#include "symbol.inc"
! #define testfock
      MODULE dmatrix
      USE prec
      USE fock
      IMPLICIT NONE

      LOGICAL, SAVE :: LDO_DMATRIX

      REAL(q) :: DMAT(6)=0._q

!            \mu_0
! DSCALE = - ----- g_e^2 \mu_B^2 [Mhz]/[Angst]^3
!             4pi
!
! \mu_0     = 4pi x 10^-7 [T][m]/[A]
! g_e \mu_B = -2.00231930436153 x 5.7883818066 x 10^-5 [eV]/[T]
! 1 J       = 6.24150934 x 10^18 eV
! 1 eV      = 2.417989348 x 10^8 MHz
!
! [eV]^2 [T][m]   1     [eV]^2
! ------ ------ ----- = ------   (since [T]=[kg]/[A]/[s]^2 and [J]= [kg][m]^2/[s]^2)
! [T]^2   [A]   [m]^3    [J]
! 
!          4pi x 10^-7 (-2.00231930436153 x 5.7883818066 x 10^-5)^2 
! DSCALE = ----------- -------------------------------------------- x 10^30 [=(Angst/m)^3] x 2.417989348 x 10^8 [Mhz/eV]
!             4pi            6.24150934 x 10^18 [=eV/J]
!
!        = 52041.016040125 [MHz]/[Angst]^3
!
      REAL(q), PARAMETER, PRIVATE :: DSCALE=52041.016040125_q
!     REAL(q), PARAMETER, PRIVATE :: DSCALE=653970.892970065_q

      ! angular intermediates
      REAL(q), POINTER, SAVE :: KSI_poly(:,:,:)=>NULL(), &
                                KSI_step(:,:,:)=>NULL()
      !  radial intermediates (type dependent)
      REAL(q), POINTER, SAVE :: SL4_poly(:,:,:,:,:)=>NULL(), &
                                SL4_step(:,:,:,:,:)=>NULL(), &
                                SL4_delt(:,:,:,:,:)=>NULL()

      ! type for which Srs and Srg were last calculated
      INTEGER, PRIVATE, SAVE :: STYPE=-1

      CONTAINS

!******************** SUBROUTINE DMATRIX_READER ************************
!
!***********************************************************************
      SUBROUTINE DMATRIX_READER(IU5,IU6,IU0)
      USE base
      INTEGER IU5,IU6,IU0
      INTEGER NTYP
      ! local variables
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM,LRPA
      CHARACTER (1) CHARAC
      CHARACTER (40) SZNAM

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
 
      LDO_DMATRIX=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LDMATRIX','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LDO_DMATRIX,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LDMATRIX'' from file INCAR.'
         LDO_DMATRIX=.FALSE.
      ENDIF
 
      CALL XML_INCAR('LDMATRIX','L',IDUM,RDUM,CDUM,LDO_DMATRIX,CHARAC,N)

      CLOSE(IU5)
      RETURN
      END SUBROUTINE DMATRIX_READER


!******************** SUBROUTINE DMATRIX_CALC **************************
!
!***********************************************************************
      SUBROUTINE DMATRIX_CALC(WDES,W,T_INFO,LATT_CUR,P,NONL_S,NONLR_S,GRID,GRIDC,CHTOT,LMDIM,CRHODE,IU6,IU0)
      USE mgrid
      USE lattice
      USE wave_high
      USE nonl_high
      USE pseudo
      USE poscar
      TYPE (wavedes) WDES
      TYPE (wavespin) W
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (grid_3d) GRID
      TYPE (grid_3d) GRIDC
      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ)
      INTEGER LMDIM,IU6,IU0
      OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      ! symmetry related quantities (common block)
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
     &   GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
 
      ! local variables
      REAL(q) A(3,3),EIGV(3,3),EIG(3),TMP,TMPV(3)
      INTEGER, PARAMETER :: LWORK=6
      REAL(q) WORK(3*LWORK)
      INTEGER I,IFAIL

      REAL(q) STOT
      REAL(q), EXTERNAL :: RHO0

      IF (.NOT.LDO_DMATRIX) RETURN

      STOT=ABS(RHO0(GRIDC,CHTOT(1,2)))/2._q
      IF (STOT<0.5_q) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,'(A,F14.7,A)') 'DMATRIX_CALC: ERROR: S=',STOT,' < 1/2, skipping calculation of D-matrix ...'
         ENDIF
         RETURN
      ENDIF

      DMAT=0._q

!     CALL DMATRIX_CALCULATE_JIJ_ALT(GRIDC,LATT_CUR,CHTOT(1,2),IU6,IU0)
      CALL DMATRIX_CALCULATE_JIJ(WDES,W,P,LATT_CUR,LMDIM,NONL_S,NONLR_S,GRID,IU6,IU0)
      CALL DMATRIX_CALCULATE_KIJ(WDES,W,P,LATT_CUR,NONL_S,NONLR_S,GRID,IU6,IU0)
      CALL SET_DAB_PAW(WDES,T_INFO,P,LMDIM,CRHODE,IU6,IU0)

      DMAT=DMAT/STOT/(2*STOT-1)

      A=0
      A(1,1)=DMAT(1)
      A(1,2)=DMAT(4); A(2,1)=A(1,2)
      A(1,3)=DMAT(5); A(3,1)=A(1,3)
      A(2,2)=DMAT(2)
      A(2,3)=DMAT(6); A(3,2)=A(2,3)
      A(3,3)=DMAT(3)
      ! Symmetrize
      CALL TSYM(A,ISYMOP,NROTK,LATT_CUR%A) 
      DMAT(1)=A(1,1)
      DMAT(4)=A(1,2)
      DMAT(5)=A(1,3)
      DMAT(2)=A(2,2)
      DMAT(6)=A(2,3)
      DMAT(3)=A(3,3)
      ! Get eigenvalues and eigenvectors
      CALL DSYEV('V','U',3,A,3,EIG,WORK,3*LWORK,IFAIL)
      EIGV(:,:)=A

      DO I=1,2
         IF (ABS(EIG(1))>ABS(EIG(2))) THEN
            TMP=EIG(2); EIG(2)=EIG(1); EIG(1)=TMP
            TMPV(:)=EIGV(:,2); EIGV(:,2)=EIGV(:,1); EIGV(:,1)=TMPV(:)
         ENDIF
         IF (ABS(EIG(2))>ABS(EIG(3))) THEN
            TMP=EIG(3); EIG(3)=EIG(2); EIG(2)=TMP
            TMPV(:)=EIGV(:,3); EIGV(:,3)=EIGV(:,2); EIGV(:,2)=TMPV(:)
         ENDIF            
      ENDDO

      IF (IU6>=0) THEN
         WRITE(IU6,'(/A)') ' Spin-spin contribution to zero-field splitting tensor (MHz)'
         WRITE(IU6,'(A)')  ' ---------------------------------------------------------------'
         WRITE(IU6,'(A)')  '      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz'
         WRITE(IU6,'(A)')  ' ---------------------------------------------------------------'

         WRITE(IU6,'(X,6F10.3)') DMAT(:)
         WRITE(IU6,'(A)')  ' ---------------------------------------------------------------'

         WRITE(IU6,'(/A)') ' after diagonalization'
         WRITE(IU6,'(A)')  ' ---------------------------------------------'
         WRITE(IU6,'(A)')  '     D_diag          eigenvector (x,y,z)'
         WRITE(IU6,'(A)')  ' ---------------------------------------------'

         DO I=1,3
            WRITE(IU6,'(X,F10.3,2X,3F10.3)') EIG(I),EIGV(:,I)
         ENDDO
         WRITE(IU6,'(A/)') '---------------------------------------------'
      ENDIF

      RETURN
      END SUBROUTINE DMATRIX_CALC


!******************** SUBROUTINE DMATRIX_CALCULATE_KIJ *****************
!
!***********************************************************************
      SUBROUTINE DMATRIX_CALCULATE_KIJ(WDES,W,P,LATT_CUR,NONL_S,NONLR_S,GRID,IU6,IU0)
      USE sym_prec
      USE wave_high
      USE nonl_high
      USE full_kpoints
      USE pseudo
      USE lattice
      USE augfast
      USE mgrid
      USE constant

      TYPE (wavedes) WDES
      TYPE (wavespin) W
      TYPE (potcar) P(:)
      TYPE (latt) LATT_CUR
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (grid_3d) GRID

      INTEGER IU6,IU0
      ! local variables
      TYPE (wavespin) WHF
      TYPE (wavedes1) WDESQ,WDESQ_IRZ,WDESK,WDESQP
      TYPE (wavefun1) WQ
      TYPE (wavefun1), ALLOCATABLE :: WK(:)
      GDEF GWORK(m_ GRIDHF%MPLWV)
      GDEF, ALLOCATABLE :: CRHOLM(:)
#ifdef testfock
      REAL(q) AEXX_SAVE,HFSCREEN_SAVE
      REAL(q), ALLOCATABLE :: POTFAK(:)
      REAL(q) FSG
      COMPLEX(q) CDCHF
#else
      REAL(q), ALLOCATABLE :: POTFAK(:,:)
      REAL(q) KIJ(6)
#endif
      INTEGER ISGN
      REAL(q) WEIGHT,RES
      INTEGER ISP1,ISP2,ISP_IRZ,ISPINOR,IDIR
      INTEGER NK,NQ,NBQ,NBK,N,NP,MM
      INTEGER NSTRIP,NSTRIP_ACT
      INTEGER IDO,NDO
      LOGICAL LSHIFT

      TYPE( rotation_handle), POINTER :: ROT_HANDLE

      IF (.NOT.LDO_DMATRIX) RETURN

      IF (WDES%NCDIJ/=2) THEN
         IF (IU0>=0) WRITE(IU0,'(A)') 'ERROR: DMATRIX_CALCULATE_KIJ: ISPIN/=2 not supported, sorry ...'
         RETURN
      ENDIF

      NSTRIP=2*WDES%NSIM

      CALL CHECK_FULL_KPOINTS

      WHF=W
      WHF%WDES=>WDES_FOCK

      CALL SETWDES(WHF%WDES,WDESK,0)
      ALLOCATE(WK(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WK(N),WDESK,.TRUE.)
      ENDDO

      CALL SETWDES(WHF%WDES,WDESQ,0)
      CALL NEWWAV(WQ,WDESQ,.TRUE.)

      ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS))
#ifdef testfock
      AEXX_SAVE=AEXX; AEXX=1._q; HFSCREEN_SAVE=HFSCREEN; HFSCREEN=0
      ALLOCATE(POTFAK(GRIDHF%MPLWV))
      CDCHF=0._q
#else
      ALLOCATE(POTFAK(GRIDHF%MPLWV,6))
      KIJ=0._q
#endif
!     NDO=CEILING(REAL(WDES%NKPTS/WDES%COMM_KINTER%NCPU))*WDES%NB_TOT*KPOINTS_FULL%NKPTS*WHF%WDES%NBANDS/50

      sp1: DO ISP1=1,WDES%ISPIN
      sp2: DO ISP2=1,WDES%ISPIN
#ifdef testfock
         IF (ISP1/=ISP2) CYCLE sp2
#else
#ifdef gammareal
         IF (ISP1>ISP2) CYCLE sp2
         ISGN=1; IF (ISP1/=ISP2) ISGN=-2
#else
         ISGN=1; IF (ISP1/=ISP2) ISGN=-1
#endif
#endif
         IF (IU0>=0) WRITE(IU0,'(X,"isp1:",I3,2X,"isp2:",I3)',ADVANCE='No') ISP1,ISP2

         IDO=0
         ! kpoints loop runs over the IBZ
         kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE kpoint
#endif
         NULLIFY(ROT_HANDLE)

         IF (NONLR_S%LREAL) THEN
            CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
         ELSE
            CALL PHASE(WDES,NONL_S,NK)
         ENDIF

#ifdef testfock
         ! average electrostatic potential for k=k' and n=n'
         FSG=FSG_STORE(NK)
#endif
         kband: DO NBK=1,WDES%NB_TOT,NSTRIP
            NSTRIP_ACT=MIN(NSTRIP,WDES%NB_TOT-NBK+1)
            CALL SETWDES(WHF%WDES,WDESK,NK)
            CALL W1_GATHER_GLB(WHF,NBK,NBK+NSTRIP_ACT-1,ISP1,WK)

            ! qpoints loop runs over full BZ
            qpoint: DO NQ=1,KPOINTS_FULL%NKPTS
#ifdef testfock
               ! set POTFAK for this q and k point
               CALL SET_GFAC(GRIDHF,LATT_CUR,NK,NQ,FSG,POTFAK)
#else
               CALL SET_GFAC_DAB(GRIDHF,LATT_CUR,KPOINTS_FULL%VKPT(:,NK)-KPOINTS_FULL%VKPT(:,NQ),POTFAK)
#endif
               CALL SETWDES(WHF%WDES,WDESQ,NQ)
               CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

               ISP_IRZ=ISP2
               IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
                  ISP_IRZ=3-ISP2
               ENDIF

               qband: DO NBQ=1,WHF%WDES%NBANDS
                  IF (ABS(WHF%FERWE(NBQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE qband
                  IF (NQ<=WHF%WDES%NKPTS) THEN
                     CALL W1_COPY(ELEMENT(WHF,WDESQ,NBQ,ISP2),WQ)
                     CALL FFTWAV_W1(WQ)
                  ELSE
#ifndef gammareal
                     LSHIFT=.FALSE.
                     IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                         (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                         (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
                     CALL W1_ROTATE_AND_FFT(WQ,ELEMENT(WHF,WDESQ_IRZ,NBQ,ISP_IRZ),ROT_HANDLE,P,LATT_CUR,LSHIFT)
#endif
                  ENDIF

                  strip: DO N=1,NSTRIP_ACT
                     IF (ABS(WHF%FERTOT(NBK+N-1,NK,ISP1))<=1E-10) CYCLE strip

!                    IF ((WHF%WDES%NB_LOW+WHF%WDES%NB_PAR*(NBQ-1))==(NBK+N-1).AND.(ISP1/=ISP2)) CYCLE strip

                     CALL FOCK_CHARGE(WK(N),WQ,GWORK(:),CRHOLM)
                     ! fft to reciprocal space
                     CALL FFT3D(GWORK(1),GRIDHF,-1)
#ifdef testfock
                     WEIGHT=WHF%FERTOT(NBK+N-1,NK,ISP1)*WHF%FERWE(NBQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)* &
                    &   WHF%WDES%WTKPT(NK)*0.5_q*WHF%WDES%RSPIN/GRIDHF%NPLWV

                     CALL CONTRACT_GFAC(GRIDHF,GWORK(1),POTFAK(1),RES)
                     IF (IU0>=0) WRITE(IU0,'(X,A,F14.7,3I6)') 'res:',RES,GRID%RC%NP,GRIDHF%NPLWV,WDESK%NGVECTOR
                     CDCHF=CDCHF+RES*WEIGHT
#else
                     WEIGHT=WHF%FERTOT(NBK+N-1,NK,ISP1)*WHF%FERWE(NBQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)* &
                    &   WHF%WDES%WTKPT(NK)*KPOINTS_FULL%WTKPT(1)*WHF%WDES%RSPIN/GRIDHF%NPLWV

                     CALL GNORM(GRIDHF,GWORK(1))

                     tensor: DO IDIR=1,6
!                       CALL CONTRACT_GFAC(GRIDHF,GWORK(1),POTFAK(1,IDIR),RES) 
                        CALL MULT_GFAC(GRIDHF,GWORK(1),POTFAK(1,IDIR),RES)
                        KIJ(IDIR)=KIJ(IDIR)+RES*WEIGHT*ISGN
                     ENDDO tensor
#endif
                     IDO=IDO+1
!                    IF (IU0>=0.AND.NDO/=0) THEN
!                       IF (MOD(IDO,NDO)==0) THEN
!                          IF (MOD(IDO-NDO,NDO*5)==0) THEN
!                             WRITE(IU0,'(A)',ADVANCE='No') '|'
!                          ELSE
!                             WRITE(IU0,'(A)',ADVANCE='No') '.'
!                          ENDIF
!                       ENDIF
!                    ENDIF
                  ENDDO strip
               ENDDO qband
            ENDDO qpoint
         ENDDO kband

         CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

         ENDDO kpoint

         CALLMPI( M_sum_i(WDES%COMM_KIN,   IDO,1))
         CALLMPI( M_sum_i(WDES%COMM_KINTER,IDO,1))
         IF (IU0>=0) WRITE(IU0,*) ' IDO:',IDO

      ENDDO sp2
      ENDDO sp1

      ! communicate (COMM_KINTER) and (COMM_KIN)
#ifdef testfock
      CALLMPI( M_sum_z(WDES%COMM_KIN,   CDCHF,1) )
      CALLMPI( M_sum_z(WDES%COMM_KINTER,CDCHF,1) )
#else
      CALLMPI( M_sum_d(WDES%COMM_KIN,   KIJ(1),6) )
      CALLMPI( M_sum_d(WDES%COMM_KINTER,KIJ(1),6) )
#endif
      CALL DELWAV(WQ,.TRUE.)
      DEALLOCATE(CRHOLM)
      DO N=1,NSTRIP
         CALL DELWAV(WK(N),.TRUE.)
      ENDDO
      DEALLOCATE(WK)
      DEALLOCATE(POTFAK)
#ifdef testfock
      AEXX=AEXX_SAVE; HFSCREEN=HFSCREEN_SAVE
      IF (IU6>=0) THEN
         WRITE(IU0,'(A,2F14.7)') 'cdchf:',cdchf
      ENDIF
#else
      IF (IU6>=0) THEN
         WRITE(IU0,'(A,6F14.7)') '-Kij:',-Kij*DSCALE/4._q*4._q*PI
      ENDIF
      DMAT=DMAT-4._q*PI*KIJ*DSCALE/4._q
#endif
      RETURN
      END SUBROUTINE DMATRIX_CALCULATE_KIJ


!******************** SUBROUTINE DMATRIX_CALCULATE_JIJ *****************
!
!***********************************************************************
      SUBROUTINE DMATRIX_CALCULATE_JIJ(WDES,W,P,LATT_CUR,LMDIM,NONL_S,NONLR_S,GRID,IU6,IU0)
      USE sym_prec
      USE wave_high
      USE nonl_high
      USE full_kpoints
      USE pseudo
      USE lattice
      USE augfast
      USE mgrid
      USE constant

      TYPE (wavedes) WDES
      TYPE (wavespin) W
      TYPE (potcar) P(:)
      TYPE (latt) LATT_CUR
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (grid_3d) GRID

      INTEGER LMDIM
      INTEGER IU6,IU0
      ! local variables
      TYPE (wavespin) WHF
      TYPE (wavedes1) WDESQ,WDESQ_IRZ,WDESK,WDESQP
      TYPE (wavefun1) WQ
      TYPE (wavefun1), ALLOCATABLE :: WK(:)
      GDEF GWORK(m_ GRIDHF%MPLWV),GWORK2(m_ GRIDHF%MPLWV)
      GDEF, ALLOCATABLE :: GWORK1(:,:)
      GDEF, ALLOCATABLE :: CRHOLM(:)

      REAL(q), ALLOCATABLE :: POTFAK(:,:)
      REAL(q) KIJ(6)

      INTEGER ISGN
      REAL(q) WEIGHT,RES
      INTEGER ISP1,ISP2,ISP_IRZ,ISPINOR,IDIR
      INTEGER NK,NQ,NBQ,NBK,N,NP,MM
      INTEGER NSTRIP,NSTRIP_ACT
      INTEGER IDO,NDO
      LOGICAL LSHIFT

      REAL(q) DUMMY(3)

      TYPE( rotation_handle), POINTER :: ROT_HANDLE

      IF (.NOT.LDO_DMATRIX) RETURN

      IF (WDES%NCDIJ/=2) THEN
         IF (IU0>=0) WRITE(IU0,'(A)') 'ERROR: DMATRIX_CALCULATE_KIJ: ISPIN/=2 not supported, sorry ...'
         RETURN
      ENDIF

      NSTRIP=2*WDES%NSIM

      CALL CHECK_FULL_KPOINTS

      WHF=W
      WHF%WDES=>WDES_FOCK

      CALL SETWDES(WHF%WDES,WDESK,0)
      ALLOCATE(WK(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WK(N),WDESK,.TRUE.)
      ENDDO

      CALL SETWDES(WHF%WDES,WDESQ,0)
      CALL NEWWAV(WQ,WDESQ,.TRUE.)

      ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS))

      ALLOCATE(POTFAK(GRIDHF%MPLWV,6))
      DUMMY=0._q; CALL SET_GFAC_DAB(GRIDHF,LATT_CUR,DUMMY,POTFAK)

      ALLOCATE(GWORK1(m_ GRIDHF%MPLWV,NSTRIP)) 

      KIJ=0._q

!     NDO=CEILING(REAL(WDES%NKPTS/WDES%COMM_KINTER%NCPU))*WDES%NB_TOT*KPOINTS_FULL%NKPTS*WHF%WDES%NBANDS/50

      sp1: DO ISP1=1,WDES%ISPIN
      sp2: DO ISP2=1,WDES%ISPIN
#ifdef gammareal
         IF (ISP1>ISP2) CYCLE sp2
         ISGN=1; IF (ISP1/=ISP2) ISGN=-2
#else
         ISGN=1; IF (ISP1/=ISP2) ISGN=-1
#endif
         IF (IU0>=0) WRITE(IU0,'(X,"isp1:",I3,2X,"isp2:",I3)',ADVANCE='No') ISP1,ISP2

         IDO=0
         ! kpoints loop runs over the IBZ
         kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE kpoint
#endif
         NULLIFY(ROT_HANDLE)

         IF (NONLR_S%LREAL) THEN
            CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
         ELSE
            CALL PHASE(WDES,NONL_S,NK)
         ENDIF

         kband: DO NBK=1,WDES%NB_TOT,NSTRIP
            NSTRIP_ACT=MIN(NSTRIP,WDES%NB_TOT-NBK+1)
            CALL SETWDES(WHF%WDES,WDESK,NK)
            CALL W1_GATHER_GLB(WHF,NBK,NBK+NSTRIP_ACT-1,ISP1,WK)

            strip1: DO N=1,NSTRIP_ACT
               IF (ABS(WHF%FERTOT(NBK+N-1,NK,ISP1))<=1E-10) CYCLE strip1
               CALL FOCK_CHARGE(WK(N),WK(N),GWORK1(:,N),CRHOLM)
               ! fft to reciprocal space
               CALL FFT3D(GWORK1(1,N),GRIDHF,-1)
            ENDDO strip1

            ! qpoints loop runs over full BZ
            qpoint: DO NQ=1,KPOINTS_FULL%NKPTS

               CALL SETWDES(WHF%WDES,WDESQ,NQ)
               CALL SETWDES(WHF%WDES,WDESQ_IRZ,KPOINTS_FULL%NEQUIV(NQ))

               ISP_IRZ=ISP2
               IF (KPOINTS_FULL%SPINFLIP(NQ)==1) THEN
                  ISP_IRZ=3-ISP2
               ENDIF

               qband: DO NBQ=1,WHF%WDES%NBANDS
                  IF (ABS(WHF%FERWE(NBQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ))<=1E-10) CYCLE qband
                  IF (NQ<=WHF%WDES%NKPTS) THEN
                     CALL W1_COPY(ELEMENT(WHF,WDESQ,NBQ,ISP2),WQ)
                     CALL FFTWAV_W1(WQ)
                  ELSE
#ifndef gammareal
                     LSHIFT=.FALSE.
                     IF ((ABS(KPOINTS_FULL%TRANS(1,NQ))>TINY) .OR. &
                         (ABS(KPOINTS_FULL%TRANS(2,NQ))>TINY) .OR. &
                         (ABS(KPOINTS_FULL%TRANS(3,NQ))>TINY)) LSHIFT=.TRUE.
                     CALL W1_ROTATE_AND_FFT(WQ,ELEMENT(WHF,WDESQ_IRZ,NBQ,ISP_IRZ),ROT_HANDLE,P,LATT_CUR,LSHIFT)
#endif
                  ENDIF

                  CALL FOCK_CHARGE(WQ,WQ,GWORK2(:),CRHOLM)
                  ! fft to reciprocal space
                  CALL FFT3D(GWORK2(1),GRIDHF,-1)

                  strip2: DO N=1,NSTRIP_ACT
                     IF (ABS(WHF%FERTOT(NBK+N-1,NK,ISP1))<=1E-10) CYCLE strip2

!                    IF ((WHF%WDES%NB_LOW+WHF%WDES%NB_PAR*(NBQ-1))==(NBK+N-1).AND.(ISP1/=ISP2)) CYCLE strip2

                     WEIGHT=WHF%FERTOT(NBK+N-1,NK,ISP1)*WHF%FERWE(NBQ,KPOINTS_FULL%NEQUIV(NQ),ISP_IRZ)* &
                    &   WHF%WDES%WTKPT(NK)*KPOINTS_FULL%WTKPT(1)*WHF%WDES%RSPIN/GRIDHF%NPLWV

                     GWORK=GWORK2
                     CALL MULT_GCHG(GRIDHF,GWORK1(1,N),GWORK(1))

                     tensor: DO IDIR=1,6
                        CALL MULT_GFAC(GRIDHF,GWORK(1),POTFAK(1,IDIR),RES)
                        KIJ(IDIR)=KIJ(IDIR)+RES*WEIGHT*ISGN
                     ENDDO tensor

                     IDO=IDO+1
!                    IF (IU0>=0.AND.NDO/=0) THEN
!                       IF (MOD(IDO,NDO)==0) THEN
!                          IF (MOD(IDO-NDO,NDO*5)==0) THEN
!                             WRITE(IU0,'(A)',ADVANCE='No') '|'
!                          ELSE
!                             WRITE(IU0,'(A)',ADVANCE='No') '.'
!                          ENDIF
!                       ENDIF
!                    ENDIF
                  ENDDO strip2
               ENDDO qband
            ENDDO qpoint
         ENDDO kband

         CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)

         ENDDO kpoint

         CALLMPI( M_sum_i(WDES%COMM_KIN,   IDO,1))
         CALLMPI( M_sum_i(WDES%COMM_KINTER,IDO,1))
         IF (IU0>=0) WRITE(IU0,*) ' IDO:',IDO

      ENDDO sp2
      ENDDO sp1

      ! communicate (COMM_KINTER) and (COMM_KIN)
      CALLMPI( M_sum_d(WDES%COMM_KIN,   KIJ(1),6) )
      CALLMPI( M_sum_d(WDES%COMM_KINTER,KIJ(1),6) )

      DEALLOCATE(GWORK1)

      CALL DELWAV(WQ,.TRUE.)
      DEALLOCATE(CRHOLM)
      DO N=1,NSTRIP
         CALL DELWAV(WK(N),.TRUE.)
      ENDDO
      DEALLOCATE(WK)
      DEALLOCATE(POTFAK)

      IF (IU6>=0) THEN
         WRITE(IU0,'(A,6F14.7)') ' Jij:',Kij*DSCALE/4._q*4._q*PI
      ENDIF
      DMAT=DMAT+4._q*PI*KIJ*DSCALE/4._q

      RETURN
      END SUBROUTINE DMATRIX_CALCULATE_JIJ


!******************** SUBROUTINE SET_GFAC_DAB **************************
!
!***********************************************************************
      SUBROUTINE SET_GFAC_DAB(GRID,LATT_CUR,VKPT,POTFAK)
      USE constant
      USE mgrid
      USE nonl_high
      USE lattice
      USE full_kpoints
      USE fock
      IMPLICIT NONE
      TYPE (grid_3d) GRID
      TYPE (latt) LATT_CUR
      REAL(q) VKPT(3)
      REAL(q) POTFAK(GRID%MPLWV,6)
      ! local variables
      INTEGER NI,NC,N1,N2,N3
      REAL(q) DKX,DKY,DKZ,GX,GY,GZ,GSQU,SCALE,FACTM

      CALL CHECK_FULL_KPOINTS

      ! all k-points in the full grid have equal weight equal to first one
      SCALE=1._q/LATT_CUR%OMEGA

      ! set correct phase in FAST_AUG structure
      CALL PHASER_HF(GRID,LATT_CUR,FAST_AUG_FOCK,VKPT)

      DKX=VKPT(1)*LATT_CUR%B(1,1)+VKPT(2)*LATT_CUR%B(1,2)+VKPT(3)*LATT_CUR%B(1,3)
      DKY=VKPT(1)*LATT_CUR%B(2,1)+VKPT(2)*LATT_CUR%B(2,2)+VKPT(3)*LATT_CUR%B(2,3)
      DKZ=VKPT(1)*LATT_CUR%B(3,1)+VKPT(2)*LATT_CUR%B(3,2)+VKPT(3)*LATT_CUR%B(3,3)

      NI=0
      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         row: DO N1=1,GRID%RC%NROW
            NI=NI+1

            FACTM=1._q
#ifdef gammareal
            IF (N1/=1) FACTM=2._q
#endif
            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))
!           GSQU=(DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2
            GX=GX+DKX; GY=GY+DKY; GZ=GZ+DKZ
            GSQU=GX**2+GY**2+GZ**2
            IF (GSQU<KPOINTS_FULL%VKPTMINDIST2/40) THEN
               POTFAK(NI,1:3)=0._q
            ELSE
               POTFAK(NI,1)=FACTM*(GX*GX/GSQU-1._q/3._q)
               POTFAK(NI,2)=FACTM*(GY*GY/GSQU-1._q/3._q)
               POTFAK(NI,3)=FACTM*(GZ*GZ/GSQU-1._q/3._q)
               POTFAK(NI,4)=FACTM*(GX*GY/GSQU)
               POTFAK(NI,5)=FACTM*(GX*GZ/GSQU)
               POTFAK(NI,6)=FACTM*(GY*GZ/GSQU)
            ENDIF
         ENDDO row
      ENDDO col

      CALL SETUNB_REAL(POTFAK,GRID)

      POTFAK(1:GRID%RC%NP,:)=POTFAK(1:GRID%RC%NP,:)*SCALE/GRID%NPLWV

      RETURN
      END SUBROUTINE SET_GFAC_DAB


!******************** SUBROUTINE DMATRIX_CALCULATE_JIJ_ALT *************
!
!***********************************************************************
      SUBROUTINE DMATRIX_CALCULATE_JIJ_ALT(GRIDC,LATT_CUR,CHTOT,IU6,IU0)
      USE mgrid
      USE lattice
      USE constant
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      COMPLEX(q) CHTOT(GRIDC%MPLWV)
      INTEGER IU6,IU0
      ! local variables
      INTEGER NI,N1,N2,N3,NC,IDIR
      REAL(q) JIJ(6)
      REAL(q) SCALE,GX,GY,GZ,GSQU,FACTM
      COMPLEX(q) CVD(GRIDC%RC%NP,6)

      IF (.NOT.LDO_DMATRIX) RETURN

      SCALE=1._q/LATT_CUR%OMEGA

      JIJ=0._q

      NI=0
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
 
         NI=NI+1
         SETFACT1
         SETFACT
 
         GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))

         GSQU=GX**2+GY**2+GZ**2
         IF ((GRIDC%LPCTX(N1)==0).AND.(GRIDC%LPCTY(N2)==0).AND.(GRIDC%LPCTZ(N3)==0)) THEN
             CVD(NI,:)=(0.0_q,0.0_q)
          ELSE
             CVD(NI,1)=(GX*GX/GSQU-1._q/3._q)
             CVD(NI,2)=(GY*GY/GSQU-1._q/3._q)
             CVD(NI,3)=(GZ*GZ/GSQU-1._q/3._q)
             CVD(NI,4)=(GX*GY/GSQU)
             CVD(NI,5)=(GX*GZ/GSQU)
             CVD(NI,6)=(GY*GZ/GSQU)
             CVD(NI,:)=CVD(NI,:)*CHTOT(NI)*SCALE
         ENDIF
      ENDDO row
      ENDDO col
      DO IDIR=1,6
         CALL SETUNB(CVD(:,IDIR),GRIDC)
      ENDDO

      NI=0
      col2: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row2: DO N1=1,GRIDC%RC%NROW
  
         NI=NI+1
         SETFACT1
         SETFACT
  
         JIJ(:)=JIJ(:)+MULFACT CVD(NI,:)*CONJG(CHTOT(NI))

      ENDDO row2
      ENDDO col2

      CALLMPI( M_sum_d(GRIDC%COMM,JIJ(1),6))

      IF (IU6>=0) THEN
         WRITE(IU0,'(A,6F14.7)') ' Jij:',JIJ*DSCALE/4._q*4._q*PI
      ENDIF

      RETURN
      END SUBROUTINE DMATRIX_CALCULATE_JIJ_ALT


!********************** SUBROUTINE SET_DAB_PAW *************************
!
!***********************************************************************
      SUBROUTINE SET_DAB_PAW(WDES,T_INFO,P,LMDIM,CRHODE,IU6,IU0)
      USE asa
      USE pseudo
      USE poscar
      USE wave
      USE paw
      TYPE (wavedes) WDES
      TYPE (type_info) T_INFO
      TYPE (potcar), TARGET :: P(T_INFO%NTYP)
      INTEGER LMDIM
      OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      INTEGER IU6,IU0
      ! local variables
      OVERLAP, ALLOCATABLE :: COCC(:,:) 
      TYPE (potcar), POINTER :: PP
      REAL(q) DAB(6),JIJ(6),KIJ(6)
      INTEGER IBASE,IDONE
      INTEGER NI,NIP,NT
      INTEGER ISP,NCDIJ

      LOGICAL LSKIP
      INTEGER LMAX,LMMAX
      INTEGER, EXTERNAL :: MAXL

      IF (.NOT.LDO_DMATRIX) RETURN

      IF (WDES%NCDIJ/=2) THEN
         WRITE(*,*) 'SET_DAB_PAW: ERROR: ISPIN/=2 not supported.'
!        STOP
      ENDIF

      LMAX=MAXL(T_INFO%NTYP,P)
!     LMAX=MIN(6,LMAX+2); LMMAX=(LMAX+1)*(LMAX+1)
      LMAX=MIN(LMAX*2,6)

      CALL DAB_POLY_POLAR(LMAX,KSI_poly)
      CALL DAB_HEAVYSIDE(LMAX,KSI_step)

      ALLOCATE(COCC(LMDIM,LMDIM))

      DAB=0; IDONE=0

      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI,WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)

         LSKIP=.FALSE.
#ifdef MPI
         ! DO_LOCAL represents a distribution of the work on the
         ! one-center terms over the procs in COMM_INB and COMM_INTER (=COMM_KIN).
         ! The following allows an additional round-robin distribution over COMM_KINTER as well.
         IF (DO_LOCAL(NI)) THEN
            IDONE=IDONE+1; LSKIP=(MOD(IDONE,WDES%COMM_KINTER%NCPU)+1/=WDES%COMM_KINTER%NODE_ME)
         ENDIF
#endif
         ! if this element is not treated locally CYCLE
         IF (.NOT.DO_LOCAL(NI).OR.LSKIP) CYCLE ion

         COCC=CRHODE(:,:,NIP,2)

         PP=>PP_POINTER(P,NI,NT)

         CALL SETSL4(NT,PP,SL4_poly,SL4_step,SL4_delt)

         KIJ=0; JIJ=0
         CALL CALC_DAB_PAW(LMAX,PP,COCC,JIJ,KIJ)

         DAB(:)=DAB(:)+JIJ(:)-KIJ(:)
      ENDDO ion

      CALLMPI( M_sum_d(WDES%COMM,DAB,6))

      CALL UNSETSL4(SL4_poly,SL4_step,SL4_delt)
      CALL UNSETKSI(KSI_poly,KSI_step)
      DEALLOCATE(COCC)

      IF (IU6>=0) THEN
         WRITE(IU0,'(A,6F14.7)') ' D1c:',-DAB*DSCALE/4._q
      ENDIF
      DMAT=DMAT-DAB*DSCALE/4._q

      RETURN
      END SUBROUTINE SET_DAB_PAW


!********************** SUBROUTINE CALC_DAB_PAW ************************
!
! The interaction kernel
!
!              3 (r-r')_a (r-r')_b - |r-r'|^2 \delta_ab
!  D_ab(r-r')= ----------------------------------------
!                             |r-r'|^5
!
! can be derived from
!
!  d^2           1                     4pi
!  --------- ( ------ ) = D_ab(r-r') - ---\delta(r-r')\delta_ab
!  dr_a dr_b   |r-r'|                   3
!
! where second term on the RHS above stems from the fact that
!
!  \nabla^2 (1/|r-r'|) = -4pi\delta(r-r')
!
! i.e.,
!               d^2           1        4pi 
!  D_ab(r-r') = --------- ( ------ ) + ---\delta(r-r')\delta_ab
!               dr_a dr_b   |r-r'|      3
!
! The second derivative of the Coulomb kernel can written in terms
! of the "poly-polar expansion of irregular spherical harmonics", plus
! addtional contributions that arise from the case |r|=|r'| (or more
! correctly, from the double derivative of the Heavyside step functions
! H(r-r') and H(r-r') with respect to r, that are used to define r_< and r_>,
! found in the radial part of the spherical harmonics expansion of the
! Coulomb kernel).
!
! More details can be found in the comment blocks of DAB_POLY_POLAR
! and DAB_HEAVYSIDE.
!
! The expansion of the interaction kernel in spherical harmonics is
! then given by
!
!  D_ab(r-r') = \sum_lmm' Y_lm(r_<) KSI_poly(ab, lm, l+2 m') r_<^l / r_>^(l+3) Y_l+2,m(r_>)
!
!     + \sum_lm \sum_l'm' Y_lm(r) KSI_step(ab, lm,l'm') \delta(|r|-|r'|)/|r|^2 Y_l'm'(r')
!
!              + 4pi/3 \delta_ab \delta(\hat r - \hat r') \delta(|r|-|r'|)/|r|^2 
!
! where
!              / r   for |r| < |r'|            / r'  for |r| < |r'|
!       r_< = |                         r_> = | 
!              \ r'  for |r'|< |r|             \ r   for |r'|< |r|
!
! (clearly, in the above r and r' are understood to be vectors)
!
! The coefficients KSI_poly and KSI_step are computed by the subroutines
! DAB_POLY_POLAR and DAB_HEAVYSIDE, respectively.
!
! From the first line in the expansion of the interaction kernel we
! see that a charge density Q_l(r)Y_lm(r) gives rise to potential 
! contribution at l+2:
!
!                        Y_l+2,m'(r')
!  KSI_poly(:,lm,l+2 m') ------------ \int_0^r' Q_l(r) r^l dr
!                          r'^(l+3)
! and at l-2:
!                                                         Q_l(r)
!  KSI_poly(:,l-2 m',lm) Y_l-2,m'(r') r'^(l-2) \int_r'^oo ------- dr
!                                                         r^(l+1)
!
! The action of this potential on a charge density \sum_l'm' P_l'(r')Y_l'm'(r')
! then yields contributions:
!                                      P_l+2(r')
!  KSI_poly(:,lm,l+2 m') \int_0^oo dr' --------- \int_0^r' Q_l(r) r^l dr
!                                      r'^(l+3)
! and
!                                                                Q_l(r)
!  KSI_poly(:,l-2 m',lm) \int_0^oo P_l-2(r') r'^(l-2) \int_r'^oo ------- dr
!                                                                r^(l+1)
! With
!                                P_l+2(r')
!  S_poly(l,P,Q) = \int_0^oo dr' --------- \int_0^r' Q_l(r) r^l dr
!                                r'^(l+3)
! and 
!                                          Q_l(r)
!  \int_0^oo P_l-2(r') r'^(l-2) \int_r'^oo ------- dr = S_poly(l-2,Q,P)
!                                          r^(l+1)
!
! (see the description of CALCSL4AE and CALCSL4PS as well.)
!
! we get an interaction energy
!
!  KSI_poly(:,lm,l+2 m') S_poly(l,P,Q) + KSI_poly(:,l-2 m',lm) S_poly(l-2,Q,P)
!
! 
! The second line in the expansion of the interaction kernel yields contributions
! between charge densities with |l-l'| = 2 or 0:
!
!  KSI_step(:,lm,l+2 m') S_step(l,P,Q) + KSI_step(:,l-2 m',lm) S_step(l-2,Q,P)
!     
!  + KSI_step(:,lm,lm') S_delt(l,P,Q)
!
! where
!
!  S_step(l,P,Q) = \int_0^oo P_l+2(r) Q_l(r) / r^2 dr
!
! and
!
!  S_delt(l,P,Q) = \int_0^oo P_l(r) Q_l(r) / r^2 dr
!
! (again see the description of CALCSL4AE and CALCSL4PS as well.)
!
! 
! The charge densities that enter into the one-center interaction energy
! are of the form
!
!  Q_ij(r) = <Y_LM|Y_lmi Y_lmj> [<\phi_j|r><r|\phi_i>+ \hat rho^L_ij(r)] \rho_ij
!
! where
!
!  \rho_ij = \sum_nk w_k f_nk <\Psi_nk|p_j><p_i|\Psi_nk> = COCC(j,i)
!
! where the functions \phi are either all-electron or pseudo partial waves.
! (In case the \phi are taken to be all-electron partial waves, then the
! augmentation density \rho^L is zero.)
!
! With this definition we get a total of 6 contribution to J_ab ("Hartree"-like):
!
! L <--> L+2
!
! \rho_ba < Y_a Y_b | Y_LM > SL4_poly(L,a,b,m,n) KSI_poly(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_nm
!
!
! \rho_ba < Y_a Y_b | Y_LM > SL4_step(L,a,b,m,n) KSI_step(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_nm
!
!
! L-2 <--> L
!
! \rho_ba < Y_a Y_b | Y_L-2 M'> SL4_poly(L-2,m,n,a,b) KSI_poly(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_nm
!
!
! \rho_ba < Y_a Y_b | Y_L-2 M'> SL4_step(L-2,m,n,a,b) KSI_step(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_nm
!
!
! L <--> L
!
! \rho_ba < Y_a Y_b | Y_LM'> SL4_delt(L,a,b,m,n) KSI_step(:,LM',LM) < Y_LM | Y_m Y_n > \rho_nm
!
!
! LM <--> LM
!
! \rho_ba < Y_a Y_b | Y_LM> SL4_delt(L,a,b,m,n) 4Pi/3 < Y_LM | Y_m Y_n > \rho_nm
!

! and to K_ab ("Exchange"-like):
!
! L <--> L+2
!
! \rho_na < Y_a Y_b | Y_LM > SL4_poly(L,a,b,m,n) KSI_poly(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_bm
!
!
! \rho_na < Y_a Y_b | Y_LM > SL4_step(L,a,b,m,n) KSI_step(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_bm
!
!
! L-2 <--> L
!
! \rho_na < Y_a Y_b | Y_L-2 M'> SL4_poly(L-2,m,n,a,b) KSI_poly(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_bm
!
!
! \rho_na < Y_a Y_b | Y_L-2 M'> SL4_step(L-2,m,n,a,b) KSI_step(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_bm
!
!
! L <--> L
!
! \rho_na < Y_a Y_b | Y_LM'> SL4_delt(L,a,b,m,n) KSI_step(:,LM',LM) < Y_LM | Y_m Y_n > \rho_bm
!
!
! LM <--> LM
!
! \rho_na < Y_a Y_b | Y_LM> SL4_delt(L,a,b,m,n) 4Pi/3 < Y_LM | Y_m Y_n > \rho_bm
!
!***********************************************************************
      SUBROUTINE CALC_DAB_PAW(LMAX,PP,COCC,JIJ,KIJ)
      USE asa
      USE pseudo
      USE constant
      TYPE (potcar) PP
      OVERLAP COCC(:,:)
      REAL(q) JIJ(6),KIJ(6)
      INTEGER LMAX
      ! local variables
      REAL(q) FACT
      INTEGER IA,IB,IM,IN,LA,LB,LM,LN,MA,MB,MM,MN,LMA,LMB,LMM,LMN
      INTEGER LMIND1,ISTART1,IEND1,LMIND2,ISTART2,IEND2,IC1,IC2
      INTEGER JS1,JS2,JL1,JL2
      INTEGER Nc

      Nc=PP%LMAX

      LMA=1
      a__: DO IA=1,Nc
      LMB=1
      b__: DO IB=1,Nc
            
         LA=PP%LPS(IA)
         LB=PP%LPS(IB)

         LMM=1
         m__: DO IM=1,Nc
         LMN=1
         n__: DO IN=1,Nc

            LM=PP%LPS(IM)
            LN=PP%LPS(IN)

            IF (MOD(LM+LN,2)/=MOD(LA+LB,2)) THEN
               LMN=LMN+2*LN+1; CYCLE n__
            ENDIF

            CALL YLM3LOOKUP(LA,LB,LMIND1)

            ma__: DO MA=1,2*LA+1
            mb__: DO MB=1,2*LB+1

               LMIND1=LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               CALL YLM3LOOKUP(LM,LN,LMIND2)

               mm__: DO MM=1,2*LM+1
               mn__: DO MN=1,2*LN+1

                  LMIND2 = LMIND2+1
                  ISTART2=INDCG(LMIND2)
                  IEND2=INDCG(LMIND2+1)
 
                  LM__: DO IC2=ISTART2,IEND2-1
                     JL2=JL(IC2); JS2=JS(IC2)
                  LMP_: DO IC1=ISTART1,IEND1-1
                     JL1=JL(IC1); JS1=JS(IC1)

                     IF (JL1>LMAX.OR.JL2>LMAX) CYCLE LMP_

                     IF (JL2-JL1==2) THEN
!
! \rho_ba < Y_a Y_b | Y_LM > SL4_poly(L,a,b,m,n) KSI_poly(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_nm
!
                        JIJ(:)=JIJ(:)+ &
                      &   SL4_poly(JL1/2,IA,IB,IM,IN)*KSI_poly(:,JS1,JS2)* &
                      &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_LM > SL4_poly(L,a,b,m,n) KSI_poly(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_bm
!
                        KIJ(:)=KIJ(:)+ &
                      &   SL4_poly(JL1/2,IA,IB,IM,IN)*KSI_poly(:,JS1,JS2)* &
                      &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
!
! \rho_ba < Y_a Y_b | Y_LM > SL4_step(L,a,b,m,n) KSI_step(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_nm
!
                        JIJ(:)=JIJ(:)+ &
                       &   SL4_step(JL1/2,IA,IB,IM,IN)*KSI_step(:,JS1,JS2)* &
                       &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_LM > SL4_step(L,a,b,m,n) KSI_step(:,LM,L+2 M') < Y_L+2 M' | Y_m Y_n > \rho_bm
!
                        KIJ(:)=KIJ(:)+ &
                       &   SL4_step(JL1/2,IA,IB,IM,IN)*KSI_step(:,JS1,JS2)* &
                       &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
                     ENDIF

                     IF (JL1-JL2==2) THEN
!
! \rho_ba < Y_a Y_b | Y_L-2 M'> SL4_poly(L-2,m,n,a,b) KSI_poly(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_nm
!
                        JIJ(:)=JIJ(:)+ &
                       &   SL4_poly(JL2/2,IM,IN,IA,IB)*KSI_poly(:,JS2,JS1)* &
                       &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_L-2 M'> SL4_poly(L-2,m,n,a,b) KSI_poly(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_bm
!
                        KIJ(:)=KIJ(:)+ &
                       &   SL4_poly(JL2/2,IM,IN,IA,IB)*KSI_poly(:,JS2,JS1)* &
                       &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
!
! \rho_ba < Y_a Y_b | Y_L-2 M'> SL4_step(L-2,m,n,a,b) KSI_step(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_nm
!
                        JIJ(:)=JIJ(:)+ &
                       &   SL4_step(JL2/2,IM,IN,IA,IB)*KSI_step(:,JS2,JS1)* &
                       &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_L-2 M'> SL4_step(L-2,m,n,a,b) KSI_step(:,L-2 M',LM) < Y_LM | Y_m Y_n > \rho_bm
!
                        KIJ(:)=KIJ(:)+ &
                       &   SL4_step(JL2/2,IM,IN,IA,IB)*KSI_step(:,JS2,JS1)* &
                       &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
                     ENDIF

                     IF (JL1==JL2) THEN
!
! \rho_ba < Y_a Y_b | Y_LM'> SL4_delt(L,a,b,m,n) KSI_step(:,LM',LM) < Y_LM | Y_m Y_n > \rho_nm
!
                        JIJ(:)=JIJ(:)+ &
                       &   SL4_delt(JL1/2,IA,IB,IM,IN)*KSI_step(:,JS1,JS2)* &
                       &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_LM'> SL4_delt(L,a,b,m,n) KSI_step(:,LM',LM) < Y_LM | Y_m Y_n > \rho_bm
!
                        KIJ(:)=KIJ(:)+ &
                       &   SL4_delt(JL1/2,IA,IB,IM,IN)*KSI_step(:,JS1,JS2)* &
                       &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
                     ENDIF

                     IF (JS1==JS2) THEN
!
! \rho_ba < Y_a Y_b | Y_LM> SL4_delt(L,a,b,m,n) 4Pi/3 < Y_LM | Y_m Y_n > \rho_nm
!
                        JIJ(1:3)=JIJ(1:3)+ &
                       &   SL4_delt(JL1/2,IA,IB,IM,IN)*4._q*PI/3._q* &
                       &   COCC(LMA+MA-1,LMB+MB-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMN+MN-1)
!
! \rho_na < Y_a Y_b | Y_LM> SL4_delt(L,a,b,m,n) 4Pi/3 < Y_LM | Y_m Y_n > \rho_bm
!
                        KIJ(1:3)=KIJ(1:3)+ &
                       &   SL4_delt(JL1/2,IA,IB,IM,IN)*4._q*PI/3._q* &
                       &   COCC(LMA+MA-1,LMN+MN-1)*YLM3(IC1)*YLM3(IC2)*COCC(LMM+MM-1,LMB+MB-1)
                     ENDIF

                  ENDDO LMP_
                  ENDDO LM__
               ENDDO mn__
               ENDDO mm__

            ENDDO mb__
            ENDDO ma__

         LMN=LMN+2*LN+1
         ENDDO n__
         LMM=LMM+2*LM+1
         ENDDO m__

      LMB=LMB+2*LB+1
      ENDDO b__
      LMA=LMA+2*LA+1
      ENDDO a__

      RETURN
      END SUBROUTINE CALC_DAB_PAW


!********************** SUBROUTINE DAB_POLY_POLAR **********************
!
!            3x^2-r^2
! D(1): xx:  -------- = -1/2 ( 4 sqrt(pi/5) Y_20 - 12 sqrt(pi/15) Y_22 )
!              r^2
!
!            3y^2-r^2
! D(2): yy:  -------- = -1/2 ( 4 sqrt(pi/5) Y_20 + 12 sqrt(pi/15) Y_22 )
!              r^2
!
!            3z^2-r^2
! D(3): zz:  -------- =        4 sqrt(pi/5) Y_20 
!              r^2
!
!              3xy
! D(4): xy:    ---    =       6 sqrt(pi/15) Y_2-2
!              r^2
!
!              3xz
! D(5): xz:    ---    =       6 sqrt(pi/15) Y_21
!              r^2
!
!              3yz
! D(6): yz:    ---    =       6 sqrt(pi/15) Y_2-1
!              r^2
!
! In this routine
! 
!  D(i)= \sum_j XIYJ2YLM(i,j) Y_2j   (j=-2,..,2)
!
! (where obviously XIXJ2YLM(i,j) = <Y_2j|D(i)>)
!
! and
!
!  |Y_lM>= \sum_m YlM2ylm(m,M) |y_lm>    (m=-l,..,l)
!
!
! With these transformations we express D(i) in complex spherical harmonics
!
!  d(i)= \sum_mm' |y_2m'><y_2m'|Y_2m ><Y_2m|D(i)>
!
!      = \sum_m' XIXJ_in_ylm(i,m') |y_2m'>   (m'=-2,..,2)
!
! From the poly-polar expansion (see Eqs. 6 and 7 in PRB 24, 4240 (1981))
! we have that
!
!  y_2m(r-R)                                                r^l'
!  --------- = \sum_l' \sum_m' (-1)^m' A_2m,l'm' y_l'm'(r) -------- y_2+l',m-m'(R)
!  |r-R>|^3                                                R^(l'+3)
!
! where
!                               2l+1         (l+l'+m-m')!(l+l'-m+m')!
!  a_lm,l'm' = sqrt( 4pi ----------------- ---------------------------- )
!                        (2l+2l'+1)(2l'+1) (l'+m')!(l'-m')!(l+m)!(l-m)!
!
!
!  <Y_LM|\hat r><\hat r| d(i)/(r-R)^3 |\hat R><\hat R|Y_L+2,M'> = \sum_m XIXJ_in_ylm(i,m) *
!
!      \sum_m' (-1)^m' A_2m,Lm' <Y_LM|y_Lm'> <Y_L+2,M'|y_L+2,m-m'> r^L / R^(L+3)
!
!    = KSI(i,ind(L,M),ind(L+2,M')) r^L / R^(L+3)
!
! where the compound LM index is given by
!
!  ind(L,M)=(L+1)^2-L+M
!
! and
!
!  R > r (or in the notation that is commonly used: r = r_< and R = r_>) 
!
!
! Output:
!
!  KSI(i,ind(L,M),ind(L+2,M'))=
!
!   \sum_m XIXJ_in_ylm(i,m) \sum_m'(-1)^m' A_2m,Lm' <Y_LM|y_Lm'> <Y_L+2,M'|y_L+2,m-m'>
!
! for L=0,LMAX-2 (i.e. L must be >= 2)
!
!***********************************************************************
      SUBROUTINE DAB_POLY_POLAR(LMAX,KSI)
      USE constant
      INTEGER LMAX
      REAL(q), POINTER :: KSI(:,:,:)

      ! local variables
      INTEGER I
      INTEGER, PARAMETER :: IMAX=24
      REAL(q) FAC(0:IMAX)

      INTEGER LMMAX,L,M,LP,MP,M_,mu,LM,LMP
      COMPLEX(q), ALLOCATABLE :: YlM2ylm(:,:,:)

      REAL(q), ALLOCATABLE :: TMP(:,:,:)

      REAL(q) XIXJ2YLM(6,-2:2)
      REAL(q), PARAMETER :: SQPi54=4._q*SQRT(PI/5._q),SQPi52=2._q*SQRT(PI/5._q),mSQPi52=-2._q*SQRT(PI/5._q),SQPi15=6._q*SQRT(PI/15),mSQPi15=-6._q*SQRT(PI/15)
! 
!     DATA XIXJ2YLM / 0._q,   0._q,-SQPi52,   0._q, SQPi15, &
!    &                0._q,   0._q,-SQPi52,   0._q,-SQPi15, &
!    &                0._q,   0._q, SQPi54,   0._q,   0._q, &
!    &              SQPi15,   0._q,   0._q,   0._q,   0._q, &
!    &                0._q,   0._q,   0._q, SQPi15,   0._q, &
!    &                0._q, SQPi15,   0._q,   0._q,   0._q  /
               
      DATA XIXJ2YLM / 0._q,   0._q,   0._q, SQPi15,   0._q,   0._q, &
     &                0._q,   0._q,   0._q,   0._q,   0._q, SQPi15, &
     &             mSQPi52,mSQPi52, SQPi54,   0._q,   0._q,   0._q, &
     &                0._q,   0._q,   0._q,   0._q, SQPi15,   0._q, &
     &              SQPi15,mSQpi15,   0._q,   0._q,   0._q,   0._q /
 
      COMPLEX(q) XIXJ_in_ylm(6,-2:2)

      IF (LMAX<2) THEN
         WRITE(*,'(A,I4)') 'DAB_POLY_POLAR: ERROR: LMAX<2, LMAX=',LMAX
         STOP
      ENDIF

      LMMAX=(LMAX+1)*(LMAX+1)

      IF (ASSOCIATED(KSI)) THEN
         DEALLOCATE(KSI); NULLIFY(KSI)
      ENDIF
      ALLOCATE(KSI(6,LMMAX,LMMAX))

! set up transformation matrix real->complex spherical harmonics
!
!   YlM2ylm(m,M) = <y_lm|Y_lM>
! 
! where y_lm and Y_lm are, complex and real spherical harmonics,
! respectively.
!
      ALLOCATE(YlM2ylm(-LMAX:LMAX,-LMAX:LMAX,0:LMAX))
      YlM2ylm=(0._q,0._q)

      DO L=0,LMAX
         DO M=-L,L
            IF (M>0) THEN
               YlM2ylm( M,M,L)=(-1)**M/SQRT(2._q)
               YlM2ylm(-M,M,L)=      1/SQRT(2._q)
            ENDIF
            IF (M==0) THEN
               YlM2ylm(0,0,L)=1._q
            ENDIF
            IF (M<0) THEN
               YlM2ylm( M,M,L)= CMPLX(0._q,   1._q/SQRT(2._q),q)
               YlM2ylm(-M,M,L)=-CMPLX(0._q,(-1)**M/SQRT(2._q),q)
            ENDIF
         ENDDO
      ENDDO

!
!  XIXJ_in_ylm(i,m')= \sum_m <y_2m'|Y_2m ><Y_2m|D(i)>
! 
      XIXJ_in_ylm=(0._q,0._q)
      DO M=-2,2; DO MP=-2,2; DO I=1,6
         XIXJ_in_ylm(I,MP)=XIXJ_in_ylm(I,MP)+YLM2ylm(MP,M,2)*XIXJ2YLM(I,M)
      ENDDO; ENDDO; ENDDO

! set up table with factorials
      FAC(0)=1._q
      DO I=1,IMAX
         FAC(I)=I*FAC(I-1)
      ENDDO

      ALLOCATE(TMP(-2:2,-LMAX+2:LMAX-2,0:LMAX-2))
      TMP=(0._q,0._q)

      DO LP=0,LMAX-2
         DO MP=-LP,LP; DO M=-2,2
            TMP(M,MP,LP)=APOLY(2,M,LP,MP,FAC)*(-1)**MP
         ENDDO; ENDDO
      ENDDO

      KSI=(0._q,0._q)

      LM=0
      DO L=0,LMAX-2
         DO M=-L,L
            LM=LM+1; LMP=(L+2)*(L+2)
            DO MP=-L-2,L+2
               LMP=LMP+1
               DO M_=-L,L; DO mu=-2,2
                  KSI(:,LM,LMP)=KSI(:,LM,LMP)+XIXJ_in_ylm(:,mu)*TMP(mu,M_,L)*CONJG(YlM2ylm(M_,M,L))*CONJG(YlM2ylm(mu-M_,MP,L+2))
               ENDDO; ENDDO
            ENDDO
         ENDDO
      ENDDO

      DEALLOCATE(TMP,YlM2ylm) 

      RETURN   
      END SUBROUTINE DAB_POLY_POLAR


!********************** FUNCTION APOLY *********************************
!
!                               2l+1         (l+l'+m-m')!(l+l'-m+m')!
!  a_lm,l'm' = sqrt( 4pi ----------------- ---------------------------- )
!                        (2l+2l'+1)(2l'+1) (l'+m')!(l'-m')!(l+m)!(l-m)!
!
!***********************************************************************
      FUNCTION APOLY(L,M,LP,MP,FAC)
      USE constant
      INTEGER L,M,LP,MP
      REAL(q) FAC(0:)
      REAL(q) APOLY

      APOLY=REAL(2*L+1,q)/REAL(2*L+2*LP+1,q)/REAL(2*LP+1,q)
      APOLY=APOLY*FAC(L+LP+M-MP)/FAC(L+M)/FAC(LP-MP)
      APOLY=APOLY*FAC(L+LP-M+MP)/FAC(L-M)/FAC(LP+MP)
      APOLY=SQRT(4._q*PI*APOLY)

      END FUNCTION APOLY


!********************** SUBROUTINE DAB_HEAVYSIDE ***********************
!
! With the following definition of r_< and r_>
!
!  r_< = r H(r'-r) + r'H(r-r') and  r_> = r H(r-r') +r'H(r'-r)
!
! where H is the Heayside step function, and r=|\vec r| and r'=|\vec r'|
!
! we have that
!
!   d^2       r_<^l              r_i r_j              r^(l-2) 
! ---------(---------) = l[(l-2) ------- + \delta_ij] -------- H(r'-r)
! dr_i dr_j r_>^(l+1)              r^2                r'^(l+1) 
!
!                                r_i r_j               r'^l 
!                  + (l+1)[(l+3) ------- - \delta_ij] -------  H(r-r')
!                                  r^2                r^(l+3)
!
!                       r^(l-1)          r'^l    r_i r_j 
!                  - (l -------- + (l+1)-------) ------- \delta(r-r')
!                       r'^(l+1)        r^(l+2)    r^2   
!
! The first two contributions are included in the poly-polar expansion
! of the interaction kernel (see SUBROUTINE DAB_POLY_POLAR), the last
! line contributes as
!
!               r_i r_j        \delta(r-r')
!  - 4pi <Y_lm| ------- |Y_LM> ------------
!                 r^2              r^2
!
! Output:
!                                          r_i r_j         
!  KSI(ij,ind(l,m),ind(L,M))= - 4pi <Y_lm| ------- |Y_LM>  (ij=xx,yy,zz,xy,xz,yz)
!                                            r^2          
!***********************************************************************
      SUBROUTINE DAB_HEAVYSIDE(LMAX,KSI)
      USE asa
      USE constant
      INTEGER LMAX
      REAL(q), POINTER :: KSI(:,:,:)
      ! local variables
      REAL(q), ALLOCATABLE :: YLM_XIXJ_YLM(:,:,:)
      INTEGER LMMAX,I

      IF (LMAX<2) THEN
         WRITE(*,'(A,I4)') 'DAB_HEAVYSIDE: ERROR: LMAX<2, LMAX=',LMAX
         STOP
      ENDIF

      LMMAX=(LMAX+1)*(LMAX+1)

      IF (ASSOCIATED(KSI)) THEN
         DEALLOCATE(KSI); NULLIFY(KSI)
      ENDIF
      ALLOCATE(KSI(6,LMMAX,LMMAX))

      ALLOCATE(YLM_XIXJ_YLM(LMMAX,LMMAX,6))
      CALL SETYLM_XIXJ_YLM(LMAX,YLM_XIXJ_YLM)

      DO I=1,6
         KSI(I,:,:)=-4._q*PI*YLM_XIXJ_YLM(:,:,I)
      ENDDO

      DEALLOCATE(YLM_XIXJ_YLM) 

      RETURN
      END SUBROUTINE DAB_HEAVYSIDE


!********************** SUBROUTINE UNSETKSI ****************************
!
!***********************************************************************
      SUBROUTINE UNSETKSI(K1,K2)
      REAL(q), POINTER :: K1(:,:,:),K2(:,:,:)
      IF (ASSOCIATED(K1)) THEN
         DEALLOCATE(K1); NULLIFY(K1)
      ENDIF
      IF (ASSOCIATED(K2)) THEN
         DEALLOCATE(K2); NULLIFY(K2)
      ENDIF
      RETURN
      END SUBROUTINE UNSETKSI


!********************** SUBROUTINE UNSETSL4 ****************************
!
!***********************************************************************
      SUBROUTINE UNSETSL4(S1,S2,S3)
      REAL(q), POINTER :: S1(:,:,:,:,:),S2(:,:,:,:,:),S3(:,:,:,:,:)
      IF (ASSOCIATED(S1)) THEN
         DEALLOCATE(S1); NULLIFY(S1)
      ENDIF
      IF (ASSOCIATED(S2)) THEN
         DEALLOCATE(S2); NULLIFY(S2)
      ENDIF
      IF (ASSOCIATED(S3)) THEN
         DEALLOCATE(S3); NULLIFY(S3)
      ENDIF
      STYPE=-1
      RETURN
      END SUBROUTINE UNSETSL4


!********************** SUBROUTINE SETSL4 ******************************
!
!***********************************************************************
      SUBROUTINE SETSL4(NT,PP,S1,S2,S3)
      USE pseudo
      INTEGER NT 
      TYPE (potcar) PP
      REAL(q), POINTER :: S1(:,:,:,:,:),S2(:,:,:,:,:),S3(:,:,:,:,:)
      ! local variables
      INTEGER Nc,LYMAX
      INTEGER, EXTERNAL :: MAXL1

      IF (NT==STYPE) RETURN

      IF (ASSOCIATED(S1)) THEN
         DEALLOCATE(S1); NULLIFY(S1)
      ENDIF
      IF (ASSOCIATED(S2)) THEN
         DEALLOCATE(S2); NULLIFY(S2)
      ENDIF
      IF (ASSOCIATED(S3)) THEN
         DEALLOCATE(S3); NULLIFY(S3)
      ENDIF

      Nc=PP%LMAX
      LYMAX=2*MAXL1(PP)
      ALLOCATE(S1(0:LYMAX/2,Nc,Nc,Nc,Nc),S2(0:LYMAX/2,Nc,Nc,Nc,Nc),S3(0:LYMAX/2,Nc,Nc,Nc,Nc))

      S1=0._q; S2=0._q; S3=0._q

      CALL CALCSL4PS(PP%WPS,PP%LPS,PP%R,Nc,PP%AUG_SOFT,PP%QPAW,PP%AUG_FOCK,PP%QPAW_FOCK,S1,S2,S3)
      S1=-S1; S2=-S2; S3=-S3

      CALL CALCSL4AE(PP%WAE,PP%LPS,PP%R,Nc,S1,S2,S3)

      STYPE=NT

      RETURN
      END SUBROUTINE SETSL4


!********************** SUBROUTINE CALCSL4AE ***************************
!
!  \rho^L_kl(r) = <phi_k|r>< r|phi_l>
!
! where <r|phi_i> are the all-electron partial waves
!                  1
!  V1^L_kl(r) = ------- \int_0^r \rho^L_kl(r') r'^L dr'
!               r^(L+3)
!                                \rho^L_kl(r')
!  V2^L_kl(r) =  r^L-2 \int_r^oo ------------- dr'
!                                   r'^(L+1)
!
!  V3^L_kl(r) = \rho^L_kl(r)/r^2
!
! and with
!
!  \rho^L'_ij(r) = <phi_i|r>< r|phi_j>
!
! we define (the output of this subroutine) 
!
!  S1(L/2,i,j,k,l) = \int V1^L_kl(r) \rho^L+2_ij(r) dr
!
!  S2(L/2,i,j,k,l) = \int V3^L_kl(r) \rho^L+2_ij(r) dr
!
!  S3(L/2,i,j,k,l) = \int V3^L_kl(r) \rho^L_ij(r) dr
!
!
! N.B.: the potential V2^L(r) is not used here, instead lateron we
!       exploit the fact that:
!
!  \int V2^L_ij(r) \rho^L-2_kl(r) dr = S1((L-2)/2,k,l,i,j)  with L>=2
!
!***********************************************************************
      SUBROUTINE CALCSL4AE(W,L,R,N,S1,S2,S3)
      USE radial
      REAL(q) W(:,:)
      INTEGER L(:)
      TYPE (rgrid) R
      INTEGER N
      REAL(q) S1(0:,:,:,:,:),S2(0:,:,:,:,:),S3(0:,:,:,:,:)
      ! local variables
      REAL(q) RHO(R%NMAX),V1(R%NMAX),V2(R%NMAX),SUM
      INTEGER CH1,CH2,CH3,CH4
      INTEGER LLMAX,LL,LLP,LMIN12,LMAX12,LMIN34,LMAX34,LL1,LL2,I

      LLMAX=2*MAXVAL(L)
      IF (LLMAX/2>SIZE(S1,1).OR.LLMAX/2>SIZE(S2,1)) THEN
         WRITE(*,*) 'CALCSL4AE: ERROR: internal inconsistency:', &
        &   LLMAX/2,SIZE(S1,1),SIZE(S2,1),SIZE(S3,1)
         STOP
      ENDIF

      DO CH4=1,N
      DO CH3=1,N

         LL =L(CH3)
         LLP=L(CH4)
         LMIN34=ABS(LL-LLP); LMAX34=MIN(ABS(LL+LLP),6)

         RHO(:)=W(:,CH3)*W(:,CH4)

         ! loop over L
         DO LL1=LMIN34,LMAX34,2

            V1(1:R%NMAX)=0; V2(1:R%NMAX)=0
            CALL RAD_POT_DIP(LL1,R,RHO,V1,V2)

            DO CH2=1,N
            DO CH1=1,N

               LL =L(CH1)
               LLP=L(CH2)
               LMIN12=ABS(LL-LLP); LMAX12=MIN(ABS(LL+LLP),6)
               IF (MOD(LMAX12,2)/=MOD(LMAX34,2)) CYCLE

               LL2=LL1+2
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN
                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+V1(I)*W(I,CH1)*W(I,CH2)*R%SI(I)
                  ENDDO
                  S1(LL1/2,CH1,CH2,CH3,CH4)=S1(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

               LL2=LL1+2
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN
                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+RHO(I)/R%R(I)/R%R(I)*W(I,CH1)*W(I,CH2)*R%SI(I)
                  ENDDO
                  S2(LL1/2,CH1,CH2,CH3,CH4)=S2(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

               LL2=LL1
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN
                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+RHO(I)/R%R(I)/R%R(I)*W(I,CH1)*W(I,CH2)*R%SI(I)
                  ENDDO
                  S3(LL1/2,CH1,CH2,CH3,CH4)=S3(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

            ENDDO
            ENDDO
         ENDDO
      ENDDO
      ENDDO

      RETURN
      END SUBROUTINE CALCSL4AE


!********************** SUBROUTINE CALCSL4PS ****************************
!
!  \rho^L_kl(r) = <\tilde phi_k|r>< r|\tilde phi_l> + \hat \rho^L_kl(r)
!
! where <r|\tilde phi_i> are the pseudo partial waves and \hat \rho^L_ij(r)
! is the augmentation charge in the L-channel of the charge density arising
! from the product of <r|\tilde phi_i> and <r|\tilde phi_j>
!
!                  1
!  V1^L_kl(r) = ------- \int_0^r \rho^L_kl(r') r'^L dr'
!               r^(L+3)
!                                \rho^L_kl(r')
!  V2^L_kl(r) =  r^L-2 \int_r^oo ------------- dr'
!                                   r'^(L+1)
!
!  V3^L_kl(r) = \rho^L_kl(r)/r^2
!
! and with
!
!  \rho^L'_ij(r) = <\tilde phi_i|r>< r|\tilde phi_j> + \hat \rho^L_ij(r)
!
! we define (the output of this subroutine) 
!
!  S1(L/2,i,j,k,l) = \int V1^L_kl(r) \rho^L+2_ij(r) dr
!
!  S2(L/2,i,j,k,l) = \int V3^L_kl(r) \rho^L+2_ij(r) dr
!
!  S3(L/2,i,j,k,l) = \int V3^L_kl(r) \rho^L_ij(r) dr
!
!
! N.B.: the potential V2^L(r) is not used here, instead lateron we
!       exploit the fact that:
!
!  \int V2^L_ij(r) \rho^L-2_kl(r) dr = S1((L-2)/2,k,l,i,j)  with L>=2
!
!***********************************************************************
      SUBROUTINE CALCSL4PS(W,L,R,N,AUG,QPAW,AUG_FOCK,QPAW_FOCK,S1,S2,S3)
      USE radial
      REAL(q) W(:,:)
      INTEGER L(:)
      TYPE (rgrid) R
      INTEGER N
      REAL(q) AUG(:,0:)
      REAL(q) QPAW(:,:,0:)
      REAL(q), POINTER :: AUG_FOCK(:,:,:)
      REAL(q), POINTER :: QPAW_FOCK(:,:,:,:)
      REAL(q) S1(0:,:,:,:,:),S2(0:,:,:,:,:),S3(0:,:,:,:,:)
      ! local variables
      REAL(q) RHO12(R%NMAX),RHO34(R%NMAX),RHOAUG(R%NMAX),V1(R%NMAX),V2(R%NMAX),V3(R%NMAX),SUM
      INTEGER CH1,CH2,CH3,CH4
      INTEGER LLMAX,LL,LLP,LMIN12,LMAX12,LMIN34,LMAX34,LL1,LL2,I
      INTEGER LMAX_FOCKAE,NMAX_FOCKAE

      LLMAX=2*MAXVAL(L)
      IF (LLMAX/2>SIZE(S1,1).OR.LLMAX/2>SIZE(S2,1).OR.LLMAX/2>SIZE(S3,1)) THEN
         WRITE(*,*) 'CALCSL4PS: ERROR: internal inconsistency:', &
        &   LLMAX/2,SIZE(S1,1),SIZE(S2,1),SIZE(S3,1)
         STOP
      ENDIF

      IF (ASSOCIATED(QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(QPAW_FOCK,3)-1
      ELSE
         NMAX_FOCKAE=1
         LMAX_FOCKAE=-1
      ENDIF

      DO CH4=1,N
      DO CH3=1,N

         LL =L(CH3)
         LLP=L(CH4)
         LMIN34=ABS(LL-LLP); LMAX34=MIN(ABS(LL+LLP),6)

         RHO34(:)=W(:,CH3)*W(:,CH4)

         ! loop over L
         DO LL1=LMIN34,LMAX34,2

            RHOAUG(1:R%NMAX)=RHO34(1:R%NMAX)+AUG(1:R%NMAX,LL1)*QPAW(CH3,CH4,LL1)
            IF (LL1<=LMAX_FOCKAE) THEN
               DO I=1,NMAX_FOCKAE
                  RHOAUG(1:R%NMAX)=RHOAUG(1:R%NMAX)+AUG_FOCK(1:R%NMAX,LL1,I)*QPAW_FOCK(CH3,CH4,LL1,I)
               ENDDO
            ENDIF

            V1(1:R%NMAX)=0; V2(1:R%NMAX)=0
            CALL RAD_POT_DIP(LL1,R,RHOAUG,V1,V2)

            V3(1:R%NMAX)=RHOAUG(1:R%NMAX)/R%R(1:R%NMAX)/R%R(1:R%NMAX)

            DO CH2=1,N
            DO CH1=1,N

               LL =L(CH1)
               LLP=L(CH2)
               LMIN12=ABS(LL-LLP); LMAX12=MIN(ABS(LL+LLP),6)
               IF (MOD(LMAX12,2)/=MOD(LMAX34,2)) CYCLE

               RHO12(:)=W(:,CH1)*W(:,CH2)

               LL2=LL1+2
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN

                  RHOAUG(1:R%NMAX)=RHO12(1:R%NMAX)+AUG(1:R%NMAX,LL2)*QPAW(CH1,CH2,LL2)
                  IF (LL2<=LMAX_FOCKAE) THEN
                     DO I=1,NMAX_FOCKAE
                        RHOAUG(1:R%NMAX)=RHOAUG(1:R%NMAX)+AUG_FOCK(1:R%NMAX,LL2,I)*QPAW_FOCK(CH1,CH2,LL2,I)
                     ENDDO
                  ENDIF

                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+V1(I)*RHOAUG(I)*R%SI(I)
                  ENDDO
                  S1(LL1/2,CH1,CH2,CH3,CH4)=S1(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

               LL2=LL1+2
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN
                  RHOAUG(1:R%NMAX)=RHO12(1:R%NMAX)+AUG(1:R%NMAX,LL2)*QPAW(CH1,CH2,LL2)
                  IF (LL2<=LMAX_FOCKAE) THEN
                     DO I=1,NMAX_FOCKAE
                        RHOAUG(1:R%NMAX)=RHOAUG(1:R%NMAX)+AUG_FOCK(1:R%NMAX,LL2,I)*QPAW_FOCK(CH1,CH2,LL2,I)
                     ENDDO
                  ENDIF

                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+V3(I)*RHOAUG(I)*R%SI(I)
                  ENDDO
                  S2(LL1/2,CH1,CH2,CH3,CH4)=S2(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

               LL2=LL1
               IF (LL2>=LMIN12.AND.LL2<=LMAX12.AND.LL2<=LLMAX) THEN
                  RHOAUG(1:R%NMAX)=RHO12(1:R%NMAX)+AUG(1:R%NMAX,LL2)*QPAW(CH1,CH2,LL2)
                  IF (LL2<=LMAX_FOCKAE) THEN
                     DO I=1,NMAX_FOCKAE
                        RHOAUG(1:R%NMAX)=RHOAUG(1:R%NMAX)+AUG_FOCK(1:R%NMAX,LL2,I)*QPAW_FOCK(CH1,CH2,LL2,I)
                     ENDDO
                  ENDIF

                  SUM=0._q
                  DO I=1,R%NMAX
                     SUM=SUM+V3(I)*RHOAUG(I)*R%SI(I)
                  ENDDO
                  S3(LL1/2,CH1,CH2,CH3,CH4)=S3(LL1/2,CH1,CH2,CH3,CH4)+SUM
               ENDIF

            ENDDO
            ENDDO
         ENDDO
      ENDDO
      ENDDO

      RETURN
      END SUBROUTINE CALCSL4PS


!********************** SUBROUTINE RAD_POT_DIP *************************
!
!               1
!  POT1(r) = ------- \int_0^r \rho(r') r'^L dr'
!            r^(L+3)
!                             \rho(r')
!  POT2(r) =  r^L-2 \int_r^oo -------- dr'
!                             r'^(L+1)
!
!***********************************************************************
      SUBROUTINE RAD_POT_DIP(LL,R,RHO,POT1,POT2)
      USE radial
      USE constant
      IMPLICIT NONE
      INTEGER LL
      TYPE (rgrid) R
      REAL(q) RHO(:),POT1(:),POT2(:)
      ! local variables
      REAL(q) T1(R%NMAX),T2(R%NMAX),V1(R%NMAX),V2(R%NMAX),RL(R%NMAX)
      REAL(q) H3,EXT,RGR,RSM
      INTEGER LMMAX,M,LM,LMBASE,LMP
      INTEGER N,I,J,K,L

      IF (SIZE(RHO)<R%NMAX) THEN
         WRITE(*,*) 'RAD_POT_DIP: ERROR: RHO incorrectly dimensioned:', &
        &   R%NMAX,SIZE(RHO)
         STOP
      ENDIF
      IF (SIZE(POT1)<R%NMAX.OR.SIZE(POT2)<R%NMAX) THEN
         WRITE(*,*) 'RAD_POT_DIP: ERROR: POT incorrectly dimensioned:', &
        &   R%NMAX,SIZE(POT1),SIZE(POT2)
         STOP
      ENDIF

      N=R%NMAX      
      I=0
      DO K=N,1,-1
         RL(K)=R%R(K)**LL
         I=I+1
         T2(I)=RHO(K)/RL(K)
         T1(I)=RL(K)*R%R(K)*RHO(K)
      ENDDO
      H3=R%H/ 3.0_q
      ! integrate inward (assuming zero potential for grid point NMAX)
      ! V1 = \int_R^Inf RHO(r) r^l dr
      ! V1 is essentially the moment l of the charge
      V1(1)=0
      DO L=3,N,2
         V1(L)  =V1(L-2)+H3*(T1(L-2)+4.0_q*T1(L-1)+T1(L))
         V1(L-1)=V1(L-2)+H3*(1.25_q*T1(L-2)+2.0_q*T1(L-1)-0.25_q*T1(L))
      ENDDO
      IF (MOD(N,2)==0) V1(N)=V1(N-2)+H3*(T1(N-2)+4.0_q*T1(N-1)+T1(N))

      ! V2 = \int_R^Inf RHO(r) r^(-l-1) dr
      V2(1)=0
      DO L=3,N,2
         V2(L)  =V2(L-2)+H3*(T2(L-2)+4.0_q*T2(L-1)+T2(L))
         V2(L-1)=V2(L-2)+H3*(1.25_q*T2(L-2)+2.0_q*T2(L-1)-0.25_q*T2(L))
      ENDDO
      IF (MOD(N,2)==0) V2(N)=V2(N-2)+H3*(T2(N-2)+4.0_q*T2(N-1)+T2(N))
 
      EXT=V1(N); I=0
      DO K=N,1,-1
         I=I+1
         POT1(I)=(EXT-V1(K))/RL(I)/R%R(I)/R%R(I)/R%R(I)
         POT2(I)=V2(K)*RL(I)/R%R(I)/R%R(I)
      ENDDO

      RETURN
      END SUBROUTINE RAD_POT_DIP

      END MODULE dmatrix


!********************** SUBROUTINE CONTRACT_GFAC ***********************
!
!  this subroutine multiplies the charge density by the potential kernel
!  (essentially e^2 epsilon_0 / 4 pi (G +k-q)^2)
!
!***********************************************************************

     SUBROUTINE CONTRACT_GFAC(GRID, CWORK, POTFAK, RES)
     USE prec
     USE mgrid
     IMPLICIT NONE
     TYPE (grid_3d) GRID
     COMPLEX(q) CWORK(GRID%MPLWV)
     REAL(q) POTFAK(GRID%MPLWV)
     REAL(q) RES
     ! local variables
     INTEGER NP
     RES=0._q
     DO NP=1,GRID%RC%NP
        RES=RES+CONJG(CWORK(NP))*POTFAK(NP)*CWORK(NP)
     ENDDO
     END SUBROUTINE CONTRACT_GFAC


!********************** SUBROUTINE GNORM *******************************
!
!***********************************************************************

     SUBROUTINE GNORM(GRID, CWORK)
     USE prec
     USE mgrid
     IMPLICIT NONE
     TYPE (grid_3d) GRID
     COMPLEX(q) CWORK(GRID%MPLWV)
     ! local variables
     INTEGER NP
     DO NP=1,GRID%RC%NP
        CWORK(NP)=CONJG(CWORK(NP))*CWORK(NP)
     ENDDO
     END SUBROUTINE GNORM


!********************** SUBROUTINE MULT_GCHG ***************************
!
!***********************************************************************

     SUBROUTINE MULT_GCHG(GRID, CWORK1, CWORK2)
     USE prec
     USE mgrid
     IMPLICIT NONE
     TYPE (grid_3d) GRID
     COMPLEX(q) CWORK1(GRID%MPLWV)
     COMPLEX(q) CWORK2(GRID%MPLWV)
     ! local variables
     INTEGER NP
     DO NP=1,GRID%RC%NP
!       CWORK2(NP)=CONJG(CWORK1(NP))*CWORK2(NP)
        CWORK2(NP)=CWORK1(NP)*CONJG(CWORK2(NP))
     ENDDO
     END SUBROUTINE MULT_GCHG


!********************** SUBROUTINE MULT_GFAC ***************************
!
!  this subroutine multiplies the charge density by the potential kernel
!  (essentially e^2 epsilon_0 / 4 pi (G +k-q)^2)
!
!***********************************************************************

     SUBROUTINE MULT_GFAC(GRID, CWORK, POTFAK, RES)
     USE prec
     USE mgrid
     IMPLICIT NONE
     TYPE (grid_3d) GRID
     COMPLEX(q) CWORK(GRID%MPLWV)
     REAL(q) POTFAK(GRID%MPLWV)
     REAL(q) RES
     ! local variables
     INTEGER NP
     RES=0._q
     DO NP=1,GRID%RC%NP
        RES=RES+POTFAK(NP)*CWORK(NP)
     ENDDO
     END SUBROUTINE MULT_GFAC


!******************** SUBROUTINE GRID_TO_RADIAL ************************
!
!***********************************************************************
      SUBROUTINE GRID_TO_RADIAL(GRID,WCW,LATT_CUR,R,RMAX,LYDIM,ENCUT)
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE asa
      IMPLICIT NONE
      TYPE (grid_3d) GRID
      COMPLEX(q) WCW(GRID%MPLWV)
      TYPE (latt) LATT_CUR
      INTEGER LYDIM
      REAL(q) R(3)
      REAL(q) RMAX
      REAL(q) ENCUT
      ! local variables
      INTEGER LLMAX,LMMAX,PHPTS,THPTS,NPTS
      INTEGER I,J,IFAIL
      REAL(q) DELTAPHI,SIM_FAKT
      REAL(q), ALLOCATABLE :: RADPTS(:,:), XYZPTS(:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)
      REAL(q), ALLOCATABLE :: YLM(:,:)
      EXTERNAL GAUSSI2

      INTEGER, PARAMETER :: NR=101
      REAL(q), ALLOCATABLE :: RP(:,:,:),VR(:,:),CR(:,:),VTMP(:,:),CTMP(:,:)
      REAL(q) X,Y,Z

      REAL(q) GX,GY,GZ,GSQU,POTFAK,FACTM
      INTEGER NC,N1,N2,N3,IG,G1,G2,G3,LM

      LLMAX=LYDIM
      LMMAX=(LLMAX+1)**2
      ! number of theta and phi pivot points
      ! the value below perform the integration exactly without error
      ! less is not possible more is not required
      PHPTS=(LLMAX+1)*3
      THPTS=FLOOR(REAL(LLMAX/2+1,KIND=q))*3
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
      ! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX))
      ALLOCATE(XYZPTS(NPTS,3),RADPTS(NPTS,2))
      ALLOCATE(WEIGHT(THPTS),ABSCIS(THPTS))

      RADPTS=0; WEIGHT=0; ABSCIS=0
      ! set phi positions, equally spaces
      DO I=1,PHPTS
         DO J=1,THPTS
            RADPTS((J-1)*PHPTS+I,2)=(I-1)*DELTAPHI
         ENDDO
      ENDDO
      ! get theta positions (actually get cos(theta)) (Gauss integration)
      CALL GAUSSI(GAUSSI2,-1._q,1._q,0,THPTS,WEIGHT,ABSCIS,IFAIL)
      DO I=1,THPTS
         RADPTS((I-1)*PHPTS+1:I*PHPTS,1)=ABSCIS(I)
      ENDDO
      ! convert radial to cartesian coordinates
      DO I=1,NPTS
         XYZPTS(I,1)=COS(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! x
         XYZPTS(I,2)=SIN(RADPTS(I,2))*SQRT(1_q-RADPTS(I,1)**2_q) ! y
         XYZPTS(I,3)=RADPTS(I,1)                                 ! z
      ENDDO

      CALL SETYLM(LLMAX,NPTS,YLM,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

      ALLOCATE(RP(3,NPTS,NR),VR(NR,LMMAX),CR(NR,LMMAX),VTMP(NPTS,NR),CTMP(NPTS,NR))

      DO I=1,NR
         DO J=1,NPTS
            X=XYZPTS(J,1)*MAX((I-1)*RMAX/(NR-1),1.E-4_q)
            Y=XYZPTS(J,2)*MAX((I-1)*RMAX/(NR-1),1.E-4_q)
            Z=XYZPTS(J,3)*MAX((I-1)*RMAX/(NR-1),1.E-4_q)
            ! transform to direct coordinates
            RP(1,J,I)=X*LATT_CUR%B(1,1)+Y*LATT_CUR%B(2,1)+Z*LATT_CUR%B(3,1)+R(1)
            RP(2,J,I)=X*LATT_CUR%B(1,2)+Y*LATT_CUR%B(2,2)+Z*LATT_CUR%B(3,2)+R(2)
            RP(3,J,I)=X*LATT_CUR%B(1,3)+Y*LATT_CUR%B(2,3)+Z*LATT_CUR%B(3,3)+R(3)
         ENDDO
      ENDDO

      VTMP=0._q; CTMP=0._q

      IG=0
      col: DO NC=1,GRID%RC%NCOL
         N2=GRID%RC%I2(NC)
         N3=GRID%RC%I3(NC)
         row: DO N1=1,GRID%RC%NROW
            IG=IG+1

            FACTM=1._q
#ifdef gammareal
            IF (N1/=1) FACTM=2._q
#endif
            G1=GRID%LPCTX(N1); G2=GRID%LPCTY(N2); G3=GRID%LPCTZ(N3)

            GX=(GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3))
            GY=(GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3))
            GZ=(GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)* &
                 LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3))

            GSQU=GX**2+GY**2+GZ**2
            IF (GSQU<1.E-6_q) THEN
               POTFAK=0._q
            ELSE
               POTFAK=FACTM*(GX*GX/GSQU-1._q/3._q)
!              POTFAK=FACTM*(GY*GY/GSQU-1._q/3._q)
!              POTFAK=FACTM*(GZ*GZ/GSQU-1._q/3._q)
!              POTFAK=FACTM*(GX*GY/GSQU)
!              POTFAK=FACTM*(GX*GZ/GSQU)
!              POTFAK=FACTM*(GY*GZ/GSQU)
            ENDIF

!           IF (HSQDTM*TPI*TPI*GSQU>ENCUT) THEN
!              POTFAK=0._q
!           ENDIF

            DO I=1,NR
               DO J=1,NPTS
                  CTMP(J,I)=CTMP(J,I)+WCW(IG)*EXP(CITPI*(RP(1,J,I)*G1+RP(2,J,I)*G2+RP(3,J,I)*G3))*FACTM
                  VTMP(J,I)=VTMP(J,I)+WCW(IG)*EXP(CITPI*(RP(1,J,I)*G1+RP(2,J,I)*G2+RP(3,J,I)*G3))*POTFAK
!                 VTMP(J,I)=VTMP(J,I)+EXP(CITPI*(RP(1,J,I)*G1+RP(2,J,I)*G2+RP(3,J,I)*G3))*POTFAK
               ENDDO
            ENDDO

         ENDDO row
      ENDDO col

      DO LM=1,LMMAX
         CR(:,LM)=0._q; VR(:,LM)=0._q
         DO I=1,NR
            DO J=1,NPTS
               SIM_FAKT=DELTAPHI*WEIGHT((INT((J-1)/PHPTS)+1))
               VR(I,LM)=VR(I,LM)+YLM(J,LM)*VTMP(J,I)*SIM_FAKT
               CR(I,LM)=CR(I,LM)+YLM(J,LM)*CTMP(J,I)*SIM_FAKT
            ENDDO
         ENDDO
! test_
         VR(:,LM)=VR(:,LM)/LATT_CUR%OMEGA/GRID%NPLWV
         CR(:,LM)=CR(:,LM)/LATT_CUR%OMEGA/GRID%NPLWV
         DO I=1,NR
            WRITE(9000+LM,'(3E15.6)') MAX((I-1)*RMAX/(NR-1),1.E-4_q),CR(I,LM),VR(I,LM)
         ENDDO
! test_
      ENDDO

      DEALLOCATE(RP,VR,CR,VTMP,CTMP)

      RETURN
      END SUBROUTINE GRID_TO_RADIAL


!******************** FUNCTION LDMATRIX ********************************
!
!***********************************************************************
      FUNCTION LDMATRIX()
      USE dmatrix
      LOGICAL LDMATRIX
      LDMATRIX=LDO_DMATRIX
      END FUNCTION LDMATRIX
