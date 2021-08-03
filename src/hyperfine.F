#include "symbol.inc"
!#define s_waves_only
!#define fc_ae1c_only
!#define zora_simple
#define zora_bloechl
!#define corepol
#define corepol_my
      MODULE hyperfine
      USE prec

      IMPLICIT NONE

      LOGICAL, PRIVATE, SAVE :: LDO_HYPERFINE

      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: NUCLEAR_GYROMAGN_RATIO(:)

      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_PW_ISO(:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_PW_ANI(:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_RAD_ISO_AE(:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_RAD_ISO_PS(:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_RAD_ISO_C(:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_RAD_ANI_AE(:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: HYPERFINE_RAD_ANI_PS(:,:)

      REAL(q), PRIVATE, SAVE :: HYPERFINE_S_TOT
      
      PRIVATE :: POLINT

      CONTAINS

!******************** SUBROUTINE HYPERFINE_READER **********************
!
!***********************************************************************
      SUBROUTINE HYPERFINE_READER(IU5,IU6,IU0,NTYP)
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

      LDO_HYPERFINE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LHYPERFINE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LDO_HYPERFINE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LHYPERFINE'' from file INCAR.'
         LDO_HYPERFINE=.FALSE.
      ENDIF

      CALL XML_INCAR('LHYPERFINE','L',IDUM,RDUM,CDUM,LDO_HYPERFINE,CHARAC,N)

      ALLOCATE(NUCLEAR_GYROMAGN_RATIO(NTYP))
      NUCLEAR_GYROMAGN_RATIO=1._q
      CALL RDATAB(LOPEN,INCAR,IU5,'NGYROMAG','=','#',';','F', &
     &            IDUM,NUCLEAR_GYROMAGN_RATIO,CDUM,LDUM,CHARAC,N,NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<NTYP))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''NGYROMAG'' from file INCAR.'
         NUCLEAR_GYROMAGN_RATIO=1._q
      ENDIF

      CALL XML_INCAR_V('NGYROMAG','F',IDUM,NUCLEAR_GYROMAGN_RATIO,CDUM,LDUM,CHARAC,N)

      CLOSE(IU5)

      RETURN
      END SUBROUTINE HYPERFINE_READER


!******************** SUBROUTINE HYPERFINE_WRITER **********************
!
!***********************************************************************
      SUBROUTINE HYPERFINE_WRITER(IU6)
      INTEGER IU6
      IF (IU6>=0) THEN
         WRITE(IU6,10) LDO_HYPERFINE
      ENDIF
   10 FORMAT(' Calculation of hyperfine parameters:'/ &
     &       '   LHYPERFINE =',L6/)
      RETURN
      END SUBROUTINE HYPERFINE_WRITER


!******************** FUNCTION HYPERFINE *******************************
!
!***********************************************************************
      FUNCTION LHYPERFINE()
      LOGICAL LHYPERFINE
      LHYPERFINE=LDO_HYPERFINE
      END FUNCTION LHYPERFINE


!******************** SUBROUTINE HYPERFINE_WRITER **********************
!
!***********************************************************************
      SUBROUTINE HYPERFINE_WRITE_TENSORS(T_INFO,WDES,IO)
      USE base
      USE poscar
      USE wave_high
      USE paw
      USE constant
      TYPE(type_info) T_INFO
      TYPE(wavedes) WDES
      TYPE(in_struct) IO
      ! local variables
      INTEGER NI,I
      REAL(q), ALLOCATABLE :: HYPERFINE_TENSOR(:,:)
      REAL(q), ALLOCATABLE :: HYPERFINE_TENSOR_DIAG(:,:)
      REAL(q), ALLOCATABLE :: EIGV(:,:,:)
      REAL(q) TMP,ETA,S

      INTEGER IFAIL,IDONE
      INTEGER, PARAMETER :: LWORK=6
      REAL(q) A(3,3),EIG(3),WORK(3*LWORK)

      LOGICAL LSKIP

      REAL(q), PARAMETER :: SCALE_iso=2._q/3._q*4._q*PI*928.476377_q*2._q*1.E-3_q
      REAL(q), PARAMETER :: SCALE_ani=928.476377*2._q*1.E-3_q

      ! quick return if possible
      IF (.NOT.LHYPERFINE()) RETURN

      ! if this node didn't do any work in SET_DD_PAW then the
      ! HYPERFINE_RAD_* arrays were not allocated before
      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ISO_AE)) THEN
         ALLOCATE(HYPERFINE_RAD_ISO_AE(T_INFO%NIONS),HYPERFINE_RAD_ISO_PS(T_INFO%NIONS), &
        &   HYPERFINE_RAD_ISO_C(T_INFO%NIONS),HYPERFINE_RAD_ANI_AE(6,T_INFO%NIONS),HYPERFINE_RAD_ANI_PS(6,T_INFO%NIONS))
      ENDIF

      IF (ALLOCATED(DO_LOCAL)) THEN
         ! zero one-center contributions that
         ! were not calculated on this node
         IDONE=0
         DO NI=1,T_INFO%NIONS
            LSKIP=.FALSE.
#ifdef MPI
            ! DO_LOCAL represents a distribution of the work on the
            ! one-center terms over the procs in COMM_INB and COMM_INTER (=COMM_KIN).
            ! The following allows an additional round-robin distribution over COMM_KINTER as well.
            IF (DO_LOCAL(NI)) THEN
               IDONE=IDONE+1; LSKIP=(MOD(IDONE,WDES%COMM_KINTER%NCPU)+1/=WDES%COMM_KINTER%NODE_ME)
            ENDIF
#endif
            IF (.NOT.DO_LOCAL(NI).OR.LSKIP) THEN
               HYPERFINE_RAD_ISO_AE(NI)=0._q
               HYPERFINE_RAD_ISO_PS(NI)=0._q
               HYPERFINE_RAD_ISO_C(NI)=0._q
               HYPERFINE_RAD_ANI_AE(:,NI)=0._q
               HYPERFINE_RAD_ANI_PS(:,NI)=0._q
            ENDIF
         ENDDO
         ! communicate one-center contributions (is not done in SET_DD_PAW)
         CALLMPI( M_sum_d(WDES%COMM,HYPERFINE_RAD_ISO_AE,  T_INFO%NIONS) )
         CALLMPI( M_sum_d(WDES%COMM,HYPERFINE_RAD_ISO_PS,  T_INFO%NIONS) )
         CALLMPI( M_sum_d(WDES%COMM,HYPERFINE_RAD_ISO_C ,  T_INFO%NIONS) )
         CALLMPI( M_sum_d(WDES%COMM,HYPERFINE_RAD_ANI_AE,6*T_INFO%NIONS) )
         CALLMPI( M_sum_d(WDES%COMM,HYPERFINE_RAD_ANI_PS,6*T_INFO%NIONS) )
      ELSE
         HYPERFINE_RAD_ISO_AE=0._q; HYPERFINE_RAD_ISO_PS=0._q; HYPERFINE_RAD_ISO_C=0._q
         HYPERFINE_RAD_ANI_AE=0._q; HYPERFINE_RAD_ANI_PS=0._q
      ENDIF

      S=1._q
      IF (ABS(HYPERFINE_S_TOT)>1.E-3) THEN
         S=ABS(HYPERFINE_S_TOT)
      ENDIF

      DO NI=1,T_INFO%NIONS 
         HYPERFINE_PW_ISO(NI)=HYPERFINE_PW_ISO(NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_iso/S
         HYPERFINE_RAD_ISO_AE(NI)=HYPERFINE_RAD_ISO_AE(NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_iso/S
         HYPERFINE_RAD_ISO_PS(NI)=HYPERFINE_RAD_ISO_PS(NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_iso/S
         HYPERFINE_RAD_ISO_C(NI)=HYPERFINE_RAD_ISO_C(NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_iso/S
         HYPERFINE_PW_ANI(:,NI)=HYPERFINE_PW_ANI(:,NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_ani/S
         HYPERFINE_RAD_ANI_AE(:,NI)=HYPERFINE_RAD_ANI_AE(:,NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_ani/S
         HYPERFINE_RAD_ANI_PS(:,NI)=HYPERFINE_RAD_ANI_PS(:,NI)*NUCLEAR_GYROMAGN_RATIO(T_INFO%ITYP(NI))*SCALE_ani/S
      ENDDO

      ! quick return if node is not supposed to do io
      IF(IO%IU6<0) RETURN

      WRITE(IO%IU6,'(/A,F14.7)') ' Total magnetic moment S=',S

      ALLOCATE(HYPERFINE_TENSOR(6,T_INFO%NIONS),HYPERFINE_TENSOR_DIAG(3,T_INFO%NIONS),EIGV(3,3,T_INFO%NIONS))

      WRITE(IO%IU6,'(/A)')  ' Fermi contact (isotropic) hyperfine coupling parameter (MHz) '
      WRITE(IO%IU6,'(A)')   ' -------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')   '  ion      A_pw      A_1PS     A_1AE     A_1c      A_tot'
      WRITE(IO%IU6,'(A)')   ' -------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
         WRITE(IO%IU6,'(I4,2X,6F10.3)') NI, &
        &   HYPERFINE_PW_ISO(NI),HYPERFINE_RAD_ISO_PS(NI),HYPERFINE_RAD_ISO_AE(NI),HYPERFINE_RAD_ISO_C(NI), &
        &   HYPERFINE_PW_ISO(NI)-HYPERFINE_RAD_ISO_PS(NI)+HYPERFINE_RAD_ISO_AE(NI)!+HYPERFINE_RAD_ISO_C(NI)
      ENDDO
      WRITE(IO%IU6,'(A/)')  ' -------------------------------------------------------------'

      HYPERFINE_TENSOR=HYPERFINE_PW_ANI+HYPERFINE_RAD_ANI_AE-HYPERFINE_RAD_ANI_PS

      WRITE(IO%IU6,'(/A)')  ' Dipolar hyperfine coupling parameters (MHz)'
      WRITE(IO%IU6,'(A)')   ' ---------------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')   '  ion      A_xx      A_yy      A_zz      A_xy      A_xz      A_yz'
      WRITE(IO%IU6,'(A)')   ' ---------------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
!#define debug
#ifdef debug
         WRITE(1000+NI,'(A)') ' PW'
         WRITE(1000+NI,'(I4,2X,6F10.3)') NI,HYPERFINE_PW_ANI(:,NI)
         WRITE(1000+NI,'(A)') ' one-center PS'
         WRITE(1000+NI,'(I4,2X,6F10.3)') NI,HYPERFINE_RAD_ANI_PS(:,NI)
         WRITE(1000+NI,'(A)') ' one-center AE'
         WRITE(1000+NI,'(I4,2X,6F10.3)') NI,HYPERFINE_RAD_ANI_AE(:,NI)
#endif
         WRITE(IO%IU6,'(I4,2X,6F10.3)') NI,HYPERFINE_TENSOR(:,NI)
      ENDDO
      WRITE(IO%IU6,'(A/)')   ' ---------------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
         HYPERFINE_TENSOR(1:3,NI)=HYPERFINE_TENSOR(1:3,NI)+HYPERFINE_PW_ISO(NI)- &
        &   HYPERFINE_RAD_ISO_PS(NI)+HYPERFINE_RAD_ISO_AE(NI)!+HYPERFINE_RAD_ISO_C(NI)
      ENDDO

      WRITE(IO%IU6,'(/A)')  ' Total hyperfine coupling parameters after diagonalization (MHz)'
      WRITE(IO%IU6,'(A)')   ' (convention: |A_zz| > |A_xx| > |A_yy|)'
      WRITE(IO%IU6,'(A)')   ' ----------------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')   '  ion      A_xx      A_yy      A_zz     asymmetry (A_yy - A_xx)/ A_zz '
      WRITE(IO%IU6,'(A)')   ' ----------------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
         ! Get eigenvalues and eigenvectors
         A=0
         A(1,1)=HYPERFINE_TENSOR(1,NI)
         A(1,2)=HYPERFINE_TENSOR(4,NI)
         A(1,3)=HYPERFINE_TENSOR(5,NI)
         A(2,2)=HYPERFINE_TENSOR(2,NI)
         A(2,3)=HYPERFINE_TENSOR(6,NI)
         A(3,3)=HYPERFINE_TENSOR(3,NI)
         CALL DSYEV('V','U',3,A,3,EIG,WORK,3*LWORK,IFAIL)
         EIGV(:,:,NI)=A
         HYPERFINE_TENSOR_DIAG(1,NI)=EIG(1)
         HYPERFINE_TENSOR_DIAG(2,NI)=EIG(2)
         HYPERFINE_TENSOR_DIAG(3,NI)=EIG(3)

         ! (convention: |A_zz| > |A_xx| > |A_yy|)
         DO I=1,2
            IF (ABS(HYPERFINE_TENSOR_DIAG(1,NI))>ABS(HYPERFINE_TENSOR_DIAG(2,NI))) THEN
               TMP=HYPERFINE_TENSOR_DIAG(2,NI)
               HYPERFINE_TENSOR_DIAG(2,NI)=HYPERFINE_TENSOR_DIAG(1,NI)
               HYPERFINE_TENSOR_DIAG(1,NI)=TMP
            ENDIF
            IF (ABS(HYPERFINE_TENSOR_DIAG(2,NI))>ABS(HYPERFINE_TENSOR_DIAG(3,NI))) THEN
               TMP=HYPERFINE_TENSOR_DIAG(3,NI)
               HYPERFINE_TENSOR_DIAG(3,NI)=HYPERFINE_TENSOR_DIAG(2,NI)
               HYPERFINE_TENSOR_DIAG(2,NI)=TMP
            ENDIF            
         ENDDO
         ETA=(HYPERFINE_TENSOR_DIAG(1,NI)-HYPERFINE_TENSOR_DIAG(2,NI))/HYPERFINE_TENSOR_DIAG(3,NI)
         WRITE(IO%IU6,'(I4,2X,3F10.3,2X,F10.3)') &
        &   NI,HYPERFINE_TENSOR_DIAG(2,NI),HYPERFINE_TENSOR_DIAG(1,NI),HYPERFINE_TENSOR_DIAG(3,NI),ETA
      ENDDO
      WRITE(IO%IU6,'(A/)')   ' ---------------------------------------------------------------------'

      DEALLOCATE(HYPERFINE_TENSOR,HYPERFINE_TENSOR_DIAG,EIGV)

      ! cleanup
      IF (ALLOCATED(HYPERFINE_PW_ISO)) DEALLOCATE(HYPERFINE_PW_ISO)
      IF (ALLOCATED(HYPERFINE_PW_ANI)) DEALLOCATE(HYPERFINE_PW_ANI)
      IF (ALLOCATED(HYPERFINE_RAD_ISO_AE)) DEALLOCATE(HYPERFINE_RAD_ISO_AE)
      IF (ALLOCATED(HYPERFINE_RAD_ISO_PS)) DEALLOCATE(HYPERFINE_RAD_ISO_PS)
      IF (ALLOCATED(HYPERFINE_RAD_ISO_C)) DEALLOCATE(HYPERFINE_RAD_ISO_C)
      IF (ALLOCATED(HYPERFINE_RAD_ANI_AE)) DEALLOCATE(HYPERFINE_RAD_ANI_AE)
      IF (ALLOCATED(HYPERFINE_RAD_ANI_PS)) DEALLOCATE(HYPERFINE_RAD_ANI_PS)

      RETURN
      END SUBROUTINE HYPERFINE_WRITE_TENSORS


!******************** SUBROUTINE HYPERFINE_PW **************************
!
! N.B.: CHTOT must contain the magnetisation density
!
!***********************************************************************
      SUBROUTINE HYPERFINE_PW(T_INFO,LATT_CUR,GRIDC,CHTOT)
      USE poscar
      USE lattice
      USE mgrid
      USE constant
      TYPE(type_info) T_INFO
      TYPE(latt) LATT_CUR
      TYPE(grid_3d) GRIDC
      COMPLEX(q) CHTOT(GRIDC%RC%NP)
      ! local variables
      INTEGER NI,I,N1,N2,N3,NC
      REAL(q) FACTM,GX,GY,GZ,GSQU
      COMPLEX(q) CGR

      ! quick return if possible
      IF (.NOT.LHYPERFINE()) RETURN

      IF (.NOT.ALLOCATED(HYPERFINE_PW_ISO)) ALLOCATE(HYPERFINE_PW_ISO(T_INFO%NIONS))
      HYPERFINE_PW_ISO=0._q
      
      IF (.NOT.ALLOCATED(HYPERFINE_PW_ANI)) ALLOCATE(HYPERFINE_PW_ANI(6,T_INFO%NIONS))
      HYPERFINE_PW_ANI=0._q

      HYPERFINE_S_TOT=0._q

      I=0
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
 
         I=I+1
         SETFACT1
         SETFACT
 
         ! x,y,z components and norm of G vector
         GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
         GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
         GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI
 
         GSQU=GX**2+GY**2+GZ**2

         ions: DO NI=1,T_INFO%NIONS
            ! phase factor         
            CGR=EXP(CITPI*(GRIDC%LPCTX(N1)*T_INFO%POSION(1,NI)+GRIDC%LPCTY(N2)*T_INFO%POSION(2,NI)+GRIDC%LPCTZ(N3)*T_INFO%POSION(3,NI)))

            ! isotropic (Fermi contact) contribution 
            HYPERFINE_PW_ISO(NI)=HYPERFINE_PW_ISO(NI)+MULFACT CHTOT(I)*CGR

            IF (GSQU<1.E-6) THEN
               ! Store G=0 component
               HYPERFINE_S_TOT=CHTOT(I)
            ELSE
               ! anisotropic (dipolar) contribution
               HYPERFINE_PW_ANI(1,NI)=HYPERFINE_PW_ANI(1,NI)+MULFACT CHTOT(I)*(GX*GX/GSQU-1._q/3._q)*CGR
               HYPERFINE_PW_ANI(2,NI)=HYPERFINE_PW_ANI(2,NI)+MULFACT CHTOT(I)*(GY*GY/GSQU-1._q/3._q)*CGR
               HYPERFINE_PW_ANI(3,NI)=HYPERFINE_PW_ANI(3,NI)+MULFACT CHTOT(I)*(GZ*GZ/GSQU-1._q/3._q)*CGR
               HYPERFINE_PW_ANI(4,NI)=HYPERFINE_PW_ANI(4,NI)+MULFACT CHTOT(I)*(GX*GY/GSQU)*CGR
               HYPERFINE_PW_ANI(5,NI)=HYPERFINE_PW_ANI(5,NI)+MULFACT CHTOT(I)*(GX*GZ/GSQU)*CGR
               HYPERFINE_PW_ANI(6,NI)=HYPERFINE_PW_ANI(6,NI)+MULFACT CHTOT(I)*(GY*GZ/GSQU)*CGR
            ENDIF
         ENDDO ions

      ENDDO row
      ENDDO col

      HYPERFINE_PW_ISO= HYPERFINE_PW_ISO/LATT_CUR%OMEGA
      HYPERFINE_PW_ANI=-HYPERFINE_PW_ANI/LATT_CUR%OMEGA*4._q*PI

      ! communicate
      CALLMPI( M_sum_d(GRIDC%COMM,HYPERFINE_PW_ISO,  T_INFO%NIONS) )
      CALLMPI( M_sum_d(GRIDC%COMM,HYPERFINE_PW_ANI,6*T_INFO%NIONS) )
      CALLMPI( M_sum_d(GRIDC%COMM,HYPERFINE_S_TOT,1) )

      RETURN
      END SUBROUTINE HYPERFINE_PW


!******************** SUBROUTINE HYPERFINE_RAD *************************
!
! N.B.: RHOAE and RHOPS must contain the LM decomposed all-electron
!       and pseudo total and magnetisation density (tot,mag),
!       COCC contains the occupations in (up,dn) storage mode
!
!***********************************************************************
      SUBROUTINE HYPERFINE_RAD(T_INFO,NI,PP,RHOPS,RHOAE,POTAE,COCC)
      USE poscar
      USE pseudo
      USE constant
      USE radial
#ifdef corepol
      USE rhfatm
      USE setexm
#endif
#ifdef corepol_my
      USE cl
      USE setexm
#endif
      TYPE(type_info) T_INFO
      TYPE(potcar), POINTER :: PP
      INTEGER NI
      REAL(q) RHOAE(:,:,:),RHOPS(:,:,:),POTAE(:,:,:)
      OVERLAP COCC(:,:,:)
      ! local variables
#ifdef corepol
      TYPE(atomic), ALLOCATABLE, SAVE :: ATOM(:)
      REAL(q), ALLOCATABLE :: C(:,:) 
      REAL(q) DV(PP%R%NMAX),RHOC(PP%R%NMAX)
      INTEGER NT

      REAL(q) RHOUP,RHODN
      REAL(q) EXC,EXCD(2),EXCDD(2,2),EXCDDD(2,2,2)
#endif      
#ifdef corepol_my
      REAL(q) RHOC(PP%R%NMAX),RHO(PP%R%NMAX,1,2),POT(PP%R%NMAX,1,2),DMY(PP%R%NMAX),RHOP(PP%R%NMAX,2),DRHO
      REAL(q), ALLOCATABLE :: WUP(:,:),WDN(:,:),EUP(:),EDN(:)
      INTEGER, ALLOCATABLE :: N(:),LC(:)
      INTEGER MAXNL

      REAL(q), PARAMETER :: MIX=0.2_q, RCONV=1.E-8_q
      INTEGER, PARAMETER :: NSCF=100

      REAL(q) DV(PP%R%NMAX)
      REAL(q) RHOUP,RHODN
      REAL(q) EXC,EXCD(2),EXCDD(2,2),EXCDDD(2,2,2)
#endif
      INTEGER LM,I,J,K,LM1,LM2
      REAL(q) SCALE,SUM(5),RHOW(PP%R%NMAX),RES
      REAL(q) TMP(0:4),R(4),DUMMY
      INTEGER, PARAMETER :: ISTRIDE=1

      OVERLAP COCC_SPIN
      REAL(q), ALLOCATABLE :: RHOAE_(:),RHOPS_(:)

      ! quick return if possible
      IF (.NOT.LHYPERFINE().OR.SIZE(COCC,3)==1) RETURN

      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ISO_AE)) ALLOCATE(HYPERFINE_RAD_ISO_AE(T_INFO%NIONS))
      HYPERFINE_RAD_ISO_AE(NI)=0._q

      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ISO_PS)) ALLOCATE(HYPERFINE_RAD_ISO_PS(T_INFO%NIONS))
      HYPERFINE_RAD_ISO_PS(NI)=0._q

      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ISO_C)) ALLOCATE(HYPERFINE_RAD_ISO_C(T_INFO%NIONS))
      HYPERFINE_RAD_ISO_C(NI)=0._q

      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ANI_AE)) ALLOCATE(HYPERFINE_RAD_ANI_AE(6,T_INFO%NIONS))
      HYPERFINE_RAD_ANI_AE(:,NI)=0._q

      IF (.NOT.ALLOCATED(HYPERFINE_RAD_ANI_PS)) ALLOCATE(HYPERFINE_RAD_ANI_PS(6,T_INFO%NIONS))
      HYPERFINE_RAD_ANI_PS(:,NI)=0._q

      ! Y00
      SCALE=1/(2._q*SQRT(PI))

      ALLOCATE(RHOAE_(SIZE(RHOAE,1)),RHOPS_(SIZE(RHOPS,1)))
#ifdef s_waves_only
      ! construct the s-wave spin density only
      RHOAE_=0._q; RHOPS_=0._q
      LM1=0
      DO I=1,PP%LMAX
         LM1=LM1+(2*PP%LPS(I)+1)
!        IF (PP%LPS(I)/=0.OR.(.NOT.PP%QATO(I,I)/=0._q)) CYCLE
         IF (PP%LPS(I)/=0) CYCLE
         LM2=0
         DO J=1,PP%LMAX 
            LM2=LM2+(2*PP%LPS(J)+1)
!           IF (PP%LPS(J)/=0.OR.(.NOT.PP%QATO(J,J)/=0._q)) CYCLE
            IF (PP%LPS(J)/=0) CYCLE
            COCC_SPIN=COCC(LM1,LM2,1)-COCC(LM1,LM2,2)
            DO K=1,PP%R%NMAX
               RHOAE_(K)=RHOAE_(K)+PP%WAE(K,I)*PP%WAE(K,J)*COCC_SPIN*SCALE!*PP%R%R(K)**2
!              RHOPS_(K)=RHOPS_(K)+PP%WPS(K,I)*PP%WPS(K,J)*COCC_SPIN*SCALE!*PP%R%R(K)**2
            ENDDO
         ENDDO
      ENDDO

      RHOPS_(:)=RHOPS(:,1,2)
#else
      ! simple copy of total s spin density
      RHOAE_(:)=RHOAE(:,1,2); RHOPS_(:)=RHOPS(:,1,2)
#endif

#ifdef corepol
      IF (.NOT.ALLOCATED(ATOM)) THEN
         ALLOCATE(ATOM(T_INFO%NTYP)); ATOM(:)%Z=-1
      ENDIF
      NT=T_INFO%ITYP(NI)
      IF (ATOM(NT)%Z<0) CALL RHFATM_DFATOM(PP,ATOM(NT))

!     DV(1:PP%R%NMAX)=(POTAE(1:PP%R%NMAX,1,1)-POTAE(1:PP%R%NMAX,1,2))/2._q
      DV=0._q
      DO K=1,PP%R%NMAX
         RHOUP=(RHOAE(K,1,1)+RHOAE(K,1,2)+PP%RHOAE(K))*SCALE/2._q/PP%R%R(K)**2
         RHODN=(RHOAE(K,1,1)-RHOAE(K,1,2)+PP%RHOAE(K))*SCALE/2._q/PP%R%R(K)**2
         CALL EXCOR_DER(RHOUP,RHODN,0,EXC,EXCD,EXCDD,EXCDDD,.FALSE.)
         DV(K)=(EXCD(1)-EXCD(2))/SCALE/2._q
      ENDDO
      
      ALLOCATE(C(ATOM(NT)%NSCORE,ATOM(NT)%NS)); C=0._q
      RHOC=0._q 
      DO I=1,ATOM(NT)%NSCORE
         IF (ATOM(NT)%L(I)/=0) CYCLE
         DO J=ATOM(NT)%NSCORE+1,ATOM(NT)%NS
!           IF (I==J.OR.ATOM(NT)%L(J)/=0) CYCLE
            IF (ATOM(NT)%L(J)/=0) CYCLE
            RHOW=0._q
            DO K=1,PP%R%NMAX
!              IF (PP%RDEP>0.AND.PP%R%R(K)-PP%RDEP>-5E-3) EXIT
               RHOW(K)=ATOM(NT)%W(K,J)*DV(K)*ATOM(NT)%W(K,I)
            ENDDO
            CALL SIMPI(PP%R,RHOW,RES)
            C(I,J)=RES/(ATOM(NT)%E(I)-ATOM(NT)%E(J))
            DO K=1,PP%R%NMAX
               RHOC(K)=RHOC(K)+4._q*C(I,J)*ATOM(NT)%W(K,J)*ATOM(NT)%W(K,I)
            ENDDO
         ENDDO 
      ENDDO
      DEALLOCATE(C)
    
      RHOC(:)=RHOC(:)*SCALE
    
      CALL HYPERFINE_FC_ZORA(PP,RHOC,HYPERFINE_RAD_ISO_C(NI))
!     RHOAE_=RHOAE_+RHOC
#endif

#ifdef corepol_my
      ! Switch to the exchange-correlation functional
      ! that was used for the generation of the PAW data set
      CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q,0.0_q)

      CALL CL_INIT_CORE_CONF(PP,MAXNL)
      ALLOCATE(WUP(PP%R%NMAX,MAXNL),WDN(PP%R%NMAX,MAXNL),N(MAXNL),LC(MAXNL),EUP(MAXNL),EDN(MAXNL))

      CALL SET_CORE_WF(PP%RHOAE,PP%POTAE,PP%R,PP%ZCORE,PP%ZVALF_ORIG, &
     &   WUP,N,LC,EUP,(POTAE(1:PP%R%NMAX,1,1)+POTAE(1:PP%R%NMAX,1,2))/2._q) 

      RHOC=0._q
      DO I=1,MAXNL
         DO K=1,PP%R%NMAX
            RHOC(K)=RHOC(K)+2._q*(WUP(K,I))*REAL(2*LC(I)+1,KIND=q)
         ENDDO
      ENDDO
      RHOC=RHOC*SCALE

      RHO(:,1,1)=RHOAE(:,1,1)+RHOC(:)
!     RHO(:,1,1)=RHOAE(:,1,1)+PP%RHOAE(:)
      RHO(:,1,2)=RHOAE(:,1,2)
      DMY=0._q

      ! start scf cycle here
      RHOP(:,1)=RHOC(:); RHOP(:,2)=0._q
      scf: DO J=1,NSCF

         CALL RAD_POT(PP%R,2,0,0,.FALSE.,RHO,DMY,DMY,POT,DUMMY,DUMMY)

         ! spin up
         CALL SET_CORE_WF(DMY,DMY,PP%R,PP%ZCORE,PP%ZVALF_ORIG, &
        &   WUP,N,LC,EUP,POT(1:PP%R%NMAX,1,1)) !,NMAX= RNMAX_CL)
         ! spin down
         CALL SET_CORE_WF(DMY,DMY,PP%R,PP%ZCORE,PP%ZVALF_ORIG, &
        &   WDN,N,LC,EDN,POT(1:PP%R%NMAX,1,2)) !,NMAX= RNMAX_CL)

#ifdef debug
         DO I=1,MAXNL
            WRITE(*,'(a,i4,x,a,i4,x,a,f14.7,x,a,f14.7)') 'n:',n(i),'l:',lc(i),'eup:',eup(i),'edn:',edn(i)
         ENDDO
#endif
         RHO=0._q
         DO I=1,MAXNL
            DO K=1,PP%R%NMAX
               RHO(K,1,1)=RHO(K,1,1)+(WUP(K,I)+WDN(K,I))*REAL(2*LC(I)+1,KIND=q)
               RHO(K,1,2)=RHO(K,1,2)+(WUP(K,I)-WDN(K,I))*REAL(2*LC(I)+1,KIND=q)
            ENDDO
         ENDDO
         RHO=RHO*SCALE

         ! check for convergence
         DRHO=0._q
         DO K=1,PP%R%NMAX
            DRHO=DRHO+ABS(RHO(K,1,2)-RHOP(K,2))
         ENDDO
         IF (DRHO<RCONV) EXIT scf

         RHO(:,1,1)=RHOP(:,1)+MIX*(RHO(:,1,1)-RHOP(:,1))
         RHO(:,1,2)=RHOP(:,2)+MIX*(RHO(:,1,2)-RHOP(:,2))

         RHOP(:,1)=RHO(:,1,1)
         RHOP(:,2)=RHO(:,1,2)

         RHO(:,1,1)=RHO(:,1,1)+RHOAE(:,1,1)
         RHO(:,1,2)=RHO(:,1,2)+RHOAE(:,1,2)

      ENDDO scf
#ifdef debug
      WRITE(*,*) 'iscf:',J
      DO I=1,MAXNL
         WRITE(*,'(a,i4,x,a,i4,x,a,f14.7,x,a,f14.7)') 'n:',n(i),'l:',lc(i),'eup:',eup(i),'edn:',edn(i)
      ENDDO
#endif

      RHOC=0._q
      DO I=1,MAXNL
         DO K=1,PP%R%NMAX
            RHOC(K)=RHOC(K)+(WUP(K,I)-WDN(K,I))*REAL(2*LC(I)+1,KIND=q)
         ENDDO
      ENDDO
      RHOC=RHOC*SCALE

      DEALLOCATE(WUP,WDN,N,LC,EUP,EDN)
      CALL CL_CLEAR_CORE_CONF

      ! Switch back to requested xc-type
      CALL POP_XC_TYPE

      CALL HYPERFINE_FC_ZORA(PP,RHOC,HYPERFINE_RAD_ISO_C(NI))
#endif

      ! isotropic (Fermi contact) contribution
      DO J=1,4
         I=(J-1)*ISTRIDE+1
         R(J)=PP%R%R(I)
         TMP(J)=RHOAE_(I)/PP%R%R(I)**2
      ENDDO
      ! extrapolate TMP to R=0
      CALL POLINT(R(1:4),TMP(1:4),4,0._q,TMP(0),DUMMY)
      HYPERFINE_RAD_ISO_AE(NI)=TMP(0)*SCALE

      CALL HYPERFINE_FC_ZORA(PP,RHOAE_,HYPERFINE_RAD_ISO_AE(NI))

      DO J=1,4
         I=(J-1)*ISTRIDE+1
         R(J)=PP%R%R(I)
         TMP(J)=RHOPS_(I)/PP%R%R(I)**2
      ENDDO
      ! extrapolate TMP to R=0
      CALL POLINT(R(1:4),TMP(1:4),4,0._q,TMP(0),DUMMY)
      HYPERFINE_RAD_ISO_PS(NI)=TMP(0)*SCALE

!     DO I=1,PP%R%NMAX
!        WRITE(1000+NI,'(2F14.7)') PP%R%R(I),RHOAE_(I)/PP%R%R(I)**2
!        WRITE(2000+NI,'(2F14.7)') PP%R%R(I),RHOPS_(I)/PP%R%R(I)**2
!     ENDDO

      DEALLOCATE(RHOAE_,RHOPS_)

      ! anisotropic (dipolar) contribution
      IF (SIZE(RHOAE,2)<5) RETURN

      SCALE=SQRT(12._q*PI/5._q)

      DO LM=5,9
         DO I=1,PP%R%NMAX
            RHOW(I)=RHOAE(I,LM,2)/PP%R%R(I)**3
         ENDDO
         CALL SIMPI(PP%R,RHOW,SUM(LM-4))
      ENDDO

      HYPERFINE_RAD_ANI_AE(1,NI)=SCALE*( SUM(5)-SUM(3)/SQRT(3._q))
      HYPERFINE_RAD_ANI_AE(2,NI)=SCALE*(-SUM(5)-SUM(3)/SQRT(3._q))
      HYPERFINE_RAD_ANI_AE(3,NI)=SCALE*( SUM(3)*SQRT(4._q/3._q))
      HYPERFINE_RAD_ANI_AE(4,NI)=SCALE*( SUM(1))
      HYPERFINE_RAD_ANI_AE(5,NI)=SCALE*( SUM(4))
      HYPERFINE_RAD_ANI_AE(6,NI)=SCALE*( SUM(2))

      DO LM=5,9
         DO I=1,PP%R%NMAX
            RHOW(I)=RHOPS(I,LM,2)/PP%R%R(I)**3
         ENDDO
         CALL SIMPI(PP%R,RHOW,SUM(LM-4))
      ENDDO

      HYPERFINE_RAD_ANI_PS(1,NI)=SCALE*( SUM(5)-SUM(3)/SQRT(3._q))
      HYPERFINE_RAD_ANI_PS(2,NI)=SCALE*(-SUM(5)-SUM(3)/SQRT(3._q))
      HYPERFINE_RAD_ANI_PS(3,NI)=SCALE*( SUM(3)*SQRT(4._q/3._q))
      HYPERFINE_RAD_ANI_PS(4,NI)=SCALE*( SUM(1))
      HYPERFINE_RAD_ANI_PS(5,NI)=SCALE*( SUM(4))
      HYPERFINE_RAD_ANI_PS(6,NI)=SCALE*( SUM(2))

      RETURN
      END SUBROUTINE HYPERFINE_RAD


!******************** SUBROUTINE HYPERFINE_FC_ZORA *********************
!
!***********************************************************************
      SUBROUTINE HYPERFINE_FC_ZORA(PP,RHOAE,FC)
      USE pseudo
      USE constant
      USE radial
      TYPE(potcar), POINTER :: PP
      REAL(q) RHOAE(:),FC
      ! local variables
      INTEGER I,J,K
      REAL(q) SCALE,RHOW(PP%R%NMAX)
      REAL(q) TMP(0:4),R(4),DUMMY
      INTEGER, PARAMETER :: ISTRIDE=1

      REAL(q) R_Thomson,Z
      ! classical electron radius
      REAL(q), PARAMETER :: Re=AUTOA/CLIGHT/CLIGHT

      INTEGER, PARAMETER :: NP=10000
      REAL(q) X(0:NP),Y(0:NP),RES,H

      INTEGER IPVT(4),INFO
      REAL(q) LAMBDA,A(4,4),B(4),R2,QQ,JSUM,IFAC,RESL
      INTEGER, PARAMETER :: JMAX=200

      ! Y00
      SCALE=1/(2._q*SQRT(PI))

#ifdef zora_simple
      ! isotropic (Fermi contact) contribution
      DO J=1,4
         I=(J-1)*ISTRIDE+1
         R(J)=PP%R%R(I)
         TMP(J)=RHOAE(I)/PP%R%R(I)**2
      ENDDO
      ! extrapolate TMP to R=0
      CALL POLINT(R(1:4),TMP(1:4),4,0._q,TMP(0),DUMMY)

      Z=(PP%ZVALF+PP%ZCORE)
      R_Thomson=Z*Re
 
      RHOW=0._q
      DO I=1,PP%R%NMAX
         IF (PP%RDEP>0.AND.PP%R%R(I)-PP%RDEP>-5E-3) EXIT
         RHOW(I)=SCALE*RHOAE(I)*(R_Thomson/2._q/(PP%R%R(I)+R_Thomson/2._q)**2)/PP%R%R(I)**2
!        RHOW(I)=(R_Thomson/2._q/(PP%R%R(I)+R_Thomson/2._q)**2)
      ENDDO
      CALL SIMPI(PP%R,RHOW,RES)

      X(0)=0
      DO I=1,NP
         X(I)=I*PP%R%R(1)/NP
      ENDDO

      H=(X(NP)-X(0))/NP/3._q
      DO I=0,NP
         CALL POLINT(R(1:4),TMP(1:4),4,X(I),TMP(0),DUMMY) 
         Y(I)=TMP(0)*SCALE*(R_Thomson/2._q/(X(I)+R_Thomson/2._q)**2)*H
!        Y(I)=(R_Thomson/2._q/(X(I)+R_Thomson/2._q)**2)*H
      ENDDO

      ! Simpson integration
      RES=RES+Y(0)
      DO I=1,NP,2
         RES=RES+4*Y(I) 
      ENDDO
      DO I=2,NP-1,2
         RES=RES+2*Y(I)
      ENDDO
      RES=RES+Y(NP) 
      
!     WRITE(*,'(I4,X,A,F14.7,X,A,F14.7)') NI,'res:',RES,'ext:',HYPERFINE_RAD_ISO_AE(NI)
!     DO I=1,NP
!        WRITE(3000+NI,'(2F20.12)') X(I),Y(I)
!     ENDDO
!     DO I=1,PP%R%NMAX
!        WRITE(3000+NI,'(2F20.12)') PP%R%R(I),(R_Thomson/2._q/(PP%R%R(I)+R_Thomson/2._q)**2)
!     ENDDO

      FC=RES
#endif

#ifdef zora_bloechl
      Z=(PP%ZVALF+PP%ZCORE)
      R_Thomson=Z*Re
      LAMBDA=SQRT(1-Z*Z/CLIGHT/CLIGHT)
      
      RHOW=0._q
      DO I=1,PP%R%NMAX
         IF (PP%RDEP>0.AND.PP%R%R(I)-PP%RDEP>-5E-3) EXIT
         RHOW(I)=SCALE*RHOAE(I)*(R_Thomson/2._q/(PP%R%R(I)+R_Thomson/2._q)**2)/PP%R%R(I)**2
      ENDDO
      CALL SIMPI(PP%R,RHOW,RES)

      DO J=1,4
         DO I=1,4
            A(I,J)=PP%R%R((J-1)*ISTRIDE+1)**(2._q*LAMBDA-3._q+I)
         ENDDO
      ENDDO
      A=A/PP%R%R(1)**(2._q*LAMBDA-2._q)
      DO I=1,4
         J=(I-1)*ISTRIDE+1
         B(I)=SCALE*RHOAE(J)/PP%R%R(J)**2
      ENDDO

      CALL DGETRF(4,4,A,4,IPVT,INFO)
      IF (INFO/=0) THEN
         WRITE(*,*) 'HYPERFINE_RAD: ERROR: DGETRF failed:',INFO
         STOP
      ENDIF

      CALL DGETRS('T',4,1,A,4,IPVT,B,4,INFO)
      IF (INFO/=0) THEN
         WRITE(*,*) 'HYPERFINE_RAD: ERROR: DGETRS failed:',INFO
         STOP
      ENDIF

      DO I=1,NP
         X(I)=I*PP%R%R(1)/NP
      ENDDO
      Y=0
      DO I=1,NP
         DO J=1,4
            Y(I)=Y(I)+B(J)*X(I)**(2._q*LAMBDA-3._q+J)
         ENDDO
      ENDDO
      Y=Y/PP%R%R(1)**(2._q*LAMBDA-2._q)
!     DO I=1,NP
!        WRITE(4000,'(2F20.12)') X(I),Y(I)/SCALE
!     ENDDO

      B=B*(R_Thomson/2._q/PP%R%R(1))**(2._q*LAMBDA-2._q)

      QQ=2._q*(LAMBDA-1._q)+1._q
      R2=2._q*PP%R%R(1)/R_Thomson
      DO I=0,3
         IFAC=B(I+1)*R2**(QQ+I)/(QQ+I)/(1._q+R2)**2._q*(R_Thomson/2._q)**I
         DO J=0,JMAX
            JSUM=(J+1._q) 
            IF (J>=1) THEN
               DO K=1,J
                  JSUM=JSUM*(1._q/(1._q+(QQ+I)/K)*R2/(1._q+R2))
               ENDDO
            ENDIF
            RES=RES+IFAC*JSUM
            IF (J>0) THEN
               IF (ABS(RES-RESL)<1.E-6) EXIT
            ENDIF
            RESL=RES
         ENDDO
         IF (J>JMAX) THEN
            WRITE(*,*) 'HYPERFINE_RAD: ERROR: taylor series did not converge:',I,JMAX,J
            STOP
         ENDIF
      ENDDO
!     WRITE(*,'(I4,X,A,F14.7,X,A,F14.7)') NI,'res bloechl:',RES

      FC=RES
#endif
      RETURN
      END SUBROUTINE HYPERFINE_FC_ZORA


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE POLINT(XA,YA,N,X,Y,DY)

      USE prec

      INTEGER N,NMAX
      INTEGER I,M,NS
      REAL(q) X,Y,DY
      REAL(q) XA(N),YA(N)
      REAL(q) DEN,DIF,DIFT,HO,HP,W
      PARAMETER(NMAX=10)
      REAL(q) C(NMAX),D(NMAX)

      NS=1
      DIF=ABS(X-XA(1))
      DO I=1,N
         DIFT=ABS(X-XA(I))
         IF (DIFT.LT.DIF) THEN
            NS=I
            DIF=DIFT
         ENDIF
         C(I)=YA(I)
         D(I)=YA(I)
      ENDDO

      Y=YA(NS)
      NS=NS-1
      DO M=1,N-1
         DO I=1,N-M
            HO=XA(I)-X
            HP=XA(I+M)-X
            W=C(I+1)-D(I)
            DEN=HO-HP
! MM. When input XA's are all inequivalent (to within rounoff)
!           IF(DEN.EQ.0._q)PAUSE 'FAILURE IN POLINT'
! MM.
            DEN=W/DEN
            D(I)=HP*DEN
            C(I)=HO*DEN
         ENDDO
         IF (2*NS.LT.N-M)THEN
            DY=C(NS+1)
         ELSE
            DY=D(NS)
            NS=NS-1
         ENDIF
         Y=Y+DY
      ENDDO

      RETURN
      END SUBROUTINE POLINT

      END MODULE hyperfine
