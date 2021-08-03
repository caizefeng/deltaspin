#include "symbol.inc"
!***********************************************************************
!***********************************************************************

      MODULE egrad
      USE prec
      
      PRIVATE :: POLINT
      
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: EFG_PW(:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: EFG_RAD(:,:)

      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: R_STORE(:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: VPS_STORE(:,:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: VAE_STORE(:,:,:)
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: QUAD_EFG(:)

      LOGICAL, PRIVATE, SAVE :: LCALCEFG
      
      CONTAINS


!***********************************************************************
!***********************************************************************

      SUBROUTINE EGRAD_READER(IU5, IU6, IU0, NTYP)
      USE prec
      USE base
      USE vaspxml
      IMPLICIT NONE
      INTEGER IU5,IU6,IU0,NTYP
      ! local variables
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC
      CHARACTER (40)   SZNAM

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      LCALCEFG=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LEFG','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LCALCEFG,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
            WRITE(IU0,*)'Error reading item ''LEFG'' from file INCAR.'
         LCALCEFG=.FALSE.
      ENDIF

      CALL XML_INCAR('LEFG','L',IDUM,RDUM,CDUM,LCALCEFG,CHARAC,N)

      ALLOCATE(QUAD_EFG(NTYP))
 
      QUAD_EFG=1.0_q
      CALL RDATAB(LOPEN,INCAR,IU5,'QUAD_EFG','=','#',';','F', &
     &            IDUM,QUAD_EFG,CDUM,LDUM,CHARAC,N,NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
          WRITE(IU0,*)'Error reading item ''QUAD_EFG'' from file INCAR.'
         QUAD_EFG=1.0_q
      ENDIF

      CALL XML_INCAR_V('QUAD_EFG','F',IDUM,QUAD_EFG,CDUM,LDUM,CHARAC,N)


      CLOSE(IU5)      
      
      RETURN
      END SUBROUTINE EGRAD_READER


!***********************************************************************
!***********************************************************************

      SUBROUTINE EGRAD_WRITE_EFG(T_INFO,WDES,IO)
      USE prec
      USE constant
      USE base
      USE paw
      USE poscar
      USE wave_high
      USE vaspxml
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(in_struct) IO
      TYPE(type_info) T_INFO
      ! local variables
      INTEGER I,J
      INTEGER NI ,NT
      REAL(q), ALLOCATABLE :: EFG(:,:), EIGV(:,:,:)
      REAL(q) A(3,3),EIG(3)
      INTEGER IFAIL,IDONE
      INTEGER, PARAMETER :: LWORK=6
      REAL(q) WORK(3*LWORK)
      REAL(q) TMP, ETA, CQ_UNIT, PLANCK, TMPVEC(3)

      LOGICAL LSKIP

      ! quick return if electric field gradienst are not required
      IF (.NOT.EGRAD_CALC_EFG()) RETURN

      IF (.NOT.ALLOCATED(EFG_RAD)) THEN
         ALLOCATE(EFG_RAD(6,T_INFO%NIONS))
         ALLOCATE(R_STORE(4,T_INFO%NIONS))
         ALLOCATE(VPS_STORE(0:4,5,T_INFO%NIONS))
         ALLOCATE(VAE_STORE(0:4,5,T_INFO%NIONS))
      ENDIF
      IF (.NOT.ALLOCATED(EFG_PW)) RETURN

      ! zero all elements in EFG_RAD that have not been
      ! calculated on this node
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
            EFG_RAD(:,NI)=0
            R_STORE(:,NI)=0
            VPS_STORE(:,:,NI)=0
            VAE_STORE(:,:,NI)=0
         ENDIF
      ENDDO
      
      ! communicate EFG_RAD
      CALLMPI( M_sum_d(WDES%COMM,EFG_RAD,6*T_INFO%NIONS))
      CALLMPI( M_sum_d(WDES%COMM,R_STORE,4*T_INFO%NIONS))
      CALLMPI( M_sum_d(WDES%COMM,VPS_STORE,5*5*T_INFO%NIONS))
      CALLMPI( M_sum_d(WDES%COMM,VAE_STORE,5*5*T_INFO%NIONS))

      
      ! quick return if node is not supposed to do io
      IF(IO%IU6<0) RETURN

      DO NI=1,T_INFO%NIONS
         WRITE(*,'(A,I4)') 'ion',NI
         WRITE(*,'(6F14.7)') 0._q,(VPS_STORE(0,J,NI),J=1,5)
         DO I=1,4
            WRITE(*,'(6F14.7)') R_STORE(I,NI),(VPS_STORE(I,J,NI),J=1,5)
         ENDDO
         WRITE(*,*)
         WRITE(*,'(6F14.7)') 0._q,(VAE_STORE(0,J,NI),J=1,5)
         DO I=1,4
            WRITE(*,'(6F14.7)') R_STORE(I,NI),(VAE_STORE(I,J,NI),J=1,5)
         ENDDO
         WRITE(*,*)
      ENDDO

      ALLOCATE(EFG(6,T_INFO%NIONS),EIGV(3,3,T_INFO%NIONS))
      EFG=EFG_PW+EFG_RAD
!FMV OPPOSITE SIGN CONVENTION
      EFG = -EFG
!FMV OPPOSITE SIGN CONVENTION
      WRITE(IO%IU6,'(/A)') ' Electric field gradients (V/A^2)'
      WRITE(IO%IU6,'(A)')   '---------------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')   ' ion       V_xx      V_yy      V_zz      V_xy      V_xz      V_yz'
      WRITE(IO%IU6,'(A)')   '---------------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
         WRITE(IO%IU6,'(I4,2X,6F10.3)') NI,EFG(:,NI)
      ENDDO
      WRITE(IO%IU6,'(A/)')   '---------------------------------------------------------------------'


      ! Get eigenvalues and eigenvectors
      DO NI=1,T_INFO%NIONS
         A=0
         A(1,1)=EFG(1,NI)
         A(1,2)=EFG(4,NI)
         A(1,3)=EFG(5,NI)
         A(2,2)=EFG(2,NI)
         A(2,3)=EFG(6,NI)
         A(3,3)=EFG(3,NI)
         CALL DSYEV('V','U',3,A,3,EIG,WORK,3*LWORK,IFAIL)
         EIGV(:,:,NI)=A
         EFG(1,NI)=EIG(1)
         EFG(2,NI)=EIG(2)
         EFG(3,NI)=EIG(3)
      ENDDO

! test
!     DO NI=1,T_INFO%NIONS
!        WRITE(IO%IU6,*) NI
!        WRITE(IO%IU6,*) EFG_PW(:,NI)
!        WRITE(IO%IU6,*) EFG_RAD(:,NI)
!        WRITE(IO%IU6,*) EFG(1:3,NI)
!     ENDDO
! test
        
      ! (convention: |V_zz| > |V_xx| > |V_yy|)
      DO NI=1,T_INFO%NIONS
         DO I=1,2
            IF (ABS(EFG(1,NI))>ABS(EFG(2,NI))) THEN
               TMP=EFG(2,NI)
               EFG(2,NI)=EFG(1,NI)
               EFG(1,NI)=TMP

               TMPVEC=EIGV(:,2,NI)
               EIGV(:,2,NI)=EIGV(:,1,NI)
               EIGV(:,1,NI)=TMPVEC
            ENDIF
            IF (ABS(EFG(2,NI))>ABS(EFG(3,NI))) THEN
               TMP=EFG(3,NI)
               EFG(3,NI)=EFG(2,NI)
               EFG(2,NI)=TMP

               TMPVEC=EIGV(:,3,NI)
               EIGV(:,3,NI)=EIGV(:,2,NI)
               EIGV(:,2,NI)=TMPVEC
            ENDIF            
         ENDDO
      ENDDO

      WRITE(IO%IU6,'(/A)')  ' Electric field gradients after diagonalization (V/A^2)'
      WRITE(IO%IU6,'(A)')   ' (convention: |V_zz| > |V_xx| > |V_yy|)'
      WRITE(IO%IU6,'(A)')   '----------------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')   ' ion       V_xx      V_yy      V_zz     asymmetry (V_yy - V_xx)/ V_zz '
      WRITE(IO%IU6,'(A)')   '----------------------------------------------------------------------'

      DO NI=1,T_INFO%NIONS
         ETA=(EFG(1,NI)-EFG(2,NI))/EFG(3,NI)
         WRITE(IO%IU6,'(I4,2X,3F10.3,2X,F10.3)') NI,EFG(2,NI),EFG(1,NI),EFG(3,NI),(EFG(1,NI)-EFG(2,NI))/EFG(3,NI)
      ENDDO
      WRITE(IO%IU6,'(A/)')  '----------------------------------------------------------------------'
      WRITE(IO%IU6,'(/A)')  ' Eigenvectors'
      WRITE(IO%IU6,'(A/)')  '----------------------------------------------------------------------'
      DO NI=1,T_INFO%NIONS 
         WRITE(IO%IU6,'(A,I4)') ' ion',NI
         WRITE(IO%IU6,'(X,A2,2X,3F10.3)') 'xx' , EIGV(:,2,NI)
         WRITE(IO%IU6,'(X,A2,2X,3F10.3)') 'yy' , EIGV(:,1,NI)
         WRITE(IO%IU6,'(X,A2,2X,3F10.3)') 'zz' , EIGV(:,3,NI)
         WRITE(IO%IU6,'(A/)')  ''
      ENDDO
      
      WRITE(IO%IU6,'(A/)')  '----------------------------------------------------------------------'

     
!this seems to be incorrect: CQ_UNIT=2.0d0*RYTOEV/(AUTOA*AUTOA)*(EVTOJ*1e15_q)/6.62620d0
!     e V_zz[V/Ang2] V/Ang2 Q[millibarn] 1e-31 m2 /(h[Js] Js) = 
!     eV 1e20 1e-31 /(h[Js] Js)                   * V_zz[V/Ang2] Q[millibarn] =
!     EVTOJ J 1e20 1e-31 /(h[Js] Js)              * V_zz[V/Ang2] Q[millibarn] =
!     EVTOJ   1e20 1e-31 Hz /(6.6262e-34)         * V_zz[V/Ang2] Q[millibarn] =
!     EVTOJ   1e20 1e-31 1e-6 MHz /(6.6262e-34)   * V_zz[V/Ang2] Q[millibarn] =
!     EVTOJ   1e20 1e3 Hz / (6.6262)              * V_zz[V/Ang2] Q[millibarn] =
!     EVTOJ   1e20 1e-3 MHz / (6.6262)            * V_zz[V/Ang2] Q[millibarn]
      PLANCK = 6.62606957e-34  ! Js, 2010 CODATA
      CQ_UNIT = EVTOJ/1e17/PLANCK

      WRITE(IO%IU6,'(A)')  '           NMR quadrupolar parameters'
      WRITE(IO%IU6,'(A)')  ' ' 
      WRITE(IO%IU6,'(A)')  ' Cq : quadrupolar parameter    Cq=e*Q*V_zz/h'
      WRITE(IO%IU6,'(A)')  ' eta: asymmetry parameters     (V_yy - V_xx)/ V_zz'
      WRITE(IO%IU6,'(A)')  ' Q  : nuclear electric quadrupole moment in mb (millibarn)'
      WRITE(IO%IU6,'(A)')  '----------------------------------------------------------------------'
      WRITE(IO%IU6,'(A)')  ' ion       Cq(MHz)       eta       Q (mb) '
      WRITE(IO%IU6,'(A)')  '----------------------------------------------------------------------'
      DO NI=1,T_INFO%NIONS
         NT=T_INFO%ITYP(NI) 
         ETA=(EFG(1,NI)-EFG(2,NI))/EFG(3,NI)
         WRITE(IO%IU6,'(I4,2X,F10.3,2X,F10.3,2X,F10.3)') NI,EFG(3,NI)*CQ_UNIT*QUAD_EFG(NT), ETA, QUAD_EFG(NT)
      ENDDO
      WRITE(IO%IU6,'(A/)') '----------------------------------------------------------------------'

      DEALLOCATE(EFG,EIGV)

      RETURN
      END SUBROUTINE EGRAD_WRITE_EFG
      

!***********************************************************************
!***********************************************************************

      FUNCTION EGRAD_CALC_EFG()
      LOGICAL EGRAD_CALC_EFG
      EGRAD_CALC_EFG=LCALCEFG
      END FUNCTION EGRAD_CALC_EFG


!***********************************************************************
!***********************************************************************
      
      SUBROUTINE EGRAD_EFG_PW_HAR_ONLY( &
     &   T_INFO,LATT_CUR,WDES,GRIDC,P,CSTRF,CHTOT)
      USE prec
      USE mgrid
      USE poscar
      USE pseudo
      USE lattice
      USE constant
      USE wave_high
      IMPLICIT NONE
      TYPE(wavedes) WDES
      TYPE(grid_3d) GRIDC
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE (potcar) P(T_INFO%NTYP)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      COMPLEX(q) CHTOT(GRIDC%RC%NP)
      ! local variables
      INTEGER NI,NT
      INTEGER I,NC,N1,N2,N3
      REAL(q) GX,GY,GZ,GSQU
      REAL(q) SCALE,FACTM
      COMPLEX(q) CGR
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
      REAL(q) E,ENCUT,DFUN
      REAL(q), PARAMETER :: CUTOFF = 0.2

      REAL(q), PARAMETER :: A=64

      ! quick return if electric field gradients are not required
      IF (.NOT.EGRAD_CALC_EFG()) RETURN

      IF (.NOT.ALLOCATED(EFG_PW)) THEN
         ALLOCATE(EFG_PW(6,T_INFO%NIONS))
      ENDIF

      ! allocate workspace
      ALLOCATE(CWORK(SIZE(CHTOT)))

      EFG_PW=0

      ENCUT=WDES%ENMAX

      SCALE=EDEPS/LATT_CUR%OMEGA

      CWORK=0

      ions: DO NI=1,T_INFO%NIONS

      I=0
      col1: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row1: DO N1=1,GRIDC%RC%NROW

         I=I+1
         SETFACT1
         SETFACT

         ! x,y,z components and norm of G vector
         GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
         GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
         GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

         GSQU=GX**2+GY**2+GZ**2

         ! damping of high Fourier components between (1-CUTOFF) ENCUT and ENCUT
         DFUN=1
!        E=HSQDTM*GSQU
!        IF (E<ENCUT) THEN   
!           DFUN=(1+COS((E-ENCUT*(1-CUTOFF))/(ENCUT*CUTOFF)*PI))/2
!           DFUN=1
!        ELSE
!           DFUN=0
!        ENDIF

         ! phase factor         
         CGR=CITPI*(GRIDC%LPCTX(N1)*T_INFO%POSION(1,NI)+ &
        &            GRIDC%LPCTY(N2)*T_INFO%POSION(2,NI)+ &
        &             GRIDC%LPCTZ(N3)*T_INFO%POSION(3,NI))

!=======================================================================
! since the G=0 coulomb contributions to the hartree, ewald and
! electron-ion energies are individually divergent but together sum to
! zero, set the hartree potential at G=0 to zero.
!=======================================================================
         IF ((GRIDC%LPCTX(N1)==0).AND.(GRIDC%LPCTY(N2)==0).AND.(GRIDC%LPCTZ(N3)==0)) THEN
            CWORK(I)=(0.0_q,0.0_q)
         ELSE
            CWORK(I)=CHTOT(I)
            DO NT=1,T_INFO%NTYP
               CWORK(I)=CWORK(I)-P(NT)%ZVALF*CSTRF(I,NT)*EXP(-GSQU/A)
            ENDDO
            CWORK(I)=CWORK(I)/GSQU*SCALE
            CWORK(I)=CWORK(I)*EXP(CGR)*DFUN
         ENDIF

      ENDDO row1
      ENDDO col1
      CALL SETUNB(CWORK,GRIDC)

      I=0
      col2: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row2: DO N1=1,GRIDC%RC%NROW

         I=I+1
         SETFACT1
         SETFACT

         ! x,y,z components and norm of G vector
         GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*2*PI
         GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*2*PI
         GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*2*PI

         GSQU=GX**2+GY**2+GZ**2

         EFG_PW(1,NI)=EFG_PW(1,NI)-MULFACT CWORK(I)*(GX*GX-GSQU/3)
         EFG_PW(2,NI)=EFG_PW(2,NI)-MULFACT CWORK(I)*(GY*GY-GSQU/3)
         EFG_PW(3,NI)=EFG_PW(3,NI)-MULFACT CWORK(I)*(GZ*GZ-GSQU/3)
         EFG_PW(4,NI)=EFG_PW(4,NI)-MULFACT CWORK(I)*(GX*GY)
         EFG_PW(5,NI)=EFG_PW(5,NI)-MULFACT CWORK(I)*(GX*GZ)
         EFG_PW(6,NI)=EFG_PW(6,NI)-MULFACT CWORK(I)*(GY*GZ)

      ENDDO row2
      ENDDO col2

      ENDDO ions

      ! communicate
      CALLMPI( M_sum_d(GRIDC%COMM,EFG_PW,6*T_INFO%NIONS))

      ! deallocate workspace
      DEALLOCATE(CWORK)

      RETURN
      END SUBROUTINE EGRAD_EFG_PW_HAR_ONLY


!***********************************************************************
!***********************************************************************
      
      SUBROUTINE EGRAD_EFG_RAD_HAR_ONLY(T_INFO,NI_GLOBAL,PP,RHOPS,RHOAE)
      USE prec
      USE poscar
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE(potcar) PP
      TYPE(type_info) T_INFO
      INTEGER NI_GLOBAL
      REAL(q) RHOPS(:,:,:),RHOAE(:,:,:)
      ! local variables
      INTEGER K,M,IR
      REAL(q) SCALE,DUMMY
      REAL(q), ALLOCATABLE :: POTPS(:,:),POTAE(:,:) 
      REAL(q) R(4),VPS(0:4,5),VAE(0:4,5)
      INTEGER, PARAMETER :: ISTRIDE=10

      ! quick return if electric field gradienst are not required
      IF (.NOT.EGRAD_CALC_EFG()) RETURN

      IF (.NOT.ALLOCATED(EFG_RAD)) THEN
         ALLOCATE(EFG_RAD(6,T_INFO%NIONS))
         ALLOCATE(R_STORE(4,T_INFO%NIONS))
         ALLOCATE(VPS_STORE(0:4,5,T_INFO%NIONS))
         ALLOCATE(VAE_STORE(0:4,5,T_INFO%NIONS))
      ENDIF
      
      EFG_RAD(:,NI_GLOBAL)=0
      R_STORE(:,NI_GLOBAL)=0
      VPS_STORE(:,:,NI_GLOBAL)=0
      VAE_STORE(:,:,NI_GLOBAL)=0
      
      ! quick return if the charge density does not contain
      ! components of angular momentum l=2
      IF (SIZE(RHOPS,2)<5) RETURN

      ! compute the AE and PS Hartree potential for l=2
      ALLOCATE(POTAE(SIZE(RHOAE,1),5),POTPS(SIZE(RHOPS,1),5))
      DO M=1,5
         CALL RAD_POT_HAR(2,PP%R,POTAE(:,M),RHOAE(:,M+4,1),DUMMY)
         CALL RAD_POT_HAR(2,PP%R,POTPS(:,M),RHOPS(:,M+4,1),DUMMY)
      ENDDO

      ! copy the first four grid points of POTPS and POTAE
      ! into the temporary arrays VPS and VAE
      VPS=0
      VAE=0
      DO M=1,5
         DO K=1,4
            IR=(K-1)*ISTRIDE+1
            R(K)=PP%R%R(IR)
            VPS(K,M)=POTPS(IR,M)/PP%R%R(IR)**2
            VAE(K,M)=POTAE(IR,M)/PP%R%R(IR)**2
         ENDDO
      ENDDO      
      
      ! extrapolate the VPS and VAE to R=0
      DO M=1,5
         CALL POLINT(R(1:4),VPS(1:4,M),4,0._q,VPS(0,M),DUMMY)
         CALL POLINT(R(1:4),VAE(1:4,M),4,0._q,VAE(0,M),DUMMY)
      ENDDO
      
      R_STORE(1:4,NI_GLOBAL)=R(1:4)
      VPS_STORE(0:4,1:5,NI_GLOBAL)=VPS(0:4,1:5)
      VAE_STORE(0:4,1:5,NI_GLOBAL)=VAE(0:4,1:5)
      
#ifdef debug
      WRITE(*,*)  'ion',NI_GLOBAL
      WRITE(*,'(10F20.10)') 0._q, (VPS(0,M), M=1,5)
      DO K=1,4
         WRITE(*,'(10F20.10)') R(K), (VPS(K,M), M=1,5) 
      ENDDO
      WRITE(*,*)
      WRITE(*,'(10F20.10)') 0._q, (VAE(0,M), M=1,5)
      DO K=1,4
         WRITE(*,'(10F20.10)') R(K), (VAE(K,M), M=1,5) 
      ENDDO
!     DO K=1,PP%R%NMAX
!        WRITE(71,'(10F20.10)') PP%R%R(K), (POTPS(K,M), M=1,5)
!     ENDDO
!     DO K=1,PP%R%NMAX
!        WRITE(72,'(10F20.10)') PP%R%R(K), (POTAE(K,M), M=1,5)
!     ENDDO
!     DO K=1,100
!        DO M=1,5
!           CALL POLINT(R(1:4),VAE(1:4,M),4,(K-1)*PP%R%R(45)/100,VAE(0,M),DUMMY)
!        ENDDO
!        WRITE(73,'(10F20.10)') (K-1)*PP%R%R(45)/100, (VAE(0,M), M=1,5)
!     ENDDO
!     STOP
#endif

      SCALE=1/SQRT(16._q*PI)
      EFG_RAD(1,NI_GLOBAL)=( (VAE(0,5)-VPS(0,5))*SQRT(15._q)-(VAE(0,3)-VPS(0,3))*SQRT(5._q))*SCALE*2
      EFG_RAD(2,NI_GLOBAL)=(-(VAE(0,5)-VPS(0,5))*SQRT(15._q)-(VAE(0,3)-VPS(0,3))*SQRT(5._q))*SCALE*2
      EFG_RAD(3,NI_GLOBAL)=  (VAE(0,3)-VPS(0,3))*SQRT( 5._q)*SCALE*4
      EFG_RAD(4,NI_GLOBAL)=  (VAE(0,1)-VPS(0,1))*SQRT(60._q)*SCALE
      EFG_RAD(5,NI_GLOBAL)=  (VAE(0,4)-VPS(0,4))*SQRT(60._q)*SCALE
      EFG_RAD(6,NI_GLOBAL)=  (VAE(0,2)-VPS(0,2))*SQRT(60._q)*SCALE
      
      DEALLOCATE(POTPS,POTAE)
      
      RETURN
      END SUBROUTINE EGRAD_EFG_RAD_HAR_ONLY

      
!***********************************************************************
!***********************************************************************
      
      SUBROUTINE EGRAD_EFG_RAD(T_INFO,NI_GLOBAL,PP,POTPS,POTAE)
      USE prec
      USE poscar
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar) PP
      TYPE(type_info) T_INFO
      INTEGER NI_GLOBAL
      REAL(q) POTPS(:,:,:),POTAE(:,:,:)
      ! local variables
      INTEGER K,LM
      REAL(q) DUMMY
      INTEGER IR
      REAL(q) R(4),VPS(0:4,5),VAE(0:4,5)
      INTEGER, PARAMETER :: ISTRIDE=10

      ! quick return if electric field gradienst are not required
      IF (.NOT.EGRAD_CALC_EFG()) RETURN

      IF (.NOT.ALLOCATED(EFG_RAD)) THEN
         ALLOCATE(EFG_RAD(6,T_INFO%NIONS))
      ENDIF
      
!     EFG_RAD(:,NI_GLOBAL)=0
      
      ! quick return if the potential does not contain
      ! components of angular momentum l=2
      IF (SIZE(POTPS,2)<5) RETURN

      ! copy the first four grid points of POTPS and POTAE
      ! into the temporary arrays VPS and VAE
      VPS=0
      VAE=0
      DO LM=1,5
         DO K=1,4
            IR=(K-1)*ISTRIDE+1
            R(K)=PP%R%R(IR)
            IF (SIZE(POTPS,3)==2) THEN
               VPS(K,LM)=(POTPS(IR,LM+4,1)+POTPS(IR,LM+4,2))/2/PP%R%R(IR)**2
               VAE(K,LM)=(POTAE(IR,LM+4,1)+POTAE(IR,LM+4,2))/2/PP%R%R(IR)**2
            ELSE
               VPS(K,LM)=POTPS(IR,LM+4,1)/PP%R%R(IR)**2
               VAE(K,LM)=POTAE(IR,LM+4,1)/PP%R%R(IR)**2
            ENDIF
         ENDDO
      ENDDO      
      
      ! extrapolate the VPS and VAE to R=0
      DO LM=1,5
         CALL POLINT(R(1:4),VPS(1:4,LM),4,0._q,VPS(0,LM),DUMMY)
         CALL POLINT(R(1:4),VAE(1:4,LM),4,0._q,VAE(0,LM),DUMMY)
      ENDDO
      
! test
      WRITE(*,*)  'ion',NI_GLOBAL
      WRITE(*,'(10F20.10)') 0._q, (VPS(0,LM), LM=1,5)
      DO K=1,4
         WRITE(*,'(10F20.10)') R(K), (VPS(K,LM), LM=1,5) 
      ENDDO
      WRITE(*,*)
      WRITE(*,'(10F20.10)') 0._q, (VAE(0,LM), LM=1,5)
      DO K=1,4
         WRITE(*,'(10F20.10)') R(K), (VAE(K,LM), LM=1,5) 
      ENDDO
!     DO K=1,PP%R%NMAX
!        WRITE(71,'(10F20.10)') PP%R%R(K), (POTPS(K,LM,1), LM=5,9)
!     ENDDO
!     DO K=1,PP%R%NMAX
!        WRITE(72,'(10F20.10)') PP%R%R(K), (POTAE(K,LM,1), LM=5,9)
!     ENDDO
!     DO K=1,100
!        DO LM=1,5
!           CALL POLINT(R(1:4),VAE(1:4,LM),4,(K-1)*PP%R%R(45)/100,VAE(0,LM),DUMMY)
!        ENDDO
!        WRITE(73,'(10F20.10)') (K-1)*PP%R%R(45)/100, (VAE(0,LM), LM=1,5)
!     ENDDO
!     STOP
! test
            
      RETURN
      END SUBROUTINE EGRAD_EFG_RAD
      
      
!***********************************************************************
!***********************************************************************

      SUBROUTINE POLINT(XA,YA,N,X,Y,DY)

      USE prec

      IMPLICIT NONE

      INTEGER                    N,NMAX
      INTEGER                    I,M,NS

      REAL(q)                    X,Y,DY
      REAL(q)                    XA(N),YA(N)
      REAL(q)                    DEN,DIF,DIFT,HO,HP,W

      PARAMETER(NMAX=10)

      REAL(q)                    C(NMAX),D(NMAX)


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

      
      END MODULE egrad
