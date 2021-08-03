#include "symbol.inc"

      MODULE spline
      USE prec

      TYPE splfnc
         INTEGER :: NP
         INTEGER :: NGEN
         REAL(q) :: RMAX
         REAL(q), POINTER :: SPL(:,:)
      END TYPE splfnc

      CONTAINS


      SUBROUTINE COPYSPL(SPL1,SPL2)
      IMPLICIT NONE
      TYPE(splfnc) SPL1,SPL2
      SPL2%NP=SPL1%NP
      SPL2%NGEN=SPL1%NGEN
      SPL2%RMAX=SPL1%RMAX
      SPL2%SPL=SPL1%SPL
      RETURN
      END SUBROUTINE COPYSPL


      SUBROUTINE LINPOL(R,SPL,FR)
      USE prec
      IMPLICIT NONE
      TYPE (splfnc) SPL
      REAL(q) R,FR
      ! local variables
      INTEGER I
      REAL(q) C1,C2

      FR=0

      IF (R>=SPL%RMAX) RETURN

      DO I=SPL%NP,1,-1
         IF (SPL%SPL(I,1)<=R) EXIT
      ENDDO

      ! fr=f(1) for r<R(1)
!     IF (I==0) THEN
!        FR=SPL%SPL(1,2)/SPL%SPL(1,1)/SPL%SPL(1,1)
!     ENDIF

      ! backward differences for r<R(1)
      IF (I==0) THEN
         C1=SPL%SPL(1,2)/SPL%SPL(1,1)/SPL%SPL(1,1)
         C2=SPL%SPL(2,2)/SPL%SPL(2,1)/SPL%SPL(2,1)
         FR=C1-(C2-C1)/(SPL%SPL(2,1)-SPL%SPL(1,1))*(SPL%SPL(1,1)-R)
         RETURN
      ENDIF

      ! forward differences for r>=R(N)
      IF (I==SPL%NP) THEN
         C1=SPL%SPL(I-1,2)/SPL%SPL(I-1,1)/SPL%SPL(I-1,1)
         C2=SPL%SPL(I,2)/SPL%SPL(I,1)/SPL%SPL(I,1)
         FR=C2+(C2-C1)/(SPL%SPL(I,1)-SPL%SPL(I-1,1))*(R-SPL%SPL(I,1))
         RETURN
      ENDIF

      ! interpolate between f(i) and f(i+1) for R(i)<=r<R(i+1)
      C1=SPL%SPL(I,2)/SPL%SPL(I,1)/SPL%SPL(I,1)
      C2=SPL%SPL(I+1,2)/SPL%SPL(I+1,1)/SPL%SPL(I+1,1)
      FR=C1+(C2-C1)/(SPL%SPL(I+1,1)-SPL%SPL(I,1))*(R-SPL%SPL(I,1))
      
      RETURN
      END SUBROUTINE LINPOL

      END MODULE spline


      MODULE chgfit
      USE prec
      IMPLICIT NONE

      INTEGER NGAUS
      REAL(q), ALLOCATABLE :: ZCT(:),RGAUS(:)

      REAL(q) TAU0

      REAL(q) ENCHG
      LOGICAL LCHGFIT
      LOGICAL, PRIVATE :: LGAUSRC

      CONTAINS

!**********************************************************************
!                                           
! this subroutine reads the relevant tags from the INCAR file
!
!**********************************************************************

      SUBROUTINE CHGFIT_READER(IO,NTYPP)
      USE prec
      USE base
      IMPLICIT NONE
      TYPE (in_struct) IO
      INTEGER NTYPP
      ! local variables
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM

      ! read relevant stuff from INCAR 
      LOPEN=.FALSE.
      OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
      LCHGFIT=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LCHGFIT','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LCHGFIT,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LCHGFIT'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LCHGFIT','L',IDUM,RDUM,CDUM,LCHGFIT,CHARAC,N)      

      LGAUSRC=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LGAUSRC','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LGAUSRC,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LGAUSRC'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LGAUSRC','L',IDUM,RDUM,CDUM,LGAUSRC,CHARAC,N)      

      ENCHG=-1
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'ENCHG','=','#',';','F', &
     &            IDUM,ENCHG,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''ENCHG'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('ENCHG','F',IDUM,ENCHG,CDUM,LDUM,CHARAC,N)      

      ! the number of Gaussians that define the "charge-transfer"
      ! charges for non-scf calculations (default is 0)
      NGAUS=0
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'NGAUS','=','#',';','I', &
     &            NGAUS,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''NGAUS'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('NGAUS','I',NGAUS,RDUM,CDUM,LDUM,CHARAC,N)

      IF (NGAUS>0) THEN
      ALLOCATE(ZCT(NTYPP*NGAUS),RGAUS(NTYPP*NGAUS))
      ! "charge transfer" charges for non-scf calculations (default is 0)
      ZCT=0
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'ZCT','=','#',';','F', &
     &            IDUM,ZCT,CDUM,LDUM,CHARAC,N,NGAUS*NTYPP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<NGAUS*NTYPP))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''ZCT'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR_V('ZCT','F',IDUM,ZCT,CDUM,LDUM,CHARAC,N)

      ! widths of Gaussian CT charge distributions
      RGAUS=0
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'RGAUS','=','#',';','F', &
     &            IDUM,RGAUS,CDUM,LDUM,CHARAC,N,NGAUS*NTYPP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<NGAUS*NTYPP))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''RGAUS'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR_V('RGAUS','F',IDUM,RGAUS,CDUM,LDUM,CHARAC,N)
      ENDIF

      TAU0=-1    
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'TAU0','=','#',';','F', &
     &            IDUM,TAU0,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''TAU0'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('TAU0','F',IDUM,TAU0,CDUM,LDUM,CHARAC,N)

      CLOSE(IO%IU5)
      RETURN

  150 CONTINUE
      IF (IO%IU0>=0) &
      WRITE(IO%IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP

      RETURN
      END SUBROUTINE CHGFIT_READER


      SUBROUTINE CHGFIT_WRITER(IO)
      USE prec
      USE base
      IMPLICIT NONE
      TYPE (in_struct) IO
      ! local variables
      INTEGER I

      IF (IO%IU6>=0.AND.NGAUS/=0) THEN
         WRITE(IO%IU6,'(X,A)') 'Adding atom centered Gaussian charge distributions'
         WRITE(IO%IU6,'(4X,A,I6)') 'NGAUS =',NGAUS
!        WRITE(IO%IU6,'(2X,A)')  'Charge-transfer charges'   
         WRITE(IO%IU6,'(4X,A)',ADVANCE='No') 'ZCT   ='
         DO I=1,SIZE(ZCT)
            WRITE(IO%IU6,'(F14.7)',ADVANCE='No') ZCT(I)
         ENDDO
         WRITE(IO%IU6,*)
!        WRITE(IO%IU6,'(2X,A)')  'Gaussian widths'
         WRITE(IO%IU6,'(4X,A)',ADVANCE='No') 'RGAUS ='
         DO I=1,SIZE(RGAUS)
            WRITE(IO%IU6,'(F14.7)',ADVANCE='No') RGAUS(I)
         ENDDO
         WRITE(IO%IU6,'(/)')
      ENDIF

      IF (IO%IU6>=0.AND.TAU0>-1._q) THEN
         WRITE(IO%IU6,'(X,A)') 'Set total kinetic energy density'
         WRITE(IO%IU6,'(4X,A,F14.7)') 'TAU0  =',TAU0
         WRITE(IO%IU6,*)
      ENDIF

      RETURN
      END SUBROUTINE CHGFIT_WRITER


      FUNCTION LWRT_CHGFIT()
      IMPLICIT NONE
      LOGICAL LWRT_CHGFIT
      LWRT_CHGFIT=LCHGFIT
      END FUNCTION LWRT_CHGFIT


!******************** SUBROUTINE ATOMIC_CHARGES ************************
!
!***********************************************************************
      SUBROUTINE ATOMIC_CHARGES(T_INFO,LATT_CUR,P,GRIDC,CSTRF,A,B,CHTOT)
      USE prec
      USE poscar
      USE lattice
      USE mgrid
      USE pseudo
      USE charge
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      REAL(q) A,B
      COMPLEX(q) CHTOT(GRIDC%MPLWV)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      ! local variables
      COMPLEX(q) CTMP(GRIDC%MPLWV)
#ifdef debug
      REAL(q) RHOT
      REAL(q), EXTERNAL :: RHO0
      RHOT=RHO0(GRIDC,CHTOT(1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif

      CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CTMP)

      CHTOT=A*CHTOT+B*CTMP

#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif
      RETURN
      END SUBROUTINE ATOMIC_CHARGES


!******************** SUBROUTINE WRITE_CHARGE_RC ***********************
!
!***********************************************************************

      SUBROUTINE WRITE_CHARGE_RC(INFO,T_INFO,LATT_CUR,GRIDC,CHTOT,IO,IU)
      USE prec
      USE base
      USE mgrid
      USE poscar
      USE lattice
      USE constant
      IMPLICIT NONE
      TYPE (info_struct) INFO
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      COMPLEX(q) CHTOT(GRIDC%MPLWV)
      INTEGER IU
      ! local variables
      INTEGER N1,N2,N3,NG
      INTEGER NT,NIS,NI
      REAL(q) GX,GY,GZ,GSQU,SFAC
      REAL(q) ECUT
      COMPLEX(q) CSTRF
      COMPLEX(q), ALLOCATABLE :: CWORK(:)

      IF (ENCHG<0) THEN
         ECUT=INFO%ENAUG
      ELSE
         ECUT=ENCHG
      ENDIF

      ALLOCATE(CWORK(GRIDC%NGY_rd*GRIDC%NGZ_rd))

      NG=0
      DO N1=1,GRIDC%RC%NROW
      DO N2=1,GRIDC%NGY_rd
      DO N3=1,GRIDC%NGZ_rd
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*TPI
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*TPI
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*TPI
         GSQU=GX**2+GY**2+GZ**2
         IF (GSQU>0._q.AND.HSQDTM*GSQU<=ECUT) NG=NG+1
      ENDDO
      ENDDO
      ENDDO
      IF (IO%IU6>=0) THEN
         OPEN(UNIT=IU,FILE='CHGFIT',STATUS='UNKNOWN')
         WRITE(IU,'(2I6)',ADVANCE='No') NG,T_INFO%NTYP
         DO NT=1,T_INFO%NTYP
            WRITE(IU,'(I6)',ADVANCE='No') T_INFO%NITYP(NT)
         ENDDO
         WRITE(IU,*)
      ENDIF

      DO N1=1,GRIDC%RC%NROW
         CALL MRG_GRID_RC_PLANE(GRIDC,CWORK,CHTOT,N1)
         IF (IO%IU6>=0) THEN
            NG=0
            DO N2=1,GRIDC%NGY_rd
            DO N3=1,GRIDC%NGZ_rd
               NG=NG+1
               GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*TPI
               GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*TPI
               GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*TPI
               GSQU=GX**2+GY**2+GZ**2

               IF (GSQU>0._q.AND.HSQDTM*GSQU<=ECUT) THEN
                  SFAC=SQRT(2._q)
                  IF ((GRIDC%LPCTX(N1)==0.AND.GRIDC%NGX/=GRIDC%NGX_rd).OR. &
                 &    (GRIDC%LPCTZ(N3)==0.AND.GRIDC%NGZ/=GRIDC%NGZ_rd)) SFAC=1._q
                  WRITE(IU,'(3I6,3E18.10)',ADVANCE='No') GRIDC%LPCTX(N1),GRIDC%LPCTY(N2),GRIDC%LPCTZ(N3),GSQU,CWORK(NG)*SFAC/SQRT(GSQU)
!                 WRITE(IU,'(3I6,3E18.10)',ADVANCE='No') GRIDC%LPCTX(N1),GRIDC%LPCTY(N2),GRIDC%LPCTZ(N3),GSQU,CWORK(NG)*SFAC/GSQU
 
                  NIS=1
                  type: DO NT=1,T_INFO%NTYP
                  CSTRF=0
                  ions: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
                     CSTRF=CSTRF+EXP(-CITPI*(T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)+ &
                    &                         T_INFO%POSION(2,NI)*GRIDC%LPCTX(N2)+ &
                    &                          T_INFO%POSION(3,NI)*GRIDC%LPCTX(N3)))
                  ENDDO ions
                  NIS = NIS+T_INFO%NITYP(NT) 
                  WRITE(IU,'(2E18.10)',ADVANCE='No') CSTRF*SFAC/SQRT(GSQU)
!                 WRITE(IU,'(2E18.10)',ADVANCE='No') CSTRF*SFAC/GSQU
                  ENDDO type

                  WRITE(IU,*)
               ENDIF 

            ENDDO
            ENDDO
         ENDIF
      ENDDO

      DEALLOCATE(CWORK)

      IF (IO%IU6>=0) CLOSE(IU)

      RETURN
      END SUBROUTINE WRITE_CHARGE_RC


!******************** SUBROUTINE RHOADD_GAUSSIANS **********************
!
!***********************************************************************
      SUBROUTINE RHOADD_GAUSSIANS(T_INFO,LATT_CUR,P,GRIDC,NCDIJ,CHTOT, CSTRF)
      USE prec
      USE poscar
      USE lattice
      USE mgrid
      USE pseudo
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
#ifdef debug
      REAL(q) RHOT
      REAL(q), EXTERNAL :: RHO0
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif
      ! quick return if possible
      IF (NGAUS==0) RETURN

      IF (.NOT. LGAUSRC) THEN
         ! take total charge to real space
         CALL FFT3D(CHTOT(1,1),GRIDC,1)

         CALL RHOADD_RL_GAUSSIANS(T_INFO,LATT_CUR,P,GRIDC,CHTOT(1,1))

         ! and back to reciprocal space
         CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      ELSE
         CALL RHOADD_RC_GAUSSIANS(T_INFO,LATT_CUR,P,GRIDC,CHTOT(1,1),CSTRF(1,1))
      ENDIF
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif
      RETURN
      END SUBROUTINE RHOADD_GAUSSIANS


!******************** SUBROUTINE RHOADD_GAUSSIANS_LIST *****************
!
!***********************************************************************
      SUBROUTINE RHOADD_GAUSSIANS_LIST(LATT_CUR,GRIDC,NCDIJ,CHTOT)
      USE prec
      USE lattice
      USE mgrid
      IMPLICIT NONE
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)
      ! local variables
      INTEGER Num_Gaus
      REAL(q), ALLOCATABLE :: Z_Gaus(:),Sigma_Gaus(:),Pos_Gaus(:,:)
      INTEGER I
      LOGICAL LFOUND
#ifdef debug
      REAL(q) RHOT
      REAL(q), EXTERNAL :: RHO0
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif

      INQUIRE(FILE='GAUSSIANS',EXIST=LFOUND)
      ! quick return if possible
      IF (.NOT.LFOUND) RETURN

      ! Read list of Gaussians from file GAUSSIANS
      OPEN(UNIT=99,FILE='GAUSSIANS',STATUS='OLD')
      READ(99,*) Num_Gaus
      ALLOCATE(Z_Gaus(Num_Gaus),Sigma_Gaus(Num_Gaus),Pos_Gaus(3,Num_Gaus))
      DO I=1,Num_Gaus
         READ(99,*,ERR=100) Pos_Gaus(1,I),Pos_Gaus(2,I),Pos_Gaus(3,I),Z_Gaus(I),Sigma_Gaus(I)
!        WRITE(*,'(5F14.7)') Pos_Gaus(:,I),Z_Gaus(I),Sigma_Gaus(I)
      ENDDO
      CLOSE(99)

      ! take total charge to real space
      CALL FFT3D(CHTOT(1,1),GRIDC,1)

      CALL RHOADD_RL_GAUSSIANS_LIST(Num_Gaus,Z_Gaus,Sigma_Gaus,Pos_Gaus,LATT_CUR,GRIDC,CHTOT(1,1))

      ! and back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
#endif
      ! deallocation
      DEALLOCATE(Z_Gaus,Sigma_Gaus,Pos_Gaus)
      RETURN
      ! Error reading GAUSSIANS?
 100  CONTINUE
      WRITE(*,'(A,I7,A)') 'RHOADD_GAUSSIANS_LIST: ERROR: unable to read line',I,' of GAUSSIANS'
      DEALLOCATE(Z_Gaus,Sigma_Gaus,Pos_Gaus)
      RETURN
      END SUBROUTINE RHOADD_GAUSSIANS_LIST


!******************** SUBROUTINE SET_TAU0 ******************************
!
!***********************************************************************
      SUBROUTINE SET_TAU0(GRIDC,KINEDEN)
      USE prec
      USE mgrid
      USE meta
      IMPLICIT NONE
      TYPE (grid_3d) GRIDC
      TYPE (tau_handle) KINEDEN
      ! quick return if possible
      IF (TAU0<0._q) RETURN
      ! take the kinetic energy density to reciprocals space
      CALL FFT_RC_SCALE(KINEDEN%TAU(1,1),KINEDEN%TAU(1,1),GRIDC)
      CALL SETUNB_COMPAT(KINEDEN%TAU(1,1),GRIDC)
      ! set the G=0 element to TAU0
      CALL SET_RHO0(GRIDC,KINEDEN%TAU,TAU0)
      ! and back to real space
      CALL FFT3D(KINEDEN%TAU(1,1),GRIDC,1)
      RETURN
      END SUBROUTINE SET_TAU0

      END MODULE chgfit


      MODULE stockholder
      USE prec
      IMPLICIT NONE

      LOGICAL LSTOCKHOLDER

      REAL(q), ALLOCATABLE :: AngularGrids(:,:)
      INTEGER, ALLOCATABLE :: AngularGridOffset(:),NAng(:)

      PRIVATE :: RHOATO_PARTICLE_MESH,POLINT,POLIN3,SPLVL

      CONTAINS

      SUBROUTINE STOCKHOLDER_READER(IO)
      USE prec
      USE base
      IMPLICIT NONE
      TYPE (in_struct) IO
      ! local variables
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM

      ! read relevant stuff from INCAR 
      LOPEN=.FALSE.
      OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
      LSTOCKHOLDER=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LSTOCKHOLDER','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LSTOCKHOLDER,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LSTOCKHOLDER'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LSTOCKHOLDER','L',IDUM,RDUM,CDUM,LSTOCKHOLDER,CHARAC,N)      

      CLOSE(IO%IU5)
      RETURN

  150 CONTINUE
      IF (IO%IU0>=0) &
      WRITE(IO%IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP

      RETURN
      END SUBROUTINE STOCKHOLDER_READER


!******************** SUBROUTINE ANALYSE_CHARGES ***********************
!
!***********************************************************************
      SUBROUTINE ANALYSE_CHARGES(T_INFO,LATT_CUR,P,GRIDC,NCDIJ,CHTOT,IO)
      USE prec
      USE base
      USE poscar
      USE lattice
      USE mgrid
      USE pseudo
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)

      ! take total charge to real space
      CALL FFT3D(CHTOT(1,1),GRIDC,1)

      CALL ANALYSE_RL3(T_INFO,LATT_CUR,P,GRIDC,CHTOT(1,1),IO)

      ! and back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)

      RETURN
      END SUBROUTINE ANALYSE_CHARGES


!******************** SUBROUTINE RHOATO_PARTICLE_MESH ******************
!
!***********************************************************************
      SUBROUTINE RHOATO_PARTICLE_MESH(INFO,T_INFO,LATT_CUR,P,GRIDC,NCDIJ,CHTOT)
      USE prec
      USE base
      USE poscar
      USE lattice
      USE mgrid
      USE pseudo
      USE constant
      USE spline
      IMPLICIT NONE
      TYPE (info_struct) INFO
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)
      ! local variables
      INTEGER, PARAMETER :: NFFT=32768
      TYPE (splfnc) SPLINES(T_INFO%NTYP)
      REAL(q) NORMS(T_INFO%NTYP)
      INTEGER NT
      REAL(q) G1,G2,GMAX
!#ifdef debug
      REAL(q) RHOT
      REAL(q), EXTERNAL :: RHO0
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
!#endif
      ! take total charge to real space
      CALL FFT3D(CHTOT(1,1),GRIDC,1)

      DO NT=1,T_INFO%NTYP
         GMAX=SQRT(INFO%ENMAX/HSQDTM)
         G1=2.0_q*GMAX; G2=MIN(3.0_q*GMAX,P(NT)%PSGMAX)
         ! Fourier-filter atomic charge density,
         ! transform to real space and calculate spline interpolation
         ALLOCATE(SPLINES(NT)%SPL(NFFT/2,5))
         SPLINES(NT)%NP=NFFT/2
         CALL ATORHO_FILTER(G1,G2,NPSPTS,P(NT)%PSPRHO,P(NT)%PSGMAX/NPSPTS,NFFT,SPLINES(NT)%SPL,SPLINES(NT)%RMAX,-1)
         NORMS(NT)=P(NT)%ZVALF
         P(NT)%RCUTATO=SPLINES(NT)%RMAX
      ENDDO

!     CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLINES,NORMS,GRIDC,CHTOT(1,1))

      ! and back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
!#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
!#endif
      RETURN
      END SUBROUTINE RHOATO_PARTICLE_MESH


!******************** SUBROUTINE STOCKHOLDER_ANALYSIS ******************
!
!***********************************************************************
      SUBROUTINE STOCKHOLDER_ANALYSIS2(INFO,T_INFO,LATT_CUR,P,GRIDC,NCDIJ,CHTOT,DENCOR,IO)
      USE prec
      USE base
      USE poscar
      USE lattice
      USE mgrid
      USE pseudo
      USE constant
      USE spline
      IMPLICIT NONE
      TYPE (info_struct) INFO
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)
      RGRID      DENCOR(GRIDC%RL%NP) 
      ! local variables
      TYPE (splfnc) SPLF0(T_INFO%NTYP),SPLF1(T_INFO%NTYP),SPLFC(T_INFO%NTYP)
      INTEGER I
      INTEGER NT,IR
      REAL(q) D,DQ,DQD,QN,F
      REAL(q) G1,G2,GMAX
      REAL(q) NORMS(T_INFO%NTYP)
      COMPLEX(q) CHTOTW(GRIDC%MPLWV)

      REAL(q), PARAMETER :: TINY=1.E-5_q

      INTEGER, PARAMETER :: NSCF=20
      REAL(q), PARAMETER :: AMIX=0.5_q,CONV=1E-4
      INTEGER, PARAMETER :: NR=200,NFFT=32768

!#ifdef debug
      REAL(q) RHOT
      REAL(q), EXTERNAL :: RHO0
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
!#endif

      ! take total charge to real space
      CALL FFT3D(CHTOT(1,1),GRIDC,1)
      ! and add the partial core charge
      CALL ADDRHOC(GRIDC,CHTOT(1,1),1._q,DENCOR(1),1._q)

      DO NT=1,T_INFO%NTYP
         GMAX=SQRT(INFO%ENMAX/HSQDTM)
         G1=2.0_q*GMAX; G2=MIN(3.0_q*GMAX,P(NT)%PSGMAX)

         ! Fourier-filter atomic charge density,
         ! transform to real space and calculate spline interpolation
         ALLOCATE(P(NT)%ATOSPL(NFFT/2,5))
         CALL ATORHO_FILTER(G1,G2,NPSPTS,P(NT)%PSPRHO,P(NT)%PSGMAX/NPSPTS,NFFT,P(NT)%ATOSPL,P(NT)%RCUTATO,-1)
         ! transfer to f(r)*r^2 to SPLF0
         ALLOCATE(SPLF0(NT)%SPL(NR,5)); SPLF0(NT)%SPL=0
         SPLF0(NT)%NP=NR
         SPLF0(NT)%RMAX=P(NT)%RCUTATO
         DO IR=1,NR
            D=SPLF0(NT)%RMAX*(IR-0.5)/NR
            SPLF0(NT)%SPL(IR,1)=SPLF0(NT)%RMAX*(IR-0.5)/SPLF0(NT)%NP
            CALL SPLVAL(D,DQ,DQD,P(NT)%ATOSPL,SIZE(P(NT)%ATOSPL,1),SIZE(P(NT)%ATOSPL,1))
            SPLF0(NT)%SPL(IR,1)=D
            SPLF0(NT)%SPL(IR,2)=DQ*D*D
         ENDDO
         NORMS(NT)=P(NT)%ZVALF

         ! allocate SPLF1
         ALLOCATE(SPLF1(NT)%SPL(NR,5)); SPLF1(NT)%SPL=0
         SPLF1(NT)%NP=NR
         SPLF1(NT)%RMAX=P(NT)%RCUTATO

         ! allocate SPLFC
         ALLOCATE(SPLFC(NT)%SPL(NR,5)); SPLFC(NT)%SPL=0
         SPLFC(NT)%NP=NR
         SPLFC(NT)%RMAX=SPLF0(NT)%RMAX
         SPLFC(NT)%SPL(:,1)=SPLF0(NT)%SPL(:,1)
         IF (ASSOCIATED(P(NT)%PSPCOR)) THEN
            ! Fourier-filter partial core charge density, stored in SPLFC
            CALL ATORHO_FILTER(G1,G2,NPSPTS,P(NT)%PSPCOR,P(NT)%PSGMAX/NPSPTS,NFFT,P(NT)%ATOSPL,P(NT)%RCUTATO,-1)
            DO IR=1,NR
               D=SPLFC(NT)%SPL(IR,1)
               IF (D>P(NT)%RCUTATO) EXIT
               CALL SPLVAL(D,DQ,DQD,P(NT)%ATOSPL,SIZE(P(NT)%ATOSPL,1),SIZE(P(NT)%ATOSPL,1))
               SPLFC(NT)%SPL(IR,2)=DQ*D*D 
            ENDDO
         ENDIF
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(100)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR+1
            D=SPLF0(NT)%RMAX/(NR+1)*(IR-1)+TINY
            CALL LINPOL(D,SPLF0(NT),DQ)
            WRITE(100,'(2F14.7)') D,DQ
         ENDDO
         WRITE(100,*)
      ENDDO
      CLOSE(100)
      OPEN(110)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR+1
            D=SPLFC(NT)%RMAX/(NR+1)*(IR-1)+TINY
            CALL LINPOL(D,SPLFC(NT),DQ)
            WRITE(110,'(2F14.7)') D,DQ
         ENDDO
         WRITE(110,*)
      ENDDO
      CLOSE(100)
      OPEN(300)
      DO NT=1,T_INFO%NTYP
         DO IR=1,1000
            D=P(NT)%RCUTATO/1000*IR
            CALL SPLVAL(D,DQ,DQD,P(NT)%ATOSPL,SIZE(P(NT)%ATOSPL,1),SIZE(P(NT)%ATOSPL,1))
            WRITE(300,'(2F14.7)') D,DQ!*LATT_CUR%OMEGA/GRIDC%NPLWV
         ENDDO
         WRITE(300,*)
      ENDDO
      CLOSE(300)
      ENDIF
!#endif

      ! set up CHTOTW
      CHTOTW=0
      CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLF0,NORMS,.TRUE.,GRIDC,CHTOTW(1),IO)
 
      sc: DO I=1,NSCF
         ! Compute the new weight functions
         CALL AVERAGE_RL(T_INFO,LATT_CUR,GRIDC,CHTOT(1,1),CHTOTW(1),SPLF0,SPLF1,IO)
         ! mix SPLF0 and SPLF1
         DQ=0; DQD=0
         DO NT=1,T_INFO%NTYP
         DO IR=1,MIN(SPLF0(NT)%NP,SPLF1(NT)%NP)
            D=SPLF1(NT)%SPL(IR,2)-SPLF0(NT)%SPL(IR,2)
            DQ=DQ+ABS(D); DQD=DQD+D
            SPLF0(NT)%SPL(IR,2)=SPLF0(NT)%SPL(IR,2)+AMIX*D
         ENDDO
         ENDDO
         ! check for convergence
         IF (DQ<CONV) EXIT sc
!#ifdef debug
         IF (IO%IU6>=0) THEN
            WRITE(*,'(A,I4,A,F14.7,A,F14.7)') 'CYCLE=',I,' DQ=',DQ,' DQD=',DQD
            OPEN(151)
            DO NT=1,T_INFO%NTYP
               DO IR=1,NR+1
                  D=SPLF0(NT)%RMAX/(NR+1)*(IR-1)+TINY
                  CALL LINPOL(D,SPLF0(NT),F)
                  WRITE(151,'(2F14.7)') D,F
              ENDDO
              WRITE(151,*)
            ENDDO
            CLOSE(151)
         ENDIF
!#endif
         ! set up CHTOTW with new weight functions
         CHTOTW=0
         CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLF0,NORMS,.FALSE.,GRIDC,CHTOTW(1),IO)
      ENDDO sc

      IF (IO%IU6>=0) THEN
         IF (I>NSCF) THEN
            WRITE(*,*) 'STOCKHOLDER_ANALYSIS: Convergence was not reached in:',NSCF,' cycles'
            WRITE(*,*) '                    total change in weight functions:',DQ
         ELSE
            !
         ENDIF
      ENDIF

      ! subtract the partial core charge
      DO NT=1,T_INFO%NTYP
         SPLF0(NT)%SPL(:,2)=SPLF0(NT)%SPL(:,2)-SPLFC(NT)%SPL(:,2)
      ENDDO
      ! set up CHTOTW 
      CHTOTW=0
      CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLF0,NORMS,.FALSE.,GRIDC,CHTOTW(1),IO)     

      ! 
      DQ=0; QN=0
      DO NT=1,T_INFO%NTYP
         DQ=DQ+T_INFO%NITYP(NT)*NORMS(NT)
         QN=QN+T_INFO%NITYP(NT)*ABS(NORMS(NT))
      ENDDO
      DQ=DQ-INFO%NELECT

! test
      IF (IO%IU6>=0) THEN
         WRITE(*,*) 'DQ=',DQ
      ENDIF
! test
      NORMS(:)=NORMS(:)-DQ*ABS(NORMS(:))/QN
      CHTOT(:,1)=0
      CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLF0,NORMS,.TRUE.,GRIDC,CHTOT(1,1),IO)
      
!     CALL ANALYSE_RL3(T_INFO,LATT_CUR,P,GRIDC,CHTOTW(1),IO)

      ! take CHTOT back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
!#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'STOCKHOLDER_ANALYSIS: Total number of electrons =',RHOT
!#endif
      RETURN
      END SUBROUTINE STOCKHOLDER_ANALYSIS2


!******************** SUBROUTINE STOCKHOLDER_ANALYSIS ******************
!
!***********************************************************************
      SUBROUTINE STOCKHOLDER_ANALYSIS(INFO,T_INFO,LATT_CUR,P,GRIDC,NCDIJ,CHTOT,DENCOR,IO)
      USE prec
      USE base
      USE mgrid
      USE poscar
      USE lattice
      USE pseudo
      USE constant
      USE spline
      IMPLICIT NONE
      TYPE (info_struct) INFO
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      INTEGER NCDIJ
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ)
      RGRID      DENCOR(GRIDC%RL%NP)
      ! local variables
      TYPE (splfnc) SPLFTMP
      TYPE (splfnc) SPLF0(T_INFO%NTYP),SPLF1(T_INFO%NTYP),SPLFC(T_INFO%NTYP)
      TYPE (splfnc) SPLF2(T_INFO%NTYP)
      TYPE (splfnc) SPLF0_(T_INFO%NIONS),SPLF1_(T_INFO%NIONS)
      REAL(q) NORMS(T_INFO%NTYP)
      REAL(q), ALLOCATABLE :: CHGLOBAL(:,:,:)

      INTEGER I,NT,IR,NI,NIS
      INTEGER, PARAMETER :: NR=200,NFFT=32768
      REAL(q), PARAMETER :: RCUT=2.5_q

      REAL(q) SCALE,FRICTION
      INTEGER, PARAMETER :: NSCF=800
      REAL(q), PARAMETER :: AMIX=0.5_q,CONV=1E-4
      REAL(q), ALLOCATABLE :: VEL(:,:)

      REAL(q) RHOT
      REAL(q) G1,G2,GMAX,D,DQ,DQP,DQD,XX,R0,QN
      REAL(q), PARAMETER :: RM=1._q
      REAL(q), EXTERNAL :: RHO0

      INTEGER, PARAMETER :: NFFTP=4096
      REAL(q) DELQL,RMAXF,RDEL,WINDOW
      REAL(q), ALLOCATABLE :: FFT(:)

      ! quick return if possible
      IF (.NOT.LSTOCKHOLDER) RETURN

!#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      IF (IO%IU6>=0) WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT
!#endif

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(190)
      DO NT=1,T_INFO%NTYP
         DELQL=P(NT)%PSGMAX/NPSPTS
         DO I=1,NPSPTS
            WRITE(190,'(2F14.7)') DELQL*(I-1),P(NT)%PSPRHO(I) 
         ENDDO
         WRITE(190,*) 
      ENDDO
      CLOSE(190)
      ENDIF
!#endif

      DO NT=1,T_INFO%NTYP
         GMAX=SQRT(INFO%ENMAX/HSQDTM)
         G1=2.0_q*GMAX; G2=MIN(3.0_q*GMAX,P(NT)%PSGMAX)
         ! Fourier-filter atomic charge density,
         ! transform to real space and calculate spline interpolation
         ALLOCATE(P(NT)%ATOSPL(NFFT/2,5))
         CALL ATORHO_FILTER(G1,G2,NPSPTS,P(NT)%PSPRHO,P(NT)%PSGMAX/NPSPTS,NFFT,P(NT)%ATOSPL,P(NT)%RCUTATO,-1)
         ! store into SPLF0
         SPLF0(NT)%RMAX=MIN(P(NT)%RCUTATO,RCUT)
         ! determine the grid dimensions
         SPLF0(NT)%NGEN=NR; SPLF0(NT)%NP=NR
         XX=-COS(PI/REAL(NR+1,KIND=q))
         R0=1E-5_q-RM*SQRT((1+XX)/(1-XX))
         DO IR=1,NR
            XX=-COS(REAL(IR,KIND=q)*PI/REAL(NR+1,KIND=q))
            D=RM*SQRT((1+XX)/(1-XX))+R0
            IF (D>SPLF0(NT)%RMAX) THEN
               SPLF0(NT)%RMAX=D; SPLF0(NT)%NP=IR; EXIT
            ENDIF
         ENDDO    
         ! setup grid
         ALLOCATE(SPLF0(NT)%SPL(SPLF0(NT)%NP,5)); SPLF0(NT)%SPL=0
         XX=-COS(PI/REAL(NR+1,KIND=q))
         R0=1E-5_q-RM*SQRT((1+XX)/(1-XX))
         DO IR=1,SPLF0(NT)%NP
            XX=-COS(REAL(IR,KIND=q)*PI/REAL(NR+1,KIND=q))
            D=RM*SQRT((1+XX)/(1-XX))+R0
            SPLF0(NT)%SPL(IR,1)=D
            DQ=0
            IF (D.LE.SPLF0(NT)%RMAX) THEN
               CALL SPLVAL(D,DQ,DQD,P(NT)%ATOSPL,SIZE(P(NT)%ATOSPL,1),SIZE(P(NT)%ATOSPL,1)) 
            ENDIF
            SPLF0(NT)%SPL(IR,2)=DQ
         ENDDO
         CALL SPLCOF(SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1),0._q)
         ! allocate SPLF1 and SPLF2
         ALLOCATE(SPLF1(NT)%SPL(SPLF0(NT)%NP,5),SPLF2(NT)%SPL(SPLF0(NT)%NP,5))
         CALL COPYSPL(SPLF0(NT),SPLF1(NT)); CALL COPYSPL(SPLF0(NT),SPLF2(NT))
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(230)
      DO NT=1,T_INFO%NTYP
         DO IR=1,1000
            D=IR*SPLF0(NT)%RMAX/1000 
            CALL SPLVAL(D,DQ,DQD,SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1))
            WRITE(230,'(2F14.7)') D,DQ
         ENDDO
         WRITE(230,*) 
      ENDDO
      CLOSE(230)
      ENDIF
!#endif

      DO NT=1,T_INFO%NTYP
         GMAX=SQRT(INFO%ENMAX/HSQDTM)
         G1=2.0_q*GMAX; G2=MIN(3.0_q*GMAX,P(NT)%PSGMAX)
         ! allocate and setup SPLFC
         ALLOCATE(SPLFC(NT)%SPL(SPLF0(NT)%NP,5))
         CALL COPYSPL(SPLF0(NT),SPLFC(NT)); SPLFC(NT)%SPL(:,2:5)=0
         IF (ASSOCIATED(P(NT)%PSPCOR)) THEN
            ! Fourier-filter partial core charge density, stored in SPLFC
            CALL ATORHO_FILTER(G1,G2,NPSPTS,P(NT)%PSPCOR,P(NT)%PSGMAX/NPSPTS,NFFT,P(NT)%ATOSPL,P(NT)%RCUTATO,-1)
            DO IR=1,SPLFC(NT)%NP
               D=SPLFC(NT)%SPL(IR,1)
               IF (D>P(NT)%RCUTATO) EXIT
               CALL SPLVAL(D,DQ,DQD,P(NT)%ATOSPL,SIZE(P(NT)%ATOSPL,1),SIZE(P(NT)%ATOSPL,1))
               SPLFC(NT)%SPL(IR,2)=DQ 
            ENDDO
         ENDIF
         ! Add partial core charge SPLFC to atomic charge density SPLF0
         SPLF0(NT)%SPL(:,2)=SPLF0(NT)%SPL(:,2)+SPLFC(NT)%SPL(:,2)
         SPLF0(NT)%SPL(:,2)=MAX(ABS(SPLF0(NT)%SPL(:,2)),1E-6_q)
         CALL SPLCOF(SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1),0._q)
         ! backup in SPLF2
         SPLF2(NT)%SPL=SPLF0(NT)%SPL
      ENDDO

! test
!     DO NT=1,T_INFO%NTYP
!        DO IR=1,SPLF0(NT)%NP
!           ! initialize with an exponential
!           SPLF0(NT)%SPL(IR,2)=EXP(-6._q*LOG(10._q)/SPLF0(NT)%RMAX*SPLF0(NT)%SPL(IR,1))
!           ! or with a uniform distribution
!!          SPLF0(NT)%SPL(IR,2)=1._q 
!        ENDDO
!        CALL SPLCOF(SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1),0._q)
!        SPLF1(NT)%SPL=SPLF0(NT)%SPL
!     ENDDO
! test

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(330)
      DO NT=1,T_INFO%NTYP
         DO IR=1,1000
            D=IR*SPLF0(NT)%RMAX/1000 
            CALL SPLVAL(D,DQ,DQD,SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1))
            WRITE(330,'(2F14.7)') D,DQ
         ENDDO
         WRITE(330,*) 
      ENDDO
      CLOSE(330)
      ENDIF
!#endif

      ! take total charge to real space
      CALL FFT3D(CHTOT(1,1),GRIDC,1)
      ! and add the partial core charge
      CALL ADDRHOC(GRIDC,CHTOT(1,1),1._q,DENCOR(1),1._q)

      ! accumulate the complete charge density into CHGLOBAL
      ALLOCATE(CHGLOBAL(GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ))
      CALL MRG_GRID_RL(GRIDC,CHGLOBAL,CHTOT)

      ! setup Lebedev-Laikov grids
      CALL PrepareAngularGrids

      NIS=1
      DO NT=1,T_INFO%NTYP
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            ALLOCATE(SPLF0_(NI)%SPL(SPLF0(NT)%NP,5),SPLF1_(NI)%SPL(SPLF0(NT)%NP,5))
            CALL COPYSPL(SPLF0(NT),SPLF0_(NI)); CALL COPYSPL(SPLF0(NT),SPLF1_(NI))
         ENDDO
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO 


      ALLOCATE(VEL(NR,T_INFO%NIONS)); VEL=0._q
      FRICTION=2._q; SCALE=AMIX

      sc: DO I=1,NSCF
         CALL AVERAGE_RL2(T_INFO,LATT_CUR,GRIDC,CHGLOBAL,SPLF0_,SPLF1_,IO)
!#ifdef debug

!#endif
         ! mix
!        IF (I>35) FRICTION=0.6_q
         DQ=0; DQD=0
         DO NI=1,T_INFO%NIONS
            DO IR=1,MIN(SPLF0_(NI)%NP,SPLF1_(NI)%NP)
               D=SPLF1_(NI)%SPL(IR,2)-SPLF0_(NI)%SPL(IR,2)
               DQ=DQ+ABS(D); DQD=DQD+ABS(D)*SPLF0_(NI)%SPL(IR,1)**2
               VEL(IR,NI)=((1-FRICTION/2)*VEL(IR,NI)+2*SCALE*D)/(1+FRICTION/2)
               SPLF0_(NI)%SPL(IR,2)=SPLF0_(NI)%SPL(IR,2)+VEL(IR,NI)
            ENDDO
            CALL SPLCOF(SPLF0_(NI)%SPL,SPLF0_(NI)%NP,SIZE(SPLF0_(NI)%SPL,1),0._q)
         ENDDO

         ! check for convergence
         IF (DQ<CONV) EXIT sc
!#ifdef debug
         IF (IO%IU6>=0) THEN
            WRITE(*,'(A,I4,A,F14.7,A,F14.7)') 'CYCLE=',I,' DQ=',DQ,' DQD=',DQD
            ! Average over atoms of like species
            NIS=1
            DO NT=1,T_INFO%NTYP 
               SPLF1(NT)%SPL(:,2)=0
               DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
                  SPLF1(NT)%SPL(:,2)=SPLF1(NT)%SPL(:,2)+SPLF0_(NI)%SPL(:,2)
               ENDDO
               SPLF1(NT)%SPL(:,2)=SPLF1(NT)%SPL(:,2)/T_INFO%NITYP(NT)
               CALL SPLCOF(SPLF1(NT)%SPL,SPLF1(NT)%NP,SIZE(SPLF1(NT)%SPL,1),0._q)
               NIS=NIS+T_INFO%NITYP(NT)
            ENDDO
            ! Write out
            OPEN(220);OPEN(225)
            DO NT=1,T_INFO%NTYP
               DO IR=1,1000
                  D=IR*SPLF1(NT)%RMAX/1000 
                  CALL SPLVAL(D,DQ ,DQD,SPLF1(NT)%SPL,SPLF1(NT)%NP,SIZE(SPLF1(NT)%SPL,1))
                  CALL SPLVAL(D,DQP,DQD,SPLF2(NT)%SPL,SPLF2(NT)%NP,SIZE(SPLF2(NT)%SPL,1))
                  WRITE(220,'(2F14.7)') D,DQ
                  WRITE(225,'(2F14.7)') D,DQ-DQP
               ENDDO
               WRITE(220,*);WRITE(225,*)
               SPLF0(NT)%SPL=SPLF1(NT)%SPL
            ENDDO
            CLOSE(220);CLOSE(225)
         ENDIF
!#endif
      ENDDO sc

      DEALLOCATE(VEL)

      ! Average over atoms of like species
      NIS=1
      DO NT=1,T_INFO%NTYP 
         SPLF0(NT)%SPL(:,2)=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            SPLF0(NT)%SPL(:,2)=SPLF0(NT)%SPL(:,2)+SPLF0_(NI)%SPL(:,2)
         ENDDO
         SPLF0(NT)%SPL(:,2)=SPLF0(NT)%SPL(:,2)/T_INFO%NITYP(NT)
         CALL SPLCOF(SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1),0._q)
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO

      ! subtract the partial core charge
      DO NT=1,T_INFO%NTYP
         SPLF0(NT)%SPL(:,2)=SPLF0(NT)%SPL(:,2)-SPLFC(NT)%SPL(:,2)
         CALL SPLCOF(SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1),0._q)
         ! compute the norm
         CALL RADINT(SPLF0(NT),NORMS(NT))
      ENDDO

      ! enforce the right amount of charge
      DQ=0; QN=0
      DO NT=1,T_INFO%NTYP
         DQ=DQ+T_INFO%NITYP(NT)*NORMS(NT)
         QN=QN+T_INFO%NITYP(NT)*ABS(NORMS(NT))
      ENDDO
      DQ=DQ-REAL(INFO%NELECT,KIND=q)
      NORMS(:)=NORMS(:)-DQ*ABS(NORMS(:))/QN
!#ifdef debug
      IF (IO%IU6>=0) THEN
         WRITE(*,*) 'DQ=',DQ,' NELECT=',INFO%NELECT
         DQ=0
         DO NT=1,T_INFO%NTYP
            WRITE(*,'(I4,X,A,F14.7)') NT,'Z=',NORMS(NT)
            DQ=DQ+T_INFO%NITYP(NT)*NORMS(NT)
         ENDDO
         WRITE(*,*) 'DQ=',DQ
      ENDIF
!#endif

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(430)
      DO NT=1,T_INFO%NTYP
         DO IR=1,1000
            D=IR*SPLF1(NT)%RMAX/1000 
            CALL SPLVAL(D,DQ,DQD,SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1))
            WRITE(430,'(2F14.7)') D,DQ
         ENDDO
         WRITE(430,*) 
      ENDDO
      CLOSE(430)
      ENDIF
!#endif

      ! FFT SPLF0 to reciprocal space and store in PSPRHO
      SPLFTMP%NP=NFFTP
      ALLOCATE(SPLFTMP%SPL(SPLFTMP%NP,5))
      DO NT=1,T_INFO%NTYP
         DELQL=2*P(NT)%PSGMAX/NFFTP
         RMAXF=2*PI/DELQL
         RDEL=RMAXF/NFFTP
         ALLOCATE(FFT(NFFTP)); FFT=0
         DO I=1,NFFTP
            D=RDEL*(I-1)
            IF (D>SPLF0(NT)%RMAX) EXIT
            CALL SPLVAL(D,DQ,DQD,SPLF0(NT)%SPL,SPLF0(NT)%NP,SIZE(SPLF0(NT)%SPL,1))
            FFT(I)=DQ*D
         ENDDO
         ! sine transform to reciprocal space
         CALL SINFT(FFT,NFFTP)
         ! setup spline function
         SPLFTMP%SPL=0
         SPLFTMP%SPL(1,1)=0._q
         SPLFTMP%SPL(1,2)=NORMS(NT)
         DO I=2,NFFTP
            SPLFTMP%SPL(I,1)=DELQL*(I-1)/2._q
            SPLFTMP%SPL(I,2)=FFT(I)*RDEL*4*PI/(DELQL*(I-1))*2._q
         ENDDO
         CALL SPLCOF(SPLFTMP%SPL,SPLFTMP%NP,SIZE(SPLFTMP%SPL,1),0._q)
         ! backfill PSPRHO
         DO I=1,NPSPTS
            D=(I-1)*P(NT)%PSGMAX/NPSPTS
            CALL SPLVAL(D,DQ,DQD,SPLFTMP%SPL,SPLFTMP%NP,SIZE(SPLFTMP%SPL,1))
            P(NT)%PSPRHO(I)=DQ
         ENDDO
         DEALLOCATE(FFT)
      ENDDO

      ! output to STOCKCAR
      IF (IO%IU6>=0) THEN
      OPEN(UNIT=99,FILE='STOCKCAR',STATUS='UNKNOWN')
      DO NT=1,T_INFO%NTYP 
         WRITE(99,'(A,2X,A)') 'atomic pseudo charge-density:',P(NT)%ELEMENT
         WRITE(99,'(5E16.8)') P(NT)%PSPRHO(1:NPSPTS)
      ENDDO
      CLOSE(99)
      ENDIF
!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(440)
      DO NT=1,T_INFO%NTYP
         DO I=1,NPSPTS
            WRITE(440,'(2F14.7)') P(NT)%PSGMAX/NPSPTS*(I-1),P(NT)%PSPRHO(I) 
         ENDDO
         WRITE(440,*) 
      ENDDO
      CLOSE(440)
      ENDIF
!#endif

      ! subtract the partial core charge again
      CALL ADDRHOC(GRIDC,CHTOT(1,1),1._q,DENCOR(1),-1._q)
! test
!     CHTOT=0
!     CALL RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLF0,NORMS,.TRUE.,GRIDC,CHTOT(1,1),IO)
! test
      ! take CHTOT back to reciprocal space
      CALL FFT_RC_SCALE(CHTOT(1,1),CHTOT(1,1),GRIDC)
      CALL SETUNB_COMPAT(CHTOT(1,1),GRIDC)
!#ifdef debug
      RHOT=RHO0(GRIDC,CHTOT(1,1))
      WRITE(*,'(A,F20.10)') 'STOCKHOLDER_ANALYSIS: Total number of electrons =',RHOT
!#endif

      DEALLOCATE(CHGLOBAL,AngularGrids,AngularGridOffset,NAng)

      ! Deallocation
      NIS=1
      DO NT=1,T_INFO%NTYP 
         DEALLOCATE(SPLF0(NT)%SPL,SPLF1(NT)%SPL,SPLFC(NT)%SPL)
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            DEALLOCATE(SPLF0_(NI)%SPL,SPLF1_(NI)%SPL)
         ENDDO
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO
      DEALLOCATE(SPLFTMP%SPL)

      RETURN
      END SUBROUTINE STOCKHOLDER_ANALYSIS


!******************** SUBROUTINE ANALYSE_RL4 ***************************
!
!***********************************************************************
      SUBROUTINE ANALYSE_RL4(T_INFO,LATT_CUR,GRIDC,CHGLOBAL,SPL,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE wave
      USE spline
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      TYPE (splfnc) SPL(T_INFO%NTYP)
      REAL(q) CHGLOBAL(GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ)
      ! local variables
      INTEGER I,J
      INTEGER NI,NIP,NIS,NT,NR
      INTEGER ANGNUM
      REAL(q) XX,R,W,AREA
      REAL(q) RHO,SUM1,SUM2,RADWEIGHT
      REAL(q) TMP(3),POS(3),POSION(3)
      REAL(q), ALLOCATABLE :: FR(:,:),RR(:,:),Z(:)
      REAL(q), PARAMETER :: RM=1.0_q,ANGPARAM=0.8_q

      ALLOCATE(FR(MAXVAL(SPL(:)%NP),T_INFO%NIONS),RR(MAXVAL(SPL(:)%NP),T_INFO%NIONS),Z(T_INFO%NIONS))
      FR=0; RR=0; Z=0

      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI,GRIDC%COMM)
         NT=T_INFO%ITYP(NI)
         IF (NIP==0) CYCLE ion
         
         NR=SPL(NT)%NP

         ! atomic position in cartesian coordinates
         POSION(1)=T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         POSION(2)=T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         POSION(3)=T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)

         SUM2=0
         DO I=1,NR
            XX=COS(REAL(I,KIND=q)*PI/REAL(NR+1,KIND=q))
            R=RM*SQRT((1+XX)/(1-XX))

            RR(I,NI)=R

            IF (R>SPL(NT)%RMAX) CYCLE

            W=PI/REAL(NR+1,KIND=q)*(SIN(PI*REAL(I,KIND=q)/REAL(NR+1,KIND=q)))**2

            AREA=4*PI*R**2/(ANGPARAM**2)
            ANGNUM=0
            DO J=1,5
               IF (ANGNUM==0.AND.AREA<REAL(NAng(J),KIND=q)) ANGNUM=J
            ENDDO
            IF (ANGNUM==0) ANGNUM=5
 
            SUM1=0
            DO J=1,NAng(ANGNUM)
               TMP(1)=POSION(1)+R*AngularGrids(1,AngularGridOffset(ANGNUM)+J)
               TMP(2)=POSION(2)+R*AngularGrids(2,AngularGridOffset(ANGNUM)+J)
               TMP(3)=POSION(3)+R*AngularGrids(3,AngularGridOffset(ANGNUM)+J)
               ! In direct coordinates
               POS(1)=TMP(1)*LATT_CUR%B(1,1)+TMP(2)*LATT_CUR%B(2,1)+TMP(3)*LATT_CUR%B(3,1)
               POS(2)=TMP(1)*LATT_CUR%B(1,2)+TMP(2)*LATT_CUR%B(2,2)+TMP(3)*LATT_CUR%B(3,2)
               POS(3)=TMP(1)*LATT_CUR%B(1,3)+TMP(2)*LATT_CUR%B(2,3)+TMP(3)*LATT_CUR%B(3,3)
               ! 
               CALL INTERPOLATE_CHARGE(POS,LATT_CUR,GRIDC,CHGLOBAL(1,1,1),RHO)
               SUM1=SUM1+RHO*AngularGrids(4,AngularGridOffset(ANGNUM)+J)
            ENDDO
            RADWEIGHT=W*(1-XX)**(-3)*RM**3
            SUM2=SUM2+SUM1*RADWEIGHT*4*PI/LATT_CUR%OMEGA
            FR(I,NI)=SUM1/LATT_CUR%OMEGA
         ENDDO
         Z(NI)=SUM2
!        WRITE(*,*) NI,' Z=',SUM2
      ENDDO ion

      ! Communicate
      CALLMPI( M_sum_d(GRIDC%COMM, FR, T_INFO%NIONS*MAXVAL(SPL(:)%NP)) )
      CALLMPI( M_sum_d(GRIDC%COMM, RR, T_INFO%NIONS*MAXVAL(SPL(:)%NP)) )
      CALLMPI( M_sum_d(GRIDC%COMM,  Z, T_INFO%NIONS) )

      ! Average over atoms of like species
      NIS=1
      DO NT=1,T_INFO%NTYP 
         SPL(NT)%SPL(:,2)=0
         NR=SPL(NT)%NP
         SPL(NT)%SPL(1:NR,1)=RR(1:NR,NIS)
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            SPL(NT)%SPL(1:NR,2)=SPL(NT)%SPL(1:NR,2)+FR(1:NR,NI)
         ENDDO
         SPL(NT)%SPL(:,2)=SPL(NT)%SPL(:,2)/T_INFO%NITYP(NT)
         CALL SPLCOF(SPL(NT)%SPL,SPL(NT)%NP,SIZE(SPL(NT)%SPL,1),0._q)
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(210)
      DO NT=1,T_INFO%NTYP
         DO I=1,SPL(NT)%NP
            WRITE(210,'(2F14.7)') SPL(NT)%SPL(I,1),SPL(NT)%SPL(I,2)
         ENDDO
         WRITE(210,*)
      ENDDO
      CLOSE(210)
      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I4,X,A,F14.7)') NI,'Z=',Z(NI)
      ENDDO
      ENDIF
!#endif

      DEALLOCATE(FR,RR,Z)

      RETURN
      END SUBROUTINE ANALYSE_RL4


!******************** SUBROUTINE AVERAGE_RL2 ***************************
!
!***********************************************************************
      SUBROUTINE AVERAGE_RL2(T_INFO,LATT_CUR,GRIDC,CHGLOBAL,SPL0,SPL1,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE wave
      USE spline
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      TYPE (splfnc) SPL0(T_INFO%NIONS),SPL1(T_INFO%NIONS)
      REAL(q) CHGLOBAL(GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ)
      ! local variables
      INTEGER I,J,K
      INTEGER N,NI,NIP,NIS,NT,NR
      INTEGER ANGNUM
      REAL(q) XX,R,W,AREA,R0
      REAL(q) RHO,SUM1,SUM2,SUMW,RADWEIGHT,DQ,DQD
      REAL(q) TMP(3),POS(3),POSION(3)
      REAL(q), ALLOCATABLE :: FR(:,:),RR(:,:),Z(:),ZNT(:)
      REAL(q), PARAMETER :: RM=1.0_q,ANGPARAM=0.8_q

      REAL(q) BP(3,8),MIND
      REAL(q), PARAMETER :: TINY=1E-7_q

      ALLOCATE(FR(MAXVAL(SPL0(:)%NP),T_INFO%NIONS),RR(MAXVAL(SPL0(:)%NP),T_INFO%NIONS))
      ALLOCATE(Z(T_INFO%NIONS),ZNT(T_INFO%NTYP))
      FR=0; RR=0; Z=0; ZNT=0

      ! Calculate unit cell nodes 
      N=0
      DO I=0,1; DO J=0,1; DO K=0,1
         N=N+1; BP(:,N)=I*LATT_CUR%A(:,1)+J*LATT_CUR%A(:,2)+K*LATT_CUR%A(:,3)
      ENDDO; ENDDO; ENDDO
      ! Compute minimal distance between nodes
      MIND=LATT_CUR%ANORM(1)
      DO I=1,8; DO J=1,8
         IF (I/=J) THEN
            R=SQRT((BP(1,I)-BP(1,J))**2+(BP(2,I)-BP(2,J))**2+(BP(3,I)-BP(3,J))**2)
            MIND=MIN(MIND,R)
         ENDIF
      ENDDO; ENDDO
! test
!     WRITE(*,*) 'MIND=',MIND
! test
      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI,GRIDC%COMM)
         NT=T_INFO%ITYP(NI)
         IF (NIP==0) CYCLE ion
         
         XX=-COS(PI/REAL(SPL0(NI)%NGEN+1,KIND=q))
         R0=1E-5_q-RM*SQRT((1+XX)/(1-XX))

         ! atomic position in cartesian coordinates
         POSION(1)=T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         POSION(2)=T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         POSION(3)=T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)

         SUM2=0
         DO I=1,SPL0(NI)%NP
            XX=-COS(REAL(I,KIND=q)*PI/REAL(SPL0(NI)%NGEN+1,KIND=q))
! test
!           R=RM*SQRT((1+XX)/(1-XX))
            R=RM*SQRT((1+XX)/(1-XX))+R0
! test

            RR(I,NI)=R

            IF (R>SPL0(NI)%RMAX) CYCLE

! test
!           W=PI/REAL(NR+1,KIND=q)*(SIN(PI*REAL(I,KIND=q)/REAL(NR+1,KIND=q)))**2
            W=PI/REAL(SPL0(NI)%NGEN+1,KIND=q)*SIN(PI*REAL(I,KIND=q)/REAL(SPL0(NI)%NGEN+1,KIND=q))
! test

            AREA=4*PI*R**2/(ANGPARAM**2)
            ANGNUM=0
            DO J=1,5
               IF (ANGNUM==0.AND.AREA<REAL(NAng(J),KIND=q)) ANGNUM=J
            ENDDO
            IF (ANGNUM==0) ANGNUM=5

            ! Get the weight of atom NI at radius R
! test
!           CALL SPLVAL(R,DQ,DQD,SPL0(NT)%SPL,SPL0(NT)%NP,SIZE(SPL0(NT)%SPL,1))
            CALL SPLVL (R,DQ,DQD,SPL0(NI)%SPL,SPL0(NI)%NGEN,SIZE(SPL0(NI)%SPL,1))
! test

            SUM1=0
            DO J=1,NAng(ANGNUM)
               TMP(1)=POSION(1)+R*AngularGrids(1,AngularGridOffset(ANGNUM)+J)
               TMP(2)=POSION(2)+R*AngularGrids(2,AngularGridOffset(ANGNUM)+J)
               TMP(3)=POSION(3)+R*AngularGrids(3,AngularGridOffset(ANGNUM)+J)
               ! In direct coordinates
               POS(1)=TMP(1)*LATT_CUR%B(1,1)+TMP(2)*LATT_CUR%B(2,1)+TMP(3)*LATT_CUR%B(3,1)
               POS(2)=TMP(1)*LATT_CUR%B(1,2)+TMP(2)*LATT_CUR%B(2,2)+TMP(3)*LATT_CUR%B(3,2)
               POS(3)=TMP(1)*LATT_CUR%B(1,3)+TMP(2)*LATT_CUR%B(2,3)+TMP(3)*LATT_CUR%B(3,3)
               ! Check whether this point belongs to the basin of atom NI
!              IF (.NOT.MYBASIN(POS,NI,T_INFO,LATT_CUR)) CYCLE
               ! Get the SCF charge density at POS
               CALL INTERPOLATE_CHARGE(POS,LATT_CUR,GRIDC,CHGLOBAL(1,1,1),RHO)
               ! Compute the sum of the weights at POS
               CALL SUMWEIGHTS(TMP,T_INFO,LATT_CUR,MIND,SPL0,SUMW)

! test
!              IF (RHO<0._q) CYCLE
               IF (RHO<0._q) THEN
                  WRITE(*,*) 'RHO=',RHO
                  CYCLE
               ENDIF
!              IF (I==SPL0(NI)%NP) WRITE(*,*) 'SUMW=',SUMW,R
! test

! test
               IF (ABS(SUMW)>TINY) &
                  SUM1=SUM1+RHO*AngularGrids(4,AngularGridOffset(ANGNUM)+J)*DQ/SUMW
! test
            ENDDO
! test
!           RADWEIGHT=W*(1-XX)**(-3)*RM**3
            RADWEIGHT=(RM**3)*R*R/(1-XX)/(1-XX)/SQRT((1+XX)/(1-XX))*W
! test
            SUM2=SUM2+SUM1*RADWEIGHT*4*PI/LATT_CUR%OMEGA
            FR(I,NI)=SUM1/LATT_CUR%OMEGA
         ENDDO
         Z(NI)=SUM2
!        WRITE(*,*) NI,' Z=',SUM2
      ENDDO ion

      ! Communicate
      CALLMPI( M_sum_d(GRIDC%COMM, FR, T_INFO%NIONS*MAXVAL(SPL0(:)%NP)) )
      CALLMPI( M_sum_d(GRIDC%COMM, RR, T_INFO%NIONS*MAXVAL(SPL0(:)%NP)) )
      CALLMPI( M_sum_d(GRIDC%COMM,  Z, T_INFO%NIONS) )

      ! Copy to SPL1 and compute spline coefficients
      DO NI=1,T_INFO%NIONS
         SPL1(NI)%SPL=0
         SPL1(NI)%NP=SPL0(NI)%NP; NR=SPL0(NI)%NP
         SPL1(NI)%SPL(1:NR,1)=RR(1:NR,NI)
         SPL1(NI)%SPL(1:NR,2)=FR(1:NR,NI)
         CALL SPLCOF(SPL1(NI)%SPL,SPL1(NI)%NP,SIZE(SPL1(NI)%SPL,1),0._q)
      ENDDO

      ! Average over atoms of like species
      NIS=1
      DO NT=1,T_INFO%NTYP 
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            ZNT(NT)=ZNT(NT)+Z(NI)
         ENDDO
         ZNT(NT)=ZNT(NT)/T_INFO%NITYP(NT)
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(210)
      DO NI=1,T_INFO%NIONS
         DO I=1,SPL1(NI)%NP
            WRITE(210,'(2F14.7)') SPL1(NI)%SPL(I,1),SPL1(NI)%SPL(I,2)
         ENDDO
         WRITE(210,*)
      ENDDO
      CLOSE(210)
!     DO NI=1,T_INFO%NIONS
!        WRITE(*,'(I4,X,A,F14.7)') NI,'Z=',Z(NI)
!     ENDDO
      DO NT=1,T_INFO%NTYP
         WRITE(*,'(I4,X,A,F14.7)') NT,'Z=',ZNT(NT)
      ENDDO
      ENDIF
!#endif

      DEALLOCATE(FR,RR,Z,ZNT)

      RETURN
      END SUBROUTINE AVERAGE_RL2


      SUBROUTINE RADINT(SPLF,RINT)
      USE prec
      USE spline
      USE constant
      IMPLICIT NONE
      TYPE (splfnc) SPLF
      REAL(q) RINT
      ! local variables
      INTEGER I
      REAL(q) R0,XX,R,W,RADWEIGHT,RHO,DQD
      INTEGER, PARAMETER :: NR=1000
      REAL(q), PARAMETER :: RM=1._q

!     NR=SPLF%NP

      XX=-COS(PI/REAL(NR+1,KIND=q))
      R0=1E-5_q-RM*SQRT((1+XX)/(1-XX))

      RINT=0
      DO I=1,NR
         XX=-COS(REAL(I,KIND=q)*PI/REAL(NR+1,KIND=q))
         R=RM*SQRT((1+XX)/(1-XX))+R0
         IF (R>SPLF%RMAX) EXIT
         W=PI/REAL(NR+1,KIND=q)*SIN(PI*REAL(I,KIND=q)/REAL(NR+1,KIND=q))
         RADWEIGHT=(RM**3)*R*R/(1-XX)/(1-XX)/SQRT((1+XX)/(1-XX))*W
         CALL SPLVL (R,RHO,DQD,SPLF%SPL,SPLF%NGEN,SIZE(SPLF%SPL,1))
         RINT=RINT+RHO*RADWEIGHT*4*PI
     ENDDO

      RETURN
      END SUBROUTINE RADINT


      SUBROUTINE INTERPOLATE_CHARGE(R,LATT_CUR,GRIDC,CHGLOBAL,RHO)
      USE prec
      USE mgrid
      USE lattice
      IMPLICIT NONE
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      REAL(q) R(3),RHO
      REAL(q) CHGLOBAL(GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ)
      ! local variables
      INTEGER X0,Y0,Z0
      INTEGER I,I1,I2,I3,N1,N2,N3,X,Y,Z
      INTEGER, PARAMETER :: N=4,DLO=1,DHI=2
      REAL(q) DRHO
      REAL(q) XA(N),YA(N),ZA(N),CHG(N,N,N)

      X0=INT(FLOOR(R(1)*GRIDC%NGX))
      Y0=INT(FLOOR(R(2)*GRIDC%NGY))
      Z0=INT(FLOOR(R(3)*GRIDC%NGZ))

      DO I=1,N
         XA(I)=REAL(X0+I-1-DLO,KIND=q)/REAL(GRIDC%NGX,KIND=q)
         YA(I)=REAL(Y0+I-1-DLO,KIND=q)/REAL(GRIDC%NGY,KIND=q)
         ZA(I)=REAL(Z0+I-1-DLO,KIND=q)/REAL(GRIDC%NGZ,KIND=q)
      ENDDO

      Z=0
      DO I3=Z0-DLO,Z0+DHI
      N3=MODULO(I3,GRIDC%NGZ)+1; Z=Z+1

      Y=0
      DO I2=Y0-DLO,Y0+DHI
      N2=MODULO(I2,GRIDC%NGY)+1; Y=Y+1

      X=0
      DO I1=X0-DLO,X0+DHI
      N1=MODULO(I1,GRIDC%NGX)+1; X=X+1

         CHG(X,Y,Z)=CHGLOBAL(N1,N2,N3)

      ENDDO
      ENDDO
      ENDDO

      CALL POLIN3(XA,YA,ZA,CHG,N,N,N,R(1),R(2),R(3),RHO,DRHO)

      RETURN
      END SUBROUTINE INTERPOLATE_CHARGE


      SUBROUTINE SUMWEIGHTS(POS,T_INFO,LATT_CUR,MIND,SPL,SUMW)
      USE prec
      USE spline
      USE poscar
      USE lattice
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (splfnc) SPL(T_INFO%NIONS)
      REAL(q) POS(3),MIND,SUMW
      ! local variables
      INTEGER I,J,K
      INTEGER N,NT,NI,NIS
      REAL(q) DQ,DQD
      REAL(q) POSION(3),R(3),RNORM

      REAL(q), PARAMETER :: TINY=1E-5_q

      SUMW=0
      NIS=1
      DO NT=1,T_INFO%NTYP
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            N=INT(2*SPL(NI)%RMAX/MIND); N=MAX(N,1)
            ! atomic position in cartesian coordinates
            POSION(1)=T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
           &             T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
           &                T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
            POSION(2)=T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
           &             T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
           &                T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
            POSION(3)=T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
           &             T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
           &                T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
            ! apply periodic boundary conditions and find out if the atom is
            ! within a sphere with radius RMAX of position POS
            DO I=-N,N; DO J=-N,N; DO K=-N,N
               R(:)=POSION(:)+I*LATT_CUR%A(:,1)+J*LATT_CUR%A(:,2)+K*LATT_CUR%A(:,3)  
               R(:)=R(:)-POS(:)
               RNORM=SQRT(R(1)*R(1)+R(2)*R(2)+R(3)*R(3))
!              IF (RNORM>SPL(NT)%RMAX.OR.RNORM<TINY) CYCLE
!              IF (RNORM>SPL(NI)%RMAX) CYCLE
               IF ((RNORM-SPL(NI)%RMAX)>TINY) CYCLE
               ! if yes, this atom contributes to SUMW
! test
!              CALL SPLVAL(RNORM,DQ,DQD,SPL(NT)%SPL,SPL(NT)%NP,SIZE(SPL(NT)%SPL,1))
               CALL SPLVL (RNORM,DQ,DQD,SPL(NI)%SPL,SPL(NI)%NGEN,SIZE(SPL(NI)%SPL,1))
! test
!              IF (DQ<0) DQ=0
               SUMW=SUMW+DQ
            ENDDO; ENDDO; ENDDO
         ENDDO
         NIS=NIS+T_INFO%NITYP(NT)
      ENDDO
      RETURN
      END SUBROUTINE SUMWEIGHTS


      FUNCTION MYBASIN(POS,NIP,T_INFO,LATT_CUR)
      USE prec
      USE poscar
      USE lattice
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      INTEGER NIP
      REAL(q) POS(3)
      LOGICAL MYBASIN
      ! local variables
      INTEGER I,J,K,NI
      REAL(q) POSION(3),R(3),RNORM
      REAL(q), PARAMETER :: RMIN=0.5_q

      MYBASIN=.TRUE.
      ion: DO NI=1,T_INFO%NIONS
         POSION(1)=T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         POSION(2)=T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         POSION(3)=T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
        &             T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
        &                T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         DO I=-1,1; DO J=-1,1; DO K=-1,1
            IF (NI==NIP.AND.I==0.AND.J==0.AND.K==0) CYCLE
            R(:)=POSION(:)+I*LATT_CUR%A(:,1)+J*LATT_CUR%A(:,2)+K*LATT_CUR%A(:,3)  
            R(:)=R(:)-POS(:)
            RNORM=SQRT(R(1)*R(1)+R(2)*R(2)+R(3)*R(3))
            IF (RNORM<RMIN) THEN
               MYBASIN=.FALSE.
               EXIT ion
            ENDIF
         ENDDO; ENDDO; ENDDO
      ENDDO ion

      END FUNCTION MYBASIN


      SUBROUTINE SPLVL(X,F,FDER,P,NAC,NDIM)
      USE prec
      USE constant
      IMPLICIT NONE
      INTEGER NAC,NDIM
      REAL(q) X,F,FDER
      REAL(q) P(NDIM,5)
      ! local variables
      INTEGER K
      REAL(q) DX
      K=MAX(INT(NAC+1-REAL(NAC+1,KIND=q)/PI*ACOS((X*X-1._q)/(X*X+1._q))),1)
      DX=X-P(K,1); DX=MAX(DX,0._q)
      F=((P(K,5)*DX+P(K,4))*DX+P(K,3))*DX+P(K,2)
! test
!     IF (K>NDIM) WRITE(*,*) 'SPLVL error',K,NDIM
!     IF (K>1.AND.K<NDIM) THEN
!        F=(P(K+1,2)-P(K,2))/(P(K+1,1)-P(K,1))*DX+P(K,2)
!     ELSE
!        F=P(K,2)
!     ENDIF
! test
      FDER=(3.0_q*P(K,5)*DX+2.0_q*P(K,4))*DX+P(K,3)
      END SUBROUTINE SPLVL


      subroutine PrepareAngularGrids
      use prec
      implicit none
      real(q) work(4,1000)
      integer N,offset

      if (.not. allocated(AngularGrids)) then

      allocate(AngularGrids(4,214))
      allocate(AngularGridOffset(5))
      allocate(NAng(5))
      AngularGridOffset(1)=0
      AngularGridOffset(2)=14
      AngularGridOffset(3)=40
      AngularGridOffset(4)=78
      AngularGridOffset(5)=128
      NAng(1)=14
      NAng(2)=26
      NAng(3)=38
      NAng(4)=50
      NAng(5)=86

      offset=AngularGridOffset(1)
      call LD0014(work(1,:),work(2,:),work(3,:), work(4,:),N)          
      AngularGrids(:,offset+1:offset+N)=work(:,1:N)

      offset=AngularGridOffset(2)
      call LD0026(work(1,:),work(2,:),work(3,:), work(4,:),N)          
      AngularGrids(:,offset+1:offset+N)=work(:,1:N)

      offset=AngularGridOffset(3)
      call LD0038(work(1,:),work(2,:),work(3,:), work(4,:),N)          
      AngularGrids(:,offset+1:offset+N)=work(:,1:N)

      offset=AngularGridOffset(4)
      call LD0050(work(1,:),work(2,:),work(3,:), work(4,:),N)          
      AngularGrids(:,offset+1:offset+N)=work(:,1:N)

      offset=AngularGridOffset(5)
      call LD0086(work(1,:),work(2,:),work(3,:), work(4,:),N)          
      AngularGrids(:,offset+1:offset+N)=work(:,1:N)

      endif
      end subroutine PrepareAngularGrids


      SUBROUTINE POLIN3(XA,YA,ZA,FA,NX,NY,NZ,X,Y,Z,F,DF)
      USE prec
      IMPLICIT NONE
      INTEGER                    I,J
      INTEGER                    NX,NY,NZ,NXMAX,NYMAX,NZMAX
      REAL(q)                    X,Y,Z
      REAL(q)                    F,DF
      REAL(q)                    XA(NX),YA(NY),ZA(NZ),FA(NX,NY,NZ)
      PARAMETER(NXMAX=10,NYMAX=10,NZMAX=10)
      REAL(q)                    FXTMP(NXMAX),FYTMP(NYMAX),FZTMP(NZMAX)
      
      DO I=1,NZ
         DO J=1,NY
            FXTMP(1:4)=FA(1:4,J,I)
            CALL POLINT(XA,FXTMP(1:NX),NX,X,FYTMP(J),DF)
         ENDDO
         CALL POLINT(YA,FYTMP(1:NY),NY,Y,FZTMP(I),DF)
      ENDDO
      CALL POLINT(ZA,FZTMP(1:NZ),NZ,Z,F,DF)

      RETURN
      END SUBROUTINE POLIN3


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
! MM. When closest table entry is known to be XA(2)
!     NS=2
!     DIF=ABS(X-XA(2))
!     C=YA
!     D=YA
! MM.
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

      END MODULE stockholder


!******************** SUBROUTINE RHOADD_RL_BESSEL **********************
!
!***********************************************************************
      SUBROUTINE RHOADD_RL_BESSEL(T_INFO,LATT_CUR,P,GRIDC,DHTOT)
      USE prec
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,DQ,QN,RHOT,QSM1
      INTEGER NTOT(T_INFO%NIONS)
      REAL(q) QTOT(T_INFO%NIONS)
      REAL(q) QSPH(T_INFO%NIONS)

      INTEGER, PARAMETER :: NQ=2
      REAL(q) QQ(NQ),A(NQ),BSUM

      REAL(q), PARAMETER :: TINY = 1.E-5_q

! To enforce strict charge neutrality
      DQ=0; QN=0
      DO NT=1,T_INFO%NTYP
         DQ=DQ+T_INFO%NITYP(NT)*T_INFO%ZCT(NT)
         QN=QN+T_INFO%NITYP(NT)*ABS(T_INFO%ZCT(NT))
      ENDDO

      ! quick return if possible
      IF (QN<TINY) RETURN

      T_INFO%ZCT(:)=T_INFO%ZCT(:)-DQ*ABS(T_INFO%ZCT(:))/QN

      NIS=1

      type1: DO NT=1,T_INFO%NTYP

      CALL AUG_SETQ(0,P(NT)%R,T_INFO%RWIGS(NT),QQ,A,.FALSE.)
! test
      WRITE(*,*) 'Q=',T_INFO%ZCT
      WRITE(*,*) QQ
      WRITE(*,*) A
!     OPEN(200+NT)
!     DO IND=0,100
!        D= REAL(IND,KIND=q)/100._q*T_INFO%RWIGS(NT)
!        BSUM=0
!        DO I=1,NQ
!           QR=QQ(I)*D
!           CALL SBESSEL(QR,BJ,0)
!           BSUM=BSUM+BJ*A(I) 
!        ENDDO
!        WRITE(200+NT,'(2F20.10)') D,BSUM
!     ENDDO
!     CLOSE(200+NT)
! test

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         IND=IND+1
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         ! sum Bessel-functions in sphere
         QSUM=QSUM+BSUM!*PI*PI/3._q
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         IND=IND+1
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         ! sum Bessel-functions in sphere
         QSUM=QSUM+BSUM!*PI*PI/3._q
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM
      NTOT(NI)=IND

      QSPH(NI)=QSM1

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, T_INFO%NIONS))
!#ifdef debug
      DO NI=1,T_INFO%NIONS
!        WRITE(*,'(I6,I8,F14.7)') NI,NTOT(NI),QTOT(NI)/NTOT(NI)!*LATT_CUR%OMEGA*F1*F2*F3
         WRITE(*,'(I6,2F14.7)') NI,QTOT(NI)/GRIDC%NPLWV*LATT_CUR%OMEGA/4._q/PI,QSPH(NI)/GRIDC%NPLWV

      ENDDO
!     STOP
!#endif

      NIS=1

      type2: DO NT=1,T_INFO%NTYP

      CALL AUG_SETQ(0,P(NT)%R,T_INFO%RWIGS(NT),QQ,A,.FALSE.)

      ions2: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         QSUM=QSUM+BSUM/QTOT(NI)
         DHTOT(IND)=DHTOT(IND)+BSUM/QTOT(NI)*T_INFO%ZCT(NT)*GRIDC%NPLWV
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         QSUM=QSUM+BSUM/QTOT(NI)
         DHTOT(IND)=DHTOT(IND)+BSUM/QTOT(NI)*T_INFO%ZCT(NT)*GRIDC%NPLWV
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM

      ENDDO ions2
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type2

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
#ifdef debug
      RHOT=0
      DO IND=1,GRIDC%RL%NP
         RHOT=RHOT+DHTOT(IND)
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RHOT, 1))

      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I6,I8,F14.7)') NI,NTOT(NI),QTOT(NI)!*LATT_CUR%OMEGA*F1*F2*F3
      ENDDO
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT/GRIDC%NPLWV
#endif
      RETURN
      END SUBROUTINE RHOADD_RL_BESSEL


!******************** SUBROUTINE RHOADD_RL_GAUSSIAN ********************
!
!***********************************************************************
      SUBROUTINE RHOADD_RL_GAUSSIAN(T_INFO,LATT_CUR,P,GRIDC,DHTOT)
      USE prec
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,DQ,QN,RHOT,QSM1
      REAL(q) A,SIGMA,FAC
      INTEGER NTOT(T_INFO%NIONS)
      REAL(q) QTOT(T_INFO%NIONS)
      REAL(q) QSPH(T_INFO%NIONS)

      INTEGER, PARAMETER :: NQ=2

      REAL(q), PARAMETER :: TINY = 1.E-5_q

! To enforce strict charge neutrality
      DQ=0; QN=0
      DO NT=1,T_INFO%NTYP
         DQ=DQ+T_INFO%NITYP(NT)*T_INFO%ZCT(NT)
         QN=QN+T_INFO%NITYP(NT)*ABS(T_INFO%ZCT(NT))
      ENDDO

      ! quick return if possible
      IF (QN<TINY) RETURN

      T_INFO%ZCT(:)=T_INFO%ZCT(:)-DQ*ABS(T_INFO%ZCT(:))/QN

      NIS=1

      type1: DO NT=1,T_INFO%NTYP

      SIGMA=T_INFO%RWIGS(NT)/5._q
      A=T_INFO%ZCT(NT)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
! test
      WRITE(*,*) 'Q=',T_INFO%ZCT
      WRITE(*,*) A,SIGMA
! test

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         IND=IND+1
         ! sum Gaussian function in sphere
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         IND=IND+1
         ! sum Gaussian function in sphere
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM
      NTOT(NI)=IND

      QSPH(NI)=QSM1

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, T_INFO%NIONS))
!#ifdef debug
      DO NI=1,T_INFO%NIONS
!        WRITE(*,'(I6,I8,F14.7)') NI,NTOT(NI),QTOT(NI)/NTOT(NI)!*LATT_CUR%OMEGA*F1*F2*F3
!        WRITE(*,'(I6,2F14.7)') NI,QTOT(NI)/GRIDC%NPLWV*LATT_CUR%OMEGA/4._q/PI,QSPH(NI)/GRIDC%NPLWV
         WRITE(*,'(I6,I10,2F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV,QSPH(NI)/GRIDC%NPLWV
      ENDDO
!     STOP
!#endif

      NIS=1

      type2: DO NT=1,T_INFO%NTYP

      SIGMA=T_INFO%RWIGS(NT)/5._q
      A=T_INFO%ZCT(NT)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA

      ions2: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      FAC=T_INFO%ZCT(NT)*GRIDC%NPLWV/QTOT(NI)
! test
      WRITE(*,*) NI,' FAC=',FAC
! test
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA*FAC
         ! add Gaussian function in sphere
         DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA*FAC
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA*FAC
         ! add Gaussian function in sphere
         DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA*FAC
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM

      ENDDO ions2
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type2

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
!#ifdef debug
      RHOT=0
      DO IND=1,GRIDC%RL%NP
         RHOT=RHOT+DHTOT(IND)
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RHOT, 1))

      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I6,I8,F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV
      ENDDO
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT/GRIDC%NPLWV
!#endif
      RETURN
      END SUBROUTINE RHOADD_RL_GAUSSIAN



!******************** SUBROUTINE RHOADD_RL_GAUSSIANS *******************
!
!***********************************************************************
      SUBROUTINE RHOADD_RL_GAUSSIANS(T_INFO,LATT_CUR,P,GRIDC,DHTOT)
      USE prec
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      USE chgfit
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,DQ,QN,RHOT,QSM1
      REAL(q) A,SIGMA,FAC,ZCORR,RWIGS
      INTEGER NTOT(T_INFO%NIONS)
      REAL(q) QTOT(T_INFO%NIONS)
      REAL(q) QSPH(T_INFO%NIONS)

      INTEGER NQ

      NQ=NGAUS
      NIS=1

      type1: DO NT=1,T_INFO%NTYP

!     SIGMA=MAXVAL(RGAUS(2*NT-1:2*NT+NQ-2))
      SIGMA=MAXVAL(RGAUS(NQ*(NT-1)+1:NQ*NT))
      RWIGS=5._q*SIGMA

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         ! sum Gaussian function in sphere
         DO I=1,NQ
!           SIGMA=RGAUS(2*NT+I-2)
!           A=ZCT(2*NT+I-2)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA 
            SIGMA=RGAUS(NQ*(NT-1)+I)
            A=ZCT(NQ*(NT-1)+I)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ENDDO
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         ! sum Gaussian function in sphere
         DO I=1,NQ
!           SIGMA=RGAUS(2*NT+I-2)
!           A=ZCT(2*NT+I-2)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            SIGMA=RGAUS(NQ*(NT-1)+I)
            A=ZCT(NQ*(NT-1)+I)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ENDDO
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM
      NTOT(NI)=IND

      QSPH(NI)=QSM1

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, T_INFO%NIONS))
#ifdef debug
      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I6,I10,2F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV,QSPH(NI)/GRIDC%NPLWV
      ENDDO
!     STOP
#endif

      NIS=1

      type2: DO NT=1,T_INFO%NTYP

!     SIGMA=MAXVAL(RGAUS(2*NT-1:2*NT+NQ-1))
      SIGMA=MAXVAL(RGAUS(NQ*(NT-1)+1:NQ*NT))
      RWIGS=5._q*SIGMA

      QN=0._q
      DO I=1,NQ
!        QN=QN+ZCT(2*NT+I-2)
         QN=QN+ZCT(NQ*(NT-1)+I)
      ENDDO

      ions2: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      ZCORR=QTOT(NI)/GRIDC%NPLWV-QN
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         ! add Gaussian function in sphere
         DO I=1,NQ
!           SIGMA=RGAUS(2*NT+I-2)
!           A=ZCT(2*NT+I-2)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            SIGMA=RGAUS(NQ*(NT-1)+I)
            A=ZCT(NQ*(NT-1)+I)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
            DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA-ZCORR/NTOT(NI)*GRIDC%NPLWV/NGAUS
         ENDDO
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         ! add Gaussian function in sphere
         DO I=1,NQ
!           SIGMA=RGAUS(2*NT+I-2)
!           A=ZCT(2*NT+I-2)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            SIGMA=RGAUS(NQ*(NT-1)+I)
            A=ZCT(NQ*(NT-1)+I)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
            QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
            DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA-ZCORR/NTOT(NI)*GRIDC%NPLWV/NGAUS
         ENDDO
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM

      ENDDO ions2
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type2

!     CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
#ifdef debug
      RHOT=0
      DO IND=1,GRIDC%RL%NP
         RHOT=RHOT+DHTOT(IND)
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RHOT, 1))

      DO NI=1,T_INFO%NIONS
         WRITE(*,'(I6,I8,F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV
      ENDDO
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT/GRIDC%NPLWV
#endif
      RETURN
      END SUBROUTINE RHOADD_RL_GAUSSIANS


!******************** SUBROUTINE RHOADD_RL_GAUSSIANS_LIST **************
!
!***********************************************************************
      SUBROUTINE RHOADD_RL_GAUSSIANS_LIST( &
     &   Num_Gaus,Z_Gaus,Sigma_Gaus,Pos_Gaus,LATT_CUR,GRIDC,DHTOT)
      USE prec
      USE constant
      USE lattice
      USE mgrid
      USE mpimy
      USE radial
      USE chgfit
      IMPLICIT NONE
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      INTEGER Num_Gaus
      REAL(q) Z_Gaus(Num_Gaus),Sigma_Gaus(Num_Gaus),Pos_Gaus(3,Num_Gaus)
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I,NG
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,DQ,RHOT,QSM1
      REAL(q) A,SIGMA,FAC,ZCORR,RWIGS
      INTEGER NTOT(Num_Gaus)
      REAL(q) QTOT(Num_Gaus)
      REAL(q) QSPH(Num_Gaus)

      F1=1._q/GRIDC%NGX; F2=1._q/GRIDC%NGY; F3=1._q/GRIDC%NGZ

      gaussians1: DO NG=1,Num_Gaus

      SIGMA=Sigma_Gaus(NG)
      RWIGS=5._q*SIGMA

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(Pos_Gaus(3,NG)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(Pos_Gaus(2,NG)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(Pos_Gaus(1,NG)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(Pos_Gaus(3,NG)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(Pos_Gaus(2,NG)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(Pos_Gaus(1,NG)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-Pos_Gaus(2,NG))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-Pos_Gaus(1,NG))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-Pos_Gaus(3,NG))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         SIGMA=Sigma_Gaus(NG)
         A=Z_Gaus(NG)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-Pos_Gaus(3,NG))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-Pos_Gaus(2,NG))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-Pos_Gaus(1,NG))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         SIGMA=Sigma_Gaus(NG)
         A=Z_Gaus(NG)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NG)=QSUM
      NTOT(NG)=IND

      QSPH(NG)=QSM1

      ENDDO gaussians1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, Num_Gaus))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, Num_Gaus))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, Num_Gaus))
#ifdef debug
      DO NG=1,Num_Gaus
         WRITE(*,'(I6,I10,2F14.7)') NG,NTOT(NG),QTOT(NG)/GRIDC%NPLWV,QSPH(NG)/GRIDC%NPLWV
      ENDDO
!     STOP
#endif

      gaussians2: DO NG=1,Num_Gaus

      SIGMA=Sigma_Gaus(NG)
      RWIGS=5._q*SIGMA

      ZCORR=Z_Gaus(NG)/(QTOT(NG)/GRIDC%NPLWV)
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(Pos_Gaus(3,NG)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(Pos_Gaus(2,NG)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(Pos_Gaus(1,NG)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(Pos_Gaus(3,NG)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(Pos_Gaus(2,NG)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(Pos_Gaus(1,NG)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-Pos_Gaus(2,NG))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-Pos_Gaus(1,NG))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-Pos_Gaus(3,NG))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         ! add Gaussian function in sphere
         SIGMA=Sigma_Gaus(NG)
         A=Z_Gaus(NG)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA*ZCORR
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-Pos_Gaus(3,NG))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-Pos_Gaus(2,NG))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-Pos_Gaus(1,NG))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         ! add Gaussian function in sphere
         SIGMA=Sigma_Gaus(NG)
         A=Z_Gaus(NG)/SQRT(8*PI*PI*PI)/SIGMA/SIGMA/SIGMA*ZCORR
         QSUM=QSUM+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
         DHTOT(IND)=DHTOT(IND)+A*EXP(-D*D/SIGMA/SIGMA/2._q)*LATT_CUR%OMEGA
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NG)=QSUM

      ENDDO gaussians2

      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, Num_Gaus))
!#ifdef debug
      RHOT=0
      DO IND=1,GRIDC%RL%NP
         RHOT=RHOT+DHTOT(IND)
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RHOT, 1))

      DO NG=1,Num_Gaus
         WRITE(*,'(I6,3F14.7,I8,F14.7)') NG,Pos_Gaus(1:3,NG),NTOT(NG),QTOT(NG)/GRIDC%NPLWV
      ENDDO
      WRITE(*,'(A,F20.10)') 'Total number of electrons =',RHOT/GRIDC%NPLWV
!#endif
      RETURN
      END SUBROUTINE RHOADD_RL_GAUSSIANS_LIST



!***********************************************************************
!  add Gaussian charged in reciprocal space
!***********************************************************************


      SUBROUTINE RHOADD_RC_GAUSSIANS(T_INFO,LATT_CUR,P,GRIDC,CHTOT,CSTRF)
      USE prec
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE chgfit

      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      COMPLEX(q)   CHTOT(GRIDC%RC%NP)
      COMPLEX(q)   CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

! local variables
      INTEGER :: NT, N, N1, NC, N2, N3, I, NQ
      REAL(q) :: GX, GY, GZ, G2, A, EVF, SIGMA

      NQ=NGAUS
!=======================================================================
! loop over all types of atoms
! if no pseudocharge for this ion next ion
!=======================================================================
      typ: DO NT=1,T_INFO%NTYP
!=======================================================================
! calculate the scaling factor ARGSC that converts the magnitude of a
! reciprocal lattice vector to the correponding position in the
! pseudopotential arrays
!=======================================================================

      DO I=1,NQ
       SIGMA=RGAUS(NQ*(NT-1)+I)
       A=ZCT(NQ*(NT-1)+I)
       EVF=-0.5_q*SIGMA*SIGMA

       DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G2=(GX**2+GY**2+GZ**2)*(TPI*TPI)

!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
        CHTOT(N)=CHTOT(N)+CSTRF(N,NT)*A*EXP(EVF*G2)
       ENDDO
      ENDDO
      ENDDO typ

      RETURN
      END SUBROUTINE


!************************ SUBROUTINE FORHAR_GAUSSIAN *******************
!
! this subroutine calculates the correction to the
! hellman feynman forces for moving Gaussian charges
!
!***********************************************************************


      SUBROUTINE FORHAR_GAUSSIAN(GRIDC,P,T_INFO,LATT_CUR,CHGGA,HARFOR)
      USE prec

      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      USE chgfit
      IMPLICIT NONE

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      COMPLEX(q)   CHGGA(GRIDC%MPLWV)
      REAL(q)      HARFOR(3,T_INFO%NIONS)
! work arrays
      REAL(q), ALLOCATABLE :: WORK(:)
      REAL(q) :: HARFOR_LOCAL(3,T_INFO%NIONS)
      INTEGER :: NIS, NT, NIADD, N1, NC, N2, N3, I, N, NQ, N1P, NI, NGP, NG
      REAL(q) :: G2, A, EVF, SIGMA, GX, GY, GZ, FOR, FOR1, FOR2, FOR3, G1, G3, FACTM
      COMPLEX(q) :: CX, CEXPF, CE

      ! quick return if possible
      IF (NGAUS==0) RETURN

      ALLOCATE(WORK(GRIDC%RC%NP))

      NQ=NGAUS
      HARFOR_LOCAL=0
!=======================================================================
! loop over all types of atoms
!=======================================================================
      NIS=1
      typ: DO NT=1,T_INFO%NTYP
      NIADD=T_INFO%NITYP(NT)

gaus: DO I=1,NQ
       SIGMA=RGAUS(NQ*(NT-1)+I)
       A=ZCT(NQ*(NT-1)+I)
       EVF=-0.5_q*SIGMA*SIGMA
!=======================================================================
! interpolate the pseudopotential on the grid of reciprocal
! lattice-vectors
!=======================================================================

       DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G2=(GX**2+GY**2+GZ**2)*(TPI*TPI)
!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
        WORK(N)=A*EXP(EVF*G2)
      ENDDO

      ion: DO NI=NIS,NIADD+NIS-1
!=======================================================================
! initialise the force on the ion to zero
!=======================================================================
         FOR1=0._q
         FOR2=0._q
         FOR3=0._q
!=======================================================================
! calculate the total force on the ions by summing over reciprocal
! lattice vectors
! first calculate phase factor:
! there are two version for calculating the phase factor
! on vector machines you might try the first version
! but usually the second version is much faster (less calls to CEXP)
!=======================================================================
#ifdef vector
         DO NG=1,GRIDC%RC%NP

           N1= MOD((NG-1),GRIDC%RC%NROW) +1
           NC= (NG-1)/GRIDC%RC%NROW+1
           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)

           G1=T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           SETFACT1
           SETFACT
           CEXPF=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)
#else
         CX =EXP(-CITPI*T_INFO%POSION(1,NI))
         G1 =T_INFO%POSION(1,NI)*xmin(GRIDC%NGX)

         DO NC=1,GRIDC%RC%NCOL
           NGP=(NC-1)*GRIDC%RC%NROW+1

           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           CE=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)

           DO N1P=0,GRIDC%RC%NROW-1
              SETN1fromN1P
              NG=NGP+N1
              N1=N1+1
              SETFACT1
              SETFACT
              CEXPF=CE
              CE=CE*CX
#endif
!=======================================================================
! add the contribution to the force from the present reciprocal lattice
! vector  and multiply by i (ie take imaginary part)
!=======================================================================
              FOR=WORK(NG)*MULFACT AIMAG(CONJG(CHGGA(NG))*CEXPF)
              FOR1=FOR1-GRIDC%LPCTX_(N1)*FOR
              FOR2=FOR2-GRIDC%LPCTY_(N2)*FOR
              FOR3=FOR3-GRIDC%LPCTZ_(N3)*FOR
           ENDDO
#ifndef vector
         ENDDO
#endif
!=======================================================================
! multiply forces by 2*Pi
!=======================================================================
         HARFOR_LOCAL(1,NI)=HARFOR_LOCAL(1,NI)+FOR1*TPI
         HARFOR_LOCAL(2,NI)=HARFOR_LOCAL(2,NI)+FOR2*TPI
         HARFOR_LOCAL(3,NI)=HARFOR_LOCAL(3,NI)+FOR3*TPI
       ENDDO ion
       ENDDO gaus
       NIS=NIS+NIADD
      ENDDO typ
!=======================================================================
! forces are now in the reciprocal lattice transform it to
! cartesian coordinates
!=======================================================================
      CALLMPI( M_sum_d(GRIDC%COMM, HARFOR_LOCAL(1,1),T_INFO%NIONS*3))
      CALL  DIRKAR(T_INFO%NIONS,HARFOR_LOCAL,LATT_CUR%B)
      DEALLOCATE(WORK)

      HARFOR=HARFOR+HARFOR_LOCAL

      RETURN
      END SUBROUTINE


!******************** SUBROUTINE RHOADD_RL_SPLINE **********************
!
!***********************************************************************
      SUBROUTINE RHOADD_RL_SPLINE(T_INFO,LATT_CUR,SPLINES,NORMS,LNORM,GRIDC,DHTOT,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE radial
      USE spline
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (splfnc) SPLINES(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      REAL(q) NORMS(T_INFO%NTYP)
      LOGICAL LNORM
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QSUM,QN,DQ,DQD,RHOT,QSM1
      REAL(q) FAC,RWIGS
      INTEGER NTOT(T_INFO%NIONS)
      REAL(q) QTOT(T_INFO%NIONS)
      REAL(q) QSPH(T_INFO%NIONS)

      REAL(q), PARAMETER :: TINY=1.E-5_q

      F1=1._q/GRIDC%NGX; F2=1._q/GRIDC%NGY; F3=1._q/GRIDC%NGZ

      IF (LNORM) THEN

      NIS=1

      type1: DO NT=1,T_INFO%NTYP

      RWIGS=SPLINES(NT)%RMAX

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
!        CALL LINPOL(D,SPLINES(NT),DQ)
         CALL SPLVAL(D,DQ,DQD,SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1))
         QSUM=QSUM+DQ*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
!        CALL LINPOL(D,SPLINES(NT),DQ)
         CALL SPLVAL(D,DQ,DQD,SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1))
         QSUM=QSUM+DQ*LATT_CUR%OMEGA
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM
      NTOT(NI)=IND

      QSPH(NI)=QSM1

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, T_INFO%NIONS))
!#ifdef debug
!     DO NI=1,T_INFO%NIONS
!        WRITE(*,'(I6,I10,2F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV,QSPH(NI)/GRIDC%NPLWV
!     ENDDO
!     NI=1
!     WRITE(*,'(I6,I10,2F14.7)') NI,NTOT(NI),QTOT(NI)/GRIDC%NPLWV,QSPH(NI)/GRIDC%NPLWV
!     STOP
!#endif
      
      ENDIF


      NIS=1

      type2: DO NT=1,T_INFO%NTYP

      RWIGS=SPLINES(NT)%RMAX

      ions2: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      IF (LNORM.AND.ABS(NORMS(NT))>TINY) THEN
         FAC=NORMS(NT)*GRIDC%NPLWV/QTOT(NI)
      ELSE
         FAC=1._q
      ENDIF
! test
!     IF (NI==1) WRITE(*,*) 'NI=',NI,' FAC1=',FAC
! test
!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
!        CALL LINPOL(D,SPLINES(NT),DQ)
         CALL SPLVAL(D,DQ,DQD,SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1))
         QSUM=QSUM+DQ*LATT_CUR%OMEGA*FAC
         DHTOT(IND)=DHTOT(IND)+DQ*LATT_CUR%OMEGA*FAC 
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
!        CALL LINPOL(D,SPLINES(NT),DQ)
         CALL SPLVAL(D,DQ,DQD,SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1))
         QSUM=QSUM+DQ*LATT_CUR%OMEGA*FAC
         DHTOT(IND)=DHTOT(IND)+DQ*LATT_CUR%OMEGA*FAC
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM

      ENDDO ions2
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type2

!     CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
!#ifdef debug
      RHOT=0
      DO IND=1,GRIDC%RL%NP
         RHOT=RHOT+DHTOT(IND)
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RHOT, 1))

      NIS=1
      DO NT=1,T_INFO%NTYP
         QSUM=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            QSUM=QSUM+QTOT(NI)
         ENDDO
         NORMS(NT)=QSUM/T_INFO%NITYP(NT)/GRIDC%NPLWV
         NIS = NIS+T_INFO%NITYP(NT)
      ENDDO

      IF (IO%IU6>=0) THEN
!        DO NI=1,T_INFO%NIONS
!           WRITE(*,'(I6,F14.7)') NI,QTOT(NI)/GRIDC%NPLWV
!        ENDDO
         DO NT=1,T_INFO%NTYP
            WRITE(*,'(I6,F14.7)') NT,NORMS(NT)
         ENDDO
!        NI=1
!        WRITE(*,'(I6,F14.7)') NI,QTOT(NI)/GRIDC%NPLWV
         WRITE(*,'(A,F20.10)') 'RHOADD_RL_SPLINE: Total number of electrons =',RHOT/GRIDC%NPLWV
      ENDIF
!#endif
      RETURN
      END SUBROUTINE RHOADD_RL_SPLINE


!******************** SUBROUTINE ANALYSE_RL ****************************
!
!***********************************************************************
      SUBROUTINE ANALYSE_RL(T_INFO,LATT_CUR,P,GRIDC,DHTOT,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,DQ,QN,RHOT,QSM1
      INTEGER NTOT(T_INFO%NIONS)
      REAL(q) QTOT(T_INFO%NIONS)
      REAL(q) QSPH(T_INFO%NIONS)

      INTEGER IR
      INTEGER, PARAMETER :: NR=100
      REAL(q) RWIGS
      REAL(q) QRAD(T_INFO%NIONS,NR),QRD1(T_INFO%NTYP,NR),RNT(T_INFO%NTYP,NR)

      INTEGER, PARAMETER :: NQ=2
      REAL(q) QQ(NQ),A(NQ),BSUM

      REAL(q), PARAMETER :: TINY = 1.E-5_q

! To enforce strict charge neutrality
      DQ=0; QN=0
      DO NT=1,T_INFO%NTYP
         DQ=DQ+T_INFO%NITYP(NT)*T_INFO%ZCT(NT)
         QN=QN+T_INFO%NITYP(NT)*ABS(T_INFO%ZCT(NT))
      ENDDO

      ! quick return if possible
      IF (QN<TINY) RETURN

      T_INFO%ZCT(:)=T_INFO%ZCT(:)-DQ*ABS(T_INFO%ZCT(:))/QN

      radius: DO IR=1,NR

      NIS=1

      type1: DO NT=1,T_INFO%NTYP

      RWIGS=T_INFO%RWIGS(NT)*IR/NR
      RNT(NT,IR)=RWIGS

      CALL AUG_SETQ(0,P(NT)%R,RWIGS,QQ,A,.FALSE.)

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

      IND=0; QSUM=0; QSM1=0
!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         ! sum Bessel-functions in sphere
         QSUM=QSUM+BSUM!*PI*PI/3._q
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=RWIGS) THEN
         IND=IND+1
         BSUM=0
         DO I=1,NQ
            QR=QQ(I)*D
            CALL SBESSEL(QR,BJ,0)
            BSUM=BSUM+BJ*A(I) 
         ENDDO
         ! sum Bessel-functions in sphere
         QSUM=QSUM+BSUM!*PI*PI/3._q
         ! sum total charge in sphere
         QSM1=QSM1+DHTOT(IND1)
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      QTOT(NI)=QSUM
      NTOT(NI)=IND

      QSPH(NI)=QSM1

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QTOT, T_INFO%NIONS))
      CALLMPI( M_sum_d(GRIDC%COMM, QSPH, T_INFO%NIONS))

      QRAD(:,IR)=QSPH(:)

      ENDDO radius

      DO IR=1,NR
      NIS=1
      DO NT=1,T_INFO%NTYP
         QSUM=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            QSUM=QSUM+QRAD(NI,IR)
         ENDDO
         QRD1(NT,IR)=QSUM/T_INFO%NITYP(NT)/GRIDC%NPLWV
         NIS = NIS+T_INFO%NITYP(NT)
      ENDDO
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(200)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR
            WRITE(200,'(2F14.7)') RNT(NT,IR),QRD1(NT,IR)
         ENDDO
         WRITE(200,*)
      ENDDO
      CLOSE(200)
      ENDIF
!#endif

      RETURN
      END SUBROUTINE ANALYSE_RL


!******************** SUBROUTINE ANALYSE_RL2 ***************************
!
!***********************************************************************
      SUBROUTINE ANALYSE_RL2(T_INFO,LATT_CUR,P,GRIDC,DHTOT,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D
      REAL(q) QR,BJ,QSUM,RHOT,QSM1

      INTEGER IR
      INTEGER, PARAMETER :: NR=100
      REAL(q) QRAD(T_INFO%NIONS,NR),QRD1(T_INFO%NTYP,NR),RNT(T_INFO%NTYP,NR)

      REAL(q), PARAMETER :: TINY = 1.E-5_q

      QRAD=0; QRD1=0; RNT=0

      NIS=1
 
      type1: DO NT=1,T_INFO%NTYP

      DO IR=1,NR
         RNT(NT,IR)=T_INFO%RWIGS(NT)*(IR-1)/NR
      ENDDO

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= T_INFO%RWIGS(NT)*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         QRAD(NI,INT(D/T_INFO%RWIGS(NT)*NR)+1)=QRAD(NI,INT(D/T_INFO%RWIGS(NT)*NR)+1)+DHTOT(IND1)/4/PI/D/D
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D<=T_INFO%RWIGS(NT)) THEN
         QRAD(NI,INT(D/T_INFO%RWIGS(NT)*NR)+1)=QRAD(NI,INT(D/T_INFO%RWIGS(NT)*NR)+1)+DHTOT(IND1)/4/PI/D/D
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_d(GRIDC%COMM, QRAD, T_INFO%NIONS*NR))

      QRD1=0

      DO IR=1,NR
      NIS=1
      DO NT=1,T_INFO%NTYP
         QSUM=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            QSUM=QSUM+QRAD(NI,IR)
         ENDDO
         QRD1(NT,IR)=QSUM/T_INFO%NITYP(NT)/GRIDC%NPLWV
         NIS = NIS+T_INFO%NITYP(NT)
      ENDDO
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(200)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR
            WRITE(200,'(2F14.7)') RNT(NT,IR),QRD1(NT,IR)
         ENDDO
         WRITE(200,*)
      ENDDO
      CLOSE(200)
      ENDIF
!#endif

      RETURN
      END SUBROUTINE ANALYSE_RL2


!******************** SUBROUTINE ANALYSE_RL3 ***************************
!
!***********************************************************************
      SUBROUTINE ANALYSE_RL3(T_INFO,LATT_CUR,P,GRIDC,DHTOT,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      USE spline
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (potcar) P(T_INFO%NTYP)
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV))
      ! local variables
      TYPE (splfnc) SPLINES(T_INFO%NTYP)
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D,RWIGS
      REAL(q) QSUM,QSM1,DQ,DQD,FAC

      INTEGER IR,ILAST,NP
      INTEGER, PARAMETER :: NR=200
      INTEGER NTOT(T_INFO%NTYP,NR)
      REAL(q) QRAD(T_INFO%NIONS,NR),QRD1(T_INFO%NTYP,NR),RNT(T_INFO%NTYP,NR)

      REAL(q), PARAMETER :: TINY = 1.E-5_q

      QRAD=0; QRD1=0; RNT=0; NTOT=0

      NIS=1
 
      type1: DO NT=1,T_INFO%NTYP

      RWIGS=P(NT)%RCUTATO
      DO IR=1,NR
!        RNT(NT,IR)=RWIGS*(IR-1)/NR+TINY
         RNT(NT,IR)=RWIGS*(IR-0.5)/NR
      ENDDO

      FAC=P(NT)%RCUTATO/NR
! test
      WRITE(*,*) 'FAC=',FAC
! test

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D>TINY.AND.D<RWIGS) THEN
         QRAD(NI,INT(D/RWIGS*NR)+1)=QRAD(NI,INT(D/RWIGS*NR)+1)+DHTOT(IND1)/4/PI/FAC !/D/D
         NTOT(NT,INT(D/RWIGS*NR)+1)=NTOT(NT,INT(D/RWIGS*NR)+1)+1
      ENDIF
      
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D>TINY.AND.D<RWIGS) THEN
         QRAD(NI,INT(D/RWIGS*NR)+1)=QRAD(NI,INT(D/RWIGS*NR)+1)+DHTOT(IND1)/4/PI/FAC !/D/D
         NTOT(NT,INT(D/RWIGS*NR)+1)=NTOT(NT,INT(D/RWIGS*NR)+1)+1
      ENDIF 
      
      ENDDO
      ENDDO
      ENDDO
#endif

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      CALLMPI( M_sum_d(GRIDC%COMM, QRAD, T_INFO%NIONS*NR))
      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NTYP*NR ))

      QRD1=0

      DO IR=1,NR
      NIS=1
      DO NT=1,T_INFO%NTYP
         QSUM=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            QSUM=QSUM+QRAD(NI,IR)
         ENDDO
         QRD1(NT,IR)=QSUM/T_INFO%NITYP(NT)/GRIDC%NPLWV
         NIS = NIS+T_INFO%NITYP(NT)
      ENDDO
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(200)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR
            WRITE(200,'(2F14.7,I10)') RNT(NT,IR),QRD1(NT,IR),NTOT(NT,IR)
         ENDDO
         WRITE(200,*)
      ENDDO
      CLOSE(200)
      ENDIF
!#endif
      ! construct spline functions
      DO NT=1,T_INFO%NTYP
         ALLOCATE(SPLINES(NT)%SPL(NR,5))
         SPLINES(NT)%SPL=0
         SPLINES(NT)%RMAX=P(NT)%RCUTATO
         NP=0
         DO IR=1,NR
            IF (NTOT(NT,IR)/=0) THEN
               NP=NP+1
               SPLINES(NT)%SPL(NP,1)=RNT(NT,IR)
               SPLINES(NT)%SPL(NP,2)=QRD1(NT,IR)
               ILAST=IR
            ENDIF
         ENDDO
         IF (ILAST<NR) THEN
            SPLINES(NT)%SPL(NP+1:NP+NR-ILAST,1)=RNT(NT,ILAST+1:NR)
            SPLINES(NT)%SPL(NP+1:NP+NR-ILAST,2)=0
         ENDIF
         SPLINES(NT)%NP=NP+NR-ILAST 
!        ! first point, f(0)=0
!        SPLINES(NT)%SPL(1,1)=0._q; SPLINES(NT)%SPL(1,2)=0._q
!        SPLINES(NT)%SPL(1:NR,1)=RNT(NT,1:NR); SPLINES(NT)%SPL(1:NR,2)=QRD1(NT,1:NR)
!        CALL SPLCOF(SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1),1.0E30_q)
         CALL SPLCOF(SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1),0._q)
      ENDDO
!#ifdef debug
      IF (IO%IU6>=0) THEN
      WRITE(*,*) 'ILAST=',ILAST,' NP=',NP,' NR=',NR
      OPEN(250)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR+1
            D=SPLINES(NT)%RMAX/(NR+1)*(IR-1)+TINY
            CALL SPLVAL(D,DQ,DQD,SPLINES(NT)%SPL,SPLINES(NT)%NP,SIZE(SPLINES(NT)%SPL,1))
            WRITE(250,'(2F14.7)') D,DQ
         ENDDO
         WRITE(250,*)
      ENDDO
      CLOSE(250)
      OPEN(150)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR+1
            D=SPLINES(NT)%RMAX/(NR+1)*(IR-1)+TINY
            CALL LINPOL(D,SPLINES(NT),DQ)
            WRITE(150,'(2F14.7)') D,DQ
         ENDDO
         WRITE(150,*)
      ENDDO
      CLOSE(150)
      ENDIF
!#endif

      RETURN
      END SUBROUTINE ANALYSE_RL3


!******************** SUBROUTINE AVERAGE_RL ****************************
!
!***********************************************************************
      SUBROUTINE AVERAGE_RL(T_INFO,LATT_CUR,GRIDC,DHTOT,DHTOTW,SPLF0,SPLF1,IO)
      USE prec
      USE base
      USE constant
      USE poscar
      USE lattice
      USE mgrid
      USE mpimy
      USE pseudo
      USE radial
      USE spline
      IMPLICIT NONE
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      TYPE (grid_3d) GRIDC
      TYPE (in_struct) IO
      TYPE (splfnc) SPLF0(T_INFO%NTYP),SPLF1(T_INFO%NTYP)
      RGRID DHTOT(DIMREAL(GRIDC%MPLWV)),DHTOTW(DIMREAL(GRIDC%MPLWV))
      ! local variables
      INTEGER I
      INTEGER NIS,NT,NI
      INTEGER N1LOW,N2LOW,N3LOW,N1HI,N2HI,N3HI
      INTEGER N1,N2,N3,N1P,N2P,N3P,NCOL,IND,IND1
      REAL(q) F1,F2,F3,D1,D2,D3,X1,X2,X3
      REAL(q) XC,YC,ZC,D,RWIGS
      REAL(q) QSUM,QSM1,DQ,FAC

      INTEGER IR,ILAST,NP,NR

      INTEGER, ALLOCATABLE :: NTOT(:,:)
      REAL(q), ALLOCATABLE :: QRAD(:,:),QRD1(:,:),RNT(:,:)

      REAL(q), PARAMETER :: TINY = 1.E-5_q

      NR=MAXVAL(SPLF1(1:T_INFO%NTYP)%NP)
      ALLOCATE(NTOT(T_INFO%NTYP,NR),QRAD(T_INFO%NIONS,NR),QRD1(T_INFO%NTYP,NR),RNT(T_INFO%NTYP,NR))
      QRAD=0; QRD1=0; RNT=0; NTOT=0

      NIS=1

      type1: DO NT=1,T_INFO%NTYP

      NR=SPLF1(NT)%NP
      RWIGS=SPLF1(NT)%RMAX
      DO IR=1,NR
         RNT(NT,IR)=RWIGS*(IR-0.5)/NR
      ENDDO
      FAC=RWIGS/NR

      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRIDC%NGX
      F2=1._q/GRIDC%NGY
      F3=1._q/GRIDC%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= RWIGS*LATT_CUR%BNORM(1)*GRIDC%NGX
      D2= RWIGS*LATT_CUR%BNORM(2)*GRIDC%NGY
      D3= RWIGS*LATT_CUR%BNORM(3)*GRIDC%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRIDC%NGZ-D3+10*GRIDC%NGZ+.99_q)-10*GRIDC%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRIDC%NGY-D2+10*GRIDC%NGY+.99_q)-10*GRIDC%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRIDC%NGX-D1+10*GRIDC%NGX+.99_q)-10*GRIDC%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRIDC%NGZ+D3+10*GRIDC%NGZ)-10*GRIDC%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRIDC%NGY+D2+10*GRIDC%NGY)-10*GRIDC%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRIDC%NGX+D1+10*GRIDC%NGX)-10*GRIDC%NGX

!-----------------------------------------------------------------------
! loop over cubus
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)

      NCOL=GRIDC%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE ! not on local node, move on
      IF (GRIDC%RL%I2(NCOL) /= N1P+1 .OR. GRIDC%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'RHOADD: internal ERROR:', &
           GRIDC%RL%I2(NCOL),N1P+1, GRIDC%RL%I3(NCOL),N2P+1
        STOP
      ENDIF
!OCL SCALAR
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)
      IND1=N3P+(NCOL-1)*GRIDC%NGZ+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D>TINY.AND.D<RWIGS) THEN
         CALL LINPOL(D,SPLF0(NT),DQ)
         IF (ABS(DHTOTW(IND1))>TINY) &
        &   QRAD(NI,INT(D/RWIGS*NR)+1)=QRAD(NI,INT(D/RWIGS*NR)+1)+ &
        &      DHTOT(IND1)/DHTOTW(IND1)*DQ*LATT_CUR%OMEGA/4/PI/FAC !/D/D
         NTOT(NT,INT(D/RWIGS*NR)+1)=NTOT(NT,INT(D/RWIGS*NR)+1)+1
      ENDIF
 
      ENDDO
      ENDDO
      ENDDO
#else
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))
      N3P=MOD(N3+10*GRIDC%NGZ,GRIDC%NGZ)

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))
      N2P=MOD(N2+10*GRIDC%NGY,GRIDC%NGY)

      NCOL=GRIDC%RL%INDEX(N2P,N3P)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))
      N1P=MOD(N1+10*GRIDC%NGX,GRIDC%NGX)
      IND1=N1P+(NCOL-1)*GRIDC%NGX+1

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)

      IF (D>TINY.AND.D<RWIGS) THEN
         CALL LINPOL(D,SPLF0(NT),DQ)
         IF (ABS(DHTOTW(IND1))>TINY) &
        &   QRAD(NI,INT(D/RWIGS*NR)+1)=QRAD(NI,INT(D/RWIGS*NR)+1)+ &
        &      DHTOT(IND1)/DHTOTW(IND1)*DQ*LATT_CUR%OMEGA/4/PI/FAC !/D/D
         NTOT(NT,INT(D/RWIGS*NR)+1)=NTOT(NT,INT(D/RWIGS*NR)+1)+1
      ENDIF 
 
      ENDDO
      ENDDO
      ENDDO
#endif

      ENDDO ions1
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

      NR=MAXVAL(SPLF1(1:T_INFO%NTYP)%NP)

      CALLMPI( M_sum_d(GRIDC%COMM, QRAD, T_INFO%NIONS*NR))
      CALLMPI( M_sum_i(GRIDC%COMM, NTOT, T_INFO%NTYP*NR ))

      QRD1=0

      DO IR=1,NR
      NIS=1
      DO NT=1,T_INFO%NTYP
         QSUM=0
         DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
            QSUM=QSUM+QRAD(NI,IR)
         ENDDO
         QRD1(NT,IR)=QSUM/T_INFO%NITYP(NT)/GRIDC%NPLWV
         NIS = NIS+T_INFO%NITYP(NT)
      ENDDO
      ENDDO

!#ifdef debug
      IF (IO%IU6>=0) THEN
      OPEN(200)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR
            WRITE(200,'(2F14.7)') RNT(NT,IR),QRD1(NT,IR)
         ENDDO
         WRITE(200,*)
      ENDDO
      CLOSE(200)
      ENDIF
!#endif

      ! construct spline functions
      DO NT=1,T_INFO%NTYP
         SPLF1(NT)%SPL=0
         NP=0
         NR=SPLF1(NT)%NP
         DO IR=1,NR
            IF (NTOT(NT,IR)/=0) THEN
               NP=NP+1
               SPLF1(NT)%SPL(NP,1)=RNT(NT,IR)
               SPLF1(NT)%SPL(NP,2)=QRD1(NT,IR)
               ILAST=IR
            ENDIF
         ENDDO
         IF (ILAST<NR) THEN
            SPLF1(NT)%SPL(NP+1:NP+NR-ILAST,1)=RNT(NT,ILAST+1:NR)
            SPLF1(NT)%SPL(NP+1:NP+NR-ILAST,2)=0
         ENDIF
!        SPLF1(NT)%NP=NP+NR-ILAST 
         CALL SPLCOF(SPLF1(NT)%SPL,SPLF1(NT)%NP,SIZE(SPLF1(NT)%SPL,1),0._q)
      ENDDO
!#ifdef debug
      IF (IO%IU6>=0) THEN
      WRITE(*,*) 'ILAST=',ILAST,' NP=',NP,' NR=',NR
!     OPEN(250)
!     DO NT=1,T_INFO%NTYP
!        DO IR=1,NR+1
!           D=SPLF1(NT)%RMAX/(NR+1)*(IR-1)
!           CALL SPLVAL(D,DQ,DQD,SPLF1(NT)%SPL,SPLF1(NT)%NP,SIZE(SPLF1(NT)%SPL,1))
!           WRITE(250,'(2F14.7)') D,DQ
!        ENDDO
!        WRITE(250,*)
!     ENDDO
!     CLOSE(250)
      OPEN(150)
      DO NT=1,T_INFO%NTYP
         DO IR=1,NR+1
            D=SPLF1(NT)%RMAX/(NR+1)*(IR-1)+TINY
            CALL LINPOL(D,SPLF1(NT),DQ)
            WRITE(150,'(2F14.7)') D,DQ
         ENDDO
         WRITE(150,*)
      ENDDO
      CLOSE(150)
      ENDIF
!#endif

      DEALLOCATE(QRAD,QRD1,RNT,NTOT)

      RETURN
      END SUBROUTINE AVERAGE_RL


      SUBROUTINE ADDRHOC(GRIDC,RHO,A,RHOC,B)
      USE prec
      USE mgrid
      IMPLICIT NONE
      TYPE (grid_3d) GRIDC
      REAL(q) A,B
      RGRID RHO(DIMREAL(GRIDC%MPLWV)),RHOC(GRIDC%RL%NP)
      ! local variables
      INTEGER I

      DO I=1,GRIDC%RL%NP
         RHO(I)=A*RHO(I)+B*RHOC(I)
      ENDDO

      RETURN
      END SUBROUTINE ADDRHOC
