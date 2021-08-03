#define Zmethod3
#include "symbol.inc"
!***********************************************************************
!***********************************************************************
!
!***********************************************************************
!***********************************************************************

  MODULE setxcmeta
    USE prec
    
    INTEGER, SAVE :: ID_METAGGA=-1
    
    LOGICAL, SAVE :: LDOMETAGGA=.FALSE.
    
    LOGICAL, SAVE :: LMETA_NEEDS_POT=.FALSE.
    LOGICAL, SAVE :: LMETA_NEEDS_MU=.FALSE.

    INTEGER, SAVE :: LMAXTAU=6

    LOGICAL, SAVE :: LMIXTAU=.FALSE.
    
    ! MBJ related constants and variables
    LOGICAL, SAVE :: LMETA_SC_CMBJ=.FALSE.
    REAL(q), SAVE :: CMBJ=-1,CMBJA,CMBJB
    REAL(q),ALLOCATABLE, SAVE :: CMBJ_TYP(:)
    RGRID, ALLOCATABLE, SAVE  :: CMBJ_AUX(:)      ! grid quantity, usually stores C values for MBJ

    ! auxiliary variables needed for the selfconsistent MBJ
    REAL(q), SAVE :: GRHO_OVER_RHO_PW
    REAL(q), SAVE :: GRHO_OVER_RHO_ONE_CENTER
    REAL(q), SAVE :: GRHO_OVER_RHO_FACT
    
    ! MSx related variables
    REAL(q), SAVE :: MSX_RKAPPA,MSX_CFC,MSX_CFE

    CONTAINS
    
      SUBROUTINE XC_META_READER(IO,NTYP)
      USE prec
      USE base
      USE setexm
      USE vaspxml
      IMPLICIT NONE
      TYPE (in_struct) IO
      INTEGER NTYP
      ! local variables
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) CHARAC
      CHARACTER (40) SZNAM

      LOGICAL LASPH
      
      LOPEN=.FALSE.
      OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

      SZNAM='--'
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'METAGGA','=','#',';','S', &
     &            IDUM,RDUM,CDUM,LDUM,SZNAM,N,40,IERR)
      IF ((IERR/=0).AND.(IERR/=3)) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''METAGGA'' from file INCAR.'
         STOP
      ENDIF
      CALL XML_INCAR('METAGGA','S',IDUM,RDUM,CDUM,LDUM,SZNAM,N)

      CALL STRIP(SZNAM,N,'L'); CALL UPPER(SZNAM)
      
      IF (SZNAM(1:2)=='--') THEN
         ID_METAGGA=-1
      ELSEIF (SZNAM(1:4)=='PKZB') THEN
         ID_METAGGA=10
         LMETA_NEEDS_POT=.FALSE.
         LMETA_NEEDS_MU=.FALSE.
      ELSEIF (SZNAM(1:5)=='RTPSS') THEN
         ID_METAGGA=20
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
      ELSEIF (SZNAM(1:4)=='TPSS') THEN
         ID_METAGGA=25
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
      ELSEIF (SZNAM(1:3)=='MBJ') THEN
         ID_METAGGA=30
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.FALSE.
      ELSEIF (SZNAM(1:4)=='M06L') THEN
         ID_METAGGA=35
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
      ELSEIF (SZNAM(1:4)=='MS0') THEN
         ID_METAGGA=40
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
         MSX_RKAPPA=0.29_q
         MSX_CFC=0.28771_q
         MSX_CFE=1.0_q
      ELSEIF (SZNAM(1:4)=='MS1') THEN
         ID_METAGGA=41
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
         MSX_RKAPPA=0.404_q
         MSX_CFC=0.18150_q
         MSX_CFE=1.0_q
      ELSEIF (SZNAM(1:4)=='MS2') THEN
         ID_METAGGA=42
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
         MSX_RKAPPA=0.504_q
         MSX_CFC=0.14601_q
         MSX_CFE=4.0_q
      ELSEIF (SZNAM(1:4)=='SCAN') THEN
         ID_METAGGA=50
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.TRUE.
      ELSEIF (SZNAM(1:3)=='PBE') THEN
         ID_METAGGA=999
         LMETA_NEEDS_POT=.TRUE.
         LMETA_NEEDS_MU=.FALSE.
      ENDIF

      IF (ID_METAGGA/=-1) LDOMETAGGA=.TRUE.

      ! additional tags for MBJ
      IF (ID_METAGGA==30) THEN
         CMBJ=-1
         ALLOCATE(CMBJ_TYP(NTYP))
         CMBJ_TYP=-1
         CALL RDATAB(LOPEN,INCAR,IO%IU5,'CMBJ','=','#',';','F', &
        &            IDUM,CMBJ_TYP,CDUM,LDUM,CHARAC,N,NTYP,IERR)
         IF ((IERR/=0).AND.(IERR/=3)) THEN
            IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''CMBJ'' from file INCAR.'
            STOP
         ELSE
            CMBJ=CMBJ_TYP(1)
            IF (N<NTYP .OR. N==1 ) THEN
               CMBJ_TYP=-1
            ENDIF
         ENDIF
         IF (N==1) THEN
            CALL XML_INCAR('CMBJ','F',IDUM,CMBJ,CDUM,LDUM,CHARAC,1)
         ELSE
            CALL XML_INCAR_V('CMBJ','F',IDUM,CMBJ_TYP,CDUM,LDUM,CHARAC,N)
         ENDIF
         ! If CMBJ is not set then it will be calculated (default)
         IF (CMBJ==-1) THEN
            LMETA_SC_CMBJ=.TRUE.
            CMBJ=1.0_q
            CMBJA=-0.012_q
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'CMBJA','=','#',';','F', &
           &            IDUM,CMBJA,CDUM,LDUM,CHARAC,N,NTYP,IERR)
            IF ((IERR/=0).AND.(IERR/=3)) THEN
               IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''CMBJA'' from file INCAR.'
               CMBJA=-0.012_q
            ENDIF
            CALL XML_INCAR('CMBJA','F',IDUM,CMBJA,CDUM,LDUM,CHARAC,N)

            CMBJB=1.023_q 
            CALL RDATAB(LOPEN,INCAR,IO%IU5,'CMBJB','=','#',';','F', &
           &            IDUM,CMBJB,CDUM,LDUM,CHARAC,N,NTYP,IERR)
            IF ((IERR/=0).AND.(IERR/=3)) THEN
               IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''CMBJB'' from file INCAR.'
               CMBJB=1.023_q
            ENDIF           
            CALL XML_INCAR('CMBJB','F',IDUM,CMBJB,CDUM,LDUM,CHARAC,N)
         ENDIF
      ENDIF

      LASPH=.FALSE.      
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LASPH','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LASPH,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LASPH'' from file INCAR.'
         LASPH=.FALSE.
      ENDIF
      CALL XML_INCAR('LASPH','L',IDUM,RDUM,CDUM,LASPH,CHARAC,N)

      IF (LASPH) THEN
         LMAXTAU=6
         CALL RDATAB(LOPEN,INCAR,IO%IU5,'LMAXTAU','=','#',';','I', &
        &            LMAXTAU,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF ((IERR/=0).AND.(IERR/=3)) THEN
            IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LMAXTAU'' from file INCAR.'
            STOP
         ENDIF
      ELSE
         LMAXTAU=0
      ENDIF
      CALL XML_INCAR('LMAXTAU','I',LMAXTAU,RDUM,CDUM,LDUM,CHARAC,N)

      LMIXTAU=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IO%IU5,'LMIXTAU','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LMIXTAU,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LMIXTAU'' from file INCAR.'
         LMIXTAU=.FALSE.
      ENDIF
      CALL XML_INCAR('LMIXTAU','I',LMIXTAU,RDUM,CDUM,LDUM,CHARAC,N)

      CLOSE(IO%IU5)   

      ! Write to OUTCAR
      IF (IO%IU6>=0 .AND. LDOMETAGGA) THEN
         WRITE(IO%IU6,'(/X,A,X,A5,4X,A,I3,4X,A,L3)',ADVANCE="No") 'METAGGA =',SZNAM(1:5),'LMAXTAU =',LMAXTAU,'LMIXTAU =',LMIXTAU
         IF (ID_METAGGA==30) THEN
            IF (CMBJ_TYP(1)==-1) THEN
               WRITE(IO%IU6,'(4X,A,F10.4/)') 'CMBJ =',CMBJ
            ELSE
               WRITE(IO%IU6,'(4X,A,20F10.4/)') 'CMBJ =',CMBJ_TYP
            ENDIF
         ELSE
            WRITE(IO%IU6,'(/)')
         ENDIF
      ENDIF
      
      RETURN
      END SUBROUTINE XC_META_READER


!***********************************************************************
!
!***********************************************************************

      SUBROUTINE SET_CMBJ_PW(I)
      INTEGER I
      IF (ID_METAGGA==30.AND.CMBJ_TYP(1)/=-1) CMBJ=CMBJ_AUX(I)
      RETURN
      END SUBROUTINE SET_CMBJ_PW


!***********************************************************************
!
!***********************************************************************

      SUBROUTINE SET_CMBJ_RAD(NT)
      INTEGER NT
      IF (LDOMETAGGA.AND.ID_METAGGA==30) THEN
         IF (CMBJ_TYP(1)/=-1) CMBJ=CMBJ_TYP(NT)
      ENDIF
      RETURN
      END SUBROUTINE SET_CMBJ_RAD


!***********************************************************************
!
!***********************************************************************
      FUNCTION LscMBJ()
      IMPLICIT NONE
      LOGICAL LscMBJ
      LscMBJ=LMETA_SC_CMBJ
      END FUNCTION LscMBJ


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE SUM_GRHO_OVER_RHO_PW(RHO,GRHO)
      USE prec
      REAL(q) RHO,GRHO
      GRHO_OVER_RHO_PW=GRHO_OVER_RHO_PW+GRHO/RHO
      RETURN
      END SUBROUTINE SUM_GRHO_OVER_RHO_PW


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE SUM_GRHO_OVER_RHO_ONE_CENTER(RHO,GRHO)
      USE prec
      REAL(q) RHO,GRHO
      GRHO_OVER_RHO_ONE_CENTER=GRHO_OVER_RHO_ONE_CENTER+ &
     &   GRHO_OVER_RHO_FACT*GRHO/RHO
      RETURN
      END SUBROUTINE SUM_GRHO_OVER_RHO_ONE_CENTER


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE SET_CMBJ_ONE_CENTER_FACT(FACT)
      USE prec
      REAL(q) FACT
      GRHO_OVER_RHO_FACT=FACT
      RETURN
      END SUBROUTINE SET_CMBJ_ONE_CENTER_FACT


!***********************************************************************
!***********************************************************************
  END MODULE setxcmeta
!***********************************************************************
!***********************************************************************


!***********************************************************************
!***********************************************************************
!
!***********************************************************************
!***********************************************************************

  MODULE metalib
    USE prec
    USE constant
    USE setxcmeta

    PRIVATE :: IR_MATCH,IR_PSMAX

    CONTAINS
    
      SUBROUTINE METAGGASPIN(&
     &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,LAPLUP,LAPLDW,TAUUP,TAUDW, &
     &   EXC,dEXCdRHOup,dEXCdRHOdw,dEXCdABSNABup,dEXCdABSNABdw,dEXCdABSNAB, &
     &   dEXCdTAUup,dEXCdTAUdw)
      IMPLICIT NONE
      REAL(q) RHOUP,RHODW
      REAL(q) ABSNABUP,ABSNABDW,ABSNAB
      REAL(q) LAPLUP,LAPLDW
      REAL(q) TAUUP,TAUDW

      REAL(q) EXC
      REAL(q) dEXCdRHOup,dEXCdRHOdw
      REAL(q) dEXCdABSNABup,dEXCdABSNABdw,dEXCdABSNAB
      REAL(q) dEXCdTAUup,dEXCdTAUdw

      ! local variables
      REAL(q) EXL,ECL    
      REAL(q) Ex_TPSS,Ec_TPSS
      REAL(q) Ex_revTPSS,Ec_revTPSS
      REAL(q) Ex_SCAN,Ec_SCAN
      REAL(q) Ex_M06,Ec_M06
      REAL(q) Ex_MSX,Ec_MSX
      REAL(q) VXD1,VXD2,VXDD1,VXDD2,AMUXD1,AMUXD2
      REAL(q) VCD1,VCD2,VCDD1,VCDD2,AMUCD1,AMUCD2

      ! VMBJ related variables
      REAL(q), PARAMETER :: THRD=1._q/3._q
      REAL(q) D,DTHRD,RS,ZETA,FK,SK,G,T
      REAL(q) ECLDA,ECD1LDA,ECD2LDA,EC,ECD1,ECD2,ECQ

      EXC=0
      dEXCdRHOup=0; dEXCdRHOdw=0
      dEXCdABSNABup=0; dEXCdABSNABdw=0; dEXCdABSNAB=0
      dEXCdTAUup=0; dEXCdTAUdw=0

      IF (ID_METAGGA==-1) THEN
      ! Just a stub, no metaGGA requested
      ELSEIF (ID_METAGGA==10) THEN
      ! PKZB
         CALL EPKZB(RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
        &   EXL,ECL)
         EXC=(EXL+ECL)/(RHOUP+RHODW)
         ! from Hartree to Rydberg
         EXC=EXC*2
      ELSEIF (ID_METAGGA==20) THEN
      ! revTPSS
         ! Exchange
         CALL VrevTPSSx(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ex_revTPSS,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

         ! Correlation
         CALL VrevTPSSc(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ec_revTPSS,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

         ! Sum everything
         EXC=(Ex_revTPSS+Ec_revTPSS)/(RHOUP+RHODW)
         dEXCdRHOup=VXD1+VCD1
         dEXCdRHOdw=VXD2+VCD2
         dEXCdABSNABup=VXDD1+VCDD1
         dEXCdABSNABdw=VXDD2+VCDD2
         dEXCdTAUup=AMUXD1+AMUCD1
         dEXCdTAUdw=AMUXD2+AMUCD2
         ! from Hartree to Rydberg
         EXC=EXC*2
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
         dEXCdABSNABup=dEXCdABSNABup*2
         dEXCdABSNABdw=dEXCdABSNABdw*2
         dEXCdTAUup=dEXCdTAUup*2
         dEXCdTAUdw=dEXCdTAUdw*2
      ELSEIF (ID_METAGGA==25) THEN
      ! TPSS
         ! Exchange
         CALL VTPSSx(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ex_TPSS,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

         ! Correlation
         CALL VTPSSc(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ec_TPSS,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

         ! Sum everything
         EXC=(Ex_TPSS+Ec_TPSS)/(RHOUP+RHODW)
         dEXCdRHOup=VXD1+VCD1
         dEXCdRHOdw=VXD2+VCD2
         dEXCdABSNABup=VXDD1+VCDD1
         dEXCdABSNABdw=VXDD2+VCDD2
         dEXCdTAUup=AMUXD1+AMUCD1
         dEXCdTAUdw=AMUXD2+AMUCD2
         ! from Hartree to Rydberg
         EXC=EXC*2
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
         dEXCdABSNABup=dEXCdABSNABup*2
         dEXCdABSNABdw=dEXCdABSNABdw*2
         dEXCdTAUup=dEXCdTAUup*2
         dEXCdTAUdw=dEXCdTAUdw*2
      ELSEIF (ID_METAGGA==30) THEN
      ! MBJ
         ! spin up
         CALL VMBJ_BRENT(RHOUP,ABSNABUP,LAPLUP,2._q*TAUUP,CMBJ,dEXCdRHOup)
         ! spin down
         CALL VMBJ_BRENT(RHODW,ABSNABDW,LAPLDW,2._q*TAUDW,CMBJ,dEXCdRHOdw)

         ! get the LDA correlation potential (using a call to CORPBE)
         D=RHOUP+RHODW
         DTHRD=exp(log(D)*THRD)
         RS=(0.75_q/PI)**THRD/DTHRD
         ZETA=(RHOUP-RHODW)/D
         ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
         FK=(3._q*PI*PI)**THRD*DTHRD
         SK=SQRT(4.0_q*FK/PI)
         G=(exp((2*THRD)*log(1._q+ZETA))+exp((2*THRD)*log(1._q-ZETA)))/2._q
         T=ABSNAB/(D*2._q*SK*G)

         CALL CORPBE(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA,G,SK,T,EC,ECD1,ECD2,ECQ,.TRUE.)

         ! add the LDA correlation potential
         dEXCdRHOup=dEXCdRHOup+ECD1LDA
         dEXCdRHOdw=dEXCdRHOdw+ECD2LDA

         ! from Hartree to Rydberg
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
      ELSEIF (ID_METAGGA==35) THEN
      ! M06-L
         ! Exchange
         CALL VM06x(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ex_M06,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2,1)

         ! Correlation
         CALL VM06c(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ec_M06,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2,1)

         ! Sum everything
         EXC=(Ex_M06+Ec_M06)/(RHOUP+RHODW)
         dEXCdRHOup=VXD1+VCD1
         dEXCdRHOdw=VXD2+VCD2
         dEXCdABSNABup=VXDD1+VCDD1
         dEXCdABSNABdw=VXDD2+VCDD2
         dEXCdTAUup=AMUXD1+AMUCD1
         dEXCdTAUdw=AMUXD2+AMUCD2
         ! from Hartree to Rydberg
         EXC=EXC*2
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
         dEXCdABSNABup=dEXCdABSNABup*2
         dEXCdABSNABdw=dEXCdABSNABdw*2
         dEXCdTAUup=dEXCdTAUup*2
         dEXCdTAUdw=dEXCdTAUdw*2
      ELSEIF (ID_METAGGA==40.OR.ID_METAGGA==41.OR.ID_METAGGA==42) THEN
      ! revTPSS
         ! Exchange
         CALL VMSXx(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ex_MSX,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

         ! Correlation
         CALL VMSXc(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ec_MSX,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

         ! Sum everything
         EXC=(Ex_MSX+Ec_MSX)/(RHOUP+RHODW)
         dEXCdRHOup=VXD1+VCD1
         dEXCdRHOdw=VXD2+VCD2
         dEXCdABSNABup=VXDD1+VCDD1
         dEXCdABSNABdw=VXDD2+VCDD2
         dEXCdTAUup=AMUXD1+AMUCD1
         dEXCdTAUdw=AMUXD2+AMUCD2
         ! from Hartree to Rydberg
         EXC=EXC*2
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
         dEXCdABSNABup=dEXCdABSNABup*2
         dEXCdABSNABdw=dEXCdABSNABdw*2
         dEXCdTAUup=dEXCdTAUup*2
         dEXCdTAUdw=dEXCdTAUdw*2
      ELSEIF (ID_METAGGA==50) THEN
      ! SCAN
         ! Exchange
         CALL VSCANx(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ex_SCAN,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

         ! Correlation
         CALL VSCANc(&
       &   RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB,TAUUP,TAUDW, &
       &   Ec_SCAN,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

         ! Sum everything
         EXC=(Ex_SCAN+Ec_SCAN)/(RHOUP+RHODW)
         dEXCdRHOup=VXD1+VCD1
         dEXCdRHOdw=VXD2+VCD2
         dEXCdABSNABup=VXDD1+VCDD1
         dEXCdABSNABdw=VXDD2+VCDD2
         dEXCdTAUup=AMUXD1+AMUCD1
         dEXCdTAUdw=AMUXD2+AMUCD2
         ! from Hartree to Rydberg
         EXC=EXC*2
         dEXCdRHOup=dEXCdRHOup*2
         dEXCdRHOdw=dEXCdRHOdw*2
         dEXCdABSNABup=dEXCdABSNABup*2
         dEXCdABSNABdw=dEXCdABSNABdw*2
         dEXCdTAUup=dEXCdTAUup*2
         dEXCdTAUdw=dEXCdTAUdw*2
      ELSEIF (ID_METAGGA==999) THEN
      ! PBE, for testing mainly
         CALL GGASPIN(RHOUP,RHODW,ABSNABUP,ABSNABDW,ABSNAB, &
       &    EXC,dEXCdRHOup,dEXCdRHOdw,dEXCdABSNABup,dEXCdABSNABdw,dEXCdABSNAB,.TRUE.)
      ENDIF

      RETURN
      END SUBROUTINE METAGGASPIN

!***********************************************************************
! Below we should enter all functionals
!***********************************************************************

!****************** SUBROUTINE VMBJ_BRENT ******************************
!
! calculates modified Becke-Johnson exchange potential
! according to Tran and Blaha, PRL 102, 226401 (2009), which is
! based on Becke and Roussel, PRA 39, 3761 (1989) and
! Becke and Jhonson, J. Chem. Phys. 124, 221101 (2006)
!
! This routine find the root using the Brent's method
! And it implemented in VASP by Yoon-Suk Kim. 20100922
!
! everything in Hartree unit and Bohr
!
! ATTENTION: All values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RHO        electron density (up or down)
! GRHO       abs. val. gradient of density (up or down)
! G2RHO      abs. val. second gradient of total density (up or down)
! TAU        kinetic energy density (up or down)
! XCCONST    fitted value by int{|nabla rho|/rho}dr3
! VXBRJ      return value
!
!***********************************************************************

      SUBROUTINE VMBJ_BRENT(RHO,GRHO,G2RHO,TAU,XCCONST,VXBRJ)
      USE prec
      USE constant
      IMPLICIT NONE

      REAL(q) RHO,GRHO,G2RHO,TAU,XCCONST,VXBRJ
      ! local variables
      REAL(q) SMALL,TMP
      REAL(q) TOL,X,Y,X1,Y1,X2,Y2,X21,Y21,X22,S,M
      REAL(q) DSIGMA,QSIGMA,BSIGMA
      INTEGER NLOOP
      LOGICAL LMETHOD

      REAL(q), PARAMETER :: THRD=1._q/3._q
      REAL(q), PARAMETER :: TTHRD=2._q*THRD
      REAL(q), PARAMETER :: FTHRD=1._q+TTHRD
      REAL(q), PARAMETER :: GAM=0.8_q

      SMALL=1.E-8_q
      VXBRJ=0._q

      ! Now start the MBJ routine
      IF (RHO>SMALL) THEN
         TOL=1.E-6_q
         NLOOP=0
       
         DSIGMA=TAU-0.25_q*GRHO**2._q/RHO
         QSIGMA=(1._q/6._q)*(G2RHO-2._q*GAM*DSIGMA)
      
         LMETHOD = .TRUE. 
         X1=0._q; X2=0._q; Y1=1._q ; Y2=1._q
         DO WHILE (Y1*Y2>0._q)
            X1=X2
            X2=X2+1._q
            Y1=X1*EXP(-TTHRD*X1)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X1-2._q)
            Y2=X2*EXP(-TTHRD*X2)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X2-2._q)
            NLOOP = NLOOP +1
            IF (NLOOP>=400) THEN
               WRITE(*,*) 'VMBJ_BRENT: INTERNAL ERROR. Unable to find initial points.'
               STOP
            ENDIF
         ENDDO
         IF (ABS(Y1)<ABS(Y2)) THEN
            TMP=X1;X1=X2;X2=TMP;TMP=Y1;Y1=Y2;Y2=TMP
         ENDIF
         X21=X1; Y21=Y1
         !WRITE(*,*) 'The interval X1,Y1=',X1,Y1
         !WRITE(*,*) 'The interval X2,Y2=',X2,Y2
         !FY=(1._q-TTHRD*X)*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD
         !TOL=2._q*TOL*ABS(X2)

         NLOOP=0
         !Now start the Brent's method 
         DO WHILE (ABS(Y2)>TOL)
            S=X2-(X2-X21)/(Y2-Y21)*Y2
            M=(X1+X2)*0.5_q
         
            IF (LMETHOD) THEN
               IF (ABS(X2-X21)>TOL .AND. ABS(S-X2)<ABS(X2-X21)*0.5_q) THEN
                  X=S
                  Y=X*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X-2._q)
                  IF (Y1*Y<0._q) THEN
                     X22=X21; X21=X2; Y21=Y2; X2=S; Y2=Y
                  ELSE
                     X22=X21; X21=X1; Y21=Y1; X1=S; Y1=Y
                  ENDIF
                  LMETHOD = .FALSE.
               ELSE
                  X=M
                  Y=X*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X-2._q)
                  IF (Y1*Y<0._q) THEN
                     X22=X21; X21=X2; Y21=Y2; X2=M; Y2=Y
                  ELSE
                     X22=X21; X21=X2; Y21=Y2; X1=M; Y1=Y
                  ENDIF
                  LMETHOD = .TRUE.
               ENDIF
            ELSE
               IF (ABS(X21-X22)>TOL .AND. ABS(S-X2)<ABS(X21-X22)*0.5_q) THEN
                  X=S
                  Y=X*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X-2._q)
                  IF (Y1*Y<0._q) THEN
                     X22=X21; X21=X2; Y21=Y2; X2=S; Y2=Y
                  ELSE
                     X22=X21; X21=X1; Y21=Y2; X1=S; Y1=Y
                  ENDIF
                  LMETHOD = .FALSE.
               ELSE
                  X=M
                  Y=X*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X-2._q)
                  IF (Y1*Y<0._q) THEN
                     X21=X2; Y21=Y2; X2=M; Y2=Y
                  ELSE
                     X21=X1; Y21=Y1; X1=M; Y1=Y
                  ENDIF
                  LMETHOD = .TRUE.
               ENDIF
            ENDIF
            IF (ABS(X1-X2)<TOL .AND. Y1*Y2<0._q) THEN
               Y2=TOL/10._q
            ENDIF

            !Now X1 and X2 are closed to the root.
            !|Y2| should be less than or equal to |Y1|, so that X2 is a better guess
            !for the unknown solution than X1.
            IF (ABS(Y1)<ABS(Y2)) THEN
               TMP=X1;X1=X2;X2=TMP;TMP=Y1;Y1=Y2;Y2=TMP
            ENDIF
            !WRITE(*,*) 'X2,Y2=',X2,Y2
            NLOOP = NLOOP +1
            IF (NLOOP>=400) THEN
               WRITE(*,*) 'VMBJ_BRENT: INTERNAL ERROR. Too many iterations.. I am tired.'
               WRITE(*,*) 'The interval X1,Y1=',X1,Y1
               WRITE(*,*) 'The interval X2,Y2=',X2,Y2
               WRITE(*,*) 'The interval X21,Y21=',X21,Y21
               WRITE(*,*) 'QSIGMA,RHO=',QSIGMA,RHO
               STOP
            ENDIF
         ENDDO
         !WRITE(*,*) 'Final X2,Y2=',X2,Y2

         X=X2
         BSIGMA=(X**3._q*EXP(-X)/(8._q*PI*RHO))**THRD

         IF (BSIGMA>SMALL) THEN
           VXBRJ=-(1._q-EXP(-X)-0.5_q*X*EXP(-X))/BSIGMA
           VXBRJ=XCCONST*VXBRJ+(3._q*XCCONST-2._q)/PI*SQRT(5._q/12._q)*SQRT(TAU/RHO)
         ELSE
           VXBRJ=0._q
         ENDIF

      ELSE
         VXBRJ=0._q
      ENDIF

      RETURN
      END SUBROUTINE VMBJ_BRENT

!****************** SUBROUTINE VMBJ ************************************
!
! calculates modified Becke-Johnson exchange potential
! according to Tran and Blaha, PRL 102, 226401 (2009), which is
! based on Becke and Roussel, PRA 39, 3761 (1989) and
! Becke and Jhonson, J. Chem. Phys. 124, 221101 (2006)
!
! The original source code was written by Fabien Tran and Peter Blaha
! And it implemented in VASP by Yoon-Suk Kim. 20090612
!
! everything in Hartree unit and Bohr
!
! ATTENTION: All values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RHO        electron density (up or down)
! GRHO       abs. val. gradient of density (up or down)
! G2RHO      abs. val. second gradient of total density (up or down)
! TAU        kinetic energy density (up or down)
! XCCONST    fitted value by int{|nabla rho|/rho}dr3
! VXBRJ      return value
!
!***********************************************************************

      SUBROUTINE VMBJ(RHO,GRHO,G2RHO,TAU,XCCONST,VXBRJ)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

      PARAMETER (THRD=1._q/3._q)
      PARAMETER (TTHRD=2._q*THRD)
      PARAMETER (FTHRD=1._q+TTHRD)
      PARAMETER (GAM=0.8_q)

      REAL(q) RHO,GRHO,G2RHO,TAU,XCCONST,VXBRJ
      ! local variables
      REAL(q) SMALL,BIG
      REAL(q) TOL,F,DF,X,X1,X2,DSIGMA,QSIGMA,BSIGMA
      INTEGER NLOOP

      SMALL=1.E-8_q
      BIG=1.E10_q
      VXBRJ=0._q

      ! Now start the MBJ routine
      IF (RHO>SMALL) THEN
      TOL=1.E-6_q
      F=10._q*TOL
      NLOOP=0
      X1=0._q; X2=0._q

      DSIGMA=TAU-0.25_q*GRHO**2._q/RHO
      QSIGMA=(1._q/6._q)*(G2RHO-2._q*GAM*DSIGMA)

      ! Newton-Raphson iterative method (N-R),
      ! Press, Flannery, Teukolsky, and Vettering, Numerical Recipes,
      ! (Cambridge University Press, Cambridge, England, 1986)
   10    DO WHILE (ABS(F)>=TOL)
           F=X*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD*(X-2._q)
           DF=(1._q-TTHRD*X)*EXP(-TTHRD*X)*QSIGMA-TTHRD*PI**TTHRD*RHO**FTHRD
           IF (ABS(DF)>SMALL) THEN
             X=X-F/DF
           ELSE
             X=X-F/SIGN(SMALL,DF)
           ENDIF
           ! Check the status of iteration..
           ! if too many loops, too large X, or too big F,
           ! then change the initial values and start over N-R.
           IF ((NLOOP>=400) .OR. (ABS(X)>=BIG) .OR. (ABS(F)>=BIG)) THEN
             X=X1
             X1=X1+1._q
             F=10._q*TOL
             NLOOP=0
             GOTO 10
           ENDIF
           NLOOP=NLOOP+1
         ENDDO ! end do while

         ! If the solution X has negative value then start over N-R.
         IF (X<0._q) THEN
           X=X2
           X2=X2+1._q
           F=10._q*TOL
           NLOOP=0
           GOTO 10
         ENDIF

         BSIGMA=(X**3._q*EXP(-X)/(8._q*PI*RHO))**THRD

         IF (BSIGMA>SMALL) THEN
           VXBRJ=-(1._q-EXP(-X)-0.5_q*X*EXP(-X))/BSIGMA
           VXBRJ=XCCONST*VXBRJ+(3._q*XCCONST-2._q)/PI*SQRT(5._q/12._q)*SQRT(TAU/RHO)
         ELSE
           VXBRJ=0._q
         ENDIF

      ELSE
         VXBRJ=0._q
      ENDIF

      RETURN
      END SUBROUTINE VMBJ


!************************ SUBROUTINE EPKZB *****************************
!
! calculates local contribution to metagga Exc according to 
! Perdew et. al. PRL 82, 12 (1999)
!
! RH 20001119
!
! everything in Hartree units
!
! ATTENTION: Every values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RU,RD      density up,down
! DRU, DRD   abs. val. gradient of density up/down
! DRT        abs. val. gradient of total density
! TAUU,TAUD  kinetic energy density up/down
! TAUWU,TAUWD Weizsaecker kinetic energy density up/down
! EXC        return value
!
!***********************************************************************

      SUBROUTINE EPKZB(RU,RD,DRU,DRD,DRT,TAUU,TAUD,EX,EC)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

      ! the following parameters are given by Perdew et.al.
      PARAMETER (RKAPPA=0.804_q)
      PARAMETER (D=0.113_q)
      PARAMETER (C=0.53_q)
      ! other parameters
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (TTHRD=2._q*THRD)
      PARAMETER (FTHRD=1._q+TTHRD)
      PARAMETER (ETHRD=1._q+FTHRD)
      PARAMETER (PISQ=PI*PI)

      TAUWU=0.125_q*DRU**2._q/RU
      TAUWU=MIN(TAUWU,TAUU)

      TAUWD=0.125_q*DRD**2._q/RD
      TAUWD=MIN(TAUWD,TAUD)
      
      EX=0._q;EC=0._q
      ! exchange energy
      ! spin up
      P=(2._q*DRU)**2._q/(4._q*(3._q*PISQ)**TTHRD*(2._q*RU)**ETHRD)
      QQS=6._q*TAUU/(2._q*(3._q*PISQ)**TTHRD*(2._q*RU)**FTHRD)-9._q/20._q-P/12._q
      X=10._q/81._q*P+146._q/2025._q*QQS*QQS-73._q/405._q*QQS*P
      X=X+(D+1._q/RKAPPA*(10._q/81._q)**2._q)*P*P
      FX=1._q+RKAPPA-RKAPPA/(1._q+(X/RKAPPA))
      EX=EX-RU*(3._q/(4._q*PI))*(3._q*PISQ*2._q*RU)**THRD*FX
      ! spin down
      P=(2._q*DRD)**2._q/(4._q*(3._q*PISQ)**TTHRD*(2._q*RD)**ETHRD)
      QQS=6._q*TAUD/(2._q*(3._q*PISQ)**TTHRD*(2._q*RD)**FTHRD)-9._q/20._q-P/12._q
      X=10._q/81._q*P+146._q/2025._q*QQS*QQS-73._q/405._q*QQS*P
      X=X+(D+1._q/RKAPPA*(10._q/81._q)**2._q)*P*P
      FX=1._q+RKAPPA-RKAPPA/(1._q+(X/RKAPPA))
      EX=EX-RD*(3._q/(4._q*PI))*(3._q*PISQ*2._q*RD)**THRD*FX

      ! correlation energy
      CALL GGASPINCOR(RU,RD,DRT,ECT)
      TAUK=(TAUWU+TAUWD)/(TAUU+TAUD)
      ECM1=(RU+RD)*ECT*(1._q+C*TAUK**2._q)

!     CALL GGACOR(RU,DRU,ECU)
      CALL GGASPINCOR(RU,0.0_q,DRU,ECU)
      TAUK=TAUWU/TAUU
      ECM2=TAUK**2._q*RU*ECU

!     CALL GGACOR(RD,DRD,ECD)
      CALL GGASPINCOR(RD,0.0_q,DRD,ECD)
      TAUK=TAUWD/TAUD
      ECM3=TAUK**2._q*RD*ECD
      EC=ECM1-(1._q+C)*(ECM2+ECM3)

      RETURN
      END SUBROUTINE EPKZB


!************************ SUBROUTINE VrevTPSSx *****************************
!
! calculates the first order derivatives of Ex wrt n and |grad(n)|
! Perdew et. al. PRL (2009)
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! everything in Hartree units
!
! ATTANTION: Every values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VXD1 VXD2                    THE DERIVATIVES OF EX WRT n 
! VXDD1,VXDD2                  THE DERIVATIVES OF EX WRT |grad n| 
! AMUXD1, AMUXD2               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VrevTPSSx(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ex_revTPSS,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the exchange part
      PARAMETER (RKAPPA=0.804_q)
      PARAMETER (CFB=0.40_q)
      PARAMETER (CFC=2.35203946_q)
      PARAMETER (CFE=2.16769874_q)
      PARAMETER (CFMU=0.14_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (AX=-0.738558766382022405884230032680836_q)
! test
      PARAMETER (thresh=1.e-10_q)
      PARAMETER (xorder=12._q)
      PARAMETER (zinfinity=2._q)
! test
      VXD1=0._q;VXD2=0._q;
      VXDD1=0._q;VXDD2=0._q;
      AMUXD1=0._q;AMUXD2=0._q;


      CX1=10._q/81._q
      CX2=146._q/2025._q
      CX3=73._q/405._q
      CFE12=SQRT(CFE)

! Suspect that TAUWU and TAUWD are not well described. Use TAUW_TEMP=0.125_q*DRT**2/RT
! IF WANT TO TEST TAUW, SIMPLY OVERWRITE TAUW_TEMP BY TAUW
      TAUWU_TEMP=0.125_q*DRU**2._q/RU
!     TAUWU_TEMP=MIN(TAUWU_TEMP,TAUU)
      
      TAUWD_TEMP=0.125_q*DRD**2._q/RD
!     TAUWD_TEMP=MIN(TAUWD_TEMP,TAUD)

! spin up
! IN EXD1(2*RU), TAUWU AND TAUU SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RU
      DRHO=TWO*DRU
      TAUW_RHO=TWO*TAUWU_TEMP
      TAU_RHO=TWO*TAUU

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|
      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P
      Z=TAUW_RHO/TAU_RHO
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2 
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test      
      Z2=Z*Z
      Z3=Z2*Z
      ZTF=0.6_q*Z
      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5
      
!     ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      ALPHA=5._q/3._q*P*(1._q/Z-1._q)
      
      QB_NUM=9._q/20._q*(ALPHA-ONE)
      QB_DEN=SQRT(ONE+CFB*ALPHA*(ALPHA-ONE))
      QB=QB_NUM/QB_DEN+ TWO*P/THREE
      OPZ=ONE+Z2
      SFP=Z3/OPZ**TWO
      FQ=SQRT((ZTF**TWO+P2)/TWO)
      S1=(CX1+CFC*SFP)*P
      S2=CX2*QB**TWO
      S3=-CX3*FQ*QB
      S4=(CX1*P)**TWO/RKAPPA
      S5=TWO*CFE12*CX1*ZTF**TWO
      S6=CFE*CFMU*P2*P
      XN=S1+S2+S3+S4+S5+S6
      XD=(ONE+CFE12*P)**TWO
      X=XN/XD

!    GET THE VALUE FOR FX
      FX=ONE+RKAPPA-RKAPPA/(ONE+(X/RKAPPA))

!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

      DALPHAD=THRD5*(ONE/Z-ONE)*DPD-THRD5*P*DZD/Z**TWO
!     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   &  
!    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

      DALPHADD=THRD5*(ONE/Z-ONE)*DPDD-THRD5*P*DZDD/Z**TWO
!     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

      DALPHADTAU=-THRD5*P*DZDTAU/Z**TWO
!     DALPHADTAU=ONE/TAU_UNIF

      DQB_NUM_D=9._q/20._q*DALPHAD
      DQB_NUM_DD=9._q/20._q*DALPHADD
      DQB_NUM_DTAU=9._q/20._q*DALPHADTAU
      C_DQB_DEN=ONE/TWO*CFB*(TWO*ALPHA-ONE)/QB_DEN
      DQB_DEN_D=C_DQB_DEN*DALPHAD
      DQB_DEN_DD=C_DQB_DEN*DALPHADD
      DQB_DEN_DTAU=C_DQB_DEN*DALPHADTAU

      DQBD=(DQB_NUM_D*QB_DEN-QB_NUM*DQB_DEN_D)/QB_DEN**TWO+THRD2*DPD
      DQBDD=(DQB_NUM_DD*QB_DEN-QB_NUM*DQB_DEN_DD)/QB_DEN**TWO+THRD2*DPDD
      DQBDTAU=(DQB_NUM_DTAU*QB_DEN-QB_NUM*DQB_DEN_DTAU)/QB_DEN**TWO+THRD2*DPDTAU

      C1_DS1=P*Z2*(THREE*CFC-CFC*Z2)/OPZ**THREE
      C2_DS1=(CX1+CFC*Z3/OPZ**TWO)
      DS1D=C1_DS1*DZD+C2_DS1*DPD
      DS1DD=C1_DS1*DZDD+C2_DS1*DPDD
      DS1DTAU=C1_DS1*DZDTAU+C2_DS1*DPDTAU

      C_DS2=CX2*TWO*QB
      DS2D=C_DS2*DQBD
      DS2DD=C_DS2*DQBDD
      DS2DTAU=C_DS2*DQBDTAU

      DFQD=ONE/TWO/FQ*(0.6_q**TWO*Z*DZD+P*DPD)
      DFQDD=ONE/TWO/FQ*(0.6_q**TWO*Z*DZDD+P*DPDD)
      DFQDTAU=ONE/TWO/FQ*(0.6_q**TWO*Z*DZDTAU+P*DPDTAU)
      DS3D=-CX3*(FQ*DQBD+QB*DFQD)
      DS3DD=-CX3*(FQ*DQBDD+QB*DFQDD)
      DS3DTAU=-CX3*(FQ*DQBDTAU+QB*DFQDTAU)

      C_S4=TWO/RKAPPA*CX1**TWO*P
      DS4D=C_S4*DPD
      DS4DD=C_S4*DPDD
      DS4DTAU=C_S4*DPDTAU

      C_S5=FOUR*CFE12*CX1*0.6_q**TWO*Z
      DS5D=C_S5*DZD
      DS5DD=C_S5*DZDD
      DS5DTAU=C_S5*DZDTAU

      C_S6=THREE*CFE*CFMU*P2
      DS6D=C_S6*DPD
      DS6DD=C_S6*DPDD
      DS6DTAU=C_S6*DPDTAU

      C_XD=TWO*(ONE+CFE12*P)*CFE12
      DXD_D=C_XD*DPD
      DXD_DD=C_XD*DPDD
      DXD_DTAU=C_XD*DPDTAU

      DXN_D=DS1D+DS2D+DS3D+DS4D+DS5D+DS6D
      DXN_DD=DS1DD+DS2DD+DS3DD+DS4DD+DS5DD+DS6DD
      DXN_DTAU=DS1DTAU+DS2DTAU+DS3DTAU+DS4DTAU+DS5DTAU+DS6DTAU

      DX_D=(XD*DXN_D-XN*DXD_D)/XD**TWO
!     DX_DD=(XD*DXN_DD-XN*DXD_DD)/XD**TWO
      DX_DD=DXN_DD/XD-X*DXD_DD/XD
      DX_DTAU=(XD*DXN_DTAU-XN*DXD_DTAU)/XD**TWO

      C_FX=ONE/(ONE+X/RKAPPA)**TWO
      DFX_D=C_FX*DX_D
      DFX_DD=C_FX*DX_DD
      DFX_DTAU=C_FX*DX_DTAU

!   OUTPUT THE VXD1,VXDD1 AND AMUXD1
      VXD1=EXDLDA*FX+EXLDA*DFX_D
      VXDD1=EXLDA*DFX_DD
      AMUXD1=EXLDA*DFX_DTAU
      EX_REVTPSS=0._q
      EX_REVTPSS=EX_REVTPSS+EXLDA*FX

! spin down
! IN EXD1(2*RD), TAUWD AND TAUD SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RD
      DRHO=TWO*DRD
      TAUW_RHO=TWO*TAUWD_TEMP
      TAU_RHO=TWO*TAUD

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|

      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P
      Z=TAUW_RHO/TAU_RHO
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
         DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
         Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test     
      Z2=Z*Z
      Z3=Z2*Z
      ZTF=0.6_q*Z
      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5

!     ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      ALPHA=5._q/3._q*P*(1._q/Z-1._q)
      
      QB_NUM=9._q/20._q*(ALPHA-ONE)
      QB_DEN=SQRT(ONE+CFB*ALPHA*(ALPHA-ONE))
      QB=QB_NUM/QB_DEN+ TWO*P/THREE
      OPZ=ONE+Z2
      SFP=Z3/OPZ**TWO
      FQ=SQRT((ZTF**TWO+P2)/TWO)
      S1=(CX1+CFC*SFP)*P
      S2=CX2*QB**TWO
      S3=-CX3*FQ*QB
      S4=(CX1*P)**TWO/RKAPPA
      S5=TWO*CFE12*CX1*ZTF**TWO
      S6=CFE*CFMU*P2*P
      XN=S1+S2+S3+S4+S5+S6
      XD=(ONE+CFE12*P)**TWO
      X=XN/XD

!    GET THE VALUE FOR FX
      FX=ONE+RKAPPA-RKAPPA/(ONE+(X/RKAPPA))

!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

      DALPHAD=THRD5*(ONE/Z-ONE)*DPD-THRD5*P*DZD/Z**TWO
!     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   & 
!    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

      DALPHADD=THRD5*(ONE/Z-ONE)*DPDD-THRD5*P*DZDD/Z**TWO
!     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

      DALPHADTAU=-THRD5*P*DZDTAU/Z**TWO
!     DALPHADTAU=ONE/TAU_UNIF

      DQB_NUM_D=9._q/20._q*DALPHAD
      DQB_NUM_DD=9._q/20._q*DALPHADD
      DQB_NUM_DTAU=9._q/20._q*DALPHADTAU
      C_DQB_DEN=ONE/TWO*CFB*(TWO*ALPHA-ONE)/QB_DEN
      DQB_DEN_D=C_DQB_DEN*DALPHAD
      DQB_DEN_DD=C_DQB_DEN*DALPHADD
      DQB_DEN_DTAU=C_DQB_DEN*DALPHADTAU

      DQBD=(DQB_NUM_D*QB_DEN-QB_NUM*DQB_DEN_D)/QB_DEN**TWO+THRD2*DPD
      DQBDD=(DQB_NUM_DD*QB_DEN-QB_NUM*DQB_DEN_DD)/QB_DEN**TWO+THRD2*DPDD
      DQBDTAU=(DQB_NUM_DTAU*QB_DEN-QB_NUM*DQB_DEN_DTAU)/QB_DEN**TWO+THRD2*DPDTAU

      C1_DS1=P*Z2*(THREE*CFC-CFC*Z2)/OPZ**THREE
      C2_DS1=(CX1+CFC*Z3/OPZ**TWO)
      DS1D=C1_DS1*DZD+C2_DS1*DPD
      DS1DD=C1_DS1*DZDD+C2_DS1*DPDD
      DS1DTAU=C1_DS1*DZDTAU+C2_DS1*DPDTAU

      C_DS2=CX2*TWO*QB
      DS2D=C_DS2*DQBD
      DS2DD=C_DS2*DQBDD
      DS2DTAU=C_DS2*DQBDTAU

      DFQD=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZD+P*DPD)
      DFQDD=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZDD+P*DPDD)
      DFQDTAU=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZDTAU+P*DPDTAU)
      DS3D=-CX3*(FQ*DQBD+QB*DFQD)
      DS3DD=-CX3*(FQ*DQBDD+QB*DFQDD)
      DS3DTAU=-CX3*(FQ*DQBDTAU+QB*DFQDTAU)

      C_S4=TWO/RKAPPA*CX1**TWO*P
      DS4D=C_S4*DPD
      DS4DD=C_S4*DPDD
      DS4DTAU=C_S4*DPDTAU

      C_S5=FOUR*CFE12*CX1*0.6_q**TWO*Z
      DS5D=C_S5*DZD
      DS5DD=C_S5*DZDD
      DS5DTAU=C_S5*DZDTAU

      C_S6=THREE*CFE*CFMU*P2
      DS6D=C_S6*DPD
      DS6DD=C_S6*DPDD
      DS6DTAU=C_S6*DPDTAU

      C_XD=TWO*(ONE+CFE12*P)*CFE12
      DXD_D=C_XD*DPD
      DXD_DD=C_XD*DPDD
      DXD_DTAU=C_XD*DPDTAU

      DXN_D=DS1D+DS2D+DS3D+DS4D+DS5D+DS6D
      DXN_DD=DS1DD+DS2DD+DS3DD+DS4DD+DS5DD+DS6DD
      DXN_DTAU=DS1DTAU+DS2DTAU+DS3DTAU+DS4DTAU+DS5DTAU+DS6DTAU

      DX_D=(XD*DXN_D-XN*DXD_D)/XD**TWO
!     DX_DD=(XD*DXN_DD-XN*DXD_DD)/XD**TWO
      DX_DD=DXN_DD/XD-X*DXD_DD/XD
      DX_DTAU=(XD*DXN_DTAU-XN*DXD_DTAU)/XD**TWO

      C_FX=ONE/(ONE+X/RKAPPA)**TWO
      DFX_D=C_FX*DX_D
      DFX_DD=C_FX*DX_DD
      DFX_DTAU=C_FX*DX_DTAU

!   OUTPUT THE VXD2,VXDD2 AND AMUXD2
      VXD2=EXDLDA*FX+EXLDA*DFX_D
      VXDD2=EXLDA*DFX_DD
      AMUXD2=EXLDA*DFX_DTAU

      EX_REVTPSS=EX_REVTPSS+EXLDA*FX
      EX_REVTPSS=EX_REVTPSS/TWO
      RETURN
      END SUBROUTINE VrevTPSSx


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VCD1 VCD2                    THE DERIVATIVES OF EC WRT n 
! VCDD1,VCDD2                  THE DERIVATIVES OF EC WRT |grad n| 
! AMUCD1, AMUCD2                   THE DERIVATIVES OF EC WRT TAU
!
!***********************************************************************

      SUBROUTINE VrevTPSSc(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_revTPSS,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the correlatin part
      PARAMETER (CFD=2.8_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
! test
      PARAMETER (thresh=1.e-10_q)
      PARAMETER (xorder=12._q)
      PARAMETER (zinfinity=2._q)
! test

      VCD1=0._q;VCD2=0._q;
      VCDD1=0._q;VCDD2=0._q;
      AMUCD1=0._q;AMUCD2=0._q;

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
!    YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q
      TAUW=1._q/8._q*(Y/RT)
      TAU=TAUU+TAUD

!     TAUW=MIN(TAUW,TAU)

      Z=TAUW/TAU
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test
      Z2=Z*Z
      Z3=Z2*Z

      CALL COE_RPKZB(RU,RD,DRU,DRD,DRT,CRPKZB,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2)

      CALL CORGGA_REVTPSS(RU,RD,DRU,DRD,DRT,EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

! EPPGGA2_D1, EPPGGA1_D2 EPPGGA1_DD2 AND EPPGGA2_DD1 ARE ZEROS
      CALL CORGGA_REVTPSS(RU,0._q,DRU,0._q,DRU,EPPGGA1,EPPGGA1_D1,EPPGGA1_D2,EPPGGA1_DD1,EPPGGA1_DD2)
      EPPGGA1_D2=0._q
      EPPGGA1_DD2=0._q

      CALL CORGGA_REVTPSS(RD,0._q,DRD,0._q,DRD,EPPGGA2,EPPGGA2_D2,EPPGGA2_D1,EPPGGA2_DD2,EPPGGA2_DD1)
      EPPGGA2_D1=0._q
      EPPGGA2_DD1=0._q

      IF (EPPGGA1 .LT. EPPGGA)THEN
          EPPGGA1=EPPGGA
          EPPGGA1_D1=EPPGGA_D1
          EPPGGA1_D2=EPPGGA_D2
          EPPGGA1_DD1=EPPGGA_DD1
          EPPGGA1_DD2=EPPGGA_DD2
      ENDIF

      IF (EPPGGA2 .LT. EPPGGA)THEN
          EPPGGA2=EPPGGA
          EPPGGA2_D1=EPPGGA_D1
          EPPGGA2_D2=EPPGGA_D2
          EPPGGA2_DD1=EPPGGA_DD1
          EPPGGA2_DD2=EPPGGA_DD2
      ENDIF

      WEIRD=(RU*EPPGGA1+RD*EPPGGA2)/RT
      WEIRD_D1=RD*EPPGGA1/RT**TWO+RU/RT*EPPGGA1_D1-RD*EPPGGA2/RT**TWO+RD/RT*EPPGGA2_D1
      WEIRD_D2=-RU*EPPGGA1/RT**TWO+RU/RT*EPPGGA1_D2+RU*EPPGGA2/RT**TWO+RD/RT*EPPGGA2_D2
      WEIRD_DD1=RU/RT*EPPGGA1_DD1+RD/RT*EPPGGA2_DD1
      WEIRD_DD2=RU/RT*EPPGGA1_DD2+RD/RT*EPPGGA2_DD2

      EPSC_REVPKZB=EPPGGA*(ONE+CRPKZB*Z2)-(ONE+CRPKZB)*Z2*WEIRD

      FP_D_EPSC_REVPKZB_D1=(ONE+CRPKZB*Z2)*EPPGGA_D1+EPPGGA*(D_CRPKZB_D1*Z2+TWO*Z*CRPKZB*DZD1)
      FP_D_EPSC_REVPKZB_D2=(ONE+CRPKZB*Z2)*EPPGGA_D2+EPPGGA*(D_CRPKZB_D2*Z2+TWO*Z*CRPKZB*DZD2)

      SD_D_EPSC_REVPKZB_D1=D_CRPKZB_D1*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZD1*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_D1
      SD_D_EPSC_REVPKZB_D2=D_CRPKZB_D2*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZD2*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_D2

      D_EPSC_REVPKZB_D1=FP_D_EPSC_REVPKZB_D1-SD_D_EPSC_REVPKZB_D1
      D_EPSC_REVPKZB_D2=FP_D_EPSC_REVPKZB_D2-SD_D_EPSC_REVPKZB_D2

      FP_D_EPSC_REVPKZB_DD1=(ONE+CRPKZB*Z2)*EPPGGA_DD1+EPPGGA*(D_CRPKZB_DD1*Z2+TWO*Z*CRPKZB*DZDD1)
      FP_D_EPSC_REVPKZB_DD2=(ONE+CRPKZB*Z2)*EPPGGA_DD2+EPPGGA*(D_CRPKZB_DD2*Z2+TWO*Z*CRPKZB*DZDD2)
      SD_D_EPSC_REVPKZB_DD1=D_CRPKZB_DD1*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZDD1*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_DD1
      SD_D_EPSC_REVPKZB_DD2=D_CRPKZB_DD2*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZDD2*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_DD2

      D_EPSC_REVPKZB_DD1=FP_D_EPSC_REVPKZB_DD1-SD_D_EPSC_REVPKZB_DD1
      D_EPSC_REVPKZB_DD2=FP_D_EPSC_REVPKZB_DD2-SD_D_EPSC_REVPKZB_DD2

      D_EPSC_REVPKZB_DTAU=EPPGGA*CRPKZB*TWO*Z*DZDTAU-(ONE+CRPKZB)*TWO*Z*DZDTAU*WEIRD

      VCD1=(EPSC_REVPKZB+RT*D_EPSC_REVPKZB_D1)*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_D1+THREE*Z2*EPSC_REVPKZB*DZD1)

      VCD2=(EPSC_REVPKZB+RT*D_EPSC_REVPKZB_D2)*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_D2+THREE*Z2*EPSC_REVPKZB*DZD2)

      VCDD1=RT*D_EPSC_REVPKZB_DD1*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DD1+THREE*Z2*EPSC_REVPKZB*DZDD1)

      VCDD2=RT*D_EPSC_REVPKZB_DD2*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DD2+THREE*Z2*EPSC_REVPKZB*DZDD2)

      AMUCD1=RT*D_EPSC_REVPKZB_DTAU*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DTAU+THREE*Z2*EPSC_REVPKZB*DZDTAU)
      AMUCD2=AMUCD1

      EC_REVTPSS=RT*EPSC_REVPKZB*(ONE+CFD*EPSC_REVPKZB*Z3)
      RETURN
      END SUBROUTINE VrevTPSSC


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! CRPKZB                       C(ZETA,XI)
! D_CRPKZB_D1,D_CRPKZB_D2      THE DERIVATIVE OF C(ZETA,XI) WRT n
! D_CRPKZB_DD1,D_CRPKZB_DD2    THE DERIVATIVE OF C(ZETA,XI) WRT |GRAD n|
!
!***********************************************************************

      SUBROUTINE COE_RPKZB(&
     &   RU,RD,DRU,DRD,DRT, &
     &   CRPKZB,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the correlatin part
      PARAMETER (CF1=0.59_q)
      PARAMETER (CF2=0.9269_q)
      PARAMETER (CF3=0.6225_q)
      PARAMETER (CF4=2.1540_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)

      CRPKZB=0._q
      D_CRPKZB_D1=0._q;D_CRPKZB_D2=0._q;
      D_CRPKZB_DD1=0._q;D_CRPKZB_DD2=0._q;

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
! YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
      ZETA2=ZETA*ZETA
      OPZETA=ONE/(ONE+ZETA)
      OMZETA=ONE/(ONE-ZETA)
      D_ZETA_D1=TWO*RD/RT**TWO
      D_ZETA_D2=-TWO*RU/RT**TWO

! |GRAD ZETA|, 
! IN SOME EXTREME CASES, THE TERM (YA*RD**TWO-TWO*RU*RD*YC+YB*RU**TWO)
! GOES TO NEGATIVE BECAUSE OF THE NUMERICAL PRESICION OF COMPUTERS. SO, WE USE 2/4 AS THE POWER
! INSTEAD OF 1/2
     
      DEL_ZETA=TWO/RT**TWO*((YA*RD**TWO-TWO*RU*RD*YC+YB*RU**TWO)**TWO)**(ONE/FOUR)
      DEL_ZETA=MAX(DEL_ZETA,1E-10_q)
      D_DEL_ZETA_D1=-TWO*DEL_ZETA/RT+TWO/(RT**FOUR*DEL_ZETA)*(-TWO*YC*RD+TWO*RU*YB)
      D_DEL_ZETA_D2=-TWO*DEL_ZETA/RT+TWO/(RT**FOUR*DEL_ZETA)*(-TWO*YC*RU+TWO*RD*YA)
      D_DEL_ZETA_DD1=TWO/(RT**FOUR*DEL_ZETA)*(TWO*RD**TWO*DRU-TWO*RU*RD*YC/DRU)
      D_DEL_ZETA_DD2=TWO/(RT**FOUR*DEL_ZETA)*(TWO*RU**TWO*DRD-TWO*RU*RD*YC/DRD)

! XI
      XIDEN=TWO*(THREE*PISQ*RT)**THRD
      XI=DEL_ZETA/XIDEN
      D_XI_D1=(D_DEL_ZETA_D1-THRD*DEL_ZETA/RT)/XIDEN
      D_XI_D2=(D_DEL_ZETA_D2-THRD*DEL_ZETA/RT)/XIDEN
      D_XI_DD1=D_DEL_ZETA_DD1/XIDEN
      D_XI_DD2=D_DEL_ZETA_DD2/XIDEN

      CNUM=CF1+(CF2+(CF3+CF4*ZETA2)*ZETA2)*ZETA2
      D_CNUM_D1=D_ZETA_D1*(TWO*CF2+(FOUR*CF3+SIX*CF4*ZETA2)*ZETA2)*ZETA
      D_CNUM_D2=D_ZETA_D2*(TWO*CF2+(FOUR*CF3+SIX*CF4*ZETA2)*ZETA2)*ZETA

      OPZETA43=OPZETA**THRD4
      OMZETA43=OMZETA**THRD4
      CDENL=ONE+XI**TWO*(OPZETA43+OMZETA43)/TWO
      CDEN=CDENL**FOUR
      D_CDENL_DXI=XI*(OPZETA43+OMZETA43)
      D_CDENL_DZETA=-THRD2*XI**TWO*(OPZETA43*OPZETA-OMZETA43*OMZETA)
      D_CDENL_D1=D_CDENL_DXI*D_XI_D1+D_CDENL_DZETA*D_ZETA_D1
      D_CDENL_D2=D_CDENL_DXI*D_XI_D2+D_CDENL_DZETA*D_ZETA_D2
      D_CDENL_DD1=D_CDENL_DXI*D_XI_DD1
      D_CDENL_DD2=D_CDENL_DXI*D_XI_DD2

      D_CDEN_D1=FOUR*CDENL**THREE*D_CDENL_D1
      D_CDEN_D2=FOUR*CDENL**THREE*D_CDENL_D2
      D_CDEN_DD1=FOUR*CDENL**THREE*D_CDENL_DD1
      D_CDEN_DD2=FOUR*CDENL**THREE*D_CDENL_DD2
      
! OUTPUT CRPKZB,,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2
      CRPKZB=CNUM/CDEN
      D_CRPKZB_D1=(CDEN*D_CNUM_D1-CNUM*D_CDEN_D1)/CDEN**TWO
      D_CRPKZB_D2=(CDEN*D_CNUM_D2-CNUM*D_CDEN_D2)/CDEN**TWO
      D_CRPKZB_DD1=-CNUM*D_CDEN_DD1/CDEN**TWO
      D_CRPKZB_DD2=-CNUM*D_CDEN_DD2/CDEN**TWO
      RETURN
      END SUBROUTINE COE_RPKZB


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! EPPGGA                       GGA ENERGY PER PARTICLE
! EPPGGA_D1,EPPGGA_D2          THE DERIVATIVE OF EPPGGA WRT n
! EPPGGA_DD1,EPPGGA_DD2        THE DERIVATIVE OF EPPGGA WRT |GRAD n|
!
!***********************************************************************
      SUBROUTINE CORGGA_REVTPSS(&
     &   RU,RD,DRU,DRD,DRT, &
     &   EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

      USE prec
      USE constant

      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
  
      PARAMETER (GAMMA=0.03109069086965489503494086371273_q)
      PARAMETER (BETA_mb=0.06672455060314922_q)

      EPPGGA  =0._q
      EPPGGA_D1=0._q
      EPPGGA_D2=0._q
      EPPGGA_DD1=0._q
      EPPGGA_DD2=0._q

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
!     YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
      DZETAD1=TWO*RD/RT**TWO
      DZETAD2=-TWO*RU/RT**TWO


      DTHRD=exp(log(RT)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD
      DRSD1=-THRD/RT*RS
      DRSD2=-THRD/RT*RS


      PHI = (exp((TWO*THRD)*log(1._q+ZETA)) &
                +exp((TWO*THRD)*log(1._q-ZETA)))/2._q
      D_PHI_DZETA=THRD*((ONE+ZETA)**(-THRD)-(ONE-ZETA)**(-THRD))
      D_PHI_D1=D_PHI_DZETA*DZETAD1
      D_PHI_D2=D_PHI_DZETA*DZETAD2

      AFIX_T=SQRT(PI/FOUR)*(9._q*PI/FOUR)**(ONE/SIX)
      S=DRT/(TWO*(THREE*PISQ)**THRD*RT**THRD4)
      DSD1=-THRD4*S/RT
      DSD2=-THRD4*S/RT
! |GRAD N|/|GRAD N(UP OR DOWN)|
      GNGNU=ONE/(TWO*DRT)*(TWO*DRU+TWO*YC/DRU)
      GNGND=ONE/(TWO*DRT)*(TWO*DRD+TWO*YC/DRD)
      DSDD1=S/DRT*GNGNU
      DSDD2=S/DRT*GNGND

      T=AFIX_T*S/SQRT(RS)/PHI
      T2 = T*T
      T4 = T2*T2

      DTD1=AFIX_T*(PHI*RS*DSD1-0.5_q*S*PHI*DRSD1-S*RS*D_PHI_D1)/(PHI**TWO*RS**(THREE/TWO))
      DTD2=AFIX_T*(PHI*RS*DSD2-0.5_q*S*PHI*DRSD2-S*RS*D_PHI_D2)/(PHI**TWO*RS**(THREE/TWO))
      DTDD1=AFIX_T*DSDD1/(SQRT(RS)*PHI)
      DTDD2=AFIX_T*DSDD2/(SQRT(RS)*PHI)

      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)
! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
      CALL CORPBE_revtpss(RS,ZETA,EC,VCUPLDA,VCDNLDA,PHI,SK, &
           T,H,DVCUP,DVCDN,ECQ,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=EC
!      EPPGGA_D1=(VCUPLDA-EC)/RT
!      EPPGGA_D2=(VCDNLDA-EC)/RT
      D_EC_DZETA=(VCUPLDA-VCDNLDA)/TWO
      D_EC_DRS=(EC-ZETA*D_EC_DZETA-(VCUPLDA+VCDNLDA)/TWO)*THREE/RS

! THE RS DEPENDENCE OF BETA
      AFACTOR=0.1_q
      BFACTOR=0.1778_Q
      BETA_NUM=ONE + AFACTOR*RS
      BETA_DEN=ONE+ BFACTOR*RS
      D_BETA_NUM=AFACTOR
      D_BETA_DEN=BFACTOR
      BETA = BETA_MB*BETA_NUM/BETA_DEN
      D_BETA_DRS=BETA_MB*(BETA_DEN*D_BETA_NUM-BETA_NUM*D_BETA_DEN)/BETA_DEN**TWO
      
 
      PHI3=PHI**THREE
      PON=-EC/(PHI3*gamma)
      W=DEXP(PON)-ONE
      D_W_DRS=-(W+ONE)*D_EC_DRS/(GAMMA*PHI3)
      D_W_DZETA=-(W+ONE)/(GAMMA*PHI3)*(D_EC_DZETA-THREE*EC*D_PHI_DZETA/PHI)
      D_W_DT=0._q


      A=BETA/(GAMMA*W)
      D_A_DRS=(W*D_BETA_DRS-BETA*D_W_DRS)/(GAMMA*W**TWO)
      D_A_DZETA=-BETA*D_W_DZETA/(GAMMA*W**TWO)
      D_A_DT=0._q

      V=A*T2
      D_V_DRS=T2*D_A_DRS
      D_V_DZETA=T2*D_A_DZETA
      D_V_DT=T2*D_A_DT+TWO*A*T
        
      FUNKG=ONE/(ONE+V+V**TWO)
      D_FUNKG_DV=-(ONE+TWO*V)*FUNKG**TWO
      
      
      HCORE=ONE+W*(ONE-FUNKG)
      AH=GAMMA*PHI3
      H=AH*DLOG(HCORE)
   
      DH1=ONE-FUNKG
      DH2=W*D_FUNKG_DV
      D_H_DRS=(DH1*D_W_DRS-DH2*D_V_DRS)*AH/HCORE
      D_H_DZETA=THREE*H*D_PHI_DZETA/PHI+(DH1*D_W_DZETA-DH2*D_V_DZETA)*AH/HCORE
      D_H_DT=(DH1*D_W_DT-DH2*D_V_DT)*AH/HCORE

! OUTPUT EPPGGA AND ITS DERIVATIVES EPPGGA_D1,EPPGGA_D2, EPPGGA_DD1, EPPGGA_DD2
      EPPGGA=EC+H
      EPPGGA_D1=D_EC_DRS*DRSD1+D_EC_DZETA*DZETAD1+D_H_DRS*DRSD1+D_H_DZETA*DZETAD1+D_H_DT*DTD1
      EPPGGA_D2=D_EC_DRS*DRSD2+D_EC_DZETA*DZETAD2+D_H_DRS*DRSD2+D_H_DZETA*DZETAD2+D_H_DT*DTD2
      EPPGGA_DD1=D_H_DT*DTDD1
      EPPGGA_DD2=D_H_DT*DTDD2

      RETURN
      END SUBROUTINE CORGGA_REVTPSS


      SUBROUTINE GGASPINCOR_revTPSS(D1,D2,DDA,EC)
!     D1   density up
!     D2   density down
!     DDA  |gradient of the total density|

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (THRD=1._q/3._q)

      D=D1+D2
      DTHRD=exp(log(D)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD

      ZETA=(D1-D2)/D
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)

      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)
      G = (exp((2*THRD)*log(1._q+ZETA)) &
                +exp((2*THRD)*log(1._q-ZETA)))/2._q
      T = DDA/(D*2._q*SK*G)

      CALL CORPBE_revtpss(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA,G,SK, &
           T,EC,ECD1,ECD2,ECQ,.TRUE.)

      EC  =(EC  +ECLDA)
      RETURN
      END SUBROUTINE GGASPINCOR_revTPSS


! Modified by Adrienn Ruzsinszky
!######################################################################
!----------------------------------------------------------------------
      SUBROUTINE CORPBE_revtpss(RS,ZET,EC,VCUP,VCDN,g,sk, &
     &                  T,H,DVCUP,DVCDN,ecdd,lgga)
!----------------------------------------------------------------------
!  Official PBE correlation code. K. Burke, May 14, 1996.
!  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3)
!       : ZET=RELATIVE SPIN POLARIZATION = (rhoup-rhodn)/rho
!       : t=ABS(GRAD rho)/(rho*2.*KS*G)  -- only needed for PBE
!       : lgga=flag to do gga (0=>LSD only)
!       : lmetagga=flag to do metagga (revTPSS)
!       : lpot=flag to do potential (0=>energy only)
!  output: ec=lsd correlation energy from [a]
!        : vcup=lsd up correlation potential
!        : vcdn=lsd dn correlation potential
!        : h=NONLOCAL PART OF CORRELATION ENERGY PER ELECTRON
!        : dvcup=nonlocal correction to vcup
!        : dvcdn=nonlocal correction to vcdn
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! References:
! [a] J.P.~Perdew, K.~Burke, and M.~Ernzerhof, 
!     {\sl Generalized gradient approximation made simple}, sub.
!     to Phys. Rev.Lett. May 1996.
! [b] J. P. Perdew, K. Burke, and Y. Wang, {\sl Real-space cutoff
!     construction of a generalized gradient approximation:  The PW91
!     density functional}, submitted to Phys. Rev. B, Feb. 1996.
! [c] J. P. Perdew and Y. Wang, Phys. Rev. B {\bf 45}, 13244 (1992).
!----------------------------------------------------------------------
!----------------------------------------------------------------------
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      logical lgga
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.      
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet_mb=0.06672455060314922_q)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor2(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      CALL gcor2(0.01554535_q,0.20548_q,14.1189_q,6.1977_q,3.3662_q, &
     &    0.62517_q,rtRS,EP,EPRS)
      CALL gcor2(0.0168869_q,0.11125_q,10.357_q,3.6231_q,0.88026_q, &
     &    0.49671_q,rtRS,ALFM,ALFRSM)
      ALFC = -ALFM
      Z4 = ZET**4
      F=((1._q+ZET)**THRD4+(1._q-ZET)**THRD4-2._q)/GAM
      EC = EU*(1._q-F*Z4)+EP*F*Z4-ALFM*F*(1._q-Z4)/FZZ
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! LSD potential from [c](A1)
! ECRS = dEc/drs [c](A2)
! ECZET=dEc/dzeta [c](A3)
! FZ = dF/dzeta [c](A4)
      ECRS = EURS*(1._q-F*Z4)+EPRS*F*Z4-ALFRSM*F*(1._q-Z4)/FZZ
      FZ = THRD4*((1._q+ZET)**THRD-(1._q-ZET)**THRD)/GAM
      ECZET = 4._q*(ZET**3)*F*(EP-EU+ALFM/FZZ)+FZ*(Z4*EP-Z4*EU &
     &        -(1._q-Z4)*ALFM/FZZ)
      COMM = EC -RS*ECRS/3._q-ZET*ECZET
      VCUP = COMM + ECZET
      VCDN = COMM - ECZET
!write(*,*)'rs,VCUP,VCDN',rs,VCUP,VCDN
      if (.not.lgga) return
!----------------------------------------------------------------------
! PBE correlation energy
! G=phi(zeta), given after [a](3)
! DELT=bet/gamma
! B=A of [a](8)
      bet = bet_mb*(1._q + 0.1_q*RS)/(1._q + 0.1778_q*RS)

      delt=bet/gamma
      G3 = G**3
      PON=-EC/(G3*gamma)
      B = DELT/(DEXP(PON)-1._q)
      B2 = B*B
      T2 = T*T
      T4 = T2*T2
      RS2 = RS*RS
      RS3 = RS2*RS
      Q4 = 1._q+B*T2
      Q5 = 1._q+B*T2+B2*T4
      H = G3*(BET/DELT)*DLOG(1._q+DELT*Q4*T2/Q5)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! ENERGY DONE. NOW THE POTENTIAL, using appendix E of [b].
      G4 = G3*G
      T6 = T4*T2
      RSTHRD = RS/3._q
      GZ=(((1._q+zet)**2+eta)**sixthm- &
     &((1._q-zet)**2+eta)**sixthm)/3._q
      FAC = DELT/B+1._q
      BG = -3._q*B2*EC*FAC/(BET*G4)
      BEC = B2*FAC/(BET*G3)
      Q8 = Q5*Q5+DELT*Q4*Q5*T2
      Q9 = 1._q+2._q*B*T2
      hB = -BET*G3*B*T6*(2._q+B*T2)/Q8
      hRS = -RSTHRD*hB*BEC*ECRS
      hZ = 3._q*GZ*h/G + hB*(BG*GZ+BEC*ECZET)
      hT = 2._q*BET*G3*Q9/Q8
      COMM = H+HRS-7.0_q*T2*HT/6._q
      PREF = HZ-GZ*T2*HT/G
      COMM = COMM-PREF*ZET
      DVCUP = COMM + PREF
      DVCDN = COMM - PREF
      ecdd=0.5_q/(sk*g)*t*ht
      RETURN
      END SUBROUTINE CORPBE_revtpss


!************************ SUBROUTINE VTPSSx ****************************
!
! calculates the first order derivatives of Ex wrt n and |grad(n)|
! Perdew et. al. PRL (2009)
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! everything in Hartree units
!
! ATTANTION: Every values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VXD1 VXD2                    THE DERIVATIVES OF EX WRT n 
! VXDD1,VXDD2                  THE DERIVATIVES OF EX WRT |grad n| 
! AMUXD1, AMUXD2               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VTPSSx(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ex_revTPSS,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the exchange part
      PARAMETER (RKAPPA=0.804_q)
      PARAMETER (CFB=0.40_q)
      PARAMETER (CFC=1.59096_q)
      PARAMETER (CFE=1.537_q)
      PARAMETER (CFMU=0.21951_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (AX=-0.738558766382022405884230032680836_q)
! test
      PARAMETER (thresh=1.e-10_q)
      PARAMETER (xorder=12._q)
      PARAMETER (zinfinity=2._q)
! test

      VXD1=0._q;VXD2=0._q;
      VXDD1=0._q;VXDD2=0._q;
      AMUXD1=0._q;AMUXD2=0._q;


      CX1=10._q/81._q
      CX2=146._q/2025._q
      CX3=73._q/405._q
      CFE12=SQRT(CFE)

! Suspect that TAUWU and TAUWD are not well described. Use TAUW_TEMP=0.125_q*DRT**2/RT
! IF WANT TO TEST TAUW, SIMPLY OVERWRITE TAUW_TEMP BY TAUW
      TAUWU_TEMP=0.125_q*DRU**2._q/RU
!     TAUWU_TEMP=MIN(TAUWU_TEMP,TAUU)
      
      TAUWD_TEMP=0.125_q*DRD**2._q/RD
!     TAUWD_TEMP=MIN(TAUWD_TEMP,TAUD)

! spin up
! IN EXD1(2*RU), TAUWU AND TAUU SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RU
      DRHO=TWO*DRU
      TAUW_RHO=TWO*TAUWU_TEMP
      TAU_RHO=TWO*TAUU

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|
      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P
      Z=TAUW_RHO/TAU_RHO
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2 
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test      
      Z2=Z*Z
      Z3=Z2*Z
      ZTF=0.6_q*Z
      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5
      
!     ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      ALPHA=5._q/3._q*P*(1._q/Z-1._q)
      
      QB_NUM=9._q/20._q*(ALPHA-ONE)
      QB_DEN=SQRT(ONE+CFB*ALPHA*(ALPHA-ONE))
      QB=QB_NUM/QB_DEN+ TWO*P/THREE
      OPZ=ONE+Z2
      SFP=Z2/OPZ**TWO
      FQ=SQRT((ZTF**TWO+P2)/TWO)
      S1=(CX1+CFC*SFP)*P
      S2=CX2*QB**TWO
      S3=-CX3*FQ*QB
      S4=(CX1*P)**TWO/RKAPPA
      S5=TWO*CFE12*CX1*ZTF**TWO
      S6=CFE*CFMU*P2*P
      XN=S1+S2+S3+S4+S5+S6
      XD=(ONE+CFE12*P)**TWO
      X=XN/XD

!    GET THE VALUE FOR FX
      FX=ONE+RKAPPA-RKAPPA/(ONE+(X/RKAPPA))

!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

      DALPHAD=THRD5*(ONE/Z-ONE)*DPD-THRD5*P*DZD/Z**TWO
!     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   &  
!    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

      DALPHADD=THRD5*(ONE/Z-ONE)*DPDD-THRD5*P*DZDD/Z**TWO
!     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

      DALPHADTAU=-THRD5*P*DZDTAU/Z**TWO
!     DALPHADTAU=ONE/TAU_UNIF

      DQB_NUM_D=9._q/20._q*DALPHAD
      DQB_NUM_DD=9._q/20._q*DALPHADD
      DQB_NUM_DTAU=9._q/20._q*DALPHADTAU
      C_DQB_DEN=ONE/TWO*CFB*(TWO*ALPHA-ONE)/QB_DEN
      DQB_DEN_D=C_DQB_DEN*DALPHAD
      DQB_DEN_DD=C_DQB_DEN*DALPHADD
      DQB_DEN_DTAU=C_DQB_DEN*DALPHADTAU

      DQBD=(DQB_NUM_D*QB_DEN-QB_NUM*DQB_DEN_D)/QB_DEN**TWO+THRD2*DPD
      DQBDD=(DQB_NUM_DD*QB_DEN-QB_NUM*DQB_DEN_DD)/QB_DEN**TWO+THRD2*DPDD
      DQBDTAU=(DQB_NUM_DTAU*QB_DEN-QB_NUM*DQB_DEN_DTAU)/QB_DEN**TWO+THRD2*DPDTAU

      C1_DS1=TWO*CFC*P*Z*(ONE-Z2)/OPZ**THREE
      C2_DS1=(CX1+CFC*Z2/OPZ**TWO)
      DS1D=C1_DS1*DZD+C2_DS1*DPD
      DS1DD=C1_DS1*DZDD+C2_DS1*DPDD
      DS1DTAU=C1_DS1*DZDTAU+C2_DS1*DPDTAU

      C_DS2=CX2*TWO*QB
      DS2D=C_DS2*DQBD
      DS2DD=C_DS2*DQBDD
      DS2DTAU=C_DS2*DQBDTAU

      DFQD=ONE/TWO/FQ*(0.6_q**TWO*Z*DZD+P*DPD)
      DFQDD=ONE/TWO/FQ*(0.6_q**TWO*Z*DZDD+P*DPDD)
      DFQDTAU=ONE/TWO/FQ*(0.6_q**TWO*Z*DZDTAU+P*DPDTAU)
      DS3D=-CX3*(FQ*DQBD+QB*DFQD)
      DS3DD=-CX3*(FQ*DQBDD+QB*DFQDD)
      DS3DTAU=-CX3*(FQ*DQBDTAU+QB*DFQDTAU)

      C_S4=TWO/RKAPPA*CX1**TWO*P
      DS4D=C_S4*DPD
      DS4DD=C_S4*DPDD
      DS4DTAU=C_S4*DPDTAU

      C_S5=FOUR*CFE12*CX1*0.6_q**TWO*Z
      DS5D=C_S5*DZD
      DS5DD=C_S5*DZDD
      DS5DTAU=C_S5*DZDTAU

      C_S6=THREE*CFE*CFMU*P2
      DS6D=C_S6*DPD
      DS6DD=C_S6*DPDD
      DS6DTAU=C_S6*DPDTAU

      C_XD=TWO*(ONE+CFE12*P)*CFE12
      DXD_D=C_XD*DPD
      DXD_DD=C_XD*DPDD
      DXD_DTAU=C_XD*DPDTAU

      DXN_D=DS1D+DS2D+DS3D+DS4D+DS5D+DS6D
      DXN_DD=DS1DD+DS2DD+DS3DD+DS4DD+DS5DD+DS6DD
      DXN_DTAU=DS1DTAU+DS2DTAU+DS3DTAU+DS4DTAU+DS5DTAU+DS6DTAU

      DX_D=(XD*DXN_D-XN*DXD_D)/XD**TWO
!     DX_DD=(XD*DXN_DD-XN*DXD_DD)/XD**TWO
      DX_DD=DXN_DD/XD-X*DXD_DD/XD
      DX_DTAU=(XD*DXN_DTAU-XN*DXD_DTAU)/XD**TWO

      C_FX=ONE/(ONE+X/RKAPPA)**TWO
      DFX_D=C_FX*DX_D
      DFX_DD=C_FX*DX_DD
      DFX_DTAU=C_FX*DX_DTAU

!   OUTPUT THE VXD1,VXDD1 AND AMUXD1
      VXD1=EXDLDA*FX+EXLDA*DFX_D
      VXDD1=EXLDA*DFX_DD
      AMUXD1=EXLDA*DFX_DTAU
      EX_REVTPSS=0._q
      EX_REVTPSS=EX_REVTPSS+EXLDA*FX

! spin down
! IN EXD1(2*RD), TAUWD AND TAUD SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RD
      DRHO=TWO*DRD
      TAUW_RHO=TWO*TAUWD_TEMP
      TAU_RHO=TWO*TAUD

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|

      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P
      Z=TAUW_RHO/TAU_RHO
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
         DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
         Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD=-Z/RHO*DZTILDEDZ
        DZDD=DRHO/(FOUR*RHO*TAU_RHO)*DZTILDEDZ
        DZDTAU=-Z/TAU_RHO*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test     
      Z2=Z*Z
      Z3=Z2*Z
      ZTF=0.6_q*Z
      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5

!     ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      ALPHA=5._q/3._q*P*(1._q/Z-1._q)
      
      QB_NUM=9._q/20._q*(ALPHA-ONE)
      QB_DEN=SQRT(ONE+CFB*ALPHA*(ALPHA-ONE))
      QB=QB_NUM/QB_DEN+ TWO*P/THREE
      OPZ=ONE+Z2
      SFP=Z2/OPZ**TWO
      FQ=SQRT((ZTF**TWO+P2)/TWO)
      S1=(CX1+CFC*SFP)*P
      S2=CX2*QB**TWO
      S3=-CX3*FQ*QB
      S4=(CX1*P)**TWO/RKAPPA
      S5=TWO*CFE12*CX1*ZTF**TWO
      S6=CFE*CFMU*P2*P
      XN=S1+S2+S3+S4+S5+S6
      XD=(ONE+CFE12*P)**TWO
      X=XN/XD

!    GET THE VALUE FOR FX
      FX=ONE+RKAPPA-RKAPPA/(ONE+(X/RKAPPA))

!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

      DALPHAD=THRD5*(ONE/Z-ONE)*DPD-THRD5*P*DZD/Z**TWO
!     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   & 
!    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

      DALPHADD=THRD5*(ONE/Z-ONE)*DPDD-THRD5*P*DZDD/Z**TWO
!     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

      DALPHADTAU=-THRD5*P*DZDTAU/Z**TWO
!     DALPHADTAU=ONE/TAU_UNIF

      DQB_NUM_D=9._q/20._q*DALPHAD
      DQB_NUM_DD=9._q/20._q*DALPHADD
      DQB_NUM_DTAU=9._q/20._q*DALPHADTAU
      C_DQB_DEN=ONE/TWO*CFB*(TWO*ALPHA-ONE)/QB_DEN
      DQB_DEN_D=C_DQB_DEN*DALPHAD
      DQB_DEN_DD=C_DQB_DEN*DALPHADD
      DQB_DEN_DTAU=C_DQB_DEN*DALPHADTAU

      DQBD=(DQB_NUM_D*QB_DEN-QB_NUM*DQB_DEN_D)/QB_DEN**TWO+THRD2*DPD
      DQBDD=(DQB_NUM_DD*QB_DEN-QB_NUM*DQB_DEN_DD)/QB_DEN**TWO+THRD2*DPDD
      DQBDTAU=(DQB_NUM_DTAU*QB_DEN-QB_NUM*DQB_DEN_DTAU)/QB_DEN**TWO+THRD2*DPDTAU

      C1_DS1=TWO*CFC*P*Z*(ONE-Z2)/OPZ**THREE
      C2_DS1=(CX1+CFC*Z2/OPZ**TWO)
      DS1D=C1_DS1*DZD+C2_DS1*DPD
      DS1DD=C1_DS1*DZDD+C2_DS1*DPDD
      DS1DTAU=C1_DS1*DZDTAU+C2_DS1*DPDTAU

      C_DS2=CX2*TWO*QB
      DS2D=C_DS2*DQBD
      DS2DD=C_DS2*DQBDD
      DS2DTAU=C_DS2*DQBDTAU

      DFQD=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZD+P*DPD)
      DFQDD=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZDD+P*DPDD)
      DFQDTAU=ONE/TWO/FQ*(0.6_Q**TWO*Z*DZDTAU+P*DPDTAU)
      DS3D=-CX3*(FQ*DQBD+QB*DFQD)
      DS3DD=-CX3*(FQ*DQBDD+QB*DFQDD)
      DS3DTAU=-CX3*(FQ*DQBDTAU+QB*DFQDTAU)

      C_S4=TWO/RKAPPA*CX1**TWO*P
      DS4D=C_S4*DPD
      DS4DD=C_S4*DPDD
      DS4DTAU=C_S4*DPDTAU

      C_S5=FOUR*CFE12*CX1*0.6_q**TWO*Z
      DS5D=C_S5*DZD
      DS5DD=C_S5*DZDD
      DS5DTAU=C_S5*DZDTAU

      C_S6=THREE*CFE*CFMU*P2
      DS6D=C_S6*DPD
      DS6DD=C_S6*DPDD
      DS6DTAU=C_S6*DPDTAU

      C_XD=TWO*(ONE+CFE12*P)*CFE12
      DXD_D=C_XD*DPD
      DXD_DD=C_XD*DPDD
      DXD_DTAU=C_XD*DPDTAU

      DXN_D=DS1D+DS2D+DS3D+DS4D+DS5D+DS6D
      DXN_DD=DS1DD+DS2DD+DS3DD+DS4DD+DS5DD+DS6DD
      DXN_DTAU=DS1DTAU+DS2DTAU+DS3DTAU+DS4DTAU+DS5DTAU+DS6DTAU

      DX_D=(XD*DXN_D-XN*DXD_D)/XD**TWO
!     DX_DD=(XD*DXN_DD-XN*DXD_DD)/XD**TWO
      DX_DD=DXN_DD/XD-X*DXD_DD/XD
      DX_DTAU=(XD*DXN_DTAU-XN*DXD_DTAU)/XD**TWO

      C_FX=ONE/(ONE+X/RKAPPA)**TWO
      DFX_D=C_FX*DX_D
      DFX_DD=C_FX*DX_DD
      DFX_DTAU=C_FX*DX_DTAU

!   OUTPUT THE VXD2,VXDD2 AND AMUXD2
      VXD2=EXDLDA*FX+EXLDA*DFX_D
      VXDD2=EXLDA*DFX_DD
      AMUXD2=EXLDA*DFX_DTAU

      EX_REVTPSS=EX_REVTPSS+EXLDA*FX
      EX_REVTPSS=EX_REVTPSS/TWO
      RETURN
      END SUBROUTINE VTPSSx


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VCD1 VCD2                    THE DERIVATIVES OF EC WRT n 
! VCDD1,VCDD2                  THE DERIVATIVES OF EC WRT |grad n| 
! AMUCD1, AMUCD2                   THE DERIVATIVES OF EC WRT TAU
!
!***********************************************************************

      SUBROUTINE VTPSSc(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_revTPSS,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the correlatin part
      PARAMETER (CFD=2.8_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
! test
      PARAMETER (thresh=1.e-10_q)
      PARAMETER (xorder=12._q)
      PARAMETER (zinfinity=2._q)
! test

      VCD1=0._q;VCD2=0._q;
      VCDD1=0._q;VCDD2=0._q;
      AMUCD1=0._q;AMUCD2=0._q;

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
!    YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q
      TAUW=1._q/8._q*(Y/RT)
      TAU=TAUU+TAUD

      Z=TAUW/TAU
! test
      !Z=max(min(Z,0.9999999999999_q),1.E-20_q)
#ifdef Zmethod1
      IF (Z>1) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=1
      ELSE IF(Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE
        DZTILDEDZ=1
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
      ENDIF
#endif
#ifdef Zmethod2
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+Z**xorder)**(1._q/xorder) - Z**xorder/(1._q+Z**xorder)**(1._q+1._q/xorder)
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=Z/(1._q+Z**xorder)**(1._q/xorder)
      ENDIF
#endif
#ifdef Zmethod3
      IF (Z<0) THEN
        DZTILDEDZ=0
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=0
      ELSE       
        DZTILDEDZ=1._q/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder) - (Z/zinfinity)**xorder/(1._q+(Z/zinfinity)**xorder)**(1._q+1._q/xorder)
        DZD1=-Z/RT*DZTILDEDZ
        DZD2=DZD1
        DYCDD1=YC/DRU
        DYCDD2=YC/DRD
        DZDD1=ONE/FOUR/(RT*TAU)*(DRU+DYCDD1)*DZTILDEDZ
        DZDD2=ONE/FOUR/(RT*TAU)*(DRD+DYCDD2)*DZTILDEDZ
        DZDTAU=-Z/TAU*DZTILDEDZ
        Z=Z/(1._q+(Z/zinfinity)**xorder)**(1._q/xorder)
      ENDIF
#endif
! test
      Z2=Z*Z
      Z3=Z2*Z

      CALL COE_RPKZB_TPSS(RU,RD,DRU,DRD,DRT,CRPKZB,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2)

      CALL CORGGA_TPSS(RU,RD,DRU,DRD,DRT,EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

! EPPGGA2_D1, EPPGGA1_D2 EPPGGA1_DD2 AND EPPGGA2_DD1 ARE ZEROS
      CALL CORGGA_TPSS(RU,0._q,DRU,0._q,DRU,EPPGGA1,EPPGGA1_D1,EPPGGA1_D2,EPPGGA1_DD1,EPPGGA1_DD2)
      EPPGGA1_D2=0._q
      EPPGGA1_DD2=0._q

      CALL CORGGA_TPSS(RD,0._q,DRD,0._q,DRD,EPPGGA2,EPPGGA2_D2,EPPGGA2_D1,EPPGGA2_DD2,EPPGGA2_DD1)
      EPPGGA2_D1=0._q
      EPPGGA2_DD1=0._q

      IF (EPPGGA1 .LT. EPPGGA)THEN
          EPPGGA1=EPPGGA
          EPPGGA1_D1=EPPGGA_D1
          EPPGGA1_D2=EPPGGA_D2
          EPPGGA1_DD1=EPPGGA_DD1
          EPPGGA1_DD2=EPPGGA_DD2
      ENDIF

      IF (EPPGGA2 .LT. EPPGGA)THEN
          EPPGGA2=EPPGGA
          EPPGGA2_D1=EPPGGA_D1
          EPPGGA2_D2=EPPGGA_D2
          EPPGGA2_DD1=EPPGGA_DD1
          EPPGGA2_DD2=EPPGGA_DD2
      ENDIF

      WEIRD=(RU*EPPGGA1+RD*EPPGGA2)/RT
      WEIRD_D1=RD*EPPGGA1/RT**TWO+RU/RT*EPPGGA1_D1-RD*EPPGGA2/RT**TWO+RD/RT*EPPGGA2_D1
      WEIRD_D2=-RU*EPPGGA1/RT**TWO+RU/RT*EPPGGA1_D2+RU*EPPGGA2/RT**TWO+RD/RT*EPPGGA2_D2
      WEIRD_DD1=RU/RT*EPPGGA1_DD1+RD/RT*EPPGGA2_DD1
      WEIRD_DD2=RU/RT*EPPGGA1_DD2+RD/RT*EPPGGA2_DD2

      EPSC_REVPKZB=EPPGGA*(ONE+CRPKZB*Z2)-(ONE+CRPKZB)*Z2*WEIRD

      FP_D_EPSC_REVPKZB_D1=(ONE+CRPKZB*Z2)*EPPGGA_D1+EPPGGA*(D_CRPKZB_D1*Z2+TWO*Z*CRPKZB*DZD1)
      FP_D_EPSC_REVPKZB_D2=(ONE+CRPKZB*Z2)*EPPGGA_D2+EPPGGA*(D_CRPKZB_D2*Z2+TWO*Z*CRPKZB*DZD2)

      SD_D_EPSC_REVPKZB_D1=D_CRPKZB_D1*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZD1*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_D1
      SD_D_EPSC_REVPKZB_D2=D_CRPKZB_D2*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZD2*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_D2

      D_EPSC_REVPKZB_D1=FP_D_EPSC_REVPKZB_D1-SD_D_EPSC_REVPKZB_D1
      D_EPSC_REVPKZB_D2=FP_D_EPSC_REVPKZB_D2-SD_D_EPSC_REVPKZB_D2

      FP_D_EPSC_REVPKZB_DD1=(ONE+CRPKZB*Z2)*EPPGGA_DD1+EPPGGA*(D_CRPKZB_DD1*Z2+TWO*Z*CRPKZB*DZDD1)
      FP_D_EPSC_REVPKZB_DD2=(ONE+CRPKZB*Z2)*EPPGGA_DD2+EPPGGA*(D_CRPKZB_DD2*Z2+TWO*Z*CRPKZB*DZDD2)
      SD_D_EPSC_REVPKZB_DD1=D_CRPKZB_DD1*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZDD1*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_DD1
      SD_D_EPSC_REVPKZB_DD2=D_CRPKZB_DD2*Z2*WEIRD+(ONE+CRPKZB)*TWO*Z*DZDD2*WEIRD+(ONE+CRPKZB)*Z2*WEIRD_DD2

      D_EPSC_REVPKZB_DD1=FP_D_EPSC_REVPKZB_DD1-SD_D_EPSC_REVPKZB_DD1
      D_EPSC_REVPKZB_DD2=FP_D_EPSC_REVPKZB_DD2-SD_D_EPSC_REVPKZB_DD2

      D_EPSC_REVPKZB_DTAU=EPPGGA*CRPKZB*TWO*Z*DZDTAU-(ONE+CRPKZB)*TWO*Z*DZDTAU*WEIRD

      VCD1=(EPSC_REVPKZB+RT*D_EPSC_REVPKZB_D1)*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_D1+THREE*Z2*EPSC_REVPKZB*DZD1)

      VCD2=(EPSC_REVPKZB+RT*D_EPSC_REVPKZB_D2)*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_D2+THREE*Z2*EPSC_REVPKZB*DZD2)

      VCDD1=RT*D_EPSC_REVPKZB_DD1*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DD1+THREE*Z2*EPSC_REVPKZB*DZDD1)

      VCDD2=RT*D_EPSC_REVPKZB_DD2*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DD2+THREE*Z2*EPSC_REVPKZB*DZDD2)

      AMUCD1=RT*D_EPSC_REVPKZB_DTAU*(ONE+CFD*EPSC_REVPKZB*Z3) &
          & +RT*EPSC_REVPKZB*CFD*(Z3*D_EPSC_REVPKZB_DTAU+THREE*Z2*EPSC_REVPKZB*DZDTAU)
      AMUCD2=AMUCD1

      EC_REVTPSS=RT*EPSC_REVPKZB*(ONE+CFD*EPSC_REVPKZB*Z3)
      RETURN
      END SUBROUTINE VTPSSc


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! CRPKZB                       C(ZETA,XI)
! D_CRPKZB_D1,D_CRPKZB_D2      THE DERIVATIVE OF C(ZETA,XI) WRT n
! D_CRPKZB_DD1,D_CRPKZB_DD2    THE DERIVATIVE OF C(ZETA,XI) WRT |GRAD n|
!
!***********************************************************************

      SUBROUTINE COE_RPKZB_TPSS(&
     &   RU,RD,DRU,DRD,DRT, &
     &   CRPKZB,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the correlatin part
      PARAMETER (CF1=0.53_q)
      PARAMETER (CF2=0.87_q)
      PARAMETER (CF3=0.50_q)
      PARAMETER (CF4=2.26_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)

      CRPKZB=0._q
      D_CRPKZB_D1=0._q;D_CRPKZB_D2=0._q;
      D_CRPKZB_DD1=0._q;D_CRPKZB_DD2=0._q;

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
! YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
      ZETA2=ZETA*ZETA
      OPZETA=ONE/(ONE+ZETA)
      OMZETA=ONE/(ONE-ZETA)
      D_ZETA_D1=TWO*RD/RT**TWO
      D_ZETA_D2=-TWO*RU/RT**TWO

! |GRAD ZETA|, 
! IN SOME EXTREME CASES, THE TERM (YA*RD**TWO-TWO*RU*RD*YC+YB*RU**TWO)
! GOES TO NEGATIVE BECAUSE OF THE NUMERICAL PRESICION OF COMPUTERS. SO, WE USE 2/4 AS THE POWER
! INSTEAD OF 1/2
     
      DEL_ZETA=TWO/RT**TWO*((YA*RD**TWO-TWO*RU*RD*YC+YB*RU**TWO)**TWO)**(ONE/FOUR)
      DEL_ZETA=MAX(DEL_ZETA,1E-10_q)
      D_DEL_ZETA_D1=-TWO*DEL_ZETA/RT+TWO/(RT**FOUR*DEL_ZETA)*(-TWO*YC*RD+TWO*RU*YB)
      D_DEL_ZETA_D2=-TWO*DEL_ZETA/RT+TWO/(RT**FOUR*DEL_ZETA)*(-TWO*YC*RU+TWO*RD*YA)
      D_DEL_ZETA_DD1=TWO/(RT**FOUR*DEL_ZETA)*(TWO*RD**TWO*DRU-TWO*RU*RD*YC/DRU)
      D_DEL_ZETA_DD2=TWO/(RT**FOUR*DEL_ZETA)*(TWO*RU**TWO*DRD-TWO*RU*RD*YC/DRD)

! XI
      XIDEN=TWO*(THREE*PISQ*RT)**THRD
      XI=DEL_ZETA/XIDEN
      D_XI_D1=(D_DEL_ZETA_D1-THRD*DEL_ZETA/RT)/XIDEN
      D_XI_D2=(D_DEL_ZETA_D2-THRD*DEL_ZETA/RT)/XIDEN
      D_XI_DD1=D_DEL_ZETA_DD1/XIDEN
      D_XI_DD2=D_DEL_ZETA_DD2/XIDEN

      CNUM=CF1+(CF2+(CF3+CF4*ZETA2)*ZETA2)*ZETA2
      D_CNUM_D1=D_ZETA_D1*(TWO*CF2+(FOUR*CF3+SIX*CF4*ZETA2)*ZETA2)*ZETA
      D_CNUM_D2=D_ZETA_D2*(TWO*CF2+(FOUR*CF3+SIX*CF4*ZETA2)*ZETA2)*ZETA

      OPZETA43=OPZETA**THRD4
      OMZETA43=OMZETA**THRD4
      CDENL=ONE+XI**TWO*(OPZETA43+OMZETA43)/TWO
      CDEN=CDENL**FOUR
      D_CDENL_DXI=XI*(OPZETA43+OMZETA43)
      D_CDENL_DZETA=-THRD2*XI**TWO*(OPZETA43*OPZETA-OMZETA43*OMZETA)
      D_CDENL_D1=D_CDENL_DXI*D_XI_D1+D_CDENL_DZETA*D_ZETA_D1
      D_CDENL_D2=D_CDENL_DXI*D_XI_D2+D_CDENL_DZETA*D_ZETA_D2
      D_CDENL_DD1=D_CDENL_DXI*D_XI_DD1
      D_CDENL_DD2=D_CDENL_DXI*D_XI_DD2

      D_CDEN_D1=FOUR*CDENL**THREE*D_CDENL_D1
      D_CDEN_D2=FOUR*CDENL**THREE*D_CDENL_D2
      D_CDEN_DD1=FOUR*CDENL**THREE*D_CDENL_DD1
      D_CDEN_DD2=FOUR*CDENL**THREE*D_CDENL_DD2
      
! OUTPUT CRPKZB,,D_CRPKZB_D1,D_CRPKZB_D2,D_CRPKZB_DD1,D_CRPKZB_DD2
      CRPKZB=CNUM/CDEN
      D_CRPKZB_D1=(CDEN*D_CNUM_D1-CNUM*D_CDEN_D1)/CDEN**TWO
      D_CRPKZB_D2=(CDEN*D_CNUM_D2-CNUM*D_CDEN_D2)/CDEN**TWO
      D_CRPKZB_DD1=-CNUM*D_CDEN_DD1/CDEN**TWO
      D_CRPKZB_DD2=-CNUM*D_CDEN_DD2/CDEN**TWO
      RETURN
      END SUBROUTINE COE_RPKZB_TPSS


!***********************************************************************
!
! Written by Jianwei Sun and Yoon-Suk Kim 06/18/2009
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! EPPGGA                       GGA ENERGY PER PARTICLE
! EPPGGA_D1,EPPGGA_D2          THE DERIVATIVE OF EPPGGA WRT n
! EPPGGA_DD1,EPPGGA_DD2        THE DERIVATIVE OF EPPGGA WRT |GRAD n|
!
!***********************************************************************
      SUBROUTINE CORGGA_TPSS(&
     &   RU,RD,DRU,DRD,DRT, &
     &   EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

      USE prec
      USE constant

      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
  
      PARAMETER (GAMMA=0.03109069086965489503494086371273_q)
      PARAMETER (BETA_mb=0.06672455060314922_q)

      EPPGGA  =0._q
      EPPGGA_D1=0._q
      EPPGGA_D2=0._q
      EPPGGA_DD1=0._q
      EPPGGA_DD2=0._q

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
!     YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)
      DZETAD1=TWO*RD/RT**TWO
      DZETAD2=-TWO*RU/RT**TWO


      DTHRD=exp(log(RT)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD
      DRSD1=-THRD/RT*RS
      DRSD2=-THRD/RT*RS


      PHI = (exp((TWO*THRD)*log(1._q+ZETA)) &
                +exp((TWO*THRD)*log(1._q-ZETA)))/2._q
      D_PHI_DZETA=THRD*((ONE+ZETA)**(-THRD)-(ONE-ZETA)**(-THRD))
      D_PHI_D1=D_PHI_DZETA*DZETAD1
      D_PHI_D2=D_PHI_DZETA*DZETAD2

      AFIX_T=SQRT(PI/FOUR)*(9._q*PI/FOUR)**(ONE/SIX)
      S=DRT/(TWO*(THREE*PISQ)**THRD*RT**THRD4)
      DSD1=-THRD4*S/RT
      DSD2=-THRD4*S/RT
! |GRAD N|/|GRAD N(UP OR DOWN)|
      GNGNU=ONE/(TWO*DRT)*(TWO*DRU+TWO*YC/DRU)
      GNGND=ONE/(TWO*DRT)*(TWO*DRD+TWO*YC/DRD)
      DSDD1=S/DRT*GNGNU
      DSDD2=S/DRT*GNGND

      T=AFIX_T*S/SQRT(RS)/PHI
      T2 = T*T
      T4 = T2*T2

      DTD1=AFIX_T*(PHI*RS*DSD1-0.5_q*S*PHI*DRSD1-S*RS*D_PHI_D1)/(PHI**TWO*RS**(THREE/TWO))
      DTD2=AFIX_T*(PHI*RS*DSD2-0.5_q*S*PHI*DRSD2-S*RS*D_PHI_D2)/(PHI**TWO*RS**(THREE/TWO))
      DTDD1=AFIX_T*DSDD1/(SQRT(RS)*PHI)
      DTDD2=AFIX_T*DSDD2/(SQRT(RS)*PHI)

      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)
! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
      CALL CORPBE_tpss(RS,ZETA,EC,VCUPLDA,VCDNLDA,PHI,SK, &
           T,H,DVCUP,DVCDN,ECQ,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=EC
!      EPPGGA_D1=(VCUPLDA-EC)/RT
!      EPPGGA_D2=(VCDNLDA-EC)/RT
      D_EC_DZETA=(VCUPLDA-VCDNLDA)/TWO
      D_EC_DRS=(EC-ZETA*D_EC_DZETA-(VCUPLDA+VCDNLDA)/TWO)*THREE/RS

! THE RS DEPENDENCE OF BETA
      AFACTOR=0.1_q
      BFACTOR=0.1778_Q
      BETA_NUM=ONE + AFACTOR*RS
      BETA_DEN=ONE+ BFACTOR*RS
      D_BETA_NUM=AFACTOR
      D_BETA_DEN=BFACTOR
!      BETA = BETA_MB*BETA_NUM/BETA_DEN
!      D_BETA_DRS=BETA_MB*(BETA_DEN*D_BETA_NUM-BETA_NUM*D_BETA_DEN)/BETA_DEN**TWO
      BETA=BETA_MB
      D_BETA_DRS=0._q     
 
      PHI3=PHI**THREE
      PON=-EC/(PHI3*gamma)
      W=DEXP(PON)-ONE
      D_W_DRS=-(W+ONE)*D_EC_DRS/(GAMMA*PHI3)
      D_W_DZETA=-(W+ONE)/(GAMMA*PHI3)*(D_EC_DZETA-THREE*EC*D_PHI_DZETA/PHI)
      D_W_DT=0._q


      A=BETA/(GAMMA*W)
      D_A_DRS=(W*D_BETA_DRS-BETA*D_W_DRS)/(GAMMA*W**TWO)
      D_A_DZETA=-BETA*D_W_DZETA/(GAMMA*W**TWO)
      D_A_DT=0._q

      V=A*T2
      D_V_DRS=T2*D_A_DRS
      D_V_DZETA=T2*D_A_DZETA
      D_V_DT=T2*D_A_DT+TWO*A*T
        
      FUNKG=ONE/(ONE+V+V**TWO)
      D_FUNKG_DV=-(ONE+TWO*V)*FUNKG**TWO
      
      
      HCORE=ONE+W*(ONE-FUNKG)
      AH=GAMMA*PHI3
      H=AH*DLOG(HCORE)
   
      DH1=ONE-FUNKG
      DH2=W*D_FUNKG_DV
      D_H_DRS=(DH1*D_W_DRS-DH2*D_V_DRS)*AH/HCORE
      D_H_DZETA=THREE*H*D_PHI_DZETA/PHI+(DH1*D_W_DZETA-DH2*D_V_DZETA)*AH/HCORE
      D_H_DT=(DH1*D_W_DT-DH2*D_V_DT)*AH/HCORE

! OUTPUT EPPGGA AND ITS DERIVATIVES EPPGGA_D1,EPPGGA_D2, EPPGGA_DD1, EPPGGA_DD2
      EPPGGA=EC+H
      EPPGGA_D1=D_EC_DRS*DRSD1+D_EC_DZETA*DZETAD1+D_H_DRS*DRSD1+D_H_DZETA*DZETAD1+D_H_DT*DTD1
      EPPGGA_D2=D_EC_DRS*DRSD2+D_EC_DZETA*DZETAD2+D_H_DRS*DRSD2+D_H_DZETA*DZETAD2+D_H_DT*DTD2
      EPPGGA_DD1=D_H_DT*DTDD1
      EPPGGA_DD2=D_H_DT*DTDD2

      RETURN
      END SUBROUTINE CORGGA_TPSS


      SUBROUTINE GGASPINCOR_TPSS(D1,D2,DDA,EC)
!     D1   density up
!     D2   density down
!     DDA  |gradient of the total density|

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (THRD=1._q/3._q)

      D=D1+D2
      DTHRD=exp(log(D)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD

      ZETA=(D1-D2)/D
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)

      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)
      G = (exp((2*THRD)*log(1._q+ZETA)) &
                +exp((2*THRD)*log(1._q-ZETA)))/2._q
      T = DDA/(D*2._q*SK*G)

      CALL CORPBE_tpss(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA,G,SK, &
           T,EC,ECD1,ECD2,ECQ,.TRUE.)

      EC  =(EC  +ECLDA)
      RETURN
      END SUBROUTINE GGASPINCOR_TPSS


! Modified by Adrienn Ruzsinszky
!######################################################################
!----------------------------------------------------------------------
      SUBROUTINE CORPBE_tpss(RS,ZET,EC,VCUP,VCDN,g,sk, &
     &                  T,H,DVCUP,DVCDN,ecdd,lgga)
!----------------------------------------------------------------------
!  Official PBE correlation code. K. Burke, May 14, 1996.
!  INPUT: RS=SEITZ RADIUS=(3/4pi rho)^(1/3)
!       : ZET=RELATIVE SPIN POLARIZATION = (rhoup-rhodn)/rho
!       : t=ABS(GRAD rho)/(rho*2.*KS*G)  -- only needed for PBE
!       : lgga=flag to do gga (0=>LSD only)
!       : lmetagga=flag to do metagga (revTPSS)
!       : lpot=flag to do potential (0=>energy only)
!  output: ec=lsd correlation energy from [a]
!        : vcup=lsd up correlation potential
!        : vcdn=lsd dn correlation potential
!        : h=NONLOCAL PART OF CORRELATION ENERGY PER ELECTRON
!        : dvcup=nonlocal correction to vcup
!        : dvcdn=nonlocal correction to vcdn
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! References:
! [a] J.P.~Perdew, K.~Burke, and M.~Ernzerhof, 
!     {\sl Generalized gradient approximation made simple}, sub.
!     to Phys. Rev.Lett. May 1996.
! [b] J. P. Perdew, K. Burke, and Y. Wang, {\sl Real-space cutoff
!     construction of a generalized gradient approximation:  The PW91
!     density functional}, submitted to Phys. Rev. B, Feb. 1996.
! [c] J. P. Perdew and Y. Wang, Phys. Rev. B {\bf 45}, 13244 (1992).
!----------------------------------------------------------------------
!----------------------------------------------------------------------
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      logical lgga
! thrd*=various multiples of 1/3
! numbers for use in LSD energy spin-interpolation formula, [c](9).
!      GAM= 2^(4/3)-2
!      FZZ=f''(0)= 8/(9*GAM)
! numbers for construction of PBE
!      gamma=(1-log(2))/pi^2
!      bet=coefficient in gradient expansion for correlation, [a](4).
!      eta=small number to stop d phi/ dzeta from blowing up at 
!          |zeta|=1.      
      parameter(thrd=1._q/3._q,thrdm=-thrd,thrd2=2._q*thrd)
      parameter(sixthm=thrdm/2._q)
      parameter(thrd4=4._q*thrd)
      parameter(GAM=0.5198420997897463295344212145565_q)
      parameter(fzz=8._q/(9._q*GAM))
      parameter(gamma=0.03109069086965489503494086371273_q)
      parameter(bet_mb=0.06672455060314922_q)
      parameter(eta=1.e-12_q)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! find LSD energy contributions, using [c](10) and Table I[c].
! EU=unpolarized LSD correlation energy
! EURS=dEU/drs
! EP=fully polarized LSD correlation energy
! EPRS=dEP/drs
! ALFM=-spin stiffness, [c](3).
! ALFRSM=-dalpha/drs
! F=spin-scaling factor from [c](9).
! construct ec, using [c](8)
      rtrs=dsqrt(rs)
      CALL gcor2(0.0310907_q,0.21370_q,7.5957_q,3.5876_q,1.6382_q, &
     &    0.49294_q,rtrs,EU,EURS)
      CALL gcor2(0.01554535_q,0.20548_q,14.1189_q,6.1977_q,3.3662_q, &
     &    0.62517_q,rtRS,EP,EPRS)
      CALL gcor2(0.0168869_q,0.11125_q,10.357_q,3.6231_q,0.88026_q, &
     &    0.49671_q,rtRS,ALFM,ALFRSM)
      ALFC = -ALFM
      Z4 = ZET**4
      F=((1._q+ZET)**THRD4+(1._q-ZET)**THRD4-2._q)/GAM
      EC = EU*(1._q-F*Z4)+EP*F*Z4-ALFM*F*(1._q-Z4)/FZZ
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! LSD potential from [c](A1)
! ECRS = dEc/drs [c](A2)
! ECZET=dEc/dzeta [c](A3)
! FZ = dF/dzeta [c](A4)
      ECRS = EURS*(1._q-F*Z4)+EPRS*F*Z4-ALFRSM*F*(1._q-Z4)/FZZ
      FZ = THRD4*((1._q+ZET)**THRD-(1._q-ZET)**THRD)/GAM
      ECZET = 4._q*(ZET**3)*F*(EP-EU+ALFM/FZZ)+FZ*(Z4*EP-Z4*EU &
     &        -(1._q-Z4)*ALFM/FZZ)
      COMM = EC -RS*ECRS/3._q-ZET*ECZET
      VCUP = COMM + ECZET
      VCDN = COMM - ECZET
!write(*,*)'rs,VCUP,VCDN',rs,VCUP,VCDN
      if (.not.lgga) return
!----------------------------------------------------------------------
! PBE correlation energy
! G=phi(zeta), given after [a](3)
! DELT=bet/gamma
! B=A of [a](8)
!  in TPSS beta is rs-independent
!      bet = bet_mb*(1._q + 0.1_q*RS)/(1._q + 0.1778_q*RS)
       bet = bet_mb

      delt=bet/gamma
      G3 = G**3
      PON=-EC/(G3*gamma)
      B = DELT/(DEXP(PON)-1._q)
      B2 = B*B
      T2 = T*T
      T4 = T2*T2
      RS2 = RS*RS
      RS3 = RS2*RS
      Q4 = 1._q+B*T2
      Q5 = 1._q+B*T2+B2*T4
      H = G3*(BET/DELT)*DLOG(1._q+DELT*Q4*T2/Q5)
!----------------------------------------------------------------------
!----------------------------------------------------------------------
! ENERGY DONE. NOW THE POTENTIAL, using appendix E of [b].
      G4 = G3*G
      T6 = T4*T2
      RSTHRD = RS/3._q
      GZ=(((1._q+zet)**2+eta)**sixthm- &
     &((1._q-zet)**2+eta)**sixthm)/3._q
      FAC = DELT/B+1._q
      BG = -3._q*B2*EC*FAC/(BET*G4)
      BEC = B2*FAC/(BET*G3)
      Q8 = Q5*Q5+DELT*Q4*Q5*T2
      Q9 = 1._q+2._q*B*T2
      hB = -BET*G3*B*T6*(2._q+B*T2)/Q8
      hRS = -RSTHRD*hB*BEC*ECRS
      hZ = 3._q*GZ*h/G + hB*(BG*GZ+BEC*ECZET)
      hT = 2._q*BET*G3*Q9/Q8
      COMM = H+HRS-7.0_q*T2*HT/6._q
      PREF = HZ-GZ*T2*HT/G
      COMM = COMM-PREF*ZET
      DVCUP = COMM + PREF
      DVCDN = COMM - PREF
      ecdd=0.5_q/(sk*g)*t*ht
      RETURN
      END SUBROUTINE CORPBE_tpss


!************************ SUBROUTINE M06 family ****************************
!
! calculates the first order derivatives of Ex wrt n, tau, and |grad(n)|
! Zhao & Truhlar TCA (2008)
!
! Written by Yan Zhao 09/22/2011
!
! everything in Hartree units
!
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VXD1 VXD2                    THE DERIVATIVES OF EX WRT n 
! VXDD1,VXDD2                  THE DERIVATIVES OF EX WRT |grad n| 
! AMUXD1, AMUXD2               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VM06x(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ex_M06,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (TWO=2._q)

      STAUU = TAUU*TWO
      STAUD = TAUD*TWO
      Ex_M06 = 0._q
      EM6U  = 0._q
      EVSXU = 0._q
      EM6D  = 0._q
      EVSXD = 0._q
      VXD1=0._q;VXD2=0._q;
      VXDD1=0._q;VXDD2=0._q;
      AMUXD1=0._q;AMUXD2=0._q;
 
      CALL M6X(EM6U,RU,DRU,STAUU,EM6D1,EM6DD1,EM6AMUXD1,IJZY)
      CALL VS98X(EVSXU,RU,DRU,STAUU,VSD1,VSDD1,VSAMUXD1,IJZY+1)

      CALL M6X(EM6D,RD,DRD,STAUD,EM6D2,EM6DD2,EM6AMUXD2,IJZY)
      CALL VS98X(EVSXD,RD,DRD,STAUD,VSD2,VSDD2,VSAMUXD2,IJZY+1)
       
      Ex_M06 = EM6U + EVSXU + EM6D + EVSXD
      VXD1 = EM6D1 + VSD1
      VXD2 = EM6D2 + VSD2
!  EM6DD and VSDD are  DERIVATIVES OF EX WRT |grad n|^2 
!  so We need to multiply them by 2*|grad n|
      VXDD1 = (EM6DD1 + VSDD1)*TWO*DRU
      VXDD2 = (EM6DD2 + VSDD2)*TWO*DRD
! EM6AMUXD and VSAMUXD are  DERIVATIVES OF EX WRT 2Tau
! so We need to multiply them by 2
      AMUXD1 = (EM6AMUXD1 + VSAMUXD1)*TWO
      AMUXD2 = (EM6AMUXD2 + VSAMUXD2)*TWO
   
      RETURN
      END SUBROUTINE VM06x

!************************ SUBROUTINE M06 family ****************************
!
! calculates the first order derivatives of Ex wrt n, tau, and |grad(n)|
! Zhao & Truhlar TCA (2008)
!
! Written by Yan Zhao 09/22/2011
!
! everything in Hartree units
!
!
! R                     density up,down
! DR                    abs. val. gradient of density up/down
! TAU                   kinetic energy density up/down
! M6D                   THE DERIVATIVES OF EX WRT n 
! M6DD                  THE DERIVATIVES OF EX WRT |grad n| 
! M6AMUXD               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE M6X(EM6,R,DR,TAU,EM6D,EM6DD,EM6AMUXD,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the coefficients 
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (SEVEN=7._q)
      PARAMETER (EIGHT=8._q)
      PARAMETER (NINE=9._q)

!      PI    = FOUR*ATAN(ONE)
      F1O3  = ONE / THREE
      F1O4  = ONE / FOUR
      F1O8  = ONE / EIGHT
      F2O3  = TWO / THREE
      F3O2  = THREE / TWO
      F4O3  = FOUR / THREE
      F4O9  = FOUR / NINE
      F3O5  = THREE / FIVE
      F5O3  = FIVE / THREE
      F5O2  = FIVE / TWO
      F7O3  = SEVEN / THREE
      F8O27 = EIGHT / ( THREE * NINE )
      F12   = TWO * SIX
      F18   = TWO * NINE
      F10 = TWO*FIVE
      F11 = 11._q
      DTol = 1.0e-10_q
 

!
!     PBE FUNCTIONAL'S PARAMETERS.
!
      C1     = 3.36116e-03_q
      C2     = 4.49267e-03_q
!     PARAMETERS FOR M06-L
      IF (IJZY.EQ.1) THEN
        AT0=    3.987756e-01_q
        AT1=    2.548219e-01_q
        AT2=    3.923994e-01_q
        AT3=    -2.103655e+00_q
        AT4=    -6.302147e+00_q
        AT5=    1.097615e+01_q
        AT6=    3.097273e+01_q
        AT7=    -2.318489e+01_q
        AT8=    -5.673480e+01_q
        AT9=    2.160364e+01_q
        AT10=   3.421814e+01_q
        AT11=   -9.049762e+00_q
       ELSEIF (IJZY.EQ.2) THEN
!     PARAMETERS FOR M06-HF
        AT0=    1.179732e-01_q
        AT1=    -1.066708e+00_q
        AT2=    -1.462405e-01_q
        AT3=    7.481848e+00_q
        AT4=    3.776679e+00_q
        AT5=    -4.436118e+01_q
        AT6=    -1.830962e+01_q
        AT7=    1.003903e+02_q
        AT8=    3.864360e+01_q
        AT9=    -9.806018e+01_q
        AT10=   -2.557716e+01_q
        AT11=   3.590404e+01_q
       ELSEIF (IJZY.EQ.3) THEN
!     PARAMETERS FOR M06
        AT0=    5.877943e-01_q
        AT1=    -1.371776e-01_q
        AT2=    2.682367e-01_q
        AT3=    -2.515898e+00_q
        AT4=    -2.978892e+00_q
        AT5=    8.710679e+00_q
        AT6=    1.688195e+01_q
        AT7=    -4.489724e+00_q
        AT8=    -3.299983e+01_q
        AT9=    -1.449050e+01_q
        AT10=   2.043747e+01_q
        AT11=   1.256504e+01_q
       ELSEIF (IJZY.EQ.4) THEN
!     PARAMETERS FOR M06-2X
        AT0=    4.600000e-01_q
        AT1=    -2.206052e-01_q
        AT2=    -9.431788e-02_q
        AT3=    2.164494e+00_q
        AT4=    -2.556466e+00_q
        AT5=    -1.422133e+01_q
        AT6=    1.555044e+01_q
        AT7=    3.598078e+01_q
        AT8=    -2.722754e+01_q
        AT9=    -3.924093e+01_q
        AT10=   1.522808e+01_q
        AT11=   1.522227e+01_q
      ENDIF

      
      IF ((R .GT. DTol) .AND. (Tau .GT. DTol)) THEN
        Ax = -F3o2*(F4o3*PI)**(-F1o3)
        RHOO = R
        RHO43 = RHOO**F4O3  
        RRHO = ONE/RHOO       ! RECIPROCAL OF RHO
        RHO13 = RHO43*RRHO
        RHO53 = RHOO**F5O3
        TAUN = TAU
        TAUUEG=F3O5*((SIX*PI*PI)**F2O3)*RHO53
        TSIG =TAUUEG/TAUN
	   
        WSIG =(TSIG-ONE)/(TSIG+ONE)
        FSIG=(AT0 + WSIG*(AT1 + WSIG*(AT2 + WSIG*(AT3 + WSIG*( &
     &            AT4 + WSIG*(AT5 + WSIG*(AT6 + WSIG*(AT7 + WSIG*( &
     &            AT8 + WSIG*(AT9 + WSIG*(AT10+WSIG*AT11)))))))))))
        Gamma = DR
        X = GAMMA/RHO43
        X2 = X*X
        EN = C1*X2
        ED = ONE + C2*X2
        E  = -EN/ED
        EM6=(AX+E)*FSIG*RHO43
       
        DEN   = TWO*C1*X   
        DED   = TWO*C2*X
        DE    = -(DEN*ED-EN*DED)/(ED*ED)
        DFDW=(AT1 + WSIG*(TWO  *AT2 + WSIG*(THREE*AT3 + WSIG*( &
     &           FOUR *AT4 + WSIG*(FIVE *AT5 + WSIG*(SIX  *AT6 + WSIG*(&
     &           SEVEN*AT7 + WSIG*(EIGHT*AT8 + WSIG*(NINE *AT9 + WSIG*(&
     &           F10 *AT10+ WSIG*F11*AT11))))))))))
         DWDT = TWO/((ONE + TSIG)**2)
         DTDR = ((SIX*PI*PI)**F2O3)*(RHOO**F2O3)/TAUN
         DTDTAU = -TAUUEG/TAUN**2
         DGGADR = F4O3*RHO13*(AX+(E-X*DE))
         DFDR = DFDW*DWDT*DTDR
         DFDTAU=DFDW*DWDT*DTDTAU
         DGGADG =DE/(TWO*GAMMA)
!       DGGADG =(DE/(TWO*GAMMA)) 
       
!        DF/DRHOA
         EM6D = DGGADR*FSIG + (AX+E)*RHO43*DFDR
!        DF/DGAMMAAA
         EM6DD = DGGADG*FSIG
!        DF/DTAUA
         EM6AMUXD = RHO43*(AX+E)*DFDTAU
      ELSE
         EM6= 0._q
         EM6D = 0._q
         EM6DD = 0._q
         EM6AMUXD = 0._q
      ENDIF 
      RETURN
      END SUBROUTINE M6X

!************************ SUBROUTINE VS98 family ****************************
!
! calculates the first order derivatives of Ex wrt n, tau, and |grad(n)|
! Zhao & Truhlar TCA (2008)
!
! Written by Yan Zhao 09/22/2011
!
! everything in Hartree units
!
!
! R                     density up,down
! DR                    abs. val. gradient of density up/down
! TAU                   kinetic energy density up/down
! VSD                   THE DERIVATIVES OF EX WRT n 
! VSDD                  THE DERIVATIVES OF EX WRT |grad n| 
! VSAMUXD               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VS98X(EVS,R,DR,TAU,VSD,VSDD,VSAMUXD,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

      REAL(q) KX,NINE

!      PARAMETER( PI = 3.1415926535897932384626433832795_q )

      PARAMETER (CF = 9.115599720_q)
      PARAMETER (AXLSDA = -0.9305257363491_q)
      PARAMETER (GG  = 0.00186726_q)
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (FIVE=5._q)
      PARAMETER (SIX=6._q)
      PARAMETER (SEVEN=7._q)
      PARAMETER (EIGHT=8._q)
      PARAMETER (NINE=9._q)
      PARAMETER (F10=10._q)
      PARAMETER (F11=11._q)

      DTol = 1.0e-10_q
!      IF ((R .LT. DTol) .OR. (Tau .LT. DTol)) Return
      F13 = ONE/THREE
      F43 = FOUR/THREE
      F53 = FIVE/THREE
      F83 = EIGHT/THREE
      F113 = F11/THREE
      F3O5 = THREE/FIVE
      F2O3 = TWO/THREE
      
      IF (IJZY.EQ.1) THEN

!     PARAMETERS FOR VS98

        R1=  -9.800683E-01_q
        R2=  -3.556788E-03_q
        R3=   6.250326E-03_q
        R4=  -2.354518E-05_q
        R5=  -1.282732E-04_q
        R6=   3.574822E-04_q
      ELSEIF (IJZY.EQ.2) THEN

!     PARAMETERS FOR M06-L

        R1 =   6.012244E-01_q*AXLSDA
        R2 =   4.748822E-03_q*AXLSDA
        R3 =  -8.635108E-03_q*AXLSDA
        R4 =  -9.308062E-06_q*AXLSDA
        R5 =   4.482811E-05_q*AXLSDA
        R6 =   0.000000E+00_q
      ELSEIF (IJZY.EQ.3) THEN

!     PARAMETERS FOR M06-HF

        R1 =   -1.179732E-01_q*AXLSDA
        R2 =   -2.500000E-03_q*AXLSDA
        R3 =   -1.180065E-02_q*AXLSDA
        R4 =   0.000000E+00_q
        R5 =   0.000000E+00_q
        R6 =   0.000000E+00_q
      ELSEIF (IJZY.EQ.4) THEN

!     PARAMETERS FOR M06

        R1 =   1.422057E-01_q*AXLSDA
        R2 =   7.370319E-04_q*AXLSDA
        R3 =   -1.601373E-02_q*AXLSDA
        R4 =   0.000000E+00_q
        R5 =   0.000000E+00_q
        R6 =   0.000000E+00_q
      ENDIF

      IF ((R .GT. DTol) .AND. (Tau .GT. DTol)) THEN 
        RHOO = R
        RHO43 = RHOO**F43  
        RRHO = ONE/RHOO       ! RECIPROCAL OF RHO
        RHO13 = RHO43*RRHO
        RHO53 = RHOO**F53
        TAUN =  TAU
        TAUUEG=F3O5*((SIX*PI*PI)**F2O3)*RHO53
	TSIG = TAUUEG/TAUN

        RHO83 = RHO53*RHOO
        GAMMA = DR*DR
        X = GAMMA/RHO83
        DXDR = -F83*X*RRHO
        DXDG = ONE/RHO83
        Z = TAUN/RHO53 - CF
        DZDR = -F53 * TAUN/RHO83
        DZDT = ONE/RHO53
        KX = ONE + GG*X + GG*Z
        XK = X/KX
        ZK = Z/KX
        CALL GVT4(GX,DGDX,DGDZ,XK,ZK,KX,GG,R1,R2,R3,R4,R5,R6)
        EVS =  RHO43*GX
        VSD =  F43*RHO13*GX + RHO43*(DGDX*DXDR + DGDZ*DZDR)
        VSDD =  RHO43*(DGDX*DXDG)
        VSAMUXD =  RHO43*(DGDZ*DZDT)
      ELSE
        EVS = 0._q
        VSD = 0._q
        VSDD = 0._q
        VSAMUXD = 0._q
      ENDIF
      
      RETURN 
      END SUBROUTINE VS98X 


      SUBROUTINE GVT4(GVT,DG_DX,DG_DZ,XG,ZG,GAMA,CT,A,B,C,D,E,F)
       USE prec
       USE constant
       IMPLICIT REAL(q) (A-H,O-Z)       
!    SOME WORKING VARIABLES
       PARAMETER (F1= 1._q)
       PARAMETER (F2= 2._q)
       PARAMETER (F3= 3._q)


       G=GAMA
       G2=GAMA*GAMA

       GVT =(A + B*XG + C*ZG + D*XG*XG + E*ZG*XG + F*ZG*ZG)/G
       DG_DX =(-A*CT+B*(F1-F2*CT*XG)-F2*C*ZG*CT+D*(F2*XG-F3*XG*XG*CT) &
     &  +E*(ZG -F3*ZG*XG*CT)-F3*F*ZG*ZG*CT )/G2
       DG_DZ =(-A*CT -F2*B*XG*CT +C*(F1-F2*ZG*CT)-F3*D*XG*XG*CT &
     &  +E*(XG-F3*XG*ZG*CT)+F*(F2*ZG-F3*ZG*ZG*CT))/G2

       RETURN
       END SUBROUTINE GVT4 


!************************ SUBROUTINE M06 family ****************************
!
! calculates the first order derivatives of Ex wrt n, tau, and |grad(n)|
! Zhao & Truhlar TCA (2008)
!
! Written by Yan Zhao 09/22/2011
!
! everything in Hartree units
!
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VXD1 VXD2                    THE DERIVATIVES OF EX WRT n 
! VXDD1,VXDD2                  THE DERIVATIVES OF EX WRT |grad n| 
! AMUXD1, AMUXD2               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VM06c(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_M06,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (F1=1._q)
      PARAMETER (F2=2._q)
      PARAMETER (F3=3._q)
      PARAMETER (F4=4._q)
      PARAMETER (COpp=0.0031_q)
      IF (IJZY.EQ.1) THEN
!     PARAMETERS FOR M06-L CORRELATION
         SOPP0= 6.042374e-01_q
         SOPP1= 1.776783e+02_q
         SOPP2= -2.513252e+02_q
         SOPP3= 7.635173e+01_q
         SOPP4= -1.255699e+01_q
      ELSEIF (IJZY.EQ.2) THEN
!     PARAMETERS FOR M06-HF CORRELATION
         SOPP0= 1.674634e+00_q
         SOPP1= 5.732017e+01_q
         SOPP2= 5.955416e+01_q
         SOPP3= -2.311007e+02_q
         SOPP4= 1.255199e+02_q
      ELSEIF (IJZY.EQ.3) THEN
!     PARAMETERS FOR M06 CORRELATION
         SOPP0= 3.741539e+00_q
         SOPP1= 2.187098e+02_q
         SOPP2= -4.531252e+02_q
         SOPP3= 2.936479e+02_q
         SOPP4= -6.287470e+01_q
      ELSEIF (IJZY.EQ.4) THEN
!     PARAMETERS FOR M06-2X CORRELATION
         SOPP0= 8.833596e-01_q
         SOPP1= 3.357972e+01_q
         SOPP2= -7.043548e+01_q
         SOPP3= 4.978271e+01_q
         SOPP4= -1.852891e+01_q
      ENDIF
      DTol = 1.0e-8_q
      Ec_M06=0._q
      VCD1=0._q;VCD2=0._q;
      VCDD1=0._q;VCDD2=0._q;
      AMUCD1=0._q;AMUCD2=0._q;
     
       
      CALL VS98C(DTol, &
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_VS,VSD1,VSDD1,VSD2,VSDD2,VSAMUCD1,VSAMUCD2,IJZY+1)

!      PI = F4*ATAN(F1)
      F6=6.0D0
      F43 = F4 / F3
      PI34 = F3 / (F4*PI)
      F13 = F1 / F3
      
!     parallel spin case UP
        PA = RU
      IF (RU .GT. DTol .AND. TAUU .GT. DTol) THEN
        GAA = DRU*DRU
        TAUA = F2*TAUU
        CALL M06CSS(DTol,PA,GAA,TAUA,FA,FPA,FGA,FTA,EUA, &
     &                CHIA,EUPA,CHIAP,CHIAG,IJZY)
      ELSE
        FA = 0._q
        FPA = 0._q
        FGA = 0._q
        FTA = 0._q
        EUA = 0._q
        CHIA = 0._q
        EUPA = 0._q
        CHIAP = 0._q
        CHIAG = 0._q
      ENDIF
!     parallel spin case DOWN
      PB = RD
      IF (RD .GT. DTol .AND. TAUD .GT. DTol) THEN
      GBB = DRD*DRD
      TAUB= F2*TAUD
      CALL M06CSS(DTol,PB,GBB,TAUB,FB,FPB,FGB,FTB,EUB, &
     &                CHIB,EUPB,CHIBP,CHIBG,IJZY)

       ELSE
        FB = 0._q
        FPB = 0._q
        FGB = 0._q
        FTB = 0._q
        EUB = 0._q
        CHIB = 0._q
        EUPB = 0._q
        CHIBP = 0._q
        CHIBG = 0._q
      ENDIF

!    antiparallel spin case
        P = PA + PB
!      IF (PB.GT.DTol.AND.PA.GT.DTol.AND.TAUA.GT.DTol.AND.TAUB.GT.DTol) THEN
       IF (PB.GT.DTol.AND.PA.GT.DTol) THEN
        RS = (PI34/P) ** F13
        RSP = -RS/(F3*P)
        ZETA = (PA-PB)/P
        DZDA = (F1-ZETA)/P
        DZDB = (-F1-ZETA)/P
!        CALL LSDAC(RS,ZETA,POTLC,DLDS,DLDZ,D2LDSS,D2LDSZ, &
!     &      D2LDZZ)

! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
       CALL CORPBE(RS,ZETA,PotLC,VCUPLDA,VCDNLDA,unknown,unknown, &
         &        unknown,unknown,unknown,unknown,unknown,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=PotLC
!      EPPGGA_D1=(VCUPLDA-PotLC)/RT
!      EPPGGA_D2=(VCDNLDA-PotLC)/RT
       dLdz=(VCUPLDA-VCDNLDA)/F2
       dLdS=(PotLC-ZETA*dLdZ-(VCUPLDA+VCDNLDA)/F2)*F3/RS

        EUEG = P*POTLC - EUA - EUB
        U = COPP*(CHIA+CHIB)/(F1 + COPP*(CHIA+CHIB))
        W = SOPP0+U*(SOPP1+U*(SOPP2+U*(SOPP3+U*SOPP4)))
        EAB = EUEG*W
        DUDCHIA =COPP/(F1 + COPP*(CHIA+CHIB))**2
        DUDCHIB =COPP/(F1 + COPP*(CHIA+CHIB))**2
        DUDPA= DUDCHIA*CHIAP
        DUDPB= DUDCHIB*CHIBP
        DUDGA= DUDCHIA*CHIAG
        DUDGB= DUDCHIB*CHIBG
        DWDU =SOPP1+U*(F2*SOPP2+U*(F3*SOPP3+U*F4*SOPP4))
        DWDPA= DWDU*DUDPA
        DWDPB= DWDU*DUDPB
        DWDGA= DWDU*DUDGA
        DWDGB= DWDU*DUDGB
        EUEGPA = POTLC + P*DLDS*RSP + P*DLDZ*DZDA - EUPA
        EUEGPB = POTLC + P*DLDS*RSP + P*DLDZ*DZDB - EUPB
        DEABDPA = EUEGPA*W + EUEG*DWDPA
        DEABDGAA = EUEG*DWDGA
        DEABDPB = EUEGPB*W + EUEG*DWDPB
        DEABDGBB = EUEG*DWDGB
      ELSE
         EAB = 0._q
         DEABDPA = 0._q
         DEABDGAA = 0._q
         DEABDPB = 0._q
         DEABDGBB = 0._q
      ENDIF
      Ec_M06 = FA + FB + EAB + Ec_VS
      VCD1 = FPA + DEABDPA + VSD1  
      VCD2 = FPB + DEABDPB + VSD2
      VCDD1 = F2*(FGA + DEABDGAA)*DRU + VSDD1
      VCDD2 = F2*(FGB + DEABDGBB)*DRD + VSDD2
      AMUCD1 = F2 * FTA + VSAMUCD1
      AMUCD2 = F2 * FTB + VSAMUCD2  
!debug
!      if (abs(Ec_M06) .gt. 0.00001_q) then
!       write (*,*) "RU, RD, DRU, DRD, TAUU, TAUD", RU, RD, DRU, DRD, TAUU, TAUD
!       write (*,*) " Ec_M061, Ec_VS, Ec_M06", FA + FB + EAB, Ec_VS, Ec_M06
!       write (*,*) "VCD1, VCD2  ", VCD1, VCD2
!       write (*,*) "VCDD1, VCDD2 =", VCDD1, VCDD2
!       write (*,*) "AMUCD1,  AMUCD2 = ", AMUCD1,  AMUCD2 
!       write (*,*) "VSD1,  VSD2", VSD1,  VSD2
!       write (*,*) "VSDD1, VSDD2", VSDD1, VSDD2
!       write (*,*) "VSAMUCD1, VSAMUCD2", VSAMUCD1, VSAMUCD2
!       write (*,*) "M06C FPA + DEABDPA", FPA + DEABDPA
!       write (*,*) "M06C FPB + DEABDPB", FPB + DEABDPB
!       write (*,*) "FGA + DEABDGAA", FGA + DEABDGAA
!       write (*,*) "FGB + DEABDGBB", FGB + DEABDGBB
!       write (*,*) "FTA, FTB", FTA, FTB
!       stop
!      ENDIF
      RETURN
      END SUBROUTINE VM06c

      SUBROUTINE M06CSS(DTol,PX,GX,TX,F,FP,FG,FT,EUEG,CHI,EUEGP, &
     &                   CHIP,CHIG,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

!     COMPUTE THE SAME-SPIN PART OF THE M06 CORRELATION FUNCTIONAL FOR ONE GRID
!     POINT AND ONE SPIN-CASE.
!     
      PARAMETER (ZERO = 0._q)    
      PARAMETER (F1=1._q)
      PARAMETER (F2=2._q)
      PARAMETER (F3=3._q)
      PARAMETER (F4=4._q)
      PARAMETER (F5=5._q)
      PARAMETER (F6=6._q)
      PARAMETER (F8=8._q)
      PARAMETER (F11=11._q)
      PARAMETER (CSS = 0.06_q)
      PARAMETER (PT25 = 0.25_q) 
!   
      SS=1._q 
      IF (IJZY.EQ.1) THEN
!     PARAMETERS FOR M06-L CORRELATION
         SSS0=  5.349466e-01_q
         SSS1=  5.396620e-01_q
         SSS2=  -3.161217e+01_q
         SSS3=  5.149592e+01_q
         SSS4=  -2.919613e+01_q
      ELSEIF (IJZY.EQ.2) THEN
!     PARAMETERS FOR M06-HF CORRELATION
         SSS0=  1.023254e-01_q
         SSS1=  -2.453783e+00_q
         SSS2=  2.913180e+01_q
         SSS3=  -3.494358e+01_q
         SSS4=  2.315955e+01_q
      ELSEIF (IJZY.EQ.3) THEN
!     PARAMETERS FOR M06 CORRELATION
         SSS0=  5.094055e-01_q
         SSS1=  -1.491085e+00_q
         SSS2=  1.723922e+01_q
         SSS3=  -3.859018e+01_q
         SSS4=  2.845044e+01_q
      ELSEIF (IJZY.EQ.4) THEN
!     PARAMETERS FOR M06-2X CORRELATION
         SSS0=  3.097855e-01_q
         SSS1=  -5.528642e+00_q
         SSS2=  1.347420e+01_q
         SSS3=  -3.213623e+01_q
         SSS4=  2.846742e+01_q
      ENDIF

!      IF ((PX.LE.DTol).OR.(TX.LE.DTol).OR.(GX.LE.DTol))  THEN
       IF ((PX.LE.DTol))  THEN
        EUEG = ZERO
        CHI = ZERO
        EUEGP = ZERO
        CHIP = ZERO
        CHIG = ZERO
        PX = ZERO
        GX = ZERO
        TX = ZERO
        F  = ZERO
        FP = ZERO
        FG = ZERO
        FT = ZERO
      ELSE
!        PI = F4*ATAN(F1)
        PI34 = F3 / (F4*PI)
        F13 = F1 / F3
        F23 = F2 / F3
        F43 = F2 * F23
        F53 = F5 / F3
        F83 = F8 / F3
        F113 = F11 / F3
        FDUEG = (F3/F5)*(F6*PI*PI)**F23
        RS = (PI34/PX) ** F13
!        CALL LSDAC(RS,F1,POTLC,DLDS,DLDZ,D2LDSS,D2LDSZ,D2LDZZ)
! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
       CALL CORPBE(RS,F1,PotLC,VCUPLDA,VCDNLDA,1.,1., &
          &  1.,unknown,unknown,unknown,unknown,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=PotLC
!      EPPGGA_D1=(VCUPLDA-PotLC)/RT
!      EPPGGA_D2=(VCDNLDA-PotLC)/RT
       dLdz=(VCUPLDA-VCDNLDA)/F2
       dLdS=(PotLC-F1*dLdZ-(VCUPLDA+VCDNLDA)/F2)*F3/RS
!c        Call lsdac(RS,F1,PotLC,dLdS,dLdZ)^M
!c  End of modification

        EUEG = PX*POTLC
        D = TX - PT25*GX/PX
!        DUEG = FDUEG*PX**F53
        CHI = GX/PX**F83
        U = CSS*CHI/(F1 + CSS*CHI)
        W = SSS0+U*(SSS1+U*(SSS2+U*(SSS3+U*SSS4)))
        FSCC=D/TX
        E = FSCC*W*EUEG
        F = E*SS
        RSP = -RS/(F3*PX)
        CHIG = F1/PX**F83
        CHIP = -F83*CHI/PX
        DFSCCP=PT25*GX/(TX*PX**2)
        DFSCCG=-PT25/(TX*PX)
        DFSCCT=PT25*GX/(PX*TX**2)
        DUDCHI=CSS/((F1+CSS*CHI)**2)
        DWDU=SSS1+U*(F2*SSS2+U*(F3*SSS3+U*F4*SSS4))
        DWDP=DWDU*DUDCHI*CHIP
        DWDG=DWDU*DUDCHI*CHIG
        EUEGP = POTLC + PX*DLDS*RSP
        FP = SS*(DFSCCP*W*EUEG            &
     &                 + FSCC*DWDP*EUEG   &
     &                 + FSCC*W*EUEGP)
        FG = SS*(DFSCCG*W*EUEG            &
     &                 + FSCC*DWDG*EUEG)

        FT = SS*(DFSCCT*W*EUEG)
       ENDIF

       RETURN
       END SUBROUTINE M06CSS



      SUBROUTINE VS98C(DTol, &
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_VS,VSD1,VSDD1,VSD2,VSDD2,VSAMUCD1,VSAMUCD2,IJZY)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      REAL(q) KAB
!     EVALUATE THE INTERPOLATION FUNCTION FOR PW91 LOCAL CORRELATION.
      PARAMETER (F1=1._q)
      PARAMETER (F2=2._q)
      PARAMETER (F3=3._q)
      PARAMETER (F4=4._q)
      PARAMETER (GAB=0.00304966e0_q)
      PARAMETER (CF=9.115599720e0_q)

      
!     PARAMETERS FOR VS98
      IF (IJZY.EQ.1) THEN
              R7=   7.035010e-01_q
              R8=   7.694574e-03_q
              R9=   5.152765e-02_q
              R10=   3.394308e-05_q
              R11=  -1.269420e-03_q
              R12=   1.296118e-03_q
!     PARAMETERS FOR M06-L
      ELSEIF (IJZY.EQ.2) THEN
              R7=      3.957626e-01_q
              R8=      -5.614546e-01_q
              R9=      1.403963e-02_q
              R10=     9.831442e-04_q
              R11=     -3.577176e-03_q
              R12=     0.000000e+00_q
!     PARAMETERS FOR M06-HF
      ELSEIF (IJZY.EQ.3) THEN
              R7=    -6.746338e-01_q
              R8=    -1.534002e-01_q
              R9=    -9.021521e-02_q
              R10=   -1.292037e-03_q
              R11=   -2.352983e-04_q
              R12=   0.000000e+00_q

!     PARAMETERS FOR M06
      ELSEIF (IJZY.EQ.4) THEN
               R7= -2.741539e+00_q
               R8= -6.720113e-01_q
               R9= -7.932688e-02_q
               R10=1.918681e-03_q
               R11=-2.032902e-03_q
               R12=0.000000e+00_q

!     PARAMETERS FOR M06-2X
      ELSEIF (IJZY.EQ.5) THEN
              R7=  1.166404e-01_q
              R8=  -9.120847e-02_q
              R9=  -6.726189e-02_q
              R10= 6.720580e-05_q
              R11= 8.448011e-04_q
              R12= 0.000000e+00_q
      ENDIF
!      PI = F4*ATAN(F1)
      F6=6._q
      F43 = F4 / F3
      PI34 = F3 / (F4*PI)
      F13 = F1 / F3

!     parallel spin case UP
        PA = RU
      IF (RU .GT. DTol .AND. TAUU .GT. DTol) THEN
        GAA = DRU*DRU
        TAUA = F2*TAUU
        CALL VS98SS(DTol,PA,GAA,TAUA,FA,FPA,FGA,FTA,EUA,ZA,&
     &                CHIA,EUPA,CHIAP,CHIAG,ZAP,ZAT,IJZY)
      ELSE
       FA = 0._q
       FPA = 0._q
       FGA = 0._q
       FTA = 0._q
       EUA = 0._q
       ZA = 0._q
       CHIA = 0._q
       EUPA = 0._q
       CHIAP = 0._q
       CHIAG = 0._q
      ENDIF
!     parallel spin case DOWN
      PB = RD
      IF (RD .GT. DTol .AND. TAUD .GT. DTol) THEN
      GBB = DRD*DRD
      TAUB= F2*TAUD
      CALL VS98SS(DTol,PB,GBB,TAUB,FB,FPB,FGB,FTB,EUB,ZB,&
     &                CHIB,EUPB,CHIBP,CHIBG,ZBP,ZBT,IJZY)
      ELSE
       FB = 0._q
       FPB = 0._q
       FGB = 0._q
       FTB = 0._q
       EUB = 0._q
       ZB = 0._q
       CHIB = 0._q
       EUPB = 0._q
       CHIBP = 0._q
       CHIBG = 0._q
      ENDIF


!    antiparallel spin case
      P=PA+PB
!      IF (PB.GT.DTol.AND.PA.GT.DTol.AND.TAUA.GT.DTol.AND.TAUB.GT.DTol) THEN
       IF (PB.GT.DTol.AND.PA.GT.DTol) THEN
        RS = (PI34/P) ** F13
        RSP = -RS/(F3*P)
        ZETA = (PA-PB)/P
        DZDA = (F1-ZETA)/P
        DZDB = (-F1-ZETA)/P
!        CALL LSDAC(RS,ZETA,POTLC,DLDS,DLDZ,D2LDSS,D2LDSZ, &
!     &      D2LDZZ)
! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
        CALL CORPBE(RS,ZETA,PotLC,VCUPLDA,VCDNLDA,1.,1., &
     &  1.,unknown,unknown,unknown,unknown,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=PotLC
!      EPPGGA_D1=(VCUPLDA-PotLC)/RT
!      EPPGGA_D2=(VCDNLDA-PotLC)/RT
        dLdz=(VCUPLDA-VCDNLDA)/F2
        dLdS=(PotLC-ZETA*dLdZ-(VCUPLDA+VCDNLDA)/F2)*F3/RS

        EUEG = P*POTLC - EUA - EUB
        ZAB = ZA + ZB
        XAB = CHIA+CHIB
        KAB = F1 + GAB*(XAB+ZAB)
        XK = XAB/KAB
        ZK = ZAB/KAB
        CALL GVT4(GCAB,DGDX,DGDZ,XK,ZK,KAB,GAB,R7,R8,R9,R10,R11,R12)
        EAB =  GCAB*EUEG
        DGDPA = DGDX*CHIAP + DGDZ*ZAP
        DGDGA = DGDX*CHIAG
        DGDTA = DGDZ*ZAT
        DGDPB = DGDX*CHIBP + DGDZ*ZBP
        DGDGB = DGDX*CHIBG
        DGDTB = DGDZ*ZBT
        EUEGPA = POTLC + P*DLDS*RSP + P*DLDZ*DZDA - EUPA
        EUEGPB = POTLC + P*DLDS*RSP + P*DLDZ*DZDB - EUPB
        DEABDPA =  (EUEGPA*GCAB + EUEG*DGDPA)
        DEABDPB =  (EUEGPB*GCAB + EUEG*DGDPB)
        DEABDGAA =  EUEG*DGDGA
        DEABDGBB =  EUEG*DGDGB
        DEABDTA =   EUEG*DGDTA
        DEABDTB =   EUEG*DGDTB
      ELSE
        EAB = 0._q
        DEABDPA = 0._q  
        DEABDPB = 0._q
        DEABDGAA = 0._q
        DEABDGBB = 0._q
        DEABDTA =  0._q
        DEABDTB =  0._q
      ENDIF
      Ec_VS = EAB + FA + FB
      VSD1 = FPA + DEABDPA    
      VSD2 = FPB + DEABDPB  
      VSDD1 = F2*(FGA + DEABDGAA)*DRU  
      VSDD2 = F2*(FGB + DEABDGBB)*DRD  
      VSAMUCD1 = F2 * (FTA + DEABDTA)  
      VSAMUCD2 = F2 * (FTB + DEABDTB)  

      RETURN
      END SUBROUTINE VS98C


      SUBROUTINE VS98SS(DTol,PX,GX,TX,F,FP,FG,FT,EUEG,Z,CHI,EUEGP, &
     &                   CHIP,CHIG,ZP,ZT,IJZY)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)


!     COMPUTE THE SAME-SPIN PART OF THE M06 CORRELATION FUNCTIONAL FOR ONE GRID
!     POINT AND ONE SPIN-CASE.



      REAL(q) KC
      PARAMETER (ZERO = 0._q)    
      PARAMETER (F1=1._q)
      PARAMETER (F2=2._q)
      PARAMETER (F3=3._q)
      PARAMETER (F4=4._q)
      PARAMETER (F5=5._q)
      PARAMETER (F6=6._q)
      PARAMETER (F8=8._q)
      PARAMETER (F11=11._q)
      PARAMETER (GCC = 0.00515088_q)
      PARAMETER (CF = 9.11559972_q)
      PARAMETER (PT25 = 0.25_q) 

      F4O3 = F4/F3
!     PARAMETERS FOR VS98
      IF (IJZY.EQ.1) THEN
              R13=   3.270912e-01_q
              R14=  -3.228915e-02_q
              R15=  -2.942406e-02_q
              R16=   2.134222e-03_q
              R17=  -5.451559e-03_q
              R18=   1.577575e-02_q
!     PARAMETERS FOR M06-L
      ELSEIF (IJZY.EQ.2) THEN
              R13=   4.650534e-01_q
              R14=   1.617589e-01_q
              R15=   1.833657e-01_q
              R16=   4.692100e-04_q
              R17=  -4.990573e-03_q
              R18=   0.000000e+00_q
!     PARAMETERS FOR M06-HF
      ELSEIF (IJZY.EQ.3) THEN
              R13=   8.976746e-01_q
              R14=  -2.345830e-01_q
              R15=   2.368173e-01_q
              R16=  -9.913890e-04_q
              R17=  -1.146165e-02_q
              R18=   0.000000e+00_q
!     PARAMETERS FOR M06
      ELSEIF (IJZY.EQ.4) THEN
               R13=  4.905945e-01_q
               R14= -1.437348e-01_q
               R15=  2.357824e-01_q
               R16=  1.871015e-03_q
               R17= -3.788963e-03_q
               R18=  0.000000e+00_q
!     PARAMETERS FOR M06-2X
      ELSEIF (IJZY.EQ.5) THEN
              R13=  6.902145e-01_q
              R14=  9.847204e-02_q
              R15=  2.214797e-01_q
              R16= -1.968264e-03_q
              R17= -6.775479e-03_q
              R18=  0.000000e+00_q
      ENDIF

!      IF(PX.LE.DTol.OR.TX.LE.DTol.OR.GX.LE.DTol) THEN
       IF(PX.LE.DTol) THEN
        EUEG = ZERO
        CHI = ZERO
        EUEGP = ZERO
        CHIP = ZERO
        CHIG = ZERO
        PX = ZERO
        GX = ZERO
        TX = ZERO
        F  = ZERO
        FP = ZERO
        FG = ZERO
        FT = ZERO
        Z  = ZERO
        ZP = ZERO
        ZT = ZERO
      ELSE
!        PI = F4*ATAN(F1)
        PI34 = F3 / (F4*PI)
        F13 = F1 / F3
        F23 = F2 / F3
        F43 = F2 * F23
        F53 = F5 / F3
        F83 = F8 / F3
        F113 = F11 / F3
        RHOO = PX
        RHO43 = RHOO**F4O3
        RHO13 = RHO43*RRHO
        RHO53 = RHOO**F53
        RHO83 = RHO53*RHOO

        RS = (PI34/PX) ** F13
!        CALL LSDAC(RS,F1,POTLC,DLDS,DLDZ,D2LDSS,D2LDSZ,D2LDZZ)

! Only the local part (EC,VCUPLDA,VCDNLDA) are used from the following call to CORPBE
        CALL CORPBE(RS,F1,PotLC,VCUPLDA,VCDNLDA,1.,1., &
     &  1.,unknown,unknown,unknown,unknown,.FALSE.)
! OBTAIN THE CONTRIBUTION FROM LDA PART AND D_EC_DRS AND D_EC_DZETA
!      EPPGGA=PotLC
!      EPPGGA_D1=(VCUPLDA-PotLC)/RT
!      EPPGGA_D2=(VCDNLDA-PotLC)/RT

        dLdz=(VCUPLDA-VCDNLDA)/F2
        dLdS=(PotLC-F1*dLdZ-(VCUPLDA+VCDNLDA)/F2)*F3/RS
!          Call lsdac(RS,F1,PotLC,dLdS,dLdZ)
!  End of modification
        EUEG = PX*POTLC
        CHI = GX/RHO83
        Z = (TX/RHO53) - CF
        KC = F1 + GCC*(CHI + Z)
        XK = CHI/KC
        ZK = Z/KC
        D = F1 - CHI/(F4*(Z + CF))
        CALL GVT4(GC,DGDX,DGDZ,XK,ZK,KC,GCC,R13,R14,R15,R16,R17,R18)
        E = D*EUEG*GC
!         WRITE (*,*) "CHI, Z, GC", CHI, Z, GC
        F = E
        RSP = -RS/(F3*PX)
        CHIG = F1/PX**F83
        CHIP = -F83*CHI/PX
        ZP = -F53 * TX/RHO83
        ZT =  F1/RHO53
        DZ = CHI/(F4*(Z + CF)*(Z + CF))
        DX = -F1/(F4*(Z + CF))
        DP = DZ*ZP + DX*CHIP
        DG = DX*CHIG
        DT = DZ*ZT
        DGDP = DGDX*CHIP + DGDZ*ZP
        DGDG = DGDX*CHIG
        DGDT = DGDZ*ZT
        EUEGP = POTLC + PX*DLDS*RSP
        FP = DP*EUEG*GC + D*EUEGP*GC + D*EUEG*DGDP
        FG = DG*EUEG*GC + D*EUEG*DGDG
        FT = DT*EUEG*GC + D*EUEG*DGDT
       ENDIF
       RETURN
       END SUBROUTINE VS98SS


!************************ SUBROUTINE MSx family ****************************
!
!***********************************************************************

      SUBROUTINE VMSXx(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ex_revTPSS,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the exchange part
! modified by Jianwei Sun 06/29/2011
      REAL(q) :: RKAPPA
      REAL(q) :: CFC
      PARAMETER (CFD=1.0_q)
      PARAMETER (CFG=0.0_q)
      REAL(q) :: CFE 
      PARAMETER (CFMUSO=0.12345679_q)
      PARAMETER (CFMUAK=0.12345679_q)
      PARAMETER (CFN=1.0_q)
      PARAMETER (CFM=1.0_q)

!end of modification

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (AX=-0.738558766382022405884230032680836_q)

      RKAPPA=MSX_RKAPPA
      CFC=MSX_CFC
      CFE=MSX_CFE

      VXD1=0._q;VXD2=0._q;
      VXDD1=0._q;VXDD2=0._q;
      AMUXD1=0._q;AMUXD2=0._q;


      CX1=10._q/81._q
      CX2=146._q/2025._q
      CX3=73._q/405._q
      CFE12=SQRT(CFE)

! Suspect that TAUWU and TAUWD are not well described. Use TAUW_TEMP=0.125_q*DRT**2/RT
! IF WANT TO TEST TAUW, SIMPLY OVERWRITE TAUW_TEMP BY TAUW
      TAUWU_TEMP=0.125_q*DRU**2._q/RU
!     TAUWU_TEMP=MIN(TAUWU_TEMP,TAUU)
      
      TAUWD_TEMP=0.125_q*DRD**2._q/RD
!     TAUWD_TEMP=MIN(TAUWD_TEMP,TAUD)

! spin up
! IN EXD1(2*RU), TAUWU AND TAUU SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RU
      DRHO=TWO*DRU
      TAUW_RHO=TWO*TAUWU_TEMP
      TAU_RHO=TWO*TAUU

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|
      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P

      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5
      ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
       FX_PBEsol=ONE+RKAPPA-RKAPPA/(ONE+(CFMUAK*P/RKAPPA))
       FX_SO=ONE+RKAPPA-RKAPPA/(ONE+((CFMUSO*P+CFC)/RKAPPA))

       ALPHA2=ALPHA*ALPHA
       ALPHA3=ALPHA*ALPHA2
       ALPHA4=ALPHA2*ALPHA2
       ALPHA6=ALPHA4*ALPHA2
       OMA=ONE-ALPHA2
       OMA2=OMA*OMA
       OMA3=OMA2*OMA
       OPA=ONE+CFD*ALPHA3+CFE*ALPHA6
       FINT=OMA3/OPA

       FX=FX_PBEsol+FINT*(FX_SO-FX_PBEsol)

!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   &  
    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

     DALPHADTAU=ONE/TAU_UNIF


! d Falpha / dy
      OPA2=OPA*OPA
      D_FINT_ALPHA=-6._q*ALPHA*OMA2/OPA-(THREE*CFD+6._q*CFE*ALPHA3)*ALPHA2*OMA3/OPA2


      D_FINT_D=D_FINT_ALPHA*DALPHAD
      D_FINT_DD=D_FINT_ALPHA*DALPHADD
      D_FINT_DTAU=D_FINT_ALPHA*DALPHADTAU


      D_FXPBESOL_P=CFMUAK/(ONE+CFMUAK*P/RKAPPA)**TWO
      D_FXSO_P=CFMUSO/(ONE+(CFMUSO*P+CFC)/RKAPPA)**TWO


      COE_FX_P=D_FXPBESOL_P+FINT*(D_FXSO_P-D_FXPBESOL_P)
      DFX_D=COE_FX_P*DPD+(FX_SO-FX_PBEsol)*D_FINT_D
      DFX_DD=COE_FX_P*DPDD+(FX_SO-FX_PBEsol)*D_FINT_DD
      DFX_DTAU=COE_FX_P*DPDTAU+(FX_SO-FX_PBEsol)*D_FINT_DTAU




!   OUTPUT THE VXD1,VXDD1 AND AMUXD1
      VXD1=EXDLDA*FX+EXLDA*DFX_D
      VXDD1=EXLDA*DFX_DD
      AMUXD1=EXLDA*DFX_DTAU
      EX_REVTPSS=0._q
      EX_REVTPSS=EX_REVTPSS+EXLDA*FX

! spin down
! IN EXD1(2*RD), TAUWD AND TAUD SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RD
      DRHO=TWO*DRD
      TAUW_RHO=TWO*TAUWD_TEMP
      TAU_RHO=TWO*TAUD

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|

      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P

      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5

      ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      

       FX_PBEsol=ONE+RKAPPA-RKAPPA/(ONE+(CFMUAK*P/RKAPPA))
       FX_SO=ONE+RKAPPA-RKAPPA/(ONE+((CFMUSO*P+CFC)/RKAPPA))

       ALPHA2=ALPHA*ALPHA
       ALPHA3=ALPHA*ALPHA2
       ALPHA4=ALPHA2*ALPHA2
       ALPHA6=ALPHA4*ALPHA2
       OMA=ONE-ALPHA2
       OMA2=OMA*OMA
       OMA3=OMA2*OMA
       OPA=ONE+CFD*ALPHA3+CFE*ALPHA6
       FINT=OMA3/OPA

       FX=FX_PBEsol+FINT*(FX_SO-FX_PBEsol)


!    NOW, DERIVATIVES COME
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q

      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5

     DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   & 
    &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)

     DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)

     DALPHADTAU=ONE/TAU_UNIF


! d Falpha / dy
      OPA2=OPA*OPA
      D_FINT_ALPHA=-6._q*ALPHA*OMA2/OPA-(THREE*CFD+6._q*CFE*ALPHA3)*ALPHA2*OMA3/OPA2



      D_FINT_D=D_FINT_ALPHA*DALPHAD
      D_FINT_DD=D_FINT_ALPHA*DALPHADD
      D_FINT_DTAU=D_FINT_ALPHA*DALPHADTAU


      D_FXPBESOL_P=CFMUAK/(ONE+CFMUAK*P/RKAPPA)**TWO
      D_FXSO_P=CFMUSO/(ONE+(CFMUSO*P+CFC)/RKAPPA)**TWO


      COE_FX_P=D_FXPBESOL_P+FINT*(D_FXSO_P-D_FXPBESOL_P)
      DFX_D=COE_FX_P*DPD+(FX_SO-FX_PBEsol)*D_FINT_D
      DFX_DD=COE_FX_P*DPDD+(FX_SO-FX_PBEsol)*D_FINT_DD
      DFX_DTAU=COE_FX_P*DPDTAU+(FX_SO-FX_PBEsol)*D_FINT_DTAU


!   OUTPUT THE VXD2,VXDD2 AND AMUXD2
      VXD2=EXDLDA*FX+EXLDA*DFX_D
      VXDD2=EXLDA*DFX_DD
      AMUXD2=EXLDA*DFX_DTAU

      EX_REVTPSS=EX_REVTPSS+EXLDA*FX
      EX_REVTPSS=EX_REVTPSS/TWO
      RETURN
      END SUBROUTINE VMSXx


      SUBROUTINE VMSXc(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_revTPSS,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

! the following parameters are given by Perdew et.al.
! the coefficients for the correlatin part
      PARAMETER (CFD=2.8_q)

! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)

      VCD1=0._q;VCD2=0._q;
      VCDD1=0._q;VCDD2=0._q;
      AMUCD1=0._q;AMUCD2=0._q;

      RT=RU+RD
      YA=DRU**2._q
      YB=DRD**2._q
      Y=DRT**2._q
!    YC IS DEL RU DOT DEL RD
      YC=(Y-YA-YB)/2._q
      TAUW=1._q/8._q*(Y/RT)
      TAU=TAUU+TAUD

!     TAUW=MIN(TAUW,TAU)

      CALL CORGGA_REVTPSS(RU,RD,DRU,DRD,DRT,EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

      VCD1=EPPGGA+RT*EPPGGA_D1

      VCD2=EPPGGA+RT*EPPGGA_D2

      VCDD1=RT*EPPGGA_DD1

      VCDD2=RT*EPPGGA_DD2

      AMUCD1=0._q
      AMUCD2=AMUCD1

      EC_REVTPSS=RT*EPPGGA


      RETURN
      END SUBROUTINE VMSXc


!************************ SUBROUTINE VSCANx ****************************
!
! calculates the first order derivatives of Ex wrt n and |grad(n)|
! Perdew et. al. PRL (2009)
!
! Written by Jianwei Sun
!
! everything in Hartree units
!
! ATTANTION: Every values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VXD1 VXD2                    THE DERIVATIVES OF EX WRT n 
! VXDD1,VXDD2                  THE DERIVATIVES OF EX WRT |grad n| 
! AMUXD1, AMUXD2               THE DERIVATIVES OF EX WRT TAU
!
!***********************************************************************

      SUBROUTINE VSCANx(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ex_SCAN,VXD1,VXDD1,VXD2,VXDD2,AMUXD1,AMUXD2)

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)

      PARAMETER (CFC1X=0.667_q)
      PARAMETER (CFC2X=0.800_q)
      PARAMETER (CFD1X=1.240_q)
      PARAMETER (CFK1=0.065_q)
! other parameters
      PARAMETER (ONE=1._q)
      PARAMETER (TWO=2._q)
      PARAMETER (THREE=3._q)
      PARAMETER (FOUR=4._q)
      PARAMETER (THRD=1._q/3._q)
      PARAMETER (THRD2=2._q*THRD)
      PARAMETER (THRD4=4._q*THRD)
      PARAMETER (THRD5=1._q+THRD2)
      PARAMETER (THRD8=1._q+THRD5)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (AX=-0.738558766382022405884230032680836_q)

      VXD1=0._q;VXD2=0._q;
      VXDD1=0._q;VXDD2=0._q;
      AMUXD1=0._q;AMUXD2=0._q;


      CX1=10._q/81._q
      CX2=146._q/2025._q
      CX3=73._q/405._q
      CFE12=SQRT(CFE)

! Suspect that TAUWU and TAUWD are not well described. Use TAUW_TEMP=0.125_q*DRT**2/RT
! IF WANT TO TEST TAUW, SIMPLY OVERWRITE TAUW_TEMP BY TAUW
      TAUWU_TEMP=0.125_q*DRU**2._q/RU
!     TAUWU_TEMP=MIN(TAUWU_TEMP,TAUU)
      
      TAUWD_TEMP=0.125_q*DRD**2._q/RD
!     TAUWD_TEMP=MIN(TAUWD_TEMP,TAUD)

! spin up
! IN EXD1(2*RU), TAUWU AND TAUU SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RU
      DRHO=TWO*DRU
      TAUW_RHO=TWO*TAUWU_TEMP
      TAU_RHO=TWO*TAUU

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|
      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P

      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5
      ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q
      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5
      DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   &  
     &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)
      DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)
      DALPHADTAU=ONE/TAU_UNIF


! calculate the exchange enhancement factor and its derivatives wrt p
! and alpha
       call metaggafx(cfc1x,cfc2x,cfd1x,cfk1, &
     & p, alpha, Fx, fx1p, fx1a, fx2p2, fx2pa, fx2a2)

! Calculate fx DERIVATIVES WRT n, |grad n|, and tau
      DFX_D=fx1p*DPD+fx1a*DALPHAD
      DFX_DD=fx1p*DPDD+fx1a*DALPHADD
      DFX_DTAU=fx1a*DALPHADTAU



!   OUTPUT THE VXD1,VXDD1 AND AMUXD1
      VXD1=EXDLDA*FX+EXLDA*DFX_D
      VXDD1=EXLDA*DFX_DD
      AMUXD1=EXLDA*DFX_DTAU
      EX_SCAN=0._q
      EX_SCAN=EX_SCAN+EXLDA*FX

! spin down
! IN EXD1(2*RD), TAUWD AND TAUD SCALES AS 2 AND TAU0 SCALES AS 2**FTHRD
      RHO=TWO*RD
      DRHO=TWO*DRD
      TAUW_RHO=TWO*TAUWD_TEMP
      TAU_RHO=TWO*TAUD

!----------------------------------------------------------------------
! construct LDA exchange energy density AND ITS DERIVATIVE WRT n
      EXUNIF=AX*RHO**THRD
      EXLDA=EXUNIF*RHO
      EXDLDA=EXUNIF*THRD4


! CONSTRUCT FX AND ITS DERIVATIVES WRT n AND |grad n|
      P=(DRHO)**TWO/(FOUR*(THREE*PISQ)**THRD2*(RHO)**THRD8)
      P2=P*P

      TAU0=0.3_q*(THREE*PISQ)**THRD2*(RHO)**THRD5
      ALPHA=(TAU_RHO-TAUW_RHO)/TAU0
      DPD=-THRD8*P/RHO
      DPDD=TWO*P/DRHO
      DPDTAU=0._q
      TAU_UNIF=THREE/10._q*(THREE*PI**TWO)**THRD2*RHO**THRD5
      DALPHAD=-TAU_RHO*(THREE*PI**TWO*RHO)**THRD2/(TWO*TAU_UNIF**TWO)+   &
     &   DRHO**TWO/RHO**(11._q/THREE)*(10._q/9._q/(THREE*PI**TWO)**THRD2)
      DALPHADD=-DRHO/(FOUR*RHO*TAU_UNIF)
      DALPHADTAU=ONE/TAU_UNIF


! calculate the exchange enhancement factor and its derivatives wrt p
! and alpha
       call metaggafx(cfc1x,cfc2x,cfd1x,cfk1, &
     & p, alpha, Fx, fx1p, fx1a, fx2p2, fx2pa, fx2a2)

! Calculate fx DERIVATIVES WRT n, |grad n|, and tau
      DFX_D=fx1p*DPD+fx1a*DALPHAD
      DFX_DD=fx1p*DPDD+fx1a*DALPHADD
      DFX_DTAU=fx1a*DALPHADTAU



!   OUTPUT THE VXD2,VXDD2 AND AMUXD2
      VXD2=EXDLDA*FX+EXLDA*DFX_D
      VXDD2=EXLDA*DFX_DD
      AMUXD2=EXLDA*DFX_DTAU

      EX_SCAN=EX_SCAN+EXLDA*FX
      EX_SCAN=EX_SCAN/TWO
      RETURN
      END SUBROUTINE VSCANx


      SUBROUTINE metaggafx(cfc1,cfc2,cfd1,cfk1, &
     &            p, a, fx, fx1p, fx1a, fx2p2, fx2pa, fx2a2)

      IMPLICIT REAL*8 (A-H,O-Z)

      INTEGER I

      PARAMETER (cfk0=0.1740d0)
      PARAMETER (CFA1=4.9479d0)

      PARAMETER (CFMUAK=10.00d0/81.0d0)
      PARAMETER (CFB1=0.156632d0)
      PARAMETER (CFB2=0.12083d0)
      PARAMETER (CFB3=0.5d0)

      PARAMETER (TWO=2.0d0)
      PARAMETER (THREE=3.0d0)
      PARAMETER (FOUR=4.0d0)

      fx=0.0d0
      fx1p=0.0d0
      fx1a=0.0d0
      fx2p2=0.0d0
      fx2pa=0.0d0
      fx2a2=0.0d0


      P2=P*P
      oma=1.0d0-a
      oma2=oma*oma

          HX0=1.0d0+CFK0

          cfb4=cfmuak**two/cfk1-0.112654d0
          if (cfb4 .gt. 0.0d0) then
             wfac=cfb4*p2*dexp(-cfb4*p/cfmuak)
             d_wfac_dp=cfb4*p*dexp(-cfb4*p/cfmuak)*(two-cfb4*p/cfmuak)
          else
             wfac=cfb4*p2*dexp(cfb4*p/cfmuak)
             d_wfac_dp=cfb4*p*dexp(cfb4*p/cfmuak)*(two+cfb4*p/cfmuak)
          endif
          vfac=cfb1*p+cfb2*oma*dexp(-cfb3*oma2)
          yfac=cfmuak*p+wfac+vfac**two
          hx1=1.0d0+cfk1-cfk1/(1.0d0+yfac/cfk1)

          FA=0.d0
          if (a .lt. 1.0d0) then
             FA=dExp(-cfc1*a/oma)
          endif
          if (a .gt. 1.0d0) then
             FA=-cfd1*dExp(cfc2/oma)
          endif

         p14=p**(1.0d0/four)
         gx=1.d0
         if (p .gt. 0.d0) then
             gx=1.0d0-dexp(-cfa1/p14)
         endif

          Fx1 = hx1+FA*(hx0-hx1)

          Fx = Fx1*gx



            d_hx0_dp=0.d0

            d_vfac_dp=cfb1
            d_yfac_dp=cfmuak+d_wfac_dp+two*vfac*d_vfac_dp
            d_hx1_dp=d_yfac_dp/(1.0d0+yfac/cfk1)**two

            d_vfac_da=-cfb2*(1.0d0-two*cfb3*oma2)*dexp(-cfb3*oma2)
            d_yfac_da=two*vfac*d_vfac_da
            d_hx1_da=d_yfac_da/(1.0d0+yfac/cfk1)**two

          d_FA_da=0.d0
          if (a .lt. 1.0d0) then
            d_FA_da=-cfc1*dExp(-cfc1*a/oma)/oma**two
          endif
          if (a .gt. 1.0d0) then
            d_FA_da=-cfc2*cfd1*dExp(cfc2/oma)/oma**two
          endif

         d_gx_dp=0.d0
         if (p .gt. 0.d0) then
             d_gx_dp=-cfa1/four/p/p14*dexp(-cfa1/p14)
         endif

            d_Fx1_dp=d_hx1_dp+FA*(d_hx0_dp-d_hx1_dp)

            d_Fx1_da=(1.0d0-FA)*d_hx1_da+d_FA_da*(hx0-hx1)


            fx1p=d_Fx1_dp*gx+Fx1*d_gx_dp

            fx1a=d_Fx1_da*gx

      RETURN
      END SUBROUTINE metaggafx


!************************ SUBROUTINE VSCANc ****************************
!
! RU,RD                        density up,down
! DRU, DRD                     abs. val. gradient of density up/down
! DRT                          abs. val. gradient of total density
! TAUU,TAUD                    kinetic energy density up/down
! TAUWU,TAUWD                  Weizsaecker kinetic energy density up/down
! VCD1 VCD2                    THE DERIVATIVES OF EC WRT n 
! VCDD1,VCDD2                  THE DERIVATIVES OF EC WRT |grad n| 
! AMUCD1, AMUCD2                   THE DERIVATIVES OF EC WRT TAU
!
!***********************************************************************

      SUBROUTINE VSCANc(&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD,& 
     &   Ec_SCAN,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      IMPLICIT REAL*8 (A-H,O-Z)

! the coefficients for the exchange part
      PARAMETER (CFC1C=0.640d0)
      PARAMETER (CFC2C=1.500d0)
      PARAMETER (CFD1C=0.700d0)



! other parameter
      PARAMETER (ONE=1.0d0)
      PARAMETER (TWO=2.0d0)
      PARAMETER (THREE=3.0d0)
      PARAMETER (FOUR=4.0d0)
      PARAMETER (PI =3.141592653589793238d0)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (THRD=1.0d0/3.0d0)
      PARAMETER (THRD2=2.0d0*THRD)
      PARAMETER (THRD4=4.0d0*THRD)
      PARAMETER (THRD5=1.0d0+THRD2)
      PARAMETER (THRD8=1.0d0+THRD5)

      call Vmetaggac(cfc1c,cfc2c,cfd1c,&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD,&
     &   Ec_SCAN,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      RETURN
      END SUBROUTINE VSCANc


      SUBROUTINE Vmetaggac(cfc1,cfc2,cfd1,&
     &   RU,RD,DRU,DRD,DRT,TAUU,TAUD, &
     &   Ec_SCAN,VCD1,VCDD1,VCD2,VCDD2,AMUCD1,AMUCD2)

      IMPLICIT REAL*8 (A-H,O-Z)

      PARAMETER (TWO=2.0d0)
      PARAMETER (THREE=3.0d0)
      PARAMETER (FOUR=4.0d0)
      PARAMETER (PI =3.141592653589793238d0)
      PARAMETER (PISQ=PI*PI)
      PARAMETER (THRD=1.0d0/3.0d0)
      PARAMETER (THRD2=2.0d0*THRD)
      PARAMETER (THRD4=4.0d0*THRD)
      PARAMETER (THRD5=1.0d0+THRD2)
      PARAMETER (THRD8=1.0d0+THRD5)


      VCD1=0.0d0
      VCD2=0.0d0
      VCDD1=0.0d0
      VCDD2=0.0d0
      AMUCD1=0.0d0
      AMUCD2=0.0d0

      RT=RU+RD
      YA=DRU**2.0d0
      YB=DRD**2.0d0
      Y=DRT**2.0d0
      YC=(Y-YA-YB)/2.0d0
      TAUW=1.0d0/8.0d0*(Y/RT)
      TAU=TAUU+TAUD
      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.99999999999990d0),0.99999999999990d0)
      DS_ZETA=((1.0d0+ZETA)**THRD5+(1.0d0-ZETA)**THRD5)/TWO
      DX_ZETA=((1.0d0+ZETA)**THRD4+(1.0d0-ZETA)**THRD4)/TWO
      TAU0=0.30d0*(THREE*PISQ)**THRD2*RT**THRD5*DS_ZETA
      ALPHA=(TAU-TAUW)/TAU0

      D_ZETA_D1=TWO*RD/RT**TWO
      D_ZETA_D2=-TWO*RU/RT**TWO
      D_DS_ZETA_D1=THRD5*((1.0d0+ZETA)**THRD2 &
     &             -(1.0d0-ZETA)**THRD2)*D_ZETA_D1/TWO
      D_DS_ZETA_D2=THRD5*((1.0d0+ZETA)**THRD2 &
     &             -(1.0d0-ZETA)**THRD2)*D_ZETA_D2/TWO
      DTAU0D1=THRD5*TAU0/RT+TAU0*D_DS_ZETA_D1/DS_ZETA
      DTAU0D2=THRD5*TAU0/RT+TAU0*D_DS_ZETA_D2/DS_ZETA

      DTAUWD1=-TAUW/RT
      DTAUWD2=DTAUWD1

      D_ALPHA_D1=(-DTAUWD1-ALPHA*DTAU0D1)/TAU0
      D_ALPHA_D2=(-DTAUWD2-ALPHA*DTAU0D2)/TAU0

      DYCDD1=YC/DRU
      DYCDD2=YC/DRD
      DTAUWDD1=1.0d0/FOUR/RT*(DRU+DYCDD1)
      DTAUWDD2=1.0d0/FOUR/RT*(DRD+DYCDD2)

      D_ALPHA_DD1=-DTAUWDD1/TAU0
      D_ALPHA_DD2=-DTAUWDD2/TAU0

      D_ALPHA_DTAU=1.0d0/TAU0

      ALPHA2=ALPHA*ALPHA
      ALPHA3=ALPHA*ALPHA2
      ALPHA4=ALPHA*ALPHA3

      FUNKC_ALPHA=0.0d0
      IF (ALPHA .LT. 1.0d0) THEN
          FUNKC_ALPHA=EXP(CFC1*ALPHA/(ALPHA-1.0d0))
       ENDIF
      IF (ALPHA .GT. 1.0d0) THEN
          FUNKC_ALPHA=-cfd1*EXP(-CFC2/(ALPHA-1.0d0))
       ENDIF

      D_FUNKC_ALPHA_DALPHA=0.0d0
      IF (ALPHA .lt. 1.0d0) THEN
          D_FUNKC_ALPHA_DALPHA=-cfc1*FUNKC_ALPHA &
     &                         /(ALPHA-1.0d0)**TWO 
       ENDIF
      IF (ALPHA .gt. 1.0d0) THEN
          D_FUNKC_ALPHA_DALPHA=cfc2*FUNKC_ALPHA &
     &                         /(ALPHA-1.0d0)**TWO
      endif

      CALL CORGGA_0(RU,RD,DRU,DRD,DRT,&
     &        EPPGGA0,EPPGGA0_D1,EPPGGA0_D2,EPPGGA0_DD1,EPPGGA0_DD2)
      CALL CORGGA_1(RU,RD,DRU,DRD,DRT, &
     &       EPPGGA1,EPPGGA1_D1,EPPGGA1_D2,EPPGGA1_DD1,EPPGGA1_DD2)


      EPP=EPPGGA1+FUNKC_ALPHA*(EPPGGA0-EPPGGA1)
      D_EPP_D1=EPPGGA1_D1 &
     &          +(EPPGGA0-EPPGGA1)*D_FUNKC_ALPHA_DALPHA*D_ALPHA_D1 &
     &          +FUNKC_ALPHA*(EPPGGA0_D1-EPPGGA1_D1) 
      D_EPP_D2=EPPGGA1_D2 &
     &          +(EPPGGA0-EPPGGA1)*D_FUNKC_ALPHA_DALPHA*D_ALPHA_D2 &
     &          +FUNKC_ALPHA*(EPPGGA0_D2-EPPGGA1_D2)
      D_EPP_DD1=EPPGGA1_DD1 &
     &          +(EPPGGA0-EPPGGA1)*D_FUNKC_ALPHA_DALPHA*D_ALPHA_DD1 &
     &          +FUNKC_ALPHA*(EPPGGA0_DD1-EPPGGA1_DD1)
      D_EPP_DD2=EPPGGA1_DD2 &
     &          +(EPPGGA0-EPPGGA1)*D_FUNKC_ALPHA_DALPHA*D_ALPHA_DD2 &
     &          +FUNKC_ALPHA*(EPPGGA0_DD2-EPPGGA1_DD2) 
      D_EPP_DTAU=(EPPGGA0-EPPGGA1)*D_FUNKC_ALPHA_DALPHA*D_ALPHA_DTAU



      VCD1=EPP+RT*D_EPP_D1
      VCD2=EPP+RT*D_EPP_D2
      VCDD1=RT*D_EPP_DD1
      VCDD2=RT*D_EPP_DD2
      AMUCD1=RT*D_EPP_DTAU
      AMUCD2=AMUCD1

      EC_SCAN=RT*EPP

      RETURN
      END SUBROUTINE Vmetaggac


      SUBROUTINE CORGGA_1( &
     &   RU,RD,DRU,DRD,DRT, &
     &   EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)


      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER (TWO=2.0d0)
      PARAMETER (THREE=3.0d0)
      PARAMETER (FOUR=4.0d0)
      PARAMETER (FIVE=5.0d0)
      PARAMETER (SIX=6.0d0)
      PARAMETER (THRD=1.0d0/3.0d0)
      PARAMETER (THRD2=2.0d0*THRD)
      PARAMETER (THRD4=4.0d0*THRD)
      PARAMETER (THRD5=1.0d0+THRD2)
      PARAMETER (THRD8=1.0d0+THRD5)
      PARAMETER (PI =3.141592653589793238d0)
      PARAMETER (PISQ=PI*PI)
  
      PARAMETER (GAMMA=0.031090690869654895034940863712730d0)
      PARAMETER (BETA_mb=0.066724550603149220d0)

      EPPGGA  =0.0d0
      EPPGGA_D1=0.0d0
      EPPGGA_D2=0.0d0
      EPPGGA_DD1=0.0d0
      EPPGGA_DD2=0.0d0

      RT=RU+RD
      YA=DRU**2.0d0
      YB=DRD**2.0d0
      Y=DRT**2.0d0
      YC=(Y-YA-YB)/2.0d0

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.99999999999990d0),0.99999999999990d0)
      DZETAD1=TWO*RD/RT**TWO
      DZETAD2=-TWO*RU/RT**TWO


      DTHRD=exp(log(RT)*THRD)
      RS=(0.750d0/PI)**THRD/DTHRD
      DRSD1=-THRD/RT*RS
      DRSD2=-THRD/RT*RS


      PHI = (exp((TWO*THRD)*log(1.0d0+ZETA)) &
     &       +exp((TWO*THRD)*log(1.0d0-ZETA)))/2.0d0
      D_PHI_DZETA=THRD*((1.0d0+ZETA)**(-THRD)-(1.0d0-ZETA)**(-THRD))
      D_PHI_D1=D_PHI_DZETA*DZETAD1
      D_PHI_D2=D_PHI_DZETA*DZETAD2

      AFIX_T=SQRT(PI/FOUR)*(9.0d0*PI/FOUR)**(1.0d0/SIX)
      S=DRT/(TWO*(THREE*PISQ)**THRD*RT**THRD4)
      DSD1=-THRD4*S/RT
      DSD2=-THRD4*S/RT
      GNGNU=1.0d0/(TWO*DRT)*(TWO*DRU+TWO*YC/DRU)
      GNGND=1.0d0/(TWO*DRT)*(TWO*DRD+TWO*YC/DRD)
      DSDD1=S/DRT*GNGNU
      DSDD2=S/DRT*GNGND

      T=AFIX_T*S/SQRT(RS)/PHI
      T2 = T*T
      T4 = T2*T2

      DTD1=AFIX_T*(PHI*RS*DSD1-0.50d0*S*PHI*DRSD1-S*RS*D_PHI_D1) &
     &           /(PHI**TWO*RS**(THREE/TWO))
      DTD2=AFIX_T*(PHI*RS*DSD2-0.50d0*S*PHI*DRSD2-S*RS*D_PHI_D2) &
     &           /(PHI**TWO*RS**(THREE/TWO))
      DTDD1=AFIX_T*DSDD1/(SQRT(RS)*PHI)
      DTDD2=AFIX_T*DSDD2/(SQRT(RS)*PHI)

      FK=(3.0d0*PI*PI)**THRD*DTHRD
      SK = SQRT(4.00d0*FK/PI)
      CALL CORPBE_revtpss(RS,ZETA,EC,VCUPLDA,VCDNLDA,PHI,SK, &
     &           T,H,DVCUP,DVCDN,ECQ,.FALSE.)
      D_EC_DZETA=(VCUPLDA-VCDNLDA)/TWO
      D_EC_DRS=(EC-ZETA*D_EC_DZETA-(VCUPLDA+VCDNLDA)/TWO)*THREE/RS

      AFACTOR=0.10d0
      BFACTOR=0.17780d0
      BETA_NUM=1.0d0 + AFACTOR*RS
      BETA_DEN=1.0d0+ BFACTOR*RS
      D_BETA_NUM=AFACTOR
      D_BETA_DEN=BFACTOR
      BETA = BETA_MB*BETA_NUM/BETA_DEN
      D_BETA_DRS=BETA_MB*(BETA_DEN*D_BETA_NUM-BETA_NUM*D_BETA_DEN) &
     &                  /BETA_DEN**TWO
      
 
      PHI3=PHI**THREE
      PON=-EC/(PHI3*gamma)
      W=DEXP(PON)-1.0d0
      D_W_DRS=-(W+1.0d0)*D_EC_DRS/(GAMMA*PHI3)
      D_W_DZETA=-(W+1.0d0)/(GAMMA*PHI3) &
     &                  *(D_EC_DZETA-THREE*EC*D_PHI_DZETA/PHI)
      D_W_DT=0.0d0


      A=BETA/(GAMMA*W)
      D_A_DRS=(W*D_BETA_DRS-BETA*D_W_DRS)/(GAMMA*W**TWO)
      D_A_DZETA=-BETA*D_W_DZETA/(GAMMA*W**TWO)
      D_A_DT=0.0d0

      V=A*T2
      D_V_DRS=T2*D_A_DRS
      D_V_DZETA=T2*D_A_DZETA
      D_V_DT=T2*D_A_DT+TWO*A*T
        

      FUNKG=1.0d0/(1.0d0+4.0d0*V)**0.250d0
      D_FUNKG_DV=-1.0d0/(1.0d0+4.0d0*V)**1.250d0


      
      HCORE=1.0d0+W*(1.0d0-FUNKG)
      AH=GAMMA*PHI3
      H=AH*DLOG(HCORE)
   
      DH1=1.0d0-FUNKG
      DH2=W*D_FUNKG_DV
      D_H_DRS=(DH1*D_W_DRS-DH2*D_V_DRS)*AH/HCORE
      D_H_DZETA=THREE*H*D_PHI_DZETA/PHI &
     &          +(DH1*D_W_DZETA-DH2*D_V_DZETA)*AH/HCORE
      D_H_DT=(DH1*D_W_DT-DH2*D_V_DT)*AH/HCORE

      EPPGGA=EC+H
      EPPGGA_D1=D_EC_DRS*DRSD1+D_EC_DZETA*DZETAD1+D_H_DRS*DRSD1 &
     &                        +D_H_DZETA*DZETAD1+D_H_DT*DTD1
      EPPGGA_D2=D_EC_DRS*DRSD2+D_EC_DZETA*DZETAD2+D_H_DRS*DRSD2 &
     &                        +D_H_DZETA*DZETAD2+D_H_DT*DTD2
      EPPGGA_DD1=D_H_DT*DTDD1
      EPPGGA_DD2=D_H_DT*DTDD2

      RETURN
      END SUBROUTINE CORGGA_1


      SUBROUTINE CORGGA_0(RU,RD,DRU,DRD,DRT, &
     &   EPPGGA,EPPGGA_D1,EPPGGA_D2,EPPGGA_DD1,EPPGGA_DD2)

      IMPLICIT REAL*8 (A-H,O-Z)

      PARAMETER (CFB1=0.0285764d0)
      PARAMETER (CFB2=0.0889d0) 
      PARAMETER (CFB3=0.125541d0)
      PARAMETER (cfkaiLD=0.12802585262625815d0)

      PARAMETER (TWO=2.0d0)
      PARAMETER (THREE=3.0d0)
      PARAMETER (FOUR=4.0d0)
      PARAMETER (FIVE=5.0d0)
      PARAMETER (SIX=6.0d0)
      PARAMETER (THRD=1.0d0/3.0d0)
      PARAMETER (THRD2=2.0d0*THRD)
      PARAMETER (THRD4=4.0d0*THRD)
      PARAMETER (THRD5=1.0d0+THRD2)
      PARAMETER (THRD8=1.0d0+THRD5)
      PARAMETER (PI =3.141592653589793238d0)
      PARAMETER (PISQ=PI*PI)

      PARAMETER (GAMMA=0.031090690869654895034940863712730d0)
      PARAMETER (BETA_mb=0.066724550603149220d0)

      EPPGGA  =0.0d0
      EPPGGA_D1=0.0d0
      EPPGGA_D2=0.0d0
      EPPGGA_DD1=0.0d0
      EPPGGA_DD2=0.0d0
      AX_LDA=-THREE/(FOUR*PI)*(9.00d0*PI/FOUR)**THRD

      RT=RU+RD
      YA=DRU**2.0d0
      YB=DRD**2.0d0
      Y=DRT**2.0d0
      YC=(Y-YA-YB)/2.0d0

      ZETA=(RU-RD)/RT
      ZETA=MIN(MAX(ZETA,-0.99999999999990d0),0.99999999999990d0)
      DZETAD1=TWO*RD/RT**TWO
      DZETAD2=-TWO*RU/RT**TWO


      DTHRD=exp(log(RT)*THRD)
      RS=(0.750d0/PI)**THRD/DTHRD
      DRSD1=-THRD/RT*RS
      DRSD2=-THRD/RT*RS


      PHI = (exp((TWO*THRD)*log(1.0d0+ZETA)) &
     &           +exp((TWO*THRD)*log(1.0d0-ZETA)))/2.0d0
      D_PHI_DZETA=THRD*((1.0d0+ZETA)**(-THRD)-(1.0d0-ZETA)**(-THRD))
      D_PHI_D1=D_PHI_DZETA*DZETAD1
      D_PHI_D2=D_PHI_DZETA*DZETAD2

      AFIX_T=SQRT(PI/FOUR)*(9.0d0*PI/FOUR)**(1.0d0/SIX)
      S=DRT/(TWO*(THREE*PISQ)**THRD*RT**THRD4)
      DSD1=-THRD4*S/RT
      DSD2=-THRD4*S/RT
      GNGNU=1.0d0/(TWO*DRT)*(TWO*DRU+TWO*YC/DRU)
      GNGND=1.0d0/(TWO*DRT)*(TWO*DRD+TWO*YC/DRD)
      DSDD1=S/DRT*GNGNU
      DSDD2=S/DRT*GNGND

      RSHALF=SQRT(RS)
      FACTOR1=1.0d0+CFB2*RSHALF+CFB3*RS
      EC0LDA=-CFB1/FACTOR1

      D_EC0LDA_DRS=(CFB3+CFB2/RSHALF/TWO)*EC0LDA**TWO/cfb1
      
      dx_zeta=((1.0d0+zeta)**THRD4+(1.0d0-zeta)**THRD4)/two
      d_dx_zeta_dzeta=two*((1.0d0+zeta)**THRD-(1.0d0-zeta)**THRD)/three
      TWO13=TWO**THRD
!      gc_zeta=(two13-dx_zeta)/(two13-1.0d0)
!      d_gc_zeta_dzeta=-d_dx_zeta_dzeta/(two13-1.0d0)

      gc_zeta=(1.0d0-2.363d0*(dx_zeta-1.0d0))*(1.0d0-zeta**12.0d0)
      d_gc_zeta_dzeta=-(1.0d0-2.363d0*(dx_zeta-1.0d0))*12.d0*zeta**11.d0
      d_gc_zeta_dzeta=d_gc_zeta_dzeta -2.363d0*d_dx_zeta_dzeta*(1.0d0-zeta**12.0d0)


      w0=dexp(-EC0LDA/cfb1)-1.0d0
      d_w0_drs=-(w0+1.0d0)*D_EC0LDA_DRS/cfb1
      
      gfunkinf=1.0d0/(1.0d0+four*cfkaiLD*s*s)**(1.0d0/four)
      d_gfunkinf_ds=-two*cfkaiLD*s*gfunkinf**five

      hcore0=1.0d0+w0*(1.0d0-gfunkinf)
      h0=cfb1*dlog(hcore0)
      d_h0_drs=cfb1*d_w0_drs*(1.0d0-gfunkinf)/hcore0
      d_h0_ds=-cfb1*w0*d_gfunkinf_ds/hcore0


      EPPGGA=(EC0LDA+h0)*gc_zeta

      d_EPPGGA_dzeta=(EC0LDA+h0)*d_gc_zeta_dzeta
      d_EPPGGA_drs=(D_EC0LDA_DRS+d_h0_drs)*gc_zeta
      d_EPPGGA_ds=d_h0_ds*gc_zeta

      EPPGGA_D1=d_EPPGGA_dzeta*DZETAD1+d_EPPGGA_drs*DRSD1 &
     &          +d_EPPGGA_ds*DSD1
      EPPGGA_D2=d_EPPGGA_dzeta*DZETAD2+d_EPPGGA_drs*DRSD2 &
     &          +d_EPPGGA_ds*DSD2



      EPPGGA_DD1=d_EPPGGA_ds*DSDD1
      EPPGGA_DD2=d_EPPGGA_ds*DSDD2

      RETURN
      END SUBROUTINE CORGGA_0


!***********************************************************************
!***********************************************************************
  END MODULE metalib
!***********************************************************************
!***********************************************************************


!***********************************************************************
!***********************************************************************
!
!***********************************************************************
!***********************************************************************

  MODULE meta
    USE prec
    USE setxcmeta
    
    ! Methods that are private
    PRIVATE :: FEXCGS_METAGGA
    
    TYPE tau_handle
       COMPLEX(q),POINTER :: TAU(:,:)    ! kinetic energy density
       COMPLEX(q),POINTER :: TAUL(:,:)   ! kinetic energy density (previous step)
       COMPLEX(q),POINTER :: TAUW(:,:)   ! Weisaecker kinetic energy density
       RGRID, POINTER     :: TAUC(:)     ! kinetic energy density of the core electrons
    END TYPE tau_handle
    !
    ! parameters that determine the spherical integration mesh 
    ! for potentials in the METAGGA
    !
    INTEGER, PARAMETER, PRIVATE :: PHPTS_FACT=3, THPTS_FACT=3        

    ! auxiliary functions needed by RAD_KINEDEN and RAD_PROJ_METAGGA,
    ! allocated and constructed by RAD_AUXILIARY_FUNCTIONS_METAGGA
    REAL(q), ALLOCATABLE, PRIVATE, SAVE :: X_YLM_X_YLM(:,:,:)
    REAL(q), ALLOCATABLE, PRIVATE, SAVE :: YLMD_YLMD(:,:,:)
    REAL(q), ALLOCATABLE, PRIVATE, SAVE :: X_YLM_YLMD(:,:,:)

   CONTAINS
      
!***********************************************************************
!
!***********************************************************************

      SUBROUTINE ALLOCATE_MU(MU,MUTOT,GRID,GRIDC,WDES)
      USE mgrid
      USE ini
      USE wave
      IMPLICIT NONE
      TYPE (grid_3d) GRID, GRIDC
      TYPE (wavedes) WDES
      COMPLEX(q),POINTER :: MUTOT(:,:)  ! d e_xc / d tau
      RGRID , POINTER    :: MU(:,:)     ! soft version of the above

      IF (LCALCMU()) THEN
         ALLOCATE(MU(DIMREAL(GRID%MPLWV),WDES%NCDIJ),MUTOT(GRIDC%MPLWV,WDES%NCDIJ))
         CALL REGISTER_ALLOCATE(16._q*(SIZE(MUTOT)+SIZE(MU)), "grid")
         MU=0; MUTOT=0
      ELSE
         NULLIFY(MUTOT, MU)
      ENDIF

      RETURN
      END SUBROUTINE ALLOCATE_MU


!***********************************************************************
!
!***********************************************************************

      SUBROUTINE DEALLOCATE_MU(MU,MUTOT)
      USE mgrid
      USE ini
      IMPLICIT NONE
      COMPLEX(q),POINTER :: MUTOT(:,:)  ! d e_xc / d tau
      RGRID , POINTER    :: MU(:,:)     ! soft version of the above

      IF (LCALCMU()) THEN
         CALL DEREGISTER_ALLOCATE(16._q*(SIZE(MUTOT)+SIZE(MU)), "grid")
         DEALLOCATE(MU,MUTOT)
      ENDIF
      NULLIFY(MUTOT, MU)

      RETURN      
      END SUBROUTINE DEALLOCATE_MU


!!!***********************************************************************
!!!
!!! generate the tau_handle
!!!
!!!***********************************************************************
!!
!!      SUBROUTINE GENERATE_TAU_HANDLE(KINEDEN, GRIDC, NCDIJ)
!!      USE mgrid
!!      IMPLICIT NONE
!!      TYPE(tau_handle), POINTER :: KINEDEN
!!      TYPE(grid_3d) GRIDC
!!      INTEGER NCDIJ
!!
!!      IF (.NOT.ASSOCIATED(KINEDEN).AND.LDO_METAGGA()) THEN
!!         ALLOCATE(KINEDEN)
!!         ALLOCATE(KINEDEN%TAU(GRIDC%MPLWV,NCDIJ),KINEDEN%TAUC(GRIDC%RL%NP))
!!         KINEDEN%TAU=0; KINEDEN%TAUC=0
!!         IF (LMIX_TAU()) THEN
!!            ALLOCATE(KINEDEN%TAUL(GRIDC%MPLWV,NCDIJ))
!!            KINEDEN%TAUL=0
!!         ELSE
!!            NULLIFY(KINEDEN%TAUL)
!!         ENDIF
!!      ENDIF
!!
!!      RETURN
!!      END SUBROUTINE GENERATE_TAU_HANDLE


!!!***********************************************************************
!!!
!!! deallocate the tau_handle
!!!
!!!***********************************************************************
!!
!!      SUBROUTINE DEALLOCATE_TAU_HANDLE(KINEDEN)
!!      IMPLICIT NONE
!!      TYPE(tau_handle), POINTER :: KINEDEN
!!
!!      IF (ASSOCIATED(KINEDEN)) THEN
!!         IF (ASSOCIATED(KINEDEN%TAU)) DEALLOCATE(KINEDEN%TAU)
!!         IF (ASSOCIATED(KINEDEN%TAUL)) DEALLOCATE(KINEDEN%TAUL)
!!         IF (ASSOCIATED(KINEDEN%TAUC)) DEALLOCATE(KINEDEN%TAUC)
!!         DEALLOCATE(KINEDEN)
!!      ENDIF
!!
!!      RETURN
!!      END SUBROUTINE DEALLOCATE_TAU_HANDLE


!***********************************************************************
!
! generate the tau_handle
!
!***********************************************************************

      SUBROUTINE ALLOCATE_TAU(TAU, TAUC, TAUL, GRIDC, NCDIJ)
      USE mgrid
      IMPLICIT NONE
      COMPLEX(q), POINTER :: TAU(:,:),TAUL(:,:)
      RGRID, POINTER      :: TAUC(:)
      TYPE(grid_3d) GRIDC
      INTEGER NCDIJ

      IF (LDO_METAGGA()) THEN
         ALLOCATE(TAU(GRIDC%MPLWV,NCDIJ),TAUC(GRIDC%RL%NP))
         TAU=0; TAUC=0
         IF (LMIX_TAU()) THEN
            ALLOCATE(TAUL(GRIDC%MPLWV,NCDIJ))
            TAUL=0
         ELSE
            NULLIFY(TAUL)
         ENDIF
      ELSE
         NULLIFY(TAU,TAUC,TAUL)
      ENDIF

      RETURN
      END SUBROUTINE ALLOCATE_TAU


!***********************************************************************
!
! deallocate the tau_handle
!
!***********************************************************************

      SUBROUTINE DEALLOCATE_TAU(TAU,TAUC,TAUL)
      IMPLICIT NONE
      COMPLEX(q), POINTER :: TAU(:,:),TAUL(:,:)
      RGRID, POINTER      :: TAUC(:)

      IF (LDO_METAGGA()) THEN
         DEALLOCATE(TAU,TAUC)
         IF (LMIX_TAU()) DEALLOCATE(TAUL)
      ENDIF
      NULLIFY(TAU,TAUC,TAUL)

      RETURN
      END SUBROUTINE DEALLOCATE_TAU


!***********************************************************************
!
!***********************************************************************

      SUBROUTINE CREATE_CMBJ_AUX(GRIDC,T_INFO,LATT_CUR)
      USE poscar
      USE mgrid
      USE lattice
      IMPLICIT NONE
      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER NG, NC, N1, N2, N3, NX, NY, NZ, NI, NI_FOUND
      REAL(q) X, Y, Z, X1, X2, X3, D, DMIN

      IF (LDO_METAGGA().AND.ID_METAGGA==30) THEN
         IF (CMBJ_TYP(1)==-1) RETURN
         IF (ALLOCATED(CMBJ_AUX)) DEALLOCATE(CMBJ_AUX)
         ALLOCATE(CMBJ_AUX(GRIDC%RL%NP))
         
         ! loop over all grid points
         NG=0
         DO NC=1,GRIDC%RL%NCOL
         N2= GRIDC%RL%I2(NC)
         N3= GRIDC%RL%I3(NC)

         DO N1=1,GRIDC%RL%NROW
            NG=NG+1
            IF (GRIDC%RL%NFAST==3) THEN
               NX=N2
               NY=N3
               NZ=N1
            ELSE
               NX=N1
               NY=N2
               NZ=N3
            ENDIF

            ! now search the ion that is closest to this grid point
            DMIN=1000
            NI_FOUND=-1

            DO NI=1,T_INFO%NIONS
               X1=MOD((REAL(NX,q)-1)*(1._q/GRIDC%NGX)-T_INFO%POSION(1,NI)+10.5_q,1._q)-0.5_q
               X2=MOD((REAL(NY,q)-1)*(1._q/GRIDC%NGY)-T_INFO%POSION(2,NI)+10.5_q,1._q)-0.5_q
               X3=MOD((REAL(NZ,q)-1)*(1._q/GRIDC%NGZ)-T_INFO%POSION(3,NI)+10.5_q,1._q)-0.5_q


               X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
               Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
               Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

               D=SQRT(X*X+Y*Y+Z*Z)
            !   WRITE(*,'(7F14.7)') X1, X2, X3, X, Y, Z, D
               
               IF (D<DMIN) THEN
                  DMIN=D
                  NI_FOUND=NI
               ENDIF
            ENDDO
            IF (NI_FOUND<0) THEN
               WRITE(0,*) 'internal error in CREATE_CMBJ_AUX: NI_FOUND not correct', NI_FOUND
            ENDIF
            CMBJ_AUX(NG)=CMBJ_TYP(T_INFO%ITYP(NI_FOUND))
         !   WRITE(*,*) 'found',NG, KINEDEN%AUX(NG)
         ENDDO
         ENDDO
      ENDIF

      RETURN
      END SUBROUTINE CREATE_CMBJ_AUX


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IU6)
      USE poscar
      USE mgrid
      USE lattice 
      USE vaspxml

      IMPLICIT NONE
      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (latt)        LATT_CUR
      INTEGER IU6
      ! local variables
      REAL(q) GRHO_OVER_RHO

      IF (LDO_METAGGA().AND.ID_METAGGA==30.AND.LscMBJ()) THEN
         ! write current CMBJ to OUTCAR and vasprun.xml
         IF (IU6>=0) THEN
            WRITE(IU6,'(4X,A,F10.4/)') 'CMBJ =',CMBJ
            CALL XML_TAG_REAL("CMBJ", CMBJ)
         ENDIF
         ! communicate \nabla rho / rho
         CALLMPI( M_sum_d(GRIDC%COMM, GRHO_OVER_RHO_PW, 1) )
         CALLMPI( M_sum_d(GRIDC%COMM, GRHO_OVER_RHO_ONE_CENTER, 1) )
         ! calculate CMBJ
         GRHO_OVER_RHO=GRHO_OVER_RHO_PW/GRIDC%NPLWV+GRHO_OVER_RHO_ONE_CENTER/LATT_CUR%OMEGA
         CMBJ=CMBJA+CMBJB*SQRT(GRHO_OVER_RHO) 
         ! just to be sure
         IF (CMBJ<1._q) CMBJ=1._q
         IF (CMBJ>2._q) CMBJ=2._q
         ! initialize for the next step
         GRHO_OVER_RHO_PW=0._q; GRHO_OVER_RHO_ONE_CENTER=0._q
      ENDIF

      RETURN
      END SUBROUTINE UPDATE_CMBJ


!***********************************************************************
!
!***********************************************************************
      SUBROUTINE DESTROY_CMBJ_AUX()
      IF (ALLOCATED(CMBJ_AUX))  DEALLOCATE(CMBJ_AUX)
      RETURN
      END SUBROUTINE DESTROY_CMBJ_AUX


!***********************************************************************
!
!***********************************************************************
      FUNCTION LDO_METAGGA()
      IMPLICIT NONE
      LOGICAL LDO_METAGGA
      LDO_METAGGA=LDOMETAGGA
      END FUNCTION LDO_METAGGA


!***********************************************************************
!
!***********************************************************************
      FUNCTION LCALCPOT()
      IMPLICIT NONE
      LOGICAL LCALCPOT
      LCALCPOT=LMETA_NEEDS_POT
      END FUNCTION LCALCPOT


!***********************************************************************
!
!***********************************************************************
      FUNCTION LCALCMU()
      IMPLICIT NONE
      LOGICAL LCALCMU
      LCALCMU=LMETA_NEEDS_MU
      END FUNCTION LCALCMU


!***********************************************************************
!
!***********************************************************************
      FUNCTION LMIX_TAU()
      IMPLICIT NONE
      LOGICAL LMIX_TAU
      LMIX_TAU=LMIXTAU
      END FUNCTION LMIX_TAU


!******************** SUBROUTINE POTLOK_METAGGA ************************
!
!***********************************************************************

      SUBROUTINE POTLOK_METAGGA(&
     &   KINEDEN, &
     &   GRID,GRIDC,GRID_SOFT,COMM_INTER,WDES,INFO,P,T_INFO,E,LATT_CUR,  &
     &   CHDEN,CHTOT,DENCOR,CVTOT,SV,MUTOT,MU,SOFT_TO_C,XCSIF &
     &   )
      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE setexm
      USE base
      USE xcgrad
      USE wave

      IMPLICIT NONE
      
      TYPE (tau_handle)  KINEDEN
      TYPE (grid_3d)     GRID,GRIDC,GRID_SOFT
      TYPE (wavedes)     WDES
      TYPE (transit)     SOFT_TO_C
      TYPE (info_struct) INFO
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (energy)      E
      TYPE (latt)        LATT_CUR
      TYPE (communic)    COMM_INTER
            
      RGRID SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ),DENCOR(GRIDC%RL%NP)
      COMPLEX(q) CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ),CVTOT(GRIDC%MPLWV,WDES%NCDIJ)

      RGRID , TARGET :: MU(:,:)
      COMPLEX(q), TARGET :: MUTOT(:,:)

      REAL(q) XCSIF(3,3)      

      ! local variables
      INTEGER ISP
      INTEGER MWORK1
      REAL(q) RINPL
      REAL(q) EXC,EXCG,XCENCG
      REAL(q) TMPSIF(3,3)
      COMPLEX(q) CVZERG
      RGRID , POINTER :: MU_(:,:)
      COMPLEX(q), POINTER :: MUTOT_(:,:)
      ! work arrays (allocated after call to FEXCG)
      COMPLEX(q), ALLOCATABLE:: CWORK1(:),CWORK(:,:)
#ifdef noAugXCmeta
      COMPLEX(q), ALLOCATABLE :: CTMP(:,:)
#endif
      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
#ifdef noAugXCmeta
      ALLOCATE(CTMP(GRIDC%MPLWV,WDES%NCDIJ))
#else
#define CTMP CHTOT
#endif
      ! just a failsafe
      IF (.NOT.ASSOCIATED(KINEDEN%TAU)) THEN
         WRITE(*,*) 'POTLOK_METAGGA: internal ERROR, KIN_EDEN handle not allocated'
         STOP
      ENDIF

      MU_ => MU
      MUTOT_ => MUTOT

      MWORK1=MAX(GRIDC%MPLWV,GRID_SOFT%MPLWV)
      ALLOCATE(CWORK1(MWORK1),CWORK(GRIDC%MPLWV,WDES%NCDIJ))
!-----------------------------------------------------------------------
!
!  calculate the exchange correlation potential and the dc. correction
!
!-----------------------------------------------------------------------

      EXCG  =0
      XCENCG=0
      CVZERG=0
      TMPSIF=0

      CWORK=0

#ifdef noAugXCmeta
      ! We will use the soft charge density instead of the augmented one
      CTMP=0
      DO ISP=1,WDES%NCDIJ
         CALL ADD_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,CHDEN(1,ISP),CTMP(1,ISP))
         CALL SETUNB_COMPAT(CTMP(1,ISP),GRIDC)
      ENDDO
#endif
      ! Bring the charge density to real space
      DO ISP=1,WDES%NCDIJ
         CALL FFT3D(CTMP(1,ISP),GRIDC,1)
      ENDDO
  
      CALL FEXCGS_METAGGA( &
     &   WDES%NCDIJ,GRIDC,LATT_CUR,XCENCG,EXCG,CVZERG,TMPSIF, &
     &   KINEDEN,CTMP,DENCOR,CWORK,MUTOT_)
       
      XCSIF=XCSIF+TMPSIF
      E%EXCG=E%EXCG+EXCG
      E%XCENC=E%XCENC+XCENCG
      E%CVZERO=E%CVZERO+CVZERG

#ifndef noAugXCmeta
      ! Take the charge density back to reciprocal space
      DO ISP=1,WDES%NCDIJ
         CALL FFT_RC_SCALE(CTMP(1,ISP),CTMP(1,ISP),GRIDC)
         CALL SETUNB_COMPAT(CTMP(1,ISP),GRIDC)
      ENDDO
#endif
!-----------------------------------------------------------------------
! Add CWORK to CVTOT in real space, store the result in CVTOT
! and take it to reciprocal space
!-----------------------------------------------------------------------
#ifdef noAugXCmeta
      ! Store CVTOT in CTMP
      CTMP=CVTOT
#endif
      RINPL=1._q/GRIDC%NPLWV
      DO  ISP=1,WDES%NCDIJ 
         CALL RL_ADD(CVTOT(1,ISP),RINPL,CWORK(1,ISP),RINPL,CVTOT(1,ISP),GRIDC)
         CALL FFT3D(CVTOT(1,ISP),GRIDC,-1)
      ENDDO
!-----------------------------------------------------------------------
! copy CVTOT to SV and set contribution of unbalanced lattice-vectors
! to zero, then FFT of SV and CVTOT to real space
!-----------------------------------------------------------------------
      DO ISP=1,WDES%NCDIJ
         CALL SETUNB_COMPAT(CVTOT(1,ISP),GRIDC)
         CALL CP_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,CVTOT(1,ISP),CWORK1)
         CALL SETUNB(CWORK1,GRID_SOFT)
         CALL FFT3D(CWORK1,GRID_SOFT, 1)
         CALL RL_ADD(CWORK1,1.0_q,CWORK1,0.0_q,SV(1,ISP),GRID_SOFT)

    !  final result is only correct for first in-band-group
    ! (i.e. proc with nodeid 1 in COMM_INTER)
    !  copy to other in-band-groups using COMM_INTER
    ! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)
#ifdef realmode
         CALLMPI( M_bcast_d(COMM_INTER, SV(1,ISP), GRID%RL%NP))
#else
         CALLMPI( M_bcast_z(COMM_INTER, SV(1,ISP), GRID%RL%NP))
#endif
#ifndef noAugXCmeta
         CALL FFT3D(CVTOT(1,ISP),GRIDC,1)
#endif
      ENDDO

#ifdef noAugXCmeta
      ! Restore CVTOT from CTMP
      CVTOT=CTMP
#endif
      IF (ASSOCIATED(MUTOT_).AND.ASSOCIATED(MU_)) THEN
!-----------------------------------------------------------------------
! The same procedure to calculate MU from MUTOT
!-----------------------------------------------------------------------
      RINPL=1._q/GRIDC%NPLWV
      DO  ISP=1,WDES%NCDIJ 
         CALL RL_ADD(MUTOT_(1,ISP),RINPL,MUTOT_(1,ISP),0._q,MUTOT_(1,ISP),GRIDC)
         CALL FFT3D(MUTOT_(1,ISP),GRIDC,-1)
      ENDDO
!-----------------------------------------------------------------------
! copy MUTOT_ to MU_ and set contribution of unbalanced lattice-vectors
! to zero, then FFT of MU_ and MUTOT_ to real space
!-----------------------------------------------------------------------
      DO ISP=1,WDES%NCDIJ
         CALL SETUNB_COMPAT(MUTOT_(1,ISP),GRIDC)
         CALL CP_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,MUTOT_(1,ISP),CWORK1)
         CALL SETUNB(CWORK1,GRID_SOFT)
         CALL FFT3D(CWORK1,GRID_SOFT, 1)
         CALL RL_ADD(CWORK1,1.0_q,CWORK1,0.0_q,MU_(1,ISP),GRID_SOFT)

    !  final result is only correct for first in-band-group
    ! (i.e. proc with nodeid 1 in COMM_INTER)
    !  copy to other in-band-groups using COMM_INTER
    ! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)
#ifdef realmode
         CALLMPI( M_bcast_d(COMM_INTER, MU_(1,ISP), GRID%RL%NP))
#else
         CALLMPI( M_bcast_z(COMM_INTER, MU_(1,ISP), GRID%RL%NP))
#endif
         CALL FFT3D(MUTOT_(1,ISP),GRIDC,1)
      ENDDO
      ENDIF

      DEALLOCATE(CWORK1,CWORK)
      NULLIFY(MU_,MUTOT_)
#ifdef noAugXCmeta
      DEALLOCATE(CTMP)
#endif
      RETURN
      END SUBROUTINE POTLOK_METAGGA


!******************** SUBROUTINE POTXC_METAGGA *************************
!
!***********************************************************************

      SUBROUTINE POTXC_METAGGA(&
     &   KINEDEN,GRID,GRIDC,GRID_SOFT,SOFT_TO_C,WDES,LATT_CUR,  &
     &   CHDEN,CHTOT,DENCOR,CVTOT,MUTOT &
     &   )
      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE setexm
      USE base
      USE xcgrad
      USE wave

      IMPLICIT NONE
      
      TYPE (tau_handle)  KINEDEN
      TYPE (grid_3d)     GRID,GRIDC,GRID_SOFT
      TYPE (wavedes)     WDES
      TYPE (transit)     SOFT_TO_C
      TYPE (latt)        LATT_CUR
            
      RGRID DENCOR(GRIDC%RL%NP)
      COMPLEX(q) CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ),CVTOT(GRIDC%MPLWV,WDES%NCDIJ)

      COMPLEX(q), TARGET :: MUTOT(:,:)

      ! local variables
      INTEGER ISP
      REAL(q) RINPL
      REAL(q) EXC,EXCG,XCENCG
      REAL(q) TMPSIF(3,3)
      COMPLEX(q) CVZERG
      COMPLEX(q), POINTER :: MUTOT_(:,:)
#ifdef noAugXCmeta
      COMPLEX(q), ALLOCATABLE :: CTMP(:,:)
#endif
      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
#ifdef noAugXCmeta
      ALLOCATE(CTMP(GRIDC%MPLWV,WDES%NCDIJ))
#else
#define CTMP CHTOT
#endif
      ! just a failsafe
      IF (.NOT.ASSOCIATED(KINEDEN%TAU)) THEN
         WRITE(*,*) 'POTLOK_METAGGA: internal ERROR, KIN_EDEN handle not allocated'
         STOP
      ENDIF

      MUTOT_ => MUTOT

      CVTOT=0; MUTOT_=0
      XCENCG=0; EXCG=0; CVZERG=0; TMPSIF=0

#ifdef noAugXCmeta
      ! We will use the soft charge density instead of the augmented one
      CTMP=0
      DO ISP=1,WDES%NCDIJ
         CALL ADD_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,CHDEN(1,ISP),CTMP(1,ISP))
         CALL SETUNB_COMPAT(CTMP(1,ISP),GRIDC)
      ENDDO
#endif
      ! Bring the charge density to real space
      DO ISP=1,WDES%NCDIJ
         CALL FFT3D(CTMP(1,ISP),GRIDC,1)
      ENDDO
  
      CALL FEXCGS_METAGGA( &
     &   WDES%NCDIJ,GRIDC,LATT_CUR,XCENCG,EXCG,CVZERG,TMPSIF, &
     &   KINEDEN,CTMP,DENCOR,CVTOT,MUTOT_)
       
#ifndef noAugXCmeta
      ! Take the charge density back to reciprocal space
      DO ISP=1,WDES%NCDIJ
         CALL FFT_RC_SCALE(CTMP(1,ISP),CTMP(1,ISP),GRIDC)
         CALL SETUNB_COMPAT(CTMP(1,ISP),GRIDC)
      ENDDO
#endif
!-----------------------------------------------------------------------
! Add CWORK to CVTOT in real space, store the result in CVTOT
! and take it to reciprocal space
!-----------------------------------------------------------------------
      RINPL=1._q/GRIDC%NPLWV
      DO  ISP=1,WDES%NCDIJ 
         CALL RL_ADD(CVTOT(1,ISP),RINPL,CVTOT(1,ISP),0._q,CVTOT(1,ISP),GRIDC)
         CALL FFT3D(CVTOT(1,ISP),GRIDC,-1)
         CALL SETUNB_COMPAT(CVTOT(1,ISP),GRIDC)         
      ENDDO
     ! Flip CVTOT from (up,down) to (total,magnetization) storage mode
     ! This is done because the routine FORTAU needs the "total" component
     CALL RC_FLIP_POTENTIAL(CVTOT,GRIDC,WDES%NCDIJ,.FALSE.)

      IF (ASSOCIATED(MUTOT_)) THEN
!-----------------------------------------------------------------------
! The same procedure with MUTOT
!-----------------------------------------------------------------------
      RINPL=1._q/GRIDC%NPLWV
      DO  ISP=1,WDES%NCDIJ 
         CALL RL_ADD(MUTOT_(1,ISP),RINPL,MUTOT_(1,ISP),0._q,MUTOT_(1,ISP),GRIDC)
         CALL FFT3D(MUTOT_(1,ISP),GRIDC,-1)
         CALL SETUNB_COMPAT(MUTOT_(1,ISP),GRIDC)         
      ENDDO
      ! Flip MUTOT from (up,down) to (total,magnetization) storage mode
      ! This is done because the routine FORTAU needs the "total" component 
      CALL RC_FLIP_POTENTIAL(MUTOT_,GRIDC,WDES%NCDIJ,.FALSE.)
      ENDIF

      NULLIFY(MUTOT_)
#ifdef noAugXCmeta
      DEALLOCATE(CTMP)
#endif
      RETURN
      END SUBROUTINE POTXC_METAGGA


!******************** SUBROUTINE FEXCGS_METAGGA ************************
!
!***********************************************************************

      SUBROUTINE FEXCGS_METAGGA(&
     &   NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &   KINEDEN,CHTOT,DENCOR,CWORK,MUTOT &
     &)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE setexm

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR
      TYPE (tau_handle)  KINEDEN

      RGRID DENCOR(GRIDC%RL%NP)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ),CWORK(GRIDC%MPLWV,NCDIJ)
      DIMENSION XCSIF(3,3)
      COMPLEX(q), POINTER :: MUTOT(:,:)
      ! work arrays
      COMPLEX(q),ALLOCATABLE :: CWGRAD(:,:)
      REAL(q), ALLOCATABLE   :: DWORKG(:,:),DWORK1(:,:),DWORK2(:,:),DWORK3(:,:),DVC(:)

      REAL(q) LAPLUP,LAPLDW

      ! for rVV10
      REAL(q),ALLOCATABLE :: DWORK4(:), DWORK6(:), DWORK7(:)
      RGRID , ALLOCATABLE :: DWORK5(:)

      NP1=GRIDC%RL%NP
      
      IF (NCDIJ==1.OR.NCDIJ==2) THEN
      ! (Non)spinpolarized
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))

         ! Flip CHTOT from (rho,mag) to (rho_up,rho_down)
         CALL RL_FLIP(CHTOT,GRIDC,NCDIJ,.TRUE.)

         IF (.NOT.LUSE_VDW) THEN
            CALL FEXCGS_METAGGA_(LCALCPOT(),LCALCMU(), &
        &            NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK,KINEDEN%TAU,CWGRAD,CHTOT,CWORK,KINEDEN%TAU, &
        &            DENCOR,KINEDEN%TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC,MUTOT)
         ELSE
            ALLOCATE(DWORK4(NP1),DWORK5(NP1),DWORK6(NP1),DWORK7(NP1) )
            CALL FEXCGS_METAGGA_VDW_(LCALCPOT(),LCALCMU(), &
        &            NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK,KINEDEN%TAU,CWGRAD,CHTOT,CWORK,KINEDEN%TAU, &
        &            DENCOR,KINEDEN%TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC,MUTOT, &
        &            DWORK4,DWORK5,DWORK6,DWORK7)
            DEALLOCATE(DWORK4,DWORK5,DWORK6,DWORK7)
         ENDIF
         ! Flip CHTOT back to (rho,mag)
         CALL RL_FLIP(CHTOT,GRIDC,NCDIJ,.FALSE.)
         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC)
      ELSEIF (NCDIJ==4) THEN
      ! Non-collinear magnetism: gradient corrections in the
      ! noncollinear case are calculated a bit differently than in the collinear case
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ/2), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))

         ! CHTOT and KINEDEN should be in (rho,mag) representation
         CALL FEXCGS_METAGGA_NONCOL(LCALCPOT(),LCALCMU(), &
        &            NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK,KINEDEN%TAU,CWGRAD,CHTOT,CWORK,KINEDEN%TAU, &
        &            DENCOR,KINEDEN%TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC,MUTOT)

         IF (ASSOCIATED(MUTOT).AND.LCALCMU()) THEN
! mM: just until all is done
!           WRITE(*,*) 'non-collinear mode for revTPSS not yet fully implemented, sorry ...'
!           STOP
! mM
            ! MUTOT and CWORK come out as two component (up,down) quantities, where
            ! "up" and "down" are taken w.r.t. the local magnetization direction,
            ! this is now cast into (rho, mag) form.
            CALL MAG_DIRECTION_KINDENS(CHTOT(1,1), KINEDEN%TAU(1,1), MUTOT(1,1), CWORK(1,1), GRIDC, NCDIJ, LATT_CUR)
            ! and rearranged from (rho,mag) to spinor representation 
            CALL POT_FLIP_RL(MUTOT(1,1), GRIDC, NCDIJ)    
         ELSE
            ! CWORK(potential) comes out with (spin up and down) representation
            ! It needs to flip to (total, mag) representation
            CALL MAG_DIRECTION(CHTOT(1,1), CWORK(1,1), GRIDC, NCDIJ)
         ENDIF
         ! rearrange the potential in (spinor) representation 
         CALL POT_FLIP_RL(CWORK(1,1), GRIDC, NCDIJ)
         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC)         
      ENDIF

      RETURN
      END SUBROUTINE FEXCGS_METAGGA


!******************** SUBROUTINE SET_PAW_METAGGA ***********************
!
!***********************************************************************
      SUBROUTINE SET_PAW_METAGGA(P)
      USE prec
      USE pseudo
      USE radial
      USE constant
      IMPLICIT NONE
      TYPE (potcar), POINTER :: P
      ! local variables
      INTEGER I,NMIN,NMAX
      REAL(q) SCALE
      REAL(q) RHO(P%R%NMAX),DRHO(P%R%NMAX)

      SCALE=1/(2*SQRT(PI)) ! Y00

      NMAX=P%R%NMAX

!     NMIN=NMAX
!     DO I=1,P%LMAX
!        NMIN=MIN(NMIN,IR_MATCH(P,I))
!     ENDDO
      NMIN=IR_PSMAX(P)

      RHO(:)=SCALE*P%RHOAE(:)/(P%R%R(:)*P%R%R(:))

      CALL GRAD(P%R,RHO,DRHO)
      DRHO=ABS(DRHO)
      DO I=1,NMAX
         IF ((DRHO(I)*DRHO(I)/RHO(I)/8._q*2._q*HSQDTM) > (P%TAUAE(I)*SCALE/(P%R%R(I)*P%R%R(I))) ) &
        &   P%TAUAE(I)=(DRHO(I)*DRHO(I)/RHO(I)/8._q*2._q*HSQDTM)/SCALE*(P%R%R(I)*P%R%R(I)) !*1.00001
      ENDDO

!     P%TAUAE(NMIN:NMAX)=0
      P%RHOAE(NMIN:NMAX)=P%RHOPS(NMIN:NMAX)

      RETURN
      END SUBROUTINE SET_PAW_METAGGA


!******************** SUBROUTINE SET_KINEDEN ***************************
!
! SET_KINEDEN calculates the kinetic energy density of the PW part of
! the wavefunctions (0.5*|grad psi|**2).
!
! The actual work is done in TAU_PW_DIRECT
!
!***********************************************************************

      SUBROUTINE SET_KINEDEN(&
     &   GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM,NIOND,W,WDES,KINEDEN &
     &)
      USE prec
      USE lattice
      USE mgrid
      USE msymmetry
      USE base
      USE wave
      USE mpimy
      
      IMPLICIT NONE

      TYPE (grid_3d)     GRIDC,GRID,GRID_SOFT
      TYPE (latt)        LATT_CUR
      TYPE (transit)     SOFT_TO_C
      TYPE (symmetry)    SYMM
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (tau_handle)  KINEDEN

      INTEGER NIOND
      
      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
!     IF ((.NOT.LDO_METAGGA()).OR.(.NOT.LCALCMU())) RETURN

      ! just a failsafe
      IF (.NOT.ASSOCIATED(KINEDEN%TAU)) THEN
         WRITE(*,*) 'SET_KINEDEN: internal ERROR, KIN_EDEN handle not allocated'
         STOP
      ENDIF
      
      CALL TAU_PW(&
     &   GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM,NIOND,W,WDES,KINEDEN%TAU)
   
      RETURN
      END SUBROUTINE SET_KINEDEN


!************************* SUBROUTINE TAUPAR ***************************
!
!  subroutine to calculate the partial kinetic energy density of the 
!  core electrons from overlapping atoms in real space and store
!  it in an real array
!  NOTE: the DENS array is not necessarily suffiently large
!  to allow for 3d-FFT, hence a work array is temporarily allocated
!
!***********************************************************************

      SUBROUTINE TAUPAR(GRIDC,T_INFO,B,OMEGA,P,CSTRF,DENS)
      USE prec
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d)     GRIDC
      RGRID        DENS(GRIDC%RL%NP)
      COMPLEX(q)   CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      REAL(q)      B(3,3),OMEGA
      ! work arrays
      COMPLEX(q),ALLOCATABLE::    CWORK1(:),CWORK2(:)

      ALLOCATE(CWORK1(GRIDC%MPLWV),CWORK2(GRIDC%MPLWV))
      CALL TAUATO(.TRUE.,GRIDC,T_INFO,B,P,CSTRF,CWORK1,CWORK2)
      CALL RL_ADD(CWORK1,1.0_q/OMEGA,CWORK1,0.0_q,DENS,GRIDC)

      DEALLOCATE(CWORK1,CWORK2)

      RETURN
      END SUBROUTINE TAUPAR


!************************* SUBROUTINE TAUATO ***************************
!
!  This routine calculates the partial kinetic energy density of the
!  core electrons corresponding to overlapping atoms
!
!  LFOUR
!   .FALSE. set up density in reciprocal space
!   .TRUE.  set up density in real space
!
!***********************************************************************

      SUBROUTINE TAUATO(LFOUR,GRIDC,T_INFO,B,P,CSTRF,CHTOT,CHDER)
      USE prec
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d)     GRIDC
      COMPLEX(q)   CHTOT(GRIDC%RC%NP),CHDER(GRIDC%RC%NP)
      COMPLEX(q)   CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      REAL(q)      B(3,3)
      LOGICAL LFOUR
      ! local variables
      REAL(q), POINTER :: PRHO(:)

      CHTOT=0
      CHDER=0
!=======================================================================
! loop over all types of atoms
! if no pseudocharge for this ion next ion
!=======================================================================
      typ: DO NT=1,T_INFO%NTYP
       PRHO=>P(NT)%PSPTAU
       IF (.NOT.ASSOCIATED(PRHO)) GOTO 200
!=======================================================================
! calculate the scaling factor ARGSC that converts the magnitude of a
! reciprocal lattice vector to the correponding position in the
! pseudopotential arrays
!=======================================================================
      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS*3

      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*B(1,1)+GRIDC%LPCTY(N2)*B(1,2)+GRIDC%LPCTZ(N3)*B(1,3)
        GY= GRIDC%LPCTX(N1)*B(2,1)+GRIDC%LPCTY(N2)*B(2,2)+GRIDC%LPCTZ(N3)*B(2,3)
        GZ= GRIDC%LPCTX(N1)*B(3,1)+GRIDC%LPCTY(N2)*B(3,2)+GRIDC%LPCTZ(N3)*B(3,3)

        G=SQRT(GX**2+GY**2+GZ**2)*TPI

        IF (G/=0 .AND.G<PSGMA2) THEN
!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
          ARG=(G*ARGSC)+1
          NADDR=MAX(INT(ARG),2)
          REM=ARG-NADDR
          V1=PRHO(NADDR-1)
          V2=PRHO(NADDR)
          V3=PRHO(NADDR+1)
          V4=PRHO(NADDR+2)
          T0=V2
          T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
          T2=(V1+V3-(2*V2))/2._q
          T3=(V4-V1+(3*(V2-V3)))/6._q
          CHTOT(N)=CHTOT(N)+(T0+REM*(T1+REM*(T2+REM*T3))) *CSTRF(N,NT)
          CHDER(N)=CHDER(N)+(T1+REM*(2*T2+REM*3*T3))*ARGSC*CSTRF(N,NT)
        ELSE IF (G==0) THEN
          CHTOT(N)=CHTOT(N)+PRHO(1)*CSTRF(N,NT)
          CHDER(N)=0
        ENDIF
      ENDDO
  200 CONTINUE
      ENDDO typ
!=======================================================================
! set the charge-density of unbalanced lattice-vectors to 0
! and transform the charge-density to real space
!=======================================================================
      CALL SETUNB(CHTOT(1),GRIDC)
      CALL SETUNB(CHDER(1),GRIDC)

      IF (LFOUR) THEN
        CALL FFT3D(CHTOT,GRIDC,1)
      ENDIF

      RETURN
      END SUBROUTINE TAUATO

!
!  small subroutine which allocates CHDER calls the subroutine
!  above, and then returns
!

      SUBROUTINE TAUATO_WORK(LFOUR,GRIDC,T_INFO,B,P,CSTRF,CHTOT)
      USE prec
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d)     GRIDC
      COMPLEX(q)   CHTOT(GRIDC%RC%NP)
      COMPLEX(q), ALLOCATABLE :: CHDER(:)
      COMPLEX(q)   CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      REAL(q)      B(3,3)
      LOGICAL LFOUR
      ! local variables
      REAL(q), POINTER :: PRHO(:)

      ALLOCATE(CHDER(GRIDC%RC%NP))
      CALL TAUATO(LFOUR,GRIDC,T_INFO,B,P,CSTRF,CHTOT,CHDER)
      DEALLOCATE(CHDER)

      END SUBROUTINE TAUATO_WORK
      

!************************ SUBROUTINE FORTAU ****************************
!
! This subroutine calculates the correction to the Hellmann-Feynman
! forces due to the (possible) presence of a partial core kinetic energy
! density
!
!***********************************************************************

      SUBROUTINE FORTAU(GRIDC,P,T_INFO,LATT_CUR,MUTOT,TAUFOR)
      USE prec

      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      COMPLEX(q)   MUTOT(GRIDC%MPLWV)
      REAL(q)      TAUFOR(3,T_INFO%NIONS)
      ! work arrays
      REAL(q), POINTER :: PRHO(:)
      REAL(q), ALLOCATABLE :: WORK(:)

      ALLOCATE(WORK(GRIDC%RC%NP))
      
      SCALE=1._q/(RYTOEV*AUTOA2)

      TAUFOR(1:3,1:T_INFO%NIONS)=0
!=======================================================================
! loop over all types of atoms
!=======================================================================
      NIS=1
      typ: DO NT=1,T_INFO%NTYP
      NIADD=T_INFO%NITYP(NT)

      PRHO=>P(NT)%PSPTAU

       IF (.NOT.ASSOCIATED(PRHO)) GOTO 200
!=======================================================================
! iterpolate the pseudopotential on the grid of reciprocal
! lattice-vectors
!=======================================================================
      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS*3

      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        SETFACT1
        SETFACT
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G=SQRT(GX**2+GY**2+GZ**2)*TPI

        IF (G/=0 .AND.G<PSGMA2) THEN
!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
          ARG=(G*ARGSC)+1
          NADDR=MAX(INT(ARG),2)
          REM=ARG-NADDR
          V1=PRHO(NADDR-1)
          V2=PRHO(NADDR)
          V3=PRHO(NADDR+1)
          V4=PRHO(NADDR+2)
          T0=V2
          T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
          T2=(V1+V3-(2*V2))/2._q
          T3=(V4-V1+(3*(V2-V3)))/.6_q
          WORK(N)=T0+REM*(T1+REM*(T2+REM*T3))
!       ELSE IF (G==0) THEN
!         WORK(N)=PRHO(1)
        ELSE
          WORK(N)=0
        ENDIF

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
           FOR=WORK(NG)*MULFACT AIMAG(CONJG(MUTOT(NG))*CEXPF)

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
         TAUFOR(1,NI)=FOR1*TPI*SCALE
         TAUFOR(2,NI)=FOR2*TPI*SCALE
         TAUFOR(3,NI)=FOR3*TPI*SCALE

      ENDDO ion
  200 NIS=NIS+NIADD
      ENDDO typ
!=======================================================================
! forces are now in the reciprocal lattice transform it to
! cartesian coordinates
!=======================================================================
      CALLMPI( M_sum_d(GRIDC%COMM,TAUFOR(1,1),T_INFO%NIONS*3))
      CALL DIRKAR(T_INFO%NIONS,TAUFOR,LATT_CUR%B)
      DEALLOCATE(WORK)

      RETURN
      END SUBROUTINE FORTAU


!******************** SUBROUTINE RAD_KINEDEN ***************************
!
!***********************************************************************
      SUBROUTINE RAD_KINEDEN(&
     &   P,W,LMAX_TAU,NCDIJ,COCC,TAU)
      USE prec
      USE asa
      USE radial
      USE constant
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      REAL(q) W(:,:)
      INTEGER LMAX_TAU,NCDIJ
      OVERLAP COCC(:,:,:)
      REAL(q) TAU(:,:,:)
      ! local variables
      INTEGER ISP,I,Ch1,Ch2
      INTEGER LLMAX_TAU,LMMAX_TAU,LM,LMP,LL,LLP,M,MP,MPLOW,INDYLM,INDYLMP

      REAL(q) FAKT
      REAL(q) W1(P%R%NMAX),W2(P%R%NMAX),DW1(P%R%NMAX),DW2(P%R%NMAX)

      TAU=0

      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
                  
      ! check consistency
      LMMAX_TAU=(LMAX_TAU+1)**2
      IF (SIZE(TAU,2)<LMMAX_TAU) THEN
         WRITE(*,*) 'RAD_KINEDEN: ERROR, size(TAU,2)<LMMAX_TAU',SIZE(TAU,2),LMMAX_TAU
         STOP
      ENDIF
      
      ! restriction for the angular development of \tau and \mu
      LLMAX_TAU=MIN(LMAXTAU,LMAX_TAU); LMMAX_TAU=(LLMAX_TAU+1)**2

      spin: DO ISP=1,NCDIJ
         
         ! loops over channels (l,epsilon)
         LM=1; DO Ch1=1,P%LMAX
         ! d(w1/r)/dr
         W1(:)=W(:,Ch1)/P%R%R(:); CALL GRAD(P%R,W1,DW1)
         
         LMP=LM; DO Ch2=Ch1,P%LMAX
         ! d(w2/r)/dr
         W2(:)=W(:,Ch2)/P%R%R(:); CALL GRAD(P%R,W2,DW2)
         
            LL=P%LPS(Ch1); LLP=P%LPS(Ch2)
            DO M=1,2*LL+1
            MPLOW=1; IF(Ch1==Ch2) MPLOW=M
            DO MP=MPLOW,2*LLP+1
               INDYLM= LL**2 +M
               INDYLMP=LLP**2+MP

               FAKT=REAL(COCC(LM+M-1,LMP+MP-1,ISP)+COCC(LMP+MP-1,LM+M-1,ISP),q)
               IF ((LM+M-1)==(LMP+MP-1)) THEN
                  FAKT=REAL(COCC(LM+M-1,LMP+MP-1,ISP),q)
               ENDIF
               
               DO I=1,LMMAX_TAU
                  TAU(:,I,ISP)=TAU(:,I,ISP)+FAKT*&
                 &   (DW1(:)*X_YLM_X_YLM(INDYLM,INDYLMP,I)*DW2(:)*P%R%R(:)*P%R%R(:) + &
                 &    DW1(:)*X_YLM_YLMD(INDYLM,INDYLMP,I)*W2(:)*P%R%R(:) + &
                 &    W1(:)*X_YLM_YLMD(INDYLMP,INDYLM,I)*DW2(:)*P%R%R(:) + &
                 &    W1(:)*YLMD_YLMD(INDYLM,INDYLMP,I)*W2(:))
               ENDDO
                              
            ENDDO
            ENDDO
         LMP=LMP+2*LLP+1
         ENDDO
         LM =LM +2*LL +1
         ENDDO

      ENDDO spin
      
      TAU=TAU*HSQDTM

      RETURN
      END SUBROUTINE RAD_KINEDEN


!********************* SUBROUTINE RAD_TAU_ATOM *************************
!
!***********************************************************************
      SUBROUTINE RAD_TAU_ATOM( &
     &   P,W,LMAX_TAU,TAU)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      REAL(q) W(:,:)  
      REAL(q) TAU(:,:,:)
      INTEGER LMAX_TAU
      ! local variables
      OVERLAP, ALLOCATABLE :: COCC(:,:,:)
      INTEGER Ch,LM,M,LMMAX_TAU

      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
                  
      ! check consistency
      LMMAX_TAU=(LMAX_TAU+1)**2
      IF (SIZE(TAU,2)<LMMAX_TAU) THEN
         WRITE(*,*) 'RAD_TAU_ATOM: ERROR, size(TAU,2)<LMMAX_TAU',SIZE(TAU,2),LMMAX_TAU
         STOP
      ENDIF
      
      ALLOCATE(COCC(P%LMDIM,P%LMDIM,1))
      COCC=0
      
      LM=1
      DO Ch=1,P%LMAX
         DO M=1,2*P%LPS(Ch)+1
            COCC(LM,LM,1)=P%QATO(Ch,Ch)
            LM=LM+1
         ENDDO
      ENDDO
      
      CALL RAD_KINEDEN(P,W,LMAX_TAU,1,COCC,TAU)
      
      DEALLOCATE(COCC)
      
      RETURN
      END SUBROUTINE RAD_TAU_ATOM


!********************* SUBROUTINE RAD_POT_METAGGA **********************
!
!***********************************************************************
      SUBROUTINE RAD_POT_METAGGA(&
     &   R,ISPIN,LMAX_CALC,LMAX_TAU,LASPH,RHO,RHOPS,RHOC,POTC,TAU,POTH,POT,DOUBLEC,EXCG,MU,TAUC)
      USE prec
      USE constant
      USE radial
      IMPLICIT NONE
      TYPE (rgrid) R            
      INTEGER ISPIN
      INTEGER LMAX_CALC,LMAX_TAU
      REAL(q) RHO(:,:,:)     ! charge distribution see above (charge,magnetization)
      REAL(q) RHOPS(:,:,:)   ! same as above (but possibly without augmentation)
      REAL(q) RHOC(:)        ! core charge distribution
      REAL(q) POTC(:)        ! frozen core potential (Hartree)
      REAL(q) TAU(:,:,:)     ! kinetic energy density (up,down)
      REAL(q) POTH(:,:,:)    ! Hartree potential (up,down)
      REAL(q) POT(:,:,:)     ! Total potential (up,down)
      REAL(q) EXCG           ! exchange energy only
      REAL(q) DOUBLEC        ! double counting corrections      
      REAL(q) MU(:,:,:)      ! d e_xc/d tau (up,down)
      REAL(q), OPTIONAL :: TAUC(:) ! kinetic energy density of the core electrons
      LOGICAL LASPH
      ! local variables
      INTEGER N,L,M,LM,K
      REAL(q) SCALE
      REAL(q) SUM,DHARTREE
      REAL(q) DEXC,DVXC

      MU=0

      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN

      POT=0; POTH=0

      SCALE=2*SQRT(PI)
      N=R%NMAX

      ! Hartree contributions 
      DHARTREE=0
      DO L=0,LMAX_CALC
      DO M=0,2*L
         LM=L*L+M+1
         CALL RAD_POT_HAR(L,R,POTH(:,LM,1),RHO(:,LM,1),SUM)
         IF (ISPIN==2) POTH(:,LM,2)=POTH(:,LM,1)
         DHARTREE=DHARTREE+SUM
      ENDDO
      ENDDO
      DO K=1,N
         POTH(K,1,1)=POTH(K,1,1)+POTC(K)*SCALE
      ENDDO
      IF (ISPIN==2) POTH(:,1,2)=POTH(:,1,1)

      ! Exchange-correlation
      DEXC=0
      DVXC=0

      POT=POTH
      
      IF (PRESENT(TAUC)) THEN
         CALL RAD_METAGGA_XC(R,ISPIN,LMAX_CALC,LMAX_TAU,LASPH,RHOPS,RHOC,POTC,TAU,DEXC,DVXC,POT,MU,TAUC)
      ELSE
         CALL RAD_METAGGA_XC(R,ISPIN,LMAX_CALC,LMAX_TAU,LASPH,RHOPS,RHOC,POTC,TAU,DEXC,DVXC,POT,MU)
      ENDIF

      EXCG=DEXC
      DOUBLEC=-DHARTREE/2+DEXC-DVXC
#ifdef debug
      WRITE(*,1)  DHARTREE/2+DEXC, DOUBLEC 
1     FORMAT('  Hartree + exc, -Hartree + exc -vxc',6F14.7)
#endif
      
      RETURN
      END SUBROUTINE RAD_POT_METAGGA     


!********************* SUBROUTINE RAD_METAGGA_XC ***********************
!
!***********************************************************************
      SUBROUTINE RAD_METAGGA_XC(&
     &   R,ISPIN,LMAX_CALC,LMAX_TAU,LASPH,RHO,RHOC,POTC,TAU,DEXC,DVXC,POT,MU,TAUC)
      USE prec
      USE asa
      USE radial
      USE constant
      USE metalib

      IMPLICIT NONE
      TYPE (rgrid) :: R
      INTEGER ISPIN
      INTEGER LMAX_CALC,LMAX_TAU
      REAL(q) RHOC(:)        ! core charge distribution
      REAL(q) POTC(:)        ! frozen core potential (Hartree)
      REAL(q) RHO(:,:,:)     ! charge distribution see above (charge,magnetization)
      REAL(q) TAU(:,:,:)     ! kinetic energy density (up,down)
      REAL(q) DEXC, DVXC
      REAL(q) POT(:,:,:)     ! potential (up,down)
      REAL(q) MU(:,:,:)      ! d e_xc/d tau (up,down)
      REAL(q), OPTIONAL :: TAUC(:) ! kinetic energy density of the core electrons
      LOGICAL LASPH
      ! local variables
      REAL(q) RHOT(R%NMAX,2),RHO_ANG(R%NMAX,2,3),T1(R%NMAX,2),T2(R%NMAX,2)
      REAL(q) TAUU(R%NMAX),TAUD(R%NMAX)
      INTEGER ISP,K,I,J,IFAIL
      INTEGER LM,LMP,LLMAX,LMMAX,LMMAX_alloc,LMMAX_CALC,LLMAX_TAU,LMMAX_TAU
      INTEGER NP,PHPTS,THPTS,NPTS
      REAL(q) SCALE,SUM,DELTAPHI,EVTOH

      REAL(q) DVINI,DCMU

      REAL(q) SIM_FAKT
      REAL(q) TEMP1
      REAL(q), ALLOCATABLE :: RADPTS(:,:), XYZPTS(:,:)
      REAL(q), ALLOCATABLE :: YLM(:,:),YLMD(:,:,:),YLMDD(:,:,:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: dFXC(:,:,:,:),dFXCdT(:,:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)
      REAL(q), ALLOCATABLE :: RHOLM(:,:,:), RHOLMD(:,:,:)

      REAL(q) XU,YU,ZU,XD,YD,ZD
      REAL(q) TG,LAPLUP,LAPLDW,MUUP,MUDW
      REAL(q) EXT,DEXC1,DEXC2,DVXC1,DVXC2,DVC
      REAL(q) TMP(ISPIN,3)
      EXTERNAL GAUSSI2

      REAL(q) :: CHGMIN=1E-10_q, NABMIN=1E-10_q, TAUMIN=1E-10_q, LAPMIN=1E-10_q

      ! initial double counting corrections
      DVINI=0
      IF (LCALCPOT()) THEN
         CALL RAD_POT_IN_RHO(R,ISPIN,LMAX_CALC,RHO,POT,DVINI)
      ENDIF
      
      SCALE=2*SQRT(PI) ! 1/Y00
      
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5
      
      LMMAX_CALC=(LMAX_CALC+1)**2
      ! restriction for the angular development of \rho
      LLMAX=MIN(6,LMAX_CALC)
      ! in GGA this needs to be increased by 1 (nabla-comment)
      LLMAX=LLMAX+1; LMMAX=(LLMAX+1)**2
      
      ! restriction for the angular development of \tau and \mu
      LLMAX_TAU=MIN(LMAXTAU,LMAX_TAU); LMMAX_TAU=(LLMAX_TAU+1)**2

      ! restrict everything to spherical contributions only (LASPH=.FALSE.)
      IF (.NOT.LASPH) THEN
         LLMAX=1; LMMAX=4; LLMAX_TAU=0; LMMAX_TAU=1; LMMAX_CALC=1
      ENDIF

      LMMAX_alloc=(MAX(LLMAX,LLMAX_TAU)+1)**2

!========================================================================
! number of theta and phi pivot points to perform angular integration
! since Exc=f(a*Yllmax,m) we need more pivot points than theoretically 
! needed to integrate Yllmax,m. 
! the factor 2 is the minium, 3 is more accurate
!========================================================================
      PHPTS=PHPTS_FACT*(MAX(LLMAX,LLMAX_TAU)+1)
      THPTS=THPTS_FACT*FLOOR(REAL(MAX(LLMAX,LLMAX_TAU)/2+1,KIND=q))
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(XYZPTS(NPTS,3),RADPTS(NPTS,2),WEIGHT(THPTS),ABSCIS(THPTS), & 
     &     YLM(NPTS,LMMAX_alloc),YLMD(NPTS,LMMAX_alloc,3),YLMDD(NPTS,LMMAX_alloc,6), &
     &     YLM_NABLA_YLM(LMMAX,LMMAX,0:3),YLM_X_YLM(LMMAX,LMMAX,0:3), &
     &     RHOLM(R%NMAX,LMMAX,2),RHOLMD(R%NMAX,LMMAX,2))
      
      IF (LCALCPOT()) THEN
         ALLOCATE(dFXC(R%NMAX,LMMAX,0:3,2)); dFXC=0
         CALL SETYLM_NABLA_YLM(LLMAX,YLM_NABLA_YLM,YLM_X_YLM)
      ENDIF
      IF (LCALCMU()) THEN
         ALLOCATE(dFXCdT(R%NMAX,LMMAX_TAU,2)); dFXCdT=0
      ENDIF

      ! set phi positions, equally spaced
      RADPTS=0; WEIGHT=0; ABSCIS=0
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
      
      ! get |r| Y_lm on a unit sphere and its derivatives
      YLM=0; YLMD=0; YLMDD=0
!     CALL SETYLM_GRAD2(LLMAX,NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))
      CALL SETYLM_GRAD3(MAX(LLMAX,LLMAX_TAU),NPTS,YLM,YLMD,YLMDD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))
!========================================================================
! main loop over all grid points
! prepare the charge density array RHOT
!========================================================================
      DEXC=0

      ! loop over all points in the angular grid
      points: DO NP=1,NPTS
         ! weight of this points
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))

         RHOT=0
         RHO_ANG=0
         TAUU=0; TAUD=0

         ! calculate the total charge RHOT = RHO/r^2 (up and down)
         ! and the gradient of the charge RHO_ANG
         DO K=1,R%NMAX
            DO LM=1,LMMAX_CALC
               ! total charge
               RHOT(K,1)=RHOT(K,1)+YLM(NP,LM)*RHO(K,LM,1) ! rho(r) Y_L(r)
               RHO_ANG(K,1,:)=RHO_ANG(K,1,:)+YLMD(NP,LM,:)*RHO(K,LM,1) ! rho(r) nabla Y_L(r)
               IF (ISPIN==1) THEN
               ! LM resolved charge density (up,down)
                  RHOLM(K,LM,1)=0.5_q*RHO(K,LM,1)/(R%R(K)*R%R(K))
                  RHOLM(K,LM,2)=0.5_q*RHO(K,LM,1)/(R%R(K)*R%R(K))
               ! magnetization
                  RHOT(K,2)=0
                  RHO_ANG(K,2,:)=0
               ELSE
               ! LM resolved charge density (up,down)
                  RHOLM(K,LM,1)=0.5_q*(RHO(K,LM,1)+RHO(K,LM,2))/(R%R(K)*R%R(K))
                  RHOLM(K,LM,2)=0.5_q*(RHO(K,LM,1)-RHO(K,LM,2))/(R%R(K)*R%R(K))               
               ! magnetization
                  RHOT(K,2)=RHOT(K,2)+YLM(NP,LM)*RHO(K,LM,2)
                  RHO_ANG(K,2,:)=RHO_ANG(K,2,:)+YLMD(NP,LM,:)*RHO(K,LM,2)              
               ENDIF
            ENDDO
            ! add core charge (spherical) and divide by 1/r^2
            RHOT(K,1)=(RHOT(K,1)+RHOC(K)*YLM(NP,1))/(R%R(K)*R%R(K))
!           RHOT(K,1)=(RHOT(K,1))/(R%R(K)*R%R(K))
            RHOT(K,2)= RHOT(K,2)/(R%R(K)*R%R(K))
            ! divide nabla Y_L through r^3 (1/r is missing in YLMD)
            RHO_ANG(K,:,:)=RHO_ANG(K,:,:)/(R%R(K)*R%R(K))/R%R(K)
            ! add core charge to RHOLM
            RHOLM(K,1,1)=RHOLM(K,1,1)+0.5_q*RHOC(K)/(R%R(K)*R%R(K))
            RHOLM(K,1,2)=RHOLM(K,1,2)+0.5_q*RHOC(K)/(R%R(K)*R%R(K))

            DO LM=1,LMMAX_TAU
               IF (ISPIN==1) THEN
               ! kinetic energy density
                  TAUU(K)=TAUU(K)+0.5_q*TAU(K,LM,1)*YLM(NP,LM)/(R%R(K)*R%R(K))
                  TAUD(K)=TAUU(K)
               ELSE
               ! kinetic energy density
                  TAUU(K)=TAUU(K)+TAU(K,LM,1)*YLM(NP,LM)/(R%R(K)*R%R(K))
                  TAUD(K)=TAUD(K)+TAU(K,LM,2)*YLM(NP,LM)/(R%R(K)*R%R(K))
               ENDIF
            ENDDO
            ! kinetic energy density of the core electrons
            IF (PRESENT(TAUC)) THEN
               TAUU(K)=TAUU(K)+0.5_q*TAUC(K)*YLM(NP,1)/(R%R(K)*R%R(K))
               TAUD(K)=TAUD(K)+0.5_q*TAUC(K)*YLM(NP,1)/(R%R(K)*R%R(K))
            ENDIF
            TAUU(K)=MAX(TAUU(K),TAUMIN)
            TAUD(K)=MAX(TAUD(K),TAUMIN)
         ENDDO

         ! d RHOLM / dr
         DO LM=1,LMMAX_CALC
            CALL GRAD(R,RHOLM(1:R%NMAX,LM,1),RHOLMD(1:R%NMAX,LM,1))
            CALL GRAD(R,RHOLM(1:R%NMAX,LM,2),RHOLMD(1:R%NMAX,LM,2))
         ENDDO         

         DO K=1,R%NMAX
            TEMP1=RHOT(K,1)
            RHOT(K,1)=(RHOT(K,1)+RHOT(K,2))/2    ! spin up
            RHOT(K,2)=(TEMP1-RHOT(K,2))/2        ! spin down
            
            RHOT(K,1)=MAX(RHOT(K,1), CHGMIN)
            RHOT(K,2)=MAX(RHOT(K,2), CHGMIN)
         ENDDO

         ! d rhot / dr (up,down)
         CALL GRAD(R,RHOT,T1)
         CALL GRAD(R,RHOT(1:R%NMAX,2),T1(1:R%NMAX,2))
         ! d^2 rhot / dr^2 (up,down)
         CALL GRAD(R,T1,T2)
         CALL GRAD(R,T1(1:R%NMAX,2),T2(1:R%NMAX,2))

         DO K=1,R%NMAX
            ! calculate the laplacian of the density
            LAPLUP=0; LAPLDW=0
            DO LM=1,LMMAX_CALC
               LAPLUP=LAPLUP+ &
              &    YLMDD(NP,LM,1)*RHOLM(K,LM,1)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,1)*XYZPTS(NP,1)+YLMD(NP,LM,1)*XYZPTS(NP,1))/R%R(K)*RHOLMD(K,LM,1) &
              &   +YLMDD(NP,LM,4)*RHOLM(K,LM,1)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,2)*XYZPTS(NP,2)+YLMD(NP,LM,2)*XYZPTS(NP,2))/R%R(K)*RHOLMD(K,LM,1) &
              &   +YLMDD(NP,LM,6)*RHOLM(K,LM,1)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,3)*XYZPTS(NP,3)+YLMD(NP,LM,3)*XYZPTS(NP,3))/R%R(K)*RHOLMD(K,LM,1)
               LAPLDW=LAPLDW+ &
              &    YLMDD(NP,LM,1)*RHOLM(K,LM,2)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,1)*XYZPTS(NP,1)+YLMD(NP,LM,1)*XYZPTS(NP,1))/R%R(K)*RHOLMD(K,LM,2) &
              &   +YLMDD(NP,LM,4)*RHOLM(K,LM,2)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,2)*XYZPTS(NP,2)+YLMD(NP,LM,2)*XYZPTS(NP,2))/R%R(K)*RHOLMD(K,LM,2) &
              &   +YLMDD(NP,LM,6)*RHOLM(K,LM,2)/(R%R(K)*R%R(K)) &
              &   +(YLMD(NP,LM,3)*XYZPTS(NP,3)+YLMD(NP,LM,3)*XYZPTS(NP,3))/R%R(K)*RHOLMD(K,LM,2)
            ENDDO
            
            LAPLUP=LAPLUP- &
           &    XYZPTS(NP,1)*XYZPTS(NP,1)/(R%R(K))*T1(K,1)+XYZPTS(NP,1)*XYZPTS(NP,1)*T2(K,1) &
           &   -XYZPTS(NP,2)*XYZPTS(NP,2)/(R%R(K))*T1(K,1)+XYZPTS(NP,2)*XYZPTS(NP,2)*T2(K,1) &
           &   -XYZPTS(NP,3)*XYZPTS(NP,3)/(R%R(K))*T1(K,1)+XYZPTS(NP,3)*XYZPTS(NP,3)*T2(K,1) &
           &   +3*T1(K,1)/R%R(K)
           
            LAPLDW=LAPLDW- &
           &    XYZPTS(NP,1)*XYZPTS(NP,1)/(R%R(K))*T1(K,2)+XYZPTS(NP,1)*XYZPTS(NP,1)*T2(K,2) &
           &   -XYZPTS(NP,2)*XYZPTS(NP,2)/(R%R(K))*T1(K,2)+XYZPTS(NP,2)*XYZPTS(NP,2)*T2(K,2) &
           &   -XYZPTS(NP,3)*XYZPTS(NP,3)/(R%R(K))*T1(K,2)+XYZPTS(NP,3)*XYZPTS(NP,3)*T2(K,2) &
           &   +3*T1(K,2)/R%R(K)

            ! norm of gradient for spin up
            XU  =T1(K,1)*XYZPTS(NP,1)+(RHO_ANG(K,1,1)+RHO_ANG(K,2,1))/2
            YU  =T1(K,1)*XYZPTS(NP,2)+(RHO_ANG(K,1,2)+RHO_ANG(K,2,2))/2
            ZU  =T1(K,1)*XYZPTS(NP,3)+(RHO_ANG(K,1,3)+RHO_ANG(K,2,3))/2
            T1(K,1)=SQRT(XU*XU+YU*YU+ZU*ZU)
            ! norm of gradient for spin down
            XD  =T1(K,2)*XYZPTS(NP,1)+(RHO_ANG(K,1,1)-RHO_ANG(K,2,1))/2
            YD  =T1(K,2)*XYZPTS(NP,2)+(RHO_ANG(K,1,2)-RHO_ANG(K,2,2))/2
            ZD  =T1(K,2)*XYZPTS(NP,3)+(RHO_ANG(K,1,3)-RHO_ANG(K,2,3))/2

            T1(K,2)=SQRT(XD*XD+YD*YD+ZD*ZD)
!           TG=T1(K,1)+T1(K,2) incorrect definition
            TG=SQRT((XU+XD)**2+(YU+YD)**2+(ZU+ZD)**2)
            
            T1(K,1)=MAX(T1(K,1),NABMIN); T1(K,2)=MAX(T1(K,2),NABMIN); TG=MAX(TG,2*NABMIN)

            CALL METAGGASPIN(&
           &   RHOT(K,1)*AUTOA3,RHOT(K,2)*AUTOA3,T1(K,1)*AUTOA4,T1(K,2)*AUTOA4,TG*AUTOA4, &
           &   LAPLUP*AUTOA5,LAPLDW*AUTOA5,TAUU(K)*EVTOH,TAUD(K)*EVTOH, &
           &   EXT,DEXC1,DEXC2,DVXC1,DVXC2,DVC,MUUP,MUDW)

            CALL SUM_GRHO_OVER_RHO_ONE_CENTER( &
           &   (RHOT(K,1)+RHOT(K,2)),TG*R%R(K)**2*AUTOA*R%SI(K)*SIM_FAKT)

            DEXC=DEXC+(EXT*RYTOEV)*(RHOT(K,1)+RHOT(K,2))*R%R(K)*R%R(K)*R%SI(K)*SIM_FAKT

            ! develop
            !             d    f_xc     grad rho
            !             ------------  --------  = vec dFXC(r)
            !             d |grad rho| |grad rho|
            ! into spherical harmonics
            ! afterwards vec dFXC can be reconstructed using
            ! dFXC_i(r) =  \sum_L dFXC_Li(r) Y_L(r)

            IF (LCALCPOT()) THEN
               DEXC1=DEXC1*RYTOEV
               DEXC2=DEXC2*RYTOEV
               DVXC1=DVXC1*RYTOEV*AUTOA
               DVXC2=DVXC2*RYTOEV*AUTOA
               DVC  =DVC  *RYTOEV*AUTOA

               DVXC1=DVXC1/ MAX(T1(K,1),NABMIN)
               DVXC2=DVXC2/ MAX(T1(K,2),NABMIN)
               DVC  =DVC  / MAX(TG,2*NABMIN)

               DO LM=1,LMMAX
                  dFXC(K,LM,0,1) = dFXC(K,LM,0,1)+DEXC1*SIM_FAKT*YLM(NP,LM)
                  dFXC(K,LM,0,2) = dFXC(K,LM,0,2)+DEXC2*SIM_FAKT*YLM(NP,LM)

                  dFXC(K,LM,1,1) = dFXC(K,LM,1,1)+(XU* DVXC1 + (XU+XD) * DVC)*SIM_FAKT*YLM(NP,LM)
                  dFXC(K,LM,2,1) = dFXC(K,LM,2,1)+(YU* DVXC1 + (YU+YD) * DVC)*SIM_FAKT*YLM(NP,LM)
                  dFXC(K,LM,3,1) = dFXC(K,LM,3,1)+(ZU* DVXC1 + (ZU+ZD) * DVC)*SIM_FAKT*YLM(NP,LM)
                         
                  dFXC(K,LM,1,2) = dFXC(K,LM,1,2)+(XD* DVXC2 + (XU+XD) * DVC)*SIM_FAKT*YLM(NP,LM)
                  dFXC(K,LM,2,2) = dFXC(K,LM,2,2)+(YD* DVXC2 + (YU+YD) * DVC)*SIM_FAKT*YLM(NP,LM)
                  dFXC(K,LM,3,2) = dFXC(K,LM,3,2)+(ZD* DVXC2 + (ZU+ZD) * DVC)*SIM_FAKT*YLM(NP,LM)
               ENDDO
            ENDIF
            
            IF (LCALCMU()) THEN
               MUUP=0.5_q*MUUP*RYTOEV*AUTOA2
               MUDW=0.5_q*MUDW*RYTOEV*AUTOA2
               DO LM=1,LMMAX_TAU
                  dFXCdT(K,LM,1) = dFXCdT(K,LM,1)+MUUP*SIM_FAKT*YLM(NP,LM)
                  dFXCdT(K,LM,2) = dFXCdT(K,LM,2)+MUDW*SIM_FAKT*YLM(NP,LM)
               ENDDO
            ENDIF
         ENDDO
      ENDDO points

!========================================================================
!
! GGA part related to d e_XC / grad n
!
! add -\sum_i Y_L  d (dFXC_L'i(x) Y_L'(x))/ d x_i to the potential V_L(r)
!
!                          x_i d dFXC_L'i(|x|)
! -\sum_L'i Y_L(x) Y_L'(x)  -  --------------- + dFXC_L'i(x) Y_L nabla_i Y_L'(x)
!                          |x|   d |x|     
!
!========================================================================
      IF (LCALCPOT()) THEN
         DO ISP=1,ISPIN
            DO LM=1,LMMAX_CALC
               DO K=1,R%NMAX
                  POT(K,LM,ISP)=POT(K,LM,ISP)+dFXC(K,LM,0,ISP)
               ENDDO
            ENDDO
         ENDDO
         ! nabla comment: 
         ! the matrix elements < Y_L| nabla | Y_L'> are non zero
         ! for L=L'+-1
         DO ISP=1,ISPIN
            DO I=1,3
               DO LMP=1,LMMAX
                  CALL GRAD(R,dFXC(1:R%NMAX,LMP,I,ISP),T1(1:R%NMAX,1))
                  DO LM=1,LMMAX_CALC
                     IF (ABS(YLM_NABLA_YLM(LM,LMP,I))>1E-4 .OR. ABS(YLM_X_YLM(LM,LMP,I))>1E-4) THEN
                        DO K=1,R%NMAX
                           POT(K,LM,ISP)=POT(K,LM,ISP)-( &
                                dFXC(K,LMP,I,ISP)*YLM_NABLA_YLM(LM,LMP,I)/ R%R(K)+ & 
                                T1(K,1)*YLM_X_YLM(LM,LMP,I) &
                              )
                        ENDDO
                  ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         DEALLOCATE(dFXC)
      ENDIF

      DCMU=0
      IF (LCALCMU()) THEN
         DO ISP=1,ISPIN
            DO LM=1,LMMAX_TAU
               DO K=1,R%NMAX
                  MU(K,LM,ISP)=dFXCdT(K,LM,ISP)
                  DCMU=DCMU+MU(K,LM,ISP)*TAU(K,LM,ISP)*R%SI(K)/HSQDTM
               ENDDO
            ENDDO
         ENDDO
         DEALLOCATE(dFXcdT)
      ENDIF

      ! finally calculate the double counting corrections again
      DVXC=0
      IF (LCALCPOT()) THEN
         CALL RAD_POT_IN_RHO(R,ISPIN,LMAX_CALC,RHO,POT,DVXC)
         DVXC=DVXC-DVINI
      ENDIF

      IF (LCALCMU()) THEN
         DVXC=DVXC+DCMU
!        WRITE(*,'(A,F24.14)') 'dcmu =',DCMU
      ENDIF

      DEALLOCATE(YLM,YLMD,YLMDD,XYZPTS,RADPTS,YLM_NABLA_YLM,YLM_X_YLM,WEIGHT,ABSCIS,RHOLM,RHOLMD)
      
      RETURN
      END SUBROUTINE RAD_METAGGA_XC


!********************* SUBROUTINE RAD_PROJ_METAGGA *********************
!
!***********************************************************************
      SUBROUTINE RAD_PROJ_METAGGA(&
     &   P,W,LMAX_TAU,MU,A,DLLMM)
      USE prec
      USE asa
      USE pseudo
      USE radial
      USE constant
      USE paw
      IMPLICIT NONE
      TYPE (potcar) P
      INTEGER LMAX_TAU
      REAL(q) A
      REAL(q) W(:,:)
      REAL(q) MU(:,:,:)
      OVERLAP DLLMM(:,:,:)
      ! local variables
      INTEGER ISP,I,K,Ch1,Ch2
      INTEGER LLMAX_TAU,LMMAX_TAU,LM,LMP,LL,LLP,M,MP,MPLOW,INDYLM,INDYLMP

      REAL(q) SUM
      REAL(q) W1(P%R%NMAX),W2(P%R%NMAX),DW1(P%R%NMAX),DW2(P%R%NMAX)
      REAL(q), ALLOCATABLE :: TAU(:,:)      

      ! quick return if possible
      IF ((.NOT.LDO_METAGGA()).OR.(.NOT.LCALCMU())) RETURN

      ! non-collinear magnetism
!     IF (SIZE(DLLMM,3)==4) THEN
!        WRITE(*,*) 'RAD_PROJ_METAGGA: internal error; non-collinear case not supported (yet).'
!        STOP
!     ENDIF
            
      ! check consistency
      LMMAX_TAU=(LMAX_TAU+1)**2
      IF (SIZE(MU,2)<LMMAX_TAU) THEN
         WRITE(*,*) 'RAD_PROJ_METAGGA: ERROR, size(MU,2)<LMMAX_TAU',SIZE(MU,2),LMMAX_TAU
         STOP
      ENDIF
      
      ! restriction for the angular development of \tau and \mu
      LLMAX_TAU=MIN(LMAXTAU,LMAX_TAU); LMMAX_TAU=(LLMAX_TAU+1)**2

      ALLOCATE(TAU(P%R%NMAX,LMMAX_TAU))
      
      ! loops over channels (l,epsilon)
      LM=1; DO Ch1=1,P%LMAX

      ! d(w1/r)/dr
      W1(:)=W(:,Ch1)/P%R%R(:); CALL GRAD(P%R,W1,DW1)
      
      LMP=LM; DO Ch2=Ch1,P%LMAX
      ! d(w2/r)/dr
      W2(:)=W(:,Ch2)/P%R%R(:); CALL GRAD(P%R,W2,DW2)
      
         LL=P%LPS(Ch1); LLP=P%LPS(Ch2)
         DO M=1,2*LL+1
         MPLOW=1; IF(Ch1==Ch2) MPLOW=M
         DO MP=MPLOW,2*LLP+1
            INDYLM= LL**2 +M
            INDYLMP=LLP**2+MP

            TAU=0
               
            DO I=1,LMMAX_TAU
               TAU(:,I)=TAU(:,I)+ &
              &   (DW1(:)*X_YLM_X_YLM(INDYLM,INDYLMP,I)*DW2(:)*P%R%R(:)*P%R%R(:) + &
              &    DW1(:)*X_YLM_YLMD(INDYLM,INDYLMP,I)*W2(:)*P%R%R(:) + &
              &    W1(:)*X_YLM_YLMD(INDYLMP,INDYLM,I)*DW2(:)*P%R%R(:) + &
              &    W1(:)*YLMD_YLMD(INDYLM,INDYLMP,I)*W2(:))
            ENDDO
            
            DO ISP=1,SIZE(DLLMM,3)
               SUM=0
               DO I=1,LMMAX_TAU
                  ! Integrate over r (Simpson integration)
                  DO K=1,P%R%NMAX
                     SUM=SUM+MU(K,I,ISP)*TAU(K,I)*P%R%SI(K)
                  ENDDO
               ENDDO
               DLLMM(LM+M-1,LMP+MP-1,ISP)=DLLMM(LM+M-1,LMP+MP-1,ISP)+A*SUM
               DLLMM(LMP+MP-1,LM+M-1,ISP)=DLLMM(LM+M-1,LMP+MP-1,ISP)
            ENDDO
            
         ENDDO
         ENDDO
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

!     CALL DUMP_DLLMM( "NABLA-MU-NABLA",DLLMM(:,:,1),P)

      DEALLOCATE(TAU)

      RETURN
      END SUBROUTINE RAD_PROJ_METAGGA


!********************* SUBROUTINE RAD_AUXILIARY_FUNCTIONS_METAGGA ******
!
!***********************************************************************
      SUBROUTINE RAD_AUXILIARY_FUNCTIONS_METAGGA(LLMAX)
      USE prec
      USE asa
      USE radial
      USE constant
      USE pseudo
      IMPLICIT NONE
      INTEGER LLMAX
      ! local variables
      INTEGER I,J,Ch1,Ch2
      INTEGER IFAIL,PHPTS,THPTS,NPTS,NP
      INTEGER LMMAX,LLMAX_TAU,LMMAX_TAU
      INTEGER LM,LMP,LL,LLP,M,MP,INDYLM,INDYLMP

      REAL(q), ALLOCATABLE :: RADPTS(:,:), XYZPTS(:,:)
      REAL(q), ALLOCATABLE :: YLM(:,:),YLMD(:,:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)

      REAL(q) DELTAPHI,SIM_FAKT
      EXTERNAL GAUSSI2

      ! quick return if possible
      IF (.NOT.LDO_METAGGA()) RETURN
     
      IF (ALLOCATED(X_YLM_X_YLM).AND.ALLOCATED(YLMD_YLMD).AND.ALLOCATED(X_YLM_YLMD)) RETURN

      LMMAX=(LLMAX+1)**2
      LLMAX_TAU=MIN(2*LLMAX+2,LMAXTAU); LMMAX_TAU=(LLMAX_TAU+1)**2
      
      IF (ALLOCATED(X_YLM_X_YLM)) DEALLOCATE(X_YLM_X_YLM)
      ALLOCATE(X_YLM_X_YLM(LMMAX,LMMAX,LMMAX_TAU)); X_YLM_X_YLM=0

      IF (ALLOCATED(YLMD_YLMD)) DEALLOCATE(YLMD_YLMD)
      ALLOCATE(YLMD_YLMD(LMMAX,LMMAX,LMMAX_TAU)); YLMD_YLMD=0

      IF (ALLOCATED(X_YLM_YLMD)) DEALLOCATE(X_YLM_YLMD)
      ALLOCATE(X_YLM_YLMD(LMMAX,LMMAX,LMMAX_TAU)); X_YLM_YLMD=0


!========================================================================
! number of theta and phi pivot points to perform angular integration
! since Exc=f(a*Yllmax,m) we need more pivot points than theoretically 
! needed to integrate Yllmax,m. 
! the factor 2 is the minium, 3 is more accurate
!========================================================================
      PHPTS=PHPTS_FACT*(MAX(LLMAX,LLMAX_TAU)+1)
      THPTS=THPTS_FACT*FLOOR(REAL(MAX(LLMAX,LLMAX_TAU)/2+1,KIND=q))
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(XYZPTS(NPTS,3),RADPTS(NPTS,2),WEIGHT(THPTS),ABSCIS(THPTS), & 
     &     YLM(NPTS,MAX(LMMAX,LMMAX_TAU)),YLMD(NPTS,MAX(LMMAX,LMMAX_TAU),3))
      
      ! set phi positions, equally spaced
      RADPTS=0; WEIGHT=0; ABSCIS=0
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

      ! get |r| Y_lm on a unit sphere and its derivatives
      YLM=0; YLMD=0
      CALL SETYLM_GRAD2(MAX(LLMAX,LLMAX_TAU),NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

      ! loop over all points in the angular grid
      points: DO NP=1,NPTS

         ! weight of this points
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))

         ! loops over channels (l,epsilon)
         DO LL =0,LLMAX
         DO LLP=0,LLMAX

            DO M=1,2*LL+1
            DO MP=1,2*LLP+1
               INDYLM= LL**2 +M
               INDYLMP=LLP**2+MP
               DO I=1,LMMAX_TAU
                  X_YLM_X_YLM(INDYLM,INDYLMP,I)=X_YLM_X_YLM(INDYLM,INDYLMP,I)+ SIM_FAKT*YLM(NP,I)* &
                 &   (XYZPTS(NP,1)*XYZPTS(NP,1)+XYZPTS(NP,2)*XYZPTS(NP,2)+XYZPTS(NP,3)*XYZPTS(NP,3))* & 
                 &   YLM(NP,INDYLM)*YLM(NP,INDYLMP)
                  X_YLM_YLMD(INDYLM,INDYLMP,I)=X_YLM_YLMD(INDYLM,INDYLMP,I)+ SIM_FAKT*YLM(NP,I)* &
                 &   (XYZPTS(NP,1)*YLMD(NP,INDYLMP,1)+XYZPTS(NP,2)*YLMD(NP,INDYLMP,2)+XYZPTS(NP,3)*YLMD(NP,INDYLMP,3))* &
                 &   YLM(NP,INDYLM)
                  YLMD_YLMD(INDYLM,INDYLMP,I)=YLMD_YLMD(INDYLM,INDYLMP,I)+ SIM_FAKT*YLM(NP,I)* &
                 &   (YLMD(NP,INDYLM,1)*YLMD(NP,INDYLMP,1)+YLMD(NP,INDYLM,2)*YLMD(NP,INDYLMP,2)+YLMD(NP,INDYLM,3)*YLMD(NP,INDYLMP,3))
               ENDDO
            ENDDO
            ENDDO
         ENDDO
         ENDDO

      ENDDO points

      DEALLOCATE(XYZPTS,RADPTS,WEIGHT,ABSCIS,YLM,YLMD)
      
      RETURN
      END SUBROUTINE RAD_AUXILIARY_FUNCTIONS_METAGGA


!******************** FUNCTION IR_MATCH ********************************
!        
!***********************************************************************
      FUNCTION IR_MATCH(PP,CHANNEL)
      USE prec
      USE pseudo 
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER CHANNEL
      INTEGER IR_MATCH
      ! local variables
      INTEGER K
      IR_MATCH=PP%R%NMAX
      DO K=PP%R%NMAX,1,-1
         IF (PP%WAE(K,CHANNEL)/=PP%WPS(K,CHANNEL)) EXIT
         IR_MATCH=K
      ENDDO
      END FUNCTION IR_MATCH


!******************** FUNCTION IR_PSMAX ********************************
!      
!***********************************************************************
      FUNCTION IR_PSMAX(PP)
      USE prec
      USE pseudo
      IMPLICIT NONE
      TYPE(potcar), POINTER :: PP
      INTEGER IR_PSMAX
      ! local variables
      INTEGER K
      IR_PSMAX=PP%R%NMAX
      DO K=PP%R%NMAX,2,-1
         IF ((PP%R%R(K)<PP%PSDMAX).OR.(PP%R%R(K)<PP%PSRMAX)) EXIT
         IR_PSMAX=K
      ENDDO
      END FUNCTION IR_PSMAX


!***********************************************************************
!***********************************************************************
  END MODULE meta
!***********************************************************************
!***********************************************************************


!******************** SUBROUTINE FEXCGS_METAGGA_ ***********************
!
!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
!***********************************************************************

      SUBROUTINE FEXCGS_METAGGA_(LPOT,LMU,&
     &   ISPIN,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &   CWGRAD,CHTOT,CWORK,CKINEDEN,DWGRAD,DHTOT,DWORK,DKINEDEN, &
     &   DENCOR,TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC,DMUWORK &
     &)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE metalib
      USE setxcmeta

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      LOGICAL LPOT,LMU

      COMPLEX(q) CWGRAD(GRIDC%MPLWV,ISPIN),CHTOT(GRIDC%MPLWV,ISPIN), &
     &   CKINEDEN(GRIDC%MPLWV,ISPIN),CWORK(GRIDC%MPLWV,ISPIN)
      RGRID DWGRAD(DIMREAL(GRIDC%MPLWV),ISPIN),DHTOT(DIMREAL(GRIDC%MPLWV),ISPIN), &
     &   DKINEDEN(DIMREAL(GRIDC%MPLWV),ISPIN),DWORK(DIMREAL(GRIDC%MPLWV),ISPIN)
              
      RGRID DENCOR(GRIDC%RL%NP),TAUC(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      REAL(q) DWORKG(GRIDC%RL%NP,ISPIN),DWORK1(GRIDC%RL%NP,ISPIN), &
     &   DWORK2(GRIDC%RL%NP,ISPIN),DWORK3(GRIDC%RL%NP,ISPIN),DVC(GRIDC%RL%NP)

      RGRID  DMUWORK(DIMREAL(GRIDC%MPLWV),ISPIN)

      REAL(q) LAPLUP,LAPLDW,LAPLACIAN(GRIDC%RL%NP,ISPIN)
      REAL(q) MUUP,MUDW,EDCMU,MU1
      REAL(q) :: CHGMIN=1E-10_q, NABMIN=1E-10_q, TAUMIN=1E-9_q, LAPMIN=1E-10_q

#ifdef debug2
      INTEGER, SAVE :: I_DBG=0
      REAL(Q) MU_DBG,TAU_DBG
#endif
     
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif

      LAPLACIAN=0

      RINPL=1._q/GRIDC%NPLWV                    ! Scaling of Energy
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5             ! KinEDens eV to Hartree

!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
    spin: DO ISP=1,ISPIN
      ! set CWORK to total real charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I,ISP)=(DENCOR(I)/ISPIN+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! GRAD_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! Compute the laplacian of the density
      DO IDIR=1,3  
         DO I=1,GRIDC%RC%NP  ! loop over all grid points NP in the reciprocal (RC) grid
            ! index of grid point along three reciprocal lattice vectors (N1, N2, N3)
            N1= MOD((I-1),GRIDC%RC%NROW) +1
            NC= (I-1)/GRIDC%RC%NROW+1
            N2= GRIDC%RC%I2(NC)
            N3= GRIDC%RC%I3(NC)
            ! convert to lattice vector component in direction x, y or z (corresponding to index J)
            GG=(GRIDC%LPCTX(N1)*LATT_CUR%B(IDIR,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(IDIR,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(IDIR,3))**2
            CWORK(I,ISP)=(CWGRAD(I,ISP))*GG*CITPI*CITPI
            !write(*,*)'xcgrad:cwgrad ', CWORK4(I), CWGRAD(I), G1, G2, CITPI
         ENDDO
         CALL SETUNB(CWORK(1,ISP),GRIDC)
         CALL FFT3D(CWORK(1,ISP),GRIDC,1)
         CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
         DO I=1,GRIDC%RL%NP
            LAPLACIAN(I,ISP)=LAPLACIAN(I,ISP)+REAL( DWORK(I,ISP) ,KIND=q)
         ENDDO
      ENDDO
    
      ENDDO spin
      

!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0; EDCMU=0
#ifdef debug2
      OPEN(100+I_DBG)
      TAU_DBG=0; MU_DBG=0
#endif
      DO I=1,GRIDC%RL%NP
         IF (ISPIN==2) THEN
         ! Spin-polarized case
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                          +DWORK3(I,1)*DWORK3(I,1))

            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                          +DWORK3(I,2)*DWORK3(I,2))

            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                    (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                    (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNAB=SQRT(ABSNAB)

            ABSNABUP=MAX(ABSNABUP,NABMIN)
            ABSNABDW=MAX(ABSNABDW,NABMIN)
            ABSNAB=MAX(ABSNAB,2*NABMIN)
!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
            ABSNAB=ABSNABUP+ABSNABDW
#endif
            ! kinetic energy density
            TAUU=MAX(REAL(DKINEDEN(I,1)+TAUC(I)/2._q,KIND=q), TAUMIN)
            TAUD=MAX(REAL(DKINEDEN(I,2)+TAUC(I)/2._q,KIND=q), TAUMIN)
            
            ! Laplacian of the density
            LAPLUP=REAL(LAPLACIAN(I,1),KIND=q)
            LAPLDW=REAL(LAPLACIAN(I,2),KIND=q)
         ELSE
         ! nonspin-polarized case
         ! up and down-spin variables get half of the total values
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I))/2._q/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= RHO1

            ABSNAB=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                 +DWORK3(I,1)*DWORK3(I,1))

            ABSNABUP= 0.5_q*ABSNAB
            ABSNABDW= 0.5_q*ABSNAB

            ABSNABUP=MAX(ABSNABUP,NABMIN)
            ABSNABDW=MAX(ABSNABDW,NABMIN)
            ABSNAB=MAX(ABSNAB,2*NABMIN)

            ! kinetic energy density
            TAUU= MAX(0.5_q*REAL(DKINEDEN(I,1)+TAUC(I),KIND=q), TAUMIN)
            TAUD= TAUU

            ! Laplacian of the density
            LAPLUP=0.5_q*REAL(LAPLACIAN(I,1),KIND=q)
            LAPLDW=LAPLUP
         ENDIF

         ! The CMBJ parameter may depend on the spatial position,
         ! if so, it is set here to the right entry in CMBJ_AUX
         CALL SET_CMBJ_PW(I)

         CALL METAGGASPIN(&
        &   RHO1*AUTOA3,RHO2*AUTOA3,ABSNABUP*AUTOA4,ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
        &   LAPLUP*AUTOA5,LAPLDW*AUTOA5,TAUU*EVTOH,TAUD*EVTOH, &
        &   EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,MUUP,MUDW)

         RHO=RHO1+RHO2

         IF (LscMBJ()) CALL SUM_GRHO_OVER_RHO_PW(RHO*AUTOA3,ABSNAB*AUTOA4)
#ifdef debug2
         IF (ABS(MUUP)>1._q) THEN
            WRITE(100+I_DBG,'(A,I9,7F18.11)') 'in ',I,RHO1,RHO2,ABSNABUP,ABSNABDW,ABSNAB,TAUU,TAUD
            WRITE(100+I_DBG,'(A,I9,8F18.11)') 'out',I,EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,MUUP,MUDW
         ENDIF
#endif

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA

         ! Double counting contribution stemming from dExc/d\mu
         EDCMU=EDCMU+0.5_q*(MUUP*TAUU+MUDW*TAUD-(MUUP+MUDW)*TAUC(I)/2._q)*LATT_CUR%OMEGA
#ifdef debug2
         TAU_DBG=TAU_+(TAUU+TAUD)*LATT_CUR%OMEGA
         MU_DBG=MU_DBG+MUUP
#endif
         MUUP=0.5_q*MUUP*RYTOEV*AUTOA2 ! HSQDTM = (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
         MUDW=0.5_q*MUDW*RYTOEV*AUTOA2

#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif

         ! Store d f_xc / d \tau_up,down in DMUWORK if required
         IF (LMU) THEN
            DMUWORK(I,1)=MUUP
            IF (ISPIN==2) DMUWORK(I,2)=MUDW
         ENDIF

         ! Store d f_x/ d (|\nabla \rho_up,down| ) / |\nabla \rho_up,down| in DWORK
         ! and  d f_xc/ d \rho_up,down  in DWORKG
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP, NABMIN)
         DWORKG(I,1) = DEXC1*RYTOEV
         IF (ISPIN==2) THEN
            DWORK(I,2)  = DVXC2 / MAX(ABSNABDW, NABMIN)
            DWORKG(I,2) = DEXC2*RYTOEV
         ENDIF
         ! Store d f_c/ d(|\nabla \rho|) / |\nabla \rho| in DVC
         DVC(I)=DVC_/MAX(ABSNAB, NABMIN)
      ENDDO
#ifdef debug2
      CLOSE(100+I_DBG)
      I_DBG=I_DBG+1
#endif

#ifdef debug
      REWIND(77)
      REWIND(78)
      REWIND(79)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
#endif      
#ifdef debug2
      OPEN(I_DBG+500-1)
      I=1
      DO N3=0,GRIDC%NGZ-1
         X3=MOD(N3*1._q/GRIDC%NGZ+10.5,1._q)-0.5
      DO N2=0,GRIDC%NGY-1
         X2=MOD(N2*1._q/GRIDC%NGY+10.5,1._q)-0.5
      DO N1=0,GRIDC%NGX-1
         X1=MOD(N1*1._q/GRIDC%NGX+10.5,1._q)-0.5
         X=X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
         Y=X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
         Z=X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)
         IF (ABS(REAL(DMUWORK(I,1),q)*2._q/RYTOEV/AUTOA2) > 1._q) THEN
            WRITE(I_DBG+500-1,'(3F14.7,F18.11)') X,Y,Z,REAL(DMUWORK(I,1),q)*2._q/RYTOEV/AUTOA2
         ENDIF
         I=I+1
      ENDDO
      ENDDO
      ENDDO               
      CLOSE(I_DBG+500-1)
#endif

!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      IF (ISPIN==2) THEN
         DO ISP=1,ISPIN
         DO I=1,GRIDC%RL%NP
            SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
            SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
            SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
            SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
            SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
            SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
 
            SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
            SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
            SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
            SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
            SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
            SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
         ENDDO
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            SIF11=SIF11+DWORK1(I,1)*DWORK1(I,1)*DWORK(I,1)/2
            SIF22=SIF22+DWORK2(I,1)*DWORK2(I,1)*DWORK(I,1)/2
            SIF33=SIF33+DWORK3(I,1)*DWORK3(I,1)*DWORK(I,1)/2
            SIF12=SIF12+DWORK1(I,1)*DWORK2(I,1)*DWORK(I,1)/2
            SIF23=SIF23+DWORK2(I,1)*DWORK3(I,1)*DWORK(I,1)/2
            SIF31=SIF31+DWORK3(I,1)*DWORK1(I,1)*DWORK(I,1)/2
 
            SIF11=SIF11+DWORK1(I,1)*(DWORK1(I,1))*DVC(I)
            SIF22=SIF22+DWORK2(I,1)*(DWORK2(I,1))*DVC(I)
            SIF33=SIF33+DWORK3(I,1)*(DWORK3(I,1))*DVC(I)
            SIF12=SIF12+DWORK1(I,1)*(DWORK2(I,1))*DVC(I)
            SIF23=SIF23+DWORK2(I,1)*(DWORK3(I,1))*DVC(I)
            SIF31=SIF31+DWORK3(I,1)*(DWORK1(I,1))*DVC(I)
         ENDDO
      ENDIF
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      IF (ISPIN==2) THEN
         DO I=1,GRIDC%RL%NP
            ANAB1U= DWORK1(I,1)
            ANAB2U= DWORK2(I,1)
            ANAB3U= DWORK3(I,1)
            ANAB1D= DWORK1(I,2)
            ANAB2D= DWORK2(I,2)
            ANAB3D= DWORK3(I,2)

            DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

            DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            ANAB1U= DWORK1(I,1)/2
            ANAB2U= DWORK2(I,1)/2
            ANAB3U= DWORK3(I,1)/2
            ANAB1D= ANAB1U
            ANAB2D= ANAB2U
            ANAB3D= ANAB3U

            DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)
         ENDDO
      ENDIF

      spin2: DO ISP=1,ISPIN
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)

      ENDDO spin2

!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      IF (ISPIN==2) THEN
         DO I=1,GRIDC%RL%NP
         ! Spin-polarized case
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
            VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
            DWORK(I,1)=VXC1
            DWORK(I,2)=VXC2
            VXC = 0.5_q*(VXC1+VXC2)
            CVZERO=CVZERO+VXC
            XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
            XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q) -VXC2* REAL( DHTOT(I,2) ,KIND=q)
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP      
         ! nonspin-polarized case
            RHO= MAX(REAL((DHTOT(I,1)+DENCOR(I))/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            
            VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
            
            DWORK(I,1)=VXC1
            CVZERO=CVZERO+VXC1
            XCENCC=XCENCC-VXC1*RHO*LATT_CUR%OMEGA
            XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q)  
         ENDDO
      ENDIF

      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC-EDCMU)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC   =EXC*RINPL
      EDCMU =EDCMU*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31

      CALLMPI( M_sum_s(GRIDC%COMM, 3, EXC, XCENC, EDCMU, 0._q))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
         WRITE(*,'(A,F24.14)') '    xcencc  =',XCENCC
         WRITE(*,'(A,F24.14)') '     edcmu  =',EDCMU         
#ifdef debug2
         WRITE(*,'(A,F24.14)') '     tau    =',TAU_DBG*RINPL
         WRITE(*,'(A,F24.14)') '     mu     =',MU_DBG*RINPL
#endif
      ENDIF

      RETURN
      END SUBROUTINE FEXCGS_METAGGA_


!******************** SUBROUTINE FEXCGS_METAGGA_VDW_ ***********************
!
!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
!***********************************************************************

      SUBROUTINE FEXCGS_METAGGA_VDW_(LPOT,LMU,&
     &   ISPIN,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &   CWGRAD,CHTOT,CWORK,CKINEDEN,DWGRAD,DHTOT,DWORK,DKINEDEN, &
     &   DENCOR,TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC,DMUWORK, &
     &   DWORK4,DWORK5,DWORK6,DWORK7)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE metalib
      USE setxcmeta
      USE VDW_rVV10

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      LOGICAL LPOT,LMU

      COMPLEX(q) CWGRAD(GRIDC%MPLWV,ISPIN),CHTOT(GRIDC%MPLWV,ISPIN), &
     &   CKINEDEN(GRIDC%MPLWV,ISPIN),CWORK(GRIDC%MPLWV,ISPIN)
      RGRID DWGRAD(DIMREAL(GRIDC%MPLWV),ISPIN),DHTOT(DIMREAL(GRIDC%MPLWV),ISPIN), &
     &   DKINEDEN(DIMREAL(GRIDC%MPLWV),ISPIN),DWORK(DIMREAL(GRIDC%MPLWV),ISPIN)
              
      RGRID DENCOR(GRIDC%RL%NP),TAUC(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3),STRESS(3,3)
      REAL(q) DWORKG(GRIDC%RL%NP,ISPIN),DWORK1(GRIDC%RL%NP,ISPIN), &
     &   DWORK2(GRIDC%RL%NP,ISPIN),DWORK3(GRIDC%RL%NP,ISPIN),DVC(GRIDC%RL%NP)

      RGRID  DMUWORK(DIMREAL(GRIDC%MPLWV),ISPIN)

      REAL(q) DWORK4(GRIDC%RL%NP),DWORK6(GRIDC%RL%NP),DWORK7(GRIDC%RL%NP)
      RGRID   DWORK5(GRIDC%RL%NP)

      REAL(q) LAPLUP,LAPLDW,LAPLACIAN(GRIDC%RL%NP,ISPIN)
      REAL(q) MUUP,MUDW,EDCMU,MU1
      REAL(q) :: CHGMIN=1E-10_q, NABMIN=1E-10_q, TAUMIN=1E-9_q, LAPMIN=1E-10_q

#ifdef debug2
      INTEGER, SAVE :: I_DBG=0
      REAL(Q) MU_DBG,TAU_DBG
#endif
     
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif

      LAPLACIAN=0

      RINPL=1._q/GRIDC%NPLWV                    ! Scaling of Energy
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5             ! KinEDens eV to Hartree

!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
    spin: DO ISP=1,ISPIN
      ! set CWORK to total real charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I,ISP)=(DENCOR(I)/ISPIN+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! GRAD_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! Compute the laplacian of the density
      DO IDIR=1,3  
         DO I=1,GRIDC%RC%NP  ! loop over all grid points NP in the reciprocal (RC) grid
            ! index of grid point along three reciprocal lattice vectors (N1, N2, N3)
            N1= MOD((I-1),GRIDC%RC%NROW) +1
            NC= (I-1)/GRIDC%RC%NROW+1
            N2= GRIDC%RC%I2(NC)
            N3= GRIDC%RC%I3(NC)
            ! convert to lattice vector component in direction x, y or z (corresponding to index J)
            GG=(GRIDC%LPCTX(N1)*LATT_CUR%B(IDIR,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(IDIR,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(IDIR,3))**2
            CWORK(I,ISP)=(CWGRAD(I,ISP))*GG*CITPI*CITPI
            !write(*,*)'xcgrad:cwgrad ', CWORK4(I), CWGRAD(I), G1, G2, CITPI
         ENDDO
         CALL SETUNB(CWORK(1,ISP),GRIDC)
         CALL FFT3D(CWORK(1,ISP),GRIDC,1)
         CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
         DO I=1,GRIDC%RL%NP
            LAPLACIAN(I,ISP)=LAPLACIAN(I,ISP)+REAL( DWORK(I,ISP) ,KIND=q)
         ENDDO
      ENDDO
    
      ENDDO spin
      

!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0; EDCMU=0
#ifdef debug2
      OPEN(100+I_DBG)
      TAU_DBG=0; MU_DBG=0
#endif
      DO I=1,GRIDC%RL%NP
         IF (ISPIN==2) THEN
         ! Spin-polarized case
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                          +DWORK3(I,1)*DWORK3(I,1))

            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                          +DWORK3(I,2)*DWORK3(I,2))

            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                    (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                    (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNAB=SQRT(ABSNAB)

            ABSNABUP=MAX(ABSNABUP,NABMIN)
            ABSNABDW=MAX(ABSNABDW,NABMIN)
            ABSNAB=MAX(ABSNAB,2*NABMIN)
!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
            ABSNAB=ABSNABUP+ABSNABDW
#endif
            ! kinetic energy density
            TAUU=MAX(REAL(DKINEDEN(I,1)+TAUC(I)/2._q,KIND=q), TAUMIN)
            TAUD=MAX(REAL(DKINEDEN(I,2)+TAUC(I)/2._q,KIND=q), TAUMIN)
            
            ! Laplacian of the density
            LAPLUP=REAL(LAPLACIAN(I,1),KIND=q)
            LAPLDW=REAL(LAPLACIAN(I,2),KIND=q)

            DWORK4(I)=MAX(REAL((DENCOR(I)+DHTOT(I,1)+DHTOT(I,2))/LATT_CUR%OMEGA, kind=q),CHGMIN)
            DWORK7(I)=ABSNAB
         ELSE
         ! nonspin-polarized case
         ! up and down-spin variables get half of the total values
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I))/2._q/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= RHO1

            ABSNAB=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                 +DWORK3(I,1)*DWORK3(I,1))

            ABSNABUP= 0.5_q*ABSNAB
            ABSNABDW= 0.5_q*ABSNAB

            ABSNABUP=MAX(ABSNABUP,NABMIN)
            ABSNABDW=MAX(ABSNABDW,NABMIN)
            ABSNAB=MAX(ABSNAB,2*NABMIN)

            ! kinetic energy density
            TAUU= MAX(0.5_q*REAL(DKINEDEN(I,1)+TAUC(I),KIND=q), TAUMIN)
            TAUD= TAUU

            ! Laplacian of the density
            LAPLUP=0.5_q*REAL(LAPLACIAN(I,1),KIND=q)
            LAPLDW=LAPLUP

            DWORK4(I)=MAX(REAL((DENCOR(I)+DHTOT(I,1))/LATT_CUR%OMEGA, kind=q),CHGMIN)
            DWORK7(I)=ABSNAB
         ENDIF

         ! The CMBJ parameter may depend on the spatial position,
         ! if so, it is set here to the right entry in CMBJ_AUX
         CALL SET_CMBJ_PW(I)

         CALL METAGGASPIN(&
        &   RHO1*AUTOA3,RHO2*AUTOA3,ABSNABUP*AUTOA4,ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
        &   LAPLUP*AUTOA5,LAPLDW*AUTOA5,TAUU*EVTOH,TAUD*EVTOH, &
        &   EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,MUUP,MUDW)

         RHO=RHO1+RHO2

         IF (LscMBJ()) CALL SUM_GRHO_OVER_RHO_PW(RHO*AUTOA3,ABSNAB*AUTOA4)
#ifdef debug2
         IF (ABS(MUUP)>1._q) THEN
            WRITE(100+I_DBG,'(A,I9,7F18.11)') 'in ',I,RHO1,RHO2,ABSNABUP,ABSNABDW,ABSNAB,TAUU,TAUD
            WRITE(100+I_DBG,'(A,I9,8F18.11)') 'out',I,EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,MUUP,MUDW
         ENDIF
#endif

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA

         ! Double counting contribution stemming from dExc/d\mu
         EDCMU=EDCMU+0.5_q*(MUUP*TAUU+MUDW*TAUD-(MUUP+MUDW)*TAUC(I)/2._q)*LATT_CUR%OMEGA
#ifdef debug2
         TAU_DBG=TAU_+(TAUU+TAUD)*LATT_CUR%OMEGA
         MU_DBG=MU_DBG+MUUP
#endif
         MUUP=0.5_q*MUUP*RYTOEV*AUTOA2 ! HSQDTM = (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
         MUDW=0.5_q*MUDW*RYTOEV*AUTOA2

#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif

         ! Store d f_xc / d \tau_up,down in DMUWORK if required
         IF (LMU) THEN
            DMUWORK(I,1)=MUUP
            IF (ISPIN==2) DMUWORK(I,2)=MUDW
         ENDIF

         ! Store d f_x/ d (|\nabla \rho_up,down| ) / |\nabla \rho_up,down| in DWORK
         ! and  d f_xc/ d \rho_up,down  in DWORKG
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP, NABMIN)
         DWORKG(I,1) = DEXC1*RYTOEV
         IF (ISPIN==2) THEN
            DWORK(I,2)  = DVXC2 / MAX(ABSNABDW, NABMIN)
            DWORKG(I,2) = DEXC2*RYTOEV
         ENDIF
         ! Store d f_c/ d(|\nabla \rho|) / |\nabla \rho| in DVC
         DVC(I)=DVC_/MAX(ABSNAB, NABMIN)
      ENDDO
#ifdef debug2
      CLOSE(100+I_DBG)
      I_DBG=I_DBG+1
#endif

#ifdef debug
      REWIND(77)
      REWIND(78)
      REWIND(79)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
#endif      
#ifdef debug2
      OPEN(I_DBG+500-1)
      I=1
      DO N3=0,GRIDC%NGZ-1
         X3=MOD(N3*1._q/GRIDC%NGZ+10.5,1._q)-0.5
      DO N2=0,GRIDC%NGY-1
         X2=MOD(N2*1._q/GRIDC%NGY+10.5,1._q)-0.5
      DO N1=0,GRIDC%NGX-1
         X1=MOD(N1*1._q/GRIDC%NGX+10.5,1._q)-0.5
         X=X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
         Y=X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
         Z=X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)
         IF (ABS(REAL(DMUWORK(I,1),q)*2._q/RYTOEV/AUTOA2) > 1._q) THEN
            WRITE(I_DBG+500-1,'(3F14.7,F18.11)') X,Y,Z,REAL(DMUWORK(I,1),q)*2._q/RYTOEV/AUTOA2
         ENDIF
         I=I+1
      ENDDO
      ENDDO
      ENDDO               
      CLOSE(I_DBG+500-1)
#endif

      IF (ISPIN==1) THEN
         CALL VDW_NONLOC_RVV10(DWORK4, DWORK(:,1), DWORKG(:,1), DWORK7,EXC,GRIDC,LATT_CUR,STRESS)
      ELSEIF (ISPIN==2) THEN
         DWORK5=0._q
         DWORK6=0._q
      
         CALL VDW_NONLOC_RVV10_spin( DWORK4, DWORK5, DWORK6, DWORK7,EXC,GRIDC,LATT_CUR,STRESS)
         DO I=1,GRIDC%RL%NP 
            DWORK(I,1)=DWORK(I,1)+DWORK5(I)/ MAX(SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
        &             +DWORK3(I,1)*DWORK3(I,1)),1.E-10_q)
            DWORK(I,2)=DWORK(I,2)+DWORK5(I)/ MAX(SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
        &             +DWORK3(I,2)*DWORK3(I,2)),1.E-10_q)
         ENDDO
  
         DO I=1,GRIDC%RL%NP
            DWORKG(I,1)=DWORKG(I,1)+DWORK6(I)
            DWORKG(I,2)=DWORKG(I,2)+DWORK6(I)
         ENDDO
      ENDIF

!=======================================================================
! gradient terms in STRESS tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      IF (ISPIN==2) THEN
         DO ISP=1,ISPIN
         DO I=1,GRIDC%RL%NP
            SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
            SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
            SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
            SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
            SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
            SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
 
            SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
            SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
            SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
            SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
            SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
            SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
         ENDDO
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            SIF11=SIF11+DWORK1(I,1)*DWORK1(I,1)*DWORK(I,1)/2
            SIF22=SIF22+DWORK2(I,1)*DWORK2(I,1)*DWORK(I,1)/2
            SIF33=SIF33+DWORK3(I,1)*DWORK3(I,1)*DWORK(I,1)/2
            SIF12=SIF12+DWORK1(I,1)*DWORK2(I,1)*DWORK(I,1)/2
            SIF23=SIF23+DWORK2(I,1)*DWORK3(I,1)*DWORK(I,1)/2
            SIF31=SIF31+DWORK3(I,1)*DWORK1(I,1)*DWORK(I,1)/2
 
            SIF11=SIF11+DWORK1(I,1)*(DWORK1(I,1))*DVC(I)
            SIF22=SIF22+DWORK2(I,1)*(DWORK2(I,1))*DVC(I)
            SIF33=SIF33+DWORK3(I,1)*(DWORK3(I,1))*DVC(I)
            SIF12=SIF12+DWORK1(I,1)*(DWORK2(I,1))*DVC(I)
            SIF23=SIF23+DWORK2(I,1)*(DWORK3(I,1))*DVC(I)
            SIF31=SIF31+DWORK3(I,1)*(DWORK1(I,1))*DVC(I)
         ENDDO
      ENDIF
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      IF (ISPIN==2) THEN
         DO I=1,GRIDC%RL%NP
            ANAB1U= DWORK1(I,1)
            ANAB2U= DWORK2(I,1)
            ANAB3U= DWORK3(I,1)
            ANAB1D= DWORK1(I,2)
            ANAB2D= DWORK2(I,2)
            ANAB3D= DWORK3(I,2)

            DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

            DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            ANAB1U= DWORK1(I,1)/2
            ANAB2U= DWORK2(I,1)/2
            ANAB3U= DWORK3(I,1)/2
            ANAB1D= ANAB1U
            ANAB2D= ANAB2U
            ANAB3D= ANAB3U

            DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
            DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
            DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)
         ENDDO
      ENDIF

      spin2: DO ISP=1,ISPIN
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)

      ENDDO spin2

!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      IF (ISPIN==2) THEN
         DO I=1,GRIDC%RL%NP
         ! Spin-polarized case
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/ISPIN)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

            VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
            VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
            DWORK(I,1)=VXC1
            DWORK(I,2)=VXC2
            VXC = 0.5_q*(VXC1+VXC2)
            CVZERO=CVZERO+VXC
            XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
            XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q) -VXC2* REAL( DHTOT(I,2) ,KIND=q)
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP      
         ! nonspin-polarized case
            RHO= MAX(REAL((DHTOT(I,1)+DENCOR(I))/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            
            VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
            
            DWORK(I,1)=VXC1
            CVZERO=CVZERO+VXC1
            XCENCC=XCENCC-VXC1*RHO*LATT_CUR%OMEGA
            XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q)  
         ENDDO
      ENDIF

      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC-EDCMU)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC   =EXC*RINPL
      EDCMU =EDCMU*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31

      XCSIF = XCSIF - STRESS

      CALLMPI( M_sum_s(GRIDC%COMM, 3, EXC, XCENC, EDCMU, 0._q))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
         WRITE(*,'(A,F24.14)') '    xcencc  =',XCENCC
         WRITE(*,'(A,F24.14)') '     edcmu  =',EDCMU         
#ifdef debug2
         WRITE(*,'(A,F24.14)') '     tau    =',TAU_DBG*RINPL
         WRITE(*,'(A,F24.14)') '     mu     =',MU_DBG*RINPL
#endif
      ENDIF

      RETURN
      END SUBROUTINE FEXCGS_METAGGA_VDW_


!***************** SUBROUTINE FEXCGS_METAGGA_NONCOL ********************
!
!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
!***********************************************************************

      SUBROUTINE FEXCGS_METAGGA_NONCOL(LPOT,LMU,&
     &   NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &   CWGRAD,CHTOT,CWORK,CKINEDEN,DWGRAD,DHTOT,DWORK,DKINEDEN, &
     &   DENCOR,TAUC,DWORKG,DWORK1,DWORK2,DWORK3,DVC, &
     &   DMUWORK &
     &)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE metalib
      USE setxcmeta

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      LOGICAL LPOT,LMU

      COMPLEX(q) CWGRAD(GRIDC%MPLWV,NCDIJ),CHTOT(GRIDC%MPLWV,NCDIJ), &
     &   CKINEDEN(GRIDC%MPLWV,NCDIJ),CWORK(GRIDC%MPLWV,NCDIJ)
      RGRID DWGRAD(DIMREAL(GRIDC%MPLWV),NCDIJ),DHTOT(DIMREAL(GRIDC%MPLWV),NCDIJ), &
     &   DKINEDEN(DIMREAL(GRIDC%MPLWV),NCDIJ),DWORK(DIMREAL(GRIDC%MPLWV),NCDIJ)

      RGRID DENCOR(GRIDC%RL%NP),TAUC(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      REAL(q) DWORKG(GRIDC%RL%NP,NCDIJ/2),DWORK1(GRIDC%RL%NP,NCDIJ), &
     &   DWORK2(GRIDC%RL%NP,NCDIJ),DWORK3(GRIDC%RL%NP,NCDIJ),DVC(GRIDC%RL%NP)
      REAL(q) MAG_NORM,NABMAG(3),TAU_NORM,TAU_DOT_MDIR,LAP_NORM,LAP_DOT_MDIR

      RGRID  DMUWORK(DIMREAL(GRIDC%MPLWV),NCDIJ)

      REAL(q) LAPLUP,LAPLDW,LAPLACIAN(GRIDC%RL%NP,NCDIJ)
      REAL(q) MUUP,MUDW,EDCMU
      REAL(q) :: CHGMIN=1E-10_q, NABMIN=1E-10_q, TAUMIN=1E-10_q
      
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif

      LAPLACIAN=0

      RINPL=1._q/GRIDC%NPLWV                    ! Scaling of Energy
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5             ! KinEDens eV to Hartree

!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
    spin: DO ISP=1,NCDIJ
      IF (ISP==1) THEN
      ! Add core charge density to pseudo charge density
         DO I=1,GRIDC%RL%NP
            DWORK(I,ISP)=(DENCOR(I)+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            DWORK(I,ISP)=DHTOT(I,ISP)*RINPL/LATT_CUR%OMEGA
         ENDDO
      ENDIF

      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO

! Compute the laplacian of the density
      DO IDIR=1,3  
         DO I=1,GRIDC%RC%NP  ! loop over all grid points NP in the reciprocal (RC) grid
            ! index of grid point along three reciprocal lattice vectors (N1, N2, N3)
            N1= MOD((I-1),GRIDC%RC%NROW) +1
            NC= (I-1)/GRIDC%RC%NROW+1
            N2= GRIDC%RC%I2(NC)
            N3= GRIDC%RC%I3(NC)
            ! convert to lattice vector component in direction x, y or z (corresponding to index J)
            GG=(GRIDC%LPCTX(N1)*LATT_CUR%B(IDIR,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(IDIR,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(IDIR,3))**2
            !write(*,*)'xcgrad:cwgrad ', CWORK4(I), CWGRAD(I), G1, G2, CITPI
            CWORK(I,ISP)=(CWGRAD(I,ISP))*GG*CITPI*CITPI
         ENDDO
         CALL SETUNB(CWORK(1,ISP),GRIDC)
         CALL FFT3D(CWORK(1,ISP),GRIDC,1)
         CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
         DO I=1,GRIDC%RL%NP
            LAPLACIAN(I,ISP)=LAPLACIAN(I,ISP)+REAL( DWORK(I,ISP) ,KIND=q)
         ENDDO
      ENDDO
      
      ENDDO spin

!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0; EDCMU=0
      DO I=1,GRIDC%RL%NP
!     
! | m |
!
         MAG_NORM=MAX(SQRT(ABS( &
        &    DHTOT(I,2)*DHTOT(I,2)+DHTOT(I,3)*DHTOT(I,3)+DHTOT(I,4)*DHTOT(I,4))), CHGMIN)        
!       
! RHO1: \rho_up   = ( \rho_tot + | m | )/2
! RHO2: \rho_down = ( \rho_tot - | m | )/2
!
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)+MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,1)+DENCOR(I)-MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
!        
! \nabla | m |
!
         NABMAG(1)=(DWORK1(I,2)*DHTOT(I,2)+DWORK1(I,3)*DHTOT(I,3)+DWORK1(I,4)*DHTOT(I,4))/MAG_NORM
         NABMAG(2)=(DWORK2(I,2)*DHTOT(I,2)+DWORK2(I,3)*DHTOT(I,3)+DWORK2(I,4)*DHTOT(I,4))/MAG_NORM
         NABMAG(3)=(DWORK3(I,2)*DHTOT(I,2)+DWORK3(I,3)*DHTOT(I,3)+DWORK3(I,4)*DHTOT(I,4))/MAG_NORM
!        
! | ( \nabla \rho + \nabla | m | )/2 |
!
         ABSNABUP=SQRT( &
        &            (DWORK1(I,1)+NABMAG(1))*(DWORK1(I,1)+NABMAG(1)) + &
        &             (DWORK2(I,1)+NABMAG(2))*(DWORK2(I,1)+NABMAG(2)) + &
        &              (DWORK3(I,1)+NABMAG(3))*(DWORK3(I,1)+NABMAG(3)) ) / 2
!
! | ( \nabla \rho - \nabla | m | )/2 |
!
         ABSNABDW=SQRT( &
        &            (DWORK1(I,1)-NABMAG(1))*(DWORK1(I,1)-NABMAG(1)) + &
        &             (DWORK2(I,1)-NABMAG(2))*(DWORK2(I,1)-NABMAG(2)) + &
        &              (DWORK3(I,1)-NABMAG(3))*(DWORK3(I,1)-NABMAG(3)) ) / 2
!
! | \nabla \rho |
!
         ABSNAB=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1)+DWORK3(I,1)*DWORK3(I,1))
!
! Refill DWORK[1..3](:,2) with ( \nabla \rho - \nabla | m | )/2
!
         DWORK1(I,2)=(DWORK1(I,1)-NABMAG(1))/2
         DWORK2(I,2)=(DWORK2(I,1)-NABMAG(2))/2
         DWORK3(I,2)=(DWORK3(I,1)-NABMAG(3))/2
!
! Refill DWORK[1..3](:,1) with ( \nabla \rho + \nabla | m | )/2
!
         DWORK1(I,1)=(DWORK1(I,1)+NABMAG(1))/2
         DWORK2(I,1)=(DWORK2(I,1)+NABMAG(2))/2
         DWORK3(I,1)=(DWORK3(I,1)+NABMAG(3))/2

!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         ABSNAB=ABSNABUP+ABSNABDW
#endif
         !     
         ! | m |
         !
         TAU_NORM=MAX(SQRT(ABS( &
        &    DKINEDEN(I,2)*DKINEDEN(I,2)+DKINEDEN(I,3)*DKINEDEN(I,3)+DKINEDEN(I,4)*DKINEDEN(I,4))), TAUMIN)        
         TAU_DOT_MDIR=(DHTOT(I,2)*DKINEDEN(I,2)+DHTOT(I,3)*DKINEDEN(I,3)+DHTOT(I,4)*DKINEDEN(I,4))/MAG_NORM
         !       
         ! kinetic energy density
         ! TAUU: \tau_up   = ( \tau_tot + | m | )/2
         ! TAUD: \tau_down = ( \tau_tot - | m | )/2
         !
!        TAUU=MAX(REAL((DKINEDEN(I,1)+TAU_NORM)/2,KIND=q), TAUMIN)
!        TAUD=MAX(REAL((DKINEDEN(I,1)-TAU_NORM)/2,KIND=q), TAUMIN)
         TAUU=MAX(REAL((DKINEDEN(I,1)+TAU_DOT_MDIR)/2._q+TAUC(I)/2._q,KIND=q), TAUMIN)
         TAUD=MAX(REAL((DKINEDEN(I,1)-TAU_DOT_MDIR)/2._q+TAUC(I)/2._q,KIND=q), TAUMIN)
            
         !     
         ! | m |
         !
         LAP_NORM=MAX(SQRT(ABS( &
        &    LAPLACIAN(I,2)*LAPLACIAN(I,2)+LAPLACIAN(I,3)*LAPLACIAN(I,3)+LAPLACIAN(I,4)*LAPLACIAN(I,4))), TAUMIN)
         LAP_DOT_MDIR=(DHTOT(I,2)*LAPLACIAN(I,2)+DHTOT(I,3)*LAPLACIAN(I,3)+DHTOT(I,4)*LAPLACIAN(I,4))/MAG_NORM
         !       
         ! Laplacian of the density
!        LAPLUP=REAL((LAPLACIAN(I,1)+LAP_NORM)/2,KIND=q)
!        LAPLDW=REAL((LAPLACIAN(I,1)-LAP_NORM)/2,KIND=q)
         LAPLUP=REAL((LAPLACIAN(I,1)+LAP_DOT_MDIR)/2,KIND=q)
         LAPLDW=REAL((LAPLACIAN(I,1)-LAP_DOT_MDIR)/2,KIND=q)

         ! The CMBJ parameter may depend on the spatial position,
         ! if so, it is set here to the right entry in CMBJ_AUX
         CALL SET_CMBJ_PW(I)

         CALL METAGGASPIN(&
        &   RHO1*AUTOA3,RHO2*AUTOA3,ABSNABUP*AUTOA4,ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
        &   LAPLUP*AUTOA5,LAPLDW*AUTOA5,TAUU*EVTOH,TAUD*EVTOH, &
        &   EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,MUUP,MUDW)

         RHO=RHO1+RHO2

         IF (LscMBJ()) CALL SUM_GRHO_OVER_RHO_PW(RHO*AUTOA3,ABSNAB*AUTOA4)

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA

         ! Double counting contribution stemming from dExc/d\mu
         EDCMU=EDCMU+0.5_q*(MUUP*TAUU+MUDW*TAUD-(MUUP+MUDW)*TAUC(I)/2._q)*LATT_CUR%OMEGA

         MUUP=0.5_q*MUUP*RYTOEV*AUTOA2 ! HSQDTM = (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
         MUDW=0.5_q*MUDW*RYTOEV*AUTOA2

#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif

!test
!         DVXC1=0
!         DVXC2=0
!         DVC_ =0
!test
         ! Store d f_xc / d \tau_up,down in DMUWORK if required
         IF (LMU) THEN
            DMUWORK(I,1)=MUUP
            DMUWORK(I,2)=MUDW
         ENDIF

!
!   store d f/ d (|d rho| ) / |d rho|  in DWORK
!
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP,NABMIN)
         DWORK(I,2)  = DVXC2 / MAX(ABSNABDW,NABMIN)
         DVC(I)      = DVC_  / MAX(ABSNAB,NABMIN)
!
!   store d f/ d rho  in DWORKG
!
         DWORKG(I,1) = DEXC1*RYTOEV
         DWORKG(I,2) = DEXC2*RYTOEV

      ENDDO

!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO ISP=1,2
      DO I=1,GRIDC%RL%NP
         SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
         SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
         SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
         SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
         SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
         SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)

         SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
         SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
         SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
         SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
         SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
         SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
      ENDDO
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      DO I=1,GRIDC%RL%NP
         ANAB1U= DWORK1(I,1)
         ANAB2U= DWORK2(I,1)
         ANAB3U= DWORK3(I,1)
         ANAB1D= DWORK1(I,2)
         ANAB2D= DWORK2(I,2)
         ANAB3D= DWORK3(I,2)

         DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

         DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
      ENDDO

      spin2: DO ISP=1,2
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)

      ENDDO spin2


!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
!
! | m |
!
         MAG_NORM=SQRT(ABS(DHTOT(I,2)*DHTOT(I,2)+ DHTOT(I,3)*DHTOT(I,3) + DHTOT(I,4)*DHTOT(I,4)))
!
! RHO1: \rho_up   = ( \rho_tot + | m | )/2
! RHO2: \rho_down = ( \rho_tot - | m | )/2
!
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)+MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,1)+DENCOR(I)-MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
         VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
         DWORK(I,1)=VXC1
         DWORK(I,2)=VXC2
         VXC = 0.5_q*(VXC1+VXC2)
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC1* REAL( (DHTOT(I,1)+MAG_NORM)/2 ,KIND=q) &
        &             -VXC2* REAL( (DHTOT(I,1)-MAG_NORM)/2 ,KIND=q)
      ENDDO

      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC-EDCMU)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC   =EXC*RINPL
      EDCMU =EDCMU*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31

      CALLMPI( M_sum_s(GRIDC%COMM, 3, EXC, XCENC, EDCMU, 0._q ))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
         WRITE(*,'(A,F24.14)') '    xcencc  =',XCENCC
         WRITE(*,'(A,F24.14)') '     edcmu  =',EDCMU         
      ENDIF

      RETURN
      END SUBROUTINE FEXCGS_METAGGA_NONCOL


      SUBROUTINE MUxTAU(GRIDC,NCDIJ,LATT_CUR,DMUTOT,DKINEDEN,TAUC,DE)
      USE prec
      USE constant
      USE lattice
      USE mgrid
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE(grid_3d) GRIDC
      TYPE(latt) LATT_CUR
      INTEGER NCDIJ      
      RGRID DMUTOT(DIMREAL(GRIDC%MPLWV),NCDIJ)
      RGRID DKINEDEN(DIMREAL(GRIDC%MPLWV),NCDIJ),TAUC(GRIDC%RL%NP)
      REAL(q) DE
      ! local variables
      INTEGER I

      DE=0
      RINPL=1._q/GRIDC%NPLWV

      IF (NCDIJ==1) THEN
         DO I=1,GRIDC%RL%NP
            DE=DE+DMUTOT(I,1)*CHGCONJG(DKINEDEN(I,1)+TAUC(I))
         ENDDO
      ELSEIF (NCDIJ==2) THEN
         DO I=1,GRIDC%RL%NP
            DE=DE+DMUTOT(I,1)*CHGCONJG(DKINEDEN(I,1)+TAUC(I)/2._q)+ &
           &       DMUTOT(I,2)*CHGCONJG(DKINEDEN(I,2)+TAUC(I)/2._q)
         ENDDO
      ELSEIF (NCDIJ==4) THEN
         DO I=1,GRIDC%RL%NP
            DE=DE+DMUTOT(I,1)*CHGCONJG(DKINEDEN(I,1)+TAUC(I)/2._q)+ &
           &       DMUTOT(I,2)*CHGCONJG(DKINEDEN(I,2))+DMUTOT(I,3)*CHGCONJG(DKINEDEN(I,3))+ &
           &        DMUTOT(I,4)*CHGCONJG((DKINEDEN(I,4)+TAUC(I)/2._q))
         ENDDO
      ELSE
         WRITE(*,*) 'MUxTAU: internal error, NCDIJ=',NCDIJ
         STOP
      ENDIF

      DE=DE*RINPL*LATT_CUR%OMEGA/(RYTOEV*AUTOA2)

      CALLMPI( M_sum_d(GRIDC%COMM,DE,1))

      RETURN
      END SUBROUTINE MUxTAU


!************************** M E T A G G A . F ******************************
! Implementation of the METAGGA according to 
! Perdew, Kurth, Zupan and Blaha (PRL 82, 2544)
!
! All subroutines in this File were written by Robin Hirschl in Dec. 2000
! using templates supplied by Georg Kresse
! Thanks to Georg for his encouragement and support
!
!***************************************************************************


!************************ SUBROUTINE GGASPINCOR ************************
!
!  calculate the correlation energy density according to the
!  Perdew, Burke and Ernzerhof functional
!
!***********************************************************************

    SUBROUTINE GGASPINCOR(D1,D2,DDA, EC)

!     D1   density up
!     D2   density down
!     DDA  |gradient of the total density|

      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (THRD=1._q/3._q)

      D=D1+D2
      DTHRD=exp(log(D)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD

      ZETA=(D1-D2)/D
      ZETA=MIN(MAX(ZETA,-0.9999999999999_q),0.9999999999999_q)

      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)
      G = (exp((2*THRD)*log(1._q+ZETA)) &
                +exp((2*THRD)*log(1._q-ZETA)))/2._q
      T = DDA/(D*2._q*SK*G)

      CALL corpbe(RS,ZETA,ECLDA,ECD1LDA,ECD2LDA,G,SK, &
           T,EC,ECD1,ECD2,ECQ,.TRUE.)

      EC  =(EC  +ECLDA)

      RETURN
    END SUBROUTINE GGASPINCOR

!************************ SUBROUTINE GGACOR *****************************
!
!  calculate the correlation energy density according to the
!  Perdew, Burke and Ernzerhof functional
!
!***********************************************************************

    SUBROUTINE GGACOR(D, DD, EC)
      USE prec
      USE constant
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (THRD=1._q/3._q)

      IF (D<0) THEN
         EC   = 0._q
         RETURN
      ENDIF

      DTHRD=exp(log(D)*THRD)
      RS=(0.75_q/PI)**THRD/DTHRD
      FK=(3._q*PI*PI)**THRD*DTHRD
      SK = SQRT(4.0_q*FK/PI)

      IF(D>1.E-10_q)THEN
         T=DD/(D*SK*2._q)
      ELSE
         T=0.0_q
      ENDIF

      CALL CORunspPBE(RS,ECLDA,ECDLDA,SK, &
           T,EC,ECD,ECDD,.TRUE.)

      EC = (EC+ECLDA)

      RETURN
    END SUBROUTINE GGACOR


!=======================================================================
! 
! SUBROUTINE TAU_PW
!
! This subroutine calculates the kinetic energy of the PW part of the
! wavefunctions (0.5*|grad psi|**2) 
! and the Weizsaecker kinetic energy density, output on GRIDC
!
! INPUT: GRIDC,  LATT_CUR, SYMM, NIOND, W, WDES
! OUTPUT: TAU(GRIDC%MPLWC, WDES%NCDIJ)
!
! Robin Hirschl 20001221 
!=======================================================================

SUBROUTINE TAU_PW(GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM,NIOND, &
        W,WDES,TAU)
      USE prec
      USE lattice
      USE mgrid
      USE msymmetry
      USE base
      USE wave
      USE mpimy
      USE constant
      
      IMPLICIT NONE

      TYPE (grid_3d)     GRIDC,GRID,GRID_SOFT
      TYPE (latt)        LATT_CUR
      TYPE (transit)     SOFT_TO_C
      TYPE (symmetry)    SYMM
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
! result
      RGRID ::  TAU(DIMREAL(GRIDC%MPLWV),WDES%NCDIJ)  ! kinetic energy density
! dynamic work array
      RGRID,ALLOCATABLE :: CKIN(:)

      COMPLEX(q),ALLOCATABLE :: CFAR(:),CFBR(:)
      COMPLEX(q) :: CFA(WDES%NRPLWV,3),CFB(WDES%NRPLWV,3)
      
      INTEGER :: ISPINOR,ISPINOR_
      INTEGER :: MPLWV,ISP,N,NK,NPL,IDIR,I,NIOND
      REAL(q) :: WEIGHT,G1,G2,G3
      COMPLEX(q) :: GC

      MPLWV=MAX(GRID%MPLWV, GRID_SOFT%MPLWV)
      ALLOCATE(CFAR(MPLWV),CFBR(MPLWV))
      ALLOCATE(CKIN(DIMREAL(MPLWV)))
      
      TAU=0
      
spin: DO ISP=1,WDES%ISPIN

      DO ISPINOR=0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1
      
      CFAR=0 ; CFBR=0; CKIN=0
      
      band: DO N=1,WDES%NBANDS
         kpoints: DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            WEIGHT=WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)/LATT_CUR%OMEGA
            NPL=WDES%NGVECTOR(NK)
            CFA=0; CFB=0
! loop over plane wave coefficients            
            DO I=1,NPL
! get k-vector of respective k-point and coefficient
               G1=WDES%IGX(I,NK)+WDES%VKPT(1,NK)
               G2=WDES%IGY(I,NK)+WDES%VKPT(2,NK)
               G3=WDES%IGZ(I,NK)+WDES%VKPT(3,NK)
! loop over cartesian directions
               DO IDIR=1,3
                  GC=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))
                  CFA(I,IDIR)=GC*W%CPTWFP(I+ISPINOR *NPL,N,NK,ISP)*CITPI
                  CFB(I,IDIR)=GC*W%CPTWFP(I+ISPINOR_*NPL,N,NK,ISP)*CITPI
               ENDDO
            ENDDO

! fourier trafo of gradient of wave-function 
! loop over cartesian directions
            DO IDIR=1,3
               CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CFAR(1),CFA(1,IDIR),GRID)
               CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CFBR(1),CFB(1,IDIR),GRID)
! update kinetic energy density (in CW2) and grad rho^2 (in CW4)
               DO I=1,GRID%RL%NP
#ifdef realmode
                  CKIN(I)=CKIN(I)+HSQDTM*REAL(CONJG(CFBR(I))*CFAR(I),KIND=q)*WEIGHT
#else
                  CKIN(I)=CKIN(I)+HSQDTM*(CONJG(CFBR(I))*CFAR(I))*WEIGHT
#endif            
               ENDDO
            ENDDO

        ENDDO kpoints
      ENDDO band

! merge results from nodes
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CKIN(1), GRID%RL%NP))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, CKIN(1), GRID%RL%NP))
#else
      CALLMPI( M_sum_z(WDES%COMM_INTER, CKIN(1), GRID%RL%NP))
      CALLMPI( M_sum_z(WDES%COMM_KINTER, CKIN(1), GRID%RL%NP))
#endif

! rescaling
      CKIN=CKIN/GRID_SOFT%NPLWV
    
! to rec space
      CALL FFT3D(CKIN(1),GRID_SOFT,-1)
      
! transition to finer grid
      CALL CPB_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,CKIN(1),TAU(1,ISP+ISPINOR_+2*ISPINOR))
!     CALL SETUNB_COMPAT(TAU(1,ISP+ISPINOR_+2*ISPINOR),GRIDC)
      ENDDO ! ispinor_
      ENDDO ! ispinor
      ENDDO spin


      IF (WDES%LNONCOLLINEAR) THEN
         ! needs (total,mag) instead of up,dw
         CALL RC_FLIP(TAU,GRIDC,WDES%NCDIJ,.FALSE.)
         IF (SYMM%ISYM>0) THEN
            CALL RHOSYM(TAU(1,1),GRIDC,SYMM%PTRANS,NIOND,SYMM%MAGROT,1)
            IF (.NOT.WDES%LSPIRAL) THEN
               CALL SYMFIELD(TAU(1,2),GRIDC,SYMM%PTRANS,NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
            ENDIF
         ENDIF
      ELSE
         IF (SYMM%ISYM>0) THEN
            ! symmetrization of result TAU(:,ISP)
            ! needs (total,mag) instead of up,dw
            CALL RC_FLIP(TAU, GRIDC,WDES%NCDIJ,.FALSE.)
            DO ISP=1,WDES%ISPIN
               CALL RHOSYM(TAU(1,ISP),GRIDC,SYMM%PTRANS,NIOND,SYMM%MAGROT,ISP)
            ENDDO
            ! in the collinear case we go back to up,dw
            CALL RC_FLIP(TAU,GRIDC,WDES%NCDIJ,.TRUE.)
         ENDIF      
      ENDIF
 
! back to real space
      DO ISP=1,WDES%NCDIJ
         CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, TAU(1,ISP))
         CALL FFT3D(TAU(1,ISP),GRIDC,1)
      ENDDO
      TAU=REAL(TAU,KIND=q)
   
      DEALLOCATE(CFAR,CFBR,CKIN)
   
      RETURN
END SUBROUTINE TAU_PW


!************************ SUBROUTINE METAGGA ***************************
!
! calculates local contribution to metagga Exc according to 
! Perdew et. al. PRL 82, 12 (1999)
!
! RH 20001119
!
! everything in Hartree units
!
! ATTANTION: Every values are passed "as they are", i.e. including
! possibly unphysical numerical errors (e.g. negative charge densities)
! values need to be checked accordingly
!***********************************************************************

SUBROUTINE METAGGA(RU,RD,DRU,DRD,DRT,TAUU,TAUD,TAUWU,TAUWD,EX,EC,I)

! RU,RD      density up,down
! DRU, DRD   abs. val. gradient of density up/down
! DRT        abs. val. gradient of total density
! TAUU,TAUD  kinetic energy density up/down
! TAUWU,TAUWD Weizsaecker kinetic energy density up/down
! EXC        return value

  USE prec
  USE constant
  IMPLICIT REAL(q) (A-H,O-Z)

  INTEGER I
! the following parameters are given by Perdew et.al.
  PARAMETER (RKAPPA=0.804_q)
  PARAMETER (D=0.113_q)
  PARAMETER (C=0.53_q)
! other parameters
  PARAMETER (THRD=1._q/3._q)
  PARAMETER (TTHRD=2._q*THRD)
  PARAMETER (FTHRD=1._q+TTHRD)
  PARAMETER (ETHRD=1._q+FTHRD)
  PARAMETER (PISQ=PI*PI)


  EX=0._q;EC=0._q
! exchange energy
! spin up
     P=(2._q*DRU)**2._q/(4._q*(3._q*PISQ)**TTHRD*(2._q*RU)**ETHRD)
     QQS=6._q*TAUU/(2._q*(3._q*PISQ)**TTHRD*(2._q*RU)**FTHRD)-9._q/20._q-P/12._q
     X=10._q/81._q*P+146._q/2025._q*QQS*QQS-73._q/405._q*QQS*P
     X=X+(D+1._q/RKAPPA*(10._q/81._q)**2._q)*P*P
     FX=1._q+RKAPPA-RKAPPA/(1._q+(X/RKAPPA))
     EX=EX-RU*(3._q/(4._q*PI))*(3._q*PISQ*2._q*RU)**THRD*FX
! spin down
     P=(2._q*DRD)**2._q/(4._q*(3._q*PISQ)**TTHRD*(2._q*RD)**ETHRD)
     QQS=6._q*TAUD/(2._q*(3._q*PISQ)**TTHRD*(2._q*RD)**FTHRD)-9._q/20._q-P/12._q
     X=10._q/81._q*P+146._q/2025._q*QQS*QQS-73._q/405._q*QQS*P
     X=X+(D+1._q/RKAPPA*(10._q/81._q)**2._q)*P*P
     FX=1._q+RKAPPA-RKAPPA/(1._q+(X/RKAPPA))
     EX=EX-RD*(3._q/(4._q*PI))*(3._q*PISQ*2._q*RD)**THRD*FX

! correlation energy
     CALL GGASPINCOR(RU,RD,DRT,ECT)
     TAUK=(TAUWU+TAUWD)/(TAUU+TAUD)
     ECM1=(RU+RD)*ECT*(1._q+C*TAUK**2._q)

!     CALL GGACOR(RU,DRU,ECU)
     CALL GGASPINCOR(RU,0.0_q,DRU,ECU)
     TAUK=TAUWU/TAUU
     ECM2=TAUK**2._q*RU*ECU

!     CALL GGACOR(RD,DRD,ECD)
     CALL GGASPINCOR(RD,0.0_q,DRD,ECD)
     TAUK=TAUWD/TAUD
     ECM3=TAUK**2._q*RD*ECD
     EC=ECM1-(1._q+C)*(ECM2+ECM3)

  RETURN
END SUBROUTINE METAGGA
