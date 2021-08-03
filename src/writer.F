#include "symbol.inc"
!***********************************************************************
!
! routine for writing the local magnetic moments at each atom
!
!***********************************************************************   

      MODULE writer
      USE prec
      USE base

      LOGICAL, PRIVATE,SAVE :: L_WR_MOMENTS
      LOGICAL, PRIVATE,SAVE :: L_WR_DENSITY
      INTEGER, PRIVATE,SAVE :: LDIMP,LMDIMP
      REAL(q), PRIVATE,ALLOCATABLE,SAVE :: ION_SUM_DETAIL(:,:,:)
      
      CONTAINS

      SUBROUTINE WRITER_READER(IU0,IU5)

      USE vaspxml
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      
      LOGICAL :: LOPEN,LDUM
      CHARACTER (1) :: CHARAC
! Reading the appropriate tokens from INCAR
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')     

      L_WR_MOMENTS=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'L_WR_MOMENTS','=','#',';','L', &
     &            IDUM,RDUM,CDUM,L_WR_MOMENTS,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''L_WR_MOMENTS'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('L_WR_MOMENTS','L',IDUM,RDUM,CDUM,L_WR_MOMENTS,CHARAC,N)

      L_WR_DENSITY=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'L_WR_DENSITY','=','#',';','L', &
     &            IDUM,RDUM,CDUM,L_WR_DENSITY,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''L_WR_DENSITY'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('L_WR_DENSITY','L',IDUM,RDUM,CDUM,L_WR_DENSITY,CHARAC,N)
      
      CLOSE(IU5)      
      
      RETURN

  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP

      END SUBROUTINE


      SUBROUTINE INIT_WRITER(P,T_INFO,WDES)
      
      USE prec
      USE poscar
      USE pseudo
      USE us
      USE wave
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      
      TYPE(type_info) T_INFO
      TYPE(potcar)    P(T_INFO%NTYP)
      TYPE(wavedes)   WDES

      LDIMP=MAXL(T_INFO%NTYP,P)+1
      LMDIMP=(LDIMP)**2
      ALLOCATE(ION_SUM_DETAIL(LMDIMP,T_INFO%NIONS,WDES%NCDIJ))     

      RETURN
      END SUBROUTINE INIT_WRITER

!***********************************************************************
!
! function to query whether we want to write the 
! magnetic moments to MAGCAR
!
!***********************************************************************      

      FUNCTION  WRITE_MOMENTS()
      IMPLICIT NONE
      LOGICAL WRITE_MOMENTS

      IF (L_WR_MOMENTS) THEN
         WRITE_MOMENTS=.TRUE.
      ELSE
         WRITE_MOMENTS=.FALSE.
      ENDIF

      END FUNCTION WRITE_MOMENTS
      
!***********************************************************************
!
! function to query whether we want to write the 
! magnetic moments to MAGCAR
!
!***********************************************************************      

      FUNCTION  WRITE_DENSITY()
      IMPLICIT NONE
      LOGICAL WRITE_DENSITY

      IF (L_WR_DENSITY) THEN
         WRITE_DENSITY=.TRUE.
      ELSE
         WRITE_DENSITY=.FALSE.
      ENDIF

      END FUNCTION WRITE_DENSITY
      
!************************ SUBROUTINE WR_MOMENTS ************************
!
! Yeah guys, I know this routine looks a lot like SPHPRO_FAST, but this
! is all done in the name of modular programming
!
!***********************************************************************   

      SUBROUTINE WR_MOMENTS( &
          GRID,LATT_CUR,P,T_INFO,W,WDES,LWRITE)
          
      USE prec
      USE main_mpi
      USE constant
      USE wave
      USE lattice
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES_1K
      
      TYPE (potcar), POINTER :: PP
      REAL(q) SUMION(LDIMP),SUMTOT(LDIMP)
      REAL(q) ION_SUM(LDIMP,T_INFO%NIONS,WDES%NCDIJ)
      REAL(q) MAG(T_INFO%NIONS,WDES%NCDIJ),X,Y,Z,DUMMY
      COMPLEX(q) :: CSUM(WDES%NBANDS,WDES%NKPTS,WDES%NCDIJ)
      LOGICAL LWRITE

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME= WDES%COMM%NODE_ME
      IONODE = WDES%COMM%IONODE
      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('WR_MOMENTS: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif
      RSPIN=WDES%RSPIN
!=======================================================================
! some initialization
!=======================================================================

      ION_SUM=0
      ION_SUM_DETAIL=0

      LMBASE =0
      NIS=1

      typ: DO NT=1,T_INFO%NTYP
      ion: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
      PP=>PP_POINTER(P, NI, NT)

!=======================================================================
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion ! not on local node

      LOW=1
      block: DO
      LL=PP%LPS(LOW)
      DO LHI=LOW,PP%LMAX
         IF (LL/=PP%LPS(LHI)) EXIT
      ENDDO
      LHI=LHI-1

      MMAX=2*LL+1

      DO L =LOW,LHI
      DO LP=LOW,LHI

      DO M=1,MMAX
      LM=LL*LL+M

      IF (ASSOCIATED (PP%QTOT) ) THEN

      DO ISP=1,WDES%ISPIN
      DO ISPINOR=0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1

      LMIND  =LMBASE +(L -LOW) *MMAX+M + ISPINOR *WDES%NPRO/2
      LMIND_ =LMBASE +(LP-LOW) *MMAX+M + ISPINOR_*WDES%NPRO/2
      II=ISP+ISPINOR_+2*ISPINOR

         DO NK=1,WDES%NKPTS
         DO NB=1,WDES%NBANDS
            CSUM(NB,NK,II)=  &
                 W%CPROJ(LMIND,NB,NK,ISP)*PP%QTOT(LP,L)*GCONJG(W%CPROJ(LMIND_,NB,NK,ISP))
         ENDDO
         ENDDO
      ENDDO
      ENDDO
      ENDDO
      ND=WDES%NBANDS*WDES%NKPTS

      IF (WDES%LNONCOLLINEAR) &
      CALL C_FLIP(CSUM,ND,ND,WDES%NCDIJ,.FALSE.)

      DO ISP=1,WDES%NCDIJ
      ISP_=MIN(ISP,WDES%ISPIN)
      DO NK=1 ,WDES%NKPTS
      CALL SETWDES(WDES,WDES_1K,NK)
      DO NB=1,WDES%NB_TOT
         NB_=NB_LOCAL(NB,WDES_1K)
         IF(NB_==0) CYCLE
         ION_SUM(LL+1,NI,ISP)=ION_SUM(LL+1,NI,ISP)+CSUM(NB_,NK,ISP)*RSPIN* &
              WDES%WTKPT(NK)*W%FERWE(NB_,NK,ISP_)
         ION_SUM_DETAIL(LM,NI,ISP)=ION_SUM_DETAIL(LM,NI,ISP)+CSUM(NB_,NK,ISP)*RSPIN* &
              WDES%WTKPT(NK)*W%FERWE(NB_,NK,ISP_)
      ENDDO
      ENDDO
      ENDDO
         
      ENDIF

      ENDDO
      ENDDO
      ENDDO

!-----------------------------------------------------------------------
      LMBASE =LMBASE +(LHI-LOW+1)*MMAX
      LOW=LHI+1
      IF (LOW > PP%LMAX) EXIT block
      ENDDO block

      ENDDO ion
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO typ

      CALLMPI( M_sum_d( WDES%COMM, ION_SUM, LDIMP*T_INFO%NIONS*WDES%NCDIJ))
      CALLMPI( M_sum_d( WDES%COMM, ION_SUM_DETAIL, LMDIMP*T_INFO%NIONS*WDES%NCDIJ))

      ND=LDIMP*T_INFO%NIONS
      IF (.NOT.WDES%LNONCOLLINEAR) &
           CALL R_FLIP(ION_SUM,ND,ND,WDES%NCDIJ,.FALSE.)
      io_begin
      IF (LWRITE) THEN
!=======================================================================
!   write to MAGCAR
!=======================================================================
      DUMMY=0
      DO ISP=1,WDES%NCDIJ

      DO NI=1,T_INFO%NIONS
      PARSUM=0
      DO NL=1,LDIMP
        PARSUM=PARSUM+ION_SUM(NL,NI,ISP)
      ENDDO
      MAG(NI,ISP)=PARSUM
      ENDDO
      ENDDO

      OPEN(UNIT=72,FILE=DIR_APP(1:DIR_LEN)//'MAGCAR',STATUS='UNKNOWN')

      DO IA=1,3
         WRITE(72,'(1X,3F12.6)' ) LATT_CUR%A(1:3,IA)
      ENDDO
      
      WRITE(72,'(I4)') T_INFO%NIONS
      DO NT=1,T_INFO%NTYP
         WRITE(72,'(I4)',ADVANCE='NO') T_INFO%NITYP(NT)
      ENDDO
      WRITE(72,'(/A12)') 'Comment line'
      DO NI=1,T_INFO%NIONS
         X=T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
            T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
             T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         Y=T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
            T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
             T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         Z=T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
            T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
             T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         IF (WDES%NCDIJ==2) &
            WRITE(72,'((1X,A2),3X,3(1X,F6.3),3X,3(1X,F6.3))') P(T_INFO%ITYP(NI))%ELEMENT,X,Y,Z,DUMMY,DUMMY,MAG(NI,2)
         IF (WDES%NCDIJ==4) &   
            WRITE(72,'((1X,A2),3X,3(1X,F6.3),3X,3(1X,F6.3))') P(T_INFO%ITYP(NI))%ELEMENT,X,Y,Z,MAG(NI,2:4)
      ENDDO
      CLOSE(72)
      ENDIF
      io_end

      RETURN
      END SUBROUTINE WR_MOMENTS


!************************ SUBROUTINE WR_PROJ_CHARG *********************
!
! Mind: WR_MOMENTS has to be called before WR_PROJ_CHARG, otherwise
!       the array ION_SUM_DETAIL is not filled with the right values
!
!***********************************************************************   

      SUBROUTINE WR_PROJ_CHARG(GRID,P,LATT_CUR,T_INFO,WDES)

      USE prec
      USE pseudo
      USE main_mpi
      USE mpimy
      USE mgrid
      USE lattice
      USE constant
      USE wave
      USE asa
      USE poscar
      USE us
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES

      DIMENSION NLIMAX(T_INFO%NIONS)
      REAL(q),ALLOCATABLE :: RPROJ(:,:)
! work arrays
      REAL(q),ALLOCATABLE :: DIST(:),XS(:),YS(:),ZS(:),VPS(:),YLM(:,:)

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME= WDES%COMM%NODE_ME
      IONODE = WDES%COMM%IONODE
#endif
      io_begin
      
      OPEN(UNIT=72,FILE=DIR_APP(1:DIR_LEN)//'MAGCAR',STATUS='UNKNOWN',POSITION='APPEND')

      LYDIM=MAXL(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs
!=======================================================================
! first get me IRMAX
!=======================================================================
      NIS=1

      type1: DO NT=1,T_INFO%NTYP
      IF (P(NT)%LMMAX==0) GOTO 600
      ions1: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      ARGSC=NPSRNL/P(NT)%PSRMAX

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
      F1=1._q/GRID%NGX
      F2=1._q/GRID%NGY
      F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= P(NT)%PSRMAX*LATT_CUR%BNORM(1)*GRID%NGX
      D2= P(NT)%PSRMAX*LATT_CUR%BNORM(2)*GRID%NGY
      D3= P(NT)%PSRMAX*LATT_CUR%BNORM(3)*GRID%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX
      
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      IND=1
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)
      ARG=(D*ARGSC)+1
      NADDR=INT(ARG)

      IF (NADDR<=NPSRNL) IND=IND+1
      
      ENDDO
      ENDDO
      ENDDO

      IRMAX=MAX(IRMAX,IND)
!=======================================================================
! end of loop over ions
!=======================================================================
      ENDDO ions1
  600 NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type1

! allocate projectors
      ALLOCATE(RPROJ(IRMAX,LMYDIM))
! and allocate work arrays
      ALLOCATE(DIST(IRMAX),XS(IRMAX),YS(IRMAX),ZS(IRMAX),VPS(IRMAX),YLM(IRMAX,LMYDIM))
      XS=0;YS=0;ZS=0;DIST=0;VPS=0;YLM=0
!=======================================================================
! now set up the projectors
! loop over all ions
!=======================================================================
      NIS=1

      type2: DO NT=1,T_INFO%NTYP
      IF (P(NT)%LMMAX==0) GOTO 601
      ions2: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      RPROJ=0      

      ARGSC=NPSRNL/P(NT)%PSRMAX

!=======================================================================
! find lattice points contained within the cutoff-sphere
! this loop might be done in scalar unit
!=======================================================================
      F1=1._q/GRID%NGX
      F2=1._q/GRID%NGY
      F3=1._q/GRID%NGZ

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= P(NT)%PSRMAX*LATT_CUR%BNORM(1)*GRID%NGX
      D2= P(NT)%PSRMAX*LATT_CUR%BNORM(2)*GRID%NGY
      D3= P(NT)%PSRMAX*LATT_CUR%BNORM(3)*GRID%NGZ

      N3LOW= INT(T_INFO%POSION(3,NI)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
      N2LOW= INT(T_INFO%POSION(2,NI)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
      N1LOW= INT(T_INFO%POSION(1,NI)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

      N3HI = INT(T_INFO%POSION(3,NI)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
      N2HI = INT(T_INFO%POSION(2,NI)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
      N1HI = INT(T_INFO%POSION(1,NI)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX      
!-----------------------------------------------------------------------
! loop over cubus around one ion
! conventional version x is fast index
!-----------------------------------------------------------------------
      IND=1
      DO N3=N3LOW,N3HI
      X3=(N3*F3-T_INFO%POSION(3,NI))

      DO N2=N2LOW,N2HI
      X2=(N2*F2-T_INFO%POSION(2,NI))

      DO N1=N1LOW,N1HI
      X1=(N1*F1-T_INFO%POSION(1,NI))

      XC= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      YC= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      ZC= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(XC*XC+YC*YC+ZC*ZC)
      ARG=(D*ARGSC)+1
      NADDR=INT(ARG)

      IF (NADDR<=NPSRNL) THEN

        IF (D<1E-4_q) THEN
          DIST(IND)=1E-4_q
        ELSE
          DIST(IND)=D
        ENDIF

        XS(IND)  =XC/DIST(IND)
        YS(IND)  =YC/DIST(IND)
        ZS(IND)  =ZC/DIST(IND)

        IND=IND+1
      ENDIF
      ENDDO
      ENDDO
      ENDDO
      INDMAX=IND-1
!=======================================================================
! now calculate the tables containing the spherical harmonics
! multiplied by the pseudopotential
!=======================================================================
      LYDIM=MAXL1(P(NT))

      CALL SETYLM(LYDIM,INDMAX,YLM,XS,YS,ZS)

      l_loop: DO L=1,P(NT)%LMAX
!-----------------------------------------------------------------------
! interpolate the non-local pseudopotentials
! and multiply by (LATT_CUR%OMEGA/4*PI)^(1/2)
! interpolation is done here using spline-fits this inproves the
! numerical stability of the forces the MIN operation takes care
! that the index is between  1 and NPSRNL
!-----------------------------------------------------------------------
        FAKT= SQRT(LATT_CUR%OMEGA)

!DIR$ IVDEP
!OCL NOVREC
        DO IND=1,INDMAX
          I  =MIN(INT(DIST(IND)*ARGSC)+1,NPSRNL-1)

          REM=DIST(IND)-P(NT)%PSPRNL(I,1,L)
          VPS(IND)=(P(NT)%PSPRNL(I,2,L)+REM*(P(NT)%PSPRNL(I,3,L)+ &
     &         REM*(P(NT)%PSPRNL(I,4,L)+REM*P(NT)%PSPRNL(I,5,L))))*FAKT
        ENDDO

        LL=P(NT)%LPS(L)
        MMAX=2*LL

        LMBASE=LL**2+1

        DO LM=0,MMAX
        DO IND=1,INDMAX
           RPROJ(IND,LMBASE+LM)=RPROJ(IND,LMBASE+LM)+VPS(IND)*YLM(IND,LM+LMBASE)
        ENDDO
        ENDDO

      ENDDO l_loop

!=======================================================================
! write projectors and projections (PROJ and ION_SUM_DETAIL) to MAGCAR
!=======================================================================
      LMYDIM=(LYDIM+1)**2
      SELECT CASE (LMYDIM)
      CASE (1)
      WRITE(72,'(3I3,I9)') NI,LMYDIM,WDES%NCDIJ,INDMAX
      DO ISP=1,WDES%NCDIJ
         WRITE(72,'(1X,F6.3)') ION_SUM_DETAIL(1:LMYDIM,NI,ISP)
      ENDDO
      DO IND=1,INDMAX
         X=XS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         Y=YS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         Z=ZS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         WRITE(72,'(3(1X,E12.5),2X,(1X,E9.3))') X,Y,Z,RPROJ(IND,1:LMYDIM)
      ENDDO
      CASE (4)
      WRITE(72,'(3I3,I9)') NI,LMYDIM,WDES%NCDIJ,INDMAX
      DO ISP=1,WDES%NCDIJ
         WRITE(72,'(4(1X,F6.3))') ION_SUM_DETAIL(1:LMYDIM,NI,ISP)
      ENDDO
      DO IND=1,INDMAX
         X=XS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         Y=YS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         Z=ZS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         WRITE(72,'(3(1X,E12.5),2X,(1X,E9.3),1X,3(1X,E9.3))') X,Y,Z,RPROJ(IND,1:LMYDIM)
      ENDDO
      CASE (9)
      WRITE(72,'(3I3,I9)') NI,LMYDIM,WDES%NCDIJ,INDMAX
      DO ISP=1,WDES%NCDIJ
         WRITE(72,'(9(1X,F6.3))') ION_SUM_DETAIL(1:LMYDIM,NI,ISP)
      ENDDO
      DO IND=1,INDMAX
         X=XS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         Y=YS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         Z=ZS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         WRITE(72,'(3(1X,E12.5),2X,(1X,E9.3),1X,3(1X,E9.3),1X,5(1X,E9.3))') X,Y,Z,RPROJ(IND,1:LMYDIM)
!          IF (NI==1) WRITE(72,'(3(1X,E12.5))') X,Y,Z
      ENDDO
!      DO IND=1,INDMAX
!         IF (NI==1) WRITE(72,'((1X,E9.3))') RPROJ(IND,7)
!      ENDDO
      CASE (16)
      WRITE(72,'(3I3,I9)') NI,LMYDIM,WDES%NCDIJ,INDMAX
      DO ISP=1,WDES%NCDIJ
         FSUM=SUM(ION_SUM_DETAIL(10:16,NI,ISP))
         WRITE(72,'(10(1X,F6.3))') ION_SUM_DETAIL(1:9,NI,ISP),FSUM
      ENDDO
      DO IND=1,INDMAX
         X=XS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(1,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(1,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(1,3)
         Y=YS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(2,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(2,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(2,3)
         Z=ZS(IND)*DIST(IND)+T_INFO%POSION(1,NI)*LATT_CUR%A(3,1)+ &
                              T_INFO%POSION(2,NI)*LATT_CUR%A(3,2)+ &
                               T_INFO%POSION(3,NI)*LATT_CUR%A(3,3)
         FSUM=SUM(RPROJ(IND,10:16))
         WRITE(72,'(3(1X,E12.5),2X,10(1X,E9.3))') X,Y,Z,RPROJ(IND,1:9),FSUM
      ENDDO
      END SELECT

!=======================================================================
! end of loop over ions
!=======================================================================
      ENDDO ions2
  601 NIS = NIS+T_INFO%NITYP(NT)
      ENDDO type2

      DEALLOCATE(DIST,XS,YS,ZS,VPS,YLM,RPROJ)
      CLOSE(72)
      io_end
      RETURN
      END SUBROUTINE WR_PROJ_CHARG

      
      END MODULE writer
