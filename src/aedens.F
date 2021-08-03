#include "symbol.inc"
      MODULE aedens

      USE prec

      REAL(q), SAVE,PRIVATE  :: ENCUTAE

      LOGICAL, SAVE,PRIVATE  :: LAECHG
      
  INTERFACE
      SUBROUTINE AUGCHG(WDES, GRID_SOFT,GRIDC_,GRIDUS,C_TO_US, &
        LATT_CUR,P,T_INFO,SYMM, LOVERL, SOFT_TO_C,&
        LMDIM,CRHODE, CHTOT_,CHDEN, IRDMAX ,LPSEUDO, LCORE)
      USE prec
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d)     GRID_SOFT,GRIDC_,GRIDUS
      TYPE (transit)     C_TO_US
      TYPE (transit)     SOFT_TO_C
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES
      TYPE (symmetry)    SYMM

      INTEGER IRDMAX,ISP      ! allocation required for augmentation
      OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      COMPLEX(q) CHTOT_(GRIDC_%MPLWV,WDES%NCDIJ)
      COMPLEX(q) CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      LOGICAL LOVERL,LPSEUDO,LCORE
    END SUBROUTINE

  END INTERFACE

      CONTAINS

      SUBROUTINE INIT_AEDENS(IU0,IU5)

      USE prec
      USE base
      USE vaspxml
      
      INTEGER IU0,IU5

      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      CHARACTER(1) CHARAC
      CHARACTER(255) INPLIN
      LOGICAL LOPEN,LDUM
      
! read relevant stuff from INCAR 
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
      LAECHG=.FALSE.     
      CALL RDATAB(LOPEN,INCAR,IU5,'LAECHG','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LAECHG,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LAECHG'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LAECHG','L',IDUM,RDUM,CDUM,LAECHG,CHARAC,N)      

      ENCUTAE=-1     
      CALL RDATAB(LOPEN,INCAR,IU5,'ENCUTAE','=','#',';','F', &
     &            IDUM,ENCUTAE,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''ENCUTAE'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('ENCUTAE','F',IDUM,ENCUTAE,CDUM,LDUM,CHARAC,N)      

      CLOSE(IU5)
      RETURN

  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP
      
      END SUBROUTINE INIT_AEDENS
      
      FUNCTION LWRT_AECHG()
      IMPLICIT NONE
      LOGICAL LWRT_AECHG
      LWRT_AECHG=LAECHG
      END FUNCTION LWRT_AECHG

      FUNCTION LWRTSTRF()
      IMPLICIT NONE
      LOGICAL LWRTSTRF
      LWRTSTRF=(ENCUTAE>0._q)
      END FUNCTION LWRTSTRF


!*******************************************************************
!
!*******************************************************************

      SUBROUTINE WRTSTRF(GRIDC,LATT_CUR,CHTOT,IU)
      USE mgrid
      USE mpimy
      USE lattice
      USE constant
      IMPLICIT NONE
      TYPE (grid_3d) GRIDC
      TYPE (latt) LATT_CUR
      COMPLEX(q) CHTOT(GRIDC%RC%NP)
      INTEGER IU
      ! local variables
      INTEGER NALLOC,ISTAT
      INTEGER NODE_ME,IONODE
      INTEGER N3,N2,N1,NI,NC
      REAL(q) GX,GY,GZ,GSQU
      COMPLEX(q), ALLOCATABLE :: CWORK(:)
      
      NALLOC=GRIDC%NGY_rd*GRIDC%NGZ_rd
      ALLOCATE(CWORK(NALLOC),STAT=ISTAT)
      IF (ISTAT>0) RETURN ! can not write the charge immediate exit

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
#endif
      
      do_io WRITE(IU,'(1X,3F12.6)') LATT_CUR%B*TPI

! test      
!     NI=0
!     CHTOT=0
!     col: DO NC=1,GRIDC%RC%NCOL
!     N2= GRIDC%RC%I2(NC)
!     N3= GRIDC%RC%I3(NC)
!     row: DO N1=1,GRIDC%RC%NROW
!        NI=NI+1
!        IF ((GRIDC%LPCTX(N1)==1).AND.(GRIDC%LPCTY(N2)<0).AND.(GRIDC%LPCTZ(N3)>4)) THEN
!           CHTOT(NI)=(0.5_q,0.25_q)
!        ENDIF
!     ENDDO row
!     ENDDO col      
! test
      
      DO N1=1,GRIDC%RC%NROW
         CALL MRG_GRID_RC_PLANE(GRIDC,CWORK,CHTOT,N1)
         io_begin
         NI=0
         DO N2=1,GRIDC%NGY_rd
         DO N3=1,GRIDC%NGZ_rd
            NI=NI+1
            GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))*TPI
            GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))*TPI
            GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))*TPI
            GSQU=GX**2+GY**2+GZ**2
            IF (HSQDTM*GSQU<=ENCUTAE) THEN
               WRITE(IU,'(3I6,3X,2G15.5)') GRIDC%LPCTX(N1),GRIDC%LPCTY(N2),GRIDC%LPCTZ(N3),CWORK(NI)
            ENDIF 
         ENDDO
         ENDDO
         io_end
      ENDDO
      
      DEALLOCATE(CWORK)
      
      RETURN
      END SUBROUTINE WRTSTRF
      
      
!*******************************************************************
!
!*******************************************************************

      SUBROUTINE WRT_RHO_RAD(WDES, P, T_INFO, LOVERL, LMDIM, CRHODE, IU)

      USE pseudo
      USE poscar
      USE wave
      USE constant
      USE paw

      IMPLICIT NONE

      TYPE (wavedes) WDES
      TYPE (type_info) T_INFO
      TYPE (potcar), TARGET :: P(T_INFO%NTYP)
      INTEGER LMDIM,IU

      OVERLAP CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      LOGICAL LOVERL

    ! local variables
      TYPE (potcar), POINTER :: PP
      
      REAL(q), ALLOCATABLE :: RHOLM(:,:)
      REAL(q), ALLOCATABLE :: RHO(:,:,:),RTMP(:,:)
      
      INTEGER NT,NI,NIP,ISP,I,J
      INTEGER NELEMENTS,LYMAX,LMMAX
      INTEGER, EXTERNAL :: MAXL_AUG
      INTEGER NODE_ME, IONODE

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif

!=======================================================================
! quick return if possible
!=======================================================================
      IF (.NOT.LOVERL .OR. MIMIC_US ) RETURN

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)
      LMMAX=(LYMAX+1)**2

      ALLOCATE(RHOLM(LMMAX*LMMAX,WDES%NCDIJ))
!=======================================================================
! cycle all ions and collect the required occupancies
!=======================================================================
      ion: DO NI=1,T_INFO%NIONS
         NT=T_INFO%ITYP(NI)
         PP=> P(NT)

         NELEMENTS =0
         RHOLM=0
         IF (DO_LOCAL(NI)) THEN
            NIP=NI_LOCAL(NI,WDES%COMM_INB)
            DO ISP=1,WDES%NCDIJ
               CALL TRANS_RHOLM(CRHODE(:,:,NIP,ISP),RHOLM(:,ISP),PP)
            ENDDO
            NELEMENTS=LMMAX*LMMAX*WDES%NCDIJ
         ENDIF
         ! transfer a "local" CRHODE to all nodes, i.e.,
         ! also to the node responsible for writing to files
         CALLMPI( M_sum_i(WDES%COMM, NELEMENTS, 1))
         CALLMPI( M_sum_d(WDES%COMM, RHOLM, NELEMENTS))

         io_begin
         ! calculate the radial densities
         ALLOCATE(RHO(PP%R%NMAX,LMMAX,WDES%NCDIJ),RTMP(PP%R%NMAX,LMMAX))
         RHO=0   
         DO ISP=1,WDES%NCDIJ
            ! calculate the AE density
            RTMP=0
            CALL RAD_CHARGE(RTMP(:,:),PP%R,RHOLM(:,ISP),PP%LMAX,PP%LPS,PP%WAE)
            RHO(:,:,ISP)=RTMP(:,:)
            ! calculate the PS density
            RTMP=0
            CALL RAD_CHARGE(RTMP(:,:),PP%R,RHOLM(:,ISP),PP%LMAX,PP%LPS,PP%WPS)
            ! add the compensation charge density
            CALL RAD_AUG_CHARGE(RTMP(:,:),PP%R,RHOLM(:,ISP),PP%LMAX,PP%LPS, &
     &                     LYMAX,PP%AUG,PP%QPAW)
            ! AE minus PS density
!           RTMP=0 ! uncomment this line to get the AE density            
            RHO(:,:,ISP)=RHO(:,:,ISP)-RTMP(:,:)
         ENDDO
         ! write atomic coordinates to file
         ! write radial densities to file
         WRITE(IU,'(2I4)') NI,PP%R%NMAX
         DO I=1,PP%R%NMAX
            WRITE(IU,FMT='(F15.8)',ADVANCE='NO') PP%R%R(I)
            WRITE(IU,FMT='(28F15.8)') (RHO(I,J,1), J=1,LMMAX)
         ENDDO        
         DEALLOCATE(RHO,RTMP)
         io_end

      ENDDO ion

      DEALLOCATE(RHOLM)

      RETURN
      END SUBROUTINE WRT_RHO_RAD


      
      END MODULE aedens
      
!************************ SUBROUTINE AUGCHG  ***************************
!
! This subroutine is used to construct all-electron densities in
! real space.
!
! If LCORE=.TRUE.: AUGCHG adds the all-electron frozen core charge
! density minus the partial core charge (nc_ae - nc_ps) in the PAW
! spheres to CHTOT, in real space.
! On entry CHTOT should contain the partial core charge on the plane
! wave grid. On exit CHTOT will contain the AE core charge density in
! reciprocal space.
!
! If LPSEUDO=.TRUE.: AUGCHG adds the all-electron one-center charge
! density minus the pseudo one-center charge density in the PAW
! spheres to the pseudo charge density on the plane wave grid (CHDEN).
! The result is stored in CHTOT, in reciprocal space.
!
! If LPSEUDO= .FALSE.: AUGCHG adds the all-electron one-center charge
! density minus the pseudo one-center charge density and minus the
! augmentation charge density in the PAW spheres to the charge density
! contained in CHTOT. This mode is only used to construct the
! all-electron charge density of overlapping atomic charge densities.
!
!***********************************************************************

      SUBROUTINE AUGCHG(WDES, GRID_SOFT,GRIDC_,GRIDUS,C_TO_US, &
        LATT_CUR,P,T_INFO,SYMM, LOVERL, SOFT_TO_C,&
        LMDIM,CRHODE, CHTOT_,CHDEN, IRDMAX, LPSEUDO,LCORE)
      USE prec
      USE base
      USE charge
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE constant
      USE us

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET :: GRID_SOFT,GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER :: GRIDC
      TYPE (transit)     C_TO_US
      TYPE (transit)     SOFT_TO_C
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES
      TYPE (symmetry)    SYMM

      INTEGER   LDEP_INDEX
      INTEGER   IRDMAX,ISP      ! allocation required for augmentation
      OVERLAP   CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      RGRID,TARGET  :: CHTOT_(DIMREAL(GRIDC_%MPLWV),WDES%NCDIJ)
      RGRID,POINTER :: CHTOT(:)
      COMPLEX(q) CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      LOGICAL   LOVERL,LADDITIONAL,LPSEUDO,LCORE
!  work arrays
      REAL(q)   RHOLM(256)
      REAL(q),ALLOCATABLE ::   DIST(:),DEP(:),QIJ(:),SUM(:),YLM(:,:)
      INTEGER,ALLOCATABLE ::   NLI(:)
      LOGICAL L_SYM
!-MM- spin spiral stuff
      REAL(q)   QVEC(3),QR
      REAL(q),ALLOCATABLE :: XS(:),YS(:),ZS(:)
      REAL(q)   RHOLMX(256),RHOLMY(256)
!-MM- end of addition
#ifdef MPI
      ! in the MPI version CRHODE holds the contribution to the augmentation
      ! occupation only for ions and bands which are local
      ! CTMP holds all elements (merged)
      ! to achive good load balancing CTMP is first merged, and then each node
      ! calculates augmentation charges for his columns
      OVERLAP,ALLOCATABLE :: CTMP(:,:,:,:)
      ALLOCATE(CTMP(LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
#else
#define CTMP CRHODE
#endif
      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)

      ALLOCATE(XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX))
!=======================================================================
! if no overlap copy CHDEN to CHTOT and that s it
!=======================================================================
      overl: IF (.NOT.LOVERL) THEN
         DO ISP=1,WDES%NCDIJ
            CALL RC_ADD(CHDEN(1,ISP),1.0_q,CHDEN(1,ISP),0.0_q,CHTOT_(1,ISP),GRID_SOFT)
         ENDDO
      ELSE overl

! find the maximum L for augmentation charge (usually just 2 l)
      LYDIM=MAXL_AUG(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE( &
     &          DIST(IRDMAX),DEP(IRDMAX),SUM(IRDMAX),YLM(IRDMAX,LMYDIM), &
     &          NLI(IRDMAX),QIJ(IRDMAX))

      IF (LADDITIONAL) THEN
         ALLOCATE(CHTOT(DIMREAL(GRIDUS%MPLWV)))
      ENDIF
#ifdef MPI
!=======================================================================
! merge CRHODE from all nodes
! for simplicity I do this with M_sum_d but there are of course better
! ways to do this
!=======================================================================
      CTMP=0
      DO ISP=1,WDES%NCDIJ
         DO NI=1,T_INFO%NIONS
            NIP=NI_LOCAL(NI, WDES%COMM_INB)
            IF (NIP/=0) THEN
               CTMP(:,:,NI,ISP)=CRHODE(:,:,NIP,ISP)
            ENDIF
         ENDDO
      ENDDO
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INB,CTMP,LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_d(WDES%COMM_INB,CTMP,LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ*2))
#endif
#endif
!-----------------------------------------------------------------------
! do  symmetrization of the CRHODE
! (in MPI version this is the only position where I can do that
!  without additional communication)
!-----------------------------------------------------------------------
! if PAW is selected, do symmetrization in any case
      L_SYM=.FALSE.
      DO NT=1,T_INFO%NTYP
        IF ( ASSOCIATED(P(NT)%QPAW) ) L_SYM=.TRUE.
      ENDDO
! no symmetry used, well switch it off
      IF (SYMM%ISYM<=0) L_SYM=.FALSE.
! CHDEN is symmetrized and not CHTOT, in that case do symmetrization in any case
      IF (SYMM%ISYM==2) L_SYM=.TRUE.
! now do the symmetrization
      IF (L_SYM) THEN
         IF (WDES%LNONCOLLINEAR) THEN
#ifdef MPI
           CALL AUGSYM_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP, &
                CTMP(1,1,1,1), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), LATT_CUR%A, LATT_CUR%B, 1)
! Marsman insert symmetrization here
! Symmetrize the vectors (DX,DY,DZ)
           IF (.NOT.WDES%LSPIRAL) &
          &   CALL AUGSYM_NONCOL_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP, &
                CTMP(1,1,1,2), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), WDES%SAXIS, LATT_CUR%A, LATT_CUR%B)
! store result back to original storage position
           DO NI=1,T_INFO%NIONS
              NIP=NI_LOCAL(NI, WDES%COMM_INB)
              IF (NIP/=0) THEN
                 CRHODE(:,:,NIP,:)=CTMP(:,:,NI,:)
              ENDIF
           ENDDO
#else
           CALL AUGSYM_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP(1), &
              CRHODE(1,1,1,1), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), LATT_CUR%A, LATT_CUR%B, 1)
! Marsman insert symmetrization here
! Symmetrize the vectors (DX,DY,DZ)
           IF (.NOT.WDES%LSPIRAL) &
          &   CALL AUGSYM_NONCOL_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP, &
                CTMP(1,1,1,2), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), WDES%SAXIS, LATT_CUR%A, LATT_CUR%B)
#endif
         ELSE
         DO ISP=1,WDES%NCDIJ
#ifdef MPI
           CALL AUGSYM_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP, &
                CTMP(1,1,1,ISP), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), LATT_CUR%A, LATT_CUR%B, ISP)
        ! store result back to original storage position
           DO NI=1,T_INFO%NIONS
              NIP=NI_LOCAL(NI, WDES%COMM_INB)
              IF (NIP/=0) THEN
                 CRHODE(:,:,NIP,ISP)=CTMP(:,:,NI,ISP)
              ENDIF
           ENDDO
#else
           CALL AUGSYM_(P,LMDIM,T_INFO%NIONS,T_INFO%NIOND,T_INFO%NTYP,T_INFO%NITYP(1), &
              CRHODE(1,1,1,ISP), SYMM%ROTMAP(1,1,1), SYMM%MAGROT(1,1), LATT_CUR%A, LATT_CUR%B, ISP)
#endif
         ENDDO
        ENDIF
      ENDIF
!=======================================================================
! now the actual work starts
!=======================================================================
      spin:DO ISP=1,WDES%NCDIJ
      IF (LADDITIONAL) THEN
         GRIDC => GRIDUS
      ELSE
         CHTOT => CHTOT_(:,ISP)
         GRIDC => GRIDC_
      ENDIF
!=======================================================================
! loop over all ions
!=======================================================================
      IF (LPSEUDO) CHTOT=0
      ion: DO NI=1,T_INFO%NIONS
      NT=T_INFO%ITYP(NI)
!-----------------------------------------------------------------------
! for this ion (this type of ion) no depletion charge
!-----------------------------------------------------------------------
      IF (P(NT)%PSDMAX==0) CYCLE
!-----------------------------------------------------------------------
! calulate the spherical harmonics (DEP is Work-arrays)
!-----------------------------------------------------------------------
      DISX=0
      DISY=0
      DISZ=0

      LYMAX=MAXL1(P(NT))
      IF ( ASSOCIATED(P(NT)%QPAW) ) THEN
         LYMAX=MIN(4,LYMAX*2)
      ENDIF
      CALL SETYLM_AUG(GRIDC,LATT_CUR,T_INFO%POSION(1,NI),P(NT)%PSDMAX,NPSRNL, &
     &        LMYDIM,LYMAX,YLM(1,1),IRDMAX,INDMAX, &
     &        DISX,DISY,DISZ,DIST(1),NLI(1),XS,YS,ZS)

      SUM=0
!=======================================================================
! include only core charge density
!=======================================================================
      IF (LCORE) THEN
         SUM=0
         CALL GETQIJ(0,0,P(NT),INDMAX,DIST(1:INDMAX),QIJ(1:INDMAX),.TRUE.)
         DO IND=1,INDMAX
            SUM(IND)=SUM(IND)+QIJ(IND)*LATT_CUR%OMEGA/(2*SQRT(PI))
         ENDDO
      ELSE
!=======================================================================
! loop over pseudopotential indexes L and LP
!=======================================================================
       ! loop over all channels (l,epsilon)
         LDEP_INDEX=1
         LM=1
         l_loop:  DO L =1,P(NT)%LMAX
         LMP=LM
         lp_loop: DO LP=L,P(NT)%LMAX
         CALL GETQIJ(L,LP,P(NT),INDMAX,DIST(1:INDMAX),QIJ(1:INDMAX),.FALSE.)
         QIJ=QIJ*LATT_CUR%OMEGA

       ! quantum numbers l and lp of these two channels
         LL =P(NT)%LPS(L )
         LLP=P(NT)%LPS(LP)

       ! loop over all m mp
         m_loop:  DO M=1,2*LL+1
         MPLOW=1
         IF (L==LP) MPLOW=M
         mp_loop: DO MP=MPLOW,2*LLP+1
            FAKT=1
            IF (LMP+MP/=LM+M) FAKT=2

         !   calculate the indexes into the array containing the spherical
         !   harmonics
            INDYLM =LL **2  +M
            INDPYL =LLP**2  +MP

            TFAKT=CTMP(LM+M-1,LMP+MP-1,NI,ISP)*FAKT

         !   add augmentation charge (augmentation charge is real)
!DIR$ IVDEP
!OCL NOVREC
            DO IND=1,INDMAX
               SUM(IND)=SUM(IND)+QIJ(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)*TFAKT
            ENDDO

         ENDDO mp_loop
         ENDDO m_loop
         LMP=LMP+2*LLP+1
         ENDDO lp_loop
         LM =LM +2*LL +1
         ENDDO l_loop
!========================================================================
! subtract compensation density
! only necessary in case we want to calculate the AE charge arising
! from overlapping atomic charge densities, since the compensation
! density was already added on the POTCAR file
!========================================================================      
         IF (.NOT.LPSEUDO) THEN
            CALL CALC_RHOLM( LYMAX, CTMP(:,:,NI,ISP) , RHOLM, P(NT))
            DO L =0,LYMAX
            ! WRITE(0,'("RHOLM",I2,10F10.6)') L,(RHOLM(L**2+M),M=1,(L*2)+1)
            CALL SETDEP(P(NT)%QDEP(1,1,L),P(NT)%PSDMAX,NPSRNL, &
                 LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
            DO M=1,(L*2)+1
               INDYLM =L **2  +M
               TFAKT=RHOLM(INDYLM)
!DIR$ IVDEP
!OCL NOVREC
               DO IND=1,INDMAX
                  SUM(IND)=SUM(IND)-DEP(IND)*YLM(IND,INDYLM)*TFAKT
               ENDDO
            ENDDO
            ENDDO
         ENDIF
      ENDIF
!=======================================================================
! add the calculated augmentation charge to the total charge
!=======================================================================
      SUMN=0
      DO IND=1,INDMAX
        CHTOT(NLI(IND))=CHTOT(NLI(IND))+SUM(IND)
        SUMN=SUMN+SUM(IND)
      ENDDO
!-----------------------------------------------------------------------
      ENDDO ion
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
! transform the charge-density to reciprocal space
! and add valenz-charge-density
!-----------------------------------------------------------------------
      CALL FFT_RC_SCALE(CHTOT(1),CHTOT(1),GRIDC)
      IF (LADDITIONAL) CALL CP_GRID(GRIDUS,GRIDC_,C_TO_US,CHTOT(1),CHTOT_(1,ISP))
#define debug
#ifdef debug
      RHO_AUG =RHO0(GRIDC_, CHTOT_(1,ISP))
      RHO_SOFT=RHO0(GRID_SOFT , CHDEN)
      WRITE(*,*) "augmentation electrons", RHO_AUG
      WRITE(*,*) "soft         electrons", RHO_SOFT
#endif

      CALL ADD_GRID(GRIDC_,GRID_SOFT,SOFT_TO_C,CHDEN(1,ISP),CHTOT_(1,ISP))

      CALL SETUNB_COMPAT(CHTOT_(1,ISP),GRIDC_)
#ifdef debug
      RHO_AUG =RHO0(GRIDC_, CHTOT_(1,ISP))
      WRITE(*,*) "total        electrons", RHO_AUG
#endif
      ENDDO spin

      IF (LADDITIONAL) DEALLOCATE(CHTOT)
      DEALLOCATE(DIST,DEP,SUM,YLM,NLI,QIJ,XS,YS,ZS)

      ENDIF overl
#ifdef MPI
      DEALLOCATE(CTMP)
#else
#undef CTMP
#endif

      RETURN
      END SUBROUTINE


      SUBROUTINE GETQIJ(L,LP,P,INDMAX,DIST,QIJ,LCORE)
      
      USE prec
      USE pseudo
      
      IMPLICIT NONE
      
      TYPE (potcar) P
      
      INTEGER L,LP
      INTEGER INDMAX
      
      REAL(q), DIMENSION(INDMAX) :: DIST
      REAL(q), DIMENSION(INDMAX) :: QIJ 
      
      LOGICAL LCORE
      
      INTEGER I,N,LO,HI
      REAL(q) X(P%R%NMAX),Y(P%R%NMAX)
      REAL(q) DY
      
      IF (LCORE) THEN
         DO I=1,P%R%NMAX
            X(I)=P%R%R(I)
            Y(I)=(P%RHOAE(I)-P%RHOPS(I))/P%R%R(I)/P%R%R(I)
         ENDDO
      ELSE
         DO I=1,P%R%NMAX
            X(I)=P%R%R(I)
            Y(I)=(P%WAE(I,L)*P%WAE(I,LP)-P%WPS(I,L)*P%WPS(I,LP))/P%R%R(I)/P%R%R(I)
!           WRITE(100,'(2F20.8)') X(I),Y(I)
         ENDDO
      ENDIF
            
      QIJ=0
      DO I=1,INDMAX
         N=1+INT(LOG(DIST(I)/P%R%RSTART)/P%R%H)
         IF (N<=1) QIJ(I)=Y(1)
         IF (N==2) THEN
            LO=1; HI=4
            CALL POLINT(X(LO:HI),Y(LO:HI),4,DIST(I),QIJ(I),DY)
         ENDIF
         IF (N>2.AND.(N+2)<=P%R%NMAX) THEN
            LO=N-1; HI=N+2
            CALL POLINT(X(LO:HI),Y(LO:HI),4,DIST(I),QIJ(I),DY)
         ENDIF
      ENDDO
      
      RETURN
      END SUBROUTINE
      

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
      
      
!     END MODULE aedens
