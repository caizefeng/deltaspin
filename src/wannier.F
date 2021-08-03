!#define dotiming
#define debug
#include "symbol.inc"

      MODULE WANNIER
      USE prec

      LOGICAL, PRIVATE, SAVE :: LWANNIER_,LROTATE
      
      CONTAINS

!************************ SUBROUTINE WANNIER_READER *********************
!
!************************************************************************

      SUBROUTINE WANNIER_READER(IU0,IU5,IU6,IDIOT)

      USE vaspxml
      USE base
      IMPLICIT NONE

      INTEGER IU0,IU5,IU6,IDIOT
      INTEGER IDUM,N,IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC
      
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

      LWANNIER_=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LWANNIER','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LWANNIER_,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LWANNIER'' from file INCAR.'
         GOTO 150
      ENDIF
#ifndef gammareal
      IF (LWANNIER_) THEN     
         CALL VTUTOR('S','NOWANNIER',RDUM,1,IDUM,1,CDUM,1,LDUM,1,IU6,IDIOT)
      ENDIF
#else      
      CALL XML_INCAR('LWANNIER','L',IDUM,RDUM,CDUM,LWANNIER_,CHARAC,N)

      LROTATE=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LOCALIZE','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LROTATE,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LOCALIZE'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('LOCALIZE','L',IDUM,RDUM,CDUM,LROTATE,CHARAC,N)
#endif
      CLOSE(IU5)      
      
      RETURN

  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
      STOP
                      
      END SUBROUTINE WANNIER_READER


!***********************************************************************
!
! function to query whether we want calculate the maximally localized
! wannier functions
!
!***********************************************************************    

      FUNCTION  LWANNIER()
      IMPLICIT NONE
      LOGICAL LWANNIER

      IF (LWANNIER_) THEN
         LWANNIER=.TRUE.
      ELSE
         LWANNIER=.FALSE.
      ENDIF

      END FUNCTION LWANNIER


!************************ SUBROUTINE SET_DIJ_WANNIER ********************
!************************************************************************

      SUBROUTINE SET_DIJ_WANNIER(G,WDES,GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P, &
     &  T_INFO,LOVERL,LMDIM,d_IJ,IRDMAX,CVALUE)
     
      USE prec
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE asa
      USE paw
      USE us
      USE elpol
      USE constant
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER ::  GRIDC
      TYPE (transit)             C_TO_US  ! index table between GRIDC and GRIDUS
      TYPE (latt)                LATT_CUR
      TYPE (wavedes)             WDES

      INTEGER                    G(3)
      GDEF                       d_IJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      
      INTEGER                    IRDMAX   ! allocation required for augmentation
      REAL(q)                    GX,GY,GZ,POS_X,POS_Y,POS_Z,GDR
      GDEF                       CSUM,d_LM(256)
      COMPLEX(q)                 CVALUE
      LOGICAL                    LOVERL,LADDITIONAL

      REAL(q), ALLOCATABLE ::    XS(:),YS(:),ZS(:)
      REAL(q), ALLOCATABLE ::    DIST(:),DEP(:),YLM(:,:)
      TYPE (potcar), POINTER :: PP

#ifdef MPI
      GDEF , ALLOCATABLE ::      GTMP(:,:,:,:)
      ALLOCATE(GTMP(LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
#else
# define GTMP d_IJ
#endif

      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)
      IF (LADDITIONAL) THEN
         GRIDC => GRIDUS
      ELSE
         GRIDC => GRIDC_
      ENDIF

!=======================================================================
! express G in cartesian coordinates
!=======================================================================

      GX=LATT_CUR%B(1,1)*G(1)+LATT_CUR%B(1,2)*G(2)+LATT_CUR%B(1,3)*G(3)
      GY=LATT_CUR%B(2,1)*G(1)+LATT_CUR%B(2,2)*G(2)+LATT_CUR%B(2,3)*G(3)
      GZ=LATT_CUR%B(3,1)*G(1)+LATT_CUR%B(3,2)*G(2)+LATT_CUR%B(3,3)*G(3)

!=======================================================================
      GTMP  =0
!=======================================================================

      IF (.NOT.LOVERL) GOTO 2000

      LYDIM=MAXL_AUG(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE(XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX), &
     &  DIST(IRDMAX),DEP(IRDMAX),YLM(IRDMAX,LMYDIM))
!=======================================================================
! loop over all ions
!=======================================================================
      RINPL=1._q/GRIDC%NPLWV

      ion: DO NI=1,T_INFO%NIONS
      
      ! express POSION in cartesian coordinates
      POS_X=LATT_CUR%A(1,1)*T_INFO%POSION(1,NI)+ &
     &       LATT_CUR%A(1,2)*T_INFO%POSION(2,NI)+ &
     &        LATT_CUR%A(1,3)*T_INFO%POSION(3,NI)
      POS_Y=LATT_CUR%A(2,1)*T_INFO%POSION(1,NI)+ &
     &       LATT_CUR%A(2,2)*T_INFO%POSION(2,NI)+ &
     &        LATT_CUR%A(2,3)*T_INFO%POSION(3,NI)
      POS_Z=LATT_CUR%A(3,1)*T_INFO%POSION(1,NI)+ &
     &       LATT_CUR%A(3,2)*T_INFO%POSION(2,NI)+ &
     &        LATT_CUR%A(3,3)*T_INFO%POSION(3,NI)

      NT=T_INFO%ITYP(NI)

      PP=>PP_POINTER(P, NI, NT)
      
      ! for this ion (this type of ion) no depletion charge
      IF (PP%PSDMAX==0) CYCLE
!-----------------------------------------------------------------------
! calculate the spherical harmonics YLM and the distance DIST  between
! grid-points and central atom (DEP is a work-arrays)
!-----------------------------------------------------------------------
      LYMAX=MAXL1(PP)
      IF ( ASSOCIATED(PP%QPAW) ) THEN
      ! in paw method we truncate the augmentation charge at L=2
         LYMAX=MIN(4,LYMAX*2)
      ENDIF
      CALL SETYLM_AUG2(GRIDC,LATT_CUR,T_INFO%POSION(1:3,NI), &
     &  PP%PSDMAX,NPSRNL,LMYDIM,LYMAX,YLM(1,1),IRDMAX,INDMAX, &
     &  0._q,0._q,0._q,XS(1),YS(1),ZS(1),DIST(1))

      IF (WDES%LNONCOLLINEAR) THEN
         ISP_INC=3
      ELSE
         ISP_INC=1
      ENDIF
 spin:DO ISP=1,WDES%NCDIJ,ISP_INC  

 lpaw:IF ( .NOT. ASSOCIATED(PP%QPAW) ) THEN
!=======================================================================
! US-PP
!=======================================================================
      LDEP_INDEX=1

    ! loop over all channels (l,epsilon)
      LM =1
      l_loop:  DO L =1,PP%LMAX
      LMP=LM
      lp_loop: DO LP=L,PP%LMAX
      IF (PP%NDEP(L,LP)==0) GOTO 510

      CALL SETDEP(PP%QDEP(1,1,LDEP_INDEX),PP%PSDMAX,NPSRNL, &
     &  LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
      LDEP_INDEX=LDEP_INDEX+ABS(PP%NDEP(L,LP))

    ! quantum numbers l and lp of these two channels
      LL =PP%LPS(L )
      LLP=PP%LPS(LP)

    ! loop over all m mp
      m_loop:  DO M=1,2*LL+1
      MPLOW=1 ; IF (L==LP) MPLOW=M
      mp_loop: DO MP=MPLOW,2*LLP+1

      !   calculate the indices into the array containing the spherical
      !   harmonics
         INDYLM =LL**2   +M
         INDPYL =LLP**2  +MP

         CSUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC

         DO IND=1,INDMAX
            GDR=GX*(XS(IND)+POS_X)+GY*(YS(IND)+POS_Y)+GZ*(ZS(IND)+POS_Z)
            CSUM=CSUM+CVALUE*EXP(CITPI*GDR)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
         ENDDO

         !   add to array d_IJ and make symmetric
         GTMP(LM+M-1,LMP+MP-1,NI,ISP)=CSUM*RINPL
         GTMP(LMP+MP-1,LM+M-1,NI,ISP)=CSUM*RINPL
!         WRITE(0,90) LL,M,LLP,MP,d_IJ(1:3,LM+M-1,LMP+MP-1,NI),TSUM*RINPL,PP%QION(L,LP)
! 90      FORMAT('l =',I2,'  m =',I2,'  lp =',I2,'  mp =',I2,'  d = (',3F12.5,' )',2F12.5)       
      ENDDO mp_loop
      ENDDO m_loop

 510  LMP=LMP+2*LLP+1
      ENDDO lp_loop
      LM =LM +2*LL +1
      ENDDO l_loop
   ELSE lpaw
!=======================================================================
! PAW approach
!=======================================================================
      d_LM=0
      DO L=0,LYMAX
         CALL SETDEP(PP%QDEP(1,1,L),PP%PSDMAX,NPSRNL, &
        &        LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
         
         DO M=1,(L*2)+1
            INDYLM=L*L+M
            CSUM=0
            ! sum over all r
!DIR$ IVDEP
!OCL NOVREC
            DO IND=1,INDMAX
               GDR=GX*(XS(IND)+POS_X)+GY*(YS(IND)+POS_Y)+GZ*(ZS(IND)+POS_Z)
               CSUM=CSUM+CVALUE*EXP(CITPI*GDR)*DEP(IND)*YLM(IND,INDYLM)
            ENDDO
            d_LM(INDYLM)=CSUM*RINPL
         ENDDO
      ENDDO
      !   and transform LM -> ll'mm' i.e. d_LM -> d_IJ
      CALL CALC_DLLMM_WANNIER( GTMP(:,:,NI,ISP),d_LM(:),PP)

      ENDIF lpaw
      
      ENDDO spin

!=======================================================================
      ENDDO ion
!=======================================================================
      DEALLOCATE(XS,YS,ZS,DIST,DEP,YLM)

#ifdef gammareal
      CALLMPI( M_sum_d(GRIDC%COMM, GTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))      
#else
      CALLMPI( M_sum_d(GRIDC%COMM, GTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ*2))
#endif

 2000 CONTINUE      
      
#ifdef MPI
      ion2: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion2
         DO ISP=1,WDES%NCDIJ
            d_IJ(:,:,NIP,ISP)=GTMP(:,:,NI,ISP)
         ENDDO
      ENDDO ion2

      DEALLOCATE(GTMP)
#else
#undef GTMP
#endif

      RETURN
      END SUBROUTINE

      
!***********************************************************************
!
!***********************************************************************
      
      SUBROUTINE CALC_DLLMM_WANNIER( DLLMM, DLM, P)
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      GDEF DLLMM(:,:)  ! net augmentation charge
      GDEF DLM(:)      ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP

    ! initialize everything to 0
      DLLMM=0

   ! loop over all channels (l,epsilon)
      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

   ! transform coefficients and multiply with QPAW (which gives the moment
   ! of the corresponding charge)

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                      DLM(JS(IC))*YLM3(IC)*P%QPAW(CH1,CH2,JL(IC))
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
      END SUBROUTINE CALC_DLLMM_WANNIER


!************************* SUBROUTINE OVERL_WANNIER *******************
!
! calculate the result of the overlap-operator acting onto a set of
! wavefunctions
!**********************************************************************

      SUBROUTINE OVERL_WANNIER(WDES, LOVERL,LMDIM, ISP, CQIJ, CPROF,CRESUL)
      USE prec
      USE wave
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavedes) WDES
      GDEF CRESUL(WDES%NPROD,WDES%NBANDS),CPROF(WDES%NPROD,WDES%NBANDS)

      LOGICAL LOVERL
      GDEF CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      IF (LOVERL) THEN

       BANDS: DO NB=1,WDES%NBANDS

       DO NP=1,WDES%NPRO
        CRESUL(NP,NB)=0
       ENDDO

       spinor: DO ISPINOR=0,WDES%NRSPINORS-1
       DO ISPINOR_=0,WDES%NRSPINORS-1

       NPRO =ISPINOR *WDES%NPRO/2
       NPRO_=ISPINOR_*WDES%NPRO/2

       NIS =1
       DO NT=1,WDES%NTYP
         LMMAXC=WDES%LMMAX(NT)
         IF (LMMAXC==0) GOTO 100
         DO NI=NIS,WDES%NITYP(NT)+NIS-1

           DO L =1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
           DO LP=1,LMMAXC
             CRESUL(L+NPRO,NB)=CRESUL(L+NPRO,NB)+CQIJ(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)*CPROF(LP+NPRO_,NB)
           ENDDO
           ENDDO
           NPRO = LMMAXC+NPRO
           NPRO_= LMMAXC+NPRO_
         ENDDO
 100     NIS = NIS+WDES%NITYP(NT)
       ENDDO
       ENDDO
       ENDDO spinor

       ENDDO BANDS
      ENDIF

      RETURN
      END SUBROUTINE OVERL_WANNIER


!************************* SUBROUTINE SHIFT_WAV ************************
!
! this subroutine calculates the product of e^iGr (or cos(Gr), or sin(Gr))
! with a wavefunction stored in CR and returns the result in CVR
! (akin to what VHAMIL does for a local potential)
!
!***********************************************************************

      SUBROUTINE SHIFT_WAV(WDES1,GRID,SV,CR,CVR)
      USE prec
      USE mgrid
      USE wave
      IMPLICIT NONE

      TYPE (grid_3d)     GRID
      TYPE (wavedes1)    WDES1

      COMPLEX(q)   SV(GRID%MPLWV) ! "shifting potential"

      COMPLEX(q) :: CR(GRID%MPLWV*WDES1%NRSPINORS),CVR(GRID%MPLWV*WDES1%NRSPINORS)
! local variables
      REAL(q) RINPLW
      INTEGER M,MM,ISPINOR

      RINPLW=1._q/GRID%NPLWV
!
! calculate the shifted wavefunction
! 
      IF (WDES1%NRSPINORS==1) THEN
         DO M=1,GRID%RL%NP
            CVR(M)= SV(M) *CR(M)*RINPLW
         ENDDO
      ELSE
         CVR(1:GRID%MPLWV*2)=0
         DO ISPINOR =0,1
            DO M=1,GRID%RL%NP
               MM =M+ISPINOR *GRID%MPLWV
               CVR(MM)= SV(M) *CR(MM)*RINPLW
            ENDDO
         ENDDO
      ENDIF
      
      END SUBROUTINE SHIFT_WAV      


!************************ SUBROUTINE SET_SV_WANNIER *********************
!************************************************************************

      SUBROUTINE SET_SV_WANNIER(G,GRID,WDES1,CVALUE,SV)
      
      USE prec
      USE mgrid
      USE wave

      IMPLICIT NONE      
      TYPE(grid_3d)  GRID
      TYPE(wavedes1) WDES1
      
      INTEGER G(3)
      
      REAL(q) FFTSCA
      COMPLEX(q) CVALUE
      COMPLEX(q) SVWORK(GRID%MPLWV), SV(GRID%MPLWV)! "shifting potential"
      INTEGER IPW,IX,IY,IZ
      
      SVWORK=0

      FFTSCA=1._q
#ifdef gammareal
      IF (G(1)/=0.OR.G(2)/=0.OR.G(3)/=0) THEN
         FFTSCA=1/SQRT(2._q)
      ENDIF
#endif      

      DO IPW=1,WDES1%NGVECTOR
         IX=WDES1%IGX(IPW)
         IY=WDES1%IGY(IPW)
         IZ=WDES1%IGZ(IPW)
         IF (IX==G(1).AND.IY==G(2).AND.IZ==G(3)) THEN
            SVWORK(IPW)=CVALUE*FFTSCA
         ENDIF
      ENDDO

      ! take the "shifting potential" to real space
      CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),SV,SVWORK,GRID)

      RETURN
      END SUBROUTINE


!************************ SUBROUTINE WANNMAT ***************************
!
! this subroutine calculates the following matrix elements
!
!  CHAM(n2,n1) = < C(n2) | e (i G r) | C(n1) >
!
! where the reciprocal lattice vector G is handled down by the 
! calling subroutine in the array IG
!
!***********************************************************************

      SUBROUTINE WANNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, IRDMAX, IG, NK, ISP, CHAM, LIMAG)
     
      USE prec
      USE wave_mpi
      USE wave
      USE wave_high
      USE lattice
      USE mpimy
      USE mgrid
      
      USE nonl_high
      USE hamil
      USE constant
      USE jacobi
      USE scala
      USE pot
      USE poscar
      USE pseudo
      USE main_mpi

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID,GRIDC,GRIDUS
      TYPE (transit)     C_TO_US
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
! G-vector for which the operator < C(n2) | e (i G r) | C(n1) >
! should be evaluated
      INTEGER  IG(3),NK,ISP
      LOGICAL  LIMAG

! work array to store the matrix int Q_ij(r) e(i G r)
      GDEF, ALLOCATABLE :: CDIJ(:,:,:,:)

      COMPLEX(q) CSET_IN_SV

      GDEF    CHAM(WDES%NB_TOT, WDES%NB_TOT)
      LOGICAL LOVERL, DO_REDIS

! work arrays (do max of 16 strips simultaneously)
      PARAMETER (NSTRIPD=16)

      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      TYPE (wavefun1)    W1             ! current wavefunction

! wavefunction in real space
      COMPLEX(q),ALLOCATABLE,TARGET::   CR(:),CH(:,:),CVR(:)
      GDEF,ALLOCATABLE,TARGET::  CPROW(:,:)
! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:),CH_RED(:,:)
      GDEF   , POINTER :: CPROW_RED(:,:),CPROJ_RED(:,:)
! maybe we will need this in the future
      COMPLEX(q)  SV(GRID%MPLWV) ! local potential

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NODE_ME=0
      IONODE =0
      NCPU=1
#endif
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
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

      ! set NSTRIP between [1 and 32]
      NSTRIP=MAX(MIN(NSTRIPD,32/NCPU,NBANDS),1)

! allocate work space
      ALLOCATE(CPROW(WDES%NPROD,NBANDS), &
           CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
           CR(GRID%MPLWV*WDES%NRSPINORS),CVR(GRID%MPLWV*WDES%NRSPINORS),CH(WDES%NRPLWV,NSTRIP))

      W1%CR=>CR

!=======================================================================
!
!=======================================================================
      IF (LIMAG) THEN
         CSET_IN_SV=(0._q,-1._q)
      ELSE
         CSET_IN_SV=(1._q,0._q)
      ENDIF

!=======================================================================
      CALL SET_DIJ_WANNIER(IG,WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P, &
     &  T_INFO,LOVERL,LMDIM,CDIJ,IRDMAX,CSET_IN_SV)
     
      CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

!   get pointers for redistributed wavefunctions
!   I can not guarantee that this works with all f90 compilers
!   please see comments in wave_mpi.F
      IF (DO_REDIS) THEN
        CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
        CALL SET_WPOINTER(CH_RED,   NRPLWV_RED, NCPU*NSTRIP, CH(1,1))
        CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROW_RED, NPROD_RED, NB_TOT, CPROW(1,1))
      ELSE
        CW_RED    => W%CW(:,:,NK,ISP)
        CH_RED    => CH(:,:)
        CPROJ_RED => W%CPROJ(:,:,NK,ISP)
        CPROW_RED => CPROW(:,:)
      ENDIF

!   set number of wavefunctions after redistribution
      NPL = WDES1%NPL     ! number of plane waves/node after data redistribution
      NPRO= WDES1%NPRO    ! number of projected wavef. after data redistribution
      NPRO_O=NPRO
      IF (.NOT. LOVERL) NPRO_O=0

      CALLMPI( SET_NPL_NPRO(WDES1, NPL, NPRO))

!=======================================================================
!  calculate the matrix
!=======================================================================
!  calculate D |cfin_n>   (D = \int Q_ij(r) e(i G r))
      CALL OVERL_WANNIER(WDES, .TRUE.,LMDIM,ISP,CDIJ, W%CPROJ(:,:,NK,ISP),CPROW)

! redistribute the wavefunctions ("bands" -> "planewave coefs")
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS, W%CW(1,1,NK,ISP))
         CALL REDIS_PROJ(WDES1, NBANDS, CPROW(1,1))
         CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
      ENDIF

      CHAM=0
      NDONE=0
      
!  get me the shifting potential SV (= e^iGr, or cos(Gr), or sin(Gr)) in real space
      CALL SET_SV_WANNIER(IG,GRID,WDES1,CSET_IN_SV,SV)

  strip: DO NPOS=1,NBANDS,NSTRIP
      NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

! calculate e ^(i G r) |phi> for a block containing NSTRIP_ACT wavefunctions
! first redistribute W%CW from NPOS to NPOS+NSTRIP_ACT ("planewave coefs" -> "bands")
      IF (DO_REDIS) CALL REDIS_PW(WDES1,NSTRIP_ACT,W%CW(1,NPOS,NK,ISP))
       
      DO N=NPOS,NPOS+NSTRIP_ACT-1

         NDONE=NDONE+1
         NP=N-NPOS+1

         ! fft |phi> to real space
         DO ISPINOR=0,WDES%NRSPINORS-1
            CALL FFTWAV(WDES%NGVECTOR(NK),WDES%NINDPW(1,NK),CR(1+ISPINOR*GRID%MPLWV), &
           &             W%CW(1+ISPINOR*WDES%NGVECTOR(NK),N,NK,ISP),GRID)
         ENDDO

         ! SV(r) * phi(r)
         RINPLW=1._q/GRID%NPLWV
         CALL SHIFT_WAV(WDES1,GRID,SV,CR(1),CVR(1))
         
         ! fft of SV(r) * phi(r)
         spinors: DO ISPINOR=0,WDES%NRSPINORS-1
         CALL FFTEXT(WDES%NGVECTOR(NK),WDES%NINDPW(1,NK),CVR(1+ISPINOR*GRID%MPLWV), &
        &             CH(1+ISPINOR*WDES%NGVECTOR(NK),NP),GRID,.FALSE.)
         ENDDO spinors

      ENDDO

! redistribute W%CW and CH ("bands" -> "planewave coefs")
! W%CW from NPOS to NPOS+NSTRIP_ACT (see above)
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NSTRIP_ACT, W%CW(1,NPOS,NK,ISP))
         CALL REDIS_PW(WDES1, NSTRIP_ACT, CH(1,1))
      ENDIF

      NPOS_RED  =(NPOS-1)*NCPU+1
      NSTRIP_RED=NSTRIP_ACT*NCPU

! take the inproduce between current stripe and all other wavefunctions
      CALL ORTH1('L', &
        CW_RED(1,1),CH_RED(1,1),CPROJ_RED(1,1), &
        CPROW_RED(1,NPOS_RED),NB_TOT, &
        NPOS_RED, NSTRIP_RED, NPL,NPRO,NRPLWV_RED,NPROD_RED,CHAM(1,1))

      ENDDO strip

!  just for safety everything ok ?
!  Mind that all wavefunctions are redistributed at this point
      IF (NDONE/=NBANDS .OR. NPOS_RED+NSTRIP_RED/=NB_TOT+1 ) THEN
       WRITE(*,*)'EDDIAG: fatal internal error (2):',NDONE,NBANDS,NPOS_RED+NSTRIP_RED,NB_TOT+1
       STOP
      ENDIF
      statmg
      CALLMPI( M_sum_g(WDES%COMM,CHAM(1,1),NB_TOT*NB_TOT))
!-----------------------------------------------------------------------
! dump some arrays
!-----------------------------------------------------------------------
#ifdef debug
      io_begin
        NPL2=MIN(10,NB_TOT)
        DO N1=1,NPL2
          WRITE(*,1)N1,(REAL( CHAM(N1,N2) ,KIND=q) ,N2=1,NPL2)
        ENDDO
        WRITE(*,*)
#ifndef gammareal
        DO N1=1,NPL2
          WRITE(*,1)N1,(AIMAG(CHAM(N1,N2)),N2=1,NPL2)
        ENDDO
        WRITE(*,*)
#endif
      1 FORMAT(1I2,3X,20F9.5)
      io_end
#endif

!-----------------------------------------------------------------------
!  back redistribution of date from over plane wave coefficients
!  to over bands
!-----------------------------------------------------------------------

      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
        DWRITE "redis ok"
      ENDIF

      DEALLOCATE(CPROW, CR, CVR, CH, CDIJ )

      RETURN
      END SUBROUTINE WANNMAT

      
!************************ SUBROUTINE ROTORB ****************************
!
! this subroutine minimizes the spread functional of Silvestrelli
!
!  \Omega = 1/(2\pi)^2 \sum_n \sum_I \omega_I (1-|z_{I,n}|^2)
!
! using the orbital rotation method [see PRB. 61, 10040 (2000)].
!
!***********************************************************************     

      SUBROUTINE ROTORB(W,WDES,Z_,NOCC,NOCCD,NCOMP,LATT_CUR,THRESH,U,R,RR,IU)

      USE prec
      USE mpimy
      USE constant
      USE lattice
      USE wave
      
      IMPLICIT NONE      

      TYPE (latt) LATT_CUR
      TYPE (wavespin) W
      TYPE (wavedes) WDES

      INTEGER I,J,N,NOCC,NOCCD,II,JJ,IJ,IN,JN,ISP,ITER
      
      INTEGER ICOMP,NCOMP
      INTEGER IU
      
      REAL(q) A11,A12,A13,A21,A22,A23,A31,A32,A33      
      REAL(q) PHI,A,B,WNEW,WPREV,WDIFF,THRESH
      REAL(q) ALPHA,COSPHI,SINPHI
      REAL(q) UI,UJ
      REAL(q) WEIGHT(6)      
      GDEF    U(WDES%NB_TOT,WDES%NB_TOT)
      
      COMPLEX(q) QII,QJJ,QIJ,QI,QJ      
      COMPLEX(q) Z_(WDES%NB_TOT,WDES%NB_TOT,NCOMP)
      COMPLEX(q), ALLOCATABLE :: Z(:,:)

      REAL(q),PARAMETER :: TINY=1E-4_q
      
      REAL(q) R(3,NOCCD),RR(8,NOCCD)
      
! Move < \psi_n | e^{i G_I r} | \psi_m > to lower triangular storage mode
      ALLOCATE(Z(NOCC*(NOCC+1)/2,NCOMP))

      DO ICOMP=1,NCOMP
      N=0
      DO I=1,NOCC
      DO J=1,I
         N=N+1
         Z(N,ICOMP)=Z_(I,J,ICOMP)
      ENDDO
      ENDDO
      ENDDO
      
! Initialize rotation matrix U to identity
      U=0
      DO N=1,WDES%NB_TOT
         U(N,N)=1._q
      ENDDO         
       
      A11=LATT_CUR%A(1,1);A21=LATT_CUR%A(2,1);A31=LATT_CUR%A(3,1)
      A12=LATT_CUR%A(1,2);A22=LATT_CUR%A(2,2);A32=LATT_CUR%A(3,2)
      A13=LATT_CUR%A(1,3);A23=LATT_CUR%A(2,3);A33=LATT_CUR%A(3,3)

! Calculate the weights for <r^2>      
      WEIGHT(4)=A11*A12+A21*A22+A31*A32
      WEIGHT(5)=A11*A13+A21*A23+A31*A33
      WEIGHT(6)=A12*A13+A22*A23+A32*A33
      WEIGHT(1)=A11*A11+A21*A21+A31*A31-WEIGHT(4)-WEIGHT(5)
      WEIGHT(2)=A12*A12+A22*A22+A32*A32-WEIGHT(4)-WEIGHT(6)
      WEIGHT(3)=A13*A13+A23*A23+A33*A33-WEIGHT(5)-WEIGHT(6)

! Initial value of function to be maximized
      WNEW=0      
      DO ICOMP=1,NCOMP
         DO N=1,NOCC
            WNEW=WNEW+WEIGHT(ICOMP)*Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))/(TPI*TPI)
         ENDDO
      ENDDO
      
      ITER=0
! Main loop of the 2x2 rotations
   10 ITER=ITER+1
   
      DO I=2,NOCC
      II=IND_(I,I)
      DO J=1,I-1
      IJ=IND_(I,J); JJ=IND_(J,J)
      
      A=0; B=0
      DO ICOMP=1,NCOMP
         A=A+WEIGHT(ICOMP)*(Z(IJ,ICOMP)*CONJG(Z(IJ,ICOMP))-&
        &   (Z(JJ,ICOMP)-Z(II,ICOMP))*CONJG(Z(JJ,ICOMP)-Z(II,ICOMP))/4)
         B=B+WEIGHT(ICOMP)*REAL(Z(IJ,ICOMP)*CONJG(Z(JJ,ICOMP)-Z(II,ICOMP)),q)
      ENDDO
            
      ALPHA=ASIN(B/SQRT(A*A+B*B))/4
      
! Get the optimal rotation      
      IF (A>0._q.AND.B>0._q) THEN
         COSPHI=COS(PI/4-ALPHA)
         SINPHI=SIN(PI/4-ALPHA)
      ELSE IF (A>0._q.AND.B<0._q) THEN
         COSPHI=COS(-PI/4-ALPHA)
         SINPHI=SIN(-PI/4-ALPHA)
      ELSE IF (A<0._q .AND.B<0._q) THEN
         COSPHI=COS(ALPHA)
         SINPHI=SIN(ALPHA)
      ELSE IF (A<0._q.AND.B>0._q) THEN      
         COSPHI=COS(ALPHA)
         SINPHI=SIN(ALPHA)
      ELSE IF (A==0._q.AND.B>0._q) THEN      
         COSPHI=COS(PI/8)
         SINPHI=SIN(PI/8)
      ELSE IF (A==0._q.AND.B<0._q) THEN
         COSPHI=COS(3*PI/8)
         SINPHI=SIN(3*PI/8)
      ELSE IF (A>0._q.AND.B==0._q) THEN
         COSPHI=COS(PI/4)
         SINPHI=SIN(PI/4)
      ELSE IF (A<0._q.AND.B==0._q) THEN
         COSPHI=1
         SINPHI=0
      ELSE         
         COSPHI=1
         SINPHI=0
      ENDIF      
 
! Update transformation matrix
      DO N=1,NOCC
         UI= COSPHI*U(N,I)-SINPHI*U(N,J)
         UJ= SINPHI*U(N,I)+COSPHI*U(N,J)
         U(N,I)=UI
         U(N,J)=UJ
      ENDDO
 
! Rotate integrals 
      DO ICOMP=1,NCOMP
         QII=COSPHI*COSPHI*Z(II,ICOMP)+SINPHI*SINPHI*Z(JJ,ICOMP)-&
        &     2*COSPHI*SINPHI*Z(IJ,ICOMP)
         QJJ=SINPHI*SINPHI*Z(II,ICOMP)+COSPHI*COSPHI*Z(JJ,ICOMP)+&
        &     2*COSPHI*SINPHI*Z(IJ,ICOMP)
         QIJ=(COSPHI*COSPHI-SINPHI*SINPHI)*Z(IJ,ICOMP)+&
        &     COSPHI*SINPHI*(Z(II,ICOMP)-Z(JJ,ICOMP))
         DO N=1,NOCC
            IN=IND_(I,N)
            JN=IND_(J,N)
            QI= COSPHI*Z(IN,ICOMP)-SINPHI*Z(JN,ICOMP)
            QJ= SINPHI*Z(IN,ICOMP)+COSPHI*Z(JN,ICOMP)
            Z(IN,ICOMP)=QI
            Z(JN,ICOMP)=QJ
         ENDDO
         Z(II,ICOMP)=QII
         Z(IJ,ICOMP)=QIJ
         Z(JJ,ICOMP)=QJJ
      ENDDO

      ENDDO ! loop over j
      ENDDO ! loop over i
      
! Recalculate functional
      WPREV=WNEW
      WNEW=0
      DO ICOMP=1,NCOMP
         DO N=1,NOCC
            WNEW=WNEW+WEIGHT(ICOMP)*Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))/(TPI*TPI)
         ENDDO
      ENDDO
      
      WDIFF=WNEW-WPREV
      WDIFF=WDIFF/NOCC

! Check convergence
      IF (ABS(WDIFF)>THRESH.OR.ITER<10) THEN
#ifdef debug
      IF (IU>=0) THEN
         WRITE(*,*) 'Iteration:',ITER,' W_new:',WNEW,' W_prev:',WPREV
      ENDIF
#endif
         GOTO 10
      ENDIF             

#ifdef debug
      IF (IU>=0) THEN
        N=MIN(10,NOCC)
        DO ICOMP=1,NCOMP
        WRITE(*,*) 'component',ICOMP
        DO I=1,N
           WRITE(*,1) I,(REAL(Z(IND_(I,J),ICOMP),q), J=1,I)
        ENDDO
        WRITE(*,*)
        DO I=1,N
           WRITE(*,1) I,(AIMAG(Z(IND_(I,J),ICOMP)), J=1,I)
        ENDDO
        WRITE(*,*)
        ENDDO
      1 FORMAT(1I2,3X,20F9.5)
      ENDIF
#endif 

      DO N=1,NOCC
      DO ICOMP=1,3
         R(1:3,N)=R(1:3,N)+LATT_CUR%A(1:3,ICOMP)*AIMAG(LOG(Z(IND_(N,N),ICOMP)))/TPI
      ENDDO 
      ENDDO
#ifdef debug
      IF (IU>=0) THEN
      WRITE(IU,*) ' Maximally localized Wannier functions:'
      WRITE(IU,*)
      WRITE(IU,'(A,I5,A,F12.5)') '# of occ. bands: ',NOCC,'  Z total: ',WNEW
      WRITE(IU,*)
      WRITE(IU,*) '  n      <x>      <y>      <z>    '
      WRITE(IU,*) '----------------------------------'
      DO N=1,NOCC
         WRITE(IU,2) N,R(1:3,N)
      ENDDO
      WRITE(IU,*) '----------------------------------'
      WRITE(IU,*)
    2 FORMAT(1I4,3X,3(20F9.5))
      ENDIF
#endif      
! \Omega
      DO N=1,NOCC
      DO ICOMP=1,NCOMP
         RR(1,N)=RR(1,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO

! <r^2>
!     DO N=1,NOCC
!     DO ICOMP=1,NCOMP
!        RR(2,N)=RR(2,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!       &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
!     ENDDO
!     ENDDO


! <xx>
      WEIGHT(1)=A11*A11 - A11*A12 - A11*A13
      WEIGHT(2)=A12*A12 - A11*A12 - A12*A13
      WEIGHT(3)=A13*A13 - A11*A13 - A12*A13
      WEIGHT(4)=A11*A12
      WEIGHT(5)=A11*A13
      WEIGHT(6)=A12*A13

      
      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!        RR(N,3)=RR(N,3)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!       &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
         RR(2,N)=RR(2,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO

! <yy>
      WEIGHT(1)=A21*A21 - A21*A22 - A21*A23
      WEIGHT(2)=A22*A22 - A21*A22 - A22*A23
      WEIGHT(3)=A23*A23 - A21*A23 - A22*A23
      WEIGHT(4)=A21*A22
      WEIGHT(5)=A21*A23
      WEIGHT(6)=A22*A23

      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!        RR(N,4)=RR(N,4)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!       &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
         RR(3,N)=RR(3,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO
      
! <zz>
      WEIGHT(1)=A31*A31 - A31*A32 - A31*A33
      WEIGHT(2)=A32*A32 - A31*A32 - A32*A33
      WEIGHT(3)=A33*A33 - A31*A33 - A32*A33
      WEIGHT(4)=A31*A32
      WEIGHT(5)=A31*A33
      WEIGHT(6)=A32*A33

      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!        RR(N,5)=RR(N,5)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!       &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
         RR(4,N)=RR(4,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO
      
! <xy>
      WEIGHT(1)=A11*A21 - (A11*A22+A12*A21)/2 - (A11*A23+A13*A21)/2
      WEIGHT(2)=A12*A22 - (A11*A22+A12*A21)/2 - (A12*A23+A13*A22)/2
      WEIGHT(3)=A13*A23 - (A11*A23+A13*A21)/2 - (A12*A23+A13*A22)/2
      WEIGHT(4)=(A11*A22+A12*A21)/2
      WEIGHT(5)=(A11*A23+A13*A21)/2
      WEIGHT(6)=(A12*A23+A13*A22)/2

      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!        RR(N,6)=RR(N,6)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!       &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
         RR(5,N)=RR(5,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO

! <xz>
      WEIGHT(1)=A11*A31 - (A11*A32+A12*A31)/2 - (A11*A33+A13*A31)/2
      WEIGHT(2)=A12*A32 - (A11*A32+A12*A31)/2 - (A12*A33+A13*A32)/2
      WEIGHT(3)=A13*A33 - (A11*A33+A13*A31)/2 - (A12*A33+A13*A32)/2
      WEIGHT(4)=(A11*A32+A12*A31)/2   
      WEIGHT(5)=(A11*A33+A13*A31)/2
      WEIGHT(6)=(A12*A33+A13*A32)/2
      
      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!       RR(N,7)=RR(N,7)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!      &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
        RR(6,N)=RR(6,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO

! <yz>
      WEIGHT(1)=A21*A31 - (A21*A32+A22*A31)/2 - (A21*A33+A23*A31)/2
      WEIGHT(2)=A22*A32 - (A21*A32+A22*A31)/2 - (A22*A33+A23*A32)/2
      WEIGHT(3)=A23*A33 - (A21*A33+A23*A31)/2 - (A22*A33+A23*A32)/2
      WEIGHT(4)=(A21*A32+A22*A31)/2
      WEIGHT(5)=(A21*A33+A23*A31)/2
      WEIGHT(6)=(A22*A33+A23*A32)/2
      
      DO N=1,NOCC
      DO ICOMP=1,NCOMP
!       RR(N,8)=RR(N,8)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP))+ &
!      &                       AIMAG(LOG(Z(IND_(N,N),ICOMP)))**2)/(TPI*TPI)
        RR(7,N)=RR(7,N)+WEIGHT(ICOMP)*(1-Z(IND_(N,N),ICOMP)*CONJG(Z(IND_(N,N),ICOMP)))/(TPI*TPI)
      ENDDO
      ENDDO

! Anisotropy
      DO N=1,NOCC
         RR(8,N)=SQRT( RR(2,N)**2+RR(3,N)**2+RR(4,N)**2 + &
        &                3*(RR(5,N)**2+RR(6,N)**2+RR(7,N)**2) - &
        &                 RR(5,N)*RR(6,N)-RR(5,N)*RR(7,N)-RR(6,N)*RR(7,N) )
      ENDDO

! Write second moment matrix
#ifdef debug
      IF (IU>=0) THEN
      WRITE(IU,*) '  n      Omega    Anis.    <xx>     <yy>     <zz>     <xy>     <xz>     <yz>   '
      WRITE(IU,*) '-------------------------------------------------------------------------------'
      DO N=1,NOCC
         WRITE(IU,3) N,RR(1,N),RR(8,N),RR(2:7,N)
      ENDDO
      WRITE(IU,*) '-------------------------------------------------------------------------------'
      WRITE(IU,*)
    3 FORMAT(1I4,3X,8F9.5)
      ENDIF
#endif
      DEALLOCATE(Z)

      RETURN
      END SUBROUTINE ROTORB


      FUNCTION IND_(I,J)      
      IMPLICIT NONE
      INTEGER IND_,I,J
      
      IND_= ((MAX(I,J)*(MAX(I,J)-1)/2)+MIN(I,J))
      
      END FUNCTION IND_


!************************ SUBROUTINE ROTWAV ****************************
!
!***********************************************************************
      
      SUBROUTINE ROTWAV(W,WDES,GRID,LOVERL,NK,ISP,CHAM)
      
      USE prec
      USE wave_mpi
      USE wave
      USE wave_high
      USE lattice
      USE mpimy
      USE mgrid
      
      USE nonl_high
      USE hamil
      USE constant
      USE jacobi
      USE scala
      USE pot
      USE poscar
      USE pseudo
      USE main_mpi
      USE dfast

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES

      GDEF    CHAM(WDES%NB_TOT, WDES%NB_TOT)
      LOGICAL LOVERL, DO_REDIS

      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      TYPE (wavefun1)    W1             ! current wavefunction

! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:)
      GDEF   , POINTER :: CPROJ_RED(:,:)
! maybe we will need this in the future

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NODE_ME=0
      IONODE =0
      NCPU=1
#endif
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
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

      CALL SETWDES(WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

!   get pointers for redistributed wavefunctions
!   I can not guarantee that this works with all f90 compilers
!   please see comments in wave_mpi.F
      IF (DO_REDIS) THEN
        CALL SET_WPOINTER(CW_RED,   NRPLWV_RED, NB_TOT, W%CW(1,1,NK,ISP))
        CALL SET_GPOINTER(CPROJ_RED, NPROD_RED, NB_TOT, W%CPROJ(1,1,NK,ISP))
      ELSE
        CW_RED    => W%CW(:,:,NK,ISP)
        CPROJ_RED => W%CPROJ(:,:,NK,ISP)
      ENDIF

      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
      ENDIF

!   set number of wavefunctions after redistribution
      NPL = WDES1%NPL     ! number of plane waves/node after data redistribution
      NPRO= WDES1%NPRO    ! number of projected wavef. after data redistribution

      CALLMPI(SET_NPL_NPRO(WDES1, NPL, NPRO))

      CALL LINCOM('F',CW_RED(:,:),CPROJ_RED(:,:),CHAM(1,1), &
             NB_TOT,NB_TOT,NPL,NPRO,NRPLWV_RED,NPROD_RED,NB_TOT, &
             CW_RED(:,:),CPROJ_RED(:,:))
      DWRITE "lincom ok"

!-----------------------------------------------------------------------
!  back redistribution of date from over plane wave coefficients
!  to over bands
!-----------------------------------------------------------------------

      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1, NBANDS, W%CW   (1,1,NK,ISP))
        DWRITE "redis ok"
      ENDIF
      
      RETURN
      END SUBROUTINE ROTWAV


      SUBROUTINE LOCALIZE(W, WDES, GRID, GRIDC, GRIDUS, C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, IRDMAX, IU0, IU6)

     
      USE prec
      USE wave
      USE wave_high
      USE mpimy
      USE lattice
      USE mgrid
      USE constant
      USE pot
      USE poscar
      USE pseudo

      IMPLICIT NONE

      TYPE (grid_3d)     GRID,GRIDC,GRIDUS
      TYPE (transit)     C_TO_US
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      
      INTEGER I,J,NB,N,ISP
      INTEGER IRDMAX,LMDIM,IU0,IU6
      INTEGER ICOMP,NOCC(WDES%ISPIN)
      INTEGER IREPEAT,NREPEAT
      INTEGER, PARAMETER :: NCOMP=6
      REAL(q), PARAMETER :: THRESH=1.0E-12_q
      REAL(q), PARAMETER :: TINY=1E-4_q

      GDEF   U(WDES%NB_TOT,WDES%NB_TOT)

      GDEF       Z_(WDES%NB_TOT,WDES%NB_TOT)
      COMPLEX(q) Z(WDES%NB_TOT,WDES%NB_TOT,NCOMP)

      REAL(q), ALLOCATABLE :: R(:,:,:),RR(:,:,:)
      
      LOGICAL LOVERL
      
      INTEGER IG(3,6)
      
      IG(1,1)=1;IG(2,1)=0;IG(3,1)=0
      IG(1,2)=0;IG(2,2)=1;IG(3,2)=0
      IG(1,3)=0;IG(2,3)=0;IG(3,3)=1
      IG(1,4)=1;IG(2,4)=1;IG(3,4)=0
      IG(1,5)=1;IG(2,5)=0;IG(3,5)=1
      IG(1,6)=0;IG(2,6)=1;IG(3,6)=1

! Get the number of occupied bands per spin component
      NOCC=0
      DO ISP=1,WDES%ISPIN
      DO I=1,WDES%NB_TOT
         IF (W%FERTOT(I,1,ISP).LE.TINY) CYCLE
         NOCC(ISP)=NOCC(ISP)+1
      ENDDO
      ENDDO
      
      ALLOCATE(R(3,MAXVAL(NOCC(1:WDES%ISPIN)),WDES%ISPIN), &
     &            RR(8,MAXVAL(NOCC(1:WDES%ISPIN)),WDES%ISPIN))

      NREPEAT=1
      DO IREPEAT=1,NREPEAT
      
      R=0;RR=0
      
      spin: DO ISP=1,WDES%ISPIN
      
      Z=0
      DO ICOMP=1,NCOMP
#ifdef debug
      IF (IU0>=0) THEN
         write(*,*) 'icomp = ',icomp,' real part'
      ENDIF
#endif
         CALL WANNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, IRDMAX, IG(:,ICOMP), 1, ISP, Z_, .FALSE.)
         Z(:,:,ICOMP)=Z(:,:,ICOMP)+Z_(:,:)

#ifdef debug
      IF (IU0>=0) THEN
         write(*,*) 'icomp = ',icomp,' imaginary part'
      ENDIF
#endif
#ifdef gammareal
         CALL WANNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, IRDMAX, IG(:,ICOMP), 1, ISP, Z_, .TRUE.)
         Z(:,:,ICOMP)=Z(:,:,ICOMP)+CMPLX(0._q,Z_(:,:),q)      
#endif
      ENDDO
      CALL ROTORB(W,WDES,Z,NOCC(ISP),MAXVAL(NOCC(1:WDES%ISPIN)),NCOMP,LATT_CUR,THRESH, &
     &             U,R(:,:,ISP),RR(:,:,ISP),IU6)

#ifdef debug
      IF (IU0>=0) THEN
        NB=MIN(10,WDES%NB_TOT)
        DO I=1,NB
          WRITE(*,1) I,(U(I,J), J=1,NB)
        ENDDO
      1 FORMAT(1I2,3X,20F9.5)
      ENDIF
#endif      
      IF (LROTATE) CALL ROTWAV(W,WDES,GRID,LOVERL,1,ISP,U)

      ENDDO spin

      ENDDO ! repetition of localization cycle

      IF (IU6>=0) THEN
      WRITE(IU6,*) ' Maximally localized Wannier functions:'
      WRITE(IU6,*)
      DO ISP=1,WDES%ISPIN
      IF (WDES%ISPIN==2) THEN
         IF (ISP==1) THEN
            WRITE(IU6,*) ' Spin-up'
            WRITE(IU6,*)
         ELSE
            WRITE(IU6,*) ' Spin-down'
            WRITE(IU6,*)
         ENDIF
      ENDIF
      
      WRITE(IU6,*) '  n      <x>      <y>      <z>    '
      WRITE(IU6,*) '----------------------------------'
      DO N=1,NOCC(ISP)
         WRITE(IU6,2) N,R(1:3,N,ISP)
      ENDDO
      WRITE(IU6,*) '----------------------------------'
      WRITE(IU6,*)
    2 FORMAT(1I4,3X,3(20F9.5))

      WRITE(IU6,*) '  n      Omega    Anis.    <xx>     <yy>     <zz>     <xy>     <xz>     <yz>   '
      WRITE(IU6,*) '-------------------------------------------------------------------------------'
      DO N=1,NOCC(ISP)
         WRITE(IU6,3) N,RR(1,N,ISP),RR(8,N,ISP),RR(2:7,N,ISP)
      ENDDO
      WRITE(IU6,*) '-------------------------------------------------------------------------------'
      WRITE(IU6,*)
    3 FORMAT(1I4,3X,8F9.5)
      ENDDO
      ENDIF

      CALL MNMAT_RSPACE(GRID, GRIDC, GRIDUS, C_TO_US, &
     &   LATT_CUR, T_INFO, W, WDES, P, IRDMAX, LMDIM, LOVERL, &
     &   1, 1, NOCC(1), R(:,:,1), IU6)

      DEALLOCATE(R,RR)
      
      RETURN
      END SUBROUTINE LOCALIZE
      

!************************ SUBROUTINE SETYLM_AUG_TRUNC ******************
!
! this subroutine performes the following tasks
! ) finds the points, which are within a certain cutoff around one ion
! ) calculates the distance of each of this points from the ion
! ) calculates the spherical harmonics Y_lm(Omega(r-R(ion))
! DISX,Y,Z are additional displacements of the ions
!
! N.B. the only difference between this subroutine and the SETYLM_AUG
! in us.F, is that SETYLM_AUG2 also writes the arrays XS,YS and ZS 
! containing the coordinates of the points at which the spherical
! harmonics are given.
! 
!***********************************************************************

      SUBROUTINE SETYLM_AUG_TRUNC(GRID,LATT_CUR,POSION,PSDMAX,NPSRNL, &
     &  LMYDIM,LYDIM,YLM,IRMAX,INDMAX,N1L,N1H,N2L,N2H,N3L,N3H,XS,YS,ZS,DIST)
       
      USE prec
      USE mpimy
      USE mgrid
      USE lattice
      USE asa
      USE constant

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)             GRID
      TYPE (latt)                LATT_CUR

      REAL(q)                    POSION(3)
      REAL(q)                    DIST(IRMAX)
      REAL(q)                    YLM(IRMAX,LMYDIM)

      REAL(q)                    XS(IRMAX),YS(IRMAX),ZS(IRMAX)

      IF ((LYDIM+1)**2 > LMYDIM) THEN
         WRITE(0,*)'internal error: LMYDIM is too small',LYDIM,LMYDIM
         STOP
      ENDIF

      XS=0;YS=0;ZS=0;DIST=0

!=======================================================================
! find lattice points contained within the cutoff-sphere
!=======================================================================
      F1=1._q/GRID%NGX
      F2=1._q/GRID%NGY
      F3=1._q/GRID%NGZ

      ARGSC=NPSRNL/PSDMAX

!-----------------------------------------------------------------------
! restrict loop to points contained within a cubus around the ion
!-----------------------------------------------------------------------
      D1= PSDMAX*LATT_CUR%BNORM(1)*GRID%NGX
      D2= PSDMAX*LATT_CUR%BNORM(2)*GRID%NGY
      D3= PSDMAX*LATT_CUR%BNORM(3)*GRID%NGZ

      N3LOW= INT(POSION(3)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
      N2LOW= INT(POSION(2)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
      N1LOW= INT(POSION(1)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX

      N3HI = INT(POSION(3)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
      N2HI = INT(POSION(2)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
      N1HI = INT(POSION(1)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX
           
      IF (N3L>N3LOW) N3LOW=N3L; IF (N3H<N3HI) N3HI=N3H
      IF (N2L>N2LOW) N2LOW=N2L; IF (N2H<N2HI) N2HI=N2H
      IF (N1L>N1LOW) N1LOW=N1L; IF (N1H<N1HI) N1HI=N1H

#ifdef debug
      WRITE(*,*) 'setylm reports'
!     WRITE(*,*) d1,d2,d3
      WRITE(*,'("POS =",3F12.7)') POSION(1:3)
      WRITE(*,'("N1LO =",I4,"  N1HI =",I4)') N1LOW,N1HI
      WRITE(*,'("N2LO =",I4,"  N2HI =",I4)') N2LOW,N2HI
      WRITE(*,'("N3LO =",I4,"  N3HI =",I4)') N3LOW,N3HI
      WRITE(*,*)
#endif

      IND=1

      DO N3=N3LOW,N3HI
      X3=(N3*F3-POSION(3))

      DO N2=N2LOW,N2HI
      X2=(N2*F2-POSION(2))

      DO N1=N1LOW,N1HI
      X1=(N1*F1-POSION(1))

      X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
      Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
      Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

      D=SQRT(X*X+Y*Y+Z*Z)
      ARG=(D*ARGSC)+1
      NADDR=INT(ARG)

      IF (NADDR<NPSRNL) THEN
        DIST(IND)=MAX(D,1E-4_q)

        XS(IND)  =X/DIST(IND)
        YS(IND)  =Y/DIST(IND)
        ZS(IND)  =Z/DIST(IND)

        IND=IND+1
      ENDIF
      ENDDO; ENDDO; ENDDO
!-----------------------------------------------------------------------
!  compare maximum index with INDMAX
!-----------------------------------------------------------------------
      INDMAX=IND-1
      IF (INDMAX>IRMAX) THEN
        WRITE(*,*) &
     &  'internal ERROR: DEPLE:  IRDMAX must be increased to',INT(INDMAX*1.1)
        STOP
      ENDIF

      CALL SETYLM(LYDIM,INDMAX,YLM,XS,YS,ZS)

      DO IND=1,INDMAX 
         XS(IND)=XS(IND)*DIST(IND)
         YS(IND)=YS(IND)*DIST(IND)
         ZS(IND)=ZS(IND)*DIST(IND)
      ENDDO

      RETURN
      END SUBROUTINE SETYLM_AUG_TRUNC


!************************ SUBROUTINE SET_DIJ_WANNIER_RSPACE *************
!************************************************************************

      SUBROUTINE SET_DIJ_WANNIER_RSPACE(WDES,GRIDC_,GRIDUS,C_TO_US, &
     &   LATT_CUR,P,T_INFO,LOVERL,LMDIM,IRDMAX,N1L,N1H,N2L,N2H,N3L,N3H, &
     &   NIONS,POSION,MAP_,d_IJ)
     
      USE prec
      USE pseudo
      USE poscar
      USE mpimy
      USE mgrid
      USE lattice
      USE wave
      USE wave_high
      USE asa
      USE paw
      USE us
      USE elpol
      USE constant
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER ::  GRIDC
      TYPE (transit)             C_TO_US  ! index table between GRIDC and GRIDUS
      TYPE (latt)                LATT_CUR
      TYPE (wavedes)             WDES

      INTEGER                    IRDMAX   ! allocation required for augmentation
      INTEGER                    LMDIM
      INTEGER                    NIONS
      INTEGER                    MAP_(NIONS)
      REAL(q)                    POSION(3,NIONS)
      REAL(q)                    POS_X,POS_Y,POS_Z
      GDEF                       SUM(0:9),d_LM(256,0:9)
      LOGICAL                    LOVERL,LADDITIONAL

      REAL(q), ALLOCATABLE ::    XS(:),YS(:),ZS(:)
      REAL(q), ALLOCATABLE ::    DIST(:),DEP(:),YLM(:,:)

      GDEF                       d_IJ(LMDIM,LMDIM,0:9,NIONS,WDES%NCDIJ)

      TYPE (potcar), POINTER ::  PP
      
      LADDITIONAL=(GRIDUS%NGX/=GRIDC_%NGX) .OR. &
                  (GRIDUS%NGY/=GRIDC_%NGY) .OR. &
                  (GRIDUS%NGZ/=GRIDC_%NGZ)
      IF (LADDITIONAL) THEN
         GRIDC => GRIDUS
      ELSE
         GRIDC => GRIDC_
      ENDIF

!=======================================================================
      d_IJ  =0
!=======================================================================

      IF (.NOT.LOVERL) GOTO 2000

      LYDIM=MAXL_AUG(T_INFO%NTYP,P)
      LMYDIM=(LYDIM+1)**2          ! number of lm pairs

      ALLOCATE(XS(IRDMAX),YS(IRDMAX),ZS(IRDMAX), &
     &  DIST(IRDMAX),DEP(IRDMAX),YLM(IRDMAX,LMYDIM))
!=======================================================================
! loop over all ions
!=======================================================================
      RINPL=1._q/GRIDC%NPLWV

      ion: DO NI=1,NIONS
            
      ! express POSION in cartesian coordinates
      POS_X=LATT_CUR%A(1,1)*POSION(1,NI)+ &
     &       LATT_CUR%A(1,2)*POSION(2,NI)+ &
     &        LATT_CUR%A(1,3)*POSION(3,NI)
      POS_Y=LATT_CUR%A(2,1)*POSION(1,NI)+ &
     &       LATT_CUR%A(2,2)*POSION(2,NI)+ &
     &        LATT_CUR%A(2,3)*POSION(3,NI)
      POS_Z=LATT_CUR%A(3,1)*POSION(1,NI)+ &
     &       LATT_CUR%A(3,2)*POSION(2,NI)+ &
     &        LATT_CUR%A(3,3)*POSION(3,NI)

#ifdef debug
      WRITE(*,'("ion: ",I4,"  prim: ",I4)') NI,MAP_(NI)
      WRITE(*,'("POS: ",3F12.7)') POS_X,POS_Y,POS_Z
#endif

      NT=T_INFO%ITYP(MAP_(NI))
      PP=>PP_POINTER(P,MAP_(NI),NT)
      ! for this ion (this type of ion) no depletion charge
      IF (PP%PSDMAX==0) CYCLE
!-----------------------------------------------------------------------
! calculate the spherical harmonics YLM and the distance DIST  between
! grid-points and central atom (DEP is a work-arrays)
!-----------------------------------------------------------------------
      LYMAX=MAXL1(PP)
      IF ( ASSOCIATED(PP%QPAW) ) THEN
      ! in paw method we truncate the augmentation charge at L=2
         LYMAX=MIN(4,LYMAX*2)
      ENDIF

      CALL SETYLM_AUG_TRUNC(GRIDC,LATT_CUR,POSION(1:3,NI), &
     &  PP%PSDMAX,NPSRNL,LMYDIM,LYMAX,YLM(1,1),IRDMAX,INDMAX, &
     &  N1L,N1H,N2L,N2H,N3L,N3H,XS(1),YS(1),ZS(1),DIST(1))

      IF (WDES%LNONCOLLINEAR) THEN
         ISP_INC=3
      ELSE
         ISP_INC=1
      ENDIF
 spin:DO ISP=1,WDES%NCDIJ,ISP_INC  

 lpaw:IF ( .NOT. ASSOCIATED(PP%QPAW) ) THEN
!=======================================================================
! US-PP
!=======================================================================
      LDEP_INDEX=1

    ! loop over all channels (l,epsilon)
      LM =1
      l_loop:  DO L =1,PP%LMAX
      LMP=LM
      lp_loop: DO LP=L,PP%LMAX
      IF (PP%NDEP(L,LP)==0) GOTO 510

      CALL SETDEP(PP%QDEP(1,1,LDEP_INDEX),PP%PSDMAX,NPSRNL, &
     &  LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
      LDEP_INDEX=LDEP_INDEX+ABS(PP%NDEP(L,LP))

    ! quantum numbers l and lp of these two channels
      LL =PP%LPS(L )
      LLP=PP%LPS(LP)

    ! loop over all m mp
      m_loop:  DO M=1,2*LL+1
      MPLOW=1 ; IF (L==LP) MPLOW=M
      mp_loop: DO MP=MPLOW,2*LLP+1

      !   calculate the indices into the array containing the spherical
      !   harmonics
         INDYLM =LL**2   +M
         INDPYL =LLP**2  +MP

         SUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC

         DO IND=1,INDMAX
            SUM(0)=SUM(0)+DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(1)=SUM(1)+(XS(IND)+POS_X)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(2)=SUM(2)+(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(3)=SUM(3)+(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(4)=SUM(4)+(XS(IND)+POS_X)*(XS(IND)+POS_X)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(5)=SUM(5)+(YS(IND)+POS_Y)*(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(6)=SUM(6)+(ZS(IND)+POS_Z)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(7)=SUM(7)+(XS(IND)+POS_X)*(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(8)=SUM(8)+(XS(IND)+POS_X)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(9)=SUM(9)+(YS(IND)+POS_Y)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
         ENDDO

         !   add to array d_IJ and make symmetric
         d_IJ(LM+M-1,LMP+MP-1,:,NI,ISP)=SUM(:)*RINPL
         d_IJ(LMP+MP-1,LM+M-1,:,NI,ISP)=SUM(:)*RINPL
!       WRITE(0,90) LL,M,LLP,MP,d_IJ(1:3,LM+M-1,LMP+MP-1,NI),TSUM*RINPL,PP%QION(L,LP)
! 90      FORMAT('l =',I2,'  m =',I2,'  lp =',I2,'  mp =',I2,'  d = (',3F12.5,')',2F12.5)	 
      ENDDO mp_loop
      ENDDO m_loop

 510  LMP=LMP+2*LLP+1
      ENDDO lp_loop
      LM =LM +2*LL +1
      ENDDO l_loop
      ELSE lpaw
!=======================================================================
! PAW approach
!=======================================================================
      d_LM=0
      DO L=0,LYMAX
         CALL SETDEP(PP%QDEP(1,1,L),PP%PSDMAX,NPSRNL, &
        &        LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
         
         DO M=1,(L*2)+1
            INDYLM=L*L+M
            SUM=0
            ! sum over all r
!DIR$ IVDEP
!OCL NOVREC
            DO IND=1,INDMAX
               SUM(0)=SUM(0)+DEP(IND)*YLM(IND,INDYLM)
               SUM(1)=SUM(1)+(XS(IND)+POS_X)*DEP(IND)*YLM(IND,INDYLM)
               SUM(2)=SUM(2)+(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)
               SUM(3)=SUM(3)+(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)
               SUM(4)=SUM(4)+(XS(IND)+POS_X)*(XS(IND)+POS_X)*DEP(IND)*YLM(IND,INDYLM)
               SUM(5)=SUM(5)+(YS(IND)+POS_Y)*(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)
               SUM(6)=SUM(6)+(ZS(IND)+POS_Z)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)
               SUM(7)=SUM(7)+(XS(IND)+POS_X)*(YS(IND)+POS_Y)*DEP(IND)*YLM(IND,INDYLM)
               SUM(8)=SUM(8)+(XS(IND)+POS_X)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)
               SUM(9)=SUM(9)+(YS(IND)+POS_Y)*(ZS(IND)+POS_Z)*DEP(IND)*YLM(IND,INDYLM)
            ENDDO
            d_LM(INDYLM,0:9)=SUM(0:9)*RINPL
         ENDDO
      ENDDO
      !   and transform LM -> ll'mm' i.e. d_LM -> d_IJ
      DO I=0,9
         CALL CALC_DLLMM_WANNIER( d_IJ(:,:,I,NI,ISP),d_LM(:,I),PP)
      ENDDO

      ENDIF lpaw
      
      ENDDO spin

!=======================================================================
      ENDDO ion
!=======================================================================
      DEALLOCATE(XS,YS,ZS,DIST,DEP,YLM)

 2000 CONTINUE      
      
      RETURN
      END SUBROUTINE SET_DIJ_WANNIER_RSPACE
      
      
      SUBROUTINE GET_IMAGES(GRID,LATT_CUR,R0,R1,N,R)
      
      USE prec
      USE lattice
      USE mgrid
      
      IMPLICIT NONE
      
      TYPE (latt)    LATT_CUR
      TYPE (grid_3d) GRID

      INTEGER I,J,K,N
      REAL(q) R0(3),R1(3),D(3),R(3,0:8)
      REAL(q) R0_(3),R1_(3),R_(3)
      REAL(q) A1L0,A2L0,A3L0,A1H0,A2H0,A3H0,A1L1,A2L1,A3L1,A1H1,A2H1,A3H1

      REAL(q),PARAMETER :: TINY=1E-4_q
      
      R0_=R0; R1_=R1
! Express centroids in direct coordinates       
      CALL KARDIR(1,R0_,LATT_CUR%B)
      CALL KARDIR(1,R1_,LATT_CUR%B)
! Bring both into primitive cell
      R0_(1)=MOD(R0_(1)+10,1._q); R0_(2)=MOD(R0_(2)+10,1._q); R0_(3)=MOD(R0_(3)+10,1._q)
      R1_(1)=MOD(R1_(1)+10,1._q); R1_(2)=MOD(R1_(2)+10,1._q); R1_(3)=MOD(R1_(3)+10,1._q)
      
      R(:,0)=R0_(:)
      N=0
! test
      R(:,1)=R1_(:)
      N=1
! test
      A1L0=R0_(1)-0.5_q; A1H0=R0_(1)+0.5_q-1._q/GRID%NGX
      A2L0=R0_(2)-0.5_q; A2H0=R0_(2)+0.5_q-1._q/GRID%NGY
      A3L0=R0_(3)-0.5_q; A3H0=R0_(3)+0.5_q-1._q/GRID%NGZ

! Get images (in direct coordinates)      
      DO K=-1,1
      DO J=-1,1
      DO I=-1,1
! test
         IF (I==0.AND.J==0.AND.K==0) CYCLE
! test
         R_(1)=R1_(1)+I; R_(2)=R1_(2)+J; R_(3)=R1_(3)+K
         
         A1L1=R_(1)-0.5_q; A1H1=R_(1)+0.5_q-1._q/GRID%NGX
         A2L1=R_(2)-0.5_q; A2H1=R_(2)+0.5_q-1._q/GRID%NGY
         A3L1=R_(3)-0.5_q; A3H1=R_(3)+0.5_q-1._q/GRID%NGZ

         IF (((A1L0<=A1L1.AND.A1L1<A1H0).OR.(A1L0<A1H1.AND.A1H1<=A1H0)) .AND. &
        &    ((A2L0<=A2L1.AND.A2L1<A2H0).OR.(A2L0<A2H1.AND.A2H1<=A2H0)) .AND. &
        &    ((A3L0<=A3L1.AND.A3L1<A3H0).OR.(A3L0<A3H1.AND.A3H1<=A3H0))) THEN

            N=N+1
            R(1,N)=R_(1)
            R(2,N)=R_(2)
            R(3,N)=R_(3)
            
         ENDIF
      ENDDO
      ENDDO
      ENDDO
      
      RETURN
      END SUBROUTINE GET_IMAGES


      SUBROUTINE MAP2IMAGE(GRID,LATT_CUR,T_INFO,P,N1L,N1H,N2L,N2H,N3L,N3H, &
     &   NIONS,POSION,MAP_)
      
      USE prec
      USE mgrid
      USE lattice
      USE poscar
      USE pseudo
      
      IMPLICIT NONE

      TYPE (grid_3d)   GRID
      TYPE (latt)      LATT_CUR
      TYPE (type_info) T_INFO
      TYPE (potcar)    P(T_INFO%NTYP)
      
      INTEGER I,J,K      
      INTEGER NI,NT,NIONS
      INTEGER N1L,N1H,N2L,N2H,N3L,N3H
      INTEGER N1LO,N1HI,N2LO,N2HI,N3LO,N3HI
      
      REAL(q) D1,D2,D3
      
      INTEGER MAP_(8*T_INFO%NIONS)
      REAL(q) PSDMAX,POS(3),POSION(3,8*T_INFO%NIONS)

      TYPE (potcar), POINTER :: PP
            
      NIONS=0
      ions: DO NI=1,T_INFO%NIONS
      
      NT=T_INFO%ITYP(NI)
      PP=>PP_POINTER(P,NI,NT)
      PSDMAX=PP%PSDMAX
      IF (PSDMAX==0) CYCLE ions
  
      D1= PSDMAX*LATT_CUR%BNORM(1)*GRID%NGX
      D2= PSDMAX*LATT_CUR%BNORM(2)*GRID%NGY
      D3= PSDMAX*LATT_CUR%BNORM(3)*GRID%NGZ

      kloop: DO K=-1,1
         POS(3)=T_INFO%POSION(3,NI)+K
         N3LO = INT(POS(3)*GRID%NGZ-D3+10*GRID%NGZ+.99_q)-10*GRID%NGZ
         N3HI = INT(POS(3)*GRID%NGZ+D3+10*GRID%NGZ)-10*GRID%NGZ
         IF (N3LO<N3L.AND.N3HI<N3L) CYCLE kloop
         IF (N3LO>N3H.AND.N3HI>N3H) CYCLE kloop
      jloop: DO J=-1,1
         POS(2)=T_INFO%POSION(2,NI)+J
         N2LO = INT(POS(2)*GRID%NGY-D2+10*GRID%NGY+.99_q)-10*GRID%NGY
         N2HI = INT(POS(2)*GRID%NGY+D2+10*GRID%NGY)-10*GRID%NGY
         IF (N2LO<N2L.AND.N2HI<N2L) CYCLE jloop
         IF (N2LO>N2H.AND.N2HI>N2H) CYCLE jloop
      iloop: DO I=-1,1
         POS(1)=T_INFO%POSION(1,NI)+I         
         N1LO = INT(POS(1)*GRID%NGX-D1+10*GRID%NGX+.99_q)-10*GRID%NGX
         N1HI = INT(POS(1)*GRID%NGX+D1+10*GRID%NGX)-10*GRID%NGX
         IF (N1LO<N1L.AND.N1HI<N1L) CYCLE iloop
         IF (N1LO>N1H.AND.N1HI>N1H) CYCLE iloop

         NIONS=NIONS+1
         POSION(:,NIONS)=POS(:)
         MAP_(NIONS)=NI
#ifdef debug
         WRITE(*,*) 'map2image reports'
         WRITE(*,'("ion: ",I4,"  prim: ",I4)') NIONS,NI
         WRITE(*,'("POS =",3F12.7)') POS(1:3)
         WRITE(*,'("N1LO =",I4,"  N1HI =",I4)') N1LO,N1HI
         WRITE(*,'("N2LO =",I4,"  N2HI =",I4)') N2LO,N2HI
         WRITE(*,'("N3LO =",I4,"  N3HI =",I4)') N3LO,N3HI
         WRITE(*,*)
#endif
      ENDDO iloop 
      ENDDO jloop
      ENDDO kloop
      
      ENDDO ions
      
      RETURN
      END SUBROUTINE MAP2IMAGE


      SUBROUTINE ONSITE_RSPACE(CPROJ_BRA, CPROJ_KET, WDES, LMDIM, T_INFO, NIONS, MAP_, d_IJ, SUM)

      USE prec
      USE wave
      USE wave_high
      USE poscar

      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (wavedes)     WDES

      INTEGER ISPINOR, NPRO, LMMAXC, NT, IT, NI, NIP, NI_, L, LP, IM
      INTEGER LMDIM,NIONS, MAP_(NIONS)
      GDEF    SUM(0:9)
      GDEF    d_IJ(LMDIM,LMDIM,0:9,NIONS,WDES%NCDIJ)

      GDEF  CPROJ_BRA(WDES%NPROD)
      GDEF  CPROJ_KET(WDES%NPROD)

      spinor: DO ISPINOR=0,WDES%NRSPINORS-1
            
         DO NI=1,NIONS
            NIP=MAP_(NI)
            NT=T_INFO%ITYP(NIP)
            LMMAXC=WDES%LMMAX(NT)
            IF (LMMAXC==0) CYCLE
! Get position in CPROJ array for atom NIP
            NPRO=0; NI_=0
            IF (NT>1) THEN
               DO IT=1,NT-1
                  NPRO=NPRO+WDES%LMMAX(IT)*T_INFO%NITYP(IT)
                  NI_=NI_+T_INFO%NITYP(IT)
               ENDDO
            ENDIF
            NPRO=NPRO+(NIP-NI_-1)*LMMAXC+ISPINOR *WDES%NPRO/2
            DO IM=0,9
               DO L=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                  DO LP=1,LMMAXC
!                    write(71,'(7i7)') ni,map_(ni),nt,lmmaxc,im,l,lp,npro
                     SUM(IM)=SUM(IM)+ d_IJ(LP,L,IM,NI,1+ISPINOR*3)* &
                    &   CPROJ_KET(LP+NPRO)*GCONJG(CPROJ_BRA(L+NPRO))
!                   &   ABS(CPROJ_KET(LP+NPRO)*GCONJG(CPROJ_BRA(L+NPRO)))
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
      ENDDO spinor      

      RETURN
      END SUBROUTINE ONSITE_RSPACE


      SUBROUTINE MNMAT_RSPACE(GRID, GRIDC, GRIDUS, C_TO_US, &
     &   LATT_CUR, T_INFO, W, WDES, P, IRDMAX, LMDIM, LOVERL, &
     &   NK, ISP, NOCC, R, IU6)

      USE sym_prec
      USE constant
      USE mgrid
      USE wave_high
      USE lattice
      USE poscar
      USE pseudo

      IMPLICIT NONE 

! maximal number of bands done at a time
      INTEGER, PARAMETER  :: NSTRIPD=16

! passed variables
      TYPE (wavedes)    WDES
      TYPE (wavespin)   W
      TYPE (grid_3d)    GRID,GRIDC,GRIDUS
      TYPE (transit)    C_TO_US
      TYPE (latt)       LATT_CUR
      TYPE (type_info)  T_INFO
      TYPE (potcar)     P(T_INFO%NTYP)

      INTEGER NK,ISP,IU6,NOCC
      LOGICAL LOVERL
      REAL(q) R(3,NOCC) ! Centroid positions of the (occupied) wannier functions

! local variables
      TYPE (wavedes1) WDES1

      INTEGER ISPINOR
      INTEGER IM,NIM
      INTEGER NIONS_,MAP_(8*T_INFO%NIONS)
      INTEGER N,M,NB_GLOBAL,MB_GLOBAL
      INTEGER N1,N2,N3,N1P,N2P,N3P,N1L,N1H,N2L,N2H,N3L,N3H
      INTEGER NODE_ME,NODE_ME_I,IONODE,NCPU,NSTRIP,NSTRIP_ACT,NLOC,NPOS,NDONE
      INTEGER LMDIM,IRDMAX,IND,IERR,IDUM
      
      REAL(q) RINPL
      REAL(q) F1,F2,F3
      REAL(q) X,Y,Z
      REAL(q) A1,A2,A3,A1L0,A1H0,A2L0,A2H0,A3L0,A3H0,A1L1,A1H1,A2L1,A2H1,A3L1,A3H1
      REAL(q) R_(3,0:8),POSION_(3,8*T_INFO%NIONS)
      GDEF    SUM(0:9),OVL

      TYPE (wavedes1), TARGET :: WDES_LOC,WDES_STRIP
      TYPE (wavefun1) :: WLOC
      TYPE (wavefun1), ALLOCATABLE :: WSTRIP(:)

      GDEF, ALLOCATABLE :: d_IJ(:,:,:,:,:)

      GDEF  EXPVAL(0:9,NOCC,NOCC,8)

      LOGICAL DO_REDIS

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      NODE_ME_I=WDES%COMM_INTER%NODE_ME
      IONODE=WDES%COMM%IONODE
      NCPU=WDES%COMM_INTER%NCPU              ! number of band groups
#else
      NODE_ME=0
      NODE_ME_I=1
      IONODE=0
      NCPU=1
#endif
! determine whether redistribution is required
      IF (NCPU /= 1) THEN                    ! more than one band-group
         DO_REDIS=.TRUE.
      ELSE 
         DO_REDIS=.FALSE.
      ENDIF

! allocate temporary wavefuncs
      NSTRIP=MAX(MIN(NSTRIPD,WDES%NBANDS),1)
      
      CALL SETWDES(WDES,WDES_LOC,NK)
      CALL NEWWAV(WLOC,WDES_LOC,.TRUE.)
      
      CALL SETWDES(WDES,WDES_STRIP,NK)
      ALLOCATE(WSTRIP(NSTRIP*NCPU))
      DO N=1,NSTRIP*NCPU
         CALL NEWWAV(WSTRIP(N),WDES_STRIP,.TRUE.)
      ENDDO

! set some more variables
      RINPL=1._q/GRID%NPLWV

      F1=1._q/GRID%NGX
      F2=1._q/GRID%NGY
      F3=1._q/GRID%NGZ

! test
      write(*,*) 'grid%rl%nfast=',grid%rl%nfast
! test            
!=============================================================================
!    spin: DO ISP=1,WDES%ISPIN
!=============================================================================
      EXPVAL=0

! loop over all blocks, NPOS is the first band in the current block
      NDONE=0
      block: DO NPOS=1,WDES%NBANDS,NSTRIP
         IF (NDONE>=NOCC) CYCLE block

! set NSTRIP_ACT and NLOC to new values in case we are at the last strip
         NSTRIP_ACT=MIN(WDES%NBANDS-NPOS+1,NSTRIP)
         NLOC=NSTRIP_ACT*NCPU

! fourier transform the local bands of the strip into array CWR
! consider the round-robin layout of the bands
         DO N=NPOS,NPOS+NSTRIP_ACT-1
            CALL W1_COPY( ELEMENT(W,WDES_STRIP,N,ISP),WSTRIP((N-NPOS)*NCPU+NODE_ME_I) )
            CALL FFTWAV_W1( WSTRIP((N-NPOS)*NCPU+NODE_ME_I) )
         ENDDO

! if redistribution is required communicate the transformed wavefunctions
! and corresponding projectors to all nodes (bcast)
#ifdef MPI
         IF (DO_REDIS) THEN
            DO N=1,NLOC
               CALL M_bcast_z_from(WDES_STRIP%COMM_INTER,WSTRIP(N)%CR(1), &
                  GRID%MPLWV*W%WDES%NRSPINORS,MOD(N-1,NCPU)+1)
#ifndef gammareal
               IF (W%WDES%LOVERL) CALL M_bcast_z_from(WDES_STRIP%COMM_INTER,WSTRIP(N)%CPROJ(1), &
                  W%WDES%NPROD,MOD(N-1,NCPU)+1)
#else
               IF (W%WDES%LOVERL) CALL M_bcast_d_from(WDES_STRIP%COMM_INTER,WSTRIP(N)%CPROJ(1), &
                  W%WDES%NPROD,MOD(N-1,NCPU)+1)
#endif
            ENDDO
         ENDIF
#endif

!-----------------------------------------------------------------------------
! main 
!-----------------------------------------------------------------------------

! loop over local bands (M-loop)
         lband: DO M=1,WDES%NBANDS
            MB_GLOBAL=(M-1)*NCPU+NODE_ME_I
            IF (MB_GLOBAL>NOCC) CYCLE lband

            CALL W1_COPY( ELEMENT(W,WDES_LOC,M,ISP),WLOC )
            CALL FFTWAV_W1( WLOC )

! loop over bands in strip (N-loop)
            sband: DO N=1,NLOC
               NB_GLOBAL=NDONE+N
               IF (NB_GLOBAL>NOCC) CYCLE sband
               
               CALL GET_IMAGES(GRID,LATT_CUR,R(:,MB_GLOBAL),R(:,NB_GLOBAL),NIM,R_)
#ifdef debug
               WRITE(*,*) 'bra:',MB_GLOBAL,' ket:',NB_GLOBAL,' # of images:',NIM
               DO IM=0,NIM
                  WRITE(*,'(I4,3F8.3)') IM,R_(1:3,IM)
               ENDDO
#endif
               images: DO IM=1,NIM
               SUM=0
               spinor: DO ISPINOR=0,WDES%NRSPINORS-1
      
               A1L0=R_(1,0)-0.5_q; A1H0=R_(1,0)+0.5_q-1._q/GRID%NGX
               A2L0=R_(2,0)-0.5_q; A2H0=R_(2,0)+0.5_q-1._q/GRID%NGY
               A3L0=R_(3,0)-0.5_q; A3H0=R_(3,0)+0.5_q-1._q/GRID%NGZ
      
               A1L1=R_(1,IM)-0.5_q; A1H1=R_(1,IM)+0.5_q-1._q/GRID%NGX
               A2L1=R_(2,IM)-0.5_q; A2H1=R_(2,IM)+0.5_q-1._q/GRID%NGY
               A3L1=R_(3,IM)-0.5_q; A3H1=R_(3,IM)+0.5_q-1._q/GRID%NGZ
#ifdef debug
               WRITE(*,*) 'Cell boundaries'
               WRITE(*,'("A1L0=",F12.7,"  A1H0=",F12.7)') A1L0,A1H0
               WRITE(*,'("A2L0=",F12.7,"  A2H0=",F12.7)') A2L0,A2H0
               WRITE(*,'("A3L0=",F12.7,"  A3H0=",F12.7)') A3L0,A3H0
               WRITE(*,*)
               WRITE(*,'("A1L1=",F12.7,"  A1H1=",F12.7)') A1L1,A1H1
               WRITE(*,'("A2L1=",F12.7,"  A2H1=",F12.7)') A2L1,A2H1
               WRITE(*,'("A3L1=",F12.7,"  A3H1=",F12.7)') A3L1,A3H1
#endif      
               IF (A1L0<A1L1.AND.A1L1<A1H0) A1L0=A1L1
               IF (A2L0<A2L1.AND.A2L1<A2H0) A2L0=A2L1
               IF (A3L0<A3L1.AND.A3L1<A3H0) A3L0=A3L1

               IF (A1L0<A1H1.AND.A1H1<A1H0) A1H0=A1H1
               IF (A2L0<A2H1.AND.A2H1<A2H0) A2H0=A2H1
               IF (A3L0<A3H1.AND.A3H1<A3H0) A3H0=A3H1

               N3L=INT(A3L0*GRID%NGZ+10*GRID%NGZ)-10*GRID%NGZ
               N2L=INT(A2L0*GRID%NGY+10*GRID%NGY)-10*GRID%NGY
               N1L=INT(A1L0*GRID%NGX+10*GRID%NGX)-10*GRID%NGX
      
               N3H=INT(A3H0*GRID%NGZ+10*GRID%NGZ)-10*GRID%NGZ
               N2H=INT(A2H0*GRID%NGY+10*GRID%NGY)-10*GRID%NGY
               N1H=INT(A1H0*GRID%NGX+10*GRID%NGX)-10*GRID%NGX
               
               SUM=0
#ifdef debug
               WRITE(*,*)
               WRITE(*,*) 'Resulting boundaries'
               WRITE(*,'("A1L0=",F12.7,"  A1H0=",F12.7)') A1L0,A1H0
               WRITE(*,'("A2L0=",F12.7,"  A2H0=",F12.7)') A2L0,A2H0
               WRITE(*,'("A3L0=",F12.7,"  A3H0=",F12.7)') A3L0,A3H0
               WRITE(*,*)
               WRITE(*,*) 'Grid boundaries'
               WRITE(*,'("N1L =",I4,"  N1H =",I4)') N1L,N1H
               WRITE(*,'("N2L =",I4,"  N2H =",I4)') N2L,N2H
               WRITE(*,'("N3L =",I4,"  N3H =",I4)') N3L,N3H
               WRITE(*,*)
               WRITE(*,*) 'Grid info'
               WRITE(*,*) 'NGX=',GRID%NGX,'NGY=',GRID%NGY,'NGZ=',GRID%NGZ
               WRITE(*,*) '# of grid points:',GRID%RL%NP
#endif
               IF (GRID%RL%NFAST==3) THEN
! z is fast index
               DO N2=N2L,N2H
               A2=N2*F2       
               N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

               DO N1=N1L,N1H
               A1=N1*F1
               N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

               DO N3=N3L,N3H
               A3=N3*F3
               N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)
       
               X=A1*LATT_CUR%A(1,1)+A2*LATT_CUR%A(1,2)+A3*LATT_CUR%A(1,3)
               Y=A1*LATT_CUR%A(2,1)+A2*LATT_CUR%A(2,2)+A3*LATT_CUR%A(2,3)
               Z=A1*LATT_CUR%A(3,1)+A2*LATT_CUR%A(3,2)+A3*LATT_CUR%A(3,3)

               IND=N3P+N1P*GRID%NGZ+N2P*GRID%NGZ*GRID%NGX+1+ISPINOR*GRID%MPLWV
       
               OVL=WSTRIP(N)%CR(IND)*CONJG(WLOC%CR(IND))*RINPL
               SUM(0)=SUM(0)+OVL
               SUM(1)=SUM(1)+OVL*X
               SUM(2)=SUM(2)+OVL*Y
               SUM(3)=SUM(3)+OVL*Z
               SUM(4)=SUM(4)+OVL*X*X
               SUM(5)=SUM(5)+OVL*Y*Y
               SUM(6)=SUM(6)+OVL*Z*Z
               SUM(7)=SUM(7)+OVL*X*Y
               SUM(8)=SUM(8)+OVL*X*Z
               SUM(9)=SUM(9)+OVL*Y*Z
               
               ENDDO
               ENDDO
               ENDDO
               
               ELSE
! x is fast index
               DO N3=N3L,N3H
               A3=N3*F3
               N3P=MOD(N3+10*GRID%NGZ,GRID%NGZ)

               DO N2=N2L,N2H
               A2=N2*F2       
               N2P=MOD(N2+10*GRID%NGY,GRID%NGY)
       
               DO N1=N1L,N1H
               A1=N1*F1
               N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

               X=A1*LATT_CUR%A(1,1)+A2*LATT_CUR%A(1,2)+A3*LATT_CUR%A(1,3)
               Y=A1*LATT_CUR%A(2,1)+A2*LATT_CUR%A(2,2)+A3*LATT_CUR%A(2,3)
               Z=A1*LATT_CUR%A(3,1)+A2*LATT_CUR%A(3,2)+A3*LATT_CUR%A(3,3)

               IND=N1P+N2P*GRID%NGX+N3P*GRID%NGX*GRID%NGY+1+ISPINOR*GRID%MPLWV
                      
               OVL=WSTRIP(N)%CR(IND)*CONJG(WLOC%CR(IND))*RINPL
               SUM(0)=SUM(0)+OVL
               SUM(1)=SUM(1)+OVL*X
               SUM(2)=SUM(2)+OVL*Y
               SUM(3)=SUM(3)+OVL*Z
               SUM(4)=SUM(4)+OVL*X*X
               SUM(5)=SUM(5)+OVL*Y*Y
               SUM(6)=SUM(6)+OVL*Z*Z
               SUM(7)=SUM(7)+OVL*X*Y
               SUM(8)=SUM(8)+OVL*X*Z
               SUM(9)=SUM(9)+OVL*Y*Z
       
               ENDDO
               ENDDO
               ENDDO

               ENDIF
               
               ENDDO spinor
!              write(*,'(3F20.5)') sum(0)
!              write(*,'(3F20.5,/,3F20.5,/,3F20.5)') sum(1:9)

! add augmentation part
               IF (W%WDES%LOVERL) THEN
                  A1L0=R_(1,0)-0.5_q; A1H0=R_(1,0)+0.5_q-1._q/GRIDC%NGX
                  A2L0=R_(2,0)-0.5_q; A2H0=R_(2,0)+0.5_q-1._q/GRIDC%NGY
                  A3L0=R_(3,0)-0.5_q; A3H0=R_(3,0)+0.5_q-1._q/GRIDC%NGZ
      
                  A1L1=R_(1,IM)-0.5_q; A1H1=R_(1,IM)+0.5_q-1._q/GRIDC%NGX
                  A2L1=R_(2,IM)-0.5_q; A2H1=R_(2,IM)+0.5_q-1._q/GRIDC%NGY
                  A3L1=R_(3,IM)-0.5_q; A3H1=R_(3,IM)+0.5_q-1._q/GRIDC%NGZ
#ifdef debug
                  WRITE(*,*) 'Cell boundaries'
                  WRITE(*,'("A1L0=",F12.7,"  A1H0=",F12.7)') A1L0,A1H0
                  WRITE(*,'("A2L0=",F12.7,"  A2H0=",F12.7)') A2L0,A2H0
                  WRITE(*,'("A3L0=",F12.7,"  A3H0=",F12.7)') A3L0,A3H0
                  WRITE(*,*)
                  WRITE(*,'("A1L1=",F12.7,"  A1H1=",F12.7)') A1L1,A1H1
                  WRITE(*,'("A2L1=",F12.7,"  A2H1=",F12.7)') A2L1,A2H1
                  WRITE(*,'("A3L1=",F12.7,"  A3H1=",F12.7)') A3L1,A3H1
#endif      
                  IF (A1L0<A1L1.AND.A1L1<A1H0) A1L0=A1L1
                  IF (A2L0<A2L1.AND.A2L1<A2H0) A2L0=A2L1
                  IF (A3L0<A3L1.AND.A3L1<A3H0) A3L0=A3L1

                  IF (A1L0<A1H1.AND.A1H1<A1H0) A1H0=A1H1
                  IF (A2L0<A2H1.AND.A2H1<A2H0) A2H0=A2H1
                  IF (A3L0<A3H1.AND.A3H1<A3H0) A3H0=A3H1

                  N3L=INT(A3L0*GRIDC%NGZ+10*GRIDC%NGZ)-10*GRIDC%NGZ
                  N2L=INT(A2L0*GRIDC%NGY+10*GRIDC%NGY)-10*GRIDC%NGY
                  N1L=INT(A1L0*GRIDC%NGX+10*GRIDC%NGX)-10*GRIDC%NGX
      
                  N3H=INT(A3H0*GRIDC%NGZ+10*GRIDC%NGZ)-10*GRIDC%NGZ
                  N2H=INT(A2H0*GRIDC%NGY+10*GRIDC%NGY)-10*GRIDC%NGY
                  N1H=INT(A1H0*GRIDC%NGX+10*GRIDC%NGX)-10*GRIDC%NGX
#ifdef debug
                  WRITE(*,*)
                  WRITE(*,*) 'Resulting boundaries'
                  WRITE(*,'("A1L0=",F12.7,"  A1H0=",F12.7)') A1L0,A1H0
                  WRITE(*,'("A2L0=",F12.7,"  A2H0=",F12.7)') A2L0,A2H0
                  WRITE(*,'("A3L0=",F12.7,"  A3H0=",F12.7)') A3L0,A3H0
                  WRITE(*,*)
                  WRITE(*,*) 'Grid boundaries'
                  WRITE(*,'("N1L =",I4,"  N1H =",I4)') N1L,N1H
                  WRITE(*,'("N2L =",I4,"  N2H =",I4)') N2L,N2H
                  WRITE(*,'("N3L =",I4,"  N3H =",I4)') N3L,N3H
                  WRITE(*,*)
                  WRITE(*,*) 'GridC info'
                  WRITE(*,*) 'NGX=',GRIDC%NGX,'NGY=',GRIDC%NGY,'NGZ=',GRIDC%NGZ
                  WRITE(*,*) '# of grid points:',GRIDC%RL%NP
#endif
                  CALL MAP2IMAGE(GRIDC,LATT_CUR,T_INFO,P,N1L,N1H,N2L,N2H,N3L,N3H, &
                 &   NIONS_,POSION_,MAP_)
#ifdef debug
                  WRITE(*,*) 'ions mapped to current image'
                  DO N1=1,NIONS_
                     WRITE(*,'(I4,I4,3F8.3)') N1,MAP_(N1),POSION_(1:3,N1)
                  ENDDO
#endif
! calculate <Y_lm|O|Y_l'm'> where O=x,y,z,xx,yy,zz,xy,xz, and yz
                  IF (NIONS_>0) THEN
                  ALLOCATE(d_IJ(LMDIM,LMDIM,0:9,NIONS_,WDES%NCDIJ))
                  
                  CALL SET_DIJ_WANNIER_RSPACE(WDES,GRIDC,GRIDUS,C_TO_US, &
                 &   LATT_CUR,P,T_INFO,LOVERL,LMDIM,IRDMAX, &
                 &   N1L,N1H,N2L,N2H,N3L,N3H,NIONS_,POSION_(1:3,1:NIONS_),MAP_(1:NIONS_),d_IJ)

                  CALL ONSITE_RSPACE(WLOC%CPROJ(:),WSTRIP(N)%CPROJ(:),WDES,LMDIM, &
                 &   T_INFO,NIONS_,MAP_(1:NIONS_),d_IJ,SUM)

                  DEALLOCATE(d_IJ)
                  ENDIF
               ENDIF

               EXPVAL(:,MB_GLOBAL,NB_GLOBAL,IM)=SUM(:)
#ifdef debug
               WRITE(*,'(I4," ",6I4,F20.5)') IM,N1L,N1H,N2L,N2H,N3L,N3H,EXPVAL(0,MB_GLOBAL,NB_GLOBAL,IM)
!              write(*,'(3F20.5)') sum(0)
               write(*,'(3F20.5,/,3F20.5,/,3F20.5)') sum(1:9)
#endif
               ENDDO images
#ifdef debug
               WRITE(*,*)
#endif
            ENDDO sband
         ENDDO lband
       
      NDONE=NDONE+NLOC

      ENDDO block

! Communicate
      CALLMPI( M_sum_g(WDES%COMM_INTER,EXPVAL(0,1,1,1),10*NOCC*NOCC*8) )

      io_begin
      DO N=1,NOCC
      DO M=1,NOCC
         WRITE(*,'(I4,I4,4F10.5)') N,M,EXPVAL(0:3,N,M,1)
      ENDDO
      ENDDO
      io_end

!=============================================================================
!      ENDDO spin
!=============================================================================

! deallocate temporary wavefuncs
      CALL DELWAV(WLOC,.TRUE.)
      DO N=1,NSTRIP*NCPU
         CALL DELWAV(WSTRIP(N),.TRUE.)
      ENDDO
      DEALLOCATE(WSTRIP)
      
      RETURN
      END SUBROUTINE MNMAT_RSPACE      


      END MODULE WANNIER
