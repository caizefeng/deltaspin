#include "symbol.inc"

!*********************************************************************
!
!module pade_fit includes some routines for the analytic continuation 
!of diagonal elements of imaginary frequency-dependent Sigma(iw) to 
!the real axis, and calculate the QP energies and spectral functions
!
!*********************************************************************
MODULE pade_fit
   USE prec
   USE wave
   USE vaspxml
   IMPLICIT NONE
   REAL(q),PARAMETER           :: ZMIN=0.60_q
   REAL(q),PARAMETER           :: ZMAX=1.05_q
   LOGICAL,SAVE                :: LPADE_FIT               !whether you want to show the fitted G and Sigma 
   INTEGER,SAVE                :: NBANDS_OUT              !how many bands do you want to output for comparison
   INTEGER,SAVE                :: KPTS_OUT                !how many k-points do you want to output for comparison
   INTEGER,SAVE                :: ISP_OUT                 !which spin channel ISP is going to be outputed
   INTEGER,SAVE                :: NOMEGA_OUT              !which OMEGA is going to be outputed
   INTEGER,ALLOCATABLE,SAVE    :: NBANDS_INDEX(:)         !which band index do you want to output
   INTEGER,ALLOCATABLE,SAVE    :: KPTS_INDEX(:)           !which kpoint index do you want to output
   INTEGER,ALLOCATABLE,SAVE    :: ISP_INDEX(:)            !which spin index do you want to output
   INTEGER,ALLOCATABLE,SAVE    :: NOMEGA_INDEX(:)         !which OMEGA index do you want to output
   COMPLEX(q),ALLOCATABLE,SAVE :: SIGMA(:,:,:)            !SIGMA(NBANDSGW,NKPTS_IRZ,ISPIN) self-energy at DFT eigenvalue
   COMPLEX(q),ALLOCATABLE,SAVE :: PADE_COEFF_UP(:,:,:,:)  !PADE_COEFF_UP(NOMEGA,NBANDSGW,NKPTS_IRZ,ISPIN)
   COMPLEX(q),ALLOCATABLE,SAVE :: PADE_COEFF_DN(:,:,:,:)  !PADE_COEFF_DN(NOMEGA,NBANDSGW,NKPTS_IRZ,ISPIN)
   COMPLEX(q),ALLOCATABLE,SAVE :: QP_ZEROS(:,:,:,:)       !QP_ZEROS(2,NBANDSGW,NKPTS_IRZ,ISPIN)
   REAL(q),ALLOCATABLE,SAVE    :: QP_LINEAR(:,:,:,:)      !QP_LINEAR(2,NBANDSGW,NKPTS_IRZ,ISPIN)
   REAL(q),ALLOCATABLE,SAVE    :: QP_DIFF(:,:,:)          !QP_DIFF(NBANDSGW,NKPTS_IRZ,ISPIN)

 
CONTAINS


!*********************************************************************
!
! reading related parameters from INCAR for pade fit
!
!*********************************************************************
SUBROUTINE QP_PADE_READER(W,IU5,IU0,IU6,NBANDSGW,NKPTS_IRZ,IMAG_GRIDS)
  USE minimax
  IMPLICIT NONE
  TYPE (wavespin)   :: W
  INTEGER           :: IU5,IU0,IU6
  INTEGER           :: NBANDSGW                   !number of bands to be written to file
  INTEGER           :: NKPTS_IRZ                  !number of KPOINTS in the IBZ
  TYPE(imag_grid_handle),OPTIONAL :: IMAG_GRIDS   !time and frequency grid handle
  !local
  INTEGER           :: I,IDUM,N,IERR
  REAL(q)           :: RDUM
  COMPLEX(q)        :: CDUM
  LOGICAL           :: LOPEN, LDUM
  CHARACTER(1)      :: CHARAC
  CHARACTER(40)     :: STRING
  CHARACTER(LEN=10) :: INCAR='INCAR'

  LOPEN=.FALSE.
  OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

  NBANDS_OUT=NBANDSGW
  CALL RDATAB(LOPEN,INCAR,IU5,'NBANDS_OUT','=','#',';','I', &
      &            NBANDS_OUT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<1))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''NBANDS_OUT'' from file INCAR. We set NBANDS_OUT=NBANDSGW'
     NBANDS_OUT=NBANDSGW
  ENDIF
  IF (IU0>0 .AND. NBANDS_OUT>NBANDSGW) THEN
      WRITE(IU0,*)'Error:NBANDS_OUT should be <= 2*NELECT'
      NBANDS_OUT=NBANDSGW
  ENDIF
  CALL XML_INCAR('NBANDS_OUT','I',NBANDS_OUT,RDUM,CDUM,LDUM,CHARAC,1)
  IF (ALLOCATED(NBANDS_INDEX)) DEALLOCATE(NBANDS_INDEX)
  ALLOCATE(NBANDS_INDEX(NBANDS_OUT))

  DO I=1,NBANDS_OUT 
     NBANDS_INDEX(I)=I
  ENDDO
  CALL RDATAB(LOPEN,INCAR,IU5,'NBANDS_INDEX','=','#',';','I', &
      &            NBANDS_INDEX,RDUM,CDUM,LDUM,CHARAC,N,NBANDS_OUT,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<NBANDS_OUT))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''NBANDS_INDEX'' from file INCAR.'
     DO I=1,NBANDS_OUT 
        NBANDS_INDEX(I)=I
     ENDDO
  ENDIF
  CALL XML_INCAR_V('NBANDS_INDEX','I',NBANDS_INDEX,RDUM,CDUM,LDUM,CHARAC,NBANDS_OUT)

  KPTS_OUT=NKPTS_IRZ  !by default
  CALL RDATAB(LOPEN,INCAR,IU5,'KPTS_OUT','=','#',';','I', &
      &            KPTS_OUT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<1))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'ERROR ''KPTS_OUT'' in INCAR, We set KPTS_OUT=NKPTS_IRZ.'
     KPTS_OUT=NKPTS_IRZ
  ENDIF
  IF (IU0>0 .AND. KPTS_OUT>NKPTS_IRZ) THEN
      WRITE(IU0,*)'Error:KPTS_OUT should be <= NKPTS_IRZ'
      KPTS_OUT=NKPTS_IRZ
  ENDIF
  CALL XML_INCAR('KPTS_OUT','I',KPTS_OUT,RDUM,CDUM,LDUM,CHARAC,1)
  IF (ALLOCATED(KPTS_INDEX)) DEALLOCATE(KPTS_INDEX)
  ALLOCATE(KPTS_INDEX(KPTS_OUT))

  DO I=1,KPTS_OUT
     KPTS_INDEX(I)=I
  ENDDO
  CALL RDATAB(LOPEN,INCAR,IU5,'KPTS_INDEX','=','#',';','I', & 
      &            KPTS_INDEX,RDUM,CDUM,LDUM,CHARAC,N,KPTS_OUT,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<KPTS_OUT))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''KPTS_INDEX'' from file INCAR.'
     DO I=1,KPTS_OUT
        KPTS_INDEX(I)=I
     ENDDO
  ENDIF
  CALL XML_INCAR_V('KPTS_INDEX','I',KPTS_INDEX,RDUM,CDUM,LDUM,CHARAC,KPTS_OUT)

  ISP_OUT=W%WDES%ISPIN  !by default
  CALL RDATAB(LOPEN,INCAR,IU5,'ISP_OUT','=','#',';','I', &
      &            ISP_OUT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<1))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'ERROR ''ISP_OUT'' in INCAR, We set ISP_OUT=W%WDES%ISPIN.'
     ISP_OUT=W%WDES%ISPIN
  ENDIF
  IF (IU0>0 .AND. ISP_OUT>W%WDES%ISPIN) THEN
      WRITE(IU0,*)'Error:ISP_OUT should be <= W%WDES%ISPIN'
      ISP_OUT=W%WDES%ISPIN
  ENDIF
  CALL XML_INCAR('ISP_OUT','I',ISP_OUT,RDUM,CDUM,LDUM,CHARAC,1)
  IF (ALLOCATED(ISP_INDEX)) DEALLOCATE(ISP_INDEX)
  ALLOCATE(ISP_INDEX(ISP_OUT))

  DO I=1,ISP_OUT
     ISP_INDEX(I)=I
  ENDDO
  CALL RDATAB(LOPEN,INCAR,IU5,'ISP_INDEX','=','#',';','I', & 
      &            ISP_INDEX,RDUM,CDUM,LDUM,CHARAC,N,ISP_OUT,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<ISP_OUT))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''ISP_INDEX'' from file INCAR.'
     DO I=1,ISP_OUT
        ISP_INDEX(I)=I
     ENDDO
  ENDIF
  CALL XML_INCAR_V('ISP_INDEX','I',ISP_INDEX,RDUM,CDUM,LDUM,CHARAC,ISP_OUT)

IF (PRESENT(IMAG_GRIDS)) THEN
  NOMEGA_OUT=IMAG_GRIDS%NOMEGA  !by default
  CALL RDATAB(LOPEN,INCAR,IU5,'NOMEGA_OUT','=','#',';','I', &
      &            NOMEGA_OUT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<1))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'ERROR ''NOMEGA_OUT'' in INCAR, We set NOMEGA_OUT=IMAG_GRIDS%NOMEGA.'
     NOMEGA_OUT=IMAG_GRIDS%NOMEGA
  ENDIF
  IF (IU0>0 .AND. NOMEGA_OUT>IMAG_GRIDS%NOMEGA) THEN
      WRITE(IU0,*)'Error:NOMEGA_OUT should be <= IMAG_GRIDS%NOMEGA'
      NOMEGA_OUT=IMAG_GRIDS%NOMEGA
  ENDIF
  CALL XML_INCAR('NOMEGA_OUT','I',NOMEGA_OUT,RDUM,CDUM,LDUM,CHARAC,1)
  IF (ALLOCATED(NOMEGA_INDEX)) DEALLOCATE(NOMEGA_INDEX)
  ALLOCATE(NOMEGA_INDEX(NOMEGA_OUT))

  DO I=1,NOMEGA_OUT
     NOMEGA_INDEX(I)=I
  ENDDO
  CALL RDATAB(LOPEN,INCAR,IU5,'NOMEGA_INDEX','=','#',';','I', & 
      &            NOMEGA_INDEX,RDUM,CDUM,LDUM,CHARAC,N,NOMEGA_OUT,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<NOMEGA_OUT))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''NOMEGA_INDEX'' from file INCAR.'
     DO I=1,NOMEGA_OUT
        NOMEGA_INDEX(I)=I
     ENDDO
  ENDIF
  CALL XML_INCAR_V('NOMEGA_INDEX','I',NOMEGA_INDEX,RDUM,CDUM,LDUM,CHARAC,NOMEGA_OUT)
ENDIF

  LPADE_FIT=.FALSE.
  CALL RDATAB(LOPEN,INCAR,IU5,'LPADE_FIT','=','#',';','L', &
   &            IDUM,RDUM,CDUM,LPADE_FIT,CHARAC,N,1,IERR)
  IF (((IERR/=0).AND.(IERR/=3)).OR. ((IERR==0).AND.(N<1))) THEN
     IF (IU0>=0)  WRITE(IU0,*)'Error reading item ''LPADE_FIT'' from file INCAR.'
     LPADE_FIT=.FALSE.
  ENDIF
  CALL XML_INCAR('LPADE_FIT','L',IDUM,RDUM,CDUM,LPADE_FIT,CHARAC,1)

END SUBROUTINE QP_PADE_READER


!*********************************************************************
!
!Main routine to do the Pade fit for G0W0R, GW0R 
!
!*********************************************************************
SUBROUTINE QP_PADE_FIT(W,WMEAN,FREQCOS,EFERMI,SIGMA_COS,SIGMA_SIN,NKPTS_IRZ,NBANDSGW,IU6,IU0)
    USE mpimy
    IMPLICIT NONE
    TYPE(wavespin)  :: W                       !wave function 
    TYPE(wavespin)  :: WMEAN                   !identical to W, except for CELTOT and FERTOT
    REAL(q)         :: FREQCOS(:)              !frequecny
    REAL(q)         :: EFERMI(W%WDES%ISPIN)    !fermi energy
    COMPLEX(q)      :: SIGMA_COS(:,:,:,:)      !diagonal of cos transformed Self-energy
    COMPLEX(q)      :: SIGMA_SIN(:,:,:,:)      !diagonal of sin transformed Self-energy
    INTEGER         :: NKPTS_IRZ               !number of KPOINTS in the IBZ
    INTEGER         :: NBANDSGW                !number of bands to be written to file
    INTEGER         :: IU6                     !for output OUTCAR
    INTEGER         :: IU0
    !local variables 
    INTEGER,PARAMETER      :: NP=5             !NP-point finite difference stencil
    INTEGER,PARAMETER      :: NADD=2
    COMPLEX(q),ALLOCATABLE :: IW_UP(:),SELF_IW_UP(:)
    COMPLEX(q),ALLOCATABLE :: IW_DN(:),SELF_IW_DN(:)
    COMPLEX(q)             :: EQP_TMP,E1,E2
    COMPLEX(q)             :: SIGMA_TMP1,SIGMA_TMP2
    REAL(q)                :: QP_TMP1(2), QP_TMP2(2)
    REAL(q)                :: RETMP,IMTMP,Z1,Z2,KK,Z_MEAN1,Z_MEAN2
    INTEGER                :: NODE_ME,IONODE,NCPU
    INTEGER                :: NOMEGA,IND,NOMEGA_TMP
    INTEGER                :: I,J,K,L,NN
    INTEGER                :: NB,NK,ISP
    REAL(q),ALLOCATABLE    :: QP_ZZ(:,:,:,:)      
    REAL(q),ALLOCATABLE    :: QP_AA(:,:,:,:)      
    COMPLEX(q),ALLOCATABLE :: SIGMA_ZZ(:,:,:)     
    COMPLEX(q),ALLOCATABLE :: SIGMA_AA(:,:,:)      

    NOMEGA=SIZE(SIGMA_COS,1)

    IF ( ALLOCATED(QP_ZZ) )  DEALLOCATE(QP_ZZ)
    ALLOCATE( QP_ZZ(2,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    QP_ZZ=0.0_q

    IF ( ALLOCATED(QP_AA) )  DEALLOCATE(QP_AA)
    ALLOCATE( QP_AA(2,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    QP_AA=0.0_q

    IF ( ALLOCATED(QP_LINEAR) )  DEALLOCATE(QP_LINEAR)
    ALLOCATE( QP_LINEAR(2,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    QP_LINEAR=0.0_q

    IF ( ALLOCATED(QP_DIFF) )  DEALLOCATE(QP_DIFF)
    ALLOCATE( QP_DIFF(NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    QP_DIFF=0.0_q

    IF ( ALLOCATED(QP_ZEROS)  )  DEALLOCATE(QP_ZEROS)
    ALLOCATE( QP_ZEROS(2,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    QP_ZEROS=(0.0_q,0.0_q)

    IF ( ALLOCATED(PADE_COEFF_UP))  DEALLOCATE(PADE_COEFF_UP)
    ALLOCATE( PADE_COEFF_UP(NOMEGA,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    PADE_COEFF_UP=(0.0_q,0.0_q)

    IF ( ALLOCATED(SIGMA)     )  DEALLOCATE(SIGMA)
    ALLOCATE( SIGMA(NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    SIGMA=(0.0_q,0.0_q)

    IF ( ALLOCATED(SIGMA_ZZ)     )  DEALLOCATE(SIGMA_ZZ)
    ALLOCATE( SIGMA_ZZ(NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    SIGMA_ZZ=(0.0_q,0.0_q)

    IF ( ALLOCATED(SIGMA_AA)     )  DEALLOCATE(SIGMA_AA)
    ALLOCATE( SIGMA_AA(NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
    SIGMA_AA=(0.0_q,0.0_q)

    IF (LPADE_FIT) THEN
      IF ( ALLOCATED(PADE_COEFF_DN))  DEALLOCATE(PADE_COEFF_DN)
      ALLOCATE( PADE_COEFF_DN(NOMEGA,NBANDSGW,NKPTS_IRZ,W%WDES%ISPIN) )
      PADE_COEFF_DN=(0.0_q,0.0_q)
    ENDIF

#ifdef MPI
    NODE_ME= W%WDES%COMM%NODE_ME
    IONODE = W%WDES%COMM%IONODE
    NCPU   = W%WDES%COMM%NCPU
#else
    NODE_ME= 1
    IONODE = 1
    NCPU   = 1
#endif
  
    !number of frequency points 
    ALLOCATE( IW_UP(NOMEGA) )
    ALLOCATE( SELF_IW_UP(NOMEGA) )
    IW_UP=(0.0_q,0.0_q)
    SELF_IW_UP=(0.0_q,0.0_q)

    IF (LPADE_FIT) THEN
       ALLOCATE( IW_DN(NOMEGA) )
       ALLOCATE( SELF_IW_DN(NOMEGA) )
       IW_DN=(0.0_q,0.0_q)
       SELF_IW_DN=(0.0_q,0.0_q)
    ENDIF


    !=====================================================================!
    !first, we pre-caculate NOMEGA linearization case
    !=====================================================================!
    DO ISP=1, W%WDES%ISPIN
      DO NK=1,NKPTS_IRZ
         DO NB=1,NBANDSGW
#ifdef MPI
          IND=NBANDSGW*(NK-1)+NB
          IF ( MOD(IND-1,NCPU) /= (NODE_ME-1) ) CYCLE
#endif
             !store self-energy to auxillary arrays            
             DO I=1,NOMEGA
                !pL: NOTE WMEAN%CELECT stores HF eigenvalues
                RETMP= SIGMA_COS(I,NB,NK,ISP)+WMEAN%CELTOT(NB,NK,ISP)
                !pL: "-1" sign comes from G_unocc(paper)=-G_unocc(VASP)
                IMTMP=-SIGMA_SIN(I,NB,NK,ISP)
                IW_UP(I)     =CMPLX(0.0_q,FREQCOS(I))
                SELF_IW_UP(I)=CMPLX(RETMP,IMTMP)
                IF (LPADE_FIT) THEN
                   IW_DN(I)     =CMPLX(0.0_q,-FREQCOS(I))
                   SELF_IW_DN(I)=CMPLX(RETMP,-IMTMP)
                ENDIF
             ENDDO

             IF (LPADE_FIT) THEN
                !NOTE: negative imaginary axis to negative real axis to avoid the branch cut
                PADE_COEFF_DN(:,NB,NK,ISP)=(0.0_q,0.0_q)
                CALL PADE_FIT_COEFF(NOMEGA,IW_DN(:),SELF_IW_DN(:),PADE_COEFF_DN(:,NB,NK,ISP),IU0)
             ENDIF

             !get QP shift by linearization around E_DFT for NOMEGA case
             NOMEGA_TMP=NOMEGA
             CALL CALCULATE_QP(NOMEGA_TMP,QP_ZZ,SIGMA_ZZ)

         ENDDO !NB
      ENDDO !NK
    ENDDO !ISP

#ifdef MPI
    IF (LPADE_FIT) THEN
       CALL M_sum_z(W%WDES%COMM,PADE_COEFF_DN,SIZE(PADE_COEFF_DN))
    ENDIF
    CALL M_sum_z(W%WDES%COMM,SIGMA_ZZ,SIZE(SIGMA_ZZ))
    CALL M_sum_d(W%WDES%COMM,QP_ZZ,SIZE(QP_ZZ))
#endif
    !=====================================================================!


    !=====================================================================!
    !second, we pre-caculate NOMEGA-1 linearization case
    !=====================================================================!
    DO ISP=1, W%WDES%ISPIN
      DO NK=1,NKPTS_IRZ
         DO NB=1,NBANDSGW
#ifdef MPI
          IND=NBANDSGW*(NK-1)+NB
          IF ( MOD(IND-1,NCPU) /= (NODE_ME-1) ) CYCLE
#endif
             !store self-energy to auxillary arrays            
             DO I=1,NOMEGA
                !pL: NOTE WMEAN%CELECT stores HF eigenvalues
                RETMP= SIGMA_COS(I,NB,NK,ISP)+WMEAN%CELTOT(NB,NK,ISP)
                !pL: "-1" sign comes from G_unocc(paper)=-G_unocc(VASP)
                IMTMP=-SIGMA_SIN(I,NB,NK,ISP)
                IW_UP(I)     =CMPLX(0.0_q,FREQCOS(I))
                SELF_IW_UP(I)=CMPLX(RETMP,IMTMP)
             ENDDO

             !get QP shift by linearization around E_DFT for NOMEGA case
             NOMEGA_TMP=NOMEGA-1
             CALL CALCULATE_QP(NOMEGA_TMP,QP_AA,SIGMA_AA)

         ENDDO !NB
      ENDDO !NK
    ENDDO !ISP

#ifdef MPI
    CALL M_sum_z(W%WDES%COMM,SIGMA_AA,SIZE(SIGMA_AA))
    CALL M_sum_d(W%WDES%COMM,QP_AA,SIZE(QP_AA))
#endif
    !=====================================================================!


    !=====================================================================!
    !third, we compare NOMEGA and NOMEGA-1 cases to select the 'best' one
    !=====================================================================!
    DO ISP=1, W%WDES%ISPIN  
      DO NK=1,NKPTS_IRZ
         DO NB=1,NBANDSGW
#ifdef MPI
          IND=NBANDSGW*(NK-1)+NB
          IF ( MOD(IND-1,NCPU) /= (NODE_ME-1) ) CYCLE
#endif
             !store self-energy to auxillary arrays            
             DO I=1,NOMEGA
                !pL: NOTE WMEAN%CELECT stores HF eigenvalues
                RETMP= SIGMA_COS(I,NB,NK,ISP)+WMEAN%CELTOT(NB,NK,ISP) 
                !pL: "-1" sign comes from G_unocc(paper)=-G_unocc(VASP)
                IMTMP=-SIGMA_SIN(I,NB,NK,ISP)
                IW_UP(I)     =CMPLX(0.0_q,FREQCOS(I))
                SELF_IW_UP(I)=CMPLX(RETMP,IMTMP)
             ENDDO

             !get QP shift by linearization around E_DFT
             QP_LINEAR(:,NB,NK,ISP)=0.0_q

             !for NOMEGA case
             SIGMA_TMP1=SIGMA_ZZ(NB,NK,ISP)
             QP_TMP1(:)=QP_ZZ(:,NB,NK,ISP)

             !for NOMEGA-1 case
             SIGMA_TMP2=SIGMA_AA(NB,NK,ISP)
             QP_TMP2(:)=QP_AA(:,NB,NK,ISP)

             QP_DIFF(NB,NK,ISP)=ABS(QP_TMP1(1)-QP_TMP2(1))


             !determine the mean value of Z factor for band index NB
             IF (NB>=NADD+1 .AND. NB<=NBANDSGW-NADD .AND. NBANDSGW>=2*NADD+1) THEN
                Z_MEAN2=0.0_q
                DO I=NB-NADD,NB+NADD
                  Z_MEAN2=Z_MEAN2+QP_ZZ(2,NB,NK,ISP)
                ENDDO
                Z_MEAN2=Z_MEAN2/REAL(2*NADD+1,q)

                Z_MEAN1=0.0_q
                J=0
                DO I=NB-NADD,NB+NADD
                   IF (ABS(QP_ZZ(2,NB,NK,ISP)-Z_MEAN2)>0.1_q) CYCLE
                   Z_MEAN1=Z_MEAN1+QP_ZZ(2,NB,NK,ISP)
                   J=J+1
                ENDDO
                IF (J/=0) THEN
                  Z_MEAN1=Z_MEAN1/REAL(J,q)
                ELSE
                  Z_MEAN1=0.8_q
                ENDIF
             ELSE
                Z_MEAN1=0.8_q
             ENDIF


             !the criteria is that we try to keep the QP and Z from NOMEGA case as many as possible
             !otherwise we select the one, Z factot of which is closer to the mean value
             IF (0.65_q<=QP_TMP1(2) .AND. QP_TMP1(2)<1.0_q .AND. &
               & QP_DIFF(NB,NK,ISP)<0.5_q .AND. ABS(QP_TMP1(2)-QP_TMP2(2))<0.2_q ) THEN
                QP_LINEAR(:,NB,NK,ISP)=QP_TMP1(:)
                SIGMA(NB,NK,ISP)=SIGMA_TMP1
             ELSE
                IF (ABS(QP_TMP1(2)-Z_MEAN1) <= ABS(QP_TMP2(2)-Z_MEAN1)) THEN
                   QP_LINEAR(:,NB,NK,ISP)=QP_TMP1(:)
                   SIGMA(NB,NK,ISP)=SIGMA_TMP1
                ELSE
                   QP_LINEAR(:,NB,NK,ISP)=QP_TMP2(:)
                   SIGMA(NB,NK,ISP)=SIGMA_TMP2
                ENDIF
             ENDIF

             !if still problematic, we report error and use original NOMEGA one 
             IF (QP_LINEAR(2,NB,NK,ISP)>ZMAX .OR. QP_LINEAR(2,NB,NK,ISP)<ZMIN ) THEN
                IF (IU0>=0) WRITE(IU0,'(A,3I6)') "ERROR: GET_QP_LINEAR in Z", NB,NK,ISP
                SIGMA(NB,NK,ISP)=SIGMA_ZZ(NB,NK,ISP)
                QP_LINEAR(:,NB,NK,ISP)=QP_ZZ(:,NB,NK,ISP)
             ENDIF


             !get QP shift by finding the roots of QP equation using Brent-bisection method
             EQP_TMP=CMPLX(QP_LINEAR(1,NB,NK,ISP),0.0_q)

             NOMEGA_TMP=NOMEGA
             DO WHILE(.TRUE.)
                PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
                CALL PADE_FIT_COEFF(NOMEGA_TMP,IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                     & SELF_IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),IU0)
                QP_ZEROS(:,NB,NK,ISP)=(0.0_q,0.0_q)
                CALL GET_QP_ZEROS_BRENT(NOMEGA_TMP,ISP,NK,NB,EQP_TMP,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                         & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),QP_ZEROS(:,NB,NK,ISP),IU0)
                IF (REAL(QP_ZEROS(2,NB,NK,ISP),q)>ZMAX .OR. REAL(QP_ZEROS(2,NB,NK,ISP),q)<ZMIN ) THEN
                   NOMEGA_TMP=NOMEGA_TMP-1
                ELSE
                   EXIT
                ENDIF
                IF (NOMEGA_TMP<MAX(NOMEGA-6,8) ) THEN
                   IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NOMEGA_TMP<MAX(NOMEGA-6,8) in GET_QP_ZEROS_BRENT",NB,NK,ISP
                   EXIT
                ENDIF
             ENDDO

             !if still problematic, we report error and use original NOMEGA one 
             IF (NOMEGA_TMP<MAX(NOMEGA-6,MIN(8,NOMEGA)) .OR. REAL(QP_ZEROS(2,NB,NK,ISP),q)>ZMAX .OR. REAL(QP_ZEROS(2,NB,NK,ISP),q)<ZMIN ) THEN
                IF (IU0>=0) WRITE(IU0,'(A,3I6)') "ERROR: GET_QP_ZEROS_BRENT in Z ", NB,NK,ISP
                PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
                CALL PADE_FIT_COEFF(NOMEGA,IW_UP,SELF_IW_UP,PADE_COEFF_UP(:,NB,NK,ISP),IU0)
                QP_ZEROS(:,NB,NK,ISP)=(0.0_q,0.0_q)
                CALL GET_QP_ZEROS_BRENT(NOMEGA,ISP,NK,NB,EQP_TMP,EFERMI(ISP),IW_UP,&
                            & PADE_COEFF_UP(:,NB,NK,ISP),QP_ZEROS(:,NB,NK,ISP),IU0)
             ENDIF

             !return to the origianl one for the plot
             PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
             CALL PADE_FIT_COEFF(NOMEGA,IW_UP,SELF_IW_UP,PADE_COEFF_UP(:,NB,NK,ISP),IU0)

         ENDDO !NB
      ENDDO !NK
    ENDDO !ISP

#ifdef MPI
    CALL M_sum_z(W%WDES%COMM,SIGMA,SIZE(SIGMA))
    CALL M_sum_z(W%WDES%COMM,PADE_COEFF_UP,SIZE(PADE_COEFF_UP))
    CALL M_sum_d(W%WDES%COMM,QP_LINEAR,SIZE(QP_LINEAR))
    CALL M_sum_d(W%WDES%COMM,QP_DIFF,SIZE(QP_DIFF))
    CALL M_sum_z(W%WDES%COMM,QP_ZEROS,SIZE(QP_ZEROS))
#endif
    !=====================================================================!


    IF (NODE_ME==1) THEN
      !write out the QP_shift to OUTCAR
      CALL WRITE_QP_SHIFT(W,NKPTS_IRZ,NBANDSGW,IU6)
      !write out the Pade fitted sigma(w) for specified bands and kpoints in vasprun.xml file
      IF (LPADE_FIT) THEN
         CALL WRITE_SIGMA_XML(IW_UP,PADE_COEFF_UP,IW_DN,PADE_COEFF_DN,EFERMI(1),1001,1,IU0)
      ENDIF
    ENDIF  

! gK, keep SIGMA for later use
!    DEALLOCATE(SIGMA,SIGMA_ZZ,QP_ZZ,SIGMA_AA,QP_AA)
    DEALLOCATE(SIGMA_ZZ,QP_ZZ,SIGMA_AA,QP_AA)
    DEALLOCATE(IW_UP,SELF_IW_UP,PADE_COEFF_UP,QP_DIFF)
    IF (LPADE_FIT) DEALLOCATE(IW_DN,SELF_IW_DN,PADE_COEFF_DN)


CONTAINS


!*********************************************************************
!get QP shift by linearization around E_DFT for NOMEGA_TMP case
!*********************************************************************
SUBROUTINE CALCULATE_QP(NOMEGA_TMP,QP_BB,SIGMA_BB)
    IMPLICIT NONE
    INTEGER     :: NOMEGA_TMP
    REAL(q)     :: QP_BB(:,:,:,:)
    COMPLEX(q)  :: SIGMA_BB(:,:,:)
    INTEGER     :: NOMEGA_BACK

    NOMEGA_BACK=NOMEGA_TMP

    PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
    CALL PADE_FIT_COEFF(NOMEGA_TMP,IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
         & SELF_IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),IU0)
    QP_BB(:,NB,NK,ISP)=0.0_q
    CALL GET_QP_LINEAR(NOMEGA_TMP,W%CELTOT(NB,NK,ISP),EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
          & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)


    !if the initial try gives wierd results
    !--------------------------------------------------------------!
    IF (QP_BB(2,NB,NK,ISP)>ZMAX .OR. QP_BB(2,NB,NK,ISP)<ZMIN ) THEN
    !--------------------------------------------------------------!

       !if fails, then try to find a better slope at the position at DFT+delta
       E1=W%CELTOT(NB,NK,ISP)+CMPLX(0.2_q,0.0_q)
       QP_BB(:,NB,NK,ISP)=0.0_q
       CALL GET_QP_LINEAR(NOMEGA_TMP,E1,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
             & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
       Z1=QP_BB(2,NB,NK,ISP)
       NN=0
       DO WHILE(.TRUE.)
          E2=E1+CMPLX(0.2_q,0.0_q)
          QP_BB(:,NB,NK,ISP)=0.0_q
          CALL GET_QP_LINEAR(NOMEGA_TMP,E2,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
          Z2=QP_BB(2,NB,NK,ISP)
 
          !0.03 should be tested
          IF (Z1<ZMAX .AND. Z1>ZMIN .AND. Z2<ZMAX .AND. Z2>ZMIN .AND. ABS(Z2-Z1)<0.03_q) THEN
             EXIT
          ELSE
             E1=E2
             Z1=Z2
             NN=NN+1
             IF (NN>=40) THEN
               IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NN>=40",NB,NK,ISP
               EXIT
             ENDIF
          ENDIF
       ENDDO


       !if still gives wierd results, then try to find a better slope
       !at the position at DFT-delta
       IF (NN>=40 .OR. QP_BB(2,NB,NK,ISP)>ZMAX .OR. QP_BB(2,NB,NK,ISP)<ZMIN ) THEN
          E1=W%CELTOT(NB,NK,ISP)-CMPLX(0.2_q,0.0_q)
          QP_BB(:,NB,NK,ISP)=0.0_q
          CALL GET_QP_LINEAR(NOMEGA_TMP,E1,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
          Z1=QP_BB(2,NB,NK,ISP)
          NN=0
          DO WHILE(.TRUE.)
             E2=E1-CMPLX(0.2_q,0.0_q)
             QP_BB(:,NB,NK,ISP)=0.0_q
             CALL GET_QP_LINEAR(NOMEGA_TMP,E2,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                   & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
             Z2=QP_BB(2,NB,NK,ISP)
 
             !0.03 should be tested
             IF (Z1<ZMAX .AND. Z1>ZMIN .AND. Z2<ZMAX .AND. Z2>ZMIN .AND. ABS(Z2-Z1)<0.03_q) THEN
                EXIT
             ELSE
                E1=E2
                Z1=Z2
                NN=NN+1
                IF (NN>=40) THEN
                  IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NN>=40",NB,NK,ISP
                  EXIT
                ENDIF
             ENDIF
          ENDDO
       ENDIF


       IF (NN<40 .AND. QP_BB(2,NB,NK,ISP)<ZMAX .AND. QP_BB(2,NB,NK,ISP)>ZMIN) THEN
          !here successfull for problematic cases
          KK=1.0_q-1.0_q/QP_BB(2,NB,NK,ISP)
          QP_BB(:,NB,NK,ISP)=0.0_q
          CALL GET_QP_LINEAR_2(NOMEGA_TMP,W%CELTOT(NB,NK,ISP),E2,KK,EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                 & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0)
       ELSE
          !we try to remove the first few frequencies used to fitting
          !until Z is correct
          NOMEGA_TMP=NOMEGA_TMP-1
          DO WHILE(.TRUE.)
            PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
            CALL PADE_FIT_COEFF(NOMEGA_TMP,IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                  & SELF_IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),IU0)
            QP_BB(:,NB,NK,ISP)=0.0_q
            CALL GET_QP_LINEAR(NOMEGA_TMP,W%CELTOT(NB,NK,ISP),EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                   & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
            IF (QP_BB(2,NB,NK,ISP)>ZMAX .OR. QP_BB(2,NB,NK,ISP)<ZMIN ) THEN
               NOMEGA_TMP=NOMEGA_TMP-1
            ELSE
               EXIT
            ENDIF
            IF (NOMEGA_TMP<MAX(NOMEGA-6,8)) THEN
               IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NOMEGA_TMP<MAX(NOMEGA-6,8) in GET_QP_LINEAR",NB,NK,ISP
               EXIT
            ENDIF
          ENDDO
       ENDIF

       !if still problematic, we report error and use original one 
       IF (NOMEGA_TMP<MAX(NOMEGA-6,MIN(8,NOMEGA)) .OR. QP_BB(2,NB,NK,ISP)>ZMAX .OR. QP_BB(2,NB,NK,ISP)<ZMIN ) THEN
          IF (IU0>=0) WRITE(IU0,'(A,3I6)') "ERROR: GET_QP_LINEAR in Z", NB,NK,ISP
          NOMEGA_TMP=NOMEGA_BACK
          PADE_COEFF_UP(:,NB,NK,ISP)=(0.0_q,0.0_q)
          CALL PADE_FIT_COEFF(NOMEGA_TMP,IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
               & SELF_IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),IU0)
          QP_BB(:,NB,NK,ISP)=0.0_q
          CALL GET_QP_LINEAR(NOMEGA_TMP,W%CELTOT(NB,NK,ISP),EFERMI(ISP),IW_UP(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                & PADE_COEFF_UP(1:NOMEGA_TMP,NB,NK,ISP),SIGMA_BB(NB,NK,ISP),QP_BB(:,NB,NK,ISP),IU0,NP)
       ENDIF

    !--------------------------------------------------------------!
    ENDIF
    !--------------------------------------------------------------!
END SUBROUTINE CALCULATE_QP

END SUBROUTINE QP_PADE_FIT


!*********************************************************************
!
!write self-energy to vasprun.xml
!
!*********************************************************************
SUBROUTINE WRITE_SIGMA_IW_XML(ISP,NK,NB,OMEGA,SIGMA_IMAG)
    IMPLICIT NONE
    INTEGER    :: ISP,NK,NB
    REAL(q)    :: OMEGA(:)
    COMPLEX(q) :: SIGMA_IMAG(:)
    !local
    INTEGER    :: NO
    REAL(q),ALLOCATABLE    :: SELF_IW(:,:)
    CHARACTER(len=100)     :: STR

    ALLOCATE(SELF_IW(3,SIZE(OMEGA)))

    IF (ISMEMBER(ISP,ISP_INDEX) .AND. ISMEMBER(NK,KPTS_INDEX) .AND. ISMEMBER(NB,NBANDS_INDEX)) THEN
        DO NO=1,SIZE(OMEGA)
              SELF_IW(1,NO)=OMEGA(NO)
              SELF_IW(2,NO)=REAL(SIGMA_IMAG(NO),q)
              SELF_IW(3,NO)=AIMAG(SIGMA_IMAG(NO))
        ENDDO
        WRITE(STR,"(A,I0,A,I0)") " ISP=1 NK=", NK, " NB=", NB
        CALL XML_VECARRAY("selfenergy along imaginary axis"//TRIM(STR))
        CALL XML_ARRAY_REAL(SELF_IW)
        CALL XML_CLOSE_TAG
    ENDIF
    DEALLOCATE(SELF_IW)
END SUBROUTINE WRITE_SIGMA_IW_XML


!*********************************************************************
!
!*********************************************************************
FUNCTION ISMEMBER(A,B)
 IMPLICIT NONE
 INTEGER :: A,I
 INTEGER :: B(:)
 LOGICAL :: ISMEMBER

 ISMEMBER=.FALSE.
 DO I=1,SIZE(B)
    IF (B(I)==A) THEN
        ISMEMBER=.TRUE.
    ENDIF
 ENDDO
END FUNCTION ISMEMBER
!*********************************************************************


!*********************************************************************
!
!write self-energy to vasprun.xml
!
!*********************************************************************
SUBROUTINE WRITE_SIGMA_XML(IW_UP,COEFF_UP,IW_DN,COEFF_DN,EFERMI,NGRIDS,ISP,IU0)
    IMPLICIT NONE
    COMPLEX(q) :: IW_UP(:), COEFF_UP(:,:,:,:)
    COMPLEX(q) :: IW_DN(:), COEFF_DN(:,:,:,:)
    REAL(q)    :: EFERMI
    INTEGER    :: NGRIDS 
    INTEGER    :: ISP
    INTEGER    :: IU0
    !local
    INTEGER    :: I,J,NK,NB

    DO J=1,KPTS_OUT
       NK=KPTS_INDEX(J)
       DO I=1,NBANDS_OUT
          NB=NBANDS_INDEX(I)
          CALL WRITE_G_SIGMA_FIT(IW_UP,COEFF_UP(:,NB,NK,ISP),IW_DN,COEFF_DN(:,NB,NK,ISP),EFERMI,NGRIDS,NK,NB,IU0)
       ENDDO
    ENDDO

END SUBROUTINE WRITE_SIGMA_XML


!*********************************************************************
!
!this help routine is used to do the analytical continuation by Pade fit 
!in the old GW code  with Merzuk imaginary grids
!
!*********************************************************************
SUBROUTINE QP_PADE_FIT_SIGMA(NB,NK,ISP,OMEGA,SIGMA_IMAG,EDFT,EFERMI,COEFF,SIGMA_QP,QP_LIN,&
                          &  QPDIFF,QP_ZERO,IU0)
    IMPLICIT NONE
    INTEGER         :: NB,NK,ISP
    REAL(q)         :: OMEGA(:)                !imaginary frequency
    COMPLEX(q)      :: SIGMA_IMAG(:)           !self-energy
    REAL(q)         :: EDFT                    !CELNEW
    REAL(q)         :: EFERMI                  !fermi energy
    COMPLEX(q)      :: COEFF(:)                !pade coefficients
    COMPLEX(q)      :: SIGMA_QP                !self-energy at E_DFT
    REAL(q)         :: QP_LIN(2)               !linearized QP energy and Z
    REAL(q)         :: QPDIFF                  !difference between NOMEGA and NOMEGA-1
    COMPLEX(q)      :: QP_ZERO(2)              !QP energy and Z of Zeros solution
    INTEGER         :: IU0
    !local variables 
    INTEGER,PARAMETER      :: NP=5             !NP-point finite difference stencil
    INTEGER                :: NOMEGA,NOMEGA_TMP,NN
    REAL(q)                :: Z1,Z2,KK
    REAL(q)                :: QP_TMP1(2), QP_TMP2(2)
    COMPLEX(q)             :: C_DFT,EQP_TMP,E1,E2
    COMPLEX(q)             :: SIGMA_TMP1,SIGMA_TMP2
    COMPLEX(q),ALLOCATABLE :: IW(:)

    NOMEGA=SIZE(OMEGA)
    ALLOCATE(IW(NOMEGA))

    C_DFT=CMPLX(EDFT,0.0_q,q)
    IW=CMPLX(0.0_q,OMEGA,q)

    !get QP shift by linearization around E_DFT
    !intial try for both NOMEGA and NOMEGA-1 cases, and select the 'best' solution
    QP_LIN(:)=0.0_q

    !for NOMEGA-1 cases
    NOMEGA_TMP=NOMEGA-1
    COEFF(:)=(0.0_q,0.0_q)
    CALL PADE_FIT_COEFF(NOMEGA_TMP,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
         & SIGMA_IMAG(NOMEGA-NOMEGA_TMP+1:NOMEGA),COEFF(1:NOMEGA_TMP),IU0)
    QP_TMP2(:)=0.0_q
    CALL GET_QP_LINEAR(NOMEGA_TMP,C_DFT,EFERMI,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
          & COEFF(1:NOMEGA_TMP),SIGMA_TMP2,QP_TMP2(:),IU0,NP)

    !for NOMEGA cases
    NOMEGA_TMP=NOMEGA
    COEFF(:)=(0.0_q,0.0_q)
    CALL PADE_FIT_COEFF(NOMEGA_TMP,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
         & SIGMA_IMAG(NOMEGA-NOMEGA_TMP+1:NOMEGA),COEFF(1:NOMEGA_TMP),IU0)
    QP_TMP1(:)=0.0_q
    CALL GET_QP_LINEAR(NOMEGA_TMP,C_DFT,EFERMI,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
          & COEFF(1:NOMEGA_TMP),SIGMA_TMP1,QP_TMP1(:),IU0,NP)

    !the criteria is that if Z fator for NOMEGA cases is "reasonable", say [0.65,1.0)
    !otherwise choose the one that is closer to 0.8_q
    IF (0.65_q<=QP_TMP1(2) .AND. QP_TMP1(2)<1.0_q ) THEN
      QP_LIN(:)=QP_TMP1(:)
      SIGMA_QP=SIGMA_TMP1
    ELSE
      IF (ABS(QP_TMP1(2)-0.8_q) <= ABS(QP_TMP2(2)-0.8_q)) THEN
        QP_LIN(:)=QP_TMP1(:)
        SIGMA_QP=SIGMA_TMP1
      ELSE
        QP_LIN(:)=QP_TMP2(:)
        SIGMA_QP=SIGMA_TMP2
      ENDIF
    ENDIF
    QPDIFF=ABS(QP_TMP1(1)-QP_TMP2(1))

    !if the first try gives wierd results, then try to find a better slope
    !at the position DFT+delta
    !=========================================!
    IF (QP_LIN(2)>ZMAX .OR. QP_LIN(2)<ZMIN ) THEN
    !=========================================!

       E1=C_DFT+CMPLX(0.2_q,0.0_q)
       QP_LIN=0.0_q
       CALL GET_QP_LINEAR(NOMEGA,E1,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0,NP)
       Z1=QP_LIN(2)
       NN=0
       DO WHILE(.TRUE.)
          E2=E1+CMPLX(0.2_q,0.0_q)
          QP_LIN=0.0_q
          CALL GET_QP_LINEAR(NOMEGA,E2,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0,NP)
          Z2=QP_LIN(2)

          !0.03 should be tested
          IF (Z1<ZMAX .AND. Z1>ZMIN .AND. Z2<ZMAX .AND. Z2>ZMIN .AND. ABS(Z2-Z1)<0.03_q) THEN
             EXIT
          ELSE
             E1=E2
             Z1=Z2
             NN=NN+1
             IF (NN>=40) THEN
               IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NN>=40",NB,NK,ISP
               EXIT
             ENDIF
          ENDIF
       ENDDO

       !if the second try gives wierd results, then try to find a better slope
       !at the position DFT-delta
       IF (NN>=40 .OR. QP_LIN(2)>ZMAX .OR. QP_LIN(2)<ZMIN) THEN
          E1=C_DFT-CMPLX(0.2_q,0.0_q)
          QP_LIN=0.0_q
          CALL GET_QP_LINEAR(NOMEGA,E1,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0,NP)
          Z1=QP_LIN(2)
          NN=0
          DO WHILE(.TRUE.)
             E2=E1-CMPLX(0.2_q,0.0_q)
             QP_LIN=0.0_q
             CALL GET_QP_LINEAR(NOMEGA,E2,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0,NP)
             Z2=QP_LIN(2)

             !0.03 should be tested
             IF (Z1<ZMAX .AND. Z1>ZMIN .AND. Z2<ZMAX .AND. Z2>ZMIN .AND. ABS(Z2-Z1)<0.03_q) THEN
                EXIT
             ELSE
                E1=E2
                Z1=Z2
                NN=NN+1
                IF (NN>=40) THEN
                  IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NN>=40",NB,NK,ISP
                  EXIT
                ENDIF
             ENDIF
          ENDDO
       ENDIF


       IF (NN<40 .AND. Z2<ZMAX .AND.Z2>ZMIN) THEN
          !here successfull for problematic cases
          KK=1.0_q-1.0_q/Z2
          QP_LIN=0.0_q
          CALL GET_QP_LINEAR_2(NOMEGA,C_DFT,E2,KK,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0)
       ELSE
          !if still fails, we try the last way by descreaing the number of NOMEGA until Z is correct
          NOMEGA_TMP=NOMEGA-2
          DO WHILE(.TRUE.)
            COEFF(:)=(0.0_q,0.0_q)
            CALL PADE_FIT_COEFF(NOMEGA_TMP,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                 & SIGMA_IMAG(NOMEGA-NOMEGA_TMP+1:NOMEGA),COEFF(1:NOMEGA_TMP),IU0)
            QP_LIN=0.0_q
            CALL GET_QP_LINEAR(NOMEGA_TMP,C_DFT,EFERMI,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
                & COEFF(1:NOMEGA_TMP),SIGMA_QP,QP_LIN(:),IU0,NP)
            IF (QP_LIN(2)>ZMAX .OR. QP_LIN(2)<ZMIN ) THEN
               NOMEGA_TMP=NOMEGA_TMP-1
            ELSE
               EXIT
            ENDIF
            IF (NOMEGA_TMP<MAX(NOMEGA-6,8) ) THEN
               IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NOMEGA_TMP<MAX(NOMEGA-6,8) in GET_QP_LINEAR",NB,NK,ISP
               EXIT
            ENDIF
          ENDDO
       ENDIF

       !if still problematic, we report error and use original one
       IF (NOMEGA_TMP<MAX(NOMEGA-6,MIN(8,NOMEGA)) .OR. QP_LIN(2)>ZMAX .OR. QP_LIN(2)<ZMIN ) THEN
          IF (IU0>=0) WRITE(IU0,'(A,3I6)') "ERROR: GET_QP_LINEAR in Z", NB,NK,ISP
          COEFF=(0.0_q,0.0_q)
          CALL PADE_FIT_COEFF(NOMEGA,IW(:),SIGMA_IMAG(:),COEFF(:),IU0)
          QP_LIN=0.0_q
          CALL GET_QP_LINEAR(NOMEGA,C_DFT,EFERMI,IW(:),COEFF(:),SIGMA_QP,QP_LIN(:),IU0,NP)
       ENDIF
    !=========================================!
    ENDIF
    !=========================================!

    !get QP shift by finding the roots of QP equation using Brent-bisection method
    EQP_TMP=CMPLX(QP_LIN(1),0.0_q)

    NOMEGA_TMP=NOMEGA
    DO WHILE(.TRUE.)
       COEFF(:)=(0.0_q,0.0_q)
       CALL PADE_FIT_COEFF(NOMEGA_TMP,IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),&
            & SIGMA_IMAG(NOMEGA-NOMEGA_TMP+1:NOMEGA),COEFF(1:NOMEGA_TMP),IU0)
       QP_ZERO=(0.0_q,0.0_q)
       CALL GET_QP_ZEROS_BRENT(NOMEGA_TMP,ISP,NK,NB,EQP_TMP,EFERMI,&
            & IW(NOMEGA-NOMEGA_TMP+1:NOMEGA),COEFF(1:NOMEGA_TMP),QP_ZERO(:),IU0)
       IF (REAL(QP_ZERO(2),q)>ZMAX .OR. REAL(QP_ZERO(2),q)<ZMIN ) THEN
          NOMEGA_TMP=NOMEGA_TMP-1
       ELSE
          EXIT
       ENDIF
       IF (NOMEGA_TMP<MAX(NOMEGA-6,8) ) THEN
          IF (IU0>=0) WRITE(IU0,'(A,3I6)') "Warning: NOMEGA_TMP<MAX(NOMEGA-6,8) in GET_QP_ZEROS_BRENT",NB,NK,ISP
          EXIT
       ENDIF
    ENDDO

    !if still problematic, we report error and use original one
    IF (NOMEGA_TMP<MAX(NOMEGA-6,MIN(8,NOMEGA)) .OR. REAL(QP_ZERO(2),q)>ZMAX .OR. REAL(QP_ZERO(2),q)<ZMIN ) THEN
       IF (IU0>=0) WRITE(IU0,'(A,3I6)') "ERROR: GET_QP_ZEROS_BRENT in Z", NB,NK,ISP
       COEFF=(0.0_q,0.0_q)
       CALL PADE_FIT_COEFF(NOMEGA,IW(:),SIGMA_IMAG(:),COEFF(:),IU0)
       QP_ZERO=(0.0_q,0.0_q)
       CALL GET_QP_ZEROS_BRENT(NOMEGA,ISP,NK,NB,EQP_TMP,EFERMI,IW(:),COEFF(:),QP_ZERO(:),IU0)
    ENDIF

    !return to the orginal coefficient for plot
    COEFF=(0.0_q,0.0_q)
    CALL PADE_FIT_COEFF(NOMEGA,IW(:),SIGMA_IMAG(:),COEFF(:),IU0)

    DEALLOCATE(IW)
END SUBROUTINE QP_PADE_FIT_SIGMA



!*********************************************************************
!
!write QP energies to OUTCAR
!
!*********************************************************************
SUBROUTINE WRITE_QP_SHIFT(W,NKPTS_IRZ,NBANDSGW,IU6)
    IMPLICIT NONE
    TYPE(wavespin)    :: W                       !wave function 
    INTEGER           :: NKPTS_IRZ,NBANDSGW
    INTEGER           :: ISP,NK,NB,IU6

    DO ISP=1, W%WDES%ISPIN
       IF (W%WDES%ISPIN==2 .AND. IU6>=0) WRITE(IU6,'(/A,I1)') ' spin component ',ISP
       DO NK=1,NKPTS_IRZ  !KPOINT INDEX
          IF (W%WDES%WTKPT(NK)==0) CYCLE ! zero k-point weighted points are useless right now in the GW code
          IF (IU6>=0) THEN
             WRITE(IU6,777) NK,W%WDES%VKPT(1,NK),W%WDES%VKPT(2,NK),W%WDES%VKPT(3,NK)
          ENDIF
          DO NB=1,NBANDSGW  !BAND INDEX
             !write the output to OUTCAR file         
             IF (IU6>=0)  WRITE(IU6,888) NB,REAL(W%CELTOT(NB,NK,ISP),q),REAL(SIGMA(NB,NK,ISP),q),QP_LINEAR(1,NB,NK,ISP),&
                           & QP_LINEAR(2,NB,NK,ISP),REAL(QP_ZEROS(1,NB,NK,ISP),q), REAL(QP_ZEROS(2,NB,NK,ISP),q),&
                           & W%FERTOT(NB,NK,ISP)*W%WDES%RSPIN,AIMAG(QP_ZEROS(1,NB,NK,ISP)), QP_DIFF(NB,NK,ISP)
          ENDDO
       ENDDO
    ENDDO

777  FORMAT(/' k-point ',I3,' :',3X,3F10.4/  &
   & " band No.  KS-energies   sigma(KS)    QP-e(linear)    Z         QP-e(zeros)     Z        occupation    Imag(E_QP)    QP_DIFF "/)
888  FORMAT((3X,I4,3X,9(F10.4,3X)))

END SUBROUTINE WRITE_QP_SHIFT


!*********************************************************************
!
!help routine to write self-energy to vasprun.xml
!
!*********************************************************************
SUBROUTINE WRITE_G_SIGMA_FIT(IW_UP,COEFF_UP,IW_DN,COEFF_DN,EFERMI,NOUT,NK,NB,IU0)
    IMPLICIT NONE
    COMPLEX(q)      :: IW_UP(:),COEFF_UP(:)
    COMPLEX(q)      :: IW_DN(:),COEFF_DN(:)
    REAL(q)         :: EFERMI
    INTEGER         :: NOUT,NK,NB,IU0
    !local
    INTEGER                 :: NO
    REAL(q)                 :: RFREQ
    COMPLEX(q),ALLOCATABLE  :: CFREQ_UP(:)
    COMPLEX(q),ALLOCATABLE  :: SELF_W_UP(:)
    COMPLEX(q),ALLOCATABLE  :: CFREQ_DN(:)
    COMPLEX(q),ALLOCATABLE  :: SELF_W_DN(:)
    COMPLEX(q),ALLOCATABLE  :: CFREQ2(:)
    COMPLEX(q),ALLOCATABLE  :: SELF_W2(:)
    REAL(q),ALLOCATABLE     :: SIGMA_W_FIT(:,:)
    REAL(q),ALLOCATABLE     :: G_W_FIT(:,:)
    REAL(q),ALLOCATABLE     :: SIGMA_IW_FIT(:,:)
    CHARACTER(len=100)      :: STR

    WRITE(STR,"(A,I0,A,I0)") " ISP=1 NK=", NK, " NB=", NB

    ALLOCATE( CFREQ_UP(NOUT),  SELF_W_UP(NOUT) )
    ALLOCATE( CFREQ_DN(NOUT),  SELF_W_DN(NOUT) )
    ALLOCATE( CFREQ2(NOUT), SELF_W2(NOUT) )
    ALLOCATE( SIGMA_W_FIT(3,2*NOUT-1)) 
    ALLOCATE( G_W_FIT(3,2*NOUT-1)) 
    ALLOCATE(SIGMA_IW_FIT(3,NOUT)) 

    DO NO=1,NOUT
       RFREQ=EFERMI+(200.0_q-EFERMI)/(NOUT-1)*(NO-1)
       CFREQ_UP(NO)=CMPLX(RFREQ,0.0_q) 
       CFREQ2(NO)=CMPLX(0.0_q,RFREQ)
    ENDDO

    DO NO=1,NOUT
       RFREQ=-200.0_q+(EFERMI+200.0_q)/(NOUT-1)*(NO-1)
       CFREQ_DN(NO)=CMPLX(RFREQ,0.0_q) 
    ENDDO


    SIGMA_W_FIT(1,1:NOUT)=  REAL(CFREQ_DN,q)
    SIGMA_W_FIT(1,NOUT+1:2*NOUT-1)=  REAL(CFREQ_UP(2:NOUT),q)
    SIGMA_IW_FIT(1,:)=AIMAG(CFREQ2)

    CFREQ_UP(:)= CFREQ_UP(:)-CMPLX(EFERMI,0.0_q)  !w=w-EF 
    CFREQ_DN(:)= CFREQ_DN(:)-CMPLX(EFERMI,0.0_q)  !w=w-EF 

    CALL PADE_FIT_FUNC(SIZE(IW_UP),IW_UP,COEFF_UP,NOUT,CFREQ_UP,SELF_W_UP,IU0)
    CALL PADE_FIT_FUNC(SIZE(IW_DN),IW_DN,COEFF_DN,NOUT,CFREQ_DN,SELF_W_DN,IU0)

    SIGMA_W_FIT(2,1:NOUT)= REAL(SELF_W_DN,q)
    SIGMA_W_FIT(2,NOUT+1:2*NOUT-1)= REAL(SELF_W_UP(2:NOUT),q)
    SIGMA_W_FIT(3,1:NOUT)=AIMAG(SELF_W_DN)
    SIGMA_W_FIT(3,NOUT+1:2*NOUT-1)=AIMAG(SELF_W_UP(2:NOUT))

    CALL SIGMA_W_TO_G_W(SIGMA_W_FIT,G_W_FIT)

    CALL PADE_FIT_FUNC(SIZE(IW_UP),IW_UP,COEFF_UP,NOUT,CFREQ2,SELF_W2,IU0)
    SIGMA_IW_FIT(2,:)= REAL(SELF_W2,q)
    SIGMA_IW_FIT(3,:)=AIMAG(SELF_W2)

    CALL XML_VECARRAY("FIT_SIGMA_W along real axis"//TRIM(STR))
    CALL XML_ARRAY_REAL( SIGMA_W_FIT )
    CALL XML_CLOSE_TAG

    CALL XML_VECARRAY("FIT_G_W along real axis"//TRIM(STR))
    CALL XML_ARRAY_REAL( G_W_FIT )
    CALL XML_CLOSE_TAG

    CALL XML_VECARRAY("FIT_SIGMA_IW along imaginary axis"//TRIM(STR))
    CALL XML_ARRAY_REAL( SIGMA_IW_FIT )
    CALL XML_CLOSE_TAG

    DEALLOCATE( SIGMA_W_FIT, G_W_FIT, SIGMA_IW_FIT ) 
    DEALLOCATE( CFREQ_UP, SELF_W_UP ) 
    DEALLOCATE( CFREQ_DN, SELF_W_DN ) 
    DEALLOCATE( CFREQ2, SELF_W2 ) 
END SUBROUTINE WRITE_G_SIGMA_FIT


!*********************************************************************
!
!calculation of G(w) from Sigma(w)
!
!*********************************************************************
SUBROUTINE SIGMA_W_TO_G_W(SIGMA_W,GREEN_W)
  REAL(q):: SIGMA_W(:,:), GREEN_W(:,:)
  !local
  COMPLEX(q) :: OMEGA, SELF, G, ONE
  INTEGER :: I

  ONE=CMPLX(1.0_q,0.0_q)
  DO I=1,SIZE(SIGMA_W,2)
     OMEGA=CMPLX(SIGMA_W(1,I),0.0_q)
     SELF=CMPLX(SIGMA_W(2,I),SIGMA_W(3,I))
     G=ONE/(OMEGA-SELF)
     GREEN_W(1,I)=SIGMA_W(1,I)
     GREEN_W(2,I)=REAL(G)
     GREEN_W(3,I)=AIMAG(G)
  ENDDO
END SUBROUTINE


!*********************************************************************
!
!calculation of G(iw) from Sigma(iw)
!G(iw)=1/(iw+EF-Sigma(iw))
!
!*********************************************************************
SUBROUTINE SIGMA_IW_TO_G_IW(FREQ,SIGMA,G,EFERMI)
  IMPLICIT NONE
  REAL(q)     :: FREQ        !freqency
  COMPLEX(q)  :: G           !green function G
  COMPLEX(q)  :: SIGMA       !self-energy SIGMA
  REAL(q)     :: EFERMI
  G=1.0_q/(cmplx(0.0,FREQ,q)+EFERMI-SIGMA)
END SUBROUTINE SIGMA_IW_TO_G_IW


!*********************************************************************
!
!calculation of Sigma(iw) from G(iw)
!G(iw)=1/(iw+EF-Sigma(iw))
!
!*********************************************************************
SUBROUTINE G_IW_TO_SIGMA_IW(FREQ,G,SIGMA,EFERMI)
  IMPLICIT NONE
  REAL(q)     :: FREQ        !freqency
  COMPLEX(q)  :: SIGMA       !self-energy SIGMA
  COMPLEX(q)  :: G           !green function G
  REAL(q)     :: EFERMI
  SIGMA=cmplx(0.0,FREQ,q)+EFERMI-1.0_q/G
END SUBROUTINE G_IW_TO_SIGMA_IW


!*********************************************************************
!
!calculate the QP energies by linearization around DFT eigenvalues
!
!*********************************************************************
SUBROUTINE GET_QP_LINEAR(NOMEGA,EDFT,EFERMI,IW,COEFF,SIGMA,QP,IU0,NP)
    IMPLICIT NONE
    INTEGER               :: NOMEGA
    COMPLEX(q)            :: EDFT
    REAL(q)               :: EFERMI
    COMPLEX(q),INTENT(IN) :: IW(:)   
    COMPLEX(q),INTENT(IN) :: COEFF(:)  
    COMPLEX(q)            :: SIGMA        ! self-energy at DFT energy (including T+V_H)
    REAL(q)               :: QP(:)        ! approximate QP energy
    INTEGER               :: IU0
    INTEGER               :: NP           !NP-point ifferentiation stencils
    !local 
    REAL(q)               :: KK
    COMPLEX(q)            :: E_DFT(NP),H_DFT(NP)
    COMPLEX(q),PARAMETER  :: DELTA=(0.1_q,0.0_q) !0.1 seems best

    !quick return
    IF (NP/=3 .AND. NP/=5 .AND. NP/=7 ) THEN
      IF (IU0>=0) WRITE(IU0,*) "ERROR: stencils NP in GET_QP_LINEAR", NP
      STOP
    ENDIF    
    
    SELECT CASE (NP)
      CASE(3)
        E_DFT(1)=EDFT-CMPLX(EFERMI,0.0_q)
        E_DFT(2)=E_DFT(1)+DELTA  
        E_DFT(3)=E_DFT(1)-DELTA  

        CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,NP,E_DFT,H_DFT,IU0)
        SIGMA=H_DFT(1)

        KK=REAL(H_DFT(2)-H_DFT(3),q)/(2.0_q*REAL(DELTA,q))
        QP(2)=1.0_q/(1.0_q-KK)
        QP(1)=REAL(SIGMA-EDFT,q)*QP(2)+REAL(EDFT,q)

      CASE(5)
        E_DFT(1)=EDFT-CMPLX(EFERMI,0.0_q)
        E_DFT(2)=E_DFT(1)+DELTA  
        E_DFT(3)=E_DFT(1)+DELTA*2.0_q
        E_DFT(4)=E_DFT(1)-DELTA  
        E_DFT(5)=E_DFT(1)-DELTA*2.0_q  

        CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,NP,E_DFT,H_DFT,IU0)
        SIGMA=H_DFT(1)

        KK=REAL(-1.0_q*H_DFT(3)+8.0_q*H_DFT(2)-8.0_q*H_DFT(4)+1.0_q*H_DFT(5),q)/(12.0_q*REAL(DELTA,q))
        QP(2)=1.0_q/(1.0_q-KK)
        QP(1)=REAL(SIGMA-EDFT,q)*QP(2)+REAL(EDFT,q)

      CASE(7)
        E_DFT(1)=EDFT-CMPLX(EFERMI,0.0_q)
        E_DFT(2)=E_DFT(1)+DELTA  
        E_DFT(3)=E_DFT(1)+DELTA*2.0_q
        E_DFT(4)=E_DFT(1)+DELTA*3.0_q
        E_DFT(5)=E_DFT(1)-DELTA  
        E_DFT(6)=E_DFT(1)-DELTA*2.0_q  
        E_DFT(7)=E_DFT(1)-DELTA*3.0_q  

        CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,NP,E_DFT,H_DFT,IU0)
        SIGMA=H_DFT(1)

        KK=REAL(0.016666667_q*H_DFT(4)-0.15_q*H_DFT(3)+0.75_q*H_DFT(2)-0.75_q*H_DFT(5)+0.15_q*H_DFT(6)-0.016666667_q*H_DFT(7),q)/REAL(DELTA,q)
        QP(2)=1.0_q/(1.0_q-KK)
        QP(1)=REAL(SIGMA-EDFT,q)*QP(2)+REAL(EDFT,q)
     END SELECT

END SUBROUTINE GET_QP_LINEAR


!*********************************************************************
!
!*********************************************************************
SUBROUTINE GET_QP_LINEAR_2(NOMEGA,EDFT,ETMP,K_SLOP,EFERMI,IW,COEFF,SIGMA,QP,IU0)
    IMPLICIT NONE
    INTEGER               :: NOMEGA
    COMPLEX(q)            :: EDFT
    COMPLEX(q)            :: ETMP
    REAL(q)               :: K_SLOP
    REAL(q)               :: EFERMI
    COMPLEX(q),INTENT(IN) :: IW(:)
    COMPLEX(q),INTENT(IN) :: COEFF(:)
    COMPLEX(q)            :: SIGMA        ! self-energy at DFT energy (including T+V_H)
    REAL(q)               :: QP(:)        ! approximate QP energy
    INTEGER               :: IU0
    !local 
    COMPLEX(q),PARAMETER  :: DELTA=(0.1_q,0.0_q) !0.1 seems best
    COMPLEX(q)            :: E_FIT(1),H_FIT(1)

    E_FIT(1)=ETMP-CMPLX(EFERMI,0.0_q)

    CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,1,E_FIT,H_FIT,IU0)
    SIGMA=K_SLOP*(EDFT-ETMP)+H_FIT(1)

    QP(2)=1.0_q/(1.0_q-K_SLOP)
    QP(1)=REAL(SIGMA-EDFT,q)*QP(2)+REAL(EDFT,q)

END SUBROUTINE GET_QP_LINEAR_2


!*********************************************************************
!
!calculate the QP energies by searching the root of E_QP=\Sigma(E_QP)
!via damped version
!
!*********************************************************************
SUBROUTINE GET_QP_ZEROS_DAMP(ISP,NK,NB,EDFT,EFERMI,IW,COEFF,QP,IU0)
    IMPLICIT NONE
    INTEGER               :: ISP, NK, NB
    COMPLEX(q)            :: EDFT
    REAL(q)               :: EFERMI
    COMPLEX(q),INTENT(IN) :: IW(:)   
    COMPLEX(q),INTENT(IN) :: COEFF(:)  
    COMPLEX(q)            :: QP(:)
    INTEGER               :: IU0
    !local 
    COMPLEX(q),PARAMETER  :: DELTA=(0.1_q,0.0_q)
    COMPLEX(q)            :: E_DFT(1),H_DFT(1)
    COMPLEX(q)            :: E_TMP(3),H_TMP(3)
    COMPLEX(q)            :: E_QP,SELF_QP
    INTEGER               :: NOMEGA,NUM
    REAL(q), PARAMETER    :: DAMP=0.6_q
    
    NOMEGA=SIZE(IW)

    ! damped version
    SELF_QP=EDFT
    NUM=0
    
    DO
       E_QP=REAL(SELF_QP,q) ! take real part (we seek for roots along real axis)
       E_DFT(1)=REAL(SELF_QP,q)-CMPLX(EFERMI,0.0_q)
       CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,1,E_DFT,H_DFT,IU0)
       ! "mix" old and new QP energy
       SELF_QP=H_DFT(1)*DAMP+(1-DAMP)*E_QP
       NUM=NUM+1
       IF ((ABS(REAL(SELF_QP-E_QP,q))< 0.0001 .OR. NUM>=20)) EXIT
    ENDDO

    ! no mixing in last step
    SELF_QP=H_DFT(1)
    IF (NUM==20) THEN 
       IF (IU0>=0) WRITE(IU0,'(A,3I5,5F10.4)') "internal error in VASP: damped iteration in GET_QP_ZEROS failed", &
                                 & ISP, NK, NB, ABS(REAL(SELF_QP-E_QP,q)),E_QP,SELF_QP
    ENDIF
    QP(1)=SELF_QP

    !determine the Z factor
    E_TMP(1)=SELF_QP-CMPLX(EFERMI,0.0_q)
    E_TMP(2)=E_TMP(1)+DELTA
    E_TMP(3)=E_TMP(1)-DELTA

    CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,3,E_TMP,H_TMP,IU0)
    QP(2)=1.0_q/(1-REAL(H_TMP(2)-H_TMP(3),q)/(2.0_q*REAL(DELTA,q)) )

END SUBROUTINE GET_QP_ZEROS_DAMP


!*********************************************************************
!
!calculate the QP energies by searching the root of E_QP=\Sigma(E_QP)
!via Brent version
!
!*********************************************************************
SUBROUTINE GET_QP_ZEROS_BRENT(NOMEGA,ISP,NK,NB,EDFT,EFERMI,IW,COEFF,QP,IU0)
    IMPLICIT NONE
    INTEGER               :: NOMEGA
    INTEGER               :: ISP,NK,NB
    COMPLEX(q)            :: EDFT   !here is the sulution of linearized version
    REAL(q)               :: EFERMI
    COMPLEX(q),INTENT(IN) :: IW(:)   
    COMPLEX(q),INTENT(IN) :: COEFF(:)  
    COMPLEX(q)            :: QP(:)
    INTEGER               :: IU0
    !local 
    COMPLEX(q),PARAMETER  :: DELTA=(0.1_q,0.0_q)
    COMPLEX(q)            :: E_TMP(3),H_TMP(3)
    INTEGER,PARAMETER     :: NUM=100
    INTEGER               :: NTOT,I,J
    REAL(q)               :: X1,X2,DE,EPS
    REAL(q), DIMENSION(:),POINTER :: XB1,XB2
    REAL(q), ALLOCATABLE  :: ROOT(:),DIST(:)
    CHARACTER(len=100)    :: STR
   
    NULLIFY(XB1,XB2)
 
    DE=2.0_q ; NTOT=0
    X1=REAL(EDFT,q)-DE
    X2=REAL(EDFT,q)+DE

    CALL FIND_BRACKET(IW,COEFF,EFERMI,X1,X2,NUM,XB1,XB2,NTOT)
    IF (NTOT==0) THEN
       IF (IU0>=0) WRITE(IU0,'(A,3I5)') &
                    & "internal error in VASP: NTOT==0,GET_QP_ZEROS_BRENT failed", NB,NK,ISP
       QP(1)=(-1000.0_q,-1000.0_q)
       QP(2)=(-1000.0_q,-1000.0_q)
       RETURN
    ENDIF

    ALLOCATE(ROOT(NTOT),DIST(NTOT))

    DO I=1,NTOT
      ROOT(I)=ROOT_FIND_BRENT(IW,COEFF,EFERMI,XB1(I),XB2(I),0.001_q)
      IF (NTOT==1 .AND. ABS(ROOT(1)-43210.0_q)<0.0001_q) THEN
          QP(1)=(-1000.0_q,-1000.0_q)
          QP(2)=(-1000.0_q,-1000.0_q)
          RETURN
      ENDIF
      DIST(I)=ABS(ROOT(I)-REAL(EDFT,q))
    ENDDO

    J=1
    EPS=10000.0_q
    DO I=1,NTOT
       IF (DIST(I)<EPS) THEN 
          EPS=DIST(I)  
          J=I
       ENDIF
    ENDDO

#ifdef debug
444 format("(A,2X,",I0,"F9.3,4X,A,F9.3)")
555 format(3(A,I5,2X),4X,A,F9.3)
    WRITE(STR,444) NTOT
    WRITE(*,555) "NK=",NK, "NB=",NB, "NROOT=",NTOT, "E_QP(linear)=", REAL(EDFT,q)
    WRITE(*,trim(STR)) "TOTAL ROOTS=", ROOT, "FINAL_QP=", ROOT(J)
    WRITE(*,*)
#endif

    !determine the Z factor
    E_TMP(1)=ROOT(J)-CMPLX(EFERMI,0.0_q)
    E_TMP(2)=E_TMP(1)+DELTA
    E_TMP(3)=E_TMP(1)-DELTA

    CALL PADE_FIT_FUNC(NOMEGA,IW,COEFF,3,E_TMP,H_TMP,IU0)
    QP(1)=H_TMP(1)
    QP(2)=1.0_q/(1-REAL(H_TMP(2)-H_TMP(3),q)/(2.0_q*REAL(DELTA,q)) )
 
    DEALLOCATE(ROOT,DIST)
    IF (ASSOCIATED(XB1)) DEALLOCATE(XB1)
    IF (ASSOCIATED(XB2)) DEALLOCATE(XB2)
END SUBROUTINE GET_QP_ZEROS_BRENT


!*********************************************************************
!
!find the bracket to search the root
!Given a function fx defined on the interval from x1-x2 subdivide the 
!interval into n equally spaced segments, and search for zero crossings 
!of the function. nb is input as the maximum number of roots sought, and 
!is reset to the number of bracketing pairs xb1(1:nb),xb2(1:nb) that 
!are found.  
!
!*********************************************************************
SUBROUTINE FIND_BRACKET(IW,COEFF,EFERMI,X1,X2,N,XB1,XB2,NB)
    IMPLICIT NONE
    COMPLEX(q),INTENT(IN) :: IW(:)
    COMPLEX(q),INTENT(IN) :: COEFF(:)
    REAL(q), INTENT(IN)   :: EFERMI,X1,X2
    INTEGER, INTENT(IN)   :: N
    INTEGER, INTENT(OUT)  :: NB
    REAL(q), DIMENSION(:), POINTER :: XB1,XB2
    !LOCAL
    INTEGER :: I
    REAL(q) :: DX
    !!!REAL(q), EXTERNAL    :: FUNC
    REAL(q), DIMENSION(0:N) :: F,X
    LOGICAL, DIMENSION(1:N) :: MASK
    LOGICAL, SAVE :: INIT=.TRUE.

    IF (INIT) THEN
      INIT=.FALSE.
      NULLIFY(XB1,XB2)
    END IF

    IF (ASSOCIATED(XB1)) DEALLOCATE(XB1)
    IF (ASSOCIATED(XB2)) DEALLOCATE(XB2)
   
    DX=(X2-X1)/N
    X=X1+DX*ARTH_I(0,1,N+1)
    DO I=0,N
       F(I)=FUNC(SIZE(IW),IW,COEFF,EFERMI,X(I))
    END DO

    MASK=(F(1:N)*F(0:N-1) <= 0.0)
    
    NB=COUNT(MASK)
    ALLOCATE(XB1(NB),XB2(NB))
    XB1(1:NB)=PACK(X(0:N-1),MASK)
    XB2(1:NB)=PACK(X(1:N),MASK)

END SUBROUTINE FIND_BRACKET

!*********************************************************************
!
!*********************************************************************
FUNCTION ARTH_I(FIRST,INCREMENT,N)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: FIRST,INCREMENT,N
    INTEGER, DIMENSION(N) :: ARTH_I
    !LOCAL
    INTEGER, PARAMETER    :: NPAR_ARTH=16, NPAR2_ARTH=8
    INTEGER :: K,K2,TEMP
    
    IF (N > 0) ARTH_I(1)=FIRST
    IF (N <= NPAR_ARTH) THEN
       DO K=2,N
          ARTH_I(K)=ARTH_I(K-1)+INCREMENT
       END DO
    ELSE
       DO K=2,NPAR2_ARTH
          ARTH_I(K)=ARTH_I(K-1)+INCREMENT
       END DO
       TEMP=INCREMENT*NPAR2_ARTH
       K=NPAR2_ARTH
       DO
         IF (K >= N) EXIT
         K2=K+K
         ARTH_I(K+1:MIN(K2,N))=TEMP+ARTH_I(1:MIN(K,N-K))
         TEMP=TEMP+TEMP
         K=K2
       END DO
    END IF
END FUNCTION ARTH_I
!*********************************************************************


!*********************************************************************
FUNCTION ROOT_FIND_BRENT(IW,COEFF,EFERMI,X1,X2,TOL)
    IMPLICIT NONE
    COMPLEX(q),INTENT(IN) :: IW(:)
    COMPLEX(q),INTENT(IN) :: COEFF(:)
    REAL(q), INTENT(IN)   :: EFERMI,X1,X2,TOL
    REAL(q) :: ROOT_FIND_BRENT
    !local
    INTEGER, PARAMETER :: ITMAX=100
    REAL(q), PARAMETER :: EPS=EPSILON(X1)
    !!!REAL(q), EXTERNAL :: FUNC
    INTEGER(q) :: ITER
    REAL(q)    :: A,B,C,D,E,FA,FB,FC
    REAL(q)    :: P,Q1,R,S,TOL1,XM

    A=X1
    B=X2
    FA=FUNC(SIZE(IW),IW,COEFF,EFERMI,A)
    FB=FUNC(SIZE(IW),IW,COEFF,EFERMI,B)
    IF ((FA > 0.0 .AND. FB > 0.0) .OR. (FA < 0.0 .AND. FB < 0.0)) THEN
        CALL NRERROR('root must be bracketed for ROOT_FIND_BRENT')
        ROOT_FIND_BRENT=43210.0_q
        RETURN
    ENDIF    
    C=B
    FC=FB
    DO ITER=1,ITMAX
       IF ((FB > 0.0 .AND. FC > 0.0) .OR. (FB < 0.0 .AND. FC < 0.0)) THEN
                C=A
                FC=FA
                D=B-A
                E=D
       END IF
       IF (ABS(FC) < ABS(FB)) THEN
                A=B
                B=C
                C=A
                FA=FB
                FB=FC
                FC=FA
       END IF
       TOL1=2.0_q*EPS*ABS(B)+0.5_q*TOL
       XM=0.5_q*(C-B)
       IF (ABS(XM) <= TOL1 .OR. FB == 0.0) THEN
                ROOT_FIND_BRENT=B
                RETURN
       END IF
       IF (ABS(E) >= TOL1 .AND. ABS(FA) > ABS(FB)) THEN
                S=FB/FA
                IF (A == C) THEN
                        P=2.0_q*XM*S
                        Q1=1.0_q-S
                ELSE
                        Q1=FA/FC
                        R=FB/FC
                        P=S*(2.0_q*XM*Q1*(Q1-R)-(B-A)*(R-1.0_q))
                        Q1=(Q1-1.0_q)*(R-1.0_q)*(S-1.0_q)
                END IF
                IF (P > 0.0) Q1=-Q1
                P=ABS(P)
                IF (2.0_q*P  <  MIN(3.0_q*XM*Q1-ABS(TOL1*Q1),ABS(E*Q1))) THEN
                        E=D
                        D=P/Q1
                ELSE
                        D=XM
                        E=D
                END IF
       ELSE
          D=XM
          E=D
       END IF
       A=B
       FA=FB
       B=B+MERGE(D,SIGN(TOL1,XM), ABS(D) > TOL1 )
       FB=FUNC(SIZE(IW),IW,COEFF,EFERMI,B)
    END DO
    CALL NRERROR('ROOT_FIND_BRENT: exceeded maximum iterations: 100')
    ROOT_FIND_BRENT=43210.0_q
    RETURN
END FUNCTION ROOT_FIND_BRENT


!**********************************************************************
!
!*********************************************************************
SUBROUTINE NRERROR(STRING)
    IMPLICIT NONE
    CHARACTER(LEN=*), INTENT(IN) :: STRING
    WRITE (*,*) 'NRERROR: ',STRING
END SUBROUTINE NRERROR


!**********************************************************************
!
!*********************************************************************
FUNCTION FUNC(N,IZ,A,EFERMI,X)
    IMPLICIT NONE
    INTEGER,   INTENT(IN)   :: N        !number of imaginary data sets (iz(i),ih(i),i=1,...N))
    COMPLEX(q),INTENT(IN)   :: IZ(N)    !imaginary frequencies iz(i)
    COMPLEX(q),INTENT(IN)   :: A(N)     !coefficients of pade
    REAL(q),   INTENT(IN)   :: EFERMI,X
    REAL(q)                 :: FUNC
    !local
    INTEGER    :: IU0
    COMPLEX(q) :: RZ(1),RH(1)

    RZ(1)=CMPLX(X-EFERMI,0.0_q)
    CALL PADE_FIT_FUNC(N,IZ,A,1,RZ(1),RH(1),IU0)
    FUNC=REAL(RH(1),q)-X
END FUNCTION FUNC


!**********************************************************************
!
!this subroutine is implemented by following the idea from             
!G. A. Baker Jr., Essentials of Pade Approximants                      
!(Academic Press, New York, 1975),Chapter 8                            
!See also related works in                                             
!K. Lee and K. Chang, Phys. Rev. B 54, R8285 (1996)                    
!H. J. Vidberg and J. W. Serene, J, Low, Temp. Phys. 29, 179 (1977)    
!
!**********************************************************************
SUBROUTINE ACONT_PADE_FIT(N,IZ,IH, M,RZ,RH,IU0)
    IMPLICIT NONE
    INTEGER,   INTENT(IN)   :: N        !number of imaginary data sets (iz(i),ih(i),i=1,...N))
    COMPLEX(q),INTENT(IN)   :: IZ(N)    !imaginary frequencies iz(i)
    COMPLEX(q),INTENT(IN)   :: IH(N)    !function evaluated at imaginary frequencies iz(i)
    INTEGER,   INTENT(IN)   :: M        !number of real data sets (rz(i),rh(i),i=1,...M))
    COMPLEX(q),INTENT(IN)   :: RZ(M)    !real frequencies for which analytic continuation is to be performed
    COMPLEX(q),INTENT(OUT)  :: RH(M)    !function evaluated at real frequencies rz(i)
    INTEGER,   INTENT(IN)   :: IU0
    !local
    COMPLEX(q), ALLOCATABLE :: ATMP(:), BTMP(:), A(:), G(:, :)
    COMPLEX(q)              :: CTMP
    INTEGER                 :: I,J

    IF (N<2) THEN
       IF (IU0>=0) WRITE (IU0,'(A)') 'internal ERROR in VASP: N should be >=2 in ACONT_PADE_FIT(N,IZ,IH,M,RZ,RH,IU0) Stop! '
       STOP
    END IF

    ALLOCATE(ATMP(N), BTMP(0:N), A(N), G(N,N))

    A(1)=IH(1)     !A(1)=G(1,1)=F(1)  
    G(1,:)=IH(:)   !(G_1(Z(i))=F(i), i=1,N) =>  G(1,1:N)=F(1:N) 
    DO I=2,N
       DO J=2,N
          CTMP=(IZ(J)-IZ(I-1))*G(I-1,J)
          IF ( ABS(CTMP) < 1.0E-6  ) THEN
             G(I,J)=0.0_q
          ELSE
             G(I,J) = ( A(I-1)-G(I-1,J) ) / CTMP
          ENDIF 
       ENDDO
       A(I)=G(I,I)   
    ENDDO

    DO I=1,M
       CTMP=RZ(I)
       BTMP(:)=CMPLX(1.0_q, 0.0_q)
       BTMP(0)=CMPLX(0.0_q, 0.0_q)
       ATMP(1)=A(1)
       DO J=2,N
          ATMP(J)=A(J)*(CTMP-IZ(J-1)) 
       ENDDO
       CALL PADE_EVALUATE(N, ATMP, BTMP, RH(I))
    ENDDO

    DEALLOCATE(ATMP, BTMP, A, G)
END SUBROUTINE ACONT_PADE_FIT


!*********************************************************************
!
!obtain the coefficients A(:) of pade fit
!
!*********************************************************************
SUBROUTINE PADE_FIT_COEFF(N,IZ,IH,A,IU0)
    IMPLICIT NONE
    INTEGER,   INTENT(IN)   :: N        !number of imaginary data sets (iz(i),ih(i),i=1,...N))
    COMPLEX(q),INTENT(IN)   :: IZ(N)    !imaginary frequencies iz(i)
    COMPLEX(q),INTENT(IN)   :: IH(N)    !function evaluated at imaginary frequencies iz(i)
    COMPLEX(q),INTENT(OUT)  :: A(N)     !coefficients of pade fit
    INTEGER,   INTENT(IN)   :: IU0
    !local
    COMPLEX(q), ALLOCATABLE :: G(:,:)
    INTEGER                 :: I,J

    IF (N<2) THEN
       IF (IU0>=0) WRITE (IU0,'(A)') 'internal ERROR in VASP: N should be >=2 in PADE_FIT_COEFF(N,IZ,IH,A,IU0) Stop! '
       STOP
    END IF
    ALLOCATE(G(N,N))

    A(1)=IH(1)     !A(1)=G(1,1)=F(1)  
    G(1,:)=IH(:)   !(G_1(Z(i))=F(i), i=1,N) =>  G(1,1:N)=F(1:N) 
    DO I=2,N
       DO J=2,N
          G(I,J) = ( A(I-1)-G(I-1,J) ) / ( (IZ(J)-IZ(I-1))*G(I-1,J) )
       ENDDO
       A(I)=G(I,I)   
    ENDDO

    DEALLOCATE(G)
END SUBROUTINE PADE_FIT_COEFF


!*********************************************************************
!
!Evaluation of the pade_fit at specified real frequency
!PADE_FIT_COEFF should be called before you call PADE_FIT_FUNC
!
!*********************************************************************
SUBROUTINE PADE_FIT_FUNC(N,IZ,A,M,RZ,RH,IU0)
    IMPLICIT NONE
    INTEGER,   INTENT(IN)   :: N        !number of imaginary data sets (iz(i),ih(i),i=1,...N))
    COMPLEX(q),INTENT(IN)   :: IZ(N)    !imaginary frequencies iz(i)
    COMPLEX(q),INTENT(IN)   :: A(N)     !coefficients of pade
    INTEGER,   INTENT(IN)   :: M        !number of real data sets (rz(i),rh(i),i=1,...M))
    COMPLEX(q),INTENT(IN)   :: RZ(M)    !real frequencies for which analytic continuation is to be performed
    COMPLEX(q),INTENT(OUT)  :: RH(M)    !function evaluated at real frequencies rz(i)
    INTEGER,   INTENT(IN)   :: IU0
    !local
    COMPLEX(q), ALLOCATABLE :: ATMP(:), BTMP(:)
    COMPLEX(q)              :: CTMP
    INTEGER                 :: I,J

    ALLOCATE(ATMP(N), BTMP(0:N))

    DO I=1,M
       CTMP=RZ(I)
       BTMP(:)=CMPLX(1.0_q, 0.0_q)
       BTMP(0)=CMPLX(0.0_q, 0.0_q)
       ATMP(1)=A(1)
       DO J=2,N
          ATMP(J)=A(J)*(CTMP-IZ(J-1)) 
       ENDDO
       CALL PADE_EVALUATE(N, ATMP, BTMP, RH(I))
    ENDDO

    DEALLOCATE(ATMP, BTMP)
END SUBROUTINE PADE_FIT_FUNC


!*********************************************************************
!
!*********************************************************************
SUBROUTINE PADE_EVALUATE(N, A, B, FUN)
    IMPLICIT NONE
    INTEGER,   INTENT(IN)  :: N
    COMPLEX(q),INTENT(IN)  :: A(N)
    COMPLEX(q),INTENT(IN)  :: B(0:N)
    COMPLEX(q),INTENT(OUT) :: FUN
    !local
    INTEGER                :: I
    FUN=B(N)
    DO I=N,1,-1
       FUN=B(I-1)+A(I)/FUN
    ENDDO
END SUBROUTINE PADE_EVALUATE


!*********************************************************************
!
!set up the NABNDSGW modulated by NCPU
!
!*********************************************************************
  SUBROUTINE SET_NBANDSGW_pL(W,NBANDSGW)
    IMPLICIT NONE
    TYPE (wavespin)    W
    INTEGER :: NBANDSGW
    NBANDSGW=((NBANDSGW+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)*W%WDES%NB_PAR
    NBANDSGW=MIN(NBANDSGW,W%WDES%NB_TOT)
  END SUBROUTINE SET_NBANDSGW_pL


END MODULE pade_fit
