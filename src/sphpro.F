#include "symbol.inc"

#define pro_details

      MODULE msphpro
      USE prec
      CHARACTER (LEN=5), ALLOCATABLE, SAVE :: LMTABLE(:, :)

      CONTAINS
!************************ SUBROUTINE SPHPRO ****************************
! RCS:  $Id: sphpro.F,v 1.7 2003/06/27 13:22:23 kresse Exp kresse $
!
! SPHPRO calculates the projection of the wavefunctions onto spherical
! waves and from that the local charge on each ion and the 
! partial density of states
!
!***********************************************************************

      SUBROUTINE SPHPRO(GRID,LATT_CUR,P,T_INFO,W,WDES,IUP,IU6, &
          LOVERL,LMDIM,CQIJ,LPAR,LDIMP,LMDIMP,LTRUNC,LORBIT,PAR)
      USE prec
      USE main_mpi
      USE constant
      USE wave
      USE lattice
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      USE nonl
      USE relativistic

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NIONS)
      TYPE (potcar)      BET(1)
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES,WDES_1K
      TYPE (nonl_struct) NONL_S

      LOGICAL LOVERL
      INTEGER LORBIT
#ifndef realmode
! Orbital moment: start
      COMPLEX(q), PARAMETER :: ci=(0._q,1._q)
      INTEGER LMINDp,LMSp
      REAL(q), ALLOCATABLE :: PAR_lmom(:,:,:,:,:,:)
      REAL(q) PARAUG_lmom(LDIMP,T_INFO%NIONP,3,WDES%ISPIN)  
      REAL(q) ION_SUM_lmom(LDIMP,T_INFO%NIONP,3,WDES%ISPIN)
#ifdef separate_aug_lmom
      REAL(q), ALLOCATABLE :: AUG_lmom(:,:,:,:,:,:)
      REAL(q) ION_SUMAUG_lmom(LDIMP,T_INFO%NIONP,3,WDES%ISPIN) 
#endif
      REAL(q) SUMION_lmom(LDIMP),SUMTOT_lmom(LDIMP)
      COMPLEX(q) CSUM_ABS_lmom(LMDIMP,3,WDES%ISPIN)
      COMPLEX(q), ALLOCATABLE :: L_OP(:,:,:,:)
      COMPLEX(q), ALLOCATABLE :: DUMMY(:,:,:,:)
! Orbital moment: stop
#endif
      TYPE (potcar), POINTER :: PP
      OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q) PAR(WDES%NB_TOT,WDES%NKPTS,LPAR,T_INFO%NIONP,WDES%NCDIJ)
      REAL(q) WORKR(WDES%NRPLWV), WORKI(WDES%NRPLWV)
      COMPLEX(q) CSUM_ABS(LMDIMP,WDES%NCDIJ),CSUM_PHASE(LMDIMP,WDES%NCDIJ)
      COMPLEX(q) CSUM(LMDIMP*LTRUNC,WDES%NCDIJ)
      REAL(q) PARAUG(LPAR,T_INFO%NIONP,WDES%NCDIJ)
      REAL(q) SUMION(LPAR),SUMTOT(LPAR)

      REAL(q) ION_SUM(LPAR,T_INFO%NIONP,WDES%NCDIJ)
      REAL(q) ION_SUM2(LDIMP,T_INFO%NIONP,WDES%NCDIJ)

      COMPLEX(q),ALLOCATABLE :: PHAS(:,:,:,:,:)
      REAL(q) QZERO(LTRUNC)
      CHARACTER (8) :: STR
      CHARACTER(LEN=3) :: LCHAR(4)=(/'  s','  p','  d','  f'/)
      CHARACTER(LEN=5) :: LMCHAR(16)=(/'    s','   py','   pz','   px','  dxy', &
         '  dyz','  dz2','  dxz','x2-y2','fy3x2',' fxyz',' fyz2','  fz3',' fxz2',' fzx2','  fx3'/)
      CHARACTER(LEN=10) :: LMCHAR_LONG(16)=(&
        &/'     s    ','    py    ','    pz    ','    px    ',&
        & '    dxy   ','    dyz   ','    dz2   ','    dxz   ','  dx2-y2  ',&
        & 'fy(3x2-y2)','   fxyz   ','   fyz2   ','    fz3   ','   fxz2   ',' fz(x2-y2)','fx(x2-3y2)'/)

      REAL(q), ALLOCATABLE :: RHO(:)
      REAL(q) QIJ

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME= WDES%COMM%NODE_ME
      IONODE = WDES%COMM%IONODE
#endif
!=======================================================================
! some initialization
!=======================================================================
      PAR=0._q

#ifndef realmode
! Orbital moment: start
      ALLOCATE(L_OP(2*LDIMP+1,2*LDIMP+1,3,0:LDIMP),DUMMY(2*LDIMP+1,2*LDIMP+1,4,0:LDIMP))
      L_OP=(0._q,0._q)      
      DO L=1,LDIMP
         CALL SETUP_LS(L,0._q,0._q,L_OP(1:2*L+1,1:2*L+1,1:3,L),DUMMY(1:2*L+1,1:2*L+1,1:4,L))
      ENDDO

      ALLOCATE(PAR_lmom(LDIMP,T_INFO%NIONP,3,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      PAR_lmom=0._q
#ifdef separate_aug_lmom
      ALLOCATE(AUG_lmom(LDIMP,T_INFO%NIONP,3,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
      AUG_lmom=0._q
#endif
! Orbital moment: stop
#endif

      IF (LORBIT==2) THEN
         ALLOCATE(PHAS(LMDIMP,T_INFO%NIONP,WDES%NKPTS,WDES%NB_TOT,WDES%ISPIN))
         PHAS=(0._q,0._q)
      ENDIF

      IF (LORBIT==5) THEN
         ALLOCATE(PHAS(LMDIMP,T_INFO%NIONP,WDES%NKPTS,WDES%NB_TOT,WDES%NCDIJ))
         PHAS=(0._q,0._q)
      ENDIF

! compute volume per type
      io_begin
      DO NT=1,T_INFO%NTYPP
         IF ( NT<=T_INFO%NTYP ) THEN
            WRITE(IU6,2220) NT,100*T_INFO%NITYP(NT)*2*TPI*T_INFO%RWIGS(NT)**3/3/LATT_CUR%OMEGA
         ELSE
            WRITE(IU6,2221) NT,100*T_INFO%NITYP(NT)*2*TPI*T_INFO%RWIGS(NT)**3/3/LATT_CUR%OMEGA
         ENDIF

 2220  FORMAT(/'volume of typ          ',I3,':  ',F6.1,' %')
 2221  FORMAT(/'volume of empty sphere ',I3,':  ',F6.1,' %')
      ENDDO

! write header fo file PROOUT
      IF (LORBIT==5) THEN
         DO ISP=1,WDES%NCDIJ
            WRITE(STR,'(A,I1)') "PROOUT.",ISP
            OPEN(UNIT=IUP+ISP-1,FILE=DIR_APP(1:DIR_LEN)//STR,STATUS='UNKNOWN')
         ENDDO
         DO ISP=1,WDES%NCDIJ
            WRITE(IUP+ISP-1,*) 'PROOUT'
            WRITE(IUP+ISP-1,3200) WDES%NKPTS,WDES%NB_TOT,T_INFO%NIONP
            WRITE(IUP+ISP-1,'(9I4)') T_INFO%NTYPP,T_INFO%NTYP,(T_INFO%NITYP(I),I=1,T_INFO%NTYPP)
            WRITE(IUP+ISP-1,'(9F7.3)') ((W%FERTOT(NB,NK,MIN(ISP,WDES%ISPIN)),NK=1,WDES%NKPTS),NB=1,WDES%NB_TOT)
         ENDDO
      ELSE
         OPEN(UNIT=IUP,FILE=DIR_APP(1:DIR_LEN)//'PROCAR',STATUS='UNKNOWN')
         IF (LORBIT==1) THEN
            WRITE(IUP,'(A)')'PROCAR lm decomposed'
         ELSEIF (LORBIT==2) THEN
            WRITE(IUP,'(A)')'PROCAR lm decomposed + phase factor'
         ELSE
            WRITE(IUP,'(A)')'PROCAR new format'
         ENDIF
      ENDIF
      io_end


! allocate descriptor for 1 kpoints
      CALL CREATE_SINGLE_KPOINT_WDES(WDES,WDES_1K,1)
! allocate BET
      ALLOCATE(BET(1)%PSPNL(0:NPSNL,LDIMP*LTRUNC), &
               BET(1)%LPS(LDIMP*LTRUNC))
      BET(1)%LMDIM   =LMDIMP*LTRUNC
      BET(1)%LMAX    =LDIMP*LTRUNC
      BET(1)%LMMAX   =LMDIMP*LTRUNC

      CALL NONL_ALLOC_SPHPRO(NONL_S,BET(1),WDES_1K)
!=======================================================================
      NIS=1

      typ: DO NT=1,T_INFO%NTYPP
!-----------------------------------------------------------------------
!    set up table with BETAs:
!
!    BET(1)%PSPNL=  4 Pi \int_0^R_{cut}
!                j_l(qr) j_l(q_n r) r^2 dr / Sqrt(A(q,qp))
!           q_n is chosen so that j_l(q_n R_{cut}B[MaB) = 0
!     we use the relation
!     A(q,qp) = \int_0^R j_l(qr) j_l(qp r) r^2 dr
!             = R^2/(q^2-qp^2) [ j(qR) j(qpR)' - j(qp R) j(qR)')
!-----------------------------------------------------------------------
      GMAX =SQRT(WDES%ENMAX/HSQDTM)*1.2_q

      BET(1)%PSMAXN = GMAX

      PSTEP=GMAX/(NPSNL-1)

      INDEX=0
    setbet: DO LL= 0,LDIMP-1
      CALL BEZERO(QZERO,LL,LTRUNC)
      DO I=1,LTRUNC
        INDEX=INDEX+1

        Q1=QZERO(I)/T_INFO%RWIGS(NT)
        QR=QZERO(I)
        CALL SBESSE2(QR, BQ, BQP, LL)
        A= 1/Q1* T_INFO%RWIGS(NT)**2/2*(BQ*BQP+QR*BQP**2+ &
                            QR*BQ**2- (LL+1)*LL/QR*BQ**2)
        SQRTIA=1/SQRT(A)
        DO N = 0,NPSNL-1
          QQ=PSTEP*N
          IF (QQ==0) QQ=1E-5_q
          CALL SBESSE2(T_INFO%RWIGS(NT)*QQ, BQQ , BQQP, LL)
          IF (ABS(QQ-Q1)<1E-5_q) THEN
          A= 1/Q1* T_INFO%RWIGS(NT)**2/2*(BQ*BQP+QR*BQP**2+ &
                          QR*BQ**2- (LL+1)*LL/QR*BQ**2)
          ELSE
          A= T_INFO%RWIGS(NT)**2/(Q1**2-QQ**2)*(BQ*BQQP*QQ-BQQ*BQP*Q1)
          ENDIF
          BET(1)%PSPNL(N+1,INDEX)=TPI*2*A*SQRTIA
        ENDDO
      IF (MOD(LL,2)==0) THEN
        BET(1)%PSPNL(0,INDEX)=BET(1)%PSPNL(2,INDEX)
      ELSE
        BET(1)%PSPNL(0,INDEX)=BET(1)%PSPNL(2,INDEX)
      ENDIF
      BET(1)%LPS(INDEX)=LL
      ENDDO
      ENDDO setbet
!=======================================================================
      kpoint: DO NK=1,WDES%NKPTS
!=======================================================================
      CALL CREATE_SINGLE_KPOINT_WDES(WDES,WDES_1K,NK)

      IZERO =1
      CALL SPHER(GRID,NONL_S,BET,WDES_1K,LATT_CUR,  IZERO)
!=======================================================================
      ion: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
!=======================================================================
      NONL_S%POSION=>T_INFO%POSION(:,NI:NI)
      CALL PHASE(WDES_1K,NONL_S,0)  ! reset phase factor
      CALL PHASE(WDES_1K,NONL_S,1)  ! and force calculation
! phase factor e(i k R)
      GXDX=T_INFO%POSION(1,NI)
      GYDY=T_INFO%POSION(2,NI)
      GZDZ=T_INFO%POSION(3,NI)
      CGDR=EXP(CITPI*(WDES%VKPT(1,NK)*GXDX+WDES%VKPT(2,NK)*GYDY+WDES%VKPT(3,NK)*GZDZ))

      band: DO NB=1,WDES%NBANDS
!=======================================================================
! multiply with phasefactor and divide into real and imaginary part
!=======================================================================
      NPL = WDES%NGVECTOR(NK)

      CSUM      =0
      CSUM_PHASE=0
      CSUM_ABS  =0
#ifndef realmode
! Orbital moment: start
      CSUM_ABS_lmom =0
! Orbital moment: stop
#endif
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE kpoint
#endif
!-MM- spin spiral stuff
      ISPIRAL = 1
!-MM- end of addition
      spin: DO ISP=1,WDES%ISPIN
      DO ISPINOR=0,WDES%NRSPINORS-1

      DO K=1,NPL
         KK=K+ISPINOR*NPL
         CTMP=NONL_S%CREXP(K,1)*W%CPTWFP(KK,NB,NK,ISP)
         WORKR(K)=REAL(CTMP,KIND=q)
         WORKI(K)=AIMAG(CTMP)
      ENDDO
!=======================================================================
! loop over indices L,M,N and calculate 
! CSUM(lmn,alp) = < phi(alpha) | beta lmn >
!=======================================================================

      LMS=0
      DO LL=0,LDIMP-1
         DO I=1,LTRUNC
            DO M=0,2*LL
               LMS=LMS+1
               LM=LL*LL+M+1
               SUMR=0
               SUMI=0
!DIR$ IVDEP
!OCL NOVREC
              ! loop over G-vectors
               DO K=1,NPL
!-MM- changes to accomodate spin spirals
! original statements
!                 SUMR=SUMR+WORKR(K)*NONL_S%QPROJ(K,LMS,1,1)
!                 SUMI=SUMI+WORKI(K)*NONL_S%QPROJ(K,LMS,1,1)
                  SUMR=SUMR+WORKR(K)*NONL_S%QPROJ(K,LMS,1,1,ISPIRAL)
                  SUMI=SUMI+WORKI(K)*NONL_S%QPROJ(K,LMS,1,1,ISPIRAL)
!-MM- end of alterations
               ENDDO
               CTMP=GREAL(CMPLX(SUMR,SUMI,KIND=q)*NONL_S%CQFAK(LMS,1)*CGDR)
               CSUM(LMS,ISP+ISPINOR)=CTMP
            ENDDO
         ENDDO
      ENDDO
!-MM- spin spiral stuff
      IF (NONL_S%LSPIRAL) ISPIRAL=2
!-MM- end of addition
      ENDDO
      ENDDO spin

      CALLMPI( M_sum_z(WDES%COMM_INB,CSUM,LMDIMP*LTRUNC*(WDES%ISPIN+WDES%NRSPINORS-1)))

      LMS=0
      DO LL=0,LDIMP-1
         DO I=1,LTRUNC
            DO M=0,2*LL
               LMS=LMS+1
               LM=LL*LL+M+1
               CSUM_PHASE(LM,:)=CSUM_PHASE(LM,:)+CSUM(LMS,:)*ABS(CSUM(LMS,:))
            ENDDO
         ENDDO
      ENDDO

      !
      ! now calculate rho(lm,alp,bet) = 
      ! sum_n < phi(alp) | beta lmn > <beta lmn | phi(bet) >
      !
      DO ISP=1,WDES%ISPIN
      DO ISPINOR =0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1
      
      II=ISP+ISPINOR_+2*ISPINOR

      LMS=0
      DO LL=0,LDIMP-1
         DO I=1,LTRUNC
            DO M =1,2*LL+1
               LMS=LMS+1
               LM=LL*LL+M
               CSUM_ABS(LM,II)=CSUM_ABS(LM,II)+CSUM(LMS,ISP+ISPINOR)*CONJG(CSUM(LMS,ISP+ISPINOR_))               
            ENDDO
         ENDDO
      ENDDO
#ifndef realmode
! Orbital moment: start
! Calculate the orbital moment from < beta lmn | phi >, to be augmented lateron
      LMS=0
      DO LL=0,LDIMP-1
         DO I=1,LTRUNC
            LMS_BASE=LMS
            DO M =1,2*LL+1
               LMS=LMS+1
               LM=LL*LL+M
               LMS_=LMS_BASE
               DO M_=1,2*LL+1
                  LMS_=LMS_+1
                  CPRODCSUM_x=0._q;CPRODCSUM_y=0._q;CPRODCSUM_z=0._q
                  IF (ISPINOR==ISPINOR_) THEN
                     CPRODCSUM_x=CSUM(LMS,ISP+ISPINOR)*CONJG(CSUM(LMS_,ISP+ISPINOR_))*L_OP(M_,M,1,LL)
                     CPRODCSUM_y=CSUM(LMS,ISP+ISPINOR)*CONJG(CSUM(LMS_,ISP+ISPINOR_))*L_OP(M_,M,2,LL)
                     CPRODCSUM_z=CSUM(LMS,ISP+ISPINOR)*CONJG(CSUM(LMS_,ISP+ISPINOR_))*L_OP(M_,M,3,LL)
                  ENDIF
                  CSUM_ABS_lmom(LM,1,ISP)=CSUM_ABS_lmom(LM,1,ISP)+ CPRODCSUM_x
                  CSUM_ABS_lmom(LM,2,ISP)=CSUM_ABS_lmom(LM,2,ISP)+ CPRODCSUM_y
                  CSUM_ABS_lmom(LM,3,ISP)=CSUM_ABS_lmom(LM,3,ISP)+ CPRODCSUM_z
               ENDDO         
            ENDDO
         ENDDO
      ENDDO   
! Orbital moment: stop
#endif
      ENDDO
      ENDDO
      ENDDO
      
      IF (LORBIT==2) THEN
         IF (WDES%LNONCOLLINEAR) THEN
            ! the phase factor is only qualitative, we just sum over up and down
            ! for the non collinear case
            PHAS(:,NI,NK,WDES%NB_LOW+(NB-1)*WDES%NB_PAR,1)= & 
               CSUM_PHASE(:,1)*SQRT(ABS(CSUM_ABS(:,1)))/ABS(CSUM_PHASE(:,1))+ &
               CSUM_PHASE(:,2)*SQRT(ABS(CSUM_ABS(:,2)))/ABS(CSUM_PHASE(:,2))
         ELSE
            PHAS(:,NI,NK,WDES%NB_LOW+(NB-1)*WDES%NB_PAR,1:WDES%ISPIN)=CSUM_PHASE(:,1:WDES%ISPIN)* &
               SQRT(ABS(CSUM_ABS(:,1:WDES%ISPIN)))/ABS(CSUM_PHASE(:,1:WDES%ISPIN))
         ENDIF
      ENDIF

      IF (WDES%LNONCOLLINEAR) THEN
         CALL C_FLIP(CSUM_ABS,LMDIMP,LMDIMP,WDES%NCDIJ,.FALSE.)
      ENDIF

      IF (LORBIT==5) THEN
         PHAS(:,NI,NK,WDES%NB_LOW+(NB-1)*WDES%NB_PAR,1:WDES%NCDIJ)=CSUM_PHASE(:,1:WDES%NCDIJ)* &
            SQRT(ABS(CSUM_ABS(:,1:WDES%NCDIJ)))/ABS(CSUM_PHASE(:,1:WDES%NCDIJ))
      ENDIF

      DO ISP=1,WDES%NCDIJ
         DO LL=0,LDIMP-1
            IF (LORBIT==1.OR.LORBIT==2) THEN
               DO M=0,2*LL
                  LM=LL*LL+M+1
                  PAR(WDES%NB_LOW+(NB-1)*WDES%NB_PAR,NK,LM,NI,ISP)=CSUM_ABS(LM,ISP)
               ENDDO
            ELSE
               SUML=0
               DO M=0,2*LL
                  LM=LL*LL+M+1
                  SUML=SUML+CSUM_ABS(LM,ISP)
               ENDDO
               PAR(WDES%NB_LOW+(NB-1)*WDES%NB_PAR,NK,LL+1,NI,ISP)=SUML
            ENDIF
         ENDDO
      ENDDO
#ifndef realmode
! Orbital moment: start
      DO ISP=1,WDES%ISPIN
         DO IDIR=1,3
            DO LL=0,LDIMP-1
               SUML2=0
               DO M=0,2*LL
                  LM=LL*LL+M+1
                  SUML2=SUML2+CSUM_ABS_lmom(LM,IDIR,ISP)
               ENDDO
               PAR_lmom(LL+1,NI,IDIR,WDES%NB_LOW+(NB-1)*WDES%NB_PAR,NK,ISP)=SUML2
            ENDDO
         ENDDO
      ENDDO
! Orbital moment: stop
#endif
      ENDDO band
      ENDDO ion
      ENDDO kpoint

      NIS=NIS+T_INFO%NITYP(NT)
      ENDDO typ

      CALL NONL_DEALLOC_SPHPRO(NONL_S)
      DEALLOCATE(BET(1)%PSPNL,BET(1)%LPS)

      IF (ALLOCATED(PHAS)) THEN
         CALLMPI(M_sum_z(WDES%COMM_INTER, PHAS,SIZE(PHAS)))
         CALLMPI(M_sum_z(WDES%COMM_KINTER,PHAS,SIZE(PHAS)))
      ENDIF

      IF (LORBIT==5) THEN
         io_begin
         DO ISP=1,WDES%NCDIJ
            NIS=1
            DO NT=1,T_INFO%NTYP
               DO NK=1,WDES%NKPTS
               DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
               DO NB=1,WDES%NB_TOT
!                 WRITE(IUP+ISP-1,'(32F12.6)') PHAS(:,NI,NK,NB,ISP)
                  WRITE(IUP+ISP-1,'(9F12.6)') PHAS(:,NI,NK,NB,ISP)
               ENDDO
               ENDDO
               ENDDO
               NIS=NIS+T_INFO%NITYP(NT)
            ENDDO
         ENDDO
         io_end
      ENDIF

!=======================================================================
! calculate contribution from augmentation-part
!=======================================================================
      overl: IF (LOVERL) THEN

      IF (LORBIT==5) PHAS=(0._q,0._q)
#ifndef realmode
! Orbital moment: start
#ifdef separate_aug_lmom
      AUG_lmom=0
#endif
! Orbital moment: stop
#endif
    kpoint_aug: DO NK=1,WDES%NKPTS
    band_aug:   DO N=1 ,WDES%NBANDS

      NIS=1
      NPRO=0
      SUMAUG=0
      PARAUG=0
#ifndef realmode
! Orbital moment: start
      PARAUG_lmom=0
! Orbital moment: stop
#endif
    typ_aug: DO NT= 1,T_INFO%NTYP
    ion_aug: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1
      PP=>PP_POINTER(P, NI, NT)

      NIP=NI_LOCAL(NI, WDES%COMM_INB)     !  local storage index
      IF (NIP==0) CYCLE ion_aug

      IF (ALLOCATED(RHO)) DEALLOCATE(RHO)
      ALLOCATE(RHO(PP%R%NMAX))

!-----------------------------------------------------------------------
! find blocks with same quantum number L
! assuming that the block is continously arranged in the arrays
!-----------------------------------------------------------------------
      LOW=1
      LMBASE=1

      block: DO

      LL=PP%LPS(LOW)
!-----------------------------------------------------------------------
! only terms with equal L L'
!-----------------------------------------------------------------------
      DO LHI=LOW,PP%LMAX
        IF (LL/=PP%LPS(LHI)) EXIT
      ENDDO

      LHI=LHI-1
      MMAX=2*LL+1
      CSUM_ABS=0
#ifndef realmode
! Orbital moment: start
      CSUM_ABS_lmom=0
! Orbital moment: stop
#endif
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE kpoint_aug
#endif
      DO ISP=1,WDES%ISPIN
      DO ISPINOR =0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1
      II=ISP+ISPINOR_+2*ISPINOR

      DO L =LOW,LHI
      DO LP=LOW,LHI

      RHO=0._q
      DO I=1,PP%R%NMAX
         IF (PP%R%R(I)>T_INFO%RWIGS(NT)) EXIT
         RHO(I)=PP%WAE(I,L)*PP%WAE(I,LP)-PP%WPS(I,L)*PP%WPS(I,LP)
      ENDDO
      CALL SIMPI(PP%R,RHO,QIJ)
!     QIJ=PP%QION(LP,L)

      DO M =0,MMAX-1
         LMIND =NPRO+LMBASE+(L -LOW)*MMAX+M+ISPINOR *WDES%NPRO/2
         LMIND_=NPRO+LMBASE+(LP-LOW)*MMAX+M+ISPINOR_*WDES%NPRO/2

         CTMP=W%CPROJ(LMIND,N,NK,ISP)*GCONJG(W%CPROJ(LMIND_,N,NK,ISP))*QIJ

         CSUM_ABS(LL*LL+M+1,II)=CSUM_ABS(LL*LL+M+1,II)+CTMP
#ifndef realmode
! Orbital moment: start
! Add augmentation part to orbital moments
         DO M_=0,MMAX-1
            LMINDp = NPRO+LMBASE+(LP-LOW)*MMAX+M_+ISPINOR_*WDES%NPRO/2
            CTMP_x=0._q;CTMP_y=0._q;CTMP_z=0._q                    
            IF (ISPINOR==ISPINOR_) THEN            
               CTMP_x=W%CPROJ(LMIND,N,NK,ISP)*GCONJG(W%CPROJ(LMINDp,N,NK,ISP)) * &
              &                    QIJ * L_OP(M_+1,M+1,1,LL)
               CTMP_y=W%CPROJ(LMIND,N,NK,ISP)*GCONJG(W%CPROJ(LMINDp,N,NK,ISP)) * &
              &                    QIJ * L_OP(M_+1,M+1,2,LL)
               CTMP_z=W%CPROJ(LMIND,N,NK,ISP)*GCONJG(W%CPROJ(LMINDp,N,NK,ISP)) * &
              &                    QIJ * L_OP(M_+1,M+1,3,LL)
            ENDIF
            CSUM_ABS_lmom(LL*LL+M+1,1,ISP)=CSUM_ABS_lmom(LL*LL+M+1,1,ISP)+CTMP_x 
            CSUM_ABS_lmom(LL*LL+M+1,2,ISP)=CSUM_ABS_lmom(LL*LL+M+1,2,ISP)+CTMP_y
            CSUM_ABS_lmom(LL*LL+M+1,3,ISP)=CSUM_ABS_lmom(LL*LL+M+1,3,ISP)+CTMP_z
         ENDDO
! Orbital moment: stop
#endif
      ENDDO
      ENDDO
      ENDDO

      ENDDO
      ENDDO
      ENDDO

      IF (WDES%LNONCOLLINEAR) THEN
         CALL C_FLIP(CSUM_ABS,LMDIMP,LMDIMP,WDES%NCDIJ,.FALSE.)
      ENDIF

      IF (LORBIT==5) THEN
         DO M=0,MMAX-1
            PHAS(LL*LL+M+1,NI,NK,WDES%NB_LOW+(N-1)*WDES%NB_PAR,1:WDES%NCDIJ)=CSUM_ABS(LL*LL+M+1,1:WDES%NCDIJ)
         ENDDO
      ENDIF

      IF (LORBIT==1.OR.LORBIT==2) THEN
         DO ISP=1,WDES%NCDIJ
            DO M=0,MMAX-1
               PARAUG(LL*LL+M+1,NI,ISP)=PARAUG(LL*LL+M+1,NI,ISP)+REAL(CSUM_ABS(LL*LL+M+1,ISP),q)
            ENDDO
         ENDDO         
      ELSE
         DO ISP=1,WDES%NCDIJ
            PARAUG(LL+1,NI,ISP)=PARAUG(LL+1,NI,ISP)+SUM(REAL(CSUM_ABS(:,ISP),q))
         ENDDO
      ENDIF
#ifndef realmode
! Orbital moment: start
      DO ISP=1,WDES%NCDIJ
         DO IDIR=1,3
            PARAUG_lmom(LL+1,NI,IDIR,ISP)=PARAUG_lmom(LL+1,NI,IDIR,ISP)+SUM(REAL(CSUM_ABS_lmom(:,IDIR,ISP),q))
         ENDDO
      ENDDO
! Orbital moment: stop
#endif
      LMBASE=LMBASE+(LHI-LOW+1)*MMAX
      LOW=LHI+1
      IF (LOW > PP%LMAX) EXIT block
      ENDDO block

      NPRO=NPRO+LMBASE-1
      ENDDO ion_aug

      NIS=NIS+T_INFO%NITYP(NT)
      ENDDO typ_aug

      IF (ALLOCATED(RHO)) DEALLOCATE(RHO)

      CALLMPI( M_sum_d(WDES%COMM_INB,PARAUG,LPAR*T_INFO%NIONP*WDES%NCDIJ))
      PAR(WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,:,:,:)=PAR(WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,:,:,:)+PARAUG(:,:,:)
#ifndef realmode
! Orbital moment: start
      CALLMPI( M_sum_d( WDES%COMM_INB,PARAUG_lmom,WDES%ISPIN*T_INFO%NIONP*LDIMP*3))

      PAR_lmom(:,:,:,WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,1:WDES%ISPIN)=PAR_lmom(:,:,:,WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,1:WDES%ISPIN)+PARAUG_lmom(:,:,:,1:WDES%ISPIN)
#ifdef separate_aug_lmom
      AUG_lmom(:,:,:,WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,1:WDES%ISPIN)=AUG_lmom(:,:,:,WDES%NB_LOW+(N-1)*WDES%NB_PAR,NK,1:WDES%ISPIN)+PARAUG_lmom(:,:,:,1:WDES%ISPIN)
#endif
! Orbital moment: stop
#endif
      ENDDO band_aug
      ENDDO kpoint_aug

      IF (LORBIT==5) THEN
         CALLMPI(M_sum_z(WDES%COMM,PHAS,SIZE(PHAS)))
         io_begin
         DO ISP=1,WDES%NCDIJ
            WRITE(IUP+ISP-1,*) 'augmentation part'
            DO NK=1,WDES%NKPTS
            DO NB=1,WDES%NB_TOT
!              DO NI=1,T_INFO%NIONS
!                 WRITE(IUP+ISP-1,'(32F12.6)')  PHAS(:,NI,NK,NB,ISP)
!              ENDDO
               WRITE(IUP+ISP-1,'(9F12.6)') REAL(PHAS(:,:,NK,NB,ISP),q)
            ENDDO
            ENDDO
         ENDDO
         io_end
      ENDIF

      ENDIF overl


      CALLMPI( M_sum_d(WDES%COMM_INTER, PAR,SIZE(PAR)))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,PAR,SIZE(PAR)))
#ifndef realmode
! Orbital moment: start
      CALLMPI( M_sum_d(WDES%COMM_INTER, PAR_lmom,SIZE(PAR_lmom)))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,PAR_lmom,SIZE(PAR_lmom)))
#ifdef separate_aug_lmom
      CALLMPI( M_sum_d(WDES%COMM_INTER, AUG_lmom,SIZE(AUG_lmom)))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,AUG_lmom,SIZE(AUG_lmom)))
#endif
! Orbital moment: stop
#endif

! calculate the ionic occupancies
      ION_SUM=0
      DO ISP=1,WDES%NCDIJ
      ISP_=MIN(ISP,WDES%ISPIN)
      DO NK=1,WDES%NKPTS
      DO NB=1,WDES%NB_TOT
         ION_SUM(:,:,ISP)=ION_SUM(:,:,ISP)+PAR(NB,NK,:,:,ISP)*WDES%RSPIN* &
              WDES%WTKPT(NK)*W%FERTOT(NB,NK,ISP_)
      ENDDO
      ENDDO
      ENDDO
#ifndef realmode
! Orbital moment: start
      ION_SUM_lmom=0._q
#ifdef separate_aug_lmom
      ION_SUMAUG_lmom=0._q
#endif
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
      DO NB=1,WDES%NB_TOT
         ION_SUM_lmom(:,:,:,ISP)=ION_SUM_lmom(:,:,:,ISP)+PAR_lmom(:,:,:,NB,NK,ISP)* &
              WDES%RSPIN*WDES%WTKPT(NK)*W%FERTOT(NB,NK,ISP)
#ifdef separate_aug_lmom
         ION_SUMAUG_lmom(:,:,:,ISP)=ION_SUMAUG_lmom(:,:,:,ISP)+AUG_lmom(:,:,:,NB,NK,ISP)* &
              WDES%RSPIN*WDES%WTKPT(NK)*W%FERTOT(NB,NK,ISP)
#endif
      ENDDO
      ENDDO
      ENDDO
! Orbital moment: stop
#endif
      ND=LPAR*T_INFO%NIONP
      IF (.NOT.WDES%LNONCOLLINEAR) THEN
         CALL R_FLIP(ION_SUM,ND,ND,WDES%NCDIJ,.FALSE.)
      ENDIF
      io_begin

      IF (LORBIT==1.OR.LORBIT==2) THEN
         ION_SUM2=0
         DO LL=0,LDIMP-1
            DO M=1,2*LL+1
               LM=LL*LL+M
               ION_SUM2(LL+1,:,:)=ION_SUM2(LL+1,:,:)+ION_SUM(LM,:,:)
            ENDDO
         ENDDO
      ELSE
         ION_SUM2=ION_SUM
      ENDIF
!=======================================================================
!   write PAR on file PROCAR
!=======================================================================
      IF (LORBIT/=5) THEN

      DO ISP=1,WDES%ISPIN

      WRITE(IUP,3200) WDES%NKPTS,WDES%NB_TOT,T_INFO%NIONP
      DO NK=1,WDES%NKPTS
      WRITE(IUP,3201) NK,WDES%VKPT(1,NK),WDES%VKPT(2,NK),WDES%VKPT(3,NK),WDES%WTKPT(NK)
      DO NB=1,WDES%NB_TOT
      NI=1

      WRITE(IUP,3203) NB,REAL( W%CELTOT(NB,NK,ISP) ,KIND=q),WDES%RSPIN*W%FERTOT(NB,NK,ISP)
      WRITE(IUP,*)
      
      WRITE(IUP,'(A3)',ADVANCE='No') "ion"
      IF (LORBIT==1.OR.LORBIT==2) THEN
         DO NL=1,LPAR
            WRITE(IUP,'(A7)',ADVANCE='No') LMCHAR(NL)
         ENDDO
      ELSE
         DO NL=1,LPAR
            WRITE(IUP,'(A7)',ADVANCE='No') LCHAR(NL)
         ENDDO
      ENDIF
      WRITE(IUP,'(A7)',ADVANCE='yes') "tot"

      DO II=0,WDES%NRSPINORS*WDES%NRSPINORS-1
      PARSUM=0
      SUMION=0
      DO NI=1,T_INFO%NIONP
         S=0
         PARSUM=0
         DO NL=1,LPAR
            PARSUM=PARSUM+PAR(NB,NK,NL,NI,ISP+II)
            SUMION(NL)=SUMION(NL)+PAR(NB,NK,NL,NI,ISP+II)
            S=S+SUMION(NL)
         ENDDO
         WRITE(IUP,3204) NI,(PAR(NB,NK,NL,NI,ISP+II),NL=1,LPAR),PARSUM
      ENDDO
      IF (T_INFO%NIONP>1) THEN
         WRITE(IUP,3205) (SUMION(NL),NL=1,LPAR),S
      ENDIF
      ENDDO

      IF (LORBIT==2) THEN
         WRITE(IUP,'(A3)',ADVANCE='No') "ion"
         DO NL=1,LMDIMP
            WRITE(IUP,'(A15)',ADVANCE='No') LMCHAR_LONG(NL)
         ENDDO
         WRITE(IUP,*)

         DO NI=1,T_INFO%NIONP
            WRITE(IUP,3214) NI, (PHAS(M,NI,NK,NB,ISP),M=1,LMDIMP)
         ENDDO
      ENDIF
      WRITE(IUP,*)

      ENDDO  ! bands
      ENDDO  ! kpoints
      ENDDO  ! spin

      ENDIF
!=======================================================================
! set the LMTABLE
!=======================================================================
! allocate the LMTABLE
      IF (ALLOCATED(LMTABLE)) THEN
         DEALLOCATE(LMTABLE)
      ENDIF
      ALLOCATE(LMTABLE(LPAR,T_INFO%NIONP))
      LMTABLE="     "

      DO NI=1,T_INFO%NIONP
         IF (LORBIT==1.OR.LORBIT==2) THEN
            DO NL=1,LPAR
               LMTABLE(NL,NI)=LMCHAR(NL)
            ENDDO
         ELSE
            DO NL=1,LPAR
               LMTABLE(NL,NI)=LCHAR(NL)
            ENDDO
         ENDIF
      ENDDO
!=======================================================================
!   write condensed form of PAR on file OUTCAR
!=======================================================================
      DO ISP=1,WDES%NCDIJ
      IF (IU6>=0) THEN
         WRITE(IU6,*)
         SELECT CASE (ISP)
         CASE (1)
            WRITE(IU6,'(A)') 'total charge     '
         CASE (2)
            WRITE(IU6,'(A)') 'magnetization (x)'
         CASE (3)
            WRITE(IU6,'(A)') 'magnetization (y)'
         CASE (4)
            WRITE(IU6,'(A)') 'magnetization (z)'
         END SELECT
         WRITE(IU6,*)
      ENDIF

      IF (IU6>=0) THEN
         IF (LDIMP==4) THEN
            WRITE(IU6,12212)
         ELSE
            WRITE(IU6,2212)
         ENDIF
      ENDIF

      NI=1
      PARSUM=0
      DO NL=1,LDIMP
         SUMION(NL)=ION_SUM2(NL,1,ISP)
         PARSUM    =PARSUM+SUMION(NL)
         SUMTOT(NL)=SUMION(NL)
      ENDDO

      IF (IU6>0) &
         WRITE(IU6,2213) NI,(SUMION(NL),NL=1,LDIMP),PARSUM

      IF (T_INFO%NIONP>1) THEN
         DO NI=2,T_INFO%NIONP
            S=0
            PARSUM=0
            DO NL=1,LDIMP
               SUMION(NL)=ION_SUM2(NL,NI,ISP)
               PARSUM    =PARSUM+SUMION(NL)
               SUMTOT(NL)=SUMTOT(NL)+SUMION(NL)
               S         =S  +SUMTOT(NL)
            ENDDO
            IF (IU6>=0) &
               WRITE(IU6,2213) NI,(SUMION(NL),NL=1,LDIMP),PARSUM
         ENDDO
         IF (IU6>=0) &
            WRITE(IU6,2215) (SUMTOT(NL),NL=1,LDIMP),S
      ENDIF

      ENDDO
#ifndef realmode
! Orbital moment: start
!=======================================================================
! Print Orbital moments in OUTCAR
!=======================================================================      
      DO ISP=1,WDES%ISPIN
      DO IDIR=1,3
         IF (IU6>=0) THEN
            WRITE(IU6,*)
            SELECT CASE (IDIR)
            CASE (1)
                WRITE(IU6,'(A)') 'orbital moment (x)'
            CASE (2)
                WRITE(IU6,'(A)') 'orbital moment (y)'
            CASE (3)
                WRITE(IU6,'(A)') 'orbital moment (z)'
            END SELECT
            WRITE(IU6,*)
         ENDIF

         IF (IU6>=0) THEN
            IF (LDIMP==4) THEN
               WRITE(IU6,12216)
            ELSE
               WRITE(IU6,2216)
            ENDIF
         ENDIF
         NI=1 
         PARSUM2=0 
         DO NL=1,LDIMP
            SUMION_lmom(NL)=ION_SUM_lmom(NL,NI,IDIR,ISP) 
            PARSUM2=PARSUM2+SUMION_lmom(NL) 
            SUMTOT_lmom(NL)=SUMION_lmom(NL) 
         ENDDO

         IF (IU6>=0) &
            WRITE(IU6,2217) NI,(SUMION_lmom(NL),NL=1,LDIMP),PARSUM2

         IF (T_INFO%NIONP>1) THEN   
            DO NI=2,T_INFO%NIONP
               S2=0  
               PARSUM2=0
               DO NL=1,LDIMP   
                  SUMION_lmom(NL)=ION_SUM_lmom(NL,NI,IDIR,ISP)  
                  PARSUM2    =PARSUM2+SUMION_lmom(NL)  
                  SUMTOT_lmom(NL)=SUMTOT_lmom(NL)+SUMION_lmom(NL) 
                  S2         =S2  +SUMTOT_lmom(NL) 
               ENDDO   
               IF (IU6>=0) &
                  WRITE(IU6,2217) &
                     NI,(SUMION_lmom(NL),NL=1,LDIMP),PARSUM2
            ENDDO
            IF (IU6>=0) &
               WRITE(IU6,2218) (SUMTOT_lmom(NL),NL=1,LDIMP),S2
         ENDIF
      ENDDO

#ifdef separate_aug_lmom
      DO IDIR=1,3
         IF (IU6>=0) THEN
            WRITE(IU6,*)
            SELECT CASE (IDIR)
            CASE (1)
                WRITE(IU6,'(A)') 'orbital moment (x) (augmentation contribution)'
            CASE (2)
                WRITE(IU6,'(A)') 'orbital moment (y) (augmentation contribution)'
            CASE (3)
                WRITE(IU6,'(A)') 'orbital moment (z) (augmentation contribution)'
            END SELECT
            WRITE(IU6,*)
         ENDIF

         IF (IU6>=0) THEN
            IF (LDIMP==4) THEN
               WRITE(IU6,12216)
            ELSE
               WRITE(IU6,2216)
            ENDIF
         ENDIF
         NI=1 
         PARSUM2=0 
         DO NL=1,LDIMP
            SUMION_lmom(NL)=ION_SUMAUG_lmom(NL,NI,IDIR,ISP) 
            PARSUM2=PARSUM2+SUMION_lmom(NL) 
            SUMTOT_lmom(NL)=SUMION_lmom(NL) 
         ENDDO

         IF (IU6>=0) &
            WRITE(IU6,2217) NI,(SUMION_lmom(NL),NL=1,LDIMP),PARSUM2

         IF (T_INFO%NIONP>1) THEN   
            DO NI=2,T_INFO%NIONP
               S2=0  
               PARSUM2=0
               DO NL=1,LDIMP   
                  SUMION_lmom(NL)=ION_SUMAUG_lmom(NL,NI,IDIR,ISP)  
                  PARSUM2    =PARSUM2+SUMION_lmom(NL)  
                  SUMTOT_lmom(NL)=SUMTOT_lmom(NL)+SUMION_lmom(NL) 
                  S2         =S2  +SUMTOT_lmom(NL) 
               ENDDO   
               IF (IU6>=0) &
                  WRITE(IU6,2217) &
                     NI,(SUMION_lmom(NL),NL=1,LDIMP),PARSUM2
            ENDDO
            IF (IU6>=0) &
               WRITE(IU6,2218) (SUMTOT_lmom(NL),NL=1,LDIMP),S2
         ENDIF
      ENDDO
#endif
      ENDDO

 2217 FORMAT(I5,3X,5(E14.6))

 2216 FORMAT('# of ion     s             p             d             tot'/ &
     &       '-----------------------------------------------------------------')
12216 FORMAT('# of ion     s             p             d             f             tot'/ &
     &       '------------------------------------------------------------------------------')
 2218 FORMAT('------------------------------------------------------------------------------',/ &
     &          'tot  ',4X,5(E14.6)/)

      DEALLOCATE(PAR_lmom)
#ifdef separate_aug_lmom
      DEALLOCATE(AUG_lmom)
#endif
! Orbital moment: stop
#endif
      io_end

 2212 FORMAT('# of ion       s       p ', &
     &       '      d       tot'/ &
     &          '------------------------------------------')
12212 FORMAT('# of ion       s       p ', &
     &       '      d       f       tot'/ &
     &          '--------------------------------------------------')

 2213 FORMAT(I5,5X,5(F18.9))
 2215 FORMAT('--------------------------------------------------',/ &
     &          'tot  ',5X,5(F18.9)/)

 3200 FORMAT('# of k-points:',I5,9X,'# of bands:',I5,9X,'# of ions:',I5)
 3201 FORMAT(/' k-point ',I5,' :',3X,3F11.8,'     weight = ',F10.8/)

 3203 FORMAT('band ',I5,' # energy',F14.8,' # occ.',F12.8)

 3204 FORMAT(I5,17(1X,F6.3))
 3214 FORMAT(I5,17(1X,F6.3,1X,F6.3,1X))
 3205 FORMAT('tot  ',17(1X,F6.3)/)

      do_io CLOSE (IUP)
      IF (IU6>=0) WRITE(IU6,*)
      IF (LORBIT==2) DEALLOCATE(PHAS)

      RETURN
      END SUBROUTINE SPHPRO


!************************ SUBROUTINE SPHPRO_FAST************************
!
! fast partial density of states using the build in projector functions
! of the pseudopotentials
!
!***********************************************************************

      SUBROUTINE SPHPRO_FAST( &
          GRID,LATT_CUR, P,T_INFO,W,WDES, IUP,IU6, &
          LOVERL,LMDIM,CQIJ,LPAR,LDIMP,LMDIMP,LFINAL, LORBIT,PAR, &
          EFERMI, EMIN, EMAX)
      USE prec
      USE main_mpi
      USE constant
      USE wave
      USE lattice
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      USE nonl

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar), TARGET :: P(T_INFO%NTYP)
      TYPE (potcar)      BET(1)
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES_1K
      TYPE (nonl_struct) NONL_S

      LOGICAL LOVERL,LFINAL
      INTEGER LORBIT
      INTEGER :: LMDIM   ! maximum number of nlm channels
      INTEGER :: LDIMP   ! maximum L quantum number considered
      INTEGER :: LMDIMP  ! maximum number of LM quantum number considered
      INTEGER :: LPAR    ! for LORBIT<=1 equal LDIMP; for LORBIT>1  LMDIMP

      OVERLAP   CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
      REAL(q) PAR(WDES%NB_TOT,WDES%NKPTS,LPAR, T_INFO%NIONP,WDES%NCDIJ)
      REAL(q), OPTIONAL :: EFERMI, EMIN, EMAX
!  allocate dynamically
      INTEGER :: I,J 
      TYPE (potcar), POINTER :: PP
      REAL(q) SUMION(LPAR),SUMTOT(LPAR)
      REAL(q) ION_SUM(LDIMP,T_INFO%NIONS,WDES%NCDIJ)
      REAL(q) ION_SUM_DETAIL(LMDIMP,T_INFO%NIONS,WDES%NCDIJ)
      COMPLEX(q) :: CSUM(WDES%NBANDS,WDES%NKPTS,WDES%NCDIJ)
      COMPLEX(q),ALLOCATABLE :: PHAS(:,:,:,:,:)
      CHARACTER(LEN=3) :: LCHAR(4)=(/'  s','  p','  d','  f'/)
      CHARACTER(LEN=5) :: LMCHAR(16)=(/'    s','   py','   pz','   px','  dxy', &
         '  dyz','  dz2','  dxz','x2-y2','fy3x2',' fxyz',' fyz2','  fz3',' fxz2',' fzx2','  fx3'/)
      CHARACTER(LEN=10) :: LMCHAR_LONG(16)=(&
        &/'     s    ','    py    ','    pz    ','    px    ',&
        & '    dxy   ','    dyz   ','    dz2   ','    dxz   ','  dx2-y2  ',&
        & 'fy(3x2-y2)','   fxyz   ','   fyz2   ','    fz3   ','   fxz2   ',' fz(x2-y2)','fx(x2-3y2)'/)

      INTEGER, PARAMETER :: NDMAT=4        ! maximum main quantum numbers per L channel
      COMPLEX(q) :: PHAS_TMP(LMDIMP,NDMAT) 
      REAL(q) ::    VEC(NDMAT)      ! note that PHAS_TMP and VEC loops are manually unrolled
      REAL(q)    :: EV(NDMAT,NDMAT), OCC(NDMAT,NDMAT)
      INTEGER    :: NSET_OPTPROJ(T_INFO%NTYP, LDIMP)

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME= WDES%COMM%NODE_ME
      IONODE = WDES%COMM%IONODE
#endif
      RSPIN=WDES%RSPIN
!=======================================================================
! some initialization
!=======================================================================
      io_begin
!   write header fo file PROCAR and 
      IF (LFINAL) THEN
         OPEN(UNIT=IUP,FILE=DIR_APP(1:DIR_LEN)//'PROCAR',STATUS='UNKNOWN')
         IF (LORBIT==11) THEN
            WRITE(IUP,'(A)')'PROCAR lm decomposed'
         ELSE  IF (LORBIT==12) THEN
            WRITE(IUP,'(A)')'PROCAR lm decomposed + phase'
         ELSE  IF (LORBIT==13) THEN
            WRITE(IUP,'(A)')'PROCAR lm decomposed + phase; optimize projector for each state'
         ELSE  IF (LORBIT==14) THEN
            WRITE(IUP,'(A,2F14.7)')'PROCAR lm decomposed + phase; opt. projector for intervall (eV)',EMIN,EMAX
         ELSE
            WRITE(IUP,'(A)')'PROCAR new format'
         ENDIF
      ENDIF
      io_end

      IF (LORBIT==14) THEN
         IF (.NOT. PRESENT(EFERMI) .OR. .NOT. PRESENT(EMIN) .OR. .NOT. PRESENT(EMAX)) THEN
            WRITE(0,*) 'internal error in SPHPRO_FAST: optional arguments are missing'
            STOP
         ENDIF
      ENDIF

      IF (LFINAL) THEN
         PAR=0
         IF (LORBIT>=12 .AND. LORBIT<=14) THEN
            ALLOCATE(PHAS(LMDIMP,T_INFO%NIONS,WDES%NKPTS,WDES%NB_TOT,WDES%ISPIN))
            PHAS=0
         ENDIF
      ENDIF

      ! optimized projectors
      IF (LORBIT==14) THEN
         NSET_OPTPROJ=0
         DO NT=1,T_INFO%NTYP
            IF (ASSOCIATED(P(NT)%OPTPROJ)) THEN
               DEALLOCATE(P(NT)%OPTPROJ)
            ENDIF
            ALLOCATE(P(NT)%OPTPROJ(NDMAT, 1, LDIMP))
            P(NT)%OPTPROJ=0
         ENDDO
      ENDIF

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

      ! diagonalize the AE density matrix QTOT= <psi_i | psi_j>
      N_CHANNELS=LHI-LOW+1
      IF (N_CHANNELS > NDMAT) THEN
         WRITE(0,*) 'internal error in VASP: SPHPRO_FAST does not support more then 4 channels per L quantum number'
         STOP
      ENDIF
      EV(1:N_CHANNELS,1:N_CHANNELS)=PP%QTOT(LOW:LHI,LOW:LHI)
      ! determine QTOT =  O+ lambda O;  EV = lambda^(1/2) O
      ! so that the charge is simply give by projection onto 
      CALL DIAG_AND_SORT( EV, N_CHANNELS )

      DO L =LOW,LHI
      DO LP=LOW,LHI

      DO M=1,MMAX
      LM=LL*LL+M

      IF (LFINAL .AND. (LORBIT>=11 .AND. LORBIT<=14) .AND. LM > LPAR) THEN
         WRITE(0,*) 'internal ERROR: LPAR is too small in SPHPRO_FAST (LM)',LM,LPAR
         STOP
      ELSE IF (LFINAL .AND. LL+1 > LPAR) THEN
         WRITE(0,*) 'internal ERROR: LPAR is too small in SPHPRO_FAST (LL)',LL+1,LPAR
         STOP
      ENDIF

      IF (ASSOCIATED (PP%QTOT) ) THEN

      DO ISP=1,WDES%ISPIN
      DO ISPINOR=0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1

      LMIND  =LMBASE +(L -LOW) *MMAX+M + ISPINOR *WDES%NPRO/2
      LMIND_ =LMBASE +(LP-LOW) *MMAX+M + ISPINOR_*WDES%NPRO/2
      II=ISP+ISPINOR_+2*ISPINOR

         DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
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
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(WDES,WDES_1K,NK)
      DO NB=1,WDES%NB_TOT
         NB_=NB_LOCAL(NB,WDES_1K)
         IF(NB_==0) CYCLE
         ION_SUM(LL+1,NI,ISP)=ION_SUM(LL+1,NI,ISP)+CSUM(NB_,NK,ISP)*RSPIN* &
              WDES%WTKPT(NK)*W%FERWE(NB_,NK,ISP_)
         ION_SUM_DETAIL(LM,NI,ISP)=ION_SUM_DETAIL(LM,NI,ISP)+CSUM(NB_,NK,ISP)*RSPIN* &
              WDES%WTKPT(NK)*W%FERWE(NB_,NK,ISP_)

         IF (LFINAL) THEN
          IF (LORBIT>=11 .AND. LORBIT<=14) THEN 
            PAR(NB,NK,LM,NI,ISP)=PAR(NB,NK,LM,NI,ISP)+CSUM(NB_,NK,ISP)
          ELSE
            PAR(NB,NK,LL+1,NI,ISP)=PAR(NB,NK,LL+1,NI,ISP)+CSUM(NB_,NK,ISP)
          ENDIF
         ENDIF

      ENDDO
      ENDDO
      ENDDO
         
      ENDIF

      ENDDO
      ENDDO
      ENDDO
!=======================================================================
! determine optimized projectors for supplied energy window
!=======================================================================
      IF (LORBIT==14) THEN
         ! set occupancy matrix for this L quantum number to 0
         OCC=0

         DO ISP=1,WDES%ISPIN
         DO II=0,WDES%NRSPINORS-1
         DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         DO NB=1,WDES%NB_TOT
            NB_=NB_LOCAL(NB,WDES_1K)
            IF(NB_==0) CYCLE
            PHAS_TMP=0
            DO L =LOW,LHI
            DO M=1,MMAX
               LM=LL*LL+M
               ! project onto four sqrt(eigenvalue) eigenvectors of QTOT (possibly just zero)
               ! hard coded to avoid another deep do loop
               CTMP= W%CPROJ(LMBASE+(L-LOW)*MMAX+M+II*WDES%NPRO/2,NB_,NK,ISP)
               PHAS_TMP(M,1)=PHAS_TMP(M,1)+CTMP*EV(L-LOW+1,1)
               PHAS_TMP(M,2)=PHAS_TMP(M,2)+CTMP*EV(L-LOW+1,2)
               PHAS_TMP(M,3)=PHAS_TMP(M,3)+CTMP*EV(L-LOW+1,3)
               PHAS_TMP(M,4)=PHAS_TMP(M,4)+CTMP*EV(L-LOW+1,4)
            ENDDO
            ENDDO

            VEC=0
            
            ! add to occupancy matrix (see above)
            IF ( EMIN < REAL(W%CELTOT(NB,NK,ISP)) .AND. REAL(W%CELTOT(NB,NK,ISP)) < EMAX) THEN
               DO I=1,N_CHANNELS
                  DO J=1,N_CHANNELS
                     OCC(I,J)=OCC(I,J)+2*SUM(PHAS_TMP(1:MMAX,I)*CONJG(PHAS_TMP(1:MMAX,J)))*W%WDES%WTKPT(NK)
                  ENDDO
               ENDDO
            ENDIF
         ENDDO
         ENDDO
         ENDDO
         ENDDO

         ! sum OCC over bands
         CALLMPI( M_sum_d( WDES%COMM_INTER, OCC, SIZE(OCC)))
         CALLMPI( M_sum_d( WDES%COMM_KINTER, OCC, SIZE(OCC)))
         
         ! diagonalize OCC and store largest eigenvalue, specifically OCC(:,:) VEC(:)
         CALL DIAG_OCC(OCC, MMAX, N_CHANNELS, VEC)

         PP%OPTPROJ(1:N_CHANNELS,1,LL+1)=PP%OPTPROJ(1:N_CHANNELS,1,LL+1)+MATMUL(EV(1:N_CHANNELS,1:N_CHANNELS),VEC(1:N_CHANNELS))
         NSET_OPTPROJ(NT,LL+1)=NSET_OPTPROJ(NT,LL+1)+1

         ! now project onto "best" linear combination
         IF (LFINAL) THEN
            VEC(1:N_CHANNELS)=MATMUL(EV(1:N_CHANNELS,1:N_CHANNELS),VEC(1:N_CHANNELS))

            DO ISP=1,WDES%ISPIN
            DO II=0,WDES%NRSPINORS-1
            DO NK=1,WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            DO NB=1,WDES%NB_TOT
               NB_=NB_LOCAL(NB,WDES_1K)
               IF(NB_==0) CYCLE
               PHAS_TMP=0
               DO L =LOW,LHI
               DO M=1,MMAX
                  LM=LL*LL+M
                  ! project onto optimized linear combination of projectors
                  ! hard coded to avoid another deep do loop
                  CTMP= W%CPROJ(LMBASE+(L-LOW)*MMAX+M+II*WDES%NPRO/2,NB_,NK,ISP)
                  PHAS(LM,NI,NK,NB,ISP)=PHAS(LM,NI,NK,NB,ISP)+CTMP*VEC(L-LOW+1)
               ENDDO
               ENDDO            
            ENDDO
            ENDDO
            ENDDO
            ENDDO
         ENDIF
      ENDIF
!=======================================================================
! determine phase factors (LORBIT>=12 and LORBIT=13)
!=======================================================================
      IF ((LORBIT>=12 .AND. LORBIT<=13) .AND. LFINAL) THEN
         DO ISP=1,WDES%ISPIN
         DO II=0,WDES%NRSPINORS-1
         DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         DO NB=1,WDES%NB_TOT
            NB_=NB_LOCAL(NB,WDES_1K)
            IF(NB_==0) CYCLE
            PHAS_TMP=0
            DO L =LOW,LHI
            DO M=1,MMAX
               LM=LL*LL+M
               ! project onto four sqrt(eigenvalue) eigenvectors of QTOT (possibly just zero)
               ! hard coded to avoid another deep do loop
               CTMP= W%CPROJ(LMBASE+(L-LOW)*MMAX+M+II*WDES%NPRO/2,NB_,NK,ISP)
               PHAS_TMP(M,1)=PHAS_TMP(M,1)+CTMP*EV(L-LOW+1,1)
               PHAS_TMP(M,2)=PHAS_TMP(M,2)+CTMP*EV(L-LOW+1,2)
               PHAS_TMP(M,3)=PHAS_TMP(M,3)+CTMP*EV(L-LOW+1,3)
               PHAS_TMP(M,4)=PHAS_TMP(M,4)+CTMP*EV(L-LOW+1,4)
            ENDDO
            ENDDO

            VEC=0
            IF (LORBIT==12) THEN
               ! pick projector that yields the largest one-center all-electron density
               ! for this orbital
               VEC(1)=SUM(PHAS_TMP(1:MMAX,1)*CONJG(PHAS_TMP(1:MMAX,1)))
               VEC(2)=SUM(PHAS_TMP(1:MMAX,2)*CONJG(PHAS_TMP(1:MMAX,2)))
               VEC(3)=SUM(PHAS_TMP(1:MMAX,3)*CONJG(PHAS_TMP(1:MMAX,3)))
               VEC(4)=SUM(PHAS_TMP(1:MMAX,4)*CONJG(PHAS_TMP(1:MMAX,4)))
               IF (VEC(1)>VEC(2) .AND. VEC(1)>VEC(3) .AND. VEC(1)>VEC(4)) THEN
                  VEC=0; VEC(1)=1
               ELSE IF (VEC(2) > VEC(3) .AND. VEC(2) > VEC(4)) THEN
                  VEC=0; VEC(2)=1
               ELSE IF (VEC(3) > VEC(4)) THEN
                  VEC=0; VEC(3)=1
               ELSE
                  VEC=0; VEC(4)=1
               ENDIF
            ELSE IF (LORBIT==13) THEN
               ! build occupancy matrix for this L channel (summed over MMAX)
               ! for the  occupancy matrix C C+   +   C+ C = 2 Real (C C+)
               ! use eigenvector that is most strongly occupied
               DO I=1,N_CHANNELS
                  DO J=1,N_CHANNELS
                     OCC(I,J)=2*SUM(PHAS_TMP(1:MMAX,I)*CONJG(PHAS_TMP(1:MMAX,J)))
                  ENDDO
               ENDDO
               CALL DIAG_OCC(OCC, MMAX, N_CHANNELS, VEC )
            ENDIF
            DO M=1,MMAX
               LM=LL*LL+M
               PHAS(LM,NI,NK,NB,ISP)=PHAS(LM,NI,NK,NB,ISP)+ & 
                    PHAS_TMP(M,1)*VEC(1)+PHAS_TMP(M,2)*VEC(2)+PHAS_TMP(M,3)*VEC(3)+PHAS_TMP(M,4)*VEC(4)
            ENDDO
         ENDDO
         ENDDO
         ENDDO
         ENDDO
      ENDIF

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
      IF (LFINAL) THEN
        CALLMPI( M_sum_d( WDES%COMM, PAR(1,1,1,1,1), WDES%NB_TOT*WDES%NKPTS*LPAR*T_INFO%NIONP*WDES%NCDIJ))
        IF (LORBIT>=12 .AND. LORBIT<=14) THEN
           CALLMPI( M_sum_d( WDES%COMM, PHAS(1,1,1,1,1), LMDIMP*T_INFO%NIONS*WDES%NKPTS*WDES%NB_TOT*WDES%ISPIN*2))
        ENDIF
      ENDIF

      ND=LDIMP*T_INFO%NIONS
      IF (.NOT.WDES%LNONCOLLINEAR) CALL R_FLIP(ION_SUM,ND,ND,WDES%NCDIJ,.FALSE.)
!=======================================================================
!   write optimized projection operators to OUTCAR
!=======================================================================
      IF (LORBIT==14) THEN
         IF (IU6>=0) &
              WRITE(IU6,'(/" optimized coefficients for linear combined projectors, energy window ",2F10.2)') EMIN, EMAX

         CALLMPI( M_sum_i( WDES%COMM, NSET_OPTPROJ, SIZE(NSET_OPTPROJ)))
         DO NT=1,T_INFO%NTYP
            ! TODO use of PP_POINTER here
            PP=>P(NT)
            CALLMPI( M_sum_d( WDES%COMM, PP%OPTPROJ, SIZE(PP%OPTPROJ)))
            LOW=1
            DO
               LL=PP%LPS(LOW)
               DO LHI=LOW,PP%LMAX
                  IF (LL/=PP%LPS(LHI)) EXIT
               ENDDO
               LHI=LHI-1
               PP%OPTPROJ(1:LHI-LOW+1,1,LL+1)=PP%OPTPROJ(1:LHI-LOW+1,1,LL+1)/NSET_OPTPROJ(NT,LL+1)
               IF (IU6>=0) &
                    WRITE(IU6,'(" species ",A2," NT=",I4," L=",I4,10F10.3)') PP%ELEMENT,NT,LL,PP%OPTPROJ(1:LHI-LOW+1,1,LL+1)
               LOW=LHI+1
               IF (LOW > PP%LMAX) EXIT
            ENDDO
            IF (IU6>=0) WRITE(IU6,*)
         ENDDO
      ENDIF

      io_begin
!=======================================================================
!   write PAR on file PROCAR
!=======================================================================
      IF (LFINAL) THEN

      DO ISP=1,WDES%ISPIN

      WRITE(IUP,3200) WDES%NKPTS,WDES%NB_TOT,T_INFO%NIONP
      DO NK=1,WDES%NKPTS
      WRITE(IUP,3201) NK,WDES%VKPT(1,NK),WDES%VKPT(2,NK),WDES%VKPT(3,NK),WDES%WTKPT(NK)
      DO NB=1,WDES%NB_TOT
      NI=1

      WRITE(IUP,3203) NB,REAL( W%CELTOT(NB,NK,ISP) ,KIND=q),WDES%RSPIN*W%FERTOT(NB,NK,ISP)
      WRITE(IUP,*)
      
      WRITE(IUP,'(A3)',ADVANCE='No') "ion"
      IF (LORBIT>=11 .AND. LORBIT<=14) THEN
         DO NL=1,LPAR
            WRITE(IUP,'(A7)',ADVANCE='No') LMCHAR(NL)
         ENDDO
      ELSE
         DO NL=1,LPAR
            WRITE(IUP,'(A7)',ADVANCE='No') LCHAR(NL)
         ENDDO
      ENDIF
      WRITE(IUP,'(A7)',ADVANCE='yes') "tot"

      DO II=0,WDES%NRSPINORS*WDES%NRSPINORS-1
      PARSUM=0
      SUMION=0
      DO NI=1,T_INFO%NIONP
         S=0
         PARSUM=0
            DO NL=1,LPAR
               PARSUM=PARSUM+PAR(NB,NK,NL,NI,ISP+II)
               SUMION(NL)=SUMION(NL)+PAR(NB,NK,NL,NI,ISP+II)
               S=S+SUMION(NL)
            ENDDO
         WRITE(IUP,3204) NI,(PAR(NB,NK,NL,NI,ISP+II),NL=1,LPAR),PARSUM
      ENDDO
      IF (T_INFO%NIONP>1) THEN
         WRITE(IUP,3205) (SUMION(NL),NL=1,LPAR),S
      ENDIF
      ENDDO

      IF (LORBIT>=12 .AND. LORBIT<=14) THEN
         WRITE(IUP,'(A3)',ADVANCE='No') "ion"
         DO NL=1,LMDIMP
            WRITE(IUP,'(A15)',ADVANCE='No') LMCHAR_LONG(NL)
         ENDDO
         WRITE(IUP,*)
         SUMION=0
         DO NI=1,T_INFO%NIONP
!            WRITE(IUP,3204) NI, (REAL (PHAS(M,NI,NK,NB,ISP)),M=1,LMDIMP)
!            WRITE(IUP,3204) NI, (AIMAG(PHAS(M,NI,NK,NB,ISP)),M=1,LMDIMP)
!            WRITE(IUP,3204) NI, (REAL(PHAS(M,NI,NK,NB,ISP)*CONJG(PHAS(M,NI,NK,NB,ISP))),M=1,LMDIMP), &
            WRITE(IUP,3214) NI, (PHAS(M,NI,NK,NB,ISP) ,M=1,LMDIMP), &
                 REAL(SUM(PHAS(1:LMDIMP,NI,NK,NB,ISP)*CONJG(PHAS(1:LMDIMP,NI,NK,NB,ISP))))
            SUMION(1:LMDIMP)=SUMION(1:LMDIMP)+REAL(PHAS(1:LMDIMP,NI,NK,NB,ISP)*CONJG(PHAS(1:LMDIMP,NI,NK,NB,ISP)))
         ENDDO
         IF (T_INFO%NIONP>1) THEN
            WRITE(IUP,3215) SUMION(1:LMDIMP),SUM(SUMION(1:LMDIMP))
         ENDIF
      ENDIF
      WRITE(IUP,*)

      ENDDO
      ENDDO
      ENDDO

      ENDIF
!=======================================================================
!   set the LMTABLE
!=======================================================================
      IF (LFINAL) THEN
         IF (ALLOCATED(LMTABLE)) THEN
            DEALLOCATE(LMTABLE)
         ENDIF
         ALLOCATE(LMTABLE(LPAR, T_INFO%NIONP))
         LMTABLE="     "

         DO NI=1,T_INFO%NIONP
            IF (LORBIT==11.OR.LORBIT==12.OR.LORBIT==13) THEN
               DO NL=1,LPAR
                  LMTABLE(NL,NI)=LMCHAR(NL)
               ENDDO
            ELSE
               DO NL=1,LPAR
                  LMTABLE(NL,NI)=LCHAR(NL)
               ENDDO
            ENDIF
         ENDDO
      ENDIF
!=======================================================================
!   write condensed information on file OUTCAR
!=======================================================================
      DO ISP=1,WDES%NCDIJ
      IF (IU6>=0) THEN
         WRITE(IU6,*)
         SELECT CASE (ISP)
         CASE (1)
            WRITE(IU6,2011) 'total charge     '
         CASE (2)
            WRITE(IU6,2011) 'magnetization (x)'
         CASE (3)
            WRITE(IU6,2011) 'magnetization (y)'
         CASE (4)
            WRITE(IU6,2011) 'magnetization (z)'
         END SELECT
         WRITE(IU6,*)
      ENDIF
 2011 FORMAT(//A18)

      IF (IU6>=0) THEN
          IF (LDIMP==4) THEN
             WRITE(IU6,12212)
          ELSE
             WRITE(IU6,2212)
          ENDIF
      ENDIF

      NI=1
      PARSUM=0
      DO NL=1,LDIMP
        SUMION(NL)=ION_SUM(NL,1,ISP)
        PARSUM    =PARSUM+SUMION(NL)
        SUMTOT(NL)=SUMION(NL)
      ENDDO
      IF (IU6>=0) &
          WRITE(IU6,2213) NI,(SUMION(NL),NL=1,LDIMP),PARSUM
      IF (T_INFO%NIONS>1) THEN
        DO NI=2,T_INFO%NIONS
        S=0
        PARSUM=0
        DO NL=1,LDIMP
          SUMION(NL)=ION_SUM(NL,NI,ISP)
          PARSUM    =PARSUM+SUMION(NL)
          SUMTOT(NL)=SUMTOT(NL)+SUMION(NL)
          S         =S  +SUMTOT(NL)
        ENDDO
        IF (IU6>=0) &
            WRITE(IU6,2213) NI,(SUMION(NL),NL=1,LDIMP),PARSUM
        ENDDO
        IF (IU6>=0) &
            WRITE(IU6,2215) (SUMTOT(NL),NL=1,LDIMP),S
      ENDIF

      ENDDO
      io_end


 2212 FORMAT('# of ion       s       p ', &
     &       '      d       tot'/ &
     &          '------------------------------------------')
12212 FORMAT('# of ion       s       p ', &
     &       '      d       f       tot'/ &
     &          '--------------------------------------------------')

 2213 FORMAT(I5,5X,5(F18.9))
 2215 FORMAT('--------------------------------------------------',/ &
     &          'tot  ',5X,5(F18.9)/)


 3200 FORMAT('# of k-points:',I5,9X,'# of bands:',I5,9X,'# of ions:',I5)
 3201 FORMAT(/' k-point ',I5,' :',3X,3F11.8,'     weight = ',F10.8/)

 3203 FORMAT('band ',I5,' # energy',F14.8,' # occ.',F12.8)

 3204 FORMAT(I5,17(1X,F6.3))
 3214 FORMAT(I5,17(1X,F6.3,1X,F6.3,1X))
 3205 FORMAT('tot  ',17(1X,F6.3)/)
 3215 FORMAT('charge',17(F6.3,1X,'      ',2X)/)


      do_io CLOSE (IUP)
      IF (IU6>=0) WRITE(IU6,*)
      IF ((LORBIT==12.OR.LORBIT==13) .AND. LFINAL) DEALLOCATE(PHAS)

      RETURN
!*******************************************************************
!
! small helper routine to diagonalize an overlap matrix
!
! e.g. Q= <phi_i^AE | phi_j^AE >
!
! and calculate Q^(1/2)
! the sorting of the eigenvectors is such that the original order
! is largerly maintained
! EV is also zeroed if N_CHANNELS is not using the entire array
! note: identical routine in paw.F
!
!********************************************************************

      CONTAINS

      SUBROUTINE DIAG_AND_SORT(EV, N_CHANNELS )
        IMPLICIT NONE
        INTEGER N_CHANNELS
        REAL(q) :: EV(:,:)
      ! local
        REAL(q) :: R(NDMAT)
        REAL(q) :: EV_MAX, EV_WORK(NDMAT,NDMAT), TMP(NDMAT), RTMP
        INTEGER :: IFAIL, I, J, IMAX

        CALL DSYEV &
             ('V','U',N_CHANNELS,EV,SIZE(EV,1), &
             R,EV_WORK,SIZE(EV_WORK), IFAIL)

        ! eigenvalues should be positive definite, multiply in sqrt(EW)
        ! and resort that the one with largest norm comes first
        EV_WORK=0
        DO I=1,N_CHANNELS
           EV_WORK(1:N_CHANNELS,N_CHANNELS-I+1)=EV(1:N_CHANNELS,I)*SQRT(ABS(R(I)))
        ENDDO
        EV=EV_WORK

        ! now sort so that eigenvector with largest value in component=line 1 comes first
        ! largest value in line 2 comes second and so on
        DO I=1,N_CHANNELS-1
           EV_MAX=ABS(EV(I,I))
           IMAX=I
           DO J=I+1,N_CHANNELS
              ! compare with eigenvector J
              IF (ABS(EV(I,J))>EV_MAX) THEN
                 EV_MAX=ABS(EV(I,J))
                 IMAX=J
              ENDIF
           ENDDO
           ! swap vector I and IMAX
           TMP(1:N_CHANNELS)=EV(1:N_CHANNELS,I)
           EV(1:N_CHANNELS,I)=EV(1:N_CHANNELS,IMAX)
           EV(1:N_CHANNELS,IMAX)=TMP(1:N_CHANNELS)
           ! make entry EV(I,I) positive (change sign of eigenvector)
           IF (EV(I,I)<0) THEN
              EV(1:N_CHANNELS,I)=-EV(1:N_CHANNELS,I)
           ENDIF           
        ENDDO
      END SUBROUTINE DIAG_AND_SORT


!*******************************************************************
!
! small helper routine to diagonalize an one-center occupancy
! matrix and yield a new projector that yields (when
! projected onto it) a maximal occupancy
!
!********************************************************************

      SUBROUTINE DIAG_OCC( OCC, MMAX, NUMBER_OF_EV, VEC )
        IMPLICIT NONE
        REAL(q) :: OCC(NDMAT,NDMAT)
        INTEGER :: MMAX, NUMBER_OF_EV
        REAL(q) :: VEC(:)
        ! matrix
        REAL(q) :: R(NDMAT)
        REAL(q) :: OCC_WORK(NDMAT,NDMAT)
        INTEGER :: I,J, IFAIL, IMAX

        VEC=0

        CALL DSYEV &
             ('V','U',N_CHANNELS,OCC,SIZE(OCC,1), &
             R,OCC_WORK,SIZE(OCC_WORK), IFAIL)

        ! determine largest value in last eigenvector corresponding to 
        ! largest charge
        IMAX=1
        DO I=2,N_CHANNELS
           IF (ABS(OCC(I,N_CHANNELS))>ABS(OCC(IMAX,N_CHANNELS))) THEN
              IMAX=I
           ENDIF
        ENDDO
        ! and make it positive (possibly changing sign of eigenvector)
        IF (OCC(IMAX,N_CHANNELS)<0) OCC(1:N_CHANNELS,N_CHANNELS)=-OCC(1:N_CHANNELS,N_CHANNELS)
        
        ! pick larges eigenvalue and copy eigenvector to VEC
        VEC(1)=OCC(1,N_CHANNELS)
        IF (N_CHANNELS>=2) VEC(2)=OCC(2,N_CHANNELS)
        IF (N_CHANNELS>=3) VEC(3)=OCC(3,N_CHANNELS)
        IF (N_CHANNELS>=4) VEC(4)=OCC(4,N_CHANNELS)
       
      END SUBROUTINE DIAG_OCC

     END SUBROUTINE SPHPRO_FAST

!*******************************************************************
!  SUBROUTINE BEZERO
!  searches for NMAX zeros in the sperical Bessel functions
!  i/o:
!         XNULL(NMAX) result
!         L           quantum number l
!  great full spaghetti code (writen by gK)
!********************************************************************

      SUBROUTINE BEZERO(XNULL,L,NMAX)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (STEP=.1_q, BREAK= 1E-10_q )
      DIMENSION XNULL(NMAX)
! initialization
      LBES = L
      X=STEP
      N=0
! entry point for next q_n
  30  CALL SBESSEL(X, BJ1, L)
! coarse search
  10  X=X+STEP
      CALL SBESSEL(X, BJ2, L)
! found one point
      IF(BJ1*BJ2 <0) THEN
        ETA=0.0_q
! intervall bisectioning
        SSTEP=STEP
        XX   =X
  20    SSTEP=SSTEP/2
        IF (BJ1*BJ2<0) THEN
          XX=XX-SSTEP
        ELSE
          XX=XX+SSTEP
        ENDIF
        CALL SBESSEL(XX, BJ2, L)
        IF (SSTEP>BREAK) GOTO 20

        N=N+1
        XNULL(N)=XX
        IF (N==NMAX) RETURN
        GOTO 30
      ENDIF
      GOTO 10

      END SUBROUTINE
      END MODULE msphpro


!************************ SUBROUTINE C_FLIP ***************************
!
! rearranges the storage mode of an array from 
! (up, down) (i.e. spinor representation) to (total,magnetization)
! also the reverse operation is possible if setting LBACK=.TRUE.
!
!***********************************************************************

      SUBROUTINE C_FLIP(C,NDIM,NELM,NCDIJ,LBACK)
      USE prec
      IMPLICIT NONE

      LOGICAL LBACK
      INTEGER NCDIJ,NDIM,NELM,N
      COMPLEX(q) :: C(NDIM,NCDIJ)
      REAL(q) FAC
      COMPLEX(q) :: CQU,CQD,C01,C10
      COMPLEX(q) :: C11,C00,CX,CY,CZ

      IF (NCDIJ==2 ) THEN
!=======================================================================
         FAC=1._q
         IF (LBACK) FAC=0.5_q
      
         DO N=1,NELM
            CQU=C(N,1)
            CQD=C(N,2)
            C(N,1)=FAC*(CQU+CQD)
            C(N,2)=FAC*(CQU-CQD)
         ENDDO

      ELSE IF ( NCDIJ==4 .AND. .NOT. LBACK) THEN
!=======================================================================
         DO N=1,NELM
            C00=C(N,1)
            C01=C(N,2)
            C10=C(N,3)
            C11=C(N,4)

            C(N,1)= C00+C11
            C(N,2)= C01+C10
            C(N,3)=(C01-C10)*(0._q,1._q)
            C(N,4)= C00-C11             
         ENDDO
      ELSE IF ( NCDIJ==4 .AND. LBACK) THEN
!=======================================================================
         FAC=0.5_q
         DO N=1,NELM
            C00=C(N,1)
            CX =C(N,2)
            CY =C(N,3)
            CZ =C(N,4)
            
            C(N,1)= (C00+CZ)*FAC
            C(N,2)= (CX-CY*(0._q,1._q))*FAC
            C(N,3)= (CX+CY*(0._q,1._q))*FAC
            C(N,4)= (C00-CZ)*FAC
         ENDDO
      ENDIF

      END SUBROUTINE
 

!************************ SUBROUTINE R_FLIP ***************************
!
! rearranges the storage mode of an array from 
! (up, down) (i.e. spinor representation) to (total,magnetization)
! also the reverse operation is possible if setting LBACK=.TRUE.
! (collinear version only)
!***********************************************************************

      SUBROUTINE R_FLIP(C,NDIM,NELM,NCDIJ,LBACK)
      USE prec
      IMPLICIT NONE

      LOGICAL LBACK
      INTEGER NCDIJ,NDIM,NELM,N
      REAL(q) :: C(NDIM,NCDIJ)
      REAL(q) FAC
      REAL(q) :: CQU,CQD

      IF (NCDIJ==2 ) THEN
         FAC=1._q
         IF (LBACK) FAC=0.5_q
      
         DO N=1,NELM
            CQU=C(N,1)
            CQD=C(N,2)
            C(N,1)=FAC*(CQU+CQD)
            C(N,2)=FAC*(CQU-CQD)
         ENDDO
      ENDIF

      END SUBROUTINE
      


!************************ SUBROUTINE SPHPRO_DESCRIPTION ****************
!
! this subroutine can be used to get a content description
! of the partial DOSCAR array
! it returns an empty string if the table is not yet set up
!
!***********************************************************************
    
      SUBROUTINE SPHPRO_DESCRIPTION(ni, l, lmtype)
      USE prec
      USE msphpro
      INTEGER ni, nis
      INTEGER l
      CHARACTER (LEN=5) :: lmtype

      IF (.NOT.ALLOCATED(LMTABLE)) THEN
        lmtype=" err "
        RETURN
      ENDIF

      IF (ni > SIZE( LMTABLE, 2) .OR.  l > SIZE( LMTABLE, 1)) THEN
         lmtype=" err "
         RETURN
      ENDIF


      IF (ni==0) THEN
         lmtype="     "
         DO nis=1,SIZE( LMTABLE, 2)
            IF (LMTABLE(l, nis) /= "     ") THEN
               lmtype=LMTABLE(l, nis)
               EXIT
            ENDIF
         ENDDO
      ELSE
         lmtype=LMTABLE(l, ni)
      ENDIF

      END SUBROUTINE
