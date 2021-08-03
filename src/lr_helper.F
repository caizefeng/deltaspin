#include "symbol.inc"

MODULE lr_helper
  USE prec
  USE vaspxml
  IMPLICIT NONE
CONTAINS

!*********************************************************************
!
! displace an atom in the cartesian direction IDIR by a step STEP
!
!*********************************************************************

  SUBROUTINE DISPLACE_ATOM(INITIAL_POSITIONS, POS, ION, IDIR, A, B, STEP)
    USE lattice
    REAL(q) :: INITIAL_POSITIONS(:,:)
    REAL(q) :: POS(:,:)
    REAL(q) :: A(3,3), B(3,3)
    INTEGER ION, IDIR
    REAL(q) :: STEP
  ! local 
    REAL(q) :: TMP(3)


    POS=INITIAL_POSITIONS
    TMP=0
    TMP(IDIR)=STEP
    CALL KARDIR(1,TMP,B)
    POS(:,ION) = POS(:,ION)+TMP
  END SUBROUTINE DISPLACE_ATOM

!************************ ADD_NABLA_ONE_CENTRE   ***********************
!
! add the term  
! - hbar^2/ m_e 
!  < psi_i | nabla | psi_j > - < tilde psi_i | nabla | tilde psi_j >
! to CDIJ
!
!***********************************************************************

  SUBROUTINE ADD_NABLA_ONE_CENTRE(WDES, T_INFO, P, LMDIM, IDIR, CDIJ)
    USE poscar
    USE pseudo
    USE wave
    USE constant
    IMPLICIT NONE

    INTEGER LMDIM
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    INTEGER IDIR
  ! local
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

    INTEGER NT, NI, NIP, NIS, LM, LMP, ISP, ISP_INC


    IF (WDES%LNONCOLLINEAR) THEN 
       ISP_INC=3
    ELSE
       ISP_INC=1
    ENDIF

    NIS=1
    typ: DO NT=1,T_INFO%NTYP
      ion: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      NIP=NI_LOCAL(NI, WDES%COMM_INB)  ! local storage index
      IF (NIP==0) CYCLE ion            ! projected wavefunction not on local node
      IF (.NOT. ASSOCIATED(P(NT)%NABLA)) CYCLE

      DO ISP=1,ISP_INC
      DO LM =1,P(NT)%LMMAX
      DO LMP=1,P(NT)%LMMAX
         CDIJ(LM,LMP,NIP,ISP)=CDIJ(LM,LMP,NIP,ISP)+P(NT)%NABLA(IDIR,LM,LMP)*(2*HSQDTM)
      ENDDO
      ENDDO
      ENDDO

      ENDDO ion
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO typ
    END SUBROUTINE ADD_NABLA_ONE_CENTRE


!************************ SUBROUTINE OVERL_ALL **************************
!
! calculate the result of the overlap-operator acting onto the
! wave function character 
!
!  W2%CPROJ_N,nlm += sum_n'l'm' (Q_N,n'l'm',nlm) W1%CPROJ_N,n'l'm'  
! 
! W2 and W1 may be identical
! if LZERO is set W2%CPROJ is set to zero prior to adding the term
!
!***********************************************************************

    SUBROUTINE OVERL_ALL(WDES, W1, W2, LZERO, LMDIM, CQIJ)
      USE prec

      USE wave
      USE mpimy
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES1
      TYPE (wavespin)    W1, W2
      LOGICAL LZERO
      INTEGER LMDIM
      OVERLAP    CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
! local variables
      GDEF CRESUL(WDES%NPRO)
      INTEGER :: ISP, NK, N, M

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CALL SETWDES(WDES,WDES1,NK)

         DO N=1,WDES%NBANDS
            CALL OVERL1(WDES1, LMDIM,CQIJ,CQIJ, 0.0_q, W1%CPROJ(1,N,NK,ISP),CRESUL)
            
            IF (LZERO) THEN
               DO M=1,WDES%NPRO
                  W2%CPROJ(M,N,NK,ISP)=CRESUL(M)
               ENDDO
            ELSE
               DO M=1,WDES%NPRO
                  W2%CPROJ(M,N,NK,ISP)=W2%CPROJ(M,N,NK,ISP)+CRESUL(M)
               ENDDO
            ENDIF
         ENDDO
      ENDDO
      ENDDO

      RETURN
    END SUBROUTINE OVERL_ALL



!************************ SUBROUTINE SUM_INPROD *************************
!
! this subroutine evaluates calculates the inproduct between two sets
! of wavefunctions 
! ) LOVERL = .FALSE.
!  sum_nk (<phi(1)_nk| phi(2)_nk >+c.c) * f_nk * w_k (plane wave part only)
! ) LOVERL = .TRUE., CQIJ associated
!  sum_nk (<phi(1)_nk| (1+ sum_ij | p_i> Q_ij <p_j |)  phi(2)_nk >+c.c)
!              * f_nk * w_k
! ) LOVERL = .TRUE., CQIJ not associated
!  sum_nk (<phi(1)_nk| phi(2)_nk > + \sum_i phi(1)_nk^i * phi(2)_nk^i + c.c)
!              * f_nk * w_k
!  where phi(1)_nk^i is the wavefunction character stored in CPROJ
!
!***********************************************************************

    SUBROUTINE SUM_INPROD( WDES, W1, W2, SUM, LOVERL, LMDIM, CQIJ)
      USE wave_high
      USE mpimy
      USE hamil
      USE hamil_lr
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES1
      TYPE (wavespin)    W1, W2
      LOGICAL LOVERL
      REAL(q)    SUM
      INTEGER    LMDIM
      OVERLAP, OPTIONAL :: CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
! local variables
      TYPE (wavefun1)  W1_,W2_
      COMPLEX(q) CE
      INTEGER :: NK, NGVECTOR, ISP, N, M, MM, ISPINOR

      SUM=0
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         NGVECTOR=WDES%NGVECTOR(NK)
         CALL SETWDES(WDES,WDES1,NK)


         DO N=1,WDES%NBANDS
            CALL SETWAV(W1,W1_,WDES1,N,ISP)
            CALL SETWAV(W2,W2_,WDES1,N,ISP)
            CE=0
            IF (LOVERL) THEN
               IF (PRESENT(CQIJ)) THEN
                  CALL ECCP_NL_ALL(WDES1,W1_,W2_,CQIJ,CQIJ,0.0_q,CE)
               ELSE
                  DO M=1,WDES%NPRO
                     CE=CE+W1_%CPROJ(M)*GCONJG(W2_%CPROJ(M))
                  ENDDO
               ENDIF
            ENDIF
            DO ISPINOR=0,WDES%NRSPINORS-1
               DO M=1,NGVECTOR
                  MM=M+ISPINOR*NGVECTOR
                  CE=CE+W1_%CW(MM)*CONJG(W2_%CW(MM))
               ENDDO
            ENDDO
!            write(*,'(3I4,2e14.7)') isp,nk,n,ce
            SUM=SUM+(CE+CONJG(CE))*WDES%RSPIN*WDES%WTKPT(NK)*W1%FERWE(N,NK,ISP)
         ENDDO
      ENDDO
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM, SUM, 1))

      RETURN
    END SUBROUTINE SUM_INPROD


    SUBROUTINE SUM_INPROD_C( WDES, W1, W2, SUM, LOVERL, LMDIM, CQIJ)
      USE wave_high
      USE mpimy
      USE hamil
      USE hamil_lr
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (wavedes1)    WDES1
      TYPE (wavespin)    W1, W2
      LOGICAL LOVERL
      COMPLEX(q) SUM
      INTEGER    LMDIM
      OVERLAP, OPTIONAL :: CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
! local variables
      TYPE (wavefun1)  W1_,W2_
      COMPLEX(q) CE
      INTEGER :: NK, NGVECTOR, ISP, N, M, MM, ISPINOR

#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('SUM_INPROD_C: KPAR>1 not tested (but implemented), sorry.') )
         STOP
      END IF
#endif

      SUM=0
      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         NGVECTOR=WDES%NGVECTOR(NK)
         CALL SETWDES(WDES,WDES1,NK)


         DO N=1,WDES%NBANDS
            CALL SETWAV(W1,W1_,WDES1,N,ISP)
            CALL SETWAV(W2,W2_,WDES1,N,ISP)
            CE=0
            IF (LOVERL) THEN
               IF (PRESENT(CQIJ)) THEN
                  CALL ECCP_NL_ALL(WDES1,W1_,W2_,CQIJ,CQIJ,0.0_q,CE)
               ELSE
                  DO M=1,WDES%NPRO
                     CE=CE+W1_%CPROJ(M)*GCONJG(W2_%CPROJ(M))
                  ENDDO
               ENDIF
            ENDIF
            DO ISPINOR=0,WDES%NRSPINORS-1
               DO M=1,NGVECTOR
                  MM=M+ISPINOR*NGVECTOR
                  CE=CE+W1_%CW(MM)*CONJG(W2_%CW(MM))
               ENDDO
            ENDDO
!            write(*,'(3I4,2e14.7)') isp,nk,n,ce
            SUM=SUM+CE*WDES%RSPIN*WDES%WTKPT(NK)*W1%FERWE(N,NK,ISP)
         ENDDO
      ENDDO
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM, SUM, 2))

      RETURN
    END SUBROUTINE SUM_INPROD_C


!************************ FUNCTION ONE_CENTRE_CHARGE *******************
!
! calculate the total one centre charge density
!
!***********************************************************************

    FUNCTION ONE_CENTRE_CHARGE(WDES, T_INFO, P, ION, LMDIM, CQIJ, CRHODE)
      USE prec
      USE wave
      USE poscar
      USE mpimy
      USE pseudo
      IMPLICIT NONE

      TYPE (wavedes)     WDES
      TYPE (type_info)   T_INFO
      INTEGER ION
      INTEGER LMDIM
      OVERLAP    CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
      OVERLAP    CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NRSPINORS*WDES%NRSPINORS)
      TYPE (potcar) :: P(T_INFO%NTYP)
      REAL(q) ONE_CENTRE_CHARGE
! local
      INTEGER NIP, NT
      INTEGER LM, LL, LHI, LOW, MMAX, L, LP, M
      REAL(q) SUM

      NIP=NI_LOCAL(ION, WDES%COMM_INB)  ! local storage index in CRHODE1
      NT=T_INFO%ITYP(ION)

      SUM=0
 
      IF (NIP/=0) THEN

      LOW=1
      LM =1
      block: DO
         LL=P(NT)%LPS(LOW)
         ! search block with same L
         DO LHI=LOW,P(NT)%LMAX
            IF (LL/=P(NT)%LPS(LHI)) EXIT
         ENDDO
         LHI=LHI-1
         MMAX=2*LL+1

         DO L =LOW,LHI
         DO LP=LOW,LHI
            DO M =0,MMAX-1
               SUM=SUM+CQIJ(LM+(L-LOW)*MMAX+M,LM+(LP-LOW)*MMAX+M,NIP,1) &
                    *CRHODE(LM+(L-LOW)*MMAX+M,LM+(LP-LOW)*MMAX+M,NIP,1)
            ENDDO
         ENDDO
         ENDDO
      
         ! set new LOW value and LM value and go on
         LM=LM+(LHI-LOW+1)*MMAX
         LOW=LHI+1
         IF (LOW > P(NT)%LMAX) EXIT block
      ENDDO block

      ENDIF
      
      CALLMPI( M_sum_s(WDES%COMM_INB, 1, SUM, 0._q, 0._q, 0._q))
      ONE_CENTRE_CHARGE=-SUM
      RETURN
    END FUNCTION  ONE_CENTRE_CHARGE



!************************ SUBROUTINE SETDIJ_R **************************
!
! This subroutine calculates minus the dipole moment of the augmentation
! charges
!  tau_idir,i,j = - sum_r (r-R)_idir Q_i,j(r-R)
! 
!***********************************************************************

    SUBROUTINE SETDIJ_R(WDES,GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,d_IJ,IRDMAX,IDIR)

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

      IMPLICIT NONE

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER ::  GRIDC
      TYPE (transit)             C_TO_US  ! index table between GRIDC and GRIDUS
      TYPE (latt)                LATT_CUR
      TYPE (wavedes)             WDES

      REAL(q)                    SUM(3),d_LM(256)

      INTEGER                    IRDMAX   ! allocation required for augmentation
      LOGICAL                    LOVERL,LADDITIONAL
      INTEGER                    LMDIM 
      OVERLAP                    d_IJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q), ALLOCATABLE ::    XS(:),YS(:),ZS(:)
      REAL(q), ALLOCATABLE ::    DIST(:),DEP(:),YLM(:,:)
! local
      TYPE (potcar), POINTER :: PP

      INTEGER :: IDIR, LYDIM, LMYDIM, NI, NT, LYMAX, INDMAX, ISP_INC, ISP, LDEP_INDEX
      INTEGER :: LM, L, LMP, LP, LL, LLP, M, MPLOW, MP, INDYLM, INDPYL, IND
      REAL(q) :: RINPL, TSUM
      INTEGER, EXTERNAL :: MAXL_AUG, MAXL1
#ifdef MPI
      OVERLAP, ALLOCATABLE ::    RTMP(:,:,:,:)
      INTEGER NIP
      ALLOCATE(RTMP(LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
#else
#define RTMP d_IJ
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
      RTMP  =0
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
         LYMAX=LYMAX*2
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

         SUM=0
         TSUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC
         DO IND=1,INDMAX
            SUM(1)=SUM(1)-XS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(2)=SUM(2)-YS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(3)=SUM(3)-ZS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
         ENDDO

         !   add to array d_IJ and make symmetric
         RTMP(LM+M-1,LMP+MP-1,NI,ISP)=SUM(IDIR)*RINPL
         RTMP(LMP+MP-1,LM+M-1,NI,ISP)=SUM(IDIR)*RINPL
      ENDDO mp_loop
      ENDDO m_loop

 510  LMP=LMP+2*LLP+1
      ENDDO lp_loop
      LM =LM +2*LL +1
      ENDDO l_loop
   ELSE lpaw
!=======================================================================
!
! PAW approach
! 
!=======================================================================
      IF (LYMAX==0) CYCLE ! Is L=1 present?

      CALL SETDEP(PP%QDEP(1,1,1),PP%PSDMAX,NPSRNL, &
     &        LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))

      d_LM=0
      ! Only L=1 matters
      DO M=1,3 
         INDYLM=M+1
         SUM=0
         TSUM=0
      !   sum over all r
!DIR$ IVDEP
!OCL NOVREC
         DO IND=1,INDMAX
            SUM(1)=SUM(1)-XS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(2)=SUM(2)-YS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(3)=SUM(3)-ZS(IND)*DEP(IND)*YLM(IND,INDYLM)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)
         ENDDO
         d_LM(INDYLM)=SUM(IDIR)*RINPL
      ENDDO

      !   and transform LM -> ll'mm' i.e. d_LM -> d_IJ
      CALL CALC_DLLMM( RTMP(:,:,NI,ISP),d_LM,PP)

      ENDIF lpaw

      RTMP(:,:,NI,ISP)=RTMP(:,:,NI,ISP)*T_INFO%VCA(NT)
      
      ENDDO spin

!=======================================================================
      ENDDO ion
!=======================================================================
      DEALLOCATE(XS,YS,ZS,DIST,DEP,YLM)
#ifdef realmode
      CALLMPI( M_sum_d(GRIDC%COMM, RTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_z(GRIDC%COMM, RTMP, LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))
#endif

 2000 CONTINUE      

#ifdef debug
      d_IJ_=RTMP
#endif

#ifdef MPI
      ion2: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion2
         DO ISP=1,WDES%NCDIJ
            d_IJ(:,:,NIP,ISP)=RTMP(:,:,NI,ISP)
         ENDDO
      ENDDO ion2
      DEALLOCATE(RTMP)
#else
#undef RTMP
#endif
      RETURN
    END SUBROUTINE SETDIJ_R

!=======================================================================
!  calculate first derivative of potential 
!  and derivatives of non local strenght parameters
!  this presently done using finite differences
!  the results are stored in 
!   CVTOT1
!   SV1
!   CDIJ1
!  the subroutine requires that the arrays
!   CSTRF1 and CSTRF2
!  are correctly set up (corresponding to the phase factors for
!  displacement +-POTIM/2.
!
!  second derivative
!  grid point
!            -2    -1      1      2
!                  -1/2    1/2
!            1/12 -8/12    8/12  -1/12
!
!=======================================================================

    SUBROUTINE POT_DER( WDES, GRID, GRIDC, GRIDUS, GRID_SOFT, C_TO_US, SOFT_TO_C, &
         INFO, T_INFO, P,  LATT_CUR, E1, E2, CSTRF, CSTRF1, CSTRF2, &
         CHTOT, CHTOT1, CHTOT_, DENCOR, CVTOT1, CVTOT2, SV1, SV2, & 
         LMDIM, CDIJ1, CRHODE, CRHODE1, CQIJ, CRHODE_, &
         N_MIX_PAW, RHOLM, RHOLM1, RHOLM_, &
         IRDMAX, DISPL, POTIM)

    USE base
    USE wave
    USE mgrid
    USE poscar
    USE pseudo
    USE lattice
    USE pawm
    USE us
    USE pot

    TYPE (wavedes)     WDES
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (info_struct) INFO
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (latt)        LATT_CUR
    TYPE (energy)      E1,E2      ! energy arrays
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! original structure factor
    COMPLEX(q)  CSTRF1(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at positive displ
    COMPLEX(q)  CSTRF2(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at negative displ
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density
    COMPLEX(q)  CHTOT1(GRIDC%MPLWV,WDES%NCDIJ)! derivative of charge density
    COMPLEX(q)  CHTOT_(GRIDC%MPLWV,WDES%NCDIJ)! work array charge density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT1(GRIDC%MPLWV,WDES%NCDIJ)! derivative of potential on GRIDC (real space)
    COMPLEX(q)  CVTOT2(GRIDC%MPLWV,WDES%NCDIJ)! work array
    RGRID       SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! derivative of potential on GRID
    RGRID       SV2(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! work array
    INTEGER LMDIM
    OVERLAP  CDIJ1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! derivative of strength parameters
    OVERLAP  CDIJ2(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)   ! work array
    OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)  ! one center occupancies
    OVERLAP  CRHODE1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! derivative of one center occupancies
    OVERLAP  CRHODE_(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! work array
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)      ! one centre occupancies passed through mixer
    REAL(q)  RHOLM1(N_MIX_PAW,WDES%NCDIJ)     ! derivative of one centre occupancies
    REAL(q)  RHOLM_(N_MIX_PAW,WDES%NCDIJ)     ! work array
    INTEGER IRDMAX
    REAL(q) DISPL(3,T_INFO%NIONS)             ! displacement vector
    REAL(q) POTIM                             ! step size used for finite differences
! local
    INTEGER :: IRDMAA
    REAL(q)   XCSIF(3,3)
    INTEGER :: ISP

! positive displacement
    CHTOT_=CHTOT  +CHTOT1*(POTIM/2)

    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF1,DENCOR,-1)
    CVTOT1=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E1,LATT_CUR, &
         CHTOT_,CSTRF1,CVTOT1,DENCOR,SV1, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=DISPL*(POTIM/2)
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,.TRUE.,IRDMAA,IRDMAX, DISPL)

    CRHODE_=CRHODE+CRHODE1*(POTIM/2)
    RHOLM_=RHOLM+RHOLM1*(POTIM/2)
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E1,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    
! negative displacement
    CHTOT_=CHTOT  -CHTOT1*(POTIM/2)
    
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF2,DENCOR,-1)
    CVTOT2=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E2,LATT_CUR, &
         CHTOT_,CSTRF2,CVTOT2,DENCOR,SV2, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=-DISPL
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ2,CQIJ,CVTOT2,.TRUE.,IRDMAA,IRDMAX,DISPL)
    
    CRHODE_=CRHODE-CRHODE1*(POTIM/2)
    RHOLM_=RHOLM-RHOLM1*(POTIM/2)
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ2(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E2,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    
! restore  displacement
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF,DENCOR,-1)

!
! derivative of potential CVTOT1=(CVTOT1-CVTOT2)*(1/POTIM)
!                         SV1   =(SV1   -SV2  )*(1/POTIM)
       
    DO ISP=1,WDES%NCDIJ
       CALL RL_ADD(CVTOT1(1,ISP),1/POTIM,CVTOT2(1,ISP),-1/POTIM,CVTOT1(1,ISP),GRIDC)
       CALL RL_ADD(SV1(1,ISP)   ,1/POTIM,SV2(1,ISP)  ,-1/POTIM,SV1(1,ISP)   ,GRID)
    ENDDO

! derivative of CDIJ
! this contains two contributions
! one from the displaced ion, and one from the change of the potential
! one call to SETDIJ could be saved by calculating the first term
! once and forever, and evaluating CDIJ1 from the derivative of the potential
    CDIJ1 =(CDIJ1 -CDIJ2)*(1/POTIM)

  END SUBROUTINE POT_DER

    SUBROUTINE POT_DER4( WDES, GRID, GRIDC, GRIDUS, GRID_SOFT, C_TO_US, SOFT_TO_C, &
         INFO, T_INFO, P,  LATT_CUR, E1, E2, CSTRF, CSTRF1, CSTRF2, CSTRF3, CSTRF4, &
         CHTOT, CHTOT1, CHTOT_, DENCOR, CVTOT1, CVTOT2, SV1, SV2, & 
         LMDIM, CDIJ1, CRHODE, CRHODE1, CQIJ, CRHODE_, &
         N_MIX_PAW, RHOLM, RHOLM1, RHOLM_, &
         IRDMAX, DISPL, POTIM)

    USE base
    USE wave
    USE mgrid
    USE poscar
    USE pseudo
    USE lattice
    USE pawm
    USE us
    USE pot

    TYPE (wavedes)     WDES
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (info_struct) INFO
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (latt)        LATT_CUR
    TYPE (energy)      E1,E2      ! energy arrays
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! original structure factor
    COMPLEX(q)  CSTRF1(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at positive displ
    COMPLEX(q)  CSTRF2(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at negative displ
    COMPLEX(q)  CSTRF3(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at positive displ*2
    COMPLEX(q)  CSTRF4(GRIDC%MPLWV,T_INFO%NTYP) ! structure factor at negative displ*2
    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density
    COMPLEX(q)  CHTOT1(GRIDC%MPLWV,WDES%NCDIJ)! derivative of charge density
    COMPLEX(q)  CHTOT_(GRIDC%MPLWV,WDES%NCDIJ)! work array charge density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT1(GRIDC%MPLWV,WDES%NCDIJ)! derivative of potential on GRIDC (real space)
    COMPLEX(q)  CVTOT2(GRIDC%MPLWV,WDES%NCDIJ)! work array
    RGRID       SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! derivative of potential on GRID
    RGRID       SV2(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! work array
    INTEGER LMDIM
    OVERLAP  CDIJ1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! derivative of strength parameters
    OVERLAP  CDIJ2(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)   ! work array
    OVERLAP  CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)  ! one center occupancies
    OVERLAP  CRHODE1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! derivative of one center occupancies
    OVERLAP  CRHODE_(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! work array
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)      ! one centre occupancies passed through mixer
    REAL(q)  RHOLM1(N_MIX_PAW,WDES%NCDIJ)     ! derivative of one centre occupancies
    REAL(q)  RHOLM_(N_MIX_PAW,WDES%NCDIJ)     ! work array
    INTEGER IRDMAX
    REAL(q) DISPL(3,T_INFO%NIONS)             ! displacement vector
    REAL(q) POTIM                             ! step size used for finite differences
! local
    INTEGER :: IRDMAA
    REAL(q)   XCSIF(3,3)
    INTEGER :: ISP
    TYPE (energy)      E3,E4      ! energy arrays
    REAL(q) :: WEIGHT

! positive displacement
    CHTOT_=CHTOT  +CHTOT1*(POTIM/2)

    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF1,DENCOR,-1)
    CVTOT1=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E1,LATT_CUR, &
         CHTOT_,CSTRF1,CVTOT1,DENCOR,SV1, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=DISPL*(POTIM/2)
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,.TRUE.,IRDMAA,IRDMAX, DISPL)

    CRHODE_=CRHODE+CRHODE1*(POTIM/2)
    RHOLM_=RHOLM+RHOLM1*(POTIM/2)
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E1,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    
! negative displacement
    CHTOT_=CHTOT  -CHTOT1*(POTIM/2)
    
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF2,DENCOR,-1)
    CVTOT2=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E2,LATT_CUR, &
         CHTOT_,CSTRF2,CVTOT2,DENCOR,SV2, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=-DISPL
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ2,CQIJ,CVTOT2,.TRUE.,IRDMAA,IRDMAX,DISPL)
    
    CRHODE_=CRHODE-CRHODE1*(POTIM/2)
    RHOLM_=RHOLM-RHOLM1*(POTIM/2)
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ2(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E2,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

! restore  displacement
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF,DENCOR,-1)

!
! derivative of potential CVTOT1=(CVTOT1-CVTOT2)*(1/POTIM)
!                         SV1   =(SV1   -SV2  )*(1/POTIM)
    WEIGHT=16._q/12._q/POTIM
      
    DO ISP=1,WDES%NCDIJ
       CALL RL_ADD(CVTOT1(1,ISP),WEIGHT,CVTOT2(1,ISP),-WEIGHT,CVTOT1(1,ISP),GRIDC)
       CALL RL_ADD(SV1(1,ISP)   ,WEIGHT,SV2(1,ISP)   ,-WEIGHT,SV1(1,ISP)   ,GRID)
    ENDDO
    CDIJ1 =(CDIJ1 -CDIJ2)*WEIGHT

! positive displacement (doubled)
    CHTOT_=CHTOT  +CHTOT1*POTIM

    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF3,DENCOR,-1)
    CVTOT2=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E3,LATT_CUR, &
         CHTOT_,CSTRF3,CVTOT2,DENCOR,SV2, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=-DISPL*2
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ2,CQIJ,CVTOT2,.TRUE.,IRDMAA,IRDMAX, DISPL)

    CRHODE_=CRHODE+CRHODE1*POTIM
    RHOLM_=RHOLM+RHOLM1*POTIM
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ2(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E3,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

    WEIGHT=-2._q/12._q/POTIM

    DO ISP=1,WDES%NCDIJ
       CALL RL_ADD(CVTOT1(1,ISP),1.0_q,CVTOT2(1,ISP),WEIGHT,CVTOT1(1,ISP),GRIDC)
       CALL RL_ADD(SV1(1,ISP)   ,1.0_q,SV2(1,ISP)   ,WEIGHT,SV1(1,ISP)   ,GRID)
    ENDDO
    CDIJ1 =CDIJ1+WEIGHT*CDIJ2

! negative displacement (doubled)
    CHTOT_=CHTOT  -CHTOT1*POTIM
    
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF4,DENCOR,-1)
    CVTOT2=0
    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E4,LATT_CUR, &
         CHTOT_,CSTRF4,CVTOT2,DENCOR,SV2, SOFT_TO_C,XCSIF)
    
    ! SETDIJ routine can handle finite displacements directly
    DISPL=-DISPL
    CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ2,CQIJ,CVTOT2,.TRUE.,IRDMAA,IRDMAX,DISPL)
    
    CRHODE_=CRHODE-CRHODE1*POTIM
    RHOLM_=RHOLM-RHOLM1*POTIM
    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ2(1,1,1,1), RHOLM_ , CRHODE_(1,1,1,1), &
         E4,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

    WEIGHT=2._q/12._q/POTIM

    DO ISP=1,WDES%NCDIJ
       CALL RL_ADD(CVTOT1(1,ISP),1.0_q,CVTOT2(1,ISP),WEIGHT,CVTOT1(1,ISP),GRIDC)
       CALL RL_ADD(SV1(1,ISP)   ,1.0_q,SV2(1,ISP)   ,WEIGHT,SV1(1,ISP)   ,GRID)
    ENDDO
    CDIJ1 =CDIJ1+WEIGHT*CDIJ2
    
! restore  displacement
    IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF,DENCOR,-1)

  END SUBROUTINE POT_DER4

!*********************************************************************
!
! determine the change of a wavefunction with respect to k_i
! as arguments the change of the wavefunction in the IRZ with respect
! to external fields is supplied
! the cartesian index i is supplied in IDIR (IDIR=1,...,3)  
!
!*********************************************************************

  SUBROUTINE KDER_WAVE(W, GRID, WDES, IDIR, RPHI, RPHI_CPROJ, &
       T_INFO, P, LATT_CUR, KPOINTS_TRANS )
    USE prec
    USE wave
    USE mgrid
    USE full_kpoints
    USE kpoints_change
    USE fock
    USE poscar
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE pawsym
    IMPLICIT NONE

    TYPE (wavespin)    W
    TYPE (grid_3d)     GRID
    TYPE (wavedes)     WDES
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (latt)        LATT_CUR
    TYPE (skpoints_trans) :: KPOINTS_TRANS
    INTEGER IDIR
! variables to store G [H,r] phi = d phi/ d k_i
    COMPLEX(qs) :: RPHI(:,:,:,:,:)
    GDEFS       :: RPHI_CPROJ(:,:,:,:,:)
! local
    INTEGER ISP, NK, ISPINOR, N, NK_OLD, ISP_OLD, NPL
! 
    INTEGER LDIM, MMAX, NI, NT, NPRO, NIP, NPROP, NIS, IROT, L, K, I, J, IDIR_OLD
    INTEGER, EXTERNAL :: MAXL
    INTEGER :: NPRO_NI(WDES%NIONS)
    REAL(q),ALLOCATABLE :: SL(:,:,:)
    REAL(q) :: S(3,3), FAKT
    REAL(q) :: CPROJ(1:100)

    PROFILING_START('kder_wave')

! orbitals at other k-points are required
    CALL KPAR_SYNC_ALL(WDES,W)
!
! k-points transformation matrix is not properly set up
! simple copy fall back
!
    IF (.NOT. ASSOCIATED(KPOINTS_TRANS%CPHASE)) THEN
       W%CW   =RPHI(:,:,:,:,IDIR)
       W%CPROJ=RPHI_CPROJ(:,:,:,:,IDIR)
       PROFILING_STOP('kder_wave')
       RETURN
    ENDIF
!
! arrays required to transform wave function character
!
    LDIM=MAXL(T_INFO%NTYP,P)    ! maximum l quantum number
    MMAX=(2*LDIM+1)             ! maximum m quantum number
    ALLOCATE (SL(MMAX,MMAX,0:LDIM))
!
! setup index array NPRO_NI into the CPROJ array
! (required to transform wave function characters)
!
    NPRO=1
    NIS=1
    DO NT=1, WDES%NTYP
       DO NI=NIS, WDES%NITYP(NT)+NIS-1
          NPRO_NI(NI)=NPRO
          NPRO= WDES%LMMAX(NT)+NPRO
       ENDDO
       NIS=NIS+WDES%NITYP(NT)
    ENDDO
    W%CW=0
    W%CPROJ=0

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          NK_OLD=KPOINTS_TRANS%NK_OLD(NK)
!new
          ISP_OLD=ISP
          IF (KPOINTS_TRANS%SPINFLIP(NK)==1) THEN
             ISP_OLD=3-ISP
          ENDIF
!new
    ! determine the unitary transformation matrix in real space when going from
    ! old to new k-point
          S=0
          DO L=1,3
          DO K=1,3
             DO J=1,3
             DO I=1,3
                S(L,I)=S(L,I)+LATT_CUR%A(L,K)*KPOINTS_TRANS%ISYMOP(J,K,NK)*LATT_CUR%B(I,J)
             ENDDO
             ENDDO
          ENDDO
          ENDDO
direction:DO IDIR_OLD=1,3
    ! a field in direction IDIR_OLD will rotate into a field 
    !  S(:,IDIR_OLD) we seek the response for a field in direction IDIR
          FAKT=S(IDIR,IDIR_OLD)
          IF (ABS(FAKT) < 1E-6) CYCLE
    !
    ! and perform the transformation of the wavefunction coefficients
    !
          DO N=1,WDES%NBANDS
             DO ISPINOR=0,WDES%NRSPINORS-1
                CALL ROTATE_WAVE_ADD(WDES%NGVECTOR(NK), &
                     W%CW(1+ISPINOR*WDES%NGVECTOR(NK),N,NK,ISP), & 
                     RPHI(1+ISPINOR*WDES%NGVECTOR(NK_OLD),N,NK_OLD,ISP_OLD,IDIR_OLD), &
                     KPOINTS_TRANS%CPHASE(1,NK), KPOINTS_TRANS%NINDPW(1,NK),  &
                     KPOINTS_TRANS%LINV(NK), KPOINTS_TRANS%LSHIFT(NK),FAKT)
             ENDDO
          ENDDO

    !
    ! and perform the transformation of the wavefunction caracter
    !
          CALL SETUP_SYM_LL(MMAX,LDIM,KPOINTS_TRANS%ISYMOP(:,:,NK),SL,LATT_CUR%A,LATT_CUR%B)

          DO N=1,WDES%NBANDS
             DO ISPINOR=0,WDES%NRSPINORS-1
                NIS=1
                NPRO =ISPINOR *WDES%NPRO/2+1
                DO NT=1, WDES%NTYP
                   DO NI=NIS, WDES%NITYP(NT)+NIS-1
                      NIP=KPOINTS_TRANS%ROTMAP(NI,NK)
                      NPROP=NPRO_NI(NIP)
                      CALL ROTATE_VECTOR_ADD( KPOINTS_TRANS%LINV(NK),RPHI_CPROJ(NPROP,N,NK_OLD,ISP_OLD,IDIR_OLD), &
                           W%CPROJ(NPRO,N,NK,ISP),MMAX,LDIM,SL(1,1,0),P(NT),FAKT)
                      NPRO= WDES%LMMAX(NT)+NPRO
                   ENDDO
                   NIS=NIS+WDES%NITYP(NT)
                ENDDO
             ENDDO
          ENDDO
          ENDDO direction
       ENDDO
      ENDDO
    DEALLOCATE(SL)

    PROFILING_STOP('kder_wave')

  END SUBROUTINE KDER_WAVE

END MODULE lr_helper

!************************ SUBROUTINE AUG_DER ***************************
!
! this subroutine calculates  the derivated of the 
! augmentation charge density  in reciprocal space with respect to
! the ion ION and the direction IDIR (cartesian index)
! finite differences are used as documented in us.F
!
!***********************************************************************



  SUBROUTINE AUG_DER( &
           WDES, GRIDC, GRIDUS, C_TO_US, &
           LATT_CUR, P, T_INFO, SYMM, LOVERL, &
           LMDIM, CRHODE, CHTOT, IRDMAX, ION, IDIR) 
    USE base
    USE pseudo
    USE poscar
    USE mgrid
    USE lattice
    USE wave
    USE constant
    USE us

    IMPLICIT NONE

    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (grid_3d),TARGET ::  GRIDC,GRIDUS
    TYPE (transit)     C_TO_US
    TYPE (latt)        LATT_CUR
    TYPE (wavedes)     WDES
    TYPE (symmetry)    SYMM

    INTEGER   IRDMAX         ! allocation required for augmentation
    INTEGER   LMDIM
    OVERLAP   CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    RGRID     CHTOT(DIMREAL(GRIDC%MPLWV),WDES%NCDIJ)
    LOGICAL   LOVERL,LADDITIONAL
    INTEGER   ION, IDIR

  ! internally finite differences are used
    REAL(q),PARAMETER   :: DIS=1E-4
    REAL(q)   DISPL(3,T_INFO%NIONS)
    OVERLAP,ALLOCATABLE :: CTMP(:,:,:,:)
    INTEGER   NIP, ISP

    ALLOCATE(CTMP(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ))

  ! only one ion is moved, set occupancies for other ions to 0
    NIP=NI_LOCAL(ION, WDES%COMM_INB)  ! local storage index in CRHODE

    CTMP=0
    !  copy the one center occupancy matrix to CTMP
    DO ISP=1,WDES%NCDIJ
       IF (NIP/=0) THEN
          CTMP(:,:,NIP,ISP)=CRHODE(:,:,NIP,ISP)
       ENDIF
    ENDDO

    CHTOT=0

    DISPL=0
    DISPL(IDIR,:)=-DIS
    CALL AUGMENTATION_CHARGE( &
           WDES, GRIDC, GRIDUS, C_TO_US, &
           LATT_CUR, P, T_INFO, SYMM, LOVERL, &
           LMDIM, CTMP, CHTOT, IRDMAX, DISPL) 

    CHTOT=-CHTOT

    DISPL(IDIR,:)=+DIS
    CALL AUGMENTATION_CHARGE( &
           WDES, GRIDC, GRIDUS, C_TO_US, &
           LATT_CUR, P, T_INFO, SYMM, LOVERL, &
           LMDIM, CTMP, CHTOT, IRDMAX, DISPL) 

    CHTOT=CHTOT*(1._q/2/DIS)
    
    DEALLOCATE (CTMP)

  END SUBROUTINE AUG_DER


!************************ SUBROUTINE POT_CHARGE ***********************
!
! this subroutine calculates Tr[ rho V] where the potential
! is supplied in real space (CVTOT) and the charge in rec. space
!
!***********************************************************************

   FUNCTION TR_POT_CHARGE( GRID, NCDIJ, CVTOT, CHTOT )
      USE prec
      USE mgrid

      IMPLICIT NONE
      TYPE (grid_3d)     GRID
      INTEGER NCDIJ
      RGRID CHTOT(DIMREAL(GRID%MPLWV), NCDIJ), &
            CVTOT(DIMREAL(GRID%MPLWV), NCDIJ)
      REAL(q) TR_POT_CHARGE
! LOCAL
      RGRID, ALLOCATABLE::  CWORK(:)
      INTEGER ISP, I
      REAL(q) SUM

      ALLOCATE(CWORK(DIMREAL(GRID%MPLWV)))

      SUM=0
      DO ISP=1,NCDIJ
         CALL RC_ADD(CHTOT(1,ISP),1.0_q,CHTOT(1,ISP),0.0_q,CWORK,GRID)
         CALL FFT3D(CWORK,GRID,1)
         DO I=1,GRID%RL%NP
            SUM=SUM+CVTOT(I,ISP)*CWORK(I)
         ENDDO
      ENDDO
      SUM=SUM/GRID%NPLWV

      CALLMPI( M_sum_d(GRID%COMM,SUM,1))
      TR_POT_CHARGE=sum

      DEALLOCATE(CWORK)

    END FUNCTION TR_POT_CHARGE


!************************ SUBROUTINE POTXC2     ***********************
!
! this subroutine calculates the second derivative of the xc
! correlation potential and evaluates the double counting corrections
!
!***********************************************************************

    SUBROUTINE POTXC2(GRID,LATT_CUR, CHTOT0, DENCOR, CHTOT1, CWORK, E)

      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar

      IMPLICIT NONE
      TYPE (latt)        LATT_CUR
      TYPE (grid_3d)     GRID
      INTEGER NCDIJ

      RGRID CHTOT1(DIMREAL(GRID%MPLWV))
      RGRID CHTOT0(DIMREAL(GRID%MPLWV))
      RGRID CVTOT(DIMREAL(GRID%MPLWV))
      RGRID CWORK(DIMREAL(GRID%MPLWV)),CHTOT0_(DIMREAL(GRID%MPLWV))
      RGRID DENCOR(GRID%RL%NP)
      REAL(q) SUM, E, XCENC, EXC, XCSIF(3,3)
      COMPLEX(q) EZERO

      INTEGER I

      CALL RC_ADD(CHTOT0,1.0_q,CHTOT0,0.0_q,CHTOT0_,GRID)
      CALL FFT3D(CHTOT0_,GRID,1)

      CALL FEXCP(GRID,LATT_CUR%OMEGA, &
               CHTOT0_,DENCOR,CWORK,CVTOT,EZERO,EXC,XCENC,XCSIF,.FALSE.)

      SUM=0
      CALL RC_ADD(CHTOT1,1.0_q,CHTOT1,0.0_q,CWORK,GRID)
      CALL FFT3D(CWORK,GRID,1)
      DO I=1,GRID%RL%NP
         SUM=SUM+CVTOT(I)*CWORK(I)*CWORK(I)
      ENDDO
      SUM=-SUM/GRID%NPLWV/GRID%NPLWV*LATT_CUR%OMEGA

      CALLMPI( M_sum_d(GRID%COMM,SUM,1))

      E=SUM
    END SUBROUTINE POTXC2


