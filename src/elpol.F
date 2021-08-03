#include "symbol.inc"
      MODULE ELPOL
      USE prec
!***********************************************************************
!
! This module contains subroutines to evaluate the Berry-phase 
! expression, given by D.Vanderbilt and R.D. King-Smith, to calculate
! the electronic polarization of an insulating crystal within the
! Vanderbilt US-PP scheme.
! The subroutines have been written and tested by Martijn Marsman
!
! References: 
! 1)  David Vanderbilt, R.D. King-Smith
!     "Electronic polarization in the ultrasoft pseudopotential formalism"
!     Unpublished report 17-1-1998
!     http://xxx.lanl.gov/ps/cond-mat/9801177
! 2)  R.D. King-Smith, David Vanderbilt
!     "Theory of polarization of crystalline solids"
!     Phys.Rev.B., v.47, n.3, p.1651, (1993)
!
!***********************************************************************

      CONTAINS

!************************ SUBROUTINE SETYLM_AUG2 ***********************
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

      SUBROUTINE SETYLM_AUG2(GRID,LATT_CUR,POSION,PSDMAX,NPSRNL, &
     &  LMYDIM,LYDIM,YLM,IRMAX,INDMAX,DISX,DISY,DISZ,XS,YS,ZS,DIST)
       
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
!-----------------------------------------------------------------------
! MPI version z ist the fast index
!-----------------------------------------------------------------------
#ifdef MPI
      IND=1

      DO N2=N2LOW,N2HI
      X2=(N2*F2-POSION(2))
      N2P=MOD(N2+10*GRID%NGY,GRID%NGY)

      DO N1=N1LOW,N1HI
      X1=(N1*F1-POSION(1))
      N1P=MOD(N1+10*GRID%NGX,GRID%NGX)

      NCOL=GRID%RL%INDEX(N1P,N2P)
      IF (NCOL==0) CYCLE
      IF (GRID%RL%I2(NCOL) /= N1P+1 .OR. GRID%RL%I3(NCOL) /= N2P+1) THEN
        WRITE(*,*)'internal ERROR SETYLM:',N1P+1,N2P+1,NCOL, &
           GRID%RL%I2(NCOL),N1P , GRID%RL%I3(NCOL),N2P
        STOP
      ENDIF

!OCL SCALAR
      DO N3=N3LOW,N3HI
        X3=(N3*F3-POSION(3))

        X= X1*LATT_CUR%A(1,1)+X2*LATT_CUR%A(1,2)+X3*LATT_CUR%A(1,3)
        Y= X1*LATT_CUR%A(2,1)+X2*LATT_CUR%A(2,2)+X3*LATT_CUR%A(2,3)
        Z= X1*LATT_CUR%A(3,1)+X2*LATT_CUR%A(3,2)+X3*LATT_CUR%A(3,3)

        D=SQRT(X*X+Y*Y+Z*Z)
        ARG=(D*ARGSC)+1
        NADDR=INT(ARG)

        IF (NADDR<NPSRNL) THEN
          ZZ=Z-DISZ
          YY=Y-DISY
          XX=X-DISX
        ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
        ! is done using the well known formula  | R+d | = | R | + d . R/|R|
        ! this improves the stability of finite differences considerable
          IF (D<1E-4_q) THEN
            DIST(IND)=1E-4_q
          ELSE
            DIST(IND)=MAX(D-(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
          ENDIF

          XS(IND)  =XX/DIST(IND)
          YS(IND)  =YY/DIST(IND)
          ZS(IND)  =ZZ/DIST(IND)

          IND=IND+1
        ENDIF
      ENDDO; ENDDO; ENDDO
#else
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
        ZZ=Z-DISZ
        YY=Y-DISY
        XX=X-DISX
        ! the calculation of the | R(ion)-R(mesh)+d | for displaced ions
        ! is done using the well known formula  | R+d | = | R | + d . R/|R|
        ! this improves the stability of finite differences considerable
        IF (D<1E-4_q) THEN
          DIST(IND)=1E-4_q
        ELSE
          DIST(IND)=MAX(D-(DISX*X+DISY*Y+DISZ*Z)/D,1E-10_q)
        ENDIF

        XS(IND)  =XX/DIST(IND)
        YS(IND)  =YY/DIST(IND)
        ZS(IND)  =ZZ/DIST(IND)

        IND=IND+1
      ENDIF
      ENDDO; ENDDO; ENDDO
#endif
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
      END SUBROUTINE SETYLM_AUG2


!************************ SUBROUTINE SETd_IJ ***************************
!
! This subroutine calculates the dipole moment of the augmentation
! charges (according to eq.2 of ref.1).
! 
!***********************************************************************
#ifdef debug
      SUBROUTINE SETd_IJ(WDES,GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,d_IJ,IRDMAX,d_IJ_)
#else
      SUBROUTINE SETd_IJ(WDES,GRIDC_,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,d_IJ,IRDMAX)
#endif      
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

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (grid_3d),TARGET  ::  GRIDC_,GRIDUS
      TYPE (grid_3d),POINTER ::  GRIDC
      TYPE (transit)             C_TO_US  ! index table between GRIDC and GRIDUS
      TYPE (latt)                LATT_CUR
      TYPE (wavedes)             WDES
      TYPE (potcar), POINTER :: PP

      REAL(q)                    SUM(3),d_LM(3,256)

      INTEGER                    IRDMAX   ! allocation required for augmentation
      LOGICAL                    LOVERL,LADDITIONAL

      REAL(q)                    d_IJ(3,LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
#ifdef debug
      REAL(q)                    d_IJ_(3,LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ)
#endif
      REAL(q), ALLOCATABLE ::    XS(:),YS(:),ZS(:)
      REAL(q), ALLOCATABLE ::    DIST(:),DEP(:),YLM(:,:)

#ifdef MPI
      REAL(q), ALLOCATABLE ::    RTMP(:,:,:,:,:)
      ALLOCATE(RTMP(3,LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ))
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
!      WRITE(0,*) 'ion ',NI
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
     &     LATT_CUR%OMEGA,INDMAX,DIST(1),DEP(1))
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
            SUM(1)=SUM(1)+XS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(2)=SUM(2)+YS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            SUM(3)=SUM(3)+ZS(IND)*DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)*YLM(IND,INDPYL)
         ENDDO

         !   add to array d_IJ and make symmetric
         RTMP(1:3,LM+M-1,LMP+MP-1,NI,ISP)=SUM*RINPL
         RTMP(1:3,LMP+MP-1,LM+M-1,NI,ISP)=SUM*RINPL
      ENDDO mp_loop
      ENDDO m_loop

 510  LMP=LMP+2*LLP+1
      ENDDO lp_loop
      LM =LM +2*LL +1
      ENDDO l_loop
   ELSE lpaw
!=======================================================================
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
            SUM(1)=SUM(1)+XS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(2)=SUM(2)+YS(IND)*DEP(IND)*YLM(IND,INDYLM)
            SUM(3)=SUM(3)+ZS(IND)*DEP(IND)*YLM(IND,INDYLM)
            TSUM=TSUM+DEP(IND)*YLM(IND,INDYLM)
         ENDDO
         d_LM(1:3,INDYLM)=SUM*RINPL
      ENDDO

      !   and transform LM -> ll'mm' i.e. d_LM -> d_IJ
      DO IND=1,3
         CALL CALC_DLLMM_BERRY( RTMP(IND,:,:,NI,ISP),d_LM(IND,:),PP)
      ENDDO
      ENDIF lpaw
      ENDDO spin

!=======================================================================
      ENDDO ion
!=======================================================================
      DEALLOCATE(XS,YS,ZS,DIST,DEP,YLM)

      CALLMPI( M_sum_d(GRIDC%COMM, RTMP, 3*LMDIM*LMDIM*T_INFO%NIONS*WDES%NCDIJ))      

 2000 CONTINUE      

#ifdef debug
      d_IJ_=RTMP
#endif

#ifdef MPI
      ion2: DO NI=1,T_INFO%NIONS
      NIP=NI_LOCAL(NI, WDES%COMM_INB)
      IF (NIP==0) CYCLE ion2
         DO ISP=1,WDES%NCDIJ
            d_IJ(:,:,:,NIP,ISP)=RTMP(:,:,:,NI,ISP)
         ENDDO
      ENDDO ion2
      DEALLOCATE(RTMP)
#else
#undef RTMP
#endif   
 
      RETURN
      END SUBROUTINE SETd_IJ


!************************ SUBROUTINE SET_EV ****************************
!
! This subroutine calculates the "expectation-value" (EV) term, which 
! appears in eq.19 of ref.1, for a specific k-point, i.e.
!
!   Sum_t{ Sum_ij{ d_t_ij <U_nk|B_k_ti> <B_k_tj|U_nk> } }
!
!***********************************************************************

      SUBROUTINE SET_EV(WDES,W,ISP,LMDIM,d_IJ,LOVERL,RSUM)
      
      USE prec
      USE wave
      USE mpimy
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavedes)             WDES
      TYPE (wavespin)            W
      TYPE (wavedes1)            WDES1
      TYPE (wavefun1)            W1
      REAL(q)                    RSUM(3),RSUM_(3)
      LOGICAL                    LOVERL
      REAL(q)                    d_IJ(3,LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      

      RSUM=0
   
      IF (LOVERL) THEN
#ifdef MPI
         IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
            CALLMPI( M_stop('SET_EV: KPAR>1 not implemented, sorry.') )
!PK Probably skipping non-owned k-points is sufficient
            STOP
         END IF
#endif
      kpoint: DO IK=1,WDES%NKPTS
      bands: DO IB=1,WDES%NBANDS
          
         CALL SETWDES(WDES,WDES1,IK)
         CALL SETWAV(W,W1,WDES1,IB,ISP)
         
         RSUM_=0
          
         spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
         DO ISPINOR_=0,WDES1%NRSPINORS-1

         NPRO =ISPINOR *WDES1%NPRO/2
         NPRO_=ISPINOR_*WDES1%NPRO/2

         NIS=1

         DO NT=1,WDES1%NTYP
         LMMAXC=WDES1%LMMAX(NT)
         IF (LMMAXC==0) GOTO 230

         DO NI=NIS,WDES1%NITYP(NT)+NIS-1
               DO L=1,LMMAXC
               DO LP=1,LMMAXC
                  RSUM_(1:3)=RSUM_(1:3)+ &
                 &  d_IJ(1:3,LP,L,NI,ISP+ISPINOR_+2*ISPINOR)*W1%CPROJ(LP+NPRO_)*GCONJG(W1%CPROJ(L+NPRO))
               ENDDO
               ENDDO

               NPRO = LMMAXC+NPRO
               NPRO_= LMMAXC+NPRO_
         ENDDO
 230     NIS = NIS+WDES1%NITYP(NT)
         ENDDO
         ENDDO
         ENDDO spinor

      WEIGHT=WDES1%RSPIN*W1%FERWE*WDES%WTKPT(IK)
      RSUM=RSUM+RSUM_*WEIGHT

      ENDDO bands
      ENDDO kpoint
      ENDIF

#ifdef MPI
      CALLMPI( M_sum_d( WDES1%COMM, RSUM, 3))
#endif      

      RETURN
      END SUBROUTINE SET_EV

      
!***********************************************************************
!
!***********************************************************************
      
      SUBROUTINE CALC_DLLMM_BERRY( DLLMM, DLM, P)
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      REAL(q) DLLMM(:,:)  ! net augmentation charge
      REAL(q) DLM(:)      ! local charge for each L,M
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
      END SUBROUTINE CALC_DLLMM_BERRY


!************************* SUBROUTINE OVERL_BERRY *********************
!
! calculate the result of the overlap-operator acting onto a set of
! wavefunctions
!**********************************************************************

      SUBROUTINE OVERL_BERRY(WDES, LOVERL,LMDIM,ISP,CQIJ,CPROF,CRESUL)
      USE prec
      USE wave
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavedes) WDES
      GDEF CRESUL(WDES%NPROD,WDES%NBANDS),CPROF(WDES%NPROD,WDES%NBANDS)

      LOGICAL LOVERL
      OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

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
      END SUBROUTINE OVERL_BERRY


!************************* SUBROUTINE SHIFT_PROJ **********************
!
! calculate the result of the overlap-operator acting onto a set of
! wavefunctions
!
!**********************************************************************

      SUBROUTINE SHIFT_PROJ(IG,CVALUE,WDES1,LOVERL,LATT_CUR,T_INFO,CPROJ)

      USE prec
      USE wave
      USE lattice
      USE poscar
      USE wave
      USE constant

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (latt) LATT_CUR
      TYPE (type_info) T_INFO
      TYPE (wavedes1)  WDES1
      
      INTEGER IG(3)
      
      COMPLEX(q) CVALUE
      
      GDEF CPROJ(WDES1%NPROD,WDES1%NBANDS)

      LOGICAL LOVERL

      IF (LOVERL) THEN

         BANDS: DO NB=1,WDES1%NBANDS

         spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

         NPRO =ISPINOR *WDES1%NPRO/2

         NIS =1
         DO NT=1,WDES1%NTYP
            LMMAXC=WDES1%LMMAX(NT)
            IF (LMMAXC==0) GOTO 100
            DO NI=NIS,WDES1%NITYP(NT)+NIS-1
               
               NIP=NI_GLOBAL(NI,WDES1%COMM_INB)

               CGDR=CITPI*(T_INFO%POSION(1,NIP)*IG(1)+ &
              &              T_INFO%POSION(2,NIP)*IG(2)+ &
              &               T_INFO%POSION(3,NIP)*IG(3))

!DIR$ IVDEP
!OCL NOVREC
               DO L =1,LMMAXC
                  CPROJ(L+NPRO,NB)=CPROJ(L+NPRO,NB)*EXP(CGDR)*CVALUE
               ENDDO
               NPRO = LMMAXC+NPRO
            ENDDO
 100        NIS = NIS+WDES1%NITYP(NT)
         ENDDO
         ENDDO spinor

         ENDDO BANDS
      ENDIF

      RETURN
      END SUBROUTINE SHIFT_PROJ


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


!************************ SUBROUTINE SET_SV_BERRY ***********************
!************************************************************************

      SUBROUTINE SET_SV_BERRY(G,GRID,WDES1,CVALUE,SV)
      
      USE prec
      USE mgrid
      USE wave

      IMPLICIT NONE      
      TYPE(grid_3d)  GRID
      TYPE(wavedes1) WDES1
      
      INTEGER G(3)
      
      REAL(q) FFTSCA
      COMPLEX(q) CVALUE,CVALUE_
      COMPLEX(q) SVWORK(GRID%MPLWV), SV(GRID%MPLWV)! "shifting potential"
      INTEGER IPW,IX,IY,IZ
      
      SVWORK=0

      FFTSCA=1._q
      CVALUE_=CVALUE
#ifdef gammareal
      IF (G(1)/=0.OR.G(2)/=0.OR.G(3)/=0) THEN
         FFTSCA=1/SQRT(2._q)
      ENDIF
#endif      

      DO IPW=1,WDES1%NGVECTOR
         IX=WDES1%IGX(IPW)
         IY=WDES1%IGY(IPW)
         IZ=WDES1%IGZ(IPW)
#ifdef gammareal
         IF (IX==-G(1).AND.IY==-G(2).AND.IZ==-G(3)) THEN
            IF (CVALUE==(0._q,-1._q)) CVALUE_=-CVALUE
#else
         IF (IX==G(1).AND.IY==G(2).AND.IZ==G(3)) THEN
#endif
            SVWORK(IPW)=CVALUE_*FFTSCA
         ENDIF
      ENDDO
      ! take the "shifting potential" to real space
      CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),SV,SVWORK,GRID)

      RETURN
      END SUBROUTINE



!************************ SUBROUTINE MNMAT *****************************
!
! this subroutine calculates the following matrix elements
!
!  CHAM(n2,n1) = < C(n2) | e (i G r) | C(n1) >
!
! where the reciprocal lattice vector G is handled down by the 
! calling subroutine in the array IG
!
!***********************************************************************

      SUBROUTINE MNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, CQIJ, IG, NK, NKP, ISP, CHAM, LIMAG)
     
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
      INTEGER  IG(3),NK,NKP,ISP
      LOGICAL  LIMAG

! the matrix \int Q_ij(r) dr
      OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      COMPLEX(q) CSET_IN_SV

      GDEF    CHAM(WDES%NB_TOT, WDES%NB_TOT)
      GDEF    CHAM_(WDES%NB_TOT, WDES%NB_TOT)
      LOGICAL LOVERL, DO_REDIS

! work arrays (do max of 16 strips simultaneously)
      PARAMETER (NSTRIPD=16)

      TYPE (wavedes1)    WDES1,WDES2    ! descriptors for one k-point
      TYPE (wavefun1)    W1             ! current wavefunction

! wavefunction in real space
      COMPLEX(q),ALLOCATABLE,TARGET::   CR(:),CH(:,:),CVR(:)
      GDEF,ALLOCATABLE,TARGET::  CPROW(:,:)
! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW_RED(:,:),CH_RED(:,:)
      GDEF   , POINTER :: CPROW_RED(:,:),CPROJ_RED(:,:)
! shifting potential
      COMPLEX(q)  SV(GRID%MPLWV)

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.

      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('MNMAT: KPAR>1 not implemented, sorry.') )
!PK All the Berry phase service routines need updating/checking too
         STOP
      END IF
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

      CALL SETWDES(WDES,WDES1,NK) ; CALL SETWGRID_OLD(WDES1,GRID)
      CALL SETWDES(WDES,WDES2,NKP); CALL SETWGRID_OLD(WDES2,GRID)

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
      CALL OVERL_BERRY(WDES, .TRUE.,LMDIM,ISP,CQIJ,W%CPROJ(:,:,NKP,ISP),CPROW)

      CALL SHIFT_PROJ(IG,CSET_IN_SV,WDES2,LOVERL,LATT_CUR,T_INFO,CPROW(1,1))

    ! redistribute the projected wavefunctions
    ! wavefunctions are still required at this point
      IF (DO_REDIS) THEN
        CALL REDIS_PROJ(WDES2, NBANDS, CPROW(1,1))
        CALL REDIS_PROJ(WDES1, NBANDS, W%CPROJ(1,1,NK,ISP))
      ENDIF

      CHAM=0
      CHAM_=0
      NDONE=0
      
!  get me the shifting potential SV (= e^iGr, or cos(Gr), or sin(Gr)) in real space
      CALL SET_SV_BERRY(IG,GRID,WDES2,CSET_IN_SV,SV)

!  redistribute the bra
      IF (DO_REDIS) CALL REDIS_PW(WDES1, NBANDS, W%CW(1,1,NK,ISP))

  strip: DO NPOS=1,NBANDS,NSTRIP
        NSTRIP_ACT=MIN(NBANDS+1-NPOS,NSTRIP)

       !  calculate e ^(i G r) phi
       !  for a block containing NSTRIP_ACT wavefunctions
        IF (DO_REDIS.AND.NK==NKP) CALL REDIS_PW(WDES1, NSTRIP_ACT, W%CW(1,NPOS,NK,ISP))

        DO N=NPOS,NPOS+NSTRIP_ACT-1

          NDONE=NDONE+1
          NP=N-NPOS+1
          ! fft to real space
          ! shift wavefunction stored in W%CW(1+ISPINOR*NGVECTOR,N,NK,ISP)

!  shift wavefunction stored in W%CW(1+ISPINOR*NGVECTOR,N,NK,ISP)
!  and store result in CH(MM,NP)

          DO ISPINOR=0,WDES%NRSPINORS-1
             CALL FFTWAV(WDES%NGVECTOR(NKP),WDES%NINDPW(1,NKP),CR(1+ISPINOR*GRID%MPLWV),&
            &             W%CW(1+ISPINOR*WDES%NGVECTOR(NKP),N,NKP,ISP),GRID)
          ENDDO

!          !  SV(r) * phi(r)
          RINPLW=1._q/GRID%NPLWV
          CALL SHIFT_WAV(WDES2,GRID,SV,CR(1),CVR(1))
          
!          ! fft of SV(r) * phi(r)
          spinors: DO ISPINOR=0,WDES%NRSPINORS-1
          CALL FFTEXT(WDES%NGVECTOR(NK),WDES%NINDPW(1,NK),CVR(1+ISPINOR*GRID%MPLWV), &
         &             CH(1+ISPINOR*WDES%NGVECTOR(NK),NP),GRID,.FALSE.)
          ENDDO spinors

        ENDDO

    ! redistribute wavefunctions at this point
    ! W%CW is then redistributed up to and including 1...NPOS+NSTRIP_ACT
      IF (DO_REDIS) THEN
         IF (NK==NKP) CALL REDIS_PW(WDES1, NSTRIP_ACT, W%CW(1,NPOS,NKP,ISP))
!        CALL REDIS_PW(WDES1, NSTRIP_ACT, W%CW(1,NPOS,NK,ISP))
         CALL REDIS_PW(WDES1, NSTRIP_ACT, CH(1,1))
      ENDIF

      NPOS_RED  =(NPOS-1)*NCPU+1
      NSTRIP_RED=NSTRIP_ACT*NCPU

! take the inproduce between current stripe and all other wavefunctions
      CALL ORTH1('U', &
        CW_RED(1,1),CH_RED(1,1),CPROJ_RED(1,1), &
        CPROW_RED(1,NPOS_RED),NB_TOT, &
        NPOS_RED, NSTRIP_RED, NPL,NPRO,NRPLWV_RED,NPROD_RED,CHAM(1,1))

      CALL ORTH1('L', &
        CW_RED(1,1),CH_RED(1,1),CPROJ_RED(1,1), &
        CPROW_RED(1,NPOS_RED),NB_TOT, &
        NPOS_RED, NSTRIP_RED, NPL,NPRO,NRPLWV_RED,NPROD_RED,CHAM_(1,1))

!     IF (DO_REDIS) CALL REDIS_PW(WDES1, NBANDS, W%CW(1,1,NK,ISP))

  ENDDO strip
! We have the upper triangle in CHAM, now fill in the lower triangle from CHAM_
      DO N1=1,WDES%NB_TOT-1
      DO N2=N1+1,WDES%NB_TOT
         CHAM(N2,N1)=CHAM_(N2,N1)
      ENDDO
      ENDDO
!
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
          WRITE(*,2)N1,(AIMAG(CHAM(N1,N2)),N2=1,NPL2)
        ENDDO
        WRITE(*,*)
#endif
      1 FORMAT(1I2,3X,20F9.5)
      2 FORMAT(1I2,3X,20F9.5)
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

      DEALLOCATE(CPROW, CR, CVR, CH)

      RETURN
      END SUBROUTINE MNMAT
     
      
!************************ SUBROUTINE READER_ADD_ON *********************
!
! This subroutine reads LBERRY, IGPAR and NSTR from INCAR
! ICHARG is set to 11 if the flag LBERRY is found in the INCAR file
! since this is the only sensible option
!
!***********************************************************************

      SUBROUTINE READER_ADD_ON(IU5,IU0,LBERRY,IGPAR,NPPSTR, &
           ICHARG,ISMEAR,SIGMA)
      
      USE prec
      USE base
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      CHARACTER (1)              CHARAC
      
      LOGICAL                    LOPEN,LDUM,LBERRY
      
      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
!
! Calculate Berry-Phase? Default -> .FALSE.
      LBERRY=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LBERRY','=','#',';','L', &
     &  IDUM,RDUM,CDUM,LBERRY,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LBERRY'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
            LBERRY=.FALSE. ! In case of error set LBERRY=.FALSE.
         ENDIF
      ENDIF
      CALL XML_INCAR('LBERRY','L',IDUM,RDUM,CDUM,LBERRY,CHARAC,N)
!
      IF (LBERRY) THEN
! overwrite user defaults for ISMEAR and SIGMA
! this should be save in all cases
         ISMEAR  = 0
         SIGMA   = 0.0001
! If Berry-Phase should be calculated, find G_parallel
         CALL RDATAB(LOPEN,INCAR,IU5,'IGPAR','=','#',';','I', &
        &  IGPAR,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF ((IERR/=0).OR.((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) THEN 
               WRITE(IU0,*)'Error reading item ''IGPAR'' from file INCAR.'
               WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
               WRITE(IU0,*)'Defaulting to LBERRY=.FALSE.'
               LBERRY=.FALSE. ! In case of error set LBERRY=.FALSE. 
            ENDIF
         ENDIF
         IF ((IGPAR.LT.1).OR.(IGPAR.GT.3)) THEN
            IF (IU0>=0) THEN 
               WRITE(IU0,*)'ERROR: IGPAR <1 or >3'
               WRITE(IU0,*)'Defaulting to LBERRY=.FALSE.'
               LBERRY=.FALSE. ! In case of error set LBERRY=.FALSE. 
            ENDIF
         ENDIF
         CALL XML_INCAR('IGPAR','I',IGPAR,RDUM,CDUM,LDUM,CHARAC,N)

         CALL RDATAB(LOPEN,INCAR,IU5,'NPPSTR','=','#',';','I', &
        &  NPPSTR,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF ((IERR/=0).OR.((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) THEN 
               WRITE(IU0,*)'Error reading item ''NPPSTR'' from file INCAR.'
               WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
               WRITE(IU0,*)'Defaulting to LBERRY=.FALSE.'
               LBERRY=.FALSE. ! In case of error set LBERRY=.FALSE. 
            ENDIF
         ENDIF
         CALL XML_INCAR('NPPSTR','I',NPPSTR,RDUM,CDUM,LDUM,CHARAC,N)
! When not doing a single point Berry-Phase calculation override user default for ICHARG
         IF (NPPSTR/=1) ICHARG=11
      ENDIF
      
      CLOSE(IU5)

      RETURN 
      END SUBROUTINE READER_ADD_ON

!************************ SUBROUTINE XML_WRITE_BERRY *******************
!
! This subroutine writes the parameters of the berry phase routine
!
!***********************************************************************

      SUBROUTINE XML_WRITE_BERRY(LBERRY,IGPAR,NPPSTR)
      
      USE prec
      IMPLICIT NONE
      LOGICAL LBERRY
      INTEGER IGPAR, NPPSTR
    ! local
      REAL(q) :: RDUM
      INTEGER :: IDUM
      COMPLEX(q) :: CDUM
      CHARACTER (1)              CHARAC
      LOGICAL                    LDUM
      
      CALL XML_INCAR('LBERRY','L',IDUM,RDUM,CDUM,LBERRY,CHARAC,1)
      IF (.NOT.LBERRY) RETURN

      CALL XML_INCAR('IGPAR','I',IGPAR,RDUM,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('NPPSTR','I',NPPSTR,RDUM,CDUM,LDUM,CHARAC,1)


      RETURN 
    END SUBROUTINE XML_WRITE_BERRY

!************************ SUBROUTINE WRITE_BERRY_PARA ******************
!
! This subroutine writes LBERRY, IGPAR and NPPSTR to OUTCAR
!
!***********************************************************************

      SUBROUTINE WRITE_BERRY_PARA(IUNOUT,LBERRY,IGPAR,NPPSTR)
      
      USE prec

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      LOGICAL                    LBERRY
      
      IF (IUNOUT>=0 .AND. LBERRY ) THEN
         WRITE(IUNOUT,1) LBERRY
         IF (LBERRY) WRITE(IUNOUT,2) IGPAR,NPPSTR
      ENDIF
      
    1 FORMAT(' Parameters for Berry phase evaluation:'/ &
     &       '   LBERRY = ',L6,'    evaluation of Berry phase?')
    2 FORMAT('   IGPAR  = ',I6,'    parallel direction'/ &
     &       '   NPPSTR = ',I6,'    k-mesh size in direction IGPAR')      

      RETURN 
      END SUBROUTINE WRITE_BERRY_PARA
      

!************************ SUBROUTINE IONIC_POL *************************
!
! calculate the contribution from the ions to the polarization
! this requires two calculations
!  1) first for the unpolarized system
!  2) second for the polarized system
! a central point is supplied by the vector POSCEN in the INCAR file
! this central point must be chosen so that the ions are on the
! same side of POSCEN (in terms of a minimum image convention) 
! for calculation 1 and 2. Otherwise the results are unpredictable
!
!***********************************************************************

      SUBROUTINE IONIC_POL(LATT_CUR, T_INFO, ZVAL, POSCEN, DIP_ION)

      USE prec
      USE lattice
      USE poscar

      TYPE (latt)  :: LATT_CUR
      TYPE (type_info)   T_INFO
      REAL(q)     :: ZVAL(T_INFO%NTYP)
      REAL(q)     :: POSCEN(3)
      REAL(q)     :: DIP_ION(3)
!     local
      
      REAL(q)     :: POS(3)
      

      IF (POSCEN(1) == -100) POSCEN(1) = 0.5
      IF (POSCEN(2) == -100) POSCEN(2) = 0.5
      IF (POSCEN(3) == -100) POSCEN(3) = 0.5

      DIP_ION=0

      NIS=1
      DO NT =1,T_INFO%NTYP
      DO ION=NIS,T_INFO%NITYP(NT)+NIS-1

         POS=MOD(T_INFO%POSION(:,ION)-POSCEN(:)+10.5_q,1._q)-0.5_q 
         DIP_ION=DIP_ION-ZVAL(NT)*POS
      ENDDO
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO

      CALL DIRKAR(1,DIP_ION ,LATT_CUR%A)

      END SUBROUTINE IONIC_POL
      

!************************ SUBROUTINE BERRY *****************************
!
! Output to OUTCAR:
!   <R>ev and <R>bp, respectively the "expectation value" and
!   "Berry phase" contributions to the polarization (see ref.1).
!    (Type: grep '<R>' OUTCAR)
!
! The sum of these contributions
!
!   <R>ev + <R>bp = (Omega*P)/(|e|)
!
! where 
!
!   Omega = cell volume
!       P = polarization
!       f = occupation number of valence states
!           (f=2 for spin degenerate systems)
!       e = unit of charge
!
!***********************************************************************
      SUBROUTINE BERRY(WDES,W,GRID,GRIDC,GRIDUS,C_TO_US,LATT_CUR,KPOINTS, &
     &  IGPAR,NPPSTR,P,T_INFO,LMDIM,CQIJ,IRDMAX,LOVERL,IUNOUT,POSCEN)

      USE prec
      USE pseudo
      USE poscar
      USE mgrid
      USE mkpoints
      USE lattice
      USE wave
      USE constant
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)           T_INFO
      TYPE (potcar)              P(T_INFO%NTYP)
      TYPE (latt)                LATT_CUR
      TYPE (grid_3d)             GRID,GRIDC,GRIDUS
      TYPE (transit)             C_TO_US
      TYPE (kpoints_struct)      KPOINTS
      TYPE (wavedes)             WDES
      TYPE (wavespin)            W
      TYPE (wavedes1)            WDES1
      TYPE (wavefun1)            W1

      INTEGER                    IRDMAX
      INTEGER                    IG(3)
      INTEGER, ALLOCATABLE ::    KPTSTR(:,:)
      
      OVERLAP                    CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q)                    d_IJ(3,LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
#ifdef debug
      REAL(q)                    d_IJ_(3,LMDIM,LMDIM,T_INFO%NIONS,WDES%NCDIJ)
#endif
      GDEF                       CTMP(WDES%NB_TOT,WDES%NB_TOT)
      COMPLEX(q)                 CMK(WDES%NB_TOT,WDES%NB_TOT)
      COMPLEX(q)                 CMK_(WDES%NB_TOT,WDES%NB_TOT)

      REAL(q)                    R_EV(3,WDES%ISPIN)
      REAL(q)                    R_BP(3,WDES%ISPIN)   
      REAL(q)                    RSUM(3)

      LOGICAL                    LOVERL,LIMAG

      REAL(q)                    POSCEN(3)
      REAL(q)                    DIP_ION(3)

      REAL(q),PARAMETER ::       TINY=1E-4_q

      INTEGER, ALLOCATABLE ::    IPVT(:)
      COMPLEX(q), ALLOCATABLE :: CWORK(:)

#ifdef SCSL
      INTEGER                       :: INFO
      DOUBLE PRECISION              :: ANORM
      DOUBLE PRECISION, ALLOCATABLE :: AWORK(:), RWORK(:)
      COMPLEX(q)                    :: CDET, CDETMK
#else
      COMPLEX(q)                    :: CDET(2),CDETMK
#endif

      REAL(q)                    IMLNBP
      COMPLEX(q)                 BP_MEAN
      COMPLEX(q), ALLOCATABLE :: BP(:,:)

#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
      NCPU   =WDES%COMM_INTER%NCPU ! number of procs involved in band dis.
#else
      NODE_ME=0
      IONODE =0
      NCPU=1
#endif
!
! Calculate dipole moment of augmentation charges
!      

#ifdef debug
      CALL SETd_IJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,d_IJ,IRDMAX,d_IJ_)

      IF (NODE_ME==IONODE) THEN

      WRITE(IUNOUT,'(/A)')'Dipole moment of augmentation charges:'

      ion: DO NI=1,T_INFO%NIONS
      
        WRITE(IUNOUT,'(/A,I4)')'ION #',NI
      
        NT=T_INFO%ITYP(NI)
        ! for this ion (this type of ion) no depletion charge
        IF (P(NT)%PSDMAX==0) CYCLE

        ! loop over all channels (l,epsilon)
        LM =1
        l_loop:  DO L =1,P(NT)%LMAX
        LMP=LM
        lp_loop: DO LP=L,P(NT)%LMAX
!         IF (P(NT)%NDEP(L,LP)==0) GOTO 510

          ! quantum numbers l and lp of these two channels
          LL =P(NT)%LPS(L )
          LLP=P(NT)%LPS(LP)
          
          ! loop over all m mp
          m_loop:  DO M=1,2*LL+1
          MPLOW=1 ; IF (L==LP) MPLOW=M
          mp_loop: DO MP=MPLOW,2*LLP+1

            !   calculate the indices into the array containing the spherical
            !   harmonics
            INDYLM =LL**2   +M
            INDPYL =LLP**2  +MP

            WRITE(IUNOUT,90) LL,M,LLP,MP,d_IJ_(1:3,LM+M-1,LMP+MP-1,NI,1)
90          FORMAT('l =',I2,'  m =',I2,'  lp =',I2,'  mp =',I2,'  d = (',3F12.5,' )')
  	 
          ENDDO mp_loop
          ENDDO m_loop

510       LMP=LMP+2*LLP+1
        ENDDO lp_loop
        LM =LM +2*LL +1
        ENDDO l_loop

      ENDDO ion
      ENDIF
#else
      CALL SETd_IJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &  LOVERL,LMDIM,d_IJ,IRDMAX)
#endif
      
!=======================================================================
! Setup k-mesh
!=======================================================================
      NSTR=KPOINTS%NKPTS/NPPSTR     ! Number of points per string
      
      IF (NPPSTR*NSTR.NE.KPOINTS%NKPTS) GOTO 1001 ! not a (nppstr x nstr) kpoints mesh

      IF (IUNOUT>=0) THEN
         WRITE(IUNOUT,'(/A)')'k-mesh defined as follows:'
         WRITE(IUNOUT,110) IGPAR
      ENDIF      
 110  FORMAT(/'IGPAR = ',I4)    

      IF (IUNOUT>=0) THEN
         WRITE(IUNOUT,120) KPOINTS%NKPTS,NSTR,NPPSTR
         WRITE(IUNOUT,*)
      ENDIF      
 120  FORMAT('NKPTS = ',I4,'  NSTR = ',I4,'  NPPSTR = ',I4) 
 
! Allocate and fill kpoint strings index table
      ALLOCATE(KPTSTR(NPPSTR,NSTR))
      NK=0
      DO ISTR=1,NSTR
         DO IK=1,NPPSTR
            NK=NK+1
            KPTSTR(IK,ISTR)=NK
         ENDDO
      ENDDO
      
      ALLOCATE(BP(NSTR,WDES%ISPIN))

      R_EV=0
      
 spin:DO ISP=1,WDES%ISPIN
!=======================================================================
! Sum <R>ev terms over k-points and occupied bands
!=======================================================================
      CALL SET_EV(WDES,W,ISP,LMDIM,d_IJ,LOVERL,R_EV(:,ISP))

!=======================================================================
! Calculate and sum <R>bp terms
!=======================================================================
      k_perp: DO ISTR=1,NSTR
      CDETMK=1
      k_para: DO IK=1,NPPSTR

         NK=KPTSTR(IK,ISTR)
         IF (IK.LT.NPPSTR) THEN
            NKP=KPTSTR(IK+1,ISTR)
            IG=0
         ELSE
            NKP=KPTSTR(1,ISTR)
            IG=0
            IG(IGPAR)=-1
         ENDIF
                  
         CTMP=0         
         LIMAG=.FALSE.
         CALL MNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, CQIJ, IG, NK, NKP, ISP, CTMP, LIMAG)
         CMK_=CTMP
#ifdef gammareal
         CTMP=0
         LIMAG=.TRUE.
         CALL MNMAT(W, WDES,GRID,GRIDC,GRIDUS,C_TO_US, LATT_CUR, T_INFO, P, &
     &     LMDIM, LOVERL, CQIJ, IG, NK, NKP, ISP, CTMP, LIMAG)
         CMK_=CMK_+CMPLX(0,CTMP,q)
#endif
         CMK=0
         N=0;MN=0
         loop_n: DO N1=1,WDES%NB_TOT
            IF (W%FERTOT(N1,NK,ISP).LE.TINY) CYCLE loop_n
            N=N+1;M=0
            loop_m: DO N2=1,WDES%NB_TOT  
               IF (W%FERTOT(N2,NKP,ISP).LE.TINY) CYCLE loop_m
               M=M+1
               CMK(N,M)=CMK_(N1,N2)
            ENDDO loop_m
            MN=MN+M
         ENDDO loop_n
         
!=======================================================================
! Calculate determinant
!=======================================================================
         IFAIL=0
         IF (MN.EQ.N**2) THEN
            ! Find LU decomposition and reciprocal condition of M_k
#ifdef SCSL
            ALLOCATE(IPVT(N), CWORK(2*N), AWORK(N), RWORK(2*N))
            ANORM = ZLANGE('1', WDES%NB_TOT, N, CMK, WDES%NB_TOT, AWORK)
            CALL ZGETRF(WDES%NB_TOT, N, CMK, IPVT, INFO)
            CALL ZGECON('1', N, CMK, WDES%NB_TOT, ANORM, RCOND,         &
     &                  CWORK, RWORK, INFO)
            IF (INFO/=0) THEN
               WRITE(*,*) 'SCSL: Routine ZGETRF failed!', INFO
               STOP
            ENDIF
#else
            ALLOCATE(IPVT(N),CWORK(N))
            CALL ZGECO(CMK,WDES%NB_TOT,N,IPVT,RCOND,CWORK)
#endif
            IF (RCOND.GT.0._q) THEN
               ! Calculate determinant of M_k
#ifdef SCSL
               CDET = (1.0D0, 0.0D0)
               DO I = 1, N
                  CDET = CMK(I,I) * CDET
                  IF (IPVT(I) .NE. I) THEN
                     IPVT(IPVT(I)) = IPVT(I)
                     CDET = -CDET
                  ENDIF
               END DO
               CDETMK = CDETMK * CDET
#else
               CALL ZGEDI(CMK,WDES%NB_TOT,N,IPVT,CDET,CWORK,10)
               CDETMK=CDETMK*CDET(1)*10._q**CDET(2)
#endif
            ELSE
               IFAIL=-1 ! Matrix M_k is singular to working precision
            ENDIF
#ifdef SCSL
            DEALLOCATE(IPVT, CWORK, RWORK, AWORK)
#else
            DEALLOCATE(IPVT,CWORK)
#endif
         ELSE
            IFAIL=1     ! Matrix M_k is not a nxn matrix
         ENDIF

         IF (IFAIL.NE.0) GOTO 1000 ! Error in calculation of determinant

         ENDDO k_para

         BP(ISTR,ISP)=CDETMK/SQRT(REAL(CDETMK, q)**2+AIMAG(CDETMK)**2)
!        BP(ISTR,ISP)=CDETMK

      ENDDO k_perp
      ENDDO spin

!=======================================================================
! Postprocessing Berry phases, and output
!=======================================================================

      R_BP=0
      DO ISP=1,WDES%ISPIN
         BP_MEAN=0
         DO ISTR=1,NSTR
            BP_MEAN=BP_MEAN+WDES%WTKPT((ISTR-1)*NPPSTR+1)*NPPSTR*BP(ISTR,ISP)
         ENDDO

         IF (IUNOUT>=0) THEN
            WRITE(IUNOUT,'(/A,2F10.5,A)') 'Average Det|M_k|=  (',BP_MEAN,' )'
            WRITE(IUNOUT,*)
         ENDIF
         
         DO ISTR=1,NSTR
            IMLNBP=AIMAG(LOG(BP(ISTR,ISP)/BP_MEAN))
            IF (IUNOUT>=0) THEN
               WRITE(IUNOUT,'(A,I4,A,F12.8)')'K-point string # ',ISTR,' weight=', &
               WDES%WTKPT((ISTR-1)*NPPSTR+1)*NPPSTR
               DO IK=1,NPPSTR
                  WRITE(IUNOUT,'(3F10.6)') KPOINTS%VKPT(1:3,KPTSTR(IK,ISTR))
               ENDDO
               WRITE(IUNOUT,'(14X,A,2F10.5,A)') '       Det|M_k|=  (',BP(ISTR,ISP),' )'
               WRITE(IUNOUT,'(14X,A,F12.5,A)') 'Im ln[Det|M_k|]= ',-AIMAG(LOG(BP(ISTR,ISP)))/TPI,' electrons'
               WRITE(IUNOUT,'(A,F12.5)',ADVANCE='No') 'Im ln[Det|M_k|/<Det|M_k|>_av]= ',ABS(IMLNBP)
               IF ((ABS(IMLNBP)>PI/2).AND.(IUNOUT>=0)) THEN
                  WRITE(IUNOUT,'(A)') ' <= WARNING: larger than pi/2, not well clustered.'
               ELSE
                  WRITE(IUNOUT,*)
               ENDIF
               WRITE(IUNOUT,*)
            ENDIF
            R_BP(:,ISP)=R_BP(:,ISP)+ &
           &   WDES%WTKPT((ISTR-1)*NPPSTR+1)*NPPSTR*LATT_CUR%A(:,IGPAR)*IMLNBP
         ENDDO
         
         R_BP(:,ISP)=-(R_BP(:,ISP)+LATT_CUR%A(:,IGPAR)*AIMAG(LOG(BP_MEAN)))*WDES%RSPIN/TPI
      ENDDO

      CALL IONIC_POL(LATT_CUR, T_INFO, P%ZVALF, POSCEN, DIP_ION)

      IF (IUNOUT>=0) THEN
         WRITE(IUNOUT,*)
         IF (WDES%ISPIN==2) THEN
            WRITE(IUNOUT,'(A)',ADVANCE='No') ' Spin component 1'
            WRITE(IUNOUT,130) R_EV(:,1)
            WRITE(IUNOUT,135) R_BP(:,1)
            WRITE(IUNOUT,*)
            WRITE(IUNOUT,'(A)',ADVANCE='No') ' Spin component 2'            
            WRITE(IUNOUT,140) R_EV(:,2)
            WRITE(IUNOUT,145) R_BP(:,2)
            WRITE(IUNOUT,*)
            RSUM(:)=R_EV(:,1)+R_EV(:,2)+R_BP(:,1)+R_BP(:,2)
         ELSE
            WRITE(IUNOUT,'(A)',ADVANCE='No') '                 '            
            WRITE(IUNOUT,130) R_EV(:,1)
            WRITE(IUNOUT,135) R_BP(:,1)         
            RSUM(:)=R_EV(:,1)+R_BP(:,1)
         ENDIF
         WRITE(IUNOUT,*)
         WRITE(IUNOUT,150) RSUM
         WRITE(IUNOUT,*)
         WRITE(IUNOUT,160) DIP_ION
         WRITE(IUNOUT,*) 
      ENDIF

 130  FORMAT(                 '               e<r>_ev=(',3F12.5,' ) e*Angst')
 135  FORMAT('                                e<r>_bp=(',3F12.5,' ) e*Angst')
 140  FORMAT(                 '               e<r>_ev=(',3F12.5,' ) e*Angst')
 145  FORMAT('                                e<r>_bp=(',3F12.5,' ) e*Angst')
 150  FORMAT(' Total electronic dipole moment: p[elc]=(',3F12.5,' ) e*Angst')
 160  FORMAT('            ionic dipole moment: p[ion]=(',3F12.5,' ) e*Angst')


      DEALLOCATE(BP)
      DEALLOCATE(KPTSTR)

      RETURN
!
! Error handling 
!
 1000 CONTINUE   ! Case IFAIL<>0
      IF (IUNOUT>=0) THEN
      WRITE(IUNOUT,*)'Error in subroutine BERRY: did not find all determinants'
      SELECT CASE(IFAIL.LT.0)
         CASE(.TRUE.)
            WRITE(IUNOUT,200) ISTR,IK-1
         CASE(.FALSE.)
            WRITE(IUNOUT,210) ISTR,IK-1
      END SELECT
      ENDIF
 200  FORMAT(/'Matrix CMK is singular to working precision for'/, &
     &  'ISTR = ',I4,' j = ',I4/)
 210  FORMAT(/'Matrix CMK is not an nxn matrix for'/, &
     &  'ISTR = ',I4,' j = ',I4/)
      DEALLOCATE(KPTSTR)
      STOP

 1001 CONTINUE   ! Case IK<>NPPSTR
      IF (IUNOUT>=0) THEN
      WRITE(IUNOUT,*)'Error in subroutine BERRY: problem with k-mesh'
      WRITE(IUNOUT,300) KPOINTS%NKPTS,NSTR,NPPSTR
      ENDIF
 300  FORMAT(/'NKPTS = ',I4,' NSTR = ',I4,' NPPSTR = ',I4)     
      DEALLOCATE(KPTSTR)
      STOP
           
      END SUBROUTINE BERRY



!************************************************************************
!
!  Read UNIT=14: KPOINTS
!  and generate the k-points for calculating polarization in direction
!   IGPAR
!  this reader supports both manual and automatic k-point generations
!  the gambler mode A should not be selected 
!  (the generating vectors must be parallel to the reciprocal lattice
!   vectors)
!
!************************************************************************

      SUBROUTINE RD_KPOINTS_BERRY(KPOINTS,NPPSTR,IGPAR, &
           LATT_CUR,LINVERSION,IU6,IU0)
      USE prec
      USE mkpoints
      USE lattice

      IMPLICIT NONE

      TYPE (kpoints_struct) KPOINTS
      TYPE (latt)        LATT_CUR
      INTEGER   NPPSTR  ! number of points per string
      INTEGER   IGPAR   ! direction G_parallel
      INTEGER   IU0,IU6 ! units for output
      CHARACTER (1) CSEL
      REAL(q)    BK(3,3),SHIFT(3)
      LOGICAL   LINVERSION
! required for reallocation
      REAL(q),POINTER   :: VKPT(:,:),WTKPT(:)
      INTEGER,POINTER:: IDTET(:,:)

! local variables
      INTEGER KTH,NKPX,NKPY,NKPZ,NKP,ITET,I,NT,NK
      REAL(q) RKLEN,WSUM,TMP(3),WEIGHT
      INTEGER NKPTS_REPLICATED,INDEX_BASE,ISTR
      REAL(q) EW(3,3)
      DATA EW /1,0,0, 0,1,0 , 0,0,1/

      OPEN(UNIT=14,FILE='KPOINTS',STATUS='OLD')

      KPOINTS%NKDIM=NKDIMD
      KPOINTS%NTET=0

      ALLOCATE(VKPT(3,NKDIMD),WTKPT(NKDIMD),IDTET(0:4,NTETD))
      ALLOCATE(KPOINTS%IKPT(IKPTD,IKPTD,IKPTD))

      IF (IU6>=0) WRITE(IU6,*)
!-----k-points
      READ(14,'(A40)',ERR=70111,END=70111) KPOINTS%SZNAMK
      IF (IU6>=0) WRITE(IU6,*)'KPOINTS: ',KPOINTS%SZNAMK
      READ(14,*,ERR=70111,END=70111) KPOINTS%NKPTS
      IF (KPOINTS%NKPTS>KPOINTS%NKDIM) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'ERROR in RD_KPOINTS (mkpoints.F): increase NKDIMD to', KPOINTS%NKPTS
        STOP
      ENDIF
      READ(14,'(A1)',ERR=70111,END=70111) CSEL
      IF (CSEL=='K'.OR.CSEL=='k'.OR. &
     &    CSEL=='C'.OR.CSEL=='c') THEN
        CSEL='K'
        IF (IU6 >=0 .AND. KPOINTS%NKPTS>0)  &
     &      WRITE(IU6,*)' k-points in cartesian coordinates'
      ELSE
        IF (IU6 >= 0 .AND. KPOINTS%NKPTS>0)  &
     &     WRITE(IU6,*)' k-points in reciprocal lattice'
      ENDIF

      IF (KPOINTS%NKPTS>0) THEN
! Read in a given set of arbitrary k-points:
         WSUM=0
         DO NKP=1,KPOINTS%NKPTS
            READ(14,*,ERR=70111,END=70111) &
     &  VKPT(1,NKP),VKPT(2,NKP),VKPT(3,NKP), &
     &  WTKPT(NKP)
            WSUM=WSUM+WTKPT(NKP)
         ENDDO

         IF (WSUM==0) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'ERROR: sum of weights is zero'
            STOP
         ENDIF

         IF (CSEL=='K') THEN

            VKPT(:,1:KPOINTS%NKPTS)=  &
     &          VKPT(:,1:KPOINTS%NKPTS)/LATT_CUR%SCALE

            CALL KARDIR(KPOINTS%NKPTS,VKPT,LATT_CUR%A)
         ENDIF

         WTKPT(1:KPOINTS%NKPTS)=WTKPT(1:KPOINTS%NKPTS)/WSUM

         IF (KPOINTS%LTET) THEN
! Read in tetrahedra if you want to use tetrahedron method:
            READ(14,'(A)',ERR=70111,END=70111) CSEL
            IF (CSEL/='T' .AND. CSEL/='t') GOTO 70111
            READ(14,*,ERR=70111,END=70111) KPOINTS%NTET,KPOINTS%VOLWGT
            DO ITET=1,KPOINTS%NTET
               READ(14,*,ERR=70111,END=70111) (IDTET(KTH,ITET),KTH=0,4)
            ENDDO
         ENDIF

      ELSE
! Automatic generation of a mesh if KPOINTS%NKPTS<=0:
         IF (IU6>=0 ) WRITE(IU6,'(/A)') 'Automatic generation of k-mesh.'
         SHIFT(1)=0._q
         SHIFT(2)=0._q
         SHIFT(3)=0._q
! k-lattice basis vectors in cartesian or reciprocal coordinates?
         IF ((CSEL/='M').AND.(CSEL/='m').AND. &
     &       (CSEL/='G').AND.(CSEL/='g').AND. &
     &       (CSEL/='A').AND.(CSEL/='a')) THEN
! Here give a basis and probably some shift (warning this shift is
! always with respect to the point (0,0,0) ... !
            IF (CSEL=='K'.OR.CSEL=='k'.OR. &
     &          CSEL=='C'.OR.CSEL=='c') THEN
               CSEL='K'
               IF (IU6>=0 ) WRITE(IU6,*)' k-lattice basis in cartesian coordinates'
            ELSE
               IF (IU6>=0 )WRITE(IU6,*)' k-lattice basis in reciprocal lattice'
            ENDIF
! Read in the basis vectors for the k-lattice (unscaled!!):
            READ(14,*,ERR=70111,END=70111) BK(1,1),BK(2,1),BK(3,1)
            READ(14,*,ERR=70111,END=70111) BK(1,2),BK(2,2),BK(3,2)
            READ(14,*,ERR=70111,END=70111) BK(1,3),BK(2,3),BK(3,3)
! Correct scaling with LATT_CUR%SCALE ('lattice constant'):
            IF (CSEL=='K') BK=BK/LATT_CUR%SCALE
! Routine IBZKPT needs cartesian coordinates:
            IF (CSEL/='K') THEN
               CALL DIRKAR(3,BK,LATT_CUR%B)
            ENDIF
! Read in the shift of the k-mesh: these values must be given in
! k-lattice basis coordinates (usually 0 or 1/2 ...):
            READ(14,*,ERR=70112,END=70112) SHIFT(1),SHIFT(2),SHIFT(3)
70112       CONTINUE
         ELSE IF ((CSEL=='A').OR.(CSEL=='a')) THEN
            READ(14,*) RKLEN
            NKPX =MAX(1._q,RKLEN*LATT_CUR%BNORM(1)+0.5_q)
            NKPY =MAX(1._q,RKLEN*LATT_CUR%BNORM(2)+0.5_q)
            NKPZ =MAX(1._q,RKLEN*LATT_CUR%BNORM(3)+0.5_q)
            IF (IU6 >= 0 ) THEN
              IF (IU0>=0) &
              WRITE(IU0,99502)NKPX,NKPY,NKPZ
              WRITE(IU6,99502)NKPX,NKPY,NKPZ
            ENDIF
99502       FORMAT( 'generate k-points for:',3I4)
            DO 99501 I=1,3
               BK(I,1)=LATT_CUR%B(I,1)/FLOAT(NKPX)
               BK(I,2)=LATT_CUR%B(I,2)/FLOAT(NKPY)
               BK(I,3)=LATT_CUR%B(I,3)/FLOAT(NKPZ)
99501       CONTINUE
         ELSE
! Here we give the Monkhorst-Pack conventions ... :
            READ(14,*,ERR=70111,END=70111) NKPX,NKPY,NKPZ
! modifications for the Berry phase
            IF      (IGPAR == 1) THEN
               NKPX=1
            ELSE IF (IGPAR == 2) THEN
               NKPY=1
            ELSE IF (IGPAR == 3) THEN
               NKPZ=1
            ENDIF
! end modifications

! Shift (always in units of the reciprocal lattice vectors!):
            READ(14,*,ERR=70113,END=70113) SHIFT(1),SHIFT(2),SHIFT(3)
70113       CONTINUE
! Internal rescaling and centering according to Monkhorst-Pack:
            IF ((CSEL=='M').OR.(CSEL=='m')) THEN
               SHIFT(1)=SHIFT(1)+0.5_q*MOD(NKPX+1,2)
               SHIFT(2)=SHIFT(2)+0.5_q*MOD(NKPY+1,2)
               SHIFT(3)=SHIFT(3)+0.5_q*MOD(NKPZ+1,2)
            ENDIF
            DO I=1,3
               BK(I,1)=LATT_CUR%B(I,1)/FLOAT(NKPX)
               BK(I,2)=LATT_CUR%B(I,2)/FLOAT(NKPY)
               BK(I,3)=LATT_CUR%B(I,3)/FLOAT(NKPZ)
            ENDDO
! At this point hope that routine IBZKPT accepts all ... !
         ENDIF
! Find all irreducible points in the first Brillouin zone ... :
         CALL IBZKPT(LATT_CUR%B(1,1),BK,SHIFT,KPOINTS%NKPTS, &
              VKPT(1,1),WTKPT(1),KPOINTS%NKDIM, &
              KPOINTS%LTET,KPOINTS%NTET,IDTET(0,1),NTETD,KPOINTS%VOLWGT, &
              KPOINTS%IKPT(1,1,1),IKPTD,LATT_CUR%SCALE,LINVERSION,.FALSE.,.FALSE.,IU6)

! modifications for the Berry phase
         NKPTS_REPLICATED=KPOINTS%NKPTS*NPPSTR
         DO I=KPOINTS%NKPTS,1,-1
            INDEX_BASE=(I-1)*NPPSTR
            TMP(:)=VKPT(:,I)
            WEIGHT=WTKPT(I)
            DO ISTR=1,NPPSTR
               VKPT(:,INDEX_BASE+ISTR)=VKPT(:,I)+(ISTR-1)*EW(:,IGPAR)/NPPSTR
               WTKPT(INDEX_BASE+ISTR)=WEIGHT/NPPSTR
            ENDDO
         ENDDO
         KPOINTS%NKPTS=NKPTS_REPLICATED
! end modifications
      ENDIF

! set old k-points
      GOTO 70222
70111 CONTINUE
      IF (IU6>=0) &
      WRITE(IU6,*) 'Error reading file KPOINTS. Stopping execution.'
      IF (IU0>=0) &
      WRITE(IU0,*) 'Error reading file KPOINTS. Stopping execution.'
      STOP
70222 CONTINUE
#ifdef gammareal
      IF (KPOINTS%NKPTS/=1._q .OR. &
     &    VKPT(1,1)**2+VKPT(2,1)**2 &
     &   +VKPT(3,1)**2>1E-10_q) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'ERROR: You are using the Gamma-point only version', &
                     ' with a wrong k-point setting'
        STOP
      ENDIF
#endif
      DEALLOCATE(KPOINTS%IKPT)
!
!  reallocate everthing with the minimum number of kpoints set
!
      NK=KPOINTS%NKPTS
      NT=MAX(KPOINTS%NTET,1)

      ALLOCATE(KPOINTS%VKPT(3,NK),KPOINTS%WTKPT(NK),KPOINTS%IDTET(0:4,NT))
      KPOINTS%VKPT = VKPT(1:3,1:NK)
      KPOINTS%WTKPT= WTKPT(1:NK)
      KPOINTS%IDTET= IDTET(0:4,1:NT)
      DEALLOCATE(VKPT,WTKPT,IDTET)
      KPOINTS%NKDIM=NK

      CLOSE(UNIT=14)

      CALL SETUP_KPOINTS_STATIC(KPOINTS)      

      RETURN
      END SUBROUTINE


      END MODULE ELPOL
