!#define debug
#include "symbol.inc"
!***********************************************************************
!
!  this module contains all the routines required to support 
!  reciprocal space presentation of the non local projection operators
!  on a plane wave grid
!
!***********************************************************************

MODULE nonl_struct_def
  USE wave
  USE prec
  !
  !  structure required to support non local projection operators in recip space
  !
  TYPE nonl_struct
     !only NONL_S
     LOGICAL LRECIP                ! structure set up ?
     INTEGER NTYP                  ! number of types
     INTEGER NIONS                 ! number of ions
     INTEGER NK                    ! kpoint for which CREXP is set up
     INTEGER SELECTED_ION          ! allows to generate a projector for a single ion
     INTEGER, POINTER :: NITYP(:)  ! number of ions for each type
     INTEGER, POINTER :: LMMAX(:)  ! max l-quantum number for each type
     LOGICAL LSPIRAL               ! do we want to calculate spin spirals
!    REAL(q), POINTER, CONTIGUOUS ::QPROJ(:,:,:,:,:) ! projectors in reciprocal space for each k-point
     REAL(q), POINTER ::QPROJ(:,:,:,:,:)
     COMPLEX(q),POINTER, CONTIGUOUS ::CREXP(:,:)  ! phase factor exp (i (G+k) R(ion))
     REAL(q),  POINTER  ::POSION(:,:) ! positions (required for setup)
     REAL(q),  POINTER  ::VKPT_SHIFT(:,:) => NULL()
                                   ! k-point shift for each ion
     COMPLEX(q),POINTER ::CQFAK(:,:)  ! i^l
  END TYPE nonl_struct

END MODULE nonl_struct_def

MODULE nonl

  USE nonl_struct_def
  USE prec
  
  INTERFACE
     SUBROUTINE VNLAC0(NONL_S,WDES1,CPROJ_LOC,CACC)
       USE nonl_struct_def
       TYPE (nonl_struct) NONL_S
       TYPE (wavedes1)    WDES1
       GDEF    CPROJ_LOC
       COMPLEX(q) CACC
     END SUBROUTINE VNLAC0
  END INTERFACE

  INTERFACE
     SUBROUTINE VNLAC0_DER(NONL_S,WDES1,CPROJ_LOC,CACC, LATT_CUR, ION, IDIR)
       USE nonl_struct_def
       USE lattice
       TYPE (nonl_struct) NONL_S
       TYPE (wavedes1)    WDES1
       COMPLEX(q)   CACC
       GDEF         CPROJ_LOC
       TYPE (latt)  LATT_CUR
       INTEGER      ION
       INTEGER      IDIR
     END SUBROUTINE VNLAC0_DER
  END INTERFACE

!***********************************************************************
! 
! the wavefunction character is strictly defined as
!
!  C_n = e (i k R) \sum_G p_n,G+k e (i G R) C_G
!
! VASP however does not include the term  e (i k R), since
! it does not change the occupancy matrix or energy. Hence 
!
!  W%CPROJ(n) = \sum_G p_n,G+k e (i G R) C_G
!
! the derivative of C_n with respect to R can either include the
! term from e (i k R)
!
!    i k \sum_G  p_n,G+k e (i G R) C_G = i k C_n
!
! or not. This contribution does not contribute to the one center occupancy 
! matrix or the total energy either, since it is "complex" 
! (expectation values involving C_m* D_mn C_n are complex)
! the Berry phase seems to require  shift_der_k however
! nonlr.F uses a similar convention
!***********************************************************************
#define shift_der_k
  
CONTAINS

!****************** subroutine NONL_ALLOC  *****************************
! RCS:  $Id: nonl.F,v 1.2 2002/08/14 13:59:41 kresse Exp $
!
! allocate required arrays
! base on T_INFO and P structure
! i.e.
! number of ions and types is taken from T_INFO
! LMDIM  is taken from P structure
!***********************************************************************

  SUBROUTINE  NONL_ALLOC(NONL_S,T_INFO,P,WDES, LREAL)
    USE prec
    USE pseudo
    USE poscar
    USE ini
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    LOGICAL LREAL

    ! local var
    INTEGER NIONS,NTYPD,LMDIM,NRPLWV,NKPTS,NT

    NIONS =T_INFO%NIONS
    NTYPD =T_INFO%NTYPD
    LMDIM =P(1)%LMDIM
    NRPLWV=WDES%NGDIM
    NKPTS =WDES%NKPTS   ! number of k-points in the IBZ
    ! (required for HF type calculations)
    NONL_S%LRECIP=.NOT. LREAL
    NONL_S%NTYP  =T_INFO%NTYP
    NONL_S%NK    =0
    NONL_S%NIONS =T_INFO%NIONS
    NONL_S%NITYP =>T_INFO%NITYP
    NONL_S%POSION=>T_INFO%POSION
    NONL_S%LSPIRAL=WDES%LSPIRAL
    NULLIFY(NONL_S%VKPT_SHIFT)

    ALLOCATE(NONL_S%LMMAX(NONL_S%NTYP))
    DO NT=1,T_INFO%NTYP
       NONL_S%LMMAX(NT)=P(NT)%LMMAX
    ENDDO

    NULLIFY(NONL_S%CREXP,NONL_S%QPROJ,NONL_S%CQFAK)

    IF ((.NOT.LREAL).AND.(.NOT.NONL_S%LSPIRAL)) THEN
         ALLOCATE(NONL_S%CREXP(NRPLWV,NIONS), &
         NONL_S%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,1), &
         NONL_S%CQFAK(LMDIM,NTYPD))
         CALL REGISTER_ALLOCATE(16._q*SIZE(NONL_S%CREXP)+8._q*SIZE(NONL_S%QPROJ), "nonl-proj")
    ENDIF
    IF ((.NOT.LREAL).AND. NONL_S%LSPIRAL) THEN
         ALLOCATE(NONL_S%CREXP(NRPLWV,NIONS), &
         NONL_S%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,2), &
         NONL_S%CQFAK(LMDIM,NTYPD))
         CALL REGISTER_ALLOCATE(16._q*SIZE(NONL_S%CREXP)+8._q*SIZE(NONL_S%QPROJ), "nonl-proj")
    ENDIF

    RETURN
  END SUBROUTINE NONL_ALLOC


  SUBROUTINE  NONL_DEALLOC(NONL_S)
    USE ini
    IMPLICIT NONE
    TYPE (nonl_struct) NONL_S
    LOGICAL LREAL

    LREAL=.NOT.NONL_S%LRECIP

    IF (.NOT.LREAL) THEN
       CALL DEREGISTER_ALLOCATE(16._q*SIZE(NONL_S%CREXP)+8._q*SIZE(NONL_S%QPROJ), "nonl-proj")
       DEALLOCATE(NONL_S%CREXP,NONL_S%QPROJ,NONL_S%CQFAK)
    ENDIF
       

    RETURN
  END SUBROUTINE NONL_DEALLOC

!****************** subroutine NONL_ALLOC_SPHPRO ***********************
!
! allocate required arrays to describe projectors for
! one ion for one k-point
!
!***********************************************************************

  SUBROUTINE  NONL_ALLOC_SPHPRO(NONL_S,P,WDES)
    USE pseudo
    USE poscar
    IMPLICIT NONE


    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P
    TYPE (wavedes)     WDES
    ! local var
    INTEGER NIONS,NTYPD,LMDIM,NRPLWV,NKPTS,NT

    NIONS =1
    NTYPD =1
    LMDIM =P%LMDIM
    NRPLWV=WDES%NGDIM
    NKPTS =WDES%NKPTS

    NONL_S%LRECIP=.TRUE.
    NONL_S%NTYP  =NTYPD
    NONL_S%NK    =0
    NONL_S%NIONS =NIONS
    NONL_S%LSPIRAL=WDES%LSPIRAL
    ALLOCATE(NONL_S%NITYP(NTYPD)); NONL_S%NITYP(1)=1
    ALLOCATE(NONL_S%LMMAX(NTYPD)); NONL_S%LMMAX(1)=P%LMMAX

    IF (.NOT.NONL_S%LSPIRAL) &
         ALLOCATE(NONL_S%CREXP(NRPLWV,NIONS), &
         NONL_S%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,1), &
         NONL_S%CQFAK(LMDIM,NTYPD))
    IF (NONL_S%LSPIRAL) &
         ALLOCATE(NONL_S%CREXP(NRPLWV,NIONS), &
         NONL_S%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,2), &
         NONL_S%CQFAK(LMDIM,NTYPD))
    RETURN
  END SUBROUTINE NONL_ALLOC_SPHPRO

!****************** subroutine NONL_DEALLOC_SPHPRO *********************
!
! allocate required arrays to describe projectors for
! one ion for one k-point
!
!***********************************************************************

  SUBROUTINE  NONL_DEALLOC_SPHPRO(NONL_S)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    DEALLOCATE(NONL_S%CREXP,NONL_S%QPROJ,NONL_S%CQFAK)
    DEALLOCATE(NONL_S%NITYP)
    DEALLOCATE(NONL_S%LMMAX)

    RETURN
  END SUBROUTINE NONL_DEALLOC_SPHPRO


!****************** subroutine SPHER  ********************************
!
!  subroutine SPHER calculates the sperical harmonics multiplied
!  by the pseudopotential QPROJ on the  compressed reciprocal
!  lattice grid
!  the full non local Pseudopotential is given by
!  <G|V(K)|GP> = SUM(L,LP,site,R)
!                     L                                    LP
!    QPROJ(G,L site) i * DIJ(L,LP site) QPROJ(GP,LP site) i *
!                        EXP(-i(G-GP) R)
!  i^L is stored in CQFAK
!
!  IZERO -1   set NONL_S to the negative projectors
!  IZERO  1   set NONL_S to the positive projectors
!  IZERO  0   add to NONL_S the positive projectors
!
!*********************************************************************

  SUBROUTINE SPHER(GRID, NONL_S, P, WDES, LATT_CUR, IZERO, NK_SELECT, DK)

    USE pseudo
    USE mpimy
    USE mgrid
    USE lattice
    USE constant
    USE asa

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(NONL_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    INTEGER  IZERO
    INTEGER, OPTIONAL :: NK_SELECT
    REAL(q), OPTIONAL :: DK(3)

    ! work arrays
    REAL(q),ALLOCATABLE :: GLEN(:),XS(:),YS(:),ZS(:),VPS(:),FAKTX(:),YLM(:,:),GLENP(:)

    PROFILING_START('spher')

    LYDIM=MAXL(NONL_S%NTYP,P)
    LMYDIM=(LYDIM+1)**2          ! number of lm pairs

    NA=WDES%NGDIM

    ALLOCATE(GLEN(NA),XS(NA),YS(NA),ZS(NA),VPS(NA),FAKTX(NA),YLM(NA,LMYDIM))

    IF (PRESENT(DK)) ALLOCATE(GLENP(NA))

#ifdef gammareal
    IF (.NOT.WDES%LGAMMA) THEN
       WRITE(*,*)'internal error in SPHER: WDES%LGAMMA is incorrect',WDES%LGAMMA
       STOP
    ENDIF
#else
    IF (WDES%LGAMMA) THEN
       WRITE(*,*)'internal error in SPHER: WDES%LGAMMA is incorrect',WDES%LGAMMA
       STOP
    ENDIF
#endif

!-----------------------------------------------------------------------
! spin spiral propagation vector in cartesian coordinates
! is simply zero when LSPIRAL=.FALSE.
!-----------------------------------------------------------------------
    QX= (WDES%QSPIRAL(1)*LATT_CUR%B(1,1)+WDES%QSPIRAL(2)*LATT_CUR%B(1,2)+WDES%QSPIRAL(3)*LATT_CUR%B(1,3))/2
    QY= (WDES%QSPIRAL(1)*LATT_CUR%B(2,1)+WDES%QSPIRAL(2)*LATT_CUR%B(2,2)+WDES%QSPIRAL(3)*LATT_CUR%B(2,3))/2
    QZ= (WDES%QSPIRAL(1)*LATT_CUR%B(3,1)+WDES%QSPIRAL(2)*LATT_CUR%B(3,2)+WDES%QSPIRAL(3)*LATT_CUR%B(3,3))/2

!=======================================================================
! Loop over NSPINORS: here only in case of spin spirals NRSPINOR=2
!=======================================================================
    IF (WDES%LSPIRAL) THEN 
       NSPINORS=2
    ELSE
       NSPINORS=1
    ENDIF
    spinor: DO ISPINOR=1,NSPINORS
!=======================================================================
! main loop over all special points
!=======================================================================
       kpoint: DO NK=1,WDES%NKPTS
          IF (PRESENT(NK_SELECT)) THEN
             IF ( NK_SELECT/=NK) CYCLE
          ENDIF
!         unfortunately this presently breaks the sc-GW code
!          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
!=======================================================================
! now calculate the necessary tables:
! containing the length, phasefactor exp(i m phi) and
! sin(theta),cos(theta)
!=======================================================================
          DO IND=1,WDES%NGVECTOR(NK)

             N1=MOD(WDES%IGX(IND,NK)+GRID%NGX,GRID%NGX)+1
             N2=MOD(WDES%IGY(IND,NK)+GRID%NGY,GRID%NGY)+1 
             N3=MOD(WDES%IGZ(IND,NK)+GRID%NGZ,GRID%NGZ)+1

             G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
             G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
             G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))

             IF (ASSOCIATED(NONL_S%VKPT_SHIFT)) THEN
                G1= G1 +NONL_S%VKPT_SHIFT(1,NI)
                G2= G2 +NONL_S%VKPT_SHIFT(2,NI)
                G3= G3 +NONL_S%VKPT_SHIFT(3,NI)
             ENDIF

             FACTM=1.00
             IF (WDES%LGAMMA .AND. (N1/=1 .OR. N2/=1 .OR. N3/=1)) FACTM=SQRT(2._q)

             GX= (G1*LATT_CUR%B(1,1)+G2*LATT_CUR%B(1,2)+G3*LATT_CUR%B(1,3)-QX) *TPI
             GY= (G1*LATT_CUR%B(2,1)+G2*LATT_CUR%B(2,2)+G3*LATT_CUR%B(2,3)-QY) *TPI
             GZ= (G1*LATT_CUR%B(3,1)+G2*LATT_CUR%B(3,2)+G3*LATT_CUR%B(3,3)-QZ) *TPI


             GLEN(IND)=MAX(SQRT(GX*GX+GY*GY+GZ*GZ),1E-10_q)
             FAKTX(IND)=FACTM
             XS(IND)  =GX/GLEN(IND)
             YS(IND)  =GY/GLEN(IND)
             ZS(IND)  =GZ/GLEN(IND)

             IF (PRESENT(DK)) THEN
                G1P= G1 +DK(1)
                G2P= G2 +DK(2)
                G3P= G3 +DK(3)
                GXP= (G1P*LATT_CUR%B(1,1)+G2P*LATT_CUR%B(1,2)+G3P*LATT_CUR%B(1,3)-QX) *TPI
                GYP= (G1P*LATT_CUR%B(2,1)+G2P*LATT_CUR%B(2,2)+G3P*LATT_CUR%B(2,3)-QY) *TPI
                GZP= (G1P*LATT_CUR%B(3,1)+G2P*LATT_CUR%B(3,2)+G3P*LATT_CUR%B(3,3)-QZ) *TPI
                GLENP(IND)=MAX(SQRT(GXP*GXP+GYP*GYP+GZP*GZP),1E-10_q)
             ENDIF

          ENDDO

          INDMAX=IND-1
!=======================================================================
! now calculate the tables containing the spherical harmonics
! multiply by the pseudopotential and 1/(OMEGA)^(1/2)
!=======================================================================
          CALL SETYLM(LYDIM,INDMAX,YLM,XS,YS,ZS)

          typ: DO NT=1,NONL_S%NTYP

             LMIND=1
             l_loop: DO L=1,P(NT)%LMAX
!=======================================================================
! first interpolate the non-local pseudopotentials
! and multiply by 1/(OMEGA)^(1/2)
!=======================================================================
                FAKT= 1/SQRT(LATT_CUR%OMEGA)
                ARGSC=NPSNL/P(NT)%PSMAXN
                DO IND=1,INDMAX
                   ARG=(GLEN(IND)*ARGSC)+1
                   NADDR=INT(ARG)

                   VPS(IND)=0._q

                   IF (ASSOCIATED(P(NT)%PSPNL_SPLINE)) THEN
                      IF (NADDR<NPSNL) THEN
                         NADDR  =MIN(INT(GLEN(IND)*ARGSC)+1,NPSNL-1)
                         REM=GLEN(IND)-P(NT)%PSPNL_SPLINE(NADDR,1,L)
                         VPS(IND)=(P(NT)%PSPNL_SPLINE(NADDR,2,L)+REM*(P(NT)%PSPNL_SPLINE(NADDR,3,L)+ &
                           &         REM*(P(NT)%PSPNL_SPLINE(NADDR,4,L)+REM*P(NT)%PSPNL_SPLINE(NADDR,5,L))))*FAKT*FAKTX(IND)
                      ENDIF
                   ELSE IF (NADDR<NPSNL-2) THEN
                      REM=MOD(ARG,1.0_q)
                      V1=P(NT)%PSPNL(NADDR-1,L)
                      V2=P(NT)%PSPNL(NADDR,L  )
                      V3=P(NT)%PSPNL(NADDR+1,L)
                      V4=P(NT)%PSPNL(NADDR+2,L)
                      T0=V2
                      T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
                      T2=(V1+V3-(2*V2))/2._q
                      T3=(V4-V1+(3*(V2-V3)))/6._q
                      VPS(IND)=(T0+REM*(T1+REM*(T2+REM*T3)))*FAKT*FAKTX(IND)
                   ENDIF

                   IF (VPS(IND)/=0._q.AND.PRESENT(DK)) THEN
                      ARG=(GLENP(IND)*ARGSC)+1
                      NADDR=INT(ARG)
                      IF (NADDR>NPSNL-1.OR.(.NOT.ASSOCIATED(P(NT)%PSPNL_SPLINE).AND.NADDR>NPSNL-3)) VPS(IND)=0._q
                   ENDIF
                ENDDO
!=======================================================================
! initialize to 0
!=======================================================================
                LL=P(NT)%LPS(L)
                MMAX=2*LL

                IF (IZERO==1 .OR. IZERO==-1) THEN
                   DO LM=0,MMAX
!DIR$ IVDEP
!OCL NOVREC
                      DO IND=1,INDMAX
                         NONL_S%QPROJ(IND,LMIND+LM,NT,NK,ISPINOR)=0
                      ENDDO
                   ENDDO
                ENDIF
                IF (IZERO==-1) THEN
                   DO IND=1,INDMAX
                      VPS(IND)=-VPS(IND)
                   ENDDO
                ENDIF

!=======================================================================
! now multiply with the spherical harmonics
! and set "phase factor" CSET
!=======================================================================
                IF (LL==0) THEN
                   CSET=1.0_q
                ELSE IF (LL==1) THEN
                   CSET=(0.0_q,1.0_q)
                ELSE IF (LL==2) THEN
                   CSET=-1.0_q
                ELSE IF (LL==3) THEN
                   CSET=(0.0_q,-1.0_q)
                ENDIF

                DO LM=0,MMAX
                   NONL_S%CQFAK(LMIND+LM,NT)=CSET
                ENDDO
                LMBASE=LL**2+1

                DO LM=0,MMAX
                   DO IND=1,INDMAX
                      NONL_S%QPROJ(IND,LMIND+LM,NT,NK,ISPINOR)= NONL_S%QPROJ(IND,LMIND+LM,NT,NK,ISPINOR)+ &
                           VPS(IND)*YLM(IND,LM+LMBASE)
                   ENDDO
                ENDDO

                LMIND=LMIND+MMAX+1

             ENDDO l_loop
             IF (LMIND-1/=P(NT)%LMMAX) THEN
                WRITE(*,*)'internal ERROR: SPHER:  LMMAX is wrong',P(NT)%LMMAX,LMIND-1
                STOP
             ENDIF
          ENDDO typ

!=======================================================================
! and of loop over special-points
!=======================================================================
       ENDDO kpoint

       ! conjugate phase alteration for spin down: -q/2 -> q/2
       QX=-QX
       QY=-QY
       QZ=-QZ
    ENDDO spinor

    DEALLOCATE(GLEN,XS,YS,ZS,VPS,FAKTX,YLM)

    IF (PRESENT(DK)) DEALLOCATE(GLENP)

    PROFILING_STOP('spher')

    RETURN
  END SUBROUTINE SPHER


!****************** subroutine SPHER_DER  ****************************
!
!  subroutine SPHER_DER calculates the derivative
!  of the non local projector with respect to k for the cartesian
!  component IDIR times -i = -i nabla_k <p_k|
!  this is equivalent to the projector times r_i  (i=IDIR)
!
!  <p r_i| G+k> =             \int p(r) r_i Y_lm(r) e^ir(G+k) dr
!               = 1/i  d/dk_i \int p(r)     Y_lm(r) e^ir(G+k) dr
!               = -i d/dk_i <p | G+k>
!
!  for simplicity finite differences are used
!
!*********************************************************************

  SUBROUTINE SPHER_DER(GRID,NONL_S,P,WDES,LATT_CUR,  IDIR)
    USE pseudo
    USE mpimy
    USE mgrid
    USE lattice
    USE constant
    USE asa

    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(NONL_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    INTEGER IDIR
    ! local
    REAL(q),POINTER  :: VKPT_ORIG(:,:)
    REAL(q) :: V(3)
    ! displacement for finite differences
    REAL(q), PARAMETER :: DIS=fd_displacement
    INTEGER IDIS,NK

    PROFILING_START('spher_der')

    ! reset all entries
    NONL_S%QPROJ=0

    ! first generate a pointer to the original k-point array
    VKPT_ORIG=>WDES%VKPT
    NULLIFY(WDES%VKPT)
    ALLOCATE(WDES%VKPT(3,WDES%NKPTS))

    ! k-displacement
    V=0
    V(IDIR)=DIS/TPI
    CALL KARDIR(1,V(1),LATT_CUR%A)

    ! negative displacement
    DO NK=1,WDES%NKPTS
       WDES%VKPT(:,NK)=VKPT_ORIG(:,NK)-V
    ENDDO
!   CALL SPHER(GRID,NONL_S,P,WDES,LATT_CUR,-1)
    CALL SPHER(GRID,NONL_S,P,WDES,LATT_CUR,-1,DK= 2*V)

    ! positive displacement
    DO NK=1,WDES%NKPTS
       WDES%VKPT(:,NK)=VKPT_ORIG(:,NK)+V
    ENDDO
!   CALL SPHER(GRID,NONL_S,P,WDES,LATT_CUR, 0)
    CALL SPHER(GRID,NONL_S,P,WDES,LATT_CUR, 0,DK=-2*V)

    NONL_S%QPROJ=NONL_S%QPROJ/DIS/2

    NONL_S%CQFAK=NONL_S%CQFAK*(0.0_q,-1.0_q)


    DEALLOCATE(WDES%VKPT)
    WDES%VKPT=>VKPT_ORIG

    PROFILING_STOP('spher_der')

  END SUBROUTINE SPHER_DER


  SUBROUTINE  NONL_ALLOC_DER(NONL_S, NONL_D)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S, NONL_D
    ! local var
    INTEGER NTYPD, NRPLWV, LMDIM, NKPTS

    NRPLWV=SIZE(NONL_S%QPROJ,1)
    LMDIM =SIZE(NONL_S%QPROJ,2)
    NTYPD =SIZE(NONL_S%QPROJ,3)
    NKPTS =SIZE(NONL_S%QPROJ,4)

    NONL_D=NONL_S

    NULLIFY(NONL_D%QPROJ,NONL_D%CQFAK)
    IF (NONL_S%LSPIRAL) THEN
       ALLOCATE(NONL_D%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,2), &
            NONL_D%CQFAK(LMDIM,NTYPD))
    ELSE
       ALLOCATE(NONL_D%QPROJ(NRPLWV,LMDIM,NTYPD,NKPTS,1), &
            NONL_D%CQFAK(LMDIM,NTYPD))
    ENDIF

    RETURN
  END SUBROUTINE NONL_ALLOC_DER

  SUBROUTINE  NONL_DEALLOC_DER(NONL_D)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_D
    DEALLOCATE(NONL_D%QPROJ,NONL_D%CQFAK)

    RETURN
  END SUBROUTINE NONL_DEALLOC_DER


!************************ SUBROUTINE PHASE *****************************
!
! this subroutine calculates the phasefactor CREXP (exp(ig.r))
! for one k-point, on the compressed grid of  lattice vectors
! CREXP is only calculated if NK changes, if ions positions change
! PHASE must be called with NK=0 to force the routine to recalculate
! the phase-factor in the next call
!***********************************************************************

  SUBROUTINE PHASE(WDES,NONL_S,NK)
    USE constant

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (wavedes)     WDES
    TYPE (nonl_struct) NONL_S

    PROFILING_START('phase')

!=======================================================================
! check if special point changed
!=======================================================================
    IF (NK==0 .OR.NK==NONL_S%NK) THEN
       NONL_S%NK=NK
       PROFILING_STOP('phase')
       RETURN
    ENDIF
    NONL_S%NK=NK

    ! number of G-vectors
    NPL= WDES%NGVECTOR(NK)

    ! set the phase-factor
    ion:  DO NI=1,NONL_S%NIONS
       GXDX=NONL_S%POSION(1,NI)
       GYDY=NONL_S%POSION(2,NI)
       GZDZ=NONL_S%POSION(3,NI)
       DO M=1,NPL
          CGDR=CITPI*(WDES%IGX(M,NK)*GXDX+WDES%IGY(M,NK)*GYDY+WDES%IGZ(M,NK)*GZDZ)
          NONL_S%CREXP(M,NI)=EXP(CGDR)
       ENDDO
    ENDDO ion

    PROFILING_STOP('phase')

    RETURN
  END SUBROUTINE PHASE


!************************ SUBROUTINE VNLACC  ***************************
!
!  subroutine for calculating the non local contribution of
!  the Hamiltonian, using reciprocal space projection scheme
!  the result of the wavefunction projected on the projection operatores
!  must be given in CPROJ
!
!***********************************************************************

  SUBROUTINE VNLACC(NONL_S, W1, CDIJ, CQIJ, ISP, EVALUE,  CACC)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    REAL(q) EVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CACC(:)
  ! local
    GDEF       CRESUL(W1%WDES1%NPRO)

    CACC=0
    CALL OVERL1(W1%WDES1, SIZE(CDIJ,1),CDIJ(1,1,1,ISP),CQIJ(1,1,1,ISP), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0(NONL_S,W1%WDES1,CRESUL(1),CACC(1))

    RETURN
  END SUBROUTINE VNLACC

  SUBROUTINE VNLACC_ADD(NONL_S, W1, CDIJ, CQIJ, ISP, EVALUE,  CACC)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    REAL(q) EVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CACC(:)
    ! work array
    GDEF       CRESUL(W1%WDES1%NPRO)

    CALL OVERL1(W1%WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0(NONL_S,W1%WDES1,CRESUL(1),CACC(1))

    RETURN
  END SUBROUTINE VNLACC_ADD

  SUBROUTINE VNLACC_C(NONL_S, W1, CDIJ, CQIJ, ISP, CEVALUE,  CACC)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    INTEGER    LMDIM
    COMPLEX(q) CEVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CACC(:)
    ! work array
    GDEF       CRESUL(W1%WDES1%NPRO)

    CACC=0
    CALL OVERL1_C(W1%WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), CEVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0(NONL_S,W1%WDES1,CRESUL(1),CACC(1))

    RETURN
  END SUBROUTINE VNLACC_C


  SUBROUTINE VNLACC_ADD_C(NONL_S,W1, CDIJ, CQIJ, ISP, CEVALUE,  CACC)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    INTEGER    LMDIM
    COMPLEX(q) CEVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CACC(:)
    ! work array
    GDEF       CRESUL(W1%WDES1%NPRO)

    CALL OVERL1_C(W1%WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), CEVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0(NONL_S,W1%WDES1,CRESUL(1),CACC(1))

    RETURN
  END SUBROUTINE VNLACC_ADD_C

! here CDIJ and QCIJ are always complex
  SUBROUTINE VNLACC_ADD_CCDIJ(NONL_S, W1, CDIJ, CQIJ, ISP, EVALUE,  CACC)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    REAL(q) EVALUE
    COMPLEX(q) CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    INTEGER :: ISP
    COMPLEX(q) CACC(:)
    ! work array
    GDEF       CRESUL(W1%WDES1%NPRO)

    CALL OVERL1_CCDIJ(W1%WDES1, SIZE(CDIJ,1), CDIJ(1,1,1,ISP), CQIJ(1,1,1,ISP), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0(NONL_S,W1%WDES1,CRESUL(1),CACC(1))

    RETURN
  END SUBROUTINE VNLACC_ADD_CCDIJ


!************************ SUBROUTINE PROJ    ***************************
!
! this subroutine calculates the projection of all bands of one
! specific k-point onto the reciprocal space projection operator
!
!***********************************************************************
  
  SUBROUTINE PROJ(NONL_S,WDES,W,NK)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    INTEGER NK
  ! local
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1
    INTEGER ISP, N

    PROFILING_START('proj')

    CALL SETWDES(WDES,WDES1,NK)

    DO ISP=1,WDES%ISPIN
       DO N=1,WDES%NBANDS
          CALL SETWAV(W,W1,WDES1,N,ISP)
          CALL PROJ1(NONL_S,WDES1,W1)
       ENDDO
    ENDDO

    PROFILING_STOP('proj')

  END SUBROUTINE PROJ

!************************ SUBROUTINE PROJ1   ***************************
!
! this subroutine calculates the scalar product of one wavefunction with
! all projector functions in reciprocal space
! thesis gK Equ. (10.34)
!
!
!***********************************************************************

  SUBROUTINE PROJ1(NONL_S,WDES1,W1)
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1), TARGET :: W1

  ! local
    INTEGER :: NPL, LMBASE, ISPIRAL, ISPINOR, NIS, NT, LMMAXC, NI, &
         K, KK, LM
    REAL(q) :: SUMR, SUMI
    COMPLEX(q) :: CTMP
    REAL(q) :: WORK(WDES1%NGVECTOR*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)

    PROFILING_START('proj1')


    IF (WDES1%NK /= NONL_S%NK) THEN
       WRITE(*,*) 'internal error in PROJ1: PHASE not properly set up',WDES1%NK, NONL_S%NK
       STOP
    ENDIF

    NPL=WDES1%NGVECTOR
!=======================================================================
! performe loop over ions
!=======================================================================
    LMBASE= 0
    ISPIRAL = 1

    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
       NIS=1

       typ: DO NT=1,NONL_S%NTYP
          LMMAXC=NONL_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
!=======================================================================
! multiply with phasefactor and divide into real and imaginary part
!=======================================================================
             CALL CREXP_MUL_WAVE( NPL,  NONL_S%CREXP(1,NI), W1%CPTWFP(NPL*ISPINOR+1), WORK(1))
!=======================================================================
! loop over composite indexes L,M
!=======================================================================
#if defined(do_loops) || defined(pro_loop)
             l_loop: DO LM=1,LMMAXC
                SUMR=0
                SUMI=0
!DIR$ IVDEP
!OCL NOVREC
                DO K=1,NPL
                   SUMR = SUMR + WORK(K)     * NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)
                   SUMI = SUMI + WORK(K+NPL) * NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)
                ENDDO
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *NONL_S%CQFAK(LM,NT))
             ENDDO l_loop
#else
             CALL DGEMV( 'T' , NPL, LMMAXC, 1._q , NONL_S%QPROJ(1,1,NT,WDES1%NK,ISPIRAL), &
                  WDES1%NGDIM, WORK(1) , 1 , 0._q ,  TMP(1,1), 1)
             CALL DGEMV( 'T' , NPL, LMMAXC, 1._q , NONL_S%QPROJ(1,1,NT,WDES1%NK,ISPIRAL), &
                  WDES1%NGDIM, WORK(1+NPL) , 1 , 0._q ,  TMP(1,2), 1)

             l_loop: DO LM=1,LMMAXC
                SUMR=TMP(LM,1)
                SUMI=TMP(LM,2)
                CPROJ(LM+LMBASE)=GREAL(CMPLX( SUMR , SUMI ,KIND=q) *NONL_S%CQFAK(LM,NT))
             ENDDO l_loop
#endif
             LMBASE=LMBASE+LMMAXC
          ENDDO ion

600       NIS = NIS+NONL_S%NITYP(NT)
       ENDDO typ
       IF (NONL_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

    ! distribute the projected wavefunctions to nodes
    CALL DIS_PROJ(WDES1,CPROJ(1),W1%CPROJ(1))

    PROFILING_STOP('proj1')

    RETURN
  END SUBROUTINE PROJ1


!************************ SUBROUTINE PROJXYZ  **************************
!
! this subroutine calculates the first order change of the 
! wave function character upon moving the ions for one selected k-point
! and spin component
! the results are stored in CPROJXYZ
! a factor 2 stemming from variations of the bra or kat is included
! another important issue is that LADDK can be included, which
! forces the treatment usually done only for #define shift_der_k
! this allows for correct results for the Born effective charges
! even if shift_der_k is not defined
!
!
!***********************************************************************

  SUBROUTINE PROJXYZ(NONL_S, WDES, W, LATT_CUR, ISP, NK, CPROJXYZ, LADDK)
    USE lattice
    USE constant
    USE openmp, ONLY : omp_nthreads_hi

    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    INTEGER            ISP, NK
    GDEF :: CPROJXYZ(WDES%NPROD, WDES%NBANDS, 3)
    LOGICAL, OPTIONAL :: LADDK
    ! local variable
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    GDEF       :: CX(WDES%NPRO_TOT) ,CY(WDES%NPRO_TOT) ,CZ(WDES%NPRO_TOT)
    GDEF       :: CX_, CY_, CZ_
    INTEGER    :: NPRO, NPL, N, LMBASE, LMMAXC, ISPINOR, ISPIRAL, K, KK, NI, NIS, NT, LM
    COMPLEX(q) :: CMUL
    GDEF       :: CVAL

    PROFILING_START('projxyz')

    NPRO=WDES%NPRO_TOT
    NPL= WDES%NGVECTOR(NK)


    CALL SETWDES(WDES,WDES1,NK)
    CALL PHASE(WDES,NONL_S,NK)

    band: DO N=1,WDES%NBANDS

       CX=0
       CY=0
       CZ=0

       LMBASE= 0

       ISPIRAL=1
       spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
          NIS=1
          typ:  DO NT=1,NONL_S%NTYP
             LMMAXC=NONL_S%LMMAX(NT)
             IF (LMMAXC==0) GOTO 100

             ions:   DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
                l_loop: DO LM=1,LMMAXC
                   CMUL=NONL_S%CQFAK(LM,NT)*CITPI
                   CX_=0
                   CY_=0
                   CZ_=0
                   IF (PRESENT (LADDK)) THEN
                   DO K=1,NPL                 
                      KK=K+NPL*ISPINOR
                      CVAL =  2*NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*W%CPTWFP(KK,N,NK,ISP)*NONL_S%CREXP(K,NI)*CMUL
                      CX_=CX_+(WDES%IGX(K,NK)+WDES%VKPT(1,NK))*CVAL
                      CY_=CY_+(WDES%IGY(K,NK)+WDES%VKPT(2,NK))*CVAL
                      CZ_=CZ_+(WDES%IGZ(K,NK)+WDES%VKPT(3,NK))*CVAL
                   ENDDO
                   ELSE
                   DO K=1,NPL                 
                      KK=K+NPL*ISPINOR
                      ! factor 2 because either the bra or kat can vary
                      CVAL =  2*NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*W%CPTWFP(KK,N,NK,ISP)*NONL_S%CREXP(K,NI)*CMUL
#ifdef shift_der_k
                      CX_=CX_+(WDES%IGX(K,NK)+WDES%VKPT(1,NK))*CVAL
                      CY_=CY_+(WDES%IGY(K,NK)+WDES%VKPT(2,NK))*CVAL
                      CZ_=CZ_+(WDES%IGZ(K,NK)+WDES%VKPT(3,NK))*CVAL
#else
                      CX_=CX_+(WDES%IGX(K,NK))*CVAL
                      CY_=CY_+(WDES%IGY(K,NK))*CVAL
                      CZ_=CZ_+(WDES%IGZ(K,NK))*CVAL
#endif
                   ENDDO
                   ENDIF

                   CX(LM+LMBASE)=CX_*LATT_CUR%B(1,1)+CY_*LATT_CUR%B(1,2)+CZ_*LATT_CUR%B(1,3)
                   CY(LM+LMBASE)=CX_*LATT_CUR%B(2,1)+CY_*LATT_CUR%B(2,2)+CZ_*LATT_CUR%B(2,3)
                   CZ(LM+LMBASE)=CX_*LATT_CUR%B(3,1)+CY_*LATT_CUR%B(3,2)+CZ_*LATT_CUR%B(3,3)

                ENDDO l_loop
                LMBASE= LMMAXC+LMBASE
             ENDDO ions

100          NIS = NIS+NONL_S%NITYP(NT)
          ENDDO typ

          IF (NONL_S%LSPIRAL) ISPIRAL=2
       ENDDO spinor

       CALL DIS_PROJ(WDES1,CX,CPROJXYZ(1,N,1))
       CALL DIS_PROJ(WDES1,CY,CPROJXYZ(1,N,2))
       CALL DIS_PROJ(WDES1,CZ,CPROJXYZ(1,N,3))

    ENDDO band

    PROFILING_STOP('projxyz')

    RETURN
  END SUBROUTINE PROJXYZ


!************************ SUBROUTINE PROJXYZ_WA  ***********************
!
! This subroutine does the same as PROJXYZ, however, is accepts
! a single array for the orbitals' PW coefficients instead 
! of a wavefunction array W
! this version is used by the Green's function module 
!
!
!***********************************************************************

  SUBROUTINE PROJXYZ_WA(NONL_S, WDES, CW, LATT_CUR, NK, CPROJXYZ, LADDK)
    USE lattice
    USE constant
    USE openmp, ONLY : omp_nthreads_hi

    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes)     WDES
    COMPLEX(q),CONTIGUOUS :: CW(:,:)
    TYPE (latt)        LATT_CUR
    INTEGER            ISP, NK
    GDEF, CONTIGUOUS :: CPROJXYZ(:,:,:)
    LOGICAL, OPTIONAL :: LADDK
    ! local variable
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    GDEF       :: CX(WDES%NPRO_TOT) ,CY(WDES%NPRO_TOT) ,CZ(WDES%NPRO_TOT)
    GDEF       :: CX_, CY_, CZ_
    INTEGER    :: NPRO, NPL, N, LMBASE, LMMAXC, ISPINOR, ISPIRAL, K, KK, NI, NIS, NT, LM
    COMPLEX(q) :: CMUL
    GDEF       :: CVAL

    PROFILING_START('projxyz')

    NPRO=WDES%NPRO_TOT
    NPL= WDES%NGVECTOR(NK)

    CALL SETWDES(WDES,WDES1,NK)
    CALL PHASE(WDES,NONL_S,NK)

    band: DO N=1,WDES%NBANDS

       CX=0
       CY=0
       CZ=0

       LMBASE= 0

       ISPIRAL=1
       spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
          NIS=1
          typ:  DO NT=1,NONL_S%NTYP
             LMMAXC=NONL_S%LMMAX(NT)
             IF (LMMAXC==0) GOTO 100

             ions:   DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
                l_loop: DO LM=1,LMMAXC
                   CMUL=NONL_S%CQFAK(LM,NT)*CITPI
                   CX_=0
                   CY_=0
                   CZ_=0
                   IF (PRESENT (LADDK)) THEN
                   DO K=1,NPL                 
                      KK=K+NPL*ISPINOR
                      CVAL =  2*NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*CW(KK,N)*NONL_S%CREXP(K,NI)*CMUL
                      CX_=CX_+CVAL*(WDES%IGX(K,NK)+WDES%VKPT(1,NK))
                      CY_=CY_+CVAL*(WDES%IGY(K,NK)+WDES%VKPT(2,NK))
                      CZ_=CZ_+CVAL*(WDES%IGZ(K,NK)+WDES%VKPT(3,NK))
                   ENDDO
                   ELSE
                   DO K=1,NPL                 
                      KK=K+NPL*ISPINOR
                      ! factor 2 because either the bra or kat can vary
                      CVAL =  2*NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*CW(KK,N)*NONL_S%CREXP(K,NI)*CMUL
#ifdef shift_der_k
                      CX_=CX_+CVAL*(WDES%IGX(K,NK)+WDES%VKPT(1,NK))
                      CY_=CY_+CVAL*(WDES%IGY(K,NK)+WDES%VKPT(2,NK))
                      CZ_=CZ_+CVAL*(WDES%IGZ(K,NK)+WDES%VKPT(3,NK))
#else
                      CX_=CX_+CVAL*(WDES%IGX(K,NK))
                      CY_=CY_+CVAL*(WDES%IGY(K,NK))
                      CZ_=CZ_+CVAL*(WDES%IGZ(K,NK))
#endif
                   ENDDO
                   ENDIF

                   CX(LM+LMBASE)=CX_*LATT_CUR%B(1,1)+CY_*LATT_CUR%B(1,2)+CZ_*LATT_CUR%B(1,3)
                   CY(LM+LMBASE)=CX_*LATT_CUR%B(2,1)+CY_*LATT_CUR%B(2,2)+CZ_*LATT_CUR%B(2,3)
                   CZ(LM+LMBASE)=CX_*LATT_CUR%B(3,1)+CY_*LATT_CUR%B(3,2)+CZ_*LATT_CUR%B(3,3)

                ENDDO l_loop
                LMBASE= LMMAXC+LMBASE
             ENDDO ions

100          NIS = NIS+NONL_S%NITYP(NT)
          ENDDO typ

          IF (NONL_S%LSPIRAL) ISPIRAL=2
       ENDDO spinor

       CALL DIS_PROJ(WDES1,CX,CPROJXYZ(1,N,1))
       CALL DIS_PROJ(WDES1,CY,CPROJXYZ(1,N,2))
       CALL DIS_PROJ(WDES1,CZ,CPROJXYZ(1,N,3))

    ENDDO band

    PROFILING_STOP('projxyz')

    RETURN
  END SUBROUTINE PROJXYZ_WA


!************************ SUBROUTINE PROJLATT_DER***********************
!
! this subroutine calculates the first order change of the 
! wave function character upon changing the lattice
! the results are stored in CPROJXYZ
!
!***********************************************************************

  SUBROUTINE PROJLAT_DER(P, NONL_S, WDES, W, LATT_CUR, ISP, NK, CPROJXYZ)
    USE lattice
    USE constant
    USE pseudo 

    IMPLICIT NONE

    TYPE (potcar)      P(:)
    TYPE (nonl_struct) NONL_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    INTEGER            ISP, NK
    GDEF, TARGET :: CPROJXYZ(WDES%NPROD, WDES%NBANDS, 6)
    ! local variable
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point
    TYPE (wavefun1)    W1
    INTEGER    :: N
    INTEGER :: IDIR, JDIR, IJDIR
    TYPE (latt)        LATT_FIN1,LATT_FIN2
    REAL(q) :: DIS=fd_displacement

    PROFILING_START('projlat_der')

    CALL SETWDES(WDES,WDES1,NK)
    CALL PHASE(WDES,NONL_S,NK)

    IJDIR=0
    DO IDIR=1,3
       DO JDIR=1,IDIR
          IJDIR=IJDIR+1
          LATT_FIN1=LATT_CUR
          LATT_FIN1%A(IDIR,:)=LATT_CUR%A(IDIR,:)+DIS*LATT_CUR%A(JDIR,:)

          LATT_FIN2=LATT_CUR
          LATT_FIN2%A(IDIR,:)=LATT_CUR%A(IDIR,:)-DIS*LATT_CUR%A(JDIR,:)

          CALL LATTIC(LATT_FIN1)
          CALL LATTIC(LATT_FIN2)

          CALL SPHER(WDES%GRID,NONL_S,P,WDES,LATT_FIN2,  IZERO=-1, NK_SELECT=NK)
          CALL SPHER(WDES%GRID,NONL_S,P,WDES,LATT_FIN1,  IZERO=0, NK_SELECT=NK)

          band: DO N=1,WDES%NBANDS
             CALL SETWAV(W,W1,WDES1,N,ISP)
             W1%CPROJ => CPROJXYZ(:,N,IJDIR)
             CALL PROJ1(NONL_S,WDES1,W1) ! calculate W1%CPROJ (linked to CWORK)
             CPROJXYZ(:,N,IJDIR)=CPROJXYZ(:,N,IJDIR)/DIS

          ENDDO band
       ENDDO
    ENDDO

    CALL SPHER(WDES%GRID, NONL_S, P, WDES, LATT_CUR,  1, NK_SELECT=NK)

    PROFILING_STOP('projlat_der')

    RETURN
  END SUBROUTINE PROJLAT_DER


!************************ SUBROUTINE PROJ_DER  *************************
!
! this subroutine calculates the first order change of the 
! wave function character upon moving one ion ION in the direction IDIR
! a term  (i k)  is included as well to "center" the derivate at the 
! ion
!
!***********************************************************************

  SUBROUTINE PROJ_DER(NONL_S, WDES1, W1, LATT_CUR, ION, IDIR)
    USE lattice
    USE constant

    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes1)    WDES1
    TYPE (wavefun1)    W1
    TYPE (latt)        LATT_CUR
    INTEGER            ION, IDIR
    ! local variable
    GDEF       :: C(WDES1%NPRO_TOT)
    GDEF       :: CVAL
    INTEGER    :: NPL, N, LMBASE, LMMAXC, ISPINOR, ISPIRAL, K, KK, NI, NIS, NT, LM
    COMPLEX(q) :: CMUL
    COMPLEX(q) :: CTMP(WDES1%NRPLWV)

    PROFILING_START('proj_der')

    NPL= WDES1%NGVECTOR

    C=0

    LMBASE= 0
    ISPIRAL=1

    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
       NIS=1
       typ:  DO NT=1,NONL_S%NTYP
          LMMAXC=NONL_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 100

          ions:   DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
             IF (NI==ION) THEN

                IF (IDIR==0) THEN
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL
                      KK=K+NPL*ISPINOR
                      CTMP(K)=W1%CPTWFP(KK)*NONL_S%CREXP(K,NI)
                   ENDDO
                ELSE
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL
                      KK=K+NPL*ISPINOR
                      CTMP(K)=W1%CPTWFP(KK)*NONL_S%CREXP(K,NI)* &
#ifdef shift_der_k
                          ((WDES1%IGX(K)+WDES1%VKPT(1))*LATT_CUR%B(IDIR,1)+ &
                           (WDES1%IGY(K)+WDES1%VKPT(2))*LATT_CUR%B(IDIR,2)+ &
                           (WDES1%IGZ(K)+WDES1%VKPT(3))*LATT_CUR%B(IDIR,3))
#else
                          ((WDES1%IGX(K))*LATT_CUR%B(IDIR,1)+ &
                           (WDES1%IGY(K))*LATT_CUR%B(IDIR,2)+ &
                           (WDES1%IGZ(K))*LATT_CUR%B(IDIR,3))
#endif
                   ENDDO
                ENDIF

                l_loop: DO LM=1,LMMAXC
                   IF (IDIR==0) THEN
                      CMUL=NONL_S%CQFAK(LM,NT)
                   ELSE
                      CMUL=NONL_S%CQFAK(LM,NT)*CITPI
                   ENDIF

                   CVAL=0
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL                 
                      CVAL = CVAL+NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)*GREAL(CTMP(K)*CMUL)
                   ENDDO

                   C(LM+LMBASE)=CVAL
                ENDDO l_loop
             ENDIF
             LMBASE= LMMAXC+LMBASE
          ENDDO ions

100       NIS = NIS+NONL_S%NITYP(NT)
       ENDDO typ

       IF (NONL_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor


    CALL DIS_PROJ(WDES1,C,W1%CPROJ(1))

    PROFILING_STOP('proj_der')

    RETURN
  END SUBROUTINE PROJ_DER


!************************ SUBROUTINE VNLACC_DER  ***********************
!
!  subroutine for calculating the non local contribution of
!  the first derivative of the Hamiltonian
!  upon moving one ion ION in the direction IDIR
!  more specifically the  contribution 
!
!     | d p_i/ d R > (D_ij - epsilon Q_ij) c_j
!
!  for IDIR=0 the derivative is replaced by | p_i >
!
!***********************************************************************

  SUBROUTINE VNLACC_DER(NONL_S, W1, &
       &     CDIJ,CQIJ,EVALUE,  CACC, LATT_CUR, ION, IDIR)
    USE lattice

    IMPLICIT NONE
    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    INTEGER LMDIM
    REAL(q) EVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    COMPLEX(q)   CACC(:)
    TYPE (latt)  LATT_CUR
    INTEGER      ION, IDIR
    ! work arrays
    GDEF         CRESUL(W1%WDES1%NPRO)

    CALL OVERL1(W1%WDES1,SIZE(CDIJ,1),CDIJ(1,1,1,1),CQIJ(1,1,1,1), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0_DER(NONL_S,W1%WDES1,CRESUL(1),CACC(1), LATT_CUR, ION, IDIR )

    RETURN
  END SUBROUTINE VNLACC_DER

  SUBROUTINE VNLACC_DER_C(NONL_S, W1, &
       &     CDIJ, CQIJ,EVALUE,  CACC, LATT_CUR, ION, IDIR)
    USE lattice

    IMPLICIT NONE
    TYPE (nonl_struct) NONL_S
    TYPE (wavefun1)    W1
    INTEGER LMDIM
    COMPLEX(q)   EVALUE
    OVERLAP CDIJ(:,:,:,:),CQIJ(:,:,:,:)
    COMPLEX(q)   CACC(W1%WDES1%NPL)
    TYPE (latt)  LATT_CUR
    INTEGER      ION, IDIR
    ! work arrays
    GDEF         CRESUL(W1%WDES1%NPRO)

    CALL OVERL1_C(W1%WDES1,SIZE(CDIJ,1),CDIJ(1,1,1,1),CQIJ(1,1,1,1), EVALUE, W1%CPROJ(1),CRESUL(1))
    CALL VNLAC0_DER(NONL_S,W1%WDES1,CRESUL(1),CACC(1), LATT_CUR, ION, IDIR )

    RETURN
  END SUBROUTINE VNLACC_DER_C



!************************ SUBROUTINE FORNL  ****************************
!
! this subroutine calculates the forces related to the non local
! pseudopotential
! the projection of the wavefunction onto the projector functions
! must be stored in CPROJ on entry
! Algorithm:
! ACC(G') = SUM(L,L',R)
!         SUM(G)   QPROJ(G ,L)  EXP( iG  R) C(G)  iG *
!         D(L,L') + Evalue  Q(L,L')
!         SUM(GP)  QPROJ(GP,LP) EXP(-iGP R) C(GP)
!
!***********************************************************************

  SUBROUTINE FORNL(NONL_S,WDES,W,LATT_CUR, LMDIM,CDIJ,CQIJ,EINL)
    USE lattice
    USE constant

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    TYPE (wavedes1)    WDES1         ! descriptor for one k-point

    DIMENSION EINL(3,NONL_S%NIONS)
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    ! work arrays
    REAL(q) ::  ENL(NONL_S%NIONS)
    GDEF :: CM
    GDEF,ALLOCATABLE :: CX(:) ,CY(:) ,CZ(:), C(:)
    GDEF,ALLOCATABLE :: CXL(:),CYL(:),CZL(:),CL(:)

    PROFILING_START('fornl')

    N =WDES%NPRO_TOT
    NL=WDES%NPRO

    ALLOCATE(CX(N),CY(N),CZ(N),C(N),CXL(NL),CYL(NL),CZL(NL),CL(NL))

    EINL=0._q
    ENL =0._q
!=======================================================================
! loop over special points, and bands
!=======================================================================

    kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
       IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
       NPL= WDES%NGVECTOR(NK)
       CALL SETWDES(WDES,WDES1,NK)

       CALL PHASE(WDES,NONL_S,NK)

       spin:   DO ISP=1,WDES%ISPIN
          band: DO N=1,WDES%NBANDS

             EVALUE=W%CELEN(N,NK,ISP)
             WEIGHT=W%FERWE(N,NK,ISP)*WDES%WTKPT(NK)*WDES%RSPIN
!=======================================================================
! first build up CX, CY, CZ for tables
!=======================================================================
             CX=0
             CY=0
             CZ=0
             C=0

             LMBASE= 0

             ISPIRAL = 1
             spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

                NIS=1
                typ:  DO NT=1,NONL_S%NTYP
                   LMMAXC=NONL_S%LMMAX(NT)
                   IF (LMMAXC==0) GOTO 100

                   ions: DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
                      l_loop: DO LM=1,LMMAXC
                         CMUL=NONL_S%CQFAK(LM,NT)*CITPI

                         !DIR$ IVDEP
                         !OCL NOVREC
                         DO K=1,NPL                 
                            KK=K+NPL*ISPINOR
                            CVAL =  (NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*NONL_S%CREXP(K,NI)*W%CPTWFP(KK,N,NK,ISP))*CMUL
                            ! new version calculate the projected wave function character on the fly
                            C(LM+LMBASE) =C(LM+LMBASE) +(NONL_S%QPROJ(K,LM,NT,NK,ISPIRAL)*NONL_S%CREXP(K,NI)*W%CPTWFP(KK,N,NK,ISP))*NONL_S%CQFAK(LM,NT)
#ifdef shift_der_k
                            ! new version add i k which contributes nothing to the energy derivative
                            CX(LM+LMBASE)=CX(LM+LMBASE)+(WDES%IGX(K,NK)+WDES%VKPT(1,NK))*CVAL
                            CY(LM+LMBASE)=CY(LM+LMBASE)+(WDES%IGY(K,NK)+WDES%VKPT(2,NK))*CVAL
                            CZ(LM+LMBASE)=CZ(LM+LMBASE)+(WDES%IGZ(K,NK)+WDES%VKPT(3,NK))*CVAL
#else
                            CX(LM+LMBASE)=CX(LM+LMBASE)+(WDES%IGX(K,NK))*CVAL
                            CY(LM+LMBASE)=CY(LM+LMBASE)+(WDES%IGY(K,NK))*CVAL
                            CZ(LM+LMBASE)=CZ(LM+LMBASE)+(WDES%IGZ(K,NK))*CVAL
#endif
                         ENDDO

                      ENDDO l_loop
                      LMBASE= LMMAXC+LMBASE
                   ENDDO ions

100                NIS = NIS+NONL_S%NITYP(NT)
                ENDDO typ
                IF (NONL_S%LSPIRAL) ISPIRAL=2
             ENDDO spinor

             CALL DIS_PROJ(WDES1,C,CL)
             CALL DIS_PROJ(WDES1,CX,CXL)
             CALL DIS_PROJ(WDES1,CY,CYL)
             CALL DIS_PROJ(WDES1,CZ,CZL)
!=======================================================================
! sum up local contributions
! calculate SUM_LP  D(LP,L)-E Q(LP,L) * C(LP)
!=======================================================================
             spinor2 : DO ISPINOR=0,WDES%NRSPINORS-1
                DO ISPINOR_=0,WDES%NRSPINORS-1

                   LMBASE =ISPINOR *WDES%NPRO/2
                   LMBASE_=ISPINOR_*WDES%NPRO/2

                   NIS   =1
                   typ2:  DO NT=1,WDES%NTYP
                      LMMAXC=WDES%LMMAX(NT)
                      IF (LMMAXC==0) GOTO 600

                      ions2: DO NI=NIS,WDES%NITYP(NT)+NIS-1
                         NIP=NI_GLOBAL(NI, WDES%COMM_INB) !  local storage index
                         l_loop2: DO LM=1,LMMAXC
                            CM=0
                            DO LMP=1,LMMAXC
                               CM=     (CDIJ(LMP,LM,NI,ISP+ISPINOR_+2*ISPINOR)-EVALUE*CQIJ(LMP,LM,NI,ISP+ISPINOR_+2*ISPINOR))* &
                                    ! new version use the calculated projected wave function character
                                    GCONJG(CL(LM+LMBASE))
!                                    GCONJG(W%CPROJ(LM+LMBASE,N,NK,ISP))
                               ENL(NIP)=ENL(NIP) + (W%CPROJ(LMP+LMBASE_,N,NK,ISP)*CM)*WEIGHT
                               EINL(1,NIP)=EINL(1,NIP)-(2*WEIGHT)*(CXL(LMP+LMBASE_)*CM)
                               EINL(2,NIP)=EINL(2,NIP)-(2*WEIGHT)*(CYL(LMP+LMBASE_)*CM)
                               EINL(3,NIP)=EINL(3,NIP)-(2*WEIGHT)*(CZL(LMP+LMBASE_)*CM)
                            ENDDO
                         ENDDO l_loop2
                         LMBASE = LMMAXC+LMBASE
                         LMBASE_= LMMAXC+LMBASE_
                      ENDDO ions2
600                   NIS = NIS+WDES%NITYP(NT)
                   ENDDO typ2
                ENDDO
             ENDDO spinor2
!=======================================================================
          ENDDO band
       ENDDO spin
    ENDDO kpoint
!=======================================================================
! forces are now in the reciprocal lattice transform it to
! cartesian coordinates
!=======================================================================
    CALLMPI( M_sum_d(WDES%COMM, EINL(1,1),NONL_S%NIONS*3))
    CALLMPI( M_sum_d(WDES%COMM, ENL (1)  ,NONL_S%NIONS))
    CALL  DIRKAR(NONL_S%NIONS,EINL,LATT_CUR%B)
    DEALLOCATE(CX,CY,CZ,C,CXL,CYL,CZL,CL)
 
    PROFILING_STOP('fornl')

    RETURN
  END SUBROUTINE FORNL
  
!************************ SUBROUTINE STRENL  ****************************
!
! calculate non-local contributions to stress,
! easiest to implement and definitly quite fast, is an approach based
! on finite differences, the implementation took only 3 hours (and this is
! -I think- the most compelling feature of the routine)
!
!***********************************************************************

  SUBROUTINE STRENL(GRID,NONL_S,P,W,WDES,LATT_CUR, &
       LMDIM,CDIJ,CQIJ, ISIF,FNLSIF)
    USE pseudo
    USE mpimy
    USE mgrid
    USE lattice
    USE constant

    IMPLICIT COMPLEX(q) (C)
    IMPLICIT REAL(q) (A-B,D-H,O-Z)

    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(NONL_S%NTYP)
    TYPE (wavedes)     WDES
    TYPE (wavedes1)    WDES1
    TYPE (wavespin)    W
    TYPE (wavefun1)    W1
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR,LATT_FIN

    DIMENSION FNLSIF(3,3)
    OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    OVERLAP CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    ! work arrays
    GDEF,ALLOCATABLE,TARGET ::  CWORK(:)
    REAL(q) FNLSIFTMP

    PROFILING_START('strenl')

    DIS=fd_displacement
!TEST which step should be used in the finite differences
!      DIS=1E-3
! 1000 DIS=DIS/2
!TEST
    ALLOCATE(CWORK(WDES%NPROD))

    FNLSIF=0

    DO IDIR=1,3
       DO JDIR=1,3
          FNLSIFTMP=0
!=======================================================================
! use central differences to calculate the stress
! set up QPROJ so that central differences can be  calculated
!=======================================================================
          LATT_FIN=LATT_CUR
          IF (ISIF==1) THEN
             !  only isotrop pressure
             DO I=1,3; DO J=1,3
                LATT_FIN%A(I,J)=LATT_CUR%A(I,J)*(1+DIS/3)
             ENDDO; ENDDO
          ELSE
             !  all directions
             DO I=1,3
                LATT_FIN%A(IDIR,I)=LATT_CUR%A(IDIR,I)+DIS*LATT_CUR%A(JDIR,I)
             ENDDO
          ENDIF
          CALL LATTIC(LATT_FIN)

          CALL SPHER(GRID,NONL_S,P,WDES,LATT_FIN,  IZERO=-1)

          LATT_FIN=LATT_CUR
          IF (ISIF==1) THEN
             !  only isotrop pressure
             DO I=1,3; DO J=1,3
                LATT_FIN%A(I,J)=LATT_CUR%A(I,J)*(1-DIS/3)
             ENDDO; ENDDO
          ELSE 
             !  all directions
             DO I=1,3
                LATT_FIN%A(IDIR,I)=LATT_CUR%A(IDIR,I)-DIS*LATT_CUR%A(JDIR,I)
             ENDDO
          ENDIF
          CALL LATTIC(LATT_FIN)

          CALL SPHER(GRID,NONL_S,P,WDES,LATT_FIN,  IZERO=0)

!=======================================================================
! loop over all k-points spin and bands
!=======================================================================
          kpoint: DO NK=1,WDES%NKPTS
#ifdef MPI
             IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
             CALL PHASE(WDES,NONL_S,NK)
             CALL SETWDES(WDES,WDES1,NK)

             spin: DO ISP=1,WDES%ISPIN
                band: DO N=1,WDES%NBANDS
                   CALL SETWAV(W,W1,WDES1,N,ISP); W1%CPROJ => CWORK
                   CALL PROJ1(NONL_S,WDES1,W1)  ! calculate W1%CPROJ (linked to CWORK)

                   WEIGHT=WDES%WTKPT(NK)*WDES%RSPIN*W1%FERWE
                   EVALUE=W1%CELEN

                   LBASE= 0
                   spinor: DO ISPINOR=0,WDES1%NRSPINORS-1
                      DO ISPINOR_=0,WDES%NRSPINORS-1

                         LBASE =ISPINOR *WDES%NPRO/2
                         LBASE_=ISPINOR_*WDES%NPRO/2

                         NIS=1

                         DO NT=1,WDES%NTYP
                            LMMAXC=WDES%LMMAX(NT)
                            IF (LMMAXC==0) GOTO 270

                            DO NI=NIS,WDES%NITYP(NT)+NIS-1
                               DO L=1 ,LMMAXC
                                  DO LP=1,LMMAXC
                                     FNLSIFTMP = FNLSIFTMP+WEIGHT*CWORK(LBASE_+LP)* &
                                          GCONJG(W%CPROJ(LBASE+L,N,NK,ISP))* &
                                          (CDIJ(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)-EVALUE*CQIJ(LP,L,NI,ISP+ISPINOR_+2*ISPINOR))
                                  ENDDO
                               ENDDO

                               LBASE=  LMMAXC+LBASE
                               LBASE_= LMMAXC+LBASE_
                            ENDDO
270                         NIS = NIS+WDES%NITYP(NT)
                         ENDDO
                      ENDDO
                   ENDDO spinor
                ENDDO band
             ENDDO spin
          ENDDO kpoint
          FNLSIF(IDIR,JDIR)=FNLSIFTMP
          !
          !  only isotrop pressure finish now
          !
          IF (ISIF==1) THEN
             FNLSIF(2,2)= FNLSIF(1,1)
             FNLSIF(3,3)= FNLSIF(1,1)
             GOTO 310  ! terminate (not very clean but who cares)
          ENDIF
!=======================================================================
! next direction
!=======================================================================
       ENDDO
    ENDDO

310 CONTINUE
    CALLMPI( M_sum_d(WDES%COMM, FNLSIF, 9))

    FNLSIF=FNLSIF/DIS

#ifdef debug
    WRITE(*,'(E10.3,3E14.7)')DIS,((FNLSIF(I,J),I=1,3),J=1,3)
!    IF (DIS>1E-10) GOTO 1000
!    GOTO 1000
#endif
!=======================================================================
! recalculate the projection operators
! (the array was used as a workspace)
!=======================================================================
    IZERO=1
    CALL SPHER(GRID,NONL_S,P,WDES,LATT_CUR,  IZERO)

    DEALLOCATE(CWORK)

    PROFILING_STOP('strenl')

    RETURN
  END SUBROUTINE STRENL


END MODULE nonl

!************************ SUBROUTINE VNLAC0  ***************************
!
! this subroutine calculates a linear combination of
! projection operatores in reciprocal space
! the result is added to  CACC
!               -----
!***********************************************************************

  SUBROUTINE VNLAC0(NONL_S,WDES1,CPROJ_LOC,CACC)
    USE nonl_struct_def

    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes1)    WDES1
    GDEF    CPROJ_LOC(WDES1%NPRO)
    COMPLEX(q) CACC(WDES1%NRPLWV)
  ! local
    INTEGER :: NPL, LMBASE, ISPIRAL, ISPINOR, NIS, NT, LMMAXC, NI, LM, &
         K, KK
    COMPLEX(q) :: CTMP
#ifdef avoidalloc
    REAL(q) :: WORK(WDES1%NGVECTOR*2),TMP(101,2)
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
#else
    REAL(q),ALLOCATABLE :: WORK(:),TMP(:,:)
    GDEF,ALLOCATABLE    :: CPROJ(:)

    ALLOCATE(WORK(WDES1%NGVECTOR*2),TMP(101,2),CPROJ(WDES1%NPRO_TOT))
#endif
    PROFILING_START('vnlac0')

    IF (WDES1%NK /= NONL_S%NK) THEN
       WRITE(*,*) 'internal error in VNLAC0: PHASE not properly set up',WDES1%NK, NONL_S%NK
    ENDIF
    NPL=WDES1%NGVECTOR

    ! merge projected wavefunctions from all nodes
    CALL MRG_PROJ(WDES1,CPROJ(1),CPROJ_LOC(1))
!=======================================================================
! performe loop over ions
!=======================================================================
    LMBASE= 0
    ISPIRAL = 1

    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NIS=1
       typ: DO NT=1,NONL_S%NTYP
          LMMAXC=NONL_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ion: DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
             DO LM=1,LMMAXC
                CTMP= CPROJ(LMBASE+LM)*CONJG(NONL_S%CQFAK(LM,NT))
                TMP(LM,1)= REAL( CTMP ,KIND=q)
                TMP(LM,2)= AIMAG(CTMP)
             ENDDO
#ifdef do_loops
!DIR$ IVDEP
!OCL NOVREC
             DO K=1,NPL
                WORK(K)=0
                WORK(K+NPL)=0
             ENDDO
!=======================================================================
! call BLAS to calculate SUM(LM=1,LMMAX) QPROJ(K,LM) * TMP(LM)
!=======================================================================
             DO LM=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                DO K=1,NPL
                   WORK(K)    =WORK(K)    +TMP(LM,1)*NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)
                   WORK(K+NPL)=WORK(K+NPL)+TMP(LM,2)*NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)
                ENDDO
             ENDDO
#else
             PROFILING_START('gemm')

             CALL DGEMV( 'N' , NPL, LMMAXC, 1._q , NONL_S%QPROJ(1,1,NT,WDES1%NK,ISPIRAL), &
                  WDES1%NGDIM, TMP(1,1) , 1 , 0._q , WORK(1), 1)
             CALL DGEMV( 'N' , NPL, LMMAXC, 1._q , NONL_S%QPROJ(1,1,NT,WDES1%NK,ISPIRAL), &
                  WDES1%NGDIM, TMP(1,2) , 1 , 0._q , WORK(1+NPL), 1)

             PROFILING_STOP('gemm')
#endif
!=======================================================================
! add acceleration from this ion to CACC
!=======================================================================
             CALL WORK_MUL_CREXP( NPL, WORK(1), NONL_S%CREXP(1,NI), CACC(1+NPL*ISPINOR))

             LMBASE= LMMAXC+LMBASE
          ENDDO ion
600       NIS = NIS+NONL_S%NITYP(NT)
       ENDDO typ
       IF (NONL_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

#ifndef avoidalloc
    DEALLOCATE(WORK,TMP,CPROJ)
#endif

    PROFILING_STOP('vnlac0')

    RETURN
  END SUBROUTINE VNLAC0


!************************ SUBROUTINE VNLAC0_DER  *************************
!
!  subroutine for calculating the non local contribution of
!  the first derivative of the Hamiltonian
!  upon moving one ion ION in the direction IDIR
!  more specifically the  contribution 
!     | d p_i/ d R > c_i
!  for IDIR=0 the derivative is replaced by | p_i >
! a term  (i k)  is included as well to "center" the derivate at the 
! ion
!
!***********************************************************************

  SUBROUTINE VNLAC0_DER(NONL_S,WDES1,CPROJ_LOC,CACC, LATT_CUR, ION, IDIR)
    USE nonl_struct_def
    USE lattice
    USE constant
    IMPLICIT NONE

    TYPE (nonl_struct) NONL_S
    TYPE (wavedes1)    WDES1
    COMPLEX(q) CACC(WDES1%NRPLWV)
    GDEF       CPROJ_LOC(WDES1%NPRO)
    TYPE (latt)  LATT_CUR
    INTEGER      ION, IDIR

    ! work arrays
    GDEF    :: CPROJ(WDES1%NPRO_TOT)
    INTEGER :: NPL, LMBASE, LMMAXC, ISPINOR,  ISPIRAL, K, KK, NI, NIS, NT, LM, L
    COMPLEX(q) CMUL
    COMPLEX(q) CACC_ADD(WDES1%NRPLWV)

    PROFILING_START('vnlac0_der')

    IF (WDES1%NK /= NONL_S%NK) THEN
       WRITE(*,*) 'internal error in VNLAC0_DER: PHASE not properly set up',WDES1%NK, NONL_S%NK
       STOP
    ENDIF

    NPL=WDES1%NGVECTOR

    ! merge projected wavefunctions from all nodes
    CALL MRG_PROJ(WDES1,CPROJ(1),CPROJ_LOC(1))
!=======================================================================
! performe loop over ions
!=======================================================================
    LMBASE= 0
    ISPIRAL = 1

    spinor: DO ISPINOR=0,WDES1%NRSPINORS-1

       NIS=1
       typ: DO NT=1,NONL_S%NTYP
          LMMAXC=NONL_S%LMMAX(NT)
          IF (LMMAXC==0) GOTO 600

          ions: DO NI=NIS,NONL_S%NITYP(NT)+NIS-1
             IF (NI==ION) THEN
                CACC_ADD=0
                l_loop: DO LM=1,LMMAXC
                   CMUL=CPROJ(LMBASE+LM)*CONJG(NONL_S%CQFAK(LM,NT))
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL
                      CACC_ADD(K)=CACC_ADD(K)+NONL_S%QPROJ(K,LM,NT,WDES1%NK,ISPIRAL)*CMUL* &
                           (CONJG(NONL_S%CREXP(K,NI)))
                   ENDDO
                ENDDO l_loop
!=======================================================================
! add acceleration from this ion
!=======================================================================

                IF (IDIR==0) THEN
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL
                      KK=K+NPL*ISPINOR
                      CACC(KK)=CACC(KK)+CACC_ADD(K)
                   ENDDO
                ELSE
!DIR$ IVDEP
!OCL NOVREC
                   DO K=1,NPL
                      KK=K+NPL*ISPINOR
                      CACC(KK)= CACC(KK)+CACC_ADD(K)*CONJG(CITPI)* &
#ifdef shift_der_k
                          ((WDES1%IGX(K)+WDES1%VKPT(1))*LATT_CUR%B(IDIR,1)+ &
                           (WDES1%IGY(K)+WDES1%VKPT(2))*LATT_CUR%B(IDIR,2)+ &
                           (WDES1%IGZ(K)+WDES1%VKPT(3))*LATT_CUR%B(IDIR,3))
#else
                          ((WDES1%IGX(K))*LATT_CUR%B(IDIR,1)+ &
                           (WDES1%IGY(K))*LATT_CUR%B(IDIR,2)+ &
                           (WDES1%IGZ(K))*LATT_CUR%B(IDIR,3))
#endif
                   ENDDO
                ENDIF

             ENDIF
             LMBASE= LMMAXC+LMBASE
          ENDDO ions

600       NIS = NIS+NONL_S%NITYP(NT)
       ENDDO typ
       IF (NONL_S%LSPIRAL) ISPIRAL=2
    ENDDO spinor

    PROFILING_STOP('vnlac0_der')

    RETURN
  END SUBROUTINE VNLAC0_DER


!***********************************************************************
!
! small f77 helper  routine to 
! multiply with phasefactor and divide into real and imaginary part
!
!***********************************************************************


  SUBROUTINE CREXP_MUL_WAVE( NPL, CREXP, CW, WORK)
    USE prec
    IMPLICIT NONE
    INTEGER NPL
    COMPLEX(q) :: CREXP(NPL), CW(NPL)
    REAL(q)    :: WORK(2*NPL)
    COMPLEX(q) :: CTMP
  ! local
    INTEGER K

    DO K=1,NPL                 
       CTMP=    CREXP(K) *CW(K)
       WORK(K)    = REAL( CTMP ,KIND=q)
       WORK(K+NPL)= AIMAG(CTMP)
    ENDDO
  END SUBROUTINE CREXP_MUL_WAVE


  SUBROUTINE WORK_MUL_CREXP( NPL, WORK, CREXP, CW)
    USE prec
    IMPLICIT NONE
    INTEGER NPL
    COMPLEX(q) :: CREXP(NPL), CW(NPL)
    REAL(q)    :: WORK(2*NPL)
    COMPLEX(q) :: CTMP
  ! local
    INTEGER K

    PROFILING_START('work_mul_crexp')

    DO K=1,NPL
       CW(K)=CW(K)+ CMPLX( WORK(K) , WORK(K+NPL) ,KIND=q) *CONJG(CREXP(K))
    ENDDO

    PROFILING_STOP('work_mul_crexp')

  END SUBROUTINE WORK_MUL_CREXP
