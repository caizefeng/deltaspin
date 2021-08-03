#include "symbol.inc" 
!*********************************************************************************
!
! this module implements the base local Hartree Fock routines
! on the logarithmic PAW grid (one centre terms)
!
! for details on the KLI method see
!   Krieger, Li, and Iafrate, Phys. Rev. A46, 5453 (1992)
! the local Hartree Fock method was introduced in
!   Sala and Goerling, J. Chem. Phys. 115, 5718 (2001)
!
! Both methods require the calculation of a correction term and 
! the Slater energy density
! e(r) = \sum_ab rho_ab phi_b(r) V_x(r,r') phi_a(r')
! where V_x is the exact exchange potential, c_ab are one centre
! occupancy matrices, and phi the AE or PS partial waves.
!
!*********************************************************************************

MODULE PAWKLI
  USE prec
  USE pseudo
  USE pawfock
  USE paw
  USE constant

  TYPE ONE_CENTER_OCCUPANCY_AE
     OVERLAP, POINTER :: QOC(:,:)
  END TYPE ONE_CENTER_OCCUPANCY_AE

  TYPE (ONE_CENTER_OCCUPANCY_AE), ALLOCATABLE :: COCC_ADD(:,:)

  REAL(q), ALLOCATABLE, SAVE :: POTAE_CORR(:,:,:)    ! AE correction potential for "local" ions
  REAL(q), ALLOCATABLE, SAVE :: POTPS_CORR(:,:,:)    ! PS correction potential for "local" ions
  INTEGER, SAVE :: NDIM_POT=0

  REAL(q), PARAMETER  :: MIN_CHARGE=1E-4

! this flag determines whether core is handled by the subroutine
!  LOGICAL, PARAMETER  :: USE_CORE=.TRUE.
! presently USE_CORE= .TRUE. is most likely not working at all
  LOGICAL, PARAMETER  :: USE_CORE=.FALSE.

#define LMAXFOCKAE
CONTAINS
! **************************** SUBROUTINE CALC_PAW_SLATERPOT ***********************
!
!  calculates and stores the PAW one centre (on site) Slater energy density e(r)
!  required for EXX-KLI and EXX-LHF
!  presently only the spherical contributions are calculated
!  The routine determines the Slater energy density
!  the Slater potential must be calculated by the calling routine by division
!  of the energy density by the charge density rho(r)
!
!  e_s(r)   = Sum_1234 rho_14*K_1234(r)*rho_32
!  rho(r) = Sum_12   c_00l1m1l2m2 *rho_12
!
!  K_1234(r) = Sum_LM c_LM,l1m1,l2m2 * S_n1n2n3n4L(r) * c_LM,l3m3,l4m4
!
!  S_n1n2n3n4 L(r) =phi_n1(r) phi_n2(r) \int dr' G_L(r,r') phi_n3(r') phi_n4(r')
!
!  where G_L(r',r) is the Coloumb kernel (or Greens function)
!  for the angular quantum number L
!
!  G_L(r',r) =  4 pi/(2L+1) (r',r)<^(L) /(r',r)>^(L+1) 
!
!  and
!  c_LM,l1m1,l2m2 are the Gaunt coefficients stored in YLM3
!
!  1,2,3 and 4     are compound indices including the channel and l,m indices
!                   e.g. 1=n1l1m1
!
!  POTPS(AE)_AVERAGE is the average local pseudo (all electron) potential 
!                    (valence only)
!  RHOPS(AE)_AVERAGE is the average local pseudo (all electron) charge 
!                    (valence only)
!
! **********************************************************************************

    SUBROUTINE CALC_PAW_SLATERPOT(COCC, PP, POTAE_AVERAGE, POTPS_AVERAGE, &
         RHOAE_AVERAGE, RHOPS_AVERAGE,  DFOCKAE, DFOCKPS)
      USE asa
      IMPLICIT NONE
      
      OVERLAP :: COCC(:,:), DFOCKAE(:,:), DFOCKPS(:,:)
      TYPE (potcar),POINTER:: PP
      REAL(q) :: RHOAE_AVERAGE(:), RHOPS_AVERAGE(:)
      REAL(q) :: POTAE_AVERAGE(:), POTPS_AVERAGE(:)

      INTEGER, EXTERNAL :: MAXL1, FOCK_LMAXONECENTER
     !local variables
      INTEGER LYMAX, CHANNELS
      INTEGER I,J,K,L,LI,LJ,LK,LL,MI,MJ,MK,ML,ISTART1,ISTART2,IEND1,IEND2,LMIND1,LMIND2,IC1,IC2,NAE
      INTEGER LMI,LMJ,LMK,LML
      INTEGER LLOOP, N
      REAL(q), ALLOCATABLE :: POTPS(:,:), POTAE(:,:), SUM(:), S_PS(:), S_AE(:)
      REAL(q) :: SUMC(0:0)
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE

      ! the energy density can be calculated using the
      ! improved reconstruction (LMAXFOCKAE) 
      ! or using the simple reconstruction
      ! the simple yields a better OEP potential
      IF (ASSOCIATED(PP%QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(PP%QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(PP%QPAW_FOCK,3)-1
      ELSE
         NMAX_FOCKAE=1
         LMAX_FOCKAE=-1
      ENDIF
#ifndef LMAXFOCKAE
      LMAX_FOCKAE=-1
#endif

      LYMAX =MAXL1(PP)*2/2   ! LYMAX is half the maximum L augmentation number
                             ! since the L index in S is divided by 2
      CHANNELS=PP%LMAX

      ALLOCATE(POTPS(PP%R%NMAX,0:LYMAX),POTAE(PP%R%NMAX,0:LYMAX), & 
           SUM(0:LYMAX), S_AE(0:LYMAX), S_PS(0:LYMAX))



      POTPS_AVERAGE=0
      POTAE_AVERAGE=0
      RHOPS_AVERAGE=0
      RHOAE_AVERAGE=0

      DFOCKAE=0
      DFOCKPS=0
!-----------------------------------------------------------------------
! loop i,j
!-----------------------------------------------------------------------
      LMI=1
      DO I=1,CHANNELS
      LMJ=1
      DO J=1,CHANNELS
            
         LI = PP%LPS(I)
         LJ = PP%LPS(J)
!-----------------------------------------------------------------------
!  determine 
!  \int dr' G_L(r,r') phi_i(r') phi_j(r')
!-----------------------------------------------------------------------
         CALL COLOUMB_2POT( PP%WAE(:,I), PP%WAE(:,J), & 
              LI, LJ, PP%R, POTAE)
         IF (ASSOCIATED(PP%QPAW_FOCK) .AND. LMAX_FOCKAE>=0) THEN
            CALL COLOUMB_2POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), &
              LI, LJ, PP%R, POTPS, NMAX_FOCKAE, LMAX_FOCKAE, & 
              PP%AUG_FOCK, PP%QPAW_FOCK(I,J,:,:))
         ELSE
            CALL COLOUMB_2POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), &
              LI, LJ, PP%R, POTPS, NMAX_FOCKAE, LMAX_FOCKAE )
         ENDIF
!-----------------------------------------------------------------------
! loop k,l
! determine Slater integrals between four orbitals
!-----------------------------------------------------------------------
         LMK=1
         DO K=1,CHANNELS
         LML=1
         DO L=1,CHANNELS

            CALL YLM3LOOKUP(LI,LJ,LMIND1)
            LK = PP%LPS(K)
            LL = PP%LPS(L)

            S_PS = 0
            S_AE = 0
            DO LLOOP=ABS(LI-LJ),MIN(ABS(LI+LJ),LYMAX*2),2
               DO N=1,PP%R%NMAX
                  S_AE(LLOOP/2) = S_AE(LLOOP/2) + POTAE(N,LLOOP/2)* &
                       (PP%WAE(N,K)*PP%WAE(N,L)*PP%R%SI(N))
                  S_PS(LLOOP/2) = S_PS(LLOOP/2) + POTPS(N,LLOOP/2)* &
                       (PP%WPS(N,K)*PP%WPS(N,L)+PP%AUG_SOFT(N,LLOOP)*PP%QPAW(L,K,LLOOP))*PP%R%SI(N)
               ENDDO
               IF (LLOOP <= LMAX_FOCKAE) THEN
                  DO NAE=1,NMAX_FOCKAE
                  DO N=1,PP%R%NMAX
                     S_PS(LLOOP/2) = S_PS(LLOOP/2) + POTPS(N,LLOOP/2)* &
                          PP%AUG_FOCK(N,LLOOP,NAE)*PP%QPAW_FOCK(L,K,LLOOP,NAE)*PP%R%SI(N)
                  ENDDO
                  ENDDO
               ENDIF
               
            ENDDO
!-----------------------------------------------------------------------
!  S_klij L(r) =phi_k(r) phi_l(r) \int dr' G_L(r,r') phi_i(r') phi_j(r')
!-----------------------------------------------------------------------
            ! loop mi,mj
            SUM=0
            SUMC=0

            DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)
!-----------------------------------------------------------------------
! charge density  SUMC = sum_mi,mj c_00,limi,ljmj rho_ij
!-----------------------------------------------------------------------
               DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  IF ( JS(IC1)==1) THEN
                     SUMC(0) = SUMC(0)+COCC(LMI+MI-1,LMJ+MJ-1)*YLM3(IC1)
                  ENDIF
               ENDDO
 
               CALL YLM3LOOKUP(LK,LL,LMIND2)
               
               ! loop mk,ml
               DO MK=1,2*LK+1
               DO ML=1,2*LL+1
                  LMIND2 = LMIND2+1
                  ISTART2=INDCG(LMIND2)
                  IEND2=INDCG(LMIND2+1)
 
                  DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  DO IC2=ISTART2,IEND2-1  ! IC2 compound index for LM
                  IF (JS(IC2) == JS(IC1)) THEN

!-----------------------------------------------------------------------
!     SUM(L) = SUM_{M} c_LM,limi,ljmj* c_LM,lkmk,llml
!-----------------------------------------------------------------------
                     SUM(JL(IC2)/2) = SUM(JL(IC2)/2)+COCC(LMI+MI-1,LML+ML-1)* &
                          YLM3(IC1)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                     DFOCKPS(LMI+MI-1,LML+ML-1) = DFOCKPS(LMI+MI-1,LML+ML-1) &
                        + YLM3(IC1)*S_PS(JL(IC2)/2)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                     DFOCKAE(LMI+MI-1,LML+ML-1) = DFOCKAE(LMI+MI-1,LML+ML-1) &
                        + YLM3(IC1)*S_AE(JL(IC2)/2)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                  ENDIF
                  ENDDO
                  ENDDO
               ENDDO ! ML
               ENDDO ! MK

            ENDDO ! MI
            ENDDO ! MJ

            CALL ADD_WEIGHTED_POT_AE( PP%WAE(:,K), PP%WAE(:,L), & 
              LK, LL, PP%R, SUM, POTAE_AVERAGE, POTAE)

         IF (ASSOCIATED(PP%QPAW_FOCK).AND. LMAX_FOCKAE>=0) THEN
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,K), PP%WPS(:,L), PP%AUG_SOFT, PP%QPAW(K,L,:), &
                 LK, LL, PP%R, SUM, POTPS_AVERAGE, NMAX_FOCKAE, LMAX_FOCKAE, POT=POTPS, &
                 AUG_FOCK=PP%AUG_FOCK, QPAW_FOCK = PP%QPAW_FOCK(K,L,:,:))
         ELSE
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,K), PP%WPS(:,L), PP%AUG_SOFT, PP%QPAW(K,L,:), &
                 LK, LL, PP%R, SUM, POTPS_AVERAGE , NMAX_FOCKAE, LMAX_FOCKAE, POT=POTPS)
         ENDIF

            LML=LML+2*LL+1
         ENDDO ! L
         LMK=LMK+2*LK+1
         ENDDO ! K


         ! construct charge density
         CALL ADD_WEIGHTED_POT_AE( PP%WAE(:,I), PP%WAE(:,J), & 
              LI, LJ, PP%R, SUMC, RHOAE_AVERAGE )

         IF (ASSOCIATED(PP%QPAW_FOCK).AND. LMAX_FOCKAE>=0) THEN
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), &
                 LI, LJ, PP%R, SUMC, RHOPS_AVERAGE, NMAX_FOCKAE, LMAX_FOCKAE, &
                 AUG_FOCK=PP%AUG_FOCK, QPAW_FOCK = PP%QPAW_FOCK(I,J,:,:))
         ELSE
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), &
                 LI, LJ, PP%R, SUMC, RHOPS_AVERAGE, NMAX_FOCKAE, LMAX_FOCKAE )
         ENDIF

         LMJ=LMJ+2*LJ+1
      ENDDO ! J

      LMI=LMI+2*LI+1
      ENDDO ! I

      DEALLOCATE(POTPS, POTAE, SUM, S_AE, S_PS)
 
    END SUBROUTINE CALC_PAW_SLATERPOT


! **************************** SUBROUTINE CALC_PAW_CORE_SLATERPOT ******************
!
!  calculates and stores the PAW one centre (on site) EXX-KLI energy density for the
!  core electrons (Slater potential)
!  for documentation see above
!
! **********************************************************************************

    SUBROUTINE CALC_PAW_CORE_SLATERPOT(COCC, R, LYMAX, CHANNELS, LPS, WAE, & 
         POTAE_AVERAGE, RHOAE_AVERAGE, DFOCK )
      USE asa
      IMPLICIT NONE

      OVERLAP :: COCC(:,:), DFOCK(:,:)
      TYPE (rgrid) R
      INTEGER LYMAX           ! LYMAX is half the maximum L augmentation number
      INTEGER CHANNELS        ! number of channels
      INTEGER LPS(:)          ! L-quantum number of each channel
      REAL(q) WAE(:,:)        ! AE wavefunctions
      REAL(q) :: POTAE_AVERAGE(:)
      REAL(q) :: RHOAE_AVERAGE(:)

      INTEGER, EXTERNAL :: MAXL1, FOCK_LMAXONECENTER
     !local variables
      INTEGER I,J,K,L,LI,LJ,LK,LL,MI,MJ,MK,ML,ISTART1,ISTART2,IEND1,IEND2,LMIND1,LMIND2,IC1,IC2
      INTEGER LMI,LMJ,LMK,LML
      INTEGER LLOOP, N
      REAL(q) :: SUM(0:LYMAX), SUMC(0:0), S(0:LYMAX)
      REAL(q), ALLOCATABLE :: POTAE(:,:)

      ALLOCATE(POTAE(R%NMAX,0:LYMAX))

      POTAE_AVERAGE=0
      RHOAE_AVERAGE=0

      DFOCK=0
!-----------------------------------------------------------------------
! loop i,j
!-----------------------------------------------------------------------
      LMI=1
      DO I=1,CHANNELS
      LMJ=1
      DO J=1,CHANNELS
            
         LI = LPS(I)
         LJ = LPS(J)
!-----------------------------------------------------------------------
!  determine 
!  \int dr' G_L(r,r') phi_i(r') phi_j(r')
!-----------------------------------------------------------------------
         CALL COLOUMB_2POT( WAE(:,I), WAE(:,J), & 
              LI, LJ, R, POTAE)
!-----------------------------------------------------------------------
! loop k,l
!-----------------------------------------------------------------------
         LMK=1
         DO K=1,CHANNELS
         LML=1
         DO L=1,CHANNELS

            S = 0
            DO LLOOP=ABS(LI-LJ),MIN(ABS(LI+LJ),LYMAX*2),2
               DO N=1,R%NMAX
                  S(LLOOP/2) = S(LLOOP/2) + POTAE(N,LLOOP/2)*WAE(N,K)*WAE(N,L)*R%SI(N)
               ENDDO
            ENDDO

            CALL YLM3LOOKUP(LI,LJ,LMIND1)
            LK = LPS(K)
            LL = LPS(L)
!-----------------------------------------------------------------------
!  determine 
!  S_klij L(r) =phi_k(r) phi_l(r) \int dr' G_L(r,r') phi_i(r') phi_j(r')
!-----------------------------------------------------------------------
            ! loop mi,mj
            SUM=0
            SUMC=0

            DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

!-----------------------------------------------------------------------
! charge density  SUMC = sum_mi,mj c_00,limi,ljmj rho_ij
!-----------------------------------------------------------------------
               DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  IF ( JS(IC1)==1) THEN
                     SUMC(0) = SUMC(0)+COCC(LMI+MI-1,LMJ+MJ-1)*YLM3(IC1)
                  ENDIF
               ENDDO
 
               CALL YLM3LOOKUP(LK,LL,LMIND2)
               
               ! loop mk,ml
               DO MK=1,2*LK+1
               DO ML=1,2*LL+1
                  LMIND2 = LMIND2+1
                  ISTART2=INDCG(LMIND2)
                  IEND2=INDCG(LMIND2+1)
 
                  DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  DO IC2=ISTART2,IEND2-1  ! IC2 compound index for LM
                  IF (JS(IC2) == JS(IC1)) THEN

!-----------------------------------------------------------------------
!     SUM(L) = SUM_{M} c_LM,limi,ljmj* c_LM,lkmk,llml
!-----------------------------------------------------------------------
                     SUM(JL(IC2)/2) = SUM(JL(IC2)/2)+COCC(LMI+MI-1,LML+ML-1)* &
                          YLM3(IC1)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                     DFOCK(LMI+MI-1,LML+ML-1) = DFOCK(LMI+MI-1,LML+ML-1) &
                        + YLM3(IC1)*S(JL(IC2)/2)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                  ENDIF
                  ENDDO
                  ENDDO
               ENDDO ! ML
               ENDDO ! MK

            ENDDO ! MI
            ENDDO ! MJ

            CALL ADD_WEIGHTED_POT_AE( WAE(:,K), WAE(:,L), & 
              LK, LL, R, SUM, POTAE_AVERAGE, POTAE )

            LML=LML+2*LL+1
         ENDDO ! L
         LMK=LMK+2*LK+1
         ENDDO ! K

         CALL ADD_WEIGHTED_POT_AE( WAE(:,I), WAE(:,J), & 
              LI, LJ, R, SUMC, RHOAE_AVERAGE )
         
         LMJ=LMJ+2*LJ+1
      ENDDO ! J

      LMI=LMI+2*LI+1
      ENDDO ! I

      DEALLOCATE(POTAE )
 
    END SUBROUTINE CALC_PAW_CORE_SLATERPOT
    
! **************************** SUBROUTINE CALC_PAW_LOCAL_CORR **********************
!
! determine one centre strenght parameter from current local potential
!  D_ij = D_ni li mi, nj lj m = int r^2 dr V(r) 
!           phi_ni(r) phi_nj(r) C_LM,li mi,lj mj
! 
! where V(r) is the current local approximation to the exchange potential
! only spherical contributions are considered
!
! **********************************************************************************

    SUBROUTINE CALC_PAW_LOCAL_CORR( DCORR_AE, DCORR_PS, PP, POTAE, POTPS )
      USE asa
      IMPLICIT NONE

      OVERLAP :: DCORR_AE(:,:), DCORR_PS(:,:)
      TYPE (potcar),POINTER:: PP
      REAL(q) :: POTAE(:), POTPS(:)

     !local variables
      INTEGER I,J,LI,LJ,MI,MJ,ISTART1,IEND1,LMIND1,IC1,LMI,LMJ, NN
      INTEGER N
      REAL(q) :: POT(PP%R%NMAX)
      REAL(q) :: SUMC_PS(0:0),SUMC_AE(0:0)

      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE, NAE

      IF (ASSOCIATED(PP%QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(PP%QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(PP%QPAW_FOCK,3)-1
      ELSE
         NMAX_FOCKAE=1
         LMAX_FOCKAE=-1
      ENDIF
#ifndef LMAXFOCKAE
      LMAX_FOCKAE=-1
#endif

      DCORR_AE=0
      DCORR_PS=0
      
      LMI=1
      DO I=1,PP%LMAX
      LMJ=1
      DO J=1,PP%LMAX
         ! calculate int r^2 dr V(r) phi_ni(r) phi_nj(r)
         DO N=1,PP%R%NMAX
            POT(N)=PP%WAE(N,I)*PP%WAE(N,J)*POTAE(N)
         ENDDO
         CALL SIMPI(PP%R, POT, SUMC_AE(0))

         DO N=1,PP%R%NMAX
            POT(N)=(PP%WPS(N,I)*PP%WPS(N,J)+PP%AUG_SOFT(N,0)*PP%QPAW(I,J,0))*POTPS(N)
         ENDDO
         IF (LMAX_FOCKAE >= 0 ) THEN
            DO NAE=1,NMAX_FOCKAE
            DO N=1,PP%R%NMAX
               POT(N)=POT(N)+PP%AUG_FOCK(N,0,NAE)*PP%QPAW_FOCK(I,J,0,NAE)*POTPS(N)
            ENDDO
            ENDDO
         ENDIF

         CALL SIMPI(PP%R, POT, SUMC_PS(0))

         LI = PP%LPS(I)
         LJ = PP%LPS(J)
         CALL YLM3LOOKUP(LI,LJ,LMIND1)

         DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
 
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)
               DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  IF ( JS(IC1)==1) THEN  ! only L=0 component implemented right now
                     DCORR_AE(LMI+MI-1,LMJ+MJ-1) = SUMC_AE(0)*YLM3(IC1)
                     DCORR_PS(LMI+MI-1,LMJ+MJ-1) = SUMC_PS(0)*YLM3(IC1)
                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         LMJ=LMJ+2*LJ+1
      ENDDO
      LMI=LMI+2*LI+1
      ENDDO
    END SUBROUTINE CALC_PAW_LOCAL_CORR

    SUBROUTINE CALC_PAW_CORE_LOCAL_CORR( DCORR, R,  CHANNELS, LPS, WAE, POTAE )
      USE asa
      IMPLICIT NONE

      TYPE (rgrid) R
      INTEGER CHANNELS        ! number of channels
      INTEGER LPS(:)          ! L-quantum number of each channel
      REAL(q) WAE(:,:)        ! AE wavefunctions
      OVERLAP :: DCORR(:,:)
      REAL(q) :: POTAE(:)

     !local variables
      INTEGER I,J,LI,LJ,MI,MJ,ISTART1,IEND1,LMIND1,IC1,LMI,LMJ
      INTEGER N
      REAL(q) :: POT(R%NMAX)
      REAL(q) :: SUM(0:0)

      DCORR=0
      LMI=1
      DO I=1,CHANNELS
      LMJ=1
      DO J=1,CHANNELS
         ! calculate int r^2 dr V(r) phi_ni(r) phi_nj(r)
         DO N=1,R%NMAX
            POT(N)=WAE(N,I)*WAE(N,J)*POTAE(N)
         ENDDO
         CALL SIMPI(R, POT, SUM(0))

         LI = LPS(I)
         LJ = LPS(J)
         CALL YLM3LOOKUP(LI,LJ,LMIND1)

         DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
 
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)
               DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  IF ( JS(IC1)==1) THEN  ! only L=0 component implemented right now
                     DCORR(LMI+MI-1,LMJ+MJ-1) = SUM(0)*YLM3(IC1)
                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         LMJ=LMJ+2*LJ+1
      ENDDO
      LMI=LMI+2*LI+1
      ENDDO

    END SUBROUTINE CALC_PAW_CORE_LOCAL_CORR


! **************************** SUBROUTINE CALC_PAW_CORR ***************************
!
!  calculates the PAW one centre correction terms to the energy density e(r)
!  in the LHF approach
!  (only spherical contribution e_s(r) is considered, which can be obtained
!   by calculating e_s(r) = 1/Y_00 \int d Omega_r e(r) Y_00(r))
!
!  e_s(r) = 1/Y_00 SUM_14 c_00,l1m1,l4m4 rho_14    phi_n1(r) phi_n4(r)
!
!  c_LM,l1m1,l2m2 are the Gaunt coefficients stored in YLM3
!
!  1 and 4    are compound indices including the channel and l,m indices
!             e.g. 1=n1l1m1
!
! **********************************************************************************


    SUBROUTINE CALC_PAW_CORR(COCC, PP, POTAE, POTPS )
      USE asa
      IMPLICIT NONE

      OVERLAP :: COCC(:,:)
      TYPE (potcar),POINTER:: PP
      REAL(q) :: POTAE(:), POTPS(:)

      INTEGER, EXTERNAL :: MAXL1, FOCK_LMAXONECENTER
     !local variables
      INTEGER I,J,LI,LJ,MI,MJ,ISTART1,IEND1,LMIND1,IC1
      INTEGER LMI,LMJ
      INTEGER N
      REAL(q) :: POT(PP%R%NMAX)
      REAL(q) :: SUM(0:0)
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE

      IF (ASSOCIATED(PP%QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(PP%QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(PP%QPAW_FOCK,3)-1
      ELSE
         NMAX_FOCKAE=1
         LMAX_FOCKAE=-1
      ENDIF
#ifndef LMAXFOCKAE
      LMAX_FOCKAE=-1
#endif
      
      POTAE=0
      POTPS=0
!-----------------------------------------------------------------------
! loop i,j
!-----------------------------------------------------------------------
      LMI=1
      DO I=1,PP%LMAX
      LMJ=1
      DO J=1,PP%LMAX
            
         LI = PP%LPS(I)
         LJ = PP%LPS(J)

         SUM=0
!-----------------------------------------------------------------------
! loop k,l
!-----------------------------------------------------------------------
         CALL YLM3LOOKUP(LI,LJ,LMIND1)

         ! loop mi,mj
         DO MI=1,2*LI+1
         DO MJ=1,2*LJ+1

            LMIND1 = LMIND1+1
            ISTART1=INDCG(LMIND1)
            IEND1=INDCG(LMIND1+1)
            DO IC1=ISTART1,IEND1-1 !IC for Comp.Ind. LM
               IF ( JL(IC1)==0) THEN ! only L=0 component implemented right now
                  ! loop mk,ml
                  SUM(0)  = SUM(0) +YLM3(IC1)*COCC(LMI+MI-1,LMJ+MJ-1)
               ENDIF
            ENDDO
         ENDDO ! MI
         ENDDO ! MJ

         CALL ADD_WEIGHTED_POT_AE( PP%WAE(:,I), PP%WAE(:,J), & 
              LI, LJ, PP%R, SUM, POTAE)
         IF (ASSOCIATED(PP%QPAW_FOCK).AND. LMAX_FOCKAE>=0) THEN
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), & 
                 LI, LJ, PP%R, SUM, POTPS, NMAX_FOCKAE, LMAX_FOCKAE,  &
                 AUG_FOCK=PP%AUG_FOCK, QPAW_FOCK = PP%QPAW_FOCK(I,J,:,:))
         ELSE
            CALL ADD_WEIGHTED_POT_PS( PP%WPS(:,I), PP%WPS(:,J), PP%AUG_SOFT, PP%QPAW(I,J,:), & 
                 LI, LJ, PP%R, SUM, POTPS, NMAX_FOCKAE, LMAX_FOCKAE )
         ENDIF

         LMJ=LMJ+2*LJ+1
      ENDDO ! J

      LMI=LMI+2*LI+1
      ENDDO ! I
 
    END SUBROUTINE CALC_PAW_CORR

! **************************** SUBROUTINE CALC_PAW_CORE_CORR ***********************
!
!  calculates the PAW one centre correction terms to the energy density e(r)
!  (only spherical contributions are considered)
!  includes an update related to core electrons
!
! **********************************************************************************

    SUBROUTINE CALC_PAW_CORE_CORR(COCC, R,  CHANNELS, LPS, WAE, POTAE )
      USE asa
      IMPLICIT NONE

      OVERLAP :: COCC(:,:)
      TYPE (rgrid) R
      INTEGER CHANNELS        ! number of channels
      INTEGER LPS(:)          ! L-quantum number of each channel
      REAL(q) WAE(:,:)        ! AE wavefunctions
      REAL(q) :: POTAE(:)

      INTEGER, EXTERNAL :: MAXL1, FOCK_LMAXONECENTER
     !local variables
      INTEGER I,J,K,L,LI,LJ,MI,MJ,ISTART1,IEND1,LMIND1,IC1
      INTEGER LMI,LMJ
      INTEGER N
      REAL(q) :: POT(R%NMAX)
      REAL(q) :: SUM(0:0), SUM_

!-----------------------------------------------------------------------
! loop i,j
!-----------------------------------------------------------------------
      LMI=1
      DO I=1,CHANNELS
      LMJ=1
      DO J=1,CHANNELS
            
         LI = LPS(I)
         LJ = LPS(J)
         SUM=0

         CALL YLM3LOOKUP(LI,LJ,LMIND1)

         ! loop mi,mj
         DO MI=1,2*LI+1
         DO MJ=1,2*LJ+1

            LMIND1 = LMIND1+1
            ISTART1=INDCG(LMIND1)
            IEND1=INDCG(LMIND1+1)
            DO IC1=ISTART1,IEND1-1 !IC for Comp.Ind. LM
               IF ( JL(IC1)==0) THEN ! only L=0 component implemented right now
                  ! loop mk,ml
                  SUM(0) = SUM(0)+YLM3(IC1)*COCC(LMI+MI-1,LMJ+MJ-1)
               ENDIF
            ENDDO
         ENDDO ! MI
         ENDDO ! MJ

         CALL ADD_WEIGHTED_POT_AE( WAE(:,I), WAE(:,J), & 
              LI, LJ, R, SUM, POTAE)

         LMJ=LMJ+2*LJ+1
      ENDDO ! J
      
      LMI=LMI+2*LI+1
    ENDDO ! I
 
    END SUBROUTINE CALC_PAW_CORE_CORR

!*******************************************************************
!
! small helper routine that  calculates
!  sum_ij D_ij rho_ij 
! and dumps the energy
!
!*******************************************************************


    SUBROUTINE KLI_ENERGY_OCC_STRENGH(STRING, LMMAX, DFOCK, CRHODE)
      IMPLICIT NONE
      CHARACTER (LEN=*) STRING
      INTEGER LMMAX
      OVERLAP :: DFOCK(:,:), CRHODE(:,:)

    ! local
      INTEGER LM1, LM2
      REAL (q) :: HF_ENERGY

      HF_ENERGY=0
      DO LM1=1,LMMAX
         DO LM2=1,LMMAX
            HF_ENERGY=HF_ENERGY+DFOCK(LM1,LM2)*CHGCONJG(CRHODE(LM1,LM2))
         ENDDO
      ENDDO

      WRITE(*,*) STRING, HF_ENERGY
    END SUBROUTINE KLI_ENERGY_OCC_STRENGH



! **********************************************************************************
!   CALC_LHF_ONE_CENTRE
!
! calculate the local Hartree Fock one centre correction terms 
! DFOCK_LOCAL
!   D_ij= - <phi^ps_i| V_x + V_local | phi^ps_j> +
!           <phi_i   | V_x + V_local | phi_j> 
! DLOCAL
!   D_ij= - <phi^ps_i| V_local | phi^ps_j> +
!           <phi_i   | V_local | phi_j> 
!
! the local exchange potential is updated if CRHODE_ADD is
! passed to the routine
! the update is performed using a straight "mixing"
!
!
! **********************************************************************************


    SUBROUTINE CALC_LHF_ONE_CENTRE(NT, PP, CRHODE, DFOCK_LOCAL, DLOCAL, D_CORE, &
         POTAE_CORR, POTPS_CORR, LCORR, LCORE, DOUBLEC_HF, COCC_ADD, CRHODE_ADD , &
         POTAE_FINAL, POTPS_FINAL )
      USE cl
      USE pawfock_inter
      IMPLICIT NONE

      INTEGER NT
      TYPE (potcar), POINTER ::  PP
      OVERLAP :: CRHODE(:,:),  DFOCK_LOCAL(:,:), DLOCAL(:,:)
      OVERLAP :: D_CORE
      OVERLAP, POINTER :: COCC_ADD(:,:)
      OVERLAP, OPTIONAL :: CRHODE_ADD(:,:)
      REAL(q), OPTIONAL :: POTAE_FINAL(:), POTPS_FINAL(:)
      ! this is tricky
      ! the pseudo charge density can become negative
      ! only by adding the pseudo core positive definite values can be obtained
      ! if LCORE is set the pseudo core is therefore added
      LOGICAL LCORE
    ! one centre correction energy density
      REAL(q) :: POTAE_CORR(:), POTPS_CORR(:)
      LOGICAL :: LCORR   ! corrections terms to potential are initialised
      REAL(q) :: DOUBLEC_HF
    !local
      INTEGER CHANNELS_CORE, CHANNELS
      REAL(q), ALLOCATABLE :: W(:,:), A(:,:),B(:,:), EIG(:)
      INTEGER, ALLOCATABLE :: N(:), LC(:)
      OVERLAP, ALLOCATABLE :: COCC(:,:), DFOCKAE(:,:),  DFOCKAE_CORE(:,:), &
           DFOCKPS(:,:), DCORRAE(:,:), DCORRPS(:,:)
      INTEGER :: CH1,CH2, LYMAX, LMMAX, LMMAX_C, RNMAX, M, LM1, LM2
      REAL(q),ALLOCATABLE :: POTAE_SLATER(:), POTAE_SLATER_VAL(:), POTPS_SLATER(:), &
                             RHOAE(:), RHOAE_VAL(:), RHOPS(:),POTAE(:), POTPS(:), TMP(:)
      REAL(q)             :: HF_ENERGY     ! AE valence and core Fock energy
      REAL(q)             :: HF_ENERGY_VAL ! valence only Fock energy
      REAL(q)             :: HF_ENERGY_PS  ! pseudo Fock energy
      REAL(q)             :: HF_ENERGY_PS_VAL ! pseudo valence Fock energy
      REAL(q)             :: POT_SHIFT_FROM_CORE 
      REAL(q)             :: SUM, RHO_CORE
      INTEGER I
      REAL(q) :: AEXX

      CALL HFPARAMETERS(AEXX)        ! amount of exact exchange
      RNMAX =PP%R%NMAX               ! number of grid points

      ! determine the number of core channels
      CALL CL_INIT_CORE_CONF(PP,CHANNELS_CORE)

      CHANNELS=CHANNELS_CORE+PP%LMAX
      ALLOCATE( W(RNMAX,CHANNELS), A(RNMAX,CHANNELS), B(RNMAX,CHANNELS), & 
           N(CHANNELS), LC(CHANNELS), EIG(CHANNELS))
      ALLOCATE( POTAE_SLATER(RNMAX), POTAE_SLATER_VAL(RNMAX), POTPS_SLATER(RNMAX), &
           RHOAE(RNMAX), RHOAE_VAL(RNMAX), RHOPS(RNMAX), &
           POTAE(RNMAX), POTPS(RNMAX),TMP(RNMAX))


      IF (.NOT.LCORR) THEN
         POTAE_CORR=0
         POTPS_CORR=0
      ENDIF

!-----------------------------------------------------------------------
! determine core wavefunctions
! and set required occupancy matrix
!-----------------------------------------------------------------------
      ! core part
      CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
           W, N, LC, EIG, A_=A, B_=B)

      CALL CL_CLEAR_CORE_CONF
   
      ! add valence information
      DO CH1=1,PP%LMAX
         ! copy valence partial waves to A
         A(:,CHANNELS_CORE+CH1)=PP%WAE(:,CH1)
         LC(CHANNELS_CORE+CH1) =PP%LPS(CH1)
      END DO

      LYMAX=3    ! f orbitals are the hard coded upper limit
      LMMAX=0
      DO CH1=1,CHANNELS
         LMMAX=LMMAX+(LC(CH1)*2+1)
      ENDDO

      IF (.NOT.ASSOCIATED(COCC_ADD)) THEN
         ALLOCATE(COCC_ADD(LMMAX, LMMAX))
      ELSE IF (SIZE(COCC_ADD,1) /= LMMAX) THEN
         WRITE(0,*) 'internal error in CALC_LHF_ONE_CENTRE: SIZE(COCC_ADD) has changed',SIZE(COCC_ADD,1), LMMAX
         STOP         
      ENDIF
         
      ALLOCATE(COCC(LMMAX,LMMAX), &
           DFOCKPS(LMMAX,LMMAX), DFOCKAE(LMMAX,LMMAX), & 
           DFOCKAE_CORE(LMMAX,LMMAX), & 
           DCORRAE(LMMAX,LMMAX), DCORRPS(LMMAX,LMMAX))

      ! set diagonal components of the occupancy matrix (core only)
      COCC=0
      LMMAX_C=0
      DO CH1=1,CHANNELS_CORE
         DO M=1,LC(CH1)*2+1
            COCC(LMMAX_C+M,LMMAX_C+M)=1
         ENDDO
         LMMAX_C=LMMAX_C+(LC(CH1)*2+1)
      ENDDO

      ! valence part of occupancy matrix
      COCC(LMMAX_C+1:LMMAX,LMMAX_C+1:LMMAX)=CRHODE(1:LMMAX-LMMAX_C,1:LMMAX-LMMAX_C)
!-----------------------------------------------------------------------
! Slater potential
!-----------------------------------------------------------------------
      CALL CALC_PAW_SLATERPOT(CRHODE, PP, & 
           POTAE_SLATER, POTPS_SLATER, RHOAE, RHOPS, DFOCKAE, DFOCKPS)

      RHOAE_VAL=RHOAE
      POTAE_SLATER_VAL=POTAE_SLATER*AEXX
      CALL SIMPI(PP%R, POTAE_SLATER_VAL, HF_ENERGY)

      DFOCKAE_CORE=0
      IF (USE_CORE) THEN
         CALL CALC_PAW_CORE_SLATERPOT(COCC, PP%R, LYMAX, CHANNELS, LC, A, & 
              POTAE_SLATER, RHOAE, DFOCKAE_CORE)
         DFOCKAE=DFOCKAE_CORE(LMMAX_C+1:LMMAX,LMMAX_C+1:LMMAX)
      ENDIF

      POTAE_SLATER=POTAE_SLATER*AEXX
      POTPS_SLATER=POTPS_SLATER*AEXX

      DFOCKAE_CORE=DFOCKAE_CORE*AEXX
      DFOCKAE=DFOCKAE*AEXX
      DFOCKPS=DFOCKPS*AEXX
!-----------------------------------------------------------------------
! update correction potential
!-----------------------------------------------------------------------
      IF (PRESENT (CRHODE_ADD)) THEN
         POTPS_CORR=0
         POTAE_CORR=0

         ! integration over the energy density should yield
         ! the total energy, but the energy density is defined through
         !   e(vec r) = e(r) Y_00 (Omega_r)
         ! therefore the factor 2 sqrt(pi) = int_Omega Y_00 d Omega = 4 pi Y_00 
         CALL CALC_PAW_CORR(CRHODE_ADD, PP, POTAE_CORR, POTPS_CORR )
         ! the low level subroutines use the negative exchange potential
         ! change sign in the correction (tested against plane wave part)
         POTAE_CORR=-POTAE_CORR*2*SQRT(PI)
         POTPS_CORR=-POTPS_CORR*2*SQRT(PI)

         IF (USE_CORE) THEN
            COCC_ADD(LMMAX_C+1:LMMAX,LMMAX_C+1:LMMAX)=CRHODE_ADD(1:LMMAX-LMMAX_C,1:LMMAX-LMMAX_C)
            ! presently core and valence are not very accurately
            ! orthogonal hence the off diagonal (LHF) terms
            ! are not taken into account
            COCC_ADD(LMMAX_C+1:LMMAX,1:LMMAX_C)=0
            COCC_ADD(1:LMMAX_C,LMMAX_C+1:LMMAX)=0

            CALL SIMPI(PP%R, POTAE_CORR , SUM)

            POTAE_CORR=0
            CALL CALC_PAW_CORE_CORR(COCC_ADD, PP%R,  CHANNELS, LC, A, POTAE_CORR )
            POTAE_CORR=-POTAE_CORR*2*SQRT(PI)
            CALL SIMPI(PP%R, POTAE_CORR , POT_SHIFT_FROM_CORE )

            POT_SHIFT_FROM_CORE=POT_SHIFT_FROM_CORE-SUM
            WRITE(*,*) 'potential shift from core',POT_SHIFT_FROM_CORE*2
         ENDIF
      ENDIF

! determine slater plus correction energy density
      POTAE=POTAE_CORR +POTAE_SLATER
      POTPS=POTPS_CORR +POTPS_SLATER

      CALL SIMPI(PP%R, POTAE ,HF_ENERGY_VAL)
      CALL SIMPI(PP%R, POTPS ,HF_ENERGY_PS)

! divide energy density by charge
      POTAE=POTAE/MAX(RHOAE,MIN_CHARGE)
      IF (LCORE) THEN
         POTPS=POTPS/MAX(RHOPS+PP%RHOPS/2,MIN_CHARGE)
      ELSE
         POTPS=POTPS/MAX(RHOPS,MIN_CHARGE)
      ENDIF

! check potential after division by charge density
      SUM=HF_ENERGY_PS
      TMP=POTPS*RHOPS
      CALL SIMPI(PP%R, TMP , HF_ENERGY_PS_VAL)

      IF (LCORE) THEN
         TMP=POTPS*(RHOPS+PP%RHOPS/2)
         CALL SIMPI(PP%R, TMP ,HF_ENERGY_PS)
      ELSE
         HF_ENERGY_PS=HF_ENERGY_PS_VAL
      ENDIF

      IF (ABS(HF_ENERGY_PS-SUM)>MIN_CHARGE*10) THEN
         WRITE(*,*) 'internal error in LHF_ONE_CENTRE: potential shift caused by negative charge density',HF_ENERGY_PS,SUM
         STOP
      ENDIF

      SUM=HF_ENERGY_VAL
      TMP=POTAE*RHOAE_VAL
      CALL SIMPI(PP%R, TMP, HF_ENERGY_VAL)
      IF (.NOT.USE_CORE .AND. ABS(HF_ENERGY_VAL-SUM)>MIN_CHARGE*10) THEN
         WRITE(*,*) 'internal error in LHF_ONE_CENTRE: potential shift caused by negative charge density',HF_ENERGY_VAL,SUM
         STOP
      ENDIF
!-----------------------------------------------------------------------
! constant shift of potential v(r) to match 
!  \int v(r) rho_c(r) dr = \int e(r)/ (rho(r)+rho_c(r)) rho_c(r) dr 
! on the plane wave grid
! mind that the pseudo 
! v(r)=e(r)/rho(r) is undetermined but not e(r)/(rho(r)+rho_c(r))
!-----------------------------------------------------------------------
      IF (LCORE) THEN
         ! total core charge
         CALL SIMPI(PP%R, PP%RHOPS/2 , SUM)
         SUM=SUM*2*SQRT(PI)
         ! core charge 0, bypass potential shifts for this atom
         IF (SUM/=0) THEN
            TMP=PP%RHOPS/2*POTPS
            CALL SIMPI(PP%R, TMP , RHO_CORE)
            RHO_CORE=RHO_CORE
#ifdef debug
            WRITE(*,*) 'correction strength on radial and pw grid',RHO_CORE,D_CORE,SUM
#endif
            RHO_CORE=-RHO_CORE-D_CORE

            ! determine how a constant shift of the energy density will affect
            ! the previous integral
            TMP=2*SQRT(PI)*RHOPS/MAX(RHOPS+PP%RHOPS/2,MIN_CHARGE)*PP%RHOPS/2
            CALL SIMPI(PP%R, TMP , SUM)
            RHO_CORE=RHO_CORE/SUM
            
            ! constant shift of energy density such that radial grid
            ! reproduces pw grid
            POTPS=RHO_CORE*2*SQRT(PI)*RHOPS/MAX(RHOPS+PP%RHOPS/2,MIN_CHARGE)+POTPS
! old version shift AE potential accordingly
            POTAE=RHO_CORE*2*SQRT(PI)+POTAE
            POTAE_SLATER_VAL=RHO_CORE*2*SQRT(PI)*RHOAE_VAL+POTAE_SLATER_VAL
            CALL SIMPI(PP%R, POTAE_SLATER_VAL, HF_ENERGY)
            
            TMP=PP%RHOPS/2*POTPS
            CALL SIMPI(PP%R, TMP , SUM)
            IF (ABS(-SUM-REAL(D_CORE,q))>=1E-6) THEN
               WRITE(*,*) 'correction on strength radial grid not correct',SUM,REAL(D_CORE,q)
               STOP
            ENDIF
            ! new version finally shift the potential back 
            ! to match AE potential
!            POTPS=-RHO_CORE*2*SQRT(PI)*RHOPS/MAX(RHOPS+PP%RHOPS/2,MIN_CHARGE)+POTPS
         ENDIF
      ENDIF

! check potential after shift
      SUM=HF_ENERGY_VAL-HF_ENERGY_PS

      TMP=POTAE*RHOAE_VAL
      CALL SIMPI(PP%R, TMP ,HF_ENERGY_VAL)

      TMP=POTPS*RHOPS
      CALL SIMPI(PP%R, TMP , HF_ENERGY_PS_VAL)

      IF (LCORE) THEN
         TMP=POTPS*(RHOPS+PP%RHOPS/2)
         CALL SIMPI(PP%R, TMP ,HF_ENERGY_PS)
      ELSE
         HF_ENERGY_PS=HF_ENERGY_PS_VAL
      ENDIF

      IF (ABS(HF_ENERGY_VAL-HF_ENERGY_PS-SUM)>MIN_CHARGE*10) THEN
         WRITE(*,*) 'internal error in LHF_ONE_CENTRE: potential shift caused change in Hartree-Fock energy or the charge density is negative', &
              HF_ENERGY_VAL-HF_ENERGY_PS,SUM
         STOP
      ENDIF
!-----------------------------------------------------------------------
! debugging and finalisation
!-----------------------------------------------------------------------
#ifdef debug
      WRITE(*,'(A,3F14.7)') 'KLI energy valence AE',HF_ENERGY_VAL
      CALL  KLI_ENERGY_OCC_STRENGH('Slater energy valence AE',LMMAX-LMMAX_C, DFOCKAE, CRHODE)
      WRITE(*,*) ' radius, charge, potential valence AE'
      DO M= PP%R%NMAX-80, PP%R%NMAX, 10
        WRITE(*,'(I4,4F14.7)') M,PP%R%R(M),RHOAE(M)*2,RHOAE(M)*SQRT(3.1415*4)/PP%R%R(M)**2,POTAE(M)/SQRT(3.1415*4)
     ENDDO


      WRITE(*,'(A,3F14.7)') 'KLI valence PS energy',HF_ENERGY_PS,HF_ENERGY_PS_VAL
      CALL  KLI_ENERGY_OCC_STRENGH('Slater energy valence PS',LMMAX-LMMAX_C, DFOCKPS, CRHODE)
!     WRITE(*,*) ' radius, charge, potential valence PS'
!     DO M= PP%R%NMAX-80, PP%R%NMAX, 10
!        WRITE(*,'(I4,4F14.7)') M,PP%R%R(M),RHOPS(M)*2,RHOPS(M)*SQRT(3.1415*4)/PP%R%R(M)**2,POTPS(M)/SQRT(3.1415*4)
!     ENDDO
      WRITE(*,'(A,3F14.7)') 'KLI valence AE-PS energy',-HF_ENERGY_PS/2 +HF_ENERGY_VAL/2
#endif
      CALL CALC_PAW_LOCAL_CORR( DCORRAE, DCORRPS, PP, POTAE, POTPS )

      ! there is a sign change here, compared to what one whould expect
      ! because the pawlhf low level routines process the negative exchange potentials
      DFOCK_LOCAL=DFOCKPS-DCORRPS-  (DFOCKAE-DCORRAE)
      DLOCAL     =DCORRPS-           DCORRAE
      
      ! now determine the update occupancies for the core states
      COCC_ADD=0
      CALL CALC_PAW_CORE_LOCAL_CORR( COCC_ADD, PP%R,  CHANNELS, LC, A, POTAE )
      COCC_ADD=-COCC_ADD+DFOCKAE_CORE

      SUM=0
      DO LM1=1,LMMAX
         DO LM2=1,LMMAX
            SUM=SUM+ &
                 COCC_ADD(LM1,LM2)*COCC(LM1,LM2)
         ENDDO
      ENDDO
!      WRITE(*,*) 'sum of diagonal terms',SUM
#ifdef debug
      CALL DUMP_DLLMM( "ONE-CENTRE-CORRECTION local AE",DCORRAE, PP)
      CALL DUMP_DLLMM( "ONE-CENTRE-CORRECTION AE",DFOCKAE, PP)
#endif
! double counting correction
      DOUBLEC_HF =(-HF_ENERGY_PS_VAL+HF_ENERGY_PS/2) +HF_ENERGY_VAL/2
!      WRITE(*,*) 'old double counting version', DOUBLEC_HF
!      DOUBLEC_HF =(-HF_ENERGY_PS_VAL+HF_ENERGY_PS/2) + HF_ENERGY_VAL - HF_ENERGY/2
!      WRITE(*,*) HF_ENERGY , HF_ENERGY_VAL
!      WRITE(*,*) 'new double counting version', DOUBLEC_HF
      IF (PRESENT(POTPS_FINAL)) POTPS_FINAL=POTPS
      IF (PRESENT(POTAE_FINAL)) POTAE_FINAL=POTAE

! quick return
      DEALLOCATE(COCC, DFOCKPS, DFOCKAE, DFOCKAE_CORE, DCORRAE, DCORRPS)
      DEALLOCATE(W, A, B, N, LC, EIG)
      DEALLOCATE(POTAE_SLATER, POTAE_SLATER_VAL, POTPS_SLATER, RHOAE, RHOAE_VAL, RHOPS, POTPS, POTAE, TMP)

      RETURN

    END SUBROUTINE CALC_LHF_ONE_CENTRE


! **********************************************************************************
!   CALC_OEP_ONE_CENTRE
!
! calculate the OEP one centre correction terms 
! the one centre occupancies are supplied in CRHODE_ADD
! from those the one centre charge density is calculated and this is projected
! onto the local basis functions
! the strength parameters are returned in DLOCAL
!
! **********************************************************************************


    SUBROUTINE CALC_OEP_ONE_CENTRE(NT, PP, CRHODE_ADD, DLOCAL)
      USE cl
      USE pawfock_inter
      IMPLICIT NONE

      INTEGER NT
      TYPE (potcar), POINTER ::  PP
      OVERLAP :: CRHODE_ADD(:,:), DLOCAL(:,:)
    !local
      OVERLAP :: DCORRAE(SIZE(DLOCAL,1),SIZE(DLOCAL,1)), DCORRPS(SIZE(DLOCAL,1),SIZE(DLOCAL,1))
      INTEGER :: RNMAX
      REAL(q),ALLOCATABLE :: POTAE(:), POTPS(:)
      INTEGER I
      REAL(q) :: AEXX
      ! number of grid points

      CALL HFPARAMETERS(AEXX)
      RNMAX =PP%R%NMAX

      ! determine the number of core channels
      ALLOCATE( POTAE(RNMAX), POTPS(RNMAX))

      POTPS=0
      POTAE=0

      CALL CALC_PAW_CORR(CRHODE_ADD, PP, POTAE, POTPS )
      POTAE=-POTAE*2*SQRT(PI)
      POTPS=-POTPS*2*SQRT(PI)

      CALL CALC_PAW_LOCAL_CORR( DCORRAE, DCORRPS, PP, POTAE, POTPS )
      ! there is a sign change here, compared to what one whould expect
      ! because the low level routines process the negative exchange potentials
      DLOCAL     =DCORRPS-           DCORRAE

#ifdef debug
      CALL DUMP_DLLMM( "ONE-CENTRE-CORRECTION local AE",DCORRAE, PP)
      CALL DUMP_DLLMM( "ONE-CENTRE-CORRECTION AE",DFOCKAE, PP)
#endif
      DEALLOCATE(POTAE, POTPS)

    END SUBROUTINE CALC_OEP_ONE_CENTRE

!*******************************************************************
!
! subroutine to determine to local magnetisation axis inside
! one PAW sphere
! the supplied occupancy matrix must be in the (total,magnetisation)
! representation
!
!*******************************************************************

    SUBROUTINE MAG_DIRECTION( PP, CRHODE, MAG )
      USE asa
      IMPLICIT NONE

      OVERLAP :: CRHODE(:,:,:)
      REAL(q) :: MAG(:)
      TYPE (potcar),POINTER:: PP
    ! local
      INTEGER :: I, J, CHANNELS, MI, MJ, LMI, LMJ, LI, LJ, LMIND1, ISTART1, IEND1, IC1, ISP
      REAL(q) :: ABS_MAG

      IF (SIZE(MAG) /= SIZE(CRHODE,3)) THEN
         WRITE(0,*)'internal error in MAG_DIRECTION: size of MAG and CRHODE do not match',SIZE(MAG),SIZE(CRHODE,3)
         STOP
      ENDIF


      MAG=0

      CHANNELS=PP%LMAX
      LMI=1
      DO I=1,CHANNELS
      LMJ=1
      DO J=1,CHANNELS
            
         LI = PP%LPS(I)
         LJ = PP%LPS(J)

         CALL YLM3LOOKUP(LI,LJ,LMIND1)

         DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)
               DO IC1=ISTART1,IEND1-1  ! IC1 compound index for LM
                  IF ( JS(IC1)==1) THEN
                     DO ISP=1,SIZE(CRHODE,3)
                        MAG(ISP) = MAG(ISP)+CRHODE(LMI+MI-1,LMJ+MJ-1,ISP)*YLM3(IC1)*PP%QTOT(I,J)
                     ENDDO
                  ENDIF
               ENDDO
            ENDDO
         ENDDO
      LMJ=LMJ+2*LJ+1
      ENDDO ! J

      LMI=LMI+2*LI+1
      ENDDO ! I

!      WRITE(*,'(A,4F14.7)') 'local charge and magnetisation direction is',MAG
      ABS_MAG=SQRT(MAG(2)*MAG(2)+MAG(3)*MAG(3)+MAG(4)*MAG(4))
      ! magnetisation axis
      IF (ABS_MAG/=0) THEN
         MAG(2)=MAG(2)/ABS_MAG
         MAG(3)=MAG(3)/ABS_MAG
         MAG(4)=MAG(4)/ABS_MAG
      ELSE
         ! default magnetisation axis along z
         MAG(2)=0
         MAG(3)=0
         MAG(4)=1
      ENDIF

    END SUBROUTINE MAG_DIRECTION

!*******************************************************************
!
! subroutine to determines the up and down occupancies from
! the occupancies in the presentation (total,magnetisation)
!
!*******************************************************************

    SUBROUTINE UP_AND_DOWN_OCCUPANCIES( PP, CRHODE, MAG, CRHODE_UP_DOWN)
      USE asa
      USE pawfock_inter
      IMPLICIT NONE

      OVERLAP :: CRHODE(:,:,:)
      OVERLAP :: CRHODE_UP_DOWN(:,:,:)
      REAL(q) :: MAG(:)
      TYPE (potcar),POINTER:: PP
    ! local
      INTEGER ISP

      IF (SIZE(CRHODE,3)/=4 .OR. SIZE(CRHODE_UP_DOWN,3)/=2) THEN
         WRITE(0,*)'internal error in MAG_DIRECTION: size of CRHODE is inappropriate',SIZE(CRHODE,3),SIZE(CRHODE_UP_DOWN,3)
         STOP
      ENDIF
      IF (SIZE(CRHODE,1)/= SIZE(CRHODE_UP_DOWN,1)) THEN
         WRITE(0,*)'internal error in MAG_DIRECTION: size of CRHODE and CRHODE_UP_DOWN do not match',SIZE(CRHODE,1),SIZE(CRHODE_UP_DOWN,1)
         STOP
      ENDIF

      ! total, magnetisation
      CRHODE_UP_DOWN(:,:,1)=REAL(CRHODE(:,:,1),q)
      CRHODE_UP_DOWN(:,:,2)=0
         
      DO ISP=2,4
         CRHODE_UP_DOWN(:,:,2)=CRHODE_UP_DOWN(:,:,2)+REAL(CRHODE(:,:,ISP),q)*MAG(ISP)
      ENDDO

      ! up and down occupancies
      CALL OCC_FLIP2(CRHODE_UP_DOWN,SIZE(CRHODE_UP_DOWN,1))

    END SUBROUTINE UP_AND_DOWN_OCCUPANCIES

!*******************************************************************
!
! subroutine to determines the PP strenght matrix
! from the up and down strenght matrices
!
!*******************************************************************

    SUBROUTINE FULLMATRIX_FROM_UP_DOWN( PP, CDIJ_UP_DOWN , CDIJ , MAG)
      USE asa
      IMPLICIT NONE

      OVERLAP :: CDIJ(:,:,:)
      OVERLAP :: CDIJ_UP_DOWN(:,:,:)
      REAL(q) :: MAG(:)
      TYPE (potcar),POINTER:: PP
    ! local
      INTEGER ISP, L, LP
      REAL(q) CQU, CQD
      OVERLAP CDIJ_MAG

      IF (SIZE(CDIJ,3)/=4 .OR. SIZE(CDIJ_UP_DOWN,3)/=2) THEN
         WRITE(0,*)'internal error in  FULLMATRIX_FROM_UP_DOWN: size of CDIJ is inappropriate',SIZE(CDIJ,3),SIZE(CDIJ_UP_DOWN,3)
         STOP
      ENDIF
      IF (SIZE(CDIJ,1)/= SIZE(CDIJ_UP_DOWN,1)) THEN
         WRITE(0,*)'internal error in  FULLMATRIX_FROM_UP_DOWN: size of CDIJ and CDIJ_UP_DOWN do not match',SIZE(CDIJ,1),SIZE(CDIJ_UP_DOWN,1)
         STOP
      ENDIF

      ! transform to total and magnetisation strenght
      DO L=1,PP%LMMAX
         DO LP=1,PP%LMMAX
            CQU=CDIJ_UP_DOWN(L,LP,1)
            CQD=CDIJ_UP_DOWN(L,LP,2)
            CDIJ(L,LP,1)=(CQU+CQD)
            CDIJ_MAG=(CQU-CQD)
            CDIJ(L,LP,2)=MAG(2)*CDIJ_MAG
            CDIJ(L,LP,3)=MAG(3)*CDIJ_MAG
            CDIJ(L,LP,4)=MAG(4)*CDIJ_MAG
         ENDDO
      ENDDO

    END SUBROUTINE FULLMATRIX_FROM_UP_DOWN

!*******************************************************************
!
! subroutine to determine the up and down on site correction
! it takes a 4 component array in spinor representation
! transforms it to total, magnetisation and projects it onto
! the local magnetisation axis and returns the up and down corrections
!
!*******************************************************************

    SUBROUTINE UP_DOWN_CORRECTION(D_CORE,D_CORE_UP_DOWN, MAG)
      IMPLICIT NONE

      OVERLAP :: D_CORE(:)
      OVERLAP :: D_CORE_UP_DOWN(:)
      REAL(q) :: MAG(:)
      ! local
      REAL(q) :: D(4),ABS_MAG
      OVERLAP C00,C01,C10,C11


      ! essentially same transformation as in US_FLIP(....,.FALSE.)
      C00=D_CORE(1)
      C01=D_CORE(2)
      C10=D_CORE(3)
      C11=D_CORE(4)
      ! calculate total and magnetisation
      D(1)= C00+C11
      D(2)= C01+C10
      D(3)=(C01-C10)*(0._q,1._q)
      D(4)= C00-C11
      ! project onto local axis
      ABS_MAG=MAG(2)*D(2)+MAG(3)*D(3)+MAG(4)*D(4)

      ! why a factor 2??
      D_CORE_UP_DOWN(1)=(D(1)+ABS_MAG)/2
      D_CORE_UP_DOWN(2)=(D(1)-ABS_MAG)/2

!      WRITE(*,'(A,2F14.7)') 'dcore is',REAL(D_CORE_UP_DOWN,q)

    END SUBROUTINE UP_DOWN_CORRECTION

!*******************************************************************
!
! this is the central routine of the PAW LHF routine
! if CRHODE_ADD is supplied it updates the correction potential
! and afterwards calculates the new interaction strenght
!  upon return the routine sets
!   CDIJC        D_ij= - <phi^ps_i| V^ps_x + V^ps_local | phi^ps_j> +
!                        <phi_i   | V_x + V_local | phi_j> 
!   CDIJ         D_ij= - <phi^ps_i| V^ps_local | phi^ps_j> +
!                        <phi_i   | V_local | phi_j> 
!   DOUBLEC_HF   double counting corrections
!
! for the collinear case the routine operates seperately on the
! up and down spin, whereas for the non collinear case a transformation
! to total magnetisation is performed, and the final result is
! stored back in a spinor representation
! in any case the routine expects quantities in a spinor representation
! as input (and returns spinor like quantities as output)
!
!*******************************************************************

    SUBROUTINE SET_DD_LHF(WDES, P , T_INFO, LOVERL, LCORR, LCORE, LMDIM, & 
         CDIJC, CDIJ, D_CORE, CRHODE, DOUBLEC_HF, CRHODE_ADD )
      USE asa
      USE poscar
      USE wave
      USE radial
      USE base
      USE setexm
      IMPLICIT NONE

      TYPE (wavedes)  WDES
      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      LOGICAL  LOVERL
      TYPE (energy)   E
      LOGICAL LCORR             ! corrections terms to potential are initialised
      LOGICAL LCORE             ! partial core 
      INTEGER LMDIM
      OVERLAP  CDIJC(:,:,:,:)
      OVERLAP  CDIJ(:,:,:,:)
      OVERLAP  D_CORE(:,:)
      OVERLAP  CRHODE(:,:,:,:)  ! onsite occupancies               
      OVERLAP, OPTIONAL :: CRHODE_ADD(:,:,:,:)
      REAL(q) :: DOUBLEC_HF
    ! local variables
      INTEGER NT,NI,NDIM,NIP,ISP,IBASE,K,LMAX,NIONS_LOCAL, NI_L
      TYPE (potcar), POINTER ::  PP
! variables required to store core wavefunctions
      OVERLAP :: CRHODE_UP_DOWN(LMDIM,LMDIM,2),CRHODE_ADD_UP_DOWN(LMDIM,LMDIM,2), &
           CDIJC_UP_DOWN(LMDIM,LMDIM,2), CDIJ_UP_DOWN(LMDIM,LMDIM,2)
      OVERLAP :: D_CORE_UP_DOWN(2)
      REAL(q) :: MAG(4)
      REAL(q) :: DOUBLEC
      REAL(q) :: SCALE_DENSITY
!=======================================================================
! quick return and allocation of work space
!=======================================================================
      DOUBLEC_HF=0
! no pseudopotentials with overlap
      IF (.NOT.LOVERL) RETURN

      CDIJC=0
      CDIJ=0
! mimic US-PP just return
      IF (MIMIC_US) RETURN
! determine maximal radial grid points
      NDIM=0
      DO NT=1,T_INFO%NTYP
         IF (ASSOCIATED(P(NT)%QPAW)) THEN
            NDIM=MAX(NDIM, P(NT)%R%NMAX)
         END IF
      ENDDO
      IF (NDIM == 0) RETURN

! allocate required arrays 
      IF (NDIM_POT/=NDIM) THEN
         IF (NDIM_POT/=0) THEN
            DO NI=1,NIONS_LOCAL
               DO ISP=1,WDES%NCDIJ
                  DEALLOCATE(COCC_ADD(NI,ISP)%QOC)
               ENDDO
            ENDDO
            DEALLOCATE(POTAE_CORR, POTPS_CORR,COCC_ADD)
         ENDIF
         NDIM_POT=NDIM
         NIONS_LOCAL=0
         DO NI=1,T_INFO%NIONS
            IF (DO_LOCAL(NI)) NIONS_LOCAL=NIONS_LOCAL+1
         ENDDO
         ALLOCATE(POTAE_CORR(NDIM, NIONS_LOCAL,WDES%NCDIJ), &
                  POTPS_CORR(NDIM, NIONS_LOCAL,WDES%NCDIJ), &
                  COCC_ADD(NIONS_LOCAL, WDES%NCDIJ))
         DO NI=1,NIONS_LOCAL
            DO ISP=1,WDES%NCDIJ
               NULLIFY(COCC_ADD(NI,ISP)%QOC)
            ENDDO
         ENDDO
         POTAE_CORR=0
         POTPS_CORR=0
      ENDIF

      IF (WDES%NCDIJ==1) THEN
         SCALE_DENSITY=2   ! need to scale total density by 1/2 to get up 
      ELSE
         SCALE_DENSITY=1
      ENDIF

      IF (WDES%LNONCOLLINEAR) THEN
         ! go to magnetisation presentation
         CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .FALSE.) 
      ENDIF
!=======================================================================
! cycle all ions and add corrections to pseudopotential strength CDIJ
!=======================================================================
      IBASE=1
      
      NI_L=0
      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)
      ! if this element is not treated locally CYCLE
         IF (DO_LOCAL(NI)) THEN
            NI_L=NI_L+1
            PP=> PP_POINTER(P, NI, NT)
            IF (WDES%LNONCOLLINEAR) THEN
               CALL MAG_DIRECTION( PP, CRHODE(:,:,NIP,:), MAG )
               CALL UP_AND_DOWN_OCCUPANCIES( PP, CRHODE(:,:,NIP,:), MAG, CRHODE_UP_DOWN)
               IF (PRESENT(CRHODE_ADD)) &
                  CALL UP_AND_DOWN_OCCUPANCIES( PP, CRHODE_ADD(:,:,NIP,:), MAG, CRHODE_ADD_UP_DOWN)
               CALL UP_DOWN_CORRECTION(D_CORE(NI,:),D_CORE_UP_DOWN, MAG)

               DO ISP=1,2
                  IF (PRESENT(CRHODE_ADD)) THEN
                     CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE_UP_DOWN(:,:,ISP), &
                     CDIJC_UP_DOWN(:,:,ISP), CDIJ_UP_DOWN(:,:,ISP), D_CORE_UP_DOWN(ISP), &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     LCORR, LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC ,  &
                     CRHODE_ADD_UP_DOWN(:,:,ISP))
                  ELSE
                     CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE_UP_DOWN(:,:,ISP), &
                     CDIJC_UP_DOWN(:,:,ISP), CDIJ_UP_DOWN(:,:,ISP), D_CORE_UP_DOWN(ISP),  &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     LCORR, LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC )
                  ENDIF
                  DOUBLEC_HF=DOUBLEC_HF+DOUBLEC*SCALE_DENSITY
               ENDDO

               CALL FULLMATRIX_FROM_UP_DOWN( PP, CDIJ_UP_DOWN , CDIJ(:,:,NIP,:) , MAG)
               CALL FULLMATRIX_FROM_UP_DOWN( PP, CDIJC_UP_DOWN, CDIJC(:,:,NIP,:), MAG)

            ELSE
               DO ISP=1,WDES%NCDIJ
                  IF (PRESENT(CRHODE_ADD)) THEN
                     CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE(:,:,NIP,ISP)*(1._q/SCALE_DENSITY), &
                     CDIJC(:,:,NIP,ISP), CDIJ(:,:,NIP,ISP), D_CORE(NI,ISP), &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     LCORR, LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC ,  &
                     CRHODE_ADD(:,:,NIP,ISP)*(1._q/SCALE_DENSITY))
                  ELSE
                     CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE(:,:,NIP,ISP)*(1._q/SCALE_DENSITY), &
                     CDIJC(:,:,NIP,ISP), CDIJ(:,:,NIP,ISP), D_CORE(NI,ISP),  &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     LCORR, LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC )
                  ENDIF
                  DOUBLEC_HF=DOUBLEC_HF+DOUBLEC*SCALE_DENSITY
               ENDDO
            END IF
         END IF
      ENDDO ion
!=======================================================================
! now distribute the DIJ to all nodes which hold DIJ (using global sum)
!=======================================================================
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJC,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
#else
      CALLMPI( M_sum_z(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
      CALLMPI( M_sum_z(WDES%COMM_INTER, CDIJC,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
#endif
!      DO ISP=1,WDES%NCDIJ
!         CALL DUMP_DLLMM( "FOCK-LOCAL",CDIJC(:,:,1,ISP), P(1))
!      ENDDO


      IF (WDES%LNONCOLLINEAR) THEN
         ! go to spinor presentation
         CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .TRUE.) 
         CALL US_FLIP(WDES, LMDIM, CDIJ, LOVERL, .TRUE.) 
         CALL US_FLIP(WDES, LMDIM, CDIJC, LOVERL, .TRUE.)          
      ENDIF

!      DO ISP=1,WDES%NCDIJ
!         CALL DUMP_DLLMM( "FOCK-LOCAL",CDIJC(:,:,1,ISP), P(1))
!      ENDDO
!      DO ISP=1,WDES%NCDIJ
!         CALL DUMP_DLLMM( "FOCK-LOCAL",CRHODE(:,:,1,ISP), P(1))
!      ENDDO

      CALLMPI( M_sum_d(WDES%COMM, DOUBLEC_HF, 1))

    END SUBROUTINE SET_DD_LHF



!*******************************************************************
!
! this routine allows to switch from KLI to EXX-OEP
! this requires to calculate the Slater potential and to add
! the potential to the correction potential
! from then on POTPS_CORR and POTAE_CORR hold the entire
! local one centre potentials instead of the corrections only
!
!*******************************************************************

    SUBROUTINE LHF_TO_OEP(WDES, P , T_INFO, LOVERL, LCORE, LMDIM, &
      D_CORE, CRHODE )
      USE asa
      USE poscar
      USE wave
      USE radial
      USE base
      USE setexm
      IMPLICIT NONE

      TYPE (wavedes)  WDES
      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      LOGICAL  LOVERL
      TYPE (energy)   E
      LOGICAL LCORE   ! partial core 
      INTEGER LMDIM
      OVERLAP  D_CORE(:,:)
      OVERLAP  CRHODE(:,:,:,:)
    ! local variables
      INTEGER NT,NI,NDIM,NIP,ISP,IBASE,K,LMAX,NIONS_LOCAL, NI_L
      TYPE (potcar), POINTER ::  PP
! variables required to store core wavefunctions
      OVERLAP :: CRHODE_UP_DOWN(LMDIM,LMDIM,2)
      OVERLAP :: CDIJ(LMDIM,LMDIM), CDIJC(LMDIM, LMDIM)
      OVERLAP :: D_CORE_UP_DOWN(2)
      REAL(q) :: MAG(4)
      REAL(q) :: SCALE_DENSITY
      REAL(q) :: DOUBLEC

! no pseudopotentials with overlap quick return
      IF (.NOT.LOVERL .OR. MIMIC_US) RETURN

      IF (WDES%NCDIJ==1) THEN
         SCALE_DENSITY=2   ! need to scale total density by 1/2 to get up 
      ELSE
         SCALE_DENSITY=1
      ENDIF

      IF (WDES%LNONCOLLINEAR) THEN
         ! go to magnetisation presentation
         CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .FALSE.) 
      ENDIF
!=======================================================================
! cycle all ions and add Slater potential to that ion
!=======================================================================
      IBASE=1
      
      NI_L=0
      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)
      ! if this element is not treated locally CYCLE
         IF (DO_LOCAL(NI)) THEN
            NI_L=NI_L+1
            PP=> PP_POINTER(P, NI, NT)
            IF (WDES%LNONCOLLINEAR) THEN
               CALL MAG_DIRECTION( PP, CRHODE(:,:,NIP,:), MAG )
               CALL UP_AND_DOWN_OCCUPANCIES( PP, CRHODE(:,:,NIP,:), MAG, CRHODE_UP_DOWN)
               CALL UP_DOWN_CORRECTION(D_CORE(NI,:),D_CORE_UP_DOWN, MAG)

               DO ISP=1,2
                  CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE_UP_DOWN(:,:,ISP), &
                     CDIJC, CDIJ, D_CORE_UP_DOWN(ISP),  &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     .FALSE., LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC , &
                     POTPS_FINAL=POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), &
                     POTAE_FINAL=POTAE_CORR(1:PP%R%NMAX,NI_L,ISP))
               ENDDO
            ELSE
               DO ISP=1,WDES%NCDIJ
                  CALL CALC_LHF_ONE_CENTRE( NT, PP, CRHODE(:,:,NIP,ISP)*(1._q/SCALE_DENSITY), &
                     CDIJC, CDIJ, D_CORE(NI,ISP),  &
                     POTAE_CORR(1:PP%R%NMAX,NI_L,ISP), POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), & 
                     .FALSE., LCORE, DOUBLEC, COCC_ADD(NI_L,ISP)%QOC , &
                     POTPS_FINAL=POTPS_CORR(1:PP%R%NMAX,NI_L,ISP), &
                     POTAE_FINAL=POTAE_CORR(1:PP%R%NMAX,NI_L,ISP))
               ENDDO
            END IF
         END IF
      ENDDO ion

      IF (WDES%LNONCOLLINEAR) THEN
         ! go to spinor presentation
         CALL US_FLIP(WDES, LMDIM, CRHODE, LOVERL, .TRUE.) 
      ENDIF


    END SUBROUTINE LHF_TO_OEP


!*******************************************************************
!
! this routine calculates the one center corrections
! to the one-center strength parameters
! the strength parameters CDIJ must be and are returned in the 
! spinor respresentation
!   e.g. spin up, spin down in collinear case
! the occupancies CRHODE must be in the 
!    (total, magnetization (x,y,z)) representation
!
! unfortunately restoring the exact all electron charge density
! does not seem to work, if one restricts the space to valence
! orbitals only. The resulting potential is most likely unphysical
! in the area of the core states, if the core states are not
! included
! What seem to work, is to just accurately restore the "shape"
! of the all electron charge density using LMAXFOCKAE 
! (without actually restoring its nodal structure)
! at least this yields stable results
! this option is selected using EXXOEP=3
! (local LPS_ONE_CENTER=.TRUE.)
!
!*******************************************************************

    SUBROUTINE SET_DD_OEP(WDES, P , T_INFO, LOVERL, &
         ISPIN, LMDIM, CDIJ, CRHODE , LPS_ONE_CENTER )
      USE asa
      USE poscar
      USE wave
      USE radial
      USE base
      IMPLICIT NONE

      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      TYPE (wavedes)  WDES
      INTEGER LMDIM, ISPIN
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      LOGICAL  LOVERL
      LOGICAL  LPS_ONE_CENTER
    ! local variables
      TYPE (potcar),POINTER:: PP
      INTEGER NT,LYMAX,NI,NDIM,LMMAX,NIP,ISP,K,NCDIJ
      INTEGER, EXTERNAL :: MAXL_AUG,MAXL1
      REAL(q) DDLM(LMDIM*LMDIM), RHOLM_(LMDIM*LMDIM,WDES%NCDIJ)
      OVERLAP CTMP(LMDIM,LMDIM,MAX(2,WDES%NCDIJ))
      
      REAL(q),ALLOCATABLE :: POT(:,:,:), RHO(:,:,:), POTAE(:,:,:), RHOAE(:,:,:)
      REAL(q),ALLOCATABLE :: RHOCOL(:,:,:)
      REAL(q) :: ALPHA,BETA
      ! core level shifts
      INTEGER II,RNMAX

!=======================================================================
! quick return and allocation of work space
!=======================================================================
      IF (.NOT.LOVERL .OR. MIMIC_US) THEN
         RETURN
      ENDIF

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)

      NDIM=0
      DO NT=1,T_INFO%NTYP
         IF (ASSOCIATED(P(NT)%QPAW)) THEN
            NDIM=MAX(NDIM, P(NT)%R%NMAX)
         END IF
      ENDDO

      IF (NDIM == 0) RETURN

      LMMAX=(LYMAX+1)**2
      NCDIJ = WDES%NCDIJ
      ALLOCATE ( POT( NDIM, LMMAX, NCDIJ ), RHO( NDIM, LMMAX, NCDIJ ), &
           POTAE( NDIM, LMMAX, NCDIJ ), RHOAE( NDIM, LMMAX, NCDIJ))
      ALLOCATE (RHOCOL( NDIM, LMMAX, NCDIJ ))
      
! for spin orbit coupling set the euler angles
      IF ( WDES%LSORBIT ) &
           CALL EULER(WDES%SAXIS, ALPHA, BETA)
!=======================================================================
! cycle all ions and add corrections to pseudopotential strength CDIJ
!=======================================================================
      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)
      ! if this element is not treated locally CYCLE
         IF (.NOT. DO_LOCAL(NI)) THEN
            ! for PAW, set CDIJ to zero if it resides on local node
            ! and if the element is not treated locally
            ! (required since global sum is performed)
            IF (ASSOCIATED(P(NT)%QPAW)) THEN
               IF (NIP /= 0) THEN
                  DO ISP=1,NCDIJ
                     CDIJ(:,:,NIP,ISP)=0
                  ENDDO
               ENDIF
            ELSE
            ! US PP: initialize to zero if we are not on first node in COMM_INTER
            ! (at the end, we use a global sum over COMM_INTER) 
#ifdef MPI
               IF (WDES%COMM_INTER%NODE_ME /=1 .AND. NIP /=0 ) THEN
                  DO ISP=1,NCDIJ
                     CDIJ(:,:,NIP,ISP)=0
                  ENDDO
               ENDIF
#endif
            ENDIF
            CYCLE ion
         ENDIF

         PP=> PP_POINTER(P, NI, NT)
         LYMAX =MAXL1(PP)*2
         RNMAX =PP%R%NMAX
  !-----------------------------------------------------------------------
  ! first set RHOLM (i.e. the on site occupancy matrix)
  ! and then the lm dependent charge densities RHO and RHOAE
  ! (excluding augmentation charges yet)
  !-----------------------------------------------------------------------
         RHOLM_=0
  !      WRITE(*,'("RHOMIX",6F10.6)') RHOLM_STORE

         DO ISP=1,NCDIJ
    ! retrieve the one center on site charge densities to RHOLM_
            CALL TRANS_RHOLM( CRHODE(:,:,NIP,ISP), RHOLM_(:,ISP), PP )

    ! calculate the total radial angular decomposed charge distributions
            LMMAX=(LYMAX+1)**2
            RHOAE(:,:,ISP)=0; RHO(:,:,ISP)=0
            CALL RAD_CHARGE( RHOAE(:,:,ISP), PP%R,RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WAE )
            CALL RAD_CHARGE( RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WPS )
         ENDDO
  !-----------------------------------------------------------------------
  ! add augmentation charges now
  !-----------------------------------------------------------------------
         DO ISP=1,NCDIJ
            CALL RAD_AUG_CHARGE(  RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS,  &
                  LYMAX, PP%AUG_SOFT, PP%QPAW )
!            CALL RAD_INT( PP%R,  LYMAX, RHO(:,:,ISP), RHOAE(:,:,ISP) )
         ENDDO

         ! overwrite the AE charge by the pseudo charge and make it spatially more accurate
         ! using QPAW_FOCK
         IF (LPS_ONE_CENTER) THEN
            DO ISP=1,NCDIJ
               RHOAE=RHO
               CALL RAD_AUG_CHARGE_FOCK(  RHOAE(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS,  &
               LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK )
            ENDDO
         ENDIF
  !-----------------------------------------------------------------------
  ! calculate the local radial potential
  ! mind in the non-collinear case the potential V(r) = d E(r) / d rho (r)
  ! and the potential vec mu(r) = d E(r) / d vec m (r) are stored in
  ! POT and POTAE (potentials need to be real), whereas 
  ! in the collinear case the spin up and down potentials
  ! are stored in POT and POTAE
  ! probably one should rewrite this in the collinear case
  !-----------------------------------------------------------------------
         IF ( WDES%LNONCOLLINEAR ) THEN
            CALL RAD_MAG_DENSITY( RHO, RHOCOL, LYMAX, PP%R)
            CALL RAD_OEP_POT( PP%R, 2, PP%LMAX_CALC,   &
                 RHOCOL, POT )
            CALL RAD_MAG_DIRECTION( RHO, RHOCOL, POT, LYMAX, PP%R)

            CALL RAD_MAG_DENSITY( RHOAE, RHOCOL, LYMAX, PP%R)
            CALL RAD_OEP_POT( PP%R, 2, PP%LMAX_CALC,  &
                 RHOCOL, POTAE )
            CALL RAD_MAG_DIRECTION( RHOAE, RHOCOL, POTAE, LYMAX, PP%R)

         ELSE
            RHOCOL=RHO
            CALL RAD_OEP_POT( PP%R, ISPIN, PP%LMAX_CALC, &
                 RHOCOL, POT )

            RHOCOL=RHOAE
            CALL RAD_OEP_POT( PP%R, ISPIN, PP%LMAX_CALC, &
                 RHOCOL, POTAE )

         ENDIF
  !-----------------------------------------------------------------------
  ! calculate the PAW correction terms to the pseudopotential strength D
  ! I have defined the PAW contribution in a way that in the limit of
  ! atomic occupancies no contributions are added
  !-----------------------------------------------------------------------
         ! multiply potentials by simpson weights
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POTAE)
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POT)
         CTMP=0

         POTAE=-POTAE ! this is stupid, since RAD_AUG_PROJ and RAD_AUG_PROJ_FOCK
                      ! subtract the contributions from DDLM invert sign of POTAE
         DO ISP=1,NCDIJ
            DDLM=0
            CALL RAD_PROJ(  POTAE(:,:,ISP), PP%R, -1._q, DDLM, PP%LMAX, PP%LPS, PP%WPS )
            ! add the augmentation contributions including accurate augmentation
            IF (LPS_ONE_CENTER) THEN
               CALL RAD_AUG_PROJ( POTAE(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
               LYMAX, PP%AUG_SOFT, PP%QPAW )
               CALL RAD_AUG_PROJ_FOCK( POTAE(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
               LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK )
            ENDIF

            CALL RAD_PROJ(  POT(:,:,ISP)  , PP%R,-1._q, DDLM, PP%LMAX, PP%LPS, PP%WPS )
            CALL RAD_AUG_PROJ( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                  LYMAX, PP%AUG_SOFT, PP%QPAW )
         ! transform them using Clebsch Gordan coefficients and add to CDIJ
            CALL TRANS_DLM( CTMP(:,:,ISP), DDLM , PP )
         ENDDO
    
         ! non-collinear case: strength parameters need to go to the spinor presentation now
         IF (WDES%LNONCOLLINEAR) CALL DIJ_FLIP(CTMP,LMDIM)

         CDIJ(:,:,NIP,:)=CDIJ(:,:,NIP,:)+CTMP
      ENDDO ion
!=======================================================================
! now distribute the DIJ to all nodes which hold DIJ (using global sum)
!=======================================================================
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ))
#else
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ*2))
#endif

      DEALLOCATE(RHOCOL)
      DEALLOCATE( POTAE, RHOAE, POT, RHO)
    END SUBROUTINE SET_DD_OEP


!*******************************************************************
!
!  RAD_OEP_POT
!  calculate the one center potential corrections from the 
!  charge density corrections
!  the charge density rho(r) is 
!     rho(r) =  \sum_lm rho_lm(r) * Y_lm(r)  / r^2
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!
!  where rho_lm(r) is stored in RHO(2l+1+m,..) l=0,..,LMAX, m=0,..,2*l
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
! the charge density is supplied as 
!  total=RHO(:,:,1), magnetization=RHO(:,:,1)
! the potential however is returned for spin up and spin down 
!  spin up=POT(:,:,1), spin down=POT(:,:,1)
!
!*******************************************************************


    SUBROUTINE RAD_OEP_POT( R, ISPIN, LMAX_CALC, &
         RHO, POT )
      USE constant
      USE setexm

      IMPLICIT NONE
      TYPE (rgrid) :: R
      INTEGER ISPIN, LMAX_CALC
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      REAL(q) :: POT(:,:,:)     ! potential
    ! local variables
      INTEGER N,L,M,LM,K,KP,KPP,ISP
      REAL(q) V,DELTA,RP
      LOGICAL :: LBROADENING=.FALSE.
      REAL(q), PARAMETER :: DISTANCE=0.2
      REAL(q) :: TMP(R%NMAX),SUM,ALPHA
      
      POT=0
      N=R%NMAX
      ALPHA=1/DISTANCE**2
    ! restrict to spherical contributions
      DO L=0,MIN(0,LMAX_CALC)
      DO M=0,2*L
         LM=L*L+M+1
         DO ISP=1,ISPIN
            DO K=1,N
               POT(K,LM,ISP)  =RHO(K,LM,ISP)/ (R%R(K)*R%R(K))
            ENDDO
            IF (LBROADENING) THEN
            ! integral from other side (assuming even function)
            DO K=1,N
               SUM=0
               DO KP=1,N
                  IF ((R%R(K)+R%R(KP))**2*ALPHA >10 ) EXIT
                  SUM=SUM+EXP(-(R%R(K)+R%R(KP))**2*ALPHA)*R%SI(KP)*POT(KP,LM,ISP)
               ENDDO
               ! integral from other side (assuming even function)
               DO KP=1,N
                  IF ((R%R(K)-R%R(KP))**2*ALPHA >10 ) CYCLE
                  SUM=SUM+EXP(-(R%R(K)-R%R(KP))**2*ALPHA)*R%SI(KP)*POT(KP,LM,ISP)
               ENDDO
               SUM=SUM*SQRT(ALPHA)/SQRT(PI)
               TMP(K)=SUM
            ENDDO
            WRITE(77,*)
            WRITE(77,'(3F14.7)') (R%R(K),POT(K,LM,ISP),TMP(K),K=1,N)
            POT(1:N,LM,ISP)=TMP(1:N)
            ENDIF
         ENDDO

         IF (ISPIN==2) THEN 
            DO K=1,N
               V=POT(K,LM,1)
               DELTA=POT(K,LM,2)
               POT(K,LM,1)=V+DELTA
               POT(K,LM,2)=V-DELTA
            ENDDO
         ENDIF
      ENDDO
      ENDDO

    END SUBROUTINE RAD_OEP_POT

END MODULE PAWKLI

