#include "symbol.inc"
!#define vector
!*******************************************************************
! RCS:  $Id: radial.F,v 1.11 2003/06/27 13:22:22 kresse Exp kresse $
!
!  MODULE which supports operations on radial grid 
!  all routines written by gK with the exception of the
!  routines required for metaGGA, which were written by
!  Robin Hirsch, Dec 2000
!
!*******************************************************************
  MODULE radial
    USE prec

    ! structure which is used for the logarithmic grid
    ! the grid points are given by R(i) = RSTART * exp [H (i-1)]
    TYPE rgrid
       REAL(q)  :: RSTART          ! starting point
       REAL(q)  :: REND            ! endpoint
       REAL(q)  :: RMAX            ! radius of augmentation sphere
       REAL(q)  :: D               ! R(N+1)/R(N) = exp(H)
       REAL(q)  :: H               !
       REAL(q),POINTER :: R(:)     ! radial grid (r-grid)
       REAL(q),POINTER :: SI(:)    ! integration prefactors on r-grid
       INTEGER  :: NMAX            ! number of grid points
    END TYPE rgrid

    ! This parameter determines at which magnetization the aspherical contributions
    ! to the one center magnetization are truncated in the non collinear case
    !   Without any truncation the aspherical terms for non magnetic atoms
    ! tend to yield spurious but meaningless contributions to the potential
    ! so that convergence to the groundstate can not be achieved
    ! for details see the routines RAD_MAG_DIRECTION and RAD_MAG_DENSITY
    REAL(q), PARAMETER :: MAGMIN=1E-2

    ! for non collinear calculations, setting
    ! the parameter USE_AVERAGE_MAGNETISATION  means that the aspherical 
    ! contributions to the one center magnetisation are projected onto the
    ! average magnetization direction in the PAW sphere instead of the
    ! local moment of the spherical magnetization density at
    ! each grid-point
    ! USE_AVERAGE_MAGNETISATION improves the numerical stability significantly
    ! and must be set
    LOGICAL :: USE_AVERAGE_MAGNETISATION=.TRUE.

    !
    ! parameters that determine the spherical integration mesh 
    ! for potentials in the GGA/LDA case
    !
    INTEGER, PARAMETER, PRIVATE :: PHPTS_FACT=3, THPTS_FACT=3

#ifndef noPAW
  CONTAINS

!*******************************************************************
!
! RAD_ALIGN
! alligns R%RMAX to the radial grid
! this should be used to avoid that R%RMAX is slightly off
! from one of the grid points (due to rounding errors on reading)
!
!*******************************************************************

    SUBROUTINE RAD_ALIGN(R,RMAX)
      IMPLICIT REAL(q) (A-H,O-Z)
      TYPE (rgrid) R
      REAL(q) RMAX

      RMAX=RMAX-1E-5_q

      DO N=1,R%NMAX
         IF (R%R(N).GT. RMAX) THEN
            RMAX=R%R(N)
            EXIT
         ENDIF
      ENDDO

      CALL SET_SIMP(R)
    END SUBROUTINE RAD_ALIGN

!*******************************************************************
!
!  RAD_CHECK_QPAW
!  subroutine checks the consistency of the array QPAW
!  with the all electron and pseustored wavefunctions WAE and WPS
!    Q(PAW,ll') = \int (phi^AE(l,n,r)^2  -  phi^PS(l',n,r)^2) dr
!  if there are large errors the program stops and reports
!  the error
!  in all cases QPAW is corrected, so that it is exactly equal to
!  the integral
!
!  remember the wavefunction psi(r) can be obtained from the stored
!  coefficients using:
!      psi(r) = \sum_lmn Y_lm(r) w_ln(r) / r
!
!*******************************************************************

    SUBROUTINE RAD_CHECK_QPAW( R, CHANNELS, WAE, WPS , QPAW, QTOT, L, RDEP )
      IMPLICIT NONE

      TYPE (rgrid) R
      INTEGER CHANNELS
      REAL(q) :: WAE(:,:),WPS(:,:)   ! AE and soft wavefunctions
      REAL(q) :: QPAW(:,:,0:)        ! moments of compensation charge
      REAL(q) :: QTOT(:,:)           ! L=0 moments of AE charge
      INTEGER :: L(:)
      REAL(q) :: RDEP                ! outermost radius of augmentation charge 
! local variables
      REAL(q) :: RHOT(R%NMAX)
      INTEGER CH1,CH2,I
      REAL(q) :: RES
      INTEGER :: LL,LLP,LMIN,LMAX,LMAIN
      INTEGER :: IRMAX,NMAX_STORE

      DO IRMAX=1,R%NMAX-1
         ! because of rounding errors we need to be somewhat sloppy for the comparison
         IF (RDEP>0 .AND. R%R(IRMAX)-RDEP > -5E-3) EXIT
      ENDDO

      ! set the simpson weights in accordance with IRMAX
      NMAX_STORE=R%NMAX; R%NMAX=IRMAX
!      WRITE(*,*) NMAX_STORE, IRMAX
      CALL SET_SIMP(R)

      DO CH1=1,CHANNELS
      DO CH2=1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
         IF (LL==LLP) THEN
            RHOT=0
            DO I=1,IRMAX
               RHOT(I)=WAE(I,CH1)*WAE(I,CH2)
            ENDDO
            CALL SIMPI(R, RHOT , RES)
            QTOT(CH1,CH2)=RES
         ENDIF
      ENDDO
      ENDDO

      ! reset the simpson weights
      R%NMAX=NMAX_STORE
      CALL SET_SIMP(R)

      DO CH1=1,CHANNELS
      DO CH2=1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         DO LMAIN=LMIN,LMAX,2
            DO I=1,R%NMAX
               RHOT(I)=(WAE(I,CH1)*WAE(I,CH2)-WPS(I,CH1)*WPS(I,CH2))*R%R(I)**LMAIN
            ENDDO
            CALL SIMPI(R, RHOT , RES)
         ! screwed if we do not have the correct integrated charge
            IF ( ABS(RES-QPAW(CH1,CH2,0)) > 1E-4 .AND. LMAIN==0 ) THEN
               WRITE(0,1) CH1,CH2,RES,QPAW(CH1,CH2,0),RHOT(R%NMAX-4:R%NMAX)
 1             FORMAT('internal error RAD_CHECK_QPAW: QPAW is incorrect',/ &
               '      channels',2I3,' QPAW=',E20.10,' int=',10E20.10)
               STOP
            ENDIF
            QPAW(CH1,CH2,LMAIN)=RES
         ENDDO
      ENDDO
      ENDDO
    END SUBROUTINE RAD_CHECK_QPAW


!*******************************************************************
!
!  RAD_AUG_CHARGE
!  add the compensation charge density on the radial grid
!  to RHO
!  RHO(L,M,r) = RHO(L,M,r) +  Q(r,L) Q(PAW,ll' L) RHOLM(ll',LM)
!
!  Q(r,L) are the L-dependent 1-normalized compensation charges
!
!*******************************************************************

    SUBROUTINE RAD_AUG_CHARGE( RHO, R, RHOLM, CHANNELS, L, &
         LYMAX, AUG, QPAW )
      USE constant
      IMPLICIT NONE

      REAL(q) :: RHO(:,:)    ! charge on radial grid
      TYPE (rgrid) R
      REAL(q) :: RHOLM(:)    ! occupancy of each llLM channel
      REAL(q) :: AUG(:,0:)   ! 1-normalized L-dep compensation charge
      REAL(q) :: QPAW(:,:,0:)! moments of compensation charge Q(PAW,ll L)
      INTEGER :: LYMAX       ! maximum L
      INTEGER CHANNELS, L(:)
   ! local variables
      REAL(q) :: RHOLMT((LYMAX+1)*(LYMAX+1))
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I,LMAX
      INTEGER IBASE,JBASE,LMIN,LMAIN,MMAIN,LMMAIN
!-----------------------------------------------------------------------
! first contract to L and M dependent Q(LM)
!-----------------------------------------------------------------------
     IBASE=0
      RHOLMT=0

      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN
      ! add to LM dependent charge
         DO LMAIN=LMIN,MIN(LMAX,LYMAX),2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            RHOLMT(LMMAIN)=RHOLMT(LMMAIN)+QPAW(CH1,CH2,LMAIN)*RHOLM(LMMAIN+JBASE)
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
!-----------------------------------------------------------------------
! then add to charge on radial grid
!-----------------------------------------------------------------------
      ! could be replaced by matrix vecor DGEMV
      DO LMAIN=0,LYMAX
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            DO I=1,R%NMAX
               RHO(I,LMMAIN)=RHO(I,LMMAIN)+AUG(I,LMAIN)*RHOLMT(LMMAIN)
            ENDDO
         ENDDO
      ENDDO
#ifdef debug
      WRITE(0,*) 'RAD_AUG_CHARGE: L compensation charges are'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.5)') LL,(RHOLMT(LL**2+I)*2*SQRT(PI),I=1,LL*2+1)
      ENDDO
#endif
    END SUBROUTINE RAD_AUG_CHARGE


!*******************************************************************
!
!  RAD_AUG_CHARGE_FOCK
!  add the compensation charge density on the radial grid
!  to RHO
!  RHO(L,M,r) = RHO(L,M,r) +  Q(r,L) Q(PAW,ll' L) RHOLM(ll',LM)
!
!  this version adds the fock contributions which
!  restores the shape of the AE charge more accurately
!
!  Q(r,L) are the L-dependent 1-normalized compensation charges
!
!*******************************************************************

    SUBROUTINE RAD_AUG_CHARGE_FOCK( RHO, R, RHOLM, CHANNELS, L, &
         LYMAX, AUG_FOCK, QPAW_FOCK )
      USE constant
      IMPLICIT NONE

      REAL(q) :: RHO(:,:)    ! charge on radial grid
      TYPE (rgrid) R
      REAL(q) :: RHOLM(:)    ! occupancy of each llLM channel
      REAL(q),POINTER :: AUG_FOCK(:,:,:)   ! 1-normalized L-dep compensation charge
      REAL(q),POINTER :: QPAW_FOCK(:,:,:,:)! moments of compensation charge Q(PAW,ll L)
      INTEGER :: LYMAX       ! maximum L
      INTEGER CHANNELS, L(:)
   ! local variables
      REAL(q) :: RHOLMT((LYMAX+1)*(LYMAX+1))
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I,LMAX,N
      INTEGER IBASE,JBASE,LMIN,LMAIN,MMAIN,LMMAIN
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE

      IF (ASSOCIATED(QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(QPAW_FOCK,3)-1
      ELSE
         RETURN
      ENDIF

      DO N=1,NMAX_FOCKAE
!-----------------------------------------------------------------------
! first contract to L and M dependent Q(LM)
!-----------------------------------------------------------------------
      IBASE=0
      RHOLMT=0
      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN
      ! add to LM dependent charge
         DO LMAIN=LMIN,MIN(LMAX,LMAX_FOCKAE,LYMAX),2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            RHOLMT(LMMAIN)=RHOLMT(LMMAIN)+QPAW_FOCK(CH1,CH2,LMAIN,N)*RHOLM(LMMAIN+JBASE)
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
!-----------------------------------------------------------------------
! then add to charge on radial grid
!-----------------------------------------------------------------------
      ! could be replaced by matrix vecor DGEMV
      DO LMAIN=0,MIN(LMAX_FOCKAE,LYMAX)
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            DO I=1,R%NMAX
               RHO(I,LMMAIN)=RHO(I,LMMAIN)+AUG_FOCK(I,LMAIN,N)*RHOLMT(LMMAIN)
            ENDDO
         ENDDO
      ENDDO
      ENDDO
#ifdef debug
      WRITE(0,*) 'RAD_AUG_CHARGE: L compensation charges are'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.5)') LL,(RHOLMT(LL**2+I)*2*SQRT(PI),I=1,LL*2+1)
      ENDDO
#endif
    END SUBROUTINE RAD_AUG_CHARGE_FOCK

!*******************************************************************
!
!  RAD_AUG_PROJ
!  calculate the integral
!   D(ll'LM) =  \int V(r,L,M) Q(r,L) Q(PAW,ll' L) dr
!  on a radial grid
!  Q(r,L) are the L-dependent 1-normalized compensation charges
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
!*******************************************************************

    SUBROUTINE RAD_AUG_PROJ( POT, R, DLM, CHANNELS, L, &
         LYMAX, AUG, QPAW )
      USE constant
      IMPLICIT NONE

      REAL(q) :: POT(:,:)
      TYPE (rgrid) R
      REAL(q) :: DLM(:)
      REAL(q) :: AUG(:,0:)    ! 1-normalized L-dep compensation charge
      REAL(q) :: QPAW(:,:,0:) ! moments of compensation charges Q(PAW,ll L)
      INTEGER :: LYMAX        ! maximum L
      INTEGER CHANNELS,L(:)
   ! local variables
      REAL(q) :: RHOLMT((LYMAX+1)*(LYMAX+1)),SUM
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I,LMAX
      INTEGER IBASE,JBASE,LMIN,LMAIN,MMAIN,LMMAIN
!-----------------------------------------------------------------------
! first calculate \int V(L,M) Q(L,M)
!-----------------------------------------------------------------------
      RHOLMT=0
      ! could be replaced by matrix vecor DGEMV
      DO LMAIN=0,LYMAX
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            SUM=0
            DO I=1,R%NMAX
               SUM=SUM+POT(I,LMMAIN)*AUG(I,LMAIN)
            ENDDO
            RHOLMT(LMMAIN)=SUM
         ENDDO
      ENDDO
!-----------------------------------------------------------------------
! than multiply with QPAW(llp, L) and add the DLM
!-----------------------------------------------------------------------
      IBASE=0

      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN
      ! add to LM dependet charge
         DO LMAIN=LMIN,MIN(LYMAX,LMAX),2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            DLM(LMMAIN+JBASE)=DLM(LMMAIN+JBASE)-RHOLMT(LMMAIN)*QPAW(CH1,CH2,LMAIN)
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
#ifdef debug
      WRITE(0,*) 'RAD_AUG_PROJ: int V Q(LM) is'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.5)') LL,(RHOLMT(LL**2+I)*2*SQRT(PI),I=1,LL*2+1)
      ENDDO
#endif
    END SUBROUTINE RAD_AUG_PROJ

!*******************************************************************
!
!  RAD_AUG_PROJ_FOCK
!  calculate the integral
!   D(ll'LM) =  \int V(r,L,M) Q_FOCK(r,L) Q_FOCK(PAW,ll' L) dr
!  on a radial grid
!  Q(r,L) are the L-dependent 0-normalized compensation charges
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
!*******************************************************************

    SUBROUTINE RAD_AUG_PROJ_FOCK( POT, R, DLM, CHANNELS, L, &
         LYMAX, AUG_FOCK, QPAW_FOCK )
      USE constant
      IMPLICIT NONE

      REAL(q) :: POT(:,:)
      TYPE (rgrid) R
      REAL(q) :: DLM(:)
      REAL(q),POINTER :: AUG_FOCK(:,:,:)    ! 1-normalized L-dep compensation charge
      REAL(q),POINTER :: QPAW_FOCK(:,:,:,:) ! moments of compensation charges Q(PAW,ll L)
      INTEGER :: LYMAX        ! maximum L
      INTEGER CHANNELS,L(:)
   ! local variables
      REAL(q) :: RHOLMT((LYMAX+1)*(LYMAX+1)),SUM
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I,LMAX,NAE
      INTEGER IBASE,JBASE,LMIN,LMAIN,MMAIN,LMMAIN
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE

      IF (ASSOCIATED(QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(QPAW_FOCK,3)-1
      ELSE
         RETURN
      ENDIF

      DO NAE=1,NMAX_FOCKAE
!-----------------------------------------------------------------------
! first calculate \int V(L,M) Q(L,M)
!-----------------------------------------------------------------------
      RHOLMT=0
      ! could be replaced by matrix vecor DGEMV
      DO LMAIN=0,MIN(LYMAX,LMAX_FOCKAE)
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            SUM=0
            DO I=1,R%NMAX
               SUM=SUM+POT(I,LMMAIN)*AUG_FOCK(I,LMAIN,NAE)
            ENDDO
            RHOLMT(LMMAIN)=SUM
         ENDDO
      ENDDO
!      WRITE(*,*) '2',RHOLMT(1) ! can be compared to DLM in SETDIJ in us.F
!-----------------------------------------------------------------------
! then multiply with QPAW(llp, L) and add the DLM
!-----------------------------------------------------------------------
      IBASE=0

      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN
      ! add to LM dependet charge
         DO LMAIN=LMIN,MIN(LMAX,LMAX_FOCKAE,LYMAX),2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            DLM(LMMAIN+JBASE)=DLM(LMMAIN+JBASE)-RHOLMT(LMMAIN)*QPAW_FOCK(CH1,CH2,LMAIN,NAE)
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
#ifdef debug
      WRITE(0,*) 'RAD_AUG_PROJ: int V Q(LM) is'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.5)') LL,(RHOLMT(LL**2+I)*2*SQRT(PI),I=1,LL*2+1)
      ENDDO
#endif
      ENDDO
    END SUBROUTINE RAD_AUG_PROJ_FOCK


!*******************************************************************
!
!  RAD_CHARGE
!  calculate the soft pseudo/AE charge density on the radial grid
!  from a set of frozen wavefunctions and the ll LM dependent
!  occupancies
!  the normalised wavefunction psi(r) can be obtained from the stored 
!  coefficients w(l,n,r) = W(:,:) in the following way
!      psi(r) = \sum_lmn Y_lm(r) w_ln(r) / r
!  here 
!'      rho_LM(r)= \sum_n w_ln(r) w_l'n'(r)  RHOLM(ll',LM)
!  is calculated and stored in RHO(2l+1+m,..) l=0,..,LMAX, m=0,..,2*l
!  RHOLM is the occupancy of each channel (see TRANS_RHOLM)
!  thus the charge density can be obtained from the stored
!  coefficients rho_lm(r) in the following way
!     rho(r) =  \sum_lm rho_lm(r) * Y_lm(r)  / r^2
!
!*******************************************************************

    SUBROUTINE RAD_CHARGE( RHOAE, R, RHOLM, CHANNELS, L, W )
      IMPLICIT NONE

      REAL(q) :: RHOAE(:,:)
      TYPE (rgrid) R
      INTEGER CHANNELS, L(:)
      REAL(q) :: RHOLM(:)
      REAL(q) :: W(:,:)
! local variables
      REAL(q) :: RHOT(R%NMAX)
      INTEGER CH1,CH2,LL,LLP,I
      INTEGER IBASE,JBASE,LMIN,LMAX,LMAIN,MMAIN,LMMAIN
   ! loop over all channels (l,epsilon)
      IBASE=0

      DO CH1=1,CHANNELS
      DO CH2=CH1,CHANNELS
         DO I=1,R%NMAX
            RHOT(I)=W(I,CH1)*W(I,CH2)
         ENDDO
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN
      ! add to LM dependent charge
         ! loop could be replaced by matrix vector DGEMV
         DO LMAIN=LMIN,LMAX,2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            DO I=1,R%NMAX
               RHOAE(I,LMMAIN)=RHOAE(I,LMMAIN)+RHOT(I)*RHOLM(LMMAIN+JBASE)
            ENDDO
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      ENDDO
      ENDDO

    END SUBROUTINE RAD_CHARGE


!*******************************************************************
!
!  RAD_KINETIC_EDENS
!  calculate the radial kinetic energy denstiy and weizsaecker kin Edens
!  from a set of frozen wavefunctions and the ll LM dependent
!  occupancies
!  the kinetic energy density can be obtained from the stored 
!  coefficients KINDENS in the following way
!     tau(r) =  \sum_lm KINDENS(r) * Y_lm(r)  / r^2
!  charge density and occupancies are passed as up and down spin respectively
!  for spin polarized calculations
!
! Robin Hirschl 20001222
!*******************************************************************

    SUBROUTINE RAD_KINETIC_EDENS( KINDENS, WKD, R, RHOLM, CHANNELS, L, W, RHOC, TAUC, RHO,ISPIN)
      USE constant
      IMPLICIT NONE

      REAL(q) :: KINDENS(:)     ! result, kinetic energy density
      REAL(q) :: WKD(:)         ! Weizsaecker kinetic edens
      TYPE (rgrid) R            ! descriptor for radial grid
      INTEGER CHANNELS, L(:)    ! number of channels and corsp. L
      REAL(q) :: RHOLM(:)       ! occupancy of each channel
      REAL(q) :: W(:,:)         ! wave function
      REAL(q) :: RHO(:,:)       ! radial part of charge density       
      REAL(q),POINTER :: RHOC(:)! core charge density
      REAL(q),POINTER :: TAUC(:)! kinetic energy density of core electrons
      ! local variables
      REAL(q) :: K(R%NMAX), DRHO(R%NMAX), WTMP(R%NMAX),DW1(R%NMAX),DW2(R%NMAX)
      REAL(q) :: TOTKINE 
      INTEGER CH1,CH2,LL,LLP,I,KONST
      INTEGER IBASE,JBASE,LMIN,LMAX,LMAIN,MMAIN,LMMAIN,ISPIN

      ! loop over all channels (l,epsilon)
      IBASE=0

      DO CH1=1,CHANNELS
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
         IF (LL == LLP) THEN
            ! here one has to calculate the spherical contribution
            ! to the kinetic energy density
            ! first calculate radial differentiations of the two channels
            ! W is first stored in WTMP which is destroyed on return from GRAD_
            
            WTMP(:)=W(:,CH1)/R%R
            CALL GRAD_(R,WTMP,DW1)
            WTMP(:)=W(:,CH2)/R%R
            CALL GRAD_(R,WTMP,DW2)

            ! the spherical contributions to the kinetic energy density are of the form
            ! (d/dr w1*d/dr w2+k*w1*w2/r^2)*sum_m Ylm^2; k depends on l only
            ! we need the result multiplied by r^2 

            ! set up constants
            ! of course the same as in the radial representation of the laplace operator
            KONST=(LL+1)*LL  
            ! calculate radial part
            DO I=1,R%NMAX
               K(I)=(DW1(I)*DW2(I)*R%R(I)**2+ &
                    KONST*W(I,CH1)*W(I,CH2)/R%R(I)**2)
            ENDDO

            ! now add to kinetic energy density (only L=0 presently)
            LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
            JBASE=IBASE-LMIN*LMIN      !
            LMAIN=0                    ! l=0
            MMAIN=1                    ! m=0 (we start indexing at 1)

            LMMAIN=LMAIN*LMAIN+MMAIN 
            ! currently LMMAIN is allways 1, and JBASE=IBASE
            DO I=1,R%NMAX
               KINDENS(I)=KINDENS(I)+K(I)*RHOLM(LMMAIN+JBASE)*HSQDTM
            ENDDO
         ENDIF

         IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      ENDDO
      ENDDO

      IF (ASSOCIATED(TAUC).AND.ISPIN/=4) THEN
         KINDENS(1:R%NMAX)=KINDENS(1:R%NMAX)+TAUC(1:R%NMAX)/ISPIN
      ENDIF

      ! Weizsaecker kinetic energy density
      WTMP=RHO(1:R%NMAX,1)/(R%R**2)
      IF (ASSOCIATED(RHOC)) THEN
         WTMP=WTMP+RHOC/(ISPIN*R%R**2)
      ENDIF

      CALL GRAD_(R,WTMP,DRHO)
      DO I=1,R%NMAX
         IF (WTMP(I)==0) THEN
            WKD(I)=0._q
         ELSE
!gK         WKD(I)=0.25*HSQDTM*DRHO(I)**2._q/RHO(I)*R%R(I)**4._q
            WKD(I)=0.25*HSQDTM*DRHO(I)**2._q/WTMP(I)*R%R(I)**2._q
         ENDIF
         
         ! force physical condition tauw <= tau
         ! may be violated due to numerical errors
         WKD(I)=MIN(WKD(I),KINDENS(I))
      ENDDO
      
    END SUBROUTINE RAD_KINETIC_EDENS

!*******************************************************************
!
!  FLIP_RAD
!  Flips an arbitrary real(!) array from total, magnetization to
!  up,down spin
!   
!  Robin Hirschl 20010109
!*******************************************************************
    SUBROUTINE FLIP_RAD(WORKIN,WORKOUT,N)
      IMPLICIT NONE

      REAL(q) :: WORKIN(:,:),WORKOUT(:,:)
      INTEGER I,N
      REAL(q) TEMP

      DO I=1,N
         TEMP=WORKIN(I,1)
         WORKOUT(I,1)=(WORKIN(I,1)+WORKIN(I,2))/2._q
         WORKOUT(I,2)=(TEMP-WORKIN(I,2))/2._q
      ENDDO
    END SUBROUTINE FLIP_RAD

!*******************************************************************
!
!  RAD_PROJ
!  calculate
!'  D(ll',LM)= D(ll',LM)+ A *\int dr phi_l'(r) pot_lm(r) phi_l(r) dr
!  on the radial grid
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
!*******************************************************************

    SUBROUTINE RAD_PROJ( POT, R, A, DLM, CHANNELS, L, W )
      IMPLICIT NONE

      REAL(q) :: POT(:,:)     ! radial potential V(r,L,M)
      TYPE (rgrid) R
      REAL(q) :: DLM(:)
      REAL(q) :: W(:,:)       ! wavefunctions phi(r,l)
      REAL(q) :: A            ! scaling factor
      INTEGER CHANNELS, L(:)
! local variables
      REAL(q) :: RHOT(R%NMAX),SUM
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I
      INTEGER IBASE,JBASE,LMIN,LMAX,LMAIN,MMAIN,LMMAIN
   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
         DO I=1,R%NMAX
            RHOT(I)=W(I,CH1)*W(I,CH2)
         ENDDO
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
      ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         JBASE=IBASE-LMIN*LMIN

         DO LMAIN=LMIN,LMAX,2
         DO MMAIN=1,LMAIN*2+1
            LMMAIN=LMAIN*LMAIN+MMAIN
            SUM=0

            ! integrate RHOT POT(L,M) (the potentials are already weighted)
            DO I=1,R%NMAX
               SUM=SUM+RHOT(I)*POT(I,LMMAIN)
            ENDDO
            DLM(LMMAIN+JBASE)=DLM(LMMAIN+JBASE)+SUM*A
         ENDDO
         ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

    END SUBROUTINE RAD_PROJ

!*******************************************************************
!
!  RAD_PROJ_KINPOT
!  calculate the integral
!'   D(ll',LM) =  D(ll',LM) + A * < nabla phi_l' | V(L,M) | nabla phi_l >
!  on the radial grid
!  only spherical contributions are used
!
!*******************************************************************

    SUBROUTINE RAD_PROJ_KINPOT( POT, R, A, DLM, CHANNELS, L, W )
      IMPLICIT NONE

      REAL(q) :: POT(:)       ! radial kinetic energy potential V(r,L,M)
      TYPE (rgrid) R
      REAL(q) :: DLM(:)
      REAL(q) :: W(:,:)       ! wavefunctions phi(r,l)
      REAL(q) :: A            ! scaling factor
      INTEGER CHANNELS, L(:)
! local variables
      REAL(q) :: K(R%NMAX),SUM
      INTEGER CH1,CH2,LL,LLP,LM,LMP,I
      INTEGER IBASE,JBASE,LMIN,LMAX,LMAIN,MMAIN,LMMAIN
   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,CHANNELS
      LMP=LM
      DO CH2=CH1,CHANNELS
      ! quantum numbers l and lp of these two channels
         LL =L(CH1)
         LLP=L(CH2)
         
         IF (LL == LLP) THEN

            ! here one has to calculate the spherical contribution
            ! to the kinetic energy density

            DO I=1,R%NMAX
               K(I)=W(I,CH1)*W(I,CH2)
            ENDDO

            ! Lmin and Lmax
            LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
            JBASE=IBASE-LMIN*LMIN

            LMAIN=0
            MMAIN=1

              ! currently LMMAIN is allways 1, and JBASE=IBASE
              LMMAIN=LMAIN*LMAIN+MMAIN
              ! integrate K *  POT (the potential is already weighted)
              SUM=0
              DO I=1,R%NMAX
                 SUM=SUM+K(I)*POT(I)
              ENDDO
              DLM(LMMAIN+JBASE)=DLM(LMMAIN+JBASE)+SUM*A
         ENDIF

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

    END SUBROUTINE RAD_PROJ_KINPOT

!*******************************************************************
!
!  RAD_INT
!  integrate the soft (including compensation charge)
!  and AE chargedensity and calculate
!  the moments (if they are not the same we have a problem)
!
!*******************************************************************

    SUBROUTINE RAD_INT(  R,  LYMAX, RHO, RHOAE )
      USE constant
      REAL(q) :: RHOAE(:,:),RHO(:,:)
      TYPE (rgrid) R
    ! local
      INTEGER LL,I,II
      REAL(q)  RHOL(LYMAX*2+1), RHOLAE(LYMAX*2+1)
      REAL(q)  RHOT(R%NMAX)

      DO LL=0,LYMAX
         DO I=1,LL*2+1
            DO K=1,R%NMAX
               RHOT(K)=RHO(K,LL*LL+I)*R%R(K)**LL
            ENDDO
            CALL SIMPI(R, RHOT, RHOL(I))
            DO K=1,R%NMAX
               RHOT(K)=RHOAE(K,LL*LL+I)*R%R(K)**LL
            ENDDO
            CALL SIMPI(R, RHOT, RHOLAE(I))
         ENDDO
         DO II=1,LL*2+1
         ! we are screwed
         ! the moments are not correctly conserved
         IF ( ABS(RHOL(II)-RHOLAE(II)) >1E-4_q) THEN
            WRITE(0,'(I3,10F14.9)') LL,(RHOL(I)*2*SQRT(PI),I=1,LL*2+1)
            WRITE(0,'(I3,10F14.9)') LL,(RHOLAE(I)*2*SQRT(PI),I=1,LL*2+1)
            WRITE(0,*)' internal error in RAD_INT: RHOPS /= RHOAE'
            STOP
         ENDIF
         ENDDO
      ENDDO

    END SUBROUTINE RAD_INT

!*******************************************************************
!
!  print out the charge deficit for each L,M channel
!
!*******************************************************************

    SUBROUTINE RAD_DEF(  R,  LYMAX, RHO, RHOAE )
      USE constant
      REAL(q) :: RHOAE(:,:),RHO(:,:)
      TYPE (rgrid) R
    ! local
      INTEGER LL,I,II
      REAL(q)  RHOL(LYMAX*2+1), RHOLAE(LYMAX*2+1)
      REAL(q)  RHOT(R%NMAX)

      DO LL=0,LYMAX
         DO I=1,LL*2+1
            DO K=1,R%NMAX
               RHOT(K)=RHO(K,LL*LL+I)*R%R(K)**LL
            ENDDO
            CALL SIMPI(R, RHOT, RHOL(I))
            DO K=1,R%NMAX
               RHOT(K)=RHOAE(K,LL*LL+I)*R%R(K)**LL
            ENDDO
            CALL SIMPI(R, RHOT, RHOLAE(I))
         ENDDO
         WRITE(0,'("RADDEF",I2,14F10.6)') LL,(RHOL(II)-RHOLAE(II),II=1,LL*2+1)
      ENDDO

    END SUBROUTINE RAD_DEF

!*******************************************************************
!
!  RAD_POT
!  calculate the radial potential from the radial chargedensity
!  the charge density rho(r) is given by
!     rho(r) =  \sum_lm rho_lm(r) * Y_lm(r)  / r^2
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!
!  where rho_lm(r) is stored in RHO(2l+1+m,..) l=0,..,LMAX, m=0,..,2*l
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
! in many places we use a scaling factor 2 sqrt(pi)
! the "real" charge density for L=0 angular quantum number is
!   n_0= rho_00 Y_00 = RHO(r,0) / (2 sqrt(pi))
! for other channels it is
!   n_lm= rho_lm Y_lm
! in comments we will always distinct between n_L and rho_L
!
! the charge density is supplied as 
!  total=RHO(:,:,1), magnetization=RHO(:,:,2)
! the potential however is returned for spin up and spin down 
!  spin up=POT(:,:,1), spin down=POT(:,:,2)
!
!*******************************************************************


    SUBROUTINE RAD_POT( R, ISPIN, LMAX, LMAX_CALC, LASPH, &
         RHO, RHOC, POTC, POT, DOUBLEC, EXCG)
      USE constant
      USE setexm

      IMPLICIT NONE
      INTEGER LMAX, ISPIN, LMAX_CALC
      LOGICAL :: LASPH          ! non spherical contributions
      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      REAL(q) :: POTC(:)        ! froze core potential
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      ! RHO(:,:,1) contains total charge distribution
      ! RHO(:,:,2) magnetization charge distribution
      REAL(q) :: POT(:,:,:)     ! potential
      ! POT(:,:,1) up   component
      ! POT(:,:,2) down component
      TYPE (rgrid) :: R
      REAL(q) :: EXCG           ! exchange energy only
      REAL(q) :: DOUBLEC        ! double counting corrections
    ! local variables
      REAL(q) RHOT(R%NMAX,ISPIN)
      INTEGER K,N,I,L,M,LM
      REAL(q) SCALE,SUM
      
      LOGICAL,PARAMETER :: TREL=.TRUE. ! use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TLDA=.TRUE. ! calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      ! in this case non spherical contributions are missing
      REAL(q) :: DHARTREE,DEXC_LDA,DVXC_LDA,DEXC_GGA,DVXC_GGA
      REAL(q) :: TMP((LMAX+1)*(LMAX+1),ISPIN)
!
      REAL(q) SIM_FAKT, RHOP, EXT, VXT, DEXC1, DVXC1
      REAL(q) T1(R%NMAX),T2(R%NMAX),V1(R%NMAX)

      POT=0

      SCALE=2*SQRT(PI)
      N=R%NMAX

      DHARTREE=0
      DO L=0,LMAX_CALC
      DO M=0,2*L
         LM=L*L+M+1
         CALL RAD_POT_HAR(L,R,POT(:,LM,1),RHO(:,LM,1),SUM)
         IF (ISPIN==2) POT(:,LM,2)=POT(:,LM,1)
         DHARTREE=DHARTREE+SUM
         TMP(LM,1)=SUM
      ENDDO
      ! WRITE(0,'(I2,10F12.7)') L, (TMP(LM,1),LM=L*L+1,(L+1)*(L+1))
      ENDDO
      DO K=1,N
         POT(K,1,1)=POT(K,1,1)+POTC(K)*SCALE
      ENDDO
      IF (ISPIN==2) POT(:,1,2)=POT(:,1,1)
!========================================================================
! exchange correlation energy, potential
! and double counting corrections
!========================================================================
      DEXC_LDA=0
      DVXC_LDA=0

      IF (ISPIN==1) THEN
        DO K=1,N
          RHOT(K,1)=(RHO(K,1,1)+RHOC(K))/ (SCALE*R%R(K)*R%R(K)) ! charge density rho_0 Y(0)
        ENDDO

      ELSE
        DO K=1,N
         RHOT(K,1)=(RHO(K,1,1)+RHOC(K))/ (SCALE*R%R(K)*R%R(K)) ! charge density n_0=rho_0 Y(0)
         RHOT(K,2)= RHO(K,1,2)/ (SCALE*R%R(K)*R%R(K))          ! magnetization
        ENDDO

      ENDIF
      ! WRITE(*,'(9F14.7)') POT(R%NMAX-10,:,:)
!========================================================================
! LDA+ GGA if required
!========================================================================
      IF (.NOT. LASPH) THEN
         ilda: IF (TLDA) THEN
            
            IF (ISPIN==1) THEN
               CALL RAD_LDA_XC( R, TREL, LMAX_CALC, RHOT(:,1), RHO(:,:,1), POT(:,:,1), DEXC_LDA, DVXC_LDA, .TRUE.)
            ELSE
               CALL RAD_LDA_XC_SPIN( R, TREL, LMAX_CALC, RHOT, RHO, POT, DEXC_LDA, DVXC_LDA, .TRUE.)
            ENDIF
            
         ENDIF ilda
         DEXC_GGA=0
         DVXC_GGA=0
         
         gga: IF (ISGGA()) THEN
            
            IF (ISPIN==1) THEN
               CALL RAD_GGA_XC( R, TLDA, RHOT(:,1), RHO(:,1,1), POT(:,1,1), DEXC_GGA, DVXC_GGA)
            ELSE
               DO K=1,N
                  RHOT(K,1)=(RHO(K,1,1)+RHOC(K)+RHO(K,1,2))/(2*SCALE*R%R(K)*R%R(K)) ! up
                  RHOT(K,1)=MAX(RHOT(K,1), 1E-7_q)
                  RHOT(K,2)=(RHO(K,1,1)+RHOC(K)-RHO(K,1,2))/(2*SCALE*R%R(K)*R%R(K)) ! down
                  RHOT(K,2)=MAX(RHOT(K,2), 1E-7_q)
               ENDDO
               
               CALL RAD_GGA_XC_SPIN( R, TLDA, RHOT, RHO(:,1,:), POT(:,1,:), DEXC_GGA, DVXC_GGA)
            ENDIF
         ENDIF gga
      ELSE
         CALL RAD_GGA_ASPH( R, ISPIN, LMAX_CALC, &
              RHO, RHOC, POTC, DEXC_LDA, DEXC_GGA, DVXC_LDA, DVXC_GGA, POT )
      ENDIF
!========================================================================
! thats it
!========================================================================
      DOUBLEC= -DHARTREE/2+DEXC_LDA-DVXC_LDA+DEXC_GGA-DVXC_GGA
      EXCG= DEXC_LDA+DEXC_GGA
#ifdef debug
      WRITE(*,1)  N,-DHARTREE/2, -DVXC_LDA-DVXC_GGA, DEXC_LDA+DEXC_GGA
1     FORMAT(' -Hartree, -vxc, exc:',I4,3F14.7)
#endif

    END SUBROUTINE RAD_POT


!*******************************************************************
!
!  RAD_POT_HAR_ONLY
!  calculate the radial potential from the radial chargedensity
!  Hartree contribution only
!  see above for details
!
!*******************************************************************


    SUBROUTINE RAD_POT_HAR_ONLY( R, ISPIN, LMAX, LMAX_CALC, &
         RHO, POT, DOUBLEC )
      USE constant

      IMPLICIT NONE
      INTEGER LMAX, ISPIN, LMAX_CALC
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      ! RHO(:,:,1) contains total charge distribution
      ! RHO(:,:,2) magnetization charge distribution
      REAL(q) :: POT(:,:,:)     ! potential
      ! POT(:,:,1) up   component
      ! POT(:,:,2) down component
      TYPE (rgrid) :: R
      REAL(q) :: DOUBLEC
    ! local variables
      
      REAL(q) :: DHARTREE, SCALE, SUM
      INTEGER L,M,LM,N

      POT(:,1:(LMAX+1)*(LMAX+1),:)=0

      SCALE=2*SQRT(PI)
      N=R%NMAX

      DHARTREE=0
      DO L=0,LMAX_CALC
      DO M=0,2*L
         LM=L*L+M+1
         CALL RAD_POT_HAR(L,R,POT(:,LM,1),RHO(:,LM,1),SUM)
         IF (ISPIN==2) POT(:,LM,2)=POT(:,LM,1)
         DHARTREE=DHARTREE+SUM
      ENDDO
      ENDDO
      IF (ISPIN==2) POT(:,1,2)=POT(:,1,1)

      DOUBLEC= -DHARTREE/2

    END SUBROUTINE RAD_POT_HAR_ONLY


!*******************************************************************
!
!  RAD_POT_DARWIN
!  the potential is given by
!       V(r) =  \sum_lm pot_lm(r) * Y_lm(r)
!
!  and   pot_lm(r) is stored in POT(2l+1+m,..)
!
!  calculate darwin shift to adjust the gaps and the dispersion
!  This is done by adding to the effective potentials
!  an additional external potential V_w(r).
!
!  the external potential is given by
!       V_w(r) = V_0 {r_0/r} EXP [ -(r/r_0)^2 ]
!
!*******************************************************************


    SUBROUTINE RAD_POT_DARWIN( R, ISPIN, POT, DARWIN_V, DARWIN_R )
      USE constant

      IMPLICIT NONE
      INTEGER ISPIN
      REAL(q) :: POT(:,:,:)     ! potential
      ! POT(:,:,1) up   component
      ! POT(:,:,2) down component
      REAL(q) :: DARWIN_V, DARWIN_R ! Parameters of the external potential
      TYPE (rgrid) :: R
      ! local variables
      
      REAL(q) :: SCALE, SUM
      INTEGER :: L,M,LM,N

      SCALE=2*SQRT(PI)
      N=R%NMAX

      L=0
      M=0
      LM=L*L+M+1
      DO N=1,R%NMAX
         POT(:,LM,1)=DARWIN_V*(DARWIN_R/R%R(N))*EXP(-(R%R(N)/DARWIN_R)**2_q)*SCALE
         IF (ISPIN==2) POT(:,LM,2)=DARWIN_V*(DARWIN_R/R%R(N))*EXP(-(R%R(N)/DARWIN_R)**2_q)*SCALE
      ENDDO

    END SUBROUTINE RAD_POT_DARWIN

!*******************************************************************
!
!  RAD_GGA_ASPH
! calculate the one center exchange energy and the potential
! by performing a angular integration for each radial grid
! point
!
! Robin Hirschl   20010118
! potential added 20050225 gK
!
!*******************************************************************
    
    SUBROUTINE RAD_GGA_ASPH( R, ISPIN, LMAX_CALC, &
         RHO, RHOC, POTC, EXC_LDA, EXC_GGA, DVXC_LDA, DVXC_GGA, POT )
      USE constant
      USE asa
      USE setexm

      IMPLICIT NONE
      INTEGER ISPIN, LMAX_CALC
      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      REAL(q) :: POTC(:)        ! froze core potential
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      ! RHO(:,:,1) contains total charge distribution
      ! RHO(:,:,2) magnetization charge distribution
      REAL(q), OPTIONAL :: POT(:,:,:)     ! potential
      ! POT(:,:,1) up   component
      ! POT(:,:,2) down component
      TYPE (rgrid) :: R
      REAL(q) :: EXC_LDA, EXC_GGA, DVXC_LDA, DVXC_GGA
    ! set potential
      LOGICAL :: LPOT           ! set potential or not
    ! local variables
      REAL(q) RHOT(R%NMAX,ISPIN),RHO_ANG(R%NMAX,ISPIN,3),POTXC(R%NMAX,1,ISPIN)
      INTEGER ISP,K,I,J,LM,LMP,NP,IFAIL,LLMAX,LMMAX,LMMAX_CALC,PHPTS,THPTS,NPTS
      REAL(q) SCALE,SUM,DELTAPHI

      LOGICAL,PARAMETER :: TREL=.TRUE. ! use relativistic corrections to exchange
      LOGICAL,PARAMETER :: TLDA=.TRUE. ! calculate LDA contribution seperately
      ! TLDA=.FALSE. works only for Perdew Burke Ernzerhof
      REAL(q) :: DVINI
!
      REAL(q) SIM_FAKT
      REAL(q) TEMP1
      REAL(q), ALLOCATABLE :: RADPTS(:,:), XYZPTS(:,:),YLM(:,:),YLMD(:,:,:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: MU(:,:,:,:)
      REAL(q), ALLOCATABLE :: WEIGHT(:),ABSCIS(:)

      REAL(q) T1(R%NMAX,ISPIN)
      REAL(q) XU,YU,ZU,XD,YD,ZD,TG,EXT,DEXC,DEXC1,DEXC2,DVXC,DVXC1,DVXC2,DVC
      REAL(q) TMP(ISPIN,3)
      EXTERNAL GAUSSI2
      
      IF (PRESENT(POT)) THEN
         LPOT=.TRUE.
      ELSE
         LPOT=.FALSE.
      ENDIF
      ! initial double counting corrections
      DVINI=0
      IF (LPOT) THEN
         CALL RAD_POT_IN_RHO( R, ISPIN, LMAX_CALC, RHO, POT, DVINI)
      ENDIF
      
      SCALE=2*SQRT(PI)                 ! 1/Y00

      LMMAX_CALC=(LMAX_CALC+1)**2
      ! restriction for the angular development
      LLMAX=MIN(6,LMAX_CALC)
      ! in GGA this needs to be increased by 1 (nabla-comment)
      IF (ISGGA()) LLMAX=LLMAX+1
      LMMAX=(LLMAX+1)**2

!========================================================================
! number of theta and phi pivot points to perform angular integration
! since Exc=f(a*Yllmax,m) we need more pivot points than theoretically 
! needed to integrate Yllmax,m. 
! the factor 2 is the minium, 3 is more accurate
!========================================================================
! update 23.02.2011 by Georg Kresse
! the following lines cause sizeable errors in the one-electron energies e.g. for
! LDA+U calculations on NiO
! all versions up to vasp.5.2.11
!      PHPTS=2*(LLMAX+1)
!      THPTS=2*FLOOR(REAL(LLMAX/2+1,KIND=q))
! this version is much much better, yielding the correct degeneracies in the
! meV regime
      PHPTS=PHPTS_FACT*(LLMAX+1)
      THPTS=THPTS_FACT*FLOOR(REAL(LLMAX/2+1,KIND=q))
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX),YLMD(NPTS,LMMAX,3),XYZPTS(NPTS,3),RADPTS(NPTS,2), & 
           YLM_NABLA_YLM(LMMAX,LMMAX,0:3),YLM_X_YLM(LMMAX,LMMAX,0:3), &
           WEIGHT(THPTS),ABSCIS(THPTS))
      

      IF (LPOT) THEN
         ALLOCATE (MU(R%NMAX,LMMAX,0:3,ISPIN))
         MU=0
         CALL SETYLM_NABLA_YLM(LLMAX,YLM_NABLA_YLM,YLM_X_YLM)
      ENDIF

      RADPTS=0; WEIGHT=0; ABSCIS=0
      ! set phi positions, equally spaces
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
      YLM=0 ; YLMD=0

      CALL SETYLM_GRAD2(LLMAX,NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

!========================================================================
! main loop over all grid points
! prepare the charge density array RHOT
!========================================================================
      EXC_LDA=0
      EXC_GGA=0

! loop over all points in the angular grid
      points: DO NP=1,NPTS

         ! weight of this points
         SIM_FAKT=DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))

         RHOT=0
         RHO_ANG=0

         ! calculate the total charge RHOT = RHO/r^2 (up and down)
         ! and the gradient of the charge RHO_ANG
         DO K=1,R%NMAX
            DO LM=1,LMMAX_CALC
               IF (ISPIN==1) THEN
                  ! rho(r) Y_L(r)
                  RHOT(K,1)     =RHOT(K,1)     +YLM(NP,LM)   *RHO(K,LM,1)
                  ! rho(r) nabla Y_L(r)
                  RHO_ANG(K,1,:)=RHO_ANG(K,1,:)+YLMD(NP,LM,:)*RHO(K,LM,1)
               ELSE
                  RHOT(K,1)=RHOT(K,1)+YLM(NP,LM)*RHO(K,LM,1)  ! total charge
                  RHOT(K,2)=RHOT(K,2)+YLM(NP,LM)*RHO(K,LM,2)  ! magnetization
                  
                  RHO_ANG(K,1,:)=RHO_ANG(K,1,:)+YLMD(NP,LM,:)*RHO(K,LM,1)  ! total charge
                  RHO_ANG(K,2,:)=RHO_ANG(K,2,:)+YLMD(NP,LM,:)*RHO(K,LM,2)  ! magnetization
               ENDIF
            ENDDO
            ! add core charge (spherical) and divide by 1/r^2
            RHOT(K,1)=(RHOT(K,1)+RHOC(K)*YLM(NP,1))/(R%R(K)*R%R(K))
            IF (ISPIN==2) RHOT(K,2)=RHOT(K,2)/(R%R(K)*R%R(K))

            ! divide nabla Y_L through r^3 (1/r is missing in YLMD)
            RHO_ANG(K,:,:)=RHO_ANG(K,:,:)/(R%R(K)*R%R(K))/R%R(K)
         ENDDO

!========================================================================
! LDA part
!========================================================================
         POTXC=0
         ilda: IF (TLDA) THEN
            DEXC=0
            IF (ISPIN==1) THEN
               CALL RAD_LDA_XC( R, TREL, LMAX_CALC, RHOT(:,1), &
                    RHO(:,:,1), POTXC(:,:,1), DEXC, DVXC, .FALSE.)
            ELSE
               CALL RAD_LDA_XC_SPIN( R, TREL, LMAX_CALC, RHOT, RHO, &
                    POTXC, DEXC, DVXC, .FALSE.)
            ENDIF
            ! the routines RAD_LDA_XC has already performed a radial integration
            ! incorrectly adding a factor 4*PI
            DEXC=DEXC/(4*PI)
            EXC_LDA= EXC_LDA+(DEXC)*SIM_FAKT

            ! multiply potential with Y_00 and store it
            POTXC(1:R%NMAX,1,:)=POTXC(1:R%NMAX,1,:)*(1.0_q/SCALE)
            IF (LPOT) THEN
               DO LM=1,LMMAX_CALC
                  DO K=1,R%NMAX
                     ! POTXC must be multiplied by Y_00 to get V_xc(r)
                     POT(K,LM,:)=POT(K,LM,:)+YLM(NP,LM)*SIM_FAKT*POTXC(K,1,:)
                  ENDDO
               ENDDO
            ENDIF

         ENDIF ilda
!========================================================================
! GGA if required
!========================================================================
         gga: IF (ISGGA()) THEN
            
            IF (ISPIN==1) THEN
               CALL GRAD(R,RHOT,T1)

               DO K=1,R%NMAX
                  ! norm of gradient for spin up
                  XU  =T1(K,1)*XYZPTS(NP,1)+RHO_ANG(K,1,1)
                  YU  =T1(K,1)*XYZPTS(NP,2)+RHO_ANG(K,1,2)
                  ZU  =T1(K,1)*XYZPTS(NP,3)+RHO_ANG(K,1,3)
                  T1(K,1)=SQRT(XU*XU+YU*YU+ZU*ZU)

                  CALL GGAALL(RHOT(K,1)*AUTOA3,T1(K,1)*AUTOA4,EXT,DEXC1,DVXC1,.NOT.TLDA)
                  EXC_GGA= EXC_GGA+(EXT*RYTOEV)*RHOT(K,1)*(R%R(K)*R%R(K))*R%SI(K)*SIM_FAKT

                  ! develop
                  !             d    f_xc     grad rho
                  !             ------------  --------  = vec mu(r)
                  !             d |grad rho| |grad rho|
                  ! into spherical harmonics
                  ! afterwards vec mu can be reconstructed using
                  ! mu_i(r) =  \sum_L mu_Li(r) Y_L(r)

                  IF (LPOT) THEN
                     DEXC1=DEXC1*RYTOEV
                     DVXC1=DVXC1*RYTOEV*AUTOA

                     DVXC1=DVXC1/ MAX(T1(K,1),1.E-10_q)

                     DO LM=1,LMMAX
                     ! add to local potential to POTXC
                        MU(K,LM,0,1) = MU(K,LM,0,1)+DEXC1*SIM_FAKT*YLM(NP,LM)

                        MU(K,LM,1,1) = MU(K,LM,1,1)+(XU* DVXC1)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,2,1) = MU(K,LM,2,1)+(YU* DVXC1)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,3,1) = MU(K,LM,3,1)+(ZU* DVXC1)*SIM_FAKT*YLM(NP,LM)
                     ENDDO
                  ENDIF
               ENDDO

            ELSE
               DO K=1,R%NMAX
                  TEMP1=RHOT(K,1)
                  RHOT(K,1)=(RHOT(K,1)+RHOT(K,2))/2    ! spin up
                  RHOT(K,2)=(TEMP1-RHOT(K,2))/2        ! spin down
                  
                  RHOT(K,1)=MAX(RHOT(K,1), 1E-7_q)
                  RHOT(K,2)=MAX(RHOT(K,2), 1E-7_q)
                ENDDO

               CALL GRAD(R,RHOT,T1)
               CALL GRAD(R,RHOT(1:R%NMAX,2),T1(1:R%NMAX,2))

               DO K=1,R%NMAX
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
!                  TG=T1(K,1)+T1(K,2) incorrect definition
                  TG=SQRT((XU+XD)**2+(YU+YD)**2+(ZU+ZD)**2)

                  CALL GGASPIN(RHOT(K,1)*AUTOA3,RHOT(K,2)*AUTOA3, &
                       T1(K,1)*AUTOA4,T1(K,2)*AUTOA4,TG*AUTOA4, &
                       EXT,DEXC1,DEXC2,DVXC1,DVXC2,DVC,.NOT.TLDA)
                  EXC_GGA= EXC_GGA+(EXT*RYTOEV)*(RHOT(K,1)+RHOT(K,2))*R%R(K)*R%R(K)*R%SI(K)*SIM_FAKT

                  ! develop
                  !             d    f_xc     grad rho
                  !             ------------  --------  = vec mu(r)
                  !             d |grad rho| |grad rho|
                  ! into spherical harmonics
                  ! afterwards vec mu can be reconstructed using
                  ! mu_i(r) =  \sum_L mu_Li(r) Y_L(r)

                  IF (LPOT) THEN
                     DEXC1=DEXC1*RYTOEV
                     DEXC2=DEXC2*RYTOEV
                     DVXC1=DVXC1*RYTOEV*AUTOA
                     DVXC2=DVXC2*RYTOEV*AUTOA
                     DVC  =DVC  *RYTOEV*AUTOA

                     DVXC1=DVXC1/ MAX(T1(K,1),1.E-10_q)
                     DVXC2=DVXC2/ MAX(T1(K,2),1.E-10_q)
                     DVC  =DVC  / MAX(TG,1.E-10_q)
#ifdef debug
                     IF (K==R%NMAX-10) WRITE(0,'(I4,6F14.7)') NP, & 
                           (XU* DVXC1 + (XU+XD) * DVC ), &
                           (YU* DVXC1 + (YU+YD) * DVC ), &
                           (ZU* DVXC1 + (ZU+ZD) * DVC ), &                  
                           (XD* DVXC2 + (XU+XD) * DVC ), &
                           (YD* DVXC2 + (YU+YD) * DVC ), &
                           (ZD* DVXC2 + (ZU+ZD) * DVC )
#endif
                     DO LM=1,LMMAX
                     ! add to local potential to POTXC
                        MU(K,LM,0,1) = MU(K,LM,0,1)+DEXC1*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,0,2) = MU(K,LM,0,2)+DEXC2*SIM_FAKT*YLM(NP,LM)

                        MU(K,LM,1,1) = MU(K,LM,1,1)+(XU* DVXC1 + (XU+XD) * DVC)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,2,1) = MU(K,LM,2,1)+(YU* DVXC1 + (YU+YD) * DVC)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,3,1) = MU(K,LM,3,1)+(ZU* DVXC1 + (ZU+ZD) * DVC)*SIM_FAKT*YLM(NP,LM)
                               
                        MU(K,LM,1,2) = MU(K,LM,1,2)+(XD* DVXC2 + (XU+XD) * DVC)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,2,2) = MU(K,LM,2,2)+(YD* DVXC2 + (YU+YD) * DVC)*SIM_FAKT*YLM(NP,LM)
                        MU(K,LM,3,2) = MU(K,LM,3,2)+(ZD* DVXC2 + (ZU+ZD) * DVC)*SIM_FAKT*YLM(NP,LM)
                     ENDDO
                  ENDIF
               ENDDO

            ENDIF
         ENDIF gga

      ENDDO points


      ! double counting corrections
      DVXC_LDA=0
      IF (LPOT) THEN
         CALL RAD_POT_IN_RHO( R, ISPIN, LMAX_CALC, RHO, POT, DVXC_LDA)
         DVXC_LDA=DVXC_LDA-DVINI
      ENDIF
!========================================================================
!
! GGA part related to d e_XC / grad n
!
! add -\sum_i Y_L  d (mu_L'i(x) Y_L'(x))/ d x_i to the potential V_L(r)
!
!                          x_i d mu_L'i(|x|)
! -\sum_L'i Y_L(x) Y_L'(x)  -  -----------  + mu_L'i(x)  Y_L nabla_i Y_L'(x)
!                          |x|   d |x|     
!
!========================================================================

      IF (LPOT .AND. ISGGA()) THEN
#ifdef debug
         WRITE(0,*) 'result of the potential fit'
         DO NP=1,NPTS
         WRITE(0,'(I4)',ADVANCE='NO') NP
         DO ISP=1,ISPIN
            DO I=1,3
               K=R%NMAX-10
               XU=0
               DO LM=1,LMMAX
                  XU=XU+YLM(NP,LM)*MU(K,LM,I,ISP)
               ENDDO
               WRITE(0,'(F14.7)',ADVANCE='NO') XU
            ENDDO
         ENDDO
         WRITE(0,*)
         ENDDO
#endif
         DO ISP=1,ISPIN
            DO LM=1,LMMAX_CALC
               DO K=1,R%NMAX
                  POT(K,LM,ISP)=POT(K,LM,ISP)+MU(K,LM,0,ISP)
               ENDDO
            ENDDO
         ENDDO
         ! nabla comment: 
         ! the matrix elements < Y_L| nabla | Y_L'> are non zero
         ! for L=L'+-1
         DO ISP=1,ISPIN
            DO I=1,3
               DO LMP=1,LMMAX
                  CALL GRAD(R,MU(1:R%NMAX,LMP,I,ISP),T1(1:R%NMAX,1))
                  DO LM=1,LMMAX_CALC
                     IF (ABS(YLM_NABLA_YLM(LM,LMP,I))>1E-4 .OR. ABS(YLM_X_YLM(LM,LMP,I))>1E-4) THEN
                        DO K=1,R%NMAX
                           POT(K,LM,ISP)=POT(K,LM,ISP)-( &
                                MU(K,LMP,I,ISP)*YLM_NABLA_YLM(LM,LMP,I)/ R%R(K)+ & 
                                T1(K,1)*YLM_X_YLM(LM,LMP,I) &
                              )
                        ENDDO
                  ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         DEALLOCATE(MU)
      ENDIF

      ! finally calculate the double counting corrections again
      DVXC_GGA=0
      IF (LPOT) THEN
         CALL RAD_POT_IN_RHO( R, ISPIN, LMAX_CALC, RHO, POT, DVXC_GGA)
         DVXC_GGA=DVXC_GGA-DVXC_LDA-DVINI
      ENDIF

      DEALLOCATE(YLM,YLMD,XYZPTS,RADPTS,YLM_NABLA_YLM,YLM_X_YLM,WEIGHT,ABSCIS)
    END SUBROUTINE RAD_GGA_ASPH


!*******************************************************************
!
!  RAD_POT_IN_RHO
!  small helper routine to calculate the integral double counting
!  corrections
!
!    \int V(r) rho(r) r^2 dr d Omega_r
!    \sum_LM \int pot_LM(r) (rho_LM(r) r^2) Y_LM(r) Y_LM(r) dr d Omega_r
!    \sum_LM \int pot_LM(r) (rho_LM(r) r^2) dr 
!    \sum_LM \sum_l POT(r,LM) (RHO(r,LM) SI
!
!*******************************************************************

    SUBROUTINE RAD_POT_IN_RHO( R, ISPIN, LMAX_CALC, RHO, POT, DVXC)
      IMPLICIT NONE
      TYPE (rgrid) :: R
      INTEGER ISPIN
      INTEGER LMAX_CALC
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      REAL(q) :: POT(:,:,:)     ! potential
      REAL(q) :: DVXC
   ! local
      INTEGER LMMAX, LM, K

      LMMAX=(LMAX_CALC+1)**2

      DVXC=0
      IF (ISPIN==1) THEN
         ! non spin polarized case
         DO LM=1,LMMAX
            DO K=1,R%NMAX
               DVXC=DVXC+POT(K,LM,1)*RHO(K,LM,1)*R%SI(K)
            ENDDO
         ENDDO
      ELSE
         ! non spin polarized case
         ! charge is stored as total magnetisation 
         DO LM=1,LMMAX
            DO K=1,R%NMAX
               DVXC=DVXC+POT(K,LM,1)*(RHO(K,LM,1)+RHO(K,LM,2))*R%SI(K)
            ENDDO
         ENDDO
         DO LM=1,LMMAX
            DO K=1,R%NMAX
               DVXC=DVXC+POT(K,LM,2)*(RHO(K,LM,1)-RHO(K,LM,2))*R%SI(K)
            ENDDO
         ENDDO
         DVXC=DVXC/2
      ENDIF

    END SUBROUTINE RAD_POT_IN_RHO

!*******************************************************************
!
!  RAD_META_GGA_ASPH
!  calculate spherical and aspherical contributions to the 
!  metagga exchange and correlation energy. Only the charge density 
!  is treated aspherically. Vasp currently restricts the maximum 
!  angular quantum number to 4. Integration over the sphere is done
!  numerically.
!   
! Robin Hirschl 20010115
!*******************************************************************

    SUBROUTINE RAD_META_GGA_ASPH( R, ISPIN, LMAX, LMAX_CALC, &
         RHO, RHOC, POT, TAU, TAUW, EXC)
      USE constant
      USE asa
      
      IMPLICIT NONE
      INTEGER LMAX, ISPIN, LMAX_CALC
      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      !  RHO,POT(:,:,1) total
      !  RHO,POT(:,:,2) magnetization
      REAL(q) :: TAU(:,:)       ! kinetic energy density
      REAL(q) :: TAUW(:,:)      ! Weizsaecker kinetic energy density
      REAL(q) :: POT(:,:,:)     ! potential
      !  TAU(:,:,1) up   component
      !  TAU(:,:,2) down component
      REAL(q) :: EXC            ! result (total Exc for particular atom)
      TYPE (rgrid) :: R
    ! local variables
      REAL(q), ALLOCATABLE :: RHOT(:,:), RADPTS(:,:), XYZPTS(:,:),YLM(:,:), YLMD(:,:,:)
      REAL(q), ALLOCATABLE :: RHO_ANG(:,:,:),WEIGHT(:),ABSCIS(:)
      INTEGER N,K,LMMAX,LLMAX,LM,NP,NPTS,PHPTS,THPTS,I,J,IFAIL
      REAL(q) DELTAPHI
      REAL(q) SCALE, EVTOH, EXL, ECL, EX, EC, SIM_FAKT
      REAL(q) X,Y,Z,XU,XD,YU,YD,ZU,ZD
      REAL(q) :: T1(R%NMAX),T2(R%NMAX)
      REAL(q) :: RHO1, RHO2, ABSNAB, ABSNABUP, ABSNABDW
      REAL(q) :: TAUU, TAUD, TAUWU, TAUWD, TAUWTOT, TAUDIFF
      REAL(q) TMP(ISPIN,3)
      EXTERNAL GAUSSI2

      SCALE=2*SQRT(PI)   ! 1/Y00
      N=R%NMAX
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5                ! KinEDens eV to Hartree

!========================================================================
! get maximum L number
! find out number of pivot points needed for each sphere,
! calculate the xyz-positions of the pivot points
! and get sherical harmonics
!========================================================================

! LMAX in vasp is (currently) restricted to 6 (f-electrons)
      LLMAX=MIN(6,LMAX_CALC)
      LMMAX=(LLMAX+1)**2
! number of theta and phi pivot points
 ! since Exc=f(a*Yllmax,m) we need more pivor points than theoretically needed to integrate
! Yllmax,m. To be on the safe side (routine is only called once), we multiply by factor of 3
      PHPTS=PHPTS_FACT*(LLMAX+1)
      THPTS=THPTS_FACT*FLOOR(REAL(LLMAX/2+1,KIND=q))
      NPTS=PHPTS*THPTS
      DELTAPHI=REAL(2_q*PI/PHPTS,KIND=q)
! allocate arrays
      ALLOCATE(YLM(NPTS,LMMAX),YLMD(NPTS,LMMAX,3),XYZPTS(NPTS,3),RADPTS(NPTS,2))
      ALLOCATE(RHOT(N,ISPIN),RHO_ANG(N,ISPIN,3),WEIGHT(THPTS),ABSCIS(THPTS))
      RADPTS=0; WEIGHT=0; ABSCIS=0
! save phi positions
      DO I=1,PHPTS
         DO J=1,THPTS
            RADPTS((J-1)*PHPTS+I,2)=(I-1)*DELTAPHI
         ENDDO
      ENDDO
      ! get theta positions (actually get cos(theta))
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
      ! get values of ylm
      YLM=0; YLMD=0
      CALL SETYLM_GRAD(LLMAX,NPTS,YLM,YLMD,XYZPTS(:,1),XYZPTS(:,2),XYZPTS(:,3))

      EXC    =0
      EX=0;EC=0
      
      ! loop over all points in the angular grid
      points: DO NP=1,NPTS
         ! get charge density and gradient components at point NP
         RHOT=0
         RHO_ANG=0
         DO K=1,N
            DO LM=1,LMMAX
               IF (ISPIN==1) THEN
                  RHOT(K,1)=RHOT(K,1)+YLM(NP,LM)*RHO(K,LM,1)
                  RHO_ANG(K,1,:)=RHO_ANG(K,1,:)+YLMD(NP,LM,:)*RHO(K,LM,1)
               ELSE
                  RHOT(K,1)=RHOT(K,1)+YLM(NP,LM)*&
                       (RHO(K,LM,1)+RHO(K,LM,2))/2._q       ! spin up
                  RHOT(K,2)=RHOT(K,2)+YLM(NP,LM)*&
                       (RHO(K,LM,1)-RHO(K,LM,2))/2._q       ! spin down
                  RHO_ANG(K,1,:)=RHO_ANG(K,1,:)+YLMD(NP,LM,:)*&
                       (RHO(K,LM,1)+RHO(K,LM,2))/2._q       ! spin up
                  RHO_ANG(K,2,:)=RHO_ANG(K,2,:)+YLMD(NP,LM,:)*&
                       (RHO(K,LM,1)-RHO(K,LM,2))/2._q       ! spin down
               ENDIF
            ENDDO
            ! add core charge (spherical)
            IF (ISPIN==1) THEN
               RHOT(K,1)=(RHOT(K,1)+RHOC(K)*YLM(NP,1))/(R%R(K)*R%R(K))
            ELSE
               RHOT(K,1)=(RHOT(K,1)+RHOC(K)*YLM(NP,1)/2._q)/(R%R(K)*R%R(K))
               RHOT(K,2)=(RHOT(K,2)+RHOC(K)*YLM(NP,1)/2._q)/(R%R(K)*R%R(K))
            ENDIF
            ! divide gradient by R**2
            RHO_ANG(K,:,:)=RHO_ANG(K,:,:)/(R%R(K)*R%R(K))
            ! we have calculated the derivative d Y_lm(x')/ d x'
            !  x' = x / || x || ;   dx'_i / dx_j = 1/r (delta_ij - x_i' x_j')
            TMP=RHO_ANG(K,:,:)/R%R(K)
            DO I=1,3
               DO J=1,3
                  TMP(:,I)=TMP(:,I)-RHO_ANG(K,:,J)/R%R(K)*XYZPTS(NP,I)*XYZPTS(NP,J)
               ENDDO
            ENDDO
            RHO_ANG(K,:,:)=TMP
         ENDDO

!========================================================================
! exchange correlation energy
!========================================================================         
         IF (ISPIN==1) THEN
            CALL GRAD(R,RHOT(1:N,1),T1)            
         ELSE
            CALL GRAD(R,RHOT(1:N,1),T1)
            CALL GRAD(R,RHOT(1:N,2),T2)
         ENDIF
         
!========================================================================
! metagga calculation
!========================================================================
         DO K=1,N
            IF (ISPIN==1) THEN
               RHO1=MAX(RHOT(K,1)/2._q,1.E-10_q)
               RHO2=RHO1
               ! norm of gradient for spin up
               X   =T1(K)*XYZPTS(NP,1)+RHO_ANG(K,1,1)
               Y   =T1(K)*XYZPTS(NP,2)+RHO_ANG(K,1,2)
               Z   =T1(K)*XYZPTS(NP,3)+RHO_ANG(K,1,3)
               T1(K)=SQRT(X*X+Y*Y+Z*Z)
               ABSNAB=T1(K)
               ABSNABUP=T1(K)/2._q
               ABSNABDW=ABSNABUP
               TAUU =MAX(TAU(K,1)/(2._q*SCALE*R%R(K)*R%R(K)),1.E-10_q)
               TAUWU=MAX(TAUW(K,1)/(2._q*SCALE*R%R(K)*R%R(K)),1.E-10_q)
               TAUD= TAUU
               TAUWD=TAUWU        
            ELSE
               RHO1=MAX(RHOT(K,1),1.E-10_q)
               RHO2=MAX(RHOT(K,2),1.E-10_q)
               ! norm of gradient for spin up
               XU  =T1(K)*XYZPTS(NP,1)+RHO_ANG(K,1,1)
               YU  =T1(K)*XYZPTS(NP,2)+RHO_ANG(K,1,2)
               ZU  =T1(K)*XYZPTS(NP,3)+RHO_ANG(K,1,3)
               T1(K)=SQRT(XU*XU+YU*YU+ZU*ZU)
               ! norm of gradient for spin down
               XD  =T2(K)*XYZPTS(NP,1)+RHO_ANG(K,2,1)
               YD  =T2(K)*XYZPTS(NP,2)+RHO_ANG(K,2,2)
               ZD  =T2(K)*XYZPTS(NP,3)+RHO_ANG(K,2,3)
               T2(K)=SQRT(XD*XD+YD*YD+ZD*ZD)
!               ABSNAB=T1(K)+T2(K) incorrect
               ABSNAB=SQRT((XU+XD)**2+(YU+YD)**2+(ZU+ZD)**2)
               ABSNABUP=T1(K)
               ABSNABDW=T2(K)
               TAUU=MAX(TAU(K,1)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
               TAUD=MAX(TAU(K,2)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
               TAUWU=MAX(TAUW(K,1)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
               TAUWD=MAX(TAUW(K,2)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
            ENDIF
            ! All parameters for subroutine Metagga must be passed in Hartree
            
            CALL METAGGA(RHO1*AUTOA3,RHO2*AUTOA3, &
                 &         ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
                 &         TAUU*EVTOH,TAUD*EVTOH, &
                 &         TAUWU*EVTOH,TAUWD*EVTOH,EXL,ECL,K)
            SIM_FAKT=R%SI(K)*DELTAPHI*WEIGHT((INT((NP-1)/PHPTS)+1))
            ! factor 2 for Hartree -> Rydberg conversion
            EX=EX+2*EXL*RYTOEV/AUTOA3*R%R(K)*R%R(K)*SIM_FAKT
            EC=EC+2*ECL*RYTOEV/AUTOA3*R%R(K)*R%R(K)*SIM_FAKT
         ENDDO
      ENDDO points
      EXC=EX+EC      
      DEALLOCATE(YLM,YLMD,XYZPTS,RADPTS,RHOT,RHO_ANG,WEIGHT,ABSCIS)
    END SUBROUTINE RAD_META_GGA_ASPH


!*******************************************************************
!
!  RAD_META_GGA
!  calculate the spherical contributions to the metagga exchange and 
!  correlation energy. 
!
! Robin Hirschl 20001223
!*******************************************************************

    SUBROUTINE RAD_META_GGA( R, ISPIN, LMAX, LMAX_CALC, &
         RHO, RHOC, POT, TAU, TAUW, EXC)
      USE constant
      IMPLICIT NONE
!      IMPLICIT REAL(q) (A-H,O-Z)
      INTEGER LMAX, ISPIN, LMAX_CALC
      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      REAL(q) :: RHO(:,:,:)     ! charge distribution see above
      !  RHO,POT(:,:,1) total
      !  RHO,POT(:,:,2) magnetization
      REAL(q) :: TAU(:,:)       ! kinetic energy density
      REAL(q) :: TAUW(:,:)      ! Weizsaecker kinetic energy density
      REAL(q) :: POT(:,:,:)     ! potential
      ! TAU(:,:,1) up   component
      ! TAU(:,:,2) down component
      REAL(q) :: EXC            ! result (total Exc for particular atom)
      TYPE (rgrid) :: R
    ! local variables
      REAL(q) RHOT(R%NMAX,ISPIN)
      INTEGER N,K
      REAL(q) SCALE, EVTOH, EXL, ECL, EX, EC, SIM_FAKT
      REAL(q) :: T1(R%NMAX),T2(R%NMAX)
      REAL(q) :: RHO1, RHO2, ABSNAB, ABSNABUP, ABSNABDW
      REAL(q) :: TAUU, TAUD, TAUWU, TAUWD, TAUWTOT, TAUDIFF
      SCALE=2*SQRT(PI)
      N=R%NMAX
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5                ! KinEDens eV to Hartree

!========================================================================
! exchange correlation energy
!========================================================================
      EXC    =0
      EX=0;EC=0

      IF (ISPIN==1) THEN
        DO K=1,N
! charge density rho_0 Y(0)
          RHOT(K,1)=(RHO(K,1,1)+RHOC(K))/ (SCALE*R%R(K)*R%R(K))
        ENDDO
        CALL GRAD(R,RHOT,T1)

      ELSE
        DO K=1,N
! charge density n_0=rho_0 Y(0) spin up
         RHOT(K,1)=(RHO(K,1,1)+RHO(K,1,2)+RHOC(K))/ (2*SCALE*R%R(K)*R%R(K)) 
! charge density n_0=rho_0 Y(0) spin down
         RHOT(K,2)=(RHO(K,1,1)-RHO(K,1,2)+RHOC(K))/ (2*SCALE*R%R(K)*R%R(K))
        ENDDO
        CALL GRAD(R,RHOT,T1)
        CALL GRAD(R,RHOT(1:N,2),T2)
      ENDIF

!========================================================================
! metagga calculation
!========================================================================
      DO K=1,N
         IF (ISPIN==1) THEN
            RHO1=MAX(RHOT(K,1)/2._q,1.E-10_q)
            RHO2=RHO1
            ABSNAB=T1(K)
            ABSNABUP=T1(K)/2._q
            ABSNABDW=ABSNABUP
            TAUU =MAX(TAU(K,1)/(2._q*SCALE*R%R(K)*R%R(K)),1.E-10_q)
            TAUWU=MAX(TAUW(K,1)/(2._q*SCALE*R%R(K)*R%R(K)),1.E-10_q)

            TAUD= TAUU
            TAUWD=TAUWU        
        ELSE
            RHO1=MAX(RHOT(K,1),1.E-10_q)
            RHO2=MAX(RHOT(K,2),1.E-10_q)
            ABSNAB=T1(K)+T2(K)
            ABSNABUP=T1(K)
            ABSNABDW=T2(K)
            TAUU=MAX(TAU(K,1)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
            TAUD=MAX(TAU(K,2)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
            TAUWU=MAX(TAUW(K,1)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
            TAUWD=MAX(TAUW(K,2)/(SCALE*R%R(K)*R%R(K)),1.E-10_q)
         ENDIF
! All parameters for subroutine Metagga must be passed in Hartree

         CALL METAGGA(RHO1*AUTOA3,RHO2*AUTOA3, &
              &               ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
              &           TAUU*EVTOH,TAUD*EVTOH, &
              &           TAUWU*EVTOH,TAUWD*EVTOH,EXL,ECL,K)
         SIM_FAKT=R%SI(K)*SCALE
! factor 2 for Hartree -> Rydberg conversion
         EX=EX+2*EXL*RYTOEV/AUTOA3*SCALE*R%R(K)*R%R(K)*SIM_FAKT
         EC=EC+2*ECL*RYTOEV/AUTOA3*SCALE*R%R(K)*R%R(K)*SIM_FAKT
      ENDDO

!      WRITE(*,*)
!      WRITE(*,'(2(A,F14.6))') 'Exchange energy    eV:',EX,'  Hartree:',EX/(2*RYTOEV)
!      WRITE(*,'(2(A,F14.6))') 'Correlation energy eV:',EC,'  Hartree:',EC/(2*RYTOEV)
!      WRITE(*,*)
     
      EXC=EX+EC
    END SUBROUTINE RAD_META_GGA


!*******************************************************************
!
!  RAD_POT_WEIGHT
!  because POT will be required only for integration we
!  multiply now with the weights
!
!*******************************************************************

    SUBROUTINE RAD_POT_WEIGHT( R, ISPIN, LMAX, POT)
      IMPLICIT NONE
      REAL(q) :: POT(:,:,:)     ! radial potential
      TYPE (rgrid) :: R
      INTEGER LMAX,ISPIN,I,L,M,LM,K

      DO I=1,ISPIN
      DO L=0,LMAX
      DO M=0,2*L
         LM=L*L+M+1
         DO K=1,R%NMAX
           POT(K,LM,I)=POT(K,LM,I)*R%SI(K)
         ENDDO      
      ENDDO
      ENDDO
      ENDDO
    END SUBROUTINE RAD_POT_WEIGHT

!*******************************************************************
!
! Coloumb potential
! we use the equation
! POT_lm(R) = 4 pi/(2l+1)[ 1/R^(l+1) \int_0^R RHO_lm(r) r^(l) dr
!                       + R^l \int_R^Infinity RHO_lm(r) r^(-l-1) dr ]
!           = 4 pi \int_0^Infinity dr RHO_lm(r) (r,R)<^l / (r,R)>^(l+1)
!
!  rho(r) = \sum_lm  RHO_lm(r) Y_lm(r) / r^2
!    V(r) = \sum_lm  POT_lm(r) Y_lm(r)
!
! which can be obtained by partial integration of
! V  = \int dR 1/R^2  \int  rho(r) dr
!
!*******************************************************************


      SUBROUTINE RAD_POT_HAR(LL,R,POT,RHO,DHARTREE)
      USE constant
      IMPLICIT NONE

      TYPE (rgrid) :: R
      INTEGER LL
      REAL(q) :: RHO(:),POT(:)
      REAL(q) T1(R%NMAX),T2(R%NMAX),V1(R%NMAX),V2(R%NMAX),RL(R%NMAX)
      REAL(q) DHARTREE,H3,EXT
      INTEGER N,I,K,L

      N=R%NMAX

      I=0
    ! additional factor r stems from logarithmic grid
      DO K=N,1,-1
         RL(K)=R%R(K)**LL
         I=I+1
         T2(I)=RHO(K)/RL(K)
         T1(I)=RL(K)*R%R(K)*RHO(K)
      ENDDO
      H3=R%H/ 3.0_q
    ! integrate inward (assuming zero moment for grid point NMAX)
    ! V1 = \int_R^Inf RHO(r) r^l dr
      V1(1)=0
      DO L=3,N,2
        V1(L)  =V1(L-2)+H3*(T1(L-2)+4.0_q*T1(L-1)+T1(L))
        V1(L-1)=V1(L-2)+H3*(1.25_q*T1(L-2)+2.0_q*T1(L-1)-0.25_q*T1(L))
      ENDDO
      IF (MOD(N,2)==0) V1(N)=V1(N-2)+H3*(T1(N-2)+4.0_q*T1(N-1)+T1(N))
    ! V2 = \int_R^Inf RHO(r) r^(-l-1) dr
      V2(1)=0
      DO L=3,N,2
         V2(L)  =V2(L-2)+H3*(T2(L-2)+4.0_q*T2(L-1)+T2(L))
         V2(L-1)=V2(L-2)+H3*(1.25_q*T2(L-2)+2.0_q*T2(L-1)-0.25_q*T2(L))
      ENDDO
      IF (MOD(N,2)==0) V2(N)=V2(N-2)+H3*(T2(N-2)+4.0_q*T2(N-1)+T2(N))

     ! EXT total moment (i.e. charge)
     ! EXT-V1(K) = \int_0^R RHO(r) r^l dr 
      EXT=V1(N)
      I=0
      DO K=N,1,-1
         I=I+1
         POT(I)=(V2(K)*RL(I)+(EXT-V1(K))/(R%R(I)*RL(I)))*(FELECT*4*PI/(2*LL+1))
      ENDDO
    ! double counting corrections are simply obtained by summing over grid
    ! since int Y_lm(Omega_r) Y_lm(Omega_r) d_Omega_r= 1
      DHARTREE=0
      DO K=1,N
         DHARTREE=DHARTREE+POT(K)*(RHO(K)*R%SI(K))
      ENDDO
      END SUBROUTINE




!*******************************************************************
!
! calculate the LDA contribution to the exchange correlation
! energy and to the potentials
! non spin polarised case and below the spinpolarised case
!
! following Bloechl we use a Taylor expansion of the
! exchange correlation energy around the spherical density
!  E_xc = \int d Omega r^2 dr
!'        eps_xc( n_0(r)) + 1/2 v_xc'(n_0) Y_L rho_L(r) Y_L rho_L(r)
! (tick denotes derivative w.r.t. n_0)
! the potential is defined as
!  v_L(r) = var E_xc / var n_L(r) / Y_L
! (since  the factor  Y_L is applied later in the program (see above)
!  we have to divide by Y_L)
!
!
!*******************************************************************

   SUBROUTINE RAD_LDA_XC( R, TREL, LMAX_CALC, RHOT, RHO, POT, DEXC, DVXC, LASPH)
      USE constant
      USE setexm
      IMPLICIT NONE
      TYPE (rgrid) :: R
      LOGICAL TREL      ! relativistic corrections to exchange
      LOGICAL LASPH     ! wether to calculate aspherical corrections
      INTEGER LMAX_CALC ! maximum L
      REAL(q) RHOT(:)   ! spherical charge + core charge density
      REAL(q) RHO(:,:)  ! charge distribution see above
      REAL(q) POT(:,:)  ! potential
      REAL(q) DEXC      ! exchange correlation energy
      REAL(q) DVXC      ! V_xc rho
! local
      REAL(q) EXCA(4)
      REAL(q) SIM_FAKT, RHOP, EXT, VXT, DVXC1, RHOPA, SCALE
      INTEGER K,LM

      SCALE=2*SQRT(PI)
      DO K=1,R%NMAX
! MM
!        IF (RHOT(K) <= 0 ) CYCLE
         IF (RHOT(K) <= 1E-99_q ) CYCLE
! MM
         CALL EXCOR_DER_PARA(RHOT(K),LMAX_CALC,EXCA,TREL)

         SIM_FAKT= R%SI(K)*R%R(K)*R%R(K)
         RHOP    = RHO(K,1)/(R%R(K)*R%R(K))

         ! store v_xc(r) / Y_0
         POT(K,1)=POT(K,1)+EXCA(2)*SCALE
         !  \int v_xc(r)  rho_0(r) Y_0 4 Pi r^2 dr
         DVXC    =DVXC    +EXCA(2) *  RHOP*SCALE*SIM_FAKT
         !  \int eps_xc(r) 4 pi r^2 dr
         DEXC    =DEXC    +EXCA(1) *4*PI*SIM_FAKT
         EXT=0
         VXT=0
         IF (LASPH) THEN
         DO LM=2,(LMAX_CALC+1)*(LMAX_CALC+1)
            RHOPA=RHO(K,LM)/ (R%R(K)*R%R(K)) ! rho_L
            ! corrections to energy
            ! 1/2 \int v_xc p(n_0) Y_L rho_L(r) Y_L rho_L(r) d Omega r^2 dr
            !  = 1/2 \int v_xc p(n_0) rho_L(r) rho_L(r) r^2 dr
            EXT =EXT +(EXCA(3)*RHOPA*RHOPA)/2
            ! correction to L=0 potential
            ! 1/2  \int v_xc''(n_0) Y_L rho_L(r) Y_L rho_L(r) d Omega =
            ! \int Y_0 (v_xc''(n_0) rho_L(r) rho_L(r) Y_0/2) d Omega
            DVXC1 = (EXCA(4)*RHOPA*RHOPA)/2/SCALE
            ! \int 1/2 v_xc''(n_0) Y(L) rho_L(r) Y(L) rho_L(r) Y(0) rho_0 d Omega r^2 dr
            ! =\int  Y_0/2 v_xc''(n_0) rho_L(r) rho_L(r)  r^2 dr
            VXT=VXT + DVXC1*RHOP
            POT(K,1)=POT(K,1)+DVXC1

            ! L/=0 potentials
            ! \int Y_L (v_xc p(n_0) \rho_L)
            POT(K,LM)=POT(K,LM)+ (EXCA(3)*RHOPA)
            ! \int v_xc p(n_0) \rho_L Y_L \rho_L Y_L d Omega r^2 dr
            ! =  \int v_xc p(n_0)  \rho_L \rho_L r^2 dr
            VXT=VXT+EXCA(3)*RHOPA*RHOPA
         ENDDO
         ENDIF
         DEXC=DEXC + EXT*SIM_FAKT
         DVXC=DVXC + VXT*SIM_FAKT
      ENDDO
   END SUBROUTINE RAD_LDA_XC



   SUBROUTINE RAD_LDA_XC_SPIN( R, TREL, LMAX_CALC, RHOT, RHO, POT, DEXC, DVXC, LASPH)
      USE constant
      USE setexm
      IMPLICIT NONE
      TYPE (rgrid) :: R
      LOGICAL TREL      ! relativistic corrections to exchange
      LOGICAL LASPH     ! wether to calculate aspherical corrections
      INTEGER LMAX_CALC ! maximum L
      REAL(q) RHOT(:,:) ! spherical charge + core charge density
      REAL(q) RHO(:,:,:)! charge distribution see above
      REAL(q) POT(:,:,:)! potential
      REAL(q) DEXC      ! exchange correlation energy
      REAL(q) DVXC      ! V_xc rho
! local
      REAL(q) EXC,EXCD(2),EXCDD(2,2),EXCDDD(2,2,2)
      REAL(q) SIM_FAKT, RHOP, RHOM, EXT, VXT, DVXC1, DVXC2, RHOPA, RHOMA, SCALE, RHOMA_, RHOPA_
      INTEGER K,LM

      SCALE=2*SQRT(PI)

      DO K=1,R%NMAX
         IF (RHOT(K,1) <= 0) CYCLE

         CALL EXCOR_DER(0.5*(RHOT(K,1)+RHOT(K,2)),0.5*(RHOT(K,1)-RHOT(K,2)), &
              LMAX_CALC,EXC,EXCD,EXCDD,EXCDDD,TREL)

         ! store v_xc(r) / Y_0
         POT(K,1,1)=POT(K,1,1)+(EXCD(1)*SCALE)                 ! up potential
         POT(K,1,2)=POT(K,1,2)+(EXCD(2)*SCALE)                 ! down potential pot_L(r)

         RHOP =0.5_q*(RHO(K,1,1)+RHO(K,1,2))/ (R%R(K)*R%R(K))  ! up valence density   rho_0
         RHOM =0.5_q*(RHO(K,1,1)-RHO(K,1,2))/ (R%R(K)*R%R(K))  ! down valence density

         SIM_FAKT= R%SI(K)*R%R(K)*R%R(K)                       ! r^2 dr

         !  \int eps_xc(r) 4 pi r^2 dr
         DEXC =DEXC  +EXC * SIM_FAKT *4*PI
         !  \int v_xc(r)  rho_0(r) Y_0 4 Pi r^2 dr
         DVXC =DVXC  +(EXCD(1)*RHOP + EXCD(2)*RHOM)* SIM_FAKT*SCALE

         ! L/=0 terms
         ! mind that the factors Y_L are not stored in POT (see above)
         EXT=0
         VXT=0

         IF (LASPH) THEN
         DO LM=2,(LMAX_CALC+1)*(LMAX_CALC+1)
            RHOPA=0.5_q*(RHO(K,LM,1)+RHO(K,LM,2))/ (R%R(K)*R%R(K)) ! rho_L
            RHOMA=0.5_q*(RHO(K,LM,1)-RHO(K,LM,2))/ (R%R(K)*R%R(K))
            ! corrections to energy
            ! 1/2 \int v_xc p(n_0) Y_L rho_L(r) Y_L rho_L(r) d Omega r^2 dr
            !  = 1/2 \int v_xc p(n_0) rho_L(r) rho_L(r) r^2 dr
            EXT =EXT +(EXCDD(1,1)*RHOPA*RHOPA + EXCDD(2,2)*RHOMA*RHOMA + &
                     2*EXCDD(1,2)*RHOPA*RHOMA)/2

            ! correction to L=0 potential
            ! 1/2 v_xc''(n_0) Y_L rho_L(r) Y_L rho_L(r)
            ! spherically averaged:  Y_0 (v_xc''(n_0) rho_L(r) rho_L(r) Y_0/2)
             DVXC1 = (EXCDDD(1,1,1)*RHOPA*RHOPA + EXCDDD(1,2,2)*RHOMA*RHOMA + &
                        2*EXCDDD(1,1,2)*RHOPA*RHOMA)/2/SCALE
             DVXC2 = (EXCDDD(2,1,1)*RHOPA*RHOPA + EXCDDD(2,2,2)*RHOMA*RHOMA + &
                        2*EXCDDD(2,1,2)*RHOPA*RHOMA)/2/SCALE
            ! \int 1/2 v_xc''(n_0) Y(L) rho_L(r) Y(L) rho_L(r) Y(0) rho_0 d Omega r^2 dr
            ! =\int  Y_0/2 v_xc''(n_0) rho_L(r) rho_L(r)  r^2 dr
             POT(K,1,1)=POT(K,1,1)+DVXC1
             POT(K,1,2)=POT(K,1,2)+DVXC2
            ! double counting correction
             VXT=VXT+(DVXC1*RHOP) + (DVXC2*RHOM)

            ! L/=0 potentials
            ! Y_L (v_xc p(n_0) \rho_L)
             POT(K,LM,1)=POT(K,LM,1)+(EXCDD(1,1)*RHOPA+EXCDD(1,2)*RHOMA)
             POT(K,LM,2)=POT(K,LM,2)+(EXCDD(1,2)*RHOPA+EXCDD(2,2)*RHOMA)

            ! double counting correction
            ! \int v_xc p(n_0) \rho_L Y_L \rho_L Y_L d Omega r^2 dr
            ! =  \int v_xc p(n_0)  \rho_L \rho_L r^2 dr
             VXT=VXT+(EXCDD(1,1)*RHOPA*RHOPA +EXCDD(2,2)*RHOMA*RHOMA + &
                    2*EXCDD(1,2)*RHOPA*RHOMA)
         ENDDO
         ENDIF



         DEXC=DEXC + EXT*SIM_FAKT
         DVXC=DVXC + VXT*SIM_FAKT
      ENDDO
    END SUBROUTINE RAD_LDA_XC_SPIN

!*******************************************************************
!
! calculate the GGA contribution to the exchange correlation
! energy and to the potentials
! only spherical contributions are accounted for
!
!*******************************************************************

    SUBROUTINE RAD_GGA_XC( R, TLDA, RHOT, RHO, POT, DEXC_GGA, DVXC_GGA)
      USE constant
      IMPLICIT NONE
      TYPE (rgrid) :: R
      LOGICAL TLDA      ! include LDA contributions  (usually .FALSE.)
      REAL(q) RHOT(:)   ! spherical charge + core charge density
      REAL(q) RHO(:)    ! spherical charge
      REAL(q) POT(:)    ! potential
      REAL(q) DEXC_GGA  ! exchange correlation energy
      REAL(q) DVXC_GGA  ! V_xc rho
! local
      REAL(q) SIM_FAKT, RHOP, EXT, VXT, DEXC1, DVXC1, SCALE
      REAL(q) T1(R%NMAX),T2(R%NMAX),V1(R%NMAX)
      INTEGER K

      SCALE=2*SQRT(PI)


      CALL GRAD(R,RHOT,T1)
      DO K=1,R%NMAX
#ifdef vector
         CALL GGA91_WB(RHOT(K)*AUTOA3,T1(K)*AUTOA4,EXT,DEXC1,DVXC1)
#else
         CALL GGAALL(RHOT(K)*AUTOA3,T1(K)*AUTOA4,EXT,DEXC1,DVXC1,.NOT.TLDA)
#endif
         SIM_FAKT=R%SI(K)*SCALE
         DEXC_GGA=DEXC_GGA+(EXT*RYTOEV)*RHOT(K)*(SCALE*R%R(K)*R%R(K))*SIM_FAKT
         
         !  store d f/ d (d rho )   in T2
         T2(K) = DVXC1*RYTOEV*AUTOA
         !  store d f/ d rho  in T1
         T1(K) = DEXC1*RYTOEV
      ENDDO
      CALL GRAD(R,T2,V1)
      
      DO K=1,R%NMAX
         VXT     = T1(K) - V1(K) - 2*T2(K)/ R%R(K)
         SIM_FAKT=R%SI(K)*SCALE
         DVXC_GGA=DVXC_GGA+VXT*RHO(K)*SIM_FAKT
         POT(K)=POT(K)+VXT*SCALE
      ENDDO

    END SUBROUTINE RAD_GGA_XC


    SUBROUTINE RAD_GGA_XC_SPIN( R, TLDA, RHOT, RHO, POT, DEXC_GGA, DVXC_GGA)
      USE constant
      IMPLICIT NONE
      TYPE (rgrid) :: R
      LOGICAL TLDA      ! include LDA contributions  (usually .FALSE.)
      REAL(q) RHOT(:,:) ! up and down charge density (including core)
      REAL(q) RHO(:,:)  ! total charge and magnetisation
      REAL(q) POT(:,:)  ! potential
      REAL(q) DEXC_GGA  ! exchange correlation energy
      REAL(q) DVXC_GGA  ! V_xc rho
! local
      REAL(q) SIM_FAKT, RHOP, EXT, VXT, DEXC1, DEXC2, DVXC1, DVXC2, DVC, SCALE
      REAL(q) T1(R%NMAX,2),T2(R%NMAX,2),V1(R%NMAX,2)
      INTEGER K

      SCALE=2*SQRT(PI)

      CALL GRAD(R,RHOT,T1)
      CALL GRAD(R,RHOT(1:R%NMAX,2),T1(1:R%NMAX,2))

      DO K=1,R%NMAX
         CALL GGASPIN(RHOT(K,1)*AUTOA3,RHOT(K,2)*AUTOA3, &
     &              T1(K,1)*AUTOA4,T1(K,2)*AUTOA4,(T1(K,1)+T1(K,2))*AUTOA4, &
     &                EXT,DEXC1,DEXC2,DVXC1,DVXC2,DVC,.NOT.TLDA)
         DVXC1=DVXC1+DVC
         DVXC2=DVXC2+DVC

         SIM_FAKT=R%SI(K)*SCALE
         DEXC_GGA=DEXC_GGA+(EXT*RYTOEV)*(RHOT(K,1)+RHOT(K,2))*SCALE*R%R(K)*R%R(K)*SIM_FAKT

         !   store d f/ d ( d rho )  in T2
         T2(K,1)  = DVXC1*RYTOEV*AUTOA
         T2(K,2)  = DVXC2*RYTOEV*AUTOA

         !   store d f/ d rho  in T1
         T1(K,1) = DEXC1*RYTOEV
         T1(K,2) = DEXC2*RYTOEV
      ENDDO

      CALL GRAD(R,T2,V1)
      CALL GRAD(R,T2(1,2),V1(1,2))

      DO K=1,R%NMAX
         SIM_FAKT=R%SI(K)*SCALE
         
         VXT     = T1(K,1)- V1(K,1) - 2*T2(K,1)/ R%R(K)
         DVXC_GGA=DVXC_GGA+VXT* (RHO(K,1)+RHO(K,2))*0.5_q*SIM_FAKT
         POT(K,1)=POT(K,1)+VXT*SCALE
         
         VXT     = T1(K,2)- V1(K,2) - 2*T2(K,2)/ R%R(K)
         DVXC_GGA=DVXC_GGA+VXT* (RHO(K,1)-RHO(K,2))*0.5_q*SIM_FAKT
         POT(K,2)=POT(K,2)+VXT*SCALE
      ENDDO

    END SUBROUTINE RAD_GGA_XC_SPIN


!*******************************************************************
!
! this subroutine calculates the absolute magnitude of the 
! magnetization density for non collinear calculations
!
!*******************************************************************

      SUBROUTINE RAD_MAG_DENSITY( RHO, RHOCOL, LMAX, R)
      USE constant
      IMPLICIT NONE
      REAL(q) :: RHO(:,:,:)     ! density
      REAL(q) :: RHOCOL(:,:,:)  ! total density
      TYPE (rgrid) :: R
      INTEGER LMAX,ISPIN,I,L,M,LM,K
      REAL(q) AVMAG(3),ABS_AVMAG
      !
      ! L=0 just copy rho, and calculate the absolute
      ! magnitude of m
      LM=1
      AVMAG=0
      DO K=1,R%NMAX
         RHOCOL(K,LM,1)=RHO(K,LM,1)
         RHOCOL(K,LM,2)=SQRT(RHO(K,LM,2)*RHO(K,LM,2)+RHO(K,LM,3)*RHO(K,LM,3)+ &
              RHO(K,LM,4)*RHO(K,LM,4))
         AVMAG(1)=AVMAG(1)+RHO(K,LM,2)*R%SI(K)
         AVMAG(2)=AVMAG(2)+RHO(K,LM,3)*R%SI(K)
         AVMAG(3)=AVMAG(3)+RHO(K,LM,4)*R%SI(K)
      ENDDO
      ABS_AVMAG=SQRT(AVMAG(1)**2+AVMAG(2)**2+AVMAG(3)**2)

      !
      ! L/=0 calculate  m_0 / |m_0| m_lm
      !
      IF (.NOT. USE_AVERAGE_MAGNETISATION) THEN
         DO L=1,LMAX
         DO M=0,2*L
            LM=L*L+M+1
            DO K=1,R%NMAX
               RHOCOL(K,LM,1)=RHO(K,LM,1)
               RHOCOL(K,LM,2)=(RHO(K,1,2)*RHO(K,LM,2)+RHO(K,1,3)*RHO(K,LM,3)+ &
               RHO(K,1,4)*RHO(K,LM,4))/MAX(RHOCOL(K,1,2),MAGMIN)
            ENDDO
         ENDDO
         ENDDO
      ELSE
         DO L=1,LMAX
         DO M=0,2*L
            LM=L*L+M+1
            DO K=1,R%NMAX
               RHOCOL(K,LM,1)=RHO(K,LM,1)
               RHOCOL(K,LM,2)=(AVMAG(1)*RHO(K,LM,2)+AVMAG(2)*RHO(K,LM,3)+ &
               AVMAG(3)*RHO(K,LM,4))/MAX(ABS_AVMAG,MAGMIN)
            ENDDO
         ENDDO
         ENDDO
      ENDIF

      END SUBROUTINE

      SUBROUTINE RAD_MAG_DENSITY_KINDENS(RHO, RHOCOL, KINDENS, KINDENSCOL, LMAX, LMAX_TAU, R, RHOPS)
      USE constant
      IMPLICIT NONE
      REAL(q) :: RHO(:,:,:)        ! density
      REAL(q) :: RHOCOL(:,:,:)     ! total density
      REAL(q) :: KINDENS(:,:,:)    ! kinetic energy density (total,magnetization)
      REAL(q) :: KINDENSCOL(:,:,:) ! return: collinear kinetic energy density (up,down)
      TYPE (rgrid) :: R
      REAL(q), OPTIONAL :: RHOPS(:,:,:) ! the non-augmented pseudo density (treated like RHO) 
      INTEGER LMAX
      INTEGER LMAX_TAU
      ! local variables
      INTEGER ISPIN,I,L,M,LM,K
      REAL(q) AVMAG(3),ABS_AVMAG,A,B

      RHOCOL=0; KINDENSCOL=0

      AVMAG=0
      DO K=1,R%NMAX
         AVMAG(1)=AVMAG(1)+RHO(K,1,2)*R%SI(K)
         AVMAG(2)=AVMAG(2)+RHO(K,1,3)*R%SI(K)
         AVMAG(3)=AVMAG(3)+RHO(K,1,4)*R%SI(K)
      ENDDO
      ABS_AVMAG=SQRT(AVMAG(1)**2+AVMAG(2)**2+AVMAG(3)**2)

      !
      ! L=0 component of the kinetic energy density is
      ! computed seperately, so that at least this component
      ! is set in case USE_AVERAGE_MAGNETISATION=.FALSE.
      DO K=1,R%NMAX
         A=KINDENS(K,1,1)
         B=(AVMAG(1)*KINDENS(K,1,2)+AVMAG(2)*KINDENS(K,1,3)+ &
        &    AVMAG(3)*KINDENS(K,1,4))/MAX(ABS_AVMAG,MAGMIN)
         KINDENSCOL(K,1,1)=(A+B)/2
         KINDENSCOL(K,1,2)=(A-B)/2
      ENDDO

      IF (.NOT. USE_AVERAGE_MAGNETISATION) THEN
         !
         ! L=0 just copy rho, and calculate the absolute
         ! magnitude of m
         DO K=1,R%NMAX
            RHOCOL(K,1,1)=RHO(K,1,1)
            RHOCOL(K,1,2)=SQRT(RHO(K,1,2)*RHO(K,1,2)+RHO(K,1,3)*RHO(K,1,3)+ &
                 RHO(K,1,4)*RHO(K,1,4))
         ENDDO
         !
         ! L/=0 calculate  m_0 / |m_0| m_lm
         !
         DO L=1,LMAX
         DO M=0,2*L
            LM=L*L+M+1
            DO K=1,R%NMAX
               RHOCOL(K,LM,1)=RHO(K,LM,1)
               RHOCOL(K,LM,2)=(RHO(K,1,2)*RHO(K,LM,2)+RHO(K,1,3)*RHO(K,LM,3)+ &
              &   RHO(K,1,4)*RHO(K,LM,4))/MAX(RHOCOL(K,1,2),MAGMIN)
            ENDDO
         ENDDO
         ENDDO
         ! possibly we have a non-augmented pseudo charge density as well
         IF (PRESENT(RHOPS)) THEN
            DO L=0,LMAX
            DO M=0,2*L
               LM=L*L+M+1
               DO K=1,R%NMAX
                  RHOPS(K,LM,1)=RHOPS(K,LM,1)
                  RHOPS(K,LM,2)=(RHO(K,1,2)*RHOPS(K,LM,2)+RHO(K,1,3)*RHOPS(K,LM,3)+ &
                 &   RHO(K,1,4)*RHOPS(K,LM,4))/MAX(RHOCOL(K,1,2),MAGMIN)
               ENDDO
            ENDDO
            ENDDO
         ENDIF
      ELSE
         ! charge density
         DO L=0,LMAX
         DO M=0,2*L
            LM=L*L+M+1
            DO K=1,R%NMAX
               RHOCOL(K,LM,1)=RHO(K,LM,1)
               RHOCOL(K,LM,2)=(AVMAG(1)*RHO(K,LM,2)+AVMAG(2)*RHO(K,LM,3)+ &
              &    AVMAG(3)*RHO(K,LM,4))/MAX(ABS_AVMAG,MAGMIN)
            ENDDO
         ENDDO
         ENDDO
         ! kinetic energy density
         DO L=1,LMAX_TAU
         DO M=0,2*L
            LM=L*L+M+1
            DO K=1,R%NMAX
               A=KINDENS(K,LM,1)
               B=(AVMAG(1)*KINDENS(K,LM,2)+AVMAG(2)*KINDENS(K,LM,3)+ &
              &    AVMAG(3)*KINDENS(K,LM,4))/MAX(ABS_AVMAG,MAGMIN)
               KINDENSCOL(K,LM,1)=(A+B)/2
               KINDENSCOL(K,LM,2)=(A-B)/2
            ENDDO
         ENDDO
         ENDDO
         ! possibly we have a non-augmented pseudo charge density as well
         IF (PRESENT(RHOPS)) THEN
            DO L=0,LMAX
            DO M=0,2*L
               LM=L*L+M+1
               DO K=1,R%NMAX
                  RHOPS(K,LM,1)=RHOPS(K,LM,1)
                  RHOPS(K,LM,2)=(AVMAG(1)*RHOPS(K,LM,2)+AVMAG(2)*RHOPS(K,LM,3)+ &
                 &    AVMAG(3)*RHOPS(K,LM,4))/MAX(ABS_AVMAG,MAGMIN)
               ENDDO
            ENDDO
            ENDDO
         ENDIF
      ENDIF

      END SUBROUTINE


!************************ SUBROUTINE  MAG_DIRECTION  *******************
!
! on entry POT must contain the v_xc(up) and v_xc(down)
! on return 
!  POT(:,1) =  (v_xc(up) + v_xc(down))/2
!  POT(:,2) =  m_x (v_xc(up) - v_xc(down))/2
!  POT(:,3) =  m_y (v_xc(up) - v_xc(down))/2
!  POT(:,4) =  m_z (v_xc(up) - v_xc(down))/2
! where m is the unit vector of the magnetization
!
!***********************************************************************


      SUBROUTINE RAD_MAG_DIRECTION( RHO, RHOCOL, POT, LMAX, R)
      IMPLICIT NONE
      REAL(q) :: RHO(:,:,:)     ! density
      REAL(q) :: POT(:,:,:)     ! potential
      REAL(q) :: RHOCOL(:,:,:)  ! total density
      TYPE (rgrid) :: R
      INTEGER LMAX,ISPIN,I,L,M,LM,K
      REAL(q) :: NORM,DELTAV,V0,MLMPROJ
      REAL(q) :: AVMAG(3),ABS_AVMAG

      LM=1
      AVMAG=0
      DO K=1,R%NMAX
         V0       =(POT(K,LM,1)+POT(K,LM,2))/2
         DELTAV   =(POT(K,LM,1)-POT(K,LM,2))/2
!        NORM     = MAX(RHOCOL(K,LM,2),MAGMIN)
         NORM     = MAX(ABS(RHOCOL(K,LM,2)),MAGMIN)*SIGN(1._q,RHOCOL(K,LM,2))
         POT(K,LM,1) = V0
         POT(K,LM,2) = DELTAV * RHO(K,LM,2) / NORM
         POT(K,LM,3) = DELTAV * RHO(K,LM,3) / NORM
         POT(K,LM,4) = DELTAV * RHO(K,LM,4) / NORM
         AVMAG(1)=AVMAG(1)+RHO(K,LM,2)*R%SI(K)
         AVMAG(2)=AVMAG(2)+RHO(K,LM,3)*R%SI(K)
         AVMAG(3)=AVMAG(3)+RHO(K,LM,4)*R%SI(K)
      ENDDO
      ABS_AVMAG=SQRT(AVMAG(1)**2+AVMAG(2)**2+AVMAG(3)**2)

      IF (.NOT. USE_AVERAGE_MAGNETISATION) THEN
      !
      ! for aspherical elements the subroutine has returned
      !           e_xc              
      ! d ------------------  this yields two contributions
      !    m_0 . m_lm  / |m_0|   
      !   e_xc             e_xc          m_lm    (m_0 m_lm) m_0    
      ! d ---- = d ------------------ (  ----  - ---------  ---  )
      !   m_0       m_0 m_lm  / |m_0|    |m_0|   |m_0|^2   |m_0|
      !   e_xc             e_xc            m_0
      ! d ---- = d ------------------     ----
      !   m_lm       m_0 m_lm  / |m_0|    |m_0|

         DO L=1,LMAX
         DO M=0,2*L
         LM=L*L+M+1
         DO K=1,R%NMAX

            V0    =(POT(K,LM,1)+POT(K,LM,2))/2
            DELTAV=(POT(K,LM,1)-POT(K,LM,2))/2

            NORM     = MAX(RHOCOL(K,1,2),MAGMIN)
            POT(K,LM,1) = V0
            POT(K,LM,2) = DELTAV * RHO(K,1,2) / NORM
            POT(K,LM,3) = DELTAV * RHO(K,1,3) / NORM
            POT(K,LM,4) = DELTAV * RHO(K,1,4) / NORM

            MLMPROJ = (RHO(K,LM,2) *RHO(K,1,2) +RHO(K,LM,3) *RHO(K,1,3) +RHO(K,LM,4) *RHO(K,1,4)) /NORM/NORM
            IF (RHOCOL(K,1,2)<MAGMIN) THEN
               MLMPROJ =0
            ENDIF
            POT(K,1,2)  = POT(K,1,2)+DELTAV*(RHO(K,LM,2) - MLMPROJ*RHO(K,1,2) )/ NORM
            POT(K,1,3)  = POT(K,1,3)+DELTAV*(RHO(K,LM,3) - MLMPROJ*RHO(K,1,3) )/ NORM
            POT(K,1,4)  = POT(K,1,4)+DELTAV*(RHO(K,LM,4) - MLMPROJ*RHO(K,1,4) )/ NORM
         ENDDO
         ENDDO
         ENDDO

      ELSE
         DO L=1,LMAX
         DO M=0,2*L
         LM=L*L+M+1
         DO K=1,R%NMAX

            V0    =(POT(K,LM,1)+POT(K,LM,2))/2
            DELTAV=(POT(K,LM,1)-POT(K,LM,2))/2

            NORM     = MAX(ABS_AVMAG,MAGMIN)
            POT(K,LM,1) = V0
            POT(K,LM,2) = DELTAV * AVMAG(1) / NORM
            POT(K,LM,3) = DELTAV * AVMAG(2) / NORM
            POT(K,LM,4) = DELTAV * AVMAG(3) / NORM

            MLMPROJ = (RHO(K,LM,2) *AVMAG(1) +RHO(K,LM,3) *AVMAG(2) +RHO(K,LM,4) *AVMAG(3)) /NORM/NORM
            IF (ABS_AVMAG<MAGMIN) THEN
               MLMPROJ =0
            ENDIF
            POT(K,1,2)  = POT(K,1,2)+DELTAV*(RHO(K,LM,2) - MLMPROJ*AVMAG(1) )/ NORM*R%SI(K)
            POT(K,1,3)  = POT(K,1,3)+DELTAV*(RHO(K,LM,3) - MLMPROJ*AVMAG(2) )/ NORM*R%SI(K)
            POT(K,1,4)  = POT(K,1,4)+DELTAV*(RHO(K,LM,4) - MLMPROJ*AVMAG(3) )/ NORM*R%SI(K)
         ENDDO
         ENDDO
         ENDDO
      ENDIF

      END SUBROUTINE


      SUBROUTINE RAD_MAG_DIRECTION_KINDENS( RHO, KINDENS, LMAX_TAU, POT, MU, R)
      USE constant
      IMPLICIT NONE
      REAL(q) :: RHO(:,:,:)        ! density
      REAL(q) :: KINDENS(:,:,:)    ! kinetic energy density (total,magnetization)
      REAL(q) :: POT(:,:,:)        ! potential (total,magnetization)
      REAL(q) :: MU(:,:,:)         ! \mu_xc= d E_xc / d \tau, on entry (up,down), on return: (total,magnetization)
      TYPE (rgrid) :: R
      INTEGER LMAX_TAU
      ! local variables
      INTEGER ISPIN,I,L,M,LM,K
      REAL(q) NORM,DELTAV,V0,MLMPROJ
      REAL(q) AVMAG(3),ABS_AVMAG

      AVMAG=0
      DO K=1,R%NMAX
         AVMAG(1)=AVMAG(1)+RHO(K,1,2)*R%SI(K)
         AVMAG(2)=AVMAG(2)+RHO(K,1,3)*R%SI(K)
         AVMAG(3)=AVMAG(3)+RHO(K,1,4)*R%SI(K)
      ENDDO
      ABS_AVMAG=SQRT(AVMAG(1)**2+AVMAG(2)**2+AVMAG(3)**2)

      IF (.NOT. USE_AVERAGE_MAGNETISATION) THEN
         WRITE(*,*) 'ERROR: RAD_MAG_DIRECTION_KINDENS: USE_AVERAGE_MAGNETISATION=.FALSE. is not implemented'
         STOP
      ELSE
         DO L=0,LMAX_TAU
         DO M=0,2*L
         LM=L*L+M+1
         DO K=1,R%NMAX

            V0    =(MU(K,LM,1)+MU(K,LM,2))/2
            DELTAV=(MU(K,LM,1)-MU(K,LM,2))/2

            NORM=MAX(ABS_AVMAG,MAGMIN)

            MU(K,LM,1) = V0
            MU(K,LM,2) = DELTAV * AVMAG(1) / NORM
            MU(K,LM,3) = DELTAV * AVMAG(2) / NORM
            MU(K,LM,4) = DELTAV * AVMAG(3) / NORM

            MLMPROJ = (KINDENS(K,LM,2) *AVMAG(1) +KINDENS(K,LM,3) *AVMAG(2) +KINDENS(K,LM,4) *AVMAG(3)) /NORM/NORM
!           IF (ABS_AVMAG<MAGMIN) MLMPROJ=0
            IF (ABS_AVMAG<MAGMIN) THEN
               POT(K,1,2) = POT(K,1,2)+DELTAV*(KINDENS(K,LM,2)-MLMPROJ*AVMAG(1) )/NORM/HSQDTM*R%SI(K)!*2._q
               POT(K,1,3) = POT(K,1,3)+DELTAV*(KINDENS(K,LM,3)-MLMPROJ*AVMAG(2) )/NORM/HSQDTM*R%SI(K)!*2._q
               POT(K,1,4) = POT(K,1,4)+DELTAV*(KINDENS(K,LM,4)-MLMPROJ*AVMAG(3) )/NORM/HSQDTM*R%SI(K)!*2._q
            ENDIF
         ENDDO
         ENDDO
         ENDDO
      ENDIF

      END SUBROUTINE


!*******************************************************************
!
!  RAD_CORE_XC
!  calculate the exchange correlation energy of the core charge
!  density on the radial grid
!  this energy will be subtracted (definining the zero of energy)
!  this has the advantage that small differences in the potentials
!  in the PP-generation program and VASP will not affect the
!  energy
!
!*******************************************************************

    SUBROUTINE RAD_CORE_XC( R, RHOC, DEXC)
      USE constant
      USE setexm
      IMPLICIT NONE

      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      TYPE (rgrid) :: R
    ! local variables
      REAL(q) DRH(R%NMAX),ADRH(R%NMAX),DADRH(R%NMAX),DDRH(R%NMAX)
      REAL(q) T1(R%NMAX)
      REAL(q) RHOT(R%NMAX)
      INTEGER K,N
      REAL(q) SCALE, RS, DEXC, DEXC_GGA, VXT, EXT
      REAL(q) DEXC1,DVXC1,EXC,DEXC_
      LOGICAL,PARAMETER :: TREL=.TRUE.
      REAL(q) :: SIM_FAKT

      SCALE=2*SQRT(PI)
      N=R%NMAX

      DEXC    =0

      DO K=1,N
         RHOT(K) =RHOC(K)/ (SCALE*R%R(K)*R%R(K))  ! charge density
      ENDDO
     
      DO K=1,N
         IF (RHOT(K) /=0 ) THEN
            CALL  EXCOR_PARA(RHOT(K),EXC,DEXC_,TREL)
            EXT = EXC/ RHOT(K)
         ELSE
            EXT=0
         ENDIF

         SIM_FAKT=R%SI(K)*SCALE
         DEXC    =DEXC    +EXT  *RHOC(K)*SIM_FAKT
      ENDDO

! GGA part
      DEXC_GGA=0

      IF (ISGGA()) THEN
         DEXC_GGA=0
         CALL GRAD(R,RHOT,T1)

         DO K=1,N
         IF (RHOT(K) > 1E-10_q) THEN
#ifdef vector
            CALL GGA91_WB(RHOT(K)*AUTOA3,T1(K)*AUTOA4,EXT,DEXC1,DVXC1)
#else
            CALL GGAALL(RHOT(K)*AUTOA3,T1(K)*AUTOA4,EXT,DEXC1,DVXC1,.FALSE.)
#endif
            SIM_FAKT=R%SI(K)*SCALE
            DEXC_GGA=DEXC_GGA+EXT*RHOC(K)*RYTOEV*SIM_FAKT
         ENDIF
         ENDDO
      ENDIF
#ifdef debug
      WRITE(0,1) DEXC/RYTOEV, DEXC_GGA/RYTOEV,DEXC/RYTOEV+DEXC_GGA/RYTOEV
 1    FORMAT(' core xc energy',3F14.7)
#endif
#undef debug
      DEXC=DEXC+DEXC_GGA
    END SUBROUTINE RAD_CORE_XC


!*******************************************************************
!
!  RAD_CORE_META_XC
!  calculate the exchange correlation energy of the core charge
!  density on the radial grid for metaGGA Exchange and correlation
!
!  as kinetic energy we use the Weizsaecker kinetic energy
!  (therefore the coefficient Ec is always unity)
!
! Robin Hirschl 20001225
!*******************************************************************

    SUBROUTINE RAD_CORE_META_XC( R, RHOC, TAUC, DEXC_MGGA)
      USE constant
      IMPLICIT NONE

      REAL(q) :: RHOC(:)        ! core charge for exchange correlation
      REAL(q) :: TAUC(:)        ! kinetic energy density of core
      TYPE (rgrid) :: R
    ! local variables
      REAL(q) TAUW(R%NMAX)
      REAL(q) T1(R%NMAX)
      REAL(q) RHOT(R%NMAX)
      INTEGER K,N
      REAL(q) SCALE,DEXC_MGGA, EXL,ECL
      REAL(q) RHO1,ABSNAB,TAUL,TAUWL
      LOGICAL,PARAMETER :: TREL=.TRUE.
      REAL(q) :: SIM_FAKT,EVTOH

      SCALE=2*SQRT(PI)
      N=R%NMAX
      EVTOH=1._q/(2.*HSQDTM)*AUTOA5                ! KinEDens eV to Hartree

      DO K=1,N
         RHOT(K) =RHOC(K)/ (SCALE*R%R(K)*R%R(K))  ! charge density
      ENDDO
     
      CALL GRAD(R,RHOT,T1)
! Weitzsaecker kinetic energy
      DO K=1,N
         IF (RHOT(K)>0) THEN
            TAUW(K)=0.25*HSQDTM*T1(K)**2/RHOT(K)
         ELSE
            TAUW(K)=0
         ENDIF
      ENDDO

      DEXC_MGGA=0

      DO K=1,N
         RHO1=MAX(RHOT(K)/2._q,1.E-10_q)
         ABSNAB=T1(K)
         TAUL =MAX(TAUC(K)/(2*SCALE*R%R(K)*R%R(K)),1.E-10_q)
         TAUWL=MAX(TAUW(K)/2._q,1.E-10_q)
! gK test was ok for Li at this place
      !  WRITE(78,*) TAUL,TAUWL 

      ! to avoid problems due to rounding errors force tauw<=tau
         TAUWL=MIN(TAUWL,TAUL)

         CALL METAGGA(RHO1*AUTOA3,RHO1*AUTOA3, &
              0.5*ABSNAB*AUTOA4, 0.5*ABSNAB*AUTOA4,ABSNAB*AUTOA4, &
              TAUL*EVTOH,TAUL*EVTOH, &
              TAUWL*EVTOH,TAUWL*EVTOH,EXL,ECL,K)
         
         SIM_FAKT=R%SI(K)*SCALE
         DEXC_MGGA=DEXC_MGGA+2*(EXL+ECL)*RYTOEV/AUTOA3*SCALE*R%R(K)*R%R(K)* &
              SIM_FAKT
      ENDDO
    END SUBROUTINE RAD_CORE_META_XC


!*******************************************************************
!
!  RAD_METRIC
!  evaluate the integral
!   (phi^AE_i * phi^AE_j)^2 - ( phi^PS_i * phi^PS_j + Q_ij^L )^2
!  this corresponds to the metric 
!  int_rad  | rho_AE(r)-rho_PS(r) | ^ 2 dr
!
!*******************************************************************

   SUBROUTINE RAD_METRIC( R, AUG, QPAW, WAE1, WAE2, WPS1, WPS2, AMETRIC)
      USE constant
      IMPLICIT NONE

      TYPE (rgrid) R
      REAL(q) :: AUG(:)          ! 1-normalized compensation charge
      REAL(q) :: WAE1(:),WAE2(:) ! all electron wavefunction
      REAL(q) :: WPS1(:),WPS2(:) ! pseudo wavefunctions
      REAL(q) :: QPAW            ! moments of compensation charge
      REAL(q) AMETRIC
! local variables
      REAL(q) :: RHOT(R%NMAX)
      INTEGER I


      DO I=1,R%NMAX
         RHOT(I)=(WAE1(I)*WAE2(I))**2-(WPS1(I)*WPS2(I)+AUG(I)*QPAW)**2
      ENDDO
      CALL SIMPI(R, RHOT, AMETRIC)
      AMETRIC=AMETRIC*2*SQRT(PI)

    END SUBROUTINE RAD_METRIC


!*******************************************************************
!
! calculate the gradient of a function on the radial grid
! nabla rho(r) = d/dr rho(r)
! (adopted from some routine of a program called atoms)
!
!*******************************************************************

      SUBROUTINE GRAD_(R,RH,DRH)
      IMPLICIT NONE
      TYPE (rgrid) R      ! defines the radial grid
      REAL(q) RH(R%NMAX)  ! charge density
      REAL(q) DRH(R%NMAX) ! gradient of the function
      REAL(q) H
      INTEGER NR,NM2,I

      H=R%H
      NR=R%NMAX
      NM2=NR-2
! For the first and second point the forward differences are used
      DRH(1)=((6._q*RH(2)+20._q/3._q*RH(4)+1.2_q*RH(6)) &
     &      -(2.45_q*RH(1)+7.5_q*RH(3)+3.75_q*RH(5)+1._q/6._q*RH(7)))/H
      DRH(2)=((6._q*RH(3)+20._q/3._q*RH(5)+1.2_q*RH(7)) &
     &        -(2.45_q*RH(2)+7.5_q*RH(4)+3.75_q*RH(6)+1._q/6._q*RH(8)))/H
! Five points formula
      DO  I=3,NM2
         DRH(I)=((RH(I-2)+8._q*RH(I+1))-(8._q*RH(I-1)+RH(I+2)))/12._q/H
      ENDDO
! Five points formula for the last two points ('backward differences')
      DRH(NR-1)=(-1._q/12._q*RH(NR-4)+0.5_q*RH(NR-3)-1.5_q*RH(NR-2) &
     &           +5._q/6._q*RH(NR-1)+0.25_q*RH(NR))/H
      DRH(NR)=  (0.25_q*RH(NR-4)-4._q/3._q*RH(NR-3)+3._q*RH(NR-2) &
     &           -4._q*RH(NR-1)+25._q/12._q*RH(NR))/H
! account for logarithmic mesh
      DO  I=1,NR
         DRH(I)=DRH(I)/R%R(I)
      ENDDO

      RETURN
      END SUBROUTINE

!*******************************************************************
!
! calculate the gradient of a function on the radial grid
! using 2.order differentiation (this is good enough 
! and less suseptible to noise than the previous routine)
! nabla rho(r) = d/dr rho(r)
!
!*******************************************************************

      SUBROUTINE GRAD(R,RH,DRH)
      IMPLICIT NONE
      TYPE (rgrid) R      ! defines the radial grid
      REAL(q) RH(R%NMAX)  ! charge density
      REAL(q) DRH(R%NMAX) ! gradient of the function
      REAL(q) H
      INTEGER NR,NM1,I

      H=R%H
      NR=R%NMAX
      NM1=NR-1
! 1. point use first order differantion
      DRH(1)=(RH(2)-RH(1))/H
! three point formula
      DO  I=2,NM1
         DRH(I)=(RH(I+1)-RH(I-1))/2/H
      ENDDO
! last point
      DRH(NR)=(RH(NR)-RH(NR-1))/H

      DO  I=1,NR
         DRH(I)=DRH(I)/R%R(I)
      ENDDO

      RETURN
      END SUBROUTINE

!*******************************************************************
!
!  AUG_SETE
!  calculate coefficients for generalized gaussian
!   exp ( r^2 / alpha)
!  not used
!*******************************************************************


    SUBROUTINE AUG_SETE(R, ALPHA, A, TH)
      IMPLICIT REAL(q) (A-H,O-Z)
      TYPE (rgrid) R

      REAL(q)  ALPHA, A, TH
      REAL(q)  TMP(R%NMAX)

      RC= R%RMAX
      ALPHA= LOG( TH) / (RC*RC)

      DO N=1,R%NMAX
         TMP(N)=0
         IF (R%R(N) <= RC) THEN
            VAL=EXP(ALPHA*R%R(N)*R%R(N))
            TMP(N)=R%R(N)*R%R(N)*VAL
         ENDIF
      ENDDO
      CALL SIMPI(R,TMP,SUM)
      A = 1/SUM

    END SUBROUTINE AUG_SETE

!*******************************************************************
!
!  AUG_SETQ
!  find a set of 2 q_i and coefficients A_i such that
!     sum_i  A_i j_l(q_i,Rc)  =0
!     sum_i  A_i j_l(q_i,Rc)' =0'
!     sum_i  A_i j_l(q_i,Rc)''=0
!  and
!     sum_i int_0^Rc A_i j_l(q_i r) r^(l+2) dr = 1
!  the augmentation charge is then represented by
!     sum_i  A_i j_l(q_i, r)
!  PS: I confess I have an obsession with spherical Besselfunctions
!*******************************************************************


    SUBROUTINE AUG_SETQ(L,R,RMAX,QQ,A,LCOMPAT)
      IMPLICIT REAL(q) (A-H,O-Z)
      TYPE (rgrid) R

      PARAMETER (NQ=2)
      REAL(q)  QQ(NQ)
      REAL(q)  A(NQ)
      LOGICAL LCOMPAT
      REAL(q)  AMAT(NQ,NQ)
      REAL(q)  B(3)
      INTEGER  IPIV(NQ)

      REAL(q)  TMP(R%NMAX)
!-----gaussian integration
      PARAMETER (M=32)
      REAL(q)  WR(M),RA(M)
      INTEGER IFAIL
      EXTERNAL GAUSSI2
!-----------------------------------------------------------------------
! search for q so that j(q Rc)=0
!-----------------------------------------------------------------------
      CALL AUG_BEZERO(QQ,L,NQ)
!-----------------------------------------------------------------------
! second set the matrix
!-----------------------------------------------------------------------
      ITYPE=0
      CALL GAUSSI(GAUSSI2,0.0_q,RMAX, ITYPE,M,WR,RA,IFAIL)

      DO I=1,NQ
         QQ(I)=QQ(I)/RMAX
         QR  =QQ(I)*RMAX
         CALL SBESSE3( QR, BJ, BJP, BJPP, L)
         BJP =BJP *QQ(I)
         BJPP=BJPP*QQ(I)*QQ(I)
         AMAT(1,I) = BJP
!  I could not find an analytical expression for \int j_l(qr) r^(2+l) dr
!  I will check again probably it is simple enough
         IF (LCOMPAT) THEN
           DO N=1,R%NMAX
              IF (R%R(N) < RMAX) THEN
                 QR=QQ(I)*R%R(N)
                 CALL SBESSEL(QR,BJ,L)
                 TMP(N)=BJ*R%R(N)**(2+L)
              ELSE
                 TMP(N)=0
              ENDIF
           ENDDO
           CALL SIMPI(R,TMP,SUM)
           AMAT(2,I) = SUM
         ELSE

! Gauss integration, more accurate !
           SUM2=0
           DO N=1,M
              QR=QQ(I)*RA(N)
              CALL SBESSEL(QR,BJ,L)
              SUM2=SUM2+BJ*RA(N)**(2+L)*WR(N)
           ENDDO

           AMAT(2,I) = SUM2
         ENDIF
      ENDDO
!-----------------------------------------------------------------------
!  solve the linear equations  B_i = AMAT_ii' A_i'
!-----------------------------------------------------------------------
      IFAIL=0
      B(1)=0
      B(2)=1

      A(1)=0
      A(2)=0

      CALL DGETRF( NQ, NQ, AMAT, NQ, IPIV, IFAIL )
      CALL DGETRS('N', NQ, 1, AMAT, NQ, IPIV, B, NQ, IFAIL)
      A=B(1:2)

      B(1)=0
      B(2)=0
      B(3)=0

      DO I=1,NQ
         SUM=0
         QR  =QQ(I)*RMAX
         CALL SBESSE3( QR, BJ, BJP, BJPP, L)
         B(1)=B(1)+BJ*A(I)
         B(2)=B(2)+BJP*A(I)*QQ(I)
         B(3)=B(3)+BJPP*A(I)*QQ(I)*QQ(I)
      ENDDO
    END SUBROUTINE

!*******************************************************************
!  SUBROUTINE BEZERO
!  searches for NQ zeros j(qr)
!  i/o:
!         XNULL(NQ) result
!         L           quantum number l
!  great-full spaghetti code (written by gK)
!********************************************************************

    SUBROUTINE AUG_BEZERO(XNULL,L,NQ)
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (STEP=.1_q, BREAK= 1E-10_q)
      DIMENSION XNULL(NQ)
! initialization
      X=STEP
      N=0
! entry point for next q_n
  30  CALL SBESSE2(X, BJ1, DUMMY,  L)
! coarse search
  10  X=X+STEP
      CALL SBESSE2(X, BJ2, DUMMY,  L)
! found one point
      IF(BJ1*BJ2 < 0) THEN
        ETA=0.0_q
! interval bisectioning
        SSTEP=STEP
        XX   =X
  20    SSTEP=SSTEP/2
        IF (BJ1*BJ2 < 0) THEN
          XX=XX-SSTEP
        ELSE
          XX=XX+SSTEP
        ENDIF
        CALL SBESSE2(XX, BJ2, DUMMY,  L)
        IF (SSTEP > BREAK) GOTO 20

        N=N+1
        XNULL(N)=XX
        IF (N == NQ) RETURN
        GOTO 30
      ENDIF
      GOTO 10

    END SUBROUTINE

!********************************************************************
!
!  SUBROUTINE SIMPI
!  integrate a function on the logarithmic grid
!  uses the previously setup weights
!
!********************************************************************

    SUBROUTINE SIMPI(R,F,FI)
      IMPLICIT NONE
      TYPE (rgrid) R
      REAL(q)  F(:),FI,SUM
      INTEGER   K

      SUM=0
!OCL SCALAR
      DO K=1,R%NMAX
         SUM=SUM+F(K)*R%SI(K)
      ENDDO

      FI=SUM

    END SUBROUTINE

!********************************************************************
!
!  SUBROUTINE SET_SIMP
!  setup weights for simpson integration on radial grid
!  any radial integral can then be evaluated by just summing all
!  radial grid points with the weights SI
!
!  int dr = sum_i si(i) * f(i)
!  the factors  R%R(K)  *R%H stem from the logarithmic grid
!********************************************************************

    SUBROUTINE SET_SIMP(R)
      IMPLICIT NONE
      TYPE (rgrid) R
      INTEGER   K

      ALLOCATE(R%SI(R%NMAX))
      R%SI=0
      DO K=R%NMAX,3,-2
         R%SI(K)=    R%R(K)  *R%H/3._q+R%SI(K)
         R%SI(K-1)=4*R%R(K-1)*R%H/3._q
         R%SI(K-2)=  R%R(K-2)*R%H/3._q
      ENDDO
    END SUBROUTINE

!******************************************************************** 
!
!  SPHERICAL_BESSEL_TRANSFORM_R2Q
!
!********************************************************************
      SUBROUTINE SPHERICAL_BESSEL_TRANSFORM_R2Q( &
     & L,R,FR,G,FG &
     & )
      USE constant
      IMPLICIT NONE
      TYPE(rgrid) R
      INTEGER L
      REAL(q) FR(:),G(:),FG(:)
      ! local variables
      INTEGER IG,IR
      REAL(q) QR,BJ,SUM,SCALE
      REAL(q), ALLOCATABLE :: TMP(:),SI(:)
      
      ! quick consistency checks
      IF (SIZE(FR)<R%NMAX.OR.SIZE(FG)<SIZE(G)) THEN
         WRITE(*,*) 'SPHERICAL_BESSEL_TRANSFORM: ERROR: array(s) not sufficiently large:',&
        & SIZE(FR),R%NMAX,SIZE(FG),SIZE(G)
         STOP
      ENDIF

!     SCALE=SQRT(2._q/PI)
      
      ALLOCATE(TMP(R%NMAX))

      DO IG=1,SIZE(G)
         TMP=0
         DO IR=1,R%NMAX
            QR=G(IG)*R%R(IR)
            CALL SBESSEL(QR,BJ,L)
!           TMP(IR)=FR(IR)*QR*BJ
            TMP(IR)=FR(IR)*BJ*R%R(IR)*R%R(IR)
         ENDDO
         CALL SIMPI(R,TMP,SUM)
!        FG(IG)=SCALE*SUM
         FG(IG)=(4._q*PI)*SUM
      ENDDO  

      RETURN
      END SUBROUTINE SPHERICAL_BESSEL_TRANSFORM_R2Q

!********************************************************************
!
!  BESSEL_TRANSFORM_RADIAL_FUNCTION
!
!********************************************************************
      SUBROUTINE BESSEL_TRANSFORM_RADIAL_FUNCTION( &
     & L,R,FR,DG,FG &
     &)
      IMPLICIT NONE
      TYPE(rgrid) R
      INTEGER L
      REAL(q) FR(:)
      REAL(q) DG
      REAL(q) FG(:,:)
      ! local variables
      INTEGER IG

      DO IG=1,SIZE(FG,1)
         FG(IG,1)=(IG-1)*DG 
      ENDDO
      CALL SPHERICAL_BESSEL_TRANSFORM_R2Q(L,R,FR(:),FG(:,1),FG(:,2))

      CALL SPLCOF(FG,SIZE(FG,1),SIZE(FG,1),0._q) 

      RETURN
      END SUBROUTINE BESSEL_TRANSFORM_RADIAL_FUNCTION
#endif
  END MODULE radial
