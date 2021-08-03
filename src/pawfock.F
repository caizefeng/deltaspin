!#define dotiming
!#define debug
#include "symbol.inc" 

  MODULE RS_GreensFunc_kernel
  USE prec
  IMPLICIT NONE

  PRIVATE :: GreensFunc,RS_GreensFunc,RS_GreensFunc_type

  TYPE GreensFunc
     INTEGER ITYPE,L
     REAL(q) RMU
     LOGICAL LRHF
     REAL(q), POINTER :: GF(:,:)
  END TYPE GreensFunc

  TYPE (GreensFunc), ALLOCATABLE, SAVE :: RS_GreensFunc(:)
  INTEGER, SAVE :: RS_GreensFunc_type=-1

  CONTAINS

  SUBROUTINE SET_RSGF_TYPE(NT)
    INTEGER NT
    RS_GreensFunc_type=NT
  END SUBROUTINE SET_RSGF_TYPE

  SUBROUTINE UNSET_RSGF_TYPE
    RS_GreensFunc_type=-1
  END SUBROUTINE UNSET_RSGF_TYPE

  SUBROUTINE SET_RSGF(R,RMU,LL,LRHF,GRS)
    USE radial
    IMPLICIT NONE
    TYPE (rgrid) R
    REAL(q) RMU
    INTEGER LL
    LOGICAL LRHF
    REAL(q) GRS(R%NMAX,R%NMAX)
    ! local variables
    TYPE (GreensFunc), ALLOCATABLE :: TMP(:)
    INTEGER I,NMAX
    LOGICAL LFOUND

    IF (RS_GreensFunc_type<1) THEN
       ! compute on-the-fly
       CALL RS_COULOMB_GREEN_FUNC(R,RMU,LL,GRS,LRHF)
    ELSE
       ! upon first call
       IF (.NOT.ALLOCATED(RS_GreensFunc)) THEN
          ALLOCATE(RS_GreensFunc(1))
          ALLOCATE(RS_GreensFunc(1)%GF(R%NMAX,R%NMAX))
          RS_GreensFunc(1)%ITYPE=RS_GreensFunc_type
          RS_GreensFunc(1)%L=LL
          RS_GreensFunc(1)%RMU=RMU
          RS_GreensFunc(1)%LRHF=LRHF
          RS_GreensFunc(1)%GF=0._q
          CALL RS_COULOMB_GREEN_FUNC(R,RMU,LL,RS_GreensFunc(1)%GF,LRHF)   
       ENDIF
       ! check if this type and angular momentum has already been computed
       LFOUND=.FALSE.
       DO I=1,SIZE(RS_GreensFunc)
          IF (RS_GreensFunc(I)%ITYPE==RS_GreensFunc_type.AND.RS_GreensFunc(I)%L==LL.AND. &
         &   RS_GreensFunc(I)%RMU==RMU.AND.(RS_GreensFunc(I)%LRHF.EQV.LRHF)) THEN
             LFOUND=.TRUE.; GRS=RS_GreensFunc(I)%GF; EXIT
          ENDIF
       ENDDO
       ! if not  
       IF (.NOT.LFOUND) THEN
          ! copy RS_GreensFunc to TMP
          ALLOCATE(TMP(SIZE(RS_GreensFunc)))
          DO I=1,SIZE(RS_GreensFunc)
             TMP(I)%ITYPE=RS_GreensFunc(I)%ITYPE
             TMP(I)%L=RS_GreensFunc(I)%L
             TMP(I)%RMU=RS_GreensFunc(I)%RMU
             TMP(I)%LRHF=RS_GreensFunc(I)%LRHF
             NMAX=SIZE(RS_GreensFunc(I)%GF,1)
             ALLOCATE(TMP(I)%GF(NMAX,NMAX))
             TMP(I)%GF=RS_GreensFunc(I)%GF
             DEALLOCATE(RS_GreensFunc(I)%GF)
             NULLIFY(RS_GreensFunc(I)%GF)
          ENDDO
          DEALLOCATE(RS_GreensFunc)
          ! increase the size of RS_GreensFunc and copy TMP back
          ALLOCATE(RS_GreensFunc(SIZE(TMP)+1))
          DO I=1,SIZE(TMP)
             RS_GreensFunc(I)%ITYPE=TMP(I)%ITYPE
             RS_GreensFunc(I)%L=TMP(I)%L
             RS_GreensFunc(I)%RMU=TMP(I)%RMU
             RS_GreensFunc(I)%LRHF=TMP(I)%LRHF
             NMAX=SIZE(TMP(I)%GF,1)
             ALLOCATE(RS_GreensFunc(I)%GF(NMAX,NMAX))
             RS_GreensFunc(I)%GF=TMP(I)%GF
             DEALLOCATE(TMP(I)%GF)
             NULLIFY(TMP(I)%GF)
          ENDDO
          DEALLOCATE(TMP)
          ! add the range-separated Greens function for
          ! type RS_GreensFunc_type and angular moment LL
          RS_GreensFunc(I)%ITYPE=RS_GreensFunc_type
          RS_GreensFunc(I)%L=LL
          RS_GreensFunc(I)%RMU=RMU
          RS_GreensFunc(I)%LRHF=LRHF
          ALLOCATE(RS_GreensFunc(I)%GF(R%NMAX,R%NMAX))
          GRS=0._q
          CALL RS_COULOMB_GREEN_FUNC(R,RMU,LL,GRS,LRHF)
          RS_GreensFunc(I)%GF=GRS
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE SET_RSGF

  SUBROUTINE SET_RSGF_SIMPLE(PP)
    USE pseudo
    TYPE (potcar) PP
    ! local variables
    REAL(q) RMU,BRSEXX,BEXX
    INTEGER LMAX,I
    LOGICAL LRHF

    IF (RS_GreensFunc_type<=0) RETURN

    CALL RSPARAMETERS(RMU,LRHF,BRSEXX,BEXX)
    IF (BRSEXX==0) RETURN

    IF (ALLOCATED(RS_GreensFunc)) CALL DEALLOCATE_RSGF

    LMAX=MAXVAL(PP%LPS(:))
    LMAX=MIN(2*LMAX,8)

    ALLOCATE(RS_GreensFunc(LMAX+1))
    DO I=1,LMAX+1
       RS_GreensFunc(I)%ITYPE=RS_GreensFunc_type
       RS_GreensFunc(I)%L=I-1
       RS_GreensFunc(I)%RMU=RMU
       RS_GreensFunc(I)%LRHF=LRHF
       ALLOCATE(RS_GreensFunc(I)%GF(PP%R%NMAX,PP%R%NMAX))
       CALL RS_COULOMB_GREEN_FUNC(PP%R,RMU,I-1,RS_GreensFunc(I)%GF,LRHF)
    ENDDO
  END SUBROUTINE SET_RSGF_SIMPLE

!
!
  SUBROUTINE SET_RSGF_ALL(P) 
    USE pseudo
    TYPE (potcar) P(:)
    ! local variables
    REAL(q) RMU,BRSEXX,BEXX
    INTEGER NT,N,LMAX,I
    LOGICAL LRHF

    PROFILING_START('set_rsgf_all')

    CALL RSPARAMETERS(RMU,LRHF,BRSEXX,BEXX)

    ! quick return if possible
    IF (BRSEXX==0) THEN
       PROFILING_STOP('set_rsgf_all')
       RETURN
    ENDIF
    ! and another quick return
    IF (ALLOCATED(RS_GreensFunc)) THEN
       PROFILING_STOP('set_rsgf_all')
       RETURN
    ENDIF
    ! determine the number of Greens functions
    N=0
    DO NT=1,SIZE(P)
       LMAX=MAXVAL(P(NT)%LPS(:))
       LMAX=MIN(2*LMAX,8)
       N=N+LMAX+1
    ENDDO
    ! allocate RS_GreensFunc
    ALLOCATE(RS_GreensFunc(N))
    ! and fill it
    N=0
    DO NT=1,SIZE(P)
       LMAX=MAXVAL(P(NT)%LPS(:))
       LMAX=MIN(2*LMAX,8)
       DO I=1,LMAX+1
          N=N+1
          RS_GreensFunc(N)%ITYPE=NT
          RS_GreensFunc(N)%L=I-1
          RS_GreensFunc(N)%RMU=RMU
          RS_GreensFunc(N)%LRHF=LRHF
          ALLOCATE(RS_GreensFunc(N)%GF(P(NT)%R%NMAX,P(NT)%R%NMAX))
          CALL RS_COULOMB_GREEN_FUNC(P(NT)%R,RMU,I-1,RS_GreensFunc(N)%GF,LRHF)
       ENDDO
    ENDDO
    PROFILING_STOP('set_rsgf_all')

  END SUBROUTINE SET_RSGF_ALL

  SUBROUTINE DEALLOCATE_RSGF
    INTEGER I
    IF (.NOT.ALLOCATED(RS_GreensFunc)) RETURN
    DO I=1,SIZE(RS_GreensFunc)
       DEALLOCATE(RS_GreensFunc(I)%GF)
       NULLIFY(RS_GreensFunc(I)%GF)
    ENDDO
    DEALLOCATE(RS_GreensFunc)
  END SUBROUTINE DEALLOCATE_RSGF

  SUBROUTINE RS_COULOMB_GREEN_FUNC(R,MU,L,G,LONGRANGE)

    USE prec
    USE radial
    USE constant
    USE c2f_interface, ONLY : ERRFC
    
    IMPLICIT NONE
! input: grid, separation parameter, L quantum number
    TYPE (rgrid) R
    REAL(q) MU
    INTEGER L
! input: LONGRANGE=.False. : short range part
!                  .True.  :  long range part
    LOGICAL LONGRANGE
! maximal L quantum number supported
    INTEGER, PARAMETER :: LMAX=8
! output: range separated coulomb green function G_{l}_{rs}(r,r')
    REAL(q) G(R%NMAX,R%NMAX)
! local variables
    INTEGER N,M,P,IGRID,JGRID
    REAL(q) E,F(0:L)
    REAL(q) PHI
    REAL(q) X,Y
    REAL(q) S
    REAL(q) TWON
    REAL(q) EXPP,EXPM,EXPX
    REAL(q) ERFP,ERFM,ERFX
! unable to construct G_l_sr for L>LMAX 
    IF (L>LMAX) THEN
       WRITE(*,*) 'RS_COULOMB_GREEN_FUNC: L>Lmax:',L,LMAX
       STOP
    END IF
! initialize to zero
    G=0 
! start
    DO IGRID=1,R%NMAX
    DO JGRID=1,MIN(IGRID+1,R%NMAX)
       ! scale variables
       X=MU*R%R(IGRID)
       Y=MU*R%R(JGRID)
       ! expanded or full form
! test
       IF (Y<0.1_q) THEN
!      IF (.FALSE.) THEN
! test
       ! expanded form
          EXPX=EXP(-X**2)
          ERFX=ERRFC(X)
          TWON=REAL(2*L)
          S=0
          DO P=1,L
             S=S+(2._q**P)*(X**(2*P-1))/(FFACTOR(2*P-1)) 
          ENDDO
          PHI=(Y**(L)/X**(L+1))* &
         &    (ERFX+S*(EXPX/SQRT(PI))+ &
         &       ((2._q**(L+1))*(X**(2*L+1))* &
         &          EXPX/FFACTOR(2*L-1)/(TWON+3._q)/SQRT(PI))*(Y**2)+ &
         &       ((2._q**L)*(2._q*X**2-TWON-3._q)*X**(2*L+1)*EXPX/FFACTOR(2*L-1)/ &
         &          (TWON+3._q)/(TWON+5._q)/SQRT(PI))*(Y**4))
       ELSE
       ! full form
          ! precalculate exponentials and erfc             
          EXPP=EXP(-(X+Y)**2)
          EXPM=EXP(-(X-Y)**2)
          ERFP=ERRFC(X+Y)
          ERFM=ERRFC(X-Y)
          ! precalculate auxiliary function E
          E=((X**(2*L+1)+Y**(2*L+1))*ERFP- &
         &   (X**(2*L+1)-Y**(2*L+1))*ERFM)/(2._q*(X*Y)**(l+1))
          ! precalculate auxiliary function F up to order L
          DO N=0,L
             S=0._q
             DO P=0,N
                S=S+((-1._q)**(P+1))*FACTPROD(N,P)* &
               &  (((-1._q)**(N-P))*EXPP-EXPM)/((4._q*X*Y)**(P+1))
             ENDDO      
             F(N)=S*(2._q/SQRT(PI))
!              WRITE(*,'(A, I2, A,F12.8,A,F12.8,A,2F24.16)') & 
!             &            "F(",N,",",X,",",Y,")=",F(N),E(N)
          ENDDO
          ! calculate the radial function
          PHI=F(L)+E
          DO M=1,L
             PHI=PHI+F(L-M)*((X/Y)**M+(Y/X)**M)
          ENDDO
       ENDIF
       ! store G_l_sr (r,r')
       G(IGRID,JGRID)=MU*PHI
       IF (LONGRANGE) THEN
       ! G_l_lr=G_l-G_l_sr
          G(IGRID,JGRID)=MU*(Y**(L))/(X**(L+1))-G(IGRID,JGRID)
! test
!         G(IGRID,JGRID)=MU*(Y**(L))/(X**(L+1))
! test
       ENDIF
    ENDDO
    ENDDO
    RETURN      
  END SUBROUTINE RS_COULOMB_GREEN_FUNC

  FUNCTION FACTPROD(N,P)
    USE prec
! this routine is based on the routine "bico" of
! numerical recipes. uses the logarithm of the
! factorials to form the expression
!      (n+p)!
!     ----------
!      p! (n-p)!
! in order to avoid overflow for the components.
! in double precision its use is limited to n=8.
    IMPLICIT NONE
    INTEGER P,N
    REAL(q) FACTPROD
    IF (N>8) THEN
       WRITE(*,*) 'FACTPROD: N too large (>8) for factorial expression:',N
       STOP
    ENDIF
    IF (N<P) THEN
       WRITE(*,*) 'FACTPROD: invalid call: N<PR:',N,P
       STOP
    ENDIF
    FACTPROD=REAL(NINT(EXP(FACTLN(N+P)-FACTLN(P)-FACTLN(N-P))))
  END FUNCTION FACTPROD

  FUNCTION GAMMLN(XX)
    USE prec
! numerical recipes routine to calculate 
! the logarithm of the gamma function
    IMPLICIT NONE
    REAL(q) GAMMLN,XX
    INTEGER J
    REAL(q) SER,STP,TMP,X,Y,COF(6)
    SAVE COF,STP
    DATA COF,STP/76.18009172947146_q, &
   & -86.50532032941677_q,  24.01409824083091_q, &
   &  -1.231739572450155_q,   .1208650973866179E-2_q, &
   &   -.5395239384953E-5_q, 2.5066282746310005_q/
    X=XX
    Y=X
    TMP=X+5.5_q
    TMP=(X+0.5_q)*LOG(TMP)-TMP
    SER=1.000000000190015_q
    DO J=1,6
      Y=Y+1._q
      SER=SER+COF(J)/Y
    ENDDO
    GAMMLN=TMP+LOG(STP*SER/X)
  END FUNCTION GAMMLN

  FUNCTION FACTLN(N)
    USE prec
! numerical recipes routine to calculate 
! the logarithm of the factorial using gammln
    IMPLICIT NONE
    INTEGER N
    REAL(q) FACTLN
    REAL(q) A(100)
    SAVE A
    DATA A/100*-1._q/
    IF (N<0) WRITE(*,*) 'FACTLN: negative factorial in FACTLN'
    IF (N<=99) THEN
      IF (A(N+1)<0._q) A(N+1)=GAMMLN(REAL(N)+1._q)
      FACTLN=A(N+1)
    ELSE
      FACTLN=GAMMLN(REAl(N)+1._q)
    ENDIF
  END FUNCTION FACTLN

  FUNCTION FFACTOR(N)
    IMPLICIT NONE
    INTEGER FFACTOR
    INTEGER N,I,SUM
    IF (N<0) THEN
       SUM=1
    ELSE
       SUM=N
       DO I=N-2,1,-2
          SUM=SUM*I
       ENDDO
    ENDIF
    FFACTOR=SUM
  END FUNCTION FFACTOR

  END MODULE RS_GreensFunc_kernel


!*********************************************************************************
!
! module implements the one centre (on site) Hartree Fock (exact exchange)
! routines
! some routines (COLOUMB_4TERM, CALC_DHARTREE) 
! were initially written by Adrian Rohrbach but rewritten and cleanded up 
! extensively by gK
!
!*********************************************************************************

MODULE PAWFOCK
  USE prec

  IMPLICIT NONE

CONTAINS

!************************ SUBROUTINE COLOUMB_4TERM *******************************
!
!  calculate and stores the four electron Coloumb integrals (Slater integrals)
!
!  S(i,j,l,k,L) = int dr' phi_i(r') phi_j(r') \int dr G_L(r',r) phi_l(r) phi_k(r)
!
!  where G_L(r',r) is the Coloumb kernel (or Greens function)
!  for the angular quantum number L
!
!  G_L(r',r) =  4 pi/(2L+1) (r',r)<^(L) /(r',r)>^(L+1) 
!
!  MIND: S must be set to zero before call, otherwise the results are added to S
!  the L index in S(i,j,l,k,L) is divided by 2 to save space, since
!  only odd or even L indices are allowed
!
!*********************************************************************************

    SUBROUTINE COLOUMB_4TERM (W, L, R, CHANNELS, S, LONECENTERMAX)
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      INTEGER :: CHANNELS
      INTEGER L(:)             ! l-quantum number of each channel
      TYPE (rgrid) R
      REAL(q) :: W(:,:)        ! wavefunction of each channel
      REAL(q) :: S(:,:,:,:,0:) ! Slater integrals
      INTEGER :: LONECENTERMAX ! maximum l for pseudo one center development
       
       ! local variables
      REAL(q) :: M(R%NMAX),SUM
      INTEGER CH1,CH2,LL,LLP,CHI1,CHI2,I,J,LLOOP,LMIN,LMAX, LDIM
      REAL(q) :: RHOAE(R%NMAX)
      LDIM=SIZE(S,5)

      IF (SIZE(S,1)<CHANNELS .OR. SIZE(S,2)<CHANNELS & 
           .OR. SIZE(S,3)<CHANNELS .OR. SIZE(S,4)<CHANNELS) THEN
         WRITE(0,*) 'internal ERROR in COLOUMB_4TERM: S is too small',SIZE(S)
         STOP
      ENDIF

      ! loop over k,l
      DO CH1=1, CHANNELS
      DO CH2=1, CHANNELS
 
         ! calculate rho_kl(r)
         DO I=1,R%NMAX
            RHOAE(I)=W(I,CH1)*W(I,CH2)
         ENDDO
         
         LL = L(CH1)
         LLP = L(CH2)
         LMIN = ABS(LL-LLP) ; LMAX = MIN(ABS(LL+LLP),LONECENTERMAX)
         IF (LMAX/2>LDIM) THEN
            WRITE(0,*) 'internal ERROR in COLOUMB_4TERM: LDIM is not sufficiently large',LMAX/2,LDIM
            STOP
         ENDIF
         M(1:R%NMAX) = 0
 
         ! loop over L
         DO LLOOP=LMIN,LMAX,2
            ! calculated Hartree Pot. M_klL M(r,k,l,L)
            CALL RAD_POT_EX_HAR(RHOAE,R,LLOOP,M)
            ! loop over i,j
            DO CHI1=1,CHANNELS
            DO CHI2=1,CHANNELS              
 
               SUM = 0
 
               DO J=1,R%NMAX
                  SUM = SUM + M(J)*W(J,CHI1)*W(J,CHI2)*R%SI(J)
               ENDDO

               S(CHI1,CHI2,CH1,CH2,LLOOP/2) = SUM+S(CHI1,CHI2,CH1,CH2,LLOOP/2)

            ENDDO
            ENDDO !J
         ENDDO !I
      ENDDO !L
      ENDDO !K
    END SUBROUTINE COLOUMB_4TERM

!************************ SUBROUTINE COLOUMB_4TERM_PS ****************************
!
!  calculate and store the four electron Coloumb integrals (Slater integrals)
!  for pseudo wave functions (requires proper augmentation)
!
!  S(i,j,l,k,L) = int dr' phi_i(r') phi_j(r') \int dr G_L(r',r) phi_l(r) phi_k(r)
!
!  where G_L(r',r) is the Coloumb kernel (or Greens function)
!  for the angular quantum number L
!
!  G_L(r',r) =  4 pi/(2L+1) (r',r)<^(L) /(r',r)>^(L+1) 
!
!  MIND: S must be set to zero before call, otherwise the results are added to S
!  the L index in S(i,j,l,k,L) is divided by 2 to save space, since
!  only odd or even L indices are allowed
!
!*********************************************************************************

    SUBROUTINE COLOUMB_4TERM_PS (W, L, R, CHANNELS, AUG, QPAW, S, LONECENTERMAX, & 
         AUG_FOCK, QPAW_FOCK)
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      INTEGER :: CHANNELS      ! number of LN channels in the non local PP
      INTEGER L(:)             ! L quantum number of each channel
      TYPE (rgrid) R           ! radial grid descriptor
      REAL(q) :: W(:,:)        ! wavefunctions
      REAL(q) :: AUG(:,0:)     ! 1-normalized L-dep compensation charge
      REAL(q) :: QPAW(:,:,0:)  ! moments of compensation charge Q(PAW,ll L)
      REAL(q) :: S(:,:,:,:,0:) ! Slater integrals
      INTEGER :: LONECENTERMAX ! maximum l for pseudo one center development
   ! these additional arrays are for accurate AE restoration of the charge density
      REAL(q), POINTER :: AUG_FOCK(:,:,:)    ! 1-normalized L-dep compensation charge
      REAL(q), POINTER :: QPAW_FOCK(:,:,:,:) ! how much of each component AUG_FOCK needs to be added
       
       ! local variables
      REAL(q) :: M(R%NMAX),SUM
      INTEGER CH1,CH2,LL,LLP,CHI1,CHI2,I,J,LLOOP,LMIN,LMAX, LDIM,N
      REAL(q) :: RHOPS(R%NMAX),RHOAUG(R%NMAX)
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE

      LDIM=SIZE(S,5)

      IF (SIZE(S,1)<CHANNELS .OR. SIZE(S,2)<CHANNELS & 
           .OR. SIZE(S,3)<CHANNELS .OR. SIZE(S,4)<CHANNELS) THEN
         WRITE(0,*) 'internal ERROR in COLOUMB_4TERM_PS: S is too small',SIZE(S)
         STOP
      ENDIF
      IF (ASSOCIATED(QPAW_FOCK)) THEN
         NMAX_FOCKAE=SIZE(QPAW_FOCK,4)
         LMAX_FOCKAE=SIZE(QPAW_FOCK,3)-1
      ELSE
         NMAX_FOCKAE=1
         LMAX_FOCKAE=-1
      ENDIF

      ! loop over k,l
      DO CH1=1, CHANNELS
      DO CH2=1, CHANNELS
 
         ! calculate rho_kl(r)
         DO I=1,R%NMAX
            RHOPS(I)=W(I,CH1)*W(I,CH2)
         ENDDO
         
         LL = L(CH1)
         LLP = L(CH2)
         LMIN = ABS(LL-LLP) ; LMAX = MIN(ABS(LL+LLP),LONECENTERMAX)
         IF (LMAX/2>LDIM) THEN
            WRITE(0,*) 'internal ERROR in COLOUMB_4TERM_PS: LDIM is not sufficiently large',LMAX/2,LDIM
            STOP
         ENDIF
         IF (SIZE(AUG,2)-1<LMAX) THEN
            WRITE(0,*) 'internal ERROR in COLOUMB_4TERM_PS: SIZE(AUG) is too small',SIZE(AUG,2)-1,LMAX
            STOP
         ENDIF
         M(1:R%NMAX) = 0
 
         ! loop over L
         DO LLOOP=LMIN,LMAX,2
            ! calculated Hartree Pot. M_klL M(r,k,l,L)
            RHOAUG(1:R%NMAX)=RHOPS(1:R%NMAX)+AUG(1:R%NMAX,LLOOP)*QPAW(CH1,CH2,LLOOP)
            IF (LLOOP <= LMAX_FOCKAE) THEN
               DO N=1,NMAX_FOCKAE
                  RHOAUG(1:R%NMAX)=RHOAUG(1:R%NMAX)+AUG_FOCK(1:R%NMAX,LLOOP,N)*QPAW_FOCK(CH1,CH2,LLOOP,N)
               ENDDO
            ENDIF
            CALL RAD_POT_EX_HAR(RHOAUG,R,LLOOP,M)
            ! loop over i,j
            DO CHI1=1,CHANNELS
            DO CHI2=1,CHANNELS
 
               SUM = 0
               DO J=1,R%NMAX
                  SUM = SUM + M(J)*R%SI(J)* &
                       (W(J,CHI1)*W(J,CHI2)+AUG(J,LLOOP)*QPAW(CHI1,CHI2,LLOOP))
               ENDDO
               IF (LLOOP <= LMAX_FOCKAE) THEN
                  DO N=1,NMAX_FOCKAE
                     DO J=1,R%NMAX
                        SUM = SUM + M(J)*R%SI(J)*AUG_FOCK(J,LLOOP,N)*QPAW_FOCK(CHI1,CHI2,LLOOP,N)
                     ENDDO
                  ENDDO
               ENDIF
               S(CHI1,CHI2,CH1,CH2,LLOOP/2) = S(CHI1,CHI2,CH1,CH2,LLOOP/2)+SUM

            ENDDO
            ENDDO !J
         ENDDO !I
      ENDDO !L
      ENDDO !K
      
    END SUBROUTINE COLOUMB_4TERM_PS


! **************************** SUBROUTINE RAD_POT_EX_HAR *************************
!
! calculate the Coloumb potential
!   V(r') = \int dr G_L(r',r) rho_L(r) d r
!
! where G_L(r',r) is the Coloumb kernel (or Greens function)
! for the angular quantum number L
!
!  G_L(r',r) =  4 pi/(2L+1) (r',r)<^(L) /(r',r)>^(L+1) 
! (essentially identical to RAD_POT_HAR )
!
! ********************************************************************************

    SUBROUTINE RAD_POT_EX_HAR (RHO,R,LL,POT)
      USE constant
      USE radial
      USE pseudo 
      USE RS_GreensFunc_kernel, ONLY : SET_RSGF
      IMPLICIT NONE
 
      REAL(q) :: RHO(:)
      TYPE (rgrid) R
      INTEGER LL
      REAL(q) :: POT(:)
      !local variables
      REAL(q) T1(R%NMAX),T2(R%NMAX),V1(R%NMAX),V2(R%NMAX),RL(R%NMAX)
      REAL(q) DHARTREE,H3,EXT
      INTEGER N,I,J,K,L,JBOUND
      ! variables for range-separated HF hybrids
      REAL(q) :: RMU     ! range seperation parameter
      REAL(q) :: BRSEXX  ! amount of range seperated exchange
      REAL(q) :: BEXX    ! amount of exact exchange
      REAL(q), ALLOCATABLE :: GRS(:,:)
      LOGICAL LRHF
#ifdef debug
      REAL(q) :: POT2(SIZE(POT))
#endif
      CALL RSPARAMETERS(RMU, LRHF, BRSEXX, BEXX )

      IF (BRSEXX/=0) THEN
         ALLOCATE(GRS(R%NMAX,R%NMAX))
         ! get range-separated coulomb interaction green function
         CALL SET_RSGF(R,RMU,LL,LRHF,GRS)
!        CALL RS_COULOMB_GREEN_FUNC(R,RMU,LL,GRS,LRHF)
         ! calculate the radial potential
         DO I=1,R%NMAX
            ! first segment \int_0^r G(r,r') rho(r') dr'
            DO J=1,MIN(I+1,R%NMAX)
               T1(J)=RHO(J)*GRS(I,J)*R%R(J)
            ENDDO
            V1(1)=0
            DO L=3,MIN(I+1,R%NMAX),2
               V1(L)  =V1(L-2)+R%H/3.0_q*(T1(L-2)+4.0_q*T1(L-1)+T1(L))
               V1(L-1)=V1(L-2)+R%H/3.0_q*(1.25_q*T1(L-2)+2.0_q*T1(L-1)-0.25_q*T1(L))               
            ENDDO
            ! second segment \int_r^Rmax G(r',r) rho(r') dr'
            DO J=R%NMAX,MAX(I-1,1),-1
               T2(J)=RHO(J)*GRS(J,I)*R%R(J)
            ENDDO
            V2(R%NMAX)=0
            DO L=R%NMAX-2,MAX(I-1,1),-2
               V2(L)  =V2(L+2)+R%H/3.0_q*(T2(L+2)+4.0_q*T2(L+1)+T2(L))
               V2(L+1)=V2(L+2)+R%H/3.0_q*(1.25_q*T2(L+2)+2.0_q*T2(L+1)-0.25_q*T2(L))
            ENDDO
#ifdef debug
            POT2(I)=V2(I)+V1(I)
#endif
            POT(I)=V2(I)+V1(I)
         ENDDO
         POT(:)=POT(:)*FELECT*4*PI/(2*LL+1)*BRSEXX
         DEALLOCATE(GRS)
      ELSE 
         POT=0
      ENDIF

      IF (BEXX/=0) THEN
         N=R%NMAX      
         I=0
         DO K=N,1,-1
            RL(K)=R%R(K)**LL
            I=I+1
            T2(I)=RHO(K)/RL(K)
            T1(I)=RL(K)*R%R(K)*RHO(K)
         ENDDO
         H3=R%H/ 3.0_q
         ! integrate inward (assuming zero potential for grid point NMAX)
         ! V1 = \int_R^Inf RHO(r) r^l dr
         ! V1 is essentially the moment l of the charge
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
         
         EXT=V1(N)
         I=0
         DO K=N,1,-1
            I=I+1
            POT(I)=POT(I)+(V2(K)*RL(I)+(EXT-V1(K))/(R%R(I)*RL(I)))*(FELECT*4*PI/(2*LL+1))*BEXX
         ENDDO
#ifdef debug
         POT2(:)=POT2(:)*FELECT*4*PI/(2*LL+1)
         WRITE(110,'(A,I4,A,I3)') 'GRIDSIZE=',R%NMAX,'  L=',LL
         WRITE(110,'(5F14.7)') &
              &   (R%R(I),RHO(I),POT(I),POT2(I),POT(I)-POT2(I), I=1,R%NMAX,1)
#endif
      ENDIF
      RETURN
    END SUBROUTINE RAD_POT_EX_HAR


! **************************** SUBROUTINE CALC_DHARTREE ****************************
!
!   calculates and stores the PAW strenght matrix D_ij for the exact Hartree and 
!   Fock Hamiltonian
!
!   D(Hartree)_12 = sum_34 K_1234*rho_34, where 
!   D(Fock)_12    = sum_34 K_1432*rho_34, where 
!
!   K_1234 = sum_LM c_LM,l1m1,l2m2 * S_n1n2n3n4L * c_LM,l3m3,l4m4 
!
!   c_LM,l1m1,l2m2 are the Gaunt coefficients stored in YLM3
!   S_n1n2n3n4   are the four electron Coloumb integrals
!   1,2,3 and 4  are compound indices including the channel and l,m indices
!                e.g. 1=n1l1m1
!
! **********************************************************************************

    SUBROUTINE CALC_DHARTREE(S,COCC,CHANNELS,LPS,DHARTREE,DFOCK)
      USE pseudo
      USE constant
      USE asa
      IMPLICIT NONE
      
      REAL(q) :: S(:,:,:,:,0:)
      INTEGER CHANNELS
      INTEGER LPS(:)
      OVERLAP :: COCC(:,:)
      OVERLAP :: DHARTREE(:,:), DFOCK(:,:)

     !local variables
      INTEGER I,J,K,L,LI,LJ,LK,LL,MI,MJ,MK,ML,ISTART1,ISTART2,IEND1,IEND2,LMIND1,LMIND2,IC1,IC2
      INTEGER LMI,LMJ,LMK,LML

      DHARTREE=0
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
! loop k,l
!-----------------------------------------------------------------------
         LMK=1
         DO K=1,CHANNELS
         LML=1
         DO L=1,CHANNELS

            CALL YLM3LOOKUP(LI,LJ,LMIND1)
            LK = LPS(K)
            LL = LPS(L)

            ! loop mi,mj

            DO MI=1,2*LI+1
            DO MJ=1,2*LJ+1
               LMIND1 = LMIND1+1
 
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               CALL YLM3LOOKUP(LK,LL,LMIND2)
               ! loop mk,ml
               DO MK=1,2*LK+1
               DO ML=1,2*LL+1
                  LMIND2 = LMIND2+1
 
                  ISTART2=INDCG(LMIND2)
                  IEND2=INDCG(LMIND2+1)
 
                  DO IC1=ISTART1,IEND1-1  !IC Comp.Ind. LM
                  DO IC2=ISTART2,IEND2-1
                  IF (JS(IC2) == JS(IC1)) THEN

! **********************************************************************************
!     K_IJKL = Sum_{LM} c_LM,ij * S_ijklL * c_LM,kl                                
!     KFOCK_ILKJ = Sum_{LM} c_LM,ij * S_ijklL * c_LM,kl  (S_ijkl = F_ilkj !)       
!     I is actually {ni,li,mi}                                                   

!     DHARTREE_IJ = Sum_{KL} K_IJKL * RHO_KL                                     
!     DFOCK_IL = Sum_{KJ} KFOCK_ILKJ * RHO_KJ                                    
!     DIM (DHARTREE, DFOCK) = DIM (RHO_KL)                                       
! **********************************************************************************
                     DHARTREE(LMI+MI-1,LMJ+MJ-1) = DHARTREE(LMI+MI-1,LMJ+MJ-1) &
                        +YLM3(IC1)*S(I,J,K,L,JL(IC2)/2)*YLM3(IC2)*COCC(LMK+MK-1,LML+ML-1)
                     DFOCK(LMI+MI-1,LML+ML-1) = DFOCK(LMI+MI-1,LML+ML-1) &
                        - YLM3(IC1)*S(I,J,K,L,JL(IC2)/2)*YLM3(IC2)*COCC(LMK+MK-1,LMJ+MJ-1) 
                  ENDIF
                  ENDDO
                  ENDDO
               ENDDO ! ML
               ENDDO ! MK

            ENDDO ! MI
            ENDDO ! MJ

            LML=LML+2*LL+1
         ENDDO ! L
         LMK=LMK+2*LK+1
         ENDDO ! K

         LMJ=LMJ+2*LJ+1
      ENDDO ! J

      LMI=LMI+2*LI+1
      ENDDO ! I
 
    END SUBROUTINE CALC_DHARTREE



!************************ SUBROUTINE COLOUMB_2POT  *******************************
!
!  calculate and stores the two electron Coloumb potentials
!
!  POT_L = \int dr G_L(r',r) phi_l(r) phi_k(r)
!
!  where G_L(r',r) is the Coloumb kernel (or Greens function)
!  for the angular quantum number L
!
!  G_L(r',r) =  4 pi/(2L+1) (r',r)<^(L) /(r',r)>^(L+1) 
!
!  the subroutine COLOUMB_2POT_PS adds the local augmentation charges
!
!*********************************************************************************

    SUBROUTINE COLOUMB_2POT (W1, W2,  L1, L2,  R, POT)
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      REAL(q) :: W1(:),W2(:)   ! wavefunctions for each channel
      INTEGER :: L1, L2        ! l-quantum number of each channel
      TYPE (rgrid) R           ! grid structure
      REAL(q) :: POT(:,0:)     ! potential
       
       ! local variables
      INTEGER I, LLOOP,LMIN,LMAX, LDIM
      REAL(q) :: RHO(R%NMAX)

      LDIM=SIZE(POT,2)
 
      LMIN = ABS(L1-L2) ; LMAX = ABS(L1+L2)
      IF (LMAX/2>LDIM) THEN
         WRITE(0,*) 'internal ERROR in COLOUMB_2POT: LDIM is not sufficiently large',LMAX/2,LDIM
         STOP
      ENDIF

      ! calculate rho_kl(r)
      DO I=1,R%NMAX
         RHO(I)=W1(I)*W2(I)
      ENDDO
         
      POT=0
      ! loop over L
      DO LLOOP=LMIN,LMAX,2
         CALL RAD_POT_EX_HAR(RHO,R,LLOOP,POT(:,LLOOP/2))
      ENDDO
    END SUBROUTINE COLOUMB_2POT

    SUBROUTINE COLOUMB_2POT_PS (W1, W2,  AUG, QPAW, L1, L2,  R, POT, & 
         NMAX_FOCKAE, LMAX_FOCKAE, AUG_FOCK, QPAW_FOCK )
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      REAL(q) :: W1(:),W2(:)   ! wavefunctions for each channel
      INTEGER :: L1, L2        ! l-quantum number of each channel
      TYPE (rgrid) R           ! grid structure
      REAL(q) :: POT(:,0:)     ! potential
      REAL(q) :: AUG(:,0:)     ! 1-normalized L-dep compensation charge
      REAL(q) :: QPAW(0:)      ! moments of compensation charge Q(PAW,ll L)
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE
      REAL(q), OPTIONAL :: AUG_FOCK(:,0:,:)! 1-normalized L-dep compensation charge
      REAL(q), OPTIONAL :: QPAW_FOCK(0:,:) ! moments of compensation charge Q(PAW,ll L)
      
       ! local variables
      INTEGER I, LLOOP, LMIN, LMAX, LDIM, NAE
      REAL(q) :: RHO(R%NMAX)

      LDIM=SIZE(POT,2)
 
      LMIN = ABS(L1-L2) ; LMAX = ABS(L1+L2)
      IF (LMAX/2>LDIM) THEN
         WRITE(0,*) 'internal ERROR in COLOUMB_2POT_PS: LDIM is not sufficiently large',LMAX/2,LDIM
         STOP
      ENDIF

      POT=0
      ! loop over L
      DO LLOOP=LMIN,LMAX,2
         ! calculate rho_kl(r)
         DO I=1,R%NMAX
            RHO(I)=W1(I)*W2(I)+AUG(I,LLOOP)*QPAW(LLOOP)
         ENDDO
         IF (PRESENT(QPAW_FOCK) .AND. LLOOP <= LMAX_FOCKAE) THEN
            DO NAE=1,NMAX_FOCKAE
               DO I=1,R%NMAX
                  RHO(I)=RHO(I)+AUG_FOCK(I,LLOOP,NAE)*QPAW_FOCK(LLOOP,NAE)
               ENDDO
            ENDDO
         ENDIF
         CALL RAD_POT_EX_HAR(RHO,R,LLOOP,POT(:,LLOOP/2))
      ENDDO
    END SUBROUTINE COLOUMB_2POT_PS


!************************ SUBROUTINE ADD_WEIGHTED_POT_AE ************************
!
!  adds the density weighted exchange potential to the average potential 
!
!  POT_AVERAGE = SUM_L WEIGHT(L)  phi_l(r) phi_k(r) POT_L 
!
!*********************************************************************************

    SUBROUTINE ADD_WEIGHTED_POT_AE(W1, W2,  L1, L2,  R, & 
         WEIGHT, POT_ADD, POT )
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      REAL(q) :: W1(:),W2(:)   ! wavefunctions for each channel
      INTEGER :: L1, L2        ! l-quantum number of each channel
      TYPE (rgrid) R           ! grid structure
      REAL(q) :: WEIGHT(0:)
      REAL(q) :: POT_ADD(:)
      REAL(q),OPTIONAL :: POT(:,0:)     ! potential
       
       ! local variables
      INTEGER LLOOP, I, LMIN, LMAX
      REAL(q) :: RHO(R%NMAX)

      LMIN = ABS(L1-L2) ; LMAX = MIN(ABS(L1+L2),UBOUND(WEIGHT,1)*2)
      ! calculate rho_kl(r)
      DO I=1,R%NMAX
         RHO(I)=W1(I)*W2(I)
      ENDDO

      ! loop over L
      DO LLOOP=LMIN,LMAX,2
         IF (ABS(WEIGHT(LLOOP/2))>=1E-10) THEN
            IF (PRESENT(POT)) THEN
               DO I=1,R%NMAX
                  POT_ADD(I)=POT_ADD(I)+(POT(I,LLOOP/2)*WEIGHT(LLOOP/2))*RHO(I)
               ENDDO
            ELSE
               DO I=1,R%NMAX
                  POT_ADD(I)=POT_ADD(I)+WEIGHT(LLOOP/2)*RHO(I)
               ENDDO
            ENDIF
         ENDIF
      ENDDO

    END SUBROUTINE ADD_WEIGHTED_POT_AE

    SUBROUTINE ADD_WEIGHTED_POT_PS(W1, W2, AUG, QPAW, L1, L2,  R, &
         WEIGHT, POT_ADD, NMAX_FOCKAE, LMAX_FOCKAE, POT, AUG_FOCK, QPAW_FOCK )
      USE radial
      USE pseudo
      USE constant
      IMPLICIT NONE
 
      REAL(q) :: W1(:),W2(:)   ! wavefunctions for each channel
      INTEGER :: L1, L2        ! l-quantum number of each channel
      REAL(q) :: AUG(:,0:)     ! 1-normalized L-dep compensation charge
      REAL(q) :: QPAW(0:)      ! moments of compensation charge Q(PAW,ll L)
      TYPE (rgrid) R           ! grid structure
      REAL(q) :: WEIGHT(0:)
      REAL(q) :: POT_ADD(:)
      INTEGER :: NMAX_FOCKAE, LMAX_FOCKAE
      REAL(q), OPTIONAL :: POT(:,0:)     ! potential
      REAL(q), OPTIONAL :: AUG_FOCK(:,0:,:)! 1-normalized L-dep compensation charge
      REAL(q), OPTIONAL :: QPAW_FOCK(0:,:) ! moments of compensation charge Q(PAW,ll L)
       
       ! local variables
      INTEGER LLOOP, I, LMIN, LMAX, NAE
      REAL(q) :: RHO(R%NMAX)

      LMIN = ABS(L1-L2) ; LMAX = MIN(ABS(L1+L2),UBOUND(WEIGHT,1)*2)
      
      ! loop over L
      DO LLOOP=LMIN,LMAX,2
         IF (ABS(WEIGHT(LLOOP/2))>=1E-10) THEN
            ! calculate rho_kl(r)
            DO I=1,R%NMAX
               RHO(I)=W1(I)*W2(I)+AUG(I,LLOOP)*QPAW(LLOOP)
            ENDDO
            IF (PRESENT(QPAW_FOCK) .AND. LLOOP <= LMAX_FOCKAE) THEN
               DO NAE=1,NMAX_FOCKAE
                  DO I=1,R%NMAX
                     RHO(I)=RHO(I)+AUG_FOCK(I,LLOOP,NAE)*QPAW_FOCK(LLOOP,NAE)
                  ENDDO
               ENDDO
            ENDIF

            IF (PRESENT(POT)) THEN
               DO I=1,R%NMAX
                  POT_ADD(I)=POT_ADD(I)+(POT(I,LLOOP/2)*WEIGHT(LLOOP/2))*RHO(I)
               ENDDO
            ELSE
               DO I=1,R%NMAX
                  POT_ADD(I)=POT_ADD(I)+WEIGHT(LLOOP/2)*RHO(I)
               ENDDO
            ENDIF
         ENDIF
      ENDDO

    END SUBROUTINE ADD_WEIGHTED_POT_PS

  END MODULE PAWFOCK


! **********************************************************************************
!
! interface module that keeps track of the four electron
! Coloumb (Slater) integrals and of the type for which they have been set up in the 
! internal hash array S
!
! **********************************************************************************

  MODULE PAWFOCK_INTER
    USE prec
    USE pawfock
    USE RS_GreensFunc_kernel, ONLY : SET_RSGF_TYPE,UNSET_RSGF_TYPE,DEALLOCATE_RSGF
    IMPLICIT NONE

    INTEGER,SAVE :: NTYP_SLATER=0
    REAL(q),ALLOCATABLE,SAVE :: S(:,:,:,:,:)
  CONTAINS


! **********************************************************************************
!    SETUP_PAWFOCK
!
! setup the array S and the NTYP_SLATER variable
!
! **********************************************************************************

    SUBROUTINE SETUP_PAWFOCK(NT, PP)
      USE pseudo
      TYPE (potcar),POINTER:: PP
      INTEGER NT
    ! local
      INTEGER LYMAX
      INTEGER CHANNELS
      INTEGER, EXTERNAL :: MAXL1, FOCK_LMAXONECENTER

      IF (NT==NTYP_SLATER) RETURN

      IF (ALLOCATED(S)) THEN
         DEALLOCATE(S)
      ENDIF

      LYMAX =MAXL1(PP)*2/2   ! LYMAX is half the maximum L augmentation number
                             ! since the L index in S is divided by 2
      CHANNELS=PP%LMAX

      ALLOCATE(S(CHANNELS, CHANNELS, CHANNELS, CHANNELS, 0:LYMAX))
      S=0
      ! there are many options and combinations for the maximum L quantum number
      ! on the plane wave grid and for handling the one center terms
      ! currently the one center terms are calculated up to the maximum possible L
      ! for pseudo and AE contributions
      !
      ! But maybe one should truncate at FOCK_LMAXONECENTER (maximum L on plane wave grid)
      ! or one should truncate only the pseudo term at  FOCK_LMAXONECENTER
      ! the last choice makes a lot of sense for localized electrons
      ! but does not work well for most systems
      !
      ! Currently for the one center terms all possible L are included 
      ! this overall seems to work best and just implies neglecting higher
      ! L quantum numbers on the plane wave grid
      !
      ! maximum value of 8 suffices for g
      ! CALL COLOUMB_4TERM_PS( PP%WPS, PP%LPS, PP%R, CHANNELS, PP%AUG_SOFT, PP%QPAW, S, FOCK_LMAXONECENTER(), & 
      !                        PP%AUG_FOCK, PP%QPAW_FOCK )
      CALL COLOUMB_4TERM_PS( PP%WPS, PP%LPS, PP%R, CHANNELS, PP%AUG_SOFT, PP%QPAW, S, 8, &
                             PP%AUG_FOCK, PP%QPAW_FOCK )
      S=-S
      CALL COLOUMB_4TERM( PP%WAE, PP%LPS, PP%R, CHANNELS, S, 8)

      NTYP_SLATER=NT
      
    END SUBROUTINE SETUP_PAWFOCK

! **********************************************************************************
!    SETUP_PAWFOCK_AE
!
! setup the array S and the NTYP_SLATER variable
! for  the AE wavefunctions only (i.e. not compensated by pseudo contribution)
! 
! **********************************************************************************

    SUBROUTINE SETUP_PAWFOCK_AE(NT, PP)
      USE pseudo
      TYPE (potcar),POINTER:: PP
      INTEGER NT
    ! local
      INTEGER LYMAX
      INTEGER CHANNELS
      INTEGER, EXTERNAL :: MAXL1

      IF (NT==NTYP_SLATER) RETURN

      IF (ALLOCATED(S)) THEN
         DEALLOCATE(S)
      ENDIF

      LYMAX =MAXL1(PP)*2/2   ! LYMAX is half the maximum L augmentation number
                             ! since the L index in S is divided by 2
      CHANNELS=PP%LMAX

      ALLOCATE(S(CHANNELS, CHANNELS, CHANNELS, CHANNELS, 0:LYMAX))
      S=0
      CALL COLOUMB_4TERM( PP%WAE, PP%LPS, PP%R, CHANNELS, S, 8)
      NTYP_SLATER=NT
      
    END SUBROUTINE SETUP_PAWFOCK_AE

! **********************************************************************************
!    SETUP_PAWFOCK_PS
!
! setup the array S and the NTYP_SLATER variable
! for  the PS wavefunctions only (i.e. not compensated by AE contribution)
! 
! **********************************************************************************

    SUBROUTINE SETUP_PAWFOCK_PS(NT, PP)
      USE pseudo
      TYPE (potcar),POINTER:: PP
      INTEGER NT
    ! local
      INTEGER LYMAX
      INTEGER CHANNELS
      INTEGER, EXTERNAL :: MAXL1

      IF (NT==NTYP_SLATER) RETURN

      IF (ALLOCATED(S)) THEN
         DEALLOCATE(S)
      ENDIF

      LYMAX =MAXL1(PP)*2/2   ! LYMAX is half the maximum L augmentation number
                             ! since the L index in S is divided by 2
      CHANNELS=PP%LMAX

      ALLOCATE(S(CHANNELS, CHANNELS, CHANNELS, CHANNELS, 0:LYMAX))
      S=0
      CALL COLOUMB_4TERM_PS( PP%WPS, PP%LPS, PP%R, CHANNELS, PP%AUG_SOFT, PP%QPAW, S, 8, &
                             PP%AUG_FOCK, PP%QPAW_FOCK )


      NTYP_SLATER=NT
      
    END SUBROUTINE SETUP_PAWFOCK_PS


! **********************************************************************************
!    RELEASE_PAWFOCK
!
! deallocate  the array S and set the NTYP_SLATER varibale to 0
!
! **********************************************************************************

    SUBROUTINE RELEASE_PAWFOCK
      IF (ALLOCATED(S)) THEN
         DEALLOCATE(S)
      ENDIF
      NTYP_SLATER=0
    END SUBROUTINE RELEASE_PAWFOCK


! **********************************************************************************
!    SETUP_PAWFOCK_MATRIX
!
! setup a square matrix consisting of the four orbital integrals
! the rank of the matrix is ENTRIES_FOR_TYPE, and both indices run from
! one to ENTRIES_FOR_TYPE
! CHANNEL1, CHANNEL2 and L store original CHANNEL indices and the L quantum number
! i.e. the compound index is  i=(channel1,channel2,L,M)
! return:
!  POT   four orbital integrals 
!        a one dimensional array is used here entries 
!        I1+ENTRIES_FOR_TYPE*(I2-1)
!
! **********************************************************************************

    SUBROUTINE SETUP_PAWFOCK_MATRIX( ENTRIES_FOR_TYPE, CHANNEL1, CHANNEL2, L, LM, & 
         PP, POT)
      USE pseudo
      INTEGER :: ENTRIES_FOR_TYPE
      INTEGER :: CHANNEL1(:), CHANNEL2(:), L(:), LM(:)
      INTEGER :: NT
      TYPE (potcar),POINTER:: PP
      GDEF    :: POT(:)
    ! local
      INTEGER :: I1, I2

      DO I1=1, ENTRIES_FOR_TYPE
         DO I2=1, ENTRIES_FOR_TYPE
            IF (LM(I1)==LM(I2)) THEN
               POT(I1+ENTRIES_FOR_TYPE*(I2-1))=S(CHANNEL1(I1),CHANNEL2(I1),CHANNEL1(I2),CHANNEL2(I2),L(I1)/2)
            ELSE
               POT(I1+ENTRIES_FOR_TYPE*(I2-1))=0
            ENDIF
         ENDDO
      ENDDO

    END SUBROUTINE SETUP_PAWFOCK_MATRIX


! **********************************************************************************
!   CALC_PAWFOCK
!
! calculate the Hartree Fock PAW term and the corresponding double counting 
! corrections
!
! **********************************************************************************

    SUBROUTINE CALC_PAWFOCK(NT, PP, COCC, DFOCK, DOUBLEC_HF)
      USE pseudo
      IMPLICIT NONE

      INTEGER NT
      TYPE (potcar), POINTER ::  PP
      OVERLAP :: COCC(:,:,:),  DFOCK(:,:,:)
      REAL(q) :: DOUBLEC_HF
      ! local
      INTEGER :: I, NCDIJ, NCDIJ_STEP
      OVERLAP :: DHARTREE(SIZE(DFOCK,1),SIZE(DFOCK,1))
      INTEGER :: LM1, LM2
      REAL(q) :: HF_ENERGY

      ! setup the 
      CALL SETUP_PAWFOCK(NT, PP)
! test use AE contribution only
!     CALL RELEASE_PAWFOCK
!     CALL SETUP_PAWFOCK_AE(NT, PP)

      ! determine the last dimension of the DFOCK matrix (NCDIJ)
      NCDIJ=SIZE(DFOCK,3)
      ! step width is usually 1
      NCDIJ_STEP=1
      ! for the non collinear case, only (up,up) and (down,down) must be considered
      IF (NCDIJ==4) NCDIJ_STEP=1

      
      DO I=1,NCDIJ,NCDIJ_STEP
         CALL CALC_DHARTREE(S, COCC(:,:,I), PP%LMAX,  PP%LPS, DHARTREE, DFOCK(:,:,I))
         !CALL DUMP_DLLMM( "hartree",DHARTREE, PP)
         !CALL DUMP_DLLMM( "fock   ",DFOCK(:,:,I)   , PP)
      ENDDO

      HF_ENERGY=0
      DO I=1,NCDIJ,NCDIJ_STEP
         DO LM1=1,PP%LMMAX
         DO LM2=1,PP%LMMAX
#ifdef realmode
            HF_ENERGY=HF_ENERGY+ &
                 DFOCK(LM1,LM2,I)*COCC(LM1,LM2,I)
#else
            HF_ENERGY=HF_ENERGY+ &
                 DFOCK(LM1,LM2,I)*CONJG(COCC(LM1,LM2,I))
#endif
         ENDDO
         ENDDO
      ENDDO

      ! non spin polarized
      ! in this case there would be a second contribution from the
      ! second spin (which is not explicitly calculated)
      IF (NCDIJ==1) THEN
         HF_ENERGY=HF_ENERGY*2
      ENDIF
      DOUBLEC_HF=-HF_ENERGY/2
!      WRITE(*,1)  DOUBLEC_HF
1     FORMAT(' -1/2 Fock energy',6F14.7)
    END SUBROUTINE CALC_PAWFOCK

  END MODULE PAWFOCK_INTER
