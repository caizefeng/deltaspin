#include "symbol.inc"
!***********************SUBROUTINE CHCJG ******************************
! RCS:  $Id: mix.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
! subroutine performes a steeptes descent or conjugate gradient
! step to optimize the chargedensity
! the method introduced by G. Kresse
!
! the routine uses one flags to exchange information with the main
! program
!  IMIX       0  initialisation main program supplies the old
!                chargedensity and old energy
!             1  conventional Kerker-mixing
!  B          recip. lattice  vectors
!  RMST       norm of residual vector
!
!  Mind: RMST must be set to 0 before call
!
!**********************************************************************

      SUBROUTINE MIX_SIMPLE(GRIDC,MIX,ISPIN, CHTOT,CHTOTL, &
                       N_MIX_PAW, RHOLM, RHOLM_LAST, B, OMEGA, RMST)
      USE prec

      USE mpimy
      USE mgrid
      USE base
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRIDC
      TYPE (mixing)  MIX

      COMPLEX(q) CHTOT(GRIDC%MPLWV,ISPIN),CHTOTL(GRIDC%MPLWV,ISPIN)
      REAL(q)    B(3,3)
      REAL(q)    RHOLM(N_MIX_PAW,ISPIN),RHOLM_LAST(N_MIX_PAW,ISPIN)
      REAL(q)    OMEGA
      COMPLEX(q),SAVE, ALLOCATABLE :: CHVEL(:,:)
      REAL(q),SAVE, ALLOCATABLE    :: RHOVEL(:,:)
      REAL(q)  :: MU

      LOGICAL,SAVE :: INI=.TRUE.
      IF (INI .AND. MIX%IMIX==2) THEN
        ALLOCATE(CHVEL(GRIDC%RC%NP,ISPIN)) ;  INI=.FALSE.
        CHVEL=0
        IF (N_MIX_PAW>0) THEN
           ALLOCATE(RHOVEL(N_MIX_PAW,ISPIN))
           RHOVEL=0
        ENDIF
      ENDIF

      FAKT=1._q/OMEGA
!=======================================================================
! IMIX==0
! on init just copy CHTOT to CHTOTL
!=======================================================================
      IF (MIX%IMIX==0) THEN
          CHTOTL(1:GRIDC%RC%NP,:)=CHTOT(1:GRIDC%RC%NP,:)
          RHOLM_LAST=RHOLM
!======================================================================
! IMIX==1
! conventional Kerker-mixing
!======================================================================
      ELSE IF (MIX%IMIX==1) THEN
      AMIX=MIX%AMIX
      BMIX=MIX%BMIX

      DO ISP=1,ISPIN
      FLAM=BMIX**2
      DO K=1,GRIDC%RC%NP
        N1= MOD((K-1),GRIDC%RC%NROW) +1
        NC= (K-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        SETFACT1
        SETFACT

        GX= (GRIDC%LPCTX(N1)*B(1,1)+GRIDC%LPCTY(N2)*B(1,2)+GRIDC%LPCTZ(N3)*B(1,3))
        GY= (GRIDC%LPCTX(N1)*B(2,1)+GRIDC%LPCTY(N2)*B(2,2)+GRIDC%LPCTZ(N3)*B(2,3))
        GZ= (GRIDC%LPCTX(N1)*B(3,1)+GRIDC%LPCTY(N2)*B(3,2)+GRIDC%LPCTZ(N3)*B(3,3))

        GSQU =(GX**2+GY**2+GZ**2)*TPI*TPI
        IF (N1==1 .AND. N2==1 .AND. N3==1) GSQU=1E30_q
        CHNEW    =CHTOTL(K,ISP)+AMIX*GSQU/(GSQU+FLAM)*(CHTOT(K,ISP)-CHTOTL(K,ISP))
        RMST     =RMST+ FAKT*MULFACT &
           CONJG((CHTOT(K,ISP)-CHTOTL(K,ISP)))*(CHTOT(K,ISP)-CHTOTL(K,ISP))
        CHTOTL(K,ISP)=CHTOT(K,ISP)
        CHTOT (K,ISP)=CHNEW
      ENDDO

      AMIX_PAW=AMIX
      DO K=1,N_MIX_PAW
         DEL =(RHOLM(K,ISP)-RHOLM_LAST(K,ISP))
         RMST=RMST+DEL*DEL
         RHO_NEW=RHOLM_LAST(K,ISP)+DEL*AMIX_PAW
         RHOLM_LAST(K,ISP)=RHOLM(K,ISP)
         RHOLM(K,ISP)=RHO_NEW
      ENDDO

      AMIX=MIX%AMIX_MAG
      BMIX=MIX%BMIX_MAG

      ENDDO

      CALLMPI( M_sum_d(GRIDC%COMM, RMST, 1))
      RMST=SQRT(RMST)
!======================================================================
! IMIX==2
! Tshebyshef Kerker-mixing (velocity damping algorithm)
!======================================================================
      ELSE IF (MIX%IMIX==2) THEN
      AMIX=MIX%AMIX
      BMIX=MIX%BMIX
      MU =MIX%AMIN

      DO ISP=1,ISPIN
      FLAM=MIX%BMIX**2
      DO K=1,GRIDC%RC%NP
        N1= MOD((K-1),GRIDC%RC%NROW) +1
        NC= (K-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        SETFACT1
        SETFACT

        GX= (GRIDC%LPCTX(N1)*B(1,1)+GRIDC%LPCTY(N2)*B(1,2)+GRIDC%LPCTZ(N3)*B(1,3))
        GY= (GRIDC%LPCTX(N1)*B(2,1)+GRIDC%LPCTY(N2)*B(2,2)+GRIDC%LPCTZ(N3)*B(2,3))
        GZ= (GRIDC%LPCTX(N1)*B(3,1)+GRIDC%LPCTY(N2)*B(3,2)+GRIDC%LPCTZ(N3)*B(3,3))

        RMST     =RMST+ FAKT* MULFACT &
           CONJG((CHTOT(K,ISP)-CHTOTL(K,ISP)))*(CHTOT(K,ISP)-CHTOTL(K,ISP))

        GSQU =(GX**2+GY**2+GZ**2)*TPI*TPI
        IF (N1==1 .AND. N2==1 .AND. N3==1) GSQU=1E30_q
        CACC = AMIX*GSQU/(GSQU+FLAM)*(CHTOT(K,ISP)-CHTOTL(K,ISP))

        CHVEL(K,ISP)= ((1-MU/2) * CHVEL(K,ISP) + 2* CACC)/(1+MU/2)
        CHNEW=CHTOTL(K,ISP)+CHVEL(K,ISP)

        CHTOTL(K,ISP)=CHTOT(K,ISP)
        CHTOT (K,ISP)=CHNEW
      ENDDO
      AMIX_PAW=AMIX
      DO K=1,N_MIX_PAW
         DEL =(RHOLM(K,ISP)-RHOLM_LAST(K,ISP))
         RMST=RMST+DEL*DEL
         RHOVEL(K,ISP)= ((1-MU/2) * RHOVEL(K,ISP) + 2* DEL*AMIX_PAW)/(1+MU/2)

         RHO_NEW=RHOLM_LAST(K,ISP)+RHOVEL(K,ISP)
         RHOLM_LAST(K,ISP)=RHOLM(K,ISP)
         RHOLM(K,ISP)=RHO_NEW
      ENDDO

      AMIX=MIX%AMIX_MAG
      BMIX=MIX%BMIX_MAG
      ENDDO
      CALLMPI( M_sum_d(GRIDC%COMM, RMST, 1))

      RMST=SQRT(RMST)
      ENDIF


      RETURN
      END
