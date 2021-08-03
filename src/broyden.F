#include "symbol.inc"
! RCS:  $Id: broyden.F,v 1.3 2002/04/16 07:28:37 kresse Exp $
#define USE_TMPBROYD
!
! you have the option to do the mixing on disc or in storage
! if USE_TMPBROYD is defined required arrays are stored on disc
! if it is not defined dynamic arrays are allocated as required
#undef USE_TMPBROYD

      MODULE broyden
      USE prec
      INCLUDE "broyden.inc"
      SAVE tmp_storage
      SAVE broyden_storage
      CONTAINS

!***********************************************************************
!
!  calling interface to BROYD
!  this routine extracts the chargedensity on the small grid
!  and calls the broyden-mixing
!  for wavevectors, which are not contained in the small grid a simple
!  mixing is used
! NGX,NGY,NGZ     is the reduced grid on which Broyden mixing is done
!                 corresponds to NGXB,NGYB,NGZB in main program
! NGXC,NGYC,NGZC  is the full grid
!
!***********************************************************************

      SUBROUTINE BRMIX( &
         KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
         NGIGA,CHTOT,CHTOTL,ISPIN,B,OMEGA, &
         N_MIX_PAW, RHOLM, RHOLM_LAST, &
         RMST,RMS,RMSP,WEIGHT,LWARN,IERRBR, &
         N_RHO_ONE_CENTRE, RHO_ONE_CENTRE , RHO_ONE_CENTRE_LAST )
      USE prec

      USE base
      USE mpimy
      USE mgrid
      USE charge
      USE meta
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (tau_handle)  KINEDEN
      TYPE (grid_3d)     GRIDB
      TYPE (grid_3d)     GRIDC
      TYPE (transit)     B_TO_C
      TYPE (in_struct)   IO
      TYPE (mixing)      MIX

      COMPLEX(q)   CHTOT(GRIDC%MPLWV,ISPIN),CHTOTL(GRIDC%MPLWV,ISPIN)
      INTEGER      N_MIX_PAW
      REAL(q)      RHOLM(N_MIX_PAW,ISPIN),RHOLM_LAST(N_MIX_PAW,ISPIN)
      INTEGER,OPTIONAL :: N_RHO_ONE_CENTRE
      REAL(q),OPTIONAL :: RHO_ONE_CENTRE(:),RHO_ONE_CENTRE_LAST(:)
      REAL(q)      B(3,3)
      LOGICAL LWARN
! local variables
      REAL(q)      OMEGA,FAKT,ISFAKT,SFAKT
      SAVE IERR
      DATA IERR /0/
      INTEGER NDATA,NCHARGE
      COMPLEX(q),ALLOCATABLE :: CWRK1(:),CWRK2(:),CWRK3(:),CWRK4(:),CHP(:)

      PROFILING_START('brmix')

      ! number of grid points to be mixed
      NCHARGE= GRIDB%RC%NP*ISPIN
      ! double NCHARGE to hold tau for metaGGA's
      IF (LDO_METAGGA().AND.LMIX_TAU()) NCHARGE=2*NCHARGE
      ! add number of occupancies to be mixed
      NDATA  = NCHARGE+N_MIX_PAW*ISPIN
      ! and the one-center density for the relaxed-core PAW
      IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
         NDATA=NDATA+N_RHO_ONE_CENTRE
      ENDIF

      ALLOCATE( CWRK1(MAX(NGIGA,NDATA*2)),CWRK2(NDATA), &
                CWRK3(NDATA),CWRK4(NDATA),CHP(NDATA))

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME= GRIDB%COMM%NODE_ME
      IONODE = GRIDB%COMM%IONODE
#endif
      IDUMP=0     ! set to 1 on serial computers for add. inform.
      io_begin
      IDUMP=0     ! set to 1 on parallel computers
      io_end

      NUPDZ=-NGIGA/(MAX(NDATA,1))

      CALLMPI ( M_max_i( GRIDB%COMM, NUPDZ ,1)) ! global minimum ;-)
      NUPDZ=-NUPDZ

!      IF (NDATA > GRIDC%MPLWV) THEN
!         WRITE(*,*) 'Fatal error BRMIX: Insufficient workspace ...'
!         STOP
!      ENDIF

      FAKT=1._q/OMEGA

      TOTNEL=RHO0(GRIDC,CHTOT (1,1))
      TOTOLD=RHO0(GRIDC,CHTOTL(1,1))

! Set CHP (contains the metric to be used later):
      MIXPMA=MIX%MIXPRE
      IF (MIXPMA>=10) MIXPMA=MIXPMA-10
! charge density metric
      CALL BRPRE(GRIDB,CHP,B,MIXPMA,MIX%BMIX,MIX%LRESET.AND.(IDUMP/=0))

! magnetisation density metric
      MIXPMA=MIX%MIXPRE
      ! special treatment for potential mixing (MIXPMA>10, e.g. OEP): 
      ! in this case  both channels are treated in the same manner
      IF (MIXPMA>=10) THEN
        MIXPMA=MIXPMA-10
      ELSE
        MIXPMA=0._q
      ENDIF
      DO ISP=2,ISPIN
         CALL BRPRE(GRIDB,CHP(GRIDB%RC%NP*(ISP-1)+1),B,MIXPMA,MIX%BMIX,MIX%LRESET.AND.(IDUMP/=0))
      ENDDO

! kinetic energy density metric
      IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
         DO ISP=1,ISPIN
            CALL BRPRE(GRIDB,CHP(GRIDB%RC%NP*(ISP-1+ISPIN)+1),B,MIXPMA,MIX%BMIX,MIX%LRESET.AND.(IDUMP/=0))
!           CHP(GRIDB%RC%NP*(ISP-1+ISPIN)+1:GRIDB%RC%NP*(ISP+ISPIN))=CHP(GRIDB%RC%NP*(ISP-1+ISPIN)+1:GRIDB%RC%NP*(ISP+ISPIN))
            CHP(GRIDB%RC%NP*(ISP-1+ISPIN)+1:GRIDB%RC%NP*(ISP+ISPIN))=CHP(GRIDB%RC%NP*(ISP-1+ISPIN)+1:GRIDB%RC%NP*(ISP+ISPIN))*1.E-4_q
         ENDDO
      ENDIF

! Copy CHTOT to CWRK3 and CHTOTL to CWRK4 - reduction to a smaller mesh:

      NP=NCHARGE

      DO ISP=1,ISPIN
         ! copy charge and magnetization
         CALL CP_GRID(GRIDC,GRIDB,B_TO_C,CHTOT(1,ISP), CWRK3(GRIDB%RC%NP*(ISP-1)+1))
         CALL CP_GRID(GRIDC,GRIDB,B_TO_C,CHTOTL(1,ISP),CWRK4(GRIDB%RC%NP*(ISP-1)+1))
         ! copy kinetic energy density
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            CALL CP_GRID(GRIDC,GRIDB,B_TO_C,KINEDEN%TAU(1,ISP), CWRK3(GRIDB%RC%NP*(ISP-1+ISPIN)+1))
            CALL CP_GRID(GRIDC,GRIDB,B_TO_C,KINEDEN%TAUL(1,ISP),CWRK4(GRIDB%RC%NP*(ISP-1+ISPIN)+1))
         ENDIF
         ! one center occupancy matrix
         CWRK3(NP+1:NP+N_MIX_PAW)  =RHOLM(:,ISP)-RHOLM_LAST(:,ISP)
         CWRK4(NP+1:NP+N_MIX_PAW)  =RHOLM_LAST(:,ISP)
         CHP  (NP+1:NP+N_MIX_PAW)  =RHOLM(:,ISP)-RHOLM_LAST(:,ISP)
         NP=NP+N_MIX_PAW
      ENDDO

      ! additional terms for relaxed core methods
      IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
         CWRK3(NP+1:NP+N_RHO_ONE_CENTRE)=RHO_ONE_CENTRE-RHO_ONE_CENTRE_LAST
         CWRK4(NP+1:NP+N_RHO_ONE_CENTRE)=RHO_ONE_CENTRE_LAST
         CHP  (NP+1:NP+N_RHO_ONE_CENTRE)=RHO_ONE_CENTRE-RHO_ONE_CENTRE_LAST
         NP=NP+N_RHO_ONE_CENTRE
      ENDIF

      IF (NP /= NDATA) THEN
         WRITE(*,*) 'internal error 1 in BRMIX',NP,NDATA ; STOP
      ENDIF

! straight mixing for components which are not mixed by the Pulay mixer
      AMIX0=0.8_q
      BMIX0=0.001_q
      CALL SETG0(GRIDC,CWRK1,B,MIX%INIMIX,AMIX0,BMIX0,MIX%AMIN,.FALSE.)
      RMST=0._q
      RMS_tau=0._q
!-----------------------------------------------------------------------
! calculation of norm of residual on full grid !
! and mix all components
!-----------------------------------------------------------------------
      DO ISP=1,ISPIN
         ! charge density
         DO I=1,GRIDC%RC%NP
            N1= MOD((I-1),GRIDC%RC%NROW) +1
            NC= (I-1)/GRIDC%RC%NROW+1
            N2= GRIDC%RC%I2(NC)
            N3= GRIDC%RC%I3(NC)
            SETFACT1
            SETFACT
            RMST=RMST+FAKT*MULFACT CONJG(CHTOT(I,ISP)-CHTOTL(I,ISP))* &
     &           (CHTOT(I,ISP)-CHTOTL(I,ISP))
            ALPHA=CWRK1(I)
            CHTOT(I,ISP)=(1._q-ALPHA)*CHTOTL(I,ISP)+ALPHA*CHTOT(I,ISP)
         ENDDO
         ! kinetic energy density
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            DO I=1,GRIDC%RC%NP
               N1= MOD((I-1),GRIDC%RC%NROW) +1
               NC= (I-1)/GRIDC%RC%NROW+1
               N2= GRIDC%RC%I2(NC)
               N3= GRIDC%RC%I3(NC)
               SETFACT1
               SETFACT

!              RMST=RMST+FAKT*MULFACT CONJG(KINEDEN%TAU(I,ISP)-KINEDEN%TAUL(I,ISP))* &
!    &              (KINEDEN%TAU(I,ISP)-KINEDEN%TAUL(I,ISP))
               RMS_tau=RMS_tau+FAKT*MULFACT CONJG(KINEDEN%TAU(I,ISP)-KINEDEN%TAUL(I,ISP))* &
     &              (KINEDEN%TAU(I,ISP)-KINEDEN%TAUL(I,ISP))

               ALPHA=CWRK1(I)
               KINEDEN%TAU(I,ISP)=(1._q-ALPHA)*KINEDEN%TAUL(I,ISP)+ALPHA*KINEDEN%TAU(I,ISP)
            ENDDO
         ENDIF
         ! one center terms
         DO I=1,N_MIX_PAW
            DEL =(RHOLM(I,ISP)-RHOLM_LAST(I,ISP))
            RMST=RMST+DEL*DEL
         ENDDO
      ENDDO

      ! additional terms for relaxed core methods
      IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
         DO I=1,N_RHO_ONE_CENTRE
            DEL =(RHO_ONE_CENTRE(I)-RHO_ONE_CENTRE_LAST(I))
            RMST=RMST+DEL*DEL
         ENDDO
      ENDIF

      CALLMPI( M_sum_d(GRIDC%COMM, RMST, 1))
      CALLMPI( M_sum_d(GRIDC%COMM, RMS_tau, 1))
      RMST=SQRT(RMST)
      RMS_tau=SQRT(RMS_tau)
!-----------------------------------------------------------------------
! residual vector on reduced grid (i.e. difference CHTOT-CHTOTL)
! store in CWRK3, CHP is set to metric * CWRK3
!-----------------------------------------------------------------------
      SFAKT=SQRT(FAKT)
      DO ISP=1,ISPIN
         ! charge density
!DIR$ IVDEP
!OCL NOVREC
         DO I=1,GRIDB%RC%NP
            IP= I+(ISP-1)*GRIDB%RC%NP
            N1= MOD((I-1),GRIDB%RC%NROW) +1
            NC= (I-1)/GRIDB%RC%NROW+1
            N2= GRIDB%RC%I2(NC)
            N3= GRIDB%RC%I3(NC)
            SETFACT1
            SETSQRT
            CWRK3(IP)=MULFACT (CWRK3(IP)-CWRK4(IP))*SFAKT
            CWRK4(IP)=MULFACT  CWRK4(IP)*SFAKT
         ENDDO
         ! kinetic energy density
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            DO I=1,GRIDB%RC%NP
               IP= I+(ISP-1+ISPIN)*GRIDB%RC%NP
               N1= MOD((I-1),GRIDB%RC%NROW) +1
               NC= (I-1)/GRIDB%RC%NROW+1
               N2= GRIDB%RC%I2(NC)
               N3= GRIDB%RC%I3(NC)
               SETFACT1
               SETSQRT
               CWRK3(IP)=MULFACT (CWRK3(IP)-CWRK4(IP))*SFAKT
               CWRK4(IP)=MULFACT  CWRK4(IP)*SFAKT
            ENDDO
         ENDIF
      ENDDO

    ! Do not mix the G=0 component here ...
      CALL SET_RHO0(GRIDB, CWRK3(1), 0._q)

    ! set CHP = metric * (CHTOT-CHTOTL):
      ! charge and magnetization
      DO I=1,GRIDB%RC%NP*ISPIN
         CHP(I)=CWRK3(I)*CHP(I)
      ENDDO
      ! kinetic energy density
      IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
         DO I=GRIDB%RC%NP*ISPIN+1,2*GRIDB%RC%NP*ISPIN
            CHP(I)=CWRK3(I)*CHP(I)
         ENDDO
      ENDIF

!-----------------------------------------------------------------------
! Start mixing (given as input to BROYD on array CWRK1 if reset ...):
!-----------------------------------------------------------------------
      IF (MIX%LRESET) THEN
         ! set initial step or mixing for charge
         CALL SETG0(GRIDB,CWRK1,B,MIX%INIMIX,MIX%AMIX,MIX%BMIX,MIX%AMIN,(IDUMP/=0))
         NP=NCHARGE
         ! one center terms
         CWRK1 (NP+1:NP+N_MIX_PAW)= MIN(MIX%AMIX,0.5_q)
         NP=NP+N_MIX_PAW

         ! set initial step for magnetization
         INIMA=MIX%INIMIX
         DO ISP=2,ISPIN
            CALL SETG0(GRIDB,CWRK1(GRIDB%RC%NP*(ISP-1)+1),B,INIMA,MIX%AMIX_MAG,MIX%BMIX_MAG,MIX%AMIN,(IDUMP/=0))
            ! one center terms
            CWRK1(NP+1:NP+N_MIX_PAW)  = MIN(MIX%AMIX_MAG,2.0_q)
            NP=NP+N_MIX_PAW
         ENDDO

         ! set initial step for kinetic energy density
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            DO ISP=1,ISPIN
               CALL SETG0(GRIDB,CWRK1(GRIDB%RC%NP*(ISP-1+ISPIN)+1),B,INIMA,MIX%AMIX_MAG,MIX%BMIX_MAG,MIX%AMIN,(IDUMP/=0))
            ENDDO
         ENDIF

         ! initial mixing for relaxed core method
         IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
            CWRK1(NP+1:NP+N_RHO_ONE_CENTRE)= MIX%AMIX
            NP=NP+N_RHO_ONE_CENTRE
         ENDIF

         IF (NP /= NDATA) THEN
            WRITE(*,*) 'internal error 2 in BRMIX',NP,NDATA ; STOP
         ENDIF
      ENDIF
! Broyden mixing:
      IB=0
      IF (MIX%WC==0._q) IB=1
      IF (IERR/=0) MIX%LRESET=.TRUE.
      IERR=0

      CALL BROYD(NDATA,CWRK3,CWRK4,CHP,MIX%WC,IB,NUPDZ, &
                 IABS(MIX%MAXMIX),MIX%MAXMIX>0 .AND. .NOT. MIX%HARD_RESET,MIX%MREMOVE,B,NGIGA, &
                 CWRK1,CWRK2,MIX%LRESET,MIX%IUBROY,IO%LOPEN,IO%ICMPLX,IO%MRECL, &
                 RMS,RMSP,WEIGHT,MIX%NEIG,MIX%EIGENVAL,MIX%AMEAN,IERR &
#ifdef MPI
                 ,GRIDC%COMM &
#endif
                )

      IF (MIX%LRESET) MIX%HARD_RESET=.FALSE.

      DWRITE0 'rms are',RMS,RMSP,RMST

! Hmmm ... . Disk was full? Invalid/corrupted file TMPBROYD? Or???
      IF (IERR/=0) THEN
! 'last chance': get some disk space and reset mixing at next step:
#ifdef USE_TMPBROYD
         CALL CLEAN(MIX%IUBROY)
#endif
! we can not only do 'simple mixing' here (take initial mixing ...):
! this has been done 'on entry' --- just forget the following update!
         GOTO 50
      ENDIF
!-----------------------------------------------------------------------
! now CWRK4 (representing the image of CHTOTL) containes the mixed
! density - expand to the larger grid and update CHTOT ... :
!-----------------------------------------------------------------------
      ISFAKT=1._q/SFAKT
      SUM_=0
      NP=NCHARGE

      DO ISP=1,ISPIN
         ! charge density
!DIR$ IVDEP
!OCL NOVREC
         DO I=1,GRIDB%RC%NP
            IP= I+(ISP-1)*GRIDB%RC%NP
            N1= MOD((I-1),GRIDB%RC%NROW) +1
            NC= (I-1)/GRIDB%RC%NROW+1
            N2= GRIDB%RC%I2(NC)
            N3= GRIDB%RC%I3(NC)

            SETFACT1
            SETISQRT
            CWRK4(IP)= MULFACT CWRK4(IP)*ISFAKT
            SUM_=SUM_+CWRK4(IP)*CONJG(CWRK4(IP))
         ENDDO
         CALL CPB_GRID(GRIDC,GRIDB,B_TO_C,CWRK4(GRIDB%RC%NP*(ISP-1)+1),CHTOT(1,ISP))

         ! kinetic energy density
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            DO I=1,GRIDB%RC%NP
               IP= I+(ISP-1+ISPIN)*GRIDB%RC%NP
               N1= MOD((I-1),GRIDB%RC%NROW) +1
               NC= (I-1)/GRIDB%RC%NROW+1
               N2= GRIDB%RC%I2(NC)
               N3= GRIDB%RC%I3(NC)

               SETFACT1
               SETISQRT
               CWRK4(IP)= MULFACT CWRK4(IP)*ISFAKT
            ENDDO
            CALL CPB_GRID(GRIDC,GRIDB,B_TO_C,CWRK4(GRIDB%RC%NP*(ISP-1+ISPIN)+1),KINEDEN%TAU(1,ISP))
         ENDIF

         ! one center terms
         RHOLM(:,ISP) = CWRK4(NP+1:NP+N_MIX_PAW)
         NP=NP+N_MIX_PAW
      ENDDO

      IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
         RHO_ONE_CENTRE=CWRK4(NP+1:NP+N_RHO_ONE_CENTRE)
         NP=NP+N_RHO_ONE_CENTRE
      ENDIF

      IF (NP /= NDATA) THEN
         WRITE(*,*) 'internal error 3 in BRMIX',NP,NDATA ; STOP
      ENDIF

      CALLMPI( M_sum_d(GRIDC%COMM, SUM_, 1))
      DWRITE0 'summed soft charge is',SUM_

! Check whether charge density changed
     io_begin
      IF (ABS(TOTNEL-TOTOLD)>1E-5_q*ABS(TOTNEL) .AND. LWARN) WRITE(*,11) TOTOLD,TOTNEL
   11 FORMAT('BRMIX: very serious problems',/ &
             ' the old and the new charge density differ',/ &
             ' old charge density: ',F11.5,' new',F11.5)
     io_end
      CALL SET_RHO0(GRIDC, CHTOT(1,1), TOTNEL)
! copy CHTOT to CHTOTL
   50 IERRBR=IERR
! We average CHTOT over the KPAR (possibly slighty) different instances ... first the summation
      CALLMPI( M_bcast_z(GRIDC%COMM_KINTER,CHTOT(1,1),SIZE(CHTOT)))
      SUM_=0
      DO ISP=1,ISPIN
         DO I=1,GRIDC%RC%NP
            CHTOTL(I,ISP)=CHTOT(I,ISP)
            SUM_=SUM_+CHTOT(I,ISP)*CONJG(CHTOT(I,ISP))
         ENDDO
      ENDDO
! copy kinetic energy density
      IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
! TAU needs to be averaged over KPAR instances as well
         CALLMPI( M_bcast_z(GRIDC%COMM_KINTER,KINEDEN%TAU(1,1),SIZE(KINEDEN%TAU)))
         DO ISP=1,ISPIN
            DO I=1,GRIDC%RC%NP
               KINEDEN%TAUL(I,ISP)=KINEDEN%TAU(I,ISP)
            ENDDO
         ENDDO
      ENDIF
! copy occupancies
#ifdef MPI
      CALL M_bcast_d(GRIDC%COMM_KINTER,RHOLM(1,1),SIZE(RHOLM))
#endif 
      RHOLM_LAST=RHOLM
! copy one-center densities
      IF (PRESENT(N_RHO_ONE_CENTRE)) THEN
         RHO_ONE_CENTRE_LAST=RHO_ONE_CENTRE
      ENDIF

      CALLMPI( M_sum_d(GRIDC%COMM, SUM_, 1))
      DWRITE0 'summed charge is',SUM_

      DEALLOCATE(CWRK1,CWRK2,CWRK3,CWRK4,CHP)

      PROFILING_STOP('brmix')

      RETURN
      END SUBROUTINE


!***********************************************************************
!
! Find the zero of a function vector F(X) of a vector X using the
! second form of the modified Broyden scheme of Vanderbilt and Louie
! as "described" in D. Johnsons paper (PRB 38, 12807 [Dec. 1988]).
! (don't take this comment too serious, Johnson's paper contains at
!  least 6 hard errors, which we have corrected ...)
!
! details can be found in
! G. Kresse, J. Furthmueller,  Comput. Mat. Sci. 6, 15-50 (1996)
! (written by  jF with contributions of gK [small corrections only])
!
! there is a routine in VASP which is much easier to understand and
! does more or less the same see dynbr.F,
! this routine is so complicated
! because it can use a file to save the iteration history
!
!***********************************************************************

      SUBROUTINE BROYD(NDIM,F,X,FP,WC,IB,NUPDZ,MAXIT,LKEEP,REMOVE, &
     &                  B,NGIGA,WRK1,WRK2,INI,IU,LOPEN,ICMPLX,MRECL, &
     &                  RMS,RMSP,WEIGHT,NEIG,EIGENVAL,AMEAN,IERR &
#ifdef MPI
                        ,COMM &
#endif
                       )
      USE prec
      USE mpimy
      IMPLICIT REAL(q) (A-H,O-Z)

#ifdef MPI
      TYPE (communic) COMM
#endif

      PARAMETER(WEIMAX=33._q)
      LOGICAL INI,LOPEN,LREDUC
      LOGICAL LKEEP     ! keep Hessian matrix even at reset
      COMPLEX(q) CTMP
      COMPLEX(q) F(NDIM),X(NDIM),FP(NDIM),WRK1(NGIGA),WRK2(NDIM)
      REAL(q)    B(3,3)

      REAL(q)    BETAQ(MAXIT,MAXIT),AMAT(MAXIT,MAXIT),BETA(MAXIT,MAXIT)
      REAL(q)    AUX(MAXIT,MAX(8,MAXIT)),VV(MAXIT)
      REAL(q)    GP(MAXIT,MAXIT),SP(MAXIT,MAXIT)

      INTEGER    IOD(3),INDEX(MAXIT)
      REAL(q)    AUXR(MAXIT),AUXI(MAXIT),AUXBET(MAXIT)
      COMPLEX(q) AUXC(MAXIT)
      REAL(q)    EIGENVAL(512)

      REAL(q), ALLOCATABLE,SAVE :: WI(:),FINF(:,:), &
                      GMAT(:,:),SMAT(:,:)
      SAVE ITER,ICALL,IOD,LASTIT
      DATA ITER /0/,ICALL /0/,LASTIT /0/

      INTEGER REMOVE   ! how many vectors are removed once MAXMIX is reached

      PROFILING_START('broyd')

#if defined(USE_TMPBROYD) && defined(MPI)
      MPI + use of file for MIXING is not supported please undef USE_TMPBROYD
#endif
      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME= COMM%NODE_ME
      IONODE = COMM%IONODE
#endif
      NEIG=0
!=======================================================================
! First call: initialise some I/O-things ...
!=======================================================================
      IF (ICALL==0) THEN
         ALLOCATE(WI(MAXIT),FINF(MAXIT,MAXIT), &
                  GMAT(MAXIT,MAXIT),SMAT(MAXIT,MAXIT))
         WI=0
         FINF=0
         GMAT=0
         SMAT=0

         IRECL=NDIM*ICMPLX
! Save some I/O-data ... :
         IOD(1)=IU
         IOD(2)=-1
         IOD(3)=1
         ICALL=1

#ifdef USE_TMPBROYD
! Open the save file for iteration data ... :
         IF (LOPEN) THEN
            OPEN(UNIT=IU,FILE='TMPBROYD',ACCESS='DIRECT', &
     &                                    FORM='UNFORMATTED',RECL=IRECL)
         ELSE
            OPEN(UNIT=IU,ACCESS='DIRECT',FORM='UNFORMATTED',RECL=IRECL)
         END IF
#else
         CALL BRNULL()
#endif
      END IF
!=======================================================================
! calculate some necessary quantities
!=======================================================================

! Restart of mixing forced externally by setting INI = .TRUE. (this must
! be used for example after an ionic step / setup of a new geometry):
      IF (INI .AND. ITER/=0 ) THEN
      IF (.NOT. LKEEP .OR. MAXIT-ITER < -1 ) THEN
          ITER=0
          LASTIT=0
      ELSE
! keep mixing information alive (i.e. use it as an initial approximation
! for  the present charge dielectric function)
         ITER=ITER-1
         LASTIT=ITER
      ENDIF
      ENDIF
! usually it makes no sense to proceed if one cannot converge after so
! many steps - if there is some little hope at all, then by removing
! vectors from the iteration history
! currently REMOVE vectors are removed if MAXIT is reached
      IF (ITER == MAXIT ) THEN
         ISHIFT=MIN(REMOVE,ITER)
!        WRITE(*,*) 'removing ',ISHIFT,' from iteration history'
         ITER  =ITER-ISHIFT
         LASTIT=MAX(0,LASTIT-ISHIFT)
         ! shift all saved arrays and vectors
         CALL BR_SHIFT_MATRIX(GMAT,MAXIT,ITER,ISHIFT)
         CALL BR_SHIFT_MATRIX(SMAT,MAXIT,ITER,ISHIFT)
         CALL BR_SHIFT_MATRIX(FINF,MAXIT,ITER,ISHIFT)
         CALL BR_SHIFT_VEC   (WI,1,ITER,ISHIFT)
         ! shift all stored vectors
         DO I=1,ITER-1
            CALL BRGET(NDIM,WRK1,'F',I+ISHIFT,IOD,ITER,IERR)
            CALL BRGET(NDIM,WRK2,'U',I+ISHIFT,IOD,ITER,IERR)
            CALL BRSAV(NDIM,WRK1,'F',I,IOD,ITER,IERR)
            CALL BRSAV(NDIM,WRK2,'U',I,IOD,ITER,IERR)
            CALL BRGET(NDIM,WRK1,'Z',I+ISHIFT,IOD,ITER,IERR)
            CALL BRSAV(NDIM,WRK2,'Z',I,IOD,ITER,IERR)
         ENDDO
      ENDIF
! Restart of mixing if MAXIT is the same as ITER
! not effective with the few lines above
      ITER=MOD(ITER,MAXIT)
! Increment iteration counter:
      ITER=ITER+1
! Number of previous iteration:
      ITERM1=ITER-1
! Change of vector |X>
      SUM_=0
      SUMP=0
      DO I=1,NDIM
         SUMP=SUMP+CONJG(F(I))*FP(I)
         SUM_=SUM_  +CONJG(F(I))*F(I)
      ENDDO
      CALLMPI( M_sum_d(COMM, SUM_, 1))
      CALLMPI( M_sum_d(COMM, SUMP, 1))
      RMSP=SQRT(SUMP)
      RMS =SQRT(SUM_)
      DWRITE0 'rms',RMS,RMSP
! Relative weight (W_iter/W_0) used for the current iteration ... :
      IF (WC>=0._q) THEN
        WI(ITER)=WC
      ELSE
! Try to set weights automatically (not recommended)
        WI(ITER)=0.01_q*ABS(WC)/SUMP
      END IF
! Do not allow too strange weights ... :
      IF (WI(ITER) < 1)     WI(ITER)=1
      IF (WI(ITER) > 1E6  ) WI(ITER)=1E6
      IF (WC==0 )           WI(ITER)=100

! Here Broydens 2nd method (slightly generalized) can be switched on:
      IF (IB/=0) THEN
         DO I=1,ITERM1
           WI(I)=0
         ENDDO
         DO I=MAX(1,ITERM1-IB+1),ITERM1
           WI(I)=100
         ENDDO
      ENDIF
! it is possible to give all iterations including LASTIT
! zero weight
! this means that information collected up to LASTIT is overwritten
! when updating the Hessian matrix
!      WI(1:LASTIT)=0

      WEIGHT=WI(ITER)
!=======================================================================
! First iteration is a conventional linear mixing using G^(1):
!=======================================================================
      IF (ITER==1 .OR.INI) THEN
         IF (.NOT.INI) THEN
! Can only be reset by ITER>MAXIT here, try again initial start mixing:
            IF (NODE_ME /= IONODE) &
            WRITE(*,*) 'Iteration count exceeded MAXIT =',MAXIT
! Get initial (diagonal) mixing used previously ... :
            CALL BRGET(NDIM,WRK1,'G',ITER,IOD,ITER,IERR)
            IF (IERR/=0) THEN
               PROFILING_STOP('broyd')
               RETURN
            END IF
         END IF
         IF (ITER==1) THEN
#ifdef USE_TMPBROYD
      ! Delete and reopen file TMPBROYD (or clean temproray work area)...
           CALL CLEAN(IU)
#else
           CALL CLBROYD(IU)
#endif
       ! allocate for next 4 steps
           CALL BRSAV(NDIM,F,'Z',ITER+3,IOD,ITER+3,IERR)
         ENDIF
! First save some things needed later: |XLAST>, |F^(1)> and |FP^(1)>:
         CALL BRSAV(NDIM,FP,'P',ITER,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF
         CALL BRSAV(NDIM,X,'X',ITER,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF
         CALL BRSAV(NDIM,F,'I',ITER,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF
! Now save also the start mixing G^(1) supplied on array WRK1:
         CALL BRSAV(NDIM,WRK1,'G',ITER,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF

         GOTO 2000
! Mix (here we need no metric because for simple linear mixing
! using a diagonal G0 the result is the same with and without metric.):
         DO I=1,NDIM
           X(I)=X(I)+WRK1(I)*F(I)
         ENDDO
! That is all for the first ... :
         PROFILING_STOP('broyd')
         RETURN
      END IF

      IF (MOD(ITER,5)==1) THEN
! force allocation in junks of 5 data blocks to avoid memory fragmentation
         CALL BRSAV(NDIM,F,'Z',ITER+3,IOD,ITER+3,IERR)
      ENDIF
!=======================================================================
! BROYDEN updating for ITER > 1 ... :
!=======================================================================
!Get |F> and |FP> from last iteration
      CALL BRGET(NDIM,WRK1,'P',ITERM1,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
      CALL BRGET(NDIM,WRK2,'I',ITERM1,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Save actual |F> and |FP>:
      CALL BRSAV(NDIM,F,'I',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
      CALL BRSAV(NDIM,FP,'P',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
      DO I=1,NDIM
       ! |FP^(iter)>-|FP^(iter-1)>:
         FP(I)=FP(I)-WRK1(I)
       ! |F^(iter)>-|F^(iter-1)>:
        F(I)=F(I)-WRK2(I)
      ENDDO
! Norm of this vector (with integration weights/metric. included ...):
      FNORM=0._q
      DO  I=1,NDIM
        FNORM=FNORM+FP(I)*CONJG(F(I))
      ENDDO
      CALLMPI( M_sum_d( COMM, FNORM, 1))
      FNORM=1._q/SQRT(FNORM)
! Build |Delta F^(iter-1)>, |Delta FP^(iter-1)>:
      DO I=1,NDIM
         FP(I)=FP(I)*FNORM
         F(I)=F(I)*FNORM
      ENDDO
! Save |Delta F^(iter-1)>:
      CALL BRSAV(NDIM,F,'F',ITERM1,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Get |X^(iter-1)> from last iteration:
      CALL BRGET(NDIM,WRK2,'X',ITERM1,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Save actual |X^(iter)>:
      CALL BRSAV(NDIM,X,'X',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Build |Delta X^(iter-1)> = (|X^(iter)>-|X^(iter-1)>)/||F||_w :
      DO I=1,NDIM
        WRK1(I)=(X(I)-WRK2(I))*FNORM
      ENDDO
! Vector |U^(iter-1)> = G^(1) |Delta F^(iter-1)> + |Delta X^(iter-1)>:
      CALL BRGET(NDIM,WRK2,'G',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
      DO I=1,NDIM
        CTMP=WRK1(I)
        WRK1(I)=WRK2(I)*F(I)
        WRK1(I+NDIM)=CTMP+WRK2(I)*F(I)
        WRK2(I)=CTMP+WRK2(I)*F(I)
      ENDDO
! Save |U^(iter-1)>:
      CALL BRSAV(NDIM,WRK2,'U',ITERM1,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
!=======================================================================
! Matrix FINF: we must only add the elements (iter-1,j) and (j,iter-1):
!   FINF(j,iter-1) = <D F(j) | metric |  D F(iter-1) >
!=======================================================================
      DO J=1,ITERM1
         CALL BRGET(NDIM,WRK2,'F',J,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF
         SUM_=0
         DO I=1,NDIM
           SUM_=SUM_+CONJG(WRK2(I))*FP(I)
         ENDDO

         FINF(J,ITERM1)=SUM_
         FINF(ITERM1,J)=SUM_

         SUM1=0
         SUM2=0
        !  SUM1=<D F(j)| G(1) |D F(iter-1)>
        !  SUM2=<D F(j)| U^(iter-1)> = <D F(j)| G(m)-G(1) | D F^(iter-1)>
         DO I=1,NDIM
           SUM1=SUM1+CONJG(WRK2(I))*WRK1(I)
           SUM2=SUM2-CONJG(WRK2(I))*WRK1(I+NDIM)
         ENDDO
         SMAT(J,ITERM1)=SUM1
         SMAT(ITERM1,J)=SUM1
         GMAT(J,ITERM1)=SUM2
      ENDDO

      CALL BRGET(NDIM,WRK1,'F',ITERM1,IOD,ITER,IERR)

      DO J=1,ITERM1
         CALL BRGET(NDIM,WRK2,'U',J,IOD,ITER,IERR)
         IF (IERR/=0) THEN
            PROFILING_STOP('broyd')
            RETURN
         END IF
       ! SUM_=<D F(iter-1)| U^(j)>
         SUM_=0
         DO I=1,NDIM
           SUM_=SUM_+CONJG(WRK1(I))*WRK2(I)
         ENDDO
         GMAT(ITERM1,J)=-SUM_
      ENDDO

      CALLMPI( M_sum_d(COMM, SMAT(1,ITERM1), ITERM1))
      CALLMPI( M_sum_d(COMM, FINF(1,ITERM1), ITERM1))
      CALLMPI( M_sum_d(COMM, GMAT(1,ITERM1), ITERM1))
      CALLMPI( M_sum_d(COMM, GMAT(ITERM1,1:ITERM1), ITERM1-1))
      FINF(ITERM1,ITERM1)=1._q

      ! correct transposed elements
      DO J=1,ITERM1-1
        FINF(ITERM1,J)=FINF(J,ITERM1)
        SMAT(ITERM1,J)=SMAT(J,ITERM1)
      ENDDO

! Matrix A:
      DO J=1,ITERM1
      AMAT(J,J)=1._q+WI(J)*WI(J)*FINF(J,J)
      DO K=1,J-1
         AMAT(J,K)=FINF(J,K)*WI(J)*WI(K)
         AMAT(K,J)=FINF(K,J)*WI(K)*WI(J)
      ENDDO
      ENDDO
      IDUMP=0
      io_begin
      IF (IDUMP==1) THEN
        WRITE(*,*)
        WRITE(*,*)'AMAT'
        DO I=1,ITERM1
          WRITE(*,'(10F10.4)') (AMAT (I,J),J=1,ITERM1)
        ENDDO
        WRITE(*,*)'FINF'
        DO I=1,ITERM1
          WRITE(*,'(10F10.4)') (FINF(I,J),J=1,ITERM1)
        ENDDO
      ENDIF
      io_end


! Matrix Beta:
      IF (ITERM1==1) THEN
! Trivial case: A is a 1x1 matrix ...
         BETA(1,1)=1._q/AMAT(1,1)
      ELSE
! General case: Invert matrix A ...
         CALL INVERS(AMAT,ITERM1,MAXIT,BETA,AUX,INDEX,VV)
      END IF
!=======================================================================
!  solve eigenvalue problem
!   G e = lambda G^1 e
!  where G is the approximation of the Hessian matrix
!  works only for Pulay mixing because we assume  G |d F> = | d X>
!=======================================================================
     io_begin
      DO I=1,ITERM1
      DO J=1,ITERM1
        GP(I,J)=GMAT(I,J)
        SP(I,J)=SMAT(I,J)
      ENDDO
      ENDDO

      IDUMP=0
      IF (IDUMP==1) THEN
        WRITE(*,*)'SMAT'
        DO  I=1,ITERM1
          WRITE(*,'(10F10.4)') (SP(I,J),J=1,ITERM1)
        ENDDO
        WRITE(*,*)'GMAT'
        DO I=1,ITERM1
          WRITE(*,'(10F10.4)') (GP(I,J),J=1,ITERM1)
        ENDDO
      ENDIF
      INFO=1
      IF (INFO/=0) THEN
#ifdef essl
        CALL DGEGV(0,GP,MAXIT,SP,MAXIT,AUXC,AUXBET,AUX,1,ITERM1, &
       &        AUX,MAX(8*MAXIT,MAXIT*MAXIT))
#else
        CALL DGEGV('N','N',ITERM1,GP,MAXIT,SP,MAXIT,AUXR,AUXI,AUXBET, &
       &        AUX,1,AUX,1,AUX,MAX(8*MAXIT,MAXIT*MAXIT),INFO)
        AUXC= CMPLX( AUXR , AUXI ,KIND=q)
#endif
        NEIG=ITERM1
        DO I=1,ITERM1
          EIGENVAL(I)= ABS(AUXC(I)/MAX(AUXBET(I),1E-10_q)+1)
        ENDDO
        AMEAN=0
        DO I=1,ITERM1
          AMEAN=AMEAN+ABS(AUXC(I)/MAX(AUXBET(I),1E-10_q)+1)
        END DO
        AMEAN=AMEAN/ITERM1
      ENDIF
     io_end
!=======================================================================
! Calculate BETAQ used in the update of Z.(formula in Johnsons paper is
! wrong if weights are not equal, here is the correct version - by gK):
! a few comments here:
!   for Pulays      approach BETAQ is strictly 0 (equ. 103)
!   for Broydens 2. approach BETAQ and BETA have the structure
!            BETAQ     1   0   0      BETA=0 except for
!                      0   1   0      BETA(ITERM1,ITERM1)=1
!                      0   0   1
!                     g_1 g_2  g_3
!            (see equ. 104 and 105 in gK)
!=======================================================================
      LREDUC=.TRUE.
      BETAQ=0
      DO I=1,ITERM1
        LREDUC=LREDUC.AND.(WI(I)>=WEIMAX)
        DO IT=1,ITERM1-1
          BETAQ(I,IT)=0._q
          DO J=1,ITERM1
            BETAQ(I,IT)=BETAQ(I,IT)-WI(I)*BETA(I,J)*WI(J)*FINF(J,IT)
          ENDDO
        ENDDO
        BETAQ(I,I)=BETAQ(I,I)+1
      ENDDO

      IDUMP=0
      io_begin
      IF (IDUMP==1) THEN
        WRITE(*,*)'BETA'
        DO I=1,ITERM1
          WRITE(*,'(10F10.4)') (BETA (I,J)*WI(I)*WI(J),J=1,ITERM1)
        ENDDO
        WRITE(*,*)'BETAQ'
        DO I=1,ITERM1
          WRITE(*,'(10F10.4)') (BETAQ(I,J),J=1,ITERM1-1)
        ENDDO
      ENDIF
      io_end
      IDUMP=0

! Reload |FP^(iter)>:
      CALL BRGET(NDIM,FP,'P',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
!=======================================================================
! update all the vectors |Z_it^(m-1)> - if WRK1 is dimensioned large
! enough several vectors Z are updated simultaneously to reduce I/O!
!=======================================================================
      DO IT=1,ITERM1,NUPDZ
! Number of vectors to be updated simultaneously:
         NUPDM=MIN(NUPDZ,ITERM1-IT+1)
         DO I=1,NDIM*NUPDM
           WRK1(I)=(0._q,0._q)
         ENDDO
! Sum_{im=1,iter-2} BETAQ_{it,im} |Z_im^(iter-2)> - warning, do not be
! confused: array F is here used as a workarray ... !
         IF (.NOT.LREDUC) THEN
! Remark: this part is not used at all if all weigths are very large!
            DO IM=1,ITERM1-1
! Selective treatment for steps with "smaller" weights ... :
               IF (WI(IM)>=(10._q*WEIMAX)) CYCLE
               CALL BRGET(NDIM,F,'Z',IM,IOD,ITER,IERR)
               IF (IERR/=0) THEN
                  PROFILING_STOP('broyd')
                  RETURN
               END IF
               DO IUPD=0,NUPDM-1
                  IF (BETAQ(IT+IUPD,IM)==0._q) CYCLE
                  IOFF=IUPD*NDIM
                  DO I=1,NDIM
                  WRK1(I+IOFF)=WRK1(I+IOFF)+BETAQ(IT+IUPD,IM)*F(I)
                  ENDDO
               ENDDO
            ENDDO
         ENDIF
! Add sum_{im=1,iter-1} WI_im WI_it BETA_{it,im} |U^(im)>:
         DO IM=1,ITERM1
            CALL BRGET(NDIM,F,'U',IM,IOD,ITER,IERR)
            IF (IERR/=0) THEN
               PROFILING_STOP('broyd')
               RETURN
            END IF
            DO IUPD=0,NUPDM-1
               IOFF=IUPD*NDIM
               DO  I=1,NDIM
               WRK1(I+IOFF)=WRK1(I+IOFF)+ &
     &                          WI(IT+IUPD)*WI(IM)*BETA(IT+IUPD,IM)*F(I)
               ENDDO
            ENDDO
         ENDDO
         IF (LREDUC) THEN
! Save |Z>-vectors (only allowed in reduced version for large weigths):
            DO IUPD=0,NUPDM-1
               IOFF=IUPD*NDIM
               CALL BRSAV(NDIM,WRK1(1+IOFF),'Z',IT+IUPD,IOD,ITER,IERR)
               IF (IERR/=0) THEN
                  PROFILING_STOP('broyd')
                  RETURN
               END IF
            ENDDO
         ELSE
! Save temporarily (we may not yet destroy the old |Z>-vectors!):
            DO IUPD=0,NUPDM-1
               IOFF=IUPD*NDIM
               IF (WI(IT+IUPD)<(10._q*WEIMAX)) THEN
                 CALL BRSAV(NDIM,WRK1(1+IOFF),'T',IT+IUPD,IOD,ITER,IERR)
                 IF (IERR/=0) THEN
                    PROFILING_STOP('broyd')
                    RETURN
                 END IF
               ELSE
! ... except unused |Z>-vectors (iterations with large weights):
                 CALL BRSAV(NDIM,WRK1(1+IOFF),'Z',IT+IUPD,IOD,ITER,IERR)
                 IF (IERR/=0) THEN
                    PROFILING_STOP('broyd')
                    RETURN
                 END IF
               ENDIF
            ENDDO
         ENDIF
      ENDDO

! Finally swap all updated |Z_it^(iter-1)> from temporary records 'T'
! to records 'Z' (if necessary) to be prepared for the next step ... :
      DO IT=1,ITERM1
         IF ((.NOT.LREDUC).AND.(WI(IT)<(10._q*WEIMAX))) THEN
            CALL BRGET(NDIM,WRK1,'T',IT,IOD,ITER,IERR)
            IF (IERR/=0) THEN
               PROFILING_STOP('broyd')
               RETURN
            END IF
            CALL BRSAV(NDIM,WRK1,'Z',IT,IOD,ITER,IERR)
            IF (IERR/=0) THEN
               PROFILING_STOP('broyd')
               RETURN
            END IF
         ENDIF
      ENDDO
!=======================================================================
! calculate (G-G^(1)) |F^(iter)>
!=======================================================================
 2000 CONTINUE

! Reload |FP^(iter)>:
      CALL BRGET(NDIM,FP,'P',ITER,IOD,ITER,IERR)

      DO IT=1,ITERM1
! Get the |Delta F^(it)>:
            CALL BRGET(NDIM,F,'F',IT,IOD,ITER,IERR)
            IF (IERR/=0) THEN
               PROFILING_STOP('broyd')
               RETURN
            END IF
            SUM_=0._q
! Build <Delta F^(it)|FP^(iter)>:
            DO I=1,NDIM
              SUM_=SUM_+CONJG(F(I))*FP(I)
            ENDDO
            CALLMPI ( M_sum_d(COMM, SUM_, 1))
! Subtract |Z_it^(iter-1)><Delta F^(it)|FP^(iter)> from |X^(iter)>:
            CALL BRGET(NDIM,WRK1(1),'Z',IT,IOD,ITER,IERR)
            DO I=1,NDIM
              X(I)=X(I)-SUM_*WRK1(I)
            ENDDO
      ENDDO
!=======================================================================
! calculate G^(1) |F^(iter)>
!=======================================================================
! Reload |F^(iter)>:
      CALL BRGET(NDIM,F,'I',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Reload start mixing G^(1):
      CALL BRGET(NDIM,WRK1,'G',ITER,IOD,ITER,IERR)
      IF (IERR/=0) THEN
         PROFILING_STOP('broyd')
         RETURN
      END IF
! Add G^(1) |F^(iter)> to vector |X^(iter)>:
      DO I=1,NDIM
        X(I)=X(I)+WRK1(I)*F(I)
      ENDDO

! Now we have the output vector |X(iter+1)> and all is done - good bye:
#ifndef USE_TMPBROYD
      CALL BRRMTMP()
#endif

      PROFILING_STOP('broyd')

      RETURN
      END SUBROUTINE

!************************* SUBROUTINE SETG0 ****************************
!
! set the initial inverse Jacobi matrix G^(1)
! use a Kerker like matrix or simple mixing
!
!***********************************************************************

      SUBROUTINE SETG0(GRIDB,G0,B,IMIX,AMIX,BMIX,AMIN,LW)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRIDB

! Sets up initial mixing ...
      LOGICAL LW
      COMPLEX(q) G0(GRIDB%RC%NP)
      REAL(q)    B(3,3)

      PROFILING_START('setg0')

      TPI=8._q*ATAN(1._q)
      IF (IMIX==2) THEN

! 'No mixing'

         IF (LW) WRITE(*,*) 'Initial mixing matrix = unity matrix (no mixing!)'
         G0=1._q
      ELSE IF (IMIX==1) THEN

! Kerker mixing:

         IF (LW) WRITE(*,*) &
           'Initial mixing = Kerker mixing. AMIX=',AMIX,'    BMIX=',BMIX
         FLAM=BMIX**2
         G0MIN=1E10_q
         IND0=0
         DO K=1,GRIDB%RC%NP
            N1= MOD((K-1),GRIDB%RC%NROW) +1
            NC= (K-1)/GRIDB%RC%NROW+1
            N2= GRIDB%RC%I2(NC)
            N3= GRIDB%RC%I3(NC)

            IGX=GRIDB%LPCTX(N1)
            IGY=GRIDB%LPCTY(N2)
            IGZ=GRIDB%LPCTZ(N3)
            GX=IGX*B(1,1)+IGY*B(1,2)+IGZ*B(1,3)
            GY=IGX*B(2,1)+IGY*B(2,2)+IGZ*B(2,3)
            GZ=IGX*B(3,1)+IGY*B(3,2)+IGZ*B(3,3)
            GSQU=(GX*GX+GY*GY+GZ*GZ)*TPI*TPI
            G0(K)=MAX(AMIX*GSQU/(GSQU+FLAM),AMIN)
            IF (N1==1 .AND. N2==1 .AND. N3==1) THEN
              IND0=K
            ELSE
              G0MIN=MIN(G0MIN,AMIX*GSQU/(GSQU+FLAM))
            ENDIF
         ENDDO
         G0MIN=-G0MIN; CALLMPI ( M_max_d( GRIDB%COMM, G0MIN,1))
         IF (IND0/=0) G0(IND0)=-G0MIN
      ELSE

! Linear mixing using only AMIX:

         IF (LW) WRITE(*,*) 'Initial mixing = linear mixing. ALP =',AMIX
         G0=AMIX
      END IF

      PROFILING_STOP('setg0')

      RETURN
      END SUBROUTINE

!************************* SUBROUTINE BRPRE ****************************
!
! set up metric, it is assumed that small components are more
! important than large G-componentes, therefore
! 1   inverse of Kerker like matrix with automatic determination
!     such that long wave lenght components are weighted 10 times stronger
! 2   inverse of Kerker like matrix with automatic determination
!     such that long wave lenght components are weighted 100 times stronger
! 3   inverse Kerker metric determined by BMIX
! 4-5 weight the long wave vectors more strongly (not very reasonable)
! else no metric is used
!
!***********************************************************************


      SUBROUTINE BRPRE(GRIDB,P,B,MIXPRE,BMIX,LWRITE)
      USE prec

      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d)  GRIDB

      PARAMETER(FACMAX=20._q)

      LOGICAL LWRITE
      COMPLEX(q) P(GRIDB%RC%NP)
      REAL(q)    B(3,3)

      PROFILING_START('brpre')
!
      TPI=8._q*ATAN(1._q)
      IF ((MIXPRE<1).OR.(MIXPRE>5)) THEN
! Unknown type: 'No metric':
         IF (LWRITE) WRITE(*,*) 'Routine BRPRE: No metric'
         P=1.0_q
      ELSE
         IF (MIXPRE==1 .OR. MIXPRE==2 .OR. MIXPRE==4 .OR. MIXPRE==5) THEN
! (Inverse) Hartree like metric (not really exactly - it is
! essentially an inverse Kerker-type metric with automatically
! adjusted parameter BMIX so that the maximum differences between the
! maximum and minimum weights are about a given predefined value ...):
            FACMAX_=FACMAX
            IF (MIXPRE==2.OR. MIXPRE==5) FACMAX_=FACMAX*10

            IF (LWRITE) WRITE(*,*) 'Routine BRPRE: ''Kerker-metric'''
! Get smallest Q-vector (Q_min^2):
            GSQUM=1.E30_q
            DO K=1,GRIDB%RC%NP
               N1= MOD((K-1),GRIDB%RC%NROW) +1
               NC= (K-1)/GRIDB%RC%NROW+1
               N2= GRIDB%RC%I2(NC)
               N3= GRIDB%RC%I3(NC)

               IGX=GRIDB%LPCTX(N1)
               IGY=GRIDB%LPCTY(N2)
               IGZ=GRIDB%LPCTZ(N3)
               GX=IGX*B(1,1)+IGY*B(1,2)+IGZ*B(1,3)
               GY=IGX*B(2,1)+IGY*B(2,2)+IGZ*B(2,3)
               GZ=IGX*B(3,1)+IGY*B(3,2)+IGZ*B(3,3)
               GSQU=(GX*GX+GY*GY+GZ*GZ)*TPI*TPI
               IF (GSQU>1.E-10_q) GSQUM=MIN(GSQUM,GSQU)
            ENDDO
! Set 'automatic BMIX':
            GSQUM=-GSQUM
            CALLMPI( M_max_d(GRIDB%COMM, GSQUM, 1))
            GSQUM=-GSQUM
            FLAM=(FACMAX_-1._q)*GSQUM
            IF (LWRITE) WRITE(*,*) '               Automatic BMIX_pre =',SQRT(FLAM)
         ELSE
! (Inverse) Kerker-mixing like metric:
            IF (LWRITE) WRITE(*,*) 'Routine BRPRE: Kerker-metric with BMIX =',BMIX
            FLAM=BMIX**2
         END IF
! Now set up the metric:
         IF (MIXPRE==1 .OR. MIXPRE==2 .OR. MIXPRE==3) THEN
         DO K=1,GRIDB%RC%NP
            N1= MOD((K-1),GRIDB%RC%NROW) +1
            NC= (K-1)/GRIDB%RC%NROW+1
            N2= GRIDB%RC%I2(NC)
            N3= GRIDB%RC%I3(NC)

            IGX=GRIDB%LPCTX(N1)
            IGY=GRIDB%LPCTY(N2)
            IGZ=GRIDB%LPCTZ(N3)
            GX=IGX*B(1,1)+IGY*B(1,2)+IGZ*B(1,3)
            GY=IGX*B(2,1)+IGY*B(2,2)+IGZ*B(2,3)
            GZ=IGX*B(3,1)+IGY*B(3,2)+IGZ*B(3,3)
            GSQU=(GX*GX+GY*GY+GZ*GZ)*TPI*TPI
            P(K)=(GSQU+FLAM)/MAX(GSQU,1.E-10_q)
         ENDDO
         ELSE
         IF (LWRITE) WRITE(*,*) 'using inverse metric'
         DO K=1,GRIDB%RC%NP
            N1= MOD((K-1),GRIDB%RC%NROW) +1
            NC= (K-1)/GRIDB%RC%NROW+1
            N2= GRIDB%RC%I2(NC)
            N3= GRIDB%RC%I3(NC)

            IGX=GRIDB%LPCTX(N1)
            IGY=GRIDB%LPCTY(N2)
            IGZ=GRIDB%LPCTZ(N3)
            GX=IGX*B(1,1)+IGY*B(1,2)+IGZ*B(1,3)
            GY=IGX*B(2,1)+IGY*B(2,2)+IGZ*B(2,3)
            GZ=IGX*B(3,1)+IGY*B(3,2)+IGZ*B(3,3)
            GSQU=(GX*GX+GY*GY+GZ*GZ)*TPI*TPI
            P(K)=MAX(GSQU,1.E-10_q)/(GSQU+FLAM)
         ENDDO
         ENDIF
      END IF

      PROFILING_STOP('brpre')

      RETURN
      END SUBROUTINE

!************************* SUBROUTINE BRGRID ***************************
!
! Set up the grid size to be used for Broyden mixing (the rest will be
! mixed using conventional linear mixing or Kerker mixing ...):
!***********************************************************************

      SUBROUTINE BRGRID(GRID,GRIDB,ENMAX,IU6,B)
      USE prec
      USE mpimy
      USE mgrid

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (grid_3d)     GRIDB

      PARAMETER(SCALE=1._q,EUNITS=3.8100198741_q)
      DIMENSION B(3,3)

      PROFILING_START('brgrid')

      TPI=8._q*ATAN(1._q)
! The 'cutoff' were to truncate the grid is a matter of taste and a
! matter of empirical experiences. A surely not too bad guess is to
! take some value around the wavefunction-cutoff (parameter "SCALE"
! gives us some freedom for adjustement to some special multiple of
! GCUT_pw). Usually the mesh for the charge density is cut at 2*GCUT
! (or some value close to it) - so this would lead to approximately
! half size of the grid in each direction ... . The crude experience
! is that within 1.5-1.7*GCUT one finds already 99.9999...% of all
! charges - so cutting the grid here almost nothing would be left
! outside the Broyden-grid (except for augmentation charges which
! occur for ultrasoft potentials). Already within GCUT one should
! have most of the charge (80%] and all(99.9999%] of the really
! bad converging components which need treatment by a Broyden-mixing.
      GCUTSQ=ENMAX/EUNITS*SCALE
      GRIDB%NGX=0
      GRIDB%NGY=0
      GRIDB%NGZ=0
! Find largest indexes for all Q-vectors with Q^2 < THRESH * Q_min^2:
      col: DO NC=1,GRID%RC%NCOL
      N2= GRID%RC%I2(NC)
      N3= GRID%RC%I3(NC)
      row: DO N1=1,GRID%RC%NROW

         IGZ=GRID%LPCTZ(N3)
         IGY=GRID%LPCTY(N2)
         IGX=GRID%LPCTX(N1)

         GX=IGX*B(1,1)+IGY*B(1,2)+IGZ*B(1,3)
         GY=IGX*B(2,1)+IGY*B(2,2)+IGZ*B(2,3)
         GZ=IGX*B(3,1)+IGY*B(3,2)+IGZ*B(3,3)
         GSQU=(GX*GX+GY*GY+GZ*GZ)*TPI*TPI
         IF (GSQU>GCUTSQ) CYCLE
         GRIDB%NGX=MAX(ABS(IGX),GRIDB%NGX)
         GRIDB%NGY=MAX(ABS(IGY),GRIDB%NGY)
         GRIDB%NGZ=MAX(ABS(IGZ),GRIDB%NGZ)

      ENDDO row
      ENDDO col
      GRIDB%NGX=MIN(2*GRIDB%NGX+1,GRID%NGX)
      GRIDB%NGY=MIN(2*GRIDB%NGY+1,GRID%NGY)
      GRIDB%NGZ=MIN(2*GRIDB%NGZ+1,GRID%NGZ)

      CALLMPI ( M_max_i(GRID%COMM, GRIDB%NGX, 1))
      CALLMPI ( M_max_i(GRID%COMM, GRIDB%NGY, 1))
      CALLMPI ( M_max_i(GRID%COMM, GRIDB%NGZ, 1))
      IF (IU6>=0) &
      WRITE(IU6,1) GRIDB%NGX,GRIDB%NGY,GRIDB%NGZ, &
                   GRID%NGX ,GRID%NGY ,GRID%NGZ , &
                   GRIDB%NGX*GRIDB%NGY*GRIDB%NGZ
      
    1 FORMAT(' Broyden mixing: mesh for mixing (old mesh)'/ &
     &        '   NGX =',I3,'   NGY =',I3,'   NGZ =',I3/ &
     &        '  (NGX  =',I3,'   NGY  =',I3,'   NGZ  =',I3,')'/ &
     &        '  gives a total of ',I6,' points'/)

      PROFILING_STOP('brgrid')

      RETURN
      END SUBROUTINE

!************************* SUBROUTINE BRSAV ****************************
!
! Save a vector necessary for Broyden mixing
! in each iteration 3 datasets are used
!  F   gradient vector   |Delta F> (|F> for last iteration!!!)
!  Z   vector Z as defined in Johnsons paper (and corrected by gK)
!  U   vector U as defined in Johnsons paper
!      i.e |U> = G^(1) |Delta F> + |Delta X>:
!  the first 3 records of the file  (or broyden_storage)
!  are used to store
!  G   start mixing G^(1)
!  X   |X^(iter)> (used as |X^(iter-1)> in next iteration)
!  P   metric * residual vector |FP^(iter)>
!  in addition temporary 'swap  areas' are used
!  T   temporary
!***********************************************************************


      SUBROUTINE BRSAV(NDIM,DATA,WHAT,ITER,IOD,ITNOW,IERR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      CHARACTER (1) WHAT
      COMPLEX(q) DATA
      DIMENSION DATA(NDIM),IOD(3)

      PROFILING_START('brsav')

      IERR=0
#ifdef USE_TMPBROYD
        IU=IOD(1)
        SELECT CASE (WHAT)
          CASE ('F')
            IREC=3*ITER+2
          CASE ('U')
            IREC=3*ITER+3
          CASE ('Z')
            IREC=3*ITER+4
          CASE ('G')
            IREC=1
          CASE ('X')
            IREC=2
          CASE ('P')
            IREC=3
          CASE ('I')
            IREC=4
          CASE ('T')
          ! 'T' is put at the very end of the file
            IREC=3*ITNOW+1+ITER
        END SELECT

        WRITE(IU,REC=IREC,ERR=2) CMPLX(DATA,KIND=qs)

        PROFILING_STOP('brsav')

        RETURN
 2      IERR=1

      PROFILING_STOP('brsav')

      RETURN
#else
      IF (WHAT=='T') THEN
        CALL dyn_put(tmp_storage,NDIM,ITER,DATA,IERR)
      ELSE
        SELECT CASE (WHAT)
          CASE ('F')
            IREC=3*ITER+2
          CASE ('U')
            IREC=3*ITER+3
          CASE ('Z')
            IREC=3*ITER+4
          CASE ('G')
            IREC=1
          CASE ('X')
            IREC=2
          CASE ('P')
            IREC=3
          CASE ('I')
            IREC=4
        END SELECT
        CALL dyn_put(broyden_storage,NDIM,IREC,DATA,IERR)
      ENDIF

      PROFILING_STOP('brsav')

      RETURN
#endif
      END SUBROUTINE

!************************* SUBROUTINE BRGET ****************************
!
! Get some data stored previously (conventions see routine BRSAV):
!
!***********************************************************************

      SUBROUTINE BRGET(NDIM,DATA,WHAT,ITER,IOD,ITNOW,IERR)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      CHARACTER (1) WHAT
      COMPLEX(q) DATA
      DIMENSION DATA(NDIM),IOD(3)
      COMPLEX(qs) DATA_TMP(NDIM)

      PROFILING_START('brget')

      IERR=0
#ifdef USE_TMPBROYD
        IU=IOD(1)
        SELECT CASE (WHAT)
          CASE ('F')
            IREC=3*ITER+2
          CASE ('U')
            IREC=3*ITER+3
          CASE ('Z')
            IREC=3*ITER+4
          CASE ('G')
            IREC=1
          CASE ('X')
            IREC=2
          CASE ('P')
            IREC=3
          CASE ('I')
            IREC=4
          CASE ('T')
          ! 'T' is put at the very end of the file
            IREC=3*ITNOW+1+ITER
        END SELECT

        READ(IU,REC=IREC,ERR=2) DATA_TMP
        DATA=DATA_TMP

        PROFILING_STOP('brget')

        RETURN
 2      IERR=1

      PROFILING_STOP('brget')

      RETURN
#else
      IF (WHAT=='T') THEN
        CALL dyn_get(tmp_storage,NDIM,ITER,DATA,IERR)
      ELSE
        SELECT CASE (WHAT)
          CASE ('F')
            IREC=3*ITER+2
          CASE ('U')
            IREC=3*ITER+3
          CASE ('Z')
            IREC=3*ITER+4
          CASE ('G')
            IREC=1
          CASE ('X')
            IREC=2
          CASE ('P')
            IREC=3
          CASE ('I')
            IREC=4
        END SELECT
        CALL dyn_get(broyden_storage,NDIM,IREC,DATA,IERR)
      ENDIF

      PROFILING_STOP('brget')

      RETURN
#endif
      END SUBROUTINE

!***********************************************************************
!
!  remove temporary data
!
!***********************************************************************
      SUBROUTINE CLBROYD(IUBROY)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
#ifdef USE_TMPBROYD
      CLOSE(IUBROY,STATUS='DELETE',ERR=100)
  100 CONTINUE
#else
      CALL dyn_free(tmp_storage)
      CALL dyn_free(broyden_storage)
#endif
      RETURN
      END SUBROUTINE

#ifndef USE_TMPBROYD

!***********************************************************************
!
!  remove tmp_storage
!
!***********************************************************************
      SUBROUTINE BRRMTMP()
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      CALL dyn_free(tmp_storage)
      RETURN
      END SUBROUTINE

!***********************************************************************
!
!  remove temporary data
!
!***********************************************************************
      SUBROUTINE BRNULL()
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      NULLIFY(tmp_storage)
      NULLIFY(broyden_storage)
      RETURN

      END SUBROUTINE


!***********************************************************************
! dynamic storage retrival system by gK
! dyn_put   puts one complex array <data(ndim)> at position <irec> in a
!           list <root>, the list is extened dynamically if required
! dyn_get   retrives one complex array data(ndim) from position irec
! dyn_free  frees the whole list
! MIND:     ierr is not jet supported (not used at all!)
!
! written recursively because it is much shorter
! the problem with an iterative implementation is that there is nothing
! like a pointer to a pointer in F90, therefore we would have
! complicated IF cases for initial node in the list
! (whereas dummy arguments to pointer pass their association status back
!  to the CALLER, in some way they behave like pointers to pointer)
!***********************************************************************

      RECURSIVE SUBROUTINE dyn_put(root,ndim,irec,data,ierr)
      IMPLICIT NONE
      INTEGER :: irec,ierr,ndim
      COMPLEX(q) data(ndim)
      TYPE (dyn_storage),POINTER :: root

      IF (.NOT.ASSOCIATED(root)) THEN
         ALLOCATE(root)
         NULLIFY(root%store)
         ALLOCATE(root%store(ndim))
         NULLIFY(root%next)
      ENDIF
      IF (irec==1) THEN
        IF  (.NOT.ASSOCIATED(root%store)) THEN
          ALLOCATE(root%store(ndim))
        ENDIF
        root%store= data
      ELSE
        CALL dyn_put(root%next,ndim,irec-1,data,ierr)
      ENDIF
      END SUBROUTINE dyn_put


      RECURSIVE SUBROUTINE dyn_get(root,ndim,irec,data,ierr)
      IMPLICIT NONE
      INTEGER :: irec,ierr,ndim
      COMPLEX(q) data(ndim)
      TYPE (dyn_storage),POINTER :: root

      IF (.NOT.ASSOCIATED(root)) THEN
         WRITE(*,*)'BROYDEN: fatal error: dyn_get leaf not allocate'
         STOP
         ALLOCATE(root)
         NULLIFY(root%store)
         NULLIFY(root%next)
      ENDIF
      IF (irec==1) THEN
        IF  (.NOT.ASSOCIATED(root%store)) THEN
          WRITE(*,*)'BROYDEN: fatal error: dyn_get store not allocated'
          STOP
        ENDIF
        data=root%store
      ELSE
        CALL dyn_get(root%next,ndim,irec-1,data,ierr)
      ENDIF
      END SUBROUTINE dyn_get


      RECURSIVE SUBROUTINE dyn_free(root)
      IMPLICIT NONE
      TYPE (dyn_storage),POINTER :: root

      IF (.NOT.ASSOCIATED(root)) RETURN
      CALL dyn_free(root%next)
      IF (ASSOCIATED(root%store)) DEALLOCATE(root%store)
      DEALLOCATE(root)
      NULLIFY(root)
      RETURN
      END SUBROUTINE dyn_free
#endif
      END MODULE

!***********************************************************************
! SOME SMALL HELPER ROUTINES
! Inverts a square matrix using routines of "NUMERICAL RECIPES",
!   W. Press et al., Cambridge U.P., 1987.
! Arguments:
!   A   : Input square matrix
!   N   : Input dimension of the matrix
!   NP  : Input first physical dimension of matrices A and B
!   B   : Output inverse matrix. Make B=A for 'in place' inversion
!   AUX : Auxiliary array of minimun size N*N. If A and B are different
!         and matrix A needs not be preserved, you can make AUX=A
! Written by Jose Soler (JSOLER AT EMDUAM11) 22/4/90.
! Modified by J. Furthmueller 11/12/92
!
!***********************************************************************
      SUBROUTINE INVERS(A,N,NP,B,AUX,INDEX,VV)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (ZERO=0._q,ONE=1._q)
      DIMENSION A(NP,NP),B(NP,NP),AUX(NP,NP),INDEX(N),VV(N)
!
      DO I=1,N
       DO J=1,N
          AUX(J,I)=A(J,I)
       ENDDO
      ENDDO
      CALL LUDCM(AUX,N,NP,INDEX,VV)
      DO 30 I=1,N
         DO 20 J=1,N
            B(J,I)=ZERO
   20    CONTINUE
         B(I,I)=ONE
         CALL LUBKS(AUX,N,NP,INDEX,B(1,I))
   30 CONTINUE
      END SUBROUTINE

      SUBROUTINE LUDCM(A,N,NP,INDX,VV)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
!
! Decomposes a matrix in upper-lower triangular form.
! Ref: W. Press et al., "NUMERICAL RECIPES", Cambridge U.P., 1987
!
      PARAMETER (TINY=1.E-20_q, ZERO=0._q,ONE=1._q)
      DIMENSION A(NP,NP),INDX(N),VV(N)
!
      DO 12 I=1,N
         AAMAX=ZERO
         DO 11 J=1,N
            IF (ABS(A(I,J))>AAMAX) AAMAX=ABS(A(I,J))
  11     CONTINUE
         IF (AAMAX==ZERO) CALL ERROR(' LUDCMP',' SINGULAR MATRIX ',-3)
         VV(I)=ONE/AAMAX
  12  CONTINUE
      DO 19 J=1,N
         IF (J>1) THEN
            DO 14 I=1,J-1
               SUM=A(I,J)
               IF (I>1) THEN
                  DO 13 K=1,I-1
                     SUM=SUM-A(I,K)*A(K,J)
  13              CONTINUE
                  A(I,J)=SUM
               END IF
  14        CONTINUE
         ENDIF
         AAMAX=ZERO
         DO 16 I=J,N
            SUM=A(I,J)
            IF (J>1) THEN
               DO 15 K=1,J-1
                  SUM=SUM-A(I,K)*A(K,J)
  15           CONTINUE
               A(I,J)=SUM
            END IF
            DUM=VV(I)*ABS(SUM)
            IF (DUM>=AAMAX) THEN
               IMAX=I
               AAMAX=DUM
            END IF
  16     CONTINUE
         IF (J/=IMAX) THEN
            DO 17 K=1,N
               DUM=A(IMAX,K)
               A(IMAX,K)=A(J,K)
               A(J,K)=DUM
  17        CONTINUE
            VV(IMAX)=VV(J)
         END IF
         INDX(J)=IMAX
         IF (J/=N) THEN
            IF (A(J,J)==ZERO) A(J,J)=TINY
            DUM=ONE/A(J,J)
            DO 18 I=J+1,N
               A(I,J)=A(I,J)*DUM
  18        CONTINUE
         END IF
  19  CONTINUE
      IF (A(N,N)==ZERO) A(N,N)=TINY
      RETURN
      END SUBROUTINE


!***********************************************************************
!
! Solves a system of linear equations in combination with 'LUDCMP'
! Ref: W. Press et al., "NUMERICAL RECIPES", Cambridge U.P., 1987
!
!***********************************************************************
      SUBROUTINE LUBKS(A,N,NP,INDX,B)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      PARAMETER (ZERO=0._q)
      DIMENSION A(NP,NP),INDX(N),B(N)
!
      II=0
      DO 12 I=1,N
         LL=INDX(I)
         SUM=B(LL)
         B(LL)=B(I)
         IF (II/=0) THEN
            DO 11 J=II,I-1
               SUM=SUM-A(I,J)*B(J)
  11        CONTINUE
         ELSE IF (SUM/=ZERO) THEN
            II=I
         END IF
         B(I)=SUM
  12  CONTINUE
      DO 14 I=N,1,-1
         SUM=B(I)
         IF (I<N) THEN
            DO 13 J=I+1,N
               SUM=SUM-A(I,J)*B(J)
  13        CONTINUE
         END IF
         B(I)=SUM/A(I,I)
  14  CONTINUE
      RETURN
      END SUBROUTINE
