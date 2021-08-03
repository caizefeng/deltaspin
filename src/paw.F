!#define debug
#include "symbol.inc"
!*******************************************************************
! RCS:  $Id: paw.F,v 1.14 2003/06/27 13:22:22 kresse Exp kresse $
!
!  PAW-module
!  implements the top level PAW functions
!  most of the functionality is implemented in the MODULES radial
!
!  all routine written by Georg Kresse
!  (even the symmetrization routines :)
!*******************************************************************
  MODULE pawm
    USE prec
    USE pseudo
    USE paw
    CONTAINS

!*******************************************************************
!
!  start up procedure for PAW
!  checks internal consistency of the QPAW
!  and sets up the compensation charges
!  on the radial grid, and spline coefficients which are used
!  to interpolate compensation charges in us.F
!
!*******************************************************************

      SUBROUTINE SET_PAW_AUG(NTYPD, P, IU6, LMAX_CALC, LCOMPAT)
        USE radial
        USE LDAPLUSU_MODULE
        IMPLICIT NONE
        INTEGER NTYPD
        INTEGER IU6
        TYPE (potcar),TARGET :: P(NTYPD)
        TYPE (rgrid),POINTER :: R
        INTEGER LMAX_CALC  ! if -1 mimic US PP (see below)
      ! local variables
        INTEGER NTYP,CHANNELS,LMAX,L,LP,N,I
        INTEGER, PARAMETER :: NQ=2
        REAL(q)  QQ(NQ)
        REAL(q)  A(NQ),B,ALPHA
        REAL(q)  SUM,QR,BJ,STEP,X
        REAL(q),PARAMETER ::  TH=1E-6_q
        LOGICAL LCOMPAT


      ! if LMAX_CALC == -1 the PAW will run in a special mode
      ! resulting in essentially US-PP like behaviour
      ! i.e. all terms are linearized around the atomic reference configuration
      ! and one center terms are not evaluated

        IF (LMAX_CALC ==-1) THEN
           MIMIC_US=.TRUE.
        ELSE
           MIMIC_US=.FALSE.
        ENDIF

        LMAX_MIX=0

        typ: DO NTYP=1,NTYPD
           IF (.NOT. ASSOCIATED(P(NTYP)%QPAW)) CYCLE

           ! maximal L mixed in Broyden mixer
           LMAX_MIX=MAX(LMAX_MIX,P(NTYP)%LMAX_CALC)

           R => P(NTYP)%R
           CALL  RAD_ALIGN(R,R%RMAX) ! reallign RMAX with grid

           CALL RAD_CHECK_QPAW( R, P(NTYP)%LMAX, &
                P(NTYP)%WAE, P(NTYP)%WPS, P(NTYP)%QPAW, P(NTYP)%QTOT , P(NTYP)%LPS, P(NTYP)%RDEP )

           CALL SET_OPT_PROJ(P(NTYP))

           IF ((USELDApU().OR.LCALC_ORBITAL_MOMENT()).AND.INTEGRALS_LDApU()) THEN
              CALL OVERLAP_AE(R,P(NTYP)%RDEP,NTYP,NTYPD,P(NTYP)%LMAX,P(NTYP)%WAE,P(NTYP)%LPS)
           ENDIF

           CHANNELS=P(NTYP)%LMAX
           DO L=1, CHANNELS
              DO LP=1, P(NTYP)%LMAX
                 IF (P(NTYP)%LPS(L)==P(NTYP)%LPS(LP)) THEN
                    P(NTYP)%QION(L,LP)=P(NTYP)%QPAW(L,LP,0)
                 ENDIF
              ENDDO
           ENDDO

           LMAX=0
           DO I=1,CHANNELS
              LMAX=MAX( P(NTYP)%LPS(I),LMAX )
           ENDDO

           LMAX=LMAX*2+1              ! maximum l in augmentation charges
                                      ! to allow use of one-center dipole operators increase by one
           ALLOCATE(P(NTYP)%QDEP(NPSRNL,5,0:LMAX), &
                    P(NTYP)%AUG (R%NMAX,0:LMAX) )

!           IF (IU6>=0) WRITE(IU6,"(' L augmenation charges for type=',I4)") NTYP

           ll: DO L=0,LMAX

        ! find q values
              CALL AUG_SETQ(L,R,R%RMAX,QQ,A,LCOMPAT)
!              IF (IU6>=0) WRITE(IU6,2) L,QQ,A

        ! setup augmentation charge on radial grid rho(r) r^2

              DO N=1,R%NMAX
                 SUM=0
                 IF (R%R(N) <= R%RMAX) THEN
                    DO I=1,NQ
                       QR=QQ(I)*R%R(N)
                       CALL SBESSEL( QR, BJ, L)
                       SUM=SUM+BJ*A(I)*R%R(N)*R%R(N)
                    ENDDO
                 ENDIF
                 P(NTYP)%AUG(N,L)=SUM
              ENDDO

        ! setup spline for augmentation charge
              ! the spline ends at PSDMAX*(NPSRNL-1)/NPSRNL see SETDEP
              STEP= R%RMAX/(NPSRNL-1)
        ! this resets PSDMAX which is from now on no longer identical to R%RMAX
              P(NTYP)%PSDMAX=NPSRNL*STEP

              DO N=1,NPSRNL
                 X=STEP*(N-1)
                 SUM=0
                 DO I=1,NQ
                    QR=QQ(I)*X
                    CALL SBESSEL( QR, BJ, L)
                    SUM=SUM+BJ*A(I)
                 ENDDO
                 P(NTYP)%QDEP(N,1,L) = X
                 P(NTYP)%QDEP(N,2,L) = SUM
              ENDDO
              ! derivative at startpoint
              X=STEP/1000
              SUM=0
              DO I=1,NQ
                 QR=QQ(I)*X
                 CALL SBESSEL( QR, BJ, L)
                 SUM=SUM+BJ*A(I)
              ENDDO
              SUM=(SUM-P(NTYP)%QDEP(1,2,L))/X
              ! derivative is zero for all but L=1
              IF (L/=1) THEN
                 SUM=0
              ENDIF
              CALL SPLCOF(P(NTYP)%QDEP(1,1,L),NPSRNL,NPSRNL,SUM)
           ENDDO ll
           P(NTYP)%AUG_SOFT =>P(NTYP)%AUG

        ENDDO typ

 ! mix only L=SET_LMAX_MIX_TO components in Broyden mixer
        LMAX_MIX=MIN(LMAX_MIX,SET_LMAX_MIX_TO)

 ! if US-PP are mimicked no need to mix any onsite components
        IF (MIMIC_US) LMAX_MIX=-1

        RETURN

        CONTAINS

!**********************************************************************
!
! small subroutine to set OPTPROJ
!  QTOT= O lambda O
!  OPTPROJ is set to lambda^(1/2) O
! this allows to define well behaved projection operators
!
!**********************************************************************

        SUBROUTINE SET_OPT_PROJ(PP)
          TYPE (potcar) PP
          INTEGER :: LOW, LL, LHI, N_CHANNELS, I
        ! local
          INTEGER, PARAMETER :: NDMAT=4        ! maximum main quantum numbers per L channel
          REAL(q) :: EV(NDMAT,NDMAT)

        ! not very elegant just limit maximum number of channels to 4
          ALLOCATE(PP%OPTPROJ(NDMAT, NDMAT, PP%LMAX))
          PP%OPTPROJ=0

        ! now loop over all L quantum numbers
          LOW=1
          DO
             ! seek all channels with same L quantum number
             LL=PP%LPS(LOW)
             DO LHI=LOW,PP%LMAX
                IF (LL/=PP%LPS(LHI)) EXIT
             ENDDO
             LHI=LHI-1
             IF (LL>PP%LMAX) THEN
                WRITE(0,*) 'internal error in SET_OPT_PROJ: increase PP%LMAX',PP%LMAX
                STOP
             ENDIF

             N_CHANNELS=LHI-LOW+1
             IF (N_CHANNELS> NDMAT) THEN
                WRITE(0,*) 'internal error in SET_OPT_PROJ: increase NDMAT', NDMAT
                STOP
             ENDIF

             ! diagonalize EV and set 
             EV(1:N_CHANNELS,1:N_CHANNELS)=PP%QTOT(LOW:LHI,LOW:LHI)

             CALL DIAG_AND_SORT(EV, N_CHANNELS)
!            DO I=1,N_CHANNELS
!               WRITE(*,'(5F14.7)') EV(1:N_CHANNELS,I)
!            ENDDO
             PP%OPTPROJ(1:N_CHANNELS,1:N_CHANNELS,LL+1)=EV(1:N_CHANNELS,1:N_CHANNELS)

             LOW=LHI+1
             IF (LOW > PP%LMAX) EXIT
          ENDDO
        END SUBROUTINE SET_OPT_PROJ

!*******************************************************************
!
! small helper routine to diagonalize an overlap matrix
!
! e.g. Q= <phi_i^AE | phi_j^AE >
!
! and calculate Q^(1/2)
! the sorting of the eigenvectors is such that the original order
! is largely maintained
! EV is also zeroed if N_CHANNELS is not using the entire array
! note: identical routine in sphpro.F
!
!********************************************************************

        SUBROUTINE DIAG_AND_SORT(EV, N_CHANNELS )
          IMPLICIT NONE
          INTEGER N_CHANNELS
          REAL(q) :: EV(:,:)
          ! local
          INTEGER, PARAMETER :: NDMAT=4        ! maximum main quantum numbers per L channel
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


      END SUBROUTINE SET_PAW_AUG


!*******************************************************************
!
! WRT_RHO_PAW write the PAW occupancies to a file specified by
! IU
!
!*******************************************************************

    SUBROUTINE WRT_RHO_PAW(P, T_INFO, LOVERL, RHOLM_STORE, COMM, IU )
      USE pseudo
      USE poscar
      USE wave
      USE constant
      IMPLICIT NONE

      INTEGER IU               ! io unit
      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      LOGICAL  LOVERL          ! overlap matrix used ?
      REAL(q) RHOLM_STORE(:)   ! storage for the channel occupancies
      TYPE(communic) :: COMM

    ! local variables
      TYPE (potcar),POINTER :: PP
      INTEGER NT, NI, I
      INTEGER IBASE, IADD, NELEMENTS, LYMAX, LMMAX
      INTEGER, EXTERNAL :: MAXL_AUG
      INTEGER NODE_ME, IONODE

      REAL(q), ALLOCATABLE::  BUFFER(:)

#ifdef MPI
      NODE_ME=COMM%NODE_ME
      IONODE =COMM%IONODE
#endif

!=======================================================================
! quick return if possible
!=======================================================================
      IF (.NOT.LOVERL .OR. MIMIC_US ) RETURN

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)
      LMMAX=(LYMAX+1)**2

      ALLOCATE( BUFFER(LMMAX*LMMAX))

!=======================================================================
! cycle all ions and write the required elements
!=======================================================================
      IBASE=1

      ion: DO NI=1,T_INFO%NIONS
         BUFFER=0
         IADD  =0

         NT=T_INFO%ITYP(NI)
         PP=> P(NT)

         NELEMENTS =0 
         IF (DO_LOCAL(NI)) THEN
            CALL RETRIEVE_RHOLM( BUFFER, RHOLM_STORE(IBASE:), &
                 METRIC(IBASE:), IADD, PP, .TRUE., NELEMENTS)

            IF (NELEMENTS > LMMAX*LMMAX) THEN
               WRITE(*,*)'internal ERROR: WRT_RHO_PAW running out of buffer'
               STOP
            ENDIF
            
            IBASE=IBASE+IADD
         ENDIF

         CALLMPI( M_sum_i(COMM, NELEMENTS, 1))
         CALLMPI( M_sum_d(COMM, BUFFER, NELEMENTS))
         
         io_begin
         WRITE(IU,'("augmentation occupancies",2I4)') NI, NELEMENTS
         WRITE(IU,'(5E15.7)') (BUFFER(I),I=1,NELEMENTS)
         io_end

      ENDDO ion

#ifdef RHOLM_complex
      IBASE=1

      again: DO NI=1,T_INFO%NIONS
         BUFFER=0
         IADD  =0

         NT=T_INFO%ITYP(NI)
         PP=> P(NT)

         NELEMENTS =0 
         IF (DO_LOCAL(NI)) THEN
            CALL RETRIEVE_RHOLM( BUFFER, RHOLM_STORE(IBASE+UBOUND(RHOLM_STORE,1)/2:), &
                 METRIC(IBASE+UBOUND(RHOLM_STORE,1)/2:), IADD, PP, .TRUE., NELEMENTS)

            IF (NELEMENTS > LMMAX*LMMAX) THEN
               WRITE(*,*)'internal ERROR: WRT_RHO_PAW running out of buffer'
               STOP
            ENDIF
            
            IBASE=IBASE+IADD
         ENDIF

         CALLMPI( M_sum_i(COMM, NELEMENTS, 1))
         CALLMPI( M_sum_d(COMM, BUFFER, NELEMENTS))
         
         io_begin
         WRITE(IU,'("augmentation occupancies (imaginary part)",2I4)') NI, NELEMENTS
         WRITE(IU,'(5E15.7)') (BUFFER(I),I=1,NELEMENTS)
         io_end

      ENDDO again
#endif
      DEALLOCATE(BUFFER)
    END SUBROUTINE WRT_RHO_PAW

!*******************************************************************
!
! RD_RHO_PAW read the PAW occupancies from a file specified by
! IU
! only components up to LMAX_MIX are read in
! the remaining components are required by the routine
! SET_DD_PAW
! the current program logic is that these are determined from CRHODE
! which is initialised according to atomic occupancies
!
!*******************************************************************

    SUBROUTINE RD_RHO_PAW(P, T_INFO, LOVERL, RHOLM_STORE, COMM, IU, IERR )
      USE pseudo
      USE poscar
      USE wave
      USE constant
      IMPLICIT NONE

      INTEGER IU               ! io unit
      INTEGER IERR             ! error status on return
                               ! 0 ok, 1 error occured
      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      LOGICAL  LOVERL          ! overlap matrix used ?
      REAL(q) RHOLM_STORE(:)   ! storage for the channel occupancies
      TYPE(communic) :: COMM

    ! local variables
      TYPE (potcar),POINTER :: PP
      INTEGER NT, NI, I
      INTEGER IBASE, IADD, NELEMENTS, LYMAX, LMMAX
      INTEGER, EXTERNAL :: MAXL_AUG
      INTEGER NI_READ, NELEMENTS_READ
      INTEGER NODE_ME, IONODE
      INTEGER IFLAG

      REAL(q), ALLOCATABLE::  BUFFER(:)
      CHARACTER (1) CH

#ifdef MPI
      NODE_ME=COMM%NODE_ME
      IONODE =COMM%IONODE
#endif

!=======================================================================
! quick return if possible
!=======================================================================
      IF (.NOT.LOVERL .OR. MIMIC_US ) RETURN

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)
      LMMAX=(LYMAX+1)**2

      ALLOCATE( BUFFER(LMMAX*LMMAX))

!=======================================================================
! cycle all ions and write the required elements
!=======================================================================
      IBASE=1

      ion: DO NI=1,T_INFO%NIONS
         BUFFER=0
         IADD  =0

         NT=T_INFO%ITYP(NI)
         PP=> P(NT)

         IERR=0


         io_begin
         NI_READ=0
         NELEMENTS_READ=0

         READ(IU,'(24X,2I4)',IOSTAT=IERR) &
              NI_READ, NELEMENTS_READ

         IF (NELEMENTS_READ > LMMAX*LMMAX) THEN
            WRITE(*,*)'internal ERROR: RD_RHO_PAW running out of buffer'
            STOP
         ENDIF

         IF (IERR == 0 ) READ(IU,*,IOSTAT=IERR) (BUFFER(I),I=1,NELEMENTS_READ)
         IF (NI_READ /= NI .OR. IERR/=0 ) THEN
            IERR=1
         ENDIF
         io_end

         CALLMPI( M_bcast_i( COMM, NELEMENTS_READ , 1))
         CALLMPI( M_sum_d(COMM, BUFFER, NELEMENTS_READ))

         NELEMENTS=0
         IF (DO_LOCAL(NI)) THEN
            CALL STORE_RHOLM( BUFFER, RHOLM_STORE(IBASE:), &
                 METRIC(IBASE:), IADD, PP, .TRUE. , NELEMENTS)
            IBASE=IBASE+IADD
         ENDIF

         CALLMPI( MPI_barrier( COMM%MPI_COMM, I ))
         CALLMPI( M_sum_i(COMM, NELEMENTS, 1))

         io_begin
         IF (NELEMENTS /= NELEMENTS_READ)  THEN
           IERR=1
         ENDIF
         io_end

         CALLMPI( M_bcast_i( COMM, IERR, 1))
         
         IF (IERR/=0) THEN
            do_io WRITE(*,*) 'RD_RHO_PAW: ion', NI,'data corrupt'
            DEALLOCATE (BUFFER)
            RETURN
         ENDIF
         
      ENDDO ion

      ! Now try to find out whether or not the imaginary part
      ! of the occupancies were written to this file as well
      IFLAG=0; CH=' '
      io_begin
      READ(IU,'(A)',ADVANCE='No',END=100) CH
      IF (CH=='a') IFLAG=1
      io_end
 100  CONTINUE 
      CALLMPI( M_sum_i(COMM, IFLAG, 1))

      ! if so then read the imaginary parts from file
      IF (IFLAG>0) THEN
      do_io WRITE(*,*) 'reading imaginary part of occupancies ...'

      IBASE=1

      again: DO NI=1,T_INFO%NIONS
         BUFFER=0
         IADD  =0

         NT=T_INFO%ITYP(NI)
         PP=> P(NT)

         IERR=0


         io_begin
         NI_READ=0
         NELEMENTS_READ=0

         IF (NI==1) THEN
            READ(IU,'(40X,2I4)',IOSTAT=IERR) &
                 NI_READ, NELEMENTS_READ
         ELSE
            READ(IU,'(41X,2I4)',IOSTAT=IERR) &
                 NI_READ, NELEMENTS_READ
         ENDIF

         IF (NELEMENTS_READ > LMMAX*LMMAX) THEN
            WRITE(*,*)'internal ERROR: RD_RHO_PAW running out of buffer'
            STOP
         ENDIF

         IF (IERR == 0 ) READ(IU,*,IOSTAT=IERR) (BUFFER(I),I=1,NELEMENTS_READ)
         IF (NI_READ /= NI .OR. IERR/=0 ) THEN
            IERR=1
         ENDIF
         io_end
#ifdef RHOLM_complex
         CALLMPI( M_bcast_i( COMM, NELEMENTS_READ , 1))
         CALLMPI( M_sum_d(COMM, BUFFER, NELEMENTS_READ))

         NELEMENTS=0
         IF (DO_LOCAL(NI)) THEN
            CALL STORE_RHOLM( BUFFER, RHOLM_STORE(IBASE+UBOUND(RHOLM_STORE,1)/2:), &
                 METRIC(IBASE+UBOUND(RHOLM_STORE,1)/2:), IADD, PP, .TRUE. , NELEMENTS)
            IBASE=IBASE+IADD
         ENDIF

         CALLMPI( MPI_barrier( COMM%MPI_COMM, I ))
         CALLMPI( M_sum_i(COMM, NELEMENTS, 1))

         io_begin
         IF (NELEMENTS /= NELEMENTS_READ)  THEN
           IERR=1
         ENDIF
         io_end

         CALLMPI( M_bcast_i( COMM, IERR, 1))
         
         IF (IERR/=0) THEN
            do_io WRITE(*,*) 'RD_RHO_PAW: ion', NI,'data corrupt'
            DEALLOCATE (BUFFER)
            RETURN
         ENDIF
#endif   
      ENDDO again
      ENDIF

      DEALLOCATE(BUFFER)

    END SUBROUTINE RD_RHO_PAW

!*******************************************************************
!
! this subroutine calculates the average local potential
! inside the PAW sphere it is required for relaxed core calculations
!
!*******************************************************************

    SUBROUTINE SET_AVERAGEPOT(PP)
      USE prec
      USE constant
      USE radial
      USE pseudo
      
      IMPLICIT NONE
      
      TYPE (potcar) PP

      REAL(q) :: POT(PP%R%NMAX)

   ! local variables
      INTEGER I
      REAL(q) SUM
      REAL(q) SCALE
      REAL(q) DHARTREE

      ! scale is simply Y_00
      SCALE=1/(2*SQRT(PI))
      ! calculate the Hatree potential of the core electrons
      CALL RAD_POT_HAR(0,PP%R,POT, PP%RHOAE, DHARTREE)
!     WRITE(0,'(8F14.7)') APOT(R%NMAX-7:R%NMAX)* R%R(R%NMAX-7:R%NMAX)/RYTOEV/AUTOA*SCALE
      ! add the potential of the nucleus (delta like charge at origin)
      POT=POT*SCALE-FELECT/PP%R%R*(PP%ZCORE+PP%ZVALF_ORIG)

      DO I=1,PP%R%NMAX
         POT(I)=POT(I)+PP%POTPSC(PP%R%NMAX)-POT(PP%R%NMAX)
      ENDDO
! test
!      WRITE(*,*) 'SET_AVERAGEPOT: V_h[Nc](Rmax)+V_h[Z](Rmax):',POT(PP%R%NMAX)
! test      
      ! add the valence contributions
      POT=POT-PP%POTAE
! test
!      WRITE(*,*) 'SET_AVERAGEPOT: VAERMAX:',POT(PP%R%NMAX)
! test      

!-----------------------------------------------------------------------
! calculate \int V(r) aug(r)
!-----------------------------------------------------------------------
      SUM=0
      DO I=1,PP%R%NMAX
         SUM=SUM+POT(I)*PP%AUG(I,0)*PP%R%SI(I)
      ENDDO

      PP%AVERAGEPOT(1)=SUM
! test
!      WRITE(*,*) 'SET_AVERAGEPOT: PP%AVERAGEPOT(1)=',PP%AVERAGEPOT(1)
! test      

    END SUBROUTINE SET_AVERAGEPOT

!*******************************************************************
!
! this subroutine calculates the one-centre (on site) potentials
! for atomic occupancies within the augmentation spheres;
! these are stored as the "reference" potentials in POTAE and POTPS
! the actual potentials are referenced to these potentials
!
! it is called to reduce numerical errors introduced by
! different implementations of the exchange correlation functional
! used in the pseudopotential generation code and VASP
!
! additionally it calculates an energy correction to the
! atomic reference energy if the type of the exchange correlation
! functional has changed
! The following entires in PP are updated
!   PP%ZCORE      number of core electrons (integer)
!   PP%DEXCCORE   exchange correlation energy of frozen core
!   PP%DEXCCOREM  exchange correlation energy of frozen core (MetaGGA)
!                 (contribution outside PAW core region is added)
!   PP%POTAE      AE one electron potential in reference state
!   PP%POTAE_XCUPDATED update to new xc potential
!   PP%POTPS      PS one electron potential 
!   PP%POTPSC     PS one electron potential core only
!   PP%EATOM      atomic energy
!   PP%EATOM_CORRECTED 
!   PP%DION       one center contribution 
!
! TODO: Cu atom: energy using new updated potential is 
!                very different for LMAXPAW = -1
!                PAW terms are identical to old version
!                probably energy has to be reconsidered
!
!*******************************************************************

    SUBROUTINE SET_PAW_ATOM_POT( P , T_INFO, LOVERL, LMDIM, EALLAT, LMETAGGA, IU6 )

      USE pseudo
      USE asa
      USE poscar
      USE wave
      USE constant
      USE radial
      USE setexm
      USE pawfock_inter
      USE pawkli
      USE meta
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::      P(T_INFO%NTYP)
      INTEGER LMDIM
      LOGICAL LMETAGGA
      INTEGER  IU6              ! io unit for OUTCAR 
      REAL(q) EALLAT            ! atomic reference energy
      LOGICAL  LOVERL

    ! local variables
      TYPE (potcar),POINTER:: PP
      INTEGER NT,LYMAX,NI,NDIM,LMMAX,M,K
      INTEGER,PARAMETER :: ISPIN=1
      INTEGER, EXTERNAL :: MAXL_AUG,MAXL1,MAXL
      REAL(q)  RHOLM(LMDIM*LMDIM), DLM(LMDIM*LMDIM)
      OVERLAP  CRHODE(LMDIM,LMDIM,1),CHF(LMDIM,LMDIM,1)
      REAL(q),ALLOCATABLE :: POT(:,:,:), RHO(:,:,:),POTOLD(:),POTDIFF(:)
      REAL(q) :: DOUBLEC_AE,DOUBLEC_PS,EXCG_OLD,EXCG,VX
      REAL(q) :: DOUBLEPS,DOUBLEAE,DOUBLEC_HF,EX_HF,SCALE
      REAL(q) :: DOUBLEPS_OLD,DOUBLEAE_OLD
      REAL(q) :: Z
      REAL(q) :: DEXC,DEXC_OLD,DEXCM
      REAL(q) :: DEATOM             ! change of valence-valence and core-valence energy
                                    ! related to change of the exchange correlation functional
    ! variables required to store core wavefunctions
      INTEGER MAXNL
      REAL(q), ALLOCATABLE :: W(:,:), EIG(:)
      INTEGER, ALLOCATABLE :: N(:), LC(:)
      LOGICAL, EXTERNAL :: USEFOCK_ONECENTER, USEFOCK_AE_ONECENTER
      INTEGER, EXTERNAL :: ONE_CENTER_NMAX_FOCKAE
      INTEGER :: NAE

      INTEGER :: NTP, NTPP, NTMAX
      REAL(q) :: WTOTAL

      LOGICAL :: VERSION46=.FALSE.
!=======================================================================
! quick return and allocation of work space
!=======================================================================
      DOUBLEC_AE=0
      DOUBLEC_PS=0
      IF (.NOT.LOVERL) RETURN

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)
      LMMAX=(LYMAX+1)**2

      CALL RAD_AUXILIARY_FUNCTIONS_METAGGA(MAXL(T_INFO%NTYP,P))

      NDIM=0
      DO NT=1,T_INFO%NTYP
         IF (ASSOCIATED(P(NT)%QPAW)) THEN
            NDIM=MAX(NDIM, P(NT)%R%NMAX)
         END IF
      ENDDO

      IF (NDIM == 0) RETURN
      ALLOCATE ( POT( NDIM, LMMAX, ISPIN ), RHO( NDIM, LMMAX, ISPIN ),POTOLD(NDIM), POTDIFF(NDIM))
!=======================================================================
! cycle all ions and recalculate the local potentials
! in VASP the local reference potentials are those for the atom
! if the XC potential has been changed on the fly, the calculation
! of the reference energy (atomic energy) is somewhat involved
!=======================================================================
      SCALE=2*SQRT(PI)

      DO NT=1,T_INFO%NTYP
         DEATOM=0
         PP=> P(NT)
    ! restore original values read from files
         PP%EATOM   =PP%EATOM_ORIG
         PP%EATOM_CORRECTED   =PP%EATOM_CORRECTED_ORIG
         PP%DEXCCORE=PP%DEXCCORE_ORIG
         PP%DION    =PP%DION_ORIG

         IF ( .NOT. ASSOCIATED(P(NT)%QPAW )) CYCLE

         PP%POTAE   =PP%POTAE_ORIG
         PP%POTPS   =PP%POTPS_ORIG
         PP%POTPSC  =PP%POTPSC_ORIG

    ! calculate  total electronic core charge 
    ! round up in all cases, since some electronic charge might be outside the
    ! augmentation sphere
         CALL SIMPI(PP%R, PP%RHOAE, Z)
         PP%ZCORE= AINT(Z*sqrt(4*PI)+0.9)

    ! calculate the LDA/GGA exchange correlation energy of the core
    ! the contribution from outside the core radius (R%NMAX)
    ! was read from the POTCAR file (entry DEXC) and stored in P%DEXCCORE
    ! (vasp.5.1: the latest potentials are constructed such that DEXC is essentially zero)
         IF (LUSE_THOMAS_FERMI) &
              CALL PUSH_XC_TYPE(LEXCH, LDAX, ALDAC, AGGAX, AGGAC, 0.0_q)

         CALL APPLY_ONE_CENTER_AEXX()
         CALL RAD_CORE_XC( PP%R, PP%RHOAE, DEXC)
         CALL RESTORE_ONE_CENTER_AEXX

         IF (LUSE_THOMAS_FERMI) CALL POP_XC_TYPE

         ! the same for metaGGA
         IF (LMETAGGA) THEN
            CALL RAD_CORE_META_XC( PP%R, PP%RHOAE, PP%TAUAE, DEXCM)
         ELSE
            DEXCM=0
         ENDIF
         ! DEXCCORE is later subtracted from the energy
         PP%DEXCCOREM=PP%DEXCCORE+DEXCM
         PP%DEXCCORE =PP%DEXCCORE+DEXC

         ! switch to original potential type
         IF(.NOT. VERSION46) CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q, 0.0_q)
         CALL RAD_CORE_XC( PP%R, PP%RHOAE, DEXC_OLD)
         IF(.NOT. VERSION46) CALL POP_XC_TYPE

         ! negative change of core energy due to change of exchange correlation functional
         DEATOM=DEXC_OLD-DEXC

    ! calculate the energies and potential of the spherical atom

         ! set RHOLM and CRHODE to atomic occupancies
         CALL SET_CRHODE_ATOM(CRHODE(:,:,1), PP)
         ! transform CRHODE to llp,LM
         RHOLM=0
         CALL TRANS_RHOLM( CRHODE(:,:,1), RHOLM(:), PP )
         ! get the spin up density
         CRHODE=CRHODE/2

         LYMAX =MAXL1(PP)*2

!-----------------------------------------------------------------------
! update reference one centre AE potential
! PP%POTAE stores the "valence only contribution"
! (except for the DFT exchange correlation part)
!  V_H(rho_val) + V_xc(rho_val+rho_core)
!-----------------------------------------------------------------------
         CALL SET_AVERAGEPOT(PP)

         POTOLD=0
         POTOLD(1:SIZE(PP%POTAE))=PP%POTAE
         PP%POTAE=0
         RHO=0
     ! get atomic charge density
         CALL RAD_CHARGE( RHO(:,:,1), PP%R, RHOLM, PP%LMAX, PP%LPS, PP%WAE )

         IF (LUSE_THOMAS_FERMI) &
              CALL PUSH_XC_TYPE(LEXCH, LDAX, ALDAC, AGGAX, AGGAC, 0.0_q)
         CALL APPLY_ONE_CENTER_AEXX()
     ! recalculate  V_H(rho_val) + V_xc(rho_val+rho_core) using the present
     ! one-centre xc potential
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,   &
                 RHO, PP%RHOAE, PP%POTAE,  POT, DOUBLEAE, EXCG)
! test
!        CALL RAD_POT_METAGGA( PP%R, ISPIN, 1,   &
!                RHO, PP%RHOAE, PP%POTAE,  POT, DOUBLEAE, EXCG)
! test
         CALL RESTORE_ONE_CENTER_AEXX
         IF (LUSE_THOMAS_FERMI) CALL POP_XC_TYPE

         ! WRITE(*,*) 'all electron core+ valence ',EXCG

     ! recalculate  V_H(rho_val) + V_xc(rho_val+rho_core) using the original
     ! one-centre xc potential
         IF(.NOT. VERSION46) CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q, 0.0_q)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOAE, PP%POTAE,  POT, DOUBLEAE_OLD, EXCG_OLD)
! test
!        CALL RAD_POT_METAGGA( PP%R, ISPIN, 1,  &
!                RHO, PP%RHOAE, PP%POTAE,  POT, DOUBLEAE_OLD, EXCG_OLD)
! test
         ! WRITE(0,*) 'all electron core+ valence old',EXCG_OLD
         IF(.NOT. VERSION46) CALL POP_XC_TYPE

    ! change of (core+val) energy due to change of exchange correlation functional
    ! minus change of core energy
         DEATOM=DEATOM-EXCG_OLD+EXCG

    ! "double counting" correction for PAW
    ! if the PAW one center terms are not updated:
    ! it is given by  \int dr (e_xc_new(r) - v_xc_old(r)) rho(r)
         VX=EXCG_OLD-DOUBLEAE_OLD
         DOUBLEAE=EXCG-VX

         DO K=1,PP%R%NMAX
           PP%POTAE(K)=-POT(K,1,1)/SCALE
           PP%POTAE_XCUPDATED(K)=-POT(K,1,1)/SCALE
         ENDDO
#define update_potential
#ifdef update_potential
         PP%POTAE_XCUPDATED=0
    ! at this point POT contains the old "valence" only one-electron potential
    ! calculate one center contributions and store them in CHF
    ! CHF = -\int Q_ij(r) (V^H(rho_v) + V^xc(rho_v+rho_c))
    ! this term cancels the contribution stored in DION by the
    ! pseudo-potential generation code
         DLM=0
         CALL RAD_POT_WEIGHT( PP%R, 1, 0, POT)
         CALL RAD_PROJ(  POT(:,:,1)  , PP%R,1._q, DLM, PP%LMAX, PP%LPS, PP%WAE )
         ! transform them using Clebsch Gordan coefficients and store in CHF
         CHF=0
         CALL TRANS_DLM( CHF(:,:,1), DLM , PP )
         CHF=-CHF
!#define DFT_core_valence
#ifdef DFT_core_valence
!
! the flag DFT_core_valence allows to calculate the core-valence
! interaction on the DFT level
! by calling RAD_POT with the valence density set to zero (RHO*0.0_q)
! and adding this to CHF the core DFT exchange correlation contribution is added to PP%DION
!   CHF += \int Q_ij(r) V_xc(rho_core)
! can be used for HF/hybrid or GW calculations
         CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q, 0.0_q)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO*0.0_q, PP%RHOAE, PP%POTAE_XCUPDATED,  POT, DOUBLEAE_OLD, EXCG_OLD)
         CALL POP_XC_TYPE
         DLM=0
         CALL RAD_POT_WEIGHT( PP%R, 1, 0, POT)
         CALL RAD_PROJ(  POT(:,:,1)  , PP%R,1._q, DLM, PP%LMAX, PP%LPS, PP%WAE )
         CALL TRANS_DLM( CHF(:,:,1), DLM , PP )
#endif
    ! get new one center AE potential for atom ("valence only")
    ! this might be different from above if the exchange correlation functional
    ! has changed
    !  V_H(rho_val) + V_xc_updated(rho_val+rho_core)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOAE, PP%POTAE_XCUPDATED,  POT, DOUBLEAE_OLD, EXCG_OLD)

         DLM=0
         CALL RAD_POT_WEIGHT( PP%R, 1, 0, POT)
         CALL RAD_PROJ(  POT(:,:,1)  , PP%R,1._q, DLM, PP%LMAX, PP%LPS, PP%WAE )
         ! transform them using Clebsch Gordan coefficients and add to CHF
         CALL TRANS_DLM( CHF(:,:,1), DLM , PP )

         ! add those contributions to PP%DION (PP strength parameters)
         CALL ADD_CDIJ(CHF(:,:,1), PP)

     ! recalculate potential again (RAD_POT_WEIGHT has multiplied it by a weight)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOAE, PP%POTAE_XCUPDATED,  POT, DOUBLEAE_OLD, EXCG_OLD)
         DO K=1,PP%R%NMAX
            PP%POTAE_XCUPDATED(K)=-POT(K,1,1)/SCALE
         ENDDO
#endif

         IF (USEFOCK_ONECENTER()) THEN
            ! exact exchange energy for one-center AE wave functions
            CALL SETUP_PAWFOCK_AE(NT, PP)
            CALL CALC_PAWFOCK(NT, PP, CRHODE, CHF, EX_HF)

            ! WRITE(*,*) 'all electron contribution HF',EX_HF
            ! valence-valence non-local exchange for atom
            DEATOM=DEATOM-EX_HF

            ! free the one-center terms so that AE-PS is calculated
            ! upon calling CALC_PAWFOCK
            IF (.NOT. USEFOCK_AE_ONECENTER()) CALL RELEASE_PAWFOCK

            ! difference between one center AE and PS Fock energies
            !  E_x(AE)-E_x(PS) = - DOUBLEC_HF
            
            CALL CALC_PAWFOCK(NT, PP, CRHODE, CHF, DOUBLEC_HF)

            DOUBLEAE = DOUBLEAE- DOUBLEC_HF
            ! WRITE(*,*) 'doublec HF',DOUBLEC_HF
            CALL RELEASE_PAWFOCK

#ifndef DFT_core_valence
#ifndef update_potential
            IF (.NOT. MIMIC_US) THEN
#endif
               ! for convenience the core-valence exchange is added directly to PP%QATOM
               ! this however is only allowed if the one center terms
               ! are added to CDIJ (or if the update_potential switch is used)
               ! the main reason for this is that the DFT core-valence interaction
               ! needs to be subtracted which is done by setting the reference AE
               ! potential to V_xc^DFT(rho_val + rho_core) and going through SET_DD_PAW
               EX_HF=0

               !core-valence HF
               ! AEXX_OLD=AEXX
               ! HFSCREEN_OLD=HFSCREEN
               ! AEXX=1.0
               ! HFSCREEN=0.0
               !end core-valence HF

               CALL SET_PAW_CORE_FOCK(PP, CRHODE(:,:,1), EX_HF)

               !core-valence HF
               ! AEXX=AEXX_OLD
               ! HFSCREEN=HFSCREEN_OLD
               ! PP%RHOAE=0
               !end core-valence HF

               ! valence-core non-local exchange for atom
                IF (.NOT. MIMIC_US) DEATOM=DEATOM+EX_HF
#ifndef update_potential
            ENDIF
#endif
#endif
         ENDIF
!-----------------------------------------------------------------------
! update the local one centre pseudo potential
!-----------------------------------------------------------------------
         ! store the PS potential value at PP%R%NMAX
         PP%VPSRMAX=-PP%POTPS(PP%R%NMAX)+PP%POTPSC(PP%R%NMAX)

         POTOLD=0
         POTOLD(1:SIZE(PP%POTPS))=PP%POTPS
         PP%POTPS=0
         RHO=0
         CALL RAD_CHARGE( RHO(:,:,1), PP%R, RHOLM, PP%LMAX, PP%LPS, PP%WPS )
         CALL RAD_AUG_CHARGE(  RHO(:,:,1), PP%R, RHOLM, PP%LMAX, PP%LPS,  &
              LYMAX, PP%AUG, PP%QPAW )

         ! get double counting corrections
         IF (LUSE_THOMAS_FERMI) CALL PUSH_XC_TYPE(LEXCH, LDAX, ALDAC, AGGAX, AGGAC, 0.0_q)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS, EXCG)
! test
!        CALL RAD_POT_METAGGA( PP%R, ISPIN, 1,  &
!                RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS, EXCG)
! test
         IF (LUSE_THOMAS_FERMI) CALL POP_XC_TYPE

         IF(.NOT. VERSION46) CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q, 0.0_q)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS_OLD, EXCG_OLD)
! test
!        CALL RAD_POT_METAGGA( PP%R, ISPIN, 1,  &
!                RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS_OLD, EXCG_OLD)
! test
         IF(.NOT. VERSION46) CALL POP_XC_TYPE

    ! double counting correction for PAW one centre pseudo terms
    ! as above \int dr (e_xc_new(r) - v_xc_old(r)) rho_PS(r)
         VX=EXCG_OLD-DOUBLEPS_OLD
         DOUBLEPS=EXCG-VX

#ifdef update_potential
    ! at this point POT contains the old valence only one-electron potential
    ! add the core potential (PP%POTPSC-PP%POTPS, PP%POTPS is zero here)
         POT(1:SIZE(PP%POTPS),1,1)=POT(1:SIZE(PP%POTPS),1,1)+(PP%POTPSC-PP%POTPS)*SCALE
         POTOLD(1:SIZE(PP%POTPS))=POT(1:SIZE(PP%POTPS),1,1)

         ! calculate one center contributions and store them in CHF
         DLM=0
         CALL RAD_POT_WEIGHT( PP%R, 1, 0, POT)
         CALL RAD_PROJ(  POT(:,:,1)  , PP%R,-1._q, DLM, PP%LMAX, PP%LPS, PP%WPS )
         CALL RAD_AUG_PROJ( POT(:,:,1), PP%R, DLM, PP%LMAX, PP%LPS, &
                  0, PP%AUG, PP%QPAW )

         ! transform them using Clebsch Gordan coefficients and store in CHF
         CHF=0
         CALL TRANS_DLM( CHF(:,:,1), DLM , PP )
         ! add those contributions to PP%DION (PP strength parameters)
         CHF=-CHF

    ! get new one center potential for atom (valence only)
    ! this might be different from above if the exchange correlation potential 
    ! or the way the augmentation is done has changed
    
    ! if required do more accurate augmentation of charge density using
    ! accurate charge restoration
         IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
            CALL RAD_AUG_CHARGE_FOCK(  RHO(:,:,1), PP%R, RHOLM, PP%LMAX, PP%LPS,  &
                    LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK)
         ENDIF


! redefine local ionic pseudopotential
! ) by adding the valence contribution for the conventional standard augmentation scheme
!   and the corresponding xc functional
! ) and subtracting the current valence contribution for the PP xc functional
         IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
            ! actually it is an open matter what to chose as local reference potential
            ! e.g.  the new functional or the old
            ! for OEP methods I lean towards PP generation potential
            ! for other cases towards new xc type
            CALL PUSH_XC_TYPE(PP%LEXCH,1.0_q,1.0_q,1.0_q,1.0_q, 0.0_q)
            CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS_OLD, EXCG_OLD)
            CALL POP_XC_TYPE
            POT(1:SIZE(PP%POTPS),1,1)=POT(1:SIZE(PP%POTPS),1,1)+(PP%POTPSC-PP%POTPS)*SCALE

            POTDIFF(1:SIZE(PP%POTPS))=(POT(1:SIZE(PP%POTPS),1,1)-POTOLD(1:SIZE(PP%POTPS)))/SCALE

            ! redefine core pseudo potential on plane wave grid
            CALL FOURPOT_TO_Q( PP%RDEP, POTDIFF, PP%PSP(:,2), SIZE(PP%PSP,1), PP%PSGMAX/ SIZE(PP%PSP,1), PP%R, IU6)
            PP%PSCORE=PP%PSP(1,2)
            CALL SPLCOF(PP%PSP(1,1), SIZE(PP%PSP,1), SIZE(PP%PSP,1), 0._q)

            ! redefine core pseudo potential on radial grid
            PP%POTPSC=PP%POTPSC-POTDIFF

            ! sanity test compare fourier transfrom of new PP%PSP with new POTPSC
            ! WRITE(78,'(2E16.7)') (PP%R%R(K),PP%POTPSC(K),K=1,PP%R%NMAX)
            ! WRITE(78,*)
            ! CALL POTTORHO(  PP%ZVALF, SIZE(PP%PSP,1), PP%PSP(:,2), PP%PSGMAX/ SIZE(PP%PSP,1), &
            !      .TRUE. , PP%R%NMAX, PP%R%R ,  PP%POTPSC )
            ! WRITE(78,'(2E16.7)') (PP%R%R(K),PP%POTPSC(K),K=1,PP%R%NMAX)

            ! recalculate the total local potential for the atom on the grid (core+ valence)
            ! should be identical to original POTLOK
         ENDIF

         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS_OLD, EXCG_OLD)
         POT(1:SIZE(PP%POTPS),1,1)=POT(1:SIZE(PP%POTPS),1,1)+(PP%POTPSC-PP%POTPS)*SCALE

         DLM=0
         CALL RAD_POT_WEIGHT( PP%R, 1, 0, POT)
         CALL RAD_PROJ(  POT(:,:,1)  , PP%R,-1._q, DLM, PP%LMAX, PP%LPS, PP%WPS )
         CALL RAD_AUG_PROJ( POT(:,:,1), PP%R, DLM, PP%LMAX, PP%LPS, &
                  0, PP%AUG, PP%QPAW )

         IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
            CALL RAD_AUG_PROJ_FOCK( POT(:,:,1), PP%R, DLM, PP%LMAX, PP%LPS, &
                 0, PP%AUG_FOCK, PP%QPAW_FOCK )
         ENDIF
         ! transform them using Clebsch Gordan coefficients and add to CDIJ
         CALL TRANS_DLM( CHF(:,:,1), DLM , PP )

         ! add those contributions to PP%DION (PP strength parameters)
         CALL ADD_CDIJ(CHF(:,:,1), PP)

     ! recalculate reference potential again (to get rid of weights)
         CALL RAD_POT( PP%R, ISPIN, 1, 1, .FALSE.,  &
                 RHO, PP%RHOPS, PP%POTPS, POT, DOUBLEPS_OLD, EXCG_OLD)
#endif
    ! now update the local potential to the new potential
         DO K=1,PP%R%NMAX
           PP%POTPS(K)=-POT(K,1,1)/SCALE
           ! WRITE(77,'(I4,5F14.7)') K,PP%R%R(K)/AUTOA,RHO(K,1,1)*AUTOA*SCALE,PP%POTPS(K),PP%POTPS(K)-POTOLD(K)
         ENDDO

         DOUBLEC_PS= DOUBLEC_PS- DOUBLEPS*T_INFO%NITYP(NT)*T_INFO%VCA(NT)
         DOUBLEC_AE= DOUBLEC_AE+(DOUBLEAE-PP%DEXCCORE)*T_INFO%NITYP(NT)*T_INFO%VCA(NT)

         ! update energy of atom
         PP%EATOM=PP%EATOM-DEATOM                      ! atomic energy without convergence correction
         PP%EATOM_CORRECTED=PP%EATOM_CORRECTED-DEATOM  ! convergence corrected energy

         !core-valence HF
         ! IF (.NOT. USEFOCK_ONECENTER()) THEN
         !   AEXX=1.0
         !   HFSCREEN = 0.0
         !   CALL SET_PAW_CORE_FOCK(PP, CRHODE(:,:,1), EX_HF)
         !   AEXX=AEXX_OLD
         !   HFSCREEN=HFSCREEN_OLD
         !   PP%RHOAE=0      ! switch off DFT core/valence exchange and correlation
         ! ENDIF
         !end core-valence HF

         IF (T_INFO%DARWIN_R(NT) /=0) THEN
            WRITE(*,*) 'pseudopotential strength before Darwin'
            DO M=1,PP%LMAX
               WRITE(*,'(10F14.7)') PP%DION(:,M)
            ENDDO
            CALL RAD_POT_DARWIN( PP%R, 1, POT , T_INFO%DARWIN_V(NT), T_INFO%DARWIN_R(NT))
            ! weight properly for simpson integration
            CALL RAD_POT_WEIGHT( PP%R, 1, 1, POT)
            ! extract DARWIN potential contribution into array RHOLM
            RHOLM=0
            CALL RAD_PROJ(  POT(:,:,1)  , PP%R,1._q, RHOLM, PP%LMAX, PP%LPS, PP%WAE )
            ! transform them using Clebsch Gordan coefficients and add to CDIJ
            CHF=0
            CALL TRANS_DLM( CHF(:,:,1), RHOLM , PP )
            ! add those contributions to PP%DION (PP strength parameters)
            CALL ADD_CDIJ(CHF(:,:,1), PP)
            WRITE(*,*) 'pseudopotential strength after Darwin'
            DO M=1,PP%LMAX
               WRITE(*,'(10F14.7)') PP%DION(:,M)
            ENDDO
         ENDIF

!        CALL SET_PAW_METAGGA(PP)

      ENDDO

      EALLAT=0
      DO NT=1,T_INFO%NTYP 
         EALLAT=EALLAT+P(NT)%EATOM_CORRECTED*T_INFO%NITYP(NT)*T_INFO%VCA(NT)
      ENDDO

      
  !  store double counting corrections for the atomic case
  !  applied only for MIMIC_US
      DOUBLEC_PS_ATOM=DOUBLEC_PS 
      DOUBLEC_AE_ATOM=DOUBLEC_AE

      DEALLOCATE( POT, RHO , POTOLD, POTDIFF )

      RETURN
    END SUBROUTINE SET_PAW_ATOM_POT

    SUBROUTINE SET_PAW_ATOM_POT_RHF(P,T_INFO,LOVERL,EALLAT,IO)

      USE prec
      USE base
      USE poscar
      USE rhfatm
      IMPLICIT NONE
      TYPE(type_info) T_INFO
      TYPE(potcar), TARGET :: P(T_INFO%NTYP)
      TYPE(in_struct) IO
      REAL(q) EALLAT
      LOGICAL LOVERL
      ! local variables
      TYPE (potcar), POINTER:: PP
      INTEGER NT
      REAL(q) EATOM_OLD,DEATOM

      IF (.NOT.LOVERL) RETURN

      CALL RHFATM_WRITER(IO)

      DO NT=1,T_INFO%NTYP
         PP=> P(NT)

         EATOM_OLD=PP%EATOM

         IF (.NOT.ASSOCIATED(P(NT)%QPAW )) CYCLE

         CALL RHFATM_SET_PP(PP,IO)

         DEATOM=PP%EATOM-EATOM_OLD

         EALLAT=EALLAT+T_INFO%NITYP(NT)*DEATOM
      ENDDO

      RETURN
    END SUBROUTINE SET_PAW_ATOM_POT_RHF

!*******************************************************************
!
!  SET_PAW_CORE_FOCK
! calculates the core-valence exchange contribution
! for Hartree-Fock
!
! sum_c int dr' phi_c(r') phi_v(r') int dr phi_c(r) phi_v(r)/(r-r')
!
! this is boundary condition independent since the
! valence and core are orthogonal int dr phi_c(r) phi_v(r)=0. 
! The Coloumb potential 
!  V(r') = int dr phi_c(r) phi_v(r) / (r-r') 
! is therefore boundary condition independent!
!
!*******************************************************************

    SUBROUTINE SET_PAW_CORE_FOCK(PP, CRHODE, HF_ENERGY)
! variables required to store core wavefunctions
      USE pseudo
      USE pawfock
      USE prec
      USE cl
      IMPLICIT NONE

      TYPE (potcar),POINTER:: PP
      OVERLAP              :: CRHODE(:,:)   ! valence occupancy
      REAL(q)              :: HF_ENERGY     ! valence-core interaction energy
    !local
      INTEGER CHANC,CHANNELS
      REAL(q), ALLOCATABLE :: W(:,:), A(:,:),B(:,:), EIG(:)
      INTEGER, ALLOCATABLE :: N(:), LC(:)
      REAL(q), ALLOCATABLE :: S(:,:,:,:,:)
      REAL(q) :: NORM
      OVERLAP, ALLOCATABLE :: COCC(:,:),DFOCK(:,:),DHARTREE(:,:)
      INTEGER :: CH1,CH2, LYMAX, LMMAX, RNMAX, M, LM1, LM2
      INTEGER, EXTERNAL :: MAXL1

      ! number of grid points
      RNMAX =PP%R%NMAX

      ! determine the number of core channels
      CALL CL_INIT_CORE_CONF(PP,CHANC)
      CHANNELS=CHANC+PP%LMAX
      ALLOCATE( W(RNMAX,CHANNELS),A(RNMAX,CHANNELS), B(RNMAX,CHANNELS), N(CHANNELS), LC(CHANNELS), EIG(CHANNELS))

      CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
           W, N, LC, EIG, A_=A, B_=B)

      DO CH1=1,PP%LMAX
         ! copy valence partial waves to A
         A(:,CHANC+CH1)=PP%WAE(:,CH1)
         LC(CHANC+CH1) =PP%LPS(CH1)
      END DO
#ifdef debug
      ! test orthogonality to core wave functions
      DO CH1=1,CHANC+PP%LMAX
      DO CH2=1,CH1
         IF( LC(CH2)==PP%LPS(CH1)) THEN
            W(:,1)=A(:,CH1)*A(:,CH2)
            CALL SIMPI(PP%R, W(:,1) , NORM)
            WRITE(*,*) 'norm', CH2, CH1, LC(CH2), NORM
         ENDIF
      ENDDO
      ENDDO
      STOP
#endif
      LYMAX =3    ! no g orbitals at this point

      ALLOCATE(S(CHANNELS, CHANNELS, CHANNELS, CHANNELS, 0:LYMAX))

      S=0
      CALL COLOUMB_4TERM( A, LC, PP%R, CHANNELS, S, 8)
      DEALLOCATE(A, B, W)

      LMMAX=0
      DO CH1=1,CHANNELS
         LMMAX=LMMAX+(LC(CH1)*2+1)
      ENDDO
      ALLOCATE(COCC(LMMAX,LMMAX), DHARTREE(LMMAX,LMMAX), DFOCK(LMMAX,LMMAX))
      !
      ! set diagonal components of the occupancy matrix (core only)
      !
      COCC=0
      LMMAX=0
      DO CH1=1,CHANC
         DO M=1,LC(CH1)*2+1
            COCC(LMMAX+M,LMMAX+M)=1
         ENDDO
         LMMAX=LMMAX+(LC(CH1)*2+1)
      ENDDO

      ! partially include valence states in the setup
      ! this does not work properly but might be required if 
      ! NBANDSGW_LOW is supported
      !
      ! set atomic occupancies
      ! CALL SET_CRHODE_ATOM(COCC(LMMAX+1:,LMMAX+1:), PP)
      ! spin
      ! COCC(LMMAX+1:,LMMAX+1:)=COCC(LMMAX+1:,LMMAX+1:)/2 
      ! selectively clear states
      ! COCC(8:10,8:10)=0
      ! COCC(12:,12:)=0


      CALL CALC_DHARTREE(S,COCC,CHANNELS,LC,DHARTREE,DFOCK)

      !DO CH1=1,SIZE(COCC,1)
      !   WRITE(*,'(40F7.2)') COCC(:,CH1)
      !ENDDO
      !DO CH1=1,SIZE(COCC,1)
      !   WRITE(*,'(40F7.2)') DFOCK(:,CH1)
      !ENDDO

      CALL ADD_CDIJ(DFOCK(LMMAX+1:,LMMAX+1:),PP)

      HF_ENERGY=0
      DO LM1=1,PP%LMMAX
         DO LM2=1,PP%LMMAX
#ifdef realmode
            HF_ENERGY=HF_ENERGY+ &
                 DFOCK(LMMAX+LM1,LMMAX+LM2)*CRHODE(LM1,LM2)
#else
            HF_ENERGY=HF_ENERGY+ &
                 DFOCK(LMMAX+LM1,LMMAX+LM2)*CONJG(CRHODE(LM1,LM2))
#endif
         ENDDO
      ENDDO
      ! double the energy since only up component is stored in CRHODE
      HF_ENERGY=HF_ENERGY*2

      DEALLOCATE(COCC,DHARTREE,DFOCK)
      DEALLOCATE(S)
      DEALLOCATE( N, LC, EIG)
      CALL CL_CLEAR_CORE_CONF()
    END SUBROUTINE SET_PAW_CORE_FOCK




!*******************************************************************
!
! this is the central routine of the PAW method
! it calculates the on site terms and adds them to D(I,J)
! at the same time double counting corrections are calculated
!
!
!*******************************************************************

    SUBROUTINE SET_DD_PAW(WDES, P , T_INFO, LOVERL, &
         ISPIN, LMDIM, CDIJ, RHOLM_STORE, CRHODE, &
         E, LMETA, LASPH, LCOREL )
      USE pseudo
      USE asa
      USE poscar
      USE wave
      USE constant
      USE radial
      USE base
      USE relativistic
      USE LDAPLUSU_MODULE
      USE cl
      USE pawfock_inter
      USE setexm
      USE egrad
      USE meta
      USE setxcmeta
      USE hyperfine
      IMPLICIT NONE

      TYPE (type_info) T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      TYPE (wavedes)  WDES
      TYPE (energy)   E
      INTEGER LMDIM, ISPIN
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      REAL(q)  RHOLM_STORE(:,:)
      LOGICAL  LOVERL
      LOGICAL  LMETA      ! calculate meta GGA contribution
      LOGICAL  LASPH      ! calculate aspherical corrections to potential
      LOGICAL  LCOREL     ! calculate accurate core level shifts
    ! local variables
      TYPE (potcar),POINTER:: PP
      INTEGER NT,LYMAX,NI,NDIM,LMMAX,NIP,ISP,IBASE,IADD,ISIZE,K,ITMP,NCDIJ,LMAX
      INTEGER, EXTERNAL :: MAXL_AUG,MAXL1
      LOGICAL, EXTERNAL :: USEFOCK_CONTRIBUTION, USEFOCK_AE_ONECENTER
! automatic arrays crash on IBM and SGI (thank's to Lucian Anton NAG, Zhengji Zhao SGI)
!      REAL(q) DDLM(LMDIM*LMDIM),RHOLM(LMDIM*LMDIM),RHOLM_(LMDIM*LMDIM,WDES%NCDIJ)
!      OVERLAP CTMP(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),CSO(LMDIM,LMDIM,WDES%NCDIJ), &
!              CHF(LMDIM,LMDIM,WDES%NCDIJ)
!      OVERLAP COCC(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),COCC_IM(LMDIM,LMDIM)
      REAL(q),ALLOCATABLE :: DDLM(:),RHOLM(:),RHOLM_(:,:)
      OVERLAP,ALLOCATABLE :: CTMP(:,:,:),CSO(:,:,:),CHF(:,:,:)
      OVERLAP,ALLOCATABLE :: COCC(:,:,:),COCC_IM(:,:)
      OVERLAP,ALLOCATABLE :: CMETAGGA(:,:,:)
      REAL(q),ALLOCATABLE :: POT(:,:,:), RHO(:,:,:), POTAE(:,:,:), RHOAE(:,:,:)
#ifdef noAugXCmeta
      REAL(q),ALLOCATABLE :: RHOPS_STORE(:,:,:),POTH(:,:,:)
#else
#define RHOPS_STORE RHOCOL
#define POTH POT
#endif
      REAL(q),ALLOCATABLE :: RHOCOL(:,:,:), KINDENSCOL(:,:,:),KINDENSWCOL(:,:,:)
      ! core level shifts
      REAL(q),ALLOCATABLE :: DRHOCORE(:)
      REAL(q) :: DOUBLEC_AE,DOUBLEC_PS
      REAL(q) :: DOUBLEPS,DOUBLEAE
      REAL(q) :: EXCM,EXCG,EXCGA
      REAL(q) :: DOUBLEC_LDAU, DOUBLEC_HF
      ! euler angles of the global spin quantisation axis
      REAL(q) :: ALPHA,BETA
      INTEGER :: LMAX_TAU, LMMAX_TAU
      REAL(q),ALLOCATABLE :: TAUAE(:,:,:),TAUPS(:,:,:),MUAE(:,:,:),MUPS(:,:,:)
    ! kinetic energy density (true and Weizsaecker)
      REAL(q),ALLOCATABLE :: KINDENSAE(:,:),KINDENSPS(:,:)
      REAL(q),ALLOCATABLE :: WKDAE(:,:),WKDPS(:,:)
      REAL(q),ALLOCATABLE :: RHOUPD(:,:,:),RHOLMUPD(:,:)
      REAL(q),POINTER :: NULPOINTER(:)
      REAL(q) :: SPI2,TMP
      INTEGER II,RNMAX, RNMAX_CL
      INTEGER, EXTERNAL :: ONE_CENTER_NMAX_FOCKAE
      REAL(q) :: EPAWPSM,EPAWPSG,EPAWAEM,EPAWAEG,EPAWCORE,EPAWCOREM
! variables required to store core wavefunctions
      INTEGER MAXNL
      REAL(q), ALLOCATABLE :: W(:,:), EIG(:)
      INTEGER, ALLOCATABLE :: N(:), LC(:)
! needed to distribute over COMM_KINTER
      INTEGER IDONE
      LOGICAL LSKIP


      PROFILING_START('set_dd_paw')

!=======================================================================
! quick return and allocation of work space
!=======================================================================

      E%PAWPSM=0; E%PAWAEM=0; E%PAWCORE=0
      E%PAWPSG=0; E%PAWAEG=0; E%PAWCOREM=0

      DOUBLEC_AE=0
      DOUBLEC_PS=0

      EPAWPSM=0; EPAWAEM=0; EPAWCORE=0
      EPAWPSG=0; EPAWAEG=0; EPAWCOREM=0

      CL_SHIFT= 0

      NULLIFY(NULPOINTER)

      IF (.NOT.LOVERL) THEN
         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

! mimic US-PP just set the double counting corrections correctly
      IF (MIMIC_US) THEN
         DOUBLEC_AE=DOUBLEC_AE_ATOM
         DOUBLEC_PS=DOUBLEC_PS_ATOM

         E%PAWAE=DOUBLEC_AE
         E%PAWPS=DOUBLEC_PS

         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

      SPI2= 2*SQRT(PI)

      LYMAX =MAXL_AUG(T_INFO%NTYP,P)

      NDIM=0
      DO NT=1,T_INFO%NTYP
         IF (ASSOCIATED(P(NT)%QPAW)) THEN
            NDIM=MAX(NDIM, P(NT)%R%NMAX)
         END IF
      ENDDO

      IF (NDIM == 0) THEN
         PROFILING_STOP('set_dd_paw')
         RETURN
      ENDIF

      IF (LUSE_THOMAS_FERMI) CALL PUSH_XC_TYPE(LEXCH, LDAX, ALDAC, AGGAX, AGGAC, 0.0_q)

      ALLOCATE(DDLM(LMDIM*LMDIM),RHOLM(LMDIM*LMDIM))
      ALLOCATE(RHOLM_(LMDIM*LMDIM,WDES%NCDIJ))
      ALLOCATE(CTMP(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),CSO(LMDIM,LMDIM,WDES%NCDIJ),CHF(LMDIM,LMDIM,WDES%NCDIJ))
      ALLOCATE(COCC(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),COCC_IM(LMDIM,LMDIM))

      LMMAX=(LYMAX+1)**2
      NCDIJ = WDES%NCDIJ

      ALLOCATE ( POT( NDIM, LMMAX, NCDIJ ), RHO( NDIM, LMMAX, NCDIJ ), &
     &   POTAE( NDIM, LMMAX, NCDIJ ), RHOAE( NDIM, LMMAX, NCDIJ), DRHOCORE(NDIM))
      ALLOCATE (RHOCOL( NDIM, LMMAX, NCDIJ ))
#ifdef noAugXCmeta
      ALLOCATE(RHOPS_STORE(NDIM,LMMAX,NCDIJ),POTH(NDIM,LMMAX,NCDIJ))
#endif
! allocate kinetic energy density arrays for metagga
      ALLOCATE (KINDENSAE(NDIM,NCDIJ),KINDENSPS(NDIM,NCDIJ)) 
      ALLOCATE (WKDAE(NDIM,NCDIJ),WKDPS(NDIM,NCDIJ))
      ALLOCATE (RHOUPD(NDIM,1,NCDIJ),RHOLMUPD(LMDIM*LMDIM,NCDIJ))
      ALLOCATE (CMETAGGA(LMDIM,LMDIM,NCDIJ))

! for spin orbit coupling set the euler angles
      IF ( WDES%LSORBIT ) &
         CALL EULER(WDES%SAXIS, ALPHA, BETA)
!=======================================================================
! cycle all ions and add corrections to pseudopotential strength CDIJ
!=======================================================================
      IBASE=1; IDONE=0


      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         NT=T_INFO%ITYP(NI)

         LSKIP=.FALSE.
#ifdef MPI
         ! DO_LOCAL represents a distribution of the work on the
         ! one-center terms over the procs in COMM_INB and COMM_INTER (=COMM_KIN).
         ! The following allows an additional round-robin distribution over COMM_KINTER as well.
         IF (DO_LOCAL(NI)) THEN
            IDONE=IDONE+1; LSKIP=(MOD(IDONE,WDES%COMM_KINTER%NCPU)+1/=WDES%COMM_KINTER%NODE_ME)
         ENDIF
#endif
         ! if this element is not treated locally CYCLE
         IF (.NOT. DO_LOCAL(NI).OR.LSKIP) THEN
            ! for PAW, set CDIJ to zero if it resides on local node
            ! and if the element is not treated locally
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
               IF (WDES%COMM_INTER%NODE_ME*WDES%COMM_KINTER%NODE_ME /=1 .AND. NIP /=0 ) THEN
                  DO ISP=1,NCDIJ
                     CDIJ(:,:,NIP,ISP)=0
                  ENDDO
               ENDIF
#endif
            ENDIF
            ! in case we skip work on this ion, we still have to advance IBASE
            IF (LSKIP) THEN
               PP=> PP_POINTER(P, NI, NT)
               IBASE=IBASE+COUNT_RHO_PAW_ELEMENTS(PP)
            ENDIF
            CYCLE ion
         ENDIF

         ! The CMBJ parameter may depend on the atomic type
         ! if so, it is set here to the right entry in CMBJ_TYP
         CALL SET_CMBJ_RAD(NT)

         PP=> PP_POINTER(P, NI, NT)
         CALL SET_RSGF_TYPE(NT)
!        CALL SET_RSGF_SIMPLE(PP)

         LYMAX =MAXL1(PP)*2
         RNMAX =PP%R%NMAX

         LMAX_TAU=LYMAX+2; LMMAX_TAU=(LMAX_TAU+1)**2
         ALLOCATE (TAUAE(NDIM,LMMAX_TAU,NCDIJ),TAUPS(NDIM,LMMAX_TAU,NCDIJ), &
        &   MUAE(NDIM,LMMAX_TAU,NCDIJ),MUPS(NDIM,LMMAX_TAU,NCDIJ),KINDENSCOL(NDIM,LMMAX_TAU,NCDIJ))
  !-----------------------------------------------------------------------
  ! first set RHOLM (i.e. the on site occupancy matrix)
  ! and then the lm dependent charge densities RHO and RHOAE
  ! (excluding augmentation charges yet)
  !-----------------------------------------------------------------------
         RHOLM_=0
  !      WRITE(*,'("RHOMIX",6F10.6)') RHOLM_STORE
         COCC=0
         ISIZE=UBOUND(RHOLM_STORE,1)

         DO ISP=1,NCDIJ
    ! retrieve the one center on site charge densities to RHOLM_
            IF ( LMAX_MIX < PP%LMAX_CALC) &
                 CALL TRANS_RHOLM( CRHODE(:,:,NIP,ISP), RHOLM_(:,ISP), PP )
    ! retrieve mixed elements from RHOLM_STORE and overwrite them in RHOLM_
            CALL RETRIEVE_RHOLM( RHOLM_(:,ISP), RHOLM_STORE(IBASE:,ISP), &
                           METRIC(IBASE:), IADD, PP, .FALSE.,  ITMP)

    ! calculate the total radial angular decomposed charge distributions
            LMMAX=(LYMAX+1)**2
            RHOAE(:,:,ISP)=0; RHO(:,:,ISP)=0
            CALL RAD_CHARGE( RHOAE(:,:,ISP), PP%R,RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WAE )
            CALL RAD_CHARGE( RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS, PP%WPS )
#ifdef noAugXCmeta
    ! We'll need the pseudo charge density without augmentation in the metagga
            RHOPS_STORE(:,:,ISP)=RHO(:,:,ISP)
#endif
    ! for LDA+U or Hartree fock we need the mixed occupancy matrix COCC
            IF (USELDApU() .OR. LDO_METAGGA() .OR. & 
           &   USEFOCK_CONTRIBUTION() .OR. USEFOCK_AE_ONECENTER() .OR. &
           &   LHYPERFINE()) THEN
               ! calculate the occupancy matrix COCC from RHOLM_(:,ISP)
               CALL TRANS_RHOLMI( COCC(:,:,ISP), RHOLM_(:,ISP), PP )
#ifndef realmode
               ! retrieve imaginary part from CRHODE and store in RHOLM
               CALL TRANS_RHOLM_IM( CRHODE(:,:,NIP,ISP), RHOLM, PP )
               ! orverwrite by elements passed through mixer
#ifdef RHOLM_complex
               CALL RETRIEVE_RHOLM( RHOLM, RHOLM_STORE(IBASE+ISIZE/2:,ISP), &
                                 METRIC(IBASE+ISIZE/2:), IADD, PP, .FALSE.,  ITMP)
#endif
               COCC_IM=0
               CALL TRANS_RHOLMI_IM( COCC_IM, RHOLM, PP )
               ! join imaginary and real parts
               COCC(:,:,ISP)=COCC(:,:,ISP)+(0._q,1._q)*COCC_IM(:,:)
!              WRITE(*,*) 'spin',ISP
!              CALL DUMP_DLLMM_IM(COCC(:,:,ISP),PP)
!              CALL DUMP_DLLMM_IM(CRHODE(:,:,NIP,ISP),PP)
#endif
            ENDIF
!              WRITE(*,*) 'spin',ISP
!              CALL DUMP_DLLMM("cocc",COCC(:,:,ISP),PP)
!              CALL DUMP_DLLMM("crhode",CRHODE(:,:,NIP,ISP),PP)
         ENDDO

         ! bring COCC to (up,down)
         IF (WDES%ISPIN==2) CALL OCC_FLIP2(COCC,LMDIM)

         CALL RAD_KINEDEN(PP,PP%WAE,LMAX_TAU,NCDIJ,COCC,TAUAE)
         CALL RAD_KINEDEN(PP,PP%WPS,LMAX_TAU,NCDIJ,COCC,TAUPS)

         ! bring COCC to spinor representation
         IF (WDES%LNONCOLLINEAR) CALL OCC_FLIP4(COCC,LMDIM)

  !-----------------------------------------------------------------------
  ! calculation and output of radial kinetic energy density
  ! (remember augmentation charges are still excluded)
  !-----------------------------------------------------------------------
         IF ( LMETA ) THEN
            ! RHOLM_ is the (total, magnetization) representation
            DO ISP=1,NCDIJ
               RHOLMUPD(:,ISP)=RHOLM_(:,ISP)
            ENDDO

            KINDENSAE=0; WKDAE=0
            KINDENSPS=0; WKDPS=0
            IF (ISPIN==2) THEN
               ! go to (spin up, down) representation
               CALL FLIP_RAD(RHOLMUPD,RHOLMUPD,LMDIM*LMDIM)
               CALL FLIP_RAD(RHOAE(:,1,1:2),RHOUPD(:,1,1:2),RNMAX)
            ELSE
               ! for ISPIN =1 and ISPIN = 4 stay in total, magnetization representation
               RHOUPD(:,1,1)=RHOAE(:,1,1)
            ENDIF

            DO ISP=1,ISPIN
               IF (ISPIN ==4 .AND. ISP==1) THEN 
               CALL RAD_KINETIC_EDENS(KINDENSAE(:,ISP),WKDAE(:,ISP),PP%R,RHOLMUPD(:,ISP), &
                    PP%LMAX,PP%LPS, PP%WAE, PP%RHOAE, PP%TAUAE, RHOUPD(:,:,ISP), 1)
               ELSE
               CALL RAD_KINETIC_EDENS(KINDENSAE(:,ISP),WKDAE(:,ISP),PP%R,RHOLMUPD(:,ISP), &
                    PP%LMAX,PP%LPS, PP%WAE, PP%RHOAE, PP%TAUAE, RHOUPD(:,:,ISP), ISPIN)
               ENDIF
            ENDDO

            IF (ISPIN==2) THEN
               CALL FLIP_RAD(RHO(:,1,1:2),RHOUPD(:,1,1:2),RNMAX)
            ELSE
               RHOUPD(:,1,1)=RHO(:,1,1)
            ENDIF
            DO ISP=1,ISPIN
               CALL RAD_KINETIC_EDENS(KINDENSPS(:,ISP),WKDPS(:,ISP),PP%R,RHOLMUPD(:,ISP), &
                    PP%LMAX,PP%LPS, PP%WPS, NULPOINTER, NULPOINTER, RHOUPD(:,:,ISP),ISPIN)
            ENDDO
         ENDIF
  !-----------------------------------------------------------------------
  ! add augmentation charges now
  !-----------------------------------------------------------------------
         DO ISP=1,NCDIJ
            CALL RAD_AUG_CHARGE(  RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS,  &
                  LYMAX, PP%AUG, PP%QPAW )
            IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
               CALL RAD_AUG_CHARGE_FOCK(  RHO(:,:,ISP), PP%R, RHOLM_(:,ISP), PP%LMAX, PP%LPS, &
                    LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK)
            ENDIF
            CALL RAD_INT( PP%R,  LYMAX, RHO(:,:,ISP), RHOAE(:,:,ISP) )
         ENDDO
         IBASE=IBASE+IADD
  !-----------------------------------------------------------------------
  ! now finish the meta GGA stuff
  !-----------------------------------------------------------------------
            IF ( LMETA ) THEN
            ! calculate E(xc) for meta-GGA
!              CALL RAD_META_GGA(PP%R, ISPIN, LYMAX, PP%LMAX_CALC, &
!                   RHO, PP%RHOPS, POT,KINDENSPS,WKDPS,EXCM)
               CALL RAD_META_GGA_ASPH(PP%R, ISPIN, LYMAX, PP%LMAX_CALC, &
                    RHO, PP%RHOPS, POT,KINDENSPS,WKDPS,EXCM)
               E%PAWPSM=E%PAWPSM-EXCM
!              CALL RAD_META_GGA(PP%R, ISPIN, LYMAX, PP%LMAX_CALC, &
!                   RHOAE, PP%RHOAE, POTAE,KINDENSAE,WKDAE,EXCM)
               CALL RAD_META_GGA_ASPH(PP%R, ISPIN, LYMAX, PP%LMAX_CALC, &
                    RHOAE, PP%RHOAE, POTAE,KINDENSAE,WKDAE,EXCM)

               EPAWAEM=EPAWAEM+EXCM-PP%DEXCCOREM
               EPAWCOREM=EPAWCOREM+PP%DEXCCOREM
            ENDIF

         CALL EGRAD_EFG_RAD_HAR_ONLY(T_INFO,NI,PP,RHO,RHOAE)
  !-----------------------------------------------------------------------
  ! calculate the local radial potential
  ! mind in the non-collinear case the potential V(r) = d E(r) / d rho (r)
  ! and the potential vec mu(r) = d E(r) / d vec m (r) are stored in
  ! POT and POTAE (potentials need to be real), whereas 
  ! in the collinear case the spin up and down potentials
  ! are stored in POT and POTAE
  ! probably one should rewrite this in the collinear case
  !-----------------------------------------------------------------------
    ! initialise the spin orbit contributions to D_ij to 0
         CSO=0
    ! Hartree Fock contribution set to zero
         CHF=0

         IF ( WDES%LNONCOLLINEAR ) THEN
            ! bring KINDENSPS from spinor to 2 component spin up and spin down presentation
#ifndef noAugXCmeta            
            CALL RAD_MAG_DENSITY_KINDENS(RHO, RHOCOL, TAUPS, KINDENSCOL, LYMAX, LMAX_TAU, PP%R)
#else
            CALL RAD_MAG_DENSITY_KINDENS(RHO, RHOCOL, TAUPS, KINDENSCOL, LYMAX, LMAX_TAU, PP%R, RHOPS_STORE)
#endif
            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            IF (L_NO_LSDA()) RHOCOL(:,:,2:WDES%NCDIJ)=0
#ifdef noAugXCmeta
            IF (L_NO_LSDA()) RHOPS_STORE(:,:,2:WDES%NCDIJ)=0
#endif
            CALL RAD_POT( PP%R, 2, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOPS, PP%POTPS, POT, DOUBLEPS, EXCG)

            CALL SET_CMBJ_ONE_CENTER_FACT(-1._q)
            CALL RAD_POT_METAGGA( PP%R, 2, PP%LMAX_CALC, LMAX_TAU, LASPH,  &
               RHOCOL, RHOPS_STORE, PP%RHOPS, PP%POTPS, KINDENSCOL, POTH, POT, DOUBLEPS, EXCG, MUPS, PP%TAUPS)

            EPAWPSG=EPAWPSG-EXCG

            CALL RAD_MAG_DIRECTION( RHO, RHOCOL, POT, LYMAX, PP%R)
#ifdef noAugXCmeta
            CALL RAD_MAG_DIRECTION( RHO, RHOCOL, POTH, LYMAX, PP%R)
#endif
            CALL RAD_MAG_DIRECTION_KINDENS( RHO, TAUPS, LMAX_TAU, POT, MUPS, PP%R)

            ! bring KINDENSAE from spinor to 2 component spin up and spin down presentation
            CALL RAD_MAG_DENSITY_KINDENS(RHOAE, RHOCOL, TAUAE, KINDENSCOL, LYMAX, LMAX_TAU, PP%R)

            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            IF (L_NO_LSDA()) RHOCOL(:,:,2:WDES%NCDIJ)=0

            CALL APPLY_ONE_CENTER_AEXX()
            CALL RAD_POT( PP%R, 2, LYMAX, PP%LMAX_CALC, LASPH,  &
                 RHOCOL, PP%RHOAE, PP%POTAE_XCUPDATED,  POTAE, DOUBLEAE,EXCG)

            CALL SET_CMBJ_ONE_CENTER_FACT( 1._q)
            CALL RAD_POT_METAGGA( PP%R, 2, PP%LMAX_CALC, LMAX_TAU, LASPH, &
                 RHOCOL, RHOCOL, PP%RHOAE, PP%POTAE_XCUPDATED, KINDENSCOL, POTAE, POTAE, DOUBLEAE, EXCG, MUAE, PP%TAUAE)

            CALL RESTORE_ONE_CENTER_AEXX

            EPAWAEG=EPAWAEG+EXCG-PP%DEXCCORE
            EPAWCORE=EPAWCORE+PP%DEXCCORE

            CALL RAD_MAG_DIRECTION( RHOAE, RHOCOL, POTAE, LYMAX, PP%R)

            CALL RAD_MAG_DIRECTION_KINDENS( RHOAE, TAUAE, LMAX_TAU, POTAE, MUAE, PP%R)

            IF (WDES%LSORBIT) &
              CALL SPINORB_STRENGTH(POTAE(:,1,1), PP%RHOAE, PP%POTAE_XCUPDATED, PP%R, CSO, &
                PP%LMAX, PP%LPS ,PP%WAE, PP%ZCORE+PP%ZVALF_ORIG, THETA=BETA, PHI=ALPHA)

         ELSE
! collinear case
            DRHOCORE=0

            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            RHOCOL=RHO
            IF (L_NO_LSDA()) RHOCOL(:,:,2)=0
#ifdef noAugXCmeta
            IF (L_NO_LSDA()) RHOPS_STORE(:,:,2)=0
#endif
            ! cl shifts DRHOCORE is zero here, since for the pseudo terms
            ! we do not include the core electron in the exchange correlation term
            ! but only in the Hartree term
            CALL RAD_POT( PP%R, ISPIN, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOPS-DRHOCORE(1:RNMAX), PP%POTPS, POT, DOUBLEPS, EXCG)

            CALL SET_CMBJ_ONE_CENTER_FACT(-1._q)
            CALL RAD_POT_METAGGA( PP%R, ISPIN, PP%LMAX_CALC, LMAX_TAU, LASPH, &
               RHOCOL, RHOPS_STORE, PP%RHOPS-DRHOCORE(1:RNMAX), PP%POTPS, TAUPS, POTH, POT, DOUBLEPS, EXCG, MUPS, PP%TAUPS)

            CALL SET_CL_DRHOCORE_PS(DRHOCORE, NT, PP%R, PP%AUG)
            CALL ADD_CL_HARTREE_POT(DRHOCORE, NT, ISPIN, POT, PP%R)
            !CALL RAD_POT_HAR_ONLY( PP%R, ISPIN, LYMAX, PP%LMAX_CALC,RHOCOL,  POT, DOUBLEPS)

            EPAWPSG=EPAWPSG-EXCG

            DRHOCORE=0
            
            ! do LDA+U instead of LSDA+U (set magnetisation density to 0)
            RHOCOL=RHOAE
            IF (L_NO_LSDA()) RHOCOL(:,:,2)=0
 
            CALL SET_CL_DRHOCORE_AE(DRHOCORE, NT, PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG )

            CALL APPLY_ONE_CENTER_AEXX()

            CALL RAD_POT( PP%R, ISPIN, LYMAX, PP%LMAX_CALC, LASPH,   &
               RHOCOL, PP%RHOAE-DRHOCORE(1:RNMAX), PP%POTAE_XCUPDATED,  POTAE, DOUBLEAE,EXCG)

            CALL SET_CMBJ_ONE_CENTER_FACT( 1._q)
            CALL RAD_POT_METAGGA( PP%R, ISPIN, PP%LMAX_CALC, LMAX_TAU, LASPH, &
               RHOCOL, RHOCOL, PP%RHOAE-DRHOCORE(1:RNMAX), PP%POTAE_XCUPDATED, TAUAE, POTAE, POTAE, DOUBLEAE, EXCG, MUAE, PP%TAUAE)

            CALL RESTORE_ONE_CENTER_AEXX

            CALL ADD_CL_HARTREE_POT(DRHOCORE, NT, ISPIN, POTAE, PP%R)

            !CALL RAD_POT_HAR_ONLY( PP%R, ISPIN, LYMAX, PP%LMAX_CALC,RHOCOL,  POTAE, DOUBLEAE)

            EPAWAEG=EPAWAEG+EXCG-PP%DEXCCORE
            EPAWCORE=EPAWCORE+PP%DEXCCORE
         ENDIF

         DOUBLEC_PS= DOUBLEC_PS-DOUBLEPS*T_INFO%VCA(NT)
         ! the core-core exchange correlation energy is included
         ! up to this point in \int dr (e_xc(rho_c+rho_v) - v_xc(rho_c+rho_v) rho_v(r)
         ! subtract it now
         DOUBLEC_AE= DOUBLEC_AE+DOUBLEAE*T_INFO%VCA(NT)-PP%DEXCCORE*T_INFO%VCA(NT)

         CALL HYPERFINE_RAD(T_INFO,NI,PP,RHO,RHOAE,POTAE,COCC)
  !-----------------------------------------------------------------------
  ! calculate core level shift for averaged up and down potential
  ! or total potential (in non collinear case stored in POTAE(..,..,1))
  !-----------------------------------------------------------------------
         DO RNMAX_CL=1,RNMAX
           IF (PP%RDEP>0 .AND.  PP%R%R(RNMAX_CL)-PP%RDEP > -5E-3) EXIT
         ENDDO

         IF (.NOT. LCOREL) THEN
           CL_SHIFT(1,NI)=0
           IF (NCDIJ==2) THEN
              CALL RAD_CL_SHIFT( (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/SPI2/2, &
                      (POTAE(1:RNMAX,1,1)+POTAE(1:RNMAX,1,2))/SPI2/2, PP%R, CL_SHIFT(1,NI), PP%AUG(:,0))
           ELSE
              CALL RAD_CL_SHIFT( POT(1:RNMAX,1,1)/SPI2, POTAE(1:RNMAX,1,1)/SPI2, PP%R, CL_SHIFT(1,NI), PP%AUG(:,0))
           ENDIF
         ELSE

           CALL CL_INIT_CORE_CONF(PP,MAXNL)
           ALLOCATE( W(RNMAX,MAXNL), N(MAXNL), LC(MAXNL), EIG(MAXNL))

           ! first version for cl-shifts
           ! ---------------------------
           ! calculate core wavefunctions in PAW sphere for atomic reference potential

           CL_SHIFT(1:MAXNL,NI) =0
           CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
             W, N, LC, EIG)
           IF (MAXNL > SIZE(CL_SHIFT,DIM=1)) THEN
              WRITE(0,*) 'internal error: increase CL_MAXNL in main.F',MAXNL,SIZE(CL_SHIFT,DIM=1)
              STOP
           ENDIF

           ! now calculate the first order change caused by the current potential
           ! and subtract the pseudo contribution
           IF (NCDIJ==2) THEN
             CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/2/SPI2 , &
                   PP%R, CL_SHIFT(:,NI), &
                   PP%AUG(:,0),  MAXNL, W, EIG, (POTAE(:,1,1)+POTAE(:,1,2))/SPI2/2)
           ELSE
             CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ POT(1:RNMAX,1,1)/SPI2 , &
                   PP%R, CL_SHIFT(:,NI), &
                   PP%AUG(:,0),  MAXNL, W, EIG, POTAE(:,1,1)/SPI2)
           ENDIF
          !  WRITE(0,*) CL_SHIFT(1:MAXNL,NI) 

          ! CL_SHIFT(1:MAXNL,NI) =0
           ! version for cl-shifts that uses exact potential
           ! ------------------------------------------------
           ! solve radial Schroedinger equation for *current* potential
          ! IF (NCDIJ==2) THEN
          !    CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
          !    W, N, LC, EIG, (POTAE(1:RNMAX,1,1)+POTAE(1:RNMAX,1,2))/2 , NMAX= RNMAX_CL)

           ! subtract only the pseudo contribution
          !    CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ (POT(1:RNMAX,1,1)+POT(1:RNMAX,1,2))/2/SPI2 , &
          !         PP%R, CL_SHIFT(:,NI), &
          !         PP%AUG(:,0),  MAXNL, W, EIG)
          ! ELSE
          !    CALL SET_CORE_WF( PP%RHOAE, PP%POTAE , PP%R, PP%ZCORE, PP%ZVALF_ORIG , &
          !    W, N, LC, EIG, POTAE(1:RNMAX,1,1),  NMAX= RNMAX_CL)

           ! subtract only the pseudo contribution
          !    CALL RAD_CL_SHIFT_AE( (PP%POTPSC-PP%POTPS)+ POT(1:RNMAX,1,1)/SPI2 , &
          !         PP%R, CL_SHIFT(:,NI), &
          !         PP%AUG(:,0),  MAXNL, W, EIG)
          ! ENDIF
           ! WRITE(0,*)  CL_SHIFT(1:MAXNL,NI)
           DEALLOCATE(W, N, LC, EIG)
           CALL CL_CLEAR_CORE_CONF
         ENDIF

  !-----------------------------------------------------------------------
  ! calculate the PAW correction terms to the pseudopotential strength D
  ! I have defined the PAW contribution in a way that in the limit of
  ! atomic occupancies no contributions are added
  !-----------------------------------------------------------------------

         ! multiply potentials by simpson weights
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POTAE)
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POT)
#ifdef noAugXCmeta
         CALL RAD_POT_WEIGHT( PP%R, NCDIJ, LYMAX, POTH)
#endif
         CTMP=0
         DO ISP=1,NCDIJ
            DDLM=0
            CALL RAD_PROJ(  POTAE(:,:,ISP), PP%R, 1._q, DDLM, PP%LMAX, PP%LPS, PP%WAE )
            CALL RAD_PROJ(  POT(:,:,ISP)  , PP%R,-1._q, DDLM, PP%LMAX, PP%LPS, PP%WPS )
#ifndef noAugXCmeta
            CALL RAD_AUG_PROJ( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                  LYMAX, PP%AUG, PP%QPAW )
#else
            IF (LDO_METAGGA()) THEN
               CALL RAD_AUG_PROJ( POTH(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                     LYMAX, PP%AUG, PP%QPAW )            
            ELSE
               CALL RAD_AUG_PROJ( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                     LYMAX, PP%AUG, PP%QPAW )
            ENDIF
#endif
            IF (ONE_CENTER_NMAX_FOCKAE()>0) THEN
               CALL RAD_AUG_PROJ_FOCK( POT(:,:,ISP), PP%R, DDLM, PP%LMAX, PP%LPS, &
                    LYMAX, PP%AUG_FOCK, PP%QPAW_FOCK )
            ENDIF
         ! transform them using Clebsch Gordan coefficients and add to CDIJ
            CALL TRANS_DLM( CTMP(:,:,ISP), DDLM , PP )
         ENDDO

         
         CMETAGGA=0
         CALL RAD_PROJ_METAGGA(PP,PP%WAE,LMAX_TAU,MUAE, 1._q,CMETAGGA)
         CALL RAD_PROJ_METAGGA(PP,PP%WPS,LMAX_TAU,MUPS,-1._q,CMETAGGA)

         ! non-collinear case: strength parameters need to go to the spinor presentation now
         IF (WDES%LNONCOLLINEAR) THEN
            CALL DIJ_FLIP(CTMP,LMDIM)
            CALL DIJ_FLIP(CMETAGGA,LMDIM)
         ENDIF

         IF (USELDApU() .OR. USEFOCK_CONTRIBUTION() .OR. USEFOCK_AE_ONECENTER()) THEN
            IF (WDES%ISPIN==1.AND.(.NOT.WDES%LNONCOLLINEAR)) THEN
               COCC(:,:,1)=COCC(:,:,1)*0.5_q
               ! LDA+U requires up and down density
               COCC(:,:,2)=COCC(:,:,1)
            ENDIF

          !---------------------------------------------------------------
            IF (USEFOCK_AE_ONECENTER()) THEN
               CALL SETUP_PAWFOCK_AE(NT, PP)  ! this initializes AE part only
               CALL CALC_PAWFOCK(NT, PP, COCC, CHF, DOUBLEC_HF)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_HF*T_INFO%VCA(NT)
            ELSE IF (USEFOCK_CONTRIBUTION()) THEN
               CALL CALC_PAWFOCK(NT, PP, COCC, CHF, DOUBLEC_HF)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_HF*T_INFO%VCA(NT)
            ENDIF
         
           ! correction terms from LDA+U         
            IF (USELDApU()) THEN
          !---------------------------------------------------------------
               CALL LDAPLUSU(LMDIM,NI,NT,COCC, CTMP, PP, DOUBLEC_LDAU)
               DOUBLEC_AE = DOUBLEC_AE + DOUBLEC_LDAU*T_INFO%VCA(NT)
            ENDIF
         ENDIF

         IF (LCALC_ORBITAL_MOMENT().AND.WDES%LNONCOLLINEAR) THEN
            COCC=CRHODE(:,:,NIP,:)
            CALL OCC_FLIP4(COCC,LMDIM) ! go to spinor representation
            CALL CALC_ORBITAL_MOMENT(LMDIM, NI, NT, COCC, PP, 0._q,0._q)
         ENDIF

         IF (WDES%LSORBIT) THEN
            COCC=CRHODE(:,:,NIP,:)
            CALL OCC_FLIP4(COCC,LMDIM) ! go to spinor representation
            CALL CALC_SPINORB_MATRIX_ELEMENTS(WDES,PP,T_INFO,NI,CSO,COCC)
         ENDIF

         CDIJ(:,:,NIP,:)=CDIJ(:,:,NIP,:)+(CTMP+CSO+CHF+CMETAGGA)*T_INFO%VCA(NT)

         DEALLOCATE(TAUAE,TAUPS,MUAE,MUPS,KINDENSCOL)
         CALL UNSET_RSGF_TYPE
         ! we can deallocate the calculated range-separated Greens functions here,
         ! which means that these kernels are recalculated for each ion 
         CALL DEALLOCATE_RSGF
      ENDDO ion
      ! or we could deallocate here, which means that the Greens functions are
      ! only thrown away after SET_DD_PAW has finished completely
!=======================================================================
! now distribute the DIJ to all nodes which hold DIJ (using global sum)
!=======================================================================
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ))
#else
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ*2))
      CALLMPI( M_sum_d(WDES%COMM_KINTER,CDIJ, LMDIM*LMDIM*WDES%NIONS*NCDIJ*2))
#endif
      CALLMPI( M_sum_d(WDES%COMM, CL_SHIFT, SIZE(CL_SHIFT)))

      CALLMPI( M_sum_d(WDES%COMM, DOUBLEC_AE, 1))
      CALLMPI( M_sum_d(WDES%COMM, DOUBLEC_PS, 1))

      CALLMPI( M_sum_d(WDES%COMM, EPAWPSG, 1))
      CALLMPI( M_sum_d(WDES%COMM, EPAWAEG, 1))
      CALLMPI( M_sum_d(WDES%COMM, EPAWCORE, 1))

      IF ( LMETA ) THEN
         CALLMPI( M_sum_d(WDES%COMM, EPAWPSM, 1))
         CALLMPI( M_sum_d(WDES%COMM, EPAWAEM, 1))
         CALLMPI( M_sum_d(WDES%COMM, EPAWCOREM, 1))
      ENDIF
#ifdef debug
      DO K=1,WDES%COMM%NCPU
      IF (WDES%COMM%NODE_ME == K) THEN
         DO NI=1,WDES%NIONS
            CALL DUMP_DLLMM( "CDIJ",CDIJ(:,:,NI,1), PP)
            IF (ISPIN==2) CALL DUMP_DLLMM( "CDIJ",CDIJ(:,:,NI,2), PP)
         ENDDO
         WRITE(*,*)
      ENDIF
      CALL MPI_barrier( WDES%COMM%MPI_COMM, k )
      ENDDO
#endif
      DEALLOCATE(KINDENSAE,WKDAE,WKDPS,KINDENSPS,RHOUPD,RHOLMUPD)
      DEALLOCATE(RHOCOL)
#ifdef noAugXCmeta
      DEALLOCATE(RHOPS_STORE,POTH)
#endif
      DEALLOCATE(POTAE, RHOAE, POT, RHO, DRHOCORE )
      DEALLOCATE(DDLM, RHOLM, RHOLM_)
      DEALLOCATE(CTMP,CSO,CHF,CMETAGGA)
      DEALLOCATE(COCC,COCC_IM)

      E%PAWAE=DOUBLEC_AE
      E%PAWPS=DOUBLEC_PS

      E%PAWPSG  =EPAWPSG
      E%PAWAEG  =EPAWAEG
      E%PAWPSM  =EPAWPSM
      E%PAWAEM  =EPAWAEM
      E%PAWCORE =EPAWCORE
      E%PAWCOREM=EPAWCOREM

      IF (LUSE_THOMAS_FERMI) CALL POP_XC_TYPE

      CALL RELEASE_PAWFOCK

      PROFILING_STOP('set_dd_paw')

    END SUBROUTINE SET_DD_PAW

  END MODULE pawm
