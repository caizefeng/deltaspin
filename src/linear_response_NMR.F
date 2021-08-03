#include "symbol.inc"

#define use_davidson
! #define apply_greens_function
! #define provide_w1
! #define exact_diagonalisation

      MODULE mlr_main_nmr
      USE prec
      USE vaspxml
      IMPLICIT NONE

      COMPLEX(q), PRIVATE, SAVE :: QQ(3,3)

      REAL(q), PRIVATE, SAVE :: CHI_MAG(3,3)

      COMPLEX(q), ALLOCATABLE, PRIVATE, SAVE :: ABS_CHEM_SHIFT(:,:,:)
      REAL(q)   , ALLOCATABLE, PRIVATE, SAVE :: ABS_CHEM_SHIFT_NICS(:,:,:)

      INTEGER, PRIVATE, PARAMETER :: NSTEPS_DAVIDSON=15

      INTEGER, PRIVATE, SAVE :: NKPTS_ORIG
! NICS_
      REAL(q), ALLOCATABLE, PRIVATE, SAVE :: POSROT(:,:)
      INTEGER, PRIVATE, SAVE :: NPOSROT=1
! NICS_
      CONTAINS

!************************ SUBROUTINE MLR_SKELETON **********************
!
!***********************************************************************
      SUBROUTINE MLR_SKELETON( &
     &   HAMILTONIAN,W,WDES,GRID,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,KPOINTS,LATT_CUR,LATT_INI, &
     &   T_INFO,DYN,SYMM,P,NONL_S,NONLR_S,LMDIM,CDIJ,CQIJ,SV,E,INFO,IRDMAX,IO)
      USE prec
      USE base
      USE pseudo
      USE poscar
      USE wave_high
      USE nonl_high
      USE hamil_high
      USE mgrid
      USE lattice
      USE constant
      USE kpoints_change
      USE msymmetry
      USE david
      USE subrot
      USE paw
      USE pead
      USE morbitalmag
      USE choleski
      IMPLICIT NONE
      TYPE (ham_handle) HAMILTONIAN
      TYPE (wavespin) W
      TYPE (wavedes) WDES
      TYPE (grid_3d) GRID
      TYPE (grid_3d) GRID_SOFT               ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d) GRIDC                   ! full, fine grid
      TYPE (grid_3d) GRIDUS
      TYPE (transit) C_TO_US
      TYPE (transit) SOFT_TO_C
      TYPE (kpoints_struct) KPOINTS
      TYPE (latt) LATT_CUR
      TYPE (latt) LATT_INI
      TYPE (type_info) T_INFO
      TYPE (dynamics) DYN
      TYPE (symmetry) SYMM
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct)NONLR_S
      TYPE (energy) E
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (potcar), TARGET :: P(:)
      INTEGER LMDIM,IRDMAX
      OVERLAP CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      RGRID   SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)

      ! local variables
      INTEGER I,J,NI
      INTEGER BDIR ! cartesian component of B-field, 1=x, 2=y, 3=z

      COMPLEX(qs), ALLOCATABLE :: RPHI(:,:,:,:,:)
      GDEFS, ALLOCATABLE ::  RPHI_CPROJ(:,:,:,:,:)
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
           GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
      REAL(q) TMP(3,3), TMP_TENS_AT(3,3)
      REAL(q) TMP_TENS(3,3,T_INFO%NIOND)
      REAL(q) ISO1,SPAN1,SKEW1
      REAL(q) ISO2,SPAN2,SKEW2
      REAL(q), ALLOCATABLE :: CORE_SHIFT(:)
      REAL(q) CORE_CHI, CHI_FACT
      INTEGER K1,K2
      CHARACTER*5 MODUS

      TYPE (skpoints_trans) :: KPOINTS_TRANS
      INTEGER NKORIG
#ifndef use_davidson
      REAL(q) DESUM
#else
      INTEGER NELM
      INTEGER ICOUEV,NSIM
      REAL(q) RMS,DESUM1,TOTEN
! NICS_
      INTEGER NPOSROTMAX,NSPTMP,NSXTMP
      INTEGER, DIMENSION(1):: NANICS
      INTEGER, ALLOCATABLE :: MAPNICS(:,:,:,:)
      REAL(q), ALLOCATABLE :: TRNICS(:,:),TAUWRK(:,:)
! NICS_

      NSIM=W%WDES%NSIM*2
#ifdef MPI
      NSIM=((W%WDES%NSIM*2+W%WDES%COMM_INTER%NCPU-1)/W%WDES%COMM_INTER%NCPU)*W%WDES%COMM_INTER%NCPU
#endif
#endif

      ! setup one-centre partial wave matrix elements for current augmentation
      IF (LLRAUG) CALL CALC_JPAW(T_INFO, P)

      QQ = 0
      CHI_MAG = 0

      ALLOCATE(ABS_CHEM_SHIFT(3,3,T_INFO%NIONS))

      ! possibly we want to reduce the symmetry to strictly
      ! keep the q-stars mapped on themselves for all symmetry operations
      IF (LNMR_SYM_RED.AND.SYMM%ISYM>0) THEN
         NKORIG=W%WDES%NKPTS

         CALL MLR_SYM_REDUCE(LATT_CUR,T_INFO,SYMM,IO%IU6)
#ifdef oldsym
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#else
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#endif
         CALL KPAR_SYNC_ALL(WDES,W)
         CALL RE_GEN_LAYOUT(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
         IF (LUSEPEAD()) CALL PEAD_RESETUP_WDES(WDES,GRID,KPOINTS,LATT_CUR,LATT_INI,IO)
         CALL REALLOCATE_WAVE(W,GRID,WDES,NONL_S,T_INFO,P,LATT_CUR,KPOINTS_TRANS)

         CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)
#ifndef use_davidson 
         ! Loewdin perturbation theory to improve states at added k-points
         CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
        &     LMDIM,CDIJ,CQIJ,4,SV,T_INFO,P,IO%IU0,DESUM,NKSTART=NKORIG+1)
         CALL KPAR_SYNC_ALL(WDES,W)
#else
         ! optimize the states at added k-points with Davidson optimization
         TOTEN=0; INFO%IALGO=8
         DO NELM=1,NSTEPS_DAVIDSON
            CALL EDDAV(HAMILTONIAN,P,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,NSIM, &
           &     LMDIM,CDIJ,CQIJ,RMS,DESUM1,ICOUEV,SV,E%EXHF,IO%IU6,IO%IU0, &
           &     LDELAY=.FALSE.,LSUBROTI=.TRUE.,LEMPTY=.FALSE.,LHF=.TRUE., &
           &     NKSTART=NKORIG+1)

            E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES,W)

            IF (IO%IU0>=0) WRITE(17, 200)      NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
            IF (IO%IU0>=0) WRITE(IO%IU0, 200)  NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
            TOTEN=E%EBANDSTR

 200        FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
                 &       I6,'  ',E10.3)
              
            CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

            IF (ABS(DESUM1) < ABS(INFO%EDIFF) .AND. NELM>1) EXIT
         ENDDO
         CALL KPAR_SYNC_ALL(W%WDES,W)
#endif
      ENDIF

! NICS_
      ! we might want to have some NICS, we need to prepare
      IF(NUCIND.AND..NOT.LNICSALL) THEN
        NPOSROTMAX=NROTK*NPCELL*NPOS_NICS
        ALLOCATE(TAUWRK(NPOSROTMAX,3))
! set up all symmetry equivalent positions of sites read from POSNICS,
! non-primitive translations are included, but not needed.
        CALL PSYMMAKE(TAUNICS,NPOS_NICS,ISYMOP,GTRANS,SYMM%PTRANS,NROTK,NPCELL,TAUWRK,NPOSROT,NPOSROTMAX)
! POSROT starts with NPOS_NICS sites from POSNICS and has all symmetry equivalent positions. Total: NPOSROT
        ALLOCATE(POSROT(NPOSROT,3))
        DO K1=1,NPOSROT
           POSROT(K1,:)=TAUWRK(K1,:)
        ENDDO
        DEALLOCATE(TAUWRK)
        NANICS(1)=NPOSROT
        ALLOCATE(MAPNICS(NPOSROT,1,NROTK,NPCELL))
        ALLOCATE(TRNICS(NPOSROT,3))
        NSPTMP=1
        NSXTMP=1
        CALL POSMAP(POSROT,ISYMOP,GTRANS,NROTK,SYMM%PTRANS,NPCELL,T_INFO%NIOND,NSXTMP,NSPTMP,NPOSROT,NANICS,MAPNICS,TRNICS)
        DEALLOCATE(TRNICS)
        ALLOCATE(ABS_CHEM_SHIFT_NICS(3,3,NPOSROT))
      ENDIF
! NICS_
      ! loop over magnetic field directions
      bloop: DO BDIR=1,3

         IF (LUSEPEAD()) THEN
            ALLOCATE(RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3), &
           &   RPHI_CPROJ(W%WDES%NPROD,W%WDES%NBANDS,W%WDES%NKPTS,W%WDES%ISPIN,3))
            RPHI=0; RPHI_CPROJ=0
            CALL PEAD_DPSI_DK_ALL(W,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,RPHI,RPHI_CPROJ)
            ! calculate induced field
            CALL MLR_B_MAIN( &
           &   HAMILTONIAN,W,GRID,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,KPOINTS,LATT_CUR,LATT_INI,T_INFO,SYMM,P, &
           &   NONL_S,NONLR_S,LMDIM,CDIJ,CQIJ,SV,E,INFO,IRDMAX,IO,BDIR,RPHI)
            DEALLOCATE(RPHI,RPHI_CPROJ)
         ELSE
            CALL MLR_B_MAIN( &
           &   HAMILTONIAN,W,GRID,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,KPOINTS,LATT_CUR,LATT_INI,T_INFO,SYMM,P, &
           &   NONL_S,NONLR_S,LMDIM,CDIJ,CQIJ,SV,E,INFO,IRDMAX,IO,BDIR)
         ENDIF

      ENDDO bloop

      CALLMPI( M_sum_z(W%WDES%COMM,QQ,9))

      DO I=1,3
      DO J=1,3
         QQ(I,J) = QQ(I,J) /DQ/DQ/TPI/TPI/LATT_CUR%OMEGA
         IF (I.NE.J) QQ(I,J) = 2.*QQ(I,J)
         QQ(I,J) = QQ(I,J) * 8._q * PI/3._q
         QQ(I,J) = QQ(I,J) * 1E6 *MOMTOMOM
      ENDDO
      ENDDO

      IF (IO%IU6>=0) THEN
         WRITE (IO%IU6,'(/A)') &
        &                ' -------------------------------------------------------------'
         WRITE (IO%IU6,*) ' Absolute Chemical Shift tensors'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         TMP_TENS=0._q
         DO NI=1,T_INFO%NIONS

!           DO K1=1,3
!           DO K2=1,3
!              TMP_TENS(K1,K2,NI)=REAL(ABS_CHEM_SHIFT(K2,K1,NI),q)
!           ENDDO
!           ENDDO
            TMP_TENS(1:3,1:3,NI)=REAL(ABS_CHEM_SHIFT(1:3,1:3,NI),q)
   
            IF (NI.GT.T_INFO%NIOND) THEN
               WRITE (*,*) 'ERROR, STOP'
               STOP
            ENDIF
         ENDDO
         WRITE (IO%IU6,*) ' UNSYMMETRIZED TENSORS '
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(" ion ",I4)') NI 
            DO BDIR=1,3
!              WRITE(IO%IU6,'(3E18.6)') REAL(ABS_CHEM_SHIFT(:,BDIR,NI),q)
!              WRITE(IO%IU6,'(3E18.6)') TMP_TENS(:,BDIR,NI)
               WRITE(IO%IU6,'(3F18.6)') TMP_TENS(:,BDIR,NI)
            ENDDO
         ENDDO
!        CALL TSYM(TMP,ISYMOP,NROTK,LATT_CUR%A)
         CALL TENSYM(TMP_TENS, &
      &     SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,LATT_CUR%A)
         WRITE (IO%IU6,*) ' SYMMETRIZED TENSORS '
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(" ion ",I4)') NI 
            DO BDIR=1,3
!              WRITE(IO%IU6,'(3E18.6)') REAL(ABS_CHEM_SHIFT(:,BDIR,NI),q)
!              WRITE(IO%IU6,'(3E18.6)') TMP_TENS(:,BDIR,NI)
               WRITE(IO%IU6,'(3F18.6)') TMP_TENS(:,BDIR,NI)
            ENDDO
         ENDDO 
         WRITE (IO%IU6,'(A/)') &
        &                ' -------------------------------------------------------------'

! NICS_
         IF (NUCIND.AND..NOT.LNICSALL) THEN
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) ' Nucleus Independent Chemical Shift, NICS (absolute) '
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) ' UNSYMMETRIZED TENSORS '
         DO NI=1,NPOSROT   ! .NE. NPOS_NICS IN GENERAL
            TMP=0._q
            TMP(1:3,1:3)=ABS_CHEM_SHIFT_NICS(1:3,1:3,NI)
            WRITE (IO%IU6,'(" nics ",I4)') NI
            DO BDIR=1,3
               WRITE(IO%IU6,'(3F18.6)') TMP(1,BDIR),TMP(2,BDIR),TMP(3,BDIR)
            ENDDO
         ENDDO
         WRITE (IO%IU6,*) ' SYMMETRIZED TENSORS '
         CALL TENSYM(ABS_CHEM_SHIFT_NICS(1,1,1), &
      &     MAPNICS,1,NANICS,NPOSROT,LATT_CUR%A)
         DO NI=1,NPOS_NICS
            TMP=0._q
            TMP(1:3,1:3)=ABS_CHEM_SHIFT_NICS(1:3,1:3,NI)
            WRITE (IO%IU6,'(" nics ",I4)') NI
            DO BDIR=1,3
               WRITE(IO%IU6,'(3F18.6)') ABS_CHEM_SHIFT_NICS(:,BDIR,NI)
            ENDDO
         ENDDO
         WRITE (IO%IU6,'(A/)') &
        &         '-------------------------------------------------------------'
         ENDIF 
! NICS_
 
         TMP(1:3,1:3)=REAL(QQ(1:3,1:3),q)
         CALL TSYM(TMP,ISYMOP,NROTK,LATT_CUR%A)
         CHI_FACT= 3._q/8._q/PI*LATT_CUR%OMEGA*6.022142E23/1.E24 ! N_Avogadro * Ang_to_cm^3
         CHI_MAG(1:3,1:3)= TMP(1:3,1:3) * CHI_FACT
         WRITE (IO%IU6,*)
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) ' G=0 CONTRIBUTION TO CHEMICAL SHIFT (field along BDIR) '
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) '  BDIR                X                 Y                 Z'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         DO BDIR=1,3
            WRITE (IO%IU6,'(I6,6F18.6)')  BDIR,TMP(1,BDIR),TMP(2,BDIR),TMP(3,BDIR)
         ENDDO
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) 
         WRITE (IO%IU6,*) ' Approximate magnetic susceptibility, pGv (10^-6 cm^3/mole) '
         DO BDIR=1,3
            WRITE (IO%IU6,'(I6,6F18.6)') BDIR,CHI_MAG(1,BDIR),CHI_MAG(2,BDIR),CHI_MAG(3,BDIR)
         ENDDO
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) 
         WRITE (IO%IU6,*) 

      ENDIF

      ! restore the original symmetry
      IF (LNMR_SYM_RED.AND.SYMM%ISYM>0) THEN
         CALL INISYM(LATT_CUR%A,T_INFO%POSION,DYN%VEL,T_INFO%LSFOR, &
              T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS, &
              SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
              SYMM%INDROT,T_INFO%ATOMOM,WDES%SAXIS,SYMM%MAGROT,WDES%NCDIJ,IO%IU6)
#ifdef oldsym
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LSORBIT.AND..NOT.WDES%LSPIRAL, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#else
         CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
              SYMM%ISYM>=0.AND..NOT.WDES%LNONCOLLINEAR, &
              T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
#endif
         CALL RE_GEN_LAYOUT(GRID,WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
         IF (LUSEPEAD()) CALL PEAD_RESETUP_WDES(WDES,GRID,KPOINTS,LATT_CUR,LATT_INI,IO)
         CALL REALLOCATE_WAVE(W,GRID,WDES,NONL_S,T_INFO,P,LATT_CUR)

         CALL DEALLOCATE_KPOINTS_TRANS(KPOINTS_TRANS)
      ENDIF

      DEALLOCATE(ABS_CHEM_SHIFT)
      IF (NUCIND.AND..NOT.LNICSALL) THEN
        DEALLOCATE(ABS_CHEM_SHIFT_NICS)
        DEALLOCATE(POSROT,MAPNICS)
      ENDIF

      ALLOCATE(CORE_SHIFT(T_INFO%NTYP))

      CALL MLR_CORE_SHIFTS(P,T_INFO,LATT_CUR,IO%IU6,CORE_SHIFT,CORE_CHI)

      IF (IO%IU6>=0) THEN

      ! write csa info

         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'
         WRITE (IO%IU6,'(A)') '  CSA tensor (J. Mason, Solid State Nucl. Magn. Reson. 2, 285 (1993))             '
         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'
         WRITE (IO%IU6,'(A)') '             EXCLUDING G=0 CONTRIBUTION             INCLUDING G=0 CONTRIBUTION    '
         WRITE (IO%IU6,'(A)') '         -----------------------------------   -----------------------------------'
         WRITE (IO%IU6,'(A)') '  ATOM    ISO_SHIFT        SPAN        SKEW     ISO_SHIFT        SPAN        SKEW '
         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'
         WRITE (IO%IU6,'(A)') '  (absolute, valence only)                                                                  '
         MODUS='SHIFT'
         DO NI=1,T_INFO%NIONS
            TMP_TENS_AT(:,:) = TMP_TENS(:,:,NI)
            CALL DIAGSHIFT(TMP_TENS_AT(1,1),MODUS,ISO1,SPAN1,SKEW1)
            TMP_TENS_AT(:,:) = TMP_TENS(:,:,NI)+TMP(:,:)
            CALL DIAGSHIFT(TMP_TENS_AT(1,1),MODUS,ISO2,SPAN2,SKEW2)
            IF (SPAN1.LT.0.00005_q) SKEW1=0._q
            IF (SPAN2.LT.0.00005_q) SKEW2=0._q
            WRITE (IO%IU6,'(I6,X,3F12.4,2X,3F12.4)') NI, ISO1, SPAN1, SKEW1, ISO2, SPAN2, SKEW2
         ENDDO
         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'
         WRITE (IO%IU6,'(A)') '  (absolute, valence and core)                                                                  '
         DO NI=1,T_INFO%NIONS
            TMP_TENS_AT(:,:) = TMP_TENS(:,:,NI)        
            TMP_TENS_AT(1,1) = TMP_TENS_AT(1,1) + CORE_SHIFT(T_INFO%ITYP(NI))
            TMP_TENS_AT(2,2) = TMP_TENS_AT(2,2) + CORE_SHIFT(T_INFO%ITYP(NI))
            TMP_TENS_AT(3,3) = TMP_TENS_AT(3,3) + CORE_SHIFT(T_INFO%ITYP(NI))
            CALL DIAGSHIFT(TMP_TENS_AT(1,1),MODUS,ISO1,SPAN1,SKEW1)
            TMP_TENS_AT(:,:) = TMP_TENS(:,:,NI) + TMP(:,:)
            TMP_TENS_AT(1,1) = TMP_TENS_AT(1,1) + CORE_SHIFT(T_INFO%ITYP(NI)) + CORE_CHI/CHI_FACT
            TMP_TENS_AT(2,2) = TMP_TENS_AT(2,2) + CORE_SHIFT(T_INFO%ITYP(NI)) + CORE_CHI/CHI_FACT
            TMP_TENS_AT(3,3) = TMP_TENS_AT(3,3) + CORE_SHIFT(T_INFO%ITYP(NI)) + CORE_CHI/CHI_FACT
            CALL DIAGSHIFT(TMP_TENS_AT(1,1),MODUS,ISO2,SPAN2,SKEW2)
            IF (SPAN1.LT.0.00005_q) SKEW1=0._q
            IF (SPAN2.LT.0.00005_q) SKEW2=0._q
            WRITE (IO%IU6,'(I6,X,3F12.4,2X,3F12.4)') NI, ISO1, SPAN1, SKEW1, ISO2, SPAN2, SKEW2
         ENDDO
         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'
         WRITE (IO%IU6,'(A)') '  IF SPAN.EQ.0, THEN SKEW IS ILL-DEFINED                                          '
         WRITE (IO%IU6,'(A)') ' ---------------------------------------------------------------------------------'

         WRITE (IO%IU6,*)
         WRITE (IO%IU6,*)

      ENDIF

      DEALLOCATE(CORE_SHIFT)

      RETURN
      END SUBROUTINE MLR_SKELETON


!************************ SUBROUTINE MLR_CORE_SHIFT ********************
!
!***********************************************************************
      SUBROUTINE MLR_CORE_SHIFTS(P,T_INFO,LATT_CUR,IU6,CORE_SHIFT,CORE_CHI)
      USE pseudo
      USE poscar
      USE rhfatm
      USE radial
      USE lattice
      USE constant
      TYPE (potcar), TARGET :: P(:)
      TYPE (type_info) T_INFO
      TYPE (latt) LATT_CUR
      INTEGER IU6
      REAL(q) CORE_SHIFT(*),CORE_CHI
      ! local variables
      TYPE (potcar), POINTER :: PP
      TYPE (atomic) :: ATOM
      INTEGER NT,I
      REAL(q), ALLOCATABLE :: RHOC(:)
      REAL(q) RES
 
      REAL(q), PARAMETER :: C=137.037_q

      IF (IU6<0) RETURN

      DO NT=1,T_INFO%NTYP
         CORE_SHIFT(NT)=0._q
      ENDDO
      CORE_CHI=0._q

      DO NT=1,T_INFO%NTYP
         PP=>P(NT)
         IF (.NOT.ASSOCIATED(PP%ATOMIC_N)) THEN
            WRITE(IU6,'(/A,I4)') ' WARNING: MLR_CORE_SHIFTS: can not compute core contributions for atomic type:',NT
            WRITE(IU6,'(A/)')    '          CORE CONTRIBUTION TO SHIELDING AND SUSCEPTIBILITY ARE INCORRECT'
            CYCLE
         ENDIF
         CALL RHFATM_DFATOM(PP,ATOM)
         ALLOCATE(RHOC(ATOM%R%NMAX))
         RHOC=0._q
         DO I=1,ATOM%NSCORE
!           RHOC(:)=RHOC(:)+2*(2*ATOM%L(I)+1)*ATOM%W(:,I)**2/ATOM%R%R(:)
            RHOC(:)=RHOC(:)+2*(2*ATOM%L(I)+1)*(ATOM%A(:,I)**2+ATOM%B(:,I)**2/C/C)/ATOM%R%R(:)
         ENDDO
         CALL SIMPI(ATOM%R,RHOC,RES)
         CORE_SHIFT(NT)=-RES/2._q/C*2._q/3._q/C*AUTOA

         RHOC=0._q
         DO I=1,ATOM%NSCORE
            RHOC(:)=RHOC(:)+2*(2*ATOM%L(I)+1)*ATOM%W(:,I)**2*ATOM%R%R(:)**2
         ENDDO
         CALL SIMPI(ATOM%R,RHOC,RES)
         CORE_CHI=CORE_CHI-T_INFO%NITYP(NT)*RES/C/C*AUTOA

         DEALLOCATE(RHOC)
         CALL RHFATM_DEALLOCW(ATOM)
      ENDDO

      WRITE(IU6,'(A)')  ' --------------------------------------------------------------------------'
      WRITE(IU6,'(A)')  '  Core NMR properties        '

      WRITE(IU6,'(/A)') '  typ  El   Core shift (ppm)'
      WRITE(IU6,'(A)')  ' ----------------------------'
      DO NT=1,T_INFO%NTYP
         WRITE(IU6,'(X,I4,2X,A,2X,F14.7)') NT,P(NT)%ELEMENT,CORE_SHIFT(NT)*1.E6_q
      ENDDO
      WRITE(IU6,'(A)')  ' ----------------------------'

      WRITE(IU6,'(/A,F10.2,X,A)') '  Core contribution to magnetic susceptibility:',CORE_CHI*1.E5_q,' 10^-6 cm^3/mole'
      WRITE(IU6,'(A)')  ' --------------------------------------------------------------------------'
      WRITE(IU6,*)
      WRITE(IU6,*)

      DO NT=1,T_INFO%NTYP
         CORE_SHIFT(NT) = CORE_SHIFT(NT)*1.E6_q
      ENDDO
      CORE_CHI = CORE_CHI*1.E5_q

      RETURN
      END SUBROUTINE MLR_CORE_SHIFTS


!************************ SUBROUTINE MLR_B_MAIN ************************
!
!***********************************************************************
      SUBROUTINE MLR_B_MAIN( &
     &   HAMILTONIAN,W,GRID,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,KPOINTS,LATT_CUR,LATT_INI,T_INFO,SYMM,P, &
     &   NONL_S,NONLR_S,LMDIM,CDIJ,CQIJ,SV,E,INFO,IRDMAX,IO,BDIR,RPHI)
      USE prec
      USE base
      USE pseudo
      USE poscar
      USE wave_high
      USE nonl_high
      USE hamil_high
      USE mgrid
      USE lattice
      USE constant
      USE kpoints_change
      USE david
      USE subrot
      USE paw
      USE choleski
      USE main_mpi
      USE morbitalmag
      IMPLICIT NONE
      TYPE (ham_handle) HAMILTONIAN
      TYPE (wavespin) W
      TYPE (grid_3d) GRID
      TYPE (grid_3d) GRID_SOFT               ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d) GRIDC                   ! full, fine grid
      TYPE (grid_3d) GRIDUS
      TYPE (transit) C_TO_US
      TYPE (transit) SOFT_TO_C
      TYPE (kpoints_struct) KPOINTS
      TYPE (latt) LATT_CUR
      TYPE (latt) LATT_INI
      TYPE (type_info) T_INFO
      TYPE (symmetry) SYMM
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct)NONLR_S
      TYPE (energy) E
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (potcar), TARGET :: P(:)
      INTEGER LMDIM,IRDMAX
      OVERLAP CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      RGRID   SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
      INTEGER BDIR ! cartesian component of B-field, 1=x, 2=y, 3=z
      COMPLEX(qs), OPTIONAL :: RPHI(:,:,:,:,:)

      RGRID :: S_BARE(W%WDES%GRID%RL%NP,3)   ! three component vector storing current density in real space,
                                             ! we should be careful allocating this

      ! local variables
      TYPE (wavespin) WGKPQ ! G u_i v_k+q, k| u>
      TYPE (wavespin) WXI

      TYPE (wavedes), TARGET  :: WDES_kpq

      TYPE (nonl_struct) NONL_D
      TYPE (nonlr_struct), TARGET :: NONLR_D

      TYPE (nonl_struct) NONL_S_tmp

      INTEGER IDIR ! cartesian component of k-point shift
      INTEGER VDIR ! cartesian component of velocity operator, 1=x, 2=y, 3=z
      INTEGER NK,ISP,IQ
      INTEGER N,NELM
      INTEGER NSIM
      REAL(q) DISPL(3),DISPL_CART(3) 
      INTEGER ICOUEV
      REAL(q) RMS,DESUM1,TOTEN
      REAL(q) WSCAL

      COMPLEX(q) B_OUT_PARA_PS(3,T_INFO%NIONS)  ! output magnetic fields
      REAL(q) B_NICS_OUT_PARA_PS(3,NPOSROT)     ! output NICS magnetic fields
      REAL(q) B_PARA_ONE_CENTR(3,T_INFO%NIONS)  ! for direction BDIR
      REAL(q) B_DIA_ONE_CENTR(3,T_INFO%NIONS)
      REAL(q) B_TMP(3)  ! helper array
      TYPE (potcar), POINTER :: PP
      COMPLEX(q) CRHODE(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      COMPLEX(q) CRHODE_AUG(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ) ! augmentation strengths
      ! because the current augmentation has to be done in PS_BIOT_SAVART_DR the strengths have to be
      ! processed and carried around separately. alternatively the calls to CALC_B_MAGATOM_LR might be
      ! merged so that CRHODE and CRHODE_AUG can be identical arrays, however, this makes for more
      ! complications
      COMPLEX(q) CLQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      INTEGER NI, NIP, NT, IDP, I, J

      COMPLEX(q) NAB_QQ(3)
      INTEGER ICART_K_DIR
      INTEGER IQDIR, JQDIR
      CHARACTER(2) :: DIR_TEXT(3)=(/"BX","BY","BZ"/)

      REAL(q) STENCIL(7,3)
      DATA STENCIL /  1.0_q,            -2.0_q,           1.0_q,  0.0_q,           0.0_q,             0.0_q,  0.0_q, &
                     -0.083333333333_q,  1.3333333333_q, -2.5_q,  1.3333333333_q, -0.083333333333_q,  0.0_q,  0.0_q, &
                      0.011111111111_q, -0.15_q,          1.5_q, -2.7222222222_q,  1.5_q,            -0.15_q, 0.011111111111_q /

      INTEGER CHI_BARE_STENCIL

#ifdef exact_diagonalisation
      INTEGER IFLAG
#endif
      CHI_BARE_STENCIL=ICHIBARE 

      B_OUT_PARA_PS=0     ! initialize fields
      B_PARA_ONE_CENTR=0
      B_DIA_ONE_CENTR=0
     
      NSIM=W%WDES%NSIM*2
#ifdef MPI
      NSIM=((W%WDES%NSIM*2+W%WDES%COMM_INTER%NCPU-1)/W%WDES%COMM_INTER%NCPU)*W%WDES%COMM_INTER%NCPU
#endif

      S_BARE = 0._q

      ! allocate before doubling the k-point grid
      CALL ALLOCW(W%WDES,WGKPQ)
      CALL ALLOCW(W%WDES,WXI)

      CALL CHECK_FULL_KPOINTS

      ! original number of k-points
      NKPTS_ORIG=W%WDES%NKPTS

      ! double the number of k-points
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0,W%WDES%VKPT(:,1:NKPTS_ORIG))
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0,W%WDES%VKPT(:,1:NKPTS_ORIG))
#endif
      CALL KPAR_SYNC_ALL(W%WDES,W)
      CALL RE_GEN_LAYOUT(GRID,W%WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
      CALL REALLOCATE_WAVE(W,GRID,W%WDES,NONL_S,T_INFO,P,LATT_CUR)

!     CALL SETWDES(W%WDES,WDES1,0)
!     CALL NEWWAV(W1, WDES1, .FALSE.)

      ! WGKPQ and WXI will live at k+q, their WDES remains the same except WDES%NKPTS, WDES%VKPT,
      ! WDES%DATAKE should point to the upper panel of NKPTS_ORIG k-points
      WDES_kpq=W%WDES
      WDES_kpq%NKPTS=NKPTS_ORIG
      WDES_kpq%VKPT=>W%WDES%VKPT(:,NKPTS_ORIG+1:W%WDES%NKPTS)
      WDES_kpq%DATAKE=>W%WDES%DATAKE(:,:,NKPTS_ORIG+1:W%WDES%NKPTS)

      WGKPQ%WDES=>WDES_kpq; WXI%WDES=>WDES_kpq

      CRHODE_AUG=0._q

      iloop: DO IDIR=1,3
         ! skip the component along the B-field
         IF (IDIR==BDIR) CYCLE iloop

         ! the direction of the velocity operator we need here is \hat{B}_bdir x \hat{q}_idir 
         VDIR=6-BDIR-IDIR

         ! setup  r | p_i > 
         IF (INFO%LREAL) THEN
!           NONLR_D=NONLR_S
            CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
            CALL NONLR_ALLOC_CRREXP(NONLR_D)
         ELSE
            CALL NONL_ALLOC_DER(NONL_S,NONL_D)
!           CALL SPHER_DER(GRID,NONL_D,P,W%WDES,LATT_CUR,VDIR)
         ENDIF

         DISPL_CART=0; DISPL_CART(IDIR)=DQ
         DISPL=DISPL_CART
         CALL KARDIR(1, DISPL(1), LATT_CUR%A)

         qloop: DO IQ=-CHI_BARE_STENCIL,CHI_BARE_STENCIL
            IF (IO%IU0>=0) WRITE(*,'("BDIR=",I2,2X,"IDIR=",I2,2X,"IQ=",I3)') BDIR,IDIR,IQ

            WGKPQ%CW=(0._q,0._q); WGKPQ%CPROJ=zero; WXI%CW=(0._q,0._q); WXI%CPROJ=zero

            ! set additional k-points to k+q
            DO NK=1,NKPTS_ORIG
               W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)+IQ*DISPL(:)
!              W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)
            ENDDO

            CALL SPHER_DER(GRID,NONL_D,P,W%WDES,LATT_CUR,VDIR)

            ! fill additional wave functions
            IF (PRESENT(RPHI)) THEN
               DO ISP=1,W%WDES%ISPIN
                  DO NK=1,NKPTS_ORIG
                     DO N=1,W%WDES%NBANDS
                        W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)+TPI*( &
                       &   RPHI(:,N,NK,ISP,1)*(0.0_q,1.0_q)*IQ*DISPL_CART(1)+ &
                       &   RPHI(:,N,NK,ISP,2)*(0.0_q,1.0_q)*IQ*DISPL_CART(2)+ &
                       &   RPHI(:,N,NK,ISP,3)*(0.0_q,1.0_q)*IQ*DISPL_CART(3))
                     ENDDO
                  ENDDO
               ENDDO
            ELSE
               DO ISP=1,W%WDES%ISPIN
                  DO NK=1,NKPTS_ORIG
                     DO N=1,W%WDES%NBANDS
                        W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)
                     ENDDO
                  ENDDO
               ENDDO
            ENDIF           
            
            CALL SET_DATAKE(W%WDES,LATT_CUR%B)
            IF (NONLR_S%LREAL) THEN
               CALL RSPHER(GRID,NONLR_S,LATT_CUR)
            ELSE
               CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR,1)
            ENDIF
            NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)

            CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

!           IF (IQ/=0) THEN
#ifdef exact_diagonalisation
!              CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
!                          LMDIM,CDIJ,CQIJ,3,SV,T_INFO,P,IO%IU0,E%EXHF, &
!                          NKSTART=NKPTS_ORIG+1)
               CALL EDDIAG_EXACT(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
                           LMDIM,CDIJ,CQIJ,IFLAG,SV,T_INFO,P,IO%IU0,IO%IU6,E%EXHF)
               CALL KPAR_SYNC_ALL(W%WDES,W)
#else
               ! let's do a bit of Davidson optimization
               W%FERTOT(:,NKPTS_ORIG+1:W%WDES%NKPTS,:)=W%FERTOT(:,1:NKPTS_ORIG,:)
               W%WDES%WTKPT(NKPTS_ORIG+1:W%WDES%NKPTS)=W%WDES%WTKPT(1:NKPTS_ORIG)

               TOTEN=0; INFO%IALGO=8
               DO NELM=1,NSTEPS_DAVIDSON
                  CALL EDDAV(HAMILTONIAN,P,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,NSIM, &
                 &     LMDIM,CDIJ,CQIJ,RMS,DESUM1,ICOUEV,SV,E%EXHF,IO%IU6,IO%IU0, &
                 &     LDELAY=.FALSE.,LSUBROTI=.TRUE.,LEMPTY=.FALSE.,LHF=.TRUE., &
                 &     NKSTART=NKPTS_ORIG+1)

                  E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES,W)

                  IF (IO%IU0>=0) WRITE(17, 200)      NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
                  IF (IO%IU0>=0) WRITE(IO%IU0, 200)  NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
                  TOTEN=E%EBANDSTR

 200              FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
                       &       I6,'  ',E10.3)
              
                  CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

                  IF (ABS(DESUM1) < ABS(INFO%EDIFF) .AND. NELM>1) EXIT
               ENDDO
               CALL KPAR_SYNC_ALL(W%WDES,W)
#endif
!           ENDIF
            ! let's start the linear response stuff to get the contribution | psi^(1e) >
            CALL MLR_PSI_RESPONSE_EMPTY(W,GRID,P,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CDIJ,CQIJ,SV,INFO,IO,VDIR,WXI,WGKPQ)

            ! here we should get the contributions Q(q)
            CALL NABIJ_ASYM(W, WGKPQ, GRID, LATT_CUR, NKPTS_ORIG, INFO, NAB_QQ)
            DO ICART_K_DIR=1,3
               IQDIR =6-IDIR-ICART_K_DIR
               JQDIR =6-IDIR-VDIR
               IF (IQDIR==0.OR.JQDIR==0) CYCLE
!              IF ((IQ==-1).OR.(IQ==1)) QQ(IQDIR,JQDIR)=QQ(IQDIR,JQDIR)  &
!    &                               + NAB_QQ(ICART_K_DIR)*EIJK(IQDIR,IDIR,ICART_K_DIR)*EIJK(JQDIR,IDIR,VDIR)
!              IF (IQ==0)               QQ(IQDIR,JQDIR)=QQ(IQDIR,JQDIR)  &
!    &                        - 2._q * NAB_QQ(ICART_K_DIR)*EIJK(IQDIR,IDIR,ICART_K_DIR)*EIJK(JQDIR,IDIR,VDIR)
               QQ(IQDIR,JQDIR)=QQ(IQDIR,JQDIR)+ &
              &   NAB_QQ(ICART_K_DIR)*EIJK(IQDIR,IDIR,ICART_K_DIR)*EIJK(JQDIR,IDIR,VDIR)*STENCIL(IQ+CHI_BARE_STENCIL+1,CHI_BARE_STENCIL)
            ENDDO

            IF (IQ==0.OR.ABS(IQ)>1) CYCLE qloop

            ! and the contribution | \psi^(1o) >
            CALL MLR_PSI_RESPONSE_OCC(W,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CQIJ,IO,INFO,VDIR,WXI,WGKPQ)

            WGKPQ%CW=WGKPQ%CW*(0._q,-1._q)

            ! we should use the upper panel of NONL_S%QPROJ, i.e., the projectors at k+q
            NONL_S_tmp=NONL_S; NONL_S_tmp%NK=-1
            NONL_S_tmp%QPROJ=>NONL_S%QPROJ(:,:,:,NKPTS_ORIG+1:W%WDES%NKPTS,:)
            CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S_tmp,WGKPQ) 

            ! the sign of \hat{B}_bdir x \hat{q}_idir is given by the Levi-Civita symbol
            WGKPQ%CW=EIJK(VDIR,BDIR,IDIR)*WGKPQ%CW; WGKPQ%CPROJ=EIJK(VDIR,BDIR,IDIR)*WGKPQ%CPROJ

            ! mupltiply with the sign of q
            WGKPQ%CW=WGKPQ%CW*IQ; WGKPQ%CPROJ=WGKPQ%CPROJ*IQ

            ! compute current contributions to S_bare(q).
            CALL MLR_SBARE(W,WGKPQ,S_BARE)

            ! compute current contributions to B_Dp.
            CALL DEPSUM_ASYM(W ,WGKPQ , W%WDES, LMDIM, CRHODE, W%WDES%LOVERL)   ! or cc?

            CALL US_FLIP_CMPLX(W%WDES, LMDIM, CRHODE, W%WDES%LOVERL, .FALSE.)

            ! augmentation strenghts
            CRHODE_AUG(:,:,:,:)=CRHODE_AUG(:,:,:,:)+CRHODE(:,:,:,:)

            IDP=1                       ! only paramagnetic contribution to field

            DO NI=1,T_INFO%NIONS        ! index of ion

               NIP=NI_LOCAL(NI, W%WDES%COMM_INB)   ! local index of ion
               IF (NIP==0) CYCLE                   ! not stored locally return
               NT=T_INFO%ITYP(NI)                  ! type of ion
!              PP=> P(NT)
               PP=>PP_POINTER(P,NI,NT)

               IF ( ASSOCIATED(PP%QPAW)) THEN
                 IF (DO_LOCAL(NI)) THEN
                    CALL CALC_B_MAGATOM_LR(PP, CRHODE(:,:,NIP,1),BDIR,IDP,NIP,NI, &
                         B_TMP(:),LLRAUG)
! TEST
                    B_TMP(:)=-B_TMP(:)
! TEST
                    B_PARA_ONE_CENTR(:,NI) = B_PARA_ONE_CENTR(:,NI) + B_TMP(:)
                 ENDIF
               ENDIF
            ENDDO

         ENDDO qloop

         ! some deallocation
         IF (INFO%LREAL) THEN
            CALL NONLR_DEALLOC_CRREXP(NONLR_D)
         ELSE
            CALL NONL_DEALLOC_DER(NONL_D)
         ENDIF

      ENDDO iloop

      ! here one has to divide by q, both soft and one centre
      DO I=1,W%WDES%GRID%RL%NP
         DO J=1,3
            S_BARE(I,J) = S_BARE(I,J)/DQ/2._q/TPI
         ENDDO
      ENDDO
      DO I=1,3
         DO J=1,T_INFO%NIONS
            B_PARA_ONE_CENTR(I,J) = B_PARA_ONE_CENTR(I,J)/DQ/2._q/TPI
         ENDDO
      ENDDO
      CRHODE_AUG(:,:,:,:)=CRHODE_AUG(:,:,:,:)/DQ/2._q/TPI

      ! here one has to add j_bare,Q_R and j_dp,Q_R
      CALL MLR_CALC_CLQIJ(W%WDES,P,T_INFO,LMDIM,BDIR,CLQIJ)
! test
!     CALL DUMP_DLLMM( "CLQIJ",CLQIJ(:,:,1,1), P(1))
!     CALL DUMP_DLLMM( "CQIJ",  CQIJ(:,:,1,1), P(1))
! test
      ! everything here will be at k
      W%WDES%VKPT(:,NKPTS_ORIG+1:W%WDES%NKPTS)=W%WDES%VKPT(:,1:NKPTS_ORIG)
      W%CW(:,:,NKPTS_ORIG+1:W%WDES%NKPTS,:)=W%CW(:,:,1:NKPTS_ORIG,:)

      CALL SET_DATAKE(W%WDES,LATT_CUR%B)
      IF (NONLR_S%LREAL) THEN
         CALL RSPHER(GRID,NONLR_S,LATT_CUR)
      ELSE
         CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR,1)
      ENDIF
      NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)

      CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

#ifdef exact_diagonalisation
      CALL EDDIAG_EXACT(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
                  LMDIM,CDIJ,CQIJ,IFLAG,SV,T_INFO,P,IO%IU0,IO%IU6,E%EXHF)
#endif
      CALL KPAR_SYNC_ALL(W%WDES,W)

      WGKPQ%WDES%VKPT=>W%WDES%VKPT(:,1:NKPTS_ORIG); WXI%WDES%VKPT=>WGKPQ%WDES%VKPT
      WGKPQ%WDES%DATAKE=>W%WDES%DATAKE(:,:,1:NKPTS_ORIG); WXI%WDES%DATAKE=>WGKPQ%WDES%DATAKE

      WGKPQ%CW=(0._q,0._q); WGKPQ%CPROJ=zero; WXI%CW=(0._q,0._q); WXI%CPROJ=zero

      CALL MLR_PSI_RESPONSE_LQ(W,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CDIJ,CQIJ,CLQIJ,SV,INFO,IO,WXI,WGKPQ)

      ! 
      NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,WGKPQ)

      ! compute current contributions to S_bare(q).
      CALL MLR_SBARE(W,WGKPQ,S_BARE)

      ! compute current contributions to B_Dp.
      CALL DEPSUM_ASYM(W ,WGKPQ , W%WDES, LMDIM, CRHODE, W%WDES%LOVERL)   ! or cc?

      CRHODE_AUG(:,:,:,:)=CRHODE_AUG(:,:,:,:)+CRHODE(:,:,:,:)

      IDP=1                       ! only paramagnetic contribution to field

      DO NI=1,T_INFO%NIONS        ! index of ion

         NIP=NI_LOCAL(NI, W%WDES%COMM_INB)   ! local index of ion
         IF (NIP==0) CYCLE                   ! not stored locally return
         NT=T_INFO%ITYP(NI)                  ! type of ion
!        PP=> P(NT)
         PP=>PP_POINTER(P,NI,NT)

         IF ( ASSOCIATED(PP%QPAW)) THEN
           IF (DO_LOCAL(NI)) THEN
              CALL CALC_B_MAGATOM_LR(PP, CRHODE(:,:,NIP,1),BDIR,IDP,NIP,NI, &
                   B_TMP(:),LLRAUG)
! TEST
              B_TMP(:)=-B_TMP(:)
! TEST
              B_PARA_ONE_CENTR(:,NI) = B_PARA_ONE_CENTR(:,NI) + B_TMP(:)
           ENDIF
         ENDIF
      ENDDO

      ! sum contributions to one-centre magnetic fields along BDIR
      CALLMPI( M_sum_d(W%WDES%COMM_KIN, B_PARA_ONE_CENTR(1,1), SIZE(B_PARA_ONE_CENTR)))

      ! we might want to look at the current
      IF (LWRTCUR) THEN
         IF (IO%IU6>=0) THEN
            OPEN(UNIT=99,FILE=DIR_APP(1:DIR_LEN)//'JCAR'//DIR_TEXT(BDIR),STATUS='UNKNOWN')
            ! write header
            CALL OUTPOS(99,.FALSE.,INFO%SZNAM1,T_INFO,LATT_CUR%SCALE,LATT_CUR%A,.FALSE.,T_INFO%POSION)
            WRITE(99,'(3I5)') GRID_SOFT%NGX,GRID_SOFT%NGY,GRID_SOFT%NGZ
            WRITE(99,'(5(1X,E17.11))') (0._q, I=1,GRID_SOFT%NGX*GRID_SOFT%NGY*GRID_SOFT%NGZ)
            DO J=1,3
               WRITE(99,'(3I5)') GRID_SOFT%NGX,GRID_SOFT%NGY,GRID_SOFT%NGZ
               WRITE(99,'(5(1X,E17.11))') &
              &  (REAL(S_BARE(I,J),q),I=1,GRID_SOFT%NGX*GRID_SOFT%NGY*GRID_SOFT%NGZ)
            ENDDO
            CLOSE(99)
         ENDIF
      ENDIF

      ! here one has to call Biot-Savart for the PS current
      CALL PS_BIOT_SAVART_DR(S_BARE,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,T_INFO,LATT_CUR,W%WDES,B_OUT_PARA_PS,B_NICS_OUT_PARA_PS,CRHODE_AUG,LMDIM,IRDMAX,P,IO%IU0,IO%IU6)

      ! calculate the diamagnetic one-centre field
      CALL DEPSUM_ASYM(W, W, W%WDES, LMDIM, CRHODE, W%WDES%LOVERL)

      IDP=2                       ! only diamagnetic contribution to field

      DO NI=1,T_INFO%NIONS        ! index of ion

         NIP=NI_LOCAL(NI, W%WDES%COMM_INB)   ! local index of ion
         IF (NIP==0) CYCLE                   ! not stored locally return
         NT=T_INFO%ITYP(NI)                  ! type of ion
!        PP=> P(NT)
         PP=>PP_POINTER(P,NI,NT)

         IF ( ASSOCIATED(PP%QPAW)) THEN
           IF (DO_LOCAL(NI)) THEN
              CALL CALC_B_MAGATOM_LR(PP, CRHODE(:,:,NIP,1),BDIR,IDP,NIP,NI, &
                   B_TMP(:),LLRAUG)
              B_DIA_ONE_CENTR(:,NI) = B_TMP(:)
           ENDIF
         ENDIF
      ENDDO

      CALLMPI( M_sum_d(W%WDES%COMM_KIN, B_DIA_ONE_CENTR(1,1), SIZE(B_DIA_ONE_CENTR)))

      ! write out one column of the tensor, should maybe go elsewhere

      B_OUT_PARA_PS(:,:)     =B_OUT_PARA_PS(:,:)   *1E6 *MOMTOMOM
      B_NICS_OUT_PARA_PS(:,:)=B_NICS_OUT_PARA_PS(:,:)   *MOMTOMOM
      B_PARA_ONE_CENTR(:,:)  =B_PARA_ONE_CENTR(:,:)*1E6 *MOMTOMOM
      B_DIA_ONE_CENTR(:,:)   =B_DIA_ONE_CENTR(:,:) *1E6

      DO NI=1,T_INFO%NIONS
         ABS_CHEM_SHIFT(:,BDIR,NI)=B_OUT_PARA_PS(:,NI)+B_PARA_ONE_CENTR(:,NI)+B_DIA_ONE_CENTR(:,NI)
      ENDDO

      IF (IO%IU6>=0) THEN
         WRITE (IO%IU6,*)
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) ' EXTERNAL FIELD APPLIED IN DIRECTION',BDIR
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) '                       MAGNETIC FIELD (mu_B/Angstrom3/1e6)'     ! jaja
         WRITE (IO%IU6,*) ' ATOM                 X                 Y                 Z'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(I6,3E18.6)') NI, &
     &      REAL(B_OUT_PARA_PS(1,NI),q)+B_PARA_ONE_CENTR(1,NI)+B_DIA_ONE_CENTR(1,NI), &
     &      REAL(B_OUT_PARA_PS(2,NI),q)+B_PARA_ONE_CENTR(2,NI)+B_DIA_ONE_CENTR(2,NI), &
     &      REAL(B_OUT_PARA_PS(3,NI),q)+B_PARA_ONE_CENTR(3,NI)+B_DIA_ONE_CENTR(3,NI)
         ENDDO 
         WRITE (IO%IU6,*) 
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) '  plane wave contribution'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(I6,6E18.6)') NI,REAL(B_OUT_PARA_PS(1,NI),q), &
     &                                   REAL(B_OUT_PARA_PS(2,NI),q), &
     &                                   REAL(B_OUT_PARA_PS(3,NI),q)
         ENDDO 
         WRITE (IO%IU6,*) 
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) '  one center paramagnetic contribution'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(I6,3E18.6)') NI,B_PARA_ONE_CENTR(1,NI), &
     &                                   B_PARA_ONE_CENTR(2,NI), &
     &                                   B_PARA_ONE_CENTR(3,NI)
         ENDDO
         WRITE (IO%IU6,*)
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*) '  one center diamagnetic contribution'
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         DO NI=1,T_INFO%NIONS
            WRITE (IO%IU6,'(I6,3E18.6)') NI,B_DIA_ONE_CENTR(1,NI), &
     &                                   B_DIA_ONE_CENTR(2,NI), &
     &                                   B_DIA_ONE_CENTR(3,NI)
         ENDDO
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         IF (NUCIND.AND..NOT.LNICSALL) THEN
            WRITE (IO%IU6,*)
            WRITE (IO%IU6,*) '-------------------------------------------------------------'
            WRITE (IO%IU6,*) '  NICS at the position given in file POSNICS'
            WRITE (IO%IU6,*) '-------------------------------------------------------------'         
            DO NI=1,NPOS_NICS
               WRITE (IO%IU6,'(I6,3E18.6)') NI,B_NICS_OUT_PARA_PS(1,NI), &
     &                                         B_NICS_OUT_PARA_PS(2,NI), &
     &                                         B_NICS_OUT_PARA_PS(3,NI)
            ENDDO
            DO NI=1,NPOSROT
               ABS_CHEM_SHIFT_NICS(:,BDIR,NI)=B_NICS_OUT_PARA_PS(:,NI)
            ENDDO
         ENDIF
         WRITE (IO%IU6,*) '-------------------------------------------------------------'
         WRITE (IO%IU6,*)
      ENDIF

      CALL DEALLOCW(WGKPQ)
      CALL DEALLOCW(WXI)

      ! restore to the original number of k-points
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#endif
      CALL RE_GEN_LAYOUT(GRID,W%WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
      CALL REALLOCATE_WAVE(W,GRID,W%WDES,NONL_S,T_INFO,P,LATT_CUR)     

      RETURN
      END SUBROUTINE MLR_B_MAIN


!************************ SUBROUTINE MLR_B_QQ **************************
!
!***********************************************************************
      SUBROUTINE MLR_B_QQ( &
     &   HAMILTONIAN,W,GRID,GRID_SOFT,GRIDC,SOFT_TO_C,KPOINTS,LATT_CUR,LATT_INI,T_INFO,SYMM,P,NONL_S,NONLR_S, &
     &   LMDIM,CDIJ,CQIJ,SV,E,INFO,IO,BDIR,RPHI)
      USE prec
      USE base
      USE pseudo
      USE poscar
      USE wave_high
      USE nonl_high
      USE hamil_high
      USE mgrid
      USE lattice
      USE constant
      USE kpoints_change
      USE david
      USE subrot
      USE paw
      USE choleski
      USE main_mpi
      USE morbitalmag
      IMPLICIT NONE
      TYPE (ham_handle) HAMILTONIAN
      TYPE (wavespin) W
      TYPE (grid_3d) GRID
      TYPE (grid_3d) GRID_SOFT               ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d) GRIDC                   ! full, fine grid
      TYPE (transit) SOFT_TO_C
      TYPE (kpoints_struct) KPOINTS
      TYPE (latt) LATT_CUR
      TYPE (latt) LATT_INI
      TYPE (type_info) T_INFO
      TYPE (symmetry) SYMM
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct)NONLR_S
      TYPE (energy) E
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (potcar), TARGET :: P(:)
      INTEGER LMDIM
      OVERLAP CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      OVERLAP CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      RGRID   SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
      INTEGER BDIR ! cartesian component of B-field, 1=x, 2=y, 3=z
      COMPLEX(qs), OPTIONAL :: RPHI(:,:,:,:,:)

      ! local variables
      TYPE (wavespin) WGKPQ ! G u_i v_k+q, k| u>
      TYPE (wavespin) WXI

      TYPE (wavedes), TARGET  :: WDES_kpq

      TYPE (nonl_struct) NONL_D
      TYPE (nonlr_struct), TARGET :: NONLR_D

      TYPE (nonl_struct) NONL_S_tmp

      INTEGER IDIR ! cartesian component of k-point shift
      INTEGER VDIR ! cartesian component of velocity operator, 1=x, 2=y, 3=z
      INTEGER NK,ISP,IQ
      INTEGER N,NELM
      INTEGER NSIM
      REAL(q) DISPL(3),DISPL_CART(3) 
      INTEGER ICOUEV
      REAL(q) RMS,DESUM1,TOTEN
      REAL(q) WSCAL

      TYPE (potcar), POINTER :: PP

      COMPLEX(q) NAB_QQ(3)
      INTEGER ICART_K_DIR
      INTEGER IQDIR, JQDIR
      CHARACTER(2) :: DIR_TEXT(3)=(/"BX","BY","BZ"/)

      REAL(q) STENCIL(7,3)
      DATA STENCIL /  1.0_q,            -2.0_q,           1.0_q,  0.0_q,           0.0_q,             0.0_q,  0.0_q, &
                     -0.083333333333_q,  1.3333333333_q, -2.5_q,  1.3333333333_q, -0.083333333333_q,  0.0_q,  0.0_q, &
                      0.011111111111_q, -0.15_q,          1.5_q, -2.7222222222_q,  1.5_q,            -0.15_q, 0.011111111111_q /
#ifdef exact_diagonalisation
      INTEGER IFLAG
#endif

      NSIM=W%WDES%NSIM*2
#ifdef MPI
      NSIM=((W%WDES%NSIM*2+W%WDES%COMM_INTER%NCPU-1)/W%WDES%COMM_INTER%NCPU)*W%WDES%COMM_INTER%NCPU
#endif

      ! allocate before doubling the k-point grid
      CALL ALLOCW(W%WDES,WGKPQ)
      CALL ALLOCW(W%WDES,WXI)

      CALL CHECK_FULL_KPOINTS

      ! original number of k-points
      NKPTS_ORIG=W%WDES%NKPTS

      ! double the number of k-points
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0,W%WDES%VKPT(:,1:NKPTS_ORIG))
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0,W%WDES%VKPT(:,1:NKPTS_ORIG))
#endif
      CALL KPAR_SYNC_ALL(W%WDES,W)
      CALL RE_GEN_LAYOUT(GRID,W%WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
      CALL REALLOCATE_WAVE(W,GRID,W%WDES,NONL_S,T_INFO,P,LATT_CUR)

!     CALL SETWDES(W%WDES,WDES1,0)
!     CALL NEWWAV(W1, WDES1, .FALSE.)

      IF (W%WDES%NKPTS/= NKPTS_ORIG*2) THEN
         WRITE(*,*) 'MLR_B_QQ: ERROR: k-points are not properly doubled',NKPTS_ORIG,W%WDES%NKPTS
         STOP
      ENDIF

      ! WGKPQ and WXI will live at k+q, their WDES remains the same except WDES%NKPTS, WDES%VKPT,
      ! WDES%DATAKE should point to the upper panel of NKPTS_ORIG k-points
      WDES_kpq=W%WDES
      WDES_kpq%NKPTS=NKPTS_ORIG
      WDES_kpq%VKPT=>W%WDES%VKPT(:,NKPTS_ORIG+1:W%WDES%NKPTS)
      WDES_kpq%DATAKE=>W%WDES%DATAKE(:,:,NKPTS_ORIG+1:W%WDES%NKPTS)

      WGKPQ%WDES=>WDES_kpq; WXI%WDES=>WDES_kpq

      iloop: DO IDIR=1,3
         ! skip the component along the B-field
         IF (IDIR==BDIR) CYCLE iloop

         ! the direction of the velocity operator we need here is \hat{B}_bdir x \hat{q}_idir 
         VDIR=6-BDIR-IDIR

         ! setup  r | p_i > 
         IF (INFO%LREAL) THEN
!           NONLR_D=NONLR_S
            CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
            CALL NONLR_ALLOC_CRREXP(NONLR_D)
         ELSE
            CALL NONL_ALLOC_DER(NONL_S,NONL_D)
!           CALL SPHER_DER(GRID,NONL_D,P,W%WDES,LATT_CUR,VDIR)
         ENDIF

         DISPL_CART=0; DISPL_CART(IDIR)=DQ
         DISPL=DISPL_CART
         CALL KARDIR(1, DISPL(1), LATT_CUR%A)

         qloop: DO IQ=-3,3
            IF (IO%IU0>=0) WRITE(*,'("BDIR=",I2,2X,"IDIR=",I2,2X,"IQ=",I3)') BDIR,IDIR,IQ

            WGKPQ%CW=(0._q,0._q); WGKPQ%CPROJ=zero; WXI%CW=(0._q,0._q); WXI%CPROJ=zero

            ! set additional k-points to k+q
            DO NK=1,NKPTS_ORIG
               W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)+IQ*DISPL(:)
!              W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)
            ENDDO

            CALL SPHER_DER(GRID,NONL_D,P,W%WDES,LATT_CUR,VDIR)

            ! fill additional wave functions
            IF (PRESENT(RPHI)) THEN
               DO ISP=1,W%WDES%ISPIN
                  DO NK=1,NKPTS_ORIG
                     DO N=1,W%WDES%NBANDS
                        W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)+TPI*( &
                       &   RPHI(:,N,NK,ISP,1)*(0.0_q,1.0_q)*IQ*DISPL_CART(1)+ &
                       &   RPHI(:,N,NK,ISP,2)*(0.0_q,1.0_q)*IQ*DISPL_CART(2)+ &
                       &   RPHI(:,N,NK,ISP,3)*(0.0_q,1.0_q)*IQ*DISPL_CART(3))
                     ENDDO
                  ENDDO
               ENDDO
            ELSE
               DO ISP=1,W%WDES%ISPIN
                  DO NK=1,NKPTS_ORIG
                     DO N=1,W%WDES%NBANDS
                        W%CW(:,N,NKPTS_ORIG+NK,ISP)=W%CW(:,N,NK,ISP)
                     ENDDO
                  ENDDO
               ENDDO
            ENDIF           
            
            CALL SET_DATAKE(W%WDES,LATT_CUR%B)
            IF (NONLR_S%LREAL) THEN
               CALL RSPHER(GRID,NONLR_S,LATT_CUR)
            ELSE
               CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR,1)
            ENDIF
            NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)

            CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

!           IF (IQ/=0) THEN
#ifdef exact_diagonalisation
!              CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
!                          LMDIM,CDIJ,CQIJ,3,SV,T_INFO,P,IO%IU0,E%EXHF, &
!                          NKSTART=NKPTS_ORIG+1)
               CALL EDDIAG_EXACT(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
                           LMDIM,CDIJ,CQIJ,IFLAG,SV,T_INFO,P,IO%IU0,IO%IU6,E%EXHF)
#else
               ! let's do a bit of Davidson optimization
               W%FERTOT(:,NKPTS_ORIG+1:W%WDES%NKPTS,:)=W%FERTOT(:,1:NKPTS_ORIG,:)
               W%WDES%WTKPT(NKPTS_ORIG+1:W%WDES%NKPTS)=W%WDES%WTKPT(1:NKPTS_ORIG)

               TOTEN=0; INFO%IALGO=8
               DO NELM=1,NSTEPS_DAVIDSON
                  CALL EDDAV(HAMILTONIAN,P,GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,NSIM, &
                 &     LMDIM,CDIJ,CQIJ,RMS,DESUM1,ICOUEV,SV,E%EXHF,IO%IU6,IO%IU0, &
                 &     LDELAY=.FALSE.,LSUBROTI=.TRUE.,LEMPTY=.FALSE.,LHF=.TRUE., &
                 &     NKSTART=NKPTS_ORIG+1)
                  E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES,W)

                  IF (IO%IU0>=0) WRITE(17, 200)      NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
                  IF (IO%IU0>=0) WRITE(IO%IU0, 200)  NELM,E%EBANDSTR,E%EBANDSTR-TOTEN,DESUM1,ICOUEV,RMS
                  TOTEN=E%EBANDSTR


 200              FORMAT('DAV: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5, &
                       &       I6,'  ',E10.3)
              
                  CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ)

                  IF (ABS(DESUM1) < ABS(INFO%EDIFF) .AND. NELM>1) EXIT
               ENDDO
#endif
!           ENDIF
            ! let's start the linear response stuff to get the contribution | psi^(1e) >
            CALL MLR_PSI_RESPONSE_EMPTY(W,GRID,P,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CDIJ,CQIJ,SV,INFO,IO,VDIR,WXI,WGKPQ)

            ! here we should get the contributions Q(q)
            CALL NABIJ_ASYM(W, WGKPQ, GRID, LATT_CUR, NKPTS_ORIG, INFO, NAB_QQ)
            DO ICART_K_DIR=1,3
               IQDIR =6-IDIR-ICART_K_DIR
               JQDIR =6-IDIR-VDIR
               QQ(IQDIR,JQDIR)=QQ(IQDIR,JQDIR)+ &
              &   NAB_QQ(ICART_K_DIR)*EIJK(IQDIR,IDIR,ICART_K_DIR)*EIJK(JQDIR,IDIR,VDIR)*STENCIL(IQ+4,3)
            ENDDO

         ENDDO qloop

         ! some deallocation
         IF (INFO%LREAL) THEN
            CALL NONLR_DEALLOC_CRREXP(NONLR_D)
         ELSE
            CALL NONL_DEALLOC_DER(NONL_D)
         ENDIF

      ENDDO iloop

      CALL DEALLOCW(WGKPQ)
      CALL DEALLOCW(WXI)

      ! restore to the original number of k-points
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#endif
      CALL RE_GEN_LAYOUT(GRID,W%WDES,KPOINTS,LATT_CUR,LATT_INI,IO%IU6,IO%IU0)
      CALL REALLOCATE_WAVE(W,GRID,W%WDES,NONL_S,T_INFO,P,LATT_CUR)     

      RETURN
      END SUBROUTINE MLR_B_QQ


!************************ SUBROUTINE MLR_PSI_RESPONSE_EMPTY ************
!
!***********************************************************************
      SUBROUTINE MLR_PSI_RESPONSE_EMPTY( &
     &   W0,GRID,P,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CDIJ,CQIJ,SV,INFO,IO,IDIR,WXI,W1)
      USE prec
      USE base
      USE wave_high
      USE nonl_high
      USE lattice
      USE mgrid
      USE pseudo
      USE rmm_diis_mlr
      USE hamil
      USE vaspxml
      USE ini
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (potcar) P(:)
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (nonl_struct) NONL_D            ! k derivative of projector in real space
      TYPE (nonlr_struct) NONLR_D          ! projector times r (only CRREXP distinct)
      TYPE (wavespin) WXI                  ! stores H(1) - epsilon S(1) | phi_0>
      TYPE (wavespin) W1                   ! first order change of wavefunction
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (latt) LATT_CUR
      INTEGER LMDIM
      OVERLAP CDIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ), &
              CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      RGRID   SV(DIMREAL(GRID%MPLWV),W0%WDES%NCDIJ)
      INTEGER IDIR ! component of velocity operator 1=x, 2=y, 3=z
      ! local variables
      TYPE (wavespin) W0_tmp
      TYPE (nonl_struct) NONL_S_tmp

      COMPLEX(q), POINTER :: SW0(:,:,:,:)=>NULL()

      INTEGER N
      REAL(q) TOTENL,TOTEN,DE

      INTEGER ICOUEV,IERROR
      REAL(q) DESUM,RMS(INFO%NELM),CSHIFT
      REAL(q) EBREAK_STORE
      LOGICAL LRESET

      ! calculate  v_k+q,k | u_nk >
      CALL MLR_COMMUTATOR(W0,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CDIJ,CQIJ,INFO,IDIR,WXI)
#ifdef apply_greens_function
      CALL MLR_APPLY_GREENS_FUNC(W0,WXI,W1)
#else
#ifdef provide_w1
      CALL MLR_APPLY_GREENS_FUNC(W0,WXI,W1)
#endif
      ! project onto empty subspace | xi_k+q > = P_e v_k+q,k | u_nk >
!     CALL START_TIMING("PROJ")
!     CALL MLR_PROJ_IN_EMPTY_SUBSPACE(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
      CALL MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
!     CALL STOP_TIMING("PROJ",IO%IU6)

!     we need to trick LINEAR_RESPONSE_DIIS to work with the second panel of NKPTS_ORIG k-points
!     as far as some quantities are concerned (WDES%DATAKE, WDES%VKPT, W0%CW and NONL_S), but for
!     instance with eigenvalues from the lower panel (e(0)_k S_k+q - H_k+q)|u(1)_k+q> = |xi_k+q>,
!     (W0%CELTOT, and W0%CELEN).
      W0_tmp=W0
!     W0_tmp%WDES%NKPTS=NKPTS_ORIG
!     W0_tmp%WDES%VKPT=>W0%WDES%VKPT(:,NKPTS_ORIG+1:W0%WDES%NKPTS)
!     W0_tmp%WDES%DATAKE=>W0%WDES%DATAKE(:,:,NKPTS_ORIG+1:W0%WDES%NKPTS)
      W0_tmp%WDES=>W1%WDES

      W0_tmp%CW=>W0%CW(:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)
      W0_tmp%CPROJ=>W0_tmp%CPROJ(:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)

      NONL_S_tmp=NONL_S; NONL_S_tmp%NK=-1
      NONL_S_tmp%QPROJ=>NONL_S%QPROJ(:,:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)

      NONL_S_tmp%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S_tmp,WXI)

      EBREAK_STORE=INFO%EBREAK; INFO%EBREAK=INFO%EDIFF*0.25_q
      W0_tmp%OVER_BAND=.FALSE.

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,142)
         WRITE(17,142)
      ENDIF
  142 FORMAT('       N       E                     dE             ' &
           ,'d eps       ncg     rms')


  130 FORMAT (5X, //, &
       &'----------------------------------------------------', &
       &'----------------------------------------------------'//)

  140 FORMAT (5X, //, &
       &'----------------------------------------- Iteration ', &
       &I4,'(',I4,')  ---------------------------------------'//)

!=======================================================================
!  main selfconsistent loop for linear response
!=======================================================================
      TOTEN=0

      CSHIFT=0._q
      LRESET=.TRUE.
#ifdef provide_w1
      NONL_S_tmp%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S_tmp,W1)
      LRESET=.FALSE.; W1%CW=-W1%CW; W1%CPROJ=-W1%CPROJ
#endif
      iter: DO N=1,INFO%NELM
         CALL START_TIMING("LOOP")

         IF(IO%IU6>=0) WRITE(IO%IU6,140) IDIR,N
         CALL XML_TAG("scstep")

         CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S_tmp,W1,WXI,W0_tmp,W0_tmp%WDES, &
        &   LMDIM,CDIJ,CQIJ,RMS(N),DESUM,ICOUEV,SV,CSHIFT,IO%IU6,IO%IU0,LRESET,IERROR,SW0) 
!        CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S_tmp,W1,WXI,W0_tmp,W0_tmp%WDES, &
!       &   LMDIM,CDIJ,CQIJ,RMS(N),DESUM,ICOUEV,SV,CSHIFT,IO%IU6,IO%IU0,LRESET,IERROR)

         CALL MRG_CEL(W1%WDES,W1)
         LRESET=.FALSE.
         ! break criterion?
         W1%FERTOT(:,1:NKPTS_ORIG,:)=W0%FERTOT(:,1:NKPTS_ORIG,:)
         TOTENL=TOTEN
         TOTEN =BANDSTRUCTURE_ENERGY(W1%WDES,W1)
         DE= (TOTEN-TOTENL)

         IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,200) N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
            WRITE(17,200)     N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
         ENDIF
 200     FORMAT('RMM: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)
!=======================================================================
! total time used for this step
!=======================================================================
         CALL SEPERATOR_TIMING(IO%IU6)
         CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')

         INFO%LABORT=.FALSE.
         IF(ABS(DESUM)<INFO%EDIFF.AND.ABS(DE)<INFO%EDIFF) INFO%LABORT=.TRUE.
!        IF(N>=3) THEN
!           IF (ABS((RMS(N)-RMS(N-1))/RMS(N))<1E-1 .AND. ABS((RMS(N)-RMS(N-2))/RMS(N))<1E-1) INFO%LABORT=.TRUE.
!        ENDIF

         IF (IO%IU6>=0)  THEN
            WRITE(IO%IU6,210) TOTEN

210   FORMAT(/ &
             '  free energy    TOTEN  = ',F18.8,' eV'/ &
             '  ---------------------------------------------------')

            IF (IO%LOPEN) CALL WFORCE(IO%IU6)
            IF (IO%LOPEN) CALL WFORCE(17)
         ENDIF

!=======================================================================
!  xml related output
!=======================================================================
         CALL XML_TAG("energy")
         IF (INFO%LABORT .OR. N==1) THEN
            CALL XML_ENERGY(TOTEN, TOTEN, TOTEN)
         ELSE
            CALL XML_ENERGY(TOTEN, TOTEN, TOTEN)
         ENDIF
         CALL XML_CLOSE_TAG

         CALL XML_CLOSE_TAG("scstep")


         IF (INFO%LABORT) THEN
            IF (IO%IU6>=0)  THEN

               WRITE(IO%IU6,131)
131       FORMAT (5X, //, &
       &  '------------------------ aborting loop because EDIFF', &
       &  ' is reached ----------------------------------------'//)
            ENDIF
            EXIT  iter
         ENDIF

      ENDDO iter

      CALL REDIS_PW_OVER_BANDS(W0_tmp%WDES,W0_tmp)
      DEALLOCATE(SW0); NULLIFY(SW0)

      INFO%EBREAK=EBREAK_STORE

      ! a bit heuristically
      W1%CW=-W1%CW
#endif

      RETURN
      END SUBROUTINE MLR_PSI_RESPONSE_EMPTY


!************************ SUBROUTINE MLR_PSI_RESPONSE_OCC **************
!
!***********************************************************************
      SUBROUTINE MLR_PSI_RESPONSE_OCC( &
     & W0,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CQIJ,IO,INFO,IDIR,WXI,W1)
      USE prec
      USE base
      USE lattice
      USE wave_high
      USE nonl_high
      USE mgrid
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (nonl_struct) NONL_D            ! k derivative of projector in real space
      TYPE (nonlr_struct) NONLR_D          ! projector times r (only CRREXP distinct)
      TYPE (wavespin) WXI                  ! stores -i [r_idir,S_k+q,k] |u(0)_nk>
      TYPE (wavespin) W1
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (latt) LATT_CUR
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      INTEGER IDIR ! component of velocity operator 1=x, 2=y, 3=z
      ! local variables

      ! compute (1/i) [r_idir,S_k+q,k] |u(0)_nk> and store it in WXI
      CALL MLR_COMMUTATOR_RQ(W0,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CQIJ,INFO,IDIR,WXI)

      ! add - \sum_j | u(0)_j,k+q > < u(0)_j,k+q | (1/i) [r_idir,S_k+q,k] |u(0)_nk> to  | u(1)_n,k+q >
      ! where j runs over all occupied states
!     CALL MLR_PROJ_IN_OCC_SUBSPACE(W0,WXI,W1)
      CALL MLR_PROJ_IN_OCC_SUBSPACE_BLAS(W0,WXI,W1)
      
      RETURN
      END SUBROUTINE MLR_PSI_RESPONSE_OCC


!************************ SUBROUTINE MLR_PSI_RESPONSE_LQ ***************
!
!***********************************************************************
      SUBROUTINE MLR_PSI_RESPONSE_LQ( &
     &   W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CDIJ,CQIJ,CLQIJ,SV,INFO,IO,WXI,W1)
      USE prec
      USE base
      USE mgrid
      USE wave_high
      USE nonl_high
      USE lattice
      USE rmm_diis_mlr
      USE hamil
      USE vaspxml
      USE ini
      USE constant
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (wavespin) WXI
      TYPE (wavespin) W1
      INTEGER LMDIM
      OVERLAP CDIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ), &
              CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      COMPLEX(q) CLQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      RGRID   SV(DIMREAL(GRID%MPLWV),W0%WDES%NCDIJ)
      ! local variables 
      TYPE (wavespin) W0_tmp

      COMPLEX(q), POINTER :: SW0(:,:,:,:)=>NULL()

      INTEGER N
      REAL(q) TOTENL,TOTEN,DE

      INTEGER ICOUEV,IERROR
      REAL(q) DESUM,RMS(INFO%NELM),CSHIFT
      REAL(q) EBREAK_STORE
      LOGICAL LRESET

      ! calculate L_R Q_R | u_nk >
      CALL MLR_LQ_PSI(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CLQIJ,INFO,WXI)
! unit conversion: perturbation entering the Greens function has same units as with S
      WXI%CW=WXI%CW*AUTOA*AUTOA*RYTOEV*2._q

#ifdef apply_greens_function
      CALL MLR_APPLY_GREENS_FUNC(W0,WXI,W1)
#else
#ifdef provide_w1
      CALL MLR_APPLY_GREENS_FUNC(W0,WXI,W1)
#endif
      ! project onto empty subspace | xi_k > = P_e L_R Q_R | u_nk >
!     CALL MLR_PROJ_IN_EMPTY_SUBSPACE(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
      CALL MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)

      W0_tmp=W0
!     W0_tmp%WDES%NKPTS=W1%WDES%NKPTS
      W0_tmp%WDES=>W1%WDES

      NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,WXI)

      EBREAK_STORE=INFO%EBREAK; INFO%EBREAK=INFO%EDIFF*0.25_q
      W0_tmp%OVER_BAND=.FALSE.

      IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,142)
         WRITE(17,142)
      ENDIF
  142 FORMAT('       N       E                     dE             ' &
           ,'d eps       ncg     rms')


  130 FORMAT (5X, //, &
       &'----------------------------------------------------', &
       &'----------------------------------------------------'//)

  140 FORMAT (5X, //, &
       &'----------------------------------------- Iteration ', &
       &'(',I4,')  ---------------------------------------'//)

      TOTEN=0

      CSHIFT=0._q
      LRESET=.TRUE.
#ifdef provide_w1
      NONL_S%NK=-1; CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W1)
      LRESET=.FALSE.; W1%CW=-W1%CW; W1%CPROJ=-W1%CPROJ
#endif
      iter: DO N=1,INFO%NELM
         CALL START_TIMING("LOOP")

         IF(IO%IU6>=0) WRITE(IO%IU6,140) N
         CALL XML_TAG("scstep")

         CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W1,WXI,W0_tmp,W0_tmp%WDES, &
        &   LMDIM,CDIJ,CQIJ,RMS(N),DESUM,ICOUEV,SV,CSHIFT,IO%IU6,IO%IU0,LRESET,IERROR,SW0) 

         CALL MRG_CEL(W1%WDES,W1)

         LRESET=.FALSE.
         ! break criterion?
         W1%FERTOT(:,1:NKPTS_ORIG,:)=W0%FERTOT(:,1:NKPTS_ORIG,:)
         TOTENL=TOTEN
         TOTEN =BANDSTRUCTURE_ENERGY(W1%WDES,W1)
         DE= (TOTEN-TOTENL)

         IF (IO%IU0>=0) THEN
            WRITE(IO%IU0,200) N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
            WRITE(17,200)     N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
         ENDIF
 200     FORMAT('RLQ: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)
!=======================================================================
! total time used for this step
!=======================================================================
         CALL SEPERATOR_TIMING(IO%IU6)
         CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')

         INFO%LABORT=.FALSE.
         IF(ABS(DESUM)<INFO%EDIFF.AND.ABS(DE)<INFO%EDIFF) INFO%LABORT=.TRUE.
!        IF(N>=3) THEN
!           IF (ABS((RMS(N)-RMS(N-1))/RMS(N))<1E-1 .AND. ABS((RMS(N)-RMS(N-2))/RMS(N))<1E-1) INFO%LABORT=.TRUE.
!        ENDIF

         IF (IO%IU6>=0)  THEN
            WRITE(IO%IU6,210) TOTEN

210   FORMAT(/ &
             '  free energy    TOTEN  = ',F18.8,' eV'/ &
             '  ---------------------------------------------------')

            IF (IO%LOPEN) CALL WFORCE(IO%IU6)
            IF (IO%LOPEN) CALL WFORCE(17)
         ENDIF

!=======================================================================
!  xml related output
!=======================================================================
         CALL XML_TAG("energy")
         IF (INFO%LABORT .OR. N==1) THEN
            CALL XML_ENERGY(TOTEN, TOTEN, TOTEN)
         ELSE
            CALL XML_ENERGY(TOTEN, TOTEN, TOTEN)
         ENDIF
         CALL XML_CLOSE_TAG

         CALL XML_CLOSE_TAG("scstep")


         IF (INFO%LABORT) THEN
            IF (IO%IU6>=0)  THEN

               WRITE(IO%IU6,131)
131       FORMAT (5X, //, &
       &  '------------------------ aborting loop because EDIFF', &
       &  ' is reached ----------------------------------------'//)
            ENDIF
            EXIT  iter
         ENDIF

      ENDDO iter

      CALL REDIS_PW_OVER_BANDS(W0_tmp%WDES,W0_tmp)
      DEALLOCATE(SW0); NULLIFY(SW0)

      INFO%EBREAK=EBREAK_STORE

      ! a bit heuristically
      W1%CW=-W1%CW
#endif

      RETURN
      END SUBROUTINE MLR_PSI_RESPONSE_LQ 


!************************ SUBROUTINE MLR_CALC_CLQIJ ********************
!
!***********************************************************************
      SUBROUTINE MLR_CALC_CLQIJ(WDES,P,T_INFO,LMDIM,IDIR,CLQIJ)
      USE prec
      USE pseudo
      USE wave_high
      USE poscar 
      USE relativistic
      USE radial
      USE constant
      USE morbitalmag, ONLY : LZORA
      IMPLICIT NONE
      TYPE (wavedes) WDES
      TYPE (type_info) T_INFO
      TYPE (potcar) P(T_INFO%NTYP)
      INTEGER LMDIM
      COMPLEX(q) CLQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      INTEGER IDIR
      ! local variables
      TYPE (potcar), POINTER :: PP
      INTEGER NI,NT,NIP
      INTEGER LYMAX,LDIMP,L,LL,LLP
      INTEGER LM,LMP,M,MP,CH1,CH2
      COMPLEX(q), ALLOCATABLE :: L_OP_R(:,:,:,:),DUMMY(:,:,:)

      INTEGER, EXTERNAL :: MAXL1

      REAL(q), ALLOCATABLE :: POT(:),KQ(:)
      REAL(q) RDUM,KQION

      CLQIJ=0

      ion: DO NI=1,T_INFO%NIONS
         NT=T_INFO%ITYP(NI)

         ! cycle if this ion is not treated by this node
         NIP=NI_LOCAL(NI,WDES%COMM_INB)
         IF (NIP==0) CYCLE ion

         PP=>PP_POINTER(P, NI, NT)

         IF (LZORA) THEN
            ALLOCATE(POT(PP%R%NMAX),KQ(PP%R%NMAX))
            POT=0._q
            ! compute V_H[n^a_c]
            CALL RAD_POT_HAR(0,PP%R,POT,PP%RHOAE,RDUM)
            ! add -Z/r
            POT=POT/(2._q*SQRT(PI))-(PP%ZCORE+PP%ZVALF_ORIG)*FELECT/PP%R%R
            ! add V_H[n^a_v] + V_xc[n^a_v+n^a_c]
            POT=POT-PP%POTAE
            ! and make sure this AE-"atomic reference"-potential lines up exactly with
            ! the potential that was constructed by the pseudopotential generation code.
            POT=POT+(PP%VPSRMAX-POT(PP%R%NMAX))
!           WRITE(1001,'(F14.7,4X,2F22.10)') (PP%R%R(M),POT(M),-PP%POTPS(M)+PP%POTPSC(M),M=1,SIZE(PP%POTAE))
         ENDIF

         ! cycle if this ion has no depletion charge
         IF (PP%PSDMAX==0) CYCLE ion
         LYMAX=MAXL1(PP)

         LDIMP=3
         IF (LYMAX>2) LDIMP=LYMAX+1

         ALLOCATE(L_OP_R(2*LDIMP+1,2*LDIMP+1,3,LDIMP),DUMMY(2*LDIMP+1,2*LDIMP+1,4))
         L_OP_R=(0._q,0._q)
         DO L=1,LDIMP
            CALL SETUP_LS(L,0._q,0._q,L_OP_R(1:2*L+1,1:2*L+1,1:3,L),DUMMY(1:2*L+1,1:2*L+1,1:4))
         ENDDO

         LM=1
         DO CH1=1,PP%LMAX
         LMP=1
         DO CH2=1,PP%LMAX
!           IF (PP%NDEP(CH1,CH2)==0) GOTO 510 
            LL =PP%LPS(CH1)
            LLP=PP%LPS(CH2)
            IF (LL == LLP .AND. LL>0 .AND. LL<=LDIMP ) THEN
               IF (LZORA) THEN
                  ! ZORA: radially variant relativistic mass correction 1/(1-V(r)/2/c^2) in Hartree atomic units,
                  ! in accordance with Eq. A13 of Yates et al., J. Chem. Phys. 118, 5746 (2003).
                  KQ(:)=1._q/(1._q-POT(:)/(2._q*RYTOEV)/CLIGHT/CLIGHT/2._q)*PP%WAE(:,CH1)*PP%WAE(:,CH2)-PP%WPS(:,CH1)*PP%WPS(:,CH2)
                  CALL SIMPI(PP%R,KQ,KQION)
!                 WRITE(*,*) 'QION=',PP%QION(CH1,CH2),'KQION=',KQION
               ELSE
                  KQION=PP%QION(CH1,CH2)
               ENDIF

               DO M =1,2*LL+1
               DO MP=1,2*LLP+1
                  CLQIJ(LMP+MP-1,LM+M-1,NIP,1)=CLQIJ(LMP+MP-1,LM+M-1,NIP,1)+L_OP_R(M,MP,IDIR,LL)*KQION
               ENDDO
               ENDDO
            ENDIF
 510        LMP=LMP+(2*LLP+1)
         ENDDO
         LM=LM+(2*LL+1)
         ENDDO

         DEALLOCATE(L_OP_R,DUMMY)
         IF (LZORA) DEALLOCATE(POT,KQ)
      ENDDO ion

      IF (WDES%ISPIN==2) CLQIJ(:,:,:,2)=CLQIJ(:,:,:,1)
      IF (WDES%NCDIJ==4) CLQIJ(:,:,:,4)=CLQIJ(:,:,:,1)

      RETURN
      END SUBROUTINE MLR_CALC_CLQIJ


!************************ SUBROUTINE MLR_LQ_PSI ************************
!
!***********************************************************************
      SUBROUTINE MLR_LQ_PSI(W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CLQIJ,INFO,WXI)
      USE prec
      USE base
      USE mgrid
      USE wave_high
      USE nonl_high
      USE lattice
      USE dfast
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      TYPE (wavespin) WXI
      INTEGER LMDIM
      COMPLEX(q) CLQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      ! local variables
      TYPE (wavefun1) W0_1(W0%WDES%NSIM)  ! current wavefunction
      TYPE (wavedes1) WDES1

      COMPLEX(q),ALLOCATABLE :: CH(:,:),CWORK1(:,:)

      INTEGER ISP,NK,ISPINOR,M,MM
      INTEGER NSIM,NGVECTOR
      INTEGER NB(W0%WDES%NSIM),NB_DONE,NP,N
      LOGICAL LDO(W0%WDES%NSIM)
      REAL(q) DUMMY(W0%WDES%NSIM)

      LOGICAL LSTOP

      NSIM=W0%WDES%NSIM
      DUMMY=0._q

      ALLOCATE(CH(W0%WDES%NRPLWV,NSIM))

      IF (INFO%LREAL) THEN
         DO NP=1,NSIM
            ALLOCATE(W0_1(NP)%CR(GRID%MPLWV*W0%WDES%NRSPINORS))
         ENDDO
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSIM))
      ENDIF

      WXI%CPROJ=0

!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,WXI%WDES%NKPTS
#ifdef MPI   
         IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================     
         CALL SETWDES(W0%WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

         NGVECTOR=WDES1%NGVECTOR

         IF (INFO%LREAL) THEN
            CALL PHASER (GRID,LATT_CUR,NONLR_S,NK,W0%WDES)
         ELSE
            NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK)
         ENDIF 

         NB_DONE=0  ! index of the bands allready optimised
         bands: DO
            NB=0    ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
            newband: DO NP=1,NSIM
               IF (NB_DONE < WXI%WDES%NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  NB(NP)=NB_DONE

                  CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

                  IF (INFO%LREAL) THEN
                     ! fft to real space
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
                     ENDDO
                  ENDIF
               ENDIF
            ENDDO newband
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
            LSTOP=.TRUE.
            LDO  =.FALSE.
            DO NP=1,NSIM
               IF (NB(NP)/=0) THEN
                  LSTOP  =.FALSE.
                  LDO(NP)=.TRUE.     ! band not finished yet
               ENDIF
            ENDDO
            IF (LSTOP) EXIT bands

            CH=(0._q,0._q)

            ! contribution | p_i,k > LQ_ij < p_j,k | u_nk >
            IF (INFO%LREAL) THEN
               CWORK1=0
               CALL RACCMU_CCDIJ(NONLR_S,WDES1,W0_1,LMDIM,CLQIJ,CLQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSIM,LDO)
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
                     ENDDO
                  ENDIF
               ENDDO
            ELSE
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     CALL VNLACC_ADD_CCDIJ(NONL_S,W0_1(NP),CLQIJ,CLQIJ,1,DUMMY(NP),CH(:,NP))
                  ENDIF
               ENDDO
            ENDIF

            ! store in WXI
            DO NP=1,NSIM
               N=NB(NP); IF (.NOT.LDO(NP)) CYCLE
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,NGVECTOR
                     MM=M+ISPINOR*NGVECTOR
                     WXI%CW(MM,N,NK,ISP)=CH(MM,NP) 
                  ENDDO
               ENDDO             
            ENDDO

         ENDDO bands
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      RETURN
      END SUBROUTINE MLR_LQ_PSI


!************************ SUBROUTINE MLR_COMMUTATOR ********************
!
!***********************************************************************
      SUBROUTINE MLR_COMMUTATOR( &
     &   W0,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CDIJ,CQIJ,INFO,IDIR,WXI)
      USE prec
      USE base
      USE wave_high
      USE nonl_high
      USE lattice
      USE mgrid
      USE constant
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_D
      TYPE (nonlr_struct) NONLR_D          ! projector times r (only CRREXP distinct)
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (latt) LATT_CUR
      INTEGER LMDIM
      OVERLAP CDIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ), &
              CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      INTEGER IDIR ! component of velocity operator 1=x, 2=y, 3=z
      TYPE (wavespin) WXI                  ! stores H(1) - epsilon S(1) | phi_0>
      ! local variables
      TYPE (wavedes1)  WDES1               ! descriptor for one k-point
      TYPE (wavefun1)  W0_1(W0%WDES%NSIM)  ! current wavefunction
      TYPE (nonl_struct) NONL_S_tmp
      TYPE (nonl_struct) NONL_D_tmp

      REAL(q), ALLOCATABLE :: GC(:)
      COMPLEX(q),ALLOCATABLE :: CH(:,:),CWORK1(:,:)
      GDEF, TARGET, ALLOCATABLE ::  CPROJ(:,:)

      INTEGER ISP,NK,M,MM,ISPINOR
      INTEGER NSIM,NGVECTOR
      INTEGER NB(W0%WDES%NSIM),NB_DONE,NP,N
      LOGICAL LDO(W0%WDES%NSIM) 
      LOGICAL LSTOP

      REAL(q) EVALUE0(W0%WDES%NSIM)
      REAL(q) G1,G2,G3

      NSIM=W0%WDES%NSIM

      ALLOCATE(CH(W0%WDES%NRPLWV,NSIM),GC(W0%WDES%NRPLWV),CPROJ(W0%WDES%NPROD,NSIM))

      IF (INFO%LREAL) THEN
         DO NP=1,NSIM
            ALLOCATE(W0_1(NP)%CR(GRID%MPLWV*W0%WDES%NRSPINORS))
         ENDDO
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSIM))
      ENDIF

      WXI%CPROJ=0

      spin:    DO ISP=1,W0%WDES%ISPIN 
! We have wave functions at k and k+q, the first set have the k-point index running from 1,..,NKPTS/2,
! the latter from NKPTS/2+1,2*NKPTS 
      kpoints: DO NK=1,NKPTS_ORIG  ! loop over k
#ifdef MPI   
         IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CALL SETWDES(W0%WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

         NGVECTOR=WDES1%NGVECTOR

         IF (INFO%LREAL) THEN
            CALL PHASER (GRID,LATT_CUR,NONLR_S,NK+NKPTS_ORIG,W0%WDES); NONLR_S%NK=NK
            CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK,W0%WDES,IDIR)
         ELSE
            NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK)
            NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            NONL_S_tmp=NONL_S
            NONL_S_tmp%QPROJ=>NONL_S%QPROJ(:,:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)
         ENDIF

         NB_DONE=0  ! index of the bands allready optimised
        bands: DO
            NB=0    ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
            newband: DO NP=1,NSIM
               IF (NB_DONE < WXI%WDES%NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  NB(NP)=NB_DONE

                  CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

                  IF (INFO%LREAL) THEN
                     ! fft to real space
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
                     ENDDO
                  ENDIF
                  ! W0_1 points to elements of W0, we don't want to destroy them so we
                  ! create some space to hold the projection of the wave functions onto
                  ! the derivative of the projectors w.r.t. k, as used below.
                  W0_1(NP)%CPROJ => CPROJ(:,NP)

                  EVALUE0(NP)=W0_1(NP)%CELEN
               ENDIF
            ENDDO newband
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
            LSTOP=.TRUE.
            LDO  =.FALSE.
            DO NP=1,NSIM
               IF (NB(NP)/=0) THEN
                  LSTOP  =.FALSE.
                  LDO(NP)=.TRUE.     ! band not finished yet
               ENDIF
            ENDDO
            IF (LSTOP) EXIT bands

            !  -i\nabla + k | u_nk > = hbar^2/ m_e (G+k) C_G | u_nk >
            DO M=1,NGVECTOR
               G1=WDES1%IGX(M)+W0%WDES%VKPT(1,NK)
               G2=WDES1%IGY(M)+W0%WDES%VKPT(2,NK)
               G3=WDES1%IGZ(M)+W0%WDES%VKPT(3,NK)
               GC(M)=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))*(TPI*(2*HSQDTM))
            ENDDO

            CH=(0._q,0._q)

            DO NP=1,NSIM
               IF (LDO(NP)) THEN
                  DO ISPINOR =0,WDES1%NRSPINORS-1
                     DO M=1,NGVECTOR
                        MM=M+ISPINOR*NGVECTOR
                        ! mind the factor -i here, it is removed lateron
                        CH(MM,NP)=CH(MM,NP)+(W0_1(NP)%CW(MM)*GC(M))*(0._q,-1._q)
                     ENDDO
                  ENDDO
               ENDIF
            ENDDO

            ! contribution   | p_i,k+q > [D_ij - e(0)Q_ij] < p_j,k|  r_idir | u_nk >
            IF (INFO%LREAL) THEN
               CWORK1=0
               CALL RPROMU(NONLR_D,WDES1,W0_1,NSIM,LDO)
               CALL RACCMU(NONLR_S,WDES1,W0_1,LMDIM,CDIJ,CQIJ,EVALUE0,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSIM,LDO)
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
                     ENDDO
                  ENDIF
               ENDDO
            ELSE
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     CALL PROJ1(NONL_D,WDES1,W0_1(NP)) 
                     CALL VNLACC_ADD(NONL_S_tmp,W0_1(NP),CDIJ,CQIJ,1,EVALUE0(NP),CH(:,NP))
                  ENDIF
               ENDDO
            ENDIF

            ! store in WXI
            DO NP=1,NSIM
               N=NB(NP); IF (.NOT.LDO(NP)) CYCLE
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,NGVECTOR
                     MM=M+ISPINOR*NGVECTOR
                     WXI%CW(MM,N,NK,ISP)=CH(MM,NP)*(0._q,1._q) 
                  ENDDO
                  ! we store  i< p_j,k | r_idir | u_nk >  in WXI%CPROJ
                  WXI%CPROJ(:,N,NK,ISP)=W0_1(NP)%CPROJ*(0._q,1._q)
               ENDDO             
            ENDDO
         ENDDO bands



         IF (INFO%LREAL) THEN
            CALL PHASER (GRID,LATT_CUR,NONLR_S,NK,W0%WDES)
            CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK+NKPTS_ORIG,W0%WDES,IDIR); NONLR_D%NK=NK
         ELSE
! Gilles and Martijn think this does not need to be recomputed
!           CALL PHASE(WDES,NONL_S,NK)
!           NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            NONL_D_tmp=NONL_D
            NONL_D_tmp%QPROJ=>NONL_D%QPROJ(:,:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)
         ENDIF


         NB_DONE=0  ! index of the bands allready optimised
         bands2: DO
            NB=0    ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
            newband2: DO NP=1,NSIM
               IF (NB_DONE < WXI%WDES%NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  NB(NP)=NB_DONE

                  CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

                  IF (INFO%LREAL) THEN
                     ! fft to real space
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
                     ENDDO
                  ENDIF

                  EVALUE0(NP)=W0_1(NP)%CELEN
               ENDIF
            ENDDO newband2
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
            LSTOP=.TRUE.
            LDO  =.FALSE.
            DO NP=1,NSIM
               IF (NB(NP)/=0) THEN
                  LSTOP  =.FALSE.
                  LDO(NP)=.TRUE.     ! band not finished yet
               ENDIF
            ENDDO
            IF (LSTOP) EXIT bands2

            CH=(0._q,0._q)

            ! contribution - r_idir | p_i,k+q > [D_ij - e(0)Q_ij] < p_j,k | u_nk >
            IF (INFO%LREAL) THEN
               CWORK1=0
               CALL RACCMU(NONLR_D,WDES1,W0_1,LMDIM,CDIJ,CQIJ,EVALUE0,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSIM,LDO)
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
                     ENDDO
                  ENDIF
               ENDDO
            ELSE
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     CALL VNLACC_ADD(NONL_D_tmp,W0_1(NP),CDIJ,CQIJ,1,EVALUE0(NP),CH(:,NP))
                  ENDIF
               ENDDO
            ENDIF

            ! store in WXI
            DO NP=1,NSIM
               N=NB(NP); IF (.NOT.LDO(NP)) CYCLE
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,NGVECTOR
                     MM=M+ISPINOR*NGVECTOR
                     WXI%CW(MM,N,NK,ISP)=WXI%CW(MM,N,NK,ISP)-CH(MM,NP)*(0._q,1._q) 
                  ENDDO
               ENDDO             
            ENDDO

         ENDDO bands2
      ENDDO kpoints
      ENDDO spin

      IF (INFO%LREAL) THEN
         DO NP=1,NSIM
            DEALLOCATE(W0_1(NP)%CR)
         ENDDO
         DEALLOCATE(CWORK1)
      ENDIF

      DEALLOCATE(CH,GC,CPROJ)

      RETURN
      END SUBROUTINE MLR_COMMUTATOR


!************************ SUBROUTINE MLR_COMMUTATOR_RQ *****************
!
!***********************************************************************
      SUBROUTINE MLR_COMMUTATOR_RQ( &
     &   W0,GRID,NONL_S,NONLR_S,NONL_D,NONLR_D,LATT_CUR,LMDIM,CQIJ,INFO,IDIR,WXI)
      USE prec
      USE base
      USE wave_high
      USE nonl_high
      USE lattice
      USE mgrid
      USE constant
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_D
      TYPE (nonlr_struct) NONLR_D          ! projector times r (only CRREXP distinct)
      TYPE (info_struct) INFO
      TYPE (in_struct) IO
      TYPE (latt) LATT_CUR
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      INTEGER IDIR ! component of velocity operator 1=x, 2=y, 3=z
      TYPE (wavespin) WXI                  ! S(1) | phi_0>
      ! local variables
      TYPE (wavedes1)  WDES1               ! descriptor for one k-point
      TYPE (wavefun1)  W0_1(W0%WDES%NSIM)  ! current wavefunction
      TYPE (nonl_struct) NONL_S_tmp
      TYPE (nonl_struct) NONL_D_tmp

      COMPLEX(q),ALLOCATABLE :: CH(:,:),CWORK1(:,:)
      GDEF, TARGET, ALLOCATABLE ::  CPROJ(:,:)

      INTEGER ISP,NK,M,MM,ISPINOR
      INTEGER NSIM,NGVECTOR
      INTEGER NB(W0%WDES%NSIM),NB_DONE,NP,N
      LOGICAL LDO(W0%WDES%NSIM) 
      LOGICAL LSTOP

      REAL(q) DUMMY(W0%WDES%NSIM)

      NSIM=W0%WDES%NSIM
      DUMMY=0._q

      ALLOCATE(CH(W0%WDES%NRPLWV,NSIM),CPROJ(W0%WDES%NPROD,NSIM))

      IF (INFO%LREAL) THEN
         DO NP=1,NSIM
            ALLOCATE(W0_1(NP)%CR(GRID%MPLWV*W0%WDES%NRSPINORS))
         ENDDO
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSIM))
      ENDIF

      WXI%CPROJ=0

      spin:    DO ISP=1,W0%WDES%ISPIN 
! We wave functions at k and k+q, the first set have the k-point index running from 1,..,NKPTS/2,
! the latter from NKPTS/2+1,2*NKPTS 
      kpoints: DO NK=1,NKPTS_ORIG  ! loop over k
#ifdef MPI   
         IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CALL SETWDES(W0%WDES,WDES1,NK); CALL SETWGRID_OLD(WDES1,GRID)

         NGVECTOR=WDES1%NGVECTOR

         IF (INFO%LREAL) THEN
            CALL PHASER (GRID,LATT_CUR,NONLR_S,NK+NKPTS_ORIG,W0%WDES); NONLR_S%NK=NK
            CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK,W0%WDES,IDIR)
         ELSE
            NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK)
            NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            NONL_S_tmp=NONL_S
            NONL_S_tmp%QPROJ=>NONL_S%QPROJ(:,:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:)
         ENDIF

         NB_DONE=0  ! index of the bands allready optimised
         bands: DO
            NB=0    ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
            newband: DO NP=1,NSIM
               IF (NB_DONE < WXI%WDES%NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  NB(NP)=NB_DONE

                  CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

                  IF (INFO%LREAL) THEN
                     ! fft to real space
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
                     ENDDO
                  ENDIF
                  ! W0_1 points to elements of W0, we don't want to destroy them so we
                  ! create some space to hold the projection of the wave functions onto
                  ! the derivative of the projectors w.r.t. k, as used below.
                  W0_1(NP)%CPROJ => CPROJ(:,NP)
               ENDIF
            ENDDO newband
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
            LSTOP=.TRUE.
            LDO  =.FALSE.
            DO NP=1,NSIM
               IF (NB(NP)/=0) THEN
                  LSTOP  =.FALSE.
                  LDO(NP)=.TRUE.     ! band not finished yet
               ENDIF
            ENDDO
            IF (LSTOP) EXIT bands

            CH=(0._q,0._q)

            ! contribution   | p_i,k+q > Q_ij < p_j,k|  r_idir | u_nk >
            IF (INFO%LREAL) THEN
               CWORK1=0
               CALL RPROMU(NONLR_D,WDES1,W0_1,NSIM,LDO)
               CALL RACCMU(NONLR_S,WDES1,W0_1,LMDIM,CQIJ,CQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSIM,LDO)
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
                     ENDDO
                  ENDIF
               ENDDO
            ELSE
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     CALL PROJ1(NONL_D,WDES1,W0_1(NP)) 
                     CALL VNLACC_ADD(NONL_S_tmp,W0_1(NP),CQIJ,CQIJ,1,DUMMY(NP),CH(:,NP))
                  ENDIF
               ENDDO
            ENDIF

            ! store in WXI
            DO NP=1,NSIM
               N=NB(NP); IF (.NOT.LDO(NP)) CYCLE
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,NGVECTOR
                     MM=M+ISPINOR*NGVECTOR
                     WXI%CW(MM,N,NK,ISP)=CH(MM,NP)*(0._q,1._q) 
                  ENDDO
                  ! we store  i< p_j,k | r_idir | u_nk >  in WXI%CPROJ
                  WXI%CPROJ(:,N,NK,ISP)=W0_1(NP)%CPROJ*(0._q,1._q)
               ENDDO             
            ENDDO
         ENDDO bands



         IF (INFO%LREAL) THEN
            CALL PHASER (GRID,LATT_CUR,NONLR_S,NK,W0%WDES)
            CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK+NKPTS_ORIG,W0%WDES,IDIR); NONLR_D%NK=NK
         ELSE
! Gilles and Martijn think this does not need to be recomputed
!           CALL PHASE(WDES,NONL_S,NK)
!           NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            NONL_D_tmp=NONL_D
            NONL_D_tmp%QPROJ=>NONL_D%QPROJ(:,:,:,NKPTS_ORIG+1:W0%WDES%NKPTS,:) 
         ENDIF


         NB_DONE=0  ! index of the bands allready optimised
         bands2: DO
            NB=0    ! empty the list of bands, which are optimized currently
!
!  check the NB list, whether there is any empty slot
!  fill in a not yet optimized wavefunction into the slot
!
            newband2: DO NP=1,NSIM
               IF (NB_DONE < WXI%WDES%NBANDS ) THEN
                  NB_DONE=NB_DONE+1
                  N     =NB_DONE
                  NB(NP)=NB_DONE

                  CALL SETWAV(W0,W0_1(NP),WDES1,N,ISP)  ! fill band N into W0_1(NP)

                  IF (INFO%LREAL) THEN
                     ! fft to real space
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W0_1(NP)%CR(1+ISPINOR*GRID%MPLWV),W0_1(NP)%CW(1+ISPINOR*NGVECTOR),GRID)
                     ENDDO
                  ENDIF
               ENDIF
            ENDDO newband2
!=======================================================================
! if the NB list is now empty end the bands DO loop
!=======================================================================
            LSTOP=.TRUE.
            LDO  =.FALSE.
            DO NP=1,NSIM
               IF (NB(NP)/=0) THEN
                  LSTOP  =.FALSE.
                  LDO(NP)=.TRUE.     ! band not finished yet
               ENDIF
            ENDDO
            IF (LSTOP) EXIT bands2

            CH=(0._q,0._q)

            ! contribution - r_idir | p_i,k+q > Q_ij < p_j,k | u_nk >
            IF (INFO%LREAL) THEN
               CWORK1=0
               CALL RACCMU(NONLR_D,WDES1,W0_1,LMDIM,CQIJ,CQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSIM,LDO)
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     DO ISPINOR=0,WDES1%NRSPINORS-1
                        CALL FFTEXT(NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,NP),CH(1+ISPINOR*NGVECTOR,NP),GRID,.TRUE.)
                     ENDDO
                  ENDIF
               ENDDO
            ELSE
               DO NP=1,NSIM
                  IF (LDO(NP)) THEN
                     CALL VNLACC_ADD(NONL_D_tmp,W0_1(NP),CQIJ,CQIJ,1,DUMMY(NP),CH(:,NP))
                  ENDIF
               ENDDO
            ENDIF

            ! store in WXI
            DO NP=1,NSIM
               N=NB(NP); IF (.NOT.LDO(NP)) CYCLE
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,NGVECTOR
                     MM=M+ISPINOR*NGVECTOR
                     WXI%CW(MM,N,NK,ISP)=WXI%CW(MM,N,NK,ISP)-CH(MM,NP)*(0._q,1._q) 
                  ENDDO
               ENDDO             
            ENDDO

         ENDDO bands2
      ENDDO kpoints
      ENDDO spin
! test
!     WXI%CW=WXI%CW*AUTOA
      WXI%CW=WXI%CW/2._q
! test

      IF (INFO%LREAL) THEN
         DO NP=1,NSIM
            DEALLOCATE(W0_1(NP)%CR)
         ENDDO
         DEALLOCATE(CWORK1)
      ENDIF

      DEALLOCATE(CH,CPROJ)

      RETURN
      END SUBROUTINE MLR_COMMUTATOR_RQ


!************************ SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE ********
!
!***********************************************************************
      SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE(&
     & W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
      USE prec
      USE base
      USE dfast      
      USE lattice
      USE mgrid
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) WXI
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      ! local variables
      TYPE (wavedes1) WDES1
      TYPE (wavefun1), ALLOCATABLE :: WTMP(:)

      INTEGER ISP,NK,NPOS,N,NB,M,MM,ISPINOR
      INTEGER NB_TOT,NBANDS,NB_START,NB_STOP,NSTRIP,NSTRIP_ACT

      REAL(q), ALLOCATABLE :: DUMMY(:)
      LOGICAL, ALLOCATABLE :: LDO(:)

      GDEF COVL
      COMPLEX(q), ALLOCATABLE :: CH(:,:),CWORK1(:,:)

      NB_TOT=W0%WDES%NB_TOT
      
      ! set NSTRIP between [1 and 32]
      NSTRIP=NSTRIP_STANDARD_GLOBAL

      CALL SETWDES(W0%WDES,WDES1,0)
      ALLOCATE(WTMP(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WTMP(N),WDES1,INFO%LREAL)
      ENDDO

      ALLOCATE(DUMMY(NSTRIP),LDO(NSTRIP))

      ALLOCATE(CH(W0%WDES%NRPLWV,NSTRIP))
      IF (INFO%LREAL) THEN
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSTRIP))
      ENDIF
 
!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

!-----------------------------------------------------------------------
! calculate |phi(1)_i> - \sum_j S|phi(0)_j><phi(0)_j|phi(1)_i>
! where j runs over occupied orbitals
!-----------------------------------------------------------------------

      IF (INFO%LREAL) THEN
         CALL PHASER(GRID,LATT_CUR,NONLR_S,NK+NKPTS_ORIG,W0%WDES)
      ELSE
         NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK+NKPTS_ORIG) 
      ENDIF

      DO NB_START=1,NB_TOT,NSTRIP
         NSTRIP_ACT=MIN(NSTRIP,NB_TOT-NB_START+1)
         NB_STOP=MIN(NB_START+NSTRIP-1,NB_TOT)
         ! collect the wave functions from NB_START to NB_STOP (global band index)
         ! and store into WTMP (for LREAL=.TRUE. WTMP(:)%CR is calculated as well).
         IF (INFO%LREAL) THEN
            CALL W1_GATHER_GLB(W0,NB_START,NB_STOP,ISP,WTMP)
         ELSE
            CALL W1_GATHER_GLB_NOCR(W0,NB_START,NB_STOP,ISP,WTMP)
         ENDIF

         CH=(0._q,0._q)
         DO N=1,NSTRIP_ACT
            CH(1:WDES1%NRSPINORS*WDES1%NGVECTOR,N)=WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
         ENDDO

         ! compute S|phi(0)> and store it in CH
         IF (INFO%LREAL) THEN
            CWORK1=0; LDO=.FALSE.; LDO(1:NSTRIP_ACT)=.TRUE.; DUMMY=0
            CALL RACCMU(NONLR_S,WDES1,WTMP,LMDIM,CQIJ,CQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSTRIP,LDO)
            DO N=1,NSTRIP_ACT
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  CALL FFTEXT(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,N),CH(1+ISPINOR*WDES1%NGVECTOR,N),GRID,.TRUE.)
               ENDDO
            ENDDO
         ELSE
            DO N=1,NSTRIP_ACT 
               CALL VNLACC_ADD(NONL_S,WTMP(N),CQIJ,CQIJ,1,0._q,CH(:,N))
            ENDDO
         ENDIF 

         ! subtract S|phi(0)_j><phi(0)_j|phi(1)_i> from |phi(1)_i>
         ! the latter are distributed over nodes
         DO NB=1,WXI%WDES%NBANDS
            DO N=1,NSTRIP_ACT
               IF (W0%FERTOT(NB_START+N-1,NK,ISP)<1E-4_q) CYCLE
               COVL=zero
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,WDES1%NGVECTOR
                     MM=M+WDES1%NGVECTOR*ISPINOR
                     COVL=COVL+GCONJG(WTMP(N)%CW(MM))*WXI%CW(MM,NB,NK,ISP)
                  ENDDO
               ENDDO               
! test
!              WRITE(70+W0%WDES%COMM%NODE_ME,'(2I4,2F14.7)') N+NB_START-1,W0%WDES%NB_LOW+(NB-1)*W0%WDES%NB_PAR,COVL
! test
               WXI%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)=WXI%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)- &
              &   COVL*CH(1:WDES1%NRSPINORS*WDES1%NGVECTOR,N)
            ENDDO
         ENDDO
      ENDDO
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      DO N=1,NSTRIP
         CALL DELWAV(WTMP(N),INFO%LREAL)
      ENDDO
      DEALLOCATE(WTMP,DUMMY,LDO,CH)
      IF (INFO%LREAL) DEALLOCATE(CWORK1)

      RETURN
      END SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE


!************************ SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS ***
!
!***********************************************************************
      SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS(&
     & W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
      USE prec
      USE base
      USE dfast
      USE lattice
      USE mgrid
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) WXI
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      ! local variables
      TYPE (wavedes1) WDES1
      TYPE (wavefun1), ALLOCATABLE :: WTMP(:)
      ! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW0_RED(:,:),CW1_RED(:,:),CH_RED(:,:)
      INTEGER NCPU,NRPLWV_RED,NPL,NPRO
      LOGICAL DO_REDIS

      INTEGER ISP,NK,ISPINOR
      INTEGER NB_TOT,NBANDS,NSTRIP,NSTRIP_ACT,NB_START,N
      INTEGER NPOSB1,NPOSB2,NSTRPB,NPOSPL,NSTRPL

      REAL(q), ALLOCATABLE :: DUMMY(:)
      LOGICAL, ALLOCATABLE :: LDO(:)

      COMPLEX(q), ALLOCATABLE, TARGET :: CH(:,:)
      COMPLEX(q), ALLOCATABLE :: CWORK1(:,:)

      GDEF,ALLOCATABLE,TARGET :: COVL(:,:)

#ifdef MPI
      ! number of procs involved in band distribution
      NCPU=W0%WDES%COMM_INTER%NCPU
#else
      NCPU=1
#endif
      IF (NCPU/=1) THEN
         DO_REDIS=.TRUE.
         NRPLWV_RED=W0%WDES%NRPLWV/NCPU
      ELSE
         DO_REDIS=.FALSE.
         NRPLWV_RED=W0%WDES%NRPLWV
      ENDIF

      NB_TOT=W0%WDES%NB_TOT
      NBANDS=W0%WDES%NBANDS

!     ! set NSTRIP between [1 and 32]
!     NSTRIP=NSTRIP_STANDARD_GLOBAL
      ! the above was much too large
      NSTRIP=MIN(4,NBANDS)

      CALL SETWDES(W0%WDES,WDES1,0)
      ALLOCATE(WTMP(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WTMP(N),WDES1,INFO%LREAL)
      ENDDO

      ALLOCATE(DUMMY(NSTRIP),LDO(NSTRIP))

      IF (INFO%LREAL) THEN
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSTRIP))
      ENDIF

      ALLOCATE(CH(W0%WDES%NRPLWV,NBANDS))
      ! get pointers for redistribution to "over-plane-wave coefficients"
      IF (NCPU/=1) THEN
         CALL SET_WPOINTER( CH_RED, NRPLWV_RED, NB_TOT, CH(1,1))
      ELSE
         CH_RED => CH(:,:)
      ENDIF

      ALLOCATE(COVL(NB_TOT,NB_TOT))
!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

      IF (INFO%LREAL) THEN
         CALL PHASER(GRID,LATT_CUR,NONLR_S,NK+NKPTS_ORIG,W0%WDES)
      ELSE
         NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK+NKPTS_ORIG) 
      ENDIF

      ! compute S|phi(0)> and store it in CH
      CH=(0._q,0._q)
      DO NB_START=1,NBANDS,NSTRIP
         NSTRIP_ACT=MIN(NSTRIP,NBANDS-NB_START+1)

         DO N=1,NSTRIP_ACT
            CALL W1_COPY(ELEMENT(W0,WDES1,NB_START+N-1,ISP),WTMP(N)); IF (INFO%LREAL) CALL FFTWAV_W1(WTMP(N))
            CH(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB_START+N-1)=WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
         ENDDO

         IF (INFO%LREAL) THEN
            CWORK1=0; LDO=.FALSE.; LDO(1:NSTRIP_ACT)=.TRUE.; DUMMY=0
            CALL RACCMU(NONLR_S,WDES1,WTMP,LMDIM,CQIJ,CQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSTRIP,LDO)
            DO N=1,NSTRIP_ACT
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  CALL FFTEXT(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,N),CH(1+ISPINOR*WDES1%NGVECTOR,NB_START+N-1),GRID,.TRUE.)
               ENDDO
            ENDDO
         ELSE
            DO N=1,NSTRIP_ACT 
               CALL VNLACC_ADD(NONL_S,WTMP(N),CQIJ,CQIJ,1,0._q,CH(:,NB_START+N-1))
            ENDDO
         ENDIF 
      ENDDO

      ! get pointers for redistribution to "over-plane-wave coefficients"
      IF (NCPU/=1) THEN
         CALL SET_WPOINTER(CW0_RED, NRPLWV_RED, NB_TOT,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL SET_WPOINTER(CW1_RED, NRPLWV_RED, NB_TOT, WXI%CW(1,1,NK,ISP))
      ELSE
         CW0_RED =>  W0%CW(:,:,NK+NKPTS_ORIG,ISP)
         CW1_RED => WXI%CW(:,:,NK,ISP)
      ENDIF

      ! set number of wavefunctions after redistribution
      NPL = WDES1%NPL
      NPRO= WDES1%NPRO
      CALLMPI(SET_NPL_NPRO(WDES1, NPL, NPRO))

      ! redistribute to "over-plane-wave coefficients"
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
         CALL REDIS_PW(WDES1, NBANDS,     CH(1,1)       )
      ENDIF

      ! calculate COVL(i,j)=<phi(0)_i|phi(1)_j>
      COVL=zero
      IF (NPL/=0) THEN
!     CALL GGEMM('C','N',NB_TOT,NB_TOT,m_ NPL,one,CW0_RED(1,1),m_ NRPLWV_RED,CW1_RED(1,1),m_ NRPLWV_RED,zero,COVL(1,1),NB_TOT)
!     NSTRPB=NSTRIP
      NSTRPB=NB_TOT
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSB2=1,NB_TOT,NSTRPB
      CALL GGEMM('C','N',MIN(NSTRPB,NB_TOT-NPOSB2+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),m_ NPL,one,CW0_RED(1,NPOSB2),m_ NRPLWV_RED,CW1_RED(1,NPOSB1),m_ NRPLWV_RED,zero,COVL(NPOSB2,NPOSB1),NB_TOT)
      ENDDO; ENDDO
      ENDIF

      CALLMPI( M_sum_g(W0%WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))

! test
!     IF (W0%WDES%COMM%IONODE==W0%WDES%COMM%NODE_ME) THEN
!        DO N=1,W0%WDES%NB_TOT
!           IF (W0%FERTOT(N,NK,ISP)<1E-4_q) COVL(N,:)=zero
!           WRITE(80,'(32(2F14.7,2X))') COVL(N,:)
!        ENDDO
!     ENDIF
! test
      ! set COVL(i,j)=0 if |phi(0)_i> is an empty state.
      DO N=1,W0%WDES%NB_TOT
         IF (W0%FERTOT(N,NK,ISP)<1E-4_q) COVL(N,:)=zero
      ENDDO

      ! subtract S|phi(0)_j><phi(0)_j|phi(1)_i> from |phi(1)_i>
!     CALL GGEMM('N','N',m_ NPL,NB_TOT,NB_TOT,-one,CH_RED(1,1),m_ NRPLWV_RED,COVL(1,1),NB_TOT,one,CW1_RED(1,1),m_ NRPLWV_RED)
!     NSTRPL=m_ NPL
!     DO NPOSPL=1,m_ NPL,NSTRPL
!     CALL GGEMM('N','N',MIN(NSTRPL,m_ NPL-NPOSPL+1),NB_TOT,NB_TOT,-one,CH_RED(NPOSPL,1),m_ NRPLWV_RED,COVL(1,1),NB_TOT,one,CW1_RED(NPOSPL,1),m_ NRPLWV_RED)
!     ENDDO
      IF (NPL/=0) THEN
      NSTRPB=NB_TOT; NSTRPL=m_ NPL
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSPL=1,m_ NPL,NSTRPL
      CALL GGEMM('N','N',MIN(NSTRPL,m_ NPL-NPOSPL+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),NB_TOT,-one,CH_RED(NPOSPL,1),m_ NRPLWV_RED,COVL(1,NPOSB1),NB_TOT,one,CW1_RED(NPOSPL,NPOSB1),m_ NRPLWV_RED)
      ENDDO; ENDDO
      ENDIF

      ! redistribute back to original data distribution
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
      ENDIF

!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      ! deallocation
      IF (NCPU/=1) NULLIFY(CW0_RED,CW1_RED,CH_RED)

      DO N=1,NSTRIP
         CALL DELWAV(WTMP(N),INFO%LREAL)
      ENDDO

      DEALLOCATE(WTMP,DUMMY,LDO,CH,COVL)

      IF (INFO%LREAL) DEALLOCATE(CWORK1)

      RETURN
      END SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS


!************************ SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS ***
!
!***********************************************************************
      SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS_(&
     & W0,GRID,NONL_S,NONLR_S,LATT_CUR,LMDIM,CQIJ,INFO,WXI)
      USE prec
      USE base
      USE dfast
      USE lattice
      USE mgrid
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) WXI
      TYPE (grid_3d) GRID
      TYPE (nonl_struct) NONL_S
      TYPE (nonlr_struct) NONLR_S
      TYPE (latt) LATT_CUR
      TYPE (info_struct) INFO
      INTEGER LMDIM
      OVERLAP CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
      ! local variables
      TYPE (wavedes1) WDES1
      TYPE (wavefun1), ALLOCATABLE :: WTMP(:)
      ! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW0_RED(:,:),CW1_RED(:,:),CH_RED(:,:)
      INTEGER NCPU,NRPLWV_RED,NPL,NPRO
      LOGICAL DO_REDIS

      INTEGER ISP,NK,ISPINOR
      INTEGER NB_TOT,NBANDS,NSTRIP,NSTRIP_ACT,NB_START,N
      INTEGER NPOSB1,NPOSB2,NSTRPB,NPOSPL,NSTRPL

      REAL(q), ALLOCATABLE :: DUMMY(:)
      LOGICAL, ALLOCATABLE :: LDO(:)

      COMPLEX(q), ALLOCATABLE, TARGET :: CH(:,:)
      COMPLEX(q), ALLOCATABLE :: CWORK1(:,:)

      GDEF,ALLOCATABLE,TARGET :: COVL(:,:)

#ifdef MPI
      ! number of procs involved in band distribution
      NCPU=W0%WDES%COMM_INTER%NCPU
#else
      NCPU=1
#endif
      IF (NCPU/=1) THEN
         DO_REDIS=.TRUE.
         NRPLWV_RED=W0%WDES%NRPLWV/NCPU
      ELSE
         DO_REDIS=.FALSE.
         NRPLWV_RED=W0%WDES%NRPLWV
      ENDIF

      NB_TOT=W0%WDES%NB_TOT
      NBANDS=W0%WDES%NBANDS

!     ! set NSTRIP between [1 and 32]
!     NSTRIP=NSTRIP_STANDARD_GLOBAL
      ! the above was much too large
      NSTRIP=MIN(4,NBANDS)

      CALL SETWDES(W0%WDES,WDES1,0)
      ALLOCATE(WTMP(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WTMP(N),WDES1,INFO%LREAL)
      ENDDO

      ALLOCATE(DUMMY(NSTRIP),LDO(NSTRIP))

      IF (INFO%LREAL) THEN
         ALLOCATE(CWORK1(GRID%MPLWV*W0%WDES%NRSPINORS,NSTRIP))
      ENDIF

      ALLOCATE(CH(W0%WDES%NRPLWV,NBANDS))
      ! get pointers for redistribution to "over-plane-wave coefficients"
      IF (NCPU/=1) THEN
         CALL SET_WPOINTER( CH_RED, NRPLWV_RED, NB_TOT, CH(1,1))
      ELSE
         CH_RED => CH(:,:)
      ENDIF

      ALLOCATE(COVL(NB_TOT,NB_TOT))

!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

      IF (INFO%LREAL) THEN
         CALL PHASER(GRID,LATT_CUR,NONLR_S,NK+NKPTS_ORIG,W0%WDES)
      ELSE
         NONL_S%NK=-1; CALL PHASE(W0%WDES,NONL_S,NK+NKPTS_ORIG) 
      ENDIF

      ! compute S|phi(0)> and store it in CH
      CH=(0._q,0._q)
      DO NB_START=1,NBANDS,NSTRIP
         NSTRIP_ACT=MIN(NSTRIP,NBANDS-NB_START+1)

         DO N=1,NSTRIP_ACT
            CALL W1_COPY(ELEMENT(W0,WDES1,NB_START+N-1,ISP),WTMP(N)); IF (INFO%LREAL) CALL FFTWAV_W1(WTMP(N))
            CH(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB_START+N-1)=WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
         ENDDO

         IF (INFO%LREAL) THEN
            CWORK1=0; LDO=.FALSE.; LDO(1:NSTRIP_ACT)=.TRUE.; DUMMY=0
            CALL RACCMU(NONLR_S,WDES1,WTMP,LMDIM,CQIJ,CQIJ,DUMMY,CWORK1,GRID%MPLWV*WDES1%NRSPINORS,NSTRIP,LDO)
            DO N=1,NSTRIP_ACT
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  CALL FFTEXT(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK1(1+ISPINOR*WDES1%GRID%MPLWV,N),CH(1+ISPINOR*WDES1%NGVECTOR,NB_START+N-1),GRID,.TRUE.)
               ENDDO
            ENDDO
         ELSE
            DO N=1,NSTRIP_ACT 
               CALL VNLACC_ADD(NONL_S,WTMP(N),CQIJ,CQIJ,1,0._q,CH(:,NB_START+N-1))
            ENDDO
         ENDIF 
      ENDDO

      ! get pointers for redistribution to "over-plane-wave coefficients"
      IF (NCPU/=1) THEN
         CALL SET_WPOINTER(CW0_RED, NRPLWV_RED, NB_TOT,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL SET_WPOINTER(CW1_RED, NRPLWV_RED, NB_TOT, WXI%CW(1,1,NK,ISP))
      ELSE
         CW0_RED =>  W0%CW(:,:,NK+NKPTS_ORIG,ISP)
         CW1_RED => WXI%CW(:,:,NK,ISP)
      ENDIF

      ! set number of wavefunctions after redistribution
      NPL = WDES1%NPL
      NPRO= WDES1%NPRO
      CALLMPI(SET_NPL_NPRO(WDES1, NPL, NPRO))

      ! redistribute to "over-plane-wave coefficients"
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
         CALL REDIS_PW(WDES1, NBANDS,     CH(1,1)       )
      ENDIF

      ! calculate COVL(i,j)=<phi(0)_i|S|phi(1)_j>
      COVL=zero
!     CALL GGEMM('C','N',NB_TOT,NB_TOT,m_ NPL,one,CW0_RED(1,1),m_ NRPLWV_RED,CW1_RED(1,1),m_ NRPLWV_RED,zero,COVL(1,1),NB_TOT)
!     NSTRPB=NSTRIP
      NSTRPB=NB_TOT
      IF (NPL/=0) THEN
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSB2=1,NB_TOT,NSTRPB
      CALL GGEMM('C','N',MIN(NSTRPB,NB_TOT-NPOSB2+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),m_ NPL,one,CH_RED(1,NPOSB2),m_ NRPLWV_RED,CW1_RED(1,NPOSB1),m_ NRPLWV_RED,zero,COVL(NPOSB2,NPOSB1),NB_TOT)
      ENDDO; ENDDO
      ENDIF

      CALLMPI( M_sum_g(W0%WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))

! test
!     IF (W0%WDES%COMM%IONODE==W0%WDES%COMM%NODE_ME) THEN
!        DO N=1,W0%WDES%NB_TOT
!           IF (W0%FERTOT(N,NK,ISP)<1E-4_q) COVL(N,:)=zero
!           WRITE(80,'(32(2F14.7,2X))') COVL(N,:)
!        ENDDO
!     ENDIF
! test
      ! set COVL(i,j)=0 if |phi(0)_i> is an empty state.
      DO N=1,W0%WDES%NB_TOT
         IF (W0%FERTOT(N,NK,ISP)<1E-4_q) COVL(N,:)=zero
      ENDDO

      ! subtract |phi(0)_j><phi(0)_j|S|phi(1)_i> from |phi(1)_i>
!     CALL GGEMM('N','N',m_ NPL,NB_TOT,NB_TOT,-one,CH_RED(1,1),m_ NRPLWV_RED,COVL(1,1),NB_TOT,one,CW1_RED(1,1),m_ NRPLWV_RED)
!     NSTRPL=m_ NPL
!     DO NPOSPL=1,m_ NPL,NSTRPL
!     CALL GGEMM('N','N',MIN(NSTRPL,m_ NPL-NPOSPL+1),NB_TOT,NB_TOT,-one,CH_RED(NPOSPL,1),m_ NRPLWV_RED,COVL(1,1),NB_TOT,one,CW1_RED(NPOSPL,1),m_ NRPLWV_RED)
!     ENDDO
      NSTRPB=NB_TOT; NSTRPL=m_ NPL
      IF (NPL/=0) THEN
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSPL=1,m_ NPL,NSTRPL
      CALL GGEMM('N','N',MIN(NSTRPL,m_ NPL-NPOSPL+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),NB_TOT,-one,CW0_RED(NPOSPL,1),m_ NRPLWV_RED,COVL(1,NPOSB1),NB_TOT,one,CW1_RED(NPOSPL,NPOSB1),m_ NRPLWV_RED)
      ENDDO; ENDDO
      ENDIF

      ! redistribute back to original data distribution
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
      ENDIF

!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      ! deallocation
      IF (NCPU/=1) NULLIFY(CW0_RED,CW1_RED,CH_RED)

      DO N=1,NSTRIP
         CALL DELWAV(WTMP(N),INFO%LREAL)
      ENDDO

      DEALLOCATE(WTMP,DUMMY,LDO,CH,COVL)

      IF (INFO%LREAL) DEALLOCATE(CWORK1)

      RETURN
      END SUBROUTINE MLR_PROJ_IN_EMPTY_SUBSPACE_BLAS_


!************************ SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE **********
!
!***********************************************************************
      SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE(W0,WXI,W1)
      USE prec
      USE dfast
      USE wave_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) W1
      TYPE (wavespin) WXI
      ! local variables
      TYPE (wavedes1) WDES1
      TYPE (wavefun1), ALLOCATABLE :: WTMP(:)

      INTEGER ISP,NK,NB,N,ISPINOR,M,MM
      INTEGER NB_TOT,NB_START,NB_STOP,NSTRIP,NSTRIP_ACT

      GDEF COVL

      NB_TOT=W0%WDES%NB_TOT
      
      ! set NSTRIP between [1 and 32]
      NSTRIP=NSTRIP_STANDARD_GLOBAL

      CALL SETWDES(W0%WDES,WDES1,0)
      ALLOCATE(WTMP(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WTMP(N),WDES1,.FALSE.)
      ENDDO

!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

!-----------------------------------------------------------------------
! calculate
!  |u(1)_k+q,n> 
!   - \sum_j | u(0)_j,k+q >< u(0)_j,k+q | (1/i) [r_idir,S_k+q,k] |u(0)_nk>
!
! where j runs over occupied orbitals
!-----------------------------------------------------------------------

      DO NB_START=1,NB_TOT,NSTRIP
         NSTRIP_ACT=MIN(NSTRIP,NB_TOT-NB_START+1)
         NB_STOP=MIN(NB_START+NSTRIP-1,NB_TOT)
         ! collect the wave functions from NB_START to NB_STOP (global band index) into WTMP
         CALL W1_GATHER_GLB_NOCR(W0,NB_START,NB_STOP,ISP,WTMP)

         DO NB=1,W1%WDES%NBANDS
            DO N=1,NSTRIP_ACT
               IF (W0%FERTOT(NB_START+N-1,NK,ISP)<1E-4_q) CYCLE
               COVL=zero
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,WDES1%NGVECTOR 
                     MM=M+WDES1%NGVECTOR*ISPINOR
                     COVL=COVL+GCONJG(WTMP(N)%CW(MM))*WXI%CW(MM,NB,NK,ISP)
                  ENDDO
               ENDDO
! test
!              W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)=W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)- &
!             &   COVL*WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
               W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)=W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)+ &
              &   COVL*WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
! test
            ENDDO
         ENDDO
      ENDDO
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      DO N=1,NSTRIP
         CALL DELWAV(WTMP(N),.FALSE.)
      ENDDO
      DEALLOCATE(WTMP)

      RETURN
      END SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE


!************************ SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE_BLAS *****
!
!***********************************************************************
      SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE_BLAS(&
     &   W0,WXI,W1)
      USE prec
      USE base
      USE dfast
      USE lattice
      USE mgrid
      USE wave_high
      USE nonl_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) WXI
      TYPE (wavespin) W1
      ! local variables
      TYPE (wavedes1) WDES1
      ! redistributed plane wave coefficients
      COMPLEX(q), POINTER :: CW0_RED(:,:),CW1_RED(:,:),CWX_RED(:,:)
      INTEGER NCPU,NRPLWV_RED,NPL,NPRO
      LOGICAL DO_REDIS

      INTEGER ISP,NK
      INTEGER NB_TOT,NBANDS,N
      INTEGER NPOSB1,NPOSB2,NSTRPB,NPOSPL,NSTRPL

      GDEF,ALLOCATABLE,TARGET :: COVL(:,:)

#ifdef MPI
      ! number of procs involved in band distribution
      NCPU=W0%WDES%COMM_INTER%NCPU
#else
      NCPU=1
#endif
      IF (NCPU/=1) THEN
         DO_REDIS=.TRUE.
         NRPLWV_RED=W0%WDES%NRPLWV/NCPU
      ELSE
         DO_REDIS=.FALSE.
         NRPLWV_RED=W0%WDES%NRPLWV
      ENDIF

      NB_TOT=W0%WDES%NB_TOT
      NBANDS=W0%WDES%NBANDS

      ALLOCATE(COVL(NB_TOT,NB_TOT))
!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

      ! get pointers for redistribution to "over-plane-wave coefficients"
      IF (NCPU/=1) THEN
         CALL SET_WPOINTER(CW0_RED, NRPLWV_RED, NB_TOT,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL SET_WPOINTER(CW1_RED, NRPLWV_RED, NB_TOT,  W1%CW(1,1,NK,ISP))
         CALL SET_WPOINTER(CWX_RED, NRPLWV_RED, NB_TOT, WXI%CW(1,1,NK,ISP))
      ELSE
         CW0_RED =>  W0%CW(:,:,NK+NKPTS_ORIG,ISP)
         CW1_RED =>  W1%CW(:,:,NK,ISP)
         CWX_RED => WXI%CW(:,:,NK,ISP)
      ENDIF

      ! set number of wavefunctions after redistribution
      NPL = WDES1%NPL
      NPRO= WDES1%NPRO
      CALLMPI(SET_NPL_NPRO(WDES1, NPL, NPRO))

      ! redistribute to "over-plane-wave coefficients"
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS,  W1%CW(1,1,NK,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
      ENDIF

      ! calculate COVL(i,j)=<phi(0)_i|WXI_j>
      COVL=zero
      IF (NPL/=0) THEN
      NSTRPB=NB_TOT
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSB2=1,NB_TOT,NSTRPB
      CALL GGEMM('C','N',MIN(NSTRPB,NB_TOT-NPOSB2+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),m_ NPL,one,CW0_RED(1,NPOSB2),m_ NRPLWV_RED,CWX_RED(1,NPOSB1),m_ NRPLWV_RED,zero,COVL(NPOSB2,NPOSB1),NB_TOT)
      ENDDO; ENDDO
      ENDIF

      CALLMPI( M_sum_g(W0%WDES%COMM_KIN,COVL(1,1),NB_TOT*NB_TOT))

      ! set COVL(i,j)=0 if |phi(0)_i> is an empty state.
      DO N=1,W0%WDES%NB_TOT
         IF (W0%FERTOT(N,NK,ISP)<1E-4_q) COVL(N,:)=zero
      ENDDO

      ! add |phi(0)_j><phi(0)_j|WXI_i> to |phi(1)_i>
      IF (NPL/=0) THEN
      NSTRPB=NB_TOT; NSTRPL=m_ NPL
      DO NPOSB1=1,NB_TOT,NSTRPB; DO NPOSPL=1,m_ NPL,NSTRPL
      CALL GGEMM('N','N',MIN(NSTRPL,m_ NPL-NPOSPL+1),MIN(NSTRPB,NB_TOT-NPOSB1+1),NB_TOT,one,CW0_RED(NPOSPL,1),m_ NRPLWV_RED,COVL(1,NPOSB1),NB_TOT,one,CW1_RED(NPOSPL,NPOSB1),m_ NRPLWV_RED)
      ENDDO; ENDDO
      ENDIF

      ! redistribute back to original data distribution
      IF (DO_REDIS) THEN
         CALL REDIS_PW(WDES1, NBANDS,  W0%CW(1,1,NK+NKPTS_ORIG,ISP))
         CALL REDIS_PW(WDES1, NBANDS,  W1%CW(1,1,NK,ISP))
         CALL REDIS_PW(WDES1, NBANDS, WXI%CW(1,1,NK,ISP))
      ENDIF

!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      ! deallocation
      IF (NCPU/=1) NULLIFY(CW0_RED,CW1_RED,CWX_RED)
      DEALLOCATE(COVL)

      RETURN
      END SUBROUTINE MLR_PROJ_IN_OCC_SUBSPACE_BLAS


!************************ SUBROUTINE MLR_APPLY_GREENS_FUNC *************
!
!***********************************************************************
      SUBROUTINE MLR_APPLY_GREENS_FUNC(W0,WXI,W1)
      USE prec
      USE dfast
      USE wave_high
      IMPLICIT NONE
      TYPE (wavespin) W0
      TYPE (wavespin) W1
      TYPE (wavespin) WXI
      ! local variables
      TYPE (wavedes1) WDES1
      TYPE (wavefun1), ALLOCATABLE :: WTMP(:)

      INTEGER ISP,NK,NB,N,ISPINOR,M,MM
      INTEGER NB_TOT,NB_START,NB_STOP,NSTRIP,NSTRIP_ACT

      REAL(q), PARAMETER :: TINY=1.E-6_q

      GDEF COVL

      NB_TOT=W0%WDES%NB_TOT
      
      ! set NSTRIP between [1 and 32]
      NSTRIP=NSTRIP_STANDARD_GLOBAL

      CALL SETWDES(W0%WDES,WDES1,0)
      ALLOCATE(WTMP(NSTRIP))
      DO N=1,NSTRIP
         CALL NEWWAV(WTMP(N),WDES1,.FALSE.)
      ENDDO

!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoints: DO NK=1,NKPTS_ORIG
#ifdef MPI   
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
!=======================================================================
      CALL SETWDES(W0%WDES,WDES1,NK+NKPTS_ORIG)

!-----------------------------------------------------------------------
! calculate
!  |u(1)_k+q,n> 
!            | u(0)_j,k+q >< u(0)_j,k+q |
!   + \sum_j ---------------------------- |Xi(0)_nk+q>
!                e(0)_n,k - e(0)_j,k+q
!
! where j runs over occupied orbitals
!-----------------------------------------------------------------------

      DO NB_START=1,NB_TOT,NSTRIP
         NSTRIP_ACT=MIN(NSTRIP,NB_TOT-NB_START+1)
         NB_STOP=MIN(NB_START+NSTRIP-1,NB_TOT)
         ! collect the wave functions from NB_START to NB_STOP (global band index) into WTMP
         CALL W1_GATHER_GLB_NOCR(W0,NB_START,NB_STOP,ISP,WTMP)

         DO NB=1,W1%WDES%NBANDS
            DO N=1,NSTRIP_ACT
               IF (W0%FERTOT(NB_START+N-1,NK,ISP)>1E-4_q) CYCLE
               COVL=zero
               DO ISPINOR=0,WDES1%NRSPINORS-1
                  DO M=1,WDES1%NGVECTOR 
                     MM=M+WDES1%NGVECTOR*ISPINOR
                     COVL=COVL+GCONJG(WTMP(N)%CW(MM))*WXI%CW(MM,NB,NK,ISP)
                  ENDDO
               ENDDO
               IF (ABS(REAL(W0%CELEN(NB,NK,ISP)-W0%CELTOT(NB_START+N-1,NK+NKPTS_ORIG,ISP),q))>TINY) THEN
                  W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)=W1%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR,NB,NK,ISP)- &
                 &   COVL/REAL(W0%CELEN(NB,NK,ISP)-W0%CELTOT(NB_START+N-1,NK+NKPTS_ORIG,ISP),q) &
                 &   *WTMP(N)%CW(1:WDES1%NRSPINORS*WDES1%NGVECTOR)
               ENDIF
            ENDDO
         ENDDO
      ENDDO
!=======================================================================
      ENDDO kpoints
      ENDDO spin
!=======================================================================

      DO N=1,NSTRIP
         CALL DELWAV(WTMP(N),.FALSE.)
      ENDDO
      DEALLOCATE(WTMP)

      RETURN
      END SUBROUTINE MLR_APPLY_GREENS_FUNC


!************************ SUBROUTINE MLR_SBARE *************************
!
!***********************************************************************
      SUBROUTINE MLR_SBARE(W,WP,S_BARE)
      USE prec
      USE wave_high
      IMPLICIT NONE
      TYPE (wavespin) W,WP
      RGRID :: S_BARE(W%WDES%GRID%RL%NP,3) 
      ! local variables
      TYPE (wavedes1) WDES1,WDESP1
      TYPE (wavefun1) W1,WP1 
      COMPLEX(q), ALLOCATABLE :: CWORK1(:)
      COMPLEX(q), ALLOCATABLE :: CWORK2(:)
      INTEGER ISP,NK,N
      REAL(q) WEIGHT

      ALLOCATE(CWORK1(W%WDES%NRPLWV))
      ALLOCATE(CWORK2(W%WDES%GRID%MPLWV))

      CALL SETWDES(W%WDES ,WDES1 ,0)
      CALL SETWDES(WP%WDES,WDESP1,0)
      CALL NEWWAV(W1 ,WDES1, .TRUE.)
      CALL NEWWAV(WP1,WDESP1,.TRUE.)

      spin:  DO ISP=1,W%WDES%ISPIN
      kpoint: DO NK=1,WP%WDES%NKPTS
#ifdef MPI   
         IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         CALL SETWDES(W%WDES ,WDES1 ,NK)
         CALL SETWDES(WP%WDES,WDESP1,NK)  ! but this one is shifted with q

         DO N=1,W%WDES%NBANDS
            CALL W1_COPY( ELEMENT( W , WDES1 , N, ISP), W1 )
            CALL W1_COPY( ELEMENT( WP, WDESP1, N, ISP), WP1)
            CALL FFTWAV_W1(W1 )
            CALL FFTWAV_W1(WP1)
            WEIGHT=WDES1%RSPIN*W%FERWE(N,NK,ISP)*W%WDES%WTKPT(NK)     ! occupancies assumed identical

            CALL PSCURRENT_PARA_TWO_KPT(WDES1, WDES1%GRID, W1%CR(1), WP1%CR(1), WDES1%IGX(1), &
           &     WDES1%IGY(1), WDES1%IGZ(1), WDES1%VKPT, WDESP1%VKPT, WEIGHT, CWORK1(1), CWORK2(1), &
           &     W1%CW(1), WP1%CW(1), S_BARE(1,1))

         ENDDO
      ENDDO kpoint
      ENDDO spin

      CALL DELWAV(W1  ,.TRUE.)
      CALL DELWAV(WP1,.TRUE.)

      DEALLOCATE(CWORK1)
      DEALLOCATE(CWORK2)

      RETURN
      END SUBROUTINE MLR_SBARE


!****************************** CALC_B_MAGATOM_LR **********************************
!
! Taken from CALC_B_MAGATOM, augmentation removed
!
! To calculate one-centre contribution to magnetic field at atomic nuclei.
! First paramagnetic and diamagnetic "currents" are calculated:
!
! COMMENT: change from p -> -i nabla
! COMMENT: and  A to M x r / r^3
! para: J_LM^alpha(r) =   Sum_ij \int \phi_i ((-i nabla)^* + -i nabla )/2 Y_LM \phi_j dOmega \rho_ji
! dia:  J_LM^alpha(r) = - Sum_ij \int \phi_i  M x r / r^3   Y_LM \phi_j dOmega \rho_ji
!                         * MOMTOMOM           
!
! These are collected into one array
!
! Next the augmentation currents are NOT subracted
!
! Finally Biot-Savart is applied to obtain the field:
!
!   B = - r x J /|r^3|
!
! A=(Bxr)/2*MAGMOMTOENERGY
!
! This routine is similar to and derived from CALC_MU_MAGATOM
! It is different though, as the augmentation current has to be subtracted in
! a "different" place.
!
!***********************************************************************************

      SUBROUTINE CALC_B_MAGATOM_LR(PP,CRHODE,BDIR,IDP,NIP,NI,B_OUT,LLRAUG)
      USE pseudo
      USE constant
      USE asa
      USE radial 
      USE wave
      USE paw
      USE poscar
      USE prec
      USE morbitalmag, ONLY : LBONE

      IMPLICIT NONE
     
      TYPE (potcar), POINTER :: PP   ! pseudopotential descriptor
      COMPLEX(q) :: CRHODE(:,:)         ! one center occupancy matrix
      REAL(q) :: B_OUT(3)
      INTEGER :: NIP,NI,BDIR
      LOGICAL LLRAUG

      INTEGER, EXTERNAL :: MAXL1

      !local variables
      INTEGER CHANNELS          ! number of channels
      INTEGER NMAX,I,NT
      INTEGER I0,I1
      INTEGER L0,L1,L2,L3
      INTEGER M0,M1,M3
      INTEGER LM0,LM1,LM2,LM3,LM2MAX
      INTEGER ISTART1,IEND1,LMIND1,IC1
      INTEGER LMAXNABLA,LMAX,K,IDP
      INTEGER II1,II2,II3,LL,M
      INTEGER LRHO,LMRHO,MRHO,L2MAX
      INTEGER IRMAX
      REAL(q) :: CGOR
      REAL(q) :: SUM
      REAL(q), ALLOCATABLE :: WTMP(:),DWAE(:),DWPS(:),TMP(:)
      REAL(q), ALLOCATABLE :: WTMP2(:),WTMP3(:),WBAE(:,:),DWBAE(:)
      REAL(q), ALLOCATABLE :: YLM_NABLA_YLM(:,:,:),YLM_X_YLM(:,:,:),YLM_BxR_YLM(:,:,:)
      REAL(q), ALLOCATABLE :: JPAWRAD(:,:,:)
      REAL(q) :: RHOLM(256)
      COMPLEX(q) :: JLM(256,3)
      REAL(q) :: OSUM
      REAL(q) :: OSUM1,OSUM2
      REAL(q), ALLOCATABLE :: OTMP(:)
      COMPLEX(q) :: BF(3)
      REAL(q) :: BCONST_LR(3)

      REAL(q), PARAMETER :: C = 137.037  ! speed of light in a.u.
      REAL(q), PARAMETER :: INVMC2=7.45596E-6
!                           invmc2=hbar^2/2(m_e c)^2 in A^2
      REAL(q), ALLOCATABLE :: POT(:),KQ(:)
      REAL(q), ALLOCATABLE :: WEIGHT_WAAE(:,:),WEIGHT_WBAE(:,:)
      REAL(q) RDUM
      REAL(q) AAE_CHAR,BAE_CHAR,WEIGHT_TMP

      IF (.NOT. ASSOCIATED(PP%JPAW)) RETURN
      BCONST_LR=0._q
      IF (IDP.EQ.2) THEN  ! diamagnetic
         BCONST_LR(BDIR)=1._q
      ENDIF

      ! allocate
      NMAX=PP%R%NMAX
      ALLOCATE(WTMP(NMAX),DWAE(NMAX),DWPS(NMAX),TMP(NMAX),OTMP(NMAX))
      ALLOCATE(WTMP2(NMAX),WTMP3(NMAX),WBAE(NMAX,PP%LMAX),DWBAE(NMAX))
      ALLOCATE(POT(NMAX),KQ(NMAX),WEIGHT_WAAE(NMAX,PP%LMAX),WEIGHT_WBAE(NMAX,PP%LMAX))

      CHANNELS = PP%LMAX
      LMAX=MAXL1(PP)
      LMAXNABLA=MAXL1(PP)+1   ! nabla Y_lm has components up to LMAX+1
      ALLOCATE(JPAWRAD(PP%R%NMAX,(LMAXNABLA+LMAX+1)*(LMAXNABLA+LMAX+1),3))        ! be ware

      ALLOCATE(YLM_NABLA_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAXNABLA+2)*(2*LMAXNABLA+2),0:3))
      ALLOCATE(YLM_X_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))
      ALLOCATE(YLM_BxR_YLM((2*LMAX+2)*(2*LMAX+2),(2*LMAX+2)*(2*LMAX+2),0:3))

      YLM_NABLA_YLM=0._q
      YLM_X_YLM=0._q
      YLM_BxR_YLM=0._q

!GIL  CALL SETYLM_NABLA_YLM(LMAXNABLA,YLM_NABLA_YLM,YLM_X_YLM)
! in principle we should go up to LMAX+LMAXNABLA+1 = 2 LMAX + 2
! see below, but that does not work for f electrons
      CALL SETYLM_NABLA_YLM(2*LMAX+1,YLM_NABLA_YLM,YLM_X_YLM)

      YLM_BxR_YLM(:,:,1)=BCONST_LR(2)*YLM_X_YLM(:,:,3)-BCONST_LR(3)*YLM_X_YLM(:,:,2)
      YLM_BxR_YLM(:,:,2)=BCONST_LR(3)*YLM_X_YLM(:,:,1)-BCONST_LR(1)*YLM_X_YLM(:,:,3)
      YLM_BxR_YLM(:,:,3)=BCONST_LR(1)*YLM_X_YLM(:,:,2)-BCONST_LR(2)*YLM_X_YLM(:,:,1)
      YLM_BxR_YLM(:,:,:)=YLM_BxR_YLM(:,:,:)/2. !*MAGMOMTOENERGY
 
      JPAWRAD=0._q   
      B_OUT=0._q

      IF (LLRAUG) THEN
         DO I=1,3
            CALL CALC_RHOLM_JVEC_CMPLX(  CRHODE, JLM(:,I), PP, I)
         ENDDO
      ENDIF

      IF (LBONE) THEN
         POT=0._q
         ! compute V_H[n^a_c]
         CALL RAD_POT_HAR(0,PP%R,POT,PP%RHOAE,RDUM)
         ! add -Z/r
         POT=POT/(2._q*SQRT(PI))-(PP%ZCORE+PP%ZVALF_ORIG)*FELECT/PP%R%R
         ! add V_H[n^a_v] + V_xc[n^a_v+n^a_c]
         POT=POT-PP%POTAE
         ! and make sure this AE-"atomic reference"-potential lines up exactly with
         ! the potential that was constructed by the pseudopotential generation code.
         POT=POT+(PP%VPSRMAX-POT(NMAX))

         KQ(:)=1._q/(1._q-POT(:)/(2._q*RYTOEV)/CLIGHT/CLIGHT/2._q)
         KQ(:)=KQ(:)*(INVMC2**0.5_q)

         DO I=1,NMAX-1
            IF ((PP%RDEP>0).AND.(PP%R%R(I)-PP%RDEP>-5E-3)) EXIT
         ENDDO
!        WRITE(*,*) 'outermost point',PP%R%R(I),PP%RDEP
         IRMAX=I
      ELSE
         KQ(:)=1._q
      ENDIF

!     OPEN (UNIT=99,FILE='ABTMPAE',STATUS="UNKNOWN")
      DO I1=1,PP%LMAX   !CHANNELS
         WTMP2(:)=PP%WAE(:,I1)/PP%R%R(:)
         CALL GRAD_(PP%R,WTMP2,WTMP3)
         WTMP3(:)=WTMP3(:)*KQ(:)/SQRT(2._q)
!        WTMP3(:)=WTMP3(:)/(2._q**0.5_q)
         WBAE(:,I1)=WTMP3(:)*PP%R%R(:)
!        WRITE (99,*) I1
!        DO I=1,NMAX
!           WRITE (99,*) PP%R%R(I),PP%WAE(I,I1),WBAE(I,I1)
!        ENDDO
      ENDDO
!     CLOSE (99)
!     OPEN (UNIT=99,FILE='ABTMPPS',STATUS="UNKNOWN")
!     DO I1=1,PP%LMAX   !CHANNELS
!        WTMP2(:)=PP%WPS(:,I1)/PP%R%R
!        CALL GRAD_(PP%R,WTMP2,WTMP3)
!        WTMP2=WTMP3*PP%R%R
!        WRITE (99,*) I1
!        DO I=1,NMAX
!           WRITE (99,*) PP%R%R(I),PP%WPS(I,I1),WTMP2(I)
!        ENDDO
!     ENDDO
!     CLOSE (99)

      WEIGHT_WAAE(:,:)=1._q
      WEIGHT_WBAE(:,:)=0._q
      IF (LBONE) THEN
         DO I1=1,PP%LMAX   !CHANNELS
            WTMP2(:)=0._q
            DO I=1,IRMAX
               WTMP2(I)=PP%WAE(I,I1)*PP%WAE(I,I1)
            ENDDO
            CALL SIMPI(PP%R,WTMP2,AAE_CHAR)
            WTMP2(:)=0._q
            DO I=1,IRMAX
               WTMP2(I)=WBAE(I,I1)*WBAE(I,I1)
            ENDDO
            CALL SIMPI(PP%R,WTMP2,BAE_CHAR)
            WEIGHT_TMP=SQRT(AAE_CHAR/(AAE_CHAR+BAE_CHAR))
!           WRITE (*,*) 'charge',AAE_CHAR,BAE_CHAR,WEIGHT_TMP
            DO I=1,IRMAX
               WEIGHT_WAAE(I,I1)=WEIGHT_TMP
               WEIGHT_WBAE(I,I1)=WEIGHT_TMP
            ENDDO
         ENDDO
      ENDIF

! diapara: DO IDP=1,2

! if IDP.eq.1 paramagnetic contribution
! if IDP.eq.2 diamagnetic contribution

!-----------------------------------------------------------------------
! loop i0,i1
! i0 corresponds to nl
! i1 corresponds to n'l'
! LMI combined index (n,l,m)
!-----------------------------------------------------------------------
      LM2MAX=0
      L2MAX=0

      LM0=1
      DO I0=1,PP%LMAX   !CHANNELS
      LM1=1
      DO I1=1,PP%LMAX   !CHANNELS
           
         L0 = PP%LPS(I0)
         L1 = PP%LPS(I1)

         WTMP(:)=PP%WAE(:,I1)/PP%R%R(:)
! perferable maybe to use y d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r]
         CALL GRAD_(PP%R,WTMP,DWAE)
         DWAE=DWAE*PP%R%R

         WTMP(:)=WBAE(:,I1)/PP%R%R(:)
         CALL GRAD_(PP%R,WTMP,DWBAE)
         DWBAE=DWBAE*PP%R%R

         WTMP(:)=PP%WPS(:,I1)/PP%R%R(:)
         CALL GRAD_(PP%R,WTMP,DWPS)
         DWPS=DWPS*PP%R%R

         DO M1=1,2*L1+1

            LM3=1
            DO L3=0,LMAXNABLA

            CALL YLM3LOOKUP(L0,L3,LMIND1)

            DO M0=1,2*L0+1
            DO M3=1,2*L3+1

               LMIND1 = LMIND1+1
               ISTART1=INDCG(LMIND1)
               IEND1=INDCG(LMIND1+1)

               DO IC1=ISTART1,IEND1-1

                  CGOR=YLM3(IC1)
                  LM2 =JS(IC1)
                  L2  =JL(IC1)
                  IF (LM2 <=0 .OR. LM2 > SIZE(JPAWRAD,2)) THEN
                     WRITE(0,*) 'internal error in CALC_B_MAGATOM_LR: index 2 into JPAWRAD exceeds bounds'
                     STOP
                  ENDIF
                  LM2MAX=MAX(LM2, LM2MAX)
                  L2MAX=MAX(L2, L2MAX)

                  DO I=1,3

!-----------------------------------------------------------------------
! paramagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.1) THEN
                     IF (ABS(YLM_X_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8 .OR. ABS(YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                        TMP=0._q
                        SUM=0._q
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * ( &
        &   WEIGHT_WAAE(K,I0) *                                                     WEIGHT_WAAE(K,I1)   * &
        &        PP%WAE(K,I0) * (  DWAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                               &
        &                              + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) *  PP%WAE(K,I1) )   &
        & + WEIGHT_WBAE(K,I0) *                                                     WEIGHT_WBAE(K,I1)   * &
        &          WBAE(K,I0) * ( DWBAE(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                               &
        &                              + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) *    WBAE(K,I1) )   &
        & -    ( PP%WPS(K,I0) * (  DWPS(K) * YLM_X_YLM(LM3+M3-1,L1*L1+M1,I)                               &
        &                              + YLM_NABLA_YLM(LM3+M3-1,L1*L1+M1,I)/PP%R%R(K) *  PP%WPS(K,I1) ))  &
        &                   )
                        ENDDO

!                       IF (LM0+M0-1 <=0 .OR. LM0+M0-1 > SIZE(PP%JPAW,1)) THEN
!                          WRITE(0,*) 'internal error in CALC_JPAW: index 1 into JPAR exceeds bounds'
!                          STOP
!                       ENDIF
!                       IF (LM1+M1-1 <=0 .OR. LM1+M1-1 > SIZE(PP%JPAW,2)) THEN
!                          WRITE(0,*) 'internal error in CALC_JPAW: index 2 into JPAR exceeds bounds'
!                          STOP
!                       ENDIF
!                       PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) = PP%JPAW(LM1+M1-1,LM0+M0-1,LM2,I) + SUM

                        !multipy by -i and make symmetric (!)
                        DO K=1,PP%R%NMAX
#ifdef realmode
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)-AIMAG(TMP(K)*(   CONJG(CRHODE(LM1+M1-1,LM0+M0-1))+CRHODE(LM0+M0-1,LM1+M1-1))/2)
#else
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)-AIMAG(TMP(K)*(CHGCONJG(CRHODE(LM1+M1-1,LM0+M0-1))+CRHODE(LM0+M0-1,LM1+M1-1))/2)
#endif
! note that one can save 4 lines above
                        ENDDO
                     ENDIF
                     ENDIF
!-----------------------------------------------------------------------
! diamagnetic contribution
!-----------------------------------------------------------------------
                     IF (IDP.EQ.2) THEN
                     IF (ABS(YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I))>1E-8) THEN
                        TMP=0._q
                        SUM=0._q
!                       |r| from Y*x/y/z*Y
                        DO K=1,PP%R%NMAX
                           TMP(K) = CGOR * PP%R%R(K) * ( &
                               WEIGHT_WAAE(K,I0) * WEIGHT_WAAE(K,I1)                                      * &
                                    PP%WAE(K,I0) *      PP%WAE(K,I1) *( YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I) )  &
                             + WEIGHT_WBAE(K,I0) * WEIGHT_WBAE(K,I1)                                      * &
                                      WBAE(K,I0) *        WBAE(K,I1) *( YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I) )  &
                             -      PP%WPS(K,I0) *      PP%WPS(K,I1) *( YLM_BxR_YLM(LM3+M3-1,L1*L1+M1,I) )  &
                            )
                        ENDDO
!                       WRITE (*,'(3I3,2f16.10)') I1,I0,L2,PP%QPAW(I0,I1,L2)

                        DO K=1,PP%R%NMAX
!gK change
!                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM1+M1-1,LM0+M0-1)
                           JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I)+TMP(K)*MOMTOMOM*(-1._q)*CRHODE(LM0+M0-1,LM1+M1-1)
                        ENDDO
                     ENDIF
                     ENDIF
                  ENDDO  ! I

               ENDDO  ! IC1

            ENDDO ! M3
            ENDDO ! M0

            LM3=LM3+2*L3+1
            ENDDO ! L3

         ENDDO ! M1

      LM1=LM1+2*L1+1
      ENDDO ! I1
      LM0=LM0+2*L0+1
      ENDDO ! I0

      IF (IDP==1) THEN                                                
!-----------------------------------------------------------------------                                                     !NO AUG
! paramagnetic current augmentation contribution (matches                                                                    !NO AUG
!  CURRENT_AUGMENTATION in us.F)                                                                                             !NO AUG
! i.e. depends on whether                                                                                                    !NO AUG
!   j_para = e hbar/m_e sum_n Im(psi nabla psi)                                                                              !NO AUG
! is augmented with augmentation currents or not                                                                             !NO AUG
!-----------------------------------------------------------------------                                                     !NO AUG
      IF (LLRAUG) THEN
      LMRHO=1                                                         
      DO LRHO=0,2*LMAX                                                
         DO MRHO=1,2*LRHO+1                                           
            DO I=1,3                                                  
               DO K=1,PP%R%NMAX                                       
                  JPAWRAD(K,LMRHO+MRHO-1,I)=JPAWRAD(K,LMRHO+MRHO-1,I) +  & ! -1 as above -1 for augmentation correction 
                       PP%AUG(K,LRHO)*AIMAG((1.0_q,0.0_q)*JLM(LMRHO+MRHO-1,I)) ! R^2 in PP%AUG ?                        
               ENDDO                                                  
            ENDDO                                                     
         ENDDO                                                        
         LMRHO=LMRHO+2*LRHO+1                                         
      ENDDO                                                           
      ENDIF
!-----------------------------------------------------------------------                                                     !NO AUG
! diagmagnetic augmentation correction                                                                                       !NO AUG
!  depends on whether augmented charge density or plane wave                                                                 !NO AUG
!  contribution only is used when                                                                                            !NO AUG
!   j_dia =  e^2/m_e c psi^2 A                                                                                               !NO AUG
! is calculated                                                                                                              !NO AUG
!-----------------------------------------------------------------------                                                     !NO AUG
      ELSE IF (IDP==2) THEN                                           
!NO AUG #ifdef current_augmentation
!NO AUG       ! loop over all L,M pairs in the current density
!NO AUG !      DO LM2=1,(LMAX+LMAXNABLA+1)*(LMAX+LMAXNABLA+1)  ! this does not yet work
!NO AUG       DO LM2=1,(LMAX+LMAX+1)*(LMAX+LMAX+1)
!NO AUG          LMRHO=1
!NO AUG          DO LRHO=0,2*LMAX
!NO AUG          DO MRHO=1,2*LRHO+1
!NO AUG             DO I=1,3
!NO AUG                IF (LM2 <1 .OR. LM2 > SIZE(JPAWRAD,2)) THEN
!NO AUG                   WRITE(0,*) 'internal error CALC_B_MAGATOM, JPAWRAD',LM2,SIZE(JPAWRAD,2)
!NO AUG                   STOP
!NO AUG                ENDIF
!NO AUG                IF (LM2 <1 .OR. LM2 > SIZE(YLM_BxR_YLM,1)) THEN
!NO AUG                   WRITE(0,*) 'internal error CALC_B_MAGATOM, LM2',LM2,SIZE(YLM_BxR_YLM,1)
!NO AUG                   STOP
!NO AUG                ENDIF
!NO AUG                IF (LMRHO+MRHO-1 <1 .OR. LMRHO+MRHO-1 > SIZE(YLM_BxR_YLM,2)) THEN
!NO AUG                   WRITE(0,*) 'internal error CALC_B_MAGATOM, YLM_MxR_YLM, LMRHO+MRHO-1',LMRHO, MRHO
!NO AUG                   STOP
!NO AUG                ENDIF
!NO AUG                IF (LMRHO+MRHO-1 <1 .OR. LMRHO+MRHO-1 > SIZE(RHOLM,1)) THEN
!NO AUG                   WRITE(0,*) 'internal error CALC_B_MAGATOM, RHOLM, LMRHO+MRHO-1',LMRHO, MRHO
!NO AUG                   STOP
!NO AUG                ENDIF
!NO AUG                IF (LRHO <0 .OR. LRHO > SIZE(PP%AUG,2)-1) THEN
!NO AUG                   WRITE(0,*) 'internal error CALC_B_MAGATOM,PPAUG',LRHO,SIZE(PP%AUG,2)
!NO AUG                   STOP
!NO AUG                ENDIF
!NO AUG 
!NO AUG                DO K=1,PP%R%NMAX
!NO AUG                   JPAWRAD(K,LM2,I)=JPAWRAD(K,LM2,I) +  &
!NO AUG                      MOMTOMOM*  &                                             ! -1 for units, -1 for augmentation correction
!NO AUG                      PP%AUG(K,LRHO)*RHOLM(LMRHO+MRHO-1)*YLM_BxR_YLM(LM2,LMRHO+MRHO-1,I)*PP%R%R(K)
!NO AUG                ENDDO
!NO AUG             ENDDO
!NO AUG          ENDDO
!NO AUG          LMRHO=LMRHO+2*LRHO+1
!NO AUG          ENDDO
!NO AUG       ENDDO
!NO AUG #endif
       ENDIF                                                          

! calculate moment
! \int r \times J_LM^alpha Y_LM dV
! x Y_LM = sqrt(4pi) Y_00 x Y_LM = sqrt(4pi) |r|/|r| (Y_00 x Y_LM) = sqrt(4pi) |r|/|r| YLM_X_YLM(LM=0,LM,x)

      BF=0._q

      DO LM2=1,LM2MAX
         DO I=1,3
            ! please put in some guard against going over bounds

            DO K=1,PP%R%NMAX
               IF (I.EQ.1) OTMP(K)=(YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,3)-YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,2)) *PP%R%R(K)
               IF (I.EQ.2) OTMP(K)=(YLM_X_YLM(1,LM2,3)*JPAWRAD(K,LM2,1)-YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,3)) *PP%R%R(K)
               IF (I.EQ.3) OTMP(K)=(YLM_X_YLM(1,LM2,1)*JPAWRAD(K,LM2,2)-YLM_X_YLM(1,LM2,2)*JPAWRAD(K,LM2,1)) *PP%R%R(K)
               OTMP(K)=OTMP(K)/PP%R%R(K)**3
            ENDDO

            OSUM=0._q
            DO K=1,PP%R%NMAX
               OSUM=OSUM+OTMP(K)*PP%R%SI(K)
            ENDDO
            BF(I)=BF(I)+OSUM *SQRT(4._q*PI) *2._q !normalisation Y00 * 2

         ENDDO
      ENDDO

!      WRITE (*,'("one center magnetic field*1E6",I3,6E16.7)') NI,REAL(BF,q)*1E6

! default here: calculate dia and paramagnetic contr. seperately
        JPAWRAD=0

! make one-centre contributions to field real
        B_OUT(1)=REAL(BF(1),q)
        B_OUT(2)=REAL(BF(2),q)
        B_OUT(3)=REAL(BF(3),q)

!     ENDDO diapara

      DEALLOCATE(YLM_NABLA_YLM, YLM_X_YLM, YLM_BxR_YLM)
      DEALLOCATE(WTMP,DWAE,DWPS,TMP,OTMP)
      DEALLOCATE(WTMP2,WTMP3,WBAE,DWBAE)
      DEALLOCATE(JPAWRAD)
      DEALLOCATE(POT,KQ)
      DEALLOCATE(WEIGHT_WAAE,WEIGHT_WBAE)

      RETURN
      END SUBROUTINE CALC_B_MAGATOM_LR


!************************ SUBROUTINE DEPSUM_ASYM ***********************
!
! modelled on DEPSUM from us.F, left-right wave functions are different
! and from different k-points
!
! this subroutine calculates  the total one center (on site) "occupancy" 
! matrix of each ll'mm'augmentation channel from the FERMI weights, 
! the weight of each k-point WTKPT and  the wavefunction character 
! of all bands
! result is stored in CRHODE (thesis gK 10.32)
!
!***********************************************************************

      SUBROUTINE DEPSUM_ASYM(W_LEFT, W_RIGHT ,WDES, LMDIM, CRHODE, LOVERL)
      USE prec
      USE wave
      IMPLICIT NONE

      TYPE (wavespin)    W_LEFT
      TYPE (wavespin)    W_RIGHT
      TYPE (wavedes)     WDES
      LOGICAL LOVERL
      INTEGER LMDIM
      COMPLEX(q) CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

      INTEGER ISP, NT, NK, N, ISPINOR, ISPINOR_, LMBASE, LMBASE_, NIS, &
           LMMAXC, NI, L, LP
      REAL(q) WEIGHT

      IF (.NOT.LOVERL) RETURN
!=======================================================================
! initialise to zero
!=======================================================================
      CRHODE=0
!=======================================================================
! loop over all bands and k-points
!=======================================================================
      spin:   DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,NKPTS_ORIG    ! beware original number of k-points
#ifdef MPI   
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      band:   DO N=1,WDES%NBANDS

      WEIGHT=WDES%RSPIN*W_LEFT%FERWE(N,NK,ISP)*WDES%WTKPT(NK)    ! assumed identical occupancies for left and right channel

      spinor: DO ISPINOR =0,WDES%NRSPINORS-1
      DO ISPINOR_=0,WDES%NRSPINORS-1

      LMBASE =ISPINOR *(WDES%NPRO/2)
      LMBASE_=ISPINOR_*(WDES%NPRO/2)

      NIS   =1
      typ:  DO NT=1,WDES%NTYP
      LMMAXC=WDES%LMMAX(NT)
      IF (LMMAXC==0) GOTO 210

      ion: DO NI=NIS,WDES%NITYP(NT)+NIS-1

!DIR$ IVDEP
!OCL NOVREC
        DO L=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
        DO LP=1,LMMAXC
           CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)=CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)+ &
              WEIGHT*W_RIGHT%CPROJ(L+LMBASE,N,NK,ISP)*GCONJG(W_LEFT%CPROJ(LP+LMBASE_,N,NK,ISP))
        ENDDO
        ENDDO
      
      LMBASE = LMMAXC+LMBASE
      LMBASE_= LMMAXC+LMBASE_

      ENDDO ion

  210 NIS = NIS+WDES%NITYP(NT)
      ENDDO typ
      ENDDO
      ENDDO spinor

      ENDDO band
      ENDDO kpoint
      ENDDO spin
      ! sum over all bands
!#ifdef realmode
!      CALLMPI( M_sum_d(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
!      CALLMPI( M_sum_d(WDES%COMM_KINTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
!#else
      CALLMPI( M_sum_z(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
      CALLMPI( M_sum_z(WDES%COMM_KINTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ))
!#endif

      RETURN
      END SUBROUTINE DEPSUM_ASYM


!************************ SUBROUTINE DEPSUM_SYM_ASYM *******************
!
! modelled on DEPSUM from us.F, left-right wave functions are different
! and from different k-points
!
! this subroutine calculates  the total one center (on site) "occupancy" 
! applying 
! symmetry to determine the wavefunctions at each k-point
! it should be fully compatible with the Fock like routines but
! of course is slightly slower than DEPSUM
! if this routine is used to construct CRHODE no
! a posteriori symmetrization of the charge density should be required
!
!***********************************************************************

      SUBROUTINE DEPSUM_SYM_ASYM(W_LEFT, W_RIGHT, WDES, LMDIM, CRHODE, LOVERL, P, LATT_CUR)
      USE prec
      USE sym_prec
      USE wave_high
      USE full_kpoints
      USE pseudo
      USE lattice
      IMPLICIT NONE

      TYPE (wavespin)    W_LEFT, W_RIGHT
      TYPE (wavedes)     WDES
      LOGICAL LOVERL
      INTEGER LMDIM
      COMPLEX(q) CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      TYPE (potcar)       P(:)
      TYPE (latt) LATT_CUR
      
! local
      INTEGER ISP, ISP_IRZ, NT, NK, N, ISPINOR, ISPINOR_, LMBASE, LMBASE_, NIS, &
           LMMAXC, NI, L, LP
      LOGICAL :: LSHIFT
      TYPE( rotation_handle), POINTER :: ROT_HANDLE
      TYPE (wavedes1)     WDES1, WDES1_IRZ
      TYPE (wavefun1)     W1_LEFT
      TYPE (wavefun1)     W1_RIGHT
      REAL(q) WEIGHT

      IF (.NOT.LOVERL) RETURN

      CALL CHECK_FULL_KPOINTS
      NULLIFY(ROT_HANDLE)

      CALL SETWDES(WDES, WDES1, 0)
      CALL NEWWAV(W1_LEFT  , WDES1, .FALSE.)
      CALL NEWWAV(W1_RIGHT , WDES1, .FALSE.)
!=======================================================================
! initialise to zero
!=======================================================================
      CRHODE=0
!=======================================================================
! loop over all bands and k-points
!=======================================================================
      spin:   DO ISP=1,WDES%ISPIN
      kpoint: DO NK=1,KPOINTS_FULL%NKPTS       ! BE CAREFUL
         CALL SETWDES(WDES, WDES1, NK)
         CALL SETWDES(WDES, WDES1_IRZ, KPOINTS_FULL%NEQUIV(NK))
         
         ISP_IRZ=ISP
         IF (KPOINTS_FULL%SPINFLIP(NK)==1) THEN
            ISP_IRZ=3-ISP
         ENDIF
         
         band:   DO N=1,WDES%NBANDS

            WEIGHT=WDES%RSPIN*KPOINTS_FULL%WTKPT(NK)*W_LEFT%FERWE(N,KPOINTS_FULL%NEQUIV(NK),ISP)
                                                     ! assume identical occupancies left and right

            IF (NK <= WDES%NKPTS) THEN
               CALL W1_COPY(ELEMENT(W_LEFT,  WDES1, N, ISP), W1_LEFT )
               CALL W1_COPY(ELEMENT(W_RIGHT, WDES1, N, ISP), W1_RIGHT)
            ELSE
#ifndef gammareal
           ! 
           ! use symmetry to construct wave function charakter at k
               CALL ROTATE_WAVE_CHARACTER(ROT_HANDLE, P, LATT_CUR, WDES1, ELEMENT(W_LEFT, WDES1_IRZ, N, ISP_IRZ), W1_LEFT )
               CALL ROTATE_WAVE_CHARACTER(ROT_HANDLE, P, LATT_CUR, WDES1, ELEMENT(W_RIGHT, WDES1_IRZ, N, ISP_IRZ), W1_RIGHT)
#endif
            ENDIF

            spinor: DO ISPINOR =0,WDES%NRSPINORS-1
               DO ISPINOR_=0,WDES%NRSPINORS-1

                  LMBASE =ISPINOR *(WDES%NPRO/2)
                  LMBASE_=ISPINOR_*(WDES%NPRO/2)
                  
                  NIS   =1
                  typ:  DO NT=1,WDES%NTYP
                     LMMAXC=WDES%LMMAX(NT)
                     IF (LMMAXC==0) GOTO 210

                     ion: DO NI=NIS,WDES%NITYP(NT)+NIS-1
!DIR$ IVDEP
!OCL NOVREC
                        DO L=1,LMMAXC
!DIR$ IVDEP
!OCL NOVREC
                           DO LP=1,LMMAXC
                              CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)=CRHODE(LP,L,NI,ISP+ISPINOR_+2*ISPINOR)+ &
                                   WEIGHT*W1_RIGHT%CPROJ(L+LMBASE)*GCONJG(W1_LEFT%CPROJ(LP+LMBASE_))
                           ENDDO
                        ENDDO
                        
                        LMBASE = LMMAXC+LMBASE
                        LMBASE_= LMMAXC+LMBASE_

                     ENDDO ion

210                  NIS = NIS+WDES%NITYP(NT)
                  ENDDO typ
               ENDDO
            ENDDO spinor

         ENDDO band
      ENDDO kpoint
      ENDDO spin
      ! sum over all bands
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ*2))
#else
      CALLMPI( M_sum_d(WDES%COMM_INTER,CRHODE,LMDIM*LMDIM*WDES%NIONS*WDES%NCDIJ*2))
#endif
! note that one can save 4 lines above

      CALL DEALLOCATE_ROT_HANDLE(ROT_HANDLE)
      CALL DELWAV(W1_LEFT,  .FALSE.)
      CALL DELWAV(W1_RIGHT, .FALSE.)

      RETURN
      END SUBROUTINE DEPSUM_SYM_ASYM


!***************************** SUBROUTINE PS_BIOT_SAVART_DR ************************
!
! changes representation current density from reciprocal to direct,
! brings current to fine grid, entirely unnecessary, but nice for cosmetics,
! call to Biot-Savart in reciprocal space
!
! routine destroys contents input current JVEC_IN,
! routine returns magnetic fields on atoms B_OUT_PARA
!
! modelled from SUBROUTINE CURRENT
!
!***********************************************************************************

      SUBROUTINE PS_BIOT_SAVART_DR(JVEC_IN,GRID_SOFT,GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,T_INFO,LATT_CUR,WDES,B_OUT_PARA,B_NICS_OUT_PARA,CRHODE_AUG,LMDIM,IRDMAX,P,IU0,IU6)
      USE prec
      USE lattice
      USE morbitalmag
      USE mgrid
      USE poscar
      USE wave_high
      USE fileio
      IMPLICIT NONE
      TYPE (grid_3d)     GRIDC         ! full, fine grid
      TYPE (grid_3d)     GRID_SOFT     ! soft grid for pseudized potentials/ charge etc.
      TYPE (grid_3d) GRIDUS
      TYPE (transit) C_TO_US
      TYPE (transit)     SOFT_TO_C
      TYPE (latt)        LATT_CUR      ! lattice
      TYPE (type_info):: T_INFO        ! type information
      TYPE (wavedes)     WDES          ! descriptor for wavefunction
      TYPE (potcar), TARGET :: P(:)
      COMPLEX(q) :: B_OUT_PARA(3,T_INFO%NIONS)   ! output field
      COMPLEX(q) CRHODE_AUG(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ) ! augmentation input
      REAL(q) :: B_NICS_OUT_PARA(3,NPOSROT)      ! output NICS, used
      RGRID :: JVEC_IN(WDES%GRID%RL%NP,3)
      INTEGER LMDIM,IRDMAX
      INTEGER IU0, IU6

      ! local
      COMPLEX(q) TMP(3)
      COMPLEX(q), ALLOCATABLE :: JVEC(:)
      COMPLEX(q), ALLOCATABLE :: JTOT(:,:)
      REAL(q) DISPL_NIONS(3,T_INFO%NIONS)
      INTEGER M, I

!-----------------------------------------------------------------------
! change from reciprocal space presentation to cartesian coordinates
! sign change because electronic charge is negative
!-----------------------------------------------------------------------
      DO M=1,WDES%GRID%RL%NP
         DO I=1,3
            TMP(I)=JVEC_IN(M,I)             ! make real from ... well
         ENDDO
         CALL CDIRKAR(1,TMP,LATT_CUR%B)
         DO I=1,3
            JVEC_IN(M,I)=TMP(I)
         ENDDO
      ENDDO

!-----------------------------------------------------------------------
! bring the current density to the fine grid
!-----------------------------------------------------------------------

      ALLOCATE(JVEC(GRID_SOFT%MPLWV))
      ALLOCATE(JTOT(GRIDC%MPLWV,3))

      JTOT=0

      DO I=1,3
! now merge the current from all nodes
#ifdef realmode
         CALLMPI( M_sum_d(WDES%COMM_INTER, JVEC_IN(1,I), WDES%GRID%RL%NP))
         CALLMPI( M_sum_d(WDES%COMM_KINTER,JVEC_IN(1,I), WDES%GRID%RL%NP))
#else
         CALLMPI( M_sum_z(WDES%COMM_INTER, JVEC_IN(1,I), WDES%GRID%RL%NP))
         CALLMPI( M_sum_z(WDES%COMM_KINTER,JVEC_IN(1,I), WDES%GRID%RL%NP))
#endif
         CALL FFT_RC_SCALE(JVEC_IN(1,I),JVEC(1),GRID_SOFT)
         ! set the current density of unbalanced lattic-vectors to 0
         CALL SETUNB(JVEC(1),GRID_SOFT)

         ! bring to full grid
         IF (.NOT.WDES%LOVERL) THEN
            CALL RC_ADD(JVEC(1),1.0_q,JVEC(1),0.0_q,JTOT(1,I),GRID_SOFT)
         ELSE
           CALL ADD_GRID(GRIDC, GRID_SOFT, SOFT_TO_C,JVEC(1),JTOT(1,I))
           CALL SETUNB_COMPAT(JTOT(1,I),GRIDC)
        ENDIF
      ENDDO

      IF (LLRAUG) THEN
         DISPL_NIONS=0
         CRHODE_AUG(:,:,:,:)=-CRHODE_AUG(:,:,:,:) !GIL empirical test
         CALL CURRENT_AUGMENTATION_CMPLX( &
              WDES, GRIDC, GRIDUS, C_TO_US, LATT_CUR, P, T_INFO, &
              LMDIM, CRHODE_AUG(1,1,1,1), JTOT, IRDMAX, DISPL_NIONS, .TRUE.)
      ENDIF

      ! bring total pseudo current and total charge to real space
      DO I=1,3
         CALL FFT3D(JTOT(1,I),GRIDC,1)
      ENDDO

!-----------------------------------------------------------------------
! call Biot-Savart
!-----------------------------------------------------------------------


      CALL PS_BFIELD_C(JTOT(1,1),GRIDC,LATT_CUR,T_INFO,NPOSROT,POSROT,B_OUT_PARA,B_NICS_OUT_PARA,IU0,IU6)

      DEALLOCATE(JTOT)
      DEALLOCATE(JVEC)

      END SUBROUTINE PS_BIOT_SAVART_DR


!***********************************************************************
!
!***********************************************************************
      FUNCTION EIJK(I,J,K)
      USE prec
      INTEGER I,J,K
      REAL(q) EIJK
      EIJK=REAL((J-I)*(K-I)*(K-J)/2,q)
      END FUNCTION EIJK


!***********************************************************************
!
! Modelled from NABIJ_SOFT from optics.F
!
!***********************************************************************

      SUBROUTINE NABIJ_ASYM(W, WGKPQ, GRID, LATT_CUR, NKPT, INFO, NAB_QQ)
      USE prec
      USE lattice
      USE mgrid
      USE wave
      USE constant
      USE base
      IMPLICIT NONE

      TYPE (wavespin) W
      TYPE (wavespin) WGKPQ ! G u_i v_k+q, k| u>
      TYPE (grid_3d) GRID
      TYPE (latt)    LATT_CUR
      TYPE (info_struct) INFO
      COMPLEX(q)  NAB_QQ(3)     !not GDEF
      INTEGER NKPT

      !local
      INTEGER I, NB, NBP, IDIR, NK, ISP
      REAL(q) G1, G2, G3, GC, WEIGHT
      COMPLEX(q) CSUM

      NAB_QQ = 0

! loop over all (current) k-points, spin component and cartesian directions
      DO IDIR=1,3

      DO NK=1,NKPT
#ifdef MPI   
         IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         DO ISP=1,INFO%ISPIN
            DO NB =1,W%WDES%NBANDS
               WEIGHT=W%WDES%RSPIN*W%FERWE(NB,NK,ISP)*W%WDES%WTKPT(NK)     ! occupancies assumed identical
               DO I  =1,W%WDES%NPLWKP(NK)
                  G1=W%WDES%IGX(I,NK)+W%WDES%VKPT(1,NK)   ! beware take k-points from W.
                  G2=W%WDES%IGY(I,NK)+W%WDES%VKPT(2,NK)
                  G3=W%WDES%IGZ(I,NK)+W%WDES%VKPT(3,NK)
                  GC=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))*TPI

                  CSUM=WGKPQ%CW(I,NB,NK,ISP)*GC*CONJG(W%CW(I,NB,NK,ISP)) !*(0._q,1._q)  ! Gilles doesn't understand the i
                  NAB_QQ(IDIR)=NAB_QQ(IDIR)+CSUM*WEIGHT
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      ENDDO

!     CALLMPI(M_sum_z(W%WDES%COMM_KINTER,NAB_QQ(1),3))

      RETURN
      END SUBROUTINE NABIJ_ASYM

      END MODULE mlr_main_nmr


!************************* SUBROUTINE PSCURRENT_PARA_TWO_KPT ***********
!
! gradient psi contribution to current: Im[psi^*(r) nabla psi(r)]
! for a supplied weighted k-point and psi
! with psi(r) = e(iGr) 
! we get contributions proportional to (iG) e(iGr)
!
!***********************************************************************

      SUBROUTINE PSCURRENT_PARA_TWO_KPT(WDES1, GRID, CR_1, CR_2, IGX, IGY, IGZ, &
     &       VKPT_1, VKPT_2, WEIGHT, CWORK1, CWORK2, CW_1, CW_2, J)

      USE prec
      USE wave
      USE mgrid
      USE constant
      IMPLICIT NONE

      TYPE (wavedes1)    WDES1
      TYPE (grid_3d)     GRID
      COMPLEX(q) :: CR_1(GRID%MPLWV*WDES1%NRSPINORS)  ! wavefunction in real space
      COMPLEX(q) :: CR_2(GRID%MPLWV*WDES1%NRSPINORS)  ! wavefunction in real space
      COMPLEX(q) :: CW_1(WDES1%NRPLWV)
      COMPLEX(q) :: CW_2(WDES1%NRPLWV)
      REAL(q)    :: VKPT_1(3)                         ! k-point in reciprocal lattice
      REAL(q)    :: VKPT_2(3)                         ! k-point in reciprocal lattice
      REAL(q)    :: WEIGHT            ! same for both k-points
      INTEGER    :: IGX(WDES1%NGDIM)  ! component of each G vector in direction of first rec lattice vector
      INTEGER    :: IGY(WDES1%NGDIM)  ! component of each G vector in direction of second rec lattice vector
      INTEGER    :: IGZ(WDES1%NGDIM)  ! component of each G vector in direction of third rec lattice vector
! we assume both states consist of the same set of G-vectors
      COMPLEX(q) :: CWORK1(WDES1%NRPLWV)
      COMPLEX(q) :: CWORK2(GRID%MPLWV)
      RGRID      :: J(GRID%RL%NP,3)   ! three component vector storing current density in real space

! local
      INTEGER    :: ISPINOR, M, MM

      DO ISPINOR=0,WDES1%NRSPINORS-1

      ! first we take the gradient of wave function 2, fft that to real space,
      ! and multiply with 1

         ! component along first axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_2(MM)*(IGX(M)+VKPT_2(1))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)
  
         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,1)=J(M,1)+(CWORK2(M)*CONJG(CR_1(MM)))*WEIGHT / 2._q
         ENDDO

         ! component along second axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_2(MM)*(IGY(M)+VKPT_2(2))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)

         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,2)=J(M,2)+(CWORK2(M)*CONJG(CR_1(MM)))*WEIGHT / 2._q
         ENDDO

         ! component along third axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_2(MM)*(IGZ(M)+VKPT_2(3))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)

         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,3)=J(M,3)+(CWORK2(M)*CONJG(CR_1(MM)))*WEIGHT / 2._q
         ENDDO

      ! next we take the gradient of wave function 1, fft that to real space,
      ! and multiply with 2
      ! we keep the conjugation on the wave function 1

         ! component along first axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_1(MM)*(IGX(M)+VKPT_1(1))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)

         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,1)=J(M,1)+(CONJG(CWORK2(M))*CR_2(MM))*WEIGHT / 2._q
         ENDDO
           
         ! component along second axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_1(MM)*(IGY(M)+VKPT_1(2))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)

         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,2)=J(M,2)+(CONJG(CWORK2(M))*CR_2(MM))*WEIGHT / 2._q
         ENDDO
 
         ! component along third axis
         DO M=1,WDES1%NGVECTOR
            MM=M+ISPINOR*WDES1%NGVECTOR
            CWORK1(M)=CW_1(MM)*(IGZ(M)+VKPT_1(3))*TPI
         ENDDO
         ! FFT to real space
         CALL FFTWAV(WDES1%NGVECTOR,WDES1%NINDPW(1),CWORK2(1),CWORK1(1),GRID)

         DO M=1,GRID%RL%NP
            MM =M+ISPINOR *WDES1%GRID%MPLWV
            J(M,3)=J(M,3)+(CONJG(CWORK2(M))*CR_2(MM))*WEIGHT / 2._q
         ENDDO

      ENDDO

      END SUBROUTINE PSCURRENT_PARA_TWO_KPT


!************************* SUBROUTINE MLR_SYM_REDUCE *******************
!
!***********************************************************************
!#define debug
      SUBROUTINE MLR_SYM_REDUCE(LATT_CUR,T_INFO,SYMM,IU6)
      USE prec
      USE base
      USE lattice
      USE poscar
      IMPLICIT NONE
      TYPE(latt) LATT_CUR
      TYPE(type_info) T_INFO
      TYPE(symmetry) SYMM
      INTEGER IU6
      ! local variables
      INTEGER I,J,K,IS
      INTEGER INOT,INROT,ISORNOT,LIST(48)
      REAL(q) QST(3,3),RQST(3,3)
      REAL(q) V1,V2,V3
      REAL(q) MODQST,MODRQST,PRODMOD,DOT
      INTEGER ISYMOPTMP(3,3,48),IGRPOPTMP(3,3,48)
      REAL(q) GTRANSTMP(3,48)
      LOGICAL ISMAPPED(3)
      INTEGER IPGIND
      CHARACTER(4) GRPNAM
      REAL(q) TAU(T_INFO%NIONS,3),TAUROT(T_INFO%NIONS,3)
      ! symmetry common block
      INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
      REAL(q)  GTRANS,AP
      COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
           GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

#ifdef debug
      IF (IU6>0) THEN
      WRITE(*,'(A)') 'OLD ISYMOP'
      DO IS=1,NROTK 
         WRITE(*,'(A,I5)') 'SYM. OP',IS
         DO I=1,3
            WRITE(*,'(7I5)') (ISYMOP(I,J,IS),J=1,3), (IGRPOP(I,J,IS),J=1,3),INVMAP(IS)
         ENDDO
         WRITE(*,*) 'OLD GTRANS'
         WRITE(*,'(3F8.5)') (GTRANS(J,IS),J=1,3) 
      ENDDO  
      WRITE(*,*) 'NROT=',NROT,'NROTK=',NROTK
      ENDIF
#endif

      INOT=0; INROT=0; LIST=1 

      ! loop on sym. op. 
      isym: DO IS=1,NROTK
!        WRITE (*,'(A,I5)') 'SYMOP=',IS
         ! q-star
         QST=0._q; QST(1,1)=1._q; QST(2,2)=1._q; QST(3,3)=1._q

         ! cartesian to direct 
         DO I=1,3 ! 3 vectors
            V1=QST(1,I)*LATT_CUR%B(1,1)+QST(2,I)*LATT_CUR%B(2,1)+QST(3,I)*LATT_CUR%B(3,1)
            V2=QST(1,I)*LATT_CUR%B(1,2)+QST(2,I)*LATT_CUR%B(2,2)+QST(3,I)*LATT_CUR%B(3,2)
            V3=QST(1,I)*LATT_CUR%B(1,3)+QST(2,I)*LATT_CUR%B(2,3)+QST(3,I)*LATT_CUR%B(3,3)
            QST(1,I)=V1
            QST(2,I)=V2
            QST(3,I)=V3
         ENDDO

         ! rotate the q-star
         RQST=0._q
         DO K=1,3 ! x y z direction of IQST
            DO I=1,3 ! component of IRQST
               DO J=1,3 ! summation index (row column) 
                  RQST(I,K)=RQST(I,K)+IGRPOP(I,J,IS)*QST(J,K)     
               ENDDO
            ENDDO
         ENDDO

         ISORNOT = 0

         ! test colinearity abs ( u dot v ) == u*v
         idir: DO K=1,3 ! x y z direction for QST
            ISMAPPED=.FALSE.
            MODQST=QST(1,K)*QST(1,K)+QST(2,K)*QST(2,K)+QST(3,K)*QST(3,K)
            MODQST=SQRT(MODQST)
            DO I=1,3 ! x y z direction for RQST 
               DOT = 0.0_q
               MODRQST = RQST(1,I)*RQST(1,I) + RQST(2,I)*RQST(2,I) + RQST(3,I)*RQST(3,I)
               MODRQST = SQRT(MODRQST)
               PRODMOD = MODQST * MODRQST
               ! dot product   
               DO J=1,3 
                  DOT=DOT+RQST(J,I)*QST(J,K)
               ENDDO
               IF (ABS(ABS(DOT)-PRODMOD).LT.1E-8) THEN 
                  ISMAPPED(I)=.TRUE.
               ENDIF
#ifdef debug
              WRITE(*,'(A,3F9.5,A,L,3F9.5)') 'rotated ',RQST(:,I),' is it mapped ? ',ISMAPPED(I),DOT,PRODMOD,ABS(DABS(DOT)-PRODMOD)
#endif
            ENDDO

            IF ((.NOT.ISMAPPED(1)).AND.(.NOT.ISMAPPED(2)).AND.(.NOT.ISMAPPED(3)))  THEN
               ISORNOT=ISORNOT+1 
            ENDIF
#ifdef debug
           WRITE(*,*) IS , K,ISORNOT,ISMAPPED(1),ISMAPPED(2),ISMAPPED(3)
#endif
         ENDDO idir

         
         IF (ISORNOT.NE.0) THEN
#ifdef debug
            IF (IU6>0) WRITE(*,'(A,I3,A)') 'SYM. OP.',IS,' is NOT SUITABLE FOR LR_NMR and set to zero'
#endif
            INOT=INOT+1
            ISYMOP(:,:,IS)=0
            IGRPOP(:,:,IS)=0
            GTRANS(:,IS)=0
            LIST(IS)=0
         ENDIF
      ENDDO isym

#ifdef debug
      IF (IU6>0) WRITE(*,*) 'NUMBER OF SYMMETRY OPERATIONS NOT SUITABLE',INOT
#endif

      ! reorganize the arrays
      K=0
      DO IS=1,NROTK
         IF (LIST(IS).EQ.1) THEN
            K=K+1
            ISYMOPTMP(:,:,K)=ISYMOP(:,:,IS)
            IGRPOPTMP(:,:,K)=IGRPOP(:,:,IS)
            GTRANSTMP(:,K)   =GTRANS(:,IS)
            IF ((GTRANS(1,IS).EQ.0).AND.(GTRANS(2,IS).EQ.0).AND.(GTRANS(3,IS).EQ.0)) INROT=INROT+1
         ENDIF
      ENDDO

      ISYMOP(:,:,:)=ISYMOPTMP(:,:,:)
      IGRPOP(:,:,:)=IGRPOPTMP(:,:,:)
      GTRANS(:,:)  =GTRANSTMP(:,:)
      NROTK=NROTK-INOT
      NROT=INROT 

#ifdef debug
      IF (IU6>0) THEN
      WRITE(*,'(A)') 'NEW ISYMOP'
      DO IS=1,NROTK 
         WRITE(*,'(A,I5)') 'SYM. OP',IS
         DO I=1,3
            WRITE(*,'(7I5)') (ISYMOP(I,J,IS),J=1,3),(IGRPOP(I,J,IS),J=1,3),INVMAP(IS)
         ENDDO
         WRITE(*,*) 'NEW GTRANS'
         WRITE(*,'(3F8.5)') (GTRANS(J,IS),J=1,3) 
      ENDDO
      ENDIF
#endif

!=======================================================================
! Now try to find out which symmetry operations we have here:
!=======================================================================
      CALL PGROUP(ISYMOP,NROT,IPGIND,GRPNAM)
      IF (IU6>0) &
      WRITE(IU6,'(/4A)') '(MLR_SYM_REDUCE) The static configuration has the ', &
     &                                      'point symmetry ',GRPNAM,'.'
      IF (NROTK>NROT) THEN
         CALL PGROUP(ISYMOP,NROTK,IPGIND,GRPNAM)
         IF (IU6>0) &
         WRITE(IU6,'(/4A)') '(MLR_SYM_REDUCE) The point group associated with ', &
     &                             'its full space group is ',GRPNAM,'.'
      END IF
      DO I=1,T_INFO%NIONS
         TAU(I,1)=T_INFO%POSION(1,I)
         TAU(I,2)=T_INFO%POSION(2,I)
         TAU(I,3)=T_INFO%POSION(3,I)
      ENDDO
      IF (ASSOCIATED(SYMM%ROTMAP)) THEN
         DEALLOCATE(SYMM%ROTMAP)
      ENDIF
      ALLOCATE(SYMM%ROTMAP(T_INFO%NIONS,NROTK,NPCELL))
      CALL POSMAP(TAU,ISYMOP,GTRANS,NROTK,SYMM%PTRANS,NPCELL,T_INFO%NIONS,1, &
     &            T_INFO%NTYP,T_INFO%NIONS,T_INFO%NITYP,SYMM%ROTMAP(1,1,1),TAUROT)

!=======================================================================
! map of the inverse elements of each (reciprocal space) group element
!======================================================================
      CALL INVGRP(IGRPOP,NROTK,INVMAP)

      SYMM%NROT=NROTK

      RETURN
      END SUBROUTINE MLR_SYM_REDUCE
