#include "symbol.inc"
MODULE subrotscf
      USE prec
      USE wave
      USE mgrid
      
      TYPE (wavedes), POINTER, SAVE :: WDES2
      TYPE (grid_3d), POINTER, SAVE :: GRID2
      TYPE (grid_3d), POINTER, SAVE :: GRID_SOFT2
      TYPE (transit), POINTER, SAVE :: SOFT2_TO_C
      
      REAL(q), SAVE :: ENCUT_SUBROT_SCF
      
      LOGICAL, SAVE :: LINIT_SUBROT_SCF_DONE=.FALSE.
      
      CONTAINS
!**********************************************************************
! RCS:  $Id: electron.F,v 1.12 2003/06/27 13:22:15 kresse Exp kresse $
!
! subroutine to perform a complete selfconsistent optimization in
! the subspace spanned by the present set of wavefunctions
! written by Martijn Marsman, some cleanup by gK
!
! LHAM=.TRUE.
!     the wavefunctions are not updated instead a Hamiltonian
!     is returned that upon diagonalization will yield a rotation
!     matrix that leads directly to the selfconsistent minimum 
!     of the DFT part of the Hamiltonian
! LHAM=.FALSE.
!     the final wavefunction are unitarily transformed such 
!     that they diagonalize the selfconsistent Hamiltonian
!
!**********************************************************************

      SUBROUTINE SUBROT_SCF( &
          HAMILTONIAN,KINEDEN, &
          P,WDES,NONLR_S,NONL_S,W_ORIG,LATT_CUR, &
          T_INFO,INFO,IO,MIX,KPOINTS,SYMM, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,E_ORIG, &
          DENCOR,CSTRF,N_MIX_PAW,LMDIM,IRDMAX, &
          CHTOTL,RHOLM_LAST,CHAM,LHAM)

      USE prec
      USE pseudo
      USE lattice
      USE us
      USE pot
      USE nonl_high
      USE ini
      USE wave_high
      USE broyden
      USE msymmetry
      USE subrot
      USE base
      USE mpimy
      USE mgrid
      USE mkpoints
      USE constant
      USE poscar
      USE wave
      USE hamil_high
      USE pawm
      USE pawfock
      USE meta
      USE fock
! solvation__
      USE solvation
! solvation__
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
!=======================================================================
!  structures
!=======================================================================
      TYPE (ham_handle)  HAMILTONIAN
      TYPE (tau_handle)  KINEDEN
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (wavedes)     WDES
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (wavespin)    W_ORIG,W_F,W
      TYPE (latt)        LATT_CUR
      TYPE (info_struct) INFO
      TYPE (in_struct)   IO
      TYPE (mixing)      MIX
      TYPE (kpoints_struct) KPOINTS
      TYPE (symmetry)    SYMM
!     TYPE (grid_3d)     GRID       ! grid for wavefunctions
!     TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
      TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
      TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
      TYPE (grid_3d)     GRIDB      ! Broyden grid
      TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
      TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
!     TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
      TYPE (energy)      E_ORIG
      
      INTEGER LMDIM,IRDMAX
      REAL(q) :: TOTEN,EFERMI
      LOGICAL :: LHFCALC_SAVE
      COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
      COMPLEX(q)  CHTOTL(GRIDC%MPLWV,WDES%NCDIJ) 
      RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
      COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
      COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor

      GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)

! augmentation related quantities
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
               CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
               CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
! paw sphere charge density
      INTEGER N_MIX_PAW
      REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)
      REAL(q)  RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)
! charge-density and potential on soft grid
      COMPLEX(q)  CHDEN(GRID_SOFT2%MPLWV,WDES%NCDIJ)
      RGRID       SV(DIMREAL(GRID2%MPLWV),WDES%NCDIJ)
! density of states
      INTEGER, PARAMETER :: NEDOS=301
      REAL(q)    DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
      REAL(q) :: XCSIF(3,3)
! Hamiltonian or rotation matrix
      LOGICAL LHAM
! local 
      TYPE (energy) E
      REAL(q) :: TOTENL=0
      REAL(q) :: EDIFF,DESUM1,DESUM(INFO%NELM)
      INTEGER :: IONODE, NODE_ME
! local l-projected wavefunction characters (not really used here)
      REAL(q)    PAR(1,1,1,1,WDES%NCDIJ),DOSPAR(1,1,1,WDES%NCDIJ)

      INTEGER N,ISP,IRDMAA,IFLAG,ICOUEV,IERRBR
      REAL(q) RMS, RMST, WEIGHT, RMSC, RMSP, RMST_FIRST, EXHF_DUMMY
      LOGICAL LDELAY,LFIRST,LLAST

      INTEGER NK,N1,N2
      REAL(q) DIFFER,DIFCEL
      REAL(q), PARAMETER :: DIFMAX=1E-8
      LOGICAL :: OVER_BAND
      GDEF CROT,CEXPRE

      IONODE=0
      NODE_ME=0
#ifdef MPI
      IONODE  = WDES%COMM%IONODE
      NODE_ME = WDES%COMM%NODE_ME
#endif
      E=E_ORIG
      EDIFF=INFO%EDIFF

      NELM=INFO%NELM
      ! to make timing more sensefull syncronize now
      CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror ))
      CALL START_TIMING("SUBRT+")
      CALL START_TIMING("SUBROT")

      IF (IO%IU6>=0) WRITE(IO%IU6,*)

      IF (IO%IU0>=0) WRITE(IO%IU0,142)
142   FORMAT('         N     E                     dE            rms(c)')

      DESUM1=0
      INFO%LMIX=.FALSE.

      DWRITE0 'electron entered'

! well yet undecided, PUSH_FOCK implies that the onsite terms 
! are evaluated without the one-center Fock contributions
! this is consistent with the plane wave part
! and speeds up the code
      LHFCALC_SAVE=LHFCALC
      CALL PUSH_FOCK
      CALL ALLOCW(WDES,W)
      W%WDES => WDES2
!=======================================================================
! calculate CHTOT, CHDEN, CRHODE, and RHOLM
! done here to make the routine more selfcontained
!=======================================================================
      LFIRST=.TRUE.
      LLAST=.FALSE.
      ! reset mixer
      MIX%LRESET=.TRUE.

      ! store original storage convention
      OVER_BAND=W_ORIG%OVER_BAND
      ! distribute over bands, no matter what we had originally
      CALL REDIS_PW_OVER_BANDS(W%WDES, W_ORIG)

      W%CW    =W_ORIG%CW
      W%CPROJ =W_ORIG%CPROJ
      W%CELTOT=W_ORIG%CELTOT
      W%FERTOT=W_ORIG%FERTOT
#ifdef MPI
      IF (LHFCALC_SAVE.AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
         CALL MRG_FERWE(WDES,W)
      END IF
#endif
      CALL SET_CHARGE(W, W%WDES, INFO%LOVERL, &
           GRID2, GRIDC, GRID_SOFT2, GRIDUS, C_TO_US, SOFT2_TO_C, &
           LATT_CUR, P, SYMM, T_INFO, &
           CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

      CALL SET_KINEDEN(GRID2,GRID_SOFT2,GRIDC,SOFT2_TO_C,LATT_CUR,SYMM, &
           T_INFO%NIONS,W,W%WDES,KINEDEN)      

      CALL POTLOK(GRID2,GRIDC,GRID_SOFT2, W%WDES%COMM_INTER, W%WDES, &
           INFO,P,T_INFO,E,LATT_CUR, &
           CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT2_TO_C,XCSIF)

      CALL POTLOK_METAGGA(KINEDEN, &
           GRID2,GRIDC,GRID_SOFT2,W%WDES%COMM_INTER,W%WDES,INFO,P,T_INFO,E,LATT_CUR, &
           CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT2_TO_C,XCSIF)

      CALL SETDIJ(W%WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
           LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

      CALL SET_DD_PAW(W%WDES, P , T_INFO, INFO%LOVERL, &
           W%WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
           E,  LMETA=.FALSE., LASPH=(INFO%LASPH.AND.LDO_METAGGA()), LCOREL= .FALSE.  )
!           E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

      CALL UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IO%IU6)

      IF (INFO%LDIAG) THEN
         IFLAG=3    ! exact diagonalization
      ELSE
         IFLAG=4    ! using Loewdin perturbation theory
      ENDIF
         
      W%CW    =W_ORIG%CW
      W%CPROJ =W_ORIG%CPROJ
      W%CELTOT=W_ORIG%CELTOT
      W%FERTOT=W_ORIG%FERTOT
      W%OVER_BAND=W_ORIG%OVER_BAND
         
      CALL EDDIAG(HAMILTONIAN,GRID2,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
           LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY, &
           CHAM,LFIRST,LLAST)
      LFIRST=.FALSE.

      CHTOT=CHTOTL
      RHOLM=RHOLM_LAST
!=======================================================================
      electron: DO N=1,NELM
!=======================================================================
   
!=======================================================================
! if recalculation of total lokal potential is necessary
! ) the hartree potential from the electronic  charge density
! ) the exchange correlation potential
! ) and the total lokal potential
!  in addition all double counting correction and forces are calculated
! &
! call SETDIJ
! calculates the Integral of the depletion charges * local potential
! and sets CDIJ
!=======================================================================
      CALL POTLOK(GRID2,GRIDC,GRID_SOFT2, W%WDES%COMM_INTER, W%WDES, &
                  INFO,P,T_INFO,E,LATT_CUR, &
                  CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT2_TO_C,XCSIF)
      CALL POTLOK_METAGGA(KINEDEN, &
                  GRID2,GRIDC,GRID_SOFT2,W%WDES%COMM_INTER,W%WDES,INFO,P,T_INFO,E,LATT_CUR, &
                  CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT2_TO_C,XCSIF)
      DWRITE0 'potlok is ok'
!     CALL STOP_TIMING("SUBROT",IO%IU6,"POTLOK")

      CALL SETDIJ(W%WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                  LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

      CALL SET_DD_PAW(W%WDES, P , T_INFO, INFO%LOVERL, &
         W%WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
         E,  LMETA=.FALSE., LASPH=(INFO%LASPH.AND.LDO_METAGGA()), LCOREL= .FALSE.  )
!         E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
      DWRITE0 'setdij is ok'
!     CALL STOP_TIMING("SUBROT",IO%IU6,"SETDIJ")

      CALL UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IO%IU6)

      DESUM1=0
      RMS   =0
      ICOUEV=0
!=======================================================================
! sub space rotation
!=======================================================================
      IF (INFO%LDIAG) THEN
         IFLAG=3    ! exact diagonalization
      ELSE
         IFLAG=4    ! using Loewdin perturbation theory
      ENDIF

      W%CW    =W_ORIG%CW
      W%CPROJ =W_ORIG%CPROJ
      W%CELTOT=W_ORIG%CELTOT
      W%FERTOT=W_ORIG%FERTOT
      W%OVER_BAND=W_ORIG%OVER_BAND

      CALL EDDIAG(HAMILTONIAN,GRID2,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
           LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY, &
           CHAM,LFIRST,LLAST)
      DWRITE0 "eddiag is ok"
!     CALL STOP_TIMING("SUBROT",IO%IU6,"EDDIAG")

!=======================================================================
! recalculate the broadened density of states and fermi-weights
! recalculate depletion charge size
!=======================================================================
      CALL MRG_CEL(W%WDES,W)

      E%EENTROPY=0
      DOS=0
      DOSI=0
      CALL DENSTA( IO%IU0, IO%IU6, W%WDES, W, KPOINTS, INFO%NELECT, &
               INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE., &
               NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)
!=======================================================================
! calculate free-energy and bandstructur-energy
! EBANDSTR = sum of the energy eigenvalues of the electronic states
!         weighted by the relative weight of the special k point
! TOTEN = total free energy of the system
!=======================================================================
      E%EBANDSTR=BANDSTRUCTURE_ENERGY(W%WDES, W)

      TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+INFO%EALLAT+E%EXHF+Ediel_sol

!---- write total energy to OSZICAR file and stdout
      DESUM(N)=TOTEN-TOTENL
      ECONV=DESUM(N)

      IF (IO%IU0>=0) WRITE(IO%IU0,  400,ADVANCE='NO')  N,TOTEN,DESUM(N)
400   FORMAT('   ROT: ',I2,' ',E20.12,'   ',E12.5)
!========================= subroutine CHSP  ============================
! if charge density is updated
!  ) first copy current charge to CHTOTL
!  ) set INFO%LMIX to .T.
!  ) call subroutine SET_CHARGE to generate the new charge density
!  ) then performe mixing
!=======================================================================
      INFO%LMIX=.FALSE.
      MIX%NEIG=0

      DO ISP=1,W%WDES%NCDIJ
         CALL RC_ADD(CHTOT(1,ISP),1.0_q,CHTOT(1,ISP),0.0_q,CHTOTL(1,ISP),GRIDC)
      ENDDO

      IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
         DO ISP=1,W%WDES%NCDIJ
            CALL RC_ADD(KINEDEN%TAU(1,ISP),1.0_q,KINEDEN%TAU(1,ISP),0.0_q,KINEDEN%TAUL(1,ISP),GRIDC)
         ENDDO
      ENDIF

      RHOLM_LAST=RHOLM
#ifdef MPI
      IF (LHFCALC_SAVE.AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
         CALL MRG_FERWE(WDES,W)
      END IF
#endif
      CALL SET_CHARGE(W, W%WDES, INFO%LOVERL, &
           GRID2, GRIDC, GRID_SOFT2, GRIDUS, C_TO_US, SOFT2_TO_C, &
           LATT_CUR, P, SYMM, T_INFO, &
           CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

      CALL SET_KINEDEN(GRID2,GRID_SOFT2,GRIDC,SOFT2_TO_C,LATT_CUR,SYMM, &
           T_INFO%NIONS,W,W%WDES,KINEDEN)      

!     CALL STOP_TIMING("SUBROT",IO%IU6,"CHARGE")

!-----------------------------------------------------------------------

      IF (MIX%IMIX/=0) THEN
         INFO%LMIX=.TRUE.
         IF (MIX%IMIX==4) THEN
!  broyden mixing ... :
            CALL BRMIX(KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
            &   (2*GRIDC%MPLWV),CHTOT,CHTOTL,W%WDES%NCDIJ,LATT_CUR%B, &
            &   LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
            &   RMST,RMSC,RMSP,WEIGHT,.TRUE.,IERRBR)
            MIX%LRESET=.FALSE.
         ELSE
!  simple mixing ... :
            RMST=0
            CALL MIX_SIMPLE(GRIDC,MIX,W%WDES%NCDIJ, CHTOT,CHTOTL, &
            N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
         ENDIF

         IF (N==1) RMST_FIRST=RMST
         DWRITE0 "mixing is ok"
      ENDIF
!     CALL STOP_TIMING("SUBROT",IO%IU6,"MIXING")
!=======================================================================
! total time used for this step
!=======================================================================
      CALL STOP_TIMING("SUBROT",IO%IU6)
!=======================================================================
!  important write statements
!=======================================================================
!=======================================================================
!  Test for Break condition
!=======================================================================
      INFO%LABORT=.FALSE.
!-----conjugated gradient eigenvalue and energy must be converged
      IF(ABS(DESUM(N))<EDIFF/10.AND.ABS(DESUM1)<EDIFF/10.AND.N>1) INFO%LABORT=.TRUE.
!-----but stop after INFO%NELM steps no matter where we are now
      IF (N>=INFO%NELM) INFO%LABORT=.TRUE.
!-----rms decreased by factor 20
      IF(ABS(RMST)<ABS(RMST_FIRST/20) .AND. N>=2) INFO%LABORT=.TRUE.

 308  FORMAT('   ',E10.3)
      IF ( INFO%LMIX .AND. MIX%IMIX==4 .AND. IO%IU0>=0) THEN
         WRITE(IO%IU0,308) RMST
         IF (IERRBR/=0) THEN
            WRITE(IO%IU0,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
            'mixing'' now and reset mixing at next step!'
            WRITE(IO%IU6,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
            'mixing'' now and reset mixing at next step!'
         ENDIF
      ELSE IF (INFO%LMIX .AND. IO%IU0>=0) THEN
         WRITE(IO%IU0,308) RMST
      ELSE IF (IO%IU0>=0) THEN
         WRITE(IO%IU0,*)
      ENDIF

!======================== end of loop ENDLSC ===========================
! This is the end of the selfconsistent calculation loop
!=======================================================================
      IF (INFO%LABORT) EXIT electron
      TOTENL=TOTEN

      ENDDO electron

      IF (IO%LOPEN) CALL WFORCE(IO%IU6)
      CALL SEPERATOR_TIMING(IO%IU6)
      CALL STOP_TIMING("SUBRT+",IO%IU6)

      IF (MIX%IMIX==4 .AND. (IO%NWRITE>=3 .OR. INFO%LABORT) .AND. IO%IU6>=0) THEN
 2441 FORMAT(/ &
     &       ' Broyden mixing:'/ &
     &       '  rms(total) =',E12.5,'    rms(broyden)=',E12.5,/ &
     &       '  rms(prec ) =',E12.5/ &
     &       '  weight for this iteration ',F10.2)

 2442 FORMAT(/' eigenvalues of (default mixing * dielectric matrix)' / &
             '  average eigenvalue GAMMA= ',F8.4,/ (10F8.4))

         WRITE(IO%IU6,2441) RMST,RMSC,RMSP,WEIGHT
         IF (ABS(RMST-RMSC)/RMST> 0.1_q) THEN
            WRITE(IO%IU6,*) ' WARNING: grid for Broyden might be too small'
         ENDIF
         IF (MIX%NEIG > 0) THEN
            WRITE(IO%IU6,2442) MIX%AMEAN,MIX%EIGENVAL(1:MIX%NEIG)
         ENDIF
         WRITE(IO%IU6,*)
      ENDIF

      LLAST=.TRUE.
! determine the Hamiltonian in the subspace spanned by the wavefunctions
! one last time to get the self-consistent Hamiltonian matrix
      IF (LHAM) THEN
         IFLAG=1
      ELSE IF (INFO%LDIAG) THEN
         IFLAG=3    ! exact diagonalization
      ELSE
         IFLAG=4    ! using Loewdin perturbation theory
      ENDIF

      W%CW       =W_ORIG%CW
      W%CPROJ    =W_ORIG%CPROJ
      W%CELTOT   =W_ORIG%CELTOT
      W%FERTOT   =W_ORIG%FERTOT
      W%OVER_BAND=W_ORIG%OVER_BAND

      CALL EDDIAG(HAMILTONIAN,GRID2,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
           LMDIM,CDIJ,CQIJ, IFLAG,SV,T_INFO,P,IO%IU0,EXHF_DUMMY, &
           CHAM,LFIRST,LLAST)

      IF (.NOT. LHAM) THEN
      ! return the updated wavefunctions, und eigenvalues
         W_ORIG%CW       =W%CW
         W_ORIG%CPROJ    =W%CPROJ
         W_ORIG%CELTOT   =W%CELTOT
         W_ORIG%FERTOT   =W%FERTOT
         W_ORIG%OVER_BAND=W%OVER_BAND
         E_ORIG=E
      ENDIF

      IF (OVER_BAND) THEN
         CALL REDIS_PW_ALL(W%WDES, W_ORIG)
      ENDIF

      CALL DEALLOCW(W)

!     io_begin
!     CALL DUMP_HAM( "Hamilton matrix",W%WDES, CHAM(:,:,1,1))
!     io_end

      CALL POP_FOCK
      DWRITE0 'electron left'

      RETURN
      END SUBROUTINE SUBROT_SCF


      SUBROUTINE SETUP_SUBROT_SCF(INFO,WDES,LATT_CUR,GRID,GRIDC,GRID_SOFT,SOFT_TO_C,IU0,IU5,IU6)
      USE prec
      USE base
      USE constant
      USE wave
      USE mgrid
      USE lattice
      USE full_kpoints
      IMPLICIT NONE
      TYPE (latt) LATT_CUR
      TYPE (wavedes), TARGET :: WDES
      TYPE (grid_3d), TARGET :: GRID
      TYPE (grid_3d), TARGET :: GRIDC
      TYPE (grid_3d), TARGET :: GRID_SOFT
      TYPE (transit), TARGET :: SOFT_TO_C
      TYPE (info_struct) :: INFO
      INTEGER :: IU0,IU5,IU6

      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC
      
      INTEGER :: NGX,NGY,NGZ
      INTEGER :: NK, N1, N2, N3, IND
      REAL(q) :: G1, G2, G3, GIX, GIY, GIZ, ENERGI
      REAL(q) :: WFACT
      COMPLEX(q), ALLOCATABLE :: CWORK1(:)

!     IF (LINIT_SUBROT_SCF_DONE.OR.(.NOT.INFO%LDIAG).OR.(.NOT.INFO%LONESW)) RETURN
      IF (LINIT_SUBROT_SCF_DONE.OR.(.NOT.INFO%LSUBROT).OR.(.NOT.INFO%LONESW)) RETURN

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
      ENCUT_SUBROT_SCF=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'ENCUTSUBROTSCF','=','#',';','F', &
           &            IDUM,ENCUT_SUBROT_SCF,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''ENCUTSUBROTSCF'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('ENCUTSUBROTSCF','F',IDUM,ENCUT_SUBROT_SCF,CDUM,LDUM,CHARAC,N)
      CLOSE(IU5)

      IF (ENCUT_SUBROT_SCF<0) THEN
         ALLOCATE(WDES2,GRID2,GRID_SOFT2,SOFT2_TO_C)
         WDES2=>WDES
         GRID2=>GRID
         GRID_SOFT2=>GRID_SOFT
         SOFT2_TO_C=>SOFT_TO_C
         LINIT_SUBROT_SCF_DONE=.TRUE.
         RETURN
      ENDIF

      ALLOCATE(GRID2)
      ALLOCATE(WDES2)
            
      GRID2=GRID
      WDES2=WDES
      
      ! determine minimum required values for NGX, Y and NGZ
      NGX=1
      NGY=1
      NGZ=1

      ! loop over all k-points in the entire BZ
      ! to determine the required NGX
      DO N3=1,GRID%NGZ_rd
         DO N2=1,GRID%NGY
            DO NK=1,WDES%NKPTS
               G3=(GRID%LPCTZ(N3)+WDES%VKPT(3,NK))
               G2=(GRID%LPCTY(N2)+WDES%VKPT(2,NK))
               DO N1=1,GRID%NGX_rd
                  G1=(GRID%LPCTX(N1)+WDES%VKPT(1,NK))
                  GIX= (G1*LATT_CUR%B(1,1)+G2*LATT_CUR%B(1,2)+G3*LATT_CUR%B(1,3)) *TPI
                  GIY= (G1*LATT_CUR%B(2,1)+G2*LATT_CUR%B(2,2)+G3*LATT_CUR%B(2,3)) *TPI
                  GIZ= (G1*LATT_CUR%B(3,1)+G2*LATT_CUR%B(3,2)+G3*LATT_CUR%B(3,3)) *TPI
                  ENERGI=HSQDTM*((GIX**2)+(GIY**2)+(GIZ**2))

                  ! exclude some components for gamma-only version (C(G)=C*(-G))
                  IF (GRID%NGZ/=GRID%NGZ_rd) THEN
                     IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)<0) CYCLE
                     IF (GRID%LPCTZ(N3)==0 .AND.GRID%LPCTY(N2)==0 .AND.GRID%LPCTX(N1)<0) CYCLE
                  ENDIF

                  IF(ENERGI<WDES%ENMAX) THEN
                     NGX=MAX(NGX,ABS(GRID%LPCTX(N1))*2+2)
                     NGY=MAX(NGY,ABS(GRID%LPCTY(N2))*2+2)
                     NGZ=MAX(NGZ,ABS(GRID%LPCTZ(N3))*2+2)
                  ENDIF
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      ! in principle that should do
      ! however the previous statments sometimes leads to meshes that violate
      ! symmetry
      ! search for the next larger values that conserves symmetry
      WFACT=MAX(NGX/LATT_CUR%ANORM(1),NGY/LATT_CUR%ANORM(2),NGZ/LATT_CUR%ANORM(3))+1E-6
      NGX=WFACT*LATT_CUR%ANORM(1)
      NGY=WFACT*LATT_CUR%ANORM(2)
      NGZ=WFACT*LATT_CUR%ANORM(3)

      GRID2%NGPTAR(1)=NGX
      GRID2%NGPTAR(2)=NGY
      GRID2%NGPTAR(3)=NGZ            

      CALL FFTCHK(GRID2%NGPTAR)
      ! reinitialize the loop counters
      CALL INILGRD(GRID2%NGPTAR(1),GRID2%NGPTAR(2),GRID2%NGPTAR(3),GRID2)
      ! regenerate the layout
#ifdef MPI
      CALL MAPSET(GRID2)
#endif
      IF (IU6>=0) THEN
         WRITE(IU6,20) GRID2%NGPTAR
20       FORMAT(/' FFT grid used in determination of subspace rotation ',/ &
              '  NGX =',I3,'; NGY =',I3,'; NGZ =',I3,/)
      ENDIF
      
      CALL GEN_LAYOUT(GRID2, WDES2, LATT_CUR%B, LATT_CUR%B, IU6, .TRUE.)
      CALL GEN_INDEX (GRID2, WDES2, LATT_CUR%B, LATT_CUR%B,-1, -1, .TRUE.)
      CALL CHECK_GEN_LAYOUT( WDES,  WDES2, WDES%NKPTS)
      !  init FFT (required if real to complex FFT is used)
      CALL FFTINI(WDES2%NINDPW(1,1),WDES2%NGVECTOR(1),WDES2%NKPTS,WDES2%NGDIM,GRID2)
      ! initialize FFT
      ALLOCATE(CWORK1(GRID2%MPLWV+1024))
      DO IND=1,MIN(4,WDES%NBANDS*WDES%NKPTS)
         CALL INIDAT(GRID2%RC%NP,CWORK1)
         CALL FFTMAKEPLAN(CWORK1(IND),GRID2)
      ENDDO
      DEALLOCATE(CWORK1)

      ! generate GRID_SOFT2
      ALLOCATE(GRID_SOFT2,SOFT2_TO_C)
      GRID_SOFT2=GRID_SOFT
      CALL INILGRD(GRID2%NGX,GRID2%NGY,GRID2%NGZ,GRID_SOFT2)
      CALL GEN_RC_SUB_GRID(GRID_SOFT2, GRIDC, SOFT2_TO_C, .TRUE., .TRUE.)
      CALL SET_RL_GRID(GRID_SOFT2,GRID2)
#ifdef MPI
      CALL MAPSET(GRID_SOFT2)
#endif

      LINIT_SUBROT_SCF_DONE=.TRUE.

      RETURN
      END SUBROUTINE SETUP_SUBROT_SCF


END MODULE subrotscf
