#include "symbol.inc"

MODULE mlrf_main
  USE prec
  USE vaspxml
  USE lr_helper
  USE subrot_cluster
  IMPLICIT NONE

  LOGICAL :: LEPSILON=.FALSE.
  LOGICAL :: LNABLA=.FALSE.
  LOGICAL :: LVEL=.FALSE.
  LOGICAL :: LINTERFAST=.FALSE.
  INTEGER :: KINTER=0
  LOGICAL :: LRPA=.FALSE.
  REAL(q) :: LCSHIFT=0.1_q
  REAL(q) :: OMEGAMAX_OPTIC
  REAL(q) :: RTIME=0.0_q      !
  REAL(q) :: WPLASMAI=0.0_q   ! imaginary part of plasma frequency in eV
                              ! this is somehow related to RTIME, however units of RTIME are complicated
  REAL(q) :: DFIELD(3)
  LOGICAL, SAVE :: DFIELD_DONE=.FALSE.
  REAL(q) :: WPLASMA(3,3)     ! tensor of squared plasma frequency

CONTAINS

!*********************************************************************
!
! calculate first derivatives of the wavefunctions with respect
! to an external field using linear response theory
! implemented by gK
! this allows to calculate the macroscopic static dielectric tensor
! epsilon_macro or epsilon^infinity
! with the inclusion of local field effects
!
! comment on resolving the degeneracy problem:
! presently the same treatment as in ilinear_response.F is used
!
! however, this can be commented out without significantly 
! changing the results (uncomment SUBROT_DEG_ALL)
! reasoning:
! ) this routine is for insulators only, where the problem is
!   generally less severe (no splitting of initially
!    partially occupied states)
! ) <phi(0) | H(1)| phi(0)> approx
!         <phi(0) | xi(0)>+ \sum_i  <phi(0) | xi(0)_i |p_i>) approx
!         <phi(0) | S(0) | xi(0)> = 0 for the occupied subspace
!   hence the response into other occupied states is essentially zero
!   this holds in particular for degenerate eigenvalue pairs
!
!*********************************************************************

  SUBROUTINE LRF_MAIN( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W0,CXI0,CXI0_CPROJ,LATT_CUR, &
          T_INFO,DYN,INFO,IO,MIX,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C,E, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV,DOS,DOSI, &
          LMDIM,IRDMAX,NEDOS, &
          TOTEN,EFERMI, DEG_CLUSTER, KPOINTS_TRANS, EPSILON, BORN_CHARGES, PIEZO, & 
          IDIR, UPDATE_CHARGE)

    USE base
    USE lattice
    USE finite_differences
    USE charge
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE msymmetry
    USE mpimy
    USE mgrid
    USE mkpoints
    USE constant
    USE poscar
    USE wave
    USE pot
    USE ini
    USE pawm
    USE hamil_lrf
    USE rmm_diis
    USE rmm_diis_lr
    USE subrot_lr
    USE us
    USE wave_high
    USE choleski
    USE broyden
    USE subrot
    USE subrot_cluster
    USE kpoints_change
    USE hamil_high
    USE setexm
    USE meta
! solvation__
      USE solvation
! solvation__
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (tau_handle)  KINEDEN
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W0         ! unperturbed wavefunctions
    COMPLEX(qs)        CXI0(:,:,:,:,:)
    GDEFS              CXI0_CPROJ(:,:,:,:,:)
    TYPE (latt)        LATT_CUR
    TYPE (dynamics)    DYN
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (mixing)      MIX
    TYPE (kpoints_struct) KPOINTS
    TYPE (symmetry)    SYMM
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (grid_3d)     GRIDB      ! Broyden grid
    TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE (energy)      E
    
    INTEGER LMDIM,IRDMAX,NEDOS
    REAL(q) TOTEN,EFERMI

    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q)  CHTOTL(GRIDC%MPLWV,WDES%NCDIJ)! old charge-density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ),RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
!  density of states
    REAL(q)    DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
! local l-projected wavefunction characters (not really used here)
    REAL(q)    PAR(1,1,1,1,WDES%NCDIJ),DOSPAR(1,1,1,WDES%NCDIJ)
    TYPE (skpoints_trans)   :: KPOINTS_TRANS
    TYPE (eigenf_cluster_pointer) :: DEG_CLUSTER(WDES%NKPTS,WDES%ISPIN)
    INTEGER :: IDIR                      ! direction
    REAL(q) :: EPSILON(:), BORN_CHARGES(:,:), PIEZO(:,:)
! flag whether potential and charge is updated
! if UPDATE_CHARGE is .FALSE. local field effects are not included
    LOGICAL :: UPDATE_CHARGE
!
! variable determines the step width for the finite differences
! used to calculate derivatives with respect to the charge
! here a relatively large step can be used, because the changes upon
! an external field are small
    REAL(q) :: POTIM
! local variables
    INTEGER :: IONODE, NODE_ME
    INTEGER :: IRDMAA
#ifdef gammareal
    REAL(q) :: CSHIFT=0.000
#else
    REAL(q) :: CSHIFT=0.002              ! complex shift
                                         ! improves stability in particular
                                         ! if degenerated eigenvalue pairs have not been resolved
#endif
    REAL(q) :: DESUM,DE                  ! change of energy
    REAL(q) :: RMS(INFO%NELM)            ! magnitude of residual vector
    REAL(q) :: RMST(INFO%NELM)           ! magnitude of charge residual
    INTEGER :: ICOUEV                    ! number of H | phi> evaluations
    REAL(q) :: TVPUL0,TCPUL0,TVPUL,TCPUL ! timing related
    INTEGER :: N                         ! main loop counter (electronic step)
    INTEGER :: NORDER                    ! order in the MP smearing
    REAL(q) :: TOTENL                    ! total energy in previous step
    COMPLEX(q),ALLOCATABLE :: CHTOT1(:,:)! derivative change of total charge
    COMPLEX(q),ALLOCATABLE :: CHTOT2(:,:)! derivative change of total charge
    COMPLEX(q),ALLOCATABLE :: CVTOT1(:,:)! derivative of local potential
    COMPLEX(q),ALLOCATABLE :: CWORK (:,:)! derivative of local potential
    OVERLAP,ALLOCATABLE :: CDIJ1(:,:,:,:)! derivative of non local strength
    OVERLAP,ALLOCATABLE :: CDIJ2(:,:,:,:)! work array
    OVERLAP,ALLOCATABLE::CRHODE1(:,:,:,:)! derivative of onsite occupancies
    OVERLAP,ALLOCATABLE::CTMP(:,:,:,:)   ! temporary
    RGRID,ALLOCATABLE   :: SV1(:,:)      ! derivative of soft potential
    RGRID,ALLOCATABLE   :: SV2(:,:)      ! work array
    COMPLEX(q),ALLOCATABLE:: CHDEN1(:,:) ! 1st order change of pseudo charge
    REAL(q)  RHOLM1(N_MIX_PAW,WDES%NCDIJ)! one center occupancy matrix (mixed)

    TYPE (wavespin)     :: WXI           ! stores H(1) - epsilon S(1) | phi_0>
    TYPE (wavespin)     :: W1            ! first order change of wavefunction
    TYPE (wavefun)      :: WTMP          ! temporary
    TYPE (energy)          E1,E2         ! energy arrays
    INTEGER I, J                         ! loop counter for various cases
    INTEGER NI
    INTEGER ISP, NK, M
    REAL(q)             :: RMSC,RMSP,WEIGHT ! mixer 
    INTEGER             :: IERRBR
    REAL(q)   POL(3)
    REAL(q), EXTERNAL :: TR_POT_CHARGE
    INTEGER :: ierror
    LOGICAL :: LCORR_STORE
! forces and stress tensors
    REAL(q)   XCSIF(3,3),EWSIF(3,3),TSIF(3,3),PRESS,VTMP(3),F(3,T_INFO%NIONS),SIF(3,3)
    REAL(q)   TIFOR(3,T_INFO%NIOND),  FACT
    GDEF, ALLOCATABLE :: CPROJXYZ(:,:,:)
    INTEGER :: ISPINOR, LBASE, L, LP, LMMAXC, NIS, NIP, NT, NDIR
    REAL(q) :: ENL(T_INFO%NIONS), EWISIF(6), EWIFOR(3,T_INFO%NIOND)
! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

    PROFILING_START('lrf_main')

    IONODE=0
    NODE_ME=0
#ifdef MPI
    IONODE  = WDES%COMM%IONODE
    NODE_ME = WDES%COMM%NODE_ME
#endif
!=======================================================================
!  allocate all required work arrays
!=======================================================================
    POTIM=0.1
    CALL REINIT_DEG_CLUSTERS(WDES,DEG_CLUSTER)

    ALLOCATE(CHTOT1(GRIDC%MPLWV,WDES%NCDIJ), &
             CHTOT2(GRIDC%MPLWV,WDES%NCDIJ), &
             CVTOT1(GRIDC%MPLWV,WDES%NCDIJ), &
             CWORK(GRIDC%MPLWV,WDES%NCDIJ), &
             CDIJ1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CDIJ2(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CRHODE1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CTMP(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             SV1(DIMREAL(GRID%MPLWV),WDES%NCDIJ), &
             SV2(DIMREAL(GRID%MPLWV),WDES%NCDIJ), &
             CHDEN1(GRID_SOFT%MPLWV,WDES%NCDIJ))

    CALL ALLOCW(WDES,WXI,WTMP,WTMP)
    CALL ALLOCW(WDES,W1,WTMP,WTMP)

    W1%CW   =0
    W1%CPROJ=0
!=======================================================================
!  initialise required work arrays
!=======================================================================

!  1st derivative of charge in the first iteration is 0
    CRHODE1=0
    RHOLM1=0
    CHTOT1=0

    ! to make timing more sensefull syncronize now
    CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror ))
    CALL START_TIMING("LOOP")

    
    IF (IO%IU0>=0) THEN
       WRITE(IO%IU0,142)
       WRITE(17,142)
      ENDIF
142 FORMAT('       N       E                     dE             ' &
         ,'d eps       ncg     rms          rms(c)')

    INFO%LMIX=.FALSE.

130 FORMAT (5X, //, &
     &'----------------------------------------------------', &
     &'----------------------------------------------------'//)

140 FORMAT (5X, //, &
     &'----------------------------------------- Iteration ', &
     &I4,'(',I4,')  ---------------------------------------'//)
    DWRITE0 'electron entered'

!     CALL DIPOL_RESET()
!=======================================================================
!  main selfconsistent loop for linear response
!=======================================================================
    MIX%LRESET=.TRUE.
    TOTEN=0

    INFO%LCHCON=.FALSE.

!   uncomment these lines if you want to include only Hartree terms
    IF (IO%IU0>=0 .AND. LRPA .AND. UPDATE_CHARGE) & 
      WRITE(IO%IU0,*) "local field effects only on Hartree level (RPA) included"
    IF (LRPA) CALL PUSH_LEXCH(-1)

!   zero charge residuum
    RMST=0
electron: DO N=1,INFO%NELM
       IF(IO%IU6>=0) WRITE(IO%IU6,140) IDIR,N
       CALL XML_TAG("scstep")
!=======================================================================
!  calculate first derivative of potential 
!  and derivatives of non local strenght
!  this presently done using finite differences 
!  the results are stored in 
!   CVTOT1
!   SV1
!   CDIJ1
!=======================================================================
       CALL START_TIMING("G")
! positive displacement
       CHTOTL=CHTOT  +CHTOT1*(POTIM/2)
       CVTOT1=0
       CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
            INFO,P,T_INFO,E1,LATT_CUR, &
            CHTOTL,CSTRF, CVTOT1,DENCOR,SV1, SOFT_TO_C,XCSIF)

       CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
            LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)

       CTMP=CRHODE+CRHODE1*(POTIM/2)
       RHOLM_LAST=RHOLM+RHOLM1*(POTIM/2)
       CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
            WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1), RHOLM_LAST , CTMP(1,1,1,1), &
            E1,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

! negative displacement

       CHTOTL=CHTOT  -CHTOT1*(POTIM/2)
       CWORK=0
       CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
            INFO,P,T_INFO,E2,LATT_CUR, &
            CHTOTL,CSTRF, CWORK,DENCOR,SV2, SOFT_TO_C,XCSIF)

       CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
            LMDIM,CDIJ2,CQIJ,CWORK,IRDMAA,IRDMAX)

       CTMP=CRHODE-CRHODE1*(POTIM/2)
       RHOLM_LAST=RHOLM-RHOLM1*(POTIM/2)
       CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
            WDES%NCDIJ, LMDIM, CDIJ2(1,1,1,1), RHOLM_LAST , CTMP(1,1,1,1), &
            E2,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
!
! derivative of potential CVTOT1=(CVTOT1-CWORK)*(1/POTIM)
!                         SV1   =(SV1   -SV2  )*(1/POTIM)
       DO ISP=1,WDES%NCDIJ
          CALL RL_ADD(CVTOT1(1,ISP),1/POTIM,CWORK(1,ISP),-1/POTIM,CVTOT1(1,ISP),GRIDC)
          CALL RL_ADD(SV1(1,ISP)   ,1/POTIM,SV2(1,ISP)  ,-1/POTIM,SV1(1,ISP)   ,GRID)
       ENDDO

! derivative of CDIJ (due to change of the potential)
       CDIJ1 =(CDIJ1 -CDIJ2)*(1/POTIM)
       

! second order change of double counting corrections
! E1%DENC = - Tr [ rho(1) V_H rho(1) ]/2 
       E1%DENC =(E1%DENC + E2%DENC -2*E%DENC )*(1/POTIM)**2*2
       E1%XCENC=(E1%XCENC+ E2%XCENC-2*E%XCENC)*(1/POTIM)**2*2
       E1%PAWPS=(E1%PAWPS+ E2%PAWPS-2*E%PAWPS)*(1/POTIM)**2*2
       E1%PAWAE=(E1%PAWAE+ E2%PAWAE-2*E%PAWAE)*(1/POTIM)**2*2

       CALL STOP_TIMING("G",IO%IU6,"POT+DIJ")
!=======================================================================
!
!  calculate |xi> =(H(1) - e(1) S(0))| phi(0)>+ |xi(0)>+ \sum_i xi(0)_i |p_i> 
!  where H(1) is the first order change of the
!  e(1) is the first order change of the eigenenergy
!  |xi(0)>  = -i d/dk phi_nk> 
!  xi(0)_i  = -i d/dk <p_i |phi>= -i(<p_i | d/dk phi>+< d/dk p_i | phi>)
! 
!=======================================================================
       CALL KDER_WAVE (WXI, GRID, WDES, IDIR, CXI0, CXI0_CPROJ,  &
            T_INFO, P, LATT_CUR, KPOINTS_TRANS)

       ! rotate the polarization vector in the same manner as W1
       CALL SUBROT_DEG_ALL(WDES, WXI, .FALSE., .TRUE., DEG_CLUSTER )

       CALL LRF_HAMIL(GRID,INFO,LATT_CUR, &
            NONLR_S, NONL_S, W0, WXI, WDES, &
            LMDIM,CDIJ1, CQIJ, SV1, RMS(N), ICOUEV)

       CALL MRG_CEL(WDES,WXI)
       CALL MRG_FERWE(WDES,WXI)

       CALL STOP_TIMING("G",IO%IU6,"HAMIL1")
!
! the double counting corrections are 
!  - Tr [ rho(1)-rho_c(1) V_Hxc  rho(1)-rho_c(1) ]
! where rho(1) is the total 1st order change of the charge
! and rho_c(1) is the change of the charge density for fixed 
! wavefunction coefficients
! the d.c calculated above are however  Tr [ rho(1) V_Hxc  rho(1) ]
! adding Tr [  (rho(1) V_Hxc rho_c(1)) ] should work, but seems to be 
! too inaccurate
!       E1%DENC = E1%DENC+E1%XCENC
! test
!       CALL POTHAR(GRIDC,LATT_CUR, CHTOT1,CWORK,E2%DENC)
!       WRITE(*,*) E2%DENC ,E1%DENC,E2%DENC-E1%DENC
!       CALL POTXC2(GRIDC,LATT_CUR, CHTOT, DENCOR, CHTOT1,CWORK,E2%XCENC)
!       WRITE(*,*) E2%XCENC,E1%XCENC,E2%XCENC-E1%XCENC
! Hartree and LDA contribution
       CALL POTHAR(GRIDC,LATT_CUR, CHTOT1,CWORK,E1%DENC)
       IF (ISLDAXC()) THEN
          CALL POTXC2(GRIDC,LATT_CUR, CHTOT, DENCOR, CHTOT1,CWORK,E1%XCENC)
       ELSE
          CWORK=0
          E1%XCENC=0
       ENDIF
!=======================================================================
!
!  calculate the first order change of the wavefunction
!    H(0) - epsilon S(0) | phi_1> = - |xi>
!  the first order change of the energy is returned in WXI%CELEN
!  the first order change of the norm in WXI%FERWE
!
!=======================================================================
       CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER)
       CALL STOP_TIMING("G",IO%IU6,"LRDIAG")

       CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W1,WXI,W0,WDES, &
            LMDIM,CDIJ,CQIJ, RMS(N),DESUM,ICOUEV, SV, CSHIFT, IO%IU6,IO%IU0, LRESET=(N==1))

       CALL MRG_CEL(WDES,W1)
       CALL STOP_TIMING("G",IO%IU6,"LRDIIS")

       IF (N<=RESOLVE_DEG_NTIMES) THEN
          CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER,.TRUE.)
       ELSE
          CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER)
       ENDIF

       CALL STOP_TIMING("G",IO%IU6,"LRDIAG")
       
       ! sum f(0) (<phi(1)|xi> + c.c + <phi(1)| H(0)|phi(1)>)
       W1%FERTOT=W0%FERTOT  ! set the occupancy matrix W1 to W0
       E1%EBANDSTR= BANDSTRUCTURE_ENERGY(WDES, W1)
       ! derivative of occupancies is set to zero since the dielectric
       ! constant makes only sense for an insulator
       W1%FERTOT  =0
       E1%EENTROPY=0
       ! and set the first order change of eigenvalues (until this point stored in WXI)
       W1%CELTOT=   WXI%CELTOT

       TOTENL=TOTEN
       TOTEN =E1%EBANDSTR+E1%DENC+E1%XCENC+E1%PAWPS+E1%PAWAE+E1%EENTROPY+Ediel_sol
       DE= (TOTEN-TOTENL)

       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,200,ADVANCE='NO') N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
          WRITE(17,200,ADVANCE='NO')     N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
       ENDIF
 200   FORMAT('RMM: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)

!=======================================================================
! calculate first derivative of charge density
!=======================================================================
    chg:IF ( UPDATE_CHARGE .AND. N>=RESOLVE_DEG_NTIMES) THEN

       ! store CHTOTL for later use in the mixer
       CHTOTL=CHTOT1
       RHOLM_LAST=RHOLM1
       ! determine first order change of one center occupancy matrix 
       CALL DEPSUM1(W0, W1, WDES, LMDIM, CRHODE1, INFO%LOVERL)

       ! change storage convention to (total, magnetization)
       CALL US_FLIP(WDES, LMDIM, CRHODE1, INFO%LOVERL, .FALSE.)

       ! soft pseudo charge
       CALL SOFT_CHARGE1(GRID,GRID_SOFT,W0,W1,WDES, CHDEN1)
       ! change storage convention to (total, magnetization)
       CALL RC_FLIP(CHDEN1, GRID_SOFT, WDES%NCDIJ, .FALSE.)

       ! symmetrisation of soft pseudo charge
       IF (SYMM%ISYM ==2) THEN
          IF (WDES%LNONCOLLINEAR) THEN
             CALL RHOSYM(CHDEN1(1,1),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,1)
             IF (.NOT.WDES%LSPIRAL) &
                  CALL SYMFIELD(CHDEN1(1,2),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
          ELSE
             DO ISP=1,WDES%ISPIN
                CALL RHOSYM(CHDEN1(1,ISP),GRID_SOFT,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
             ENDDO
          ENDIF
       ENDIF

       CHTOT1=0
       ! add first order change of one center occupancy matrix times
       ! augmentation charge distribution rho(1)_ij Q_ij(r)
       CALL DEPLE_ADD(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
                 LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
                 LMDIM, CRHODE1, CHTOT1, CHDEN1, IRDMAX)

       ! set the one center occupancy matrix passed through mixer (RHOLM1)
       CALL SET_RHO_PAW(WDES, P, T_INFO, INFO%LOVERL, WDES%NCDIJ, LMDIM, &
            CRHODE1, RHOLM1)

       ! symmetrise total pseudo charge density CHTOT if required 
       IF (SYMM%ISYM ==1) THEN
          IF (WDES%LNONCOLLINEAR) THEN
             CALL RHOSYM(CHTOT1(1,1),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,1)
             IF (.NOT.WDES%LSPIRAL) &
                  &   CALL SYMFIELD(CHTOT1(1,2),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,WDES%SAXIS,LATT_CUR)
          ELSE
             DO ISP=1,WDES%ISPIN
                CALL RHOSYM(CHTOT1(1,ISP),GRIDC,SYMM%PTRANS,T_INFO%NIOND,SYMM%MAGROT,ISP)
             ENDDO
          ENDIF
       ENDIF

!=======================================================================
! mix charge density
!=======================================================================
       IF (MIX%IMIX/=0) THEN
          CALL START_TIMING("G")
          INFO%LMIX=.TRUE.

          CALL SET_RHO0(GRIDC, CHTOT1, 0.0_q)

          IF (MIX%IMIX==4) THEN
!  broyden mixing ... :
             CALL BRMIX(KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
                  (2*GRIDC%MPLWV),CHTOT1,CHTOTL,WDES%NCDIJ,LATT_CUR%B, &
                  LATT_CUR%OMEGA, N_MIX_PAW, RHOLM1, RHOLM_LAST, &
                  RMST(N),RMSC,RMSP,WEIGHT,.TRUE.,IERRBR)
             MIX%LRESET=.FALSE.
          ELSE
!  simple mixing ... :
             RMST(N)=0
             CALL MIX_SIMPLE(GRIDC,MIX,WDES%NCDIJ, CHTOT1,CHTOTL, &
                  N_MIX_PAW, RHOLM1, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST(N))
          ENDIF
          CALL STOP_TIMING("G",IO%IU6,"MIXING")
          DWRITE0 "mixing is ok"

          IF (IO%IU0>=0) THEN
             WRITE(IO%IU0,300) RMST(N)
             WRITE(17,300) RMST(N)
          ENDIF
300       FORMAT('   ',E10.3)
       ELSE
          IF (IO%IU0>=0) THEN
             WRITE(IO%IU0,*)
             WRITE(17,*)
          ENDIF
       ENDIF
    ELSE chg
       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,*)
          WRITE(17,*)
       ENDIF
    ENDIF chg
!=======================================================================
! total time used for this step
!=======================================================================
       CALL SEPERATOR_TIMING(IO%IU6)
       CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')

       NORDER=0 ; IF (KPOINTS%ISMEAR>=0) NORDER=KPOINTS%ISMEAR

!======================== end of loop ENDLSC ===========================
! end of the selfconsistent calculation loop
! and write commands
!=======================================================================
       INFO%LABORT=.FALSE.
       IF(ABS(DESUM)<INFO%EDIFF.AND.ABS(DE)<INFO%EDIFF) INFO%LABORT=.TRUE.
       ! rms stable for three steps, ok stop, no way to improve
       IF(N>=3) THEN
          IF (ABS((RMS(N)-RMS(N-1))/RMS(N))<1E-1 .AND. ABS((RMS(N)-RMS(N-2))/RMS(N))<1E-1) THEN
             ! if charge is updated require also the same for charge residuum
             IF (UPDATE_CHARGE .AND. N>=RESOLVE_DEG_NTIMES) THEN
                IF (ABS((RMST(N)-RMST(N-1))/RMST(N))<1E-1 .AND. ABS((RMST(N)-RMST(N-2))/RMST(N))<1E-1) THEN
                   INFO%LABORT=.TRUE.
                ENDIF
             ELSE
                INFO%LABORT=.TRUE.
             ENDIF
          ENDIF
       ENDIF
! finite difference for potential 1E-6 for charge convergence is the
! maximum attainable precision
       IF(UPDATE_CHARGE .AND. N>=RESOLVE_DEG_NTIMES .AND. ABS(RMST(N)) <1E-6)  INFO%LABORT=.TRUE.
       !-----do not stop before minimum number of iterations is reached
       IF (N < ABS(INFO%NELMIN)) INFO%LABORT=.FALSE.

       io_begin
    ! mixing related output
      IF ( INFO%LMIX .AND. MIX%IMIX==4 ) THEN
        WRITE(IO%IU6,2441) RMST(N),RMSC,RMSP,WEIGHT
        IF (MIX%NEIG > 0) THEN
           WRITE(IO%IU6,2442) MIX%AMEAN,MIX%EIGENVAL(1:MIX%NEIG)
        ENDIF
      ENDIF

 2441 FORMAT(/ &
     &       ' Broyden mixing:'/ &
     &       '  rms(total) =',E12.5,'    rms(broyden)=',E12.5,/ &
     &       '  rms(prec ) =',E12.5/ &
     &       '  weight for this iteration ',F10.2)

 2442 FORMAT(/' eigenvalues of (default mixing * dielectric matrix)' / &
             '  average eigenvalue GAMMA= ',F8.4,/ (10F8.4))


      WRITE(IO%IU6,210) E1%DENC,E1%XCENC,E1%PAWPS,E1%PAWAE, &
           E1%EENTROPY,E1%EBANDSTR,TOTEN,(TOTEN-E1%EENTROPY)

210   FORMAT(/ &
              ' Free energy of the ion-electron system (eV)'/ &
     &        '  ---------------------------------------------------'/ &
     &        '  -Hartree energ DENC   = ',F18.8/ &
     &        '  -V(xc)+E(xc)   XCENC  = ',F18.8/ &
     &        '  PAW double counting   = ',2F18.8/ &
     &        '  entropy T*S    EENTRO = ',F18.8/ &
     &        '  eigenvalues    EBANDS = ',F18.8/ &
     &        '  ---------------------------------------------------'/ &
     &        '  free energy    TOTEN  = ',F18.8,' eV'// &
     &        '  energy without entropy =',F18.8)


      io_end
!=======================================================================
!  xml related output
!=======================================================================
       CALL XML_TAG("energy")
       IF (INFO%LABORT .OR. N==1) THEN
          CALL XML_ENERGY(TOTEN, TOTEN-E1%EENTROPY, TOTEN-E1%EENTROPY/(2+NORDER))
       ELSE
          CALL XML_ENERGY(TOTEN, TOTEN-E1%EENTROPY, TOTEN-E1%EENTROPY/(2+NORDER))
       ENDIF
       CALL XML_CLOSE_TAG

       CALL XML_CLOSE_TAG("scstep")

       IF (INFO%LABORT) THEN
          IF (IO%IU6>=0) WRITE(IO%IU6,131)
131       FORMAT (5X, //, &
     &  '------------------------ aborting loop because EDIFF', &
     &  ' is reached ----------------------------------------'//)
          EXIT electron
       ENDIF
       INFO%LSOFT=.FALSE.

    ENDDO electron
!======================== end of loop ENDLSC ===========================
!

!=======================================================================
    DO I=1,3
       CALL KDER_WAVE (WXI, GRID, WDES, I, CXI0, CXI0_CPROJ,  &
            T_INFO, P, LATT_CUR, KPOINTS_TRANS)

       ! rotate the polarization vector in the same manner as W1
       CALL SUBROT_DEG_ALL(WDES, WXI, .FALSE., .TRUE., DEG_CLUSTER )
       WXI%FERTOT=W0%FERTOT
       CALL SUM_INPROD(WDES, WXI, W1, POL(I), .TRUE., LMDIM)
    ENDDO

    POL=-POL
! test
!   IF (SYMM%ISYM>0) CALL POLSYM(POL,LATT_CUR%A)
! test

    IF (IO%IU0>=0) THEN
       WRITE(IO%IU0,"(' change of polarisation eV/A/(eV/A) component ',I2,' :',3F10.3)") IDIR,POL
       WRITE(IO%IU6,"(' change of polarisation eV/A/(eV/A) component ',I2,' :',3F10.3)") IDIR,POL
    ENDIF

    ! in a.u 4 pi e^2/ volume
    POL=2*TPI*POL/(LATT_CUR%OMEGA)*FELECT
    POL(IDIR)=POL(IDIR)+1

    IF (IO%IU0>=0) THEN
       WRITE(IO%IU0,"(' dielectric tensor                  component ',I2,' :',3F10.3)") IDIR,POL
       WRITE(IO%IU6,"(' dielectric tensor                  component ',I2,' :',3F10.3)") IDIR,POL
    ENDIF
    EPSILON=POL
    IF (IO%IU6>=0) WRITE(IO%IU6,130)

!=======================================================================
!
! force and stress in order to obtain the Born effective charges
! and the (ion clamped) piezoelectric tensor 
! 
! The total energy functional for a finite field lambda is given by
!  E^DFT+lambda x \sum_nk < -i d/d_k phi_nk | phi_nk > +
!                 \sum_nk  \sum_j beta_j* <p_j | phi_nk > +   c.c.
!
! (beta_j is defined at the top of LRF_RPHI)
!
! The second derivative of E^DFT w.r.t. ions (and strain) and field
! is evaluated using the force and stress differences at
! (phi^0_nk + lambda phi^1_nk) and (phi^0_nk - lambda phi^1_nk)
! 
! The derivative of the second term is evaluated directly
!
!=======================================================================
    IF (LRPA) CALL POP_LEXCH

    IF (UPDATE_CHARGE) THEN
    ! this POTIM is a fairly good compromise
    ! Born effective charges and piezoe. tensor are correct to roughly 6 figures
    ! dynamic charges deteriorate with larger steps, whereas piezoe. tensor 
    ! deteriorates with smaller steps
    POTIM=0.01

    LCORR_STORE=INFO%LCORR
    INFO%LCORR =.FALSE.
    CHTOT2=CHTOT    +CHTOT1*(POTIM/2)
    CTMP=CRHODE     +CRHODE1*(POTIM/2)
    RHOLM_LAST=RHOLM+RHOLM1*(POTIM/2)

    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E1,LATT_CUR, &
         CHTOT2,CSTRF,CVTOT1,DENCOR,SV1, SOFT_TO_C,XCSIF)

    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)

    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1), RHOLM_LAST , CTMP(1,1,1,1), &
         E1,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

    WXI%CW    =W0%CW    +(POTIM/2)* W1%CW
    WXI%CPROJ =W0%CPROJ +(POTIM/2)* W1%CPROJ
    WXI%CELTOT=W0%CELTOT+(POTIM/2)* W1%CELTOT
    WXI%FERTOT=W0%FERTOT+(POTIM/2)* W1%FERTOT

    EWIFOR=0
    EWSIF=0
    CALL FORCE_AND_STRESS( &
         KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,WXI,LATT_CUR, &
         T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
         GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
         CHTOT2,CHTOTL,DENCOR,CVTOT1,CSTRF, &
         CDIJ1,CQIJ,CTMP,N_MIX_PAW,RHOLM_LAST,RHOLM_LAST, &
         CHDEN,SV1, &
         LMDIM, IRDMAX, .FALSE.,DYN%ISIF/=0,.FALSE.,.FALSE.,  &
         XCSIF, EWSIF, TSIF, EWIFOR, TIFOR, PRESS, TOTEN )
    F=TIFOR
    SIF=TSIF
!--------------------------------------------------------------------

    CHTOT2=CHTOT-CHTOT1*(POTIM/2)
    CTMP=CRHODE-CRHODE1*(POTIM/2)
    RHOLM_LAST=RHOLM-RHOLM1*(POTIM/2)

    CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
         INFO,P,T_INFO,E2,LATT_CUR, &
         CHTOT2,CSTRF,CVTOT1,DENCOR,SV1, SOFT_TO_C,XCSIF)

    CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ1,CQIJ,CVTOT1,IRDMAA,IRDMAX)

    CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ1(1,1,1,1), RHOLM_LAST , CTMP(1,1,1,1), &
         E2,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )

    WXI%CW    =W0%CW    -(POTIM/2)* W1%CW
    WXI%CPROJ =W0%CPROJ -(POTIM/2)* W1%CPROJ
    WXI%CELTOT=W0%CELTOT-(POTIM/2)* W1%CELTOT
    WXI%FERTOT=W0%FERTOT-(POTIM/2)* W1%FERTOT

    EWIFOR=0
    EWSIF=0
    CALL FORCE_AND_STRESS( &
         KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,WXI,LATT_CUR, &
         T_INFO,T_INFO,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
         GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
         CHTOT2,CHTOTL,DENCOR,CVTOT1,CSTRF, &
         CDIJ1,CQIJ,CTMP,N_MIX_PAW,RHOLM_LAST,RHOLM_LAST, &
         CHDEN,SV1, &
         LMDIM, IRDMAX, .FALSE.,DYN%ISIF/=0,.FALSE.,.FALSE.,  &
         XCSIF, EWSIF, TSIF, EWIFOR, TIFOR, PRESS, TOTEN )

    INFO%LCORR=LCORR_STORE

    F=(F-TIFOR)*(1/POTIM)
    SIF=(SIF-TSIF)*(1/POTIM)

    ! contribution from derivative of projectors with positions 
    ! \sum_j beta_j* <p_j | phi_nk >
    EWIFOR=0
    EWISIF=0

    IF (W0%WDES%LOVERL) THEN
       CALL KDER_WAVE (WXI, GRID, WDES, IDIR, CXI0, CXI0_CPROJ,  &
            T_INFO, P, LATT_CUR, KPOINTS_TRANS)

       CALL SUBROT_DEG_ALL(WDES, WXI, .FALSE., .TRUE., DEG_CLUSTER )

       IF (DYN%ISIF/=0) THEN
          NDIR=9
       ELSE
          NDIR=3
       ENDIF
       ALLOCATE(CPROJXYZ(W0%WDES%NPROD, W0%WDES%NBANDS, NDIR))

       DO ISP=1,W0%WDES%ISPIN
       DO NK=1,W0%WDES%NKPTS
#ifdef MPI
          IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
          ! first derivative of  wavefunction character with respect to all ionic positions
          ! factor two is accounted for in CPROJXYZ (bra or kat variation)
          IF (NONLR_S%LREAL) THEN             
             CALL PHASER(W0%WDES%GRID,LATT_CUR,NONLR_S,NK,W0%WDES)
             CALL RPROXYZ(W0%WDES%GRID, NONLR_S, P, LATT_CUR, W0, W0%WDES, ISP, NK, CPROJXYZ(1,1,1))
             IF (NDIR>3) &
                  CALL RPROLAT_DER(W0%WDES%GRID, NONLR_S, P, LATT_CUR, W0, W0%WDES, ISP, NK, CPROJXYZ(1,1,4))
          ELSE
             CALL PHASE(W0%WDES,NONL_S,NK)
             CALL PROJXYZ(NONL_S, W0%WDES, W0, LATT_CUR, ISP, NK, CPROJXYZ, .TRUE.)
             IF (NDIR>3) &
                  CALL PROJLAT_DER(P, NONL_S, W0%WDES, W0, LATT_CUR, ISP, NK, CPROJXYZ(1,1,4))
          ENDIF
          ! \sum_j beta_j* <p_j | phi_nk >
          DO I=1,NDIR
             DO N=1,WDES%NBANDS
                ENL=0
                DO ISPINOR=0,W0%WDES%NRSPINORS-1
                   LBASE =ISPINOR *W0%WDES%NPRO/2
                   
                   NIS=1
                   typ: DO NT=1,W0%WDES%NTYP
                      LMMAXC=W0%WDES%LMMAX(NT)
                      IF (LMMAXC==0) GOTO 510
                      ion: DO NI=NIS,W0%WDES%NITYP(NT)+NIS-1
#ifdef  ForceTest
                      DO L=1,LMMAXC
                      DO LP=1,LMMAXC
                      !test for derivative of wave function character with positions
                         ENL(NI)=ENL(NI)+W0%CPROJ(LBASE+LP,N,NK,ISP)*GCONJG(CPROJXYZ(LBASE+L,N,I))* &
                              (CDIJ(LP,L,NI,1)-W0%CELEN(N,NK,ISP)*CQIJ(LP,L,NI,1))
                      ENDDO
                      ENDDO
#else
                      DO L=1,LMMAXC
                         ENL(NI)=ENL(NI)+CPROJXYZ(LBASE+L,N,I)*GCONJG(WXI%CPROJ(LBASE+L,N,NK,ISP))
                      ENDDO
#endif
                      LBASE = LMMAXC+LBASE
                      ENDDO ion
510                   NIS = NIS+W0%WDES%NITYP(NT)
                   ENDDO typ
                ENDDO
                IF (I<=3) THEN
                   DO NI=1,W0%WDES%NIONS
                      NIP=NI_GLOBAL(NI, W0%WDES%COMM_INB)
                      EWIFOR(I,NIP)=EWIFOR(I,NIP)-ENL(NI)*WDES%RSPIN*WDES%WTKPT(NK)*W0%FERWE(N,NK,ISP)
                   ENDDO
                ELSE
                   DO NI=1,W0%WDES%NIONS
                      NIP=NI_GLOBAL(NI, W0%WDES%COMM_INB)
                      EWISIF(I-3)=EWISIF(I-3)+ENL(NI)*WDES%RSPIN*WDES%WTKPT(NK)*W0%FERWE(N,NK,ISP)
                   ENDDO
                ENDIF
             ENDDO
          ENDDO
       ENDDO
       ENDDO
       CALLMPI( M_sum_d(WDES%COMM, EWIFOR(1,1),T_INFO%NIONS*3))
       CALLMPI( M_sum_d(WDES%COMM, EWISIF(1),6))

       NI=0
       DO I=1,3
          DO J=1,I
             NI=NI+1
             EWSIF(I,J)=EWISIF(NI)
             EWSIF(J,I)=EWISIF(NI)
          ENDDO
       ENDDO

       IF (SYMM%ISYM>0) THEN
          CALL FORSYM(EWIFOR,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,  & 
               SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)
          CALL TSYM(EWSIF,ISYMOP,NROTK,LATT_CUR%A)
       ENDIF

       DEALLOCATE(CPROJXYZ)
       F  =F  +EWIFOR
       SIF=SIF+EWSIF 
       CALL STOP_TIMING("G",IO%IU6,"FORNLD")
    ENDIF
!   add rigid ion contribution
    F=-F
    DO NI=1,T_INFO%NIONS
       ENL(NI)=ONE_CENTRE_CHARGE(WDES, T_INFO, P, NI, LMDIM, CQIJ, CRHODE)
       F(IDIR,NI)=F(IDIR,NI)+P(T_INFO%ITYP(NI))%ZVALF*T_INFO%VCA(T_INFO%ITYP(NI))+ENL(NI)
    ENDDO

    io_begin
! write final piezoelectric tensor to OUTCAR
    IF (DYN%ISIF/=0) THEN
       FACT=EVTOJ*1E20_q/LATT_CUR%OMEGA
       WRITE(IO%IU6,160) IDIR, -SIF(:,1),-EWSIF(:,1),-SIF(:,2),-EWSIF(:,2),-SIF(:,3),-EWSIF(:,3)
       WRITE(IO%IU6,170) IDIR, -SIF(:,:)*FACT
       PIEZO(:,:)=-SIF
    ENDIF

160 FORMAT(/ ' PIEZOELECTRIC TENSOR FIELD DIRECTION ',I1,'  (e Angst)', / &
         &   ' -----------------------------------------------------------------------------',/ &
               (3X,3F9.5,10X," (",3F9.5,")"))

170 FORMAT(/ ' PIEZOELECTRIC TENSOR FIELD DIRECTION ',I1,'  (C/m^2)', / &
         &   ' -----------------------------------------------------------------------------',/ &
               (3X,3F9.5))

! write final Born effective charges to OUTCAR
    WRITE(IO%IU6,100) IDIR

    DO NI=1,T_INFO%NIONS
       VTMP=T_INFO%POSION(1:3,NI)
       CALL  DIRKAR(1,VTMP,LATT_CUR%A)
       WRITE(IO%IU6,110) VTMP,F(:,NI),ENL(NI),P(T_INFO%ITYP(NI))%ZVALF*T_INFO%VCA(T_INFO%ITYP(NI))
       BORN_CHARGES(:,NI)=F(:,NI)
    ENDDO
    VTMP=SUM(F,2)

    WRITE(IO%IU6,120) VTMP

100 FORMAT(// '      POSITION',8X,' DIRECTION ',I1,'           BORN EFFECTIVE CHARGE    (rigid.aug.   ionic)'/ &
              ' -----------------------------------------------------------------------------------------')
110 FORMAT((3F13.5,3X,3F9.5," (",2F9.5,")"))
120 FORMAT(   ' -----------------------------------------------------------------------------------------',/ &
              '    total drift (improves with k-points): ',3F9.5//)

    IF (IO%IU6>=0) WRITE(IO%IU6,130)
    IF (IO%LOPEN) CALL WFORCE(IO%IU6)
    IF (IO%LOPEN) CALL WFORCE(17)
    io_end

    ENDIF

    CALL SUBROT_DEG_ALL(WDES, W0, .TRUE., .TRUE., DEG_CLUSTER )

    CALL DEALLOCW(WXI)
    CALL DEALLOCW(W1)
      
    DEALLOCATE(CHTOT1,CHTOT2,CVTOT1,CWORK,CDIJ1,CDIJ2,CRHODE1,CTMP,SV1,SV2,CHDEN1)

    PROFILING_STOP('lrf_main')

  END SUBROUTINE LRF_MAIN


!*********************************************************************
!
! calculate the cell periodic part of -r | phi(0)_nk> using 
! - ( H(0)- S(0) e(0)_nk) (r | phi(0)_nk>) =
!                        ([H(0), r] - e(0) [S(0),r]) | phi(0)_nk>
! this is equivalent to determining -i d/dk | phi_nk>
! since [H(0),r] = - i d/dk H_k
!
! this is -i beta, where beta is defined in Equ. (26) of
! M. Gaidos, K. Hummer, G. Kresse, 
! Phys. Rev. B {\bf 73}, 045112 (2006).
!
! more specifically
! RPHI (W1%CW internally)  is set to
!
!   -i  d/dk phi_nk
!
! RPHI_CPROJ (W1%CPROJ  internally) is set to 
!
!   beta_j = \sum_i
!  -i Q_ji(<p_i | d/dk phi_nk>+< d/dk p_i | phi_nk>)-  tau_ji < p_i | phi_nk>
!
!*********************************************************************

  SUBROUTINE LRF_RPHI( &
          P,WDES,NONLR_S,NONL_S,W0,LATT_CUR, &
          T_INFO,INFO,IO,GRID,GRIDC,GRIDUS,C_TO_US,IRDMAX, &
          CDIJ,CQIJ,SV,LMDIM, DEG_CLUSTER, IDIR, RPHI, RPHI_CPROJ, LORTHO)

    USE base
    USE lattice
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE mpimy
    USE mgrid
    USE constant
    USE poscar
    USE wave
    USE pot
    USE ini
    USE paw
    USE hamil_lrf
    USE subrot_lr
    USE rmm_diis_lr
    USE us
    USE wave_high
    USE subrot_cluster
    USE hamil
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W0         ! unperturbed wavefunctions
    TYPE (latt)        LATT_CUR
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS      ! grid for potentials/charge
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    
    INTEGER LMDIM

!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  potential on soft grid
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
    TYPE (eigenf_cluster_pointer) :: DEG_CLUSTER(WDES%NKPTS,WDES%ISPIN)
    INTEGER :: IDIR                      ! direction
    COMPLEX(qs) :: RPHI(:,:,:,:)
    GDEFS       :: RPHI_CPROJ(:,:,:,:)
    LOGICAL     :: LORTHO
! local variables
#ifdef gammareal
    REAL(q) :: CSHIFT=0.000
#else
    REAL(q) :: CSHIFT=0.002              ! complex shift should avoid too large components 
                                         ! among eigenvalues that are close in energy
                                         ! however does not work for Gamma-only
                                         ! also seems to have very little effect on the final results
#endif
    REAL(q) :: TOTENL=0                  ! total energy in previous step
    REAL(q) :: TOTEN
    TYPE (nonlr_struct),TARGET :: NONLR_D! projector times r (only CRREXP distinct)
    TYPE (nonl_struct)  NONL_D           ! k derivative of projector in real space
    INTEGER :: IONODE, NODE_ME
    REAL(q) :: DESUM,DE                  ! change of energy
    REAL(q) :: RMS(INFO%NELM)            ! magnitude of residual vector
    INTEGER :: ICOUEV                    ! number of H | phi> evaluations
    INTEGER :: N                         ! main loop counter (electronic step)
    OVERLAP,ALLOCATABLE :: CDIJ1(:,:,:,:)! derivative of non local strength

    TYPE (wavespin)     :: WXI           ! stores H(1) - epsilon S(1) | phi_0>
    TYPE (wavespin)     :: W1            ! first order change of wavefunction
    TYPE (wavefun)      :: WTMP          ! temporary
    INTEGER I                            ! loop counter for various cases
    INTEGER IRDMAX
    INTEGER :: ierror
!=======================================================================
! the routine can either calculate and store 
!
!  xi =([H(0), r] - e(0) [S(0),r] ) | phi(0)>       (LI=.FALSE.)
!
! or
!
!  xi= (H(1)-e(0) S(1)) phi(0) = i([H(0), r] - e(0) [S(0),r]) | phi(0)>
!                                                   (LI=.TRUE.)
!
! in the former case  H(1)=[H(0), r] is antihermitian (antisymmetric
! but real at the Gamma point), 
! whereas in the latter H(1)=i [H(0), r] is Hermitian (only imaginary at 
! the Gamma point)
! presently the routine that resolves degeneracies can handle only
! the second case LI=.TRUE.
! that's why this is the preferred option but it does not work
! for the Gamma point only version
! i.e. the calls
! CALL EDDIAG_LR(............,.TRUE.) 
! CALL SUBROT_DEG_ALL(WDES, W0,  .TRUE., .TRUE., DEG_CLUSTER )
! CALL SUBROT_DEG_ALL(WDES, W1,  .TRUE., .TRUE., DEG_CLUSTER )
!=======================================================================
    LOGICAL             :: LI=.TRUE.

    PROFILING_START('lrf_rphi')

    IONODE=0
    NODE_ME=0
#ifdef MPI
    IONODE  = WDES%COMM%IONODE
    NODE_ME = WDES%COMM%NODE_ME
#endif
!=======================================================================
!  allocate all required work arrays
!=======================================================================
    CALL REINIT_DEG_CLUSTERS(WDES,DEG_CLUSTER)

    ALLOCATE(CDIJ1(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ))

    CALL ALLOCW(WDES,WXI,WTMP,WTMP)
    CALL ALLOCW(WDES,W1,WTMP,WTMP)

    W1%CW   =0
    W1%CPROJ=0

    IF (INFO%LREAL) THEN
!      NONLR_D=NONLR_S
       CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
       CALL NONLR_ALLOC_CRREXP(NONLR_D)
    ELSE
       CALL NONL_ALLOC_DER(NONL_S, NONL_D)
       CALL SPHER_DER(GRID,NONL_D,P,WDES,LATT_CUR,  IDIR)
    ENDIF
!=======================================================================
!  initialise required work arrays
!=======================================================================
    ! to make timing more sensefull syncronize now
    CALLMPI( MPI_barrier( WDES%COMM%MPI_COMM, ierror ))
    CALL START_TIMING("LOOP")

    CALL START_TIMING("G")
!=======================================================================
!  < psi_i | nabla | psi_j > - < tilde psi_i | nabla | tilde psi_j >
!=======================================================================
    CDIJ1=0
    IF (LNABLA) THEN
       CALL ADD_NABLA_ONE_CENTRE(WDES, T_INFO, P, LMDIM, IDIR, CDIJ1)
    ENDIF

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
    DWRITE0 'electron entered'

!     CALL DIPOL_RESET()
!=======================================================================
!  main selfconsistent loop for linear response
!=======================================================================
    TOTEN=0

electron: DO N=1,INFO%NELM

!=======================================================================
!  calculate |xi> = [H,r] | phi(0)>
!=======================================================================
       IF(IO%IU6>=0) WRITE(IO%IU6,140) IDIR,N
       CALL XML_TAG("scstep")

       IF (N<=RESOLVE_DEG_NTIMES+1) THEN
          CALL LRF_COMMUTATOR(GRID,INFO,LATT_CUR, &
               NONLR_S, NONLR_D, NONL_S, NONL_D, W0, WXI, WDES, &
               LMDIM,CDIJ1, CDIJ, CQIJ, LNABLA, LI, IDIR, CSHIFT, RMS(N), ICOUEV)
      
          CALL MRG_CEL(WDES,WXI)
          CALL STOP_TIMING("G",IO%IU6,"HAMIL1")
       ENDIF

       IF (N<=RESOLVE_DEG_NTIMES) THEN
          CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER,.TRUE.)
       ELSE
          CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER)
       ENDIF

       CALL STOP_TIMING("G",IO%IU6,"LRDIAG")
       CALL LINEAR_RESPONSE_DIIS(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W1,WXI,W0,WDES, &
            LMDIM,CDIJ,CQIJ, RMS(N),DESUM,ICOUEV, SV, CSHIFT, IO%IU6,IO%IU0, LRESET=(N==1)) 
       CALL MRG_CEL(WDES,W1)
       CALL STOP_TIMING("G",IO%IU6,"LRDIIS")

       ! sum f(0) (<phi(1)|xi> + c.c + <phi(1)| H(0)|phi(1)>)
       W1%FERTOT=W0%FERTOT  ! set the occupancy matrix W1 to W0
       TOTENL=TOTEN
       TOTEN =BANDSTRUCTURE_ENERGY(WDES, W1)
       DE= (TOTEN-TOTENL)

       IF (IO%IU0>=0) THEN
          WRITE(IO%IU0,200) N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
          WRITE(17,200)     N, TOTEN, DE, DESUM, ICOUEV, RMS(N)
       ENDIF
 200   FORMAT('RMM: ',I3,'   ',E20.12,'   ',E12.5,'   ',E12.5,I6,'  ',E10.3)
!=======================================================================
! total time used for this step
!=======================================================================
       CALL SEPERATOR_TIMING(IO%IU6)
       CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')

       INFO%LABORT=.FALSE.
       IF(ABS(DESUM)<INFO%EDIFF.AND.ABS(DE)<INFO%EDIFF) INFO%LABORT=.TRUE.
       IF(N>=3) THEN
          IF (ABS((RMS(N)-RMS(N-1))/RMS(N))<1E-1 .AND. ABS((RMS(N)-RMS(N-2))/RMS(N))<1E-1) INFO%LABORT=.TRUE.
       ENDIF


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
          EXIT electron
       ENDIF
       INFO%LSOFT=.FALSE.

    ENDDO electron
!======================== end of loop ENDLSC ===========================
!=======================================================================

! add the derivative of the projector 
! <p_i | d/dk phi_nk> + < d/dk p_i | phi_nk>
! to obtain the change of the wavefunction character with respect to ik
    IF (.NOT.LNABLA) THEN
       W1%CPROJ =W1%CPROJ+ WXI%CPROJ
    ENDIF
! remove unitary part from i([H(0), r] - e(0) [S(0),r]) phi
! any unitary rotation in the subspace spanned by the occupied orbitals
! leaves the density and the energy unchanged
    IF (LORTHO) THEN
       CALL ORTHO_LR(W1,W0,WDES,LMDIM,CQIJ,INFO%LOVERL,IO%IU0,LHERM=.NOT.LI)
    ENDIF
    IF (LI) THEN
       W1%CW=W1%CW *(0.0_q,-1.0_q)
       W1%CPROJ=W1%CPROJ *(0.0_q,-1.0_q)
       WXI%CPROJ=WXI%CPROJ *(0.0_q,-1.0_q)
    ENDIF

! W1%CW contains now -i <G | d/dk phi_nk> =  <G | d/d ik phi_nk> 
!                    ( <G | r phi_nk>  for a finite system)
! set W1%CPROJ to 
! \sum_i -i Q_ji (<p_i | d/dk phi_nk> + < d/dk p_i | phi_nk>) 
    CALL  OVERL_ALL(WDES, W1, W1, .TRUE., LMDIM, CQIJ)

    IF (.NOT. LNABLA) THEN
! add - tau_ji < p_i | phi_nk>
! where tau is the dipole moment of the augmentation charges
       CALL SETDIJ_R(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
     &      INFO%LOVERL,LMDIM,CDIJ1,IRDMAX,IDIR)
       CALL  OVERL_ALL(WDES, W0, W1, .FALSE. , LMDIM, CDIJ1)
! test: if ilinear_response.F has already set up RPHI (pead), we may inspect it here
!       W1%CW=RPHI
!       W1%CPROJ=RPHI_CPROJ
!       CALL ORTHO_LR_TEST(W1,W0,WDES,LMDIM,CQIJ,INFO%LOVERL,IO%IU0,LHERM=.TRUE.)
! test end
    ENDIF
!
! back rotation of the wavefunctions
! to resolve degeneracies the wavefunctions might have been rotated
    CALL SUBROT_DEG_ALL(WDES, W0,  .TRUE., .TRUE., DEG_CLUSTER )
    CALL SUBROT_DEG_ALL(WDES, W1,  .TRUE., .TRUE., DEG_CLUSTER )

! store the result in RPHI and RPHI_CPROJ
    CALL KPAR_SYNC_ALL(WDES,W1)
    RPHI=W1%CW
    RPHI_CPROJ=W1%CPROJ

    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    CALL DEALLOCW(WXI)
    CALL DEALLOCW(W1)

    IF (INFO%LREAL) THEN
       CALL  NONLR_DEALLOC_CRREXP(NONLR_D)
    ELSE
       CALL  NONL_DEALLOC_DER(NONL_D)
    ENDIF

    DEALLOCATE(CDIJ1)

    PROFILING_STOP('lrf_rphi')

  END SUBROUTINE LRF_RPHI


!*********************************************************************
!
! calculate the cell periodic part of -r | phi(0)_nk> in the
! subspace spanned by all calculated orbitals using
! -( H(0)- S(0) e(0)_nk) r | phi(0)_nk> =
!                         ([H(0), r] - e(0) [S(0),r]) | phi(0)_nk>
! this is equivalent to determining -i d/dk | phi_nk>
! i.e. the change of the wavefunction with respect to ik
! this is -i beta, beta defined in Equ. (26)
! M. Gaidos, K. Hummer, G. Kresse, "Linear optical properties in the PAW"
! 
! since the response is only calculated in the subspace of the
! known orbitals this can be  done by multiplication from the
! left with  <phi(0)_n'k|
!
! -r|phi(0)_nk> (approximately)
!       |phi(0)_n'k><phi(0)_n'k| ([H(0), r]-e(0)[S(0),r]) |phi(0)_nk>                  
! sum_n'------------------------------------------------------------ 
!                                  e(0)_nk- e(0)_n'k
!
! the first order change of the eigenvalues with respect to k is 
! returned in W1%CELTOT
!
!*********************************************************************

  SUBROUTINE LRF_RPHI0( &
          P,NONLR_S,NONL_S,W0,LATT_CUR, &
          T_INFO,INFO,IO,GRID,GRIDC,GRIDUS,C_TO_US,IRDMAX, &
          CDIJ,CQIJ,SV,LMDIM,DEG_CLUSTER, IDIR, W1, LADD)

    USE base
    USE lattice
    USE pseudo
    USE lattice
    
    USE nonl_high
    USE mpimy
    USE mgrid
    USE constant
    USE poscar
    USE wave
    USE pot
    USE ini
    USE paw
    USE hamil_lrf
    USE subrot_lr
    USE rmm_diis_lr
    USE us
    USE wave_high
    USE subrot_cluster
    IMPLICIT NONE
!  structures
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W0         ! unperturbed wavefunctions
    TYPE (latt)        LATT_CUR
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS      ! grid for potentials/charge
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    
    INTEGER LMDIM

!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ)
!  potential on soft grid
    RGRID       SV(DIMREAL(GRID%MPLWV),W0%WDES%NCDIJ)
    TYPE (eigenf_cluster_pointer) :: DEG_CLUSTER(W0%WDES%NKPTS,W0%WDES%ISPIN)
    INTEGER :: IDIR                      ! direction
    LOGICAL, OPTIONAL :: LADD            ! W1 contains the Fock contribution
! local variables
    REAL(q) :: CSHIFT=0.00
    TYPE (nonlr_struct) NONLR_D          ! projector times r (only CRREXP distinct)
    TYPE (nonl_struct)  NONL_D           ! k derivative of projector in real space
    INTEGER :: IONODE, NODE_ME
    REAL(q) :: DESUM,DE                  ! change of energy
    REAL(q) :: RMS                       ! magnitude of residual vector
    INTEGER :: ICOUEV                    ! number of H | phi> evaluations
    INTEGER :: N                         ! main loop counter (electronic step)
    OVERLAP,ALLOCATABLE :: CDIJ1(:,:,:,:)! derivative of non local strength

    TYPE (wavespin)     :: WXI           ! stores H(1) - epsilon S(1) | phi_0>
    TYPE (wavespin)     :: W1            ! first order change of wavefunction
    TYPE (wavefun)      :: WTMP          ! temporary
    INTEGER I                            ! loop counter for various cases
    REAL(q)             :: SCALE=1.0_q
    INTEGER IRDMAX
    INTEGER :: ierror
!=======================================================================
! the routine can either calculate and store  (LI=.FALSE.)
!  xi =([H(0), r] - e(0) [S(0),r] ) | phi(0)>
! (xi -> WXI) or  (LI=.TRUE.)
!  xi= (H(1)-e(0) S(1)) phi(0) = i([H(0), r] - e(0) [S(0),r]) | phi(0)>
! in the former case  H(1)=[H(0), r] is antihermitian, whereas 
! in the latter H(1)=i [H(0), r] is hermitian
! this routine does not need to deal with degeneracies, hence
! LI can be set to .FALSE.
!=======================================================================
    LOGICAL             :: LI=.FALSE.

    PROFILING_START('lrf_rphi0')

    IONODE=0
    NODE_ME=0
#ifdef MPI
    IONODE  = W0%WDES%COMM%IONODE
    NODE_ME = W0%WDES%COMM%NODE_ME
#endif
!=======================================================================
!  allocate all required work arrays
!=======================================================================
    CALL REINIT_DEG_CLUSTERS(W0%WDES, DEG_CLUSTER)

    ALLOCATE(CDIJ1(LMDIM,LMDIM,W0%WDES%NIONS,W0%WDES%NCDIJ))

    CALL ALLOCW(W1%WDES,WXI,WTMP,WTMP)

    IF (INFO%LREAL) THEN
!      NONLR_D=NONLR_S
       CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
       CALL NONLR_ALLOC_CRREXP(NONLR_D)
    ELSE
       CALL NONL_ALLOC_DER(NONL_S, NONL_D)
       CALL SPHER_DER(GRID,NONL_D,P,W0%WDES,LATT_CUR,  IDIR)
    ENDIF
!=======================================================================
!  initialise required work arrays
!=======================================================================
    ! to make timing more sensefull syncronize now
    CALLMPI( MPI_barrier( W0%WDES%COMM%MPI_COMM, ierror ))
    CALL START_TIMING("LOOP")

    CALL START_TIMING("G")
!=======================================================================
!  < psi_i | nabla | psi_j > - < tilde psi_i | nabla | tilde psi_j >
!=======================================================================
    CDIJ1=0
    IF (LNABLA) THEN
       CALL ADD_NABLA_ONE_CENTRE(W0%WDES, T_INFO, P, LMDIM, IDIR, CDIJ1)
    ENDIF
!=======================================================================
!  calculate |xi> = [H,r] | phi(0)>
!  WXI%CELEN stores the first order energy change
!=======================================================================
    IF (LADD) THEN
       WXI%CW=W1%CW
    ENDIF

    CALL LRF_COMMUTATOR(GRID, INFO, LATT_CUR, &
         NONLR_S, NONLR_D, NONL_S, NONL_D, W0, WXI, W0%WDES, &
         LMDIM, CDIJ1, CDIJ, CQIJ, LNABLA, LI, IDIR, CSHIFT, RMS, ICOUEV, LADD)
    
    CALL MRG_CEL(WXI%WDES,WXI)

    W1%CW   =0
    W1%CPROJ=0
!=======================================================================
! calculate
!               |phi(0)_n'k><phi(0)_n'k| xi_n> 
!  W1=  sum_n' ------------------------------  
!                  e(0)_nk - e(0)_n'k                          
!=======================================================================

!test resolve deg
!    CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER,.TRUE.)

!    CALL LRF_COMMUTATOR(GRID, INFO, LATT_CUR, &
!         NONLR_S, NONLR_D, NONL_S, NONL_D, W0, WXI, W0%WDES, &
!         LMDIM, CDIJ1, CDIJ, CQIJ, LNABLA, LI, IDIR, CSHIFT, RMS, ICOUEV, LADD)
   
!    CALL MRG_CEL(WXI%WDES,WXI)

!    W1%CW   =0
!    W1%CPROJ=0

!test resolve deg end
      CALL EDDIAG_LR(W1,W0,WXI,LMDIM,CQIJ,INFO%LOVERL,.FALSE.,CSHIFT,IO%IU0,DEG_CLUSTER)

    IF (LI) THEN
       W1%CW=W1%CW *(0.0_q,-1.0_q)
       W1%CPROJ=W1%CPROJ *(0.0_q,-1.0_q)
       WXI%CPROJ=WXI%CPROJ *(0.0_q,-1.0_q)
    ENDIF

! W1%CW contains now -i <G | d/dk phi_nk> =  <G | d/d ik phi_nk> 
!                    ( <G | - r phi_nk>  for a finite system)
! this is the total change of the wavefunction with respect to ik
    IF (LNABLA) THEN
       CALL  OVERL_ALL(W1%WDES, W1, W1, .TRUE.  , LMDIM, CQIJ)
    ELSE
!=======================================================================
! W1%CPROJ now contains  <p_j | d/d ik phi_nk> 
! set W1%CPROJ to 
! sum_ij Q_ij -i(<p_j | d/dk phi_nk> + < d/dk p_j | phi_nk>) 
! - sum_ij  tau_ij < p_j | phi_nk>
! where tau is the dipole moment of the augmentation charges
! this corresponds the change of the wavefunction character with respect to ik
! plus a dipole term stemming from the augmentation charges
!=======================================================================
       W1%CPROJ = W1%CPROJ+WXI%CPROJ
       CALL  OVERL_ALL(W1%WDES, W1, W1, .TRUE.  , LMDIM, CQIJ)
       CALL SETDIJ_R(W1%WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO, &
            INFO%LOVERL,LMDIM,CDIJ1,IRDMAX,IDIR)
       CALL OVERL_ALL(W1%WDES, W0, W1, .FALSE. , LMDIM, CDIJ1)
    ENDIF

! back rotation of the wavefunctions
! to resolve degeneracies the wavefunctions might have been rotated
! presently this does not apply since EDDIAG_LR is not resolving degeneracies
!test resolve deg
!   CALL SUBROT_DEG_ALL(W0%WDES, W0, .TRUE., .TRUE., DEG_CLUSTER )
!   CALL SUBROT_DEG_ALL(W1%WDES, W1, .TRUE., .TRUE., DEG_CLUSTER )
!test resolve deg

! store the first order change of the eigenvalues in W1
    IF (LI) THEN
       W1%CELTOT=WXI%CELTOT
    ELSE
       W1%CELTOT=WXI%CELTOT*(0.0_q,1.0_q)
    ENDIF

    CALL DEALLOCW(WXI)

    IF (INFO%LREAL) THEN
       CALL  NONLR_DEALLOC_CRREXP(NONLR_D)
    ELSE
       CALL  NONL_DEALLOC_DER(NONL_D)
    ENDIF

    DEALLOCATE(CDIJ1)

    PROFILING_STOP('lrf_rphi0')

  END SUBROUTINE LRF_RPHI0


!*************************** LR_READER ********************************
!                                           
! this subroutine reads the INCAR file to determine parameters for
! the linear response routine

!**********************************************************************
  
    SUBROUTINE LR_READER(EDIFF,IU0,IU5,IU6)
      
      USE vaspxml
      USE base
      IMPLICIT NONE 
      REAL(q) EDIFF
      INTEGER IU5   ! input device (usually INCAR)
      INTEGER IU0   ! stderr
      INTEGER IU6   ! stdout
   ! local
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER(1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
!
! direction of dipole correction IDIPOL=0 (off) or 1-3 (1.d slab) or 4 (all direction)
!
      LEPSILON=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LEPSILON','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LEPSILON,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LEPSILON'' from file INCAR.'
         LEPSILON=.FALSE.
         RETURN
      ENDIF
      CALL XML_INCAR('LEPSILON','L',IDUM,RDUM,CDUM,LEPSILON,CHARAC,N)
!
! reread ediff using tighter default
!
      IF (LEPSILON) EDIFF=1E-6
      CALL RDATAB(LOPEN,INCAR,IU5,'EDIFF','=','#',';','F', &
     &            IDUM,EDIFF,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''EDIFF'' from file INCAR.'
      ENDIF
      EDIFF=MAX(ABS(EDIFF),1.E-12_q)
!
! DEGENERACY_THRESHOLD
!
      
      CALL RDATAB(LOPEN,INCAR,IU5,'DEG_THRESHOLD','=','#',';','F', &
     &            IDUM,DEGENERACY_THRESHOLD,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''DEG_THRESHOLD'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('DEG_THRESHOLD','F',IDUM,DEGENERACY_THRESHOLD,CDUM,LDUM,CHARAC,N)
!
! direction of dipole correction IDIPOL=0 (off) or 1-3 (1.d slab) or 4 (all direction)
!
      LRPA=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LRPA','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LRPA,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LRPA'' from file INCAR.'
         LRPA=.FALSE.
      ENDIF
      CALL XML_INCAR('LRPA','L',IDUM,RDUM,CDUM,LRPA,CHARAC,N)
!
! nabla operator
!
      LNABLA=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LNABLA','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LNABLA,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LNABLA'' from file INCAR.'
         LNABLA=.FALSE.
      ENDIF
      CALL XML_INCAR('LNABLA','L',IDUM,RDUM,CDUM,LNABLA,CHARAC,N)

!
! interpolation in k using linear response routines
!
      LINTERFAST=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LINTERFAST','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LINTERFAST,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LINTERFAST'' from file INCAR.'
         LINTERFAST=.TRUE.
      ENDIF
      CALL XML_INCAR('LINTERFAST','L',IDUM,RDUM,CDUM,LINTERFAST,CHARAC,N)

!
! additional finer grid
!
      KINTER=0
      CALL RDATAB(LOPEN,INCAR,IU5,'KINTER','=','#',';','I', &
     &            KINTER,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''KINTER'' from file INCAR.'
         KINTER=0
      ENDIF
      CALL XML_INCAR('KINTER','I',KINTER,RDUM,CDUM,LDUM,CHARAC,N)

!
! velocity operator
!
      LVEL=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LVEL','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LVEL,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LVEL'' from file INCAR.'
         LVEL=.FALSE.
      ENDIF
      CALL XML_INCAR('LVEL','L',IDUM,RDUM,CDUM,LVEL,CHARAC,N)

!
! CSHIFT from INCAR
!
      LCSHIFT=0.1_q
      CALL RDATAB(LOPEN,INCAR,IU5,'CSHIFT','=','#',';','F', &
      &            IDUM,LCSHIFT,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
           &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''CSHIFT'' from file INCAR.'
         LCSHIFT=0.1_q
      ENDIF
      CALL XML_INCAR('CSHIFT','F',IDUM,LCSHIFT,CDUM,LDUM,CHARAC,N)

!
! OMEGAMAX from INCAR
!
      OMEGAMAX_OPTIC=-1
      CALL RDATAB(LOPEN,INCAR,IU5,'OMEGAMAX','=','#',';','F', &
      &            IDUM,OMEGAMAX_OPTIC,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<1))) THEN
      IF (IU0>=0) &
      WRITE(IU0,*)'Error reading item ''OMEGAMAX'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('OMEGAMAX','F',IDUM,OMEGAMAX_OPTIC,CDUM,LDUM,CHARAC,N)
      
!
! relaxation time RTIME from INCAR
!
      RTIME=-1.E-1_q
      CALL RDATAB(LOPEN,INCAR,IU5,'RTIME','=','#',';','F', &
      &            IDUM,RTIME,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<1))) THEN
      IF (IU0>=0) &
      WRITE(IU0,*)'Error reading item ''RTIME'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('RTIME','F',IDUM,RTIME,CDUM,LDUM,CHARAC,N)
!
! imaginary part of plasma frequency WPLASMAI in eV
! this can be used as alternative input for RTIME
!
      WPLASMAI=0
      IF (RTIME>0) THEN
         WPLASMAI=0.65821220/RTIME
      ENDIF
      CALL RDATAB(LOPEN,INCAR,IU5,'WPLASMAI','=','#',';','F', &
      &            IDUM,WPLASMAI,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<1))) THEN
      IF (IU0>=0) &
      WRITE(IU0,*)'Error reading item ''WPLASMAI'' from file INCAR.'
      ENDIF
      CALL XML_INCAR('WPLASMAI','F',IDUM,WPLASMAI,CDUM,LDUM,CHARAC,N)
      IF (WPLASMAI>0) THEN
         ! 1 eV-1 = 0.6582122 fs
         RTIME=0.65821220/WPLASMAI
      ENDIF
!
! read in plasma frequency, this can be used to set the Drude term
!
      WPLASMA=0
      CALL RDATAB(LOPEN,INCAR,IU5,'WPLASMA','=','#',';','F', &
      &            IDUM,WPLASMA,CDUM,LDUM,CHARAC,N,9,IERR)
      IF (IERR==0 .AND. N==1) THEN
         WPLASMA(2,2)=WPLASMA(1,1)
         WPLASMA(3,3)=WPLASMA(1,1)
      ELSE IF (((IERR/=0).AND.(IERR/=3)).OR. &
      &                    ((IERR==0).AND.(N<9))) THEN
         IF (IU0>=0) &
              WRITE(IU0,*)'Error reading item ''WPLASMA'' from file INCAR.'
         WPLASMA=0
      ENDIF
      CALL XML_INCAR_V('WPLASMA','F',IDUM,WPLASMA,CDUM,LDUM,CHARAC,N)

!
! delta impulse in time
!
      DFIELD=0
      CALL RDATAB(LOPEN,INCAR,IU5,'DFIELD','=','#',';','F', &
           &   IDUM,DFIELD,CDUM,LDUM,CHARAC,N,3,IERR)

      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=3))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''DFIELD'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
      ENDIF
      CALL XML_INCAR_V('DFIELD','F',IDUM,DFIELD,CDUM,LDUM,CHARAC,N)

      CLOSE(IU5)
      
    END SUBROUTINE LR_READER


!**************************** LR_WRITER *******************************
!
! write the parameters of the LR routine to the OUTCAR file
!
!**********************************************************************

    SUBROUTINE LR_WRITER(IU6)

      IMPLICIT NONE
      INTEGER IU6               ! output unit

      IF (IU6>=0) THEN
         WRITE(IU6,100) LEPSILON, LRPA, LNABLA, LVEL, LINTERFAST, KINTER, LCSHIFT, OMEGAMAX_OPTIC, DEGENERACY_THRESHOLD, RTIME, WPLASMAI, DFIELD
         IF (SUM(ABS(WPLASMA))>0) THEN
            WRITE(IU6,110) WPLASMA
         ELSE
            WRITE(IU6,*)
         ENDIF
      ENDIF

100   FORMAT( &
             ' Linear response parameters'/  &
             '   LEPSILON=',L6,  '    determine dielectric tensor' / &
             '   LRPA    =',L6,  '    only Hartree local field effects (RPA)' / &
             '   LNABLA  =',L6,  '    use nabla operator in PAW spheres' / &
             '   LVEL    =',L6,  '    velocity operator in full k-point grid' / &
             '   LINTERFAST=',L4,  '  fast interpolation' / &
             '   KINTER  =',I6,  '    interpolate to denser k-point grid' / &
             '   CSHIFT  =',F6.4,'    complex shift for real part using Kramers Kronig' / & 
             '   OMEGAMAX=',F6.1,'    maximum frequency' / &
             '   DEG_THRESHOLD=',E14.7,' threshold for treating states as degnerate' / &
             '   RTIME   =',F9.3,   ' relaxation time in fs' / &
             '  (WPLASMAI=',F9.3,  ' imaginary part of plasma frequency in eV, 0.658/RTIME)' / &
             '   DFIELD  =',3F10.7,' field for delta impulse in time')
110   FORMAT('   WPLASMA =',9F9.3/)

      
       
    END SUBROUTINE LR_WRITER

!**************************** XML_WRITE_LR ****************************
!
! write the parameters of the LR routine to the XML file   
!
!**********************************************************************

    SUBROUTINE XML_WRITE_LR
      USE pseudo
      USE vaspxml
      IMPLICIT NONE

      LOGICAL :: LDUM
      INTEGER :: IDUM
      REAL(q) :: RDUM
      COMPLEX(q)  :: CDUM
      CHARACTER(1) :: CHARAC

      CALL XML_TAG("separator","linear response parameters")
      CALL XML_INCAR('LEPSILON','L',IDUM,RDUM,CDUM,LEPSILON,CHARAC,1)
      CALL XML_INCAR('LRPA','L',IDUM,RDUM,CDUM,LRPA,CHARAC,1)
      CALL XML_INCAR('LNABLA','L',IDUM,RDUM,CDUM,LNABLA,CHARAC,1)
      CALL XML_INCAR('LVEL','L',IDUM,RDUM,CDUM,LVEL,CHARAC,1)
      CALL XML_INCAR('KINTER','I',KINTER,RDUM,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('CSHIFT','F',IDUM,LCSHIFT,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('OMEGAMAX','F',IDUM,OMEGAMAX_OPTIC,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('DEG_THRESHOLD','F',IDUM,DEGENERACY_THRESHOLD,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('RTIME','F',IDUM,RTIME,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('WPLASMAI','F',IDUM,WPLASMAI,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('DFIELD','F',IDUM,DFIELD,CDUM,LDUM,CHARAC,3)
      CALL XML_INCAR_V('WPLASMA','F',IDUM,WPLASMA,CDUM,LDUM,CHARAC,9)
      CALL XML_CLOSE_TAG
    END SUBROUTINE XML_WRITE_LR


END MODULE mlrf_main
