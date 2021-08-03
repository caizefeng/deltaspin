#include "symbol.inc"

MODULE mlr_optic
  USE prec
  USE vaspxml
  USE lr_helper
  USE dfast
  IMPLICIT NONE

!*********************************************************************
!
!  CDER_BETWEEN_STATES(m,n,1:NK,1:ISP,j)= 
!          <u_m|  -i d/dk_j u_n> = - <u_m| r_j |u_n>
!  is the derivative of the orbital n with respect to k
!  projected onto the orbital m
!  this is an Hermitian matrix (quite obvious for  - <u_m| r_j |u_n>)
!  for the derivative operator this is not immedeatly obvious:
!  one minus comes from i, the second one from partial differentiation
!  (<u_n|  -i d/dk_j u_m>)* = <u*_n|  i d/dk_j u*_m> =  (partial int.)
!          <u_m|  i d/dk_j u_n>
!
!*********************************************************************

  INTEGER :: NBANDS_CDER        ! number of bands n considered 
                                ! m always runs from 1 to NB_TOT

!
! stores the optical transition elements   <u_m|- i d/dk_j | u_n> = - <u_m| r_j |u_n>
!
  GDEFS, ALLOCATABLE,   SAVE :: CDER_BETWEEN_STATES(:,:,:,:,:)
!
! optical transition elements between two k-points (at finite q)
!          <u_m,k+q| e^(i q r) | u_n,k>
! u_n is restricted to occupied states, whereas m is 
! in principle allowed to run over all states
! 
!
  COMPLEX, ALLOCATABLE,   SAVE :: CDER_BETWEEN_STATES_Q(:,:,:,:)

!
! velocity operator or derivative of eigenvalue with respect to k
!
  REAL(q), ALLOCATABLE, SAVE :: ENERGY_DER(:,:,:,:)

!
! position of innermost node in dielectric function
!
  REAL(q), SAVE :: NODES_IN_DIELECTRIC_FUNCTION=-1


  REAL(q), SAVE :: WPLASMON(3,3)

  REAL(q), PARAMETER :: CONTCON = 0.0204370_q ! e0(SI)/hq^2(eV^2)*10^-15*10^-6 = 8.85x10^-12/(6.58x10^-16)^2*10^-15*10^-6 (tau in fs, sigma in Mega Siemens/ meter)
  !k-downsampling array
  INTEGER, ALLOCATABLE,SAVE :: IKPT_SELECT(:)

  PRIVATE :: F,G
CONTAINS

!*********************************************************************
!
! calculate the macroscopic frequency dependent dielectric function
!
!  epsilon_q->0(G=0,G'=0,w) 
!
! in the random phase approximation (independent particle approximation)
! the name optical conductivity is maybe more appropriate
! 
! epsilon is calculated using a sum over unoccupied bands
! in the RPA approximation, that is the macroscopic tensor is 
! approximated by
!
! epsilon_q->0(G=0,G'=0,w) = 1 + 4 pi e^2/q^2 P(G=0,G'=0,w)
! 
! where P is the head of the irreducable polarizability
! respectively the macroscopic polarizability
! written by gK
!
! optionally the matrix of the derivative of the cell periodic part
! of the wavefunctions with respect to i k is set up and stored
!  CDER_BETWEEN_STATES(m,n,1:NK,1:ISP,j)= 
!          <u_m|- i d/dk_j | u_n> = - <u_m| r_j |u_n>
! this quantity is real in the case of gamma point only calculations
!
!
! More details_
! there are two equivalent ways to calculate the dielectric function
! (1) Either using the density-density response function
!     the matrix elements are in this case given by <u_m|  -i d/dk_j u_n> 
!     stored globally in the array CDER_BETWEEN_STATES 
!     (locally in many subroutines temporarily stored in CHAM)
!     in the function EPSILON_IMAG  EPSDD(w) is set to
!           (f_n-f_m) delta(w-(e_m-e_n)) (e 4 pi^2 e^2/volume) <u_m| -i d/dk_j u_n> <u_m|  -i d/dk_j u_n>^*
!     then a Kramers-Kronig transformation is performed using the
!     function EPSILON_REAL
!     the dielectric function is then finally calculated as 
!      1 + EPS_REAL(w) + i 2/pi
!            1/w e 4 pi^2 e^2/volume Imag(<u_m|  -i d/dk_j u_n> <u_m|  -i d/dk_j u_n>^*) (f_n-f_m)
!     the later term (summed over m and n) is stored in BIMAG
!     this term is related to the anomalous Hall contribution (AHC)
! (2) instead the velocity-velocity response function can be used 
!     the velocity operator can be calculated as
!          i v_mn =  <u_m|  [H,r_j] | u_n> = <u_m| r_j | u_n> (e_m-e_n) 
!                 =  CHAM(m,n) * (e_m-e_m)
!     from this the response function EPSJJ is calculated as 
!           (f_n-f_m) delta(w-(e_m-e_n)) (e 4 pi^2 e^2/volume) v_mn  v^*_mn
!     then a Kramers-Kronig transformation is performed using EPSILON_REAL
!     and the dielectric function is then given by 1/w^2 (1 + EPSJJ(w)-EPSJJ(0))
!     the equivalence is quite simple to prove by using a many-pole model
!     for the spectral function
!
! for the Hall term see in particular:
!     Sipe and Ghahramani, Phys. Rev. B 48, 11705 (1993).
!     Kunes, Physica Scripta. Vol. T109, 116 (2004).
!     Sangalli, Marini, Debernardi, PRB 86, 12539 (2012).
!
! sign convention: signs are on first sight different from e.g. Marini, however, this
! is related to the adopted convention of the Kramsers Kronig transform which
! is in it's "standard" form
!
!  2/pi int_0^infty w' eps_imag(w') / (w'^2-w^2) d w' =
!  2/pi (-1/2) int_0^infty (eps_imag(w')/(w-w') - eps_imag^*(w')/(w+w'))
!
! see function EPSILON_REAL. This implies that EPSDD has a different
! sign than in Marini paper. Also sign from AHC differs.
! The routine usually calculates the causal or retarded polarizability,
! however, 
!#define timeordered
! allows to use the timeordered version
! 
! tetrahedron method added by Michiel van Setten 25/11/2004
! m.vansetten@science.ru.nl
!
!*********************************************************************

  SUBROUTINE LR_OPTIC( &
          P,WDES,NONLR_S,NONL_S,W,LATT_CUR,LATT_INI, &
          T_INFO,INFO,IO,KPOINTS,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDUS,C_TO_US,SOFT_TO_C, &
          CHTOT,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM, &
          CHDEN,SV,LMDIM,IRDMAX,EFERMI,NEDOS_IN, LSTORE, LPOT)

    USE base
    USE lattice
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
    USE pawm
    USE wave_high
    USE subrot_cluster
    USE mlrf_main
    USE meta
    USE pead
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W          ! wavefunction
    TYPE (latt)        LATT_CUR, LATT_INI
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (kpoints_struct) KPOINTS
    TYPE (symmetry)    SYMM
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    TYPE (energy)      E
    
    INTEGER LMDIM,IRDMAX,IRDMAA,NEDOS_IN
    LOGICAL, OPTIONAL :: LSTORE

    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
    REAL(q) :: EFERMI                         ! position of Fermi-energy
    LOGICAL, OPTIONAL :: LPOT
!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
!  Hamiltonian
    REAL(q)   XCSIF(3,3)
! local variables
    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
    INTEGER :: NEDOS
!   response functions
    COMPLEX(q), ALLOCATABLE:: EPSDD(:,:,:) ! density-density response function 
    COMPLEX(q), ALLOCATABLE:: EPSJJ(:,:,:) ! velocity-velocity response function
    REAL(q), ALLOCATABLE:: COND_ENERGY(:,:,:,:), EDOS(:)
    COMPLEX(q) :: WPLASMA_INTER(3,3)
    REAL(q) :: CON(3,3), BIMAG(3,3)
    REAL(q) :: EMAX, EMAX_COND, DELTAE
    INTEGER IDIR, JDIR, I, ISTEP

130 FORMAT (5X, //, &
     &'----------------------------------------------------', &
     &'----------------------------------------------------'//)
!=======================================================================
! any unoccupied states ?
!=======================================================================
    EMAX=MAX_ENERGY_UNOCCUPIED(WDES,W)*1.2
    IF (EMAX<=0) THEN
       RETURN
    ENDIF
    IF (OMEGAMAX_OPTIC/=-1) THEN
       EMAX=OMEGAMAX_OPTIC
    ENDIF

    EMAX_COND=5
    IF (OMEGAMAX_OPTIC/=-1) THEN
       EMAX_COND=OMEGAMAX_OPTIC
    ENDIF

    CALL CHECK_OCCUPATIONS(W)
    IF ((KPOINTS%ISMEAR<=-4).AND.(.NOT.IS_INSULATING())) THEN
       CALL VTUTOR('W','TETNOMETAL',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
       CALL VTUTOR('W','TETNOMETAL',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
       RETURN
    ENDIF

    IF (IO%IU0>0) WRITE(IO%IU0,*) 'optical routines'

    ! smaller values are dangerous and fail more often than not
    NEDOS=MAX(NEDOS_IN,1000)
      
    DELTAE=EMAX/(NEDOS-1)
    ALLOCATE(EPSDD(NEDOS,3,3),EPSJJ(NEDOS,3,3))
    ALLOCATE(COND_ENERGY(NEDOS,3,3,WDES%NCDIJ), EDOS(NEDOS))

    IF (IO%IU0>=0) THEN
       WRITE(IO%IU0,*)'imaginary and real dielectric function'
       WRITE(17,*)'imaginary and real dielectric function'
    ENDIF
    IF (IO%IU6>=0) WRITE(IO%IU6,130)
!=======================================================================
! reset the potential
!=======================================================================
    ! just in case sync orbitals and eigenvalues among nodes
    CALL KPAR_SYNC_ALL(WDES,W)

    IF (INFO%LREAL) THEN
       CALL RSPHER(GRID,NONLR_S,LATT_CUR)
    ENDIF

      
    IF (LPOT) THEN
    IF (IO%IU0>=0) &
       WRITE(IO%IU0,*)'recalculating local potential from charge density'
    ! in most cases it is safer to recalculate the potential
      CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES,  &
         INFO,P,T_INFO,E,LATT_CUR, &
         CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
                  
      CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
         LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)

      CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
         WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
         E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
    ENDIF
    IF (.NOT. LUSEPEAD() .AND. LDO_METAGGA()) THEN
       CALL VTUTOR('W','METAGGARESPONSE',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
       CALL VTUTOR('W','METAGGARESPONSE',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
    ENDIF

    IF (IO%IU0>=0 .AND. IO%LOPEN) CALL WFORCE(17)
!=======================================================================
! response to external field
!=======================================================================
    NULLIFY(DEG_CLUSTER)
    CALL FIND_DEG_CLUSTERS(WDES, W, DEG_CLUSTER)

    CALL SET_NABIJ_AUG(P,T_INFO%NTYP)

    IF (ALLOCATED(CDER_BETWEEN_STATES)) THEN
       DEALLOCATE(CDER_BETWEEN_STATES) 
    ENDIF
    IF (ALLOCATED(ENERGY_DER)) THEN
       DEALLOCATE(ENERGY_DER)
    ENDIF

    NBANDS_CDER=MIN(LAST_FILLED_OPTICS(W)*2, WDES%NB_TOT)
    IF (LVEL) THEN
       NBANDS_CDER=WDES%NB_TOT
       IF (IO%IU0>0) WRITE(IO%IU0,*) 'derivatives for all bands'
    ENDIF
    ALLOCATE(CDER_BETWEEN_STATES(WDES%NB_TOT, NBANDS_CDER, WDES%NKPTS, WDES%ISPIN, 3))
    ALLOCATE(ENERGY_DER(WDES%NB_TOT, WDES%NKPTS, WDES%ISPIN, 3))
    CDER_BETWEEN_STATES=0
    ENERGY_DER=0

    CALL LRF_EPSILON( P, WDES, W, LATT_CUR, LATT_INI, T_INFO, INFO, IO, KPOINTS, EFERMI, &
      NONL_S, NONLR_S, IRDMAX, SV, DEG_CLUSTER, &
      GRID, GRIDC, GRIDUS, C_TO_US, EMAX, EMAX_COND, NEDOS, &
      INFO%LOVERL, CDIJ, CQIJ, LMDIM, SYMM, EPSDD, EPSJJ, COND_ENERGY, WPLASMON, WPLASMA_INTER,  CON, BIMAG, & 
      DELTAE, CDER_BETWEEN_STATES, ENERGY_DER)


900  FORMAT(/ & 
              ' plasma frequency squared (from intraband transitions) (eV^2)'/ & 
              '  ----------------------------------------------------', &
              '----------------------------------'/ &
              (3F12.3))

9002  FORMAT(/ &
              ' electrical conductivity sigma (Mega S m-1) (frequency dependency in vasprun.xml)'    / &
              '  ----------------------------------------------------', &
              '----------------------------------'/ &
              (3F12.3))

910  FORMAT(/ &
              ' plasma frequency squared (from interband transitions, int dw w eps(2)(w)) (eV^2)'/ &
              '  ----------------------------------------------------', &
              '----------------------------------'/ &
               3(6F12.3/), &
              '  ----------------------------------------------------', &
              '----------------------------------'/ &
              ' sumrule: sum of plasma frequencies squared should yield (valence only):',  1F12.3)

920  FORMAT(/ & 
              ' Anomalous Hall contribution (eV) 2/pi Imag((f_n-f_m)<u_m| d/dk_a u_n> <u_m| d/dk_b u_n>^*) '/ & 
              '  ----------------------------------------------------', &
              '----------------------------------'/ &
              (3F12.3))


1000 FORMAT(/ &
     &        '  frequency dependent IMAGINARY DIELECTRIC FUNCTION' &
     &       ,' (independent particle, no local field effects)',A,/&
     &        '     E(ev)  ', &
     &        4X,'X', 9X,'Y', 9X,'Z', 8X,'XY', 8X,'YZ', 8X,'ZX'/ &
     &        '  ----------------------------------------------------', &
     &        '----------------------------------------------------------'/ &
     &        (7F12.6))

1100 FORMAT(/ &
     &        '  frequency dependent      REAL DIELECTRIC FUNCTION' &
     &       ,' (independent particle, no local field effects)',A,/&
     &        '     E(ev)  ', &
     &        6X,'X', 11X,'Y', 11X,'Z', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
     &        '  ----------------------------------------------------', &
     &        '----------------------------------------------------------'/ &
     &        (7F12.6))


1200 FORMAT(' The outermost node in the dielectric function lies at epsilon=',F6.1)

!=======================================================================
! seek outermost node in the dielectric function
!=======================================================================
    DO IDIR=1,3
       DO I=NEDOS,2,-1
          IF (REAL(EPSDD(I,IDIR,IDIR),q)*REAL(EPSDD(I-1,IDIR,IDIR),q)<0) THEN
             IF (IDIR==1) THEN
                NODES_IN_DIELECTRIC_FUNCTION=(I-1)*DELTAE
             ELSE
                NODES_IN_DIELECTRIC_FUNCTION=MAX(NODES_IN_DIELECTRIC_FUNCTION,(I-1)*DELTAE)
             ENDIF
          ENDIF
       ENDDO
    ENDDO

    IF (IO%IU6>=0) THEN
       ISTEP=MIN(10,NEDOS/40)
       IF (KPOINTS%ISMEAR<0) ISTEP = 1

       WRITE(IO%IU6,900) WPLASMON
       !  in CGS the conversion factor is h^2/m_e  (4 pi e^2)
       WRITE(IO%IU6,9002) CON
       WRITE(IO%IU6,910) WPLASMA_INTER, INFO%NELECT/LATT_CUR%OMEGA*EDEPS*2*HSQDTM

       WRITE(IO%IU6,920) BIMAG

       WRITE(IO%IU6,1000) " density-density", & 
                      (DELTAE*(I-1),AIMAG(EPSDD(I,1,1)),AIMAG(EPSDD(I,2,2)),AIMAG(EPSDD(I,3,3)), &
                                    AIMAG(EPSDD(I,1,2)),AIMAG(EPSDD(I,2,3)),AIMAG(EPSDD(I,3,1)), &
                                    I=1,NEDOS,ISTEP)
       WRITE(IO%IU6,1100) " density-density", &
                      (DELTAE*(I-1),REAL(EPSDD(I,1,1),q),REAL(EPSDD(I,2,2),q),REAL(EPSDD(I,3,3),q), &
                                    REAL(EPSDD(I,1,2),q),REAL(EPSDD(I,2,3),q),REAL(EPSDD(I,3,1),q), &
                                    I=1,NEDOS,ISTEP)

       WRITE(IO%IU6,1000) " current-current (2nd set in vasprun.xml)", & 
                      (DELTAE*(I-1),AIMAG(EPSJJ(I,1,1)),AIMAG(EPSJJ(I,2,2)),AIMAG(EPSJJ(I,3,3)), &
                                    AIMAG(EPSJJ(I,1,2)),AIMAG(EPSJJ(I,2,3)),AIMAG(EPSJJ(I,3,1)), &
                                    I=1,NEDOS,ISTEP)
       WRITE(IO%IU6,1100) " current-current (2nd set in vasprun.xml)", &
                      (DELTAE*(I-1),REAL(EPSJJ(I,1,1),q),REAL(EPSJJ(I,2,2),q),REAL(EPSJJ(I,3,3),q), &
                                    REAL(EPSJJ(I,1,2),q),REAL(EPSJJ(I,2,3),q),REAL(EPSJJ(I,3,1),q), &
                                    I=1,NEDOS,ISTEP)


       WRITE(IO%IU6,*)
       WRITE(IO%IU6,1200) NODES_IN_DIELECTRIC_FUNCTION
       WRITE(IO%IU6,130)
    ENDIF

    ! write derivative of wavefunctions to file
    CALL WRT_CDER_BETWEEN_STATES(WDES, IO%IU0, IU=55)
!   CALL WRT_CDER_BETWEEN_STATES_FORMATTED(WDES,W,IO%IU0,IU=55)

    IF (LVEL) THEN
       CALL FULL_KPOINT_AND_ENERGY_DER( W, LATT_CUR,  IO)
    ENDIF

    IF (.NOT. PRESENT(LSTORE)) THEN
       DEALLOCATE(CDER_BETWEEN_STATES, ENERGY_DER)
    ELSE
       IF (.NOT. LSTORE) THEN
          DEALLOCATE(CDER_BETWEEN_STATES, ENERGY_DER)
       ENDIF
    ENDIF

    CALL XML_EPSILON_W(DELTAE, REAL(EPSDD,q), AIMAG(EPSDD), NEDOS)
    CALL XML_EPSILON_W(DELTAE, REAL(EPSJJ,q), AIMAG(EPSJJ), NEDOS)
    
    DO I=1,NEDOS
       EDOS(I)=EFERMI-EMAX_COND+(EMAX_COND*2)/(NEDOS-1)*(I-1)
    ENDDO

    DO I=1,WDES%ISPIN
       IF (I==1) THEN
          CALL XML_EPSILON_COND(EDOS, COND_ENERGY(1,1,1,I), NEDOS, "spin=1" )
       ELSE
          CALL XML_EPSILON_COND(EDOS, COND_ENERGY(1,1,1,I), NEDOS, "spin=2" )
       ENDIF
    ENDDO

    CALL FREE_DEG_CLUSTERS(WDES, DEG_CLUSTER)

    DEALLOCATE(EPSDD,EPSJJ)
    DEALLOCATE(EDOS,COND_ENERGY)


  END SUBROUTINE LR_OPTIC

!************************ SUBROUTINE FULL_KPOINT_AND_ENERGY_DER ********
!
! this subroutine calculates the velocity operatore
!     <phi_nk | d H / d k | phi_nk >
! and writes the result onto the file vasprun.xml
! the velocity operator is written using the full 
! (non symmetry reduce) k-point grid
!
!***********************************************************************

  SUBROUTINE FULL_KPOINT_AND_ENERGY_DER( W, LATT_CUR, IO)
    USE vaspxml
    USE base
    USE wave
    USE lattice
    USE full_kpoints

    TYPE (wavespin)    W         ! unperturbed wavefunctions
    TYPE (latt)        LATT_CUR  ! lattice parameters
    TYPE (in_struct)   IO        ! IO structure

    ! temporary array to store k-points in extended BZ, eigenvalues and second energy derivative
    REAL(q), ALLOCATABLE :: EIGENVAL(:, :, :, :), VKPT(:,:), WTKPT(:)
    INTEGER, PARAMETER :: NREP=1
    REAL(q) :: DER(3)
    INTEGER :: INDEX,  N1, N2, N3, ISP, NB, NK, I, J, K, L
    INTEGER :: NK_IRZ, ISP_IRZ, IDIR_OLD
    REAL(q) :: S(3,3)

    CALL CHECK_FULL_KPOINTS

    IF (KPOINTS_FULL%NKPTS/=KPOINTS_FULL%NKPX*KPOINTS_FULL%NKPY*KPOINTS_FULL%NKPZ) THEN
       CALL VTUTOR('W','KPOINTS HF',0.0_q,1, &
            &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
       CALL VTUTOR('W','KPOINTS HF',0.0_q,1, &
            &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
       RETURN
    ENDIF

    ALLOCATE(EIGENVAL(4, W%WDES%NB_TOT, KPOINTS_FULL%NKPTS*NREP*NREP*NREP, W%WDES%ISPIN), & 
             VKPT(3, KPOINTS_FULL%NKPTS*NREP*NREP*NREP), & 
             WTKPT(KPOINTS_FULL%NKPTS*NREP*NREP*NREP))
    WTKPT=1.0_q/KPOINTS_FULL%NKPTS

    INDEX=0
    DO N3=-KPOINTS_FULL%NKPZ*(NREP-1), KPOINTS_FULL%NKPZ-1
       DO N2=-KPOINTS_FULL%NKPY*(NREP-1), KPOINTS_FULL%NKPY-1
          DO N1=-KPOINTS_FULL%NKPX*(NREP-1), KPOINTS_FULL%NKPX-1
             INDEX=INDEX+1
             VKPT(1,INDEX)=REAL(N1,q)/KPOINTS_FULL%NKPX
             VKPT(2,INDEX)=REAL(N2,q)/KPOINTS_FULL%NKPY
             VKPT(3,INDEX)=REAL(N3,q)/KPOINTS_FULL%NKPZ


             NK= KPOINT_IN_FULL_GRID(VKPT(:,INDEX), KPOINTS_FULL)
             NK_IRZ=KPOINTS_FULL%NEQUIV(NK)

             
             ! determine transformation matrix in real space when going from
             ! old to new k-point
             S=0
             DO L=1,3
                DO K=1,3
                   DO J=1,3
                      DO I=1,3
                         S(L,I)=S(L,I)+LATT_CUR%A(L,K)*KPOINTS_FULL%ISYMOP(J,K,NK)*LATT_CUR%B(I,J)
                      ENDDO
                   ENDDO
                ENDDO
             ENDDO

             DO ISP=1, W%WDES%ISPIN
                ISP_IRZ=ISP
                IF (KPOINTS_FULL%SPINFLIP(NK)==1) THEN
                   ISP_IRZ=3-ISP
                ENDIF
                DO NB=1, W%WDES%NB_TOT

                   ! a vector in direction IDIR_OLD will rotate into a vector
                   !  S(:,IDIR_OLD)
                   DER=0
                   DO IDIR_OLD=1,3
                      DER(:)=DER(:)+S(:,IDIR_OLD)*ENERGY_DER(NB, NK_IRZ, ISP_IRZ, IDIR_OLD)
                   ENDDO

                   EIGENVAL(1  , NB, INDEX, ISP)= W%CELTOT( NB, NK_IRZ, ISP_IRZ)
                   EIGENVAL(2:4, NB, INDEX, ISP)= DER
                ENDDO
             ENDDO
             
          ENDDO
       ENDDO
    ENDDO

    CALL XML_TAG("electronvelocities")
    CALL XML_TAG("kpoints")
    CALL XML_KPOINTS_LIST( VKPT, WTKPT)
    CALL XML_CLOSE_TAG("kpoints")
    CALL XML_EIGENVALUES_EXT( EIGENVAL, 4, W%WDES%NB_TOT, KPOINTS_FULL%NKPTS*NREP*NREP*NREP, W%WDES%ISPIN)
    CALL XML_CLOSE_TAG("electronvelocities")

    DEALLOCATE(EIGENVAL, VKPT, WTKPT)

    
  END SUBROUTINE FULL_KPOINT_AND_ENERGY_DER

!************************ SUBROUTINE LRF_EPSILON ***********************
!
! this subroutine calculates imaginary part of the macroscopic 
! dielectric function
!
!  epsilon_q->0(G=0,G'=0,w) 
!
! in the random phase approximation (independent particle approximation)
! using a sum over unoccupied states
!
! optionally the matrix
!  CHAM(m,n,NK,ISP,j)= <u_m|- i d/dk_j | u_n> = - <u_m| r_j |u_n>
! is returned
! where m and n are band indices
! j is a cartesian index
!
!***********************************************************************

  SUBROUTINE LRF_EPSILON( P, WDES, W0, LATT_CUR, LATT_INI, T_INFO, INFO, IO, KPOINTS, EFERMI, &
      NONL_S, NONLR_S, IRDMAX, SV, DEG_CLUSTER, &
      GRID, GRIDC, GRIDUS, C_TO_US, EMAX, EMAX_COND, NEDOS, &
      LOVERL, CDIJ, CQIJ, LMDIM, SYMM, EPSDD, EPSJJ, COND_ENERGY, WPLASMON, WPLASMA_INTER, CON, BIMAG,  & 
      DELTAE, CHAM, ENERGY_DER)

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
    USE mlrf_main
    USE subrot_cluster
    USE kpoints_change
    USE fock
    USE pead
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W0         ! unperturbed wavefunctions
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    INTEGER IRDMAX
    TYPE (latt)        LATT_CUR, LATT_INI
    TYPE (info_struct)    INFO
    TYPE (in_struct)      IO
    TYPE (kpoints_struct) KPOINTS
    REAL(q)            EFERMI     ! fermi-energy
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS

    INTEGER LMDIM
    LOGICAL LOVERL
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
    TYPE (eigenf_cluster_pointer),POINTER :: DEG_CLUSTER(:,:)
    INTEGER NEDOS                 ! number of grid points in DOS
    TYPE (symmetry)    SYMM
    COMPLEX(q)  :: EPSDD(:,:,:)
    COMPLEX(q)  :: EPSJJ(:,:,:)
    REAL(q)  :: COND_ENERGY(:,:,:,:)
! local variables
    GDEFS   :: CHAM(:,:,:,:,:)
    REAL(q) :: ENERGY_DER(:,:,:,:)
    REAL(q) :: WPLASMON(3,3), CON(3,3), BIMAG(3,3)
    COMPLEX(q) :: WPLASMA_INTER(3,3)
    INTEGER IDIR, JDIR,I
    REAL(q) EMAX, EMAX_COND, DELTAE  ! maximum energy
    LOGICAL LDONE
    TYPE (wavespin)     :: W1     ! stores  the derivative of the wavefunction with respect to k_i
    TYPE (wavefun)      :: WTMP   ! temporary
    TYPE (wavedes)      :: WDES1  ! descriptor for W1
    LOGICAL :: LWARN

! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    INTEGER  IWINDOW
    REAL(q)  GTRANS,AP, W
    INTEGER N1,N2,NK,ISP

    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

    ENERGY_DER=0

    WDES1=WDES
    WDES1%NB_TOT=NBANDS_CDER
    WDES1%NBANDS=NBANDS_CDER/WDES1%NB_PAR

    CALL ALLOCW(WDES1,W1,WTMP,WTMP)

    W1%CW   =0
    W1%CPROJ=0

    DO IDIR=1,3
       IF (IO%IU0>=0) WRITE(IO%IU0,*) 'direction ',IDIR

       IF (LUSEPEAD()) THEN
          W1%CW=0
          W1%CPROJ=0
          W1%CELTOT=0
          CALL PEAD_DPSI_DK_IDIR(W0,KPOINTS,P,CQIJ,LATT_CUR,T_INFO,IDIR,W1)
       ELSE       
!=======================================================================
! determine exact first order change of wavefunction with respect to ik 
!  - i d/dk  u_n = - r u_n
! see LRF_RPHI0 for details
!=======================================================================
          CALL FOCK_K_DER_ANALYT(KPOINTS, GRID, LATT_CUR, LATT_INI, T_INFO,  NONLR_S, NONL_S, W0, W1, &
         &   LMDIM, P, CQIJ, SYMM, IDIR, LDONE, IO%IU0, IO%IU6)
!test
!         CALL INPROD_W(W1,W0,CHAM(:,:,:,:,IDIR),LOVERL,IO%IU0)
!         STOP
!testend
          CALL LRF_RPHI0( &
               P,NONLR_S,NONL_S,W0,LATT_CUR, &
               T_INFO,INFO,IO,GRID,GRIDC,GRIDUS,C_TO_US,IRDMAX, &
               CDIJ,CQIJ,SV,LMDIM,DEG_CLUSTER, IDIR, W1, LDONE)
       ENDIF

       ! CHAM(n,m) = - <u_n| r | u_m > 
       CALL INPROD_W(W1,W0,CHAM(:,:,:,:,IDIR),LOVERL,IO%IU0)

       
       DO ISP=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS
#ifdef MPI
             IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
             DO N1=1,WDES1%NB_TOT
                ENERGY_DER(N1,NK,ISP,IDIR)=W1%CELTOT(N1,NK,ISP)
             ENDDO
          ENDDO
       ENDDO
    ENDDO
!test
!    DO NK=1,WDES%NKPTS
!       DO N1=1,WDES%NB_TOT
!          IF (W0%FERTOT(N1,NK,1)>0.2) THEN
!             WRITE(*,'(3I,50F10.5)') 1,NK,N1,WDES%VKPT(:,NK),ENERGY_DER(N1,NK,1,1)
!             WRITE(*,'(3I,50F10.5)') 2,NK,N1,WDES%VKPT(:,NK),ENERGY_DER(N1,NK,1,2)
!             WRITE(*,'(3I,50F10.5)') 3,NK,N1,WDES%VKPT(:,NK),ENERGY_DER(N1,NK,1,3)
!          END IF
!       END DO
!    END DO
!    STOP
!test
#ifdef gammareal
    CALLMPI( M_sum_single(WDES%COMM_KINTER, CHAM, SIZE(CHAM)))
#else
    CALLMPI( M_sum_single(WDES%COMM_KINTER, CHAM, 2*SIZE(CHAM)))
#endif
    CALLMPI( M_sum_d(WDES%COMM_KINTER, ENERGY_DER, SIZE(ENERGY_DER)))
       
!=======================================================================
! determine imaginary part of the frequency dependent macroscopic 
! dielectric matrix
!=======================================================================
    BIMAG=0
    DO IDIR=1,3
      DO JDIR=1,3
        IF (KPOINTS%ISMEAR >= -1) THEN
          CALL EPSILON_IMAG( WDES, W0, CHAM(:,:,:,:,IDIR), CHAM(:,:,:,:,JDIR), & 
          ENERGY_DER(:,:,:,IDIR), ENERGY_DER(:,:,:,JDIR), EFERMI, &
          NEDOS, EPSDD(:,IDIR,JDIR), EPSJJ(:,IDIR,JDIR), DELTAE, KPOINTS%ISMEAR, KPOINTS%SIGMA, & 
          LATT_CUR%OMEGA, WPLASMON(IDIR, JDIR), CON(IDIR, JDIR), BIMAG(IDIR, JDIR), RTIME, LCSHIFT)

        ELSEIF (KPOINTS%ISMEAR <= -4) THEN
          CALL EPSILON_IMAG_TET( WDES, W0, CHAM(:,:,:,:,IDIR), CHAM(:,:,:,:,JDIR), EMAX, &
          NEDOS, EPSDD(:,IDIR,JDIR), DELTAE, LATT_CUR%OMEGA, IO, INFO, KPOINTS)
        ENDIF
      ENDDO
    ENDDO
      
    COND_ENERGY=0
    CALL CONDUCTIVITY_ENERGY_RESOLVED( WDES, W0, ENERGY_DER(:,:,:,:), EFERMI, &
         NEDOS, EMAX_COND, COND_ENERGY(:,:,:,:), INFO%NELECT, INFO%NUP_DOWN, KPOINTS, & 
         LATT_CUR%OMEGA, RTIME)
    ! first pin is almost always screwed (since it contains integrated contribution
    ! from eigenvalues outside range
    COND_ENERGY(1,:,:,:)=COND_ENERGY(2,:,:,:)
    
    IF (KPOINTS%ISMEAR >= -1) THEN
       CALLMPI( M_sum_z(WDES%COMM, EPSDD, SIZE(EPSDD)))
       CALLMPI( M_sum_z(WDES%COMM, EPSJJ, SIZE(EPSJJ)))
       CALLMPI( M_sum_d(WDES%COMM, BIMAG, SIZE(BIMAG)))
       CALLMPI( M_sum_d(WDES%COMM_KINTER, WPLASMON, SIZE(WPLASMON)))
       CALLMPI( M_sum_d(WDES%COMM_KINTER, CON, SIZE(CON)))
    ENDIF

    ! CON is presently not calculated by EPSILON_IMAG_TET
    IF (KPOINTS%ISMEAR <= -4) THEN
       CON=COND_ENERGY(NEDOS/2,:,:,1)
       IF (WDES%ISPIN==2) THEN
          CON=COND_ENERGY(NEDOS/2,:,:,1)+COND_ENERGY(NEDOS/2,:,:,2)
       ENDIF
       WPLASMON=CON/ABS(RTIME)/CONTCON
    ENDIF

    DO I=1,NEDOS
       IF (SYMM%ISYM>0) CALL TSYM_CMPLX(EPSDD(I,:,:),ISYMOP,NROTK,LATT_CUR%A)
       IF (SYMM%ISYM>0) CALL TSYM_CMPLX(EPSJJ(I,:,:),ISYMOP,NROTK,LATT_CUR%A)
       DO ISP=1,WDES%NCDIJ
          IF (SYMM%ISYM>0) CALL TSYM(COND_ENERGY(I,:,:,ISP),ISYMOP,NROTK,LATT_CUR%A)
       ENDDO
    ENDDO

    IF (SYMM%ISYM>0) CALL TSYM(WPLASMON(:,:),ISYMOP,NROTK,LATT_CUR%A)
    IF (SYMM%ISYM>0) CALL TSYM(CON(:,:),ISYMOP,NROTK,LATT_CUR%A)
    IF (SYMM%ISYM>0) CALL TSYM(BIMAG(:,:),ISYMOP,NROTK,LATT_CUR%A)

    IWINDOW=0
    LWARN=.FALSE.
    DO IDIR=1,3
       DO JDIR=1,3
          CALL EPSILON_REAL( WDES%COMM, NEDOS, EPSDD(:,IDIR,JDIR), LWARN, &
               DELTAE, LCSHIFT, IWINDOW, WPLASMA_INTER(IDIR, JDIR))
          IF (JDIR==IDIR) EPSDD(:,IDIR,IDIR)=EPSDD(:,IDIR,IDIR)+1

          CALL EPSILON_REAL( WDES%COMM, NEDOS, EPSJJ(:,IDIR,JDIR), LWARN, & 
               DELTAE, LCSHIFT, IWINDOW)
       ENDDO
    ENDDO

    IF (LWARN) THEN
       CALL VTUTOR('W','SMALL CSHIFT',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU6,2)
       CALL VTUTOR('W','SMALL CSHIFT',0.0_q,1, &
       &           0,1,(0.0_q,0.0_q),1,.FALSE.,1,IO%IU0,2)
    ENDIF
    ! add off diagonal anomalous Hall terms
    DO IDIR=1,3
       DO JDIR=1,3
          IF (IDIR==JDIR) CYCLE
#ifdef timeordered
          DO I=2,NEDOS
             EPSDD(I,IDIR,JDIR)=EPSDD(I,IDIR,JDIR)+(0._q,1._q)*BIMAG(IDIR,JDIR)/(DELTAE*(I-1))
          ENDDO
          EPSDD(1,IDIR,JDIR)=EPSDD(2,IDIR,JDIR)
#else
          DO I=1,NEDOS
             EPSDD(I,IDIR,JDIR)=EPSDD(I,IDIR,JDIR)+(0._q,1._q)*BIMAG(IDIR,JDIR)/(DELTAE*(I-1)+CMPLX(0,LCSHIFT,q))
          ENDDO
#endif
       ENDDO
   ENDDO


    DO IDIR=1,3
       DO JDIR=1,3
          DO I=2,NEDOS
             ! shifting the poles does not improve results
             ! EPSJJ(I,IDIR,JDIR)=(EPSJJ(I,IDIR,JDIR)-EPSJJ(1,IDIR,JDIR))/(DELTAE*(I-1)+CMPLX(0,LCSHIFT,q))**2
             EPSJJ(I,IDIR,JDIR)=(EPSJJ(I,IDIR,JDIR)-EPSJJ(1,IDIR,JDIR))/(DELTAE*(I-1))**2
          ENDDO
          IF (JDIR==IDIR) EPSJJ(:,IDIR,IDIR)=EPSJJ(:,IDIR,IDIR)+1
       ENDDO
    ENDDO
    ! set first frequency point to second time point
    EPSJJ(1,:,:)=EPSJJ(2,:,:)
    ! add Drude term, if WPLASMAI (or RTIME) has been read
    ! - w_p^2 d/dw delta(w) = w_p^2 d/dw 1/(w + i wi)
    ! TODO: one should diagonalize WPLASMON and add contributions along principle axis
    IF (WPLASMAI>0) THEN
       DO IDIR=1,3
          DO I=1,NEDOS
             ! avoid zero frequency
             W=DELTAE*(I-1)
             EPSDD(I,IDIR,IDIR)=EPSDD(I,IDIR,IDIR)+ & 
                  WPLASMON(IDIR,IDIR)/(W**2+WPLASMAI**2)* & 
                  CMPLX(-1.0_q,WPLASMAI/MAX(W,DELTAE))
             EPSJJ(I,IDIR,IDIR)=EPSJJ(I,IDIR,IDIR)+ &
                  WPLASMON(IDIR,IDIR)/(W**2+WPLASMAI**2)* & 
                  CMPLX(-1.0_q,WPLASMAI/MAX(W,DELTAE))
          ENDDO
       ENDDO
    ENDIF

    CALL DEALLOCW(W1)

  END SUBROUTINE LRF_EPSILON

!***********************************************************************
! 
! function that yields .TRUE. if a state is empty
!
!***********************************************************************

  FUNCTION EMPTY(F)
    USE prec
    IMPLICIT NONE
    LOGICAL EMPTY
    REAL(q) :: F
    IF (ABS(F) <0.5) THEN
       EMPTY=.TRUE.
    ELSE
       EMPTY=.FALSE.
    ENDIF
    
  END FUNCTION EMPTY

  FUNCTION FILLED(F)
    USE prec
    IMPLICIT NONE
    LOGICAL FILLED
    REAL(q) :: F
    IF (ABS(F-1)<0.5) THEN
       FILLED=.TRUE.
    ELSE
       FILLED=.FALSE.
    ENDIF
    
  END FUNCTION FILLED


!************************ SUBROUTINE MAX_ENERGY_UNOCCUPIED *************
! 
! determine the energy range of the unoccupied states with respect
! to the valence band minimum
!
!***********************************************************************

  FUNCTION MAX_ENERGY_UNOCCUPIED(WDES, W)
    USE prec
    USE wave
    IMPLICIT NONE
    REAL(q) :: MAX_ENERGY_UNOCCUPIED
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W

    INTEGER NK, N, ISP
    REAL(q) :: EMIN, EMAX

    EMIN= 1000
    EMAX=-1000

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          ! few eigenstates no problem
          IF (WDES%NB_TOTK(NK,ISP)<100) THEN
             DO N=1,WDES%NB_TOTK(NK,ISP)
                EMIN=MIN(EMIN,REAL(W%CELTOT(N,NK,ISP),q))
                EMAX=MAX(EMAX,REAL(W%CELTOT(N,NK,ISP),q))
             ENDDO
          ELSE
          ! very many eigenstates
          ! sometimes the diagonalization screws up some eigenvalues
          ! at the top (need to protect us from those)
             DO N=1,WDES%NB_TOTK(NK,ISP)-4
                EMIN=MIN(EMIN,REAL(W%CELTOT(N,NK,ISP),q))
                EMAX=MAX(EMAX,REAL(W%CELTOT(N,NK,ISP),q))
             ENDDO
             DO N=WDES%NB_TOTK(NK,ISP)-3,WDES%NB_TOTK(NK,ISP)
                ! screwed eigenvalues are usually well seperated from the rest
                ! 40 eV seems to be a good safe guard
                IF (REAL(W%CELTOT(N,NK,ISP),q)>REAL(W%CELTOT(N-1,NK,ISP),q)+40) THEN
                   EXIT
                ENDIF
                EMAX=MAX(EMAX,REAL(W%CELTOT(N,NK,ISP),q))
             ENDDO
          ENDIF
       ENDDO
    ENDDO

    MAX_ENERGY_UNOCCUPIED=EMAX-EMIN
     
  END FUNCTION MAX_ENERGY_UNOCCUPIED


  FUNCTION MAX_ENERGY_OCC_UNOCCUPIED(WDES, W, NB)
    USE prec
    USE wave
    IMPLICIT NONE
    REAL(q) :: MAX_ENERGY_OCC_UNOCCUPIED
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W
    INTEGER NB ! maximum band to be considered

    INTEGER NK, N, ISP
    REAL(q) :: EMIN, EMAX

    EMIN= 1000
    EMAX=-1000

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          DO N=1,NB
             EMIN=MIN(EMIN,REAL(W%CELTOT(N,NK,ISP),q))
             EMAX=MAX(EMAX,REAL(W%CELTOT(N,NK,ISP),q))
          ENDDO
       ENDDO
    ENDDO

    MAX_ENERGY_OCC_UNOCCUPIED=EMAX-EMIN
     
  END FUNCTION MAX_ENERGY_OCC_UNOCCUPIED

!************************ SUBROUTINE MIN_ENERGY_UNOCCUPIED *************
! 
! determine the energy range of the unoccupied states with respect
! to the bottom of the conduction band
!
!***********************************************************************

  FUNCTION MIN_ENERGY_UNOCCUPIED(WDES, W)
    USE prec
    USE wave
    IMPLICIT NONE
    REAL(q) :: MIN_ENERGY_UNOCCUPIED
    TYPE (wavedes)     WDES
    TYPE (wavespin)    W

    INTEGER NK, N, ISP
    REAL(q) :: EMIN, EMAX

    EMIN= 1000
    EMAX= 1000

    DO ISP=1,WDES%ISPIN
       DO NK=1,WDES%NKPTS
          DO N=1,WDES%NB_TOTK(NK,ISP)
             EMIN=MIN(EMIN,REAL(W%CELTOT(N,NK,ISP),q))
             IF (W%FERTOT(N,NK,ISP)<=0.5) THEN
                EMAX=MIN(EMAX,REAL(W%CELTOT(N,NK,ISP),q))
             ENDIF
          ENDDO
       ENDDO
    ENDDO

    MIN_ENERGY_UNOCCUPIED=EMAX-EMIN
     
  END FUNCTION MIN_ENERGY_UNOCCUPIED

!************************ SUBROUTINE INPROD_W **************************
!
! this subroutine calculates the inproduct between two sets of
! wavefunctions W1 and W0
!  CHAM(j,i) = <W0(j) | W1(i) >
! 
!***********************************************************************

  SUBROUTINE INPROD_W(W1,W0,CHAM,LOVERL,IU0)
      USE prec
      USE wave_high
      USE dfast
      USE mpimy

      IMPLICIT NONE

      TYPE (wavespin)    W1
      TYPE (wavespin)    W0
      INTEGER IU0                    ! debug output
      LOGICAL LOVERL                 ! overlap matrix used (i.e. US-PP, PAW)
      GDEFS :: CHAM(:,:,:,:)
! local variables
      INTEGER ISP, NK, NSTRIP, NSTRIP_ACT
      INTEGER NPOS, N, NB_TOT, NB_TOT_W1
      INTEGER N1, N2, NPL2
      TYPE (wavedes1)   WDES1          ! descriptor for one k-point
      TYPE (wavedes1)    WDES1_W1       ! descriptor for one k-point
      TYPE (wavefuna)    WA             ! array to store wavefunction
      GDEF, ALLOCATABLE   :: CHAM_NORMAL(:,:)
!-----------------------------------------------------------------------
! determine whether redistribution is required
!-----------------------------------------------------------------------
      NB_TOT=W0%WDES%NB_TOT
      NB_TOT_W1=W1%WDES%NB_TOT

      IF (SIZE(CHAM,1) /= NB_TOT .OR. SIZE(CHAM,2) /= NB_TOT_W1) THEN
         WRITE(*,*) 'internal error in INPROD_W: shape not conform',SIZE(CHAM,1), NB_TOT, SIZE(CHAM,2), NB_TOT_W1
         STOP
      ENDIF

      NSTRIP=MIN(NSTRIP_STANDARD_GLOBAL,W1%WDES%NB_TOT)

      ALLOCATE(CHAM_NORMAL(NB_TOT, NSTRIP))
!=======================================================================
      spin:  DO ISP=1,W0%WDES%ISPIN
      kpoint: DO NK=1,W0%WDES%NKPTS
!=======================================================================
#ifdef MPI
      IF (MOD(NK-1,W0%WDES%COMM_KINTER%NCPU).NE.W0%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      CALL SETWDES(W0%WDES,WDES1,NK)
      CALL SETWDES(W1%WDES,WDES1_W1,NK)
      WA=ELEMENTS(W1, WDES1_W1, ISP)
      
    ! redistribute the projected wavefunctions
      IF (WDES1%DO_REDIS) THEN
        CALL REDIS_PW  (WDES1, W0%WDES%NBANDS, W0%CW   (1,1,NK,ISP))
        CALL REDIS_PROJ(WDES1, W0%WDES%NBANDS, W0%CPROJ(1,1,NK,ISP))
        CALL REDIS_PW  (WDES1_W1, W1%WDES%NBANDS, W1%CW   (1,1,NK,ISP))
        CALL REDIS_PROJ(WDES1_W1, W1%WDES%NBANDS, W1%CPROJ(1,1,NK,ISP))
      ENDIF
      
      DO NPOS=1,NB_TOT_W1,NSTRIP
         NSTRIP_ACT=MIN(NB_TOT_W1+1-NPOS, NSTRIP)
         CHAM_NORMAL=0
         CALL ORTH2_NOSUBINDEX( &
              W0%CW(1,1,NK,ISP),WA%CW_RED(1,NPOS),W0%CPROJ(1,1,NK,ISP), &
              WA%CPROJ_RED(1,NPOS),NB_TOT, &
              NPOS,NSTRIP_ACT,WDES1%NPL_RED,WDES1%NPRO_O_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,CHAM_NORMAL(1,1))

         CALLMPI( M_sum_g(WDES1%COMM_KIN,CHAM_NORMAL(1,1),NB_TOT*NSTRIP_ACT))

         DO N2=1,NSTRIP_ACT
            DO N1=1,NB_TOT
               CHAM(N1,N2+NPOS-1,NK,ISP)=CHAM_NORMAL(N1,N2)
            ENDDO
         ENDDO
      ENDDO
      !
      !  W1 might be correct only for occupied  states such that only the lower
      ! triangle is correct, fill the upper one
      DO  N1=1,NB_TOT_W1
         DO  N2=N1+1,NB_TOT_W1
            CHAM(N1,N2,NK,ISP)=GCONJG(CHAM(N2,N1,NK,ISP))
         ENDDO
      ENDDO
#ifdef debug
      IF (IU0>=0) CALL DUMP_HAM_SINGLE( " <phi_j| nabla |phi_i(1)>",W0%WDES, CHAM(1,1,NK,ISP))
#endif

      IF (WDES1%DO_REDIS) THEN
         CALL REDIS_PW  (WDES1, W0%WDES%NBANDS, W0%CW   (1,1,NK,ISP))
         CALL REDIS_PROJ(WDES1, W0%WDES%NBANDS, W0%CPROJ(1,1,NK,ISP))
         DWRITE "redis ok"
      ENDIF
!=======================================================================
      ENDDO kpoint
      ENDDO spin
!=======================================================================

      DEALLOCATE(CHAM_NORMAL)

      RETURN
    END SUBROUTINE INPROD_W

!***********************************************************************
!
! perform a unitary transformation between the optical matrix elements
! for a selected k point and spin component
! this routine is scheduled for removal since pead should be used 
! instead
!   M' = U+ M U
! since only part of the matrix am is stored the algebra is somewhat
! complicated
!
!
!***********************************************************************

!
! old version certainly wrong
!
    SUBROUTINE ROTATE_CDER_(WDES, CTRANS, NK, ISP)
      USE wave_high
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      GDEF CTRANS(:,:)
      INTEGER :: NK, ISP
    ! local
      GDEFS, ALLOCATABLE ::  C(:,:), CTRANS_single(:,:)
      INTEGER :: NB_TOT, IDIR, NB

      NB_TOT=WDES%NB_TOT

      ALLOCATE(C(NB_TOT, NBANDS_CDER), CTRANS_single(NBANDS_CDER, NBANDS_CDER))

      ! initialize to unity matrix
      CTRANS_single=0
      DO NB=1,SIZE(CTRANS_single,1)
         CTRANS_single(NB,NB)=1
      ENDDO

      NB=MIN(NBANDS_CDER,SIZE(CTRANS,1))

      ! overwrite with supplied matrix
      CTRANS_single(1:NB,1:NB)=CTRANS(1:NB,1:NB)

      IF (NK==1) THEN
         WRITE(*,*) NBANDS_CDER
         WRITE(*,'(16F8.4)') CDER_BETWEEN_STATES(1:8,1:NBANDS_CDER,1,1,1)
         WRITE(*,*)
         WRITE(*,'(16F8.4)') CTRANS_single(1:8,1:8)
      ENDIF
      
      DO IDIR=1,3
#ifdef gammareal
         CALL SGEMM('N', 'N', NB_TOT, NBANDS_CDER, NBANDS_CDER, 1.0, &
              &        CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT, CTRANS_single(1,1), &
              &        NBANDS_CDER, 0.0, C(1,1), NB_TOT)
         CALL SGEMM('T', 'N', NBANDS_CDER, NBANDS_CDER, NBANDS_CDER, 1.0, &
              &        CTRANS_single(1,1), NBANDS_CDER, C(1,1), &
              &        NB_TOT, 0.0, CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT)
#else
         CALL CGEMM('N', 'N', NB_TOT, NBANDS_CDER, NBANDS_CDER, (1.0,0.0), &
              &        CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT, CTRANS_single(1,1), &
              &        NBANDS_CDER, (0.0,0.0), C(1,1), NB_TOT)
         CALL CGEMM('C', 'N', NBANDS_CDER, NBANDS_CDER, NBANDS_CDER, (1.0,0.0), &
              &        CTRANS_single(1,1), NBANDS_CDER, C(1,1), &
              &        NB_TOT, (0.0,0.0), CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT)
#endif
         CDER_BETWEEN_STATES(NBANDS_CDER+1:NB_TOT,1:NBANDS_CDER,NK,ISP,IDIR)=  &
                      C(NBANDS_CDER+1:NB_TOT,1:NBANDS_CDER)
      ENDDO

      IF (NK==1) THEN
         WRITE(*,*)
         WRITE(*,'(16F8.4)') CDER_BETWEEN_STATES(1:8,1:NBANDS_CDER,1,1,1)
      ENDIF
      DEALLOCATE(C, CTRANS_single)

    END SUBROUTINE ROTATE_CDER_

!
! new version, also does not seem to give satisfactory results
!
    SUBROUTINE ROTATE_CDER(WDES, CTRANS, NK, ISP)
      USE wave_high
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      GDEF CTRANS(:,:)
      INTEGER :: NK, ISP
    ! local
      GDEFS, ALLOCATABLE ::  C(:,:),C0(:,:), CTRANS_single(:,:)
      INTEGER :: NB_TOT, IDIR, NB

      NB_TOT=WDES%NB_TOT
      ALLOCATE(C(NB_TOT, NBANDS_CDER), C0(NB_TOT,NB_TOT), CTRANS_single(NB_TOT, NB_TOT))

      ! initialize to unity matrix
      CTRANS_single=0
      DO NB=1,SIZE(CTRANS_single,1)
         CTRANS_single(NB,NB)=1
      ENDDO

      NB=MIN(NB_TOT,SIZE(CTRANS,1))
      ! overwrite with supplied matrix
      CTRANS_single(1:NB,1:NB)=CTRANS(1:NB,1:NB)

      
      DO IDIR=1,3
         C0=0
         C0(1:NB_TOT, 1:NBANDS_CDER)= CDER_BETWEEN_STATES(1:NB_TOT, 1:NBANDS_CDER,NK,ISP,IDIR)
         DO NB=1,NBANDS_CDER
            C0(NB,NBANDS_CDER+1:NB_TOT)=GCONJG( CDER_BETWEEN_STATES(NBANDS_CDER+1:NB_TOT,NB,NK,ISP,IDIR))
         ENDDO
#ifdef gammareal
         CALL SGEMM('N', 'N', NB_TOT, NBANDS_CDER, NB_TOT, 1.0, &
              &        C0(1,1), NB_TOT, CTRANS_single(1,1), &
              &        NB_TOT, 0.0, C(1,1), NB_TOT)
         CALL SGEMM('T', 'N', NB_TOT, NBANDS_CDER, NB_TOT, 1.0, &
              &        CTRANS_single(1,1), NB_TOT, C(1,1), &
              &        NB_TOT, 0.0, CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT)
#else
         CALL CGEMM('N', 'N', NB_TOT, NBANDS_CDER, NB_TOT, (1.0,0.0), &
              &        C0(1,1), NB_TOT, CTRANS_single(1,1), &
              &        NB_TOT, (0.0,0.0), C(1,1), NB_TOT)
         CALL CGEMM('C', 'N', NB_TOT, NBANDS_CDER, NB_TOT, (1.0,0.0), &
              &        CTRANS_single(1,1), NB_TOT, C(1,1), &
              &        NB_TOT, (0.0,0.0), CDER_BETWEEN_STATES(1,1,NK,ISP,IDIR), NB_TOT)
#endif
      ENDDO

      DEALLOCATE(C0, C, CTRANS_single)

    END SUBROUTINE ROTATE_CDER

!************************ SUBROUTINE EPSILON_IMAG **********************
!
! this subroutine calculates the imaginary part of the dielectric function
! for a specific direction
! as argument it requires the matrix of inproducts between
! all bands and the derivative of the bands  with respect 
! to a specific cartesian component of k
!  CHAM(m,n) = <u_m| - i d/dk_i  |u_n> = <u_m|- r_i |u_n>
! ideally the index n should be restricted to filled states
!  DER1 = <u_m| d H / d k_1 | u_m> = d epsilon_k,m / d k
! the off diagonal terms of the spectral function of the optical
! conductivity can have an imaginary contribution, which is calculated
! if the flag LIMAG is set
!
!***********************************************************************

    SUBROUTINE EPSILON_IMAG( WDES, W0, CHAM1, CHAM2, DER1, DER2, EFERMI, & 
      NEDOS, EPSDD, EPSJJ, DELTAE, ISMEAR, SIGMA, OMEGA, WPLASMON, CON, BIMAG, TAU, CSHIFT)
      USE wave
      USE constant
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      TYPE (wavespin)    W0
      GDEFS :: CHAM1(:,:,:,:)
      GDEFS :: CHAM2(:,:,:,:)
      REAL(q) :: DER1(:,:,:), DER2(:,:,:), EFERMI
      INTEGER NEDOS
      COMPLEX(q) EPSDD(:), EPSJJ(:)
      REAL(q) DELTAE
      INTEGER ISMEAR
      REAL(q) SIGMA
      REAL(q) OMEGA
      REAL(q) WPLASMON, CON, BIMAG, TAU, CSHIFT
    !local
      INTEGER ISP, NK, N1, N2
      REAL(q) WEIGHT
      COMPLEX(q) A
      REAL(q)  DECEL, ENERGY, DFUN, SFUN

      EPSDD=0
      EPSJJ=0
      WPLASMON=0
      CON=0

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         ! N1 = n  corresponds to "occupied" bands
         DO N1=1,WDES%NB_TOT
         ! N1 must be smaller tan NBANDS_CDER
         IF (N1>NBANDS_CDER) CYCLE

         ENERGY=W0%CELTOT(N1,NK,ISP)-EFERMI
         IF (ISMEAR==-1) THEN
            SFUN=F(-ENERGY,ABS(SIGMA))
            DFUN=G(ENERGY,ABS(SIGMA))
         ELSE
            CALL DELSTP(ISMEAR, ENERGY/ABS(SIGMA), DFUN, SFUN)
            DFUN=DFUN/ABS(SIGMA)
         ENDIF
         ! DFUN d f / d epsilon (= delta)
         ! conversion factor 4 pi e^2/ volume to obtain polarization
         WPLASMON=WPLASMON+DFUN*DER1(N1,NK,ISP)*DER2(N1,NK,ISP)*WDES%RSPIN*WDES%WTKPT(NK)*(4*PI*FELECT/OMEGA)
         CON=CON          +DFUN*DER1(N1,NK,ISP)*DER2(N1,NK,ISP)*WDES%RSPIN*WDES%WTKPT(NK)*(4*PI*FELECT/OMEGA)*ABS(TAU)*CONTCON 

         ! N2 = m corresponds to "empty" bands
         ! distribute this in round robin fashion among COMM_KIN
#ifdef MPI
         DO N2=N1+WDES%COMM_KIN%NODE_ME,WDES%NB_TOT, WDES%COMM_KIN%NCPU
#else
         DO N2=N1+1,WDES%NB_TOT
#endif
            ! CHAM stores  <u_m|  -i d/dk_j u_n>
            ! the polarizability is then given by    4 pi^2 e/omega  <u_m|  -i d/dk_j u_n>^* <u_m|  -i d/dk_j u_n> 
            ! one can also calculate the polarizability from the related velocity operator 
            !   v= - <u_m|  [H,r] | u_n> = - <u_m| r | u_n> (e_m-e_n) (collected in EPSJJ)
            WEIGHT=(W0%FERTOT(N1,NK,ISP)-W0%FERTOT(N2,NK,ISP))*WDES%RSPIN*WDES%WTKPT(NK)
            DECEL =(W0%CELTOT(N2,NK,ISP)-W0%CELTOT(N1,NK,ISP))
            ! contribution <u_m| - i d/dk_i u_n> <u_m| - i d/dk_j  u_n>*
            ! TODO gK: 28.12.16 conjugated to have same sign as Kunes, Marini on off diagonal terms e.g. xy 
            A =CHAM1(N2,N1,NK,ISP)*GCONJG(CHAM2(N2,N1,NK,ISP))
            ! conversion factor 4 pi^2 e^2/ volume to obtain polarization
            CALL SLOT( REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE,  WEIGHT*A*4*PI*PI*FELECT/OMEGA, EPSDD)
            CALL SLOT( REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE,  WEIGHT*A*4*PI*PI*FELECT/OMEGA*DECEL*DECEL, EPSJJ)
            BIMAG=BIMAG+WEIGHT*4*PI*PI*FELECT/OMEGA*AIMAG(A)*2/PI

            ! add contribution for negative frequencies (exchanged pair)
            WEIGHT=-WEIGHT
#ifdef timeordered
            CALL SLOT(-REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE, WEIGHT*A*4*PI*PI*FELECT/OMEGA, EPSDD)
            CALL SLOT(-REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE, WEIGHT*A*4*PI*PI*FELECT/OMEGA*DECEL*DECEL, EPSJJ)
#else
            CALL SLOT(-REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE, WEIGHT*CONJG(A)*4*PI*PI*FELECT/OMEGA, EPSDD)
            CALL SLOT(-REAL(DECEL,q), ISMEAR, SIGMA, NEDOS, DELTAE, WEIGHT*CONJG(A)*4*PI*PI*FELECT/OMEGA*DECEL*DECEL, EPSJJ)
#endif


         ENDDO
         ENDDO
      ENDDO
      ENDDO

      ! recalculate BIMAG from the pinned values in the array EPSDD
      ! this is more accurate than above, since it is compatible with what we
      ! subsequently use
      ! (factor 2/PI compare calculation of WPLASMA_INTER in EPSILON_REAL)
      BIMAG=0
      DO N1=1,NEDOS
         BIMAG=BIMAG+AIMAG(EPSDD(N1))
      ENDDO
      BIMAG=BIMAG*DELTAE*2/PI

    END SUBROUTINE EPSILON_IMAG


    FUNCTION F(E,SIG)
      REAL(q) F,E,SIG
      F=  1/(1 + EXP(E /SIG))
    END FUNCTION F
    
    FUNCTION G(E,SIG)
      REAL(q) G,E,SIG
      G=  1/EXP(E/SIG)/(1 + EXP(-E/SIG))**2/SIG
    END FUNCTION G


!************************ SUBROUTINE CONDUCTIVITY **********************
!
! slimmed down version that calculates only conductivity tensor
!
!***********************************************************************

    SUBROUTINE CONDUCTIVITY( WDES, W0, DER1, DER2, EFERMI, & 
      ISMEAR, SIGMA, OMEGA, WPLASMON, CON, TAU)
      USE wave
      USE constant
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      TYPE (wavespin)    W0
      REAL(q) :: DER1(:,:,:), DER2(:,:,:), EFERMI
      INTEGER ISMEAR       ! broadening method
      REAL(q) SIGMA        ! broadening parameter
      REAL(q) OMEGA        ! volume of unit cell
      REAL(q) WPLASMON, CON
      REAL(q) TAU          ! relaxation time
    !local
      INTEGER ISP, NK, N1
      REAL(q) WEIGHT, DECEL
      REAL(q) A, ENERGY, DFUN, SFUN

      WPLASMON=0
      CON=0

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
#ifdef MPI
         IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
         ! N1 could be restricted to occupied bands
         DO N1=1,WDES%NB_TOT
         ! N1 must be smaller tan NBANDS_CDER
         IF (N1>NBANDS_CDER) CYCLE

         ENERGY=W0%CELTOT(N1,NK,ISP)-EFERMI
         IF (ISMEAR==-1) THEN
            SFUN=F(-ENERGY,ABS(SIGMA))
            DFUN=G(ENERGY,ABS(SIGMA))
         ELSE
            CALL DELSTP(ISMEAR, ENERGY/ABS(SIGMA), DFUN, SFUN)
            DFUN=DFUN/ABS(SIGMA)
         ENDIF
         ! DFUN d f / d epsilon (= delta)
         ! conversion factor 4 pi e^2/ volume to obtain polarization
         WPLASMON=WPLASMON+DFUN*DER1(N1,NK,ISP)*DER2(N1,NK,ISP)*WDES%RSPIN*WDES%WTKPT(NK)*(4*PI*FELECT/OMEGA)
         CON=CON          +DFUN*DER1(N1,NK,ISP)*DER2(N1,NK,ISP)*WDES%RSPIN*WDES%WTKPT(NK)*(4*PI*FELECT/OMEGA)*ABS(TAU)*CONTCON

         ENDDO
      ENDDO
      ENDDO

    END SUBROUTINE CONDUCTIVITY


!************************ SUBROUTINE CONDUCTIVITY_ENERGY_RESOLVED ******
!
! calculate the energy resolve conductivity tensor
!
!***********************************************************************

    SUBROUTINE CONDUCTIVITY_ENERGY_RESOLVED( WDES, W0, ENERGY_DER,  EFERMI, & 
         NEDOS, EMAX_COND, COND_ENERGY, NELECT, NUP_DOWN, KPOINTS, OMEGA, TAU)
      USE wave
      USE constant
      USE mkpoints
      IMPLICIT NONE
      TYPE (wavedes)     WDES
      TYPE (wavespin)    W0
      REAL(q) :: ENERGY_DER(:,:,:,:) 
      REAL(q) :: EFERMI    ! position of Fermi level
      INTEGER :: NEDOS
      REAL(q) :: EMAX_COND
      REAL(q) :: COND_ENERGY(:,:,:,:)
      REAL(q) :: NELECT, NUP_DOWN
      TYPE (kpoints_struct) :: KPOINTS
      REAL(q) :: OMEGA     ! volume of unit cell
      REAL(q) TAU          ! relaxation time
    !local
      INTEGER ISP, NK, N1
      REAL(q) WEIGHT, DECEL
      REAL(q) A, ENERGY, DFUN, SFUN
      TYPE (kpoints_struct) :: KPOINTS_FOR_COND
      REAL(q),ALLOCATABLE :: SIGMA(:,:,:,:,:), DOS(:,:), DOSI(:,:)
      REAL(q) :: ENTROPY
      REAL(q) :: EFERMI_LOCAL
      INTEGER :: IDIR, JDIR


      ALLOCATE(DOS(NEDOS, WDES%NCDIJ), DOSI(NEDOS, WDES%NCDIJ))
      ALLOCATE(SIGMA(WDES%NB_TOT, WDES%NKPTS, 3,3, WDES%NCDIJ))

      KPOINTS_FOR_COND=KPOINTS
      KPOINTS_FOR_COND%EMAX=EFERMI+EMAX_COND
      KPOINTS_FOR_COND%EMIN=EFERMI-EMAX_COND

      DO ISP=1,WDES%ISPIN
      DO NK=1,WDES%NKPTS
         ! N1 could be restricted to occupied bands
         DO N1=1,WDES%NB_TOT
            ! N1 must be smaller tan NBANDS_CDER
            IF (N1>NBANDS_CDER) CYCLE
            ! TODO adapt to spin orbit coupled case
            DO IDIR=1,3
               DO JDIR=1,3
                  SIGMA(N1,NK,IDIR,JDIR,ISP)=ENERGY_DER(N1,NK,ISP,IDIR)*ENERGY_DER(N1,NK,ISP,JDIR)*(4*PI*FELECT/OMEGA)*ABS(TAU)*CONTCON
               ENDDO
            ENDDO
         ENDDO
      ENDDO
      ENDDO

      EFERMI_LOCAL=EFERMI
      CALL  DENSTA( -1, -1, WDES, W0, KPOINTS_FOR_COND, NELECT, &
           NUP_DOWN, ENTROPY, EFERMI_LOCAL, KPOINTS_FOR_COND%SIGMA, .TRUE.,  &
              NEDOS, 3, 3, DOS, DOSI, SIGMA, COND_ENERGY)

      DEALLOCATE(SIGMA)
      DEALLOCATE(DOS, DOSI)

    END SUBROUTINE CONDUCTIVITY_ENERGY_RESOLVED


!************************ SUBROUTINE WRT_CDER_BETWEEN_STATES_FORMATTED *
!
! this subroutine writes CDER_BETWEEN_STATES to a formatted file
!
!***********************************************************************

    SUBROUTINE WRT_CDER_BETWEEN_STATES_FORMATTED(WDES,W,IU0,IU)
      USE wave
      USE main_mpi
      IMPLICIT NONE
      TYPE (wavedes) WDES
      TYPE (wavespin) W
      INTEGER IU0,IU
      ! local variables
      INTEGER ISP,NK,NB1,NB2
      COMPLEX(qs) CDUM(3)

      ! quick return if possible
      IF (.NOT.ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      IF (IU0>=0) THEN
         ! open file
         OPEN(IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDERF',STATUS='UNKNOWN')
         ! write header information
         WRITE(IU,'(3I6)') WDES%ISPIN,WDES%NKPTS,NBANDS_CDER
         ! write CDER_BETWEEN_STATES
         DO ISP=1,WDES%ISPIN
         DO NK=1,WDES%NKPTS
            DO NB1=1,NBANDS_CDER
            DO NB2=1,NBANDS_CDER
#ifdef gammareal
               CDUM(1:3)=CMPLX(CDER_BETWEEN_STATES(NB1,NB2,NK,ISP,1:3),0._qs,KIND=qs)
#else
               CDUM(1:3)=CDER_BETWEEN_STATES(NB1,NB2,NK,ISP,1:3)
#endif
               WRITE(IU,'(I6,2F14.7,I6,2F14.7,3(2F20.10))') &
              &   NB1,REAL(W%CELTOT(NB1,NK,ISP),KIND=q),W%FERTOT(NB1,NK,ISP), &
              &   NB2,REAL(W%CELTOT(NB2,NK,ISP),KIND=q),W%FERTOT(NB2,NK,ISP), &
              &   CDUM(1:3)
            ENDDO
            ENDDO
         ENDDO
         ENDDO
         ! close
         CLOSE(IU)
      ENDIF
      RETURN
    END SUBROUTINE WRT_CDER_BETWEEN_STATES_FORMATTED



!***********************************************************************
!
!  rotate the matrix elements
!  CHAM(n1,n2) = <u_n1|  - i d/dk_i u_n2> = <u_n1|- r_i |u_n2>
!  from the IRZ to an arbitrary point NK in the full BZ
!  and return  <u_n1|  - i d/dk_i u_n2> at the desired
!  k-point
!
!***********************************************************************

    SUBROUTINE CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN)
      USE lattice
      USE full_kpoints
      USE kpoints_change
      IMPLICIT NONE
      INTEGER NK, ISP, N1_IN, N2_IN
      INTEGER IDIR
      GDEF CDER_BETWEEN_STATE(:)
      TYPE (latt)        LATT_CUR
    ! local
      INTEGER NK_OLD, ISP_OLD, L, K, J, I, IDIR_OLD, N1, N2
      REAL(q) :: S(3,3)
      LOGICAL :: LINV

      CDER_BETWEEN_STATE=0
    !
    ! somewhat dirty, if CDER_BETWEEN_STATES_Q is allocated call 
      IF (ALLOCATED(CDER_BETWEEN_STATES_Q)) THEN
         CALL CDER_BETWEEN_STATES_ROTATED_Q(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN)
         RETURN
      ENDIF

      IF (.NOT. ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      NK_OLD=KPOINTS_FULL_ORIG%NEQUIV(NK)
      ISP_OLD=ISP
      IF (KPOINTS_FULL_ORIG%SPINFLIP(NK)==1) THEN
         ISP_OLD=3-ISP
      ENDIF
      LINV=  KPOINTS_FULL_ORIG%LINV(NK)

      IF (N1_IN<=N2_IN) THEN
         N1=N2_IN
         N2=N1_IN
         LINV=.NOT. LINV
      ELSE
         N1=N1_IN
         N2=N2_IN
      ENDIF

      IF (N2 > NBANDS_CDER) THEN
         WRITE(*,*) 'internal error in CDER_BETWEEN_STATES_ROTATED: optical matrix elements not calculated',NK, N1, N2, NBANDS_CDER
         STOP
      ENDIF

    ! determine transformation matrix in real space when going from
    ! old to new k-point
      S=0
      DO L=1,3
         DO K=1,3
            DO J=1,3
               DO I=1,3
                  S(L,I)=S(L,I)+LATT_CUR%A(L,K)*KPOINTS_FULL_ORIG%ISYMOP(J,K,NK)*LATT_CUR%B(I,J)
               ENDDO
            ENDDO
         ENDDO
      ENDDO
  ! a vector in direction IDIR_OLD will rotate into a vector
  !  S(:,IDIR_OLD)
      DO IDIR_OLD=1,3
         CDER_BETWEEN_STATE(:)=CDER_BETWEEN_STATE(:)+S(:,IDIR_OLD)*CDER_BETWEEN_STATES(N1,N2,NK_OLD,ISP_OLD,IDIR_OLD)
      ENDDO
      IF (LINV) THEN
         CDER_BETWEEN_STATE(:)=GCONJG(CDER_BETWEEN_STATE(:))
      ENDIF
!      WRITE(*,'(3F14.7)') CDER_BETWEEN_STATES(N1,N2,NK_OLD,ISP_OLD,:)
!      WRITE(*,'(3F14.7)') CDER_BETWEEN_STATE(:)
    END SUBROUTINE CDER_BETWEEN_STATES_ROTATED


    SUBROUTINE CDER_BETWEEN_STATES_ROTATED_NOERR(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN)
      USE lattice
      USE full_kpoints
      USE kpoints_change
      IMPLICIT NONE
      INTEGER NK, ISP, N1_IN, N2_IN
      INTEGER IDIR
      GDEF CDER_BETWEEN_STATE(:)
      TYPE (latt)        LATT_CUR
    ! local
      INTEGER NK_OLD, ISP_OLD, L, K, J, I, IDIR_OLD, N1, N2
      REAL(q) :: S(3,3)
      LOGICAL :: LINV

      CDER_BETWEEN_STATE=0
    !
    ! somewhat dirty, if CDER_BETWEEN_STATES_Q is allocated call 
      IF (ALLOCATED(CDER_BETWEEN_STATES_Q)) THEN
         CALL CDER_BETWEEN_STATES_ROTATED_Q(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN)
         RETURN
      ENDIF

      IF (.NOT. ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      NK_OLD=KPOINTS_FULL_ORIG%NEQUIV(NK)
      ISP_OLD=ISP
      IF (KPOINTS_FULL_ORIG%SPINFLIP(NK)==1) THEN
         ISP_OLD=3-ISP
      ENDIF
      LINV=  KPOINTS_FULL_ORIG%LINV(NK)

      IF (N1_IN<=N2_IN) THEN
         N1=N2_IN
         N2=N1_IN
         LINV=.NOT. LINV
      ELSE
         N1=N1_IN
         N2=N2_IN
      ENDIF

      IF (N2 > NBANDS_CDER .OR. N1 > SIZE(CDER_BETWEEN_STATES,1) ) THEN
         CDER_BETWEEN_STATE=0
         RETURN
      ENDIF

    ! determine transformation matrix in real space when going from
    ! old to new k-point
      S=0
      DO L=1,3
         DO K=1,3
            DO J=1,3
               DO I=1,3
                  S(L,I)=S(L,I)+LATT_CUR%A(L,K)*KPOINTS_FULL_ORIG%ISYMOP(J,K,NK)*LATT_CUR%B(I,J)
               ENDDO
            ENDDO
         ENDDO
      ENDDO
  ! a vector in direction IDIR_OLD will rotate into a vector
  !  S(:,IDIR_OLD)
      DO IDIR_OLD=1,3
         CDER_BETWEEN_STATE(:)=CDER_BETWEEN_STATE(:)+S(:,IDIR_OLD)*CDER_BETWEEN_STATES(N1,N2,NK_OLD,ISP_OLD,IDIR_OLD)
      ENDDO
      IF (LINV) THEN
         CDER_BETWEEN_STATE(:)=GCONJG(CDER_BETWEEN_STATE(:))
      ENDIF
!      WRITE(*,'(3F14.7)') CDER_BETWEEN_STATES(N1,N2,NK_OLD,ISP_OLD,:)
!      WRITE(*,'(3F14.7)') CDER_BETWEEN_STATE(:)
    END SUBROUTINE CDER_BETWEEN_STATES_ROTATED_NOERR



!***********************************************************************
!
!  store the transition matrix elements 
!  CDER_BETWEEN_STATES_Q( n2, n1, nk, ispin)
!  < c_k1+q,n2 | e^-i q r | v_k1,n1 > 
!
!
!***********************************************************************

!
! allocate CDER_BETWEEN_STATES_Q
!
    SUBROUTINE ALLOCATE_CDER_BETWEEN_STATES_Q( NB_TOT, NBANDS_CDER, NKPTS, ISPIN)
      INTEGER NB_TOT, NBANDS_CDER, NKPTS, ISPIN

      ALLOCATE(CDER_BETWEEN_STATES_Q( NB_TOT, NBANDS_CDER, NKPTS, ISPIN))
      CDER_BETWEEN_STATES_Q=0
      
    END SUBROUTINE ALLOCATE_CDER_BETWEEN_STATES_Q

!
! deallocate CDER_BETWEEN_STATES_Q
!
    SUBROUTINE DEALLOCATE_CDER_BETWEEN_STATES_Q

      IF (ALLOCATED(CDER_BETWEEN_STATES_Q)) THEN
         DEALLOCATE(CDER_BETWEEN_STATES_Q)
      ENDIF
      
    END SUBROUTINE DEALLOCATE_CDER_BETWEEN_STATES_Q

    SUBROUTINE SET_CDER_BETWEEN_STATES_Q(CDER_BETWEEN_STATE, NK, ISP, N1, N2)
      USE lattice
      USE full_kpoints
      USE kpoints_change
      IMPLICIT NONE
      INTEGER NK, ISP, N1, N2
      COMPLEX(q) CDER_BETWEEN_STATE
      TYPE (latt)        LATT_CUR
    ! local

      IF (.NOT. ALLOCATED(CDER_BETWEEN_STATES_Q)) RETURN

      IF (N1 > SIZE(CDER_BETWEEN_STATES_Q,1)) THEN
         WRITE(0,*) 'internal warning: SET_CDER_BETWEEN_STATES matrix elements skipped N1', N1, SIZE(CDER_BETWEEN_STATES_Q,1)
         RETURN
      ENDIF

      IF (N2 > SIZE(CDER_BETWEEN_STATES_Q,2)) THEN
         WRITE(0,*) 'internal warning: SET_CDER_BETWEEN_STATES matrix elements skipped N2', N2, SIZE(CDER_BETWEEN_STATES_Q,2)
         RETURN
      ENDIF

      IF (NK > SIZE(CDER_BETWEEN_STATES_Q,3)) THEN
         WRITE(0,*) 'internal warning: SET_CDER_BETWEEN_STATES matrix elements skipped NK', NK, SIZE(CDER_BETWEEN_STATES_Q,3)
         RETURN
      ENDIF

      CDER_BETWEEN_STATES_Q(N1, N2, NK, ISP)=CDER_BETWEEN_STATE

    END SUBROUTINE SET_CDER_BETWEEN_STATES_Q

!***********************************************************************
!
!  retrieve the matrix elements
!    <u_nk+q| e^(i qr)   |u_mk>
!  from the IRZ to an arbitrary point NK in the wull BZ
!
!***********************************************************************

    SUBROUTINE CDER_BETWEEN_STATES_ROTATED_Q(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN)
      USE lattice
      USE full_kpoints
      USE kpoints_change
      IMPLICIT NONE
      INTEGER NK, ISP, N1_IN, N2_IN
      GDEF CDER_BETWEEN_STATE(:)
      TYPE (latt)        LATT_CUR
    ! local
      INTEGER NK_OLD, ISP_OLD, L, K, J, I, N1, N2
      LOGICAL :: LINV

      CDER_BETWEEN_STATE=0
      IF (.NOT. ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      LINV=.FALSE.
      IF (N1_IN<=N2_IN) THEN
         N1=N2_IN
         N2=N1_IN
         LINV=.NOT. LINV
      ELSE
         N1=N1_IN
         N2=N2_IN
      ENDIF

      CDER_BETWEEN_STATE(:)=CDER_BETWEEN_STATES_Q(N1, N2, NK, ISP)
      IF (LINV) THEN
         CDER_BETWEEN_STATE(:)=GCONJG(CDER_BETWEEN_STATE(:))
      ELSE
         WRITE(*,*) 'not inverted'
         STOP
      ENDIF
    END SUBROUTINE CDER_BETWEEN_STATES_ROTATED_Q

!#ifdef VASP2WANNIER90

!***********************************************************************
! rotates CDER_BETWEEN_STATES and projects onto correlated subspace
! needed for CRPA 
!***********************************************************************

    SUBROUTINE CDER_CRPA(CDER_BETWEEN_STATE, LATT_CUR, NK, ISP, N1_IN, N2_IN, CTRANS, NK1)
      USE lattice
      USE full_kpoints
      USE kpoints_change
      IMPLICIT NONE
      INTEGER NK, ISP, N1_IN, N2_IN, NK1
      INTEGER IDIR
      GDEF CDER_BETWEEN_STATE(:)
      TYPE (latt)        LATT_CUR
      GDEF :: CTRANS(:,:,:,:)
    ! local
      INTEGER NK_OLD, ISP_OLD, L, K, J, I, IDIR_OLD, N1, N2
      REAL(q) :: S(3,3)
      LOGICAL :: LINV, LINV_K
      INTEGER, SAVE :: NK_LAST=-1
      GDEFS, ALLOCATABLE ::  C(:,:),C0(:,:),CTRANS_single(:,:)
      GDEFS, ALLOCATABLE, SAVE :: CTMP(:,:,:)
      INTEGER NB_TOT,NB

      CDER_BETWEEN_STATE=0
      IF (.NOT. ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      IF ( SIZE(CDER_BETWEEN_STATES,1) /= SIZE(CTRANS,1) ) THEN
         WRITE(*,*)'ERROR in CDER_CRPA: # of bands dont match',SIZE(CDER_BETWEEN_STATES,1),SIZE(CTRANS,1) 
         STOP
      ENDIF
      IF ( SIZE(CDER_BETWEEN_STATES,2) /= NBANDS_CDER) THEN
         WRITE(*,*)'ERROR in CDER_CRPA: # of NBANDS_CDER does not match',SIZE(CDER_BETWEEN_STATES,2),NBANDS_CDER 
         STOP
      ENDIF
      IF ( SIZE(CDER_BETWEEN_STATES,4) /= SIZE(CTRANS,4) ) THEN
         WRITE(*,*)'ERROR in CDER_CRPA: # of spins dont match',SIZE(CDER_BETWEEN_STATES,4),SIZE(CTRANS,4) 
         STOP
      ENDIF

    setup: IF (NK_LAST/=NK) THEN

      NK_LAST=NK
      NB_TOT=SIZE(CDER_BETWEEN_STATES,1)

      ALLOCATE(CTRANS_single(NB_TOT,NB_TOT)) 
      CTRANS_single(:,:)=REAL(CTRANS(:,:,NK1,ISP),KIND=qs)
      
      !NK_OLD is the k-point in the IBZ
      NK_OLD=KPOINTS_FULL_ORIG%NEQUIV(NK)
      ISP_OLD=ISP
      IF (KPOINTS_FULL_ORIG%SPINFLIP(NK)==1) THEN
         ISP_OLD=3-ISP
      ENDIF
      LINV_K=  KPOINTS_FULL_ORIG%LINV(NK)


    ! determine transformation matrix in Cartesian space when going from
    ! old to new k-point
      S=0
      DO L=1,3
         DO K=1,3
            DO J=1,3
               DO I=1,3
                  S(L,I)=S(L,I)+LATT_CUR%A(L,K)*KPOINTS_FULL_ORIG%ISYMOP(J,K,NK)*LATT_CUR%B(I,J)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      
     !the transformation S only acts in Cartesian space on the vector r and can be left unchanged as it is.
     !this means that a vector in direction IDIR_OLD will rotate into a vector S(:,IDIR_OLD)

     !the matrix element 
     !\sum_{N'N''ISP'} CTRANS*(NK,N1_IN,N',ISP') <N', NK, ISP'| r | N'',NK,ISP''> CTRANS(NK,N'',N2_IN,ISP'')
     !is needed for NK in the full BZ 
     !this means we need to blow up CDER not only for N1_IN, N2_IN to the full BZ, but for 
     !the complete range 1:NB_TOT, 1:NB_TOT
     !hence we need an auxillary matrix CTMP(1:NBANDS,1:NBANDS,1:3)
     IF (.NOT. ALLOCATED(CTMP)) THEN
        ALLOCATE(CTMP(NB_TOT,NB_TOT,3))
     ENDIF

     !first we need to blow up CDER_BETWEEN_STATES to full BZ 
     !and store the result in the matrix CTMP(:,:,IDIR) for each direction IDIR = 1,2,3
     !for one matrix element I, J, this is done by
     DO I = 1, NB_TOT
        DO J = 1, NB_TOT 
           !only the triangle of CDER_BETWEEN_STATES is saved (its symmetric wrt bands)
           IF (I<=J) THEN
              N1=J
              N2=I
              LINV=.NOT. LINV_K
           ELSE
              N1=I
              N2=J
              LINV=LINV_K
           ENDIF
           CTMP(I, J, :)=0

           !the matrix <n|r|n'> is hermitian, so that for matrix elements N2 > NBANDS_CDER 
           !the complex conjugated element of the transposed matrix can be used
           !note that only the "wings and head of width NBANDS_CDER" are non-zero  
           IF ( N2 > NBANDS_CDER ) THEN
              IF ( N1 <= NBANDS_CDER ) THEN 
                 DO IDIR = 1, 3
                    DO IDIR_OLD=1,3
                       CTMP(I, J, IDIR)=CTMP(I, J, IDIR)+S(IDIR,IDIR_OLD)*&
                                      CDER_BETWEEN_STATES(N2,N1,NK_OLD,ISP_OLD,IDIR_OLD)
                    ENDDO
                 ENDDO
              ENDIF
           ELSE 
              DO IDIR = 1, 3
                 DO IDIR_OLD=1,3
                    CTMP(I, J, IDIR)=CTMP(I, J, IDIR)+S(IDIR,IDIR_OLD)*&
                                   GCONJG(CDER_BETWEEN_STATES(N1,N2,NK_OLD,ISP_OLD,IDIR_OLD) )
                 ENDDO
              ENDDO
           ENDIF

           IF (LINV) THEN
              CTMP(I, J, : )=GCONJG( CTMP( I, J, : ))
           ENDIF
        ENDDO
     ENDDO

     !at this stage CTMP contains the matix elements at NK, ISP for Bloch states
     ALLOCATE(C(NB_TOT,NB_TOT))
    C = 0 

     !now comes the rotation of CTMP to the correlated sub space
     !care must be taken, which indices to choose for the transformation. 
     !the matrix element is needed at NK, N1_IN, N2_IN and ISP, for IDIR = 1,2,3. thus
     !CTMP <- \sum_{N', N"} CTRANS(N1_IN,N',NK,ISP)CTMP(N',N",:)CTRANS(N2_IN,N",:)

     DO IDIR = 1, 3
        C(:,:) = 0
#ifdef gammareal
        CALL SGEMM('N', 'N', NB_TOT, NB_TOT, NB_TOT, 1._qs, &
             CTMP(1,1,IDIR), NB_TOT, CTRANS_single(1,1), &
             NB_TOT, 0._qs, C(1,1), NB_TOT)
        CALL SGEMM('T', 'N', NB_TOT, NB_TOT, NB_TOT, 1._qs, &
             CTRANS_single(1,1), NB_TOT, C(1,1), &
             NB_TOT, 0._qs, CTMP(1,1,IDIR), NB_TOT)
#else
        CALL CGEMM('N', 'N', NB_TOT, NB_TOT, NB_TOT, (1._qs,0._qs), &
             CTMP(1,1,IDIR), NB_TOT, CTRANS_single(1,1), &
             NB_TOT, (0._qs,0._qs), C(1,1), NB_TOT)
        CALL CGEMM('C', 'N', NB_TOT, NB_TOT, NB_TOT, (1._qs,0._qs), &
             CTRANS_single(1,1), NB_TOT, C(1,1), &
             NB_TOT, (0._qs,0._qs), CTMP(1,1,IDIR), NB_TOT)
#endif
      ENDDO
      DEALLOCATE(C)
      DEALLOCATE (CTRANS_single)

  ENDIF setup
    
      !it turs out that the complex conjugate is the correct element
      CDER_BETWEEN_STATE(:) = GCONJG(CTMP(N1_IN,N2_IN,:))
      RETURN
    END SUBROUTINE CDER_CRPA

!#endif

!***********************************************************************
!
! add a value VALUE at the energy EPS to a density of states array DOS
! ISMEAR/SIGMA  specify the broadening
! NEDOS  number of slots
! DELTAE energy step between slots
!
!***********************************************************************

    SUBROUTINE SLOT(EPS, ISMEAR, SIGMA, NEDOS, DELTAE, VALUE, DOS)
      USE prec
      IMPLICIT NONE
      REAL(q) EPS    ! energy
      INTEGER ISMEAR ! type of smearing
      REAL(q) SIGMA  ! width
      INTEGER NEDOS  ! number of slots
      REAL(q) DELTAE ! slot distance
      COMPLEX(q) VALUE  ! magnitude of value to add to DOS
      COMPLEX(q) DOS(:)
    ! local
      INTEGER NELOW
      INTEGER NEHIG
      REAL(q) E, DFUN, SFUN, SFUN_DONE, EPSDOS
      INTEGER I

      NELOW=(EPS-8._q*SIGMA)/DELTAE+1
      ! use a least two points
      NEHIG=(EPS+8._q*SIGMA)/DELTAE+2
      IF (NELOW<0)     NELOW=0
      IF (NELOW>NEDOS) NELOW=NEDOS
      IF (NEHIG<1)     NEHIG=1
      IF (NEHIG>NEDOS) NEHIG=NEDOS
            
      SFUN_DONE=0
      DO I=NELOW,NEHIG
         E=DELTAE*(I-1)-EPS
!     evaluate the value of derivative in the centre of slot
!         CALL DELSTP(ISMEAR,(E/SIGMA),DFUN,SFUN)
!         EPSDOS=DFUN/SIGMA
!     more accurate take the value at the boundaries of the slots
!     and perform a numerical differentiation
         CALL DELSTP(ISMEAR,((E+DELTAE/2)/SIGMA),DFUN,SFUN)
         EPSDOS=(SFUN-SFUN_DONE)/DELTAE
         SFUN_DONE=SFUN
         IF (I>=1) THEN
            DOS(I) =DOS(I) +(VALUE*EPSDOS)
         ENDIF
      ENDDO
    END SUBROUTINE SLOT

!************************ SUBROUTINE EPSILON_IMAG_TET ******************
!
! this subroutine calculates the imaginary part of the dielectric function
! for a specific direction
! as argument it requires the matrix of inproducts between
! all bands and the derivative of the bands  with respect 
! to a specific cartesian component of k
!  CHAM(n,m) = <u_n| - i d/dk_i  |u_m> = <u_n|- r_i |u_m>
! 
! routine for the case of ISMEAR = -4, -5 
! uses subroutines taken from optics by Furtmuller
!
!***********************************************************************

    SUBROUTINE EPSILON_IMAG_TET( WDES, W0, CHAM1, CHAM2, EMAX, NEDOS, DOS, DELTAE, &
                   OMEGA, IO, INFO, KPOINTS)
      USE wave
      USE constant
      USE base
      USE kpoints_change
      IMPLICIT NONE
      TYPE (wavedes)        WDES
      TYPE (wavespin)       W0
      TYPE (kpoints_struct) KPOINTS

      GDEFS :: CHAM1(:,:,:,:)
      GDEFS :: CHAM2(:,:,:,:)

      COMPLEX(q) DOS(:)

      REAL(q) OMEGA, DELTAE
      INTEGER NEDOS
    !local
      INTEGER ISP, NK, N1, N2, NC, N, NTRANS, NKPTS, NBCON, NBVAL, NBFULL, NBTOT, ISPIN, J
      REAL(q) EMIN, EMAX, EFERMI, SUMWEI, SUME, CONST
      TYPE (in_struct)      IO
      TYPE (info_struct)    INFO
      REAL(q), ALLOCATABLE :: AMP(:,:,:),ETRANS(:,:,:),DOST(:,:),DOSI(:,:), &
                  FERWE(:,:,:),PAR(:,:,:,:,:),DOSPAR(:,:,:,:),WTKPT(:)


      DOS    = 0._q

      NBTOT  = WDES%NB_TOT

      DO ISP = 1, WDES%ISPIN

         NBVAL  = LAST_FILLED_OPTICS_NO_MOD(W0,ISP,0.01_q)
         NBFULL = FIRST_EMPTY_OPTICS_NO_MOD(W0,ISP,0.99_q)-1
! test_
!        WRITE(*,*) 'NBVAL=',NBVAL,' NBFULL=',NBFULL
! test_

         NBCON  = NBTOT-NBFULL
         NTRANS = NBCON * NBVAL

         NKPTS  = KPOINTS%NKPTS
         CONST  = RYTOEV * 2 * AUTOA * TPI * PI / OMEGA * WDES%RSPIN
         EMIN   = 0._q

         ALLOCATE( AMP(NTRANS,NKPTS,1), ETRANS(NTRANS,NKPTS,1) )
         ALLOCATE( DOST(NEDOS,1), DOSI(NEDOS,1))
         ALLOCATE( FERWE(NTRANS,NKPTS,1), WTKPT(NKPTS) )
         ALLOCATE( PAR(NTRANS,NKPTS,1,1,1), DOSPAR(NEDOS,1,1,1) )
 
         DOSI   = 0._q
         DOST   = 0._q
         AMP    = 0._q
         ETRANS = 0._q
         PAR    = 0._q
         DOSPAR = 0._q

         DO NK = 1, NKPTS
            DO N1 = 1, NBVAL
               DO N2 = 1, NBCON
                  J = (N1 - 1) * NBCON + N2
                  NC = NBFULL + N2
                  IF (NC>N1) &
                  AMP(J,NK,1) = GCONJG(CHAM1(NC,N1,NK,ISP)) * CHAM2(NC,N1,NK,ISP) * CONST * &
                                (W0%FERTOT(N1,NK,ISP)-W0%FERTOT(NC,NK,ISP))
                  ETRANS(J,NK,1) = W0%CELTOT(NC,NK,ISP) - W0%CELTOT(N1,NK,ISP)
               ENDDO
            ENDDO
         ENDDO
       
         CALL BZINTS_WEIGHT(0,FERWE,ETRANS,AMP,WTKPT,NTRANS,NTRANS, &
                        NKPTS,KPOINTS%IDTET,KPOINTS%NTET,1,KPOINTS%VOLWGT,EMIN,EMAX, &
                        DOST(1,1),DOSI(1,1),NEDOS,EFERMI, &
                        SUMWEI,SUME,IO%IU6,PAR,DOSPAR,NKPTS,1,1,1,0)

     
         DOS(:) = DOS(:) + DOST(:,1)   

         DEALLOCATE(AMP, ETRANS, DOST, DOSI, FERWE, WTKPT, PAR, DOSPAR )
      ENDDO
 

    END SUBROUTINE EPSILON_IMAG_TET

!************************ SUBROUTINE EPSILON_REAL **********************
!
! this subroutine calculates the real part of the dielectric function
! given the imaginary part
! a Hann or Welch window function can be applied to remove
! artefacts of the finite energy interval (but this is not
! encouraged)
! Kramers-Kronig is used to perform the transformation
! with a small positive imaginary shift (SIGMA)
!
!  2/pi P int_0^infty w' eps_imag(w') / (w'^2-w^2) d w'
!  2/pi (-1/2) P int_0^infty (eps_imag(w')/(w-w') - eps_imag^*(w')/(w+w'))
!
! implemented windows functions:
!  0 no window
!  1 Welch window 
!  2 Hann window
! presently the routine uses the simplest conceivable integration
! formula
!
!***********************************************************************

    SUBROUTINE EPSILON_REAL( COMM, NEDOS, EPSDD, LWARN, DELTAE, CSHIFT, IWINDOW, WPLASMA_INTER )
      USE constant
      IMPLICIT NONE
      TYPE (communic) :: COMM
      INTEGER NEDOS
      COMPLEX(q) EPSDD(:)
      REAL(q) DELTAE, CSHIFT
      INTEGER IWINDOW
      COMPLEX(q), OPTIONAL :: WPLASMA_INTER
    ! local
      COMPLEX(q) EPS_REAL(SIZE(EPSDD))
      INTEGER I_I, I_R
      REAL(q) W_R, W_I, ARG
      COMPLEX(q) SUM, VAL
      LOGICAL LWARN             ! return .TRUE. if the input data are complex
      LOGICAL LCOPY

      ! calculate the "interband" contribution to the squared plasma frequency
      IF (PRESENT(WPLASMA_INTER)) THEN
         SUM=0
         DO I_I=1,NEDOS
            W_I=DELTAE*(I_I-1)
            VAL=EPSDD(I_I)
            SUM=SUM+VAL*W_I
         ENDDO
         SUM=SUM*2/PI*DELTAE
         WPLASMA_INTER=SUM
      ENDIF

      ! check whether input data (imaginary part of dielectric function)
      ! contains any imaginary contributions
      ! if so we can not simply overwrite the imaginary part by the originally
      ! supplied function
      LCOPY=.TRUE.
      DO I_R=1,NEDOS
         IF (ABS(AIMAG(EPSDD(I_R)))>1E-2) THEN
            LCOPY=.FALSE.
         ENDIF
      ENDDO

      EPS_REAL=0
      ! loop over w=W_R
#ifdef MPI
      DO I_R=COMM%NODE_ME,NEDOS,COMM%NCPU
#else
      DO I_R=1,NEDOS
#endif
         W_R=DELTAE*(I_R-1)
         SUM=0
         ! loop over w'=W_I
         DO I_I=1,NEDOS
            W_I=DELTAE*(I_I-1)
            ! this is the more traditional KG version: complex shift compatible to calculation of Xi in GW routines
            ! VAL=EPSDD(I_I)*(W_I+CMPLX(0,CSHIFT,q))/((W_I+CMPLX(0,CSHIFT,q))**2-W_R**2)
            !
            ! time ordered version, poles above real axis for positive frequencies
            ! and below real axis for negative frequencies Imag xi(-w) = Imag xi(w)
            ! can cause  finite values for the imaginary part for finite shifts CSHIFT
            ! VAL=EPSDD(I_I)*(1/(W_R-W_I-CMPLX(0,CSHIFT,q))-1/(W_R+W_I+CMPLX(0,CSHIFT,q)))*(-0.5)

            ! gK: updated 27.12.2016 poles from above real axis to below real axis (more common in literature)
            ! causal, all poles below real axis Imag xi(-w) = -Imag xi(w) and (Imag xi(0)=0)
            ! in the GW routines poles are currenlty above the real axis
            ! the choice of the imaginary shift is such that 
            ! Imag  2/pi int_0^infty w' eps_imag(w') / (w'^2-w^2) d w' = eps_imag(w')
#ifdef timeordered
            VAL=(EPSDD(I_I)/(W_R-W_I+CMPLX(0,CSHIFT,q))-CONJG(EPSDD(I_I))/(W_R+W_I-CMPLX(0,CSHIFT,q)))*(-0.5)
#else
            VAL=(EPSDD(I_I)/(W_R-W_I+CMPLX(0,CSHIFT,q))-CONJG(EPSDD(I_I))/(W_R+W_I+CMPLX(0,CSHIFT,q)))*(-0.5)
#endif

            ARG=REAL(I_I-1,q)/(NEDOS-1)
            SELECT CASE (IWINDOW)
               CASE (1)
                  VAL=VAL*(1-ARG**2)
               CASE (2)
                  VAL=VAL*0.5_q*(COS(PI*ARG)+1)
               CASE DEFAULT
            END SELECT
            SUM=SUM+VAL
         ENDDO
         SUM=SUM*2/PI*DELTAE
         EPS_REAL(I_R)=SUM
      ENDDO

      CALLMPI( M_sum_z(COMM, EPS_REAL, SIZE(EPS_REAL)))

! this line overwrites the imaginary part by the original data
! passed to the subroutine
! the option is usually selected if the complex shift is small,
! since in this case the KK transformation above yields inaccurate results
! for the imaginary part
! however, this, option can not be used if the input data is already complex
! values
      IF (ABS(CSHIFT) < DELTAE .AND. .NOT. LCOPY) THEN
         LWARN=.TRUE.
      ENDIF
      IF (ABS(CSHIFT) < DELTAE .AND. LCOPY) THEN
         EPS_REAL=CMPLX(REAL(EPS_REAL,q),REAL(EPSDD,q))
      ENDIF

      EPSDD=EPS_REAL

    END SUBROUTINE EPSILON_REAL


!************************ SUBROUTINE FOCK_K_DER_ANALYT ****************
!
! calculate the commutator of the [H_x,r]  phi_i
! where H_x is the exact exchange part of the Hamiltonian
! this is equivalent to the first derivative with respect to k multiplied
! by -i
!
!   [H_x,r] = -i d/d k H_x
! this is the analytical version of the previous routine
! 
!***********************************************************************


    SUBROUTINE FOCK_K_DER_ANALYT(KPOINTS, GRID, LATT_CUR, LATT_INI, T_INFO,  & 
            NONLR_S, NONL_S, W, W1, &
            LMDIM, P, CQIJ, SYMM, IDIR, LDONE, IU0, IU6)
      USE prec
      USE wave_high
      USE lattice
      USE poscar
      USE mpimy
      USE mgrid
      USE nonl_high
      USE base
      USE pseudo
      USE kpoints_change
      USE fock
      USE constant
      USE choleski
      IMPLICIT NONE

      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (latt)        LATT_INI
      TYPE (type_info)   T_INFO
      TYPE (nonlr_struct)NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (potcar)      P(:)
      TYPE (wavespin), TARGET :: W
      TYPE (wavespin)    W1
      INTEGER LMDIM
      OVERLAP            CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      TYPE (symmetry)    SYMM
      LOGICAL LDONE                  ! flag indicating whether HF exchange has been determined
      INTEGER IU0, IU6
      INTEGER IDIR
!    local
      TYPE (nonlr_struct),TARGET :: NONLR_D! projector times r (only CRREXP distinct)
      TYPE (nonl_struct)  NONL_D           ! k derivative of projector in real space
      INTEGER NKPTS_ORIG, I, ISP, NK, NSTRIP, NSTRIP_ACT, NPOS
      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      LOGICAL, EXTERNAL :: USEFOCK_CONTRIBUTION
      INTEGER :: ISPINOR, M, MM
   INTERFACE
     SUBROUTINE FOCK_QDER(GRID, LMDIM, LATT_CUR, W, &
          NONLR_S, NONLR_D, NONL_S, NONL_D, IDIR, NK, ISP, NPOS, NSTRIPN, &
          CH, P, CQIJ)

       USE mgrid
       USE lattice
       USE pseudo
       USE wave
       USE nonl_high
       IMPLICIT NONE

       ! passed variables
       TYPE (grid_3d) GRID
       INTEGER LMDIM
       TYPE (latt) LATT_CUR
       TYPE (wavespin) W
       TYPE (nonlr_struct) NONLR_S, NONLR_D
       TYPE (nonl_struct) NONL_S, NONL_D
       TYPE (potcar)      P(:)
       OVERLAP              CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
       INTEGER IDIR, NK, ISP, NPOS, NSTRIPN
       COMPLEX(q)       :: CH
     END SUBROUTINE FOCK_QDER
  END INTERFACE

      IF (.NOT. USEFOCK_CONTRIBUTION()) THEN
         LDONE=.FALSE.
         RETURN
      ENDIF

      IF (NONLR_S%LREAL) THEN
         CALL VTUTOR('E','GW-HF optics', &
         &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
         STOP
      ENDIF

      LDONE=.TRUE.

      IF (NONLR_S%LREAL) THEN
!        NONLR_D=NONLR_S
         CALL NONLR_ASSIGN(NONLR_D,NONLR_S)
         CALL NONLR_ALLOC_CRREXP(NONLR_D)
         CALL RSPHER(GRID,NONLR_S,LATT_CUR)
      ELSE
         CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
         CALL NONL_ALLOC_DER(NONL_S, NONL_D)
         CALL SPHER_DER(GRID,NONL_D, P, W%WDES, LATT_CUR, IDIR)
      ENDIF

      W1%CW=0

      NSTRIP=MIN(NSTRIP_STANDARD,W1%WDES%NBANDS)

      DO ISP=1,W%WDES%ISPIN
         DO NK=1,W%WDES%NKPTS
#ifdef MPI
            IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
            CALL SETWDES(W%WDES,WDES1,NK)
            IF (NONLR_S%LREAL) THEN
               CALL PHASER(GRID ,LATT_CUR,NONLR_S,NK,W%WDES)
               CALL PHASERR(GRID,LATT_CUR,NONLR_D,NK,W%WDES,IDIR)
            ELSE
               CALL PHASE(W%WDES, NONL_S, NK)
               NONL_D%NK=NONL_S%NK        ! uses the same phasefactor array
            ENDIF
               
            DO NPOS=1,W1%WDES%NBANDS,NSTRIP
               NSTRIP_ACT=MIN(W1%WDES%NBANDS+1-NPOS,NSTRIP)
               IF (AEXX/=0) THEN
                  CALL FOCK_QDER(GRID, LMDIM, LATT_CUR, W,  &
                       NONLR_S, NONLR_D, NONL_S, NONL_D, IDIR, NK, ISP, NPOS, NSTRIP_ACT, &
                       W1%CW(1,NPOS,NK,ISP), P, CQIJ )
               ENDIF
            ENDDO
         ENDDO
      ENDDO

      IF (NONLR_S%LREAL) THEN
         CALL  NONLR_DEALLOC_CRREXP(NONLR_D)
      ELSE
         CALL  NONL_DEALLOC_DER(NONL_D)
      ENDIF

    END SUBROUTINE FOCK_K_DER_ANALYT

!************************ SUBROUTINE FOCK_K_DER ***********************
!
! calculate the commutator of the [H_x,r]  phi_i
! where H_x is the exact exchange part of the Hamiltonian
! this is equivalent to the first derivative with respect to k multiplied
! by -i
!
!   [H_x,r] = -i d/d k H_x
!
! this version uses finite differences, and is scheduled for removal
! instead the previous version relying on an analytical derivative
! should be used
! the finite difference version fails for the Gamma-point only
! 
!***********************************************************************

    SUBROUTINE FOCK_K_DER(KPOINTS, GRID, LATT_CUR, LATT_INI, T_INFO,  NONLR_S, NONL_S, W, W1, &
       &    LMDIM, P, CQIJ, SYMM, IDIR, LDONE, IU0, IU6)
      USE prec
      USE wave_high
      USE lattice
      USE poscar
      USE mpimy
      USE mgrid
      USE nonl_high
      USE base
      USE pseudo
      USE kpoints_change
      USE fock
      USE constant
      USE choleski
      IMPLICIT NONE

      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (latt)        LATT_INI
      TYPE (type_info)   T_INFO
      TYPE (nonlr_struct)NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (potcar)      P(:)
      TYPE (wavespin), TARGET :: W
      TYPE (wavespin)    W1
      INTEGER LMDIM
      OVERLAP            CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
      TYPE (symmetry)    SYMM
      LOGICAL LDONE                  ! flag indicating whether HF exchange has been determined
      INTEGER IU0, IU6
      INTEGER IDIR
!    local
      INTEGER NKPTS_ORIG, I, ISP, NK, NSTRIP, NSTRIP_ACT, NPOS, NB
      REAL(q) DISPL(3)
      COMPLEX(q) CDCHF
      TYPE (wavedes1)    WDES1          ! descriptor for one k-point
      TYPE (wavefuna)    WHAM
      LOGICAL, EXTERNAL :: USEFOCK_CONTRIBUTION
      INTEGER :: ISPINOR, M, MM
#ifdef MPI
      IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('FOCK_K_DER: KPAR>1 not tested (but implemented), sorry.') )
         STOP
      END IF
#endif
!-----------------------------------------------------------------------
! double k-points in new structure
!-----------------------------------------------------------------------
      IF (.NOT. USEFOCK_CONTRIBUTION()) THEN
         LDONE=.FALSE.
         RETURN
      ENDIF
      LDONE=.TRUE.

      ! original number of k-points
      NKPTS_ORIG=W%WDES%NKPTS
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM, IU6,-1, W%WDES%VKPT(:,1:NKPTS_ORIG))
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM, IU6,-1, W%WDES%VKPT(:,1:NKPTS_ORIG))
#endif
      CALL KPAR_SYNC_ALL(W%WDES,W)
      CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, IU6, IU0)
      CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)
      CALL SETWDES(W%WDES,WDES1,0)

      NSTRIP=((W%WDES%NSIM*2+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)
      CALL NEWWAVA(WHAM, WDES1, NSTRIP)

      ! shift k-point indices by 1/2 of minimum distance between k-points
      DISPL=0
      DISPL(IDIR)=SQRT(KPOINTS_FULL%VKPTMINDIST2)/40
      CALL KARDIR(1, DISPL(1), LATT_CUR%A)
      
      DO NK=1,NKPTS_ORIG
         W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)-DISPL
      ENDDO
!-----------------------------------------------------------------------
! calculate numerical derivative
!-----------------------------------------------------------------------
      W1%CW=0
      
      DO I=-1,1,2
         ! set convergence corrections to 0
         ! seems to be more reliable
         FSG_STORE(NKPTS_ORIG+1:W%WDES%NKPTS)=0

         ! since we have changed the k-points h^2/(G+k)^2 needs to be recalculated
         CALL SET_DATAKE(W%WDES, LATT_CUR%B)
         IF (NONLR_S%LREAL) THEN
            CALL RSPHER(GRID,NONLR_S,LATT_CUR)
         ELSE
            CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
         ENDIF
         CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W)

         CDCHF=0

         DO ISP=1,W%WDES%ISPIN
            DO NK=NKPTS_ORIG+1,W%WDES%NKPTS
#ifdef MPI
               IF (MOD(NK-1,W%WDES%COMM_KINTER%NCPU).NE.W%WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
               ! find first empty band for this k-point
               DO NB=W%WDES%NB_TOT, 1, -1
                  ! non empty band required
                  IF (ABS(W%FERTOT( NB, NK, ISP))>1E-5 ) EXIT
               ENDDO
               ! round to next larger value modulo W%WDES%NB_PAR
               NB=((NB+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)*W%WDES%NB_PAR

               IF (NONLR_S%LREAL) THEN
                  CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,W%WDES)
               ELSE
                  CALL PHASE(W%WDES,NONL_S,NK)
               ENDIF
               
               CALL SETWDES(W%WDES,WDES1,NK)
               DO NPOS=1,NB,NSTRIP
                  NSTRIP_ACT=MIN(NB+1-NPOS,NSTRIP)
                  IF (AEXX/=0) THEN
                     CALL FOCK_ACC(GRID, LMDIM, LATT_CUR, W,  &
                          NONLR_S, NONL_S, NK, ISP, NPOS, NSTRIP_ACT, &
                          WHAM%CW(:,:), P, CQIJ, CDCHF )
                     W1%CW(:,NPOS:NPOS+NSTRIP_ACT-1,NK-NKPTS_ORIG,ISP)= &
                          W1%CW(:,NPOS:NPOS+NSTRIP_ACT-1,NK-NKPTS_ORIG,ISP) &
                          +REAL(I,q)*WHAM%CW(:,1:NSTRIP_ACT)
                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         DO NK=1,NKPTS_ORIG
            W%WDES%VKPT(:,NKPTS_ORIG+NK)=W%WDES%VKPT(:,NK)+DISPL
         ENDDO
      ENDDO
      ! central differences factor two
      ! factor 2pi from the fact that 2pi is missing in our k vector definition
      ! [H_x,r] = -i d/d (2pi k) H_x
      W1%CW=W1%CW/(SQRT(KPOINTS_FULL%VKPTMINDIST2)/40)*(0.0_q,-1.0_q)/2/(2*PI)
      
      CALL DELWAVA(WHAM)
!-----------------------------------------------------------------------
! restore old data layout
!-----------------------------------------------------------------------
#ifdef oldsym
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#else
      CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
           SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
           T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#endif
      CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, -1, -1)
      CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)

    END SUBROUTINE FOCK_K_DER

!***********************************************************************
!  function LAST_FILLED_OPTICS returns the last filled band
!  modulo the number of bands done in parallel
!***********************************************************************

  FUNCTION LAST_FILLED_OPTICS( W)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_OPTICS
    INTEGER K1, ISP
  ! local
    INTEGER NB, NB_MAX

    NB_MAX=0
    DO ISP=1,W%WDES%ISPIN
       DO K1=1,W%WDES%NKPTS
          DO NB=W%WDES%NB_TOT, 1, -1
             IF (W%FERTOT( NB, K1, ISP)>0.05 ) EXIT
          ENDDO
          NB_MAX=MAX(NB,NB_MAX)
       ENDDO
    ENDDO

    ! round to next larger value modulo W%WDES%NB_PAR
    LAST_FILLED_OPTICS=((NB_MAX+W%WDES%NB_PAR-1)/W%WDES%NB_PAR)*W%WDES%NB_PAR
  END FUNCTION LAST_FILLED_OPTICS

!***********************************************************************
!  function LAST_FILLED_OPTICS returns the last filled band
!***********************************************************************

  FUNCTION LAST_FILLED_OPTICS_NO_MOD( W, ISPIN, THRESH)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_OPTICS_NO_MOD
    INTEGER, OPTIONAL :: ISPIN
    REAL(q), OPTIONAL :: THRESH
  ! local
    INTEGER ISP,K1,NB,NB_MAX
    REAL(q) THRSH

    THRSH=0.05_q
    IF (PRESENT(THRESH)) THRSH=THRESH

    NB_MAX=0
    DO ISP=1,W%WDES%ISPIN
       IF (PRESENT(ISPIN)) THEN
          IF (ISP/=ISPIN) CYCLE
       ENDIF
       DO K1=1,W%WDES%NKPTS
          DO NB=W%WDES%NB_TOT, 1, -1
             IF (W%FERTOT( NB, K1, ISP)>THRSH) EXIT
          ENDDO
          NB_MAX=MAX(NB,NB_MAX)
       ENDDO
    ENDDO

    ! round to next larger value modulo W%WDES%NB_PAR
    LAST_FILLED_OPTICS_NO_MOD=NB_MAX
  END FUNCTION LAST_FILLED_OPTICS_NO_MOD

!***********************************************************************
!  function FIRST_EMPTY_OPTICS returns the first (partially) empty band
!***********************************************************************

  FUNCTION FIRST_EMPTY_OPTICS_NO_MOD( W, ISPIN, THRESH)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_OPTICS_NO_MOD
    INTEGER, OPTIONAL :: ISPIN
    REAL(q), OPTIONAL :: THRESH
  ! local
    INTEGER ISP,K1,NB,NB_MIN
    REAL(q) THRSH

    THRSH=0.95_q
    IF (PRESENT(THRESH)) THRSH=THRESH

    NB_MIN=W%WDES%NB_TOT
    DO ISP=1,W%WDES%ISPIN
       IF (PRESENT(ISPIN)) THEN
          IF (ISP/=ISPIN) CYCLE
       ENDIF
       DO K1=1,W%WDES%NKPTS
          DO NB=1,W%WDES%NB_TOT
             IF (W%FERTOT( NB, K1, ISP)<THRSH) EXIT
          ENDDO
          NB_MIN=MIN(NB,NB_MIN)
       ENDDO
    ENDDO

    ! round to next larger value modulo W%WDES%NB_PAR
    FIRST_EMPTY_OPTICS_NO_MOD=NB_MIN
  END FUNCTION FIRST_EMPTY_OPTICS_NO_MOD

!************************ SUBROUTINE WRT_CDER_BETWEEN_STATES ***********
!
! this subroutine writes CDER_BETWEEN_STATES  to a file it is required
! for GW calculations
!
!***********************************************************************

    SUBROUTINE WRT_CDER_BETWEEN_STATES(WDES, IU0, IU)
      USE wave
      USE main_mpi
      TYPE (wavedes)     WDES
      INTEGER IU0
      INTEGER IU
      

      IF (IU0>=0) THEN
         OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDER', &
              FORM='UNFORMATTED',STATUS='UNKNOWN')
         
         WRITE(IU) WDES%NB_TOT, NBANDS_CDER, WDES%NKPTS, WDES%ISPIN
         WRITE(IU) NODES_IN_DIELECTRIC_FUNCTION
         WRITE(IU) WPLASMON
         WRITE(IU) CDER_BETWEEN_STATES
         CLOSE(IU)
      ENDIF
    END SUBROUTINE WRT_CDER_BETWEEN_STATES

!************************ SUBROUTINE WRT_CDER_BETWEEN_STATES ***********
!
! this subroutine read CDER_BETWEEN_STATES  from a file 
! required for GW calculations
!
!***********************************************************************


    SUBROUTINE READ_CDER_BETWEEN_STATES(WDES, IU0, IU, EXT)
      USE wave
      USE main_mpi
      USE mlrf_main

      TYPE (wavedes)     WDES
      INTEGER IU0
      INTEGER IU
      CHARACTER (LEN=3), OPTIONAL :: EXT
    ! local
      INTEGER NB_TOT, NKPTS, ISPIN, I, IERR, NKPTS_NON_ZERO
      GDEFS,ALLOCATABLE :: CDER_TMP(:,:,:,:,:)
      INTEGER NKPTS_COUNTER,ISP
      LOGICAL  LREAD

      IF (ALLOCATED(CDER_BETWEEN_STATES)) RETURN

      WPLASMON=0
      IF (SUM(ABS(WPLASMA))>0) THEN
         WPLASMON=WPLASMA
      ENDIF

      !allow for downsampling from denser k-grids
      LREAD=.TRUE.
      IF ( ALLOCATED(IKPT_SELECT))THEN
         NKPTS=0
         IF (IU0>=0) THEN
            IF (PRESENT(EXT)) THEN
               OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDER.'//EXT, &
               FORM='UNFORMATTED',STATUS='OLD', IOSTAT=IERR)
            ELSE
               OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDER', &
               FORM='UNFORMATTED',STATUS='OLD', IOSTAT=IERR)
            ENDIF
            IF (IERR==0) THEN 
              READ(IU, IOSTAT=IERR) NB_TOT, NBANDS_CDER, NKPTS, ISPIN
              IF ( ISPIN /= WDES%ISPIN .AND. IU0>=0 ) WRITE(IU0,'(A)')' Warning WAVEDER contains more spin cannels than current wavefunction'
            ENDIF
            CLOSE(IU)
         ENDIF
         CALLMPI(M_bcast_i(WDES%COMM,NKPTS,1)) 
         IF ( SIZE(IKPT_SELECT)/=NKPTS) THEN
!            IF(IU0>=0) WRITE(IU0,*)'ERROR in READ_CDER_BETWEEN_STATES:'//&
!               ' number of kpoint not compatible', SIZE(IKPT_SELECT),NKPTS
            LREAD=.FALSE.
         ENDIF 
         IF ( IU0>=0 ) WRITE(IU0,'(A,I4,A)',ADVANCE='NO')' WAVEDER contains more than',&
            WDES%NKPTS,' k-points, downsampling... '
         NKPTS=0
      ENDIF

      NKPTS_NON_ZERO=0
      DO NKPTS=1,WDES%NKPTS
         IF (WDES%WTKPT(NKPTS)==0) EXIT
         NKPTS_NON_ZERO=NKPTS_NON_ZERO+1
      ENDDO

      WPLASMON=0
      NODES_IN_DIELECTRIC_FUNCTION=0
      IERR=0
      NBANDS_CDER=0

      IF (IU0>=0) THEN
         IF (PRESENT(EXT)) THEN
            OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDER.'//EXT, &
            FORM='UNFORMATTED',STATUS='OLD', IOSTAT=IERR)
         ELSE
            OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'WAVEDER', &
            FORM='UNFORMATTED',STATUS='OLD', IOSTAT=IERR)
         ENDIF
         IF (IERR==0) THEN 
           READ(IU, IOSTAT=IERR) NB_TOT, NBANDS_CDER, NKPTS, ISPIN
           IF (NB_TOT/= WDES%NB_TOT .OR. ISPIN /= WDES%ISPIN .OR. & 
              (NKPTS /= WDES%NKPTS .AND. NKPTS /= NKPTS_NON_ZERO .AND. .NOT. ALLOCATED(IKPT_SELECT)) ) THEN
            WRITE(IU0,'(A)')' ----------------------------------------------------------------------------- '
            IF (NB_TOT/= WDES%NB_TOT .AND. IU0>=0 ) THEN 
               WRITE(IU0,'(A,I8,I8)') ' WAVEDER not read: bands not compatible',NB_TOT, WDES%NB_TOT
            ENDIF
            IF (ISPIN /= WDES%ISPIN .AND. IU0>=0 ) THEN 
               WRITE(IU0,'(A,I8,I8)') ' WAVEDER not read: spin not compatible',ISPIN, WDES%ISPIN
            ENDIF
            IF ((NKPTS /= WDES%NKPTS .AND. NKPTS /= NKPTS_NON_ZERO) .AND. .NOT. ALLOCATED(IKPT_SELECT) .AND. IU0>=0) THEN 
               WRITE(IU0,'(A,I8,I8)') ' WAVEDER not read: k-points not compatible',NKPTS, WDES%NKPTS
            ENDIF
            IERR=1
           ENDIF
         ENDIF

         IF (IERR==0) READ(IU, IOSTAT=IERR) NODES_IN_DIELECTRIC_FUNCTION
         IF (IERR==0) READ(IU, IOSTAT=IERR) WPLASMON
         IF (IERR==0) THEN
            ALLOCATE(CDER_BETWEEN_STATES( WDES%NB_TOT, NBANDS_CDER, WDES%NKPTS, WDES%ISPIN, 3))
            CDER_BETWEEN_STATES=0
            !store CDER appropriately
            IF ( ALLOCATED(IKPT_SELECT) .AND. LREAD )THEN
               ALLOCATE(CDER_TMP(WDES%NB_TOT, NBANDS_CDER, NKPTS, WDES%ISPIN, 3))
               CDER_TMP=0
               READ(IU,IOSTAT=IERR) CDER_TMP(:,:,1:NKPTS,:,:)
               NKPTS_COUNTER = 0
               DO I=1,NKPTS
                  IF ( IKPT_SELECT(I)>0)THEN
                     NKPTS_COUNTER = NKPTS_COUNTER + 1
                     CDER_BETWEEN_STATES(:,:,IKPT_SELECT(I),:,:)=CDER_TMP(:,:,I,:,:)
                  ENDIF
               ENDDO
               DEALLOCATE(CDER_TMP)
               IF (NKPTS_COUNTER == WDES%NKPTS)THEN
                  WRITE(IU0,'(A)')'Success'
               ELSE
                  WRITE(IU0,'(A,I5,A)')'FAILED! ',NKPTS_COUNTER,' k-points found'
               ENDIF
            ELSE
               READ(IU,IOSTAT=IERR) CDER_BETWEEN_STATES(:,:,1:NKPTS,:,:)
            ENDIF
         ENDIF
         CLOSE(IU)

         CALLMPI( M_bcast_i(WDES%COMM, NBANDS_CDER , 1))
         CALLMPI( M_bcast_i(WDES%COMM, IERR, 1))
      ELSE
         CALLMPI( M_bcast_i(WDES%COMM, NBANDS_CDER , 1))
         CALLMPI( M_bcast_i(WDES%COMM, IERR, 1))
         IF (IERR==0) THEN
            ALLOCATE(CDER_BETWEEN_STATES( WDES%NB_TOT, NBANDS_CDER, WDES%NKPTS, WDES%ISPIN, 3))
            CDER_BETWEEN_STATES=0
         ENDIF
      ENDIF

      IF (IERR/=0) THEN
         CALL VTUTOR('E','GW optics', &
              &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
      ENDIF

      IF (IERR==0) THEN
#ifdef gammareal
         CALLMPI( M_bcast_s(WDES%COMM, CDER_BETWEEN_STATES(1,1,1,1,1), SIZE(CDER_BETWEEN_STATES)))
#else
         CALLMPI( M_bcast_s(WDES%COMM, CDER_BETWEEN_STATES(1,1,1,1,1), SIZE(CDER_BETWEEN_STATES)*2))
#endif

         CALLMPI( M_bcast_d(WDES%COMM, NODES_IN_DIELECTRIC_FUNCTION, 1))
         CALLMPI( M_bcast_d(WDES%COMM, WPLASMON, 9))
         IF (PRESENT(EXT)) THEN
            IF (IU0>=0) WRITE(IU0,*) 'the WAVEDER.'//EXT//' file was read successfully'
         ELSE
            IF (IU0>=0) WRITE(IU0,*) 'the WAVEDER file was read successfully'
         ENDIF
      ELSE 
         IF (IU0>=0) WRITE(IU0,*) 'the WAVEDER file was not read'
         IF (ALLOCATED(CDER_BETWEEN_STATES)) DEALLOCATE(CDER_BETWEEN_STATES)
      ENDIF

    END SUBROUTINE READ_CDER_BETWEEN_STATES

END MODULE mlr_optic


SUBROUTINE ROTATE_CDER_NOINTER(WDES, CTRANS, NB, NK, ISP)
  USE mlr_optic
  USE prec
  USE wave
  IMPLICIT NONE
  TYPE (wavedes)     WDES
  INTEGER NB,NK, ISP
  GDEF CTRANS(NB,NB)
  CALL ROTATE_CDER(WDES, CTRANS, NK, ISP)

END SUBROUTINE ROTATE_CDER_NOINTER
