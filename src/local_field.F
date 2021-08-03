#include "symbol.inc"
MODULE local_field
  USE prec
  USE fock
  USE twoelectron4o
  USE chi_base
  USE wpot
  USE lattice
  IMPLICIT NONE

!**********************************************************************
!
! this subroutine handles local field effects on the level
! of screened exchange and model GW
! the GW people might call this vertex corrections
! In TD-HF and TD-DFT this term is usually included by solving
! the Casida equation.
!
! It uses the methods suggested by
! L. Reining et al. Phys. Rev. Lett. 88, 66404 (2002).
! G. Adragna, R.Del Sole, A. Marini, Phys. Rev. B 68, 165108 (2003).
! F. Sottile, V. Olevano, and L. Reining, Phys. Rev. Lett. 91, 56402 (2003).
! 
! i.e. the four orbital integrals of the BSE matrix are
! contracted into a plane wave basis set.
!
! For further reading:
! Stratmann et al., J. Chem. Phys. 109, 8218 (1998).
! Casida in Recent Advances in Density Function Methods, Vol 1, 
!     (World Sci., Singapore, 1995).
!
! It is a complicated routine, because memory requirements have
! to be balanced with efficiency.
! Hence the routine heavily relies on blocking
! The following two contributions are evaluated:
! The  direct contribution (A = <e,h | e',h'>)
! (two electrons at the same spatial coordinate, and two holes)
! describing the ladders between resonant-resonant
!
! T_q(G',G) = 
!  sum_n1,k1, n2,k2, n3, n4
!    <v_k1,n1   | e-i(G+q)r   | c_k1+q,n3> / (e_k1,n1-e_k1+q,n3)
!
! x ( int_d3r d3r' c*_k1+q,n3(r') c'_k2+q,n4(r') v'*_k2,n2(r)   v_k1,n1(r)  W(r',r)
!
! x  <c'_k2+q,n4| e i(G'+q)r' | v'_k2,n2>  / (e'_k2+q,n4 -e'_k2,n2)
!
! and a second  exchange like contribution (B= <e,h | h',e'>) (TWOELECTRON4O_ACC_EX2)
! correspodnding to resonant - anti resonant coupling
! T_q(G',G) = 
!  sum_n1,k1, n2,k2, n3, n4
!    <c_k1-q,n3 | e-i(G+q)r | v_k1,n1 > / (e_k1,n1-e_k1-q,n3)
!
! x ( int_d3r d3r' c_k1-q,n3(r')  v'*_k2,n2(r') c'_k2+q,n4(r)  v*_k1,n1(r)  W(r',r)
!
! x  <c'_k2+q,n4| e i(G'+q)r' | v'_k2,n2>  / (e'_k2+q,n4 -e'_k2,n2)
!
! The second term can be also calculated as (TWOELECTRON4O_ACC_EX)
! (this is more efficient since many operation can be combined with direct term)
!
!  sum_n1,k1, n2,k2, n3, n4
!    <v_k1,n1  | e-i(G+q)r   | c_k1+q,n3 > / (e_k1,n1-e_k1+q,n3)
!
! x ( int_d3r d3r' c*_k1+q,n3(r')  v'*_k2,n2(r') c'_k2+q,n4(r)  v_k1,n1(r)  W(r',r)
!
! x  <c'_k2+q,n4| e i(G'+q)r' | v'_k2,n2>  / (e'_k2+q,n4 -e'_k2,n2)
!
!
! The term c'_k2+q,n4(r)  v_k1,n1(r)  might look odd, but using the relation
! c'_k2+q,n4(r) = c'*_-k2-q,n4(r) and v'*_k2,n2(r') = v'_-k2,n2(r'),
! it is fairly easy to show that it is correct
! (if time inversion symmetry applies, SO might be a problem).
!
! Note that the notation A and B refers to the quantum chemists notation,
! where the full TDFT-TDHF response function between equal spins is often
! written as (a=k1,n1, b=k2,n2,   i=k3,n3,   j=k4,n4)
!
!
!        K_ai,bj    K_ai,jb         A_ai,bj +v_ai,bj     B_ai,bj +v_ai,jb
!                                =  
!        K_ia,bj    K_ia,jb         B*_ai,bj+v*_ai,jb    A*_ai,bj+v*_ai,bj
!
! between non equal spins it is given by
!
!        K_ai,bj    K_ai,jb         v_ai, bj             v_ai,jb
!                                =  
!        K_ia,bj    K_ia,jb         v*_ai,jb             v*_ai,bj
!
! The Hartree and local exchange correlation term is given by
!
!  int_d3r d3r' v'*_k2,n2(r') c'_k4,n4(r') c*_k3,n3(r) v_k1,n1(r) vbar(r',r)
!
! vbar is the bare Coulomb operator and the second derivative of the exchange correlation
! potential with respect to the charge density
! 
! v states are restricted to valence band states 
! whereas c are conduction band states
! the sums are restricted to NBANDSO bands below and above the Fermi-level
! loops over n1/n2 and n3/n4 are blocked, and the blocking parameters 
! are NSTRIPV and NSTRIP respectively
!
! outline of the structure of the routine:
!
!   allocate array W1 for merged block n1
!   allocate array W2 for merged block n2
!
!   allocate array W3 for merged block n3
!   allocate array W4 for merged block n4
!   allocate charge array (WKAPPA) of size merged block n1 x NBANDSO
!     (this costs possibly a lot of memory and restricts n1)
!
!   allocate charge array of size merged WA block n4 x merged block n1
!
!   loop over all k1-points
!     loop overal all v1-states in blocks (preferably all n1)
!        gather wavefunction in the current block
!        loop over all k2-points
!           loop overal all v2-states in blocks (preferably all n2)
!              gather wavefunction in the current block
!
!              calculate F(n1,n3,n2,n4) = c*_k1+q,n3  c'_k2+q,n4(r') W(r',r) v'*_k2,n2(r), v_k1,n1(r)
!
!              loop over n3 in blocks
!                gather wavefunctions in block 
!                loop over block n3
!                  loop over block n1
!                     calculate charge <v_k1,n1| e-i(G+q)r | c_k1+q,n3> / (e_k1,n1-e_k1+q,n3)
!
!                contract sum_n1,n3  <v_k1,n1| e-i(G-q)r | c_k1+q,n3> F(n1,n3,n2,n4)= kappa_n2,n4 (G)
!                (to allow ZGEMM it would be convenient to have a flat indexing in n1,n3)
!
!
!              loop over n4 in blocks
!                loop over block n4
!                   fft wavefunction n4
!                   loop over block n1
!                      calculate charge <c'_k2+q,n4| e i(G'+q)r' | v'_k2,n2> / (e'_k2+q,n4 -e'_k2,n2)
!
!                add to local field effects <c'_k2+q,n4| e i(G'+q)r | v'_k2,n2> kappa_n2,n4 (G)
!
!
! implemented test flags:
!
! third test results should equal Xi^2 (tests almost all routines)
!#define Xi2_test
! fourth tests results should equal Xi fxc Xi (tests all routines)
!#define XifXi_test
!
! Note on the scale of the returned result
! it is assumed that the f_xc kernel is multiplied by the 
! response function including resonant and antiresonant parts
!
!              
!**********************************************************************

 ! global array to store local field effects in DFT f_xc(r)
 ! for test purpose only

  GDEF, ALLOCATABLE :: FXCR(:)

 ! diagonal of BSE matrix related to the q->0 term 
 ! of the screened potential W(r',r)
 !  c*_k1+q,n3  c'_k2+q,n4(r') W(r',r) v'*_k1,n1(r), v_k1,n1(r)
 ! this term is precalculated by CALCULATE_LOCAL_FIELD_PREPARE
 ! CALCULATE_LOCAL_FIELD_PREPARE might also set this to zero, since
 ! one can show that the effect of this correction is just
 ! a constant shift the optical-gap by DELTA_COND
  REAL(q) :: DELTA_COND

 ! WPOTH is a handle for the screened Coulomb potential W_k(G,G')
 ! reading of the potential is handled by wpot.F

  TYPE(wpothandle), POINTER, SAVE :: WPOTH => NULL()

 ! this flag decides whether parallelization over k-points
 ! is performed or wether parallelization over bands is done
 ! NOTE: no reason that this is a global variable
 ! it is only used in the main calling routines
                                      
  LOGICAL :: LKPOINT_PARALLEL=.FALSE.

 ! frequency dependent TBSE
 ! the routines in this module can calculate frequency dependent effective
 ! two point kernels 
 ! currently this can not be switched on in the INCAR
 ! and needs to be compiled in statically by chosing

  LOGICAL :: FREQUENCY_DEPENDENT_TBSE=.FALSE.

 ! the routines that calculate the four-point Coulomb integrals
 ! can multiply in the fermi-weights (1-f_a) f_i or not
 ! for BSE calculations it is necessary to multiply them in,
 ! whereas the routine that calculates effective two point kernels (this module)
 ! take care of the weights in the routines CONSTRUCT_KAPPA.
 ! Presently, the behaviour needs to be selected by setting
 ! the global variable MULTIPLY_FRACTIONAL_OCC
 ! before calling any of the routines calculating the four orbital integrals

  LOGICAL :: MULTIPLY_FRACTIONAL_OCC=.FALSE.


 ! the routines that calculate the four-point Coulomb integrals
 ! can include the head directly in the four-point integrals
 ! this yields the same results, however, it is less flexible
 ! since it works only for cubic materials

  LOGICAL :: LHEAD_DIRECT=.FALSE.


CONTAINS
  SUBROUTINE CALCULATE_LOCAL_FIELD_FOCK( &
          P,WDES,W,LATT_CUR,T_INFO,IO,KPOINTS, &
          WGW, TBSE, TBSEA, CHI0, ENCUTLF, ENCUTGW, ENCUTGWSOFT, COMEGA, LDIRECT, LINVXI, & 
          NBANDSO, NBANDSV, NKREDLFX, NKREDLFY, NKREDLFZ, NQ, ANTIRES, LGWLF, NELM)

    USE base
    USE pseudo
    USE mpimy
    USE mkpoints
    USE constant
    USE poscar
    USE pot
    USE pawm
    USE wave_high
    USE mlr_optic
    USE ini
    IMPLICIT NONE
! structures
    TYPE (type_info)     T_INFO
    TYPE (potcar)        P(T_INFO%NTYP)
    TYPE (wavedes)       WDES
    TYPE (wavespin)      W
    TYPE (latt)          LATT_CUR
    TYPE (in_struct)     IO
    TYPE (kpoints_struct) KPOINTS
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (responsefunction) :: TBSE ! local field effects from screened exchange
    TYPE (responsefunction) :: TBSEA! local field effects from screened exchange
    TYPE (responsefunction) :: CHI0 ! determined responsefunction
    REAL(q) :: ENCUTLF              ! cutoff for local field effects
    REAL(q) :: ENCUTGW, ENCUTGWSOFT ! soft cutoffs
    TYPE (responsefunction) :: CHI  ! irreducible polarizability
    COMPLEX(q)           COMEGA(:)  ! complex frequency for local field effects
    LOGICAL              LDIRECT    ! include direct contribution
    LOGICAL              LINVXI     ! left and right multiply with inverted polarizability 
    INTEGER              NBANDSO    ! number of occupied states included in BSE
    INTEGER              NBANDSV    ! number of virtual states included in BSE
    INTEGER              NKREDLFX, NKREDLFY, NKREDLFZ
    INTEGER              NQ         ! q-point for which local field correction are required
    INTEGER              ISP        ! spin index to be considered
    INTEGER              ANTIRES    ! how to handle antiresonant part
    LOGICAL              LGWLF      ! use screened W from GW for local field effects
    INTEGER              NELM       ! if NELM > 1 the routine reads the interaction from a file
! local
    GDEF, ALLOCATABLE :: TWOELECTRON3O(:,:,:,:)
    TYPE (wavespin) WHF
    TYPE (wavefun1),ALLOCATABLE :: W1(:), W2(:), W3(:), W3P(:), W4(:)
    TYPE (wavefuna), ALLOCATABLE :: WKAPPA(:)
    GDEF,ALLOCATABLE  :: WKAPPA0(:,:,:), CWKAPPA0(:,:,:)
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2, WDESK3, WDESK3P, WDESK4, WGWQ
    INTEGER :: IERR              ! error indicator
    INTEGER :: NSTRIP            ! block size
    INTEGER :: NGLB              ! block size in conduction band
    INTEGER :: NGLB4             ! block size in conduction band index 4 
    INTEGER :: NSTRIPV           ! block size in valence band
    INTEGER :: N
    INTEGER :: NPOS1, NSTRIP1    ! base index and width of the n1 block
    INTEGER :: NPOS2, NSTRIP2    ! base index and width of the n1 block
    INTEGER :: K1, K2, K3, K3P, K4, K2_LOCAL, K4_LOCAL, K2_COLLECT, K2_DONE
    INTEGER :: VBMIN, VBMAX      ! band index for the VB minimum and VB maximum
    INTEGER :: CBMIN, CBMAX      ! band index for the CB minimum and CB maximum
    INTEGER :: CBMIN4, CBMAX4    ! band index for the CB minimum and CB maximum for n4
                                 ! this might be reduced for parallelization over bands
    REAL(q) :: SCALE
    REAL(q) :: NFFTW             ! number of FFTs for  wavefunctions
    REAL(q) :: NFLOAT4O, NFFT4O  ! number of BLAS3 operations in 4 orbital routines
    REAL(q) :: NFLOATKA, NFFTKA  ! number of BLAS3 operations in construction of kappa
    REAL(q) :: NFLOATXI, NFFTXI  ! number of BLAS3 operations in construction of XI
    INTEGER NKREDXS, NKREDYS, NKREDZS
    INTEGER :: NOMEGA
    INTEGER LAST_FILLED, FIRST_EMPTY
    LOGICAL :: LEX_CONJG         ! use the conjugated version for the exchange like contribution (B)
    LOGICAL :: W1EQUALW2         ! W1 and W2, and W3 and W4 are strictly equal
    COMPLEX(q) :: COMEGA_STATIC=(0.0_q,0.0_q)

! make shure fractional occupancies are not multiplied into the matrix elements
    MULTIPLY_FRACTIONAL_OCC=.FALSE.
!=======================================================================
! allocation of response related functions
!=======================================================================
    IF (LGWLF .AND. .NOT. ASSOCIATED(WPOTH)) & 
       CALL INIT_WPOT_HANDLE( WPOTH, WGW, KPOINTS_FULL%NKPTS, IO%IU6, IO%IU0, NKREDLFX, NKREDLFY, NKREDLFZ )

    IF (NBANDSO<=0) RETURN      ! quick return if NBANDSO is not set

      
    IF (.NOT. LDIRECT .AND. ANTIRES ==0) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_FOCK: neither the direct nor the exchange contribution is selected'
       WRITE(*,*) '  the routine will do nothing, I think it is safer to stop right here !!'
       STOP
    ENDIF

    IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_FOCK: TBSE not allocated'
       STOP
    ENDIF
    IF (.NOT. ASSOCIATED(TBSEA%RESPONSEFUN) .AND. ANTIRES>=2) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_FOCK: TBSEA not allocated'
       STOP
    ENDIF

! determine frequency grid
    IF (FREQUENCY_DEPENDENT_TBSE) THEN
       NOMEGA=SIZE(COMEGA)
       IF (NOMEGA/= SIZE(TBSE%RESPONSEFUN,3)) THEN
          WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_FOCK: size mismatch ',NOMEGA, SIZE(TBSE%RESPONSEFUN,3)
       ENDIF
    ELSE
       NOMEGA=1
    ENDIF

! allocate and clear response function arrays
    IF (FREQUENCY_DEPENDENT_TBSE) THEN
       TBSE%COMEGA=COMEGA
    ELSE 
       TBSE%COMEGA=COMEGA_STATIC
    ENDIF

    CALL CLEAR_RESPONSE(TBSE)
    CALL SET_RESPONSE_KPOINT(TBSE, WDES%VKPT(:,NQ), NQ)
    TBSE%SHIFT =0.0

    TBSE%NOMEGA_LOW=1
    TBSE%NOMEGA_HIGH=NOMEGA

    IF (ASSOCIATED(TBSEA%RESPONSEFUN)) THEN
       IF (FREQUENCY_DEPENDENT_TBSE) THEN ; TBSEA%COMEGA=COMEGA
       ELSE ; TBSEA%COMEGA=COMEGA_STATIC ; ENDIF
       CALL CLEAR_RESPONSE(TBSEA)
       CALL SET_RESPONSE_KPOINT(TBSEA, WDES%VKPT(:,NQ), NQ)
       TBSEA%SHIFT = 0 ! CHI0%SHIFT
       TBSEA%NOMEGA_LOW=1
       TBSEA%NOMEGA_HIGH=NOMEGA
    ENDIF
    IF (NELM>1) THEN
       CALL READ_LOCAL_FIELD_FOCK( TBSE, WGW, "FXC", NQ, IERR)
       IF (IERR==0 .AND. ANTIRES>=2) CALL READ_LOCAL_FIELD_FOCK( TBSEA, WGW, "FXCA", NQ, IERR)
       IF (IERR==0) RETURN
    ENDIF

    IF (FREQUENCY_DEPENDENT_TBSE) THEN
       CALL ALLOCATE_RESPONSEFUN(CHI, WGW%NGDIM, WGW%LGAMMA, .FALSE., SIZE(COMEGA))
    ELSE
       CALL ALLOCATE_RESPONSEFUN(CHI, WGW%NGDIM, WGW%LGAMMA, .FALSE., 1)
    ENDIF
    CALL CLEAR_RESPONSE(CHI)
    CALL SET_RESPONSE_KPOINT(CHI, WDES%VKPT(:,NQ), NQ)
    CHI%SHIFT  = 0 ! CHI0%SHIFT
    CHI%NOMEGA_LOW=1
    CHI%NOMEGA_HIGH=NOMEGA
    IF (FREQUENCY_DEPENDENT_TBSE) THEN
        CHI%COMEGA=COMEGA
    ELSE 
        CHI%COMEGA=COMEGA_STATIC
    ENDIF

!=======================================================================
! preparation of four-orbital related quantities
!=======================================================================
    WHF=W      ! use temporarily another WDES
    WHF%WDES => WDES_FOCK

!  save current setting for NKRED and switch to setting for local field effects
    NKREDXS=NKREDX
    NKREDYS=NKREDY
    NKREDZS=NKREDZ
    NKREDX=NKREDLFX
    NKREDY=NKREDLFY
    NKREDZ=NKREDLFZ
! determine wether the k1+k2 is found in the full k-point grid or not
    DO K1=1,KPOINTS_FULL%NKPTS
       IF (LIDENTICAL_KPOINT(WHF%WDES%VKPT(:,1)+WHF%WDES%VKPT(:,2),KPOINTS_FULL%VKPT(:,K1))) EXIT
    ENDDO
    
    IF (K1==KPOINTS_FULL%NKPTS+1 .AND. ANTIRES>=1) THEN
       LEX_CONJG=.TRUE.      ! k1+k2 not found in k-point grid use LEX_CONJG (slower)
    ELSE
       LEX_CONJG=.FALSE.     ! found, LEX_CONJG not required
    ENDIF
! conjugated version in any case if full treatment of antiresonant part is required
! LEX_CONJG can be always set (even for ANTIRES=1)
    IF (ANTIRES>=2) THEN
       LEX_CONJG=.TRUE.
    ENDIF

!test
!    LEX_CONJG=.FALSE.
!testend

    IF (LEX_CONJG .AND. IO%IU6>=0) THEN
       WRITE(IO%IU6,*) 'conjugated version for coupling between resonant and antiresonant part'
    ENDIF
! set the wavefunction descriptors
    CALL SETWDES(WGW, WGWQ, NQ)

    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK2,0)
    CALL SETWDES(WHF%WDES,WDESK3,0)
    CALL SETWDES(WHF%WDES,WDESK3P,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    LAST_FILLED=LAST_FILLED_XI_NOMOD(W,1,1)
    FIRST_EMPTY=FIRST_EMPTY_XI_NOMOD(W,1,1)

    DO ISP=1,W%WDES%ISPIN
       DO K1=1,W%WDES%NKPTS
          LAST_FILLED=MAX(LAST_FILLED_XI_NOMOD(W,K1,ISP),LAST_FILLED)
          FIRST_EMPTY=MIN(FIRST_EMPTY_XI_NOMOD(W,K1,ISP),FIRST_EMPTY)
       ENDDO
    ENDDO

    ! determine VBMIN and VBMAX
    VBMAX=LAST_FILLED
    VBMIN=MAX(LAST_FILLED-NBANDSO+1,1)

    ! NSTRIPV determines the blocking for the valence band states
    ! it can be made smaller than NBANDSO (tested at least a little bit)
    NSTRIPV=VBMAX-VBMIN+1
!   NSTRIPV=MIN(NSTRIPV,64)

    NBANDSV=MIN(WHF%WDES%NB_TOT-LAST_FILLED, NBANDSV)

    CBMIN=FIRST_EMPTY
    CBMAX=FIRST_EMPTY+NBANDSV-1

! more k-points than cores, always parallelization over k-points
! or number of k-points larger than number of conduction bands/4
    IF (WDES%NKPTS >= WDES%NB_PAR*2 .OR. WDES%NKPTS*4 >= (CBMAX-CBMIN)) THEN
       LKPOINT_PARALLEL=.TRUE.
       W1EQUALW2=.FALSE.
    ELSE
       LKPOINT_PARALLEL=.FALSE.
       W1EQUALW2=.FALSE.
       IF (WDES%NKPTS==1) W1EQUALW2=.TRUE.
    ENDIF

    IF (LKPOINT_PARALLEL) THEN
       IF (IO%IU6>=0) THEN
          WRITE(IO%IU6,*) 'parallelization over k-points'
       ENDIF

       CBMIN4=CBMIN
       CBMAX4=CBMAX
    ELSE
       IF (IO%IU6>=0) THEN
          WRITE(IO%IU6,*) 'parallelization over bands'
       ENDIF

       ! loops over the fourth index are distributed over nodes
       ! determine data distribution
       NGLB4=(CBMAX-CBMIN)/WDES%NB_PAR+1
       CBMIN4=CBMIN
       DO N=1,WDES%NB_PAR
          CBMAX4=MIN(CBMIN4+NGLB4-1,CBMAX)
          IF (N==WDES%NB_LOW) EXIT
          CBMIN4=CBMAX4+1
       ENDDO
    ENDIF

    NGLB =CBMAX -CBMIN +1
    NGLB4=CBMAX4-CBMIN4+1

    ! NSTRIP determines the blocking for the conduction band states
    ! applied only internally in the TWOELECTRON4O_ACC
    ! it limits the matrix size in BLAS level three, but values around 32-64 should
    ! be fine for maximal performance
    NSTRIP= MIN(NGLB,64)

    ! NGLB must always factorize NBANDSV, otherwise I get into problems with
    ! the blocking in the routines  CONSTRUCT_KAPPA
    IF (MOD(NBANDSV, NGLB)/=0) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD: NGLB does not factorize NBANDSV'
       STOP
    ENDIF

    ALLOCATE(W1(VBMAX-VBMIN+1))
    ALLOCATE(W2(VBMAX-VBMIN+1))
    ALLOCATE(W3(NGLB))
    ALLOCATE(W3P(NGLB))
    ALLOCATE(W4(NGLB))

    DO N=1,(VBMAX-VBMIN+1)
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL NEWWAV(W2(N) , WDESK2,.TRUE.)
    ENDDO

    DO N=1,NGLB
       CALL NEWWAV(W3(N) , WDESK3,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL NEWWAV(W4(N) , WDESK4,.TRUE.)
       IF (.NOT. W1EQUALW2 .AND. LEX_CONJG) CALL NEWWAV(W3P(N) , WDESK3P,.TRUE.)
    ENDDO

    IF (IO%IU6>=0) WRITE(IO%IU6,'(A,4I4)') 'allocating two-electron 4 orbital integral table',NSTRIPV, NGLB, NSTRIPV, NGLB4

    ALLOCATE(TWOELECTRON3O( NSTRIPV, NGLB, NSTRIPV, NGLB4))

    ALLOCATE(WKAPPA(NOMEGA))
    ! size of kappa must match last two dimensions of TWOELECTRON3O
    DO N=1,NOMEGA
       CALL NEWWAVA(WKAPPA(N), WGWQ, NSTRIPV * NGLB4 )
    ENDDO

    ! G=0 component is direction dependent store the three components
    ALLOCATE( WKAPPA0(  3, NSTRIPV * NGLB4, NOMEGA))
    ALLOCATE( CWKAPPA0( 3, NSTRIPV * NGLB4, NOMEGA))

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(A,/2(A,I5,2X,A,I5/))')' Bands included in NANO-quanta kernel f_xc= X^-1 GGWGG X^-1', & 
            ' VB(min)=',VBMIN,'VB(max)=',VBMAX,' CB(min)=',CBMIN,'CB(max)=',CBMAX
       IF (LGWLF) THEN
          WRITE(IO%IU6,'(A,3F14.7)')' W is read from the files WXXXX.tmp or WFULLXXXX.tmp if present'
       ELSE
          WRITE(IO%IU6,'(A,F14.7,A,F14.7)')' parameters for screened Coulomb W: AEXX=',AEXX,' HFSCREEN=',HFSCREEN
       ENDIF
       IF (.NOT. LDIRECT) THEN
          WRITE(IO%IU6,'(A,3F14.7)')' only exchange term included'
       ENDIF
    ENDIF

    IF (.NOT. LGWLF .AND. AEXX==0.0) THEN
       CALL VTUTOR('W','AEXX=0', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
       CALL VTUTOR('W','AEXX=0', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
    ENDIF

    SCALE=NKREDLFX*NKREDLFY*NKREDLFZ

    NFFTW = 0
    NFLOAT4O=0 ; NFFT4O=0
    NFLOATKA=0 ; NFFTKA=0
    NFLOATXI=0 ; NFFTXI=0

    CALL START_TIMING("I")
!==========================================================================
!  set up contracted four orbital integrals
!==========================================================================
    DO ISP=1,WDES%ISPIN
    DO K1=1,WDES%NKPTS
#ifdef MPI
       IF (MOD(K1-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
       ! generate the proper descriptor for W1 wavefunctions
       CALL SETWDES(WHF%WDES,WDESK1,K1)
       IF (WHF%WDES%WTKPT(K1)==0) CYCLE
       IF (SKIP_THIS_KPOINT_IN_LF(WHF%WDES%VKPT(:,K1), NKREDLFX, NKREDLFY, NKREDLFZ)) CYCLE

       CALL W1_GATHER_GLB( WHF, VBMIN, VBMAX, ISP, W1)
       NFFTW=NFFTW+VBMAX-VBMIN+1

       K3 =KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,NQ),KPOINTS_FULL)
       K3P=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ),KPOINTS_FULL)
       CALL SETWDES(WHF%WDES,WDESK3 ,K3)
       CALL SETWDES(WHF%WDES,WDESK3P,K3P)

       CALL W1_GATHER_GLB( WHF, CBMIN, CBMAX, ISP, W3)
       NFFTW=NFFTW+CBMAX-CBMIN+1

       IF (LEX_CONJG) THEN
          IF (W1EQUALW2) THEN
             W3P=W3
          ELSE
             CALL W1_GATHER_GLB( WHF, CBMIN, CBMAX, ISP, W3P)
             NFFTW=NFFTW+CBMAX-CBMIN+1
          ENDIF
       ENDIF

       DO NPOS1=VBMIN, VBMAX, NSTRIPV
          CALL GWPROGRESS(IO%IU0, K1, WDES%NKPTS, NPOS1-VBMIN+1, VBMAX-VBMIN+1 )
          NSTRIP1=MIN(VBMAX+1-NPOS1,NSTRIPV)

          K2=0
          DO
             K2=K2+1
             ! distribute the bands over k-points in a round robin fashion
             K2_DONE =0                   ! counts the number of k-points that have been collected
             K2_LOCAL=-1                  ! determine which k-point treated locally
             DO K2_COLLECT=K2, WDES%NKPTS ! loop from present K2 upto WDES%NKPTS
                ! generate the proper descriptor for W2 wavefunctions
                CALL SETWDES(WHF%WDES,WDESK2,K2_COLLECT)

                IF (WHF%WDES%WTKPT(K2_COLLECT)==0) CYCLE
                IF (SKIP_THIS_KPOINT_IN_LF(WHF%WDES%VKPT(:,K2_COLLECT), NKREDLFX, NKREDLFY, NKREDLFZ)) CYCLE

                K2_DONE=K2_DONE+1  ! new k-point to be included

                IF (LKPOINT_PARALLEL) THEN
                   CALL W1_GATHER_KSEL( WHF, VBMIN, VBMAX, ISP, W2, K2_DONE)
                   NFFTW=NFFTW+(CBMAX-CBMIN+1)/WDES%NB_PAR
                ELSE
                   IF (W1EQUALW2) THEN
                      W2=W1
                   ELSE
                      CALL W1_GATHER_GLB( WHF, VBMIN, VBMAX, ISP, W2)
                      NFFTW=NFFTW+VBMAX-VBMIN+1
                   ENDIF
                ENDIF

                K4=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K2_COLLECT)+WHF%WDES%VKPT(:,NQ),KPOINTS_FULL)
                CALL SETWDES(WHF%WDES,WDESK4,K4)

                IF (LKPOINT_PARALLEL) THEN
                   CALL W1_GATHER_KSEL( WHF, CBMIN, CBMAX, ISP, W4, K2_DONE)
                   NFFTW=NFFTW+(CBMAX-CBMIN+1)/WDES%NB_PAR
                ELSE
                   IF (W1EQUALW2) THEN
                      W4=W3
                   ELSE
                      CALL W1_GATHER_GLB( WHF, CBMIN, CBMAX, ISP, W4)
                      NFFTW=NFFTW+CBMAX-CBMIN+1
                   ENDIF
                ENDIF

                IF (K2_DONE==WDES%NB_LOW .OR. .NOT. LKPOINT_PARALLEL) THEN
                   K2_LOCAL=K2_COLLECT
                   K4_LOCAL=K4
                ENDIF
                IF (K2_DONE==WDES%NB_PAR .OR. .NOT. LKPOINT_PARALLEL) EXIT
             ENDDO

             ! at this point each CPU holds the k-points corresponding to K2_LOCAL 
             ! in W2 (and corresponding to K2-NQ in W4)
             K2=K2_LOCAL
             K4=K4_LOCAL
!==========================================================================
!  set up contracted four orbital integrals
!==========================================================================
             IF (K2>=1) THEN

                ! set K2 and K4 again
                CALL SETWDES(WHF%WDES,WDESK2,K2)
                CALL SETWDES(WHF%WDES,WDESK4,K4)

                DO NPOS2=VBMIN, VBMAX, NSTRIPV
                   NSTRIP2=MIN(VBMAX+1-NPOS2,NSTRIPV)

                   DO N=1,NOMEGA
                      WKAPPA(N)%CW=0
                   ENDDO
                   WKAPPA0  =0
                   CWKAPPA0 =0
                   TWOELECTRON3O=0

                   IF (LDIRECT) THEN
                      CALL TWOELECTRON4O_ACC_DIRECT(WHF, P, LATT_CUR, ISP, WGW, DELTA_COND,  &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT4O, NFFT4O, NSTRIP, & 
                           ENCUTGW, ENCUTGWSOFT )
                   ENDIF

                   IF (ANTIRES==1 .AND. .NOT. LEX_CONJG) THEN
                      CALL TWOELECTRON4O_ACC_EX(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT4O, NFFT4O, NSTRIP, & 
                           ENCUTGW, ENCUTGWSOFT )
                   ENDIF

                   ! Hartree and local terms are included elsewhere hence skip them usually
                   ! CALL TWOELECTRON4O_ACC_HARTREE(WHF, P, LATT_CUR, ISP, ISP, WGW, & 
                   !     .FALSE., .FALSE., 2*WDES%RSPIN, &
                   !      W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                   !      TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT4O, NFFT4O, NSTRIP)

                   ! add result to WKAPPA
                   CALL CONSTRUCT_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE,  &
                        W1, K1, NPOS1, NSTRIP1, NSTRIP2 , W3, K3, &
                        TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, & 
                        CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATKA, NFFTKA, NSTRIP, LCONJG=.FALSE., IANTI=0)
                   ! anti-resonant part handled separately and stored in TBSEA
                   ! add results stored in WKAPPA to TBSE
                   ! (TWOELECTRON3O is only used to get proper dimensions of arrays)
                   IF (ANTIRES>=2) THEN
                      CALL ADD_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE, &
                           W2, K2, K4, NPOS2, NSTRIP2, W4, K1==K2, &
                           TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, CHI, & 
                           CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATXI, NFFTXI, NSTRIP, IANTI=0)
                      DO N=1,NOMEGA
                         WKAPPA(N)%CW=0
                      ENDDO
                      WKAPPA0  =0
                      CWKAPPA0 =0
                   ENDIF
                   IF (LEX_CONJG) THEN
                      TWOELECTRON3O=0
!                     old version now replaced by ACC_EX2_INTER
!                      CALL TWOELECTRON4O_ACC_EX2(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
!                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3P, K3P, W4, K4, &
!                           TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, & 
!                           NFLOAT4O, NFFT4O, NSTRIP, & 
!                           ENCUTGW, ENCUTGWSOFT )

                      CALL TWOELECTRON4O_ACC_EX2_INTER(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                          W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3P, K3P, W4, K4, &
                           TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, & 
                           NFLOAT4O, NFFT4O, NSTRIP, LATT_CUR%B, ENCUTGW, ENCUTGWSOFT )
                      ! LCONJG implies density from W1 and W3P is conjugated 
                      CALL CONSTRUCT_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE,  &
                           W1, K1, NPOS1, NSTRIP1, NSTRIP2 , W3P, K3P, &
                           TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, & 
                           CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATKA, NFFTKA, NSTRIP , LCONJG= .TRUE., IANTI=0)
                   ELSE IF (ANTIRES>=2) THEN
                      TWOELECTRON3O=0
                      CALL TWOELECTRON4O_ACC_EX_INTER(WHF, P, LATT_CUR, ISP, WGW, 0.0_q, &
                           W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
                           TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT4O, NFFT4O, NSTRIP, & 
                           LATT_CUR%B, ENCUTGW, ENCUTGWSOFT )
                      ! add result to WKAPPA
                      CALL CONSTRUCT_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE,  &
                           W1, K1, NPOS1, NSTRIP1, NSTRIP2 , W3, K3, &
                           TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, & 
                           CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATKA, NFFTKA, NSTRIP, LCONJG=.FALSE., IANTI=0)
                   ENDIF

                   ! add results stored in WKAPPA to TBSEA (for ANTIRES>=2) or TBSE
                   IF (ANTIRES>=2) THEN
                      CALL ADD_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE, &
                           W2, K2, K4, NPOS2, NSTRIP2, W4, .FALSE., &
                           TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSEA, CHI, & 
                           CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATXI, NFFTXI, NSTRIP, IANTI=0)
                   ELSE
                      CALL ADD_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, 0.0_q, SCALE, &
                           W2, K2, K4, NPOS2, NSTRIP2, W4, K1==K2, &
                           TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, CHI,& 
                           CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOATXI, NFFTXI, NSTRIP, IANTI=0)
                   ENDIF
                ENDDO
             ENDIF
!==========================================================================
!  end of loop body
!==========================================================================
             K2=K2_COLLECT
             IF (K2>=WDES%NKPTS) EXIT
          ENDDO
       ENDDO
    ENDDO
    ENDDO

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT4O, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT4O  , 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOATKA, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFTKA  , 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOATXI, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFTXI  , 1))

    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFLOAT4O, 1))
    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFFT4O  , 1))
    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFLOATKA, 1))
    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFFTKA  , 1))
    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFLOATXI, 1))
    CALLMPI( M_sum_d(WGW%COMM_KINTER, NFFTXI  , 1))

10  FORMAT(" BLAS level 3 operations / number of FFT's:"/ &
           " number of FFTs for wave wavefunctions          ",F10.0," fft"/ & 
           " number of operations in four-orbital integrals ",F10.2," Gflops, ",F10.0," fft"/ &
           " number of operations in update of kappa        ",F10.2," Gflops, ",F10.0," fft"/ &
           " number of operations in construction of GG W GG",F10.2," Gflops, ",F10.0," fft")

    IF (IO%IU6>0) THEN
       WRITE(IO%IU6,10) NFFTW, NFLOAT4O/1E9, NFFT4O, NFLOATKA/1E9, NFFTKA,  NFLOATXI/1E9, NFFTXI
    ENDIF

    CALL SUM_RESPONSE_LF(CHI, WGW )
    CALL SUM_RESPONSE_LF(TBSE, WGW )
    IF (ANTIRES>=2) CALL SUM_RESPONSE_LF(TBSEA, WGW )

    CALL STOP_TIMING("I",IO%IU6,"LFIELDS")

    IF (WGW%LGAMMA) THEN
       CALL CHI_REAL2CMPLX( CHI)
       CALL CHI_REAL2CMPLX( TBSE)
       IF (ANTIRES>=2) CALL CHI_REAL2CMPLX(TBSEA)
    ENDIF
    CALL DUMP_RESPONSE( CHI, WGWQ, NOMEGA, & 
                  " Xi from CALCULATE_LOCAL_FIELD_FOCK", IO%IU6, 3, .FALSE.)
    CALL DUMP_RESPONSE( CHI0, WGWQ, NOMEGA, & 
                  " Xi from GW routine", IO%IU6, 3, .FALSE.)
    CALL DUMP_RESPONSE( TBSE, WGWQ, NOMEGA, & 
                  " GG W GG from CALCULATE_LOCAL_FIELD_FOCK", IO%IU6, 3, .FALSE.)

    ! need to scale the response by factor 2 if only antiresonant part is calculated
    IF (ANTIRES==-1) THEN
       SCALE=2
    ELSE
       SCALE=1
    ENDIF

    ! the supplied response function CHI0 at w=0 is used to calculate X^-1 GG W GG X^-1
    ! using this seems to improve the stability
    ! most likely because the calling routine multiplies by similar X
    ! without this copy the contributions are way overestimated !
    IF (.NOT. FREQUENCY_DEPENDENT_TBSE) THEN
       CALL COPY_CHI( CHI0, 1, CHI, 1, SCALE)
       CALL SEND_RESPONSE_LF(CHI, WGW )
    ENDIF

    CALL FORCE_HERM_CHI(CHI)
    CALL CLEAR_HIGH_ENCUTLF( CHI, WGWQ, LATT_CUR, ENCUTLF)
    CALL FORCE_HERM_CHI(TBSE)
    CALL DUMP_RESPONSE_G_OMEGA( TBSE, WGWQ, LATT_CUR, "GGWGG_diag")

    IF (LINVXI) THEN
       CALL DETERMINE_FXC_FROM_TBSE(CHI, TBSE, WGW, LATT_CUR, IO%IU0, IO%IU6)

       CALL DUMP_RESPONSE( TBSE, WGWQ, NOMEGA, & 
            " fxc from DETERMINE_FXC_FROM_TBSE", IO%IU6, 3, .FALSE.)
       CALL DUMP_RESPONSE_G( TBSE, WGWQ, LATT_CUR, "fxc_diag")
       CALL FORCE_HERM_CHI(TBSE)
    ENDIF

    CALL SEND_RESPONSE_LF(TBSE, WGW )
    
    IF (ANTIRES>=2) THEN
       CALL DUMP_RESPONSE( TBSEA, WGWQ, NOMEGA, & 
            " GG W GG antiresonant from CALCULATE_LOCAL_FIELD_FOCK", IO%IU6, 3, .FALSE.)
       CALL DUMP_RESPONSE_G( TBSEA, WGWQ, LATT_CUR, "T_anti_diag")
       CALL FORCE_HERM_CHI(TBSEA)

       IF (LINVXI) THEN
          CALL DETERMINE_FXC_FROM_TBSE(CHI, TBSEA, WGW, LATT_CUR, IO%IU0, IO%IU6)
       
          CALL DUMP_RESPONSE( TBSEA, WGWQ, NOMEGA, & 
               " fxc antiresonant from DETERMINE_FXC_FROM_TBSE", IO%IU6, 3, .FALSE.)
          CALL DUMP_RESPONSE_G( TBSEA, WGWQ, LATT_CUR, "fxc_anti_diag")
          CALL FORCE_HERM_CHI(TBSEA)
       ENDIF

       CALL SEND_RESPONSE_LF(TBSEA, WGW )
! if the following lines are undocumented ANTIRES=2 should behave
! exactly like ANTIRES=1
!       TBSE%RESPONSEFUN=(TBSE%RESPONSEFUN+TBSEA%RESPONSEFUN)
!       TBSE%WING =(TBSE%WING +TBSEA%WING)
!       TBSE%CWING=(TBSE%CWING+TBSEA%CWING)
!       TBSE%HEAD= (TBSE%HEAD+TBSEA%HEAD)
!       TBSEA%RESPONSEFUN=0
!       TBSEA%WING =0
!       TBSEA%CWING=0
!       TBSEA%HEAD =0
    ENDIF

!==========================================================================
! deallocation
!==========================================================================
    DO N=1,NGLB
       CALL DELWAV(W3(N) ,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL DELWAV(W4(N) ,.TRUE.)
       IF (.NOT. W1EQUALW2 .AND. LEX_CONJG)  CALL DELWAV(W3P(N) ,.TRUE.)
    ENDDO

    DO N=1,(VBMAX-VBMIN+1)
       CALL DELWAV(W1(N) ,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL DELWAV(W2(N) ,.TRUE.)
    ENDDO

    CALL DEALLOCATE_RESPONSEFUN( CHI )
    
    DO N=1,NOMEGA
       CALL DELWAVA(WKAPPA(N))
    ENDDO
    DEALLOCATE(WKAPPA)

    DEALLOCATE(WKAPPA0, CWKAPPA0)
    DEALLOCATE(TWOELECTRON3O)
    DEALLOCATE(W1,W2,W3,W4,W3P)

    ! restore NKRED setting
    NKREDX=NKREDXS
    NKREDY=NKREDYS
    NKREDZ=NKREDZS
    CALL WRITE_LOCAL_FIELD_FOCK( TBSE, WGW, "FXC", NQ, IERR)
    IF (ANTIRES>=2) CALL WRITE_LOCAL_FIELD_FOCK( TBSEA, WGW, "FXCA", NQ, IERR)

    RETURN

  END SUBROUTINE CALCULATE_LOCAL_FIELD_FOCK


!***********************************************************************
!
! the routine DEALLOCATE_LOCAL_FIELD_FOCK 
! deallocates all major arrays allocated by CALCULATE_LOCAL_FIELD_FOCK
! specifically the WPOTH
! 
!***********************************************************************

  SUBROUTINE DEALLOCATE_LOCAL_FIELD_FOCK

    IF (ASSOCIATED(WPOTH)) CALL DESTROY_WPOT_HANDLE(WPOTH)

  END SUBROUTINE DEALLOCATE_LOCAL_FIELD_FOCK

!***********************************************************************
!
! Determine whether a specific q point (usually k1-k2) should
! be used in the calculation of Bloch integrals such 
! as exact exchange
! 
!***********************************************************************

    FUNCTION SKIP_THIS_KPOINT_IN_LF(VKPT, NKREDX, NKREDY, NKREDZ)
      USE full_kpoints
      REAL(q) VKPT(3)
      INTEGER NKREDX, NKREDY, NKREDZ
      LOGICAL SKIP_THIS_KPOINT_IN_LF

      SKIP_THIS_KPOINT_IN_LF=.FALSE.
      IF (NKREDX>=2 .OR. NKREDY>=2 .OR. NKREDZ>=2) THEN
         IF (MOD(FLOOR((VKPT(1)+32)*KPOINTS_FULL%NKPX+.5),NKREDX)/=0) SKIP_THIS_KPOINT_IN_LF=.TRUE.
         IF (MOD(FLOOR((VKPT(2)+32)*KPOINTS_FULL%NKPY+.5),NKREDY)/=0) SKIP_THIS_KPOINT_IN_LF=.TRUE.
         IF (MOD(FLOOR((VKPT(3)+32)*KPOINTS_FULL%NKPZ+.5),NKREDZ)/=0) SKIP_THIS_KPOINT_IN_LF=.TRUE.
      ENDIF
    END FUNCTION SKIP_THIS_KPOINT_IN_LF

!*********************************************************************
!
! This subroutine performs a simple global summation 
! of all entries in CHI 
! the first version is for summing
! the second version for sending the results from the first node
! to all other nodes
!
!*********************************************************************


  SUBROUTINE SUM_RESPONSE_LF(CHI, WGW )
    IMPLICIT NONE
    TYPE (responsefunction) CHI       ! full response function
    TYPE (wavedes):: WGW              ! communicator for WPOT
  ! local
    INTEGER :: NALLOC, NOMEGA

    NALLOC=SIZE(CHI%RESPONSEFUN,1)
    NOMEGA=SIZE(CHI%RESPONSEFUN,3)

    ! sum result over all nodes
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%HEAD(1,1,1)       , 3*3*NOMEGA))


    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%HEAD(1,1,1)       , 3*3*NOMEGA))

#ifdef MPI
    IF (WGW%COMM_INTER%NODE_ME /= WGW%COMM_INTER%IONODE .OR. WGW%COMM_KINTER%NODE_ME /= WGW%COMM_KINTER%IONODE) THEN
       CALL CLEAR_RESPONSE( CHI )
    ENDIF
#endif

  END SUBROUTINE SUM_RESPONSE_LF

  SUBROUTINE SEND_RESPONSE_LF(CHI, WGW )
    IMPLICIT NONE
    TYPE (responsefunction) CHI       ! full response function
    TYPE (wavedes):: WGW              ! communicator for WPOT
  ! local
    INTEGER :: NALLOC, NOMEGA

    NALLOC=SIZE(CHI%RESPONSEFUN,1)
    NOMEGA=SIZE(CHI%RESPONSEFUN,3)

#ifdef MPI
    IF (WGW%COMM_INTER%NODE_ME /= WGW%COMM_INTER%IONODE .OR. WGW%COMM_KINTER%NODE_ME /= WGW%COMM_KINTER%IONODE) THEN
       CALL CLEAR_RESPONSE( CHI )
    ENDIF
#endif

    ! sum result over all nodes
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, CHI%HEAD(1,1,1)       , 3*3*NOMEGA))

    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, CHI%HEAD(1,1,1)       , 3*3*NOMEGA))

  END SUBROUTINE SEND_RESPONSE_LF


!**********************************************************************
!
! read the local field effects f_xc from a file
!
!**********************************************************************

  SUBROUTINE READ_LOCAL_FIELD_FOCK(TBSE, WGW, NAME, NQ, IERR) 

    IMPLICIT NONE
    TYPE (responsefunction) :: TBSE ! local field effects from screened exchange
    CHARACTER (LEN=*)       :: NAME
    TYPE (wavedes)          :: WGW  ! communicator
    INTEGER              NQ         ! q-point for which local field correction are required
    INTEGER              IERR       ! error status
  ! local
    CHARACTER (4) :: APP
    INTEGER :: IU=72, NALLOC, NOMEGA

    IERR=0
    TBSE%HEAD=0
    TBSE%WING=0
    TBSE%CWING=0
    TBSE%RESPONSEFUN=0
#ifdef MPI
    IF (WGW%COMM_INTER%NODE_ME==WGW%COMM_INTER%IONODE .AND.  WGW%COMM_KINTER%NODE_ME == WGW%COMM_KINTER%IONODE) THEN
#endif
       WRITE (APP  , "(4I1)") MOD(NQ/1000,10),MOD(NQ/100,10),MOD(NQ/10,10), MOD(NQ,10)
       OPEN( UNIT=IU, FILE=NAME//APP//".tmp", STATUS="OLD", IOSTAT=IERR, FORM='UNFORMATTED')

       IF (IERR==0) READ(IU, IOSTAT=IERR) TBSE%HEAD
       IF (IERR==0) READ(IU, IOSTAT=IERR) TBSE%WING
       IF (IERR==0) READ(IU, IOSTAT=IERR) TBSE%CWING
       IF (IERR==0) READ(IU, IOSTAT=IERR) TBSE%RESPONSEFUN
       CLOSE(IU)
#ifdef MPI
    ENDIF
#endif

    NALLOC=SIZE(TBSE%RESPONSEFUN,1)
    NOMEGA=SIZE(TBSE%RESPONSEFUN,3)

    CALLMPI( M_sum_i(WGW%COMM_INTER, IERR, 1))
    CALLMPI( M_sum_z(WGW%COMM_INTER, TBSE%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, TBSE%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, TBSE%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_INTER, TBSE%HEAD(1,1,1)       , 3*3*NOMEGA))

    CALLMPI( M_sum_i(WGW%COMM_KINTER, IERR, 1))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, TBSE%RESPONSEFUN(1,1,1), NALLOC*NALLOC*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, TBSE%WING(1,1,1)       , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, TBSE%CWING(1,1,1)      , NALLOC*3*NOMEGA))
    CALLMPI( M_sum_z(WGW%COMM_KINTER, TBSE%HEAD(1,1,1)       , 3*3*NOMEGA))


  END SUBROUTINE READ_LOCAL_FIELD_FOCK


  SUBROUTINE WRITE_LOCAL_FIELD_FOCK(TBSE, WGW, NAME, NQ, IERR) 

    IMPLICIT NONE
    TYPE (responsefunction) :: TBSE ! local field effects from screened exchange
    TYPE (wavedes)          :: WGW  ! basis set descriptor
    CHARACTER (LEN=*)       :: NAME
    INTEGER              NQ         ! q-point for which local field correction are required
    INTEGER              IERR       ! error status
  ! local
    CHARACTER (4) :: APP
    INTEGER :: IU=72

    IERR=0
#ifdef MPI
    IF (WGW%COMM_INTER%NODE_ME==WGW%COMM_INTER%IONODE .AND. WGW%COMM_KINTER%NODE_ME==WGW%COMM_KINTER%IONODE) THEN
#endif
       WRITE (APP  , "(4I1)") MOD(NQ/1000,10),MOD(NQ/100,10),MOD(NQ/10,10), MOD(NQ,10)
       OPEN( UNIT=IU, FILE=NAME//APP//".tmp", IOSTAT=IERR, FORM='UNFORMATTED')

       IF (IERR==0) WRITE(IU) TBSE%HEAD
       IF (IERR==0) WRITE(IU, IOSTAT=IERR) TBSE%WING
       IF (IERR==0) WRITE(IU, IOSTAT=IERR) TBSE%CWING
       IF (IERR==0) WRITE(IU, IOSTAT=IERR) TBSE%RESPONSEFUN

       CLOSE(IU)
#ifdef MPI
    ENDIF
#endif
    CALLMPI( M_sum_i(WGW%COMM_INTER, IERR, 1))
    CALLMPI( M_sum_i(WGW%COMM_KINTER, IERR, 1))

  END SUBROUTINE WRITE_LOCAL_FIELD_FOCK


!**********************************************************************
!
! CALCULATE_LOCAL_FIELD_PREPARE
! performs a number of task to prepare the calculation of the
! Nano-quanta kernel
!
! ) first read in the W????.tmp or WFULL????.tmp files
! ) for exchange related kernels calculate the singularity correction
!   and store in DELTA_COND
!   the kernel is then calculate on the fly using HF routines
! ) if LSHIFT is set the conduction bands are shifted down
!   which alternatively takes care of the singularity correction
!   (so DELTA_COND is set to zero, or the G=0, G'0 in the
!    screened potential read from the file is set to zero)
!
!**********************************************************************

  SUBROUTINE CALCULATE_LOCAL_FIELD_PREPARE( NBANDSO, W, WGW, LBSE, LGWLF, &
      LSHIFT,  & 
      IDIR_MAX, COMEGA, TBSE, TBSEA, ANTIRES, &
      LATT_CUR, NKREDLFX, NKREDLFY, NKREDLFZ, IU0, IU6)
    USE lattice
    INTEGER NBANDSO
    TYPE (wavespin)      W          ! wavefunction array
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    LOGICAL LBSE                    ! do BSE calculation
    LOGICAL              LSHIFT     ! apply scissor to conduction band
    LOGICAL              LGWLF      ! use W from file for particle hole ladders
    INTEGER              IDIR_MAX   ! number of independent components in the head
    COMPLEX(q)           COMEGA(:)  ! complex frequency grid
    TYPE (responsefunction) TBSE    ! local field effects from screened exchange
    TYPE (responsefunction) TBSEA   ! local field effects from screened exchange
    TYPE (responsefunction) CHI     ! response function
    INTEGER ANTIRES                 ! anti resonant part required
    TYPE (latt) LATT_CUR
    INTEGER              NKREDLFX, NKREDLFY, NKREDLFZ
  ! local
    INTEGER NK, N, ISP
    INTEGER NKREDXS, NKREDYS, NKREDZS
  ! non direction dependent fxc
    INTEGER :: NDIR, K1, K2
    REAL(q) :: DELTA
    INTEGER :: IU0, IU6
    LOGICAL :: LFULL
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
! quick return if NBANDSO is not set
    IF (NBANDSO<=0 .AND. .NOT. LBSE) THEN
       RETURN
    ENDIF
  ! TODO implementation of direction dependent fxc really sucks:
  ! if a direction dependent fxc is used (which is required if the head
  !  is taken into account at Gamma) the x, y and z are stored in 2,3,4
  ! if a speficic component is need it is copied back to 1
  ! this should and must be cleaned up but requires carefull
  ! checking of DETERMINE_FXC_FROM_TBSE_IDIR
    IF (FREQUENCY_DEPENDENT_TBSE .AND. IDIR_MAX>1) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_PREPARE: frequency dependent non isotropic fxc is presently not implemented'
       STOP
    ENDIF
    IF (IDIR_MAX==3) THEN
       NDIR=4
    ELSE
       NDIR=1
    ENDIF

    ! read in all kernels (this is safer, since sometimes it happens that the WXXXX.tmp is overwritten
    ! before it is read and this can cause symmetry problems

    IF (.NOT. LBSE .AND. LGWLF .AND. .NOT. ASSOCIATED(WPOTH)) THEN
       CALL INIT_WPOT_HANDLE( WPOTH, WGW, KPOINTS_FULL%NKPTS, IU6, IU0, NKREDLFX, NKREDLFY, NKREDLFZ )
       DO K1=1,KPOINTS_FULL%NKPTS
          IF (W%WDES%WTKPT(K1)==0) CYCLE
!          IF (SKIP_THIS_KPOINT_IN_LF(W%WDES%VKPT(:,K1), NKREDLFX, NKREDLFY, NKREDLFZ)) CYCLE
          K2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K1)-W%WDES%VKPT(:,1),KPOINTS_FULL)
          IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, K2, POTFAK, LFULL)
       ENDDO
    ENDIF
!
! calculate DELTA_COND the diagonal part of BSE kernel
! q-> of W(q,q')
!
    IF ( LGWLF ) THEN
       DELTA=0
       IF (LSHIFT) THEN
          DELTA= GET_WPOT_GZERO(WPOTH)
       ENDIF
    ELSE
       ! little stupid problem, the NKRED is statically coded
       ! we need to switch temporarily to a different downsampling
       ! TODO: this needs to be cleaned up
       NKREDXS=NKREDX
       NKREDYS=NKREDY
       NKREDZS=NKREDZ
       NKREDX=NKREDLFX
       NKREDY=NKREDLFY
       NKREDZ=NKREDLFZ

       ! precalculate singularity correction from q=0
       DELTA_COND= SET_FSG(GRIDHF, LATT_CUR, 1)

       ! restore NKRED setting
       NKREDX=NKREDXS
       NKREDY=NKREDYS
       NKREDZ=NKREDZS
       ! alternatively the effect can be take care of by shifting
       ! the conduction band by DELTA
       ! the amount of exchange at q=0 (AEXX) then however needs
       ! to be multiplied in beforehand
       IF (L_MODEL_HF) THEN
          DELTA=DELTA_COND
       ELSE
          DELTA=DELTA_COND*AEXX
       ENDIF
    ENDIF
! now shift conduction band states by the calculated diagonal
! of the BSE kernel [see. Phys. Rev. Lett. 91, 256402 (2005)]
    IF (.NOT. LBSE .AND. LSHIFT) THEN
       IF (IU0>=0) THEN
          WRITE(IU0,'(A,F10.3)') 'WARNING: conduction band is shifted by ',DELTA
       ENDIF
       IF (IU6>=0) THEN
          WRITE(IU6,'(A,/,A,F10.3,/,/A)') ' For the Nanoquanta kernel the diagonal of the BSE matrix causes', & 
                             ' a conduction band shift by ',DELTA, & 
                             ' this shift is now applied to the conduction band states'
       ENDIF

       spin:   DO ISP=1,W%WDES%ISPIN
          kpoint: DO NK=1,W%WDES%NKPTS
             DO N=1,W%WDES%NB_TOT
                W%CELTOT(N, NK, ISP)= W%CELTOT(N, NK, ISP)-DELTA*(1-W%FERTOT(N, NK, ISP))
             ENDDO
          ENDDO kpoint
       ENDDO spin
! no singularity correction from here on as we have taken
! care of it by a shift of conduction band states
       DELTA_COND=0
    ENDIF
    IF (.NOT. LBSE) THEN
       ! initialisation and allocation
       ! if a frequency dependent xc kernel is required SIZE(COMEGA) determines the 
       ! number of entries
       IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) THEN
          IF (FREQUENCY_DEPENDENT_TBSE) THEN
             CALL ALLOCATE_RESPONSEFUN(TBSE,  WGW%NGDIM, WGW%LGAMMA, .FALSE., SIZE(COMEGA))
          ELSE
             CALL ALLOCATE_RESPONSEFUN(TBSE,  WGW%NGDIM, WGW%LGAMMA, .FALSE., NDIR)
          ENDIF
       ENDIF
       CALL CLEAR_RESPONSE(TBSE)

       IF (.NOT. ASSOCIATED(TBSEA%RESPONSEFUN) .AND. ANTIRES>=2) THEN
          IF (FREQUENCY_DEPENDENT_TBSE) THEN
             CALL ALLOCATE_RESPONSEFUN(TBSEA,  WGW%NGDIM, WGW%LGAMMA, .FALSE., SIZE(COMEGA))
          ELSE
             CALL ALLOCATE_RESPONSEFUN(TBSEA,  WGW%NGDIM, WGW%LGAMMA, .FALSE., NDIR)
          ENDIF
       ENDIF
       IF (ANTIRES>=2) CALL CLEAR_RESPONSE(TBSEA)
    ENDIF

  END SUBROUTINE CALCULATE_LOCAL_FIELD_PREPARE

!**********************************************************************
!
! determine  the exchange correlation kernel 
! requires the calculation of
!
!    -1        -1
!   Xi   TBSE Xi  -> TBSE
!
! unfortunately the inversion is ill-conditioned (actually highly ill
! conditioned to be precisse)
!
!**********************************************************************

  SUBROUTINE DETERMINE_FXC_FROM_TBSE(CHI, TBSE, WGW, LATT_CUR, IU0, IU6 )
    USE prec
    USE constant
    USE lattice
    USE wave 
    USE full_kpoints

    TYPE (responsefunction) CHI
    TYPE (responsefunction) TBSE
    TYPE (wavedes) WGW
    TYPE (latt) LATT_CUR
    INTEGER IU0, IU6
  ! local
    TYPE (wavedes1), TARGET :: WGWQ
    INTEGER :: NQ_IN_WGW, N, IDIR
    COMPLEX(q) :: CHI_WORK(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
  ! required for matrix inversion
    INTEGER   IPIV(SIZE(CHI%RESPONSEFUN,1)), INFO
    INTEGER, PARAMETER :: NWORK=64
    INTEGER :: NOMEGA, NOMEGA_MAX

    IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) RETURN

    IF (CHI%LGAMMA) THEN
       CALL DETERMINE_FXC_FROM_TBSE_IDIR(CHI, TBSE, WGW, LATT_CUR, IU0, IU6 )
!       CALL DETERMINE_FXC_FROM_TBSE_DIAG(CHI, TBSE, WGW, LATT_CUR, IU0, IU6 )
       RETURN
    ENDIF
       
    NQ_IN_WGW=KPOINT_IN_FULL_GRID(CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ_IN_WGW )

    N= WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) N=N*2

    NOMEGA_MAX=SIZE(TBSE%RESPONSEFUN,3)
    IF (.NOT. FREQUENCY_DEPENDENT_TBSE) NOMEGA_MAX= 1

    DO NOMEGA=1,NOMEGA_MAX
    ! invert Xi
    CHI_WORK(1:N, 1:N)=CHI%RESPONSEFUN(1:N, 1:N, NOMEGA)

! inversion using diagonalization and possibly removing singular contributions
    CALL ROTINV( CHI_WORK, N, IU6)

    INFO=0
!    CALL ZGETRF( N, N, CHI_WORK, SIZE(CHI_WORK,1), IPIV, INFO )
!    IF (INFO==0) &
!         CALL ZGETRI( N, CHI_WORK, SIZE(CHI_WORK,1), IPIV, & 
!         WORK, SIZE(CHI%RESPONSEFUN,NOMEGA)*NWORK, INFO )

    IF (INFO>0) THEN
       CALL VTUTOR('E','Xi singular', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
       CALL VTUTOR('E','Xi singular', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
    ENDIF

    ! left hand and right hand multiplication
    CHI_WORK(1:N,1:N)=MATMUL(CHI_WORK(1:N,1:N),MATMUL(TBSE%RESPONSEFUN(1:N,1:N,NOMEGA),CHI_WORK(1:N,1:N)))

    TBSE%RESPONSEFUN(1:N,1:N,NOMEGA)= CHI_WORK(1:N,1:N)

!    CALL DUMP_CHI(" fxc from DETERMINE_FXC_FROM_TBSE", TBSE,NOMEGA, WGWQ%NGVECTOR)
    ENDDO

  END SUBROUTINE DETERMINE_FXC_FROM_TBSE

!**********************************************************************
!
! determine  the exchange correlation kernel using block inversion
!
!    -1        -1
!   Xi   TBSE Xi  -> TBSE
!
! block inversion means that first the body and the head is inverted
! and then the entire matrix is determined from the inverted body
! and head
!
!**********************************************************************


  SUBROUTINE DETERMINE_FXC_FROM_TBSE_IDIR(CHI, TBSE, WGW, LATT_CUR, IU0, IU6 )
    USE prec
    USE constant
    USE lattice
    USE wave 
    USE full_kpoints

    TYPE (responsefunction) CHI
    TYPE (responsefunction) TBSE
    TYPE (wavedes) WGW
    TYPE (latt) LATT_CUR
    INTEGER IU0, IU6
  ! local
    TYPE (wavedes1), TARGET :: WGWQ
    INTEGER :: NQ_IN_WGW, NI, NIP, N, IDIR
  ! inversion of body
    COMPLEX(q) :: C(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
  ! full inverted matrix
    COMPLEX(q) :: E(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
  ! average FXC
    COMPLEX(q) :: FXC(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
  ! head and wings and 
    COMPLEX(q) :: T(SIZE(CHI%RESPONSEFUN,1)), CB(SIZE(CHI%RESPONSEFUN,1)), & 
       CBC(SIZE(CHI%RESPONSEFUN,1)), W(SIZE(CHI%RESPONSEFUN,1)),  WC(SIZE(CHI%RESPONSEFUN,1))
    COMPLEX(q) :: D, H, A
  ! required for matrix inversion
    INTEGER   IPIV(SIZE(CHI%RESPONSEFUN,1)), INFO
    INTEGER, PARAMETER :: NWORK=64
    INTEGER    :: IDIR_MAX, NOMEGA_MAX, NOMEGA

    IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) RETURN
       
    NQ_IN_WGW=KPOINT_IN_FULL_GRID(CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ_IN_WGW )

    N= WGWQ%NGVECTOR    
    IF (WGWQ%LGAMMA) N=N*2


    ! this is so unclean that one should cry out
    ! usually the number of frequency points correspond to the size of the responsefunction array
    ! however here it is possibly missused to store the f_xc kernel for the three
    ! cartesian directions...
    IDIR_MAX=1
    NOMEGA_MAX=SIZE(TBSE%RESPONSEFUN,3)
    IF (.NOT. FREQUENCY_DEPENDENT_TBSE) NOMEGA_MAX= 1
    IF (.NOT. FREQUENCY_DEPENDENT_TBSE .AND. SIZE(TBSE%RESPONSEFUN,3)==4) IDIR_MAX=3

    DO NOMEGA=1,NOMEGA_MAX
    ! transfer Xi to C
    C=CHI%RESPONSEFUN(:,:,NOMEGA)

    ! clear head and wing
    C(1,:)=0 ; C(:,1)=0 ; C(1,1)=1
    ! the second column and row are exactly zero, get rid of it
    IF (CHI%LREAL) C(2,2)=100
    ! inversion using diagonalization and possibly removing singular contributions
    CALL ROTINV( C, N, IU6)

    INFO=0
    ! direct inversion, right now sufficient
!    CALL ZGETRF( N, N, C, SIZE(C,1), IPIV, INFO )
!    IF (INFO==0) &
!         CALL ZGETRI( N, C, SIZE(C,1), IPIV, & 
!         WORK, SIZE(CHI%RESPONSEFUN,1)*NWORK, INFO )

    IF (CHI%LREAL) C(2,2)=0

    IF (INFO>0) THEN
       CALL VTUTOR('E','Xi singular', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU0,3)
       CALL VTUTOR('E','Xi singular', &
            &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IU6,3)
       STOP
    ENDIF

    C(1,:)=0;  C(:,1)=0 ! clear wings (required by coding)

    ! clear accumulator
    FXC   =0

    DO IDIR=1,IDIR_MAX
       A=REAL(CHI%HEAD(IDIR,IDIR,NOMEGA),q)
       ! protect against division by zero (usually the case if WAVEDER is missing)
       IF (ABS(A)<1E-4) A=1E-4
       T(1:N)=MATMUL(C(1:N,1:N),CHI%WING(1:N,IDIR,NOMEGA))
       ! head of matrix
       D=1/(A-SUM(CHI%CWING(1:N,IDIR,NOMEGA)*T(1:N)))
       CB(1:N)=MATMUL(C(1:N,1:N), CHI%WING(1:N,IDIR,NOMEGA))
       CBC(1:N)=MATMUL(CHI%CWING(1:N,IDIR,NOMEGA), C(1:N,1:N))

       ! body of new matrix
       DO NI=1,N
          DO NIP=1,N
             ! updated
             E(NI,NIP)=C(NI,NIP)+D*CB(NI)*CBC(NIP)
          ENDDO
       ENDDO

       W(1:N) =-MATMUL(E(1:N,1:N),CHI%WING(1:N,IDIR,NOMEGA))/A
       WC(1:N)=-MATMUL(CHI%CWING(1:N,IDIR,NOMEGA),E(1:N,1:N))/A

       T(1:N)=MATMUL(CHI%CWING(1:N,IDIR,NOMEGA),E(1:N,1:N))

       ! head again
       H=1/A+1/A**2*SUM(T(1:N)*CHI%WING(1:N,IDIR,NOMEGA))

       ! two formulas for head must coincide
       IF (ABS(H-D)>=1E-4) THEN
          WRITE(*,*) 'internal error in DETERMINE_FXC_FROM_TBSE_IDIR: head not properly determined',H,D
       ENDIF

       ! set wings and head
       E(1:N,1)=W(1:N)
       E(1,1:N)=WC(1:N)
       E(1,1)=H
       
       CALL BODY_FROM_WING( TBSE, IDIR, NOMEGA)
       ! left hand and right hand multiplication with inverse
       E(1:N,1:N)=MATMUL(E(1:N,1:N),MATMUL(TBSE%RESPONSEFUN(1:N,1:N,NOMEGA),E(1:N,1:N)))

       IF (IDIR_MAX==3) THEN
          TBSE%RESPONSEFUN(:,:,IDIR+1)=E
       ENDIF

       ! accumulate
       FXC(1:N,1:N)=FXC(1:N,1:N)+E(1:N,1:N)*(1.0_q/IDIR_MAX)

       CALL SET_WING_FROM_MAT( E, TBSE, NOMEGA, IDIR)
    ENDDO

    TBSE%RESPONSEFUN(:,:,NOMEGA)=FXC
    TBSE%HEAD(1,2,NOMEGA)= 0
    TBSE%HEAD(1,3,NOMEGA)= 0
    TBSE%HEAD(2,3,NOMEGA)= 0
    TBSE%HEAD(2,1,NOMEGA)= 0
    TBSE%HEAD(3,1,NOMEGA)= 0
    TBSE%HEAD(3,2,NOMEGA)= 0

    ENDDO

  END SUBROUTINE DETERMINE_FXC_FROM_TBSE_IDIR

!
! this is a test routine to determine a "best" diagonal exchange correlation kernel
! not yet applicable
!
  SUBROUTINE DETERMINE_FXC_FROM_TBSE_DIAG(CHI, TBSE, WGW, LATT_CUR, IU0, IU6 )
    USE prec
    USE constant
    USE lattice
    USE wave 
    USE full_kpoints

    TYPE (responsefunction) CHI
    TYPE (responsefunction) TBSE
    TYPE (wavedes) WGW
    TYPE (latt) LATT_CUR
    INTEGER IU0, IU6
  ! local
    TYPE (wavedes1), TARGET :: WGWQ
    INTEGER :: NQ_IN_WGW, N, M, NA, NB
  ! average FXC
    COMPLEX(q) :: FXC(SIZE(CHI%RESPONSEFUN,1),SIZE(CHI%RESPONSEFUN,1))
  ! head and wings and 
    COMPLEX(q) :: A, B
    REAL(q) :: RMS, G(SIZE(CHI%RESPONSEFUN,1))
    REAL(q), PARAMETER :: DELTA=1.0

    IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) RETURN
       
    NQ_IN_WGW=KPOINT_IN_FULL_GRID(CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ_IN_WGW )

    N= WGWQ%NGVECTOR    
    IF (WGWQ%LGAMMA) N=N*2

    ! transfer Xi to C
    CALL BODY_FROM_WING( CHI, 1, 1)
    CALL BODY_FROM_WING( TBSE, 1, 1)

    FXC=0
    DO M=1,N
       FXC(M,M)=-0.1
    ENDDO

    DO

    RMS=0
    G=0
    DO M=1,N
       DO NA=1,N
          DO NB=1,N
             A=CHI%RESPONSEFUN(NA,M,1)*CHI%RESPONSEFUN(M,NB,1)
             B=CHI%RESPONSEFUN(NA,M,1)*CHI%RESPONSEFUN(M,NB,1)*FXC(M,M)-TBSE%RESPONSEFUN(NA,NB,1)
             RMS=RMS+B*CONJG(B)
             G(M)=G(M)+A*CONJG(B)+CONJG(B)*A
          ENDDO
       ENDDO
    ENDDO

    DO M=1,N
       FXC(M,M)=FXC(M,M)-G(M)*DELTA
    ENDDO

    WRITE(*,*) RMS
    WRITE(*,'(10F14.7)') (FXC(M,M),M=1,10)
    ENDDO


    TBSE%RESPONSEFUN(:,:,1)=FXC
    CALL WING_FROM_BODY( TBSE, 1, 1)
    TBSE%HEAD(1,2,1)= 0
    TBSE%HEAD(1,3,1)= 0
    TBSE%HEAD(2,3,1)= 0
    TBSE%HEAD(2,1,1)= 0
    TBSE%HEAD(3,1,1)= 0
    TBSE%HEAD(3,2,1)= 0


    IF (SIZE(TBSE%RESPONSEFUN,3)==4) THEN
       TBSE%RESPONSEFUN(:,:,2)=TBSE%RESPONSEFUN(:,:,1)
       TBSE%RESPONSEFUN(:,:,3)=TBSE%RESPONSEFUN(:,:,1)
       TBSE%RESPONSEFUN(:,:,4)=TBSE%RESPONSEFUN(:,:,1)
    ENDIF


  END SUBROUTINE DETERMINE_FXC_FROM_TBSE_DIAG



!**********************************************************************
!
! set diagonal components of response function beyond a certain
! cutoff ENCUTLF to a very large value
! inversion will than kill these elements
! gK 09.11.2012: bug fix, the indexing was into the array IGX
!    was using a Gamma point indexing, k-vector incorrectly
!    calculated
!
!**********************************************************************

  SUBROUTINE CLEAR_HIGH_ENCUTLF( CHI, WGWQ, LATT_CUR, ENCUTLF)
    USE constant
    USE lattice
    USE wave 

    IMPLICIT NONE
    TYPE (responsefunction) CHI
    TYPE (wavedes1) WGWQ
    TYPE (latt) LATT_CUR
    REAL(q) ENCUTLF
    ! local
    INTEGER    NI, NI_, NP, NOMEGA
    REAL(q) :: DKX, DKY, DKZ, GX, GY, GZ, GSQU, SCALE

    SCALE=1/TPI**2

    NP= WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) NP=NP*2

    DKX=(CHI%VKPT(1))*LATT_CUR%B(1,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(1,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(1,3)
    DKY=(CHI%VKPT(1))*LATT_CUR%B(2,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(2,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(2,3)
    DKZ=(CHI%VKPT(1))*LATT_CUR%B(3,1)+ &
        (CHI%VKPT(2))*LATT_CUR%B(3,2)+ &
        (CHI%VKPT(3))*LATT_CUR%B(3,3)

    DO NOMEGA=1,SIZE(CHI%RESPONSEFUN,3)
    DO NI=1,NP
       NI_=NI
       IF (WGWQ%LGAMMA) NI_=(NI-1)/2+1

       GX=(WGWQ%IGX(NI_)*LATT_CUR%B(1,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(1,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(1,3))
       GY=(WGWQ%IGX(NI_)*LATT_CUR%B(2,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(2,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(2,3))
       GZ=(WGWQ%IGX(NI_)*LATT_CUR%B(3,1)+WGWQ%IGY(NI_)* &
            LATT_CUR%B(3,2)+WGWQ%IGZ(NI_)*LATT_CUR%B(3,3))
          
       GSQU=HSQDTM*((DKX+GX)**2+(DKY+GY)**2+(DKZ+GZ)**2)*(TPI*TPI)

       ! useless to try inversion of well screened elements 
       ! well actually I now have the impression it is rather
       ! better to do it correctly
       ! IF (ABS(CHI%RESPONSEFUN(NI,NI,NOMEGA)*WGWQ%DATAKE(NI, 1))<1E-2) THEN
       !    CHI%RESPONSEFUN(NI,NI,NOMEGA)=1E4
       ! ENDIF

       IF (GSQU>ENCUTLF) THEN
          CHI%RESPONSEFUN(NI,NI,NOMEGA)=1E4
       ENDIF

    ENDDO
    ENDDO
  END SUBROUTINE CLEAR_HIGH_ENCUTLF

!**********************************************************************
!
! set TBSE to 
!
!   Xi TVXC Xi 
!
! this is mainly to test the previous routine
!
!**********************************************************************

  SUBROUTINE DETERMINE_TBSE_FROM_FXC(CHI, TBSE, TVXC, WGW, LATT_CUR, IU0, IU6 )
    USE prec
    USE constant
    USE lattice
    USE wave 
    USE full_kpoints

    TYPE (responsefunction) CHI
    TYPE (responsefunction) TBSE, TVXC
    TYPE (wavedes) WGW
    TYPE (latt) LATT_CUR
    INTEGER IU0, IU6
  ! local
    TYPE (wavedes1), TARGET :: WGWQ
    INTEGER :: NQ_IN_WGW, N, IDIR

#ifndef  XifXi_test
    RETURN
#endif
    IF (.NOT. ASSOCIATED(TBSE%RESPONSEFUN)) RETURN

    NQ_IN_WGW=KPOINT_IN_FULL_GRID(CHI%VKPT(:),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ_IN_WGW )

    CALL SET_RESPONSE_KPOINT(TVXC, CHI%VKPT(:), NQ_IN_WGW)
    CALL DUMP_CHI(" fxc from DETERMINE_TBSE_FROM_FXC", TVXC, 1, WGWQ%NGVECTOR)

    N= WGWQ%NGVECTOR
    IF (WGWQ%LGAMMA) N=N*2

    DO IDIR=1,3
       IF (CHI%LGAMMA) THEN
          CALL BODY_FROM_WING( TVXC, IDIR, 1)
          CALL BODY_FROM_WING( CHI, IDIR, 1)
       ENDIF
       ! left hand and right hand multiplication
       TBSE%RESPONSEFUN(1:N,1:N,1)=MATMUL(CHI%RESPONSEFUN(1:N,1:N,1),MATMUL(TVXC%RESPONSEFUN(1:N,1:N,1),CHI%RESPONSEFUN(1:N,1:N,1)))
       
       IF (CHI%LGAMMA) THEN
          TBSE%LGAMMA=.TRUE.
          CALL WING_FROM_BODY( TBSE, IDIR, 1)
       ENDIF
       IF (.NOT. CHI%LGAMMA) EXIT ! no need to cycle for of Gamma
    ENDDO

    CALL DUMP_CHI(" Xi fxc Xi from DETERMINE_TBSE_FROM_FXC", TBSE,1, WGWQ%NGVECTOR)
    TVXC%HEAD=0
    TVXC%WING=0
    TVXC%CWING=0
    TVXC%RESPONSEFUN=0

  END SUBROUTINE DETERMINE_TBSE_FROM_FXC

!**********************************************************************
!
! determine local field kernel f_xc on the local density functional
! level
! restrictions: presently only LDA is implemented
!               spin polarization is lacking as well
!
!**********************************************************************

  SUBROUTINE CALCULATE_LOCAL_FIELD_DFT( &
          GRIDC, WDES, LATT_CUR, CHTOT, DENCOR, &
          WGW, NQ, ISP, TVXC)
    USE mgrid
    USE full_kpoints

    TYPE (grid_3d)     GRIDC        ! grid for charge densities
    TYPE (wavedes)     WDES         ! wavefunction descriptor
    TYPE (latt)        LATT_CUR     ! basis vectors
    COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge density
    RGRID      DENCOR(GRIDC%RL%NP)
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    INTEGER              NQ         ! k-points for which the local field effects are required
    TYPE (responsefunction), OPTIONAL ::  TVXC    ! DFT local field effects
    INTEGER              ISP        ! spin index to be considered
  ! local
    TYPE (wavedes1)  :: WGWQ
    REAL(q)    XCSIF(3,3), EXC, XCENC
    COMPLEX(q) CVZERO
    COMPLEX(q), ALLOCATABLE :: CVTOT(:,:) ! exchange correlation potential
    COMPLEX(q), ALLOCATABLE :: CWORK(:)   ! first derivative of ex-cor. potential
    INTEGER :: I 

    IF (ISP/=1) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD_DFT: ISP=2 is not yet implemented'
       STOP
    ENDIF

    IF (PRESENT(TVXC)) THEN
       CALL SET_RESPONSE_KPOINT(TVXC, WDES%VKPT(:,NQ), NQ)

       IF (.NOT. ASSOCIATED(TVXC%RESPONSEFUN)) THEN
          CALL ALLOCATE_RESPONSEFUN(TVXC, WGW%NGDIM, WGW%LGAMMA, .FALSE., 1)
       ENDIF
       TVXC%RESPONSEFUN=0
    ENDIF

    ALLOCATE(CWORK(GRIDC%MPLWV), CVTOT(GRIDC%MPLWV,WDES%NCDIJ) )

  ! charge density to real space
    DO I=1,WDES%NCDIJ
       CALL FFT3D(CHTOT(1,I),GRIDC,1)
    ENDDO

  ! get second derivative of exchange correlation potential
    CALL FEXCP(GRIDC, LATT_CUR%OMEGA, &
         CHTOT, DENCOR, CVTOT, CWORK, CVZERO, EXC, XCENC, XCSIF,.FALSE.)

  ! second derivative to reciprocal space
  ! the factor 1/LATT_CUR%OMEGA is required
  ! needs further consideration
    CALL RL_ADD(CWORK(1),(1._q/LATT_CUR%OMEGA),CWORK(1),0.0_q,CWORK(1),GRIDC)
#ifdef debug
    CALL WRT_RL_LINE(6,GRIDC,CWORK)
#endif

    CALL RL_ADD(CWORK(1),(1._q/GRIDC%NPLWV),CWORK(1),0.0_q,CWORK(1),GRIDC)
    CALL FFT3D(CWORK(1),GRIDC,-1)

  ! determine the descriptor for the present k-point
    CALL SETWDES(WGW, WGWQ, NQ )

  ! extract the local field effects to the matrix TVXC%RESPONSEFUN
    IF (PRESENT(TVXC)) THEN
       CALL EXTRACT_LOCAL_FIELD_EFFECTS( GRIDC, CWORK, WGWQ, LATT_CUR, TVXC%RESPONSEFUN(:,:,1))
  ! head and wing stores the q and q^2 divergences for each direction
  ! only this will "survive" when the matrix is multiplied by Xi in the
  ! limit q->0
       TVXC%WING=0
       TVXC%CWING=0
       TVXC%HEAD=0
    ELSE
       CALL EXTRACT_LOCAL_FIELD_EFFECTS( GRIDC, CWORK, WGWQ, LATT_CUR)
    ENDIF


  ! charge back to reciprocal space (restore it for later use)
    DO I=1,WDES%NCDIJ
       CALL FFT_RC_SCALE(CHTOT(1,I),CHTOT(1,I),GRIDC)
       CALL SETUNB_COMPAT(CHTOT(1,I),GRIDC)
    ENDDO

    DEALLOCATE(CWORK, CVTOT)
  END SUBROUTINE CALCULATE_LOCAL_FIELD_DFT


!**********************************************************************
!
! the local field effects in DFT depend only on G-G'
! CWORK contains the fourier transforms for the second derivatives
! of the exchange and correlation potential fxc(G)
!
!   fxc(G) = sum_r fxc(r) e -i G r
!
! The kernel is defined as
! 
!   fxc(G,G') = sum_r e^iGr fxc(r) e^ -iG'r = fxc(G'-G)
!
! we need to go through the array fxc(G,G')
! and determine the corresponding storage position in CWORK (G'-G)
! somewhat complicated in the parallel version, and also
! rather involved if the storage layout is reduced in the x
! y or z direction
!
! furthermore the routine sets up the second derivative of the
! exchange correlation potential on the grid GRIDHF
! this is required in the BSE based routines
! equally complicated, actually, since many different cases
! have to be considered (Alas, I hate this; is there a way
! to have more transparent routines, well I am not sure, gK)
! 
! the routine has been tested for the following cases
! ) Si-GW  LRPA=.FALSE., parallel, parallel z-reduced, serial x-reduced
! ) Si-BSE LRPA=.FALSE., parallel, parallel z-reduced, serial x-reduced
!                        gamma only, z-reduced serial and parallel
! ) gives identical results on 1-4 nodes
!
!**********************************************************************

  SUBROUTINE EXTRACT_LOCAL_FIELD_EFFECTS( GRIDC, CWORK, WGWQ, LATT_CUR, TVXC)

    USE mgrid

    TYPE (grid_3d)       GRIDC      ! grid for charge densities
    COMPLEX(q) :: CWORK(:)          ! first derivative of ex-cor. potential
    TYPE (wavedes1)  :: WGWQ
    COMPLEX(q), OPTIONAL :: TVXC(:,:)         ! local field effects
    TYPE (latt)        LATT_CUR     ! basis vectors

  ! local
    COMPLEX(q) :: POT(GRIDC%NGX_rd,GRIDC%NGY,GRIDC%NGZ_rd)
    COMPLEX(q) :: XC, XC1, XC2
    INTEGER :: NP, NI, NI_, NIP, NIP_, IGX, IGY, IGZ, IG1, IG2, IG3, NR
    LOGICAL :: LCONJG
    COMPLEX(q) :: FXC(GRIDHF%MPLWV)
    REAL(q)    :: S2

    S2=SQRT(2.0_q)

    NP= WGWQ%NGVECTOR    
    IF (WGWQ%LGAMMA) NP=NP*2

! merge charge from CWORK to POT, POT has a simpler serial data layout
    CALL MRG_GRID_RC(GRIDC, POT(1,1,1), CWORK(1))
    IF (PRESENT(TVXC)) THEN
    TVXC=0
!
! now determine exchange correlation kernel f_xc(G,G')
!
    IF (WGWQ%LGAMMA) THEN
       ! this version is really a pain in the a..
       ! to get the xc kernel for the case that
       ! the cosine and sin transforms are stored
       ! we need to determine the following elements
       !   cos(Gr) fxc(r) cos(Gp r)       cos(Gr)  fxc(r) -sin(Gp r)
       !  -sin(Gr) fxc(r) cos(Gp r)       sind(Gr) fxc(r)  sin(Gp r)
       !
       !  to construct the kernel one has to calculate the elements
       !  elements f1= f(Gp-G) = e i(G-Gp) and f2= f(Gp +G) = e i(-G-Gp)
       !   cos(Gr) fxc(r) cos(Gp r) = Re ( f1+f2)
       !  -sin(Gr) fxc(r) cos(Gp r) = Im (-f1+f2)
       !  -cos(Gr) fxc(r) sin(Gp r) = Im ( f1+f2)
       !   sin(Gr) fxc(r) sin(Gp r) = Re ( f1-f2)
       DO NI=1,NP-1,2
          NI_=(NI-1)/2+1
          
          DO NIP=1,NP-1,2
             NIP_=(NIP-1)/2+1
             IGX=(WGWQ%IGX(NIP_)-WGWQ%IGX(NI_))
             IGY=(WGWQ%IGY(NIP_)-WGWQ%IGY(NI_))
             IGZ=(WGWQ%IGZ(NIP_)-WGWQ%IGZ(NI_))

             XC1= EXTRACT_LOCAL_FIELD_EFFECTS_G( GRIDC, IGX, IGY, IGZ, POT)

             IF (NI==1 .AND. NIP==1) THEN
                ! Kernel for cosine / cosine 
                TVXC(NI,NIP)    =XC1
             ELSE IF (NIP==1) THEN
                ! fxc(G,0) = e + iGr fxc(r) = (cos (Gr) + i sin (Gr)) fxc(r)
                ! Kernel for cos
                TVXC(NI,NIP)    =REAL( XC1,q)*S2
                ! Kernel for sin
                TVXC(NI+1,NIP)  =-AIMAG( XC1)*S2
                ! Kernel for cos
                TVXC(NI  ,NIP+1)=REAL( XC1,q)*S2
                ! Kernel for sin
                TVXC(NI+1,NIP+1)=-AIMAG(XC1)*S2
             ELSE IF (NI==1) THEN
                ! fxc(0,Gp) = e +-i Gp r fxc(r) = (cos (Gp r) - i sin (Gp r)) fxc(r)
                ! Kernel for cos
                TVXC(NI,NIP)    =REAL( XC1,q)*S2
                ! Kernel for cos
                TVXC(NI+1,NIP)  =REAL( XC1,q)*S2
                ! Kernel for sin
                TVXC(NI  ,NIP+1)=AIMAG(XC1)*S2
                ! Kernel for sin
                TVXC(NI+1,NIP+1)=AIMAG(XC1)*S2
             ELSE
                IGX=(WGWQ%IGX(NIP_)+WGWQ%IGX(NI_))
                IGY=(WGWQ%IGY(NIP_)+WGWQ%IGY(NI_))
                IGZ=(WGWQ%IGZ(NIP_)+WGWQ%IGZ(NI_))
                
                XC2= EXTRACT_LOCAL_FIELD_EFFECTS_G( GRIDC, IGX, IGY, IGZ, POT)

                ! Kernel for cosine / cosine 
                TVXC(NI,NIP)    =REAL( XC1+XC2,q)
                ! Kernel for cosine / sin
                TVXC(NI+1,NIP)  =AIMAG(-XC1+XC2)
                ! Kernel for sin / cosine
                TVXC(NI  ,NIP+1)=AIMAG( XC1+XC2)
                ! Kernel for sin / sin
                TVXC(NI+1,NIP+1)=REAL(XC1-XC2,q)
             ENDIF
          ENDDO
       ENDDO
    ELSE
       DO NI=1,NP
          DO NIP=1,NP
             IGX=(WGWQ%IGX(NIP)-WGWQ%IGX(NI))
             IGY=(WGWQ%IGY(NIP)-WGWQ%IGY(NI))
             IGZ=(WGWQ%IGZ(NIP)-WGWQ%IGZ(NI))
             
             XC = EXTRACT_LOCAL_FIELD_EFFECTS_G( GRIDC, IGX, IGY, IGZ, POT)
             ! fxc(G,G') = sum_r e^iGr fxc(r) e^ -iG'r = fxc(G'-G)
             TVXC(NI,NIP)=XC
          ENDDO
       ENDDO
    ENDIF
#ifdef debug
    DO NI=1,NP
       NI_=NI
       IF (WGWQ%LGAMMA) NI_=(NI-1)/2+1
       DO NIP=1,NP
          NIP_=NIP
          IF (WGWQ%LGAMMA) NIP_=(NIP-1)/2+1

          IGX=(WGWQ%IGX(NIP_)-WGWQ%IGX(NI_))
          IGY=(WGWQ%IGY(NIP_)-WGWQ%IGY(NI_))
          IGZ=(WGWQ%IGZ(NIP_)-WGWQ%IGZ(NI_))

          GX=(IGX*LATT_CUR%B(1,1)+IGY*LATT_CUR%B(1,2)+IGZ*LATT_CUR%B(1,3))
          GY=(IGX*LATT_CUR%B(2,1)+IGY*LATT_CUR%B(2,2)+IGZ*LATT_CUR%B(2,3))
          GZ=(IGX*LATT_CUR%B(3,1)+IGY*LATT_CUR%B(3,2)+IGZ*LATT_CUR%B(3,3))
          
          GSQU=(GX)**2+(GY)**2+(GZ)**2

          WRITE(*,'(4F14.5,2F10.3)') GSQU, GX, GY, GZ, TVXC(NI, NIP)
       ENDDO
    ENDDO
#endif
    ENDIF
!
! get second (LDA) derivative on GRIDHF
!  
      IF (.NOT. ALLOCATED(FXCR)) THEN
         ALLOCATE(FXCR(GRIDHF%RL%NP))
      ENDIF
      NI=1

      DO NIP=1,GRIDHF%RC%NCOL

         IGY=GRIDHF%LPCTY(GRIDHF%RC%I2(NIP))
         IGZ=GRIDHF%LPCTZ(GRIDHF%RC%I3(NIP))
         IG2=MOD(IGY+GRIDC%NGY*10,GRIDC%NGY)
         IG3=MOD(IGZ+GRIDC%NGZ*10,GRIDC%NGZ)
         
         ! GRIDC reduced in x-direction, omygod
         ! 
         IF (GRIDC%NGX_rd== GRIDC%NGX/2+1) THEN
            IG2=MOD(GRIDC%NGY-IG2,GRIDC%NGY)
            IG3=MOD(GRIDC%NGZ-IG3,GRIDC%NGZ)
            DO NR=1,GRIDHF%RC%NROW
               IGX=GRIDHF%LPCTX(NR)
               IG1=MOD(IGX+GRIDC%NGX*10,GRIDC%NGX)
               ! reflection through mid point
               IF (IG1 >  (GRIDC%NGX/2)) THEN
                  IG1=MOD(GRIDC%NGX-IG1,GRIDC%NGX)
                  IG2=MOD(GRIDC%NGY-IG2,GRIDC%NGY)
                  IG3=MOD(GRIDC%NGZ-IG3,GRIDC%NGZ)
                  FXC(NI)=CONJG(POT(IG1+1,IG2+1,IG3+1))
               ELSE
                  FXC(NI)=POT(IG1+1,IG2+1,IG3+1)
               ENDIF
               NI=NI+1
            ENDDO
         ELSE IF (GRIDC%NGZ_rd== GRIDC%NGZ/2+1 .AND. IG3> (GRIDC%NGZ/2)) THEN
            ! reflection through mid point
            IG2=MOD(GRIDC%NGY-IG2,GRIDC%NGY)
            IG3=MOD(GRIDC%NGZ-IG3,GRIDC%NGZ)
            DO NR=1,GRIDHF%RC%NROW
               IGX=GRIDHF%LPCTX(NR)
               IG1=MOD(IGX+GRIDC%NGX*10,GRIDC%NGX)
               IG1=MOD(GRIDC%NGX-IG1,GRIDC%NGX)
               FXC(NI)=CONJG(POT(IG1+1,IG2+1,IG3+1))
               NI=NI+1
            ENDDO
         ELSE
            ! conventional version
            DO NR=1,GRIDHF%RC%NROW
               IGX=GRIDHF%LPCTX(NR)
               IG1=MOD(IGX+GRIDC%NGX*10,GRIDC%NGX)
               FXC(NI)=POT(IG1+1,IG2+1,IG3+1)
               NI=NI+1
            ENDDO
         ENDIF
      ENDDO
      CALL FFT3D(FXC(1), GRIDHF, 1)
#ifdef debug
      CALL WRT_RL_LINE(6,GRIDHF,FXC)
#endif
      CALL LF_COPY( FXC(1), FXCR(1), GRIDHF%RL%NP)
  END SUBROUTINE EXTRACT_LOCAL_FIELD_EFFECTS


  FUNCTION EXTRACT_LOCAL_FIELD_EFFECTS_G( GRIDC, IGX, IGY, IGZ, POT)
    USE mgrid
    IMPLICIT NONE
    COMPLEX(q) :: EXTRACT_LOCAL_FIELD_EFFECTS_G
    TYPE (grid_3d)       GRIDC
    INTEGER :: IGX, IGY, IGZ
    COMPLEX(q) :: POT(:,:,:)
  ! local
    INTEGER :: IG1, IG2, IG3
    COMPLEX(q) :: XC
    
    ! bring it between 0 and GRIDC%NG?
    IGX=MOD(IGX+GRIDC%NGX*10,GRIDC%NGX)
    IGY=MOD(IGY+GRIDC%NGY*10,GRIDC%NGY)
    IGZ=MOD(IGZ+GRIDC%NGZ*10,GRIDC%NGZ)
    IF (GRIDC%NGX_rd== GRIDC%NGX/2+1) THEN
       IF (IGX<= (GRIDC%NGX/2)) THEN
          IG1=IGX
          IG2=IGY
          IG3=IGZ
          XC=POT(IG1+1,IG2+1,IG3+1)
       ELSE
          IG1=MOD(GRIDC%NGX-IGX,GRIDC%NGX)
          IG2=MOD(GRIDC%NGY-IGY,GRIDC%NGY)
          IG3=MOD(GRIDC%NGZ-IGZ,GRIDC%NGZ)
          XC=CONJG(POT(IG1+1,IG2+1,IG3+1))
       ENDIF
    ELSE IF (GRIDC%NGZ_rd== GRIDC%NGZ/2+1) THEN
       IF (IGZ<= (GRIDC%NGZ/2)) THEN
          IG1=IGX
          IG2=IGY
          IG3=IGZ
          XC=POT(IG1+1,IG2+1,IG3+1)
       ELSE
          IG1=MOD(GRIDC%NGX-IGX,GRIDC%NGX)
          IG2=MOD(GRIDC%NGY-IGY,GRIDC%NGY)
          IG3=MOD(GRIDC%NGZ-IGZ,GRIDC%NGZ)
          XC=CONJG(POT(IG1+1,IG2+1,IG3+1))
       ENDIF
    ELSE
       IG1=IGX
       IG2=IGY
       IG3=IGZ
       XC=POT(IG1+1,IG2+1,IG3+1)
    ENDIF
    EXTRACT_LOCAL_FIELD_EFFECTS_G=XC
  END FUNCTION EXTRACT_LOCAL_FIELD_EFFECTS_G

!************************ SUBROUTINE FORCE_HERM_CHI      ***************
!
! force the response functions to be Hermitian
!
!***********************************************************************

  SUBROUTINE FORCE_HERM_CHI(TBSE)
    TYPE (responsefunction) :: TBSE ! local field effects from screened exchange
    INTEGER N, NP, NOMEGA

    DO NOMEGA=1,SIZE(TBSE%RESPONSEFUN,3)
    ! force response function to be hermitian (exact for w=0 and zero shift)
    DO N=1,SIZE(TBSE%RESPONSEFUN,1)
       DO NP=N,SIZE(TBSE%RESPONSEFUN,1)
          TBSE%RESPONSEFUN(N,NP,NOMEGA)=(TBSE%RESPONSEFUN(N,NP,NOMEGA)+CONJG(TBSE%RESPONSEFUN(NP,N,NOMEGA)))/2
          TBSE%RESPONSEFUN(NP,N,NOMEGA)=CONJG(TBSE%RESPONSEFUN(N,NP,NOMEGA))
       ENDDO
    ENDDO
    TBSE%WING(:,:,NOMEGA)=(TBSE%WING(:,:,NOMEGA)+CONJG(TBSE%CWING(:,:,NOMEGA)))/2
    TBSE%CWING      =CONJG(TBSE%WING)
    TBSE%HEAD(:,:,NOMEGA)=REAL(TBSE%HEAD(:,:,NOMEGA),q)
    ENDDO
  END SUBROUTINE FORCE_HERM_CHI

!************************ SUBROUTINE COPY_CHI **************************
!
! copy a response function from one array to another
! copied entries are frequency N1 -> N2
!
!***********************************************************************

  SUBROUTINE COPY_CHI(CHI1, N1, CHI2, N2, SCALE)
    TYPE (responsefunction) :: CHI1, CHI2
    INTEGER :: N1, N2
    REAL(q) :: SCALE

    CHI2%RESPONSEFUN(:,:,N2)=CHI1%RESPONSEFUN(:,:,N1)*SCALE
    CHI2%WING (:,:,N2)      =CHI1%WING (:,:,N1)*SCALE
    CHI2%CWING(:,:,N2)      =CHI1%CWING(:,:,N1)*SCALE
    CHI2%HEAD (:,:,N2)      =CHI1%HEAD (:,:,N1)*SCALE

  END SUBROUTINE COPY_CHI

!************************ SUBROUTINE ROTINV   **************************
!
! this subroutine calculates a matrix to the power -1
! but kills the very low frequency components
!
!***********************************************************************

  SUBROUTINE ROTINV(CUNI, NBANDS, IU6 )
    INTEGER NBANDS
    COMPLEX(q) CUNI(:,:)
    INTEGER ::  IU6
    ! local
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    COMPLEX(q), ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)
    
    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))
!=======================================================================
! diagononalize the matrix
!=======================================================================
    CALL ZHEEV &
         &         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         &           W,  IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTINV: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    IF (IU6>=0) WRITE(IU6,'(10F10.6)') HFEIG
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
    DO N1=1,NBANDS
       DO N2=1,NBANDS
          IF (ABS(HFEIG(N2))<1E-8) THEN
             CTMP(N1,N2)=0
          ELSE
             CTMP(N1,N2)=CUNI(N1,N2)/HFEIG(N2)
          ENDIF
       ENDDO
    ENDDO
    CALL ZGEMM( 'N','C', NBANDS, NBANDS, NBANDS, (1.0_q, 0.0_q), CTMP, &
         &              NBANDS, CUNI(1,1), NDIM, (0.0_q, 0.0_q), CEIDB, NBANDS)
   
    CUNI=0 
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE ROTINV

!************************ SUBROUTINE ROTHALF   *************************
!
! this subroutine calculates a matrix to the power -1/2
! but kills the very low frequency components
!
!***********************************************************************

  SUBROUTINE ROTHALF(CUNI, NBANDS, IU6 )
    INTEGER NBANDS
    COMPLEX(q) CUNI(:,:)
    INTEGER ::  IU6
    ! local
    REAL(q) :: HFEIG(NBANDS),W(3*NBANDS)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    COMPLEX(q), ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)
    
    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(NBANDS,NBANDS),CEIDB(NBANDS,NBANDS))
!=======================================================================
! diagononalize the matrix
!=======================================================================
    CALL ZHEEV &
         &         ('V','U',NBANDS,CUNI(1,1),NDIM,HFEIG,CTMP,NBANDS*NBANDS, &
         &           W,  IFAIL)
    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in ROTHALF: Call to routine ZHEEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
    IF (IU6>=0) WRITE(IU6,'(10F10.6)') HFEIG
!=======================================================================
! set up the unitary transformation CEIDB for wavefunctions
!=======================================================================
    DO N1=1,NBANDS
       DO N2=1,NBANDS
          IF (ABS(HFEIG(N2))<1E-8) THEN
             CTMP(N1,N2)=0
          ELSE
             CTMP(N1,N2)=CUNI(N1,N2)/SQRT(ABS(HFEIG(N2)))
          ENDIF
       ENDDO
    ENDDO
    CALL ZGEMM( 'N','C', NBANDS, NBANDS, NBANDS, (1.0_q, 0.0_q), CTMP, &
         &              NBANDS, CUNI(1,1), NDIM, (0.0_q, 0.0_q), CEIDB, NBANDS)
   
    CUNI=0 
    CUNI(1:NBANDS,1:NBANDS)=CEIDB(1:NBANDS,1:NBANDS)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE ROTHALF

!**********************************************************************
!
! loop over n3 in blocks
!   gather wavefunctions in current block 
!   loop over block n3
!     loop over block n1
!        calculate charge <v_k1,n1| e-i(G+q)r | c_k1+q,n3> / (e_k1,n1-e_k1+q,n3)
!
!   contract sum_n1,n3 F(n1,n3,n2,n4) <v_k1,n1| e-i(G-q)r | c_k1+q,n3> = kappa_n2,n4 (G)
!
!**********************************************************************
  
  SUBROUTINE CONSTRUCT_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, DELTA, SCALE, &
               W1, K1, NPOS1, NSTRIP1, NSTRIP2, W3, K3, &
               TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, & 
               CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, LCONJG , IANTI)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE subrot_cluster
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (wavedes1)    WGWQ
    TYPE (potcar)      P(:)
    INTEGER ISP, NQ                    
    TYPE (wavefun1)    W1(:)
    REAL(q) DELTA   ! shift of QP spectrum in BSE
    REAL(q) SCALE
    INTEGER K1, K3, NPOS1, NSTRIP1, NSTRIP2
    TYPE (wavefun1)    W3(:)
    GDEF :: TWOELECTRON3O(:,:,:,:)
    TYPE (wavefuna) :: WKAPPA(:)
    GDEF      :: WKAPPA0(:,:,:)
    GDEF      :: CWKAPPA0(:,:,:)
    TYPE (responsefunction) :: TBSE       ! local field effects from screened exchange
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP
    LOGICAL :: LCONJG                     ! add conjugated contribution 
    INTEGER :: IANTI
! local 
    INTEGER NPOS3, NSTRIP3, N1, N3, NP
    INTEGER NB1_INTO_TOT, NB3_INTO_TOT, NB1, NB3
    COMPLEX(q) :: WEIGHT
    COMPLEX(q), ALLOCATABLE :: GCHG(:,:,:)      ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHGW(:,:,:)     ! charge weighted
    COMPLEX(q), ALLOCATABLE :: GWORK(:)
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)      
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    LOGICAL LPHASE
    INTEGER I, J, NK1_IN_KPOINTS_FULL_ORIG
    GDEF :: CDER_BETWEEN_STATE(3)
    INTEGER :: NOMEGA
    COMPLEX(q) :: COMEGA

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    IF (NSTRIP1 /= SIZE(TWOELECTRON3O,1)) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: NSTRIP1 must be equal first dimension of TWOELECTRON3O', &
            NSTRIP1, SIZE(TWOELECTRON3O,1)
       STOP
    ENDIF
    IF (SIZE(TWOELECTRON3O,3)*SIZE(TWOELECTRON3O,4)/= SIZE(WKAPPA(1)%CW,2)) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: size problem with WKAPPA', &
            SIZE(TWOELECTRON3O,3)*SIZE(TWOELECTRON3O,4), SIZE(WKAPPA(1)%CW,2)
       STOP
    ENDIF

    IF (TBSE%SHIFT/=0 .AND. WGWQ%LGAMMA) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: Gamma point only version does not support complex shift'
       STOP
    ENDIF

    ! determined the index of this k-point in the original full k-point grid
    NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)

    NP=WGWQ%NGVECTOR

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHG(NP, SIZE(TWOELECTRON3O,1), NSTRIP), &
         GCHGW(NP, SIZE(TWOELECTRON3O,1), NSTRIP), &
         GWORK( MAX(GRIDHF%MPLWV,WGWQ%GRID%MPLWV)))
    ! set the phase factors q'=k3-k1
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K1))

    ! unfortunately q'=k_3-k_1-q might be any reciprocal lattice vector G
    ! CPHASE stores the required "shift" e+iGr (see chi_base.F)
    IF (LCONJG) THEN
       CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,NQ), &
            GRIDHF,CPHASE(1),LPHASE)
    ELSE
       CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,NQ), &
            GRIDHF,CPHASE(1),LPHASE)
    ENDIF


    DO NPOS3=CBMIN, CBMAX, NSTRIP
       NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

       GCHG=0
       DO N1=1,NSTRIP1
          NB1_INTO_TOT=NPOS1-1+N1
          NB1         =NPOS1-VBMIN+N1
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE
             
             !  v_k1,n1*(r) c_k3,n3 (r) 
             CALL FOCK_CHARGE_NOINT( W3(NB3), W1(NB1), GWORK(1), CRHOLM,  SIZE(CRHOLM))

             ! apply phase factor e^iGr if required 
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             ! conjugate charge in real space (same as interchanging the states)
             IF (LCONJG .AND. .NOT. WGWQ%LGAMMA) THEN
                GWORK(1:GRIDHF%RL%NP)=GCONJG(GWORK(1:GRIDHF%RL%NP))
             ENDIF

             ! extract data using wave function FFT (impose cutoff at the same time)
             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG(1,N1,N3),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             ! G-> 0 limit, consider all three directions and store result in WKAPPA0
             IF (ABS(SUM(WHF%WDES%VKPT(:,NQ)**2))<G2ZERO) THEN    
                CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK1_IN_KPOINTS_FULL_ORIG, ISP, NB1_INTO_TOT, NB3_INTO_TOT)
                ! CDER_BETWEEN_STATE =  <v_k1,n1|- i d/dq_j | c_k1+q,n3>
                ! we need \int r v*_k1,n1(r) c_k1+q,n3(r) = i CDER_BETWEEN_STATE
                IF (LCONJG) THEN
                   CDER_BETWEEN_STATE=GCONJG(CDER_BETWEEN_STATE)
                ENDIF
                IF (.NOT. WGWQ%LGAMMA) THEN
                   CDER_BETWEEN_STATE=(0.0_q,1.0_q)*CDER_BETWEEN_STATE
                ENDIF

                DO NOMEGA=1,SIZE(WKAPPA)
                   COMEGA=TBSE%COMEGA(NOMEGA) ! positive frequencies

                   WEIGHT=0
                   IF (REAL(WHF%CELTOT(NB3_INTO_TOT, K3, ISP)-WHF%CELTOT(NB1_INTO_TOT,K1,ISP),q)>DEGENERACY_THRESHOLD) THEN

                      IF (IANTI==0 .OR. IANTI==1) &
                           WEIGHT=WEIGHT+1/( COMEGA+(REAL(WHF%CELTOT(NB1_INTO_TOT,K1,ISP)-WHF%CELTOT(NB3_INTO_TOT, K3, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                      IF (IANTI==0 .OR. IANTI==2) &
                           WEIGHT=WEIGHT+1/(-COMEGA+(REAL(WHF%CELTOT(NB1_INTO_TOT,K1,ISP)-WHF%CELTOT(NB3_INTO_TOT, K3, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                      ! weighttest
                      WEIGHT=WEIGHT*SCALE*WHF%WDES%WTKPT(K1)*(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)-WHF%FERTOT(NB3_INTO_TOT, K3, ISP))
                      WEIGHT=WEIGHT*WHF%AUXTOT(NB3_INTO_TOT, K3, ISP)
                   ENDIF

                   DO J=0, SIZE(TWOELECTRON3O,4)-1
                   DO I=0, SIZE(TWOELECTRON3O,3)-1
                      WKAPPA0(1,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=WKAPPA0(1,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(1)*WEIGHT
                      WKAPPA0(2,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=WKAPPA0(2,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(2)*WEIGHT
                      WKAPPA0(3,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=WKAPPA0(3,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(3)*WEIGHT

                      CWKAPPA0(1,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=CWKAPPA0(1,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           GCONJG(TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(1))*WEIGHT
                      CWKAPPA0(2,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=CWKAPPA0(2,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           GCONJG(TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(2))*WEIGHT
                      CWKAPPA0(3,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)=CWKAPPA0(3,I+1+J*SIZE(TWOELECTRON3O,3),NOMEGA)+ & 
                           GCONJG(TWOELECTRON3O(N1, (NPOS3-CBMIN)+N3,I+1,J+1) * CDER_BETWEEN_STATE(3))*WEIGHT
                   ENDDO
                   ENDDO
                ENDDO
             ENDIF
          ENDDO
       ENDDO

       GCHGW=0
       DO NOMEGA=1,SIZE(WKAPPA)

       COMEGA=TBSE%COMEGA(NOMEGA) ! positive frequencies
          
       DO N1=1,NSTRIP1
          NB1_INTO_TOT=NPOS1-1+N1
          NB1         =NPOS1-VBMIN+N1
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

             WEIGHT=0
             IF (REAL(WHF%CELTOT(NB3_INTO_TOT, K3, ISP)-WHF%CELTOT(NB1_INTO_TOT,K1,ISP),q)>DEGENERACY_THRESHOLD) THEN

                IF (IANTI==0 .OR. IANTI==1) &
                     WEIGHT=WEIGHT+1/( COMEGA+(REAL(WHF%CELTOT(NB1_INTO_TOT,K1,ISP)-WHF%CELTOT(NB3_INTO_TOT, K3, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                IF (IANTI==0 .OR. IANTI==2) &
                     WEIGHT=WEIGHT+1/(-COMEGA+(REAL(WHF%CELTOT(NB1_INTO_TOT,K1,ISP)-WHF%CELTOT(NB3_INTO_TOT, K3, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                ! factor 2 is for compatibility with previous coding, but that should/must really go
                ! weighttest
                WEIGHT=WEIGHT*SCALE*WHF%WDES%WTKPT(K1)*(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)-WHF%FERTOT(NB3_INTO_TOT, K3, ISP))
                WEIGHT=WEIGHT*WHF%AUXTOT(NB3_INTO_TOT, K3, ISP)
             ENDIF

             GCHGW(1:NP,N1,N3)=GCHG(1:NP,N1,N3)*(WEIGHT/GRIDHF%NPLWV)
          ENDDO
       ENDDO

       IF (WGWQ%LGAMMA) THEN
!     contract sum_n1,n3 <v_k1,n1| cos(G-q)r / sin(G-q)r  | c_k1+q,n3> F(n1,n3,n2,n4) = kappa_n2,n4 (G)
          CALL DGEMM('N','N', 2*NP,  SIZE(WKAPPA(NOMEGA)%CW,2), SIZE(TWOELECTRON3O,1)*NSTRIP3, 1.0_q , &
            GCHGW(1,1,1), 2*SIZE(GCHGW,1),  &
            TWOELECTRON3O(1,(NPOS3-CBMIN)+1,1,1), SIZE(TWOELECTRON3O,1)*SIZE(TWOELECTRON3O,2), &
            1.0_q , WKAPPA(NOMEGA)%CW(1,1), 2*SIZE(WKAPPA(NOMEGA)%CW,1))
          NFLOAT=NFLOAT+NP*SIZE(WKAPPA(NOMEGA)%CW,2)*SIZE(TWOELECTRON3O,1)*NSTRIP3*4
       ELSE
!     contract sum_n1,n3 <v_k1,n1| e-i(G-q)r | c_k1+q,n3> F(n1,n3,n2,n4)  = kappa_n2,n4 (G)
          CALL ZGEMM('N','N', NP,  SIZE(WKAPPA(NOMEGA)%CW,2), SIZE(TWOELECTRON3O,1)*NSTRIP3, (1.0_q, 0.0_q), &
            GCHGW(1,1,1), SIZE(GCHGW,1),  &
            TWOELECTRON3O(1,(NPOS3-CBMIN)+1,1,1), SIZE(TWOELECTRON3O,1)*SIZE(TWOELECTRON3O,2), &
            (1.0_q, 0.0_q) , WKAPPA(NOMEGA)%CW(1,1), SIZE(WKAPPA(NOMEGA)%CW,1))
          NFLOAT=NFLOAT+NP*SIZE(WKAPPA(NOMEGA)%CW,2)*SIZE(TWOELECTRON3O,1)*NSTRIP3*8
       ENDIF
    ENDDO
    ENDDO

    DEALLOCATE(CRHOLM, GCHGW, GCHG, GWORK)


  END SUBROUTINE CONSTRUCT_KAPPA

!**********************************************************************
!
! loop over n4 in blocks
!   loop over block n4
!      fft wavefunction n4
!      loop over block n2
!         calculate charge <c'_k2+q,n4| e i(G'+q)r | v'_k2,n2> / (e'_k2+q,n4 -e'_k2,n2)
!
!   add to local field effects  <c'_k2+q,n4| e i(G'+q)r | v'_k2,n2> kappa_n2,n4 (G)
!
!**********************************************************************

  
  SUBROUTINE ADD_KAPPA(WHF, P, LATT_CUR, ISP, NQ, WGWQ, DELTA, SCALE, &
               W2, K2, K4, NPOS2, NSTRIP2, W4, LCHI, &
               TWOELECTRON3O, WKAPPA, WKAPPA0, CWKAPPA0, TBSE, CHI, & 
               CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, IANTI)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    USE subrot_cluster
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin)    WHF
    TYPE (wavedes1)    WGWQ
    TYPE (potcar)      P(:)
    INTEGER ISP, NQ                    
    TYPE (wavefun1)    W2(:), W4(:)
    LOGICAL :: LCHI     ! add to polarizability
    REAL(q)  DELTA
    INTEGER K2, K4, NPOS2, NSTRIP2
    REAL(q) SCALE
    GDEF :: TWOELECTRON3O(:,:,:,:)
    TYPE (wavefuna) :: WKAPPA(:)
    GDEF            :: WKAPPA0(:,:,:)
    GDEF            :: CWKAPPA0(:,:,:)
    TYPE (responsefunction) :: TBSE, CHI ! local field effects from screened exchange
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP
    INTEGER :: IANTI  ! 0 resonant and antiresonant, 1 resonant only, 2 anti resonant only
! local 
    INTEGER NPOS4, NSTRIP4, N2, N4, NP
    INTEGER NB2_INTO_TOT, NB2, NB4, NB4_INTO_TOT, NK2_IN_KPOINTS_FULL_ORIG
    COMPLEX(q), ALLOCATABLE :: GCHG(:,:,:)      ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHGW(:,:,:)     ! weighted charge 
    COMPLEX(q), ALLOCATABLE :: GWORK(:)         ! charge weighted by 1/(w-e1-e2)
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF :: WEIGHT, WEIGHT_REL, WEIGHT_CHI
    LOGICAL LPHASE
    INTEGER I, J
    GDEF :: CDER_BETWEEN_STATE(3), C(3), CC(3)
    INTEGER :: NOMEGA
    COMPLEX(q) :: COMEGA

    IF (NSTRIP2 /= SIZE(TWOELECTRON3O,3)) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: NSTRIP2 must be equal to third dimension of TWOELECTRON3O', &
            NSTRIP2, SIZE(TWOELECTRON3O,3)
       STOP
    ENDIF
    IF (SIZE(TWOELECTRON3O,3)*SIZE(TWOELECTRON3O,4)/= SIZE(WKAPPA(1)%CW,2)) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: size problem with WKAPPA', &
            SIZE(TWOELECTRON3O,3)*SIZE(TWOELECTRON3O,4), SIZE(WKAPPA(1)%CW,2)
       STOP       
    ENDIF

    IF (TBSE%SHIFT/=0 .AND. WGWQ%LGAMMA) THEN
       WRITE(*,*) 'internal error in CONSTRUCT_KAPPA: Gamma point only version does not support complex shift'
       STOP
    ENDIF

    ! determined the index of this k-point in the original full k-point grid
    NK2_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K2),KPOINTS_FULL_ORIG)

    NP=WGWQ%NGVECTOR

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), & 
         GCHGW(NP, SIZE(TWOELECTRON3O,3), NSTRIP), &
         GCHG (NP, SIZE(TWOELECTRON3O,3), NSTRIP), &
         GWORK( MAX(GRIDHF%MPLWV,WGWQ%GRID%MPLWV)))
    ! set the phase factors q'=k4-k1
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, KPOINTS_FULL%VKPT(:,K4)-KPOINTS_FULL%VKPT(:,K2))

    ! unfortunately q'=k_4-k_2-q might be any reciprocal lattice vector G
    ! CPHASE stores the required "shift" e+iGr (see chi_base.F)
    CALL SETPHASE(WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), &
         GRIDHF,CPHASE(1),LPHASE)


    DO NOMEGA=1,SIZE(WKAPPA)
    COMEGA=TBSE%COMEGA(NOMEGA)

    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       GCHGW=0
       GCHG =0
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN4+N4
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE
                
          DO N2=1,NSTRIP2
             NB2_INTO_TOT=NPOS2-1+N2
             NB2         =NPOS2-VBMIN+N2
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

             WEIGHT_CHI=0
             WEIGHT=0
             WEIGHT_REL=0
             
             IF (REAL(WHF%CELTOT(NB4_INTO_TOT, K4, ISP)-WHF%CELTOT(NB2_INTO_TOT,K2,ISP),q)>DEGENERACY_THRESHOLD) THEN
                WEIGHT_CHI= &
                     (1/( COMEGA+(REAL(WHF%CELTOT(NB2_INTO_TOT,K2,ISP)-WHF%CELTOT(NB4_INTO_TOT, K4, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))+ &
                     1/(-COMEGA+(REAL(WHF%CELTOT(NB2_INTO_TOT,K2,ISP)-WHF%CELTOT(NB4_INTO_TOT, K4, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q))))

                IF (IANTI==0) THEN
                   WEIGHT=WEIGHT_CHI
                ELSE IF (IANTI==1) THEN
                   ! we calculate only (resonant-antiresonant coupling, multiply by 2 to correct for this error)
                   WEIGHT=2/( COMEGA+(REAL(WHF%CELTOT(NB2_INTO_TOT,K2,ISP)-WHF%CELTOT(NB4_INTO_TOT, K4, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                ELSE IF (IANTI==2) THEN
                   WEIGHT=2/(-COMEGA+(REAL(WHF%CELTOT(NB2_INTO_TOT,K2,ISP)-WHF%CELTOT(NB4_INTO_TOT, K4, ISP),q)+DELTA-CMPLX(0, TBSE%SHIFT,q)))
                ENDIF

                ! see comment below
                WEIGHT_REL=WEIGHT_CHI/WEIGHT
                ! weighttest            
                WEIGHT    =SCALE*WHF%WDES%RSPIN*WHF%WDES%WTKPT(K2)*WEIGHT*(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)-WHF%FERTOT(NB4_INTO_TOT, K4, ISP))
                WEIGHT_CHI=SCALE*WHF%WDES%RSPIN*WHF%WDES%WTKPT(K2)*(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)-WHF%FERTOT(NB4_INTO_TOT, K4, ISP))*WEIGHT_CHI
                WEIGHT=WEIGHT*WHF%AUXTOT(NB4_INTO_TOT, K4, ISP)
                WEIGHT_CHI=WEIGHT_CHI*WHF%AUXTOT(NB4_INTO_TOT, K4, ISP)

             ENDIF
             !  v_k2,n2*(r') c_k4,n4 (r') 
             CALL FOCK_CHARGE_NOINT( W4(NPOS4+N4-CBMIN), W2(NB2), GWORK(1), CRHOLM,  SIZE(CRHOLM))

             ! apply phase factor e^iGr if required
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             ! extract data using wave function FFT (impose cutoff at the same time)
             ! v_k2,n2*(r') c_k4,n4(r') e-iG'r'
             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHGW(1,N2,N4),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1
             !  conjugate GCHGW to get the required
             !  <c'_k2+q,n4| e i(G'+q)r' | v'_k2,n2>  / (e'_k2+q,n4 -e'_k2,n2)
             GCHG (1:NP,N2,N4)=      (GCHGW(1:NP,N2,N4)*(1.0_q/GRIDHF%NPLWV))
             IF (.NOT.WGWQ%LGAMMA) THEN
                GCHGW(1:NP,N2,N4)= CONJG(GCHG(1:NP,N2,N4))
             ELSE
                GCHGW(1:NP,N2,N4)= GCHG(1:NP,N2,N4)
             ENDIF

             ! determine head and wing

             IF (ABS(SUM(WHF%WDES%VKPT(:,NQ)**2))<G2ZERO) THEN
                CALL  CDER_BETWEEN_STATES_ROTATED(CDER_BETWEEN_STATE,LATT_CUR, NK2_IN_KPOINTS_FULL_ORIG, ISP, NB2_INTO_TOT, NB4_INTO_TOT)
                ! CDER_BETWEEN_STATE =  <v'_k2,n2|- i d/dq_j | c'_k2+q,n4>
                ! we need \int r c'_k2+q,n4(r) v'_k2,n2(r)  = CONJG(i CDER_BETWEEN_STATE)
                IF (.NOT.WGWQ%LGAMMA) THEN
                   CDER_BETWEEN_STATE=CONJG((0.0_q,1.0_q)*CDER_BETWEEN_STATE)
                ENDIF

                C=  WKAPPA0(:,(NB4-1)*SIZE(TWOELECTRON3O,3)+N2, NOMEGA )
                CC=CWKAPPA0(:,(NB4-1)*SIZE(TWOELECTRON3O,3)+N2, NOMEGA )

                DO I=1,3
                   DO J=1,3
                      TBSE%HEAD(J,I,NOMEGA)= TBSE%HEAD(J,I,NOMEGA)+CDER_BETWEEN_STATE(J)*C(I)*WEIGHT
                   ENDDO
                ENDDO

                DO I=1,3
                   ! for gamma point only the complex array is interpreted as a real array
                   ! with twice as many elements (since C, CC and weight are real this works)
                   TBSE%WING(1:NP,I,NOMEGA) =TBSE%WING(1:NP,I,NOMEGA) +GCHGW(1:NP,N2,N4)*C(I)*WEIGHT
                   TBSE%CWING(1:NP,I,NOMEGA)=TBSE%CWING(1:NP,I,NOMEGA)+GCHG(1:NP,N2,N4)*CC(I)*WEIGHT
                ENDDO

                IF (LCHI) THEN
                   ! update response function
                   DO I=1,3
                      DO J=1,3
                         CHI%HEAD(J,I,NOMEGA)= CHI%HEAD(J,I,NOMEGA)+CDER_BETWEEN_STATE(J)*GCONJG(CDER_BETWEEN_STATE(I))*WEIGHT_CHI
                      ENDDO
                   ENDDO
                   
                   DO I=1,3
                      CHI%WING(1:NP,I,NOMEGA) =CHI%WING(1:NP,I,NOMEGA) +GCHGW(1:NP,N2,N4)*GCONJG(CDER_BETWEEN_STATE(I))*WEIGHT_CHI
                      CHI%CWING(1:NP,I,NOMEGA)=CHI%CWING(1:NP,I,NOMEGA)+GCHG(1:NP,N2,N4)*CDER_BETWEEN_STATE(I) *WEIGHT_CHI
                   ENDDO

                ENDIF
             ENDIF
             GCHGW(1:NP,N2,N4)=GCHGW(1:NP,N2,N4)*WEIGHT
             ! this is a little bit unclean: for response function we want simply the WEIGHT_CHI
             ! GCHGW is weighted by WEIGHT instead of WEIGHT_CHI so weight GCHG by the relative weight
             GCHG(1:NP,N2,N4) =GCHG(1:NP,N2,N4) *WEIGHT_REL
          ENDDO
       ENDDO

       IF (WGWQ%LGAMMA) THEN
          ! TBSE(G',G) = <c'_k2+q,n4| e i(G'+q)r | v'_k2,n2> kappa_n2,n4 (G)
          ! for gamma point only the complex array is interpreted as a real array
          CALL DGEMM('N','T', 2*NP, 2*NP,  SIZE(TWOELECTRON3O,3)*NSTRIP4, 1.0_q , &
            GCHGW(1,1,1), 2*SIZE(GCHGW,1), &
            WKAPPA(NOMEGA)%CW(1,(NPOS4-CBMIN4)*SIZE(TWOELECTRON3O,3)+1 ), 2*SIZE(WKAPPA(NOMEGA)%CW,1), & 
            1.0_q , TBSE%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(TBSE%RESPONSEFUN,1) )
          NFLOAT=NFLOAT+NSTRIP4*SIZE(TWOELECTRON3O,3)*NP*NP*8

          IF (LCHI) THEN
             CALL DGEMM('N','T', 2*NP, 2*NP,  SIZE(TWOELECTRON3O,3)*NSTRIP4, 1.0_q, &
               GCHGW(1,1,1), 2*SIZE(GCHGW,1), &
               GCHG (1,1,1), 2*SIZE(GCHG ,1), &
               1.0_q, CHI%RESPONSEFUN(1,1,NOMEGA), 2*SIZE(CHI%RESPONSEFUN,1) )
             NFLOAT=NFLOAT+NSTRIP4*SIZE(TWOELECTRON3O,3)*NP*NP*8
          ENDIF
       ELSE
          ! TBSE(G',G) = <c'_k2+q,n4| e i(G'+q)r | v'_k2,n2> kappa_n2,n4 (G)
          CALL ZGEMM('N','T', NP, NP,  SIZE(TWOELECTRON3O,3)*NSTRIP4, (1.0_q, 0.0_q) , &
            GCHGW(1,1,1), SIZE(GCHGW,1), &
            WKAPPA(NOMEGA)%CW(1,(NPOS4-CBMIN4)*SIZE(TWOELECTRON3O,3)+1 ), SIZE(WKAPPA(NOMEGA)%CW,1), & 
            (1.0_q, 0.0_q) , TBSE%RESPONSEFUN(1,1,NOMEGA), SIZE(TBSE%RESPONSEFUN,1) )
          NFLOAT=NFLOAT+NSTRIP4*SIZE(TWOELECTRON3O,3)*NP*NP*8

          IF (LCHI) THEN
             CALL ZGEMM('N','T', NP, NP,  SIZE(TWOELECTRON3O,3)*NSTRIP4, (1.0_q, 0.0_q) , &
               GCHGW(1,1,1), SIZE(GCHGW,1), &
               GCHG (1,1,1), SIZE(GCHG ,1), &
               (1.0_q, 0.0_q) , CHI%RESPONSEFUN(1,1,NOMEGA), SIZE(CHI%RESPONSEFUN,1) )
             NFLOAT=NFLOAT+NSTRIP4*SIZE(TWOELECTRON3O,3)*NP*NP*8
          ENDIF

       ENDIF
    ENDDO

    ENDDO

    DEALLOCATE(CRHOLM, GCHGW, GCHG, GWORK)
    
  END SUBROUTINE ADD_KAPPA

!****************** SUBROUTINE  CHI_REAL2CMPLX ************************
!
! small helper routine to restore a response function from
! real data layout to complex data layout
! 
!**********************************************************************

  SUBROUTINE CHI_REAL2CMPLX( CHI)
    USE prec
    IMPLICIT NONE
    TYPE (responsefunction) :: CHI ! local field effects from screened exchange
    ! local
    INTEGER N
    INTEGER :: NOMEGA
    COMPLEX(q) :: WORK(SIZE(CHI%RESPONSEFUN,1))
    
    IF (CHI%LREAL) THEN
       DO NOMEGA=1,SIZE(CHI%RESPONSEFUN,3)
          DO N=1,SIZE(CHI%RESPONSEFUN,2)
             WORK=0
             ! go from real to complex storate layout
             CALL ZDAXPY( SIZE(CHI%RESPONSEFUN,1), (1.0_q, 0.0_q), CHI%RESPONSEFUN(1,N,NOMEGA), 1,  WORK(1), 1)
             CHI%RESPONSEFUN(:,N,NOMEGA)=WORK
          ENDDO

          DO N=1,3
             WORK=0
             CALL ZDAXPY( SIZE(CHI%WING,1), (1.0_q, 0.0_q), CHI%WING(1,N,NOMEGA), 1,  WORK(1), 1)
             CHI%WING(:,N,NOMEGA)=WORK

             WORK=0
             CALL ZDAXPY( SIZE(CHI%CWING,1), (1.0_q, 0.0_q), CHI%CWING(1,N,NOMEGA), 1,  WORK(1), 1)
             CHI%CWING(:,N,NOMEGA)=WORK
          ENDDO
          
       ENDDO
    ENDIF
    
  END SUBROUTINE CHI_REAL2CMPLX


!****************** SUBROUTINE  TWOELECTRON4O_ACC_DIRECT ***************
! calculate
!
!    < c_k3,n3 v'_k2,n2 | v |  c'_k4,n4 v_k1,n1  > = 
! x ( int_d3r d3r' c*_k3,n3(r') c'_k4,n4(r') v'*_k2,n2(r) v_k1,n1(r)  W(r',r)
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
! for k4=k2+NQ
!
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_DIRECT(WHF, P, LATT_CUR, ISP, WGW, FSG, &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, &
               ENCUTGW, ENCUTGWSOFT)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG12(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG34(:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    LOGICAL :: LFULL
    REAL(q), OPTIONAL  :: ENCUTGW, ENCUTGWSOFT

    ! set the descriptor for wavefunctions at the point NQ=K2-K1
    ! the charge  v_k1 n1(r') v*_k2 n2(r') = c_k1-k2
    ! is transformed corresponding to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_DIRECT: q-point changed',NQ,NQ_
       STOP
    ENDIF

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG12(NP, NSTRIP1* NSTRIP2), &
         GCHG34(NP, NSTRIP * NSTRIP), &
         CHAM(NSTRIP1, NSTRIP2, NSTRIP, NSTRIP))

    IF (PRESENT(ENCUTGWSOFT)) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF

    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)

    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2))
    ! k1-k2-q might be any reciprocal lattice vector G
    ! in that case the result is shifted by G with respect to the 
    ! internal data layout (FFTEXT), we now apply a shift e^iGr in real space
    ! to cure the problem
    CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==================================================================================
! calculate the Fourier transformed density originating from each pair of states
!  n(G, n1+ (n2-1) max(n1) ) = v'*_k2,n2(r), v_k1,n1(r) * f_k2,n2 f_k1,n1
!                              e^(-i Gr)
!==================================================================================
    GCHG12=0

    DO N1=1,NSTRIP1
       NB1_INTO_TOT=NPOS1-1+N1
       NB1         =NPOS1-VBMIN+N1
       IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

       DO N2=1,NSTRIP2
          NB2_INTO_TOT=NPOS2-1+N2
          NB2         =NPOS2-VBMIN+N2
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

          CALL FOCK_CHARGE_NOINT( W1(NB1), W2(NB2), GWORK(1), CRHOLM, SIZE(CRHOLM))

          ! apply phase factor to shift by reciprocal lattice vector
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

          ! now we have v'*_k2,n2(r'), v_k1,n1(r')
          ! extract using wave function FFT
          CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
               GWORK(1),GCHG12(1,N1+(N2-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
          NFFT=NFFT+1
          IF (MULTIPLY_FRACTIONAL_OCC) THEN
             GCHG12(1:NP,N1+(N2-1)*NSTRIP1)= GCHG12(1:NP,N1+(N2-1)*NSTRIP1) &
               *SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
          ENDIF

          IF (.NOT. LFULL) THEN
             ! multiply with potential factor for simple diagonal kernel such as 4 pi e^2/ G^2 
             CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG12(1,N1+(N2-1)*NSTRIP1), POTFAK(1))
             GCHG12(1:NP,N1+(N2-1)*NSTRIP1)=-CONJG(GCHG12(1:NP,N1+(N2-1)*NSTRIP1))
          ENDIF
       ENDDO
    ENDDO
    
    IF (LFULL) THEN
       DO N1=1, NSTRIP1*NSTRIP2, NSTRIP*NSTRIP
          N2=MIN(NSTRIP1*NSTRIP2-N1+1,NSTRIP*NSTRIP)
          IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
             CALL DGEMM('N','N', 2*NP, N2, 2*NP, 1.0_q, &
                  WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG12(1,N1), 2*SIZE(GCHG12,1), &
                  0.0_q, GCHG34(1,1), 2*SIZE(GCHG34,1) )
          ELSE
             CALL ZGEMM('T','N', NP, N2, NP, (1.0_q,0.0_q), &
                  WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG12(1,N1), SIZE(GCHG12,1), &
                  (0.0_q,0.0_q), GCHG34(1,1), SIZE(GCHG34,1) )
          ENDIF
          CALL ZCOPY( SIZE(GCHG12,1)*N2,  GCHG34(1,1), 1,  GCHG12(1,N1), 1)
       ENDDO
       ! conjugate the results
       GCHG12=-CONJG(GCHG12)
    ENDIF
!    WRITE(*,'(8F14.7)') GCHG12(1:4,1,:)
!    STOP

    ! set the phase factors q'=k3-k4
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4))
    CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
! calculate the density originating from each pair of states
! n(G', n3+(n4-1) max(n3)) =  c*_n3,k3(r')  c_n4,k4(r') (1-f_n3,k3 f_n4, k4)
!                              e^-(i G'r')
!==========================================================================
    DO NPOS3=CBMIN, CBMAX, NSTRIP
       NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

       DO NPOS4=CBMIN4, CBMAX4, NSTRIP
          NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

          GCHG34=0

          DO N4=1,NSTRIP4
             NB4_INTO_TOT=NPOS4-1+N4
             NB4         =NPOS4-CBMIN+N4
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE

             DO N3=1,NSTRIP3
                NB3_INTO_TOT=NPOS3-1+N3
                NB3         =NPOS3-CBMIN+N3
                IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

                CALL FOCK_CHARGE_NOINT( W3(NB3), W4(NB4), GWORK(1), CRHOLM, SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_n3,k3(r')  c*_n4,k4(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG34(1,N3+(N4-1)*NSTRIP),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                ! conjugate to get: c*_n3,k3(r')  c_n4,k4(r')
                GCHG34(1:NP,N3+(N4-1)*NSTRIP)=CONJG(GCHG34(1:NP,N3+(N4-1)*NSTRIP))*(1.0_q/GRIDHF%NPLWV)
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG34(1:NP,N3+(N4-1)*NSTRIP)= GCHG34(1:NP,N3+(N4-1)*NSTRIP)  & 
                     *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))        
                ENDIF
             ENDDO
          ENDDO
!==========================================================================
!    c*_n3,k3(G')  c_n4,k4(G')  w(G'G) v'*_k2,n2(G), v_k1,n1(G)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP2 , SIZE(CHAM,3)*NSTRIP4, m_ NP, one, &
               GCHG12(1,1), m_ SIZE(GCHG12,1), &
               GCHG34(1,1), m_ SIZE(GCHG34,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*SIZE(CHAM,3)*NSTRIP4*NP*8

!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_DIRECT: out of bounds 4 ',NB4
                      STOP
                   ENDIF
                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N2, N3, N4)
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==1) THEN
                         WRITE(78,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N2, N3, N4)
                      ENDIF
#endif
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    DEALLOCATE(CRHOLM, GCHG12, GCHG34, CHAM)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_DIRECT


!****************** SUBROUTINE  TWOELECTRON4O_ACC_EX  ******************
! reasonant anti-resonant coupling exchange term
!
! calculate (compare  TWOELECTRON4O_ACC_EX2))
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
! int_d3r d3r' c_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v*_k1,n1(r) W(r',r)
!
! actually coded is however
! int_d3r d3r' c*_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v_k1,n1(r)  W(r',r)
!
! if we apply time-inversion symmetry, we can use
! v*_k1,n1(r) = v_-k1,n1(r)    c*_k3,n3(r') = c*_-k3,n3(r')
! 
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
! for k4=k2+NQ
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_EX(WHF, P, LATT_CUR, ISP, WGW, FSG, &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, &
               ENCUTGW, ENCUTGWSOFT)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    LOGICAL :: LFULL
    REAL(q), OPTIONAL  :: ENCUTGW, ENCUTGWSOFT

    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX: q-point changed',NQ,NQ_
       STOP
    ENDIF

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NSTRIP1* NSTRIP ), &
         GCHG23(NP, NSTRIP2* NSTRIP ), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))

    IF (PRESENT(ENCUTGWSOFT)) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF

    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)
!==========================================================================
!   c'*_k4,n4(r)  v*_k1,n1(r)
!==========================================================================
    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       ! set phase factor
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4))
       ! k1+k4-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       CALL SETPHASE(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)

       GCHG14=0
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN+N4
          
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NO_CONJG_NOINT( W1(NB1), W4(NB4), GWORK(1), CRHOLM, SIZE(CRHOLM))

             ! apply phase factor e^iGr first (since potential is read for NQ and not K1 +K4)
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+(N4-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=GCHG14(1:NP,N1+(N4-1)*NSTRIP1)  &
                  *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF

             IF (.NOT. LFULL) THEN
             ! multiply with potential factor for simple diagonal kernel such as 4 pi e^2/ G^2 
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+(N4-1)*NSTRIP1), POTFAK(1))
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=-CONJG(GCHG14(1:NP,N1+(N4-1)*NSTRIP1))
             ENDIF
          ENDDO
       ENDDO
       IF (LFULL) THEN
          DO N1=1, NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP
             N2=MIN(NSTRIP1*NSTRIP4-N1+1,NSTRIP2*NSTRIP)
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                CALL DGEMM('N','N', 2*NP, N2, 2*NP, 1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, N2, NP, (1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
             ENDIF
             CALL ZCOPY( SIZE(GCHG14,1)*N2,  GCHG23(1,1), 1,  GCHG14(1,N1), 1)
          ENDDO
          GCHG14=-CONJG(GCHG14)
       ENDIF

       ! set the phase factors q'=-k3-k2
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2))
       CALL SETPHASE(WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c*_k3,n3(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS3=CBMIN, CBMAX, NSTRIP
          NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

          GCHG23=0

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

                CALL FOCK_CHARGE_NO_CONJG_NOINT( W3(NB3), W2(NB2), GWORK(1), CRHOLM, SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_k3,n3(r') v'_k2,n2(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG23(1,N2+(N3-1)*NSTRIP2),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                ! conjugate now to get: c*_k3,n3(r') v'*_k2,n2(r')
                GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=CONJG(GCHG23(1:NP,N2+(N3-1)*NSTRIP2))*(1.0_q/GRIDHF%NPLWV)
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)  &
                     *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))
                ENDIF
             ENDDO
          ENDDO
!==========================================================================
!    c*_k3,n3(r') v'*_k2,n2(r')  c'_k4,n4(r)  v_k1,n1(r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans ,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, m_ NP, one, &
               GCHG14(1,1), m_ SIZE(GCHG14,1), &
               GCHG23(1,1), m_ SIZE(GCHG23,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4* NP*8

!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX: out of bounds 4 ',NB4
                      STOP
                   ENDIF
                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N4, N2, N3)
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
#endif
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG14, GCHG23, CHAM, CRHOLM)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_EX



!****************** SUBROUTINE  TWOELECTRON4O_ACC_EX_INTER *************
! calculate (see comments in TWOELECTRON4O_ACC_EX)
!
! int_d3r d3r' c*_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v_k1,n1(r)  W(r',r)
!
! see comments above
! this version allows to calculate the integral using the W at
! the closest k-points to (k1+k4)
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_EX_INTER(WHF, P, LATT_CUR, ISP, WGW, FSG, &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, &
               B, ENCUTGW, ENCUTGWSOFT) 

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP
    REAL(q) :: B(3,3)

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    REAL(q) :: TMP(3)
    LOGICAL :: LFULL
    REAL(q), OPTIONAL  :: ENCUTGW, ENCUTGWSOFT

    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ= KPOINT_IN_FULL_CLOSEST(WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4),KPOINTS_FULL, B)
    NQ_=KPOINT_IN_FULL_CLOSEST(WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2),KPOINTS_FULL, B)

    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX_INTER: q-point changed',NQ,NQ_
       STOP
    ENDIF

    CALL SETWDES(WGW, WGWQ, NQ)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NSTRIP1* NSTRIP ), &
         GCHG23(NP, NSTRIP2* NSTRIP ), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))


    IF (PRESENT(ENCUTGWSOFT)) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF

    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)
!==========================================================================
!   c'*_k4,n4(r)  v*_k1,n1(r)
!==========================================================================
    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       ! set phase factor
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4))
       ! k1+k4-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       TMP=WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ)
       TMP(1)=ANINT(TMP(1))
       TMP(2)=ANINT(TMP(2))
       TMP(3)=ANINT(TMP(3))
       CALL SETPHASE(TMP, GRIDHF,CPHASE,LPHASE)

       GCHG14=0
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN+N4
          
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NO_CONJG_NOINT( W1(NB1), W4(NB4), GWORK(1), CRHOLM, SIZE(CRHOLM))

             ! apply phase factor e^iGr first (since potential is read for NQ and not K1 +K4)
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+(N4-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=GCHG14(1:NP,N1+(N4-1)*NSTRIP1)  &
                  *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF

             IF (.NOT. LFULL) THEN
             ! multiply with potential factor for simple diagonal kernel such as 4 pi e^2/ G^2 
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+(N4-1)*NSTRIP1), POTFAK(1))
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=-CONJG(GCHG14(1:NP,N1+(N4-1)*NSTRIP1))
             ENDIF
          ENDDO
       ENDDO
       IF (LFULL) THEN
          ! multiply with kernel for non-diagonal kernels 
          DO N1=1, NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP
             N2=MIN(NSTRIP1*NSTRIP4-N1+1,NSTRIP2*NSTRIP)
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                CALL DGEMM('N','N', 2*NP, N2, 2*NP, 1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, N2, NP, (1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
             ENDIF
             CALL ZCOPY( SIZE(GCHG14,1)*N2,  GCHG23(1,1), 1,  GCHG14(1,N1), 1)
          ENDDO
          GCHG14=-CONJG(GCHG14)
       ENDIF

       ! set the phase factors q'=-k3-k2
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2))
       TMP=WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ)
       TMP(1)=ANINT(TMP(1))
       TMP(2)=ANINT(TMP(2))
       TMP(3)=ANINT(TMP(3))
       CALL SETPHASE(TMP, GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c*_k3,n3(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS3=CBMIN, CBMAX, NSTRIP
          NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

          GCHG23=0

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

                CALL FOCK_CHARGE_NO_CONJG_NOINT( W3(NB3), W2(NB2), GWORK(1), CRHOLM, SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_k3,n3(r') v'_k2,n2(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG23(1,N2+(N3-1)*NSTRIP2),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                ! conjugate now to get: c*_k3,n3(r') v'*_k2,n2(r')
                GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=CONJG(GCHG23(1:NP,N2+(N3-1)*NSTRIP2))*(1.0_q/GRIDHF%NPLWV)
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)  &
                     *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))
                ENDIF

             ENDDO
          ENDDO
!==========================================================================
!    c*_k3,n3(r') v'*_k2,n2(r')  c'_k4,n4(r)  v_k1,n1(r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans ,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, m_ NP, one, &
               GCHG14(1,1), m_ SIZE(GCHG14,1), &
               GCHG23(1,1), m_ SIZE(GCHG23,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4* NP*8

!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX: out of bounds 4 ',NB4
                      STOP
                   ENDIF
                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N4, N2, N3)
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
#endif
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG14, GCHG23, CHAM, CRHOLM)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_EX_INTER

!
! this routine is similar to KPOINT_IN_FULL_GRID 
! but finds the closes k-point with zero weight
! it is applied when shifted grids are used
! it assumes that the difference vectors (for which the response function
! is known are stored with zero weight)


  FUNCTION KPOINT_IN_FULL_CLOSEST(VKPT, KPOINTS_F, B)
    !USE sym_prec
    USE lattice
    INTEGER :: KPOINT_IN_FULL_CLOSEST
    REAL(q) VKPT(:)
    TYPE (skpoints_full) :: KPOINTS_F
    REAL(q) B(3,3)    ! reciprocal lattice
    ! local
    REAL(q) T1, T2, T3, V1, V2, V3, DIS, DISMIN
    INTEGER NKFOUND
    INTEGER NK

    ! for shifted meshes we have k-points with zero weight
    ! only for those the W is stored in W????.tmp files
    ! try to seek closest k-point in this set first

    ! no kpoint found, set nk=-1
    DISMIN=100
    NKFOUND=-1
    DO NK=1,KPOINTS_F%NKPTS
       IF (KPOINTS_F%WTKPT(NK)==0) THEN
          T1=MOD(VKPT(1)-KPOINTS_F%VKPT(1,NK)+10.5_q,1._q)-0.5_q
          T2=MOD(VKPT(2)-KPOINTS_F%VKPT(2,NK)+10.5_q,1._q)-0.5_q
          T3=MOD(VKPT(3)-KPOINTS_F%VKPT(3,NK)+10.5_q,1._q)-0.5_q
          
          V1=T1*B(1,1)+T2*B(1,2)+T3*B(1,3)
          V2=T1*B(2,1)+T2*B(2,2)+T3*B(2,3)
          V3=T1*B(3,1)+T2*B(3,2)+T3*B(3,3)
          
          DIS=SQRT(V1*V1+V2*V2+V3*V3)
          IF (DIS < DISMIN) THEN
             DISMIN=DIS
             NKFOUND = NK
          ENDIF
       ENDIF
    ENDDO

    IF (NKFOUND == -1) THEN
    ! for standard meshes 
    ! we need to fall back to conventional k-points
       DO NK=1,KPOINTS_F%NKPTS
          T1=MOD(VKPT(1)-KPOINTS_F%VKPT(1,NK)+10.5_q,1._q)-0.5_q
          T2=MOD(VKPT(2)-KPOINTS_F%VKPT(2,NK)+10.5_q,1._q)-0.5_q
          T3=MOD(VKPT(3)-KPOINTS_F%VKPT(3,NK)+10.5_q,1._q)-0.5_q
          
          V1=T1*B(1,1)+T2*B(1,2)+T3*B(1,3)
          V2=T1*B(2,1)+T2*B(2,2)+T3*B(2,3)
          V3=T1*B(3,1)+T2*B(3,2)+T3*B(3,3)
          
          DIS=SQRT(V1*V1+V2*V2+V3*V3)
          IF (DIS < DISMIN) THEN
             DISMIN=DIS
             NKFOUND = NK
          ENDIF
       ENDDO

       IF (NKFOUND == -1) THEN
          WRITE(*,*) 'internal error in KPOINT_IN_FULL_CLOSEST: no k-point found',KPOINTS_F%NKPTS,B
          STOP
       ENDIF
    ENDIF

    KPOINT_IN_FULL_CLOSEST=NKFOUND
  END FUNCTION KPOINT_IN_FULL_CLOSEST



!****************** SUBROUTINE  TWOELECTRON4O_ACC_EX2 *****************
! reasonant anti-resonant coupling exchange term
!
! calculate
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
! int_d3r d3r' c_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v*_k1,n1(r) W(r',r)
!
! (compared to Hartree resonant anti-resonant coupling
!  < v_k1,n1  v'_k2,n2 | v | c_k3,n3  c'_k4,n4 >   k3 and k4 are exchanged)
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_EX2(WHF, P, LATT_CUR, ISP, WGW, FSG, &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, &
               ENCUTGW, ENCUTGWSOFT)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    LOGICAL :: LFULL
    REAL(q), OPTIONAL  :: ENCUTGW, ENCUTGWSOFT

    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ =KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL)

    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX: q-point changed',NQ,NQ_
       STOP
    ENDIF
    CALL SETWDES(WGW, WGWQ, NQ)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NSTRIP1* NSTRIP ), &
         GCHG23(NP, NSTRIP2* NSTRIP ), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))


    IF (PRESENT(ENCUTGWSOFT)) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF

    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)
!==========================================================================
!   c*'_k4,n4(r) v_k1,n1(r)
!==========================================================================
    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       ! set phase factor (already done by SET_GFAC but possibly destroyed if blocking is used
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4))
       ! average electrostatic potential for k=k' and n=n'
       ! k1+k4-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)

       GCHG14=0
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN+N4
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NOINT( W1(NB1), W4(NB4), GWORK(1), CRHOLM,  SIZE(CRHOLM))

             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+(N4-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1
             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=GCHG14(1:NP,N1+(N4-1)*NSTRIP1)  &
                  *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF

             IF (.NOT. LFULL) THEN
             ! multiply with potential factor for simple diagonal kernel such as 4 pi e^2/ G^2 
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+(N4-1)*NSTRIP1), POTFAK(1))
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=-GCHG14(1:NP,N1+(N4-1)*NSTRIP1)
             ENDIF
          ENDDO
       ENDDO

       IF (LFULL) THEN
          DO N1=1, NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP
             N2=MIN(NSTRIP1*NSTRIP4-N1+1,NSTRIP2*NSTRIP)
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                ! note -1 here and in ZGEMM call
                CALL DGEMM('N','N', 2*NP, N2, 2*NP, -1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, N2, NP, (-1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
             ENDIF
             CALL ZCOPY( SIZE(GCHG14,1)*N2,  GCHG23(1,1), 1,  GCHG14(1,N1), 1)
          ENDDO
       ENDIF

       ! set the phase factors q'=-k3-k2
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2))
       CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c_k3,n3(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS3=CBMIN, CBMAX, NSTRIP
          NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

          GCHG23=0

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

                CALL FOCK_CHARGE_NOINT( W3(NB3), W2(NB2), GWORK(1), CRHOLM,  SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_k3,n3(r') v'*_k2,n2(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG23(1,N2+(N3-1)*NSTRIP2),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)*(1.0_q/GRIDHF%NPLWV)
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)  &
                     *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))
                ENDIF

             ENDDO
          ENDDO
!==========================================================================
!    c_k3,n3(r') v'*_k2,n2(r')  c'_k4,n4(r)  v*_k1,n1(r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, m_ NP, one, &
               GCHG14(1,1), m_ SIZE(GCHG14,1), &
               GCHG23(1,1), m_ SIZE(GCHG23,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4*NP*8

!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4) .OR. NB4 <= 0) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX2: out of bounds 4 ',NB4
                      STOP
                   ENDIF

                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N4, N2, N3)
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
#endif
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG14, GCHG23, CRHOLM, CHAM)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_EX2


!****************** SUBROUTINE  TWOELECTRON4O_ACC_EX2_INTER************
! reasonant anti-resonant coupling exchange term
!
! calculate
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
! int_d3r d3r' c_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v*_k1,n1(r) W(r',r)
!
! (compared to Hartree resonant anti-resonant coupling
!  < v_k1,n1  v'_k2,n2 | v | c_k3,n3  c'_k4,n4 >   k3 and k4 are exchanged)
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_EX2_INTER(WHF, P, LATT_CUR, ISP, WGW, FSG, &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP, & 
               B, ENCUTGW, ENCUTGWSOFT)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP
    REAL(q) :: B(3,3)
  
! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    REAL(q) :: TMP(3)
    LOGICAL :: LFULL
    REAL(q), OPTIONAL :: ENCUTGW, ENCUTGWSOFT

    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ= KPOINT_IN_FULL_CLOSEST(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL, B)
    NQ_=KPOINT_IN_FULL_CLOSEST(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL, B)

    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX: q-point changed',NQ,NQ_
       STOP
    ENDIF
    CALL SETWDES(WGW, WGWQ, NQ)

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band

    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NSTRIP1* NSTRIP ), &
         GCHG23(NP, NSTRIP2* NSTRIP ), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))

    IF (PRESENT(ENCUTGWSOFT)) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF

    LFULL=.FALSE.
    IF (ASSOCIATED(WPOTH)) CALL GET_WPOT( WPOTH, NQ, POTFAK, LFULL)
!==========================================================================
!   c*'_k4,n4(r) v_k1,n1(r)
!==========================================================================
    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       ! set phase factor (already done by SET_GFAC but possibly destroyed if blocking is used
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4))
       ! average electrostatic potential for k=k' and n=n'
       ! k1+k4-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       TMP=WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ)
       TMP(1)=ANINT(TMP(1))
       TMP(2)=ANINT(TMP(2))
       TMP(3)=ANINT(TMP(3))
       CALL SETPHASE(TMP, GRIDHF,CPHASE,LPHASE)

       GCHG14=0
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN+N4
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NOINT( W1(NB1), W4(NB4), GWORK(1), CRHOLM,  SIZE(CRHOLM))

             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+(N4-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=GCHG14(1:NP,N1+(N4-1)*NSTRIP1)  &
                     *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF

             IF (.NOT. LFULL) THEN
             ! multiply with potential factor for simple diagonal kernel such as 4 pi e^2/ G^2 
                CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+(N4-1)*NSTRIP1), POTFAK(1))
                GCHG14(1:NP,N1+(N4-1)*NSTRIP1)=-GCHG14(1:NP,N1+(N4-1)*NSTRIP1)

             ENDIF
          ENDDO
       ENDDO

       IF (LFULL) THEN
          DO N1=1, NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP
             N2=MIN(NSTRIP1*NSTRIP4-N1+1,NSTRIP2*NSTRIP)
             IF (WPOTH%WPOT(NQ)%LREALSTORE) THEN
                ! note -1 here and in ZGEMM call
                CALL DGEMM('N','N', 2*NP, N2, 2*NP, -1.0_q, &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), 2*SIZE(GCHG14,1), &
                     0.0_q, GCHG23(1,1), 2*SIZE(GCHG23,1) )
             ELSE
                CALL ZGEMM('T','N', NP, N2, NP, (-1.0_q,0.0_q), &
                     WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG14(1,N1), SIZE(GCHG14,1), &
                     (0.0_q,0.0_q), GCHG23(1,1), SIZE(GCHG23,1) )
             ENDIF
             CALL ZCOPY( SIZE(GCHG14,1)*N2,  GCHG23(1,1), 1,  GCHG14(1,N1), 1)
          ENDDO
       ENDIF

       ! set the phase factors q'=-k3-k2
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2))
       TMP=WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ)
       TMP(1)=ANINT(TMP(1))
       TMP(2)=ANINT(TMP(2))
       TMP(3)=ANINT(TMP(3))
       CALL SETPHASE(TMP, GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c_k3,n3(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS3=CBMIN, CBMAX, NSTRIP
          NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

          GCHG23=0

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN+N3
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP))) CYCLE

                CALL FOCK_CHARGE_NOINT( W3(NB3), W2(NB2), GWORK(1), CRHOLM,  SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_k3,n3(r') v'*_k2,n2(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG23(1,N2+(N3-1)*NSTRIP2),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)*(1.0_q/GRIDHF%NPLWV)
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)  &
                     *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))
                ENDIF

             ENDDO
          ENDDO
!==========================================================================
!    c_k3,n3(r') v'*_k2,n2(r')  c'_k4,n4(r)  v*_k1,n1(r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, m_ NP, one, &
               GCHG14(1,1), m_ SIZE(GCHG14,1), &
               GCHG23(1,1), m_ SIZE(GCHG23,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4*NP*8

!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4) .OR. NB4 <= 0) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_EX2: out of bounds 4 ',NB4
                      STOP
                   ENDIF

                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N4, N2, N3)
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
#endif
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG14, GCHG23, CRHOLM, CHAM)
!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_EX2_INTER


!****************** SUBROUTINE  TWOELECTRON4O_ACC_HARTREE  ************
! calculate Hartree terms
! TWOELECTRON3O( n1, n3 ,  n2, n4)=
!  < c_k3,n3 v'_k2,n2 | v | v_k1,n1  c'_k4,n4 > = 
!  int_d3r d3r' v'*_k2,n2(r') c'_k4,n4(r') c*_k3,n3(r) v_k1,n1(r) v(r',r)
!
! note: in principle the interaction between resonant anti-resonant
! involves a term ( k1,k3 pair is exchanged in the resonant anti-resonant 
! coupling)
!  < v_k1,n1  v'_k2,n2 | v | c_k3,n3  c'_k4,n4 > 
! however using time inversion symmetry we can exchange the pair back to
! yield exactly
!  < c_k3,n3  v'_k2,n2 | v | v_k1,n1  c'_k4,n4 > 
!
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
! for k4=k2+NQ
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_HARTREE(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
               FEX, LCONJG, FMUL,  &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP, ISP2
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    LOGICAL :: FEX                  ! include DFT exchange
    REAL(q) :: FMUL                 ! multiplicty 
    LOGICAL :: LCONJG               ! add conjugated term
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG13(:,:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG24(:,:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV)), GCHG (MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! Hamiltonian matrix
    LOGICAL LPHASE
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    REAL(q) :: HFSCREEN_SAVE, AEXX_SAVE, HFRCUT_SAVE
    LOGICAL :: L_MODEL_HF_SAVE

    IF (FMUL==0) RETURN
! use full Coulomb kernel
    HFRCUT_SAVE  =HFRCUT
    HFSCREEN_SAVE=HFSCREEN
    AEXX_SAVE=AEXX
    L_MODEL_HF_SAVE=L_MODEL_HF

    MODEL_GW=0   ! switch off all model dielectric functions
    HFSCREEN=0   ! switch of screening
    AEXX=1       ! full exchange 
    L_MODEL_HF=.FALSE.

    ! set the descriptor for charge at the point k1-k3 = q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3),KPOINTS_FULL)
    ! set descriptor for the wavefunction at that particular q-point (wave.F)
    CALL SETWDES(WGW, WGWQ, NQ)

    ! set the descriptor for charge at the point k2-k4 = q'
    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_HARTREE: q-point changed',NQ,NQ_
       STOP
    ENDIF

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR  ! number of G-vectors in the basis (local)

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG13(NP, NSTRIP1, NSTRIP), &
         GCHG24(NP, NSTRIP2, NSTRIP), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))

    ! use bare Coulomb operator without singularity correction (see few lines above)
    ! where AEXX is set to 1
    ! SET_GFAC_WITHOUT_WEIGHT in fock.F which in turn calls SET_GFAC
    ! to setup the  Coloumb kernel  4 pi e^2 / (G+k1-(k1+q))**2 * screening
    ! or if LCONJG is set 4 pi e^2 / (G+k1+(k1+q))**2 * screening
    CALL SET_GFAC_WITHOUT_WEIGHT(GRIDHF,LATT_CUR,K1,K3,0.0_q,POTFAK)
!==========================================================================
!   c*_k3,n3(r)  v_k1,n1(r)
!==========================================================================
    DO NPOS3=CBMIN, CBMAX, NSTRIP
       NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

       ! set phase factor (already done by SET_GFAC but possibly destroyed if blocking is used
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3))
       ! k1+k3-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
       
       GCHG13=0
       
       DO N3=1,NSTRIP3
          NB3_INTO_TOT=NPOS3-1+N3
          NB3         =NPOS3-CBMIN+N3
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NOINT( W1(NB1), W3(NB3), GWORK(1), CRHOLM,  SIZE(CRHOLM))
#ifndef gammareal
             ! conjugate charge
             IF (LCONJG) THEN
                GWORK=GWORK+GCONJG(GWORK)
             ENDIF
#endif
             GCHG=GWORK

             ! fft to reciprocal space
             CALL FFT3D(GWORK(1),GRIDHF,-1)
             NFFT=NFFT+1
             ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL APPLY_GFAC(GRIDHF, GWORK(1), POTFAK(1))
             ! back to real space to get  \int psi_q(r) psi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1),GRIDHF,1)
             NFFT=NFFT+1

             ! add DFT contribution from exchange correlation f_xc
             ! this is exactly zero if LGWLF is set
             IF (FEX) THEN
! LF_ADD_MUL:  GWORK(1:GRIDHF%RL%NP)=GWORK(1:GRIDHF%RL%NP)+GCHG(1:GRIDHF%RL%NP)*FXCR(1:GRIDHF%RL%NP)
                CALL LF_ADD_MUL( GWORK(1), GCHG(1), FXCR(1), GRIDHF%RL%NP)
             ENDIF
             ! apply phase factor e^iGr if required
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG13(1,N1,N3),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1
             
             GCHG13(1:NP,N1,N3)=CONJG(GCHG13(1:NP,N1,N3))*(1.0_q/GRIDHF%NPLWV)
             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG13(1:NP,N1,N3)=GCHG13(1:NP,N1,N3) & 
                  *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF

          ENDDO
       ENDDO
       
       ! set the phase factors q'=k2-k4
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4))
       CALL SETPHASE(WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c'*_k4,n4(r') v'_k2,n2(r')
!==========================================================================
       DO NPOS4=CBMIN4, CBMAX4, NSTRIP
          NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

          GCHG24=0

          DO N4=1,NSTRIP4
             NB4_INTO_TOT=NPOS4-1+N4
             NB4         =NPOS4-CBMIN+N4
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP2))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP2))) CYCLE

                CALL FOCK_CHARGE_NOINT( W2(NB2), W4(NB4), GWORK(1), CRHOLM,  SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have v'_k2,n2(r') c'*_k4,n4(r') 
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG24(1,N2,N4),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                !  conjugate c'_k4,n4(r') v'*_k2,n2(r')
                GCHG24(1:NP,N2,N4)=CONJG(GCHG24(1:NP,N2,N4))*(1.0_q/GRIDHF%NPLWV)
                ! fractional occupancies 
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG24(1:NP,N2,N4)=GCHG24(1:NP,N2,N4) &
                     *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP)))
                ENDIF
             ENDDO
          ENDDO
!==========================================================================
!  ( int_d3r d3r' v'*_k2,n2(r') c'_k4,n4(r') c*_k3,n3(r) v_k1,n1(r) v(r',r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP3, NSTRIP2*NSTRIP4, m_ NP, one, &
               GCHG13(1,1,1), m_ SIZE(GCHG13,1), &
               GCHG24(1,1,1), m_ SIZE(GCHG24,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))

          NFLOAT=NFLOAT+NSTRIP1*NSTRIP3*NSTRIP2*NSTRIP4*NP*8
!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_HARTREE: out of bounds 4 ',NB4
                      WRITE(*,*) 'when it should be max:', SIZE(TWOELECTRON3O,4)
                      STOP
                   ENDIF

                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N3, N2, N4)*FMUL
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N3, N2, N4)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N3, N2, N4)
                      ENDIF
#endif

                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG13, GCHG24, CHAM, CRHOLM)

!  restore semilocal exchange correlation
    AEXX=AEXX_SAVE
    HFSCREEN=HFSCREEN_SAVE
    HFRCUT=HFRCUT_SAVE
    L_MODEL_HF=L_MODEL_HF_SAVE

  END SUBROUTINE TWOELECTRON4O_ACC_HARTREE

!****************** SUBROUTINE  TWOELECTRON4O_ACC_HEAD     ************
!
! calculate the head (singular contribution) related to the 
! Hartree-term
! see above subroutine for details on indexing
! 7
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_HEAD(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
               FEX, LCONJG, FMUL,  &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, NFLOAT, NFFT, NSTRIP)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP, ISP2
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    LOGICAL :: FEX                  ! include DFT exchange
    REAL(q) :: FMUL                 ! multiplicty 
    LOGICAL :: LCONJG               ! add conjugated term
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB3, NB4
    INTEGER NQ

    GDEF :: CDER_BETWEEN_STATE13(3),CDER_BETWEEN_STATE24(3)
    INTEGER :: K1_IN_FULL_ORIG,  K2_IN_FULL_ORIG

    IF (FMUL==0) RETURN
    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3),KPOINTS_FULL)

    IF (K1/=K3 .OR. .NOT. LHEAD_DIRECT) RETURN

    K1_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)
    K2_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,k2),KPOINTS_FULL_ORIG)

    ! TODO: maybe remove outer blocking, one loop N3 from CBMIN to CBMAX
    DO NPOS3=CBMIN, CBMAX, NSTRIP
       NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

       ! TODO: maybe remove outer blocking, one loop N4 from CBMIN4 to CBMAX4
       DO NPOS4=CBMIN4, CBMAX4, NSTRIP
          NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_HARTREE: out of bounds 4 ',NB4
                      WRITE(*,*) 'when it should be max:', SIZE(TWOELECTRON3O,4)
                      STOP
                   ENDIF
                   ! gK:  lhead_direct
                   IF (K2_IN_FULL_ORIG /=0) THEN
                      CALL  CDER_BETWEEN_STATES_ROTATED_NOERR( &
                           CDER_BETWEEN_STATE24,LATT_CUR, K2_IN_FULL_ORIG, ISP, NB2_INTO_TOT , NB4_INTO_TOT )
                   ENDIF

                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      ! gK:  lhead_direct
                      IF (K1_IN_FULL_ORIG/=0) THEN
                         CALL CDER_BETWEEN_STATES_ROTATED_NOERR( &
                              CDER_BETWEEN_STATE13,LATT_CUR, K1_IN_FULL_ORIG, ISP, NB3_INTO_TOT, NB1_INTO_TOT)
                         TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+(CDER_BETWEEN_STATE24(1)*CDER_BETWEEN_STATE13(1))*(EDEPS/LATT_CUR%OMEGA)*FMUL
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE TWOELECTRON4O_ACC_HEAD


!****************** SUBROUTINE  ONEELECTRON_TRANS     ***************
!
! calculate the transition element
!
! < c_k3,n3| e^i(k3-k1) | v_k1,n1 > = < c_k3,n3| e^iq | v_k1,n1 >
!
! n1 and k1 is occupied and n3 and k3 is unoccupied, and k3=k1+q
! 
!**********************************************************************

 SUBROUTINE ONEELECTRON_TRANS(WHF, P, LATT_CUR, ISP, WGW, & 
               W1, K1, VBMIN, VBMAX, &
               W3, K3, CBMIN, CBMAX, KPOINT_BSE, NFFT )

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    TYPE (wavefun1)    W1(:), W3(:)
    INTEGER K1, K3, NK1_IN_KPOINTS_FULL_ORIG
    INTEGER :: CBMIN, CBMAX, VBMIN, VBMAX
    INTEGER              KPOINT_BSE(3) ! reciprocal lattice vector g=G+q at which response is evaluated
    REAL(q) :: NFFT

! local variables
    INTEGER N1, NB1_INTO_TOT, NB3_INTO_TOT
    INTEGER N3, NB1, NB3
    TYPE (wavedes1)    WGWQ
    INTEGER NQ
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    LOGICAL LPHASE
    COMPLEX(q), EXTERNAL :: CRHO0

    ! set the descriptor for charge at the point k1-k3 = q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3),KPOINTS_FULL)

    ! determined the index of this k-point in the original full k-point grid
    ! this is the negative of the k-point supplied by the user
    NK1_IN_KPOINTS_FULL_ORIG=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1),KPOINTS_FULL_ORIG)

    ALLOCATE(CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS))

    ! set phase factor
    CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3))
    ! k1-k3-q might be any reciprocal lattice vector G
    ! in that case the result is shifted by G 
    ! we apply a shift e^iGr in real space to cure the problem
    CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,NQ)+KPOINT_BSE, GRIDHF,CPHASE,LPHASE)

    WRITE(*,*) 'evaluating response at ',WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,NQ)+KPOINT_BSE

!==========================================================================
!   c*_k3,n3(r)  v_k1,n1(r)
!==========================================================================
    DO N3=CBMIN, CBMAX
       
       NB3_INTO_TOT=N3
       NB3         =N3-CBMIN+1 ! local index into W3
       IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE
       
       DO N1=VBMIN,VBMAX
          NB1_INTO_TOT=N1
          NB1         =N1-VBMIN+1 ! ! local index into W1
          IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE
          ! calculate < c_k3,n3| r | v_k1,n1 >
          CALL FOCK_CHARGE_NOINT( W1(NB1), W3(NB3), GWORK(1), CRHOLM,  SIZE(CRHOLM))

          ! apply phase factor e^iGr if required
          IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
          
          ! fft to reciprocal space
          ! comment: fft could be avoided by summing the density on the real space grid
          CALL FFT3D(GWORK(1),GRIDHF,-1)
          NFFT=NFFT+1

          ! store < c_k3,n3| e^i(k3-k1 ) | v_k1,n1 >
          CALL  SET_CDER_BETWEEN_STATES_Q( (CRHO0(GRIDHF, GWORK)*(1.0_q/GRIDHF%NPLWV)), &
               NK1_IN_KPOINTS_FULL_ORIG, ISP, NB3_INTO_TOT, NB1_INTO_TOT)
       ENDDO
    ENDDO
    DEALLOCATE(CRHOLM)
    
  END SUBROUTINE ONEELECTRON_TRANS

!****************** SUBROUTINE  TWOELECTRON4O_ACC_HARTREE_SFLIP *******
!
! calculate Hartree terms allowing for spin flips
! and specifically other valence and conduction band windows
! for spin up and down
!
!  < c_k3,n3 v'_k2,n2 | v | v_k1,n1  c'_k4,n4 > = 
!  int_d3r d3r' v'*_k2,n2(r') c'_k4,n4(r') c*_k3,n3(r) v_k1,n1(r) v(r',r)
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
! for k4=k2+NQ
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! this is essentially identical to the above routine, but
! for the additional arguments CBMIN42, CBMAX42 that describe
! the currently considered block for the orbitals with the 4th index
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_HARTREE_SFLIP(WHF, P, LATT_CUR, ISP, ISP2, WGW, & 
               FEX, LCONJG, FMUL,  &
               W1, K1, NPOS1, NSTRIP1, W2, K2, NPOS2, NSTRIP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN, & 
               CBMIN2, CBMIN42, CBMAX42, VBMIN2, NFLOAT, NFFT, NSTRIP)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    USE pseudo
    IMPLICIT NONE

! passed variables
    TYPE (latt) LATT_CUR
    TYPE (wavespin) WHF
    TYPE (potcar)      P(:)
    INTEGER ISP, ISP2
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    LOGICAL :: FEX                  ! include DFT exchange
    REAL(q) :: FMUL                 ! multiplicty 
    LOGICAL :: LCONJG               ! add conjugated term
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEF :: TWOELECTRON3O(:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN4, CBMAX4, VBMIN
    INTEGER :: CBMIN2, CBMIN42, CBMAX42, VBMIN2
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG13(:,:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG24(:,:,:)    ! charge 
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV)), GCHG (MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    REAL(q) :: HFSCREEN_SAVE, AEXX_SAVE, HFRCUT_SAVE
    LOGICAL :: L_MODEL_HF_SAVE

    IF (FMUL==0) RETURN
! use full Coulomb kernel
    HFRCUT_SAVE  =HFRCUT
    HFSCREEN_SAVE=HFSCREEN
    AEXX_SAVE=AEXX
    L_MODEL_HF_SAVE=L_MODEL_HF

    MODEL_GW=0   ! switch off all model dielectric functions
    HFSCREEN=0   ! switch of screening
    AEXX=1       ! full exchange 
    L_MODEL_HF=.FALSE.

    ! set the descriptor for charge at the point k1-k3 = q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

    ! set the descriptor for charge at the point k2-k4 = q
    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_HARTREE: q-point changed',NQ,NQ_
       STOP
    ENDIF

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP=WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG13(NP, NSTRIP1, NSTRIP), &
         GCHG24(NP, NSTRIP2, NSTRIP), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))

    ! use bare Coulomb operator without singularty correction (see few lines above)
    ! where AEXX is set to 1
    ! no singularity correction
    CALL SET_GFAC_WITHOUT_WEIGHT(GRIDHF,LATT_CUR,K1,K3,0.0_q,POTFAK)
!==========================================================================
!   c*_k3,n3(r)  v_k1,n1(r)
!==========================================================================
    DO NPOS3=CBMIN, CBMAX, NSTRIP
       NSTRIP3=MIN(CBMAX+1-NPOS3,NSTRIP)

       ! set phase factor (already done by SET_GFAC but possibly destroyed if blocking is used
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3))
       ! k1+k3-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
       
       GCHG13=0
       
       DO N3=1,NSTRIP3
          NB3_INTO_TOT=NPOS3-1+N3
          NB3         =NPOS3-CBMIN+N3
          IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP))) CYCLE

             CALL FOCK_CHARGE_NOINT( W1(NB1), W3(NB3), GWORK(1), CRHOLM,  SIZE(CRHOLM))
#ifndef gammareal
             ! conjugate charge
             IF (LCONJG) THEN
                GWORK=GWORK+GCONJG(GWORK)
             ENDIF
#endif
             GCHG=GWORK

             ! fft to reciprocal space
             CALL FFT3D(GWORK(1),GRIDHF,-1)
             NFFT=NFFT+1
             ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
             CALL APPLY_GFAC(GRIDHF, GWORK(1), POTFAK(1))
             ! back to real space to get  \int psi_q(r) psi_k(r) / (r-r') d3r
             CALL FFT3D(GWORK(1),GRIDHF,1)
             NFFT=NFFT+1

             ! add DFT contribution from exchange correlation f_xc
             ! this is exactly zero if LGWLF is set
             IF (FEX) THEN
! LF_ADD_MUL:  GWORK(1:GRIDHF%RL%NP)=GWORK(1:GRIDHF%RL%NP)+GCHG(1:GRIDHF%RL%NP)*FXCR(1:GRIDHF%RL%NP)
                CALL LF_ADD_MUL( GWORK(1), GCHG(1), FXCR(1), GRIDHF%RL%NP)
             ENDIF
             ! apply phase factor e^iGr if required
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG13(1,N1,N3),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             GCHG13(1:NP,N1,N3)=CONJG(GCHG13(1:NP,N1,N3))*(1.0_q/GRIDHF%NPLWV)
             IF (MULTIPLY_FRACTIONAL_OCC) THEN
                GCHG13(1:NP,N1,N3)=GCHG13(1:NP,N1,N3) & 
                  *SQRT(ABS(1-WHF%FERTOT(NB3_INTO_TOT,K3,ISP)))*SQRT(ABS(WHF%FERTOT(NB1_INTO_TOT,K1,ISP)))
             ENDIF
          ENDDO
       ENDDO

       ! set the phase factors q'=k2-k4
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4))
       CALL SETPHASE(WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c'*_k4,n4(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS4=CBMIN42, CBMAX42, NSTRIP
          NSTRIP4=MIN(CBMAX42+1-NPOS4,NSTRIP)

          GCHG24=0

          DO N4=1,NSTRIP4
             NB4_INTO_TOT=NPOS4-1+N4
             NB4         =NPOS4-CBMIN2+N4
             IF (FILLED_XI_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP2))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN2+N2
                IF (EMPTY_XI_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP2))) CYCLE

                CALL FOCK_CHARGE_NOINT( W2(NB2), W4(NB4), GWORK(1), CRHOLM,  SIZE(CRHOLM))

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have v'_k2,n2(r') c'*_k4,n4(r') 
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG24(1,N2,N4),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                !  conjugate c'_k4,n4(r') v'*_k2,n2(r')
                GCHG24(1:NP,N2,N4)=CONJG(GCHG24(1:NP,N2,N4))*(1.0_q/GRIDHF%NPLWV)
                !  and eveluate fractional occupancies
                IF (MULTIPLY_FRACTIONAL_OCC) THEN
                   GCHG24(1:NP,N2,N4)=GCHG24(1:NP,N2,N4) &
                     *SQRT(ABS(1-WHF%FERTOT(NB4_INTO_TOT,K4,ISP2)))*SQRT(ABS(WHF%FERTOT(NB2_INTO_TOT,K2,ISP2)))
                ENDIF

             ENDDO
          ENDDO
!==========================================================================
!  ( int_d3r d3r' v'*_k2,n2(r') c'_k4,n4(r') c*_k3,n3(r) v_k1,n1(r) v(r',r)
!==========================================================================
          CHAM=0

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP3, NSTRIP2*NSTRIP4, m_ NP, one, &
               GCHG13(1,1,1), m_ SIZE(GCHG13,1), &
               GCHG24(1,1,1), m_ SIZE(GCHG24,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))

          NFLOAT=NFLOAT+NSTRIP1*NSTRIP3*NSTRIP2*NSTRIP4*NP*8
!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             DO N2=1,NSTRIP2
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN42+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,4)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_HARTREE_SF: out of bounds 4 ',NB4
                      WRITE(*,*) 'when it should be max:', SIZE(TWOELECTRON3O,4)
                      STOP
                   ENDIF
                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=NPOS3-1+N3
                      NB3         =NPOS3-CBMIN+N3
                      TWOELECTRON3O( N1, NB3 , N2, NB4)=TWOELECTRON3O( N1, NB3 , N2, NB4)+CHAM( N1, N3, N2, N4)*FMUL
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N3, N2, N4)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N3, N2, N4)
                      ENDIF
#endif

                   ENDDO
                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    DEALLOCATE(GCHG13, GCHG24, CHAM, CRHOLM)

!  restore semilocal exchange correlation
    AEXX=AEXX_SAVE
    HFSCREEN=HFSCREEN_SAVE
    HFRCUT=HFRCUT_SAVE
    L_MODEL_HF=L_MODEL_HF_SAVE

  END SUBROUTINE TWOELECTRON4O_ACC_HARTREE_SFLIP

END MODULE local_field

!****************** SUBROUTINE  LF_MUL ********************************
!
! small helper routine to multiply charge with potential
! using GDEF valued arrays
! 
!**********************************************************************

SUBROUTINE LF_ADD_MUL( GWORK, GCHG, FXCR, N)
  USE prec
  IMPLICIT NONE
  INTEGER :: N
  GDEF :: GWORK(N), GCHG(N), FXCR(N)

  INTEGER :: I
  
  DO I=1,N
     GWORK(I)=GWORK(I)+GCHG(I)*FXCR(I)
  ENDDO
  
END SUBROUTINE LF_ADD_MUL

SUBROUTINE LF_COPY( FXC, FXCR, N)
  USE prec
  IMPLICIT NONE
  INTEGER :: N
  GDEF :: FXC(N), FXCR(N)

  INTEGER :: I
  
  DO I=1,N
     FXCR(I)=FXC(I)
  ENDDO
  
END SUBROUTINE LF_COPY
