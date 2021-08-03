#include "symbol.inc"
MODULE mp2kpar
  USE prec
  USE fock
  USE twoelectron4o
  USE chi_base
  USE wpot
  USE lattice
  IMPLICIT NONE

!**********************************************************************
!
! this subroutine handles MP2 using similar routines as coded in
! local_field.F
! it is more efficient than the twoelectron4o routine, in 
! particular if ENCUTGW is set but the routine
! also reduces the number of FFTs significantly typically by
! a factor filled bands/2 compared to the twoelectron4o routine.
! Last not least it reduces significantly the memory requirements.
! 
! There are however some caveats to this routine. 
! One problem is that the difference vectors between any two
! k-points must be included in the k-point set. This requires to
! use Gamma centered meshes.
!
! The central quantity is:
! 
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
!  int_d3r d3r' c_k1-q,n3(r')  v'*_k2,n2(r') c'_k2+q,n4(r)  v*_k1,n1(r)  W(r',r)
!
! where W is the bare or a screened interaction
! most of the comments in the local_field.F routine apply to 
! this routine is well
!              
!**********************************************************************

  LOGICAL, PRIVATE :: LKPOINT_PARALLEL=.FALSE. ! parallelization over k-points
! test
! LOGICAL, PRIVATE :: LFORCE_DISTRIBUTE_W3=.FALSE.
  LOGICAL, PRIVATE :: LFORCE_DISTRIBUTE_W3=.TRUE.
! test

  INTEGER :: NBINS
  REAL(q) :: EKIN_BINSIZE
  REAL(q), ALLOCATABLE :: ECOR_HARTREE(:), ECOR_EXCHANGE(:), ECOR(:)

  REAL(q), PRIVATE, SAVE :: SCALE_HEAD=1.0_q
  CONTAINS


!*********************************************************************
!
! main calculational procedure
!
!*********************************************************************

  SUBROUTINE CALCULATE_MP2_KPAR( &
          P,WDES,W,LATT_CUR,T_INFO,IO,KPOINTS,WGW, ENCUTGW, ENCUTGWSOFT, LMAXMP2)

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
! test_
    USE radial
! test_
    USE c2f_interface, ONLY : ERRF
    IMPLICIT NONE
! structures
    TYPE (type_info)     T_INFO
    TYPE (potcar)        P(T_INFO%NTYP)
    TYPE (wavedes)       WDES
    TYPE (nonl_struct)   NONL_S
    TYPE (wavespin)      W
    TYPE (latt)          LATT_CUR
    TYPE (in_struct)     IO
    TYPE (kpoints_struct) KPOINTS
    TYPE (wavedes)       WGW         ! descriptor for basis set of response function
    REAL(q)              ENCUTGW, ENCUTGWSOFT
    INTEGER              LMAXMP2     ! maximum L index for one center terms
! local
    GDEFS, ALLOCATABLE :: TWOELECTRON3O(:,:,:,:,:)
    TYPE (wavespin) WHF
    TYPE (wavefun1),ALLOCATABLE :: W1(:), W2(:), W3(:), W4(:)
    TYPE (wavefun1) :: WTMP
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2, WDESK3, WDESK4
    INTEGER :: IERR              ! error indicator
    INTEGER :: NSTRIP            ! block size
    INTEGER :: NGLB              ! block size in conduction band
    INTEGER :: NGLB3             ! block size in conduction band index 3
    INTEGER :: NGLB4             ! block size in conduction band index 4 
    INTEGER :: NSTRIPV           ! block size in valence band
    INTEGER :: N, NP
    INTEGER :: NPOS1, NSTRIP1    ! base index and width of the n1 block
    INTEGER :: NPOS2, NSTRIP2    ! base index and width of the n1 block
    INTEGER :: NKMAX_4O          ! 3rd dimension of 4O integrals (1 or 2)
    INTEGER :: K1,  K3, K4, K2, K4_BASE, K4_COLLECT, K4_DONE, K2_LOCAL, K3_IN_FULL_ORIG
    INTEGER :: VBMIN, VBMAX      ! band index for the VB minimum and VB maximum
    INTEGER :: CBMIN, CBMAX      ! band index for the CB minimum and CB maximum
    INTEGER :: CBMIN3, CBMAX3    ! band index for the CB minimum and CB maximum for n4
                                 ! this might be reduced for parallelization over bands
    INTEGER :: CBMIN4, CBMAX4    ! band index for the CB minimum and CB maximum for n4
                                 ! this might be reduced for parallelization over bands
    REAL(q) :: SCALE
    REAL(q) :: NFFTW             ! number of FFTs for  wavefunctions
    REAL(q) :: NFLOAT4O, NFFT4O  ! number of BLAS3 operations in 4 orbital routines
    REAL(q) :: DELTA
    INTEGER LAST_FILLED, FIRST_EMPTY
    LOGICAL :: W1EQUALW2         ! W1 and W2, and W3 and W4 are strictly equal
  ! MP2 part 
    INTEGER :: N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT, ISP, ISP1, ISP2
    INTEGER :: I, J
    COMPLEX(q) :: EFOCK
    REAL(q), POINTER :: EKIN(:,:,:), EKIN_MAX(:,:,:)
  ! one center terms
    TYPE (one_center_handle), POINTER :: H
    
    CHARACTER (LEN=1) :: SP(2)=(/ "u", "d" /)
    
! test
    REAL(q) :: OMEGBK,QC

    IF (LRHFCALC.AND.LRSCOR) THEN
       CALL CELVOL(LATT_CUR%B(1,1),LATT_CUR%B(1,2),LATT_CUR%B(1,3),OMEGBK)
       QC=TPI*(0.75_q/PI*OMEGBK/KPOINTS_FULL%NKPTS)**(1._q/3._q)
       SCALE_HEAD=8*PI*HFSCREEN*HFSCREEN* &
      &   (-EXP(-QC*QC/4/HFSCREEN/HFSCREEN)*QC+HFSCREEN*SQRT(PI)*ERRF(QC/2/HFSCREEN))/ &
      &   (OMEGBK*TPI*TPI*TPI/KPOINTS_FULL%NKPTS)
       IF (IO%IU6>=0) THEN
          WRITE(*,'(A,F10.5)') 'scale head by :',SCALE_HEAD
       ENDIF
    ENDIF
! test

!=======================================================================
! preparation of four-orbital related quantities
!=======================================================================
    IF (LMAXMP2>=0) THEN
       CALL SET_UP_ONE_CENTER_H( WDES, P, T_INFO, LMAXMP2, H)
    ENDIF

    CALL CHECK_FULL_KPOINTS ! all set up properly ?
    CALL START_TIMING("LOOP")

    WHF=W      ! use temporarily another WDES
    WHF%WDES => WDES_FOCK
130 FORMAT (5X, //, &
         &'----------------------------------------------------', &
         &'----------------------------------------------------'//)

    IF (IO%IU6>=0) WRITE(IO%IU6,130)

    CALL SETWDES(WHF%WDES,WDESK1,0)
    CALL SETWDES(WHF%WDES,WDESK2,0)
    CALL SETWDES(WHF%WDES,WDESK3,0)
    CALL SETWDES(WHF%WDES,WDESK4,0)

    LAST_FILLED=LAST_FILLED_XI_NOMOD(W,1,1)
    FIRST_EMPTY=FIRST_EMPTY_XI_NOMOD(W,1,1)
    DO ISP=2,WDES%ISPIN
       LAST_FILLED=MAX(LAST_FILLED_XI_NOMOD(W,1,ISP),LAST_FILLED)
       FIRST_EMPTY=MIN(FIRST_EMPTY_XI_NOMOD(W,1,ISP),FIRST_EMPTY)
    ENDDO
    ! set FIRST_EMPTY to zero to allow calculation of HF energy
    FIRST_EMPTY=1

    ! determine VBMIN and VBMAX
    VBMIN=1
    VBMAX=LAST_FILLED

    ! NSTRIPV determines the blocking for the valence band states
    ! it can be made smaller than the number of VB states
! test
    NSTRIPV=VBMAX-VBMIN+1
!   NSTRIPV=MIN(VBMAX-VBMIN+1,1)
! test

    CBMIN=FIRST_EMPTY
    CBMAX=WDES%NB_TOT

    IF (WDES%NKPTS >= WDES%NB_PAR) THEN
       LKPOINT_PARALLEL=.TRUE.
       W1EQUALW2=.FALSE.
       NKMAX_4O=2
    ELSE
       LKPOINT_PARALLEL=.FALSE.
       W1EQUALW2=.FALSE.
       NKMAX_4O=2
       IF (WDES%NKPTS==1) THEN
          IF (WDES%ISPIN==1) W1EQUALW2=.TRUE.
          NKMAX_4O=1
       ENDIF
    ENDIF

    IF (LKPOINT_PARALLEL) THEN
       IF (IO%IU6>=0) THEN
          WRITE(IO%IU6,*) 'parallelization over k-points'
       ENDIF
       CBMIN4=CBMIN
       CBMAX4=CBMAX
    ELSE
       IF (IO%IU6>=0) THEN
          WRITE(IO%IU6,*) 'parallelization over band index 4'
       ENDIF
       ! make sure CBMAX-CBMIN+1 is divisible by the number of nodes
       DO N=1,WDES%NB_PAR
          IF (MOD(CBMAX-CBMIN+1,WDES%NB_PAR)==0) EXIT
          CBMIN=CBMIN-1
       ENDDO
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
    ! it limits the matrix size in BLAS level three, but values around 16-32 should
    ! be fine for maximal performance
    NSTRIP= NGLB4
!    NSTRIP= MIN(NGLB4,16)

    IF (NGLB < NSTRIPV) THEN
       WRITE(*,*) 'internal error in CALCULATE_LOCAL_FIELD: NSTRIPV is larger than NGLB'
       STOP
    ENDIF

    ! allocate storage for W1 (val.), W2 (val.), and W4 (cond.)
    ALLOCATE(W1(VBMAX-VBMIN+1))
    ALLOCATE(W2(VBMAX-VBMIN+1))
    DO N=1,VBMAX-VBMIN+1
       CALL NEWWAV(W1(N) , WDESK1,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL NEWWAV(W2(N) , WDESK2,.TRUE.)
    ENDDO
    ALLOCATE(W4(NGLB4))
    DO N=1,NGLB4
       IF (.NOT.W1EQUALW2) CALL NEWWAV(W4(N) , WDESK4,.TRUE.)
    ENDDO

    IF (IO%IU0>=0) WRITE(IO%IU0,'(A,4I5)') 'allocating two-electron 4 orbital integral table',NSTRIPV, NGLB, NSTRIPV, NGLB4

! TWOELECTRON3O(n3, n4, n2, n1, q ) = < v_k1,n1 v'_k2,n2 | W | c'_k2+q,n4 c_k1-q,n3 > *=
!                                    < c_k1-q,n3 c'_k2+q,n4  | W |  v'_k2,n2 v_k1,n1>       
    ALLOCATE(TWOELECTRON3O(WDES%NB_TOT, NGLB4, NSTRIPV, NSTRIPV, NKMAX_4O))

    ! allocate storage for W3 (cond.)
    IF (LKPOINT_PARALLEL) THEN
       CBMIN3=CBMIN
       CBMAX3=CBMAX
       NGLB3=CBMAX3-CBMIN3+1
       ALLOCATE(W3(NGLB3))
       DO N=1,NGLB3
          CALL NEWWAV(W3(N),WDESK3,.TRUE.)
       ENDDO
    ELSEIF (LFORCE_DISTRIBUTE_W3) THEN
       IF (IO%IU6>=0) THEN
          WRITE(IO%IU6,*) 'parallelization over band index 3'
       ENDIF
       CBMIN3=CBMIN4
       CBMAX3=CBMAX4
       NGLB3=CBMAX3-CBMIN3+1
       ALLOCATE(W3(NGLB3))
       DO N=1,NGLB3
          CALL NEWWAV(W3(N),WDESK3,.TRUE.)
       ENDDO
    ELSE
       CBMIN3=CBMIN
       CBMAX3=CBMAX
       NGLB3=CBMAX3-CBMIN3+1
       ALLOCATE(W3(NGLB3))
       ! try to allocate CBMAX-CBMIN+1 wave functions
       DO N=1,NGLB3
          CALL NEWWAV(W3(N),WDESK3,.TRUE.,ierr)
          IF (ierr>0) EXIT          
       ENDDO
       CALLMPI( M_sum_i(WHF%WDES%COMM_INTER, ierr, 1))
       IF (ierr>0) THEN
          ! failed, first deallocate
          DO N=1,NGLB3
             CALL DELWAV(W3(N),.TRUE.,ierr)
             IF (ierr>0) EXIT
          ENDDO
          DEALLOCATE(W3)
          ! and try with W3 distributed over nodes
          IF (IO%IU6>=0) THEN
             WRITE(IO%IU6,*) 'parallelization over band index 3'
          ENDIF
          CBMIN3=CBMIN4
          CBMAX3=CBMAX4
          NGLB3=CBMAX3-CBMIN3+1
          ALLOCATE(W3(NGLB3))
          DO N=1,NGLB3
             CALL NEWWAV(W3(N),WDESK3,.TRUE.)
          ENDDO
       ENDIF
    ENDIF

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,'(A,/2(A,I5,2X,A,I5/))')' Bands included in local field effects', & 
            'VB(min)=',VBMIN,'VB(max)=',VBMAX,'CB(min)=',CBMIN,'CB(max)=',CBMAX
       WRITE(IO%IU6,'(A,3F14.7)')' parameters for HF',AEXX, HFSCREEN
    ENDIF

    SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ
    IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2

    CALL SET_EKIN(EKIN, EKIN_MAX, WHF )

    EKIN_BINSIZE=WHF%WDES%ENMAX/100
    NBINS=CEILING(WHF%WDES%ENMAX/EKIN_BINSIZE)
    ALLOCATE(ECOR_HARTREE(NBINS),ECOR_EXCHANGE(NBINS),ECOR(NBINS))

    EFOCK=0
    ECOR_HARTREE=0
    ECOR_EXCHANGE=0
    NFFTW = 0
    NFLOAT4O=0 ; NFFT4O=0
!==========================================================================
!  outer loop construct for two electron four orbital integrals
!  these loops go over the indices k1, k2, 
!  and possibly blocks of the band indices n1 and n2
!==========================================================================
    CALL START_TIMING("G")
!==========================================================================
    sp1: DO ISP1=1,WDES%ISPIN
    sp2: DO ISP2=1,WDES%ISPIN
    IF (ISP1>ISP2) CYCLE sp2
!==========================================================================
    ! loop over all q-points N3
    DO K3=1, WDES%NKPTS
       IF (IO%IU0>=0) WRITE(IO%IU0,*)
       IF (IO%IU0>=0) THEN
          IF (WDES%ISPIN==1) THEN
             WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') K3,KPOINTS%VKPT(:,K3)
          ELSE
             WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ",A1,A1,", ")') K3,KPOINTS%VKPT(:,K3),SP(ISP1),SP(ISP2)
          ENDIF
       ENDIF

       ! collect bands for K3=K1-Q
       CALL SETWDES(WHF%WDES,WDESK3,K3)
       CALL W1_GATHER_DISTR( WHF, CBMIN3, CBMAX3, ISP2, W3)
       NFFTW=NFFTW+CBMAX3-CBMIN3+1

       K4_BASE=0
       DO
          K4_BASE=K4_BASE+1
          ! distribute the bands over k-points in a round robin fashion
          K4_DONE =0             ! counts the number of k-points that have been collected
          K4=-1                  ! determine which k-point treated locally
          DO K4_COLLECT=K4_BASE, WDES%NKPTS ! loop from present K4_BASE up to WDES%NKPTS
             IF (WHF%WDES%WTKPT(K4_COLLECT)==0) CYCLE
             K4_DONE=K4_DONE+1              ! new k-point to be included

             CALL SETWDES(WHF%WDES,WDESK4,K4_COLLECT)

             IF (LKPOINT_PARALLEL) THEN
                CALL W1_GATHER_KSEL( WHF, CBMIN, CBMAX, ISP1, W4, K4_DONE)
                NFFTW=NFFTW+(CBMAX-CBMIN+1)/WDES%NB_PAR
             ELSE
                IF (W1EQUALW2) THEN
                   W4=W3
                ELSE
                   CALL W1_GATHER_DISTR( WHF, CBMIN4, CBMAX4, ISP1, W4)
                   NFFTW=NFFTW+CBMAX4-CBMIN4+1
                ENDIF
             ENDIF
             IF (K4_DONE==WDES%NB_LOW .OR. .NOT. LKPOINT_PARALLEL) K4=K4_COLLECT
             IF (K4_DONE==WDES%NB_PAR .OR. .NOT. LKPOINT_PARALLEL) EXIT
          ENDDO
!==========================================================================
!  inner loop for four orbital integrals
!==========================================================================
          DO K1=1,KPOINTS_ORIG%NKPTS

             CALL GWPROGRESS(IO%IU0, K4, WDES%NKPTS, K1, KPOINTS_ORIG%NKPTS)

             ! generate the proper descriptor for W1 wavefunctions
             CALL SETWDES(WHF%WDES,WDESK1,K1)
             IF (WHF%WDES%WTKPT(K1)==0) CYCLE

             CALL W1_GATHER_GLB( WHF, VBMIN, VBMAX, ISP1, W1)
             NFFTW=NFFTW+VBMAX-VBMIN+1
             ! collect bands for k-point K2=K4+Q
             ! depending on whether K4 is distributed among nodes or not
             ! K2 is also accordingly assembled
             K4_DONE =0
             K2_LOCAL=-1
             DO K4_COLLECT=K4_BASE, WDES%NKPTS ! loop from present K2 upto WDES%NKPTS
                IF (WHF%WDES%WTKPT(K4_COLLECT)==0) CYCLE
                K4_DONE=K4_DONE+1   ! new k-point to be included

                K2=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)+WHF%WDES%VKPT(:,K4_COLLECT)-WHF%WDES%VKPT(:,K1),KPOINTS_FULL)
                ! generate the proper descriptor for W2 wavefunctions
                CALL SETWDES(WHF%WDES,WDESK2,K2)


                IF (LKPOINT_PARALLEL) THEN
                   CALL W1_GATHER_KSEL( WHF, VBMIN, VBMAX, ISP2, W2, K4_DONE)
                   NFFTW=NFFTW+(CBMAX-CBMIN+1)/WDES%NB_PAR
                ELSE
                   IF (W1EQUALW2) THEN
                      W2=W1
                   ELSE
                      CALL W1_GATHER_GLB( WHF, VBMIN, VBMAX, ISP2, W2)
                      NFFTW=NFFTW+VBMAX-VBMIN+1
                   ENDIF
                ENDIF

                IF (K4_DONE==WDES%NB_LOW .OR. .NOT. LKPOINT_PARALLEL) K2_LOCAL=K2
                IF (K4_DONE==WDES%NB_PAR .OR. .NOT. LKPOINT_PARALLEL) EXIT
             ENDDO
             ! at this point each CPU holds the k-points corresponding to K2 in W2
             K2=K2_LOCAL

             ! block of stripes NSTRIPV for both indices N1 and N2
             DO NPOS1=VBMIN, VBMAX, NSTRIPV
                NSTRIP1=MIN(VBMAX+1-NPOS1,NSTRIPV)

                DO NPOS2=VBMIN, VBMAX, NSTRIPV
                   NSTRIP2=MIN(VBMAX+1-NPOS2,NSTRIPV)

                   IF (K2>=1) THEN
                      TWOELECTRON3O=0

                      CALL SETWDES(WHF%WDES,WDESK2,K2)
                      CALL SETWDES(WHF%WDES,WDESK4,K4)
                      IF (ISP1==ISP2) THEN
                         IF (K4>=K3) THEN
                            ! first
                            CALL TWOELECTRON4O_ACC_MP2(WHF, H, LATT_CUR, WGW, ENCUTGW, ENCUTGWSOFT, FSG_STORE(1), &
                                 W1, K1, NPOS1, NSTRIP1, ISP1, W2, K2, NPOS2, NSTRIP2, ISP2, W3, K3, W4, K4, &
                                 TWOELECTRON3O, CBMIN, CBMAX, CBMIN3, CBMAX3, CBMIN4, CBMAX4, VBMIN, &
                                 NFLOAT4O, NFFT4O, NSTRIP, 1)
                         ENDIF
                         IF (K4> K3) THEN
                            ! interchange K3 and K4 and calculate the exchange integrals as well
                            ! store it in the second slot in TWOELECTRON3O
                            CALL TWOELECTRON4O_ACC_MP2(WHF, H, LATT_CUR, WGW, ENCUTGW, ENCUTGWSOFT, FSG_STORE(1), &
                                 W1, K1, NPOS1, NSTRIP1, ISP1, W2, K2, NPOS2, NSTRIP2, ISP2, W4, K4, W3, K3, &
                                 TWOELECTRON3O, CBMIN, CBMAX, CBMIN3, CBMAX3, CBMIN4, CBMAX4, VBMIN, &
                                 NFLOAT4O, NFFT4O, NSTRIP, 2)
                         ENDIF
                      ELSE
                         CALL TWOELECTRON4O_ACC_MP2(WHF, H, LATT_CUR, WGW, ENCUTGW, ENCUTGWSOFT, FSG_STORE(1), &
                              W1, K1, NPOS1, NSTRIP1, ISP1, W2, K2, NPOS2, NSTRIP2, ISP2, W3, K3, W4, K4, &
                              TWOELECTRON3O, CBMIN, CBMAX, CBMIN3, CBMAX3, CBMIN4, CBMAX4, VBMIN, &
                              NFLOAT4O, NFFT4O, NSTRIP, 1)
                      ENDIF
                   ENDIF
!==========================================================================
! end construction of two electron four orbital integrals
! here comes the required MP2 code
!==========================================================================
                   TWOELECTRON3O=TWOELECTRON3O*SCALE
                   DO N1=1,NSTRIP1
                      DO N2=1,NSTRIP2
                         NB1_INTO_TOT=NPOS1-1+N1
                         NB2_INTO_TOT=NPOS2-1+N2
                         IF (EMPTY_MP2_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP1))) CYCLE
                         IF (EMPTY_MP2_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP2))) CYCLE

                         IF (ISP1==ISP2) THEN
                            IF (K4==K3) THEN
                            CALL ADD_MP2(LATT_CUR, W, K1, N1, ISP1, K2, N2, ISP2, NB1_INTO_TOT, NB2_INTO_TOT, CBMAX, CBMIN, CBMAX4, CBMIN4, &
                                 K3, K4, 1, 1, & 
                                 EFOCK, ECOR_HARTREE, ECOR_EXCHANGE, TWOELECTRON3O)
                            ELSE IF (K4> K3) THEN
                            CALL ADD_MP2(LATT_CUR, W, K1, N1, ISP1, K2, N2, ISP2, NB1_INTO_TOT, NB2_INTO_TOT, CBMAX, CBMIN, CBMAX4, CBMIN4, &
                                 K3, K4, 1, 2, & 
                                 EFOCK, ECOR_HARTREE, ECOR_EXCHANGE, TWOELECTRON3O)
                            ! interchange K3 and K4
                            CALL ADD_MP2(LATT_CUR, W, K1, N1, ISP1, K2, N2, ISP2, NB1_INTO_TOT, NB2_INTO_TOT, CBMAX, CBMIN, CBMAX4, CBMIN4, &
                                 K4, K3, 2, 1, & 
                                 EFOCK, ECOR_HARTREE, ECOR_EXCHANGE, TWOELECTRON3O)
                            ENDIF
                         ELSE
                            CALL ADD_MP2(LATT_CUR, W, K1, N1, ISP1, K2, N2, ISP2, NB1_INTO_TOT, NB2_INTO_TOT, CBMAX, CBMIN, CBMAX4, CBMIN4, &
                                 K3, K4, 1, 1, &
                                 EFOCK, ECOR_HARTREE, ECOR_EXCHANGE, TWOELECTRON3O)
                         ENDIF

                      ENDDO
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
!==========================================================================
! close outer loop
!==========================================================================
          K4_BASE=K4_COLLECT
          IF (K4_BASE>=WDES%NKPTS) EXIT
       ENDDO
       CALL STOP_TIMING("G",IO%IU6,"MP2")
    ENDDO
!==========================================================================
    ENDDO sp2
    ENDDO sp1
!==========================================================================
    CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')

    CALLMPI( M_sum_d(WGW%COMM_INTER, ECOR_EXCHANGE, SIZE(ECOR_EXCHANGE)))
    CALLMPI( M_sum_d(WGW%COMM_INTER, ECOR_HARTREE,  SIZE(ECOR_HARTREE)))
    CALLMPI( M_sum_z(WGW%COMM_INTER, EFOCK,         1))

    ECOR=ECOR_EXCHANGE+ECOR_HARTREE

    CALLMPI( M_sum_d(WGW%COMM_INTER, NFLOAT4O, 1))
    CALLMPI( M_sum_d(WGW%COMM_INTER, NFFT4O  , 1))


10  FORMAT(//" BLAS level 3 operations / number of FFT's:"/ &
           " number of FFTs for wave wavefunctions          ",F10.0," fft"/ & 
           " number of operations in four-orbital integrals ",F10.2," Gflops, ",F10.0," fft")

    IF (IO%IU0>0) THEN
       WRITE(IO%IU0,*)
       WRITE(IO%IU0,11) REAL(EFOCK,Kind=q), &
      &   REAL(ECOR_HARTREE(NBINS),Kind=q),REAL(ECOR_EXCHANGE(NBINS),Kind=q),REAL(ECOR(NBINS),Kind=q)
    ENDIF
    IF (IO%IU6>0) THEN
       WRITE(IO%IU6,10) NFFTW, NFLOAT4O/1E9, NFFT4O
       WRITE(IO%IU6,*)
       WRITE(IO%IU6,*) 'Moeller Plesset 2 correlation:'
       WRITE(IO%IU6,*) '================================'
       WRITE(IO%IU6,11) REAL(EFOCK,Kind=q), &
      &   REAL(ECOR_HARTREE(NBINS),Kind=q),REAL(ECOR_EXCHANGE(NBINS),Kind=q),REAL(ECOR(NBINS),Kind=q)
       WRITE(IO%IU6,12) (N1*EKIN_BINSIZE,ECOR_HARTREE(N1),ECOR_EXCHANGE(N1), ECOR(N1),N1=1,NBINS)
       WRITE(IO%IU6,*)
    ENDIF
    
11  FORMAT('     Hartree Fock energy: ',F20.8/&
           '   Hartree contr. to MP2: ',F20.8/&
           '  Exchange contr. to MP2: ',F20.8/&
           '  MP2 correlation energy: ',F20.8/)

12  FORMAT(' E_kin(cwm)    E_MP2 (H)     E_MP2 (X)     E_MP2 (T)',/ (F9.3,3F14.6))

    IF (IO%IU6>=0) WRITE(IO%IU6,130)
!==========================================================================
! deallocation
!==========================================================================
    DO N=1,VBMAX-VBMIN+1
       CALL DELWAV(W1(N) ,.TRUE.)
       IF (.NOT. W1EQUALW2) CALL DELWAV(W2(N) ,.TRUE.)
    ENDDO
    DO N=1,NGLB3
       CALL DELWAV(W3(N) ,.TRUE.)
    ENDDO
    DO N=1,NGLB4
       IF (.NOT. W1EQUALW2) CALL DELWAV(W4(N) ,.TRUE.)
    ENDDO
    DEALLOCATE(TWOELECTRON3O)
    DEALLOCATE(W1,W2,W3,W4)

    DEALLOCATE(ECOR_HARTREE,ECOR_EXCHANGE,ECOR)

    IF (ASSOCIATED(H)) CALL DEALLOCATE_ONE_CENTER_H( H)

  END SUBROUTINE CALCULATE_MP2_KPAR


!****************** SUBROUTINE  ADD_MP2 *******************************
!
! this subroutine adds up the MP2 energies to the 
! arrays EFOCK,  ECOR_HARTREE(:), ECOR_EXCHANGE(:)
!
! TWOELECTRON3O(:,:,:,:,:) holds the required Coulomb integrals
!
! as a rule of thumb TWOELECTRON3O(N3, N4, N2, N1, KI) holds
! non interchange integrals and
! TWOELECTRON3O(N4, N3, N2, N1, KI2) the one with interchanged
! band and k-point indices
! 
!
!**********************************************************************

  SUBROUTINE ADD_MP2(LATT_CUR, W, K1, N1, ISP1, K2, N2, ISP2, NB1_INTO_TOT, NB2_INTO_TOT, CBMAX, CBMIN, CBMAX4, CBMIN4, &
       K3, K4, KI, KI2, & 
       EFOCK, ECOR_HARTREE, ECOR_EXCHANGE, TWOELECTRON3O)
    USE constant
    USE lattice 
    USE wave

    INTEGER :: ISP1, ISP2
    TYPE (latt)     LATT_CUR
    TYPE (wavespin) W
    INTEGER :: K1, K2, K3, K4, KI, KI2
    INTEGER :: N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER :: CBMAX, CBMIN, CBMAX4, CBMIN4
    COMPLEX(q) :: EFOCK
    REAL(q) :: ECOR_HARTREE(:),ECOR_EXCHANGE(:)
    GDEFS, TARGET :: TWOELECTRON3O(:,:,:,:,:)
  ! local
    INTEGER :: K3_IN_FULL_ORIG, K2_IN_FULL_ORIG, I, J, IBIN
    REAL(q) :: OCC,VIRT,DENOM,INTE_HARTREE,INTE_EXCHANGE
    REAL(q) :: FAC1,FAC2
    COMPLEX(q) :: HEAD1(3,3),HEAD2(3,3)
    GDEF :: CDER_BETWEEN_STATE12(3),CDER_BETWEEN_STATE34(3)
    GDEFS, POINTER :: TWOELECTRON3O_TRANS(:,:,:,:,:)

    IF (W%WDES%ISPIN==1) THEN
       FAC1=2.0_q
       FAC2=1.0_q
    ELSE
       IF (ISP1==ISP2) THEN
          FAC1=0.5_q
          FAC2=0.5_q
       ELSE
          FAC1=1.0_q
          FAC2=0.0_q
       ENDIF
    ENDIF

    IF (K2==K4 .AND. K1==K3 .AND. ISP1==ISP2) THEN
       ! test whether this  N4 index is on local node
       N4= NB2_INTO_TOT-CBMIN4+1     
       IF ( N4 >=1 .AND.  N4 <= CBMAX4-CBMIN4+1) THEN
       !   <n3,k3  n4,k4 | n2 n1 >  for n3=n1, k3==k1 and n4==n2, k4==k1
          EFOCK=EFOCK+TWOELECTRON3O(N1, N4,  N2, N1, KI)*KPOINTS_ORIG%WTKPT(K1) &
               *W%FERTOT(NB1_INTO_TOT,K1,ISP1)*W%FERTOT(NB2_INTO_TOT,K2,ISP1) & 
               *W%FERTOT(NB2_INTO_TOT,K2,ISP1)*W%FERTOT(NB1_INTO_TOT,K1,ISP1) &
               *FAC2
       ENDIF
    ENDIF

    IF (.NOT. LKPOINT_PARALLEL) THEN
       ! unfortunately we need to the transposed matrix for the exchange
       ALLOCATE(TWOELECTRON3O_TRANS( SIZE(TWOELECTRON3O,2),  SIZE(TWOELECTRON3O,1),  & 
            SIZE(TWOELECTRON3O,3),  SIZE(TWOELECTRON3O,4), SIZE(TWOELECTRON3O,5)))
       CALL TRANSPOSE_MP2(W%WDES, TWOELECTRON3O_TRANS, TWOELECTRON3O, CBMIN4, CBMAX4)
    ELSE
       TWOELECTRON3O_TRANS=>TWOELECTRON3O
    ENDIF
    
    DO N3=1,CBMAX-CBMIN+1
       NB3_INTO_TOT=N3+CBMIN-1
       IF (FILLED_MP2_ORBITAL(W%FERTOT(NB3_INTO_TOT,K3,ISP2))) CYCLE

       DO N4=1,CBMAX4-CBMIN4+1
          NB4_INTO_TOT=N4+CBMIN4-1
          IF (FILLED_MP2_ORBITAL(W%FERTOT(NB4_INTO_TOT,K4,ISP1))) CYCLE

          DENOM=(REAL(W%CELTOT(NB1_INTO_TOT,K1,ISP1),KIND=q)+REAL(W%CELTOT(NB2_INTO_TOT,K2,ISP2),KIND=q)&
               -REAL(W%CELTOT(NB4_INTO_TOT,K4,ISP1),KIND=q)-REAL(W%CELTOT(NB3_INTO_TOT,K3,ISP2),KIND=q))

          IF (DENOM<0._q) THEN
             OCC =W%FERTOT(NB1_INTO_TOT,K1,ISP1)*W%FERTOT(NB2_INTO_TOT,K2,ISP2)*KPOINTS_ORIG%WTKPT(K1)
             VIRT=(1._q-W%FERTOT(NB4_INTO_TOT,K4,ISP1))*(1._q-W%FERTOT(NB3_INTO_TOT,K3,ISP2))

             !   <n3,k3  n4,k4 | n2 n1 >  x c.c.
             ! - <n3,k3  n4,k4 | n2 n1 >* x <n4,k4  n3,k3 | n2 n1>

             ! term from Hartree <n3,k3  n4,k4 | n2 n1 >  x c.c.
             INTE_HARTREE=0
             HEAD1=0
             IF (K1==K4) THEN
                K3_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K3),KPOINTS_FULL_ORIG)
                CALL  CDER_BETWEEN_STATES_ROTATED( &
                     CDER_BETWEEN_STATE12,LATT_CUR, K1, ISP1, NB4_INTO_TOT, NB1_INTO_TOT)

                CALL  CDER_BETWEEN_STATES_ROTATED( &
                     CDER_BETWEEN_STATE34,LATT_CUR, K3_IN_FULL_ORIG, ISP2, NB3_INTO_TOT, NB2_INTO_TOT)

                DO I=1,3
                   DO J=1,3
                      HEAD1(J,I)= HEAD1(J,I)+CDER_BETWEEN_STATE12(J)*CDER_BETWEEN_STATE34(I)
                   ENDDO
                ENDDO
                HEAD1=-HEAD1*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(K1)*SCALE_HEAD

                DO I=1,3
                   INTE_HARTREE=INTE_HARTREE+(1._q/3._q)* &
                        ((TWOELECTRON3O(N3, N4, N2, N1, KI)+HEAD1(I,I))* &
                       &       GCONJG(TWOELECTRON3O(N3, N4, N2, N1, KI)+HEAD1(I,I)))
!                  INTE_HARTREE=INTE_HARTREE+ &
!                       ((TWOELECTRON3O(N3, N4, N2, N1, KI))* &
!                      &       GCONJG(TWOELECTRON3O(N3, N4, N2, N1, KI)))/3
                ENDDO
             ELSE
                INTE_HARTREE= &
                     ((TWOELECTRON3O(N3, N4, N2, N1, KI))* &
                    &       GCONJG(TWOELECTRON3O(N3, N4, N2, N1, KI)))
             ENDIF

             ! Term from exchange - <n3,k3  n4,k4 | n2 n1 >* x <n4,k4  n3,k3 | n2 n1>
             INTE_EXCHANGE=0
             HEAD2=0
             IF (K4==K1 .OR. K1==K3 .AND. ISP1==ISP2) THEN
                IF (K1==K3) THEN
                   CALL  CDER_BETWEEN_STATES_ROTATED( &
                   &   CDER_BETWEEN_STATE12,LATT_CUR, K1, ISP1, NB3_INTO_TOT, NB1_INTO_TOT)

                   K2_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K2),KPOINTS_FULL_ORIG)
                   CALL  CDER_BETWEEN_STATES_ROTATED( &
                   &   CDER_BETWEEN_STATE34,LATT_CUR, K2_IN_FULL_ORIG, ISP2, NB4_INTO_TOT, NB2_INTO_TOT)

                   DO I=1,3
                      DO J=1,3
                         HEAD2(J,I)= HEAD2(J,I)+CDER_BETWEEN_STATE12(J)*CDER_BETWEEN_STATE34(I)
                      ENDDO
                   ENDDO
                   HEAD2=-HEAD2*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(K1)*SCALE_HEAD
                ENDIF

                DO I=1,3
                   INTE_EXCHANGE=INTE_EXCHANGE-(1._q/3._q)* &
                        REAL(GCONJG(TWOELECTRON3O(N3, N4, N2, N1, KI)+HEAD1(I,I))* &
                        (TWOELECTRON3O_TRANS(N4, N3, N2, N1, KI2)+HEAD2(I,I)),KIND=q)
                ENDDO
             ELSE
                INTE_EXCHANGE=- &
                     REAL(GCONJG(TWOELECTRON3O(N3, N4, N2, N1, KI))* &
                     (TWOELECTRON3O_TRANS(N4, N3, N2, N1, KI2)),KIND=q)
             ENDIF
             ! note: this takes approximately 10 % of the total time 
             ! so moved ECOR to post processing
! test
!            ECOR_HARTREE(MAX(NB4_INTO_TOT,NB3_INTO_TOT):W%WDES%NB_TOT)= &
!                 &   ECOR_HARTREE(MAX(NB4_INTO_TOT,NB3_INTO_TOT):W%WDES%NB_TOT)+(OCC*VIRT*INTE_HARTREE/DENOM)
!
!            ECOR_EXCHANGE(MAX(NB4_INTO_TOT,NB3_INTO_TOT):W%WDES%NB_TOT)= &
!                 &   ECOR_EXCHANGE(MAX(NB4_INTO_TOT,NB3_INTO_TOT):W%WDES%NB_TOT)+(OCC*VIRT*INTE_EXCHANGE/DENOM)

             IBIN=CEILING(MAX(W%AUXTOT(NB3_INTO_TOT,K3,ISP2),W%AUXTOT(NB4_INTO_TOT,K4,ISP1))/EKIN_BINSIZE)

             ECOR_HARTREE(IBIN:NBINS)= &
                  &   ECOR_HARTREE(IBIN:NBINS)+(FAC1*OCC*VIRT*INTE_HARTREE/DENOM)

             ECOR_EXCHANGE(IBIN:NBINS)= &
                  &   ECOR_EXCHANGE(IBIN:NBINS)+(FAC2*OCC*VIRT*INTE_EXCHANGE/DENOM)
! test
          ENDIF
       END DO
    END DO
    
    IF (.NOT. LKPOINT_PARALLEL) THEN
       DEALLOCATE(TWOELECTRON3O_TRANS)
    ENDIF
    
  END SUBROUTINE ADD_MP2


!****************** SUBROUTINE  TRANSPOSE_MP2 *************************
!
! To calculate the exchange part the transposed matrix
! is also required. This is not trivial in the parallel version
! when the index N4 is distributed over processors
! the present implementation is not very memory conserving
! but simple
! the required data are copied from each node to a temporary
! array, a global sum is performed and the target node stores
! the data
!
!**********************************************************************

  SUBROUTINE TRANSPOSE_MP2(WDES, TWOELECTRON3O_TRANS, TWOELECTRON3O, CBMIN4_, CBMAX4_)
    USE wave
    TYPE (wavedes) WDES
    GDEFS :: TWOELECTRON3O(:,:,:,:,:)
    GDEFS :: TWOELECTRON3O_TRANS(:,:,:,:,:)
    GDEFS, ALLOCATABLE :: TWOELECTRON3O_TMP(:,:,:,:,:)
    INTEGER :: CBMIN4_, CBMAX4_
  ! local
    INTEGER :: N, CBMIN4, CBMAX4, N4, N4P
    INTEGER :: CBMIN4_MAX, CBMAX4_MAX

#ifdef MPI
    ! loop over all nodes
    DO N=1,WDES%COMM%NCPU
       ! so here is what the target node n requires
       IF (N==WDES%COMM%NODE_ME) THEN
          CBMIN4=CBMIN4_
          CBMAX4=CBMAX4_
       ENDIF
       CALL M_bcast_i_from(WDES%COMM, CBMIN4, 1, n)
       CALL M_bcast_i_from(WDES%COMM, CBMAX4, 1, n)

       ! allocate the work array where data are collected (commensurate with target node)
       ALLOCATE(TWOELECTRON3O_TMP(CBMAX4-CBMIN4+1,  SIZE(TWOELECTRON3O,1),  & 
            SIZE(TWOELECTRON3O,3),  SIZE(TWOELECTRON3O,4), SIZE(TWOELECTRON3O,5)))

       TWOELECTRON3O_TMP=0       ! clear work array

       ! now loop over local data and fill in what the target node n requires
       DO N4=1, CBMAX4_-CBMIN4_+1
          DO N4P=1, CBMAX4-CBMIN4+1
             TWOELECTRON3O_TMP(N4P, CBMIN4_+ N4-1, :, : ,:)= & 
                  TWOELECTRON3O(N4P+CBMIN4-1, N4, :, : ,:)
          ENDDO
       ENDDO

       ! merge from all nodes
#ifdef gammareal 
       CALL M_sum_single(WDES%COMM, TWOELECTRON3O_TMP, SIZE(TWOELECTRON3O_TMP))
#else
       CALL M_sum_single(WDES%COMM, TWOELECTRON3O_TMP, 2*SIZE(TWOELECTRON3O_TMP))
#endif
       IF (N==WDES%COMM%NODE_ME) THEN
          TWOELECTRON3O_TRANS=TWOELECTRON3O_TMP
       ENDIF
       DEALLOCATE( TWOELECTRON3O_TMP)
    ENDDO
#endif
    
  END SUBROUTINE TRANSPOSE_MP2


!****************** SUBROUTINE  TWOELECTRON4O_ACC_MP2 *****************
! calculate
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
! int_d3r d3r' c_k3,n3(r') v'*_k2,n2(r') c'_k4,n4(r) v*_k1,n1(r) W(r',r)
!
! for a specified k-point k1 and k2 and 
! bands n1=[NPOS1,NPOS1+NSTRIP1] and n2=[NPOS2,NPOS2+NSTRIP1]
!
! N4,K4 and N3,K3 are restricted to empty orbitals
! 
!**********************************************************************

 SUBROUTINE TWOELECTRON4O_ACC_MP2(WHF, H, LATT_CUR, WGW, ENCUTGW, ENCUTGWSOFT, FSG, &
               W1, K1, NPOS1, NSTRIP1, ISP1, W2, K2, NPOS2, NSTRIP2, ISP2, W3, K3, W4, K4, &
               TWOELECTRON3O, CBMIN, CBMAX, CBMIN3, CBMAX3, CBMIN4, CBMAX4, VBMIN, &
               NFLOAT, NFFT, NSTRIP, KINDEX)

    USE wave_high
    USE constant
    USE full_kpoints
    USE kpoints_change
    IMPLICIT NONE

! passed variables
    TYPE (wavespin) WHF
    TYPE (one_center_handle), POINTER :: H
    TYPE (latt) LATT_CUR
    INTEGER ISP1,ISP2
    TYPE (wavedes)       WGW        ! descriptor for basis set of response function
    REAL(q) ENCUTGW, ENCUTGWSOFT
    TYPE (wavefun1)    W1(:), W2(:), W3(:), W4(:)
    INTEGER K1, K2, K3, K4, NPOS1, NSTRIP1, NPOS2, NSTRIP2
    GDEFS :: TWOELECTRON3O(:,:,:,:,:)
    INTEGER :: CBMIN, CBMAX, CBMIN3, CBMAX3, CBMIN4, CBMAX4, VBMIN
    REAL(q) :: NFLOAT, NFFT
    INTEGER :: NSTRIP, KINDEX

! local variables
    INTEGER N1, N2, N3, N4, NB1_INTO_TOT, NB2_INTO_TOT, NB3_INTO_TOT, NB4_INTO_TOT
    INTEGER NP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
    TYPE (wavedes1)    WGWQ
    INTEGER NQ, NQ_
    COMPLEX(q), ALLOCATABLE :: GCHG14(:,:)    ! charge 
    COMPLEX(q), ALLOCATABLE :: GCHG23(:,:)    ! charge 
    GDEF      , ALLOCATABLE :: CRHO14(:,:)    ! one center charge
    GDEF      , ALLOCATABLE :: CRHO23(:,:)    ! one center charge
    COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
    LOGICAL LPHASE
    REAL(q) :: FSG                        ! singularity correction
    REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
    INTEGER :: ierror

    INTEGER :: node
    INTEGER :: CBMIN3_CYCLIC,CBMAX3_CYCLIC,NCYCLES

#ifdef MPI
    IF (WHF%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('TWOELECTRON4O_ACC_MP2: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#endif
    ! set the descriptor for wavefunctions at the point K1+K4
    ! the charge c'_k4,n4(r)  v_k1,n1(r)
    ! is transformed corresponding to a wavevector q
    NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
    CALL SETWDES(WGW, WGWQ, NQ)

    NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL)
    IF (NQ/=NQ_) THEN
       WRITE(*,*)'internal error in TWOELECTRON4O_ACC_MP2: q-point changed',NQ,NQ_
       STOP
    ENDIF
! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band
    NP =WGWQ%NGVECTOR

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         GCHG14(NP, NSTRIP1* NSTRIP ), &
         GCHG23(NP, NSTRIP2* NSTRIP ), &
         CHAM(NSTRIP1, NSTRIP, NSTRIP2,  NSTRIP))

    IF (ASSOCIATED(H)) THEN
       ALLOCATE(CRHO14(H%TOTAL_ENTRIES, NSTRIP1* NSTRIP ), CRHO23(H%TOTAL_ENTRIES, NSTRIP2* NSTRIP ))
    ENDIF

!    CALL SET_GFAC_WITHOUT_WEIGHT(GRIDHF,LATT_CUR,K1,K4,FSG, POTFAK)
    ! apply soft cutoff function
    IF (ENCUTGW /= ENCUTGWSOFT .AND. ENCUTGW > 0 .AND. ENCUTGWSOFT > 0) THEN
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK, ENCUTGW, ENCUTGWSOFT )
    ELSE
       CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
    ENDIF
!==========================================================================
!   c*'_k4,n4(r) v_k1,n1(r)
!==========================================================================
    DO NPOS4=CBMIN4, CBMAX4, NSTRIP
       NSTRIP4=MIN(CBMAX4+1-NPOS4,NSTRIP)

       ! set phase factor in fastaug
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4))
       ! average electrostatic potential for k=k' and n=n'
       ! k1+k4-q might be any reciprocal lattice vector G
       ! in that case the result is shifted by G with respect to the 
       ! internal data layout (FFTEXT), we apply a shift e^iGr in real space
       ! to cure the problem
       CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)

       GCHG14=0
       IF (ASSOCIATED(H)) THEN
          CRHO14=0
       ENDIF
       
       DO N4=1,NSTRIP4
          NB4_INTO_TOT=NPOS4-1+N4
          NB4         =NPOS4-CBMIN4+N4
!          IF (FILLED_MP2_ORBITAL(WHF%FERTOT(NB4_INTO_TOT,K4,ISP1))) CYCLE

          DO N1=1,NSTRIP1
             NB1_INTO_TOT=NPOS1-1+N1
             NB1         =NPOS1-VBMIN+N1
             IF (EMPTY_MP2_ORBITAL(WHF%FERTOT(NB1_INTO_TOT,K1,ISP1))) CYCLE

             IF (ASSOCIATED(H)) THEN
                CALL FOCK_CHARGE_ONE_CENTER_NOINT( W1(NB1), W4(NB4), GWORK(1), &
                     H, CRHO14(1,N1+NSTRIP1*(N4-1)), CRHOLM,  SIZE(CRHOLM))
             ELSE
                CALL FOCK_CHARGE_NOINT( W1(NB1), W4(NB4), GWORK(1), CRHOLM,  SIZE(CRHOLM))
             ENDIF
             IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

             CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                  GWORK(1),GCHG14(1,N1+NSTRIP1*(N4-1)),WGWQ%GRID,.FALSE.)
             NFFT=NFFT+1

             ! multiply with potential factor
             CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG14(1,N1+NSTRIP1*(N4-1)), POTFAK(1))
          ENDDO
       ENDDO

       IF (ASSOCIATED(H)) THEN
          DO N1=1, NSTRIP*NSTRIP1, NSTRIP*NSTRIP2
             N2=MIN(NSTRIP*NSTRIP2, NSTRIP*NSTRIP1-N1+1)
             ! use CRHO23 as temporary work array
             CALL APPLY_ONE_CENTER_H( WHF%WDES, H, CRHO14(:,N1:), CRHO23(:,:), N2)
          ENDDO
       ENDIF

       ! set the phase factors q'=-k3-k2
       CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2))
       CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
!==========================================================================
!     c_k3,n3(r') v'*_k2,n2(r')
!==========================================================================
       DO NPOS3=CBMIN3, CBMAX3, NSTRIP
          NSTRIP3=MIN(CBMAX3+1-NPOS3,NSTRIP)

          GCHG23=0
          IF (ASSOCIATED(H)) THEN
             CRHO23=0
          ENDIF

          DO N3=1,NSTRIP3
             NB3_INTO_TOT=NPOS3-1+N3
             NB3         =NPOS3-CBMIN3+N3
!             IF (FILLED_MP2_ORBITAL(WHF%FERTOT(NB3_INTO_TOT,K3,ISP2))) CYCLE

             DO N2=1,NSTRIP2
                NB2_INTO_TOT=NPOS2-1+N2
                NB2         =NPOS2-VBMIN+N2
                IF (EMPTY_MP2_ORBITAL(WHF%FERTOT(NB2_INTO_TOT,K2,ISP2))) CYCLE

                IF (ASSOCIATED(H)) THEN
                   CALL FOCK_CHARGE_ONE_CENTER_NOINT( W3(NB3), W2(NB2), GWORK(1), & 
                        H, CRHO23(1,N2+(N3-1)*NSTRIP2), CRHOLM,  SIZE(CRHOLM))
                ELSE
                   CALL FOCK_CHARGE_NOINT( W3(NB3), W2(NB2), GWORK(1), CRHOLM,  SIZE(CRHOLM))
                ENDIF

                ! apply phase factor e^iGr if required
                IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
                
                ! now we have c_k3,n3(r') v'*_k2,n2(r')
                ! extract using FFT
                CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                     GWORK(1),GCHG23(1,N2+(N3-1)*NSTRIP2),WGWQ%GRID,.FALSE.)
                NFFT=NFFT+1

                GCHG23(1:NP,N2+(N3-1)*NSTRIP2)=GCHG23(1:NP,N2+(N3-1)*NSTRIP2)*(1.0_q/GRIDHF%NPLWV)
             ENDDO
          ENDDO
                    
!==========================================================================
!    c_k3,n3(r') v'*_k2,n2(r')  c'_k4,n4(r)  v*_k1,n1(r)
!==========================================================================
          CHAM=0

          IF (ASSOCIATED(H)) THEN
             CALL APPLY_PHASE_ONE_CENTER(WHF%WDES, H, CRHO23(:,:NSTRIP2*NSTRIP3), & 
                  WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,K1)+WHF%WDES%VKPT(:,K4))
          ENDIF

          CBMIN3_CYCLIC=CBMIN3
          CBMAX3_CYCLIC=CBMAX3
          NCYCLES=1
#ifdef MPI
          IF (CBMIN3>CBMIN .OR. CBMAX3<CBMAX) NCYCLES=WHF%WDES%COMM%NCPU 
#endif
          DO node=0,NCYCLES-1

          CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, m_ NP, -one, &
               GCHG14(1,1), m_ SIZE(GCHG14,1), &
               GCHG23(1,1), m_ SIZE(GCHG23,1), &
               zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))

          NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4*NP*8

          IF (NCYCLES>1) THEN
             CALLMPI( M_cycle_d(WHF%WDES%COMM, GCHG23, 2*NP*NSTRIP2*NSTRIP3))
          ENDIF
          
          IF (ASSOCIATED(H)) THEN
             CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP4, NSTRIP2*NSTRIP3, H%TOTAL_ENTRIES, -one, &
                  CRHO14(1,1), SIZE(CRHO14,1), &
                  CRHO23(1,1), SIZE(CRHO23,1), &
                  one , CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
#ifdef gammareal
             IF (NCYCLES>1) THEN
                CALLMPI( M_cycle_d(WHF%WDES%COMM, CRHO23, H%TOTAL_ENTRIES*NSTRIP2*NSTRIP3))
             ENDIF
#else
             IF (NCYCLES>1) THEN
                CALLMPI( M_cycle_d(WHF%WDES%COMM, CRHO23, 2*H%TOTAL_ENTRIES*NSTRIP2*NSTRIP3))
             ENDIF
#endif
             NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*NSTRIP3*NSTRIP4*H%TOTAL_ENTRIES*8
          ENDIF
!==========================================================================
!   store in TWOELECTRON3O
!==========================================================================
          DO N1=1,NSTRIP1
             IF (N1 > SIZE(TWOELECTRON3O,4)) THEN
                WRITE(*,*)'internal error in TWOELECTRON4O_ACC_MP2: out of bounds 4 ',N1
                STOP
             ENDIF
             DO N2=1,NSTRIP2
                IF (N2 > SIZE(TWOELECTRON3O,3)) THEN
                   WRITE(*,*)'internal error in TWOELECTRON4O_ACC_MP2: out of bounds 3 ',N2
                   STOP
                ENDIF
                NB1_INTO_TOT=NPOS1-1+N1
                NB2_INTO_TOT=NPOS2-1+N2
                DO N4=1,NSTRIP4
                   NB4_INTO_TOT=NPOS4-1+N4
                   NB4         =NPOS4-CBMIN4+N4

                   IF (NB4 > SIZE(TWOELECTRON3O,2)) THEN
                      WRITE(*,*)'internal error in TWOELECTRON4O_ACC_MP2: out of bounds 2 ',NB4
                      STOP
                   ENDIF
                   DO N3=1,NSTRIP3
                      NB3_INTO_TOT=CBMIN3_CYCLIC+(NPOS3-CBMIN3)+N3-1
                      NB3         =NB3_INTO_TOT-CBMIN+1
                      IF ( NB3> SIZE(TWOELECTRON3O,1)) THEN
                         WRITE(*,*)'internal error in TWOELECTRON4O_ACC_MP2: out of bounds 1 ',NB3_INTO_TOT,NB3
                         STOP
                      ENDIF
                      ! WRITE(*,*) NB3, NB3_INTO_TOT, NB4, NB4_INTO_TOT
                      ! WRITE(*,'(4F14.7)') CHAM(N1, N4, N2, N3)
                      ! TWOELECTRON3O(n3, n4, q, n2, n1) = < v_k1,n1 v'_k2,n2 | W | c'_k2+q,n4 c_k1-q,n3 > *=
                      !              < c_k1-q,n3 c'_k2+q,n4  | W |  v'_k2,n2 v_k1,n1>       
                      TWOELECTRON3O( NB3, NB4, N2, N1, KINDEX)=GCONJG(CHAM( N1, N4, N2, N3))
#ifdef dump_twoe
                      IF (K1==1 .AND. K2==3) THEN
                         WRITE(77,'(6I3,2F14.7)') K1,NB1_INTO_TOT,NB3_INTO_TOT,K2,NB2_INTO_TOT,NB4_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
                      IF (K2==1 .AND. K1==3) THEN
                         WRITE(78,'(6I3,2F14.7)') K2,NB2_INTO_TOT,NB4_INTO_TOT,K1,NB1_INTO_TOT,NB3_INTO_TOT,CHAM( N1, N4, N2, N3)
                      ENDIF
#endif
                   ENDDO ! N3
                ENDDO ! N4
             ENDDO ! N2
          ENDDO ! N1
          CBMIN3_CYCLIC=CBMIN3_CYCLIC-(CBMAX3-CBMIN3+1)
          CBMAX3_CYCLIC=CBMAX3_CYCLIC-(CBMAX3-CBMIN3+1)
          IF (CBMIN3_CYCLIC<CBMIN) THEN
             CBMIN3_CYCLIC=CBMIN3_CYCLIC+(CBMAX-CBMIN+1)
             CBMAX3_CYCLIC=CBMAX3_CYCLIC+(CBMAX-CBMIN+1)
          ENDIF
          ENDDO ! node
       ENDDO ! NPOS3
    ENDDO ! NPOS4
    DEALLOCATE(GCHG14, GCHG23)

    IF (ASSOCIATED(H)) THEN
       DEALLOCATE(CRHO14, CRHO23)
    ENDIF

!    WRITE(*,'(16F10.4)') REAL(TWOELECTRON3O)

  END SUBROUTINE TWOELECTRON4O_ACC_MP2
END MODULE mp2kpar
