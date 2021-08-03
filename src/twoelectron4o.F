#include "symbol.inc"

!*********************************************************************
!
! This module implements the two electron four orbital integrals
! and related quantities such as Moeller Plesset 2
! 
!*********************************************************************

MODULE twoelectron4o
  USE prec
  USE fock
  USE dfast
  !
  ! MP2_EMPTY_THRESHHOLD sets a threshhold for emtpy orbitals
  !
  REAL(q), PARAMETER :: MP2_EMPTY_THRESHHOLD=0.5
CONTAINS

!*********************************************************************
!
! main module
!
!*********************************************************************
  SUBROUTINE TWOELECTRON4O_MAIN( &
          P,WDES,NONLR_S,NONL_S,W,LATT_CUR,LATT_INI, &
          T_INFO,DYN,INFO,IO,KPOINTS,SYMM,GRID, LMDIM)

    USE base
    USE ini
    USE lattice
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
    USE kpoints_change
    USE full_kpoints
    USE main_mpi
    USE mlr_optic
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (type_info)   T_INFO
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W
    TYPE (latt)        LATT_CUR
    TYPE (dynamics)    DYN
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (kpoints_struct) KPOINTS
    TYPE (symmetry)    SYMM
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    INTEGER :: LMDIM
    TYPE (latt)        LATT_INI
    
!  local
    INTEGER :: NSTRIP1, NSTRIP2, NSTRIP
    REAL(q),ALLOCATABLE :: FOCKE(:,:)
    INTEGER, ALLOCATABLE :: K3_(:,:)
    TYPE (wavespin)  W_REDIS
    INTEGER :: N1_GLOBAL, N1, N2_GLOBAL, N2, K1, K2, N3, K3, K4, N4, &
         NSTRIP1_ACT, NSTRIP2_ACT, NN1, KSTART, KSTRIDE, K4_STORE
    INTEGER :: N3LOW, N4LOW, N4UP
    INTEGER :: NCPU, ISP
    REAL(q) :: NFLOAT
    REAL(q) :: EFOCK,ECOR(WDES%NB_TOT)
    REAL(q) :: ECOR_HARTREE(WDES%NB_TOT),ECOR_EXCHANGE(WDES%NB_TOT)
    REAL(q) :: OCC,VIRT,DENOM,INTE_HARTREE,INTE_EXCHANGE
    COMPLEX(q) :: HEAD(3,3),ADD,SUM
    COMPLEX(q) :: HEAD1(3,3),HEAD2(3,3)
    INTEGER :: IERR,I,J,K3_IN_FULL_ORIG, K2_IN_FULL_ORIG
    GDEF :: CDER_BETWEEN_STATE12(3),CDER_BETWEEN_STATE34(3)
    GDEFS, ALLOCATABLE :: TWOELECTRON3O(:,:,:,:,:,:)
    REAL(q), POINTER :: EKIN(:,:,:), EKIN_MAX(:,:,:)
! LMP2=.TRUE. implies that the indices N1,N4 are restricted to
! filled orbitals only (in the chemist notations these incides
! are usually termed a and b)
    LOGICAL :: LMP2=.TRUE.
! LMP2_EMPTY=.TRUE. implies that the indices N2,N3 are restricted to
! empty orbitals only (in the chemist notations these incides
! are usually termed i and j)
! if one includes a lot of empty orbitals savings are very small
! furthermore, if LMP2_EMTPY=.TRUE., the Fock energy can not be calculated

    LOGICAL :: LMP2_EMPTY=.FALSE.
    IF (FOURORBIT==0) RETURN

    CALL START_TIMING("G")

    CALL CHECK_FULL_KPOINTS ! all set up properly ?

#ifdef MPI
    NCPU=W%WDES%COMM_INTER%NCPU              ! number of band groups
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('TWOELECTRON4O_MAIN: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#else
    NCPU=1
#endif
!=======================================================================
! allocate all required arrays
! the four orbital integrals are defined as
!   ( N3,K3* N4,K1-K2+K3 |v| N2,K2* N1,K1) =
!    <N3,K3  N2,K2 |v| N4,K4  N1+NN1-1,K1> =
!   int psi_N3,K3 *(r')    psi_N4,K4(r')
!       psi_N2,K2 *(r) psi_N1+NN1,K1(r) 1/|r-r'| dr dr' =
! N1 and N4 are occupied orbitals
!
! the four orbital integrals
! are calculated by looping in the calling routine over N2,K2
! and N1,K1 and supplying a strip [N1,N1+NSTRIP],K1 [N2,N2+NSTRIP],K2
! of wavefunctions to the calling routine
! N1 and N4
!
! the called routine stores the results in wavefunction arrays
!  W(NSTRIP1,NSTRIP2)
! and loops over N3, K3 these two indices become the indices into the
! W array W...%(..,N3,K3)
!=======================================================================
    ! hard coded 1
    NSTRIP1=1
    ! reduce this value if you have memory problems
    NSTRIP2=MIN(WDES%NBANDS,MAX(16/NCPU,1))

    ! blocking in ORTH, change only if you know what you are doing
    NSTRIP=NSTRIP_STANDARD

!=======================================================================
! read first derivative of wavefuntions with respect to k from file
! WAVEDER (if present), this information is needed to correctly treat
! the matrix elements with K1==K2 and K3==K4
!=======================================================================
    CALL READ_CDER_BETWEEN_STATES(WDES, IO%IU0, 55)
    ! some tricks so that the distribution of CDER_BETWEEN_STATES
    ! works as well in the case KIMAGES /= 0
    IF (KIMAGES/=0) THEN
       IERR=0
       IF (.NOT.ALLOCATED(CDER_BETWEEN_STATES)) IERR=1
       CALLMPI( M_sum_i(COMM_CHAIN, IERR, 1))
       IF (IERR==0) THEN
          CALLMPI( M_sum_g(COMM_CHAIN, CDER_BETWEEN_STATES(1,1,1,1,1), SIZE(CDER_BETWEEN_STATES)))
       ELSE
          IF (ALLOCATED(CDER_BETWEEN_STATES)) DEALLOCATE(CDER_BETWEEN_STATES)
       ENDIF
    ENDIF

!=======================================================================
! generate wavefunctions at all k-points
!=======================================================================
    ! switch of symmetry
    CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
         &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,INFO%ISPIN,IO%IU6)
       
    ! reread k-points with LINVERSION=.FALSE.  to generate full mesh
    CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR,.FALSE., &
         T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
    CALL KPAR_SYNC_ALL(WDES,W)
    CALL RE_GEN_LAYOUT( GRID, WDES, KPOINTS, LATT_CUR, LATT_CUR, IO%IU6, IO%IU0)
    CALL REALLOCATE_WAVE( W, GRID, WDES, NONL_S, T_INFO, P, LATT_CUR)

    N1=0
    DO ISP=1,WDES%ISPIN
       DO K1=1,KPOINTS_ORIG%NKPTS
          N1=MAX(N1,LAST_FILLED_MP2(W,LMP2,K1,ISP))
       ENDDO
    ENDDO

    ALLOCATE(FOCKE(NSTRIP1,NSTRIP2*NCPU))
    IF (KIMAGES==0) THEN
       ALLOCATE(TWOELECTRON3O(WDES%NB_TOT, WDES%NB_TOT,WDES%NKPTS, N1,WDES%NKPTS, NSTRIP1))
    ELSE
       write(*,*) 'allocate 2e3o, dim k4=',CEILING(REAL(WDES%NKPTS)/REAL(KIMAGES))
       ALLOCATE(TWOELECTRON3O(WDES%NB_TOT, WDES%NB_TOT,WDES%NKPTS, N1,CEILING(REAL(WDES%NKPTS)/REAL(KIMAGES)), NSTRIP1))
    ENDIF

    ALLOCATE(K3_(WDES%NKPTS,WDES%NKPTS))
!=======================================================================
! redistribute wavefunctions if required
!=======================================================================
    IF (WDES%DO_REDIS) THEN
       CALL ALLOCW(WDES,W_REDIS)
       ! copy entries
       W_REDIS%CW=W%CW
       W_REDIS%CPROJ=W%CPROJ
       W_REDIS%CELTOT=W%CELTOT
       W_REDIS%FERTOT=W%FERTOT

       ! redistribute
       CALL REDISTRIBUTE_PW( W_REDIS)
       CALL REDISTRIBUTE_PROJ( W_REDIS)
    ELSE
       W_REDIS=W
    ENDIF

    CALL SET_EKIN(EKIN, EKIN_MAX, W )
!=======================================================================
! loop over K1, K2
! restrict loop over K1 to IRZ
!=======================================================================
    EFOCK=0
! test
    ECOR_HARTREE=0
    ECOR_EXCHANGE=0
! test
    ECOR =0
    NFLOAT=0
    TWOELECTRON3O=0
    SUM=0

    DO ISP=1,WDES%ISPIN
! only k-points in IRZ
    DO K1=1,KPOINTS_ORIG%NKPTS
       IF (IO%IU0>=0) WRITE(IO%IU0,'("NK=",I4,", ")',ADVANCE='NO') K1
       DO N1_GLOBAL=1,LAST_FILLED_ACTUAL_MP2(W,LMP2,K1,ISP),NSTRIP1
          NSTRIP1_ACT=MIN(LAST_FILLED_ACTUAL_MP2(W,LMP2,K1,ISP)+1-N1_GLOBAL,NSTRIP1)

          DO K2=1,WDES%NKPTS
             IF (IO%IU0>=0) WRITE(IO%IU0,'(".")',ADVANCE='NO')
             
             DO N2_GLOBAL=1,WDES%NBANDS,NSTRIP2
                NSTRIP2_ACT=MIN(WDES%NBANDS+1-N2_GLOBAL,NSTRIP2)

                ! possible restrictions 
                ! K4 >= K1            (TWOELECTRON4O_ACC + INPROD_BETWEEN_STATES)
                ! N4 >= N1 if K4==K1  (TWOELECTRON4O_ACC + INPROD_BETWEEN_STATES)
                ! 
                !
                ! MP2:
                ! N1 restricted to filled orbitals (main loop)
                ! N4 restricted to filled orbitals (TWOELECTRON4O_ACC + INPROD_BETWEEN_STATES)
                ! N2 restricted to empty orbitals
                ! N3 restricted to empty orbitals

                CALL TWOELECTRON4O_ACC(LMDIM, LATT_CUR, W, W_REDIS, &
                     FOCKE, P, ISP,  &
                     K1,  N1_GLOBAL, NSTRIP1_ACT, K2, &
                     N2_GLOBAL, NSTRIP2_ACT, K3_(:,K2), &
                     TWOELECTRON3O, NFLOAT, LMP2, LMP2_EMPTY)
             ENDDO
          ENDDO
          !=======================================================================
          ! at this point TWOELECTRON3O(n3,  n2,k2,  n4,k4,  nn1) equals
          !  <c_n3,k3  c_n2,k2 | vbar | v_n4,k4  v_n1+nn1-1,k1>
          !   int c_n3,k3 *(r') v_n4,k4(r')
          !       c_n2,k2 *(r)  v_n1+nn1,k1(r) 1/|r-r'| dr dr' =
          ! divided by the number of k-points   k3=k4-k2+k1
          ! n1 and n4 are occupied orbitals (valence=v)
          ! n2 and n3 are virtual (empty) band orbitals (conduction=c)
          !  some symmtry relations
          !   <n3,k3  n2,k2 | n4,k4  n1,k1>= <n2,k2  n3,k3 | n1,k1  n4,k4>
          !                                = <n1,k1  n4,k4 | n2,k2  n3,k3>*
          !=======================================================================
          ! Hartree-Fock energy
#ifdef MPI
          IF (KIMAGES>0) THEN
             KSTART=COMM_CHAIN%NODE_ME
             KSTRIDE=KIMAGES
          ELSE
             KSTART=1
             KSTRIDE=1
          ENDIF
#else
          KSTART=1
          KSTRIDE=1
#endif
          DO NN1=1,NSTRIP1_ACT
             N1=N1_GLOBAL-1+NN1
             DO K2=KSTART,WDES%NKPTS,KSTRIDE
                K4=K2
                IF (KIMAGES==0) THEN
                   K4_STORE=K4
                ELSE
                   K4_STORE=CEILING(REAL(K4)/REAL(KIMAGES))
                ENDIF
                K3=K3_(K4,K2)
                IF (K3==-1) THEN
                ELSE IF (K3==K1) THEN
                   DO N2=1,LAST_FILLED_MP2(W,LMP2,K2,ISP)
                      N3=N1
                      N4=N2
                      EFOCK=EFOCK+TWOELECTRON3O(N3,  N2,K2,  N4,K4_STORE, NN1)*KPOINTS_ORIG%WTKPT(K1) &
                           *W%FERTOT(N1,K1,ISP)*W%FERTOT(N3,K3,ISP)*W%FERTOT(N2,K2,ISP)*W%FERTOT(N4,K4, ISP)
                   ENDDO
                ELSE
                   WRITE(*,*) 'internal error in TWOELECTRON4O_MAIN: the K3_ index array is corrupted',K1,K3
                   STOP
                ENDIF
             ENDDO
          ENDDO

!         DO NN1=1,NSTRIP1_ACT
!            N1=N1_GLOBAL-1+NN1
!            IF (EMPTY_MP2_ORBITAL(W%FERTOT(N1,K1,ISP))) CYCLE
!            DO K4=KSTART,WDES%NKPTS,KSTRIDE
!               IF (KIMAGES==0) THEN
!                  K4_STORE=K4
!               ELSE
!                  K4_STORE=CEILING(REAL(K4)/REAL(KIMAGES))
!               ENDIF
!               DO N4=1,LAST_FILLED_ACTUAL_MP2(W,LMP2,K4,ISP)
!                  DO K2=1,WDES%NKPTS
!                     K3=K3_(K4,K2)
!                     IF (K3==K4) THEN
!                     K3_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K3),KPOINTS_FULL_ORIG)
!                     DO N2=FIRST_EMPTY_ACTUAL_MP2(W,LMP2_EMPTY,K2,ISP),WDES%NB_TOT
!                        DO N3=FIRST_EMPTY_ACTUAL_MP2(W,LMP2_EMPTY,K3,ISP),WDES%NB_TOT
!!                          IF( N2/=N3 .OR. N1/=N4 .OR. K2/=K3) CYCLE
!!                          WRITE(*,'(10I4)') N1, N3, N2, N4, K1, K3_IN_FULL_ORIG
!                          CALL  CDER_BETWEEN_STATES_ROTATED( &
!                         &   CDER_BETWEEN_STATE12,LATT_CUR, K1, ISP, N2, N1)
!
!                          CALL  CDER_BETWEEN_STATES_ROTATED( &
!                         &   CDER_BETWEEN_STATE34,LATT_CUR, K3_IN_FULL_ORIG, ISP, N3, N4)
!                         
!                          HEAD=0
!                          DO I=1,3
!                          DO J=1,3
!                             HEAD(J,I)= HEAD(J,I)+CDER_BETWEEN_STATE12(J)*CDER_BETWEEN_STATE34(I)
!                          ENDDO
!                          ENDDO
!                          ADD=HEAD(3,3)
!                          ADD=ADD*EDEPS/LATT_CUR%OMEGA*WDES%WTKPT(K1)
!!                         SUM=SUM+ADD/(W%CELTOT(N1,K1,ISP)-W%CELTOT(N2,K2,ISP))*KPOINTS_ORIG%WTKPT(K1)*2*2
!!                         WRITE(*,'(4I4, 4F14.7)') K1, K2, N1, N2, SUM, 1.0/(W%CELTOT(N1,K1,ISP)-W%CELTOT(N2,K2,ISP))*KPOINTS_ORIG%WTKPT(K1)*2
!
!                          TWOELECTRON3O(N3, N2,K2, N4,K4_STORE, NN1)=TWOELECTRON3O(N3, N2,K2, N4,K4_STORE, NN1)+ADD
!                          
!                        ENDDO
!                     ENDDO
!                     ENDIF
!                  ENDDO
!               ENDDO
!            ENDDO
!         ENDDO
!!        WRITE(*,*) SUM

#ifdef debug
          DO NN1=1,NSTRIP1_ACT
             N1=N1_GLOBAL-1+NN1
             DO K4=KSTART,WDES%NKPTS,KSTRIDE
                IF (KIMAGES==0) THEN
                   K4_STORE=K4
                ELSE
                   K4_STORE=CEILING(REAL(K4)/REAL(KIMAGES))
                ENDIF
                DO N4=1,LAST_FILLED_MP2(W,LMP2,K4,ISP)
                   DO K2=1,WDES%NKPTS
                      K3=K3_(K4,K2)
                      IF (K3/=-1) THEN
                      DO N2=FIRST_EMPTY_MP2(W,LMP2_EMPTY,K2,ISP),WDES%NB_TOT
                         DO N3=1,WDES%NB_TOT
                            ! everything is defined here
                            ! if running in the IRZ weight by KPOINTS_ORIG%WTKPT(K1)
                            ! if running in the full BZ weight by WDES%WTKPT(K1)
                            ! occupancies (between 0 and 1) in W%FERTOT(NX, KX, ISP)
                            WRITE(77,'(8I3,2F10.4)') K3,N3,K2,N2,K4,N4,K1,N1,TWOELECTRON3O(N3,  N2,K2,  N4,K4_STORE, NN1)
                            WRITE(78,'(8I3,2F10.4)') K2,N2,K3,N3,K1,N1,K4,N4,TWOELECTRON3O(N3,  N2,K2,  N4,K4_STORE, NN1)
                            WRITE(79,'(8I3,2F10.4)') K1,N1,K4,N4,K2,N2,K3,N3,GCONJG(TWOELECTRON3O(N3,  N2,K2,  N4,K4_STORE, NN1))
                         ENDDO
                      ENDDO
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
          ENDDO
#endif

          DO NN1=1,NSTRIP1_ACT
             N1=N1_GLOBAL-1+NN1
             IF (EMPTY_MP2_ORBITAL(W%FERTOT(N1,K1,ISP))) CYCLE
             DO K4=KSTART,WDES%NKPTS,KSTRIDE
                IF (KIMAGES==0) THEN
                   K4_STORE=K4
                ELSE
                   K4_STORE=CEILING(REAL(K4)/REAL(KIMAGES))
                ENDIF
                DO N4=1,LAST_FILLED_ACTUAL_MP2(W,LMP2,K4,ISP)
                   DO K2=1,WDES%NKPTS
                      K3=K3_(K4,K2)

                      IF (K3/=-1) THEN
                         DO N2=FIRST_EMPTY_ACTUAL_MP2(W,LMP2_EMPTY,K2,ISP),WDES%NB_TOT
                         DO N3=FIRST_EMPTY_ACTUAL_MP2(W,LMP2_EMPTY,K3,ISP),WDES%NB_TOT

                            DENOM=(REAL(W%CELTOT(N1,K1,ISP),KIND=q)+REAL(W%CELTOT(N4,K4,ISP),KIND=q)&
                                  -REAL(W%CELTOT(N2,K2,ISP),KIND=q)-REAL(W%CELTOT(N3,K3,ISP),KIND=q))

                            IF (DENOM<0._q) THEN
                               OCC=W%FERTOT(N1,K1,ISP)*W%FERTOT(N4,K4,ISP)*KPOINTS_ORIG%WTKPT(K1)
                               VIRT=(1._q-W%FERTOT(N2,K2,ISP))*(1._q-W%FERTOT(N3,K3,ISP))

                               !   <N3,K3  N2,K2 | N4,K4  N1+NN1-1,K1>  x c.c.
                               ! - <N3,K3  N2,K2 | N4,K4  N1+NN1-1,K1>* x <N2,K2  N3,K3 | N4,K4  N1+NN1-1,K1>

                               HEAD1=0
                               IF (K1==K2) THEN
                                  CALL  CDER_BETWEEN_STATES_ROTATED( &
                                 &   CDER_BETWEEN_STATE12,LATT_CUR, K1, ISP, N2, N1)

                                  K3_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K3),KPOINTS_FULL_ORIG)
                                  CALL  CDER_BETWEEN_STATES_ROTATED( &
                                 &   CDER_BETWEEN_STATE34,LATT_CUR, K3_IN_FULL_ORIG, ISP, N3, N4)

                                  DO I=1,3
                                  DO J=1,3
                                     HEAD1(J,I)= HEAD1(J,I)+CDER_BETWEEN_STATE12(J)*CDER_BETWEEN_STATE34(I)
                                  ENDDO
                                  ENDDO
                                  HEAD1=HEAD1*EDEPS/LATT_CUR%OMEGA*WDES%WTKPT(K1)
                               ENDIF
                               INTE_HARTREE=INTE_HARTREE+ &
                               &             2._q*((TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1))* &
                               &             GCONJG(TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1)))

                               INTE_HARTREE=0
                               DO I=1,3
                                  INTE_HARTREE=INTE_HARTREE+ &
                                 &             2._q*((TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1)+HEAD1(I,I))* &
                                 &             GCONJG(TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1)+HEAD1(I,I)))
                               ENDDO
                               INTE_HARTREE=INTE_HARTREE/3

                             ! WRITE(*,'(4I2,4F14.7)') NN1,N4,N3,N2,TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1)


                               HEAD2=0
                               IF (K1==K3) THEN
                                  CALL  CDER_BETWEEN_STATES_ROTATED( &
                                 &   CDER_BETWEEN_STATE12,LATT_CUR, K1, ISP, N3, N1)

                                  K2_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K2),KPOINTS_FULL_ORIG)
                                  CALL  CDER_BETWEEN_STATES_ROTATED( &
                                 &   CDER_BETWEEN_STATE34,LATT_CUR, K2_IN_FULL_ORIG, ISP, N2, N4)

                                  DO I=1,3
                                  DO J=1,3
                                     HEAD2(J,I)= HEAD2(J,I)+CDER_BETWEEN_STATE12(J)*CDER_BETWEEN_STATE34(I)
                                  ENDDO
                                  ENDDO
                                  HEAD2=HEAD2*EDEPS/LATT_CUR%OMEGA*WDES%WTKPT(K1)                               
                               ENDIF
                               INTE_EXCHANGE=0
                               DO I=1,3
                                  INTE_EXCHANGE=INTE_EXCHANGE- &
                                 &              REAL(GCONJG(TWOELECTRON3O(N3,N2, K2, N4, K4_STORE, NN1)+HEAD1(I,I))* &
                                 &              (TWOELECTRON3O(N2, N3, K3, N4, K4_STORE, NN1)+HEAD2(I,I)),KIND=q)
                               ENDDO
                               INTE_EXCHANGE=INTE_EXCHANGE/3

                               ECOR_HARTREE(MAX(N2,N3):WDES%NB_TOT)= &
                               &   ECOR_HARTREE(MAX(N2,N3):WDES%NB_TOT)+(OCC*VIRT*INTE_HARTREE/DENOM)

                               ECOR_EXCHANGE(MAX(N2,N3):WDES%NB_TOT)= &
                              &   ECOR_EXCHANGE(MAX(N2,N3):WDES%NB_TOT)+(OCC*VIRT*INTE_EXCHANGE/DENOM)
                               ECOR(MAX(N2,N3):WDES%NB_TOT)= &
                              &   ECOR(MAX(N2,N3):WDES%NB_TOT)+(OCC*VIRT*(INTE_HARTREE+INTE_EXCHANGE)/DENOM)
!                              INTE=2._q*(TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1)*GCONJG(TWOELECTRON3O(N3, N2, K2, N4, K4_STORE, NN1))) &
!                                   -REAL(GCONJG(TWOELECTRON3O(N3,N2, K2, N4, K4_STORE, NN1))*TWOELECTRON3O(N2, N3, K3, N4, K4_STORE, NN1),KIND=q)
!                              ECOR(MAX(N2,N3):WDES%NB_TOT)=ECOR(MAX(N2,N3):WDES%NB_TOT) &
!                                   +(OCC*VIRT*INTE/DENOM)                            
                            ENDIF

                         ENDDO
                      ENDDO
                   ENDIF
                ENDDO
             ENDDO
             ENDDO
          ENDDO
       ENDDO
       IF (IO%IU0>=0) WRITE(IO%IU0,*)
    ENDDO
    ENDDO
    IF (NKREDX>=2 .OR. NKREDY>=2 .OR. NKREDZ>=2) THEN
       ECOR=ECOR/(NKREDX*NKREDY*NKREDZ)
    ENDIF

    IF (KIMAGES/=0) THEN
       CALLMPI( M_sum_d(COMM_CHAIN, NFLOAT, 1) )
       CALLMPI( M_sum_d(COMM_CHAIN, EFOCK, 1) )
       CALLMPI( M_sum_d(COMM_CHAIN, ECOR_HARTREE(1), WDES%NB_TOT) )
       CALLMPI( M_sum_d(COMM_CHAIN, ECOR_EXCHANGE(1), WDES%NB_TOT) )
       CALLMPI( M_sum_d(COMM_CHAIN, ECOR(1), WDES%NB_TOT) )
    ENDIF

    IF (IO%IU0>=0) WRITE(IO%IU0,10) EFOCK,ECOR(WDES%NB_TOT),NFLOAT/1E9

    IF (IO%IU6>=0) THEN
       WRITE(IO%IU6,*)
       WRITE(IO%IU6,*) 'Moeller Plesset 2 correlation:'
       WRITE(IO%IU6,*) '================================'
       WRITE(IO%IU6,10) EFOCK,ECOR(WDES%NB_TOT),NFLOAT/1E9
       WRITE(IO%IU6,11) (N1,REAL(W%CELTOT(N1,1,1),q),EKIN(N1,1,1),EKIN_MAX(N1,1,1), &
      &                  ECOR_HARTREE(N1),ECOR_EXCHANGE(N1),ECOR(N1),N1=1,WDES%NB_TOT)
       WRITE(IO%IU6,*)
    ENDIF

10  FORMAT(' total Hartree Fock energy             ',F20.8 /&
           ' total correlation energy              ',F20.8 /&
           ' number of operations (through BLAS)   ',F20.3,' GFlops'/)
11  FORMAT(' band#       E_band        E_kin       E_kin(cwm)     E_MP2 (H)     E_MP2 (X)     E_MP2 (T)',/ (I6,6F14.6))
!11  FORMAT(' band#       E_band        E_kin       E_kin(cwm)      E_MP2',/ (I6,4F14.6))
    IF (WDES%DO_REDIS) CALL DEALLOCW(W_REDIS)
    DEALLOCATE(K3_, FOCKE, TWOELECTRON3O)

    CALL STOP_TIMING("G",IO%IU6,'4ORBIT')

  END SUBROUTINE TWOELECTRON4O_MAIN

!**********************************************************************
!
! determine the kinetic energy of each wavefunction (EKIN) 
! the kinetic energy of the component with maximal intensity (EKIN_MAX)
!
!**********************************************************************

  SUBROUTINE SET_EKIN(EKIN, EKIN_MAX, W )
    USE wave_high
    USE choleski
    REAL(q), POINTER :: EKIN(:,:,:), EKIN_MAX(:,:,:)
    TYPE (wavespin) W
  ! local
    TYPE (wavedes1)  WDES1           ! descriptor for one k-point
    INTEGER :: ISP, NK, N, NP, N_GLOBAL, M
    COMPLEX(q) :: C
    REAL(q) :: MAXCW, EKIN_, EKIN_MAX_
    INTEGER :: MFOUND

#ifdef MPI
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('SET_EKIN: KPAR>1 not implemented, sorry.') )
       STOP
    END IF
#endif
    ALLOCATE(EKIN(W%WDES%NB_TOT, W%WDES%NKPTS, W%WDES%ISPIN), &
             EKIN_MAX(W%WDES%NB_TOT, W%WDES%NKPTS, W%WDES%ISPIN))

    EKIN=0
    EKIN_MAX=0
    W%AUXTOT=0
    
    DO ISP=1,W%WDES%ISPIN
       DO NK=1,W%WDES%NKPTS
          CALL SETWDES(W%WDES,WDES1,NK)
          DO N=1,W%WDES%NBANDS
             MAXCW=0
             EKIN_MAX_=0
             EKIN_=0
             ISPINOR=0
!DIR$ IVDEP
!OCL NOVREL
             DO M=1,WDES1%NGVECTOR
                C=W%CW(M, N, NK, ISP)
                EKIN_=EKIN_+ REAL(C*CONJG(C) ,KIND=q) * WDES1%DATAKE(M,ISPINOR+1)
                IF ( REAL(C*CONJG(C),KIND=q) >= MAXCW ) THEN
                   MAXCW=REAL(C*CONJG(C),KIND=q)
                   EKIN_MAX_=WDES1%DATAKE(M,ISPINOR+1)
                   MFOUND=M
                ENDIF
             ENDDO
             N_GLOBAL=(N-1)*W%WDES%NB_PAR+W%WDES%NB_LOW
             EKIN(N_GLOBAL, NK, ISP)=EKIN_
             EKIN_MAX(N_GLOBAL , NK, ISP)=EKIN_MAX_
          ENDDO
       ENDDO
    ENDDO

    CALLMPI( M_sum_d( W%WDES%COMM_INTER, EKIN(1,1,1), W%WDES%NB_TOT*W%WDES%NKPTS*W%WDES%ISPIN))
    CALLMPI( M_sum_d( W%WDES%COMM_INTER, EKIN_MAX(1,1,1), W%WDES%NB_TOT*W%WDES%NKPTS*W%WDES%ISPIN))

!   W%AUXTOT=EKIN
    W%AUXTOT=EKIN_MAX

  END SUBROUTINE SET_EKIN

!************************ SUBROUTINE  TWOELECTRON4O_ACC  **************
! calculate
!
!       psi_N4,K4(r') int dr psi_N2,K2 *(r) psi_N1,K1(r) 1/|r-r'|
!
! for a specified k-point K1 and K2 and 
! bands N1=[NPOS1,NPOS1+NSTRIP1] and N2=[NPOS2,NPOS2+NSTRIP1]
! for all K4,N4
!
! the result is FFT to reciprocal space and the in product with
! all other bands is performed at the k-points
!
!    K3=K1-K2+K4
!
! the flag LMP2 restricts the orbitals N4,K4 to filled bands
! for those that have been calculated the FERWE are set to 1
! 
!**********************************************************************

  SUBROUTINE TWOELECTRON4O_ACC(LMDIM, LATT_CUR, W, W_REDIS, &
       FOCKE, P, ISP,  &
       K1, N1_GLOBAL, NSTRIP1, K2, NPOS2, NSTRIP2 , K3_, &
       TWOELECTRON3O, NFLOAT, LMP2, LMP2_EMPTY )
    USE sym_prec
    USE nonl_high
    USE wave_high
    USE lattice
    USE constant
    USE full_kpoints
    USE pseudo
    USE fock
    USE main_mpi
    IMPLICIT NONE

! passed variables
    INTEGER LMDIM
    TYPE (latt) LATT_CUR
    TYPE (wavespin) W, W_REDIS
    REAL(q) :: FOCKE(:,:)
    TYPE (potcar)      P(:)
    INTEGER ISP, K1, N1_GLOBAL, NSTRIP1, K2, NPOS2, NSTRIP2
    INTEGER K3_(:)
    GDEFS :: TWOELECTRON3O(:,:,:,:,:,:)
    REAL(q) :: NFLOAT
    LOGICAL LMP2, LMP2_EMPTY
! local variables
    INTEGER N1, N2, N3, N4, K4, KSTART, KSTRIDE, K4_STORE, K3
    INTEGER N, NPOS1, N2_GLOBAL, NLOC1, NLOC2
    TYPE (wavedes1), TARGET :: WDESK1, WDESK2, WDESK4, WDESK3
    TYPE (wavefun1) :: W3, WTMP
    TYPE (wavefuna),POINTER :: WA_(:)
    REAL(q) :: FSG                            ! singularity correction
    INTEGER ISPINOR
    INTEGER NCPU
    GDEF, ALLOCATABLE :: GWORK(:,:)           ! fock pot in real sp
    GDEF, ALLOCATABLE :: GWORK_PHASE(:,:)
    GDEF    :: GCHG (m_ GRIDHF%MPLWV)
    COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
    TYPE (wavefun1),ALLOCATABLE :: WIN1(:), WIN2(:)
    GDEF,      ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
    GDEF,      ALLOCATABLE :: CDIJ(:,:,:,:,:)  ! D_lml'm'
    GDEF,      ALLOCATABLE :: CDIJ_PHASE(:,:,:,:,:) ! D_lml'm'
    GDEF,ALLOCATABLE,TARGET:: CDLM(:)          ! D_LM
    REAL(q),   ALLOCATABLE :: POTFAK(:)        ! 1/(G+dk)**2 (G)
    INTEGER :: ierror
    LOGICAL    :: LPHASE
    COMPLEX(q) :: CXI(GRID_FOCK%MPLWV*W%WDES%NRSPINORS)
    GDEF    :: CHAM(W%WDES%NB_TOT,W%WDES%NB_TOT)
    TYPE (wavespin) WHF
    COMPLEX(q) :: CHARGE
    REAL(q)    :: DKX, DKY, DKZ
!==========================================================================
! initialisation
!==========================================================================
! use temporarily another WDES
    WHF=W
    WHF%WDES => WDES_FOCK
! first test whether K1-K2=q belongs to the allowed k-points
    K3_(:)=-1
    IF( SKIP_THIS_KPOINT_IN_FOCK(KPOINTS_FULL%VKPT(:,K2)-WHF%WDES%VKPT(:,K1))) RETURN

#ifdef MPI
    NCPU=WHF%WDES%NB_PAR
    IF (KIMAGES>0) THEN
       KSTART=COMM_CHAIN%NODE_ME
       KSTRIDE=KIMAGES
    ELSE
       KSTART=1
       KSTRIDE=1
    ENDIF

#ifdef MPI
    IF (W%WDES%COMM_KINTER%NCPU.NE.1) THEN
       CALLMPI( M_stop('TWOELECTRON4O_ACC: KPAR>1 not implemented, sorry.') )
!PK Consider also disabling the KIMAGES code as it is now redundant
       STOP
    END IF
#endif
#else
    NCPU=1
    KSTART=1
    KSTRIDE=1
#endif

! allocate memory, we have to do the acceleration on nstripn bands
! using strips of size m for the second band

    NLOC1=(CEILING(REAL(N1_GLOBAL+NSTRIP1-1)/REAL(NCPU))-FLOOR(REAL(N1_GLOBAL-1)/REAL(NCPU)))*NCPU
    NLOC2=NSTRIP2*NCPU
    NPOS1=FLOOR(REAL(N1_GLOBAL-1)/REAL(NCPU))+1

    ALLOCATE(GWORK(m_ GRIDHF%MPLWV,NSTRIP2*NCPU))
    ALLOCATE(GWORK_PHASE(m_ GRIDHF%MPLWV,NSTRIP2*NCPU))

    ALLOCATE( &
         CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         CDIJ(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS,NSTRIP2*NCPU), &
         CDIJ_PHASE(LMDIM,LMDIM,WHF%WDES%NIONS,WHF%WDES%NRSPINORS,NSTRIP2*NCPU), &
         CDLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
         POTFAK(GRIDHF%MPLWV),WA_(NLOC2))

    CALL SETWDES(WHF%WDES,WDESK1,K1)
    CALL SETWDES(WHF%WDES,WDESK2,K2)
    CALL SETWDES(WHF%WDES,WDESK3,0)

    ALLOCATE(WIN1(NLOC1))
    ALLOCATE(WIN2(NLOC2))
    DO N=1,NLOC1
       CALL NEWWAV(WIN1(N) , WDESK1,.TRUE.)
    ENDDO
    DO N=1,NLOC2
       CALL NEWWAV(WIN2(N) , WDESK2,.TRUE.)
       CALL NEWWAVA(WA_(N), WDESK3, W%WDES%NBANDS)
    ENDDO

    CALL NEWWAV_R(W3, WDESK3)

! average electrostatic potential for k=k' and n=n'
    FSG=FSG_STORE(1)

!gk mod: got rid of CALLMPI(MPI_barrier(WHF%WDES%COMM%MPI_COMM,ierror))
    statmg
!==========================================================================
! fourier transform the bands psi [NPOS1,POS1+NSTRIP1],K1 and distribute
! to all nodes
! same for psi [NPOS2,POS2+NSTRIP2],K2
!==========================================================================
!gK mod
    CALL W1_GATHER( WHF, NPOS1, CEILING(REAL(N1_GLOBAL+NSTRIP1-1)/REAL(NCPU)), ISP, WIN1)
    CALL W1_GATHER( WHF, NPOS2, NPOS2+NSTRIP2-1, ISP, WIN2)
!gK mod end

    addtmg(TVini,TCini)

! test
    IF (ENCUT4O==-1) THEN
       CALL SET_GFAC(GRIDHF,LATT_CUR,K1,K2,FSG,POTFAK)
    ELSE
       CALL SET_GFAC(GRIDHF,LATT_CUR,K1,K2,FSG,POTFAK,ENCUT=ENCUT4O)
    ENDIF
!   CALL SET_GFAC(GRIDHF,LATT_CUR,K1,K2,FSG,POTFAK)
! test
!==========================================================================
!  psi_N1,K1(r) psi*_N2,K2(r) 
!==========================================================================
    DO N1=1,NLOC1
       IF (((NPOS1-1)*NCPU+N1)<N1_GLOBAL.OR. &
      &    ((NPOS1-1)*NCPU+N1)>(N1_GLOBAL+NSTRIP1-1).OR. &
      &    EMPTY_MP2_ORBITAL(W%FERTOT((NPOS1-1)*NCPU+N1,K1,ISP))) CYCLE

       DO N2=1,NLOC2
          N2_GLOBAL=(NPOS2-1)*NCPU+N2
          IF (FILLED_MP2_ORBITAL(W%FERTOT(N2_GLOBAL,K2,ISP)).AND.LMP2_EMPTY) CYCLE

!gK mod
          CALL FOCK_CHARGE( WIN1(N1), WIN2(N2), GWORK(:,N2), CRHOLM)
!gK mod end
!==========================================================================
!  potential 1/|r-r'| of course done via two FFT's
!==========================================================================
      ! fft to reciprocal space
          CALL FFT3D(GWORK(1,N2),GRIDHF,-1)
          GCHG=GWORK(:,N2)
      ! multiply by 4 pi e^2/G^2 and divide by # of gridpoints to obtain potential
          CALL APPLY_GFAC(GRIDHF, GWORK(1,N2), POTFAK(1))

          FOCKE((NPOS1-1)*NCPU+N1-N1_GLOBAL+1,N2)=0
          DO N=1,GRIDHF%RC%NP
             FOCKE((NPOS1-1)*NCPU+N1-N1_GLOBAL+1,N2)= &
            &   FOCKE((NPOS1-1)*NCPU+N1-N1_GLOBAL+1,N2)+GCONJG(GCHG(N))*GWORK(N,N2)*(1.0_q/GRIDHF%NPLWV)
          ENDDO

      ! back to real space to get  \int psi_q(r) psi_k(r) / (r-r') d3r
          CALL FFT3D(GWORK(1,N2),GRIDHF,1)

      ! project the potential onto the local augmentation charges
          IF (WHF%WDES%LOVERL) THEN
             WTMP%CPROJ => CDLM(:)
             AUG_DES%RINPL= 1.0_q/GRIDHF%NPLWV ! multiplicator for RPRO1
             CALL RPRO1_HF(FAST_AUG_FOCK,AUG_DES, WTMP, GWORK(:,N2))
             IF (WHF%WDES%NRSPINORS==2) CDLM(AUG_DES%NPRO+1:AUG_DES%NPRO*2)=CDLM(1:AUG_DES%NPRO)
             ! transform D_LM -> D_lml'm'
             CALL CALC_DLLMM_TRANS(WHF%WDES, AUG_DES, TRANS_MATRIX_FOCK, CDIJ(:,:,:,:,N2), CDLM)
          ENDIF
       ENDDO
       addtmga(TVpot,TCpot)

!==========================================================================
!  now multiply by psi N4,K4 loop over all N4,K4
!==========================================================================

       DO K4=KSTART,WHF%WDES%NKPTS,KSTRIDE
          K3=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,K1)-W%WDES%VKPT(:,K2)+W%WDES%VKPT(:,K4),KPOINTS_FULL)
          K3_(K4)=K3
          CALL SETWDES(WHF%WDES,WDESK4,K4)
          CALL SETWDES(WHF%WDES,WDESK3,K3)

          ! unfortunately K1-K2+K4-K3 might be any reciprocal lattice vector G
          ! in that case the result is shifted by G with respect to the 
          ! internal data layout (FFTEXT), we now apply a shift e^iGr in real space
          ! to cure the problem
          ! SETPHASE calculate the phase shift if required and set LPHASE in this case

          CALL SETPHASE((W%WDES%VKPT(:,K1)-W%WDES%VKPT(:,K2)+W%WDES%VKPT(:,K4)-W%WDES%VKPT(:,K3)), &
               GRIDHF,CPHASE,LPHASE)

          IF (LPHASE) THEN
             ! calculate the phase shifted potential
             DO N2=1,NLOC2
                N2_GLOBAL=(NPOS2-1)*NCPU+N2
                IF (FILLED_MP2_ORBITAL(W%FERTOT(N2_GLOBAL,K2,ISP)).AND.LMP2_EMPTY) CYCLE

                CALL APPLY_PHASE_GDEF( GRIDHF, CPHASE(1), GWORK(1,N2), GWORK_PHASE(1,N2) )
             ENDDO

             ! also the CDIJ need to be recalculated with the proper phase shift
             ! yes things are complicated 
             ! this is required since the D lacks the contribtion  e(i q R_ion) see below
             IF (WHF%WDES%LOVERL) THEN
                CALL APPLY_PHASE_TO_DIJ( FAST_AUG_FOCK, WHF%WDES, &
                     (W%WDES%VKPT(:,K1)-W%WDES%VKPT(:,K2)+W%WDES%VKPT(:,K4)-W%WDES%VKPT(:,K3)), &
                     CDIJ, CDIJ_PHASE)
             ENDIF
          ENDIF

          DO N4=1,WHF%WDES%NBANDS
          ! in the MP2 case empty bands can be skipped
             IF (EMPTY_MP2_ORBITAL(WHF%FERWE(N4,K4,ISP)).AND.LMP2) CYCLE
             
             CALL SETWAV(WHF,W3,WDESK4,N4,ISP)  ! fill band N into W3(NP)
             CALL FFTWAV_W1( W3)

             DO N2=1,NLOC2
                N2_GLOBAL=(NPOS2-1)*NCPU+N2
                IF (FILLED_MP2_ORBITAL(W%FERTOT(N2_GLOBAL,K2,ISP)).AND.LMP2_EMPTY) CYCLE

                CXI=0
                IF (LPHASE) THEN
                   CALL VHAMIL_TRACE(WDESK4, GRID_FOCK, GWORK_PHASE(1,N2), W3%CR(1), & 
                        CXI(1), 1.0_q/GRIDHF%NPLWV)
                ELSE
                   CALL VHAMIL_TRACE(WDESK4, GRID_FOCK, GWORK(1,N2), W3%CR(1), & 
                        CXI(1), 1.0_q/GRIDHF%NPLWV)
                ENDIF

                DO ISPINOR=0,WDESK3%NRSPINORS-1
                   CALL FFTEXT(WDESK3%NGVECTOR,WDESK3%NINDPW(1), &
                        CXI(1+ISPINOR*GRID_FOCK%MPLWV), &
                        WA_(N2)%CW(1,N4) , GRID_FOCK, .FALSE.)
                ENDDO
             
                ! add D_lml'm' to kappa_lm_N (sum over l'm')
                IF (LPHASE) THEN
                   CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ_PHASE(1,1,1,1,N2), W3%CPROJ(1), WA_(N2)%CPROJ(1,N4),.FALSE.)
                ELSE
                   CALL OVERL_FOCK(WHF%WDES, LMDIM, CDIJ(1,1,1,1,N2), W3%CPROJ(1), WA_(N2)%CPROJ(1,N4),.FALSE.)
                ENDIF
             ENDDO
          ENDDO
!==========================================================================
! proceed with the inproduct <psi_N3,K3 | and store the four electron
! integral
!==========================================================================
          IF (KIMAGES/=0) THEN
             K4_STORE=CEILING(REAL(K4)/REAL(KIMAGES))
          ELSE
             K4_STORE=K4
          ENDIF

          DO N2=1,NLOC2
             N2_GLOBAL=(NPOS2-1)*NCPU+N2
             CALL  INPROD_BETWEEN_STATES(W%WDES, ELEMENTS(W_REDIS, WDESK3, ISP), WA_(N2), & 
                  NFLOAT, CHAM, 1, 1, N4UP=LAST_FILLED_MP2(W,LMP2,K4,ISP))
             DO N4=1,LAST_FILLED_MP2(W,LMP2,K4,ISP)
                DO N3=1,W%WDES%NB_TOT
                   TWOELECTRON3O(N3,  N2_GLOBAL,K2,  N4,K4_STORE, (NPOS1-1)*NCPU+N1-N1_GLOBAL+1)=CHAM(N3,N4)
                ENDDO
             ENDDO
          ENDDO
          
       ENDDO
    ENDDO

    DO N=1,NLOC1
       CALL DELWAV(WIN1(N) ,.TRUE.)
    ENDDO
    DO N=1,NLOC2
       CALL DELWAV(WIN2(N) ,.TRUE.)
       CALL DELWAVA(WA_(N))
    ENDDO
    CALL DELWAV_R(W3)

    DEALLOCATE(CRHOLM,CDIJ,CDIJ_PHASE,CDLM,POTFAK)
    DEALLOCATE(GWORK,GWORK_PHASE)
    DEALLOCATE(WIN1, WIN2, WA_)

  END SUBROUTINE TWOELECTRON4O_ACC


!***********************************************************************
!
!  calculate the inproduct between two set of wavefunctions
!  for a selected k-points and spin component
!
!    CHAM(n1,n2) = < WA1_n1 | WA2_n2 >
!
!  usually  WA1 is the wavefunction array, whereas WA2 holds
!  psi_N4,K4(r') int dr psi_N2,K2 *(r) psi_N1,K1(r) 1/|r-r'|
!  
!***********************************************************************

  SUBROUTINE  INPROD_BETWEEN_STATES(WDES, WA1, WA2, & 
      NFLOAT, CHAM, N3LOW, N4LOW, N4UP)
    USE wave_high
    USE mgrid
    USE hamil

    IMPLICIT NONE

    TYPE (wavefuna)    WA1, WA2       ! store Hamiltonian times wavefunction
    INTEGER NK, ISP  
    REAL(q) :: NFLOAT
    TYPE (wavedes )    WDES           ! descriptor for one k-point
    GDEF ::  CHAM(WDES%NB_TOT,WDES%NB_TOT)
    INTEGER :: N3LOW                  ! lower limit for N3 (not yet supported)
    INTEGER :: N4LOW                  ! lower limit for N4
    INTEGER :: N4UP                   ! upper limit for N4
  ! local
    INTEGER :: N, NP, N4, NSTRIP_ACT

    CALL REDISTRIBUTE_PW( WA2)
    CALL REDISTRIBUTE_PROJ( WA2)

    CHAM=0
    strip: DO N4=N4LOW,N4UP,NSTRIP_STANDARD_GLOBAL
       NSTRIP_ACT=MIN(N4UP+1-N4,NSTRIP_STANDARD_GLOBAL)

       CALL ORTH2( &
            WA1%CW_RED(1,1),WA2%CW_RED(1,N4),WA1%CPROJ_RED(1,1), &
            WA2%CPROJ_RED(1,N4),WDES%NB_TOT, &
            N4, NSTRIP_ACT, WA1%WDES1%NPL_RED,WA1%WDES1%NPRO_O_RED,WA1%WDES1%NRPLWV_RED,WA1%WDES1%NPROD_RED,CHAM(1,1))

       NFLOAT=NFLOAT+WDES%NB_TOT*NSTRIP_ACT*WA1%WDES1%NRPLWV_RED*8

!       CALL ORTH1('U', &
!            WA1%CW_RED(1,1),WA2%CW_RED(1,N4),WA1%CPROJ_RED(1,1), &
!            WA2%CPROJ_RED(1,N4),WDES%NB_TOT, &
!            N4, NSTRIP_ACT, WA1%WDES1%NPL_RED,WA1%WDES1%NPRO_O_RED,WA1%WDES1%NRPLWV_RED,WA1%WDES1%NPROD_RED,CHAM(1,1))

    ENDDO strip
    CALLMPI( M_sum_g(WDES%COMM,CHAM(1,1),WDES%NB_TOT*N4UP))
  END SUBROUTINE INPROD_BETWEEN_STATES

!***********************************************************************
!  Function EMPTY_MP2_ORBITAL returns .TRUE. if the orbital
!  is considered to be empty in the MP2
!  returns true only if the orbital occupancy is 0
!  EMPTY_MP2_ORBITAL's are skipped when an occupied band is required
!***********************************************************************

  FUNCTION EMPTY_MP2_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL EMPTY_MP2_ORBITAL
    REAL(q) :: F
    IF (ABS(F) <MP2_EMPTY_THRESHHOLD) THEN
       EMPTY_MP2_ORBITAL=.TRUE.
    ELSE
       EMPTY_MP2_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION EMPTY_MP2_ORBITAL

!***********************************************************************
!  Function FILLED_MP2_ORBITAL returns .TRUE. if the orbital
!  is considered to be filled in the MP2
!  returns true only if the orbital occupancy is 1
!  FILLED_MP2_ORBITAL's are skipped when an unoccupied band is required
!***********************************************************************

  FUNCTION FILLED_MP2_ORBITAL( F)
    USE prec
    IMPLICIT NONE
    LOGICAL FILLED_MP2_ORBITAL
    REAL(q) :: F
    IF (ABS(F-1) <MP2_EMPTY_THRESHHOLD) THEN
       FILLED_MP2_ORBITAL=.TRUE.
    ELSE
       FILLED_MP2_ORBITAL=.FALSE.
    ENDIF
  END FUNCTION FILLED_MP2_ORBITAL

!***********************************************************************
!  function LAST_FILLED_MP2 returns the last filled band
!  modulo the number of bands done in parallel
!***********************************************************************

  FUNCTION LAST_FILLED_MP2( W, LMP2, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_MP2
    LOGICAL LMP2
    INTEGER K1, ISP
  ! local
    INTEGER NB
    IF (.NOT. LMP2) THEN
       LAST_FILLED_MP2=W%WDES%NB_TOT
    ELSE
       DO NB=0, W%WDES%NB_TOT-1, W%WDES%NB_PAR
          ! if first band in block is empty no longer required
          IF (EMPTY_MP2_ORBITAL(W%FERTOT( NB+1, K1, ISP))) EXIT
       ENDDO
       LAST_FILLED_MP2=NB
    ENDIF
  END FUNCTION LAST_FILLED_MP2

!***********************************************************************
!  function LAST_FILLED_ACTUAL_MP2 returns the last filled band
!***********************************************************************

  FUNCTION LAST_FILLED_ACTUAL_MP2( W, LMP2, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER LAST_FILLED_ACTUAL_MP2
    LOGICAL LMP2
    INTEGER K1, ISP
  ! local
    INTEGER NB
!   IF (.NOT. LMP2) THEN
!      LAST_FILLED_ACTUAL_MP2=W%WDES%NB_TOT
!   ELSE
       DO NB=0, W%WDES%NB_TOT-1
          IF (EMPTY_MP2_ORBITAL(W%FERTOT( NB+1, K1, ISP))) EXIT
       ENDDO
       LAST_FILLED_ACTUAL_MP2=NB
!   ENDIF
  END FUNCTION LAST_FILLED_ACTUAL_MP2

!***********************************************************************
!  function FIRST_EMPTY_MP2 returns the first empty band
!  modulo the number of bands done in parallel
!***********************************************************************

  FUNCTION FIRST_EMPTY_MP2( W, LMP2, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_MP2
    LOGICAL LMP2
    INTEGER K1, ISP
  ! local
    INTEGER NB
    IF (.NOT. LMP2) THEN
       FIRST_EMPTY_MP2=1
    ELSE
       DO NB=W%WDES%NB_TOT, 1,-W%WDES%NB_PAR
          ! if last band in block is filled block no longer required
          IF (FILLED_MP2_ORBITAL(W%FERTOT( NB, K1, ISP))) EXIT
       ENDDO
       FIRST_EMPTY_MP2=NB+1
    ENDIF
  END FUNCTION FIRST_EMPTY_MP2

!***********************************************************************
!  function FIRST_EMPTY_ACTUAL_MP2 returns the first empty band
!***********************************************************************

  FUNCTION FIRST_EMPTY_ACTUAL_MP2( W, LMP2, K1, ISP)
    USE prec
    USE wave
    IMPLICIT NONE
    TYPE (wavespin) W
    INTEGER FIRST_EMPTY_ACTUAL_MP2
    LOGICAL LMP2
    INTEGER K1, ISP
  ! local
    INTEGER NB
!   IF (.NOT. LMP2) THEN
!      FIRST_EMPTY_ACTUAL_MP2=1
!   ELSE
       DO NB=W%WDES%NB_TOT, 1, -1
          ! if last band in block is filled block no longer required
          IF (FILLED_MP2_ORBITAL(W%FERTOT( NB, K1, ISP))) EXIT
       ENDDO
       FIRST_EMPTY_ACTUAL_MP2=NB+1
!   ENDIF
  END FUNCTION FIRST_EMPTY_ACTUAL_MP2

!***********************************************************************
!
!  calculate the phase factor
!   e^(i q r)
!  for a reciprocal lattice vector q supplied in the array VKPT
!  the calculate phase factor is returned in the array CPHASE
!  and LPHASE is set to .TRUE. if VKPT is not 0
!
!***********************************************************************

  SUBROUTINE SETPHASE(VKPT, GRID, CPHASE, LPHASE)
    USE constant

    IMPLICIT NONE

    REAL(q) :: VKPT(3)
    TYPE (grid_3d) GRID
    COMPLEX(q) :: CPHASE(GRID%MPLWV)
    LOGICAL LPHASE
  ! local
    REAL(q),PARAMETER :: TINY=1E-6_q
    REAL(q) F1, F2, F3
    INTEGER NC, N, IND
    COMPLEX(q) C, CD, CSUM
    CSUM=0
#ifdef gammareal
    LPHASE=.FALSE.
    RETURN
#endif
    
    IF (ABS(VKPT(1))>TINY .OR. ABS(VKPT(2))>TINY .OR. ABS(VKPT(3))>TINY) THEN
       IF (ABS(MOD(VKPT(1)+100.5_q,1._q)-0.5_q)>TINY .OR. &
           ABS(MOD(VKPT(2)+100.5_q,1._q)-0.5_q)>TINY .OR. &
           ABS(MOD(VKPT(3)+100.5_q,1._q)-0.5_q)>TINY) THEN
           WRITE(*,*)'internal error: non integer shift in SETPHASE in twoelectron4o.F',VKPT(:)
           STOP
       ENDIF
       LPHASE=.TRUE.
       F1=TPI/GRID%NGX*VKPT(1)
       F2=TPI/GRID%NGY*VKPT(2)
       F3=TPI/GRID%NGZ*VKPT(3)

       IF (GRID%RL%NFAST==3) THEN
          CD=EXP(CMPLX(0,F3,q))
          IND=0
          DO NC=1,GRID%RL%NCOL
             C=EXP(CMPLX(0,F1*(GRID%RL%I2(NC)-1)+F2*(GRID%RL%I3(NC)-1),q))
             DO N=1,GRID%RL%NROW
                IND=IND+1
                CPHASE(IND)=C
                C=C*CD
             ENDDO
          ENDDO

       ELSE
          CD=EXP(CMPLX(0,F1,q))
          IND=0
          DO NC=1,GRID%RL%NCOL
             C=EXP(CMPLX(0,F2*(GRID%RL%I2(NC)-1)+F3*(GRID%RL%I3(NC)-1),q))
             DO N=1,GRID%RL%NROW
                IND=IND+1
                CPHASE(IND)=C
                CSUM=CSUM+C
                C=C*CD
             ENDDO
          ENDDO
       ENDIF
       LPHASE=.TRUE.
    ELSE
       LPHASE=.FALSE.
    ENDIF

  END SUBROUTINE SETPHASE


  SUBROUTINE SETPHASE_NOCHK(VKPT, GRID, CPHASE, LPHASE)
    USE constant

    IMPLICIT NONE

    REAL(q) :: VKPT(3)
    TYPE (grid_3d) GRID
    COMPLEX(q) :: CPHASE(GRID%MPLWV)
    LOGICAL LPHASE
  ! local
    REAL(q),PARAMETER :: TINY=1E-6_q
    REAL(q) F1, F2, F3
    INTEGER NC, N, IND
    COMPLEX(q) C, CD, CSUM
    CSUM=0
#ifdef gammareal
    LPHASE=.FALSE.
    RETURN
#endif
    
    IF (ABS(VKPT(1))>TINY .OR. ABS(VKPT(2))>TINY .OR. ABS(VKPT(3))>TINY) THEN
       LPHASE=.TRUE.
       F1=TPI/GRID%NGX*VKPT(1)
       F2=TPI/GRID%NGY*VKPT(2)
       F3=TPI/GRID%NGZ*VKPT(3)

       IF (GRID%RL%NFAST==3) THEN
          CD=EXP(CMPLX(0,F3,q))
          IND=0
          DO NC=1,GRID%RL%NCOL
             C=EXP(CMPLX(0,F1*(GRID%RL%I2(NC)-1)+F2*(GRID%RL%I3(NC)-1),q))
             DO N=1,GRID%RL%NROW
                IND=IND+1
                CPHASE(IND)=C
                C=C*CD
             ENDDO
          ENDDO

       ELSE
          CD=EXP(CMPLX(0,F1,q))
          IND=0
          DO NC=1,GRID%RL%NCOL
             C=EXP(CMPLX(0,F2*(GRID%RL%I2(NC)-1)+F3*(GRID%RL%I3(NC)-1),q))
             DO N=1,GRID%RL%NROW
                IND=IND+1
                CPHASE(IND)=C
                CSUM=CSUM+C
                C=C*CD
             ENDDO
          ENDDO
       ENDIF
       LPHASE=.TRUE.
    ELSE
       LPHASE=.FALSE.
    ENDIF

  END SUBROUTINE SETPHASE_NOCHK
!
! apply phase works only in complex mode
! anyhow it is bypassed in the real mode
!

  SUBROUTINE APPLY_PHASE(GRID, SV, CR , CRP)
    USE prec
    USE mgrid
    USE wave
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    COMPLEX(q) ::  SV(DIMREAL(GRID%MPLWV)) ! complex phase factor
    COMPLEX(q) ::  CR(GRID%MPLWV)
    COMPLEX(q) ::  CRP(GRID%MPLWV)
    INTEGER NRSPINORS
    ! local variables
    INTEGER ISPINOR, M, MM

#ifdef gammareal
    RETURN
#endif

!DIR$ IVDEP
!OCL NOVREL
    DO M=1,GRID%RL%NP
       CRP(M)= SV(M) *CR(M)
    ENDDO
  END SUBROUTINE APPLY_PHASE


!
! identical version but with GDEF
!

  SUBROUTINE APPLY_PHASE_GDEF(GRID, SV, CR , CRP)
    USE prec
    USE mgrid
    USE wave
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    COMPLEX(q) ::  SV(DIMREAL(GRID%MPLWV)) ! complex phase factor
    GDEF ::  CR(GRID%MPLWV)
    GDEF ::  CRP(GRID%MPLWV)
    INTEGER NRSPINORS
    ! local variables
    INTEGER ISPINOR, M, MM

#ifdef gammareal
    RETURN
#endif

!DIR$ IVDEP
!OCL NOVREL
    DO M=1,GRID%RL%NP
       CRP(M)= SV(M) *CR(M)
    ENDDO
  END SUBROUTINE APPLY_PHASE_GDEF


!***********************************************************************
!
!  apply the phase factor
!   e^(i q R_i)
!  to the ions
!  this is required since the potential is shifted by
!   V'(r) =e^(i q r) V(r)
!  if one would project this onto the local compensation charges
!  one would get
!  \int dr V'(r) e^(-i q (r-R_ion)) Q(r-R_ion)
!  as the shifts are calculated with respect to the central ion in
!  each PAW sphere
!  hence a shift of e^(i q R_ion) is missing in CDIJ
!
!***********************************************************************



  SUBROUTINE APPLY_PHASE_TO_DIJ( FAST_AUG_FOCK, WDES, VKPT, CDIJ, CDIJ_PHASE)
    USE constant

    IMPLICIT NONE
    TYPE (nonlr_struct) FAST_AUG_FOCK
    TYPE (wavedes) WDES
    REAL(q) :: VKPT(3)
    GDEF CDIJ(:,:,:,:,:)
    GDEF CDIJ_PHASE(:,:,:,:,:)
  ! local
    COMPLEX (q) CSHIFT
    INTEGER NIS, NT, LMMAXC, NI, N1, N2

    CDIJ_PHASE=0
    
    NIS=1
    type: DO NT=1,FAST_AUG_FOCK%NTYP
       LMMAXC=WDES%LMMAX(NT)
       IF (LMMAXC/=0) THEN
          ions: DO NI=NIS,FAST_AUG_FOCK%NITYP(NT)+NIS-1
             CSHIFT=EXP(CITPI*(VKPT(1)*FAST_AUG_FOCK%POSION(1,NI)+ & 
                               VKPT(2)*FAST_AUG_FOCK%POSION(2,NI)+ &
                               VKPT(3)*FAST_AUG_FOCK%POSION(3,NI)))
             CDIJ_PHASE(:,:,NI,:,:)=CDIJ(:,:,NI,:,:)*CSHIFT
          ENDDO ions
       ENDIF
       NIS = NIS+FAST_AUG_FOCK%NITYP(NT)
    ENDDO type
    
  END SUBROUTINE APPLY_PHASE_TO_DIJ

END MODULE twoelectron4o

!***********************************************************************
!
! apply phase works only in complex mode
! anyhow it is bypassed in the real mode
! this version performs an in place multiplication
! and has no explicit interface
!
!***********************************************************************

  SUBROUTINE APPLY_PHASE_INPLACE(GRID, SV, CR )
    USE prec
    USE mgrid
    USE wave
    IMPLICIT NONE

    TYPE (grid_3d)     GRID
    COMPLEX(q) ::  SV(DIMREAL(GRID%MPLWV)) ! complex phase factor
    COMPLEX(q) ::  CR(GRID%MPLWV)
    INTEGER NRSPINORS
    ! local variables
    INTEGER ISPINOR, M, MM

#ifdef gammareal
    RETURN
#endif

!DIR$ IVDEP
!OCL NOVREL
    DO M=1,GRID%RL%NP
       CR(M)= SV(M) *CR(M)
    ENDDO
  END SUBROUTINE APPLY_PHASE_INPLACE
