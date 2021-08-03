#include "symbol.inc"
MODULE mp2
      USE prec
      USE fock
      USE chi_base      
      USE lattice
      USE wpot
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
! rewritten for scaLAPACK by aG in 2008
!**********************************************************************
      
      ! FTOD_PW Stores the two-orbital response functions at q 
      ! FTOD_OC Stores the one center part of the two-orbital response functions at q 
      ! <ij|ab> (q=k_b-k_j), FTOD_PW=(ia)   
      !FTOD_PW(NGVECTOR,NBANDS,VBMAX,NKPTS,NKPTS(KQ),ISPIN)
      !FTOD_OC(H%TOTAL_ENTRIES,NBANDS,VBMAX,NKPTS,NKPTS(KQ),ISPIN)
      COMPLEX(q) , ALLOCATABLE, PRIVATE, SAVE :: FTOD_PW(:,:,:,:,:,:,:)      
      GDEF      , ALLOCATABLE, PRIVATE, SAVE :: FTOD_OC(:,:,:,:,:,:,:)      
      GDEF , ALLOCATABLE, PRIVATE, SAVE :: TWOE4ORBITAL(:,:)
      GDEF , ALLOCATABLE, PRIVATE, SAVE :: TWOE4ORBITAL_X(:,:)
      INTEGER, PRIVATE, SAVE :: NGVECTOR, NHVECTOR
      ! fac1,fac2 are used in order to distinguish between restricted(ISPIN=1) and
      ! unrestricted(WDES%ISPIN=2) MP2.
      REAL(q) :: FAC1,FAC2
      REAL(q) :: SCALE
      !Scalapack array descriptor for FTOD__ before redistribution
      integer, dimension(9)   :: desc_FTOD_PW_br, desc_FTOD_OC_br
      !Scalapack array descriptor for FTOD__ after redistribution
      integer, dimension(9)   :: desc_FTOD_PW, desc_FTOD_OC
      !Scalapack array descriptor for TWOE4ORBITAL
      integer, dimension(9)   :: desc_TWOE4ORBITAL
      !Scalapack array descriptor for TWOE4ORBITAL_X
      integer, dimension(9)   :: desc_TWOE4ORBITAL_X
      
      !BLACS related variables and function
      !PROCS.. number of processors, ME... processor number, NPROW... number of rows in process grid
      !NPCOL... number of columns in process grid, myrow,mycol... my coordinates in process grid
      !mb... blocking size of rows for block cyclic distribution
      !nb... blocking size of columns for block cyclic distribution      
      INTEGER, PRIVATE, SAVE :: PROCS,ME,NPROW,NPCOL,MYROW,MYCOL,CONTXT,CONTXT_GRID
      INTEGER, PRIVATE, SAVE :: NPROW_GRID, ncc
      INTEGER, PRIVATE, SAVE :: MB, NB, VBMAX
      INTEGER, EXTERNAL :: NUMROC
      !numroc... function that returns the number of rows or columns of the LOCAL ARRAY if you enter
      !the corresponding properties of the GLOBAL ARRAY
      COMPLEX(q) :: E_MP2, energy_c, energy_x,energy_cc,EFOCK, tmp2
      COMPLEX(q), ALLOCATABLE, PRIVATE, SAVE :: E_homo_h(:,:), E_lumo_h(:,:), E_homo_x(:,:), E_lumo_x(:,:)
      COMPLEX(q), ALLOCATABLE, PRIVATE, SAVE :: Z_homo_h(:,:),Z_homo_x(:,:),Z_lumo_h(:,:),Z_lumo_x(:,:)
      REAL(q) :: Delta_finit
      INTEGEr, ALLOCATABLE, PRIVATE, SAVE  :: HOMO(:), LUMO(:)
      
      INTEGER :: HEAD_KINTER
      INTEGEr :: HEAD_NKPTS
      REAL(q), ALLOCATABLE :: VKPT_HEAD(:,:) !k-mesh which will be used for calculating the lim_q->0 <ij|ab>
      REAL(q), ALLOCATABLE :: WTKPT_HEAD(:) !k-mesh which will be used for calculating the lim_q->0 <ij|ab>
      
      LOGICAL :: MP2_bandstructure
      TYPE(one_center_handle), POINTER, PRIVATE, SAVE :: H

#ifdef scaLAPACK
      PRIVATE :: CALC_2ORBITAL_FTOD,REDISTRIBUTE_FTOD_GRID,INIT_BLACS_COLS,INIT_BLACS_GRID

      CONTAINS

!***********************************************************************
!Main MP2 routine.        
!Here the LOOPS over the spin index(ISP), k-point indices (kq,ki,kj)
!and valence band indices(NBI,NBJ) are performed. 
!For calculating the two-electron
!4 orbital integrals scalaLAPCK is used.
!***********************************************************************

      SUBROUTINE CALCULATE_MP2(P,WDES,W,LATT_CUR,T_INFO,IO,KPOINTS,WGW, ENCUTGW, ENCUTGWSOFT, LMAXMP2, SYMM)
         USE prec
         USE poscar
         USE pseudo
         USE wave_high
         USE full_kpoints
         USE base
         USE mkpoints
         USE mpimy
         USE ini
         IMPLICIT NONE
         TYPE(wavedes) WDES
         TYPE(wavedes) WGW
         TYPE(wavespin) W        
         TYPE(type_info) T_INFO
         TYPE(potcar) P(T_INFO%NTYP)
         TYPE(latt) LATT_CUR
         INTEGER LMAXMP2         
         REAL(q) :: ENCUTGW, ENCUTGWSOFT
         TYPE (symmetry)    SYMM
         TYPE (in_struct) IO
         TYPE (kpoints_struct) KPOINTS
         !local variables
         TYPE (wavespin) WHF
         INTEGER KI,KQ,KQ_,KA,KB,ISP,KJ,NBI,NBJ
         LOGICAL :: qchange
         REAL(q) :: test         
         !write(*,*) 'lmaxmp2',lmaxmp2
         E_MP2=0
         EFOCK=0
         energy_c=0
         energy_cc=0
         energy_x=0
#ifdef gammareal
   ncc=1
#else
   ncc=2
#endif
         MP2_bandstructure=.FALSE.
         IF (MP2_bandstructure) THEN
            IF (SYMM%ISYM>=0) THEN
               CALLMPI( M_stop('CALCULATE_MP2: MP2 QP energies are not implemented for ISYM>=0. Redo all calculations setting ISYM=-1 in the INCAR file.') )
               STOP
            ENDIF
            IF (WDES%ISPIN>1) THEN
               CALLMPI( M_stop('CALCULATE_MP2: MP2 QP energies are not implemented for ISPIN>1.') )
               STOP
            ENDIF
         ENDIF
         
         IF (WDES%ISPIN>1) MP2_bandstructure=.FALSE.
         CALL CHECK_FULL_KPOINTS ! all set up properly ?
         CALL START_TIMING("LOOP")
         SCALE=KPOINTS_FULL%WTKPT(1)*NKREDX*NKREDY*NKREDZ        
         IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2.0_q
         
         ALLOCATE(HOMO(WDES%ISPIN))
         ALLOCATE(LUMO(WDES%ISPIN))
         IF (MP2_bandstructure) CALL SETUP_MP2_bands(WDES)
         
         !Initialize the 1D process grid
         CALL INIT_BLACS_COLS()
         !Calculate the FTOD functions
         CALL CALC_2ORBITAL_FTOD(WDES,WGW,W,P,T_INFO,LATT_CUR,LMAXMP2,ENCUTGW,ENCUTGWSOFT,IO,FSG_STORE(1))         
         
                  
         CALL SETUP_TWOE4ORBITAL(WDES)   
         FAC1=1.0_q
         FAC2=2.0_q
         IF (WDES%ISPIN==2) THEN
            FAC1=0.5_q
            FAC2=1.0_q
         ENDIF
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*)
            IF (.NOT. MP2_bandstructure) WRITE(IO%IU0,*)'Calculating MP2 energy:'
            IF (MP2_bandstructure) WRITE(IO%IU0,*)'Calculating MP2 quasiparticle bandstructure:'
         ENDIF
         
         spin: DO ISP=1,WDES%ISPIN
         kqloop: DO KQ=1,WDES%NKPTS
            CALL START_TIMING("G")
            IF (IO%IU0>=0) WRITE(IO%IU0,*)
            IF (IO%IU0>=0) THEN
               IF (WDES%ISPIN==1) THEN
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') KQ,KPOINTS%VKPT(:,KQ)
               ELSE
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ",A1,A1,", ")') KQ,KPOINTS%VKPT(:,KQ),ISP
               ENDIF
            ENDIF
            kiloop: DO KI=1,KPOINTS_ORIG%NKPTS
               IF (WDES%WTKPT(KI)==0) CYCLE
               ! k_a = k_i - k_q - G
               
               KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL)
               
               kjloop: DO KJ=1,WDES%NKPTS
               
                  CALL GWPROGRESS(IO%IU0, KI,KPOINTS_ORIG%NKPTS,KJ,WDES%NKPTS)
                  KB=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KJ)+WDES%VKPT(:,KQ),KPOINTS_FULL)
                  ! k_q' = k_a - k_j - G
                  KQ_=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KB),KPOINTS_FULL)
                  !KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL) 
                  ! k_b = k_j + k_q + G
                  
                  DO NBI=1,VBMAX !loop over valence bands i                     
                     IF ((.NOT. MP2_bandstructure) .AND. (EMPTY_MP2_ORBITAL(W%FERTOT(NBI,KI,ISP)))) CYCLE
                     
                     DO NBJ=1,VBMAX !loop over valence bands j
                        IF ((.NOT. MP2_bandstructure) .AND. (EMPTY_MP2_ORBITAL(W%FERTOT(NBJ,KJ,ISP)))) CYCLE
                        ! Calculate TWOE4ORBITAL <ij|ab> for all bands a and b at
                        ! k-points k_a and k_b, respectively. Here i, j, k_i and
                        ! k_j are fixed 

#ifdef gammareal
         IF ((NBJ>NBI) .and. (WDES%ISPIN==1) .and. (.NOT.MP2_bandstructure)) CYCLE
#else

#endif

#ifdef gammareal
                        CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          m_ NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP,ncc),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                        CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP,ncc),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        IF (ASSOCIATED(H)) THEN
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP,2),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP,2),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        ENDIF
                        TWOE4ORBITAL=GCONJG(TWOE4ORBITAL*SCALE)
                        ! Calculate the exchange-like part TWOE4ORBITAL_X <ij|ba>,
                        ! which is simply the transposed TWOE4ORBITAL matrix
                        ! for the case that k_q=k_q' (k_a-k_i=k_b-k_i)
                        IF (KQ==KQ_) THEN
#ifdef gammareal
                           CALL PDTRAN(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                             TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL,zero,&
                             TWOE4ORBITAL_X(1,1),1,1,desc_TWOE4ORBITAL_X)
#else
                           CALL PZTRANU(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                             TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL,zero,&
                             TWOE4ORBITAL_X(1,1),1,1,desc_TWOE4ORBITAL_X)
#endif
                        ENDIF

                        ! If k_q/=k_q', we have to calculate the exchange-like part
                        ! <ij|ba> from FTOD functions at k_q' 
                        
                        ! in fact what is calculated in the next line is:
                        ! <ba|ij>
                        IF (KQ/=KQ_) THEN
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),&
                             (PROCS*WDES%NBANDS),&
                             m_ NGVECTOR,-one, FTOD_PW(1,1,NBJ,KJ,KQ_,ISP,ncc),1,1, &
                             desc_FTOD_PW,FTOD_PW(1,1,NBI,KI,KQ_,ISP,1),1,1,&
                             desc_FTOD_PW,zero, TWOE4ORBITAL_X(1,1),1,1,&
                             desc_TWOE4ORBITAL_X)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),&
                             (PROCS*WDES%NBANDS),&
                             NGVECTOR,-one, FTOD_PW(1,1,NBJ,KJ,KQ_,ISP,ncc),1,1, &
                             desc_FTOD_PW,FTOD_PW(1,1,NBI,KI,KQ_,ISP,1),1,1,&
                             desc_FTOD_PW,zero, TWOE4ORBITAL_X(1,1),1,1,&
                             desc_TWOE4ORBITAL_X)
#endif
                           IF (ASSOCIATED(H)) THEN
#ifdef gammareal
                              CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                               NHVECTOR,-one, FTOD_OC(1,1,NBJ,KJ,KQ_,ISP,2),1,1,&
                               desc_FTOD_OC,FTOD_OC(1,1,NBI,KI,KQ_,ISP,1),1,1,&
                               desc_FTOD_OC,one, TWOE4ORBITAL_X(1,1),1,1,desc_TWOE4ORBITAL_X)
#else
                              CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                               NHVECTOR,-one, FTOD_OC(1,1,NBJ,KJ,KQ_,ISP,2),1,1,&
                               desc_FTOD_OC,FTOD_OC(1,1,NBI,KI,KQ_,ISP,1),1,1,&
                               desc_FTOD_OC,one, TWOE4ORBITAL_X(1,1),1,1,desc_TWOE4ORBITAL_X)
#endif
                           ENDIF
                           
                           TWOE4ORBITAL_X(:,:)=(TWOE4ORBITAL_X(:,:)*SCALE)
                           
                        ENDIF

                        !ADD_MP2 comes here
                        IF (MP2_bandstructure) THEN
                           call ADD_MP2_wb(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP,LATT_CUR,FSG_STORE(1))
                        ELSE
                           call ADD_MP2(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP,LATT_CUR,WDES%ISPIN)
                        ENDIF
                     ENDDO !nbj

                     IF (ISP==2) THEN
                        IF ((.NOT. MP2_bandstructure) .AND. (EMPTY_MP2_ORBITAL(W%FERTOT(NBI,KI,ISP)))) CYCLE
                        DO NBJ=1,VBMAX !loop over valence bands j
                           IF ((.NOT. MP2_bandstructure) .AND. (EMPTY_MP2_ORBITAL(W%FERTOT(NBJ,KJ,ISP-1)))) CYCLE
                           !TWOE4ORBITAL(:,:)=0
                           !there is an additional term in the expression 
                           !of the MP2 energy for the unrestricted case
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          m_ NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP-1,ncc),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP-1,ncc),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        IF (ASSOCIATED(H)) THEN
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP-1,2),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP-1,2),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        ENDIF

                        TWOE4ORBITAL(:,:)=GCONJG(TWOE4ORBITAL(:,:)*SCALE)

                        IF (MP2_bandstructure) THEN
                           call ADD_MP2_wb(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP-1,LATT_CUR,FSG_STORE(1))
                        ELSE
                           call ADD_MP2(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP-1,LATT_CUR,WDES%ISPIN)
                        ENDIF
                        ENDDO ! NBJ loop over valence bands j only
                     ENDIF

                  ENDDO ! NBI lop over valence bands i only

               ENDDO kjloop
               
            ENDDO kiloop
            CALL STOP_TIMING("G",IO%IU6,"MP2")
         ENDDO kqloop
         ENDDO spin
         
         CALL STOP_TIMING("LOOP",IO%IU6,XMLTAG='total')
         CALL M_sum_z(WGW%COMM_INTER, energy_c, 1)
         CALL M_sum_z(WGW%COMM_INTER, energy_cc, 1)
         CALL M_sum_z(WGW%COMM_INTER, energy_x, 1)
         CALL M_sum_z(WGW%COMM_INTER, EFOCK, 1)
         E_MP2=energy_c+energy_x
         IF (MP2_bandstructure) THEN
            DO ISP=1,WDES%ISPIN
              DO KI=1,WDES%NKPTS
                CALL M_sum_z(WGW%COMM_INTER, E_HOMO_h(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, E_HOMO_x(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, E_LUMO_h(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, E_LUMO_x(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, Z_HOMO_h(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, Z_HOMO_x(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, Z_LUMO_h(KI,ISP), 1)
                CALL M_sum_z(WGW%COMM_INTER, Z_LUMO_x(KI,ISP), 1)
             
                !CALCulate Z=1/(1-d Sigma/ d omega)
                Z_HOMO_h(KI,ISP)=(Z_HOMO_h(KI,ISP)-E_HOMO_h(KI,ISP))/DELTA_FINIT             
                Z_HOMO_x(KI,ISP)=(Z_HOMO_x(KI,ISP)-E_HOMO_x(KI,ISP))/DELTA_FINIT
                Z_LUMO_h(KI,ISP)=(Z_LUMO_h(KI,ISP)-E_LUMO_h(KI,ISP))/DELTA_FINIT
                Z_LUMO_x(KI,ISP)=(Z_LUMO_x(KI,ISP)-E_LUMO_x(KI,ISP))/DELTA_FINIT
             
                Z_HOMO_h(KI,ISP)=1.0_q/(1.0_q-Z_HOMO_h(KI,ISP))
                Z_HOMO_x(KI,ISP)=1.0_q/(1.0_q-Z_HOMO_x(KI,ISP))
                Z_LUMO_h(KI,ISP)=1.0_q/(1.0_q-Z_LUMO_h(KI,ISP))
                Z_LUMO_x(KI,ISP)=1.0_q/(1.0_q-Z_LUMO_x(KI,ISP))
              ENDDO
            ENDDO
         ENDIF
         IF (IO%IU6>0) THEN
10          FORMAT(//" BLAS level 3 operations / number of FFT's:"/ &
                   " number of FFTs for wave wavefunctions          ",F10.0," fft"/ &
                   " number of operations in four-orbital integrals ",F10.2," Gflops, ",F10.0," fft")
            IF (.NOT. MP2_bandstructure) THEN
               IF (IO%IU0>0) THEN
                  WRITE(IO%IU0,*)
                  WRITE(IO%IU0,11) REAL(EFOCK,Kind=q),REAL(energy_c,Kind=q),REAL(energy_x,Kind=q),REAL(E_MP2,Kind=q)
               ENDIF
               !WRITE(IO%IU6,10) NFFTW, NFLOAT4O/1E9, NFFT4O
               WRITE(IO%IU6,*)
               WRITE(IO%IU6,*) 'Moeller Plesset 2 correlation:'
               WRITE(IO%IU6,*) '================================'
               WRITE(IO%IU6,11) REAL(EFOCK,Kind=q),REAL(energy_c,Kind=q),REAL(energy_x,Kind=q),REAL(E_MP2,Kind=q)
               WRITE(IO%IU6,*)         
            ENDIF
11          FORMAT('     Hartree Fock energy: ',F20.8/&
                   '   Hartree contr. to MP2: ',F20.8/&
                   '  Exchange contr. to MP2: ',F20.8/&
                   '  MP2 correlation energy: ',F20.8/)


            IF (MP2_bandstructure) THEN               
               DO ISP=1,WDES%ISPIN
                  IF (WDES%ISPIN==2) WRITE(IO%IU6,*)'   spin component',ISP
                  WRITE(IO%IU6,*)
                  DO KI=1,WDES%NKPTS
                     WRITE(IO%IU6,*)
                     WRITE(IO%IU6,'("k-point",I4,":",3F10.4)') KI,WDES%VKPT(:,KI)
                     WRITE(IO%IU6,'(A)')'band No.  energies(HF) QP-shift   (hartree)   (exchange)    Z_h     Z_x  occupation'
                     WRITE(IO%IU6,'(I4,7F14.6)')HOMO(ISP),REAL(W%CELTOT(HOMO(ISP),KI,ISP),KIND=q),REAL(E_HOMO_h(KI,ISP)+E_HOMO_x(KI,ISP),KIND=q) &
                      ,REAL(E_HOMO_h(KI,ISP),KIND=q),REAL(E_HOMO_x(KI,ISP),KIND=q),REAL(Z_HOMO_h(KI,ISP),KIND=q),REAL(Z_HOMO_x(KI,ISP),KIND=q),W%FERTOT(HOMO(ISP),KI,ISP)
                     WRITE(IO%IU6,'(I4,7F14.6)')LUMO(ISP),REAL(W%CELTOT(LUMO(ISP),KI,ISP),KIND=q),REAL(E_LUMO_h(KI,ISP)+E_LUMO_x(KI,ISP),KIND=q) &
                     ,REAL(E_LUMO_h(KI,ISP),KIND=q),REAL(E_LUMO_x(KI,ISP),KIND=q),REAL(Z_LUMO_h(KI,ISP),KIND=q),REAL(Z_LUMO_x(KI,ISP),KIND=q),W%FERTOT(LUMO(ISP),KI,ISP)
                  ENDDO
                  WRITE(IO%IU6,*)
               ENDDO
               WRITE(IO%IU0,*)''
               WRITE(IO%IU0,*)'MP2 quasiparticle energies have been written to OUTCAR'
            ENDIF   
         ENDIF
         call BLACS_GRIDEXIT(contxt)
         call BLACS_GRIDEXIT(contxt_grid)
      RETURN
      END SUBROUTINE CALCULATE_MP2

      !***********************************************************************
      !This subroutine uses the matrices TWOE4ORBITAL(<ij|ab>) and TWOE4ORBITAL_X(<ij|ba>) 
      !in order to calculate the MP2 and Fock energy, as well as quasiparticle energy contributions:
      !The expression for the MP2 energy contribution added in this routine reads as follows:
      !\sum_a,b <ij|ab>(2<ij|ab>-<ij|ba>)*/(e_i+e_j-e_a-e_b),
      !or for the spin-polarized case
      !\sum_a,b' <ij'|ab'>(<ij'|ab'>)*/(e_i+e_j'-e_a-e_b')
      !+\sum_a,b <ij|ab>(<ij|ab>-<ij|ba>)*/(e_i+e_j-e_a-e_b)
      !+\sum_a',b' <i'j'|a'b'>(<i'j'|a'b'>-<i'j'|b'a'>)*/(e_i'+e_j'-e_a'-e_b')
      !(Here the ' denotes the first spincomponent and unprimed indices the second spnincomp.)
      !Note that all four k-point indices are kept fix in this subroutine.
      !The loops in this routine go only over the two virtual band indices a and b.
      !
      !Moreover this subroutine also evaluates MP2 quasiparticle energies for the highest
      !occupied state and lowest unoccupied state at every k-point.
      !The quasiparticle energies(E_HOMO, E_LUMO) are calculated using the expressions given in
      !J.chem.Phys. 115, 9698(2001)
      !The quasiparticle energy feature works only for ISYM=-1 calculations correctly.
      !Note that E_HOMO_h and E_LUMO_h correspond to the sum over hartree-like terms (<ij|ab><ij|ab>*),
      !whereas E_HOMO_x and E_LUMO_x correspond to the sum over exchange-like terms (<ij|ab><ij|ba>*).
      !Z_LUMO_h, Z_LUMO_x, Z_HOMO_h, Z_HOMO_x correspond to the normalization factors used in
      !the GW method.
      !Note: Due to the IF-statements in the loops, this ADD_MP2_wb routine is two
      !times slower than the subroutine called ADD_MP2. 
      !***********************************************************************

      SUBROUTINE ADD_MP2_wb(W,KI,KJ,KA,KB,NI,NJ,ISP1,ISP2,LATT_CUR,FSG)  
         USE constant
         USE full_kpoints
         USE mkpoints
         USE wave
         IMPLICIT NONE 
         TYPE (wavespin) W
         INTEGER :: KI,KJ,KA,KB,NI,NJ,I
         TYPE(latt) LATT_CUR
         REAL(q) :: FSG
         INTEGER :: NA,NB,TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS, KJ_IFO, KB_IFO
         INTEGER :: RNA,RNB,ISP1,ISP2,KI_IN_FULL_ORIG,KJ_IN_FULL_ORIG,kq1,kq2
         REAL(q) :: DENOM, OCC, VIRT, ediff_th, tmp2
         COMPLEX(q) :: test, HEAD1(3,3), HEAD2(3,3)
         COMPLEX(q) :: Inte_c, Inte_x,tmp         
         GDEF :: CDER_BETWEEN_STATE_IA(3),CDER_BETWEEN_STATE_JB(3)
         GDEF :: CDER_BETWEEN_STATE_IB(3),CDER_BETWEEN_STATE_JA(3)
         
         ediff_th=0.002_q
         !Energy_c=0.0_q
         !Energy_x=0.0_q
         TWOE4ORBITAL_ROWS = numroc(desc_TWOE4ORBITAL(3),desc_TWOE4ORBITAL(5),MYROW,0,NPROW)
         TWOE4ORBITAL_COLS = numroc(desc_TWOE4ORBITAL(4),desc_TWOE4ORBITAL(6),MYCOL,0,NPCOL)
         
         !KI_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI),KPOINTS_FULL_ORIG) 
         KJ_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KJ),KPOINTS_FULL_ORIG)
         
         !IF (KI_IN_FULL_ORIG/=KI) WRITE(*,*) 'error: ki/=ki'
         !IF (KJ_IN_FULL_ORIG/=KJ) WRITE(*,*) 'error: kj/=kj'
         kq1=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KB)-W%WDES%VKPT(:,KJ),KPOINTS_FULL)
         ! k_a = k_i + k_q - G
         KQ2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI)-W%WDES%VKPT(:,KA),KPOINTS_FULL)
         
         if (kq1/=kq2) WRITE(*,*)'error: q-point changed.'
         
         DO NB=1,TWOE4ORBITAL_COLS
            CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
            DO NA=1,TWOE4ORBITAL_ROWS
               CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)     
            
               IF ((KI==KB) .AND. (KJ==KA) .AND. (NI==RNB) .AND. (NJ==RNA) .AND. (ISP1==ISP2)) THEN
                     EFOCK=EFOCK+TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
                      test=TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
               ENDIF
               
               OCC =W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2)*KPOINTS_ORIG%WTKPT(KI)
               VIRT=(1._q-W%FERTOT(RNA,KA,ISP1))*(1._q-W%FERTOT(RNB,KB,ISP2))               
               
               DENOM=REAL(W%CELTOT(NI,KI,ISP1),KIND=q)+&
                 REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)-&
                 REAL(W%CELTOT(RNA,KA,ISP1),KIND=q)
   IF (DENOM<0._q) THEN
               IF (ISP2==ISP1) THEN
                  !IF ((W%WDES%ISPIN==2) .AND. (KA==KB) .AND. (RNA==RNB)) CYCLE
                  tmp=zero

                     HEAD1=zero
                     Inte_c=zero
                     CDER_BETWEEN_STATE_IA(:)=zero
                     CDER_BETWEEN_STATE_JB(:)=zero
                     IF (KI==KA) THEN
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_IA,LATT_CUR,KI, ISP1, RNA, NI)
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_JB,LATT_CUR,KJ_IN_FULL_ORIG, ISP1, RNB, NJ)
                        !DO i=1,3                           
                           !HEAD1(I,I)=CDER_BETWEEN_STATE_IA(I)*CDER_BETWEEN_STATE_JB(I)
                           !IF ((NJ/=RNB) .and. (ediff_th<ABS(REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)))) THEN                           
                           !   HEAD1(I,I)=zero
                           !ENDIF
                           !IF ((NI/=RNA) .and. (ediff_th<ABS(REAL(W%CELTOT(NI,KI,ISP2),KIND=q)-REAL(W%CELTOT(RNA,KA,ISP2),KIND=q)))) THEN
                              HEAD1(I,I)=zero
                           !ENDIF
                        !ENDDO
                        !important note: here we consider only terms G->0 and G'->0 (i.e. what is called the head in GW for NJ==RNB)
                        !therefore the other terms (G->0 & G'/=0, G'->0 & G/=0, as well as G->0 & G'->0 for NJ/=RNB) are not
                        !corrected, which results in a less good convergence of the quasiparticle energies with respect to the number of
                        !k-points.
                        
                        HEAD1(:,:)=-HEAD1(:,:)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)                        
                        
                        DO i=1,3
                           Inte_c=Inte_c+(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*(GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i)))*(1._q/3._q) 
                           !IF (ediff_th>ABS(REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q))) THEN
                           IF (NJ==RNB) THEN
                              Inte_c=Inte_c+CDER_BETWEEN_STATE_IA(I)*GCONJG(CDER_BETWEEN_STATE_IA(I))*(FSG)*(1._q/3._q)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                           ENDIF
                        ENDDO
                        
                     ELSE
                        Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))
                     ENDIF                     
                     
                     tmp=zero
                     HEAD2=zero
                     Inte_x=zero
                     CDER_BETWEEN_STATE_IB(:)=zero
                     CDER_BETWEEN_STATE_JA(:)=zero
                     IF ((KI==KB) .OR. (KI==KA)) THEN
                        IF ((KI==KB)) THEN 
                           CALL  CDER_BETWEEN_STATES_ROTATED( &
                         CDER_BETWEEN_STATE_IB,LATT_CUR,KI, ISP1, RNB, NI)
                           CALL  CDER_BETWEEN_STATES_ROTATED( &
                         CDER_BETWEEN_STATE_JA,LATT_CUR,KJ_IN_FULL_ORIG, ISP1, RNA, NJ)
                           !DO i=1,3
                              !HEAD2(I,I)=CDER_BETWEEN_STATE_IB(I)*CDER_BETWEEN_STATE_JA(I)
                              !IF (ediff_th<ABS(REAL(W%CELTOT(NI,KI,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q))) THEN              
                              !   HEAD2(I,I)=zero
                              !ENDIF
                              !IF ((NJ/=RNA) .and. (ediff_th<ABS(REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNA,KA,ISP2),KIND=q)))) THEN              
                              !   HEAD2(I,I)=zero
                              !ENDIF
                           !ENDDO
                           HEAD2(:,:)=-HEAD2(:,:)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                        ENDIF
                        tmp=zero
                        DO i=1,3
                           Inte_x=Inte_x+GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*((TWOE4ORBITAL_X(NA,NB)+HEAD2(i,i)))*(1._q/3._q)                           
                           IF ((NJ==RNA) .and. (NJ==RNB)) THEN
                              Inte_x=Inte_x+CDER_BETWEEN_STATE_IA(I)*GCONJG(CDER_BETWEEN_STATE_IB(I))*(FSG)*(1._q/3._q)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                           ENDIF
                        ENDDO
                     ELSE
                        Inte_x=Inte_x+GCONJG(TWOE4ORBITAL(NA,NB))*(TWOE4ORBITAL_X(NA,NB))
                     ENDIF

                  tmp2=FAC1
                  IF (W%WDES%ISPIN==2) FAC1=FAC1*2.0_q
                  OCC=W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)

                  !U(g)... where g is VBMAX orbital. electron-contribution to the quasiparticle energy
                  IF ((NJ==HOMO(ISP2)) .AND. (NI<=HOMO(ISP2)) .AND. (RNB>HOMO(ISP2)) .AND. (RNA>HOMO(ISP2))) THEN
                     E_HOMO_h(KJ,ISP2)=E_HOMO_h(KJ,ISP2)+(OCC*VIRT*FAC1*(FAC2*Inte_c)/DENOM)
                     Z_HOMO_h(KJ,ISP2)=Z_HOMO_h(KJ,ISP2)+(OCC*VIRT*FAC1*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))
                     E_HOMO_x(KJ,ISP2)=E_HOMO_x(KJ,ISP2)-(OCC*VIRT*FAC1*(Inte_x)/DENOM)
                     Z_HOMO_x(KJ,ISP2)=Z_HOMO_x(KJ,ISP2)-(OCC*VIRT*FAC1*(Inte_x)/(DENOM+DELTA_FINIT))
                  ENDIF  
                  !V(g)... where g is VBMAX orbital. hole-contribution to the quasiparticle energy
                  IF ((RNB==HOMO(ISP2)) .AND. (RNA>=LUMO(ISP2)) .and. (NI<=HOMO(ISP2)) .and. (NJ<=HOMO(ISP2))) THEN
                     E_HOMO_h(KB,ISP2)=E_HOMO_h(KB,ISP2)-(OCC*FAC1*(FAC2*Inte_c)/DENOM)
                     Z_HOMO_h(KB,ISP2)=Z_HOMO_h(KB,ISP2)-(OCC*FAC1*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                     E_HOMO_x(KB,ISP2)=E_HOMO_x(KB,ISP2)+(OCC*FAC1*(Inte_x)/DENOM)
                     Z_HOMO_x(KB,ISP2)=Z_HOMO_x(KB,ISP2)+(OCC*FAC1*(Inte_x)/(DENOM-DELTA_FINIT))
                  ENDIF
                  !U(g)... where g is CBMIN orbital. electron-contribution to the quasiparticle energy
                  IF ((NJ==LUMO(ISP2)) .AND. (NI<=HOMO(ISP2)) .AND. (RNA>=LUMO(ISP2)) .AND. (RNB>=LUMO(ISP2))) THEN                     
                     E_LUMO_h(KJ,ISP2)=E_LUMO_h(KJ,ISP2)+(VIRT*FAC1*(FAC2*Inte_c)/DENOM)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                     Z_LUMO_h(KJ,ISP2)=Z_LUMO_h(KJ,ISP2)+(VIRT*FAC1*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                     E_LUMO_x(KJ,ISP2)=E_LUMO_x(KJ,ISP2)-(VIRT*FAC1*(Inte_x)/DENOM)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                     Z_LUMO_x(KJ,ISP2)=Z_LUMO_x(KJ,ISP2)-(VIRT*FAC1*(Inte_x)/(DENOM+DELTA_FINIT))/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)                     
                  ENDIF
                  !V(g)... where g is CBMIN orbital. hole-contribution to the quasiparticle energy
                  IF ((RNB==LUMO(ISP2)) .AND. (RNA>HOMO(ISP2)) .and. (NJ<LUMO(ISP2)) .AND. (NI<LUMO(ISP2))) THEN                     
                     E_LUMO_h(KB,ISP2)=E_LUMO_h(KB,ISP2)-(VIRT*OCC*FAC1*(FAC2*Inte_c)/DENOM)
                     Z_LUMO_h(KB,ISP2)=Z_LUMO_h(KB,ISP2)-(VIRT*OCC*FAC1*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                     E_LUMO_x(KB,ISP2)=E_LUMO_x(KB,ISP2)+(VIRT*OCC*FAC1*(Inte_x)/DENOM)
                     Z_LUMO_x(KB,ISP2)=Z_LUMO_x(KB,ISP2)+(VIRT*OCC*FAC1*(Inte_x)/(DENOM-DELTA_FINIT))
                  ENDIF
                  FAC1=tmp2
                  
               ENDIF


               IF (ISP2/=ISP1) THEN
                  !write(*,*)'ni,nj,rna,rnb',ni,nj,rna,rnb,isp1,isp2,vbmax

                     HEAD1=zero
                     Inte_c=zero
                     CDER_BETWEEN_STATE_IA(:)=zero
                     CDER_BETWEEN_STATE_JB(:)=zero
                     IF (KI==KA) THEN                      
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_IA,LATT_CUR,KI,ISP1,RNA,NI)
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_JB,LATT_CUR,KJ_IN_FULL_ORIG,ISP2,RNB,NJ)
                        DO i=1,3
                           HEAD1(i,i)=CDER_BETWEEN_STATE_IA(i)*CDER_BETWEEN_STATE_JB(i)
                        ENDDO
                        IF ((NJ/=RNB) .and. (ediff_th<ABS(REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)))) THEN                           
                           HEAD1(:,:)=zero
                        ENDIF
                        IF ((NI/=RNA) .and. (ediff_th<ABS(REAL(W%CELTOT(NI,KI,ISP1),KIND=q)-REAL(W%CELTOT(RNA,KA,ISP1),KIND=q)))) THEN
                           HEAD1(:,:)=zero
                        ENDIF
                           
                        HEAD1=-HEAD1*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                        DO i=1,3
                           Inte_c=Inte_c+(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*(GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i)))*(1._q/3._q)
                           IF ((NI==RNA)) THEN
                              Inte_c=Inte_c+CDER_BETWEEN_STATE_JB(I)*GCONJG(CDER_BETWEEN_STATE_JB(I))*(FSG)*(1._q/3._q)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                           ENDIF
                           IF (NJ==RNB) THEN
                              Inte_c=Inte_c+CDER_BETWEEN_STATE_IA(I)*GCONJG(CDER_BETWEEN_STATE_IA(I))*(FSG)*(1._q/3._q)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                           ENDIF
                        ENDDO
                     ELSE
                        Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))                        
                     ENDIF
                     
                     IF ((NJ<=HOMO(ISP2)) .and. (NI<=HOMO(ISP1)) .and. (RNA>HOMO(ISP1)) .and. (RNB>HOMO(ISP2))) THEN
                        Energy_c=Energy_c+OCC*VIRT*(Inte_c)/DENOM
                        Energy_cc=Energy_cc+OCC*VIRT*(Inte_c)/DENOM
                     ENDIF
                  OCC=W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                  !U(g)... where g is VBMAX orbital. electron-contribution to the quasiparticle energy
                  !IF ((NJ==HOMO(ISP2)) .and. (NI<=HOMO(ISP1)) .and. (RNA>HOMO(ISP1)) .and. (RNB>HOMO(ISP2))) THEN
                  !   E_HOMO_h(KJ,ISP2)=E_HOMO_h(KJ,ISP2)+(OCC*VIRT*(FAC2*Inte_c)/DENOM)
                  !   Z_HOMO_h(KJ,ISP2)=Z_HOMO_h(KJ,ISP2)+(OCC*VIRT*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))
                  !ENDIF  
                  !V(g)... where g is VBMAX orbital. hole-contribution to the quasiparticle energy
                  !IF ((RNB==HOMO(ISP2)) .AND. (RNA>=LUMO(ISP1)) .and. (NI<=HOMO(ISP1)) .and. (NJ<=HOMO(ISP2))) THEN
                  !   E_HOMO_h(KB,ISP2)=E_HOMO_h(KB,ISP2)-(OCC*(FAC2*Inte_c)/DENOM)
                  !   Z_HOMO_h(KB,ISP2)=Z_HOMO_h(KB,ISP2)-(OCC*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                  !ENDIF
                  !U(g)... where g is CBMIN orbital. electron-contribution to the quasiparticle energy
                  !IF ((NJ==LUMO(ISP2)) .AND. (NI<=HOMO(ISP1)) .AND. (RNA>=LUMO(ISP1)) .AND. (RNB>=LUMO(ISP2))) THEN
                  !   E_LUMO_h(KJ,ISP2)=E_LUMO_h(KJ,ISP2)+(VIRT*(FAC2*Inte_c)/DENOM)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                  !   Z_LUMO_h(KJ,ISP2)=Z_LUMO_h(KJ,ISP2)+(VIRT*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)                  
                  !ENDIF
                  !V(g)... where g is CBMIN orbital. hole-contribution to the quasiparticle energy
                  !IF ((RNB==LUMO(ISP2)) .and. (NI<=HOMO(ISP1)) .and. (NJ<=HOMO(ISP2)) .and. (RNA>=LUMO(ISP1))) THEN
                  !   E_LUMO_h(KB,ISP2)=E_LUMO_h(KB,ISP2)-(VIRT*OCC*(FAC2*Inte_c)/DENOM)
                  !   Z_LUMO_h(KB,ISP2)=Z_LUMO_h(KB,ISP2)-(VIRT*OCC*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                  !ENDIF

                  !U(g)... where g is VBMAX orbital. electron-contribution to the quasiparticle energy
                  !IF ((NI==HOMO(ISP1)) .and. (NJ<=HOMO(ISP2)) .and. (RNA>HOMO(ISP1)) .and. (RNB>HOMO(ISP2))) THEN
                  !   E_HOMO_h(KI,ISP1)=E_HOMO_h(KI,ISP1)+(OCC*VIRT*(FAC2*Inte_c)/DENOM)
                  !   Z_HOMO_h(KI,ISP1)=Z_HOMO_h(KI,ISP1)+(OCC*VIRT*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))
                  !ENDIF  
                  !V(g)... where g is VBMAX orbital. hole-contribution to the quasiparticle energy
                  !IF ((RNA==HOMO(ISP1)) .AND. (RNB>=LUMO(ISP2)) .and. (NI<=HOMO(ISP1)) .and. (NJ<=HOMO(ISP2))) THEN
                  !   E_HOMO_h(KA,ISP1)=E_HOMO_h(KA,ISP1)-(OCC*(FAC2*Inte_c)/DENOM)
                  !   Z_HOMO_h(KA,ISP1)=Z_HOMO_h(KA,ISP1)-(OCC*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                  !ENDIF
                  !U(g)... where g is CBMIN orbital. electron-contribution to the quasiparticle energy
                  !IF ((NI==LUMO(ISP1)) .AND. (NJ<=HOMO(ISP2)) .AND. (RNA>=LUMO(ISP1)) .AND. (RNB>=LUMO(ISP2))) THEN                  
                  !   E_LUMO_h(KI,ISP1)=E_LUMO_h(KI,ISP1)+(VIRT*(FAC2*Inte_c)/DENOM)/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)
                  !   Z_LUMO_h(KI,ISP1)=Z_LUMO_h(KI,ISP1)+(VIRT*(FAC2*Inte_c)/(DENOM+DELTA_FINIT))/W%WDES%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI)                  
                  !ENDIF
                 !V(g)... where g is CBMIN orbital. hole-contribution to the quasiparticle energy
                  !IF (RNA==LUMO(ISP1) .and. (NI<=HOMO(ISP1)) .and. (NJ<=HOMO(ISP2)) .and. (RNB>=LUMO(ISP2))) THEN
                  !   E_LUMO_h(KA,ISP1)=E_LUMO_h(KA,ISP1)-(VIRT*OCC*(FAC2*Inte_c)/DENOM)
                  !   Z_LUMO_h(KA,ISP1)=Z_LUMO_h(KA,ISP1)-(VIRT*OCC*(FAC2*Inte_c)/(DENOM-DELTA_FINIT))
                  !ENDIF


                  !write(7,"(8I3,E16.5)")nj,ni,rnb,rna,kj,ki,kb,ka,Inte_c
                  !Energy_c=Energy_c+OCC*VIRT*(TWOE4ORBITAL(NA,NB)*CONJG(TWOE4ORBITAL(NA,NB)))/DENOM 
               ENDIF !isp1/=isp2

             ENDIF !denom<0 
               
           ENDDO
         ENDDO
         
      END SUBROUTINE ADD_MP2_wb

      !***********************************************************************
      !This subroutine uses the matrices TWOE4ORBITAL(<ij|ab>) and TWOE4ORBITAL_X(<ij|ba>) 
      !in order to calculate the MP2 and Fock energy contributions:
      !The expression for the MP2 energy contribution added in this routine reads as follows:
      !\sum_a,b <ij|ab>(2<ij|ab>-<ij|ba>)*/(e_i+e_j-e_a-e_b),
      !or for the spin-polarized case
      !\sum_a,b' <ij'|ab'>(<ij'|ab'>)*/(e_i+e_j'-e_a-e_b')
      !+\sum_a,b <ij|ab>(<ij|ab>-<ij|ba>)*/(e_i+e_j-e_a-e_b)
      !+\sum_a',b' <i'j'|a'b'>(<i'j'|a'b'>-<i'j'|b'a'>)*/(e_i'+e_j'-e_a'-e_b')
      !(Here the ' denotes the first spincomponent and unprimed indices the second spnincomp.)
      !Note that all four k-point indices are kept fix in this subroutine.
      !The loops in this routine go only over the two virtual band indices a and b.
      !***********************************************************************

      SUBROUTINE ADD_MP2(W,KI,KJ,KA,KB,NI,NJ,ISP1,ISP2,LATT_CUR,WISPIN)
         USE constant
         USE full_kpoints
         USE mkpoints
         USE wave
         IMPLICIT NONE 
         TYPE (wavespin) W
         INTEGER :: KI,KJ,KA,KB,NI,NJ,I,WISPIN
         TYPE(latt) LATT_CUR
         INTEGER :: NA,NB,TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS
         INTEGER :: RNA,RNB,ISP1,ISP2,KI_IN_FULL_ORIG,KJ_IN_FULL_ORIG,kq1,kq2
         REAL(q) :: DENOM, OCC, VIRT, ediff_th
         COMPLEX(q) :: test, HEAD1(3,3), HEAD2(3,3)
         COMPLEX(q) :: Inte_c, Inte_x,tmp
         GDEF :: CDER_BETWEEN_STATE_IA(3),CDER_BETWEEN_STATE_JB(3)
         GDEF :: CDER_BETWEEN_STATE_IB(3),CDER_BETWEEN_STATE_JA(3)
         
         ediff_th=0.1_q
         !Energy_c=0.0_q
         !Energy_x=0.0_q
         TWOE4ORBITAL_ROWS = numroc(desc_TWOE4ORBITAL(3),desc_TWOE4ORBITAL(5),MYROW,0,NPROW)
         TWOE4ORBITAL_COLS = numroc(desc_TWOE4ORBITAL(4),desc_TWOE4ORBITAL(6),MYCOL,0,NPCOL)
         
         !KI_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI),KPOINTS_FULL_ORIG) 
         KJ_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KJ),KPOINTS_FULL_ORIG)
         
         kq1=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KB)-W%WDES%VKPT(:,KJ),KPOINTS_FULL)
         ! k_a = k_i + k_q - G
         KQ2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI)-W%WDES%VKPT(:,KA),KPOINTS_FULL)
         
         if (kq1/=kq2) WRITE(*,*)'error: q-point changed.'
         !write(*,*)'kq1',W%WDES%VKPT(:,kq1)
         KQ2=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI)-W%WDES%VKPT(:,KB),KPOINTS_FULL)
         IF (( (ABS(W%WDES%VKPT(1,kq1))<0.01_q) .AND. (ABS(W%WDES%VKPT(2,kq1))<0.01_q) .AND. (ABS(W%WDES%VKPT(3,kq1))<0.01_q))  .OR. &
          ((ABS(W%WDES%VKPT(1,KQ2))<0.01_q) .AND. (ABS(W%WDES%VKPT(2,KQ2))<0.01_q) .AND. (ABS(W%WDES%VKPT(3,KQ2))<0.01_q) )) THEN
         
            DO NB=1,TWOE4ORBITAL_COLS
               CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
               DO NA=1,TWOE4ORBITAL_ROWS
               CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)     
               IF ((KI==KB) .AND. (KJ==KA) .AND. (NI==RNB) .AND. (NJ==RNA) .AND. (ISP1==ISP2)) THEN
                     EFOCK=EFOCK+TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
                      test=TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
               ENDIF
               
               OCC =W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2)*KPOINTS_ORIG%WTKPT(KI)
               VIRT=(1._q-W%FERTOT(RNA,KA,ISP1))*(1._q-W%FERTOT(RNB,KB,ISP2))
               
               IF (FILLED_MP2_ORBITAL(W%FERTOT(RNA,KA,ISP1))) VIRT=zero
               IF (FILLED_MP2_ORBITAL(W%FERTOT(RNB,KB,ISP2))) VIRT=zero
               
               DENOM=REAL(W%CELTOT(NI,KI,ISP1),KIND=q)+&
                 REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)-&
                 REAL(W%CELTOT(RNA,KA,ISP1),KIND=q)

#ifdef gammareal
             IF ((NJ<NI) .and. (WISPIN==1)) DENOM=DENOM*0.5_q
#else

#endif

             IF (DENOM<0._q) THEN
               IF (ISP2==ISP1) THEN
                  tmp=zero
                     HEAD1=zero
                     Inte_c=zero
                     IF ((KI==KA)) THEN
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_IA,LATT_CUR,KI, ISP1, RNA, NI)
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_JB,LATT_CUR,KJ_IN_FULL_ORIG, ISP1, RNB, NJ)
                        DO i=1,3
                           HEAD1(I,I)=CDER_BETWEEN_STATE_IA(I)*CDER_BETWEEN_STATE_JB(I)
                        ENDDO
                        HEAD1(:,:)=-HEAD1(:,:)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                        DO i=1,3
                           Inte_c=Inte_c+(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*(GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i)))*(1._q/3._q)
                        ENDDO
                     ELSE
                        Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))
                     ENDIF
                     Energy_c=Energy_c+OCC*VIRT*FAC1*(FAC2*Inte_c)/DENOM

                     HEAD2=zero
                     Inte_x=zero
                     IF ((KI==KB) .OR. (KI==KA)) THEN
                        IF ((KI==KB)) THEN                  
                           CALL  CDER_BETWEEN_STATES_ROTATED( &
                         CDER_BETWEEN_STATE_IB,LATT_CUR,KI, ISP1, RNB, NI)
                           CALL  CDER_BETWEEN_STATES_ROTATED( &
                         CDER_BETWEEN_STATE_JA,LATT_CUR,KJ_IN_FULL_ORIG, ISP1, RNA, NJ)
                           DO i=1,3
                              HEAD2(I,I)=CDER_BETWEEN_STATE_IB(I)*CDER_BETWEEN_STATE_JA(I)
                           ENDDO
                           HEAD2(:,:)=-HEAD2(:,:)*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                        ENDIF
                        DO i=1,3
                           Inte_x=Inte_x+GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*((TWOE4ORBITAL_X(NA,NB)+HEAD2(i,i)))*(1._q/3._q)
                           tmp=tmp+GCONJG(HEAD1(i,i))*HEAD2(i,i)*(1._q/3._q)
                        ENDDO                   
                     ELSE
                        Inte_x=Inte_x+GCONJG(TWOE4ORBITAL(NA,NB))*(TWOE4ORBITAL_X(NA,NB))
                     ENDIF
                     Energy_x=Energy_x-OCC*VIRT*FAC1*(Inte_x)/DENOM
                  
               ENDIF


               IF (ISP2/=ISP1) THEN
                     HEAD1=zero
                     Inte_c=zero
                     tmp=zero
                     IF ((KI==KA)) THEN
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_IA,LATT_CUR,KI,ISP1,RNA,NI)
                        CALL  CDER_BETWEEN_STATES_ROTATED( &
                      CDER_BETWEEN_STATE_JB,LATT_CUR,KJ_IN_FULL_ORIG,ISP2,RNB,NJ)
                        DO i=1,3
                           HEAD1(i,i)=CDER_BETWEEN_STATE_IA(i)*CDER_BETWEEN_STATE_JB(i)
                        ENDDO
                        HEAD1=-HEAD1*EDEPS/LATT_CUR%OMEGA*W%WDES%WTKPT(KI)
                        DO i=1,3
                           Inte_c=Inte_c+(TWOE4ORBITAL(NA,NB)+HEAD1(i,i))*(GCONJG(TWOE4ORBITAL(NA,NB)+HEAD1(i,i)))*(1._q/3._q)
                           tmp=tmp+(HEAD1(i,i))*(1._q/3._q)
                        ENDDO
                     ELSE
                        Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))
                        tmp=HEAD1(i,i)
                     ENDIF
                     Energy_c=Energy_c+OCC*VIRT*(Inte_c)/DENOM
                     Energy_cc=Energy_cc+OCC*VIRT*(Inte_c)/DENOM
               ENDIF !isp1/=isp2

             ENDIF !denom<0 
               
           ENDDO
         ENDDO
         
         ELSE
         
         DO NB=1,TWOE4ORBITAL_COLS
            CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
            DO NA=1,TWOE4ORBITAL_ROWS
               CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)     
            
               IF ((KI==KB) .AND. (KJ==KA) .AND. (NI==RNB) .AND. (NJ==RNA) .AND. (ISP1==ISP2)) THEN
                     EFOCK=EFOCK+TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
                      test=TWOE4ORBITAL(NA,NB)*KPOINTS_ORIG%WTKPT(KI) &
                      *W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) &
                      *W%FERTOT(NJ,KJ,ISP1)*W%FERTOT(NI,KI,ISP2)*FAC1
               ENDIF
               
               OCC =W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2)*KPOINTS_ORIG%WTKPT(KI)
               VIRT=(1._q-W%FERTOT(RNA,KA,ISP1))*(1._q-W%FERTOT(RNB,KB,ISP2))
               
               IF (FILLED_MP2_ORBITAL(W%FERTOT(RNA,KA,ISP1))) VIRT=zero
               IF (FILLED_MP2_ORBITAL(W%FERTOT(RNB,KB,ISP2))) VIRT=zero
               
               DENOM=REAL(W%CELTOT(NI,KI,ISP1),KIND=q)+&
                 REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)-&
                 REAL(W%CELTOT(RNA,KA,ISP1),KIND=q)
             IF (DENOM<0._q) THEN
               IF (ISP2==ISP1) THEN               
                     Inte_c=zero
                     Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))             
                     Energy_c=Energy_c+OCC*VIRT*FAC1*(FAC2*Inte_c)/DENOM
                     Inte_x=zero
                     Inte_x=Inte_x+GCONJG(TWOE4ORBITAL(NA,NB))*(TWOE4ORBITAL_X(NA,NB))
                     Energy_x=Energy_x-OCC*VIRT*FAC1*(Inte_x)/DENOM                  
               ENDIF
               
               IF (ISP2/=ISP1) THEN
                     Inte_c=zero
                     Inte_c=TWOE4ORBITAL(NA,NB)*GCONJG(TWOE4ORBITAL(NA,NB))
                     Energy_c=Energy_c+OCC*VIRT*(Inte_c)/DENOM
                     Energy_cc=Energy_cc+OCC*VIRT*(Inte_c)/DENOM
               ENDIF !isp1/=isp2

             ENDIF !denom<0 
               
           ENDDO
         ENDDO
         
         ENDIF
         

      END SUBROUTINE ADD_MP2

      !***********************************************************************
      !This routine calculates the fourier-transformed overlap integrals <i|-G|a>
      !and <j|G|b>*4*pi*e^2/(G+q)^2 and also calls the redistribution routine.
      !This is done for the plane-wave part, as well as for the one-center terms.
      !Note, that in the gamma-only version it is enough to save <i|-G|a>*SQRT(4*pi*e^2/(G+q)^2)
      !for the plane-wave part, because (<i|-G|a>)*=<i|G|a>.
      !However, this approximation cannot be made for the one-center terms in the gamma-only
      !version, because of practical reasons.
      !The <i|G|a> quantities are most important for the construction of the two electron
      !four orbital integrals <ij|ab>.  <ij|ab>=4pi e^2 \sum_G <i|-G|a> <j|G|b> /(G+q)^2
      !***********************************************************************

      SUBROUTINE CALC_2ORBITAL_FTOD(WDES,WGW,W,P,T_INFO,LATT_CUR,LMAXMP2,ENCUTGW,ENCUTGWSOFT,IO,FSG)
         USE prec
         USE poscar
         USE pseudo
         USE wave_high
         USE full_kpoints
         USE mkpoints
         USE lattice
         USE constant
         USE base
         IMPLICIT NONE
         TYPE(wavedes) WDES
         TYPE(wavedes) WGW
         TYPE(wavespin) W
         TYPE(type_info) T_INFO
         TYPE(potcar) P(T_INFO%NTYP)
         TYPE(latt) LATT_CUR
         INTEGER LMAXMP2
         TYPE (in_struct) IO
         ! local variables
         TYPE(wavespin) WHF
         TYPE(wavedes1) WGWQ
         TYPE(wavedes1) WDESKI,WDESKA,WDESKB
         TYPE(wavefun1), ALLOCATABLE :: WI(:), WA(:), WB(:)
         INTEGER KQ,KI,KA,KB,KI_IN_FULL_ORIG,KQ_
         INTEGER NSTRIP, NSTRIPA, ISP, NFFT
         INTEGER NBI,NBA,NBAA, i
         INTEGER NP ! number of plane waves for the overlap density i*(r)a(r) GCHGIA(r)
         REAL(q) :: FSG ! singularity correction
         REAL(q) :: ENCUTGW,ENCUTGWSOFT 
         REAL(q) :: POTFAK(GRIDHF%MPLWV) 
         COMPLEX(q) CPHASE(GRIDHF%MPLWV)
         COMPLEX(q) CPHASE2(GRIDHF%MPLWV)
         LOGICAL LPHASE  
         LOGICAL LPHASE2
         COMPLEX(q) :: GWORK(MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV)) !work array for calculating GCHGIA
         COMPLEX(q), ALLOCATABLE :: GCHGIA(:,:,:)  ! charge
         GDEF      , ALLOCATABLE :: CRHOIA(:,:)    ! one-center charge
         GDEF      , ALLOCATABLE :: CRHOIB(:,:)    ! one-center charge
         GDEF      , ALLOCATABLE :: CRHOLM(:)      ! augmentation occupancy matrix
         COMPLEX(q), ALLOCATABLE :: tmp_FTOD_PW(:,:,:,:)
         GDEF      , ALLOCATABLE :: tmp_FTOD_OC(:,:,:,:)
         GDEF :: CDER_BETWEEN_STATE(3)
         Real(q) :: mem_req
         
         CALL CHECK_FULL_KPOINTS
         
         IF (LMAXMP2>=0) THEN
           CALL SET_UP_ONE_CENTER_H(WDES,P,T_INFO,LMAXMP2,H)
         ENDIF
         
         WHF=W
         WHF%WDES => WDES_FOCK
         NSTRIP=30
         CALL SETWDES(WHF%WDES,WDESKI,0)
         CALL SETWDES(WHF%WDES,WDESKA,0)
         CALL SETWDES(WHF%WDES,WDESKB,0)

! do not use standard monopol correction
         IF (MCALPHA/=0) THEN
            FSG=0
         ENDIF       

         VBMAX=LAST_FILLED_XI_NOMOD(W,1,1)
         HOMO(1)=VBMAX
         LUMO(1)=FIRST_EMPTY_XI_NOMOD(W,1,1)
         DO ISP=2,WDES%ISPIN
            VBMAX=MAX(LAST_FILLED_XI_NOMOD(W,1,ISP),VBMAX)
            HOMO(ISP)=LAST_FILLED_XI_NOMOD(W,1,ISP)
            LUMO(ISP)=FIRST_EMPTY_XI_NOMOD(W,1,ISP)
         ENDDO
         IF (MP2_bandstructure) VBMAX=VBMAX+1         
         
         ALLOCATE(WI(VBMAX),WA(NSTRIP),WB(NSTRIP))
         DO NBI=1,VBMAX
            CALL NEWWAV(WI(NBI),WDESKI,.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL NEWWAV(WA(NBA),WDESKA,.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL NEWWAV(WB(NBA),WDESKB,.TRUE.)
         ENDDO
         
         NGVECTOR=MAXVAL(WGW%NGVECTOR(:))
         NHVECTOR=0
         IF (ASSOCIATED(H)) THEN
            NHVECTOR=H%TOTAL_ENTRIES
         ENDIF
         ! Make sure that NGVECTOR is suited for the BLACS process grid
         IF (MOD(NGVECTOR,NPROW_GRID)/=0) THEN
            NGVECTOR=NGVECTOR+(NPROW_GRID-MOD(NGVECTOR,NPROW_GRID))
         ENDIF
         ! Make sure that NHVECTOR is suited for the BLACS process grid
         IF (MOD(NHVECTOR,NPROW_GRID)/=0) THEN
            NHVECTOR=NHVECTOR+(NPROW_GRID-MOD(NHVECTOR,NPROW_GRID))
         ENDIF
         
         IF (ALLOCATED(FTOD_PW)) THEN
            DEALLOCATE(FTOD_PW)
         ENDIF
         
         !Initialize the 2D process grid
         CALL INIT_BLACS_GRID(WDES)
         
         CALL SETUP_FTOD(WDES)
         
         !write out an estimate for the required memory
         mem_req=2.0_q*(PROCS*PROCS*WDES%NBANDS*WDES%NBANDS)*16.0_q/1024.0_q/1024.0_q/1024.0_q !TWOE4ORBITAL arrays
         mem_req=mem_req+REAL(NGVECTOR,kind=q)*PROCS*WDES%NBANDS*VBMAX*WDES%NKPTS*WDES%NKPTS*WDES%ISPIN*2.0_q*16.0_q/1024.0_q/1024.0_q/1024.0_q 
         IF (LMAXMP2>=0) mem_req=mem_req+REAL(NHVECTOR,kind=q)*PROCS*WDES%NBANDS*VBMAX*WDES%NKPTS*WDES%NKPTS*WDES%ISPIN*2.0_q*16.0_q/1024.0_q/1024.0_q/1024.0_q
         IF (IO%IU0>0) THEN
            WRITE(*,"(A,E10.3,A)") 'For MP2 calculations approximately',mem_req,'GB RAM will be required.' 
         ENDIF
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*) 'Allocating memory...'
         ENDIF
         !Setup the descriptors for the distributed TWOE4ORBITAL matrix   
         !and allocate the TWOE4ORBITAL and TWOE4ORBITAL_X matrices
         CALL SETUP_TWOE4ORBITAL(WDES)
         IF (ALLOCATED(TWOE4ORBITAL)) THEN
            DEALLOCATE(TWOE4ORBITAL)
         ENDIF
         IF (ALLOCATED(TWOE4ORBITAL_X)) THEN
            DEALLOCATE(TWOE4ORBITAL_X)
         ENDIF
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*) 'succeeded'
         ENDIF
         
         ALLOCATE(tmp_FTOD_PW(NGVECTOR,WDES%NBANDS,VBMAX,ncc))
         IF (ASSOCIATED(H)) THEN
            ALLOCATE(tmp_FTOD_OC(NHVECTOR,WDES%NBANDS,VBMAX,2))
         ENDIF 
         
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*)
            WRITE(IO%IU0,*)'Calculating fourier transformed overlap densities:'
         ENDIF
         
         
         call BLACS_GRIDINFO(CONTXT, NPROW, NPCOL, MYROW, MYCOL)
         
         IF (ASSOCIATED(H)) THEN
            ALLOCATE(CRHOIA(NHVECTOR, NSTRIP),CRHOIB(NHVECTOR, NSTRIP))
            FTOD_OC=zero           
         ENDIF
         
         spin: DO ISP=1,WDES%ISPIN
         kqloop: DO KQ=1,WDES%NKPTS
            IF (IO%IU0>=0) WRITE(IO%IU0,*)
            IF (IO%IU0>=0) THEN
               IF (WDES%ISPIN==1) THEN
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') KQ,WDES%VKPT(:,KQ)
               ELSE
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ",A1,A1,", ")') KQ,WDES%VKPT(:,KQ),ISP
               ENDIF
            ENDIF
            CALL SETWDES(WGW,WGWQ,KQ)
         
            NP=WGWQ%NGVECTOR            
            IF (NP>NGVECTOR) THEN
               WRITE(*,*)'Internal error in "Calc_2orbital_FTOD": NP larger than NGVECTOR'
               EXIT
            ENDIF
            ALLOCATE(GCHGIA(NP,NSTRIP,2),CRHOLM(AUG_DES%NPRO*WDES%NRSPINORS))
         
            kiloop: DO KI=1,WDES%NKPTS
               tmp_FTOD_PW=(0._q,0._q)
               IF (ASSOCIATED(H)) tmp_FTOD_OC=zero
               
               CALL GWPROGRESS(IO%IU0, KI,WDES%NKPTS,KQ,WDES%NKPTS)
               KI_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI),KPOINTS_FULL_ORIG)
               ! collect all valence bands at k_i
               CALL SETWDES(WHF%WDES,WDESKI,KI)
               CALL W1_GATHER_GLB(WHF,1,VBMAX,ISP,WI)
               
               ! k_b = k_i - k_q - G
               KB=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KQ)+WDES%VKPT(:,KI),KPOINTS_FULL)
               ! k_a = k_i + k_q - G
               KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL)
               
               CALL SETWDES(WHF%WDES,WDESKA,KA)
               
               ! CPHASE(r) = e^iGr, where G = k_i - k_q - k_b
               CALL SETPHASE(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ)-WDES%VKPT(:,KA),GRIDHF,CPHASE,LPHASE)
               
               CALL SET_GFAC_WITHOUT_WEIGHT(GRIDHF,LATT_CUR,KI,KA,FSG,POTFAK)
               ! 1/(G+q)**2
               
               IF (ENCUTGW /= ENCUTGWSOFT .AND. ENCUTGW > 0 .AND. ENCUTGWSOFT > 0) THEN
                  CALL SET_GFAC_WAVEFUN(WGWQ,LATT_CUR,FSG,POTFAK,ENCUTGW,ENCUTGWSOFT)
               ELSE
                  CALL SET_GFAC_WAVEFUN(WGWQ,LATT_CUR,FSG,POTFAK)
               ENDIF
               IF (MCALPHA/=0) THEN
!                 setup routine for multipole correction
                  CALL FOCK_MULTIPOLE_CORR_SETUP(LATT_CUR, GRIDHF)
                  CALL FOCK_MULTIPOLE_SETUP_WGW(WGW, LATT_CUR, GRIDHF)
               ENDIF

#ifdef gammareal
                  POTFAK=SQRT(POTFAK)
#endif
               
               ! loop over all bands
               DO NBA=1,WDES%NBANDS,NSTRIP
                  NSTRIPA=MIN(WDES%NBANDS+1-NBA,NSTRIP)
                  ! FFT{psi_a} to real space
                  DO NBAA=1,NSTRIPA !copy and fourier transform NSTRIP wave functions
                     CALL W1_COPY( ELEMENT(WHF,WDESKA,NBA+NBAA-1,ISP),WA(NBAA))
                     CALL FFTWAV_W1(WA(NBAA))
                  ENDDO
                  ! loop over valence bands only
                  DO NBI=1,VBMAX
                     GCHGIA=0 
                     
                     IF (ASSOCIATED(H)) THEN
                        CRHOIA=zero
                        CRHOIB=zero
                        CRHOLM=zero
                     ENDIF
                     !loop over all bands in NSTRIP 
                     DO NBAA=1,NSTRIPA
                     
                     ! calculate rho(r)=psi_i(r)* psi_a(r) for,
                     ! one center terms and, on the plane wave grid.
                        IF (ASSOCIATED(H)) THEN
                           CALL FOCK_CHARGE_ONE_CENTER_NOINT( WI(NBI),WA(NBAA),&
                             GWORK(1),H,CRHOIA(1,NBAA), CRHOLM,  SIZE(CRHOLM))
                        ELSE
                           CALL FOCK_CHARGE_NOINT( WI(NBI),WA(NBAA), GWORK(1), &
                             CRHOLM, SIZE(CRHOLM))
                        ENDIF
                        ! Set phase e^iGr, where G = k_i - k_q - k_b
                        IF (LPHASE) THEN
                           CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), &
                             GWORK(1))
                        ENDIF
                   
                     ! FFT{rho} to reciprocal space
                        CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                          GWORK(1),GCHGIA(1,NBAA,1),WGWQ%GRID,.FALSE.)
                        NFFT=NFFT+1
                     ! multiply with potential factor
                        IF (MCALPHA/=0) THEN
                           !  include multipole correction
                           CALL APPLY_GFAC_MULTIPOLE_WAVEFUN(WGWQ,GCHGIA(1,NBAA,1), &
                            POTFAK(1))
                        ELSE
                           CALL APPLY_GFAC_WAVEFUN(WGWQ,GCHGIA(1,NBAA,1), &
                            POTFAK(1))
                        ENDIF
                         
                     ENDDO !NBAA (loop over all bands in NSTRIP)
                     
                     IF (ASSOCIATED(H)) THEN
                        CALL APPLY_PHASE_ONE_CENTER(WHF%WDES, H, CRHOIA(:,:), &
                          WHF%WDES%VKPT(:,KI)-WHF%WDES%VKPT(:,KA))
                     ENDIF
                     
                     IF (ASSOCIATED(H)) THEN
                        ! use CRHOIA as temporary work array
                        CALL APPLY_ONE_CENTER_H( WHF%WDES, H, CRHOIA(:,:), CRHOIB(:,:), NSTRIPA)
                     ENDIF
                     
                     DO NBAA=1,NSTRIPA
                        !copy FTOD functions to FTOD_PW and FTOD_OC
#ifdef gammareal
                           tmp_FTOD_PW(1:NP,NBAA+NBA-1,NBI,1)=(GCHGIA(1:NP,NBAA,1))*SQRT(1.0_q/GRIDHF%NPLWV)
#else
                           tmp_FTOD_PW(1:NP,NBAA+NBA-1,NBI,1)=(GCHGIA(1:NP,NBAA,1))
#endif
                        IF (ASSOCIATED(H)) THEN
                           tmp_FTOD_OC(1:H%TOTAL_ENTRIES,NBAA+NBA-1,NBI,1)=(CRHOIA(1:H%TOTAL_ENTRIES,NBAA))
                        ENDIF

                     ENDDO
                     
!!!!!!!!!!!!!!!!! GAMMA-only version !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef gammareal
                                     
                     ! calculate rho(r)=psi_i(r)* psi_a(r) for,
                     ! (1._q,0._q) center terms and, on the plane wave grid.
                IF (ASSOCIATED(H)) THEN
                     CRHOIB=0
                     CRHOLM=0
                     !loop over all bands in NSTRIP 
                     DO NBAA=1,NSTRIPA
                        CALL FOCK_CHARGE_ONE_CENTER_NOINT( WA(NBAA),WI(NBI),&
                          GWORK(1),H,CRHOIA(1,NBAA), CRHOLM,  SIZE(CRHOLM))                
                     ! FFT{rho} to reciprocal space
                        CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                          GWORK(1),GCHGIA(1,NBAA,1),WGWQ%GRID,.FALSE.)
                        NFFT=NFFT+1
                     ! multiply with potential factor
                     ENDDO !NBAA (loop over all bands in NSTRIP)
                     DO NBAA=1,NSTRIPA
                        !copy FTOD functions to FTOD_PW and FTOD_OC
                        IF (ASSOCIATED(H)) THEN
                           tmp_FTOD_OC(1:H%TOTAL_ENTRIES,NBAA+NBA-1,NBI,2)=(CRHOIA(1:H%TOTAL_ENTRIES,NBAA))    !000000000000000000000
                        ENDIF
                     ENDDO
                 ENDIF
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!GAMMA only version !!!!!!!!!!!!!!!!!!!!!!!!

                  ENDDO !NBI (loop over valence bands only)
               ENDDO !NBA (loop over all bands)
               
#ifdef gammareal
               
#else
               
               CALL SETWDES(WHF%WDES,WDESKB,KB)        
               
               ! k_i - k_a = k_q + G
               CALL PHASER_HF(GRIDHF,LATT_CUR,FAST_AUG_FOCK,WDES%VKPT(:,KB)-WDES%VKPT(:,KI))
               
               ! CPHASE(r) = e^iGr, where G = k_i - k_q - k_a
               CALL SETPHASE(WDES%VKPT(:,KB)-WDES%VKPT(:,KQ)-WDES%VKPT(:,KI),GRIDHF,CPHASE,LPHASE)
               
               ! loop over all bands
               DO NBA=1,WDES%NBANDS,NSTRIP
                  NSTRIPA=MIN(WDES%NBANDS+1-NBA,NSTRIP)
                  ! FFT{psi_a} to real space
                  DO NBAA=1,NSTRIPA !copy and fourier transform NSTRIP wave functions
                     CALL W1_COPY( ELEMENT(WHF,WDESKB,NBA+NBAA-1,ISP),WB(NBAA))
                     CALL FFTWAV_W1(WB(NBAA))
                  ENDDO
                  ! loop over valence bands only
                  DO NBI=1,VBMAX
                     GCHGIA=0 
                     
                     IF (ASSOCIATED(H)) THEN
                        CRHOIB=0
                        CRHOLM=0
                     ENDIF
                     !loop over all bands in NSTRIP
                     DO NBAA=1,NSTRIPA
                     ! GCHGIA number 2 for X-changed waves
                     ! calculate rho(r)=psi_i(r)* psi_a(r) for,
                     ! one center terms and, on the plane wave grid.
                        IF (ASSOCIATED(H)) THEN
                           CALL FOCK_CHARGE_ONE_CENTER_NOINT( WB(NBAA),WI(NBI),&
                             GWORK(1),H,CRHOIB(1,NBAA), CRHOLM,  SIZE(CRHOLM))
                        ELSE
                           CALL FOCK_CHARGE_NOINT( WB(NBAA),WI(NBI), GWORK(1), &
                            CRHOLM, SIZE(CRHOLM))
                        ENDIF
                        ! Set phase e^iGr, where G = k_i - k_q - k_a
                        IF (LPHASE) THEN
                        CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), &
                          GWORK(1) )
                           !IF (KQ==1) WRITE(*,*)'error: no apply_phase need for kq=1'
                        ENDIF  

                     ! FFT{rho} to reciprocal space
                        CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                          GWORK(1),GCHGIA(1,NBAA,2),WGWQ%GRID,.FALSE.)
                        NFFT=NFFT+1
                     ! multiply with potential factor                        
                     ENDDO !NBAA (loop over all bands in NSTRIP)
                     
                     IF (ASSOCIATED(H)) THEN
                        CALL APPLY_PHASE_ONE_CENTER(WHF%WDES, H, CRHOIB(:,:), &
                          WHF%WDES%VKPT(:,KB)-WHF%WDES%VKPT(:,KI))
                     ENDIF
                     
                     DO NBAA=1,NSTRIPA
                        !copy FTOD functions to FTOD_PW and FTOD_OC
                        
                        tmp_FTOD_PW(1:NP,NBAA+NBA-1,NBI,2)=(GCHGIA(1:NP,NBAA,2)*(1.0_q/GRIDHF%NPLWV))
                        IF (ASSOCIATED(H)) THEN
                           tmp_FTOD_OC(1:H%TOTAL_ENTRIES,NBAA+NBA-1,NBI,2)=CRHOIB(1:H%TOTAL_ENTRIES,NBAA)
                        ENDIF   

                        
                     ENDDO
                  ENDDO !NBI (loop over valence bands only)
               ENDDO !NBA (loop over all bands) 
               
#endif
               CALL REDISTRIBUTE_FTOD_GRID(WDES,KI,KQ,ISP,tmp_FTOD_PW,tmp_FTOD_OC)

            ENDDO kiloop
         
            DEALLOCATE(GCHGIA,CRHOLM)
         
         ENDDO kqloop
         ENDDO spin

         IF (ALLOCATED(CRHOIA)) DEALLOCATE(CRHOIA)
         IF (ALLOCATED(CRHOIB)) DEALLOCATE(CRHOIB)
         IF (ALLOCATED(tmp_FTOD_OC)) DEALLOCATE(tmp_FTOD_OC)
         IF (ALLOCATED(tmp_FTOD_PW)) DEALLOCATE(tmp_FTOD_PW)
         
         DO NBI=1,VBMAX
            CALL DELWAV(WI(NBI),.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL DELWAV(WA(NBA),.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL DELWAV(WB(NBA),.TRUE.)
         ENDDO
         DEALLOCATE(WI,WA,WB)
         
         
         RETURN
      END SUBROUTINE CALC_2ORBITAL_FTOD
 
      !***********************************************************************
      !This routine initializes a process grid which is used for the 
      !pre-calculation of the fourier-transformed overlap integrals <i|G|a>
      !This process grid has NPROCS(number of processors used) columns and 1 row.
      !The assigned context variable is called CONTXT.
      !*********************************************************************** 

      SUBROUTINE INIT_BLACS_COLS()
         implicit none           
         INTEGER :: a_PRCS, i
         REAL :: NPCOL_TMP
         
         !first we create a column-only process grid in order to
         !calculate the <i|-G|a> and <j|G|b> quantities
         call BLACS_PINFO(ME,PROCS)
         nprow=1
         npcol=PROCS     
         call BLACS_PINFO(ME,PROCS)         
         call BLACS_GET     (0, 0, CONTXT)
         call BLACS_GRIDINIT(CONTXT, 'R', NPROW, NPCOL)
         call BLACS_GRIDINFO(CONTXT, NPROW, NPCOL, MYROW, MYCOL)                 
        
         !since, a quadratic process grid is going to be needed for
         !the matrix-matrix multiplications, its optimal row and column
         !numbers are estimated for the given number of processors
        a_PRCS=CEILING(SQRT(Real(PROCS)))
        IF (a_PRCS==SQRT(Real(PROCS))) THEN
           NPROW_GRID=a_PRCS
           NPCOL_TMP=a_PRCS        
        ENDIF
        IF (a_PRCS/=SQRT(Real(PROCS))) THEN
           DO i=1,CEILING(SQRT(Real(PROCS)))
              NPCOL_TMP=PROCS/Real(i)
              IF ((NPCOL_TMP-INT(NPCOL_TMP))==0) THEN
                 NPROW_GRID=i
              ENDIF
           ENDDO
           NPCOL_TMP=PROCS/NPROW_GRID
        ENDIF    
        IF (ABS(NPROW_GRID-a_PRCS)>(a_PRCS/2.0)) THEN
           WRITE(*,*)'The allocated number of CPUs does not allow for the use of an efficient process grid.'
           WRITE(*,*)'Suggested number of CPUs: 4, 16, 32, ....'
        ENDIF
      END SUBROUTINE INIT_BLACS_COLS


      !***********************************************************************
      !This routine initializes the process grid, which is used for the
      !matrix-matrix multiplications and their
      !block-cyclic data distribution. Note that the context variable
      !CONTXT_GRID is used for this grid.
      !The routine tries to create the most quadratic grid possible for the given
      !number of processors.
      !*********************************************************************** 

      SUBROUTINE INIT_BLACS_GRID(WDES)
         implicit none           
         TYPE(wavedes) WDES
         
         call BLACS_PINFO(ME,PROCS)
         
         nprow=NPROW_GRID
         npcol=PROCS/NPROW
         IF (MOD(PROCS/NPROW,1)/=0) THEN
            WRITE(*,*)'internal error in INIT_BLACS_GRID: Bad process grid'
         ENDIF
         
         IF (NGVECTOR>(PROCS*WDES%NBANDS)) THEN
            IF (NPROW_GRID<(PROCS/NPROW)) THEN
               NPCOL=NPROW_GRID
               NPROW=PROCS/NPROW_GRID
               NPROW_GRID=NPROW
            ENDIF
         ENDIF         
         IF (NGVECTOR<(PROCS*WDES%NBANDS)) THEN
            IF (NPROW_GRID>(PROCS/NPROW)) THEN
               NPCOL=NPROW_GRID
               NPROW=PROCS/NPROW_GRID
               NPROW_GRID=NPROW
            ENDIF
         ENDIF
         IF ((MYROW==0) .AND. (MYCOL==0)) THEN
           !WRITE(*,*)'You are using',PROCS,' processors.'
           WRITE(*,'(A,I3,A,I3,A)')'The allocated processors form a',NPROW_GRID,'x',NPCOL,' grid.'
         ENDIF
         
         call BLACS_PINFO(ME,PROCS)
         call BLACS_GET     (0, 0, CONTXT_GRID)  
         call BLACS_GRIDINIT(CONTXT_GRID, 'R', NPROW, NPCOL)        
         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         
        
      END SUBROUTINE INIT_BLACS_GRID
      
      !***********************************************************************
      !This subroutine redistributes the fourier-transformed overlap integrals <i|G|a>
      !from the column-only process grid (CONTXT) to the quadratic 
      !process grid (CONTXT_GRID)
      !***********************************************************************

      SUBROUTINE REDISTRIBUTE_FTOD_GRID(WDES,KI,KQ,ISP,tmp_FTOD_PW,tmp_FTOD_OC)      
         IMPLICIT NONE
         TYPE(wavedes) WDES
         INTEGER :: KI,KQ,ISP
         GDEF, TARGET :: tmp_FTOD_OC(:,:,:,:)
         COMPLEX(q), TARGET :: tmp_FTOD_PW(:,:,:,:)
         INTEGER :: FTOD_PW_rows, FTOD_PW_cols,cc,FTOD_OC_rows, FTOD_OC_cols
         INTEGER :: NBI,FTOD_PW_rows_br
         
         ! Prepare array descriptors for ScaLAPACK
         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         MB=MIN(((NGVECTOR)/NPROW),50)   !Row blocking size
         FTOD_PW_rows = numroc(NGVECTOR,mb,MYROW,0,NPROW)
         desc_FTOD_PW(3) = NGVECTOR       ! global number of rows
         desc_FTOD_PW(5) = mb       ! row block size
         desc_FTOD_PW(9) = MAX(1,FTOD_PW_rows) ! leading dimension of local array      

         desc_FTOD_PW_br(1) = 1              ! descriptor type
         desc_FTOD_PW_br(2) = contxt         ! blacs context
         desc_FTOD_PW_br(3) = NGVECTOR    ! global number of rows
         desc_FTOD_PW_br(4) = (PROCS*WDES%NBANDS) ! global number of cols
         desc_FTOD_PW_br(5) = NGVECTOR     ! row block size
         desc_FTOD_PW_br(6) = 1             ! col block size
         desc_FTOD_PW_br(7) = 0              ! initial process row
         desc_FTOD_PW_br(8) = 0              ! initial process col
         desc_FTOD_PW_br(9) = MAX(1,(NGVECTOR)) ! leading dimension of local array
         !Distribute the fourier-transformed overlap integrals
         DO NBI=1,VBMAX   
            DO cc=1,ncc
               CALL PZGEMR2D(NGVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_PW(1,1,NBI,cc),1,1,&
                desc_FTOD_PW_br,FTOD_PW(1,1,NBI,KI,KQ,ISP,cc),1,1,desc_FTOD_PW,contxt_grid)
            ENDDO
         ENDDO
         
#ifdef gammareal         
         desc_FTOD_PW(9) = MAX(1,m_ desc_FTOD_PW(9)) ! leading dimension of local array
         desc_FTOD_PW(5) = m_ desc_FTOD_PW(5)
         desc_FTOD_PW(3) = m_ desc_FTOD_PW(3)  
#else

#endif
         !ONE-center part FTOD_OC
           
         IF (ASSOCIATED(H)) THEN
         
            ! Prepare array descriptors for ScaLAPACK
         
            desc_FTOD_OC_br(1) = 1            ! descriptor type
            desc_FTOD_OC_br(2) = contxt       ! blacs context
            desc_FTOD_OC_br(3) = NHVECTOR     ! global number of rows
            desc_FTOD_OC_br(4) = (PROCS*WDES%NBANDS) ! global number of cols
            desc_FTOD_OC_br(5) = NHVECTOR     ! row block size
            desc_FTOD_OC_br(6) = 1            ! col block size
            desc_FTOD_OC_br(7) = 0            ! initial process row
            desc_FTOD_OC_br(8) = 0            ! initial process col
            desc_FTOD_OC_br(9) = MAX(1,(NHVECTOR)) ! leading dimension of local array
      
            DO NBI=1,VBMAX
               DO cc=1,2
#ifdef gammareal
                  CALL PDGEMR2D(NHVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_OC(1,1,NBI,cc),1,1,&
                   desc_FTOD_OC_br,FTOD_OC(1,1,NBI,KI,KQ,ISP,cc),1,1,desc_FTOD_OC,contxt_grid)
#else
                  CALL PZGEMR2D(NHVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_OC(1,1,NBI,cc),1,1,&
                   desc_FTOD_OC_br,FTOD_OC(1,1,NBI,KI,KQ,ISP,cc),1,1,desc_FTOD_OC,contxt_grid)
#endif
               ENDDO
            ENDDO           

         ENDIF
         
      END SUBROUTINE REDISTRIBUTE_FTOD_GRID

      !***********************************************************************
      !Allocate the matrices TWOE4ORBITAL(a,b)=<ij|ab> and 
      !TWOE4ORBITAL_X(a,b)=<ij|ba> in the block-cyclic data distribution
      !and setup their descriptors(desc_TWOE4ORBITAL and desc_TWOE4ORBITAL_X),
      !which are required by the scaLAPACK routines
      !***********************************************************************
            
      SUBROUTINE SETUP_TWOE4ORBITAL(WDES)
         IMPLICIT NONE
         TYPE(wavedes) WDES
         INTEGER MB,NB,TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS
         
         CALL BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         NPROW=NPROW_GRID
         NPCOL=PROCS/NPROW

         NB=MIN(((PROCS*WDES%NBANDS)/NPCOL),desc_FTOD_PW(6))   !column(CONDUCTION BANDS) blocking size
         MB=MIN(((PROCS*WDES%NBANDS)/NPROW),desc_FTOD_PW(5))   !row(CONDUCTION BANDS) blocking size
         
         TWOE4ORBITAL_ROWS = numroc((PROCS*WDES%NBANDS),mb,MYROW,0,NPROW)
         TWOE4ORBITAL_COLS = numroc((PROCS*WDES%NBANDS),nb,MYCOL,0,NPCOL)
         
         desc_TWOE4ORBITAL(1) = 1              ! descriptor type
         desc_TWOE4ORBITAL(2) = contxt_GRID         ! blacs context
         desc_TWOE4ORBITAL(3) = (PROCS*WDES%NBANDS) ! global number of rows(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL(4) = (PROCS*WDES%NBANDS) ! global number of columns(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL(5) = mb             ! row block size
         desc_TWOE4ORBITAL(6) = nb             ! column(conduction bands) block size
         desc_TWOE4ORBITAL(7) = 0              ! initial process row
         desc_TWOE4ORBITAL(8) = 0              ! initial process column
         desc_TWOE4ORBITAL(9) = MAX(1,TWOE4ORBITAL_ROWS)       ! leading dimension of local array
         
         desc_TWOE4ORBITAL_X(1) = 1              ! descriptor type
         desc_TWOE4ORBITAL_X(2) = contxt_GRID         ! blacs context
         desc_TWOE4ORBITAL_X(3) = (PROCS*WDES%NBANDS)    ! global number of rows(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL_X(4) = (PROCS*WDES%NBANDS)    ! global number of columns(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL_X(5) = mb              ! row block size
         desc_TWOE4ORBITAL_X(6) = nb              ! column(conduction bands) block size
         desc_TWOE4ORBITAL_X(7) = 0              ! initial process row
         desc_TWOE4ORBITAL_X(8) = 0              ! initial process column
         desc_TWOE4ORBITAL_X(9) = MAX(1,TWOE4ORBITAL_ROWS) ! leading dimension of local array    
         
         IF (ALLOCATED(TWOE4ORBITAL)) THEN
            DEALLOCATE(TWOE4ORBITAL)
         ENDIF
         IF (ALLOCATED(TWOE4ORBITAL_X)) THEN
            DEALLOCATE(TWOE4ORBITAL_X)
         ENDIF
         allocate(TWOE4ORBITAL(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))
         allocate(TWOE4ORBITAL_X(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))
         TWOE4ORBITAL=zero
      END SUBROUTINE SETUP_TWOE4ORBITAL

      !***********************************************************************
      !Allocate the arrays FTOD_PW and 
      !FTOD_OC in the block-cyclic data distribution
      !and setup their descriptors(desc_FTOD_PW and desc_FTOD_OC),
      !which are required by the scalaLAPACK routines
      !***********************************************************************
      
      SUBROUTINE SETUP_FTOD(WDES)
         IMPLICIT NONE
         TYPE(wavedes) WDES              
         INTEGER :: FTOD_PW_rows, FTOD_PW_cols,cc,FTOD_OC_rows, FTOD_OC_cols         

         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         !Blocking size for block-cyclic distribution of FTOD_PW        
         MB=MIN(((NGVECTOR)/NPROW),50)   !Row blocking size
         NB=MIN((NPROW*WDES%NBANDS),MB)   !column blocking size          
         
         FTOD_PW_rows = numroc(NGVECTOR,mb,MYROW,0,NPROW)
         FTOD_PW_cols = numroc(PROCS*WDES%NBANDS,nb,MYCOL,0,NPCOL)
         If ((MYROW==0) .AND. (MYCOL==0)) THEN
            write(*,'(A,I3,A,I3)')'Blocking size of <i|G|a> matrices:',mb,'x',nb
            write(*,'(A,I6,A,I6)')'Dimension of <i|G|a> matrices:',ngvector,'x',NPROW*WDES%NBANDS
         ENDIF
         desc_FTOD_PW(1) = 1              ! descriptor type
         desc_FTOD_PW(2) = contxt_grid         ! blacs context
         desc_FTOD_PW(3) = NGVECTOR       ! global number of rows
         desc_FTOD_PW(4) = (PROCS*WDES%NBANDS)   ! global number of cols
         desc_FTOD_PW(5) = mb       ! row block size
         desc_FTOD_PW(6) = nb       ! column block size
         desc_FTOD_PW(7) = 0              ! initial process row
         desc_FTOD_PW(8) = 0              ! initial process column
         desc_FTOD_PW(9) = MAX(1,FTOD_PW_rows) ! leading dimension of local array
         
         ALLOCATE(FTOD_PW(FTOD_PW_rows,FTOD_PW_cols,VBMAX,WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,ncc))
         
         IF (ASSOCIATED(H)) THEN         
            !Blocking size for block-cyclic distribution of FTOD_PW     
            MB=MIN(((NHVECTOR)/NPROW),50)   !Row blocking size
            NB=MIN((NPROW*WDES%NBANDS),MB)   !column blocking size
            If ((MYROW==0) .AND. (MYCOL==0)) THEN
               write(*,'(A,I3,A,I3)')'Blocking size of <i|G|a>^1 matrices:',mb,'x',nb
               write(*,'(A,I6,A,I6)')'Dimension of <i|G|a>^1 matrices:',nhvector,'x',NPROW*WDES%NBANDS
            ENDIF
            FTOD_OC_rows = numroc(NHVECTOR,mb,MYROW,0,NPROW)
            FTOD_OC_cols = numroc(PROCS*WDES%NBANDS,nb,MYCOL,0,NPCOL)
            
            desc_FTOD_OC(1) = 1              ! descriptor type
            desc_FTOD_OC(2) = contxt_grid         ! blacs context
            desc_FTOD_OC(3) = NHVECTOR       ! global number of rows
            desc_FTOD_OC(4) = (PROCS*WDES%NBANDS)   ! global number of cols
            desc_FTOD_OC(5) = mb             ! row block size
            desc_FTOD_OC(6) = nb             ! column block size
            desc_FTOD_OC(7) = 0              ! initial process row
            desc_FTOD_OC(8) = 0              ! initial process column
            desc_FTOD_OC(9) = MAX(1,FTOD_OC_rows) ! leading dimension of local array
            
            ALLOCATE(FTOD_OC(FTOD_OC_rows,FTOD_OC_cols,VBMAX,WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,2))
         ENDIF

      END SUBROUTINE SETUP_FTOD
      
      !***********************************************************************
      ! TRANSFORM LOCAL TO GLOBAL ARRAY INDEX FOR BLOCK-CYCLIC ARRAY
      ! DISTRIBUTION
      !***********************************************************************
      SUBROUTINE LOC2GLOB(li,p,n,np,nb,gi)
         IMPLICIT NONE
         INTEGER :: li   ! local index
         INTEGER :: p    ! index in processor grid (either MYROW or MYCOL)
         INTEGER :: n    ! global array dimension
         INTEGER :: np   ! processor array dimension
         INTEGER :: nb   ! blocking size
         INTEGER :: gi   ! global index
         INTEGER :: litmp   
 
         litmp = li-1
         gi=(((litmp/nb)*np)+p)*nb+mod(litmp,nb)+1
         RETURN
      END SUBROUTINE LOC2GLOB
      
      !***********************************************************************
      !Allocate the variables which are used for the calculations of the
      !quasiparticle energies
      !***********************************************************************      
      SUBROUTINE SETUP_MP2_bands(WDES)
         IMPLICIT NONE
         TYPE(wavedes) WDES
         
         ALLOCATE(E_HOMO_h(WDES%NKPTS,WDES%ISPIN),Z_HOMO_h(WDES%NKPTS,WDES%ISPIN))
         ALLOCATE(E_LUMO_h(WDES%NKPTS,WDES%ISPIN),Z_LUMO_h(WDES%NKPTS,WDES%ISPIN))
         ALLOCATE(E_HOMO_x(WDES%NKPTS,WDES%ISPIN),Z_HOMO_x(WDES%NKPTS,WDES%ISPIN))
         ALLOCATE(E_LUMO_x(WDES%NKPTS,WDES%ISPIN),Z_LUMO_x(WDES%NKPTS,WDES%ISPIN))
         E_HOMO_h(:,:)=zero
         E_LUMO_h(:,:)=zero
         E_HOMO_x(:,:)=zero
         E_LUMO_x(:,:)=zero
         Z_HOMO_h(:,:)=zero
         Z_LUMO_h(:,:)=zero
         Z_HOMO_x(:,:)=zero
         Z_LUMO_x(:,:)=zero

         Delta_finit=0.001_q !This value is used for the derivative of the quasiparticle energy
         ! with respect to \omega.
 
      END SUBROUTINE SETUP_MP2_bands
#endif
END MODULE mp2
