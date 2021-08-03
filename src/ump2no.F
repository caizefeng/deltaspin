#include "symbol.inc"
MODULE ump2no
      USE prec
      USE fock
      USE chi_base      
      USE lattice
      USE wpot
      IMPLICIT NONE
      
!**********************************************************************
!
! This module calculates the MP2 natural orbitals using essentially
! the same routines as in ump2.F.
! First, the virtual-virtual block of the second-order reduced one-electron density
! matrix is constructed and stored using a block-cyclic data distribution ( D2(:,:,:) ). 
! This matrix is diagonalized and the transformation is used to rotate the orbitals.
!
! The natural orbitals are NONCANONICAL !!!
! THE EIGENVALUES ARE *NOT* UPDATED USING THE ORBITAL TRANSFORMATION!!!
!
!  the central quantity is
!  < v_k1,n1 v'_k2,n2 | W | c'_k4,n4 c_k3,n3 > =
!  int_d3r d3r' c_k1-q,n3(r')  v'*_k2,n2(r') c'_k2+q,n4(r)  v*_k1,n1(r)  W(r',r)
!
! aG 2012
!**********************************************************************
      
      ! FTOD_PW Stores the two-orbital response functions at q 
      ! FTOD_OC Stores the (1._q,0._q) center part of the two-orbital response functions at q 
      ! <ij|ab> (q=k_b-k_j), FTOD_PW=(ia)   
      COMPLEX(q) , ALLOCATABLE, PRIVATE, SAVE :: FTOD_PW(:,:,:,:,:,:,:), NW_CW(:,:), FNO_CW_T(:,:), &
         NW_CW_T(:,:), ZD2(:,:)
      GDEF      , ALLOCATABLE, PRIVATE, SAVE :: FTOD_OC(:,:,:,:,:,:,:)      
      GDEF , ALLOCATABLE, PRIVATE, SAVE :: TWOE4ORBITAL(:,:), D2(:,:,:), D2EV(:,:)
      GDEF , ALLOCATABLE ::  WORK(:)
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
      integer, dimension(9)   :: desc_TWOE4ORBITAL, desc_W_CW, desc_W_CW_T
      !Scalapack array descriptor for TWOE4ORBITAL_X
      integer, dimension(9)   :: desc_TWOE4ORBITAL_X, desc_W_CW_br
      
      !BLACS related variables and function
      !PROCS.. number of processors, ME... processor number, NPROW... number of rows in process grid
      !NPCOL... number of columns in process grid, myrow,mycol... my coordinates in process grid
      !mb... blocking size of rows for block cyclic distribution
      !nb... blocking size of columns for block cyclic distribution      
      INTEGER, PRIVATE, SAVE :: PROCS,ME,NPROW,NPCOL,MYROW,MYCOL,CONTXT,CONTXT_GRID
      INTEGER, PRIVATE, SAVE :: NPROW_GRID, ncc
      INTEGER, PRIVATE, SAVE :: MB, NB
      INTEGER, ALLOCATABLE :: VBMAX(:)
      INTEGER, EXTERNAL :: NUMROC, INDXG2P
      !numroc... function that returns the number of rows or columns of the LOCAL ARRAY if you enter
      !the corresponding properties of the GLOBAL ARRAY
      COMPLEX(q), ALLOCATABLE :: RWORK(:), CELTOT_NEW(:,:,:)
      
      INTEGER :: LWORK
      !Some experimental flags to test approximations to the second-order density matrix.
      LOGICAL :: NORBITALS_ONLY
      LOGICAL :: OEAPPROXIMATE_NOs
      LOGICAL :: APPROXIMATE_NOs
      TYPE(one_center_handle), POINTER, PRIVATE, SAVE :: H

#ifdef scaLAPACK
      PRIVATE :: CALC_2ORBITAL_FTOD,REDISTRIBUTE_FTOD_GRID,INIT_BLACS_COLS,INIT_BLACS_GRID

      CONTAINS

!***********************************************************************
!Main MP2 natural orbitals routine.	
!For the calculations of the two-electron 4 orbital integrals scalaLAPCK is used.
!***********************************************************************
      SUBROUTINE CALCULATE_FNO(P,WDES,W,LATT_INI,LATT_CUR,T_INFO,IO,KPOINTS,WGW, ENCUTGW, ENCUTGWSOFT, LMAXMP2, INFO)
         USE prec
         USE poscar
         USE pseudo
         USE wave_high
         USE full_kpoints
         USE base
         USE mkpoints
         USE mpimy
         USE ini
         USE fileio
         IMPLICIT NONE
         TYPE(wavedes) WDES
         TYPE(wavedes) WGW
         TYPE(wavespin) W  
         TYPE(type_info) T_INFO
         TYPE(potcar) P(T_INFO%NTYP)
         TYPE (latt) LATT_INI
         TYPE(latt) LATT_CUR
         INTEGER LMAXMP2
         TYPE (info_struct) INFO
         REAL(q) :: ENCUTGW, ENCUTGWSOFT
         TYPE (in_struct) IO
         TYPE (kpoints_struct) KPOINTS
         !local variables
         TYPE (wavespin) WHF
         INTEGER KI,KQ,KQ_,KA,KB,ISP,KJ,NBI,NBJ,ISP2,PINFO,N
         INTEGER :: NBANDSGW
         LOGICAL :: qchange
         DOUBLE PRECISION, ALLOCATABLE :: WEV(:)
         INTEGER, ALLOCATABLE :: IWORK(:)
         INTEGER :: NA,NB,TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS,RNA,RNB,i
         INTEGER :: TNBANDS, LIWORK,LRWORK,RKA
         INTEGER :: NRC,IROFFC,ICOFFC,ICROW,ICCOL,MpC0,NqC0,SIZEMQRLEFT
         INTEGER :: W_CW_rows, W_CW_cols, W_CW_T_rows, W_CW_T_cols

         OEAPPROXIMATE_NOs=.FALSE.
         APPROXIMATE_NOs=.FALSE.

         NBANDSGW=-1 

         NORBITALS_ONLY=.TRUE.
#ifdef gammareal
         ncc=1
#else
         ncc=2
#endif

         CALL CHECK_FULL_KPOINTS ! all set up properly ?
         CALL START_TIMING("LOOP")
         SCALE=NKREDX*NKREDY*NKREDZ 
         IF (ODDONLY .OR. EVENONLY ) SCALE=SCALE*2.0_q

         ALLOCATE(VBMAX(WDES%ISPIN))
         ALLOCATE(CELTOT_NEW(WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))

         
         !Initialize the 1D process grid
         CALL INIT_BLACS_COLS()

         IF ((IO%IU6>0)) THEN
            OPEN(unit = 7,file = "D2EVS")
            CLOSE(7,STATUS='DELETE')
         ENDIF

         !Calculate the FTOD functions
         CALL CALC_2ORBITAL_FTOD(WDES,WGW,W,P,T_INFO,LATT_CUR,LMAXMP2,ENCUTGW,ENCUTGWSOFT,IO,FSG_STORE(1))         
                  
         CALL SETUP_TWOE4ORBITAL(WDES)
         FAC1=1.0_q
         FAC2=2.0_q
         IF (WDES%ISPIN==2) THEN
            FAC1=1.0_q
            FAC2=1.0_q
         ENDIF
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*)
            WRITE(IO%IU0,*)'Calculating MP2 natural orbitals:'
         ENDIF
 
         DO RKA=1,KPOINTS_ORIG%NKPTS
            IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS%VKPT(:,RKA))) CYCLE
            !WDES%NKPTS
            call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
            D2=zero
            D2EV=zero
            TWOE4ORBITAL=zero
            TWOE4ORBITAL_X=zero
 
            IF (IO%IU0>=0) WRITE(IO%IU0,*)
            IF (IO%IU0>=0) THEN
               WRITE(IO%IU0,'("RKI=",I4,3F10.4,", ")') RKA,KPOINTS%VKPT(:,RKA)
            ENDIF

        
            spin: DO ISP=1,WDES%ISPIN
            kqloop: DO KQ=1,WDES%NKPTS
               CALL START_TIMING("G")
               CALL GWPROGRESS(IO%IU0, KI,KPOINTS_ORIG%NKPTS,KJ,WDES%NKPTS)
               kiloop: DO KI=1,KPOINTS_ORIG%NKPTS
                  IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS%VKPT(:,KI))) CYCLE
                  IF (WDES%WTKPT(KI)==0) CYCLE
                  ! k_a = k_i - k_q - G
               
                  KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL)
                  IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS%VKPT(:,KA))) CYCLE
                  kjloop: DO KJ=1,WDES%NKPTS
                     IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS%VKPT(:,KJ))) CYCLE
                     IF ((OEAPPROXIMATE_NOs) .and. (KI/=KJ)) CYCLE
                     KB=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KJ)+WDES%VKPT(:,KQ),KPOINTS_FULL)
                     IF (SKIP_THIS_KPOINT_IN_FOCK(KPOINTS%VKPT(:,KB))) CYCLE
                     IF (KB/=RKA) CYCLE

                     ! k_q' = k_i - k_b - G
                     KQ_=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KB),KPOINTS_FULL)
                  
                     DO NBI=1,VBMAX(ISP) !loop over valence bands i                     
                        IF ((OEAPPROXIMATE_NOs) .and. (NBI>1)) CYCLE
                     
                        DO NBJ=1,VBMAX(ISP) !loop over valence bands j
                           IF ((OEAPPROXIMATE_NOs) .and. (NBJ>1)) CYCLE
                           IF ((APPROXIMATE_NOs) .and. (WDES%ISPIN==2)) CYCLE
                           IF ((APPROXIMATE_NOs) .and. (NBI/=NBJ)) CYCLE
                           ! Calculate TWOE4ORBITAL <ij|ab> for all bands a and b at
                           ! k-points k_a and k_b, respectively. Here i, j, k_i and
                           ! k_j are fixed 

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


                           CALL APPLY_DENOM(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP,LATT_CUR,NBANDSGW)
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                  (PROCS*WDES%NBANDS),one, TWOE4ORBITAL(1,1),1,1,&
                                  desc_TWOE4ORBITAL,TWOE4ORBITAL_X(1,1),1,1,&
                                  desc_TWOE4ORBITAL,one, D2(1,1,ISP),1,1,desc_TWOE4ORBITAL)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                  (PROCS*WDES%NBANDS),one, TWOE4ORBITAL(1,1),1,1,&
                                  desc_TWOE4ORBITAL,TWOE4ORBITAL_X(1,1),1,1,&
                                  desc_TWOE4ORBITAL,one, D2(1,1,ISP),1,1,desc_TWOE4ORBITAL)
#endif

   
                        ENDDO !nbj

                        IF (WDES%ISPIN==2) THEN
                           DO ISP2=1,WDES%ISPIN
                              IF (ISP==ISP2) CYCLE

                              IF ((EMPTY_MP2_ORBITAL(W%FERTOT(NBI,KI,ISP)))) CYCLE
                              DO NBJ=1,VBMAX(ISP2) !loop over valence bands j
                                 IF ((OEAPPROXIMATE_NOs) .and. (NBJ>1)) CYCLE
                                 IF ((APPROXIMATE_NOs) .and. (NBI/=NBJ)) CYCLE

                              !there is an additional term in the expression 
                              !of the 2nd order density matrix for the unrestricted case
#ifdef gammareal
                                 CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                        m_ NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                                        desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP2,ncc),1,1,&
                                        desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                                 CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                        NGVECTOR,-one, FTOD_PW(1,1,NBI,KI,KQ,ISP,1),1,1,&
                                        desc_FTOD_PW,FTOD_PW(1,1,NBJ,KJ,KQ,ISP2,ncc),1,1,&
                                        desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                                 IF (ASSOCIATED(H)) THEN
#ifdef gammareal
                                    CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                           NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                                           desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP2,2),1,1,&
                                           desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                                    CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                           NHVECTOR,-one, FTOD_OC(1,1,NBI,KI,KQ,ISP,1),1,1,&
                                           desc_FTOD_OC,FTOD_OC(1,1,NBJ,KJ,KQ,ISP2,2),1,1,&
                                           desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                                 ENDIF

                                 TWOE4ORBITAL(:,:)=GCONJG(TWOE4ORBITAL(:,:)*SCALE)
                                 TWOE4ORBITAL_X=zero

                                 CALL APPLY_DENOM(W,KI,KJ,KA,KB,NBI,NBJ,ISP,ISP2,LATT_CUR,NBANDSGW)

#ifdef gammareal
                                 CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                        (PROCS*WDES%NBANDS),one, TWOE4ORBITAL(1,1),1,1,&
                                        desc_TWOE4ORBITAL,TWOE4ORBITAL(1,1),1,1,&
                                        desc_TWOE4ORBITAL,one, D2(1,1,ISP2),1,1,desc_TWOE4ORBITAL)
#else


                                 CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                                        (PROCS*WDES%NBANDS),one, TWOE4ORBITAL(1,1),1,1,&
                                        desc_TWOE4ORBITAL,TWOE4ORBITAL(1,1),1,1,&
                                        desc_TWOE4ORBITAL,one, D2(1,1,ISP2),1,1,desc_TWOE4ORBITAL)

#endif


                              ENDDO ! NBJ loop over valence bands j only
                           ENDDO !ISP2
                        ENDIF

                     ENDDO ! NBI lop over valence bands i only

                  ENDDO kjloop
               ENDDO kiloop

               CALL STOP_TIMING("G",IO%IU6,"MP2")
            ENDDO kqloop
            ENDDO spin

            D2=D2*(KPOINTS_ORIG%WTKPT(1)*KPOINTS_ORIG%WTKPT(1))*one
            call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL) 
            TWOE4ORBITAL_ROWS = numroc(desc_TWOE4ORBITAL(3),desc_TWOE4ORBITAL(5),MYROW,0,NPROW)
            TWOE4ORBITAL_COLS = numroc(desc_TWOE4ORBITAL(4),desc_TWOE4ORBITAL(6),MYCOL,0,NPCOL)
            DO ISP=1,WDES%ISPIN
               DO NB=1,TWOE4ORBITAL_COLS
                  CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                  DO NA=1,TWOE4ORBITAL_ROWS
                     CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                     IF (RNA<=VBMAX(ISP)) D2(NA,NB,ISP)=zero
                     IF (RNB<=VBMAX(ISP)) D2(NA,NB,ISP)=zero
                     IF (NBANDSGW/=-1) THEN
                        IF (RNA<=NBANDSGW) D2(NA,NB,ISP)=zero
                        IF (RNB<=NBANDSGW) D2(NA,NB,ISP)=zero
                     ENDIF

                     IF ((RNA==RNB) .and. (RNA<=VBMAX(ISP)))   D2(NA,NB,ISP)=one*1000
                     IF ((RNA==RNB) .and. (RNA<=VBMAX(ISP)))   D2(NA,NB,ISP)=one*1000
                     IF (NBANDSGW/=-1) THEN
                        IF ((RNA==RNB) .and. (RNA<=NBANDSGW))   D2(NA,NB,ISP)=one*1000
                        IF ((RNA==RNB) .and. (RNA<=NBANDSGW))   D2(NA,NB,ISP)=one*1000
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
            N=(PROCS*WDES%NBANDS)

            IF (ALLOCATED(WORK)) DEALLOCATE(WORK)
            IF (ALLOCATED(RWORK)) DEALLOCATE(RWORK)
            IF (ALLOCATED(WEV)) DEALLOCATE(WEV)
            ALLOCATE(WORK(1))
            ALLOCATE(RWORK(1))
            WORK=0
            RWORK=0
            ALLOCATE(WEV(N))
            WEV=(0.0_q)
            D2EV=zero

#ifdef gammareal
            CALL PDSYEV( 'V', 'L', N, D2(1,1,1), 1, 1, desc_TWOE4ORBITAL, WEV(1),&
                   D2EV(1,1), 1, 1, desc_TWOE4ORBITAL, WORK(1), -1, PINFO)
#else
            CALL PZHEEV( 'V', 'L', N, D2(1,1,1), 1, 1, desc_TWOE4ORBITAL, WEV(1),&
                   D2EV(1,1), 1, 1, desc_TWOE4ORBITAL, WORK(1), -1,RWORK(1),-1,PINFO)
#endif

            LWORK=WORK(1)
            DEALLOCATE(WORK)
            DEALLOCATE(RWORK)
            ALLOCATE(WORK(LWORK))
            LRWORK=2*N+2*N-2
            ALLOCATE(RWORK(LRWORK)) 
            WORK=zero
            RWORK=zero

            IF ((IO%IU6>0) .and. (RKA==1)) THEN
               OPEN(unit = 7,file = "FOCKM")
               CLOSE(7,STATUS='DELETE')
            ENDIF

            DO ISP=1,WDES%ISPIN
               WEV(:)=0.0_q
               D2EV=zero
               DO I=1,PROCS*WDES%NBANDS
                  IF (WEV(I)<INFO%EDIFF) N=I
               ENDDO


               CALL START_TIMING("PD")
#ifdef gammareal
               CALL PDSYEV( 'V', 'L', N, D2(1,1,ISP), 1, 1, desc_TWOE4ORBITAL, WEV(1),&
                      D2EV(1,1), 1, 1, desc_TWOE4ORBITAL, WORK, LWORK, PINFO )
#else
               CALL PZHEEV( 'V', 'L', N, D2(1,1,ISP), 1, 1, desc_TWOE4ORBITAL, WEV(1),&
                      D2EV(1,1), 1, 1, desc_TWOE4ORBITAL, WORK, LWORK,RWORK,LRWORK, PINFO )
#endif

               CALL STOP_TIMING("PD",IO%IU6,"PDSYEVD")
   
               D2(:,:,ISP)=D2EV(:,:)
 
               IF ((IO%IU6>0)) THEN
                  OPEN(unit = 9,file = "D2EVS") !,ACCESS="APPEND")
                  DO RNA=1,WDES%NB_TOT
                     write(9,*)RNA,RKA,ISP,WEV(WDES%NB_TOT-RNA+1)
                  ENDDO
                  CLOSE(9)
               ENDIF
               TNBANDS=0
               N=0
               DO I=1,PROCS*WDES%NBANDS
                  IF (WEV(I)<INFO%EDIFF) N=I
!                  IF (IO%IU0>=0) WRITE(IO%IU0,*)'EV',WEV(I)
               ENDDO
               IF (TNBANDS<PROCS*WDES%NBANDS) TNBANDS=PROCS*WDES%NBANDS-N
               IF (IO%IU0>=0) write(IO%IU0,*)
               IF (IO%IU0>=0) write(IO%IU0,*)'Truncating all eigenvalues below ',INFO%EDIFF,'results in ',PROCS*WDES%NBANDS-N,'frozen natural orbitals in spin channel',ISP
            ENDDO


 
            DO ISP=1,WDES%ISPIN
               D2EV=zero

#ifdef gammareal
               CALL PDTRAN(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      D2(1,1,ISP),1,1,desc_TWOE4ORBITAL,zero,&
                      D2EV(1,1),1,1,desc_TWOE4ORBITAL_X)
#else
               CALL PZTRANU(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      D2(1,1,ISP),1,1,desc_TWOE4ORBITAL,zero,&
                      D2EV(1,1),1,1,desc_TWOE4ORBITAL_X)
#endif
               D2(:,:,ISP)=D2EV(:,:)
               D2EV=zero
               DO NA=1,TWOE4ORBITAL_ROWS
                  DO NB=1,TWOE4ORBITAL_COLS
                     CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                     CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                     IF ((-RNA+(PROCS*WDES%NBANDS)+1)==RNB) D2EV(NA,NB)=one*1000
                  ENDDO
               ENDDO

#ifdef gammareal
               CALL PDGEMM('t','n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,D2(1,1,ISP),1,1,&
                      desc_TWOE4ORBITAL,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
 
               D2(:,:,ISP)=TWOE4ORBITAL(:,:)

               CALL PDTRAN(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      D2(1,1,ISP),1,1,desc_TWOE4ORBITAL,zero,&
                      TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL_X)

               D2(:,:,ISP)=TWOE4ORBITAL(:,:)

               DO NB=1,TWOE4ORBITAL_COLS
                  CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                  DO NA=1,TWOE4ORBITAL_ROWS
                     CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                     IF (RNA<=VBMAX(ISP)) D2(NA,NB,ISP)=zero
                     IF (RNB<=VBMAX(ISP)) D2(NA,NB,ISP)=zero

                     IF ((RNA==RNB) .and. (RNB<=VBMAX(ISP)))  D2(NA,NB,ISP)=one*1000
                  ENDDO
               ENDDO

#else
               CALL PZGEMM('t','n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,D2(1,1,ISP),1,1,&
                      desc_TWOE4ORBITAL,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)

               D2(:,:,ISP)=TWOE4ORBITAL(:,:)

               CALL PZTRANU(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      D2(1,1,ISP),1,1,desc_TWOE4ORBITAL,zero,&
                      TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL_X)

               D2(:,:,ISP)=TWOE4ORBITAL(:,:)

               DO NB=1,TWOE4ORBITAL_COLS
                  CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                  DO NA=1,TWOE4ORBITAL_ROWS
                     CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                     IF (RNA<=VBMAX(ISP)) D2(NA,NB,ISP)=zero
                     IF (RNB<=VBMAX(ISP)) D2(NA,NB,ISP)=zero
                     IF (NBANDSGW/=-1) THEN
                        IF (RNA<=NBANDSGW) D2(NA,NB,ISP)=zero
                        IF (RNB<=NBANDSGW) D2(NA,NB,ISP)=zero
                     ENDIF
                     IF ((RNA==RNB) .and. (RNB<=VBMAX(ISP)))  D2(NA,NB,ISP)=one*1000
                     IF (NBANDSGW/=-1) THEN
                     IF ((RNA==RNB) .and. (RNB<=NBANDSGW))  D2(NA,NB,ISP)=one*1000
                     ENDIF
                  ENDDO
               ENDDO


#endif

               IF (.not. NORBITALS_ONLY) THEN

               D2EV=zero
               DO NA=1,TWOE4ORBITAL_ROWS
                  DO NB=1,TWOE4ORBITAL_COLS
                     CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                     CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                     IF (RNA==RNB) D2EV(NA,NB)=W%CELTOT(RNA,RKA,ISP)
                  ENDDO
               ENDDO

#ifdef gammareal
               CALL PDGEMM('t','n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one, D2(1,1,ISP),1,1,&
                      desc_TWOE4ORBITAL,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)

               CALL PDTRAN(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL,zero,&
                      D2EV(1,1),1,1,desc_TWOE4ORBITAL_X)

               CALL PDGEMM('t','n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one,D2(1,1,ISP) ,1,1,&
                      desc_TWOE4ORBITAL,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,zero,TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
               CALL PZGEMM('t','n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one, D2(1,1,ISP),1,1,&
                      desc_TWOE4ORBITAL,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)

               CALL PZTRANU(PROCS*WDES%NBANDS,PROCS*WDES%NBANDS,one,&
                      TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL,zero,&
                      D2EV(1,1),1,1,desc_TWOE4ORBITAL_X)

               CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                      (PROCS*WDES%NBANDS),one,D2(1,1,ISP) ,1,1,&
                      desc_TWOE4ORBITAL,D2EV(1,1),1,1,&
                      desc_TWOE4ORBITAL,zero,TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
               CELTOT_NEW(:,RKA,ISP)=(0.0_q)
               DO I=1,PROCS
                  IF (ME==I-1) THEN

                  DO NA=1,TWOE4ORBITAL_ROWS
                     DO NB=1,TWOE4ORBITAL_COLS
                        CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)
                        CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                        IF (RNA==RNB) CELTOT_NEW(RNA,RKA,ISP)=TWOE4ORBITAL(NA,NB)
                     ENDDO
                  ENDDO
       
               ENDIF

            ENDDO

            CALLMPI( M_sum_z(WGW%COMM_INTER, CELTOT_NEW(1,RKA,ISP), WDES%NB_TOT))

            CALL FOCKM_OUT(IO,WGW,WDES,RKA,ISP)
    
         ENDIF
         ENDDO



         IF (ALLOCATED(ZD2)) DEALLOCATE(ZD2)
         ALLOCATE(ZD2(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))

!         IF (IO%IU6>0) WRITE(*,*)'calculating MP2 natural orbitals'

         DO ISP=1,WDES%ISPIN
            CALL START_TIMING("R2")
            CALL SETUP_REDISTRIBUTE_W_CW_GRID(W,WDES,RKA,ISP)
            CALL STOP_TIMING("R2",IO%IU6,"PZR2d")
       
            CALL BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
            W_CW_rows = numroc(desc_W_CW(3),desc_W_CW(5),MYROW,0,NPROW)
            W_CW_cols = numroc(desc_W_CW(4),desc_W_CW(6),MYCOL,0,NPCOL)
            W_CW_T_rows = numroc(desc_W_CW_T(3),desc_W_CW_T(5),MYROW,0,NPROW)
            W_CW_T_cols = numroc(desc_W_CW_T(4),desc_W_CW_T(6),MYCOL,0,NPCOL)

       !write(*,*)'redistribution done' !,W%CW(1:10,:,1,1)

!******************************************
            ZD2(:,:)=D2(:,:,ISP)*(1.0_q,0.0_q)
            CALL PZTRANU((PROCS*WDES%NBANDS),(WDES%NRPLWV),(1.0_q,0.0_q),&
                   NW_CW(1,1),1,1,desc_W_CW,(0.0_q,0.0_q),&
                   NW_CW_T(1,1),1,1,desc_W_CW_T)

            CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(WDES%NRPLWV),&
                   (PROCS*WDES%NBANDS),(1.0_q,0.0_q),ZD2(1,1),1,1,&
                   desc_TWOE4ORBITAL,NW_CW_T(1,1),1,1,&
                   desc_W_CW_T,(0.0_q,0.0_q),FNO_CW_T(1,1),1,1,desc_W_CW_T)
!******************************************

            CALL PZTRANU((WDES%NRPLWV),(PROCS*WDES%NBANDS),(1.0_q,0.0_q),&
                   FNO_CW_T(1,1),1,1,desc_W_CW_T,(0.0_q,0.0_q),&
                   NW_CW(1,1),1,1,desc_W_CW)

            CALL BACKDISTRIBUTE_NO_2_W_CW(W,WDES,RKA,ISP)
!******************************************

            call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)

         ENDDO


      ENDDO !RKA

    
!      CALL FOCKM_IN(IO,WDES)
         W%CELTOT=CELTOT_NEW
         IF (IO%IU6>0) WRITE(*,*)'writing MP2 natural orbitals to the WAVECAR.FNO file'
         CALL OUTWAV(IO, WDES, W, LATT_INI, 0.0_q, 'FNO')

         call BLACS_GRIDEXIT(contxt)
         call BLACS_GRIDEXIT(contxt_grid)
         CALL EXIT(0)
         RETURN
      END SUBROUTINE CALCULATE_FNO

      !****************************************************************************
      !
      ! This Subroutine applies the denominator (e_i+e_j-e_a-e_b) to
      ! TWOE4ORBITAL for the calculation of the second-order density matrix.
      !
      !***********************************************************************

      SUBROUTINE APPLY_DENOM(W,KI,KJ,KA,KB,NI,NJ,ISP1,ISP2,LATT_CUR,NBANDSGW) 
         USE constant
         USE full_kpoints
         USE mkpoints
         USE wave
         IMPLICIT NONE 
         TYPE (wavespin) W
         INTEGER :: KI,KJ,KA,KB,NI,NJ,I
         TYPE(latt) LATT_CUR
         INTEGER :: NA,NB,TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS
         INTEGER :: RNA,RNB,ISP1,ISP2,KI_IN_FULL_ORIG,KJ_IN_FULL_ORIG,kq1,kq2
         REAL(q) :: DENOM, OCC, VIRT 
         INTEGER :: NBANDSGW
         
         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
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
 
         DO NB=1,TWOE4ORBITAL_COLS
            CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
            DO NA=1,TWOE4ORBITAL_ROWS
               CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)     
               
               OCC =W%FERTOT(NI,KI,ISP1)*W%FERTOT(NJ,KJ,ISP2) !*SQRT(KPOINTS_ORIG%WTKPT(KI))
               VIRT=(1._q-W%FERTOT(RNA,KA,ISP1))*(1._q-W%FERTOT(RNB,KB,ISP2))
!               IF ((RNA<=VBMAX) .or. (RNB<=VBMAX)) VIRT=0.0_q
               IF ((NBANDSGW/=-1) .and. ((RNA<=NBANDSGW) .or. (RNB<=NBANDSGW))) THEN
                  TWOE4ORBITAL(NA,NB)=zero
                  TWOE4ORBITAL_X(NA,NB)=zero
                  VIRT=zero
               ENDIF
               IF (RNA<=VBMAX(ISP1)) THEN
                  TWOE4ORBITAL(NA,NB)=zero
                  TWOE4ORBITAL_X(NA,NB)=zero
                  VIRT=zero
               ENDIF
               IF (RNB<=VBMAX(ISP2)) THEN
                  TWOE4ORBITAL(NA,NB)=zero
                  TWOE4ORBITAL_X(NA,NB)=zero
                  VIRT=zero
               ENDIF

               DENOM=(REAL(W%CELTOT(NI,KI,ISP1),KIND=q)+&
                 REAL(W%CELTOT(NJ,KJ,ISP2),KIND=q)-REAL(W%CELTOT(RNB,KB,ISP2),KIND=q)-&
                 REAL(W%CELTOT(RNA,KA,ISP1),KIND=q))
            IF (DENOM<0._q) THEN
               IF (ISP2==ISP1) THEN
                  TWOE4ORBITAL_X(NA,NB)=(TWOE4ORBITAL(NA,NB)*FAC2-TWOE4ORBITAL_X(NA,NB))/DENOM*OCC*VIRT*FAC1 !*SQRT(KPOINTS_ORIG%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI))
                  TWOE4ORBITAL(NA,NB)=TWOE4ORBITAL(NA,NB)/DENOM*OCC*VIRT !*SQRT(KPOINTS_ORIG%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI))
               ENDIF

               IF (ISP2/=ISP1) THEN
                  TWOE4ORBITAL(NA,NB)=TWOE4ORBITAL(NA,NB)*OCC*VIRT/DENOM !*SQRT(KPOINTS_ORIG%WTKPT(KI)*KPOINTS_ORIG%WTKPT(KI))
               ENDIF !isp1/=isp2

             ENDIF !denom<0 
               
           ENDDO
         ENDDO

      END SUBROUTINE APPLY_DENOM

      !***********************************************************************
      !This routine calculates the fourier-transformed overlap integrals <i|-G|a>
      !and <j|G|b>*4*pi*e^2/(G+q)^2 and also calls the redistribution routine.
      !This is done for the plane-wave part, as well as for the one-center terms.
      !Note, that in the gamma-only version it is enough to store <i|-G|a>*SQRT(4*pi*e^2/(G+q)^2)
      !for the plane-wave part, because (<i|-G|a>)*=<i|G|a>.
      !However, this approximation cannot be made for the one-center terms in the gamma-only
      !version, only for technical reasons.
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
         
         VBMAX(1)=LAST_FILLED_XI_NOMOD(W,1,1)
         DO ISP=2,WDES%ISPIN
            VBMAX(2)=LAST_FILLED_XI_NOMOD(W,1,ISP)
         ENDDO
         
         ALLOCATE(WI(MAX(VBMAX(1),VBMAX(WDES%ISPIN))),WA(NSTRIP),WB(NSTRIP))
         DO NBI=1,MAX(VBMAX(1),VBMAX(WDES%ISPIN))
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
         mem_req=mem_req+NGVECTOR*PROCS*WDES%NBANDS*MAX(VBMAX(1),VBMAX(WDES%ISPIN))*WDES%NKPTS*WDES%NKPTS*WDES%ISPIN*2.0_q*16.0_q/1024.0_q/1024.0_q/1024.0_q 
         IF (LMAXMP2>=0) mem_req=mem_req+NHVECTOR*PROCS*WDES%NBANDS*MAX(VBMAX(1),VBMAX(WDES%ISPIN))*WDES%NKPTS*WDES%NKPTS*WDES%ISPIN*2.0_q*16.0_q/1024.0_q/1024.0_q/1024.0_q
!         IF (IO%IU0>0) THEN
!            WRITE(*,"(A,E10.3,A)") 'For MP2 calculations approximately',mem_req,'GB RAM will be required.' 
!         ENDIF
         IF (IO%IU0>0) THEN
!            WRITE(IO%IU0,*) 'Allocating memory...'
         ENDIF
         !Setup the descriptors for the distributed TWOE4ORBITAL matrix   
         !and allocate the TWOE4ORBITAL and TWOE4ORBITAL_X matrices
         CALL SETUP_TWOE4ORBITAL(WDES)
         IF (ALLOCATED(TWOE4ORBITAL)) THEN
            DEALLOCATE(TWOE4ORBITAL)
         ENDIF
         IF (ALLOCATED(D2)) THEN
            DEALLOCATE(D2)
         ENDIF
         IF (ALLOCATED(D2EV)) THEN
            DEALLOCATE(D2EV)
         ENDIF
         IF (ALLOCATED(TWOE4ORBITAL_X)) THEN
            DEALLOCATE(TWOE4ORBITAL_X)
         ENDIF
         IF (IO%IU0>0) THEN
!            WRITE(IO%IU0,*) 'succeeded'
         ENDIF
         
         ALLOCATE(tmp_FTOD_PW(NGVECTOR,WDES%NBANDS,MAX(VBMAX(1),VBMAX(WDES%ISPIN)),ncc))
         IF (ASSOCIATED(H)) THEN
            ALLOCATE(tmp_FTOD_OC(NHVECTOR,WDES%NBANDS,MAX(VBMAX(1),VBMAX(WDES%ISPIN)),2))
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
               IF (SKIP_THIS_KPOINT_IN_FOCK(WDES%VKPT(:,KI))) CYCLE

               tmp_FTOD_PW=(0._q,0._q)
               IF (ASSOCIATED(H)) tmp_FTOD_OC=zero
               
               CALL GWPROGRESS(IO%IU0, KI,WDES%NKPTS,KQ,WDES%NKPTS)
               KI_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI),KPOINTS_FULL_ORIG)
               ! collect all valence bands at k_i
               CALL SETWDES(WHF%WDES,WDESKI,KI)
               CALL W1_GATHER_GLB(WHF,1,VBMAX(ISP),ISP,WI)
               IF (OEAPPROXIMATE_NOs) THEN
                  DO NBI=2,VBMAX(ISP)
                     IF ((EMPTY_MP2_ORBITAL(W%FERTOT(NBI,KI,ISP)))) CYCLE 
                     WI(1)%CR(:)=ABS(WI(1)%CR(:))+ABS(WI(NBI)%CR(:))
                     WI(1)%CPROJ(:)=WI(1)%CPROJ(:)+WI(NBI)%CPROJ(:)
                  ENDDO
               ENDIF
               ! k_b = k_i - k_q - G
               KB=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KQ)+WDES%VKPT(:,KI),KPOINTS_FULL)
               IF (SKIP_THIS_KPOINT_IN_FOCK(WDES%VKPT(:,KB))) CYCLE

               ! k_a = k_i + k_q - G
               KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL)
               IF (SKIP_THIS_KPOINT_IN_FOCK(WDES%VKPT(:,KA))) CYCLE
               
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
                  DO NBI=1,MAX(VBMAX(1),VBMAX(WDES%ISPIN))
                     IF ((OEAPPROXIMATE_NOs) .and. (NBI>1)) CYCLE
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
                     
                        CALL APPLY_GFAC_WAVEFUN(WGWQ,GCHGIA(1,NBAA,1), &
                         POTFAK(1))
                         
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
                  DO NBI=1,MAX(VBMAX(1),VBMAX(WDES%ISPIN))
                     IF ((OEAPPROXIMATE_NOs) .and. (NBI>1)) CYCLE
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
         
         DO NBI=1,MAX(VBMAX(1),VBMAX(WDES%ISPIN))
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
         INTEGER :: NBI,FTOD_PW_rows_br,VBMAX_tmp
         
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
         VBMAX_tmp=MAX(VBMAX(1),VBMAX(WDES%ISPIN))
         IF (OEAPPROXIMATE_NOs) VBMAX_tmp=1
         DO NBI=1,VBMAX_tmp
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
     
            IF (OEAPPROXIMATE_NOs) VBMAX_tmp=1
            DO NBI=1,VBMAX_tmp
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
      !This subroutine redistributes the orbitals.
      !***********************************************************************

      SUBROUTINE SETUP_REDISTRIBUTE_W_CW_GRID(W,WDES,KI,ISP)
         IMPLICIT NONE
         TYPE(wavespin) W
         TYPE(wavedes) WDES
         INTEGER :: KI,ISP
         INTEGER :: W_CW_rows, W_CW_cols,cc, W_CW_T_rows, W_CW_T_cols
         INTEGER :: NBI,FTOD_PW_rows_br, mb_T, nb_T
         
         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         !Blocking size for block-cyclic distribution of FTOD_PW        
         MB=MIN(INT(WDES%NRPLWV/NPROW),50)   !Row blocking size
         NB=MIN((NPROW*WDES%NBANDS),MB)   !column blocking size 

         MB_T=MIN((PROCS*WDES%NBANDS/NPROW),50)   !Row blocking size
         NB_T=MIN(INT(WDES%NRPLWV/NPCOL),MB_T)   !column blocking size 
 
!         WRITE(*,*)'MB,NB',MB,NB,NPROW, NPCOL, MYROW, MYCOL, WDES%NRPLWV, WDES%NBANDS,KI,WDES%NRPLWV

!         ! Prepare array descriptors for ScaLAPACK
!         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
!         MB=MIN(((WDES%NRPLWV)/NPROW),50)   !Row blocking size
         W_CW_rows = numroc(WDES%NRPLWV,mb,MYROW,0,NPROW)
         W_CW_cols = numroc(PROCS*WDES%NBANDS,nb,MYCOL,0,NPCOL)
         W_CW_T_rows = numroc(PROCS*WDES%NBANDS,mb_T,MYROW,0,NPROW)
         W_CW_T_cols = numroc(WDES%NRPLWV,nb_T,MYCOL,0,NPCOL)
         IF (ALLOCATED(NW_CW)) DEALLOCATE(NW_CW)
         IF (ALLOCATED(NW_CW_T)) DEALLOCATE(NW_CW_T)
         IF (ALLOCATED(FNO_CW_T)) DEALLOCATE(FNO_CW_T)
         ALLOCATE(NW_CW(W_CW_rows,W_CW_cols))
         ALLOCATE(NW_CW_T(W_CW_T_rows,W_CW_T_cols))
         ALLOCATE(FNO_CW_T(W_CW_T_rows,W_CW_T_cols))
 
         NW_CW=(0.0_q,0.0_q)
         NW_CW_T=(0.0_q,0.0_q)
         FNO_CW_T=(0.0_q,0.0_q)

         desc_W_CW(1) = 1              ! descriptor type
         desc_W_CW(2) = contxt_grid         ! blacs context
         desc_W_CW(3) = (WDES%NRPLWV)       ! global number of rows
         desc_W_CW(4) = (PROCS*WDES%NBANDS)   ! global number of cols
         desc_W_CW(5) = mb       ! row block size
         desc_W_CW(6) = nb       ! column block size
         desc_W_CW(7) = 0              ! initial process row
         desc_W_CW(8) = 0              ! initial process column
         desc_W_CW(9) = MAX(1,W_CW_rows) ! leading dimension of local array      

         desc_W_CW_T(1) = 1              ! descriptor type
         desc_W_CW_T(2) = contxt_grid         ! blacs context
         desc_W_CW_T(3) = (PROCS*WDES%NBANDS)       ! global number of rows
         desc_W_CW_T(4) = WDES%NRPLWV   ! global number of cols
         desc_W_CW_T(5) = mb_T       ! row block size
         desc_W_CW_T(6) = nb_T       ! column block size
         desc_W_CW_T(7) = 0              ! initial process row
         desc_W_CW_T(8) = 0              ! initial process column
         desc_W_CW_T(9) = MAX(1,W_CW_T_rows) ! leading dimension of local array      

         desc_W_CW_br(1) = 1              ! descriptor type
         desc_W_CW_br(2) = contxt         ! blacs context
         desc_W_CW_br(3) = (WDES%NRPLWV)    ! global number of rows
         desc_W_CW_br(4) = (PROCS*WDES%NBANDS) ! global number of cols
         desc_W_CW_br(5) = (WDES%NRPLWV)     ! row block size
         desc_W_CW_br(6) = 1             ! col block size
         desc_W_CW_br(7) = 0              ! initial process row
         desc_W_CW_br(8) = 0              ! initial process col
         desc_W_CW_br(9) = MAX(1,(WDES%NRPLWV)) ! leading dimension of local array
         !Distribute the fourier-transformed overlap integrals

!         CALL PZGEMR2D((WDES%NRPLWV),(PROCS*WDES%NBANDS),W%CW(1,1,KI,ISP),1,1,&
!              desc_W_CW_br,NW_CW(1,1),1,1,desc_W_CW,contxt_grid)

         CALL PZGEMR2D((WDES%NRPLWV),(PROCS*WDES%NBANDS),W%CW(1,1,KI,ISP),1,1,&
                desc_W_CW_br,NW_CW(1,1),1,1,desc_W_CW,contxt_grid)


      END SUBROUTINE SETUP_REDISTRIBUTE_W_CW_GRID

      SUBROUTINE BACKDISTRIBUTE_NO_2_W_CW(W,WDES,KI,ISP)
         IMPLICIT NONE
         TYPE(wavespin) W
         TYPE(wavedes) WDES
         INTEGER :: KI,ISP
         INTEGER :: W_CW_rows, W_CW_cols,cc
         INTEGER :: NBI,FTOD_PW_rows_br

         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         W%CW(:,:,KI,ISP)=(0.0_q,0.0_q)
         CALL PZGEMR2D((WDES%NRPLWV),(PROCS*WDES%NBANDS),NW_CW(1,1),1,1,&
                desc_W_CW,W%CW(1,1,KI,ISP),1,1,desc_W_CW_br,desc_W_CW_br(2))

      END SUBROUTINE BACKDISTRIBUTE_NO_2_W_CW

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
         TWOE4ORBITAL_COLS = numroc((PROCS*WDES%NBANDS),mb,MYCOL,0,NPCOL)
         
         desc_TWOE4ORBITAL(1) = 1              ! descriptor type
         desc_TWOE4ORBITAL(2) = contxt_GRID         ! blacs context
         desc_TWOE4ORBITAL(3) = (PROCS*WDES%NBANDS) ! global number of rows(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL(4) = (PROCS*WDES%NBANDS) ! global number of columns(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL(5) = mb             ! row block size
         desc_TWOE4ORBITAL(6) = mb             ! column(conduction bands) block size
         desc_TWOE4ORBITAL(7) = 0              ! initial process row
         desc_TWOE4ORBITAL(8) = 0              ! initial process column
         desc_TWOE4ORBITAL(9) = MAX(1,TWOE4ORBITAL_ROWS)       ! leading dimension of local array
         
         desc_TWOE4ORBITAL_X(1) = 1              ! descriptor type
         desc_TWOE4ORBITAL_X(2) = contxt_GRID         ! blacs context
         desc_TWOE4ORBITAL_X(3) = (PROCS*WDES%NBANDS)    ! global number of rows(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL_X(4) = (PROCS*WDES%NBANDS)    ! global number of columns(conduction bands&valence bands*procs)
         desc_TWOE4ORBITAL_X(5) = mb              ! row block size
         desc_TWOE4ORBITAL_X(6) = mb              ! column(conduction bands) block size
         desc_TWOE4ORBITAL_X(7) = 0              ! initial process row
         desc_TWOE4ORBITAL_X(8) = 0              ! initial process column
         desc_TWOE4ORBITAL_X(9) = MAX(1,TWOE4ORBITAL_ROWS) ! leading dimension of local array    
         
 
         IF (ALLOCATED(TWOE4ORBITAL)) THEN
            DEALLOCATE(TWOE4ORBITAL)
         ENDIF
         IF (ALLOCATED(D2)) THEN
            DEALLOCATE(D2)
         ENDIF
         IF (ALLOCATED(D2EV)) THEN
            DEALLOCATE(D2EV)
         ENDIF
         IF (ALLOCATED(TWOE4ORBITAL_X)) THEN
            DEALLOCATE(TWOE4ORBITAL_X)
         ENDIF
         allocate(TWOE4ORBITAL(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))
         allocate(TWOE4ORBITAL_X(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))
         allocate(D2(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS,WDES%ISPIN))
         allocate(D2EV(TWOE4ORBITAL_ROWS,TWOE4ORBITAL_COLS))
         TWOE4ORBITAL=zero
         TWOE4ORBITAL_X=zero
         D2=zero
         D2EV=zero
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
         
         IF (OEAPPROXIMATE_NOs) THEN
            ALLOCATE(FTOD_PW(FTOD_PW_rows,FTOD_PW_cols,1,WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,ncc))
         ELSE
            ALLOCATE(FTOD_PW(FTOD_PW_rows,FTOD_PW_cols,MAX(VBMAX(1),VBMAX(WDES%ISPIN)),WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,ncc))
         ENDIF
         
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
            IF (OEAPPROXIMATE_NOs) THEN            
               ALLOCATE(FTOD_OC(FTOD_OC_rows,FTOD_OC_cols,1,WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,2))
            ELSE
               ALLOCATE(FTOD_OC(FTOD_OC_rows,FTOD_OC_cols,MAX(VBMAX(1),VBMAX(WDES%ISPIN)),WDES%NKPTS,WDES%NKPTS,WDES%ISPIN,2))
            ENDIF
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
      

      SUBROUTINE FOCKM_OUT(IO,WGW,WDES,KI,ISP)
         USE base
         IMPLICIT NONE
         TYPE (in_struct) IO
         TYPE(wavedes) WGW
         TYPE(wavedes) WDES
         INTEGER :: I,NB,NA,RNB,RRNB,ISP,IREC,KI
         INTEGER :: desc_FOCKM_br(9), desc_FOCKM(9)
         GDEF, ALLOCATABLE :: FOCKM_LINE(:)
         REAL(q) :: sig

         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         ALLOCATE(FOCKM_LINE(WDES%NB_TOT))
         IREC=1+(KI-1)*WDES%NB_TOT+(ISP-1)*(WDES%NKPTS*WDES%NB_TOT)
         IF ((KI/=1) .or. ISP/=1) IREC=IREC+1
         DO RNB=1,PROCS*WDES%NBANDS

               desc_FOCKM(1) = 1              ! descriptor type
               desc_FOCKM(2) = contxt_grid         ! blacs context
               desc_FOCKM(3) = (PROCS*WDES%NBANDS)    ! global number of rows
               desc_FOCKM(4) = NPCOL ! global number of cols
               desc_FOCKM(5) = (PROCS*WDES%NBANDS)     ! row block size
               desc_FOCKM(6) = 1            ! col block size
               desc_FOCKM(7) = 0              ! initial process row
               desc_FOCKM(8) = 0              ! initial process col
               IF (MYROW==0) THEN
                  desc_FOCKM(9) = MAX(1,(PROCS*WDES%NBANDS)) ! leading dimension of local array
               ELSE
                  desc_FOCKM(9) = 1
               ENDIF

               desc_FOCKM_br(1) = 1              ! descriptor type
               desc_FOCKM_br(2) = contxt_grid         ! blacs context
               desc_FOCKM_br(3) = (PROCS*WDES%NBANDS)    ! global number of rows
               desc_FOCKM_br(4) = NPCOL ! global number of cols
               desc_FOCKM_br(5) = desc_TWOE4ORBITAL(5)     ! row block size
               desc_FOCKM_br(6) = 1             ! col block size
               desc_FOCKM_br(7) = 0              ! initial process row
               desc_FOCKM_br(8) = 0              ! initial process col
               desc_FOCKM_br(9) = MAX(1,desc_TWOE4ORBITAL(9)) ! leading dimension of local array
                  !Distribute the fourier-transformed overlap integrals

               DO NB=1,(PROCS*WDES%NBANDS) !WDES%NBANDS*NPROW
                  CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RRNB)
                  sig=0.0_q
                  IF ((RNB==RRNB) .and. (MYROW==0)) sig=1.0_q
                  CALL M_sum_d(WGW%COMM_INTER, sig, 1)

                  IF ((sig==1.0_q) .and. (RNB==RRNB)) THEN
                     IF (MYROW==0) THEN
                        OPEN(UNIT=19,FILE='FOCKM',ACCESS='DIRECT', &
                        FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=((WDES%NB_TOT+3)*IO%ICMPLX*2))
                        IF (IREC==1) THEN
                            WRITE(19,REC=IREC)WDES%ISPIN,WDES%NKPTS,WDES%NB_TOT
                            IREC=IREC+1
                        ENDIF
                     ENDIF
#ifdef gammareal
                     CALL PDGEMR2D((PROCS*WDES%NBANDS),(NPCOL),TWOE4ORBITAL(1,NB),1,1,&
                        desc_FOCKM_br,FOCKM_LINE(1),1,1,desc_FOCKM,contxt_grid)
#else
                     CALL PZGEMR2D((PROCS*WDES%NBANDS),(NPCOL),TWOE4ORBITAL(1,NB),1,1,&
                        desc_FOCKM_br,FOCKM_LINE(1),1,1,desc_FOCKM,contxt_grid)
#endif
                     IF (MYROW==0) THEN
                        WRITE(19,REC=IREC) ((1.0_q,0.0_q)*FOCKM_LINE(:))
                        CLOSE(19)
                     ENDIF

                  ELSEIF ((sig==1.0_q) .and. (RNB/=RRNB)) THEN
                     IF (IREC==1) IREC=IREC+1
#ifdef gammareal
                     CALL PDGEMR2D((PROCS*WDES%NBANDS),(NPCOL),TWOE4ORBITAL(1,1),1,1,&
                        desc_FOCKM_br,FOCKM_LINE(1),1,1,desc_FOCKM,contxt_grid)
#else
                     CALL PZGEMR2D((PROCS*WDES%NBANDS),(NPCOL),TWOE4ORBITAL(1,1),1,1,&
                        desc_FOCKM_br,FOCKM_LINE(1),1,1,desc_FOCKM,contxt_grid)
#endif


                  ENDIF

               ENDDO
               IREC=IREC+1

         ENDDO
      END SUBROUTINE FOCKM_OUT

#endif

END MODULE ump2no
