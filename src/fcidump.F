#include "symbol.inc"
MODULE FCIDUMP
      USE prec
      USE fock
      USE chi_base      
      USE lattice
      USE wpot
      IMPLICIT NONE

!**********************************************************************
!
! This routine writes the 1- and 2- body terms to the FCIDUMP file.
! The FCIDUMP file can be used as input for any integral-based electronic
! structure code to calculate the electronic correlation energy.
!
! The FCIDUMP file is used as an interface between VASP and the 
! FCIQMC code of the Alavi Group for Full Configuration Interaction
! calculations.
!
! This module was written by aG with some help (for the interface to NECI) from jS and gB.
! 
!**********************************************************************

      COMPLEX(q) , ALLOCATABLE, PRIVATE, SAVE :: FTOD_PW(:,:,:,:,:,:,:)
      GDEF      , ALLOCATABLE, PRIVATE, SAVE :: FTOD_OC(:,:,:,:,:,:,:)
      INTEGER, PRIVATE, SAVE :: NGVECTOR, NHVECTOR
      !Scalapack array descriptor for FTOD_ before redistribution
      integer, dimension(9)   :: desc_FTOD_PW_br, desc_FTOD_OC_br      
      !Scalapack array descriptor for FTOD_ after redistribution
      integer, dimension(9)   :: desc_FTOD_PW, desc_FTOD_OC
      integer, dimension(9)   :: desc_TWOE4ORBITAL, desc_TWOE4ORBITAL_X
      GDEF , ALLOCATABLE, PRIVATE, SAVE :: TWOE4ORBITAL(:,:)
      GDEF , ALLOCATABLE, PRIVATE, SAVE :: TWOE4ORBITAL_X(:,:)
      !BLACS related variables and function
      !PROCS.. number of processors, ME... processor number, NPROW... number of rows in process grid
      !NPCOL... number of columns in process grid, myrow,mycol... my coordinates in process grid
      !mb... blocking size of rows for block cyclic distribution
      !nb... blocking size of columns for block cyclic distribution      
      INTEGER, PRIVATE, SAVE :: PROCS,ME,NPROW,NPCOL,MYROW,MYCOL,CONTXT_COLS,CONTXT_GRID
      INTEGER, PRIVATE, SAVE :: NPROW_GRID, ncc
      INTEGER, PRIVATE, SAVE :: VBMAX, MB, NB
      INTEGER, EXTERNAL :: NUMROC, BLACS_PNUM
      COMPLEX(q) :: E_MP2, EFOCK_tot,eVtoHar
      LOGICAL :: SHIFTED_KPOINTS, DUMPFNO
      ! Combined index lookup arrays
      integer, allocatable, save :: band_index(:)   ! nKPTS*nBands.  band index of combined index.
      integer, allocatable, save :: kpnt_index(:)   ! nKPTS*nBands.  k-point index of combined index.
      integer, allocatable, save :: b_k_index(:,:)  ! nKPTS,nBands.  combined index of the ib-th band of the ik-th k-point.
      integer, allocatable, save :: inverse_of_k(:) ! nKPTS. index of inverse k-point.
      logical, allocatable, save :: kx_negative(:)  ! nKPTS. kx_negative(ik) true if k_ik,x<0.
      TYPE(wavedes) WDES_MOD
      TYPE(one_center_handle), POINTER, PRIVATE, SAVE :: H
      GDEF, ALLOCATABLE :: FOCKM(:,:,:,:)

#ifdef scaLAPACK
      PRIVATE :: CALC_FTOD,REDISTRIBUTE_FTOD_GRID,INIT_BLACS_COLS,INIT_BLACS_GRID

      CONTAINS

!***********************************************************************
!***********************************************************************
      SUBROUTINE CALC_FCIDUMP(P,WDES,W,LATT_CUR,T_INFO,IO,KPOINTS,WGW, ENCUTGW, ENCUTGWSOFT, LMAXMP2, INFO)
         USE prec
         USE poscar
         USE pseudo
         USE wave_high
         USE full_kpoints
         USE base
         USE mkpoints
         USE mpimy
         USE ini
         ! NECI module(s)
!         USE readinput
         IMPLICIT NONE
         TYPE(wavedes) WDES
         TYPE(wavedes) WGW
         TYPE(wavespin) W        
         TYPE(type_info) T_INFO
         TYPE(potcar) P(T_INFO%NTYP)
         TYPE(latt) LATT_CUR
         TYPE (info_struct) INFO
         INTEGER LMAXMP2         
         REAL(q) :: ENCUTGW, ENCUTGWSOFT
         TYPE (in_struct) IO
         TYPE (kpoints_struct) KPOINTS
         character(255) :: neci_inputfile
         GDEF :: CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
         !local variables

         call BLACS_PINFO(ME,PROCS)
         IF (PROCS>1) THEN
            CALLMPI( M_stop('CALC_FCIDUMP: #cores>1 not implemented, sorry.') )
            STOP
         ENDIF

         WDES_MOD=WDES
         DUMPFNO=.FALSE.
#ifdef gammareal
   ncc=1
   eVtoHar=(0.03674932540_q)
#else
   ncc=2
   eVtoHar=dcmplx(0.03674932540_q,0.e0_q)
#endif

! Initialisation.
         CALL INSULATOR_CHECK(W,WDES)
         ALLOCATE(FOCKM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN))
         FOCKM=zero

!
!        This routine can be useful for non-canonical reference states.
!         CALL FOCKM_IN(IO,WDES,W)
!

         eVtoHar=dcmplx(0.03674932540_q,0.e0_q)
         CALL CHECK_FULL_KPOINTS ! all set up properly ?
         CALL CHECK_SHIFTED_KPOINTS(WDES,W,KPOINTS)
         IF ((IO%IU0>0) .and. (SHIFTED_KPOINTS)) write(IO%IU0,*)'You are using a shifted k-mesh.'
         !Initialize the 1D process grid
         CALL INIT_BLACS_COLS()
         !Calculate the ftod also calls INIT_BLACS_GRID
         CALL INVERT_KPOINT(WDES,WGW,W,IO)
         CALL CALC_FTOD(WDES,WGW,W,P,T_INFO,LATT_CUR,LMAXMP2,ENCUTGW,ENCUTGWSOFT,IO,FSG_STORE(1))
         !one contxt for every process
         IF (IO%IU0>=0) write(*,*)''
         !Allocate the TWOE4ORBITAL and TWOE4ORBITAL_X matrix
         CALL SETUP_TWOE4ORBITAL(WDES)
         !Evaluate the Fock exchange energy(test)
!         call CALC_EFOCK(WDES,WGW,IO)
! \Initialisation.
         
! Initialise NECI
         call LabelStates(W,WDES)

         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         !useful debugging routines, very slow!!!!!
         !CALL TEST_IJAB(W)
         !CALL TEST_IJAB_one(W)
         !CALL TEST_IJAB_one_MP2(W,WDES)
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*)
            WRITE(IO%IU0,*)'Writing to FCIDUMP file.'
         ENDIF

         IF (.not. DUMPFNO) THEN
            !dump out all 4 index integrals
            CALL WRITE4INDEX2DISK(W,WDES)
         ENDIF

         deallocate(band_index)
         deallocate(kpnt_index)
         deallocate(b_k_index)
         deallocate(inverse_of_k)
         deallocate(kx_negative)

! Exit BLACS cleanly.
         call BLACS_GRIDEXIT(contxt_cols)
         call BLACS_GRIDEXIT(contxt_grid)
         
      END SUBROUTINE CALC_FCIDUMP


      subroutine LabelStates(W,WDES)

         ! Set up arrays which allow conversion between the VASP indexing scheme
         ! (1 index for the spin, 1 for the band, 1 for the k-point) to the NECI
         ! indexing scheme (1 combined index).

         ! To convert from the NECI ordering scheme to the VASP ordering scheme
         ! use the SPLITINDEX subroutine.

         use full_kpoints
         use prec
         use wave
         implicit none
         type(wavespin), intent(in) :: W        
         type(wavedes), intent(in) :: WDES
         type(kpoints_struct) :: KPOINTS
         integer :: ind,ib,ik,combined_index(2)
         real(q) :: degen_tolerance,prev_eigv
         real(q), allocatable :: eigv(:,:)

         degen_tolerance=1.d-6

         ! Temporary real array of eigenvalues so we can use minloc.
         allocate(eigv(WDES%NBANDS,WDES%NKPTS))

         do ik=1,WDES%NKPTS
            do ib=1,WDES%NBANDS
               eigv(ib,ik)=real(W%CelTot(ib,ik,1),kind=q)
            end do
         end do

         allocate(band_index(WDES%NBANDS*WDES%NKPTS*PROCS))
         allocate(kpnt_index(WDES%NKPTS*WDES%NBANDS*PROCS))
         allocate(b_k_index(WDES%NBANDS,WDES%NKPTS))
         allocate(inverse_of_k(WDES%NKPTS))
         allocate(kx_negative(WDES%NKPTS))

         band_index(:)=0
         kpnt_index(:)=0
         b_k_index(:,:)=0

         combined_index=minloc(eigv)
         band_index(1)=combined_index(1)
         kpnt_index(1)=combined_index(2)
         b_k_index(band_index(1),kpnt_index(1))=1
         prev_eigv=eigv(combined_index(1),combined_index(2))

         do ind=2,WDES%NKPTS*WDES%NBANDS
            ! Check for degenerate states first.
            chkdegen: do ik=1,WDES%NKPTS
               do ib=1,WDES%NBANDS
                  if (b_k_index(ib,ik)/=0) then
                     ! Have already found the combined index for this band.
                     cycle
                  else if ((eigv(ib,ik)-prev_eigv).lt.degen_tolerance) then
                     ! Found degenerate state.
                     band_index(ind)=ib
                     kpnt_index(ind)=ik
                     b_k_index(ib,ik)=ind
                     exit chkdegen
                  end if
               end do
            end do chkdegen
            if ((kpnt_index(ind)==0) .and. (band_index(ind)==0)) then
               ! Have not found a degenerate state.  Find the next lowest energy
               ! band.
               prev_eigv=prev_eigv+degen_tolerance
               combined_index=minloc(eigv,mask=eigv.gt.prev_eigv)
               band_index(ind)=combined_index(1)
               kpnt_index(ind)=combined_index(2)
               b_k_index(band_index(ind),kpnt_index(ind))=ind
               prev_eigv=eigv(combined_index(1),combined_index(2))
            end if
         end do

         ! K-point information for determining a unique set of indices for a
         ! 4-index, 2-electron integral.
         do ik=1,WDES%NKPTS
            inverse_of_k(ik)=KPOINT_IN_FULL_GRID(-WDES%VKPT(:,ik),KPOINTS_FULL)
            kx_negative(ik)=WDES%VKPT(1,ik).lt.0.d0
         end do

         deallocate(eigv)
      
         return
      end subroutine LabelStates


      subroutine SplitIndex(ind,ik,ib,is)

         ! Convert a combined spin-orbital index (in the style of NECI) into
         ! the indexing scheme used by VASP.

         ! In:
         !    ind: combined index for all orbitals.  This refers to a specific
         !    spin-orbitial at a specific k-point in unrestricted calculations
         !    and a specific spatial orbital at a specific k-point in restricted
         !    calculations.
         ! Out:
         !    ik: k-point of orbital.
         !    ib: band index of orbital.
         !    is: spin index of orbital.

         implicit none
         integer, intent(in) :: ind
         integer, intent(out) :: ik,ib, is
         integer :: tmp_ind

         if (WDES_MOD%ISPIN==1) then
             ! restricted
             is = 1
             tmp_ind = ind
         else
             ! unrestricted.
             ! An odd index refers to an alpha (is=1) orbital.
             is = mod(ind,2) + 1
             ! kpnt_index and band_index refer to the spatial orbital
             tmp_ind = (ind+1)/2
         end if

         ik=kpnt_index(tmp_ind)
         ib=band_index(tmp_ind)

      end subroutine SplitIndex


      subroutine KPntSymInt(i,j,k,l,a,b,c,d)
!        Take a set of (combined) indices and return a unique set, using
!        the relations between k-points and their inverses.
!        Using u_{a,k1}(r)=u_{a,-k1}(r)*, then
!        < a,k1  b,k2 | c,k3 d,k4 > = < c,-k3 d,-k4 | a,-k1 b,-k2 >
!        If k1=k3 (which implies k2=k4 to meet the k-point selection rule),
!        then additional symmetry properties apply as the potential,
!        4Pi/|G-k1+k3|^2 is now symmetric. In this case, the following
!        holds:
!        < a,k1  b,k2 | c,k1 d,k2 > = < c,-k1 d,-k2 | a,-k1 b,-k2 >
!                                   = < c,-k1 b,k2  | a,-k1 d,k2  >
!                                   = < a,k1  d,-k2 | c,k1  b,-k2 >
!        Uniquely order by forcing k1,x to be positive, and (if
!        k1=k3) force k2,x to also be positive.  This fixes the
!        other k-points to take a certain value.
!        A k-point is explicitly stored if its index is less or equal to
!        nKPnt (the number of unique k-points up to inversion symmetry).
!        Return in different variables as i,j,k,l might be loop indices,
!        and we should never change a loop index inside the loop...
!        Return new combined indices in a,b,c,d

         implicit none
         integer :: i,j,k,l,a,b,c,d
         integer :: ia,ja,ka,la,ik,jk,kk,lk,is

         ! Split up state and k-point indices from each argument.
         call SplitIndex(i,ik,ia,is) ! combined index,state index,kpoint index.
         call SplitIndex(j,jk,ja,is)
         call SplitIndex(k,kk,ka,is)
         call SplitIndex(l,lk,la,is)

         if (i.eq.k.or.j.eq.l) then
             ! Require a and b to be at explicitly stored k-points.
             if (kx_negative(ik)) then
                 ik=inverse_of_k(ik)
                 kk=inverse_of_k(kk)
                 a=b_k_index(ka,kk)
                 c=b_k_index(ia,ik)
             else
                 a=i
                 c=k
             end if
             if (kx_negative(jk)) then
                 jk=inverse_of_k(jk)
                 lk=inverse_of_k(lk)
                 b=b_k_index(la,lk)
                 d=b_k_index(ja,jk)
             else
                 b=j
                 d=l
             end if
         else if (kx_negative(ik)) then
             ! Swap all k-points to their inverse.
             ! Hence generate a set of indices where ik is explicitly
             ! stored.
             ik=inverse_of_k(ik)
             jk=inverse_of_k(jk)
             kk=inverse_of_k(kk)
             lk=inverse_of_k(lk)
             a=b_k_index(ka,kk)
             b=b_k_index(la,lk)
             c=b_k_index(ia,ik)
             d=b_k_index(ja,jk)
         else
             ! Have our unique ordering already.
             a=i
             b=j
             c=k
             d=l
         end if

         return
      end subroutine KPntSymInt


      !***********************************************************************
      !This routine calculates the fourier-transformed overlap integrals(FTOD) <i|-G|a>
      !and <j|G|b>*4*pi*e^2/(G+q)^2 and also calls the redistribution routine.
      !This is done for the plane-wave part, as well as for the one-center terms.
      !Note, that in the gamma-only version it is enough to save <i|-G|a>*SQRT(4*pi*e^2/(G+q)^2)
      !for the plane-wave part, because (<i|-G|a>)*=<i|G|a>.
      !However, this approximation cannot be made for the one-center terms in the gamma-only
      !version, because of practical reasons.
      !The <i|G|a> quantities are most important for the construction of the two electron
      !four orbital integrals <ij|ab>.  <ij|ab>=4pi e^2 \sum_G <i|-G|a> <j|G|b> /(G+q)^2
      !***********************************************************************

      SUBROUTINE CALC_FTOD(WDES,WGW,W,P,T_INFO,LATT_CUR,LMAXMP2,ENCUTGW,ENCUTGWSOFT,IO,FSG)
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
         INTEGER NBI,NBA,NBAA, i, nbi_start, nbi_end,rnba
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
         
         VBMAX=LAST_FILLED_XI_NOMOD(W,1,1)
         
         DO ISP=2,WDES%ISPIN
            VBMAX=MAX(LAST_FILLED_XI_NOMOD(W,1,ISP),VBMAX)
         ENDDO
         
         ALLOCATE(WI(NSTRIP),WA(NSTRIP),WB(NSTRIP))
         DO NBI=1,NSTRIP !VBMAX
            CALL NEWWAV(WI(NBI),WDESKI,.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL NEWWAV(WA(NBA),WDESKA,.TRUE.)
         ENDDO
         DO NBA=1,NSTRIP
            CALL NEWWAV(WB(NBA),WDESKB,.TRUE.)
         ENDDO
         
         NGVECTOR=MAXVAL(WGW%NGVECTOR(:))
         NGVECTOR=NGVECTOR+1
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
         !write(*,*)'before setupf ftod'
         
         !STOP
         CALL SETUP_FTOD(WDES)
         
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
         
         ALLOCATE(tmp_FTOD_PW(NGVECTOR,WDES%NBANDS,NSTRIP,ncc)) !VBMAX
         IF (ASSOCIATED(H)) THEN
            ALLOCATE(tmp_FTOD_OC(NHVECTOR,WDES%NBANDS,NSTRIP,2)) !VBMAX
         ENDIF
         
         IF (IO%IU0>0) THEN
            WRITE(IO%IU0,*)
            WRITE(IO%IU0,*)'Calculating fourier transformed overlap densities:'
         ENDIF

         call BLACS_GRIDINFO(CONTXT_COLS, NPROW, NPCOL, MYROW, MYCOL)
         
         IF (ASSOCIATED(H)) THEN
            ALLOCATE(CRHOIA(NHVECTOR, NSTRIP),CRHOIB(NHVECTOR, NSTRIP))
            FTOD_OC=zero
         ENDIF
         
         spin: DO ISP=1,WDES%ISPIN
         kqloop: DO KQ=1,WDES%NKPTS
            IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KQ)/=0.00_q)) CYCLE
            IF (IO%IU0>=0) WRITE(IO%IU0,*)
            IF (IO%IU0>=0) THEN
               IF (WDES%ISPIN==1) THEN
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,", ")') KQ,WDES%VKPT(:,KQ)
               ELSE
                 WRITE(IO%IU0,'("NQ=",I4,3F10.4,I4", ")') KQ,WDES%VKPT(:,KQ),ISP
               ENDIF
            ENDIF
            CALL SETWDES(WGW,WGWQ,KQ)
         
            NP=WGWQ%NGVECTOR
            IF (NP>NGVECTOR) THEN
               WRITE(*,*)'Internal error in "Calc_2orbital_response": NP larger than NGVECTOR'
               EXIT
            ENDIF
            ALLOCATE(GCHGIA(NP,NSTRIP,2),CRHOLM(AUG_DES%NPRO*WDES%NRSPINORS))
            !write(*,*)'procs=',procs
            kiloop: DO KI=1,WDES%NKPTS
               IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KI)==0)) CYCLE
               tmp_FTOD_PW=(0._q,0._q)
               IF (ASSOCIATED(H)) tmp_FTOD_OC=zero
               
               CALL GWPROGRESS(IO%IU0, KI,WDES%NKPTS,KQ,WDES%NKPTS)
               KI_IN_FULL_ORIG=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,KI),KPOINTS_FULL_ORIG)
               ! collect NSTRIP nbi bands at k_i
               CALL SETWDES(WHF%WDES,WDESKI,KI)
               
               DO NBI_start=1,PROCS*WDES%NBANDS,NSTRIP
                  call BLACS_GRIDINFO(CONTXT_COLS, NPROW, NPCOL, MYROW, MYCOL)
                  NBI_end=NBI_start+MIN(PROCS*WDES%NBANDS-NBI_start+1,NSTRIP)-1
                  
                  CALL W1_GATHER_GLB(WHF,NBI_start,NBI_end,ISP,WI)
                  ! k_b = k_i - k_q - G
                  KB=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KQ)+WDES%VKPT(:,KI),KPOINTS_FULL)
                  IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KB)==0)) Write(*,*)'error in calc_ftod with shifted k-mesh'
                  ! k_a = k_i + k_q - G
                  KA=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KQ),KPOINTS_FULL)
                  IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KA)==0)) Write(*,*)'error in calc_ftod with shifted k-mesh'
               
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
                  DO NBI=1,MIN(PROCS*WDES%NBANDS-NBI_start+1,NSTRIP)
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
                        !copy response functions to RESPF_PW and RESPF_OC
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
                          !copy response functions to RESPF_PW and RESPF_OC
                           IF (ASSOCIATED(H)) THEN
                              tmp_FTOD_OC(1:H%TOTAL_ENTRIES,NBAA+NBA-1,NBI,2)=(CRHOIA(1:H%TOTAL_ENTRIES,NBAA))    !000000000000000000000
                           ENDIF
                        ENDDO
                     ENDIF
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!GAMMA only version !!!!!!!!!!!!!!!!!!!!!!!!

                     ENDDO !NBI (loop over nstrip nbi bands only)
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
                     DO NBI=1,MIN(PROCS*WDES%NBANDS-NBI_start+1,NSTRIP)
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
                           !copy response functions to RESPF_PW and RESPF_OC
                        
                           tmp_FTOD_PW(1:NP,NBAA+NBA-1,NBI,2)=(GCHGIA(1:NP,NBAA,2))*(1.0_q/GRIDHF%NPLWV)
                           IF (ASSOCIATED(H)) THEN
                              tmp_FTOD_OC(1:H%TOTAL_ENTRIES,NBAA+NBA-1,NBI,2)=CRHOIB(1:H%TOTAL_ENTRIES,NBAA)
                           ENDIF

                           IF (REAL(W%CELTOT(NBI+nbi_start-1,KI,1),kind=q)>19.4_q) then
                              !tmp_FTOD_PW(:,NBAA+NBA-1,NBI,2)=zero
                              !tmp_FTOD_PW(:,NBAA+NBA-1,NBI,1)=zero
                              !write(*,*)'ki,nbi',ki,NBI+nbi_start-1
                           endif
                           CALL LOC2GLOB(NBAA+NBA-1,MYCOL,PROCS*WDES%NBANDS,PROCS,1,rnba)
                           !write(*,*)'rnba=',rnba
                           !CALL LOC2GLOB(NS,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNS)
                           IF (REAL(W%CELTOT(rnba,KA,1),kind=q)>19.4_q) then
                              
                              !tmp_FTOD_PW(:,NBAA+NBA-1,NBI,1)=zero
                              !write(*,*)'ka,nba',ka,rnba,NBI+nbi_start-1
                           endif
                           IF (REAL(W%CELTOT(rnba,KB,1),kind=q)>19.4_q) then
                              !tmp_FTOD_PW(:,NBAA+NBA-1,NBI,2)=zero
                              
                              !write(*,*)'kb,nba',kb,rnba,NBI+nbi_start-1
                           endif
                           !write(*,*)'nbi,nba',NBI+nbi_start-1,rnba,nba+nbaa-1
                        ENDDO
                     ENDDO !NBI (loop over nstrip bands only)
                  ENDDO !NBA (loop over all bands) 
               
#endif
                  CALL REDISTRIBUTE_FTOD_GRID(WDES,KI,KQ,ISP,tmp_FTOD_PW,tmp_FTOD_OC,nbi_start,nbi_end)
               ENDDO !NBI_start (loop over all bands)
            ENDDO kiloop
         
            DEALLOCATE(GCHGIA,CRHOLM)
         
         ENDDO kqloop
         ENDDO spin
         
         IF (ALLOCATED(CRHOIA)) DEALLOCATE(CRHOIA)
         IF (ALLOCATED(CRHOIB)) DEALLOCATE(CRHOIB)
         IF (ALLOCATED(tmp_FTOD_OC)) DEALLOCATE(tmp_FTOD_OC)
         IF (ALLOCATED(tmp_FTOD_PW)) DEALLOCATE(tmp_FTOD_PW)
         
         DO NBI=1,NSTRIP
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
      END SUBROUTINE CALC_FTOD

      SUBROUTINE CONSTRUCT_IJAB(I,J,A,TWOE4ORB)
         use mkpoints
         IMPLICIT NONE
         INTEGER :: I,J,A
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,ISP1,ISP2,ISP3
         GDEF :: TWOE4ORB(:,:)
         
         call SPLITINDEX(I,NKI,NI,ISP1)
         call SPLITINDEX(J,NKJ,NJ,ISP2)
         call SPLITINDEX(A,NKA,NA,ISP3)

         if (ISP1 == ISP3) then

             NKQ=KPOINT_IN_FULL_GRID(WDES_MOD%VKPT(:,NKI)-WDES_MOD%VKPT(:,NKA),KPOINTS_FULL)

             CALL ZGEMM(trans,'N',(PROCS*WDES_MOD%NBANDS),(PROCS*WDES_MOD%NBANDS),NGVECTOR, &
                evtoHar,FTOD_PW(1,1,NI,NKI,NKQ,1,ISP1),NGVECTOR,FTOD_PW(1,1,NJ,NKJ,NKQ,ncc,ISP2),&
                NGVECTOR,(0.0_q,0.0_q),TWOE4ORB(1,1),(PROCS*WDES_MOD%NBANDS))

             IF (ASSOCIATED(H)) then
                CALL ZGEMM(trans,'N',(PROCS*WDES_MOD%NBANDS),(PROCS*WDES_MOD%NBANDS),NHVECTOR, &
                   evtoHar,FTOD_OC(1,1,NI,NKI,NKQ,1,ISP1),NHVECTOR,FTOD_OC(1,1,NJ,NKJ,NKQ,2,ISP2),&
                   NHVECTOR,(1.0_q,0.0_q),TWOE4ORB(1,1),(PROCS*WDES_MOD%NBANDS))
             endif

         else

             TWOE4ORB = zero
             
         end if

      END SUBROUTINE  CONSTRUCT_IJAB

      SUBROUTINE CONSTRUCT_IJAB_one(I,J,A,B,TWOE4ORB)
         use mkpoints
         IMPLICIT NONE
         INTEGER :: I,J,A, B
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,NB,NKB,ISP1,ISP2,ISP3,ISP4,NKBJ
         GDEF :: TWOE4ORB(1,1)

         call SPLITINDEX(I,NKI,NI,ISP1)
         call SPLITINDEX(J,NKJ,NJ,ISP2)
         call SPLITINDEX(A,NKA,NA,ISP3)
         call SPLITINDEX(B,NKB,NB,ISP4)

         NKQ=KPOINT_IN_FULL_GRID(WDES_MOD%VKPT(:,NKI)-WDES_MOD%VKPT(:,NKA),KPOINTS_FULL)
         NKBJ=KPOINT_IN_FULL_GRID(WDES_MOD%VKPT(:,NKB)-WDES_MOD%VKPT(:,NKJ),KPOINTS_FULL)
         if (ISP1 == ISP3 .and. ISP2 == ISP4 .and. NKQ == NKBJ) THEN
#ifdef gammareal
             CALL DGEMM(trans,'N',1,1,2*NGVECTOR, &
                eVtoHar,FTOD_PW(1,NA,NI,NKI,NKQ,1,ISP1),2*NGVECTOR,FTOD_PW(1,NB,NJ,NKJ,NKQ,ncc,ISP2),&
                2*NGVECTOR,zero,TWOE4ORB(1,1),1)
#else
             CALL ZGEMM(trans,'N',1,1,NGVECTOR, &
                eVtoHar,FTOD_PW(1,NA,NI,NKI,NKQ,1,ISP1),NGVECTOR,FTOD_PW(1,NB,NJ,NKJ,NKQ,ncc,ISP2),&
                NGVECTOR,(0.0_q,0.0_q),TWOE4ORB(1,1),1)
#endif

             IF (ASSOCIATED(H)) then
#ifdef gammareal
                CALL DGEMM(trans,'N',1,1,NHVECTOR, &
                   eVtoHar,FTOD_OC(1,NA,NI,NKI,NKQ,1,ISP1),NHVECTOR,FTOD_OC(1,NB,NJ,NKJ,NKQ,2,ISP2),&
                   NHVECTOR,one,TWOE4ORB(1,1),1)
#else
                CALL ZGEMM(trans,'N',1,1,NHVECTOR, &
                   eVtoHar,FTOD_OC(1,NA,NI,NKI,NKQ,1,ISP1),NHVECTOR,FTOD_OC(1,NB,NJ,NKJ,NKQ,2,ISP2),&
                   NHVECTOR,one,TWOE4ORB(1,1),1)
#endif
             endif
             TWOE4ORB = TWOE4ORB/WDES_MOD%NKPTS
         else
             TWOE4ORB = zero
         end if

      END SUBROUTINE  CONSTRUCT_IJAB_one

      SUBROUTINE TEST_IJAB(W)
         use mkpoints
         IMPLICIT NONE
         type(wavespin) :: W
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,NB
         INTEGER :: I,J,A,IS,JS
         GDEF :: fock
         GDEF :: TE4O(WDES_MOD%NBANDS,WDES_MOD%NBANDS)
         
         fock=(0.0_q,0.0_q)
         TE4O(:,:)=(0.0_q,0.0_q)
         DO I=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS
            call SPLITINDEX(I,NKI,NI,IS)
            DO J=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS
!               DO NKA=1,WDES%NKPTS
                  call SPLITINDEX(J,NKJ,NJ,JS)
!                  A=b_k_index(1,NKJ)
                  IF (W%FERTOT(NI,NKI,1)>0.1_q) then
                     IF (W%FERTOT(NJ,NKJ,1)>0.1_q) then
                        CALL CONSTRUCT_IJAB(I,J,J,TE4O)
!                        DO NA=1,VBMAX
!                           DO NB=1,VBMAX
                              fock=fock+TE4O(NJ,NI)*(WDES_MOD%WTKPT(NKI))*(WDES_MOD%WTKPT(NKJ))
!                           ENDDO
!                        ENDDO
                     ENDIF
                  ENDIF
!               ENDDO
            ENDDO
         ENDDO
         write(*,*)'fock via ijab=',fock

      END SUBROUTINE TEST_IJAB

      SUBROUTINE TEST_IJAB_one(W,WDES)
         use mkpoints
         IMPLICIT NONE
         type(wavespin) :: W
         TYPE(wavedes) :: WDES
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,NB
         INTEGER :: I,J,A,ISP1,ISP2,ISP
         GDEF :: fock,hartree
         GDEF :: TE4O(WDES_MOD%NBANDS,WDES_MOD%NBANDS)

         fock=(0.0_q,0.0_q)
         TE4O(:,:)=(0.0_q,0.0_q)
         DO I=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
            call SPLITINDEX(I,NKI,NI,ISP1)
            DO J=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
                  call SPLITINDEX(J,NKJ,NJ,ISP2)
                  IF (W%FERTOT(NI,NKI,ISP)>0.1_q) then
                     IF (W%FERTOT(NJ,NKJ,ISP2)>0.1_q) then
                        IF (ISP1==ISP2) THEN
                           CALL CONSTRUCT_IJAB_one(I,J,J,I,TE4O)
                           fock=fock+TE4O(1,1)*(WDES_MOD%WTKPT(NKI))*(WDES_MOD%WTKPT(NKJ))
                        ENDIF
                        CALL CONSTRUCT_IJAB_one(I,J,I,J,TE4O)
                        hartree=hartree+TE4O(1,1)*(WDES_MOD%WTKPT(NKI))*(WDES_MOD%WTKPT(NKJ))
                     ENDIF
                  ENDIF
            ENDDO
         ENDDO
         write(*,*)'fock via ijab=',fock
         write(*,*)'hartree via ijab=',hartree*(2.0_q/WDES%ISPIN)

      END SUBROUTINE TEST_IJAB_one

      SUBROUTINE TEST_IJAB_one_MP2(W,WDES) 
         use mkpoints
         IMPLICIT NONE
         type(wavespin) :: W
         TYPE(wavedes) :: WDES
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,NB,NKB
         INTEGER :: I,J,A,B,ISP1,ISP2,ISP3,ISP4
         COMPLEX(q) :: emp2,fock,hartree
         GDEF :: TE4O(WDES_MOD%NBANDS,WDES_MOD%NBANDS),TE4O2

         fock=(0.0_q,0.0_q)
         TE4O(:,:)=zero

         emp2=(0.0_q,0.0_q)
         DO I=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
            call SPLITINDEX(I,NKI,NI,ISP1)
            DO J=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
               call SPLITINDEX(J,NKJ,NJ,ISP2)
               DO A=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
                  call SPLITINDEX(A,NKA,NA,ISP3)
                  if (ISP3 /= ISP1) cycle
                  DO B=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
                     call SPLITINDEX(B,NKB,NB,ISP4)
                     if (ISP4 /= ISP2) cycle
                     IF (W%FERTOT(NI,NKI,ISP1)>0.1_q) then
                        IF (W%FERTOT(NJ,NKJ,ISP2)>0.1_q) then
                        IF (W%FERTOT(NA,NKA,ISP1)<0.1_q) then
                        IF (W%FERTOT(NB,NKB,ISP2)<0.1_q) then
                           TE4O2=zero
                           IF (ISP1==ISP2) THEN
                              CALL CONSTRUCT_IJAB_one(I,J,B,A,TE4O)
                              TE4O2=TE4O(1,1)
                           ENDIF
                           CALL CONSTRUCT_IJAB_one(I,J,A,B,TE4O)
                           hartree=hartree+TE4O(1,1)*(WDES_MOD%WTKPT(NKI))*(WDES_MOD%WTKPT(NKJ))
                           emp2=emp2+(TE4O(1,1)*GCONJG((2.0_q,0.0_q)/(WDES%ISPIN)*TE4O(1,1)-TE4O2))/&
                               (REAL(W%CELTOT(NI,NKI,ISP1),KIND=q)+REAL(W%CELTOT(NJ,NKJ,ISP2),KIND=q)&
                                -REAL(W%CELTOT(NA,NKA,ISP1),KIND=q)-REAL(W%CELTOT(NB,NKB,ISP2),KIND=q))/eVtoHar
                        ENDIF
                        ENDIF
                        ENDIF
                     ENDIF
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         write(*,*)'mp2 energy=',emp2

      END SUBROUTINE TEST_IJAB_one_MP2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! This routine computes and writes the integrals to the FCIDUMP file
!
! N.B.: Only integrals with an absolute value greater than SCREENER are written
! to the FCIDUMP file. The energies in the FCIDUMP file have units of Hartree!
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE WRITE4INDEX2DISK(W,WDES)
         use mkpoints
         IMPLICIT NONE
         type(wavespin) :: W
         TYPE(wavedes) WDES
         INTEGER :: NKI,NKJ,NKA,NI,NJ,NA,NKQ,NB,NKB,K,NKK,NK
         INTEGER :: I,J,A,B,ISP1,ISP2,ISP3,ISP4,SP
         GDEF :: emp2,fock,hartree
         GDEF :: TE4O(WDES_MOD%NBANDS,WDES_MOD%NBANDS),TE4O2,FAC
         REAL(q) :: SCREENER,TMP
         INTEGER :: UNPAIRED
         REAL(q) :: LINKSTR(PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS)
         REAL(q) :: SORTLINKSTR(PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS)
         INTEGER :: SORTMAP(PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS)
         LOGICAL :: ex, CONSISTENT
         INTEGER :: nbstart, nastart
         integer :: propbitlen, sym_cell(3)

         UNPAIRED=0
         IF (WDES%ISPIN>1) THEN
            DO I=1,PROCS*WDES%NBANDS
                IF (W%FERTOT(I,1,1)>0.1_q) UNPAIRED=UNPAIRED+1
                IF (W%FERTOT(I,1,2)>0.1_q) UNPAIRED=UNPAIRED-1
            ENDDO
            UNPAIRED=ABS(UNPAIRED)
         ENDIF

         SCREENER=1.e-05
         SP=0
         FAC=one
         IF (WDES%ISPIN==2) THEN
            SP=1
            FAC=(2.0_q,0.0_q)
         ENDIF
 
         fock=(0.0_q,0.0_q)
         TE4O(:,:)=(0.0_q,0.0_q)
         emp2=(0.0_q,0.0_q)
         open(unit = 7,file = "FCIDUMP")
         write(7,*)"&FCI NORB= ",PROCS*WDES_MOD%NBANDS*WDES%NKPTS*WDES%ISPIN,",NELEC=",(VBMAX*2-UNPAIRED)*WDES_MOD%NKPTS,",MS2= ",UNPAIRED,","
         IF (WDES%ISPIN==2) THEN
             write(7,*)"UHF=TRUE"
         ENDIF
         call NECISymInfo(WDES%NKPTS, WDES%VKPT, sym_cell)
         propbitlen = 15
         write(7,*)" ORBSYM="
         DO I=1,PROCS*WDES%NBANDS*WDES%NKPTS*WDES%ISPIN
            call SPLITINDEX(I,NKI,NI,ISP1)
            write(7,"(I20,',')",advance="no") ComposeNECISymLabel(WDES%VKPT(:,NKI), sym_cell, propbitlen)
         ENDDO
         write(7,*)" ISYM=",product(sym_cell)
         write(7,*)"NPROP =", sym_cell
         write(7,*)"PROPBITLEN=", propbitlen
         write(7,*)"&END"

         DO I=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
            call SPLITINDEX(I,NKI,NI,ISP1)
            DO J=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
#ifdef gammareal
               nastart=I
#else
               nastart=1
#endif
               call SPLITINDEX(J,NKJ,NJ,ISP2)
               DO A=nastart,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
#ifdef gammareal
                  nbstart=J
#else
                  nbstart=1
#endif
                  call SPLITINDEX(A,NKA,NA,ISP3)
                  if (ISP3 /= ISP1) cycle
                  DO B=nbstart,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
                      call SPLITINDEX(B,NKB,NB,ISP4)
                      if (ISP4 /= ISP2) cycle
                      CALL CONSTRUCT_IJAB_one(I,J,A,B,TE4O)
                      IF (ABS(TE4O(1,1))>SCREENER) THEN
#ifdef gammareal                                                     
                          WRITE(7,'(1X,G20.12,4I3)') REAL(TE4O(1,1),KIND=q),I,A,J,B
#else
                          WRITE(7,*) (TE4O(1,1)),I,A,J,B
#endif                           
                      ENDIF

                  ENDDO
               ENDDO
            ENDDO
         ENDDO

         DO I=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
            call SPLITINDEX(I,NKI,NI,ISP1)
            DO J=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
               TE4O2=zero
               call SPLITINDEX(J,NKJ,NJ,ISP2)
               IF (NKJ /= NKI) cycle
               DO K=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES_MOD%ISPIN
                  call SPLITINDEX(K,NKK,NK,ISP3)
                  IF ((W%FERTOT(NK,NKK,ISP3)>0.1_q)) then
                      CALL CONSTRUCT_IJAB_one(I,K,J,K,TE4O)
#ifdef gammareal
                      TE4O2=TE4O2+TE4O(1,1)*((2.0_q)/WDES%ISPIN)
#else                  
                      TE4O2=TE4O2+TE4O(1,1)*((2.0_q,0.0_q)/WDES%ISPIN)
#endif
                      CALL CONSTRUCT_IJAB_one(I,K,K,J,TE4O)
                      TE4O2=TE4O2-TE4O(1,1)
                  ENDIF
               ENDDO
               IF (I==J) THEN
                   IF (ABS( W%CELTOT(NI,NKI,ISP1)*eVtoHar-TE4O2)>SCREENER) THEN
#ifdef gammareal
                       WRITE(7,'(1X,G20.12,4I3)') REAL(W%CELTOT(NI,NKI,ISP1)*eVtoHar-TE4O2,KIND=q),I,J,0,0
#else
                       WRITE(7,*) (W%CELTOT(NI,NKI,ISP1)*eVtoHar-TE4O2),I,J,0,0
#endif
                   ENDIF
               ELSE
                   IF ((ABS(TE4O2)>SCREENER)) THEN
#ifdef gammareal
                       WRITE(7,'(1X,G20.12,4I3)') REAL(FOCKM(NI,NJ,NKI,ISP1)*eVtoHar-TE4O2,KIND=q),I,J,0,0
#else
                       WRITE(7,*) (FOCKM(NI,NJ,NKI,ISP1)*eVtoHar-TE4O2),I,J,0,0
#endif
                   ENDIF
               ENDIF
                  
            ENDDO
         ENDDO
         
         DO B=1,PROCS*WDES_MOD%NBANDS*WDES_MOD%NKPTS*WDES%ISPIN
            call SPLITINDEX(B,NKB,NB,ISP2)
#ifdef gammareal            
            WRITE(7,'(1X,G20.12,4I3)') REAL(W%CELTOT(NB,NKB,ISP2)*eVtoHar,KIND=q),B,0,0,0
#else
            WRITE(7,*) (W%CELTOT(NB,NKB,ISP2)*eVtoHar),B,0,0,0
#endif
         ENDDO

         CLOSE(7)

      END SUBROUTINE WRITE4INDEX2DISK

      subroutine NECISymInfo(nkp, kpnts, cell)
          ! Find the necessary symmetry information based upon the wavevector
          ! grid that is required by NECI.
          ! In:
          !   nkp: number of k-points in the wavevector grid.
          !   k: wavevector in terms of the reciprocal lattice vectors of the
          !      primitive cell.
          ! Out:
          !   cell: dimensions of the symmetry cell.  This is the cell (in terms
          !      of the primitive cell) over which all wavevectors are periodic.
          !      (Note this is *not* the crystal cell).
          !      This is simply the period of the smallest non-zero component
          !      wavevector in each direction.
          !      cell is "NPROP" in the terms used by the NECI FCIDUMP file.
          integer, intent(in) :: nkp
          real(q), intent(in) :: kpnts(3,nkp)
          integer, intent(out) :: cell(3)
          integer :: i, ik, recip

          ! Start by assuming a Gamma-point calculation.
          cell = (/ 1, 1, 1 /)

          ! Find the smallest non-zero component of a wavevector in each
          ! direction and hence find the dimensions of the cell in which all
          ! wavevefunctions are periodic.
          do ik = 1, nkp
              do i = 1, 3
                  if (abs(kpnts(i,ik)) > 1.e-8) then
                      recip = abs(nint(1.0_q/kpnts(i,ik)))
                      if (recip > cell(i)) cell(i) = recip
                  end if
              end do
          end do

      end subroutine NECISymInfo

      function ComposeNECISymLabel(k, cell, propbitlen) result(label)
          ! Return the 64-bit integer label corresponding to the given
          ! wavevector for use in the NECI FCIDUMP file.
          ! In:
          !   k: wavevector in terms of the reciprocal lattice vectors of the
          !      primitive cell.
          !   cell: dimensions of the symmetry cell in terms of the primitive
          !      cell that corresponds to the wavevector mesh used.
          !      See NECISymInfo.
          !   propbitlen: the number of bits used to store each component of
          !      the wavevector in the 64-bit integer.  This must be the same
          !      for all wavevectors in a given calculation and must be less
          !      than 64/3.  A value of 15 is more than sufficient for all
          !      practical purposes.
          integer(8) :: label
          real(q), intent(in) :: k(3)
          integer, intent(in) :: cell(3)
          integer, intent(in) :: propbitlen
          integer :: kcell(3)
          integer(8) :: tmp
          ! Convert wavevector to be in terms of the symmetry cell.
          ! This is an integer quantity...
          ! Useful if the quantities are non-negative.  We can add an
          ! arbitrary reciprocal vector of the crystal cell if needed.
          kcell = modulo(nint(k*cell), cell)
          ! NECI stores the wavevector as a single integer according to
          !   label = \sum_i kcell(i)*(2**p)**(i-1)
          ! where p is propbitlen.
          tmp = kcell(3)
          label = kcell(1) + ishft(kcell(2), propbitlen) + ishft(tmp, 2*propbitlen)
      end function ComposeNECISymLabel

      SUBROUTINE INSULATOR_CHECK(W,WDES)
         IMPLICIT NONE
         type(wavespin) :: W
         TYPE(wavedes) WDES
         INTEGER :: NI,KI,ISP

         DO ISP=1,WDES%ISPIN
         DO NI=1,WDES_MOD%NBANDS
            DO KI=1,WDES_MOD%NKPTS
               IF ((W%FERTOT(NI,KI,ISP)>0.001_q) .and. (W%FERTOT(NI,KI,ISP)<0.001_q)) THEN
                  WRITE(*,*)"THERE SEEMS TO BE A PROBLEM WITH THE OCCUPATION NUMBERS!!!" 
                  WRITE(*,*) "BAND NUMBER:",NI,"IS SOMEWHERE BETWEEN EMPTY AND&
                  & OCCUPIED!!!!! THIS ROUTINE CAN'T DEAL WITH PARTIAL&
                  & OCCUPANCIES."
                  WRITE(*,*) "PLEASE CHECK THE HF GROUNDSTATE CALCULATION! "
                  WRITE(*,*) "SETTING SIGMA TO A SMALLER VALUE IN THE HF CALCULATION COULD HELP (SIGMA=0.000001)."
                  EXIT
               ENDIF
            ENDDO
         ENDDO
         ENDDO

      END SUBROUTINE INSULATOR_CHECK


      !***********************************************************************
      !This subroutine redistributes the fourier-transformed overlap integrals <i|G|a>
      !from the column-only process grid (CONTXT) to the quadratic 
      !process grid (CONTXT_GRID)
      !***********************************************************************

      SUBROUTINE REDISTRIBUTE_FTOD_GRID(WDES,KI,KQ,ISP,tmp_FTOD_PW,tmp_FTOD_OC,nbi_start,nbi_end)
         IMPLICIT NONE
         TYPE(wavedes) WDES
         INTEGER :: KI,KQ,ISP, nbi_start, nbi_end,KQSHIFTED
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
         desc_FTOD_PW_br(2) = contxt_cols         ! blacs context
         desc_FTOD_PW_br(3) = NGVECTOR    ! global number of rows
         desc_FTOD_PW_br(4) = (PROCS*WDES%NBANDS) ! global number of cols
         desc_FTOD_PW_br(5) = NGVECTOR     ! row block size
         desc_FTOD_PW_br(6) = 1             ! col block size
         desc_FTOD_PW_br(7) = 0              ! initial process row
         desc_FTOD_PW_br(8) = 0              ! initial process col
         desc_FTOD_PW_br(9) = MAX(1,(NGVECTOR)) ! leading dimension of local array
         !Distribute the fourier-transformed overlap integrals
         KQSHIFTED=KQ
         IF ((SHIFTED_KPOINTS)) KQSHIFTED=KQ-WDES%NKPTS/2
         
         DO NBI=nbi_start,nbi_end
            DO cc=1,ncc
               CALL PZGEMR2D(NGVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_PW(1,1,(NBI-nbi_start+1),cc),1,1,&
                desc_FTOD_PW_br,FTOD_PW(1,1,NBI,KI,KQSHIFTED,cc,ISP),1,1,desc_FTOD_PW,contxt_grid)
            ENDDO
         ENDDO
         
#ifdef gammareal         
         desc_FTOD_PW(9) = MAX(1,m_ desc_FTOD_PW(9)) ! leading dimension of local array
         desc_FTOD_PW(5) = m_ desc_FTOD_PW(5)
         desc_FTOD_PW(3) = m_ desc_FTOD_PW(3)
#else

#endif
         !ONE-center part RESPF_OC
           
         IF (ASSOCIATED(H)) THEN
         
            ! Prepare array descriptors for ScaLAPACK
         
            desc_FTOD_OC_br(1) = 1            ! descriptor type
            desc_FTOD_OC_br(2) = contxt_cols       ! blacs context
            desc_FTOD_OC_br(3) = NHVECTOR     ! global number of rows
            desc_FTOD_OC_br(4) = (PROCS*WDES%NBANDS) ! global number of cols
            desc_FTOD_OC_br(5) = NHVECTOR     ! row block size
            desc_FTOD_OC_br(6) = 1            ! col block size
            desc_FTOD_OC_br(7) = 0            ! initial process row
            desc_FTOD_OC_br(8) = 0            ! initial process col
            desc_FTOD_OC_br(9) = MAX(1,(NHVECTOR)) ! leading dimension of local array
      
            DO NBI=nbi_start,nbi_end
               DO cc=1,2
#ifdef gammareal
                  CALL PDGEMR2D(NHVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_OC(1,1,(NBI-nbi_start+1),cc),1,1,&
                   desc_FTOD_OC_br,FTOD_OC(1,1,NBI,KI,KQSHIFTED,cc,ISP),1,1,desc_FTOD_OC,contxt_grid)
#else
                  CALL PZGEMR2D(NHVECTOR,(PROCS*WDES%NBANDS),tmp_FTOD_OC(1,1,(NBI-nbi_start+1),cc),1,1,&
                   desc_FTOD_OC_br,FTOD_OC(1,1,NBI,KI,KQSHIFTED,cc,ISP),1,1,desc_FTOD_OC,contxt_grid)
#endif
               ENDDO
            ENDDO           

         ENDIF
         
      END SUBROUTINE REDISTRIBUTE_FTOD_GRID

      !***********************************************************************
      !This routine initializes a process grid which is used for the 
      !pre-calculation of the fourier-transformed overlap integrals <i|G|a>
      !This process grid has NPROCS(number of processors used) columns and 1 row.
      !The assigned context variable is called CONTXT_COLS.
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
         call BLACS_GET     (0, 0, CONTXT_COLS)
         call BLACS_GRIDINIT(CONTXT_COLS, 'R', NPROW, NPCOL)
         call BLACS_GRIDINFO(CONTXT_COLS, NPROW, NPCOL, MYROW, MYCOL)                 
        
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
         TWOE4ORBITAL_X=zero
      END SUBROUTINE SETUP_TWOE4ORBITAL
      
      !***********************************************************************
      !Allocate the arrays RESPF_PW and 
      !RESPF_OC in the block-cyclic data distribution
      !and setup their descriptors(desc_RESPF_PW and desc_RESPF_OC),
      !which are required by the scalaLAPACK routines
      !***********************************************************************
      
      SUBROUTINE SETUP_FTOD(WDES)
         IMPLICIT NONE
         TYPE(wavedes) WDES              
         INTEGER :: FTOD_PW_rows, FTOD_PW_cols,cc,FTOD_OC_rows, FTOD_OC_cols         

         call BLACS_GRIDINFO(CONTXT_GRID, NPROW, NPCOL, MYROW, MYCOL)
         !Blocking size for block-cyclic distribution of RESPF_PW        
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
         
         IF (SHIFTED_KPOINTS) then
            ALLOCATE(FTOD_PW(FTOD_PW_rows,FTOD_PW_cols,PROCS*WDES%NBANDS,WDES%NKPTS/2,WDES%NKPTS/2,ncc,WDES%ISPIN))
         else
            ALLOCATE(FTOD_PW(FTOD_PW_rows,FTOD_PW_cols,PROCS*WDES%NBANDS,WDES%NKPTS,WDES%NKPTS,ncc,WDES%ISPIN))
         endif
         IF (ASSOCIATED(H)) THEN         
            !Blocking size for block-cyclic distribution of RESPF_PW     
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
            
            IF (SHIFTED_KPOINTS) then
               ALLOCATE(FTOD_OC(FTOD_OC_rows,FTOD_OC_cols,PROCS*WDES%NBANDS,WDES%NKPTS/2,WDES%NKPTS/2,2,WDES%ISPIN))
            else
               ALLOCATE(FTOD_OC(FTOD_OC_rows,FTOD_OC_cols,PROCS*WDES%NBANDS,WDES%NKPTS,WDES%NKPTS,2,WDES%ISPIN))
            endif
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
      ! The CALC_EFOCK routine calculates the Fock exchange energy.
      ! This energy is not required for the LCCD calculations, but
      ! serve as a test, wether the FTOD are correct.
      !***********************************************************************

      Subroutine CALC_EFOCK(WDES,WGW,IO)
         use mkpoints
         use base
         IMPLICIT NONE
         TYPE(wavedes) WDES
         TYPE(wavedes) WGW
         TYPE (in_struct) IO
         INTEGER :: KQ,KI,KA,KJ,KB,KQ_,NI,NJ,NA,RNA,NB,RNB
         INTEGER :: TWOE4ORBITAL_ROWS, TWOE4ORBITAL_COLS
         COMPLEX(q), ALLOCATABLE :: EFOCK(:,:)
         TWOE4ORBITAL_ROWS = numroc(desc_TWOE4ORBITAL(3),desc_TWOE4ORBITAL(5),MYROW,0,NPROW)
         TWOE4ORBITAL_COLS = numroc(desc_TWOE4ORBITAL(4),desc_TWOE4ORBITAL(6),MYCOL,0,NPCOL)
         
         !kqloop: DO KQ=1,WDES%NKPTS
         ALLOCATE(EFOCK(PROCS*WDES%NBANDS,WDES%NKPTS))
         EFOCK=zero
            kiloop: DO KI=1,WDES%NKPTS
               IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KI)==0.00_q)) CYCLE
!                  KB=KI
                  DO NI=1,VBMAX!PROCS*WDES%NBANDS
                  kjloop: DO KJ=1,WDES%NKPTS
                     IF ((SHIFTED_KPOINTS) .and. (WDES%WTKPT(KJ)==0.00_q)) CYCLE
                     KA=KJ
                     KB=KI
                     KQ=KPOINT_IN_FULL_GRID(WDES%VKPT(:,KI)-WDES%VKPT(:,KJ),KPOINTS_FULL)
                     IF ((SHIFTED_KPOINTS)) KQ=KQ-WDES%NKPTS/2
                     DO NJ=1,VBMAX
                     
#ifdef gammareal
                        CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          m_ NGVECTOR,one, FTOD_PW(1,1,NI,KI,KQ,1,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NJ,KJ,KQ,ncc,1),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                        CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NGVECTOR,one, FTOD_PW(1,1,NI,KI,KQ,1,1),1,1,&
                          desc_FTOD_PW,FTOD_PW(1,1,NJ,KJ,KQ,ncc,1),1,1,&
                          desc_FTOD_PW,zero, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        IF (ASSOCIATED(H)) THEN
#ifdef gammareal
                           CALL PDGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,one, FTOD_OC(1,1,NI,KI,KQ,1,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NJ,KJ,KQ,2,1),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#else
                           CALL PZGEMM(trans,'n',(PROCS*WDES%NBANDS),(PROCS*WDES%NBANDS),&
                          NHVECTOR,one, FTOD_OC(1,1,NI,KI,KQ,1,1),1,1,&
                          desc_FTOD_OC,FTOD_OC(1,1,NJ,KJ,KQ,2,1),1,1,&
                          desc_FTOD_OC,one, TWOE4ORBITAL(1,1),1,1,desc_TWOE4ORBITAL)
#endif
                        ENDIF
                        TWOE4ORBITAL=GCONJG(TWOE4ORBITAL*KPOINTS_FULL%WTKPT(1))
                        DO NA=1,TWOE4ORBITAL_ROWS
                        CALL LOC2GLOB(NA,MYROW,desc_TWOE4ORBITAL(3),NPROW,desc_TWOE4ORBITAL(5),RNA)     
                        DO NB=1,TWOE4ORBITAL_COLS
                        CALL LOC2GLOB(NB,MYCOL,desc_TWOE4ORBITAL(4),NPCOL,desc_TWOE4ORBITAL(6),RNB)
                           IF ((KI==KB) .AND. (KJ==KA) .AND. (NI==RNB) .AND. (NJ==RNA)) THEN
                              EFOCK(NI,KI)=EFOCK(NI,KI)+TWOE4ORBITAL(NA,NB)
                           ENDIF

                        ENDDO
                        ENDDO
                     ENDDO !NBJ VBMAX
                  ENDDO kjloop
                  CALL M_sum_z(WGW%COMM_INTER, EFOCK(NI,KI), 1)
                  IF (IO%IU0>0) THEN
                     !write(IO%IU0,*)'ki,Ni,EFOCK',ki,ni,EFOCK(NI,KI)
                  ENDIF
                  
                  EFOCK_tot=EFOCK_tot+EFOCK(NI,KI)*KPOINTS_FULL%WTKPT(1)
                  ENDDO !NBI VBMAX
                  
           ENDDO kiloop  
         IF (IO%IU0>0) THEN
            write(IO%IU0,*)'Total Fock energy=',EFOCK_tot           
         ENDIF   

      END subroutine CALC_EFOCK
      
      
      subroutine consistency_test(WDES,W)
         USE constant
         USE full_kpoints
         USE mkpoints
         USE wave
         implicit none
         TYPE(wavedes) WDES
         TYPE (wavespin) W
         integer :: mni,mna,mki,mka,mkq,mkq_,mng
         integer :: mnj,mnb,mkj,mkb
         complex(q) :: testsum,testsum2
         
         
         testsum=zero
         testsum2=zero
         
         DO mki=1,WDES%NKPTS
            do mka=1,WDES%NKPTS
               DO mkj=1,WDES%NKPTS
               mkq=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,mkj)-W%WDES%VKPT(:,mka),KPOINTS_FULL)
               mkq_=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,mka)-W%WDES%VKPT(:,mkj),KPOINTS_FULL)
               !mkj=2
               mkb=KPOINT_IN_FULL_GRID(W%WDES%VKPT(:,mki)-W%WDES%VKPT(:,mkq),KPOINTS_FULL)
               
               do mna=VBMAX+1,PROCS*WDES%NBANDS
               do mnb=VBMAX+1,PROCS*WDES%NBANDS
                  do mni=1,VBMAX!,PROCS*WDES%NBANDS
                  do mnj=1,vbmax
                     do mng=1,ngvector
                        !testsum=testsum+conjg(FTOD_PW(mng,mna,mni,mki,mkq,1))-GCONJG(FTOD_PW(mng,mni,mna,mka,mkq_,2))
                        !testsum=testsum+conjg(FTOD_PW(mng,mna,mni,mki,mkq,1))*FTOD_PW(mng,mnb,mnj,mkj,mkq,2)*KPOINTS_FULL%WTKPT(1)
                        testsum=testsum+GCONJG(FTOD_PW(mng,mnb,mni,mki,mkq,1,1))*(FTOD_PW(mng,mnj,mna,mka,mkq,2,1))!*&
                           !(FTOD_PW(mng,mnb,mni,mki,mkq,1))*conjg(FTOD_PW(mng,mnj,mna,mka,mkq,2))!*KPOINTS_FULL%WTKPT(1)
                        !testsum2=testsum2+conjg(FTOD_PW(mng,mni,mna,mka,mkq,1))*FTOD_PW(mng,mnj,mnb,mkb,mkq,2)!*KPOINTS_FULL%WTKPT(1)
                        testsum2=testsum2+GCONJG(FTOD_PW(mng,mnj,mna,mka,mkq_,1,1))*(FTOD_PW(mng,mnb,mni,mki,mkq_,2,1))!*&
                           !(FTOD_PW(mng,mnj,mna,mka,mkq_,1))*conjg(FTOD_PW(mng,mnb,mni,mki,mkq_,2))!*KPOINTS_FULL%WTKPT(1)
                     enddo
                  enddo
                  enddo
               enddo
               enddo
               
               enddo
            enddo
         enddo
         write(*,*)'testsum=',testsum
         write(*,*)'testsum2=',testsum2
      
      end subroutine consistency_test

      
      subroutine CHECK_SHIFTED_KPOINTS(WDES,W,KPOINTS)
         USE constant
         USE full_kpoints
         USE mkpoints
         USE wave
         implicit NONE
         TYPE(wavedes) WDES
         TYPE(wavespin) W
         TYPE (kpoints_struct) KPOINTS
         integer :: MKI
         LOGICAL :: GAMMA_FOUND
         real(q) :: GAMMA_WEIGHT
         
         !GAMMA_FOUND=.false.
         !write(*,*)LSHIFT_KPOINTS
         SHIFTED_KPOINTS=.TRUE.
         do MKI=1,WDES%NKPTS
            if ( (ABS(W%WDES%VKPT(1,MKI))<0.001_q) .and. (ABS(W%WDES%VKPT(2,MKI))<0.001_q) .and. (ABS(W%WDES%VKPT(3,MKI))<0.001_q)) then
               GAMMA_FOUND=.true.
               GAMMA_WEIGHT=W%WDES%WTKPT(MKI)
            endif
         enddo
         !write(*,*)gamma_found,gamma_weight
         if ((GAMMA_FOUND) .and. (GAMMA_WEIGHT/=0.0_q)) THEN
            SHIFTED_KPOINTS=.FALSE.
         endif
         !CALL SETUP_KPOINTS_STATIC(KPOINTS)
         !CALL CHECK_FULL_KPOINTS ! all set up properly ?
         do MKI=1,WDES%NKPTS
            !write(*,*)LSHIFT_KPOINTS,W%WDES%WTKPT(MKI)
            !write(*,*) mki,',',W%WDES%VKPT(:,MKI),W%WDES%WTKPT(MKI)
         enddo
         
      end subroutine CHECK_SHIFTED_KPOINTS

     SUBROUTINE FOCKM_IN(IO,WDES,W)
         USE base
         IMPLICIT NONE
         TYPE (in_struct) IO
         TYPE(wavedes) WDES
         TYPE (wavespin) W
         INTEGER :: NB_TOT,NBI,IREC,NISP,NKPTS,KI,ISP
         COMPLEX(q), ALLOCATABLE :: FOCKM_LINE(:)
         LOGICAL :: CONSISTENT

         CONSISTENT=.TRUE.

         IF ((MYROW==0) .and. (MYCOL==0)) THEN

            OPEN(UNIT=19,FILE='FOCKM',ACCESS='DIRECT', &
                  FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=((3)*IO%ICMPLX*2))
            READ(19,REC=1)NISP,NKPTS,NB_TOT
            IF (WDES%ISPIN/=NISP) THEN
               WRITE(*,*)'ERROR: number of SPINORS in FOCKM file differs from&
               & settings in the INCAR file'
              RETURN
            ENDIF
            IF (WDES%NKPTS/=NKPTS) THEN
               WRITE(*,*)'ERROR: number of kpoints in FOCKM file differs from&
               & settings in the KPOINTS file'
               RETURN
            ENDIF
            IF (WDES%NB_TOT/=NB_TOT) THEN
               WRITE(*,*)'WARNING: number of bands in FOCKM :',NB_TOT,'.  &
               & Now using ',WDES%NB_TOT,' bands.'
            ENDIF

            CLOSE(19)
!            WRITE(*,*)'test',NB_TOT
            ALLOCATE(FOCKM_LINE(NB_TOT))
            OPEN(UNIT=19,FILE='FOCKM',ACCESS='DIRECT', &
                  FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=((NB_TOT+3)*IO%ICMPLX*2))
            IREC=2
            DO ISP=1,WDES%ISPIN
               DO KI=1,WDES%NKPTS
            DO NBI=1,NB_TOT
               READ(19,REC=IREC) FOCKM_LINE
!               WRITE(*,*)'NBI',NBI,FOCKM_LINE
               IREC=IREC+1
               IF (NBI<=WDES%NB_TOT) THEN
                  FOCKM(:,NBI,KI,ISP)=REAL(FOCKM_LINE(1:WDES%NB_TOT),KIND=q)
                  IF (REAL(W%CELTOT(NBI,KI,ISP),KIND=q)/=REAL(FOCKM(NBI,NBI,KI,ISP),KIND=q)) CONSISTENT=.FALSE.
               ENDIF
            ENDDO
               ENDDO
            ENDDO
            CLOSE(19)
         ENDIF

         CALLMPI (M_sum_d(WDES%COMM,FOCKM(1,1,1,1), WDES%NB_TOT*WDES%NB_TOT*WDES%NKPTS*WDES%ISPIN))


         IF (CONSISTENT) WRITE(*,*)"Eigenvalues in WAVECAR and FOCKM agree. You &
              & are probably working with NATURAL ORBITALS. Everything seems to be OK!"

         IF (.not. CONSISTENT) THEN
            WRITE(*,*)"Eigenvalues in WAVECAR and FOCKM DISAGREE!! You &
              &should remove FOCKM if you are working with canonical orbitals. &
              &Anyway ONLY eigenvalues from the &
              &WAVECAR file will be used."
              FOCKM=zero
         ENDIF

      END SUBROUTINE

!      * SUBROUTINE INVERT_KPOINT
!      This subroutine makes sure that the inversion symmetry \psi_ni,ki=\phi_ni,-ki^*
!      holds. it transforms all orbitals to real space, complex conjugates them
!      and becktransforms them to the reciprocal space with an inverted
!      k-point. it would be better to use the routines in mkpoints_full.F to
!      achieve the same goal more efficiently, but i'm busy writing my thesis and
!      this workaround won't be a bottle neck for the present applications.
!      !!!!!!!! PLEASE NOTE THAT ORBITALS AT THE GAMMA POINT OR THE BZ BOUNDARY
!      ARE NOT CHANGED IN THIS ROUTINE. !!!!!!!!!
      SUBROUTINE INVERT_KPOINT(WDES,WGW,W,IO)
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
         TYPE (in_struct) IO
         ! local variables
         TYPE(wavespin) WHF
         TYPE(wavedes1) WDESKI,WDESKI_INV
         TYPE(wavefun1), ALLOCATABLE :: WI(:), WI_INV(:)
         INTEGER KQ,KI,KA,KB,KI_IN_FULL_ORIG,KI_INV
         INTEGER ISP, N
         INTEGER NBI,NBA,NBAA, i, rnba
         REAL(q) :: SMALL
         LOGICAL BZ_GAMMA_OR_BOUNDARY

         SMALL=0.00000000001_q

         write(*,*)'invert orbitals'
         ALLOCATE(WI(1))
         ALLOCATE(WI_INV(1))
         WHF=W
         WHF%WDES => WDES_FOCK
         CALL SETWDES(WHF%WDES,WDESKI,0)
         CALL SETWDES(WHF%WDES,WDESKI_INV,0)
         CALL NEWWAV(WI(1),WDESKI,.TRUE.)
         CALL NEWWAV(WI_INV(1),WDESKI_INV,.TRUE.)
         DO ISP=1,WDES%ISPIN
         DO KI=1,WDES%NKPTS
            BZ_GAMMA_OR_BOUNDARY=.TRUE.
            DO N=1,3
               IF (.not. ((ABS(W%WDES%VKPT(N,KI))< 1E-12) .or. ((ABS(W%WDES%VKPT(N,KI)-0.5_q)< 1E-12)))) BZ_GAMMA_OR_BOUNDARY=.FALSE.
            ENDDO
            IF (BZ_GAMMA_OR_BOUNDARY) CYCLE
            IF ((W%WDES%VKPT(1,KI) > -SMALL)) THEN
               IF ((ABS(W%WDES%VKPT(1,KI)) < SMALL) .and. (ABS(W%WDES%VKPT(2,KI)) < SMALL) .and. (ABS(W%WDES%VKPT(3,KI)) < SMALL)) CYCLE
               KI_INV=KPOINT_IN_FULL_GRID(-W%WDES%VKPT(:,KI),KPOINTS_FULL)
               write(*,*)W%WDES%VKPT(:,KI)
               write(*,*)W%WDES%VKPT(:,KI_INV)
               CALL SETWDES(WHF%WDES,WDESKI,KI)
               CALL SETWDES(WHF%WDES,WDESKI_INV,KI_INV)
               DO NBI=1,WDES%NB_TOT
                  CALL W1_GATHER_GLB(WHF,NBI,NBI,ISP,WI)
                  DO NBA=1,WDES%NBANDS
                     CALL LOC2GLOB(NBA,MYCOL,PROCS*WDES%NBANDS,PROCS,1,rnba)
                     IF (rnba==NBI) THEN
                        WI_INV(1)%CR(:)=GCONJG(WI(1)%CR(:))
                        CALL FFTEXT(WDESKI_INV%NGVECTOR,WDESKI_INV%NINDPW, &
                          WI_INV(1)%CR(1),W%CW(1,NBI,KI_INV,ISP),WDESKI_INV%GRID,.FALSE.)
                        W%CW(:,NBI,KI_INV,ISP)=(W%CW(:,NBI,KI_INV,ISP))*(1.0_q/WDESKI%GRID%NPLWV)
                        W%CPROJ(:,NBI,KI_INV,ISP)=GCONJG(WI(1)%CPROJ(:))
                    ENDIF
                  ENDDO
               ENDDO
            ENDIF
         ENDDO
         ENDDO
         DEALLOCATE(WI)

      END SUBROUTINE
#endif
END MODULE FCIDUMP
