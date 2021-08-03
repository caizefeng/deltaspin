#include "symbol.inc"
!************************* SUBROUTINE CALC_NABIJ ***********************
! these routines were written by Juergen Furthmueller:
! you can download the required postprocessing routines from
! http://pc06.physik.uni-jena.de/furth/pub/
!
! directory VASP/optics
! please ceck the README and Makefile
!
! You might learn more about the implementation in the following
! article:
! B. Adolph, J. Furthmueller, and F. Bechstedt, PRB 63, 125108 (2001).
!
!***********************************************************************
!
! main driver routine for the calculation of matrix elements of the
! nabla operator ( --> momentum operator --> velocity operator ... )
!
!***********************************************************************

      SUBROUTINE CALC_NABIJ(NABIJ,W,WDES,P,KPOINTS,GRID_SOFT,LATT_CUR, &
                            IO,INFO,T_INFO,COMM,IU0,IU)
      USE prec
      USE base
      USE constant
      USE lattice
      USE poscar
      USE pseudo
      USE mkpoints
      USE mgrid
      USE wave
      
      USE mpimy

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavespin)       W
      TYPE (wavedes)        WDES
      TYPE (potcar)         P
      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d)        GRID_SOFT
      TYPE (latt)           LATT_CUR
      TYPE (in_struct)      IO
      TYPE (info_struct)    INFO
      TYPE (type_info)      T_INFO
      TYPE (communic)       COMM

      GDEF  NABIJ(WDES%NB_TOT,WDES%NB_TOT)
      CHARACTER (1) CHARAC
      LOGICAL LDUM


#ifdef MPI
      IONODE  = WDES%COMM%IONODE
      NODE_ME = WDES%COMM%NODE_ME
      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('CALC_NABIJ: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif

#if defined(CRAY_MPP) && defined(MPI)
!-----------------------------------------------------------------------
! read in required parameters
!-----------------------------------------------------------------------
! CRAY C90 fails to read INCAR, if all nodes try to read it at the same time
      DO I=1,COMM%NCPU
      IF (COMM%NODE_ME == I) THEN
#endif
! if we cannot do all k-points specify the true number of k-points to be done
      NKOPT=KPOINTS%NKPTS
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NKOPT','=','#',';','I', &
     &            NKOUT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                  ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''NKOPT'' from file INCAR.'
         GOTO 8734         ! leave this block, no optics data written
      ENDIF
      CALL XML_INCAR('NKOPT','I',NKOUT,RDUM,CDUM,LDUM,CHARAC,N)

! cannot be smaller than NKPTS - readjust (without warning) if invalid input
      NKOPT=MAX(NKOPT,KPOINTS%NKPTS)
! if we cannot do all k-points specify the current k-point counter offset
      NKOFF=0
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NKOFFOPT','=','#',';','I', &
     &              NKOFF,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
           WRITE(IU0,*)'Error reading item ''NKOFFOPT'' from file INCAR.'
         GOTO 8734         ! leave this block, no optics data written
      ENDIF

      CALL XML_INCAR('NKOFFOPT','I',NKOFF,RDUM,CDUM,LDUM,CHARAC,N)

! must be a positive number and smaller than the number of k-points to be done
! minus current number of k-points - readjust (without warning) if invalid input
      NKOFF=MAX(NKOFF,0)
      NKOFF=MIN(NKOFF,NKOPT-KPOINTS%NKPTS)
! number of valence bands written on file OPTIC
      NBVAL=(NINT(INFO%NELECT)+1)/2
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NBVALOPT','=','#',';','I', &
     &            NBVAL,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                  ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
           WRITE(IU0,*)'Error reading item ''NBVALOPT'' from file INCAR.'
         GOTO 8734         ! leave this block, no optics data written
      ENDIF

      CALL XML_INCAR('NBVALOPT','I',NBVAL,RDUM,CDUM,LDUM,CHARAC,N)

! must be larger than zero and smaller or equal to the total number of bands
! - readjust (without warning) if invalid input
      NBVAL=MIN(NBVAL,WDES%NB_TOT)
      NBVAL=MAX(NBVAL,1)
! number of conduction bands written of file OPTIC
      NBCON=WDES%NB_TOT-NBVAL
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NBCONOPT','=','#',';','I', &
     &            NBCON,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                  ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
           WRITE(IU0,*)'Error reading item ''NBCONOPT'' from file INCAR.'
         GOTO 8734         ! leave this block, no optics data written
      ENDIF

      CALL XML_INCAR('NBCONOPT','I',NBCON,RDUM,CDUM,LDUM,CHARAC,N)

! must be larger than zero and smaller or equal to the total number of bands
! - readjust (without warning) if invalid input
      NBCON=MIN(NBCON,WDES%NB_TOT)
      NBCON=MAX(NBCON,1)

#if defined(CRAY_MPP) && defined(MPI)
      ENDIF
      CALL MPI_barrier( COMM%MPI_COMM, ierror )
      ENDDO
#endif
!-----------------------------------------------------------------------
! open files
!-----------------------------------------------------------------------

! we write with direct access, unformatted -> set the record length for OPTIC
      IRECLO=(MAX(NBVAL,2)+1)*IO%ICMPLX
! open OPTIC, write first record containing all data determining the layout
      io_begin
      IF (IU>=0) THEN
        IF (IO%LOPEN) THEN
          OPEN(UNIT=IU,FILE='OPTIC',ACCESS='DIRECT', &
               FORM='UNFORMATTED',RECL=IRECLO)
        ELSE
          OPEN(UNIT=IU,ACCESS='DIRECT',FORM='UNFORMATTED',RECL=IRECLO)
        ENDIF
        CALL OUTOPT_HEAD(WDES%NB_TOT,NKOPT,INFO%ISPIN,NBVAL,NBCON,IRECLO,IU)
      ENDIF
      io_end
!-----------------------------------------------------------------------
! calculate < phi_i | nabla | phi_j >
!-----------------------------------------------------------------------

! loop over all (current) k-points, spin component and cartesian directions
      DO NK=1,KPOINTS%NKPTS
       DO ISP=1,INFO%ISPIN
        DO IDIR=1,3
! get the matrix element of the nabla-operator in eV/Angstroem units
         CALL NABIJ_SOFT(NABIJ,IDIR,NK,ISP,W,WDES,GRID_SOFT,LATT_CUR)
         CALL NABIJ_AUG_ADD(NABIJ,IDIR,NK,ISP,W,WDES,P,T_INFO)

         NB_TOT=WDES%NB_TOT

         CALLMPI( M_sum_g(WDES%COMM,NABIJ(1,1),NB_TOT*NB_TOT))

         SCALE_NABIJ_TO_VIJ=AUTOA    ! hbar/m_e * 1/length in whatever unit
         NABIJ=NABIJ*SCALE_NABIJ_TO_VIJ   ! rescaling to bohr radii/Hartree
! write OPTIC
         io_begin
         IF (IU>=0) THEN
           CALL OUTOPT(NABIJ,IDIR,NK,ISP,W%CELTOT,W%FERTOT,NB_TOT, &
                       INFO%ISPIN,KPOINTS%NKPTS,WDES%VKPT, &
                       WDES%WTKPT,IRECLO,NBVAL,NBCON,NKOPT,NKOFF,IU)
         ENDIF
         io_end

        ENDDO    ! IDIR
       ENDDO     ! ISP
      ENDDO      ! NK

      io_begin
      IF (IU>=0) CLOSE(IU)
      io_end

      RETURN
! if we jumped to this label something was going wrong with the input from INCAR
 8734 CONTINUE
      IF (IU0>=0)  WRITE(IU0,*) 'No optics data calculated.'
      RETURN
      END


!************************* SUBROUTINE NABIJ_SOFT_ **********************
!
! calculates < Psi_i | \nabla | Psi_j > for the non-normconserving
! wave functions stored in CPTWFP, augmentation corrections elsewhere
! WARNING: no attempt made to optimize anything for any architecture!
!   I-loop to be split in blocks, NB- and NBP-loops and partial I-block
!   has to be replaced by ZGEMM/CGEMM calls (after setting work array
!   containing Gi*CPTWFP(i,NB) ...) --> expected to be rather SLOW now
!
!***********************************************************************

      SUBROUTINE NABIJ_SOFT_(NABIJ,IDIR,NK,ISP,W,WDES,GRID,LATT_CUR)
      USE prec
      USE constant
      USE lattice
      USE poscar
      USE mgrid
      USE wave

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR

      GDEF       NABIJ(WDES%NB_TOT,WDES%NB_TOT)
      GDEF       CSUM

      NABIJ = zero

      DO I  =1,WDES%NPLWKP(NK)
       G1=WDES%IGX(I,NK)+WDES%VKPT(1,NK)
       G2=WDES%IGY(I,NK)+WDES%VKPT(2,NK)
       G3=WDES%IGZ(I,NK)+WDES%VKPT(3,NK)
       GC=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))*TPI

       DO NB =1,WDES%NB_TOT
       DO NBP=1,WDES%NB_TOT
        CSUM=W%CPTWFP(I,NB,NK,ISP)*GC*CONJG(W%CPTWFP(I,NBP,NK,ISP))*(0._q,1._q)
        NABIJ(NBP,NB)=NABIJ(NBP,NB)+CSUM
       ENDDO
       ENDDO

      ENDDO

      RETURN
      END


!************************* SUBROUTINE NABIJ_SOFT ***********************
!
! calculates < Psi_i | \nabla | Psi_j > for the non-normconserving
! wave functions stored in CPTWFP, augmentation corrections elsewhere
!
!***********************************************************************

      SUBROUTINE NABIJ_SOFT(NABIJ,IDIR,NK,ISP,W,WDES,GRID,LATT_CUR)
      USE prec
      USE constant
      USE lattice
      USE poscar
      USE mgrid
      USE wave
      USE dfast

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR

      GDEF       NABIJ(WDES%NB_TOT,WDES%NB_TOT)
      COMPLEX(q),ALLOCATABLE :: CBLOCK(:,:),GC(:)

      ALLOCATE(CBLOCK(NBLK,WDES%NB_TOT),GC(NBLK))

      NABIJ = zero

      NB_TOT=WDES%NB_TOT
      NPLDIM=WDES%NRPLWV
      NPL=WDES%NPLWKP(NK)

      block: DO IBLOCK=0,NPL-1,NBLK
       ILENPL=MIN(NBLK,NPL-IBLOCK)
       IADDPL=MIN(IBLOCK,NPL-1)
       ILENPL=MAX(ILENPL,0)

       DO I=1,ILENPL
        G1=WDES%IGX(I+IADDPL,NK)+WDES%VKPT(1,NK)
        G2=WDES%IGY(I+IADDPL,NK)+WDES%VKPT(2,NK)
        G3=WDES%IGZ(I+IADDPL,NK)+WDES%VKPT(3,NK)
        GC(I)=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))*TPI
       ENDDO
       DO NB=1,NB_TOT
        DO I =1,ILENPL
         CBLOCK(I,NB)=W%CPTWFP(I+IADDPL,NB,NK,ISP)*GC(I)*(0._q,-1._q)
        ENDDO
       ENDDO

       CALL GGEMM(trans, 'N', NB_TOT, NB_TOT, m_ ILENPL, one, &
                  CBLOCK(1,1), m_ NBLK, W%CPTWFP(IADDPL+1,1,NK,ISP), &
                  m_ NPLDIM, one, NABIJ(1,1), NB_TOT)

      ENDDO block

      DEALLOCATE(CBLOCK,GC)

      RETURN
      END


!********************* SUBROUTINE NABIJ_AUG_ADD ************************
!
! add augmentation corrections to matrix elements of \nabla
! WARNING: this routine is not optimized at all :-)
!
!***********************************************************************

      SUBROUTINE NABIJ_AUG_ADD(NABIJ,IDIR,NK,ISP,W,WDES,P,T_INFO)
      USE prec
      USE poscar
      USE pseudo
      USE wave
      

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES

      GDEF       NABIJ(WDES%NB_TOT,WDES%NB_TOT),CSUM

!=======================================================================
! calculate augmentation correction to matrix elements
!=======================================================================

      NIS=1
      LMBASE =0

      typ: DO NT=1,T_INFO%NTYP
      ion: DO NI=NIS,T_INFO%NITYP(NT)+NIS-1

      NIP=NI_LOCAL(NI, WDES%COMM_INB)  ! local storage index
      IF (NIP==0) CYCLE ion            ! projected wavefunction not on local node

      DO LM =1,P(NT)%LMMAX
      DO LMP=1,P(NT)%LMMAX

       DO NBP=1,WDES%NB_TOT
       DO NB =1,WDES%NB_TOT
         CSUM=GCONJG(W%CPROJ(LMBASE+LM,NB,NK,ISP))* &
              P(NT)%NABLA(IDIR,LM,LMP)* &
              W%CPROJ(LMBASE+LMP,NBP,NK,ISP)
         NABIJ(NB,NBP)=NABIJ(NB,NBP)+CSUM
       ENDDO
       ENDDO

      ENDDO
      ENDDO

      LMBASE=LMBASE+P(NT)%LMMAX

      ENDDO ion
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO typ

      RETURN
      END SUBROUTINE

!***********************************************************************
!
! calculate the matrix
! n_ij= < psi_i | nabla | psi_j > - < tilde psi_i | nabla | tilde psi_j >
!
! where psi_i are the AE partial waves
! and   tilde psi_i the PS partial waves
!
! this subroutine is called once from main.F after the PAW datasets 
! have been read in 
!
!***********************************************************************

      SUBROUTINE SET_NABIJ_AUG(P,NTYP)
      USE prec
      USE constant
      USE radial
      USE pseudo

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (potcar),TARGET :: P(NTYP)
      TYPE (rgrid),POINTER :: R

! calculates the atomic matrix elements of -i \nabla for the all-electron
! and pseudo wave functions to be used later in routine NABIJ_AUG_ADD

      typ: DO NT=1,NTYP
      IF (.NOT. ASSOCIATED(P(NT)%QPAW)) CYCLE

! intel efc compiler workaround (required for 6.X versions)
!     P(NT)%NABLA=0._q   ! old command
      DO IDIR=1,3
      DO LL=1,P(NT)%LMMAX
      DO MP=1,P(NT)%LMMAX
         P(NT)%NABLA(IDIR,LL,MP)=0._q
      ENDDO
      ENDDO
      ENDDO

      R => P(NT)%R              ! grid for atom type NT
      CALL  RAD_ALIGN(R,R%RMAX) ! reallign RMAX with grid

      LM   = 0
      llloop: DO LL =1,P(NT)%LMAX       ! loop over all l and energy channels
       L=P(NT)%LPS(LL)                  ! l quantum number
       mloop: DO M=-L,L                 ! loop over all quantum numbers m
        LM=LM+1                         ! increment compound index LM
! the same for LP, MP, LMP
        LMP  = 0
        llploop: DO LLP=1,P(NT)%LMAX
         LP=P(NT)%LPS(LLP)
         mploop: DO MP=-LP,LP
          LMP=LMP+1
! add up the corresponding contributions
          CALL NABIJ_RADIAL(R,P(NT)%WPS(1,LL),P(NT)%WPS(1,LLP), &
                            P(NT)%NABLA(1,LM,LMP),L,LP,M,MP,-1._q)
          CALL NABIJ_RADIAL(R,P(NT)%WAE(1,LL),P(NT)%WAE(1,LLP), &
                            P(NT)%NABLA(1,LM,LMP),L,LP,M,MP, 1._q)
         ENDDO mploop
        ENDDO llploop

       ENDDO mloop
      ENDDO llloop

      ENDDO typ
      RETURN
      END


      SUBROUTINE NABIJ_RADIAL(R,WAVE,WAVEP,NABLA,L,LP,M,MP,SCALE)
      USE prec
      USE constant
      USE radial

! calculates atomic matrix elements of the nabla operator. Warning: it is NOT
! a hermitian operator -> matrix will be asymmetric! Our convention shall be
! that the prime quantities WAVEP(LP,MP) are the quantities RIGHT of nabla.

      IMPLICIT NONE

! global variables:
!     R                   input           grid layout/other information
!     WAVE,WAVEP          input           radial parts of atomic wave functions
!     NABLA               input/output    matrix element of nabla operator
!     L,LP,M,MP           input           momentum quantum numbers
!     SCALE               input           scaling factor (+1 or -1)
! local variables:
!     ANGLE*RSUM          atomic matrix element of nabla operator to be added
!                         with angular part ANGLE, radial part RSUM
!     ALLOWED             flag set if transition allowed (else quick return)
!     ONEBYR_SCALE        there are always two types of terms for the radial
!                         part: some with wave(r)/r and one with d/dr wave(r);
!                         this is the scaling factor for the first part ...
!     TMP,TMP2            work arrays for grid operations (on the radial grid)
!     N                   loop counter

      TYPE (rgrid) R

      LOGICAL ALLOWED
      INTEGER L,M,LP,MP,N
      REAL(q) WAVE(R%NMAX),WAVEP(R%NMAX),TMP(R%NMAX),TMP2(R%NMAX)
      REAL(q) NABLA(3),ONEBYR_SCALE,SCALE,RSUM,ANGLE(3)

      RSUM=0._q
      ANGLE=0._q
      ONEBYR_SCALE=0._q
      ALLOWED=.FALSE.

! performs all operations on the radial grid needed to calculate atomic
! matrix elements of the form rsum = < phi_lm | \nabla | phi_l'm' >

! all nonzero angular parts hard coded and selected by if statements
      IF ((L>2).OR.(LP>2)) RETURN                ! we cannot/do not handle l>2
      IF ((L==(LP+1)).OR.(L==(LP-1))) THEN       ! it must hold Delta L = +/-1
        IF ((L==1).AND.(LP==0)) THEN             ! ps-transitions
          ALLOWED=.TRUE.
          IF (M==-1) ANGLE(2)=1._q/sqrt(3._q)
          IF (M== 0) ANGLE(3)=1._q/sqrt(3._q)
          IF (M== 1) ANGLE(1)=1._q/sqrt(3._q)
! radial part is a term < psi_p(r) | d/dr psi_s(r) >  (see below)
          ONEBYR_SCALE=-1._q ! we must always reduce the factor by 1 (see below)
        ENDIF
        IF ((L==0).AND.(LP==1)) THEN             ! sp-transitions
          ALLOWED=.TRUE.
          IF (MP==-1) ANGLE(2)=1._q/sqrt(3._q)
          IF (MP== 0) ANGLE(3)=1._q/sqrt(3._q)
          IF (MP== 1) ANGLE(1)=1._q/sqrt(3._q)
! radial part is a term < psi_s(r) | 2*psi_p(r)/r + d/dr psi_p(r) >  (see below)
          ONEBYR_SCALE= 1._q ! we must always reduce the factor by 1 (see below)
        ENDIF
        IF ((L==2).AND.(LP==1)) THEN             ! dp-transitions
          IF (MP==-1) THEN
            IF (M/= 1) ALLOWED=.TRUE.
            IF (M==-2) ANGLE(1)= 1._q/SQRT(5._q)
            IF (M==-1) ANGLE(3)= 1._q/SQRT(5._q)
            IF (M== 0) ANGLE(2)=-1._q/SQRT(15._q)
            IF (M== 2) ANGLE(2)=-1._q/SQRT(5._q)
          ENDIF
          IF (MP== 0) THEN
            IF (ABS(M)/=2) ALLOWED=.TRUE.
            IF (M==-1) ANGLE(2)= 1._q/SQRT(5._q)
            IF (M== 0) ANGLE(3)= 2._q/SQRT(15._q)
            IF (M== 1) ANGLE(1)= 1._q/SQRT(5._q)
          ENDIF
          IF (MP== 1) THEN
            IF (M/=-1) ALLOWED=.TRUE.
            IF (M==-2) ANGLE(2)= 1._q/SQRT(5._q)
            IF (M== 0) ANGLE(1)=-1._q/SQRT(15._q)
            IF (M== 1) ANGLE(3)= 1._q/SQRT(5._q)
            IF (M== 2) ANGLE(1)= 1._q/SQRT(5._q)
          ENDIF
! radial part is a term < psi_d(r) | -psi_p(r)/r + d/dr psi_p(r) >  (see below)
          ONEBYR_SCALE=-2._q ! we must always reduce the factor by 1 (see below)
        ENDIF
        IF ((L==1).AND.(LP==2)) THEN             ! pd-transitions
          IF (MP==-2) THEN
            IF (M/=0) ALLOWED=.TRUE.
            IF (M==-1) ANGLE(1)= 1._q/SQRT(5._q)
            IF (M== 1) ANGLE(2)= 1._q/SQRT(5._q)
          ENDIF
          IF (MP==-1) THEN
            IF (M/=1) ALLOWED=.TRUE.
            IF (M==-1) ANGLE(3)= 1._q/SQRT(5._q)
            IF (M== 0) ANGLE(2)= 1._q/SQRT(5._q)
          ENDIF
          IF (MP== 0) THEN
            ALLOWED=.TRUE.
            IF (M==-1) ANGLE(2)=-1._q/SQRT(15._q)
            IF (M== 0) ANGLE(3)= 2._q/sqrt(15._q)
            IF (M== 1) ANGLE(1)=-1._q/SQRT(15._q)
          ENDIF
          IF (MP== 1) THEN
            IF (M/=-1) ALLOWED=.TRUE.
            IF (M== 0) ANGLE(1)= 1._q/SQRT(5._q)
            IF (M== 1) ANGLE(3)= 1._q/SQRT(5._q)
          ENDIF
          IF (MP== 2) THEN
            IF (M/=0) ALLOWED=.TRUE.
            IF (M==-1) ANGLE(2)=-1._q/SQRT(5._q)
            IF (M== 1) ANGLE(1)= 1._q/SQRT(5._q)
          ENDIF
! radial part is a term < psi_p(r) | 3*psi_d(r)/r + d/dr psi_d(r) >  (see below)
          ONEBYR_SCALE= 2._q ! we must always reduce the factor by 1 (see below)
        ENDIF
      ENDIF

! no allowed transition (angular parts are zero)
      IF (.NOT.ALLOWED) RETURN

! radial parts of the matrix elements

! We use a simple trick: since r*psi(r) is stored in WAVEP we calculate the
! quantity d/dr [r*psi(r)] = r * [d/dr psi(r) + psi(r)/r] and subtract then
! psi(r) = WAVEP(r)/r in order to yield  r * d/dr [1/r * (r*psi(r))] -> the
! subtraction can be done by subtraction of one from ONEBYR_SCALE (therefore we
! have defined a ONEBYR_SCALE already reduced by one above ...); since we have
! always to remultiply [d/dr psi(r) + ONEBYR_SCALE*psi(r)/r] with r in order
! to obtain the correct r**2 prefactor (where one factor is contained in WAVE)
! for the radial integral r**2 dr psi(r)*[d/dr psi(r) + ONEBYR_SCALE*psi(r)/r]
! -> the expression above automatically containing this factor r is what we need

      DO N=1,R%NMAX
        TMP2(N)=WAVEP(N)
      ENDDO
      CALL GRAD_(R,TMP2,TMP)                                 ! d/dr [r*psi(r)]
      DO N=1,R%NMAX
        TMP(N)=WAVE(N)*(TMP(N)+ONEBYR_SCALE*WAVEP(N)/R%R(N)) ! add psi(r)/r term
      ENDDO

      CALL SIMPI(R,TMP,RSUM)      ! integrate from 0 to RMAX

! add atomic matrix element (scaled with SCALE) to existing matrix element
      NABLA=NABLA+ANGLE*(SCALE*RSUM)
      RETURN
      END

!***********************************************************************
!
! write file
!
!***********************************************************************

      SUBROUTINE OUTOPT_HEAD(NB_TOT,NKPTS,ISPIN,NBVAL,NBCON,IRECL,IU)
      USE prec

      IMPLICIT NONE

! do not use INTEGER but REAL(q) format for unformatted output on all machines!
! -> increases exchangeability of file OPTIC across most IEEE platforms very
! significantly (apart from the remaining "little-endian/big-endian problem")
      INTEGER  IRECL, NB_TOT, NBVAL, NBCON, NKPTS, ISPIN, IU
      REAL(q) RIRECL,RNB_TOT,RNBVAL,RNBCON,RNKPTS,RISPIN
      RIRECL  = IRECL         ! use  IRECL  = NINT(RIRECL)   after input
      RNB_TOT = NB_TOT        ! use  NB_TOT = NINT(RNB_TOT)  after input
      RNBVAL  = NBVAL         ! use  NBVAL  = NINT(RNBVAL)   after input
      RNBCON  = NBCON         ! use  NBCON  = NINT(RNBCON)   after input
      RNKPTS  = NKPTS         ! use  NKPTS  = NINT(RNKPTS)   after input
      RISPIN  = ISPIN         ! use  ISPIN  = NINT(RISPIN)   after input
! write out header of file OPTIC on I/O unit IU
      WRITE(IU,REC=1) RIRECL,RNB_TOT,RNBVAL,RNBCON,RNKPTS,RISPIN

      RETURN
      END


      SUBROUTINE OUTOPT(NABIJ,IDIR,NK,ISP,CELEN,FERWE,NB_TOT,ISPIN, &
                        NKPTS,VKPT,WTKPT,IRECL,NBVAL,NBCON,NKOPT,NKOFF,IU)
      USE prec
      USE constant

      IMPLICIT REAL(q) (A-H,O-Z)

      GDEF         :: NABIJ(NB_TOT,NB_TOT)
      COMPLEX(q)   :: CELEN(NB_TOT,NKPTS,ISPIN)
      REAL(q)      :: VKPT(3,NKPTS),WTKPT(NKPTS),FERWE(NB_TOT,NKPTS,ISPIN)
      REAL(q)      :: ETMP(NB_TOT),FTMP(NB_TOT),EIG,FEIG
      INTEGER,SAVE :: NKOLD=0

! write out NABIJ on I/O unit IU

! if all is correctly coded the (complicated) layout of the file is:
!
!   REC=1:        header with control output for consistency check (NBANDS etc.)
!
!   REC=nk+1:     vkpt(1,nk),vkpt(2,nk),vkpt(3,nk),wtkpt(nk)   as control output
!
!   REC=(is+1)*nkpts+nk+1:
!                 eigen(n,nk,is),ferwe(n,nk,is)  (for n=1,nbval with is=1,ispin)
!
!   REC=(ispin+1)*nkpts+1+idir+3*(isp-1)+3*ispin*(nk-1)+3*ispin*nkpts*(n-1):
!                 eigen(nn,nk,isp),ferwe(nn,nk,isp),nabla(np,idir,isp,nk,nn)
!                          (for np=1,nbval with nn=nb_tot-nbcon+n and n=1,nbcon)

      NKR=NK+NKOFF
      IF (NK/=NKOLD) THEN
        WRITE(IU,REC=NKR+1) (VKPT(I,NK),I=1,3),WTKPT(NK)
        DO IS=1,ISPIN
         DO N=1,NBVAL
! use the unit you like
!          ETMP(N)=0.5_q*REAL(CELEN(N,NK,IS),KIND=q)/RYTOEV
          ETMP(N)=REAL(CELEN(N,NK,IS),KIND=q)
          FTMP(N)=(3-ISPIN)*FERWE(N,NK,IS)
         ENDDO
         WRITE(IU,REC=IS*NKOPT+NKR+1) (ETMP(N),FTMP(N),N=1,NBVAL)
        ENDDO
        NKOLD=NK
      ENDIF
      band: DO NN=1,NBCON
       N=NB_TOT-NBCON+NN
       IREC=(1+ISPIN)*NKOPT+1+IDIR+3*(ISP-1)+3*ISPIN*(NKR-1)+3*ISPIN*NKOPT*(NN-1)
! use the unit you like
!       EIG =0.5_q*REAL(CELEN(N,NK,ISP),KIND=q)/RYTOEV
       EIG =REAL(CELEN(N,NK,ISP),KIND=q)
       FEIG=(3-ISPIN)*FERWE(N,NK,ISP)
       WRITE(IU,REC=IREC) EIG,FEIG,(NABIJ(NP,N)*(0._q,-1._q),NP=1,NBVAL)
      ENDDO band

      RETURN
      END
