!#define debug
#include "symbol.inc"
      MODULE steep
      USE prec
      CONTAINS
!************************ SUBROUTINE EDSTEP*****************************
! RCS:  $Id: steep.F,v 1.3 2002/08/14 13:59:43 kresse Exp $
!
! this subroutine performes a minimizition of < \phi | H | \phi >
! using a sequential (i.e. band by band) algorithm
! the scheme used depends on INFO%IALGO:
!  IALGO 5 steepest descent
!  IALGO 6 conjugated gradient
!  IALGO 7 preconditioned steepest descent
!  IALGO 8 preconditioned conjugated gradient
!  IALGO 0 preconditioned conjugated gradient (Jacobi like precond)
!
! before or after EDSTEP a subspace-diagonalisation should be performed
!
!  INFO%.... controls the detailed behaviour of the minimization
!  WEIMIN  treshhold for high-quality eigenvalue minimisation
!    is the fermiweight of a band < WEIMIN, the eigenvalue
!    minimisation will break after a maximum of two iterations
!  EBREAK  absolut break condition
!    intra-band minimisation is stopped if DE is < EBREAK
!  DEPER   intra-band break condition (see below)
!  return values:
!  RMS     norm of error in eigenfunction at beginning of this routine
!  ICOUEV  number of intraband evalue minimisations
!  DESUM   change in bandstructure enrgy
!
!***********************************************************************

      SUBROUTINE EDSTEP(GRID,INFO,LATT_CUR,NONLR_S,NONL_S,W,WDES, &
        LMDIM,CDIJ,CQIJ, RMS,DESUM, ICOUEV, SV,IU6,IU0)
      USE prec
      USE wave
      USE wave_high
      USE base
      USE lattice
      USE mpimy
      USE mgrid
      USE fock
      USE pead
      USE nonl_high
      USE hamil
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (info_struct) INFO
      TYPE (latt)        LATT_CUR
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES

      RGRID   SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ) ! local potential
      OVERLAP CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ),CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)

!----- local work arrays
      TYPE (wavedes1)    WDES1         ! descriptor for one k-point
      TYPE (wavefun1)    W1            ! current wavefunction
      TYPE (wavefun1)    WSEARCH       ! current search direction
      TYPE (wavefun1)    WSEARCHL      ! last search direction

      COMPLEX(q),ALLOCATABLE:: CG(:),PRECON(:)


#ifdef MPI
    IF (W%WDES%COMM_KIN%NCPU /= W%WDES%COMM_INB%NCPU) THEN
       CALL VTUTOR('E','NPAR IALGO=8',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IU6,3)
       CALL VTUTOR('S','NPAR IALGO=8',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
            IU0,3)
       STOP
    ENDIF
#endif

      ALLOCATE(W1%CR(GRID%MPLWV*WDES%NRSPINORS),CG(WDES%NRPLWV),PRECON(WDES%NRPLWV))

      CALL SETWDES(WDES,WDES1,0)

      CALL NEWWAV(WSEARCH ,WDES1,.TRUE.)
      CALL NEWWAV(WSEARCHL,WDES1,.TRUE.)
!=======================================================================
!  INITIALISATION:
! maximum  number of iterations
! initialise sum over Energy-change to zero
!=======================================================================
      NITER=4
      IF (.NOT.INFO%LORTHO) NITER=2

      DESUM =0
      RMS   =0
      ICOUEV=0

      SLOCAL=0
      DO I=1,GRID%RL%NP
        SLOCAL=SLOCAL+SV(I,1)
      ENDDO

      CALLMPI( M_sum_d(WDES%COMM_INB, SLOCAL, 1))
      SLOCAL=SLOCAL/GRID%NPLWV

      IWARN=0
!=======================================================================
      spin:    DO ISP=1,WDES%ISPIN
      KPOINTS: DO NK=1,WDES%NKPTS
!=======================================================================
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif

      DE_ATT=ABS(W%CELEN(WDES%NBANDS,NK,ISP)-W%CELEN(1,NK,ISP)) /4

      CALL SETWDES(WDES,WDES1,NK)

      NPL=WDES1%NPL
      NGVECTOR=WDES1%NGVECTOR

      IF (INFO%LREAL) THEN
        CALL PHASER(GRID,LATT_CUR,NONLR_S,NK,WDES)
      ELSE
        CALL PHASE(WDES,NONL_S,NK)
      ENDIF

!=======================================================================
      BANDS: DO N=WDES%NBANDS,1,-1
!=======================================================================

      CALL SETWAV(W,W1,WDES1,N,ISP)  ! allocation for W1%CR done above

      IDUMP=0
#ifdef debug
      IDUMP=2
#endif
#ifdef MPI
      IF (WDES%COMM%NODE_ME /= WDES%COMM%IONODE) IDUMP=0
#endif
      IF (IDUMP==2) WRITE(*,'(I3,1X)',ADVANCE='NO') N
!-----------------------------------------------------------------------
! transform the wave-function to real space
! and calculate the result  of the Ham. acting onto the wavefunction
!-----------------------------------------------------------------------
      DO ISPINOR=0,WDES%NRSPINORS-1
        CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),W1%CR(1+ISPINOR*WDES1%GRID%MPLWV),W1%CW(1+ISPINOR*NGVECTOR),GRID)
      ENDDO
!-----------------------------------------------------------------------
! start with the exact evaluation of the eigenenergy
!-----------------------------------------------------------------------
      IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') REAL( W1%CELEN ,KIND=q)
      CALL ECCP(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W1%CELEN)
      IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') REAL( W1%CELEN ,KIND=q)
!-----------------------------------------------------------------------
! calculate the preconditioning matrix  (only once for each band)
!-----------------------------------------------------------------------
      IF (INFO%IALGO==7 .OR.INFO%IALGO==8) THEN
        EKIN=0

        DO ISPINOR=0,WDES%NRSPINORS-1
        DO M=1,NGVECTOR
          MM=M+NGVECTOR*ISPINOR
          CPT=W1%CW(MM)
          EKIN =EKIN+ REAL( CPT*CONJG(CPT) ,KIND=q) * WDES1%DATAKE(M,ISPINOR+1)
        ENDDO
        ENDDO

        CALLMPI( M_sum_d(WDES%COMM_INB, EKIN, 1))

        IF (EKIN<2.0_q) EKIN=2.0_q
        EKIN=EKIN*1.5_q
        IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') EKIN

        FAKT=2._q/EKIN
        DO ISPINOR=0,WDES%NRSPINORS-1
        DO M=1,NGVECTOR
          MM=M+NGVECTOR*ISPINOR
          X=WDES1%DATAKE(M,ISPINOR+1)/EKIN
          X2= 27+X*(18+X*(12+8*X))
          PRECON(MM)=X2/(X2+16*X*X*X*X)*FAKT
        ENDDO
        ENDDO
      ELSE IF (INFO%IALGO==0) THEN
        EVALUE=W%CELEN(N,NK,ISP)
        DO ISPINOR=0,WDES%NRSPINORS-1
        DO M=1,NGVECTOR
          MM=M+NGVECTOR*ISPINOR
          PRECON(MM)=1._q/(WDES1%DATAKE(M,ISPINOR+1)+SLOCAL-EVALUE+ CMPLX( 0 , DE_ATT ,KIND=q) )
        ENDDO
        ENDDO
      ELSE
          PRECON=1
      ENDIF
!=======================================================================
! MAIN LOOP: intra-band minimisation
!=======================================================================
      DEIT=0._q
      DELAST=0._q
      DECELL=1.E30_q

      ITER_BAND: DO ITER=1,NITER

!-----------------------------------------------------------------------
! calculate the search-directions  H-epsilon S |phi>
!-----------------------------------------------------------------------
      DEBAND=0
      EVALUE=W1%CELEN

      CALL HAMILT(W1, NONLR_S, NONL_S, EVALUE, &
     &    CDIJ, CQIJ, SV, ISP, WSEARCH%CW)
      DETEST=0
      FNORM=0
      DO M=1,WDES1%NPL
        CPT     =W1%CW(M)
        WSEARCH%CW(M)   =WSEARCH%CW(M)-EVALUE*CPT
        CG(M)   =WSEARCH%CW(M)
        DETEST =DETEST+WSEARCH%CW(M)*CONJG(W1%CW(M))
        FNORM  =FNORM+WSEARCH%CW(M)*CONJG(WSEARCH%CW(M))
      ENDDO

      CALLMPI( M_sum_s(WDES%COMM_INB, 2, DETEST, FNORM, 0._q, 0._q))
!     norm of total error vector before start
      IF (ITER==1) THEN
      IF (IDUMP==2) WRITE(*,'(F9.4)',ADVANCE='NO') SQRT(ABS(FNORM))
        RMS=RMS+WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)* &
     &       SQRT(ABS(FNORM))/WDES%NBANDS
      ENDIF
!----------------------------------------------------------------------
! PRECONDITIONING:
! explicit orthogonalisation to all bands must be done here if the
! sub-space-hamiltonian is not diagonal
! we simplify things a little bit for US-potential
!----------------------------------------------------------------------
      DO M=1,WDES1%NPL
         WSEARCH%CW(M)=WSEARCH%CW(M)*PRECON(M)
      ENDDO
!----------------------------------------------------------------------
! now orthogonalise the (preconditioned) search-direction to all bands
!----------------------------------------------------------------------
      IF (INFO%LREAL) THEN
        DO ISPINOR=0,WDES%NRSPINORS-1
           CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),WSEARCH%CR(1+ISPINOR*WDES1%GRID%MPLWV),WSEARCH%CW(1+ISPINOR*NGVECTOR),GRID)
        ENDDO
        CALL RPRO1(NONLR_S,WDES1,WSEARCH)
      ELSE
         CALL PROJ1(NONL_S,WDES1,WSEARCH)
      ENDIF

      IF (INFO%LORTHO) THEN
        CALL ORTHON(NK, W,WSEARCH, CQIJ, ISP)
      ENDIF
!----------------------------------------------------------------------
! steepest descent  ok
!----------------------------------------------------------------------
      IF (INFO%IALGO==5 .OR.INFO%IALGO==7) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO M=1,WDES1%NPL
        WSEARCHL%CW(M)=WSEARCH%CW(M)
      ENDDO
!=======================================================================
! conjugated gradient
! WSEARCHL%CW stores the last search direction (after  conjugation)
!=======================================================================
      ELSE IF (INFO%IALGO==6 .OR.INFO%IALGO==8 .OR.INFO%IALGO==0) THEN
!----------------------------------------------------------------------
! if first step that is all
! reinitialise the conjugated gradient algorithm every 20 loop
!----------------------------------------------------------------------
      IF (MOD(ITER,20)==1) THEN

!DIR$ IVDEP
!OCL NOVREC
      DO M=1,WDES1%NPL
        WSEARCHL%CW(M)=WSEARCH%CW(M)
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO NPRO=1,WDES%NPRO
        WSEARCHL%CPROJ(NPRO)= WSEARCH%CPROJ(NPRO)
      ENDDO

      GNORML=0
      DO M=1,WDES1%NPL
        GNORML =GNORML +WSEARCH%CW(M)*CONJG(CG(M))
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM_INB, GNORML, 1))
!----------------------------------------------------------------------
! calculate the conjugated direction
!----------------------------------------------------------------------
      ELSE

      GNORM=0
      DO M=1,WDES1%NPL
        GNORM =GNORM +WSEARCH%CW(M)*CONJG(CG(M))
      ENDDO
      CALLMPI( M_sum_d(WDES%COMM_INB, GNORM, 1))

      GAMMA =GNORM/GNORML
      GNORML=GNORM

!DIR$ IVDEP
!OCL NOVREC
      DO M=1,WDES1%NPL
        WSEARCH%CW (M)=WSEARCH%CW(M)+GAMMA*WSEARCHL%CW(M)
        WSEARCHL%CW(M)=WSEARCH%CW(M)
      ENDDO
!DIR$ IVDEP
!OCL NOVREC
      DO NPRO=1,WDES%NPRO
        WSEARCH%CPROJ (NPRO)=WSEARCH%CPROJ(NPRO)+ &
     &                        GAMMA*WSEARCHL%CPROJ(NPRO)
        WSEARCHL%CPROJ(NPRO)=WSEARCH%CPROJ(NPRO)
      ENDDO

      ENDIF
      ENDIF
!=======================================================================
! SEARCH -DIRECTION  is now set up -- if we orthogonalize
! perform a FFT of the search direction F to real space FR
!=======================================================================
      IF (INFO%LORTHO.OR.(.NOT.INFO%LREAL)) THEN
         DO ISPINOR=0,WDES%NRSPINORS-1
            CALL FFTWAV(NGVECTOR,WDES1%NINDPW(1),WSEARCH%CR(1+ISPINOR*WDES1%GRID%MPLWV),WSEARCH%CW(1+ISPINOR*NGVECTOR),GRID)
         ENDDO
      ELSE IF (INFO%IALGO==6 .OR.INFO%IALGO==8 .OR.INFO%IALGO==0) THEN
!=======================================================================
!  if we do not orthogonalize we can use another strategy: we need then
!  only correct for conjugation and this requires only the FFT of the
!  search-direction of the last step which can be stored in WSEARCHL ... .
!  This helps to save one FFT (only) if we use the real space scheme
!=======================================================================
        IF (MOD(ITER,20)/=1) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO ISPINOR=0,WDES%NRSPINORS-1
          DO M=1,WDES1%GRID%RL%NP
             MM=M+ISPINOR*WDES1%GRID%MPLWV
            WSEARCH%CR(MM)=WSEARCH%CR(MM) + GAMMA*WSEARCHL%CR(MM)
          ENDDO
          ENDDO
        ENDIF
        IF (ITER/=NITER) THEN
!DIR$ IVDEP
!OCL NOVREC
          DO ISPINOR=0,WDES%NRSPINORS-1
          DO M=1,WDES1%GRID%RL%NP
             MM=M+ISPINOR*WDES1%GRID%MPLWV
             WSEARCHL%CR(MM)=WSEARCH%CR(MM)
          ENDDO
          ENDDO
        ENDIF
      ENDIF
!=======================================================================
! project out the wave-function of the current Band and normalise
!    search direction
!=======================================================================
      CALL PROJCN(WSEARCH,W1, CQIJ, ISP, CSCPD)
      CALL CNORMN(WSEARCH, CQIJ, ISP, WSCAL)
!----------------------------------------------------------------------
! panic here: our search direction can not be normalised
!     to save things goon to next band
!----------------------------------------------------------------------
      IF (WSCAL<=0) GOTO 1000

!-----correct the FFT of search direction WSEARCH%CW for this operation
!DIR$ IVDEP
!OCL NOVREC
      DO ISPINOR=0,WDES%NRSPINORS-1
      DO K=1,WDES1%GRID%RL%NP
         KK=K+ISPINOR*WDES1%GRID%MPLWV
         WSEARCH%CR(KK)=(WSEARCH%CR(KK)-CSCPD*W1%CR(KK))*WSCAL
      ENDDO
      ENDDO
!=======================================================================
! calculate the Hamiltonion between two trial states
!=======================================================================
      CALL ECCP(WDES1,W1,WSEARCH,      LMDIM,CDIJ(1,1,1,ISP),GRID, SV(1,ISP), CB)
      CALL ECCP(WDES1,WSEARCH,WSEARCH, LMDIM,CDIJ(1,1,1,ISP),GRID, SV(1,ISP), CA)
!----------------------------------------------------------------------
! calculate the the following quantities B= 2*RE(<F|H|C>) A= <F|H|F>
! where F is the search direction and C the current Wavefunction
! !! due to the cutoff CB=<F|H|C> and CBP=<C|H|F> are not equal !!
!----------------------------------------------------------------------
      BB  =2* REAL( CB ,KIND=q)
      ADD =CA-W1%CELEN
!=======================================================================
! calculate now the position of the minimum  THETA
!  and the exact change in the eigenvalue DECEL0
!  and approximate changes in Energy due to 2. derivatives of Energy
!=======================================================================
      THETA=.5_q*ATAN(-BB/ADD)
      IF (ADD<0) THETA=THETA+PI/2

      DCOST=  COS(THETA)
      SINT =  SIN(THETA)

      IF (IDUMP==1) &
     &WRITE(*,'(A4,7E11.4)')'B,A',ADD,THETA,DCOST,SINT

      IF (IDUMP==2) &
     &WRITE(*,'(1X,F7.4,1X)',ADVANCE='NO') -THETA*WSCAL

      DECEL0=BB/2*SIN(2*THETA) +ADD  /2*(1-COS(2*THETA))
!----in case of serious convergence trouble one might try this ???
      IF ((.NOT.INFO%LORTHO).AND.(ABS(DECEL0)>ABS(DECELL))) GOTO 1000
      DECELL=MAX(ABS(DECEL0),INFO%EBREAK*1000)
!=======================================================================
! calculate the new-wavefunction
! and the Eigenvalue corresponding to the old Hamiltonian
! the new eigenvalue must be: CELEN(NEW)= CELEN(OLD)+DECEL0
!=======================================================================
!DIR$ IVDEP
!OCL NOVREC
      DO M=1,WDES1%NPL
        W1%CW(M)=W1%CW(M)*DCOST+WSEARCH%CW(M)*SINT
      ENDDO

!DIR$ IVDEP
!OCL NOVREC
      DO ISPINOR=0,WDES%NRSPINORS-1
      DO M=1,WDES1%GRID%RL%NP
         MM=M+ISPINOR*WDES1%GRID%MPLWV
         W1%CR(MM)=W1%CR(MM)*DCOST+WSEARCH%CR(MM)*SINT
      ENDDO
      ENDDO

!DIR$ IVDEP
!OCL NOVREC
      DO NPRO=1,WDES%NPRO
        W1%CPROJ(NPRO)= W1%CPROJ(NPRO)*DCOST+ WSEARCH%CPROJ(NPRO)*SINT
      ENDDO

      DECEL=DECEL0
      W1%CELEN=W1%CELEN+DECEL0

      CALL ECCP(WDES1,W1,W1,LMDIM,CDIJ(1,1,1,ISP),GRID,SV(1,ISP), W1%CELEN)
      DECEL=W1%CELEN-EVALUE

      IF (IDUMP==1) WRITE(*,'(A4,I3,4E14.7)')'DEC',N,DECEL,DECEL0
      IF (IDUMP==2) WRITE(*,'(E10.2)',ADVANCE='NO') DECEL

      DE    =DECEL
      DESUM =DESUM+ WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*DECEL
      ICOUEV=ICOUEV+1
!=======================================================================
! break of intra-band-minimisation
! at the moment we performe a break of the intra-band minimization if
! ) DE is less then DEPER % of the change in the first minimization
!     of this band (relative breakcondition)
! ) DE less then EBREAK (absolut breakcondition)
! ) if unoccupied band break after 2. iteration
!=======================================================================
      IF (DE>0 .AND.DELAST/=0 .AND.DE>INFO%EBREAK/10) THEN
        IF (IU6>=0) WRITE(IU6,2)  DELAST,DE
        IWARN=1
 2    FORMAT(' WARNING: EDSTEP: Energy is increasing , last change was', &
     &                  E10.2,' (set INFO%EBREAK)' / &
     &       '          current change ',E10.2/ )
      ENDIF


      IF (ABS(DE)<INFO%EBREAK) GOTO 1000
      IF (ABS(W%FERWE(N,NK,ISP))<ABS(INFO%WEIMIN) .AND. ITER >= 2)  &
     &        GOTO 1000
      IF (ABS(DE)<ABS(DEIT)) GOTO 1000
      IF (ITER==1) THEN
        DEIT=DE*INFO%DEPER
      ENDIF

      DELAST=DE
      ENDDO ITER_BAND
!=======================================================================
! move onto the next Band
!=======================================================================
 1000 CONTINUE
      W%CELEN(N,NK,ISP)=W1%CELEN
      IF (IDUMP==2) WRITE(*,'(F9.4)') REAL( W1%CELEN ,KIND=q)
      IF (IDUMP==10) WRITE(*,*)
      ENDDO BANDS
!=======================================================================
      ENDDO KPOINTS
      ENDDO spin
!=======================================================================

      IF (IWARN==1 .AND. IU0>=0) &
     & WRITE(IU0,*)'WARNING: EDSTEP: Energy increased during minimization'

      DEALLOCATE(CG,PRECON)
      CALL DELWAV(WSEARCH ,.TRUE.)
      CALL DELWAV(WSEARCHL,.TRUE.)


      CALLMPI( M_sum_d(WDES%COMM_KINTER, DESUM, 1))
      CALLMPI( M_sum_d(WDES%COMM_KINTER, RMS, 1))
      CALLMPI( M_sum_i(WDES%COMM_KINTER, ICOUEV, 1))
#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_CELTOT(WDES,W)
      ENDIF
      
      IF ((LHFCALC.OR.LUSEPEAD()).AND.WDES%COMM_KINTER%NCPU.GT.1) THEN
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)
      ENDIF
#endif

      RETURN
      END SUBROUTINE
      END MODULE








