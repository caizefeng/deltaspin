#include "symbol.inc"
      MODULE melf
      USE prec
      CONTAINS
!************************* SUBROUTINE ELF ******************************
! RCS:  $Id: elf.F,v 1.4 2003/06/27 13:22:16 kresse Exp kresse $
!
! subroutine CHSP constructs the electronic charge density according
! to the current wavefunctions and fermi-weights
!
!***********************************************************************
      SUBROUTINE ELF(GRID,GRID_SOFT,LATT_CUR,SYMM,NIOND, W,WDES,  &
               CHDEN,CELF)
       USE prec
       USE msymmetry
       USE base
       USE lattice
       USE mpimy
       USE mgrid
       USE wave
       USE constant
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID,GRID_SOFT
      TYPE (latt)        LATT_CUR
      TYPE (symmetry)    SYMM
      TYPE (wavespin)    W
      TYPE (wavedes)     WDES
! final resul
      COMPLEX(q)   CELF(GRID_SOFT%MPLWV,WDES%NCDIJ), &
                   CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
! dynamic work arrays
      COMPLEX(q) :: CF(WDES%NRPLWV)
      COMPLEX(q),ALLOCATABLE :: CR(:),CR2(:)
      RGRID,     ALLOCATABLE :: CDWORK(:),CNEW(:)

! the compaq F90 compiler has a funny bug that needs this crazy workaround
! dont ask why, it took me quite a while to find this workaround anyway
      INTEGER DECSYMM
      DECSYMM=SYMM%ISYM
! thats it. 20010606 Robin Hirschl
#ifdef MPI
      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('ELF: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif
      CELF=0
      IF (WDES%NCDIJ==4) THEN
         WRITE(*,*) 'WARNING: ELF not implemented for non collinear case'
         RETURN
      ENDIF

      MPLWV=MAX(GRID%MPLWV,GRID_SOFT%MPLWV)
      ALLOCATE(CDWORK(DIMREAL(MPLWV)),CNEW(DIMREAL(MPLWV)), &
               CR(MPLWV),CR2(MPLWV))

!=======================================================================
!    first recalculate charge-density (propably not really necessary)
!    ::  CHDEN
!=======================================================================
spin: DO ISP=1,WDES%NCDIJ

      CDWORK=0
!=======================================================================
! loop over k-points and bands
! get  real space charge density
!=======================================================================
      band_k1: DO NCOUNT=0,WDES%NBANDS*WDES%NKPTS-1

      NK=NCOUNT/WDES%NBANDS+1
      N=MOD(NCOUNT,WDES%NBANDS)+1

      WEIGHT=WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)/LATT_CUR%OMEGA
      NPL=WDES%NPLWKP(NK)
      CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CR,W%CPTWFP(1,N,NK,ISP),GRID)

      DO M=1,GRID%RL%NP
        CDWORK(M)=CDWORK(M)+ &
     &  REAL( CR(M)*CONJG(CR(M)) ,KIND=q) *WEIGHT
      ENDDO
      ENDDO band_k1
! merge charge from all nodes
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDWORK(1), GRID%RL%NP))
#else
      CALLMPI( M_sum_z(WDES%COMM_INTER, CDWORK(1), GRID%RL%NP))
#endif

! Fourier-Transformation of charge-density to recip space
      CALL FFT_RC_SCALE(CDWORK,CHDEN(1,ISP),GRID_SOFT)
!  symmetrization of charge-density
      IF (DECSYMM>0) CALL RHOSYM_REAL(CHDEN(1,ISP),GRID_SOFT, & 
              SYMM%PTRANS,NIOND,SYMM%MAGROT,ISP)
!=======================================================================
!    then calculate kinetic energy
!    :: CDWORK
!=======================================================================
      CDWORK=0._q

      band_k2: DO NCOUNT=0,WDES%NBANDS*WDES%NKPTS-1

      NK=NCOUNT/WDES%NBANDS+1
      N=MOD(NCOUNT,WDES%NBANDS)+1

      WEIGHT=WDES%RSPIN*WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)/LATT_CUR%OMEGA

      NPL=WDES%NPLWKP(NK)
      DO M=1,NPL
        CF(M)=WDES%DATAKE(M,ISP,NK)*W%CPTWFP(M,N,NK,ISP)
      ENDDO
!     fourier-transformation of wave-function
      CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CR(1),W%CPTWFP(1,N,NK,ISP),GRID)
!     fourier-transformation of wave-function *kinetic energy
      CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CR2(1),CF,GRID)

      DO M=1,GRID%RL%NP
        CDWORK(M)=CDWORK(M)+ REAL( CR(M)*CONJG(CR2(M)) ,KIND=q) *WEIGHT
      ENDDO

      ENDDO band_k2
! merge kinetic energy from nodes
#ifdef realmode
      CALLMPI( M_sum_d(WDES%COMM_INTER, CDWORK(1), GRID%RL%NP))
#else
      CALLMPI( M_sum_z(WDES%COMM_INTER, CDWORK(1), GRID%RL%NP))
#endif
!=======================================================================
!  symmetrization of result CDWORK
!=======================================================================
      IF (DECSYMM>0) THEN
        DO I=1,GRID_SOFT%RL%NP
         CDWORK(I)=CDWORK(I)/GRID_SOFT%NPLWV
        ENDDO
! kinetic energy in rec. space:
        CALL FFT3D(CDWORK,GRID_SOFT,-1)
! symmetrization of kinetic energy
        IF (DECSYMM>0) CALL RHOSYM_REAL(CDWORK,GRID_SOFT, &
             SYMM%PTRANS,NIOND,SYMM%MAGROT,ISP)
! back to real space:
        CALL FFT3D(CDWORK,GRID_SOFT,1)
      ENDIF

!=======================================================================
! calculate |grad rho|^2
! :: CNEW
!=======================================================================
      CNEW=0

! x-component:
      DO I=1,GRID_SOFT%RC%NP
         N1= MOD((I-1),GRID_SOFT%RC%NROW) +1
         NC= (I-1)/GRID_SOFT%RC%NROW+1
         N2= GRID_SOFT%RC%I2(NC)
         N3= GRID_SOFT%RC%I3(NC)
         GX=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(1,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(1,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(1,3))
         CR(I)=CHDEN(I,ISP)*GX*CITPI
      ENDDO
! grad_x to real space:
      CALL FFT3D(CR,GRID_SOFT,1)
      CALL SQADD(CR,CNEW,GRID_SOFT%RL%NP)

! y-component:
      DO I=1,GRID_SOFT%RC%NP
         N1= MOD((I-1),GRID_SOFT%RC%NROW) +1
         NC= (I-1)/GRID_SOFT%RC%NROW+1
         N2= GRID_SOFT%RC%I2(NC)
         N3= GRID_SOFT%RC%I3(NC)
         GY=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(2,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(2,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(2,3))
         CR(I)=CHDEN(I,ISP)*GY*CITPI
      ENDDO
! grad_x to real space:
      CALL FFT3D(CR,GRID_SOFT,1)
      CALL SQADD(CR,CNEW,GRID_SOFT%RL%NP)

! z-component:
      DO I=1,GRID_SOFT%RC%NP
         N1= MOD((I-1),GRID_SOFT%RC%NROW) +1
         NC= (I-1)/GRID_SOFT%RC%NROW+1
         N2= GRID_SOFT%RC%I2(NC)
         N3= GRID_SOFT%RC%I3(NC)
         GZ=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(3,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(3,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(3,3))
         CR(I)=CHDEN(I,ISP)*GZ*CITPI
      ENDDO
! grad_x to real space:
      CALL FFT3D(CR,GRID_SOFT,1)
      CALL SQADD(CR,CNEW,GRID_SOFT%RL%NP)

!=======================================================================
!  calculate div(grad Rho)
!  :: CR
!=======================================================================

      DO I=1,GRID_SOFT%RC%NP
         N1= MOD((I-1),GRID_SOFT%RC%NROW) +1
         NC= (I-1)/GRID_SOFT%RC%NROW+1
         N2= GRID_SOFT%RC%I2(NC)
         N3= GRID_SOFT%RC%I3(NC)
         GX=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(1,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(1,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(1,3))
         GY=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(2,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(2,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(2,3))
         GZ=(GRID_SOFT%LPCTX(N1)*LATT_CUR%B(3,1)+GRID_SOFT%LPCTY(N2)*LATT_CUR%B(3,2)+GRID_SOFT%LPCTZ(N3)*LATT_CUR%B(3,3))
         GSQU=GX*GX+GY*GY+GZ*GZ
         CR(I)=-CHDEN(I,ISP)*GSQU*TPI*TPI
      ENDDO

! div grad in real space:
      CALL FFT3D(CR,GRID_SOFT,1)

! charge-density in real space:
      CALL FFT3D(CHDEN(1,ISP),GRID_SOFT,1)

!=======================================================================
!  calculate ELF (Nature, 371(1994)683-686)
!=======================================================================
      CALL ELFCAL(CHDEN(1,ISP),CR,CDWORK,CNEW,GRID_SOFT)

! Fourier-Transformation of ELF to real space
      CALL FFT_RC_SCALE(CDWORK,CELF(1,ISP),GRID_SOFT)
      ENDDO spin

      DEALLOCATE(CDWORK,CR,CR2,CNEW)

      RETURN
      END SUBROUTINE
      END MODULE melf

!=======================================================================
!  small helper routine required because of REAL(q) - COMPLEX
!  problems
!=======================================================================
      SUBROUTINE SQADD(ARRAY1,ARRAY2,NP)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      RGRID ARRAY1(NP),ARRAY2(NP)

      DO I=1,NP
       ARRAY2(I)=ARRAY2(I)+ABS(ARRAY1(I))**2
      ENDDO
      RETURN
      END SUBROUTINE


      SUBROUTINE ELFCAL(CHDEN,LAPLAC,CKINE,CGRDSQ,GRID)
      USE prec
      USE mpimy
      USE mgrid
      USE constant
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRID

      RGRID CHDEN(GRID%RL%NP),CKINE(GRID%RL%NP),CGRDSQ(GRID%RL%NP)
      RGRID LAPLAC(GRID%RL%NP)
!=======================================================================
!  calculate ELF (e.g.: Nature, 371(1994)683-686)
!            _
!            h^2    *    2      T.........kinetic energy
! T    = - 2 --- Psi grad Psi   T+TCORR...pos.definite kinetic energy
!            2 m                TBOS......T of an ideal Bose-gas
!          _                                (=infimum of T+TCORR)
!        1 h^2      2           DH........T of hom.non-interact.e- - gas
! TCORR= - ---  grad rho                    (acc.to Fermi)
!        2 2 m                  ELF.......electron-localization-function
!          _             2
!        1 h^2 |grad rho|
! TBOS = - --- ----------       D = T + TCORR - TBOS
!        4 2 m    rho
!          _                                \                1
!        3 h^2        2/3  5/3          =====>    ELF = ------------
! DH   = - --- (3 Pi^2)  rho                /                   D   2
!        5 2 m                                           1 + ( --- )
!                                                              DH
!=======================================================================
      PISQ   = PI*PI
      FIVTHI = 5._q/3._q

      DO N=1,GRID%RL%NP
       T    = REAL( CKINE(N) ,KIND=q)
       TCORR= REAL( LAPLAC(N) ,KIND=q) *HSQDTM/2._q
       TBOS = HSQDTM/4._q* REAL( CGRDSQ(N)/CHDEN(N) ,KIND=q)
       IF (REAL(CHDEN(N), KIND=q) < 0.0_q) THEN
          DH = 0.0_q
       ELSE 
          DH = 0.2_q*HSQDTM/PISQ* (3*PISQ* REAL( CHDEN(N) ,KIND=q) )**FIVTHI
       ENDIF
       CKINE(N)=1/(1+((T+TCORR-TBOS)/MAX(DH,1E-8_q))**2)
      ENDDO

      RETURN

      END

!
! calling interface REAL-> COMPLEX
!
      SUBROUTINE RHOSYM_REAL(CHTOT,GRID_SOFT,PTRANS,NIOND,MAGROT,ISP)
      USE prec
      USE msymmetry
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRID_SOFT

      COMPLEX(q) CHTOT(GRID_SOFT%RC%NP)
      DIMENSION  PTRANS(NIOND+2,3)
      REAL(q)    MAGROT(48,*) 

      CALL  RHOSYM(CHTOT,GRID_SOFT,PTRANS,NIOND,MAGROT,ISP)
      END
