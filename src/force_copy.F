#include "symbol.inc"
      MODULE force
      USE prec
      CONTAINS
      SUBROUTINE mpi_dummy
      WRITE(*,*)'Im a DEC compiler so I need this line'
      END SUBROUTINE
!************************ SUBROUTINE STRELO ****************************
! RCS:  $Id: force.F,v 1.3 2002/08/14 13:59:39 kresse Exp $
!
! this calculates the stress on the unit cell
! which is related to the change in local pseudopotential on changing
! the unit vectors (EISIF + PSCSIF)
! and the stress related to the change of the hartree energy (DSIF)
! see formulas (10.30) (10.51) and (10.52) in thesis gK
!
!***********************************************************************

      SUBROUTINE STRELO(GRIDC,P,T_INFO,LATT_CUR, &
           CHTOT,CSTRF, NELECT,DSIF,EISIF,PSCSIF)
      USE prec

      USE pot
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR

      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      COMPLEX(q) CHTOT(GRIDC%RC%NP)
      REAL(q)    EISIF(3,3),DSIF(3,3),PSCSIF(3,3),NELECT

! work arrays
      COMPLEX(q), ALLOCATABLE::  CWORK1(:),CWORK2(:)

      PROFILING_START('strelo')

      ALLOCATE(CWORK1(GRIDC%MPLWV),CWORK2(GRIDC%MPLWV))
!=======================================================================
! first caclulate the local potential and its derivative
!=======================================================================
      CALL POTION(GRIDC,P,LATT_CUR,T_INFO,CWORK1,CWORK2,CSTRF,PSCENC)
!=======================================================================
! there are two contributions to the force on the unit cell,
! one due to the tendency to put the charge density  at the largest
! values of the pseudopotential and the other due to the
! 1/(cell volume) dependence
! see formulas (10.51) and (10.52) in thesis gK
!=======================================================================
      EISIF =0    ! stress due to local potential
      DSIF  =0    ! stress due to Hartree potential
      PSCSIF=0    ! stress due to q-> 0 behaviour (10.30) in thesis gK

      PSCV=0      ! 1/ volume part for local PP
      DENC=0      ! 1/ volume part for Hartree pot

      NG=1
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW

        SETFACT1
        SETFACT

        GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
        GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
        GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))

        GSQU=GX**2+GY**2+GZ**2
        GMOD=SQRT(GSQU)
        PSCV=PSCV+MULFACT REAL( CONJG(CHTOT(NG))*CWORK1(NG) ,KIND=q)
!     G=0 part is handled somewhere else
        IF (GMOD>1E-7_q) THEN ! avoid division by 0

         PSCD= MULFACT REAL( CONJG(CHTOT(NG))*CWORK2(NG) ,KIND=q)
         EISIF(1,1)=EISIF(1,1)+PSCD*GX*GX/GMOD
         EISIF(2,2)=EISIF(2,2)+PSCD*GY*GY/GMOD
         EISIF(3,3)=EISIF(3,3)+PSCD*GZ*GZ/GMOD
         EISIF(1,2)=EISIF(1,2)+PSCD*GX*GY/GMOD
         EISIF(2,3)=EISIF(2,3)+PSCD*GY*GZ/GMOD
         EISIF(3,1)=EISIF(3,1)+PSCD*GZ*GX/GMOD

         DUM=MULFACT CHTOT(NG)*CONJG(CHTOT(NG))/GSQU
         DENC=DENC+DUM
         DSIF(1,1)=DSIF(1,1)-DUM*GX*GX/GSQU
         DSIF(2,2)=DSIF(2,2)-DUM*GY*GY/GSQU
         DSIF(3,3)=DSIF(3,3)-DUM*GZ*GZ/GSQU
         DSIF(1,2)=DSIF(1,2)-DUM*GX*GY/GSQU
         DSIF(2,3)=DSIF(2,3)-DUM*GY*GZ/GSQU
         DSIF(3,1)=DSIF(3,1)-DUM*GZ*GX/GSQU

        ENDIF
        NG=NG+1
      ENDDO row
      ENDDO col
!=======================================================================
! scale the forces on the unit cell 2 Pi (and e^2/Omea/(2 pi)^2
! and add the  contribution to the total force from the
! 1/(unit cell volume)  dependence of the electron-ion energy
!=======================================================================
      EISIF(2,1)=EISIF(1,2)
      EISIF(3,2)=EISIF(2,3)
      EISIF(1,3)=EISIF(3,1)

      DSIF(2,1)=DSIF(1,2)
      DSIF(3,2)=DSIF(2,3)
      DSIF(1,3)=DSIF(3,1)

      SCALE= EDEPS/LATT_CUR%OMEGA/TPI**2
      DENC = DENC*SCALE/2

      DO K=1,3
        DO M=1,3
          EISIF(M,K)=EISIF(M,K)*TPI
          DSIF(M,K) =DSIF(M,K) *SCALE
        ENDDO
        PSCSIF(K,K)=PSCSIF(K,K)+PSCENC ! q-> 0 term
        DSIF  (K,K)=DSIF  (K,K)+DENC   ! add 1/ omega term
        EISIF (K,K)=EISIF (K,K)+PSCV   ! add 1/ omega term
      ENDDO

      CALLMPI( M_sum_d(GRIDC%COMM, DSIF, 9))
      CALLMPI( M_sum_d(GRIDC%COMM,EISIF, 9))

      DEALLOCATE(CWORK1,CWORK2)

      PROFILING_STOP('strelo')

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE STREHAR ***************************
!
! this calculates the correction to the stress
! +   if the harris-functional is used                   LPAR=.FALSE.
!        (CHGGA = gradient of Harris functional)
! +   stress due to partial core corrections
!        (CHGGA = CVXC)                                  LPAR=.TRUE.
!
! LUNIF determines if the 1/volume  dependence of the charge is taken
!       into account
! for LPAR=.TRUE. LUNIF=.F. because this part is calculated in POTEX
!
!***********************************************************************

      SUBROUTINE STREHAR(GRIDC,P,T_INFO,LATT_CUR,LPAR,CHGGA,CSTRF, HARSIF)
      USE prec


      USE charge
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR

      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      COMPLEX(q) CHGGA(GRIDC%MPLWV)
      REAL(q)    HARSIF(3,3)
      LOGICAL LUNIF,LPAR
! work arrays
      COMPLEX(q), ALLOCATABLE::  CWORK1(:),CWORK2(:)

      PROFILING_START('strehar')

      ALLOCATE(CWORK1(GRIDC%MPLWV),CWORK2(GRIDC%MPLWV))
!=======================================================================
! first calculate the chargedensity and its derivative
!=======================================================================
      LUNIF=.NOT. LPAR
      CALL RHOATO(.FALSE.,LPAR,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CWORK1,CWORK2)
!=======================================================================
! now sum up all components
! as in STRELO there is one 1/ volume term plus other terms
!=======================================================================

      HARSIF=0
      PSCV  =0       ! required to calc 1/ volume term

      NG=1
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
        SETFACT1
        SETFACT

        GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
        GY= (GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
        GZ= (GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))

        GSQU=GX**2+GY**2+GZ**2
        GMOD=SQRT(GSQU)
        PSCV=PSCV+MULFACT REAL( CONJG(CHGGA(NG))*CWORK1(NG) ,KIND=q)
!=======================================================================
! avoid G=0
!=======================================================================
        IF(GMOD>1E-7_q) THEN
         PSCD= MULFACT REAL( CONJG(CHGGA(NG))*CWORK2(NG) ,KIND=q)
         HARSIF(1,1)=HARSIF(1,1)+PSCD*GX*GX/GMOD
         HARSIF(2,2)=HARSIF(2,2)+PSCD*GY*GY/GMOD
         HARSIF(3,3)=HARSIF(3,3)+PSCD*GZ*GZ/GMOD
         HARSIF(1,2)=HARSIF(1,2)+PSCD*GX*GY/GMOD
         HARSIF(2,3)=HARSIF(2,3)+PSCD*GY*GZ/GMOD
         HARSIF(3,1)=HARSIF(3,1)+PSCD*GZ*GX/GMOD
        ENDIF
        NG=NG+1
      ENDDO row
      ENDDO col

!=======================================================================
! scale the forces on the unit cell by 2 Pi
! and add the  contribution to the total force from the
! 1/(unit cell volume)  dependence of the energy if LUNIF is TRUE
!=======================================================================
      HARSIF(2,1)=HARSIF(1,2)
      HARSIF(3,2)=HARSIF(2,3)
      HARSIF(1,3)=HARSIF(3,1)

      DO K=1,3
        DO M=1,3
          HARSIF(M,K)=HARSIF(M,K)*TPI
        ENDDO
      IF (LUNIF) HARSIF (K,K)=HARSIF (K,K)+PSCV
      ENDDO

      CALLMPI( M_sum_d(GRIDC%COMM, HARSIF, 9))

      DEALLOCATE(CWORK1,CWORK2)

      PROFILING_STOP('strehar')

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE STRKIN ****************************
!
! this subroutine calculates the stress resulting
! from the change in the kinetic
! energy of the plane wave basis states as the size of the cell changes
! formula (10.50) in thesis gK
!
!***********************************************************************

      SUBROUTINE STRKIN(W,WDES, B,SIKEF)
      USE prec


      USE constant
      USE wave
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (wavedes)     WDES
      TYPE (wavespin)    W

      DIMENSION SIKEF(3,3)
      DIMENSION B(3,3)
      DIMENSION QSP(3)
! work arrays

      PROFILING_START('strkin')

      SIKEF=0
      spin: DO ISP=1,WDES%ISPIN
!=======================================================================
! loop over all k-points and bands
!=======================================================================
      band_k: DO NCOUNT=0,WDES%NBANDS*WDES%NKPTS-1

      NK=NCOUNT/WDES%NBANDS+1
#ifdef MPI
      IF (MOD(NK-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE
#endif
      N=MOD(NCOUNT,WDES%NBANDS)+1
      NPL=WDES%NGVECTOR(NK)

      SIKE1=0
      SIKE2=0
      SIKE3=0
      SIKE12=0
      SIKE23=0
      SIKE31=0
!-MM- changes to accommodate spin spirals      
      QSP=WDES%QSPIRAL/2
!-MM- end of addition
      spinor: DO ISPINOR=0,WDES%NRSPINORS-1
      DO M=1,NPL
        MM=M+NPL*ISPINOR
        G1=WDES%IGX(M,NK)+WDES%VKPT(1,NK)-QSP(1)
        G2=WDES%IGY(M,NK)+WDES%VKPT(2,NK)-QSP(2)
        G3=WDES%IGZ(M,NK)+WDES%VKPT(3,NK)-QSP(3)

        GX= (G1*B(1,1)+G2*B(1,2)+G3*B(1,3)) *TPI
        GY= (G1*B(2,1)+G2*B(2,2)+G3*B(2,3)) *TPI
        GZ= (G1*B(3,1)+G2*B(3,2)+G3*B(3,3)) *TPI

        CPT   =W%CPTWFP(MM,N,NK,ISP)
        WFMAG =CPT*CONJG(CPT)

        SIKE1 =SIKE1 + GX*GX*HSQDTM *WFMAG
        SIKE2 =SIKE2 + GY*GY*HSQDTM *WFMAG
        SIKE3 =SIKE3 + GZ*GZ*HSQDTM *WFMAG
        SIKE12=SIKE12+ GX*GY*HSQDTM *WFMAG
        SIKE23=SIKE23+ GY*GZ*HSQDTM *WFMAG
        SIKE31=SIKE31+ GZ*GX*HSQDTM *WFMAG

      ENDDO
!-MM- changes to accommodate spin spirals      
      QSP=-QSP
!-MM- end of addition
      ENDDO spinor
!=======================================================================
! sum the contributions to the force on the unit cell from each band
! multiplying by the
! k point weight and fermi-weight, by WDES%RSPIN for electron spins and
! and additional factor 2
!=======================================================================
      SCALE=  WDES%WTKPT(NK)*W%FERWE(N,NK,ISP)*2._q*WDES%RSPIN

      SIKEF(1,1)=SIKEF(1,1)+SIKE1 *SCALE
      SIKEF(2,2)=SIKEF(2,2)+SIKE2 *SCALE
      SIKEF(3,3)=SIKEF(3,3)+SIKE3 *SCALE
      SIKEF(1,2)=SIKEF(1,2)+SIKE12*SCALE
      SIKEF(2,3)=SIKEF(2,3)+SIKE23*SCALE
      SIKEF(3,1)=SIKEF(3,1)+SIKE31*SCALE
      SIKEF(2,1)=SIKEF(2,1)+SIKE12*SCALE
      SIKEF(3,2)=SIKEF(3,2)+SIKE23*SCALE
      SIKEF(1,3)=SIKEF(1,3)+SIKE31*SCALE

      ENDDO band_k
      ENDDO spin
! reduction of SIKEF
      CALLMPI( M_sum_d(WDES%COMM,SIKEF, 9))

      PROFILING_STOP('strkin')

      RETURN
      END SUBROUTINE


!******************** SUBROUTINE STRETAU *******************************
!
!***********************************************************************
      SUBROUTINE STRETAU( &
     &   HAMILTONIAN,KINEDEN,W,WDES,CSTRF, &
     &   GRID,GRID_SOFT,GRIDC,SOFT_TO_C,P,T_INFO,LATT_CUR,SYMM, &
     &   TAUSIF)
      USE prec
      USE base
      USE pseudo
      USE mgrid
      USE poscar
      USE lattice
      USE wave_high
      USE msymmetry
      USE mkpoints
      USE hamil_high
      USE meta
      IMPLICIT NONE
      TYPE(ham_handle) HAMILTONIAN
      TYPE(tau_handle) KINEDEN
      TYPE(wavedes) WDES
      TYPE(wavespin) W
      TYPE(type_info) T_INFO
      TYPE(latt) LATT_CUR
      TYPE(symmetry) SYMM
      TYPE(grid_3d) GRID
      TYPE(grid_3d) GRID_SOFT
      TYPE(grid_3d) GRIDC
      TYPE(transit) SOFT_TO_C
      TYPE(potcar) P(T_INFO%NTYP)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      REAL(q) TAUSIF(3,3)
      ! local variables
      TYPE(latt) LATT_PERTURBED
      INTEGER I,J,K,STEP
      INTEGER ISYM_STORE
      REAL(q) :: DIS=fd_displacement
      REAL(q) DEXC,DE

      PROFILING_START('stretau')

      TAUSIF=0

      IF ((.NOT.LDO_METAGGA()).OR.(.NOT.LCALCMU())) THEN
         PROFILING_STOP('stretau')
         RETURN
      ENDIF

      ISYM_STORE=SYMM%ISYM
      IF (SYMM%ISYM>0) SYMM%ISYM=0

      DO I=1,3
      DO J=1,3

      DEXC=0

      DO STEP=-1,1,2

         LATT_PERTURBED=LATT_CUR
         DO K=1,3
            LATT_PERTURBED%A(I,K)=LATT_CUR%A(I,K)+DIS*STEP*LATT_CUR%A(J,K)
         ENDDO
         CALL LATTIC(LATT_PERTURBED)

         CALL SET_KINEDEN(GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_PERTURBED,SYMM,T_INFO%NIONS,W,WDES,KINEDEN)

         CALL TAUPAR(GRIDC,T_INFO,LATT_PERTURBED%B,LATT_PERTURBED%OMEGA,P,CSTRF,KINEDEN%TAUC)

         IF (WDES%NCDIJ==4) CALL RL_FLIP(KINEDEN%TAU,GRIDC,WDES%NCDIJ,.TRUE.)
         CALL MUxTAU(GRIDC,WDES%NCDIJ,LATT_CUR,HAMILTONIAN%MUTOT,KINEDEN%TAU,KINEDEN%TAUC,DE)
         
         DEXC=DEXC+STEP*DE
      ENDDO

      TAUSIF(I,J)=TAUSIF(I,J)-DEXC/DIS/2._q

      ENDDO
      ENDDO

      ! restore initial state
      SYMM%ISYM=ISYM_STORE
      CALL SET_KINEDEN(GRID,GRID_SOFT,GRIDC,SOFT_TO_C,LATT_CUR,SYMM,T_INFO%NIONS,W,WDES,KINEDEN)
      CALL TAUPAR(GRIDC,T_INFO,LATT_CUR%B,LATT_CUR%OMEGA,P,CSTRF,KINEDEN%TAUC)

      PROFILING_STOP('stretau')

      RETURN
      END SUBROUTINE STRETAU


!************************ SUBROUTINE CHGGRA ****************************
!
! this subroutine calculates the gradient vector associated with
! Harris-functional
! i.e.
!  CHTOTL contains the input-chargedenisty
!  CHTOT  the output-chargedenisty
!  CHGGA  is the final result i.e
!         the difference between output and input charge density
!         multiplied by the derivative of the potential with respect
!         to changes in the charge density
!
!***********************************************************************

      SUBROUTINE CHGGRA(GRIDC,LATT_CUR, &
     &     CHGGA,CHTOT,CHTOTL,DENCOR)
      USE prec

      USE mpimy
      USE mgrid
      USE lattice
      USE setexm
      USE xcgrad
      USE charge
      USE pot
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      COMPLEX(q)  CHTOT(GRIDC%RC%NP),CHTOTL(GRIDC%RC%NP)
      COMPLEX(q)  CHGGA(GRIDC%MPLWV)
      RGRID       DENCOR(GRIDC%RL%NP)
! work arrays
      COMPLEX(q),ALLOCATABLE  :: CWORK1(:),CWORK2(:)
      COMPLEX(q), ALLOCATABLE :: DWORK2(:)
      REAL(q) TMPSIF(3,3)

      PROFILING_START('chggra')

      ALLOCATE (DWORK2(GRIDC%RL%NP),CWORK1(GRIDC%MPLWV),CWORK2(GRIDC%MPLWV))
!-----------------------------------------------------------------------
!  we need the  input-charge in real space so transform CHTOTL in
!  real space -> use here CHGGA as a work array ...
!-----------------------------------------------------------------------
      DO N=1,GRIDC%RC%NP
        CHGGA(N)=CHTOTL(N)
      ENDDO
      CALL FFT3D(CHGGA,GRIDC,1)
!-----------------------------------------------------------------------
!  calculate the gradient of the exchange correlation potential
!-----------------------------------------------------------------------
      CALL FEXCP(GRIDC,LATT_CUR%OMEGA, &
             CHGGA,DENCOR,CWORK1,DWORK2,CVZERO,EXC,XCENC,TMPSIF,.FALSE.)
!-----------------------------------------------------------------------
!  calculate difference between output and input charge-density in
!  real space -> store result in CHGGA
!-----------------------------------------------------------------------
      DO N=1,GRIDC%RC%NP
        CHGGA(N)=CHTOT(N)-CHTOTL(N)
      ENDDO
      CALL FFT3D(CHGGA,GRIDC,1)
!-----------------------------------------------------------------------
! multiply by gradient of xc-potential
! in real space and FFT this to reciprocal space
!-----------------------------------------------------------------------
      RINPL=1._q/GRIDC%NPLWV
      CALL RLR_MUL(CHGGA,DWORK2,RINPL/LATT_CUR%OMEGA,CHGGA,GRIDC)
      CALL FFT3D(CHGGA,GRIDC,-1)
      CALL SETUNB(CHGGA,GRIDC)
!-----------------------------------------------------------------------
! add hartree-term
!-----------------------------------------------------------------------
      DO N=1,GRIDC%RC%NP
        CWORK1(N)=CHTOT(N)-CHTOTL(N)
      ENDDO

      CALL POTHAR(GRIDC,LATT_CUR, CWORK1,CWORK2,DENC)

      DO N=1,GRIDC%RC%NP
       CHGGA(N)=CHGGA(N)+CWORK2(N)
      ENDDO
      DEALLOCATE (DWORK2,CWORK1,CWORK2)

      PROFILING_STOP('chggra')

      RETURN
      END SUBROUTINE


!************************ SUBROUTINE FORLOC ****************************
!
! this subroutine calculates the hellmann-feynman forces exerted on the
! ions by the electrons which equals the sum over reciprocal lattice
! vectors of  c.c. of the charge density at wavevector G
!  *IG*EXP(+IG.R)* pseudopotential at wavevector G
!
!***********************************************************************

      SUBROUTINE FORLOC(GRIDC,P,T_INFO,LATT_CUR, &
              CHTOT,EIFOR)
      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR

      COMPLEX(q) CHTOT(GRIDC%RC%NP)
      REAL(q)    EIFOR(3,T_INFO%NIONS)
! work arrays
      REAL(q), ALLOCATABLE :: WORK(:)

      PROFILING_START('forloc')

      ALLOCATE(WORK(GRIDC%RC%NP))

      NIS=1
!=======================================================================
      typ: DO NT=1,T_INFO%NTYP
!=======================================================================
       NIADD=T_INFO%NITYP(NT)
!=======================================================================
! interpolate the pseudopotential on the grid of reciprocal
! lattice-vectors
!=======================================================================
      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS

      ZZ=  -4*PI*P(NT)%ZVALF*FELECT

      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        SETFACT1
        SETFACT
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G=SQRT(GX**2+GY**2+GZ**2)*TPI
        IF (G/=0 .AND. G <PSGMA2) THEN
!=======================================================================
! convert the magnitude of the reciprocal lattice vector to a position
! in the pseudopotential arrays and interpolate the pseudopotential and
! its derivative
!=======================================================================
        I  =INT(G*ARGSC)+1
        REM=G-P(NT)%PSP(I,1)
        VPST =(P(NT)%PSP(I,2)+REM*(P(NT)%PSP(I,3)+ &
     &                     REM*(P(NT)%PSP(I,4)  +REM*P(NT)%PSP(I,5))))
        WORK (N)=( VPST+ ZZ / G**2) /LATT_CUR%OMEGA

        ELSE
          WORK(N)=0
        ENDIF

      ENDDO

      ion: DO NI=NIS,NIADD+NIS-1
!=======================================================================
! initialise the force on the ion to zero
!=======================================================================
         FOR1=0
         FOR2=0
         FOR3=0
!=======================================================================
! CGXDX,Y,Z = I* the changes in the phase factor g.r on moving one
! reciprocal lattice vector in the x,y,z directions, respectively
!=======================================================================
!=======================================================================
! calculate the total force on the ions by summing over reciprocal
! lattice vectors
! first calculate phase factor:
! there are two version for calculating the phase factor
! on vector machines you might try the first version
! (see stufak.F)
!=======================================================================
#ifdef vector
         DO NG=1,GRIDC%RC%NP
           N1= MOD((NG-1),GRIDC%RC%NROW) +1
           NC= (NG-1)/GRIDC%RC%NROW+1
           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)

           G1=T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           SETFACT1
           SETFACT
           CEXPF=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)
#else
         CX =EXP(-CITPI*T_INFO%POSION(1,NI))
         G1 =T_INFO%POSION(1,NI)*xmin(GRIDC%NGX)
         DO NC=1,GRIDC%RC%NCOL
           NGP=(NC-1)*GRIDC%RC%NROW+1

           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           CE=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)

           DO N1P=0,GRIDC%RC%NROW-1
           SETN1fromN1P
           NG=NGP+N1
           N1=N1+1

           SETFACT1
           SETFACT
           CEXPF=CE
           CE=CE*CX
#endif
!=======================================================================
! add the contribution to the force from the present reciprocal lattice
! vector  and multiply by i (ie take imaginary part)
!=======================================================================
           FOR=WORK(NG)*MULFACT AIMAG(CONJG(CHTOT(NG))*CEXPF)
           FOR1=FOR1-GRIDC%LPCTX_(N1)*FOR
           FOR2=FOR2-GRIDC%LPCTY_(N2)*FOR
           FOR3=FOR3-GRIDC%LPCTZ_(N3)*FOR
         ENDDO
#ifndef vector
         ENDDO
#endif
!=======================================================================
! multiply forces by 2*Pi
!=======================================================================
         EIFOR(1,NI)=FOR1*TPI
         EIFOR(2,NI)=FOR2*TPI
         EIFOR(3,NI)=FOR3*TPI

      ENDDO ion
      NIS=NIS+NIADD
      ENDDO typ

!=======================================================================
! forces are now in the reciprocal lattice transform it to
! cartesian coordinates
!=======================================================================
      CALLMPI( M_sum_d(GRIDC%COMM, EIFOR(1,1),T_INFO%NIONS*3))

      CALL  DIRKAR(T_INFO%NIONS,EIFOR,LATT_CUR%B)
      DEALLOCATE(WORK)

      PROFILING_STOP('forloc')

      RETURN
      END SUBROUTINE


!************************ SUBROUTINE FORHAR ****************************
!
! this subroutine calculates the correction to the
! hellman feynman forces
! +   if the harris-functional is used
!        CHGGA = gradient of Harris functional,
!           PSPRHO = atom. charge                LPAR=.FALSE.
! +   forces due to partial core corrections
!        CHGGA = CVXC(xc-potential),
!           PSPRHO=partial core                  LPAR=.TRUE.
!
!***********************************************************************


      SUBROUTINE FORHAR(GRIDC,P,T_INFO,LATT_CUR,CHGGA,HARFOR,LPAR)
      USE prec

      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      COMPLEX(q)   CHGGA(GRIDC%MPLWV)
      REAL(q)      HARFOR(3,T_INFO%NIONS)
      LOGICAL   LPAR
! work arrays
      REAL(q), POINTER :: PRHO(:)
      REAL(q), ALLOCATABLE :: WORK(:)

      PROFILING_START('forhar')

      ALLOCATE(WORK(GRIDC%RC%NP))

      HARFOR(1:3,1:T_INFO%NIONS)=0
!=======================================================================
! loop over all types of atoms
!=======================================================================
      NIS=1
      typ: DO NT=1,T_INFO%NTYP
      NIADD=T_INFO%NITYP(NT)
       IF (LPAR) THEN
         PRHO=>P(NT)%PSPCOR
       ELSE
         PRHO=>P(NT)%PSPRHO
       ENDIF
       IF (.NOT.ASSOCIATED(PRHO)) GOTO 200
!=======================================================================
! iterpolate the pseudopotential on the grid of reciprocal
! lattice-vectors
!=======================================================================
      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS*3

      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

        SETFACT1
        SETFACT
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G=SQRT(GX**2+GY**2+GZ**2)*TPI

        IF (G/=0 .AND.G<PSGMA2) THEN
!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
          ARG=(G*ARGSC)+1
          NADDR=MAX(INT(ARG),2)
          REM=ARG-NADDR
          V1=PRHO(NADDR-1)
          V2=PRHO(NADDR)
          V3=PRHO(NADDR+1)
          V4=PRHO(NADDR+2)
          T0=V2
          T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
          T2=(V1+V3-(2*V2))/2._q
          T3=(V4-V1+(3*(V2-V3)))/.6_q
          WORK(N)=T0+REM*(T1+REM*(T2+REM*T3))
        ELSE
          WORK(N)=0
        ENDIF

      ENDDO

      ion: DO NI=NIS,NIADD+NIS-1
!=======================================================================
! initialise the force on the ion to zero
!=======================================================================
         FOR1=0._q
         FOR2=0._q
         FOR3=0._q
!=======================================================================
! calculate the total force on the ions by summing over reciprocal
! lattice vectors
! first calculate phase factor:
! there are two version for calculating the phase factor
! on vector machines you might try the first version
! but usually the second version is much faster (less calls to CEXP)
!=======================================================================
#ifdef vector
         DO NG=1,GRIDC%RC%NP

           N1= MOD((NG-1),GRIDC%RC%NROW) +1
           NC= (NG-1)/GRIDC%RC%NROW+1
           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)

           G1=T_INFO%POSION(1,NI)*GRIDC%LPCTX(N1)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           SETFACT1
           SETFACT
           CEXPF=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)
#else
         CX =EXP(-CITPI*T_INFO%POSION(1,NI))
         G1 =T_INFO%POSION(1,NI)*xmin(GRIDC%NGX)
         DO NC=1,GRIDC%RC%NCOL
           NGP=(NC-1)*GRIDC%RC%NROW+1

           N2= GRIDC%RC%I2(NC)
           N3= GRIDC%RC%I3(NC)
           G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
           G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
           CE=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)

           DO N1P=0,GRIDC%RC%NROW-1
           SETN1fromN1P
           NG=NGP+N1
           N1=N1+1
           SETFACT1
           SETFACT
           CEXPF=CE
           CE=CE*CX
#endif
!=======================================================================
! add the contribution to the force from the present reciprocal lattice
! vector  and multiply by i (ie take imaginary part)
!=======================================================================
           FOR=WORK(NG)*MULFACT AIMAG(CONJG(CHGGA(NG))*CEXPF)
           FOR1=FOR1-GRIDC%LPCTX_(N1)*FOR
           FOR2=FOR2-GRIDC%LPCTY_(N2)*FOR
           FOR3=FOR3-GRIDC%LPCTZ_(N3)*FOR
         ENDDO
#ifndef vector
         ENDDO
#endif
!=======================================================================
! multiply forces by 2*Pi
!=======================================================================
         HARFOR(1,NI)=FOR1*TPI
         HARFOR(2,NI)=FOR2*TPI
         HARFOR(3,NI)=FOR3*TPI

      ENDDO ion
  200 NIS=NIS+NIADD
      ENDDO typ
!=======================================================================
! forces are now in the reciprocal lattice transform it to
! cartesian coordinates
!=======================================================================
      CALLMPI( M_sum_d(GRIDC%COMM, HARFOR(1,1),T_INFO%NIONS*3))
      CALL  DIRKAR(T_INFO%NIONS,HARFOR,LATT_CUR%B)
      DEALLOCATE(WORK)

      PROFILING_STOP('forhar')

      RETURN
      END SUBROUTINE

!************************ SUBROUTINE CHARGEDER *************************
!
! this subroutine calculates the first derivative of the charge density
! with respect to the ionic positions ION in the cartesian direction
! IDIR
!
! LPAR determines whether the derivative of the partical core corrections
!      of that of the atomic charge is determined
! ION  index of ion for which derivative is calculated
! IDIR cartesian index for which derivative is determined
!
!***********************************************************************


      SUBROUTINE CHARGEDER(GRIDC,P,T_INFO,LATT_CUR,CHGGA,LPAR,ION,IDIR)
      USE prec

      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE constant

      IMPLICIT NONE

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      COMPLEX(q)         CHGGA(GRIDC%MPLWV)
      LOGICAL            LPAR
      INTEGER            ION
      INTEGER            IDIR
! work arrays
      REAL(q), POINTER :: PRHO(:)
      INTEGER  :: NI, NT, N, N1, N1P, NC, N2, N3, NADDR, NGP, NG
      REAL(q)  :: ARGSC,PSGMA2, G1, G2, G3, GX, GY, GZ, G 
      REAL(q)  :: ARG, REM, V1, V2, V3, V4, T0, T1, T2, T3
      COMPLEX(q) :: CE, CX

      PROFILING_START('chargeder')
!=======================================================================
! loop over all types of atoms
!=======================================================================
      NI =ION
      NT =T_INFO%ITYP(NI)
      IF (LPAR) THEN
         PRHO=>P(NT)%PSPCOR
      ELSE
         PRHO=>P(NT)%PSPRHO
      ENDIF
      CHGGA=0
      IF (.NOT.ASSOCIATED(PRHO)) THEN
         PROFILING_STOP('chargeder')
         RETURN
      ENDIF
!=======================================================================
! iterpolate the pseudopotential on the grid of reciprocal
! lattice-vectors
!=======================================================================
      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS*3

      DO N=1,GRIDC%RC%NP
        N1= MOD((N-1),GRIDC%RC%NROW) +1
        NC= (N-1)/GRIDC%RC%NROW+1
        N2= GRIDC%RC%I2(NC)
        N3= GRIDC%RC%I3(NC)

!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3)

        G=SQRT(GX**2+GY**2+GZ**2)*TPI

        IF (G/=0 .AND.G<PSGMA2) THEN
!=======================================================================
! convert the magnitude of the reciprocal latice vector to a position
! in the charge-dens. array  and interpolate the atomic-chargedensity
!=======================================================================
          ARG=(G*ARGSC)+1
          NADDR=MAX(INT(ARG),2)
          REM=ARG-NADDR
          V1=PRHO(NADDR-1)
          V2=PRHO(NADDR)
          V3=PRHO(NADDR+1)
          V4=PRHO(NADDR+2)
          T0=V2
          T1=((6*V3)-(2*V1)-(3*V2)-V4)/6._q
          T2=(V1+V3-(2*V2))/2._q
          T3=(V4-V1+(3*(V2-V3)))/.6_q
          CHGGA(N)=T0+REM*(T1+REM*(T2+REM*T3))
        ELSE
          CHGGA(N)=0
        ENDIF

      ENDDO
!=======================================================================
! now multiply with the derivative of the form factor
!
!   d rho / d R_i = d / d R_i e (-i G R)  rho ( G ) =
!                  - i G_i e (-i G R) rho (G)
!
! where G_i is the cartesian component i of the lattice vector G
!=======================================================================

      CX =EXP(-CITPI*T_INFO%POSION(1,NI))
      G1 =T_INFO%POSION(1,NI)*xmin(GRIDC%NGX)

      DO NC=1,GRIDC%RC%NCOL
         NGP=(NC-1)*GRIDC%RC%NROW+1

         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         G2=T_INFO%POSION(2,NI)*GRIDC%LPCTY(N2)
         G3=T_INFO%POSION(3,NI)*GRIDC%LPCTZ(N3)
         CE=EXP(-CITPI*(G3+G2+G1))*T_INFO%VCA(NT)

         DO N1P=0,GRIDC%RC%NROW-1
            SETN1fromN1P
            NG=NGP+N1
            N1=N1+1
            GX= (GRIDC%LPCTX(N1)*LATT_CUR%B(IDIR,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(IDIR,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(IDIR,3))
            CHGGA(NG)=-CHGGA(NG)*CITPI*CE*GX
            CE=CE*CX
         ENDDO
      ENDDO

      PROFILING_STOP('chargeder')

      RETURN
      END SUBROUTINE

      END MODULE

!************************ SUBROUTINE FORCE_AND_STRESS ******************
! 
!  driver routine for force and stress calculations, 
!  as well as writing the force and stress to the OUTCAR
!  file
!  upon entry the charge density must be selfconsistenly determined
!  (CHTOT) and the potential must have been set accordingly CVTOT
!
!  additionally the following components of the stress tensor/force must
!  be calculated *before* calling the routine
!    XCSIF (exchange correlation)
!    EWSIF/ EWIFOR (Ewald)
!
!  upon return the following quantities are set
!    TSIF/ TIFOR
!
! NOTE: upon exit the CVTOT array and the CDIJ arrays have been
!  destroyed and must be recalculated from CHTOT
!
!***********************************************************************

  SUBROUTINE FORCE_AND_STRESS( &
          KINEDEN,HAMILTONIAN,P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
          T_INFO,T_INFO_0,DYN,INFO,IO,MIX,SYMM,GRID,GRID_SOFT, &
          GRIDC,GRIDB,GRIDUS,C_TO_US,B_TO_C,SOFT_TO_C, &
          CHTOT,CHTOTL,DENCOR,CVTOT,CSTRF, &
          CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM,RHOLM_LAST, &
          CHDEN,SV, &
          LMDIM, IRDMAX, LWRITE_FORCE, LSTRESS, LWRITE_STRESS, LREMOVE_DRIFT,  &
          XCSIF, EWSIF, TSIF, EWIFOR, TIFOR, PRESS ,TOTEN, KPOINTS)

    USE base
    USE lattice
    USE charge
    USE pseudo
    USE lattice
    USE force    
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
    USE mdipol
    USE fock
    USE ini
    USE us
    USE broyden
    USE core_rel
    USE hamil_high
    USE meta
    USE us
    USE morbitalmag
    USE classicfields
!tb start
    USE vdwforcefield
    USE fileio
    USE mkpoints
!tb end
! solvation__
    USE solvation
! solvation__
    IMPLICIT NONE
!=======================================================================
!  structures
!=======================================================================
    TYPE (tau_handle)  KINEDEN
    TYPE (ham_handle)  HAMILTONIAN
    TYPE (type_info)   T_INFO
    TYPE (type_info)   T_INFO_0
    TYPE (potcar)      P(T_INFO%NTYP)
    TYPE (wavedes)     WDES
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin)    W          ! wavefunction
    TYPE (latt)        LATT_CUR
    TYPE (dynamics)    DYN
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    TYPE (mixing)      MIX
    TYPE (symmetry)    SYMM
    TYPE (grid_3d)     GRID       ! grid for wavefunctions
    TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
    TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
    TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
    TYPE (grid_3d)     GRIDB      ! Broyden grid
    TYPE (transit)     B_TO_C     ! index table between GRIDB and GRIDC
    TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
    TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
    
    INTEGER LMDIM,IRDMAX

    COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
    COMPLEX(q)  CHTOTL(GRIDC%MPLWV,WDES%NCDIJ)! old charge-density
    RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
    COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
    COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)

!  augmentation related quantities
    OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
             CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  paw sphere charge density
    INTEGER N_MIX_PAW
    REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ),RHOLM_LAST(N_MIX_PAW,WDES%NCDIJ)
!  charge-density and potential on soft grid
    COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
    RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
    REAL(q)   XCSIF(3,3)                      ! stress stemming from XC
    REAL(q)   EWSIF(3,3)                      ! stress from Ewald contribution
    REAL(q)   TSIF(3,3)                       ! total stress (set by routine)
    REAL(q)   EWIFOR(3,T_INFO%NIOND)          ! ewald force
    REAL(q)   OFIELD_E                        ! energy contribution from order field
    REAL(q)   OFIELD_FOR(3,T_INFO%NIOND)      ! forces from order field
    REAL(q)   TIFOR(3,T_INFO%NIOND)           ! total force (set by routine)
    REAL(q)   PRESS                           ! external pressure
    LOGICAL   LSTRESS, LWRITE_STRESS, LWRITE_FORCE, LREMOVE_DRIFT
! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL

!  local work arrays for individual force components
    REAL(q)   SIKEF(3,3),EISIF(3,3),DSIF(3,3), &
              PSCSIF(3,3),FNLSIF(3,3),AUGSIF(3,3), &
              PARSIF(3,3),FOCKSIF(3,3), &
              TAUSIF(3,3)
    REAL(q)   EIFOR(3,T_INFO%NIOND),EINL(3,T_INFO%NIOND), &
              HARFOR(3,T_INFO%NIOND),PARFOR(3,T_INFO%NIOND),FORHF(3,T_INFO%NIOND), &
              TAUFOR(3,T_INFO%NIOND)
    REAL(q)   TFORNL(3),TEIFOR(3),TEWIFO(3),THARFO(3),TFORHF(3),VTMP(3)

    REAL(q)      RMST,RMSC,RMSP,WEIGHT ! mixer 
    INTEGER      IERRBR            ! mixer
    REAL(q) FAKT                             ! scaling factor to kBar
    REAL(q) DISPL(3,T_INFO%NIONS)
    INTEGER ISP, I, J
    TYPE (energy) ETMP
    INTEGER :: IRDMAA
    LOGICAL,EXTERNAL :: USEFOCK_CONTRIBUTION
    REAL(q) :: TOTEN
! tb 
   REAL(q),ALLOCATABLE,SAVE ::  RELVOL(:)
    REAL(q),ALLOCATABLE ::  XDM(:)
    REAL(q),ALLOCATABLE,SAVE ::  Overlap(:,:),M2(:),RELCHG(:),HCD(:)
    LOGICAL,SAVE :: LOPEN_VDW=.TRUE.
    LOGICAL,SAVE :: LVDW=.FALSE.,LRELVOL=.TRUE.
    INTEGER,SAVE :: IVDW=1
    COMPLEX(q), ALLOCATABLE ::  CHTOT1(:,:), CHTOT2(:,:) ! charge-density in real / reciprocal space
    COMPLEX(q), ALLOCATABLE ::  CHDEN1(:,:)
    !COMPLEX(q),  ALLOCATABLE ::  CWORK1(:),CWORK2(:)
    RGRID,  ALLOCATABLE ::  CWORK1(:),CWORK2(:)
    INTEGER ::  NALLOC
    INTEGER :: NI,NT,NIS
    INTEGER,ALLOCATABLE :: REFSTATE(:)
    REAL(q) :: VDWFOR(3,T_INFO%NIOND)           ! vdw force (set by routine)
    REAL(q) :: VDWSIF(3,3)
    REAL(q) :: VDWEN
    TYPE (kpoints_struct), OPTIONAL :: KPOINTS

    PROFILING_START('force_and_stress')

    TIFOR=0
    TSIF=0
    IF (DYN%ISIF<0) THEN
       PROFILING_STOP('force_and_stress')
       RETURN   ! quick return if forces are not required
    ENDIF

!tb beg
    !c is some approximate dispersion corretion method used?
    IF (LOPEN_VDW) THEN
         !LOPEN_VDW=.FALSE.
         ALLOCATE(REFSTATE(T_INFO%NTYP))
         REFSTATE=0
         CALL vdw_read(IO,LVDW,IVDW,LRELVOL,T_INFO%NTYP,REFSTATE)
         IF (LVDW) ALLOCATE(RELVOL(T_INFO%NIONS))
         IF (IVDW.eq.4) ALLOCATE(Overlap(T_INFO%NIONS,T_INFO%NIONS))
         IF (IVDW.eq.4) ALLOCATE(M2(T_INFO%NIONS),RELCHG(T_INFO%NIONS),HCD(T_INFO%NIONS))
    ENDIF
    
    !c quick return if forces are not required
    !c ...but only if no vdW correction is used
    IF (.NOT. LVDW) THEN
      IF (DYN%ISIF<0) THEN
         PROFILING_STOP('force_and_stress')
         RETURN   
      ENDIF
    ENDIF
!tb end

!-----------------------------------------------------------------------
!  first set CHTOTL to CHTOT
!  if charge-density remains constant during electronic minimization
!  or Harris corrections to forces are calculates
!  set CHTOT to the chargedensity derived from the current
!  wavefunctions
!-----------------------------------------------------------------------
      CALL START_TIMING("G")

      DO ISP=1,WDES%NCDIJ
         CALL RC_ADD(CHTOT(1,ISP),1.0_q,CHTOT(1,ISP),0.0_q,CHTOTL(1,ISP),GRIDC)
      ENDDO
      IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
         DO ISP=1,WDES%NCDIJ
            CALL RC_ADD(KINEDEN%TAU(1,ISP),1.0_q,KINEDEN%TAU(1,ISP),0.0_q,KINEDEN%TAUL(1,ISP),GRIDC)
         ENDDO
      ENDIF
      RHOLM_LAST=RHOLM

      IF (INFO%LCHCON .OR. INFO%LCORR) THEN
         CALL SET_CHARGE(W, WDES, INFO%LOVERL, &
              GRID, GRIDC, GRID_SOFT, GRIDUS, C_TO_US, SOFT_TO_C, &
              LATT_CUR, P, SYMM, T_INFO, &
              CHDEN, LMDIM, CRHODE, CHTOT, RHOLM, N_MIX_PAW, IRDMAX)

         CALL STOP_TIMING("G",IO%IU6,'CHARGE')
      ENDIF
!----------------------- FORCES ON IONS    -----------------------------
! calculate the Hellmann-Feynmann forces exerted on the ions
! FORLOC local part
! FORNL  forces due to the non-local part
! FORDEP forces due to augmentation charges
!-----------------------------------------------------------------------
      EIFOR=0; EINL=0; HARFOR=0; PARFOR=0; TAUFOR=0

!     local contribution to force
      CALL START_TIMING("G")
      CALL FORLOC(GRIDC,P,T_INFO,LATT_CUR, CHTOT,EIFOR)
      CALL STOP_TIMING("G",IO%IU6,'FORLOC')


      DISPL(:,1:T_INFO%NIONS)=T_INFO%POSION(:,1:T_INFO%NIONS)-T_INFO_0%POSION(:,1:T_INFO%NIONS)
      CALL DIRKAR(T_INFO%NIONS,DISPL(1,1),LATT_CUR%A)

!     fock part of the forces
      FORHF=0
      FOCKSIF=0
      IF (USEFOCK_CONTRIBUTION()) THEN
         CALL KPAR_SYNC_WAVEFUNCTIONS(WDES,W)

         CALL FOCK_FORCE(GRID, LATT_CUR, W,  LMDIM, CQIJ, &
            NONLR_S, NONL_S, P, FORHF, FOCKSIF, LSTRESS, CDIJ, SV, IO%IU0, IO%IU6)
         CALL STOP_TIMING("G",IO%IU6,'FORHF ')
      ENDIF
      IF (SYMM%ISYM>0) &
           CALL FORSYM(FORHF,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)


      IF (INFO%LREAL) THEN
        NONLR_S%POSION => T_INFO_0%POSION
        CALL FORNLR(GRID,NONLR_S,P,LATT_CUR,W, &
     &      CDIJ, CQIJ, DISPL, EINL)
        NONLR_S%POSION => T_INFO%POSION
      ELSE
        CALL FORNL(NONL_S,WDES,W,LATT_CUR, LMDIM,CDIJ,CQIJ,EINL)
      ENDIF
!     force from augmentation part
      IF (INFO%LOVERL) THEN
      CALL FORDEP(WDES, GRIDC,GRIDUS,C_TO_US, &
         LATT_CUR,P,T_INFO_0, INFO%LOVERL, &
         LMDIM, CDIJ, CQIJ, CRHODE, CVTOT, IRDMAX, DISPL, EINL)
      ENDIF

      IF (SYMM%ISYM>0) &
           CALL FORSYM(EINL,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)
      CALL STOP_TIMING("G",IO%IU6,'FORNL ')

      ! must restore CDIJ at this point, since FORDEP destroys it
      CALL SETDIJ_(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO_0,INFO%LOVERL, &
           LMDIM,CDIJ,CQIJ,CVTOT,.TRUE.,IRDMAA,IRDMAX, DISPL)

      CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                  LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)

      CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)

      CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
           WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
           ETMP,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )


!------------------- STRESS ON UNIT CELL -------------------------------
! calculate the stress on the unit cell which is related
! to the change in local pseudopotential on changing the size of
! the cell
! then calculate the non-local contribution to the stress
!-----------------------------------------------------------------------
      IF (LSTRESS) THEN
         AUGSIF=0; FNLSIF=0; TAUSIF=0

         CALL START_TIMING("G")

! kinetic energy
         CALL STRKIN(W,WDES, LATT_CUR%B,SIKEF)
! local part
         CALL STRELO(GRIDC,P,T_INFO,LATT_CUR, &
              CHTOT,CSTRF, INFO%NELECT, DSIF,EISIF,PSCSIF)
! non-local part
         IF (INFO%LREAL) THEN
            CALL STRNLR(GRID,NONLR_S,P,LATT_CUR,W, &
                 &      CDIJ,CQIJ, DYN%ISIF,FNLSIF)
         ELSE
            CALL STRENL(GRID,NONL_S,P,W,WDES,LATT_CUR, &
                 LMDIM,CDIJ,CQIJ, DYN%ISIF,FNLSIF)
         ENDIF
#ifdef debug
         WRITE(*,'("FNLSIF",3F14.7)') FNLSIF
#endif
! augmentation part
         IF (INFO%LOVERL) &
              CALL STRDEP(WDES, GRIDC,GRIDUS,C_TO_US, &
              LATT_CUR,P,T_INFO, INFO%LOVERL, &
              LMDIM,CDIJ,CQIJ,CRHODE, CVTOT, IRDMAX, DYN%ISIF,AUGSIF)
#ifdef debug
         WRITE(*,'("AUGSIF",3F14.7)') AUGSIF
#endif
         IF (LDO_METAGGA().AND.LCALCMU()) &
              CALL STRETAU(HAMILTONIAN,KINEDEN,W,WDES,CSTRF, &
             &   GRID,GRID_SOFT,GRIDC,SOFT_TO_C,P,T_INFO,LATT_CUR,SYMM,TAUSIF)
#ifdef debug
         WRITE(*,'("TAUSIF",3F14.7)') TAUSIF
#endif
         CALL STOP_TIMING("G",IO%IU6,'STRESS')

      ENDIF

!-----------------------------------------------------------------------
! additional forces from partial core corrections
! remark for stress:
! stress deriving from the 1/volume dependency is treated in POTEX
! (this is necessary due to gradient corrections)
!-----------------------------------------------------------------------
      IF (INFO%LCORE.OR.IVDW.eq.4) THEN
         IF (LDO_METAGGA().AND.LCALCMU()) THEN
            CALL POTXC_METAGGA(KINEDEN,GRID,GRIDC,GRID_SOFT,SOFT_TO_C,WDES,LATT_CUR, &
           &   CHDEN,CHTOT,DENCOR,CVTOT,HAMILTONIAN%MUTOT)
            ! compute F_tau,i = \int \mu d\tau_c/dR_i dr
            CALL FORTAU(GRIDC,P,T_INFO,LATT_CUR,HAMILTONIAN%MUTOT,TAUFOR)
         ELSE
            IF (IVDW.eq.4) THEN
              ALLOCATE(XDM(GRIDC%RL%NP))
              CALL POTXC_ddsc(GRIDC,INFO,WDES, LATT_CUR, CVTOT(1,1),CHTOT(1,1),DENCOR,IVDW,XDM)
            ELSE
              CALL POTXC(GRIDC,INFO,WDES, LATT_CUR, CVTOT(1,1),CHTOT(1,1),DENCOR)
            ENDIF
         ENDIF

         CALL FORHAR(GRIDC,P,T_INFO,LATT_CUR,CVTOT,PARFOR,.TRUE.)

         IF (DYN%ISIF/=0) THEN
            CALL STREHAR(GRIDC,P,T_INFO,LATT_CUR,.TRUE.,CVTOT,CSTRF, PARSIF)
            XCSIF=XCSIF+PARSIF
            IF (LDO_METAGGA().AND.LCALCMU()) XCSIF=XCSIF+TAUSIF
         ENDIF

         INFO%LPOTOK=.FALSE.
         CALL STOP_TIMING("G",IO%IU6,'FORCOR')
      ENDIF

!-----------------------------------------------------------------------
! additional forces and stress for Harris-functional
! with moving atomic charge-densities
! this is only correct for paramagnetic LDA calculations
! but because Hartree term accounts for 90 % its almost ok
!-----------------------------------------------------------------------
      IF (INFO%LCHCON.OR.INFO%LCORR) THEN

         CALL CHGGRA(GRIDC,LATT_CUR, CVTOT,CHTOT,CHTOTL,DENCOR)
         
         CALL FORHAR(GRIDC,P,T_INFO,LATT_CUR, &
              CVTOT,HARFOR,.FALSE.)

         CALL FORHAR_GAUSSIAN(GRIDC,P,T_INFO,LATT_CUR, &
              CVTOT,HARFOR,.FALSE.)

         CALL STOP_TIMING("G",IO%IU6,'FORHAR')
         INFO%LPOTOK=.FALSE.

      ENDIF
!-----------------------------------------------------------------------
!    if Harris Foulkes corrections are calculated and if mixing is selected
!    mix now (this improves extrapolation of charge)
!    in all other cases set CHTOT back to CHTOTL
!-----------------------------------------------------------------------
      IF (INFO%LCORR .AND. .NOT. INFO%LCHCON  .AND. MIX%IMIX/=0 .AND. .NOT. MIX%MIXFIRST ) THEN
         CALL START_TIMING("G")
         IF (MIX%IMIX==4) THEN
!  broyden mixing ... :
            IF (DYN%IBRION/=10) THEN
               IF (LCORREL()) THEN
                  CALL BRMIX(KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
                      (2*GRIDC%MPLWV),CHTOT,CHTOTL,WDES%NCDIJ,LATT_CUR%B, &
                      LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                      RMST,RMSC,RMSP,WEIGHT,.TRUE.,IERRBR, &
                      NMIX_ONE_CENTRE,RHO_ONE_CENTRE,RHO_ONE_CENTRE_LAST)
               ELSE
                  CALL BRMIX(KINEDEN,GRIDB,GRIDC,IO,MIX,B_TO_C, &
                      (2*GRIDC%MPLWV),CHTOT,CHTOTL,WDES%NCDIJ,LATT_CUR%B, &
                      LATT_CUR%OMEGA, N_MIX_PAW, RHOLM, RHOLM_LAST, &
                      RMST,RMSC,RMSP,WEIGHT,.TRUE.,IERRBR)
               ENDIF
               IF (IERRBR/=0) THEN
                  IF (IO%IU0>=0) &
                       WRITE(IO%IU0,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
                       &                 'mixing'' now and reset mixing at next step!'
                  WRITE(IO%IU6,*) 'ERROR: Broyden mixing failed, tried ''simple '// &
                       &                 'mixing'' now and reset mixing at next step!'
               ENDIF
            ENDIF
         ELSE
!  simple mixing
            RMST=0
            CALL MIX_SIMPLE(GRIDC,MIX,WDES%NCDIJ, CHTOT,CHTOTL, &
                 N_MIX_PAW, RHOLM, RHOLM_LAST, LATT_CUR%B, LATT_CUR%OMEGA, RMST)
         ENDIF
         CALL STOP_TIMING("G",IO%IU6,'MIXING')
      ELSE
!  all other cases: restore CHTOT from CHTOTL
         DO ISP=1,WDES%NCDIJ
            CALL RC_ADD(CHTOTL(1,ISP),1.0_q,CHTOTL(1,ISP),0.0_q,CHTOT(1,ISP),GRIDC)
         ENDDO
!  and TAU from TAUL in case we run a metaGGA
         IF (LDO_METAGGA().AND.LMIX_TAU()) THEN
            DO ISP=1,WDES%NCDIJ
               CALL RC_ADD(KINEDEN%TAUL(1,ISP),1.0_q,KINEDEN%TAUL(1,ISP),0.0_q,KINEDEN%TAU(1,ISP),GRIDC)
            ENDDO
         ENDIF
         RHOLM=RHOLM_LAST
      ENDIF
!-----------------------------------------------------------------------
! ) sum total force on cell
! ) sum the total force on the ions
! ) remove spurios drift SYMVEC, and symmetrisation of forces
!-----------------------------------------------------------------------

      TSIF=0
      IF (DYN%ISIF/=0) THEN
         TSIF=SIKEF+EWSIF+EISIF+PSCSIF+XCSIF+DSIF+FNLSIF+AUGSIF+FOCKSIF
      ENDIF
      IF ( DIP%IDIPCO >0 ) THEN
         EIFOR=EIFOR+DIP%FORCE
      ENDIF

      TIFOR=EIFOR+EWIFOR+EINL+HARFOR+PARFOR+FORHF+TAUFOR

! external order field (see paricorrection.F file)
      CALL OFIELD(GRIDC%COMM, IO, LATT_CUR, DYN, T_INFO, OFIELD_FOR, OFIELD_E)
      TIFOR=TIFOR+OFIELD_FOR
      TOTEN=TOTEN+OFIELD_E
      CALL STOP_TIMING("G",IO%IU6,'OFIELD')

! solvation__
      TIFOR = TIFOR + EIFOR_SOL
! solvation__

!-----------------------------------------------------------------------
!tb start
! approximate vdW methods:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         IF (LVDW) THEN
           !c IVDW methods are not supported for DF-PT calculations!
           IF (DYN%IBRION==7 .OR. DYN%IBRION==8) THEN
             IF (IO%IU0>=0) THEN
               WRITE(IO%IU0,*)'ERROR: approximate vdW methods are not implemented for DF-PT.'
               WRITE(IO%IU0,*) 'Use finite differences (IBRION=5|6) instead.'
             ENDIF
             STOP
           ENDIF

           CALL START_TIMING("G")
           IF (LRELVOL .OR. LOPEN_VDW) THEN
             !CALL START_TIMING("G")
             !c effective volume for Tkatcheno-Schoeffler method
             RELVOL=0._q
             !c Tkatchenko-Scheffler method
             !IF (IVDW==2 .OR. IVDW==3 .OR. IVDW==4 .OR. IVDW==5 .OR. IVDW==33 .OR. IVDW==22) THEN
             IF (IVDW==2 .OR. IVDW==4 .OR. (IVDW>=20 .AND. IVDW<=22) .OR. IVDW==200 .OR. IVDW==210 & 
             &   .OR. IVDW==202 .OR. IVDW==212) THEN
               !write(*,*) "GRIDUS%NGPTAR",GRIDUS%NGPTAR
               !write(*,*) "GRIDC%NGPTAR",GRIDC%NGPTAR

               !c ADDGRID is not supported for IVDW=2xx
               IF ((GRIDUS%NGPTAR(1)/=GRIDC%NGPTAR(1)) .OR. (GRIDUS%NGPTAR(2)/=GRIDC%NGPTAR(2)) &  
                 .OR. (GRIDUS%NGPTAR(3)/=GRIDC%NGPTAR(3))) THEN
                 IF (IO%IU0>=0) THEN
                   WRITE(IO%IU0,*)'ERROR: ADDGRID=.TRUE. is not supported for TS correction methods.'
                 ENDIF
                 STOP
               ENDIF


               ALLOCATE(CHTOT1(GRIDC%MPLWV,WDES%NCDIJ), CHTOT2(GRIDC%MPLWV,WDES%NCDIJ))
               ALLOCATE(CHDEN1(GRID_SOFT%MPLWV,WDES%NCDIJ))

               !write(*,*) "GRIDC%NGX,GRIDC%NGY",GRIDC%NGX,GRIDC%NGY
               !NALLOC=GRIDC%NGX*GRIDC%NGY
               NALLOC=DIMREAL(GRIDC%MPLWV)
               ALLOCATE(CWORK1(DIMREAL(GRIDC%MPLWV)),CWORK2(DIMREAL(GRIDC%MPLWV)))

               ! overlapping atomic charges to AECCAR1
               ! add overlapping atomic charges on dense regular grid
               CHTOT1=0;CHDEN1=0
               ! TB: one of the following two calls  doesn't work if ADDGRID==.TRUE. !!!
               CALL RHOATO_WORK(.TRUE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CHTOT1)

               ! add (n_ae - n_ps - n_comp) as defined on radial grid
               CALL AUGCHG(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
               &             LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
               &              LMDIM,CRHODE, CHTOT1,CHDEN1, IRDMAX,.FALSE.,.FALSE.)

              ! core part
               CHTOT2=0
               IF (INFO%LCORE) THEN
                 CALL RHOATO_WORK(.TRUE.,.TRUE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CHTOT2)
               ENDIF
               ! add core densities (nc_ae-nc_ps) as defined on radial grid
               CALL AUGCHG(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
               &             LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
               &              LMDIM,CRHODE, CHTOT2,CHDEN1, IRDMAX,.FALSE.,.TRUE.)


               CALL RC_ADD(CHTOT1,1.0_q,CHTOT2,1.0_q,CWORK1,GRIDC)
               CALL FFT3D(CWORK1,GRIDC,1)

               ! AE-charge density
               CALL AUGCHG(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
               &             LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
               &              LMDIM,CRHODE, CHTOT,CHDEN, IRDMAX,.TRUE.,.FALSE.)

               CALL RC_ADD(CHTOT,1.0_q,CHTOT2,1.0_q,CWORK2,GRIDC)
               CALL FFT3D(CWORK2,GRIDC,1)

               ! and restore the total charge density
               CALL DEPLE(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
                    LATT_CUR,P,T_INFO,SYMM, INFO%LOVERL, SOFT_TO_C,&
                    LMDIM,CRHODE, CHTOT,CHDEN, IRDMAX)

               !c use standard Hirshfeld partitioning
               IF ( IVDW==2 .OR. IVDW==20 .OR. IVDW==200 .OR. IVDW==202) CALL weight_charge(GRIDC, IO,DYN,T_INFO,INFO,&
               &  LATT_CUR,P,  RELVOL,NALLOC,(CWORK1),(CWORK2),REFSTATE)

               !c use iterative Hirshfeld-I partitioning
               IF (IVDW==21 .OR. IVDW==210 .OR. IVDW==212) CALL hirshfeld_iterative(GRIDC, IO,DYN,T_INFO, INFO, LATT_CUR,P,  &
               &    CWORK1,CWORK2,RELVOL,IVDW)

               !c use iterative Hirshfeld-I partitioning within frozen orbital approximation
               !c for ionic state
               IF (IVDW==22 ) CALL hirshfeld_iterative(GRIDC, IO,DYN,T_INFO, INFO, LATT_CUR,P,  &
               &    CWORK1,CWORK2,RELVOL,IVDW)

               !c use iterative iterative Hirshfeld-dominant partitioning
               IF (IVDW==4) CALL hirshfeld_dominant(GRIDC, IO,DYN,T_INFO,INFO,&
               &  LATT_CUR,P,  RELVOL,NALLOC,(CWORK1),(CWORK2),REFSTATE,WDES%NCDIJ,XDM,RELCHG,Overlap,M2,HCD)
               IF (IVDW.eq.4) DEALLOCATE(XDM)

               !c use iterative iterative Hirshfeld-dominant partitioning
               !IF (IVDW==5) CALL hirschfeld_dominant_iterative(GRIDC, IO,DYN,T_INFO, LATT_CUR,P,  &
               !&    NALLOC,DIMREAL(GRIDC%MPLWV),CWORK1,CWORK2,RELVOL,.FALSE.)

               DEALLOCATE(CWORK2,CWORK1)
               DEALLOCATE(CHDEN1)
               DEALLOCATE(CHTOT2,CHTOT1)
             ENDIF
           ENDIF

           VDWSIF=0._q
           VDWFOR=0._q
           VDWEN=0._q
           !CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,P%ELEMENT,RELVOL,IVDW,REFSTATE)
           IF (PRESENT(KPOINTS)) THEN
             !CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,VDWSIF,VDWFOR,VDWEN,P%ELEMENT,& 
             !& RELVOL,IVDW,REFSTATE,KPOINTS)
             CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,VDWSIF,VDWFOR,VDWEN,P%ELEMENT,& 
             & RELVOL,IVDW,REFSTATE,Overlap,M2,RELCHG,HCD,KPOINTS)
           ELSE
             !CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,VDWSIF,VDWFOR,VDWEN,P%ELEMENT,& 
             !& RELVOL,IVDW,REFSTATE)
             CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,VDWSIF,VDWFOR,VDWEN,P%ELEMENT, &
             & RELVOL,IVDW,REFSTATE,Overlap,M2,RELCHG,HCD)
           ENDIF

           !!c perform symmetrization of vdW forces and stresses 
           !IF (SYMM%ISYM>0) THEN
           !  CALL FORSYM(VDWFOR,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)
           !  IF (DYN%ISIF/=0) CALL TSYM(VDWSIF,ISYMOP,NROTK,LATT_CUR%A)
           !ENDIF



           !c update stress and forces
           TSIF=TSIF+VDWSIF
           TIFOR=TIFOR+VDWFOR
           TOTEN=TOTEN+VDWEN

           !CALL STOP_TIMING("G",IO%IU6,'FORVDW')
           IF (LVDW) CALL STOP_TIMING("G",IO%IU6,'FORVDW')

           IF (LOPEN_VDW) THEN
             LOPEN_VDW=.FALSE.
           ENDIF
         ENDIF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!tb end
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
! symmetrization of forces and stress tensor
      IF (SYMM%ISYM>0) THEN
         CALL FORSYM(TIFOR,SYMM%ROTMAP(1,1,1),T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIOND,SYMM%TAUROT,SYMM%WRKROT,LATT_CUR%A)
         IF (DYN%ISIF/=0) CALL TSYM(TSIF,ISYMOP,NROTK,LATT_CUR%A)
      ENDIF

      IF (DYN%IBRION/=0 .OR. LCOMPAT) THEN
! remove drift from the forces
         IF (LREMOVE_DRIFT) CALL SYMVEC(T_INFO%NIONS,TIFOR)
      ENDIF

     ! average STRESS add Pullay term
      PRESS=(TSIF(1,1)+TSIF(2,2)+TSIF(3,3))/3._q &
           &      -DYN%PSTRESS/(EVTOJ*1E22_q)*LATT_CUR%OMEGA
!tb end

      ! for sake of consistency forward forces from central node to all nodes
      CALLMPI ( M_bcast_d(WDES%COMM, TIFOR, T_INFO%NIONS*3))
      ! and the stress as well
      CALLMPI ( M_bcast_d(WDES%COMM, TSIF, 9))
!=======================================================================
! write out energy, stress and forces on ions
!=======================================================================
      IF (DYN%ISIF/=0) THEN
      FAKT=EVTOJ*1E22_q/LATT_CUR%OMEGA

      IF (IO%IU6>=0 .AND. LWRITE_STRESS) THEN

      IF (DYN%ISIF==1) &
     &WRITE(IO%IU6,*)'only isotrope contributions calculated', &
     &          ' for stress because ISIF=1 (only pressure is correct)'

      !tb beg
      IF (LVDW) THEN
         WRITE(IO%IU6,7261) (PSCSIF(I,I),I=1,3), &
         &     (EWSIF(I,I),I=1,3),EWSIF(1,2),EWSIF(2,3),EWSIF(3,1), &
         &     (DSIF (I,I),I=1,3),DSIF (1,2),DSIF (2,3),DSIF (3,1), &
         &     (XCSIF(I,I),I=1,3),XCSIF(1,2),XCSIF(2,3),XCSIF(3,1), &
         &     (EISIF(I,I),I=1,3),EISIF(1,2),EISIF(2,3),EISIF(3,1), &
         &     (FNLSIF(I,I),I=1,3),FNLSIF(1,2),FNLSIF(2,3),FNLSIF(3,1), &
         &     (AUGSIF(I,I),I=1,3),AUGSIF(1,2),AUGSIF(2,3),AUGSIF(3,1), &
         &     (SIKEF(I,I),I=1,3),SIKEF(1,2),SIKEF(2,3),SIKEF(3,1), &
         &     (FOCKSIF(I,I),I=1,3),FOCKSIF(1,2),FOCKSIF(2,3),FOCKSIF(3,1), &
         &     (VDWSIF(I,I),I=1,3),VDWSIF(1,2),VDWSIF(2,3),VDWSIF(3,1), & 
         &     (TSIF (I,I),I=1,3),TSIF (1,2),TSIF (2,3),TSIF (3,1), &
         &     (TSIF (I,I)*FAKT,I=1,3), &
         &     TSIF(1,2)*FAKT,TSIF(2,3)*FAKT, &
         &     TSIF(3,1)*FAKT,PRESS*FAKT,DYN%PSTRESS
      ELSE
        WRITE(IO%IU6,7262) (PSCSIF(I,I),I=1,3), &
         &     (EWSIF(I,I),I=1,3),EWSIF(1,2),EWSIF(2,3),EWSIF(3,1), &
         &     (DSIF (I,I),I=1,3),DSIF (1,2),DSIF (2,3),DSIF (3,1), &
         &     (XCSIF(I,I),I=1,3),XCSIF(1,2),XCSIF(2,3),XCSIF(3,1), &
         &     (EISIF(I,I),I=1,3),EISIF(1,2),EISIF(2,3),EISIF(3,1), &
         &     (FNLSIF(I,I),I=1,3),FNLSIF(1,2),FNLSIF(2,3),FNLSIF(3,1), &
         &     (AUGSIF(I,I),I=1,3),AUGSIF(1,2),AUGSIF(2,3),AUGSIF(3,1), &
         &     (SIKEF(I,I),I=1,3),SIKEF(1,2),SIKEF(2,3),SIKEF(3,1), &
         &     (FOCKSIF(I,I),I=1,3),FOCKSIF(1,2),FOCKSIF(2,3),FOCKSIF(3,1), &
         &     (TSIF (I,I),I=1,3),TSIF (1,2),TSIF (2,3),TSIF (3,1), &
         &     (TSIF (I,I)*FAKT,I=1,3), &
         &     TSIF(1,2)*FAKT,TSIF(2,3)*FAKT, &
         &     TSIF(3,1)*FAKT,PRESS*FAKT,DYN%PSTRESS
      ENDIF
      !tb end

      IF (DYN%IBRION==0) THEN
         CALL STRESS_CORR( TSIF, T_INFO, LATT_CUR, DYN, IO%IU6 )
      ENDIF

      ENDIF

      !c tb beg
 7261 FORMAT(/ &
     &        '  FORCE on cell =-STRESS in cart. coord. ' &
     &       ,' units (eV):'/ &
     &        '  Direction', &
     &        4X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
     &        '  ----------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Alpha Z',3F12.5/ &
     &        '  Ewald  ',6F12.5/ &
     &        '  Hartree',6F12.5/ &
     &        '  E(xc)  ',6F12.5/ &
     &        '  Local  ',6F12.5/ &
     &        '  n-local',6F12.5/ &
     &        '  augment',6F12.5/ &
     &        '  Kinetic',6F12.5/ &
     &        '  Fock   ',6F12.5/ &
     &        '  vdW    ',6F12.5/ &
     &        '  ---------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Total  ',6F12.5/ &
     &        '  in kB  ',6F12.5/ &
     &        '  external pressure = ',F11.2,' kB', &
     &        '  Pullay stress = ',F11.2,' kB'/)
      !c tb end


 7262 FORMAT(/ &
     &        '  FORCE on cell =-STRESS in cart. coord. ' &
     &       ,' units (eV):'/ &
     &        '  Direction', &
     &        4X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
     &        '  ----------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Alpha Z',3F12.5/ &
     &        '  Ewald  ',6F12.5/ &
     &        '  Hartree',6F12.5/ &
     &        '  E(xc)  ',6F12.5/ &
     &        '  Local  ',6F12.5/ &
     &        '  n-local',6F12.5/ &
     &        '  augment',6F12.5/ &
     &        '  Kinetic',6F12.5/ &
     &        '  Fock   ',6F12.5/ &
     &        '  ---------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Total  ',6F12.5/ &
     &        '  in kB  ',6F12.5/ &
     &        '  external pressure = ',F11.2,' kB', &
     &        '  Pullay stress = ',F11.2,' kB'/)

      ENDIF

      wrtforce: IF (IO%IU6>=0 .AND. LWRITE_FORCE) THEN

        WRITE(IO%IU6,7263)
 7263 FORMAT(/' VOLUME and BASIS-vectors are now :'/ &
     &        ' ------------------------------------------------------', &
     &        '-----------------------')

        WRITE(IO%IU6,7220) INFO%ENMAX,LATT_CUR%OMEGA, &
     &    ((LATT_CUR%A(I,J),I=1,3),(LATT_CUR%B(I,J),I=1,3),J=1,3), &
     &    (LATT_CUR%ANORM(I),I=1,3),(LATT_CUR%BNORM(I),I=1,3)

 7220 FORMAT('  energy-cutoff  :  ',F10.2/ &
     &       '  volume of cell :  ',F10.2/ &
     &       '      direct lattice vectors',17X,'reciprocal lattice vectors'/ &
     &       3(2(3X,3F13.9)/) / &
     &       '  length of vectors'/ &
     &        (2(3X,3F13.9)/) /)

        TEIFOR=0;  TEWIFO=0; TFORNL=0 ; THARFO=0 ; TFORHF=0

        DO J=1,T_INFO%NIONS
        DO I=1,3
          TEIFOR(I)=TEIFOR(I)+EIFOR (I,J)+PARFOR(I,J)+TAUFOR(I,J)
! solvation__
          TEIFOR(I)=TEIFOR(I)+EIFOR_SOL(I,J)
! solvation__
          TEWIFO(I)=TEWIFO(I)+EWIFOR(I,J)
          TFORNL(I)=TFORNL(I)+EINL (I,J)
          THARFO(I)=THARFO(I)+HARFOR(I,J)
          TFORHF(I)=TFORHF(I)+FORHF(I,J)
        ENDDO
        ENDDO

        IF (INFO%LCHCON.OR.INFO%LCORR) THEN
        WRITE(IO%IU6,71) ((EIFOR (I,J)+PARFOR(I,J),I=1,3), &
     &                 (EWIFOR(I,J),I=1,3), &
     &                 (EINL (I,J),I=1,3), &
     &                 (HARFOR(I,J),I=1,3),J=1,T_INFO%NIONS)

        WRITE(IO%IU6,73) (TEIFOR(I),I=1,3), &
     &                (TEWIFO(I),I=1,3), &
     &                (TFORNL(I),I=1,3), &
     &                (THARFO(I),I=1,3)

        ELSE
        WRITE(IO%IU6,75) ((EIFOR (I,J),I=1,3), &
     &                 (EWIFOR(I,J),I=1,3), &
     &                 (EINL (I,J),I=1,3),J=1,T_INFO%NIONS)

        WRITE(IO%IU6,73) (TEIFOR(I),I=1,3), &
     &                (TEWIFO(I),I=1,3), &
     &                (TFORNL(I),I=1,3)

        ENDIF
 73     FORMAT(' ---------------------------------------------------', &
     &         '--------------------------------------------'/ &
     &          4(2X,3(1X,E9.3)))

 71     FORMAT(' FORCES acting on ions'/ &
     &    3X,' electron-ion (+dipol)',12X,'ewald-force',20X,'non-local-force',&
     &    16X,' convergence-correction' / &
     &         ' ---------------------------------------------------', &
     &         '--------------------------------------------'/ &
     &      4(2X,3(1X,E9.3)) )

 75     FORMAT(' FORCES acting on ions:'/ &
     &    3X,' Electron-Ion ',20X,'Ewald-Force',20X,'Non-Local-Force'/ &
     &         ' ---------------------------------------------------', &
     &         '--------------------------------------------'/ &
     &    3(2X,3(1X,E9.3)))

        WRITE(IO%IU6,*)
        WRITE(IO%IU6,*)

      WRITE(IO%IU6,72)
      DO J=1,T_INFO%NIONS
        VTMP=T_INFO%POSION(1:3,J)
        CALL  DIRKAR(1,VTMP,LATT_CUR%A)
        WRITE(IO%IU6,76) VTMP,(TIFOR (I,J),I=1,3)
      ENDDO

      IF (IO%INTERACTIVE) THEN
         WRITE(IO%IU0,'(A)') 'FORCES:'
         DO J=1,T_INFO%NIONS
            WRITE(IO%IU0,'(3F14.7)') (TIFOR (I,J),I=1,3)
         ENDDO
      ENDIF


 72     FORMAT( ' POSITION    ',35X,'TOTAL-FORCE (eV/Angst)'/ &
     &          ' ----------------------------------------------', &
     &          '-------------------------------------')
 76     FORMAT((3F13.5,3X,3F14.6))


        VTMP=TEIFOR+TEWIFO+TFORNL+THARFO+TFORHF

        WRITE(IO%IU6,74) VTMP

 74     FORMAT( ' ----------------------------------------------', &
     &          '-------------------------------------',/ &
     &          '    total drift:      ',20X,3F14.6)
      ENDIF wrtforce

      PROFILING_STOP('force_and_stress')

   END SUBROUTINE FORCE_AND_STRESS

!***************** SUBROUTINE FORCE_AND_STRESS_LJ **********************
!
! Use Lennard-Jones potentials only for calculating forces and stress
!
!***********************************************************************
  SUBROUTINE FORCE_AND_STRESS_LJ(GRIDC,IO,WDES,P,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN)
    USE mgrid
    USE base
    USE pseudo_struct
    USE wave
    USE lattice
    USE poscar
    USE vdwforcefield
    USE dynconstr
    USE classicfields
    IMPLICIT NONE
    TYPE(grid_3d),   INTENT(IN) :: GRIDC
    TYPE(in_struct), INTENT(IN) :: IO
    TYPE(wavedes),   INTENT(IN) :: WDES
    TYPE(potcar),    INTENT(IN) :: P
    TYPE(latt),      INTENT(IN) :: LATT_CUR
    TYPE(dynamics),  INTENT(IN) :: DYN
    TYPE(type_info), INTENT(IN) :: T_INFO
    REAL(q), INTENT(OUT) :: TSIF(3,3), TIFOR(3,T_INFO%NIONS)
    REAL(q)   OFIELD_E, OFIELD_E2             ! energy contribution from order field
    REAL(q)   OFIELD_FOR(3,T_INFO%NIOND)      ! forces from order field
    REAL(q), INTENT(OUT) :: TOTEN

    REAL(q) :: RELVOL(T_INFO%NIONS)
    REAL(q) :: EV2KB, EV2REDUCED_P
    REAL(q) :: KINETIC_STRESS(3,3), TOTAL_STRESS(3,3), PRESSURE
    REAL(q) :: REDUCED_V, REDUCED_RHO
    INTEGER :: I, J
    REAL(q) :: DIS
    INTEGER, POINTER :: REFSTATE(:)
    !tb beg
    REAL(q),ALLOCATABLE :: M2(:),Overlap(:,:),RELCHG(:),HCD(:)
    !tb end


    TSIF = 0._q
    TIFOR = 0._q
    TOTEN = 0._q
    ! for pure Lennard-Jones we invoke the vdw routines directly
    !tb beg
       !CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,P%ELEMENT,RELVOL,612,REFSTATE)
       CALL vdw_forces_main(GRIDC,IO,LATT_CUR,DYN,T_INFO,TSIF,TIFOR,TOTEN,P%ELEMENT,RELVOL,612,REFSTATE,Overlap,M2,RELCHG,HCD)
    !tb end

! external order field (see paircorrection.F file)
    CALL OFIELD(GRIDC%COMM, IO, LATT_CUR, DYN, T_INFO, OFIELD_FOR, OFIELD_E)
    TIFOR=TIFOR+OFIELD_FOR
    TOTEN=TOTEN+OFIELD_E

!    DIS=1E-7
!    DO J=1,T_INFO%NIONS
!    DO I=1,3
!       CALL DIRKAR(T_INFO%NIONS, DYN%POSION, LATT_CUR%A)
!       DYN%POSION(I,J)=DYN%POSION(I,J)+DIS
!       CALL KARDIR(T_INFO%NIONS, DYN%POSION, LATT_CUR%B)
!       CALL OFIELD(GRIDC%COMM, IO, LATT_CUR, DYN, T_INFO, OFIELD_FOR, OFIELD_E)

!       CALL DIRKAR(T_INFO%NIONS, DYN%POSION, LATT_CUR%A)
!       DYN%POSION(I,J)=DYN%POSION(I,J)-2*DIS
!       CALL KARDIR(T_INFO%NIONS, DYN%POSION, LATT_CUR%B)
!       CALL OFIELD(GRIDC%COMM, IO, LATT_CUR, DYN, T_INFO, OFIELD_FOR, OFIELD_E2)
!       CALL DIRKAR(T_INFO%NIONS, DYN%POSION, LATT_CUR%A)
!       DYN%POSION(I,J)=DYN%POSION(I,J)+DIS
!       CALL KARDIR(T_INFO%NIONS, DYN%POSION, LATT_CUR%B)

!       WRITE(*,*) OFIELD_FOR(I,J),(OFIELD_E2-OFIELD_E)/2/DIS
!    ENDDO
!    ENDDO

    CALL STOP_TIMING("G",IO%IU6,'OFIELD')

    ! distribute forces to all nodes (overwriting resutls of other nodes)
    !FIXME: Grimme & TS vdW force implementation require broadcasting the forces for consistency
!    CALLMPI(M_bcast_d(WDES%COMM, TIFOR, T_INFO%NIONS*3))

!    PRESS=(TSIF(1,1)+TSIF(2,2)+TSIF(3,3))/3._q
    IF (IO%IU6>=0) THEN
      EV2KB = EVTOJ*1E22_Q/LATT_CUR%OMEGA
      EV2REDUCED_P = LJ_SIGMA**3 / LJ_EPSILON / LATT_CUR%OMEGA

      CALL COMPUTE_KINETIC_STRESS(T_INFO,LATT_CUR,DYN,KINETIC_STRESS,TSIF,-1)
      TOTAL_STRESS = TSIF + KINETIC_STRESS
      PRESSURE = (TOTAL_STRESS(1,1)+TOTAL_STRESS(2,2)+TOTAL_STRESS(3,3)) / 3.0_q

      WRITE(IO%IU6,7221) &
        & (TSIF(I,I),I=1,3),TSIF (1,2),TSIF (2,3),TSIF (3,1), &
        & (KINETIC_STRESS(I,I),I=1,3),KINETIC_STRESS(1,2),KINETIC_STRESS(2,3),KINETIC_STRESS(3,1), &
        & (TOTAL_STRESS(I,I),I=1,3),TOTAL_STRESS(1,2),TOTAL_STRESS(2,3),TOTAL_STRESS(3,1), &
        & (EV2KB*TOTAL_STRESS(I,I),I=1,3),EV2KB*TOTAL_STRESS(1,2),EV2KB*TOTAL_STRESS(2,3),EV2KB*TOTAL_STRESS(3,1), &
        & PRESSURE*EV2KB, &
        & PRESSURE*EV2REDUCED_P, &
        & DYN%PSTRESS
 7221 FORMAT(/ &
     &        '  FORCE on cell =-STRESS in cart. coord. ' &
     &       ,' units (eV/cell):'/ &
     &        '  Direction', &
     &        4X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
     &        '  ----------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Virial ',6F12.5/ &
     &        '  Kinetic',6F12.5/ &
     &        '  ---------------------------------------------------', &
     &        '----------------------------------'/ &
     &        '  Total: ',6F12.5/ &
     &        '  in kB  ',6F12.5/ &
     &        '  Pressure:            ',F11.2,' kB'/ &
     &        '  reduced pressure p*: ',F11.2,' eps/sigma^3'/ &
     &        '  External:            ',F11.2,' kB'/)

      REDUCED_V = LATT_CUR%OMEGA / LJ_SIGMA**3
      REDUCED_RHO = T_INFO%NIONS / REDUCED_V

      WRITE(IO%IU6,7263)
 7263 FORMAT(/' VOLUME and BASIS-vectors are now :'/ &
       &        ' ------------------------------------------------------', &
       &        '-----------------------')
      WRITE(IO%IU6,7220) LATT_CUR%OMEGA, REDUCED_V, REDUCED_RHO, &
       &    ((LATT_CUR%A(I,J),I=1,3),(LATT_CUR%B(I,J),I=1,3),J=1,3), &
       &    (LATT_CUR%ANORM(I),I=1,3),(LATT_CUR%BNORM(I),I=1,3)
 7220 FORMAT('  volume of cell:        ',F10.2,' A^3'/ &
       &     '  reduced volume     V*: ',F10.2,' sigma^3'/ &
       &     '  reduced density  rho*: ',F10.5,' ions/sigma^3'/ &
       &     '      direct lattice vectors',17X,'reciprocal lattice vectors'/ &
       &       3(2(3X,3F13.9)/) / &
       &     '  length of vectors'/ &
       &        (2(3X,3F13.9)/) /)
    END IF

  END SUBROUTINE


!************************ SUBROUTINE VCA_FORCE *************************
! 
! average (or more precisely sum) force over equivalent atoms
!
!***********************************************************************

   SUBROUTINE VCA_FORCE(T_INFO, TIFOR)
     USE base
     USE poscar
     IMPLICIT NONE
     TYPE (type_info)   T_INFO
     REAL(q)   TIFOR(3,T_INFO%NIOND)           ! total force (updated by routine)
     INTEGER   NT, NI, NIP
     LOGICAL   LVCA 

     LVCA=.FALSE.
     DO NT=1,T_INFO%NTYP
        IF (T_INFO%VCA(NT)/=1.0) THEN
           LVCA=.TRUE.
        ENDIF
     ENDDO
     IF (.NOT. LVCA) RETURN

     DO NI=1,T_INFO%NIONS
        DO NIP=NI+1,T_INFO%NIONS
           IF ( SQRT(SUM((T_INFO%POSION(:,NI)-T_INFO%POSION(:,NIP))**2))<1E-10 ) THEN
              TIFOR(:,NI) =TIFOR(:,NI)+TIFOR(:,NIP)
              TIFOR(:,NIP)=TIFOR(:,NI)
           ENDIF
        ENDDO
     ENDDO

   END SUBROUTINE VCA_FORCE


!************************ SUBROUTINE VCA_DER   *************************
! 
! calculate derivate with respect to VCA parameter for all atom types
! that have a setting different from 1
! for simplicity this is done by finite differences
! this is certainly not very efficient, but since VCA is not
! a frequency used option, and the cost is less than one SCF cycle
! that approach is very reasonable
!
!***********************************************************************

   SUBROUTINE VCA_DER( & 
        HAMILTONIAN,KINEDEN, &
        P,WDES,NONLR_S,NONL_S,W,LATT_CUR, &
        T_INFO,INFO,IO,KPOINTS,GRID,GRID_SOFT, &
        GRIDC,GRIDUS,C_TO_US,SOFT_TO_C,SYMM, &
        CHTOT,DENCOR,CVTOT,CSTRF, &
        CDIJ,CQIJ,CRHODE,N_MIX_PAW,RHOLM, &
        CHDEN,SV,DOS,DOSI,CHAM, &
        LMDIM,IRDMAX,NEDOS)
      USE prec
      USE hamil_high
      USE morbitalmag
      USE pseudo
      USE lattice
      USE us
      USE pot
      USE fileio
      USE nonl_high
      USE ini
      USE ebs
      USE wave_high
      USE subrot
      USE base
      USE mpimy
      USE mkpoints
      USE mgrid
      USE poscar
      USE wave
      USE mdipol
      USE pawm
      USE ini
      USE core_rel
      USE pp_data
      USE meta
! solvation__
      USE solvation
! solvation__
      IMPLICIT NONE
      TYPE (ham_handle)  HAMILTONIAN
      TYPE (tau_handle)  KINEDEN
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P(T_INFO%NTYP)
      TYPE (wavedes)     WDES
      TYPE (nonlr_struct) NONLR_S
      TYPE (nonl_struct) NONL_S
      TYPE (wavespin)    W          ! wavefunction
      TYPE (latt)        LATT_CUR
      TYPE (info_struct) INFO
      TYPE (in_struct)   IO
      TYPE (kpoints_struct) KPOINTS
      TYPE (grid_3d)     GRID       ! grid for wavefunctions
      TYPE (grid_3d)     GRID_SOFT  ! grid for soft chargedensity
      TYPE (grid_3d)     GRIDC      ! grid for potentials/charge
      TYPE (grid_3d)     GRIDUS     ! temporary grid in us.F
      TYPE (grid_3d)     GRIDB      ! Broyden grid
      TYPE (transit)     C_TO_US    ! index table between GRIDC and GRIDUS
      TYPE (transit)     SOFT_TO_C  ! index table between GRID_SOFT and GRIDC
      TYPE (symmetry) ::   SYMM
      COMPLEX(q)  CHTOT(GRIDC%MPLWV,WDES%NCDIJ) ! charge-density in real / reciprocal space
      RGRID       DENCOR(GRIDC%RL%NP)           ! partial core
      COMPLEX(q)  CVTOT(GRIDC%MPLWV,WDES%NCDIJ) ! local potential
      COMPLEX(q)  CSTRF(GRIDC%MPLWV,T_INFO%NTYP)! structure factor
      INTEGER  LMDIM
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
               CQIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ), &
               CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
!  paw sphere charge density
      INTEGER  N_MIX_PAW
      REAL(q)  RHOLM(N_MIX_PAW,WDES%NCDIJ)
      COMPLEX(q)  CHDEN(GRID_SOFT%MPLWV,WDES%NCDIJ)
      RGRID       SV(DIMREAL(GRID%MPLWV),WDES%NCDIJ)
      INTEGER  NEDOS
      REAL(q)  DOS(NEDOS,WDES%ISPIN),DOSI(NEDOS,WDES%ISPIN)
      GDEF     CHAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
      INTEGER  IRDMAX
   ! local
     INTEGER :: NT, IDIS, IRDMAA, NORDER
     REAL(q) :: VCA_SAVE
     REAL(q) :: EFERMI
     REAL(q),PARAMETER :: VCA_STEP=0.0001
     REAL(q) :: ENERGY_CHANGE, TOTEN, EALLAT
     TYPE (energy)      E
     REAL(q) :: EWIFOR(3,T_INFO%NIONS), EWSIF(3,3), XCSIF(3,3)
! local l-projected wavefunction characters (not really used here)
     REAL(q)    PAR(1,1,1,1,WDES%NCDIJ),DOSPAR(1,1,1,WDES%NCDIJ)

     
     DO NT=1,T_INFO%NTYP
        IF (T_INFO%VCA(NT)/=1 .AND. T_INFO%VCA(NT)/=0) THEN
           VCA_SAVE=T_INFO%VCA(NT)
           ENERGY_CHANGE=0
           DO IDIS=-1,1,2
              T_INFO%VCA(NT)=VCA_SAVE+VCA_STEP*IDIS
              INFO%NELECT=SUM(T_INFO%NITYP*P%ZVALF*T_INFO%VCA)
              EALLAT=SUM(T_INFO%NITYP*P%EATOM_CORRECTED*T_INFO%VCA)

              CALL FEWALD(T_INFO%POSION,EWIFOR,LATT_CUR%A,LATT_CUR%B,LATT_CUR%ANORM,LATT_CUR%BNORM, &
                   LATT_CUR%OMEGA,EWSIF,E%TEWEN,T_INFO%NTYP,P%ZVALF,T_INFO%VCA, &
                   T_INFO%NIONS,T_INFO%NIONS,T_INFO%ITYP,T_INFO%NITYP,IO%IU6,.TRUE.)

              CALL STUFAK(GRIDC,T_INFO,CSTRF)

              IF (INFO%LCORE) CALL RHOPAR(GRIDC,T_INFO,INFO,LATT_CUR,P,CSTRF,DENCOR,IO%IU6)
              IF (LDO_METAGGA()) CALL TAUPAR(GRIDC,T_INFO,LATT_CUR%B,LATT_CUR%OMEGA,P,CSTRF,KINEDEN%TAUC)

              CALL POTLOK(GRID,GRIDC,GRID_SOFT, WDES%COMM_INTER, WDES, &
                   INFO,P,T_INFO,E,LATT_CUR, &
                   CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF)
              
              CALL POTLOK_METAGGA(KINEDEN, &
                   GRID,GRIDC,GRID_SOFT,WDES%COMM_INTER,WDES,INFO,P,T_INFO,E,LATT_CUR, &
                   CHDEN,CHTOT,DENCOR,CVTOT,SV,HAMILTONIAN%MUTOT,HAMILTONIAN%MU,SOFT_TO_C,XCSIF)
              
              CALL VECTORPOT(GRID, GRIDC, GRID_SOFT, SOFT_TO_C,  WDES%COMM_INTER, & 
                   LATT_CUR, T_INFO%POSION, HAMILTONIAN%AVEC, HAMILTONIAN%AVTOT)
              
              CALL SETDIJ(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                   LMDIM,CDIJ,CQIJ,CVTOT,IRDMAA,IRDMAX)
           
              CALL SETDIJ_AVEC(WDES,GRIDC,GRIDUS,C_TO_US,LATT_CUR,P,T_INFO,INFO%LOVERL, &
                   LMDIM,CDIJ,HAMILTONIAN%AVTOT, NONLR_S, NONL_S, IRDMAX)
           
              CALL SET_DD_MAGATOM(WDES, T_INFO, P, LMDIM, CDIJ)
           
              CALL SET_DD_PAW(WDES, P , T_INFO, INFO%LOVERL, &
                   WDES%NCDIJ, LMDIM, CDIJ(1,1,1,1),  RHOLM, CRHODE(1,1,1,1), &
                   E,  LMETA=.FALSE., LASPH=INFO%LASPH, LCOREL= .FALSE.  )
              
              CALL UPDATE_CMBJ(GRIDC,T_INFO,LATT_CUR,IO%IU6)

              CALL REDIS_PW_OVER_BANDS(WDES, W)
              
              CALL EDDIAG(HAMILTONIAN,GRID,LATT_CUR,NONLR_S,NONL_S,W,WDES,SYMM, &
                   LMDIM,CDIJ,CQIJ, 0,SV,T_INFO,P,IO%IU0,E%EXHF)

              CALL DENSTA( IO%IU0, IO%IU6, WDES, W, KPOINTS, INFO%NELECT, &
                   INFO%NUP_DOWN, E%EENTROPY, EFERMI, KPOINTS%SIGMA, .FALSE.,  &
                   NEDOS, 0, 0, DOS, DOSI, PAR, DOSPAR)

              E%EBANDSTR=BANDSTRUCTURE_ENERGY(WDES, W)
              TOTEN=E%EBANDSTR+E%DENC+E%XCENC+E%TEWEN+E%PSCENC+E%EENTROPY+E%PAWPS+E%PAWAE+EALLAT+E%EXHF+ECORE()+Ediel_sol

              IF (IO%IU6>=0) THEN
                 NORDER=0 ; IF (KPOINTS%ISMEAR>=0) NORDER=KPOINTS%ISMEAR
                 WRITE(IO%IU6,7240) & 
                      NT,T_INFO%VCA(NT),E%PSCENC,E%TEWEN,E%DENC,E%EXHF,E%XCENC,E%PAWPS,E%PAWAE, &
                      E%EENTROPY,E%EBANDSTR,EALLAT,TOTEN, &
                      TOTEN-E%EENTROPY,TOTEN-E%EENTROPY/(2+NORDER)
              ENDIF


7240  FORMAT(/ &
              ' Free energy of the ion-electron system (eV)  for NT=',I4,' VCA=',F6.4,/ &
     &        '  ---------------------------------------------------'/ &
     &        '  alpha Z        PSCENC = ',F18.8/ &
     &        '  Ewald energy   TEWEN  = ',F18.8/ &
     &        '  -Hartree energ DENC   = ',F18.8/ &
     &        '  -exchange  EXHF       = ',F18.8/ &
     &        '  -V(xc)+E(xc)   XCENC  = ',F18.8/ &
     &        '  PAW double counting   = ',2F18.8/ &
     &        '  entropy T*S    EENTRO = ',F18.8/ &
     &        '  eigenvalues    EBANDS = ',F18.8/ &
     &        '  atomic energy  EATOM  = ',F18.8/ &
     &        '  ---------------------------------------------------'/ &
     &        '  free energy    TOTEN  = ',F18.8,' eV'// &
     &        '  energy without entropy =',F18.8, &
     &        '  energy(sigma->0) =',F18.8)

              ENERGY_CHANGE=ENERGY_CHANGE+IDIS*TOTEN
           ENDDO

           ! restore number of electrons
           T_INFO%VCA(NT)=VCA_SAVE
           INFO%NELECT=SUM(T_INFO%NITYP*P%ZVALF*T_INFO%VCA)

           ! calculate derivative
           ENERGY_CHANGE=ENERGY_CHANGE/2/VCA_STEP
           IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,'(/ " VCA derivative with respect to type NT=",I4,F14.7/)') NT, ENERGY_CHANGE
           ENDIF
        ENDIF
     ENDDO
   END SUBROUTINE VCA_DER

!***********************************************************************
!
! sum the ionic kinetic contributions to the stress tensor 
!
!***********************************************************************

      SUBROUTINE STRESS_CORR( TSIF, T_INFO, LATT_CUR, DYN, IUNIT )
      USE prec
      USE poscar
      USE lattice
      USE constant
      IMPLICIT NONE

      TYPE (TYPE_INFO) :: T_INFO
      TYPE (LATT)      :: LATT_CUR
      TYPE (DYNAMICS)  :: DYN
      REAL(Q)          :: TSIF(3,3), TMP(6), FAKT, FAC, VEL(3)
      INTEGER          :: NI, NT, NOFFS, IUNIT
      REAL(q) :: pressure

      FAKT = EVTOJ*1E22_Q/LATT_CUR%OMEGA
      FAC  = AMTOKG *  &
     &     1E5_Q  *  &
     &     1E5_Q  *  &
     &     1E30_Q *  &
     &     1E-8_Q

      TMP = 0._Q
      NOFFS = 0
      DO NT=1,T_INFO%NTYP
         DO NI=1+NOFFS,T_INFO%NITYP(NT)+NOFFS
            VEL(1) = DYN%VEL(1,NI)/DYN%POTIM
            VEL(2) = DYN%VEL(2,NI)/DYN%POTIM
            VEL(3) = DYN%VEL(3,NI)/DYN%POTIM
            CALL  DIRKAR( 1, VEL, LATT_CUR%A )
            TMP(1) = TMP(1) + VEL(1)*VEL(2) * T_INFO%POMASS(NT)
            TMP(2) = TMP(2) + VEL(2)*VEL(3) * T_INFO%POMASS(NT)
            TMP(3) = TMP(3) + VEL(3)*VEL(1) * T_INFO%POMASS(NT)
            TMP(4) = TMP(4) + VEL(1)*VEL(1) * T_INFO%POMASS(NT)
            TMP(5) = TMP(5) + VEL(2)*VEL(2) * T_INFO%POMASS(NT)
            TMP(6) = TMP(6) + VEL(3)*VEL(3) * T_INFO%POMASS(NT)
         ENDDO
         NOFFS = NOFFS + T_INFO%NITYP(NT)
      ENDDO

      TMP = TMP/LATT_CUR%OMEGA*FAC         ! NOW TMP IS IN KB

      !tb beg 
      !WRITE(IUNIT,'(''  kinetic energy (ideal gas correction) = '',F9.2,'' kB'')') &
      ! &     (TMP(4)+TMP(5)+TMP(6))/3
      WRITE(IUNIT,'(''  kinetic pressure (ideal gas correction) = '',F9.2,'' kB'')') &
     &     (TMP(4)+TMP(5)+TMP(6))/3
      pressure=(TSIF(1,1)+TSIF(2,2)+TSIF(3,3))*FAKT
      pressure=pressure + (TMP(4)+TMP(5)+TMP(6))
      pressure=pressure/3.
      write(IUNIT,'(''  total pressure  = '',F9.2,'' kB'')') pressure
      !tb end
      WRITE(IUNIT,'(''  Total+kin. '',F9.3,5F12.3)') &
     &     TSIF(1,1)*FAKT + TMP(4),  &
     &     TSIF(2,2)*FAKT + TMP(5),  &
     &     TSIF(3,3)*FAKT + TMP(6),  &
     &     TSIF(1,2)*FAKT + TMP(1),  &
     &     TSIF(2,3)*FAKT + TMP(2),  &
     &     TSIF(3,1)*FAKT + TMP(3)


      RETURN
      END
