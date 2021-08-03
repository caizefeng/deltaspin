#include "symbol.inc"
      MODULE pot_gpu
      USE prec
      USE charge
      USE pot
      CONTAINS
    SUBROUTINE POTLOK_GPU(GRID,GRIDC,GRID_SOFT, COMM_INTER, WDES,  &
                  INFO,P,T_INFO,E,LATT_CUR,  &
                  CHTOT,CSTRF,CVTOT,DENCOR,SV, SOFT_TO_C,XCSIF )
      USE prec
      USE mpimy
      USE mgrid
      USE pseudo
      USE lattice
      USE poscar
      USE setexm
      USE base
      USE xcgrad
      USE wave
      USE mdipol
      USE Constrained_M_modular
#ifdef CUDA_GPU
      USE cuda_interface
#endif

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID,GRIDC,GRID_SOFT
      TYPE (wavedes)     WDES
      TYPE (transit)     SOFT_TO_C
      TYPE (info_struct) INFO
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (energy)      E
      TYPE (latt)        LATT_CUR
      TYPE (communic)    COMM_INTER

      RGRID   SV(DIMREAL(GRID%MPLWV), WDES%NCDIJ)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP), &
                 CHTOT(GRIDC%MPLWV, WDES%NCDIJ), CVTOT(GRIDC%MPLWV,WDES%NCDIJ)
      RGRID      DENCOR(GRIDC%RL%NP)
      REAL(q)    XCSIF(3,3),TMPSIF(3,3)
! work arrays (allocated after call to FEXCG)
      COMPLEX(q), ALLOCATABLE::  CWORK1(:),CWORK(:,:)
      REAL(q) ELECTROSTATIC
      LOGICAL, EXTERNAL :: L_NO_LSDA_GLOBAL
      INTEGER(c_intptr_t) :: GPU_CHTOT

nv_profile_start(NVP_POTLOK)
      
      MWORK1=MAX(GRIDC%MPLWV,GRID_SOFT%MPLWV)
      ALLOCATE(CWORK1(MWORK1),CWORK(GRIDC%MPLWV,WDES%NCDIJ))
!-----------------------------------------------------------------------
!
!  calculate the exchange correlation potential and the dc. correction
!
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_ECDC)

      EXC     =0
      E%XCENC =0
      E%EXCG  =0
      E%CVZERO=0
      XCSIF   =0

      CVTOT=0
  xc: IF (ISLDAXC()) THEN
     ! transform the charge density to real space
        EXCG  =0
        XCENCG=0
        CVZERG=0
        TMPSIF=0

        DO ISP=1,WDES%NCDIJ
#ifdef MPI
           CALL FFT3D_MPI(CHTOT(1,ISP),GRIDC,1)
#else
           WRITE(0,*) 'GPU version needs to be compiled with MPI defined!'
           STOP
#endif
        ENDDO
        
!       CALL THREADSYNCHRONIZE()
        IF (WDES%ISPIN==2) THEN

          ! get the charge and the total magnetization
          CALL MAG_DENSITY(CHTOT, CWORK, GRIDC, WDES%NCDIJ)
! do LDA+U instead of LSDA+U
          IF (L_NO_LSDA_GLOBAL()) CWORK(:,2)=0
!
          IF (ISGGA()) THEN
             ! gradient corrections to LDA
             ! unfortunately FEXCGS requires (up,down) density
             ! instead of (rho,mag)
             CALL RL_FLIP(CWORK, GRIDC, 2, .TRUE.)
             ! GGA potential
             CALL FEXCGS(2, GRIDC, LATT_CUR, XCENCG, EXCG, CVZERG, TMPSIF, &
                  CWORK, CVTOT, DENCOR)
             CALL RL_FLIP(CWORK, GRIDC, 2, .FALSE.)
          ENDIF

          ! add LDA part of potential
          CALL FEXCF(GRIDC,LATT_CUR%OMEGA, &
             CWORK(1,1), CWORK(1,2), DENCOR, CVTOT(1,1), CVTOT(1,2), &
             E%CVZERO,EXC,E%XCENC,XCSIF, .TRUE.)
!gk COH
          ! add Coulomb hole
          CALL COHSM1_RGRID(2, CWORK(1,1), CVTOT(1,1), DENCOR, GRIDC,LATT_CUR%OMEGA, .TRUE.)
!gK COHend
          ! we have now the potential for up and down stored in CVTOT(:,1) and
          ! CVTOT(:,2)

          ! get the proper direction vx = v0 + hat m delta v
          CALL MAG_DIRECTION(CHTOT(1,1), CVTOT(1,1), GRIDC, WDES%NCDIJ)
        ELSEIF (WDES%LNONCOLLINEAR) THEN
          IF (ISGGA()) THEN
             ! GGA potential
             CALL FEXCGS(4, GRIDC, LATT_CUR, XCENCG, EXCG, CVZERG, TMPSIF, &
                  CHTOT, CVTOT, DENCOR)
          ENDIF

          ! FEXCF requires (up,down) density instead of (rho,mag)
          CALL MAG_DENSITY(CHTOT, CWORK, GRIDC, WDES%NCDIJ)
! quick hack to do LDA+U instead of LSDA+U
          IF (L_NO_LSDA_GLOBAL()) CWORK(:,2)=0
! end of hack
          ! add LDA part of potential
          CALL FEXCF(GRIDC,LATT_CUR%OMEGA, &
             CWORK(1,1), CWORK(1,2), DENCOR, CVTOT(1,1), CVTOT(1,2), &
             E%CVZERO,EXC,E%XCENC,XCSIF, .TRUE.)
!gk COH
          ! add Coulomb hole
          CALL COHSM1_RGRID(2, CWORK(1,1), CVTOT(1,1), DENCOR, GRIDC,LATT_CUR%OMEGA, .TRUE.)
!gK COHend
          ! we have now the potential for up and down stored in CVTOT(:,1) and
          ! CVTOT(:,2)
          ! get the proper direction vx = v0 + hat m delta v
                    
          CALL MAG_DIRECTION(CHTOT(1,1), CVTOT(1,1), GRIDC, WDES%NCDIJ)
       ELSE
          IF (ISGGA()) THEN
             ! gradient corrections to LDA
             CALL FEXCG(GRIDC,LATT_CUR,XCENCG,EXCG,CVZERG,TMPSIF, &
                  CHTOT,CVTOT,DENCOR)
          ENDIF
                
          ! LDA part of potential
          CALL FEXCP(GRIDC,LATT_CUR%OMEGA, &
               CHTOT,DENCOR,CVTOT,CWORK,E%CVZERO,EXC,E%XCENC,XCSIF,.TRUE.)
!gk COH
          ! add Coulomb hole
          CALL COHSM1_RGRID(1, CHTOT(1,1), CVTOT(1,1), DENCOR, GRIDC,LATT_CUR%OMEGA, .TRUE.)
!gK COHend
       ENDIF

       XCSIF=XCSIF+TMPSIF
       E%EXCG=EXC+EXCG
       E%XCENC=E%XCENC+XCENCG
       E%CVZERO=E%CVZERO+CVZERG

      ELSE xc
         DO ISP=1,WDES%NCDIJ
            CALL FFT3D(CHTOT(1,ISP),GRIDC,1)
         ENDDO
      ENDIF xc

nv_profile_stop(NVP_POTLOK_ECDC)
!-MM- changes to accomodate constrained moments
!-----------------------------------------------------------------------
! add constraining potential
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_CPOT)
#if ! defined(NGXhalf) && ! defined(NGZhalf)
      IF (M_CONSTRAINED()) THEN
      ! NB. at this point both CHTOT and CVTOT must be given
      ! in (charge,magnetization) convention in real space
         CALL M_INT(CHTOT,GRIDC,WDES)
         CALL ADD_CONSTRAINING_POT(CVTOT,GRIDC,WDES)
      ENDIF
#endif
!-MM- end of addition
nv_profile_stop(NVP_POTLOK_CPOT)
!-----------------------------------------------------------------------
! calculate the total potential
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_TOTPOT)
! add external electrostatic potential
      DIP%ECORR=0
      DIP%E_ION_EXTERN=0

      IF (DIP%LCOR_DIP) THEN
          ! get the total charge and store it in CWORK
          IF  ( WDES%NCDIJ > 1) THEN
             CALL MAG_DENSITY(CHTOT,CWORK, GRIDC, WDES%NCDIJ)
          ELSE
             CALL RL_ADD(CHTOT,1.0_q,CHTOT,0.0_q,CWORK,GRIDC)
          ENDIF

           CALL CDIPOL(GRIDC, LATT_CUR,P,T_INFO, &
             CWORK,CSTRF,CVTOT(1,1), WDES%NCDIJ, INFO%NELECT )

         CALL EXTERNAL_POT(GRIDC, LATT_CUR, CVTOT(1,1))
      ENDIF

      DO ISP=1,WDES%NCDIJ
         CALL FFT_RC_SCALE(CHTOT(1,ISP),CHTOT(1,ISP),GRIDC)
         CALL SETUNB_COMPAT(CHTOT(1,ISP),GRIDC)
      ENDDO
nv_profile_stop(NVP_POTLOK_TOTPOT)
!-----------------------------------------------------------------------
! FFT of the exchange-correlation potential to reciprocal space
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_FFTEXC)
      RINPL=1._q/GRIDC%NPLWV
      DO  ISP=1,WDES%NCDIJ 
         CALL RL_ADD(CVTOT(1,ISP),RINPL,CVTOT(1,ISP),0.0_q,CVTOT(1,ISP),GRIDC)
         CALL FFT3D(CVTOT(1,ISP),GRIDC,-1)
      ENDDO
nv_profile_stop(NVP_POTLOK_FFTEXC)
!-----------------------------------------------------------------------
! add the hartree potential and the double counting corrections
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_POTHAR)
      CALL POTHAR(GRIDC, LATT_CUR, CHTOT, CWORK,E%DENC)
      DO I=1,GRIDC%RC%NP
         CVTOT(I,1)=CVTOT(I,1)+CWORK(I,1)
      ENDDO
nv_profile_stop(NVP_POTLOK_POTHAR)
!-----------------------------------------------------------------------
!  add local pseudopotential potential
!-----------------------------------------------------------------------
nv_profile_start(NVP_POTLOK_LPPOT)
      CALL POTION(GRIDC,P,LATT_CUR,T_INFO,CWORK,CWORK1,CSTRF,E%PSCENC)

      ELECTROSTATIC=0
      NG=1
      col: DO NC=1,GRIDC%RC%NCOL
      N2= GRIDC%RC%I2(NC)
      N3= GRIDC%RC%I3(NC)
      row: DO N1=1,GRIDC%RC%NROW
        SETFACT1
        SETFACT

        ELECTROSTATIC=ELECTROSTATIC+ MULFACT CWORK(NG,1)*CONJG(CHTOT(NG,1))
        NG=NG+1
      ENDDO row
      ENDDO col
      ELECTROSTATIC=ELECTROSTATIC+E%PSCENC-E%DENC+E%TEWEN

      E%PSCENC=E%PSCENC + DIP%ECORR + DIP%E_ION_EXTERN

      DO I=1,GRIDC%RC%NP
         CVTOT(I,1)=CVTOT(I,1)+CWORK(I,1)
      ENDDO
      CALL POT_FLIP(CVTOT, GRIDC,WDES%NCDIJ )
nv_profile_stop(NVP_POTLOK_LPPOT)
!=======================================================================
! if overlap is used :
! copy CVTOT to SV and set contribution of unbalanced lattice-vectors
! to zero,  then  FFT of SV and CVTOT to real space
!=======================================================================
nv_profile_start(NVP_POTLOK_CVTOTSV)

      DO ISP=1,WDES%NCDIJ
         CALL SETUNB_COMPAT(CVTOT(1,ISP),GRIDC)
         CALL CP_GRID(GRIDC,GRID_SOFT,SOFT_TO_C,CVTOT(1,ISP),CWORK1)
         CALL SETUNB(CWORK1,GRID_SOFT)
         CALL FFT3D(CWORK1,GRID_SOFT, 1)
         CALL RL_ADD(CWORK1,1.0_q,CWORK1,0.0_q,SV(1,ISP),GRID_SOFT)

    !  final result is only correct for first in-band-group
    ! (i.e. proc with nodeid 1 in COMM_INTER)
    !  copy to other in-band-groups using COMM_INTER
    ! (see SET_RL_GRID() in mgrid.F, and M_divide() in mpi.F)
#ifdef realmode
         CALLMPI( M_bcast_d(COMM_INTER, SV(1,ISP), GRID%RL%NP))
#else
         CALLMPI( M_bcast_z(COMM_INTER, SV(1,ISP), GRID%RL%NP))
#endif
         CALL FFT3D(CVTOT(1,ISP),GRIDC,1)
      ENDDO
nv_profile_stop(NVP_POTLOK_CVTOTSV)

      DEALLOCATE(CWORK1,CWORK)

nv_profile_stop(NVP_POTLOK)
      RETURN
    END SUBROUTINE POTLOK_GPU
end module pot_gpu
