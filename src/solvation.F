#include "symbol.inc"
!**********************************************************************************
!**********************************************************************************
! This file is only a placeholder for the actual solvation.F file:
! its public procedures are only stubs and the public variables are set so
! as not to interfere with the workings of VASP, but allows you to compile
! VASP with the hooks to the solvation model code VASPsol in place.
!
! The solvation.F file with the actual VASPsol code can be obtained at:
!
!  http://vaspsol.mse.cornell.edu/
!
!**********************************************************************************
!**********************************************************************************


!******************** MODULE SOLVATION ********************************************
!
!
! interfaces the solvation engine with the rest of vasp
!
!
!**********************************************************************************
MODULE solvation

  USE prec

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: SOL_READER, SOL_WRITER, XML_WRITE_SOL, SOL_VCORRECTION

  LOGICAL, SAVE :: LSOL=.FALSE.

  REAL(q), PUBLIC, SAVE :: Ediel_SOL=0._q
  REAL(q), PUBLIC, ALLOCATABLE, SAVE :: EIFOR_SOL(:,:)

CONTAINS

!******************** SUBROUTINE SOL_READER ***************************************
!
!
! Reads in the solvation model parameters
!
!      
!**********************************************************************************
  SUBROUTINE SOL_READER(NIONS,EDIFF,IO)
    USE base
    IMPLICIT NONE

    TYPE (in_struct), INTENT(in) :: IO
    REAL(q), INTENT(in) :: EDIFF
    INTEGER, INTENT(in) :: NIONS
   
    ! this has to be done ALWAYS
    IF (ALLOCATED(EIFOR_SOL)) DEALLOCATE(EIFOR_SOL)
    ALLOCATE(EIFOR_SOL(3,NIONS))

    EIFOR_SOL=0._q; Ediel_SOL=0._q
     
    RETURN
  END SUBROUTINE SOL_READER


!******************** SUBROUTINE SOL_WRITER ***************************************
!      
!
! writes the solvation model parameters to the OUTCAR file
!
!
!**********************************************************************************
  SUBROUTINE SOL_WRITER(IO)
    USE base
    TYPE (in_struct), INTENT(in) :: IO
    RETURN
  END SUBROUTINE SOL_WRITER


!******************** SUBROUTINE XML_WRITE_SOL ************************************
!      
!
! writes the solvation model parameters to vasprun.xml
!
!
!**********************************************************************************
  SUBROUTINE XML_WRITE_SOL
    RETURN
  END SUBROUTINE XML_WRITE_SOL


!******************** SUBROUTINE SOL_VCORRECTION *********************************
!
!
! Computes the potential, energy and force corrections due to solvation
!
!
!********************************************************************************
  SUBROUTINE SOL_VCORRECTION(INFO, T_INFO, LATT_CUR, P, WDES, GRIDC, CHTOT, CVTOT)
    USE base
    USE poscar
    USE lattice
    USE pseudo
    USE mgrid
    USE wave
    USE mdipol

    TYPE (info_struct), INTENT(in) :: INFO
    TYPE (type_info), INTENT(in) :: T_INFO
    TYPE (latt), INTENT(IN) :: LATT_CUR
    TYPE (potcar), INTENT(IN) :: P(T_INFO%NTYP)
    TYPE (wavedes), INTENT(IN) :: WDES
    TYPE (grid_3d), INTENT(IN) :: GRIDC
    
    COMPLEX(q) CHTOT(GRIDC%MPLWV,WDES%NCDIJ)
    COMPLEX(q) CVTOT(GRIDC%MPLWV,WDES%NCDIJ)
    
    RETURN
  END SUBROUTINE SOL_VCORRECTION

END MODULE solvation
