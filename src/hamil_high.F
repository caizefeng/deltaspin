#include "symbol.inc"
MODULE hamil_high
  USE prec
  USE hamil
  ! handle for orbital magnetization and metaGGAs
  TYPE ham_handle
     COMPLEX(q),POINTER :: AVTOT(:,:)    ! local vector magnetization potential
     RGRID  ,POINTER    :: AVEC(:,:)     ! soft part of vector magnetization potential
     COMPLEX(q),POINTER :: MUTOT(:,:)    ! derivative of energy density with respect to kinetic energy density
     RGRID  ,POINTER    :: MU(:,:)       ! same as MUTOT, but on GRID instead of GRIDC
  END TYPE ham_handle

!***********************************************************************
!
! this module implements high level routines to calculate the action
! of the Hamiltonian onto a wavefunction
! high level routines for manipulations wavefunctions are implemented
! as well
!
!***********************************************************************
END MODULE hamil_high

