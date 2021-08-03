!
! exchange correlation table file
! only included if MODULES are not supported
!
! the table has two section one for low densities
!   (1...NEXCHF(1),0...RHOEXC(1))
! and one for high densities
!   (...NEXCHF(2),...RHOEXC(2))
!
      INTEGER, PARAMETER ::  NEXCH=4000
      REAL(q) PARAM1, PARAM2, PARAM3
      REAL(q) Zab_VDW
      LOGICAL :: LUSE_VDW

! rVV10
      REAL(q) BPARAM
      REAL(q) CPARAM

      TYPE EXCTABLE
        REAL(q) EXCTAB(NEXCH,5,6) ! table including spline coeff.
        REAL(q) RHOEXC(2)         ! maximal densities
        INTEGER NEXCHF(2)         ! number of points
        INTEGER LEXCH             ! type of exchange correlation
      END TYPE
