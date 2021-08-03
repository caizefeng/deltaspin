!
! lattice include file
! only included if MODULES are not supported
! defined: LATT_CUR LATT_INI
      TYPE latt
         REAL(q) :: SCALE
         REAL(q) :: A(3,3),B(3,3)
         REAL(q) :: ANORM(3),BNORM(3)
         REAL(q) :: OMEGA
!tb start
         REAL(q) AVEL(3,3)             ! lattice velocities
         INTEGER INITlatv              ! lattice velocities initialized                  !
!tb end

      END TYPE
