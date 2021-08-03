!  Some important Parameters, to convert to a.u.
!  - AUTOA  = 1. a.u. in Angstroem
!  - RYTOEV = 1 Ry in Ev
!  - EVTOJ  = 1 eV in Joule
!  - AMTOKG = 1 atomic mass unit ("proton mass") in kg
!  - BOLKEV = Boltzmanns constant in eV/K
!  - BOLK   = Boltzmanns constant in Joule/K

      REAL(q), PARAMETER :: AUTOA=0.529177249_q,RYTOEV=13.605826_q
      REAL(q), PARAMETER :: CLIGHT = 137.037  ! speed of light in a.u.
      REAL(q), PARAMETER :: EVTOJ=1.60217733E-19_q,AMTOKG=1.6605402E-27_q, &
     &           BOLKEV=8.6173857E-5_q,BOLK=BOLKEV*EVTOJ

      REAL(q), PARAMETER :: EVTOKCAL=23.06
! FELECT = (the electronic charge)/(4*pi*the permittivity of free space)
!         in atomic units this is just e^2
! EDEPS = electron charge divided by the permittivity of free space
!         in atomic units this is just 4 pi e^2
! HSQDTM = (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
!
      REAL(q),PARAMETER  :: PI =3.141592653589793238_q,TPI=2*PI
      COMPLEX(q),PARAMETER  :: CITPI = (0._q,1._q)*TPI
      REAL(q),PARAMETER  :: FELECT = 2*AUTOA*RYTOEV,EDEPS=4*PI*2*RYTOEV*AUTOA,&
     &                   HSQDTM = RYTOEV*AUTOA*AUTOA
! vector field A times momentum times e/ (2 m_e c) is an energy
! magnetic moments are supplied in Bohr magnetons
! e / (2 m_e c) A(r) p(r) = energy
! e / (2 m_e c) m_s x ( r - r_s) / (r-r_s)^3 hbar nabla =
! e^2 hbar^2 / (2 m_e^2 c^2) 1/ lenght^3 = energy
! conversion factor from magnetic moment to energy
! checked independently in SI by Gilles de Wijs
      REAL(q),PARAMETER :: MAGMOMTOENERGY=1/CLIGHT**2*AUTOA**3*RYTOEV

! dimensionless number connecting input and output magnetic moments
! AUTOA e^2 (2 m_e c^2)
      REAL(q),PARAMETER :: MOMTOMOM=AUTOA/CLIGHT/CLIGHT/2

      REAL(q),PARAMETER :: AUTOA2=AUTOA *AUTOA
      REAL(q),PARAMETER :: AUTOA3=AUTOA2*AUTOA
      REAL(q),PARAMETER :: AUTOA4=AUTOA2*AUTOA2
      REAL(q),PARAMETER :: AUTOA5=AUTOA3*AUTOA2
