
!**************************************************************************
module npt_dynamics 
!**************************************************************************
!
!     Program:	        VASP	
!
!     module:           npt_dynamics
!
!     Variables:      
!                       
!     Created:          
!
!     Purpose:          contains the necessary subroutines for performing
!                       molecular dynamics simulations in various ensembles,
!			including microcanonical, canonical and 
!			isothermal-isobaric. For details of implementation
!			see E. Hernandez, JCP vol. 115, 10282 (2001).
!			
!			
!                                              
!     Routines used:    pre_step()                   (public)
!			post_step()                  (public)
!     Modified:         
!
!**************************************************************************
!  used modules

   use prec
!tb beg
!!   use stress_module
!tb end
!**************************************************************************
!  no implicits, please!

   implicit none
!**************************************************************************
!tb beg
   type stress_type
      logical :: calculated
      real(q), dimension(3,3) :: cartesian
      real(q), dimension(3,3) :: lattice
   end type stress_type
!tb end


!**************************************************************************
!  Public Module variables

!**************************************************************************
!  Private Module variables

   logical, private, parameter :: debug = .false.

   real(q), private, parameter :: amtokg = 1.6605402E-27_q
   real(q), private, parameter :: boltzmann_k = 8.6173857E-5_q
                                    ! in eV K^-1
   real(q), private, parameter :: evtojoule = 1.60217733E-19_q
   real(q), private, parameter :: half = 0.5e0_q
   real(q), private, parameter :: kbtointu = 6.241509E-4_q
                                    ! kbar to internal pressure units
   real(q), private, parameter :: one = 1.0e0_q
   real(q), private, parameter :: two = 2.0e0_q
   real(q), private, parameter :: three = 3.0e0_q
   real(q), private, parameter :: zero = 0.0e0_q

   real(q), private :: cos_alpha
   real(q), private :: cos_beta
   real(q), private :: cos_gamma
   real(q), private :: delta_t
   real(q), private :: E_kin_metric
   real(q), private :: H0, barostat_energy0
   real(q), private :: kinetic_energy
   real(q), private :: pressure
   real(q), private :: stress_energy
   real(q), private :: temperature

   real(q), private, dimension (3,3) :: constraint_stress
   real(q), private, dimension (3,3) :: kinetic_stress
   real(q), private, dimension (3,3) :: metric_tensor
   real(q), private, dimension (3,3) :: metric_tensor_momenta
   real(q), private, allocatable, dimension (:,:) :: momentum
   real(q), private, dimension (3,3) :: reciprocal_metric_tensor
   real(q), private, dimension (3,3) :: rotation_matrix

   type (stress_type), private :: stress ! external stress in internal units

!**************************************************************************
!  Private Module Subroutines

   private calculate_internal_pressure
   private calculate_kinetic_energy
   private impose_metric_momenta_constraints
   private set_up_MD
   private update_cell
   private update_metric_tensor_components
   private update_metric_tensor_momenta
!tb beg
   private reader_npt
!tb end 

!**************************************************************************
!  Start of module

 contains

!**************************************************************************
 subroutine reader_npt(iu5,iu0,pmass, external_stress) 
   use vaspxml
   implicit none
   integer :: iu5,iu0,idum,n,ierr
   real(q) ::  pmass, external_pressure, external_stress(6)
   logical ::  LDUM,LOPEN
   complex :: cdum
   character (1) ::   CHARAC

    LOPEN=.FALSE.
    OPEN(UNIT=IU5,FILE='INCAR',STATUS='OLD')

     PMASS = -1.0
      CALL RDATAB(LOPEN,'INCAR',IU5,'PMASS','=','#',';','F', &
     &            IDUM,PMASS,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''PMASS'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('PMASS','F',IDUM,PMASS,CDUM,LDUM,CHARAC,N)

! Target pressure tb: this is redundant - for this purpose we have variable PSTRESS 
      external_pressure = 0.0
      CALL RDATAB(LOPEN,'INCAR',IU5,'EXTERNAL_PRESSURE','=','#',';','F', &
     &            IDUM,EXTERNAL_PRESSURE,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''EXTERNAL_PRESSURE'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR('EXTERNAL_PRESSURE','F',IDUM,EXTERNAL_PRESSURE,CDUM,LDUM,CHARAC,N)

! Target stress = external_pressure + external_stress
      external_stress = 0.0
      CALL RDATAB(LOPEN,'INCAR',IU5,'EXTERNAL_STRESS','=','#',';','F', &
     &            IDUM,EXTERNAL_STRESS,CDUM,LDUM,CHARAC,N,6,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<6))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''EXTERNAL_STRESS'' from file INCAR.'
         GOTO 150
      ENDIF
      CALL XML_INCAR_V('EXTERNAL_STRESS','F',IDUM,EXTERNAL_STRESS,CDUM,LDUM,CHARAC,N)

  CLOSE(IU5)
  RETURN


  150 CONTINUE
      IF (IU0>=0) &
      WRITE(IU0,151) IERR,N
  151 FORMAT(' Error code was IERR=',I1,' ... . Found N=',I5,' data.')
end subroutine reader_npt

!**************************************************************************


!**************************************************************************
subroutine calculate_internal_pressure( debug, NIONS, NTYP, NITYP, cell,  &
                   internal_pressure, POMASS, internal_stress, SNOSE, VEL ) 
!**************************************************************************
!
!     Program:          VASP
!
!     Subroutine:       calculate_internal_pressure()
!
!     Variables:
!
!     Created:          11/06/2007 by E. Hernandez
!
!     Purpose:          it calculates the internal pressure
!
!     Routines used:    ()
!     Modified:         -
!**************************************************************************
!  used modules

   use lattice
!   use stress_module

!**************************************************************************
!  no implicits, please!

   implicit none

!**************************************************************************
!  Shared variables

   logical, intent(IN) :: debug

   integer, intent (IN) :: NIONS
   integer, intent (IN) :: NTYP

   integer, intent (IN), dimension (:) :: NITYP

   real(q), intent (OUT)  :: internal_pressure
   
   real(q), intent (IN), dimension (NTYP) :: POMASS
   real(q), intent (IN), dimension (4) :: SNOSE
   real(q), intent (IN), dimension (3,NIONS) :: VEL

   type (latt), intent (IN) :: cell
   type (stress_type), intent (IN) :: internal_stress

!**************************************************************************
!  Local variables

   integer :: i, i_species
   integer :: ni, nt

   real(q) :: ddot, factor 
   real(q) :: total_internal_stress(3,3)
   real(q) :: vector(3)

!**************************************************************************
!  Start of subroutine

   if ( debug ) write(*,*) 'Entering calculate_internal_pressure()'

! calculate the kinetic stress 

   kinetic_stress = zero

   ni = 1

   do nt=1, NTYP
      do ni=ni, NITYP(nt) + ni - 1

         kinetic_stress(1,1) = kinetic_stress(1,1) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(1,ni)
         kinetic_stress(1,2) = kinetic_stress(1,2) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(2,ni)
         kinetic_stress(1,3) = kinetic_stress(1,3) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(3,ni)

         kinetic_stress(2,1) = kinetic_stress(2,1) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(1,ni)
         kinetic_stress(2,2) = kinetic_stress(2,2) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(2,ni)
         kinetic_stress(2,3) = kinetic_stress(2,3) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(3,ni)

         kinetic_stress(3,1) = kinetic_stress(3,1) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(1,ni)
         kinetic_stress(3,2) = kinetic_stress(3,2) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(2,ni)
         kinetic_stress(3,3) = kinetic_stress(3,3) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(3,ni)

      end do
   end do

   factor = half / ( SNOSE(1) * SNOSE(1) )

   kinetic_stress = factor * kinetic_stress

   total_internal_stress = ( kinetic_stress - internal_stress % lattice -    &
                             constraint_stress ) / ( cell % omega )
!  total_internal_stress = ( kinetic_stress - internal_stress % lattice ) / &
!                            ( cell % omega )

   internal_pressure = ( two / three ) *                                     &
          ddot( 9, total_internal_stress, 1, metric_tensor, 1 )

   if ( debug ) write(*,*) 'Exiting calculate_internal_pressure()'

end subroutine calculate_internal_pressure

!**************************************************************************
subroutine calculate_kinetic_energy( NDEGREES_OF_FREEDOM, NIONS, NITYP,   &
                                     NTYP, POMASS, POTIM, SNOSE )
!**************************************************************************
!
!     Program:          VASP
!
!     Subroutine:       calculate_kinetic_energy()
!
!     Variables:      
!                       
!     Created:          1/03/07 by E. Hernandez
!
!     Purpose:          calculates the kinetic energy and temperature
!                                              
!     Routines used:    ()
!     Modified:         -
!**************************************************************************
!  used modules

!**************************************************************************
!  no implicits, please!

   implicit none

!**************************************************************************
!  Shared variables

   integer, intent (IN) :: ndegrees_of_freedom
   integer, intent (IN) :: NIONS
   integer, intent (IN) :: NTYP

   integer, intent (IN), dimension (NTYP) :: NITYP

   real(q), intent (IN) :: POTIM

   real(q), intent (IN), dimension (NTYP) :: POMASS
   real(q), intent (IN), dimension (4) :: SNOSE

!**************************************************************************
!  Local variables

   integer :: ni, nt

   real(q) :: ddot, factor

   real(q), dimension (3) :: vector

!**************************************************************************
!  Start of subroutine

   if ( debug ) write(*,*) 'Entering calculate_kinetic_energy()'

   kinetic_energy = zero

   ni=1

   do nt=1, NTYP
      do ni=ni, NITYP(nt) + ni - 1

         call dgemv( 'N', 3, 3, one, reciprocal_metric_tensor, 3, momentum(1:3,ni), 1,    &
                     zero, vector, 1 )

         kinetic_energy = kinetic_energy +                                &
             ddot( 3, vector, 1, momentum(1:3,ni), 1 ) / POMASS(nt)

      end do
   end do

   factor = half / ( SNOSE(1) * SNOSE(1) ) 
   kinetic_energy = factor * kinetic_energy 

   temperature = two * kinetic_energy /                                   &
     ( float( ndegrees_of_freedom ) * boltzmann_k )

   if ( debug ) write(*,*) 'Exiting calculate_kinetic_energy()'

end subroutine calculate_kinetic_energy

!**************************************************************************
subroutine impose_metric_momenta_constraints( ISIF, barostat_mass, SNOSE, &
                                              cell )
!**************************************************************************
!
!     Program:	        VASP	
!
!     Subroutine:       impose_momenta_constraints()
!
!     Variables:      
!                       
!     Created:          25/02/2008 by E. Hernandez
!
!     Purpose:          in the case of fixed cell shape dynamics, this 
!			subroutine imposes the necessary contraints on the
!			off-diagonal metric tensor momenta components.
!			
!                                              
!     Routines used:    ()
!     Modified:         
!
!**************************************************************************
!  used modules

   use lattice
!   use stress_module

!**************************************************************************
!  no implicits, please!

   implicit none

!**************************************************************************
!  Shared variables

   integer, intent (IN) :: ISIF

   real(q), intent (IN) :: barostat_mass

   real(q), intent (IN), dimension (4) :: SNOSE

   type (latt), intent (IN) :: cell

!**************************************************************************
!  Local variables

   real(q) :: a_modulus, b_modulus, c_modulus
   real(q) :: ddot, det_metric, factor, trace
   real(q) :: dot_a_modulus, dot_b_modulus, dot_c_modulus

   real(q), dimension(3,3) :: constraint_velocity
   real(q), dimension(3,3) :: gpi
   real(q), dimension(3,3) :: gpig
   real(q), dimension(3,3) :: matrix_1, matrix_2, matrix_3
   real(q), dimension(3) :: vector

!**************************************************************************
!  Start of subroutine

   if ( debug ) write(*,*) 'Entering impose_momenta_constraints()'

   call dgemm( 'N', 'N', 3, 3, 3, one, metric_tensor, 3,                     &
               metric_tensor_momenta, 3, zero, gpi, 3 )
   call dgemm( 'N', 'N', 3, 3, 3, one, gpi, 3,                               &
               metric_tensor, 3, zero, gpig, 3 )

   a_modulus = sqrt( ddot( 3, cell % a(1:3,1), 1, cell % a(1:3,1), 1 ) )
   b_modulus = sqrt( ddot( 3, cell % a(1:3,2), 1, cell % a(1:3,2), 1 ) )
   c_modulus = sqrt( ddot( 3, cell % a(1:3,3), 1, cell % a(1:3,3), 1 ) )

   det_metric = cell % omega * cell % omega

   factor = SNOSE(1) / ( barostat_mass * det_metric )

!  dot_a_modulus = sign( one, gpig(1,1) ) * sqrt( factor * abs( gpig(1,1) ) )
!  dot_b_modulus = sign( one, gpig(2,2) ) * sqrt( factor * abs( gpig(2,2) ) )
!  dot_c_modulus = sign( one, gpig(3,3) ) * sqrt( factor * abs( gpig(3,3) ) )

   dot_a_modulus = factor * gpig(1,1) / ( two * a_modulus )
   dot_b_modulus = factor * gpig(2,2) / ( two * b_modulus )
   dot_c_modulus = factor * gpig(3,3) / ( two * c_modulus )

! ISIF == 8 is the case when cell vectors are constrained to the same length
! as well as having constant angles between cell vectors (i.e. ISIF==7)

   if ( ISIF == 8 ) then 

      trace = ( gpig(1,1) + gpig(2,2) + gpig(3,3) ) / three
   
      gpig(1,1) = trace 
      gpig(2,2) = trace 
      gpig(3,3) = trace 

! ISIF == 9 is the case when |a| = |b| and |c| varies independently

   else if ( ISIF == 9 ) then

      trace = ( gpig(1,1) + gpig(2,2) ) / two

      gpig(1,1) = trace 
      gpig(2,2) = trace 

! ISIF == 10 specifically for Dario's calculations on graphene

   else if ( ISIF == 10 ) then

      gpig(3,3) = zero
      gpig(1,3) = zero
      gpig(2,3) = zero
      gpig(3,1) = zero
      gpig(3,2) = zero

   end if

   constraint_velocity(1,1) = gpig(1,1) * factor
   constraint_velocity(2,2) = gpig(2,2) * factor
   constraint_velocity(3,3) = gpig(3,3) * factor

   constraint_velocity(1,2) = ( dot_a_modulus * b_modulus +            &
                                a_modulus * dot_b_modulus ) * cos_gamma 

   constraint_velocity(1,3) = ( dot_a_modulus * c_modulus +            &
                                a_modulus * dot_c_modulus ) * cos_beta 

   constraint_velocity(2,3) = ( dot_b_modulus * c_modulus +            &
                                b_modulus * dot_c_modulus ) * cos_alpha 

   constraint_velocity(2,1) = constraint_velocity(1,2)
   constraint_velocity(3,1) = constraint_velocity(1,3)
   constraint_velocity(3,2) = constraint_velocity(2,3)

   call dgemm( 'N', 'N', 3, 3, 3, one, reciprocal_metric_tensor, 3,          &
               constraint_velocity, 3, zero, gpi, 3 )
   call dgemm( 'N', 'N', 3, 3, 3, one, gpi, 3,                               &
               reciprocal_metric_tensor, 3, zero, gpig, 3 )

   factor = SNOSE(1) / ( barostat_mass * det_metric )

   gpig = gpig / factor
!  print*, 'metric_tensor_momenta ', metric_tensor_momenta
!  print*, 'gpig                  ', gpig

!  metric_tensor_momenta = metric_tensor_momenta + gpig

   factor = two / ( delta_t * SNOSE(1) )

   constraint_stress = factor * ( metric_tensor_momenta - gpig )
!print'(''m '',9f10.4)',metric_tensor_momenta*factor/kbtointu
!print'(''g '',9f10.4)',gpig*factor/kbtointu
!print'(9f10.4)',constraint_stress/kbtointu 
   metric_tensor_momenta = gpig

! we must re-evaluate the metric kinetic energy once the momenta are changed

!  factor = barostat_mass * det_metric
!  factor = one / factor

!  call dgemm( 'N', 'N', 3, 3, 3, factor, metric_tensor_momenta, 3,   &
!              metric_tensor, 3, zero, matrix_1, 3 )

!  call dgemm( 'N', 'N', 3, 3, 3, one, matrix_1, 3,                       &
!              metric_tensor_momenta, 3, zero, matrix_2, 3 )

!  matrix_3 = transpose( matrix_1 )

!  factor = half * barostat_mass * det_metric

!  E_kin_metric = factor * ddot( 9, matrix_1, 1, matrix_3, 1 )

   if ( debug ) write(*,*) 'Exiting impose_metric_momenta_constraints()'

end subroutine impose_metric_momenta_constraints

!**************************************************************************
subroutine set_up_MD( cell )
!**************************************************************************
!
!     Program:	       VASP	
!
!     Subroutine:      set_up_MD() 
!
!     Variables:      
!                       
!     Created:         19/02/2007 
!
!     Purpose:         given the lattice parameters of the simulation cell,
!                      in a, b, c, it constructs the metric tensor g, it 
!                      rotates the cell so that the parameters have the 
!                      form a=(a,0,0), b=(b1,b2,0), c=(c1,c2,c3), and then
!                      the reciprocal cell and reciprocal metric tensor
!                      are constructed and returned. 
!			
!			
!                                              
!     Routines used:    ()
!     Modified:         
!
!**************************************************************************
!  used modules

   use lattice

!tb beg
   use mymath
!tb end 

!**************************************************************************
!  no implicits, please!

   implicit none

!**************************************************************************
!  Shared variables

   type (latt), intent (INOUT) :: cell

!**************************************************************************
!  Local variables

   real(q), parameter :: zero = 0.0d0 

   real(q) :: ddot
   real(q) :: det
   real(q) :: factor
   real(q) :: norm, norm1, norm2
   real(q) :: theta, ct, st, v1, v2, u1, u2, w1, w2

   real(q), dimension (3,3) :: identity
   real(q), dimension (3,3) :: matrix_1
   real(q), dimension (3,3) :: matrix_2
   real(q), dimension (3,3) :: new_cell
   real(q), dimension (3,3) :: old_cell
   real(q), dimension (3,3) :: rotation_matrix_1
   real(q), dimension (3,3) :: rotation_matrix_2
   real(q), dimension (3) :: original_vector
   real(q), dimension (3) :: rotated_vector
   real(q), dimension (3) :: sum_vector
   real(q), dimension (3) :: vector
   real(q), dimension (3) :: u, v

!tb beg
   integer :: ninfo
!tb end

!**************************************************************************
!  Start of subroutine

   if ( debug ) write(*,*) 'Entering set_up()'

! first we construct the metric tensor

   call dgemm( 'T', 'N', 3, 3, 3, one, cell % a, 3, cell % a, 3, zero,     &
              metric_tensor, 3 )

! now we 'rotate' the cell so that the lattice vectors have the following
! structure: a=(a,0,0), b=(b1,b2,0), c=(c1,c2,c3)

   new_cell(1,1) = sqrt( metric_tensor(1,1) )
   new_cell(2,1) = zero
   new_cell(3,1) = zero

   new_cell(1,2) = metric_tensor(1,2) / sqrt( metric_tensor(1,1) )
   new_cell(2,2) = sqrt( metric_tensor(2,2) -                              &
             metric_tensor(1,2) * metric_tensor(1,2) / metric_tensor(1,1) )
   new_cell(3,2) = zero

   new_cell(1,3) = metric_tensor(1,3) / sqrt( metric_tensor(1,1) )
   new_cell(2,3) = ( metric_tensor(1,1) * metric_tensor(2,3) -             &
                metric_tensor(1,2) * metric_tensor(1,3) ) /                & 
      sqrt( metric_tensor(1,1) * metric_tensor(1,1) * metric_tensor(2,2) - & 
            metric_tensor(1,1) * metric_tensor(1,2) * metric_tensor(1,2) )
   new_cell(3,3) = sqrt( metric_tensor(3,3) -                              &
           new_cell(1,3) * new_cell(1,3) - new_cell(2,3) * new_cell(2,3) )

   old_cell = cell % a

! store, for use in the case of fixed-cell dynamics, the cosines of angles

   cos_alpha = ddot( 3, cell % a(1:3,2), 1, cell % a(1:3,3), 1 ) /     &
                   ( cell % anorm(2) * cell % anorm(3) )
   cos_beta  = ddot( 3, cell % a(1:3,1), 1, cell % a(1:3,3), 1 ) /     &
                   ( cell % anorm(1) * cell % anorm(3) )
   cos_gamma = ddot( 3, cell % a(1:3,1), 1, cell % a(1:3,2), 1 ) /     &
                   ( cell % anorm(1) * cell % anorm(2) )


! now we can calculate the rotation matrices that turn these cell vectors
! back to the original ones; this is done so that as the dynamics proceeds
! we can orient the changing cell with the same orientation that the
! original cell had

   identity = zero
   identity(1,1) = one
   identity(2,2) = one
   identity(3,3) = one

! we will first calculate the rotation that is needed to superimpose the
! diagonals of the rotated cell and the original cell

!  original_vector = old_cell(1:3,1) / sqrt( metric_tensor(1,1) )
!  rotated_vector = new_cell(1:3,1) / sqrt( metric_tensor(1,1) )

   original_vector = old_cell(1:3,1) + old_cell(1:3,2) + old_cell(1:3,3)
   rotated_vector = new_cell(1:3,1) + new_cell(1:3,2) + new_cell(1:3,3)

   norm = sqrt( ddot( 3, original_vector, 1, original_vector, 1 ))
   original_vector = original_vector / norm
   norm = sqrt( ddot( 3, rotated_vector, 1, rotated_vector, 1 ))
   rotated_vector = rotated_vector / norm

   sum_vector = original_vector + rotated_vector

   factor = one / ( one + ddot( 3, original_vector, 1, rotated_vector, 1 ) )

   call dgemm( 'T', 'T', 3, 3, 1, two, original_vector, 1,               &
               rotated_vector, 3, zero, matrix_1, 3 )
   call dgemm( 'T', 'T', 3, 3, 1, factor, sum_vector, 1,                 &
               sum_vector, 3, zero, matrix_2, 3 )

   rotation_matrix_1 = identity + matrix_1 - matrix_2

! now we need to rotate around the body diagonal by some angle theta
! calculate this theta angle

   vector = matmul( rotation_matrix_1, new_cell(1:3,1) ) ! rotated a vector

   u = dot_product( old_cell(1:3,1), original_vector ) * original_vector
   v = u - old_cell(1:3,1)

   norm1 = sqrt( dot_product( v, v ) )
   vector = vector - old_cell(1:3,1)
   norm2 = sqrt( dot_product( vector, vector ) )

   theta = two * asin( half * norm2 / norm1 )

   ct = cos( theta ); st = sin( theta )
   u1 = original_vector(1)
   v1 = original_vector(2)
   w1 = original_vector(3)
   u2 = original_vector(1) * original_vector(1)
   v2 = original_vector(2) * original_vector(2)
   w2 = original_vector(3) * original_vector(3)

   rotation_matrix_2(1,1) = u2 + ( v2 + w2 ) * ct
   rotation_matrix_2(2,1) = u1 * v1 * ( one - ct ) + w1 * st
   rotation_matrix_2(3,1) = u1 * w1 * ( one - ct ) - v1 * st
   rotation_matrix_2(1,2) = u1 * v1 * ( one - ct ) - w1 * st
   rotation_matrix_2(2,2) = v2 + ( u2 + w2 ) * ct
   rotation_matrix_2(3,2) = v1 * w1 * ( one - ct ) + u1 * st
   rotation_matrix_2(1,3) = u1 * w1 * ( one - ct ) + v1 * st
   rotation_matrix_2(2,3) = v1 * w1 * ( one - ct ) - u1 * st
   rotation_matrix_2(3,3) = w2 + ( u2 + v2 ) * ct

! now the composed rotation matrix is the product of the previous two
  ! rotation_matrix = matmul( rotation_matrix_2, rotation_matrix_1 )

!tb test beg
   rotation_matrix=new_cell
   CALL SVDINVERSE(rotation_matrix,3,ninfo)
   !rotation_matrix=matmul(rotation_matrix,old_cell)
   rotation_matrix=matmul(old_cell,rotation_matrix)
!tb test end

  
! now update based on the current rotated cell the reciprocal cell etc
! strictly speaking this is not needed anymore!

   call lattic( cell )

! we can then construct the metric tensor of the reciprocal cell

   call dgemm( 'T', 'N', 3, 3, 3, one, cell % b, 3, cell % b, 3, zero,   &
               reciprocal_metric_tensor, 3 )

   if ( debug ) write(*,*) 'Exiting set_up_MD()'

end subroutine set_up_MD


!**************************************************************************
subroutine step_NPT(INIT, ISIF, NDEGREES_OF_FREEDOM, NIONS, NITYP, NTYP, & 
                  cell, PSTRESS,&
                  potential_energy,  &
                  POMASS, POSION, POSIOC, POTIM, SMASS,&
                  SNOSE, TEMP, TIFOR,TSIF, VEL, D2C, D2, IU,IU5,IU0 )
!**************************************************************************
! !**************************************************************************
! subroutine step_NPT(INIT, ISIF, NDEGREES_OF_FREEDOM, NIONS, NITYP, NTYP, & 
!                   cell, PSTRESS,&
!                   potential_energy,  barostat_energy, thermostat_energy,   &
!                   POMASS, POSION, POSIOC, POTIM, SMASS,&
!                   SNOSE, TEMP, TIFOR,TSIF, VEL, D2C, D2, IU,IU5,IU0 )
! !**************************************************************************
!
!     Program:	        VASP	
!
!     Subroutine:       step_NPT()
!
!     Variables:      
!                       
!     Created:          27/02/2007 by E. Hernandez
!
!     Purpose:          This subroutine is in charge of projecting the 
!			momenta (velocities) forward by one half of the 
!			time step, and the positions by a full time step.
!			This is done for ions, and if needed, for the 
!			Nose-Poincare thermostat and the Souza-Martins
!			barostat. This subroutine must be called before
!			the evaluation of energy, forces and stress, and
!			constitutes the first part of the time step. The
!			second part is done in subroutine post_step(),
!			called immediately after the evaluation of energy,
!			forces and stress. For details of implementation,
!			see E. Hernandez, JCP vol. 115, 10282 (2001).
!			
!                                              
!     Routines used:    ()
!     Modified:         
!
!**************************************************************************
!  used modules

   use lattice
!tb beg
   use constant
!tb end
!   use stress_module

!**************************************************************************
!  no implicits, please!

   implicit none

!**************************************************************************
!  Shared variables

   integer, intent (INOUT) :: INIT
   integer, intent (IN) :: ISIF, IU,IU5,IU0
   integer, intent (IN) :: NDEGREES_OF_FREEDOM
   integer, intent (IN) :: NIONS
   integer, intent (IN) :: NTYP

   integer, intent (IN), dimension (:) :: NITYP

   
   real(q), intent (IN) :: potential_energy
   real(q) :: barostat_energy
   real(q) :: thermostat_energy
!    real(q), intent (OUT) :: barostat_energy
!    real(q), intent (OUT) :: thermostat_energy

   real(q), intent (IN) :: PSTRESS
   real(q), intent (IN) :: POTIM
   real(q), intent (IN) :: SMASS
   real(q), intent (IN) :: TEMP

   real(q), intent (INOUT), dimension(3,NIONS) :: D2C, D2
   real(q), intent (IN), dimension(NTYP) :: POMASS
   real(q), intent (INOUT), dimension(3,NIONS) :: POSION
   real(q), intent (INOUT), dimension(3,NIONS) :: POSIOC
   real(q), intent (INOUT), dimension (4) :: SNOSE
   real(q), intent (IN), dimension (3,NIONS) :: TIFOR
   real(q), intent (INOUT), dimension (3,NIONS) :: VEL
   real(q), intent (IN), dimension (3,3) :: TSIF

   type (latt), intent (INOUT) :: cell
   

!**************************************************************************
!  Local variables

   integer :: i, n, ni, nt

   real(q) :: ddot, factor, S, S_new, S_temp, sum
   real(q) :: constant_of_motion
   real(q) :: internal_pressure
   real(q) :: total_energy

   real(q), dimension(3,3) :: matrix_a
   real(q), dimension(3,3) :: matrix_b
   real(q), dimension(3,3) :: metric_tensor_momenta_tmp
   real(q), dimension(3) :: vector
   real(q), dimension(6) :: estress
   real(q),save :: barostat_mass
   real(q) :: external_pressure   
   type (stress_type) :: external_stress
   type (stress_type) :: internal_stress
   logical,save :: lfirst=.true.

!tb remove this
   real(q) :: atmp1(3,3),atmp2(3,3),atmp3(3,3)
!tb end

!**************************************************************************
!  Start of subroutine

!   IF (INIT==0) THEN
!     SNOSE(1) = one   
!     !tb beg
!     !SNOSE(2:4) = zero
!     SNOSE(2) = zero
!     SNOSE(3)=SNOSE(1)
!     SNOSE(4)=zero
!     !tb end
!   END IF

!   IF (cell%INITlatv==1) THEN 
!     cell%avel=cell%avel/POTIM*SNOSE(3)
!   ELSE 
!     cell%avel=0._q
!   END IF

! if this is the first time the subroutine is called, do some setting up
! and return

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
initialization:   IF ( lfirst) THEN
      external_pressure=PSTRESS
      call reader_npt(iu5,iu0,barostat_mass,estress) 
  
!c tb TODO: stop if barostat_mas does not have a reasonable value!!!
      if( barostat_mass > 0.0_q) then
         external_stress % cartesian(1,1) = estress(1)
         external_stress % cartesian(2,2) = estress(2)
         external_stress % cartesian(3,3) = estress(3)
         external_stress % cartesian(1,2) = estress(4)
         external_stress % cartesian(2,3) = estress(5)
         external_stress % cartesian(3,1) = estress(6)
         external_stress % cartesian(2,1) = estress(4)
         external_stress % cartesian(3,2) = estress(5)
         external_stress % cartesian(1,3) = estress(6)
      endif
      if( barostat_mass > 0.0_q .AND. IU>=0 ) then
!         if( SMASS > 0.0_q ) then
!           write(IU,'(''Using Nose-Poincare thermostat '')')
!           write(IU,'(''WARNING: multiplying input SMASS by 50 '')')
!         endif
        if( ISIF == 3 .AND. IU>=0 ) then
          write(IU,'(''Constant Pressure Simulation; PMASS = '', f6.2)') barostat_mass
          write(IU,'(''Variable cell shape '')')
        elseif( ISIF == 7 .AND. IU>=0 ) then
          write(IU,'(''Constant Pressure Simulation; PMASS = '', f6.2)') barostat_mass
          write(IU,'(''Fixed cell shape '')')
        else
        endif
        write(IU,'(''Target Stress: '', 3f10.2,3f7.2,'' kB '')') &
        &  estress(1)+external_pressure,&
        &  estress(2)+external_pressure,estress(3)+external_pressure, &
        &  estress(4),estress(5),estress(6)
      endif

      if(IU>=0)then
        open(unit=334,file='energy.dat')
        open(unit=335,file='volume.dat')
      endif

      call set_up_MD( cell )

      if (INIT==0) THEN
        SNOSE(1) = one   
        !tb beg
        !SNOSE(2:4) = zero
        SNOSE(2) = zero
        SNOSE(3)=SNOSE(1)
        SNOSE(4)=zero
        !tb end
      endif

      !c if the velocities for the cell parameters were read-in from POSCAR:
      if (cell%INITlatv==1) then
         !wcs: metric_tensor_momenta=cell%avel

         !cell%avel=cell%avel/POTIM*SNOSE(3)  !!!*barostat_mass*cell % omega * cell % omega
         cell%avel=cell%avel/POTIM 
         metric_tensor_momenta=matmul(transpose(cell%a),(cell%avel))
         metric_tensor_momenta=metric_tensor_momenta+transpose(metric_tensor_momenta)

         metric_tensor_momenta=MATMUL(reciprocal_metric_tensor,metric_tensor_momenta)
         metric_tensor_momenta=MATMUL(metric_tensor_momenta,reciprocal_metric_tensor)
        ! metric_tensor_momenta=MATMUL(metric_tensor,metric_tensor_momenta)
!         metric_tensor_momenta=MATMUL(metric_tensor_momenta,metric_tensor)
         metric_tensor_momenta=metric_tensor_momenta *barostat_mass * cell % omega * cell % omega/SNOSE(3)
!         matrix_a=metric_tensor_momenta
!         matrix_b = transpose( matrix_a )
!         E_kin_metric = half * ddot( 9, matrix_a, 1, matrix_b, 1 )/ (barostat_mass * cell % omega * cell % omega)


       factor = one / ( barostat_mass * cell % omega * cell % omega )

        call dgemm( 'N', 'N', 3, 3, 3, factor, metric_tensor_momenta, 3,       &
                 metric_tensor, 3, zero, matrix_a, 3 )

        matrix_b = transpose( matrix_a )

        factor = half * barostat_mass * cell % omega * cell % omega

        E_kin_metric = factor * ddot( 9, matrix_a, 1, matrix_b, 1 )      
        IF (IU>=0) write(*,*) "E_kin_metric_",E_kin_metric
      else
        metric_tensor_momenta = zero
        E_kin_metric = zero
      endif

      thermostat_energy = half * SMASS * SNOSE(2) * SNOSE(2) +            &
           ndegrees_of_freedom * boltzmann_k * TEMP * log( SNOSE(1) )

      delta_t = sqrt( evtojoule / amtokg / 1.0e-20_q ) * POTIM * 1.0e-15_q

      if ( ( ISIF == 3 ) .or. ( ISIF >= 7 ) ) then

         pressure = external_pressure * kbtointu

         stress % cartesian = external_stress % cartesian * kbtointu

          call dgemm( 'N', 'N', 3, 3, 3, cell % omega, stress % cartesian, 3, &
                      reciprocal_metric_tensor, 3, zero, stress % lattice, 3 )

         !barostat_energy = E_kin_metric + pressure * cell % omega +      &
         !    half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )
         !barostat_energy0 = E_kin_metric + pressure * cell % omega +  &
         !    half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )

         barostat_energy = E_kin_metric   +  &
             half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )
         barostat_energy0 = E_kin_metric  +  &
             half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )


      else

         barostat_energy = zero

      end if

! construct the momenta conjugate to the atomic lattice positions

      if(.not.allocated(momentum)) allocate( momentum( 3, NIONS ) )
        !tb beg test
        VEL=VEL*SNOSE(3)
        !tb end test 
      ni = 1

      do nt=1, NTYP
         do ni=ni, NITYP(nt) + ni - 1
            
            call dgemv( 'N', 3, 3, POMASS(nt), metric_tensor, 3, VEL(1:3,ni), &
                        1, zero, momentum(1:3,ni), 1 )

         end do
      end do


      momentum = momentum / delta_t
     
      call calculate_kinetic_energy( ndegrees_of_freedom, NIONS, NITYP, NTYP, &
                                     POMASS, POTIM, SNOSE )
      !momentum=momentum/SNOSE(1)
!tb beg
      IF (INIT==0) THEN
        H0 = kinetic_energy + potential_energy + thermostat_energy +        &
             barostat_energy
        SNOSE(4)=H0
      !  constant_of_motion = zero
      ELSE
        H0=SNOSE(4)
      ENDIF
      !H0 = kinetic_energy + potential_energy + thermostat_energy +        &
      !       barostat_energy
      constant_of_motion = zero
!tb end
      lfirst=.false.
      INIT=1
!     return

   END IF initialization
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


!      IF (IU>=0) THEN
!        write(*,*) "kinetic_energy",kinetic_energy
!        write(*,*) "delta_t",delta_t
!        write(*,*) 'INIT',INIT
!        write(*,*) 'SNOSE(1)',SNOSE(1)
!        write(*,*) 'SNOSE(2)',SNOSE(2)
!        write(*,*) 'SNOSE(3)',SNOSE(3)
!        write(*,*) 'SNOSE(4)',SNOSE(4)
!       ENDIF


! first we need to obtain the stress as derivative with respect to metric
! tensor components
   internal_stress % cartesian = -TSIF
   matrix_a = cell % b

   call dgemm( 'N', 'N', 3, 3, 3, one, internal_stress % cartesian, 3,    &
               matrix_a, 3, zero, matrix_b, 3 )

   call dgemm( 'T', 'N', 3, 3, 3, one, matrix_a, 3, matrix_b, 3, zero,    &
               internal_stress % lattice, 3 )

   internal_stress % lattice = half * internal_stress % lattice

! first we bring the momenta in step with the positions 
! ( they are delayed by -dt/2 )

! if we have a thermostat, do it for the thermostat

   if ( SMASS > zero ) then

      !factor = half * delta_t * ( kinetic_energy - potential_energy -   &
      !         ndegrees_of_freedom * boltzmann_k * TEMP *               &
      !         ( log( SNOSE(1) ) + one ) -                              &
      !         E_kin_metric - pressure * cell % omega -                 &
      !         stress_energy - half * SMASS * SNOSE(2) * SNOSE(2) + H0 )

      factor = half * delta_t * ( kinetic_energy - potential_energy -   &
               ndegrees_of_freedom * boltzmann_k * TEMP *               &
               ( log( SNOSE(1) ) + one ) -                              &
               E_kin_metric -                  &
               stress_energy - half * SMASS * SNOSE(2) * SNOSE(2) + H0 )

      SNOSE(2) = SNOSE(2) + factor / SMASS

      thermostat_energy = half * SMASS * SNOSE(2) * SNOSE(2) +            &
           ndegrees_of_freedom * boltzmann_k * TEMP * log( SNOSE(1) )

   end if 

! now, if needed, we update the metric tensor momenta

   if ( ( ISIF == 3 ) .or. ( ISIF >= 7 ) ) then

      factor = one / ( barostat_mass * cell % omega * cell % omega )

      call dgemm( 'N', 'N', 3, 3, 3, factor, metric_tensor_momenta, 3,  &
                  metric_tensor, 3, zero, matrix_a, 3 )

      call dgemm( 'N', 'N', 3, 3, 3, one, matrix_a, 3,                  &
                  metric_tensor_momenta, 3, zero, matrix_b, 3 )

      metric_tensor_momenta_tmp = metric_tensor_momenta

      metric_tensor_momenta = metric_tensor_momenta -                   &
                                 half * delta_t * SNOSE(1) *            &
             ( internal_stress % lattice - kinetic_stress + matrix_b +  &
          ( half * pressure * cell % omega - E_kin_metric ) *  &
               reciprocal_metric_tensor + half * stress % lattice )

      if ( ISIF >= 7 ) then

         call impose_metric_momenta_constraints(ISIF,barostat_mass,SNOSE,cell)

      end if

      metric_tensor_momenta = metric_tensor_momenta_tmp -                   &
                                 half * delta_t * SNOSE(1) *            &
             ( internal_stress % lattice + constraint_stress - kinetic_stress + matrix_b +  &
          ( half * pressure * cell % omega - E_kin_metric ) *  &
               reciprocal_metric_tensor + half * stress % lattice )

      if( ISIF == 10 ) then
         metric_tensor_momenta(3,3) = zero
         metric_tensor_momenta(1,3) = zero
         metric_tensor_momenta(2,3) = zero
         metric_tensor_momenta(3,1) = zero
         metric_tensor_momenta(3,2) = zero
      endif

! and obtain the barostat energy

      factor = one / ( barostat_mass * cell % omega * cell % omega )

      call dgemm( 'N', 'N', 3, 3, 3, factor, metric_tensor_momenta, 3,       &
                  metric_tensor, 3, zero, matrix_a, 3 )

      matrix_b = transpose( matrix_a )

      factor = half * barostat_mass * cell % omega * cell % omega

      E_kin_metric = factor * ddot( 9, matrix_a, 1, matrix_b, 1 )
      IF (IU>=0) write(*,*) "E_kin_metric",E_kin_metric

      !barostat_energy = E_kin_metric + pressure * cell % omega +        &
      !       half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )

      barostat_energy = E_kin_metric +        &
             half * ddot( 9, stress % lattice, 1, metric_tensor, 1 )


   end if 

! now we update the atomic momenta

   do i=1, NIONS

      D2C(1,i) = ddot( 3, cell % a(1:3,1), 1, TIFOR(1:3,i), 1 )
      D2C(2,i) = ddot( 3, cell % a(1:3,2), 1, TIFOR(1:3,i), 1 )
      D2C(3,i) = ddot( 3, cell % a(1:3,3), 1, TIFOR(1:3,i), 1 )

   end do

   momentum = momentum + half * delta_t * SNOSE(1) * D2C

   call calculate_kinetic_energy( ndegrees_of_freedom, NIONS, NITYP, NTYP, &
                                  POMASS, POTIM, SNOSE )

! the following array is needed by VASP to calculate the correct KE

   ni=1

   do nt=1, NTYP
      do ni=ni, NITYP(nt) + ni - 1

         D2(1,ni) = ddot( 3, reciprocal_metric_tensor(1,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)
         D2(2,ni) = ddot( 3, reciprocal_metric_tensor(2,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)
         D2(3,ni) = ddot( 3, reciprocal_metric_tensor(3,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)
      enddo
   enddo

!   D2 = D2 * delta_t / SNOSE(1)
   D2 = D2 / SNOSE(1)

! calculate the internal pressure and the barostat and thermostat energies

   call calculate_internal_pressure( debug, NIONS, NTYP, NITYP, cell,       &
                     internal_pressure, POMASS, internal_stress, SNOSE, D2 )

   D2 = D2 * delta_t

   if(IU>=0)write(335,*) internal_pressure, cell % omega

! from all this we can obtain the total energy and the constant of motion

   total_energy = kinetic_energy + potential_energy + thermostat_energy +   &
                  barostat_energy

   constant_of_motion = SNOSE(1) * ( total_energy - H0 )

   if(IU>=0)then
    write(*,*) "'EKIN:",kinetic_energy+E_kin_metric,2*(kinetic_energy+E_kin_metric)/BOLKEV/(NDEGREES_OF_FREEDOM+6),2*(E_kin_metric)/BOLKEV/(6)
     write(334,'(6g15.7)') kinetic_energy, potential_energy, barostat_energy, &
               thermostat_energy, total_energy, constant_of_motion
   endif
!  E_kin = kinetic_energy

! ok, so now we can shift the momenta forward by dt/2

! first the ionic momenta

   momentum = momentum + half * delta_t * SNOSE(1) * D2C

! from the momenta, obtain the velocities and the kinetic stress

   kinetic_stress = zero

   ni=1
   
   do nt=1, NTYP
      do ni=ni, NITYP(nt) + ni - 1

         VEL(1,ni) = ddot( 3, reciprocal_metric_tensor(1,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)
         VEL(2,ni) = ddot( 3, reciprocal_metric_tensor(2,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)
         VEL(3,ni) = ddot( 3, reciprocal_metric_tensor(3,1:3), 1,      &
                                  momentum(1:3,ni), 1 ) / POMASS(nt)

         kinetic_stress(1,1) = kinetic_stress(1,1) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(1,ni)
         kinetic_stress(1,2) = kinetic_stress(1,2) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(2,ni)
         kinetic_stress(1,3) = kinetic_stress(1,3) +                   &
                                 POMASS(nt) * VEL(1,ni) * VEL(3,ni)

         kinetic_stress(2,1) = kinetic_stress(2,1) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(1,ni)
         kinetic_stress(2,2) = kinetic_stress(2,2) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(2,ni)
         kinetic_stress(2,3) = kinetic_stress(2,3) +                   &
                                 POMASS(nt) * VEL(2,ni) * VEL(3,ni)

         kinetic_stress(3,1) = kinetic_stress(3,1) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(1,ni)
         kinetic_stress(3,2) = kinetic_stress(3,2) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(2,ni)
         kinetic_stress(3,3) = kinetic_stress(3,3) +                   &
                                 POMASS(nt) * VEL(3,ni) * VEL(3,ni)

      end do
   end do

   VEL = VEL * delta_t  !/ SNOSE(1) ! velocities as needed by VASP

   kinetic_stress = half * kinetic_stress / ( SNOSE(1) * SNOSE(1) )

!  with the new momenta, obtain the kinetic energy

   call calculate_kinetic_energy( ndegrees_of_freedom, NIONS, NITYP, NTYP, &
                                  POMASS, POTIM, SNOSE )

! if this is a constant pressure, then update the metric tensor momenta
   if ( ( ISIF == 3 ) .or. ( ISIF >= 7 ) )                           &
      call update_metric_tensor_momenta( debug, barostat_mass, cell, &
                        internal_stress, SNOSE, ISIF )

! if only volume (not shape) fluctuations are allowed, impose corresponding
! restrictions on the metric tensor momenta

!   if ( ISIF >= 7 ) then

!      call impose_metric_momenta_constraints( ISIF, barostat_mass, SNOSE, cell )

!   end if

      if( ISIF == 10 ) then
         metric_tensor_momenta(3,3) = zero
         metric_tensor_momenta(1,3) = zero
         metric_tensor_momenta(2,3) = zero
         metric_tensor_momenta(3,1) = zero
         metric_tensor_momenta(3,2) = zero
      endif

! now, if needed, update the thermostat velocity (to half step) and the 
! thermostat position (to full step)

!  write(6,*) 'c'

   if ( SMASS > zero ) then

! we need the energy associated to the stress (if variable cell)

      if ( ( ISIF == 3 ) .or. ( ISIF >= 7 ) ) then  
         stress_energy = half * ddot(9, stress % lattice, 1, metric_tensor, 1) 
      else
         stress_energy = zero
      end if

      !factor = half * delta_t * ( NDEGREES_OF_FREEDOM * boltzmann_k *     &
      !           TEMP * ( one + log( SNOSE(1) ) ) - kinetic_energy +      &
      !           potential_energy + E_kin_metric +                        &
      !           pressure * cell % omega + stress_energy - H0 ) -         &
      !           SMASS * SNOSE(2)
         
      factor = half * delta_t * ( NDEGREES_OF_FREEDOM * boltzmann_k *     &
                 TEMP * ( one + log( SNOSE(1) ) ) - kinetic_energy +      &
                 potential_energy + E_kin_metric +                        &
                  stress_energy - H0 ) -         &
                 SMASS * SNOSE(2)


      SNOSE(2) = -two * factor / ( one + sqrt( one -                      &
                     factor * delta_t / SMASS ) ) / SMASS

! to advance the thermostat position we use recursion

      S = SNOSE(1)

      S_temp = S

      do 

         S_new = S + half * delta_t * ( S + S_temp ) * SNOSE(2)

         if ( abs( S_temp - S_new ) < 1.0d-7 ) exit

         S_temp = S_new
          
      end do

   else 

      S = one
      S_new = one

   end if

!  write(6,*) 'd'

! now, if needed, we can advance the components of the metric tensor to 
! full step; this, like for the momenta, requires an iterative procedure

   E_kin_metric = E_kin_metric * cell % omega * cell % omega

   if ( ( ISIF == 3 ) .or. ( ISIF >= 7 ) ) then

      call update_metric_tensor_components( debug, ISIF, barostat_mass, S,  &
                                            S_new, cell )

! now update the cell vectors, volume, etc, and calculate stress-energy
      call update_cell( debug, cell,barostat_mass,POTIM,SNOSE )
          call dgemm( 'N', 'N', 3, 3, 3, cell % omega, stress % cartesian, 3, &
                      reciprocal_metric_tensor, 3, zero, stress % lattice, 3 )

      stress_energy = half * ddot( 9, stress % lattice, 1, metric_tensor, 1 ) 

      E_kin_metric = E_kin_metric / ( cell % omega * cell % omega )

   else

      stress_energy = zero
      E_kin_metric = zero

   end if

!  write(6,*) 'e'

! store the old positions

   POSIOC = POSION

! now move the positions to full step

   POSION = POSION + half * ( VEL / S_new + VEL / S ) 

   VEL = VEL / SNOSE(1) ! velocities as needed by VASP

! impose periodic boundary conditions

   do n=1, 3
      do i=1, NIONS

         if ( POSION(n,i) >= one ) POSION(n,i) = POSION(n,i) - one
         if ( POSION(n,i) < zero ) POSION(n,i) = POSION(n,i) + one

      end do
   end do

   kinetic_energy = S * S * kinetic_energy / S_new / S_new
   kinetic_stress = S * S * kinetic_stress / S_new / S_new


   if ( SMASS > zero ) then

      S = S_new

      !tb
      SNOSE(3)=SNOSE(1)
      !tb
      SNOSE(1) = S

   end if

!c tb beg 
!c this is only work-around - poscar.F will divide AVEL by POTIM
  !cell%avel=cell%avel*POTIM/SNOSE(3)
  cell%avel=cell%avel*POTIM
!c tb end

!   barostat_energy = barostat_energy - barostat_energy0
   if ( debug ) write(*,*) 'Exiting pre_step()'

end subroutine step_NPT

!*****************************************************************************
subroutine update_cell( debug, cell ,barostat_mass,POTIM,SNOSE)
!*****************************************************************************
!
!     Program: 	        TROCADERO
!
!     Subroutine:       update_cell()
!
!     Variables:
!
!     Created:          12/02/01
!
!     Purpose:          This subroutine calculates new cell vectors, which are
!			compatible with the updated metric tensor componets
!			that come out of the Souza-Martins dynamics
!
!     subroutines used:
!
!**************************************************************************
!  used modules

   use lattice

!*****************************************************************************
!  No implicits please!
   
   implicit none

!*****************************************************************************
!  shared variables

   logical, intent (IN) :: debug

   type (latt), intent (INOUT) :: cell

   real(q), intent (IN) :: barostat_mass

   real(q), intent (IN) ::POTIM

   real(q), intent (INOUT), dimension (4) :: SNOSE

!*****************************************************************************
!  local variables

   real(q), dimension(3,3) :: matrix_1

   real(q), dimension(3,3) :: matrix_2

   real(q), dimension(3,3) :: matrix_3

   


   real(q) :: alpha, beta, gamma

!*****************************************************************************
!  start of subroutine

   if ( debug ) write(*,*) 'Entering update_cell()'

   matrix_1(1,1) = sqrt( metric_tensor(1,1) )
   matrix_1(2,1) = zero
   matrix_1(3,1) = zero

   matrix_1(1,2) = metric_tensor(1,2) / sqrt( metric_tensor(1,1) )
   matrix_1(2,2) = sqrt( metric_tensor(2,2) -                              &
             metric_tensor(1,2) * metric_tensor(1,2) / metric_tensor(1,1) )
   matrix_1(3,2) = zero

   matrix_1(1,3) = metric_tensor(1,3) / sqrt( metric_tensor(1,1) )
   matrix_1(2,3) = ( metric_tensor(1,1) * metric_tensor(2,3) -             &
                metric_tensor(1,2) * metric_tensor(1,3) ) /                &
      sqrt( metric_tensor(1,1) * metric_tensor(1,1) * metric_tensor(2,2) - &
            metric_tensor(1,1) * metric_tensor(1,2) * metric_tensor(1,2) )
   matrix_1(3,3) = sqrt( metric_tensor(3,3) -                              &
           matrix_1(1,3) * matrix_1(1,3) - matrix_1(2,3) * matrix_1(2,3) )

! rotate the cell to its original orientation

   call dgemm('N','N', 3, 3, 3, one, rotation_matrix, 3, matrix_1, 3, zero, &
              cell % a, 3 )

! once we have the new cell, construct its reciprocal cell

   call lattic( cell )

! we can then construct the metric tensor of the reciprocal cell

   call dgemm( 'T', 'N', 3, 3, 3, one, cell % b, 3, cell % b, 3, zero,   &
               reciprocal_metric_tensor, 3 )

 !  gamma = acos( dot_product( cell % a(1:3,1), cell % a(1:3,2) ) / ( cell % anorm(1) * cell % anorm(2) ) )
 !  beta = acos( dot_product( cell % a(1:3,1), cell % a(1:3,3) ) / ( cell % anorm(1) * cell % anorm(3) ) )
 !  alpha = acos( dot_product( cell % a(1:3,3), cell % a(1:3,2) ) / ( cell % anorm(3) * cell % anorm(2) ) )

!   print*, 'angles ', alpha, beta, gamma

!tb beg
  !c compute gdot:
  matrix_3=metric_tensor_momenta *SNOSE(3)/(barostat_mass * cell % omega * cell % omega)
  matrix_3=MATMUL(metric_tensor,matrix_3)
  matrix_3=MATMUL(matrix_3,metric_tensor)
  !matrix_3=MATMUL(reciprocal_metric_tensor,matrix_3)
  !matrix_3=MATMUL(matrix_3,reciprocal_metric_tensor)

  !c triangular matrix of lattice velocities
   matrix_2=zero
   cell%avel=zero

   matrix_2(1,1) = matrix_3(1,1)/matrix_1(1,1)/2
   matrix_2(2,1) = zero
   matrix_2(3,1) = zero
   matrix_2(1,2) = (matrix_3(1,2) - matrix_2(1,1)*matrix_1(1,2))/matrix_1(1,1)
   matrix_2(2,2) = (matrix_3(2,2)/2 - matrix_1(1,2)*matrix_2(1,2))/matrix_1(2,2)
   matrix_2(3,2) = zero
   matrix_2(1,3) = (matrix_3(1,3) - matrix_2(1,1)*matrix_1(1,3))/matrix_1(1,1)   
   matrix_2(2,3)= (matrix_3(2,3)-matrix_1(1,2)*matrix_2(1,3)-matrix_1(1,3)*matrix_2(1,2)-&
   &              matrix_1(2,3)*matrix_2(2,2))/matrix_1(2,2)
   matrix_2(3,3) = (matrix_3(3,3)/2 - matrix_1(1,3)*matrix_2(1,3)-&
   &               matrix_1(2,3)*matrix_2(2,3))/matrix_1(3,3)
 
  !c get the correct orientation of lattice velocity vectors 
   call dgemm('N','N', 3, 3, 3, one, rotation_matrix, 3, matrix_2, 3, zero, &
            cell % avel, 3 )


!    matrix_2=zero
!    cell%avel=zero
! 
!    matrix_2(1,1) = metric_tensor_momenta(1,1)/matrix_1(1,1)/2
!    matrix_2(2,1) = zero
!    matrix_2(3,1) = zero
!    matrix_2(1,2) = (metric_tensor_momenta(1,2) - matrix_2(1,1)*matrix_1(1,2))/matrix_1(1,1)
!    matrix_2(2,2) = (metric_tensor_momenta(2,2)/2 - matrix_1(1,2)*matrix_2(1,2))/matrix_1(2,2)
!    matrix_2(3,2) = zero
!    matrix_2(1,3) = (metric_tensor_momenta(1,3) - matrix_2(1,1)*matrix_1(1,3))/matrix_1(1,1)   
!    matrix_2(2,3)= (metric_tensor_momenta(2,3)-matrix_1(1,2)*matrix_2(1,3)-matrix_1(1,3)*matrix_2(1,2)-&
!    &              matrix_1(2,3)*matrix_2(2,2))/matrix_1(2,2)
!    matrix_2(3,3) = (metric_tensor_momenta(3,3)/2 - matrix_1(1,3)*matrix_2(1,3)-&
!    &               matrix_1(2,3)*matrix_2(2,3))/matrix_1(3,3)
!  
!    call dgemm('N','N', 3, 3, 3, one, rotation_matrix, 3, matrix_2, 3, zero, &
!             cell % avel, 3 )
!    cell % avel=cell % avel/(barostat_mass*cell%omega*cell%omega)

   !wcs: cell % avel=metric_tensor_momenta
   !cell % avel=cell % avel/sqrt(barostat_mass)
   !!cell % avel=cell % avel/POTIM

    
!tb end



   if ( debug ) write(*,*) 'Exiting update_cell()'

end subroutine update_cell

!*****************************************************************************
subroutine update_metric_tensor_components( debug, ISIF, barostat_mass, S,   &
                                            S_new, cell )
!*****************************************************************************
!
!     Program: 	        TROCADERO
!
!     Subroutine:       update_metric_tensor_components()
!
!     Variables:
!
!     Created:          06/02/01
!
!     Purpose:          This subroutine updates the metric 
!			tensor components to full-step. The equations that 
!			appear in the Souza-Martins algorithm for constant
!			pressure molecular dynamics can be integrated using
!			the Generalised Leap-Frog (GLF) integration method.
!			When this is done, the equations that appear for 
!			the numerical integration of the components of the 
!			metric tensor are implicit, and therefore
!			we must use a Newton-Raphson procedure to solve them.
!			This is what this subroutine does.
!
!     subroutines used:
!
!**************************************************************************
!  used modules

   use lattice

!*****************************************************************************
!  No implicits please!
   
   implicit none

!*****************************************************************************
!  shared variables

   logical, intent (IN) :: debug

   integer, intent (IN) :: ISIF

   real(q), intent (IN) :: barostat_mass
   real(q), intent (IN) :: S
   real(q), intent (IN) :: S_new

   type (latt), intent (IN) :: cell

!*****************************************************************************
!  local variables

   logical :: converged

   integer :: i, ifail, n_newton_raphson_steps

   integer, parameter :: n_newton_raphson_steps_max = 100

   integer :: iwork(9)

   real(q) :: det_metric, det_metric_tmp, factor_1, factor_2
   real(q) :: cofactor_11, cofactor_12, cofactor_13
   real(q) :: cofactor_22, cofactor_23, cofactor_33
   real(q) :: a_modulus, b_modulus, c_modulus
   real(q) :: trace

   real(q), parameter :: tolerance = 1.0d-7

   real(q) :: derivatives(9,9)
   real(q) :: difference(9)
   real(q) :: gpi(3,3)
   real(q) :: pig(3,3)
   real(q) :: gpig(3,3)
   real(q) :: gpig_old(3,3)
   real(q) :: metric_tensor_new(3,3)
   real(q) :: metric_tensor_tmp(3,3)
   real(q) :: recipr_metric_tensor_tmp(3,3)
   real(q) :: step(9)
   real(q) :: work(9)

!*****************************************************************************
!  start of subroutine

   if ( debug ) write(*,*) 'Entering update_metric_tensor_components()'

! the first thing is to generate the half-step velocities of the metric 
! in real space, as up to now we have been using them in reciprocal space

   call dgemm( 'N', 'N', 3, 3, 3, one, metric_tensor, 3,                     &
               metric_tensor_momenta, 3, zero, gpi, 3 )
   call dgemm( 'N', 'N', 3, 3, 3, one, gpi, 3,                               &
               metric_tensor, 3, zero, gpig, 3 )

! initialise the tmp momenta making them equal to the old

   metric_tensor_tmp = metric_tensor
   gpig_old = gpig

   det_metric = cell % omega * cell % omega

   n_newton_raphson_steps = 0

   do 

! first calculate the reciprocal metric tensor, the determinant and the
! gpig, gpi and pig matrices based on the trial metric tensor

      cofactor_11 = metric_tensor_tmp(2,2) * metric_tensor_tmp(3,3) -        &
                    metric_tensor_tmp(2,3) * metric_tensor_tmp(2,3)
      cofactor_12 = metric_tensor_tmp(1,3) * metric_tensor_tmp(2,3) -        &
                    metric_tensor_tmp(1,2) * metric_tensor_tmp(3,3)
      cofactor_13 = metric_tensor_tmp(1,2) * metric_tensor_tmp(2,3) -        &
                    metric_tensor_tmp(1,3) * metric_tensor_tmp(2,2)

      cofactor_22 = metric_tensor_tmp(1,1) * metric_tensor_tmp(3,3) -        &
                    metric_tensor_tmp(1,3) * metric_tensor_tmp(1,3)
      cofactor_23 = metric_tensor_tmp(1,2) * metric_tensor_tmp(1,3) -        &
                    metric_tensor_tmp(1,1) * metric_tensor_tmp(2,3)

      cofactor_33 = metric_tensor_tmp(1,1) * metric_tensor_tmp(2,2) -        &
                    metric_tensor_tmp(1,2) * metric_tensor_tmp(1,2)

      det_metric_tmp = metric_tensor_tmp(1,1) * cofactor_11 +                &
                       metric_tensor_tmp(1,2) * cofactor_12 +                &
                       metric_tensor_tmp(1,3) * cofactor_13

      recipr_metric_tensor_tmp(1,1) = cofactor_11 / det_metric_tmp
      recipr_metric_tensor_tmp(1,2) = cofactor_12 / det_metric_tmp
      recipr_metric_tensor_tmp(1,3) = cofactor_13 / det_metric_tmp

      recipr_metric_tensor_tmp(2,1) = cofactor_12 / det_metric_tmp
      recipr_metric_tensor_tmp(2,2) = cofactor_22 / det_metric_tmp
      recipr_metric_tensor_tmp(2,3) = cofactor_23 / det_metric_tmp
 
      recipr_metric_tensor_tmp(3,1) = cofactor_13 / det_metric_tmp
      recipr_metric_tensor_tmp(3,2) = cofactor_23 / det_metric_tmp
      recipr_metric_tensor_tmp(3,3) = cofactor_33 / det_metric_tmp

      call dgemm( 'N', 'N', 3, 3, 3, one, metric_tensor_tmp, 3,              &
                  metric_tensor_momenta, 3, zero, gpi, 3 )

      call dgemm( 'N', 'N', 3, 3, 3, one, gpi, 3,                            &
                  metric_tensor_tmp, 3, zero, gpig, 3 )
      pig = transpose( gpi )

! we then predict the tmp metric_tensor

      factor_1 = half * delta_t * S / ( det_metric * barostat_mass )
      factor_2 = half * delta_t * S_new / ( det_metric_tmp * barostat_mass )

      metric_tensor_new = metric_tensor + factor_1 * gpig_old + factor_2 * gpig

! lets calculate the differences between predicted and actual momenta
! remember: metric tensor and its momentum componets are symmetric

      difference(1) = metric_tensor_new(1,1) -                               &
                      metric_tensor_tmp(1,1)
      difference(2) = metric_tensor_new(1,2) -                               &
                      metric_tensor_tmp(1,2)
      difference(3) = metric_tensor_new(1,3) -                               &
                      metric_tensor_tmp(1,3)
      difference(4) = metric_tensor_new(2,1) -                               &
                      metric_tensor_tmp(2,1)
      difference(5) = metric_tensor_new(2,2) -                               &
                      metric_tensor_tmp(2,2)
      difference(6) = metric_tensor_new(2,3) -                               &
                      metric_tensor_tmp(2,3)
      difference(7) = metric_tensor_new(3,1) -                               &
                      metric_tensor_tmp(3,1)
      difference(8) = metric_tensor_new(3,2) -                               &
                      metric_tensor_tmp(3,2)
      difference(9) = metric_tensor_new(3,3) -                               &
                      metric_tensor_tmp(3,3)

      converged = .true.

      do i=1, 9

         if ( dabs( difference(i) ) .gt. tolerance ) converged = .false.

      end do
 
      if ( .not. converged ) then

!        metric_tensor_tmp = metric_tensor_new

         metric_tensor_tmp = half * ( metric_tensor_new +                  &
                           transpose( metric_tensor_new ) )
         
      else

         exit 

      end if

      n_newton_raphson_steps = n_newton_raphson_steps + 1

      if ( n_newton_raphson_steps > n_newton_raphson_steps_max ) then

         write(*,*) 'n_newton_raphson_max exceeded A: stopping'
         stop

      end if

   end do

! if we have fixed-cell shape constraints, impose them here

   if ( ISIF >= 7 ) then

      if ( ISIF == 8 ) then

         trace = ( metric_tensor_new(1,1) +                               &
                   metric_tensor_new(2,2) +                               &
                   metric_tensor_new(3,3) ) / three

         metric_tensor_new(1,1) = trace
         metric_tensor_new(2,2) = trace
         metric_tensor_new(3,3) = trace

      else if ( ISIF == 9 ) then

         trace = ( metric_tensor_new(1,1) +                               &
                   metric_tensor_new(2,2) ) / two

         metric_tensor_new(1,1) = trace
         metric_tensor_new(2,2) = trace

      end if

      a_modulus = sqrt( metric_tensor_new(1,1) )
      b_modulus = sqrt( metric_tensor_new(2,2) )
      c_modulus = sqrt( metric_tensor_new(3,3) )

      metric_tensor_new(1,2) = a_modulus * b_modulus * cos_gamma
      metric_tensor_new(1,3) = a_modulus * c_modulus * cos_beta
      metric_tensor_new(2,3) = b_modulus * c_modulus * cos_alpha

      metric_tensor_new(2,1) = metric_tensor_new(1,2)
      metric_tensor_new(3,1) = metric_tensor_new(1,3)
      metric_tensor_new(3,2) = metric_tensor_new(2,3)

      metric_tensor_tmp = metric_tensor_new

   end if

   metric_tensor = metric_tensor_tmp

   if ( debug ) write(*,*) 'Exiting update_metric_tensor_components()'

end subroutine update_metric_tensor_components

!*****************************************************************************
subroutine update_metric_tensor_momenta( debug, barostat_mass, cell,         &
                          internal_stress, SNOSE, ISIF )
!*****************************************************************************
!
!     Program: 	        TROCADERO
!
!     Subroutine:       update_metric_tensor_momenta()
!
!     Variables:
!
!     Created:          02/02/01
!
!     Purpose:          This subroutine updates the momenta of the metric 
!			tensor components to half-step. The equations that 
!			appear in the Souza-Martins algorithm for constant
!			pressure molecular dynamics can be integrated using
!			the Generalised Leap-Frog (GLF) integration method.
!			When this is done, the equations that appear for 
!			the numerical integration of the momenta of the 
!			metric tensor components are implicit, and therefore
!			we must use a Newton-Raphson procedure to solve them.
!			This is what this subroutine does.
!
!			IMPORTANT: this subroutine updates the components
!			           of the momenta for the reciprocal metric
!				   tensor
!
!     subroutines used:
!
!**************************************************************************
!  used modules

   use lattice
   !use stress_module

!*****************************************************************************
!  No implicits please!
   
   implicit none

!*****************************************************************************
!  shared variables

   logical, intent (IN) :: debug

   integer, intent (IN) :: ISIF

   real(q), intent (IN) :: barostat_mass

   real(q), intent (IN), dimension (4) :: SNOSE

   type (latt), intent (IN) :: cell
   type (stress_type), intent (IN) :: internal_stress

!*****************************************************************************
!  local variables

   logical :: converged

   integer :: i, ifail, n_newton_raphson_steps

   integer, parameter :: n_newton_raphson_steps_max = 100

   integer :: iwork(9)

   real(q) :: ddot, det_metric, factor

   real(q), parameter :: tolerance = 1.0d-12

   real(q) :: difference(9)
   real(q) :: gpi(3,3)
   real(q) :: pig(3,3)
   real(q) :: gpig(3,3)
   real(q) :: matrix_1(3,3)
   real(q) :: matrix_2(3,3)
   real(q) :: matrix_3(3,3)
   real(q) :: metric_tensor_momenta_new(3,3)
   real(q) :: metric_tensor_momenta_tmp(3,3)
   real(q) :: step(9)
   real(q) :: work(9)

!*****************************************************************************
!  start of subroutine

   if ( debug ) write(*,*) 'Entering update_metric_tensor_momenta()'

   det_metric = cell % omega * cell % omega

! initialise the tmp momenta making them equal to the old

   metric_tensor_momenta_tmp = metric_tensor_momenta

   n_newton_raphson_steps = 0

   do 

! we first predict the tmp momenta

      factor = barostat_mass * det_metric
      factor = one / factor 

      call dgemm( 'N', 'N', 3, 3, 3, factor, metric_tensor_momenta_tmp, 3,   &
                  metric_tensor, 3, zero, matrix_1, 3 )

      call dgemm( 'N', 'N', 3, 3, 3, one, matrix_1, 3,                       &
                  metric_tensor_momenta_tmp, 3, zero, matrix_2, 3 )

      matrix_3 = transpose( matrix_1 )

      factor = half * barostat_mass * det_metric

      E_kin_metric = factor * ddot( 9, matrix_1, 1, matrix_3, 1 )
 

!     write(6,*) 'stress % lattice ', internal_stress % lattice(1,1)
!     write(6,*) 'stress % cartesian ', internal_stress % cartesian(1,1)
!     write(6,*) 'kinetic_stress ', kinetic_stress(1,1)
!     write(6,*) 'matrix_2 ', matrix_2(1,1)
!     write(6,*) 'actual_pressure ', pressure
!     write(6,*) 'volume ', cell % omega
!     write(6,*) 'E_kin_metric ', E_kin_metric
!     write(6,*) 'reciprocal metric tensor ', reciprocal_metric_tensor(1,1)
!     write(6,*) 'external stresslatice ', stress % lattice(1,1)

      metric_tensor_momenta_new = metric_tensor_momenta -                    &
                 half * delta_t * SNOSE(1) * ( internal_stress % lattice -   &
                         kinetic_stress + matrix_2                           &
     + ( half * pressure * cell % omega - E_kin_metric ) *                   &
         reciprocal_metric_tensor + half * stress % lattice )

      if ( ISIF >= 7 ) then

!         call impose_metric_momenta_constraints(ISIF,barostat_mass,SNOSE,cell)

      end if


      metric_tensor_momenta_new = half * ( metric_tensor_momenta_new +       &
                                transpose( metric_tensor_momenta_new ) )

!     write(6,*) 'metric_momenta = ', metric_tensor_momenta_new(1,1)

! lets calculate the differences between predicted and actual momenta
! remember: metric tensor and its momentum componets are symmetric

      difference(1) = metric_tensor_momenta_new(1,1) -                       &
                      metric_tensor_momenta_tmp(1,1)
      difference(2) = metric_tensor_momenta_new(1,2) -                       &
                      metric_tensor_momenta_tmp(1,2)
      difference(3) = metric_tensor_momenta_new(1,3) -                       &
                      metric_tensor_momenta_tmp(1,3)
      difference(4) = metric_tensor_momenta_new(2,1) -                       &
                      metric_tensor_momenta_tmp(2,1)
      difference(5) = metric_tensor_momenta_new(2,2) -                       &
                      metric_tensor_momenta_tmp(2,2)
      difference(6) = metric_tensor_momenta_new(2,3) -                       &
                      metric_tensor_momenta_tmp(2,3)
      difference(7) = metric_tensor_momenta_new(3,1) -                       &
                      metric_tensor_momenta_tmp(3,1)
      difference(8) = metric_tensor_momenta_new(3,2) -                       &
                      metric_tensor_momenta_tmp(3,2)
      difference(9) = metric_tensor_momenta_new(3,3) -                       &
                      metric_tensor_momenta_tmp(3,3)

! the correspondence of indices is: 1,1-> 1, 1,2-> 2, 1,3-> 3, 
!                                   2,1-> 4, 2,2-> 5, 2,3-> 6,
!                                   3,1-> 7, 3,2-> 8, 3,3-> 6

      converged = .true.

      do i=1, 9

         if ( dabs( difference(i) ) .gt. tolerance ) converged = .false.

      end do

      if ( .not. converged ) then

         metric_tensor_momenta_tmp = metric_tensor_momenta_new

         metric_tensor_momenta_tmp = half * (                               &
                                        metric_tensor_momenta_tmp +         &
                             transpose( metric_tensor_momenta_tmp ) )
         
      else

         exit 

      end if

      n_newton_raphson_steps = n_newton_raphson_steps + 1

!     write(6,*) 'n_newton = ', n_newton_raphson_steps

      if ( n_newton_raphson_steps > n_newton_raphson_steps_max ) then

         write(6,1) metric_tensor_momenta_new(1,1:3)
         write(6,1) metric_tensor_momenta_new(2,1:3)
         write(6,1) metric_tensor_momenta_new(3,1:3)

         write(*,*) 'n_newton_raphson_max exceeded B: stopping'
         stop

      end if

   end do

   metric_tensor_momenta = metric_tensor_momenta_tmp

!  stop

 1   format(3f15.7)

   if ( debug ) write(*,*) 'Exiting update_metric_tensor_momenta()'

end subroutine update_metric_tensor_momenta

end module npt_dynamics
