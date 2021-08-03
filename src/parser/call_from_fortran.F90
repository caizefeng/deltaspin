SUBROUTINE OUTPUT_BASIS(basis_info) 

IMPLICIT NONE

TYPE BASIS_FUNC_INFO
   SEQUENCE

   ! this corresponds to the C++ type t_basis
   ! Sequence keyword guarantees that FORTRAN will not mess with order of variables

!  radial part
   INTEGER radial_source
   INTEGER index
   INTEGER shell

! spherical part
   INTEGER l
   INTEGER m

! site part
   INTEGER poscar

   REAL*8 posx
   REAL*8 posy
   REAL*8 posz

   REAL*8 sigma

END TYPE BASIS_FUNC_INFO

type (BASIS_FUNC_INFO) basis_info 

IF (basis_info%poscar .eq. -1) THEN
   WRITE(0,'(A,F5.2,F5.2,F5.2)',ADVANCE='NO') "A site on position ", basis_info%posx, basis_info%posy, basis_info%posz
ELSE
   WRITE(0,'(A,I5,A)',ADVANCE='NO') "A site on index ", basis_info%poscar, " of the POSCAR "
ENDIF

WRITE(0,'(A,I2,I2)',ADVANCE='NO') "Y ",basis_info%l,basis_info%m

IF (basis_info%radial_source .eq. 1) THEN
   IF (basis_info%shell .ne. -1) THEN
      WRITE(0,'(A,I3)') " Using the PAW function of shell", basis_info%shell
   ELSE
      WRITE(0,'(A)') " Using the PAW function of the outermost shell"
   ENDIF
ELSEIF (basis_info%radial_source .eq. 2) THEN
   IF (basis_info%shell .ne. -1) THEN
      WRITE(0,'(A,I3,A,I3)') " Using the POTCAR potential number", basis_info%index,"of shell", basis_info%shell
   ELSE
      WRITE(0,'(A,I3,A)') " Using the POTCAR potential number ",basis_info%index,"of the outermost shell"
   ENDIF
ELSE
   WRITE(0,'(A,I3,A,F5.2)') " Using the WANNIER function fallback, with shell ", basis_info%shell, " and a sigma of ", basis_info%sigma
ENDIF

END SUBROUTINE OUTPUT_BASIS

PROGRAM parse_basis_function_info

IMPLICIT NONE

TYPE BASIS_FUNC_INFO
   SEQUENCE

   ! this corresponds to the C++ type t_basis
   ! Sequence keyword guarantees that FORTRAN will not mess with order of variables

!  radial part
   INTEGER radial_source
   INTEGER index
   INTEGER shell

! spherical part
   INTEGER l
   INTEGER m

! site part
   INTEGER poscar

   REAL*8 posx
   REAL*8 posy
   REAL*8 posz

   REAL*8 sigma

END TYPE BASIS_FUNC_INFO

INTEGER n_basis

INTEGER nfilename

TYPE (BASIS_FUNC_INFO), allocatable:: basis_functions(:)

CHARACTER*100 filename

INTEGER i

   filename='in.2nd'
   nfilename = 6 ! Length of filename argument

   CALL PARSE_file(n_basis, nfilename, filename)

   write(0,*) n_basis, " n_basis"

   ALLOCATE(basis_functions(n_basis))

   do i=1,n_basis
      CALL fill_basis_info(basis_functions(i),i)   
   enddo

   do i=1,n_basis
      CALL OUTPUT_BASIS(basis_functions(i)) 
   enddo

   CALL free_parser()

   DEALLOCATE(basis_functions)

 END PROGRAM parse_basis_function_info
