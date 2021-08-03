#include "symbol.inc"

! Auxiliary functions for debugging
! Subroutines type overloaded 
! No special facilities for parallel runs!
module util
  use prec
  
  interface mat_dump
     module procedure mat_dump_cq, mat_dump_rq,mat_dump_cqs, mat_dump_rqs
  end interface mat_dump

  interface mat_diag
     module procedure mat_diag_cq
  end interface mat_diag

  interface is_hermitian
     module procedure is_hermitian_cq, is_hermitian_rq
  end interface is_hermitian

  interface is_symmetric
     module procedure is_hermitian_rq
  end interface is_symmetric

contains

  subroutine mat_dump_cq(m,txt,nc,nr)
    implicit none
    complex(q) :: m(:,:)
    character(len=*), optional :: txt
    integer, optional :: nc,nr
    integer :: i1,n1,n2

    n1=size(m,1)
    n2=size(m,2)
    if(present(nc))then
       n2=nc
       n1=nc
    end if
    if(present(nr))then
       n1=nr
    end if
    
    write(*,*) "------------------------------------------------------------------"
    if(present(txt))then
       write(*,'(a)')  txt
    end if
    do i1=1,n1
       write(*,'(100e12.3)') real(m(i1,1:n2),q)
    end do
    write(*,*)
    do i1=1,n1
       write(*,'(100e12.3)') aimag(m(i1,1:n2))
    end do

  end subroutine mat_dump_cq

  subroutine mat_dump_cqs(m,txt,nc,nr)
    implicit none
    complex(qs) :: m(:,:)
    character(len=*), optional :: txt
    integer, optional :: nc,nr
    integer :: i1,n1,n2

    n1=size(m,1)
    n2=size(m,2)
    if(present(nc))then
       n2=nc
       n1=nc
    end if
    if(present(nr))then
       n1=nr
    end if
    
    write(*,*) "------------------------------------------------------------------"
    if(present(txt))then
       write(*,'(a)')  txt
    end if
    do i1=1,n1
       write(*,'(100e12.3)') real(m(i1,1:n2),qs)
    end do
    write(*,*)
    do i1=1,n1
       write(*,'(100e12.3)') aimag(m(i1,1:n2))
    end do

  end subroutine mat_dump_cqs


  subroutine mat_dump_rq(m,txt,nc,nr)
    implicit none
    real(q) :: m(:,:)
    character(len=*), optional :: txt
    integer, optional :: nc,nr
    integer :: i1,n1,n2

    n1=size(m,1)
    n2=size(m,2)
    if(present(nc))then
       n2=nc
       n1=nc
    end if
    if(present(nr))then
       n1=nr
    end if
    
    write(*,*) "------------------------------------------------------------------"
    if(present(txt))then
       write(*,'(a)')  txt
    end if
    do i1=1,n1
       write(*,'(100e12.3)') m(i1,1:n2)
    end do

  end subroutine mat_dump_rq

  subroutine mat_dump_rqs(m,txt,nc,nr)
    implicit none
    real(qs) :: m(:,:)
    character(len=*), optional :: txt
    integer, optional :: nc,nr
    integer :: i1,n1,n2

    n1=size(m,1)
    n2=size(m,2)
    if(present(nc))then
       n2=nc
       n1=nc
    end if
    if(present(nr))then
       n1=nr
    end if
    
    write(*,*) "------------------------------------------------------------------"
    if(present(txt))then
       write(*,'(a)')  txt
    end if
    do i1=1,n1
       write(*,'(100e12.3)') m(i1,1:n2)
    end do

  end subroutine mat_dump_rqs

  function is_hermitian_cq(m,tol)
    implicit none
    logical :: is_hermitian_cq
    complex(q) :: m(:,:)
    integer :: n1,n2,i,j
    real(q), optional :: tol
    real(q) :: tol0=1e-10_q
    n1=size(m,1)
    n2=size(m,2)
    if(n1/=n2)then
       write(*,*) 'IS_HERMITIAN: matrix not quadratic'
       stop
    end if
    if(present(tol)) tol0=tol
    is_hermitian_cq=.true.
    do i=1,n1
       do j=1,n2
          if( abs(m(i,j)-conjg(m(j,i)))>tol0 )then
             is_hermitian_cq=.false.
             return
          end if
       end do
    end do
  end function is_hermitian_cq


  function is_hermitian_rq(m,tol)
    implicit none
    logical :: is_hermitian_rq
    real(q) :: m(:,:)
    integer :: n1,n2,i,j
    real(q), optional :: tol
    real(q) :: tol0=1e-8_q
    n1=size(m,1)
    n2=size(m,2)
    if(n1/=n2)then
       write(*,*) 'IS_HERMITIAN: matrix not quadratic'
       stop
    end if
    if(present(tol)) tol0=tol
    is_hermitian_rq=.true.
    do i=1,n1
       do j=1,n2
          if( abs(m(i,j)-m(j,i))>tol0 )then
             is_hermitian_rq=.false.
             return
          end if
       end do
    end do
  end function is_hermitian_rq


  subroutine mat_diag_cq(m,typ,ev,dump)
    implicit none
    complex(q), intent(in) :: m(:,:)
    character(len=1) :: typ
    real(q), intent(inout), optional :: ev(:)
    logical, optional :: dump
    !
    real(q),allocatable :: w(:),rwork(:)
    complex(q),allocatable :: mcopy(:,:),work(:)
    integer :: n,lwork,info,j
    logical :: ldump
    !
    n=size(m,2)
    lwork=max(1,2*n)
    allocate(w(n),rwork(3*n),work(lwork))
    allocate(mcopy(size(m,1),size(m,2)))
    mcopy=m
    ldump=.false.
    if(present(dump))ldump=dump

    select case (typ)
    case ('h','H')
       ! hermitian
       call zheev('N','U',n,mcopy,size(mcopy,1),w,work,lwork,rwork,info)
       if(info/=0)then
          write(*,*) 'MAT_DIAG: '
          stop
       endif
       if(ldump)then
          write(*,*) 'MAT_DIAG: Eigenvalues'
          do j=1,n
             write(*,*) w(j)
          end do
       end if
    case default
       ! general complex
       ! TODO
       write(*,*) 'MAT_DIAG: general complex matrix not implemented!'
       stop
    end select
    if(present(ev)) ev=w
    deallocate(w,rwork,work,mcopy)
  end subroutine mat_diag_cq
  
end module util


module interpolate
  use prec
  implicit none

  interface inter_1d_linear
     module procedure inter_1d_linear_rq, inter_1d_linear_cq, inter_1d_linear_rq_v, inter_1d_linear_cq_v
  end interface inter_1d_linear

  interface inter_1d_cubic
     module procedure inter_1d_cubic_rq, inter_1d_cubic_cq, inter_1d_cubic_rq_v, inter_1d_cubic_cq_v
  end interface inter_1d_cubic

  interface inter_2d_cubic
     module procedure inter_2d_cubic_rq
  end interface inter_2d_cubic

  interface inter_3d_cubic
     module procedure inter_3d_cubic_rq, inter_3d_cubic_cq
  end interface inter_3d_cubic


contains

  !=================================================
  ! LINEAR INTERPOLATION of a function at point r in [0,1]
  ! given function values
  ! f = a*r + b
  !
  ! f0=f(0)
  ! f1=f(1)
  !-------------------------------------------------
  subroutine inter_1d_linear_rq(r,f0,f1,f)
    implicit none
    real(q), intent(in) :: r,f0,f1
    real(q), intent(out) :: f
    !
    f=f0+r*(f1-f0)
  end subroutine inter_1d_linear_rq

  subroutine inter_1d_linear_cq(r,f0,f1,f)
    implicit none
    real(q), intent(in) :: r
    complex(q), intent(in) :: f0,f1
    complex(q), intent(out) :: f
    !
    f=f0+r*(f1-f0)
  end subroutine inter_1d_linear_cq
  !-------------------------------------------------
  ! vector versions
  ! Usage:
  ! 1. calculate weights w(2)
  ! 2. interpolate 
  ! (Advantage: weights can be used multiple times)
  !-------------------------------------------------
  subroutine inter_1d_linear_weights(r,w)
    implicit none
    real(q), intent(in) :: r
    real(q), intent(out) :: w(2)
    !
    ! weights for function interpolation
    w(1)=1-r     ! f(0)
    w(2)=r       ! f(1)
  end subroutine inter_1d_linear_weights

  subroutine inter_1d_linear_rq_v(w,f0,f1,f)
    implicit none
    real(q), intent(in) :: w(2),f0(:),f1(:)
    real(q), intent(out) :: f(:)
    !
    f=w(1)*f0+w(2)*f1
  end subroutine inter_1d_linear_rq_v

  subroutine inter_1d_linear_cq_v(w,f0,f1,f)
    implicit none
    real(q), intent(in) :: w(2)
    complex(q), intent(in) :: f0(:),f1(:)
    complex(q), intent(out) :: f(:)
    !
    f=w(1)*f0+w(2)*f1
  end subroutine inter_1d_linear_cq_v
  !=================================================
  

  !=================================================
  ! QUADRATIC INTERPOLATION
  ! NOT IMPLEMENTED
  !=================================================


  !=================================================
  ! CUBIC INTERPOLATION of a function at point r in [0,1]
  ! given function and derivative values
  ! f = a*r^3 + b*r^2 + c*r + d
  ! g = 3*a*r^2 + 2*b*r + c
  !
  ! f0=f(0)
  ! f1=f(1)
  ! g0=df/dr(0)=a*df/dx(x0)
  ! g1=df/dr(1)=a*df/dx(x0+a)
  ! if x=x0+a*r (i.e. scaled coordinate with scaling factor a)
  ! then df/dr = df/dx*dx/dr = a*df/dx
  !-------------------------------------------------
  subroutine inter_1d_cubic_rq(r,f0,f1,g0,g1,f,g)
    implicit none
    real(q), intent(in) :: r,f0,f1,g0,g1
    real(q), intent(out) :: f
    real(q), intent(out), optional :: g
    real(q) a,b,c,d
    !
    a=2*f0-2*f1+g0+g1
    b=-2*g0-g1+3*f1-3*f0
    c=g0
    d=f0
    !
    f=d + c*r + b*r**2 + a*r**3
    if(present(g)) g=c + 2*b*r + 3*a*r**2
  end subroutine inter_1d_cubic_rq

  subroutine inter_1d_cubic_cq(r,f0,f1,g0,g1,f,g)
    implicit none
    real(q), intent(in) :: r
    complex(q), intent(in) :: f0,f1,g0,g1
    complex(q), intent(out) :: f
    complex(q), intent(out), optional :: g
    complex(q) a,b,c,d
    !
    a=2*f0-2*f1+g0+g1
    b=-2*g0-g1+3*f1-3*f0
    c=g0
    d=f0
    !
    f=d + c*r + b*r**2 + a*r**3
    if(present(g)) g=c + 2*b*r + 3*a*r**2
  end subroutine inter_1d_cubic_cq
  !-------------------------------------------------
  ! vector versions
  !-------------------------------------------------
  subroutine inter_1d_cubic_weights(r,w,v)
    implicit none
    real(q), intent(in) :: r
    real(q), intent(out) :: w(4)
    real(q), intent(out), optional :: v(4)
    !
    ! weights for function interpolation
    w(1)=1-3*r**2+2*r**3      ! f(0)
    w(2)=3*r**2-2*r**3        ! f(1)
    w(3)=r-2*r**2+r**3        ! f'(0)
    w(4)=-r**2+r**3           ! f'(1)
    ! weights for derivative interpolation
    if(present(v))then
       v(1)=-6*r+6*r**2     ! f(0)
       v(2)=6*r-6*r**2      ! f(1)
       v(3)=1-4*r+3*r**2    ! f'(0)
       v(4)=-2*r+3*r**2     ! f'(1)
    end if
  end subroutine inter_1d_cubic_weights

  subroutine inter_1d_cubic_rq_v(w,f0,f1,g0,g1,f,v,g)
    implicit none
    real(q), intent(in) :: w(4),f0(:),f1(:),g0(:),g1(:)
    real(q), intent(out) :: f(:)
    real(q), intent(in), optional :: v(4)
    real(q), intent(out), optional :: g(:)
    real(q) a,b,c,d
    !
    f=w(1)*f0+w(2)*f1+w(3)*g0+w(4)*g1
    if(present(g)) g=v(1)*f0+v(2)*f1+v(3)*g0+v(4)*g1
  end subroutine inter_1d_cubic_rq_v

  subroutine inter_1d_cubic_cq_v(w,f0,f1,g0,g1,f,v,g)
    implicit none
    real(q), intent(in) :: w(4)
    complex(q), intent(in) :: f0(:),f1(:),g0(:),g1(:)
    complex(q), intent(out) :: f(:)
    real(q), intent(in), optional :: v(4)
    complex(q), intent(out), optional :: g(:)
    !
    f=w(1)*f0+w(2)*f1+w(3)*g0+w(4)*g1
    if(present(g)) g=v(1)*f0+v(2)*f1+v(3)*g0+v(4)*g1
  end subroutine inter_1d_cubic_cq_v
  !============================================


  !============================================  
  ! 2D INTERPOLATION
  !-------------------------------------------------
  subroutine inter_2d_cubic_rq(r,f,gx,gy,v)
    implicit none
    real(q), intent(in) :: r(2),f(2,2),gx(2,2),gy(2,2)
    real(q), intent(out) :: v
    real(q) :: f1(2),gy1(2)
    !
    ! interpolate in x direction
    call inter_1d_cubic(r(1),f(1,1),f(2,1),gx(1,1),gx(2,1),f1(1))
    call inter_1d_linear(r(1),gy(1,1),gy(2,1),gy1(1))
    call inter_1d_cubic(r(1),f(1,2),f(2,2),gx(1,2),gx(2,2),f1(2))
    call inter_1d_linear(r(1),gy(1,2),gy(2,2),gy1(2))
    !
    ! interpolate in y direction
    call inter_1d_cubic(r(2),f1(1),f1(2),gy1(1),gy1(2),v)
  end subroutine inter_2d_cubic_rq
  !============================================


  !============================================
  ! 3D INTERPOLATION
  !-------------------------------------------------
  subroutine inter_3d_cubic_rq(r,f,gx,gy,gz,v)
    implicit none
    real(q), intent(in) :: r(3),f(2,2,2),gx(2,2,2),gy(2,2,2),gz(2,2,2)
    real(q), intent(out) :: v
    real(q) :: f1(2,2),gy1(2,2),gz1(2,2),f2(2),gz2(2)
    !
    ! interpolate in x direction on 4 edges
    ! edge 1,1
    call inter_1d_cubic(r(1),f(1,1,1),f(2,1,1),gx(1,1,1),gx(2,1,1),f1(1,1))
    call inter_1d_linear(r(1),gy(1,1,1),gy(2,1,1),gy1(1,1))
    call inter_1d_linear(r(1),gz(1,1,1),gz(2,1,1),gz1(1,1))
    ! edge 1,2
    call inter_1d_cubic(r(1),f(1,1,2),f(2,1,2),gx(1,1,2),gx(2,1,2),f1(1,2))
    call inter_1d_linear(r(1),gy(1,1,2),gy(2,1,2),gy1(1,2))
    call inter_1d_linear(r(1),gz(1,1,2),gz(2,1,2),gz1(1,2))
    ! edge 2,1
    call inter_1d_cubic(r(1),f(1,2,1),f(2,2,1),gx(1,2,1),gx(2,2,1),f1(2,1))
    call inter_1d_linear(r(1),gy(1,2,1),gy(2,2,1),gy1(2,1))
    call inter_1d_linear(r(1),gz(1,2,1),gz(2,2,1),gz1(2,1))
    ! edge 2,2
    call inter_1d_cubic(r(1),f(1,2,2),f(2,2,2),gx(1,2,2),gx(2,2,2),f1(2,2))
    call inter_1d_linear(r(1),gy(1,2,2),gy(2,2,2),gy1(2,2))
    call inter_1d_linear(r(1),gz(1,2,2),gz(2,2,2),gz1(2,2))
    
    ! interpolate in y direction on 2 edges
    ! edge x,1
    call inter_1d_cubic(r(2),f1(1,1),f1(2,1),gy1(1,1),gy1(2,1),f2(1))
    call inter_1d_linear(r(2),gz1(1,1),gz1(2,1),gz2(1))    
    ! edge x,2
    call inter_1d_cubic(r(2),f1(1,2),f1(2,2),gy1(1,2),gy1(2,2),f2(2))
    call inter_1d_linear(r(2),gz1(1,2),gz1(2,2),gz2(2))    

    ! interpolate in z direction on 1 line
    call inter_1d_cubic(r(3),f2(1),f2(2),gz2(1),gz2(2),v)

  end subroutine inter_3d_cubic_rq

  subroutine inter_3d_cubic_cq(r,f,gx,gy,gz,v)
    implicit none
    real(q), intent(in) :: r(3)
    complex(q), intent(in) :: f(2,2,2),gx(2,2,2),gy(2,2,2),gz(2,2,2)
    complex(q), intent(out) :: v
    complex(q) :: f1(2,2),gy1(2,2),gz1(2,2),f2(2),gz2(2)
    !
    ! interpolate in x direction on 4 edges
    ! edge 1,1
    call inter_1d_cubic(r(1),f(1,1,1),f(2,1,1),gx(1,1,1),gx(2,1,1),f1(1,1))
    call inter_1d_linear(r(1),gy(1,1,1),gy(2,1,1),gy1(1,1))
    call inter_1d_linear(r(1),gz(1,1,1),gz(2,1,1),gz1(1,1))
    ! edge 1,2
    call inter_1d_cubic(r(1),f(1,1,2),f(2,1,2),gx(1,1,2),gx(2,1,2),f1(1,2))
    call inter_1d_linear(r(1),gy(1,1,2),gy(2,1,2),gy1(1,2))
    call inter_1d_linear(r(1),gz(1,1,2),gz(2,1,2),gz1(1,2))
    ! edge 2,1
    call inter_1d_cubic(r(1),f(1,2,1),f(2,2,1),gx(1,2,1),gx(2,2,1),f1(2,1))
    call inter_1d_linear(r(1),gy(1,2,1),gy(2,2,1),gy1(2,1))
    call inter_1d_linear(r(1),gz(1,2,1),gz(2,2,1),gz1(2,1))
    ! edge 2,2
    call inter_1d_cubic(r(1),f(1,2,2),f(2,2,2),gx(1,2,2),gx(2,2,2),f1(2,2))
    call inter_1d_linear(r(1),gy(1,2,2),gy(2,2,2),gy1(2,2))
    call inter_1d_linear(r(1),gz(1,2,2),gz(2,2,2),gz1(2,2))
    
    ! interpolate in y direction on 2 edges
    ! edge x,1
    call inter_1d_cubic(r(2),f1(1,1),f1(2,1),gy1(1,1),gy1(2,1),f2(1))
    call inter_1d_linear(r(2),gz1(1,1),gz1(2,1),gz2(1))    
    ! edge x,2
    call inter_1d_cubic(r(2),f1(1,2),f1(2,2),gy1(1,2),gy1(2,2),f2(2))
    call inter_1d_linear(r(2),gz1(1,2),gz1(2,2),gz2(2))    

    ! interpolate in z direction on 1 line
    call inter_1d_cubic(r(3),f2(1),f2(2),gz2(1),gz2(2),v)

  end subroutine inter_3d_cubic_cq
  !============================================

end module interpolate


MODULE wannier_interpolation
  USE prec
  USE vaspxml
  USE lr_helper
  USE twoelectron4o
  USE mkpoints
  USE full_kpoints
  USE wave_high
  USE lattice
  USE poscar
  USE mpimy
  USE mgrid
  USE nonl_high
  USE base
  USE pseudo
  USE kpoints_change
  USE hamil_high
  USE msymmetry
  USE fileio
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: WANNIER_INTERPOL_READER
  PUBLIC :: LWANNIERINTERPOL
  
#if defined(VASP2WANNIER90) && ! defined(gammareal)
  PUBLIC :: WANNIER_INTERPOLATE

  ! common data for all routines
  INTEGER :: KMULT                          ! local name for KINTER
  INTEGER :: NKPTS_COARSE_BZ                ! no. k-points in full BZ for coarse grid
  INTEGER :: NKPTS_COARSE_ADD               ! no. of k-points in coarse grid from which we must interpolate
  INTEGER :: NKPTS_INT                      ! no. k-points in full BZ for interpolated (dense) grid
  REAL(q), ALLOCATABLE :: DISPL_FOUND(:,:)  ! displacements of k-vectors needed for interpolation
  TYPE (kpoints_struct) KPOINTS_INTER       ! kpoint array of IBZ of interpolated grid
  TYPE (kpoints_struct) KPOINTS_BASE        ! copy of original k-points
  TYPE (skpoints_full),SAVE,POINTER ::  KPOINTS_INTER_FULL => NULL() ! full interpolated grid
  TYPE (wavedes)  :: WDES_INTER             ! descriptor for interpolated wave function
  INTEGER, ALLOCATABLE :: NK_EQUIVALENT(:)  ! for the NKPTS_COARSE_ADD k-points, the index of the 
                                            ! corresponding k-point in the IBZ
  REAL(q), ALLOCATABLE :: VKPT_COARSE(:,:)  ! coordinates of k-points in full BZ in coarse grid
  INTEGER, ALLOCATABLE :: NN(:,:,:)         ! nearest neighbour table for coarse k-grid
  GDEF,    ALLOCATABLE :: U(:,:,:,:)        ! wannier rotation matrices
  INTEGER :: NW                             ! number of wannier functions
  TYPE (wavespin), POINTER :: W_INPUT       ! pointer to input wavefunction
  TYPE (in_struct) :: WANNIER_INTERPOL_IO   ! input/output
  ! for orbitals
  COMPLEX(qs), ALLOCATABLE :: RPHI(:,:,:,:,:)  ! derivatives d phi/dk of orbitals 
  COMPLEX(q), ALLOCATABLE :: HAM(:,:,:,:),&    ! Hamiltonian on coarse grid
       HAM_INT(:,:,:,:),&                      ! Hamiltonian on interpolated grid
       DHDK_INT(:,:,:,:,:),&                   ! dH/dk on interpolated grid in direct coord.
       DHDK_INT_CART(:,:,:,:,:)                ! dH/dk on interpolated grid in cartesian coord.
  REAL(q), ALLOCATABLE :: VKPT_INT(:,:)        ! interpolated k-points in same order as in HAM_INT
#endif
  LOGICAL, SAVE :: LINTPOL_WPOT, LINTPOL_ORB, LINTPOL_KPATH,LINTPOL_KPATH_ORB,&
       LINTPOL_VELOCITY, LINTPOL_CONDUCTIVITY, LREAD_EIGENVALUES, LWANNIERINTERPOL

CONTAINS

  SUBROUTINE WANNIER_INTERPOL_READER(IO)
    USE base
    USE vaspxml
    USE crpa, ONLY: LCRPAPLOT
    USE pead, ONLY: PEAD_REQUEST
    USE mlwf, ONLY: LWANNIER90,LWANNIER90_RUN
    IMPLICIT NONE
    TYPE (in_struct) IO
    ! local variables
    INTEGER IDUM, N, IERR
    REAL(q) RDUM
    COMPLEX(q) CDUM
    LOGICAL LOPEN
    CHARACTER (1) :: CHARAC

    LOPEN=.FALSE.
    OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

    ! interpolate WPOT
    LINTPOL_WPOT=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_WPOT','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_WPOT,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_WPOT'' from file INCAR.'
       LINTPOL_WPOT=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_WPOT','L',IDUM,RDUM,CDUM,LINTPOL_WPOT,CHARAC,N)

    ! interpolate orbitals
    LINTPOL_ORB=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_ORB','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_ORB,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_ORB'' from file INCAR.'
       LINTPOL_ORB=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_ORB','L',IDUM,RDUM,CDUM,LINTPOL_ORB,CHARAC,N)

    ! interpolate bandstructure on a given k-grid
    LINTPOL_KPATH=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_KPATH','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_KPATH,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_KPATH'' from file INCAR.'
       LINTPOL_KPATH=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_KPATH','L',IDUM,RDUM,CDUM,LINTPOL_KPATH,CHARAC,N)
    IF( LCRPAPLOT ) THEN
        IF (IO%IU0>=0) WRITE(IO%IU0,*)'Wannier interpolation of k-path set by LCRPAPLOT'
        LINTPOL_KPATH = .TRUE.
    ENDIF 


    ! interpolate also orbitals on given k-grid
    LINTPOL_KPATH_ORB=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_KPATH_ORB','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_KPATH_ORB,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_KPATH_ORB'' from file INCAR.'
       LINTPOL_KPATH_ORB=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_KPATH_ORB','L',IDUM,RDUM,CDUM,LINTPOL_KPATH_ORB,CHARAC,N)

    ! Overwrite eigenvalues with those from EIGENVALUES.INT
    LREAD_EIGENVALUES=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LREAD_EIGENVALUES','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LREAD_EIGENVALUES,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LREAD_EIGENVALUES'' from file INCAR.'
       LREAD_EIGENVALUES=.FALSE.
    ENDIF
    CALL XML_INCAR('LREAD_EIGENVALUES','L',IDUM,RDUM,CDUM,LREAD_EIGENVALUES,CHARAC,N)

    ! Calculate electron velocities dH/dk
    LINTPOL_VELOCITY=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_VELOCITY','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_VELOCITY,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_VELOCITY'' from file INCAR.'
       LINTPOL_VELOCITY=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_VELOCITY','L',IDUM,RDUM,CDUM,LINTPOL_VELOCITY,CHARAC,N)

    ! Calculate conductivity
    LINTPOL_CONDUCTIVITY=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LINTPOL_CONDUCTIVITY','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LINTPOL_CONDUCTIVITY,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LINTPOL_CONDUCTIVITY'' from file INCAR.'
       LINTPOL_CONDUCTIVITY=.FALSE.
    ENDIF
    CALL XML_INCAR('LINTPOL_CONDUCTIVITY','L',IDUM,RDUM,CDUM,LINTPOL_CONDUCTIVITY,CHARAC,N)

    ! Do we need to call WANNIER_INTERPOLATE from LR_SKELETON
    LWANNIERINTERPOL=.FALSE.
    CALL RDATAB(LOPEN,INCAR,IO%IU5,'LWANNIERINTERPOL','=','#',';','L', &
         &            IDUM,RDUM,CDUM,LWANNIERINTERPOL,CHARAC,N,1,IERR)
    IF (((IERR/=0).AND.(IERR/=3)).OR. &
         &                    ((IERR==0).AND.(N<1))) THEN
       IF (IO%IU0>=0) &
            WRITE(IO%IU0,*)'Error reading item ''LWANNIERINTERPOL'' from file INCAR.'
       LWANNIERINTERPOL=.FALSE.
    ENDIF
    ! If we force a call of WANNIER_INTERPOLATE we assume it is because
    ! we are after the velocities, interpolated on a dense k-points grid
    IF (LWANNIERINTERPOL) LINTPOL_VELOCITY=.TRUE.

    LWANNIERINTERPOL=LWANNIERINTERPOL .OR. LINTPOL_WPOT .OR. LINTPOL_ORB .OR. LINTPOL_KPATH &
       .OR. LINTPOL_KPATH_ORB .OR. LINTPOL_VELOCITY .OR. LINTPOL_CONDUCTIVITY

    CALL XML_INCAR('LWANNIERINTERPOL','L',IDUM,RDUM,CDUM,LWANNIERINTERPOL,CHARAC,N)
 
    CLOSE(IO%IU5)

#if ! defined(VASP2WANNIER90) || defined(gammareal)
    IF (LWANNIERINTERPOL) THEN
       IF (IO%IU0>0) THEN
#ifndef VASP2WANNIER90
          WRITE(IO%IU0,*) 'WANNIER_INTERPOL_READER: ERROR: VASP was compiled without the VASP2WANNIER90 interface.'
#endif
#ifdef gammareal
          WRITE(IO%IU0,*) 'WANNIER_INTERPOL_READER: ERROR: Wannier interpolation is not available for gamma-only.'
#endif
       ENDIF
       STOP
    ENDIF
#endif

    IF (LWANNIERINTERPOL) THEN
       ! We need to tell the VASP2WANNIER90 interface
       ! that it has work to do
       LWANNIER90=.TRUE.
       LWANNIER90_RUN=.TRUE.
       ! and consequently the PEAD routines need to
       ! be initialized as well
       CALL PEAD_REQUEST
    ENDIF

    RETURN
  END SUBROUTINE WANNIER_INTERPOL_READER


  !************* SUBROUTINE WANNIER_INTERPOLATE ****************
  !
  ! Entry point to all routines for Wannier interpolation.
  ! Call specialized routines according to flags.
  !
  !*************************************************************
#if defined(VASP2WANNIER90) && ! defined(gammareal)
  SUBROUTINE WANNIER_INTERPOLATE(HAMILTONIAN, KPOINTS, GRID, LATT_CUR, LATT_INI, & 
       &    T_INFO, NONLR_S, NONL_S, W, LMDIM, P, SV, CQIJ, CDIJ, SYMM, INFO, IO)
    IMPLICIT NONE
    TYPE (ham_handle)   HAMILTONIAN
    TYPE (kpoints_struct) KPOINTS
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (latt)        LATT_INI
    TYPE (type_info)   T_INFO
    TYPE (nonlr_struct)NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(:)
    TYPE (wavespin), TARGET :: W
    INTEGER LMDIM
    OVERLAP         CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    OVERLAP         CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    TYPE (symmetry) SYMM
    RGRID           SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO

    IF(LINTPOL_CONDUCTIVITY)LINTPOL_VELOCITY=.TRUE.
    W_INPUT=>W
    WANNIER_INTERPOL_IO=IO

    ! KPOINTS:       coarse mesh, IBZ
    CALL WANNIER_INTERPOL_SETUP_KPOINTS(KPOINTS,LATT_CUR,SYMM,W,IO,GRID,LATT_INI,P,T_INFO,NONLR_S,NONL_S)
    ! KPOINTS:       coarse mesh, full BZ
    ! KPOINTS_BASE:  same as KPOINTS
    ! KPOINTS_INTER: dense mesh, IBZ
    IF(LINTPOL_ORB.OR.LINTPOL_KPATH.OR.LINTPOL_VELOCITY) &
         CALL WANNIER_INTERPOL_SETUP_WANNIER(W,P,LMDIM,CQIJ,T_INFO,LATT_CUR,INFO,IO)
    IF(LINTPOL_ORB.OR.LINTPOL_KPATH.OR.LINTPOL_VELOCITY) &
         CALL WANNIER_INTERPOL_SETUP_HAM(KPOINTS,W,LATT_CUR,IO)
    IF(LINTPOL_ORB.OR.LINTPOL_KPATH) &
         CALL WANNIER_INTERPOL_SETUP_ORBITALS(KPOINTS,GRID,LATT_CUR,LATT_INI,P,T_INFO,W,SYMM,NONL_S,IO)
    ! KPOINTS:       coarse mesh, full BZ, added k-points
    IF(LINTPOL_WPOT) &
         CALL WANNIER_INTERPOLATE_WPOT(W%WDES%ENMAX,GRID,LATT_CUR,IO)
    IF(LINTPOL_ORB) &
         CALL WANNIER_INTERPOLATE_ORBITALS(HAMILTONIAN,KPOINTS,GRID,LATT_CUR,LATT_INI,T_INFO,NONLR_S,&
         NONL_S,W,LMDIM,P,SV,CQIJ,CDIJ,SYMM,INFO,IO)
    IF(LINTPOL_KPATH) &
         CALL WANNIER_INTERPOLATE_KPATH(W,KPOINTS,HAMILTONIAN,GRID,LATT_CUR,T_INFO,NONLR_S,NONL_S,LMDIM, &
         & P,SV,CQIJ,CDIJ,SYMM,INFO,IO)
    IF(LINTPOL_VELOCITY) &
         CALL WANNIER_INTERPOLATE_VELOCITY(W,LATT_CUR,IO)
    IF(LINTPOL_CONDUCTIVITY) &
         CALL WANNIER_INTERPOL_CONDUCTIVITY(KPOINTS,LATT_CUR,T_INFO,W,SYMM,INFO,IO)

  END SUBROUTINE WANNIER_INTERPOLATE


  !************* SUBROUTINE WANNIER_INTERPOL_SETUP_KPOINTS ****************#
  ! 
  ! Setup everything related to k-points:
  ! - KPOINTS_INTER
  ! - DISPL_FOUND (how to shift coarse k-points to get dense k-points)
  ! - NKPTS_COARSE_ADD
  ! - VKPT_COARSE
  ! - NK_EQUIVALENT
  ! - NN 
  ! - KPOINTS_BASE
  ! 
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOL_SETUP_KPOINTS(KPOINTS,LATT_CUR,SYMM,W,IO,GRID,LATT_INI,P,T_INFO,NONLR_S,NONL_S)
    USE prec
    USE wave_high
    USE mlrf_main ! contains KINTER
    USE pead
    IMPLICIT NONE
    TYPE (kpoints_struct) KPOINTS
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (symmetry) SYMM
    TYPE (latt)        LATT_INI
    TYPE (potcar)      P(:)
    TYPE (type_info)   T_INFO
    TYPE (nonlr_struct)NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (wavespin), TARGET :: W
    RGRID           SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
    TYPE (in_struct)   IO
    !    local
    INTEGER :: IWZ,IK,IKX,IKY,IKZ,IKXP,IKYP,IKZP,N,NK
    REAL(q) :: KSTART,DIST,DISPL(3), DISPL_CART(3),DIST_FOUND
    LOGICAL, ALLOCATABLE :: K_FOUND_IN_GRID(:)
    INTEGER, DIMENSION(3) :: MULTIPL


    !-----------------------------------------------------------------------
    ! only gamma centered grids do not violate the symmetry
    ! and are therefore recommended
    IF (KPOINTS%NKPX<1 .OR. KPOINTS%NKPY<1 .OR. KPOINTS%NKPZ<1 )  THEN
       ! the routine works only using regular gamma centered grids
       WRITE(*,*) ' WANNIER_INTERPOL_SETUP_KPOINTS: requires a regular gamma centered k-point grid '
       STOP
    ENDIF
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! create denser k-point grid and store it in KPOINTS_INTER

    ! multiplicator for number of k-points
    KMULT=ABS(KINTER)

    ! initialize things such as smearing etc.
    KPOINTS_INTER=KPOINTS
    MULTIPL(:)=KMULT
    CALL RD_KPOINTS(KPOINTS_INTER, LATT_CUR, &
         SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
         SYMM%ISYM<0, IO%IU6, IO%IU0, MULTIPL = MULTIPL)
    ! generate full interpolated grid
    IF(.NOT.ASSOCIATED(KPOINTS_INTER_FULL))ALLOCATE(KPOINTS_INTER_FULL)
    CALL IBZKPT_HF(LATT_CUR,KPOINTS_INTER,KPOINTS_INTER_FULL,T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6,IO%IU0)
    CALL GEN_LAYOUT_FOR_INTERPOLATION( GRID, W%WDES, WDES_INTER, KPOINTS_INTER, LATT_CUR, LATT_INI, IO%IU6, IO%IU0)
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Precalculate all displacements
    ! Naively one might expect that the shortest displacement
    ! is given by a parallelepiped around the origin.
    ! This, however, is not true for a general bravais lattices
    ! (e.g. hexagonal, fcc of bcc). I.e. there are equivalent displacements
    ! outside the parallelepiped (equivalent means shifted by some original k-point spacing)
    ! that have a shorter distance. Actually, the k-points with the shortest
    ! displacement lie in the Wigner-Seitz cell around the origin.
    ! Calculate the shortest displacements by adding linear
    ! combinations of the original k-point spacing.
    !-----------------------------------------------------------------------
    IWZ=1     ! how many shells of neighbouring unit cells we search
    KSTART=0  ! shift of origin
    IK=0
    ALLOCATE(DISPL_FOUND(3,ABS(KMULT*KMULT*KMULT)))
    DO IKX=-ABS(KMULT-1)/2,ABS(KMULT)/2
       DO IKY=-ABS(KMULT-1)/2,ABS(KMULT)/2
          DO IKZ=-ABS(KMULT-1)/2,ABS(KMULT)/2
             IK=IK+1
             !
             ! search the equivalent displacement with the shortest distance
             DIST_FOUND=1E6
             DO IKXP=-IWZ,IWZ
                DO IKYP=-IWZ,IWZ
                   DO IKZP=-IWZ,IWZ
                      DISPL(1)=((IKX+KSTART)/ABS(KMULT)+IKXP)/KPOINTS%NKPX
                      DISPL(2)=((IKY+KSTART)/ABS(KMULT)+IKYP)/KPOINTS%NKPY
                      DISPL(3)=((IKZ+KSTART)/ABS(KMULT)+IKZP)/KPOINTS%NKPZ
                      ! k-point displacement in cartesian coordinates
                      DISPL_CART=DISPL
                      CALL DIRKAR(1, DISPL_CART(1), LATT_CUR%B)
                      DIST=DISPL_CART(1)**2+DISPL_CART(2)**2+DISPL_CART(3)**2
                      IF (DIST<=DIST_FOUND) THEN
                         DISPL_FOUND(:,IK)=DISPL
                         DIST_FOUND =DIST
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             !
          ENDDO
       ENDDO
    ENDDO
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! create KPOINTS and generate wave function in full BZ
    CALL CHECK_FULL_KPOINTS

    IF (IO%IU0>=0) write(*,*) 'switch off symmetry'
    IF (SYMM%ISYM>=0.AND. .NOT.W%WDES%LGAMMA) THEN
       ! switch off all symmetry
       CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
            &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,W%WDES%ISPIN,IO%IU6)
       ! reread k-points with LINVERSION=.FALSE. to generate full mesh
       CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR,.FALSE., &
            &   T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6, IO%IU0 )
       NKPTS_COARSE_BZ=KPOINTS%NKPTS
       CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_CUR,-1, IO%IU0)
       CALL PEAD_RESETUP_WDES(W%WDES, GRID, KPOINTS, LATT_CUR, LATT_CUR, IO)
       CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)
       CALL RESETUP_FOCK( W%WDES, LATT_CUR)
       CALL PROALL(GRID,LATT_CUR,NONLR_S,NONL_S,W)
    ENDIF
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! now loop over coarse grid and check whether any points
    ! on dense grid are created setting out from this k-point
    ! if so store the point in VKPT_COARSE and set K_FOUND_IN_GRID=.TRUE.
    ALLOCATE(K_FOUND_IN_GRID(NKPTS_COARSE_BZ))
    K_FOUND_IN_GRID=.FALSE.

    DO N=1,NKPTS_COARSE_BZ
       DO IK=1,ABS(KMULT*KMULT*KMULT)
          NK=KPOINT_IN_GRID(W%WDES%VKPT(:,N)+DISPL_FOUND(:,IK),KPOINTS_INTER%VKPT)
          IF (NK/=-1) THEN
#ifdef debug
             WRITE(*,'(I4,9F10.3)') N,W%WDES%VKPT(:,N)+DISPL_FOUND(:,IK)-KPOINTS_INTER%VKPT(:,NK),KPOINTS_INTER%VKPT(:,NK),W%WDES%VKPT(:,N)
#endif 
             K_FOUND_IN_GRID(N)=.TRUE.
             EXIT
          ENDIF
       ENDDO
    ENDDO

    ! count these points:
    ! NKPTS_COARSE_ADD is the number of k-points in coarse grid from which we have to interpolate 
    NKPTS_COARSE_ADD=COUNT(K_FOUND_IN_GRID)

    ! For IK=1..NKPTS_COARSE_ADD, NK_EQUIVALENT(IK) is the index of the base k-point in KPOINTS
    ALLOCATE(VKPT_COARSE(3,NKPTS_COARSE_ADD),NK_EQUIVALENT(NKPTS_COARSE_ADD))
    NKPTS_COARSE_ADD=0
    DO N=1,NKPTS_COARSE_BZ
       IF (K_FOUND_IN_GRID(N)) THEN
          NKPTS_COARSE_ADD=NKPTS_COARSE_ADD+1
          NK_EQUIVALENT(NKPTS_COARSE_ADD)=N
          VKPT_COARSE(:,NKPTS_COARSE_ADD)=W%WDES%VKPT(:,N)
       END IF
    ENDDO
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! set up nearest neighbour table for 3D cubic interpolation
    ALLOCATE(NN(2,3,KPOINTS%NKPTS))
    CALL NN_KPOINTS(KPOINTS,NN)
    !-----------------------------------------------------------------------

    !-----------------------------------------------------------------------
    CALL COPY_KPOINTS(KPOINTS,KPOINTS_BASE)
    !-----------------------------------------------------------------------

  END SUBROUTINE WANNIER_INTERPOL_SETUP_KPOINTS


  !************* SUBROUTINE WANNIER_INTERPOL_SETUP_WANNIER ****************
  ! 
  ! Calculate transformation matrices for Wannier orbitals and store in U.WAN
  ! or read U.WAN if available
  ! 
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOL_SETUP_WANNIER(W,P,LMDIM,CQIJ,T_INFO,LATT_CUR,INFO,IO)
    USE wave_high
    USE mlwf
    IMPLICIT NONE
    TYPE (wavespin),TARGET :: W
    TYPE (potcar)      P(:)
    TYPE (latt)        LATT_CUR
    TYPE (type_info)   T_INFO
    INTEGER            LMDIM
    OVERLAP            CQIJ(LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    ! local
    LOGICAL :: LUWAN
    INTEGER :: I,NC

    IF(.NOT.ALLOCATED(U)) ALLOCATE(U(W%WDES%NB_TOT,W%WDES%NB_TOT,W%WDES%NKPTS,W%WDES%ISPIN))
    ! check if we can read the rotation matrices
    INQUIRE(FILE='U.WAN',EXIST=LUWAN)
    IF(LUWAN)THEN
       IF(IO%IU0>=0) WRITE(*,*) 'reading U.WAN'
       OPEN(UNIT=199,FILE='U.WAN',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
       READ(199) MLWF_num_wann
       READ(199) U
       CLOSE(199)
       NW=MLWF_num_wann
    ELSE
       IF( IO%IU0>=0) WRITE(*,*) 'U.WAN not found'
       
       ! run WANNIER90
       IF (IO%IU0>=0) write(*,*) 'run wannier90'
       CALL MLWF_WANNIER90(W%WDES,W,P,CQIJ,T_INFO,LATT_CUR,INFO,IO)
       NW=MLWF_num_wann
       
       CALL MLWF_GET_U(W%WDES,U)
       
       ! write rotation matrices
       OPEN(UNIT=199,FILE='U.WAN',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
       WRITE(199) MLWF_num_wann
       WRITE(199) U
       CLOSE(199)       
    END IF

  END SUBROUTINE WANNIER_INTERPOL_SETUP_WANNIER


  !************* SUBROUTINE WANNIER_INTERPOL_SETUP_ORBITALS ****************
  ! 
  ! - generate Wannier orbitals
  ! - calculate k-derivatives of Wannier orbitals
  ! - extend KPOINTS, WDES and W by NKPTS_COURSE_ADD dummy k-points for interpolation
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOL_SETUP_ORBITALS(KPOINTS,GRID,LATT_CUR,LATT_INI,P,T_INFO,W,SYMM,NONL_S,IO)
    USE wave_high
    USE lattice
    USE poscar
    USE mgrid
    USE nonl_high
    USE full_kpoints
    USE constant
    USE msymmetry
    USE mlwf
    IMPLICIT NONE
    TYPE (kpoints_struct) KPOINTS
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (latt)        LATT_INI
    TYPE (potcar)      P(:)
    TYPE (type_info)   T_INFO
    TYPE (wavespin), TARGET :: W
    TYPE (symmetry) SYMM
    TYPE (nonl_struct) NONL_S
    TYPE (in_struct)   IO
    ! local
    TYPE (wavespin) :: W_NEW
    INTEGER         :: ISP,N,NK

#ifdef gammareal
    WRITE(*,*) 'WANNIER_INTERPOL_SETUP_ORBITALS: does not work for Gamma point-only version'
    STOP
#endif

    !-----------------------------------------------------------------------
    NKPTS_INT=NKPTS_COARSE_BZ*KMULT**3
    IF(ALLOCATED(RPHI)) DEALLOCATE(RPHI)
    ALLOCATE(RPHI(W%WDES%NRPLWV,W%WDES%NBANDS,KPOINTS%NKPTS,W%WDES%ISPIN,3))
        
    ! construct Wannier functions
    IF(IO%IU0>=0) write(*,*) 'rotate orbitals'
    CALL MLWF_ROTATE_ORBITALS_NOSYMM_U(W%WDES,W,U)
    
    ! calculate derivatives of Wannier functions wrt. k
    IF (IO%IU0>=0) write(*,*) 'calculate derivatives'
    CALL MLWF_DPSI_DK(W,KPOINTS,LATT_CUR,RPHI)
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Reread KPOINTS without! symmetry once again, but now append a copy of the VKPT_COARSE k-points
    ! (those from which we interpolate by applying all shifts).
    ! Thus we have
    ! [ 1..NKPTS_COARSE_BZ   NKPTS_COARSE_BZ+1...NKPTS_COARSE_BZ+NKPTS_COARSE_ADD ]
    !   orig.k-points        copy of irreducible orig.k-points
    ! The copy of the irreducible k-points is then shifted around to generate the interpolated k-points.
    ! This routine also resets KPOINTS_FULL by adding all symmetry equivalent k-points
    ! to the k-points in KPOINTS struct.
    CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, .FALSE., &
         T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM, IO%IU6, IO%IU0, VKPT_COARSE)

    ! This routine re-generates data layout in WDES and GRID for KPOINTS
    ! Thus, in W%WDES%VKPT we have the same as in KPOINTS:
    ! [ 1..NKPTS_COARSE_BZ    NKPTS_COARSE_BZ+1..NKPTS_COARSE_BZ+NKPTS_COARSE_ADD ]
    !   orig. k-points      copy of irreducible orig.k-pts 
    ! in the upper part of the wavefunction arrays the wavefunctions 
    ! corresponding to the new shifted k-points are stored
    ! in the lower part the original wavefunctions are stored
    CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, IO%IU6, IO%IU0 )
    ! REALLOCATE_WAVE does not work with the disentangled/Wannier orbitals!
    ! because disentanglement/Wannierization mixes different bands at each k-point
    ! CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)
    CALL ALLOCW(W%WDES,W_NEW)
    DO ISP=1,W%WDES%ISPIN
       DO N=1,W%WDES%NBANDS
          DO NK=1,NKPTS_COARSE_BZ
             W_NEW%CW(:,N,NK,ISP)=W%CW(:,N,NK,ISP)
             W_NEW%CPROJ(:,N,NK,ISP)=W%CPROJ(:,N,NK,ISP)
             W_NEW%CELTOT(N,NK,ISP)=W%CELTOT(N,NK,ISP)
             W_NEW%FERTOT(N,NK,ISP)=W%FERTOT(N,NK,ISP)
             W_NEW%AUXTOT(N,NK,ISP)=W%AUXTOT(N,NK,ISP)
          ENDDO
       END DO
    END DO
    CALL DEALLOCW(W)
    W=W_NEW
    IF (NONL_S%LRECIP) THEN
       CALL NONL_DEALLOC(NONL_S)
       CALL NONL_ALLOC (NONL_S,T_INFO,P,W%WDES, .FALSE.)
       CALL SPHER(GRID, NONL_S, P, W%WDES, LATT_CUR, 1)
       CALL PHASE(W%WDES,NONL_S,0)
    ENDIF
    !-----------------------------------------------------------------------

  END SUBROUTINE WANNIER_INTERPOL_SETUP_ORBITALS



  !************* SUBROUTINE WANNIER_INTERPOL_SETUP_HAM ********************
  ! 
  ! - apply optional EIGENVALUES.INT file
  ! - apply optional scissors operator
  ! - HAM in Wannier representation
  ! - interpolate and calculate derivatives: HAM_INT, DHDK_INT
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOL_SETUP_HAM(KPOINTS,W,LATT_CUR,IO)
    USE wave_high
    USE xi
    USE mlwf
    IMPLICIT NONE
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt)        LATT_CUR
    TYPE (wavespin), TARGET :: W
    TYPE (in_struct)   IO
    ! local
    REAL(q),ALLOCATABLE :: HELP(:)
    INTEGER :: NK,IB

    !-----------------------------------------------------------------------
    ! apply a set of eigenvalues
    IF(LREAD_EIGENVALUES)THEN
       ALLOCATE(HELP(W%WDES%NB_TOT))
       OPEN(UNIT=199,FILE='EIGENVALUES.INT',FORM='FORMATTED',ACCESS='SEQUENTIAL')
       DO NK=1,W%WDES%NKPTS
          HELP=0
          READ(199,*) HELP
          W%CELTOT(:,NK,1)=HELP
       END DO
       CLOSE(199)
       DEALLOCATE(HELP)
    END IF

    ! apply scissor operator
    IF(SCISSOR/=0)THEN
       CALL APPLY_SCISSOR(W,SCISSOR)
    END IF

#ifdef debug
    WRITE(*,*) 'Input eigenvalues'
    DO NK=1,W%WDES%NKPTS
       WRITE(*,'(''k-point: '',I,3F8.4)') NK,W%WDES%VKPT(:,NK)
       DO IB=1,W%WDES%NB_TOT
          WRITE(*,'(I,2F8.4)') IB,REAL(W%CELTOT(IB,NK,1),q),W%FERTOT(IB,NK,1)
       END DO
    END DO
#endif 
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Calculate Hamiltonian in Wannier representation
    NKPTS_INT=NKPTS_COARSE_BZ*KMULT**3
    IF(ALLOCATED(HAM)) DEALLOCATE(HAM)
    IF(ALLOCATED(HAM_INT)) DEALLOCATE(HAM_INT)
    IF(ALLOCATED(VKPT_INT)) DEALLOCATE(VKPT_INT)
    IF(ALLOCATED(DHDK_INT)) DEALLOCATE(DHDK_INT)
    ALLOCATE(HAM(W%WDES%NB_TOT,W%WDES%NB_TOT,NKPTS_COARSE_BZ,W%WDES%ISPIN),&
         HAM_INT(W%WDES%NB_TOT,W%WDES%NB_TOT,NKPTS_INT,W%WDES%ISPIN),&
         VKPT_INT(3,NKPTS_INT),&
         DHDK_INT(W%WDES%NB_TOT,W%WDES%NB_TOT,NKPTS_INT,W%WDES%ISPIN,3))
        
    IF (IO%IU0>=0) write(*,*) 'setup Wannier Hamiltonian'
    CALL MLWF_ROTATE_HAM(W%WDES,W,HAM,U)
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Interpolate HAM --> HAM_INT and calculate derivative dH/dk --> DHDK_INT
    ! VKPT_INT: interpolated k-points ordered as in HAM_INT
    IF (IO%IU0>=0) write(*,*) 'interpolate Hamiltonian'
    ! we have only NW wannierised orbitals
    CALL FOURIER_INTERPOL(KPOINTS,VKPT_INT,KMULT,HAM(1:NW,1:NW,:,:),HAM_INT,LATT_CUR,IO,DHDK_INT=DHDK_INT)
    !-----------------------------------------------------------------------

  END SUBROUTINE WANNIER_INTERPOL_SETUP_HAM



  !************* SUBROUTINE WANNIER_INTERPOLATE_ORBITALS ********************
  ! 
  ! Interpolates orbitals on a dense k-grid
  ! Needs: Wannier orbitals
  ! - calculate d phi/dk --> RPHI
  ! - interpolate (cubic or linear) and store in W_INTER
  ! - diagonalize interpolated Hamiltonian
  ! - transform interpolated orbitals to eigenfunctions
  ! 
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOLATE_ORBITALS(HAMILTONIAN,KPOINTS,GRID, LATT_CUR, LATT_INI, & 
       &    T_INFO, NONLR_S, NONL_S, W, LMDIM, P, SV, CQIJ, CDIJ, SYMM, INFO, IO)
    USE prec
    USE wave_high
    USE lattice
    USE poscar
    USE mpimy
    USE mgrid
    USE nonl_high
    USE base
    USE pseudo
    USE kpoints_change
    USE full_kpoints
    USE constant
    USE choleski
    USE subrot
    USE mlrf_main
    USE hamil_high
    USE david
    USE msymmetry
    USE fock
    USE mlwf
    USE util
    USE interpolate
    USE ini
    USE fileio
    USE wpot
    USE xi
    IMPLICIT NONE
    TYPE (ham_handle)   HAMILTONIAN
    TYPE (kpoints_struct) KPOINTS
    TYPE (grid_3d)     GRID
    TYPE (latt)        LATT_CUR
    TYPE (latt)        LATT_INI
    TYPE (type_info)   T_INFO
    TYPE (nonlr_struct)NONLR_S
    TYPE (nonl_struct) NONL_S
    TYPE (potcar)      P(:)
    TYPE (wavespin), TARGET :: W
    INTEGER LMDIM
    OVERLAP         CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    OVERLAP         CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    TYPE (symmetry) SYMM
    RGRID           SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    ! local
    INTEGER ISP, NK, NKP, NK_INTER, NB, N, IB, IK, ISPINOR
    REAL(q) :: EXHF,DK(3),DISPL(3),K_INTER(3),K(3)
    TYPE (wavespin) :: W_INTER
    TYPE (nonl_struct) NONL_S_INTER
    TYPE (wavedes1) :: WDES1,WDES1_INTER
    TYPE (wavefun1) :: W1,W1_INTER
    COMPLEX(q),ALLOCATABLE :: CPHASE(:)
    LOGICAL :: LPHASE
    INTEGER :: IDIR, JDIR
    ! for DENSTA
    INTEGER :: NEDOS
    REAL(q) :: ENTROPY,EFERMI
    REAL(q),ALLOCATABLE :: DOS(:,:),DOSI(:,:),PAR(:,:,:,:,:),DOSPAR(:,:,:,:)
#undef debug
#ifdef debug
    COMPLEX(q), ALLOCATABLE :: WC(:,:)
#endif

    CALL ALLOCW(WDES_INTER,W_INTER)

    ALLOCATE(CPHASE(W%WDES%GRID%MPLWV) )
    DK(1)=1._q/REAL(KPOINTS_BASE%NKPX,q)
    DK(2)=1._q/REAL(KPOINTS_BASE%NKPY,q)
    DK(3)=1._q/REAL(KPOINTS_BASE%NKPZ,q)


    !-----------------------------------------------------------------------
    ! loop over all displacements
    DO IK=1,ABS(KMULT*KMULT*KMULT)

       DISPL=DISPL_FOUND(:,IK)
       IF (IO%IU0>=0) WRITE(*,'("DISPL",3G15.5)') DISPL
       DO NK=1,NKPTS_COARSE_ADD
          ! this displacement is applied to all NKPTS_COARSE_ADD k-points
          W%WDES%VKPT(:,NKPTS_COARSE_BZ+NK)=W%WDES%VKPT(:,NK_EQUIVALENT(NK))+DISPL
       ENDDO


       !-----------------------------------------------------------------------
#define cubic_interpolation
#ifdef cubic_interpolation
       IF (IO%IU0>=0) write(*,*) 'cubic interpolation'

       DO NK=1,NKPTS_COARSE_ADD
          NK_INTER=KPOINT_IN_GRID(W%WDES%VKPT(:,NKPTS_COARSE_BZ+NK),KPOINTS_INTER%VKPT)
          IF(NK_INTER>0)THEN
             CALL INTERPOLATE_3D_ORBITALS_SINGLE(W,RPHI,W%CW(:,:,NKPTS_COARSE_BZ+NK,:),KPOINTS_BASE%VKPT,DK,&
                  NK_EQUIVALENT(NK),NN,W%WDES%VKPT(:,NKPTS_COARSE_BZ+NK) )
          ELSE
             NKP=NK_EQUIVALENT(NK)
             DO ISP=1,W%WDES%ISPIN
                DO N=1,W%WDES%NBANDS                     
                   W%CW(:,N,NKPTS_COARSE_BZ+NK,ISP)=W%CW(:,N,NKP,ISP)
                ENDDO
             ENDDO
          END IF
       END DO
#else
       ! linear interpolation
       IF (IO%IU0>=0) write(*,*) 'linear interpolation'
       DO ISP=1,W%WDES%ISPIN
          DO NK=1,NKPTS_COARSE_ADD
             NKP=NK_EQUIVALENT(NK)                  
             DO N=1,W%WDES%NBANDS
                W%CW(:,N,NKPTS_COARSE_BZ+NK,ISP)=W%CW(:,N,NKP,ISP)+TPI*( &
                     RPHI(:,N,NKP,ISP,1)*DISPL(1)+ & 
                     RPHI(:,N,NKP,ISP,2)*DISPL(2)+ &
                     RPHI(:,N,NKP,ISP,3)*DISPL(3))
             ENDDO
          ENDDO
       ENDDO
#endif
       !-----------------------------------------------------------------------

       !-----------------------------------------------------------------------
       ! Store orbital in W_INTER 
       ! Possibly requires a phase shift
       DO NK=1,NKPTS_COARSE_ADD
          NK_INTER=KPOINT_IN_GRID(W%WDES%VKPT(:,NKPTS_COARSE_BZ+NK),KPOINTS_INTER%VKPT)
          IF (NK_INTER>=1) THEN
             ! single wavedescriptor on coarse grid
             CALL SETWDES(W%WDES,WDES1,NKPTS_COARSE_BZ+NK)
             CALL NEWWAV(W1,WDES1,.TRUE.)
             ! single wavedescriptor on dense grid
             CALL SETWDES(WDES_INTER,WDES1_INTER,NK_INTER)
             CALL NEWWAV(W1_INTER,WDES1_INTER,.TRUE.)
             ! The interpolated wavefunction is up to now stored in the layout of WDES1
             ! and has to be transformed to the layout of WDES1_INTER.
             ! This can only be done via FFTing into real space (where all k-points
             ! have the same layout) and back.
             ! The k-point in the coarse grid and in the fine grid may also differ
             ! by some reciprocal lattice vector, which requires an additional phase
             ! shift to be applied.

             ! check if phase shift required
             K=W%WDES%VKPT(:,NKPTS_COARSE_BZ+NK)
             K_INTER=KPOINTS_INTER%VKPT(:,NK_INTER)
             CALL SETPHASE( K-K_INTER, WDES1%GRID, CPHASE, LPHASE)

             DO IB=1,W%WDES%NBANDS
                DO ISP=1,W%WDES%ISPIN
                   ! copy interpolated orbital from W to W_INTER using FFT
                   ! copy from W to W1; W1%WDES1 is set to WDES1 => W1 has layout of k-point in coarse grid
                   CALL W1_COPY(ELEMENT(W,WDES1,IB,ISP),W1)
                   ! FFT to real space using layout in coarse grid
                   CALL FFTWAV_W1(W1)
                   ! apply phase shift from K to K_INTER
                   IF(LPHASE) CALL APPLY_PHASE(WDES1%GRID,CPHASE,W1%CR,W1%CR)
                   W1_INTER%CR=W1%CR/WDES1_INTER%GRID%NPLWV
                   ! FFT to recipr. space using layout in fine grid
                   DO ISPINOR=0,WDES1%NRSPINORS-1
                      CALL FFTEXT(WDES1_INTER%NGVECTOR,WDES1_INTER%NINDPW(1),W1_INTER%CR(1+ISPINOR*WDES1_INTER%GRID%MPLWV),W1_INTER%CW(1+ISPINOR*WDES1_INTER%NGVECTOR),WDES1_INTER%GRID,.FALSE.)
                   ENDDO
                   CALL W1_COPY(W1_INTER,ELEMENT(W_INTER,WDES1_INTER,IB,ISP))
                END DO
             END DO
             CALL DELWAV(W1,.TRUE.)
             CALL DELWAV(W1_INTER,.TRUE.)            
          ENDIF
       ENDDO
       !-----------------------------------------------------------------------

    END DO
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Now work on dense k-point grid only
    !-----------------------------------------------------------------------

    !-----------------------------------------------------------------------
    ! non local projection operators for fine grid
    CALL NONL_ALLOC(NONL_S_INTER, T_INFO, P, WDES_INTER, INFO%LREAL)
    IF (NONLR_S%LREAL) THEN
       CALL RSPHER(GRID,NONLR_S,LATT_CUR)
    ELSE
       CALL SPHER(GRID,NONL_S_INTER,P,W_INTER%WDES,LATT_CUR, 1)
    ENDIF

    W_INTER%WDES%NB_TOTK(:,:)=MLWF_num_wann
    CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S_INTER,W_INTER)
    CALL ORTHCH(W_INTER%WDES,W_INTER, W_INTER%WDES%LOVERL, LMDIM, CQIJ)
    !-----------------------------------------------------------------------

    !-----------------------------------------------------------------------
    DEALLOCATE(HAM)
    ALLOCATE(HAM(WDES_INTER%NB_TOT,WDES_INTER%NB_TOT,WDES_INTER%NKPTS,WDES_INTER%ISPIN))

    ! HAM should contain the interpolated Hamiltonian only for the k-points 
    ! in WDES_INTER and also in the same order
    DO NK_INTER=1,WDES_INTER%NKPTS
       NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
       IF(NK>=1)THEN
          HAM(:,:,NK_INTER,:)=HAM_INT(:,:,NK,:)
       ELSE
          WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
          STOP
       END IF
    END DO

    ! Diagonalize interpolated Hamiltonian. The unitary matrices that diagonalize
    ! the Hamiltonian at each k-point transform from the Wannier representation back 
    ! to eigenbasis. Use this to also transform the interpolated Wannier orbitals to
    ! interpolated eigenorbitals.
    CALL EDDIAG_WANNIER(HAM,W_INTER,W_INTER%WDES,IO%IU0,IO%IU6,NKSTART=1)
    WRITE(*,*) 'Interpolated eigenvalues'
    DO NK_INTER=1,WDES_INTER%NKPTS
       WRITE(*,'(''k-point: '',I2,3F8.4)') NK_INTER,WDES_INTER%VKPT(:,NK_INTER)
       DO IB=1,MLWF_num_wann
          WRITE(*,'(I2,F8.4)') IB,REAL(W_INTER%CELTOT(IB,NK_INTER,1),q)
       END DO
    END DO
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
#ifdef debug
    IF(ALLOCATED(WC))DEALLOCATE(WC)
    ALLOCATE(WC(WDES_INTER%NB_TOT,WDES_INTER%NKPTS))
    IF (IO%IU0>=0)THEN
       WRITE(*,*) 'Diagonalization of interpolated Hamiltonian'
       DO NK_INTER=1,WDES_INTER%NKPTS
          WRITE(*,'(I2,3F8.4,50F8.4)') NK_INTER,WDES_INTER%VKPT(:,NK_INTER),REAL(W_INTER%CELTOT(1:MLWF_num_wann,NK_INTER,1),q)
       END DO
       WC=W_INTER%CELTOT(:,:,1)
    END IF

    ! Expectation value of exact Hamiltonian with interpolated orbitals
    CALL EDDIAG(HAMILTONIAN,&
         GRID,LATT_CUR,NONLR_S,NONL_S_INTER,W_INTER,WDES_INTER,SYMM, &
         LMDIM,CDIJ,CQIJ,1,SV,T_INFO,P,IO%IU0,EXHF,NKSTART=1)
    IF (IO%IU0>=0)THEN
       WRITE(*,*) 'Expectation value of exact Hamiltonian in interpolated orbitals'
       DO NK_INTER=1,WDES_INTER%NKPTS
          WRITE(*,'(I,3F8.4,50F8.4)') NK_INTER,W_INTER%WDES%VKPT(:,NK_INTER),REAL(W_INTER%CELTOT(1:MLWF_num_wann,NK_INTER,1),q)
       END DO

       WRITE(*,*) 'Difference'
       DO NK_INTER=1,WDES_INTER%NKPTS
          WRITE(*,'(I,3F8.4,50F8.4)') NK_INTER,W_INTER%WDES%VKPT(:,NK_INTER),&
               REAL( WC(1:MLWF_num_wann,NK_INTER)-W_INTER%CELTOT(1:MLWF_num_wann,NK_INTER,1),q ) , &
               MAXVAL(ABS(REAL( WC(1:MLWF_num_wann,NK_INTER)-W_INTER%CELTOT(1:MLWF_num_wann,NK_INTER,1),q)))
       END DO
    END IF

    ! Diagonalize exact Hamiltonian in interpolated orbitals
    CALL EDDIAG(HAMILTONIAN,&
         GRID,LATT_CUR,NONLR_S,NONL_S_INTER,W_INTER,WDES_INTER,SYMM, &
         LMDIM,CDIJ,CQIJ,2,SV,T_INFO,P,IO%IU0,EXHF,NKSTART=1)
    IF (IO%IU0>=0)THEN
       WRITE(*,*) 'Diagonalization of exact Hamiltonian in interpolated orbitals'
       DO NK_INTER=1,WDES_INTER%NKPTS
          WRITE(*,'(I,3F8.4,50F8.4)') NK_INTER,W_INTER%WDES%VKPT(:,NK_INTER),REAL(W_INTER%CELTOT(1:MLWF_num_wann,NK_INTER,1),q)
       END DO
    END IF
#endif
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! calculate occupancies
    ENTROPY=0
    EFERMI=0
    NEDOS=1000
    ALLOCATE(DOS(NEDOS,WDES_INTER%ISPIN),DOSI(NEDOS,WDES_INTER%ISPIN),&
         PAR(1,1,1,1,WDES_INTER%NCDIJ),DOSPAR(1,1,1,WDES_INTER%NCDIJ))
    DOS=0
    DOSI=0
    PAR=0
    DOSPAR=0
    ! bands beyond MLWF_num_wann are set to practically infinite energy
    DO N=1,WDES_INTER%NBANDS
       IB=(N-1)*W_INTER%WDES%NB_PAR+W_INTER%WDES%NB_LOW
       IF(IB>MLWF_num_wann)THEN
          DO ISP=1,WDES_INTER%ISPIN
             DO IK=1,WDES_INTER%NKPTS
                W_INTER%CELEN(N,IK,ISP)=1E30_q
                W_INTER%CELTOT(IB,IK,ISP)=1E30_q
             END DO
          END DO
       END IF
    END DO
    CALL DENSTA(IO%IU0,IO%IU6,WDES_INTER,W_INTER,KPOINTS_INTER,INFO%NELECT,&
         INFO%NUP_DOWN,ENTROPY,EFERMI,KPOINTS_INTER%SIGMA,.FALSE.,&
         NEDOS,0,0,DOS,DOSI,PAR,DOSPAR)
    DEALLOCATE(DOS,DOSI,PAR,DOSPAR)
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! Write out interpolated wave function
    ! set NB_TOT to number of wannier functions MLWF_num_wann for writing WAVECAR.INT
    W_INTER%WDES%NB_TOT=MLWF_num_wann
    IF (IO%IU0>=0)THEN
       DO ISP=1,WDES_INTER%ISPIN
          DO IK=1,WDES_INTER%NKPTS
             WRITE(*,*) 'IK',IK
             DO IB=1,WDES_INTER%NB_TOT
                WRITE(*,*) IB,REAL(W_INTER%CELTOT(IB,IK,ISP),q),W_INTER%FERTOT(IB,IK,ISP)
             ENDDO
          ENDDO
       ENDDO
    END IF
    CALL OUTWAV(IO, WDES_INTER, W_INTER, LATT_CUR, 0._q, EXT='INT')
    !-----------------------------------------------------------------------


    !-----------------------------------------------------------------------
    ! get rid of all the allocated stuff
    DEALLOCATE(HAM,HAM_INT,DHDK_INT,RPHI,CPHASE)
    CALL NONL_DEALLOC(NONL_S_INTER)
    CALL DEALLOCW(W_INTER)
    CALL DEALLOCWDES(WDES_INTER,.TRUE.)
    !-----------------------------------------------------------------------
    

    !-----------------------------------------------------------------------
    ! restore old data layout
    !-----------------------------------------------------------------------
    CALL XML_CLOSE_TAG
#ifdef oldsym
    CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
         SYMM%ISYM>=0.AND..NOT.W%WDES%LSORBIT.AND..NOT.W%WDES%LSPIRAL, &
         T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#else
    CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR, &
         SYMM%ISYM>=0.AND..NOT.W%WDES%LNONCOLLINEAR, &
         T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,-1,-1)
#endif
    CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_INI, -1, -1)
    CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)
    !-----------------------------------------------------------------------  

  END SUBROUTINE WANNIER_INTERPOLATE_ORBITALS


  !************* SUBROUTINE WANNIER_INTERPOLATE_VELOCITY ******************
  ! 
  ! Calculate electron velocities dH/dk
  ! by diagonalizing HAM_INT, transforming DHDK_INT and converting to cartesian coordinates
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOLATE_VELOCITY(W,LATT_CUR,IO)
    USE constant
    IMPLICIT NONE
    TYPE (wavespin) :: W
    TYPE (latt)        LATT_CUR
    TYPE (in_struct)   IO
    ! local
    INTEGER :: NK,NK_INTER,IB,IK,IDIR
    REAL(q),ALLOCATABLE :: EINTERPOL_FULL(:,:,:,:)


    !-----------------------------------------------------
    ! Diagonalize HAM_INT and transform DHDK_INT with the diagonalizing matrix
    ! HAM_INT and DHDK_INT have been calculated before in WANNIER_INTERPOLATE_SETUP_HAM
    IF (IO%IU0>=0) write(*,*) 'diagonalize Hamiltonian and velocities'
    CALL HAM_DIAG(NW,NKPTS_INT,W%WDES%ISPIN,HAM_INT(1:NW,1:NW,:,:),DHDK=DHDK_INT(1:NW,1:NW,:,:,:))

#ifdef debug
    WRITE(*,*) 'Interpolated eigenvalues'
    DO NK_INTER=1,WDES_INTER%NKPTS
       NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
       IF(NK>=1)THEN
          WRITE(*,'(I2,50F10.5)') NK_INTER,VKPT_INT(:,NK),(/ (REAL(HAM_INT(IB,IB,NK,1),q),IB=1,W%WDES%NB_TOT) /)
       ELSE
          WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
          STOP
       END IF
    END DO
#endif    
    !-----------------------------------------------------


    !-----------------------------------------------------
    ! Transform to cartesian coordinates
    !
    ! d/dk_i = dk_m/dk_i d/dk_m   where i=cartesian, m=direct coordinate
    ! \vec{k} = k_m \vec{b}^m
    ! 2pi k_m = \vec{a}^m . \vec{k} = a^m_i k_i
    ! Therefore: dk_m/dk_i = a^m_i/2pi
    IF(ALLOCATED(DHDK_INT_CART))DEALLOCATE(DHDK_INT_CART)
    ALLOCATE(DHDK_INT_CART(NW,NW,NKPTS_INT,W%WDES%ISPIN,3))
    DHDK_INT_CART=0
    DO IB=1,3
       DO IK=1,3
          DHDK_INT_CART(:,:,:,:,IB)=DHDK_INT_CART(:,:,:,:,IB)+DHDK_INT(1:NW,1:NW,:,:,IK)*LATT_CUR%A(IB,IK)/TPI
       END DO
    END DO
    !-----------------------------------------------------


    !-----------------------------------------------------
    ! Save derivatives in file DHDK.INT
#ifdef debug
    OPEN(UNIT=199,FILE='DHDK.INT',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
    WRITE(199) NW,WDES_INTER%NKPTS
    DO NK_INTER=1,WDES_INTER%NKPTS
       NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
       IF(NK>=1)THEN
          WRITE(199) VKPT_INT(:,NK),(/ ((REAL(DHDK_INT(IB,IB,NK,1,IK),q),IB=1,NW),IK=1,3) /)
       ELSE
          WRITE(*,*) 'error in WANNIER_INTERPOLATE_VELOCITY: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
          STOP
       END IF
    END DO
    CLOSE(199)
#endif
    !-----------------------------------------------------


    !-----------------------------------------------------
#undef debug
#ifdef debug
    ! FOR DEBUGGING
    WRITE(*,*) 'direct derivatives, real part'
    DO IK=1,3
       WRITE(*,'("# DIR = ",I)') IK
       DO NK_INTER=1,WDES_INTER%NKPTS
          NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
          IF(NK>=1)THEN
             WRITE(*,'(I,3F10.5,100F12.4)') NK,VKPT_INT(:,NK),(/ (REAL(DHDK_INT(IB,IB,NK,1,IK),q),IB=1,NW) /)
          ELSE
             WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
             STOP
          END IF
       END DO
    END DO

    WRITE(*,*) 'direct derivatives, imag. part'
    DO IK=1,3
       WRITE(*,'("# DIR = ",I)') IK
       DO NK_INTER=1,WDES_INTER%NKPTS
          NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
          IF(NK>=1)THEN
             WRITE(*,'(I,3F10.5,100F12.4)') NK,VKPT_INT(:,NK),(/ (AIMAG(DHDK_INT(IB,IB,NK,1,IK)),IB=1,NW) /)
          ELSE
             WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
             STOP
          END IF
       END DO
    END DO

    !-----------------------------------------------------


    !-----------------------------------------------------
    ! write the cartesian derivatives deps/dk
    WRITE(*,*) 'cartesian derivatives'
    DO NK_INTER=1,WDES_INTER%NKPTS
       NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
       IF(NK>=1)THEN
          DO IB=1,4
             DO IK=1,3
                WRITE(*,'(3I2,3F10.5,F10.5)') IK,NK,IB,VKPT_INT(:,NK),(/ REAL(DHDK_INT_CART(IB,IB,NK,1,IK),q) /)
             ENDDO
          ENDDO
       ELSE
          WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
          STOP
       END IF
    END DO
#endif
    !-----------------------------------------------------


    !-----------------------------------------------------
    ALLOCATE(EINTERPOL_FULL(4,NW,KPOINTS_INTER_FULL%NKPTS,1))
    DO NK_INTER=1,KPOINTS_INTER_FULL%NKPTS
       NK=KPOINT_IN_GRID( KPOINTS_INTER_FULL%VKPT(:,NK_INTER), VKPT_INT)
       DO IB=1,NW
          EINTERPOL_FULL(1,IB,NK_INTER,1)=REAL( HAM_INT(IB,IB,NK,1),q )
          DO IDIR=1,3
             EINTERPOL_FULL(1+IDIR,IB,NK_INTER,1)=REAL( DHDK_INT_CART(IB,IB,NK,1,IDIR),q )
          END DO
       END DO
    END DO
    CALL XML_TAG("wannierinterpolation")
    CALL XML_TAG("kpoints_full")
    CALL XML_KPOINTS_LIST(KPOINTS_INTER_FULL%VKPT, KPOINTS_INTER_FULL%WTKPT)
    CALL XML_CLOSE_TAG("kpoints_full")
    CALL XML_EIGENVALUES_EXT(EINTERPOL_FULL,4, W%WDES%NB_TOT, KPOINTS_INTER_FULL%NKPTS, W%WDES%ISPIN)
    CALL XML_CLOSE_TAG("wannierinterpolation")
    DEALLOCATE(EINTERPOL_FULL,DHDK_INT_CART)
    !-----------------------------------------------------

  END SUBROUTINE WANNIER_INTERPOLATE_VELOCITY


  !************* SUBROUTINE WANNIER_INTERPOLATE_CONDUCTIVITY ******************
  ! 
  ! Calculate conductivity
  ! by diagonalizing HAM_INT, transforming DHDK_INT and converting to cartesian coordinates
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOL_CONDUCTIVITY(KPOINTS,LATT_CUR,T_INFO,W,SYMM,INFO,IO)
    USE prec
    USE lattice
    USE base
    USE kpoints_change
    USE full_kpoints
    USE constant
    USE mlwf
    USE mlrf_main
    USE fileio
    USE mlr_optic
    USE pseudo
    USE util
    USE msymmetry
    IMPLICIT NONE
    TYPE (kpoints_struct) KPOINTS
    TYPE (latt)        LATT_CUR
    TYPE (type_info)   T_INFO
    TYPE (wavespin), TARGET :: W
    TYPE (symmetry) SYMM
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    ! local
    TYPE (wavespin) :: W_DER
    ! for calculation of conductivity
    INTEGER :: NEDOS, LDIMP, NIOND, IDIR, JDIR, NK_INTER, IB, NK
    REAL(q) :: ENTROPY,EFERMI,MYWPLASMON(3,3),CON(3,3)
    LOGICAL :: LNOAUTO
    REAL(q),ALLOCATABLE :: DER1(:,:,:),DER2(:,:,:)
    REAL(q),ALLOCATABLE :: DOS(:,:),DOSI(:,:),PAR(:,:,:,:,:),DOSPAR(:,:,:,:)
    ! symmetry related quantities
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
    ! dummy for INISYM
    REAL(q), ALLOCATABLE :: VEL(:,:)      


    !-----------------------------------------------------
    CALL ALLOCW_NOPLANEWAVE(WDES_INTER,W_DER)
    W_DER%CELTOT=1e6_q
    DO NK_INTER=1,W_DER%WDES%NKPTS
       NK=KPOINT_IN_GRID( W_DER%WDES%VKPT(:,NK_INTER), VKPT_INT)
       IF(NK>=1)THEN
          !
          DO IB=1,NW
             W_DER%CELTOT(IB,NK_INTER,1)=HAM_INT(IB,IB,NK,1)
          END DO
          !
       ELSE
          WRITE(*,*) 'error in WANNIER_INTERPOLATE_CONDUCTIVITY: k-point not found in VKPT_INT',W_DER%WDES%VKPT(:,NK_INTER)
          STOP
       END IF
    END DO
    ! probably FERTOT is overwritten anyway
    W_DER%FERTOT=W%FERTOT

    NEDOS=1000
    LDIMP=0
    NIOND=0
    LNOAUTO=.TRUE.      
    ALLOCATE(DOS(NEDOS,W_DER%WDES%ISPIN),DOSI(NEDOS,W_DER%WDES%ISPIN),&
         PAR(W_DER%WDES%NB_TOT, W_DER%WDES%NKPTS, LDIMP, NIOND, W_DER%WDES%NCDIJ),&
         DOSPAR(NEDOS,LDIMP,NIOND,W_DER%WDES%NCDIJ) )
    CALL DENSTA( IO%IU0, IO%IU6 ,W_DER%WDES, W_DER, KPOINTS_INTER, INFO%NELECT, INFO%NUP_DOWN, &
         ENTROPY, EFERMI, KPOINTS_INTER%SIGMA, LNOAUTO, NEDOS, LDIMP, NIOND, DOS, DOSI, PAR, DOSPAR)
    !-----------------------------------------------------


    !-----------------------------------------------------
    ALLOCATE(DER1(W_DER%WDES%NB_TOT,W_DER%WDES%NKPTS,W_DER%WDES%ISPIN),&
         DER2(W_DER%WDES%NB_TOT,W_DER%WDES%NKPTS,W_DER%WDES%ISPIN))
    DER1=0
    DER2=0
    ! this must be set, otherwise it's 0
    NBANDS_CDER=NW
    WPLASMON=0
    CON=0
    DO IDIR=1,3
       DO JDIR=1,3
          !
          !
          DO NK_INTER=1,WDES_INTER%NKPTS
             NK=KPOINT_IN_GRID( WDES_INTER%VKPT(:,NK_INTER), VKPT_INT)
             IF(NK>=1)THEN
                !
                DO IB=1,NW
                   DER1(IB,NK_INTER,:)=DHDK_INT_CART(IB,IB,NK,:,IDIR)
                   DER2(IB,NK_INTER,:)=DHDK_INT_CART(IB,IB,NK,:,JDIR)
                END DO
                !
             ELSE
                WRITE(*,*) 'error in INTERPOLATE_BAND_STR_WANNIER: k-point not found in VKPT_INT',WDES_INTER%VKPT(:,NK_INTER)
                STOP
             END IF
          END DO

          CALL CONDUCTIVITY(W_DER%WDES,W_DER,DER1,DER2,EFERMI,KPOINTS_INTER%ISMEAR,KPOINTS_INTER%SIGMA,&
               LATT_CUR%OMEGA,WPLASMON(IDIR,JDIR),CON(IDIR,JDIR),RTIME)
          !
       END DO
    END DO

    WRITE(*,*) '======================================================='
    WRITE(*,*) 'WANNIER_INTERPOL_CONDUCTIVITY:'
    WRITE(*,*) '------------------------------'
    WRITE(*,*) 'EFERMI=',EFERMI
    WRITE(*,*) 'ISMEAR=',KPOINTS_INTER%ISMEAR
    WRITE(*,*) 'SIGMA=', KPOINTS_INTER%SIGMA
    WRITE(*,*) 'RTIME=', RTIME
    CALL MAT_DUMP(CON,'CON')
    CALL MAT_DUMP(WPLASMON,'WPLASMON')

    ALLOCATE(VEL(3,T_INFO%NIONS))
    CALL INISYM(LATT_CUR%A,T_INFO%POSION,VEL,T_INFO%LSFOR,&
         T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,&
         SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,&
         SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT,&
         SYMM%INDROT,T_INFO%ATOMOM,W%WDES%SAXIS,SYMM%MAGROT,W%WDES%NCDIJ,IO%IU6)
    DEALLOCATE(VEL)
    IF (SYMM%ISYM>0) CALL TSYM(CON(:,:),ISYMOP,NROTK,LATT_CUR%A)
    CALL MAT_DUMP(CON,'CON symm.')
    WRITE(*,*) '======================================================='

    DEALLOCATE(DER1,DER2,DOS,DOSI,PAR,DOSPAR)
    !-----------------------------------------------------

  END SUBROUTINE WANNIER_INTERPOL_CONDUCTIVITY


  !************* SUBROUTINE WANNIER_INTERPOLATE_CONDUCTIVITY ******************
  ! 
  ! interpolation of W (screened Coulomb potential)
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOLATE_WPOT(ENMAX,GRID,LATT_CUR,IO)
    USE prec
    USE base
    USE constant
    USE wpot
    USE xi
    IMPLICIT NONE

    REAL(q) :: ENMAX
    TYPE (grid_3d) :: GRID
    TYPE (latt) :: LATT_CUR
    TYPE (in_struct) :: IO
    !    local
    INTEGER NK, NKP, NK_INTER, IK
    REAL(q) :: DISPL(3), K(3), K_INTER(3), DK(3)
    COMPLEX(q),ALLOCATABLE :: CPHASE(:)
    LOGICAL :: LPHASE
    TYPE(wpothandle), POINTER :: POT,POT_INT
    TYPE (wavedes), POINTER :: POTDES,POTDES_INT
    TYPE(wavedes1) :: POTDESK,POTDESK_INT
    TYPE (grid_3d), POINTER :: POTGRID,POTGRID_INT
    INTEGER :: K1,K2,NG,I,KZERO
    REAL(q),ALLOCATABLE :: POTFAK(:)
    COMPLEX(q),ALLOCATABLE :: EPS1(:,:),EPS1_INT(:,:),EPS1_REAL(:),EPS1_TRANS(:),DEPS1(:,:,:)
    TYPE(responsefunction),POINTER :: RESP_INT(:)
    LOGICAL :: LFULL
    COMPLEX(q) :: WSING
    REAL(q) :: G1,G2,G3,GX,GY,GZ
    REAL(q), ALLOCATABLE :: VKPT_INT(:,:)
    !------------------------------------


    !-----------------------------------------------
    ! wave descriptor for original POT
    ! uses full k-points
    ALLOCATE(POTDES,POTGRID)
    POTDES                     =WDES_FOCK
    POTDES%NKPTS               =KPOINTS_FULL%NKPTS
    POTDES%NKDIM               =KPOINTS_FULL%NKPTS
    POTDES%NKPTS_FOR_GEN_LAYOUT=KPOINTS_FULL%NKPTS
    POTDES%VKPT               =>KPOINTS_FULL%VKPT
    POTDES%WTKPT              =>KPOINTS_FULL%WTKPT
    ! if ENCUTGW=ENCUT, then POTGRID and GRID should be identical
    ! i.e. set ENGUTGW=ENCUT in G0W0 calculation!
    POTDES%ENMAX=ENCUTGW
    ! POTGRID is identical to GRID_FOCK, except for POTGRID%FFTSCA
    POTGRID=GRID_FOCK
    IF (POTDES%LGAMMA) THEN
       ! gamma only data layout with wavefunction stored as real in real space
       CALL GEN_LAYOUT(POTGRID, POTDES, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
       POTGRID%LREAL=.TRUE.
    ELSE
       CALL GEN_LAYOUT(POTGRID, POTDES, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
    ENDIF
    CALL GEN_INDEX (POTGRID, POTDES, LATT_CUR%B, LATT_CUR%B, IO%IU6, -1, .TRUE.)
    ! init FFT (required if real to complex FFT is used)
    CALL FFTINI(POTDES%NINDPW(1,1), POTDES%NGVECTOR(1), POTDES%NKPTS, POTDES%NGDIM, POTGRID)
    ! store Coulomb kernel in DATAKE =(4pi e^2)/|G+k|^2
    IF (ENCUTGW /= ENCUTGWSOFT .AND. ENCUTGW > 0 .AND. ENCUTGWSOFT >0 ) THEN
       CALL XI_DATAKE( POTDES, LATT_CUR, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL XI_DATAKE( POTDES, LATT_CUR)
    ENDIF
    !-----------------------------------------------


    !-----------------------------------------------
    ! wave descriptor for interpolated POT
    ! uses only IBZ of KPOINTS_INTER
    ALLOCATE(POTDES_INT,POTGRID_INT)
    POTDES_INT                     =WDES_FOCK
    POTDES_INT%NKPTS               =KPOINTS_INTER%NKPTS
    POTDES_INT%NKDIM               =KPOINTS_INTER%NKPTS
    POTDES_INT%NKPTS_FOR_GEN_LAYOUT=KPOINTS_INTER%NKPTS
    POTDES_INT%VKPT               =>KPOINTS_INTER%VKPT
    POTDES_INT%WTKPT              =>KPOINTS_INTER%WTKPT
    POTDES_INT%ENMAX=ENCUTGW
    ! POTGRID_INT is identical to GRID_FOCK, except for POTGRID_INT%FFTSCA
    POTGRID_INT=GRID_FOCK
    IF (POTDES_INT%LGAMMA) THEN
       ! gamma only data layout with wavefunction stored as real in real space
       CALL GEN_LAYOUT(POTGRID_INT, POTDES_INT, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
       POTGRID_INT%LREAL=.TRUE.
    ELSE
       CALL GEN_LAYOUT(POTGRID_INT, POTDES_INT, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
    ENDIF
    CALL GEN_INDEX (POTGRID_INT, POTDES_INT, LATT_CUR%B, LATT_CUR%B, IO%IU6, -1, .TRUE.)
    ! init FFT (required if real to complex FFT is used)
    CALL FFTINI(POTDES_INT%NINDPW(1,1), POTDES_INT%NGVECTOR(1), POTDES_INT%NKPTS, POTDES_INT%NGDIM, POTGRID_INT)
    ! store Coulomb kernel in DATAKE (4pi e^2)/|G+k|^2
    !     write(*,*) ENCUTGW, ENCUTGWSOFT
    IF (ENCUTGW /= ENCUTGWSOFT .AND. ENCUTGW > 0 .AND. ENCUTGWSOFT >0 ) THEN
       CALL XI_DATAKE( POTDES_INT, LATT_CUR, ENCUTGW, ENCUTGWSOFT)
    ELSE
       CALL XI_DATAKE( POTDES_INT, LATT_CUR)
    ENDIF
    !-----------------------------------------------


    !-----------------------------------------------
    ! Read WFULL000X.tmp or W000X.tmp files
    NULLIFY(POT)
    CALL INIT_WPOT_HANDLE(POT,POTDES,POTDES%NKPTS,IO%IU6,IO%IU0,1,1,1)
    ! read all kernels at all k-points and set them up
    ALLOCATE(POTFAK(POT%WGW%NGDIM))
    DO K1=1,POTDES%NKPTS
       ! W is stored for all _differences_ of k-points q=k-k'
       CALL GET_WPOT( POT, K1, POTFAK, LFULL )
    END DO
    ! undo division done in GET_WPOT - we want true potential, no integration
    POT%C=POT%C*GRIDHF%NPLWV
    DEALLOCATE(POTFAK)
    !-----------------------------------------------


    !-----------------------------------------------
    ! EPS1 stores the diagonal of inverse dielectric matrix eps^-1_G,G(q) = eps^-1(G+q)
    ! POT%C(NG,NQ)         = W(G+q,G+q) = (4 pi e^2)/|G+q|^2 * eps^-1_G,G(q)
    ! POTDES%DATAKE(NG,NQ) = v(G+q,G+q) = (4 pi e^2)/|G+q|^2
    ALLOCATE(EPS1(POTDES%NGDIM,POTDES%NKPTS))
    ! interpolated eps^-1
    ALLOCATE(EPS1_INT(POTDES%NGDIM,NKPTS_COARSE_ADD),EPS1_REAL(POTDES%GRID%MPLWV),EPS1_TRANS(POTDES_INT%NGDIM))
    ! Interpolated responsefunction
    ALLOCATE(RESP_INT(KPOINTS_INTER%NKPTS))
    DO IK=1,KPOINTS_INTER%NKPTS
       CALL ALLOCATE_RESPONSEFUN(RESP_INT(IK),POTDES_INT%NGDIM,POTDES_INT%LGAMMA,POTDES_INT%LGAMMA,1)
       ! only one node should write WPOT file!
       IF(IO%IU0>=0) RESP_INT(IK)%NOMEGA_LOW=1
    END DO
    ! Divide to get eps^-1(G+q)
    DO K1=1,POTDES%NKPTS
       EPS1(1:POTDES%NGVECTOR(K1),K1)=POT%C(1:POTDES%NGVECTOR(K1),K1)/POTDES%DATAKE(1:POTDES%NGVECTOR(K1),1,K1)
    END DO
    ! After interpolation, the screened Coulomb potential can be recovered by multiplication:
    ! W(G+q,G+q) = EPS1(G,q)*POTDES%DATAKE(G,q) (multiply)
    !-----------------------------------------------


    !-----------------------------------------------
    ! Special treatment for G+q=0:
    ! eps^-1(G=0,q=0) is needed as base point for interpolation.
    ! eps^-1(G=0,q=0) = isotropic average of POT%WPOT%HEAD(1:3,1:3,1)
    !                 = (eps^-1_xx + epx^-1_yy + eps^-1_zz)/3
    ! HEAD = lim_q->0 v chi = lim_q->0 eps^-1(q)-1

    ! k-point index of q=0
    K1=KPOINT_IN_GRID( (/ 0.0_q, 0.0_q, 0.0_q /), KPOINTS_FULL_ORIG%VKPT )
    KZERO=K1
    DO I=1,POTDES%NGVECTOR(K1)
       IF( POTDES%IGX(I,K1)==0 .AND. POTDES%IGY(I,K1)==0 .AND. POTDES%IGZ(I,K1)==0 ) EXIT
    END DO
    ! I is now plane wave index of G=0

    ! EPS1(G=0,q=0) is needed for interpolation
    EPS1(I,K1)=(POT%WPOT(K1)%HEAD(1,1,1)+POT%WPOT(K1)%HEAD(2,2,1)+POT%WPOT(K1)%HEAD(3,3,1))/3.0_q+1.0_q
    ! WSING = POT%C(G=0,q=0) = FSG*EPS^-1(G=0,q=0)
    WSING=POT%C(I,K1)
    ! After interpolation, the screened Coulomb potential at G+q=0 must be adjusted to the
    ! interpolated k-point grid:
    ! W(G+q=0)  = WSING*KMULT*KMULT (contains singularity correction)
    !-----------------------------------------------

    !-----------------------------------------------
    ! EPS1 = 1 + v chi
    ! Subtract 1 so that for large wave vectors the interpolant goes to zero. 
    ! Otherwise we may get problems when we shift k-points by some G, since some 
    ! plane waves from new k-point will be missing at original k-point. These
    ! coefficients are then set to 0, although they should be 1.
    ! I.e. now EPS1 = v chi
    !-----------------------------------------------
    EPS1=EPS1-1.0_q


    !-----------------------------------------------
    ! Calculate derivatives d EPS1/ dq
    ALLOCATE(DEPS1(POTDES%NGDIM,POTDES%NKPTS,3))
    CALL DEPS1_DK(EPS1,POTDES,KPOINTS_BASE,LATT_CUR,DEPS1)
    !-----------------------------------------------


    !-----------------------------------------------
    ! Interpolation
    DK(1)=1._q/REAL(KPOINTS_BASE%NKPX,q)
    DK(2)=1._q/REAL(KPOINTS_BASE%NKPY,q)
    DK(3)=1._q/REAL(KPOINTS_BASE%NKPZ,q)

    ALLOCATE(CPHASE(POTDES%GRID%MPLWV))
    ALLOCATE(VKPT_INT(3,NKPTS_COARSE_ADD))

    ! loop over all shifts
    DO IK=1,ABS(KMULT*KMULT*KMULT)

       ! this displacement is applied to all NKPTS_COARSE_ADD k-points found
       DISPL=DISPL_FOUND(:,IK)
       IF (IO%IU0>=0) WRITE(*,'("DISPL",3G15.5)') DISPL
       ! required k-points shifted by DISPL
       DO NK=1,NKPTS_COARSE_ADD
          VKPT_INT(:,NK)=KPOINTS_BASE%VKPT(:,NK_EQUIVALENT(NK))+DISPL
       ENDDO

#define cubic_interpolation
#ifdef cubic_interpolation
       IF (IO%IU0>=0) write(*,*) 'cubic'
       DO NK=1,NKPTS_COARSE_ADD
          NK_INTER=KPOINT_IN_GRID(VKPT_INT(:,NK),KPOINTS_INTER%VKPT)
          IF(NK_INTER>0)THEN
             ! interpolation required
             CALL INTERPOLATE_3D_EPS_SINGLE(EPS1,DEPS1,EPS1_INT(:,NK),POTDES,KPOINTS_BASE%VKPT,DK,&
                  NK_EQUIVALENT(NK),NN,VKPT_INT(:,NK) )
          ELSE
             ! interpolation not required, k-point not used
             EPS1_INT(:,NK)=0 
          END IF
       END DO
#else
       ! linear interpolation        
       IF (IO%IU0>=0) write(*,*) 'linear'
       DO NK=1,NKPTS_COARSE_ADD
          NKP=NK_EQUIVALENT(NK)
          EPS1_INT(:,NK)=EPS1(:,NKP)+&
               DEPS1(:,NKP,1)*DISPL(1)+ & 
               DEPS1(:,NKP,2)*DISPL(2)+ &
               DEPS1(:,NKP,3)*DISPL(3)
       ENDDO
#endif

       ! store orbital in R_INT for storing back
       ! plane waves have to be re-indexed now!
       DO NK=1,NKPTS_COARSE_ADD

          ! k-point in coarse grid
          NKP=NK_EQUIVALENT(NK)
          CALL SETWDES(POTDES,POTDESK,NKP)

          ! k-point in dense grid
          NK_INTER=KPOINT_IN_GRID(VKPT_INT(:,NK),KPOINTS_INTER%VKPT)
          ! Do we need this k-point in dense k-grid?
          IF (NK_INTER>=1) THEN
             CALL SETWDES(POTDES_INT,POTDESK_INT,NK_INTER)

             ! The k-point to which we have interpolated
             K=VKPT_INT(:,NK)
             ! The k-point included in dense grid.
             K_INTER=KPOINTS_INTER%VKPT(:,NK_INTER)
             !  The two may differ by some G.
             CALL SETPHASE( K-K_INTER, POTDES%GRID, CPHASE, LPHASE)

             ! FFT from coarse grid to real space
             EPS1_REAL=0
             CALL FFTWAV(POTDESK%NGVECTOR, POTDESK%NINDPW(1),EPS1_REAL(1),EPS1_INT(1:POTDESK%NGVECTOR,NK),POTDESK%GRID)
             IF(LPHASE) CALL APPLY_PHASE(POTDES%GRID,CPHASE,EPS1_REAL,EPS1_REAL)
             EPS1_REAL=EPS1_REAL/POTDESK%GRID%NPLWV

             ! FFT back from real space to dense grid
             EPS1_TRANS=0
             CALL FFTEXT(POTDESK_INT%NGVECTOR,POTDESK_INT%NINDPW(1),EPS1_REAL(1),EPS1_TRANS(1:POTDESK_INT%NGVECTOR),POTDESK_INT%GRID,.FALSE.)

             ! AFTER ALL FOURIER TRANSFORMS! add 1 to get eps^-1 
             EPS1_TRANS=EPS1_TRANS+1.0_q

             DO NG=1,POTDESK_INT%NGVECTOR
                ! muliply by DATAKE to get screened Coulomb potential W
                RESP_INT(NK_INTER)%RESPONSEFUN(NG,NG,1)=EPS1_TRANS(NG)*POTDES_INT%DATAKE(NG,1,NK_INTER)
             END DO
             ! 
          END IF
       END DO

    END DO
    !-----------------------------------------------


    !-----------------------------------------------
    ! Special treatment of G=0,q=0
    
    ! k-point index of q=0 in dense grid
    K1=KPOINT_IN_GRID( (/ 0.0_q, 0.0_q, 0.0_q /), KPOINTS_INTER%VKPT )
    ! Plane wave index of G=0
    DO I=1,POTDES_INT%NGVECTOR(K1)
       IF( POTDES_INT%IGX(I,K1)==0 .AND. POTDES_INT%IGY(I,K1)==0 .AND. POTDES_INT%IGZ(I,K1)==0 ) EXIT
    END DO

    ! store FSG*EPS1(G+q=0) scaled to interpolated grid 
    RESP_INT(K1)%RESPONSEFUN(I,I,1)=WSING*KMULT*KMULT
    ! store HEAD (stays the same)
    RESP_INT(K1)%HEAD(:,:,1)=POT%WPOT(KZERO)%HEAD(:,:,1)
    !-----------------------------------------------

    !-----------------------------------------------
    ! write W-files - but named Zxxx.tmp - for k-points in IRZ of interpolated k-mesh
    DO K1=1,KPOINTS_INTER%NKPTS
       CALL SETWDES(POTDES_INT,POTDESK_INT,K1)
       CALL WRITE_WPOT( RESP_INT(K1), POTDESK_INT, 0.0_q, K1, .FALSE., FILEBASE_="Z")
    END DO
    !-----------------------------------------------


    IF(ASSOCIATED(POT)) CALL DESTROY_WPOT_HANDLE(POT)
    DO IK=1,KPOINTS_INTER%NKPTS
       CALL DEALLOCATE_RESPONSEFUN(RESP_INT(IK))     
    END DO
    DEALLOCATE(RESP_INT)
    CALL DEALLOCWDES(POTDES,.TRUE.)
    CALL DEALLOCWDES(POTDES_INT,.TRUE.)
    DEALLOCATE(POTDES,POTDES_INT)
    DEALLOCATE(EPS1_REAL,EPS1_INT,EPS1_TRANS,EPS1,DEPS1)
    DEALLOCATE(CPHASE,VKPT_INT)
    CALL DEALLOC_GRD(POTGRID)
    CALL DEALLOC_GRD(POTGRID_INT)
    DEALLOCATE(POTGRID,POTGRID_INT)

  END SUBROUTINE WANNIER_INTERPOLATE_WPOT


  !************* SUBROUTINE HAM_DIAG **************************************
  ! 
  ! Diagonalize Hamiltonian matrix
  ! Optional: return diagonalising matrix as U, transform DHDK with U
  !
  !************************************************************************
  SUBROUTINE HAM_DIAG(NBANDS,NKPTS,NSPIN,HAM,DHDK,U)
    USE util
    IMPLICIT NONE
    INTEGER :: NBANDS,NKPTS,NSPIN
    COMPLEX(q) :: HAM(NBANDS,NBANDS,NKPTS,NSPIN)
    COMPLEX(q),OPTIONAL :: DHDK(NBANDS,NBANDS,NKPTS,NSPIN,3), U(NBANDS,NBANDS,NKPTS,NSPIN)
    !
    INTEGER :: ISP,IK,IB1,IB2,IB3,IB4,IDIR,IFAIL
    COMPLEX(q) :: CHAM(NBANDS,NBANDS),DD(NBANDS,NBANDS)
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*NBANDS)
    REAL(q)    R(NBANDS)
    REAL(q)    RWORK(3*NBANDS)

    spin:  DO ISP=1,NSPIN
       kpoint: DO IK=1,NKPTS

          CHAM=HAM(:,:,IK,ISP)
          CALL ZHEEV('V','U',NBANDS,CHAM(1,1),NBANDS,R,CWRK,LWORK*NBANDS,RWORK,IFAIL)
          IF(IFAIL/=0)THEN
             WRITE(*,*) 'HAM_DIAG: error in ZHEEV: ',IFAIL
             STOP
          END IF
          ! Diagonal HAM
          DO IB1=1,NBANDS
             HAM(IB1,IB1,IK,ISP)=R(IB1)
          END DO

          IF(PRESENT(U)) U(:,:,IK,ISP)=CHAM

          IF(PRESENT(DHDK))THEN
             ! transform DHDK
             DO IDIR=1,3
                DD=0
                DO IB1=1,NBANDS
                   DO IB2=1,NBANDS
                      DO IB3=1,NBANDS
                         DO IB4=1,NBANDS
                            DD(IB1,IB2)=DD(IB1,IB2)+CONJG(CHAM(IB3,IB1))*DHDK(IB3,IB4,IK,ISP,IDIR)*CHAM(IB4,IB2)
                         END DO
                      END DO
                   END DO
                END DO
                DHDK(:,:,IK,ISP,IDIR)=DD
             END DO
          END IF

       END DO kpoint
    END DO spin

  END SUBROUTINE HAM_DIAG


  !************************ SUBROUTINE EDDIAG_WANNIER ********************
  !
  ! Diagonalize Hamiltonian HAM and transform W to eigenfunctions
  ! for selected k-point range [NKSTART,NKSTOP]
  !
  !***********************************************************************
  SUBROUTINE EDDIAG_WANNIER(HAM, &
       W,WDES,IU0,IU6,NKSTART,NKSTOP)
    USE prec
    USE wave_high
    USE lattice
    USE mpimy
    USE mgrid
    USE nonl_high
    USE hamil_high
    USE constant
    USE jacobi
    USE scala
    USE main_mpi
    USE fock
    USE pseudo
    USE ini

    IMPLICIT NONE
    TYPE (wavespin)    W
    TYPE (wavedes)     WDES
    GDEF HAM(WDES%NB_TOT,WDES%NB_TOT,WDES%NKPTS,WDES%ISPIN)
    INTEGER IU0, IU6
    INTEGER, OPTIONAL :: NKSTART,NKSTOP     ! start k-point
    ! local
    ! work arrays for ZHEEV (blocksize times number of bands)
    INTEGER, PARAMETER :: LWORK=32
    GDEF       CWRK(LWORK*WDES%NB_TOT)
    REAL(q)    R(WDES%NB_TOT)
    REAL(q)    RWORK(3*WDES%NB_TOT)
    ! work arrays (do max of 16 strips simultaneously)
    TYPE (wavedes1)    WDES1          ! descriptor for one k-point
    TYPE (wavefuna)    WA             ! array to store wavefunction
    INTEGER :: NB_TOT, NBANDS, ISP, NK, N, I, J, NP, IFAIL, MY_NKSTART, MY_NKSTOP
    GDEF CHAM(WDES%NB_TOT,WDES%NB_TOT)


    !-----------------------------------------------------------
    NB_TOT=WDES%NB_TOT
    NBANDS=WDES%NBANDS
    CALL SETWDES(WDES,WDES1,0)

    IF (PRESENT(NKSTART)) THEN
       MY_NKSTART=NKSTART
    ELSE
       MY_NKSTART=1
    ENDIF
    IF (PRESENT(NKSTOP)) THEN
       MY_NKSTOP=NKSTOP
    ELSE
       MY_NKSTOP=WDES%NKPTS
    ENDIF
    !-----------------------------------------------------------


    !-----------------------------------------------------------
    spin:  DO ISP=1,WDES%ISPIN
       kpoint: DO NK=MY_NKSTART,MY_NKSTOP
         
          CALL SETWDES(WDES,WDES1,NK)
          WA=ELEMENTS(W, WDES1, ISP)         
          CHAM=HAM(:,:,NK,ISP)

#ifndef gammareal
          DO N=1,NB_TOT
             IF (ABS(AIMAG(CHAM(N,N)))>1E-2_q .AND. IU0>=0) THEN
                WRITE(IU0,'(A,I5,E14.3)')'WARNING in EDDIAG_WANNIER: sub space matrix is not hermitian',N,AIMAG(CHAM(N,N))
             ENDIF
             CHAM(N,N)= REAL( CHAM(N,N) ,KIND=q)
          ENDDO
#endif

          !-----------------------------------------------------------
          ! diagonalize
          IFAIL=0
#if defined(MPI)
          ! parallel versions
          ! use scaLAPACK if available in parallel version
          IF ( LscaLAPACK ) THEN
             CALL pDSSYEX_ZHEEVX(WDES%COMM, CHAM(1,1), R,  NB_TOT, WDES%NB_TOTK(NK,ISP))
             CALLMPI( M_sum_g(WDES%COMM, CHAM(1,1),NB_TOT*NB_TOT))
          ELSE
#endif
             !
             !  seriell codes
             !
#ifdef  gammareal
             CALL DSYEV &
                  ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, IFAIL)
#else
             CALL ZHEEV &
                  ('V','U',WDES%NB_TOTK(NK,ISP),CHAM(1,1),NB_TOT, &
                  R,CWRK,LWORK*NB_TOT, RWORK,  IFAIL)
#endif

#if defined(MPI)
          ENDIF
#endif
          IF (IFAIL/=0) THEN
             WRITE(0,*) 'ERROR in EDDIAG_WANNIER: call to ZHEEV/ZHEEVX/DSYEV/DSYEVX failed! '// &
                  &              'error code was ',IFAIL
             STOP
          ENDIF

          ! store eigenalues
          DO N=1,WDES%NB_TOTK(NK,ISP)
             W%CELTOT(N,NK,ISP)=R(N)
          ENDDO

          !-----------------------------------------------------------
          ! rotate wavefunctions
          IF (WDES%DO_REDIS) THEN
             CALL REDISTRIBUTE_PROJ( WA)
             CALL REDISTRIBUTE_PW( WA)
          ENDIF
          ! CHAM contains now unitary matrix
          CALL LINCOM('F',WA%CW_RED(:,:),WA%CPROJ_RED(:,:),CHAM(:,:), &
               WDES%NB_TOTK(NK,ISP),WDES%NB_TOTK(NK,ISP), & 
               WDES1%NPL_RED,WDES1%NPRO_RED,WDES1%NRPLWV_RED,WDES1%NPROD_RED,NB_TOT, &
               WA%CW_RED(:,:),WA%CPROJ_RED(:,:))
          IF (WDES%DO_REDIS) THEN
             CALL REDISTRIBUTE_PROJ( WA)
             CALL REDISTRIBUTE_PW( WA)
             DWRITE "redis ok"
          ENDIF
          !-----------------------------------------------------------

       ENDDO kpoint
    ENDDO spin
    !-----------------------------------------------------------

  END SUBROUTINE EDDIAG_WANNIER



  !************* SUBROUTINE WANNIER_INTERPOLATE_KPATH *********************
  ! 
  ! Interpolate energies - and, if desired, orbitals - along a path of
  ! k-points
  !
  !************************************************************************
  SUBROUTINE WANNIER_INTERPOLATE_KPATH(W,KPOINTS,HAMILTONIAN,GRID,LATT_CUR,T_INFO,&
       NONLR_S,NONL_S,LMDIM,P,SV,CQIJ,CDIJ,SYMM,INFO,IO)
    USE prec
    USE wave_high
    USE lattice
    USE poscar
    USE nonl_high
    USE base
    USE pseudo
    USE constant
    USE choleski
    USE subrot
    USE interpolate
    USE mlwf
    IMPLICIT NONE
    TYPE (wavespin), TARGET :: W
    TYPE (kpoints_struct) KPOINTS
    TYPE (ham_handle)   HAMILTONIAN
    TYPE (grid_3d)      GRID
    TYPE (latt)         LATT_CUR
    TYPE (type_info)    T_INFO
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct)  NONL_S
    INTEGER LMDIM
    TYPE (potcar)      P(:)
    RGRID           SV(DIMREAL(GRID%MPLWV),W%WDES%NCDIJ)
    OVERLAP         CQIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    OVERLAP         CDIJ (LMDIM,LMDIM,W%WDES%NIONS,W%WDES%NCDIJ)
    TYPE (symmetry) SYMM
    TYPE (info_struct) INFO
    TYPE (in_struct)   IO
    ! local
    INTEGER :: NPATH,NPATH_MAX,NPTS_PATH,NPTS_TOTAL,I,J,IK,NK,ISPINOR,IX
    REAL(q) :: DELTA_PATH(3),K(3),K0(3),SK0(3),DK(3),EXHF,DISPL(3),DIST,XX
    REAL(q) :: XSTART,XSTOP, YMIN, YMAX
    REAL(q), ALLOCATABLE :: XPATH(:)
    REAL(q), ALLOCATABLE :: KVERT(:,:),KPATH(:,:)
    TYPE (wavefun1) :: W1
    TYPE (wavedes1) :: WDES0,WDES1
    COMPLEX(q), ALLOCATABLE :: HAM_KPATH(:,:,:,:)
    CHARACTER(256) :: STR

    !-----------------------------------------------------------
    ! Path(i) goes from Vertex(2i-1) to Vertex(2i),
    ! including the starting point, but not including the end point

    ! read kpath from file
    OPEN(UNIT=199,FILE='KPOINTS.INT',FORM='FORMATTED',ACCESS='SEQUENTIAL')
    READ(199,*) STR       ! ignore first line
    READ(199,*) NPTS_PATH ! no.of points per path
    READ(199,*) STR       ! should be "Line"
    READ(199,*) STR       ! "reciprocal" or "direct"
    IF (STR(1:1)=='D'.OR.STR(1:1)=='d')THEN
       WRITE(*,*) 'WANNIER_INTERPOLATE_KPATH: Direct coordinates not implemented'
       STOP
    END IF
    ! max. no. of Paths
    NPATH_MAX=20
    ALLOCATE(KVERT(3,2*NPATH_MAX),KPATH(3,NPTS_PATH*NPATH_MAX))
    J=0
    DO
       READ(199,'(A)',END=1001) STR
       STR=ADJUSTL(STR)
       IF(LEN(TRIM(STR))==0) CYCLE
       IF(STR(1:1)=='!') CYCLE
       J=J+1
       READ(STR,*) KVERT(:,J)
    END DO
1001 NPATH=J/2
    NPTS_TOTAL=NPTS_PATH*NPATH
    CLOSE(199)

    ALLOCATE(XPATH(NPATH+1))

    ! DELTA_PATH(i) is the spacing on Path(i)
    DO I=1,NPATH
       DELTA_PATH=(KVERT(:,2*I)-KVERT(:,2*I-1))/(NPTS_PATH-1)
       DO J=1,NPTS_PATH
          KPATH(:,(I-1)*NPTS_PATH+J)=KVERT(:,2*I-1)+DELTA_PATH*(J-1)
       END DO
    END DO
    !-----------------------------------------------------------


    !-----------------------------------------------------------
    DK(1)=1._q/REAL(KPOINTS%NKPX,q)
    DK(2)=1._q/REAL(KPOINTS%NKPY,q)
    DK(3)=1._q/REAL(KPOINTS%NKPZ,q)   

    ! plot coordinate
    XX=0
    XSTART=XX

    !intialize YMIN and YMAX
    YMIN =  1E5_q
    YMAX = -1E5_q
    ! set wave descriptor for gamma point
    CALL SETWDES(W%WDES,WDES0,NKPTS_COARSE_BZ+1)

    ALLOCATE(HAM_KPATH(W%WDES%NB_TOT,W%WDES%NB_TOT,NKPTS_COARSE_BZ+1,W%WDES%ISPIN))
    HAM_KPATH(:,:,1:NKPTS_COARSE_BZ,:)=HAM

    !the master node opens the file 
    IF(IO%IU0>=0) OPEN(UNIT=199,FILE='bands_int.dat',FORM='FORMATTED',STATUS='REPLACE')

    IX = 0 
    DO IK=1,NPTS_TOTAL
       IF ( IK == 1 ) THEN
          IX = IX + 1  
          XPATH(IX)=XX
       ENDIF 
       ! the k-point we want
       K=KPATH(:,IK)
       ! find nearest neighbour of it
       NK=FIND_NN(W%WDES%VKPT(:,1:NKPTS_COARSE_BZ),K,LATT_CUR%B,SK0)
       ! SK0 is the nearest neighbour of K in the k-point mesh repeated in all BZ.
       ! Thus, SK0 is VKPT(:,NK)+integer shift, i.e. possibly shifted into another BZ.
       K0=SK0
       W%WDES%VKPT(:,NKPTS_COARSE_BZ+1)=K

       ! ORBITAL INTERPOLATION
       IF(LINTPOL_KPATH_ORB)THEN
          ! We interpolate from the orbital at k-point #NK, but using its possibly G-shifted equivalent SKO,
          ! to the k-point K.
          CALL INTERPOLATE_3D_ORBITALS_SINGLE(W,RPHI,W%CW(:,:,NKPTS_COARSE_BZ+1,:),W%WDES%VKPT,DK,NK,NN,K,SK0=K0)
          
          ! set wavedescriptor for NK
          CALL SETWDES(W%WDES,WDES1,NK)
          CALL NEWWAV(W1,WDES1,.TRUE.)
          DO I=1,W%WDES%NBANDS
             DO J=1,W%WDES%ISPIN
                ! copy in layout of NK (interpolation uses this layout)
                W1%CW=W%CW(:,I,NKPTS_COARSE_BZ+1,J)
                CALL FFTWAV_W1(W1)
                W1%CR=W1%CR/WDES1%GRID%NPLWV
                ! back-FFT into layout of NKPTS_COARSE_BZ+1 (should be gamma point)
                DO ISPINOR=0,WDES1%NRSPINORS-1
                   CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),W1%CR(1+ISPINOR*WDES1%GRID%MPLWV),W1%CW(1+ISPINOR*WDES0%NGVECTOR),WDES1%GRID,.FALSE.)
                ENDDO
                W%CW(:,I,NKPTS_COARSE_BZ+1,J)=W1%CW
             END DO
          END DO
          CALL DELWAV(W1,.TRUE.)
          
          CALL SET_DATAKE(W%WDES, LATT_CUR%B)
          IF (NONLR_S%LREAL) THEN
             CALL RSPHER(GRID,NONLR_S,LATT_CUR)
          ELSE
             CALL SPHER(GRID,NONL_S,P,W%WDES,LATT_CUR, 1)
          ENDIF
          W%WDES%NB_TOTK(:,:)=MLWF_num_wann
          CALL PROALL (GRID,LATT_CUR,NONLR_S,NONL_S,W)
          CALL ORTHCH(W%WDES,W, W%WDES%LOVERL, LMDIM, CQIJ,NKSTART=NKPTS_COARSE_BZ+1)
       END IF

       ! HAMILTONIAN INTERPOLATION
       CALL FOURIER_INTERPOL(KPOINTS,KPATH(:,IK:IK),0,HAM_KPATH(:,:,1:NKPTS_COARSE_BZ,:),HAM_KPATH(:,:,NKPTS_COARSE_BZ+1:NKPTS_COARSE_BZ+1,:),LATT_CUR,IO)
       W%WDES%NB_TOTK(:,:)=MLWF_num_wann
       CALL EDDIAG_WANNIER(HAM_KPATH,W,W%WDES,IO%IU0,IO%IU6,NKSTART=NKPTS_COARSE_BZ+1,NKSTOP=NKPTS_COARSE_BZ+1)
       IF (IO%IU0>=0) THEN
           WRITE(*,'("H",I2,3F8.4,50F8.4)') IK,K,XX,REAL(W%CELTOT(1:MLWF_num_wann,NKPTS_COARSE_BZ+1,1),q)
           WRITE(199,'(F10.6,50F14.6)')XX,REAL(W%CELTOT(1:MLWF_num_wann,NKPTS_COARSE_BZ+1,1),q)
           !find minimum and maximum energy value of interpolated bands 
           DO J=1, MLWF_num_wann 
              YMIN = MIN( YMIN, REAL(W%CELTOT(J,NKPTS_COARSE_BZ+1,1),q))
              YMAX = MAX( YMAX, REAL(W%CELTOT(J,NKPTS_COARSE_BZ+1,1),q))
           ENDDO
           IF ( MOD(IK,NPTS_PATH)==0 ) THEN
              IX = IX + 1 
 !             WRITE(*,*)IK,NPATH,NPTS_TOTAL,IX
              XPATH(IX)=XX
           ENDIF 
       ENDIF 


       IF(LINTPOL_KPATH_ORB)THEN
          ! expectation value
          CALL EDDIAG(HAMILTONIAN,&
               GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
               LMDIM,CDIJ,CQIJ,1,SV,T_INFO,P,IO%IU0,EXHF, & 
               NKSTART=NKPTS_COARSE_BZ+1,NKSTOP=NKPTS_COARSE_BZ+1)
          IF (IO%IU0>=0) WRITE(*,'("E",I2,3F8.4,50F8.4)') IK,K,XX,REAL(W%CELTOT(1:MLWF_num_wann,NKPTS_COARSE_BZ+1,1),q)
          
          ! diagonalization
          CALL EDDIAG(HAMILTONIAN,&
               GRID,LATT_CUR,NONLR_S,NONL_S,W,W%WDES,SYMM, &
               LMDIM,CDIJ,CQIJ,2,SV,T_INFO,P,IO%IU0,EXHF, & 
               NKSTART=NKPTS_COARSE_BZ+1,NKSTOP=NKPTS_COARSE_BZ+1)
          IF (IO%IU0>=0) WRITE(*,'("D",I2,3F8.4,50F8.4)') IK,K,XX,REAL(W%CELTOT(1:MLWF_num_wann,NKPTS_COARSE_BZ+1,1),q)
       END IF
          
       ! Calculate length along path in cartesian coordinates
       IF(IK<NPTS_TOTAL)THEN
          DISPL=KPATH(:,IK+1)-KPATH(:,IK)
          CALL DIRKAR(1,DISPL,LATT_CUR%B)
          XX=XX+SQRT(SUM(DISPL**2))*TPI
       END IF

       IF ( IK==NPTS_TOTAL ) XSTOP = XX 
    END DO

    !close bands_int.dat
    IF(IO%IU0>=0) CLOSE(199)

    !finally make a nice gnuplot plot file 
    IF ( IO%IU0>=0 ) THEN
      
       !increase values of YMIN an YMAX to 10% of energy range
       XX=ABS(YMAX-YMIN)*0.1_q
       YMIN=YMIN-XX
       YMAX=YMAX+XX
      
       OPEN(UNIT=199,FILE='bands_int.gnu',FORM='FORMATTED',STATUS='REPLACE')
       !write header
       WRITE(199,'(A)')'set nokey'      
       WRITE(199,'("set xrange [",F12.6,":",F12.6,"]")')XSTART,XSTOP      
       WRITE(199,'("set yrange [",F12.6,":",F12.6,"]")')YMIN,YMAX
      
       DO IK = 1,IX
         WRITE(199,'("set arrow from ",F12.6,", ",F12.6," to ",F12.6,", ",F12.6," nohead ")')&
         XPATH(IK),YMIN,XPATH(IK),YMAX
       ENDDO 
       WRITE(199,'("set xtics (",A)')'\'
       IF ( IX > 1 ) THEN
          DO IK = 1, IX-1
             WRITE(199,'(A,F12.6,A)')'"  "',XPATH(IK),',\'
          ENDDO
       ENDIF 
       WRITE(199,'(A,F12.6,A)')'"  "',XPATH(IX),')'
           

       IF ( MLWF_num_wann > 1 ) THEN
          WRITE(199,'(A)')' plot "bands_int.dat" u ($1):($2) w l lc rgb "red" ,\'  
          DO I = 1, MLWF_num_wann-1
             WRITE(STR,'(I4)')I+1
             STR=ADJUSTL(STR)
             WRITE(199,'(A)')' "bands_int.dat" u ($1):($'//TRIM(STR)//') w l lc rgb "red" ,\'
          ENDDO
          WRITE(STR,'(I4)')MLWF_num_wann+1
          STR=ADJUSTL(STR) 
          WRITE(199,'(A)')' "bands_int.dat" u ($1):($'//TRIM(STR)//') w l lc rgb "red"'
       ELSE
          WRITE(199,'(A)')' plot "bands_int.dat" u ($1):($2) w l lc rgb "red"'  
       ENDIF 
       WRITE(199,'("pause -1 ")')
       CLOSE(199)
    ENDIF 

    DEALLOCATE(HAM_KPATH,KVERT,KPATH)
    DEALLOCATE(XPATH)

  END SUBROUTINE WANNIER_INTERPOLATE_KPATH



  !************* SUBROUTINE FIND_NN ***************************************
  ! 
  ! Find nearest neighbouring k-point. Returns the index of this k-point
  ! and the value SK0, which may belong to a different primitive cell.
  ! NOTE: checks only the first neighbouring primitive cells.
  !
  !************************************************************************
  FUNCTION FIND_NN(VKPT,K,B,SK0)
    USE lattice
    IMPLICIT NONE
    REAL(q) :: VKPT(:,:),K(3),B(3,3),SK0(3)
    INTEGER :: J,FIND_NN,IKXP,IKYP,IKZP
    REAL(q) :: DIST,DIST_FOUND,DISPL(3)

    DIST_FOUND=1E6
    DO J=1,SIZE(VKPT,2)
       DO IKXP=-1,1
          DO IKYP=-1,1
             DO IKZP=-1,1
                DISPL(1)=VKPT(1,J)+IKXP-K(1)
                DISPL(2)=VKPT(2,J)+IKYP-K(2)
                DISPL(3)=VKPT(3,J)+IKZP-K(3)
                ! calculate distance in cartesian coordinates!
                CALL DIRKAR(1, DISPL(1), B)
                DIST=SUM(DISPL**2)
                IF (DIST<DIST_FOUND) THEN
                   DIST_FOUND=DIST
                   FIND_NN=J
                   SK0(1)=VKPT(1,J)+IKXP
                   SK0(2)=VKPT(2,J)+IKYP
                   SK0(3)=VKPT(3,J)+IKZP
                ENDIF
             ENDDO
          END DO
       END DO
    END DO

  END FUNCTION FIND_NN



  !************* SUBROUTINE KPOINTS_SUPERGRID *****************************
  ! 
  ! create k-points grid suitable for Fourier interpolation
  ! fastest index kx, then ky, then kz
  ! kx,ky,kz in (-0.5,0.5] but ordered starting with 0
  ! i.e. [0,0.5] + (-0.5,0)
  !
  !************************************************************************
  SUBROUTINE KPOINTS_SUPERGRID(M1,M2,M3,VKPT)
    IMPLICIT NONE
    INTEGER :: M1,M2,M3
    REAL(q) :: VKPT(3,M1*M2*M3)
    !
    REAL(q) :: TINY=1.E-6_q,D(3)
    INTEGER :: I,I1,I2,I3

    I=0
    D(1)=1.0_q/REAL(M1,q)
    D(2)=1.0_q/REAL(M2,q)
    D(3)=1.0_q/REAL(M3,q)
    DO I3=0,M3-1
       DO I2=0,M2-1
          DO I1=0,M1-1
             I=I+1
             ! map values into interval (-0.5,0.5]
             ! 1. map interval (-0.5,0.5] to (0,1]
             ! 2. then apply MOD function to x
             ! 3. map back to (-0.5,0.5]
             ! MOD( x+ 0.5-eps, 1.0 ) - (0.5-eps)
             VKPT(:,I)=(/ MOD(D(1)*I1+0.5_q-TINY,1.0_q)-(0.5_q-TINY),&
                  MOD(D(2)*I2+0.5_q-TINY,1.0_q)-(0.5_q-TINY),&
                  MOD(D(3)*I3+0.5_q-TINY,1.0_q)-(0.5_q-TINY) &
                  /)
          END DO
       END DO
    END DO
  END SUBROUTINE KPOINTS_SUPERGRID


  !************* SUBROUTINE KPOINT_IN_GRID ********************************
  ! 
  ! Search for 3-vector in array (not limited to kpoints_struct)
  !
  !************************************************************************
  FUNCTION KPOINT_IN_GRID(VKPT,GRID)
    INTEGER :: KPOINT_IN_GRID
    REAL(q) :: VKPT(3), GRID(:,:)
    REAL(q),PARAMETER :: TINY=1E-8_q
    ! local
    INTEGER NK
    DO NK=1,SIZE(GRID,2)
       IF ( ALL( ABS(MOD(VKPT(:)-GRID(:,NK)+10.5_q,1._q)-0.5_q)<TINY ) ) EXIT
    ENDDO

    IF (NK>SIZE(GRID,2)) THEN
       ! no kpoint found, set nk=-1
       NK=-1
    ENDIF

    KPOINT_IN_GRID=NK
  END FUNCTION KPOINT_IN_GRID



  !************* SUBROUTINE NN_KPOINTS ************************************
  ! 
  ! Create index array of nearest neighbours in X, Y, Z:
  ! NN( 1=forw/2=back, 1=x/2=y/3=z, k-point )
  !
  !************************************************************************
  SUBROUTINE NN_KPOINTS(KPOINTS,NN)
    USE mkpoints
    USE full_kpoints
    IMPLICIT NONE
    TYPE(kpoints_struct) :: KPOINTS
    INTEGER :: NN(2,3,KPOINTS%NKPTS) ! forw./backw, x/y/z, nk
    INTEGER :: IK,NK,ISGN,IDIR,FACT
    REAL(q) :: DK(3),K0(3),K(3)
    !
    DK(1)=1._q/REAL(KPOINTS%NKPX,KIND=q)
    DK(2)=1._q/REAL(KPOINTS%NKPY,KIND=q)
    DK(3)=1._q/REAL(KPOINTS%NKPZ,KIND=q)

    DO IK=1,KPOINTS%NKPTS
       K0(:)=KPOINTS%VKPT(:,IK)
       DO IDIR=1,3
          DO ISGN=1,2
             FACT=-3+ISGN*2
             K(:)=K0(:)
             K(IDIR)=K0(IDIR)+FACT*DK(IDIR)
             NK=KPOINT_IN_GRID(K,KPOINTS%VKPT)
             NN(ISGN,IDIR,IK)=NK
          END DO
       END DO
    END DO
  END SUBROUTINE NN_KPOINTS



  !************* SUBROUTINE INTERPOLATE_3D_ORBITALS_VECTOR*****************
  ! 
  ! Cubic 3D-interpolation of orbitals on a _vector_ of k-points VKPT_ADD.
  ! VKPT_ADD must lie in a cube whose corners are points in the coarse k-grid.
  !
  ! W                                            wavefunction containing orbitals
  ! RPHI(plane wave,band,k-point,spin,direction) 1st derivative of orbitals wrt. k
  ! WINT(plane wave,band,k-point,spin)           interpolated orbitals
  ! VKPT                                         coarse k-points
  ! DK                                           coarse k-point spacing
  ! NN                                           nearest neighbour list of coarse grid
  ! NK_EQUIV                                     list of indices of base points (cube corners) 
  !                                              used for interpolation
  ! VKPT_ADD                                     k-points you want to interpolate to
  !
  !************************************************************************
  SUBROUTINE INTERPOLATE_3D_ORBITALS_VECTOR(W,RPHI,WINT,VKPT,DK,NN,NK_EQUIV,VKPT_ADD)
    USE mkpoints
    USE wave
    USE wave_high
    USE mlwf
    USE interpolate
    USE constant
    IMPLICIT NONE
    TYPE(wavespin) :: W
    COMPLEX(qs) :: RPHI(:,:,:,:,:)
    COMPLEX(q) :: WINT(:,:,:,:)
    REAL(q) :: VKPT(:,:), DK(3)
    INTEGER :: NN(:,:,:),NK_EQUIV(:)
    REAL(q) :: VKPT_ADD(:,:)
    !
    TYPE (wavedes1) WDESK(2,2,2),WDES0
    TYPE (wavefun1) WK(2,2,2),WDKX(2,2,2),WDKY(2,2,2),WDKZ(2,2,2),&
         W1(2,2),W2(2),WX1(2,2),WY1(2,2),WZ1(2,2),WX2(2),WY2(2),WZ2(2)
    INTEGER :: NK,NK_BASE,NKPTS_ADD,CUBE(2,2,2),IC1,IC2,IC3,ISGN(3),IDIR(3),ISP,IB,ISPINOR
    REAL(q) :: DISPL(3),K0(3),K(3),KP(3),G3(4),G1(2)
    COMPLEX(q), ALLOCATABLE :: CPHASE(:) 
    LOGICAL LPHASE

    NKPTS_ADD=SIZE(VKPT_ADD,2)

    DO NK=1,NKPTS_ADD

       ! base k-point
       NK_BASE=NK_EQUIV(NK)
       K0=VKPT(:,NK_EQUIV(NK))

       ! cube orientation
       ISGN=SIGN(1.0_q,VKPT_ADD(:,NK)-K0) ! -1/1
       IDIR=(3+ISGN)/2 ! 1/2
       CALL CUBE_KPOINTS(NN,NK_BASE,IDIR,CUBE)
       CALL SETWDES(W%WDES,WDES0,NK_BASE)

       ! auxiliary wave function arrays for interpolation
       DO IC1=1,2
          DO IC2=1,2
             CALL NEWWAV(W1(IC1,IC2),WDES0,.FALSE.)
             CALL NEWWAV(WX1(IC1,IC2),WDES0,.FALSE.)
             CALL NEWWAV(WY1(IC1,IC2),WDES0,.FALSE.)
             CALL NEWWAV(WZ1(IC1,IC2),WDES0,.FALSE.)
          END DO
          CALL NEWWAV(W2(IC1),WDES0,.FALSE.)
          CALL NEWWAV(WX2(IC1),WDES0,.FALSE.)
          CALL NEWWAV(WY2(IC1),WDES0,.FALSE.)
          CALL NEWWAV(WZ2(IC1),WDES0,.FALSE.)
       END DO

       spin: DO ISP=1,W%WDES%ISPIN
          band: DO IB=1,W%WDES%NBANDS

             cube1: DO IC1=1,2
                cube2: DO IC2=1,2
                   cube3: DO IC3=1,2
                      CALL SETWDES(W%WDES,WDESK(IC1,IC2,IC3),CUBE(IC1,IC2,IC3))
                      CALL NEWWAV(WK(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                      CALL NEWWAV(WDKX(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                      CALL NEWWAV(WDKY(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                      CALL NEWWAV(WDKZ(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)

                      CALL W1_COPY(ELEMENT(W,WDESK(IC1,IC2,IC3),IB,ISP),WK(IC1,IC2,IC3))
                      WDKX(IC1,IC2,IC3)%CW=TPI*DK(1)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,1)*ISGN(1)
                      WDKY(IC1,IC2,IC3)%CW=TPI*DK(2)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,2)*ISGN(2)
                      WDKZ(IC1,IC2,IC3)%CW=TPI*DK(3)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,3)*ISGN(3)

                      K(1)=K0(1)+DK(1)*(IC1-1)*ISGN(1)
                      K(2)=K0(2)+DK(2)*(IC2-1)*ISGN(2)
                      K(3)=K0(3)+DK(3)*(IC3-1)*ISGN(3)
                      KP=VKPT(:,CUBE(IC1,IC2,IC3) )

                      ALLOCATE(CPHASE(WDESK(IC1,IC2,IC3)%GRID%MPLWV))
                      ! transform from KP to KK
                      CALL SETPHASE( KP(:)-K(:),WDESK(IC1,IC2,IC3)%GRID,CPHASE,LPHASE)

                      ! phase factor on wave function
                      CALL FFTWAV_W1(WK(IC1,IC2,IC3))
                      IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WK(IC1,IC2,IC3)%CR,WK(IC1,IC2,IC3)%CR)
                      WK(IC1,IC2,IC3)%CR=WK(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                      DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                         CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WK(IC1,IC2,IC3)%CR(1+ISPINOR*WDESK(IC1,IC2,IC3)%GRID%MPLWV),WK(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDESK(IC1,IC2,IC3)%GRID,.FALSE.)
                      ENDDO

                      ! phase factor on derivative X
                      CALL FFTWAV_W1(WDKX(IC1,IC2,IC3))
                      IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKX(IC1,IC2,IC3)%CR,WDKX(IC1,IC2,IC3)%CR)
                      WDKX(IC1,IC2,IC3)%CR=WDKX(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                      DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                         CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKX(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKX(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                      ENDDO

                      ! phase factor on derivative Y
                      CALL FFTWAV_W1(WDKY(IC1,IC2,IC3))
                      IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKY(IC1,IC2,IC3)%CR,WDKY(IC1,IC2,IC3)%CR)
                      WDKY(IC1,IC2,IC3)%CR=WDKY(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                      DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                         CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKY(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKY(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                      ENDDO

                      ! phase factor on derivative Z
                      CALL FFTWAV_W1(WDKZ(IC1,IC2,IC3))
                      IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKZ(IC1,IC2,IC3)%CR,WDKZ(IC1,IC2,IC3)%CR)
                      WDKZ(IC1,IC2,IC3)%CR=WDKZ(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                      DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                         CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKZ(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKZ(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                      ENDDO

                      DEALLOCATE(CPHASE)

                   END DO cube3
                END DO cube2
             END DO cube1

             DISPL(1)=ABS(VKPT_ADD(1,NK)-K0(1))/DK(1) ! in [0,1]
             DISPL(2)=ABS(VKPT_ADD(2,NK)-K0(2))/DK(2) ! in [0,1]
             DISPL(3)=ABS(VKPT_ADD(3,NK)-K0(3))/DK(3) ! in [0,1]

             ! interpolate in x direction on 4 edges
             CALL INTER_1D_CUBIC_WEIGHTS(DISPL(1),G3)
             CALL INTER_1D_LINEAR_WEIGHTS(DISPL(1),G1)
             ! edge 1,1
             CALL INTER_1D_CUBIC(G3,WK(1,1,1)%CW,WK(2,1,1)%CW,WDKX(1,1,1)%CW,WDKX(2,1,1)%CW,W1(1,1)%CW)
             CALL INTER_1D_LINEAR(G1,WDKY(1,1,1)%CW,WDKY(2,1,1)%CW,WY1(1,1)%CW)
             CALL INTER_1D_LINEAR(G1,WDKZ(1,1,1)%CW,WDKZ(2,1,1)%CW,WZ1(1,1)%CW)
             ! edge 1,2
             CALL INTER_1D_CUBIC(G3,WK(1,1,2)%CW,WK(2,1,2)%CW,WDKX(1,1,2)%CW,WDKX(2,1,2)%CW,W1(1,2)%CW)
             CALL INTER_1D_LINEAR(G1,WDKY(1,1,2)%CW,WDKY(2,1,2)%CW,WY1(1,2)%CW)
             CALL INTER_1D_LINEAR(G1,WDKZ(1,1,2)%CW,WDKZ(2,1,2)%CW,WZ1(1,2)%CW)
             ! edge 2,1
             CALL INTER_1D_CUBIC(G3,WK(1,2,1)%CW,WK(2,2,1)%CW,WDKX(1,2,1)%CW,WDKX(2,2,1)%CW,W1(2,1)%CW)
             CALL INTER_1D_LINEAR(G1,WDKY(1,2,1)%CW,WDKY(2,2,1)%CW,WY1(2,1)%CW)
             CALL INTER_1D_LINEAR(G1,WDKZ(1,2,1)%CW,WDKZ(2,2,1)%CW,WZ1(2,1)%CW)
             ! edge 2,2
             CALL INTER_1D_CUBIC(G3,WK(1,2,2)%CW,WK(2,2,2)%CW,WDKX(1,2,2)%CW,WDKX(2,2,2)%CW,W1(2,2)%CW)
             CALL INTER_1D_LINEAR(G1,WDKY(1,2,2)%CW,WDKY(2,2,2)%CW,WY1(2,2)%CW)
             CALL INTER_1D_LINEAR(G1,WDKZ(1,2,2)%CW,WDKZ(2,2,2)%CW,WZ1(2,2)%CW)

             ! interpolate in y direction on 2 edges
             CALL INTER_1D_CUBIC_WEIGHTS(DISPL(2),G3)
             CALL INTER_1D_LINEAR_WEIGHTS(DISPL(2),G1)
             ! edge x,1
             CALL INTER_1D_CUBIC(G3,W1(1,1)%CW,W1(2,1)%CW,WY1(1,1)%CW,WY1(2,1)%CW,W2(1)%CW)
             CALL INTER_1D_LINEAR(G1,WZ1(1,1)%CW,WZ1(2,1)%CW,WZ2(1)%CW)
             ! edge x,2
             CALL INTER_1D_CUBIC(G3,W1(1,2)%CW,W1(2,2)%CW,WY1(1,2)%CW,WY1(2,2)%CW,W2(2)%CW)
             CALL INTER_1D_LINEAR(G1,WZ1(1,2)%CW,WZ1(2,2)%CW,WZ2(2)%CW)

             ! interpolate in z direction on 1 line
             CALL INTER_1D_CUBIC_WEIGHTS(DISPL(3),G3)
             CALL INTER_1D_CUBIC(G3,W2(1)%CW,W2(2)%CW,WZ2(1)%CW,WZ2(2)%CW,W1(1,1)%CW)
             WINT(:,IB,NK,ISP)=W1(1,1)%CW

             DO IC1=1,2
                DO IC2=1,2
                   DO IC3=1,2
                      CALL DELWAV( WK(IC1,IC2,IC3),  .TRUE. )
                      CALL DELWAV( WDKX(IC1,IC2,IC3),.TRUE. )
                      CALL DELWAV( WDKY(IC1,IC2,IC3),.TRUE. )
                      CALL DELWAV( WDKZ(IC1,IC2,IC3),.TRUE. )
                   END DO
                END DO
             END DO

          END DO band
       END DO spin

       DO IC1=1,2
          DO IC2=1,2
             CALL DELWAV(W1(IC1,IC2),.FALSE.)
             CALL DELWAV(WX1(IC1,IC2),.FALSE.)
             CALL DELWAV(WY1(IC1,IC2),.FALSE.)
             CALL DELWAV(WZ1(IC1,IC2),.FALSE.)
          END DO
          CALL DELWAV(W2(IC1),.FALSE.)
          CALL DELWAV(WX2(IC1),.FALSE.)
          CALL DELWAV(WY2(IC1),.FALSE.)
          CALL DELWAV(WZ2(IC1),.FALSE.)
       END DO

    END DO

  END SUBROUTINE INTERPOLATE_3D_ORBITALS_VECTOR



  !************* SUBROUTINE INTERPOLATE_3D_ORBITALS_SINGLE ****************
  ! 
  ! Cubic 3D-interpolation of orbitals on a _single_ of k-point VKPT_ADD.
  ! VKPT_ADD must lie in a cube whose corners are points in the coarse k-grid.
  !
  ! W                                            wavefunction containing orbitals
  ! RPHI(plane wave,band,k-point,spin,direction) 1st derivative of orbitals wrt. k
  ! WINT(plane wave,band,k-point,spin)           interpolated orbitals
  ! VKPT                                         coarse k-points
  ! DK                                           coarse k-point spacing
  ! NN                                           nearest neighbour list of coarse grid
  ! NK_BASE                                      index of base point (cube corner) 
  !                                              used for interpolation
  ! VKPT_ADD                                     k-point you want to interpolate to
  ! SK0      optional: k-coordinates of the base corner NK_BASE
  !          This is necessary if your k-point VKPT_ADD is in another Primitive cell
  !          than spanned by VKPT. SK0 must be VKPT(:,NK_BASE) + integer shift 
  !          (by reciprocal lattice vector)
  !
  !************************************************************************
  SUBROUTINE INTERPOLATE_3D_ORBITALS_SINGLE(W,RPHI,WINT,VKPT,DK,NK_BASE,NN,VKPT_ADD,SK0)
    USE mkpoints
    USE wave
    USE wave_high
    USE mlwf
    USE interpolate
    USE constant
    IMPLICIT NONE
    TYPE(wavespin) :: W
    COMPLEX(qs) :: RPHI(:,:,:,:,:)
    COMPLEX(q) :: WINT(:,:,:)
    REAL(q) :: VKPT(:,:), DK(3)
    INTEGER :: NK_BASE,NN(:,:,:)
    REAL(q) :: VKPT_ADD(3)
    REAL(q), OPTIONAL :: SK0(3) ! shifted base k-point
    !
    TYPE (wavedes1) WDESK(2,2,2),WDES0
    TYPE (wavefun1) WK(2,2,2),WDKX(2,2,2),WDKY(2,2,2),WDKZ(2,2,2),&
         W1(2,2),W2(2),WX1(2,2),WY1(2,2),WZ1(2,2),WX2(2),WY2(2),WZ2(2)
    INTEGER :: NKPTS_ADD,CUBE(2,2,2),IC1,IC2,IC3,ISGN(3),IDIR(3),ISP,IB,ISPINOR
    REAL(q) :: DISPL(3),K0(3),K(3),KP(3),G3(4),G1(2)
    COMPLEX(q), ALLOCATABLE :: CPHASE(:) 
    LOGICAL LPHASE
    REAL(q) :: TINY=1E-12_q

    ! base k-point
    K0=VKPT(:,NK_BASE)
    IF(PRESENT(SK0))THEN
       IF ( .NOT. ALL( ABS(MOD(K0-SK0+10.5_q,1._q)-0.5_q)<TINY ) )THEN
          WRITE(*,*) 'INTERPOLATE_3D_ORBITALS_SINGLE: SK0 and NK_BASE not compatible',K0,SK0
          STOP
       END IF
       K0=SK0
    END IF

    ! cube orientation
    ISGN=SIGN(1.0_q,VKPT_ADD-K0) ! -1/1
    IDIR=(3+ISGN)/2 ! 1/2
    CALL CUBE_KPOINTS(NN,NK_BASE,IDIR,CUBE)
    CALL SETWDES(W%WDES,WDES0,NK_BASE)

    ! auxiliary wave function arrays for interpolation
    DO IC1=1,2
       DO IC2=1,2
          CALL NEWWAV(W1(IC1,IC2),WDES0,.FALSE.)
          CALL NEWWAV(WX1(IC1,IC2),WDES0,.FALSE.)
          CALL NEWWAV(WY1(IC1,IC2),WDES0,.FALSE.)
          CALL NEWWAV(WZ1(IC1,IC2),WDES0,.FALSE.)
       END DO
       CALL NEWWAV(W2(IC1),WDES0,.FALSE.)
       CALL NEWWAV(WX2(IC1),WDES0,.FALSE.)
       CALL NEWWAV(WY2(IC1),WDES0,.FALSE.)
       CALL NEWWAV(WZ2(IC1),WDES0,.FALSE.)
    END DO

    spin: DO ISP=1,W%WDES%ISPIN
       band: DO IB=1,W%WDES%NBANDS

          cube1: DO IC1=1,2
             cube2: DO IC2=1,2
                cube3: DO IC3=1,2
                   CALL SETWDES(W%WDES,WDESK(IC1,IC2,IC3),CUBE(IC1,IC2,IC3))
                   CALL NEWWAV(WK(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                   CALL NEWWAV(WDKX(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                   CALL NEWWAV(WDKY(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)
                   CALL NEWWAV(WDKZ(IC1,IC2,IC3),WDESK(IC1,IC2,IC3),.TRUE.)

                   CALL W1_COPY(ELEMENT(W,WDESK(IC1,IC2,IC3),IB,ISP),WK(IC1,IC2,IC3))
                   WDKX(IC1,IC2,IC3)%CW=TPI*DK(1)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,1)*ISGN(1)
                   WDKY(IC1,IC2,IC3)%CW=TPI*DK(2)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,2)*ISGN(2)
                   WDKZ(IC1,IC2,IC3)%CW=TPI*DK(3)*RPHI(:,IB,CUBE(IC1,IC2,IC3),ISP,3)*ISGN(3)

                   K(1)=K0(1)+DK(1)*(IC1-1)*ISGN(1)
                   K(2)=K0(2)+DK(2)*(IC2-1)*ISGN(2)
                   K(3)=K0(3)+DK(3)*(IC3-1)*ISGN(3)
                   KP=VKPT(:,CUBE(IC1,IC2,IC3) )

                   ALLOCATE(CPHASE(WDESK(IC1,IC2,IC3)%GRID%MPLWV))
                   ! transform from KP to KK
                   CALL SETPHASE( KP(:)-K(:),WDESK(IC1,IC2,IC3)%GRID,CPHASE,LPHASE)

                   ! phase factor on wave function
                   CALL FFTWAV_W1(WK(IC1,IC2,IC3))

                   IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WK(IC1,IC2,IC3)%CR,WK(IC1,IC2,IC3)%CR)
                   WK(IC1,IC2,IC3)%CR=WK(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                   DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                      CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WK(IC1,IC2,IC3)%CR(1+ISPINOR*WDESK(IC1,IC2,IC3)%GRID%MPLWV),WK(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDESK(IC1,IC2,IC3)%GRID,.FALSE.)
                   ENDDO

                   ! phase factor on derivative X
                   CALL FFTWAV_W1(WDKX(IC1,IC2,IC3))
                   IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKX(IC1,IC2,IC3)%CR,WDKX(IC1,IC2,IC3)%CR)
                   WDKX(IC1,IC2,IC3)%CR=WDKX(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                   DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                      CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKX(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKX(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                   ENDDO

                   ! phase factor on derivative Y
                   CALL FFTWAV_W1(WDKY(IC1,IC2,IC3))
                   IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKY(IC1,IC2,IC3)%CR,WDKY(IC1,IC2,IC3)%CR)
                   WDKY(IC1,IC2,IC3)%CR=WDKY(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                   DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                      CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKY(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKY(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                   ENDDO

                   ! phase factor on derivative Z
                   CALL FFTWAV_W1(WDKZ(IC1,IC2,IC3))
                   IF(LPHASE) CALL APPLY_PHASE(WDESK(IC1,IC2,IC3)%GRID,CPHASE,WDKZ(IC1,IC2,IC3)%CR,WDKZ(IC1,IC2,IC3)%CR)
                   WDKZ(IC1,IC2,IC3)%CR=WDKZ(IC1,IC2,IC3)%CR/WDESK(IC1,IC2,IC3)%GRID%NPLWV
                   DO ISPINOR=0,WDESK(IC1,IC2,IC3)%NRSPINORS-1
                      CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),WDKZ(IC1,IC2,IC3)%CR(1+ISPINOR*WDES0%GRID%MPLWV),WDKZ(IC1,IC2,IC3)%CW(1+ISPINOR*WDES0%NGVECTOR),WDES0%GRID,.FALSE.)
                   ENDDO

                   DEALLOCATE(CPHASE)

                END DO cube3
             END DO cube2
          END DO cube1

          DISPL(1)=ABS(VKPT_ADD(1)-K0(1))/DK(1) ! in [0,1]
          DISPL(2)=ABS(VKPT_ADD(2)-K0(2))/DK(2) ! in [0,1]
          DISPL(3)=ABS(VKPT_ADD(3)-K0(3))/DK(3) ! in [0,1]
          ! all displacements must be in the interval [0,1]
          ! otherwise something is wrong
          IF( ANY( DISPL<-TINY .OR. (DISPL-1.0_q)>TINY ) )THEN
             WRITE(*,*) 'INTERPOLATE_3D_ORBITALS_SINGLE: displacement not in [0,1]',DISPL
             STOP
          END IF

          ! interpolate in x direction on 4 edges
          CALL INTER_1D_CUBIC_WEIGHTS(DISPL(1),G3)
          CALL INTER_1D_LINEAR_WEIGHTS(DISPL(1),G1)
          ! edge 1,1
          CALL INTER_1D_CUBIC(G3,WK(1,1,1)%CW,WK(2,1,1)%CW,WDKX(1,1,1)%CW,WDKX(2,1,1)%CW,W1(1,1)%CW)
          CALL INTER_1D_LINEAR(G1,WDKY(1,1,1)%CW,WDKY(2,1,1)%CW,WY1(1,1)%CW)
          CALL INTER_1D_LINEAR(G1,WDKZ(1,1,1)%CW,WDKZ(2,1,1)%CW,WZ1(1,1)%CW)
          ! edge 1,2
          CALL INTER_1D_CUBIC(G3,WK(1,1,2)%CW,WK(2,1,2)%CW,WDKX(1,1,2)%CW,WDKX(2,1,2)%CW,W1(1,2)%CW)
          CALL INTER_1D_LINEAR(G1,WDKY(1,1,2)%CW,WDKY(2,1,2)%CW,WY1(1,2)%CW)
          CALL INTER_1D_LINEAR(G1,WDKZ(1,1,2)%CW,WDKZ(2,1,2)%CW,WZ1(1,2)%CW)
          ! edge 2,1
          CALL INTER_1D_CUBIC(G3,WK(1,2,1)%CW,WK(2,2,1)%CW,WDKX(1,2,1)%CW,WDKX(2,2,1)%CW,W1(2,1)%CW)
          CALL INTER_1D_LINEAR(G1,WDKY(1,2,1)%CW,WDKY(2,2,1)%CW,WY1(2,1)%CW)
          CALL INTER_1D_LINEAR(G1,WDKZ(1,2,1)%CW,WDKZ(2,2,1)%CW,WZ1(2,1)%CW)
          ! edge 2,2
          CALL INTER_1D_CUBIC(G3,WK(1,2,2)%CW,WK(2,2,2)%CW,WDKX(1,2,2)%CW,WDKX(2,2,2)%CW,W1(2,2)%CW)
          CALL INTER_1D_LINEAR(G1,WDKY(1,2,2)%CW,WDKY(2,2,2)%CW,WY1(2,2)%CW)
          CALL INTER_1D_LINEAR(G1,WDKZ(1,2,2)%CW,WDKZ(2,2,2)%CW,WZ1(2,2)%CW)

          ! interpolate in y direction on 2 edges
          CALL INTER_1D_CUBIC_WEIGHTS(DISPL(2),G3)
          CALL INTER_1D_LINEAR_WEIGHTS(DISPL(2),G1)
          ! edge x,1
          CALL INTER_1D_CUBIC(G3,W1(1,1)%CW,W1(2,1)%CW,WY1(1,1)%CW,WY1(2,1)%CW,W2(1)%CW)
          CALL INTER_1D_LINEAR(G1,WZ1(1,1)%CW,WZ1(2,1)%CW,WZ2(1)%CW)
          ! edge x,2
          CALL INTER_1D_CUBIC(G3,W1(1,2)%CW,W1(2,2)%CW,WY1(1,2)%CW,WY1(2,2)%CW,W2(2)%CW)
          CALL INTER_1D_LINEAR(G1,WZ1(1,2)%CW,WZ1(2,2)%CW,WZ2(2)%CW)

          ! interpolate in z direction on 1 line
          CALL INTER_1D_CUBIC_WEIGHTS(DISPL(3),G3)
          CALL INTER_1D_CUBIC(G3,W2(1)%CW,W2(2)%CW,WZ2(1)%CW,WZ2(2)%CW,W1(1,1)%CW)
          WINT(:,IB,ISP)=W1(1,1)%CW

          DO IC1=1,2
             DO IC2=1,2
                DO IC3=1,2
                   CALL DELWAV( WK(IC1,IC2,IC3),  .TRUE. )
                   CALL DELWAV( WDKX(IC1,IC2,IC3),.TRUE. )
                   CALL DELWAV( WDKY(IC1,IC2,IC3),.TRUE. )
                   CALL DELWAV( WDKZ(IC1,IC2,IC3),.TRUE. )
                END DO
             END DO
          END DO

       END DO band
    END DO spin

    DO IC1=1,2
       DO IC2=1,2
          CALL DELWAV(W1(IC1,IC2),.FALSE.)
          CALL DELWAV(WX1(IC1,IC2),.FALSE.)
          CALL DELWAV(WY1(IC1,IC2),.FALSE.)
          CALL DELWAV(WZ1(IC1,IC2),.FALSE.)
       END DO
       CALL DELWAV(W2(IC1),.FALSE.)
       CALL DELWAV(WX2(IC1),.FALSE.)
       CALL DELWAV(WY2(IC1),.FALSE.)
       CALL DELWAV(WZ2(IC1),.FALSE.)
    END DO

  END SUBROUTINE INTERPOLATE_3D_ORBITALS_SINGLE



  !************* SUBROUTINE DEPS1_DK **************************************
  ! 
  ! Calculate derivative d eps^-1(G,k)/dk
  !
  !************************************************************************
  SUBROUTINE DEPS1_DK(EPS1,WDES,KPOINTS,LAT,DEPS1)
    USE lattice
    USE mkpoints
    USE wave
    USE pead
    USE mlwf
    TYPE(kpoints_struct) :: KPOINTS
    TYPE(wavedes) :: WDES
    COMPLEX(q),intent(in) :: EPS1(WDES%NGDIM,KPOINTS%NKPTS)
    TYPE(latt) :: LAT
    COMPLEX(q),intent(out) :: DEPS1(WDES%NGDIM,KPOINTS%NKPTS,3)
    !
    INTEGER :: IORDER,IK,IDIR,IDELTA,ISGN,NK
    REAL(q) :: DK(3),K0(3),K(3),KP(3)
    COMPLEX(q) :: TMP(WDES%NGDIM)
    TYPE (wavedes1) WDESK,WDESIK
    COMPLEX(q), ALLOCATABLE :: CPHASE(:) 
    LOGICAL LPHASE
    COMPLEX(q) :: EPS1R(WDES%GRID%MPLWV), &  ! EPS1 in real space
         EPS1T(WDES%NGDIM)                   ! transformed EPS1

    IORDER=4
    DK(1)=1._q/REAL(KPOINTS%NKPX,KIND=q)
    DK(2)=1._q/REAL(KPOINTS%NKPY,KIND=q)
    DK(3)=1._q/REAL(KPOINTS%NKPZ,KIND=q)

    ALLOCATE(CPHASE(WDES%GRID%MPLWV))

    DEPS1=0
    kpoint: DO IK=1,KPOINTS%NKPTS
       K0(:)=KPOINTS%VKPT(:,IK)
       CALL SETWDES(WDES,WDESIK,IK)

       dir: DO IDIR=1,3
          delta: DO IDELTA=1,IORDER
             sgn: DO ISGN=-1,1,2

                K(:)=K0(:)
                K(IDIR)=K0(IDIR)+ISGN*IDELTA*DK(IDIR)
                NK=KPOINT_IN_GRID(K,KPOINTS%VKPT)
                IF(NK<1)THEN
                   WRITE(*,*) 'DEPS1_DK: k-point not found'
                   STOP
                END IF
                KP(:)=KPOINTS%VKPT(:,NK)

                CALL SETWDES(WDES,WDESK,NK)
                CALL SETPHASE( KP(:)-K(:),WDESK%GRID,CPHASE,LPHASE)

                ! FFT to real space EPS1
                EPS1R=0
                CALL FFTWAV(WDESK%NGVECTOR, WDESK%NINDPW(1),EPS1R(1),EPS1(1:WDESK%NGVECTOR,NK),WDESK%GRID)
                IF (LPHASE) CALL APPLY_PHASE(WDESK%GRID,CPHASE,EPS1R,EPS1R)
                EPS1R=EPS1R/WDESK%GRID%NPLWV
                ! FFT back to reciprocal space
                EPS1T=0
                CALL FFTEXT(WDESIK%NGVECTOR,WDESIK%NINDPW(1),EPS1R(1),EPS1T(1:WDESIK%NGVECTOR),WDESIK%GRID,.FALSE.)

                DEPS1(1:WDESIK%NGVECTOR,IK,IDIR)=DEPS1(1:WDESIK%NGVECTOR,IK,IDIR)+ISGN*EPS1T(1:WDESIK%NGVECTOR)*FAC(IDELTA,IORDER)/DK(IDIR)/2._q

             ENDDO sgn
          ENDDO delta
       ENDDO dir

    ENDDO kpoint
    DEALLOCATE(CPHASE)

  END SUBROUTINE DEPS1_DK



  !************* SUBROUTINE INTERPOLATE_3D_EPS_SINGLE ********************* ! 
  !
  ! Cubic 3D-interpolation of eps^-1 on a _single_ of k-point VKPT_ADD.
  ! VKPT_ADD must lie in a cube whose corners are points in the coarse k-grid.
  !
  ! EPS1                                         eps^-1 on coarse grid
  ! DEPS1                                        d eps^-1/dk 
  ! EPS1_INT                                     eps^-1 on dense grid interpolated
  ! WDES
  ! VKPT                                         coarse k-points
  ! DK                                           coarse k-point spacing
  ! NN                                           nearest neighbour list of coarse grid
  ! NK_BASE                                      index of base point (cube corner) 
  !                                              used for interpolation
  ! VKPT_ADD                                     k-point you want to interpolate to
  ! SK0      optional: k-coordinates of the base corner NK_BASE
  !          This is necessary if your k-point VKPT_ADD is in another Primitive cell
  !          than spanned by VKPT. SK0 must be VKPT(:,NK_BASE) + integer shift 
  !          (by reciprocal lattice vector)
  !
  !************************************************************************
  SUBROUTINE INTERPOLATE_3D_EPS_SINGLE(EPS1,DEPS1,EPS1_INT,WDES,VKPT,DK,NK_BASE,NN,VKPT_ADD,SK0)
    USE mkpoints
    USE mlwf
    USE interpolate
    USE constant
    IMPLICIT NONE
    COMPLEX(q) :: EPS1(:,:),DEPS1(:,:,:),EPS1_INT(:)
    TYPE(wavedes) :: WDES
    REAL(q) :: VKPT(:,:), DK(3)
    INTEGER :: NK_BASE,NN(:,:,:)
    REAL(q) :: VKPT_ADD(3)
    REAL(q), OPTIONAL :: SK0(3) ! shifted base k-point
    !
    TYPE(wavedes1) :: WDES0,WDESKP
    INTEGER :: NG,NK
    COMPLEX(q),ALLOCATABLE :: E(:,:,:,:),DEX(:,:,:,:),DEY(:,:,:,:),DEZ(:,:,:,:),&
         E1(:,:,:),DEX1(:,:,:),DEY1(:,:,:),DEZ1(:,:,:),&
         E2(:,:),DEX2(:,:),DEY2(:,:),DEZ2(:,:),&
         ER(:)  ! EPS1 in real space
    INTEGER :: NKPTS_ADD,CUBE(2,2,2),IC1,IC2,IC3,ISGN(3),IDIR(3),ISP,IB,ISPINOR
    REAL(q) :: DISPL(3),K0(3),K(3),KP(3),G3(4),G1(2)
    REAL(q) :: TINY=1E-12_q
    COMPLEX(q), ALLOCATABLE :: CPHASE(:)
    LOGICAL LPHASE
    INTEGER J


    NG=SIZE(EPS1,1)
    NK=SIZE(EPS1,2)

    ! base k-point = k-point from which cube is drawn
    K0=VKPT(:,NK_BASE)
    IF(PRESENT(SK0))THEN
       ! base k-point is not included in VKPT, i.e. in a different Brillouin zone
       IF ( .NOT. ALL( ABS(MOD(K0-SK0+10.5_q,1._q)-0.5_q)<TINY ) )THEN
          WRITE(*,*) 'INTERPOLATE_3D_EPS_SINGLE: SK0 and NK_BASE not compatible',K0,SK0
          STOP
       END IF
       K0=SK0
    END IF
    ! TODO: this should be automated

    ! cube orientation
    ISGN=SIGN(1.0_q,VKPT_ADD-K0) ! -1/1
    IDIR=(3+ISGN)/2 ! 1/2
    ! the cube is always composed of k-points in VKPT
    ! if SK0 is set, the corresponding k-points in another BZ will be used
    CALL CUBE_KPOINTS(NN,NK_BASE,IDIR,CUBE)
    CALL SETWDES(WDES,WDES0,NK_BASE)

    ! auxiliary arrays for interpolation
    ALLOCATE(E(NG,2,2,2),DEX(NG,2,2,2),DEY(NG,2,2,2),DEZ(NG,2,2,2),&
         E1(NG,2,2),DEX1(NG,2,2),DEY1(NG,2,2),DEZ1(NG,2,2),&
         E2(NG,2),DEX2(NG,2),DEY2(NG,2),DEZ2(NG,2),&
         ER(WDES%GRID%MPLWV))

    cube1: DO IC1=1,2
       cube2: DO IC2=1,2
          cube3: DO IC3=1,2

             ! function values and derivatives at this corner of the cube
             E(:,IC1,IC2,IC3)=EPS1(:,CUBE(IC1,IC2,IC3))
             DEX(:,IC1,IC2,IC3)=DEPS1(:,CUBE(IC1,IC2,IC3),1)*DK(1)*ISGN(1)
             DEY(:,IC1,IC2,IC3)=DEPS1(:,CUBE(IC1,IC2,IC3),2)*DK(2)*ISGN(2)
             DEZ(:,IC1,IC2,IC3)=DEPS1(:,CUBE(IC1,IC2,IC3),3)*DK(3)*ISGN(3)

             ! k-value at this corner
             K(1)=K0(1)+DK(1)*(IC1-1)*ISGN(1)
             K(2)=K0(2)+DK(2)*(IC2-1)*ISGN(2)
             K(3)=K0(3)+DK(3)*(IC3-1)*ISGN(3)
             ! equivalent k-value 
             KP=VKPT(:,CUBE(IC1,IC2,IC3) )

             ! transform from KP layout to K layout (which is the layout of K0)
             ! i.e. everything is then in K0 layout
             CALL SETWDES(WDES,WDESKP,CUBE(IC1,IC2,IC3))

             ALLOCATE(CPHASE(WDESKP%GRID%MPLWV))
             CALL SETPHASE( KP(:)-K(:),WDESKP%GRID,CPHASE,LPHASE)

             ! eps1
             CALL FFTWAV(WDESKP%NGVECTOR,WDESKP%NINDPW(1),ER(1),E(1:WDESKP%NGVECTOR,IC1,IC2,IC3),WDESKP%GRID)
             IF(LPHASE) CALL APPLY_PHASE(WDESKP%GRID,CPHASE,ER,ER)
             ER=ER/WDESKP%GRID%NPLWV
             CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),ER(1),E(1:WDES0%NGVECTOR,IC1,IC2,IC3),WDES0%GRID,.FALSE.)

             ! dx
             CALL FFTWAV(WDESKP%NGVECTOR,WDESKP%NINDPW(1),ER(1),DEX(1:WDESKP%NGVECTOR,IC1,IC2,IC3),WDESKP%GRID)
             IF(LPHASE) CALL APPLY_PHASE(WDESKP%GRID,CPHASE,ER,ER)
             ER=ER/WDESKP%GRID%NPLWV
             CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),ER(1),DEX(1:WDES0%NGVECTOR,IC1,IC2,IC3),WDES0%GRID,.FALSE.)

             ! dy
             CALL FFTWAV(WDESKP%NGVECTOR,WDESKP%NINDPW(1),ER(1),DEY(1:WDESKP%NGVECTOR,IC1,IC2,IC3),WDESKP%GRID)
             IF(LPHASE) CALL APPLY_PHASE(WDESKP%GRID,CPHASE,ER,ER)
             ER=ER/WDESKP%GRID%NPLWV
             CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),ER(1),DEY(1:WDES0%NGVECTOR,IC1,IC2,IC3),WDES0%GRID,.FALSE.)

             ! dz
             CALL FFTWAV(WDESKP%NGVECTOR,WDESKP%NINDPW(1),ER(1),DEZ(1:WDESKP%NGVECTOR,IC1,IC2,IC3),WDESKP%GRID)
             IF(LPHASE) CALL APPLY_PHASE(WDESKP%GRID,CPHASE,ER,ER)
             ER=ER/WDESKP%GRID%NPLWV
             CALL FFTEXT(WDES0%NGVECTOR,WDES0%NINDPW(1),ER(1),DEZ(1:WDES0%NGVECTOR,IC1,IC2,IC3),WDES0%GRID,.FALSE.)

             DEALLOCATE(CPHASE)

          END DO cube3
       END DO cube2
    END DO cube1

    DISPL(1)=ABS(VKPT_ADD(1)-K0(1))/DK(1) ! in [0,1]
    DISPL(2)=ABS(VKPT_ADD(2)-K0(2))/DK(2) ! in [0,1]
    DISPL(3)=ABS(VKPT_ADD(3)-K0(3))/DK(3) ! in [0,1]
    ! all displacements must be in the interval [0,1]
    ! otherwise something is wrong
    IF( ANY( DISPL<-TINY .OR. (DISPL-1.0_q)>TINY ) )THEN
       WRITE(*,*) 'INTERPOLATE_3D_ORBITALS_SINGLE: displacement not in [0,1]',DISPL
       STOP
    END IF

    ! interpolate in x direction on 4 edges
    CALL INTER_1D_CUBIC_WEIGHTS(DISPL(1),G3)
    CALL INTER_1D_LINEAR_WEIGHTS(DISPL(1),G1)
    ! edge 1,1
    CALL INTER_1D_CUBIC(G3,E(:,1,1,1),E(:,2,1,1),DEX(:,1,1,1),DEX(:,2,1,1),E1(:,1,1))
    CALL INTER_1D_LINEAR(G1,DEY(:,1,1,1),DEY(:,2,1,1),DEY1(:,1,1))
    CALL INTER_1D_LINEAR(G1,DEZ(:,1,1,1),DEZ(:,2,1,1),DEZ1(:,1,1))

    ! edge 1,2
    CALL INTER_1D_CUBIC(G3,E(:,1,1,2),E(:,2,1,2),DEX(:,1,1,2),DEX(:,2,1,2),E1(:,1,2))
    CALL INTER_1D_LINEAR(G1,DEY(:,1,1,2),DEY(:,2,1,2),DEY1(:,1,2))
    CALL INTER_1D_LINEAR(G1,DEZ(:,1,1,2),DEZ(:,2,1,2),DEZ1(:,1,2))
    ! edge 2,1
    CALL INTER_1D_CUBIC(G3,E(:,1,2,1),E(:,2,2,1),DEX(:,1,2,1),DEX(:,2,2,1),E1(:,2,1))
    CALL INTER_1D_LINEAR(G1,DEY(:,1,2,1),DEY(:,2,2,1),DEY1(:,2,1))
    CALL INTER_1D_LINEAR(G1,DEZ(:,1,2,1),DEZ(:,2,2,1),DEZ1(:,2,1))
    ! edge 2,2
    CALL INTER_1D_CUBIC(G3,E(:,1,2,2),E(:,2,2,2),DEX(:,1,2,2),DEX(:,2,2,2),E1(:,2,2))
    CALL INTER_1D_LINEAR(G1,DEY(:,1,2,2),DEY(:,2,2,2),DEY1(:,2,2))
    CALL INTER_1D_LINEAR(G1,DEZ(:,1,2,2),DEZ(:,2,2,2),DEZ1(:,2,2))

    ! interpolate in y direction on 2 edges
    CALL INTER_1D_CUBIC_WEIGHTS(DISPL(2),G3)
    CALL INTER_1D_LINEAR_WEIGHTS(DISPL(2),G1)
    ! edge x,1
    CALL INTER_1D_CUBIC(G3,E1(:,1,1),E1(:,2,1),DEY1(:,1,1),DEY1(:,2,1),E2(:,1))
    CALL INTER_1D_LINEAR(G1,DEZ1(:,1,1),DEZ1(:,2,1),DEZ2(:,1))
    ! edge x,2
    CALL INTER_1D_CUBIC(G3,E1(:,1,2),E1(:,2,2),DEY1(:,1,2),DEY1(:,2,2),E2(:,2))
    CALL INTER_1D_LINEAR(G1,DEZ1(:,1,2),DEZ1(:,2,2),DEZ2(:,2))

    ! interpolate in z direction on 1 line
    CALL INTER_1D_CUBIC_WEIGHTS(DISPL(3),G3)
    CALL INTER_1D_CUBIC(G3,E2(:,1),E2(:,2),DEZ2(:,1),DEZ2(:,2),E1(:,1,1))
    EPS1_INT(:)=E1(:,1,1)

    DEALLOCATE(E,DEX,DEY,DEZ,E1,DEX1,DEY1,DEZ1,E2,DEX2,DEY2,DEZ2,ER)

  END SUBROUTINE INTERPOLATE_3D_EPS_SINGLE



  !************* SUBROUTINE FOURIER_INTERPOL ******************************
  !
  ! Fourier interpolation of the Hamiltonian matrix.
  ! KINTER >=1  divisor for k-point spacing
  !             creates and returns interpolated grid in VKPT_INT
  !        ==0  interpolate on given (irregular) k-points
  !             k-points must be passed in VKPT_INT
  ! DHDK_INT    optional, calculate derivatives. Not for irregular grids.
  !
  !************************************************************************ 
  SUBROUTINE FOURIER_INTERPOL(KPOINTS,VKPT_INT,KINTER,HAM,HAM_INT,LAT,IO,DHDK_INT)
    USE base
    USE mkpoints
    USE lattice
    USE util
    USE constant
    IMPLICIT NONE
    include 'fftw3.f'
    TYPE (kpoints_struct) :: KPOINTS
    REAL(q) :: VKPT_INT(:,:)
    INTEGER :: KINTER
    COMPLEX(q) :: HAM(:,:,:,:),HAM_INT(:,:,:,:)
    TYPE (latt) :: LAT
    TYPE (in_struct)   IO
    COMPLEX(q), OPTIONAL :: DHDK_INT(:,:,:,:,:)
    !
    INTEGER :: N1,N2,N3,M1,M2,M3,NB1,NB2,NKP,NSP,&
         I1,I2,I3,IK,JK,NEQ,IB1,IB2,ISP,&
         NKP_INT,IW1,IW2,IW3,IDIR
    INTEGER(8) :: PLANF,PLANB
    COMPLEX(q), ALLOCATABLE :: C(:,:,:),D(:,:,:)
    COMPLEX(q), ALLOCATABLE :: CD(:,:,:),DD(:,:,:,:) ! arrays for storing dH/dk
    REAL(q), ALLOCATABLE :: VKPT(:,:)
    INTEGER, ALLOCATABLE :: KINDEX(:)
    INTEGER :: DISPL(3),DISPL_FOUND(3),DISPL_ALL(3,10)
    REAL(q) :: DISPL_CART(3),DIST,DIST_FOUND
    REAL(q) :: TINY=1.D-8
    LOGICAL :: DO_DHDK,DO_IRREGULAR

    DO_DHDK=.FALSE.
    IF(PRESENT(DHDK_INT))DO_DHDK=.TRUE.
    DO_IRREGULAR=.FALSE.
    IF(KINTER==0)DO_IRREGULAR=.TRUE.
    IF(DO_IRREGULAR.AND.DO_DHDK)THEN
       WRITE(*,*) 'FOURIER_INTERPOL: derivatives on irregular k-grid not implemented'
       STOP
    END IF

    ! coarse grid
    N1=KPOINTS%NKPX
    N2=KPOINTS%NKPY
    N3=KPOINTS%NKPZ
    ! fine grid
    M1=N1*KINTER
    M2=N2*KINTER
    M3=N3*KINTER   

    NB1=SIZE(HAM,1)
    NB2=SIZE(HAM,2)
    NKP=KPOINTS%NKPTS
    NSP=SIZE(HAM,4)
    ALLOCATE(C(0:N1-1,0:N2-1,0:n3-1))
    IF(.NOT.DO_IRREGULAR)THEN
       ALLOCATE(D(0:M1-1,0:M2-1,0:M3-1))
       IF(DO_DHDK) ALLOCATE(DD(0:M1-1,0:M2-1,0:M3-1,3))
    END IF

    ! create index array for k-points
    ! such that k-points are ordered by increasing x, then y, then z coordinate
    ! KINDEX(i) is the _original_ position of the i-th ordered k-point
    ! create ordered kpoints grid for Fourier interpolation
    ALLOCATE(KINDEX(NKP),VKPT(3,N1*N2*N3))
    CALL KPOINTS_SUPERGRID(N1,N2,N3,VKPT)
    DO IK=1,NKP
       ! find corresponding kpoints in original grid
       JK=KPOINT_IN_GRID(VKPT(:,IK),KPOINTS%VKPT)
       KINDEX(IK)=JK
    END DO

    IF(DO_IRREGULAR)THEN
       NKP_INT=SIZE(VKPT_INT,2)
    ELSE
       NKP_INT=M1*M2*M3
       CALL KPOINTS_SUPERGRID(M1,M2,M3,VKPT_INT)
    END IF

    IF(SIZE(HAM_INT,3)/=NKP_INT)THEN
       WRITE(*,*) 'FOURIER_INTERPOL: HAM_INT has not the correct k-point size',SIZE(HAM_INT,3),NKP_INT
       STOP
    END IF
    IF(PRESENT(DHDK_INT))THEN
       IF(SIZE(DHDK_INT,3)/=NKP_INT)THEN
          WRITE(*,*) 'FOURIER_INTERPOL: DHDK_INT has not the correct k-point size',SIZE(DHDK_INT,3),NKP_INT
          STOP
       END IF
    END IF

    ! first call: create a plan
    ! either using FFTW_MEASURE (takes a lot of time)
    ! or using FFTW_ESTIMATE (faster but FFTs less efficient)
    ! Neither of them actually performs a FFT, FFTW_MEASURE overwrites the arrays
    call dfftw_plan_dft_3d(PLANF,N1,N2,N3,C,C,FFTW_FORWARD,FFTW_MEASURE)
    ! backtransform create plan
    IF(.NOT.DO_IRREGULAR)THEN
       call dfftw_plan_dft_3d(PLANB,M1,M2,M3,D,D,FFTW_BACKWARD,FFTW_MEASURE)
    END IF

    HAM_INT=0
    DO ISP=1,NSP
       DO IB1=1,NB1
          DO IB2=1,NB2

             ! sort HAM in the same order as k-points
             IK=1
             DO I3=0,N3-1
                DO I2=0,N2-1
                   DO I1=0,N1-1
                      C(I1,I2,I3)=HAM(IB1,IB2,KINDEX(IK),ISP)
                      IK=IK+1
                   END DO
                END DO
             END DO

             call dfftw_execute_dft(PLANF,C,C)

             ! put onto fine grid

             ! ZERO PADDING IN MULTIPLE DIMENSIONS:
             ! The coarse grid values must be inserted in the Wigner-Seitz cell of
             ! the fine grid. This ensures that the SMALLEST frequencies are used
             ! as the interpolating functions!
             !
             ! HAM is defined on the reciprocal lattice:
             ! vec{k}=k1 \vec{b1} + k2 \vec{b2} + k3 \vec{b3}
             ! ki= 0 .. 1, i=1,2,3
             ! The I-th point: k1(I) = (I-1)/N1, I=1,..,N1
             !
             ! The reciprocal of the reciprocal lattice is the direct lattice
             ! - multiplied by N1, N2, N3 for the coarse grid: N1 \vec{a1} etc.
             ! - multiplied by M1, M2, M3 for the fine grid:   M1 \vec{a1} etc.
             ! \vec{R} = R1 N1 \vec{a1} + R2 N2 \vec{a2} + R3 N3 \vec{a3}
             !
             ! 1. Find out the smallest frequency for each point on the coarse grid
             ! 2. Find the corresponding frequency on the fine grid and copy the value there

             IF(.NOT.DO_IRREGULAR)THEN
                D=0
                IF(DO_DHDK) DD=0
             END IF
             DO I3=0,N3-1
                DO I2=0,N2-1
                   DO I1=0,N1-1
                      !
                      ! search the equivalent displacement with the shortest distance
                      DIST_FOUND=1E10
                      DO IW3=-N3,N3,N3
                         DO IW2=-N2,N2,N2
                            DO IW1=-N1,N1,N1
                               ! displacement in direct coordinates
                               ! The actual value of R1 is [(I1-1)+IW1]/N1*(N1*\vec{a1})
                               ! N1 cancels out, so we can use LAT%A directly in DIRKAR
                               ! analogously for R2 and R3
                               DISPL(1)=I1+IW1
                               DISPL(2)=I2+IW2
                               DISPL(3)=I3+IW3
                               ! R-point displacement in cartesian coordinates
                               DISPL_CART=DISPL
                               CALL DIRKAR(1, DISPL_CART(1), LAT%A)
                               DIST=SUM(DISPL_CART**2)
                               IF (DIST<=DIST_FOUND) THEN
                                  DISPL_FOUND=DISPL
                                  DIST_FOUND=DIST
                               ENDIF
                            ENDDO
                         ENDDO
                      ENDDO
                      ! now search all equivalent displacements with the same shortest distance DIST_FOUND
                      ! and store them in DISPL_ALL. there mAY be more than 1 of them!
                      NEQ=0
                      DO IW3=-N3,N3,N3
                         DO IW2=-N2,N2,N2
                            DO IW1=-N1,N1,N1
                               ! displacement in direct coordinates
                               DISPL(1)=I1+IW1
                               DISPL(2)=I2+IW2
                               DISPL(3)=I3+IW3
                               ! k-point displacement in cartesian coordinates
                               DISPL_CART=DISPL
                               CALL DIRKAR(1, DISPL_CART(1), LAT%A)
                               DIST=SUM(DISPL_CART**2)
                               IF (ABS(DIST-DIST_FOUND)<TINY) THEN
                                  NEQ=NEQ+1
                                  DISPL_ALL(:,NEQ)=DISPL
                               ENDIF
                            ENDDO
                         ENDDO
                      ENDDO
                      !
                      ! find the corresponding points in the fine grid
                      ! and distribute the value in C among these equivalent points evenly
                      ! 
                      DO JK=1,NEQ
                         !
                         IF(DO_IRREGULAR)THEN
                            !
                            ! Evaluate EXP( 2 pi i kj Rj Nj } (see below) for the given k-values in VKPT_INT
                            ! Rj Nj = DISPL_ALL
                            !
                            DO IK=1,NKP_INT
                               HAM_INT(IB1,IB2,IK,ISP)=HAM_INT(IB1,IB2,IK,ISP)+ &
                                    C(I1,I2,I3)*EXP( CITPI*SUM(VKPT_INT(:,IK)*DISPL_ALL(:,JK)) )/REAL(NEQ,q)
                            END DO
                            !
                         ELSE
                            !
                            ! mapping back is easy, we just have to map the R-value into [0,M1-1] etc.
                            ! 
                            DISPL(1)=MOD(DISPL_ALL(1,JK)+M1,M1)
                            DISPL(2)=MOD(DISPL_ALL(2,JK)+M2,M2)
                            DISPL(3)=MOD(DISPL_ALL(3,JK)+M3,M3)
                            !
                            D(DISPL(1),DISPL(2),DISPL(3))=D(DISPL(1),DISPL(2),DISPL(3))+C(I1,I2,I3)/REAL(NEQ,q)
                            !
                            IF(DO_DHDK)THEN
                               !
                               ! Derivative:
                               ! EXP( i \vec{k}.\vec{R} )
                               ! = EXP( i kj \vec{bj} . Rl Nl \vec{al} }  // \vec{bj} . \vec{al} = delta_jl
                               ! = EXP( 2 pi i kj Rj Nj }
                               ! d/dkj = 2p i Rj Nj 
                               ! with Rj=0..1 and thus Rj Nj = Ij = 0..Nj-1
                               !
                               ! Now multiply by (2 pi i I) to calculate derivative. The correct value of R to multiply
                               ! with at given index is given by the smallest equivalent R-value, 
                               ! and that is just what is stored in DISPL_ALL!
                               !
                               DO IDIR=1,3
                                  DD(DISPL(1),DISPL(2),DISPL(3),IDIR)=DD(DISPL(1),DISPL(2),DISPL(3),IDIR)+C(I1,I2,I3)/REAL(NEQ,q)*DISPL_ALL(IDIR,JK)*CITPI
                               END DO
                               !
                            END IF
                            !
                         END IF
                         !
                      END DO
                      !
                   ENDDO
                ENDDO
             ENDDO

             IF(.NOT.DO_IRREGULAR)THEN

                ! calculate H
                call dfftw_execute_dft(PLANB,D,D)
                D=D/REAL(N1*N2*N3,q)
                IK=0
                DO I3=0,M3-1
                   DO I2=0,M2-1
                      DO I1=0,M1-1
                         IK=IK+1
                         HAM_INT(IB1,IB2,IK,ISP)=D(I1,I2,I3)
                      END DO
                   END DO
                END DO

                IF(DO_DHDK)THEN
                   ! calculate dH/dk
                   DO IDIR=1,3
                      D=DD(:,:,:,IDIR)
                      call dfftw_execute_dft(PLANB,D,D)
                      D=D/REAL(N1*N2*N3,q)
                      IK=0
                      DO I3=0,M3-1
                         DO I2=0,M2-1
                            DO I1=0,M1-1
                               IK=IK+1
                               DHDK_INT(IB1,IB2,IK,ISP,IDIR)=D(I1,I2,I3)
                            END DO
                         END DO
                      END DO
                   END DO
                END IF

             END IF

          END DO
       END DO
    END DO
    IF(DO_IRREGULAR)HAM_INT=HAM_INT/REAL(N1*N2*N3,q)

    call dfftw_destroy_plan(PLANF)
    IF(.NOT.DO_IRREGULAR) call dfftw_destroy_plan(PLANB)

    DEALLOCATE(C,KINDEX,VKPT)
    IF(.NOT.DO_IRREGULAR)THEN
       DEALLOCATE(D)
       IF(DO_DHDK)DEALLOCATE(DD)
    END IF

  END SUBROUTINE FOURIER_INTERPOL


  !************* SUBROUTINE CUBE_KPOINTS **********************************
  ! 
  ! returns the indices CUBE of the k-points that form a cube with the given 
  ! k-point NK in the directions ISGN (x,y,z = 1 backw./2 forw.)
  ! CUBE(1,1,1) is the origin, CUBE(2,1,1) goes one step in x etc.
  !
  !************************************************************************
  SUBROUTINE CUBE_KPOINTS(NN,NK,ISGN,CUBE)
    IMPLICIT NONE
    INTEGER :: NN(2,3,*),NK,ISGN(3),CUBE(2,2,2)
    INTEGER :: I,POS(3)
    !
    ! origin
    CUBE(1,1,1)=NK
    ! step in x,y,z from 1,1,1
    DO I=1,3
       POS=1
       POS(I)=2
       CUBE(POS(1),POS(2),POS(3))=NN(ISGN(I),I,NK)
    END DO
    ! step in y,z from 2,1,1
    DO I=2,3
       POS=(/2,1,1/)
       POS(I)=2
       CUBE(POS(1),POS(2),POS(3))=NN(ISGN(I),I,CUBE(2,1,1))
    END DO
    ! step in z from 2,2,1
    DO I=3,3
       POS=(/2,2,1/)
       POS(3)=2
       CUBE(POS(1),POS(2),POS(3))=NN(ISGN(I),I,CUBE(2,2,1))
    END DO
    ! step in z from 1,2,1
    DO I=3,3
       POS=(/1,2,1/)
       POS(3)=2
       CUBE(POS(1),POS(2),POS(3))=NN(ISGN(I),I,CUBE(1,2,1))
    END DO
  END SUBROUTINE CUBE_KPOINTS
#endif

END MODULE wannier_interpolation
