#include "symbol.inc"

MODULE vdw_df_helper
  INTEGER :: IU0=6,IU6=6,ISIF=0
CONTAINS
  SUBROUTINE SET_vdW_IO_UNITS(IU0_in, IU6_in, ISIF_in)
    INTEGER IU0_in, IU6_in, ISIF_in
    IU0=IU0_in
    IU6=IU6_in
    ISIF=ISIF_in
  END SUBROUTINE SET_vdW_IO_UNITS
END MODULE
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!
!!!    module vdw_ll
!!!
!!! This is used to evaluate the non-local
!!! correlation functional of Dion et al., PRL 2004
!!! The fast procedure of Roman-Perez and Soler PRL 2010 is used
!!! which is available in Siesta and GPAW which this implementation follows
!!! Written by jik 2009-2011
!!! Please cite 
!!!   G. Rom\'{a}n-P\'{e}rez and J. Soler, Phys. Rev. Lett. 103, 096102 (2009)
!!!   J. Klime\v{s}, D. R. Bowler, and A. Michaelides, PRB 83, 195131 (2011) 
!!! and 
!!!   J. Klime\v{s}, D. R. Bowler, and A. Michaelides, J. Phys.: Cond. Matt. 22, 022201 (2010)
!!! if you use optPBE-vdW or optB88-vdW functionals
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE VDW_LL
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE vdw_df_helper, ONLY : IU0,IU6,ISIF

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: VDW_NONLOC,VDW_NONLOC_spin

  INTEGER, parameter :: Nalp=30, Nr=4096

  REAL(q) :: spl(Nalp,Nalp,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) :: phir(0:Nr,Nalp,Nalp)
  REAL(q) :: qs_vdw(Nalp), qsmin
  REAL(q), parameter :: q0cut=10.0_q, lambda=1.2_q
  REAL(q), parameter :: qAlp_ratio=31.949333280885_q
  REAL(q), parameter :: rcut = 100._q
  REAL(q), parameter :: asymptot_phi=12._q*(4._q*pi/9._q)**3
  REAL(q) :: dr
  LOGICAL :: INIT_VDW_LL=.TRUE. !do we need to initialise the stuff?
  LOGICAL :: LREAD_KERN = .FALSE.
  LOGICAL :: LHAVE_PHI_TABLE = .FALSE.
  logical :: lhave_alpha_table = .FALSE.

  !dmesh parameters, normal settings
  !integer, parameter :: nd = 20
  !real(q), parameter :: dcut = 30._q
  !real(q), parameter :: dratio = 20._q
  !dmesh parameters, tight
  integer, parameter :: nd = 35
  real(q), parameter :: dcut = 50._q
  real(q), parameter :: dratio = 20._q
  logical :: luse_asymptotic = .true.
  !careful !!! the kernel needs to be recalculated when 
  !luse_asymptotic is changed (the highest coefficients are 
  !zero for .false.)
  !normal settings, double nintg for tight
  INTEGER , parameter :: nintg=6000
  REAL(q), parameter :: maxin =200._q  
  !INTEGER , parameter :: nintg=10000
  !REAL(q), parameter :: maxin =100._q


  real(q) :: dmesh(nd)
  logical :: lhave_nd_points = .FALSE.
  real(q) :: kcut
  real(q), parameter :: dsoft = 0.5_q, phi_0 =0.8_q

  real(q) :: phi_table(0:3,0:3,nd,nd)

CONTAINS

  SUBROUTINE VDW_NONLOC(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,stress)
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
    REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
    REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input !df/d(drho)/(drho)potential on output
    REAL(q) EXC
    REAL(q) stress(3,3)
    REAL(q) E_corr
    COMPLEX(q), allocatable :: CINP(:) !, THETA(:,:)
    REAL(q) ,allocatable :: k_k(:)
    RGRID   DWORK(GRIDC%RL%NP) 

    IF (INIT_VDW_LL) THEN
      call MIN_K(GRIDC,LATT_CUR)
      call GET_SPLINES
      call GET_KERNELS
      INIT_VDW_LL=.FALSE.
    ENDIF

    ALLOCATE(K_K(GRIDC%MPLWV))
    call GET_K_IND(k_k,GRIDC,LATT_CUR)

    ALLOCATE(CINP(GRIDC%MPLWV)) 
    CALL VDW_CALC(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,CINP,spl, phik, d2phidk2,qs_vdw,qsmin,k_k,stress,ISIF,E_corr)
    DEALLOCATE(CINP) 
    DEALLOCATE(k_k) 

    IF (IU0>=0) THEN
       WRITE(IU0,'(A,F14.7)') 'Total vdW correction in eV: ',E_corr
     ENDIF
  END SUBROUTINE 

  SUBROUTINE VDW_NONLOC_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,stress)
    !spin polarised version of the previous routine 
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
    REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
    REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
    REAL(q) EXC
    REAL(q) stress(3,3) !this guy is not yet calculated and passed to xcspin.F
    REAL(q) E_corr
    COMPLEX(q), allocatable :: CINP(:) !, THETA(:,:)
    REAL(q) ,allocatable :: k_k(:)
    RGRID   DWORK(GRIDC%RL%NP)

    IF (INIT_VDW_LL) THEN
      call MIN_K(GRIDC, LATT_CUR)
      call GET_SPLINES
      call GET_KERNELS
      INIT_VDW_LL=.FALSE.
    ENDIF

    ALLOCATE(K_K(GRIDC%MPLWV))
    call GET_K_IND(k_k,GRIDC,LATT_CUR)

    ALLOCATE(CINP(GRIDC%MPLWV))
    CALL VDW_CALC_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,CINP,spl, phik, d2phidk2,qs_vdw,qsmin,k_k,stress,ISIF,E_corr)

    DEALLOCATE(CINP)
    DEALLOCATE(k_k)

    IF (IU0>=0) THEN
       WRITE(IU0,'(A,F14.7)') 'Total vdW correction in eV: ',E_corr
     ENDIF
  END SUBROUTINE

  SUBROUTINE GET_SPLINES
    !construct the spline coefficients for interpolation
    REAL(q) :: a(Nalp,Nalp), c(Nalp,Nalp), l(Nalp,Nalp), mi(Nalp,Nalp), z(Nalp,Nalp)
    REAL(q) :: b(Nalp-1,Nalp), d(Nalp-1,Nalp), h(Nalp-1), al(Nalp-1,Nalp)
    INTEGER :: i,j

    a=0._q
    b=0._q
    d=0._q
    h=0._q
    al=0._q
    c=0._q
    l=0.0_q
    mi=0._q
    z=0._q
    
    call GEN_ALPHA_POINTS()

    qsmin=qs_vdw(2)

    DO i=1,Nalp
      a(i,i)=1._q
    ENDDO
    DO i=1,Nalp-1
      h(i)=qs_vdw(i+1)-qs_vdw(i)
    ENDDO
    DO i=2,Nalp-1
      DO j=1,Nalp
        al(i,j)=3._q/h(i)*(a(i+1,j)-a(i,j))-3._q/h(i-1)*(a(i,j)-a(i-1,j))
      ENDDO
    ENDDO
    l(1,:)=1._q
    DO i=2,Nalp-1
      l(i,:)=2._q*(qs_vdw(i+1)-qs_vdw(i-1))-h(i-1)*mi(i-1,:)
      mi(i,:)=h(i)/l(i,:)
      z(i,:)=(al(i,:)-h(i-1)*z(i-1,:))/l(i,:)
    ENDDO
    l(Nalp,:)=1._q
    z(Nalp,:)=0._q
    c(Nalp,:)=0._q
    DO i=Nalp-1,1,-1
      c(i,:)=z(i,:)-mi(i,:)*c(i+1,:)
      b(i,:)=(a(i+1,:)-a(i,:))/h(i)-h(i)/3._q*(c(i+1,:)+2._q*c(i,:))
      d(i,:)=(c(i+1,:)-c(i,:))/(3._q*h(i))
    ENDDO

    spl(:,:,1)=a(:,:)
    spl(:,:,3)=c(:,:)
    DO i=1,Nalp-1
      spl(i,:,2)=b(i,:)
      spl(i,:,4)=d(i,:)
    ENDDO

  END SUBROUTINE


  SUBROUTINE GET_KERNELS
    !this procedure calculates the fourier transformed vdw kernels
    IMPLICIT NONE
    INTEGER :: alf,bet,i,j,irad, nk, ik
    REAL (q) :: q1,q2, dk, krad
    REAL (q) :: r(0:nr)
    REAL (q) :: phir_work(0:nr), phik_work(0:nr)
    REAL (q) :: d2phik_work(0:nr)
    
    !setting up the mesh of interpolating points
    !call GEN_ALPHA_POINTS()
    dr= rcut/Nr
    dk= pi/rcut
    nk= int(kcut/dk)+1

    !find the interpolation coefficients for the alpha points
    DO alf = 1, Nalp
      DO bet = 1, alf
        phik_work=0._q
        phir_work=0._q
        !desaturate q interpolation points
        q1=DESATURATE(qs_vdw(alf))
        q2=DESATURATE(qs_vdw(bet))

        !get the kernel on a real grid by using the precalculated interpolation
        !data
        DO irad = 0, Nr
          r(irad) = irad * dr
          phir(irad,alf,bet) = PHI_INTERPOLATE(q1*r(irad),q2*r(irad))
        ENDDO
       
        !smoothen for large r
        DO irad =0, Nr 
          phir(irad,alf,bet)=phir(irad,alf,bet)*S_CUT(r(irad)/rcut)
        ENDDO

        !fourier transform it and smoothen
        phir_work(:)=phir(:,alf,bet)
        call PHIK_FFT(phir_work(:), phik_work(:))
        phik(:,alf,bet)=phik_work*(2._q*pi)**1.5_q
        phik(nk:nr,alf,bet)=0._q
        DO ik =1,nk
          krad=ik*dk
          phik(ik,alf,bet)=phik(ik,alf,bet)*S_CUT(krad/kcut)
        ENDDO

        !get coefficients for cubic splines
        !phik_work to splines    
        call SPLINE(dk,phik(:,alf,bet),Nr+1,0._q,0._q,d2phik_work)

        !symmetrise
        phik(:,bet,alf)=phik(:,alf,bet)
        d2phidk2(:,bet,alf)=d2phik_work
        d2phidk2(:,alf,bet)=d2phik_work
      ENDDO
    ENDDO

  END SUBROUTINE
  
  SUBROUTINE GEN_ALPHA_POINTS()
    !find the interpolation points for the kernel interpolation q_0 grid
    !i.e. fill the qs_vdw array
    REAL(q) :: a,b,e
    INTEGER :: i
    
    qs_vdw(:)=0._q
    a=log(qAlp_ratio)/(Nalp-1._q)
    b=q0cut/(qAlp_ratio-1._q)
    
    DO i=2,Nalp-1    
      qs_vdw(i)=b*exp(a*(i-1._q))-b
    ENDDO
    qs_vdw(Nalp)=q0cut
    lhave_alpha_table = .true.

  END SUBROUTINE 

  FUNCTION DESATURATE(x)
    !find q0 corresponding to saturated q0 by interval halving 
    REAL(q) :: desaturate, x
    REAL(q) :: xdo, xup, xme, xmesat
    REAL(q), parameter :: xtol=1.e-14_q
    
    !check first
    IF ((x<0._q).OR.(x>q0cut)) then
      IF (IU0>=0) WRITE(IU0,*)   'DESATURATE: out of range', x
      stop
    endif
    !this is done for the original q0 values before saturation
    xdo =  0._q
    xup = q0cut*5._q
    DO
      xme = (xdo+xup)*0.5_q
      xmesat = SATURATE(xme)
      !if smaller 
      IF (abs(xmesat-x).LE.xtol ) THEN 
        desaturate=xup
        RETURN
      ELSE IF (x.LE.xmesat) THEN
        xup=xme
      ELSE
        xdo=xme
      ENDIF
    ENDDO

  END FUNCTION

  FUNCTION SATURATE(x)
    !the same that we use down in the main routine
    !R-P&S' equation No 4 or so
    REAL(q):: saturate, x, coe
    INTEGER :: iu

    coe=0._q
    DO iu=1,12
      coe=coe+(x/q0cut)**iu/real(iu,kind=q)
    ENDDO
    saturate=q0cut*(1._q-exp(-coe))
  END FUNCTION

  FUNCTION PHI_INTERPOLATE(d1, d2)
    !finds the kernel using phi_generate
    !returns the value of kernel for d1, d2
    REAL (q) :: phi_interpolate
    REAL (q) :: d1, d2
    REAL (q) :: dd1(0:3), dd2(0:3)
    INTEGER :: id1, id2
    INTEGER :: n1, n2
    logical,save :: lfirst_G = .TRUE.

    !the interpolation table needs to be generated before we access it
    IF (.NOT. lhave_phi_table) call PHI_GENERATE()
    !don't go over the max
      IF ((d1.GE.dcut).OR.(d2.GE.dcut)) THEN
      !or use C/d^6 
        IF (.NOT.luse_asymptotic) THEN
          phi_interpolate=0._q 
        ELSE
          !if d1 or d2 are 0 then the thing doesn't work, 
          !but as my boss of long time ago (Fiala, UFA) said: 
          !"cut-off at 1, it's close to infinity"
          !the terms here are on the order of 10^-6 compared to the big ones
          !so that the errors should be small (when d1 or d2 are not that big)
          !and presumably not cutting the thing off will help? we'll see...
          IF ((d1.GE.dmesh(2)) .AND. (d2.GE.dmesh(2))) THEN
            phi_interpolate=-asymptot_phi/(d1*d1*d2*d2*(d1*d1+d2*d2))
          ELSE
            phi_interpolate=0._q
          ENDIF
        ENDIF
      RETURN
    ENDIF

    !find the indices and interpolate the value
    id1 = IOFD(d1)
    id2 = IOFD(d2)
    dd1(0) = 1._q
    dd2(0) = 1._q 
    dd1(1) = (d1 - dmesh(id1)) / ( dmesh(id1+1) - dmesh(id1))  
    dd2(1) = (d2 - dmesh(id2)) / ( dmesh(id2+1) - dmesh(id2))
    dd1(2) = dd1(1)**2
    dd2(2) = dd2(1)**2
    dd1(3) = dd1(1)**3
    dd2(3) = dd2(1)**3
   
    phi_interpolate = 0._q
    DO n2 = 0,3
      DO n1 = 0,3
        phi_interpolate = phi_interpolate + phi_table(n1,n2,id1,id2)*dd1(n1)*dd2(n2)
      ENDDO
    ENDDO 

  END FUNCTION

  SUBROUTINE GEN_ND_POINTS()
    !find the interpolation points for the dmesh grid
    !i.e. fill the dmesh array
    !it is funny that fields in physics has the same word as 
    !the fields in the real world like corn field
    !even in French I found out today
    !Champes Elysses vs. Tres Hauts Champs
    REAL(q) :: a,b,e
    INTEGER :: i

    dmesh(:)=0._q
    a=log(dratio)/(nd-1._q)
    b=dcut/(dratio-1._q)

    DO i=1,nd-1
      dmesh(i)=b*exp(a*(i-1._q))-b
    ENDDO
    dmesh(nd)=dcut
    lhave_nd_points = .true.

  END SUBROUTINE

  FUNCTION IOFD(d)
    !finds closest smallest index point for the kernel interpolation
    !ok so the point is that we have numbers b*(exp(a*(i-1))-1)
    INTEGER :: iofd
    REAL (q) :: d
    REAL (q), save :: a,b 
    LOGICAL, save :: lfirst = .true.
   
    IF (lfirst) THEN 
      a = log(dratio)/(nd-1._q)        
      a = max( a,1.e-13_q)
      b = dcut/(dratio-1._q) 
      lfirst = .false. 
    ENDIF
    
    iofd = 1+log(1+(d-dmesh(1))/b)/a
    iofd = max ( 1, iofd)
    iofd = min (nd-1, iofd)

  END FUNCTION

  SUBROUTINE PHI_GENERATE()
    !generate the interpolation table for the kernel using phi
    logical :: lkern
    integer :: nmesh, IOstatus
    
    !ok if the kernel table is here read it
    inquire( file='vdw_kernel.bindat', exist=lkern) 
    IF (lkern) then
      open (unit=10, file='vdw_kernel.bindat', form='unformatted')
      read(10,IOSTAT=IOstatus) nmesh
      IF (nmesh .NE. nd) THEN
        IF (IU0>=0) WRITE(IU0,*)  'nmesh ', nmesh
        IF (IU0>=0) WRITE(IU0,*)  'nd ', nd
        IF (IU0>=0) WRITE(IU0,*)  'kernel differs, recalculating vdW kernel'        
        IF (IU0>=0) WRITE(IU0,*) 'it is strongly recommened to download the file vdw_kernel.bindat'
        IF (IU0>=0) WRITE(IU0,*) '(it is included in the source directory of vasp.5)'
        IF (IU0>=0) WRITE(IU0,*) 'the calculation of the kernel might take days ...'
        CALL GET_KERNEL_TABLE()
      ELSEIF (IOstatus.eq.0) THEN 
        read(10,IOSTAT=IOstatus) dmesh
        read(10,IOSTAT=IOstatus) phi_table
        IF (IOstatus.ne.0) THEN
          IF (IU0>=0) WRITE(IU0,*) 'table not legible, calculating vdW kernel'
          IF (IU0>=0) WRITE(IU0,*) 'it is strongly recommened to download the file vdw_kernel.bindat'
          IF (IU0>=0) WRITE(IU0,*) '(it is included in the source directory of vasp.5)'
          IF (IU0>=0) WRITE(IU0,*) 'the calculation of the kernel might take days ...'
          CALL GET_KERNEL_TABLE()
        ELSE
          IF (IU6>=0) WRITE(IU6,'(A/,(10F8.4))')  'vdW kernel read from vdw_kernel.bindat', dmesh(:) !, phi_table(0,2,:,5)
          IF (IU6>=0) WRITE(IU6,*)
        ENDIF
      ELSE
        IF (IU0>=0) WRITE(IU0,*) 'kernel file not legible, calculating vdW kernel'
        IF (IU0>=0) WRITE(IU0,*) 'it is strongly recommened to download the file vdw_kernel.bindat'
        IF (IU0>=0) WRITE(IU0,*) '(it is included in the source directory of vasp.5)'
        IF (IU0>=0) WRITE(IU0,*) 'the calculation of the kernel might take days ...'
        CALL GET_KERNEL_TABLE()
      ENDIF
      close(10)
    ELSE
      IF (IU0>=0) WRITE(IU0,*)  'No kernel file found, calculating vdW kernel'
      IF (IU0>=0) WRITE(IU0,*)  'it is strongly recommened to download the file vdw_kernel.bindat'
      IF (IU0>=0) WRITE(IU0,*)  '(it is included in the source directory of vasp.5)'
      IF (IU0>=0) WRITE(IU0,*)  'the calculation of the kernel might take days ...'
      CALL GET_KERNEL_TABLE()
    ENDIF
    !this probably covers the posibilities
    !so that we can set this 1
    lhave_phi_table=.true.
    
  END SUBROUTINE
  
  SUBROUTINE GET_KERNEL_TABLE()
    !here we want to find bicubic interpolation coefficients for the vdW kernel
    !and store them in variable 
    !real(q) :: phi_table(0:3,0:3,nd,nd)
    !that is we need to get the values, derivations and cross derivs for the phi(d1,d2)
    !function as defined in Dion et al and RPetS papers.
    INTEGER :: id1,id2
    REAL (q) :: d1, d2, dtot, d1m, d1p, d2m, d2p, dd
    REAL (q) :: val1
    REAL (q) :: phi(nd,nd), phi_der_d1(nd,nd), phi_der_d2(nd,nd), phi_der_d1d2(nd,nd)
    REAL (q) :: ker_d1p_d2, ker_d1m_d2, ker_d1_d2p, ker_d1_d2m 
    REAL (q) :: ker_d1p_d2p, ker_d1m_d2p, ker_d1p_d2m, ker_d1m_d2m
    REAL (q) :: by(4), by_d1(4), by_d2(4), by_d1d2(4), coefs(0:3,0:3)

    !dmesh is read from the kernel file if that exists and is legible
    !in that case this routine is not called and dmesh needs to be done
    IF (.NOT. lhave_nd_points) THEN
      CALL GEN_ND_POINTS()
    ENDIF

    do id1=3,nd-1
      d1=dmesh(id1)
      phi(id1,id1)=GET_KER_VAL(d1,d1)
      IF (IU0>=0) WRITE(IU0,*)  d1, phi(id1,id1)
    enddo

    dd=0.01_q
    !we'll call GET_KER_VAL(d1,d2) to get the phi(d1,d2) value
    DO id1=1,nd
      DO id2=1,id1
        !IF (IU0>=0) WRITE(IU0,*)  id1, id2
        !the thing is symmetric so we can do half the work
        d1 = dmesh(id1)
        d2 = dmesh(id2)
        !value of phi
        phi(id1,id2)=GET_KER_VAL(d1,d2)
        IF (id1.NE.id2) phi(id2,id1)=phi(id1,id2)
        !derivative
        !now dd should be chosen in a sensible way, the minimal values are 0.00 0.27  0.585
        !and the largest 21.5 25.4 30.0
        d1p=d1+dd 
        d1m=d1-dd
        ker_d1p_d2=GET_KER_VAL(d1p,d2)
        ker_d1m_d2=GET_KER_VAL(d1m,d2)
        phi_der_d1(id1,id2)=(ker_d1p_d2 -ker_d1m_d2 )/(2.d0*dd)
        d2p=d2+dd
        d2m=d2-dd
        ker_d1_d2p=GET_KER_VAL(d1,d2p)
        ker_d1_d2m=GET_KER_VAL(d1,d2m)
        phi_der_d2(id1,id2)=(ker_d1_d2p -ker_d1_d2m )/(2.d0*dd)
        !symmetrise
        IF (id1.NE.id2) phi_der_d2(id2,id1)=phi_der_d1(id1,id2)
        IF (id1.NE.id2) phi_der_d1(id2,id1)=phi_der_d2(id1,id2)
        ker_d1p_d2p=GET_KER_VAL(d1p,d2p)
        ker_d1m_d2p=GET_KER_VAL(d1m,d2p)
        ker_d1p_d2m=GET_KER_VAL(d1p,d2m)
        ker_d1m_d2m=GET_KER_VAL(d1m,d2m)
        phi_der_d1d2(id1,id2)=(ker_d1p_d2p+ker_d1m_d2m-ker_d1m_d2p-ker_d1p_d2m)/(4.d0*dd*dd)
        IF (id1.NE.id2) phi_der_d1d2(id2,id1)=phi_der_d1d2(id1,id2)
      ENDDO
      write (*,'(i3)', advance='no') id1 
      IF (IU0>=0) WRITE(IU0,*)  ' '
    ENDDO
    IF (IU0>=0) WRITE(IU0,*)  'done '
    !zero the border components 
    !this actually should be avoided when the asymptotic form of the kernel
    !is included in PHI_INTERPOLATE, for the "nd" parts of the arrays
    IF (.NOT.luse_asymptotic) THEN
      phi(:,nd)=0._q
      phi(nd,:)=0._q
      phi_der_d1(:,nd)=0._q
      phi_der_d1(nd,:)=0._q
      phi_der_d2(:,nd)=0._q
      phi_der_d2(nd,:)=0._q
      phi_der_d1d2(:,nd)=0._q
      phi_der_d1d2(nd,:)=0._q
    ENDIF
    phi(1,nd)=0._q
    phi(nd,1)=0._q
    phi_der_d1(1,nd)=0._q
    phi_der_d1(nd,1)=0._q
    phi_der_d2(1,nd)=0._q
    phi_der_d2(nd,1)=0._q
    phi_der_d1d2(1,nd)=0._q
    phi_der_d1d2(nd,1)=0._q

    phi_der_d1(1,:)=0._q
    phi_der_d2(:,1)=0._q
    phi_der_d1d2(:,1)=0._q
    phi_der_d1d2(1,:)=0._q

    DO id1=1,nd-1
      DO id2 = 1,id1
        by(:) =(/phi(id1,id2),phi(id1+1,id2),phi(id1+1,id2+1),phi(id1,id2+1) /) 
        by_d1(:) =(/phi_der_d1(id1,id2),phi_der_d1(id1+1,id2),phi_der_d1(id1+1,id2+1),phi_der_d1(id1,id2+1) /)
        by_d2(:) =(/phi_der_d2(id1,id2),phi_der_d2(id1+1,id2),phi_der_d2(id1+1,id2+1),phi_der_d2(id1,id2+1) /)
        by_d1d2(:) =(/phi_der_d1d2(id1,id2),phi_der_d1d2(id1+1,id2),phi_der_d1d2(id1+1,id2+1),phi_der_d1d2(id1,id2+1) /)
        CALL BCUCOF(by,by_d1,by_d2,by_d1d2,dd,dd,coefs)
        phi_table(:,:,id1,id2)=coefs
        IF (id1.NE.id2) phi_table(:,:,id2,id1)=coefs
      ENDDO
    ENDDO
    !save the data
    open(10, file='vdw_kernel.bindat',form='unformatted')
    write(10) nd
    write(10) dmesh
    write(10) phi_table
    close(10)

  END SUBROUTINE

  FUNCTION GET_KER_INT(d1,d2)
    !this function integrates the Dion et al kernel using NR p 130. remark
    !the point is to get the phi in some sensible points, fit with spline
    !and then use the integrated 3.3.3[NR]
    REAL (q) :: d1, d2, get_ker_int
    !we need to make grid for a and b, get the integrand on the grid
    !then fit it with splines and integrate using the pita but nice formula
    REAL (q) :: intgrid(nintg), dina(nintg), dinb(nintg), ders(nintg)
    REAL (q) :: a,b,sa,sb,ca,cb,b2,a2,W,T,din
    REAL (q) :: dint1
    INTEGER :: ia,ib, ii
    
    dint1=maxin/nintg
    DO ii=1,nintg
      intgrid(ii)=(ii-1)*dint1+1.d-13
    ENDDO
    dina=0._q
    dinb=0._q

    DO ia=1,nintg
      DO ib=1,nintg
        a=intgrid(ia)
        b=intgrid(ib)
        sa=sin(a)
        sb=sin(b)
        ca=cos(a)
        cb=cos(b)
        a2=a*a
        b2=b*b
        IF ((ia.GT.0).AND.(ib.GT.0)) THEN
          W = 2._q*((3._q-a2)*b*cb*sa+(3._q-b2)*a*ca*sb+(a2+b2-3._q)*sa*sb-3._q*a*b*ca*cb )/(a*b)
        ELSE IF ((ia.EQ.0).AND.(ib.EQ.0)) THEN
          W = 2._q*((3._q-a2)*b*cb+(3._q-b2)*ca*sb+(a2+b2-3._q)*sb-3._q*b*ca*cb )/(b)
        ELSE IF ((ia.GT.0).AND.(ib.EQ.0)) THEN
          W = 2._q*((3._q-a2)*cb*sa+(3._q-b2)*a*ca+(a2+b2-3._q)*sa-3._q*a*ca*cb )/(a)
        ELSE
          W = 2._q*((3._q-a2)*cb+(3._q-b2)*ca+(a2+b2-3._q)-3._q*ca*cb )
        ENDIF
        T = 0.5_q*(1._q/(NY(a,d1)+NY(b,d1))+1._q/(NY(a,d2) +NY(b,d2))) &
         *(1._q/((NY(a,d1)+NY(a,d2))*(NY(b,d1)+NY(b,d2))) + &
         1._q/((NY(a,d1)+NY(b,d2))*(NY(a,d2)+NY(b,d1))))
        dinb(ib) =  W*T
      ENDDO
      call SPLINE(dint1, dinb, nintg, 0._q, 0._q, ders)
      dina(ia)=INTSPL(intgrid, dinb, nintg, ders)
    ENDDO
    call SPLINE(dint1, dina, nintg, 0._q, 0._q, ders)
    get_ker_int=INTSPL(intgrid, dina, nintg, ders)*2._q/pi/pi
       
  END function

  FUNCTION INTSPL(x, y, n, y2)
    REAL (q) :: x(n), y(n), y2(n), intspl
    INTEGER :: n, ii
    intspl=0._q
!    DO ii=1,n-1
!      intspl=intspl+0.5_q*(y(ii)+y(ii+1))*(x(ii+1)-x(ii)) -1._q/24._q*(y2(ii)+y2(ii+1))*(x(ii+1)-x(ii))**3.
!    ENDDO
    DO ii=n-1,1,-1
      intspl=intspl+0.5_q*(y(ii)+y(ii+1))*(x(ii+1)-x(ii)) -1._q/24._q*(y2(ii)+y2(ii+1))*(x(ii+1)-x(ii))**3.
    ENDDO

     
  END FUNCTION
   
  FUNCTION GET_KER_VAL(d1,d2)
  !if dtot is smaller then dsoft the kernel is gently cut
  !this is done by using a function phi_0+dtot**2*phi_2+dtot**4*phi_4
  !the phi_0 is a parameter defined above (0.8), dsoft=0.5
  !so that only phi_2 and phi_4 need to be found
  !this can be done by calculating a value and derivative of the kernel for phi(d1',d2')
  !where d1'/d2'=d1/d2 and sqrt(d1'**2+d2'**2)=dtot, this is done numerically  
  !the phi_2 and phi_4 formulae are then straightforward
  REAL (q) :: d1,d2,dtot, phi_2, phi_4, phi_val, phi_der
  REAL (q) :: phi_pl, phi_mi, d1pl, d2pl, d1mi, d2mi, ddelta
  REAL (q) :: get_ker_val
  
  ddelta = dsoft*0.02_q
  !some sensible value
  dtot = sqrt(d1*d1+d2*d2)
  IF ((dtot .LE. 0._q).OR.(d1.LT.0._q).OR.(d2.LT.0._q)) THEN
    get_ker_val=phi_0 
    RETURN
  ELSE IF (dtot .GE. dsoft) THEN
    get_ker_val=GET_KER_INT(d1,d2)
    RETURN
  ELSE
    d1pl = d1*(dsoft+ddelta)/dtot
    d2pl = d2*(dsoft+ddelta)/dtot
    d1mi = d1*(dsoft-ddelta)/dtot
    d2mi = d2*(dsoft-ddelta)/dtot
    phi_pl = GET_KER_INT(d1pl,d2pl)
    phi_mi = GET_KER_INT(d1mi,d2mi)
    phi_val = 0.5_q*(phi_pl+phi_mi)
    phi_der = (phi_pl-phi_mi)/(2._q*ddelta)
    phi_2 = (4._q*(phi_val-phi_0)-dsoft*phi_der)/(2._q*dsoft*dsoft)
    phi_4 = (phi_0 - phi_val+0.5_q*dsoft*phi_der)/(dsoft**4.)
    get_ker_val=phi_0+phi_2*dtot*dtot +phi_4*dtot**4.
  ENDIF
  END function

  FUNCTION NY(ar,d)
    REAL(q) :: ar, d, ny,y
    REAL(q), parameter :: gamm=4._q*3.1415926535_q/9._q
    IF ((ar.GT.0._q).AND.(d.GT.0._q)) THEN
      y=ar/d
      ny = ar*ar*0.5_q/(1._q-exp(-gamm*y*y))
    ELSE IF ((ar.lt.1.e-10_q).AND.(d.GT.0._q)) THEN
      ny = 0.5_q*d*d/gamm
    ELSE
      ny = 0.5_q*ar*ar
    ENDIF
    RETURN
  END function

  SUBROUTINE BCUCOF(y,y1,y2,y12,d1,d2,c)
  !bicubic spline from NR
    IMPLICIT NONE
    REAL(q), INTENT(IN) :: d1,d2
    REAL(q), DIMENSION(4), INTENT(IN) :: y,y1,y2,y12
    REAL(q), DIMENSION(4,4), INTENT(OUT) :: c
    REAL(q), DIMENSION(16) :: x
    REAL(q), DIMENSION(16,16) :: wt
    DATA wt /1,0,-3,2,4*0,-3,0,9,-6,2,0,-6,4,&
    8*0,3,0,-9,6,-2,0,6,-4,10*0,9,-6,2*0,-6,4,2*0,3,-2,6*0,-9,6,&
    2*0,6,-4,4*0,1,0,-3,2,-2,0,6,-4,1,0,-3,2,8*0,-1,0,3,-2,1,0,-3,&
    2,10*0,-3,2,2*0,3,-2,6*0,3,-2,2*0,-6,4,2*0,3,-2,0,1,-2,1,5*0,&
    -3,6,-3,0,2,-4,2,9*0,3,-6,3,0,-2,4,-2,10*0,-3,3,2*0,2,-2,2*0,&
    -1,1,6*0,3,-3,2*0,-2,2,5*0,1,-2,1,0,-2,4,-2,0,1,-2,1,9*0,-1,2,&
    -1,0,1,-2,1,10*0,1,-1,2*0,-1,1,6*0,-1,1,2*0,2,-2,2*0,-1,1/
    x(1:4)=y
    x(5:8)=y1*d1
    x(9:12)=y2*d2
    x(13:16)=y12*d1*d2
    x=matmul(wt,x)
    c=reshape(x,(/4,4/),order=(/2,1/))
  END SUBROUTINE 

  SUBROUTINE PHIK_FFT(phir_work, phik_work)
    !this performs the FFT of the kernel which is spherically symmetric
    !and thus one just needs to perform sin(kr) transformation in principle
    !times the $k^{-1} r$ factors
    !the sin transforms are used in setlocalpp.F so I guess I copy FOUR1 from there
    !the radial array is [2^N] anyway
    !we need to know Nr, rcut
    REAL (q) :: phir_work(0:nr), phik_work(0:nr)
    REAL (q) :: work(2,0:2*nr)
    REAL (q) :: dr, dk, fac, kval, rval
    INTEGER :: ik, ir, ii
    dr=rcut/Nr
    fac=dr/sqrt(2._q*pi)
    dk=pi/rcut
    work=0._q

    do ii=1, 2*Nr-1
      !this can be loop over half but I'm tired
      if (ii.lt.Nr) then
        ir = ii
        rval = dr*ir
        work(2,ii) = -fac*phir_work(ii)*rval 
      else if (ii.eq.Nr) then
        work(2,ii) = 0._q
      else 
        ir = 2*Nr-ii
        rval = -dr*ir
        work(2,ii) = -fac*phir_work(ir)*rval
      endif 
    enddo
      
    call FOUR1(work, 2*Nr, 1)
    do ik=1, Nr
      kval=ik*dk
      phik_work(ik)=work(1,ik)/kval
    enddo

    !calculate phik_work(0)
    phik_work(0)=0._q
    do ii=1,Nr
      rval=dr*ii
      phik_work(0)=phik_work(0)+rval*rval*phir_work(ii)
    enddo
    phik_work(0)=phik_work(0)*2._q*fac

  END SUBROUTINE   

  SUBROUTINE FOUR1(DATA,NN,ISIGN)
      USE prec
      IMPLICIT NONE
      REAL(q) WR,WI,WPR,WPI,WTEMP,THETA,TEMPI,TEMPR
      REAL(q) DATA(*)
      INTEGER M,ISTEP,MMAX,ISIGN,NN,N,I,J
      N=2*NN
      J=1
      DO 11 I=1,N,2
        IF(J.GT.I)THEN
          TEMPR=DATA(J)
          TEMPI=DATA(J+1)
          DATA(J)=DATA(I)
          DATA(J+1)=DATA(I+1)
          DATA(I)=TEMPR
          DATA(I+1)=TEMPI
        ENDIF
        M=N/2
1       IF ((M.GE.2).AND.(J.GT.M)) THEN
          J=J-M
          M=M/2
        GO TO 1
        ENDIF
        J=J+M
11    CONTINUE
      MMAX=2
2     IF (N.GT.MMAX) THEN
        ISTEP=2*MMAX
        THETA=6.28318530717959_q/(ISIGN*MMAX)
        WPR=-2._q*DSIN(0.5_q*THETA)**2
        WPI=DSIN(THETA)
        WR=1._q
        WI=0._q
        DO 13 M=1,MMAX,2
          DO 12 I=M,N,ISTEP
            J=I+MMAX
            TEMPR=SNGL(WR)*DATA(J)-SNGL(WI)*DATA(J+1)
            TEMPI=SNGL(WR)*DATA(J+1)+SNGL(WI)*DATA(J)
            DATA(J)=DATA(I)-TEMPR
            DATA(J+1)=DATA(I+1)-TEMPI
            DATA(I)=DATA(I)+TEMPR
            DATA(I+1)=DATA(I+1)+TEMPI
12        CONTINUE
          WTEMP=WR
          WR=WR*WPR-WI*WPI+WR
          WI=WI*WPR+WTEMP*WPI+WI
13      CONTINUE
        MMAX=ISTEP
      GO TO 2
      ENDIF
      RETURN
  END SUBROUTINE
     
  FUNCTION S_CUT(val)
    REAL(q) :: val, s_cut
    !cut off function for the kernels, 
    !personally this looks quite harsh
    if (val .le. 0._q) then
      s_cut=1._q
    else if (val .ge. 1) then
      s_cut=0._q
    else
      s_cut=(1._q-val**8)**4
    endif 

  END FUNCTION


  SUBROUTINE SPLINE(dx,y,n,yp1,ypn,y2)
      !from Num recipes
      INTEGER n
      REAL(q) :: yp1,ypn,x(n),y(n),y2(n),dx
      INTEGER i,k
      REAL(q) :: p,qm,sig,un,u(n)

      do i=1,n
       x(i)=dx*i
      enddo
      if (yp1.gt..99e30_q) then
        y2(1)=0._q
        u(1)=0._q
      else
        y2(1)=-0.5_q
        u(1)=(3._q/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do  i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2._q
        y2(i)=(sig-1._q)/p
        u(i)=(6._q*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      enddo
      if (ypn.gt..99e30_q) then
        qm=0._q
        un=0._q
      else
        qm=0.5_q
        un=(3._q/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qm*u(n-1))/(qm*y2(n-1)+1._q)
      do k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
      enddo
      return
  END  SUBROUTINE 


  SUBROUTINE MIN_K(GRIDC,LATT_CUR)
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    INTEGER :: I, N1, N2, N3, NC, ix, iy, iz
    REAL(q) :: cell(3,3)
    REAL(q) :: k_x,k_y,k_z,kte
    !the latt_cur%B has rec vector x stored as 11 21 31
    !and similarly for the rest
    !to get total coordinate x in k-space sum is done over
    !11 12 13  
    N1=GRIDC%NGX
    N2=GRIDC%NGY
    N3=GRIDC%NGZ
    kcut = 10.e10_q
    cell=latt_cur%B*AUTOA
    !now we want to know the smallest boundary k-vector
    !IF (IU0>=0) WRITE(IU0,*)  'reciprocal cell'
    !IF (IU0>=0) WRITE(IU0,*)  cell
    !IF (IU0>=0) WRITE(IU0,*)  N1*cell(1,1), N2*cell(1,2), N3*cell(1,3)
    !IF (IU0>=0) WRITE(IU0,*) N1*cell(2,1),N2*cell(2,2), N3*cell(2,3)
    !IF (IU0>=0) WRITE(IU0,*) N1*cell(3,1), N2*cell(3,2), N3*cell(3,3)

    do ix = -1,1
      do iy = -1,1
        do iz = -1,1
          if ((abs(ix)+abs(iy)+abs(iz)).ne.0 ) then
            k_x=N1*cell(1,1)*ix+N2*cell(1,2)*iy+N3*cell(1,3)*iz
            k_y=N1*cell(2,1)*ix+N2*cell(2,2)*iy+N3*cell(2,3)*iz
            k_z=N1*cell(3,1)*ix+N2*cell(3,2)*iy+N3*cell(3,3)*iz
            kte=2._q*pi*sqrt(k_x*k_x+k_y*k_y+k_z*k_z)*0.5_q
            kcut=min(kcut, kte)   
          endif
        enddo
      enddo
    enddo 

  END SUBROUTINE


  SUBROUTINE GET_K_IND(k_k,GRIDC,LATT_CUR)
    !get the k-space interpolation indices 
    !can be done once
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) k_k(GRIDC%MPLWV)
    INTEGER :: I, N1, N2, N3, NC 
    REAL(q) :: cell(3,3)  
    REAL(q) :: k_x, k_y, k_z
  
    cell=latt_cur%B*AUTOA
    do i=1,gridc%rc%np
      n1=Mod((i-1),gridc%rc%nrow)+1
      nc=(i-1)/gridc%rc%nrow+1
      n2=gridc%rc%i2(nc)
      n3=gridc%rc%i3(nc)
      k_x=gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3)  
      k_y=gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3)  
      k_z=gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3)  
      k_k(i)=2._q*pi*sqrt(k_x*k_x+k_y*k_y+k_z*k_z)
    enddo

  END SUBROUTINE


END MODULE



SUBROUTINE VDW_CALC(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,DINP,spl,phik,d2phidk2,qs_vdw,qsmin,k_k,stress,ISIF,E_corr)
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE setexm

  IMPLICIT NONE
  INTEGER NP
!I guess LATT_CURR is the one that we need not OMEGA only
  TYPE (latt) LATT_CUR
  TYPE (grid_3d) GRIDC
  REAL(q) OMEGA
  REAL(q) EXC
  REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
  REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
  REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
  REAL(q) stress(3,3), E_corr
  INTEGER :: I, IU, alf, bet, ik_k, iofq, ii, jj, ISIF
  INTEGER :: NODE_ME, IONODE, IDUMP
  INTEGER,parameter :: Nalp=30, Nr=4096
  COMPLEX(q) :: CINP(GRIDC%MPLWV)  
  COMPLEX(q), allocatable::   THETA(:,:),f_k(:)
  
  RGRID :: DINP(DIMREAL(GRIDC%MPLWV))
  REAL(q),parameter :: lda_a=0.0310907_q, lda_a1=0.21370_q, lda_b1=7.5957_q
  REAL(q),parameter :: lda_b2=3.5876_q, lda_b3=1.6382_q, lda_b4=0.49294_q
  !REAL(q),parameter :: Zab=-0.8491_q !stored in setexm.inc to use vdw_df2 easily
  REAL(q), parameter :: rcut=100._q
  REAL(q),parameter :: dk_k=pi/rcut
  REAL(q),parameter :: soft_cor=-0.0000000000000_q
  REAL(q) :: s, nrs, rnrs, ex_LDA, ec_LDA, vc_LDA
  REAL(q) :: diff, coef(4)
  REAL(q),parameter :: q0cut=10.0_q, lambda=1.2_q
  REAL(q) :: spl(Nalp,Nalp,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) , allocatable :: q0(:),dq0(:),pa(:),f_g(:,:)
  REAL(q) :: qs_vdw(Nalp), qsmin
  REAL(q) :: k_k(GRIDC%MPLWV)
  REAL(q),allocatable :: dj_k(:)
  REAL(q) :: energ,soft_pot,soft_der
  REAL(q) :: a_k, b_k, a2_k, b2_k, a3_k, b3_k, pref
  REAL(q) :: ro,gr,dq0dn, dq0dgr, dpadq0, dtheadn, dtheadgr
  REAL(q) :: dhdx, pa_tm, vld1, vld2, vld3, vld4, phi, dphidk
  REAL(q) :: k_x(3), k_y, k_z, k_a, cell(3,3)
  REAL(q) :: max_n, max_gr, a_iofq, b_iofq
  REAL(q) :: RINPL, THRD
  RGRID   DWORK(GRIDC%RL%NP) 
  INTEGER,allocatable :: j_k(:),alp(:)
  INTEGER :: n1,n2,n3,nc
  INTEGER ::  i_max_n, i_max_gr
  NP=GRIDC%RL%NP 
  !  IF (IU0>=0) WRITE(IU0,*)  "MPLWV: ", GRIDC%MPLWV, DIMREAL(GRIDC%MPLWV), GRIDC%RC%NP
  ALLOCATE(q0(NP),dq0(NP),alp(NP),pa(NP),f_g(NP,Nalp))
  allocate(THETA(GRIDC%MPLWV,Nalp),f_k(GRIDC%MPLWV))
  allocate(dj_k(GRIDC%MPLWV),j_k(GRIDC%MPLWV))
  cell=latt_cur%B*AUTOA  
  stress=0._q
  RINPL = 1._q/(real(GRIDC%NGX,kind=q)*GRIDC%NGY*GRIDC%NGZ)
  THRD = 1._q/3._q

  energ=0.0_q

#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debugNODE_ME
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      NODE_ME=0
      IONODE =0
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
  !calculate q0 in each point and it's saturated version (Soler's paper)
  DO I=1,NP
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)
      gr=DWORK4(i)
    ENDIF
    s=gr*AUTOA4*0.5_q/ro/(3._q*pi*pi*ro)**THRD/(AUTOA4)
    nrs=(3._q/(4._q*pi*ro*AUTOA3))**THRD
    rnrs=sqrt(nrs)
    ex_LDA=-(3._q*0.25_q/pi)*(9._q*pi*0.25_q)**THRD/nrs
    ec_LDA=-2.0_q*lda_a*(1._q+lda_a1*nrs)*log(1._q+1._q/(2._q*lda_a*(lda_b1*rnrs+lda_b2*nrs+lda_b3*rnrs*nrs+lda_b4*nrs*nrs)))
    q0(i)=(3._q*pi*pi*ro*AUTOA3)**THRD*(1._q+ec_LDA/ex_LDA-Zab_VDW/9._q*s*s)
    !now saturated
    nrs=0._q
    DO iu=1,12
      nrs=nrs+(q0(i)/q0cut)**iu/real(iu,kind=q)
    ENDDO
    q0(i)=q0cut*(1._q-exp(-nrs))
  ENDDO



  !calculate alp and dq0 in each point
  !that is which interpolating point we will be using and how far from it are we
  a_iofq=log((qs_vdw(Nalp)-qs_vdw(Nalp-1))/(qs_vdw(2)-qs_vdw(1)))/(Nalp-2.0_q)
  a_iofq=max(a_iofq,1e-12_q)
  b_iofq=(qs_vdw(2)-qs_vdw(1))/(exp(a_iofq)-1.0_q)
  DO I=1,NP
    iofq=1+log(1+(q0(i)-qs_vdw(1))/b_iofq)/a_iofq
    iofq=max(1,iofq)
    iofq=min(Nalp,iofq)
    IF(q0(i).GT.(q0cut-0.00015_q)) iofq=Nalp
    alp(i)=iofq
  ENDDO
  !this is the difference from interpolating point
  DO I=1,NP
    dq0(i)=q0(i)-qs_vdw(alp(i))
  ENDDO

  !for each interpolating point create p_\alpha (Soler's paper)
  !and Fourier transform it, this is stored in \Theta array
  DO alf=1,Nalp
    DO I=1,NP
      coef(:)=spl(alp(i),alf,:)
      diff=dq0(i)
      pa(i)=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
    ENDDO
    DO I=1,NP
      IF (DCHARG(i).ge.0._q) THEN
        dinp(i)=pa(i)*DCHARG(i)*AUTOA3
      ELSE
        dinp(i)=0._q
      ENDIF
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,-1)
    DO I=1,GRIDC%RC%NP
      THETA(i,alf)=CINP(i)
    ENDDO
  ENDDO

  !find values of the kernel on the reciprocal grid
  DO i=1,GRIDC%RC%NP
    dj_k(i)=0.5_q*k_k(i)/pi*rcut
    j_k(i)=floor(dj_k(i))
    dj_k(i)=dj_k(i)-j_k(i)
    dj_k(i)=dj_k(i)*tpi/rcut
  ENDDO

  !now we are going to calculate the energy
  !this requires two nested loops over \alpha and \beta
  !first we do the sum over \Theta_\beta * kernel
  DO alf=1,Nalp
    f_k=0._q
    DO bet=1,Nalp
      DO i=1,GRIDC%RC%NP
        ik_k=k_k(i)/dk_k
        IF (ik_k.GE.Nr) THEN
          !I really don't like the if here
          phi=0._q
          dphidk=0._q
        ELSE
          a_k=((ik_k+1._q)*dk_k-k_k(i))/dk_k
          b_k=1._q-a_k
          a2_k=(3._q*a_k**2.-1._q)*dk_k/6._q
          b2_k=(3._q*b_k**2.-1._q)*dk_k/6._q
          a3_k=(a_k**3.-a_k)*dk_k**2._q/6._q
          b3_k=(b_k**3.-b_k)*dk_k**2._q/6._q
          phi=a_k*phik(ik_k, bet, alf)+b_k*phik(ik_k+1,bet,alf) &
           +a3_k*d2phidk2(ik_k,bet,alf)+b3_k*d2phidk2(ik_k+1,bet,alf)
          dphidk=(-phik(ik_k, bet, alf)+phik(ik_k+1,bet,alf))/dk_k &
                 -a2_k*d2phidk2(ik_k,bet,alf)+b2_k*d2phidk2(ik_k+1,bet,alf)

        ENDIF
        f_k(i)=f_k(i)+theta(i,bet)*phi
        IF (ISIF .ge. 1) THEN
#ifdef NGZhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        if (k_a .gt. 1.0e-13_q) then
          if ((n3.eq.1).or.(n3.eq.(GRIDC%NGZ/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj) 
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj) 
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                  (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
#ifdef NGXhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        if (k_a .gt. 1.0e-13_q) then
          if ((n1.eq.1).or.(n1.eq.(GRIDC%NGX/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)                                  
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)  
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) * &
                            (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif
        ENDIF !only calculate stress when required

      ENDDO !q points
    ENDDO  !beta



#ifdef NGZhalf
!for MPI version reduced in Z direction
    !and then we sum it with \Theta_\alpha
    DO i=1,GRIDC%RC%NP
      nc=(i-1)/gridc%rc%nrow+1
      n3=gridc%rc%i3(nc)
      IF (n3.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n3.EQ.(GRIDC%NGZ/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q) 
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
#ifdef NGXhalf
!for serial version reduced in X direction
    DO i=1,GRIDC%RC%NP
      n1=Mod((i-1),gridc%rc%nrow)+1
      IF (n1.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n1.EQ.(GRIDC%NGX/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif

    !now we fourier transform the \Theta * kernel to real space to reuse it for potential
    DO i=1,GRIDC%RC%NP
      CINP(I)=f_k(i)
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,1)
    DO i=1,GRIDC%RL%NP
      f_g(i,alf)=dinp(i)*RINPL
    ENDDO
  ENDDO

  !add the energy to the xc energy, include volume factor
  !  number of grid points since we've done FFT, and factors to get eV
  EXC=EXC + 0.5_q*energ*LATT_CUR%OMEGA*RINPL/AUTOA3*2._q*RYTOEV
  !compared to vander (my local version (jik)), 
  !we divide by (M)PLWV only once since it's done in xcgrad for the second time
  !so we can say that we get rid of the FFT 1/N factor and keep the \Omega/N for xcgrad
  E_corr=0.5_q*energ*LATT_CUR%OMEGA*RINPL**2./AUTOA3*2._q*RYTOEV

  max_n=0._q
  max_gr=0._q

  DO i=1,gridc%RL%NP
    !calc the potential 
    !here we calculate the lda_c potential, and derivatives of q0 wrt density and density gradient 
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE 
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF
    nrs=(3._q/(4._q*pi*ro))**THRD
    rnrs=sqrt(nrs)
    vld1=lda_b1*rnrs+lda_b2*nrs+lda_b3*rnrs*nrs+lda_b4*nrs*nrs 
    vld2=2._q*lda_a*vld1
    vld3=0.5_q*lda_b1/rnrs+lda_b2+1.5_q*lda_b3*rnrs+2._q*lda_b4*nrs
    vld4=log(1._q+1._q/vld2)
    ec_LDA=-2.0_q*lda_a*(1._q+lda_a1*nrs)*vld4
    vc_LDA=-2.0_q*lda_a*lda_a1*vld4+2.0_q*lda_a*(1._q+lda_a1*nrs)*vld3/(vld1*(vld2+1._q))
    vc_LDA=(ec_LDA+ro*vc_LDA*(-THRD)*(3._q/(4._q*pi))**THRD*ro**(-4._q*THRD))
    ec_LDA=ec_LDA*ro
    dq0dn=(pi/(3._q*ro))**(2._q*THRD)+4._q*pi*THRD*(ec_LDA/ro-vc_LDA)/ro+7._q/108._q*Zab_VDW*gr*gr*ro**(-10._q/3._q)/(pi*pi*3._q)**THRD
    dq0dgr=-Zab_VDW/(18._q*(3._q*pi*pi*ro)**THRD)*gr/(ro*ro)
    nrs=0._q
    rnrs=0._q
    DO iu=1,12
      rnrs=rnrs+(q0(i)/q0cut)**iu/real(iu,kind=q)
    ENDDO
    DO iu=0,11
      nrs=nrs+(q0(i)/q0cut)**iu
    ENDDO
    dhdx=nrs*exp(-rnrs)
    dtheadn=0._q
    dtheadgr=0._q
    !one loop is needed to calculate the potential
    DO alf=1,Nalp
      coef=spl(alp(i),alf,:)
      diff=dq0(i)
      pa_tm=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
      dpadq0=coef(2)+diff*(2._q*coef(3)+3._q*diff*coef(4))
      dtheadn=dtheadn+(pa_tm+ro*dpadq0*dq0dn*dhdx)*f_g(i,alf)
      dtheadgr=dtheadgr+ro*dpadq0*dhdx*dq0dgr*f_g(i,alf)
    ENDDO

    !add the potentials to the GGA ones
    !this is d/dn
    DWORKG(i)=DWORKG(i)+dtheadn*2._q*RYTOEV  
    !this is d/ddn
    DWORK(i)=DWORK(i)+dtheadgr*2._q*RYTOEV/ MAX(DWORK4(I),1.E-10_q)*AUTOA    
  ENDDO !over points on real space grid

  DEALLOCATE(q0,dq0,alp,pa)
  DEALLOCATE(dj_k,theta,f_k,f_g,j_k)
  stress=0.5_q*stress*LATT_CUR%OMEGA*RINPL**2.*2._q*RYTOEV/AUTOA3


#ifdef MPI
 CALLMPI( M_sum_d(GRIDC%COMM,e_corr,1))
#endif

! IF (NODE_ME == IONODE) THEN
!   WRITE(*,'(A,F14.7)') 'Total vdW correction in eV: ', e_corr
! ENDIF

END SUBROUTINE

SUBROUTINE VDW_CALC_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,DINP,spl,phik,d2phidk2,qs_vdw,qsmin,k_k,stress,ISIF,E_corr)
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE setexm

  IMPLICIT NONE
  INTEGER NP
  TYPE (latt) LATT_CUR
  TYPE (grid_3d) GRIDC
  REAL(q) OMEGA
  REAL(q) EXC
  REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
  REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
  REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
  REAL(q) stress(3,3),E_corr
  INTEGER :: I, IU, alf, bet, ik_k, iofq, ii, jj, ISIF
  INTEGER :: NODE_ME, IONODE, IDUMP
  INTEGER,parameter :: Nalp=30, Nr=4096
  COMPLEX(q) :: CINP(GRIDC%MPLWV)  
  COMPLEX(q), allocatable::   THETA(:,:),f_k(:)

  RGRID :: DINP(DIMREAL(GRIDC%MPLWV))
  REAL(q),parameter :: lda_a=0.0310907_q, lda_a1=0.21370_q, lda_b1=7.5957_q
  REAL(q),parameter :: lda_b2=3.5876_q, lda_b3=1.6382_q, lda_b4=0.49294_q
! REAL(q),parameter :: Zab=-0.8491_q !read from INCAR for easy-of-use vdw-df2
  REAL(q),parameter :: dk_k=pi/100._q
  REAL(q),parameter :: soft_cor=-0.0000000000000_q
  REAL(q) :: s, nrs, rnrs, ex_LDA, ec_LDA, vc_LDA
  REAL(q) :: diff, coef(4)
  REAL(q),parameter :: q0cut=10.0_q, lambda=1.2_q, rcut=100._q
  REAL(q) :: spl(Nalp,Nalp,4)
!  REAL(q) :: phi_ab(Nalp,Nalp,0:Nr/2-1,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) , allocatable :: q0(:),dq0(:),pa(:),f_g(:,:)
  REAL(q) :: qs_vdw(Nalp), qsmin
  REAL(q) :: k_k(GRIDC%MPLWV)
  REAL(q),allocatable :: dj_k(:)
  REAL(q) :: energ,soft_pot,soft_der
  REAL(q) :: a_k, b_k, a2_k, b2_k, a3_k, b3_k, pref
  REAL(q) :: ro,gr,dq0dn, dq0dgr, dpadq0, dtheadn, dtheadgr
  REAL(q) :: dhdx, pa_tm, vld1, vld2, vld3, vld4, phi, dphidk
  REAL(q) :: k_x(3), k_y, k_z, k_a, cell(3,3)
  REAL(q) :: max_n, max_gr, a_iofq, b_iofq
  RGRID   DWORK(GRIDC%RL%NP)
  INTEGER,allocatable :: j_k(:),alp(:)
  INTEGER :: n1,n2,n3,nc
  INTEGER ::  i_max_n, i_max_gr
  NP=GRIDC%RL%NP
  ALLOCATE(q0(NP),dq0(NP),alp(NP),pa(NP),f_g(NP,Nalp))
  allocate(THETA(GRIDC%MPLWV,Nalp),f_k(GRIDC%MPLWV))
  allocate(dj_k(GRIDC%MPLWV),j_k(GRIDC%MPLWV))
  stress=0._q
  cell=latt_cur%B*AUTOA

  energ=0.0_q

#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debugNODE_ME
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      NODE_ME=0
      IONODE =0
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
  DO I=1,NP
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)
      gr=DWORK4(i)
    ENDIF
    s=gr*AUTOA4*0.5_q/ro/(3._q*pi*pi*ro)**(1._q/3._q)/(AUTOA4)
    nrs=(3._q/(4._q*pi*ro*AUTOA3))**(1._q/3._q)
    rnrs=sqrt(nrs)
    ex_LDA=-(3._q*0.25_q/pi)*(9._q*pi*0.25_q)**(1._q/3._q)/nrs
    ec_LDA=-2.0_q*lda_a*(1._q+lda_a1*nrs)*log(1._q+1._q/(2._q*lda_a*(lda_b1*rnrs+lda_b2*nrs+lda_b3*rnrs*nrs+lda_b4*nrs*nrs)))
    q0(i)=(3._q*pi*pi*ro*AUTOA3)**(1._q/3._q)*(1._q+ec_LDA/ex_LDA-Zab_VDW/9._q*s*s)
    !now saturated
    nrs=0._q
    DO iu=1,12
      nrs=nrs+(q0(i)/q0cut)**iu/real(iu,q)
    ENDDO
      q0(i)=q0cut*(1._q-exp(-nrs))
  ENDDO

  !calculate alp and dq0 in each point
  a_iofq=log((qs_vdw(Nalp)-qs_vdw(Nalp-1))/(qs_vdw(2)-qs_vdw(1)))/(Nalp-2.0_q)
  a_iofq=max(a_iofq,1e-12_q)
  b_iofq=(qs_vdw(2)-qs_vdw(1))/(exp(a_iofq)-1.0_q)
  DO I=1,NP
    iofq=1+log(1+(q0(i)-qs_vdw(1))/b_iofq)/a_iofq
    iofq=max(1,iofq)
    iofq=min(Nalp,iofq)
    !!!
    IF(q0(i).GT.9.99985_q) iofq=Nalp
    alp(i)=iofq
  ENDDO
  DO I=1,NP
    dq0(i)=q0(i)-qs_vdw(alp(i))
  ENDDO


  DO alf=1,Nalp
    DO I=1,NP
      coef(:)=spl(alp(i),alf,:)
      diff=dq0(i)
      pa(i)=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
    ENDDO
    DO I=1,NP
      dinp(i)=pa(i)*DCHARG(i)*AUTOA3
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,-1)
    DO I=1,GRIDC%RC%NP
      THETA(i,alf)=CINP(i)
    ENDDO
  ENDDO

  DO i=1,GRIDC%RC%NP
    dj_k(i)=0.5_q*k_k(i)/pi*rcut
    j_k(i)=floor(dj_k(i))
    dj_k(i)=dj_k(i)-j_k(i)
    dj_k(i)=dj_k(i)*2._q*pi/rcut
  ENDDO

  DO alf=1,Nalp
    f_k=0._q

    DO bet=1,Nalp
      DO i=1,GRIDC%RC%NP
        ik_k=k_k(i)/dk_k
        IF (ik_k.GE.Nr) THEN
          !I really don't like the if here
          phi=0._q
        ELSE
          a_k=((ik_k+1._q)*dk_k-k_k(i))/dk_k
          b_k=1._q-a_k
          a2_k=(3._q*a_k**2.-1._q)*dk_k/6._q
          b2_k=(3._q*b_k**2.-1._q)*dk_k/6._q
          a3_k=(a_k**3.-a_k)*dk_k**2._q/6._q
          b3_k=(b_k**3.-b_k)*dk_k**2._q/6._q
          phi=a_k*phik(ik_k, bet, alf)+b_k*phik(ik_k+1,bet,alf)&
              +a3_k*d2phidk2(ik_k,bet,alf)+b3_k*d2phidk2(ik_k+1,bet,alf)
          dphidk=(-phik(ik_k, bet, alf)+phik(ik_k+1,bet,alf))/dk_k &
                 -a2_k*d2phidk2(ik_k,bet,alf)+b2_k*d2phidk2(ik_k+1,bet,alf)

        ENDIF
        f_k(i)=f_k(i)+theta(i,bet)*phi
        IF (ISIF .ge. 1) THEN
#ifdef NGZhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        pref=dphidk/k_a
        if (k_a .gt. 1.0e-13_q) then
          if ((n3.eq.1).or.(n3.eq.(GRIDC%NGZ/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)                                  
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)  
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                 (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
               
              enddo
            enddo
          endif
        endif
#else
#ifdef NGXhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        pref=dphidk/k_a
        if (k_a .gt. 1.0e-13_q) then
          if ((n1.eq.1).or.(n1.eq.(GRIDC%NGX/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                 (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif
        ENDIF !only calculate stress when required

      ENDDO
    ENDDO !beta

#ifdef NGZhalf
!for MPI version reduced in Z direction
    !and then we sum it with \Theta_\alpha
    DO i=1,GRIDC%RC%NP
      nc=(i-1)/gridc%rc%nrow+1
      n3=gridc%rc%i3(nc)
      IF (n3.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n3.EQ.(GRIDC%NGZ/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
#ifdef NGXhalf
!for serial version reduced in X direction
    DO i=1,GRIDC%RC%NP
      n1=Mod((i-1),gridc%rc%nrow)+1
      IF (n1.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n1.EQ.(GRIDC%NGX/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif


    DO i=1,GRIDC%RC%NP
      CINP(I)=f_k(i)
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,1)
    DO i=1,GRIDC%RL%NP
      f_g(i,alf)=dinp(i)/(real(GRIDC%NGX,q)*GRIDC%NGY*GRIDC%NGZ)
    ENDDO
  ENDDO !alf

  EXC=EXC + 0.5_q*energ*LATT_CUR%OMEGA/(real(GRIDC%NGX,q)*GRIDC%NGY*GRIDC%NGZ)/AUTOA3*2._q*RYTOEV
  !compared to vander, we divide by (M)PLWV only once since it's done in xcgrad for the second time
  !so we can say that we get rid of the FFT 1/N factor and keep the \Omega/N for xcgrad
  E_corr=0.5_q*energ*LATT_CUR%OMEGA/(real(GRIDC%NGX)*GRIDC%NGY*GRIDC%NGZ)**2./AUTOA3*2._q*RYTOEV

  max_n=0.d0
  max_gr=0.d0

  DO i=1,gridc%RL%NP
  !calc the potential
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF
    nrs=(3._q/(4._q*pi*DCHARG(i)*AUTOA3))**(1._q/3._q)
    rnrs=sqrt(nrs)
    vld1=lda_b1*rnrs+lda_b2*nrs+lda_b3*rnrs*nrs+lda_b4*nrs*nrs
    vld2=2._q*lda_a*vld1
    vld3=0.5_q*lda_b1/rnrs+lda_b2+1.5_q*lda_b3*rnrs+2._q*lda_b4*nrs
    vld4=log(1._q+1._q/vld2)
    ec_LDA=-2.0_q*lda_a*(1._q+lda_a1*nrs)*vld4
    vc_LDA=-2.0_q*lda_a*lda_a1*vld4+2.0_q*lda_a*(1._q+lda_a1*nrs)*vld3/(vld1*(vld2+1._q))
    vc_LDA=(ec_LDA+ro*vc_LDA*(-1._q/3._q)*(3._q/(4._q*pi))**(1./3.)*ro**(-4._q/3._q))
    ec_LDA=ec_LDA*ro
    dq0dn=(pi/(3._q*ro))**(2._q/3._q)+4._q*pi/3._q*(ec_LDA/ro-vc_LDA)/ro+7._q/108._q*Zab_VDW*gr*gr*ro**(-10._q/3._q)/(pi*pi*3._q)**(1._q/3._q)
    dq0dgr=-Zab_VDW/(18._q*(3._q*pi*pi*ro)**(1._q/3._q))*gr/(ro*ro)
    nrs=0._q
    rnrs=0._q
    DO iu=1,12
      rnrs=rnrs+(q0(i)/q0cut)**iu/real(iu,q)
    ENDDO
    DO iu=0,11
      nrs=nrs+(q0(i)/q0cut)**iu
    ENDDO
    dhdx=nrs*exp(-rnrs)
    dtheadn=0._q
    dtheadgr=0._q
    DO alf=1,Nalp
      coef=spl(alp(i),alf,:)
      diff=dq0(i)
      pa_tm=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
      dpadq0=coef(2)+diff*(2._q*coef(3)+3._q*diff*coef(4))
      dtheadn=dtheadn+(pa_tm+ro*dpadq0*dq0dn*dhdx)*f_g(i,alf)
      dtheadgr=dtheadgr+ro*dpadq0*dhdx*dq0dgr*f_g(i,alf)
    ENDDO

    !this is d/dn
    DWORKG(i)=DWORKG(i)+dtheadn*2._q*RYTOEV
    !this is d/ddn
    DWORK(i)=DWORK(i)+dtheadgr*2._q*RYTOEV* AUTOA
    ! in the spin polarized case the gradient is used later MAX(DWORK4(I),1.E-10_q)

  ENDDO

  DEALLOCATE(q0,dq0,alp,pa)
  DEALLOCATE(dj_k,theta,f_k,f_g,j_k)
  stress=0.5_q*stress*LATT_CUR%OMEGA/(real(GRIDC%NGX,kind=q)*GRIDC%NGY*GRIDC%NGZ)**2.*2._q*RYTOEV/AUTOA3  

#ifdef MPI
  CALLMPI( M_sum_d(GRIDC%COMM,e_corr,1))
#endif

! IF (NODE_ME == IONODE) then
!   WRITE(*,*) 'Total vdW correction in eV: ', e_corr
! ENDIF

END SUBROUTINE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!
!!! rVV10
!!!
!!! 2015/05/14
!!! Modified by Zenghui Yang, and Haowei Peng
!!! to evaluate the rVV10 non-local correlation functional of Phys. Rev. B 87, 041108(R) (2013),
!!! which is a simplified/revised version of the VV10 functional of J. Chem. Phys. 133, 244103 (2010).
!!!
!!! Please cite
!!!     Haowei Peng*, Zeng-Hui Yang, Jianwei Sun, and John P. Perdew,
!!!     "Versatile van der Waals Density Functional Based on a Meta-Generalized
!!!      Gradient Approximation", PRX 6, 041005 (2016).
!!!                                                                                                                                                                        
!!!     Haowei Peng* and John P. Perdew,
!!!     "Rehabilitation of the Perdew-Burke-Ernzerhof generalized gradient approximation
!!!      for layered materials", PRB 95, 081105(R) (2017).
!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE VDW_rVV10
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE setexm
  USE vdw_df_helper, ONLY : IU0,IU6,ISIF

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: VDW_NONLOC_RVV10,VDW_NONLOC_RVV10_spin, &
            Nalp,Nr,q0cut,rcut

! REAL(q),parameter :: bparam = 6.3_q !rVV10, not the 5.9 of VV10
! REAL(q),parameter :: betaparam = 0.03125_q * (3.0_q / (bparam**2.0_q) )**(0.75_q) * 2._q*RYTOEV  ! Already in eV 
! REAL(q),parameter :: Cparam = 0.0093_q

  INTEGER, parameter :: Nalp=24, Nr=2048
  REAL(q), parameter :: qAlp_ratio=2048._q
  REAL(q), parameter :: q0cut=0.5_q
  REAL(q), parameter :: rcut = 100._q
  REAL(q) :: spl(Nalp,Nalp,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) :: phir(0:Nr,Nalp,Nalp)
  REAL(q) :: qs_vdw(Nalp)
  REAL(q) :: dr

  LOGICAL :: INIT_VDW_LL=.TRUE. !do we need to initialise the stuff?
  LOGICAL :: LREAD_KERN = .FALSE.
  logical :: lhave_alpha_table = .FALSE.

  REAL(q) :: kcut

CONTAINS

  SUBROUTINE VDW_NONLOC_RVV10(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,stress)
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
    REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
    REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input !df/d(drho)/(drho)potential on output
    REAL(q) EXC
    REAL(q) stress(3,3)
    COMPLEX(q), allocatable :: CINP(:) !, THETA(:,:)
    REAL(q) ,allocatable :: k_k(:)
    RGRID   DWORK(GRIDC%RL%NP) 

    IF (INIT_VDW_LL) THEN
      call MIN_K(GRIDC,LATT_CUR)
      call GET_SPLINES
      call GET_KERNELS
      INIT_VDW_LL=.FALSE.
    ENDIF

    ALLOCATE(K_K(GRIDC%MPLWV))
    call GET_K_IND(k_k,GRIDC,LATT_CUR)

    ALLOCATE(CINP(GRIDC%MPLWV)) 
    CALL VDW_CALC_RVV10(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,CINP,spl, phik, d2phidk2,qs_vdw,k_k,stress,ISIF)
    DEALLOCATE(CINP) 
    DEALLOCATE(k_k)   
  END SUBROUTINE 

  SUBROUTINE VDW_NONLOC_RVV10_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,stress)
    !spin polarised version of the previous routine 
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
    REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
    REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
    REAL(q) EXC
    REAL(q) stress(3,3) !this guy is not yet calculated and passed to xcspin.F
    COMPLEX(q), allocatable :: CINP(:) !, THETA(:,:)
    REAL(q) ,allocatable :: k_k(:)
    RGRID   DWORK(GRIDC%RL%NP)

    IF (INIT_VDW_LL) THEN
      call MIN_K(GRIDC, LATT_CUR)
      call GET_SPLINES
      call GET_KERNELS
      INIT_VDW_LL=.FALSE.
    ENDIF

    ALLOCATE(K_K(GRIDC%MPLWV))
    call GET_K_IND(k_k,GRIDC,LATT_CUR)

    ALLOCATE(CINP(GRIDC%MPLWV))
    CALL VDW_CALC_RVV10_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,CINP,spl, phik, d2phidk2,qs_vdw,k_k,stress,ISIF)

    DEALLOCATE(CINP)
    DEALLOCATE(k_k)
  END SUBROUTINE

  SUBROUTINE GET_SPLINES
    !construct the spline coefficients for interpolation
    REAL(q) :: a(Nalp,Nalp), c(Nalp,Nalp), l(Nalp,Nalp), mi(Nalp,Nalp), z(Nalp,Nalp)
    REAL(q) :: b(Nalp-1,Nalp), d(Nalp-1,Nalp), h(Nalp-1), al(Nalp-1,Nalp)
    INTEGER :: i,j

    a=0._q
    b=0._q
    d=0._q
    h=0._q
    al=0._q
    c=0._q
    l=0.0_q
    mi=0._q
    z=0._q
    
    call GEN_ALPHA_POINTS()

    DO i=1,Nalp
      a(i,i)=1._q
    ENDDO
    DO i=1,Nalp-1
      h(i)=qs_vdw(i+1)-qs_vdw(i)
    ENDDO
    DO i=2,Nalp-1
      DO j=1,Nalp
        al(i,j)=3._q/h(i)*(a(i+1,j)-a(i,j))-3._q/h(i-1)*(a(i,j)-a(i-1,j))
      ENDDO
    ENDDO
    l(1,:)=1._q
    DO i=2,Nalp-1
      l(i,:)=2._q*(qs_vdw(i+1)-qs_vdw(i-1))-h(i-1)*mi(i-1,:)
      mi(i,:)=h(i)/l(i,:)
      z(i,:)=(al(i,:)-h(i-1)*z(i-1,:))/l(i,:)
    ENDDO
    l(Nalp,:)=1._q
    z(Nalp,:)=0._q
    c(Nalp,:)=0._q
    DO i=Nalp-1,1,-1
      c(i,:)=z(i,:)-mi(i,:)*c(i+1,:)
      b(i,:)=(a(i+1,:)-a(i,:))/h(i)-h(i)/3._q*(c(i+1,:)+2._q*c(i,:))
      d(i,:)=(c(i+1,:)-c(i,:))/(3._q*h(i))
    ENDDO

    spl(:,:,1)=a(:,:)
    spl(:,:,3)=c(:,:)
    DO i=1,Nalp-1
      spl(i,:,2)=b(i,:)
      spl(i,:,4)=d(i,:)
    ENDDO

  END SUBROUTINE


  SUBROUTINE GET_KERNELS
    !this procedure calculates the fourier transformed vdw kernels
    IMPLICIT NONE
    INTEGER :: alf,bet,i,j,irad, nk, ik
    REAL (q) :: q1,q2, dk, krad
    REAL (q) :: r(0:nr)
    REAL (q) :: phir_work(0:nr), phik_work(0:nr)
    REAL (q) :: d2phik_work(0:nr)
    
    !setting up the mesh of interpolating points
    !call GEN_ALPHA_POINTS()
    dr= rcut/Nr
    dk= pi/rcut
    nk= int(kcut/dk)+1

    !find the interpolation coefficients for the alpha points
    DO alf = 1, Nalp
      DO bet = 1, alf
        phik_work=0._q
        phir_work=0._q
        !desaturate q interpolation points
        q1=DESATURATE(qs_vdw(alf))
        q2=DESATURATE(qs_vdw(bet))

        !get the kernel on a real grid by using the precalculated interpolation
        !data
        DO irad = 0, Nr
          r(irad) = irad * dr
          phir(irad,alf,bet) = PHI_VALUE(q1*(r(irad)**2), q2*(r(irad)**2))
        ENDDO
       
        !smoothen for large r
        DO irad =0, Nr 
          phir(irad,alf,bet)=phir(irad,alf,bet)*S_CUT(r(irad)/rcut)
        ENDDO

        !fourier transform it and smoothen
        phir_work(:)=phir(:,alf,bet)
        call PHIK_FFT(phir_work(:), phik_work(:))
        phik(:,alf,bet)=phik_work*(2._q*pi)**1.5_q
        phik(nk:nr,alf,bet)=0._q
        DO ik =1,nk
          krad=ik*dk
          phik(ik,alf,bet)=phik(ik,alf,bet)*S_CUT(krad/kcut)
        ENDDO

        !get coefficients for cubic splines
        !phik_work to splines    
        call SPLINE(dk,phik(:,alf,bet),Nr+1,0._q,0._q,d2phik_work)

        !symmetrise
        phik(:,bet,alf)=phik(:,alf,bet)
        d2phidk2(:,bet,alf)=d2phik_work
        d2phidk2(:,alf,bet)=d2phik_work
      ENDDO
    ENDDO

  END SUBROUTINE
  
  SUBROUTINE GEN_ALPHA_POINTS()
    !find the interpolation points for the kernel interpolation q_0 grid
    !i.e. fill the qs_vdw array
    REAL(q) :: a,b,e
    INTEGER :: i

    qs_vdw(:)=0._q
    a=log(qAlp_ratio)/(Nalp-1._q)
    b=q0cut/(qAlp_ratio-1._q)
    
    DO i=2,Nalp-1    
      qs_vdw(i)=b*exp(a*(i-1._q))-b
    ENDDO
    qs_vdw(Nalp)=q0cut
    lhave_alpha_table = .true.

  END SUBROUTINE 

  FUNCTION DESATURATE(x)
    !find q0 corresponding to saturated q0 by interval halving 
    REAL(q) :: desaturate, x
    REAL(q) :: xdo, xup, xme, xmesat
    REAL(q), parameter :: xtol=1.e-14_q
    
    !check first
    IF ((x<0._q).OR.(x>q0cut)) then
      IF (IU0>=0) WRITE(IU0,*)   'DESATURATE: out of range', x
      stop
    endif
    !this is done for the original q0 values before saturation
    xdo =  0._q
    xup = q0cut*5._q
    DO
      xme = (xdo+xup)*0.5_q
      xmesat = SATURATE(xme)
      !if smaller 
      IF (abs(xmesat-x).LE.xtol ) THEN 
        desaturate=xup
        RETURN
      ELSE IF (x.LE.xmesat) THEN
        xup=xme
      ELSE
        xdo=xme
      ENDIF
    ENDDO

  END FUNCTION

  FUNCTION SATURATE(x)
    !the same that we use down in the main routine
    !R-P&S' equation No 4 or so
    REAL(q):: saturate, x, coe
    INTEGER :: iu

    coe=0._q
    DO iu=1,12
      coe=coe+(x/q0cut)**iu/real(iu,kind=q)
    ENDDO
    saturate=q0cut*(1._q-exp(-coe))
  END FUNCTION

  FUNCTION PHI_VALUE(d1, d2)
    !finds the kernel using phi_generate
    !returns the value of kernel for d1, d2
    REAL (q) :: phi_value
    REAL (q) :: d1, d2
    
    phi_value = -1.5_q/((d1+1._q)*(d2+1._q)*(d1+d2+2._q))
    
  END FUNCTION

  SUBROUTINE PHIK_FFT(phir_work, phik_work)
    !this performs the FFT of the kernel which is spherically symmetric
    !and thus one just needs to perform sin(kr) transformation in principle
    !times the $k^{-1} r$ factors
    !the sin transforms are used in setlocalpp.F so I guess I copy FOUR1 from there
    !the radial array is [2^N] anyway
    !we need to know Nr, rcut
    REAL (q) :: phir_work(0:nr), phik_work(0:nr)
    REAL (q) :: work(2,0:2*nr)
    REAL (q) :: dr, dk, fac, kval, rval
    INTEGER :: ik, ir, ii
    dr=rcut/Nr
    fac=dr/sqrt(2._q*pi)
    dk=pi/rcut
    work=0._q

    do ii=1, 2*Nr-1
      !this can be loop over half but I'm tired
      if (ii.lt.Nr) then
        ir = ii
        rval = dr*ir
        work(2,ii) = -fac*phir_work(ii)*rval 
      else if (ii.eq.Nr) then
        work(2,ii) = 0._q
      else 
        ir = 2*Nr-ii
        rval = -dr*ir
        work(2,ii) = -fac*phir_work(ir)*rval
      endif 
    enddo
      
    call FOUR1(work, 2*Nr, 1)
    do ik=1, Nr
      kval=ik*dk
      phik_work(ik)=work(1,ik)/kval
    enddo

    !calculate phik_work(0)
    phik_work(0)=0._q
    do ii=1,Nr
      rval=dr*ii
      phik_work(0)=phik_work(0)+rval*rval*phir_work(ii)
    enddo
    phik_work(0)=phik_work(0)*2._q*fac

  END SUBROUTINE   

  SUBROUTINE FOUR1(DATA,NN,ISIGN)
      USE prec
      IMPLICIT NONE
      REAL(q) WR,WI,WPR,WPI,WTEMP,THETA,TEMPI,TEMPR
      REAL(q) DATA(*)
      INTEGER M,ISTEP,MMAX,ISIGN,NN,N,I,J
      N=2*NN
      J=1
      DO 11 I=1,N,2
        IF(J.GT.I)THEN
          TEMPR=DATA(J)
          TEMPI=DATA(J+1)
          DATA(J)=DATA(I)
          DATA(J+1)=DATA(I+1)
          DATA(I)=TEMPR
          DATA(I+1)=TEMPI
        ENDIF
        M=N/2
1       IF ((M.GE.2).AND.(J.GT.M)) THEN
          J=J-M
          M=M/2
        GO TO 1
        ENDIF
        J=J+M
11    CONTINUE
      MMAX=2
2     IF (N.GT.MMAX) THEN
        ISTEP=2*MMAX
        THETA=6.28318530717959_q/(ISIGN*MMAX)
        WPR=-2._q*DSIN(0.5_q*THETA)**2
        WPI=DSIN(THETA)
        WR=1._q
        WI=0._q
        DO 13 M=1,MMAX,2
          DO 12 I=M,N,ISTEP
            J=I+MMAX
            TEMPR=SNGL(WR)*DATA(J)-SNGL(WI)*DATA(J+1)
            TEMPI=SNGL(WR)*DATA(J+1)+SNGL(WI)*DATA(J)
            DATA(J)=DATA(I)-TEMPR
            DATA(J+1)=DATA(I+1)-TEMPI
            DATA(I)=DATA(I)+TEMPR
            DATA(I+1)=DATA(I+1)+TEMPI
12        CONTINUE
          WTEMP=WR
          WR=WR*WPR-WI*WPI+WR
          WI=WI*WPR+WTEMP*WPI+WI
13      CONTINUE
        MMAX=ISTEP
      GO TO 2
      ENDIF
      RETURN
  END SUBROUTINE
     
  FUNCTION S_CUT(val)
    REAL(q) :: val, s_cut
    !cut off function for the kernels, 
    !personally this looks quite harsh
    if (val .le. 0._q) then
      s_cut=1._q
    else if (val .ge. 1) then
      s_cut=0._q
    else
      s_cut=(1._q-val**8)**4
    endif 

  END FUNCTION


  SUBROUTINE SPLINE(dx,y,n,yp1,ypn,y2)
      !from Num recipes
      INTEGER n
      REAL(q) :: yp1,ypn,x(n),y(n),y2(n),dx
      INTEGER i,k
      REAL(q) :: p,qm,sig,un,u(n)

      do i=1,n
       x(i)=dx*i
      enddo
      if (yp1.gt..99e30_q) then
        y2(1)=0._q
        u(1)=0._q
      else
        y2(1)=-0.5_q
        u(1)=(3._q/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do  i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2._q
        y2(i)=(sig-1._q)/p
        u(i)=(6._q*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      enddo
      if (ypn.gt..99e30_q) then
        qm=0._q
        un=0._q
      else
        qm=0.5_q
        un=(3._q/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qm*u(n-1))/(qm*y2(n-1)+1._q)
      do k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
      enddo
      return
  END  SUBROUTINE 


  SUBROUTINE MIN_K(GRIDC,LATT_CUR)
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    INTEGER :: I, N1, N2, N3, NC, ix, iy, iz
    REAL(q) :: cell(3,3)
    REAL(q) :: k_x,k_y,k_z,kte
    !the latt_cur%B has rec vector x stored as 11 21 31
    !and similarly for the rest
    !to get total coordinate x in k-space sum is done over
    !11 12 13  
    N1=GRIDC%NGX
    N2=GRIDC%NGY
    N3=GRIDC%NGZ
    kcut = 10.e10_q
    cell=latt_cur%B*AUTOA
    !now we want to know the smallest boundary k-vector
    !IF (IU0>=0) WRITE(IU0,*) 'reciprocal cell'
    !IF (IU0>=0) WRITE(IU0,*) cell
    !IF (IU0>=0) WRITE(IU0,*) N1*cell(1,1), N2*cell(1,2), N3*cell(1,3)
    !IF (IU0>=0) WRITE(IU0,*) N1*cell(2,1),N2*cell(2,2), N3*cell(2,3)
    !IF (IU0>=0) WRITE(IU0,*) N1*cell(3,1), N2*cell(3,2), N3*cell(3,3)

    do ix = -1,1
      do iy = -1,1
        do iz = -1,1
          if ((abs(ix)+abs(iy)+abs(iz)).ne.0 ) then
            k_x=N1*cell(1,1)*ix+N2*cell(1,2)*iy+N3*cell(1,3)*iz
            k_y=N1*cell(2,1)*ix+N2*cell(2,2)*iy+N3*cell(2,3)*iz
            k_z=N1*cell(3,1)*ix+N2*cell(3,2)*iy+N3*cell(3,3)*iz
            kte=2._q*pi*sqrt(k_x*k_x+k_y*k_y+k_z*k_z)*0.5_q
            kcut=min(kcut, kte)   
          endif
        enddo
      enddo
    enddo 

  END SUBROUTINE


  SUBROUTINE GET_K_IND(k_k,GRIDC,LATT_CUR)
    !get the k-space interpolation indices 
    !can be done once
    TYPE (latt) LATT_CUR
    TYPE (grid_3d) GRIDC
    REAL(q) k_k(GRIDC%MPLWV)
    INTEGER :: I, N1, N2, N3, NC 
    REAL(q) :: cell(3,3)  
    REAL(q) :: k_x, k_y, k_z
  
    cell=latt_cur%B*AUTOA
    do i=1,gridc%rc%np
      n1=Mod((i-1),gridc%rc%nrow)+1
      nc=(i-1)/gridc%rc%nrow+1
      n2=gridc%rc%i2(nc)
      n3=gridc%rc%i3(nc)
      k_x=gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3)  
      k_y=gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3)  
      k_z=gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3)  
      k_k(i)=2._q*pi*sqrt(k_x*k_x+k_y*k_y+k_z*k_z)
    enddo

  END SUBROUTINE


END MODULE



SUBROUTINE VDW_CALC_RVV10(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,DINP,spl,phik,d2phidk2,qs_vdw,k_k,stress,ISIF)
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE setexm
  USE VDW_rVV10, ONLY : Nalp, Nr, q0cut, rcut

  IMPLICIT NONE
  INTEGER NP
  REAL(q) RINPL, NELECT
!I guess LATT_CURR is the one that we need not OMEGA only
  TYPE (latt) LATT_CUR
  TYPE (grid_3d) GRIDC
  REAL(q) OMEGA
  REAL(q) EXC
  REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
  REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
  REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
  REAL(q) stress(3,3)
  INTEGER :: I, IU, alf, bet, ik_k, iofq, ii, jj, ISIF
  INTEGER :: NODE_ME, IONODE, IDUMP
  COMPLEX(q) :: CINP(GRIDC%MPLWV)  
  COMPLEX(q), allocatable::   THETA(:,:),f_k(:)
  
  RGRID :: DINP(DIMREAL(GRIDC%MPLWV))
  REAL(q),parameter :: dk_k=pi/rcut
  REAL(q) :: nrs, rnrs
  REAL(q) :: diff, coef(4), E_corr
  REAL(q) :: spl(Nalp,Nalp,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) , allocatable :: q0(:),dq0(:),pa(:),f_g(:,:)
  REAL(q) :: qs_vdw(Nalp)
  REAL(q) :: k_k(GRIDC%MPLWV)
  REAL(q) :: energ
  REAL(q) :: a_k, b_k, a2_k, b2_k, a3_k, b3_k, pref
  REAL(q) :: ro,gr,dq0dn, dq0dgr, dpadq0, dtheadn, dtheadgr
  REAL(q) :: dhdx, pa_tm, phi, dphidk
  REAL(q) :: k_x(3), k_y, k_z, k_a, cell(3,3)
  REAL(q) :: max_n, max_gr, a_iofq, b_iofq
  RGRID   DWORK(GRIDC%RL%NP) 
  INTEGER,allocatable :: alp(:)
  INTEGER :: n1,n2,n3,nc
  INTEGER ::  i_max_n, i_max_gr
  
  REAL(q) :: omega_g2, omega_p2, kval, omega_0
  REAL(q) :: betaparam
  
  NP=GRIDC%RL%NP 
  !  IF (IU0>=0) WRITE(IU0,*)  "MPLWV: ", GRIDC%MPLWV, DIMREAL(GRIDC%MPLWV), GRIDC%RC%NP
  ALLOCATE(q0(NP),dq0(NP),alp(NP),pa(NP),f_g(NP,Nalp))
  allocate(THETA(GRIDC%MPLWV,Nalp),f_k(GRIDC%MPLWV))
  cell=latt_cur%B*AUTOA  
  stress=0._q
  RINPL = 1._q/(real(GRIDC%NGX,kind=q)*GRIDC%NGY*GRIDC%NGZ)

  energ=0.0_q

  betaparam=0.03125_q * (3.0_q / (bparam**2.0_q) )**(0.75_q) * 2._q*RYTOEV 

#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debugNODE_ME
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      NODE_ME=0
      IONODE =0
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
  !calculate q0 in each point and it's saturated version (Soler's paper)
  DO I=1,NP
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF

    omega_p2 = 4.0_q*pi*ro
    omega_g2 = Cparam*((gr/ro)**4)
    kval=1.5_q*pi*bparam*((ro/(9.0_q*pi))**(1.0_q/6.0_q))
    omega_0 = sqrt(omega_g2 + omega_p2/3.0_q)
    q0(i) = omega_0 / kval
  
    !now saturated
    nrs=0._q
    DO iu=1,12
      nrs=nrs+(q0(i)/q0cut)**iu/real(iu,kind=q)
    ENDDO
    q0(i)=q0cut*(1._q-exp(-nrs))
  ENDDO

  !calculate alp and dq0 in each point
  !that is which interpolating point we will be using and how far from it are we
  a_iofq=log((qs_vdw(Nalp)-qs_vdw(Nalp-1))/(qs_vdw(2)-qs_vdw(1)))/(Nalp-2.0_q)
  a_iofq=max(a_iofq,1e-12_q)
  b_iofq=(qs_vdw(2)-qs_vdw(1))/(exp(a_iofq)-1.0_q)
  DO I=1,NP
    iofq=1+log(1+(q0(i)-qs_vdw(1))/b_iofq)/a_iofq
    iofq=max(1,iofq)
    iofq=min(Nalp,iofq)
    IF(q0(i).GT.(q0cut-0.00001_q)) iofq=Nalp
    alp(i)=iofq
  ENDDO
  !this is the difference from interpolating point
  DO I=1,NP
    dq0(i)=q0(i)-qs_vdw(alp(i))
  ENDDO

  !for each interpolating point create p_\alpha (Soler's paper)
  !and Fourier transform it, this is stored in \Theta array
  DO alf=1,Nalp
    DO I=1,NP
      coef(:)=spl(alp(i),alf,:)
      diff=dq0(i)
      pa(i)=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
    ENDDO
    DO I=1,NP
      IF (DCHARG(i).ge.0._q) THEN
        ro=DCHARG(i)*AUTOA3

        kval=1.5_q*pi*bparam*((ro/(9.0_q*pi))**(1.0_q/6.0_q))
        dinp(i)=pa(i)*ro/(kval**1.5_q)
      ELSE
        dinp(i)=0._q
      ENDIF
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,-1)
    DO I=1,GRIDC%RC%NP
      THETA(i,alf)=CINP(i)
    ENDDO
  ENDDO

  !now we are going to calculate the energy
  !this requires two nested loops over \alpha and \beta
  !first we do the sum over \Theta_\beta * kernel
  DO alf=1,Nalp
    f_k=0._q
    DO bet=1,Nalp
      DO i=1,GRIDC%RC%NP
        ik_k=k_k(i)/dk_k
        IF (ik_k.GE.Nr) THEN
          !I really don't like the if here
          phi=0._q
          dphidk=0._q
        ELSE
          a_k=((ik_k+1._q)*dk_k-k_k(i))/dk_k
          b_k=1._q-a_k
          a2_k=(3._q*a_k**2.-1._q)*dk_k/6._q
          b2_k=(3._q*b_k**2.-1._q)*dk_k/6._q
          a3_k=(a_k**3.-a_k)*dk_k**2._q/6._q
          b3_k=(b_k**3.-b_k)*dk_k**2._q/6._q
          phi=a_k*phik(ik_k, bet, alf)+b_k*phik(ik_k+1,bet,alf) &
           +a3_k*d2phidk2(ik_k,bet,alf)+b3_k*d2phidk2(ik_k+1,bet,alf)
          dphidk=(-phik(ik_k, bet, alf)+phik(ik_k+1,bet,alf))/dk_k &
                 -a2_k*d2phidk2(ik_k,bet,alf)+b2_k*d2phidk2(ik_k+1,bet,alf)

        ENDIF
        f_k(i)=f_k(i)+theta(i,bet)*phi
        IF (ISIF .ge. 1) THEN
#ifdef NGZhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        if (k_a .gt. 1.0e-13_q) then
          if ((n3.eq.1).or.(n3.eq.(GRIDC%NGZ/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj) 
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj) 
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                  (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
#ifdef NGXhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        if (k_a .gt. 1.0e-13_q) then
          if ((n1.eq.1).or.(n1.eq.(GRIDC%NGX/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)                                  
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)  
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) * &
                            (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif
        ENDIF !only calculate stress when required

      ENDDO !q points
    ENDDO  !beta


#ifdef NGZhalf
!for MPI version reduced in Z direction
    !and then we sum it with \Theta_\alpha
    DO i=1,GRIDC%RC%NP
      nc=(i-1)/gridc%rc%nrow+1
      n3=gridc%rc%i3(nc)
      IF (n3.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n3.EQ.(GRIDC%NGZ/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q) 
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
#ifdef NGXhalf
!for serial version reduced in X direction
    DO i=1,GRIDC%RC%NP
      n1=Mod((i-1),gridc%rc%nrow)+1
      IF (n1.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n1.EQ.(GRIDC%NGX/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif

    !now we fourier transform the \Theta * kernel to real space to reuse it for potential
    DO i=1,GRIDC%RC%NP
      CINP(I)=f_k(i)
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,1)
    DO i=1,GRIDC%RL%NP
      f_g(i,alf)=dinp(i)*RINPL
    ENDDO
  ENDDO

  !add the energy to the xc energy, include volume factor
  !  number of grid points since we've done FFT, and factors to get eV
  E_corr=0.5_q*energ*LATT_CUR%OMEGA*RINPL/AUTOA3*2._q*RYTOEV + betaparam * SUM(DCHARG)*LATT_CUR%OMEGA
  EXC=EXC + E_corr
  !compared to vander (my local version (jik)), 
  !we divide by (M)PLWV only once since it's done in xcgrad for the second time
  !so we can say that we get rid of the FFT 1/N factor and keep the \Omega/N for xcgrad
  E_corr=E_corr*RINPL

  max_n=0._q
  max_gr=0._q

  DO i=1,gridc%RL%NP
    !calc the potential 
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE 
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF
    
    dq0dn=(-39._q*Cparam*(gr**4)+8._q*pi*(ro**5)) / (9._q*(3._q**(1._q/6._q))*bparam*(pi**(5._q/6._q))*(ro**(19._q/6._q))*sqrt(3._q*Cparam*(gr**4)+4._q*pi*(ro**5)))
    dq0dgr=4._q*Cparam*(gr**3)/((3._q**(1._q/6._q))*bparam*(pi**(5._q/6._q))*(ro**(13._q/6._q))*sqrt(3._q*Cparam*(gr**4)+4._q*pi*(ro**5)))

    nrs=0._q
    rnrs=0._q
    DO iu=1,12
      rnrs=rnrs+(q0(i)/q0cut)**iu/real(iu,kind=q)
    ENDDO
    DO iu=0,11
      nrs=nrs+(q0(i)/q0cut)**iu
    ENDDO
    dhdx=nrs*exp(-rnrs)
    dtheadn=0._q
    dtheadgr=0._q
    !one loop is needed to calculate the potential
    DO alf=1,Nalp
      coef=spl(alp(i),alf,:)
      diff=dq0(i)
      pa_tm=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
      dpadq0=coef(2)+diff*(2._q*coef(3)+3._q*diff*coef(4))
      dtheadn=dtheadn+(pa_tm/(4._q*(bparam**1.5_q)*(pi**1.25_q)*(ro**0.25_q))+(ro**0.75_q)*dpadq0*dq0dn*dhdx/(3._q*(pi**1.25_q)*(bparam**1.5_q))) *f_g(i,alf)*2._q**1.5_q
      dtheadgr=dtheadgr+ 2._q**1.5_q*f_g(i,alf) * (ro**0.75_q)*dpadq0*dhdx*dq0dgr/(3._q*(pi**1.25_q)*(bparam**1.5_q))
    ENDDO

    !add the potentials to the GGA ones
    !this is d/dn
    DWORKG(i)=DWORKG(i)+dtheadn*2._q*RYTOEV  + betaparam
    !this is d/ddn
    DWORK(i)=DWORK(i)+dtheadgr*2._q*RYTOEV/ MAX(DWORK4(I),1.E-10_q)*AUTOA    
  ENDDO !over points on real space grid

  DEALLOCATE(q0,dq0,alp,pa)
  DEALLOCATE(theta,f_k,f_g)
  stress=0.5_q*stress*LATT_CUR%OMEGA*RINPL**2._q *2._q*RYTOEV/AUTOA3


NELECT = SUM(DCHARG)*LATT_CUR%OMEGA*RINPL
#ifdef MPI
 CALLMPI( M_sum_d(GRIDC%COMM,E_corr,1))
 CALLMPI( M_sum_d(GRIDC%COMM,NELECT,1))
#endif

  IF (NODE_ME == IONODE) THEN
    WRITE(*,'(A,2F14.7)') 'Total rVV10 correction in eV: ', E_corr, NELECT
  ENDIF

END SUBROUTINE

SUBROUTINE VDW_CALC_RVV10_spin(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,CINP,DINP,spl,phik,d2phidk2,qs_vdw,k_k,stress,ISIF)
  USE prec
  USE lattice
  USE mpimy
  USE mgrid
  USE constant
  USE setexm
  USE VDW_rVV10, ONLY : Nalp, Nr, q0cut, rcut

  IMPLICIT NONE
  INTEGER NP
  REAL(q) RINPL, NELECT
  TYPE (latt) LATT_CUR
  TYPE (grid_3d) GRIDC
  REAL(q) OMEGA
  REAL(q) EXC
  REAL(q) DCHARG(GRIDC%RL%NP) !charge density on input and output
  REAL(q) DWORKG(GRIDC%RL%NP) !df/drho input, add the same non-local here
                    !don't touch this one yet
  REAL(q) DWORK4(GRIDC%RL%NP) !gradient on input
                    !df/d(drho)/(drho)potential on output
  REAL(q) stress(3,3)
  INTEGER :: I, IU, alf, bet, ik_k, iofq, ii, jj, ISIF
  INTEGER :: NODE_ME, IONODE, IDUMP
  COMPLEX(q) :: CINP(GRIDC%MPLWV)  
  COMPLEX(q), allocatable::   THETA(:,:),f_k(:)

  RGRID :: DINP(DIMREAL(GRIDC%MPLWV))
  REAL(q),parameter :: dk_k=pi/rcut
  REAL(q) :: nrs, rnrs
  REAL(q) :: diff, coef(4), E_corr
  REAL(q) :: spl(Nalp,Nalp,4)
!  REAL(q) :: phi_ab(Nalp,Nalp,0:Nr/2-1,4)
  REAL(q) :: phik(0:Nr,Nalp,Nalp), d2phidk2(0:Nr,Nalp,Nalp)
  REAL(q) , allocatable :: q0(:),dq0(:),pa(:),f_g(:,:)
  REAL(q) :: qs_vdw(Nalp)
  REAL(q) :: k_k(GRIDC%MPLWV)
  REAL(q) :: energ
  REAL(q) :: a_k, b_k, a2_k, b2_k, a3_k, b3_k, pref
  REAL(q) :: ro,gr,dq0dn, dq0dgr, dpadq0, dtheadn, dtheadgr
  REAL(q) :: dhdx, pa_tm, phi, dphidk
  REAL(q) :: k_x(3), k_y, k_z, k_a, cell(3,3)
  REAL(q) :: max_n, max_gr, a_iofq, b_iofq
  RGRID   DWORK(GRIDC%RL%NP)
  INTEGER,allocatable :: alp(:)
  INTEGER :: n1,n2,n3,nc
  INTEGER ::  i_max_n, i_max_gr
  
  REAL(q) :: omega_g2, omega_p2, kval, omega_0
  REAL(q) :: betaparam
  
  NP=GRIDC%RL%NP
  RINPL = 1._q/(real(GRIDC%NGX,kind=q)*GRIDC%NGY*GRIDC%NGZ)
  ALLOCATE(q0(NP),dq0(NP),alp(NP),pa(NP),f_g(NP,Nalp))
  ALLOCATE(THETA(GRIDC%MPLWV,Nalp),f_k(GRIDC%MPLWV))
  stress=0._q
  cell=latt_cur%B*AUTOA

  energ=0.0_q

  betaparam=0.03125_q * (3.0_q / (bparam**2.0_q) )**(0.75_q) * 2._q*RYTOEV 

#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debugNODE_ME
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      NODE_ME=0
      IONODE =0
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
  DO I=1,NP
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF

    omega_p2 = 4.0_q*pi*ro
    omega_g2 = Cparam*((gr/ro)**4)
    kval=1.5_q*pi*bparam*((ro/(9.0_q*pi))**(1.0_q/6.0_q))
    omega_0 = sqrt(omega_g2 + omega_p2/3.0_q)
    q0(i) = omega_0 / kval
    
    !now saturated
    nrs=0._q
    DO iu=1,12
      nrs=nrs+(q0(i)/q0cut)**iu/real(iu,q)
    ENDDO
    q0(i)=q0cut*(1._q-exp(-nrs))
  ENDDO

  !calculate alp and dq0 in each point
  a_iofq=log((qs_vdw(Nalp)-qs_vdw(Nalp-1))/(qs_vdw(2)-qs_vdw(1)))/(Nalp-2.0_q)
  a_iofq=max(a_iofq,1e-12_q)
  b_iofq=(qs_vdw(2)-qs_vdw(1))/(exp(a_iofq)-1.0_q)
  DO I=1,NP
    iofq=1+log(1+(q0(i)-qs_vdw(1))/b_iofq)/a_iofq
    iofq=max(1,iofq)
    iofq=min(Nalp,iofq)
    !!!
    IF(q0(i).GT.(q0cut-0.00001_q)) iofq=Nalp
    alp(i)=iofq
  ENDDO
  DO I=1,NP
    dq0(i)=q0(i)-qs_vdw(alp(i))
  ENDDO


  DO alf=1,Nalp
    DO I=1,NP
      coef(:)=spl(alp(i),alf,:)
      diff=dq0(i)
      pa(i)=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
    ENDDO
    DO I=1,NP
      IF (DCHARG(i).ge.0._q) THEN
        ro=DCHARG(i)*AUTOA3

        kval=1.5_q*pi*bparam*((ro/(9.0_q*pi))**(1.0_q/6.0_q))
        dinp(i)=pa(i)*ro/(kval**1.5_q)
      ELSE
        dinp(i)=0._q
      ENDIF
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,-1)
    DO I=1,GRIDC%RC%NP
      THETA(i,alf)=CINP(i)
    ENDDO
  ENDDO

  DO alf=1,Nalp
    f_k=0._q

    DO bet=1,Nalp
      DO i=1,GRIDC%RC%NP
        ik_k=k_k(i)/dk_k
        IF (ik_k.GE.Nr) THEN
          !I really don't like the if here
          phi=0._q
          dphidk=0._q
        ELSE
          a_k=((ik_k+1._q)*dk_k-k_k(i))/dk_k
          b_k=1._q-a_k
          a2_k=(3._q*a_k**2.-1._q)*dk_k/6._q
          b2_k=(3._q*b_k**2.-1._q)*dk_k/6._q
          a3_k=(a_k**3.-a_k)*dk_k**2._q/6._q
          b3_k=(b_k**3.-b_k)*dk_k**2._q/6._q
          phi=a_k*phik(ik_k, bet, alf)+b_k*phik(ik_k+1,bet,alf)&
              +a3_k*d2phidk2(ik_k,bet,alf)+b3_k*d2phidk2(ik_k+1,bet,alf)
          dphidk=(-phik(ik_k, bet, alf)+phik(ik_k+1,bet,alf))/dk_k &
                 -a2_k*d2phidk2(ik_k,bet,alf)+b2_k*d2phidk2(ik_k+1,bet,alf)

        ENDIF
        f_k(i)=f_k(i)+theta(i,bet)*phi
        IF (ISIF .ge. 1) THEN
#ifdef NGZhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        pref=dphidk/k_a
        if (k_a .gt. 1.0e-13_q) then
          if ((n3.eq.1).or.(n3.eq.(GRIDC%NGZ/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)                                  
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)  
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                 (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
               
              enddo
            enddo
          endif
        endif
#else
#ifdef NGXhalf
        n1=Mod((i-1),gridc%rc%nrow)+1
        nc=(i-1)/gridc%rc%nrow+1
        n2=gridc%rc%i2(nc)
        n3=gridc%rc%i3(nc)
        k_x(1)=tpi*(gridc%lpctx(n1)*cell(1,1)+gridc%lpcty(n2)*cell(1,2)+gridc%lpctz(n3)*cell(1,3))
        k_x(2)=tpi*(gridc%lpctx(n1)*cell(2,1)+gridc%lpcty(n2)*cell(2,2)+gridc%lpctz(n3)*cell(2,3))
        k_x(3)=tpi*(gridc%lpctx(n1)*cell(3,1)+gridc%lpcty(n2)*cell(3,2)+gridc%lpctz(n3)*cell(3,3))
        k_a=sqrt(k_x(1)*k_x(1)+k_x(2)*k_x(2)+k_x(3)*k_x(3))  !*2._q*pi factor is here
        pref=dphidk/k_a
        if (k_a .gt. 1.0e-13_q) then
          if ((n1.eq.1).or.(n1.eq.(GRIDC%NGX/2+1))) then
            pref=dphidk/k_a*theta(i,bet)*conjg(theta(i,alf))
            do ii=1,3
              do jj=1,3
                stress(ii,jj)=stress(ii,jj)-pref*k_x(ii) * k_x(jj)
              enddo
            enddo
          else
            pref=dphidk/k_a
            do ii=1,3
              do jj=1,3
!                stress(ii,jj)=stress(ii,jj)- theta(i,bet)*pref*conjg(theta(i,alf))* k_x(ii) * k_x(jj)  &
!                                         - conjg(theta(i,bet))*pref*theta(i,alf)* k_x(ii) * k_x(jj)
                 stress(ii,jj)=stress(ii,jj)- pref*k_x(ii) * k_x(jj) *  &
                                 (theta(i,bet)*conjg(theta(i,alf))+ conjg(theta(i,bet))*theta(i,alf))
              enddo
            enddo
          endif
        endif
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif
        ENDIF !only calculate stress when required

      ENDDO
    ENDDO !beta

#ifdef NGZhalf
!for MPI version reduced in Z direction
    !and then we sum it with \Theta_\alpha
    DO i=1,GRIDC%RC%NP
      nc=(i-1)/gridc%rc%nrow+1
      n3=gridc%rc%i3(nc)
      IF (n3.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n3.EQ.(GRIDC%NGZ/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
#ifdef NGXhalf
!for serial version reduced in X direction
    DO i=1,GRIDC%RC%NP
      n1=Mod((i-1),gridc%rc%nrow)+1
      IF (n1.EQ.1) THEN
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)
      ELSEIF (n1.EQ.(GRIDC%NGX/2+1)) THEN
        energ=energ+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ELSE
        energ=energ+real(f_k(i)*conjg(theta(i,alf)),kind=q)+real(conjg(f_k(i))*(theta(i,alf)),kind=q)
      ENDIF
    ENDDO
#else
  WRITE(0,*) 'vdW-DF calculation: either NGZhalf or NGXhalf needs to be used for compilation  '
  STOP
#endif
#endif


    DO i=1,GRIDC%RC%NP
      CINP(I)=f_k(i)
    ENDDO
    CALL OPSYNC(DINP,CINP,GRIDC%NPLWV)
    CALL FFT3D(CINP,GRIDC,1)
    DO i=1,GRIDC%RL%NP
      f_g(i,alf)=dinp(i)*RINPL
    ENDDO
  ENDDO !alf
  

! EXC=EXC + 0.5_q*energ*LATT_CUR%OMEGA*RINPL/AUTOA3*2._q*RYTOEV
  E_corr=0.5_q*energ*LATT_CUR%OMEGA*RINPL/AUTOA3*2._q*RYTOEV + betaparam * SUM(DCHARG)*LATT_CUR%OMEGA
  EXC=EXC + E_corr
  !compared to vander, we divide by (M)PLWV only once since it's done in xcgrad for the second time
  !so we can say that we get rid of the FFT 1/N factor and keep the \Omega/N for xcgrad
  E_corr=E_corr*RINPL

  max_n=0.d0
  max_gr=0.d0

  DO i=1,gridc%RL%NP
  !calc the potential
    IF (DCHARG(i).LT. 1e-11_q) THEN
      ro=1e-15_q
      gr=0._q
    ELSE
      ro=DCHARG(i)*AUTOA3
      gr=DWORK4(i)*AUTOA4
    ENDIF
    
    dq0dn=(-39._q*Cparam*(gr**4)+8._q*pi*(ro**5)) / (9._q*(3._q**(1._q/6._q))*bparam*(pi**(5._q/6._q))*(ro**(19._q/6._q))*sqrt(3._q*Cparam*(gr**4)+4._q*pi*(ro**5)))
    dq0dgr=4._q*Cparam*(gr**3)/((3._q**(1._q/6._q))*bparam*(pi**(5._q/6._q))*(ro**(13._q/6._q))*sqrt(3._q*Cparam*(gr**4)+4._q*pi*(ro**5)))
    
    nrs=0._q
    rnrs=0._q
    DO iu=1,12
      rnrs=rnrs+(q0(i)/q0cut)**iu/real(iu,q)
    ENDDO
    DO iu=0,11
      nrs=nrs+(q0(i)/q0cut)**iu
    ENDDO
    dhdx=nrs*exp(-rnrs)
    dtheadn=0._q
    dtheadgr=0._q
    DO alf=1,Nalp
      coef=spl(alp(i),alf,:)
      diff=dq0(i)
      pa_tm=coef(1)+diff*(coef(2)+diff*(coef(3)+diff*coef(4)))
      dpadq0=coef(2)+diff*(2._q*coef(3)+3._q*diff*coef(4))
      dtheadn=dtheadn+(pa_tm/(4._q*(bparam**1.5_q)*(pi**1.25_q)*(ro**0.25_q))+(ro**0.75_q)*dpadq0*dq0dn*dhdx/(3._q*(pi**1.25_q)*(bparam**1.5_q)))*f_g(i,alf) *2._q**1.5_q
      dtheadgr=dtheadgr+(ro**0.75_q)*dpadq0*dhdx*dq0dgr*f_g(i,alf)/(3._q*(pi**1.25_q)*(bparam**1.5_q)) *2._q**1.5_q
    ENDDO

    !this is d/dn
    DWORKG(i)=DWORKG(i)+dtheadn*2._q*RYTOEV + betaparam
    !this is d/ddn
    DWORK(i)=DWORK(i)+dtheadgr*2._q*RYTOEV* AUTOA
    ! in the spin polarized case the gradient is used later MAX(DWORK4(I),1.E-10_q)

  ENDDO

  DEALLOCATE(q0,dq0,alp,pa)
  DEALLOCATE(theta,f_k,f_g)
  stress=0.5_q*stress*LATT_CUR%OMEGA*RINPL**2._q *2._q*RYTOEV/AUTOA3  

NELECT = SUM(DCHARG)*LATT_CUR%OMEGA*RINPL
#ifdef MPI
 CALLMPI( M_sum_d(GRIDC%COMM,E_corr,1))
 CALLMPI( M_sum_d(GRIDC%COMM,NELECT,1))
#endif

  IF (NODE_ME == IONODE) THEN
    WRITE(*,'(A,2F14.7)') 'Total rVV10 correction in eV: ', E_corr, NELECT
  ENDIF

END SUBROUTINE

