#include "symbol.inc"
#ifdef oldsym
!--------------------------------------------------------------------
! Dummy module
!--------------------------------------------------------------------
module spinsym
  contains
  subroutine spinsym_dummy
  write(*,*)'Im a DEC compiler so I need this line'
  end subroutine 
end module
#else
!--------------------------------------------------------------------
! Module to include spinor rotations
! Daniel Aberg, Sun Apr 17 13:28:23 PDT 2011
!--------------------------------------------------------------------
module spinsym
  
  use prec      ! single vs double precision
   
  implicit none
  
  public set_spinrot, rotate_wave_spin, rotate_wave_spin_recip
  complex(q),save :: rssymop(2,2,48)
  real(q),save    :: spinaxis(3,48), spindet(48), spinangle(48)

!  private
  
contains
  subroutine set_spinrot(a, b, isymop, nrotk, gtrans, iu6)
    !--------------------------------------------------
    ! This subroutine transforms the 3x3 rotation
    ! matrices in real (direct) space to 2x2 rotation matrices
    ! in spinor (cartesian) space. 
    use lattice
    use constant
    implicit none
    ! Arguments
    real(q), intent(in) :: a(3,3), b(3,3)
    integer, intent(in) :: nrotk, iu6
    integer, intent(in) :: isymop(3,3,48)
    real(q), intent(in) :: gtrans(3,48)

    ! Local variables
    integer    :: nkpt, ikpt, i, j, itmpmat(3,3), invflag, irotk
    real(q)    :: rtmpmat(3,3), rinvers(3,3), rdet, angle, axis(3)
    complex(q) :: cdet, spinmat(2,2)
    logical    :: lwrt=.false.,lwrt2=.true.


#define do_dbg    IF (lwrt)

    !-----
    ! set the configuration-space inverse
    rinvers=zero
    do i=1,3 
       rinvers(i,i)=-1.0_q
    end do
    !-----
    do_dbg write(*,*) 'nrotk=',nrotk

    do irotk=1,nrotk
       do_dbg write(*,'(a,1x,I5)') 'irotk=', irotk
       do i=1,3
          do_dbg write(*,'(3(1x,3I4))') &
               (isymop(i,j,irotk),j=1,3)
       end do
       do_dbg write(*,'(a,3(1x,F12.6))') 'trans=',gtrans(:,irotk)

       ! Transform rotation matrix to real-space...
       rtmpmat=matmul(b, matmul(real(isymop(:,:,irotk)),transpose(a)))
       do_dbg write(*,*) 'transformed matrix'
       do i=1,3
          do_dbg write(*,'(3(1x,3F12.6))') &
               (rtmpmat(i,j),j=1,3)
       end do
       ! Check if inversion is included in the operator
       invflag=1
       call calc_rdet(rtmpmat, rdet)
       do_dbg write(*,'(a,1x,F6.3)') 'det=', rdet
       ! If so, take it out, before getting the rotation
       ! axis and angle
       if(abs(rdet + 1.0_q) .lt. 1e-5 ) then
          do_dbg write(*,*) 'FOUND INVERSION'
          rtmpmat=matmul(rtmpmat,rinvers)
          rdet=rdet*(-1.0_q)
          invflag=-1
       end if
       ! Check if matrix is unitary within some limits
       if(abs(abs(rdet)-1.0_q) .gt. 1e-5) then
          write(*,*) 'Stop in set_spinrot'
          write(*,*) 'Rotation matrix not unitary.'
          write(*,*) 'rdet',rdet
          write(*,*) rtmpmat
          STOP
       end if
       ! get rotation axis and angle
       call get_rotation(irotk, rtmpmat, angle, axis)
       do_dbg write(*,'(a,1x,F12.6)') 'Rotation angle=',angle*180.0_q/pi
       do_dbg write(*,'(a,1x,3(1x,F12.6))') 'rotation axis',axis
       ! Now, get the spinor-rotation matrix
       call get_spinorrot(angle, axis, invflag, spinmat)
       cdet=spinmat(1,1)*spinmat(2,2)-spinmat(1,2)*spinmat(2,1)
       do_dbg write(iu6,*) 'spinmat'
       do_dbg write(iu6,'(2(2(1x,F12.6)))') spinmat(1,:)
       do_dbg write(iu6,'(2(2(1x,F12.6)))') spinmat(2,:)
       do_dbg write(iu6,'(a,1x,2(F12.6))') 'spinor det',cdet
       rssymop(:,:,irotk) = spinmat
       spindet(irotk)     = real(cdet)
       spinangle(irotk)   = angle*180.0_q/pi
       spinaxis(:,irotk)  = axis
       if(iu6>0) then
          if(irotk .eq. 1) then
             write(iu6,'(a)') 'Space group operators:'
             write(iu6,'(a)') ' irot       det(A)        alpha          n_x          n_y          n_z        tau_x        tau_y        tau_z'
          end if
          write(iu6,'(i5)',advance="no") irotk
          write(iu6,'(1x,F12.6)',advance="no") spindet(irotk)
          write(iu6,'(1x,F12.6)',advance="no") spinangle(irotk)
          write(iu6,'(3(1x,F12.6))',advance="no") spinaxis(:,irotk)
          write(iu6,'(3(1x,F12.6))') gtrans(:,irotk)
       end if
       do_dbg read(*,*)
    end do
  end subroutine set_spinrot


  subroutine calc_rdet(rtmpmat, rdet)
    implicit none
    real(q),intent(in)  :: rtmpmat(3,3)
    real(q),intent(out) :: rdet

    rdet = & 
         rtmpmat(1,1)*rtmpmat(2,2)*rtmpmat(3,3) - &
         rtmpmat(1,1)*rtmpmat(2,3)*rtmpmat(3,2) + &
         rtmpmat(1,2)*rtmpmat(2,3)*rtmpmat(3,1) - &
         rtmpmat(1,2)*rtmpmat(2,1)*rtmpmat(3,3) + &
         rtmpmat(1,3)*rtmpmat(2,1)*rtmpmat(3,2) - &
         rtmpmat(1,3)*rtmpmat(2,2)*rtmpmat(3,1) 

  end subroutine calc_rdet


  subroutine get_rotation(irotk, mat, angle, axis)
    use constant
    implicit none
    integer,intent(in)    :: irotk
    real(q),intent(inout) :: mat(3,3)
    real(q),intent(out)   :: angle, axis(3)
    ! local variables
    integer    :: i, j
    ! dgeev variables
    integer    :: info, iaxis
    real(q)    :: dvl(3,3), dvr(3,3), wi(3), dwork(12), wr(3), arg
    !----
    arg=((mat(1,1)+mat(2,2)+mat(3,3)-1.0_q)*0.5_q)
    if(arg>1.0_q)  arg=1.0_q
    if(arg<-1.0_q) arg=-1.0_q
    angle=acos(arg)
       
    if(abs(abs(angle) - pi) .lt. 1e-4) then 
       ! angle is 180 deg => can't find the axis the
       ! easy way. Diagonalize rotation matrix and 
       ! pick the eigenvector corresponding to 
       ! unity eigenvalue.
       call DGEEV( 'N', 'V', 3, mat, 3, wr, wi, dvl, 1, &
            dvr,  3,  dwork, 12,info)
       if(info .ne. 0) then
          write(*,*) 'error in dgeev. info=',info
          STOP
       end if
       ! find the axis...just pick the first one with e=1
       iaxis=0
       do i=1,3
          if(abs(wr(i)-1.0_q) .lt. 1e-9 .and. abs(wi(i)) .lt. 1e-9) then
             iaxis=i
             exit
          end if
       end do
       if(iaxis .lt. 1) then
          write(*,*) 'could not find rotation axis for irotk=',irotk
          STOP
       end if
    else if(abs(angle) .gt. 1e-3) then
       ! standard case. See Altmann's book
       dvr(1,1)=mat(3,2)-mat(2,3)
       dvr(2,1)=mat(1,3)-mat(3,1)
       dvr(3,1)=mat(2,1)-mat(1,2)
       dvr=dvr/sin(angle)/2.0_q
       iaxis=1
    else if(abs(angle) .lt. 1e-4) then
       dvr(1,1)=1.0_q
       dvr(2,1)=0.0_q
       dvr(3,1)=0.0_q       
       iaxis=1
    end if
    axis=dvr(:,iaxis)
  end subroutine get_rotation
 

  subroutine get_spinorrot(angle, axis, invflag, spinmat)
    !----------------------------------------------------------------
    ! This routine calculates the spinor rotation matrix according to
    ! R = exp(-i/2 th n\cdot \sigma)=exp(A)
    ! via R = v' exp(d) v, where [v,d]=eig(A) 
    implicit none
    ! input variables
    real(q),intent(in)     :: angle, axis(3)
    integer,intent(in)     :: invflag
    complex(q),intent(out) :: spinmat(2,2)
    ! local variables
    complex(q) :: thdotn(2,2), sig(2,2,3), ctmpmat(2,2)
    ! ZGEEV
    complex(q) :: zvl(2,2), zvr(2,2), zw(2), zwork(4)
    real(q)    :: rwork(4)
    integer    :: info
    ! Define Pauli-matrices
    sig=zero
    sig(1,2,1)=cmplx( 1.0_q, 0.0_q,kind=q)
    sig(2,1,1)=cmplx( 1.0_q, 0.0_q,kind=q)
    sig(1,2,2)=cmplx( 0.0_q,-1.0_q,kind=q)
    sig(2,1,2)=cmplx( 0.0_q, 1.0_q,kind=q)
    sig(1,1,3)=cmplx( 1.0_q, 0.0_q,kind=q)
    sig(2,2,3)=cmplx(-1.0_q, 0.0_q,kind=q)
    !---
    thdotn = &
         axis(1)*sig(:,:,1) + &
         axis(2)*sig(:,:,2) + &
         axis(3)*sig(:,:,3) 
    ctmpmat = thdotn*cmplx(0.0_q,-0.5_q*angle,kind=q)
    ! A=X(-1)dX
    ! exp(A)=X(-1)exp(d)X
    call zgeev( 'N', 'V', 2, ctmpmat, 2, zw, zvl, 1, &
         zvr, 2, zwork, 4, rwork, info)
    if(info .ne. 0) then
       write(*,*) 'error in zgeev. info=',info
       stop 'stop in get_spinorrot'
    end if
    ! set diagonal matrix
    ctmpmat=zero
    ctmpmat(1,1)=exp(zw(1)); ctmpmat(2,2)=exp(zw(2))
    ! transform this
    spinmat=matmul(zvr, matmul(ctmpmat,transpose(conjg(zvr))))
    ! did we have inversion? If so, put it back
    if(invflag .lt. 0) then
       spinmat=spinmat*cmplx(0.0_q,-1.0_q,kind=q)
    end if
  end subroutine get_spinorrot


  !-----------------------------
  ! Rotation subroutines follow
  !-----------------------------
  ! Externally callable routines
  !  rotate_wave_spin           - rotate 1 wfn in real and reciprocal space
  !   <-- W1_ROTATE_AND_FFT, W1_ROTATE_AND_FFT_NO_PROJ
  !  rotate_wave_spin_recip     - rotate 1 wfn in recip space (not done)
  !   <-- REALLOCATE_WAVE, CONTRACT_WAVE, APPLY_SMALL_SPACE_GROUP_OP
  !  rotate_wave_character_spin - rotate character of 1 wfn
  !   <-- REALLOCATE_WAVE, ROTATE_WAVE_CHARACTER
  ! 

  subroutine rotate_wave_spin(w_new, smat)
    !------------------------------------------------------
    ! Rotates spin-part of wfn in real and reciprocal space
    !
    use wave
    implicit none
    ! input arguments
    TYPE (wavefun1),intent(inout)   :: W_NEW         ! new wavefunction
    complex(q),intent(in)           :: smat(2,2)
    ! local variables
    integer    :: npl, nplcw
    complex(q),allocatable :: tmp(:), tmpcw(:)

!   if(.not. w_new%wdes1%lsorbit) return
    if(.not. w_new%wdes1%lnoncollinear) return
    
    npl     =  W_NEW%WDES1%GRID%MPLWV
    nplcw   =  W_NEW%WDES1%NGVECTOR 

    call rot_wave(w_new%CR, smat, npl)
    call rot_wave(w_new%CW, smat, nplcw)

  end subroutine rotate_wave_spin


  subroutine rotate_wave_spin_recip(wdes, smat, nk, spinflip, cw)
    !------------------------------------------------------
    ! Rotates spin-part of wfn in reciprocal space
    !
    use wave
    implicit none
    TYPE (wavedes),intent(in) ::     WDES
    complex(q),intent(in)     :: smat(2,2)
    integer, intent(in)       :: nk, spinflip
    complex(q),intent(inout)  :: cw(:) 
    ! local variables
    integer :: npl, nspinor
    
!   if(.not. wdes%lsorbit) return
    if(.not. wdes%lnoncollinear) return

    if(spinflip==1) then
       write(*,*) 'ERROR: spinflip encountered in rotate_wave_spin_recip'
       STOP
    end if

    npl=WDES%NGVECTOR(NK)
    nspinor=WDES%NRSPINORS

    call rot_wave(cw, smat, npl)

  end subroutine rotate_wave_spin_recip


  subroutine rot_wave(wf, smat, npl)
    !------------------------------------------------------
    ! Low-level wfn rotation
    !
    implicit none
    complex(q),intent(in)    :: smat(2,2)
    complex(q),intent(inout) :: wf(:)
    integer,intent(in)       :: npl
    ! local variables
    integer                :: ispinor, jspinor, nspinor, m
    complex(q),allocatable :: tmp(:), tmpcw(:)
    
    nspinor=2
    allocate(tmp(nspinor*npl))
    tmp=zero
    do ispinor = 0, nspinor-1
       do jspinor = 0, nspinor-1
          do m = 1, npl
             tmp(m+ispinor*npl) = tmp(m+ispinor*npl) &
                  + conjg(smat(jspinor+1,ispinor+1)) &
                  * wf(m+jspinor*npl)
          end do
       end do
    end do
!   wf=tmp
    wf(1:nspinor*npl)=tmp(1:nspinor*npl)
    deallocate(tmp)

  end subroutine rot_wave


  subroutine rotate_wave_character_spin(wdes1, cproj, smat)
    !------------------------------------------------------
    ! Rotates spin-part of wave-characters
    !
    use wave
    implicit none
    ! input arguments
    GDEF,intent(inout)              :: cproj(:)
    TYPE (wavedes1),intent(in)      :: WDES1    
    complex(q),intent(in)           :: smat(2,2)
    ! local variables
    integer    :: ispinor, jspinor, nspinor, m, nproj
    complex(q),allocatable :: tmpproj(:)

!   if(.not. wdes1%lsorbit) return
    if(.not. wdes1%lnoncollinear) return

    nspinor=WDES1%NRSPINORS
    nproj=wdes1%npro/nspinor
    allocate(tmpproj(nproj*nspinor))
    tmpproj=zero
    do ispinor = 0, nspinor-1
       do jspinor = 0, nspinor-1
          do m=1,nproj
             tmpproj(m+ispinor*nproj) = tmpproj(m+ispinor*nproj) &
                  + conjg(smat(jspinor+1,ispinor+1)) &
                  * cproj(m+jspinor*nproj)
          end do
       end do
    end do
    cproj(1:nproj*nspinor)=tmpproj
    deallocate(tmpproj)

  end subroutine rotate_wave_character_spin

end module spinsym
#endif    
