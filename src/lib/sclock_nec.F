!***********************************************************************
!
!    Timer-Routine for NEC 
!
!***********************************************************************

      SUBROUTINE VTIME(VPUTIM,CPUTIM)
      USE preclib
      IMPLICIT REAL(q) (A-H,O-Z)
      CALL CLOCK( CPUTIM )
      VPUTIM=VCLOCK() 
      RETURN
      END
      real*8 function vclock()
      real*8 t
      integer *8 i1,ir,im,ishft
      integer *8 itmtv
      intrinsic ishft
      i1=itmtv()
      ir=ishft(ishft(i1,32),-32)
      im=ishft(i1,-32)
      t=(qext(ir)+qext(im)*4294967296.d0)*4.0D-9
      vclock=t
      return
      end
