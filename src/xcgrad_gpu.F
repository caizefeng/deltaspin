#include "symbol.inc"
#define DO_NOT_USE_POTLOK_GPU
!************************ SUBROUTINE FEXCGC *****************************
! RCS:  $Id: xcgrad.F,v 1.4 2001/01/31 11:52:00 kresse Exp $
!
!  the latest version of the routine requires as input
!  the charge density in real space (CHTOT) and returns
!  the potential in real space (CWORK)
!
!  get GGA potential   (mind not LDA contribution is calculated)
!  this version supports vectorization if
!#define vector
!  is used.
!  In this case only PW-91 is supported, vectorization should be
!  possible by inlining up to 200 lines.
!  If you want to use other GGAs use
!#undef vector
!  Routine was written by jF, and rewritten  by aE and gK
!  to get the potential the algorithm proposed by
!  White and Bird Phys.Rev.B 50,7 (1994) 4954) is used
!  stress is also calculated according to this algorithm
!
! ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! We use a quite dangerous construction
! to support  REAL(q) <-> COMPLEX(q)   fft s
! several arrays are passed twice to the routine FEXCG_
! on some compilers this makes troubles,
! we call an external subroutine OPSYNC to avoid that compilers
! move DO Loops around violating our assumption that
! DWORK and CWORK point ot the same location
! (the OPSYNC subroutine actually does nothing at all)
! ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
!***********************************************************************
      MODULE xcgrad
      USE prec
      CONTAINS

      SUBROUTINE FEXCG(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
                  CHTOT,CWORK,DENCOR)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE setexm
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      COMPLEX(q)  CHTOT(GRIDC%MPLWV),CWORK(GRIDC%MPLWV)
      RGRID       DENCOR(GRIDC%RL%NP)
      REAL(q)     XCSIF(3,3)
! work arrays
      COMPLEX(q),ALLOCATABLE:: CWGRAD(:)
      REAL(q),ALLOCATABLE   :: DWORKG(:),DWORK1(:),DWORK2(:),DWORK3(:),DCHARG(:) 
!vdw jk
      REAL(q),ALLOCATABLE   :: DWORK4(:)
!vdw jk


      IF (.NOT. ISGGA()) THEN
         WRITE(*,*) 'internal ERROR:  FEXCGS called with non gradient corrected functional'
         STOP
      ENDIF

      NP1=GRIDC%RL%NP
      ALLOCATE(CWGRAD(GRIDC%MPLWV), &
               DWORKG(NP1),DWORK1(NP1),DWORK2(NP1),DWORK3(NP1),DCHARG(NP1))

      IF (.NOT.LUSE_VDW) THEN
         CALL FEXCG_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK, &
        &            CWGRAD,CHTOT,CWORK, &
        &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG)
      ELSE
!vdw jk
         ALLOCATE(DWORK4(NP1))
         CALL FEXCG_VDW_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK, &
        &            CWGRAD,CHTOT,CWORK, &
        &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG,DWORK4)
         DEALLOCATE(DWORK4)
!vdw jk
      ENDIF

      DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG)

      RETURN
      END SUBROUTINE

      !tb beg
      SUBROUTINE FEXCG_ddsc(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
                  CHTOT,CWORK,DENCOR,XDM,IVDW)
      USE prec
      USE lattice
      USE mpimy
      USE mgrid
      USE setexm
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

      COMPLEX(q)  CHTOT(GRIDC%MPLWV),CWORK(GRIDC%MPLWV)
      RGRID       DENCOR(GRIDC%RL%NP)
      REAL(q)     XCSIF(3,3)
! work arrays
      COMPLEX(q),ALLOCATABLE:: CWGRAD(:)
      REAL(q),ALLOCATABLE   :: DWORKG(:),DWORK1(:),DWORK2(:),DWORK3(:),DCHARG(:) 
      REAL(q) :: XDM(GRIDC%RL%NP)
      INTEGER :: IVDW

      IF (.NOT. ISGGA()) THEN
         WRITE(*,*) 'internal ERROR:  FEXCGS called with non gradient corrected functional'
         STOP
      ENDIF

      NP1=GRIDC%RL%NP
      ALLOCATE(CWGRAD(GRIDC%MPLWV), &
               DWORKG(NP1),DWORK1(NP1),DWORK2(NP1),DWORK3(NP1),DCHARG(NP1))

      !IF (IVDW==4) THEN
         CALL FEXCG_ddsc_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK, &
        &            CWGRAD,CHTOT,CWORK, &
        &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG,XDM,IVDW)
      !ENDIF

      DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG)

      RETURN
      END SUBROUTINE FEXCG_ddsc
      !tb end

      END MODULE

      SUBROUTINE FEXCG_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG)
      USE prec

      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE setexm

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION

! Mind CWORK and DWORK point actually to the same storagelocation
! similar to e EQUIVALENCE (CWORK(1),DWORK(1))
! same is true for (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
      COMPLEX(q) CHTOT(GRIDC%MPLWV),CWGRAD(GRIDC%MPLWV),CWORK(GRIDC%MPLWV)
      RGRID      DHTOT(DIMREAL(GRIDC%MPLWV)),DWGRAD(DIMREAL(GRIDC%MPLWV)),DWORK(DIMREAL(GRIDC%MPLWV))
      RGRID      DENCOR(GRIDC%RL%NP)
      REAL(q) DWORKG(GRIDC%RL%NP),DWORK1(GRIDC%RL%NP),DWORK2(GRIDC%RL%NP), &
              DWORK3(GRIDC%RL%NP),DCHARG(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)

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

#ifdef vector
      IF (LEXCH/=7) THEN
        do_io WRITE(*,*)'ERROR: only PW91 is supported if -Dvector is specified in the makefile'
        do_io WRITE(*,*)'       remove -Dvector from the makefile'
        STOP
      ENDIF
#endif

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
! get real charge density + core charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I)=(DENCOR(I)+DHTOT(I))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)
!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I)=CWORK(I)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I)=CWORK(I)*GX*CITPI
      ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! y-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I)=CWORK(I)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! z-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I)=CWORK(I)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO
! calculate total charge in real space, and abs nabla rho
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
         DCHARG(I)=(DHTOT(I)+DENCOR(I))/LATT_CUR%OMEGA
         G2=DWORK1(I)*DWORK1(I)+DWORK2(I)*DWORK2(I)+DWORK3(I)*DWORK3(I)
         DWORKG(I)=SQRT(G2)
      ENDDO
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is problematic
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*OMEGA)
!  the array DCHARG(I) is the real charge density (incl. part. core)
!=======================================================================

#ifdef DO_NOT_USE_POTLOK_GPU
      CALL GGAALL_GRID(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )
#else
#ifdef CUDA_GPU
      if (LEXCH .EQ. 7 .OR. LEXCH .EQ. 8 .OR. LEXCH .EQ. 9) then
        CALL GGAALL_GRID_GPU(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )
      else
        CALL GGAALL_GRID(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )
      endif
#else
   CALL GGAALL_GRID(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )
#endif
#endif


!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I)*DWORK1(I)*DWORK(I)
        SIF22=SIF22+DWORK2(I)*DWORK2(I)*DWORK(I)
        SIF33=SIF33+DWORK3(I)*DWORK3(I)*DWORK(I)
        SIF12=SIF12+DWORK1(I)*DWORK2(I)*DWORK(I)
        SIF23=SIF23+DWORK2(I)*DWORK3(I)*DWORK(I)
        SIF31=SIF31+DWORK3(I)*DWORK1(I)*DWORK(I)
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

      DO I=1,GRIDC%RL%NP
         DWORK1(I) = DWORK1(I)* REAL( DWORK(I) ,KIND=q)
         DWORK2(I) = DWORK2(I)* REAL( DWORK(I) ,KIND=q)
         DWORK3(I) = DWORK3(I)* REAL( DWORK(I) ,KIND=q)
      ENDDO
!=======================================================================
! times i G_k in reciprocal space...
!=======================================================================
! x-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I) = DWORK1(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I)=CWORK(I)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK2(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK3(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD,GRIDC)
      CALL FFT3D(CWGRAD,GRIDC,1)
      CALL OPSYNC(CWGRAD,DWGRAD,GRIDC%NPLWV)
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO= REAL( DCHARG(I) ,KIND=q)
         VXC=DWORKG(I)- REAL( DWGRAD(I) ,KIND=q) *RINPL
         DWORK(I)=VXC
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC*RHO*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC* REAL( DHTOT(I) ,KIND=q)
      ENDDO
! array reduction

      EXC   =EXC
      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC=EXC*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31

      CALLMPI( M_sum_s(GRIDC%COMM, 2, EXC, XCENC, 0._q , 0._q ))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
      ENDIF

      RETURN
      END

     !tb beg
      SUBROUTINE FEXCG_ddsc_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG,XDM,IVDW)
      USE prec

      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE setexm

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION

! Mind CWORK and DWORK point actually to the same storagelocation
! similar to e EQUIVALENCE (CWORK(1),DWORK(1))
! same is true for (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
      COMPLEX(q) CHTOT(GRIDC%MPLWV),CWGRAD(GRIDC%MPLWV),CWORK(GRIDC%MPLWV)
      RGRID      DHTOT(DIMREAL(GRIDC%MPLWV)),DWGRAD(DIMREAL(GRIDC%MPLWV)),DWORK(DIMREAL(GRIDC%MPLWV))
      RGRID      DENCOR(GRIDC%RL%NP)
      REAL(q) DWORKG(GRIDC%RL%NP),DWORK1(GRIDC%RL%NP),DWORK2(GRIDC%RL%NP), &
              DWORK3(GRIDC%RL%NP),DCHARG(GRIDC%RL%NP)
      REAL(q) XDM(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      INTEGER IVDW
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

#ifdef vector
      IF (LEXCH/=7) THEN
        do_io WRITE(*,*)'ERROR: only PW91 is supported if -Dvector is specified in the makefile'
        do_io WRITE(*,*)'       remove -Dvector from the makefile'
        STOP
      ENDIF
#endif

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
! get real charge density + core charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I)=(DENCOR(I)+DHTOT(I))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)
!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I)=CWORK(I)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I)=CWORK(I)*GX*CITPI
      ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! y-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I)=CWORK(I)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! z-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I)=CWORK(I)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO
! calculate total charge in real space, and abs nabla rho
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
         DCHARG(I)=(DHTOT(I)+DENCOR(I))/LATT_CUR%OMEGA
         G2=DWORK1(I)*DWORK1(I)+DWORK2(I)*DWORK2(I)+DWORK3(I)*DWORK3(I)
         DWORKG(I)=SQRT(G2)
      ENDDO
     
      !tb this IF is not needed anymore, this suroutine is called 
      ! only if IVDW==4
      !IF(IVDW.eq.4) then
      !  DO I=1,GRIDC%RL%NP
      !    CALL GGAapprox(ABS(DCHARG(I))*AUTOA3*0.5_q,DWORKG(I)*AUTOA4*0.5_q,XDM(I))
      !  ENDDO
      !ENDIF
      DO I=1,GRIDC%RL%NP
        CALL GGAapprox(ABS(DCHARG(I))*AUTOA3*0.5_q,DWORKG(I)*AUTOA4*0.5_q,XDM(I))
      ENDDO
      
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is problematic
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*OMEGA)
!  the array DCHARG(I) is the real charge density (incl. part. core)
!=======================================================================

      CALL GGAALL_GRID(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )

!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I)*DWORK1(I)*DWORK(I)
        SIF22=SIF22+DWORK2(I)*DWORK2(I)*DWORK(I)
        SIF33=SIF33+DWORK3(I)*DWORK3(I)*DWORK(I)
        SIF12=SIF12+DWORK1(I)*DWORK2(I)*DWORK(I)
        SIF23=SIF23+DWORK2(I)*DWORK3(I)*DWORK(I)
        SIF31=SIF31+DWORK3(I)*DWORK1(I)*DWORK(I)
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

      DO I=1,GRIDC%RL%NP
         DWORK1(I) = DWORK1(I)* REAL( DWORK(I) ,KIND=q)
         DWORK2(I) = DWORK2(I)* REAL( DWORK(I) ,KIND=q)
         DWORK3(I) = DWORK3(I)* REAL( DWORK(I) ,KIND=q)
      ENDDO
!=======================================================================
! times i G_k in reciprocal space...
!=======================================================================
! x-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I) = DWORK1(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I)=CWORK(I)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK2(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK3(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD,GRIDC)
      CALL FFT3D(CWGRAD,GRIDC,1)
      CALL OPSYNC(CWGRAD,DWGRAD,GRIDC%NPLWV)
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO= REAL( DCHARG(I) ,KIND=q)
         VXC=DWORKG(I)- REAL( DWGRAD(I) ,KIND=q) *RINPL
         DWORK(I)=VXC
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC*RHO*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC* REAL( DHTOT(I) ,KIND=q)
      ENDDO
! array reduction

      EXC   =EXC
      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC=EXC*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31

      CALLMPI( M_sum_s(GRIDC%COMM, 2, EXC, XCENC, 0._q , 0._q ))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
      ENDIF

      RETURN
      END SUBROUTINE FEXCG_ddsc_
     !tb end



      SUBROUTINE FEXCG_VDW_(GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DCHARG,DWORK4)
      USE prec

      USE lattice
      USE mpimy
      USE mgrid
      USE constant
      USE setexm
!vdw jk
      USE vdw_ll
!vdw jk

      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR

!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION

! Mind CWORK and DWORK point actually to the same storagelocation
! similar to e EQUIVALENCE (CWORK(1),DWORK(1))
! same is true for (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily
!
      COMPLEX(q) CHTOT(GRIDC%MPLWV),CWGRAD(GRIDC%MPLWV),CWORK(GRIDC%MPLWV)
      RGRID      DHTOT(DIMREAL(GRIDC%MPLWV)),DWGRAD(DIMREAL(GRIDC%MPLWV)),DWORK(DIMREAL(GRIDC%MPLWV))
      RGRID      DENCOR(GRIDC%RL%NP)
      REAL(q) DWORKG(GRIDC%RL%NP),DWORK1(GRIDC%RL%NP),DWORK2(GRIDC%RL%NP), &
              DWORK3(GRIDC%RL%NP),DCHARG(GRIDC%RL%NP)
!vdw jk
      REAL(q) DWORK4(GRIDC%RL%NP)
      REAL(q) stress(3,3)
!vdw jk
      REAL(q) XCSIF(3,3)

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

#ifdef vector
      IF (LEXCH/=7) THEN
        do_io WRITE(*,*)'ERROR: only PW91 is supported if -Dvector is specified in the makefile'
        do_io WRITE(*,*)'       remove -Dvector from the makefile'
        STOP
      ENDIF
#endif

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
! get real charge density + core charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I)=(DENCOR(I)+DHTOT(I))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)
!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I)=CWORK(I)
      ENDDO
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I)=CWORK(I)*GX*CITPI
      ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! y-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWORK(I)=CWORK(I)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO

! z-component:
      DO I=1,GRIDC%RC%NP
        CWORK(I)=CWGRAD(I)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWORK(I)=CWORK(I)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I)= REAL( DWORK(I) ,KIND=q)
      ENDDO
! calculate total charge in real space, and abs nabla rho
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
         DCHARG(I)=(DHTOT(I)+DENCOR(I))/LATT_CUR%OMEGA
         G2=DWORK1(I)*DWORK1(I)+DWORK2(I)*DWORK2(I)+DWORK3(I)*DWORK3(I)

         DWORKG(I)=SQRT(G2)
!vdw jk
         DWORK4(I)=SQRT(G2)
!vdw jk
      ENDDO
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is problematic
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*OMEGA)
!  the array DCHARG(I) is the real charge density (incl. part. core)
!=======================================================================

      CALL GGAALL_GRID(DCHARG(1), DWORKG(1), DWORK(1), EXC, GRIDC%RL%NP, LATT_CUR%OMEGA )
!vdw jk
      IF (LUSE_VDW) THEN
!        CALL VDW_NONLOC(DCHARG(1),DWORK(1),DWORKG(1),DWORK4(1),EXC,GRIDC,LATT_CUR,stress)
!PK Fix for illegal argument mismatch (vdw_nonloc interface is visible - no risk for temporaries)
        CALL VDW_NONLOC(DCHARG,DWORK,DWORKG,DWORK4,EXC,GRIDC,LATT_CUR,stress)
      ENDIF
!vdw jk

!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I)*DWORK1(I)*DWORK(I)
        SIF22=SIF22+DWORK2(I)*DWORK2(I)*DWORK(I)
        SIF33=SIF33+DWORK3(I)*DWORK3(I)*DWORK(I)
        SIF12=SIF12+DWORK1(I)*DWORK2(I)*DWORK(I)
        SIF23=SIF23+DWORK2(I)*DWORK3(I)*DWORK(I)
        SIF31=SIF31+DWORK3(I)*DWORK1(I)*DWORK(I)
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

      DO I=1,GRIDC%RL%NP
         DWORK1(I) = DWORK1(I)* REAL( DWORK(I) ,KIND=q)
         DWORK2(I) = DWORK2(I)* REAL( DWORK(I) ,KIND=q)
         DWORK3(I) = DWORK3(I)* REAL( DWORK(I) ,KIND=q)
      ENDDO
!=======================================================================
! times i G_k in reciprocal space...
!=======================================================================
! x-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I) = DWORK1(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I)=CWORK(I)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK2(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
         DWORK(I) = DWORK3(I)
      ENDDO
      CALL OPSYNC(DWORK,CWORK,GRIDC%NPLWV)
      CALL FFT3D(CWORK,GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK)

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I)=CWGRAD(I)+CWORK(I)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD,GRIDC)
      CALL FFT3D(CWGRAD,GRIDC,1)
      CALL OPSYNC(CWGRAD,DWGRAD,GRIDC%NPLWV)
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO= REAL( DCHARG(I) ,KIND=q)
         VXC=DWORKG(I)- REAL( DWGRAD(I) ,KIND=q) *RINPL
         DWORK(I)=VXC
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC*RHO*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC* REAL( DHTOT(I) ,KIND=q)
      ENDDO
! array reduction

      EXC   =EXC
      CVZERO=CVZERO*RINPL
      XCENC =(XCENC+EXC)*RINPL
      XCENCC=(XCENCC+EXC)*RINPL
      EXC=EXC*RINPL

      SIF11=SIF11-XCENCC
      SIF22=SIF22-XCENCC
      SIF33=SIF33-XCENCC
      XCSIF(1,1)=SIF11
      XCSIF(2,2)=SIF22
      XCSIF(3,3)=SIF33
      XCSIF(1,2)=SIF12
      XCSIF(2,1)=SIF12
      XCSIF(2,3)=SIF23
      XCSIF(3,2)=SIF23
      XCSIF(3,1)=SIF31
      XCSIF(1,3)=SIF31
      XCSIF=XCSIF-stress

      CALLMPI( M_sum_s(GRIDC%COMM, 2, EXC, XCENC, 0._q , 0._q ))
      CALLMPI( M_sum_d(GRIDC%COMM, XCSIF, 9))
      CALLMPI( M_sum_z(GRIDC%COMM,CVZERO,1))

! Test dumps:
      IF (IDUMP/=0) THEN
         WRITE(*,'(A,F24.14)') '<rho*excgc> =',EXC
         WRITE(*,'(A,F24.14)') '<rho*vxcgc> =',EXC-XCENC
         WRITE(*,'(A,F24.14)') '    xcencgc =',XCENC
      ENDIF

      RETURN
      END
