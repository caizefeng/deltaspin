#include "symbol.inc"
!************************ SUBROUTINE FEXGCS *****************************
! RCS:  $Id: xcspin.F,v 1.6 2003/06/27 13:22:24 kresse Exp kresse $
!
!  the latest version of the routine requires as input
!  the charge density in real space (CHTOT) and returns
!  the potential in real space (CWORK)
!
!  Routine was written by Elio Moroni, and rewritten to f90 by gK
!  to get the potentials the algorithm proposed by
!  White and Bird Phys.Rev.B 50,7 (1994) 4954) is used
!  stress is also calculated according to this algorithm
!
! ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! We use a quite dangerous construction
! to support  REAL(q) <-> COMPLEX(q)   ffts
! several arrays are passed twice to the routine FEXCG_
! on some compilers this makes troubles,
! we call an external subroutine OPSYNC to avoid that compilers
! move DO Loops around violating our assumption that
! DWORK and CWORK point to the same location in storage
! (the OPSYNC subroutine actually does nothing at all)
! ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
!*********** ************************************************************

      SUBROUTINE FEXCGS(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ),CWORK(GRIDC%MPLWV,NCDIJ)
      RGRID      DENCOR(GRIDC%RL%NP)
      DIMENSION XCSIF(3,3)
! work arrays
      COMPLEX(q),ALLOCATABLE:: CWGRAD(:,:)
      REAL(q),ALLOCATABLE   :: DWORKG(:,:),DWORK1(:,:),DWORK2(:,:),DWORK3(:,:), &
     &                      DVC(:)
!vdw jk
      REAL(q),ALLOCATABLE :: DWORK4(:), DWORK6(:), DWORK7(:)
      RGRID ,ALLOCATABLE ::  DWORK5(:)
!vdw jk
      

      IF (.NOT. ISGGA()) THEN
         WRITE(*,*) 'internal ERROR:  FEXCGS called with non gradient corrected functional'
         STOP
      ENDIF

      NP1=GRIDC%RL%NP
      IF (NCDIJ==2) THEN
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))
         IF (.NOT.LUSE_VDW) THEN
            CALL FEXCGS_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
           &            CWGRAD,CHTOT,CWORK, &
           &            CWGRAD,CHTOT,CWORK, &
           &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC)
         ELSE
!vdw jk
            ALLOCATE(DWORK4(NP1),DWORK5(NP1),DWORK6(NP1),DWORK7(NP1) )

            CALL FEXCGS_VDW_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
           &            CWGRAD,CHTOT,CWORK, &
           &            CWGRAD,CHTOT,CWORK, &
           &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC, &
           &            DWORK4,DWORK5,DWORK6,DWORK7)

            DEALLOCATE(DWORK4,DWORK5,DWORK6,DWORK7)
!vdw jk
         ENDIF
         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC) 
      ELSEIF (NCDIJ==4) THEN
!-MM- gradient corrections in the noncollinear case are calculated
!     a bit differently than in the collinear case
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ/2), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))

         CALL FEXCGS_NONCOL_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK, &
        &            CWGRAD,CHTOT,CWORK, &
        &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC)

         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC)         
!-MM- end of changes to calculation of gga in noncollinear case
      ENDIF
      
      RETURN
      END


!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily

      SUBROUTINE FEXCGS_(ISPIN,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC)
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,ISPIN),CWORK(GRIDC%MPLWV,ISPIN), &
              CWGRAD(GRIDC%MPLWV,ISPIN)
      RGRID   DHTOT(DIMREAL(GRIDC%MPLWV),ISPIN),DWORK(DIMREAL(GRIDC%MPLWV),ISPIN), &
              DWGRAD(DIMREAL(GRIDC%MPLWV),ISPIN)
      RGRID   DENCOR(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      REAL(q) DWORKG(GRIDC%RL%NP,ISPIN),DWORK1(GRIDC%RL%NP,ISPIN), &
              DWORK2(GRIDC%RL%NP,ISPIN),DWORK3(GRIDC%RL%NP,ISPIN), &
              DVC(GRIDC%RL%NP)

      REAL(q) :: CHGMIN=1E-10
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
! 
! only for the analytical PBE functional we can use a low charge density threshold
! since numerical differentiation leads to errors for the other functionals
!
! jP: adding PBEsol
      IF (LEXCH==8 .OR. LEXCH==9 .OR. LEXCH==14 .OR. LEXCH==17) THEN
! jP: adding PBEsol
         CHGMIN=1E-10
      ELSE
         CHGMIN=1E-7
      ENDIF

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
      spin: DO ISP=1,ISPIN
! set CWORK to total real charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I,ISP)=(DENCOR(I)/2+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
!
! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO


! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
      ENDDO spin
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0
      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))

         ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))

         ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
         ABSNAB=SQRT(ABSNAB)
!
! presently VASP coverges very badly if correlation_ABS_DRHOUP_ABS_DRHOD
! is not set
! it is probably the exchange potential, which is partly canceled by
! the correlation energy if ABSNAB=ABSNABUP+ABSNABDW
! so let us stick to ABSNAB=ABSNABUP+ABSNABDW for the time beeing

!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         ABSNAB=ABSNABUP+ABSNABDW
#endif
!
         CALL GGASPIN(RHO1*AUTOA3,RHO2*AUTOA3, &
     &               ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
     &           EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,.FALSE.)
!
         RHO=RHO1+RHO2

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif
!test
!         DVXC1=0
!         DVXC2=0
!         DVC_ =0
!test
!
!   store d f/ d (|d rho| ) / |d rho|  in DWORK
!
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP,1.E-10_q)
         DWORK(I,2)  = DVXC2 / MAX(ABSNABDW,1.E-10_q)
         DVC(I)      = DVC_  / MAX(ABSNAB,1.E-10_q)
!
!   store d f/ d rho  in DWORKG
!
         DWORKG(I,1) = DEXC1*RYTOEV
         DWORKG(I,2) = DEXC2*RYTOEV
      ENDDO
#ifdef debug
      REWIND(77)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
 
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)


            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
      REWIND(77)
      REWIND(78)
      REWIND(79)
#endif
!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
!
!    verify that this sum is correct also when ISPIN=2
!
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO ISP=1,ISPIN
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
        SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)

        SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
        SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
      ENDDO
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      DO I=1,GRIDC%RL%NP
         ANAB1U= DWORK1(I,1)
         ANAB2U= DWORK2(I,1)
         ANAB3U= DWORK3(I,1)
         ANAB1D= DWORK1(I,2)
         ANAB2D= DWORK2(I,2)
         ANAB3D= DWORK3(I,2)

         DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

         DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
      ENDDO

      spin2: DO ISP=1,ISPIN
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)
!
      ENDDO spin2


!
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
         VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
         DWORK(I,1)=VXC1
         DWORK(I,2)=VXC2
         VXC = 0.5_q*(VXC1+VXC2)
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q) -VXC2* REAL( DHTOT(I,2) ,KIND=q)
      ENDDO

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
      END SUBROUTINE

!tb beg
      subroutine GGAapprox(Rho,nablaRho,b)
      USE prec
! evaluates a GGA approximation to b2
! Parameters for GGA-approx
       REAL(q) s,b,rs ! reduced density gradient, b, Wigner-Seitz radius,
       REAL(q) Rho,nablaRho
       REAL(q),Parameter :: pi=3.141592653589793_q,P1=2._q,P2=1._q
       REAL(q),Parameter :: DCS=(1.D0/(2.D0*((3.0_q*(PI**2))**(1._q/3._q))))
       if((RHO.gt.1.E-10_q).and.(nablaRho.gt.1.E-10_q))then
       RS=(3._q/(4._q*pi*Rho))**(1._q/3._q) !Wigner-Seitz radius
       S=(DCS*nablaRho)/(Rho**(4._q/3._q))

!       P1=2.0 !0.97 !1.0 !2.0
!       P2=1.0 !0.57 !0.6057068643 !1.0
       b=P1*s*exp(-P2*s) !ExpExp
       b=b*rs
       b=b*b ! Our formula approximates b, but we would like to return b^2
!       b=b*rho ! this is the quantity we need for M2, so let's compute it right away
!      However, it is preferable to multiply by the total density, i.e., including the core contribution
!      Hence, we do this step in the xxx_planeDOM routine
       else
!       write(*,*) RHO,nablaRho
       b=0._q
       endif
!       b=rho
       return
       end

      SUBROUTINE FEXCGS_ddsc(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ),CWORK(GRIDC%MPLWV,NCDIJ)
      RGRID      DENCOR(GRIDC%RL%NP)
      DIMENSION XCSIF(3,3)
! work arrays
      COMPLEX(q),ALLOCATABLE:: CWGRAD(:,:)
      REAL(q),ALLOCATABLE   :: DWORKG(:,:),DWORK1(:,:),DWORK2(:,:),DWORK3(:,:), &
     &                      DVC(:)
!vdw jk
      REAL(q),ALLOCATABLE :: DWORK4(:), DWORK6(:), DWORK7(:)
      RGRID ,ALLOCATABLE ::  DWORK5(:)
!vdw jk
     REAL(q) :: XDM(GRIDC%RL%NP) 
     INTEGER :: IVDW

      IF (.NOT. ISGGA()) THEN
         WRITE(*,*) 'internal ERROR:  FEXCGS called with non gradient corrected functional'
         STOP
      ENDIF


      NP1=GRIDC%RL%NP
      IF (NCDIJ==2) THEN
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))
            CALL FEXCGS_ddsc_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
           &            CWGRAD,CHTOT,CWORK, &
           &            CWGRAD,CHTOT,CWORK, &
           &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC,XDM,IVDW)
         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC) 
      ELSEIF (NCDIJ==4) THEN
!-MM- gradient corrections in the noncollinear case are calculated
!     a bit differently than in the collinear case
         ALLOCATE(CWGRAD(GRIDC%MPLWV,NCDIJ), DWORKG(NP1,NCDIJ/2), &
           DWORK1(NP1,NCDIJ),DWORK2(NP1,NCDIJ),DWORK3(NP1,NCDIJ),DVC(NP1))

         CALL FEXCGS_NONCOL_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
        &            CWGRAD,CHTOT,CWORK, &
        &            CWGRAD,CHTOT,CWORK, &
        &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC)

         DEALLOCATE(CWGRAD,DWORKG,DWORK1,DWORK2,DWORK3,DVC)         
!-MM- end of changes to calculation of gga in noncollinear case
      ENDIF
      
      RETURN
      END SUBROUTINE FEXCGS_ddsc

      SUBROUTINE FEXCGS_ddsc_(ISPIN,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC,XDM,IVDW)
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,ISPIN),CWORK(GRIDC%MPLWV,ISPIN), &
              CWGRAD(GRIDC%MPLWV,ISPIN)
      RGRID   DHTOT(DIMREAL(GRIDC%MPLWV),ISPIN),DWORK(DIMREAL(GRIDC%MPLWV),ISPIN), &
              DWGRAD(DIMREAL(GRIDC%MPLWV),ISPIN)
      RGRID   DENCOR(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3),XDM(GRIDC%RL%NP)
      REAL(q) DWORKG(GRIDC%RL%NP,ISPIN),DWORK1(GRIDC%RL%NP,ISPIN), &
              DWORK2(GRIDC%RL%NP,ISPIN),DWORK3(GRIDC%RL%NP,ISPIN), &
              DVC(GRIDC%RL%NP)
      INTEGER :: IVDW
      REAL(q) :: CHGMIN=1E-10,tXDM
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
! 
! only for the analytical PBE functional we can use a low charge density threshold
! since numerical differentiation leads to errors for the other functionals
!
! jP: adding PBEsol
      IF (LEXCH==8 .OR. LEXCH==9 .OR. LEXCH==14 .OR. LEXCH==17) THEN
! jP: adding PBEsol
         CHGMIN=1E-10
      ELSE
         CHGMIN=1E-7
      ENDIF

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
      spin: DO ISP=1,ISPIN
! set CWORK to total real charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I,ISP)=(DENCOR(I)/2+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
!
! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO


! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
      ENDDO spin
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0
      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))

         ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))

         ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
         ABSNAB=SQRT(ABSNAB)
         if(IVDW.eq.4) then
         CALL GGAapprox(ABS(RHO1)*AUTOA3,ABSNABUP*AUTOA4,XDM(I))
         CALL GGAapprox(ABS(RHO2)*AUTOA3,ABSNABDW*AUTOA4,tXDM)
         XDM(I)=(XDM(I)+tXDM)*0.5_q
         endif
!
! presently VASP coverges very badly if correlation_ABS_DRHOUP_ABS_DRHOD
! is not set
! it is probably the exchange potential, which is partly canceled by
! the correlation energy if ABSNAB=ABSNABUP+ABSNABDW
! so let us stick to ABSNAB=ABSNABUP+ABSNABDW for the time beeing

!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         ABSNAB=ABSNABUP+ABSNABDW
#endif
!
         CALL GGASPIN(RHO1*AUTOA3,RHO2*AUTOA3, &
     &               ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
     &           EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,.FALSE.)
!
         RHO=RHO1+RHO2

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif
!test
!         DVXC1=0
!         DVXC2=0
!         DVC_ =0
!test
!
!   store d f/ d (|d rho| ) / |d rho|  in DWORK
!
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP,1.E-10_q)
         DWORK(I,2)  = DVXC2 / MAX(ABSNABDW,1.E-10_q)
         DVC(I)      = DVC_  / MAX(ABSNAB,1.E-10_q)
!
!   store d f/ d rho  in DWORKG
!
         DWORKG(I,1) = DEXC1*RYTOEV
         DWORKG(I,2) = DEXC2*RYTOEV
      ENDDO
      if(IVDW.eq.4) then
      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))

         ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))

         ABSNAB=SQRT(ABSNAB)
         CALL GGAapprox(ABS(RHO1)*AUTOA3,ABSNABUP*AUTOA4,XDM(I))
         CALL GGAapprox(ABS(RHO2)*AUTOA3,ABSNABDW*AUTOA4,tXDM)
         XDM(I)=(XDM(I)+tXDM)*0.5_q
      ENDDO
      endif
#ifdef debug
      REWIND(77)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
 
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)


            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
      REWIND(77)
      REWIND(78)
      REWIND(79)
#endif
!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
!
!    verify that this sum is correct also when ISPIN=2
!
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO ISP=1,ISPIN
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
        SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)

        SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
        SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
      ENDDO
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      DO I=1,GRIDC%RL%NP
         ANAB1U= DWORK1(I,1)
         ANAB2U= DWORK2(I,1)
         ANAB3U= DWORK3(I,1)
         ANAB1D= DWORK1(I,2)
         ANAB2D= DWORK2(I,2)
         ANAB3D= DWORK3(I,2)

         DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

         DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
      ENDDO

      spin2: DO ISP=1,ISPIN
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)
!
      ENDDO spin2


!
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
         VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
         DWORK(I,1)=VXC1
         DWORK(I,2)=VXC2
         VXC = 0.5_q*(VXC1+VXC2)
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q) -VXC2* REAL( DHTOT(I,2) ,KIND=q)
      ENDDO

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
      END SUBROUTINE FEXCGS_ddsc_

!tb end

!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily

      SUBROUTINE FEXCGS_NONCOL_(NCDIJ,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC)
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,NCDIJ),CWORK(GRIDC%MPLWV,NCDIJ), &
              CWGRAD(GRIDC%MPLWV,NCDIJ)
      RGRID   DHTOT(DIMREAL(GRIDC%MPLWV),NCDIJ),DWORK(DIMREAL(GRIDC%MPLWV),NCDIJ), &
              DWGRAD(DIMREAL(GRIDC%MPLWV),NCDIJ)
      RGRID   DENCOR(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      REAL(q) DWORKG(GRIDC%RL%NP,NCDIJ/2),DWORK1(GRIDC%RL%NP,NCDIJ), &
              DWORK2(GRIDC%RL%NP,NCDIJ),DWORK3(GRIDC%RL%NP,NCDIJ), &
              DVC(GRIDC%RL%NP)
      REAL(q) MAG_NORM,NABMAG(3)
      REAL(q) :: CHGMIN=1E-10

! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
! 
! only for the analytical PBE functional we can use a low charge density threshold
! since numerical differentiation leads to errors for the other functionals
!
! jP: adding PBEsol
      IF (LEXCH==8 .OR. LEXCH==9 .OR. LEXCH==14 .OR. LEXCH==17) THEN
! jP: adding PBEsol
         CHGMIN=1E-10
      ELSE
         CHGMIN=1E-7
      ENDIF

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
      spin: DO ISP=1,NCDIJ
      IF (ISP==1) THEN
! Add core charge density to pseudo charge density
         DO I=1,GRIDC%RL%NP
            DWORK(I,ISP)=(DENCOR(I)+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
         ENDDO
      ELSE
         DO I=1,GRIDC%RL%NP
            DWORK(I,ISP)=DHTOT(I,ISP)*RINPL/LATT_CUR%OMEGA
         ENDDO
      ENDIF
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))
!=======================================================================
! now calculate the gradients
!=======================================================================
! x-component:
      DO I=1,GRIDC%RC%NP
         CWGRAD(I,ISP)=CWORK(I,ISP)
      ENDDO
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)
         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
         DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
!
! y-component:
      DO I=1,GRIDC%RC%NP
         CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
         DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
!
! z-component:
      DO I=1,GRIDC%RC%NP
         CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
         DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
      ENDDO spin

!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0
      DO I=1,GRIDC%RL%NP
!
! | m |
!
         MAG_NORM=MAX(SQRT(ABS( & 
        &    DHTOT(I,2)*DHTOT(I,2)+DHTOT(I,3)*DHTOT(I,3)+DHTOT(I,4)*DHTOT(I,4))), 1E-10_q)        
!
! RHO1: \rho_up   = ( \rho_tot + | m | )/2
! RHO2: \rho_down = ( \rho_tot - | m | )/2
!
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)+MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,1)+DENCOR(I)-MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
!
! \nabla | m |
!
         NABMAG(1)=(DWORK1(I,2)*DHTOT(I,2)+DWORK1(I,3)*DHTOT(I,3)+DWORK1(I,4)*DHTOT(I,4))/MAG_NORM
         NABMAG(2)=(DWORK2(I,2)*DHTOT(I,2)+DWORK2(I,3)*DHTOT(I,3)+DWORK2(I,4)*DHTOT(I,4))/MAG_NORM
         NABMAG(3)=(DWORK3(I,2)*DHTOT(I,2)+DWORK3(I,3)*DHTOT(I,3)+DWORK3(I,4)*DHTOT(I,4))/MAG_NORM
!
! | ( \nabla \rho + \nabla | m | )/2 |
!
         ABSNABUP=SQRT( &
        &            (DWORK1(I,1)+NABMAG(1))*(DWORK1(I,1)+NABMAG(1)) + &
        &             (DWORK2(I,1)+NABMAG(2))*(DWORK2(I,1)+NABMAG(2)) + &
        &              (DWORK3(I,1)+NABMAG(3))*(DWORK3(I,1)+NABMAG(3)) ) / 2
!
! | ( \nabla \rho - \nabla | m | )/2 |
!
         ABSNABDW=SQRT( &
        &            (DWORK1(I,1)-NABMAG(1))*(DWORK1(I,1)-NABMAG(1)) + &
        &             (DWORK2(I,1)-NABMAG(2))*(DWORK2(I,1)-NABMAG(2)) + &
        &              (DWORK3(I,1)-NABMAG(3))*(DWORK3(I,1)-NABMAG(3)) ) / 2
!
! | \nabla \rho |
!
         ABSNAB=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1)+DWORK3(I,1)*DWORK3(I,1))
!
! Refill DWORK[1..3](:,2) with ( \nabla \rho - \nabla | m | )/2
!
         DWORK1(I,2)=(DWORK1(I,1)-NABMAG(1))/2
         DWORK2(I,2)=(DWORK2(I,1)-NABMAG(2))/2
         DWORK3(I,2)=(DWORK3(I,1)-NABMAG(3))/2
!
! Refill DWORK[1..3](:,1) with ( \nabla \rho + \nabla | m | )/2
!
         DWORK1(I,1)=(DWORK1(I,1)+NABMAG(1))/2
         DWORK2(I,1)=(DWORK2(I,1)+NABMAG(2))/2
         DWORK3(I,1)=(DWORK3(I,1)+NABMAG(3))/2
!
! presently VASP converges very badly if correlation_ABS_DRHOUP_ABS_DRHOD
! is not set
! it is probably the exchange potential, which is partly canceled by
! the correlation energy if ABSNAB=ABSNABUP+ABSNABDW
! so let us stick to ABSNAB=ABSNABUP+ABSNABDW for the time beeing

!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         ABSNAB=ABSNABUP+ABSNABDW
#endif

         CALL GGASPIN(RHO1*AUTOA3,RHO2*AUTOA3, &
     &               ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
     &           EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,.FALSE.)

         RHO=RHO1+RHO2

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif

!
!   store d f/ d (|d rho| ) / |d rho|  in DWORK
!
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP,1.E-10_q)
         DWORK(I,2)  = DVXC2 / MAX(ABSNABDW,1.E-10_q)
         DVC(I)      = DVC_  / MAX(ABSNAB,1.E-10_q)
!
!   store d f/ d rho  in DWORKG
!
         DWORKG(I,1) = DEXC1*RYTOEV
         DWORKG(I,2) = DEXC2*RYTOEV
      ENDDO
#ifdef debug
      REWIND(77)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
 
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)


            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
      REWIND(77)
      REWIND(78)
      REWIND(79)
#endif
!      stop
!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
!
!    verify that this sum is correct also when ISPIN=2
!
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO ISP=1,2
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
        SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)

        SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
        SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
      ENDDO
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      DO I=1,GRIDC%RL%NP
         ANAB1U= DWORK1(I,1)
         ANAB2U= DWORK2(I,1)
         ANAB3U= DWORK3(I,1)
         ANAB1D= DWORK1(I,2)
         ANAB2D= DWORK2(I,2)
         ANAB3D= DWORK3(I,2)

         DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

         DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
      ENDDO

      spin2: DO ISP=1,2
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)
!
      ENDDO spin2


!
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
!
! | m |
!
         MAG_NORM=SQRT(ABS(DHTOT(I,2)*DHTOT(I,2)+ DHTOT(I,3)*DHTOT(I,3) + DHTOT(I,4)*DHTOT(I,4)))
!
! RHO1: \rho_up   = ( \rho_tot + | m | )/2
! RHO2: \rho_down = ( \rho_tot - | m | )/2
!
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)+MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,1)+DENCOR(I)-MAG_NORM)/2/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
         VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
         DWORK(I,1)=VXC1
         DWORK(I,2)=VXC2
         VXC = 0.5_q*(VXC1+VXC2)
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC1* REAL( (DHTOT(I,1)+MAG_NORM)/2 ,KIND=q) &
        &               -VXC2* REAL( (DHTOT(I,1)-MAG_NORM)/2 ,KIND=q)
      ENDDO

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
      END SUBROUTINE


!  ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
! Mind CWORK and DWORK point actually to the same storagelocation
! similar to an EQUIVALENCE (CWORK(1),DWORK(1))
! the same for  (CWGRAD,DWGRAD) and   (CHTOT,DHTOT)
! so we can interchange both arrays arbitrarily

!vdw jk
      SUBROUTINE FEXCGS_VDW_(ISPIN,GRIDC,LATT_CUR,XCENC,EXC,CVZERO,XCSIF, &
     &            CWGRAD,CHTOT,CWORK, &
     &            DWGRAD,DHTOT,DWORK, &
     &            DENCOR,DWORKG,DWORK1,DWORK2,DWORK3,DVC, &
     &            DWORK4,DWORK5,DWORK6,DWORK7)

!vdw jk
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

      COMPLEX(q) CHTOT(GRIDC%MPLWV,ISPIN),CWORK(GRIDC%MPLWV,ISPIN), &
              CWGRAD(GRIDC%MPLWV,ISPIN)
      RGRID   DHTOT(DIMREAL(GRIDC%MPLWV),ISPIN),DWORK(DIMREAL(GRIDC%MPLWV),ISPIN), &
              DWGRAD(DIMREAL(GRIDC%MPLWV),ISPIN)
      RGRID   DENCOR(GRIDC%RL%NP)
      REAL(q) XCSIF(3,3)
      REAL(q) stress(3,3)

      REAL(q) DWORKG(GRIDC%RL%NP,ISPIN),DWORK1(GRIDC%RL%NP,ISPIN), &
              DWORK2(GRIDC%RL%NP,ISPIN),DWORK3(GRIDC%RL%NP,ISPIN), &
              DVC(GRIDC%RL%NP)
!vdw jk
      REAL(q) DWORK4(GRIDC%RL%NP),DWORK6(GRIDC%RL%NP),DWORK7(GRIDC%RL%NP)
      RGRID   DWORK5(GRIDC%RL%NP)

!vdw jk

      REAL(q) :: CHGMIN=1E-10
! set to one for error-dumps
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
      IDUMP=0
#ifdef debug
      IF (NODE_ME==IONODE) IDUMP=1
#endif
#else
      IDUMP=0
#ifdef debug
      IDUMP=1
#endif
#endif
! 
! only for the analytical PBE functional we can use a low charge density threshold
! since numerical differentiation leads to errors for the other functionals
!
! jP: adding PBEsol
!vdw jk
      IF (LEXCH==8 .OR. LEXCH==9 .OR. LEXCH==14 .OR. LEXCH==17 .OR. ((LEXCH.ge.40) .AND. (LEXCH.le.50))) THEN
!vdw jk
! jP: adding PBEsol
         CHGMIN=1E-10
      ELSE
         CHGMIN=1E-7
      ENDIF

      RINPL=1._q/GRIDC%NPLWV
!=======================================================================
! First phase: Transform DENCOR (core charge) and
!  CHTOT (pseudo chargedensity) to real space
!=======================================================================
      spin: DO ISP=1,ISPIN
! set CWORK to total real charge in reciprocal space
      DO I=1,GRIDC%RL%NP
         DWORK(I,ISP)=(DENCOR(I)/2+DHTOT(I,ISP))*RINPL/LATT_CUR%OMEGA
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

!=======================================================================
! now calculate the gradient of the chargedensity
!=======================================================================
      DO  I=1,GRIDC%RC%NP
        CWGRAD(I,ISP)=CWORK(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GX*CITPI
       ENDDO
! grad_x in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK1(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
!
! y-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GY*CITPI
      ENDDO
! grad_y in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK2(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO


! z-component:
      DO  I=1,GRIDC%RC%NP
        CWORK(I,ISP)=CWGRAD(I,ISP)
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
         CWORK(I,ISP)=CWORK(I,ISP)*GZ*CITPI
      ENDDO
! grad_z in real space:
      CALL SETUNB(CWORK(1,ISP),GRIDC)
      CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      DO I=1,GRIDC%RL%NP
        DWORK3(I,ISP)= REAL( DWORK(I,ISP) ,KIND=q)
      ENDDO
      ENDDO spin
!=======================================================================
!  grad rho    d    f_xc
! ---------- * ------------      (Phys.Rev.B 50,7 (1994) 4954)
! |grad rho|   d |grad rho|
!
!  MIND: the factor OMEGA is difficult to understand:
!   1/N sum_r energy_density * rho *OMEGA = Energy
!   1/N sum_r energy_density * \bar rho   = Energy (\bar rho=rho*LATT_CUR%OMEGA)
!=======================================================================
      EXC=0
      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))

         ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))

         ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
         ABSNAB=SQRT(ABSNAB)
!
! presently VASP coverges very badly if correlation_ABS_DRHOUP_ABS_DRHOD
! is not set
! it is probably the exchange potential, which is partly canceled by
! the correlation energy if ABSNAB=ABSNABUP+ABSNABDW
! so let us stick to ABSNAB=ABSNABUP+ABSNABDW for the time beeing

!#define  correlation_ABS_DRHOUP_ABS_DRHOD
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         ABSNAB=ABSNABUP+ABSNABDW
#endif
!vdw jk
!store the total gradient for later use
         DWORK7(i)=ABSNAB
!vdw jk
!
         CALL GGASPIN(RHO1*AUTOA3,RHO2*AUTOA3, &
     &               ABSNABUP*AUTOA4, ABSNABDW*AUTOA4,ABSNAB*AUTOA4, &
     &           EXCL,DEXC1,DEXC2,DVXC1,DVXC2,DVC_,.FALSE.)
!
         RHO=RHO1+RHO2

         EXC=EXC+EXCL*RHO*RYTOEV*LATT_CUR%OMEGA
         DVXC1=DVXC1*RYTOEV*AUTOA
         DVXC2=DVXC2*RYTOEV*AUTOA
         DVC_ =DVC_ *RYTOEV*AUTOA
#ifdef correlation_ABS_DRHOUP_ABS_DRHOD
         DVXC1=DVXC1+DVC_ 
         DVXC2=DVXC2+DVC_ 
         DVC_=0
#endif
!test
!         DVXC1=0
!         DVXC2=0
!         DVC_ =0
!test
!
!   store d f/ d (|d rho| ) / |d rho|  in DWORK
!
         DWORK(I,1)  = DVXC1 / MAX(ABSNABUP,1.E-10_q)
         DWORK(I,2)  = DVXC2 / MAX(ABSNABDW,1.E-10_q)
         DVC(I)      = DVC_  / MAX(ABSNAB,1.E-10_q)
!
!   store d f/ d rho  in DWORKG
!
         DWORKG(I,1) = DEXC1*RYTOEV
         DWORKG(I,2) = DEXC2*RYTOEV
      ENDDO
!vdw jk
!store total charge in DWORK4
      DO I=1,GRIDC%RL%NP
         DWORK4(I)=MAX(REAL((DENCOR(I)+DHTOT(I,1)+DHTOT(I,2))/LATT_CUR%OMEGA, kind=q),CHGMIN)
      ENDDO
!tot grad needs to be in DWORK7, done in previous loop
      DWORK5=0._q
      DWORK6=0._q
      IF (LUSE_VDW) THEN
!        CALL VDW_NONLOC_spin( DWORK4(1), DWORK5(1), DWORK6(1), DWORK7(1),EXC,GRIDC,LATT_CUR,stress)
!PK Fix for illegal argument mismatch (vdw_nonloc interface is visible - no risk for temporaries)
        CALL VDW_NONLOC_spin( DWORK4, DWORK5, DWORK6, DWORK7,EXC,GRIDC,LATT_CUR,stress)
      ENDIF

      DO I=1,GRIDC%RL%NP
        DWORK(I,1)=DWORK(I,1)+DWORK5(i)/ MAX(SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1)),1.E-10_q)
        DWORK(I,2)=DWORK(I,2)+DWORK5(i)/ MAX(SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2)),1.E-10_q)
      ENDDO

      DO I=1,GRIDC%RL%NP
        DWORKG(I,1)=DWORKG(I,1)+DWORK6(i)
        DWORKG(I,2)=DWORKG(I,2)+DWORK6(i)
      ENDDO


#ifdef debug
      REWIND(77)
      I=1
      DO NY=1,GRIDC%NGY
         DO NX=1,GRIDC%NGX
            ABSNAB= (DWORK1(I,1)+DWORK1(I,2))*(DWORK1(I,1)+DWORK1(I,2))+ &
                 (DWORK2(I,1)+DWORK2(I,2))*(DWORK2(I,1)+DWORK2(I,2))+ &
                 (DWORK3(I,1)+DWORK3(I,2))*(DWORK3(I,1)+DWORK3(I,2))
            ABSNABUP=SQRT(DWORK1(I,1)*DWORK1(I,1)+DWORK2(I,1)*DWORK2(I,1) &
                   +DWORK3(I,1)*DWORK3(I,1))
 
            ABSNABDW=SQRT(DWORK1(I,2)*DWORK1(I,2)+DWORK2(I,2)*DWORK2(I,2) &
                   +DWORK3(I,2)*DWORK3(I,2))
            RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
            RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)


            WRITE(77,'(2I4,3F14.3)') NX,NY,DWORK(I,1)*ABSNABUP,RHO1
            WRITE(78,'(3F14.3)') DWORK(I,2)*ABSNABDW
            WRITE(79,'(3F14.3)') DVC(I)*ABSNAB

            I=I+1
         ENDDO
      ENDDO
      REWIND(77)
      REWIND(78)
      REWIND(79)
#endif
!=======================================================================
! gradient terms in stress tensor
!          d    f_xc     grad rho  x grad rho
! sum_r   ------------   --------------------- * LATT_CUR%OMEGA
!         d |grad rho|        |grad rho|
!=======================================================================
!
!    verify that this sum is correct also when ISPIN=2
!
      SIF11=0
      SIF22=0
      SIF33=0
      SIF12=0
      SIF23=0
      SIF31=0
      DO ISP=1,ISPIN
      DO I=1,GRIDC%RL%NP
        SIF11=SIF11+DWORK1(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)
        SIF22=SIF22+DWORK2(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF33=SIF33+DWORK3(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF12=SIF12+DWORK1(I,ISP)*DWORK2(I,ISP)*DWORK(I,ISP)
        SIF23=SIF23+DWORK2(I,ISP)*DWORK3(I,ISP)*DWORK(I,ISP)
        SIF31=SIF31+DWORK3(I,ISP)*DWORK1(I,ISP)*DWORK(I,ISP)

        SIF11=SIF11+DWORK1(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
        SIF22=SIF22+DWORK2(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF33=SIF33+DWORK3(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF12=SIF12+DWORK1(I,ISP)*(DWORK2(I,1)+DWORK2(I,2))*DVC(I)
        SIF23=SIF23+DWORK2(I,ISP)*(DWORK3(I,1)+DWORK3(I,2))*DVC(I)
        SIF31=SIF31+DWORK3(I,ISP)*(DWORK1(I,1)+DWORK1(I,2))*DVC(I)
      ENDDO
      ENDDO
      SIF11=SIF11*RINPL*LATT_CUR%OMEGA
      SIF22=SIF22*RINPL*LATT_CUR%OMEGA
      SIF33=SIF33*RINPL*LATT_CUR%OMEGA
      SIF12=SIF12*RINPL*LATT_CUR%OMEGA
      SIF23=SIF23*RINPL*LATT_CUR%OMEGA
      SIF31=SIF31*RINPL*LATT_CUR%OMEGA

!=======================================================================
! calculate 
!              d    f_xc     grad rho
!        div  (------------  --------  ) 
!              d |grad rho| |grad rho|
!
! in reciprocal space
!=======================================================================

      DO I=1,GRIDC%RL%NP
         ANAB1U= DWORK1(I,1)
         ANAB2U= DWORK2(I,1)
         ANAB3U= DWORK3(I,1)
         ANAB1D= DWORK1(I,2)
         ANAB2D= DWORK2(I,2)
         ANAB3D= DWORK3(I,2)

         DWORK1(I,1) = ANAB1U* DWORK(I,1) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,1) = ANAB2U* DWORK(I,1) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,1) = ANAB3U* DWORK(I,1) + (ANAB3U+ANAB3D) * DVC(I)

         DWORK1(I,2) = ANAB1D* DWORK(I,2) + (ANAB1U+ANAB1D) * DVC(I)
         DWORK2(I,2) = ANAB2D* DWORK(I,2) + (ANAB2U+ANAB2D) * DVC(I)
         DWORK3(I,2) = ANAB3D* DWORK(I,2) + (ANAB3U+ANAB3D) * DVC(I)
      ENDDO

      spin2: DO ISP=1,ISPIN
! x-component:
!DIR$ IVDEP
!$DIR FORCE_VECTOR
!OCL NOVREC
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK1(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GX=(GRIDC%LPCTX(N1)*LATT_CUR%B(1,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(1,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(1,3))
         CWGRAD(I,ISP)=CWORK(I,ISP)*GX*CITPI
      ENDDO

! y-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK2(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GY=(GRIDC%LPCTX(N1)*LATT_CUR%B(2,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(2,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(2,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GY*CITPI
      ENDDO

! z-component:
      DO I=1,GRIDC%RL%NP
        DWORK(I,ISP) = DWORK3(I,ISP)
      ENDDO
      CALL OPSYNC(DWORK(1,ISP),CWORK(1,ISP),GRIDC%NPLWV)
      CALL FFT3D(CWORK(1,ISP),GRIDC,-1)
      CALL TRUNC_HIGH_FREQU(LATT_CUR, GRIDC, CWORK(1,ISP))

      DO I=1,GRIDC%RC%NP
         N1= MOD((I-1),GRIDC%RC%NROW) +1
         NC= (I-1)/GRIDC%RC%NROW+1
         N2= GRIDC%RC%I2(NC)
         N3= GRIDC%RC%I3(NC)

         GZ=(GRIDC%LPCTX(N1)*LATT_CUR%B(3,1)+GRIDC%LPCTY(N2)*LATT_CUR%B(3,2)+GRIDC%LPCTZ(N3)*LATT_CUR%B(3,3))
         CWGRAD(I,ISP)=CWGRAD(I,ISP)+CWORK(I,ISP)*GZ*CITPI
      ENDDO

      CALL SETUNB(CWGRAD(1,ISP),GRIDC)
      CALL FFT3D(CWGRAD(1,ISP),GRIDC,1)
      CALL OPSYNC(CWGRAD(1,ISP),DWGRAD(1,ISP),GRIDC%NPLWV)
!
      ENDDO spin2


!
!=======================================================================
! Now prepare the rest:
! (store rho in DWORK3 and quantity of above in DWORK1)
!=======================================================================
      XCENC=0._q
      CVZERO=0._q
      XCENCC=0._q

      DO I=1,GRIDC%RL%NP
         RHO1= MAX(REAL((DHTOT(I,1)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)
         RHO2= MAX(REAL((DHTOT(I,2)+DENCOR(I)/2)/LATT_CUR%OMEGA ,KIND=q), CHGMIN)

         VXC1=DWORKG(I,1)- REAL( DWGRAD(I,1) ,KIND=q) *RINPL
         VXC2=DWORKG(I,2)- REAL( DWGRAD(I,2) ,KIND=q) *RINPL
         DWORK(I,1)=VXC1
         DWORK(I,2)=VXC2
         VXC = 0.5_q*(VXC1+VXC2)
         CVZERO=CVZERO+VXC
         XCENCC=XCENCC-VXC1*RHO1*LATT_CUR%OMEGA-VXC2*RHO2*LATT_CUR%OMEGA
         XCENC=XCENC  -VXC1* REAL( DHTOT(I,1) ,KIND=q) -VXC2* REAL( DHTOT(I,2) ,KIND=q)
      ENDDO

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
      END SUBROUTINE
