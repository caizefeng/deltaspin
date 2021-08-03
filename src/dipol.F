#include "symbol.inc"
!************************ SUBROUTINE DIPCOR  ****************************
! RCS:  $Id: dipol.F,v 1.5 2003/06/27 13:22:15 kresse Exp kresse $
!
!   this subroutine calculates the total mono/dipol and quadrupol
!   moment in the cell and the required corrections to the total energy,
!   and possibly corrections to the forces and electrostatic field.
!   If potential corrections are switched on VASP can run for uncharged
!   systems "virtually" WITHOUT periodic boundary conditions
!
!   the strategy is straightforward, after calculating the monents
!   a Ewald-routine is called to calculate the energy of the moment
!   distribution using the periodic boundary conditions,
!   then the energy of the moment distribution without (or restricted)
!   periodic boundary conditions is calculated;
!   the energy corrections are given by
!     E(non-periodic) -E(periodic)
!
!   The introduction of a compensating electrostatic field is controlled
!   by the flag LCOR_DIP
!
!   LMONO   =T only monopole correction (for charged cells)
!   LCOR_DIP=T an external compensating field is set up;
!              in this mode CDIPOL should be called on each step
!              this is in the spirit of
!     Neugebauer und  Scheffler, Phys. Rev. B 46, 16967 (1992)
!     (mind that the energy corrections given in that paper are
!      wrong by a factor of 2)
!
!   LCOR_DIP=F no corrections to potential are calculated
!              and only energy corrections are calculated
!
!   IDIPCO controls which components of the dipol are calculated
!          and corrected
!
!    1-3       dipol in x,y or z   (should be used for slab calculations)
!    4         all directions      (should be used for isolated molecules)
!
!  I plan to extend this routine by
!  1) first fitting point charges to the current charge distribution
!  2) then correcting the energy using the equation
!        E(point charges, non-periodic) - E(point charges, periodic)
!  this would work equally well for charged and neutral systems
!
!***********************************************************************

  MODULE mdipol
    USE prec
    IMPLICIT NONE
    TYPE dipol
!only DIP
      INTEGER IDIPCO                 ! direction (0 no dipol corrections)
        LOGICAL :: LCOR_DIP          ! correct potential
        LOGICAL :: LMONO             ! monopole corrections only
        REAL(q) :: EPSILON           ! bulk charged cells: dielectric constant
        REAL(q) :: POSCEN(3)         ! position of center
        REAL(q) :: DIPOLC(3)         ! calculated dipol
        REAL(q) :: QUAD              ! trace of quadrupole
        INTEGER :: INDMIN(3)         ! position of minimum
        REAL(q) :: ECORR,EMONO,EDIPOL,EQUAD
        REAL(q) :: E_ION_EXTERN      ! energy between ions and added field
        REAL(q),POINTER :: FORCE(:,:)
        REAL(q) :: VACUUM(2)         ! vacuum level
        REAL(q) :: EFIELD
    END TYPE

    TYPE (dipol)    DIP
    LOGICAL,SAVE :: LDIPOL_RESET=.TRUE.

  CONTAINS

!*************************** FIELD_READER ****************************
!                                           
! this subroutine reads the INCAR file to determine whether the user
! wants to apply an external electric field
!
! the first version of the an external electric field has been implemented
! by Peter Feibelman
! an external electric field is only applied when LDIPOL=.TRUE.
! and when IDIPOL=1,...,3

! the present version was written by Georg Kresse
!
!**********************************************************************
  
    SUBROUTINE FIELD_READER(T_INFO, P, LATT_CUR, NELECT, IU0,IU5,IU6)
      USE base
      USE pseudo
      USE lattice
      USE poscar
      USE vaspxml
      IMPLICIT NONE 
      INTEGER IU5   ! input device (usually INCAR)
      INTEGER IU0   ! stderr
      INTEGER IU6   ! stdout
   ! local
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt) :: LATT_CUR
      REAL(q) NELECT
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER(1) :: CHARAC
      REAL(q) :: ZION

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')
!
! direction of dipole correction IDIPOL=0 (off) or 1-3 (1.d slab) or 4 (all direction)
!
      DIP%IDIPCO=0
      DIP%LCOR_DIP=.FALSE.
      DIP%POSCEN(1)=-100
      DIP%POSCEN(2)=-100
      DIP%POSCEN(3)=-100
      DIP%EPSILON=1
      DIP%EFIELD=0

      CALL RDATAB(LOPEN,INCAR,IU5,'LDIPOL','=','#',';','L', &
     &            IDUM,RDUM,CDUM,DIP%LCOR_DIP,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LDIPOL'' from file INCAR.'
         DIP%LCOR_DIP=.FALSE.
         RETURN
      ENDIF
      CALL XML_INCAR('LDIPOL','L',IDUM,RDUM,CDUM,DIP%LCOR_DIP,CHARAC,N)
      
      IF (DIP%LCOR_DIP) DIP%IDIPCO=4

      DIP%LMONO=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LMONO','=','#',';','L', &
     &            IDUM,RDUM,CDUM,DIP%LMONO,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''LMONO'' from file INCAR.'
         DIP%LMONO=.FALSE.
         RETURN
      ENDIF
      CALL XML_INCAR('LMONO','L',IDUM,RDUM,CDUM,DIP%LMONO,CHARAC,N)

      !
      ! only monopole correction: it makes no sense to perform those
      ! selfconsistently, since the potential is only shifted by a constant
      ! which can be equally well accounted for by a total energy shift
      !
      IF (DIP%LMONO) THEN
         DIP%IDIPCO=4
         DIP%LCOR_DIP=.FALSE.
      ENDIF

      CALL RDATAB(LOPEN,INCAR,IU5,'IDIPOL','=','#',';','I', &
     &            DIP%IDIPCO,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''IDIPOL'' from file INCAR.'
         DIP%IDIPCO=0
         RETURN
      ENDIF
      CALL XML_INCAR('IDIPOL','I',DIP%IDIPCO,RDUM,CDUM,LDUM,CHARAC,N)

      CALL RDATAB(LOPEN,INCAR,IU5,'EPSILON','=','#',';','F', &
     &            IDUM,DIP%EPSILON,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=1))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''EPSILON'' from file INCAR.'
         DIP%EPSILON=1
      ENDIF
      CALL XML_INCAR('EPSILON','F',IDUM,DIP%EPSILON,CDUM,LDUM,CHARAC,N)

      IF (DIP%EPSILON/=1 .AND. DIP%LCOR_DIP) THEN
         CALL VTUTOR('W','DIPOL EPSILON',DIP%EPSILON,1, &
              0,1,(0.0_q,0.0_q),1,.FALSE.,1,IU6,2)
         CALL VTUTOR('W','DIPOL EPSILON',DIP%EPSILON,1, &
              0,1,(0.0_q,0.0_q),1,.FALSE.,1,IU0,2)
      ENDIF

!
! position
!
      CALL RDATAB(LOPEN,INCAR,IU5,'DIPOL','=','#',';','F', &
     &            IDUM,DIP%POSCEN,CDUM,LDUM,CHARAC,N,3,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N/=3))) THEN
         IF (IU0>=0) &
         WRITE(IU0,*)'Error reading item ''DIPOL'' from file INCAR.'
         DIP%POSCEN=0
      ENDIF
      CALL XML_INCAR_V('DIPOL','F',IDUM,DIP%POSCEN,CDUM,LDUM,CHARAC,N)

      IF (DIP%LCOR_DIP .AND. DIP%IDIPCO>0 .AND. DIP%IDIPCO<=3) THEN

         CALL RDATAB(LOPEN,INCAR,IU5,'EFIELD','=','#',';','F', &
              &   IDUM,DIP%EFIELD,CDUM,LDUM,CHARAC,N,1,IERR)

         IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            IF (IU0>=0) THEN
               WRITE(IU0,*)'Error reading item ''EFIELD'' from file INCAR.'
               WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
            ENDIF
         ENDIF

         CALL XML_INCAR('DIP%EFIELD','F',IDUM,DIP%EFIELD,CDUM,LDUM,CHARAC,N)
      ENDIF

      CLOSE(IU5)

      ZION=SUM(T_INFO%NITYP*P%ZVALF*T_INFO%VCA)

      IF (DIP%LCOR_DIP .AND. ABS(NELECT-ZION)>1E-10) THEN
         IF (LATT_CUR%ANORM(1) /=  LATT_CUR%ANORM(2) .OR. LATT_CUR%ANORM(1) /= LATT_CUR%ANORM(3) .OR. &
              ABS(LATT_CUR%ANORM(1)*LATT_CUR%ANORM(2)*LATT_CUR%ANORM(3)-LATT_CUR%OMEGA)>1E-6) THEN
            CALL VTUTOR('S','DIPOL CUBIC',0.0_q,1,0,1,(0.0_q,0.0_q),1,.TRUE.,1, &
                 IU0,3)
         ENDIF
      ENDIF

      
    END SUBROUTINE FIELD_READER


!***************************** WRITE_EFIELD ***************************
!
!   this subroutine writes the electric field parameters to the 
!   OUTCAR file
!
!**********************************************************************

    SUBROUTINE WRITE_EFIELD(IU6)

      IMPLICIT NONE
      INTEGER IU6               ! output unit

      IF (IU6>=0) THEN
         WRITE(IU6,7225) DIP%LMONO,DIP%LCOR_DIP,DIP%IDIPCO,DIP%EPSILON
         IF (DIP%EFIELD/=0) THEN
            WRITE(IU6,100) DIP%EFIELD
         ENDIF
      ENDIF

 7225 FORMAT( &
             ' Dipole corrections'/  &
             '   LMONO  = ',L6,  '    monopole corrections only (constant potential shift)' / &
             '   LDIPOL = ',L6,  '    correct potential (dipole corrections)' / &
             '   IDIPOL = ',I6,  '    1-x, 2-y, 3-z, 4-all directions ' / & 
             '   EPSILON= ',F10.7,' bulk dielectric constant' /)
      
  100 FORMAT('   EFIELD = ',F10.7,' eV/A    external electric field' /)
       
    END SUBROUTINE WRITE_EFIELD

!************************** XML_WRITE_EFIELD **************************
!
!   this subroutine writes the electric field parameters to the XML file
!   if required
!
!**********************************************************************

    SUBROUTINE XML_WRITE_EFIELD
      USE pseudo
      USE vaspxml
      IMPLICIT NONE

      LOGICAL :: LDUM
      INTEGER :: IDUM
      REAL(q) :: RDUM
      COMPLEX(q)  :: CDUM
      CHARACTER(1) :: CHARAC

      CALL XML_TAG("separator","electronic dipolcorrection")
      CALL XML_INCAR('LDIPOL','L',IDUM,RDUM,CDUM,DIP%LCOR_DIP,CHARAC,1)
      CALL XML_INCAR('LMONO','L',IDUM,RDUM,CDUM,DIP%LMONO,CHARAC,1)
      CALL XML_INCAR('IDIPOL','I',DIP%IDIPCO,RDUM,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('EPSILON','F',IDUM,DIP%EPSILON,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('DIPOL','F',IDUM,DIP%POSCEN,CDUM,LDUM,CHARAC,3)
      CALL XML_INCAR('EFIELD','F',IDUM,DIP%EFIELD,CDUM,LDUM,CHARAC,1)
      CALL XML_CLOSE_TAG
    END SUBROUTINE XML_WRITE_EFIELD

!************************ SUBROUTINE  WRITE_DIP *************************
!
!  write dipole information
!
!************************************************************************

      SUBROUTINE WRITE_DIP(IU6)

      USE prec
      USE base
      USE vaspxml
      IMPLICIT NONE

      INTEGER IU6
! local varibales
      INTEGER IDIPCO,IMAX,IMIN,IDIP

      IDIPCO=DIP%IDIPCO

      IMAX=IDIPCO
      IMIN=IDIPCO
      IF (IDIPCO==4) THEN
         IMIN=1
         IMAX=3
      ENDIF

      IF (IU6 >=0 ) THEN
!------------------------------------------------------------------------
      WRITE(IU6,520)
 520  FORMAT(/ ' DIPCOR: dipole corrections for dipol')

      CALL XML_TAG("dipole")

 dir: DO IDIP=IMIN,IMAX
         WRITE(IU6,500,ADVANCE='No') IDIP,DIP%INDMIN(IDIP)
 500     FORMAT( ' direction ',I2,' min pos  ',I4,',')
      ENDDO dir

      WRITE(IU6,"(/,' dipolmoment     ',3F14.6,' electrons x Angstroem')") DIP%DIPOLC

      CALL XML_VEC_REAL(DIP%DIPOLC,"dipole")
      CALL XML_TAG_REAL("Tr[quadrupol]", DIP%QUAD)
      CALL XML_TAG_REAL("Echarged", DIP%EMONO)
      CALL XML_TAG_REAL("Edipol_quadrupol", DIP%EMONO)
      CALL XML_TAG_REAL("Eion", DIP%EMONO)

      WRITE(IU6,530) DIP%QUAD, DIP%EMONO,DIP%ECORR-DIP%EMONO,DIP%E_ION_EXTERN
 530  FORMAT(' Tr[quadrupol] ',F16.6,// &
             ' energy correction for charged system ',F16.6,' eV' / &
             ' dipol+quadrupol energy correction    ',F16.6,' eV' / &
             ' added-field ion interaction  ',F16.6,' eV  (added to PSCEN)'/)
!------------------------------------------------------------------------
      CALL XML_CLOSE_TAG("dipole")

      END IF
      END SUBROUTINE WRITE_DIP

!************************ SUBROUTINE  WRITE_VACUUM_LEVEL ****************
!
!  write vacuum level to the OUTCAR file
!
!************************************************************************

      SUBROUTINE WRITE_VACUUM_LEVEL(IU6)

      USE prec
      USE base
      IMPLICIT NONE

      INTEGER IU6

      IF (IU6 >=0 .AND. DIP%IDIPCO<4 ) THEN
         WRITE(IU6,510) DIP%VACUUM
 510  FORMAT(' vacuum level on the upper side and lower side of the slab',2F14.3)
      ENDIF
      END SUBROUTINE WRITE_VACUUM_LEVEL

!************************ SUBROUTINE  DIPOL_RESET ***********************
!
!  this subroutine resets the mixer in the dipol routine CDIPOL
!
!
!************************************************************************

      SUBROUTINE DIPOL_RESET()
      USE prec
      LDIPOL_RESET=.TRUE.

      END SUBROUTINE

!************************ SUBROUTINE POINT_CHARGE_DIPOL ****************
!
! this subroutine calculates the dipolmoment and the
! trace of the quadrupol moment of the point charges
! with respect to the center POSCEN
! quadrupol is only fast hack (correct only for orthogonal axes)
!
!***********************************************************************

    SUBROUTINE POINT_CHARGE_DIPOL(T_INFO,LATT_CUR,ZVAL,POSCEN, &
         DIP_DIRECT,QUAD)

      USE prec
      USE lattice
      USE poscar

      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      REAL(q) :: ZVAL(:)
      REAL(q) :: POSCEN(3)          ! center
      REAL(q) :: DIP_DIRECT(3)      ! dipol moment in direct coordinated
      REAL(q) :: QUAD               ! trace of quadrupol moment
      REAL(q), PARAMETER :: TINY=1E-2_q

      integer :: nis, nt, ni
      real(q) :: x, y, z

      DIP_DIRECT=0
      QUAD      =0

      NIS=1
      typ: DO NT=1,T_INFO%NTYP
         DO NI=NIS,NIS+T_INFO%NITYP(NT)-1
            X= MOD( T_INFO%POSION(1,NI)-POSCEN(1)+10.5_q,1._q)-0.5_q
            Y= MOD( T_INFO%POSION(2,NI)-POSCEN(2)+10.5_q,1._q)-0.5_q
            Z= MOD( T_INFO%POSION(3,NI)-POSCEN(3)+10.5_q,1._q)-0.5_q

            QUAD= QUAD-ZVAL(NT)*( (X*LATT_CUR%ANORM(1))**2+ &
                 (Y*LATT_CUR%ANORM(2))**2+(Z*LATT_CUR%ANORM(3))**2)

!jF: the atoms at the cell boundary (position -0.5) cause a problem since
!    one also had to add a contribution from the equivalent position at the
!    opposite cell boundary (position +0.5) effectively adding up to zero
!    (due to "position + -position") what should be fixed by following code.
!    Such problems only occur for bulk cells. For isolated molecules etc.
!    there are no atoms at the cell boundary, just vacuum :-). It is also no
!    problem for the quadrupole moment (due to "position**2"). If you still
!    fail to get the correct dipole moment play around with parameter TINY
!    which defines the "thickness of the cell boundary region" ...
            IF (ABS(ABS(X)-0.5_q)<TINY/LATT_CUR%ANORM(1)) X=0
            IF (ABS(ABS(Y)-0.5_q)<TINY/LATT_CUR%ANORM(2)) Y=0
            IF (ABS(ABS(Z)-0.5_q)<TINY/LATT_CUR%ANORM(3)) Z=0

            DIP_DIRECT(1)=DIP_DIRECT(1)-ZVAL(NT)*X
            DIP_DIRECT(2)=DIP_DIRECT(2)-ZVAL(NT)*Y
            DIP_DIRECT(3)=DIP_DIRECT(3)-ZVAL(NT)*Z

         ENDDO
      NIS=NIS+T_INFO%NITYP(NT)
      ENDDO typ


    END SUBROUTINE POINT_CHARGE_DIPOL
      
  END MODULE mdipol



  SUBROUTINE XML_WRITE_EFIELD_
    USE mdipol

    CALL  XML_WRITE_EFIELD

  END SUBROUTINE XML_WRITE_EFIELD_

  
!**********************************************************************
!
! CDIPOL can be called directly if the total charge density in 
! real space  is stored in CHTOT
! usually one must use the calling interface CDIPOL_CHTOT_REC
! to calculate the required quantity by a fourier transformation
!
!**********************************************************************


      SUBROUTINE CDIPOL_CHTOT_REC(GRIDC, LATT_CUR,P,T_INFO,  &
              CHTOT,CSTRF,CVTOT, NCDIJ, NELECT )

      USE prec
      USE base
      USE lattice
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      USE constant
      USE mdipol
      IMPLICIT NONE
      
      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      REAL(q) NELECT
      INTEGER NCDIJ
      RGRID  CHTOT(DIMREAL(GRIDC%MPLWV), NCDIJ)
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      RGRID :: CVTOT(DIMREAL(GRIDC%MPLWV),NCDIJ)
! local
      COMPLEX(q), ALLOCATABLE::  CWORK(:,:)
      INTEGER ISP

      ALLOCATE(CWORK(GRIDC%MPLWV,NCDIJ))
      
      DO ISP=1,NCDIJ
         CALL RC_ADD(CHTOT(1,ISP),1.0_q,CHTOT(1,ISP),0.0_q,CWORK(1,ISP),GRIDC)
         CALL FFT3D(CWORK(1,ISP),GRIDC,1)
      ENDDO
      IF (NCDIJ >1) CALL MAG_DENSITY(CWORK, CWORK, GRIDC, NCDIJ)

      CALL CDIPOL(GRIDC, LATT_CUR, P, T_INFO,  &
              CWORK, CSTRF, CVTOT, NCDIJ, NELECT )

      DEALLOCATE(CWORK)

      END SUBROUTINE



      SUBROUTINE CDIPOL(GRIDC, LATT_CUR,P,T_INFO, &
              CHTOT,CSTRF,CVTOT, NCDIJ, NELECT )

      USE prec
      USE base
      USE lattice
      USE mpimy
      USE mgrid
      USE poscar
      USE pseudo
      USE constant
      USE mdipol
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      TYPE (grid_3d)     GRIDC
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      REAL(q) NELECT
      REAL(q),SAVE :: DIP_OLD(3),QUAD_OLD

      RGRID  CHTOT(DIMREAL(GRIDC%MPLWV))
      COMPLEX(q) CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      RGRID :: CVTOT(DIMREAL(GRIDC%MPLWV),NCDIJ)
! work array
      REAL(q),ALLOCATABLE :: DENLIN(:)
      INTEGER,PARAMETER :: NWIN=3
      INTEGER,SAVE :: LASTMIN=-1000000
      REAL(q) :: WINDOW(-NWIN:NWIN)=(/0.1,0.2,0.3,0.4,0.3,0.2,0.1/)
!
! a smooth cutoff function is used for potentials and charge densities
! around the dividing plane
! WIDTH must be at least 1
! 4 grid points around the step function are usually a good choice
      REAL(q), PARAMETER :: WIDTH=4
!
! in some cases mixing of the determined dipol moment is desirable
! (I personally think it is no good, if you have convergence
!  problems increase the size of the supercell)
!
      LOGICAL,SAVE :: LMIX=.FALSE.

      LOGICAL,SAVE :: LINIT=.TRUE.
      REAL(q) :: DIP_DIRECT(3),DIP_DIRECT_POINT(3)
      REAL(q) :: POSCEN(3)
      REAL(q) :: EF(3)
      REAL(q) :: CUTOFF

      IDIPCO=DIP%IDIPCO
!=======================================================================
!     fast exit / initialization
!=======================================================================
      IF ( IDIPCO<1 .OR. IDIPCO >4) RETURN

      IF (LINIT) THEN
         ALLOCATE(DIP%FORCE(3,T_INFO%NIONS))
         LINIT=.FALSE.
      ENDIF


      DIP%INDMIN=0
      DIP%DIPOLC=0
      DIP%ECORR =0
      DIP%E_ION_EXTERN=0
      DIP%QUAD  =0
      DIP%FORCE =0
      DIP%VACUUM=0
      DIP_DIRECT=0
!=======================================================================
!     determine direction for which dipol should be calculated
!=======================================================================
      EDIPOL=0

      IMAX=IDIPCO
      IMIN=IDIPCO
      IF (IDIPCO==4) THEN
         IMIN=1
         IMAX=3
      ENDIF

      ALLOCATE(DENLIN(MAX(GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ)) )

!=======================================================================
! set up required quantities
!=======================================================================
      ZION=SUM(T_INFO%NITYP*P%ZVALF*T_INFO%VCA)
!=======================================================================
!    calculate the dipol moment
!=======================================================================
      POSCEN=0
   dir1: DO IDIP=IMIN,IMAX
      NOUT=GRIDC%NGPTAR(IDIP)
      NOUTH=NOUT/2

   ! INDMIN is the position to which integration is performed
      IF (DIP%POSCEN(IDIP) /= -100) THEN

   ! if POSCEN is set use this quantity to determine INDMIN
         IPOS=DIP%POSCEN(IDIP)*NOUT
         INDMIN=MOD(NOUTH+IPOS+10*NOUT,NOUT)+1
         POSCEN(IDIP)=MOD(REAL(INDMIN+NOUTH-1,KIND=q)/NOUT+10._q,1._q)
      ELSE

   ! of POSCEN is not set  determine INDMIN
   ! as the positions where density has a minimum

         CALL AVERAG(GRIDC,IDIP,CHTOT,DENLIN,.TRUE.)

         DENMIN=1E9

         DO IND=1,NOUT
            ! use a glinding average to get rid of wigles in charge density
            DEN=0
            DO II=-NWIN,NWIN
               II_IND=MOD(IND+II-1+NOUT,NOUT)+1
               DEN=DEN+DENLIN(II_IND)*WINDOW(II)
            ENDDO
            IF (DEN<DENMIN) THEN
               DENMIN=DEN
               INDMINDEN=IND
            ENDIF
         ENDDO
!furth   new additional strategy: determine the "center of mass" of charges :-)
         F=1._q/NOUT
         DEN=0
         DO IND=1,NOUT
            II=MOD(IND-INDMINDEN+NOUT,NOUT)-NOUTH
            IF (II==-NOUTH) II=0
            DEN=DEN+DENLIN(IND)*II*F
         ENDDO
         POSCEN(IDIP)=MOD(REAL(INDMINDEN+NOUTH-1,KIND=q)/NOUT+10._q,1._q)
         CALL POINT_CHARGE_DIPOL(T_INFO,LATT_CUR,P%ZVALF,POSCEN, &
                                 DIP_DIRECT_POINT,QUAD_POINT)
         INDMIN=INDMINDEN+NINT((DEN-DIP_DIRECT_POINT(IDIP))/(NELECT+ZION))
         IF (LASTMIN /= -1000000) INDMIN=NINT(0.5_q*REAL(INDMIN+LASTMIN,KIND=q))
         LASTMIN=INDMIN
         POSCEN(IDIP)=MOD(REAL(INDMIN+NOUTH-1,KIND=q)/NOUT+10._q,1._q)
      ENDIF
      DIP%INDMIN(IDIP)=INDMIN
!=======================================================================
!    calculate dipol moment
!  \int (r) \rho(r)  d r
!         for r from INDMIN to INDMIN+NOUT
!=======================================================================
      !   average total density over a plane
      !   perpendicular to direction IDIP

      CALL AVERAG(GRIDC,IDIP,CHTOT,DENLIN,.FALSE.)
      DENLIN(1:NOUT)=DENLIN(1:NOUT)/NOUT

      SUM_=0
      SUM2=0

      F=1._q/NOUT

      DO IND=1,NOUT
         II=MOD(IND-INDMIN+NOUT,NOUT)-NOUTH
         XX=ABS(ABS(II)-NOUTH)
         CUTOFF=1.0
         IF (XX > WIDTH)  THEN
            CUTOFF = 1.0
         ELSE
            CUTOFF= ABS(SIN(PI*XX/WIDTH/2))
         ENDIF
         SUM_=SUM_+DENLIN(IND)*II*F*CUTOFF
         SUM2=SUM2+DENLIN(IND)*(II*F)*(II*F)*CUTOFF*CUTOFF
      ENDDO
      DIPOLC=SUM_
      SUM2=SUM2*LATT_CUR%ANORM(IDIP)**2
      DIP%QUAD=DIP%QUAD+SUM2

      DIP_DIRECT(IDIP)=DIPOLC
   ENDDO dir1

! get dipol moment of point charges
      CALL POINT_CHARGE_DIPOL(T_INFO,LATT_CUR,P%ZVALF,POSCEN, &
         DIP_DIRECT_POINT,QUAD_POINT)
! each ion has a quadrupol moment because the shape of the
! charge distribution deviates from Z/r
! Q = -6 / 4pi e^2 PSCORE
!   = -6 / 4pi e^2 lim (q \to 0) 4 Pi \int_0^Inf (V(r) + e^2 Z/r ) r^2 dr
      DO NT=1,T_INFO%NTYP
         QUAD_POINT=QUAD_POINT-P(NT)%PSCORE*T_INFO%NITYP(NT)*3/FELECT/TPI
      ENDDO
!      write(*,'(''direct: dir, point'',6f9.4)') DIP_DIRECT, DIP_DIRECT_POINT

      DO  IDIP=IMIN,IMAX
        DIP_DIRECT(IDIP)=DIP_DIRECT(IDIP)+DIP_DIRECT_POINT(IDIP)
      ENDDO

! DIP_DIRECT contains now the dipol moment in direct (fractional) coordinates
! mix it with last dipol and convert to cartesian coordinates
      DIP%QUAD=DIP%QUAD+QUAD_POINT
      

      IF (LDIPOL_RESET) THEN
        LDIPOL_RESET = .FALSE.
      ELSE IF (LMIX) THEN
        DIP_DIRECT = 0.25_q * (DIP_DIRECT + 3*DIP_OLD)
        DIP%QUAD   = 0.25_q * (DIP%QUAD  + 3*QUAD_OLD)
      ENDIF

      DIP_OLD  = DIP_DIRECT
      QUAD_OLD = DIP%QUAD

      DIP%DIPOLC=DIP_DIRECT
      CALL DIRKAR(1,DIP%DIPOLC,LATT_CUR%A)
!      write(*,'(''direct: dir, point'',6f9.4)') DIP%DIPOLC
!=======================================================================
!  charged systems
!  quadrupol corrections are presently only correct for cubic boxes
!=======================================================================
      IF (ABS(NELECT-ZION)>1E-10) THEN

         CALL EWALD_MONO(EMONO , NELECT-ZION, LATT_CUR )
         EQUAD=-DIP%QUAD*(NELECT-ZION)*TPI/(LATT_CUR%OMEGA)*FELECT/3
         QUAD_FIELD=    -(NELECT-ZION)*TPI/(LATT_CUR%OMEGA)*FELECT/3
      ELSE
         EMONO=0
         EQUAD=0
         QUAD_FIELD=0
      ENDIF
!=======================================================================
! calculate the corrections to the dipol energy and the electrostatic
! field
!=======================================================================
      CALL EWALD_DIPOL(EDIPOL, EF, DIP%DIPOLC, LATT_CUR, IDIPCO)
!      write(*,'(''field: '',6f9.4)') EDIPOL,EF
 correct: IF (.NOT. DIP%LCOR_DIP) THEN
      E_ION_EXTERN=0       ! no external field
!=======================================================================
!  correct the potential using planar dipols placed at x0,y0 and z0
!  the potential should  "counterbalance" the electrostatic field
!  created by the repeated images and is equal to EF(:),
!  another way of viewing it is that we simply solve the Poison
!  equation without periodic boundary conditions
!=======================================================================
  ELSE correct
     E_ION_EXTERN=0
!  quadrupol energy will be contained in E_ION_EXTERN and bandstructure
!  energy if CVTOT is corrected
     EQUAD=0

  dir2: DO IDIP=IMIN,IMAX
     !    check whether a(IDIP) is ortogonal to the other 2 axis
     !    otherwise we are lost, and forget about the dipol corrections
     !    to the potential
      DO J=0,1
         IDIR=MOD(IDIP+J,3)+1
         SUM_=0
         DO JJ=1,3
            SUM_=SUM_+LATT_CUR%A(JJ,IDIP)*LATT_CUR%A(JJ,IDIR)
         ENDDO
         IF (SUM_>1E-4_q) THEN
            CYCLE dir2
         ENDIF
      ENDDO

      NOUT=GRIDC%NGPTAR(IDIP)
      NOUTH=NOUT/2

!  set correction field E_COMPENSATE
      E_COMPENSATE=EF(IDIP)*LATT_CUR%ANORM(IDIP)
!  test instructions, for orthorhombic cells this should be identical
!  to the previous line
!      E_COMPENSATE=-2*TPI*DIP%DIPOLC(IDIP)/(LATT_CUR%OMEGA)*FELECT
!      IF (IDIPCO==4) E_COMPENSATE=E_COMPENSATE/3


!  add energy of dipol in electrostatic external field to dipol energy
!  correction
!  EDIPOL is now a double counting correction, since the interaction between
!  the repeated dipoles is accounted for twice if the field is corrected.
!  additional energy contribution are calculated as
!  \int delta V(r) rho(r) dr 
!  \int delta V(r) rho_electron(r) dr   is contained in the band structure energy
!  \int delta V(r) rho_ion(r) dr        is E_ION_EXTERN and added to PSCENC
!  as usual this counts the electron-electron contribution twice
      EDIPOL=EDIPOL+DIP_DIRECT(IDIP)*E_COMPENSATE*LATT_CUR%ANORM(IDIP)

      INDMIN=DIP%INDMIN(IDIP)

! add electrostatic field to local potential
!  phi =  -E [x - a Theta (x0-x) ]
      DIPFAC = -(E_COMPENSATE+DIP%EFIELD)* LATT_CUR%ANORM(IDIP)/NOUT
      QUADFAC=  QUAD_FIELD*(LATT_CUR%ANORM(IDIP)/NOUT)**2
      NG=0
! NFAST tells which index is stored in a row
      IDIR=IDIP
      IF (GRIDC%RL%NFAST==3) THEN
         IDIR=MOD(IDIP,3)+1 ! mpi version: x-> N2, y-> N3, z-> N1
      ENDIF

      DO NC=1,GRIDC%RL%NCOL
         N2= GRIDC%RL%I2(NC)
         N3= GRIDC%RL%I3(NC)
         DO N1=1,GRIDC%RL%NROW
            NG=NG+1
            IF (IDIR==1) II=MOD(N1-INDMIN+NOUT,NOUT)-NOUTH
            IF (IDIR==2) II=MOD(N2-INDMIN+NOUT,NOUT)-NOUTH
            IF (IDIR==3) II=MOD(N3-INDMIN+NOUT,NOUT)-NOUTH

            XX=ABS(ABS(II)-NOUTH)
            IF (XX > WIDTH)  THEN
               CUTOFF = 1.0
            ELSE
               CUTOFF= ABS(SIN(PI*XX/WIDTH/2))
            ENDIF
            POTCOR=DIPFAC*II*CUTOFF+QUADFAC*II*II*CUTOFF*CUTOFF
            CVTOT(NG,1)=CVTOT(NG,1)+POTCOR
         ENDDO
      ENDDO
!-----------------------------------------------------------------------
!    calculate dipole correction to forces due to "correction" field
!     dF =  E  * -Z_ion  (ions have negative charge in VASP)
!    and correction to energy of ions due to "correction" potential
!     dE = phi * -Z_ion
!   (this must be done here, because the corrected potential is not used 
!    in the force calculation and the calculation of the Ewald energy)
!-----------------------------------------------------------------------
      NIS=1
      DO NT =1,T_INFO%NTYP
      DO ION=NIS,T_INFO%NITYP(NT)+NIS-1
         POS=MOD(T_INFO%POSION(IDIP,ION)-POSCEN(IDIP)+10.5_q,1._q)-0.5_q
         DIS=POS*LATT_CUR%ANORM(IDIP)
         E_ION_EXTERN=E_ION_EXTERN-P(NT)%ZVALF* &
             (-(E_COMPENSATE+DIP%EFIELD)*DIS+QUAD_FIELD*DIS*DIS)

         DIP%FORCE(IDIP,ION)=DIP%FORCE(IDIP,ION)-P(NT)%ZVALF* &
             ( ((E_COMPENSATE+DIP%EFIELD-QUAD_FIELD*DIS*2)/LATT_CUR%ANORM(IDIP)))
      ENDDO
      NIS = NIS+T_INFO%NITYP(NT)
      ENDDO
! each ion has a quadrupol moment because the shape of the
! charge distribution deviates from Z/r, this gives an additional
! energy in the quadratic field (factor 3 because each direction is treated
! seperately)
! (this term is missing essentially in the Ewald summation)
      DO NT=1,T_INFO%NTYP
         E_ION_EXTERN=E_ION_EXTERN- &
         P(NT)%PSCORE*T_INFO%NITYP(NT)*3/FELECT/TPI*QUAD_FIELD/3
      ENDDO
!-----------------------------------------------------------------------
!    calculate workfunction (left and right handed)
!    only spin up is taken into account
!      Delta(Pot_vac,E_Fermi)
!-----------------------------------------------------------------------
  ENDDO dir2
! convert from direct to cartesian
      CALL DIRKAR(T_INFO%NIONS, DIP%FORCE, LATT_CUR%A)
  ENDIF correct

      IF (IDIPCO < 4) THEN
         CALL AVERAG(GRIDC,IDIPCO,CVTOT(1,1),DENLIN,.FALSE.)

         VACPOT=DENLIN(MOD(INDMIN-INT(WIDTH)+NOUT,NOUT))
         DIP%VACUUM(1)=VACPOT

         VACPOT=DENLIN(MOD(INDMIN+INT(WIDTH)+NOUT,NOUT))
         DIP%VACUUM(2)=VACPOT
      ENDIF

!=======================================================================
! store results for later write
! if only monopole corrections are required, set the other corrections
! to zero
!=======================================================================
      IF (DIP%LMONO) THEN
         EDIPOL=0
         EQUAD =0
      ENDIF
      DIP%EDIPOL      =EDIPOL/DIP%EPSILON
      DIP%EQUAD       =EQUAD /DIP%EPSILON
      DIP%EMONO       =EMONO /DIP%EPSILON
      DIP%E_ION_EXTERN=E_ION_EXTERN

      DIP%ECORR       =DIP%EDIPOL +DIP%EQUAD +DIP%EMONO

      DEALLOCATE(DENLIN)
      RETURN
      END



!************************ SUBROUTINE AVERAG  ****************************
!
!   calculate the average (DENLIN) of an array (RGRID DWORK1)
!   over a plane along a direction (defined by IDIP)
!
!***********************************************************************

      SUBROUTINE AVERAG(GRIDC,IDIP,DWORK1,DENLIN,LABS)
      USE prec
      USE mgrid

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC

      RGRID   DWORK1(DIMREAL(GRIDC%MPLWV))
      REAL(q) DENLIN(GRIDC%NGPTAR(IDIP))
      LOGICAL LABS
!=======================================================================
!   calculate density average over a plane
!   perpendicular to slabdirection (defined by IDIP)
!=======================================================================
      NOUT=GRIDC%NGPTAR(IDIP)

      DENLIN=0

! here I do not worry about vectorization troubles
! routine is complicated enough
! NFAST tells which index is stored in a row
      IDIR=IDIP
      IF (GRIDC%RL%NFAST==3) THEN
         IDIR=MOD(IDIP,3)+1 ! x-> N2, y-> N3, z-> N1
      ENDIF

      NG=0
 col: DO NC=1,GRIDC%RL%NCOL
      N2= GRIDC%RL%I2(NC)
      N3= GRIDC%RL%I3(NC)

 row: DO N1=1,GRIDC%RL%NROW
        NG=NG+1
        VALUE= DWORK1(NG)
        IF (LABS) VALUE=ABS(VALUE)
        IF (IDIR==1) DENLIN(N1)=DENLIN(N1)+ VALUE
        IF (IDIR==2) DENLIN(N2)=DENLIN(N2)+ VALUE
        IF (IDIR==3) DENLIN(N3)=DENLIN(N3)+ VALUE

      ENDDO row
      ENDDO col

      RINPL =1._q/GRIDC%NPLWV
      FACTOR=1._q/GRIDC%NPLWV*NOUT
      DENLIN=DENLIN*FACTOR

      CALLMPI( M_sum_d(GRIDC%COMM, DENLIN, NOUT))

      END SUBROUTINE AVERAG


!************************ SUBROUTINE EWALD_DIPOL ***********************
!
! calculate energy corrections (and electric field corrections)
! due to dipol:
! the energy correction is defined to be minus the energy of the dipol
! in the current cell plus the energy of the dipol in a cell with
! x,y and/or z -> infinity
! the routine has two modes:
! ) IDIPCO=4
!    we are interest in the limit x,y AND z -> infinity
!    so we calculate the energy of two charges having a dipol moment
!    DICOLC and subtract from that the energy obtained for the same
!    two charges placed into the super cell.
!    for a cubic cell the correction is known to be
!         DIP*DIP *TPI/(LATT_T%OMEGA)*FELECT/3
!      see G. Makov and M.C.Payne, Phys. Rev. B. 51, 4014 (1995)
! ) IDICO=1,2,3
!    we want to know the limit for x, y OR z -> infinity
!    we get the limit using a trick (see below)
!    for tetragonal cells the limit is (easy to derive see for instance
!        Scheffler and Neugebauer)
!         DIP*DIP *TPI/(LATT_CUR%OMEGA)*FELECT
!
!
!***********************************************************************

      SUBROUTINE EWALD_DIPOL(EDIPOL, EF, DIPOLC, LATT_CUR, IDIPCO)
      USE prec
      USE lattice
      USE ebs
      USE constant
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      INTEGER,INTENT(IN) :: IDIPCO    ! directions
      REAL(q),INTENT(OUT):: EDIPOL    ! energy of dipol
      REAL(q),INTENT(IN) :: DIPOLC(3) ! magnitude of dipol
      REAL(q),INTENT(OUT):: EF(3)     ! -electrostatic field created by repeated dipols
      TYPE (latt)  LATT_CUR           ! lattice
! local varibales
      TYPE (latt)  LATT_T
      REAL(q), PARAMETER :: DIS=1E-1
      INTEGER, PARAMETER :: NIONS=2,NTYP=2
      REAL(q) :: POS(3,NIONS),FORCE(3,NIONS),FORCE2(3,NIONS)
      REAL(q) :: SIF(3,3),ZVAL(NTYP),VCA(NTYP)
      REAL(q) :: ENERGY
      INTEGER :: ITYP(NIONS),NITYP(NTYP)

      DIP= SQRT(SUM(DIPOLC*DIPOLC))

   ! place two charges into direction in a distance DIS
      POS(:,1)=0
      POS(:,2)=DIS*DIPOLC/DIP
      CALL KARDIR(2,POS,LATT_CUR%B)
   ! set their charge to the required value
      ZVAL(1)=-DIP/DIS
      ZVAL(2)=+DIP/DIS
      VCA    = 1
   ! initialize varibales required by EWALD-summation
      ITYP =(/ 1,2 /)  ! type of each ion
      NITYP=1          ! number of ions per typ
   ! fire up EWALD to get energy of these charges using periodic boundary cond.
      CALL FEWALD( &
           POS,FORCE,LATT_CUR%A,LATT_CUR%B,LATT_CUR%ANORM,LATT_CUR%BNORM, &
           LATT_CUR%OMEGA,SIF,ENERGY,NTYP,ZVAL,VCA, & 
           NIONS,NIONS,ITYP,NITYP,-1)

      IF (IDIPCO == 4) THEN
   ! interested in limit of very large cell (x,y and z direction -> inf)
   ! calculate energy of two point charges minus ewald energy
         EDIPOL = FELECT*ZVAL(1)*ZVAL(2)/DIS-ENERGY
         EDIPOL_=DIP*DIP *TPI/(LATT_CUR%OMEGA)*FELECT/3
         FORCE(:,1)  =-FELECT*ZVAL(1)*ZVAL(2)*DIPOLC/DIP/DIS**2-FORCE(:,1)
         FORCE(:,2)  = FELECT*ZVAL(1)*ZVAL(2)*DIPOLC/DIP/DIS**2-FORCE(:,2)
         EF          = FORCE(:,1)/ZVAL(1)
         ! this is the value for the cubic super cell should be equal EDIPOL
      ELSE
   !
   ! interested in limit of very long cell in one direction
   ! that is really tricky
         ! make cell twice as long
         LATT_T=LATT_CUR
         LATT_T%A(:,IDIPCO)= LATT_CUR%A(:,IDIPCO)*2
         CALL LATTIC(LATT_T)
         ! place two charges into direction in a distance DIS
         POS(:,1)=0
         POS(:,2)=DIS*DIPOLC/DIP
         CALL KARDIR(2,POS,LATT_T%B)

         CALL FEWALD( &
           POS,FORCE2,LATT_T%A,LATT_T%B,LATT_T%ANORM,LATT_T%BNORM, &
           LATT_T%OMEGA,SIF,ENERGY2,NTYP,ZVAL,VCA, & 
           NIONS,NIONS,ITYP,NITYP,-1)
         ! error should fall off as 1/L, doubled cell gives half the error ;-)
         EDIPOL = 2*(ENERGY2-ENERGY)
         EF     = 2*(FORCE2(:,1) -FORCE(:,1))/ZVAL(1)
         ! this is the correction for the tetragonal supercell
         EDIPOL_=DIP*DIP *TPI/(LATT_CUR%OMEGA)*FELECT
      ENDIF
      CALL KARDIR(1,EF,LATT_CUR%B)
    END SUBROUTINE EWALD_DIPOL


!************************ SUBROUTINE EWALD_MONO  ***********************
!
! calculate energy corrections for net charged system
! as before (EWALD_DIPOL) it is the energy difference between the
! real system and the artificial system with periodic boundary conditions
! (i.e. an isolated charge =0 )
!
!***********************************************************************

      SUBROUTINE EWALD_MONO(ENERGY, Z, LATT_CUR )
      USE prec
      USE lattice
      USE ebs
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      REAL(q),INTENT(OUT):: ENERGY
      REAL(q),INTENT(IN) :: Z
      TYPE (latt) LATT_CUR
! local varibales
      INTEGER, PARAMETER :: NIONS=1,NTYP=1
      REAL(q) :: POS(3,NIONS),FORCE(3,NIONS),SIF(3,3),ZVAL(NTYP),VCA(NTYP)
      INTEGER :: ITYP(NIONS),NITYP(NTYP)

      POS(:,1)=0
   ! set charge
      ZVAL=Z
      VCA =1
   ! initialize varibales required by EWALD-summation
      ITYP =(/ 1 /)    ! type of each ion
      NITYP=1          ! number of ions per typ
   ! fire up EWALD to get energy of isolated charge using ewald summation
      CALL FEWALD( &
           POS,FORCE,LATT_CUR%A,LATT_CUR%B,LATT_CUR%ANORM,LATT_CUR%BNORM, &
           LATT_CUR%OMEGA,SIF,ENERGY,NTYP,ZVAL,VCA, & 
           NIONS,NIONS,ITYP,NITYP,-1)

      ENERGY=-ENERGY
    END SUBROUTINE EWALD_MONO


!************************ SUBROUTINE CHGION ****************************
!
! this subroutine calculates the chargedensity which corresponds
! to the pseudopotential formfactor
! i.e. core-charge
! (subroutine is no longer used)
!
!***********************************************************************

      SUBROUTINE CHGION(GRID,LATT_CUR,P,T_INFO,CSTRF,CVPS)
      USE prec
      USE lattice
      USE poscar
      USE pseudo
      USE mpimy
      USE mgrid
      USE charge
      USE constant
      IMPLICIT COMPLEX(q) (C)

      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRID
      TYPE (latt)        LATT_CUR
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)

      COMPLEX(q) CVPS(GRID%RC%NP)
      COMPLEX(q) CSTRF(GRID%MPLWV,T_INFO%NTYP)

      CVPS=0
!=======================================================================
! loop over all types of atoms
! multiply structur factor by local pseudopotential
!=======================================================================
      SCALE=TPI**2/EDEPS

      typ: DO NT=1,T_INFO%NTYP

      ARGSC=NPSPTS/P(NT)%PSGMAX
      PSGMA2=P(NT)%PSGMAX-P(NT)%PSGMAX/NPSPTS

      DO N=1,GRID%RC%NP

        N1= MOD((N-1),GRID%RC%NROW) +1
        NC= (N-1)/GRID%RC%NROW+1
        N2= GRID%RC%I2(NC)
        N3= GRID%RC%I3(NC)
!=======================================================================
! calculate the magnitude of the reciprocal lattice vector
!=======================================================================
        GX= GRID%LPCTX(N1)*LATT_CUR%B(1,1)+GRID%LPCTY(N2)*LATT_CUR%B(1,2)+GRID%LPCTZ(N3)*LATT_CUR%B(1,3)
        GY= GRID%LPCTX(N1)*LATT_CUR%B(2,1)+GRID%LPCTY(N2)*LATT_CUR%B(2,2)+GRID%LPCTZ(N3)*LATT_CUR%B(2,3)
        GZ= GRID%LPCTX(N1)*LATT_CUR%B(3,1)+GRID%LPCTY(N2)*LATT_CUR%B(3,2)+GRID%LPCTZ(N3)*LATT_CUR%B(3,3)

        GSQU=GX**2+GY**2+GZ**2
        G=SQRT(GSQU)*2*PI
        IF ( ( (GRID%LPCTX(N1)/=0) .OR. (GRID%LPCTY(N2)/=0) .OR. &
     &         (GRID%LPCTZ(N3)/=0) ) .AND. (G<PSGMA2) ) THEN
!=======================================================================
! convert the magnitude of the reciprocal lattice vector to a position
! in the pseudopotential arrays and interpolate the pseudopotential and
! its derivative
!=======================================================================
           I  =INT(G*ARGSC)+1
           REM=G-P(NT)%PSP(I,1)
           VPST   =(P(NT)%PSP(I,2)+REM*(P(NT)%PSP(I,3)+ &
                REM*(P(NT)%PSP(I,4)  +REM*P(NT)%PSP(I,5))))
           CVPS(N)=CVPS(N)+( VPST*GSQU*SCALE - P(NT)%ZVALF )*CSTRF(N,NT)
        ELSE
           CVPS (N)=P(NT)%ZVALF*T_INFO%NITYP(NT)
        ENDIF

      ENDDO
      ENDDO typ

      CALL SETUNB(CVPS,GRID)

      RETURN
      END SUBROUTINE

