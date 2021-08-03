#include "symbol.inc"
!************************************************************************
! RCS:  $Id: mkpoints.F,v 1.3 2003/06/27 13:22:20 kresse Exp kresse $
!
!  this module contains some control data structures for VASP
!
!***********************************************************************
MODULE MKPOINTS
  USE prec
!
!  structure required for kpoints generation
!
  TYPE kpoints_struct
!only  KPOINTS
     INTEGER NKDIM                ! maximal number of k-points
     INTEGER NKPTS                ! actual number of k-points
     INTEGER NKPTS_NON_ZERO       ! number of k-points with non zero weight
     ! it is usually save to set this to NKPTS, then some routines might do extra work
     ! NKPTS_NON_ZERO is presently only handled and used by the GW routines
     REAL(q),POINTER:: VKPT(:,:)  ! coordinate of k-point
     REAL(q),POINTER:: WTKPT(:)   ! symmetry weight-factor for each k-point
     REAL(q)        :: VOLWGT     ! volume weight for each tetrahedron
     INTEGER,POINTER:: IDTET(:,:) ! link list for tetrahedron
     INTEGER     :: NTET          ! number of tetrahedrons
     INTEGER,POINTER:: IKPT(:,:,:)! temporary work array for tetrahedron method
     LOGICAL     :: LTET          ! use tetrahedron method ?
     INTEGER ISMEAR               ! type of smearing
     REAL(q)    SIGMA             ! type of smearing
     REAL(q)    EMIN              ! minimal E for DOS
     REAL(q)    EMAX              ! maximal E for DOS
     REAL(q)    EFERMI            ! maximal E for DOS
     INTEGER    NKPX, NKPY, NKPZ  ! integer division along rec. lattice vectors for generating the mesh
     REAL(q)    B(3,3)            ! generating lattice for k-point mesh
     CHARACTER*40  SZNAMK         ! name of k-points file
     REAL(q)    SPACING           ! spacing of k-point grid in A-1
     LOGICAL    LGAMMA            ! gamma point is included
  END TYPE kpoints_struct
  
  
!-----hard limits for k-point generation package
!     NTETD  is the maximum number of tetrahedra which can be
!            stored when using the tetrahedron integration and
!     IKPTD  is the maximum number of k-mesh points in each
!            direction that can be stored in the 'connection'
!            tables for the k-points used for the symmetry
!            reduction of the tetrahedron tiling
  INTEGER, PARAMETER :: NKDIMD=20000,NTETD=90000,IKPTD=48
      
! pointer to the currently used k-points
  TYPE (kpoints_struct),SAVE,POINTER :: KPOINTS_     

! logical variable to determine whether also all k-points
! which are the difference between two other k-points are used
! in the full k-point mesh
! this is required for GW calculations

  LOGICAL,SAVE :: LSHIFT_KPOINTS=.FALSE.

CONTAINS

!************************************************************************
!
!  stores the KPOINT set presently used in the static structure KPOINTS_
!
!************************************************************************

    SUBROUTINE SETUP_KPOINTS_STATIC(KPOINTS)
      USE prec
      TYPE (kpoints_struct) KPOINTS   ! k-points structure
      
      NULLIFY(KPOINTS_)
      ALLOCATE(KPOINTS_)
      KPOINTS_=KPOINTS

    END SUBROUTINE SETUP_KPOINTS_STATIC

!************************************************************************
!
!  deallocate the static structure KPOINTS_
!
!************************************************************************

    SUBROUTINE DEALLOC_KPOINTS_STATIC
      USE prec
    
      IF (ASSOCIATED(KPOINTS_)) THEN
         DEALLOCATE(KPOINTS_%VKPT,KPOINTS_%WTKPT,KPOINTS_%IDTET)
         DEALLOCATE(KPOINTS_)
      ENDIF

    END SUBROUTINE DEALLOC_KPOINTS_STATIC

!************************************************************************
!
!  Read UNIT=14: KPOINTS
!  number of k-points and k-points in reciprocal lattice
!  than call the symmetry package to generate the k-points
!  the call SETUP_KPOINTS also initialises the static KPOINTS_ structure
!  which stores the KPOINT set presently used
!
!************************************************************************

    SUBROUTINE SETUP_KPOINTS(KPOINTS, LATT_CUR, LINVERSION, LNOSYM, IU6, IU0)
      USE prec
      USE lattice
      IMPLICIT NONE

      TYPE (kpoints_struct) KPOINTS   ! k-points structure
      TYPE (latt)        LATT_CUR     ! lattice parameters
      LOGICAL LINVERSION              ! use time inversion symmetry 
                                      ! if SO-coupling is used time inversion symmetry misses
      LOGICAL LNOSYM                  ! LNOSYM no symmetry in k-point generation
      INTEGER   IU0,IU6               ! units for output

      CALL RD_KPOINTS(KPOINTS, LATT_CUR, LINVERSION, LNOSYM, IU6, IU0)

      CALL SETUP_KPOINTS_STATIC(KPOINTS)

    END SUBROUTINE SETUP_KPOINTS


    SUBROUTINE RD_KPOINTS(KPOINTS, LATT_CUR, LINVERSION_IN, LNOSYM, IU6, IU0, MULTIPL)
      USE prec
      USE lattice
      USE constant
      USE main_mpi

      IMPLICIT NONE

      TYPE (kpoints_struct) KPOINTS   ! k-points structure
      TYPE (latt)        LATT_CUR     ! lattice parameters
      LOGICAL LINVERSION_IN           ! use time inversion symmetry 
                                      ! time inversion symmetry is lacking if SO is used
      LOGICAL LINVERSION              ! usually LINVERSION=LINVERSION_
      LOGICAL LNOSYM                  ! LNOSYM no symmetry in k-point generation
      INTEGER   IU0,IU6               ! units for output
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL :: MULTIPL ! multiplier

! local
      CHARACTER (1)   CSEL,CLINE
      REAL(q)    BK(3,3),SHIFT(3),SUPL_SHIFT(3),BK_REC(3,3)
! required for reallocation
      REAL(q),POINTER   :: VKPT(:,:),WTKPT(:),VKPT2(:,:)
      INTEGER,POINTER:: IDTET(:,:)
      INTEGER :: IERR,INDEX,NINTER,N

      INTEGER KTH,NKPX,NKPY,NKPZ,NKP,ITET,I,NT,NK
      REAL(q) RKLEN,WSUM
! warnings from tutor
      INTEGER,PARAMETER :: NTUTOR=1
      REAL(q)     RTUT(NTUTOR),RDUM
      INTEGER  ITUT(NTUTOR),IDUM
      COMPLEX(q)  CDUM  ; LOGICAL  LDUM
      LOGICAL DUMPXML

      ! If not output to IO6 or IO0, also assume no output to XML
      DUMPXML = .TRUE.
      IF (IU6==-1.AND.IU0==-1) DUMPXML = .FALSE.
         
      LINVERSION=LINVERSION_IN

      IERR=0
      
      OPEN(UNIT=14,FILE=DIR_APP(1:DIR_LEN)//'KPOINTS',STATUS='OLD',IOSTAT=IERR)
      IF (IERR/=0) THEN
         OPEN(UNIT=14,FILE='KPOINTS',STATUS='OLD',IOSTAT=IERR)
      ENDIF

      KPOINTS%NKDIM=NKDIMD
      KPOINTS%NTET=0
      KPOINTS%NKPTS=0
      KPOINTS%NKPTS_NON_ZERO=0
      KPOINTS%SZNAMK="read from INCAR"
      IF (KPOINTS%LGAMMA) THEN
         CSEL='G'
      ELSE
         CSEL='M'
      ENDIF

      ALLOCATE(VKPT(3,NKDIMD),WTKPT(NKDIMD),IDTET(0:4,NTETD))
      ALLOCATE(KPOINTS%IKPT(IKPTD,IKPTD,IKPTD))

      IF (IU6>=0) WRITE(IU6,*)
      IF (IERR==0) THEN

      ! no error, e.g. KPOINTS file exists
      ! read it in the header

         ITUT(1)=1
         READ(14,'(A40)',ERR=70111,END=70111) KPOINTS%SZNAMK
         IF (IU6>=0) WRITE(IU6,*)'KPOINTS: ',KPOINTS%SZNAMK

         ITUT(1)=ITUT(1)+1
         READ(14,*,ERR=70111,END=70111) KPOINTS%NKPTS
         IF (KPOINTS%NKPTS>KPOINTS%NKDIM) THEN
            IF (IU0>=0) &
                 WRITE(IU0,*)'ERROR in RD_KPOINTS (mkpoints.F): increase NKDIMD to',KPOINTS%NKPTS
            STOP
         ENDIF
         ITUT(1)=ITUT(1)+1
         READ(14,'(A1)',ERR=70111,END=70111) CSEL
!
! if CSEL is starting with l for line, k-points are interpolated
! between the points read from  KPOINTS
!
         IF (CSEL=='L'.OR.CSEL=='l') THEN
            CLINE='L'
            ITUT(1)=ITUT(1)+1
            READ(14,'(A1)',ERR=70111,END=70111) CSEL
            KPOINTS%NKPTS=MAX( KPOINTS%NKPTS,2)
            IF (IU6 >=0)  &
                 WRITE(IU6,*)' interpolating k-points between supplied coordinates'
            
         ELSE
            CLINE=" "
         ENDIF
         
         IF (CSEL=='K'.OR.CSEL=='k'.OR. &
              CSEL=='C'.OR.CSEL=='c') THEN
            CSEL='K'
            IF (IU6 >=0 .AND. KPOINTS%NKPTS>0)  &
                 WRITE(IU6,*)' k-points in cartesian coordinates'
         ELSE
            IF (IU6 >= 0 .AND. KPOINTS%NKPTS>0)  &
                 WRITE(IU6,*)' k-points in reciprocal lattice'
         ENDIF
         
         KPOINTS%NKPX=-1
         KPOINTS%NKPY=-1
         KPOINTS%NKPZ=-1
         KPOINTS%B=0
      ENDIF
!=======================================================================
! read in a set of k-points and interpolate NKPTS between each
!=======================================================================
      IF (KPOINTS%NKPTS>0) THEN
#ifndef oldsym
      CALL SET_SPINROT_WRAPPER(LATT_CUR%B(1,1),IU6)
#endif
      kr: IF (CLINE=='L') THEN
         IF (KPOINTS%LTET) THEN
           CALL VTUTOR('E','LINTET',RTUT,1, &
     &          ITUT,3,CDUM,1,LDUM,1,IU6,1)
           CALL VTUTOR('E','LINTET',RTUT,1, &
     &          ITUT,3,CDUM,1,LDUM,1,IU0,1)
           STOP
         ENDIF
         ALLOCATE(VKPT2(3,NKDIMD))

         NINTER=KPOINTS%NKPTS
         NKP=0  ! counter for the number of k-points already read in
         DO 
            NKP=NKP+1
            IF (NKP>KPOINTS%NKDIM) THEN
               IF (IU0>=0) &
               WRITE(IU0,*)'ERROR in RD_KPOINTS (mkpoints.F): increase NKDIMD to',NKP
               STOP
            ENDIF
            ITUT(1)=ITUT(1)+1
            READ(14,*,IOSTAT=IERR) &
     &           VKPT2(1,NKP),VKPT2(2,NKP),VKPT2(3,NKP)
            IF (IERR/=0) EXIT
         ENDDO

         KPOINTS%NKPTS=NKP-1
         IF (CSEL=='K') THEN
            VKPT2(:,1:KPOINTS%NKPTS)=  &
     &           VKPT2(:,1:KPOINTS%NKPTS)/LATT_CUR%SCALE

            CALL KARDIR(KPOINTS%NKPTS,VKPT2,LATT_CUR%A)
         ENDIF

         INDEX=0
         ! make NKPTS even
         KPOINTS%NKPTS=(KPOINTS%NKPTS/2)*2
         DO NKP=1,KPOINTS%NKPTS-1,2
            SHIFT=(VKPT2(:,NKP+1)-VKPT2(:,NKP))/(NINTER-1)
            DO N=0,NINTER-1
               INDEX=INDEX+1
               IF (INDEX>KPOINTS%NKDIM) THEN
                  WRITE(IU0,*)'ERROR in RD_KPOINTS (mkpoints.F): increase NKDIMD to',INDEX
                  STOP
               ENDIF
               VKPT(:,INDEX)=VKPT2(:,NKP)+SHIFT*N
               WTKPT(INDEX)=1._q/(KPOINTS%NKPTS/2*NINTER)
            ENDDO
         ENDDO
         KPOINTS%NKPTS=(KPOINTS%NKPTS/2)*NINTER

         ! k-point lines
         IF (DUMPXML) &
            CALL XML_KPOINTS_3(KPOINTS%NKPTS, VKPT, WTKPT, NINTER)
      ELSE kr
!=======================================================================
! Read in a given set of arbitrary k-points:
!=======================================================================
         WSUM=0
         DO NKP=1,KPOINTS%NKPTS
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70111,END=70111) &
     &           VKPT(1,NKP),VKPT(2,NKP),VKPT(3,NKP), &
     &           WTKPT(NKP)
            WSUM=WSUM+WTKPT(NKP)
         ENDDO

         IF (WSUM==0) THEN
            IF (IU0>=0) &
            WRITE(IU0,*)'ERROR: sum of weights is zero'
            STOP
         ENDIF

         IF (CSEL=='K') THEN

            VKPT(:,1:KPOINTS%NKPTS)=  &
     &           VKPT(:,1:KPOINTS%NKPTS)/LATT_CUR%SCALE

            CALL KARDIR(KPOINTS%NKPTS,VKPT,LATT_CUR%A)
         ENDIF

         WTKPT(1:KPOINTS%NKPTS)=WTKPT(1:KPOINTS%NKPTS)/WSUM

         IF (KPOINTS%LTET) THEN
! Read in tetrahedra if you want to use tetrahedron method:
            ITUT(1)=ITUT(1)+1
            READ(14,'(A)',ERR=70119,END=70119) CSEL
            IF (CSEL/='T' .AND. CSEL/='t') GOTO 70119
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70111,END=70111) KPOINTS%NTET,KPOINTS%VOLWGT
            DO ITET=1,KPOINTS%NTET
               ITUT(1)=ITUT(1)+1
               READ(14,*,ERR=70119,END=70119) (IDTET(KTH,ITET),KTH=0,4)
            ENDDO
         ENDIF
         IF (DUMPXML) &
            CALL XML_KPOINTS_1(KPOINTS%NKPTS, VKPT, WTKPT,&
                 KPOINTS%NTET, IDTET, KPOINTS%VOLWGT)

       ENDIF kr

       ELSE
!=======================================================================
! Automatic generation of a mesh if KPOINTS%NKPTS<=0:
!=======================================================================
         IF (IU6>=0 ) WRITE(IU6,'(/A)') 'Automatic generation of k-mesh.'
         SHIFT(1)=0._q
         SHIFT(2)=0._q
         SHIFT(3)=0._q
         SUPL_SHIFT=SHIFT
! k-lattice basis vectors in cartesian or reciprocal coordinates?
         IF ((CSEL/='M').AND.(CSEL/='m').AND. &
     &       (CSEL/='G').AND.(CSEL/='g').AND. &
     &       (CSEL/='A').AND.(CSEL/='a')) THEN
! Here give a basis and probably some shift (warning this shift is
! always with respect to the point (0,0,0) ... !
            IF (CSEL=='K'.OR.CSEL=='k'.OR. &
     &          CSEL=='C'.OR.CSEL=='c') THEN
               CSEL='K'
               IF (IU6>=0 ) WRITE(IU6,*)' k-lattice basis in cartesian coordinates'
            ELSE
               IF (IU6>=0 )WRITE(IU6,*)' k-lattice basis in reciprocal lattice'
            ENDIF
! Read in the basis vectors for the k-lattice (unscaled!!):
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70111,END=70111) BK(1,1),BK(2,1),BK(3,1)
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70111,END=70111) BK(1,2),BK(2,2),BK(3,2)
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70111,END=70111) BK(1,3),BK(2,3),BK(3,3)
! Correct scaling with LATT_CUR%SCALE ('lattice constant'):
            IF (CSEL=='K') BK=BK/LATT_CUR%SCALE
! Routine IBZKPT needs cartesian coordinates:
            IF (CSEL/='K') THEN
               CALL DIRKAR(3,BK,LATT_CUR%B)
            ENDIF
            KPOINTS%B=BK
! Read in the shift of the k-mesh: these values must be given in
! k-lattice basis coordinates (usually 0 or 1/2 ...):
            ITUT(1)=ITUT(1)+1
            READ(14,*,ERR=70112,END=70112) SHIFT(1),SHIFT(2),SHIFT(3)
            SUPL_SHIFT=SHIFT
70112       CONTINUE
         ELSE IF ((CSEL=='A').OR.(CSEL=='a')) THEN
            ITUT(1)=ITUT(1)+1
            READ(14,*) RKLEN
            NKPX =MAX(1._q,RKLEN*LATT_CUR%BNORM(1)+0.5_q)
            NKPY =MAX(1._q,RKLEN*LATT_CUR%BNORM(2)+0.5_q)
            NKPZ =MAX(1._q,RKLEN*LATT_CUR%BNORM(3)+0.5_q)
            IF(PRESENT(MULTIPL)) THEN
               ! use multiplier if supplied
               NKPX=NKPX*MULTIPL(1)
               NKPY=NKPY*MULTIPL(2)
               NKPZ=NKPZ*MULTIPL(3)
            ENDIF
            KPOINTS%NKPX=NKPX
            KPOINTS%NKPY=NKPY
            KPOINTS%NKPZ=NKPZ
            IF (IU0>=0) WRITE(IU0,99502) NKPX,NKPY,NKPZ
            IF (IU6>=0) WRITE(IU6,99502) NKPX,NKPY,NKPZ
99502       FORMAT( ' generate k-points for:',3I5)
            DO 99501 I=1,3
               BK(I,1)=LATT_CUR%B(I,1)/FLOAT(NKPX)
               BK(I,2)=LATT_CUR%B(I,2)/FLOAT(NKPY)
               BK(I,3)=LATT_CUR%B(I,3)/FLOAT(NKPZ)
99501       CONTINUE
         ELSE
! Here we give the Monkhorst-Pack conventions ... :
            ITUT(1)=ITUT(1)+1
            ! use default values from INCAR
            NKPX=KPOINTS%NKPX
            NKPY=KPOINTS%NKPY
            NKPZ=KPOINTS%NKPZ
            IF(PRESENT(MULTIPL)) THEN
               ! use multiplier if supplied
               NKPX = NKPX*MULTIPL(1)
               NKPY = NKPY*MULTIPL(2)
               NKPZ = NKPZ*MULTIPL(3)
            ENDIF
            IF (IERR==0) THEN
            ! k-points file was successfully opened 
            ! continue reading it
               READ(14,*,ERR=70111,END=70111) NKPX,NKPY,NKPZ
               IF(PRESENT(MULTIPL)) THEN
                  ! use multiplier if supplied
                  NKPX = NKPX*MULTIPL(1)
                  NKPY = NKPY*MULTIPL(2)
                  NKPZ = NKPZ*MULTIPL(3)
               ENDIF
               KPOINTS%NKPX=NKPX
               KPOINTS%NKPY=NKPY
               KPOINTS%NKPZ=NKPZ
               
! Shift (always in units of the reciprocal lattice vectors!):
               ITUT(1)=ITUT(1)+1
               READ(14,*,ERR=70113,END=70113) SHIFT(1),SHIFT(2),SHIFT(3)
               SUPL_SHIFT=SHIFT
70113          CONTINUE
            ELSE
               IF (IU0>=0) WRITE(IU0,99502) NKPX,NKPY,NKPZ
               IF (IU6>=0) WRITE(IU6,99502) NKPX,NKPY,NKPZ
            ENDIF
! Internal rescaling and centering according to Monkhorst-Pack:
            IF ((CSEL=='M').OR.(CSEL=='m')) THEN
               SHIFT(1)=SHIFT(1)+0.5_q*MOD(NKPX+1,2)
               SHIFT(2)=SHIFT(2)+0.5_q*MOD(NKPY+1,2)
               SHIFT(3)=SHIFT(3)+0.5_q*MOD(NKPZ+1,2)
            ENDIF
            DO I=1,3
               BK(I,1)=LATT_CUR%B(I,1)/FLOAT(NKPX)
               BK(I,2)=LATT_CUR%B(I,2)/FLOAT(NKPY)
               BK(I,3)=LATT_CUR%B(I,3)/FLOAT(NKPZ)
            ENDDO
! At this point hope that routine IBZKPT accepts all ... !
         ENDIF
! Find all irreducible points in the first Brillouin zone ... :
         IF (.NOT.LINVERSION) THEN
!            CALL VTUTOR('W','LNOINVERSION',RTUT,1, &
!     &           ITUT,1,CDUM,1,LDUM,1,IU6,2)
!            CALL VTUTOR('W','LNOINVERSION',RTUT,1, &
!     &           ITUT,1,CDUM,1,LDUM,1,IU0,2)
         ENDIF
         CALL IBZKPT(LATT_CUR%B(1,1),BK,SHIFT,KPOINTS%NKPTS, &
              VKPT(1,1),WTKPT(1),KPOINTS%NKDIM, &
              KPOINTS%LTET,KPOINTS%NTET,IDTET(0,1),NTETD,KPOINTS%VOLWGT, &
              KPOINTS%IKPT(1,1,1),IKPTD,LATT_CUR%SCALE,LINVERSION,LNOSYM,LSHIFT_KPOINTS,IU6)

         ! NKPTS_NON_ZERO is the number of non-zero k-points
         ! presently this will differ from NKPTS only if LSHIFT_KPOINTS
         ! is used resulting in the addition of the difference vectors
         ! (differences between all k-points)
         KPOINTS%NKPTS_NON_ZERO=KPOINTS%NKPTS
         DO NK=1,KPOINTS%NKPTS
            IF (WTKPT(NK)==0) KPOINTS%NKPTS_NON_ZERO=KPOINTS%NKPTS_NON_ZERO-1
         ENDDO

         ! k-point lines
         BK_REC=BK
         CALL KARDIR(3,BK_REC,LATT_CUR%A)
         IF (DUMPXML) &
            CALL XML_KPOINTS_2(KPOINTS%NKPTS, VKPT, WTKPT,&
                 KPOINTS%NTET, IDTET, KPOINTS%VOLWGT, &
                 CSEL, RKLEN, NKPX, NKPY, NKPZ, SUPL_SHIFT, SHIFT, BK_REC )

      ENDIF

! set old k-points
      GOTO 70222
70111 CONTINUE
      CALL VTUTOR('E','KPOINTS',RTUT,1, &
     &     ITUT,1,CDUM,1,LDUM,1,IU6,1)
      CALL VTUTOR('E','KPOINTS',RTUT,1, &
     &     ITUT,1,CDUM,1,LDUM,1,IU0,1)
      STOP
70119 CONTINUE
      CALL VTUTOR('E','KPOINTSTET',RTUT,1, &
     &     ITUT,1,CDUM,1,LDUM,1,IU6,1)
      CALL VTUTOR('E','KPOINTSTET',RTUT,1, &
     &     ITUT,1,CDUM,1,LDUM,1,IU0,1)
      STOP
70222 CONTINUE
#ifdef gammareal
      IF (KPOINTS%NKPTS/=1._q .OR. &
     &    VKPT(1,1)**2+VKPT(2,1)**2 &
     &   +VKPT(3,1)**2>1E-10_q) THEN

         CALL VTUTOR('E','GAMMAK',RTUT,1, &
     &        ITUT,1,CDUM,1,LDUM,1,IU6,1)
         CALL VTUTOR('E','GAMMAK',RTUT,1, &
     &        ITUT,1,CDUM,1,LDUM,1,IU0,1)
        STOP
      ENDIF
#endif
      DEALLOCATE(KPOINTS%IKPT)
!
!  reallocate everthing with the minimum number of kpoints set
!
      NK=KPOINTS%NKPTS
      NT=MAX(KPOINTS%NTET,1)

      ALLOCATE(KPOINTS%VKPT(3,NK),KPOINTS%WTKPT(NK),KPOINTS%IDTET(0:4,NT))
      KPOINTS%VKPT = VKPT(1:3,1:NK)
      KPOINTS%WTKPT= WTKPT(1:NK)
      IF (KPOINTS%NTET==0) THEN
         KPOINTS%IDTET= 0
      ELSE
         KPOINTS%IDTET= IDTET(0:4,1:NT)
      ENDIF
      DEALLOCATE(VKPT,WTKPT,IDTET)
      KPOINTS%NKDIM=NK
      CLOSE(UNIT=14)

      ! if NKPTS_NON_ZERO is not yet set, simply copy KPOINTS%NKPTS
      IF (KPOINTS%NKPTS_NON_ZERO==0) THEN 
         KPOINTS%NKPTS_NON_ZERO=KPOINTS%NKPTS
      ENDIF

      RETURN
    END SUBROUTINE RD_KPOINTS

!***********************************************************************
! search a particular k-point and return
! the equivalent k-point in the array, if not found return -1
! this is equal to the routine KPOINT_IN_FULL_GRID, except 
! that this routine returns -1 and no error if point is not found
! in the KPOINTS_F array
!
!***********************************************************************

  FUNCTION KPOINT_IN_FULL_GRID_KINTER(VKPT,KPOINTS_F)
    !USE sym_prec
    USE lattice
    INTEGER :: KPOINT_IN_FULL_GRID_KINTER
    REAL(q) VKPT(:)
    TYPE (kpoints_struct) :: KPOINTS_F
    ! local
    REAL(q), PARAMETER :: TINY=1E-8_q
    INTEGER NK
    DO NK=1,KPOINTS_F%NKPTS
       IF ( &
         (ABS(MOD(VKPT(1)-KPOINTS_F%VKPT(1,NK)+10.5_q,1._q)-0.5_q)<TINY) .AND. &
         (ABS(MOD(VKPT(2)-KPOINTS_F%VKPT(2,NK)+10.5_q,1._q)-0.5_q)<TINY) .AND. &
         (ABS(MOD(VKPT(3)-KPOINTS_F%VKPT(3,NK)+10.5_q,1._q)-0.5_q)<TINY)) EXIT
    ENDDO

    IF (NK>KPOINTS_F%NKPTS) THEN
       ! no kpoint found, set nk=-1
       NK=-1
    ENDIF

    KPOINT_IN_FULL_GRID_KINTER=NK
  END FUNCTION KPOINT_IN_FULL_GRID_KINTER

!*************************************************************************
!
! if this subroutine is called the full k-point grid contains
! also all k-points that are difference vectors between two other k-points
!
!*************************************************************************

    SUBROUTINE USE_SHIFT_KPOINTS
      LSHIFT_KPOINTS=.TRUE.
    END SUBROUTINE USE_SHIFT_KPOINTS

!*************************************************************************
!
! duplicates KPOINTS structure
!
!*************************************************************************

    SUBROUTINE COPY_KPOINTS(K1,K2)
      IMPLICIT NONE
      TYPE(kpoints_struct) :: K1,K2
      
      K2%NKDIM=K1%NKDIM
      K2%NKPTS=K1%NKPTS
      IF(ASSOCIATED(K1%VKPT))THEN
         ALLOCATE(K2%VKPT(SIZE(K1%VKPT,1),SIZE(K1%VKPT,2)))
         K2%VKPT=K1%VKPT
      END IF
      IF(ASSOCIATED(K1%WTKPT))THEN
         ALLOCATE(K2%WTKPT(SIZE(K1%WTKPT,1)))
         K2%WTKPT=K1%WTKPT
      END IF
      K2%VOLWGT=K1%VOLWGT
      IF(ASSOCIATED(K1%IDTET))THEN
         ALLOCATE(K2%IDTET(SIZE(K1%IDTET,1),SIZE(K1%IDTET,2)))
         K2%IDTET=K1%IDTET
      END IF
      K2%NTET=K1%NTET
      IF(ASSOCIATED(K1%IKPT))THEN
         ALLOCATE(K2%IKPT(SIZE(K1%IKPT,1),SIZE(K1%IKPT,2),SIZE(K1%IKPT,3)))
         K2%IKPT=K1%IKPT
      END IF
      K2%LTET=K1%LTET
      K2%ISMEAR=K1%ISMEAR
      K2%SIGMA=K1%SIGMA
      K2%EMIN=K1%EMIN
      K2%EMAX=K1%EMAX
      K2%EFERMI=K1%EFERMI
      K2%NKPX=K1%NKPX
      K2%NKPY=K1%NKPY
      K2%NKPZ=K1%NKPZ
      K2%B=K1%B
      K2%SZNAMK=K1%SZNAMK
      K2%SPACING=K1%SPACING
      K2%LGAMMA=K1%LGAMMA
    END SUBROUTINE COPY_KPOINTS
      

END MODULE MKPOINTS
