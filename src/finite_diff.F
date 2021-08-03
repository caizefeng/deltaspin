#include "symbol.inc"

MODULE finite_differences
  USE prec
  USE vaspxml
  IMPLICIT NONE

!
! structure to store all information related to second derivative
! this entry can be read using READ_DYNMATFULL
! 
  TYPE DYNMATFULL_HANDLE 
     INTEGER :: NIONS, NTYP             ! number of ions and types
     INTEGER,ALLOCATABLE :: NITYP(:)    ! number of atoms for each type
     REAL(q),ALLOCATABLE :: POMASS(:)   ! mass of each ion in a.u.
     REAL(q)             :: TOTEN       ! energy at zero displacement
     REAL(q),ALLOCATABLE :: POS(:,:)    ! initial positions (not displaced)
     REAL(q),ALLOCATABLE :: EVAL(:)     ! eigenvalues in eV/A^2/atomic mass
     REAL(q),ALLOCATABLE :: EVEC(:,:,:) ! eigenvectors (one normalized)
  END TYPE DYNMATFULL_HANDLE

  TYPE (DYNMATFULL_HANDLE), POINTER :: DYNMH

CONTAINS

!*********************************************************************
! RCS:  $Id: finite_diff.F,v 1.4 2003/06/27 13:22:18 kresse Exp kresse $
!
! calculate second derivatives using finite differences
! implemented by Orest Dubay
! some cleanup and
! elastic moduli added and internal strain derivatives added by gK
!
!*********************************************************************

  SUBROUTINE FINITE_DIFF( LSTOP, STEP, NIONS, NTYP, NITYP,MASSES, POS, FORCE, NDISPL, &
       LSDYN, LSFOR, A, B, IU6, IU0, NWRITE)

    USE lattice

    IMPLICIT NONE

    LOGICAL :: LSTOP              ! on return: true to stop main code
    LOGICAL :: LSDYN              ! selective dynamics (yes/ no)
    INTEGER :: NIONS              ! number ions
    INTEGER :: NTYP               ! number of types of ions
    INTEGER :: NITYP(NTYP)        ! number of species
    REAL(q) :: MASSES(NTYP)       ! masses of species
    REAL(q) :: STEP               ! step size
    REAL(q) :: POS(3,NIONS)       ! positions in terms of direct lattice
    REAL(q) :: FORCE(3,NIONS)     ! forces in cartesian coordinates
    REAL(q) :: A(3,3)             ! lattice vectors
    REAL(q) :: B(3,3)             ! reciprocal lattice vectors
    LOGICAL :: LSFOR(3,NIONS)     ! selective 
    INTEGER :: IU6                ! OUTCAR file
    INTEGER :: IU0                ! stdout
    INTEGER :: NWRITE             ! how much is written out by the routine
    INTEGER :: IUDYNMAT           ! DYNMAT file
    INTEGER :: NDISPL             ! number of displacement

! local variables
    REAL(q) :: X             
    REAL(q),ALLOCATABLE,SAVE :: INITIAL_POSITIONS(:,:)
    REAL(q),ALLOCATABLE,SAVE :: INITIAL_FORCE(:,:)
    REAL(q),ALLOCATABLE,SAVE :: DISPL_FORCES(:,:,:,:)
    REAL(q),ALLOCATABLE      :: SUM_FORCES(:,:,:)
    REAL(q),ALLOCATABLE      :: SECOND_DERIV(:,:)
    INTEGER,SAVE             :: DOF
    LOGICAL,SAVE             :: INIT=.FALSE.
    INTEGER,SAVE             :: PROCESSED_DOF
    INTEGER,SAVE             :: PROCESSED_DISPL
    INTEGER                  :: I,J,K,M,N
    REAL(q),ALLOCATABLE      :: WORK(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVECTORS(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVALUES(:)
    INTEGER                  :: IERROR

!=======================================================================
! 
! initialization
!
!=======================================================================
    IF (.NOT.INIT) THEN
      IF (STEP>0.1) THEN
         CALL VTUTOR('W','POTIM large', &
     &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.FALSE.,1,IU0,3)
         CALL VTUTOR('W','POTIM large', &
     &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.FALSE.,1,IU6,3)
         STEP=0.015_q
      ENDIF

      IF (NDISPL>=4) THEN
         NDISPL=4
      ELSE IF (NDISPL<=1) THEN
         NDISPL=1
      ELSE
         NDISPL=2
      ENDIF
      
      CALL COUNT_DOF(NIONS, LSFOR, LSDYN, DOF)
      ALLOCATE(INITIAL_POSITIONS(3,NIONS))
      ALLOCATE(INITIAL_FORCE(3,NIONS))
      ALLOCATE(DISPL_FORCES(NDISPL,DOF,3,NIONS))
      INITIAL_POSITIONS             = POS
      INITIAL_FORCE                 = FORCE
      PROCESSED_DISPL               = NDISPL
      PROCESSED_DOF                 = 0
      IF (IU0>=0) WRITE (IU0,'(A,F8.5,A,I4)') ' Finite differences POTIM=',STEP,' DOF=',DOF
      IF (IU6>=0) THEN
        WRITE (IU6,*) 'Finite differences:'
        WRITE (IU6,*) '  Step               POTIM = ',STEP
        WRITE (IU6,*) '  Degrees of freedom DOF   = ',DOF
      END IF
      INIT                          = .TRUE.
    END IF

    IF (PROCESSED_DOF>0) THEN
      IF (IU0>=0) THEN
        WRITE (IU0,*)'Finite differences progress:'
        WRITE (IU0,'(A,I3,A,I3)') &
           '  Degree of freedom: ',PROCESSED_DOF,'/',DOF
        WRITE (IU0,'(A,I3,A,I3)') &
           '  Displacement:      ',PROCESSED_DISPL,"/",NDISPL
        WRITE (IU0,'(A,I3,A,I3)') &
           '  Total:             ',(PROCESSED_DOF-1)*NDISPL+PROCESSED_DISPL,&
           '/',DOF*NDISPL
      END IF
     
      IF (IU6>=0) THEN
        WRITE (IU6,*)'Finite differences progress:'
        WRITE (IU6,'(A,I3,A,I3)') &
           '  Degree of freedom: ',PROCESSED_DOF,'/',DOF
        WRITE (IU6,'(A,I3,A,I3)') &
           '  Displacement:      ',PROCESSED_DISPL,"/",NDISPL
        WRITE (IU6,'(A,I3,A,I3)') &
           '  Total:             ',(PROCESSED_DOF-1)*NDISPL+PROCESSED_DISPL,&
           '/',DOF*NDISPL
      END IF
     
      DISPL_FORCES(PROCESSED_DISPL,PROCESSED_DOF,:,:)=FORCE
    END IF

    PROCESSED_DISPL   = PROCESSED_DISPL+1
    IF (PROCESSED_DISPL>NDISPL) THEN
      PROCESSED_DISPL = 1
      PROCESSED_DOF   = PROCESSED_DOF+1
    END IF

    POS=INITIAL_POSITIONS

    IF (PROCESSED_DOF.LE.DOF) THEN
      LSTOP=.FALSE.
      ! IF (IU0>=0) WRITE (IU0,*) 'Finite differences DOF             = ',DOF
      ! IF (IU0>=0) WRITE (IU0,*) 'Finite differences PROCESSED_DOF   = ',PROCESSED_DOF
      ! IF (IU0>=0) WRITE (IU0,*) 'Finite differences PROCESSED_DISPL = ',PROCESSED_DISPL
      ! IF (IU0>=0) WRITE (IU0,*) 'Finite differences POSITIONS BEFORE DISPL'

      ! CALL PRINT_POSITIONS(NIONS,POS,IU0,A,B)

    ! PROCESSED_DISPL ordering:
    !
    ! \_           _/   
    !   \__     __/
    !      \---/
    !  |  |  |  |  |
    !       (0) 1     NDISPL=1
    !     2     1     NDISPL=2
    !  4  3     2  1  NDISPL=4
    
      IF(NDISPL<=2) THEN
         SELECT CASE(PROCESSED_DISPL)
         CASE(1)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF,   STEP,A,B)
         CASE(2)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF,  -STEP,A,B)
         END SELECT
      ELSE
         SELECT CASE(PROCESSED_DISPL)
         CASE(1)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF, 2*STEP,A,B)
         CASE(2)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF,   STEP,A,B)
         CASE(3)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF,  -STEP,A,B)
         CASE(4)
            CALL MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,PROCESSED_DOF,-2*STEP,A,B)
         END SELECT
      END IF
      ! IF (IU0>=0) WRITE (IU0,*) 'Finite differences POSITIONS AFTER DISPL'
      ! CALL PRINT_POSITIONS(NIONS,POS,IU0,A,B)

      RETURN
    ELSE

    !
    ! Final processing + output is here:
    !

      IF (IU6>=0 .AND. NWRITE >=3 ) THEN
        WRITE (IU6,*)
        WRITE (IU6,*) 'FORCES'
        WRITE (IU6,*) '------'
        WRITE (IU6,*)
        WRITE (IU6,*) 'INITIAL FORCE'
        CALL PRINT_FORCE(NIONS,INITIAL_FORCE,IU6)

        DO N=1,DOF
          CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
          DO M=1,NDISPL
            WRITE (IU6,'("DOF:",I4," ATOM:",I4," AXIS:",I4," DISPLACEMENT:",I4)')N,J,I,M
            CALL PRINT_FORCE(NIONS,DISPL_FORCES(M,N,:,:),IU6)
            WRITE (IU6,*)
          END DO
        END DO
        WRITE (IU6,*) ' '
      END IF

      ALLOCATE(SUM_FORCES(DOF,3,NIONS))
      SELECT CASE(NDISPL)
      CASE(1)
        DO N=1,DOF
          SUM_FORCES(N,:,:)=(DISPL_FORCES(1,N,:,:)-INITIAL_FORCE)/STEP
        END DO
      CASE(2)
        SUM_FORCES = (1._q/(2._q*STEP))*(DISPL_FORCES(1,:,:,:)-DISPL_FORCES(2,:,:,:))
      CASE(4)
        SUM_FORCES = (1._q/(12._q*STEP))* &
                                (8._q*DISPL_FORCES(2,:,:,:)-8._q*DISPL_FORCES(3,:,:,:) &
                                     -DISPL_FORCES(1,:,:,:)+     DISPL_FORCES(4,:,:,:))
      END SELECT

      CALL PRINT_DYNMAT(NIONS,DOF,STEP,NTYP,NITYP,MASSES,STEP*SUM_FORCES,LSDYN,LSFOR,IU6)
     
      ALLOCATE(SECOND_DERIV(DOF,DOF))
      DO N=1,DOF
        CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)       
        DO M=1,DOF
          SECOND_DERIV(M,N)=SUM_FORCES(M,I,J)
        END DO
      END DO
     
      IF (IU6>=0) THEN
        WRITE (IU6,*) 
        WRITE (IU6,*) 'SECOND DERIVATIVES (NOT SYMMETRIZED)'
        WRITE (IU6,*) '------------------------------------'
        CALL PRINT_SECOND_DERIV(NIONS,DOF,SECOND_DERIV,LSFOR,LSDYN,IU6)
      END IF

      DO N=1,DOF
        DO M=N+1,DOF
          X=0.5_q*(SECOND_DERIV(N,M)+SECOND_DERIV(M,N))
          !WRITE(0,*) N,M,SECOND_DERIV(N,M),SECOND_DERIV(M,N),X
          SECOND_DERIV(N,M)=X
          SECOND_DERIV(M,N)=X
          !WRITE(0,*) N,M,SECOND_DERIV(N,M),SECOND_DERIV(M,N),X
        END DO
      END DO

      N=1
      DO I=1,NTYP
        DO J=1,NITYP(I)
          DO K=1,3
            CALL FIND_DOF_INDEX(NIONS,LSFOR,LSDYN,K,N,M)
            IF (M>0) SECOND_DERIV(:,M)=SECOND_DERIV(:,M)/SQRT(MASSES(I))
            IF (M>0) SECOND_DERIV(M,:)=SECOND_DERIV(M,:)/SQRT(MASSES(I))
          END DO
          N=N+1
        END DO
      END DO

      ALLOCATE(WORK(DOF,32),EIGENVECTORS(DOF,DOF),EIGENVALUES(DOF))
      EIGENVECTORS=SECOND_DERIV
      CALL XML_TAG("dynmat")
        CALL XML_VECARRAY("hessian")
        CALL XML_ARRAY_REAL(SECOND_DERIV)
        CALL XML_CLOSE_TAG
  
      CALL DSYEV &
              ('V','U',DOF,EIGENVECTORS,DOF, &
              EIGENVALUES,WORK,32*DOF, IERROR)

        CALL XML_VEC_REAL(EIGENVALUES,"eigenvalues",'(ES16.8)')
        CALL XML_VECARRAY("eigenvectors")
        CALL XML_ARRAY_REAL(EIGENVECTORS,'(ES16.8)')
        CALL XML_CLOSE_TAG


      IF (IERROR/=0) THEN
        IF (IU6>=0) THEN
          WRITE(IU6,*) "Error while diagonalisation DSYEV INFO=",IERROR
          WRITE(IU6,*) "Some of (or all) eigenvectors and eigenvalues are not correct !"
        END IF
      END IF

      CALL PRINT_EIGENVECTORS(NIONS,DOF,INITIAL_POSITIONS,A, &
           EIGENVECTORS,      &
           EIGENVALUES,       &
           LSFOR,LSDYN,IU6)

      N=1
      DO I=1,NTYP
        DO J=1,NITYP(I)
          DO K=1,3
            CALL FIND_DOF_INDEX(NIONS,LSFOR,LSDYN,K,N,M)
            IF (M>0) EIGENVECTORS(M,:)=EIGENVECTORS(M,:)/SQRT(MASSES(I))
          END DO
          N=N+1
        END DO
      END DO

      IF (IU6>=0 .AND. NWRITE>=3) THEN
         WRITE(IU6,*) "Eigenvectors after division by SQRT(mass)"
         CALL PRINT_EIGENVECTORS(NIONS,DOF,INITIAL_POSITIONS,A, &
              EIGENVECTORS,      &
              EIGENVALUES,       &
              LSFOR,LSDYN,IU6)
      ENDIF


      CALL XML_CLOSE_TAG
      DEALLOCATE(WORK)
      DEALLOCATE(INITIAL_POSITIONS)
      DEALLOCATE(INITIAL_FORCE)
      DEALLOCATE(DISPL_FORCES)
      DEALLOCATE(SECOND_DERIV,SUM_FORCES)
      LSTOP=.TRUE.
    END IF


    IF (IU0>=0) WRITE (IU0,*) 'Finite differences POTIM=',STEP
    IF (IU6>=0) WRITE (IU6,*) 'Finite differences POTIM=',STEP

    ! converts a vector with three entries from direct to cartesian
    !CALL DIRKAR(1,X,A)
    
    ! converts a vector from cartesian to direct coordinates
    !CALL KARDIR(1,X,B)


!         CALL DSYEV &
!              ('V','U',NDIM,MATRIX,NDIM, &
!              EIGENVALUES,CWORK,*NDIM, IERROR)

    LSTOP=.TRUE.
  END SUBROUTINE FINITE_DIFF

!*********************************************************************
!
! calculate second derivatives using finite differences
! only symmetry-independ directions are calculated,
! the dynamical matrix is completed using the symmetry (SYDMAT routine)
! implemented by Orest Dubay
!
!*********************************************************************

  SUBROUTINE FINITE_DIFF_ID( LSTOP, STEP, T_INFO, POS, TOTEN, FORCE, SIF, NDISPL, &
       LSIF, A, B , SAXIS, SYMM, ISPIN, ISPECIAL, TEBEG, IU6, IU0, NWRITE)

    USE base
    USE lattice
    USE poscar
    USE msymmetry
    USE constant
    USE pead
    IMPLICIT NONE

    LOGICAL :: LSTOP              ! on return: true to stop main code
    TYPE (type_info)   T_INFO
    REAL(q) :: STEP               ! step size
    REAL(q) :: POS(3,T_INFO%NIONS)       ! positions in terms of direct lattice
    REAL(q) :: TOTEN              ! total energy
    REAL(q) :: FORCE(3,T_INFO%NIONS)     ! forces in cartesian coordinates
    REAL(q) :: SIF(3,3)           ! stress tensor
    REAL(q) :: A(3,3)             ! lattice vectors
    REAL(q) :: B(3,3)             ! reciprocal lattice vectors
    LOGICAL :: LSIF               ! also stress-strain derivatives
    INTEGER :: IU6                ! OUTCAR file
    INTEGER :: IU0                ! stdout
    INTEGER :: NWRITE             ! how much is written out
    INTEGER :: IUDYNMAT           ! DYNMAT file
    INTEGER :: NDISPL             ! number of displacement
    REAL(q) :: SAXIS(3)           ! quantisation axis for spin
    TYPE (symmetry)    SYMM
    INTEGER :: ISPIN
    INTEGER :: ISPECIAL           ! write POSCAR files with displaces phonon modes imposed
    REAL(q) :: TEBEG              ! temperature

! local variables
    INTEGER :: NIONS
    REAL(q) :: X             
    REAL(q), SAVE            :: INITIAL_TOTEN
    REAL(q),ALLOCATABLE,SAVE :: INITIAL_POSITIONS(:,:)
    REAL(q), SAVE            :: INITIAL_A(3,3)
    REAL(q),ALLOCATABLE,SAVE :: INITIAL_FORCE(:,:)
    REAL(q), SAVE            :: INITIAL_SIF(3,3)
    REAL(q),ALLOCATABLE,SAVE :: DISPL_FORCES(:,:,:,:)
    REAL(q),ALLOCATABLE,SAVE :: DISPL_SIF(:,:,:,:)
    REAL(q),ALLOCATABLE      :: SUM_FORCES(:,:,:)
    REAL(q),ALLOCATABLE      :: SUM_SIF(:,:,:)
    REAL(q)                  :: SECOND_DERIV(3*T_INFO%NIONS,3*T_INFO%NIONS)
    INTEGER,SAVE             :: DOF
    LOGICAL,SAVE             :: INIT=.FALSE.
    INTEGER,SAVE             :: PROCESSED_DOF
    INTEGER,SAVE             :: PROCESSED_DISPL
    INTEGER                  :: I,J,K,M,N
    REAL(q),ALLOCATABLE      :: WORK(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVECTORS(:,:)
    REAL(q),ALLOCATABLE      :: EIGENVALUES(:)
    INTEGER                  :: IERROR
    REAL(q)                  :: VEL(3,T_INFO%NIONS)
    REAL(q)                  :: WORKD(3,3,T_INFO%NIONS),WDMAT(3,T_INFO%NIONS,3,T_INFO%NIONS)
    REAL(q)                  :: ST(3,3,3,T_INFO%NIONS), AST(3,3,3,T_INFO%NIONS)
    REAL(q)                  :: ELASTIC(3,3,3,3), ELASTICP(6,3,3)
    INTEGER                  :: IWORK(T_INFO%NIONS)
    REAL(q),ALLOCATABLE,SAVE :: D(:,:,:) !D(3,3,T_INFO%NIONS)
    INTEGER                  :: NRTK, NPCLL
    INTEGER,ALLOCATABLE,SAVE :: ND(:), IDIRD(:,:)
    REAL(q)                  :: DMAT(3,3,T_INFO%NIONS,T_INFO%NIONS)
    LOGICAL                  :: LDO(T_INFO%NIONS)
    INTEGER                  :: I1,J1,I2,J2,IA1,IA2,INDFST(T_INFO%NIONS)
    CHARACTER(3)             :: IDIR_TEXT(3)=(/"x","y","z"/)
    REAL(q)                  :: FACT
    REAL(q)                  :: PIEZO(3,3,3),EPSILON(3,3)
! phonon_
    INTEGER                  :: NIONP
! phonon_
! symmetry related quantities (common block)
    INTEGER  ISYMOP,NROT,IGRPOP,NROTK,INVMAP,NPCELL
    REAL(q)  GTRANS,AP,OMEGA
    COMMON /SYMM/   ISYMOP(3,3,48),NROT,IGRPOP(3,3,48),NROTK, &
         GTRANS(3,48),INVMAP(48),AP(3,3),NPCELL
    NIONS=T_INFO%NIONS
!=======================================================================
! 
! initialization
!
!=======================================================================

    IF (.NOT.INIT) THEN
      IF (STEP>0.1) THEN
         CALL VTUTOR('W','POTIM large', &
     &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.FALSE.,1,IU0,3)
         CALL VTUTOR('W','POTIM large', &
     &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.FALSE.,1,IU6,3)
         STEP=0.015_q
       ENDIF

      ALLOCATE(D(3,3,NIONS), ND(NIONS),IDIRD(3,NIONS))

      CALL FREDOM(SYMM%ROTMAP,ISYMOP,INVMAP,NROTK,NPCELL,D,ND,1,T_INFO%NTYP,NIONS, &
             T_INFO%NITYP,WORKD,IWORK, & 
             A(1,1),A(1,2),A(1,3), &
             B(1,1),B(1,2),B(1,3),IDIRD,.FALSE.)

      DOF=SUM(ND)

      IF (IU6>=0) THEN
         WRITE(IU6,*)
         WRITE(IU6,*)
         WRITE(IU6,'(A,I5,A)') 'Found ',DOF,' degrees of freedom:'
         DO J=1,NIONS
            IF (ND(J).GT.0) THEN
               WRITE(IU6,*)
               WRITE(IU6,*)
               WRITE(IU6,'(A,I5,A)') '        Directions for atom ',J,':'
               WRITE(IU6,'(A)')      '        --------------------------'
               WRITE(IU6,*)
               DO I=1,ND(J)
                  WRITE(IU6,'(8X,3F24.16)') D(1,I,J),D(2,I,J),D(3,I,J)
               END DO
            ENDIF
         END DO
      END IF
      IF (IU0>=0) THEN
         WRITE(IU0,'(A,I5,A)') ' Found ',DOF,' degrees of freedom:'
         WRITE(17,'(A,I5,A)')  ' Found ',DOF,' degrees of freedom:'
      ENDIF

      IF (LSIF) THEN
         IF (IU6>=0) THEN
            WRITE(IU6,'(A,I5,A)') '        Strain: 6 additional degrees of freedom'
            WRITE(IU6,'(A)')      '        ---------------------------------------'
         ENDIF
         DOF=DOF+6
         IF (IU0>=0) THEN
            WRITE(IU0,'(A,I5,A)') ' Adding 6 more for cell shape distortion'
            WRITE(17,'(A,I5,A)')  ' Adding 6 more for cell shape distortion'
         ENDIF
      ENDIF

      IF (NDISPL>=4) THEN
         NDISPL=4
      ELSE IF (NDISPL<=1) THEN
         NDISPL=1
      ELSE
         NDISPL=2
      ENDIF
      
      ALLOCATE(INITIAL_POSITIONS(3,NIONS))
      ALLOCATE(INITIAL_FORCE(3,NIONS))
      ALLOCATE(DISPL_FORCES(NDISPL,DOF,3,NIONS))
      ALLOCATE(DISPL_SIF(NDISPL,DOF,3,3))
      INITIAL_POSITIONS             = POS
      INITIAL_TOTEN                 = TOTEN
      INITIAL_A                     = A
      INITIAL_FORCE                 = FORCE
      INITIAL_SIF                   = SIF
      DISPL_FORCES                  = 0
      DISPL_SIF                     = 0
      PROCESSED_DISPL               = NDISPL
      PROCESSED_DOF                 = 0
      IF (IU0>=0) WRITE (IU0,'(A,F8.5,A,I4)') ' Finite differences POTIM=',STEP,' DOF=',DOF
      IF (IU6>=0) THEN
         WRITE (IU6,*) 'Finite differences:'
         WRITE (IU6,*) '  Step               POTIM = ',STEP
         WRITE (IU6,*) '  Degrees of freedom DOF   = ',DOF
      END IF

      INIT = .TRUE.
    ENDIF
!=======================================================================
! 
! make displacement
!
!=======================================================================
    IF (PROCESSED_DOF>0) THEN
       IF (IU0>=0) THEN
          WRITE (IU0,*)'Finite differences progress:'
          WRITE (IU0,'(A,I3,A,I3)') &
               '  Degree of freedom: ',PROCESSED_DOF,'/',DOF
          WRITE (IU0,'(A,I3,A,I3)') &
               '  Displacement:      ',PROCESSED_DISPL,"/",NDISPL
          WRITE (IU0,'(A,I3,A,I3)') &
               '  Total:             ',(PROCESSED_DOF-1)*NDISPL+PROCESSED_DISPL,&
               '/',DOF*NDISPL
       END IF

       IF (IU6>=0) THEN
          WRITE (IU6,*)'Finite differences progress:'
          WRITE (IU6,'(A,I3,A,I3)') &
               '  Degree of freedom: ',PROCESSED_DOF,'/',DOF
          WRITE (IU6,'(A,I3,A,I3)') &
               '  Displacement:      ',PROCESSED_DISPL,"/",NDISPL
          WRITE (IU6,'(A,I3,A,I3)') &
               '  Total:             ',(PROCESSED_DOF-1)*NDISPL+PROCESSED_DISPL,&
               '/',DOF*NDISPL
       END IF

       DISPL_FORCES(PROCESSED_DISPL,PROCESSED_DOF,:,:)=FORCE
       DISPL_SIF   (PROCESSED_DISPL,PROCESSED_DOF,:,:)=SIF
    END IF

    PROCESSED_DISPL   = PROCESSED_DISPL+1
    IF (PROCESSED_DISPL>NDISPL) THEN
       PROCESSED_DISPL = 1
       PROCESSED_DOF   = PROCESSED_DOF+1
    END IF

    IF (PROCESSED_DOF.LE.DOF) THEN
       LSTOP=.FALSE.
       
       POS=INITIAL_POSITIONS
       A  =INITIAL_A
       CALL LATOLD(OMEGA,A,B)

       ! PROCESSED_DISPL ordering:
       !
       ! \_           _/   
       !   \__     __/
       !      \---/
       !  |  |  |  |  |
       !       (0) 1     NDISPL=1
       !     2     1     NDISPL=2
       !  4  3     2  1  NDISPL=4
    
       IF(NDISPL<=2) THEN
          SELECT CASE(PROCESSED_DISPL)
          CASE(1)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND,  STEP,A,B,IU0)
          CASE(2)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND, -STEP,A,B,IU0)
          END SELECT
       ELSE
          SELECT CASE(PROCESSED_DISPL)
          CASE(1)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND, 2*STEP,A,B,IU0)
          CASE(2)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND,   STEP,A,B,IU0)
          CASE(3)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND,  -STEP,A,B,IU0)
          CASE(4)
             CALL MAKE_DISPLACED_ID(LSIF,NIONS,POS,PROCESSED_DOF, D, ND,-2*STEP,A,B,IU0)
          END SELECT
       END IF

       RETURN
    ELSE
!=======================================================================
! 
! final processing
!
!=======================================================================
130    FORMAT (5X, //, &
            &'----------------------------------------------------', &
            &'----------------------------------------------------'//)

       IF(IU6>=0) WRITE(IU6,130)

       POS=INITIAL_POSITIONS
       A  =INITIAL_A
       CALL LATOLD(OMEGA,A,B)

       IF (SYMM%ISYM>0) THEN
          ! well here we need to get back all the original symmetry
          VEL=0
          CALL INISYM(A,POS,VEL,T_INFO%LSFOR,T_INFO%LSDYN,T_INFO%NTYP,T_INFO%NITYP,NIONS, &
               SYMM%PTRANS,SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%TAU,SYMM%TAUROT,SYMM%WRKROT, &
               SYMM%INDROT,T_INFO%ATOMOM,SAXIS,SYMM%MAGROT,ISPIN,IU6=-1)
       ENDIF
!
! some low level information mostly for debugging
!
       IF (IU6>=0 .AND. NWRITE >=3) THEN
          WRITE (IU6,*)
          WRITE (IU6,*) 'FORCES'
          WRITE (IU6,*) '------'
          WRITE (IU6,*)
          WRITE (IU6,*) 'INITIAL FORCE'
          CALL PRINT_FORCE(NIONS,INITIAL_FORCE,IU6)
          CALL PRINT_STRESS(LSIF,INITIAL_SIF,IU6)

          DO N=1,DOF
             CALL FIND_IJ_ID(LSIF,NIONS,N,ND,J,I)
             DO M=1,NDISPL
                WRITE (IU6,'("DOF:",I4," ATOM:",I4," AXIS:",I4," DISPLACEMENT:",I4)')N,J,I,M
                CALL PRINT_FORCE(NIONS,DISPL_FORCES(M,N,:,:),IU6)
                CALL PRINT_STRESS(LSIF,DISPL_SIF(M,N,:,:),IU6)
                WRITE (IU6,*)
             END DO
          END DO
          WRITE (IU6,*) ' '
       END IF
!
! contract calculated forces and stress using standard differentiation formulas
!
       ALLOCATE(SUM_FORCES(DOF,3,NIONS),SUM_SIF(DOF,3,3))
       SELECT CASE(NDISPL)
       CASE(1)
          DO N=1,DOF
             SUM_FORCES(N,:,:)=(DISPL_FORCES(1,N,:,:)-INITIAL_FORCE)/STEP
             SUM_SIF(N,:,:)   =-(DISPL_SIF(1,N,:,:)-INITIAL_SIF)/STEP
          END DO
       CASE(2)
          SUM_FORCES = (1._q/(2._q*STEP))*(DISPL_FORCES(1,:,:,:)-DISPL_FORCES(2,:,:,:))
          SUM_SIF    =-(1._q/(2._q*STEP))*(DISPL_SIF(1,:,:,:)   -DISPL_SIF(2,:,:,:))
       CASE(4)
          SUM_FORCES = (1._q/(12._q*STEP))* &
               (8._q*DISPL_FORCES(2,:,:,:)-8._q*DISPL_FORCES(3,:,:,:) &
               -DISPL_FORCES(1,:,:,:)+     DISPL_FORCES(4,:,:,:))
          SUM_SIF =-(1._q/(12._q*STEP))* &
               (8._q*DISPL_SIF(2,:,:,:)-8._q*DISPL_SIF(3,:,:,:) &
               -DISPL_SIF(1,:,:,:)+     DISPL_SIF(4,:,:,:))
       END SELECT

!
! elastic moduli and internal strain tensors
!
       IF (IU6>=0 .AND. NWRITE >=3) THEN
          DO N=1,DOF
             CALL FIND_IJ_ID(LSIF,NIONS,N,ND,J,I)
             WRITE (IU6,'("DOF:",I4," ATOM:",I4," AXIS:",I4," DISPLACEMENT:",I4)')N,J,I
             CALL PRINT_FORCE(NIONS,SUM_FORCES(N,:,:),IU6)
             CALL PRINT_STRESS(LSIF,SUM_SIF(N,:,:),IU6)
             WRITE (IU6,*)
          END DO
          WRITE (IU6,*) ' '
       END IF

       IF (LSIF .AND. IU6>=0) THEN
          FACT=EVTOJ*1E22_q/OMEGA

          WRITE(IU6,100) '  ELASTIC MODULI  (kBar)', ( &
              (SUM_SIF(J,I,I)*FACT,I=1,3), & 
               SUM_SIF(J,1,2)*FACT,SUM_SIF(J,2,3)*FACT,SUM_SIF(J,3,1)*FACT,J=1,6)
          ELASTIC(1,1,:,:)=SUM_SIF(1,:,:)
          ELASTIC(2,2,:,:)=SUM_SIF(2,:,:)
          ELASTIC(3,3,:,:)=SUM_SIF(3,:,:)
          ELASTIC(1,2,:,:)=SUM_SIF(4,:,:);  ELASTIC(2,1,:,:)=SUM_SIF(4,:,:)
          ELASTIC(2,3,:,:)=SUM_SIF(5,:,:);  ELASTIC(3,2,:,:)=SUM_SIF(5,:,:)
          ELASTIC(3,1,:,:)=SUM_SIF(6,:,:);  ELASTIC(1,3,:,:)=SUM_SIF(6,:,:)

          CALL TSYM4_HELPER(ELASTIC)
          IF (SYMM%ISYM>0) CALL TSYM4(ELASTIC,ISYMOP,NROTK,A)

          SUM_SIF(1,:,:)=ELASTIC(1,1,:,:)
          SUM_SIF(2,:,:)=ELASTIC(2,2,:,:)
          SUM_SIF(3,:,:)=ELASTIC(3,3,:,:)
          SUM_SIF(4,:,:)=ELASTIC(1,2,:,:)
          SUM_SIF(5,:,:)=ELASTIC(2,3,:,:)
          SUM_SIF(6,:,:)=ELASTIC(3,1,:,:)

          WRITE(IU6,100) '  SYMMETRIZED ELASTIC MODULI (kBar)', ( &
              (SUM_SIF(J,I,I)*FACT,I=1,3), & 
               SUM_SIF(J,1,2)*FACT,SUM_SIF(J,2,3)*FACT,SUM_SIF(J,3,1)*FACT,J=1,6)

160       FORMAT(/ & 
               ' INTERNAL STRAIN TENSOR FOR ION ',I4,' for displacements in x,y,z  (eV/Angst):',/ &
               10X,'X', 11X,'Y', 11X,'Z', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
               '  --------------------------------------------------------------------------------')
          WRITE(IU6,*) 'INTERNAL STRAIN TENSORS FROM STRAINED CELLS'
          WRITE(IU6,*) '============================================'
          DO N=1,NIONS
             WRITE(IU6,160) N
             DO I =1,3
                WRITE(IU6,140) IDIR_TEXT(I),(SUM_FORCES(M,I,N),M=1,6)
             ENDDO
          ENDDO
          WRITE(IU6,*)
140       FORMAT(2X,A1,6F12.5)
       ENDIF


!
! forces and force constants
!
       CALL PRINT_DYNMAT_ID(LSIF,NIONS,DOF,STEP,T_INFO%NTYP,T_INFO%NITYP,T_INFO%POMASS,STEP*SUM_FORCES,D,ND,IU6)

       DMAT=0
       ST=0

       DO M=1,DOF
          CALL FIND_IJ_ID(LSIF,NIONS,M,ND,I,J)
          IF (I>0) THEN
             DO N=1,NIONS
                DMAT(:,J,N,I)=SUM_FORCES(M,:,N)
             ENDDO
             ST(1:3,1:3,J,I)=-SUM_SIF(M,1:3,1:3)
          ENDIF
       END DO

       CALL MKDMAT(SYMM%ROTMAP,ISYMOP,INVMAP,NROTK,NPCELL,D,DMAT,ST,ND,      &
            1,T_INFO%NTYP,NIONS,T_INFO%NITYP,WORKD,WDMAT,AST,IWORK,SYMM%TAU,SYMM%TAUROT,   &
            SYMM%WRKROT,A(1,1),A(1,2),A(1,3),B(1,1),B(1,2),B(1,3))

       LDO=.TRUE.
       CALL SYDMAT(DMAT,SYMM%ROTMAP,ISYMOP,NROTK,NPCELL,1,T_INFO%NTYP, &
            NIONS,T_INFO%NITYP,WDMAT,A(1,1),A(1,2),A(1,3),LDO)
       CALL STMAT(ST   ,SYMM%ROTMAP,ISYMOP,NROTK,NPCELL,1,T_INFO%NTYP, &
            NIONS,T_INFO%NITYP,AST  ,A(1,1),A(1,2),A(1,3),LDO)

       DO N=1,NIONS
          DO M=1,NIONS
             SECOND_DERIV(1+3*(N-1),1+3*(M-1))=DMAT(1,1,N,M)
             SECOND_DERIV(2+3*(N-1),1+3*(M-1))=DMAT(2,1,N,M)
             SECOND_DERIV(3+3*(N-1),1+3*(M-1))=DMAT(3,1,N,M)
             SECOND_DERIV(1+3*(N-1),2+3*(M-1))=DMAT(1,2,N,M)
             SECOND_DERIV(2+3*(N-1),2+3*(M-1))=DMAT(2,2,N,M)
             SECOND_DERIV(3+3*(N-1),2+3*(M-1))=DMAT(3,2,N,M)
             SECOND_DERIV(1+3*(N-1),3+3*(M-1))=DMAT(1,3,N,M)
             SECOND_DERIV(2+3*(N-1),3+3*(M-1))=DMAT(2,3,N,M)
             SECOND_DERIV(3+3*(N-1),3+3*(M-1))=DMAT(3,3,N,M)
          END DO
       END DO

       IF (IU6>0) THEN
          WRITE(IU6,*) 'INTERNAL STRAIN TENSORS FROM DISPLACED ATOMS'
          WRITE(IU6,*) '============================================'
          DO N=1,NIONS
             WRITE(IU6,160) N
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(ST(J,J,I,N),J=1,3),ST(1,2,I,N),ST(2,3,I,N),ST(3,1,I,N)
             ENDDO
          ENDDO
       ENDIF

       IF(IU6>=0) WRITE(IU6,130)

       DOF=NIONS*3
! clumsy and unreadable output (write eigenvectors to vasprun.xml instead)
!       IF (IU6>=0) THEN
!          WRITE (IU6,*)  
!          WRITE (IU6,*) 'SECOND DERIVATIVES (NOT SYMMETRIZED)'
!          WRITE (IU6,*) '------------------------------------'
!          CALL PRINT_SECOND_DERIV(NIONS,DOF,SECOND_DERIV,T_INFO%LSFOR,T_INFO%LSDYN,IU6)
!       END IF

       DO N=1,DOF
          DO M=N+1,DOF
             X=0.5_q*(SECOND_DERIV(N,M)+SECOND_DERIV(M,N))
             SECOND_DERIV(N,M)=X
             SECOND_DERIV(M,N)=X
          END DO
       END DO

       ALLOCATE(WORK(DOF,32),EIGENVECTORS(DOF,DOF),EIGENVALUES(DOF))

       ! divide by mass  M^-1/2 Hessian M^-1/2
       EIGENVECTORS=SECOND_DERIV
       N=0
       DO I=1,T_INFO%NTYP
          DO J=1,T_INFO%NITYP(I)
             DO K=1,3
                EIGENVECTORS(:,3*N+K)=EIGENVECTORS(:,3*N+K)/SQRT(T_INFO%POMASS(I))
                EIGENVECTORS(3*N+K,:)=EIGENVECTORS(3*N+K,:)/SQRT(T_INFO%POMASS(I))
             END DO
             N=N+1
          END DO
       END DO

       CALL XML_TAG("dynmat")
       CALL XML_VECARRAY("hessian")
       CALL XML_ARRAY_REAL(EIGENVECTORS)
       CALL XML_CLOSE_TAG

       CALL DSYEV &
            ('V','U',DOF,EIGENVECTORS,DOF, &
            EIGENVALUES,WORK,32*DOF, IERROR)
       IF (IERROR/=0) THEN
          IF (IU6>=0) THEN
             WRITE(IU6,*) "Error while diagonalisation DSYEV INFO=",IERROR
             WRITE(IU6,*) "Some of (or all) eigenvectors and eigenvalues are not correct !"
          END IF
       END IF

       CALL XML_VEC_REAL(EIGENVALUES,"eigenvalues",'(ES16.8)')
       CALL XML_VECARRAY("eigenvectors")
       CALL XML_ARRAY_REAL(EIGENVECTORS,'(ES16.8)')
       CALL XML_CLOSE_TAG
! phonon_
       IF (GLOB_PRIM_CELL%LINIT) THEN
          NIONP=GLOB_PRIM_CELL%P_NUM
          CALL XML_FORCE_CONSTANTS(NIONS,NIONP,GLOB_PRIM_CELL%ATOM_IDX(1,1:NIONP),SECOND_DERIV)
       ENDIF
! phonon_
       CALL XML_CLOSE_TAG
       CALL PRINT_EIGENVECTORS_ID(NIONS,INITIAL_POSITIONS,A, &
            EIGENVECTORS,EIGENVALUES,IU6)

       IF (ISPECIAL<0) THEN
          CALL WRITE_DYNMATFULL(ISPECIAL, NIONS, INITIAL_POSITIONS, INITIAL_TOTEN, & 
               T_INFO, EIGENVECTORS, EIGENVALUES, IU6)
       ENDIF

       IF (ISPECIAL>0) THEN
          CALL GENERATE_EXCITED_STATE_POSCAR(ISPECIAL, TEBEG, NIONS,INITIAL_POSITIONS, & 
               T_INFO, A, B, EIGENVECTORS,EIGENVALUES,IU6)
       ENDIF
       
       ! calculate mass weighted eigenvectors
       N=1
       DO I=1,T_INFO%NTYP
          DO J=1,T_INFO%NITYP(I)
             DO K=1,3
                CALL FIND_DOF_INDEX(NIONS,T_INFO%LSFOR,T_INFO%LSDYN,K,N,M)
                IF (M>0) EIGENVECTORS(M,:)=EIGENVECTORS(M,:)/SQRT(T_INFO%POMASS(I))
             END DO
             N=N+1
          END DO
       END DO

       ! invert the matrix of the second derivatives (not mass weighted!)
       SECOND_DERIV=-SECOND_DERIV
       CALL INV_SECOND_DERIV(SECOND_DERIV, DOF, IU6 )

       IF ((LSIF .OR. LBORN) .AND. IU6>=0) THEN
          WRITE(IU6,130)

          ! ionic contribution to macroscopic dielectric tensor
          IF (LBORN) THEN
             CALL EPSILON_ION( T_INFO, DOF, SECOND_DERIV, BORN_CHARGES_PEAD, EPSILON )

             ! induced polariation -> field 
             EPSILON=EPSILON*EDEPS/OMEGA
             WRITE(IU6,1100) 'IONIC CONTRIBUTION',EPSILON

1100   FORMAT(// &
            " MACROSCOPIC STATIC DIELECTRIC TENSOR ",A/, &
            " -------------------------------------"/, &
            &       3(6X,3F10.3/), &
            " -------------------------------------"/)

             CALL XML_TENSOR("epsilon_ion",EPSILON)

             CALL EPSILON_ION_OMEGA( T_INFO, DOF, EIGENVECTORS, EIGENVALUES, BORN_CHARGES_PEAD, OMEGA )
          ENDIF

100       FORMAT(/ &
            A / &
            ' Direction', &
            4X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
            ' --------------------------------------------------------------------------------'/ &
            ' XX     ',6F12.4/ &
            ' YY     ',6F12.4/ &
            ' ZZ     ',6F12.4/ &
            ' XY     ',6F12.4/ &
            ' YZ     ',6F12.4/ &
            ' ZX     ',6F12.4/ &
            ' --------------------------------------------------------------------------------'/)
          CALL ELASTIC_ION( T_INFO, DOF, SECOND_DERIV, ST, ELASTIC )

          ELASTICP(1,:,:)=ELASTIC(1,1,:,:)
          ELASTICP(2,:,:)=ELASTIC(2,2,:,:)
          ELASTICP(3,:,:)=ELASTIC(3,3,:,:)
          ELASTICP(4,:,:)=ELASTIC(1,2,:,:)
          ELASTICP(5,:,:)=ELASTIC(2,3,:,:)
          ELASTICP(6,:,:)=ELASTIC(3,1,:,:)
          
          FACT=EVTOJ*1E22_q/OMEGA

          WRITE(IU6,100) ' ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)', ( &
               (ELASTICP(J,I,I)*FACT,I=1,3), &
               ELASTICP(J,1,2)*FACT,ELASTICP(J,2,3)*FACT,ELASTICP(J,3,1)*FACT,J=1,6)

          IF (LSIF) THEN
             ELASTICP=ELASTICP+SUM_SIF(1:6,:,:)
          
             WRITE(IU6,100) ' TOTAL ELASTIC MODULI (kBar)', ( &
                  (ELASTICP(J,I,I)*FACT,I=1,3), &
                  ELASTICP(J,1,2)*FACT,ELASTICP(J,2,3)*FACT,ELASTICP(J,3,1)*FACT,J=1,6)
          ENDIF

          ! ionic contribution to piezoelectric tensor
          IF (LBORN) THEN

             CALL PIEZO_ION( T_INFO, DOF, SECOND_DERIV, ST, BORN_CHARGES_PEAD, PIEZO )
             
             WRITE (IU6,180) 'PIEZOELECTRIC TENSOR IONIC CONTR'

180          FORMAT(/ ' ',A,'  for field in x, y, z        (C/m^2)',/ &
               &        10X,'XX', 10X,'YY', 10X,'ZZ', 10X,'XY', 10X,'YZ', 10X,'ZX'/ &
               &        '  ----------------------------------------------------', &
               &        '----------------------------')

             FACT=EVTOJ*1E20_q/OMEGA
             DO I =1,3
                WRITE (IU6,140) IDIR_TEXT(I),(PIEZO(I,J,J)*FACT,J=1,3), & 
                     PIEZO(I,1,2)*FACT,PIEZO(I,2,3)*FACT,PIEZO(I,3,1)*FACT
             ENDDO
          ENDIF
       ELSE IF (IU6>=0) THEN
          CALL ELASTIC_ION( T_INFO, DOF, SECOND_DERIV, ST, ELASTIC )

          ELASTICP(1,:,:)=ELASTIC(1,1,:,:)
          ELASTICP(2,:,:)=ELASTIC(2,2,:,:)
          ELASTICP(3,:,:)=ELASTIC(3,3,:,:)
          ELASTICP(4,:,:)=ELASTIC(1,2,:,:)
          ELASTICP(5,:,:)=ELASTIC(2,3,:,:)
          ELASTICP(6,:,:)=ELASTIC(3,1,:,:)
          
          FACT=EVTOJ*1E22_q/OMEGA

          WRITE(IU6,100) ' ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)', ( &
               (ELASTICP(J,I,I)*FACT,I=1,3), &
               ELASTICP(J,1,2)*FACT,ELASTICP(J,2,3)*FACT,ELASTICP(J,3,1)*FACT,J=1,6)
          
       END IF 

       DEALLOCATE(WORK,INITIAL_POSITIONS,INITIAL_FORCE,DISPL_FORCES,SUM_FORCES)
       LSTOP=.TRUE.

       IF(IU6>=0) WRITE(IU6,130)

    END IF

  END SUBROUTINE FINITE_DIFF_ID

!*********************************************************************
!
! helper routines
!
!*********************************************************************


  SUBROUTINE COUNT_DOF(NIONS, LSFOR, LSDYN, N)
    INTEGER :: NIONS,N
    LOGICAL :: LSFOR(3,NIONS)  ! selective 
    LOGICAL :: LSDYN
    ! local    
    INTEGER :: I,J

    IF (.NOT.LSDYN) THEN
      N=NIONS*3
      RETURN
    END IF

    N=0
    
    DO J=1,NIONS
      DO I=1,3
        IF (LSFOR(I,J)) N=N+1
      END DO
    END DO
  END SUBROUTINE COUNT_DOF

  ! Find the indexes of N-th degree of freedom
  SUBROUTINE FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
    INTEGER :: NIONS,N
    LOGICAL :: LSFOR(3,NIONS)
    LOGICAL :: LSDYN
    INTEGER :: I,J
    !local
    INTEGER :: M

    IF (.NOT.LSDYN) THEN
      I=1+MOD(N-1,3)
      J=1+(N-1)/3
      RETURN
    END IF

    M=0
ijloop: DO J=1,NIONS
      DO I=1,3
        IF (LSFOR(I,J)) M=M+1
        IF (M==N) EXIT ijloop
      END DO
    END DO ijloop
  END SUBROUTINE FIND_IJ

  SUBROUTINE FIND_IJ_ID(LSIF,NIONS,N,ND,II,JJ)
    LOGICAL :: LSIF
    INTEGER :: NIONS, N, NION
    INTEGER :: I,J,II,JJ,ND(NIONS)
    !local
    INTEGER :: M
    IF (LSIF) THEN
       NION=N-6
    ELSE
       NION=N
    ENDIF
    IF (NION<=0) THEN
       II=0
       JJ=0
    ELSE
       II=0
       JJ=0
       J=0
       DO I=1,NIONS
          J=J+ND(I)
          IF (J>=NION) THEN
             II=I
             JJ=NION-J+ND(I)
             RETURN
          END IF
       END DO
    ENDIF
  END SUBROUTINE FIND_IJ_ID

  SUBROUTINE FIND_DOF_INDEX(NIONS,LSFOR,LSDYN,AXE,N,DOFI)
    INTEGER     :: NIONS,N,AXE
    INTEGER     :: DOFI
    LOGICAL     :: LSFOR(3,NIONS)
    LOGICAL     :: LSDYN
    ! local    
    INTEGER :: I,J

    IF (.NOT.LSDYN) THEN
      DOFI=3*N+AXE-3
      RETURN
    END IF

    DOFI=0
    
    DO J=1,NIONS
      DO I=1,3
        IF (LSFOR(I,J)) THEN
          DOFI=DOFI+1
          IF ( (J==N).AND.(I==AXE) ) RETURN
        END IF
      END DO
    END DO
    DOFI=-1
    RETURN
  END SUBROUTINE FIND_DOF_INDEX

  SUBROUTINE MAKE_DISPLACED(NIONS,LSFOR,LSDYN,POS,N,STEP,A,B)
    USE lattice
    INTEGER :: NIONS,N
    REAL(q) :: POS(3,NIONS),STEP
    LOGICAL :: LSFOR(3,NIONS)
    LOGICAL :: LSDYN
    REAL(q) :: A(3,3)          ! lattice vectors
    REAL(q) :: B(3,3)          ! reciprocal lattice vectors

    !local
    INTEGER::I,J

    CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
    CALL DIRKAR(1,POS(:,J),A)
    POS(I,J) = POS(I,J)+STEP
    CALL KARDIR(1,POS(:,J),B)
  END SUBROUTINE MAKE_DISPLACED

  SUBROUTINE MAKE_DISPLACED_ID(LSIF,NIONS,POS,N,D,ND,STEP,A,B,OUT)
    USE lattice
    LOGICAL :: LSIF            ! strain considered ?
    INTEGER :: NIONS,N
    REAL(q) :: POS(3,NIONS),STEP
    REAL(q) :: A(3,3)          ! lattice vectors
    REAL(q) :: B(3,3)          ! reciprocal lattice vectors
    REAL(q) :: D(3,3,NIONS)
    INTEGER :: ND(NIONS),OUT
    !local
    INTEGER :: ADISPL,IDISPL
    REAL(q) :: ONEM(3,3)
    DATA       ONEM /1,0,0, 0,1,0, 0,0,1/
    REAL(q) :: DISTORTION(3,3,6)
    DATA       DISTORTION / &
          1,0,0, 0,0,0, 0,0,0, &
          0,0,0, 0,1,0, 0,0,0, &
          0,0,0, 0,0,0, 0,0,1, &
          0,1,0, 0,0,0, 0,0,0, &  ! x->y
          0,0,0, 0,0,1, 0,0,0, &  ! y->z
          0,0,0, 0,0,0, 1,0,0 /   ! z->x

    CALL FIND_IJ_ID(LSIF,NIONS,N,ND,ADISPL,IDISPL)
    
    IF (ADISPL>0) THEN
       IF (IDISPL==0) THEN
          WRITE(OUT,*) 'internal error in subroutine MAKE_DISPLACED_ID: IDISPL not determined'
          STOP
       END IF
       CALL DIRKAR(1,POS(:,ADISPL),A)
       POS(:,ADISPL) = POS(:,ADISPL)+STEP*D(:,IDISPL,ADISPL)
       CALL KARDIR(1,POS(:,ADISPL),B)
    ELSE
       A=MATMUL(ONEM+STEP*DISTORTION(:,:,N),A)
    ENDIF
    RETURN
  END SUBROUTINE MAKE_DISPLACED_ID

  SUBROUTINE PRINT_POSITIONS(NIONS,POS,OUT,A,B)
    USE lattice
    INTEGER :: NIONS,OUT
    REAL(q) :: POS(3,NIONS)
    REAL(q) :: A(3,3),B(3,3)
    !local
    INTEGER I
    REAL(q) :: X(3)
    IF (OUT>=0) THEN
      WRITE (OUT,*) 'Positions of atoms (Carthesian coordinates)'
      WRITE (OUT,*) '-------------------------------------------'
      DO I=1,NIONS
        X=POS(:,I)
        CALL DIRKAR(1,X,A)
        WRITE(OUT,'(3F11.7)') X
      END DO
    END IF
  END SUBROUTINE PRINT_POSITIONS

  SUBROUTINE PRINT_FORCE(NIONS,FORCE,OUT)
    USE lattice
    INTEGER :: NIONS,OUT
    REAL(q) :: FORCE(3,NIONS)
    !local
    INTEGER I
    IF (OUT>=0) THEN
      DO I=1,NIONS
        WRITE(OUT,'(F10.6," ",F10.6," ",F10.6)') FORCE(:,I)
      END DO
    END IF
  END SUBROUTINE PRINT_FORCE

  SUBROUTINE PRINT_STRESS(LSIF,SIF,OUT)
    USE lattice
    LOGICAL :: LSIF
    INTEGER :: NIONS,OUT
    REAL(q) :: SIF(3,3)
    !local
    INTEGER I
    IF (OUT>=0 .AND. LSIF) THEN
      WRITE(OUT,*)
      DO I=1,3
        WRITE(OUT,'(F10.6," ",F10.6," ",F10.6)') SIF(:,I)
      END DO
    END IF
  END SUBROUTINE PRINT_STRESS

  SUBROUTINE PRINT_DYNMAT(NIONS,DOF,STEP,NTYP,NITYP,MASSES,FORCES,LSDYN,LSFOR,OUT)
    USE lattice
    USE main_mpi
    INTEGER :: NIONS,OUT,DOF,NTYP
    INTEGER :: NITYP(NTYP)
    REAL(q) :: MASSES(NTYP)
    REAL(q) :: STEP
    REAL(q) :: FORCES(DOF,3,NIONS)
    LOGICAL :: LSFOR(3,NIONS)
    LOGICAL :: LSDYN    
    !local
    INTEGER :: N,I,J,OLDJ,DISPL
    INTEGER :: IU=55

    IF (OUT>=0) THEN
       OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'DYNMAT', &
              FORM='FORMATTED',STATUS='UNKNOWN')

       WRITE(IU,'(I5)',ADVANCE='NO') NTYP
       WRITE(IU,'(I5)',ADVANCE='NO') SUM(NITYP)
       WRITE(IU,'(I5)') DOF
       DO N=1,NTYP
          WRITE(IU,'(F8.3)',ADVANCE='NO') MASSES(N) 
       END DO
       WRITE(IU,*)

       OLDJ  = 0
       DISPL = 0
       DO N=1,DOF
          CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
          IF (J==OLDJ) THEN
            DISPL=DISPL+1
          ELSE
            DISPL=1
            OLDJ=J
          ENDIF
          WRITE (IU,'(I5,I5)',ADVANCE='NO') J,DISPL
          SELECT CASE(I)
          CASE(1)
             WRITE (IU,'(F8.4,F8.4,F8.4)') STEP,0._q,0._q
          CASE(2)
             WRITE (IU,'(F8.4,F8.4,F8.4)') 0._q,STEP,0._q
          CASE(3)
             WRITE (IU,'(F8.4,F8.4,F8.4)') 0._q,0._q,STEP
          CASE DEFAULT
             WRITE (IU,*) "?"
          END SELECT                            
          CALL PRINT_FORCE(NIONS,FORCES(N,:,:),IU)
       END DO
       CLOSE(IU)
    END IF
  END SUBROUTINE PRINT_DYNMAT

  SUBROUTINE PRINT_DYNMAT_ID(LSIF,NIONS,DOF,STEP,NTYP,NITYP,MASSES,FORCES,D,ND,OUT)
    USE lattice
    USE main_mpi
    LOGICAL :: LSIF
    INTEGER :: NIONS,OUT,DOF,NTYP
    INTEGER :: NITYP(NTYP)
    REAL(q) :: MASSES(NTYP)
    REAL(q) :: STEP
    REAL(q) :: FORCES(DOF,3,NIONS)
    REAL(q) :: D(3,3,NIONS)
    !local
    INTEGER :: N,I,DISPL,ND(NIONS)
    INTEGER :: IU=55

    IF (OUT>=0) THEN
       OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'DYNMAT', &
              FORM='FORMATTED',STATUS='UNKNOWN')

       WRITE(IU,'(I5)',ADVANCE='NO') NTYP
       WRITE(IU,'(I5)',ADVANCE='NO') SUM(NITYP)
       WRITE(IU,'(I5)') DOF
       DO N=1,NTYP
          WRITE(IU,'(F7.3)',ADVANCE='NO') MASSES(N) 
       END DO
       WRITE(IU,*)

       DO N=1,DOF
          CALL FIND_IJ_ID(LSIF,NIONS,N,ND,I,DISPL)
          IF (I/=0) THEN
             WRITE (IU,'(I5,I5)',ADVANCE='NO') I,DISPL
             WRITE (IU,'(F8.4,F8.4,F8.4)') STEP*D(1,DISPL,I),STEP*D(2,DISPL,I),STEP*D(3,DISPL,I)
             CALL PRINT_FORCE(NIONS,FORCES(N,:,:),IU)
          ENDIF
       END DO
       CLOSE(IU)
    END IF
  END SUBROUTINE PRINT_DYNMAT_ID

  SUBROUTINE PRINT_SECOND_DERIV(NIONS,DOF,SD,LSFOR,LSDYN,OUT)
    INTEGER::NIONS,DOF,OUT
    LOGICAL::LSFOR(3,NIONS)
    LOGICAL::LSDYN
    REAL(q)::SD(DOF,DOF)
    !local
    INTEGER   :: I,J,K,M,N
    CHARACTER :: C

    IF (OUT>=0) THEN
      WRITE(OUT,'(A)',ADVANCE='NO') "      "
      DO N=1,DOF
        CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
        SELECT CASE(I)
        CASE(1)
           C="X"
        CASE(2)
           C="Y"
        CASE(3)
           C="Z"
        CASE DEFAULT
           C="?"
        END SELECT
        WRITE(OUT,'(I10,A,A)',ADVANCE='NO') J,C," "
      END DO
      WRITE(OUT,*)

      DO N=1,DOF
        CALL FIND_IJ(NIONS,LSFOR,LSDYN,N,I,J)
        SELECT CASE(I)
        CASE(1)
           C="X"
        CASE(2)
           C="Y"
        CASE(3)
           C="Z"
        CASE DEFAULT
           C="?"
        END SELECT
        WRITE(OUT,'(I3,A,A)',ADVANCE='NO') J,C," "
     
        DO M=1,DOF
          WRITE(OUT,'(F12.6)',ADVANCE='NO') SD(N,M)
        END DO
        WRITE(OUT,*)
      END DO
      WRITE(OUT,*)
    END IF
  END SUBROUTINE PRINT_SECOND_DERIV

  SUBROUTINE PRINT_EIGENVECTORS_ID (NIONS,POS,A,EVEC,EVAL,OUT)
    USE constant
    USE lattice
    INTEGER::NIONS,OUT
    REAL(q)::EVEC(3*NIONS,3*NIONS),EVAL(3*NIONS)
    REAL(q)::POS(3,NIONS)
    REAL(q)::A(3,3)
    !local
    INTEGER   :: N,NI,AXE,DI
    REAL(q),PARAMETER :: PLANK=6.626075E-34
    REAL(q),PARAMETER :: C= 2.99792458E10
    REAL(q)::X(3)
    REAL(q)   :: FACTOR,W
                                               
!---- frequenz Sqrt(d E / M) d x (cgs System)
    FACTOR=SQRT(EVTOJ/((1E-10)**2)/AMTOKG)

    IF (OUT>=0) THEN
      WRITE(OUT,*)
      WRITE(OUT,*)'Eigenvectors and eigenvalues of the dynamical matrix'
      WRITE(OUT,*)'----------------------------------------------------'
      WRITE(OUT,*)
      DO N=1,3*NIONS
        WRITE(OUT,*)
        W=FACTOR*SQRT(ABS(EVAL(N)))
        IF (EVAL(N).GT.0) THEN
          WRITE(OUT,'(I4," f/i=",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
               N,W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
        ELSE
          WRITE(OUT,'(I4," f  =",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
               N,W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
        END IF
        WRITE(OUT,'("             X         Y         Z","           dx          dy          dz")')
        DO NI=1,NIONS
          X=POS(:,NI)
          CALL  DIRKAR(1,X,A)
          WRITE (OUT,'(A,3F10.6,A)',ADVANCE='NO')'    ',X,'   '
          DO AXE=1,3
            WRITE(OUT,'(F10.6,A)',ADVANCE='NO') EVEC((NI-1)*3+AXE,N),'  '
          END DO
          WRITE(OUT,*)
        END DO
      END DO
      WRITE(OUT,*)
    END IF
  END SUBROUTINE PRINT_EIGENVECTORS_ID

  SUBROUTINE PRINT_EIGENVECTORS (NIONS,DOF,POS,A,EVEC,EVAL,LSFOR,LSDYN,OUT)
    USE constant
    USE lattice
    INTEGER::NIONS,DOF,OUT
    LOGICAL::LSFOR(3,NIONS)
    LOGICAL::LSDYN
    REAL(q)::EVEC(DOF,DOF),EVAL(DOF)
    REAL(q)::POS(3,NIONS)
    REAL(q)::A(3,3)
    !local
    INTEGER   :: N,NI,AXE,DI
    REAL(q),PARAMETER :: PLANK=6.626075E-34
    REAL(q),PARAMETER :: C= 2.99792458E10
    REAL(q)::X(3)
!    REAL(q)   :: ELECT  = 1.602199E-19
!    REAL(q)   :: M0     = 1.6725E-27
    REAL(q)   :: FACTOR,W
                                               
!---- frequenz Sqrt(d E / M) d x (cgs System)
    FACTOR=SQRT(EVTOJ/((1E-10)**2)/AMTOKG)

    IF (OUT>=0) THEN
      WRITE(OUT,*)
      WRITE(OUT,*)'Eigenvectors and eigenvalues of the dynamical matrix'
      WRITE(OUT,*)'----------------------------------------------------'
      WRITE(OUT,*)
      DO N=1,DOF
        WRITE(OUT,*)
        W=FACTOR*SQRT(ABS(EVAL(N)))
        IF (EVAL(N).GT.0) THEN
          WRITE(OUT,'(I4," f/i=",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
               N,W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
        ELSE
          WRITE(OUT,'(I4," f  =",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
               N,W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
        END IF
        WRITE(OUT,'("             X         Y         Z","           dx          dy          dz")')
        DO NI=1,NIONS
          X=POS(:,NI)
          CALL  DIRKAR(1,X,A)
          WRITE (OUT,'(A,3F10.6,A)',ADVANCE='NO')'    ',X,'   '
          DO AXE=1,3
            CALL FIND_DOF_INDEX(NIONS,LSFOR,LSDYN,AXE,NI,DI)
            IF (DI>0) THEN
              WRITE(OUT,'(F10.6,A)',ADVANCE='NO') EVEC(DI,N),'  '
            ELSE
              WRITE(OUT,'(A)',ADVANCE='NO')'         0  '
            END IF
          END DO
          WRITE(OUT,*)
        END DO
      END DO
      WRITE(OUT,*)
    END IF
  END SUBROUTINE PRINT_EIGENVECTORS

!************************ SUBROUTINE TSYM4_HELPER **********************
!
! the elastic moduli must be symmetric with respect to interchange 
! of coordinates 
!
!***********************************************************************
  SUBROUTINE TSYM4_HELPER(ELASTIC)

    REAL(q) :: ELASTIC(:,:,:,:)
    REAL(q) :: TMP
    INTEGER I1, I2, J1, J2

    DO I1=1,3
       DO I2=1,3
          DO J1=1,3
             DO J2=1,3
                TMP=(ELASTIC(I1,I2,J1,J2)+ELASTIC(J1,J2,I1,I2))*0.5_q
                ELASTIC(I1,I2,J1,J2)=TMP
                ELASTIC(J1,J2,I1,I2)=TMP
             ENDDO
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE TSYM4_HELPER



!************************ SUBROUTINE INV_SECOND_DERIV ******************
!
! this subroutine calculates a matrix to the power -1
! but kills the very low frequency components
!
!***********************************************************************

  SUBROUTINE INV_SECOND_DERIV(CUNI, DOF, IU6 )
    INTEGER DOF
    REAL(q) CUNI(:,:)
    INTEGER ::  IU6
    ! local
    REAL(q) :: HFEIG(DOF),W(3*DOF)
    INTEGER :: N1, N2, IFAIL, NDIM
    
    REAL(q), ALLOCATABLE ::  CTMP(:,:), CEIDB(:,:)
    
    NDIM = SIZE(CUNI,1)
    ALLOCATE(CTMP(DOF,DOF),CEIDB(DOF,DOF))
!=======================================================================
! diagononalize the resulting matrix
!=======================================================================
    CALL DSYEV &
            ('V','U',DOF,CUNI(1,1),NDIM, &
            HFEIG,CTMP,DOF*DOF, IFAIL)

    IF (IFAIL/=0) THEN
       WRITE(*,*) 'internal ERROR in INV_SECOND_DERIV Call to routine DSYEV failed! '// &
            &              'Error code was ',IFAIL
       STOP
    ENDIF
!=======================================================================
! set up the unitary transformation CEIDB divided by eigenvalues
! skip lowest three eigenvalues which should correspond to 
! translational modes
! also skip soft modes close to zero
!=======================================================================
    CTMP=0
    DO N2=4,DOF
       IF (HFEIG(N2)>1E-3) THEN
          CTMP(:,N2)=CUNI(:,N2)/HFEIG(N2)
       ENDIF
    ENDDO

    CALL DGEMM( 'N',trans, DOF, DOF, DOF, one, CTMP, &
         &             DOF, CUNI(1,1), NDIM, zero, CEIDB, DOF)
   
    CUNI=0
    CUNI(1:DOF,1:DOF)=CEIDB(1:DOF,1:DOF)

    DEALLOCATE(CTMP, CEIDB)

  END SUBROUTINE INV_SECOND_DERIV

!************************ SUBROUTINE ELASTIC_ION ***********************
!
! calculate the ionic contribution the elastic  tensor
!
!***********************************************************************
  
  SUBROUTINE ELASTIC_ION( T_INFO, DOF, INV_SECOND_DERIV, ST, ELASTIC )
    USE poscar
    INTEGER DOF
    TYPE (type_info)   T_INFO
    REAL(q) :: INV_SECOND_DERIV(:,:)
    REAL(q) :: ST(:,:,:,:)
    REAL(q) :: ELASTIC(3,3,3,3)
    INTEGER :: IDIR, I, IDIRP, IP, N, NP
    INTEGER :: ALPHA, BETA, ALPHAP, BETAP

    ELASTIC=0

    DO N=1,DOF
       CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, N, IDIR, I)
       DO NP=1,DOF
          CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, NP, IDIRP, IP)
          DO ALPHA=1,3
          DO BETA=1,3
             DO ALPHAP=1,3
             DO BETAP=1,3
                ELASTIC(ALPHA,BETA,ALPHAP,BETAP)=ELASTIC(ALPHA,BETA,ALPHAP,BETAP) & 
                     -INV_SECOND_DERIV(N,NP)* & 
                     ST(ALPHA,BETA,IDIR,I)*ST(ALPHAP,BETAP,IDIRP,IP)
             END DO
             END DO
          END DO
          END DO
       END DO
    END DO
  END SUBROUTINE ELASTIC_ION


!************************ SUBROUTINE EPSILON_ION ***********************
!
! calculate the ionic contribution the dielectric tensor
! see. e.g. Wu, Vanderbilt, Hamann, PRB 72, 035105 (2005)
!
! I_SECOND_DERIV  holds the inverse of the second derivative of
!                 the energy
!   H = d^2 E / (d R_i d R_j)
! not mass weighted
!
!***********************************************************************
  
  SUBROUTINE EPSILON_ION( T_INFO, DOF, I_SECOND_DERIV, BORN_CHARGES, EPSILON )
    USE poscar
    INTEGER DOF
    TYPE (type_info)   T_INFO
    REAL(q) :: I_SECOND_DERIV(:,:)
    REAL(q) :: BORN_CHARGES(:,:,:)
    REAL(q) :: EPSILON(3,3)
    INTEGER :: N, NP, IDIR, I, IDIRP, IP
    INTEGER :: ALPHA, BETA

    EPSILON=0
    
    ! loop over eigenvectors
    DO N=1,DOF
       CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, N, IDIR, I)
       DO NP=1,DOF
          CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, NP, IDIRP, IP)
       ! loop over components of eigenvector NP
          DO ALPHA=1,3
          DO BETA=1,3
             EPSILON(ALPHA,BETA)=EPSILON(ALPHA,BETA)+I_SECOND_DERIV(N,NP)* & 
                  BORN_CHARGES(ALPHA,IDIR,I)*BORN_CHARGES(BETA,IDIRP,IP)
          END DO
          END DO
       END DO
       ! possibly dump OSZI_STRENGTH
    END DO
  END SUBROUTINE EPSILON_ION

!************************ SUBROUTINE EPSILON_ION_OMEGA *****************
!
! calculate the frequency dependent dielectric function
! this was implemented somewhat ad-hoc, but given the phonon propagator
! must by
!  1/(w-w_i) + 1/(-w-w-i)
! and the polarization is given by Z e 
! it seems the logical choice the factor 0.5 is somewhat odd, though
!
!***********************************************************************

  SUBROUTINE EPSILON_ION_OMEGA( T_INFO, DOF, EIGENVECTORS, EIGENVALUES, BORN_CHARGES, VOLUME )
    USE poscar
    USE constant
    INTEGER DOF
    TYPE (type_info)   T_INFO
    REAL(q) :: EIGENVECTORS(:,:)
    REAL(q) :: EIGENVALUES(:)
    REAL(q) :: BORN_CHARGES(:,:,:)
    REAL(q) :: VOLUME
  ! local
    INTEGER :: NO, NP, N, IDIR, NI, I
    INTEGER :: ALPHA
    REAL(q) :: POL(3), POL2(3,3), DOMEGA, OMEGA, SHIFT
    INTEGER, PARAMETER :: NOMEGA=2000
    COMPLEX(q) :: CT
    REAL(q) :: EPS(NOMEGA,3,3), EPS_IMAG(NOMEGA,3,3), E, FACTOR

    DOMEGA=SQRT(MAXVAL(-EIGENVALUES))*1.2/(NOMEGA-1)
    SHIFT =DOMEGA*2
    ! loop over eigenvectors
    EPS=0
    EPS_IMAG=0
    DO NP=1,DOF
       IF (-EIGENVALUES(NP)>1.0E-3_q) THEN
       E=SQRT(ABS(EIGENVALUES(NP)))

       ! loop over components of eigenvector NP
       POL=0
       DO N=1,DOF
          CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, N, IDIR, NI)
          POL(:)=POL(:)+EIGENVECTORS(N,NP)*BORN_CHARGES(:,IDIR,NI)/SQRT(E)
       END DO
       DO I=1,3
          POL2(:,I)=POL(:)*POL(I)
       ENDDO
       ! now add to ionic dielectric function
       DO NO=1,NOMEGA
          OMEGA=DOMEGA*(NO-1)

          CT=-0.5_q/(OMEGA-E+CMPLX(0,SHIFT,q))-0.5_q/(-OMEGA-E-CMPLX(0,SHIFT,q))
          CT=CT*EDEPS/VOLUME
          EPS(NO,:,:)=EPS(NO,:,:)+POL2*REAL(CT,q)
          EPS_IMAG(NO,:,:)=EPS_IMAG(NO,:,:)+POL2*AIMAG(CT)
       ENDDO
      ELSE
      ENDIF
    ENDDO

    ! scale to circular frequency THZ upon writing
    FACTOR=SQRT(EVTOJ/((1E-10)**2)/AMTOKG)
    DOMEGA=DOMEGA*FACTOR/1E12

    CALL XML_EPSILON_W(DOMEGA, EPS, EPS_IMAG, NOMEGA )

  END SUBROUTINE EPSILON_ION_OMEGA


!************************ SUBROUTINE  PIEZO_ION ************************
!
! calculate the ionic contribution the piezoelectric tensor
!
!***********************************************************************
  
  SUBROUTINE PIEZO_ION( T_INFO, DOF, I_SECOND_DERIV, ST, BORN_CHARGES, PIEZO )
    USE poscar
    INTEGER DOF
    TYPE (type_info)   T_INFO
    REAL(q) :: I_SECOND_DERIV(:,:)
    REAL(q) :: ST(:,:,:,:)
    REAL(q) :: BORN_CHARGES(:,:,:)
    REAL(q) :: PIEZO(3,3,3)
    INTEGER :: N, NP, IDIR, I, IDIRP, IP
    INTEGER :: ALPHA, ALPHAP, BETAP

    PIEZO=0

    DO N=1,DOF
       CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, N, IDIR, I)
       DO NP=1,DOF
          CALL FIND_IJ(T_INFO%NIONS, T_INFO%LSFOR, T_INFO%LSDYN, NP, IDIRP, IP)
          DO ALPHA=1,3
             DO ALPHAP=1,3
             DO BETAP=1,3
                PIEZO(ALPHA,ALPHAP,BETAP)=PIEZO(ALPHA,ALPHAP,BETAP)+I_SECOND_DERIV(N,NP)* & 
                     BORN_CHARGES(ALPHA,IDIR,I)*ST(ALPHAP,BETAP,IDIRP,IP)
             END DO
             END DO
          END DO
       END DO
    END DO
  END SUBROUTINE PIEZO_ION


!*********************************************************************
!
! write POSCAR files corresponding to thermodynamically "excited" 
! states at a specified temperature
! this is selected if ISPECIAL>0
! ISPECIAL also specifies the number of configurations that are written
! the temperature for which the configurations are written
! is specified by TEBEG (in the INCAR)
!
!*********************************************************************

  SUBROUTINE GENERATE_EXCITED_STATE_POSCAR(ISPECIAL,TEBEG, NIONS, POS, & 
       T_INFO, A, B, EVEC, EVAL, OUT)
    USE constant
    USE lattice
    USE poscar

    INTEGER:: ISPECIAL
    REAL(q):: TEBEG 
    INTEGER:: NIONS,OUT
    REAL(q):: EVEC(3*NIONS,3*NIONS),EVAL(3*NIONS)
    TYPE (type_info) :: T_INFO
    REAL(q):: POS(3,NIONS)
    REAL(q):: A(3,3)
    REAL(q):: B(3,3)
                                               
!---- frequenz Sqrt(d E / M) d x (cgs System)
    INTEGER   :: NPOS, NI, MODE, MODES, L1, L2
    REAL(q),PARAMETER :: PLANK=6.626075E-34
    REAL(q)   :: FACTOR, EMODE, BMP, E, EMEAN, OCC
    REAL(q), EXTERNAL :: RANG
    REAL(q)   :: X(3,NIONS)
    CHARACTER (40) :: APP
    CHARACTER (6) :: TEMP
    INTEGER :: IU=72
    
    ! to energy eV 

    EMEAN=0
    ! loop over normal modes but exclude translational modes
    MODES=3*(NIONS-1)

    ! loop over configurations
    DO NPOS=1,ISPECIAL
       X=POS
       CALL  DIRKAR(NIONS, X, A)

       E=0
       DO MODE=1,MODES
       ! units of dynamic matrix [eV/Angst^2/au]
          EMODE=ABS(EVAL(MODE))
       ! zero or imaginary frequency, forget it
          IF (EVAL(MODE)>=0) CYCLE
       ! determine random number distributed according to Gaussian distribution
          BMP=SQRT(BOLKEV*TEBEG/(EMODE))
       ! OCC has now the unit Angst/sqrt(au)
          OCC=RANG(0.0_q,BMP)
       ! energy corresponding to this mode (assuming quadratic displacement dependence)
       ! (only half is the energy, rest kinetic)
          E=E+OCC*OCC*EMODE/2

       ! solved equation is D' Y = Y <=>  M^-1/2 D  M^-1/2 M^1/2  X =  M^1/2  X 
       ! X = M^-1/2 Y -> 1/SQRT(POMASS)
          DO NI=1,NIONS
             X(:,NI)=X(:,NI)+EVEC((NI-1)*3+1:(NI-1)*3+3,MODE)*OCC/SQRT(T_INFO%POMASS(T_INFO%ITYP(NI)))
          ENDDO
       ENDDO
       EMEAN=EMEAN+E

       IF (OUT>=0) THEN
          WRITE (APP  , "(I6)") NPOS
          CALL STRIP(APP,L1,"L")

          WRITE (TEMP , "(F6.0)") TEBEG
          CALL STRIP(TEMP,L2,"L")

          OPEN(IU,FILE='POSCAR.T='//TEMP(1:L2)//APP(1:L1))
       
          WRITE(APP  , "('energy',F20.10)") E
          CALL  KARDIR(NIONS, X, B)
          CALL OUTPOS(IU, .TRUE. ,APP, T_INFO, 1.0_q, A, .FALSE., X)
          CLOSE(IU)
       ENDIF

    ENDDO
    IF (OUT>=0) THEN
       WRITE(OUT,*)
       WRITE(OUT,*) 'generated POSCAR with finite random displacements corresponding to TEBEG'
       WRITE(OUT,*) 'all normal modes except translational modes and imaginary modes where used'
       
       WRITE(OUT,'(" average potential energy",F12.5,"    k_b T x modes/2 =",F12.5)') & 
            EMEAN/ISPECIAL,BOLKEV*TEBEG*MODES*0.5
       WRITE(OUT,*)
    ENDIF

  END SUBROUTINE GENERATE_EXCITED_STATE_POSCAR


!*********************************************************************
!
! write the eigenvectors eigenvalues and initial positions
! to a file DYNMATFULL
!
!*********************************************************************

  SUBROUTINE WRITE_DYNMATFULL(ISPECIAL, NIONS, POS, TOTEN, & 
       T_INFO, EVEC, EVAL, OUT)
    USE constant
    USE lattice
    USE poscar
    USE main_mpi

    INTEGER:: ISPECIAL
    INTEGER:: NIONS
    REAL(q):: POS(3,NIONS)
    REAL(q):: TOTEN
    TYPE (type_info) :: T_INFO
    REAL(q):: EVEC(3*NIONS,3*NIONS),EVAL(3*NIONS)
    INTEGER :: OUT   ! unit for IO (usually OUTCAR)
! local
    INTEGER :: N
    INTEGER :: IU=72
    INTEGER :: MODE
    
    IF (OUT>=0) THEN
       OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'DYNMATFULL', &
              FORM='FORMATTED',STATUS='UNKNOWN')

       WRITE(IU,'("DYNMAT version 1.0")')
       WRITE(IU,'(I6,"   #number of species")') T_INFO%NTYP
       WRITE(IU,'(100I6)') T_INFO%NITYP
       WRITE(IU,'(100F8.3)') T_INFO%POMASS
       WRITE(IU,'(E18.10,"   #total energy for zero displacement")') TOTEN

       WRITE(IU,'("original positions")') 
       WRITE(IU,'(20F20.16)') POS
  
       DO MODE=1,3*NIONS
          WRITE(IU,'("mode",I6," eigenvalue ",E18.10)') MODE,EVAL(MODE)
          WRITE(IU,'(20F12.8)') EVEC(:,MODE)
       ENDDO
       CLOSE(IU)
    ENDIF

  END SUBROUTINE WRITE_DYNMATFULL
      
!*********************************************************************
!
! read the eigenvectors eigenvalues and initial positions
! from the file DYNMATFULL
!
!*********************************************************************

  SUBROUTINE READ_DYNMATFULL( NIONS, OUT)
    USE constant
    USE lattice
    USE poscar
    USE main_mpi
    IMPLICIT NONE

    INTEGER :: OUT   ! unit for IO (usually OUTCAR)
    INTEGER :: NIONS
! loca
    INTEGER :: N
    INTEGER :: IU=72
    INTEGER :: MODE, MODE2
    INTEGER :: IERR
    CHARACTER(LEN=10) :: STR1, STR2
    LOGICAL,SAVE :: READ_DYNMATFULL_TRIED=.FALSE.
    REAL(q) :: version
! structure

    ! if we have already tried to read the file quick return
    IF (READ_DYNMATFULL_TRIED) RETURN

    ! so now we will try to read once and never again
    READ_DYNMATFULL_TRIED=.TRUE.
    NULLIFY (DYNMH)

    OPEN(UNIT=IU,FILE=DIR_APP(1:DIR_LEN)//'DYNMATFULL', &
              FORM='FORMATTED',STATUS='OLD',IOSTAT=IERR )

    IF (IERR/=0) RETURN ! return with nullified DYNMH

    WRITE(OUT,*) 'reading now DYNMATFULL'

    ALLOCATE(DYNMH)
    
    READ(IU,*, IOSTAT=IERR) STR1, STR2, VERSION
    IF (IERR/=0 .OR. VERSION/=1.0) THEN
       CALL DEALLOCATE_DYNMATFULL
       RETURN
    ENDIF

    READ(IU,*,IOSTAT=IERR) DYNMH%NTYP
    ! WRITE(*,*) IERR, DYNMH%NTYP

    ! if NTYP can not be read then stop
    IF (IERR/=0) THEN
       CALL DEALLOCATE_DYNMATFULL
       RETURN
    ENDIF

    ALLOCATE(DYNMH%NITYP(DYNMH%NTYP), DYNMH%POMASS(DYNMH%NTYP))
    READ(IU,*,IOSTAT=IERR) DYNMH%NITYP
    READ(IU,*,IOSTAT=IERR) DYNMH%POMASS
    READ(IU,*,IOSTAT=IERR) DYNMH%TOTEN
    ! WRITE(*,*) IERR, DYNMH%NITYP, DYNMH%POMASS, DYNMH%TOTEN

    IF (IERR/=0) THEN
       CALL DEALLOCATE_DYNMATFULL
       RETURN
    ENDIF

    DYNMH%NIONS=SUM(DYNMH%NITYP)
    IF (DYNMH%NIONS /= NIONS) THEN
       CALL DEALLOCATE_DYNMATFULL
       IF (OUT>=0) WRITE(OUT,*)'stop reading DYNMATFULL incompatible ions',DYNMH%NIONS, NIONS
       RETURN
    ENDIF

    ALLOCATE(DYNMH%POS(3,DYNMH%NIONS),DYNMH%EVAL(3*DYNMH%NIONS), & 
         DYNMH%EVEC(3,DYNMH%NIONS,3*DYNMH%NIONS))
    
    READ(IU,*) ! skip original positions
    READ(IU,*) DYNMH%POS

    DO MODE=1,3*DYNMH%NIONS
       READ(IU,*) STR1,MODE2,STR2,DYNMH%EVAL(MODE)
!  WRITE(*,*) MODE,MODE2,DYNMH%EVAL(MODE)
       READ(IU,*) DYNMH%EVEC(:,:,MODE)
    ENDDO
    CLOSE(IU)

    IF (IERR==0) THEN
       WRITE(OUT,*) 'the DYNMATFULL file was read sucessfully'
    ENDIF


  END SUBROUTINE READ_DYNMATFULL


  SUBROUTINE DEALLOCATE_DYNMATFULL
    IMPLICIT NONE
    IF (ASSOCIATED(DYNMH)) THEN
       IF (ALLOCATED(DYNMH%NITYP)) DEALLOCATE(DYNMH%NITYP) 
       IF (ALLOCATED(DYNMH%POMASS)) DEALLOCATE(DYNMH%POMASS) 
       IF (ALLOCATED(DYNMH%POS)) DEALLOCATE(DYNMH%POS, DYNMH%EVAL, DYNMH%EVEC)
       DEALLOCATE(DYNMH)
       NULLIFY(DYNMH)
    ENDIF
    
  END SUBROUTINE DEALLOCATE_DYNMATFULL
  
END MODULE finite_differences


!*********************************************************************
!
! this subroutine adds forces corresponding to the Hessian
! matrix read from the DYNMATFULL file
! it is only invoked under rather special cases
! ) SCALEE must be set to a value different from 1
!   
!
!*********************************************************************


  SUBROUTINE DYNMATFULL_ENERGY_FORCE(SCALEE, NIONS, POS, ENERGY, FORCE, A , OUT)
    USE finite_differences
    USE lattice
    IMPLICIT NONE
    REAL(q) :: SCALEE
    INTEGER :: NIONS
    REAL(q) :: POS(3,NIONS)
    REAL(q) :: ENERGY
    REAL(q) :: FORCE(3,NIONS)
    REAL(q) :: A(3,3)
    INTEGER :: OUT

  ! local
    REAL(q) :: X(3,NIONS), Y(3,NIONS), E
    INTEGER :: MODE, NI, NT, N
    REAL(q) :: PROJ

  ! quick return if SCALEE is exactly 1
    IF (SCALEE==1) RETURN

  ! try reading DYNMATFULL
    CALL READ_DYNMATFULL(NIONS, OUT)

  ! quick return if DYNMH is not associated
    IF (.NOT. ASSOCIATED(DYNMH)) THEN
       IF (OUT>=0) THEN
          WRITE(OUT,"(' E(ab-initio)= ',E18.10)") ENERGY
          WRITE(17 ,"(' E(ab-initio)= ',E18.10)") ENERGY
       ENDIF
       ENERGY=ENERGY*SCALEE 
       FORCE =FORCE *SCALEE
       RETURN
    ENDIF

    ! subtract original zero point positions
    X=POS-DYNMH%POS
    DO N=1,DYNMH%NIONS
       X(:,N)=MOD(X(:,N)+6.5_q,1.0_q)-0.5_q
    ENDDO

    ! convert to cartesian coordinates
    CALL DIRKAR(NIONS, X, A )

    ! multiply by sqrt of mass
    N=0
    DO NT=1,DYNMH%NTYP
       DO NI=1,DYNMH%NITYP(NT)
          N=N+1
          X(:,N)=X(:,N)*SQRT(DYNMH%POMASS(NT))
       ENDDO
    ENDDO

    Y=0 ; E=0
    ! now project onto all eigenvectors except translations modes
    ! (thus NIONS-1)
    DO MODE=1,3*(NIONS-1)
       PROJ=SUM(X(:,:)* DYNMH%EVEC(:,:,MODE))
!       IF (OUT>=0) THEN
!          WRITE(77,'(E16.8)',ADVANCE='NO') PROJ
!       ENDIF

       E=E-PROJ*PROJ*DYNMH%EVAL(MODE)/2
       ! multiply by eigenvalue
       PROJ=PROJ*DYNMH%EVAL(MODE)
       Y=Y+PROJ*DYNMH%EVEC(:,:,MODE)
    ENDDO
!    IF (OUT>=0) THEN
!       WRITE(77,*) 
!    ENDIF


    ! total energy is the inproduct with the displacement
    IF (ABS(E+SUM(X*Y)*0.5_q)>=1E-8) THEN
       WRITE(0,*) 'internal error in  DYNMATFULL_ENERGY_Y: forces are energy are not compatible',E,SUM(X*Y)*0.5_q
    ENDIF

    ! multiply by sqrt of mass to get final forces
    N=0
    DO NT=1,DYNMH%NTYP
       DO NI=1,DYNMH%NITYP(NT)
          N=N+1
          Y(:,N)=Y(:,N)*SQRT(DYNMH%POMASS(NT))
       ENDDO
    ENDDO
    
!    WRITE(*,'(3F14.7)') Y
!    WRITE(*,*)
!    WRITE(*,'(3F14.7)') FORCE

    IF (OUT>=0) THEN
       WRITE(OUT,"(' E(ab-initio)= ',E18.10,' E(harmonic)= ',E18.10)") ENERGY,E+DYNMH%TOTEN
       WRITE(17 ,"(' E(ab-initio)= ',E18.10,' E(harmonic)= ',E18.10)") ENERGY,E+DYNMH%TOTEN
    ENDIF
    
    ! finally "mix" energy and forces
    ENERGY=ENERGY*SCALEE+(E+DYNMH%TOTEN)*(1-SCALEE)
    FORCE =FORCE *SCALEE+Y*(1-SCALEE)
    
  END SUBROUTINE DYNMATFULL_ENERGY_FORCE
