      MODULE internals
        USE prec
        USE mymath
        USE poscar
        USE base
        IMPLICIT NONE 
        CONTAINS

        SUBROUTINE constraints_reader(T_INFO,g_io,DPOS,lattmat,CCOORDINATES,IO)
          TYPE(type_info) :: T_INFO
          TYPE(coordstructure) :: CCOORDINATES
          TYPE(gadget_io) :: g_io
          TYPE (in_struct) :: IO
          REAL(q) :: DPOS(3,T_INFO%NIONS)        !c positions in direct
          REAL(q) :: lattmat(3,3)         !c direct and reciprocal lattice
          CHARACTER (LEN=2) :: tag
          INTEGER :: counter,primcounter,what(4),ios,info,dummy,stat
          REAL(q) :: dummyq
          INTEGER :: i,j
          CHARACTER (LEN=2000) :: line 

          counter=0
          primcounter=0
          OPEN(UNIT=g_io%CONSTRAINTS,FILE='ICONST',STATUS='UNKNOWN')
   
          DO
            what=0
            READ(g_io%CONSTRAINTS,FMT='(A2000)',IOSTAT=ios) line
            IF (ios<0) EXIT
            !write(*,*) LEN_TRIM(line),'LEN_TRIM(line)'
            IF (LEN_TRIM(line)>0) THEN
            line=TRIM(ADJUSTL(line))
            tag=line(1:2)
            line=line(3:)

            !READ(g_io%CONSTRAINTS,FMT='(A2)',IOSTAT=ios,ADVANCE='NO') tag
            !IF (ios<0) EXIT
            counter=counter+1

            IF (counter>SIZE(CCOORDINATES%COORDSTRUCT)) THEN
              CCOORDINATES%COORDSTRUCT=>coordreallocate(CCOORDINATES%COORDSTRUCT,counter+10)
            ENDIF
            nullify(CCOORDINATES%COORDSTRUCT(counter)%COEFS)
            
            stat=100
            SELECT CASE(tag)
              CASE('X ')  !c cartesian x
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),stat
                CALL sanity_test(T_INFO%NIONS,what,1,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='X '
                primcounter=primcounter+1
              CASE('Y ')  !c cartesian y
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),stat
                CALL sanity_test(T_INFO%NIONS,what,1,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='Y '
                primcounter=primcounter+1
              CASE('Z ')  !c cartesian z
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),stat
                CALL sanity_test(T_INFO%NIONS,what,1,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='Z '
                primcounter=primcounter+1
              CASE('R ')  !c interatomic distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),stat
                CALL sanity_test(T_INFO%NIONS,what,2,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='R '
                primcounter=primcounter+1
              CASE('Q ')  !c interatomic distance^2
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),stat
                CALL sanity_test(T_INFO%NIONS,what,2,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='Q '
                primcounter=primcounter+1
              CASE('A ')  !c bond angle
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),what(3),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),what(3),stat
                CALL sanity_test(T_INFO%NIONS,what,3,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='A '
                primcounter=primcounter+1
              CASE('T ')  !c torsion
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),what(3),what(4),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),what(3),what(4),stat
                CALL sanity_test(T_INFO%NIONS,what,4,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='T '
                primcounter=primcounter+1
              CASE('M ')  !c atom-midpoint distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),what(3),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),what(3),stat
                CALL sanity_test(T_INFO%NIONS,what,3,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='M '
                primcounter=primcounter+1
              CASE('B ')  !c midpoint-midpoint distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),what(3),what(4),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),what(3),what(4),stat
                CALL sanity_test(T_INFO%NIONS,what,4,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='B '
                primcounter=primcounter+1
              CASE('C ')  !c norm of vector of coordinates
                CCOORDINATES%COORDSTRUCT(counter)%TAG='C '
                ALLOCATE(CCOORDINATES%COORDSTRUCT(counter)%COEFS(primcounter))
                CCOORDINATES%COORDSTRUCT(counter)%COEFS=0
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                READ(line,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                CCOORDINATES%COORDSTRUCT(counter)%STATUS=stat
              CASE('S ')  !c sum of coordinates
                CCOORDINATES%COORDSTRUCT(counter)%TAG='S '
                ALLOCATE(CCOORDINATES%COORDSTRUCT(counter)%COEFS(primcounter))
                CCOORDINATES%COORDSTRUCT(counter)%COEFS=0
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                READ(line,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                CCOORDINATES%COORDSTRUCT(counter)%STATUS=stat
              CASE('P ')  !c ratio between coordinates
                CCOORDINATES%COORDSTRUCT(counter)%TAG='P'
                ALLOCATE(CCOORDINATES%COORDSTRUCT(counter)%COEFS(primcounter))
                CCOORDINATES%COORDSTRUCT(counter)%COEFS=0
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                READ(line,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                CCOORDINATES%COORDSTRUCT(counter)%STATUS=stat
              CASE('D ')  !c coordination number
                CCOORDINATES%COORDSTRUCT(counter)%TAG='D '
                ALLOCATE(CCOORDINATES%COORDSTRUCT(counter)%COEFS(primcounter))
                CCOORDINATES%COORDSTRUCT(counter)%COEFS=0
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                READ(line,FMT=*,IOSTAT=ios) CCOORDINATES%COORDSTRUCT(counter)%COEFS(:),stat
                CCOORDINATES%COORDSTRUCT(counter)%STATUS=stat
              !CASE('E ')
              !  CCOORDINATES%COORDSTRUCT(counter)%TAG='E '
              !  READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) stat
              !  what(1)=0;what(2)=0;what(3)=0;what(4)=0 
              !  primcounter=primcounter+1
              !CASE('F ')
              !  CCOORDINATES%COORDSTRUCT(counter)%TAG='F '
              !  READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) stat
              !  what(1)=0;what(2)=0;what(3)=0;what(4)=0 
              !  primcounter=primcounter+1
              !CASE('G ')
              !  CCOORDINATES%COORDSTRUCT(counter)%TAG='G '
              !  READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) stat
              !  what(1)=0;what(2)=0;what(3)=0;what(4)=0 
              !  stat=0
              !  primcounter=primcounter+1
              CASE('LR')  !c interatomic distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),stat
                !CALL sanity_test(T_INFO%NIONS,what,1,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='LR'
                primcounter=primcounter+1
              CASE('LA')  !c interatomic distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) what(1),what(2),stat
                READ(line,FMT=*,IOSTAT=ios) what(1),what(2),stat
                !CALL sanity_test(T_INFO%NIONS,what,2,IO,counter)
                CCOORDINATES%COORDSTRUCT(counter)%TAG='LA'
                primcounter=primcounter+1
              CASE('LV')  !c interatomic distance
                !READ(g_io%CONSTRAINTS,FMT=*,IOSTAT=ios) stat
                READ(line,FMT=*,IOSTAT=ios) stat
                CCOORDINATES%COORDSTRUCT(counter)%TAG='LV'
                primcounter=primcounter+1
              CASE DEFAULT
                IF (IO%IU0>=0) write(IO%IU0,*) 'Error reading ICONST (item',counter,'): unsupported coordinate type'
                STOP
            END SELECT

            !c terminate if STATUS was not defined for a coordinate in ICONST
            IF ( .NOT.  (stat==0 .OR. stat==5 .OR. stat==7 .OR. stat==8 )) THEN
               IF (IO%IU0>=0) write(IO%IU0,*) 'Error reading ICONST (item',counter,'): missing or invalid STATUS'
               STOP
            ENDIF

            IF (counter==primcounter) THEN
              CCOORDINATES%COORDSTRUCT(counter)%WHAT=what
              CALL where_shortest(DPOS,lattmat,CCOORDINATES%COORDSTRUCT(counter))
              CCOORDINATES%COORDSTRUCT(counter)%STATUS=stat
              SELECT CASE(tag)
                CASE('X ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=DPOS(1,CCOORDINATES%COORDSTRUCT(counter)%WHAT(1))
                CASE('Y ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=DPOS(2,CCOORDINATES%COORDSTRUCT(counter)%WHAT(1))
                CASE('Z ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=DPOS(3,CCOORDINATES%COORDSTRUCT(counter)%WHAT(1))
                CASE('R ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_DISTANCE(T_INFO%NIONS,TRANSPOSE(lattmat),&
                    & DPOS,what(1),what(2),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                  & CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:))
                CASE('Q ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_DISTANCE(T_INFO%NIONS,TRANSPOSE(lattmat),&
                    & DPOS,what(1),what(2),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                  & CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:))
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CCOORDINATES%COORDSTRUCT(counter)%VALUE**2
                CASE('A ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_ANGLE(T_INFO%NIONS,transpose(lattmat),&
                  & DPOS,what(1),what(2),what(3),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                   &   CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:),CCOORDINATES%COORDSTRUCT(counter)%WHERE(3,:),info)
                CASE('T ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_TORSION(T_INFO%NIONS,transpose(lattmat),&
                  & DPOS,what(1),what(2),what(3),what(4),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                  & CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:),CCOORDINATES%COORDSTRUCT(counter)%WHERE(3,:),&
                  & CCOORDINATES%COORDSTRUCT(counter)%WHERE(4,:),info)
                CASE('M ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_MIDDLE(T_INFO%NIONS,transpose(lattmat),&
                  & DPOS,what(1),what(2),what(3),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                   &   CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:),CCOORDINATES%COORDSTRUCT(counter)%WHERE(3,:))
                CASE('B ')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_MIDDLE2(T_INFO%NIONS,transpose(lattmat),&
                  & DPOS,what(1),what(2),what(3),what(4),CCOORDINATES%COORDSTRUCT(counter)%WHERE(1,:),&
                   &   CCOORDINATES%COORDSTRUCT(counter)%WHERE(2,:),CCOORDINATES%COORDSTRUCT(counter)%WHERE(3,:),&
                  & CCOORDINATES%COORDSTRUCT(counter)%WHERE(4,:))
                !CASE('E ')
                !  CCOORDINATES%COORDSTRUCT(counter)%VALUE=0.
                !CASE('F ')
                !  CCOORDINATES%COORDSTRUCT(counter)%VALUE=0.
                !CASE('G ')
                !  CCOORDINATES%COORDSTRUCT(counter)%VALUE=0.
                CASE('LR')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_LR(TRANSPOSE(lattmat),what(1))
                CASE('LA')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_LA(TRANSPOSE(lattmat),what(1),what(2))
                CASE('LV')
                  CCOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_LV(TRANSPOSE(lattmat))
              END SELECT
            ENDIF
            ENDIF
          ENDDO

          CLOSE(g_io%CONSTRAINTS)
          CCOORDINATES%NUMINTERNALS=counter
          CCOORDINATES%COORDSTRUCT=>coordreallocate(CCOORDINATES%COORDSTRUCT,counter)
          IF (counter/=primcounter) THEN
            DO i=1,primcounter
              CCOORDINATES%COORDSTRUCT(i)%STATUS=1
            ENDDO
            DO i=primcounter+1,counter
              CCOORDINATES%COORDSTRUCT(i)%VALUE=0._q
              SELECT CASE(CCOORDINATES%COORDSTRUCT(i)%TAG)
                CASE('C ')
                  DO j=1,SIZE(CCOORDINATES%COORDSTRUCT(i)%COEFS)
                    CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(i)%VALUE+&
                    (CCOORDINATES%COORDSTRUCT(j)%VALUE*CCOORDINATES%COORDSTRUCT(i)%COEFS(j))**2
                  ENDDO
                  CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(i)%VALUE**0.5
                  !CCOORDINATES%COORDSTRUCT(i)%STATUS=0
                CASE('S ')
                  DO j=1,SIZE(CCOORDINATES%COORDSTRUCT(i)%COEFS)
                    CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(i)%VALUE+&
                    CCOORDINATES%COORDSTRUCT(j)%VALUE*CCOORDINATES%COORDSTRUCT(i)%COEFS(j)
                  ENDDO
                  !CCOORDINATES%COORDSTRUCT(i)%STATUS=0
                CASE('D ')
                  DO j=1,SIZE(CCOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ABS(CCOORDINATES%COORDSTRUCT(i)%COEFS(j))>1e-4) THEN
                      dummyq=CCOORDINATES%COORDSTRUCT(j)%VALUE/ABS(CCOORDINATES%COORDSTRUCT(i)%COEFS(j))
                      IF (ABS(dummyq-1._q)<1e-4) THEN
                        dummyq=1._q+1e-4
                      ENDIF
                      IF (CCOORDINATES%COORDSTRUCT(i)%COEFS(j)>0._q) THEN
                        CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(i)%VALUE+&
                                                   & (1.0-(dummyq)**9)/(1.0-(dummyq)**14)
                      ELSE
                        CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(i)%VALUE-&
                                                   & (1.0-(dummyq)**9)/(1.0-(dummyq)**14)
                      ENDIF
                    ENDIF
                  ENDDO
                  
                CASE('P ')
                  dummy=0
                  DO j=1,SIZE(CCOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ABS(CCOORDINATES%COORDSTRUCT(i)%COEFS(j))>0) THEN
                      IF (dummy==0) THEN
                        CCOORDINATES%COORDSTRUCT(i)%VALUE=CCOORDINATES%COORDSTRUCT(j)%VALUE
                        dummy=dummy+1
                      ELSE
                        CCOORDINATES%COORDSTRUCT(i)%VALUE=&
                        &CCOORDINATES%COORDSTRUCT(i)%VALUE/CCOORDINATES%COORDSTRUCT(j)%VALUE
                        EXIT
                      ENDIF
                    ENDIF
                  ENDDO
              END SELECT
            ENDDO
          ENDIF
        END SUBROUTINE constraints_reader

        SUBROUTINE sanity_test(NIONS,WHAT,N,IO,num)
         !c check that the numbers used to define a constraint are in the interval <1,NIONS>
         !c quit with error message otherwise
          TYPE (in_struct) :: IO
          INTEGER :: NIONS,N
          INTEGER :: WHAT(4)
          INTEGER :: i,num
          LOGICAL :: ldum

          ldum=.FALSE.
          DO i=1,N
            IF ((WHAT(i)>NIONS) .OR. (WHAT(i)<1)) THEN
              ldum=.TRUE.
            END IF
          ENDDO
          
          IF (ldum) THEN
            IF (IO%IU0>=0) write(IO%IU0,*) 'Error reading ICONST (item',num,'): invalid definition of coordinate'
            STOP
          ENDIF
        END SUBROUTINE sanity_test

        SUBROUTINE work_coords(DPOS,A,ELEM,&
                             E_TABLE,ICOORDINATES,g_io,T_INFO,IO)
          TYPE(type_info) :: T_INFO
          TYPE(coordstructure) :: ICOORDINATES
          TYPE(elemtable) :: E_TABLE
          TYPE(gadget_io) :: g_io
          TYPE (in_struct) :: IO
          REAL(q) :: DPOS(3,T_INFO%NIONS)  ! positions in direct
          REAL(q) :: A(3,3)         ! direct and reciprocal lattice
          REAL(q) :: ATRADII(T_INFO%NTYP)  ! list of covalent radii for given elem.
          REAL(q) :: CRITERIA(3)
          INTEGER :: i,j,dummy
          CHARACTER (LEN=2) :: ELEM(T_INFO%NTYP) ! all elements, P%ELEMENT
          LOGICAL, SAVE :: init=.true.


         ! IF (init) THEN
         !   init=.false.
         !   nullify(ICOORDINATES%COORDSTRUCT,ICOORDINATES%TOPOLOGY,ICOORDINATES%FRAGMENTS)
         ! ELSE
         !   IF (associated(ICOORDINATES%COORDSTRUCT)) deallocate(ICOORDINATES%COORDSTRUCT)
         !   IF (associated(ICOORDINATES%TOPOLOGY)) deallocate(ICOORDINATES%TOPOLOGY)
         !   IF (associated(ICOORDINATES%FRAGMENTS)) deallocate(ICOORDINATES%FRAGMENTS)
         ! END IF

          ALLOCATE(ICOORDINATES%COORDSTRUCT(10))
          ICOORDINATES%NUMINTERNALS=0
          CALL constraints_reader(T_INFO,g_io,DPOS,TRANSPOSE(A),ICOORDINATES,IO)
          dummy=ICOORDINATES%NUMINTERNALS
          ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,3*T_INFO%NIONS+dummy)
          DO i=1,T_INFO%NIONS
            DO j=1,3
              nullify(ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%COEFS)
              SELECT CASE (j)
              CASE(1)
                ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%TAG='X '
              CASE(2)
                ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%TAG='Y '
              CASE(3)
                ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%TAG='Z '
              END SELECT
              ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%WHAT=(/i,0,0,0/)
              ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%WHERE=0
              ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%VALUE=DPOS(j,i)
              ICOORDINATES%COORDSTRUCT(3*i-(3-j)+dummy)%STATUS=1
            ENDDO
          ENDDO
          ICOORDINATES%NUMINTERNALS=SIZE(ICOORDINATES%COORDSTRUCT)
          CALL AT_LABELS(ICOORDINATES,ELEM,T_INFO%NTYP,T_INFO%NITYP,E_TABLE)
        END SUBROUTINE

        SUBROUTINE interfind(DPOS,A,NIONS,NTYP,ELEM,NITYP,&
                             E_TABLE,ICOORDINATES)
          TYPE(coordstructure) :: ICOORDINATES
          TYPE(elemtable) :: E_TABLE
          REAL(q) :: DPOS(3,NIONS)  ! positions in direct
          REAL(q) :: A(3,3)         ! direct and reciprocal lattice
          REAL(q) :: ATRADII(NTYP)  ! list of covalent radii for given elem.
          REAL(q) :: CRITERIA(3)
          INTEGER :: NITYP(NTYP)    ! number of ions for each type
          INTEGER :: NIONS          ! number of ions
          INTEGER :: NTYP           ! number of types
          INTEGER :: i
          CHARACTER (LEN=2) :: ELEM(NTYP) ! all elements, P%ELEMENT
          LOGICAL, SAVE :: init=.true.

          !IF (init) THEN
          !  init=.false.
          !  nullify(ICOORDINATES%COORDSTRUCT,ICOORDINATES%TOPOLOGY)
          !ELSE
          !  IF (associated(ICOORDINATES%COORDSTRUCT)) deallocate(ICOORDINATES%COORDSTRUCT)
          !  IF (associated(ICOORDINATES%TOPOLOGY)) deallocate(ICOORDINATES%TOPOLOGY)
          !END IF

          !c find atomic radii for given atoms
          ATRADII=SET_ATRADII(NIONS,NTYP,ELEM,E_TABLE)
          !c calculate 'extension' of the unit cell
          CRITERIA=SET_CRITERIA(MAXVAL(ATRADII),A)
          !c calculate in ternal coordinates
          CALL SET_BONDS(NIONS,NTYP,NITYP,DPOS,ATRADII,CRITERIA,A,ICOORDINATES)
          CALL find_fragments(ICOORDINATES)
          CALL SET_ANGLES(NIONS,DPOS,A,ICOORDINATES)
          CALL SET_TORSIONS(NIONS,DPOS,A,ICOORDINATES)
          ICOORDINATES%NUMINTERNALS=SIZE(ICOORDINATES%COORDSTRUCT)
          CALL AT_LABELS(ICOORDINATES,ELEM,NTYP,NITYP,E_TABLE)
        END SUBROUTINE

        SUBROUTINE find_fragments(ICOORDINATES)
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER, ALLOCATABLE :: frgm(:,:)
          INTEGER :: i,j,dim
          INTEGER :: dummy

          dim=SIZE(ICOORDINATES%TOPOLOGY,1)
          ALLOCATE(frgm(dim,dim))
          frgm=0

          DO i=1,dim
            DO j=1,dim
              IF (i==j) THEN
                frgm(i,j)=2
              ELSE
                frgm(i,j)=ICOORDINATES%TOPOLOGY(i,j)
              ENDIF
            ENDDO
          ENDDO


          DO i=1,dim
            dummy=0
            DO j=1,dim
              IF (frgm(i,j)==1) THEN
                dummy=1
                WHERE(frgm(i,:)==1 .AND. frgm(j,:)==0)
                  frgm(j,:)=1
                END WHERE
                WHERE(frgm(i,:)==2)
                  frgm(j,:)=2
                END WHERE
              ENDIF
            ENDDO
            IF (dummy==1) frgm(i,:)=0
          ENDDO

          ALLOCATE(ICOORDINATES%FRAGMENTS(dim,dim))
          ICOORDINATES%FRAGMENTS=0
          j=0
          DO i=1,dim
            IF (SUM(frgm(i,:))>0) THEN
              j=j+1
              ICOORDINATES%FRAGMENTS(j,:)=frgm(i,:)
            ENDIF
          ENDDO
          DEALLOCATE(frgm)
          ICOORDINATES%FRAGMENTS=>ireallocate(ICOORDINATES%FRAGMENTS,j,dim)
        END SUBROUTINE find_fragments

        SUBROUTINE AT_LABELS(ICOORDINATES,ELEM,NTYP,NITYP,E_TABLE)
          TYPE(coordstructure) :: ICOORDINATES
          TYPE(elemtable) :: E_TABLE
          INTEGER :: NTYP           ! number of types
          INTEGER :: NITYP(NTYP)    ! number of ions for each type
          INTEGER :: i,j,count
          !INTEGER, POINTER ::AT_NTYP(:)
          INTEGER, ALLOCATABLE ::AT_NTYP(:)
          INTEGER :: NELEM(NTYP)
          CHARACTER (LEN=2) :: ELEM(NTYP) ! all elements, P%ELEMENT
          !CHARACTER (LEN=2), POINTER ::AT_TYP(:)
          CHARACTER (LEN=2), ALLOCATABLE ::AT_TYP(:)

          ALLOCATE(AT_TYP(SUM(NITYP)))
          ALLOCATE(AT_NTYP(SUM(NITYP)))
          NELEM=0
          DO i=1,NTYP
            DO j=1,SIZE(E_TABLE%ELEMENTS)
              IF (ELEM(i)==E_TABLE%ELEMENTS(j)) NELEM(i)=j
            ENDDO
          ENDDO

          count=1
          DO i=1,NTYP
            DO j=1,NITYP(i)
              AT_TYP(count)=ELEM(i)
              AT_NTYP(count)=NELEM(i)
              count=count+1
            ENDDO
          ENDDO

          DO i=1,ICOORDINATES%NUMINTERNALS
            IF (ICOORDINATES%COORDSTRUCT(i)%TAG /= 'C ' .AND.  ICOORDINATES%COORDSTRUCT(i)%TAG /= 'S ' &
               &  .AND. ICOORDINATES%COORDSTRUCT(i)%TAG /= 'P ') THEN
              DO j=1,4
                IF (ICOORDINATES%COORDSTRUCT(i)%WHAT(j)/=0) THEN
                  ICOORDINATES%COORDSTRUCT(i)%ATOMS(j)=AT_TYP(ICOORDINATES%COORDSTRUCT(i)%WHAT(j))
                  ICOORDINATES%COORDSTRUCT(i)%NATOMS(j)=AT_NTYP(ICOORDINATES%COORDSTRUCT(i)%WHAT(j))
                ELSE
                  ICOORDINATES%COORDSTRUCT(i)%ATOMS(j)='xx'
                  ICOORDINATES%COORDSTRUCT(i)%NATOMS(j)=0
                ENDIF
              ENDDO
            ENDIF
          ENDDO
          DEALLOCATE(AT_NTYP,AT_TYP)
        END SUBROUTINE

        FUNCTION SET_ATRADII(NIONS,NTYP,ELEM,E_TABLE) RESULT(ATRADII)
          TYPE(elemtable) :: E_TABLE
          REAL(q) :: ATRADII(NTYP)  ! list of covalent radii for given elem.
          INTEGER :: NIONS          ! number of ions
          INTEGER :: NTYP           ! number of types
          INTEGER :: i,j
          CHARACTER (LEN=2) :: ELEM(NTYP) ! all elements, P%ELEMENT

          ATRADII=0
          DO i=1,NTYP
            DO j=1,SIZE(E_TABLE%ELEMENTS)
              IF (ELEM(i)==E_TABLE%ELEMENTS(j)) THEN
                ATRADII(i)=E_TABLE%COVALENT_RADII(j)
              ENDIF
            ENDDO
          ENDDO

        END FUNCTION

        FUNCTION SET_CRITERIA(MAXRAD,A) RESULT(CRITERIA)
          REAL(q) :: MAXRAD
          REAL(q) :: A(3,3)
          REAL(q) :: CRITERIA(3)
          REAL(q) :: norm1(3),norm2(3),norm3(3)
          REAL(q) :: cang10,cang21,cang32 

          !write(*,*) 'A',A
          !c find normal to the plane...
          norm1=CROSSPROD(3,A(:,2),A(:,3))
          norm2=CROSSPROD(3,A(:,3),A(:,1))
          norm3=CROSSPROD(3,A(:,1),A(:,2))
          !write(*,*) 'norm3',norm3
          !c ...normalize it...
          norm1=norm1/VECTORSIZE(3,norm1)
          norm2=norm2/VECTORSIZE(3,norm2)
          norm3=norm3/VECTORSIZE(3,norm3)
          !write(*,*) 'norm3_',norm3
          !c cos angles between normals and lattice vectors
          cang10=SUM(norm1*A(:,1))
          cang21=SUM(norm2*A(:,2))
          cang32=SUM(norm3*A(:,3))
          !write(*,*) 'cang32',cang32
          !CRITERIA(1)=abs(2*MAXRAD/cang10)
          !CRITERIA(2)=abs(2*MAXRAD/cang21)
          !CRITERIA(3)=abs(2*MAXRAD/cang32)
          !write(*,*) 'MAXRAD',MAXRAD
          CRITERIA(1)=abs(MAXRAD/cang10)
          CRITERIA(2)=abs(MAXRAD/cang21)
          CRITERIA(3)=abs(MAXRAD/cang32)
        END FUNCTION

        FUNCTION CAL_BOND(A,POINT1,POINT2) RESULT(BOND)
          REAL(q) :: POINT1(3),POINT2(3),VECT(3)
          REAL(q) :: A(3,3)
          REAL(q) :: BOND
          INTEGER :: i
 
          VECT=POINT1-POINT2
          CALL min_imageV(3,VECT)
          VECT=MATMUL(A,VECT)
          BOND=(SUM(VECT*VECT))**0.5
        END FUNCTION
 
        SUBROUTINE SET_BONDS(NIONS,NTYP,NITYP,DX,ATRADII,CRITERIA,A,ICOORDINATES)
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: NIONS,NTYP
          INTEGER :: i,j,k,target,counter
          INTEGER :: NITYP(NTYP),INTYP(NTYP)    ! number of ions for each type
          REAL(q) :: DX(3,NIONS),CX(3,NIONS),A(3,3)
          REAL(q) :: ATRADII(NTYP),RADII
          REAL(q) :: CRITERIA(3)
          REAL(q), DIMENSION(:,:),POINTER :: IDIR, ALLDIR
          INTEGER,DIMENSION(:,:), POINTER :: IWHATWHERE,ALLWHATWHERE
          REAL(q) :: ASCALE
          INTEGER :: PEXCLUDED(3,NIONS)
          INTEGER :: NEXCLUDED(3,NIONS)
          INTEGER :: diminter(2),totaldim
          LOGICAL, SAVE :: init=.true.
          REAL(q) :: BOND

          nullify(ALLDIR)
          nullify(ALLWHATWHERE)

          ASCALE=1.3

          !c collect those atoms which could form intra- and inter-
          !c -cell bonds with neighbords
          PEXCLUDED=0
          NEXCLUDED=0
         
          DO i=1,3
            DO j=1,NIONS
              IF (DX(i,j)<=CRITERIA(i)) THEN
                PEXCLUDED(i,j)=1
              ENDIF
              IF (DX(i,j)>=1-CRITERIA(i)) THEN
                NEXCLUDED(i,j)=1
              ENDIF
            ENDDO         
          ENDDO

          totaldim=0
          DO i=-1,1
            DO j=-1,1
              DO k=-1,1
                CALL MULTY_CELL(NIONS,DX,i,j,k,PEXCLUDED,NEXCLUDED,IDIR, &
                IWHATWHERE)
                diminter=SHAPE(IDIR)
                IF ((diminter(2)+totaldim)>totaldim) THEN
                  totaldim=totaldim+diminter(2)
                  ALLDIR=>qreallocate(ALLDIR,3,totaldim)
                  ALLWHATWHERE=>ireallocate(ALLWHATWHERE,4,totaldim)
                  ALLDIR(:,(totaldim-diminter(2)+1):totaldim)=IDIR(:,:)
                  ALLWHATWHERE(:,(totaldim-diminter(2)+1):totaldim)=IWHATWHERE(:,:)
                ENDIF
              ENDDO
            ENDDO
          ENDDO

          deallocate(IDIR,IWHATWHERE)
          nullify(IDIR,IWHATWHERE)

          !c calculate bonds, if smaller than sum of covalent
          !c radii
          !#########
          ALLOCATE(ICOORDINATES%COORDSTRUCT(NIONS))
          ALLOCATE(ICOORDINATES%TOPOLOGY(NIONS,NIONS))
          ICOORDINATES%TOPOLOGY=0

          INTYP=NITYP
          counter=0
          DO i=2,NTYP
            INTYP(i)=INTYP(i)+INTYP(i-1)
          ENDDO
          CX=MATMUL(A,DX)
          !ALLDIR=MATMUL(A,ALLDIR)

          DO i=1,NIONS
            DO j=1,totaldim
!               BOND=CAL_BOND(CX(:,i),ALLDIR(:,j))
              BOND=CAL_BOND(A,DX(:,i),ALLDIR(:,j))
           
              TARGET=0
              DO
                TARGET=TARGET+1
                IF (i<=INTYP(TARGET)) EXIT
              ENDDO
              RADII=ATRADII(TARGET)
              TARGET=0
              DO
                TARGET=TARGET+1
                IF (ALLWHATWHERE(1,j)<=INTYP(TARGET)) EXIT
              ENDDO
              RADII=RADII+ATRADII(TARGET)
              RADII=RADII*ASCALE
              IF (BOND<=RADII .AND. BOND>0.1 .AND. i<ALLWHATWHERE(1,j)) THEN
                counter=counter+1
                !###########
                IF (counter>SIZE(ICOORDINATES%COORDSTRUCT)) THEN
                  ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,counter+NIONS)
                ENDIF

                ICOORDINATES%COORDSTRUCT(counter)%TAG='R '
                ICOORDINATES%COORDSTRUCT(counter)%WHAT=(/i,ALLWHATWHERE(1,j),0,0/)
                ICOORDINATES%COORDSTRUCT(counter)%WHERE(1,:)=(/0,0,0/)
                ICOORDINATES%COORDSTRUCT(counter)%WHERE(2,:)=(/ALLWHATWHERE(2,j),&
                                             ALLWHATWHERE(3,j),ALLWHATWHERE(4,j)/)
                ICOORDINATES%COORDSTRUCT(counter)%WHERE(3,:)=(/0,0,0/)
                ICOORDINATES%COORDSTRUCT(counter)%WHERE(4,:)=(/0,0,0/)
                ICOORDINATES%COORDSTRUCT(counter)%VALUE=BOND
                ICOORDINATES%TOPOLOGY(i,ALLWHATWHERE(1,j))=1
                ICOORDINATES%TOPOLOGY(ALLWHATWHERE(1,j),i)=1
              ENDIF
            ENDDO
          ENDDO
          deallocate(ALLWHATWHERE,ALLDIR)
          nullify(ALLWHATWHERE,ALLDIR)

          ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,counter)
          ICOORDINATES%NUMBONDS=counter
        END SUBROUTINE

        SUBROUTINE MULTY_CELL(NIONS,DX,i,j,k,PEXCLUDED,NEXCLUDED, &
          INTDIRECTS,INTWHATWHERE)
          INTEGER :: NIONS
          INTEGER :: EXCLUDED(NIONS),PEXCLUDED(3,NIONS),NEXCLUDED(3,NIONS)
          INTEGER :: i,j,k,dmsn,ix,jx
          REAL(q) :: DX(3,NIONS)
          REAL(q),DIMENSION(:,:),POINTER :: INTDIRECTS
          INTEGER,DIMENSION(:,:),POINTER :: INTWHATWHERE         
          LOGICAL, SAVE :: init=.true.

          IF (init) THEN
            init=.false.
            nullify(INTDIRECTS)
            nullify(INTWHATWHERE)           
          ELSE 
            IF (associated(INTDIRECTS)) deallocate(INTDIRECTS)
            IF (associated(INTWHATWHERE)) deallocate(INTWHATWHERE)
          END IF 

          EXCLUDED=1
          IF (i==1) EXCLUDED=EXCLUDED*PEXCLUDED(1,:)
          IF (i==-1) EXCLUDED=EXCLUDED*NEXCLUDED(1,:)
          IF (j==1) EXCLUDED=EXCLUDED*PEXCLUDED(2,:)
          IF (j==-1) EXCLUDED=EXCLUDED*NEXCLUDED(2,:)
          IF (k==1) EXCLUDED=EXCLUDED*PEXCLUDED(3,:)
          IF (k==-1) EXCLUDED=EXCLUDED*NEXCLUDED(3,:)

          dmsn=SUM(EXCLUDED)
          ALLOCATE(INTDIRECTS(3,dmsn),INTWHATWHERE(4,dmsn))
          jx=1
          DO ix=1,NIONS
            IF (EXCLUDED(ix)==1) THEN 
              INTDIRECTS(:,jx)=DX(:,ix)+(/i,j,k/)
              INTWHATWHERE(:,jx)=(/ix,i,j,k/)
              jx=jx+1
            ENDIF
          ENDDO
        END SUBROUTINE

        SUBROUTINE SET_ANGLES(NIONS,DX,A,ICOORDINATES)
          TYPE(coordstructure) :: ICOORDINATES
          INTEGER :: NIONS,i,j,counter
          REAL(q) :: DX(3,NIONS)
          REAL(q) :: A(3,3)
          INTEGER :: pa,pb,pc,pd, p1,p2,p3
          INTEGER :: w1(3),w2(3),w3(3)

          counter=0
          DO i=1,ICOORDINATES%NUMBONDS-1
            IF (ICOORDINATES%COORDSTRUCT(i)%TAG=='R ') THEN
              pa=ICOORDINATES%COORDSTRUCT(i)%WHAT(1)
              pb=ICOORDINATES%COORDSTRUCT(i)%WHAT(2)
            ELSE
              CONTINUE
            ENDIF
            DO j=i+1,ICOORDINATES%NUMBONDS
              IF (ICOORDINATES%COORDSTRUCT(j)%TAG=='R ') THEN
                pc=ICOORDINATES%COORDSTRUCT(j)%WHAT(1)
                pd=ICOORDINATES%COORDSTRUCT(j)%WHAT(2)
              ELSE
                CONTINUE
              ENDIF
              IF (pa==pc) THEN
                p1=pb
                p2=pa
                p3=pd
                w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                w3=ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)
                CALL ANGLE_EVAL(NIONS,DX,A,ICOORDINATES,p1,p2,p3,w1,w2,w3,counter)
              END IF
              if (pa==pd) THEN
                p1=pb
                p2=pa
                p3=pc
                w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                w3=ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)
                CALL ANGLE_EVAL(NIONS,DX,A,ICOORDINATES,p1,p2,p3,w1,w2,w3,counter)
              END IF
              if (pb==pc) THEN
                p1=pa
                p2=pc
                p3=pd
                w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                w2=ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)
                w3=ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)
                CALL ANGLE_EVAL(NIONS,DX,A,ICOORDINATES,p1,p2,p3,w1,w2,w3,counter)
              END IF
              if (pb==pd) THEN
                p1=pa
                p2=pb
                p3=pc
                w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)-ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                w3=ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)
                CALL ANGLE_EVAL(NIONS,DX,A,ICOORDINATES,p1,p2,p3,w1,w2,w3,counter)
              END IF
            ENDDO
          ENDDO
          ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,ICOORDINATES%NUMBONDS+counter)
          ICOORDINATES%NUMANGLES=counter
          END SUBROUTINE

          SUBROUTINE ANGLE_EVAL(NIONS,DX,A,ICOORDINATES,p1,p2,p3,w1,w2,w3,counter)
            TYPE(coordstructure) :: ICOORDINATES
            INTEGER :: NIONS,counter,info
            REAL(q) :: angle
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: A(3,3)
            INTEGER :: p1,p2,p3
            INTEGER :: w1(3),w2(3),w3(3),pf(3),wf(9)

            info=0
            angle=CAL_ANGLE(NIONS,A,DX,p1,p2,p3,w1,w2,w3,info)
            IF (info==0) THEN
              !IF ((ICOORDINATES%TOPOLOGY(p1,p3)==0) .AND. p1/=p3) THEN
              !  ICOORDINATES%COORDSTRUCT(counter)%TAG='R '
              !  ICOORDINATES%COORDSTRUCT(counter)%WHAT=(/p1,p3,0,0/)
              !  ICOORDINATES%COORDSTRUCT(counter)%WHERE(1,:)=(/0,0,0/)
              !  ICOORDINATES%COORDSTRUCT(counter)%WHERE(2,:)=(/w3(:)-w1(:)/)
              !  ICOORDINATES%COORDSTRUCT(counter)%WHERE(3,:)=(/0,0,0/)
              !  ICOORDINATES%COORDSTRUCT(counter)%WHERE(4,:)=(/0,0,0/)
              !  ICOORDINATES%COORDSTRUCT(counter)%VALUE=CAL_DISTANCE(NIONS,A,DX,p1,p3,(/0,0,0/),w3-w1) 
              !  ICOORDINATES%TOPOLOGY(p1,p3)=1
              !  ICOORDINATES%TOPOLOGY(p3,p1)=1
              !  counter=counter+1
              !  IF (ICOORDINATES%NUMBONDS+counter>SIZE(ICOORDINATES%COORDSTRUCT)) THEN
              !    ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,&
              !                                ICOORDINATES%NUMBONDS+counter+NIONS)
              !  ENDIF
              !END IF
              RETURN
            ENDIF
            IF (info==1) THEN
              IF (SUM(ICOORDINATES%TOPOLOGY(p1,:))>6 .AND. SUM(ICOORDINATES%TOPOLOGY(p2,:))>6 .AND. &
                  SUM(ICOORDINATES%TOPOLOGY(p3,:))>6) info=0
            ENDIF
            IF (info==1) THEN
            counter=counter+1
            IF (ICOORDINATES%NUMBONDS+counter>SIZE(ICOORDINATES%COORDSTRUCT)) THEN
                ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,&
                                              ICOORDINATES%NUMBONDS+counter+NIONS)
            ENDIF
            IF (p1<p3) THEN
              pf=(/p1,p2,p3/)
              wf=(/w1(1),w1(2),w1(3),w2(1),w2(2),w2(3),w3(1),w3(2),w3(3)/)
            ELSE
              pf=(/p3,p2,p1/)
              wf=(/w3(1),w3(2),w3(3),w2(1),w2(2),w2(3),w1(1),w1(2),w1(3)/)
            ENDIF
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%TAG='A '
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%WHAT=(/pf(1),pf(2),pf(3),0/)
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%WHERE(1,:)=(/wf(1),wf(2),wf(3)/)
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%WHERE(2,:)=(/wf(4),wf(5),wf(6)/)
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%WHERE(3,:)=(/wf(7),wf(8),wf(9)/)
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%WHERE(4,:)=(/0,0,0/)
            ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+counter)%VALUE=angle
            ENDIF 
          END SUBROUTINE ANGLE_EVAL

         
          SUBROUTINE SET_TORSIONS(NIONS,DX,A,ICOORDINATES)
            TYPE(coordstructure) :: ICOORDINATES
            INTEGER :: i,j,k,l,counter,NIONS
            INTEGER :: p1,p2,p3,p4
            INTEGER :: p(5),pf(4),wf(12)
            INTEGER :: w1(3),w2(3),w3(3),w4(3)
            INTEGER :: info
            REAL(q) :: A(3,3),DX(3,NIONS) 
            LOGICAL :: CONDITION1,CONDITION2,CONDITION3,CONDITION4,yo
            REAL(q) :: torsion

            counter=0
            info=0
            DO i=ICOORDINATES%NUMBONDS+1,ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES
              IF (ICOORDINATES%COORDSTRUCT(i)%TAG/='A ') THEN
                CONTINUE
              ENDIF  
              p=0
              p(1:3)=ICOORDINATES%COORDSTRUCT(i)%WHAT(1:3)
              DO j=1,ICOORDINATES%NUMBONDS
                IF (ICOORDINATES%COORDSTRUCT(i)%TAG/='R ') THEN
                  CONTINUE
                ENDIF  
                p(4:5)= ICOORDINATES%COORDSTRUCT(j)%WHAT(1:2)
                CONDITION1=(p(4)==p(1) .AND. p(4)>=p(2) .AND. &
                            p(5)/=p(2) .AND. p(5)/=p(3))
                CONDITION2=(p(4)==p(3) .AND. p(4)>=p(2) .AND. & 
                            p(5)/=p(2) .AND. p(5)/=p(1))
                CONDITION3=(p(5)==p(1) .AND. p(5)>=p(2) .AND. & 
                            p(4)/=p(2) .AND. p(4)/=p(3))
                CONDITION4=(p(5)==p(3) .AND. p(5)>=p(2) .AND. & 
                            p(4)/=p(2) .AND. p(4)/=p(1))
                yo=.false.
                IF (CONDITION1) THEN
                  yo=.true.
                  p1=p(3)
                  p2=p(2)
                  p3=p(1)
                  p4=p(5)
                  w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                  w3=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  w4=ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)+&
                     ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  IF (SUM(ICOORDINATES%TOPOLOGY(p2,:))>4 .OR. SUM(ICOORDINATES%TOPOLOGY(p3,:))>4) yo=.false.
                ELSE IF (CONDITION2) THEN
                  yo=.true.
                  p1=p(1)
                  p2=p(2)
                  p3=p(3)
                  p4=p(5)
                  w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                  w3=ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  w4=ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)+&
                     ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  IF (SUM(ICOORDINATES%TOPOLOGY(p2,:))>4 .OR. SUM(ICOORDINATES%TOPOLOGY(p3,:))>4) yo=.false.
                ELSE IF (CONDITION3) THEN
                  yo=.true.
                  p1=p(3)
                  p2=p(2)
                  p3=p(1)
                  p4=p(4)
                  w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                  w3=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  w4=ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)+&
                     ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  IF (SUM(ICOORDINATES%TOPOLOGY(p2,:))>4 .OR. SUM(ICOORDINATES%TOPOLOGY(p3,:))>4) yo=.false.
                ELSE IF (CONDITION4) THEN
                  yo=.true.
                  p1=p(1)
                  p2=p(2)
                  p3=p(3)
                  p4=p(4)
                  w1=ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  w2=ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
                  w3=ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  w4=ICOORDINATES%COORDSTRUCT(j)%WHERE(1,:)-ICOORDINATES%COORDSTRUCT(j)%WHERE(2,:)+&
                     ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:)
                  IF (SUM(ICOORDINATES%TOPOLOGY(p2,:))>4 .OR. SUM(ICOORDINATES%TOPOLOGY(p3,:))>4) yo=.false.
                END IF
                
                !c dont allow torsions around atoms with more then four bonds
                IF (yo) THEN
                  counter=counter+1
                  torsion=CAL_TORSION(NIONS,A,DX,p1,p2,p3,p4,w1,w2,w3,w4,info)
                  IF (ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+&
                        counter>SIZE(ICOORDINATES%COORDSTRUCT)) THEN
                    ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,&
                                     ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter+NIONS)
                  ENDIF

                  IF (p1<p4) THEN
                    pf=(/p1,p2,p3,p4/)
                    wf=(/w1(1),w1(2),w1(3),w2(1),w2(2),w2(3), &
                         w3(1),w3(2),w3(3),w4(1),w4(2),w4(3)/)
                  ELSE
                    pf=(/p4,p3,p2,p1/)
                    wf=(/w4(1),w4(2),w4(3),w3(1),w3(2),w3(3), &
                         w2(1),w2(2),w2(3),w1(1),w1(2),w1(3)/)
                  ENDIF
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%TAG='T '
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%WHAT=pf
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%WHERE(1,:)=&
                                        (/wf(1),wf(2),wf(3)/)
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%WHERE(2,:)=&
                                        (/wf(4),wf(5),wf(6)/)
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%WHERE(3,:)=&
                                        (/wf(7),wf(8),wf(9)/)
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%WHERE(4,:)=&
                                        (/wf(10),wf(11),wf(12)/)
                  ICOORDINATES%COORDSTRUCT(ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)%VALUE=&
                                         torsion
                ENDIF
              ENDDO
            ENDDO   

            ICOORDINATES%NUMTORSIONS=counter
            ICOORDINATES%COORDSTRUCT=>coordreallocate(ICOORDINATES%COORDSTRUCT,&
                 ICOORDINATES%NUMBONDS+ICOORDINATES%NUMANGLES+counter)
          END SUBROUTINE

          FUNCTION CAL_DISTANCE(NIONS,A,DX,p1,p2,w1,w2) RESULT(dist)
            INTEGER :: NIONS
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: vect(3)
            INTEGER :: p1,p2
            INTEGER :: w1(3),w2(3)
            REAL(q) :: dist

            vect=DX(:,p2)+w2 - (DX(:,p1)+w1)
            vect=MATMUL(A,vect)
            dist=VECTORSIZE(3,vect)
          END FUNCTION

          FUNCTION CAL_LR(A,p1) RESULT(dist)
            REAL(q) :: A(3,3)
            REAL(q) :: vect(3)
            INTEGER :: p1
            REAL(q) :: dist

            vect=A(:,p1)
            dist=VECTORSIZE(3,vect)
          END FUNCTION

         FUNCTION CAL_LA(A,p1,p2) RESULT(angle)
            REAL(q) :: A(3,3)
            REAL(q) :: vect1(3),vect2(3)
            INTEGER :: p1,p2
            REAL(q) :: angle

            vect1=A(:,p1)
            vect2=A(:,p2)
            angle=sum(vect1*vect2)/VECTORSIZE(3,vect1)/VECTORSIZE(3,vect2)
            angle=ACOS(angle)
          END FUNCTION

          FUNCTION CAL_LV(A) RESULT(dist)
            REAL(q) :: A(3,3)
            REAL(q) :: vect(3)
            REAL(q) :: dist

            vect=CROSSPROD(3,A(:,1),A(:,2))
            dist=sum(vect*A(:,3))
          END FUNCTION

          FUNCTION CAL_ANGLE(NIONS,A,DX,p1,p2,p3,w1,w2,w3,info) RESULT(angle)
            INTEGER :: NIONS
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: vect1(3),vect2(3)
            INTEGER :: p1,p2,p3
            INTEGER :: w1(3),w2(3),w3(3)
            INTEGER :: info
            REAL(q) :: angle

            info=0
            vect1=DX(:,p2)+w2 - (DX(:,p1)+w1)
            CALL min_imageV(3,vect1)
            vect2=DX(:,p2)+w2 - (DX(:,p3)+w3)
            CALL min_imageV(3,vect2)
            vect1=MATMUL(A,vect1)
            vect2=MATMUL(A,vect2)
            angle=DOT_PRODUCT(vect1,vect2)
            angle=angle/VECTORSIZE(3,vect1)/VECTORSIZE(3,vect2)
            IF (angle>1) angle=1._q
            IF (angle<-1) angle=-1._q
            angle=ACOS(angle)
            IF (SIN(angle)>0.3) info=1
          END FUNCTION

          FUNCTION CAL_TORSION(NIONS,A,DX,p1,p2,p3,p4,w1,w2,w3,w4,info) RESULT(torsion)
            INTEGER :: NIONS,p1,p2,p3,p4
            INTEGER :: w1(3),w2(3),w3(3),w4(3)
            INTEGER :: info
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: vect1(3),vect2(3),vect3(3),vect4(3)
            REAL(q) :: cross1(3),cross2(3)
            REAL(q) :: torsion,tst
           
            info=0
            vect1=DX(:,p1)+w1 - (DX(:,p2)+w2)
            CALL min_imageV(3,vect1)
            vect2=DX(:,p2)+w2 - (DX(:,p3)+w3)
            CALL min_imageV(3,vect2)
            vect3=DX(:,p3)+w3 - (DX(:,p4)+w4)
            CALL min_imageV(3,vect3)
            vect1=MATMUL(A,vect1)
            vect2=MATMUL(A,vect2)
            vect3=MATMUL(A,vect3)
            tst=DOT_PRODUCT(vect1,vect2)/VECTORSIZE(3,vect1)/VECTORSIZE(3,vect2)
            tst=(1-tst**2)**0.5
            IF (tst>0.15) info=1
            tst=DOT_PRODUCT(vect2,vect3)/VECTORSIZE(3,vect2)/VECTORSIZE(3,vect3)
            tst=(1-tst**2)**0.5
            IF (tst<0.15) info=0
            cross1=CROSSPROD(3,vect1,vect2)          
            cross1=cross1/VECTORSIZE(3,cross1)
            cross2=CROSSPROD(3,vect2,vect3)           
            cross2=cross2/VECTORSIZE(3,cross2)
            torsion=DOT_PRODUCT(cross1,cross2)
            IF (torsion>1) torsion=1_q
            IF (torsion<-1) torsion=-1_q
            torsion=ACOS(torsion)
            IF (DOT_PRODUCT(cross1,vect3)>0) torsion=-torsion         
          END FUNCTION

          FUNCTION CAL_MIDDLE(NIONS,A,DX,p1,p2,p3,w1,w2,w3)  RESULT(dist)
            INTEGER :: NIONS
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: vect(3)
            INTEGER :: p1,p2,p3
            INTEGER :: w1(3),w2(3),w3(3)
            REAL(q) :: dist

            vect=((DX(:,p2)+w2) + (DX(:,p3)+w3))/2-(DX(:,p1)+w1)
            CALL min_imageV(3,vect)
            vect=MATMUL(A,vect)
            dist=VECTORSIZE(3,vect)
          END FUNCTION CAL_MIDDLE

          FUNCTION CAL_MIDDLE2(NIONS,A,DX,p1,p2,p3,p4,w1,w2,w3,w4)  RESULT(dist)
            INTEGER :: NIONS
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,NIONS)
            REAL(q) :: vect(3)
            INTEGER :: p1,p2,p3,p4
            INTEGER :: w1(3),w2(3),w3(3),w4(3)
            REAL(q) :: dist

            vect=((DX(:,p3)+w3) + (DX(:,p4)+w4))/2-((DX(:,p1)+w1) + (DX(:,p2)+w2))/2
            CALL min_imageV(3,vect)
            vect=MATMUL(A,vect)
            dist=VECTORSIZE(3,vect)
          END FUNCTION CAL_MIDDLE2

          SUBROUTINE ROT_MOM(T_INFO,A,POS,POS_old,RM)
            TYPE(type_info) :: T_INFO
            REAL(q) :: POS(3,T_INFO%NIONS),POS_old(3,T_INFO%NIONS)
            REAL(q) :: cPOS(3,T_INFO%NIONS),cPOS_old(3,T_INFO%NIONS)
            REAL(q) :: A(3,3)
            REAL(q) :: RM(3,1)
            INTEGER :: i

            RM=0._q
            cPOS=POS;cPOS_old=POS_old
            CALL minimize_difference(cPOS,cPOS_old,T_INFO%NIONS)
            cPOS=MATMUL(A,cPOS);cPOS_old=MATMUL(A,cPOS_old)
            DO i=1,T_INFO%NIONS
              RM(1,1)=RM(1,1)+T_INFO%POMASS(T_INFO%ITYP(i))*(cPOS(2,i)*cPOS_old(3,i)-cPOS(3,i)*cPOS_old(2,i))
              RM(2,1)=RM(2,1)+T_INFO%POMASS(T_INFO%ITYP(i))*(cPOS(3,i)*cPOS_old(1,i)-cPOS(1,i)*cPOS_old(3,i))
              RM(3,1)=RM(3,1)+T_INFO%POMASS(T_INFO%ITYP(i))*(cPOS(1,i)*cPOS_old(2,i)-cPOS(2,i)*cPOS_old(1,i))
            ENDDO

            !DO i=1,T_INFO%NIONS
            !  RM(1,1)=RM(1,1)+T_INFO%POMASS(T_INFO%ITYP(i))*((POS(2,i)-CM(2,1))*CM(3,1)-(POS(3,i)-CM(3,1))*CM(2,1))
            !  RM(2,1)=RM(2,1)+T_INFO%POMASS(T_INFO%ITYP(i))*((POS(3,i)-CM(3,1))*CM(1,1)-(POS(1,i)-CM(1,1))*CM(3,1))
            !  RM(3,1)=RM(3,1)+T_INFO%POMASS(T_INFO%ITYP(i))*((POS(1,i)-CM(1,1))*CM(2,1)-(POS(2,i)-CM(2,1))*CM(1,1))
            !  write(*,*) rm,'rm2'
            !ENDDO
            
          END SUBROUTINE ROT_MOM

          SUBROUTINE BMATRIX(T_INFO,DX,CX,lattmat,ICOORDINATES,BMAT,LCART)
          !c Jacobi matrix for transdformation from fractional (or cartesain) to internal coordinates
            TYPE(type_info) :: T_INFO
            TYPE(coordstructure) :: ICOORDINATES
            INTEGER :: i,j,k
            REAL(q) :: lattmat(3,3)
            REAL(q) :: DX(3,T_INFO%NIONS),CX(3,T_INFO%NIONS)
            REAL(q),DIMENSION(:,:) :: BMAT
            INTEGER :: dimn(2),dummy
            REAL(q) :: complexcoord,dummyA,dummyC,dummyD
            LOGICAL :: LCART

            dimn=SHAPE(BMAT)
            BMAT=0_q

            DO i=1,ICOORDINATES%NUMINTERNALS 
              SELECT CASE (ICOORDINATES%COORDSTRUCT(i)%TAG)
                CASE ('R ')
                  BMAT(i,:)=PREPARE_L(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('Q ')
                  BMAT(i,:)=PREPARE_Q(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('A ')
                  BMAT(i,:)=PREPARE_A(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('T ')
                  BMAT(i,:)=PREPARE_T(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('X ','Y ','Z ')
                  BMAT(i,:)=PREPARE_XYZ(DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('M ')
                  BMAT(i,:)=PREPARE_M(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                CASE ('B ')
                  BMAT(i,:)=PREPARE_B(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                !CASE('E ')
                !  BMAT(i,:)=PREPARE_ROT(T_INFO,CX,lattmat,dimn(2),1,LCART)
                !CASE('F ')
                !  BMAT(i,:)=PREPARE_ROT(T_INFO,CX,lattmat,dimn(2),2,LCART)
                !CASE('G ')
                !  BMAT(i,:)=PREPARE_ROT(T_INFO,CX,lattmat,dimn(2),3,LCART)
                CASE ('LR')
                  BMAT(i,:)=PREPARE_LR(T_INFO%NIONS,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2))
                CASE ('LA')
                  BMAT(i,:)=PREPARE_LA(T_INFO%NIONS,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2))
                CASE ('LV')
                  BMAT(i,:)=PREPARE_LV(T_INFO%NIONS,lattmat,dimn(2))
                CASE ('C ')
                  complexcoord=0
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    BMAT(i,:)=BMAT(i,:)+BMAT(j,:)*&
                    ICOORDINATES%COORDSTRUCT(i)%COEFS(j)**2*ICOORDINATES%COORDSTRUCT(j)%VALUE
                    complexcoord=complexcoord+((ICOORDINATES%COORDSTRUCT(i)%COEFS(j)**2)*&
                    ICOORDINATES%COORDSTRUCT(j)%VALUE)
                  ENDDO
                  BMAT(i,:)=BMAT(i,:)/complexcoord**(0.5)
                CASE ('S ')
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    BMAT(i,:)=BMAT(i,:)+BMAT(j,:)*ICOORDINATES%COORDSTRUCT(i)%COEFS(j)
                  ENDDO
                CASE ('D ')
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ICOORDINATES%COORDSTRUCT(i)%COEFS(j)>1e-4) THEN 
                      dummyA=ICOORDINATES%COORDSTRUCT(j)%VALUE/ABS(ICOORDINATES%COORDSTRUCT(i)%COEFS(j))
                      IF (ABS(dummyA-1._q)<1e-4) dummyA=1._q+1e-4
                      dummyC=1.0-(dummyA)**9
                      dummyD=1.0-(dummyA)**14
                      BMAT(i,:)=BMAT(i,:)-9.0*BMAT(j,:)*(dummyA**9.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)/dummyD
                      BMAT(i,:)=BMAT(i,:)+14.0*dummyC*BMAT(j,:)*(dummyA**14.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)&
                                /dummyD**2
                    ELSE IF (ICOORDINATES%COORDSTRUCT(i)%COEFS(j)<-1e-4) THEN
                      dummyA=ICOORDINATES%COORDSTRUCT(j)%VALUE/ABS(ICOORDINATES%COORDSTRUCT(i)%COEFS(j))
                      IF (ABS(dummyA-1._q)<1e-4) dummyA=1._q+1e-4
                      dummyC=1.0-(dummyA)**9
                      dummyD=1.0-(dummyA)**14
                      BMAT(i,:)=BMAT(i,:)+9.0*BMAT(j,:)*(dummyA**9.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)/dummyD
                      BMAT(i,:)=BMAT(i,:)-14.0*dummyC*BMAT(j,:)*(dummyA**14.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)&
                                /dummyD**2

                    ENDIF

!                     IF (ABS(ICOORDINATES%COORDSTRUCT(i)%COEFS(j))>1e-4) THEN 
!                       dummyA=ICOORDINATES%COORDSTRUCT(j)%VALUE/ICOORDINATES%COORDSTRUCT(i)%COEFS(j)
!                       IF (ABS(dummyA-1._q)<1e-4) dummyA=1._q+1e-4
!                       dummyC=1.0-(dummyA)**9
!                       dummyD=1.0-(dummyA)**14
!                       BMAT(i,:)=BMAT(i,:)-9.0*BMAT(j,:)*(dummyA**9.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)/dummyD
!                       BMAT(i,:)=BMAT(i,:)+14.0*dummyC*BMAT(j,:)*(dummyA**14.0/ICOORDINATES%COORDSTRUCT(j)%VALUE)&
!                                 /dummyD**2
!                     ENDIF
                  ENDDO
                CASE('P ')
                  dummy=0
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ICOORDINATES%COORDSTRUCT(i)%COEFS(j)>0) THEN
                      IF (dummy==0) THEN
                        complexcoord=ICOORDINATES%COORDSTRUCT(j)%VALUE
                        SELECT CASE (ICOORDINATES%COORDSTRUCT(j)%TAG)
                          CASE ('R ')
                            BMAT(i,:)=PREPARE_L(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('Q ')
                            BMAT(i,:)=PREPARE_Q(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                          CASE ('A ')
                            BMAT(i,:)=PREPARE_A(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('T ')
                            BMAT(i,:)=PREPARE_T(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('M ')
                            BMAT(i,:)=PREPARE_M(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('B ')
                            BMAT(i,:)=PREPARE_B(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                          CASE ('X ','Y ','Z ')
                            BMAT(i,:)=PREPARE_XYZ(DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                        END SELECT
                        dummy=1
                      ELSE
                        BMAT(i,:)=BMAT(i,:)*ICOORDINATES%COORDSTRUCT(j)%VALUE/complexcoord
                        SELECT CASE (ICOORDINATES%COORDSTRUCT(j)%TAG)
                          CASE ('R ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_L(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('Q ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_Q(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                          CASE ('A ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_A(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('T ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_T(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('M ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_M(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                          CASE ('B ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_B(T_INFO%NIONS,DX,lattmat,ICOORDINATES%COORDSTRUCT(i),dimn(2),LCART)
                          CASE ('X ','Y ','Z ')
                            BMAT(i,:)=BMAT(i,:)-PREPARE_XYZ(DX,lattmat,ICOORDINATES%COORDSTRUCT(j),dimn(2),LCART)
                        END SELECT
                        BMAT(i,:)=BMAT(i,:)*complexcoord/ICOORDINATES%COORDSTRUCT(j)%VALUE**2
                        EXIT
                      ENDIF
                    ENDIF
                ENDDO
              END SELECT
            ENDDO

            CONTAINS
            FUNCTION PREPARE_XYZ(DX,lattmat, COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: ROW(din)
              REAL(q) :: lattmat(3,3),lattinv(3,3)
              INTEGER :: dummy,ninfo
              LOGICAL :: LCART

              ROW=0_q
              SELECT CASE (COORDSTRUCT%TAG)
                CASE ('X ')
                  dummy=2
                CASE ('Y ')
                  dummy=1
                CASE ('Z ')
                  dummy=0
              END SELECT
              ROW(3*COORDSTRUCT%WHAT(1)-dummy)=1._q

              !c transform to fractional coordinates if desired
              IF (LCART) THEN
                 lattinv=lattmat
                 CALL SVDINVERSE(lattinv,3,ninfo)
                 ROW(3*COORDSTRUCT%WHAT(1)-2:3*COORDSTRUCT%WHAT(1))=MATMUL(ROW(3*COORDSTRUCT%WHAT(1)-2:3*COORDSTRUCT%WHAT(1)),TRANSPOSE(lattinv))
              ENDIF 
            END FUNCTION

            FUNCTION PREPARE_L(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),vect(3)
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              vect=(pa-pb)
              CALL min_imageV(3,vect)
              vect=MATMUL(vect,lattmat)
              !vect=vect/COORDSTRUCT%VALUE
              vect=vect/(vect(1)*vect(1)+vect(2)*vect(2)+vect(3)*vect(3))**0.5

              pa=vect
              pb=-vect

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q

              !c convert to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
              ENDIF

              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              IF (din==3*NIONS+9) THEN
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+&
                         OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:)
              ENDIF
            END FUNCTION PREPARE_L

            FUNCTION PREPARE_LR(NIONS,lattmat,COORDSTRUCT,din) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q) :: lattmat(3,3)
              REAL(q) :: ROW(din)
              REAL(q) :: dist

              ROW=0_q
              IF (din==3*NIONS+9) THEN
                dist=CAL_LR(transpose(lattmat),COORDSTRUCT%WHAT(1))
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+1)=lattmat(COORDSTRUCT%WHAT(1),1)/dist 
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+2)=lattmat(COORDSTRUCT%WHAT(1),2)/dist
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+3)=lattmat(COORDSTRUCT%WHAT(1),3)/dist
              ENDIF
            END FUNCTION PREPARE_LR

            FUNCTION PREPARE_LA(NIONS,lattmat,COORDSTRUCT,din) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q) :: lattmat(3,3)
              REAL(q) :: ROW(din)
              REAL(q) :: diffav(3),diffbv(3),dala(3),dalc(3)
              REAL(q) :: d1,d2,alpha

              ROW=0_q
              IF (din==3*NIONS+9) THEN
                diffav=lattmat(COORDSTRUCT%WHAT(1),:)
                diffbv=lattmat(COORDSTRUCT%WHAT(2),:)
                d1=CAL_LR(transpose(lattmat),COORDSTRUCT%WHAT(1))
                d2=CAL_LR(transpose(lattmat),COORDSTRUCT%WHAT(2))
                alpha=CAL_LA(transpose(lattmat),COORDSTRUCT%WHAT(1),COORDSTRUCT%WHAT(2))
                dala=-(diffbv/(d1*d2)-diffav*COS(alpha)/(d1**2))/SIN(alpha)
                dalc=-(diffav/(d1*d2)-diffbv*COS(alpha)/(d2**2))/SIN(alpha)
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+1)=dala(1)
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+2)=dala(2)
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(1)-1)+3)=dala(3) 
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(2)-1)+1)=dalc(1)
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(2)-1)+2)=dalc(2)
                ROW(3*NIONS+3*(COORDSTRUCT%WHAT(2)-1)+3)=dalc(3) 
              ENDIF
            END FUNCTION PREPARE_LA
            

            FUNCTION PREPARE_LV(NIONS,lattmat,din) RESULT(ROW)
              INTEGER :: NIONS,din
              REAL(q) :: lattmat(3,3)
              REAL(q) :: ROW(din)
              REAL(q) :: dl1(3),dl2(3),dl3(3)

              ROW=0_q
              IF (din==3*NIONS+9) THEN
                dl1=CROSSPROD(3,lattmat(2,:),lattmat(3,:))
                dl2=CROSSPROD(3,lattmat(3,:),lattmat(1,:))
                dl3=CROSSPROD(3,lattmat(1,:),lattmat(2,:))          
                ROW(3*NIONS+1:3*NIONS+3)=dl1 
                ROW(3*NIONS+4:3*NIONS+6)=dl2
                ROW(3*NIONS+7:3*NIONS+9)=dl3
              ENDIF
            END FUNCTION PREPARE_LV         


            FUNCTION PREPARE_Q(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),vect(3)
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              vect=(pa-pb)
              CALL min_imageV(3,vect)
              vect=MATMUL(vect,lattmat)
              vect=2*vect

              pa=vect
              pb=-vect

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q

              !c convert to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
              END IF

              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              IF (din==3*NIONS+9) THEN
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+&
                         OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:)
              ENDIF
            END FUNCTION PREPARE_Q


            FUNCTION PREPARE_A(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),pc(3),vect1(3),vect2(3)
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              pc=DX(:,COORDSTRUCT%WHAT(3))+COORDSTRUCT%WHERE(3,:)
              vect1=(pa-pb)
              CALL min_imageV(3,vect1)
              vect1=MATMUL(vect1,lattmat)
              vect2=(pc-pb)
              CALL min_imageV(3,vect2)
              vect2=MATMUL(vect2,lattmat)

              pa=-(vect2/(VECTORSIZE(3,vect1)*VECTORSIZE(3,vect2))- &
                   vect1*COS(COORDSTRUCT%VALUE)/(VECTORSIZE(3,vect1))**2)/SIN(COORDSTRUCT%VALUE)
              pc=-(vect1/(VECTORSIZE(3,vect1)*VECTORSIZE(3,vect2))- &
                 vect2*COS(COORDSTRUCT%VALUE)/(VECTORSIZE(3,vect2))**2)/SIN(COORDSTRUCT%VALUE)
              pb=-pa-pc

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(3))) pc(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(3))) pc(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(3))) pc(3)=0._q

            

              !c transform to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
                pc=MATMUL(pc,TRANSPOSE(lattmat))
              END IF 

              IF (COORDSTRUCT%WHAT(1)==COORDSTRUCT%WHAT(3)) pa=pa+pc

          
              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              IF (COORDSTRUCT%WHAT(1) /= COORDSTRUCT%WHAT(3)) THEN
                ROW(3*COORDSTRUCT%WHAT(3)-3+1:3*COORDSTRUCT%WHAT(3)-3+3)=pc
              ENDIF
              IF (din==3*NIONS+9) THEN
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(3,:),pc)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:) 
              ENDIF          
            END FUNCTION

            FUNCTION PREPARE_M(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),pc(3),vect(3)
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              pc=DX(:,COORDSTRUCT%WHAT(3))+COORDSTRUCT%WHERE(3,:)
              vect=(pb+pc)-2*pa
              CALL min_imageV(3,vect)
              vect=MATMUL(vect,lattmat)
              vect=vect/COORDSTRUCT%VALUE

              !pa=MATMUL(-vect/2,TRANSPOSE(lattmat))
              !pb=MATMUL(vect/4,TRANSPOSE(lattmat))
              !pc=MATMUL(vect/4,TRANSPOSE(lattmat))

              pa=-vect/2
              pb=vect/4
              pc=vect/4

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(3))) pc(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(3))) pc(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(3))) pc(3)=0._q

              !c transform to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
                pc=MATMUL(pc,TRANSPOSE(lattmat))
              END IF
             
              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              ROW(3*COORDSTRUCT%WHAT(3)-3+1:3*COORDSTRUCT%WHAT(3)-3+3)=pc

              IF (din==3*NIONS+9) THEN
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(3,:),pc)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:) 
              ENDIF          
            END FUNCTION PREPARE_M

            FUNCTION PREPARE_B(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),pc(3),pd(3),vect(3)
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              pc=DX(:,COORDSTRUCT%WHAT(3))+COORDSTRUCT%WHERE(3,:)
              pd=DX(:,COORDSTRUCT%WHAT(3))+COORDSTRUCT%WHERE(3,:)
              vect=(pa+pb)-(pc+pd)
              CALL min_imageV(3,vect)
              vect=MATMUL(vect,lattmat)
              vect=vect/COORDSTRUCT%VALUE

              pa=vect/4
              pb=vect/4
              pc=-vect/4
              pd=-vect/4

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(3))) pc(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(3))) pc(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(3))) pc(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(3))) pd(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(3))) pd(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(3))) pd(3)=0._q


              !c transform to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
                pc=MATMUL(pc,TRANSPOSE(lattmat))
                pd=MATMUL(pd,TRANSPOSE(lattmat))
              END IF


              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              ROW(3*COORDSTRUCT%WHAT(3)-3+1:3*COORDSTRUCT%WHAT(3)-3+3)=pc
              ROW(3*COORDSTRUCT%WHAT(4)-3+1:3*COORDSTRUCT%WHAT(4)-3+3)=pd

              IF (din==3*NIONS+9) THEN
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(3,:),pc)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(4,:),pd)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:) 
              ENDIF          
            END FUNCTION PREPARE_B 

            FUNCTION PREPARE_T(NIONS,DX,lattmat,COORDSTRUCT,din,LCART) RESULT(ROW)
              TYPE(coordinate) :: COORDSTRUCT
              INTEGER :: NIONS,din
              REAL(q),DIMENSION(:,:) :: DX
              REAL(q) :: lattmat(3,3)
              REAL(q) :: pa(3),pb(3),pc(3),pd(3),vect1(3),vect2(3),vect3(3)
              REAL(q) :: dr1,dr2,dr3,cospsi2,cospsi3,sinpsi2,sinpsi3
              REAL(q) :: ROW(din)
              REAL(q) :: LATDER(3,3)
              LOGICAL :: LCART

              ROW=0_q
              pa=DX(:,COORDSTRUCT%WHAT(1))+COORDSTRUCT%WHERE(1,:)
              pb=DX(:,COORDSTRUCT%WHAT(2))+COORDSTRUCT%WHERE(2,:)
              pc=DX(:,COORDSTRUCT%WHAT(3))+COORDSTRUCT%WHERE(3,:)
              pd=DX(:,COORDSTRUCT%WHAT(4))+COORDSTRUCT%WHERE(4,:)
              vect1=(pa-pb)
              CALL min_imageV(3,vect1)
              vect1=MATMUL(vect1,lattmat)
              vect2=(pb-pc)
              CALL min_imageV(3,vect2)
              vect2=MATMUL(vect2,lattmat)
              vect3=(pc-pd)
              CALL min_imageV(3,vect3)
              vect3=MATMUL(vect3,lattmat)

              dr1=VECTORSIZE(3,vect1)
              dr2=VECTORSIZE(3,vect2)
              dr3=VECTORSIZE(3,vect3)
              vect1=vect1/dr1
              vect2=vect2/dr2
              vect3=vect3/dr3
              cospsi2=DOT_PRODUCT(vect1,vect2)
              sinpsi2=(1-cospsi2**2)**0.5
              cospsi3=DOT_PRODUCT(vect2,vect3)
              sinpsi3=(1-cospsi3**2)**0.5
              
              pa=-CROSSPROD(3,vect1,vect2)/(dr1*sinpsi2**2)
              pb= (dr2+dr1*cospsi2)/(dr1*dr2*sinpsi2)* &
                  CROSSPROD(3,vect1,vect2)/sinpsi2+    &
                  cospsi3/(dr2*sinpsi3)*               &
                  CROSSPROD(3,vect2,vect3)/sinpsi3
              pc= -(dr2+dr3*cospsi3)/(dr2*dr3*sinpsi3)* &
                   CROSSPROD(3,vect2,vect3)/sinpsi3-    &
                   cospsi2/(dr2*sinpsi2)*               &
                  CROSSPROD(3,vect1,vect2)/sinpsi2
              pd= CROSSPROD(3,vect2,vect3)/(dr3*sinpsi3**2)             

              !c take care of frozen fractional coordinates
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(1))) pa(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(1))) pa(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(1))) pa(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(2))) pb(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(2))) pb(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(2))) pb(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(3))) pc(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(3))) pc(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(3))) pc(3)=0._q
              !IF (.NOT. T_INFO%LSFOR(1,COORDSTRUCT%WHAT(4))) pd(1)=0._q
              !IF (.NOT. T_INFO%LSFOR(2,COORDSTRUCT%WHAT(4))) pd(2)=0._q
              !IF (.NOT. T_INFO%LSFOR(3,COORDSTRUCT%WHAT(4))) pd(3)=0._q


              !c transform to fractional coordinates if desired
              IF (.NOT. LCART) THEN
                pa=MATMUL(pa,TRANSPOSE(lattmat))
                pb=MATMUL(pb,TRANSPOSE(lattmat))
                pc=MATMUL(pc,TRANSPOSE(lattmat))
                pd=MATMUL(pd,TRANSPOSE(lattmat))
              END IF 

              ROW(3*COORDSTRUCT%WHAT(1)-3+1:3*COORDSTRUCT%WHAT(1)-3+3)=pa
              ROW(3*COORDSTRUCT%WHAT(2)-3+1:3*COORDSTRUCT%WHAT(2)-3+3)=pb
              ROW(3*COORDSTRUCT%WHAT(3)-3+1:3*COORDSTRUCT%WHAT(3)-3+3)=pc
              ROW(3*COORDSTRUCT%WHAT(4)-3+1:3*COORDSTRUCT%WHAT(4)-3+3)=pd
              IF (din==3*NIONS+9) THEN 
                LATDER=0._q
                LATDER=OUTERPROD2(3,COORDSTRUCT%WHERE(1,:),pa)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(2,:),pb)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(3,:),pc)+ &
                       OUTERPROD2(3,COORDSTRUCT%WHERE(4,:),pd)
                ROW(3*NIONS+1:3*NIONS+3)=LATDER(1,:)
                ROW(3*NIONS+4:3*NIONS+6)=LATDER(2,:)
                ROW(3*NIONS+7:3*NIONS+9)=LATDER(3,:)  
              ENDIF            
            END FUNCTION
            
            FUNCTION PREPARE_ROT(T_INFO,CX,lattmat,din,tag,LCART) RESULT(ROW)
              TYPE(type_info) :: T_INFO
              INTEGER :: din,i
              REAL(q) :: CX(3,T_INFO%NIONS),CX_(3,T_INFO%NIONS)
              REAL(q) :: lattmat(3,3)
              REAL(q) :: ROW(din)
              INTEGER :: tag
              LOGICAL :: LCART

              ROW=0_q
              CX_=MATMUL(TRANSPOSE(lattmat),CX)
              DO i=1,T_INFO%NIONS
                SELECT CASE(tag)
                  CASE(1)
                    ROW(3*i-3+2)= T_INFO%POMASS(T_INFO%ITYP(i))*CX_(3,i)
                    ROW(3*i-3+3)=-T_INFO%POMASS(T_INFO%ITYP(i))*CX_(2,i)
                  CASE(2)
                    ROW(3*i-3+1)=-T_INFO%POMASS(T_INFO%ITYP(i))*CX_(3,i)
                    ROW(3*i-3+3)= T_INFO%POMASS(T_INFO%ITYP(i))*CX_(1,i)
                  CASE(3)
                    ROW(3*i-3+1)= T_INFO%POMASS(T_INFO%ITYP(i))*CX_(2,i)
                    ROW(3*i-3+2)=-T_INFO%POMASS(T_INFO%ITYP(i))*CX_(1,i)
                END SELECT

                IF (.NOT. LCART) THEN
                  ROW(3*i-3+1:3*i-3+3)=MATMUL(ROW(3*i-3+1:3*i-3+3),TRANSPOSE(lattmat))
                ENDIF
              ENDDO              
            END FUNCTION PREPARE_ROT

          END SUBROUTINE


          SUBROUTINE DEAL_XYZ(T_INFO,DX,CX,A,ICOORDINATES)
          !c compute values of internal coordinates
            TYPE(type_info) :: T_INFO
            TYPE(coordstructure) :: ICOORDINATES
            INTEGER :: i,j,k,info,dummy 
            REAL(q) :: A(3,3)
            REAL(q) :: DX(3,T_INFO%NIONS),CX(3,T_INFO%NIONS)
            REAL(q) :: CMASS(3,1),RM(3,1)
            REAL(q) :: pa(3),pb(3)
            REAL(q) :: complexcoord,dummyq


            !CX=MATMUL(A,DX)
            !CALL GIVE_CMASS(T_INFO,CX,CMASS)
            !CALL ROT_MOM(T_INFO,A,DX,CX,RM)
            !RM=0._q

            DO i=1,SIZE(ICOORDINATES%COORDSTRUCT)
              SELECT CASE (ICOORDINATES%COORDSTRUCT(i)%TAG)
                CASE('X ')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=DX(1,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))
                CASE('Y ')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=DX(2,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))
                CASE('Z ')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=DX(3,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))
                CASE('R ')
                  pa=DX(:,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))+ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  pb=DX(:,ICOORDINATES%COORDSTRUCT(i)%WHAT(2))+ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
!                   pa=MATMUL(A,pa)
!                   pb=MATMUL(A,pb)
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_BOND(A,pa,pb)
                CASE('Q ')
                  pa=DX(:,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))+ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:)
                  pb=DX(:,ICOORDINATES%COORDSTRUCT(i)%WHAT(2))+ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:)
!                   pa=MATMUL(A,pa)
!                   pb=MATMUL(A,pb)
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_BOND(A,pa,pb)
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=ICOORDINATES%COORDSTRUCT(i)%VALUE**2
                CASE('A ')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_ANGLE(T_INFO%NIONS,A,DX,&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(1),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(2),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(3), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:),info)
                CASE('T ')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_TORSION(T_INFO%NIONS,A,DX,&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(1),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(2), &
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(3),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(4), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(4,:),info)
                CASE('M')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_MIDDLE(T_INFO%NIONS,A,DX,&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(1),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(2),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(3), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:))
                CASE('B')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_MIDDLE2(T_INFO%NIONS,A,DX,&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(1),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(2),&
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(3), &
                  ICOORDINATES%COORDSTRUCT(i)%WHAT(4), &
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(1,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(2,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(3,:),&
                  ICOORDINATES%COORDSTRUCT(i)%WHERE(4,:))
                !CASE('E')
                !  ICOORDINATES%COORDSTRUCT(i)%VALUE=RM(1,1)
                !CASE('F')
                !  ICOORDINATES%COORDSTRUCT(i)%VALUE=RM(2,1)
                !CASE('G')
                !  ICOORDINATES%COORDSTRUCT(i)%VALUE=RM(3,1)
                CASE('LR')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_LR(A,ICOORDINATES%COORDSTRUCT(i)%WHAT(1))
                CASE('LA')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_LA(A,ICOORDINATES%COORDSTRUCT(i)%WHAT(1),ICOORDINATES%COORDSTRUCT(i)%WHAT(2))
                CASE('LV')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=CAL_LV(A)
                CASE('C ')
                  complexcoord=0
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    complexcoord=complexcoord+(ICOORDINATES%COORDSTRUCT(i)%COEFS(j)**2*&
                    ICOORDINATES%COORDSTRUCT(j)%VALUE**2)
                  ENDDO
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=complexcoord**0.5
                CASE('S')
                   ICOORDINATES%COORDSTRUCT(i)%VALUE=0._q
                   DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    ICOORDINATES%COORDSTRUCT(i)%VALUE=ICOORDINATES%COORDSTRUCT(i)%VALUE+&
                      ICOORDINATES%COORDSTRUCT(j)%VALUE*ICOORDINATES%COORDSTRUCT(i)%COEFS(j)
                  ENDDO
                CASE('D')
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=0._q
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ABS(ICOORDINATES%COORDSTRUCT(i)%COEFS(j))>1e-4) THEN
                      dummyq=ICOORDINATES%COORDSTRUCT(j)%VALUE/ABS(ICOORDINATES%COORDSTRUCT(i)%COEFS(j))
                      IF (ABS(dummyq-1._q)<1e-4) THEN
                        dummyq=1._q+1e-4
                      ENDIF
                      IF (ICOORDINATES%COORDSTRUCT(i)%COEFS(j)>0._q) THEN
                        ICOORDINATES%COORDSTRUCT(i)%VALUE=ICOORDINATES%COORDSTRUCT(i)%VALUE+&
                                                   & (1.0-(dummyq)**9)/(1.0-(dummyq)**14)
                      ELSE
                        ICOORDINATES%COORDSTRUCT(i)%VALUE=ICOORDINATES%COORDSTRUCT(i)%VALUE-&
                                                   & (1.0-(dummyq)**9)/(1.0-(dummyq)**14)
                      ENDIF
                    ENDIF
                  ENDDO

                  
                CASE('P')
                  dummy=0
                  ICOORDINATES%COORDSTRUCT(i)%VALUE=0._q
                  DO j=1,SIZE(ICOORDINATES%COORDSTRUCT(i)%COEFS)
                    IF (ICOORDINATES%COORDSTRUCT(i)%COEFS(j)>0) THEN
                      IF (dummy==0) THEN
                        ICOORDINATES%COORDSTRUCT(i)%VALUE=ICOORDINATES%COORDSTRUCT(j)%VALUE
                        dummy=dummy+1
                      ELSE
                        ICOORDINATES%COORDSTRUCT(i)%VALUE=&
                        &ICOORDINATES%COORDSTRUCT(i)%VALUE/ICOORDINATES%COORDSTRUCT(j)%VALUE
                        EXIT           
                      ENDIF
                    ENDIF
                  ENDDO
              END SELECT
            ENDDO                       
          END SUBROUTINE DEAL_XYZ

          SUBROUTINE MAKE_UMAT(BMAT,UTRANS)
            !c prepare U matrix which defines
            !c delocalised intern
            REAL(q),DIMENSION(:,:) :: BMAT,UTRANS
            REAL(q),ALLOCATABLE :: GMAT(:,:)
            INTEGER :: shp(2)
            INTEGER :: i,j            
            REAL(q),ALLOCATABLE :: d(:)         !c eigenvalues of the matrix        
            REAL(q),ALLOCATABLE :: u(:,:)     !c eigenvectors of the matrix
            REAL(q),ALLOCATABLE :: vt(:,:)

            shp=SHAPE(BMAT)
            ALLOCATE(GMAT(shp(2),shp(2)))
            GMAT=MATMUL(TRANSPOSE(BMAT),BMAT)
            ALLOCATE(d(shp(2)),u(shp(2),shp(2)),vt(shp(2),shp(2)))
            CALL SVDVALVEC(GMAT,shp(2),d,u,vt) !c calculates evectors and evalues
            DEALLOCATE(GMAT)
            ALLOCATE(GMAT(SIZE(UTRANS(:,1)),shp(2)))
            GMAT=0._q
            j=0
            DO i=1,shp(2)
              IF (ABS(d(i))>1e-06) THEN 
                j=j+1 
                GMAT(j,:)=u(:,i)/(ABS(d(i)))**0.5
              ENDIF
            ENDDO
            UTRANS=MATMUL(GMAT,TRANSPOSE(BMAT))
            DEALLOCATE(vt,u,d,GMAT)
          END SUBROUTINE
      END MODULE
