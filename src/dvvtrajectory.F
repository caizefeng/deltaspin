      MODULE dvvtrajectory
        USE prec
        USE constant
	  USE poscar
	  USE base
          USE lattice
          USE ini
          USE chain
        IMPLICIT NONE

        TYPE dvv_input
          INTEGER :: EHISTORY
          REAL(q) :: delta_0        !c target error
          REAL(q) :: vnorm_0        !c norm of velocities
          REAL(q) :: minpotim
          REAL(q) :: maxpotim
          LOGICAL :: MINUS          !c negative direction ?
        END TYPE dvv_input

        CONTAINS

        SUBROUTINE dvv_read_input(dvv_inp,IO)
          TYPE(dvv_input) :: dvv_inp
          TYPE(in_struct) :: IO
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC

          LOPEN=.FALSE.
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVEHISTORY','=','#',';','I', &
                dvv_inp%EHISTORY,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%EHISTORY=5
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            dvv_inp%EHISTORY=5
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''DVVEHISTORY'' from file INCAR'
          ENDIF
          IF (dvv_inp%EHISTORY<2 ) dvv_inp%EHISTORY=5

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVDELTA0','=','#',';','F', &
                IDUM,dvv_inp%delta_0,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%delta_0=1.5e-3 
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            dvv_inp%delta_0=1.5e-3
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''DVVDELTA0'' from file INCAR'
          ENDIF
          IF (dvv_inp%delta_0<0._q) dvv_inp%delta_0=1.5e-3

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVVNORM0','=','#',';','F', &
                IDUM,dvv_inp%vnorm_0,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%vnorm_0=0.01 
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            dvv_inp%vnorm_0=0.01 
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''DVVVNORM0'' from file INCAR'
          ENDIF
          IF (dvv_inp%vnorm_0<0._q) dvv_inp%vnorm_0=0.01

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVMINPOTIM','=','#',';','F', &
                IDUM,dvv_inp%minpotim,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%minpotim=0.025_q 
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
             dvv_inp%minpotim=0.025_q
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''DVVMINPOTIM'' from file INCAR'
          ENDIF
          IF (dvv_inp%minpotim<0._q) dvv_inp%minpotim=0.025_q

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVMAXPOTIM','=','#',';','F', &
                IDUM,dvv_inp%maxpotim,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%maxpotim=3._q
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
             dvv_inp%maxpotim=3._q
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''DVVMAXPOTIM'' from file INCAR'
          ENDIF
          IF (dvv_inp%maxpotim<dvv_inp%minpotim) dvv_inp%maxpotim=10*dvv_inp%minpotim

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DVVMINUS','=','#',';','L', &
                IDUM,RDUM,CDUM,dvv_inp%MINUS,CHARAC,N,1,IERR)
          IF (IERR==3) dvv_inp%MINUS=.FALSE.
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            dvv_inp%MINUS=.FALSE.
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''DVVMINUS'' from file INCAR'
          ENDIF

          CLOSE(IO%IU5)
        END SUBROUTINE dvv_read_input

        SUBROUTINE dvv(DYN,T_INFO,INFO,LATT_CUR,IO,EPOT,FACT)
	  TYPE(dynamics) :: DYN
	  TYPE(type_info) :: T_INFO
          TYPE (info_struct) :: INFO
	  TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE (dvv_input),SAVE :: dvv_inp
          REAL(q), ALLOCATABLE, SAVE :: F_old(:,:),V_old(:,:),X_old(:,:)
          REAL(q), ALLOCATABLE, SAVE ::  F_old2(:,:),V_old2(:,:)
          REAL(q) :: X_prime(3,T_INFO%NIONS)
          REAL(q) :: F_b(3,T_INFO%NIONS)
	  REAL(q) :: POTIM_old
          REAL(q) :: vnorm_i,step
	  REAL(q) :: delta_i
          REAL(q) :: FACT
          REAL(q) :: EPOT
          REAL(q) :: EKIN,TEIN
          REAL(q),ALLOCATABLE,SAVE :: EHISTORY(:)
          REAL(q),SAVE :: totalstep=0._q
          INTEGER,SAVE :: counter=0
          INTEGER :: i,j,NI,NT

  1234 FORMAT(/,"  DAMPED VELOCITY VERLET ALGORITHM: "&
              /,"  -----------------------------------------------------------")  
  1235 FORMAT(/,"    Specific input parameters:",/)
  1236 FORMAT(  "      DVVMINUS    ",L9)
  1237 FORMAT(  "      DVVMINPOTIM ",F9.4)
  1238 FORMAT(  "      DVVMAXPOTIM ",F9.4)
  1239 FORMAT(  "      DVVVNORM0   ",F9.4)
  1240 FORMAT(  "      DVVDELTA0   ",F9.4,/)
  1243 FORMAT(  "      DVVEHISTORY ",I2)
  1241 FORMAT(  "    IRC (A): ",F12.8," E(eV): ",E14.8)
  1242 FORMAT(  "    max. gradient (eV/A): ",F12.8)
  1250 FORMAT(X,3(F18.16,X))

          IF (IO%IU6>=0) WRITE(IO%IU6,1234)
          counter=counter+1
          IF (counter==1) THEN
            CALL dvv_read_input(dvv_inp,IO)

            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1235)
              WRITE(IO%IU6,1236) dvv_inp%MINUS
              WRITE(IO%IU6,1243) dvv_inp%EHISTORY
              WRITE(IO%IU6,1237) dvv_inp%minpotim
              WRITE(IO%IU6,1238) dvv_inp%maxpotim
              WRITE(IO%IU6,1239) dvv_inp%vnorm_0
              WRITE(IO%IU6,1240) dvv_inp%delta_0
            END IF

            ALLOCATE(EHISTORY(dvv_inp%EHISTORY))
            ALLOCATE(F_old(3,T_INFO%NIONS))
            ALLOCATE(V_old(3,T_INFO%NIONS))
            ALLOCATE(F_old2(3,T_INFO%NIONS))
            ALLOCATE(V_old2(3,T_INFO%NIONS))
            ALLOCATE(X_old(3,T_INFO%NIONS))
            EHISTORY=0._q
            F_old=0._q;V_old=0._q;X_old=0._q
            F_old2=0._q;V_old2=0._q
          ENDIF 

          IF (dvv_inp%MINUS) THEN
            IF (IO%IU6>=0) write(IO%IU6,1241) -totalstep,EPOT
          ELSE
            IF (IO%IU6>=0) write(IO%IU6,1241) totalstep,EPOT
          ENDIF

          DYN%EDIFFG=1e-12
          CALL add_energy(dvv_inp,EHISTORY,EPOT)
          !write(*,*) 'EHISTORY',EHISTORY
          IF (counter .GE. dvv_inp%EHISTORY) THEN
            INFO%LSTOP=check_finish(dvv_inp,EHISTORY)
            IF (INFO%LSTOP) THEN
              DYN%EDIFFG=1000
              RETURN
            ENDIF
          ENDIF

          FACT=(DYN%POTIM**2)*EVTOJ/AMTOKG *1E-10_q
          DYN%POSIOC=DYN%POSION
          F_b=DYN%D2C
         
          IF (counter==1) THEN
            IF (dvv_inp%MINUS) DYN%VEL=-DYN%VEL
          ELSE
            DYN%VEL=DYN%VEL+(DYN%D2C+F_old*DYN%POTIM**2)/2._q
          ENDIF
          POTIM_old=DYN%POTIM

          !c renormalize velocity
          DYN%VEL=DYN%VEL/DYN%POTIM
          DYN%VEL=MATMUL((LATT_CUR%A),DYN%VEL)
          vnorm_i=norm_three(3,T_INFO%NIONS,DYN%VEL)
          !write(*,*) 'vnorm_i',vnorm_i
          DYN%VEL=DYN%VEL*dvv_inp%vnorm_0/vnorm_i
          vnorm_i=norm_three(3,T_INFO%NIONS,DYN%VEL)
          DYN%VEL=MATMUL(TRANSPOSE(LATT_CUR%B),DYN%VEL)
          DYN%VEL=DYN%VEL*DYN%POTIM
          !write(*,*) 'DYN%VEL3',DYN%VEL
	  DYN%POSION=DYN%POSIOC+DYN%VEL+DYN%D2C/2._q
          CALL put_in_box(T_INFO,DYN%POSION)
          IF (counter>2) THEN
            X_prime=X_old+V_old2*(POTIM_old+DYN%POTIM)+F_old2*(POTIM_old+DYN%POTIM)**2/2._q
            X_prime=DYN%POSION-X_prime
            !write(*,*) 'X_prime',X_prime
            CALL put_in_box_center(T_INFO,X_prime)
            X_prime=MATMUL((LATT_CUR%A),X_prime)
            delta_i=norm_three(3,T_INFO%NIONS,X_prime)
            delta_i=MAX(delta_i,MAXVAL(ABS(X_prime)))
            !write(*,*) 'delta_i',delta_i,norm_three(3,T_INFO%NIONS,X_prime),MAXVAL(ABS(X_prime))
	    DYN%VEL=DYN%VEL/DYN%POTIM
            DYN%D2C=DYN%D2C/DYN%POTIM**2
            DYN%POTIM=DYN%POTIM*(dvv_inp%delta_0/delta_i)**(1._q/3._q)
            DYN%POTIM=MIN(DYN%POTIM,dvv_inp%maxpotim)
            DYN%POTIM=MAX(DYN%POTIM,dvv_inp%minpotim)
            !write(*,*) 'minmax',dvv_inp%maxpotim,dvv_inp%minpotim
            DYN%VEL=DYN%VEL*DYN%POTIM
            DYN%D2C=DYN%D2C*DYN%POTIM**2
          END IF	
          X_old=DYN%POSIOC
          V_old2=V_old
          F_old2=F_old
          V_old=DYN%VEL/DYN%POTIM
          F_old=DYN%D2C/DYN%POTIM**2
          X_prime=DYN%POSION-DYN%POSIOC
          CALL put_in_box_center(T_INFO,X_prime)
          X_prime=MATMUL((LATT_CUR%A),X_prime)
          step=norm_three(3,T_INFO%NIONS,X_prime)
          totalstep=totalstep+step
          !write(*,*) 'step:',POTIM_old,step
          DYN%D2C=F_b
          DYN%POSIOC=DYN%POSION
	END SUBROUTINE dvv

        SUBROUTINE add_energy(dvv_inp,EHISTORY,EPOT)
          TYPE (dvv_input) :: dvv_inp
          REAL(q) :: EHISTORY(dvv_inp%EHISTORY)
          REAL(q) :: EPOT
          INTEGER :: i

          DO i=1,dvv_inp%EHISTORY-1
            EHISTORY(dvv_inp%EHISTORY-i+1)=EHISTORY(dvv_inp%EHISTORY-i)
          ENDDO
          EHISTORY(1)=EPOT
        END SUBROUTINE add_energy

        FUNCTION check_finish(dvv_inp,EHISTORY)
          TYPE (dvv_input) :: dvv_inp
          REAL(q) :: EHISTORY(dvv_inp%EHISTORY)
          LOGICAL :: check_finish
          INTEGER :: i

          check_finish=.TRUE.
          DO i=1,dvv_inp%EHISTORY-1
            IF (EHISTORY(i)<EHISTORY(i+1)) check_finish=.FALSE.
          ENDDO
         
        END FUNCTION check_finish
 
        FUNCTION norm_three(m,n,X)
          INTEGER :: m,n,i,j
          REAL(q) :: X(m,n)
          REAL(q) :: norm_three
          
          norm_three=0._q
	  DO i=1,m
	    DO j=1,n
              norm_three=norm_three+X(i,j)**2
	    ENDDO
	  ENDDO
          norm_three=SQRT(norm_three)
        END FUNCTION norm_three

        SUBROUTINE put_in_box(T_INFO,X)
          TYPE(type_info) :: T_INFO
          REAL(q) :: X(3,T_INFO%NIONS)
          INTEGER :: i,j

          DO i=1,3
            DO j=1,T_INFO%NIONS
              DO
                IF (X(i,j) .GE. 1._q) THEN
                  X(i,j)=X(i,j)-1._q
                ELSE IF (X(i,j) .LT. 0._q) THEN
                  X(i,j)=X(i,j)+1._q
                ELSE
                  EXIT
                ENDIF
              ENDDO
            ENDDO
          ENDDO
        END SUBROUTINE put_in_box

        SUBROUTINE put_in_box_center(T_INFO,X)
          TYPE(type_info) :: T_INFO
          REAL(q) :: X(3,T_INFO%NIONS)
          INTEGER :: i,j

          DO i=1,3
            DO j=1,T_INFO%NIONS
              DO
                IF (X(i,j) .GT. 0.5_q) THEN
                  X(i,j)=X(i,j)-1._q
                ELSE IF (X(i,j) .LE. -0.5_q) THEN
                  X(i,j)=X(i,j)+1._q
                ELSE
                  EXIT
                ENDIF
              ENDDO
            ENDDO
          ENDDO
        END SUBROUTINE put_in_box_center

      END MODULE dvvtrajectory
