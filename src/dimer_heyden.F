 !***********************************************************************
! RCS:  $Id: dimer_heyden_reduced.F,v 1.2 2011/12/06 tomas bucko$
!
!  dimer method for the transition state search
!  invented by Henkelman et al.(J. Chem. Phys. 111, 7010 (1999), improved
!  by Heyden et al. (J. Chem. Phys. 123, Art. No. 224101 (2005) 
!
!*********************************************************************


      MODULE dimer_heyden
        USE prec
        USE constant
        USE poscar
        USE base
        USE lattice
        USE ini
        USE chain
        USE main_mpi
        IMPLICIT NONE

        TYPE dimer_input
          REAL(q) :: stepsize                                 !c finite difference for translation
          REAL(q) :: minrot                                   !c minimal allowed rotation of dimer
          REAL(q) :: maxstep                                  !c maximum step length for translation
          INTEGER :: SWITCH                                   !c
          INTEGER :: engine                                   !c optimization engine (1-SD,2-CG)
          INTEGER :: restartCG                                !c restart CG algorithm every **
          INTEGER :: findiff                                  !c finite differences (1-forward,2-central)
          LOGICAL :: Fletcher_Reeves                          !c
          LOGICAL :: dimerdynamics                            !c if true dynamics instead of relaxation
          LOGICAL :: LIDM_SELECTIVE                           !c consider on atoms from "reaction core" in the dimer definision?
        END TYPE

        TYPE dimer_data
          REAL(q),POINTER :: R0_NDa(:),R1_NDa(:),R2_NDa(:)     !c positions
          REAL(q),POINTER :: F0_NDa(:),F1_NDa(:),F2_NDa(:)     !c forces
          REAL(q),POINTER :: F0_dagger(:)
          REAL(q),POINTER :: FN(:)                             !c net force
          REAL(q) :: EPOT0,EPOT1,EPOT2                         !c potential energiesdynMat
          REAL(q),POINTER :: N_ND(:)                           !c unit vector along R1R2
          REAL(q),POINTER :: N_T(:)                            !c direction for translation
          REAL(q),POINTER :: THETA(:)                          !c rotational axis
          REAL(q) :: delta_R                                   !c R1R0 distance
          REAL(q) :: curve                                     !c curvature in dimer direction
          REAL(q) :: curve1                                    !c curvature in dimer direction
          REAL(q) :: dcurve_psi0                               !c derivative of the rotation curvature
          REAL(q) :: dcurve_psi1                               !c derivative of the rotation curvature
          LOGICAL :: crisis
          LOGICAL :: engine_restart    
        END TYPE

        !INCLUDE "dimer_heyden.inc"
        CONTAINS
        
        SUBROUTINE dimer(DYN,T_INFO,INFO,LATT_CUR,IO,EPOT,FACT,LSTOP2)
          TYPE(dynamics) :: DYN
          TYPE(type_info) :: T_INFO
          TYPE (info_struct) INFO
          TYPE(latt) :: LATT_CUR
          TYPE (in_struct) :: IO
          TYPE(dimer_data),SAVE :: Dimer_Dat
          TYPE(dimer_input),SAVE :: Dimer_Inp
          REAL(q) :: FACT
          REAL(q) :: EPOT,a0,a1,b1,maxForce
          REAL(q),SAVE :: alpha
          REAL(q),ALLOCATABLE,SAVE :: REF_C(:,:)
          REAL(q) :: FORCES(3,T_INFO%NIONS)
          INTEGER,SAVE :: counter=0,operation=1
          INTEGER :: i,j
          LOGICAL :: LSTOP2
       
  1234 FORMAT(/,"  DIMER METHOD: (", I1,")",/,"  -----------------------------------------------------------")  
  1235 FORMAT(/,"    Specific input parameters:",/)
  1236 FORMAT("      DIMER_DIST ",F9.4)
  1237 FORMAT("      FINDIFF    ",I9)
  1238 FORMAT("      STEP_SIZE  ",F9.4)
  1239 FORMAT("      STEP_MAX   ",F9.4)
  1240 FORMAT("      MINROT     ",F9.4,/)
  2240 FORMAT("      LIDM_SELECTIVE  ",L1,/)
  
  1241 FORMAT(/,"    operation: ",A40)
  1242 FORMAT("    max. gradient (eV/A): ",F12.8)
  1243 FORMAT("    curvature along the dimer direction: ",F9.4)
  1244 FORMAT("    trial alpha (deg): ",F9.4)
  1245 FORMAT("    alpha smaller than MINROT, no rotation ")
  1246 FORMAT("    dimer rotated by (deg.): ",F9.4)
  1250 FORMAT(X,3(F18.16,X))

          IF (DYN%ISIF>2) THEN
              IF (IO%IU6>=0) THEN
           WRITE(*,"(A,A)")'ERROR: only atomic relaxation is supported, change ISIF to 2, 1, or 0'
              ENDIF
              STOP
          END IF

          IF (IO%IU6>=0) WRITE(IO%IU6,1234) operation
          FORCES=0._q
            
          FORCES=DYN%D2C/FACT
          maxForce=0._q
          DO i=1,T_INFO%NIONS
            IF (SQRT(FORCES(1,i)**2+FORCES(2,i)**2+FORCES(3,i)**2)>maxForce) &
               & maxForce=SQRT(FORCES(1,i)**2+FORCES(2,i)**2+FORCES(3,i)**2)
          END DO
          !maxForce=SQRT(FORCES(1,i)**2+FORCES(2,i)**2+FORCES(3,i)**2)

          IF (DYN%EDIFFG<0.) THEN
            INFO%LSTOP=.TRUE.
            LSTOP2=.TRUE.

            !DO i=1,T_INFO%NIONS
            IF (maxForce>ABS(DYN%EDIFFG)) LSTOP2=.FALSE.
            !END DO

            IF (LSTOP2) THEN
              IF (IO%IU6>=0) WRITE(IO%IU6,1242) maxForce
              INFO%LSTOP=.TRUE.
              DYN%POSIOC=DYN%POSION
              RETURN
            END IF
          ENDIF
 
          DYN%POSIOC=DYN%POSION
          !write(*,*) 'FACT1',FACT
          counter=counter+1
          IF (counter==1) THEN
            ALLOCATE(REF_C(3,T_INFO%NIONS))
            REF_C=0._q
            REF_C=DYN%POSION
          ELSE
            WHERE((DYN%POSION-REF_C)>0.5) DYN%POSION=DYN%POSION-1._q
            WHERE((DYN%POSION-REF_C)<-0.5) DYN%POSION=DYN%POSION+1._q
            REF_C=DYN%POSION
          ENDIF
          
          DYN%POSION=MATMUL(LATT_CUR%A,DYN%POSION)
          
          !c read in some input and allocate necessary arrays
          IF (counter==1) THEN
            ALLOCATE(Dimer_Dat%R0_NDa(3*T_INFO%NIONS),Dimer_Dat%R1_NDa(3*T_INFO%NIONS),&
            &        Dimer_Dat%R2_NDa(3*T_INFO%NIONS))
            ALLOCATE(Dimer_Dat%F0_NDa(3*T_INFO%NIONS),Dimer_Dat%F1_NDa(3*T_INFO%NIONS),&
            &        Dimer_Dat%F2_NDa(3*T_INFO%NIONS),Dimer_Dat%F0_dagger(3*T_INFO%NIONS))
            ALLOCATE(Dimer_Dat%N_ND(3*T_INFO%NIONS),Dimer_Dat%N_T(3*T_INFO%NIONS))
            ALLOCATE(Dimer_Dat%FN(3*T_INFO%NIONS),Dimer_Dat%THETA(3*T_INFO%NIONS))

            Dimer_Dat%delta_R=0._q; Dimer_Dat%curve =0._q  
            Dimer_Dat%curve1 =0._q ; Dimer_Dat%dcurve_psi0 =0._q    
            Dimer_Dat%dcurve_psi1 =0._q
            Dimer_Dat%EPOT0=0._q;Dimer_Dat%EPOT1=0._q;Dimer_Dat%EPOT2=0._q
            Dimer_Dat%R0_NDa=0._q;Dimer_Dat%R1_NDa=0._q;Dimer_Dat%R2_NDa=0._q
            Dimer_Dat%F0_NDa=0._q;Dimer_Dat%F1_NDa=0._q;Dimer_Dat%F2_NDa=0._q;Dimer_Dat%F0_dagger=0._q
            Dimer_Dat%N_ND=0._q;Dimer_Dat%N_T=0._q
            Dimer_Dat%FN=0._q;Dimer_Dat%THETA=0._q

            CALL dimer_read_input(Dimer_Inp,IO,Dimer_Dat)

            IF (IO%IU6>=0) THEN
              WRITE(IO%IU6,1235)
              WRITE(IO%IU6,1236) Dimer_Dat%delta_R
              WRITE(IO%IU6,1237) Dimer_Inp%findiff
              WRITE(IO%IU6,1238) Dimer_Inp%stepsize
              WRITE(IO%IU6,1239) Dimer_Inp%maxstep
              WRITE(IO%IU6,1240) Dimer_Inp%minrot
              WRITE(IO%IU6,2240) Dimer_Inp%LIDM_SELECTIVE
              !WRITE(IO%IU6,1241) "Midpoint calculations" 
            END IF

            CALL THREETOONE(T_INFO%NIONS,DYN%POSION,3*T_INFO%NIONS,Dimer_Dat%R0_NDa)
            CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F0_NDa)
            Dimer_Dat%EPOT0=EPOT
            CALL THREETOONE(T_INFO%NIOND,DYN%VEL,3*T_INFO%NIONS,Dimer_Dat%N_ND)
            IF (SUM(Dimer_Dat%N_ND**2)<1e-4) THEN
              IF (IO%IU6>=0) THEN
                WRITE(*,"(A,A)")'ERROR: missing or invalid vector defining dimer'
              ENDIF
              STOP
            END IF
            Dimer_Dat%N_ND=Dimer_Dat%N_ND/SQRT(SUM(Dimer_Dat%N_ND**2))
            Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Dat%delta_R*Dimer_Dat%N_ND
            Dimer_Dat%R2_NDa=Dimer_Dat%R0_NDa-Dimer_Dat%delta_R*Dimer_Dat%N_ND
            CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R1_NDa)

            Dimer_Dat%crisis=.FALSE.
            Dimer_Dat%engine_restart=.TRUE.
            operation=6
          END IF 

          Dimer_Dat%crisis=.FALSE.
          
          IF (IO%IU6>=0) WRITE(IO%IU6,1242) maxForce

          !CALL dimer_read_input(Dimer_Inp,IO,Dimer_Dat)
          SELECT CASE (operation)
            CASE(0) !c translation - midpoint
              !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Midpoint calculations" 
              CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F1_NDa)
              CALL dimer_translation(T_INFO,Dimer_Dat,Dimer_Inp)
              CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R0_NDa)
              operation=1

            CASE(1) !c translation - endpoint
              !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Curvature calculation" 
              Dimer_Dat%EPOT0=EPOT
              CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F0_NDa)
                Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Dat%delta_R*Dimer_Dat%N_ND
                Dimer_Dat%R2_NDa=Dimer_Dat%R0_NDa-Dimer_Dat%delta_R*Dimer_Dat%N_ND
              CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R1_NDa)
              IF (Dimer_Inp%findiff==1) THEN
                operation=3
              ELSE
                operation=2
              ENDIF
            CASE(2)
              !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Curvature calculation" 
              Dimer_Dat%EPOT1=EPOT
              CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F1_NDa)
              CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R2_NDa)
              operation=3
            CASE(3) !c rotation
              IF (Dimer_Inp%findiff==1) THEN
                Dimer_Dat%EPOT1=EPOT
                CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F1_NDa)
                Dimer_Dat%F2_NDa=2*Dimer_Dat%F0_NDa-Dimer_Dat%F1_NDa
                Dimer_Dat%EPOT2=2*Dimer_Dat%EPOT0-Dimer_Dat%EPOT1- &
                  & Dimer_Dat%delta_R*SUM((Dimer_Dat%F1_NDa-Dimer_Dat%F0_NDa)*Dimer_Dat%N_ND)
                operation=5
              ELSE
                Dimer_Dat%EPOT2=EPOT
                CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F2_NDa)
                operation=4
              ENDIF
              Dimer_Dat%FN=(Dimer_Dat%F1_NDa-Dimer_Dat%F2_NDa)
              Dimer_Dat%FN=Dimer_Dat%FN-SUM(Dimer_Dat%FN*Dimer_Dat%N_ND)*Dimer_Dat%N_ND

              IF (Dimer_Inp%LIDM_SELECTIVE) THEN
                DO i=1,3*T_INFO%NIONS
                  IF (ABS(Dimer_Dat%N_ND(i))<1e-6)   Dimer_Dat%FN(i)=0._q            
                ENDDO
              ENDIF

              Dimer_Dat%THETA=Dimer_Dat%FN/SQRT(SUM(Dimer_Dat%FN**2))
              Dimer_Dat%curve=0.5*SUM((Dimer_Dat%F2_NDa-Dimer_Dat%F1_NDa)*Dimer_Dat%N_ND)/Dimer_Dat%delta_R
              CALL vib_freq(T_INFO,Dimer_Dat,IO)
              IF (IO%IU6>=0) WRITE(IO%IU6,1243) Dimer_Dat%curve     
              !IF (IO%IU0>=0) write(*,*) 'tcurve0',Dimer_Dat%curve
              Dimer_Dat%dcurve_psi0=-SUM(Dimer_Dat%FN*Dimer_Dat%THETA)/(Dimer_Dat%delta_R)
              !alpha=45._q/180._q*PI
              !alpha=0.5*ATAN(-Dimer_Dat%dcurve_psi0/(2*ABS(Dimer_Dat%curve)))
              !alpha=0.5*ATAN(-Dimer_Dat%dcurve_psi0/(2*(Dimer_Dat%curve)))
              alpha=0.5*ATAN(SUM((Dimer_Dat%F1_NDa-Dimer_Dat%F2_NDa)*Dimer_Dat%THETA)/(2*Dimer_Dat%delta_R)/&
              & ABS(Dimer_Dat%curve))
              IF (IO%IU6>=0) WRITE(IO%IU6,1244) alpha*180./PI
             ! if (alpha>0.3) alpha=0.3
             ! write(*,*) 'alpha_',alpha

              IF (Dimer_Dat%curve>0._q) THEN
                !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Uphill translation of dimer" 
                Dimer_Dat%crisis=.TRUE.
                CALL dimer_translation_crisis(T_INFO,Dimer_Dat,Dimer_Inp)
                !!!CALL dimer_translation(T_INFO,Dimer_Dat,Dimer_Inp)
                Dimer_Dat%engine_restart=.TRUE.
                CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R0_NDa)
                operation=1
              ELSE IF (ABS(alpha) .GE. Dimer_Inp%minrot)  THEN
                !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Dimer rotation" 
                CALL dimer_rotation(Dimer_Dat,alpha)
                CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R1_NDa)
              ELSE
                !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Dimer rotation" 
                IF (IO%IU6>=0) write(IO%IU6,1245) 
                alpha=0._q
                CALL dimer_translation_curvature(T_INFO,Dimer_Dat,Dimer_Inp,DYN)
                CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R1_NDa)
                operation=0
              ENDIF
            CASE(4)
              !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Dimer rotation" 
              CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F1_NDa)
              CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R2_NDa)
              operation=5
            CASE(5)
              !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Trial translation of dimer" 
              IF (Dimer_Inp%findiff==1) THEN
                CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F1_NDa)
                Dimer_Dat%F2_NDa=2*Dimer_Dat%F0_NDa-Dimer_Dat%F1_NDa
              ELSE
                CALL THREETOONE(T_INFO%NIONS,FORCES,3*T_INFO%NIONS,Dimer_Dat%F2_NDa)
              ENDIF
              Dimer_Dat%dcurve_psi1=SUM((Dimer_Dat%F2_NDa-Dimer_Dat%F1_NDa)*Dimer_Dat%THETA)/(Dimer_Dat%delta_R)
              Dimer_Dat%curve1=0.5*SUM((Dimer_Dat%F2_NDa-Dimer_Dat%F1_NDa)*Dimer_Dat%N_ND)/Dimer_Dat%delta_R
              CALL vib_freq(T_INFO,Dimer_Dat,IO)
              ! IF (IO%IU0>=0) write(*,*) 'tcurve1',Dimer_Dat%curve1
              CALL dimer_rotation(Dimer_Dat,-alpha)   !c rotate back
              CALL fourier_coefs(Dimer_Dat,alpha,a0,a1,b1)
              !IF (IO%IU0>=0) write(*,*) 'a0,a1,b1',a0,a1,b1
              IF (ABS(a1)>1e-5) THEN
                alpha=0.5*ATAN(b1/a1)
              ELSE
                alpha=PI/4._q
              ENDIF
            
              IF (ABS(alpha) .GE. Dimer_Inp%minrot)  THEN
                Dimer_Dat%curve1=a0/2._q+a1*COS(2*alpha)+b1*SIN(2*alpha)
                IF (Dimer_Dat%curve1>Dimer_Dat%curve) THEN
                  alpha=alpha-SIGN(0.5_q*PI,alpha)
                  Dimer_Dat%curve1=a0/2._q+a1*COS(2*alpha)+b1*SIN(2*alpha)
                END IF
                IF (IO%IU6>=0) WRITE(IO%IU6,1246) alpha*180./PI
                !IF (IO%IU0>=0) write(*,*) 'alpha',alpha
                Dimer_Dat%curve=Dimer_Dat%curve1
                IF (IO%IU6>=0) WRITE(IO%IU6,1243) Dimer_Dat%curve
                !IF (IO%IU0>=0) write(*,*) 'tcurve2',Dimer_Dat%curve
                CALL dimer_rotation(Dimer_Dat,alpha)
              ELSE
                IF (IO%IU6>=0) write(IO%IU6,1245) 
              ENDIF

              !IF (Dimer_Inp%dimerdynamics) THEN
              !   write(*,*) 'Dimer_Dat%R0_NDa1',Dimer_Dat%R0_NDa
              !   CALL dimer_dynamics(DYN,T_INFO,INFO,LATT_CUR,Dimer_Dat,Dimer_Inp,IO)
              !   write(*,*) 'Dimer_Dat%R0_NDa2',Dimer_Dat%R0_NDa
              !   operation=1
              !ELSE
                IF (Dimer_Dat%curve>0._q) THEN
                  Dimer_Dat%crisis=.TRUE.
                ELSE
                  Dimer_Dat%crisis=.FALSE.
                ENDIF

                IF (Dimer_Dat%crisis) THEN
                  !IF (IO%IU6>=0) WRITE(IO%IU6,1241) "Uphill translation of dimer" 
                  CALL dimer_translation_crisis(T_INFO,Dimer_Dat,Dimer_Inp)
                  Dimer_Dat%engine_restart=.TRUE.
                  CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R0_NDa)
                  operation=1
                ELSE
                  CALL dimer_translation_curvature(T_INFO,Dimer_Dat,Dimer_Inp,DYN)
                  CALL ONETOTHREE(T_INFO%NIONS,DYN%POSION,Dimer_Dat%R1_NDa)
                  operation=0
                ENDIF
              !ENDIF 
          END SELECT

          IF (operation==6) THEN
            IF (Dimer_Inp%findiff==2) THEN
              operation=2
            ELSE 
              operation=3
            ENDIF
          ENDIF


          DYN%POSION=MATMUL(TRANSPOSE(LATT_CUR%B),DYN%POSION)
          DO i=1,3
            DO j=1,T_INFO%NIONS
              DO 
                IF (DYN%POSION(i,j)<0._q) THEN
                  DYN%POSION(i,j)=DYN%POSION(i,j)+1._q
                ELSE IF (DYN%POSION(i,j)>=1._q) THEN
                   DYN%POSION(i,j)=DYN%POSION(i,j)-1._q
                ELSE
                  EXIT
                ENDIF  
              ENDDO 
            ENDDO
          ENDDO
          !DYN%POSIOC=DYN%POSION
          CALL ONETOTHREE(T_INFO%NIONS,DYN%VEL,Dimer_Dat%N_ND)

          !IF (IO%IU6>=0) THEN
          !  DO i=1,T_INFO%NIONS
          !    write(1212,*) Dimer_Dat%N_ND(3*i-2),Dimer_Dat%N_ND(3*i-1),Dimer_Dat%N_ND(3*i)
          !  ENDDO
          !ENDIF
        END SUBROUTINE dimer

        SUBROUTINE vib_freq(T_INFO,Dimer_Dat,IO)
          !c compute vibrational frequency for a given dimer direction
          TYPE(type_info) :: T_INFO
          TYPE (in_struct) :: IO
          TYPE(dimer_data) :: Dimer_Dat
          REAL(q) :: vmode(3*T_INFO%NIONS)
          REAL(q) :: FACTOR,EVAL,W,vnorm
          INTEGER :: NI,NT
          REAL(q),PARAMETER :: PLANK=6.626075E-34
          REAL(q),PARAMETER :: C= 2.99792458E10


          FACTOR=SQRT(EVTOJ/((1E-10)**2)/AMTOKG)
          !FACT=(EVTOJ/AMTOKG**0.5)*1e-10

          vmode=0._q
          !vmode=Dimer_Dat%N_ND

          vnorm=0._q
          NI=1
          DO NT=1,T_INFO%NTYP
            DO NI=NI,T_INFO%NITYP(NT)+NI-1
!               vmode(3*NI-2)=Dimer_Dat%N_ND(3*NI-2)/SQRT(T_INFO%POMASS(NT))
!               vmode(3*NI-1)=Dimer_Dat%N_ND(3*NI-1)/SQRT(T_INFO%POMASS(NT))
!               vmode(3*NI)=Dimer_Dat%N_ND(3*NI)/SQRT(T_INFO%POMASS(NT))
              vmode(3*NI-2)=Dimer_Dat%N_ND(3*NI-2)/(T_INFO%POMASS(NT))
              vmode(3*NI-1)=Dimer_Dat%N_ND(3*NI-1)/(T_INFO%POMASS(NT))
              vmode(3*NI)=Dimer_Dat%N_ND(3*NI)/(T_INFO%POMASS(NT))
              vnorm=vnorm+vmode(3*NI-2)**2+vmode(3*NI-1)**2+vmode(3*NI)**2
            ENDDO
          ENDDO
 
          !vmode=vmode/SQRT(vnorm)
          !vmode=vmode*FACT
          EVAL=0.5*SUM((Dimer_Dat%F2_NDa-Dimer_Dat%F1_NDa)*vmode)/Dimer_Dat%delta_R
          W=FACTOR*SQRT(ABS(EVAL))
          IF (IO%IU6>=0 .AND. IO%NWRITE==3) THEN
            IF (EVAL.GE.0) THEN
              WRITE(IO%IU6,'(" f  =",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
              W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
            ELSE
              WRITE(IO%IU6,'(" f/i=",F12.6," THz ",F12.6," 2PiTHz",F12.6," cm-1 ",F12.6," meV")') &
               W/(1E12*2*PI),W/(1E12),W/(C*PI*2),W*1000*PLANK/EVTOJ/2/PI
            END IF
          ENDIF

        END SUBROUTINE vib_freq

        SUBROUTINE dimer_rotation(Dimer_Dat,alpha)
         !c rotate around midpoint in the N_ND THETA plane
          TYPE(dimer_data) :: Dimer_Dat
          REAL(q) :: alpha

          Dimer_Dat%R1_NDa=Dimer_Dat%N_ND*COS(alpha)+Dimer_Dat%THETA*SIN(alpha)
          Dimer_Dat%THETA=-Dimer_Dat%N_ND*SIN(alpha)+Dimer_Dat%THETA*COS(alpha)
          !!!Dimer_Dat%THETA=Dimer_Dat%THETA/SQRT(SUM(Dimer_Dat%THETA**2))
          Dimer_Dat%N_ND=Dimer_Dat%R1_NDa
          Dimer_Dat%N_ND=Dimer_Dat%N_ND/SQRT(SUM(Dimer_Dat%N_ND**2))
          Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Dat%delta_R*Dimer_Dat%N_ND
          Dimer_Dat%R2_NDa=Dimer_Dat%R0_NDa-Dimer_Dat%delta_R*Dimer_Dat%N_ND
        END SUBROUTINE dimer_rotation

        SUBROUTINE fourier_coefs(Dimer_Dat,alpha,a0,a1,b1)
          TYPE(dimer_data) :: Dimer_Dat
          REAL(q) :: alpha,a0,a1,b1

          a0=0._q;a1=0._q;b1=0._q
          a1=(Dimer_Dat%dcurve_psi0*COS(2*alpha)-Dimer_Dat%dcurve_psi1)/(2*SIN(2*alpha))
          b1=0.5*Dimer_Dat%dcurve_psi0
          a0=2._q*(Dimer_Dat%curve-a1)
          !write(*,*) 'fourier2:',a0,a1,b1

          !a1=(Dimer_Dat%curve-Dimer_Dat%curve1+0.5_q*Dimer_Dat%dcurve_psi0*SIN(2*alpha))/&
          !&  (1._q-COS(2*alpha))
          !b1=0.5*Dimer_Dat%dcurve_psi0
          !a0=2._q*(Dimer_Dat%curve-a1)
          !write(*,*) 'fourier1:',a0,a1,b1

        END SUBROUTINE fourier_coefs

        SUBROUTINE dimer_translation_curvature(T_INFO,Dimer_Dat,Dimer_Inp,DYN)
          !c determine direction for translation, make small
          !c step to calculate curvature, use Dimer_Dat%R1_NDa
          !c as a temporary storage
          TYPE(type_info) :: T_INFO
          TYPE(dynamics) :: DYN
          TYPE(dimer_data) :: Dimer_Dat
          TYPE(dimer_input) :: Dimer_Inp

          !IF (Dimer_Dat%curve>0._q) THEN
          IF (Dimer_Dat%crisis) THEN
            !write(*,*) 'crisis!'
            Dimer_Dat%F0_dagger=-SUM(Dimer_Dat%F0_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          ELSE
            Dimer_Dat%F0_dagger=Dimer_Dat%F0_NDa-2*SUM(Dimer_Dat%F0_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          END IF

          !write(*,*) 'Dimer_Inp%engine',Dimer_Inp%engine
          SELECT CASE(Dimer_Inp%engine)
            CASE(1) !c steepest descent
              Dimer_Dat%N_T=Dimer_Dat%F0_dagger/SUM(Dimer_Dat%F0_dagger**2)**0.5
            CASE(2) !c conjugate gradient
              IF (Dimer_Dat%curve>0._q) THEN
                 !write(*,*) 'SD'
                Dimer_Dat%N_T=Dimer_Dat%F0_dagger/SUM(Dimer_Dat%F0_dagger**2)**0.5
              ELSE
                CALL conjugate_gradient(Dimer_Dat,Dimer_Inp,T_INFO)
              ENDIF
          END SELECT
          
          Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Inp%stepsize*Dimer_Dat%N_T

        END SUBROUTINE dimer_translation_curvature

        SUBROUTINE dimer_translation(T_INFO,Dimer_Dat,Dimer_Inp)
          !c calculate curvature, translate all points
          !c of dimer
          TYPE(type_info) :: T_INFO
          TYPE(dimer_data) :: Dimer_Dat
          TYPE(dimer_input) :: Dimer_Inp
          REAL(q) :: F1_dagger(3*T_INFO%NIONS)
          REAL(q) ::step

          step=0._q;F1_dagger=0._q

          IF (Dimer_Dat%crisis) THEN
            F1_dagger=-SUM(Dimer_Dat%F1_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          ELSE
            F1_dagger=Dimer_Dat%F1_NDa-2*SUM(Dimer_Dat%F1_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          END IF

          IF (Dimer_Dat%crisis) THEN
            step=-SUM(Dimer_Dat%F0_dagger*Dimer_Dat%N_T)/Dimer_Dat%curve
          ELSE
            step=-0.5*SUM((Dimer_Dat%F0_dagger+F1_dagger)*Dimer_Dat%N_T)&
            &                /(SUM((F1_dagger-Dimer_Dat%F0_dagger)*Dimer_Dat%N_T)/Dimer_Inp%stepsize) +&
            &                Dimer_Inp%stepsize/2
          ENDIF
          !write(*,*) 'step_',step
          !write(*,*) 'test',(SUM((F1_dagger-Dimer_Dat%F0_dagger)*Dimer_Dat%N_T))/Dimer_Inp%stepsize
         ! IF ((SUM((F1_dagger-Dimer_Dat%F0_dagger)*Dimer_Dat%N_T))<0._q) Dimer_Dat%crisis=.False.
          IF (ABS(step)>Dimer_Inp%maxstep) step=SIGN(Dimer_Inp%maxstep,step)
          IF (Dimer_Dat%crisis) step=SIGN(Dimer_Inp%maxstep,step)

          !write(*,*) 'step',step,Dimer_Inp%stepsize
          IF (Dimer_Inp%engine==3) THEN
            Dimer_Dat%R0_NDa=Dimer_Dat%R2_NDa+(step)*Dimer_Dat%N_T
          ELSE
            Dimer_Dat%R0_NDa=Dimer_Dat%R0_NDa+(step)*Dimer_Dat%N_T
          ENDIF
          Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Dat%delta_R*Dimer_Dat%N_ND
          Dimer_Dat%R2_NDa=Dimer_Dat%R0_NDa-Dimer_Dat%delta_R*Dimer_Dat%N_ND
        END SUBROUTINE dimer_translation

        SUBROUTINE dimer_translation_crisis(T_INFO,Dimer_Dat,Dimer_Inp)
          !c calculate curvature, translate all points
          !c of dimer
          TYPE(type_info) :: T_INFO
          TYPE(dimer_data) :: Dimer_Dat
          TYPE(dimer_input) :: Dimer_Inp
          REAL(q) ::step

          step=0._q
          
          !write(*,*) 'dimer_translation_crisis:'
          Dimer_Dat%F0_dagger=-SUM(Dimer_Dat%F0_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          Dimer_Dat%N_T=Dimer_Dat%F0_dagger/SUM(Dimer_Dat%F0_dagger**2)**0.5
          step=-SUM(Dimer_Dat%F0_dagger*Dimer_Dat%N_T)/Dimer_Dat%curve
          !write(*,*) 'step_',step
          !step=SIGN(Dimer_Inp%maxstep,step)
          !write(*,*) 'step',step,Dimer_Inp%stepsize
          !Dimer_Dat%R0_NDa=Dimer_Dat%R0_NDa+(step)*Dimer_Dat%N_T
          !!Dimer_Dat%R0_NDa=Dimer_Dat%R0_NDa+Dimer_Inp%maxstep*Dimer_Dat%F0_NDa

          Dimer_Dat%R0_NDa=Dimer_Dat%R0_NDa-Dimer_Inp%maxstep*SUM(Dimer_Dat%F0_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
          !Dimer_Dat%R0_NDa=Dimer_Dat%R0_NDa-Dimer_Inp%maxstep*SUM(Dimer_Dat%F0_NDa*Dimer_Dat%N_ND)*Dimer_Dat%N_ND/SQRT(SUM(Dimer_Dat%F0_NDa**2))
          Dimer_Dat%R1_NDa=Dimer_Dat%R0_NDa+Dimer_Dat%delta_R*Dimer_Dat%N_ND
          Dimer_Dat%R2_NDa=Dimer_Dat%R0_NDa-Dimer_Dat%delta_R*Dimer_Dat%N_ND
        END SUBROUTINE dimer_translation_crisis


        SUBROUTINE conjugate_gradient(Dimer_Dat,Dimer_Inp,T_INFO)
          TYPE(type_info) :: T_INFO
          TYPE(dimer_data) :: Dimer_Dat
          TYPE(dimer_input) :: Dimer_Inp
          REAL(q) :: gamma
          !REAL(q),POINTER :: FORCE_old(:),NT_old(:)
          REAL(q),ALLOCATABLE,SAVE :: FORCE_old(:),NT_old(:)
          INTEGER,SAVE :: counter=1

          !write(*,*) 'CG:',counter
          gamma=0._q
          IF (counter==1) THEN
            ALLOCATE(FORCE_old(3*T_INFO%NIONS),NT_old(3*T_INFO%NIONS))
            FORCE_old=0._q;NT_old=0._q
          ENDIF

          IF (counter>1 .AND. (.NOT. Dimer_Dat%engine_restart)) THEN
           ! FORCE_old=FORCE_old-SUM(FORCE_old*Dimer_Dat%N_ND)*Dimer_Dat%N_ND
            IF (Dimer_Inp%Fletcher_Reeves) THEN
              !write(*,*) 'Fletcher_Reeves'
              gamma=SUM(Dimer_Dat%F0_dagger**2)/SUM(FORCE_old**2)
            ELSE
              !write(*,*) 'Polak Ribiere'
              gamma=SUM((Dimer_Dat%F0_dagger-FORCE_old)*Dimer_Dat%F0_dagger)/SUM(FORCE_old**2)
            ENDIF

            !write(*,*) 'gamma',gamma
            IF (ABS(gamma)>10._q) gamma=0._q
            Dimer_Dat%N_T=Dimer_Dat%F0_dagger+gamma*NT_old
          ELSE
            !write(*,*) 'CGrestart'
            Dimer_Dat%N_T=Dimer_Dat%F0_dagger
            NT_old=0._q;FORCE_old=0._q
          END IF
          Dimer_Dat%N_T=Dimer_Dat%N_T/SUM(Dimer_Dat%N_T**2)**0.5
          FORCE_old=Dimer_Dat%F0_dagger
          NT_old=Dimer_Dat%N_T

          counter=counter+1
          IF (MOD(counter,Dimer_Inp%restartCG)==0) THEN
            Dimer_Dat%engine_restart=.FALSE.
          ELSE
            Dimer_Dat%engine_restart=.FALSE.
          ENDIF
        END SUBROUTINE conjugate_gradient


        SUBROUTINE dimer_read_input(Dimer_Inp,IO,Dimer_Dat)
        !c read vector from DIMERDIRECTION - no longer needed as this
        !c is read directly from POSCAR
          TYPE(dimer_input) :: Dimer_Inp
          TYPE (in_struct) :: IO
          TYPE(dimer_data) :: Dimer_Dat
          INTEGER IDUM,N,IERR
          LOGICAL :: LOPEN,LDUM
          REAL(q) :: MNORM,RDUM
          COMPLEX(q) :: CDUM
          CHARACTER*1 :: CHARAC

          LOPEN=.FALSE.
          OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
          !c SWITCH
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'SWITCH','=','#',';','I', &
                Dimer_Inp%SWITCH,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%SWITCH=2
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%SWITCH=2
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''SWITCH'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%SWITCH<1 .OR. Dimer_Inp%SWITCH>4) Dimer_Inp%SWITCH=2

          !c finite differences (1-forward,2-central)
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'findiff','=','#',';','I', &
                Dimer_Inp%findiff,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%findiff=1
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%findiff=1
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''FINDIFF'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%findiff<1 .OR. Dimer_Inp%findiff>2) Dimer_Inp%findiff=1

          !c distance between images (A)
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'DIMER_DIST','=','#',';','F', &
                IDUM,Dimer_Dat%delta_R,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Dat%delta_R=1e-2 
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Dat%delta_R=1e-2 
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''DIMER_DIST'' from file INCAR'
          ENDIF
          IF (Dimer_Dat%delta_R<0._q) Dimer_Dat%delta_R=1e-2

          !c step size for the numerical differenciation
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'STEP_SIZE','=','#',';','F', &
                IDUM,Dimer_Inp%stepsize,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%stepsize=5e-3
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%stepsize=5e-3
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''STEP_SIZE'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%stepsize<0._q) Dimer_Inp%stepsize=5e-3

         !c maximal step length for tranlsation
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'STEP_MAX','=','#',';','F', &
                IDUM,Dimer_Inp%maxstep,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%maxstep=1e-1
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%maxstep=1e-1
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''STEP_MAX'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%maxstep<0._q) Dimer_Inp%maxstep=1e-1

          !c minimal allowed rotation of dimer
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'MINROT','=','#',';','F', &
                IDUM,Dimer_Inp%minrot,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%minrot=1e-2
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%minrot=1e-2
            IF (IO%IU0>=0) &
              WRITE(IO%IU0,*)'Error reading item ''MINROT'' from file INCAR'
          ENDIF
          Dimer_Inp%minrot=ABS(Dimer_Inp%minrot)

         !c 
          CALL RDATAB(LOPEN,INCAR,IO%IU5,'ENGINE','=','#',';','I', &
                Dimer_Inp%engine,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%engine=2
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%engine=2
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''ENGINE'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%engine<1 .OR. Dimer_Inp%engine>2) Dimer_Inp%engine=2

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'RESTARTCG','=','#',';','I', &
                Dimer_Inp%restartCG,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%restartCG=10
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%restartCG=10
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''RESTARTCG'' from file INCAR'
          ENDIF
          IF (Dimer_Inp%restartCG<1) Dimer_Inp%restartCG=10

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'FLETCHER_REEVES','=','#',';','L', &
                IDUM,RDUM,CDUM,Dimer_Inp%FLETCHER_REEVES,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%FLETCHER_REEVES=.FALSE.
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%FLETCHER_REEVES=.FALSE.
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''FLETCHER_REEVES'' from file INCAR'
          ENDIF

          CALL RDATAB(LOPEN,INCAR,IO%IU5,'LIDM_SELECTIVE','=','#',';','L', &
                IDUM,RDUM,CDUM,Dimer_Inp%LIDM_SELECTIVE,CHARAC,N,1,IERR)
          IF (IERR==3) Dimer_Inp%LIDM_SELECTIVE=.FALSE.
          IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            Dimer_Inp%LIDM_SELECTIVE=.FALSE.
            IF (IO%IU0>=0) &
               WRITE(IO%IU0,*)'Error reading item ''LIDM_SELECTIVE'' from file INCAR'
          ENDIF

          CLOSE(IO%IU5)
        END SUBROUTINE dimer_read_input  

        SUBROUTINE THREETOONE(N,TVECTOR,M,OVECTOR)
        !c transforms three-column format into
        !c one-column vector
          USE prec
          INTEGER :: i,j,N,M         !M>=N!!!
          REAL(q) :: TVECTOR(3,N)    ! vector in three-column format
          REAL(q) :: OVECTOR(M)   ! vector in one-column format

          OVECTOR=0._q
          DO i=1,N
            DO j=1,3
              OVECTOR(3*i+j-3)=TVECTOR(j,i)
            ENDDO
          ENDDO
        END SUBROUTINE

        SUBROUTINE ONETOTHREE(N,TVECTOR,OVECTOR)
        !c transforms one-column format into
        !c three-column vector
          USE prec
          INTEGER :: i,j,N
          REAL(q) :: TVECTOR(3,N)    ! vector in three-column format
          REAL(q) :: OVECTOR(3*N)   ! vector in one-column format

          TVECTOR=0._q
          DO i=1,N
            DO j=1,3
              TVECTOR(j,i)=OVECTOR(3*i+j-3)
            ENDDO
          ENDDO
        END SUBROUTINE

      END MODULE dimer_heyden





























