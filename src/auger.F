#include "symbol.inc"
#define MPI_barrier_after_bcast
#undef debug

      MODULE mytimer
        USE prec
        USE wave_high

        PUBLIC :: CREATE_TIMER
        PUBLIC :: GET_TIMER_INDEX
        PUBLIC :: START_TIMER
        PUBLIC :: STOP_TIMER
        PUBLIC :: PRINT_TIMERS

        PRIVATE
        INTEGER, PARAMETER :: MAXTIMERS=100
        INTEGER, PRIVATE   :: USEDTIMERS=0
        CHARACTER(LEN=6)   :: TIMER_TAG(MAXTIMERS)
        REAL(q)            :: CPU_START(MAXTIMERS),CPU_SUM(MAXTIMERS)

        INTERFACE START_TIMER
           MODULE PROCEDURE START_TIMER_TAG,START_TIMER_INDEX
        END INTERFACE START_TIMER

        INTERFACE STOP_TIMER
           MODULE PROCEDURE STOP_TIMER_TAG,STOP_TIMER_INDEX
        END INTERFACE STOP_TIMER

      CONTAINS
        ! Creates a timer with name TAG and associates with an index
        ! MUST BE CALLED BY ALL CPUS! (otherwise different cpus have different timers)
        FUNCTION CREATE_TIMER(TAG)
          IMPLICIT NONE
          INTEGER :: CREATE_TIMER
          CHARACTER(LEN=*),INTENT(IN) :: TAG
          INTEGER :: J
          !
          USEDTIMERS=USEDTIMERS+1
          IF(USEDTIMERS>MAXTIMERS)THEN
             WRITE(*,*) 'Maximum number of timers exceeded'
             STOP
          END IF
          DO J=1,USEDTIMERS-1
             IF(TRIM(TIMER_TAG(J))==TRIM(TAG))THEN
                WRITE(*,*) 'Timer tag already in use'
                STOP
             END IF             
          END DO
          CREATE_TIMER=USEDTIMERS
          TIMER_TAG(USEDTIMERS)=TRIM(TAG)
          CPU_SUM(USEDTIMERS)=0
        END FUNCTION CREATE_TIMER

        FUNCTION GET_TIMER_INDEX(TAG)
          INTEGER :: GET_TIMER_INDEX,J
          CHARACTER(LEN=*),INTENT(IN) :: TAG
          DO J=1,USEDTIMERS
             IF(TRIM(TIMER_TAG(J))==TRIM(TAG))THEN
                GET_TIMER_INDEX=J
                RETURN
             END IF
          END DO
          GET_TIMER_INDEX=-1
        END FUNCTION GET_TIMER_INDEX

        SUBROUTINE START_TIMER_TAG(TAG)
          IMPLICIT NONE
          CHARACTER(LEN=*) :: TAG
          INTEGER :: J
          REAL(q) :: TIME
          !
          DO J=1,USEDTIMERS
             IF(TRIM(TAG)==TRIM(TIMER_TAG(J)))EXIT
          END DO
          CALL CPU_TIME(TIME)
          CPU_START(J)=TIME          
        END SUBROUTINE START_TIMER_TAG

        SUBROUTINE STOP_TIMER_TAG(TAG)
          IMPLICIT NONE
          CHARACTER(LEN=*) :: TAG
          INTEGER :: J
          REAL(q) :: TIME
          !
          DO J=1,USEDTIMERS
             IF(TRIM(TAG)==TRIM(TIMER_TAG(J)))EXIT
          END DO
          CALL CPU_TIME(TIME)
          CPU_SUM(J)=CPU_SUM(J)+(TIME-CPU_START(J))
          CPU_START(J)=0
        END SUBROUTINE STOP_TIMER_TAG

        SUBROUTINE START_TIMER_INDEX(NT)
          IMPLICIT NONE
          INTEGER :: NT
          REAL(q) :: TIME
          !
          CALL CPU_TIME(TIME)
          CPU_START(NT)=TIME          
        END SUBROUTINE START_TIMER_INDEX

        SUBROUTINE STOP_TIMER_INDEX(NT)
          IMPLICIT NONE
          INTEGER :: NT
          REAL(q) :: TIME
          !
          CALL CPU_TIME(TIME)
          CPU_SUM(NT)=CPU_SUM(NT)+(TIME-CPU_START(NT))
          CPU_START(NT)=0
        END SUBROUTINE STOP_TIMER_INDEX

        SUBROUTINE PRINT_TIMERS(WDES)
          IMPLICIT NONE
          TYPE(wavedes) :: WDES
          INTEGER :: J
          LOGICAL :: LDUMP
          !
          LDUMP=.TRUE.
#ifdef MPI
          IF(WDES%COMM%NODE_ME==WDES%COMM%IONODE) THEN
             LDUMP=.TRUE.
          ELSE
             LDUMP=.FALSE.
          ENDIF
#endif
          IF(LDUMP)THEN
             WRITE(*,'(A10,A12)') 'TAG','CPU TIME'
             WRITE(*,'(''----------------------'')')
          END IF
          DO J=1,USEDTIMERS
             IF(TIMER_TAG(J)/=' ')THEN
                CALLMPI( M_sum_d(WDES%COMM_INTER, CPU_SUM(J), 1) )
                CALLMPI( M_sum_d(WDES%COMM_KINTER, CPU_SUM(J), 1) )
                IF(LDUMP) WRITE(*,'(A10,F12.2)') TIMER_TAG(J),CPU_SUM(J)
             END IF
          END DO
          IF(LDUMP) WRITE(*,'(''----------------------'')')
        END SUBROUTINE PRINT_TIMERS

      END MODULE mytimer


!***********************************************************************
!***********************************************************************
      MODULE auger
      USE prec
      USE wpot
      USE util
      USE mytimer
      USE base

      IMPLICIT NONE

      REAL(q), PRIVATE, SAVE :: AUGER_OCCUPY_THRESHOLD=1.E-16_q
      REAL(q) :: AUGER_MU,AUGER_SIGMA,AUGER_EFERMI,AUGER_EVBHI,AUGER_ECBLO
      TYPE(wpothandle), POINTER, SAVE :: AUGER_WPOTH => NULL() ! AUGER_WPOTH stores the screened Coulomb potential at each q-points
      REAL(q), PARAMETER :: HBAR=6.58211928E-16_q  ! hbar in [eV s]
      TYPE (in_struct) :: AUGER_IO

      ! input parameters read from INCAR
      ! Width of the energy conservation window function AUGER_EWIDTH=dE/2
      REAL(q), PRIVATE :: AUGER_EWIDTH
      ! choose one of AUGER_EDENS and AUGER_HDENS, the other one is determined automatically
      REAL(q) :: AUGER_EDENS ! density of electrons in CB in [cm^-3]
      REAL(q) :: AUGER_HDENS ! density of holes in VB in [cm^-3]
      ! temperature
      REAL(q) :: AUGER_TEMP
      ! switch to turn calculation of Auger rate on/off
      LOGICAL :: LAUGER,LAUGER_EHH,LAUGER_EEH
      ! factor to determine electron/hole occupation threshold 
      REAL(q) :: AUGER_OCC_FAC_EEH,AUGER_OCC_FAC_EHH
      !
      REAL(q) :: AUGER_EMIN_EEH(4),AUGER_EMAX_EEH(4)
      INTEGER :: AUGER_BMIN_EEH(4),AUGER_BMAX_EEH(4)

      REAL(q) :: AUGER_EMIN_EHH(4),AUGER_EMAX_EHH(4)
      INTEGER :: AUGER_BMIN_EHH(4),AUGER_BMAX_EHH(4)

      LOGICAL, PRIVATE,SAVE :: LDUMP_AUGER=.FALSE.

      LOGICAL :: LAUGER_COLLECT,LAUGER_DHDK,LAUGER_JIT
      INTEGER :: NCSHMEM

    CONTAINS


      SUBROUTINE ALLOCATE_WAVEFUN(WF,WDES,ALLOC_CW,ALLOC_CR,ALLOC_CPROJ)
        IMPLICIT NONE
        TYPE (wavefun) WF
        TYPE (wavedes), TARGET :: WDES
        LOGICAL :: ALLOC_CW,ALLOC_CR,ALLOC_CPROJ
        !
        INTEGER MPLWV

        NULLIFY(WF%CPTWFP,WF%CR,WF%CPROJ)
        IF(ALLOC_CW)THEN
           ALLOCATE(WF%CPTWFP(WDES%NRPLWV,WDES%NBANDS,WDES%NKPTS))
        END IF
        IF(ALLOC_CR)THEN
           MPLWV=WDES%GRID%MPLWV*WDES%NRSPINORS
           ALLOCATE(WF%CR(MPLWV,WDES%NBANDS,WDES%NKPTS))
        END IF
        IF(ALLOC_CPROJ)THEN
           ALLOCATE(WF%CPROJ(WDES%NPROD,WDES%NBANDS,WDES%NKPTS))
        END IF
      END SUBROUTINE ALLOCATE_WAVEFUN

      SUBROUTINE DEALLOCATE_WAVEFUN(WF)
        IMPLICIT NONE
        TYPE (wavefun) WF
        
        IF(ASSOCIATED(WF%CPTWFP)) DEALLOCATE(WF%CPTWFP)
        IF(ASSOCIATED(WF%CR))     DEALLOCATE(WF%CR)
        IF(ASSOCIATED(WF%CPROJ))  DEALLOCATE(WF%CPROJ)
        NULLIFY(WF%CPTWFP,WF%CR,WF%CPROJ)
      END SUBROUTINE DEALLOCATE_WAVEFUN

#ifdef use_shmem
      SUBROUTINE ALLOCATE_WAVEFUN_SHMEM(WF,WDES,NCSHMEM,ADDRESS_CW,SHMID_CW,&
           ADDRESS_CR,SHMID_CR)
        USE c2f_interface
        USE vaspxml
        IMPLICIT NONE
        TYPE (wavefun) WF
        TYPE (wavedes), TARGET :: WDES
        INTEGER NCSHMEM
        TYPE(c_ptr), OPTIONAL :: ADDRESS_CW,ADDRESS_CR
        INTEGER(c_int), OPTIONAL :: SHMID_CW,SHMID_CR
        !
        INTEGER MPLWV
        INTEGER(c_size_t) size                   ! integer*8 has to be, see WV in getshmem.c
        INTEGER IER, I, N
        
        IF(PRESENT(ADDRESS_CW))THEN
           size=WDES%NBANDS*WDES%NKPTS*WDES%NRPLWV
           IF (MOD(WDES%COMM_INTER%NODE_ME-1,NCSHMEM).EQ.0) THEN
              ! only allocate the memory once for every group of NCSHMEM processes
              CALL getshmem(INT(8*2*size,KIND=c_size_t),shmid_cw)    ! it is a complex
              DO I=1,NCSHMEM-1
                 CALL M_send_i(WDES%COMM_INTER,WDES%COMM_INTER%NODE_ME+I,SHMID_CW,1)
              ENDDO
           ELSE
              I=(WDES%COMM_INTER%NODE_ME-1)/NCSHMEM
              I=I*NCSHMEM+1
              CALL M_recv_i(WDES%COMM_INTER,I,SHMID_CW,1)
           ENDIF
           CALLMPI( MPI_barrier(WDES%COMM_INTER%MPI_COMM, IER ))
           call attachshmem(shmid_cw, address_cw)
           call c_f_pointer(address_cw,WF%CPTWFP,[WDES%NRPLWV,WDES%NBANDS,WDES%NKPTS])
        END IF

        IF(PRESENT(ADDRESS_CR))THEN
           size=WDES%NBANDS*WDES%NKPTS*WDES%GRID%MPLWV*WDES%NRSPINORS
           IF (MOD(WDES%COMM_INTER%NODE_ME-1,NCSHMEM).EQ.0) THEN
              ! only allocate the memory once for every group of NCSHMEM processes
              CALL getshmem(INT(8*2*size,KIND=c_size_t),shmid_cr)    ! it is a complex
              DO I=1,NCSHMEM-1
                 CALL M_send_i(WDES%COMM_INTER,WDES%COMM_INTER%NODE_ME+I,SHMID_CR,1)
              ENDDO
           ELSE
              I=(WDES%COMM_INTER%NODE_ME-1)/NCSHMEM
              I=I*NCSHMEM+1
              CALL M_recv_i(WDES%COMM_INTER,I,SHMID_CR,1)
           ENDIF
           CALLMPI( MPI_barrier(WDES%COMM_INTER%MPI_COMM, IER ))
           call attachshmem(shmid_cr, address_cr)
           call c_f_pointer(address_cr,WF%CR,[WDES%GRID%MPLWV*WDES%NRSPINORS,WDES%NBANDS,WDES%NKPTS])
        END IF
        ALLOCATE(WF%CPROJ(WDES%NPROD,WDES%NBANDS,WDES%NKPTS))
        WF%WDES=>WDES

      END SUBROUTINE ALLOCATE_WAVEFUN_SHMEM
      

      SUBROUTINE DEALLOCATE_WAVEFUN_SHMEM(WF,NCSHMEM,ADDRESS_CW,SHMID_CW,&
           ADDRESS_CR,SHMID_CR)
        USE c2f_interface
        USE vaspxml
        IMPLICIT NONE
        TYPE (wavefun) WF
        INTEGER NCSHMEM
        TYPE(c_ptr), OPTIONAL :: address_cw, address_cr
        INTEGER(c_int), OPTIONAL :: shmid_cw, shmid_cr
        INTEGER IER

        IF(PRESENT(address_cw))THEN
           call detachshmem(address_cw)
           IF (MOD(WF%WDES%COMM_INTER%NODE_ME-1,NCSHMEM).EQ.0) THEN
              call destroyshmem(shmid_cw)
           ENDIF
           CALLMPI( MPI_barrier(WF%WDES%COMM_INTER%MPI_COMM, IER ))   
        END IF
        IF(PRESENT(address_cr))THEN
           call detachshmem(address_cr)
           IF (MOD(WF%WDES%COMM_INTER%NODE_ME-1,NCSHMEM).EQ.0) THEN
              call destroyshmem(shmid_cr)
           ENDIF
           CALLMPI( MPI_barrier(WF%WDES%COMM_INTER%MPI_COMM, IER ))   
        END IF
        IF(ASSOCIATED(WF%CPROJ)) DEALLOCATE(WF%CPROJ)
      END SUBROUTINE DEALLOCATE_WAVEFUN_SHMEM
#endif

      SUBROUTINE AUGER_READER(IO)
        USE base
        USE vaspxml
        IMPLICIT NONE
        TYPE(in_struct) IO
        ! local
        INTEGER IDUM, N, IERR
        REAL(q) RDUM
        COMPLEX(q) CDUM
        LOGICAL LOPEN,LDUM
        CHARACTER (1) :: SDUM

        LOPEN=.FALSE.
        OPEN(UNIT=IO%IU5,FILE=INCAR,STATUS='OLD')
        
        ! flag to switch on calculation of Auger rates
        LAUGER=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER','L',IDUM,RDUM,CDUM,LAUGER,SDUM,N)

        ! calculate EEH processes
        LAUGER_EEH=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER_EEH','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER_EEH,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER_EEH','L',IDUM,RDUM,CDUM,LAUGER_EEH,SDUM,N)

        ! calculate EHH processes
        LAUGER_EHH=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER_EHH','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER_EHH,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER_EHH','L',IDUM,RDUM,CDUM,LAUGER_EHH,SDUM,N)

        ! half(!)-width of rectangular energy window function
        ! i.e. (Theta_-AUGER_EWIDTH,+AUGER_EWIDTH)/(2*AUGER_EWIDTH)
        AUGER_EWIDTH=0.1_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EWIDTH','=','#',';','F', &
             &            IDUM,AUGER_EWIDTH,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EWIDTH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_EWIDTH','R',IDUM,AUGER_EWIDTH,CDUM,LDUM,SDUM,N)

        ! Either AUGER_EDENS or AUGER_HDENS can be specified, but not both
        ! density of conduction band electrons in [cm^-3]
        AUGER_EDENS=-1._q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EDENS','=','#',';','F', &
             &            IDUM,AUGER_EDENS,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EDENS'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_EDENS','R',IDUM,AUGER_EDENS,CDUM,LDUM,SDUM,N)

        ! density of valence band holes in [cm^-3]
        AUGER_HDENS=-1._q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_HDENS','=','#',';','F', &
             &            IDUM,AUGER_HDENS,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_HDENS'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_HDENS','R',IDUM,AUGER_HDENS,CDUM,LDUM,SDUM,N)
       
        ! temperature for Fermi distribution in [K] (same for electrons and holes)
        AUGER_TEMP=300._q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_TEMP','=','#',';','F', &
             &            IDUM,AUGER_TEMP,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_TEMP'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_TEMP','R',IDUM,AUGER_TEMP,CDUM,LDUM,SDUM,N)

        ! Fixed chemical potential in [eV] (in this case, AUGER_EDENS/AUGER_HDENS are ignored)
        AUGER_EFERMI=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EFERMI','=','#',';','F', &
             &            IDUM,AUGER_EFERMI,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EFERMI'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_EFERMI','R',IDUM,AUGER_EFERMI,CDUM,LDUM,SDUM,N)

        ! manual upper bound for valence band maximum (EVBM)
        AUGER_EVBHI=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EVBHI','=','#',';','F', &
             &            IDUM,AUGER_EVBHI,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EVBHI'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_EVBHI','R',IDUM,AUGER_EVBHI,CDUM,LDUM,SDUM,N)

        ! manual lower bound for conduction band minimum (ECBM)
        AUGER_ECBLO=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_ECBLO','=','#',';','F', &
             &            IDUM,AUGER_ECBLO,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_ECBLO'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_ECBLO','R',IDUM,AUGER_ECBLO,CDUM,LDUM,SDUM,N)

        ! we take into account only bands whose hole/electron occupation 
        ! is larger than VBM/CBM*AUGER_OCC_FAC_EEH
        AUGER_OCC_FAC_EEH=1E-4_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_OCC_FAC_EEH','=','#',';','F', &
             &            IDUM,AUGER_OCC_FAC_EEH,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_OCC_FAC_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_OCC_FAC_EEH','R',IDUM,AUGER_OCC_FAC_EEH,CDUM,LDUM,SDUM,N)

        ! we take into account only bands whose hole/lectron occupation 
        ! is larger than VBM/CBM*AUGER_OCC_FAC_EEH
        AUGER_OCC_FAC_EHH=1E-4_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_OCC_FAC_EHH','=','#',';','F', &
             &            IDUM,AUGER_OCC_FAC_EHH,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_OCC_FAC_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('AUGER_OCC_FAC_EHH','R',IDUM,AUGER_OCC_FAC_EHH,CDUM,LDUM,SDUM,N)

        ! lower bound on band energy
        AUGER_EMIN_EEH=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EMIN_EEH','=','#',';','F', &
             &            IDUM,AUGER_EMIN_EEH,CDUM,LDUM,SDUM,N,SIZE(AUGER_EMIN_EEH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_EMIN_EEH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EMIN_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_EMIN_EEH','R',IDUM,AUGER_EMIN_EEH,CDUM,LDUM,SDUM,N)
        
        ! upper bound on band energy
        AUGER_EMAX_EEH=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EMAX_EEH','=','#',';','F', &
             &            IDUM,AUGER_EMAX_EEH,CDUM,LDUM,SDUM,N,SIZE(AUGER_EMAX_EEH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_EMAX_EEH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EMAX_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_EMAX_EEH','R',IDUM,AUGER_EMAX_EEH,CDUM,LDUM,SDUM,N)

        ! lower bound on band index
        AUGER_BMIN_EEH=-1000
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_BMIN_EEH','=','#',';','I', &
             &            AUGER_BMIN_EEH,RDUM,CDUM,LDUM,SDUM,N,SIZE(AUGER_BMIN_EEH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_BMIN_EEH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_BMIN_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_BMIN_EEH','I',AUGER_BMIN_EEH,RDUM,CDUM,LDUM,SDUM,N)
        
        ! upper bound on band index
        AUGER_BMAX_EEH=-1000
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_BMAX_EEH','=','#',';','I', &
             &            AUGER_BMAX_EEH,RDUM,CDUM,LDUM,SDUM,N,SIZE(AUGER_BMAX_EEH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_BMAX_EEH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_BMAX_EEH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_BMAX_EEH','I',AUGER_BMAX_EEH,RDUM,CDUM,LDUM,SDUM,N)

        ! lower bound on band energy
        AUGER_EMIN_EHH=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EMIN_EHH','=','#',';','F', &
             &            IDUM,AUGER_EMIN_EHH,CDUM,LDUM,SDUM,N,SIZE(AUGER_EMIN_EHH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_EMIN_EHH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EMIN_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_EMIN_EHH','R',IDUM,AUGER_EMIN_EHH,CDUM,LDUM,SDUM,N)
        
        ! upper bound on band energy
        AUGER_EMAX_EHH=-2E30_q
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_EMAX_EHH','=','#',';','F', &
             &            IDUM,AUGER_EMAX_EHH,CDUM,LDUM,SDUM,N,SIZE(AUGER_EMAX_EHH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_EMAX_EHH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_EMAX_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_EMAX_EHH','R',IDUM,AUGER_EMAX_EHH,CDUM,LDUM,SDUM,N)

        ! lower bound on band index
        AUGER_BMIN_EHH=-1000
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_BMIN_EHH','=','#',';','I', &
             &            AUGER_BMIN_EHH,RDUM,CDUM,LDUM,SDUM,N,SIZE(AUGER_BMIN_EHH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_BMIN_EHH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_BMIN_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_BMIN_EHH','I',AUGER_BMIN_EHH,RDUM,CDUM,LDUM,SDUM,N)

        ! upper bound on band index
        AUGER_BMAX_EHH=-1000
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'AUGER_BMAX_EHH','=','#',';','I', &
             &            AUGER_BMAX_EHH,RDUM,CDUM,LDUM,SDUM,N,SIZE(AUGER_BMAX_EHH),IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<SIZE(AUGER_BMAX_EHH)))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''AUGER_BMAX_EHH'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR_V('AUGER_BMAX_EHH','I',AUGER_BMAX_EHH,RDUM,CDUM,LDUM,SDUM,N)

        ! F    don't collect all wavefunctions, redistribute on demand
        ! T    collect all wavefunctions at start before looping over k-points
        LAUGER_COLLECT=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER_COLLECT','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER_COLLECT,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER_COLLECT'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER_COLLECT','L',IDUM,RDUM,CDUM,LAUGER_COLLECT,SDUM,N)

        ! NCSHMEM =0     collect all wavefunctions, but don't use shared memory
        !        >=1    collect & use shared memory, NCSHMEM cores at each processor
        NCSHMEM=0
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'NCSHMEM','=','#',';','I', &
             &           NCSHMEM,RDUM,CDUM,LDUM,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''NCSHMEM'' from file INCAR.'
           NCSHMEM=0
        ENDIF
        CALL XML_INCAR_V('NCSHMEM','I',NCSHMEM,RDUM,CDUM,LDUM,SDUM,N)

        ! Use energy derivatives in DHDK file to determine width of energy window function
        ! In this case, AUGER_EWIDTH is reset automatically to the k-point spacing
        LAUGER_DHDK=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER_DHDK','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER_DHDK,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER_DHDK'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER_DHDK','L',IDUM,RDUM,CDUM,LAUGER_DHDK,SDUM,N)

        ! Redistribute wavefunctions for all k1,k2,k3,k4 only when all conditions 
        ! (on k-points, energy, occupation)
        ! Otherwise, redistribution of k1,k3 and k2,k4 is separated
        LAUGER_JIT=.FALSE.
        CALL RDATAB(LOPEN,INCAR,IO%IU5,'LAUGER_JIT','=','#',';','L', &
             &            IDUM,RDUM,CDUM,LAUGER_JIT,SDUM,N,1,IERR)
        IF (((IERR/=0).AND.(IERR/=3)).OR. &
             &                    ((IERR==0).AND.(N<1))) THEN
           IF (IO%IU0>=0) &
                WRITE(IO%IU0,*)'Error reading item ''LAUGER_JIT'' from file INCAR.'
           STOP
        ENDIF       
        CALL XML_INCAR('LAUGER_JIT','L',IDUM,RDUM,CDUM,LAUGER_JIT,SDUM,N)



        CLOSE(IO%IU5)

      END SUBROUTINE AUGER_READER


      !************************* SUBROUTINE CALCULATE_AUGER ******************
      !
      !***********************************************************************

      SUBROUTINE CALCULATE_AUGER(W,WDES,LATT_CUR,SYMM,T_INFO,GRID,P,NONL_S,KPOINTS,IO)
        USE main_mpi
        USE wave_high
        USE full_kpoints
        USE lattice
        USE constant
        USE nonl_high
        USE pseudo_struct
        USE msymmetry
        USE pead       
        USE xi
        USE ini
        USE wannier_interpolation
#ifdef use_shmem
        USE iso_c_binding
#endif
        IMPLICIT NONE
        TYPE (wavespin) W
        TYPE (wavedes) WDES
        TYPE(latt) :: LATT_CUR
        TYPE (symmetry) SYMM
        TYPE (type_info)   T_INFO
        TYPE (in_struct)   IO
        TYPE (grid_3d)     GRID
        TYPE (potcar)      P(T_INFO%NTYP)
        TYPE (nonl_struct) NONL_S
        TYPE (kpoints_struct) KPOINTS
        ! local variables
        TYPE (wavefun1), ALLOCATABLE :: W1_eeh(:),W2_eeh(:),W3_eeh(:),W4_eeh(:),&
             W1_ehh(:),W2_ehh(:),W3_ehh(:),W4_ehh(:)
        TYPE (wavefun1), POINTER :: WF_ALL(:,:)
        TYPE (wavedes1), TARGET :: WDESK1,WDESK2,WDESK3,WDESK4
        TYPE (wavedes1) :: WDES1
        INTEGER NQPOINT
        INTEGER K1_BASE,K1_COLLECT,K1_LOCAL,K1
        INTEGER K2,K3_COLLECT,K3,K4
        INTEGER ISP,IK,IB,I,I1,I2,I3,I4,IK1,IK2,IFAIL
        INTEGER,ALLOCATABLE :: K1LIST(:),K3LIST(:),K4TABLE(:)
        INTEGER K1DONE

        REAL(q) :: EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
             EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh
        
        INTEGER NB_MIN,NB_MAX

        INTEGER NB1_START_eeh,NB1_STOP_eeh,NB1_CNT_eeh
        INTEGER NB2_START_eeh,NB2_STOP_eeh,NB2_CNT_eeh
        INTEGER NB3_START_eeh,NB3_STOP_eeh,NB3_CNT_eeh
        INTEGER NB4_START_eeh,NB4_STOP_eeh,NB4_CNT_eeh

        INTEGER NB1_LO_eeh,NB1_HI_eeh
        INTEGER NB2_LO_eeh,NB2_HI_eeh
        INTEGER NB3_LO_eeh,NB3_HI_eeh
        INTEGER NB4_LO_eeh,NB4_HI_eeh
        INTEGER,ALLOCATABLE :: NB1_MIN_eeh(:),NB1_MAX_eeh(:)
        INTEGER,ALLOCATABLE :: NB2_MIN_eeh(:),NB2_MAX_eeh(:)
        INTEGER,ALLOCATABLE :: NB3_MIN_eeh(:),NB3_MAX_eeh(:)
        INTEGER,ALLOCATABLE :: NB4_MIN_eeh(:),NB4_MAX_eeh(:)
        INTEGER NB1_START_eeh_COLLECT,NB1_STOP_eeh_COLLECT,NB1_CNT_eeh_COLLECT
        INTEGER NB3_START_eeh_COLLECT,NB3_STOP_eeh_COLLECT,NB3_CNT_eeh_COLLECT

        INTEGER NB1_START_ehh,NB1_STOP_ehh,NB1_CNT_ehh
        INTEGER NB2_START_ehh,NB2_STOP_ehh,NB2_CNT_ehh
        INTEGER NB3_START_ehh,NB3_STOP_ehh,NB3_CNT_ehh
        INTEGER NB4_START_ehh,NB4_STOP_ehh,NB4_CNT_ehh
        INTEGER NB1_LO_ehh,NB1_HI_ehh
        INTEGER NB2_LO_ehh,NB2_HI_ehh
        INTEGER NB3_LO_ehh,NB3_HI_ehh
        INTEGER NB4_LO_ehh,NB4_HI_ehh
        INTEGER,ALLOCATABLE :: NB1_MIN_ehh(:),NB1_MAX_ehh(:)
        INTEGER,ALLOCATABLE :: NB2_MIN_ehh(:),NB2_MAX_ehh(:)
        INTEGER,ALLOCATABLE :: NB3_MIN_ehh(:),NB3_MAX_ehh(:)
        INTEGER,ALLOCATABLE :: NB4_MIN_ehh(:),NB4_MAX_ehh(:)
        INTEGER NB1_START_ehh_COLLECT,NB1_STOP_ehh_COLLECT,NB1_CNT_ehh_COLLECT
        INTEGER NB3_START_ehh_COLLECT,NB3_STOP_ehh_COLLECT,NB3_CNT_ehh_COLLECT

        REAL(q) :: R_tot,R_eeh,R_ehh
        REAL(q), ALLOCATABLE :: M4O_eeh(:,:,:,:),M4O_ehh(:,:,:,:)
        GDEF, ALLOCATABLE :: M4O_eeh_dir(:,:,:,:),M4O_eeh_ex(:,:,:,:),&
             M4O_ehh_dir(:,:,:,:),M4O_ehh_ex(:,:,:,:)
        REAL(q), ALLOCATABLE :: OCC_eeh(:,:,:,:),OCC_ehh(:,:,:,:),DE_eeh(:,:,:,:),DE_ehh(:,:,:,:),&
             DE_eeh_GLOBAL(:,:,:,:),DE_ehh_GLOBAL(:,:,:,:)
        REAL(q), ALLOCATABLE :: OCC1_eeh(:),OCC2_eeh(:),OCC3_eeh(:),OCC4_eeh(:),&
             OCC1_ehh(:),OCC2_ehh(:),OCC3_ehh(:),OCC4_ehh(:)

        INTEGER :: NFLOAT,NFFT,NSTRIP
        REAL(q) :: C_eeh,C_ehh,K_eeh,K_ehh
        REAL(q) :: FSG                        ! singularity correction
        REAL(q) :: EDENS,HDENS
        LOGICAL :: DO_eeh, DO_ehh
        TYPE (wavedes), POINTER :: WGW ! descriptor for basis set of response function
        TYPE (grid_3d), POINTER :: GRIDWGW
        TYPE(kpoints_struct) :: KPOINTS_IBZ
        REAL(q),ALLOCATABLE :: ELEC_OCC(:,:,:),HOLE_OCC(:,:,:)
        REAL(q) :: AUGER_TINY_eeh,AUGER_TINY_ehh

        INTEGER :: TIMER_ALL,TIMER_NQ,TIMER_RATE,TIMER_MAT,&
             TIMER_K1,TIMER_K2,TIMER_COLL,TIMER_ANY1,TIMER_ANY2,TIMER_ANY3,&
             TIMER_K1A,TIMER_K1B,TIMER_K1C,TIMER_K1D,TIMER_GATH,&
             TIMER_2E4O,TIMER_PHASE1,TIMER_PHASE2
        INTEGER, ALLOCATABLE :: BANDS13(:),BANDS14(:),BANDS23(:),BANDS24(:)
        INTEGER :: DO_MAT
        INTEGER :: J,NB_ALL,J1,J2
#ifdef use_shmem
        INTEGER(c_int), ALLOCATABLE :: SHMID(:,:)
        TYPE(c_ptr), ALLOCATABLE :: ADDRESS(:,:)
#endif
        REAL(q), ALLOCATABLE :: DHDK(:,:,:),DHDK_FULL(:,:,:)
        REAL(q) :: DHDK_VKPT(3),WIDTH,DHDK_FACTOR
        INTEGER :: DHDK_NB,DHDK_NKPTS
        INTEGER :: NB1_CNT_eeh_GLOBAL,NB2_CNT_eeh_GLOBAL,NB3_CNT_eeh_GLOBAL,NB4_CNT_eeh_GLOBAL
        INTEGER :: NB1_CNT_ehh_GLOBAL,NB2_CNT_ehh_GLOBAL,NB3_CNT_ehh_GLOBAL,NB4_CNT_ehh_GLOBAL
        LOGICAL :: DO_eeh_GLOBAL
        LOGICAL :: DO_ehh_GLOBAL
        INTEGER, ALLOCATABLE :: NN(:,:,:)

        !-----------------------------------------------------------------------
        AUGER_IO=IO


        !-----------------------------------------------------------------------
        ! CHECK INPUT PARAMETERS
#ifdef MPI
        IF(WDES%COMM%NODE_ME==WDES%COMM%IONODE)LDUMP_AUGER=.TRUE.
#else
        LDUMP_AUGER=.TRUE.
#endif

        IF(AUGER_EDENS<0 .AND. AUGER_HDENS<0.AND.AUGER_EFERMI<-1E30_q)THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'You must specify one out of AUGER_HDENS, AUGER_EDENS or AUGER_EFERMI in INCAR.'
           STOP
        END IF
        IF(AUGER_EDENS>0 .AND. AUGER_HDENS>0)THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'You cannot specify both AUGER_HDENS and AUGER_EDENS in INCAR.'
           STOP
        END IF
        IF(AUGER_EFERMI>-1E30_q)THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'AUGER_EFERMI specified, AUGER_HDENS and AUGER_EDENS are ignored.'
        END IF
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        ! Generate full k-grid
        CALL COPY_KPOINTS(KPOINTS,KPOINTS_IBZ)
        ! generate wave function in full Brillouin zone
        IF (SYMM%ISYM>=0.AND. .NOT.W%WDES%LGAMMA) THEN
           ! switch of symmetry
           CALL NOSYMM(LATT_CUR%A,T_INFO%NTYP,T_INFO%NITYP,T_INFO%NIONS,SYMM%PTRANS, &
                &   SYMM%NROT,SYMM%NPTRANS,SYMM%ROTMAP,SYMM%MAGROT,W%WDES%ISPIN,IO%IU6)
           ! reread k-points with LINVERSION=.FALSE. to generate full mesh
           CALL RE_READ_KPOINTS(KPOINTS,LATT_CUR,.FALSE., &
                &   T_INFO%NIONS,SYMM%ROTMAP,SYMM%MAGROT,SYMM%ISYM,IO%IU6, AUGER_IO%IU0 )
           ! print out KPOINTS here to check whether this was done properly
           CALL RE_GEN_LAYOUT( GRID, W%WDES, KPOINTS, LATT_CUR, LATT_CUR,-1, AUGER_IO%IU0)
           CALL PEAD_RESETUP_WDES(W%WDES, GRID, KPOINTS, LATT_CUR, LATT_CUR, IO)
           CALL REALLOCATE_WAVE( W, GRID, W%WDES, NONL_S, T_INFO, P, LATT_CUR)
           CALL RESETUP_FOCK( W%WDES, LATT_CUR)
        ENDIF
        !-----------------------------------------------------------------------

        !-----------------------------------------------------------------------
        ! generate descriptor for response function
        ! (copied from chi.F)
        ALLOCATE(WGW, GRIDWGW)
        WGW=WDES_FOCK
        WGW%NKPTS=KPOINTS_FULL%NKPTS
        WGW%NKDIM=KPOINTS_FULL%NKPTS
        WGW%NKPTS_FOR_GEN_LAYOUT=KPOINTS_FULL%NKPTS
        ! KPOINTS_FULL structure might be reallocated, better to allocate and copy data
        !        ALLOCATE(WGW%VKPT(1:3,SIZE(KPOINTS_FULL%VKPT,2)),WGW%WTKPT(SIZE(KPOINTS_FULL%WTKPT,1)))
        WGW%VKPT =>KPOINTS_FULL%VKPT
        WGW%WTKPT=>KPOINTS_FULL%WTKPT
        WGW%ENMAX=ENCUTGW
        IF (ENCUTLF==-1) ENCUTLF=WGW%ENMAX

        ! GRIDWGW is identical to GRID_FOCK, except for GRIDWGW%FFTSCA
        GRIDWGW=GRID_FOCK
        IF (WGW%LGAMMA) THEN
           ! gamma only data layout with wavefunction stored as real in real space
           CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
           GRIDWGW%LREAL=.TRUE.
        ELSE
           CALL GEN_LAYOUT(GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B, IO%IU6,.TRUE.)
        ENDIF
        CALL GEN_INDEX (GRIDWGW, WGW, LATT_CUR%B, LATT_CUR%B,IO%IU6,-1, .TRUE.)
        !  init FFT (required if real to complex FFT is used)
        CALL FFTINI(WGW%NINDPW(1,1), WGW%NGVECTOR(1), WGW%NKPTS, WGW%NGDIM, GRIDWGW)
        NULLIFY(AUGER_WPOTH)
        IF (LGWLF .AND. .NOT. ASSOCIATED(AUGER_WPOTH))THEN
           CALL INIT_WPOT_HANDLE( AUGER_WPOTH, WGW, KPOINTS_FULL%NKPTS, IO%IU6, 6, 1, 1, 1 )
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'Using WPOT from files'
        END IF

        ! singularity correction
        ! CHECK THIS!!!!
        ! FSG=FSG_STORE(1)
        FSG=SET_FSG(GRIDHF, LATT_CUR, 1)
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        ! apply energy shift
        IF(SCISSOR/=0)THEN
           CALL APPLY_SCISSOR(W,SCISSOR)
        END IF
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        ! calculate parameters of Fermi distribution
        AUGER_SIGMA=BOLKEV*AUGER_TEMP
        IF(AUGER_EFERMI>-1E30_q)THEN
           AUGER_MU=AUGER_EFERMI
        ELSE
           IF(AUGER_EDENS>0)THEN
              ! convert from [cm^-3] to [Angstrom^-3]
              EDENS=AUGER_EDENS*1E-24_q
              CALL DETERMINE_EFERMI(EDENS,'E',MINVAL(REAL(W%CELTOT,q)),MAXVAL(REAL(W%CELTOT,q)),AUGER_SIGMA,W,WDES,LATT_CUR,AUGER_MU)
           END IF
           IF(AUGER_HDENS>0)THEN
              ! convert from [cm^-3] to [Angstrom^-3]
              HDENS=AUGER_HDENS*1E-24_q
              CALL DETERMINE_EFERMI(HDENS,'H',MINVAL(REAL(W%CELTOT,q)),MAXVAL(REAL(W%CELTOT,q)),AUGER_SIGMA,W,WDES,LATT_CUR,AUGER_MU)
           END IF
        END IF
        ! calculate electron and hole concentration
        CALL EH_DENS(AUGER_MU,AUGER_SIGMA,W,WDES,LATT_CUR,EDENS=EDENS,HDENS=HDENS)
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        ! Setup for using energy derivatives to determine energy width
        IF(LAUGER_DHDK)THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'Reading DHDK file'
           OPEN(UNIT=199,FILE='DHDK',FORM='UNFORMATTED',ACCESS='SEQUENTIAL')
           READ(199) DHDK_NB,DHDK_NKPTS
           IF(DHDK_NKPTS/=KPOINTS_IBZ%NKPTS)THEN
              IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'CALCULATE_AUGER: DHDK_NKPTS /= KPOINTS_IBZ%NKPTS',DHDK_NKPTS,KPOINTS_IBZ%NKPTS
              STOP
           END IF
           ALLOCATE(DHDK(DHDK_NB,3,DHDK_NKPTS))
           DO IK1=1,DHDK_NKPTS
              READ(199) DHDK_VKPT,DHDK(:,:,IK1)
              IF( MINVAL(ABS(DHDK_VKPT-KPOINTS_IBZ%VKPT(:,IK1)))>1E-8 )THEN
                 IF(LDUMP_AUGER)WRITE(AUGER_IO%IU0,*) 'CALCULATE_AUGER: k-point changed',IK1,DHDK_VKPT,KPOINTS_IBZ%VKPT(:,IK1)
                 STOP
              END IF
           END DO
           CLOSE(199)
           ! generate derivatives in full BZ
           ALLOCATE(DHDK_FULL(DHDK_NB,3,KPOINTS_FULL%NKPTS))
           DO IK1=1,KPOINTS_FULL%NKPTS
              DHDK_FULL(:,:,IK1)=DHDK(:,:,KPOINTS_FULL_ORIG%NEQUIV(IK1))
           END DO
           DEALLOCATE(DHDK)
           AUGER_EWIDTH=0.5_q/KPOINTS_FULL%NKPX
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'Setting AUGER_EWIDTH to half of k-spacing (full width is 2*ewidth):',AUGER_EWIDTH
        END IF
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        IF(LDUMP_AUGER)THEN
           WRITE(AUGER_IO%IU0,*) 'ELECTRON DISTRIBUTION PARAMETERS:'
           WRITE(AUGER_IO%IU0,*) 'TEMPERATURE',AUGER_TEMP
           WRITE(AUGER_IO%IU0,*) 'CHEM.POT.',AUGER_MU
           WRITE(AUGER_IO%IU0,*) 'SIGMA',AUGER_SIGMA
           WRITE(AUGER_IO%IU0,*) 'EDENS',EDENS
           WRITE(AUGER_IO%IU0,*) 'HDENS',HDENS
        END IF
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        ! Determine for K1,K2,K3,K4 and for each k-point
        ! the RANGE OF BANDS that could possibly yield significant contributions

        ! 1. Determine the energy range in which significant occupation occurs
        ! (rom Fermi distribution and AUGER_MU, AUGER_SIGMA)
        CALL ENERGY_RANGE(W,WDES,&
             EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
             EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh,&
             AUGER_TINY_eeh,AUGER_TINY_ehh)
        ! 2. Optionally, lower and upper bounds on energies are applied
        IF(AUGER_EMIN_EEH(1)>-1E30) EMIN1_eeh=AUGER_EMIN_EEH(1)
        IF(AUGER_EMAX_EEH(1)>-1E30) EMAX1_eeh=AUGER_EMAX_EEH(1)           
        IF(AUGER_EMIN_EEH(2)>-1E30) EMIN2_eeh=AUGER_EMIN_EEH(2)
        IF(AUGER_EMAX_EEH(2)>-1E30) EMAX2_eeh=AUGER_EMAX_EEH(2)           
        IF(AUGER_EMIN_EEH(3)>-1E30) EMIN3_eeh=AUGER_EMIN_EEH(3)
        IF(AUGER_EMAX_EEH(3)>-1E30) EMAX3_eeh=AUGER_EMAX_EEH(3)           
        IF(AUGER_EMIN_EEH(4)>-1E30) EMIN4_eeh=AUGER_EMIN_EEH(4)
        IF(AUGER_EMAX_EEH(4)>-1E30) EMAX4_eeh=AUGER_EMAX_EEH(4)           

        IF(AUGER_EMIN_EHH(1)>-1E30) EMIN1_ehh=AUGER_EMIN_EHH(1)
        IF(AUGER_EMAX_EHH(1)>-1E30) EMAX1_ehh=AUGER_EMAX_EHH(1)           
        IF(AUGER_EMIN_EHH(2)>-1E30) EMIN2_ehh=AUGER_EMIN_EHH(2)
        IF(AUGER_EMAX_EHH(2)>-1E30) EMAX2_ehh=AUGER_EMAX_EHH(2)           
        IF(AUGER_EMIN_EHH(3)>-1E30) EMIN3_ehh=AUGER_EMIN_EHH(3)
        IF(AUGER_EMAX_EHH(3)>-1E30) EMAX3_ehh=AUGER_EMAX_EHH(3)           
        IF(AUGER_EMIN_EHH(4)>-1E30) EMIN4_ehh=AUGER_EMIN_EHH(4)
        IF(AUGER_EMAX_EHH(4)>-1E30) EMAX4_ehh=AUGER_EMAX_EHH(4)           

        ! 3. Determine band range based on energy limits
        ALLOCATE(NB1_MIN_eeh(WDES%NKPTS),NB1_MAX_eeh(WDES%NKPTS),&
             NB2_MIN_eeh(WDES%NKPTS),NB2_MAX_eeh(WDES%NKPTS),&
             NB3_MIN_eeh(WDES%NKPTS),NB3_MAX_eeh(WDES%NKPTS),&
             NB4_MIN_eeh(WDES%NKPTS),NB4_MAX_eeh(WDES%NKPTS),&
             NB1_MIN_ehh(WDES%NKPTS),NB1_MAX_ehh(WDES%NKPTS),&
             NB2_MIN_ehh(WDES%NKPTS),NB2_MAX_ehh(WDES%NKPTS),&
             NB3_MIN_ehh(WDES%NKPTS),NB3_MAX_ehh(WDES%NKPTS),&
             NB4_MIN_ehh(WDES%NKPTS),NB4_MAX_ehh(WDES%NKPTS))
        CALL BAND_RANGE(W,WDES,&
             EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,&
             EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
             EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,&
             EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh,&
             NB1_MIN_eeh,NB1_MAX_eeh,NB2_MIN_eeh,NB2_MAX_eeh,&
             NB3_MIN_eeh,NB3_MAX_eeh,NB4_MIN_eeh,NB4_MAX_eeh,&
             NB1_MIN_ehh,NB1_MAX_ehh,NB2_MIN_ehh,NB2_MAX_ehh,&
             NB3_MIN_ehh,NB3_MAX_ehh,NB4_MIN_ehh,NB4_MAX_ehh)
        ! 3. Optionally, apply lower and upper bounds on band ranges
        IF(AUGER_BMIN_EEH(1)>0) NB1_MIN_eeh=MAX(NB1_MIN_eeh,AUGER_BMIN_EEH(1))
        IF(AUGER_BMAX_EEH(1)>0) NB1_MAX_eeh=MIN(NB1_MAX_eeh,AUGER_BMAX_EEH(1))
        IF(AUGER_BMIN_EEH(2)>0) NB2_MIN_eeh=MAX(NB2_MIN_eeh,AUGER_BMIN_EEH(2))
        IF(AUGER_BMAX_EEH(2)>0) NB2_MAX_eeh=MIN(NB2_MAX_eeh,AUGER_BMAX_EEH(2))
        IF(AUGER_BMIN_EEH(3)>0) NB3_MIN_eeh=MAX(NB3_MIN_eeh,AUGER_BMIN_EEH(3))
        IF(AUGER_BMAX_EEH(3)>0) NB3_MAX_eeh=MIN(NB3_MAX_eeh,AUGER_BMAX_EEH(3))
        IF(AUGER_BMIN_EEH(4)>0) NB4_MIN_eeh=MAX(NB4_MIN_eeh,AUGER_BMIN_EEH(4))
        IF(AUGER_BMAX_EEH(4)>0) NB4_MAX_eeh=MIN(NB4_MAX_eeh,AUGER_BMAX_EEH(4))

        IF(AUGER_BMIN_EHH(1)>0) NB1_MIN_eeh=MAX(NB1_MIN_eeh,AUGER_BMIN_EHH(1))
        IF(AUGER_BMAX_EHH(1)>0) NB1_MAX_eeh=MIN(NB1_MAX_eeh,AUGER_BMAX_EHH(1))
        IF(AUGER_BMIN_EHH(2)>0) NB2_MIN_eeh=MAX(NB2_MIN_eeh,AUGER_BMIN_EHH(2))
        IF(AUGER_BMAX_EHH(2)>0) NB2_MAX_eeh=MIN(NB2_MAX_eeh,AUGER_BMAX_EHH(2))
        IF(AUGER_BMIN_EHH(3)>0) NB3_MIN_eeh=MAX(NB3_MIN_eeh,AUGER_BMIN_EHH(3))
        IF(AUGER_BMAX_EHH(3)>0) NB3_MAX_eeh=MIN(NB3_MAX_eeh,AUGER_BMAX_EHH(3))
        IF(AUGER_BMIN_EHH(4)>0) NB4_MIN_eeh=MAX(NB4_MIN_eeh,AUGER_BMIN_EHH(4))
        IF(AUGER_BMAX_EHH(4)>0) NB4_MAX_eeh=MIN(NB4_MAX_eeh,AUGER_BMAX_EHH(4))

        ! Minimum and maximum band index and max. band count over all k-points
        ! separately for K1,K2,K3,K4
        NB1_LO_eeh=MINVAL(NB1_MIN_eeh); NB1_HI_eeh=MAXVAL(NB1_MAX_eeh)
        NB2_LO_eeh=MINVAL(NB2_MIN_eeh); NB2_HI_eeh=MAXVAL(NB2_MAX_eeh)
        NB3_LO_eeh=MINVAL(NB3_MIN_eeh); NB3_HI_eeh=MAXVAL(NB3_MAX_eeh)
        NB4_LO_eeh=MINVAL(NB4_MIN_eeh); NB4_HI_eeh=MAXVAL(NB4_MAX_eeh)
        NB1_CNT_eeh=MAXVAL(NB1_MAX_eeh-NB1_MIN_eeh+1)
        NB2_CNT_eeh=MAXVAL(NB2_MAX_eeh-NB2_MIN_eeh+1)
        NB3_CNT_eeh=MAXVAL(NB3_MAX_eeh-NB3_MIN_eeh+1)
        NB4_CNT_eeh=MAXVAL(NB4_MAX_eeh-NB4_MIN_eeh+1)

        NB1_LO_ehh=MINVAL(NB1_MIN_ehh); NB1_HI_ehh=MAXVAL(NB1_MAX_ehh)
        NB2_LO_ehh=MINVAL(NB2_MIN_ehh); NB2_HI_ehh=MAXVAL(NB2_MAX_ehh)
        NB3_LO_ehh=MINVAL(NB3_MIN_ehh); NB3_HI_ehh=MAXVAL(NB3_MAX_ehh)
        NB4_LO_ehh=MINVAL(NB4_MIN_ehh); NB4_HI_ehh=MAXVAL(NB4_MAX_ehh)
        NB1_CNT_ehh=MAXVAL(NB1_MAX_ehh-NB1_MIN_ehh+1)
        NB2_CNT_ehh=MAXVAL(NB2_MAX_ehh-NB2_MIN_ehh+1)
        NB3_CNT_ehh=MAXVAL(NB3_MAX_ehh-NB3_MIN_ehh+1)
        NB4_CNT_ehh=MAXVAL(NB4_MAX_ehh-NB4_MIN_ehh+1)

        NB_MIN=MIN(NB1_LO_eeh,NB2_LO_eeh,NB3_LO_eeh,NB4_LO_eeh,&
             NB1_LO_ehh,NB2_LO_ehh,NB3_LO_ehh,NB4_LO_ehh)
        NB_MAX=MAX(NB1_HI_eeh,NB2_HI_eeh,NB3_HI_eeh,NB4_HI_eeh,&
             NB1_HI_ehh,NB2_HI_ehh,NB3_HI_ehh,NB4_HI_ehh)
        NB_ALL=NB_MAX-NB_MIN+1
        !-----------------------------------------------------------------------


        !-----------------------------------------------------------------------
        IF(LDUMP_AUGER)THEN
           WRITE(AUGER_IO%IU0,*) '----------------------------------------'
           WRITE(AUGER_IO%IU0,*) 'BAND SEARCH PARAMETERS:'
           WRITE(AUGER_IO%IU0,*) 'EWIDTH',AUGER_EWIDTH
           IF(LAUGER_EEH)THEN
              WRITE(AUGER_IO%IU0,*) 'EMIN1_eeh,EMAX1_eeh',EMIN1_eeh,EMAX1_eeh
              WRITE(AUGER_IO%IU0,*) 'EMIN2_eeh,EMAX2_eeh',EMIN2_eeh,EMAX2_eeh
              WRITE(AUGER_IO%IU0,*) 'EMIN3_eeh,EMAX3_eeh',EMIN3_eeh,EMAX3_eeh
              WRITE(AUGER_IO%IU0,*) 'EMIN4_eeh,EMAX4_eeh',EMIN4_eeh,EMAX4_eeh
           END IF
           IF(LAUGER_EHH)THEN
              WRITE(AUGER_IO%IU0,*) 'EMIN1_ehh,EMAX1_ehh',EMIN1_ehh,EMAX1_ehh
              WRITE(AUGER_IO%IU0,*) 'EMIN2_ehh,EMAX2_ehh',EMIN2_ehh,EMAX2_ehh
              WRITE(AUGER_IO%IU0,*) 'EMIN3_ehh,EMAX3_ehh',EMIN3_ehh,EMAX3_ehh
              WRITE(AUGER_IO%IU0,*) 'EMIN4_ehh,EMAX4_ehh',EMIN4_ehh,EMAX4_ehh
           END IF
           WRITE(AUGER_IO%IU0,*) 'ELIGIBLE BANDS:'
           IF(LAUGER_EEH)THEN
              WRITE(AUGER_IO%IU0,*) 'eeh: NB1_LO,NB1_HI,NB1_CNT',NB1_LO_eeh,NB1_HI_eeh,NB1_CNT_eeh
              WRITE(AUGER_IO%IU0,*) 'eeh: NB2_LO,NB2_HI,NB2_CNT',NB2_LO_eeh,NB2_HI_eeh,NB2_CNT_eeh
              WRITE(AUGER_IO%IU0,*) 'eeh: NB3_LO,NB3_HI,NB3_CNT',NB3_LO_eeh,NB3_HI_eeh,NB3_CNT_eeh
              WRITE(AUGER_IO%IU0,*) 'eeh: NB4_LO,NB4_HI,NB4_CNT',NB4_LO_eeh,NB4_HI_eeh,NB4_CNT_eeh
           END IF
           IF(LAUGER_EHH)THEN
              WRITE(AUGER_IO%IU0,*) 'ehh: NB1_LO,NB1_HI,NB1_CNT',NB1_LO_ehh,NB1_HI_ehh,NB1_CNT_ehh
              WRITE(AUGER_IO%IU0,*) 'ehh: NB2_LO,NB2_HI,NB2_CNT',NB2_LO_ehh,NB2_HI_ehh,NB2_CNT_ehh
              WRITE(AUGER_IO%IU0,*) 'ehh: NB3_LO,NB3_HI,NB3_CNT',NB3_LO_ehh,NB3_HI_ehh,NB3_CNT_ehh
              WRITE(AUGER_IO%IU0,*) 'ehh: NB4_LO,NB4_HI,NB4_CNT',NB4_LO_ehh,NB4_HI_ehh,NB4_CNT_ehh
           END IF
           WRITE(AUGER_IO%IU0,*) 'NB_MIN,NB_MAX',NB_MIN,NB_MAX
           WRITE(AUGER_IO%IU0,*) '----------------------------------------'
        END IF
        !-----------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Pre-calculate occupation probabilites to avoid repetitive calculation (quite costly)
        ALLOCATE(ELEC_OCC(NB_MIN:NB_MAX,WDES%NKPTS,WDES%ISPIN),&
             HOLE_OCC(NB_MIN:NB_MAX,WDES%NKPTS,WDES%ISPIN))
        DO ISP=1,WDES%ISPIN
           DO IK=1,WDES%NKPTS
              DO IB=NB_MIN,NB_MAX
                 ELEC_OCC(IB,IK,ISP)=FERMI(REAL(W%CELTOT(IB,IK,ISP),q),AUGER_MU,AUGER_SIGMA)
                 HOLE_OCC(IB,IK,ISP)=ANTIFERMI(REAL(W%CELTOT(IB,IK,ISP),q),AUGER_MU,AUGER_SIGMA)
              END DO
           END DO
        END DO
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Allocate wavefunctions and arrays for matrix elements
        CALL SETWDES(WDES,WDESK1,0)
        CALL SETWDES(WDES,WDESK2,0)
        CALL SETWDES(WDES,WDESK3,0)
        CALL SETWDES(WDES,WDESK4,0)

        IF(LAUGER_EEH)THEN
           ALLOCATE(&
                M4O_eeh(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh),&
                M4O_eeh_dir(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh),&
                M4O_eeh_ex(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh))
           ALLOCATE(&
                OCC1_eeh(NB1_CNT_eeh),OCC2_eeh(NB2_CNT_eeh),OCC3_eeh(NB3_CNT_eeh),OCC4_eeh(NB4_CNT_eeh))
           ALLOCATE(&
                OCC_eeh(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh),&
                DE_eeh_GLOBAL(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh),&
                DE_eeh(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh))

           ALLOCATE(W1_eeh(NB1_CNT_eeh),W2_eeh(NB2_CNT_eeh),W3_eeh(NB3_CNT_eeh),W4_eeh(NB4_CNT_eeh))
           DO I=1,NB1_CNT_eeh
              CALL NEWWAV(W1_eeh(I),WDESK1,.TRUE.)
           END DO
           DO I=1,NB2_CNT_eeh
              CALL NEWWAV(W2_eeh(I),WDESK2,.TRUE.)
           END DO
           DO I=1,NB3_CNT_eeh
              CALL NEWWAV(W3_eeh(I),WDESK3,.TRUE.)
           END DO
           DO I=1,NB4_CNT_eeh
              CALL NEWWAV(W4_eeh(I),WDESK4,.TRUE.)
           END DO

        END IF

        IF(LAUGER_EHH)THEN
           ALLOCATE(&
                M4O_ehh(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh),&
                M4O_ehh_dir(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh),&
                M4O_ehh_ex(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh))
           ALLOCATE(&
                OCC1_ehh(NB1_CNT_ehh),OCC2_ehh(NB2_CNT_ehh),OCC3_ehh(NB3_CNT_ehh),OCC4_ehh(NB4_CNT_ehh))
           ALLOCATE(&
                OCC_ehh(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh),&
                DE_ehh(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh))

           ALLOCATE(W1_ehh(NB1_CNT_ehh),W2_ehh(NB2_CNT_ehh),W3_ehh(NB3_CNT_ehh),W4_ehh(NB4_CNT_ehh))
           DO I=1,NB1_CNT_ehh
              CALL NEWWAV(W1_ehh(I),WDESK1,.TRUE.)
           END DO
           DO I=1,NB2_CNT_ehh
              CALL NEWWAV(W2_ehh(I),WDESK2,.TRUE.)
           END DO
           DO I=1,NB3_CNT_ehh
              CALL NEWWAV(W3_ehh(I),WDESK3,.TRUE.)
           END DO
           DO I=1,NB4_CNT_ehh
              CALL NEWWAV(W4_ehh(I),WDESK4,.TRUE.)
           END DO
        END IF

        ALLOCATE(K1LIST(WDES%NB_PAR),K3LIST(WDES%NB_PAR),K4TABLE(WDES%NKPTS))
        ALLOCATE(BANDS13(NB_ALL*NB_ALL),BANDS14(NB_ALL*NB_ALL),&
             BANDS23(NB_ALL*NB_ALL),BANDS24(NB_ALL*NB_ALL))
        NSTRIP=MAX(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh,&
             NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh)
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Initialize counter variables
        NFLOAT=0
        NFFT=0
        C_eeh=0
        C_ehh=0
        K_eeh=0
        K_ehh=0
        R_eeh=0
        R_ehh=0
        !---------------------------------------------------------------------------
        ! FIXED SPIN COMPONENT
        ISP=1
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Initialize timers
        TIMER_ALL=CREATE_TIMER('ALL')
        TIMER_NQ=CREATE_TIMER('NQ')
        TIMER_K1=CREATE_TIMER('K1')
        TIMER_K2=CREATE_TIMER('K2')
        TIMER_COLL=CREATE_TIMER('COLL')
        TIMER_GATH=CREATE_TIMER('GATH')
        TIMER_MAT=CREATE_TIMER('MAT')
        TIMER_RATE=CREATE_TIMER('RATE')
        ! timers for AUGER_2E4O
        TIMER_2E4O=CREATE_TIMER('2E4O')        
        TIMER_PHASE1=CREATE_TIMER('PH1')
        TIMER_PHASE2=CREATE_TIMER('PH2')
        !---------------------------------------------------------------------------


        CALL START_TIMER(TIMER_ALL)


        !---------------------------------------------------------------------------
        IF(LAUGER_COLLECT)THEN
           ! collect all wavefunctions at once before loop
           CALL START_TIMER(TIMER_COLL)           
           ! allocate array of wavefunctions
           ALLOCATE(WF_ALL(NB_MIN:NB_MAX,WDES%NKPTS))
           CALL SETWDES(WDES,WDES1,0)
#ifdef use_shmem
           IF(NCSHMEM>0)THEN
              ! use shared memory
              ALLOCATE(ADDRESS(NB_MIN:NB_MAX,WDES%NKPTS),SHMID(NB_MIN:NB_MAX,WDES%NKPTS))
              DO IK=1,WDES%NKPTS
                 DO IB=NB_MIN,NB_MAX
                    CALL NEWWAV_SHMEM(WF_ALL(IB,IK),WDES1,.TRUE.,ADDRESS(IB,IK),SHMID(IB,IK))
                 ENDDO
              ENDDO
              CALL W1_GATHER_GLB_ALLK_SHMEM(W,NB_MIN,NB_MAX,ISP,WF_ALL)
           ELSE
#endif
              ! do not use shared memory
              DO IK=1,WDES%NKPTS
                 DO IB=NB_MIN,NB_MAX
                    CALL NEWWAV_NOCW(WF_ALL(IB,IK),WDES1)
                 ENDDO
              ENDDO
              CALL W1_GATHER_GLB_ALLK(W,NB_MIN,NB_MAX,ISP,WF_ALL)
#ifdef use_shmem
           END IF
#endif
           CALL STOP_TIMER(TIMER_COLL)
        END IF
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        CALL START_TIMER(TIMER_NQ)
        ! KPAR parallelization of q-point loop (restricted to IRZ)
        qloop: DO NQPOINT=1,KPOINTS_IBZ%NKPTS

           ! q=k1-k3
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'NQ',NQPOINT

#ifdef MPI
           IF (MOD(NQPOINT-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) CYCLE qloop
           ! group of cpus:  | 1             | 2             | NCPU                |
           ! NQPOINT:        | 1,1+NCPU,...  | 2,2+NCPU,...  | NCPU,NCPU+NCPU,...  |
#endif

           ! create table of K4 for each K2 for fixed Q (k-point search is very time-consuming)
           DO K2=1,WDES%NKPTS
              K4TABLE(K2)=KPOINT_IN_FULL_GRID(WDES%VKPT(:,K2)+KPOINTS_IBZ%VKPT(:,NQPOINT),KPOINTS_FULL)
           END DO

           K1DONE=0

           !---------------------------------------------------------------------------
           ! k1loop: loop over SETS of k1-points
           k1loop: DO 
              !
              ! Check the K1LIST, whether there is any empty slot.
              ! Fill in new k1-points, as long as we have done all NKPTS k-points
              !
              K1LIST=0
              K3LIST=0

              ! local indices, different on each node!
              K1=0        
              K3=0
              NB1_START_eeh=0
              NB1_STOP_eeh=-1
              NB1_CNT_eeh=0

              NB1_START_ehh=0
              NB1_STOP_ehh=-1
              NB1_CNT_ehh=0

              NB3_START_eeh=0
              NB3_STOP_eeh=-1
              NB3_CNT_eeh=0

              NB3_START_ehh=0
              NB3_STOP_ehh=-1
              NB3_CNT_ehh=0

              !---------------------------------------------------------------------------
              ! Fill in k-points into nodes a la RMM-DIIS or similar
              ! distribute the K1 in a round robin fashion over cores
              CALL START_TIMER(TIMER_K1)
              k1slots: DO IK1=1,WDES%NB_PAR

                 ! find the next k1-point where there are bands to work on
                 nextk1: DO WHILE (K1DONE<WDES%NKPTS)

                    K1DONE=K1DONE+1
                    K1_COLLECT=K1DONE
                    ! K3=K1-Q

                    K3_COLLECT=KPOINT_IN_FULL_GRID(WDES%VKPT(:,K1_COLLECT)-KPOINTS_IBZ%VKPT(:,NQPOINT),KPOINTS_FULL)

                    !--------------------------------------------------------------
                    ! first check whether we need to work at this k1-point at all!
                    DO_eeh=.FALSE.
                    DO_ehh=.FALSE.

                    CALL SETWDES(WDES,WDESK1,K1_COLLECT)
                    CALL SETWDES(WDES,WDESK3,K3_COLLECT)

                    !------------------------------
                    ! eeh
                    IF(LAUGER_EEH)THEN
                       ! K1
                       NB1_START_eeh_COLLECT=NB1_MIN_eeh(K1_COLLECT)
                       NB1_STOP_eeh_COLLECT=NB1_MAX_eeh(K1_COLLECT)
                       NB1_CNT_eeh_COLLECT=MAX(NB1_STOP_eeh_COLLECT-NB1_START_eeh_COLLECT+1,0)
                       ! K3
                       NB3_START_eeh_COLLECT=NB3_MIN_eeh(K3_COLLECT)
                       NB3_STOP_eeh_COLLECT=NB3_MAX_eeh(K3_COLLECT)
                       NB3_CNT_eeh_COLLECT=MAX(NB3_STOP_eeh_COLLECT-NB3_START_eeh_COLLECT+1,0)

                       IF(NB1_CNT_eeh_COLLECT>0.AND.NB3_CNT_eeh_COLLECT>0)THEN
                          DO_eeh=.TRUE.

                          ! get the correct wavefunctions on this node
                          IF(LAUGER_COLLECT)THEN
                             IF (IK1==WDES%NB_LOW) THEN
                                DO I=NB1_START_eeh_COLLECT,NB1_STOP_eeh_COLLECT
                                   W1_eeh(I-NB1_START_eeh_COLLECT+1)%CR=>WF_ALL(I,K1_COLLECT)%CR(:)
                                   W1_eeh(I-NB1_START_eeh_COLLECT+1)%CPROJ=>WF_ALL(I,K1_COLLECT)%CPROJ(:)
                                END DO
                                DO I=NB3_START_eeh_COLLECT,NB3_STOP_eeh_COLLECT
                                   W3_eeh(I-NB3_START_eeh_COLLECT+1)%CR=>WF_ALL(I,K3_COLLECT)%CR(:)
                                   W3_eeh(I-NB3_START_eeh_COLLECT+1)%CPROJ=>WF_ALL(I,K3_COLLECT)%CPROJ(:)
                                END DO
                             END IF
                          ELSE
                             ! On all nodes: collect wavefunctions
                             ! Here the k-points are distributed over nodes 
                             ! (node IK1 receives, all other nodes send)
                             ! must be called by all nodes                       
                             IF(LAUGER_JIT)THEN
                                ! do nothing
                             ELSE
                                CALL START_TIMER(TIMER_GATH)
                                CALL W1_GATHER_KNODESEL(W,NB1_START_eeh_COLLECT,NB1_STOP_eeh_COLLECT,ISP,W1_eeh,IK1)                       
                                CALL W1_GATHER_KNODESEL(W,NB3_START_eeh_COLLECT,NB3_STOP_eeh_COLLECT,ISP,W3_eeh,IK1)
                                CALL STOP_TIMER(TIMER_GATH)
                             END IF
                          END IF
   
                          ! Only on the node that owns this k-point:
                          IF (IK1==WDES%NB_LOW) THEN
                             NB1_START_eeh=NB1_START_eeh_COLLECT
                             NB1_STOP_eeh =NB1_STOP_eeh_COLLECT
                             NB1_CNT_eeh  =NB1_CNT_eeh_COLLECT
                             NB3_START_eeh=NB3_START_eeh_COLLECT
                             NB3_STOP_eeh =NB3_STOP_eeh_COLLECT
                             NB3_CNT_eeh  =NB3_CNT_eeh_COLLECT
                          END IF
                       END IF
                    END IF


                    !------------------------------
                    ! eeh
                    IF(LAUGER_EHH)THEN
                       ! K1
                       NB1_START_ehh_COLLECT=NB1_MIN_ehh(K1_COLLECT)
                       NB1_STOP_ehh_COLLECT=NB1_MAX_ehh(K1_COLLECT)                       
                       NB1_CNT_ehh_COLLECT=MAX(NB1_STOP_ehh_COLLECT-NB1_START_ehh_COLLECT+1,0)
                       ! K3
                       NB3_START_ehh_COLLECT=NB3_MIN_ehh(K3_COLLECT)
                       NB3_STOP_ehh_COLLECT=NB3_MAX_ehh(K3_COLLECT)                       
                       NB3_CNT_ehh_COLLECT=MAX(NB3_STOP_ehh_COLLECT-NB3_START_ehh_COLLECT+1,0)

                       IF(NB1_CNT_ehh_COLLECT>0.AND.NB3_CNT_ehh_COLLECT>0)THEN
                          DO_ehh=.TRUE.
                          
                          ! get the correct wavefunctions on this node
                          IF(LAUGER_COLLECT)THEN
                             IF (IK1==WDES%NB_LOW) THEN
                                DO I=NB1_START_ehh_COLLECT,NB1_STOP_ehh_COLLECT
                                   W1_ehh(I-NB1_START_ehh_COLLECT+1)%CR=>WF_ALL(I,K1_COLLECT)%CR(:)
                                   W1_ehh(I-NB1_START_ehh_COLLECT+1)%CPROJ=>WF_ALL(I,K1_COLLECT)%CPROJ(:)
                                END DO
                                DO I=NB3_START_ehh_COLLECT,NB3_STOP_ehh_COLLECT
                                   W3_ehh(I-NB3_START_ehh_COLLECT+1)%CR=>WF_ALL(I,K3_COLLECT)%CR(:)
                                   W3_ehh(I-NB3_START_ehh_COLLECT+1)%CPROJ=>WF_ALL(I,K3_COLLECT)%CPROJ(:)
                                END DO
                             END IF
                          ELSE
                             ! On all nodes: collect wavefunctions
                             ! Here the k-points are distributed over nodes (node IK1 receives, all other nodes send)
                             ! must be called by all nodes                       
                             IF(LAUGER_JIT)THEN
                                ! do nothing
                             ELSE
                                CALL START_TIMER(TIMER_GATH)
                                CALL W1_GATHER_KNODESEL(W,NB1_START_ehh_COLLECT,NB1_STOP_ehh_COLLECT,ISP,W1_ehh,IK1)
                                CALL W1_GATHER_KNODESEL(W,NB3_START_ehh_COLLECT,NB3_STOP_ehh_COLLECT,ISP,W3_ehh,IK1)
                                CALL STOP_TIMER(TIMER_GATH)
                             END IF
                          END IF

                          ! Only on the node that owns this k-point:
                          IF (IK1==WDES%NB_LOW) THEN
                             NB1_START_ehh=NB1_START_ehh_COLLECT
                             NB1_STOP_ehh =NB1_STOP_ehh_COLLECT
                             NB1_CNT_ehh  =NB1_CNT_ehh_COLLECT
                             NB3_START_ehh=NB3_START_ehh_COLLECT
                             NB3_STOP_ehh =NB3_STOP_ehh_COLLECT
                             NB3_CNT_ehh  =NB3_CNT_ehh_COLLECT
                          END IF
                       END IF

                    END IF

                    IF(DO_eeh.OR.DO_ehh)THEN
                       ! On all nodes: keep track of slots 
                       K1LIST(IK1)=K1_COLLECT
                       K3LIST(IK1)=K3_COLLECT

                       ! Only on the node that owns this k-point:
                       IF (IK1==WDES%NB_LOW) THEN
                          ! set K1 and K3 indices
                          K1=K1_COLLECT
                          K3=K3_COLLECT
                       ENDIF

                       ! we are done for this slot, go on to the next slot
                       EXIT nextk1
                    END IF

                 END DO nextk1

              END DO k1slots
              CALL STOP_TIMER(TIMER_K1)
              !---------------------------------------------------------------------------


              !---------------------------------------------------------------------------
              ! If all k1-slots are empty, then all k1-points are done
              ! and we can leave the k1-loop
              IF(ALL(K1LIST==0)) THEN
                 EXIT k1loop
              END IF
              ! Otherwise, even if only 1 k-point (i.e. 1 node) has bands,
              ! ALL nodes have to continue, in order to exchange bands

              ! K1 and K3 are nonzero, if there is a k-point with bands
              ! If K1,K3 > 0, there may be only eeh or only ehh or both processes!
              ! CHECK
              IF( K1/=K1LIST(WDES%NB_LOW) ) THEN
                 IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'SOMETHING WRONG',WDES%NB_LOW,K1,K1LIST
                 STOP
              END IF
              !---------------------------------------------------------------------------


              !---------------------------------------------------------------------------              
              ! Get occupation numbers
              ! NB1_START_eeh etc. are set such that these loops are 
              ! executed only if there are bands for eeh process at K1 etc.
              ! Otherwise, START=0 > STOP=-1 and the loops are not entered
              IF(LAUGER_EEH)THEN
                 DO I=NB1_START_eeh,NB1_STOP_eeh
                    OCC1_eeh(I-NB1_START_eeh+1)=HOLE_OCC(I,K1,ISP)
                 END DO
                 DO I=NB3_START_eeh,NB3_STOP_eeh
                    OCC3_eeh(I-NB3_START_eeh+1)=ELEC_OCC(I,K3,ISP) 
                END DO                 
              END IF
              IF(LAUGER_EHH)THEN
                 DO I=NB1_START_ehh,NB1_STOP_ehh
                    OCC1_ehh(I-NB1_START_ehh+1)=HOLE_OCC(I,K1,ISP)
                 END DO
                 DO I=NB3_START_ehh,NB3_STOP_ehh
                    OCC3_ehh(I-NB3_START_ehh+1)=ELEC_OCC(I,K3,ISP)
                 END DO
              END IF
              !---------------------------------------------------------------------------


              !---------------------------------------------------------------------------
              ! wave descriptors must be reset to local K1 and K3
              CALL SETWDES(WDES,WDESK1,K1)              
              CALL SETWDES(WDES,WDESK3,K3)
              !---------------------------------------------------------------------------


              !---------------------------------------------------------------------------
              CALL START_TIMER(TIMER_K2)              
              k2calc: DO K2=1,WDES%NKPTS

                 NB2_START_eeh=0
                 NB2_STOP_eeh=-1
                 NB2_CNT_eeh=0

                 NB2_START_ehh=0
                 NB2_STOP_ehh=-1
                 NB2_CNT_ehh=0

                 NB4_START_eeh=0
                 NB4_STOP_eeh=-1
                 NB4_CNT_eeh=0

                 NB4_START_ehh=0
                 NB4_STOP_ehh=-1
                 NB4_CNT_ehh=0

                 ! k4=k2+q
                 K4=K4TABLE(K2)

                 CALL SETWDES(WDES,WDESK2,K2)
                 CALL SETWDES(WDES,WDESK4,K4)


                 !---------------------------------------------------------------------------
                 IF(LAUGER_EEH)THEN
                    ! K2
                    NB2_START_eeh=NB2_MIN_eeh(K2)
                    NB2_STOP_eeh=NB2_MAX_eeh(K2)
                    NB2_CNT_eeh=MAX(NB2_STOP_eeh-NB2_START_eeh+1,0)
                    ! K4
                    NB4_START_eeh=NB4_MIN_eeh(K4)
                    NB4_STOP_eeh=NB4_MAX_eeh(K4)
                    NB4_CNT_eeh=MAX(NB4_STOP_eeh-NB4_START_eeh+1,0)
                    !
                    DO I=NB2_START_eeh,NB2_STOP_eeh
                       OCC2_eeh(I-NB2_START_eeh+1)=HOLE_OCC(I,K2,ISP)
                    END DO
                    DO I=NB4_START_eeh,NB4_STOP_eeh
                       OCC4_eeh(I-NB4_START_eeh+1)=ELEC_OCC(I,K4,ISP)
                    END DO
                    !
                    IF(.NOT.LAUGER_COLLECT)THEN
                       IF(LAUGER_JIT)THEN
                          ! do nothing
                       ELSE
                          ! if we redistribute on demand, we must do it here,
                          ! because ALL nodes must participate !!!!
                          IF( (NB2_CNT_eeh>0 .AND. NB4_CNT_eeh>0) )THEN
                             CALL START_TIMER(TIMER_GATH)
                             CALL W1_GATHER_GLB(W,NB2_START_eeh,NB2_STOP_eeh,ISP,W2_eeh)
                             CALL W1_GATHER_GLB(W,NB4_START_eeh,NB4_STOP_eeh,ISP,W4_eeh)
                             CALL STOP_TIMER(TIMER_GATH)
                          END IF
                       END IF
                    END IF
                 END IF
                 !---------------------------------------------------------------------------


                 !---------------------------------------------------------------------------
                 IF(LAUGER_EHH)THEN          
                    ! K2
                    NB2_START_ehh=NB2_MIN_ehh(K2)
                    NB2_STOP_ehh=NB2_MAX_ehh(K2)
                    NB2_CNT_ehh=MAX(NB2_STOP_ehh-NB2_START_ehh+1,0)
                    ! K4
                    NB4_START_ehh=NB4_MIN_ehh(K4)
                    NB4_STOP_ehh=NB4_MAX_ehh(K4)
                    NB4_CNT_ehh=MAX(NB4_STOP_ehh-NB4_START_ehh+1,0)
                    !
                    DO I=NB2_START_ehh,NB2_STOP_ehh
                       OCC2_ehh(I-NB2_START_ehh+1)=HOLE_OCC(I,K2,ISP)
                    END DO
                    DO I=NB4_START_ehh,NB4_STOP_ehh
                       OCC4_ehh(I-NB4_START_ehh+1)=ELEC_OCC(I,K4,ISP)
                    END DO
                    !
                    IF(.NOT.LAUGER_COLLECT)THEN
                       IF(LAUGER_JIT)THEN
                          ! do nothing
                       ELSE
                          ! if we redistribute on demand, we must do it here,
                          ! because ALL nodes must participate !!!!
                          IF( (NB2_CNT_ehh>0 .AND. NB4_CNT_ehh>0) )THEN
                             CALL START_TIMER(TIMER_GATH)
                             CALL W1_GATHER_GLB(W,NB2_START_ehh,NB2_STOP_ehh,ISP,W2_ehh)
                             CALL W1_GATHER_GLB(W,NB4_START_ehh,NB4_STOP_ehh,ISP,W4_ehh)
                             CALL STOP_TIMER(TIMER_GATH)
                          END IF
                       END IF
                    END IF
                 END IF
                 !---------------------------------------------------------------------------
                 

                 !---------------------------------------------------------------------------
                 ! Just-In-Time redistribution of wavefunctions
                 IF(LAUGER_JIT)THEN
                    ! For all nodes check, if we need to calculate matrix elements
                    ! If so, we need to redistribute wavefunctions
                    DO_eeh=.FALSE.
                    DO_eeh_GLOBAL=.FALSE.
                    DE_eeh=0

                    !---------------------------------------------------------------------------
                    DO IK1=1,WDES%NB_PAR
                       ! check if node is active (i.e. has a K1-point)
                       IF(K1LIST(IK1)==0)CYCLE

                       !---------------------------------------------------------------------------
                       IF(LAUGER_EEH)THEN
                          NB1_CNT_eeh_GLOBAL=NB1_MAX_eeh(K1LIST(IK1))-NB1_MIN_eeh(K1LIST(IK1))+1
                          NB2_CNT_eeh_GLOBAL=NB2_CNT_eeh
                          NB3_CNT_eeh_GLOBAL=NB3_MAX_eeh(K3LIST(IK1))-NB3_MIN_eeh(K3LIST(IK1))+1
                          NB4_CNT_eeh_GLOBAL=NB4_CNT_eeh

                          IF( MIN(NB1_CNT_eeh_GLOBAL,NB2_CNT_eeh_GLOBAL,NB3_CNT_eeh_GLOBAL,NB4_CNT_eeh_GLOBAL)<=0 )&
                               GOTO 1001

                          ! occupation numbers
                          ! 1: hole, 3: elec
                          DO I1=1,NB1_CNT_eeh_GLOBAL
                             DO I2=1,NB2_CNT_eeh_GLOBAL
                                DO I3=1,NB3_CNT_eeh_GLOBAL
                                   DO I4=1,NB4_CNT_eeh_GLOBAL
                                      !
                                      OCC_eeh(I1,I2,I3,I4)=&
                                           HOLE_OCC(NB1_MIN_eeh(K1LIST(IK1))+I1-1,K1LIST(IK1),ISP)*&
                                           OCC2_eeh(I2)*&
                                           ELEC_OCC(NB3_MIN_eeh(K3LIST(IK1))+I3-1,K3LIST(IK1),ISP)*&
                                           OCC4_eeh(I4)
                                      !
                                   END DO
                                END DO
                             END DO
                          END DO

                          ! energy conservation
                          WIDTH=AUGER_EWIDTH
                          DO I1=1,NB1_CNT_eeh_GLOBAL
                             DO I2=1,NB2_CNT_eeh_GLOBAL
                                DO I3=1,NB3_CNT_eeh_GLOBAL
                                   DO I4=1,NB4_CNT_eeh_GLOBAL
                                      !
                                      IF(LAUGER_DHDK)THEN
                                         WIDTH=SQRT(&
                                                SUM(ABS(DHDK_FULL(I1+NB1_MIN_eeh(K1LIST(IK1))-1,:,K1)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I2+NB2_START_eeh           -1,:,K2)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I3+NB3_MIN_eeh(K3LIST(IK1))-1,:,K3)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              )
                                         WIDTH=AUGER_EWIDTH*WIDTH
                                      END IF
                                      !
                                      DE_eeh_GLOBAL(I1,I2,I3,I4)=DELTA_EN(REAL(&
                                           W%CELTOT(I1+NB1_MIN_eeh(K1LIST(IK1))-1,K1LIST(IK1),ISP)+&
                                           W%CELTOT(I2+NB2_START_eeh           -1,K2         ,ISP)-&
                                           W%CELTOT(I3+NB3_MIN_eeh(K3LIST(IK1))-1,K3LIST(IK1),ISP)-&
                                           W%CELTOT(I4+NB4_START_eeh           -1,K4         ,ISP) &
                                           ,q),W=WIDTH)
                                      !
                                   END DO
                                END DO
                             END DO
                          END DO

                          DE_eeh_GLOBAL=DE_eeh_GLOBAL*OCC_eeh
                          ! If all matrix elements are small, we can cycle
                          IF( MAXVAL(DE_eeh_GLOBAL(1:NB1_CNT_eeh_GLOBAL,1:NB2_CNT_eeh_GLOBAL,&
                               1:NB3_CNT_eeh_GLOBAL,1:NB4_CNT_eeh_GLOBAL))<=AUGER_TINY_eeh ) GOTO 1001

                          ! Otherwise, redistribute
                          CALL SETWDES(WDES,WDESK1,K1LIST(IK1))
                          CALL SETWDES(WDES,WDESK3,K3LIST(IK1))
                          CALL START_TIMER(TIMER_GATH)
                          CALL W1_GATHER_KNODESEL(W,NB1_MIN_eeh(K1LIST(IK1)),NB1_MAX_eeh(K1LIST(IK1)),ISP,W1_eeh,IK1)
                          CALL W1_GATHER_KNODESEL(W,NB3_MIN_eeh(K3LIST(IK1)),NB3_MAX_eeh(K3LIST(IK1)),ISP,W3_eeh,IK1)
                          CALL STOP_TIMER(TIMER_GATH)

                          ! If there is at least one K1-point that needs calculation,
                          ! we need to redistribute W2,W4 as well (after finishing K1-loop)
                          DO_eeh_GLOBAL=.TRUE.
                          ! If K1 is on local node, calculate matrix element
                          IF(IK1==WDES%NB_LOW)THEN
                             DO_eeh=.TRUE.
                             DE_eeh=DE_eeh_GLOBAL
                          END IF

1001                   END IF
                       !---------------------------------------------------------------------------

                       !---------------------------------------------------------------------------
                       IF(LAUGER_EHH)THEN
                          NB1_CNT_ehh_GLOBAL=NB1_MAX_ehh(K1LIST(IK1))-NB1_MIN_ehh(K1LIST(IK1))+1
                          NB2_CNT_ehh_GLOBAL=NB2_CNT_ehh
                          NB3_CNT_ehh_GLOBAL=NB3_MAX_ehh(K3LIST(IK1))-NB3_MIN_ehh(K3LIST(IK1))+1
                          NB4_CNT_ehh_GLOBAL=NB4_CNT_ehh

                          IF( MIN(NB1_CNT_ehh_GLOBAL,NB2_CNT_ehh_GLOBAL,NB3_CNT_ehh_GLOBAL,NB4_CNT_ehh_GLOBAL)<=0 )&
                               GOTO 1002

                          ! occupation numbers
                          ! 1: hole, 3: elec
                          DO I1=1,NB1_CNT_ehh_GLOBAL
                             DO I2=1,NB2_CNT_ehh_GLOBAL
                                DO I3=1,NB3_CNT_ehh_GLOBAL
                                   DO I4=1,NB4_CNT_ehh_GLOBAL
                                      !
                                      OCC_ehh(I1,I2,I3,I4)=&
                                           HOLE_OCC(NB1_MIN_ehh(K1LIST(IK1))+I1-1,K1LIST(IK1),ISP)*&
                                           OCC2_ehh(I2)*&
                                           ELEC_OCC(NB3_MIN_ehh(K3LIST(IK1))+I3-1,K3LIST(IK1),ISP)*&
                                           OCC4_ehh(I4)
                                      !
                                   END DO
                                END DO
                             END DO
                          END DO

                          ! energy conservation
                          WIDTH=AUGER_EWIDTH
                          DO I1=1,NB1_CNT_ehh_GLOBAL
                             DO I2=1,NB2_CNT_ehh_GLOBAL
                                DO I3=1,NB3_CNT_ehh_GLOBAL
                                   DO I4=1,NB4_CNT_ehh_GLOBAL
                                      !
                                      IF(LAUGER_DHDK)THEN
                                         WIDTH=SQRT(&
                                                SUM(ABS(DHDK_FULL(I1+NB1_MIN_ehh(K1LIST(IK1))-1,:,K1)-DHDK_FULL(I4+NB4_START_ehh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I2+NB2_START_ehh           -1,:,K2)-DHDK_FULL(I4+NB4_START_ehh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I3+NB3_MIN_ehh(K3LIST(IK1))-1,:,K3)-DHDK_FULL(I4+NB4_START_ehh-1,:,K4))**2)&
                                              )
                                         WIDTH=AUGER_EWIDTH*WIDTH
                                      END IF
                                      !
                                      DE_ehh_GLOBAL(I1,I2,I3,I4)=DELTA_EN(REAL(&
                                           W%CELTOT(I1+NB1_MIN_ehh(K1LIST(IK1))-1,K1LIST(IK1),ISP)+&
                                           W%CELTOT(I2+NB2_START_ehh           -1,K2         ,ISP)-&
                                           W%CELTOT(I3+NB3_MIN_ehh(K3LIST(IK1))-1,K3LIST(IK1),ISP)-&
                                           W%CELTOT(I4+NB4_START_ehh           -1,K4         ,ISP)&
                                           ,q),W=WIDTH)
                                      !
                                   END DO
                                END DO
                             END DO
                          END DO

                          DE_ehh_GLOBAL=DE_ehh_GLOBAL*OCC_ehh
                          ! If all matrix elements are small, we can cycle
                          IF( MAXVAL(DE_ehh_GLOBAL(1:NB1_CNT_ehh_GLOBAL,1:NB2_CNT_ehh_GLOBAL,&
                               1:NB3_CNT_ehh_GLOBAL,1:NB4_CNT_ehh_GLOBAL))<=AUGER_TINY_ehh ) GOTO 1002

                          ! Otherwise, redistribute
                          CALL SETWDES(WDES,WDESK1,K1LIST(IK1))
                          CALL SETWDES(WDES,WDESK3,K3LIST(IK1))
                          CALL START_TIMER(TIMER_GATH)
                          CALL W1_GATHER_KNODESEL(W,NB1_MIN_ehh(K1LIST(IK1)),NB1_MAX_ehh(K1LIST(IK1)),ISP,W1_ehh,IK1)
                          CALL W1_GATHER_KNODESEL(W,NB3_MIN_ehh(K3LIST(IK1)),NB3_MAX_ehh(K3LIST(IK1)),ISP,W3_ehh,IK1)
                          CALL STOP_TIMER(TIMER_GATH)

                          ! If there is at least one K1-point that needs calculation,
                          ! we need to redistribute W2,W4 as well (after finishing K1-loop)
                          DO_ehh_GLOBAL=.TRUE.
                          ! If K1 is on local node, calculate matrix element
                          IF(IK1==WDES%NB_LOW)THEN
                             DO_ehh=.TRUE.
                             DE_ehh=DE_ehh_GLOBAL
                          END IF

1002                   END IF ! LAUGER_EHH
                       !---------------------------------------------------------------------------

                    END DO ! IK1
                    !---------------------------------------------------------------------------

                    CALL SETWDES(WDES,WDESK1,K1)              
                    CALL SETWDES(WDES,WDESK3,K3)

                    IF(DO_eeh_GLOBAL)THEN
                       CALL START_TIMER(TIMER_GATH)
                       CALL W1_GATHER_GLB(W,NB2_START_eeh,NB2_STOP_eeh,ISP,W2_eeh)
                       CALL W1_GATHER_GLB(W,NB4_START_eeh,NB4_STOP_eeh,ISP,W4_eeh)
                       CALL STOP_TIMER(TIMER_GATH)
                    END IF

                    IF(DO_ehh_GLOBAL)THEN
                       CALL START_TIMER(TIMER_GATH)
                       CALL W1_GATHER_GLB(W,NB2_START_ehh,NB2_STOP_ehh,ISP,W2_ehh)
                       CALL W1_GATHER_GLB(W,NB4_START_ehh,NB4_STOP_ehh,ISP,W4_ehh)
                       CALL STOP_TIMER(TIMER_GATH)
                    END IF
                    
                 ELSE


                    !---------------------------------------------------------------------------
                    ! ... BUT only nodes with local K1,K3 need to actually calculate matrix elements
                    DO_eeh=.FALSE.
                    IF(LAUGER_EEH)THEN
                       IF( MIN(NB1_CNT_eeh,NB2_CNT_eeh,NB3_CNT_eeh,NB4_CNT_eeh)>0 ) THEN                       
                          ! occupation numbers
                          DO I1=1,NB1_CNT_eeh
                             DO I2=1,NB2_CNT_eeh
                                DO I3=1,NB3_CNT_eeh
                                   DO I4=1,NB4_CNT_eeh
                                      OCC_eeh(I1,I2,I3,I4)=OCC1_eeh(I1)*OCC2_eeh(I2)*OCC3_eeh(I3)*OCC4_eeh(I4)
                                   END DO
                                END DO
                             END DO
                          END DO
                          ! energy conservation
                          WIDTH=AUGER_EWIDTH
                          DO I1=1,NB1_CNT_eeh
                             DO I2=1,NB2_CNT_eeh
                                DO I3=1,NB3_CNT_eeh
                                   DO I4=1,NB4_CNT_eeh
                                      !
                                      IF(LAUGER_DHDK)THEN
                                         WIDTH=SQRT(&
                                              SUM(ABS(DHDK_FULL(I1+NB1_START_eeh-1,:,K1)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I2+NB2_START_eeh-1,:,K2)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I3+NB3_START_eeh-1,:,K3)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              )
                                         WIDTH=AUGER_EWIDTH*WIDTH
                                      END IF
                                      DE_eeh(I1,I2,I3,I4)=DELTA_EN(REAL(&
                                           W%CELTOT(I1+NB1_START_eeh-1,K1,ISP)+W%CELTOT(I2+NB2_START_eeh-1,K2,ISP)&
                                           -W%CELTOT(I3+NB3_START_eeh-1,K3,ISP)-W%CELTOT(I4+NB4_START_eeh-1,K4,ISP),q),W=WIDTH)
                                      !
                                   END DO
                                END DO
                             END DO
                          END DO

                          DE_eeh=DE_eeh*OCC_eeh
                          IF( MAXVAL(DE_eeh(1:NB1_CNT_eeh,1:NB2_CNT_eeh,1:NB3_CNT_eeh,1:NB4_CNT_eeh))>AUGER_TINY_eeh ) THEN
                             DO_eeh=.TRUE.
                          END IF
                       END IF
                    END IF ! LAUGER_EEH
                    !---------------------------------------------------------------------------


                    !---------------------------------------------------------------------------
                    ! ... BUT only nodes with local K1,K3 need to actually calculate matrix elements
                    DO_ehh=.FALSE.
                    IF(LAUGER_EHH)THEN
                       IF( MIN(NB1_CNT_ehh,NB2_CNT_ehh,NB3_CNT_ehh,NB4_CNT_ehh)>0 ) THEN                       
                          ! occupation numbers
                          DO I1=1,NB1_CNT_ehh
                             DO I2=1,NB2_CNT_ehh
                                DO I3=1,NB3_CNT_ehh
                                   DO I4=1,NB4_CNT_ehh
                                      OCC_ehh(I1,I2,I3,I4)=OCC1_ehh(I1)*OCC2_ehh(I2)*OCC3_ehh(I3)*OCC4_ehh(I4)
                                   END DO
                                END DO
                             END DO
                          END DO
                          ! energy conservation
                          WIDTH=AUGER_EWIDTH
                          DO I1=1,NB1_CNT_ehh
                             DO I2=1,NB2_CNT_ehh
                                DO I3=1,NB3_CNT_ehh
                                   DO I4=1,NB4_CNT_ehh
                                      IF(LAUGER_DHDK)THEN
                                         WIDTH=SQRT(&
                                              SUM(ABS(DHDK_FULL(I1+NB1_START_eeh-1,:,K1)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I2+NB2_START_eeh-1,:,K2)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              + SUM(ABS(DHDK_FULL(I3+NB3_START_eeh-1,:,K3)-DHDK_FULL(I4+NB4_START_eeh-1,:,K4))**2)&
                                              )
                                         WIDTH=AUGER_EWIDTH*WIDTH
                                      END IF
                                      DE_ehh(I1,I2,I3,I4)=DELTA_EN(REAL(&
                                           W%CELTOT(I1+NB1_START_ehh-1,K1,ISP)+W%CELTOT(I2+NB2_START_ehh-1,K2,ISP)&
                                           -W%CELTOT(I3+NB3_START_ehh-1,K3,ISP)-W%CELTOT(I4+NB4_START_ehh-1,K4,ISP),q),W=WIDTH)
                                   END DO
                                END DO
                             END DO
                          END DO
                          DE_ehh=DE_ehh*OCC_ehh
                          IF( MAXVAL(DE_ehh(1:NB1_CNT_ehh,1:NB2_CNT_ehh,1:NB3_CNT_ehh,1:NB4_CNT_ehh))>AUGER_TINY_ehh ) THEN
                             DO_ehh=.TRUE.
                          END IF
                       END IF
                    END IF ! LAUGER_EHH
                    !---------------------------------------------------------------------------

                 END IF ! LAUGER_JIT
                 !---------------------------------------------------------------------------


                 !---------------------------------------------------------------------------
                 CALL START_TIMER(TIMER_RATE)              
                 matrixeeh: IF(DO_eeh)THEN

                    !---------------------------------------------------------------------------
                    IF(LAUGER_COLLECT)THEN
                       ! copy wavefunctions only on local node
                       DO I=NB2_START_eeh,NB2_STOP_eeh
                          W2_eeh(I-NB2_START_eeh+1)%CR=>WF_ALL(I,K2)%CR(:)
                          W2_eeh(I-NB2_START_eeh+1)%CPROJ=>WF_ALL(I,K2)%CPROJ(:)
                       END DO
                       DO I=NB4_START_eeh,NB4_STOP_eeh
                          W4_eeh(I-NB4_START_eeh+1)%CR=>WF_ALL(I,K4)%CR(:)
                          W4_eeh(I-NB4_START_eeh+1)%CPROJ=>WF_ALL(I,K4)%CPROJ(:)
                       END DO
                    END IF
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! which matrix elements need to be calculated
                    BANDS13=0
                    BANDS14=0
                    BANDS24=0
                    BANDS23=0
                    DO I1=1,NB1_CNT_eeh
                       DO I2=1,NB2_CNT_eeh
                          DO I3=1,NB3_CNT_eeh
                             DO I4=1,NB4_CNT_eeh
                                IF(DE_eeh(I1,I2,I3,I4)>AUGER_TINY_eeh)THEN
                                   BANDS13((I1-1)*NB3_CNT_eeh+I3)=1
                                   BANDS14((I1-1)*NB4_CNT_eeh+I4)=1
                                   BANDS24((I2-1)*NB4_CNT_eeh+I4)=1
                                   BANDS23((I2-1)*NB3_CNT_eeh+I3)=1
                                END IF
                             END DO
                          END DO
                       END DO
                    END DO
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! eeh direct
                    M4O_eeh_dir=0
                    CALL START_TIMER(TIMER_MAT)              
                    CALL AUGER_2E4O(W, LATT_CUR, ISP, WGW, FSG, &
                         W1_eeh(1:NB1_CNT_eeh), K1, W2_eeh(1:NB2_CNT_eeh), K2, &
                         W3_eeh(1:NB3_CNT_eeh), K3, W4_eeh(1:NB4_CNT_eeh), K4, &
                         BANDS13, BANDS24, &
                         NFLOAT, NFFT, NSTRIP, M4O_eeh_dir)
                    CALL STOP_TIMER(TIMER_MAT)
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! eeh exchange
                    ! exchange 3 and 4 because the have the same number of orbitals
                    M4O_eeh_ex=0
                    ! TEST
                    IF(K3==K4)THEN
                       ! 
                       ! M4O_eeh_dir and M4O_eeh_ex are the same
                       M4O_eeh_ex=M4O_eeh_dir
                    ELSE
                       CALL START_TIMER(TIMER_MAT)              
                       CALL AUGER_2E4O(W, LATT_CUR, ISP, WGW, FSG, &
                            W1_eeh(1:NB1_CNT_eeh), K1, W2_eeh(1:NB2_CNT_eeh), K2, &
                            W4_eeh(1:NB4_CNT_eeh), K4, W3_eeh(1:NB3_CNT_eeh), K3, &
                            BANDS14, BANDS23, &
                            NFLOAT, NFFT, NSTRIP, M4O_eeh_ex)
                       CALL STOP_TIMER(TIMER_MAT)              
                       ! M40_eeh_ex must be transposed wrt. n3 and n4
                       DO I1=1,NB1_CNT_eeh
                          DO I2=1,NB2_CNT_eeh                          
                             M4O_eeh_ex(I1,I2,:,:)=TRANSPOSE(M4O_eeh_ex(I1,I2,:,:))
                          END DO
                       END DO
                    END IF
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! direct and exchange contribution
                    ! |M|^2 = 2( |M_dir|^2 + |M_ex|^2 + |M_dir+M_ex|^2 ) 
                    !       = 2( 2 |M_dir|^2 + 2 |M_ex|^2 + 2 Re[ M_dir*M_ex ] )
                    M4O_eeh=4._q*REAL( GCONJG(M4O_eeh_dir)*M4O_eeh_dir + GCONJG(M4O_eeh_ex)*M4O_eeh_ex - REAL(GCONJG(M4O_eeh_ex)*M4O_eeh_dir),q )

                    ! multiply by
                    ! - occupation probability,
                    ! - energy window function 
                    M4O_eeh=M4O_eeh*DE_eeh

                    ! sum over all bands 
                    ! and multiply by k-point integration weight (according to symmetry)
                    ! int dk1 dk2 dk3  = int dk1 dk2 dq
                    R_eeh=R_eeh+SUM(M4O_eeh(1:NB1_CNT_eeh,1:NB2_CNT_eeh,1:NB3_CNT_eeh,1:NB4_CNT_eeh) )*KPOINTS%WTKPT(K1)*KPOINTS%WTKPT(K2)*KPOINTS_IBZ%WTKPT(NQPOINT)

                    ! number of matrix elements calculated
                    C_eeh=C_eeh+NB1_CNT_eeh*NB2_CNT_eeh*NB3_CNT_eeh*NB4_CNT_eeh
                    ! number of times we pass here
                    K_eeh=K_eeh+1
                    !---------------------------------------------------------------------------

                 END IF matrixeeh
                 CALL STOP_TIMER(TIMER_RATE)              
                 !---------------------------------------------------------------------------
                 

                 !---------------------------------------------------------------------------                 
                 CALL START_TIMER(TIMER_RATE)              
                 matrixehh: IF(DO_ehh)THEN

                    !---------------------------------------------------------------------------
                    IF(LAUGER_COLLECT)THEN
                       ! copy wavefunctions only on local node
                       DO I=NB2_START_ehh,NB2_STOP_ehh
                          W2_ehh(I-NB2_START_ehh+1)%CR=>WF_ALL(I,K2)%CR(:)
                          W2_ehh(I-NB2_START_ehh+1)%CPROJ=>WF_ALL(I,K2)%CPROJ(:)
                       END DO
                       DO I=NB4_START_ehh,NB4_STOP_ehh
                          W4_ehh(I-NB4_START_ehh+1)%CR=>WF_ALL(I,K4)%CR(:)
                          W4_ehh(I-NB4_START_ehh+1)%CPROJ=>WF_ALL(I,K4)%CPROJ(:)
                       END DO
                    END IF
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! which matrix elements need to be calculated
                    BANDS13=0
                    BANDS14=0
                    BANDS24=0
                    BANDS23=0
                    DO I1=1,NB1_CNT_ehh
                       DO I2=1,NB2_CNT_ehh
                          DO I3=1,NB3_CNT_ehh
                             DO I4=1,NB4_CNT_ehh
                                IF(DE_ehh(I1,I2,I3,I4)>AUGER_TINY_ehh)THEN
                                   BANDS13((I1-1)*NB3_CNT_ehh+I3)=1
                                   BANDS14((I1-1)*NB4_CNT_ehh+I4)=1
                                   BANDS24((I2-1)*NB4_CNT_ehh+I4)=1
                                   BANDS23((I2-1)*NB3_CNT_ehh+I3)=1
                                END IF
                             END DO
                          END DO
                       END DO
                    END DO
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! ehh direct
                    M4O_ehh_dir=0
                    CALL START_TIMER(TIMER_MAT)              
                    CALL AUGER_2E4O(W, LATT_CUR, ISP, WGW, FSG, &
                         W1_ehh(1:NB1_CNT_ehh), K1, W2_ehh(1:NB2_CNT_ehh), K2, &
                         W3_ehh(1:NB3_CNT_ehh), K3, W4_ehh(1:NB4_CNT_ehh), K4, &
                         BANDS13, BANDS24, &
                         NFLOAT, NFFT, NSTRIP, M4O_ehh_dir)
                    CALL STOP_TIMER(TIMER_MAT)
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! ehh exchange:
                    ! exchange 1 and 2 - both holes in valence band, have same number of orbitals
                    M4O_ehh_ex=0
                    IF(K1==K2)THEN
                       ! 
                       ! M4O_eeh_dir and M4O_eeh_ex are the same
                       M4O_eeh_ex=M4O_eeh_dir
                    ELSE
                       CALL START_TIMER(TIMER_MAT)              
                       CALL AUGER_2E4O(W, LATT_CUR, ISP, WGW, FSG, &
                            W2_ehh(1:NB2_CNT_ehh), K2, W1_ehh(1:NB1_CNT_ehh), K1, &
                            W3_ehh(1:NB3_CNT_ehh), K3, W4_ehh(1:NB4_CNT_ehh), K4, &
                            BANDS23, BANDS14, &
                            NFLOAT, NFFT, NSTRIP, M4O_ehh_ex)
                       CALL STOP_TIMER(TIMER_MAT)           
                       ! M40_ehh_ex must be transposed wrt. 1 and 2
                       DO I3=1,NB3_CNT_ehh
                          DO I4=1,NB4_CNT_ehh                          
                             M4O_ehh_ex(:,:,I3,I4)=TRANSPOSE(M4O_ehh_ex(:,:,I3,I4))
                          END DO
                       END DO
                    END IF
                    !---------------------------------------------------------------------------

                    !---------------------------------------------------------------------------
                    ! direct and exchange contribution
                    M4O_ehh=4._q*REAL( GCONJG(M4O_ehh_dir)*M4O_ehh_dir + GCONJG(M4O_ehh_ex)*M4O_ehh_ex - REAL(GCONJG(M4O_ehh_ex)*M4O_ehh_dir),q )

                    ! multiply by
                    ! - occupation probability
                    ! - energy window function
                    M4O_ehh=M4O_ehh*DE_ehh

                    ! sum over all bands
                    ! and multiply by k-point integration weights
                    R_ehh=R_ehh+SUM(M4O_ehh(1:NB1_CNT_ehh,1:NB2_CNT_ehh,1:NB3_CNT_ehh,1:NB4_CNT_ehh) )*KPOINTS%WTKPT(K1)*KPOINTS%WTKPT(K2)*KPOINTS_IBZ%WTKPT(NQPOINT)

                    ! number of matrix elements calculated
                    C_ehh=C_ehh+NB1_CNT_ehh*NB2_CNT_ehh*NB3_CNT_ehh*NB4_CNT_ehh
                    ! number of times we pass here
                    K_ehh=K_ehh+1
                    !---------------------------------------------------------------------------

                 END IF matrixehh
                 CALL STOP_TIMER(TIMER_RATE)              
                 !---------------------------------------------------------------------------

              ENDDO k2calc
              CALL STOP_TIMER(TIMER_K2)
              !---------------------------------------------------------------------------

           ENDDO k1loop
           !---------------------------------------------------------------------------

        ENDDO qloop
        CALL STOP_TIMER(TIMER_NQ)
        !---------------------------------------------------------------------------

        CALL STOP_TIMER(TIMER_ALL)


        !---------------------------------------------------------------------------
        ! Sum Auger rate R over all nodes
        ! sum over k1
        CALLMPI( M_sum_d(WDES%COMM_INTER, R_eeh, 1) )
        ! sum over q
        CALLMPI( M_sum_d(WDES%COMM_KINTER, R_eeh, 1) )

        CALLMPI( M_sum_d(WDES%COMM_INTER, R_ehh, 1) )
        CALLMPI( M_sum_d(WDES%COMM_KINTER, R_ehh, 1) )

        ! Divide by 2 to correct for either
        ! EEH case:
        ! a) If k3/=k4 => double counting
        !    Both k3,k4 and k4,k3 appear when iterating over all k-points.
        !    Actually, the double counting comes in only when we weight the Q-points in the IBZ
        !    according to their symmetry, since for k3,k4 we have q, and for k4,k3 we have -q,
        !    i.e. only one combination is actually calculated. Thus we cannot avoid the double 
        !    counting if we want to make use of the IBZ reduction but it does not cost any
        !    additional effort, anyway. !
        ! b) If k3==k4 => 
        !    if n3/=n4 => double counting again (this time not by loop over k-points but
        !                 because for k3,k4 we calculate all matrix elements n3,n4
        !                 and thus both n3,n4 and n4,n3 occur
        !    if n3==n4 => in this case the initial state is k3n3,k3n3 and is symmetric
        !                 Thus the spin state must be antisymmetric and there is only 1 state (singlet)
        !                 <33 Singlet | A[12] Singlet > = 1/sqrt(2) ( <33|12> + <33|21> ) = sqrt(2) <33|12>
        !                 |M|^2 = 2 |M_D|^2
        !                 But we calculate 2 ( |M_D|^2 + |M_E|^2 + |M_D-M_E|^2 ) = 4 |M_D|^2 (since M_D==M_E)
        !                 Hence, also a factor 2 too much.
        !
        ! NOTE: double counting k1n1,k2n2 does not occur since the bands are always different
        !       (n1 conduction band, n2 valence band)
        R_eeh=TPI/HBAR/LATT_CUR%OMEGA*R_eeh/2._q
        R_ehh=TPI/HBAR/LATT_CUR%OMEGA*R_ehh/2._q

        IF(LDUMP_AUGER)THEN
           IF(LAUGER_EEH) WRITE(AUGER_IO%IU0,*) 'R_eeh',R_eeh
           IF(LAUGER_EHH) WRITE(AUGER_IO%IU0,*) 'R_ehh',R_ehh
        END IF
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Sum counters over all nodes
        CALLMPI( M_sum_d(WDES%COMM_INTER, C_eeh, 1) )
        CALLMPI( M_sum_d(WDES%COMM_KINTER, C_eeh, 1) )

        CALLMPI( M_sum_d(WDES%COMM_INTER, C_ehh, 1) )
        CALLMPI( M_sum_d(WDES%COMM_KINTER, C_ehh, 1) )

        CALLMPI( M_sum_d(WDES%COMM_INTER, K_eeh, 1) )
        CALLMPI( M_sum_d(WDES%COMM_KINTER, K_eeh, 1) )

        CALLMPI( M_sum_d(WDES%COMM_INTER, K_ehh, 1) )
        CALLMPI( M_sum_d(WDES%COMM_KINTER, K_ehh, 1) )

        CALLMPI( M_sum_i(WDES%COMM_INTER, NFLOAT, 1) )
        CALLMPI( M_sum_i(WDES%COMM_KINTER, NFLOAT, 1) )

        IF(LDUMP_AUGER)THEN
           WRITE(AUGER_IO%IU0,*) 'C_eeh',WDES%NB_LOW,C_eeh
           WRITE(AUGER_IO%IU0,*) 'C_ehh',WDES%NB_LOW,C_ehh
           WRITE(AUGER_IO%IU0,*) 'K_eeh',WDES%NB_LOW,K_eeh
           WRITE(AUGER_IO%IU0,*) 'K_ehh',WDES%NB_LOW,K_ehh
           WRITE(AUGER_IO%IU0,*) 'NFLOAT',NFLOAT
        END IF
        
        CALL PRINT_TIMERS(WDES)
        !---------------------------------------------------------------------------


        !---------------------------------------------------------------------------
        ! Free all resources
        IF(LAUGER_EEH)THEN
           DO I=1,SIZE(W1_eeh)
              CALL DELWAV(W1_eeh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W2_eeh)
              CALL DELWAV(W2_eeh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W3_eeh)
              CALL DELWAV(W3_eeh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W4_eeh)
              CALL DELWAV(W4_eeh(I),.TRUE.)
           END DO
           DEALLOCATE(M4O_eeh,M4O_eeh_dir,M4O_eeh_ex,&
                OCC1_eeh,OCC2_eeh,OCC3_eeh,OCC4_eeh,&
                OCC_eeh,DE_eeh,&
                W1_eeh,W2_eeh,W3_eeh,W4_eeh)
        END IF
        IF(LAUGER_EHH)THEN
           DO I=1,SIZE(W1_ehh)
              CALL DELWAV(W1_ehh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W2_ehh)
              CALL DELWAV(W2_ehh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W3_ehh)
              CALL DELWAV(W3_ehh(I),.TRUE.)
           END DO
           DO I=1,SIZE(W4_ehh)
              CALL DELWAV(W4_ehh(I),.TRUE.)
           END DO
           DEALLOCATE(M4O_ehh,M4O_ehh_dir,M4O_ehh_ex,&
                OCC1_ehh,OCC2_ehh,OCC3_ehh,OCC4_ehh,&
                OCC_ehh,DE_ehh,&
                W1_ehh,W2_ehh,W3_ehh,W4_ehh)
        END IF


        DEALLOCATE(ELEC_OCC,HOLE_OCC,K1LIST)
        CALL DEALLOCWDES(WGW,.TRUE.)
        DEALLOCATE(WGW,GRIDWGW)

        DEALLOCATE(NB1_MIN_eeh,NB2_MIN_eeh,NB3_MIN_eeh,NB4_MIN_eeh,&
             NB1_MAX_eeh,NB2_MAX_eeh,NB3_MAX_eeh,NB4_MAX_eeh,&
             NB1_MIN_ehh,NB2_MIN_ehh,NB3_MIN_ehh,NB4_MIN_ehh,&
             NB1_MAX_ehh,NB2_MAX_ehh,NB3_MAX_ehh,NB4_MAX_ehh)

        IF(LAUGER_COLLECT)THEN
#ifdef use_shmem
           IF(NCSHMEM>0)THEN
              DO IK=1,WDES%NKPTS
                 DO IB=NB_MIN,NB_MAX
                    CALL DELWAV_SHMEM(WF_ALL(IB,IK),WDES1,ADDRESS(IB,IK),SHMID(IB,IK))
                 ENDDO
              ENDDO
              DEALLOCATE(ADDRESS,SHMID)
           ELSE
#endif
              DO IK=1,WDES%NKPTS
                 DO IB=NB_MIN,NB_MAX
                    CALL DELWAV_NOCW(WF_ALL(IB,IK))
                 ENDDO
              ENDDO
#ifdef use_shmem
           END IF
#endif
        END IF
        !---------------------------------------------------------------------------

        RETURN
      END SUBROUTINE CALCULATE_AUGER


      !************************* SUBROUTINE ENERGY_RANGE ************************
      ! Determine energy ranges for K1,K2,K3,K4 for which non-negligible
      ! occupation numbers are found.
      ! Non-negligible means that the occupation is larger than
      ! (occuption at CBM)*AUGER_OCC_FAC (for electrons) or 
      ! (occuption at VBM)*AUGER_OCC_FAC (for hols)
      ! From this the value of AUGER_TINY is calculated, i.e. the minimum value
      ! of the occupation factor in the Auger rate. If we assume the matrix element
      ! is on the order of 1, than any contribution to R must be at least AUGER_TINY
      !**************************************************************************
      SUBROUTINE ENERGY_RANGE(W,WDES,&
           EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
           EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh,&
           AUGER_TINY_eeh,AUGER_TINY_ehh)
        USE wave_high
        USE constant
        IMPLICIT NONE
        TYPE(wavespin) :: W
        TYPE(wavedes) :: WDES
        REAL(q) :: AUGER_TINY_eeh,AUGER_TINY_ehh,&
             EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
             EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh
        !
        REAL(q) EVBM,ECBM,EG,EMIN
        REAL(q) EV1_eeh,EV2_eeh,EV3_eeh,EC1_eeh,EC2_eeh,EC3_eeh,EMAX_eeh
        REAL(q) EV1_ehh,EV2_ehh,EV3_ehh,EC1_ehh,EC2_ehh,EC3_ehh,EMAX_ehh
        REAL(q) :: HOLE_OCC_THRESH_eeh,ELEC_OCC_THRESH_eeh
        REAL(q) :: HOLE_OCC_THRESH_ehh,ELEC_OCC_THRESH_ehh
        REAL(q) :: F_CBM,G_VBM,EDENS,HDENS
        INTEGER :: ISP,IK,IB

        ! calculate a range of parameters that will define
        ! the various energy intervals relevant to the
        ! eeh and ehh Auger recombination processes.
        EVBM=MINVAL(REAL(W%CELTOT,q)); ECBM=MAXVAL(REAL(W%CELTOT,q))
        IF(AUGER_EVBHI>-1E30_q)THEN
           EVBM=MAXVAL(REAL(W%CELTOT,q),REAL(W%CELTOT,q)<AUGER_EVBHI)
        ELSE
           EVBM=MAXVAL(REAL(W%CELTOT,q),W%FERTOT>AUGER_OCCUPY_THRESHOLD)
        END IF
        IF(AUGER_ECBLO>-1E30_q)THEN
           ECBM=MINVAL(REAL(W%CELTOT,q),REAL(W%CELTOT,q)>AUGER_ECBLO)
        ELSE
           ECBM=MINVAL(REAL(W%CELTOT,q),W%FERTOT<=AUGER_OCCUPY_THRESHOLD)
        END IF

        ! energy gap
        EG=ECBM-EVBM
        ! minimum transition energy
        EMIN=EG-AUGER_EWIDTH

        ! occupation threshold for valence bands:
        ! we take a fraction of the occupation at the VB maximum
        ! CAUTION: do not compute 1-FERMI because FERMI is nearly 1 !
        G_VBM=ANTIFERMI(EVBM,AUGER_MU,AUGER_SIGMA)
        F_CBM=FERMI(ECBM,AUGER_MU,AUGER_SIGMA)

        !---------------------------------------------
        ! eeh: EV1,EVBM,ECBM,EC1,EC2,EC3
        IF(LAUGER_EEH)THEN
           ELEC_OCC_THRESH_eeh=F_CBM*AUGER_OCC_FAC_EEH
           HOLE_OCC_THRESH_eeh=G_VBM*AUGER_OCC_FAC_EEH
           ! occupation cannot get smaller than this (0.1 safety factor)
           AUGER_TINY_eeh=ELEC_OCC_THRESH_eeh*ELEC_OCC_THRESH_eeh*HOLE_OCC_THRESH_eeh*1E-1_q
           IF(LDUMP_AUGER)THEN
              WRITE(AUGER_IO%IU0,*) 'HOLE_OCC_THRESH_eeh',HOLE_OCC_THRESH_eeh
              WRITE(AUGER_IO%IU0,*) 'ELEC_OCC_THRESH_eeh',ELEC_OCC_THRESH_eeh
              WRITE(AUGER_IO%IU0,*) 'AUGER_TINY_eeh',AUGER_TINY_eeh
           END IF

           ! e conc. zero above EC1
           EC1_eeh=AUGER_SIGMA*LOG(1._q/ELEC_OCC_THRESH_eeh-1._q)+AUGER_MU
           ! h conc. zero below EV1
           EV1_eeh=-AUGER_SIGMA*LOG(1._q/HOLE_OCC_THRESH_eeh-1._q)+AUGER_MU
           ! maximum transition energy
           EMAX_eeh=EC1_eeh-EV1_eeh+AUGER_EWIDTH

           ! lowest accessible CB energy
           EC2_eeh=ECBM+EMIN
           ! highest accessible CB energy
           EC3_eeh=EC1_eeh+EMAX_eeh

           ! sanity check
           IF( (.NOT.(EV1_eeh<EVBM)) &
                .OR. (.NOT.(EC3_eeh>EC2_eeh .AND. EC2_eeh>ECBM .AND. EC1_eeh>ECBM)) )THEN
              IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'CALCULATE_AUGER: Error in energy bounds _eeh: ',EV1_eeh,EVBM,ECBM,EC1_eeh,EC2_eeh,EC3_eeh
              STOP
           END IF

           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'eeh: EV1,EVBM,ECBM,EC1,EC2,EC3',EV1_eeh,EVBM,ECBM,EC1_eeh,EC2_eeh,EC3_eeh              
           EMIN1_eeh=EC2_eeh
           EMAX1_eeh=EC3_eeh
           EMIN2_eeh=EV1_eeh
           EMAX2_eeh=EVBM
           EMIN3_eeh=ECBM
           EMAX3_eeh=EC1_eeh
           EMIN4_eeh=EMIN3_eeh
           EMAX4_eeh=EMAX3_eeh
        END IF


        !---------------------------------------------
        ! ehh: EV3,EV2,EV1,EVBM,ECBM,EC1
        IF(LAUGER_EHH)THEN
           ELEC_OCC_THRESH_ehh=F_CBM*AUGER_OCC_FAC_EHH
           HOLE_OCC_THRESH_ehh=G_VBM*AUGER_OCC_FAC_EHH
           ! occupation cannot get smaller than this (0.1 safety factor)
           AUGER_TINY_ehh=ELEC_OCC_THRESH_ehh*HOLE_OCC_THRESH_ehh*HOLE_OCC_THRESH_ehh*1E-1_q
           IF(LDUMP_AUGER)THEN
              WRITE(AUGER_IO%IU0,*) 'HOLE_OCC_THRESH_ehh',HOLE_OCC_THRESH_ehh
              WRITE(AUGER_IO%IU0,*) 'ELEC_OCC_THRESH_ehh',ELEC_OCC_THRESH_ehh
              WRITE(AUGER_IO%IU0,*) 'AUGER_TINY_ehh',AUGER_TINY_ehh
           END IF

           ! e conc. zero above EC1
           EC1_ehh=AUGER_SIGMA*LOG(1._q/ELEC_OCC_THRESH_ehh-1._q)+AUGER_MU
           ! h conc. zero below EV1
           EV1_ehh=-AUGER_SIGMA*LOG(1._q/HOLE_OCC_THRESH_ehh-1._q)+AUGER_MU
           ! maximum transition energy
           EMAX_ehh=EC1_ehh-EV1_ehh+AUGER_EWIDTH

           ! highest accesible VB energy
           EV2_ehh=EVBM-EMIN
           ! lowest accessible VB energy
           EV3_ehh=EV1_ehh-EMAX_ehh

           ! sanity check
           IF( (.NOT.(EV3_ehh<EV2_ehh .AND. EV2_ehh<EVBM .AND. EV1_ehh<EVBM)) &
                .OR. (.NOT.(EC1_ehh>ECBM)) )THEN
              IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'CALCULATE_AUGER: Error in energy bounds _ehh: ',EV3_ehh,EV2_ehh,EV1_ehh,EVBM,ECBM,EC1_ehh
              STOP
           END IF

           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'ehh: EV3,EV2,EV1,EVBM,ECBM,EC1',EV3_ehh,EV2_ehh,EV3_ehh,EVBM,ECBM,EC1_ehh
           EMIN1_ehh=EV1_ehh
           EMAX1_ehh=EVBM
           EMIN2_ehh=EMIN1_ehh
           EMAX2_ehh=EMAX1_ehh
           EMIN3_ehh=EV3_ehh
           EMAX3_ehh=EV2_ehh
           EMIN4_ehh=ECBM
           EMAX4_ehh=EC1_ehh

        END IF

      END SUBROUTINE ENERGY_RANGE


      !************************* SUBROUTINE BAND_RANGE************************
      ! Given min. and max. energies, find the band indices in this range
      ! for each k-point
      !***********************************************************************
      SUBROUTINE BAND_RANGE(W,WDES,&
           EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,&
           EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
           EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,&
           EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh,&
           NB1_MIN_eeh,NB1_MAX_eeh,NB2_MIN_eeh,NB2_MAX_eeh,&
           NB3_MIN_eeh,NB3_MAX_eeh,NB4_MIN_eeh,NB4_MAX_eeh,&
           NB1_MIN_ehh,NB1_MAX_ehh,NB2_MIN_ehh,NB2_MAX_ehh,&
           NB3_MIN_ehh,NB3_MAX_ehh,NB4_MIN_ehh,NB4_MAX_ehh)
        USE wave_high
        USE constant
        IMPLICIT NONE
        TYPE(wavespin) :: w
        TYPE(wavedes) :: WDES
        REAL(q) :: &
             EMIN1_eeh,EMAX1_eeh,EMIN2_eeh,EMAX2_eeh,&
             EMIN3_eeh,EMAX3_eeh,EMIN4_eeh,EMAX4_eeh,&
             EMIN1_ehh,EMAX1_ehh,EMIN2_ehh,EMAX2_ehh,&
             EMIN3_ehh,EMAX3_ehh,EMIN4_ehh,EMAX4_ehh
        INTEGER :: &
             NB1_MIN_eeh(WDES%NKPTS),NB1_MAX_eeh(WDES%NKPTS),&
             NB2_MIN_eeh(WDES%NKPTS),NB2_MAX_eeh(WDES%NKPTS),&
             NB3_MIN_eeh(WDES%NKPTS),NB3_MAX_eeh(WDES%NKPTS),&
             NB4_MIN_eeh(WDES%NKPTS),NB4_MAX_eeh(WDES%NKPTS),&
             NB1_MIN_ehh(WDES%NKPTS),NB1_MAX_ehh(WDES%NKPTS),&
             NB2_MIN_ehh(WDES%NKPTS),NB2_MAX_ehh(WDES%NKPTS),&
             NB3_MIN_ehh(WDES%NKPTS),NB3_MAX_ehh(WDES%NKPTS),&
             NB4_MIN_ehh(WDES%NKPTS),NB4_MAX_ehh(WDES%NKPTS)
        ! energy ranges for band search
        INTEGER NB1_START_eeh,NB1_STOP_eeh
        INTEGER NB2_START_eeh,NB2_STOP_eeh
        INTEGER NB3_START_eeh,NB3_STOP_eeh
        INTEGER NB4_START_eeh,NB4_STOP_eeh
        !
        INTEGER NB1_START_ehh,NB1_STOP_ehh
        INTEGER NB2_START_ehh,NB2_STOP_ehh
        INTEGER NB3_START_ehh,NB3_STOP_ehh
        INTEGER NB4_START_ehh,NB4_STOP_ehh
        REAL(q) EVBM,ECBM,EG,EMIN
        REAL(q) EV1_eeh,EV2_eeh,EV3_eeh,EC1_eeh,EC2_eeh,EC3_eeh,EMAX_eeh
        REAL(q) EV1_ehh,EV2_ehh,EV3_ehh,EC1_ehh,EC2_ehh,EC3_ehh,EMAX_ehh
        REAL(q) :: F_CBM,G_VBM,EDENS,HDENS
        INTEGER :: ISP,IK,IB

        ! eeh processes
        NB1_MIN_eeh=WDES%NB_TOT+1; NB2_MIN_eeh=WDES%NB_TOT+1; NB3_MIN_eeh=WDES%NB_TOT+1;NB4_MIN_eeh=WDES%NB_TOT+1
        NB1_MAX_eeh=0; NB2_MAX_eeh=0; NB3_MAX_eeh=0;NB4_MAX_eeh=0
        IF(LAUGER_EEH)THEN
           DO ISP=1,WDES%ISPIN; DO IK=1,WDES%NKPTS
              ! h in VB
              CALL IN_WINDOW(W,IK,ISP,EMIN2_eeh,EMAX2_eeh,NB2_START_eeh,NB2_STOP_eeh)
              NB2_MIN_eeh(IK)=NB2_START_eeh
              NB2_MAX_eeh(IK)=NB2_STOP_eeh
              ! e in CB
              CALL IN_WINDOW(W,IK,ISP,EMIN3_eeh,EMAX3_eeh,NB3_START_eeh,NB3_STOP_eeh)
              NB3_MIN_eeh(IK)=NB3_START_eeh
              NB3_MAX_eeh(IK)=NB3_STOP_eeh
              ! h in CB
              CALL IN_WINDOW(W,IK,ISP,EMIN1_eeh,EMAX1_eeh,NB1_START_eeh,NB1_STOP_eeh)
              NB1_MIN_eeh(IK)=NB1_START_eeh
              NB1_MAX_eeh(IK)=NB1_STOP_eeh
           ENDDO; ENDDO
           ! e in CB
           NB4_MIN_eeh=NB3_MIN_eeh
           NB4_MAX_eeh=NB3_MAX_eeh
        END IF

        ! ehh processes
        NB1_MIN_ehh=WDES%NB_TOT+1; NB2_MIN_ehh=WDES%NB_TOT+1; NB3_MIN_ehh=WDES%NB_TOT+1;NB4_MIN_ehh=WDES%NB_TOT+1
        NB1_MAX_ehh=0; NB2_MAX_ehh=0; NB3_MAX_ehh=0;NB4_MAX_ehh=0
        IF(LAUGER_EHH)THEN
           DO ISP=1,WDES%ISPIN; DO IK=1,WDES%NKPTS
              ! e in VB
              CALL IN_WINDOW(W,IK,ISP,EMIN3_ehh,EMAX3_ehh,NB3_START_ehh,NB3_STOP_ehh)
              NB3_MIN_ehh(IK)=NB3_START_ehh
              NB3_MAX_ehh(IK)=NB3_STOP_ehh
              ! h in VB
              CALL IN_WINDOW(W,IK,ISP,EMIN1_ehh,EMAX1_ehh,NB1_START_ehh,NB1_STOP_ehh)
              NB1_MIN_ehh(IK)=NB1_START_ehh
              NB1_MAX_ehh(IK)=NB1_STOP_ehh
              ! e in CB
              CALL IN_WINDOW(W,IK,ISP,EMIN4_ehh,EMAX4_ehh,NB4_START_ehh,NB4_STOP_ehh)
              NB4_MIN_ehh(IK)=NB4_START_ehh
              NB4_MAX_ehh(IK)=NB4_STOP_ehh
           ENDDO; ENDDO
           ! h in VB
           NB2_MIN_ehh=NB1_MIN_ehh
           NB2_MAX_ehh=NB1_MAX_ehh
        END IF

      END SUBROUTINE BAND_RANGE


      !************************* SUBROUTINE IN_WINDOW ************************
      ! find all bands with energies in [E0,E1] at k-point IK and spin ISP
      ! optional: NMIN,NMAX restrict range of bands to search
      ! returns:  N0,N1     min./max. band index
      !***********************************************************************
      SUBROUTINE IN_WINDOW(W,IK,ISP,E0,E1,N0,N1,NMIN,NMAX)
        USE wave
        TYPE (wavespin) W
        INTEGER IK,ISP
        REAL(q) E0,E1
        INTEGER N0,N1
        INTEGER, OPTIONAL :: NMIN,NMAX
        ! local variables
        INTEGER IB
        INTEGER BMIN,BMAX
        BMIN=1
        BMAX=W%WDES%NB_TOT
        IF(PRESENT(NMIN))BMIN=NMIN
        IF(PRESENT(NMAX))BMAX=NMAX
        N0=W%WDES%NB_TOT+1
        N1=-1
        DO IB=BMIN,BMAX
           IF (REAL(W%CELTOT(IB,IK,ISP),q)>=E0) N0=MIN(N0,IB)
           IF (REAL(W%CELTOT(IB,IK,ISP),q)<=E1) N1=MAX(N1,IB)
           if (REAL(W%CELTOT(IB,IK,ISP),q)>E1) EXIT
        ENDDO
        RETURN
      END SUBROUTINE IN_WINDOW



      !****************** SUBROUTINE AUGER_2E4O_EEH *******************************
      ! calculate
      !
      ! < 2 3 | W | 1 4 > 
      ! = int_d3r d3r' w3*_k3,n3(r') w4_k4,n4(r') w2*_k2,n2(r) w1_k1,n1(r)  W(r',r)
      !
      ! only those combinations of W2,W1 and W3,W4 are calculated that
      ! are specified in USE21 and USE34
      !****************************************************************************
      SUBROUTINE AUGER_2E4O(WHF, LATT_CUR, ISP, WGW, FSG, &
           W2, K2, W3, K3, &
           W1, K1, W4, K4, &
           USE21, USE34, &
           NFLOAT, NFFT, NSTRIP, M4O)

        USE wave_high
        USE constant
        USE full_kpoints
        USE kpoints_change
        USE pseudo
        USE lattice
        USE mytimer
        IMPLICIT NONE
        TYPE (wavespin) :: WHF
        TYPE (latt)     :: LATT_CUR
        INTEGER         :: ISP
        TYPE (wavedes)  :: WGW        ! descriptor for basis set of response function
        REAL(q)         :: FSG        ! singularity correction
        TYPE (wavefun1) :: W1(:), W2(:), W3(:), W4(:)
        INTEGER         :: K1, K2, K3, K4
        INTEGER         :: USE21(:),USE34(:)
        GDEF            :: M4O(:,:,:,:)
        INTEGER         :: NFLOAT, NFFT
        INTEGER         :: NSTRIP
        ! local variables
        INTEGER N1, N2, N3, N4
        INTEGER NP, NPP, NPOS3, NSTRIP3, NPOS4, NSTRIP4, NB1, NB2, NB3, NB4
        TYPE (wavedes1)    WGWQ
        INTEGER NQ, NQ_
        COMPLEX(q), ALLOCATABLE :: GCHG12(:,:)    ! charge 
        COMPLEX(q), ALLOCATABLE :: GCHG34(:,:)    ! charge 
        COMPLEX(q) :: GWORK( MAX(GRIDHF%MPLWV,WGW%GRID%MPLWV))
        COMPLEX(q) :: CPHASE(GRIDHF%MPLWV)
        GDEF, ALLOCATABLE :: CRHOLM(:)        ! augmentation occupancy matrix
        GDEF, ALLOCATABLE :: CHAM(:,:,:,:)    ! hamilton matrix
        LOGICAL LPHASE
        REAL(q) :: POTFAK(GRIDHF%MPLWV)       ! 1/(G+dk)**2 (G)
        LOGICAL :: LFULL
        INTEGER :: NSTRIP1, NSTRIP2,NB3_MAX,NB4_MAX
        INTEGER :: TIMER_2E4O,TIMER_PHASE1,TIMER_PHASE2

        TIMER_2E4O=GET_TIMER_INDEX('2E4O')
        TIMER_PHASE1=GET_TIMER_INDEX('PH1')
        TIMER_PHASE2=GET_TIMER_INDEX('PH2')

        CALL START_TIMER(TIMER_2E4O)

        ! set the descriptor for wavefunctions at the point NQ=K2-K1
        ! the charge  w1_k1 n1(r') w2*_k2 n2(r') = c_k1-k2
        ! is transformed corresponding to a wavevector q
        NQ=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2),KPOINTS_FULL)
        CALL SETWDES(WGW, WGWQ, NQ)

        NQ_=KPOINT_IN_FULL_GRID(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4),KPOINTS_FULL)
        IF (NQ/=NQ_) THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'internal error in AUGER_2E4O: q-point changed',NQ,NQ_
           STOP
        ENDIF

        ! calculation of matrix elements is done in strips, i.e.
        ! chunks of NSTRIP*NSTRIP bands from w3 and w4
        ! This makes sense for a large number of bands for memory efficiency
        ! For Auger calculations, this is mostly not useful
        NP=WGWQ%NGVECTOR
        NSTRIP1=SIZE(W1)
        NSTRIP2=SIZE(W2)
        NB3_MAX=SIZE(W3)
        NB4_MAX=SIZE(W4)
        ALLOCATE( &
             CRHOLM(AUG_DES%NPRO*WHF%WDES%NRSPINORS), &
             GCHG12(NP, NSTRIP1* NSTRIP2), &
             GCHG34(NP, NSTRIP * NSTRIP), &
             CHAM(NSTRIP1, NSTRIP2, NSTRIP, NSTRIP))

        CALL SET_GFAC_WAVEFUN(WGWQ, LATT_CUR, FSG, POTFAK )
        LFULL=.FALSE.
        IF (ASSOCIATED(AUGER_WPOTH)) CALL GET_WPOT( AUGER_WPOTH, NQ, POTFAK, LFULL)

        CALL START_TIMER(TIMER_PHASE1)
        CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2))
        CALL STOP_TIMER(TIMER_PHASE1)

        ! k1-k2-q might be any reciprocal lattice vector G
        ! in that case the result is shifted by G with respect to the 
        ! internal data layout (FFTEXT), we now apply a shift e^iGr in real space
        ! to cure the problem
        CALL START_TIMER(TIMER_PHASE2)
        CALL SETPHASE(WHF%WDES%VKPT(:,K1)-WHF%WDES%VKPT(:,K2)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
        CALL STOP_TIMER(TIMER_PHASE2)

        !  multiply v'*_k2,n2(r) v_k1,n1(r) = chg12(r) => chg12(G)
        GCHG12=0

        DO N1=1,NSTRIP1
           DO N2=1,NSTRIP2

              IF(USE21( (N2-1)*SIZE(W1)+N1 )==0) CYCLE

              CALL FOCK_CHARGE_NOINT( W1(N1), W2(N2), GWORK(1), CRHOLM, SIZE(CRHOLM))
              ! apply phase factor to shift by reciprocal lattice vector
              IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )
              ! now we have w2*_k2,n2(r'), w1_k1,n1(r')

              ! extract using wave function FFT
              CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                   GWORK(1),GCHG12(1,N1+(N2-1)*NSTRIP1),WGWQ%GRID,.FALSE.)
              NFFT=NFFT+1

              IF (.NOT. LFULL) THEN
                 ! multiply with potential factor
                 CALL APPLY_GFAC_WAVEFUN(  WGWQ, GCHG12(1,N1+(N2-1)*NSTRIP1), POTFAK(1))
                 GCHG12(1:NP,N1+(N2-1)*NSTRIP1)=-CONJG(GCHG12(1:NP,N1+(N2-1)*NSTRIP1))
              ENDIF
           ENDDO
        ENDDO

        IF (LFULL) THEN
           DO N1=1, NSTRIP1*NSTRIP2, NSTRIP*NSTRIP
              N2=MIN(NSTRIP1*NSTRIP2-N1+1,NSTRIP*NSTRIP)
              IF (AUGER_WPOTH%WPOT(NQ)%LREALSTORE) THEN
                 CALL DGEMM('N','N', 2*NP, N2, 2*NP, 1.0_q, &
                      AUGER_WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), 2*SIZE(AUGER_WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG12(1,N1), 2*SIZE(GCHG12,1), &
                      0.0_q, GCHG34(1,1), 2*SIZE(GCHG34,1) )
              ELSE
                 CALL ZGEMM('T','N', NP, N2, NP, (1.0_q,0.0_q), &
                      AUGER_WPOTH%WPOT(NQ)%RESPONSEFUN(1,1,1), SIZE(AUGER_WPOTH%WPOT(NQ)%RESPONSEFUN,1), GCHG12(1,N1), SIZE(GCHG12,1), &
                      (0.0_q,0.0_q), GCHG34(1,1), SIZE(GCHG34,1) )
              ENDIF
              CALL ZCOPY( SIZE(GCHG12,1)*N2,  GCHG34(1,1), 1,  GCHG12(1,N1), 1)
           ENDDO
           GCHG12=-CONJG(GCHG12)
        ENDIF

        ! set the phase factors q'=k3-k4
        CALL START_TIMER(TIMER_PHASE1)
        CALL PHASER_HF(GRIDHF, LATT_CUR, FAST_AUG_FOCK, WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4))
        CALL STOP_TIMER(TIMER_PHASE1)
        CALL START_TIMER(TIMER_PHASE2)
        CALL SETPHASE(WHF%WDES%VKPT(:,K3)-WHF%WDES%VKPT(:,K4)-WHF%WDES%VKPT(:,NQ), GRIDHF,CPHASE,LPHASE)
        CALL STOP_TIMER(TIMER_PHASE2)

        np3: DO NPOS3=1,NB3_MAX,NSTRIP
           NSTRIP3=MIN(NB3_MAX+1-NPOS3,NSTRIP)
           np4: DO NPOS4=1,NB4_MAX,NSTRIP
              NSTRIP4=MIN(NB4_MAX+1-NPOS4,NSTRIP)

              ! multiply w3*_n3,k3(r') w4_n4,k4(r') = chg34(r') => chg34(G')
              GCHG34=0

              DO N4=1,NSTRIP4
                 NB4=NPOS4-1+N4
                 DO N3=1,NSTRIP3
                    NB3=NPOS3-1+N3

                    IF(USE34( (NB3-1)*SIZE(W4)+NB4 )==0) CYCLE
                    
                    CALL FOCK_CHARGE_NOINT( W3(NB3), W4(NB4), GWORK(1), CRHOLM, SIZE(CRHOLM))
                    ! apply phase factor e^iGr if required
                    IF (LPHASE) CALL APPLY_PHASE( GRIDHF, CPHASE(1), GWORK(1), GWORK(1) )

                    ! now we have w3_n3,k3(r')  w4*_n4,k4(r')
                    ! extract using FFT
                    CALL FFTEXT(WGWQ%NGVECTOR, WGWQ%NINDPW(1), &
                         GWORK(1),GCHG34(1,N3+(N4-1)*NSTRIP),WGWQ%GRID,.FALSE.)
                    NFFT=NFFT+1

                    ! conjugate to get: w3*_n3,k3(r')  w4_n4,k4(r')
                    GCHG34(1:NP,N3+(N4-1)*NSTRIP)=CONJG(GCHG34(1:NP,N3+(N4-1)*NSTRIP))*(1.0_q/GRIDHF%NPLWV)
                 ENDDO
              ENDDO

              ! sum_G' sum_G w3*_n3,k3(G')  w4_n4,k4(G') w(G',G) w2*_k2,n2(G), w1_k1,n1(G) = M_n2,n3,n1,n4
              CHAM=0
              CALL GGEMM(trans,'N',  NSTRIP1*NSTRIP2 , SIZE(CHAM,3)*NSTRIP4, m_ NP, one, &
                   GCHG12(1,1), m_ SIZE(GCHG12,1), &
                   GCHG34(1,1), m_ SIZE(GCHG34,1), &
                   zero, CHAM(1,1,1,1), SIZE(CHAM,1)*SIZE(CHAM,2))
              NFLOAT=NFLOAT+NSTRIP1*NSTRIP2*SIZE(CHAM,3)*NSTRIP4*NP*8

              ! store in M4O
              DO N1=1,NSTRIP1
                 DO N2=1,NSTRIP2
                    DO N4=1,NSTRIP4
                       NB4=NPOS4-1+N4
                       IF (NB4 > SIZE(M4O,4)) THEN
                          IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'internal error in AUGER_2E4O: out of bounds 4 ',NB4
                          STOP
                       ENDIF

                       DO N3=1,NSTRIP3
                          NB3=NPOS3-1+N3
                          IF ( NB3> SIZE(M4O,2)) THEN
                             IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*)'internal error in AUGER_2E4O: out of bounds 2 ',NB3
                             STOP
                          ENDIF

                          M4O(N2,NB3,N1,NB4)=M4O(N2,NB3,N1,NB4)+CHAM( N1, N2, N3, N4)

                       ENDDO
                    ENDDO
                 ENDDO
              ENDDO

           ENDDO np4
        ENDDO np3

        DEALLOCATE(CRHOLM, GCHG12, GCHG34, CHAM)

        CALL STOP_TIMER(TIMER_2E4O)
      END SUBROUTINE AUGER_2E4O


      !************************* SUBROUTINE FERMI ************************
      ! Fermi distribution function
      !***********************************************************************
      FUNCTION FERMI(E,EFERMI,SIGMA)
        USE prec
        IMPLICIT NONE
        REAL(q) :: FERMI,E,EFERMI,SIGMA
        FERMI=1._q/(1._q+EXP((E-EFERMI)/SIGMA))
      END FUNCTION FERMI


      !************************* SUBROUTINE ANTIFERMI ************************
      ! 1-FERMI for FERMI near 1
      ! occupation probability for holes
      !***********************************************************************
      FUNCTION ANTIFERMI(E,EFERMI,SIGMA)
        USE prec
        IMPLICIT NONE
        REAL(q) :: ANTIFERMI,E,EFERMI,SIGMA
        ANTIFERMI=1._q/(1._q+EXP(-(E-EFERMI)/SIGMA))
      END FUNCTION ANTIFERMI


      !************************* SUBROUTINE DELTA_EN *************************
      ! rectangular function with width 2W and height 1/(2W) centered on 0
      !***********************************************************************
      FUNCTION DELTA_EN(E,W)
        USE prec
        IMPLICIT NONE
        REAL(q) :: DELTA_EN,E
        REAL(q) :: W
        IF(ABS(E)<W)THEN
           DELTA_EN=0.5_q/W
        ELSE
           DELTA_EN=0._q
        END IF
      END FUNCTION DELTA_EN


      !************************* SUBROUTINE EH_DENS *************************
      ! calculate density of electron and holes for given fermi distribution
      ! parameters
      ! result: EDENS and/or HDENS in [Angstrom^-3]
      ! external parameter: AUGER_OCCUPY_THRESHOLD
      !***********************************************************************
      SUBROUTINE EH_DENS(EFERMI,SIGMA,W,WDES,LAT,EDENS,HDENS)
        USE lattice
        IMPLICIT NONE
        REAL(q) EFERMI,SIGMA
        TYPE(wavespin) :: W
        TYPE(wavedes) :: WDES
        TYPE(latt) :: LAT
        REAL(q), OPTIONAL :: EDENS,HDENS
        !
        INTEGER :: ISP,IK,IB
        
        ! calculate electron and hole concentration
        IF(PRESENT(EDENS)) EDENS=0
        IF(PRESENT(HDENS)) HDENS=0
        IF(PRESENT(HDENS))THEN
           DO ISP=1,WDES%ISPIN; DO IK=1,WDES%NKPTS; DO IB=1,WDES%NB_TOT
              IF(AUGER_EVBHI>-1E30_q)THEN
                 IF(REAL(W%CELTOT(IB,IK,ISP),q)<AUGER_EVBHI) &
                      HDENS=HDENS+ANTIFERMI(REAL(W%CELTOT(IB,IK,ISP),q),EFERMI,SIGMA)*WDES%WTKPT(IK)
              ELSEIF(W%FERTOT(IB,IK,ISP)>AUGER_OCCUPY_THRESHOLD) THEN
                 ! valence band - count holes
                 HDENS=HDENS+ANTIFERMI(REAL(W%CELTOT(IB,IK,ISP),q),EFERMI,SIGMA)*WDES%WTKPT(IK)
              END IF
           ENDDO; ENDDO; ENDDO
           HDENS=HDENS*WDES%RSPIN/LAT%OMEGA
        ENDIF
        IF(PRESENT(EDENS))THEN
           DO ISP=1,WDES%ISPIN; DO IK=1,WDES%NKPTS; DO IB=1,WDES%NB_TOT
              IF(AUGER_ECBLO>-1E30_q)THEN
                 IF(REAL(W%CELTOT(IB,IK,ISP),q)>AUGER_ECBLO) &
                      EDENS=EDENS+FERMI(REAL(W%CELTOT(IB,IK,ISP),q),EFERMI,SIGMA)*WDES%WTKPT(IK)
              ELSEIF(W%FERTOT(IB,IK,ISP)<=AUGER_OCCUPY_THRESHOLD) THEN
                 ! conduction band - count electrons
                 EDENS=EDENS+FERMI(REAL(W%CELTOT(IB,IK,ISP),q),EFERMI,SIGMA)*WDES%WTKPT(IK)
              END IF
           ENDDO; ENDDO; ENDDO
           EDENS=EDENS*WDES%RSPIN/LAT%OMEGA
        ENDIF
        
      END SUBROUTINE EH_DENS


      !************************* SUBROUTINE DETERMINE_EFERMI *****************
      ! determine the chemical potential (sloppy Fermi energy) to achieve a
      ! given density of electrons (TYPE 'E') or holes (TYPE 'H')
      ! parameters
      ! result: EDENS/HDENS in [Angstrom^-3]
      ! external parameter: AUGER_OCCUPY_THRESHOLD
      !***********************************************************************
      SUBROUTINE DETERMINE_EFERMI(DENS,TYPE,EMIN,EMAX,SIGMA,W,WDES,LAT,EFERMI)
        USE lattice
        USE wave
        IMPLICIT NONE
        REAL(q) :: DENS,SIGMA,EFERMI
        CHARACTER(1) :: TYPE
        TYPE(wavespin) :: W
        TYPE(wavedes) :: WDES
        TYPE(latt) :: LAT
        !
        REAL(q) :: EMIN,EMAX,EF1,EF2,D1,D2,D
        REAL(q), PARAMETER :: TINY=1E-8
        INTEGER :: J
        
        EF1=EMIN
        EF2=EMAX
        IF(TYPE=='E')THEN
           CALL EH_DENS(EF1,SIGMA,W,WDES,LAT,EDENS=D1)
           CALL EH_DENS(EF2,SIGMA,W,WDES,LAT,EDENS=D2)
        ELSE
           CALL EH_DENS(EF1,SIGMA,W,WDES,LAT,HDENS=D1)
           CALL EH_DENS(EF2,SIGMA,W,WDES,LAT,HDENS=D2)
        END IF

        IF( (D1-DENS)*(D2-DENS)>0 ) THEN
           IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'DETERMINE_EFERMI: bisectioning not possible',EMIN,EMAX,D1,D2
           STOP
        END IF

        DO J=1,50
           IF(ABS(EF1-EF2)/SIGMA<10_q)THEN
              ! secant method
              EFERMI=(LOG(DENS)-LOG(D1))*(EF2-EF1)/(LOG(D2)-LOG(D1)) + EF1 
           ELSE
              ! bisectioning
              EFERMI=(EF1+EF2)/2
           END IF
           IF(TYPE=='E')THEN
              CALL EH_DENS(EFERMI,SIGMA,W,WDES,LAT,EDENS=D)
           ELSE
              CALL EH_DENS(EFERMI,SIGMA,W,WDES,LAT,HDENS=D)
           END IF
           
           IF(ABS(D-DENS)/DENS<TINY) GOTO 101
           IF((D-DENS)*(D1-DENS)>0)THEN
              ! shift EF1
              EF1=EFERMI
              D1=D
           ELSE
              EF2=EFERMI
              D2=D
           END IF
        END DO
        IF(LDUMP_AUGER) WRITE(AUGER_IO%IU0,*) 'DETERMINE_EFERMI: search for Fermi energy failed to converge'
        STOP
101     RETURN
        
      END SUBROUTINE DETERMINE_EFERMI


      !************************* FUNCTION EXIST_TRANSITION *******************
      !
      !***********************************************************************
      FUNCTION EXIST_TRANSITION(E,W,IK0,N0min,N0max,ISP0,IK1,N1min,N1max,ISP1)
        USE wave
        TYPE (wavespin) W
        REAL(q) E
        INTEGER IK0,IK1,ISP0,ISP1
        INTEGER N0min,N0max,N1min,N1max
        LOGICAL EXIST_TRANSITION
        ! local variables
        INTEGER I,J

        EXIST_TRANSITION=.FALSE.
        n0: DO I=N0min,N0max
           DO J=N1min,N1max
              IF (ABS(E-REAL(W%CELTOT(J,IK1,ISP1)-W%CELTOT(I,IK0,ISP0),q))<=AUGER_EWIDTH) THEN
                 EXIST_TRANSITION=.TRUE.; EXIT n0
              ENDIF
           ENDDO
        ENDDO n0

      END FUNCTION EXIST_TRANSITION

      END MODULE auger
