#include "symbol.inc"
!***********************************************************************
! RCS:  $Id: fileio.F,v 1.12 2003/06/27 13:22:18 kresse Exp kresse $
!
!  collection of subroutines to performe File IO
!  not yet realy supported heavily
!  but we want to switch all IO form the main program to this modul
!
!***********************************************************************
      MODULE fileio
      USE prec

!
! unit for the density matrix GAMMA (file GAMMA)
!
      INTEGER :: IU_GAMMA=77

!*************************SUBROUTINE OUTPOT ****************************
!   write potential     to a specified unit
!   HEADER is currently created in the main program
!***********************************************************************

      INTERFACE
      SUBROUTINE OUTPOT(GRIDC, IU,LLONG,CVTOT)
      USE prec
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRIDC
      COMPLEX(q) CVTOT(GRIDC%RC%NP)
      LOGICAL LLONG
      END SUBROUTINE
      END INTERFACE

      CONTAINS

!*************************SUBROUTINE OPENWAV ***************************
!
!   open WAVECAR file
!   optionally an three character extension can be supplied allowing the
!   open an alternative WAVECAR file
!
! - the unit 12 is always connected with the WAVECAR file
! - for the mpi chain version (repeated images) the files
!   XXX/WAVECAR are opended
! - the record length is read from the WAVECAR file
!
!***********************************************************************

      SUBROUTINE OPENWAV(IO, COM, EXT)

      USE base
      USE main_mpi
      USE mpimy
      IMPLICIT NONE

      TYPE (in_struct)   IO
      TYPE (communic)    COM
      LOGICAL  :: junk
      REAL(q)  :: RDUM, RISPIN
      INTEGER  :: IDUM
      CHARACTER (LEN=3), OPTIONAL :: EXT


      junk=.TRUE.
      IF (PRESENT(EXT)) THEN
         INQUIRE(FILE=DIR_APP(1:DIR_LEN)//'WAVECAR.'//EXT,EXIST=junk)
      ELSE
         INQUIRE(FILE=DIR_APP(1:DIR_LEN)//'WAVECAR',EXIST=junk)
      ENDIF
      CALLMPI( M_bcast_l(COM, junk, 1 ))
      IO%LFOUND=junk

! first reopen with assumed (wrong) record length ICMPLX
      IF (PRESENT(EXT)) THEN
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR.'//EXT,ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%ICMPLX)
      ELSE
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR',ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%ICMPLX)
      ENDIF
! the first record contains the record length, get it ...
      RDUM=0._q
      READ(12,REC=1,ERR=17421) RDUM,RISPIN ; IDUM=NINT(RDUM)
! in the worst case IDUM could contain completely useless data and useless is
! all <=0 or all >10000000 (since on output we use RECL=IO%ICMPLX*MAX(NRPLWV,6)
! or RECL=(NB_TOT+2)*ICMPLX more than ten millions sounds not very realistic)
      IF (IDUM<=0) IDUM=IO%ICMPLX  ! -> error reading WAVECAR
      GOTO 17422

17421 CONTINUE
      IDUM=IO%ICMPLX  ! -> error reading WAVECAR
17422 CONTINUE
      CLOSE(12)
      IO%IRECLW=IDUM
! reopen with correct record length (clumsy all that, I know ...)
      IF (PRESENT(EXT)) THEN
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR.'//EXT,ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ELSE
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR',ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ENDIF

      END SUBROUTINE OPENWAV

!*************************SUBROUTINE CLOSEWAV **************************
!
!   close unit 12 (usually connected to the  WAVECAR file)
!
!***********************************************************************

      SUBROUTINE CLOSEWAV
        CLOSE(12)
      END SUBROUTINE CLOSEWAV

!*************************SUBROUTINE INWAV  ****************************
!
!   read wavefunctions header from file
!   just parse the old cell shape old cutoff etc.
!   this is a new version that uses only real quantities and allows
!   the WAVECAR file to be exchanged between different machines
!   more easily (IEEE standard)
!
!***********************************************************************

      SUBROUTINE INWAV_HEAD(WDES, LATT_INI, LATT_CUR, ENMAXI, ISTART, IU0)
      USE prec
      USE wave
      USE lattice
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (wavedes)  WDES
      TYPE (latt)     LATT_INI,LATT_CUR
      LOGICAL     LDIFF
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif

      READ(12,REC=2,ERR=100) RKPTSF,RBANDF,ENMAXI, &
     &                       ((LATT_INI%A(I,J),I=1,3),J=1,3)

      IF (IU0 >= 0) WRITE(IU0,*) 'found WAVECAR, reading the header'

      NKPTSF=NINT(RKPTSF)
      NBANDF=NINT(RBANDF)

      CALL LATTIC(LATT_INI)

      IF (ISTART==2 .AND. ENMAXI /= WDES%ENMAX) THEN
        IF (IU0>=0) WRITE(IU0,*) 'ERROR: ENMAX changed please set ISTART to 1'
        STOP
      ENDIF

      IF (.NOT. (NBANDF==WDES%NB_TOT .OR. (WDES%NRSPINORS==2 .AND.  NBANDF*WDES%NRSPINORS==WDES%NB_TOT))) THEN
         IF (IU0 >= 0) WRITE(IU0,'(2X,A,I6,A,I6)') &
              'number of bands has changed, file:',NBANDF,' present:',WDES%NB_TOT
         IF (IU0 >= 0) WRITE(IU0,'(2X,A,I6,A,I6)') &
              'trying to continue reading WAVECAR, but it might fail'
      ENDIF
      IF (NKPTSF/=WDES%NKPTS) THEN
         IF (IU0 >= 0) WRITE(IU0,'(2X,A,I6,A,I6)') &
              'number of k-points has changed, file:',NKPTSF,' present:',WDES%NKPTS
         IF (IU0 >= 0) WRITE(IU0,'(2X,A,I6,A,I6)') &
              'trying to continue reading WAVECAR, but it might fail'
      ENDIF
      IF (ISTART ==1) THEN

      LDIFF=.FALSE.
      DO I=1,3
        DO J=1,3
          IF (ABS(LATT_INI%A(I,J)-LATT_CUR%A(I,J)) > 1E-4) LDIFF=.TRUE.
        ENDDO
      ENDDO
      IF (ENMAXI /= WDES%ENMAX) LDIFF=.TRUE.
      IF (LDIFF) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)'WAVECAR: different cutoff or change in lattice found'
      ENDIF

      ENDIF

      RETURN
!
! WAVECAR does not exist or can not be read at all 
!
  100 CONTINUE
      LATT_INI%A=LATT_CUR%A
      CALL LATTIC(LATT_INI)

      IF (ISTART==3) THEN
        IF (IU0>=0) &
        WRITE(IU0,*)"ERROR: can't restart (ISTART=3) with wavefunctions on file"
        STOP
      ELSE
        ISTART=0
      ENDIF
      RETURN

      END SUBROUTINE


!*************************SUBROUTINE INWAV_FAST ************************
!
!   read wavefunctions from file
!
!   the routine can read WAVECAR files generated for different
!   lattice vectors and cutoff (both in the serial and parallel case)
!
!   it is also possible to restart a spin polarised calculation
!   from a non spin polarised wavefunction file or
!   to restart a non collinear calculation from a collinear calculation
!
!***********************************************************************

      SUBROUTINE INWAV_FAST(IO, WDES, W, GRID, LATT_CUR, LATT_INI, ISTART, EFERMI)
      USE prec
      USE base
      USE wave_high
      USE mgrid
      USE lattice

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (in_struct)   IO
      TYPE (wavedes)  WDES
      TYPE (wavedes1) WDES1
      TYPE (wavespin) W
      TYPE (grid_3d)  GRID
      TYPE (latt)     LATT_CUR,LATT_INI
      REAL (q) :: EFERMI

! local work arrays
      REAL(q) VKPT(3)
      INTEGER,ALLOCATABLE    :: IND(:),INDI(:)
      COMPLEX(q),ALLOCATABLE :: CW1(:),CW2(:)
      LOGICAL ::  SINGLE_PREC, SPAN_RECORDS
      COMPLEX(qs),  ALLOCATABLE :: CRD(:)
      LOGICAL, POINTER :: LCONJG(:)=>null()
      INTEGER, POINTER :: INDEX(:)=>null()
      REAL(q), ALLOCATABLE :: INBUF(:)

      IU0=IO%IU0

      NALLOC = MAXVAL(WDES%NPLWKP_TOT)

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif
!
! parse the header
!
        RTAG=0
        READ(12,REC=1,ERR=200) RDUM,RISPIN,RTAG
        IF (RTAG==53300) THEN
           IF (IU0>=0) WRITE(IU0,*) "VASP.5.3 WAVECAR encountered"
           SINGLE_PREC=.TRUE.
           SPAN_RECORDS=.TRUE.
        ELSE IF (RTAG==53310) THEN
           IF (IU0>=0) WRITE(IU0,*) "VASP.5.3 double precision WAVECAR encountered"
           SINGLE_PREC=.FALSE.
           SPAN_RECORDS=.TRUE.
        ELSE IF (RTAG==45200) THEN
           SINGLE_PREC=.TRUE.
           SPAN_RECORDS=.FALSE.
        ELSE
           IF (IU0>=0) WRITE(IU0,*) "double precision WAVECAR encountered, converting it"
           SINGLE_PREC=.FALSE.
           SPAN_RECORDS=.FALSE.
        ENDIF

        ISPINREAD=NINT(RISPIN)
        READ(12,REC=2,ERR=200) RKPTSF,RBANDF,ENMAXF, &
     &                         ((LATT_INI%A(I,J),I=1,3),J=1,3),EFERMI

        IREC=2
        NKPTSF=NINT(RKPTSF)
        NBANDF=NINT(RBANDF)

        CALL LATTIC(LATT_INI)
!=======================================================================
! read WAVECAR file, number of bands agree
!=======================================================================
      IF (.NOT.(WDES%NRSPINORS==2 .AND. NBANDF*WDES%NRSPINORS == WDES%NB_TOT)) THEN

        spin:    DO ISP=1, MIN(WDES%ISPIN, ISPINREAD)
        kpoints: DO K=1,WDES%NKPTS

        IF ( K> NKPTSF ) THEN
           IREC=IRECL
        ELSE
           IRECL=IREC
        ENDIF

        CALL SETWDES(WDES,WDES1,K)
#ifdef gammareal
        CALL GEN_INDEX_GAMMA(WDES1, LATT_CUR%B, WDES%ENMAX, INDEX, LCONJG)
#endif

        io_begin
        IFAIL=0
        IF (.NOT. SPAN_RECORDS) THEN
           IREC=IREC+1
           READ(12,REC=IREC,ERR=230) RNPL,VKPT, &
                (W%CELTOT(J,K,ISP),W%FERTOT(J,K,ISP),J=1,MIN(WDES%NB_TOT,NBANDF))
        ELSE
           ALLOCATE(INBUF(4+NBANDF*3))
           CALL READ_TO_BUF(12, IREC, IO%IRECLW/IO%ICMPLX*2, INBUF, IFAIL)
           RNPL   =INBUF(1)
           VKPT(1)=INBUF(2)
           VKPT(2)=INBUF(3)
           VKPT(3)=INBUF(4)
           DO J=1,MIN(WDES%NB_TOT,NBANDF)
              W%CELTOT(J,K,ISP)=CMPLX(INBUF(5+(J-1)*3),INBUF(6+(J-1)*3),q)
              W%FERTOT(J,K,ISP)=INBUF(7+(J-1)*3)
           ENDDO
           DEALLOCATE(INBUF)

        ENDIF
        NPLREAD=NINT(RNPL)
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 230

        CALLMPI( M_bcast_i( WDES%COMM, IREC, 1))
        CALLMPI( M_bcast_i( WDES%COMM, NPLREAD, 1))
        NPL=WDES%NPLWKP_TOT(K)

        MALLOC=MAX(NPL, NPLREAD)
        ALLOCATE(CW1(MALLOC),CW2(MALLOC),CRD(MALLOC),IND(MALLOC),INDI(MALLOC))

        io_begin
        ! create an index to allow for change of cutoff or cell size
        IF (ISTART==2) THEN
           ! gK: fix ISTART=2 in INWAV_FAST
           CALL REPAD_INDEX_ARRAY(GRID, WDES%VKPT(:,K), VKPT, LATT_INI%B,  LATT_INI%B, &
              WDES%ENMAX, ENMAXF, NPL/WDES%NRSPINORS, NPLREAD/WDES%NRSPINORS, IND, INDI, MALLOC, IFAIL )
        ELSE
           CALL REPAD_INDEX_ARRAY(GRID, WDES%VKPT(:,K), VKPT, LATT_CUR%B,  LATT_INI%B, &
              WDES%ENMAX, ENMAXF, NPL/WDES%NRSPINORS, NPLREAD/WDES%NRSPINORS, IND, INDI, MALLOC, IFAIL )
        ENDIF
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 220

        band: DO J=1,NBANDF
          IREC=IREC+1
          IF (J>WDES%NB_TOT) CYCLE
          io_begin
            IF (SINGLE_PREC) THEN
               READ(12,REC=IREC,ERR=240) (CRD(I),I=1,NPLREAD)
               CW2(1:NPLREAD)=CRD(1:NPLREAD)
            ELSE
               READ(12,REC=IREC,ERR=240) (CW2(I),I=1,NPLREAD)
            ENDIF

            CW1=0
            DO IS=1,WDES%NRSPINORS
               ! store the wave function coefficients according to new cutoff
               CALL REPAD_WITH_INDEX_ARRAY( MALLOC, IND, INDI, &
                  CW1((IS-1)*NPL/WDES%NRSPINORS+1), CW2((IS-1)*NPLREAD/WDES%NRSPINORS+1))
            ENDDO

          io_end
#ifdef MPI
          IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
             CALLMPI( M_bcast_z(WDES%COMM_KINTER,CW1,SIZE(CW1)) )
          END IF
#endif
#ifdef gammareal
          ! gamma only go from seriel layout to parallel layout if required
          CALL DIS_PW_BAND_GAMMA(WDES1, J, CW1, W%CW(1,1,K,ISP), INDEX(1), LCONJG(1))
#else
          CALL DIS_PW_BAND(WDES1, J, CW1, W%CW(1,1,K,ISP))
#endif
        ENDDO band

#ifdef gammareal
        CALL FREE_INDEX_GAMMA(INDEX, LCONJG)
#endif
        DEALLOCATE(CW1,CW2,CRD,IND,INDI)

        ENDDO kpoints

        IF (NKPTSF > WDES%NKPTS) THEN
           IREC=IREC+(NKPTSF-WDES%NKPTS)*(WDES%NB_TOT+1)
        ENDIF

        ! copy eigenvalues and weights to all nodes
        NCOMM=WDES%NB_TOT*WDES%NKPTS
        CALLMPI( M_bcast_z(WDES%COMM, W%CELTOT(1,1,ISP),NCOMM ))
        CALLMPI( M_bcast_d(WDES%COMM, W%FERTOT(1,1,ISP),NCOMM ))

        ENDDO spin

        IF (ISPINREAD > WDES%ISPIN .AND. IU0>=0 ) THEN
           WRITE(IU0,*) 'down-spin wavefunctions not read'
        ENDIF

        IF (NBANDF<WDES%NB_TOT) THEN
           IF (IU0>=0) WRITE(IU0,*) 'random initialization beyond band ',NBANDF+1
           CALL WFINIT(WDES, W, 1E10_q, NBANDF+1) ! ENINI=1E10 not cutoff restriction
        ENDIF

        IF (IU0 >= 0) WRITE(IU0,*) 'the WAVECAR file was read successfully'

        IF (WDES%ISPIN<=ISPINREAD) THEN
           RETURN
        ENDIF
!
!  spin down is missing
!
        IF (IU0>=0) &
        WRITE(IU0,*) 'No down-spin wavefunctions found', &
     &             ' --> setting down-spin equal up-spin ...'

        DO K=1,WDES%NKPTS
          W%CELTOT(1:WDES%NB_TOT,K,2)=W%CELTOT(1:WDES%NB_TOT,K,1)
          W%FERTOT(1:WDES%NB_TOT,K,2)=W%FERTOT(1:WDES%NB_TOT,K,1)
          NPL=WDES%NPLWKP(K)
          W%CW(1:NPL,1:WDES%NBANDS,K,2)=W%CW(1:NPL,1:WDES%NBANDS,K,1)
        ENDDO


        RETURN
!=======================================================================
! read collinear WAVECAR file for a non collinear run
!=======================================================================
      ELSE
        W%CW=0

        spin2:    DO ISP=1,ISPINREAD
        IF (IU0>=0.AND. ISP==1) &
        WRITE(IU0,*) 'reading wavefunctions of collinear run, up'
        IF (IU0>=0.AND. ISP==2) &
        WRITE(IU0,*) 'reading wavefunctions of collinear run, down'
        kpoints2: DO K=1,NKPTSF

        IF ( K> NKPTSF ) THEN
           IREC=IRECL
        ELSE
           IRECL=IREC
        ENDIF

        CALL SETWDES(WDES,WDES1,K)

        io_begin
        IFAIL=0
        IF (.NOT. SPAN_RECORDS) THEN
           IREC=IREC+1
           READ(12,REC=IREC,ERR=230) RNPL, VKPT(1:3), &
                       (W%CELTOT(J+NBANDF*(ISP-1),K,1),W%FERTOT(J+NBANDF*(ISP-1),K,1),J=1,NBANDF)
        ELSE
           ALLOCATE(INBUF(4+NBANDF*3))

           CALL READ_TO_BUF(12, IREC, IO%IRECLW/IO%ICMPLX*2, INBUF, IFAIL)
           RNPL   =INBUF(1)
           VKPT(1)=INBUF(2)
           VKPT(2)=INBUF(3)
           VKPT(3)=INBUF(4)
           DO J=1,NBANDF
              W%CELTOT(J+NBANDF*(ISP-1),K,1)=CMPLX(INBUF(5+(J-1)*3),INBUF(6+(J-1)*3),q)
              W%FERTOT(J+NBANDF*(ISP-1),K,1)=INBUF(7+(J-1)*3)
           ENDDO
           DEALLOCATE(INBUF)

        ENDIF
        NPLREAD=NINT(RNPL)
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 230

        CALLMPI( M_bcast_i( WDES%COMM, IREC, 1))
        CALLMPI( M_bcast_i( WDES%COMM, NPLREAD, 1))
        NPL=WDES%NPLWKP_TOT(K)/2

        MALLOC=MAX(NPL, NPLREAD)
        ALLOCATE(CW1(2*MALLOC),CW2(2*MALLOC),CRD(2*MALLOC),IND(MALLOC),INDI(MALLOC))

        io_begin
        IF (ISTART==2) THEN
           ! gK: fix ISTART=2 in INWAV_FAST
           CALL REPAD_INDEX_ARRAY(GRID, WDES%VKPT(:,K), VKPT, LATT_INI%B,  LATT_INI%B, &
                WDES%ENMAX, ENMAXF, NPL, NPLREAD, IND, INDI, MALLOC, IFAIL )
        ELSE
           CALL REPAD_INDEX_ARRAY(GRID, WDES%VKPT(:,K), VKPT, LATT_CUR%B,  LATT_INI%B, &
                WDES%ENMAX, ENMAXF, NPL, NPLREAD, IND, INDI, MALLOC, IFAIL )
        ENDIF
        io_end

        CALLMPI( M_bcast_i( WDES%COMM, IFAIL, 1))
        IF (IFAIL /=0) GOTO 220

        IF (ISP==1) THEN
           DO J=1,NBANDF
              IREC=IREC+1
              io_begin
                IF (SINGLE_PREC) THEN
                   READ(12,REC=IREC,ERR=240) (CRD(I),I=1,NPLREAD)
                   CW2(1:NPLREAD)=CRD(1:NPLREAD)
                ELSE
                   READ(12,REC=IREC,ERR=240) (CW2(I),I=1,NPLREAD)
                ENDIF
                   
                      
                CW1=0
                CALL REPAD_WITH_INDEX_ARRAY( MALLOC, IND, INDI, CW1, CW2)
              io_end

#ifdef MPI
              IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
                 CALLMPI( M_bcast_z(WDES%COMM_KINTER,CW1,SIZE(CW1)) )
              END IF
#endif
              CALL DIS_PW_BAND(WDES1, J, CW1, W%CW(1,1,K,1))

              ! copy immediately to second panel
              W%CELTOT(J+NBANDF,K,1)=W%CELTOT(J,K,1)
              W%FERTOT(J+NBANDF,K,1)=W%FERTOT(J,K,1)

              CW1(NPL+1:2*NPL)=CW1(1:NPL)
              CW1(1:NPL)=0

              CALL DIS_PW_BAND(WDES1, J+NBANDF, CW1, W%CW(1,1,K,1))
           ENDDO
        ELSE
           DO J=1,NBANDF
              IREC=IREC+1
              io_begin
                IF (SINGLE_PREC) THEN
                   READ(12,REC=IREC,ERR=240) (CRD(I),I=1,NPLREAD)
                   CW2(1:NPLREAD)=CRD(1:NPLREAD)
                ELSE
                   READ(12,REC=IREC,ERR=240) (CW2(I),I=1,NPLREAD)
                ENDIF
                CW1=0
                CALL REPAD_WITH_INDEX_ARRAY( MALLOC, IND, INDI, CW1(NPL+1), CW2)
              io_end
#ifdef MPI
              IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
                 CALLMPI( M_bcast_z(WDES%COMM_KINTER,CW1,SIZE(CW1)) )
              END IF
#endif
              CALL DIS_PW_BAND(WDES1, J+NBANDF, CW1, W%CW(1,1,K,1))
           ENDDO

        ENDIF

        DEALLOCATE(CW1,CW2,CRD,IND,INDI)

        ENDDO kpoints2

        IF (NKPTSF > WDES%NKPTS) THEN
           IREC=IREC+(NKPTSF-WDES%NKPTS)*(NBANDF+1)
        ENDIF

        ENDDO spin2

        ! copy eigenvalues and weights to all nodes
        NCOMM=WDES%NB_TOT*WDES%NKPTS
        CALLMPI( M_bcast_z(WDES%COMM, W%CELTOT(1,1,1),NCOMM ))
        CALLMPI( M_bcast_d(WDES%COMM, W%FERTOT(1,1,1),NCOMM ))

        IF (IU0 >= 0) WRITE(IU0,*) 'the WAVECAR file was read successfully'
        RETURN
      ENDIF

      IF (IU0>=0) THEN
         WRITE(IU0,*)'ERROR: while reading WAVECAR, file is incompatible'
         IF (WDES%ENMAX /= ENMAXF) &
            WRITE(*,*)'the energy cutoff has changed (new,old) ',WDES%ENMAX,ENMAXF
         IF  (NBANDF /= WDES%NB_TOT) &
            WRITE(*,*)'the number of bands has changed (new,old) ',WDES%NB_TOT,NBANDF
         IF (NKPTSF /= WDES%NKPTS) &
            WRITE(*,*)'the number of k-points has changed (new,old) ',WDES%NKPTS,NKPTSF
      ENDIF
      STOP
!=======================================================================
! can not do anything with WAVECAR 
! hard stop, pull all breaks
!=======================================================================
  200 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading WAVECAR, header is corrupt'
      STOP

  220 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading WAVECAR, plane wave coefficients changed', &
          NPL,NPLREAD
      STOP

  230 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading eigenvalues from WAVECAR',K,ISP
      STOP

  240 CONTINUE

      IF (IU0>=0) &
      WRITE(IU0,*)'ERROR: while reading plane wave coef. from WAVECAR',K,ISP,J
      STOP
      
      END SUBROUTINE
      
!*************************SUBROUTINE INWAV_ALTERNATIVE *****************
!
!   read wavefunctions from alternative file
!
!   read alternative WAVECAR file 
!
!***********************************************************************

      SUBROUTINE INWAV_ALTERNATIVE(IO, WDES, W, GRID, LATT_CUR, LREAD, EXT )

      USE base
      USE wave_high
      USE lattice
      
      IMPLICIT NONE

      TYPE (in_struct)  IO
      TYPE (wavedes)  WDES
      TYPE (wavespin) W
      TYPE (grid_3d)  GRID
      TYPE (latt)     LATT_CUR
      LOGICAL         LREAD
      CHARACTER (LEN=3)  EXT
    ! local
      TYPE (latt)     LATT_INI
      INTEGER         ISTART
      REAL(q)         ENMAXI
      REAL (q)        EFERMI

      LREAD = .FALSE.

      ISTART=1
      CALL OPENWAV(IO, WDES%COMM, EXT)
      CALL INWAV_HEAD(WDES, LATT_INI, LATT_CUR, ENMAXI, ISTART, IO%IU0)

      IF (ISTART/=0) THEN
         CALL ALLOCW( WDES, W)
         CALL INWAV_FAST(IO, WDES, W, GRID, LATT_CUR, LATT_INI, ISTART, EFERMI)
         LREAD = .TRUE.
         CALL VTUTOR('W','aux WAVECAR', &
         &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
         CALL VTUTOR('W','aux WAVECAR', &
         &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
      ENDIF

      END SUBROUTINE INWAV_ALTERNATIVE

!*************************SUBROUTINE OUTWAV    ************************
!
!   write wavefunctions to file
!   the first version writes a VASP 4 file
!   the second version writes the newer VASP 5 file format
!   which requires less storage if many bands are calculated
!
!***********************************************************************

      SUBROUTINE OUTWAV_4(IO, WDES, W, LATT_INI, EXT )
      USE base
      USE wave
      USE lattice
      USE main_mpi

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (in_struct)   IO
      TYPE (wavedes)  WDES
      TYPE (wavespin) W
      TYPE (latt)     LATT_INI
      CHARACTER (LEN=3), OPTIONAL :: EXT
! local work arrays
      TYPE (wavedes1) WDES1
      COMPLEX(q),ALLOCATABLE :: CW(:),EIG(:)
#ifdef WAVECAR_double
      COMPLEX(q), ALLOCATABLE :: CRD(:)
#else
      COMPLEX(qs),  ALLOCATABLE :: CRD(:)
#endif
    ! local 
      INTEGER NPL_TOT, IU0, IRECLW
      LOGICAL, POINTER :: LCONJG(:)=>null()
      INTEGER, POINTER :: INDEX(:)=>null()


      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif

!
! report to  stdout
! 
      IF (IO%IU6>=0) WRITE(IO%IU6,*)'writing wavefunctions'
      IF (IO%IU0>=0) WRITE(IO%IU0,*)'writing wavefunctions'
!
! open file 
!
      IU0   =IO%IU0
      NPL_TOT = MAXVAL(WDES%NPLWKP_TOT)
#ifdef WAVECAR_double
      IO%IRECLW=MAX(MAX(NPL_TOT,6),((WDES%NB_TOT*3+1)/2+2))*IO%ICMPLX
#else
      IO%IRECLW=MAX(MAX((NPL_TOT+1)/2,6),((WDES%NB_TOT*3+1)/2+2))*IO%ICMPLX
#endif
      IRECLW=IO%IRECLW
      
    ! open file
      IF (PRESENT(EXT)) THEN
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR.'//EXT,ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ELSE
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR',ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ENDIF

      ALLOCATE(CW(NPL_TOT),CRD(NPL_TOT),EIG(WDES%NB_TOT))

      RISPIN=WDES%ISPIN
      RDUM  =IRECLW

#ifdef WAVECAR_double
      RTAG  =45210
#else
      RTAG  =45200
#endif
      do_io WRITE(12,REC=1) RDUM,RISPIN,RTAG

      IREC=2
      io_begin
! in order to increase exchangeability of WAVECAR files across IEEE platforms
! avoid INTEGERS on output, write REAL(q) items instead (same below with RNPL)
      RNKPTS =WDES%NKPTS
      RNB_TOT=WDES%NB_TOT
      WRITE(12,REC=2) RNKPTS,RNB_TOT,WDES%ENMAX,((LATT_INI%A(I,J),I=1,3),J=1,3)
      io_end
!
! loop over spin, kpoints, bands
!
      spin: DO ISP=1,WDES%ISPIN
      kpoints: DO K=1,WDES%NKPTS

#ifdef MPI
       IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
          IF (WDES%COMM_KINTER%NODE_ME.NE.1) THEN
             IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                CYCLE
             ELSE
                CALLMPI( M_send_z(WDES%COMM_KINTER, 1, W%CW(1,1,K,ISP),SIZE(W%CW,1)*SIZE(W%CW,2)) )
             END IF
          ELSE
             IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                CALLMPI( M_recv_z(WDES%COMM_KINTER, MOD(K-1,WDES%COMM_KINTER%NCPU)+1, W%CW(1,1,K,ISP),SIZE(W%CW,1)*SIZE(W%CW,2)) )
             ENDIF
          ENDIF
       END IF
#endif
       CALL SETWDES(WDES,WDES1,K)
#ifdef gammareal
       CALL GEN_INDEX_GAMMA(WDES1, LATT_INI%B, WDES%ENMAX, INDEX, LCONJG)
#endif
       NPL=WDES%NPLWKP_TOT(K)
       RNPL=NPL
       IREC=IREC+1
! write number of plane waves, k-point coordinates and all eigenvalues and
! occupation numbers for current k-point K
       io_begin
! write eigenvalues in real format
         DO J=1,WDES%NB_TOT
           EIG(J)=REAL(W%CELTOT(J,K,ISP),KIND=q)
         ENDDO
         WRITE(12,REC=IREC) RNPL,WDES%VKPT(1,K),WDES%VKPT(2,K), &
                       WDES%VKPT(3,K),(EIG(J),W%FERTOT(J,K,ISP),J=1,WDES%NB_TOT)
       io_end

       DO J=1,WDES%NB_TOT
#ifdef gammareal
         CALL MRG_PW_BAND_GAMMA(WDES1, J, CW, W%CW(1,1,K,ISP), INDEX(1), LCONJG(1))
#else
         CALL MRG_PW_BAND(WDES1, J, CW, W%CW(1,1,K,ISP))
#endif

         IREC=IREC+1
         do_io CRD(1:NPL)=CW(1:NPL)
         do_io WRITE(12,REC=IREC) (CRD(I),I=1,NPL)
       ENDDO

#ifdef gammareal
       CALL FREE_INDEX_GAMMA(INDEX, LCONJG)
#endif
      ENDDO kpoints
      ENDDO spin

      DEALLOCATE(CW,CRD,EIG)

      CALL CLOSEWAV

      RETURN
      END SUBROUTINE


!*************************SUBROUTINE OUTWAV    ************************
!
!   write wavefunctions to file
!   this version uses smaller record lenght
!   and if required eigenvalues/ occupancies will span
!   several records
!
!***********************************************************************

      SUBROUTINE OUTWAV(IO, WDES, W, LATT_INI, EFERMI, EXT )
      USE base
      USE wave
      USE lattice
      USE main_mpi

      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (in_struct)   IO
      TYPE (wavedes)  WDES
      TYPE (wavespin) W
      TYPE (latt)     LATT_INI
      CHARACTER (LEN=3), OPTIONAL :: EXT
      REAL (q) EFERMI
! local work arrays
      TYPE (wavedes1) WDES1
      COMPLEX(q),ALLOCATABLE :: CW(:),EIG(:)
#ifdef WAVECAR_double
      COMPLEX(q), ALLOCATABLE :: CRD(:)
#else
      COMPLEX(qs),  ALLOCATABLE :: CRD(:)
#endif
    ! local 
      INTEGER NPL_TOT, IU0
      LOGICAL, POINTER :: LCONJG(:)=>null()
      INTEGER, POINTER :: INDEX(:)=>null()
      REAL(q), ALLOCATABLE :: OUTBUF(:)

      NODE_ME=0
      IONODE=0
#ifdef MPI
      NODE_ME=WDES%COMM%NODE_ME
      IONODE =WDES%COMM%IONODE
#endif

!
! report to  stdout
! 
      IF (IO%IU6>=0) WRITE(IO%IU6,*)'writing wavefunctions'
      IF (IO%IU0>=0) WRITE(IO%IU0,*)'writing wavefunctions'
!
! open file 
!
      IU0   =IO%IU0
      NPL_TOT = MAXVAL(WDES%NPLWKP_TOT)
#ifdef WAVECAR_double
      IO%IRECLW=MAX(NPL_TOT,7)*IO%ICMPLX
      IRECLW_OLD=MAX(MAX(NPL_TOT,6),((WDES%NB_TOT*3+1)/2+2))*IO%ICMPLX
#else
      IO%IRECLW=MAX((NPL_TOT+1)/2,7)*IO%ICMPLX
      IRECLW_OLD=MAX(MAX((NPL_TOT+1)/2,6),((WDES%NB_TOT*3+1)/2+2))*IO%ICMPLX
#endif
      
    ! open file

      IF (PRESENT(EXT)) THEN
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR.'//EXT,ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ELSE
         OPEN(UNIT=12,FILE=DIR_APP(1:DIR_LEN)//'WAVECAR',ACCESS='DIRECT', &
              FORM='UNFORMATTED',STATUS='UNKNOWN',RECL=IO%IRECLW)
      ENDIF

      ALLOCATE(CW(NPL_TOT),CRD(NPL_TOT),EIG(WDES%NB_TOT))

      RISPIN=WDES%ISPIN
      RDUM  =IO%IRECLW

#ifdef WAVECAR_double
      RTAG  =53310
#else
      RTAG  =53300
#endif
      ! if the record lenght is identical to previous vasp version we can claim
      ! to write the old file format
      IF (IRECLW_OLD==IO%IRECLW) THEN
#ifdef WAVECAR_double
         RTAG  =45210
#else
         RTAG  =45200
#endif
      ENDIF

      do_io WRITE(12,REC=1) RDUM,RISPIN,RTAG

      IREC=2
      io_begin
! in order to increase exchangeability of WAVECAR files across IEEE platforms
! avoid INTEGERS on output, write REAL(q) items instead (same below with RNPL)
      RNKPTS =WDES%NKPTS
      RNB_TOT=WDES%NB_TOT
      WRITE(12,REC=2) RNKPTS,RNB_TOT,WDES%ENMAX,((LATT_INI%A(I,J),I=1,3),J=1,3),EFERMI
      io_end
!
! loop over spin, kpoints, bands
!
      spin: DO ISP=1,WDES%ISPIN
      kpoints: DO K=1,WDES%NKPTS

#ifdef MPI
       IF (WDES%COMM_KINTER%NCPU.GT.1) THEN
          IF (WDES%COMM_KINTER%NODE_ME.NE.1) THEN
             IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                CYCLE
             ELSE
                CALLMPI( M_send_z(WDES%COMM_KINTER, 1, W%CW(1,1,K,ISP),SIZE(W%CW,1)*SIZE(W%CW,2)) )
             END IF
          ELSE
             IF (MOD(K-1,WDES%COMM_KINTER%NCPU).NE.WDES%COMM_KINTER%NODE_ME-1) THEN
                CALLMPI( M_recv_z(WDES%COMM_KINTER, MOD(K-1,WDES%COMM_KINTER%NCPU)+1, W%CW(1,1,K,ISP),SIZE(W%CW,1)*SIZE(W%CW,2)) )
             ENDIF
          ENDIF
       END IF
#endif
       CALL SETWDES(WDES,WDES1,K)
#ifdef gammareal
       CALL GEN_INDEX_GAMMA(WDES1, LATT_INI%B, WDES%ENMAX, INDEX, LCONJG)
#endif
       NPL=WDES%NPLWKP_TOT(K)
       RNPL=NPL
! write number of plane waves, k-point coordinates and all eigenvalues and
! occupation numbers for current k-point K
! write eigenvalues in real format
       io_begin
       ALLOCATE(OUTBUF(4+WDES%NB_TOT*3))

       OUTBUF(1)=RNPL
       OUTBUF(2)=WDES%VKPT(1,K)
       OUTBUF(3)=WDES%VKPT(2,K)
       OUTBUF(4)=WDES%VKPT(3,K)
       DO I=1,WDES%NB_TOT
          OUTBUF(5+(I-1)*3) =REAL(W%CELTOT(I,K,ISP),q)
          OUTBUF(6+(I-1)*3) =AIMAG(W%CELTOT(I,K,ISP))
          OUTBUF(7+(I-1)*3) =W%FERTOT(I,K,ISP)
       ENDDO
       CALL WRITE_FROM_BUF(12, IREC, IO%IRECLW/IO%ICMPLX*2, OUTBUF)
       DEALLOCATE(OUTBUF)
       io_end

       DO J=1,WDES%NB_TOT
#ifdef gammareal
         CALL MRG_PW_BAND_GAMMA(WDES1, J, CW, W%CW(1,1,K,ISP), INDEX(1), LCONJG(1))
#else
         CALL MRG_PW_BAND(WDES1, J, CW, W%CW(1,1,K,ISP))
#endif

         IREC=IREC+1
         do_io CRD(1:NPL)=CW(1:NPL)
         do_io WRITE(12,REC=IREC) (CRD(I),I=1,NPL)
       ENDDO

#ifdef gammareal
       CALL FREE_INDEX_GAMMA(INDEX, LCONJG)
#endif
      ENDDO kpoints
      ENDDO spin

      DEALLOCATE(CW,CRD,EIG)

      CALL CLOSEWAV

      RETURN
      END SUBROUTINE


      SUBROUTINE WRITE_FROM_BUF(IU, IREC, IRECL_REAL, OUTBUF)
      INTEGER :: IU         ! unit for writing
      INTEGER :: IREC       ! record to write
      INTEGER :: IRECL_REAL ! record lenght in real(q) words
      REAL(q) :: OUTBUF(:)  ! buffer to written to file

      ! local
      INTEGER :: I

      DO I=1,(SIZE(OUTBUF)+IRECL_REAL-1)/IRECL_REAL
         IREC=IREC+1
         WRITE(12,REC=IREC) OUTBUF((I-1)*IRECL_REAL+1 : MIN(I*IRECL_REAL, SIZE(OUTBUF)))
      ENDDO

      END SUBROUTINE

      SUBROUTINE READ_TO_BUF(IU, IREC, IRECL_REAL, INBUF, IFAIL)
      INTEGER :: IU         ! unit for reading
      INTEGER :: IREC       ! record to read from
      INTEGER :: IRECL_REAL ! record lenght in real(q) words
      REAL(q) :: INBUF(:)   ! buffer to read to
      INTEGER :: IFAIL      ! error condition

      ! local
      INTEGER :: I

      IFAIL=0
      DO I=1,(SIZE(INBUF)+IRECL_REAL-1)/IRECL_REAL
         IREC=IREC+1
         READ(12,REC=IREC, IOSTAT=IFAIL) INBUF((I-1)*IRECL_REAL+1 : MIN(I*IRECL_REAL, SIZE(INBUF)))
         IF (IFAIL/=0) RETURN
      ENDDO

      END SUBROUTINE

!*************************SUBROUTINE OUTCHG ****************************
!
!   write chargedensity to a specified unit
!   HEADER is currently created in the main program
!
!***********************************************************************

      SUBROUTINE OUTCHG(GRIDC, IU, LLONG,CHTOT)
      USE prec
      USE mpimy
      USE mgrid
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d) GRIDC

      COMPLEX(q)  CHTOT(GRIDC%RC%NP)
      LOGICAL LLONG        ! long or short format
! work arrays
      RGRID,  ALLOCATABLE ::  CWORK(:)
      REAL(q),ALLOCATABLE ::  WORK (:)
      INTEGER NALLOC,NZ, NWRITE, NWRITTEN
      CHARACTER (40) FORM
      INTEGER ISTAT

      NALLOC=GRIDC%NGX*GRIDC%NGY

      ALLOCATE(CWORK(DIMREAL(GRIDC%MPLWV)),WORK(NALLOC),STAT=ISTAT)
      IF (ISTAT>0) RETURN ! can not write the charge immediate exit

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
#endif

      IF (GRIDC%NPLWV/= GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ) THEN
        WRITE(*,*)'internal ERROR: OUTCHG NPLWV is not compatibel', &
     &   ' with  NGX,NGY,NGZ'
        WRITE(*,*)'   ',GRIDC%NPLWV,GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ
        STOP
      ENDIF
      ! transfer to CWORK and FFT
      CALL RC_ADD(CHTOT,1.0_q,CHTOT,0.0_q,CWORK,GRIDC)
      CALL FFT3D(CWORK,GRIDC,1)

      IF (LLONG) THEN
        FORM='(1(1X,E17.11))'
        NWRITE=5
      ELSE
        FORM='(1(1X,G11.5))'
        NWRITE=10
      ENDIF

      do_io WRITE(IU,'(3I5)') GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ

      NWRITTEN=0
      DO NZ=1,GRIDC%NGZ
         CALL MRG_GRID_RL_PLANE(GRIDC, WORK, CWORK, NZ)
         io_begin
         DO N=1,NALLOC
            NWRITTEN=NWRITTEN+1
            IF ( MOD(NWRITTEN,NWRITE)==0 ) THEN
               WRITE(IU,FORM) WORK(N)
            ELSE
               WRITE(IU,FORM,ADVANCE='NO') WORK(N)
            ENDIF
         ENDDO
         io_end
      ENDDO
      IF ( MOD(NWRITTEN,NWRITE)/=0 ) WRITE(IU,*)' '

      DEALLOCATE(CWORK,WORK)

      RETURN
      END SUBROUTINE


!*************************SUBROUTINE  INCHG ****************************
!   read chargedensity form a specified unit
!   HEADER must have been read previously
!***********************************************************************

      SUBROUTINE INCHG(IU,CHTOT,GRID,IERR)
      USE prec
      USE charge
      USE mpimy
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRID
      COMPLEX(q)   CHTOT(GRID%RC%NP)
! work array
      REAL(q),ALLOCATABLE ::  WORK(:)
      INTEGER NALLOC,NZ, NWRITE, NWRITTEN
      INTEGER ISTAT

      NALLOC=GRID%NGX*GRID%NGY
      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRID%COMM%NODE_ME
      IONODE =GRID%COMM%IONODE
#endif

      ALLOCATE(WORK(NALLOC),STAT=ISTAT)
      IF (ISTAT>0) THEN
         IERR=2
         RETURN                 ! can not read potential, immediate exit
      ENDIF

      IERR=0
      IF (GRID%NPLWV/= GRID%NGX*GRID%NGY*GRID%NGZ) THEN
        WRITE(*,*)'internal ERROR: INCHG GRID%NPLWV is not compatible', &
     &   ' with  GRID%NGX,GRID%NGY,NGZC'
        WRITE(*,*)'   ',GRID%NPLWV,GRID%NGX,GRID%NGY,GRID%NGZ
        STOP
      ENDIF

      IERR=0
      NGXFIL=0; NGYFIL=0; NGZFIL=0

#ifdef MPI
      IF (NODE_ME==IONODE) THEN
#endif
        READ(IU,*,ERR=120,END=120) NGXFIL,NGYFIL,NGZFIL
 120    CONTINUE
        IF (NGXFIL==0 .AND. NGYFIL==0 .AND. NGZFIL==0) THEN
           IERR=2
        ELSE IF (NGXFIL/=GRID%NGX .OR. NGYFIL/=GRID%NGY .OR. NGZFIL/=GRID%NGZ ) THEN
          IERR=1
        ENDIF
#ifdef MPI
      ENDIF
#endif

      CALLMPI( M_bcast_i( GRID%COMM, IERR, 1))
      IF (IERR /=0 ) GOTO 200

      NWRITE=5
      NWRITTEN=0
      DO NZ=1,GRID%NGZ
         io_begin
         DO N=1,NALLOC
            NWRITTEN=NWRITTEN+1
            ! non advancing read is not the same on all machine
            ! this is the only version that works on all tested platforms
            IF ( MOD(NWRITTEN,NWRITE)==0 .OR. (N==NALLOC .AND. NZ==GRID%NGZ)) THEN
               READ(IU,'(1(1X,E17.11))',ERR=100,END=100) WORK(N)
            ELSE
               READ(IU,'(1(1X,E17.11))',ADVANCE='NO',ERR=100,END=100) WORK(N)
            ENDIF
         ENDDO
         io_end
         CALL DIS_GRID_RL_PLANE(GRID, WORK, CHTOT, .TRUE., NZ )
      ENDDO
 
      CALL FFT_RC_SCALE(CHTOT,CHTOT,GRID)
      CALL SETUNB_COMPAT(CHTOT,GRID)

  200 CONTINUE
      DEALLOCATE(WORK)
      RETURN

  100 CONTINUE
      IERR=2
      DEALLOCATE(WORK)
      RETURN
      END SUBROUTINE

!*************************SUBROUTINE READNI ****************************
!   reads total number of ions from a line consisting of numbers
!   for several species
!***********************************************************************

      SUBROUTINE READNI(ZPARSE,NIONSF)
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)
      CHARACTER (80) ZPARSE
      CHARACTER (80) ZITEM
      CHARACTER (80) ZWORK
      CHARACTER (15) ZFORM
      CHARACTER (1)  ZCHAR

!  get number of data items
      NDATA=NITEMS(ZPARSE,ZWORK,.TRUE.,'I')

!  parse and read list:
      NIONSF=0

      DO 100 IDATA=1,NDATA
         CALL SUBWRD(ZPARSE,ZITEM,IDATA,1)
         CALL CHKINT(ZITEM,ZWORK,ZCHAR,ZFORM)
         IF (ZCHAR=='Y') THEN
            ZWORK='('//ZFORM//')'
            CALL STRIP(ZWORK,LFORM,'A')
            READ(ZITEM,ZWORK(1:LFORM)) NI
         ELSE
            WRITE(*,*) 'Fatal error in READNI: Invalid data found ...'
            STOP
         ENDIF

         NIONSF=NIONSF+NI

  100 CONTINUE

      RETURN
      END SUBROUTINE

!*************************SUBROUTINE READCH ****************************
!
!  This routine reads in a chargedensity from file IU
!  when the chargedensity of the file is incompatible
!  i.e. NGX,Y,Z differ from that used in the programm a warning is
!  reported and ICHARG is set to 0
!  if the magnetisation charge density could not be read in 
!  ICHARG is set to -1
!  on entry to the routine:
!     CHTOT  may be uninitialised
!     RHOLM  must be already correctly initialised according to MAGMOM
!  on exit:
!     CHTOT  as read from file
!     RHOLM  as read from file
!  the routine is complicated by several special cases:
!  o if the magnetisation charge density could not be read in 
!     ICHARG is set to -1, CHTOT(:,2:4)=0
!  o non collinear case: if only mz could be read
!     the x and y components are set to 0 in CHTOT and RHOLM
!
!***********************************************************************

      SUBROUTINE READCH(GRIDC, LOVERL, T_INFO, CHTOT, RHOLM, ICHARG, ISPIN, &
           LATT_CUR, P, CSTRF, IU, IU0)
      USE prec
      USE lattice
      USE poscar
      USE mgrid
      USE pseudo
      USE charge
      USE pawm

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      COMPLEX(q)   CSTRF(GRIDC%MPLWV,T_INFO%NTYP)
      INTEGER ISPIN, IU, IU0, ICHARG
      REAL(q)         :: RHOLM(:,:)
! local work arrays
      TYPE (type_info)   T_INFO_OLD
      REAL(q)         :: TMP(T_INFO%NIONS)
      COMPLEX(q) CHTOT(GRIDC%MPLWV,ISPIN)
      CHARACTER (40) TITEL
      CHARACTER (80) ZPARSE
      CHARACTER (1)  CHARAC
      COMPLEX(q), ALLOCATABLE :: CD(:),CSTRF_OLD(:,:)
      INTEGER I,NIONSF,IERR,N
      LOGICAL LOVERL

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
#endif

!=======================================================================
! read in old ionic-positions and header
!=======================================================================
      T_INFO_OLD=T_INFO

      NULLIFY(T_INFO_OLD%POSION)
      ALLOCATE(T_INFO_OLD%POSION(3,T_INFO_OLD%NIONS))

      READ(IU,*,ERR=1000,END=1000) TITEL
      DO I=1,4
        READ(IU,*,ERR=1000,END=1000)
      ENDDO
      READ(IU,'(A80)') ZPARSE
      READ(ZPARSE,*) CHARAC
      IF (.NOT.(CHARAC>='0' .AND. CHARAC<='9')) READ(IU,'(A80)') ZPARSE

      CALL READNI(ZPARSE,NIONSF)
      IF (NIONSF/=T_INFO%NIONS) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: number of atoms are different on CHGCAR file'
        ICHARG=0
        RETURN
      ENDIF
      READ(IU,*,ERR=1000,END=1000)

      DO  I=1,NIONSF
        READ(IU,*) T_INFO_OLD%POSION(:,I)
      ENDDO
!=======================================================================
! read in the charge-density
!=======================================================================
      CALL INCHG(IU,CHTOT(1,1),GRIDC,IERR)
      IF (IERR==1) THEN
        IF (IU0>=0) &
        WRITE(*,*) 'WARNING: dimensions on CHGCAR file are different'
        ICHARG=0
        RETURN
      ENDIF
      IF (IERR==2) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: chargedensity file is incomplete'
        ICHARG=0
        RETURN
      ENDIF
!=======================================================================
! now subtract charge density according to old position and add
! that one according to new positions
!=======================================================================
      ALLOCATE(CD(GRIDC%MPLWV),CSTRF_OLD(GRIDC%MPLWV,T_INFO%NTYP))

!---- subtract atomic charge-denisty for old positions
      CALL STUFAK(GRIDC,T_INFO_OLD,CSTRF_OLD)
      CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO_OLD,LATT_CUR%B,P,CSTRF_OLD,CD)
      DO N=1,GRIDC%RC%NP
        CHTOT(N,1)= CHTOT(N,1)- CD(N)
      ENDDO

!---- add atomic charge-denisty for new positions
      CALL RHOATO_WORK(.FALSE.,.FALSE.,GRIDC,T_INFO,LATT_CUR%B,P,CSTRF,CD)
      DO N=1,GRIDC%RC%NP
        CHTOT (N,1)= CHTOT(N,1)+ CD(N)
      ENDDO

      DEALLOCATE(CD,CSTRF_OLD,T_INFO_OLD%POSION)

      ! just in case initialize the magnetization density to 0
      DO ISP=2,ISPIN
        CALL RC_ADD(CHTOT(1,ISP),0.0_q,CHTOT(1,ISP),0.0_q,CHTOT(1,ISP),GRIDC)
      ENDDO

      CALL RD_RHO_PAW(P, T_INFO, LOVERL, RHOLM(:,1), GRIDC%COMM, IU, IERR )
      IF (IERR/=0) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: PAW occupancies are missing on CHGCAR '
        ICHARG=0
        RETURN
      ENDIF
!=======================================================================
! set/read spin density
!=======================================================================
      IF (IU0>=0) &
        WRITE(IU0,*) 'charge-density read from file: ',TITEL

      DO ISP=2,ISPIN
! read in the spin-density
        IERR=0
        ! in vasp.4.4 the ATOMOM array was read from the CHGCAR file
        ! reading the magnetic moments from the file really makes no sense
        ! and spoiles non collinear calculations
        do_io READ(IU,*,IOSTAT=IERR) (TMP(I),I=1,T_INFO%NIONS)
        CALLMPI( M_bcast_i( GRIDC%COMM, IERR, 1))

        IF (IERR==0) CALL INCHG(IU,CHTOT(1,ISP),GRIDC,IERR)
        IF (IERR==0) CALL RD_RHO_PAW(P, T_INFO, LOVERL, RHOLM(:,ISP), GRIDC%COMM, IU, IERR )

        ! error occured
        IF (IERR/=0) THEN
           IF (ISP==2) THEN
           ! we could not read any entry in the magnetisation density return ICHARG=-1
           ! (magnetisation according to overlapping atoms)
              ICHARG=-1
              RETURN
           ELSE
           ! non collinear case we could read at least m_z(r)
           ! copy that to the right place and initialise everything else to 0
              CALL RC_ADD(CHTOT(1,2),1.0_q,CHTOT(1,2),0.0_q,CHTOT(1,4),GRIDC) ! copy
              CALL RC_ADD(CHTOT(1,2),0.0_q,CHTOT(1,2),0.0_q,CHTOT(1,2),GRIDC) ! =0
              CALL RC_ADD(CHTOT(1,3),0.0_q,CHTOT(1,3),0.0_q,CHTOT(1,3),GRIDC) ! =0
              
              RHOLM(:,4)=RHOLM(:,2)
              RHOLM(:,2)=0
              RHOLM(:,3)=0

              RETURN
           ENDIF
        ENDIF

        IF (IU0>=0) WRITE(IU0,'(A,I2)') ' magnetization density read from file',ISP-1

      ENDDO
      RETURN

      ICHARG=-1
      RETURN

 1000 CONTINUE
      IF (IU0>=0) &
       WRITE(IU0,*) 'WARNING: chargedensity file is incomplete'
      ICHARG=0
      RETURN
      END SUBROUTINE


!*************************SUBROUTINE READPOT ***************************
!
!  This routine reads in a chargedensity from file IU
!  when the chargedensity of the file is incompatible
!  i.e. NGX,Y,Z differ from that used in the programm a warning is
!  reported and ICHARG is set to 0
!  if the magnetisation charge density could not be read in 
!  ICHARG is set to -1
!  on entry to the routine:
!     CHTOT  may be uninitialised
!     RHOLM  must be already correctly initialised according to MAGMOM
!  on exit:
!     CHTOT  as read from file
!     RHOLM  as read from file
!  the routine is complicated by several special cases:
!  o if the magnetisation charge density could not be read in 
!     ICHARG is set to -1, CHTOT(:,2:4)=0
!  o non collinear case: if only mz could be read
!     the x and y components are set to 0 in CHTOT and RHOLM
!
!***********************************************************************

      SUBROUTINE READPOT(GRIDC, T_INFO, CVTOT, ICHARG, WDES, &
              LATT_CUR, P, LMDIM, CDIJ, N_MIX_PAW, IU, IU0)

      USE prec
      USE lattice
      USE poscar
      USE mgrid
      USE pseudo
      USE charge
      USE pawm
      USE wave

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      TYPE (grid_3d)     GRIDC
      TYPE (type_info)   T_INFO
      TYPE (potcar)      P (T_INFO%NTYP)
      TYPE (latt)        LATT_CUR
      TYPE (wavedes)     WDES
      INTEGER LMDIM
      OVERLAP  CDIJ(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      INTEGER N_MIX_PAW
      INTEGER IU, IU0, ICHARG
! local work arrays
      REAL(q)         :: DLM(N_MIX_PAW,WDES%NCDIJ)
      TYPE (type_info)   T_INFO_OLD
      REAL(q)         :: TMP(T_INFO%NIONS)
      COMPLEX(q) CVTOT(GRIDC%MPLWV,WDES%NCDIJ)
      CHARACTER (40) TITEL
      CHARACTER (80) ZPARSE
      CHARACTER (1)  CHARAC
      INTEGER I,NIONSF,IERR,N

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
#endif

!=======================================================================
! read in old ionic-positions and header
!=======================================================================
      T_INFO_OLD=T_INFO

      NULLIFY(T_INFO_OLD%POSION)
      ALLOCATE(T_INFO_OLD%POSION(3,T_INFO_OLD%NIONS))

      READ(IU,*,ERR=1000,END=1000) TITEL
      DO I=1,4
        READ(IU,*,ERR=1000,END=1000)
      ENDDO
      READ(IU,'(A80)') ZPARSE
      READ(ZPARSE,*) CHARAC
      IF (.NOT.(CHARAC>='0' .AND. CHARAC<='9')) READ(IU,'(A80)') ZPARSE

      CALL READNI(ZPARSE,NIONSF)
      IF (NIONSF/=T_INFO%NIONS) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: number of atoms are different on CHGCAR file'
        ICHARG=0
        RETURN
      ENDIF
      READ(IU,*,ERR=1000,END=1000)

      DO  I=1,NIONSF
        READ(IU,*) T_INFO_OLD%POSION(:,I)
      ENDDO
!=======================================================================
! read in the charge-density
!=======================================================================
      CALL INCHG(IU,CVTOT(1,1),GRIDC,IERR)
      CALL FFT3D(CVTOT(1,1),GRIDC,1)

      IF (IERR==1) THEN
        IF (IU0>=0) &
        WRITE(*,*) 'WARNING: dimensions on LOCPOT file are different'
        ICHARG=0
        RETURN
      ENDIF
      IF (IERR==2) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: potential file is incomplete'
        ICHARG=0
        RETURN
      ENDIF

      DEALLOCATE(T_INFO_OLD%POSION)

      ! just in case initialize the magnetization density to 0
      DO ISP=2,WDES%NCDIJ
        CALL RC_ADD(CVTOT(1,ISP),0.0_q,CVTOT(1,ISP),0.0_q,CVTOT(1,ISP),GRIDC)
      ENDDO

      DLM(:,1)=0
      CALL RD_RHO_PAW(P, T_INFO, WDES%LOVERL, DLM(:,1), GRIDC%COMM, IU, IERR )
      IF (IERR/=0) THEN
        IF (IU0>=0) &
        WRITE(IU0,*) 'WARNING: PAW occupancies are missing on POT file'
        ICHARG=0
        RETURN
      ENDIF
!=======================================================================
! set/read spin density
!=======================================================================
      IF (IU0>=0) &
        WRITE(IU0,*) 'local potential read from file: ',TITEL

      DO ISP=2,WDES%NCDIJ
! read in the spin-density
        IERR=0
        ! in vasp.4.4 the ATOMOM array was read from the CHGCAR file
        ! reading the magnetic moments from the file really makes no sense
        ! and spoiles non collinear calculations
        do_io READ(IU,*,IOSTAT=IERR) (TMP(I),I=1,T_INFO%NIONS)
        CALLMPI( M_bcast_i( GRIDC%COMM, IERR, 1))

        IF (IERR==0) CALL INCHG(IU,CVTOT(1,ISP),GRIDC,IERR)
        CALL FFT3D(CVTOT(1,ISP),GRIDC,1)

        IF (IERR==0) CALL RD_RHO_PAW(P, T_INFO, WDES%LOVERL, DLM(:,ISP), GRIDC%COMM, IU, IERR )

        ! error occured
        IF (IERR/=0) THEN
           IF (ISP==2) THEN
           ! we could not read any entry in the magnetisation density return ICHARG=-1
           ! (magnetisation according to overlapping atoms)
              ICHARG=-1
              RETURN
           ELSE
           ! non collinear case we could read at least m_z(r)
           ! copy that to the right place and initialise everything else to 0
              CALL RC_ADD(CVTOT(1,2),1.0_q,CVTOT(1,2),0.0_q,CVTOT(1,4),GRIDC) ! copy
              CALL RC_ADD(CVTOT(1,2),0.0_q,CVTOT(1,2),0.0_q,CVTOT(1,2),GRIDC) ! =0
              CALL RC_ADD(CVTOT(1,3),0.0_q,CVTOT(1,3),0.0_q,CVTOT(1,3),GRIDC) ! =0
              
              DLM(:,4)=DLM(:,2)
              DLM(:,2)=0
              DLM(:,3)=0

              GOTO 2000
           ENDIF
        ENDIF

        IF (IU0>=0) WRITE(IU0,'(A,I2)') ' magnetization potential read from file',ISP-1

      ENDDO

 2000 CONTINUE
      ! one centre corrections are passed through mixer and added now
      CALL RETRIVE_RHO_PAW(WDES, P, T_INFO, WDES%LOVERL, LMDIM, CDIJ, DLM)
      
      RETURN

 1000 CONTINUE
      IF (IU0>=0) &
       WRITE(IU0,*) 'WARNING: potential file is incomplete'
      ICHARG=0
      RETURN
      END SUBROUTINE


!*************************SUBROUTINE OPENGAMMA *************************
!
! the followin subroutines open and close the file GAMMA for reading
! and writing
! furthermore they write and read the header of the file
!
!***********************************************************************

      SUBROUTINE OPENGAMMA
        USE main_mpi
        OPEN(UNIT=IU_GAMMA  ,FILE=DIR_APP(1:DIR_LEN)//'GAMMA',STATUS='UNKNOWN')

      END SUBROUTINE OPENGAMMA


      SUBROUTINE CLOSEGAMMA

        CLOSE(UNIT=IU_GAMMA)

      END SUBROUTINE CLOSEGAMMA

      SUBROUTINE WRITEGAMMA_HEAD( NKPTS , NBANDS, IO)
        USE base
        IMPLICIT NONE
        TYPE (in_struct)   IO
        INTEGER :: NKPTS   ! number of k-points
        INTEGER :: NBANDS  ! number of bands
        
        IF (IO%IU0>=0) THEN
           WRITE(IO%IU0,*) 'writing the density matrix to GAMMA'
        ENDIF
        IF (IO%IU6>=0) THEN
           WRITE(IU_GAMMA,'(2I10," ! number of k-points and bands")') NKPTS, NBANDS
        ENDIF

      END SUBROUTINE WRITEGAMMA_HEAD


      SUBROUTINE WRITEGAMMA_HEAD_HAM( NKPTS , NBANDS, IO)
        USE base
        IMPLICIT NONE
        TYPE (in_struct)   IO
        INTEGER :: NKPTS   ! number of k-points
        INTEGER :: NBANDS  ! number of bands
        
        IF (IO%IU0>=0) THEN
           WRITE(IO%IU0,*) 'writing the density matrix to GAMMA'
        ENDIF
        IF (IO%IU6>=0) THEN
           WRITE(IU_GAMMA,'("Hamiltonian",2I10," ! number of k-points and bands")') NKPTS, NBANDS
        ENDIF

      END SUBROUTINE WRITEGAMMA_HEAD_HAM

      SUBROUTINE READGAMMA_HEAD( IMODE, NKPTS , NBANDS, IO)
        USE base
        IMPLICIT NONE
        TYPE (in_struct)   IO
        INTEGER :: NKPTS    ! number of k-points
        INTEGER :: NBANDS   ! number of bands
        CHARACTER (LEN=80) ::STRING ! header of Gamma file
        CHARACTER (LEN=1)  ::MODE   !  mode selector
        INTEGER :: IMODE    ! mode: IMODE=1 density matrix ; IMODE=2 Hamiltonian
     ! local
        INTEGER :: NKPTS_   ! number of k-points
        INTEGER :: NBANDS_  ! number of bands
        INTEGER :: IERR

     ! read first line
        READ(IU_GAMMA,'(A)') STRING
     ! try old format first
        IMODE=1
        READ(STRING,*, IOSTAT = IERR) NKPTS_, NBANDS_

        IF (IERR/=0) THEN
     ! try new format
           READ(STRING,*, IOSTAT = IERR) MODE, NKPTS_, NBANDS_
           IF (MODE=='H' .OR. MODE=='h') IMODE=2
        ENDIF

        ! if you do not care, one may simply supply -1 in the file
        IF (NKPTS_==-1)  NKPTS_=NKPTS
        IF (NBANDS_==-1) NBANDS_=NBANDS

        IF (IERR/= 0 .OR. NBANDS /= NBANDS_ .OR. NKPTS_ /= NKPTS) THEN
           CALL VTUTOR('S','GAMMA', &
       &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
           CALL VTUTOR('S','GAMMA', &
       &               0.0_q,1,1,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)

           STOP
        ENDIF

        WRITE(IO%IU0,*) 'reading the density matrix from GAMMA'

      END SUBROUTINE READGAMMA_HEAD


!*************************SUBROUTINE WRITEGAMMA *************************
!
! the following subroutine write and read the change of the density matrix
! for one k-point
! the matrix MUST not be stored in a distributed fashion, and
! only one core must call the routine
!
! READGAMMA may be either called by all cores, or better
! called only by one core with a subsequent broadcast of the
! data to all cores
!
!***********************************************************************

      SUBROUTINE WRITEGAMMA( NK, NB_TOT, NBDIM, CHAM , LROT)
        USE prec
        IMPLICIT NONE
        INTEGER :: NK    ! k-point index
        INTEGER :: NB_TOT! total number of bands
        INTEGER :: NBDIM ! leading dimension of CHAM
        GDEF :: CHAM(NBDIM, NB_TOT)
        LOGICAL :: LROT  ! one can also supply an approximate unitary rotation matrix
                         ! for instance obtained by Loewdin perturbation theory
        INTEGER :: N1, N2
        INTEGER :: NB_START, NB_END
        
        ! change sign in the upper triangle (convert unitary matrix to density matrix)
        IF (LROT) THEN
           DO N1=1, NB_TOT
              DO N2=N1+1, NB_TOT
                 CHAM(N1, N2)= -CHAM(N1, N2)
              ENDDO
              CHAM(N1, N1)=0
           ENDDO
        ENDIF

        ! write starting band, and final band
        NB_START = 1
        NB_END = NB_TOT
        WRITE(IU_GAMMA, '(3I10," ! k-point index, start and end band")') NK, NB_START, NB_END

        DO N1=NB_START, NB_END
           WRITE(IU_GAMMA,'(16E15.7)') CHAM(N1, NB_START:NB_END)
        ENDDO
        
        ! change back to rotation matrix)
        IF (LROT) THEN
           DO N1=1, NB_TOT
              DO N2=N1+1, NB_TOT
                 CHAM(N1, N2)= -CHAM(N1, N2)
              ENDDO
              CHAM(N1, N1)=1
           ENDDO
        ENDIF

      END SUBROUTINE WRITEGAMMA


      SUBROUTINE READGAMMA( NK, NB_TOT, NBDIM, CHAM, IO )
        USE prec
        USE base
        IMPLICIT NONE
        TYPE (in_struct)   IO
        INTEGER :: NK    ! k-point index
        INTEGER :: NB_TOT! total number of bands
        INTEGER :: NBDIM ! leading dimension of CHAM
        GDEF :: CHAM(NBDIM, NB_TOT)        
      ! local
        INTEGER :: NK_
        INTEGER :: N1, N2
        INTEGER :: NB_START, NB_END
        REAL(q) :: BUFFER(NBDIM*2)

        ! read starting band, and final band
        NB_START = 1
        NB_END = NB_TOT
        READ(IU_GAMMA, *) NK_, NB_START, NB_END

        NB_START=MAX(1, NB_START)
        NB_END  =MIN(NB_TOT, NB_END)
        IF (NK_==-1) NK_=NK

        IF (NK /= NK_) THEN
           CALL VTUTOR('S','GAMMA2', &
       &               0.0_q,1,NK ,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU0,3)
           CALL VTUTOR('S','GAMMA2', &
       &               0.0_q,1,NK ,1,(0.0_q,0.0_q),1,.TRUE.,1,IO%IU6,3)
           STOP           
        ENDIF

        CHAM=0
        DO N1=NB_START,NB_END
#ifdef gammareal
           READ(IU_GAMMA,*) CHAM(N1, NB_START:NB_END)
#else
           ! not to read into real buffer (since complex entries must be in parantheses)
           READ(IU_GAMMA,*) BUFFER(1:(NB_END-NB_START+1)*2)
           ! store as complex
           DO N2=NB_START,NB_END
              CHAM(N1, N2)=CMPLX(BUFFER((N2-NB_START)*2+1),BUFFER((N2-NB_START)*2+2))
           ENDDO
#endif

        ENDDO
      END SUBROUTINE READGAMMA

      END MODULE


!*************************SUBROUTINE OUTPOT ****************************
!   write potential     to a specified unit
!   HEADER is currently created in the main program
!***********************************************************************

      SUBROUTINE OUTPOT(GRIDC, IU, LLONG, CVTOT)
      USE prec
      USE mgrid
      IMPLICIT REAL(q) (A-H,O-Z)

      TYPE (grid_3d) GRIDC
      RGRID CVTOT(GRIDC%RL%NP)
      LOGICAL LLONG
      CHARACTER (40) FORM
! local variables
      INTEGER NALLOC,NZ, NWRITE, NWRITTEN
      REAL(q),ALLOCATABLE ::  WORK(:)
      INTEGER ISTAT

      NODE_ME=0
      IONODE =0
#ifdef MPI
      NODE_ME=GRIDC%COMM%NODE_ME
      IONODE =GRIDC%COMM%IONODE
#endif

      NALLOC=GRIDC%NGX*GRIDC%NGY

      ALLOCATE(WORK(NALLOC),STAT=ISTAT)
      IF (ISTAT>0) RETURN ! can not write the potential immediate exit

      IF (GRIDC%NPLWV/= GRIDC%NGX*GRIDC%NGY*GRIDC%NGZ) THEN
        WRITE(*,*)'internal ERROR: OUTPOT GRIDC%NPLWV is not compatibel', &
     &   ' with  GRIDC%NGX,GRIDC%NGY,NGZC'
        WRITE(*,*)'   ',GRIDC%NPLWV,GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ
        STOP
      ENDIF

      IF (LLONG) THEN
        FORM='(1(1X,E17.11))'
        NWRITE=5
      ELSE
        FORM='(1G11.5)'
        NWRITE=10
      ENDIF

      do_io WRITE(IU,'(3I5)') GRIDC%NGX,GRIDC%NGY,GRIDC%NGZ

      NWRITTEN=0
      DO NZ=1,GRIDC%NGZ
         CALL MRG_GRID_RL_PLANE(GRIDC, WORK, CVTOT, NZ)
         io_begin
         DO N=1,NALLOC
            NWRITTEN=NWRITTEN+1
            IF ( MOD(NWRITTEN,NWRITE)==0 ) THEN
               WRITE(IU,FORM) WORK(N)
            ELSE
               WRITE(IU,FORM,ADVANCE='NO') WORK(N)
            ENDIF
         ENDDO
         io_end
      ENDDO
      IF ( MOD(NWRITTEN,NWRITE)/=0 ) WRITE(IU,*)' '

      DEALLOCATE(WORK)

      RETURN
      END SUBROUTINE



!***********************************************************************
!  write out initial header for PCDAT
!***********************************************************************

      SUBROUTINE PCDAT_HEAD(IU, T_INFO, LATT_CUR, DYN, PACO, SZNAM1)
      USE prec
      USE lattice
      USE poscar
      USE base
      IMPLICIT NONE

      INTEGER IU
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_INFO
      TYPE (dynamics)  :: DYN
      TYPE (paco_struct)  PACO
      CHARACTER (40) SZNAM1
! local variables
      INTEGER I
      REAL(q) AOMEGA
      AOMEGA=LATT_CUR%OMEGA/T_INFO%NIONS

      WRITE(IU,'(4I4,2E15.7)')1,T_INFO%NIONS,1,0,AOMEGA,DYN%TEMP
      WRITE(IU,*) ' CAR '
      WRITE(IU,*) SZNAM1
      WRITE(IU,'(3I4)') 0,0,0
      WRITE(IU,'(2I4)') 1,DYN%NBLOCK
      WRITE(IU,'(3I4)') PACO%NPACO,PACO%NPACO,PACO%NPACO
      WRITE(IU,'(1I4)') PACO%NPACO
      WRITE(IU,'(1E15.7)') 1E-10
      WRITE(IU,'(1E15.7)') PACO%APACO*1E-10/PACO%NPACO
      WRITE(IU,'(1I4)') DYN%NSW/DYN%NBLOCK/DYN%KBLOCK
      WRITE(IU,'(4E15.7)') DYN%POTIM*1E-15,((LATT_CUR%ANORM(I)*1E-10),I=1,3)

      RETURN
      END SUBROUTINE


!***********************************************************************
!
! handling of vasp.lock  file
! this file created (open and close)
! there are three functions
! CREATE_VASP_LOCK creates the vasp.lock file on the root node
! REMOVE_VASP_LOCK removes the vasp.lock file on the root node
! WAIT_VASP_LOCK   waits for the file to be created by an external process
!
!***********************************************************************

      SUBROUTINE CREATE_VASP_LOCK(MYCOMM)
        USE mpimy
        USE main_mpi
        TYPE (communic) :: MYCOMM
        
#ifdef MPI
        IF (MYCOMM%NODE_ME==MYCOMM%IONODE) THEN
#endif
           OPEN(UNIT=23,FILE=DIR_APP(1:DIR_LEN)//'vasp.lock',STATUS='UNKNOWN')
           WRITE(23,*)'vasp is running happily'
           CLOSE(UNIT=23)
#ifdef MPI
        ENDIF
#endif
      END SUBROUTINE


      SUBROUTINE WAIT_VASP_LOCK(MYCOMM)
        USE mpimy
        USE main_mpi
        TYPE (communic) :: MYCOMM

      ! local
        INTEGER :: ISTAT

        DO
           ISTAT=0
           ! try opening file on root node
#ifdef MPI
           IF (MYCOMM%NODE_ME==MYCOMM%IONODE) THEN
#endif
              OPEN(UNIT=23,FILE=DIR_APP(1:DIR_LEN)//'vasp.lock',STATUS='OLD',IOSTAT=ISTAT)
#ifdef MPI
           ENDIF
#endif
           ! "propagate" error status to all nodes
           CALLMPI( M_sum_i(MYCOMM, ISTAT,1))
           IF (ISTAT==0) EXIT
           ! all nodes sleep for 2 seconds
           CALL SLEEP(2)
           ! all nodes wait for root node
           CALLMPI( M_barrier(MYCOMM))
        ENDDO

        CLOSE(UNIT=23)

      END SUBROUTINE WAIT_VASP_LOCK


      SUBROUTINE REMOVE_VASP_LOCK(MYCOMM)
        USE mpimy
        USE main_mpi
        TYPE (communic) :: MYCOMM

      ! local
        INTEGER :: ISTAT
#ifdef MPI
        IF (MYCOMM%NODE_ME==MYCOMM%IONODE) THEN
#endif
           OPEN(UNIT=23,FILE=DIR_APP(1:DIR_LEN)//'vasp.lock',STATUS='UNKNOWN')
           WRITE(23,*) 'file will be removed'
           CLOSE(UNIT=23,STATUS='DELETE')
#ifdef MPI
        ENDIF
#endif
        ! all nodes sleep for 2 seconds to be sure NFS has propagated this change
        CALL SLEEP(2)
        ! all nodes wait for root node
        CALLMPI( M_barrier(MYCOMM))

      END SUBROUTINE REMOVE_VASP_LOCK


