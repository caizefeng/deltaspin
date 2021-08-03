#include "symbol.inc"
!------------------------------------------------------------------------
!
! MODULE PARDENS
!
! This module calculates partial charge density, parameters
! are read from file "INCAR"
!
! original it was a part of the main program of vasp.3.2, written by
! Juergen Furthmueller.
!
! This module is a completly rewritten and modified version, but the
! basics are still the same.
!
! author: Karin E. Seifert-Lorenz
! date:   97-11-26
!
!------------------------------------------------------------------------

MODULE pardens
  USE prec
  USE base
  USE us
  USE charge
  USE pseudo
  USE poscar
  USE mpimy
  USE mgrid
  USE lattice
  USE wave
  USE fileio

CONTAINS
  SUBROUTINE parchg(w,wup,wdw,wdes,chden,chtot,crhode,info,grid,grid_soft, &
       gridc,gridus,c_to_us,latt_cur,p,t_info,soft_to_c, &
       symm,io,dyn,efermi,lmdim,irdmax,niond)


! all the input parameters, chtot and chden are used for the calculations and
! to store the partial charge density

    IMPLICIT NONE
    REAL(q), INTENT (IN)    :: efermi     ! Fermi energy
    INTEGER, INTENT (IN)    :: lmdim      ! no of real space proj.
    INTEGER, INTENT (IN)    :: irdmax
    INTEGER, INTENT (IN)    :: niond      ! number of ions ?

    TYPE (wavespin)    :: w          ! wavefunctions
    TYPE (wavefun)     :: wup,wdw    ! wavefunctions spin up/down
    TYPE (wavedes)     :: wdes       ! description of the wavefuncitons
    TYPE (grid_3d), INTENT (IN)  :: grid   ! grid for wavefunctions
    TYPE (grid_3d), INTENT (IN)  :: grid_soft  ! grid for soft-chargedensity
    TYPE (grid_3d), INTENT (IN)  :: gridc      ! grid for potential / charge
    TYPE (grid_3d), INTENT (IN)  :: gridus

    COMPLEX (q) chden(GRID_SOFT%MPLWV,WDES%ISPIN)  ! charge density for spin up/down
    COMPLEX (q) chtot(GRIDC%MPLWV,WDES%ISPIN) ! total charge density
    OVERLAP    crhode(LMDIM,LMDIM,WDES%NIONS,WDES%ISPIN)  ! augmentation occupations
    TYPE (latt), INTENT (IN) :: latt_cur   ! current lattice real and rez.
    TYPE (potcar), INTENT (IN)   :: p(:)          ! potential
    TYPE (type_info), INTENT (IN) :: t_info    ! infos about the atoms
    TYPE (transit), INTENT (IN)  :: soft_to_c  ! indextable: grid_soft <-> gridc
    TYPE (transit), INTENT (IN)  :: c_to_us
    TYPE (symmetry), INTENT (IN) :: symm       ! symmetrie stuff
    TYPE (in_struct), INTENT (IN) :: io        ! in/out streams
    TYPE (dynamics), INTENT (IN)  :: dyn       ! information about
    TYPE (info_struct), INTENT (IN) :: info    ! information about the whole run
#ifndef noPAW

! now the locals:
    REAL(q), DIMENSION (2)         :: eint    ! energy intervall
    REAL(q)                         :: eigv   ! eigenvalue
    REAL(q)                         :: eigvgil ! eigenvalue
    INTEGER                         :: nbmodus ! band-modus for calc.
                                               ! of partial chden
    INTEGER                          :: nkpoint ! no of k-points used
    INTEGER                          :: nband   ! no of bands used
    INTEGER, DIMENSION (wdes%nb_tot) :: nbuse   ! the band indexs
    INTEGER, DIMENSION (wdes%nkpts)  :: kpuse   ! the used k-points
    INTEGER                         :: sepmod   ! is calculated out of lsepk and lsepb (0,...,3)
    INTEGER                         :: iupar   ! input unit
    INTEGER                         :: iuout   ! output unit
    INTEGER                         :: iueigv   ! outputunit for EIGENVAL file
    INTEGER                         :: iuchg, iuchgcar ! output for total chg
    INTEGER                         :: ikpu
    LOGICAL                         :: lsepk,lsepb    ! calc. parchg for each
                                               ! k-point or not

    LOGICAL                         :: lint    ! eigenvalue in the energy range ?
    LOGICAL                         :: lopchg,lfound ! logicals for FILE I/O status
    LOGICAL                         :: lop6, lop0  ! output devices open

    LOGICAL                         :: ltest    ! this is only for testing
    LOGICAL                         :: lkptincld
    INTEGER                         :: ierr     ! error status for open file
    CHARACTER (*), PARAMETER        :: filein = "INCAR"
    CHARACTER (LEN=16)              :: fileout ! file name for output
      
#ifdef MPI
    INTEGER   IONODE,NODE_ME
#endif

! counters and such stuff
    INTEGER  :: i,nk, nb, ik, ib, isp, ni

    INTEGER :: TIU6, TIU0
    INTEGER :: ISPIN_SELECT=0

! externals:

    INTEGER, EXTERNAL :: NXTFRU
! mpi version requires to know which node does the io
#ifdef MPI
      IONODE  = WDES%COMM%IONODE
      NODE_ME = WDES%COMM%NODE_ME
      IF (WDES%COMM_KINTER%NCPU.NE.1) THEN
         CALLMPI( M_stop('PARCHG: KPAR>1 not implemented, sorry.') )
         STOP
      END IF
#endif

! ltest is just for testing !!
! so only if you want to have lots of output of  this SR set it to TRUE

    lop6 = (io%iu6 >= 0)
    lop0 = (io%iu0 >= 0)
    TIU6 = io%iu6
    TIU0 = io%iu0

    ltest = .TRUE.
! now get started: first information

    IF (lop6) &
         WRITE(TIU6,'(A)') "Calculating partial charge density."
    IF (lop0) &
         WRITE(TIU0,'(A)') "Calculating partial charge density."

! read the parameters

    CALL read_pard()


! write out the parameters, if you want to do that

    ltest1: IF (ltest) THEN
       IF (lop6) THEN
          WRITE(TIU6,'(/A)') "*****************************************************"
          WRITE(TIU6,'(A)') "* Parameters from pardens"
           WRITE(TIU6,'(A/)') "*****************************************************"

          SELECT CASE (nbmodus)
             CASE (1:)  ! nbmodus > 0: IBAND given
                WRITE(TIU6,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU6,FMT = "(A)",ADVANCE = "NO") "Selected bands: "
                WRITE(TIU6,'(5(1X,I4))') (nbuse(i),i=1,nband)

             CASE (0)  ! nbmodus = 0: all bands
                WRITE(TIU6,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU6,'(A,I4,A)') "All ",wdes%nb_tot," bands are selected."

             CASE (-1) ! nbmodus = -1: total charge density
                WRITE(TIU6,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU6,'(A)') "Calculate total charge density."

             CASE (-2,-3) ! nbmodus = -2 or -3: select bands via energy range
                WRITE(TIU6,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU6,'(A,2F9.4)') "Selected energy range (EINT) ",eint

             CASE DEFAULT
                WRITE(TIU6,'(A)') "VALUE NOT KNOWN, SO I WILL STOP"
                CALL WFORCE(io%iu6)
                STOP

             END SELECT

             IF (nkpoint == wdes%nkpts) THEN
                WRITE(TIU6,'(A)') "Selected all k-points to calculate charge density."
                DO i=1,nkpoint
                   kpuse(i)=i
                ENDDO
             ELSE
                WRITE(TIU6,'(A,I5,A)') "Selected ",nkpoint," kpoints to calculate charge density."
                WRITE(TIU6,FMT = "(A)",ADVANCE = "NO") "Selected k-points: "
                WRITE(TIU6,'(5(1X,I4))') (kpuse(i),i=1,nkpoint)
             END IF

             IF (lsepb) WRITE(TIU6,'(A)') "Seperate output for each band."
             IF (lsepk) WRITE(TIU6,'(A)') "Seperate output for each kpoint."

          END IF

          IF(lop0) THEN
             WRITE(TIU0,'(/A)') "*****************************************************"
             WRITE(TIU0,'(A)') "* Parameters from pardens"
             WRITE(TIU0,'(A/)') "*****************************************************"

          SELECT CASE (nbmodus)
             CASE (1:)  ! nbmodus > 0: IBAND given
                WRITE(TIU0,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU0,FMT = "(A)",ADVANCE = "NO") "Selected bands: "
                WRITE(TIU0,'(5(1X,I4))') (nbuse(i),i=1,nband)

             CASE (0)  ! nbmodus = 0: all bands
                WRITE(TIU0,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU0,'(A,I4,A)') "All ",wdes%nb_tot," bands are selected."

             CASE (-1) ! nbmodus = -1: total charge density
                WRITE(TIU0,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU0,'(A)') "Calculate total charge density."

             CASE (-2,-3) ! nbmodus = -2 or -3: select bands via energy range
                WRITE(TIU0,'(A,I3)') "NBMOD is set to ",nbmodus
                WRITE(TIU0,'(A,2F9.4)') "Selected energy range (EINT) ",eint

             CASE DEFAULT
                WRITE(TIU0,'(A)') "VALUE NOT KNOWN, SO I WILL STOP"
                CALL WFORCE(io%iu0)
                STOP

             END SELECT

             IF (nkpoint == wdes%nkpts) THEN
                WRITE(TIU0,'(A)') "Selected all k-points to calculate charge density."
             ELSE
                WRITE(TIU0,'(A,I5,A)') "Selected ",nkpoint," kpoints to calculate charge density."
                WRITE(TIU0,FMT = "(A)",ADVANCE = "NO") "Selected k-points: "
                WRITE(TIU0,'(5(1X,I4))') (kpuse(i),i=1,nkpoint)
             END IF

             IF (lsepb) WRITE(TIU0,'(A)') "Seperate output for each band."
             IF (lsepk) WRITE(TIU0,'(A)') "Seperate output for each kpoint."

          END IF
       END IF ltest1


! OK, everthing all right, first calculate right occupancies for
! augmentation charge, saved in CHRODE (that is out)

!    CALL DEPSUM(w,wdes,lmdim,crhode,info%loverl)

! now set the fermi weights to get partial charge densities for
! specified bands and kpoints

! if nbmodus = -1: just calculate total charge density, and stop

! for testing: allways calculate total charge density
!    total: IF (nbmodus == -1) THEN

       IF (ltest) THEN
          CALL write_eigv()
!          CLOSE(iueigv)
       END IF

       CALL calc_pard()

! OK, normal (total) charge density calculated =>  write everything to
! normal output: CHGCAR and CHG

! first make a proper starting status for the files CHG and CHGCAR
       io_begin
       
       INQUIRE(FILE="CHG",OPENED=lopchg,NUMBER=iuchg)

       IF (lopchg) THEN
          CALL CLEAN(iuchg)
       ELSE
          iuchg=NXTFRU()

          IF ((iuchg <= -1) .OR. (iuchg >= 99)) THEN  ! no I/O unit free
             CALL err_handle(io%iu6,1,"CHG","NONE","U")
             CALL err_handle(io%iu0,1,"CHG","NONE","U")
             IF (io%lopen) CALL WFORCE(io%iu6)
             STOP   ! on lack of output: stop execution
          END IF

          OPEN(UNIT=iuchg,FILE="CHG",IOSTAT=ierr,STATUS="REPLACE")
       END IF

       INQUIRE(FILE="CHGCAR",OPENED=lopchg,NUMBER=iuchgcar)

       IF (lopchg) THEN
          CALL CLEAN(iuchgcar)
       ELSE
          iuchgcar=NXTFRU()

          IF ((iuchgcar <= -1) .OR. (iuchgcar >= 99)) THEN  ! no I/O unit free
             CALL err_handle(io%iu6,1,"CHGCAR","NONE","U")
             CALL err_handle(io%iu0,1,"CHGCAR","NONE","U")
             IF (io%lopen) CALL WFORCE(io%iu6)
             STOP   ! on lack of output: stop execution
          END IF

          OPEN(UNIT=iuchgcar,FILE="CHGCAR",IOSTAT=ierr,STATUS="REPLACE")
       END IF

       io_end

! now we take just the usual output routines
! I do not call <<write_pard>> here, because I handle the total charge a little bit
! differnt, and the output to <<CHG>> is also differnt
       do_io CALL OUTPOS(iuchgcar,.FALSE.,INFO%SZNAM1,T_INFO,LATT_CUR%SCALE,LATT_CUR%A,.FALSE.,DYN%POSION)

       CALL OUTCHG(GRIDC,iuchgcar,.TRUE.,CHTOT)

       IF (INFO%ISPIN==2) THEN
          do_io WRITE(iuchgcar,'(5E20.12)') (T_INFO%ATOMOM(NI),NI=1,T_INFO%NIONS)
          CALL OUTCHG(GRIDC,iuchgcar,.TRUE.,CHTOT(1,2))
       ENDIF

       do_io CLOSE(iuchgcar)
       do_io CALL OUTPOS(iuchg,.FALSE.,INFO%SZNAM1,T_INFO,LATT_CUR%SCALE,LATT_CUR%A,.FALSE.,DYN%POSION)

       CALL OUTCHG(GRIDC,iuchg,.FALSE.,CHTOT)

       IF (INFO%ISPIN==2) CALL OUTCHG(GRIDC,iuchg,.FALSE.,CHTOT(1,INFO%ISPIN))

       do_io CLOSE(iuchg)

!    ELSE total          ! some kind of partial charge density calculat

       total: IF (nbmodus /= -1) THEN

! before starting the whole stuff: search for a free I/O unit:
      io_begin

       iuout = NXTFRU()

       IF ((iuout <= -1) .OR. (iuout >= 99) ) THEN
          CALL err_handle(io%iu6,1,"PARCHG.nb.nk","NONE","U")
          CALL err_handle(io%iu0,1,"PARCHG.nb.nk","NONE","U")
          IF (io%lopen) CALL WFORCE(io%iu6)
          STOP   ! on lack of output: stop execution
       END IF

      io_end
! there are 4 differnt cases:
!                                         sepmod         output
! seperate just everything:                  0           PARCHG.$nb.$nk
! seperate bands, but merge kpoints:         1           PARCHG.$nb.ALLK
! merge bands, but seperate kpoints:         2           PARCHG.ALLB.$nk
! merge everything:                          3           PARCHG

       sep: SELECT CASE (sepmod)

       CASE (0) sep  ! seperate everything

          kpoints0: DO ik=1,nkpoint      ! loop over all selected kpoints
             nk=kpuse(ik)

             IF (nbmodus <= -2) THEN
                nband = 0
                DO ib=1,wdes%nb_tot
                   lint = .FALSE.
                   DO isp=1,info%ispin
                      eigv = w%celtot(ib,nk,isp)
                      IF ((eint(1) <=  eigv) .AND. &
                           (eigv <= eint(2))) lint = .TRUE.
                   END DO
                   IF (lint) THEN
                      nband = nband + 1
                      nbuse(nband) = ib
                   END IF
                END DO
             END IF

             IF (ltest) THEN
                IF(lop6) THEN
                   WRITE(TIU6,'(A,I4,A,I4)') "Kpoint no.: ",nk,"; bands ",nband
                   WRITE(TIU6,'(10I4)') (nbuse(i),i=1,nband)
                END IF
                IF(lop0) THEN
                   WRITE(TIU0,'(A,I4,A,I4)') "Kpoint no.: ",nk,"; bands ",nband
                   WRITE(TIU0,'(10I4)') (nbuse(i),i=1,nband)
                END IF
             END IF
             bands: DO ib=1,nband

                ! --- first init everything
                ! --- weights for wup and wdw is pointer to weights of w

                nb = nbuse(ib)

                DO isp = 1,info%ispin
                   w%fertot(:,:,isp) = 0.
                END DO

                wdes%wtkpt = 0.
                wdes%wtkpt(nk) = 1.0

                ! --- now set the correct the one kpoint and band
                ! set the fermi weights

                DO isp=1,info%ispin
                   w%fertot(nb,nk,isp) = 1.0
                END DO

                ! --- That is it: calcualte charge density as usual

                IF (ltest) CALL write_eigv()
                CALL calc_pard()

                ! --- now output to the file PARCHG.$nb.$nk

                ! first generate filename

                WRITE(fileout,'(A7,I4.4,A1,I4.4)') "PARCHG.",nb,".",nk

                ! this file should not be open, but for safty reasons:

                CALL write_pard(fileout,iuout)

             END DO bands

          END DO kpoints0

!          IF (ltest) CLOSE(iueigv)

       CASE (1) sep     ! seperate bands but not kpoints


          ! the easier way: the bands are given

          IF (nbmodus >= 0) THEN

             DO ib=1,nband
                nb=nbuse(ib)

                DO isp = 1,info%ispin
                   w%fertot(:,:,isp) = 0.
                END DO

                DO isp=1,info%ispin
                    DO ikpu=1,nkpoint
                       w%fertot(nb,kpuse(ikpu),isp) = 1.
                    ENDDO
                ENDDO

                IF (ltest) THEN
                   IF(lop6) WRITE(TIU6,'(A,I5)') "All kpoints, Band no. ",nb
                   IF(lop0) WRITE(TIU0,'(A,I5)') "All kpoints, Band no. ",nb
                END IF

                ! now calculate charge density for this one band
                IF  (ltest) CALL write_eigv()

                CALL calc_pard()

                ! and write to file PARCHG.$nb,ALLK

                WRITE(fileout,'(A7,I4.4,A5)') "PARCHG.",nb,".ALLK"

                CALL write_pard(fileout,iuout)

             END DO

          ELSE ! now that is a little bit more difficult, because of dispersion

             bands1: DO ib=1,wdes%nb_tot

                DO isp = 1,info%ispin
                   w%fertot(:,:,isp) = 0.
                END DO


                lint = .FALSE.

                kpoints1: DO ik=1,nkpoint

                   nk = kpuse(ik)

                   DO isp=1,info%ispin
                      eigv = w%celtot(ib,nk,isp)
                      IF ((eint(1) <= eigv) .AND. &
                           (eigv <= eint(2))) THEN
                         lint = .TRUE.
                         w%fertot(ib,nk,isp) = 1.0
                      END IF

                   END DO
                END DO kpoints1

                ! calculate charge density only if band is in the energy range
                IF (lint) THEN

                   IF(ltest)THEN
                      IF(lop6) WRITE(TIU6,'(A,I4)') &
                           "All kpoints, inside intervall Band no. ",ib
                      IF(lop0) WRITE(TIU0,'(A,I4)') &
                           "All kpoints, inside intervall Band no. ",ib
                   END IF

                   IF (ltest) CALL write_eigv()

                   CALL calc_pard()

                   ! and write to file "PARCHG.nb.ALLK"

                   WRITE(fileout,'(A7,I4.4,A5)') "PARCHG.",ib,".ALLK"

                   CALL write_pard(fileout,iuout)

                END IF

             END DO bands1

          END IF

!          IF (ltest) CLOSE(iueigv)

       CASE (2) sep     ! seperate k-points and merge bands


          kpoint2: DO ik=1,nkpoint

             nk=kpuse(ik)

             DO isp = 1,info%ispin
                w%fertot(:,:,isp) = 0.
             END DO

             wdes%wtkpt = 0.
             wdes%wtkpt(nk) = 1.


             ! find out the bands in the right energy range for nbmodus <= -2

             IF (nbmodus <= -2) THEN
                nband = 0
                DO ib=1,wdes%nb_tot
                   lint = .FALSE.
                   DO isp=1,info%ispin
                      eigv = w%celtot(ib,nk,isp)
                      IF ((eint(1) <= eigv) .AND. (eigv <= eint(2))) &
                           lint = .TRUE.
                   END DO
                   IF (lint) THEN
                      nband = nband + 1
                      nbuse(nband) = ib
                   END IF
                END DO
             END IF

             IF (ltest) THEN
                IF (lop6) THEN
                   WRITE(TIU6,'(A,I4,A,I4)') &
                        "k-point no. ",nk,"; bands inside the range ",nband
                   WRITE(TIU6,'(10I4)') (nbuse(i),i=1,nband)
                END IF
                IF (lop0) THEN
                   WRITE(TIU0,'(A,I4,A,I4)') &
                        "k-point no. ",nk,"; bands inside the range ",nband
                   WRITE(TIU0,'(10I4)') (nbuse(i),i=1,nband)
                END IF
             END IF

             ! now the loop over all bands

             DO ib=1,nband
                nb=nbuse(ib)

                DO isp=1,info%ispin
                   w%fertot(nb,nk,isp) = 1.0
                END DO
             END DO

             ! now calculate charge density
             IF (ltest) CALL write_eigv()

             CALL calc_pard()

             ! and write it to the file PARCHG.ALLB.$nk

             WRITE(fileout,'(A12,I4.4)') "PARCHG.ALLB.",nk

             CALL write_pard(fileout,iuout)

          END DO kpoint2

!          IF (ltest) CLOSE(iueigv)


       CASE (3)  sep    ! no seperation at all: merge everything

          ! as usual: first init, but only fermi weights, not the weights for k-points
          ! and do it at the beginning and only once

          DO isp=1,info%ispin
             w%fertot(:,:,isp) = 0.
          END DO

          ! using all kpoints: the weights will remain as given, but
          ! using only one part of them, they all will be equall weighted

          IF( nkpoint < wdes%nkpts) THEN
             IF(lop6) THEN
                WRITE(TIU6,'(A)') &
              "Calculating partial charge density only with a part of the kpoint set."
                WRITE(TIU6,'(A)') "This may change the weights of the k-points."
                WRITE(TIU6,'(A)') "So I hope you know, what you are doing."
             END IF
             IF(lop0) THEN
                WRITE(TIU0,'(A)') &
              "Calculating partial charge density only with a part of the kpoint set."
                WRITE(TIU0,'(A)') "This may change the weights of the k-points."
                WRITE(TIU0,'(A)') "So I hope you know, what you are doing."
             END IF
             wdes%wtkpt = 0.

             DO ik=1,nkpoint
                wdes%wtkpt(kpuse(ik)) = 1.
             END DO

             wdes%wtkpt = wdes%wtkpt/MAX(SUM(wdes%wtkpt),1._q)
          END IF

          kpoint3: DO ik=1,nkpoint

             nk=kpuse(ik)

             ! find out the bands in the right energy range for nbmodus <= -2

             IF (nbmodus <= -2) THEN
                nband = 0
                DO ib=1,wdes%nb_tot
                   lint = .FALSE.
                   DO isp=1,info%ispin
                      eigv = w%celtot(ib,nk,isp)
                      IF ((eint(1) <= eigv) .AND. (eigv <= eint(2))) &
                           lint = .TRUE.
                   END DO
                   IF (lint) THEN
                      nband = nband + 1
                      nbuse(nband) = ib
                   END IF
                END DO
             END IF

             IF (ltest) THEN
                IF (lop6) THEN
                   WRITE(TIU6,'(A,I4,A,I4)') &
                        "k-point no. ",nk,"; bands inside the range ",nband
                   WRITE(TIU6,'(10I4)') (nbuse(i),i=1,nband)
                END IF
                IF (lop0) THEN
                   WRITE(TIU0,'(A,I4,A,I4)') &
                        "k-point no. ",nk,"; bands inside the range ",nband
                   WRITE(TIU0,'(10I4)') (nbuse(i),i=1,nband)
                END IF
             END IF

             ! now the loop over all bands

             DO ib=1,nband
                nb=nbuse(ib)
                ! modification by Gilles de Wijs
                ! to me that seems to make sense
                IF (nbmodus <= -2) THEN
                DO isp=1,info%ispin
                   eigvgil = w%celtot(nb,nk,isp)
                   IF ((eint(1) <= eigvgil) .AND. (eigvgil <= eint(2))) THEN
                     w%fertot(nb,nk,isp) = 1.0
                   ELSE
                   ENDIF
                ENDDO
                ELSE
                DO isp=1,info%ispin
                   w%fertot(nb,nk,isp) = 1.0
                END DO
                ENDIF
             END DO

          END DO kpoint3

          ! now calculate the charge density as usual

          IF ((ISPIN_SELECT==1.OR. ISPIN_SELECT==2) .AND. WDES%ISPIN==2) THEN
             IF (IO%IU0>=0) THEN
                WRITE(*,*) 'selecting spin channel ',ISPIN_SELECT
                WRITE(*,*) 'spin channel ',3-ISPIN_SELECT,' set to zero'
             ENDIF
             w%fertot(:,:,3-ISPIN_SELECT) = 0.0
          ENDIF


          IF (ltest) THEN
             CALL write_eigv()
!             CLOSE (iueigv)
          END IF

          CALL calc_pard()

          IF ((ISPIN_SELECT==1.OR. ISPIN_SELECT==2) .AND. WDES%ISPIN==2) THEN
             ! back to spin up/down representation
             CALL RC_FLIP(CHTOT,GRIDC,INFO%ISPIN,.TRUE.)
             ! copy charge density to second channel
             CHTOT(:,3-ISPIN_SELECT)=CHTOT(:,ISPIN_SELECT)
          ENDIF

          CALL write_pard("PARCHG",iuout)

       END SELECT sep
    END IF total

    ! as this is a tool to analyse a allready cenverged system: STOP the
    ! the calculation: sure thats not a very proper manner, but it is a
    ! quick solution, and the module is distributed with VASP

    IF(lop6) THEN
       WRITE(TIU6,'(A)') "Finished calculating partial charge density."
       WRITE(TIU6,'(A)') "VASP will stop now."
       IF (io%lopen) CALL WFORCE(io%iu6)
    END IF
    IF(lop0) THEN
       WRITE(TIU0,'(A)') "Finished calculating partial charge density."
       WRITE(TIU0,'(A)') "VASP will stop now."
    END IF

    STOP

  CONTAINS
!
! SUBROUTINE read_pard(): reades parameters for calculating of the
!                         partial charge densitie, using reader
!                         written by jF.
! Makes also some checks and set variables to default values, on lack of
! information.
!
    SUBROUTINE read_pard()

! locals: dummies for jF-reader (no optional parameters yet)
      REAL                         :: ehelp

      REAL (q), DIMENSION (1)    :: rdummy
      COMPLEX (q), DIMENSION (1) :: cdummy
      INTEGER, DIMENSION (1)       :: idummy
      LOGICAL, DIMENSION (1)       :: ldummy
      CHARACTER (LEN=1)            :: strdummy

      INTEGER                      :: ndat ! data really found
! error code returned
      INTEGER                      :: i,ierr1,ierr2,ierr3
! logicals for proving file status

      LOGICAL  :: lexist,lopen

! first get status of the input file:

      lexist = .FALSE.
      lopen = .FALSE.

      INQUIRE(FILE=filein,EXIST=lexist,OPENED=lopen,NUMBER=iupar)

      IF (.NOT.lexist) THEN
         IF (lop6) WRITE(TIU6,'(A,A)') "Can't open input file ",filein
         IF (lop0) WRITE(TIU0,'(A,A)') "Can't open input file ",filein
         IF (io%lopen) CALL WFORCE(io%iu6)
         STOP  ! Sorry: but input file does not exist
      END IF

      IF (.NOT.lopen) THEN  ! the file exists put one has to open it
         iupar = NXTFRU()
         IF (iupar < 0 .OR. iupar > 99) THEN
            IF(lop6) WRITE(TIU6,'(A)') "No IO unit free. VASP will stop."
            IF(lop0) WRITE(TIU0,'(A)') "No IO unit free. VASP will stop."
            STOP
         END IF

         IF (lop6) WRITE(TIU6,'(A,I3,1X,A,A)') "Open file ",iupar,filein," for input."
         IF (lop0) WRITE(TIU0,'(A,I3,1X,A,A)') "Open file ",iupar,filein," for input."
         OPEN(UNIT=iupar,FILE=filein,STATUS='OLD',IOSTAT=ierr)

      ELSE  ! file is allready open
         IF (lop6) WRITE(TIU6,'(A,I3,1X,A,A)') "File ",iupar,filein," allready open."
         IF (lop0) WRITE(TIU0,'(A,I3,1X,A,A)') "File ",iupar,filein," allready open."
      END IF


! try to find band index
      CALL rdatab(.FALSE.," ",iupar,"IBAND","=","!",";","I", &
           nbuse,rdummy,cdummy,ldummy,strdummy,ndat,wdes%nb_tot,ierr1)

      iband: SELECT CASE (ierr1)
      CASE (0)        ! IBAND found

         nbmodus = MIN(ndat,wdes%nb_tot)
         nband = nbmodus
         ! read nbmodus and look for a consitents
         CALL rdatab(.FALSE.," ",iupar,"NBMOD","=","!",";","I", &
              idummy,rdummy,cdummy,ldummy,strdummy,ndat,1,ierr2)

         err_nbmod1: SELECT CASE (ierr2)

         CASE(0) err_nbmod1
            IF (idummy(1) /=  nbmodus) THEN
               IF(lop6) THEN
                  WRITE(TIU6,'(A,I3,A)') &
                       "NBMOD (",idummy(1)," not consitent with IBAND "
                  WRITE(TIU6,'(A)') "NBMOD set to ",nbmodus
               END IF
               IF(lop0) THEN
                  WRITE(TIU0,'(A,I3,A)') &
                       "NBMOD (",idummy(1)," not consitent with IBAND "
                  WRITE(TIU0,'(A)') "NBMOD set to ",nbmodus
               END IF
            END IF

         CASE(3) err_nbmod1
            IF(lop6) &
                 WRITE(TIU6,'(A,I3)') "NBMOD not found, set to ",nbmodus
            IF(lop0) &
                 WRITE(TIU0,'(A,I3)') "NBMOD not found, set to ",nbmodus

         CASE DEFAULT err_nbmod1
            CALL err_handle(io%iu6,ierr3,filein,"NBMOD","I")
            CALL err_handle(io%iu0,ierr3,filein,"NBMOD","I")
            IF (io%lopen) CALL WFORCE(io%iu6)
            STOP ! ERROR !!!!!!!!!!!
         END SELECT err_nbmod1

      CASE (3)        ! IBAND not found, search for other information

         CALL rdatab(.FALSE.," ",iupar,"NBMOD","=","!",";","I", &
              idummy,rdummy,cdummy,ldummy,strdummy,ndat,1,ierr2)


         err_nbmod: SELECT CASE (ierr2)

         CASE (0)  ! nbmod found

            nbmod: SELECT CASE (idummy(1))
            CASE (-3) nbmod      ! read eint vs. fermi energy
               nbmodus = -3
               CALL rdatab(.FALSE.," ",iupar,"EINT","=","!",";","F", &
                    idummy,eint,cdummy,ldummy,strdummy,ndat,2,ierr3)


               eint1: SELECT CASE (ierr3)
               CASE (3) eint1
                  IF(lop6) &
                       WRITE(TIU6,'(A,A,A)') "EINT not found in ",filein, &
                       "Set nbmod to -1."
                  IF(lop0) &
                       WRITE(TIU0,'(A,A,A)') "EINT not found in ",filein, &
                       "Set nbmod to -1 (total charge)."
                  nbmodus = -1

               CASE (0) eint1 ! eint found

                  ! first proff: are there two values ?

                  IF (ndat == 0) THEN
                     IF(lop6) &
                          WRITE(TIU6,'(A,A,A)') "No data found for EINT in ",filein, &
                          "Set nbmod to -1."
                     IF(lop0) &
                          WRITE(TIU0,'(A,A,A)') "No data found for EINT in ",filein, &
                          "Set nbmod to -1 (total charge)."
                     nbmodus = -1
                  ELSE
                     IF (ndat == 1) THEN
                        ndat = 2
                        eint(2) = 0.0
                     END IF

                     IF(lop6) THEN
                        WRITE(TIU6,'(A,I3)') "NDMOD = ",nbmodus
                        WRITE(TIU6,'(A,A,A,2F8.3)') &
                             "Energy range vs. Fermi energy from ",filein,": ", &
                             eint
                        WRITE(TIU6,'(A,F8.3)') "Efermi = ",efermi
                     END IF
                     IF(lop0) THEN
                        WRITE(TIU0,'(A,I3)') "NDMOD = ",nbmodus
                        WRITE(TIU0,'(A,A,A,2F8.3)') &
                             "Energy range vs. Fermi energy from ",filein,": ", &
                             eint
                        WRITE(TIU0,'(A,F8.3)') "Efermi = ",efermi
                     END IF
                     eint = eint + efermi
                  END IF

               CASE DEFAULT eint1
                  CALL err_handle(io%iu6,ierr3,filein,"EINT","F")
                  CALL err_handle(io%iu0,ierr3,filein,"EINT","F")
                  IF (io%lopen) CALL WFORCE(io%iu6)
                  STOP ! ERROR !!!!!!!!!!!!!!
               END SELECT eint1

            CASE (-2) nbmod     ! read eint
               nbmodus = idummy(1)
               CALL rdatab(.FALSE.," ",iupar,"EINT","=","!",";","F", &
                    idummy,eint,cdummy,ldummy,strdummy,ndat,2,ierr3)

               eint2: SELECT CASE (ierr3)
               CASE (3) eint2
                  IF(lop6) &
                       WRITE(TIU6,'(A,A,A)') "EINT not found in ",filein, &
                       ". Set nbmod to -1 (total charge)."
                  IF(lop0) &
                       WRITE(TIU0,'(A,A,A)') "EINT not found in ",filein, &
                       ". Set nbmod to -1 (total charge)."
                  nbmodus = -1

               CASE (0) eint2

                  IF (ndat == 0) THEN
                     IF(lop6) &
                          WRITE(TIU6,'(A,A,A)') "No data found for EINT in ",filein, &
                          "Set nbmod to -1."
                     IF(lop0) &
                          WRITE(TIU0,'(A,A,A)') "No data found for EINT in ",filein, &
                          "Set nbmod to -1 (total charge)."
                     nbmodus = -1
                  ELSE
                     IF (ndat == 1) THEN
                        ndat = 2
                        eint(2) = efermi
                     END IF

                     IF(lop6) THEN
                        WRITE(TIU6,'(A,I3)') "NDMOD = ",nbmodus
                        WRITE(TIU6,'(A,A,A,2F8.3)') &
                             "Energy range energy from ",filein,": ",eint
                     END IF
                     IF(lop0) THEN
                        WRITE(TIU0,'(A,I3)') "NDMOD = ",nbmodus
                        WRITE(TIU0,'(A,A,A,2F8.3)') &
                             "Energy range energy from ",filein,": ",eint
                     END IF
                  END IF

               CASE DEFAULT eint2
                  CALL err_handle(io%iu6,ierr3,filein,"EINT","F")
                  CALL err_handle(io%iu0,ierr3,filein,"EINT","F")
                  IF (io%lopen) CALL WFORCE(io%iu6)
                  STOP ! ERROR !!!!!!!!!!
               END SELECT eint2

            CASE (-1) nbmod ! nbmodus: calculate total charge density
               nbmodus = -1
               IF(lop6) &
                    WRITE(TIU6,'(A,I3)') "NBMOD = ",nbmodus
               IF(lop0) &
                    WRITE(TIU0,'(A,I3)') "NBMOD = ",nbmodus

            CASE (0) nbmod ! take all bands: short for iband = 1,2,...,nb_tot
               nbmodus = wdes%nb_tot
               nband = nbmodus
               DO i=1,nbmodus
                  nbuse(i) = i
               ENDDO

               IF(lop6) THEN
                  WRITE(TIU6,'(A)') "Take all bands"
                    WRITE(TIU6,'(A,I3)') "NBMOD = ",nbmodus
                 END IF
               IF(lop0) THEN
                  WRITE(TIU0,'(A,I3)') "Take all bands"
                  WRITE(TIU0,'(A,I3)') "NBMOD = ",nbmodus
               END IF

            CASE (1:) nbmod ! it is the form of iband, no iband found
               IF(lop6) THEN
                  WRITE(TIU6,'(A,I3)') "NBMOD = ",nbmodus
                  WRITE(TIU6,'(A)') "But IBAND not found."
                  WRITE(TIU6,'(A)') "NBMOD set to -1 (total charge)."
               END IF
               IF(lop0) THEN
                  WRITE(TIU0,'(A,I3)') "NDMOD = ",nbmodus
                  WRITE(TIU0,'(A)') "But IBAND not found."
                  WRITE(TIU0,'(A)') "NBMOD set to -1 (total charge)."
               END IF

               nbmodus = -1
            END SELECT nbmod

         CASE (3)  err_nbmod ! nbmod not found: read eint
            CALL rdatab(.FALSE.," ",iupar,"EINT","=","!",";","F", &
                 idummy,eint,cdummy,ldummy,strdummy,ndat,2,ierr3)

            eint3: SELECT CASE (ierr3)

            CASE (3) eint3
               IF(lop6) &
                    WRITE(TIU6,'(A,A,A)') "EINT not found in ",filein, &
                    ". Set nbmod to -1."
               IF(lop0) &
                    WRITE(TIU0,'(A,A,A)') "EINT not found in ",filein, &
                    ". Set nbmod to -1 (total charge)."

               nbmodus = -1

            CASE (0) eint3
               nbmodus = -2
               IF(lop6) THEN
                  WRITE(TIU6,'(A,I3)') "NDMOD set to ",nbmodus
                  WRITE(TIU6,'(A,A,A,2F8.3)') &
                       "Energy range energy from ",filein,": ",eint
               END IF
               IF(lop0) THEN
                  WRITE(TIU0,'(A,I3)') "NDMOD = ",nbmodus
                  WRITE(TIU0,'(A,A,A,2F8.3)') &
                       "Energy range energy from ",filein,": ",eint
               END IF

            CASE DEFAULT eint3

               CALL err_handle(io%iu6,ierr3,filein,"EINT","F")
               CALL err_handle(io%iu0,ierr3,filein,"EINT","F")
               IF (io%lopen) CALL WFORCE(io%iu6)
               STOP ! ERROR !!!!!!!!!
            END SELECT eint3

         CASE DEFAULT err_nbmod

            CALL err_handle(io%iu6,ierr3,filein,"NBMOD","I")
            CALL err_handle(io%iu0,ierr3,filein,"NBMOD","I")
            IF (io%lopen) CALL WFORCE(io%iu6)
            STOP ! error !!!!!!!!!!!!!
         END SELECT err_nbmod

      CASE DEFAULT iband

         CALL err_handle(io%iu6,ierr1,filein,"IBAND","I")
         CALL err_handle(io%iu0,ierr1,filein,"IBAND","I")
         IF (io%lopen) CALL WFORCE(io%iu6)
         STOP   ! error !!!!
      END SELECT iband

      ! reading k modus
      CALL rdatab(.FALSE.," ",iupar,"KPUSE","=","!",";","I", &
           kpuse,rdummy,cdummy,ldummy,strdummy,ndat,wdes%nkpts,ierr1)

      kmod: SELECT CASE (ierr1)

      CASE (0) kmod ! kpoints found
         IF(lop6) THEN
            WRITE(TIU6,'(A,I3,A)',ADVANCE = 'NO') &
                 "Use ",ndat," KPOINTS:"
            WRITE(TIU6,*) (KPUSE(i),i=1,ndat)
         END IF
         IF(lop0) THEN
            WRITE(TIU0,'(A,I3,A)',ADVANCE = 'NO') &
                 "Use ",ndat," KPOINTS:"
            WRITE(TIU0,*) (KPUSE(i),i=1,ndat)
         END IF

         nkpoint = ndat

      CASE (3) kmod ! nothing found
         IF(lop6) &
              WRITE(TIU6,'(A)') "No KPOINTS given: use all of them."
         IF(lop0) &
              WRITE(TIU0,'(A)') "No KPOINTS given: use all of them."

         nkpoint =  wdes%nkpts
         DO i=1,nkpoint
            kpuse(i)=i
         ENDDO

      CASE DEFAULT kmod
         CALL err_handle(io%iu6,ierr1,filein,"KPUSE","I")
         CALL err_handle(io%iu0,ierr1,filein,"KPUSE","I")
         IF (io%lopen) CALL WFORCE(io%iu6)
         STOP ! error !!!

      END SELECT kmod

      CALL rdatab(.FALSE.," ",iupar,"LSEPK","=","!",";","L", &
           idummy,rdummy,cdummy,ldummy,strdummy,ndat,1,ierr1)

      sepk: SELECT CASE (ierr1)
      CASE (3) sepk ! default: merge charge of all bands
         lsepk = .FALSE.

      CASE (0) sepk
         lsepk = ldummy(1)

      CASE DEFAULT sepk
         CALL err_handle(io%iu6,ierr1,filein,"LSEPK","L")
         CALL err_handle(io%iu0,ierr1,filein,"LSEPK","L")
         IF (io%lopen) CALL WFORCE(io%iu6)
         STOP ! error case : stop !!!!!!
      END SELECT sepk

      CALL rdatab(.FALSE.," ",iupar,"LSEPB","=","!",";","L", &
           idummy,rdummy,cdummy,ldummy,strdummy,ndat,1,ierr1)

      sepb: SELECT CASE (ierr1)
      CASE (3) sepb ! default: merge charge of all bands
         lsepb = .FALSE.

      CASE (0) sepb
         lsepb = ldummy(1)

      CASE DEFAULT sepb
         CALL err_handle(io%iu6,ierr1,filein,"LSEPB","L")
         CALL err_handle(io%iu0,ierr1,filein,"LSEPB","L")
         IF (io%lopen) CALL WFORCE(io%iu6)
         STOP ! error case : stop !!!!!!
      END SELECT sepb

      ! build boolean a kind table for lsepk and lsepb

      IF (lsepk .AND. lsepb) THEN  ! everything seperated
         sepmod = 0
      ELSE IF ((.NOT.lsepk) .AND. lsepb) THEN ! merge kpoints, seperate bands
         sepmod = 1
      ELSE IF (lsepk .AND. (.NOT.lsepb)) THEN ! seperate k-points, merge bands
         sepmod = 2
      ELSE
         sepmod = 3
      END IF

      ! for security: check if eint(1) < eint(2)

      IF (nbmodus <= -2) THEN
         IF (eint(1) > eint(2)) THEN
            ehelp = eint(1)
            eint(1) = eint(2)
            eint(2) = ehelp
         END IF
      END IF

    END SUBROUTINE read_pard



!
! SUBROUTINE write_pard
! output of the chrage density, to specified io-unit:
! just calling usual routines, but still: allways the same
!
    SUBROUTINE write_pard(filename,iunit)

      IMPLICIT NONE

      CHARACTER (LEN=*)  :: filename      ! name of output file
      INTEGER            :: iunit         ! streamer number

! local
      INTEGER              :: iuocc    ! file is allready connected to IO-unit
      LOGICAL              :: lopchg   ! streamer allready open ?

      io_begin
      INQUIRE(FILE=filename,OPENED=lopchg,NUMBER=iuocc)

      IF (lopchg) CLOSE(iuchg)

      OPEN(UNIT=iunit,FILE=filename,IOSTAT=ierr,STATUS="REPLACE")

      ! and use the usual output routines
      CALL OUTPOS(iunit,.FALSE.,INFO%SZNAM1,T_INFO,LATT_CUR%SCALE,LATT_CUR%A,.FALSE.,DYN%POSION)
      io_end

      CALL OUTCHG(GRIDC,iunit,.FALSE.,CHTOT)

      IF (INFO%ISPIN==2) THEN
! these magnetic moments are not really required on the PARCHG files
!         WRITE(iunit,'(5E20.12)') (T_INFO%ATOMOM(NI),NI=1,T_INFO%NIONS)
         do_io WRITE(iunit,*)
         CALL OUTCHG(GRIDC,iunit,.FALSE.,CHTOT(1,INFO%ISPIN))
      ENDIF

      CLOSE(iunit)

    END SUBROUTINE write_pard
!
! This subroutine calculates the partial charge density:
! just call the usual VASP routines
!
    SUBROUTINE calc_pard()

      IMPLICIT NONE

! First calculate right occupancies for
! augmentation charge, saved in CHRODE (that is out)

      CALL DEPSUM(w,wdes,lmdim,crhode,info%loverl)

      CALL SOFT_CHARGE(GRID,GRID_SOFT,W,WDES, CHDEN(1,1))

      CALL RC_FLIP(CHDEN,GRID_SOFT,INFO%ISPIN,.FALSE.)
      IF (SYMM%ISYM ==2) &
        CALL RHOSYM(CHDEN,GRID_SOFT,SYMM%PTRANS,NIOND,SYMM%MAGROT,1)
      IF ((SYMM%ISYM ==2).AND.(INFO%ISPIN==2)) &
        CALL RHOSYM(CHDEN(1,2),GRID_SOFT,SYMM%PTRANS,NIOND,SYMM%MAGROT,2)

      CALL RC_FLIP(CHTOT,GRIDC,INFO%ISPIN,.FALSE.)
      CALL US_FLIP(WDES, LMDIM, CRHODE, INFO%LOVERL, .FALSE.)

      CALL DEPLE(WDES,GRID_SOFT,GRIDC,GRIDUS,C_TO_US, &
                 LATT_CUR,P,T_INFO,SYMM, &
                 INFO%LOVERL,SOFT_TO_C,LMDIM,CRHODE, &
                 CHTOT,CHDEN,IRDMAX)


!jF: I think we do not need these two calls [??], but who cares ...
      CALL US_FLIP(WDES, LMDIM, CRHODE, INFO%LOVERL, .TRUE.)
      CALL RC_FLIP(CHDEN,GRID_SOFT,INFO%ISPIN,.TRUE.)

      IF (SYMM%ISYM == 1) &
           CALL RHOSYM(CHTOT,GRIDC,SYMM%PTRANS,NIOND,SYMM%MAGROT,1)
      IF ((SYMM%ISYM == 1).AND.(INFO%ISPIN == 2)) &
           CALL RHOSYM(CHTOT(1,2),GRIDC,SYMM%PTRANS,NIOND,SYMM%MAGROT,2)

    END SUBROUTINE calc_pard

! SUBROUTINE write_eigv()
! writing eigenvalues and fermiweights for all
! k-points and bands

    SUBROUTINE write_eigv()

      IMPLICIT NONE
      INTEGER          :: NEK,NEB

      INQUIRE(FILE="EIGENVAL",NUMBER=iueigv)
      IF (iueigv >= 0 .AND. iueigv <= 99) THEN

         DO NEK=1,WDES%NKPTS
            WRITE(iueigv,*)
            WRITE(iueigv,'(I3,A)',ADVANCE = "NO") NEK,". kpoint: "
            WRITE(iueigv,'(4E15.7)') WDES%VKPT(1,NEK),WDES%VKPT(2,NEK), &
                 WDES%VKPT(3,NEK),WDES%WTKPT(NEK)
            DO NEB=1,WDES%NB_TOT
               IF (INFO%ISPIN==1) WRITE(iueigv,'(1X,I3,4X,F10.4,2X,F6.2)')  &
                    NEB,REAL( W%CELTOT(NEB,NEK,1) ,KIND=q),W%FERTOT(NEB,NEK,1)
               IF (INFO%ISPIN==2) WRITE(iueigv,'(1X,I3,4X,F10.4,2X,F10.4,2X,F6.2,1X,F6.2)') &
                    NEB,REAL( W%CELTOT(NEB,NEK,1) ,KIND=q),REAL( W%CELTOT(NEB,NEK,INFO%ISPIN) ,KIND=q), &
                    W%FERTOT(NEB,NEK,1),W%FERTOT(NEB,NEK,2)
            ENDDO
         ENDDO
         IF (IO%LOPEN) CALL WFORCE(iueigv)
      END IF

    END SUBROUTINE write_eigv
!
! SUBROUTINE err_handle
! writes some information, when an error occured reading the input file
!

    SUBROUTINE err_handle(iu,ierr,file,keyword,type)

      IMPLICIT NONE
      INTEGER, INTENT (IN)   :: iu                ! output unit
      INTEGER, INTENT (IN)   :: ierr              ! error code

      CHARACTER (LEN = *), INTENT (IN) :: file    ! file name
      CHARACTER (LEN = *), INTENT (IN) :: keyword ! keyword
      CHARACTER , INTENT (IN)          :: type    ! type of data

      CHARACTER (LEN = 10)             :: strtype  ! type of data
      LOGICAL                          :: lout     ! out put device OK ?

      lout = (iu > 0)

      IF (lout) THEN

         WRITE(iu,'(A,A,A)') "Error reading file ",file," !"

         SELECT CASE (type)

         CASE ('S')
            strtype = "STRING"

         CASE ('I')
            strtype = "INTEGER"

         CASE ('F')
            strtype = "FLOATING"

         CASE ('C')
            strtype = "COMPLEX"

         CASE ('L')
            strtype = "LOGICAL"

         CASE DEFAULT
            strtype = "UNKNOWN"

         END SELECT

         SELECT CASE (ierr)

         CASE (1)
            WRITE(iu,'(A,A,A)') &
                 "Didn't find free I/O unit to open file ",file,"."

         CASE (2)
            WRITE(iu,'(A,A,A)') "Can't open file ",file,"."

         CASE (3)
            WRITE(iu,'(A,A,A)') "Didn't find tag ",keyword,"."

         CASE (4)
            WRITE(iu,'(A,A,A)') "Invalid data typ for tag ",keyword,"."
            WRITE(iu,'(A,A)') strtype," recommended."

         CASE (5)
            WRITE(iu,'(A,A,A)') &
                 "Error reading/parsing data list for tag ", &
                 keyword,"."
            WRITE(iu,'(A,A)') &
                 "Check format ! Data type recommended: ",strtype

         CASE (6)
            WRITE(iu,'(A,A)') &
                 "Cannot open scratch file for conversion of data for tag", &
                 keyword,"."

         CASE DEFAULT
            WRITE(iu,'(A)') "Unknown error code !"

         END SELECT

         WRITE(iu,'(A)') "STOP EXECUTION"
      END IF

    END SUBROUTINE err_handle
#else
    WRITE(0,*) 'SGI f90 can not compile required routines'
    STOP


#endif

  END SUBROUTINE parchg

END MODULE pardens


