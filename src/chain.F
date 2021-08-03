#include "symbol.inc"
!**********************************************************************
! RCS:  $Id: chain.F,v 1.2 2002/08/14 13:59:37 kresse Exp $
!
! Module which implements the elastic band and the nudged
! elastic band method (for references see below)
! module becomes active if IMAGES tag is read from the INCAR files
!
!**********************************************************************

  MODULE chain
    USE prec
    USE main_mpi
    USE poscar
    USE lattice
    IMPLICIT NONE

    REAL(q),ALLOCATABLE,SAVE :: posion_all(:,:,:)
    REAL(q) :: spring=10
    INTEGER :: nions_max
 ! tag VCAIMAGES allows to perform two MD's with e.g. different POTCAR files
 ! allowing to do force averaging
    LOGICAL :: LVCAIMAGES
    REAL(q) :: VCAIMAGES   ! weight of first images, weight of second image is 1-VCAIMAGES
 ! average forces over two images
 ! it is recommended to combine this with the VCA type calculations
 ! where the POSCAR/POTCAR/INCAR etc. are strictly identical
 ! and the VCA tag is set for one type to 0 and 1 respectively
 ! in the two POTCAR files
 ! forces and energies are averaged over the two nodes
 ! IMAGES in mpi_main.F is forced to two in this case


 ! the tag LTEMPER allows to perform parallel tempering runs
 ! IMAGES images of VASP are kicked up and the 
 ! temperatures between the images (TEBEG) are swapped using
 ! a Monte-Carlo algorithm (with corresponding temperature scaling)
    LOGICAL :: LTEMPER
    INTEGER :: NTEMPER     ! replica exchange every NTEMPER steps

    INTEGER, ALLOCATABLE, SAVE :: ATTEMPTS(:)
    INTEGER, ALLOCATABLE, SAVE :: SUCCESS(:)

!**********************************************************************
!
!  routine for forces between the images on the elastic band
!
!**********************************************************************

  CONTAINS
    SUBROUTINE chain_force(nions,posion,toten,force,a,b,iu6)
      INTEGER :: nions
      INTEGER :: iu6
      REAL(q) :: posion(3,nions)
      REAL(q) :: force(3,nions),toten
      REAL(q) :: tangent(3,nions)
      REAL(q) :: a(3,3),b(3,3)
! local variables
      REAL(q) :: x(3),x1(3),x2(3)
      REAL(q) :: e_chain,e_image,norm,norm1,norm2,proj,proj2,d1,d2
      REAL(q) :: force_chain(3,nions),d,const,ftangent
      INTEGER node,i,ni
      REAL(q),ALLOCATABLE,SAVE :: force_all(:,:,:)

      IF (images==0) RETURN
      IF (spring==-1000) RETURN

#if defined(MPI) || defined (MPI_CHAIN)
      node=comm_chain%node_me

!======================================================================
! Parallel tempering return
!======================================================================
      IF (LTEMPER) THEN
         RETURN
!======================================================================
! VCA average forces over two images
!======================================================================
      ELSE IF (LVCAIMAGES) THEN
         ALLOCATE(force_all(3,nions,2))
         force_all(:,:,1:2) =0
         force_all(:,:,node)=force
#ifndef old_vca
         ! first core in image merges the data
         IF (comm%node_me==1) THEN
            CALLMPI_C( M_sum_d( comm_chain, force_all(1,1,1), nions*3*2))
            force(:,:)=force_all(:,:,1)*VCAIMAGES+force_all(:,:,2)*(1-VCAIMAGES)
         ELSE
            force=0
         ENDIF
         ! now sum over all cores in one image
         CALLMPI_C( M_sum_d( comm, force(1,1), nions*3))
#else
         CALLMPI_C( M_sum_d( comm_chain, force_all(1,1,1), nions*3*2))
         force=0
         force(:,:)=force_all(:,:,1)*VCAIMAGES+force_all(:,:,2)*(1-VCAIMAGES)
#endif
         DEALLOCATE(force_all)
         RETURN
      ELSE
!======================================================================
! nudged elastic band method:
! communicate all positions to all nodes (brute force, but simple)
!======================================================================

      posion_all(:,:,1:images)=0
      posion_all(:,:,node)    =posion
      CALLMPI_C( M_sum_d( comm_chain, posion_all(1,1,1), nions*3*images))

!======================================================================
!  calculate the tangent in each point
!  and the distance to left and right image
!======================================================================
      i=node
      norm=0
      d1=0
      d2=0

      DO ni=1,nions

     ! distance to left image
         x1(:)=posion_all(:,ni,i-1)-posion_all(:,ni,i)
         x1(:)=MOD(x1(:)+100.5_q,1._q)-0.5_q
         CALL  dirkar(1,x1,a)
         norm1=sum(x1**2)
         d1=d1+norm1

     ! distance to right image
         x2(:)=posion_all(:,ni,i+1)-posion_all(:,ni,i)
         x2(:)=MOD(x2(:)+100.5_q,1._q)-0.5_q
         CALL  dirkar(1,x2,a)
         norm2=sum(x2**2)
         d2=d2+norm2
         
      ENDDO    

      d1=sqrt(d1)
      d2=sqrt(d2)

      ! local tangent for each ion

      DO ni=1,nions
         x(:)=(MOD(posion_all(:,ni,i+1)-posion_all(:,ni,i)+100.5_q,1._q)-0.5_q)*(d1**2) &
             -(MOD(posion_all(:,ni,i-1)-posion_all(:,ni,i)+100.5_q,1._q)-0.5_q)*(d2**2)
         x=x/((d1**2)*d2+d1*(d2**2))
         CALL  dirkar(1,x,a)
         norm=norm+sum(x**2)
         tangent(:,ni)=x
      ENDDO
         
      IF (norm/=0) tangent=tangent/sqrt(norm)

      e_chain=0
      e_image=0
      force_chain=0
      ftangent=0
!======================================================================
!
!  plain elastic band method
!
!======================================================================
      IF (spring/= 0) THEN
      const=ABS(spring)

      DO i=1,images+1
         DO ni=1,nions
            x(1)=posion_all(1,ni,i)-posion_all(1,ni,i-1)
            x(2)=posion_all(2,ni,i)-posion_all(2,ni,i-1)
            x(3)=posion_all(3,ni,i)-posion_all(3,ni,i-1)
            ! minimum image convention
            x(1)=MOD(x(1)+100.5_q,1._q)-0.5_q
            x(2)=MOD(x(2)+100.5_q,1._q)-0.5_q
            x(3)=MOD(x(3)+100.5_q,1._q)-0.5_q

            CALL  dirkar(1,x,a)
            d = x(1)*x(1)+x(2)*x(2)+x(3)*x(3)

            e_chain=e_chain+ d*const/2

            ! add force to force_chain
            IF (i-1==node) THEN
               force_chain(1,ni)=force_chain(1,ni)+x(1)*const
               force_chain(2,ni)=force_chain(2,ni)+x(2)*const
               force_chain(3,ni)=force_chain(3,ni)+x(3)*const
               e_image=e_image+d*const/4
            ENDIF
            IF (i==node) THEN
               force_chain(1,ni)=force_chain(1,ni)-x(1)*const
               force_chain(2,ni)=force_chain(2,ni)-x(2)*const
               force_chain(3,ni)=force_chain(3,ni)-x(3)*const
               e_image=e_image+d*const/4
            ENDIF
         ENDDO
      ENDDO
      ENDIF
!======================================================================
!
!  nudged elastic band method
!   Hannes Jonsson and Greg Mills, preprint
!    (sub to Journal of Chem Phys.)
!  this is a little bit tricky, and the method might not work
!  with CG or molecular dynamics because no correction to energy
!  is calculated
!======================================================================
      IF ( spring <= 0 ) THEN
         ! force_chain contains already the spring force
         ! e_image must be reset to 0
         e_image=0
         e_chain=0
         ! project force and chain force onto local tangent
         proj =0
         proj2=0
         DO ni=1,nions
            proj =proj +tangent(1,ni)*force(1,ni) &
                       +tangent(2,ni)*force(2,ni) &
                       +tangent(3,ni)*force(3,ni)
            proj2=proj2+tangent(1,ni)*force_chain(1,ni) &
                       +tangent(2,ni)*force_chain(2,ni) &
                       +tangent(3,ni)*force_chain(3,ni)
         ENDDO
         ftangent   =(proj-proj2)
         ! force into direction of local tangent
         force_chain=tangent* (-proj+proj2)
      ENDIF
!======================================================================
!
! report important results to unit 6
!
!======================================================================
90    FORMAT( '  energy of chain is (eV) ',F16.6,' for this image ',F16.6,/ &
           '  tangential force (eV/A) ',F16.6, / &
           '  left and right image ',2F10.6,' A')

100   FORMAT( ' TANGENT     ',35X,'CHAIN-FORCE (eV/Angst)'/ &
           ' ----------------------------------------------', &
           '-------------------------------------')
110   FORMAT((3F13.5,3X,3F14.6))
120   FORMAT( ' ----------------------------------------------', &
           '-------------------------------------')

101   FORMAT( ' CHAIN + TOTAL  (eV/Angst)'/ &
           ' ----------------------------------------------')
111   FORMAT((3F13.5))
121   FORMAT( ' ----------------------------------------------')

      toten=toten+e_chain/images

      IF (iu6 >=0) THEN
         WRITE(iu6,90 ) e_chain,e_image,ftangent,d1,d2
         WRITE(iu6,100)
         WRITE(iu6,110) (tangent(:,ni),force_chain(:,ni),ni=1,nions)
         WRITE(iu6,120)
      ENDIF

      force=force+force_chain
      IF (iu6 >=0) THEN
         WRITE(iu6,101)
         WRITE(iu6,111) (force(:,ni),ni=1,nions)
         WRITE(iu6,121)
      ENDIF


      ENDIF
#endif
    END SUBROUTINE chain_force

!**********************************************************************
!
! there are several points where a global sum between chains
! is required in order to get correct results
! terms like the total energy / or the kinetic energy
! make only sense in a global and not per image sense
!
!**********************************************************************

    SUBROUTINE sum_chain( value )
      REAL(q) :: value
      REAL(q) :: value_all(images)
      INTEGER node

      IF (images==0    ) RETURN
      IF (spring==-1000) RETURN
      IF (LTEMPER) RETURN
#if defined(MPI) || defined (MPI_CHAIN)
      ! VCAIMAGES returns average value for energies etc.
      IF (LVCAIMAGES) THEN
         node=comm_chain%node_me
         value_all=0
         value_all(node)=value
#ifndef old_vca
         ! first core in image merges data
         IF (comm%node_me==1) THEN
            CALLMPI_C( M_sum_d( comm_chain, value_all, 2))
            value=value_all(1)*VCAIMAGES+value_all(2)*(1-VCAIMAGES)
         ELSE
            value=0
         ENDIF
         ! now sum over all cores in one image
         CALLMPI_C( M_sum_d( comm, value, 1))
#else
         CALLMPI_C( M_sum_d( comm_chain, value_all, 2))
         value=value_all(1)*VCAIMAGES+value_all(2)*(1-VCAIMAGES)
#endif
      ELSE
         CALLMPI_C( M_sum_d( comm_chain, value, 1 ))
      ENDIF
#endif

    END SUBROUTINE sum_chain

!**********************************************************************
!
! also the logical break conditionions must be
!
!**********************************************************************

    SUBROUTINE PARALLEL_TEMPERING(NSTEP, NIONS, POSION, VEL, TOTEN, FORCE, TEBEG, TEEND, & 
        A, B, IU6)
      USE constant
      IMPLICIT NONE
      INTEGER :: NSTEP           ! step
      INTEGER :: NIONS           ! number of ions
      INTEGER :: IU6             ! std-out  (unit for OUTCAR file)
      REAL(q) :: POSION(3,nions) ! position of ions
      REAL(q) :: VEL(3,nions)    ! velocity of ions
      REAL(q) :: FORCE(3,nions)
      REAL(q) :: TOTEN           ! total energy
      REAL(q) :: TEBEG           ! temperature of ensemble
      REAL(q) :: TEEND           ! temperature of ensemble (must equal TEBEG)
      REAL(q) :: A(3,3),B(3,3)   ! lattice and reciprocal lattice
! local
      INTEGER,SAVE :: SWAP_START=1
      REAL(q),ALLOCATABLE :: TEBEG_old(:), TEBEG_new(:), TOTEN_all(:)
      REAL(q) :: E1, E2
      INTEGER,ALLOCATABLE :: ID(:)
      INTEGER :: SWAP
      INTEGER :: NODE, I, J
      REAL(q) :: TEBEG_new_local, E
      REAL(q) :: value

#if defined(MPI) || defined (MPI_CHAIN)
      IF (.NOT. LTEMPER) RETURN

      IF (IU6>=0) WRITE(IU6,*) 'parallel tempering routine entered NSTEP=, NTEMPER=', NSTEP,NTEMPER
      CALL RANDOM_NUMBER(value)
      ! broadcase to all nodes
      CALLMPI_C( M_bcast_d( comm_world, value, 1))

      IF (value <1.0_q/NTEMPER) THEN

         ALLOCATE(TEBEG_old(images), TEBEG_new(images), TOTEN_all(images), ID(images))

         ! my id, i.e. which subdir I am running in
         node=comm_chain%node_me

         TEBEG_old=0
         TEBEG_old(node)=TEBEG
         TOTEN_all=0
         TOTEN_all(node)=TOTEN
         ID=0
         ID(node)=node
         
         CALLMPI_C( M_sum_d( comm_chain, TEBEG_old,  images))
         CALLMPI_C( M_sum_d( comm_chain, TOTEN_all,  images))
         CALLMPI_C( M_sum_i( comm_chain, ID,  images))

! sort the temperatures ascendingly and store the corresponding node id 
         CALL SORT_ASC_REAL(images, TEBEG_old, ID)
!======================================================================
! now select swaps randomly
! images/2 swaps 
! the array SWAP stores the list of images to be swapped 
! 0 no swap
! 1        two lowest temperatures are swapped
! 2        next two are swapped and so on
! images-1 swap last two
!======================================================================
! old code, one swap attempt each step
!         CALL RANDOM_NUMBER(value)  ! values between [0,1[
!         CALLMPI_C( M_bcast_d( comm_world, value, 1))
!         SWAP=(images-1)*value+1    ! create a random number [1,images-1]
!         CALLMPI_C( M_bcast_i( comm_chain, SWAP,  1))
! new version
       TEBEG_new=TEBEG_old

       SWAP_START=3-SWAP_START ! start swap either at first or second image

       DO SWAP=SWAP_START,images-1,2
         
         IF (IU6>=0) WRITE(IU6,'(A,16I10)')   ' attempting swapping', SWAP

         ATTEMPTS(SWAP)=ATTEMPTS(SWAP)+1
         
         E=(1.0_q/TEBEG_old(SWAP)-1.0_q/TEBEG_old(SWAP+1))/ BOLKEV * (TOTEN_all(ID(SWAP))-TOTEN_all(ID(SWAP+1)))
         E1=(1.0_q/TEBEG_old(SWAP)-1.0_q/TEBEG_old(SWAP+1))/ BOLKEV
         E2=(TOTEN_all(ID(SWAP))-TOTEN_all(ID(SWAP+1)))
         E=EXP(E)
         CALL RANDOM_NUMBER(value) ! values between [0,1[
         CALLMPI_C( M_bcast_d( comm_world, value, 1))

         IF (IU6>=0) WRITE(IU6,'(A,16F10.4)') '  1/T1-1/T2         ', E1
         IF (IU6>=0) WRITE(IU6,'(A,16F10.4)') '  E1  -E2           ', E2        
         IF (IU6>=0) WRITE(IU6,'(A,16F10.7)')   '            random  ', value

         IF (value>E) THEN
            ! Metropolis forbids swap
            IF (IU6>=0) WRITE(IU6,'(A,16I10)')   '          noswapping', SWAP
         ELSE
            SUCCESS(SWAP)=SUCCESS(SWAP)+1
            TEBEG_new(SWAP)  =TEBEG_old(SWAP+1)
            TEBEG_new(SWAP+1)=TEBEG_old(SWAP)
            IF (IU6>=0) WRITE(IU6,'(A,16I10)')   '            swapping', SWAP
         ENDIF
         
      ENDDO
 
         IF (IU6>=0) WRITE(IU6,'(A,16F14.7)') ' parallel tempering old TOTEN ', (TOTEN_all(ID(I)),I=1,images)
         IF (IU6>=0) WRITE(IU6,'(A,16F14.7)') ' parallel tempering old TEBEG ', TEBEG_old
         IF (IU6>=0) WRITE(IU6,'(A,16F14.7)') ' parallel tempering new TEBEG ', TEBEG_new
         IF (IU6>=0) WRITE(IU6,'(A,16F14.7)') ' Acceptance ratio for swaps          ', REAL(SUCCESS,q)/MAX(1,ATTEMPTS)
      
         DO I=1,images
            IF ( ID(I)==node) THEN
               IF (TEBEG /= TEBEG_old(I)) THEN
                  WRITE(*,*) 'internal error in PARALELL_TEMPERING:', I,ID(I), TEBEG, TEBEG_old(I)
               ENDIF
               TEBEG_new_local=TEBEG_new(I)
            ENDIF
         END DO

         VEL(:,:)=VEL(:,:)*SQRT(TEBEG_new_local/TEBEG)

         TEBEG=TEBEG_new_local
         TEEND=TEBEG_new_local

         IF (IU6>=0) WRITE(IU6,*)
         IF (IU6>=0) WRITE(IU6,"('   TEBEG  = ',F6.1,';   TEEND  =',F6.1)") TEBEG,TEEND
         IF (IU6>=0) WRITE(IU6,*)

         DEALLOCATE(TEBEG_old, TEBEG_new, TOTEN_all, ID)
      ENDIF

#endif
    END SUBROUTINE PARALLEL_TEMPERING


!**********************************************************************
! sorts RA in descending order, and rearanges an index array RB
! seems to be a quicksort, but I am not sure
! subroutine writen by Florian Kirchhof
!**********************************************************************

  SUBROUTINE SORT_ASC_REAL(N, RA, RB)
    REAL(q) :: RA(N)
    INTEGER :: RB(N)
    REAL(q) :: RRA
    INTEGER :: RRB
    INTEGER :: N, L, IR, J, I

    IF (N<=1) RETURN

    L=N/2+1
    IR=N
10  CONTINUE
    IF(L.GT.1)THEN
       L=L-1
       RRA=RA(L)
       RRB=RB(L)
    ELSE
       RRA=RA(IR)
       RRB=RB(IR)
       RA(IR)=RA(1)
       RB(IR)=RB(1)
       IR=IR-1
       IF(IR.EQ.1)THEN
          RA(1)=RRA
          RB(1)=RRB
          RETURN
       ENDIF
    ENDIF
    I=L
    J=L+L
20  IF(J.LE.IR)THEN
       IF(J.LT.IR)THEN
          IF(RA(J).GT.RA(J+1))J=J+1
       ENDIF
       IF(RRA.GT.RA(J))THEN
          RA(I)=RA(J)
          RB(I)=RB(J)
          I=J
          J=J+J
       ELSE
          J=IR+1
       ENDIF
       GO TO 20
    ENDIF
    RA(I)=RRA
    RB(I)=RRB
    GO TO 10
  END SUBROUTINE SORT_ASC_REAL


!**********************************************************************
!
! also the logical break conditionions must be
!
!**********************************************************************

    SUBROUTINE and_chain( value )
      LOGICAL :: value
      REAL(q) :: sum

      IF (images==0) RETURN
      IF (spring==-1000) RETURN
#if defined(MPI) || defined(MPI_CHAIN)
! if one node is .FALSE., .FALSE. is returned on all nodes
      IF (value) THEN
         sum=0
      ELSE
         sum=1
      ENDIF
      IF (LVCAIMAGES) THEN
#ifndef old_vca
         ! first core in image merges data
         IF (comm%node_me==1) THEN
            CALLMPI_C( M_sum_d( comm_chain, sum, 1 ))
         ELSE
            sum=0
         ENDIF
         ! now sum over all cores in one image
         CALLMPI_C( M_sum_d( comm, sum, 1))
#else
         CALLMPI_C( M_sum_d( comm_chain, sum, 1 ))
#endif
      ELSE
         CALLMPI_C( M_sum_d( comm_chain, sum, 1 ))
      ENDIF

      IF (sum>=1) THEN
         value=.FALSE.
      ELSE
         value=.TRUE.
      ENDIF
#endif
    END SUBROUTINE and_chain

!**********************************************************************
!
! initialize the chain (repeated image mode)
! read the spring constant
! and  the two outer images, these images are kept fixed
! during the entire simulation
!
!**********************************************************************

    SUBROUTINE chain_init (T_INFO, IO)
      USE base
      TYPE (in_struct) :: IO
      TYPE (type_info) :: T_INFO

! needed only temporarily
      INTEGER NIOND,NIONPD,NTYPPD,NTYPD
      TYPE (latt)::       LATT_CUR
      TYPE (type_info) :: T_I
      TYPE (dynamics)  :: DYN
      INTEGER     IDUM,IERR,N,idir,node
      CHARACTER (1)   CHARAC
      COMPLEX(q)  CDUM  ; LOGICAL  LDUM
      REAL(q) :: RDUM

! quick return, if we are not running in image mode
      IF (images==0) RETURN
#if defined(MPI) || defined(MPI_CHAIN)
! read the VCAIMAGES
      VCAIMAGES=-1
         
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'VCAIMAGES','=','#',';','F', &
           &        IDUM,VCAIMAGES,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) WRITE(IO%IU0,*)'Error reading item ''VCAIMAGES'' from file INCAR.'
         STOP
      ENDIF
      CALL XML_INCAR('VCAIMAGES','F',IDUM,VCAIMAGES,CDUM,LDUM,CHARAC,N)
      IF (VCAIMAGES==-1) THEN
         LVCAIMAGES=.FALSE.
      ELSE
         LVCAIMAGES=.TRUE.
      ENDIF


! LTEMPER -- use subspace diagonalization or not (default is TRUE):
      LTEMPER=.FALSE.
      CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'LTEMPER','=','#',';','L', &
     &            IDUM,RDUM,CDUM,LTEMPER,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR. &
     &                    ((IERR==0).AND.(N<1))) THEN
         IF (IO%IU0>=0) &
         WRITE(IO%IU0,*)'Error reading item ''LTEMPER'' from file INCAR.'
         STOP
      ENDIF
      CALL XML_INCAR('LTEMPER','L',IDUM,RDUM,CDUM,LTEMPER,CHARAC,N)

      IF (LTEMPER) THEN
         ! read NTEMPER
         NTEMPER=200
         CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'NTEMPER','=','#',';','I', &
              &            NTEMPER,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR. &
              &                    ((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) &
                 WRITE(IO%IU0,*)'Error reading item ''NTEMPER'' from file INCAR.'
            STOP
         ENDIF

         CALL XML_INCAR('NTEMPER','I',NTEMPER,RDUM,CDUM,LDUM,CHARAC,N)

         ALLOCATE(ATTEMPTS(images-1))
         ALLOCATE(SUCCESS(images-1))
         ATTEMPTS=0
         SUCCESS=0
      ENDIF

      IF (LVCAIMAGES .OR. LTEMPER) THEN
         nions_max=T_INFO%NIONS
#ifndef old_vca
         IF (comm%node_me==1) THEN
            CALL M_max_i(comm_chain, nions_max, 1 )
         ELSE
            nions_max=0
         ENDIF
         ! now sum over all cores in one image
         CALLMPI_C( M_sum_i( comm, nions_max, 1))
#else
         CALL M_max_i(comm_chain, nions_max, 1 )
#endif
         IF (T_INFO%NIONS /= nions_max) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'ERROR: image mode number of ions wrong'
            STOP
         ENDIF
      ELSE

! allocate work array for all positions posion_all
         ALLOCATE(posion_all(3,t_info%nions,0:images+1))
! default spring constant
         spring=-5.
! read the spring constant
         CALL RDATAB(IO%LOPEN,INCAR,IO%IU5,'SPRING','=','#',';','F', &
     &        IDUM,spring,CDUM,LDUM,CHARAC,N,1,IERR)
         IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'Error reading item ''SPRING'' from file INCAR.'
            STOP
         ENDIF
         CALL XML_INCAR('SPRING','F',IDUM,spring,CDUM,LDUM,CHARAC,N)

! read the start and end points to posion_all
! read 00/POSCAR file, a little bit of fiddling is required
         idir=0
         CALL MAKE_DIR_APP(idir)

         CALL RD_POSCAR_HEAD(LATT_CUR, T_I, &
              NIOND,NIONPD, NTYPD,NTYPPD, IO%IU0, IO%IU6)
         CALL RD_POSCAR(LATT_CUR, T_I, DYN, &
              NIOND,NIONPD, NTYPD,NTYPPD, &
              IO%IU0, IO%IU6)

         IF (T_I%NIONS /= T_INFO%NIONS) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'ERROR: image mode number of ions wrong'
            STOP
         ENDIF
         posion_all(:,:,0)= DYN%POSION
! read images+1/POSCAR file
         idir=images+1
         CALL MAKE_DIR_APP(idir)
         
         CALL RD_POSCAR_HEAD(LATT_CUR, T_I, &
              NIOND,NIONPD, NTYPD,NTYPPD, IO%IU0, IO%IU6)
         CALL RD_POSCAR(LATT_CUR, T_I, DYN, &
              NIOND,NIONPD, NTYPD,NTYPPD, &
              IO%IU0, IO%IU6)

         IF (T_I%NIONS /= T_INFO%NIONS) THEN
            IF (IO%IU0>=0) WRITE(IO%IU0,*)'ERROR: image mode number of ions wrong'
            STOP
         ENDIF
         posion_all(:,:,images+1)= DYN%POSION
      ENDIF

      node=comm_chain%node_me
      CALL MAKE_DIR_APP(node)
#endif
    END SUBROUTINE chain_init

!**********************************************************************
!
! returns true if hyper nudged elastic band method is used
!
!**********************************************************************

    FUNCTION LHYPER_NUDGE()
      LOGICAL LHYPER_NUDGE
      IF (images==0 .OR. spring /= 0 ) THEN
         LHYPER_NUDGE=.FALSE.
      ELSE
         LHYPER_NUDGE=.TRUE.
      ENDIF

    END FUNCTION LHYPER_NUDGE
END MODULE chain
