#include "symbol.inc"
!***********************************************************************
!
! LDA+U Module written by Olivier Bengone 
! please direct any queries related to this module directly to Olivier
! Bengone
!  present email address:   olivier@fysik.uu.se
! in case you can not reach Olivier you might contact 
!  Georg.Kresse@univie.ac.at
!
!***********************************************************************

      MODULE LDAPLUSU_MODULE
      
      USE prec
      USE vaspxml

      IMPLICIT NONE

      LOGICAL, PRIVATE, SAVE :: TLDA_U                        ! apply LDA+U correction
      LOGICAL, PRIVATE, SAVE :: TFIRST                        ! have the OVERLAP_AE integrals been calculated?
      LOGICAL, PRIVATE, SAVE :: LORBMOM                       ! calculate orbital moments?

      INTEGER, PRIVATE, SAVE :: LMAX_,LMMAX_,LNMAX_
      INTEGER, PRIVATE, SAVE :: NCDIJ      
      INTEGER, PRIVATE, SAVE :: PRINTOCC=0                    ! IF /= 0 print test information
      INTEGER, PRIVATE, SAVE :: TYPE_POT                      ! type of LDA+U potential (1 or 2)  
      INTEGER, PRIVATE, ALLOCATABLE,SAVE :: LANG_LDAPLUSU(:)  ! l-quantum number for LDA+U correction per type

      REAL(q), PRIVATE, SAVE :: DOUBLEC_LDAPLUSU=0._q         ! total LDA+U double counting energy
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: U(:)             ! U parameter per type
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: J(:)             ! J parameter per type
                                               
      OVERLAP , PRIVATE, ALLOCATABLE :: OCC_MAT(:,:,:)
      OVERLAP , PRIVATE, ALLOCATABLE :: OCC_MAT_ALL(:,:,:,:)
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: OVERLAP_AE_(:,:,:)
      REAL(q), PRIVATE, ALLOCATABLE, SAVE :: ORBMOM(:,:,:)

      COMPLEX(q), PRIVATE, SAVE :: V(7,7,7,7)
        
      CONTAINS

!**********************************************************************
!      LDAU_READER
!                                           
! this subroutine reads the required  parameters from the INCAR file        
!               
! LDAU     = .TRUE. or .FALSE.                                   
! LDAUTYPE = 1 or 2
!            1  use exact four center coulomb integrals
!            2  use only U_eff=U-J                                       
! LDAUL   l-quantum number on which U acts (one value for each species)
! LDAUU   U coefficient (coulomb interaction) for each species
! LDAUJ   J coefficient (exchange) for each species
!
!**********************************************************************

      SUBROUTINE LDAU_READER(NTYP,IU5,IU0) 
      USE base
      USE vaspxml
   
      IMPLICIT NONE 

      INTEGER IU5,IU0
      INTEGER ITYP,NTYP  
      INTEGER IDUM, N, IERR
      REAL(q) RDUM
      COMPLEX(q) CDUM
      LOGICAL LOPEN,LDUM
      CHARACTER (1) :: CHARAC

      LOPEN=.FALSE.
      OPEN(UNIT=IU5,FILE=INCAR,STATUS='OLD')

!=======================================================================
!    read LDA+U parameters from INCAR  
!=======================================================================
      TLDA_U=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAU','=','#',';','L', &
     &   IDUM,RDUM,CDUM,TLDA_U,CHARAC,N,1,IERR)

      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LDAU'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
            TLDA_U=.FALSE.
         ENDIF
      ENDIF
      CALL XML_INCAR('LDAU','L',IDUM,RDUM,CDUM,TLDA_U,CHARAC,N)
       
      TYPE_POT=2
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAUTYPE','=','#',';','I', &
     &   TYPE_POT,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LDAUTYPE'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
         TLDA_U=.FALSE.
      ENDIF
      
      CALL XML_INCAR_V('LDAUTYPE','I',TYPE_POT,RDUM,CDUM,LDUM,CHARAC,N)

      ALLOCATE(LANG_LDAPLUSU(NTYP),U(NTYP),J(NTYP))

      LANG_LDAPLUSU=2
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAUL','=','#',';','I', &
     &   LANG_LDAPLUSU,RDUM,CDUM,LDUM,CHARAC,N,NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<NTYP))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LDAUL'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
         TLDA_U=.FALSE.
      ENDIF
      CALL XML_INCAR_V('LDAUL','I',LANG_LDAPLUSU,RDUM,CDUM,LDUM,CHARAC,N)
   
!       DO N=1,NTYP
!       IF (LANG_LDAPLUSU(N) /= 2 .AND. LANG_LDAPLUSU(N) /= -1 ) THEN
!          IF (IU0>=0) &
!             WRITE(IU0,'(A,I3,A)')' WARNING: LDA+U for L=',LANG_LDAPLUSU(N),' LDAUTYPE must be set to 2'
!             TYPE_POT = 2 
!          ENDIF
!       ENDDO

      U=0
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAUU','=','#',';','F', &
     &   IDUM,U,CDUM,LDUM,CHARAC,N,NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<NTYP))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LDAUU'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
         TLDA_U=.FALSE.
      ENDIF
      CALL XML_INCAR_V('LDAUU','F',IDUM,U,CDUM,LDUM,CHARAC,N)

      J=0
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAUJ','=','#',';','F', &
     &   IDUM,J,CDUM,LDUM,CHARAC,N,NTYP,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<NTYP))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LDAUJ'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
         TLDA_U=.FALSE.
      ENDIF
      CALL XML_INCAR_V('LDAUJ','F',IDUM,J,CDUM,LDUM,CHARAC,N)

      PRINTOCC=0
      CALL RDATAB(LOPEN,INCAR,IU5,'LDAUPRINT','=','#',';','I', &
     &   PRINTOCC,RDUM,CDUM,LDUM,CHARAC,N,1,IERR)
      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''PRINTOCC'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
         ENDIF
      ENDIF
      IF (IU0<0) PRINTOCC=0
      CALL XML_INCAR('LDAUPRINT','I',PRINTOCC,RDUM,CDUM,LDUM,CHARAC,N)

      LORBMOM=.FALSE.
      CALL RDATAB(LOPEN,INCAR,IU5,'LORBMOM','=','#',';','L', &
     &   IDUM,RDUM,CDUM,LORBMOM,CHARAC,N,1,IERR)

      IF (((IERR/=0).AND.(IERR/=3)).OR.((IERR==0).AND.(N<1))) THEN
         IF (IU0>=0) THEN
            WRITE(IU0,*)'Error reading item ''LORBMOM'' from file INCAR.'
            WRITE(IU0,*)'Error code was IERR=',IERR,' Found N=',N,' data items'
            LORBMOM=.FALSE.
         ENDIF
      ENDIF
      CALL XML_INCAR('LORBMOM','L',IDUM,RDUM,CDUM,LORBMOM,CHARAC,N)

      CLOSE(IU5)

      RETURN
      END SUBROUTINE LDAU_READER   


      SUBROUTINE INITIALIZE_LDAU(NIONS,NTYP,P,LNONCOLLINEAR,IU,IDIOT)
      
      USE pseudo
      
      IMPLICIT NONE
      
      INTEGER NIONS,NTYP
      
      TYPE(potcar) P(NTYP)

      INTEGER ITYP,ISPIN
      INTEGER LNMAX,LMMAX
      LOGICAL LNONCOLLINEAR
      
      INTEGER IU,IDIOT,ITUT,IDUM
      REAL(q) RTUT,RDUM
      COMPLEX(q) CDUM
      LOGICAL LDUM

      NCDIJ=2
      IF (LNONCOLLINEAR) NCDIJ=4

!=======================================================================
!  Sanity check: L(S)DA+U only implemented for PAW potentials
!=======================================================================
      DO ITYP=1,NTYP
         IF (.NOT.ASSOCIATED(P(ITYP)%QPAW).AND.LANG_LDAPLUSU(ITYP)/=-1) THEN
            CALL VTUTOR('S','NOLDAU',RDUM,1,IDUM,1,CDUM,1,LDUM,1,IU,IDIOT)
         ENDIF
      ENDDO

!=======================================================================
!  allocate objects for lda+u calculations         
!=======================================================================
      LMAX_=0;LMMAX_=0;LNMAX_=0

      LNMAX_=MAXVAL(P(1:NTYP)%LMAX)
      DO ITYP=1,NTYP
         LMAX_=MAX(LMAX_,MAXVAL(P(ITYP)%LPS(1:P(ITYP)%LMAX)))
      ENDDO
      LMMAX_=(LMAX_+1)**2

      ALLOCATE(OVERLAP_AE_(LNMAX_,LNMAX_,NTYP),OCC_MAT(LMMAX_,LMMAX_,NCDIJ), &
     &          OCC_MAT_ALL(LMMAX_,LMMAX_,NCDIJ,NIONS)) 

      OCC_MAT_ALL=0._q
      OVERLAP_AE_=0._q
      TFIRST=.TRUE.

      IF (LORBMOM) THEN
         ALLOCATE(ORBMOM(LMAX_,3,NIONS))
         ORBMOM=0._q
      ENDIF
      
      RETURN 
      END SUBROUTINE INITIALIZE_LDAU 


!**********************************************************************
! write the LDA + U parameters to the OUTCAR file
!**********************************************************************
      SUBROUTINE WRITE_LDApU(IU6)

      IMPLICIT NONE

      INTEGER IU6

      IF (IU6>=0) THEN
         WRITE(IU6,100) TYPE_POT
         WRITE(IU6,110) LANG_LDAPLUSU
         WRITE(IU6,120) U
         WRITE(IU6,130) J
      ENDIF
      
  100 FORMAT(' LDA+U is selected, type is set to LDAUTYPE = ',I2)
  110 FORMAT('   angular momentum for each species LDAUL = ',20I5)
  120 FORMAT('   U (eV)           for each species LDAUU = ',20F5.1)
  130 FORMAT('   J (eV)           for each species LDAUJ = ',20F5.1)
       
      END SUBROUTINE WRITE_LDApU


!**********************************************************************
!      XML_WRITE_INITIALIZE_LDAU
!   this subroutine writes the LDA+U parameters to the XML file
!   if required
!**********************************************************************

      SUBROUTINE XML_WRITE_LDAU
      USE pseudo
      IMPLICIT NONE
 
      LOGICAL :: LDUM
      INTEGER :: IDUM
      REAL(q) :: RDUM
      COMPLEX(q)  :: CDUM
      CHARACTER (1) :: CHARAC
 
      CALL XML_INCAR('LDAU','L',IDUM,RDUM,CDUM,TLDA_U,CHARAC,1)

      IF (.NOT.TLDA_U) RETURN
 
      CALL XML_INCAR_V('LDAUTYPE','I',TYPE_POT,RDUM,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('LDAUL','I',LANG_LDAPLUSU,RDUM,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('LDAUU','F',IDUM,U,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR_V('LDAUJ','F',IDUM,J,CDUM,LDUM,CHARAC,1)
      CALL XML_INCAR('LDAUPRINT','I',PRINTOCC,RDUM,CDUM,LDUM,CHARAC,1)
      END SUBROUTINE XML_WRITE_LDAU


!**********************************************************************
!
! query function which returns .TRUE. or .FALSE. depending on
! whether LDA plus U is used or not
!
!**********************************************************************

      FUNCTION USELDApU()
      LOGICAL USELDApU

      USELDApU=TLDA_U

      END FUNCTION USELDApU

!**********************************************************************
!
! query function which returns .TRUE. or .FALSE. depending on
! whether LDA plus U is used or not
!
!**********************************************************************

      FUNCTION L_NO_LSDA()
      LOGICAL L_NO_LSDA

      L_NO_LSDA=.FALSE.
      IF (TYPE_POT==4) L_NO_LSDA=.TRUE.

      END FUNCTION L_NO_LSDA

!**********************************************************************
!
! query function which returns .TRUE. or .FALSE. depending on
! whether all overlap integrals have been calculated or not
!
!**********************************************************************

      FUNCTION INTEGRALS_LDApU()
      LOGICAL INTEGRALS_LDApU

      INTEGRALS_LDApU=TFIRST

      END FUNCTION INTEGRALS_LDApU

!***********************************************************************
!
!***********************************************************************

      FUNCTION LCALC_ORBITAL_MOMENT()
      LOGICAL LCALC_ORBITAL_MOMENT
      
      LCALC_ORBITAL_MOMENT=LORBMOM
      
      END FUNCTION LCALC_ORBITAL_MOMENT

!**********************************************************************
!
!  This routine calculates the occupancy matrix, i.e., how many 
!  electrons are in each lm (OCC_MAT), the corrections to the strength
!  parameters CDIJ, and the double counting energy DBLE_LDAU.
!
!  presently the atom number is also handled down (IATOM)
!  but not used (it could be used to kick the system into a 
!  particular state)
!
!**********************************************************************

      SUBROUTINE LDAPLUSU(LMDIM,IATOM,ITYP,CRHODE,CDIJ,PP,DBLE_LDAU)
    
      USE pseudo
      USE poscar
      USE radial

      IMPLICIT NONE

      TYPE(potcar) PP  ! pseudopotential descriptor
            
      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      
      REAL(q) DBLE_LDAU

      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ) ! occupancy matrix
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)   ! strength parameters

      INTEGER ISP,LN1,LN2,LMN1,LMN2,L1,L2,M1,M2,LM1,LM2
      INTEGER LNMAX,LMMAX

      DBLE_LDAU=0._q
      OCC_MAT=0._q
      OCC_MAT_ALL(:,:,:,IATOM)=0._q
      
      ! quick return if possible
      DO LN1=1,PP%LMAX
         IF (PP%LPS(LN1)==LANG_LDAPLUSU(ITYP)) GOTO 100
      ENDDO
      RETURN

  100 CONTINUE
                                             
!===================================================================
!  OCC(LM1,LM2) = SUM_{nk} f_nk <PSI_nk|Y_lm1> > <Y_lm2|PSI_nk> 
!  WITH
!  |PSI_NK> = SUM_{lmn3} <~P_{lmn3|~PSI_nk>  |PHI_lmn3> 
!
!  OCC(LM1,LM2) = SUM_{lmna,lmnb} delta(lma,lm1)*delta(lmb,lm2) 
!                 RHO_{lmna,lmnb} * <PHI_{ln1}|PHI_{ln2}>
!
!  LM =  (1)    (2   3   4)      (5    6     7      8     9)
!        (s)   (p_y p_z p_x)   (d_xy d_yz d_z2-r2 d_xz d_x2-y2)
!===================================================================
              
      LNMAX=PP%LMAX
      LMMAX=(MAXVAL(PP%LPS(1:LNMAX))+1)**2

      LMN1=0
      DO LN1=1,LNMAX
      L1=PP%LPS(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=PP%LPS(LN2)
         DO M2=1,2*L2+1
         LMN2=LMN2+1
         LM2=L2**2+M2
         DO ISP=1,NCDIJ
            OCC_MAT(LM1,LM2,ISP)=OCC_MAT(LM1,LM2,ISP) + &
               CRHODE(LMN1,LMN2,ISP)*OVERLAP_AE_(LN1,LN2,ITYP)
         ENDDO !(ISPIN)
         ENDDO !(M2)
         ENDDO !(LN2)
      ENDDO !(M1)
      ENDDO !(LN1)

! test
!     IF (IATOM==1) THEN
!        OCC_MAT(5:,5:,:)=0._q
!     ENDIF
! test

      OCC_MAT_ALL(:,:,:,IATOM)=OCC_MAT(:,:,:)
      
!==============================================================
!        choose lda+u potential type                   
! TYPE1: presently not supported           
! TYPE2: use only  U and J
!==============================================================

      IF (TYPE_POT == 1) THEN 
         CALL POT_TYPE1(LMDIM,IATOM,ITYP,LNMAX,PP%LPS,CRHODE,CDIJ,DBLE_LDAU) 
      ELSEIF (TYPE_POT == 2) THEN  
         CALL POT_TYPE2(LMDIM,IATOM,ITYP,LNMAX,PP%LPS,CRHODE,CDIJ,DBLE_LDAU)
      ELSEIF (TYPE_POT == 3) THEN
         CALL POT_TYPE3(LMDIM,IATOM,ITYP,LNMAX,PP%LPS,CRHODE,CDIJ,DBLE_LDAU)
      ELSEIF (TYPE_POT == 4) THEN 
         CALL POT_TYPE4(LMDIM,IATOM,ITYP,LNMAX,PP%LPS,CRHODE,CDIJ,DBLE_LDAU)
      ELSEIF (TYPE_POT == 5) THEN 
         CALL POT_TYPE5(LMDIM,IATOM,ITYP,LNMAX,PP%LPS,CDIJ,DBLE_LDAU)
      ENDIF

      RETURN
      END SUBROUTINE LDAPLUSU


!**********************************************************************
!
!  calculate the radial integral (ae_phi_i | ae_phi_j) for all 
!  lm(n)-states
!
!**********************************************************************

      SUBROUTINE OVERLAP_AE(R,RDEP,ITYP,NTYP,LNMAX,WAE,L_OF_LN)
      
      USE radial
      
      IMPLICIT NONE

      TYPE (rgrid) R
      
      INTEGER ITYP,NTYP
      INTEGER LNMAX
      REAL(q) :: RDEP
      REAL(q) WAE(:,:)       ! AE partial wavefunctions
      INTEGER L_OF_LN(LNMAX) ! l quantum number per channel
          
! local variables
      INTEGER LN1,LN2
      INTEGER IR,IRMAX 
      INTEGER L1,L2

      REAL(q) RES
      REAL(q) RHOT(R%NMAX)

      DO IR=1,R%NMAX-1
         IF (RDEP>0 .AND.  R%R(IR)-RDEP > -5E-3) EXIT
      ENDDO
!     WRITE(*,*) 'outermost point',R%R(IR),RDEP
      IRMAX=IR

      DO LN1=1,LNMAX
      DO LN2=1,LNMAX
      ! quantum numbers l1 and l2 of these two channels ln1 and ln2
         L1=L_OF_LN(LN1)
         L2=L_OF_LN(LN2)
         IF (L1 /= L2) CYCLE 

!        IRMAX=R%NMAX
         RHOT=0._q
         DO IR=1,IRMAX
            RHOT(IR)=WAE(IR,LN1)*WAE(IR,LN2)
         ENDDO

         CALL SIMPI(R,RHOT,RES)
         OVERLAP_AE_(LN1,LN2,ITYP)=RES
      ENDDO !(LN2)
      ENDDO !(LN1)

      IF (ITYP==NTYP) TFIRST=.FALSE.

      RETURN
      END SUBROUTINE OVERLAP_AE

!**********************************************************************
!
! LDAUTYPE=1
!
!**********************************************************************

      SUBROUTINE POT_TYPE1(LMDIM,IATOM,ITYP,LNMAX,L_OF_LN,CRHODE,CDIJ,DBLE_LDAU)
      
      IMPLICIT NONE
      
      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      INTEGER LNMAX,LMMAX
      INTEGER L_OF_LN(LNMAX)
      
      REAL(q) DBLE_LDAU
      
      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ)
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)
   
      INTEGER LN1,LN2,LMN1,LMN2,LM1,LM2,LM3,LM4
      INTEGER L1,L2,M1,M2,M3,M4
      INTEGER ISP
      
      REAL(q) DC1,DC2,DC3
      REAL(q) M,M_x,M_y,M_z,NEL,NUP,NDW
      
      OVERLAP POT_U(LMDIM,LMDIM,NCDIJ),TMP
      
      LMMAX=(MAXVAL(L_OF_LN(1:LNMAX))+1)**2

      CALL COULOMB_INTERACTION(ITYP)      
            
      DC1=0._q
      POT_U=0._q
            
      LMN1=0
      DO LN1=1,LNMAX
      L1=L_OF_LN(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         IF (L1 /= LANG_LDAPLUSU(ITYP)) CYCLE

         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=L_OF_LN(LN2)
         DO M2=1,2*L2+1
            LMN2=LMN2+1
            IF (L2 /= LANG_LDAPLUSU(ITYP)) CYCLE
           
            LM2=L2**2+M2
         
            DO ISP=1,NCDIJ
               TMP=0._q
               IF (ISP==1.OR.ISP==NCDIJ) THEN
               ! diagonal terms
                  DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM3=LANG_LDAPLUSU(ITYP)**2+M3
                  DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM4=LANG_LDAPLUSU(ITYP)**2+M4
                     ! Hartree terms
                     TMP=TMP+V(M1,M3,M2,M4)*(OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))
                     ! Fock term
                     TMP=TMP-V(M1,M3,M4,M2)*OCC_MAT(LM4,LM3,ISP)
                     ! Terms arising from the definition of the double counting energy
                     ! -U * \Sum_m (\rho^{\alpha\alpha}_mm+\rho^{\beta\beta}_mm)
                     ! +J * \Sum_m (\rho^{isp isp}_mm)
                     IF (LM1==LM2.AND.LM3==LM4) THEN
                        TMP=TMP-U(ITYP)*(OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))+J(ITYP)*OCC_MAT(LM4,LM3,ISP)                   
                     ENDIF
                  ENDDO ! M4
                  ENDDO ! M3
                  ! Remaining terms arising  from the definition of the double counting energy
                  ! +U/2-J/2
                  IF (LM1==LM2) TMP=TMP+U(ITYP)/2-J(ITYP)/2
               ELSE
               ! off-diagonal terms (exist only in noncollinear case)
                  DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM3=LANG_LDAPLUSU(ITYP)**2+M3
                  DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM4=LANG_LDAPLUSU(ITYP)**2+M4
                     ! Fock term 
                     TMP=TMP-V(M1,M3,M4,M2)*OCC_MAT(LM4,LM3,ISP)
                     ! Terms arising from the definition of the double counting energy
                     ! +J * \Sum_m (\rho^{\alpha\beta c.q. \beta\alpha}_mm)
                     IF (LM1==LM2.AND.LM3==LM4) THEN
                        TMP=TMP+J(ITYP)*OCC_MAT(LM4,LM3,ISP)
                     ENDIF
                  ENDDO ! M4
                  ENDDO ! M3                  
               ENDIF

               POT_U(LMN1,LMN2,ISP)=TMP*OVERLAP_AE_(LN1,LN2,ITYP)

               ! To calculate the double counting corrections to
               ! the total energy as calculated from the eigenvalues
#ifndef realmode               
               DC1=DC1+POT_U(LMN1,LMN2,ISP)*CONJG(CRHODE(LMN1,LMN2,ISP))
#else
               DC1=DC1+POT_U(LMN1,LMN2,ISP)*CRHODE(LMN1,LMN2,ISP)
#endif
               
            ENDDO ! ISP
                  
         ENDDO ! M2
         ENDDO ! LN2
      ENDDO ! M1
      ENDDO ! LN1

      CDIJ=CDIJ+POT_U

! test
! #ifndef MPI
!     IF (PRINTOCC>1) THEN
      IF (.FALSE.) THEN
!     IF (IATOM==1) THEN
! test      
         WRITE(*,*) 'potential matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (REAL(POT_U(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)
#ifndef realmode
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (AIMAG(POT_U(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)
#endif
         ENDDO

         WRITE(*,*)
         WRITE(*,*) 'Hamilton matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP         
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (REAL(CDIJ(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)
#ifndef realmode
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (AIMAG(CDIJ(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)   
#endif
         ENDDO
      ENDIF
! test
! #endif
! test

! Calculate the remaining terms necessary to determine the 
! double counting correction to the total energy as calculated
! from the eigenvalues

      DC2=0._q

      DO ISP=1,NCDIJ

      DO M1=1,2*LANG_LDAPLUSU(ITYP)+1
      LM1=LANG_LDAPLUSU(ITYP)**2+M1
      DO M2=1,2*LANG_LDAPLUSU(ITYP)+1
      LM2=LANG_LDAPLUSU(ITYP)**2+M2
         DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
         LM3=LANG_LDAPLUSU(ITYP)**2+M3
         DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
         LM4=LANG_LDAPLUSU(ITYP)**2+M4
      
         IF (ISP==1.OR.ISP==NCDIJ) THEN
         ! diagonal terms
            ! Hartree energy
            DC2=DC2+V(M1,M3,M2,M4)*OCC_MAT(LM2,LM1,ISP)* &
           &          (OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))/2
            ! Fock energy
            DC2=DC2-V(M1,M3,M4,M2)*OCC_MAT(LM2,LM1,ISP)*OCC_MAT(LM4,LM3,ISP)/2
         ELSE
         ! off-diagonal terms
            ! Fock energy
#ifndef realmode
            DC2=DC2-V(M1,M3,M4,M2)*CONJG(OCC_MAT(LM1,LM2,ISP))*OCC_MAT(LM4,LM3,ISP)/2
#endif
         ENDIF
      
         ENDDO ! M4
         ENDDO ! M3
      ENDDO ! M2
      ENDDO ! M1

      ENDDO ! ISP      
      
      DC3=0._q
      NEL=0._q ; M_x=0._q ; M_y=0._q ; M_z=0._q ; M=0._q
      
      DO M1=1,2*LANG_LDAPLUSU(ITYP)+1
      LM1=LANG_LDAPLUSU(ITYP)**2+M1
      
         NEL=NEL+OCC_MAT(LM1,LM1,1)+OCC_MAT(LM1,LM1,NCDIJ)
         M_z=M_z+OCC_MAT(LM1,LM1,1)-OCC_MAT(LM1,LM1,NCDIJ)
      
#ifndef realmode
         IF (NCDIJ==4) THEN
            M_x=M_x+2*REAL(OCC_MAT(LM1,LM1,2))
            M_y=M_y+2*AIMAG(OCC_MAT(LM1,LM1,2))
         ENDIF
#endif
      ENDDO ! M1
      
      M=M_z
      IF (NCDIJ==4) THEN
         M=SQRT(M_x*M_x+M_y*M_y+M_z*M_z)
      ENDIF
      
      NUP=(NEL+M)/2
      NDW=(NEL-M)/2
            
      DC3=U(ITYP)*NEL*(NEL-1)/2-J(ITYP)*(NUP*(NUP-1)+NDW*(NDW-1))/2
      
      DBLE_LDAU=DC2-DC1-DC3

#ifndef MPI            
      IF (PRINTOCC /= 0) THEN
         WRITE(*,*) 'LDA+U energy corrections',IATOM
         WRITE(*,*) '==================================='
         WRITE(*,*) ' E(e-e)    =',DC2
         WRITE(*,*) ' E(LSDA)   =',-DC3
         WRITE(*,*) ' E(eignv.) =',-DC1
         WRITE(*,*) ' DBLE_LDAU =',DBLE_LDAU
      ENDIF
#endif      
      
      RETURN
      END SUBROUTINE POT_TYPE1


!**********************************************************************
!
! LDAUTYPE=4
!
!**********************************************************************

      SUBROUTINE POT_TYPE4(LMDIM,IATOM,ITYP,LNMAX,L_OF_LN,CRHODE,CDIJ,DBLE_LDAU)
      
      IMPLICIT NONE
      
      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      INTEGER LNMAX,LMMAX
      INTEGER L_OF_LN(LNMAX)
      
      REAL(q) DBLE_LDAU
      
      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ)
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)
   
      INTEGER LN1,LN2,LMN1,LMN2,LM1,LM2,LM3,LM4
      INTEGER L1,L2,M1,M2,M3,M4
      INTEGER ISP
      
      REAL(q) DC1,DC2,DC3
      REAL(q) NEL
      
      OVERLAP POT_U(LMDIM,LMDIM,NCDIJ),TMP
      
      LMMAX=(MAXVAL(L_OF_LN(1:LNMAX))+1)**2

      CALL COULOMB_INTERACTION(ITYP)      
            
      DC1=0._q
      POT_U=0._q
            
      LMN1=0
      DO LN1=1,LNMAX
      L1=L_OF_LN(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         IF (L1 /= LANG_LDAPLUSU(ITYP)) CYCLE

         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=L_OF_LN(LN2)
         DO M2=1,2*L2+1
            LMN2=LMN2+1
            IF (L2 /= LANG_LDAPLUSU(ITYP)) CYCLE
           
            LM2=L2**2+M2
         
            DO ISP=1,NCDIJ
               TMP=0._q
               IF (ISP==1.OR.ISP==NCDIJ) THEN
               ! diagonal terms
                  DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM3=LANG_LDAPLUSU(ITYP)**2+M3
                  DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM4=LANG_LDAPLUSU(ITYP)**2+M4
                     ! Hartree terms
                     TMP=TMP+V(M1,M3,M2,M4)*(OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))
                     ! Fock term
                     TMP=TMP-V(M1,M3,M4,M2)*OCC_MAT(LM4,LM3,ISP)
                     ! Terms arising from the definition of the double counting energy
                     ! -U * \Sum_m (\rho^{\alpha\alpha}_mm+\rho^{\beta\beta}_mm)
                     ! +J * \Sum_m (\rho^{isp isp}_mm)
                     IF (LM1==LM2.AND.LM3==LM4) THEN
                        TMP=TMP-(U(ITYP)-J(ITYP)/2)*(OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))                   
                     ENDIF
                  ENDDO ! M4
                  ENDDO ! M3
                  ! Remaining terms arising  from the definition of the double counting energy
                  ! +U/2-J/2
                  IF (LM1==LM2) TMP=TMP+U(ITYP)/2-J(ITYP)/2
               ELSE
               ! off-diagonal terms (exist only in noncollinear case)
                  DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM3=LANG_LDAPLUSU(ITYP)**2+M3
                  DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
                  LM4=LANG_LDAPLUSU(ITYP)**2+M4
                     ! Fock term 
                     TMP=TMP-V(M1,M3,M4,M2)*OCC_MAT(LM4,LM3,ISP)
                  ENDDO ! M4
                  ENDDO ! M3                  
               ENDIF

               POT_U(LMN1,LMN2,ISP)=TMP*OVERLAP_AE_(LN1,LN2,ITYP)

               ! To calculate the double counting corrections to
               ! the total energy as calculated from the eigenvalues
#ifndef realmode               
               DC1=DC1+POT_U(LMN1,LMN2,ISP)*CONJG(CRHODE(LMN1,LMN2,ISP))
#else
               DC1=DC1+POT_U(LMN1,LMN2,ISP)*CRHODE(LMN1,LMN2,ISP)
#endif
               
            ENDDO ! ISP
                  
         ENDDO ! M2
         ENDDO ! LN2
      ENDDO ! M1
      ENDDO ! LN1

      CDIJ=CDIJ+POT_U

#ifndef MPI
      IF (PRINTOCC>1) THEN
         WRITE(*,*) 'potential matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP
         DO LMN1=1,10
            WRITE(*,'(14F12.6)') (REAL(POT_U(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)
#ifndef realmode
         DO LMN1=1,10
            WRITE(*,'(14F12.6)') (AIMAG(POT_U(LMN1,LMN2,ISP)),LMN2=1,10)
         ENDDO
         WRITE(*,*)
#endif
         ENDDO

         WRITE(*,*)
         WRITE(*,*) 'Hamilton matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP         
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (REAL(CDIJ(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
         ENDDO
         WRITE(*,*)
#ifndef realmode
         DO LMN1=1,10
            WRITE(*,'(10F12.6)') (AIMAG(CDIJ(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
         ENDDO
         WRITE(*,*)   
#endif
         ENDDO
      ENDIF
#endif

! Calculate the remaining terms necessary to determine the 
! double counting correction to the total energy as calculated
! from the eigenvalues

      DC2=0._q

      DO ISP=1,NCDIJ

      DO M1=1,2*LANG_LDAPLUSU(ITYP)+1
      LM1=LANG_LDAPLUSU(ITYP)**2+M1
      DO M2=1,2*LANG_LDAPLUSU(ITYP)+1
      LM2=LANG_LDAPLUSU(ITYP)**2+M2
         DO M3=1,2*LANG_LDAPLUSU(ITYP)+1
         LM3=LANG_LDAPLUSU(ITYP)**2+M3
         DO M4=1,2*LANG_LDAPLUSU(ITYP)+1
         LM4=LANG_LDAPLUSU(ITYP)**2+M4
      
         IF (ISP==1.OR.ISP==NCDIJ) THEN
         ! diagonal terms
            ! Hartree energy
            DC2=DC2+V(M1,M3,M2,M4)*OCC_MAT(LM2,LM1,ISP)* &
           &          (OCC_MAT(LM4,LM3,1)+OCC_MAT(LM4,LM3,NCDIJ))/2
            ! Fock energy
            DC2=DC2-V(M1,M3,M4,M2)*OCC_MAT(LM2,LM1,ISP)*OCC_MAT(LM4,LM3,ISP)/2
         ELSE
         ! off-diagonal terms
            ! Fock energy
#ifndef realmode
            DC2=DC2-V(M1,M3,M4,M2)*CONJG(OCC_MAT(LM1,LM2,ISP))*OCC_MAT(LM4,LM3,ISP)/2
#endif
         ENDIF
      
         ENDDO ! M4
         ENDDO ! M3 
      ENDDO ! M2
      ENDDO ! M1

      ENDDO ! ISP      
      
      DC3=0._q
      NEL=0._q
      
      DO M1=1,2*LANG_LDAPLUSU(ITYP)+1
      LM1=LANG_LDAPLUSU(ITYP)**2+M1
          NEL=NEL+OCC_MAT(LM1,LM1,1)+OCC_MAT(LM1,LM1,NCDIJ)
      ENDDO ! M1
      
      DC3=U(ITYP)*NEL*(NEL-1)/2-J(ITYP)*NEL*(NEL-2)/4
      
      DBLE_LDAU=DC2-DC1-DC3

#ifndef MPI            
      IF (PRINTOCC /= 0) THEN
         WRITE(*,*) 'LDA+U energy corrections',IATOM
         WRITE(*,*) '==================================='
         WRITE(*,*) ' E(e-e)    =',DC2
         WRITE(*,*) ' E(LSDA)   =',-DC3
         WRITE(*,*) ' E(eignv.) =',-DC1
         WRITE(*,*) ' DBLE_LDAU =',DBLE_LDAU
      ENDIF
#endif

      RETURN
      END SUBROUTINE POT_TYPE4
      
                                                             
!**********************************************************************
!
! LDAUTYPE=2
! simpler version using the same U and J for all M see
!
! Dudarev, Botton, Savrasov Humphreys and Sutton, PRB 57 1505
!
!**********************************************************************


      SUBROUTINE POT_TYPE2(LMDIM,IATOM,ITYP,LNMAX,L_OF_LN,CRHODE,CDIJ,DBLE_LDAU)

      IMPLICIT NONE

      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      INTEGER LNMAX,LMMAX
      INTEGER L_OF_LN(LNMAX)

      REAL(q) DBLE_LDAU

      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ)
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)

      INTEGER LN1,LN2,LMN1,LMN2,LM1,LM2,LMSTART,LMSTOP
      INTEGER L1,L2,M1,M2
      INTEGER ISP
      
      REAL(q) DC,DC_

      OVERLAP POT_U(LMDIM,LMDIM,NCDIJ),TMP

!=======================================================================

      LMMAX=(MAXVAL(L_OF_LN(1:LNMAX))+1)**2
                                                           
!-----------------------------------------------------------------------
!    loop of potential V(LMN1,LMN2)
!-----------------------------------------------------------------------

      DC=0._q
      POT_U= 0._q

      LMN1=0
      DO LN1=1,LNMAX
      L1=L_OF_LN(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         IF (L1 /= LANG_LDAPLUSU(ITYP)) CYCLE

         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=L_OF_LN(LN2)
         DO M2=1,2*L2+1
            LMN2=LMN2+1
            IF (L2 /= LANG_LDAPLUSU(ITYP)) CYCLE
           
            LM2=L2**2+M2
            
            DO ISP=1,NCDIJ                                                       
               TMP= -1._q*(U(ITYP)-J(ITYP))*OCC_MAT(LM1,LM2,ISP)

               IF (M1==M2.AND.(ISP==1.OR.ISP==NCDIJ)) THEN
                     TMP=TMP+(U(ITYP)-J(ITYP))*0.5_q
               ENDIF

               POT_U(LMN1,LMN2,ISP) = TMP*OVERLAP_AE_(LN1,LN2,ITYP)

               ! double counting correction 
#ifndef realmode
               DC=DC+POT_U(LMN1,LMN2,ISP)*CONJG(CRHODE(LMN1,LMN2,ISP))
#else
               DC=DC+POT_U(LMN1,LMN2,ISP)*CRHODE(LMN2,LMN1,ISP)
#endif
            ENDDO !(ISP)
         ENDDO !(M2)
         ENDDO !(LN2)
      ENDDO !(M1)
      ENDDO !(LN1)
      
!=======================================================================
! add LDA+U correction to hamiltonian CDIJ
!=======================================================================
                                                          
      CDIJ=CDIJ+POT_U

!=======================================================================
! LDA+U double counting corrections
! Eq. (7) of  Dudarev PRB 57, 1505
!=======================================================================
                                                            
!-----------------------------------------------------------------------
! LDA+U - LSDA contribution    
!-----------------------------------------------------------------------

      DC_=0._q
      
      LMSTART=LANG_LDAPLUSU(ITYP)**2+1
      LMSTOP=(LANG_LDAPLUSU(ITYP)+1)**2

      DO ISP=1,NCDIJ
      DO LM1=LMSTART,LMSTOP
      DO LM2=LMSTART,LMSTOP
#ifndef realmode
         DC_=DC_-0.5_q*(U(ITYP)-J(ITYP))*OCC_MAT(LM1,LM2,ISP)*CONJG(OCC_MAT(LM1,LM2,ISP))            
#else
         DC_=DC_-0.5_q*(U(ITYP)-J(ITYP))*OCC_MAT(LM1,LM2,ISP)*OCC_MAT(LM2,LM1,ISP)            
#endif
         IF (LM1==LM2.AND.(ISP==1.OR.ISP==NCDIJ)) THEN
            DC_=DC_+0.5_q*(U(ITYP)-J(ITYP))*OCC_MAT(LM1,LM1,ISP)
         ENDIF
      ENDDO !(LM1)
      ENDDO !(LM2)
      ENDDO !(ISP)
     
!-----------------------------------------------------------------------
      DBLE_LDAU=DC_-DC

#ifndef MPI
      IF (PRINTOCC /= 0) THEN
         WRITE(*,*) 'LDA+U double counting'
         WRITE(*,*) ' E(e-e)-E(LSDA) =',DC_
         WRITE(*,*) ' E(DC)          =',-DC
         WRITE(*,*) ' DBLE_LDAU      =',DBLE_LDAU
      ENDIF
#endif
      RETURN
      END SUBROUTINE POT_TYPE2


!**********************************************************************
!
! LDAUTYPE=3
!
!**********************************************************************

      SUBROUTINE POT_TYPE3(LMDIM,IATOM,ITYP,LNMAX,L_OF_LN,CRHODE,CDIJ,DBLE_LDAU)
      
      IMPLICIT NONE

      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      INTEGER LNMAX,LMMAX
      INTEGER L_OF_LN(LNMAX)

      REAL(q) DBLE_LDAU

      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ)
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)

      INTEGER LN1,LN2,LMN1,LMN2,LM1,LM2,LMSTART,LMSTOP
      INTEGER L1,L2,M1,M2
      INTEGER ISP
      
      REAL(q) DC
      REAL(q) M,M_x,M_y,M_z,NEL,NUP,NDW
      REAL(q) V(NCDIJ)

      OVERLAP POT_U(LMDIM,LMDIM,NCDIJ)

      DC=0._q
      POT_U=0._q
      
      V=0._q
      V(1)=U(ITYP)
      IF (NCDIJ/=1) V(NCDIJ)=J(ITYP)
      
      LMN1=0
      DO LN1=1,LNMAX
      L1=L_OF_LN(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         IF (L1 /= LANG_LDAPLUSU(ITYP)) CYCLE

         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=L_OF_LN(LN2)
         DO M2=1,2*L2+1
            LMN2=LMN2+1
            IF (L2 /= LANG_LDAPLUSU(ITYP)) CYCLE
           
            LM2=L2**2+M2
            
            DO ISP=1,NCDIJ,NCDIJ-1                                                       
               IF (L1==L2.AND.M1==M2) THEN
                  POT_U(LMN1,LMN2,ISP) = -1._q*V(ISP)*OVERLAP_AE_(LN1,LN2,ITYP)
                  ! energy correction 
#ifndef realmode
                  DC=DC+POT_U(LMN1,LMN2,ISP)*CONJG(CRHODE(LMN1,LMN2,ISP))
#else
                  DC=DC+POT_U(LMN1,LMN2,ISP)*CRHODE(LMN2,LMN1,ISP)
#endif
               ENDIF
            ENDDO !(ISP)
         ENDDO !(M2)
         ENDDO !(LN2)
      ENDDO !(M1)
      ENDDO !(LN1)

      CDIJ=CDIJ+POT_U
!=======================================================================
!   test printout                    
!=======================================================================
#ifndef MPI
      IF (PRINTOCC>1) THEN
         WRITE(*,*) 'potential matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP
         DO LMN1=1,LMDIM
            WRITE(*,*) 'lmn',LMN1
#ifndef realmode
            WRITE(*,*) 'real part'
#endif
            WRITE(*,'(20F8.2)') (REAL(POT_U(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
#ifndef realmode
            WRITE(*,*) 'imaginary part'
            WRITE(*,'(20F8.2)') (AIMAG(POT_U(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
#endif
         ENDDO
         WRITE(*,*)
         ENDDO

         WRITE(*,*)
         WRITE(*,*)
         WRITE(*,*) 'Hamilton matrix for atom',IATOM
         DO ISP=1,NCDIJ
         WRITE(*,*) 'component',ISP         
         DO LMN1=1,LMDIM
            WRITE(*,*) 'lmn',LMN1
#ifndef realmode
            WRITE(*,*) 'real part'
#endif
            WRITE(*,'(20F8.2)') (REAL(CDIJ(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
#ifndef realmode
            WRITE(*,*) 'imaginary part'
            WRITE(*,'(20F8.2)') (AIMAG(CDIJ(LMN1,LMN2,ISP)),LMN2=1,LMDIM)
#endif
         ENDDO
         WRITE(*,*)   
         ENDDO
      ENDIF
#endif            
!     DBLE_LDAU=-DC
      DBLE_LDAU=0._q

      NEL=0._q ; M_x=0._q ; M_y=0._q ; M_z=0._q ; M=0._q
      
      DO M1=1,2*LANG_LDAPLUSU(ITYP)+1
      LM1=LANG_LDAPLUSU(ITYP)**2+M1
      
         NEL=NEL+OCC_MAT(LM1,LM1,1)+OCC_MAT(LM1,LM1,NCDIJ)
         M_z=M_z+OCC_MAT(LM1,LM1,1)-OCC_MAT(LM1,LM1,NCDIJ)
      
#ifndef realmode
         IF (NCDIJ==4) THEN
            M_x=M_x+2*REAL(OCC_MAT(LM1,LM1,2))
            M_y=M_y+2*AIMAG(OCC_MAT(LM1,LM1,2))
         ENDIF
#endif
      ENDDO ! M1
      
      M=M_z
      IF (NCDIJ==4) THEN
         M=SQRT(M_x*M_x+M_y*M_y+M_z*M_z)
      ENDIF
      
      NUP=(NEL+M)/2
      NDW=(NEL-M)/2

#ifndef MPI
      IF (PRINTOCC /= 0) THEN
         WRITE(*,*) 'LDA+U double counting'
!        WRITE(*,*) ' DBLE_LDAU           =',DBLE_LDAU
         WRITE(*,*)
         WRITE(*,*) ' number of electrons =',NEL
         WRITE(*,*) ' magnetization       =',M
      ENDIF
#endif

      RETURN
      END SUBROUTINE POT_TYPE3
      
!**********************************************************************
!
! LDAUTYPE=5, Orbital polarization only 
!
!**********************************************************************

      SUBROUTINE POT_TYPE5(LMDIM,IATOM,ITYP,LNMAX,L_OF_LN,CDIJ,DBLE_LDAU)

      USE relativistic
      
      IMPLICIT NONE
      
      INTEGER LMDIM
      INTEGER IATOM
      INTEGER ITYP
      INTEGER LNMAX,LMMAX
      INTEGER L_OF_LN(LNMAX)
      
      REAL(q) DBLE_LDAU
      
      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ)
      OVERLAP CDIJ(LMDIM,LMDIM,NCDIJ)
   
      INTEGER LN1,LN2,LMN1,LMN2
      INTEGER L,L1,L2,M1,M2
      INTEGER ISP,I
      
      REAL(q) L_MOM(3),S_MOM(3),SNORM,L_PROJ
      
      COMPLEX(q) , ALLOCATABLE :: L_OP(:,:,:),DUMMY(:,:,:)
      COMPLEX(q) , ALLOCATABLE :: L_OP_PROJ(:,:)
            
      OVERLAP POT_U(LMDIM,LMDIM,NCDIJ)
      
! Get me the L operator
      L=LANG_LDAPLUSU(ITYP)
      ALLOCATE(L_OP(2*L+1,2*L+1,3),DUMMY(2*L+1,2*L+1,4))
      L_OP=(0._q,0._q)
      CALL SETUP_LS(L,0._q,0._q,L_OP(1:2*L+1,1:2*L+1,1:3),DUMMY(1:2*L+1,1:2*L+1,1:4))
! Calculate orbital moment
      L_MOM=0._q
      DO ISP=1,NCDIJ,NCDIJ-1
      DO I=1,3
         DO M1=1,2*L+1
         DO M2=1,2*L+1
         ! beware of the reversed storage convention of OCC_MAT
            L_MOM(I)=L_MOM(I)+L_OP(M1,M2,I)*OCC_MAT(L*L+M1,L*L+M2,ISP)
         ENDDO
         ENDDO
      ENDDO
      ENDDO      
! Calculate spin moment
      S_MOM=0._q
      DO M1=1,2*L+1
         S_MOM(1)=S_MOM(1)+(OCC_MAT(L*L+M1,L*L+M1,3)+OCC_MAT(L*L+M1,L*L+M1,2))
         S_MOM(2)=S_MOM(2)+(OCC_MAT(L*L+M1,L*L+M1,2)-OCC_MAT(L*L+M1,L*L+M1,3))*(0._q,1._q)
         S_MOM(3)=S_MOM(3)+(OCC_MAT(L*L+M1,L*L+M1,1)-OCC_MAT(L*L+M1,L*L+M1,4))
      ENDDO
      SNORM=SQRT(S_MOM(1)*S_MOM(1)+S_MOM(2)*S_MOM(2)+S_MOM(3)*S_MOM(3))
! Calculate projection of orbital moment on local spin quantization axis
      L_PROJ=(L_MOM(1)*S_MOM(1)+L_MOM(2)*S_MOM(2)+L_MOM(3)*S_MOM(3))/SNORM      
! Construct L operator along spin direction
      ALLOCATE(L_OP_PROJ(2*L+1,2*L+1))
      L_OP_PROJ=(0._q,0._q)
      DO M1=1,2*L+1
      DO M2=1,2*L+1
         L_OP_PROJ(M1,M2)=L_OP_PROJ(M1,M2)- &
        &   (L_OP(M1,M2,1)*S_MOM(1)+L_OP(M1,M2,2)*S_MOM(2)+L_OP(M1,M2,3)*S_MOM(3))/SNORM
      ENDDO
      ENDDO
! Construct spin polarizing potential
      POT_U=0._q
            
      LMN1=0
      DO LN1=1,LNMAX
      L1=L_OF_LN(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         IF (L1 /= LANG_LDAPLUSU(ITYP)) CYCLE

         LMN2=0
         DO LN2=1,LNMAX
         L2=L_OF_LN(LN2)
         DO M2=1,2*L2+1
            LMN2=LMN2+1
            IF (L2 /= LANG_LDAPLUSU(ITYP)) CYCLE
           
            DO ISP=1,NCDIJ,NCDIJ-1

               POT_U(LMN1,LMN2,ISP)=U(ITYP)*L_PROJ*L_OP_PROJ(M2,M1)*OVERLAP_AE_(LN1,LN2,ITYP)

            ENDDO ! ISP
                  
         ENDDO ! M2
         ENDDO ! LN2
      ENDDO ! M1
      ENDDO ! LN1

      CDIJ=CDIJ+POT_U
      
      DBLE_LDAU=0._q

      DEALLOCATE(L_OP,DUMMY,L_OP_PROJ)
      
      RETURN
      END SUBROUTINE POT_TYPE5

      
      SUBROUTINE COULOMB_INTERACTION(ITYP)
      
      USE prec
      
      IMPLICIT NONE
      
      INTEGER ITYP
      INTEGER L,M,M_
      INTEGER M1,M2,M3,M4,K,MA,MB,MC,MD
      
      REAL(q) RAT42,RAT62
      COMPLEX(q) PREFAC

      REAL(q), ALLOCATABLE:: CGMAT(:,:,:),F(:)
      COMPLEX(q), ALLOCATABLE :: U_R2C(:,:),U_(:)

      L=LANG_LDAPLUSU(ITYP)

      ALLOCATE(U_R2C(2*L+1,2*L+1))      
      U_R2C=(0._q,0._q)          
      DO M=1,2*L+1
         M_=M-L-1
         IF (M_>0) THEN
            U_R2C( M_+L+1,M)=(-1)**M_/SQRT(2._q)
            U_R2C(-M_+L+1,M)=1/SQRT(2._q)
         ENDIF
         IF (M_==0) THEN
            U_R2C(L+1,L+1)=1
         ENDIF
         IF (M_<0) THEN
            U_R2C( M_+L+1,M)= CMPLX(0,1/SQRT(2._q),q)
            U_R2C(-M_+L+1,M)=-CMPLX(0,(-1)**M_/SQRT(2._q),q)
         ENDIF
      ENDDO
      
!     u_r2c=(0._q,0._q)
!     do m=1,2*L+1
!        u_r2c(m,m)=1._q
!     enddo
            
      ALLOCATE(CGMAT(0:2*L,-L:L,-L:L))
      CGMAT=0._q
      CALL GAUNT_L(CGMAT,L)
      
      ALLOCATE(F(0:2*L))
      F=0._q
      SELECT CASE(L)
      CASE(1)
         F(0)=U(ITYP)
         F(2)=5*J(ITYP)
      CASE(2)
         RAT42=0.625_q
         F(0)=U(ITYP)
         F(2)=14._q/(1+RAT42)*J(ITYP)
         F(4)=RAT42*F(2)
      CASE(3)
         RAT42=0.668_q
         RAT62=0.494_q
         F(0)=U(ITYP)
         F(2)=6435._q/(286._q+195._q*RAT42+250._q*RAT62)*J(ITYP)
         F(4)=RAT42*F(2)
         F(6)=RAT62*F(2)
      END SELECT

      ALLOCATE(U_(0:2*L))      
      V=(0._q,0._q)

      DO M1=-L,L
      DO M3=-L,L
      DO M2=-L,L
      DO M4=-L,L
         U_=(0._q,0._q)

         DO MA=-L,L
         DO MB=-L,L
         DO MC=-L,L
         DO MD=-L,L
            IF ((MA-MB)/=(MD-MC)) CYCLE
            PREFAC = CONJG(U_R2C(MA+L+1,M1+L+1))*CONJG(U_R2C(MC+L+1,M3+L+1)) &
                   * U_R2C(MB+L+1,M2+L+1) * U_R2C(MD+L+1,M4+L+1)
            DO K=0,2*L,2
               U_(K)=U_(K) + PREFAC * F(K)*CGMAT(K,MA,MB)*CGMAT(K,MD,MC)
            ENDDO
         ENDDO !(MD)
         ENDDO !(MC)
         ENDDO !(MB)
         ENDDO !(MA)
         
         V(M1+L+1,M3+L+1,M2+L+1,M4+L+1)=SUM(U_(0:2*L))
     
      ENDDO !(M4)
      ENDDO !(M3)
      ENDDO !(M2)
      ENDDO !(M1)
      
      DEALLOCATE(U_R2C,CGMAT,F,U_)              
      
      END SUBROUTINE COULOMB_INTERACTION


      SUBROUTINE GAUNT_L(CGMAT,LMAX)

      USE prec
      USE constant
      USE asa
  
      IMPLICIT REAL(q) (A-H,O-Z)
      IMPLICIT INTEGER (I-N)

      INTEGER LMAX
      REAL(q) CGMAT(0:2*LMAX,-LMAX:LMAX,-LMAX:LMAX)

      REAL(q) FAC(40)
      REAL(q), PARAMETER :: SRPI =1.772453850905516027_q

      CGMAT = 0

!---------------------------------------------------------------------
! set up table for factorials
!---------------------------------------------------------------------
      IMAX=30
      FAC(1)=1._q
      DO I=1,IMAX
         FAC(I+1)= I*FAC(I)
      ENDDO
  
!---------------------------------------------------------------------
! loop over l    ,m     m =-l,+l
! loop over lp<=l,mp    mp=-lp,+lp
!---------------------------------------------------------------------
      L1=LMAX
      L2=LMAX
      K2=(2*L1+1)*(2*L2+1)

      DO M1=-L1,L1
      DO M2=-L2,L2

         M3=M1+M2
!---------------------------------------------------------------------
! loop over L given by triangular rule
!---------------------------------------------------------------------
         Q1= SQRT( REAL(K2,KIND=q)/4 )*FS(M3)

         DO L3=ABS(L1-L2),L1+L2, 2

            IF(ABS(M3)>L3) CYCLE

            T =CLEBGO(FAC(1),L1,L2,L3, M1, M2, M3)
            T0=CLEBG0(FAC(1),L1,L2,L3)
           
            CGMAT(L3,-M2,M1)= T*T0* Q1 /(SRPI* SQRT( REAL(2*L3+1, KIND=q))) &
                         *SQRT( 4 *pi)/SQRT(2.*L3+1)*FS(M2)
            ! last line rescales to maintain compatibility
         ENDDO
      ENDDO
      ENDDO

      CONTAINS

        FUNCTION FS(I)
          INTEGER I,FS
          FS=1-2*MOD(I+20,2)
        END FUNCTION FS


      END SUBROUTINE GAUNT_L


      SUBROUTINE LDAPLUSU_PRINTOCC(WDES,NIONS,ITYP,IU6)

      USE wave
      USE main_mpi
      USE mpimy

      IMPLICIT NONE

      TYPE(wavedes) WDES

      INTEGER IU6
      INTEGER NIONS
      INTEGER I,ITYP(NIONS),ISP
      INTEGER LM1,LM2,LMSTART,LMSTOP
      
      INTEGER L,M,INFO,J,K
      REAL(q), ALLOCATABLE :: W(:),RWORK(:)
      COMPLEX(q), ALLOCATABLE :: TMP(:,:),WORK(:)

#ifdef realmode
      CALLMPI( M_sum_d( WDES%COMM, OCC_MAT_ALL, LMMAX_*LMMAX_*NCDIJ*NIONS))
#else
      CALLMPI( M_sum_d( WDES%COMM, OCC_MAT_ALL, 2*LMMAX_*LMMAX_*NCDIJ*NIONS))
#endif
            
      IF (PRINTOCC /= 0 .AND. IU6>=0) THEN
         DO I=1,NIONS
            IF (LANG_LDAPLUSU(ITYP(I))<0) CYCLE

            LMSTART=LANG_LDAPLUSU(ITYP(I))**2+1
            LMSTOP=(LANG_LDAPLUSU(ITYP(I))+1)**2

            WRITE(IU6,10) I,ITYP(I),LANG_LDAPLUSU(ITYP(I))
            WRITE(IU6,*)
            WRITE(IU6,*) 'onsite density matrix'
            WRITE(IU6,*)
            DO ISP=1,NCDIJ
               WRITE(IU6,20) ISP
               WRITE(IU6,*)
               DO LM1=LMSTART,LMSTOP
               SELECT CASE(LANG_LDAPLUSU(ITYP(I)))
               CASE(0)
#ifdef realmode   
                  WRITE(IU6,'(F8.4)') (OCC_MAT_ALL(LM1,LM2,ISP,I), LM2=LMSTART,LMSTOP)
#else
                  WRITE(IU6,'(F8.4,4X,F8.4)')  (REAL(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP), &
                 &                              (AIMAG(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP)
#endif
               CASE(1)
#ifdef realmode   
                  WRITE(IU6,'(3F8.4)') (OCC_MAT_ALL(LM1,LM2,ISP,I), LM2=LMSTART,LMSTOP)
#else
                  WRITE(IU6,'(3F8.4,4X,3F8.4)')  (REAL(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP), &
                 &                                (AIMAG(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP)
#endif
               CASE(2)
#ifdef realmode
                  WRITE(IU6,'(5F8.4)') (OCC_MAT_ALL(LM1,LM2,ISP,I), LM2=LMSTART,LMSTOP)
#else
                  WRITE(IU6,'(5F8.4,4X,5F8.4)')  (REAL(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP), &
                 &                                (AIMAG(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP)
#endif               
               CASE(3)
#ifdef realmode   
                  WRITE(IU6,'(7F8.4)') (OCC_MAT_ALL(LM1,LM2,ISP,I), LM2=LMSTART,LMSTOP)
#else
                  WRITE(IU6,'(7F8.4,4X,7F8.4)')  (REAL(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP), &
                 &                                (AIMAG(OCC_MAT_ALL(LM1,LM2,ISP,I)), LM2=LMSTART,LMSTOP)
#endif               
               END SELECT
               ENDDO
               WRITE(IU6,*)
            ENDDO
            
            L=LANG_LDAPLUSU(ITYP(I))
            M=L*2+1
            
            ALLOCATE(TMP(2*M,2*M),W(2*M),WORK(MAX(1,4*M-1)),RWORK(MAX(1,6*M-2)))
            
            TMP(1:M,1:M)         = OCC_MAT_ALL(L*L+1:L*L+M,L*L+1:L*L+M,1,I)
            TMP(M+1:2*M,M+1:2*M) = OCC_MAT_ALL(L*L+1:L*L+M,L*L+1:L*L+M,NCDIJ,I)
            TMP(1:M,M+1:2*M)     = 0._q
            TMP(M+1:2*M,1:M)     = 0._q

            IF (NCDIJ==4) THEN
            TMP(1:M,M+1:2*M)     = OCC_MAT_ALL(L*L+1:L*L+M,L*L+1:L*L+M,3,I)
            TMP(M+1:2*M,1:M)     = OCC_MAT_ALL(L*L+1:L*L+M,L*L+1:L*L+M,2,I)
            ENDIF

            TMP=CONJG(TMP)

            CALL ZHEEV('V','L',2*M,TMP(1,1),2*M,W,WORK,MAX(1,4*M-1),RWORK,INFO)
            
            WRITE(IU6,*) 'occupancies and eigenvectors'
            WRITE(IU6,*)
            DO LM1=1,2*M
               SELECT CASE(LANG_LDAPLUSU(ITYP(I)))
               CASE(0)
                 WRITE(IU6,'("  o =",F8.4,"  v =",2F8.4,4X,2F8.4)') &
                 &       W(LM1),(REAL(TMP(LM2,LM1)), LM2=1,2*M),(AIMAG(TMP(LM2,LM1)), LM2=1,2*M)
               CASE(1)
                  WRITE(IU6,'("  o =",F8.4,"  v =",6F8.4,4X,6F8.4)') &
                 &       W(LM1),(REAL(TMP(LM2,LM1)), LM2=1,2*M),(AIMAG(TMP(LM2,LM1)), LM2=1,2*M)
               CASE(2)
                  WRITE(IU6,'("  o =",F8.4,"  v =",10F8.4,4X,10F8.4)') &
                 &       W(LM1),(REAL(TMP(LM2,LM1)), LM2=1,2*M),(AIMAG(TMP(LM2,LM1)), LM2=1,2*M)
               CASE(3)
                  WRITE(IU6,'("  o =",F8.4,"  v =",14F8.4,4X,14F8.4)') &
                 &       W(LM1),(REAL(TMP(LM2,LM1)), LM2=1,2*M),(AIMAG(TMP(LM2,LM1)), LM2=1,2*M)
               END SELECT
            ENDDO
            
            DEALLOCATE(TMP,W,WORK,RWORK)
            
         ENDDO      
      ENDIF

      OCC_MAT_ALL=0._q

   10 FORMAT('atom =',I4,'  type =',I3,'  l =',I2)
   20 FORMAT('spin component ',I2)

      RETURN
      END SUBROUTINE LDAPLUSU_PRINTOCC

      


!***********************************************************************
!
!***********************************************************************

      SUBROUTINE CALC_ORBITAL_MOMENT(LMDIM,IATOM,ITYP,CRHODE,PP,THETA,PHI)
      
      USE pseudo
      USE relativistic
      
      IMPLICIT NONE
      
      TYPE(potcar) PP
      
      INTEGER LMDIM,IATOM
      
      INTEGER LNMAX,LMAX,LMMAX
      INTEGER L,L1,M1,L2,M2,LM1,LM2,LMN1,LMN2,LN1,LN2
      INTEGER I,ISP,ITYP
      
      REAL(q) THETA,PHI
      
      COMPLEX(q), ALLOCATABLE :: L_OP(:,:,:,:)
      COMPLEX(q), ALLOCATABLE :: DUMMY(:,:,:,:)

      OVERLAP CRHODE(LMDIM,LMDIM,NCDIJ) ! occupancy matrix

! quick return
      IF (LMAX_==0) RETURN 

! Calculate the occupation matrix
      OCC_MAT=0._q

      LNMAX=PP%LMAX
      LMAX=MAXVAL(PP%LPS(1:LNMAX))
      LMMAX=(MAXVAL(PP%LPS(1:LNMAX))+1)**2

      LMN1=0
      DO LN1=1,LNMAX
      L1=PP%LPS(LN1)
      DO M1=1,2*L1+1
         LMN1=LMN1+1
         LM1=L1**2+M1
         LMN2=0
         DO LN2=1,LNMAX
         L2=PP%LPS(LN2)
         DO M2=1,2*L2+1
         LMN2=LMN2+1
         LM2=L2**2+M2
         DO ISP=1,NCDIJ
            OCC_MAT(LM1,LM2,ISP)=OCC_MAT(LM1,LM2,ISP) + &
               CRHODE(LMN1,LMN2,ISP)*OVERLAP_AE_(LN1,LN2,ITYP)
         ENDDO !(ISPIN)
         ENDDO !(M2)
         ENDDO !(LN2)
      ENDDO !(M1)
      ENDDO !(LN1)
      
! Get me the L operator
      ALLOCATE(L_OP(2*LMAX+1,2*LMAX+1,3,0:LMAX),DUMMY(2*LMAX+1,2*LMAX+1,4,0:LMAX))
           
      L_OP=(0._q,0._q)      
      
      DO L=1,LMAX
         CALL SETUP_LS(L,THETA,PHI,L_OP(1:2*L+1,1:2*L+1,1:3,L),DUMMY(1:2*L+1,1:2*L+1,1:4,L))
      ENDDO

! Calculate orbital moments
      ORBMOM(:,:,IATOM)=0
#ifndef realmode
      DO ISP=1,NCDIJ,NCDIJ-1
      DO I=1,3
      DO L=1,LMAX
         DO M1=1,2*L+1
         DO M2=1,2*L+1
            ORBMOM(L,I,IATOM)=ORBMOM(L,I,IATOM)+L_OP(M2,M1,I,L)*CONJG(OCC_MAT(L*L+M1,L*L+M2,ISP))
!           ORBMOM(L,I,IATOM)=ORBMOM(L,I,IATOM)+L_OP(M1,M2,I,L)*OCC_MAT(L*L+M1,L*L+M2,ISP)
         ENDDO
         ENDDO
      ENDDO
      ENDDO
      ENDDO
#endif
      
      DEALLOCATE(L_OP,DUMMY)
      
      RETURN
      END SUBROUTINE CALC_ORBITAL_MOMENT

      
!***********************************************************************
!
!***********************************************************************
      
      SUBROUTINE WRITE_ORBITAL_MOMENT(WDES,NIONS,IU6)
      
      USE wave
      USE main_mpi
      USE mpimy
      
      IMPLICIT NONE

      TYPE(wavedes) WDES
      
      INTEGER IU6
      INTEGER NIONS
      INTEGER I,N,L

! quick return
      IF (LMAX_==0) RETURN 

      CALLMPI( M_sum_d( WDES%COMM, ORBMOM, LMAX_*NIONS*3))

      IF (IU6>=0) THEN
      
      DO I=1,3
      WRITE(IU6,*)
      SELECT CASE (I)
      CASE (1)
         WRITE(IU6,'(//A19)') ' orbital moment (x)'
      CASE (2)
         WRITE(IU6,'(//A19)') ' orbital moment (y)'
      CASE (3)
         WRITE(IU6,'(//A19)') ' orbital moment (z)'
      END SELECT
      WRITE(IU6,*)

      SELECT CASE (LMAX_)
      CASE (1)
         WRITE(IU6,*) '# of ion     p       tot'
         WRITE(IU6,*) '--------------------------------'
         DO N=1,NIONS
            WRITE(IU6,'(I3,6X,2(F8.3))') N,ORBMOM(1:LMAX_,I,N),SUM(ORBMOM(1:LMAX_,I,N))         
         ENDDO
         WRITE(IU6,*) '--------------------------------'
         WRITE(IU6,'(9X,2(F8.3))') (SUM(ORBMOM(L,I,1:NIONS)), L=1,LMAX_),SUM(ORBMOM(1:LMAX_,I,1:NIONS))
      CASE (2)
         WRITE(IU6,*) '# of ion     p       d       tot'
         WRITE(IU6,*) '----------------------------------------'
         DO N=1,NIONS
            WRITE(IU6,'(I3,6X,3(F8.3))') N,ORBMOM(1:LMAX_,I,N),SUM(ORBMOM(1:LMAX_,I,N))         
         ENDDO
         WRITE(IU6,*) '----------------------------------------'
         WRITE(IU6,'(9X,3(F8.3))') (SUM(ORBMOM(L,I,1:NIONS)), L=1,LMAX_),SUM(ORBMOM(1:LMAX_,I,1:NIONS))
      CASE (3)
         WRITE(IU6,*) '# of ion     p       d       f       tot'
         WRITE(IU6,*) '------------------------------------------------'
         DO N=1,NIONS
            WRITE(IU6,'(I3,6X,4(F8.3))') N,ORBMOM(1:LMAX_,I,N),SUM(ORBMOM(1:LMAX_,I,N))         
         ENDDO
         WRITE(IU6,*) '------------------------------------------------'
         WRITE(IU6,'(9X,4(F8.3))') (SUM(ORBMOM(L,I,1:NIONS)), L=1,LMAX_),SUM(ORBMOM(1:LMAX_,I,1:NIONS))
      END SELECT

      ENDDO
      ENDIF
      
      ORBMOM=0._q
      
      RETURN
      END SUBROUTINE WRITE_ORBITAL_MOMENT
      
      END MODULE LDAPLUSU_MODULE

!**********************************************************************
!
! query function which returns .TRUE. or .FALSE. depending on
! whether LDA plus U is used or not
!
!**********************************************************************

      FUNCTION L_NO_LSDA_GLOBAL()
      USE LDAPLUSU_MODULE
      LOGICAL L_NO_LSDA_GLOBAL

      L_NO_LSDA_GLOBAL=L_NO_LSDA()

      END FUNCTION L_NO_LSDA_GLOBAL

