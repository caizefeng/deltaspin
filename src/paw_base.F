!#define debug
#include "symbol.inc"
!*******************************************************************
! RCS:  $Id: paw.F,v 1.14 2003/06/27 13:22:22 kresse Exp kresse $
!
! PAW base -module
!  implements the low level PAW functions
! most of the functionality on the logarithmic grid 
!  is implemented in the MODULES radial
! the symmetry routines can be found in the file pawsym.F
! 
!  all routine written by Georg Kresse
!
!*******************************************************************
  MODULE paw
    USE prec
    USE pseudo
      ! LMAX_MIX specifies the maximal L component mixed in the
      ! broyden mixer, the number can be customized by the users
      ! set to a large value if all components should be mixed,
      ! it seems the L=2 is sufficient, and acutally even faster
      ! than mixing all L
      INTEGER  :: SET_LMAX_MIX_TO=2

      INTEGER LMAX_MIX   ! this is the maximum L mixed by the Broyden mixer
      INTEGER IONS_LOCAL ! here we store the number of ions treated locally
                         ! in PAW
      LOGICAL, ALLOCATABLE :: DO_LOCAL(:)
      REAL(q), ALLOCATABLE :: METRIC(:)
      LOGICAL MIMIC_US
      REAL(q)       DOUBLEC_PS_ATOM,DOUBLEC_AE_ATOM

    CONTAINS

!*******************************************************************
!
! add the array CDIJ to the pseudpotential strenght parameters
! in the potential database structure PP
! this is somewhat execptional but required to properly
! handle changed core-valence interactions without too much effort
!
!*******************************************************************


    SUBROUTINE ADD_CDIJ(CDIJ, PP)
      USE prec
      USE pseudo
      IMPLICIT NONE
      OVERLAP CDIJ(:,:)

      TYPE (potcar),POINTER:: PP
      INTEGER LM, LL, LHI, LOW, MMAX, L, LP, M

!      CALL DUMP_DLLMM('CDIJ',CDIJ,PP)
      LOW=1
      LM =1
      block: DO
        LL=PP%LPS(LOW)
      ! search block with same L
        DO LHI=LOW,PP%LMAX
           IF (LL/=PP%LPS(LHI)) EXIT
        ENDDO
        LHI=LHI-1
        MMAX=2*LL+1

        DO L =LOW,LHI
        DO LP=LOW,LHI
        M =0
           PP%DION(L,LP)=PP%DION(L,LP)+CDIJ(LM+(L-LOW)*MMAX,LM+(LP-LOW)*MMAX)
        ENDDO
        ENDDO

      ! set new LOW value and LM value and go on
         LM=LM+(LHI-LOW+1)*MMAX
         LOW=LHI+1
         IF (LOW > PP%LMAX) EXIT block
      ENDDO block


    END SUBROUTINE ADD_CDIJ

!*******************************************************************
!
! set the array CRHODE such that it corresponds
! to the atomic reference occupancies
! 
!*******************************************************************

    SUBROUTINE SET_CRHODE_ATOM(CRHODE, PP)
      USE prec
      USE pseudo
      IMPLICIT NONE
      OVERLAP CRHODE(:,:)

      TYPE (potcar),POINTER:: PP
      INTEGER LM, LL, LHI, LOW, MMAX, L, LP, M

      CRHODE=0

      LOW=1
      LM =1
      block: DO
         LL=PP%LPS(LOW)
         ! search block with same L
         DO LHI=LOW,PP%LMAX
            IF (LL/=PP%LPS(LHI)) EXIT
         ENDDO
         LHI=LHI-1
         MMAX=2*LL+1

         DO L =LOW,LHI
         DO LP=LOW,LHI
            DO M =0,MMAX-1
               CRHODE(LM+(L-LOW)*MMAX+M,LM+(LP-LOW)*MMAX+M)=PP%QATO(L,LP)
            ENDDO
         ENDDO
         ENDDO
      
         ! set new LOW value and LM value and go on
         LM=LM+(LHI-LOW+1)*MMAX
         LOW=LHI+1
         IF (LOW > PP%LMAX) EXIT block
      ENDDO block
    END SUBROUTINE SET_CRHODE_ATOM


!*******************************************************************
!
! calculate the spinor representation of a  PP strength matrix
! input supplied in the total and magnetisation representation
!
!*******************************************************************

    SUBROUTINE DIJ_FLIP(CTMP,LMDIM)
      USE prec
      IMPLICIT NONE
      INTEGER LMDIM
      OVERLAP CTMP(:,:,:)
      OVERLAP C00,CX,CY,CZ
      REAL(q) :: FAC
      INTEGER L,LP

      FAC=1.0
      DO L=1,LMDIM
         DO LP=1,LMDIM
            C00=CTMP(L,LP,1)
            CX =CTMP(L,LP,2)
            CY =CTMP(L,LP,3)
            CZ =CTMP(L,LP,4)
            
            CTMP(L,LP,1)= (C00+CZ)*FAC           
            CTMP(L,LP,2)= (CX-CY*(0._q,1._q))*FAC
            CTMP(L,LP,3)= (CX+CY*(0._q,1._q))*FAC
            CTMP(L,LP,4)= (C00-CZ)*FAC           
         ENDDO
      ENDDO
    END SUBROUTINE DIJ_FLIP

!*******************************************************************
!
! calculates the spinor representation of an OCCUPANCY matrix (e.g. CRHODE)
! occupancies are supplied as total and magnetisation representaiton
! unfortunately not identical to the previous cases, 
! since a factor 1/2 is required here
!
!*******************************************************************

    SUBROUTINE OCC_FLIP4(CTMP,LMDIM)
      USE prec
      IMPLICIT NONE
      INTEGER LMDIM
      OVERLAP CTMP(:,:,:)
      OVERLAP C00,CX,CY,CZ
      REAL(q) :: FAC
      INTEGER L,LP

      FAC=0.5
      DO L=1,LMDIM
         DO LP=1,LMDIM
            C00=CTMP(L,LP,1)
            CX =CTMP(L,LP,2)
            CY =CTMP(L,LP,3)
            CZ =CTMP(L,LP,4)
            
            CTMP(L,LP,1)= (C00+CZ)*FAC           
            CTMP(L,LP,2)= (CX-CY*(0._q,1._q))*FAC
            CTMP(L,LP,3)= (CX+CY*(0._q,1._q))*FAC
            CTMP(L,LP,4)= (C00-CZ)*FAC           
         ENDDO
      ENDDO
    END SUBROUTINE OCC_FLIP4
 
!*******************************************************************
!
! calculate the up and down occupancies from
! total and magnetisation (collinear version)
!
!*******************************************************************

    SUBROUTINE OCC_FLIP2(CTMP,LMDIM)
      USE prec
      IMPLICIT NONE
      INTEGER LMDIM
      OVERLAP CTMP(:,:,:)
      OVERLAP CQU,CQD
      REAL(q) :: FAC
      INTEGER L,LP

      FAC=0.5
      DO L=1,LMDIM
         DO LP=1,LMDIM
            CQU=CTMP(L,LP,1)
            CQD=CTMP(L,LP,2)
            CTMP(L,LP,1)=FAC*(CQU+CQD)
            CTMP(L,LP,2)=FAC*(CQU-CQD)
         ENDDO
      ENDDO
    END SUBROUTINE OCC_FLIP2
   
!*******************************************************************
!
!  calculate net moment of the soft
!  augmentation charges (compensation charges in PAW language)
!  RHO(LM) at one site (this routine is called from us.F)
!
!  RHO(lm,l'm') are the occupancies of each channel
!  RHO(LM) is given by
!  RHO(LM) =  sum C(LM,ll',mm')  RHO(lm,l'm') QPAW(llp)
!  also see TRANS_RHOLM
!
!*******************************************************************


    SUBROUTINE CALC_RHOLM( LYMAX, RHOLLMM, RHOLM, P)
      USE pseudo
      USE constant
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      REAL(q) RHOLM(:)       ! local charge for each L,M
      INTEGER LYMAX          ! maximum L
    ! local varible
      INTEGER LMYMAX,CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      REAL(q) FAKT

    ! initialize everything to 0

      LMYMAX=(LYMAX+1)**2
      RHOLM=0

   ! loop over all channels (l,epsilon)
      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

   ! transform coefficients and multiply with QPAW (which gives the moment
   ! of the corresponding charge)
      FAKT=1
      IF (CH1 /= CH2) THEN
         FAKT=2   ! CH2 is >= CH1
      ENDIF

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            RHOLM(JS(IC))= RHOLM(JS(IC))+ YLM3(IC)*FAKT*P%QPAW(CH1,CH2,JL(IC))* &
     &         RHOLLMM(LM+M-1,LMP+MP-1)
         ENDDO
      ENDDO
      ENDDO

      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

#ifdef debug
      WRITE(0,*) 'CALC_RHOLM: augmentation charges are'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.7)') LL,(RHOLM(LL**2+M)*2*SQRT(PI),M=1,LL*2+1)
      ENDDO
#endif

      END SUBROUTINE CALC_RHOLM


!*******************************************************************
!
!  calculate net moment of the soft
!  augmentation charges (compensation charges in PAW language)
!  RHO(LM) at one site (this routine is called from us.F)
!
!  RHO(lm,l'm') are the occupancies of each channel
!  RHO(LM) is given by
!  RHO(LM) =  sum C(LM,ll',mm')  RHO(lm,l'm') QPAW(llp)
!  also see TRANS_RHOLM
!
!*******************************************************************


    SUBROUTINE CALC_RHOLM_FOCK( LYMAX, RHOLLMM, RHOLM, P, NAE)
      USE pseudo
      USE constant
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      INTEGER NAE
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      REAL(q) RHOLM(:)       ! local charge for each L,M
      INTEGER LYMAX          ! maximum L
    ! local varible
      INTEGER LMYMAX,CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      REAL(q) FAKT

    ! initialize everything to 0

      LMYMAX=(LYMAX+1)**2
      RHOLM=0

   ! loop over all channels (l,epsilon)
      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

   ! transform coefficients and multiply with QPAW (which gives the moment
   ! of the corresponding charge)
      FAKT=1
      IF (CH1 /= CH2) THEN
         FAKT=2   ! CH2 is >= CH1
      ENDIF

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            IF (JL(IC)  <= SIZE(P%QPAW_FOCK,3)-1) THEN
               RHOLM(JS(IC))= RHOLM(JS(IC))+ YLM3(IC)*FAKT*P%QPAW_FOCK(CH1,CH2,JL(IC),NAE)* &
     &              RHOLLMM(LM+M-1,LMP+MP-1)
            ENDIF
         ENDDO
      ENDDO
      ENDDO

      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

#ifdef debug
      WRITE(0,*) 'CALC_RHOLM_FOCK: augmentation charges are'
      DO LL=0,LYMAX
         WRITE(0,'(I3,10F10.7)') LL,(RHOLM(LL**2+M)*2*SQRT(PI),M=1,LL*2+1)
      ENDDO
#endif

    END SUBROUTINE CALC_RHOLM_FOCK


!*******************************************************************
!
!  transform the real part of the occupancies RHO(lm,l'm') 
!  to RHO(ll',L,M) using Clebsch Gordan coefficients'
!
!' RHO(ll',LM) =  sum C(LM,ll',mm')  RHO(lm,l'm')
!  where C(LM,ll',mm') = \int Y_LM Y_lm Y_l'm' d Omega
!
!  the storage layout of RHO(llp,LM) is somewhat akward
!  for each l lp pair, (2l+1) (2lp+1) elements must be stored
!  they are stored in the order
!    Lmin,M=0 ... Lmin,M=2*Lmin+1
!     ...
!    Lmax,M=0 ... Lmax,M=2*Lmax+1
!  where Lmin and Lmax are given by the triangular rule
!  Lmin = | l-l' |  and Lmax = | l+l'|
!  certain elements in this array will be always zero, because
!  the sum rule allows only L=Lmin,Lmin+2,...,Lmax
!
!*******************************************************************


    SUBROUTINE TRANS_RHOLM( RHOLLMM, RHOLM, P)
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      REAL(q) RHOLM(:)       ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX
      REAL(q) FAKT
   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

         ! quantum numbers l and lp of these two channels
         LL =P%LPS(CH1)
         LLP=P%LPS(CH2)

         CALL YLM3LOOKUP(LL,LLP,LMINDX)
         ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)
         
         ! transform coefficients
         FAKT=1
         IF (CH1 /= CH2) THEN
            FAKT=FAKT*2
         ENDIF
         ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
         JBASE=IBASE-LMIN*LMIN
         
         RHOLM(IBASE+1:IBASE+(2*LL+1)*(2*LLP+1))=0
#ifdef debug
         WRITE(0,*) 'CALC_RHOLM: RHOLLMM is',CH1,CH2
         DO MP=1,2*LLP+1
            WRITE(0,'(10F10.7)') (RHOLLMM(LM+M-1,LMP+MP-1),M=1,2*LL+1)
         ENDDO
#endif
         DO M =1,2*LL+1
            DO MP=1,2*LLP+1
               LMINDX=LMINDX+1
               
               ISTART=INDCG(LMINDX)
               IEND  =INDCG(LMINDX+1)
               DO  IC=ISTART,IEND-1
                  RHOLM(JS(IC)+JBASE)= RHOLM(JS(IC)+JBASE)+ YLM3(IC)*FAKT* &
                       &         RHOLLMM(LM+M-1,LMP+MP-1)
               ENDDO
            ENDDO
         ENDDO

#ifdef debug
      WRITE(0,*) 'CALC_RHOLM: augmentation charges are',CH1,CH2
      DO MP=LMIN,LMAX
         WRITE(0,'(I3,10F10.7)') MP,(RHOLM(JBASE+MP**2+M),M=1,MP*2+1)
      ENDDO
#endif

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO


    END SUBROUTINE TRANS_RHOLM


!*******************************************************************
!
!  transform the imaginary part of the occupancies RHO(lm,l'm') 
!  to RHO(llp,L,M) using Clebsch Gordan coefficients
!
!' RHO(ll',LM) =  sum C(LM,ll',mm')  RHO(lm,l'm')
!  where C(LM,ll',mm') = \int Y_LM Y_lm Y_l'm' d Omega
!
!
!*******************************************************************

#ifndef realmode
    SUBROUTINE TRANS_RHOLM_IM( RHOLLMM, RHOLM, P)
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      REAL(q) RHOLM(:)       ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX
      REAL(q) FAKT,FAKT2
   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)
      
      CALL YLM3LOOKUP(LL,LLP,LMINDX)
   ! Lmin and Lmax
      LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)

   ! transform coefficients
      FAKT=1
      IF (CH1 /= CH2) THEN
         FAKT=FAKT*2
      ENDIF
   ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
      JBASE=IBASE-LMIN*LMIN

      RHOLM(IBASE+1:IBASE+(2*LL+1)*(2*LLP+1))=0

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1
         IF (M<MP) THEN
           FAKT2=1
         ELSE
           FAKT2=-1
         ENDIF
         
         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)
         DO  IC=ISTART,IEND-1
            RHOLM(JS(IC)+JBASE)= RHOLM(JS(IC)+JBASE)+ YLM3(IC)*FAKT*FAKT2* &
     &         AIMAG(RHOLLMM(LM+M-1,LMP+MP-1))
         ENDDO
      ENDDO
      ENDDO

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
    END SUBROUTINE TRANS_RHOLM_IM
#endif
    
    SUBROUTINE TRANS_RHOLMI( DLLMM, DLM, P)
      USE pseudo
      USE asa
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
      REAL(q) DLM(:)       ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX,INMIN,INMAX

   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)
   ! Lmin and Lmax
      LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)

   ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
      JBASE=IBASE-LMIN*LMIN

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)
         DO  IC=ISTART,IEND-1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                            DLM(JS(IC)+JBASE)*YLM3I(IC)
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)/2
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

    END SUBROUTINE TRANS_RHOLMI
    
#ifndef realmode
    SUBROUTINE TRANS_RHOLMI_IM( DLLMM, DLM, P)
      USE pseudo
      USE asa
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
      REAL(q) DLM(:)       ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX,INMIN,INMAX
      REAL(q) :: FAKT2
   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)
   ! Lmin and Lmax
      LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)

   ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
      JBASE=IBASE-LMIN*LMIN

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1
         IF (M<MP) THEN
           FAKT2= 1
         ELSE
           FAKT2=-1
         ENDIF

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)
         DO  IC=ISTART,IEND-1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                            DLM(JS(IC)+JBASE)*YLM3I(IC)*FAKT2
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)/2
            DLLMM(LMP+MP-1,LM+M-1)=-DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

    END SUBROUTINE TRANS_RHOLMI_IM
#endif    
    
!*******************************************************************
!
!  transform D(llp,L,M) to the representation D(lm,l'm')
!  using Clebsch Gordan coefficients and add to another array
!
!  D(lm,l'm') =  sum C(LM,ll',mm') D(llp,LM)
!  where C(LM,ll',mm') = \int Y_LM Y_lm Y_l'm' d Omega
!
!  the storage layout of D(llp,LM) is somewhat akward see above
!
!*******************************************************************


    SUBROUTINE TRANS_DLM( DLLMM, DLM, P)
      USE pseudo
      USE asa
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
      REAL(q) DLM(:)       ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP
      INTEGER IBASE,JBASE,LMIN,LMAX,INMIN,INMAX

   ! loop over all channels (l,epsilon)
      IBASE=0

      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)
   ! Lmin and Lmax
      LMIN=ABS(LL-LLP) ; LMAX=ABS(LL+LLP)

   ! JS(IC) is pointing to L*L+M+1, we must subtract LMIN*LMIN
      JBASE=IBASE-LMIN*LMIN
#ifdef debug
      DO M=LMIN,LMAX
         WRITE(0,'(3I3,10F14.7)') CH1,CH2,M, &
               (DLM(JBASE+M*M+MP),MP=1,2*M+1)
      ENDDO
#endif

      INMIN=1000
      INMAX =-1000

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)
         DO  IC=ISTART,IEND-1
            INMIN=MIN(JS(IC)+JBASE,INMIN)
            INMAX=MAX(JS(IC)+JBASE,INMAX)
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                            DLM(JS(IC)+JBASE)*YLM3(IC)
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      IBASE=IBASE+(2*LL+1)*(2*LLP+1)
      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO

    END SUBROUTINE TRANS_DLM

!*******************************************************************
!
!  transform D(LM) to D(lm,l'm')
!  where D(LM) is defined as
!  D(LM) = \int V Y(L,M) Q(L)(r) d^3 r
!  and \int Q(L)(r) r^(2+L) dr is integrating to 1
!  (this routine is called from us.F)
!
!  the "pseudopotential strength" is then given by
!  D(lm,l'm') =  sum C(LM,ll',mm')  D(LM) QPAW(llp)
!  also see TRANS_DLM
!*******************************************************************


    SUBROUTINE CALC_DLLMM( DLLMM, DLM, P)
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)  ! net augmentation charge
      REAL(q) DLM(:)      ! local charge for each L,M
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP

    ! initialize everything to 0
      DLLMM=0

   ! loop over all channels (l,epsilon)
      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

   ! transform coefficients and multiply with QPAW (which gives the moment
   ! of the corresponding charge)

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                      DLM(JS(IC))*YLM3(IC)*P%QPAW(CH1,CH2,JL(IC))
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
    END SUBROUTINE CALC_DLLMM


!************************* SUBROUTINE CALC_DLLMM_FOCK  ****************
!
! this routine does the same thing as CALC_DLLMM
! but uses QPAW_FOCK instead of QPAW
!
!**********************************************************************

    SUBROUTINE CALC_DLLMM_FOCK( DLLMM, DLM, P, NAE )
      USE pseudo
      USE asa
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)  ! net augmentation charge
      REAL(q) DLM(:)      ! local charge for each L,M
      INTEGER NAE
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LM,LMP,LMINDX,ISTART,IEND,IC,M,MP

   ! loop over all channels (l,epsilon)
      LM=1
      DO CH1=1,P%LMAX
      LMP=LM
      DO CH2=CH1,P%LMAX

   ! quantum numbers l and lp of these two channels
      LL =P%LPS(CH1)
      LLP=P%LPS(CH2)

      CALL YLM3LOOKUP(LL,LLP,LMINDX)

   ! transform coefficients and multiply with QPAW (which gives the moment
   ! of the corresponding charge)

      DO M =1,2*LL+1
      DO MP=1,2*LLP+1
         LMINDX=LMINDX+1

         ISTART=INDCG(LMINDX)
         IEND  =INDCG(LMINDX+1)

         DO  IC=ISTART,IEND-1
            IF (JL(IC)  <= SIZE(P%QPAW_FOCK,3)-1) THEN
               DLLMM(LM+M-1,LMP+MP-1)=DLLMM(LM+M-1,LMP+MP-1)+ &
                      DLM(JS(IC))*YLM3(IC)*P%QPAW_FOCK(CH1,CH2,JL(IC),NAE)
            ENDIF
         ENDDO
      ENDDO
      ENDDO
  ! fill symmetric components (CH2 is >= CH1)
      IF  (CH1 /= CH2) THEN
         DO M =1,2*LL+1
         DO MP=1,2*LLP+1
            DLLMM(LMP+MP-1,LM+M-1)=DLLMM(LM+M-1,LMP+MP-1)
         ENDDO
         ENDDO
      ENDIF

      LMP=LMP+2*LLP+1
      ENDDO
      LM =LM +2*LL +1
      ENDDO
    END SUBROUTINE CALC_DLLMM_FOCK
!


!*******************************************************************
!
!  calculate net moment of the current augmentation charges 
!  RHO(LM) at one site (this routine is called from us.F) for
!  direction icart
!
!  RHO(lm,l'm') are the one center occupancies for each channel
!
!  RHO(LM) =  sum C(LM,ll',mm')  RHO(lm,l'm') JPAW(lm,l'm',LM, icart)
!
!*******************************************************************


    SUBROUTINE CALC_RHOLM_JVEC(  RHOLLMM, RHOLM, P, ICART)
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      OVERLAP RHOLM(:)       ! local current for each L,M channel
      INTEGER ICART          ! cartesian index for current
    ! local varible
      INTEGER CH1,CH2,LL,LM
      REAL(q) FAKT

    ! initialize everything to 0
      RHOLM=0

      DO LM =1,SIZE(P%JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,P%LMMAX
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,P%LMMAX
               RHOLM(LM)=RHOLM(LM)+ P%JPAW(CH2, CH1, LM, ICART)*CHGCONJG(RHOLLMM( CH2, CH1 ))
            ENDDO
         ENDDO
      ENDDO

    END SUBROUTINE CALC_RHOLM_JVEC


!*******************************************************************
! similar to above, only RHOLLMM and RHOLM are of type COMPLEX
! instead of OVERLAP
!*******************************************************************

    SUBROUTINE CALC_RHOLM_JVEC_CMPLX(  RHOLLMM, RHOLM, P, ICART)
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      COMPLEX(q) RHOLLMM(:,:)   ! net augmentation charge
      COMPLEX(q) RHOLM(:)       ! local current for each L,M channel
      INTEGER ICART          ! cartesian index for current
    ! local varible
      INTEGER CH1,CH2,LL,LM
      REAL(q) FAKT

    ! initialize everything to 0
      RHOLM=0

      DO LM =1,SIZE(P%JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,P%LMMAX
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,P%LMMAX
!              RHOLM(LM)=RHOLM(LM)+ P%JPAW(CH2, CH1, LM, ICART)*CHGCONJG(RHOLLMM( CH2, CH1 ))
               RHOLM(LM)=RHOLM(LM)+ P%JPAW(CH2, CH1, LM, ICART)*CONJG(RHOLLMM( CH2, CH1 ))
            ENDDO
         ENDDO
      ENDDO

    END SUBROUTINE CALC_RHOLM_JVEC_CMPLX
!
! similar to the above routine but without monopole terms
! these are usually treated using the gauge twisted projectors 
!
    SUBROUTINE CALC_RHOLM_JVEC_NOMONO(  RHOLLMM, RHOLM, P, ICART)
! we skip the monopole term: JPAW(:,:,1,:) contributions do not enter into the sum
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      OVERLAP RHOLM(:)       ! local current for each L,M channel
      INTEGER ICART          ! cartesian index for current
    ! local varible
      INTEGER CH1,CH2,LL,LM
      REAL(q) FAKT

    ! initialize everything to 0
      RHOLM=0

      DO LM =2,SIZE(P%JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,P%LMMAX
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,P%LMMAX
               RHOLM(LM)=RHOLM(LM)+ P%JPAW(CH2, CH1, LM, ICART)*CHGCONJG(RHOLLMM( CH2, CH1 ))
            ENDDO
         ENDDO
      ENDDO

    END SUBROUTINE CALC_RHOLM_JVEC_NOMONO

!
! similar to the first routine, but this time JPAW instead of the 
! pseudopotential pointer PP is passed down
!
    SUBROUTINE CALC_RHOLM_JVEC_(  RHOLLMM, RHOLM, JPAW, ICART)
      USE pseudo
      IMPLICIT NONE
      OVERLAP RHOLLMM(:,:)   ! net augmentation charge
      OVERLAP RHOLM(:)       ! local current for each L,M channel
      REAL(q) JPAW(:,:,:,:)  
      INTEGER ICART          ! cartesian index for current
    ! local varible
      INTEGER CH1,CH2,LL,LM
      REAL(q) FAKT

    ! initialize everything to 0
      RHOLM=0

      DO LM =1,SIZE(JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,SIZE(JPAW,1)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,SIZE(JPAW,1)
               RHOLM(LM)=RHOLM(LM)+ JPAW(CH2, CH1, LM, ICART)*CHGCONJG(RHOLLMM( CH2, CH1 ))
            ENDDO
         ENDDO
      ENDDO

    END SUBROUTINE CALC_RHOLM_JVEC_

!*******************************************************************
!
!  related routine for vector potentials:
!  transform D(LM) to D(lm,l'm')
!  where D(LM) is defined as
!  D(LM) = \int AVEC(icart) Y(L,M) Q(L)(r) d^3 r
!  and \int Q(L)(r) r^(2+l) dr is integrating to 1
!
!  the "pseudopotential strength" is then given by
!  D(lm,l'm') =  sum  D(LM) P%JPAW(lm,l'm',LM, icart)
!
!*******************************************************************


    SUBROUTINE CALC_DLLMM_AVEC( DLLMM, DLM, P, ICART)
      USE pseudo
      IMPLICIT NONE
      OVERLAP DLLMM(:,:)  ! contribution to non-local strength parameter
      OVERLAP DLM(:)      ! local moment of vector field for each LM
      TYPE (potcar) P
      INTEGER ICART       ! cartesian index of magentic vector field
    ! local varible
      INTEGER CH1,CH2,LM

      DO LM =1,SIZE(P%JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,P%LMMAX
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,P%LMMAX
               DLLMM(CH2, CH1 )=DLLMM(CH2, CH1 )+ P%JPAW(CH2, CH1, LM, ICART) *DLM(LM)
            ENDDO
         ENDDO
      ENDDO
    END SUBROUTINE CALC_DLLMM_AVEC
!
! similar to the above routine but without monopole terms
! these are usually treated using the gauge twisted projectors 
!
    SUBROUTINE CALC_DLLMM_AVEC_NOMONO( DLLMM, DLM, P, ICART)
      USE pseudo
      IMPLICIT NONE
      OVERLAP DLLMM(:,:)  ! contribution to non-local strength parameter
      OVERLAP DLM(:)      ! local moment of vector field for each LM
      TYPE (potcar) P
      INTEGER ICART       ! cartesian index of magentic vector field
    ! local varible
      INTEGER CH1,CH2,LM

      DO LM =2,SIZE(P%JPAW,3)
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels (lmepsilon)
         DO CH1 =1,P%LMMAX
!DIR$ IVDEP
!OCL NOVREC
   ! loop over all channels' (lmepsilon)
            DO CH2=1,P%LMMAX
               DLLMM(CH2, CH1 )=DLLMM(CH2, CH1 )+ P%JPAW(CH2, CH1, LM, ICART) *DLM(LM)
            ENDDO
         ENDDO
      ENDDO
    END SUBROUTINE CALC_DLLMM_AVEC_NOMONO


!*******************************************************************
!
! write out DIJ
!
!*******************************************************************


    SUBROUTINE DUMP_DLLMM( C,DLLMM, P)
      USE pseudo
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
      CHARACTER (LEN=*) C
    ! local varible
      INTEGER LM,LMP

      WRITE(*,*) C
      DO LM=1,MIN(10,P%LMMAX)
         WRITE(*,'(20(F14.7,1X))') (REAL(DLLMM(LMP,LM),q),LMP=1,MIN(10,P%LMMAX))
!#ifdef RHOLM_complex
!         WRITE(*,'(20(F8.3,1X))') (AIMAG(DLLMM(LMP,LM)),LMP=1,MIN(5,P%LMMAX))
!#endif
      ENDDO

    END SUBROUTINE DUMP_DLLMM

#ifdef RHOLM_complex

    SUBROUTINE DUMP_DLLMM_IM( DLLMM, P)
      USE pseudo
      USE constant
      IMPLICIT NONE
      TYPE (potcar) P
      OVERLAP DLLMM(:,:)   ! net augmentation charge
    ! local varible
      INTEGER LM,LMP
      WRITE(*,*) 'pseudopotential strength DIJ is'
      DO LM=1,P%LMMAX
         WRITE(*,'(18(F9.5,1X))') (AIMAG(DLLMM(LMP,LM)),LMP=1,MIN(10,P%LMMAX))
      ENDDO

    END SUBROUTINE DUMP_DLLMM_IM
#endif

!*******************************************************************
!
! retrieve the elements of the occupancy channels 
! that are mixed from RHOLM (derived from CRHODE) and store
! them in continous order in an array RHOLM_STORE
! the number of transfered elements in returned in IADD
!
! LCOMPACT = T RHOLM in a compact storage (i.e. no zero elements
!              inbetween)
! LCOMPACT = F RHOLM in conventional mode (storage layout used in
!              radial.F and paw.F)
!
!*******************************************************************

    SUBROUTINE STORE_RHOLM( RHOLM, RHOLM_STORE, METRIC, IADD, P, &
              LCOMPACT, IBASE)
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      REAL(q) RHOLM(:)       ! full RHO(llp,L,M) matrix
      REAL(q) RHOLM_STORE(:) ! storage point for elements with L=0
      REAL(q) METRIC(:)
      INTEGER IADD           ! number of elements stored in RHOLM_STORE
      INTEGER IBASE          ! maximum number of elements in RHOLM
      LOGICAL LCOMPACT       ! compact mode or not
    ! local varible
      INTEGER CH1,CH2,LL,LLP,LMIN,LMAX,JBASE,LMAIN,MMAIN,LMMAIN

      IADD=0
      IBASE=0

      DO CH1=1,P%LMAX
      DO CH2=CH1,P%LMAX

       ! quantum numbers l and lp of these two channels
         LL =P%LPS(CH1)
         LLP=P%LPS(CH2)
       ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=MIN(LMAX_MIX,P%LMAX_CALC,ABS(LL+LLP))
         JBASE=IBASE-LMIN*LMIN

         LMMAIN=LMIN*LMIN
       ! due to sum rules only LMIN to LMAX in steps of 2 are allowed
         DO LMAIN=LMIN,LMAX,2
         DO MMAIN=1,LMAIN*2+1
            IF (LCOMPACT) THEN
               ! for LCOMPACT elements with L quantum numbers that
               ! are not allowed according to sum rule are not stored IN RHOLM
               LMMAIN=LMMAIN+1
            ELSE
               LMMAIN=LMAIN*LMAIN+MMAIN
            ENDIF

            IADD=IADD+1
            RHOLM_STORE(IADD)=RHOLM(JBASE+LMMAIN)*METRIC(IADD)
         ENDDO
         ENDDO

         IF (LCOMPACT) THEN
            DO LMAIN=LMIN,ABS(LL+LLP),2
               IBASE=IBASE+LMAIN*2+1
            ENDDO
         ELSE
            IBASE=IBASE+(2*LL+1)*(2*LLP+1)
         ENDIF
      ENDDO
      ENDDO

    END SUBROUTINE STORE_RHOLM

!*******************************************************************
!
! retrieve the elements of the occupancy channels that are mixed
! from RHOLM_STORE and store them in RHOLM
! the number of transfered elements is returned in IADD
! the routine supports two mode
! LCOMPACT = T RHOLM in a compact storage (i.e. no zero elements
!              inbetween)
! LCOMPACT = F RHOLM in conventional mode (storage layout used in
!              radial.F and paw.F)
!
!*******************************************************************


    SUBROUTINE RETRIEVE_RHOLM( RHOLM, RHOLM_STORE, METRIC, IADD, P, &
              LCOMPACT, IBASE)
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      REAL(q) RHOLM(:)       ! full RHO(llp,L,M) matrix
      REAL(q) RHOLM_STORE(:) ! storage point for elements with L=0
      REAL(q) METRIC(:)      ! metric
      INTEGER IADD           ! number of elements transfered from RHOLM_STORE
      INTEGER IBASE          ! maximum number of elements in RHOLM
      LOGICAL LCOMPACT       ! compact mode or not
    ! local variable
      INTEGER CH1,CH2,LL,LLP,LMIN,LMAX,JBASE,LMAIN,MMAIN,LMMAIN
   ! loop over all channels (l,epsilon)
      IADD=0
      IBASE=0

      DO CH1=1,P%LMAX
      DO CH2=CH1,P%LMAX

       ! quantum numbers l and lp of these two channels
         LL =P%LPS(CH1)
         LLP=P%LPS(CH2)
       ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=MIN(LMAX_MIX,P%LMAX_CALC,ABS(LL+LLP))
         JBASE=IBASE-LMIN*LMIN

         LMMAIN=LMIN*LMIN

         DO LMAIN=LMIN,LMAX,2
         DO MMAIN=1,LMAIN*2+1
            IF (LCOMPACT) THEN
               ! for LCOMPACT elements with L quantum numbers that
               ! are not allowed according to sum rule are not stored IN RHOLM
               LMMAIN=LMMAIN+1
            ELSE
               LMMAIN=LMAIN*LMAIN+MMAIN
            ENDIF
               
            IADD=IADD+1
            RHOLM(JBASE+LMMAIN)=RHOLM_STORE(IADD)/METRIC(IADD)
         ENDDO
         ENDDO

         IF (LCOMPACT) THEN
            DO LMAIN=LMIN,ABS(LL+LLP),2
               IBASE=IBASE+2*LMAIN+1
            ENDDO
         ELSE
            IBASE=IBASE+(2*LL+1)*(2*LLP+1)
         ENDIF
      ENDDO
      ENDDO

    END SUBROUTINE RETRIEVE_RHOLM


    FUNCTION COUNT_RHO_PAW_ELEMENTS(P)
      USE pseudo
      IMPLICIT NONE
      TYPE (potcar) P
      INTEGER COUNT_RHO_PAW_ELEMENTS
    ! local variable
      INTEGER CH1,CH2,LL,LLP,LMIN,LMAX,JBASE,LMAIN,MMAIN,LMMAIN
    ! loop over all channels (l,epsilon)
      COUNT_RHO_PAW_ELEMENTS=0

      DO CH1=1,P%LMAX
      DO CH2=CH1,P%LMAX
       ! quantum numbers l and lp of these two channels
         LL =P%LPS(CH1)
         LLP=P%LPS(CH2)
       ! Lmin and Lmax
         LMIN=ABS(LL-LLP) ; LMAX=MIN(LMAX_MIX,P%LMAX_CALC,ABS(LL+LLP))
         DO LMAIN=LMIN,LMAX,2
            DO MMAIN=1,LMAIN*2+1
               COUNT_RHO_PAW_ELEMENTS=COUNT_RHO_PAW_ELEMENTS+1
            ENDDO
         ENDDO
      ENDDO
      ENDDO

    END FUNCTION COUNT_RHO_PAW_ELEMENTS

!*******************************************************************
!
! SET_RHO_PAW_ELEMENTS
! calculates the number of elements of the augmentation
! occupancies which must be mixed on the local node
! also allocates and sets the mask array DO_LOCAL which determines
! which ions are treated localy
!
!*******************************************************************

    SUBROUTINE SET_RHO_PAW_ELEMENTS(WDES, P , T_INFO, LOVERL, ELEMENTS )
      USE pseudo
      USE poscar
      USE wave
      USE constant
      USE asa
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::  P(T_INFO%NTYP)
      TYPE (wavedes)   WDES
      LOGICAL  LOVERL

    ! local variables
      TYPE (potcar),POINTER::  PP
      INTEGER NT,NI,NIP,LMAX_TABLE,NODE_TARGET,LM,LMP,LL,LLP,LMIN,LMAX
      INTEGER LMAIN,ELEMENTS,CH1,CH2,NI_PAW_LOCAL
      REAL(q) :: AMETRIC
!=======================================================================
! quick return and allocation of work space
!=======================================================================
      ELEMENTS=0

      IF (.NOT.LOVERL) RETURN

      ! allocate the DO_LOCAL array
      ALLOCATE (DO_LOCAL(T_INFO%NIONS))

      DO_LOCAL=.FALSE.
      IONS_LOCAL=0
!=======================================================================
! cycle all ions and calculate number of L=0 channels
!=======================================================================
      ELEMENTS=0

      ion: DO NI=1,T_INFO%NIONS
         NIP=NI_LOCAL(NI, WDES%COMM_INB)
         NT =T_INFO%ITYP(NI)
    ! not on local node, cycle
    ! no PAW for this ion, cycle
         IF ( NIP/=0 .AND. ASSOCIATED(P(NT)%QPAW) ) THEN
            IONS_LOCAL=IONS_LOCAL+1

            ! distribute ions in a round robin fashion
            ! between nodes that share one DIJ
            DO_LOCAL(NI)= .TRUE.
#ifdef MPI
            NODE_TARGET=MOD(IONS_LOCAL, WDES%COMM_INTER%NCPU)+1
            IF (NODE_TARGET /=  WDES%COMM_INTER%NODE_ME ) THEN
               DO_LOCAL(NI)= .FALSE.
            ENDIF
#endif

            IF (DO_LOCAL(NI) ) THEN
            ! how many elements are there
               DO CH1=1,P(NT)%LMAX
                  DO CH2=CH1,P(NT)%LMAX
                     ! quantum numbers l and lp of these two channels
                     LL =P(NT)%LPS(CH1)
                     LLP=P(NT)%LPS(CH2)
                     ! Lmin and Lmax
                     LMIN=ABS(LL-LLP) ; LMAX=MIN(LMAX_MIX,P(NT)%LMAX_CALC,ABS(LL+LLP))
                     DO LMAIN=LMIN,LMAX,2
                        ELEMENTS=ELEMENTS+LMAIN*2+1
#ifdef RHOLM_complex
                        ELEMENTS=ELEMENTS+LMAIN*2+1
#endif   
                     ENDDO
                  ENDDO
               ENDDO
            ENDIF
         ENDIF
      ENDDO ion
!=======================================================================
! finally calculate the metric tensor of each component that is mixed
! locally
!=======================================================================
      IF (.NOT. ALLOCATED( METRIC) ) THEN
         ALLOCATE( METRIC(ELEMENTS))
      ENDIF

      NI_PAW_LOCAL=0
      DO NI=1,T_INFO%NIONS
         IF (.NOT. DO_LOCAL(NI)) CYCLE

         NT=T_INFO%ITYP(NI)
         PP=> P(NT)
         DO CH1=1,P(NT)%LMAX
            DO CH2=CH1,P(NT)%LMAX
               ! quantum numbers l and lp of these two channels
               LL =P(NT)%LPS(CH1)
               LLP=P(NT)%LPS(CH2)
               ! Lmin and Lmax
               LMIN=ABS(LL-LLP) ; LMAX=MIN(LMAX_MIX,P(NT)%LMAX_CALC,ABS(LL+LLP))
               DO LMAIN=LMIN,LMAX,2
                 ! metric is defined as
                 ! M = ABS( rho(ae)^2 - (rho(ps)+rho(comp))^2)
                 CALL RAD_METRIC( PP%R, PP%AUG(:,LMAIN), PP%QPAW(CH1,CH2,LMAIN), &
                  PP%WAE(:,CH1), PP%WAE(:,CH2), &
                  PP%WPS(:,CH1), PP%WPS(:,CH2), AMETRIC)
                 ! generally larger L qauntum numbers are less important
                 AMETRIC=SQRT(MIN(MAX(ABS(AMETRIC),1E-6_q),1E-2_q))
                 ! if you want no metric, it will be slower ...
                 METRIC( NI_PAW_LOCAL+1:NI_PAW_LOCAL+LMAIN*2+1 ) = AMETRIC
#ifdef RHOLM_complex
                 METRIC( NI_PAW_LOCAL+1+ELEMENTS/2:NI_PAW_LOCAL+LMAIN*2+1+ELEMENTS/2 ) = AMETRIC
#endif   
                 NI_PAW_LOCAL=NI_PAW_LOCAL+LMAIN*2+1
               ENDDO
            ENDDO
         ENDDO
      ENDDO

    END SUBROUTINE SET_RHO_PAW_ELEMENTS

!*******************************************************************
!
! SET_RHO_PAW store the elements of the augmentation
! occupancies CRHODE mixed on the local node in RHOLM_STORE
!
!  calling convention for this must be changed to
!  (charge, m) instead of (up, down)
!
!*******************************************************************

    SUBROUTINE SET_RHO_PAW(WDES, P , T_INFO, LOVERL, &
         ISPIN, LMDIM, CRHODE , RHOLM_STORE )
      USE pseudo
      USE poscar
      USE wave
      USE constant
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::      P(T_INFO%NTYP)
      TYPE (wavedes)     WDES
      INTEGER LMDIM,ISPIN
      OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,ISPIN)
      LOGICAL  LOVERL
      REAL(q) RHOLM_STORE(:,:)

    ! local variables
      TYPE (potcar),POINTER:: PP
      INTEGER NT,NI,NIP,ISP,ITMP
      INTEGER ISIZE,IBASE,IADD
      INTEGER, EXTERNAL :: MAXL1
      REAL(q) RHOLM(LMDIM*LMDIM)

!=======================================================================
! quick return if possible
!=======================================================================
      IF (.NOT.LOVERL .OR. IONS_LOCAL == 0 ) RETURN

      IF (MIMIC_US) RETURN
!=======================================================================
! cycle all ions and set required elements
!=======================================================================
      IBASE=1
      ISIZE=UBOUND(RHOLM_STORE,1) ! runtimecheck for insufficient allocation

      ion: DO NI=1,T_INFO%NIONS
         IF (.NOT. DO_LOCAL(NI)) CYCLE ion

         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         IF (NIP==0) THEN
            WRITE(0,*) 'SET_RHO_PAW: internal error: ion not local'
            STOP
         ENDIF

         NT=T_INFO%ITYP(NI)

         PP=> P(NT)
         DO ISP=1,ISPIN
    ! transform CRHODE (lm,lpmp) to llp,LM
            CALL TRANS_RHOLM( CRHODE(:,:,NIP,ISP), RHOLM, PP )
    ! retrieve elements which are mixed
            CALL STORE_RHOLM( RHOLM, RHOLM_STORE(IBASE:,ISP),  &
                    METRIC(IBASE:), IADD, PP, .FALSE., ITMP )
#ifdef RHOLM_complex
            CALL TRANS_RHOLM_IM( CRHODE(:,:,NIP,ISP), RHOLM, PP )
            CALL STORE_RHOLM( RHOLM, RHOLM_STORE(IBASE+ISIZE/2:,ISP),  &
                    METRIC(IBASE+ISIZE/2:), IADD, PP, .FALSE., ITMP )
#endif           
         ENDDO
         IBASE=IBASE+IADD
         IF (IBASE > ISIZE+1) THEN
            WRITE(0,*) 'internal error SET_RHO_PAW: insufficient space'
            STOP
         ENDIF
      ENDDO ion
#ifndef RHOLM_complex
      IF (IBASE /= ISIZE+1) THEN
         WRITE(0,*) 'internal error SET_RHO_PAW: RHOLM_STORE wrong allocation',IBASE, ISIZE+1
         STOP
      ENDIF
#endif
      ! WRITE(*,'("RHOOUT",6F10.6)') RHOLM_STORE

    END SUBROUTINE SET_RHO_PAW


!*******************************************************************
!
! RETRIVE_RHO_PAW retrives the elements of the augmentation
! occupancies CRHODE from RHOLM 
! this is essentially the inverse of the previos subroutine
! 
! this is not totally trivial since simply overwriting the
! required elements in CRHODE is not possible
! instead CRHODE must be transformed using Clebsh Gordon coefficients
! and next RHOLM overwrites the required elements and finally
! the result is back transformed and stored in CRHODE
!
!  calling convention for this must be changed to
!  (charge, m) instead of (up, down)
!
!*******************************************************************

    SUBROUTINE RETRIVE_RHO_PAW(WDES, P , T_INFO, LOVERL, LMDIM, CRHODE , RHOLM_STORE )
      USE pseudo
      USE poscar
      USE wave
      USE constant
      IMPLICIT NONE

      TYPE (type_info)   T_INFO
      TYPE (potcar),TARGET::      P(T_INFO%NTYP)
      TYPE (wavedes)     WDES
      INTEGER LMDIM
      OVERLAP  CRHODE(LMDIM,LMDIM,WDES%NIONS,WDES%NCDIJ)
      LOGICAL  LOVERL
      REAL(q) RHOLM_STORE(:,:)

    ! local variables
      REAL(q) RHOLM_(LMDIM*LMDIM,WDES%NCDIJ),RHOLM(LMDIM*LMDIM)
      OVERLAP COCC(LMDIM,LMDIM,MAX(2,WDES%NCDIJ)),COCC_IM(LMDIM,LMDIM)
      TYPE (potcar),POINTER:: PP
      INTEGER NT,NI,NIP,ISP,ITMP
      INTEGER ISIZE,IBASE,IADD
      INTEGER, EXTERNAL :: MAXL1

!=======================================================================
! quick return if possible
!=======================================================================
      IF (.NOT.LOVERL .OR. IONS_LOCAL == 0 ) RETURN

      IF (MIMIC_US) RETURN
!=======================================================================
! cycle all ions and set required elements
!=======================================================================
      IBASE=1
      ISIZE=UBOUND(RHOLM_STORE,1) ! runtimecheck for insufficient allocation

      ion: DO NI=1,T_INFO%NIONS
         IF (.NOT. DO_LOCAL(NI)) THEN
            NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
            CRHODE(:,:,NIP,:)=0
            CYCLE ion
         ENDIF

         NIP=NI_LOCAL(NI, WDES%COMM_INB) ! local storage index
         IF (NIP==0) THEN
            WRITE(0,*) 'SET_RHO_PAW: internal error: ion not local'
            STOP
         ENDIF

         NT=T_INFO%ITYP(NI)

         PP=> P(NT)
         COCC=0

         DO ISP=1,WDES%NCDIJ
            ! retrieve the one center on site charge densities to RHOLM_
            IF ( LMAX_MIX < PP%LMAX_CALC) &
                 CALL TRANS_RHOLM( CRHODE(:,:,NIP,ISP), RHOLM_(:,ISP), PP )
            ! retrieve mixed elements from RHOLM_STORE and overwrite them in RHOLM_
            CALL RETRIEVE_RHOLM( RHOLM_(:,ISP), RHOLM_STORE(IBASE:,ISP), &
                           METRIC(IBASE:), IADD, PP, .FALSE.,  ITMP)
            ! calculate the occupancy matrix COCC from RHOLM_(:,ISP)
            CALL TRANS_RHOLMI( COCC(:,:,ISP), RHOLM_(:,ISP), PP )
#ifndef realmode
            ! retrieve imaginary part from CRHODE and store in RHOLM
            CALL TRANS_RHOLM_IM( CRHODE(:,:,NIP,ISP), RHOLM, PP )
               ! orverwrite by elements passed through mixer
#ifdef RHOLM_complex
            CALL RETRIEVE_RHOLM( RHOLM, RHOLM_STORE(IBASE+ISIZE/2:,ISP), &
                 METRIC(IBASE+ISIZE/2:), IADD, PP, .FALSE.,  ITMP)
#endif
            COCC_IM=0
            CALL TRANS_RHOLMI_IM( COCC_IM, RHOLM, PP )
            ! join imaginary and real parts
            COCC(:,:,ISP)=COCC(:,:,ISP)+(0._q,1._q)*COCC_IM(:,:)

!            WRITE(*,*) 'spin',ISP
!            CALL DUMP_DLLMM_IM(COCC(:,:,ISP),PP)
!            CALL DUMP_DLLMM_IM(CRHODE(:,:,NIP,ISP),PP)
#endif
!            WRITE(*,*) 'spin',ISP
!            CALL DUMP_DLLMM(COCC(:,:,ISP),PP)
!            CALL DUMP_DLLMM(CRHODE(:,:,NIP,ISP),PP)

         ENDDO
         IBASE=IBASE+IADD
         IF (IBASE > ISIZE+1) THEN
            WRITE(0,*) 'internal error SET_RHO_PAW: insufficient space'
            STOP
         ENDIF
         CRHODE(:,:,NIP,:)=COCC(:,:,:)
         
      ENDDO ion
#ifndef RHOLM_complex
      IF (IBASE /= ISIZE+1) THEN
         WRITE(0,*) 'internal error SET_RHO_PAW: RHOLM_STORE wrong allocation',IBASE, ISIZE+1
         STOP
      ENDIF
#endif

#ifdef realmode
       CALLMPI( M_sum_d(WDES%COMM_INTER, CRHODE, SIZE(CRHODE)))
#else
       CALLMPI( M_sum_z(WDES%COMM_INTER, CRHODE, SIZE(CRHODE)))
#endif

    END SUBROUTINE RETRIVE_RHO_PAW


  END MODULE paw
