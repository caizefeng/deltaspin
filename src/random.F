#include "symbol.inc"
!===========================================================================
! RCS:  $Id: random.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
! This random number generator originally appeared in Toward a Universal
! Random Number Generator by George Marsaglia and Arif Zaman.
! Florida State University Report: FSU-SCRI-87-50 (1987)
!
! It was later modified by F. James and published in A Review of Pseudo-
! random Number Generators
!
! Some final small modifications have been done by J. Furthmueller
! Technical University of Vienna, November 1993
!
! THIS IS THE BEST KNOWN RANDOM NUMBER GENERATOR AVAILABLE.
!       (However, a newly discovered technique can yield
!         a period of 10^600. But that is still in the development stage.)
!
! It passes ALL of the tests for random number generators and has a period
!   of 2^144, is completely portable (gives bit identical results on all
!   machines with at least 24-bit mantissas in the floating point
!   representation).
!
! The algorithm is a combination of a Fibonacci sequence (with lags of 97
!   and 33, and operation "subtraction plus one, modulo one") and an
!   "arithmetic sequence" (using subtraction).
!
! On a Vax 11/780, this random number generator can produce a number in
!    13 microseconds.
! (Note by J. Furthmueller: in 2.5 microseconds on a IBM RS6000/Model 580)
!========================================================================
      BLOCK DATA RMARIN_INI
      USE prec
      IMPLICIT REAL(q) (A-H,O-Z)

      LOGICAL TEST
      REAL(q) U(97), C, CD, CM
      INTEGER I97, J97
      COMMON /RASET1/ U, C, CD, CM, I97, J97, TEST
      DATA TEST /.FALSE./
      END

      SUBROUTINE RMARIN(IJ,KL)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
! This is the initialization routine for the random number generator RANMAR()
! NOTE: The seed variables can have values between:    0 <= IJ <= 31328
!                                                      0 <= KL <= 30081
! The random number sequences created by these two seeds are of sufficient
! length to complete an entire calculation with. For example, if several
! different groups are working on different parts of the same calculation,
! each group could be assigned its own IJ seed. This would leave each group
! with 30000 choices for the second seed. That is to say, this random
! number generator can create 900 million different subsequences -- with
! each subsequence having a length of approximately 10^30.
!
! Use IJ = 1802 & KL = 9373 to test the random number generator. The
! subroutine RANMAR should be used to generate 20000 random numbers.
! Then display the next six random numbers generated multiplied by 4096*4096
! If the random number generator is working properly, the random numbers
! should be:
!           6533892.0  14220222.0  7275067.0
!           6172232.0  8354498.0   10633180.0


      REAL(q) U(97), C, CD, CM
      INTEGER I97, J97
      LOGICAL TEST
      COMMON /RASET1/ U, C, CD, CM, I97, J97, TEST

      IF ( (IJ<0) .OR. (IJ>31328) .OR. &
     &    (KL<0) .OR. (KL>30081) ) THEN
          PRINT '(A)',' The first random number seed must have a value between 0 and 31328'
          PRINT '(A)',' The second seed must have a value between 0 and 30081'
          STOP
      ENDIF

      I = MOD(IJ/177, 177) + 2
      J = MOD(IJ    , 177) + 2
      K = MOD(KL/169, 178) + 1
      L = MOD(KL,     169)

      DO 2 II = 1, 97
         S = 0.0_q
         T = 0.5_q
         DO 3 jj = 1, 24
            M = MOD(MOD(I*J, 179)*K, 179)
            I = J
            J = K
            K = M
            L = MOD(53*L+1, 169)
            IF (MOD(L*M, 64) >= 32) THEN
               S = S + T
            ENDIF
            T = 0.5_q * T
3        CONTINUE
         U(II) = S
2     CONTINUE

      C = 362436.0_q / 16777216.0_q
      CD = 7654321.0_q / 16777216.0_q
      CM = 16777213.0_q /16777216.0_q

      I97 = 97
      J97 = 33

      TEST = .TRUE.

      RETURN
      END



      FUNCTION RANMAR()
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
! This is the random number generator proposed by George Marsaglia in
! Florida State University Report: FSU-SCRI-87-50
! It was slightly modified by F. James to produce an array of pseudorandom
! numbers.

      REAL(q) U(97), C, CD, CM, RANMAR
      INTEGER I97, J97
      LOGICAL TEST
      COMMON /RASET1/ U, C, CD, CM, I97, J97, TEST

      INTEGER IVEC

      IF (.NOT. TEST) THEN
         PRINT '(A)',' Call the init routine (RMARIN) before calling RANMAR!'
         PRINT '(A)',' Initializing now with built-in seeds 1802 and 9373 ...'
         CALL RMARIN(1802,9373)
      ENDIF

      UNI = U(I97) - U(J97)
      IF ( UNI < 0.0_q ) UNI = UNI + 1.0_q
      U(I97) = UNI
      I97 = I97 - 1
      IF (I97 == 0) I97 = 97
      J97 = J97 - 1
      IF (J97 == 0) J97 = 97
      C = C - CD
      IF ( C < 0.0_q ) C = C + CM
      UNI = UNI - C
      IF ( UNI < 0.0_q ) UNI = UNI + 1.0_q
      RANMAR = UNI

      RETURN
      END



      FUNCTION RANE()
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
! Simplified call interface to RANMAR using a fixed initialisation ...

      REAL(q) RANMAR, RANE
      EXTERNAL RANMAR, RMARIN
      INTEGER ICALL

      SAVE ICALL,IJ,KL
      DATA ICALL /0/, IJ /1802/, KL /9373/

      IF ( ICALL == 0 )  CALL RMARIN(IJ,KL)
      ICALL = ICALL + 1

      RANE=RANMAR()

      RETURN
      END



      FUNCTION RANG(RNULL,WIDTH)
      USE prec

      IMPLICIT REAL(q) (A-H,O-Z)
! This subroutine produces a normal distribution (Gaussian distribution)

      REAL(q) RNULL, WIDTH, TWOPI, RANE, RANG
      PARAMETER ( TWOPI = 6.283185307179586_q )
      EXTERNAL RANE

      RANG = COS( TWOPI*RANE() ) * SQRT( -2._q*LOG(RANE()) )
      RANG = WIDTH * RANG  +  RNULL

      RETURN
      END

      MODULE RANDOM_SEEDED
        USE prec
        USE constant
        IMPLICIT NONE

        PRIVATE
        PUBLIC :: RANE_ION
      CONTAINS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
      SUBROUTINE RMARIN_ION(IJ,KL,U,C,CD,CM,I97,J97,TEST)
      USE prec
      INTEGER :: I,J,K,L,M,IJ,KL,II,JJ
      REAL(q) :: S,T
      REAL(q) U(97)
      LOGICAL:: TEST
      REAL(q) ::  C, CD, CM
      INTEGER :: I97, J97

      IF ( (IJ<0) .OR. (IJ>31328) .OR. &
     &    (KL<0) .OR. (KL>30081) ) THEN
          PRINT '(A)',' The first random number seed must have a value between 0 and 31328'
          PRINT '(A)',' The second seed must have a value between 0 and 30081'
          STOP
      ENDIF

      I = MOD(IJ/177, 177) + 2
      J = MOD(IJ    , 177) + 2
      K = MOD(KL/169, 178) + 1
      L = MOD(KL,     169)

      DO II = 1, 97
         S = 0.0_q
         T = 0.5_q
         DO jj = 1, 24
            M = MOD(MOD(I*J, 179)*K, 179)
            I = J
            J = K
            K = M
            L = MOD(53*L+1, 169)
            IF (MOD(L*M, 64) >= 32) THEN
               S = S + T
            ENDIF
            T = 0.5_q * T
         ENDDO
         U(II) = S
      ENDDO

      C = 362436.0_q / 16777216.0_q
      CD = 7654321.0_q / 16777216.0_q
      CM = 16777213.0_q /16777216.0_q

      I97 = 97
      J97 = 33

      TEST = .TRUE.


      END SUBROUTINE RMARIN_ION

      SUBROUTINE RANMAR_ION(RDUMMY,U, C, CD, CM, I97, J97, TEST)
      USE prec
      REAL(q) U(97), C, CD, CM, RDUMMY
      REAL(q) :: UNI
      INTEGER I97, J97
      LOGICAL TEST

      INTEGER IVEC

      IF (.NOT. TEST) THEN
         PRINT '(A)',' Call the init routine (RMARIN_ION) before calling RANMAR_ION!'
         PRINT '(A)',' Initializing now with built-in seeds 1802 and 9373 ...'
         CALL RMARIN_ION(1802,9373,U,C,CD,CM,I97,J97,TEST)
      ENDIF

      UNI = U(I97) - U(J97)
      IF ( UNI < 0.0_q ) UNI = UNI + 1.0_q
      U(I97) = UNI
      I97 = I97 - 1
      IF (I97 == 0) I97 = 97
      J97 = J97 - 1
      IF (J97 == 0) J97 = 97
      C = C - CD
      IF ( C < 0.0_q ) C = C + CM
      UNI = UNI - C
      IF ( UNI < 0.0_q ) UNI = UNI + 1.0_q
      RDUMMY = UNI
      END SUBROUTINE RANMAR_ION

      SUBROUTINE RANE_ION(RNG_ION,PUT,GET)
      USE prec
      INTEGER, OPTIONAL :: PUT(3),GET(3)
      INTEGER,PARAMETER :: IJ_MAX=31328
      INTEGER,PARAMETER ::KL_MAX=30081
      INTEGER,PARAMETER :: SEED1_MAX=900000000
      INTEGER,SAVE :: RNG_STATE1=54217137 !IJ*(KL_MAX+1)+KL
      INTEGER,SAVE :: RNG_STATE2=0            !c number of rannd. numbers generated
      INTEGER,SAVE :: RNG_STATE3=0            !c up to this point is:
      INTEGER,PARAMETER :: COUNTLARGE=1000000 !c RNG_STATE3*COUNTLARGE+RNG_STATE2
      !INTEGER,PARAMETER :: COUNTLARGE=1000 !c RNG_STATE3*COUNTLARGE+RNG_STATE2
      REAL(q) :: rng_ion,rdummy
      INTEGER :: IMODE,i,j
      LOGICAL,SAVE :: LFIRST=.TRUE.,LINIT=.FALSE.,TEST=.FALSE.
      REAL(q),SAVE :: U(97), C, CD, CM
      INTEGER,SAVE :: I97, J97
      INTEGER, SAVE :: IJ, KL,ICALL

      IF (LFIRST) THEN
        ICALL=0
        IJ=1802
        KL=9373
        C = 362436.0_q / 16777216.0_q
        CD = 7654321.0_q / 16777216.0_q
        CM = 16777213.0_q /16777216.0_q
        I97 = 97
        J97 = 33
        TEST=.FALSE.
        LINIT=.TRUE.
      ENDIF
    
      !c just return the RNG seed (and initialize if needed)
      IF (PRESENT(GET)) THEN
        IMODE=2
        GET(1)=RNG_STATE1
        GET(2)=RNG_STATE2
        GET(3)=RNG_STATE3
        !write(*,*) "get?"
        IF ((.NOT. LFIRST)) RETURN
      !c reseed and reinitialise if SEED is provided
      ELSE IF (PRESENT(PUT)) THEN
        IMODE=1
        !write(*,*) 'put?',SEED
        !c ensure that SEED(1) doesn't exceed
        !c max. sensible value
        RNG_STATE1=MOD(PUT(1),SEED1_MAX)
        RNG_STATE2=MOD(PUT(2),COUNTLARGE)
        RNG_STATE3=PUT(3)+PUT(2)/COUNTLARGE

        !c regenerate pair of seeds used originally
        IJ=RNG_STATE1/(KL_MAX+1)
        KL=RNG_STATE1-(KL_MAX+1)*IJ

        LINIT=.TRUE.
      ELSE
        IMODE=0
      ENDIF

      IF (LINIT) THEN
         CALL RMARIN_ION(IJ,KL,U,C,CD,CM,I97,J97,TEST)

         !c skip RNG_STATE3*COUNTLONG+RNG_STATE2 numbers 
         !c (presumably already used in a previous run) 
         !c on initialization
         DO i=1,RNG_STATE3
           DO j=1, COUNTLARGE
             CALL RANMAR_ION(rdummy,U, C, CD, CM, I97, J97, TEST)
           ENDDO
         ENDDO

         DO i=1,RNG_STATE2
           CALL RANMAR_ION(rdummy,U, C, CD, CM, I97, J97, TEST)
         ENDDO
         LINIT=.FALSE.
         !write(*,*) "rng_init_finish"
      ENDIF

      !c return random number if no special action is taken
      IF (IMODE==0) THEN
       ! write(*,*) "rng_num?",RNG_STATE1,RNG_STATE2
        RNG_STATE2=RNG_STATE2+1
        IF (RNG_STATE2>=COUNTLARGE) THEN
          RNG_STATE2=RNG_STATE2-COUNTLARGE
          RNG_STATE3=RNG_STATE3+1
        ENDIF
        CALL RANMAR_ION(RNG_ION,U, C, CD, CM, I97, J97, TEST)
      ENDIF

      IF (LFIRST) LFIRST=.FALSE.
    END SUBROUTINE RANE_ION

  END MODULE RANDOM_SEEDED
