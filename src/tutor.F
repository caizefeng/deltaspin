#include "symbol.inc"
!********** TUTORIAL PACKAGE -- GIVE WARNINGS AND ADVICES **************
! RCS:  $Id: tutor.F,v 1.6 2003/06/27 13:22:23 kresse Exp kresse $
!                                                                      *
      SUBROUTINE VTUTOR(TYPE,WTOPIC,RDAT,NR,IDAT,NI, &
     &                  CDAT,NC,LDAT,NL,IU,IDIOT)
      USE prec
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      IMPLICIT COMPLEX(q) (C)
!                                                                      *
!***********************************************************************
!                                                                      *
!  This routine gives warnings, advices, error messages on very very   *
!  important things which are often done wrong by bloody newbies ...   *
!                                                                      *
!  Variables:                                                          *
!                                                                      *
!    TYPE   tells us what it is (fatal E rror, W arning, A dvice       *
!           or fatal error which makes it necessary to S top ...       *
!    TOPIC  contains some identifier string about what to talk ...     *
!    RDAT,IDAT,CDAT and LDAT are arrays containing possible data       *
!    being necessary for the one or the other message (type Real,      *
!    Integer, Complex, Logical ...) with NR,NI,NC and NL being the     *
!    dimensions of these arrays ... .                                  *
!    IDIOT  flags the 'expert level' of the user:                      *
!           0: 'complete expert' (no messages at all)                  *
!           1: 'almost complete expert' (only hard errors)             *
!           2: 'somehow experienced user' (only warnings and errors)   *
!           3: 'complete idiot' (all kind of messages ...)             *
!                                                                      *
!  On output you should receive messages on I/O-unit IU telling you    *
!  what to do or better not to do (i.e. what you have done wrong ...)  *
!                                                                      *
!***********************************************************************

      CHARACTER (1) TYPE
      CHARACTER(255) TOPIC
      CHARACTER (*) WTOPIC
      LOGICAL       LDAT, LIO
      DIMENSION RDAT(NR),IDAT(NI),CDAT(NC),LDAT(NL)

      LIO=.TRUE.
      IF (IU<0) LIO=.FALSE.

      TOPIC=WTOPIC
      CALL STRIP(TOPIC,LTOPIC,'B')

! Header of the message (if there is a message at all ...)
      IF ((TYPE=='E').OR.(TYPE=='S')) THEN
! the 'complete expert' needs no error messages, warnings or advices ...
         IF (IDIOT<=0) RETURN
         IF (IU>=0) WRITE(IU,1)
      ELSE IF ((TYPE=='W').OR.(TYPE=='U')) THEN
! the 'complete expert' needs no error messages, warnings or advices ...
! the 'almost complete expert' needs no warnings or advices ...
         IF ((TYPE=='W').AND.(IDIOT<=1)) RETURN
! ... but 'U'rgent warning (almost like error!) shall be given to all
! except for 'the complete expert' needing no help at all  :-)
         IF ((TYPE=='U').AND.(IDIOT<=0)) RETURN
         IF (LIO) WRITE(IU,2)
      ELSE
! the 'complete expert' needs no error messages, warnings or advices ...
! the 'almost complete expert' needs no warnings or advices ...
! the 'somehow experienced user' needs no advices ...
         IF (IDIOT<=2) RETURN
         IF (LIO) WRITE(IU,3)
      ENDIF
! ... but the 'complete idiot' needs all ...                   :-)
    1 FORMAT(/' ------------------------------------------------', &
     & '----------------------------- '/, &
     & '|                                                ', &
     & '                             |'/, &
     & '|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR ', &
     & '     ###     ###     ###     |'/, &
     & '|     E        R     R  R     R  O     O  R     R', &
     & '     ###     ###     ###     |'/, &
     & '|     E        R     R  R     R  O     O  R     R', &
     & '     ###     ###     ###     |'/, &
     & '|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR ', &
     & '      #       #       #      |'/, &
     & '|     E        R   R    R   R    O     O  R   R  ', &
     & '                             |'/, &
     & '|     E        R    R   R    R   O     O  R    R ', &
     & '     ###     ###     ###     |'/, &
     & '|     EEEEEEE  R     R  R     R  OOOOOOO  R     R', &
     & '     ###     ###     ###     |'/, &
     & '|                                                ', &
     & '                             |')


    2 FORMAT(/ &
     & ' -----------------------------------------------', &
     & '------------------------------ '/, &
     & '|                                               ', &
     & '                              |'/, &
     & '|           W    W    AA    RRRRR   N    N  II  ', &
     & 'N    N   GGGG   !!!           |'/, &
     & '|           W    W   A  A   R    R  NN   N  II  ', &
     & 'NN   N  G    G  !!!           |'/, &
     & '|           W    W  A    A  R    R  N N  N  II  ', &
     & 'N N  N  G       !!!           |'/, &
     & '|           W WW W  AAAAAA  RRRRR   N  N N  II  ', &
     & 'N  N N  G  GGG   !            |'/, &
     & '|           WW  WW  A    A  R   R   N   NN  II  ', &
     & 'N   NN  G    G                |'/, &
     & '|           W    W  A    A  R    R  N    N  II  ', &
     & 'N    N   GGGG   !!!           |'/, &
     & '|                                               ', &
     & '                              |')

    3 FORMAT(/ &
     & ' -----------------------------------------------', &
     & '------------------------------ '/, &
     & '|                                               ', &
     & '                              |'/, &
     & '|  ADVICE TO THIS USER RUNNING ''VASP/VAMP''  ', &
     & ' (HEAR YOUR MASTER''S VOICE ...):  |'/, &
     & '|                                               ', &
     & '                              |')

    4 FORMAT( &
     & '|                                               ', &
     & '                              |'/, &
     & ' -----------------------------------------------', &
     & '------------------------------ '/)

    5 FORMAT( &
     & '|                                               ', &
     & '                              |'/, &
     & '|      ---->  I REFUSE TO CONTINUE WITH THIS ', &
     & 'SICK JOB ..., BYE!!! <----       |'/, &
     & '|                                               ', &
     & '                              |'/, &
     & ' -----------------------------------------------', &
     & '------------------------------ '/)

! Now following the long long long long code printing all messages ...:
! =====================================================================

      IF (LIO.AND. TOPIC(1:LTOPIC)=='REAL-SPACE WITHOUT OPTIMIZATION') THEN
         WRITE(IU,'(A)')'|     The real-space-projection scheme for '// &
     &                  'the treatment of the nonlocal      |'
         WRITE(IU,'(A)')'|     pseudopotentials has been switched on'// &
     &                  ' -- but on file POTCAR I have      |'
         WRITE(IU,'(A)')'|     not found any entries signalling me '// &
     &                  'that you have ever performed a      |'
         WRITE(IU,'(A)')'|     real-space optimization of all '// &
     &                  'nonlocal projectors! BE WARNED that      |'
         WRITE(IU,'(A)')'|     a calculation using the real-space-'// &
     &                  'projection scheme together with      |'
         WRITE(IU,'(A)')'|     nonlocal projectors which were not '// &
     &                  'real-space-optimized (according      |'
         WRITE(IU,'(A)')'|     to the plane-wave-cutoff used in this '// &
     &                  'calculation) might give more      |'
         WRITE(IU,'(A)')'|     or less inaccurate results         '// &
     &                  '                                     |'
         WRITE(IU,'(A)')'|     I hope you know what you are doing   '// &
     &                  '                                   |'
         GOTO 99999
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='SPLINE INTERPOLATE PROJECTORS') THEN
         WRITE(IU,'(A)')'|     The projection operators are construc'// &
     &                  'ted using a spline interpolation.  |'
         WRITE(IU,'(A)')'|     This is needed for the calculation of'// &
     &                  ' chemical shifts by means of       |'
         WRITE(IU,'(A)')'|     linear response, but yields slightly '// &
     &                  'different total energies compared  |'
         WRITE(IU,'(A)')'|     to the default setup of the projector'// &
     &                  's (NLSPLINE=.FALSE.).              |'
         GOTO 99999
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='VASP.4.4') THEN
         WRITE(IU,'(A)')'|     You are running vasp.4.5 in the vasp.4.4 compatibility mode             |'
         WRITE(IU,'(A)')'|     vasp.4.5 has some numerical improvements, which are not applied in this |'
         WRITE(IU,'(A)')'|     mode (charge at unbalanced lattice vectors are no longer zeroed,        |'
         WRITE(IU,'(A)')'|           PAW augmentation charges are integrated more accurately)          |'
        GOTO 99999
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NO REAL-SPACE AND YOU COULD') THEN
         WRITE(IU,'(A)')'|      You have a (more or less) ''large '// &
     &              'supercell'' and for larger cells       |'
         WRITE(IU,'(A)')'|      it might be more efficient to '// &
     &           'use real space projection operators      |'
         WRITE(IU,'(A)')'|      So try LREAL= Auto  in the INCAR   '// &
     &                 'file.                               |'
         WRITE(IU,'(A)')'|      Mind: If you want to do a very accur'// &
     &                 'ate calculations keep the          |'
         WRITE(IU,'(A)')'|      reciprocal projection scheme         ' &
     &                 //' (i.e. LREAL=.FALSE.)             |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NO REAL-SPACE AND YOU SHOULD') THEN
         WRITE(IU,'(A)')'|      You have a (more or less) ''large '// &
     &              'supercell'' and for larger cells       |'
         WRITE(IU,'(A)')'|      it might be more efficient to '// &
     &           'use real space projection opertators     |'
         WRITE(IU,'(A)')'|      So try LREAL= Auto  in the INCAR   '// &
     &                 'file.                               |'
         WRITE(IU,'(A)')'|      Mind:          For very '// &
     &                ' accurate calculation you might also keep the  |'
         WRITE(IU,'(A)')'|      reciprocal projection scheme         ' &
     &                 //' (i.e. LREAL=.FALSE.)             |'

      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='IALGO8') THEN
         WRITE(IU,'(A)')'|      Recently Corning got a patent for the'// &
     &                  ' Teter Allan Payne algorithm      |'
         WRITE(IU,'(A)')'|      therefore VASP.4.5 does not support IALG'// &
     &                  'O=8 any longer                 |'
         WRITE(IU,'(A)')'|      a much faster algorithm, IALGO=38, is '// &
     &                  ' now implemented in VASP         |'
         WRITE(IU,'(A)')'|      this algorithm is a blocked Davidson'// &
     &                  ' like method and as reliable as    |'
         WRITE(IU,'(A)')'|      IALGO=8 used to be                  '// &
     &                  '                                   |'
         WRITE(IU,'(A)')'|      for ultimate performance IALGO=48 is'// &
     &                  ' still the method of choice        |' 
         WRITE(IU,'(A)')'|      -- SO MUCH ABOUT PATENTS :)         '// &
     &                  '                                   |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='REAL-SPACE NOMORE RECOMMENDED') THEN
         WRITE(IU,'(A)')'|      You have a (more or less) ''small '// &
     &              'supercell'' and for smaller cells      |'
         WRITE(IU,'(A)')'|      it is recommended  to use the '// &
     &            'reciprocal-space projection scheme!      |'
         WRITE(IU,'(A)')'|      The real space optimization is not  '// &
     &                  'efficient for small cells and it   |'
         WRITE(IU,'(A)')'|      is also less accurate ...            ' &
     &                 //'                                  |'
         WRITE(IU,'(A)')'|      Therefore set LREAL=.FALSE. in the  '// &
     &                  'INCAR file                         |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='WRONG OPTIMZATION REAL-SPACE') THEN
         WRITE(IU,'(A)') '|      One real space projector is optimized' &
     &                 //' for                              |'
         WRITE(IU,'(A,F10.2,A)') &
     &              '|      E    =',RDAT(1),', eV  but you are using a ' &
     &               //' cutoff of                   |'
         WRITE(IU,'(A,F10.2,A,F10.2,A)')  &
     &              '|      ENMAX=',RDAT(2),' eV  ( QCUT=',RDAT(3), &
     &              ' a.u.)                           |'
         WRITE(IU,'(A)') '|      This makes no sense reoptimize the  ' &
     &                 //' projector                         |'
         WRITE(IU,'(A)') '|      with the a.u. value given above      ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='long lattice') THEN
         WRITE(IU,'(A)') '|      One of the lattice vectors is very lo' &
     &                 //'ng (>50 A), but AMIN is rather    |'
         WRITE(IU,'(A,F10.2,A)') &
     &              '|      large. This can spoil convergence since cha' &
     &               //'rge sloshing might occur    |'
         WRITE(IU,'(A)') '|      along the long lattice vector. If pr' &
     &                 //'oblems with convergence are        |'
         WRITE(IU,'(A)') '|      observed, try to decrease AMIN to a ' &
     &                 //'smaller values (e.g. 0.01).        |'
         WRITE(IU,'(A)') '|      Note: this warning only applies if t' &
     &                 //'he selfconsistency cycle is used.  |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIFFERENT XCGRAD TYPES') THEN
         WRITE(IU,'(A)') '|      You have build up your multi-ion-type' &
     &                 //' POTCAR file out of POTCAR        |'
         WRITE(IU,'(A)') '|      files with incompatible specifications' &
     &                  //' for the XC-types used to        |'
         WRITE(IU,'(A)') '|      generate the pseudopotential. This '// &
     &                 'makes no sense at all!! What        |'
         WRITE(IU,'(A,I2,A,I3,A,I2,A)') '|      I found is LEXCH  = ', &
     &             IDAT(1),'  for atom types <= ',IDAT(3)-1, &
     &                    ' but LEXCH  = ',IDAT(2),'        |'
         WRITE(IU,'(A,I3,A)') '|      was found for atom type = ', &
     &            IDAT(3),'. Use identical XC-functionals for        |'
         WRITE(IU,'(A)') '|      the pseudopotential generation for '// &
     &                 'all atom types, please ... !        |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='Number of electrons') THEN
         WRITE(IU,'(A)') '|      The number of bands is not sufficient' &
     &                 //' to hold all electrons.           |'
         WRITE(IU,'(A)') '|      I am sorry, but this calculation does ' &
     &                  //' not make sense.                 |'
         WRITE(IU,'(A)') '|      If you want to calculate only selec'// &
     &                 'ted states you need to set          |'
         WRITE(IU,'(A)') '|      either EFERMI oder EREF. EREF speci'// &
     &                 'fies around which energy states     |'
         WRITE(IU,'(A)') '|      are supposed to be calculated.     '// &
     &                 '                                    |'
         WRITE(IU,'(A,I8,A,I8,A)') '|      I found NBANDS    = ', &
     &             IDAT(1),'       NELECT  =',IDAT(2),'                   |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='NBANDS changed') THEN
         WRITE(IU,'(A)') '|      The number of bands has been changed ' &
     &                 //'from the values supplied          |'
         WRITE(IU,'(A)') '|      in the INCAR file. This is a result of' &
     &                  //' running the parallel version.   |'
         WRITE(IU,'(A)') '|      The orbitals not found in the WAVEC'// &
     &                 'AR file will be initialized with    |'
         WRITE(IU,'(A)') '|      random numbers, which is usually ad'// &
     &                 'equate. For correlated              |'
         WRITE(IU,'(A)') '|      calculations, however, you should r'// &
     &                 'edo the groundstate calculation.    |'
         WRITE(IU,'(A,I8,A,I8,A)') '|      I found NBANDS    = ', &
     &             IDAT(1),'  now  NBANDS  =',IDAT(2),'                   |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIFFERENT LDA-XC TYPES') THEN
         WRITE(IU,'(A)') '|      You have build up your multi-ion-type' &
     &                 //' POTCAR file out of POTCAR        |'
         WRITE(IU,'(A)') '|      files with incompatible specifications' &
     &                  //' for the XC-types used to        |'
         WRITE(IU,'(A)') '|      generate the pseudopotential. This '// &
     &                 'makes no sense at all!! What        |'
         WRITE(IU,'(A,I2,A,I3,A,I2,A)') '|      I found is LEXCH  = ', &
     &             IDAT(1),'  for atom types <= ',IDAT(3)-1, &
     &                    ' but LEXCH  = ',IDAT(2),'        |'
         WRITE(IU,'(A,I3,A)') '|      was found for atom type = ', &
     &            IDAT(3),'. Use identical XC-functionals for        |'
         WRITE(IU,'(A)') '|      the pseudopotential generation for '// &
     &                 'all atom types, please ... !        |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIFFERENT REL-XC TYPES') THEN
         WRITE(IU,'(A)') '|      You have build up your multi-ion-type' &
     &                 //' POTCAR file out of POTCAR        |'
         WRITE(IU,'(A)') '|      files with incompatible specifications' &
     &                  //' for the XC-types used to        |'
         WRITE(IU,'(A)') '|      generate the pseudopotential. This '// &
     &                 'makes no sense at all!! What        |'
         WRITE(IU,'(A)') '|      I found is that the flag which '// &
     &             'switches on/off the relativistic        |'
         WRITE(IU,'(A,L1,A,I3,A)') '|      corrections has been set .', &
     &                        LDAT(1),'. for atom types <= ', &
     &                        IDAT(1)-1,' but it was        |'
         WRITE(IU,'(A,L1,A,I3,A)') '|      set .',LDAT(2), &
     &           '. for atom type no. ',IDAT(1), &
     &           '. Use identical XC-functionals for        |'
         WRITE(IU,'(A)') '|      the pseudopotential generation for '// &
     &                 'all atom types, please ... !        |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIFFERENT XC FILES') THEN
         WRITE(IU,'(A)') '|      The XC type on the POTCAR file is not' &
     &                 //' compatibel with the XC type on   |'
         WRITE(IU,'(A)') '|      the EXHCAR file                       ' &
     &                  //'                                 |'
         GOTO 99999
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='AM05 spin') THEN
         WRITE(IU,'(A)') '|      The AM05 functional for spin polarize' &
     &                 //'d calculations is not supported.  |'
         WRITE(IU,'(A)') '|      Maybe in a future release ...         ' &
     &                  //'                                 |'
         GOTO 99999
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOLDAU') THEN
         WRITE(IU,'(A)') '|      VASP supports LDA+U only for PAW pote' &
     &                 //'ntials and not for US or NC       |'
         WRITE(IU,'(A)') '|      pseudopotentials                      ' &
     &                  //'                                 |'
         WRITE(IU,'(A)') '|      please restart with with the approriat' &
     &                  //'e potentials                     |'
         GOTO 99999
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='Double counting') THEN
         WRITE(IU,'(A)') '|      LDA+U in combination with GW  calculat' &
     &                 //'ions causes a double             |'
         WRITE(IU,'(A)') '|      counting of unidentifiable many-body p' &
     &                  //'erturbation terms.               |'
         GOTO 99999
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOWANNIER') THEN
         WRITE(IU,'(A)') '|      VASP supports the calculation of maxi' &
     &                 //'mally localized wannier functions |'
         WRITE(IU,'(A)') '|      only in the Gamma-only version of the' &
     &                 //' code.                            |'
         GOTO 99999
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LREALA') THEN
         WRITE(IU,'(A)') '|      LREAL=A is not well tested yet, pleas' &
     &                 //'e use LREAL=O                     |'
         WRITE(IU,'(A)') '|      or test your results very carefully   ' &
     &                  //'                                 |'
         GOTO 99999
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIFFERENT SLATER-XC TYPES') THEN
         WRITE(IU,'(A)') '|      You have build up your multi-ion-type' &
     &                 //' POTCAR file out of POTCAR        |'
         WRITE(IU,'(A)') '|      files with incompatible specifications' &
     &                  //' for the XC-types used to        |'
         WRITE(IU,'(A)') '|      generate the pseudopotential. This '// &
     &                 'makes no sense at all!! What        |'
         WRITE(IU,'(A,F9.6,A,I3,A)') '|      I found is slater '// &
     &                       'parameter = ',RDAT(1),' for atom '// &
     &                       'types <= ',IDAT(1)-1,'        |'
         WRITE(IU,'(A,F9.6,A,I3,A)') '|      but slater parameter = ', &
     &               RDAT(2),' was found for atom type = ', &
     &                                    IDAT(1),'.        |'
         WRITE(IU,'(A)') '|      Use identical slater-parameters '// &
     &              'in the exchange functionals for        |'
         WRITE(IU,'(A)') '|      the pseudopotential generation for '// &
     &                 'all atom types, please ... !        |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='partial DOS') THEN
         WRITE(IU,'(A)') '|      The partial DOS and the PROCAR file are' &
     &                 //' not evaluated for NPAR/=1      |'
         WRITE(IU,'(A)') '|      please rerun with NPAR=1             '&
     &                  //'                                  |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='nooptics') THEN
         WRITE(IU,'(A)') '|      The file OPTICS can be written only for' &
     &                 //' NPAR=1                         |'
         WRITE(IU,'(A)') '|      optical properties can be found in va'&
     &                  //'sprun.xml and OUTCAR, however.    |'
         WRITE(IU,'(A)') '|      If you need the OPTICS file, rerun with '&
     &                  //'NPAR=1                         |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='POSITION') THEN
         WRITE(IU,'(A)') '|      The distance between some ions is very'&
     &                 //' small                           |'
         WRITE(IU,'(A)') '|      please check the nearest neigbor list'&
     &                  //' in the OUTCAR file               |'
         WRITE(IU,'(A)') '|          I HOPE YOU KNOW, WHAT YOU ARE ' &
     &                  //' DOING                               |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='CENTER OF MASS DRIFT') THEN
         WRITE(IU,'(A)') '|      The initial velocities result in a cen'&
     &                 //'ter of mass drift but            |'
         WRITE(IU,'(A)') '|      there must be no drift.              '&
     &                  //'                                  |'
         WRITE(IU,'(A)') '|      The drift will be removed !          '&
     &                  //'                                  |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='ENFORCED LDA') THEN
         WRITE(IU,'(A)') '|      You enforced a specific xc-type in the' &
     &                 //' INCAR file,                     |'
         WRITE(IU,'(A)') '|      a different type was found on the ' &
     &                  //'POTCAR file                          |'
         WRITE(IU,'(A)') '|          I HOPE YOU KNOW, WHAT YOU ARE ' &
     &                  //' DOING                               |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIPOL EPSILON') THEN
         WRITE(IU,'(A)') '|      You specified a dielectrion constant  ' &
     &                 //' different from 1 (EPSILON)      |'
         WRITE(IU,'(A)') '|      this indicates that you perform ca' &
     &                  //'lculations for the bulk.             |'
         WRITE(IU,'(A)') '|      At the same time LDIPOL is set to ' &
     &                  //'.TRUE., a mode which does not work   |'
         WRITE(IU,'(A)') '|      properly for bulk systems!        ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      I hope you know what you are doing' &
     &                  //' !!!                                 |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='DIPOL CUBIC') THEN
         WRITE(IU,'(A)') '|      LDIPOL = .TRUE. must be selected only ' &
     &                 //'for cubic supercells, since      |'
         WRITE(IU,'(A)') '|      the quadrupole corrections are cur' &
     &                  //'rently only implemented for this     |'
         WRITE(IU,'(A)') '|      specific geometry.                ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      Please change your box, or set LDI' &
     &                  //'POL=.FALSE.                          |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='ALGO=A ISMEAR') THEN
         WRITE(IU,'(A)') '|      ALGO=A and IALGO=5X tend to fail with ' &
     &                 //'the tetrahedron method           |'
         WRITE(IU,'(A)') '|      (e.g. Bloechls method ISMEAR=-5 is not' &
     &                 //' variational)                    |'
         WRITE(IU,'(A)') '|      please switch to IMSEAR=0-n, except' &
     &                  //' for DOS calculations               |'
         WRITE(IU,'(A)') '|      For DOS calculations use IALGO=53 afte' &
     &                 //'r preconverging with ISMEAR>=0   |'
         WRITE(IU,'(A)') '|          I HOPE YOU KNOW, WHAT YOU ARE ' &
     &                  //' DOING                               |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='model GW') THEN
         WRITE(IU,'(A)') '|      Model GW is not a variational method  ' &
     &                 //'                                 |'
         WRITE(IU,'(A)') '|      The conjugate gradient algorithm there' &
     &                 //'fore does not work               |'
         WRITE(IU,'(A)') '|      please switch to IALGO=53-55       ' &
     &                  //'                                    |'
         WRITE(IU,'(A)') '|          I HOPE YOU KNOW, WHAT YOU ARE ' &
     &                  //' DOING                               |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='HFUSPP') THEN
         WRITE(IU,'(A)') '|      HF is only implemented for the PAW method' &
     &                 //' and for norm conserving      |'
         WRITE(IU,'(A)') '|      pseudopotentials.                 ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      One of the POTCAR data sets corres' &
     &                  //'ponds to an US pseudopotential       |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LHFLINEARRESPONSE') THEN
         WRITE(IU,'(A)') '|      HF is not implemented in the linear respo' &
     &                 //'nse routines                  |'
         WRITE(IU,'(A)') '|      Please use the finite difference v' &
     &                  //'ersions.                             |'
         WRITE(IU,'(A)') '|                                        ' &
     &                  //'                                     |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LINEARRES POSCAR') THEN
         WRITE(IU,'(A)') '|      The POSCAR file is problematic for the li' &
     &                 //'near response routines,       |'
         WRITE(IU,'(A)') '|      since the displacement is not alon' &
     &                  //'g the cartesian directions.          |'
         WRITE(IU,'(A)') '|      Please send this POSCAR to vasp su' &
     &                  //'pport, we will try to fix the issue. |'
         WRITE(IU,'(A)') '|      You can use finite differences for' &
     &                  //' the time being.                     |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LINEARRESPONSE LREAL') THEN
         WRITE(IU,'(A)') '|      Linear response in principle supports LRE' &
     &                 //'AL = Auto (or LREAL=.TRUE.)   |'
         WRITE(IU,'(A)') '|      However in some cases we have foun' &
     &                  //'d that the real space projection     |'
         WRITE(IU,'(A)') '|      is not sufficiently accurate.     ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      I strongly recommend that you swit' &
     &                  //'ch it off by setting LREAL=.FALSE.   |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='FOCKFORCE') THEN
         WRITE(IU,'(A)') '|      ALLOCATE for the force calculations using' &
     &                 //' Hartree-Fock routines failed.|'
         WRITE(IU,'(A)') '|      Your forces are not correct !!!   ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      However, the WAVECAR file might be' &
     &                  //' useful for continuation.            |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='HFNPAR') THEN
         WRITE(IU,'(A)') '|      HF is only implemented for NPAR=number of' &
     &                 //' nodes.                       |'
         WRITE(IU,'(A)') '|      Otherwise communication between no' &
     &                  //'des would be required during the     |'
         WRITE(IU,'(A)') '|      calculation of the exchange potent' &
     &                  //'ial, which would be utterly slow.    |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NPAR efficiency') THEN
         WRITE(IU,'(A)') '|      For optimal performance we recommend to s' &
     &                 //'et                            |'
         WRITE(IU,'(A)') '|        NCORE= 4 - approx SQRT( number of ' &
     &                  //'cores)                             |'
         WRITE(IU,'(A)') '|      NCORE specifies how many cores sto' &
     &                  //'re one orbital (NPAR=cpu/NCORE).     |'
         WRITE(IU,'(A)') '|      This setting can  greatly improve ' &
     &                  //'the performance of VASP for DFT.     |'
         WRITE(IU,'(A)') '|      The default,   NCORE=1            m' &
     &                  //'ight be grossly inefficient         |'
         WRITE(IU,'(A)') '|      on modern multi-core architectures' &
     &                  //' or massively parallel machines.     |'
         WRITE(IU,'(A)') '|      Do your own testing !!!!          ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      Unfortunately you need to use the ' &
     &                  //'default for GW and RPA calculations. |'
         WRITE(IU,'(A)') '|      (for HF NCORE is supported but not' &
     &                  //' extensively tested yet)             |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NPAR') THEN
         WRITE(IU,'(A)') '|      VASP internal routines  have requested a ' &
     &                 //'change of the k-point set.    |'
         WRITE(IU,'(A)') '|      Unfortunately this is only possibl' &
     &                  //'e if NPAR=number of nodes.           |'
         WRITE(IU,'(A)') '|      Please remove the tag NPAR from th' &
     &                  //'e INCAR file and restart the         |'
         WRITE(IU,'(A)') '|      calculations.                     ' &
     &                  //'                                     |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='LHFONE') THEN
         WRITE(IU,'(A)') '|      HF one-center treatment and a full HF tre' &
     &                 //'atment are not compatible.    |'
         WRITE(IU,'(A)') '|      Please either set LHFCALC to .FALS' &
     &                  //'E. or remove the LHFONE tag          |'
         WRITE(IU,'(A)') '|      from the INCAR file.              ' &
     &                  //'                                     |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='HFPARAM') THEN
         WRITE(IU,'(A)') '|      A combination of HF parameters was found ' &
     &                 //'in the INCAR file, that is    |'
         WRITE(IU,'(A)') '|      not supported.                    ' &
     &                  //'      e.g.                           |'
         WRITE(IU,'(A)') '|      HFRCUT and HFKIDENT               ' &
     &                  //'                                     |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OEPdouble') THEN
         WRITE(IU,'(A)') '|      The EXX-OEP and LHF methods do not suppo' &
     &                 //'rt a different grid for HF     |'
         WRITE(IU,'(A)') '|      and the local Hamiltonian.        ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      PRECFOCK was set to Normal        ' &
     &                  //'                                     |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NMAXFOCKAE') THEN
         WRITE(IU,'(A)') '|      You use NMAXFOCKAE=2 in the INCAR file.  ' &
     &                 //'Be careful:                  |'
         WRITE(IU,'(A)') '|      Adding two functions to mimic the all ele' &
     &                 //'ctron charge density on the   |'
         WRITE(IU,'(A)') '|      plane wave grid requires very fine' &
     &                  //' FFT grids. Please set NGX manually  |'
         WRITE(IU,'(A)') '|      or reconsider your choice.        ' &
     &                  //'                                     |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='ISYM2') THEN
         WRITE(IU,'(A)') '|      You have selected ISYM=2 for HF type calc' &
     &                 //'ulation. This will symmetrize |'
         WRITE(IU,'(A)') '|      the charge density but not the exc' &
     &                  //'hange potential.                     |'
         WRITE(IU,'(A)') '|      I suggest to use ISYM=3 instead. T' &
     &                  //'his uses symmetry to obtain          |'
         WRITE(IU,'(A)') '|      the orbitals at all k-points in th' &
     &                  //'e Brillouine zone, but does not      |'
         WRITE(IU,'(A)') '|      apply symmetry to the Hartree pote' &
     &                  //'ntial directly.                      |'
         WRITE(IU,'(A)') '|      The resultant charge might have lo' &
     &                  //'wer symmetry than the crystal,       |'
         WRITE(IU,'(A)') '|      but at least, Hartree and exchange' &
     &                  //' are fully compatible.               |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='LSPECTRAL inefficient') THEN
         WRITE(IU,'(A)') '|      LSPECTRAL is very inefficient if not suff' &
     &                 //'icient memory is available.   |'
         WRITE(IU,'(A)') '|      The available memory per core shou' &
     &                  //'should be set by MAXMEM = XXXX       |'
         WRITE(IU,'(A)') '|      The default for MAXMEM is presentl' &
     &                  //'y XXXX=2800 (MByte).                 |'
         WRITE(IU,'(A)') '|      If this has already been done, and' &
     &                  //' if the warning prevails,            |'
         WRITE(IU,'(A)') '|      you should either decrease NOMEGA ' &
     &                  //'or decrease ENCUTGW, or distribute   |'
         WRITE(IU,'(A)') '|      the calculations onto more nodes. ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      Note the present calculations migh' &
     &                  //'t lead to excessive memory paging!!  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='LSPECTRALGW') THEN
         WRITE(IU,'(A)') '|      The default for LSPECTRALGW is now LSPECT' &
     &                 //'RALGW=LSPECTRAL.              |'
         WRITE(IU,'(A)') '|      This will change QP energies, but ' &
     &                  //'increases the numerical precision    |'
         WRITE(IU,'(A)') '|      of the QP slightly (for instance  ' &
     &                  //'Z factors are usually more accurate) |'
         WRITE(IU,'(A)') '|      You can use the old version by spe' &
     &                  //'cifying LSPECTRALGW=.FALSE.          |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LPOTOK') THEN
         WRITE(IU,'(A)') '|      Presently the RPA-OEP routine requ' &
     &                  //'ires one to read in a POT file       |'
         WRITE(IU,'(A)') '|      with some intial guess for the loc' &
     &                  //'al potential. This could be          |'
         WRITE(IU,'(A)') '|      generated using the KLI or EXX-OEP' &
     &                  //'routines (EXXOEP=1 or 2) and         |'
         WRITE(IU,'(A)') '|      setting LVTOT=.TRUE. to write the ' &
     &                  //' POT file.                           |'
         WRITE(IU,'(A)') '|      Read this file by setting ICHARG=4' &
     &                  //'.                                    |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='Xi singular') THEN
         WRITE(IU,'(A)') '|      The response function can not be inverted' &
     &                 //'.                             |'
         WRITE(IU,'(A)') '|      This indicates that the number of ' &
     &                  //'linearly independent transitions     |'
         WRITE(IU,'(A)') '|      is smaller than the rank of the re' &
     &                  //'sponse function.                     |'
         WRITE(IU,'(A)') '|      Try to increase the number of k-po' &
     &                  //'ints, or try to decrease ENCUTGW.    |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='AEXX=0') THEN
         WRITE(IU,'(A)') '|      You calculate the electron-hole interacti' &
     &                 //'on using AEXX=0.              |'
         WRITE(IU,'(A)') '|      This is hardly what you want to do' &
     &                  //'.                                    |'
         WRITE(IU,'(A)') '|      Maybe you have forgotten to set AE' &
     &                  //'XX in the INCAR file.                |'
         WRITE(IU,'(A)') '|    ( For LHFCALC=.TRUE. the default is ' &
     &                  //' AEXX=0.25, but if LHFCALC is not    |'
         WRITE(IU,'(A)') '|      set the default is AEXX=0.0 )     ' &
     &                  //'                                     |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='GW optics') THEN
         WRITE(IU,'(A)') '|      The derivative of the wavefunctions with ' &
     &                 //'respect to k (WAVEDER)        |'
         WRITE(IU,'(A)') '|      can not be found. You should redo ' &
     &                  //'the groundstate calculations         |'
         WRITE(IU,'(A)') '|      using LOPTICS=.TRUE. in order to w' &
     &                  //'rite the WAVEDER file.               |'
         WRITE(IU,'(A)') '|      For metals the present setting is ' &
     &                  //'however ok.                          |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='GW-HF optics') THEN
         WRITE(IU,'(A)') '|      Presently LREAL is not supported for the ' &
     &                 //'calculation of optical        |'
         WRITE(IU,'(A)') '|      properties and HF type Hamiltonian' &
     &                  //'s.                                   |'
         WRITE(IU,'(A)') '|      Please use LREAL=.FALSE.          ' &
     &                  //'                                     |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='ENCUTFOCK') THEN
         WRITE(IU,'(A)') '|      ENCUTFOCK is no longer supported!        ' &
     &                 //'                              |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='ENCUTGW LHF') THEN
         WRITE(IU,'(A)') '|      ENCUTGW is set for a local HF calculation' &
     &                 //'.                             |'
         WRITE(IU,'(A)') '|      This might cause very slow convergence wi' &
     &                 //'th respect to                 |'
         WRITE(IU,'(A)') '|      ENCUTGW, since even the Hartree potential' &
     &                 //' is cut off.                  |'
         WRITE(IU,'(A)') '|      Use with utter care (or write ENCUTGW=0) ' &
     &                 //'in the INCAR file.            |'
      ENDIF
      IF (LIO.AND. (TOPIC(1:LTOPIC)=='ENCUTFOCK' .OR. TOPIC(1:LTOPIC)=='PRECFOCK')) THEN         
         WRITE(IU,'(A)') '|  Use PRECFOCK to select the mode for' &
     &                  //' HF type calculations                   |'
         WRITE(IU,'(A)') '|      PRECFOCK= L low     (coarse grid f' &
     &                  //'or HF, normal augmentation charge)   |'
         WRITE(IU,'(A)') '|      PRECFOCK= M medium  (normal grid f' &
     &                  //'or HF, normal augmentation charge)   |'
         WRITE(IU,'(A)') '|      PRECFOCK= F fast    (coarse grid f' &
     &                  //'or HF, soft augmentation charge)     |'
         WRITE(IU,'(A)') '|      PRECFOCK= N normal  (PREC=N grid f' &
     &                  //'or HF, soft augmentation charge)     |'
         WRITE(IU,'(A)') '|      PRECFOCK= A accurate (PREC=A gri' &
     &                  //'d for HF, soft augmentation charge)    |'
         WRITE(IU,'(A)') '|                                      ' &
     &                  //'                                       |'
         WRITE(IU,'(A)') '| L  is equivalent to ENCUTFOCK=0 in va' &
     &                  //'sp.5.2.2                               |'
         WRITE(IU,'(A)') '| M  is equivalent to vasp.5.2.2 if ENC' &
     &                  //'UTFOCK is not set                      |'
         WRITE(IU,'(A)') '| M&L cause significant noise in the fo' &
     &                  //'rces and are no longer recommended     |'
         WRITE(IU,'(A)') '|                                      ' &
     &                  //'                                       |'
         WRITE(IU,'(A)') '| N  is recommended for routine calcula' &
     &                  //'tions, little noise to be expected     |'
         WRITE(IU,'(A)') '| A  accurate is now recommended for ve' &
     &                  //'ry accurate calculations               |'
         WRITE(IU,'(A)') '| F  fast is now recommended for quick ' &
     &                  //'calculations (even phonons often ok)   |'
         WRITE(IU,'(A)') '|    (speedup between 2 and 4)         ' &
     &                  //'                                       |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPOINTS HF') THEN
         WRITE(IU,'(A)') '|      Your generating k-point grid is not comme' &
     &                 //'nsurate to the symmetry       |'
         WRITE(IU,'(A)') '|      of the lattice.  This can cause   ' &
     &                  //'slow convergence with respect        |'
         WRITE(IU,'(A)') '|      to k-points for HF type calculatio' &
     &                  //'ns                                   |'
         WRITE(IU,'(A)') '|      suggested SOLUTIONS:              ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|       ) if not already the case, use au' &
     &                  //'tomatic k-point generation           |'
         WRITE(IU,'(A)') '|       ) shift your grid to Gamma (G) ' &
     &                  //'(e.g. required for hex or fcc lattice) |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPOINTS PEAD') THEN
         WRITE(IU,'(A)') '|      Your generating k-point grid is not comme' &
     &                 //'nsurate to the symmetry       |'
         WRITE(IU,'(A)') '|      of the lattice.  This does not sit' &
     &                  //' well in combination with the        |'
         WRITE(IU,'(A)') '|      PEAD routines, sorry ...          ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      suggested SOLUTIONS:              ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|       ) if not already the case, use au' &
     &                  //'tomatic k-point generation           |'
         WRITE(IU,'(A)') '|       ) shift your grid to Gamma (G) ' &
     &                  //'(e.g. required for hex or fcc lattice) |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPOINTS INTER') THEN
         WRITE(IU,'(A)') '|      Your generating k-point grid is not comme' &
     &                 //'nsurate to the symmetry       |'
         WRITE(IU,'(A)') '|      of the lattice.  Presently it is n' &
     &                  //'ot possible to write the velocity    |'
         WRITE(IU,'(A)') '|      operator on in the extended zone s' &
     &                  //'cheme for such k-point grids.        |'
         WRITE(IU,'(A)') '|      suggested SOLUTIONS:              ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|       ) if not already the case, use au' &
     &                  //'tomatic k-point generation           |'
         WRITE(IU,'(A)') '|       ) shift your grid to Gamma (G) ' &
     &                  //'(e.g. required for hex or fcc lattice) |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='GW kweight') THEN
         WRITE(IU,'(A)') '|      The weights do not sum up to one for the ' &
     &                 //'summation over q in the       |'
         WRITE(IU,'(A)') '|      GW routine. To save your live they' &
     &                  //' have been renormalied to 1.         |'
         WRITE(IU,'(A)') '|      This condition can occur under cer' &
     &                  //'tain circumstances.                  |'
         WRITE(IU,'(A)') '|      Please check your results carefull' &
     &                  //'y.                                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GW noreal') THEN
         WRITE(IU,'(A)') '|      Presently the real space GW supports only' &
     &                 //' Gamma point only             |'
         WRITE(IU,'(A)') '|      calculations. Please use the stand' &
     &                  //'ard version for arbitrary k-point    |'
         WRITE(IU,'(A)') '|      grids.                            ' &
     &                  //'                                     |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='ACFDTR not supported') THEN
         WRITE(IU,'(A)') '|      Presently the real space GW code requires' &
     &                 //' scaLAPACK.                   |'
         WRITE(IU,'(A)') '|      This code is designed for massivel' &
     &                  //'y parallel computations.             |'
         WRITE(IU,'(A)') '|      Please use the standard code for r' &
     &                  //'outine calculations.                 |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='PEAD no virtuals') THEN
         WRITE(IU,'(A)') '|      Since you included no unoccupied states, ' &
     &                 //'the PEAD routines can not     |'
         WRITE(IU,'(A)') '|      estimate whether EFIELD_PEAD excee' &
     &                  //'ds the critical field strength.      |'
         WRITE(IU,'(A)') '|      Please check your results carefull' &
     &                  //'y.                                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='PEAD large field') THEN
         WRITE(IU,'(A)') '|      One or more components of EFIELD_PEAD is ' &
     &                 //'too large for comfort, in all |'
         WRITE(IU,'(A)') '|      probability you are too near to th' &
     &                  //'e onset of Zener tunneling.          |'
         WRITE(IU,'(A)') '|                                        ' &
     &                  //'                                     |'
         WRITE(IU,'(A,F8.5,A,F8.5,A)') '|           e |E dot A_1| = ',RDAT(1),'  >  1/10' &
     &                  //' E_g/N_1 = ',RDAT(4),'              |'            
         WRITE(IU,'(A,F8.5,A,F8.5,A)') '|           e |E dot A_2| = ',RDAT(2),'  >  1/10' &
     &                  //' E_g/N_2 = ',RDAT(5),'              |'            
         WRITE(IU,'(A,F8.5,A,F8.5,A)') '|           e |E dot A_3| = ',RDAT(3),'  >  1/10' &
     &                  //' E_g/N_3 = ',RDAT(6),'              |'            
         WRITE(IU,'(A)') '|                                        ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|      Possible SOLUTIONS:               ' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|       ) choose a smaller electric field' &
     &                  //'                                     |'
         WRITE(IU,'(A)') '|       ) use a less dense grid of k-poin' &
     &                  //'ts                                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='PEAD NCORE not 1') THEN
         WRITE(IU,'(A)') '|      The PEAD routines do not work for ' &
     &                  //' NCORE /= 1 (or NPAR /= NCPU).       |'
         WRITE(IU,'(A)') '|      (Several features internally rely ' &
     &                  //'on the PEAD routines so even if you  |' 
         WRITE(IU,'(A)') '|      have not specified LPEAD=.TRUE., t' &
     &                  //'ry restarting your job with NCORE=1) |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='PEADPOLNOMETAL') THEN
         WRITE(IU,'(A)') '|      The calculation of the macroscopic po' &
     &                 //'larization by means of the        |'
         WRITE(IU,'(A)') '|      Berry-phase expressions (LCALCPOL=.TR' &
     &                 //'UE.) requires your system to be   |'
         WRITE(IU,'(A)') '|      insulating. This does not seem to be ' &
     &                 //'the case.                         |'
         WRITE(IU,'(A)') '|      VASP will skip this part of your job,' &
     &                 //' sorry ...                        |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='PEADEPSNOMETAL') THEN
         WRITE(IU,'(A)') '|      The calculation of changes in the mac' &
     &                 //'roscopic polarization due to the  |'
         WRITE(IU,'(A)') '|      application of a finite electric fiel' &
     &                 //'d, by means of the PEAD method    |'
         WRITE(IU,'(A)') '|      (LCALCEPS=.TRUE. or EFIELD_PEAD/=0)  ' &
     &                 //'requires your system to be        |'
         WRITE(IU,'(A)') '|      insulating. This does not seem to be ' &
     &                 //'the case.                         |'
         WRITE(IU,'(A)') '|      VASP will skip this part of your job,' &
     &                 //' sorry ...                        |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='LINTET') THEN
         WRITE(IU,'(A)') '|      The linear tetrahedron method can not ' &
     &                 //' be used with the KPOINTS file   |'
         WRITE(IU,'(A)') '|      (generation of strings of k-points' &
     &                  //')                                    |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GRDSHFT_NOTET') THEN
         WRITE(IU,'(A)') '|      The linear tetrahedron method can not ' &
     &                 //'be used together with the        |'
         WRITE(IU,'(A)') '|      gridshift interpolation routine.      ' &
     &                  //'                                 |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GAMMAK') THEN
         WRITE(IU,'(A)') '|      You are using the Gamma-point only ver' &
     &                 //'sion with more than one k-point  |'
         WRITE(IU,'(A)') '|      or some other non Gamma k-point   ' &
     &                  //')                                    |'
 
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPOINTS') THEN
         WRITE(IU,'(A)') '|      Error reading KPOINTS file            ' &
     &                 //'                                 |'
         WRITE(IU,'(A,I5,A)') '|      the error occured at line:       ',IDAT(1), &
     &                 '                                 |'
 
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LNOINVERSION') THEN
         WRITE(IU,'(A)') '|      Full k-point grid generated           ' &
     &                 //'                                 |'
         WRITE(IU,'(A)') '|      Inversion symmetry is not applied     ' &
     &                 //'                                 |'
 
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='FFT-GRID IS NOT SUFFICIENT') THEN
         WRITE(IU,'(A)') '|      Your FFT grids (NGX,NGY,NGZ) are not ' &
     &                 //'sufficient for an accurate        |'
         WRITE(IU,'(A)') '|      calculation.                         ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      The results might be wrong            ' &
     &                  //'                                 |'
         WRITE(IU,'(A)') '|      good settings for NGX NGY and ' &
     &               //' NGZ are                                 |'
         WRITE(IU,'(A,2I4,A,I4,A)') '|                      ', &
     &             IDAT(1),IDAT(2),'  and',IDAT(3), &
     &             '                                      |'
         WRITE(IU,'(A)') '|     Mind: This setting results in a small' &
     &                 //' but reasonable wrap around error  |'
         WRITE(IU,'(A)') '|     It is also necessary to adjust these ' &
     &                 //' values to the FFT routines you use|'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='METAGGA and forces') THEN
         WRITE(IU,'(A)') '|      You have switched METAGGA and/or ASPH' &
     &                 //'ERICAL radial calculations on,    |'
         WRITE(IU,'(A)') '|      but ions are moved based on forces.  ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      METAGGA and LASPH change only the exc' &
     &                 //'hange and correlation energy.     |'
         WRITE(IU,'(A)') '|      You might reconsider your choices.   ' &
     &                 //'                                  |'
         WRITE(IU,'(A,L4,A,L4,A)') &
                         '|      Current values: LASPH=',LDAT(1), &
     &                   ', and LMETAGGA=',LDAT(2), &
     &             '                          |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='METAGGARESPONSE') THEN
         WRITE(IU,'(A)') '|      You have switched on METAGGA and line' &
     &                 //'ar response routines.             |'
         WRITE(IU,'(A)') '|      This combination is currently not sup' &
     &                 //'ported.                           |'
         WRITE(IU,'(A)') '|      If you have selected LOPTICS=.TRUE., ' &
     &                 //'VASP will continue but some       |'
         WRITE(IU,'(A)') '|      matrix elements [H,r] are neglected  ' &
     &                 //' (you might try to use            |'
         WRITE(IU,'(A)') '|      LPEAD=.TRUE. which works irrespective' &
     &                 //' of H).                           |'
         WRITE(IU,'(A)') '|      In all other cases VASP will stop now' &
     &                 //'. SOOO sorry...                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='SMALL CSHIFT') THEN
         WRITE(IU,'(A)') '|      You are using a small complex shift. ' &
     &                 //'This can yield unreliable results |'
         WRITE(IU,'(A)') '|      when the Kramers Kronig relation is u' &
     &                 //'sed for systems with spin-orbit   |'
         WRITE(IU,'(A)') '|      interactions.                        ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      Please increase CSHIFT in the INCAR f' &
     &                 //'ile.                              |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='TETNOMETAL') THEN
         WRITE(IU,'(A)') '|      Integration of the imaginary part of ' &
     &                 //'the dielectric function by means  |'
         WRITE(IU,'(A)') '|      of the tetrahedron method is not supp' &
     &                 //'orted for metallic systems.       |'
         WRITE(IU,'(A)') '|      VASP will skip the calculation of the' &
     &                 //' optical properties.              |'
         WRITE(IU,'(A)') '|      Please rerun with ISMEAR > -4.       ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='POTIM large') THEN
         WRITE(IU,'(A)') '|      Your timestep is larger than 0.1 Angs' &
     &                 //'t.                                |'
         WRITE(IU,'(A)') '|      For finite differences this really do' &
     &                 //'es not make sense. I will         |'
         WRITE(IU,'(A)') '|      reset POTIM to 0.015. I recommend to u' &
     &                 //'se 0.01 to 0.02 for finite       |'
         WRITE(IU,'(A)') '|      differences.                         ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GRDSHFT_NOT_IN_IBZ') THEN
         WRITE(IU,'(A)') '|      The gridshift interpolation procedure' &
     &                 //'is not able to find this point    |'
         WRITE(IU,'(A)') '|      in the IBZ.                          ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GRDSHFT_K') THEN
         WRITE(IU,'(A)') '|      The gridshift interpolation procedure o' &
     &                 //'nly works for odd KINTER.         |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOOCC') THEN
         WRITE(IU,'(A)') '|      The energy of the highest lying unoccu' &
     &                 //'pied state is less than the       |'
         WRITE(IU,'(A)') '|      lowest lying occupied states. Are you ' &
     &                 //'sure your system contains         |'
         WRITE(IU,'(A)') '|      unoccupied states?                    ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOODDGRID') THEN
         WRITE(IU,'(A)') '|      The gridshift interpolation procedure ' &
     &                 //'does not work for even grid       |'
         WRITE(IU,'(A)') '|      divisions.                            ' &
     &                 //'                                  |'
      ENDIF
 
      IF (LIO.AND. TOPIC(1:LTOPIC)=='GAMMAC') THEN
         WRITE(IU,'(A)') '|      The k-points need to be Gamma centered' &
     &                 //'.                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='BSE spin') THEN
         WRITE(IU,'(A)') '|      For spinpolarized BSE calculations th' &
     &                 //'e number of conduction bands      |'
         WRITE(IU,'(A)') '|      for up and down must be equal.       ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      If you get this error message, you ne' &
     &                 //'ed to increase the number of      |'
         WRITE(IU,'(A)') '|      bands in the ground state calculation' &
     &                 //'.                                 |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='BSE ANTIRES') THEN
         WRITE(IU,'(A)') '|      BSE calculations beyond the Tamm-Dank' &
     &                 //'off approximation require you     |'
         WRITE(IU,'(A)') '|      to set the flag LORBITALREAL=.TRUE.  ' &
     &                 //'in the groundstate as well as     |'
         WRITE(IU,'(A)') '|      in the BSE calculation.              ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      Please redo the groundstate calculati' &
     &                 //'on with this flag set.            |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='BSE conjg') THEN
         WRITE(IU,'(A)') '|      BSE calculations beyond the Tamm-Dank' &
     &                 //'off approximation require you     |'
         WRITE(IU,'(A)') '|      to set the flag LORBITALREAL=.TRUE.  ' &
     &                 //'in the groundstate as well as     |'
         WRITE(IU,'(A)') '|      in the BSE calculation.              ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      Please redo the groundstate calculati' &
     &                 //'on with this flag set.            |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='BSE NCV') THEN
         WRITE(IU,'(A)') '|      The number of conduction band and val' &
     &                 //'ence band pairs is zero.          |'
         WRITE(IU,'(A)') '|      This indicates that NBANDSV or NBANDS' &
     &                 //'O are not properly set up.        |'
         WRITE(IU,'(A)') '|      The VASP default is to set them to th' &
     &                 //'e number of occupied orbitals.    |'
         WRITE(IU,'(A)') '|      Try to set them manually.            ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      It is also possible that OMEGAMAX is ' &
     &                 //'too small; please increase it.    |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='W missing') THEN
         WRITE(IU,'(A,I6,A)') '|      The file WXXXX.tmp for XXXX=',IDAT,'   can not' &
     &                 //' be read.                  |'
         WRITE(IU,'(A)') '|      Rerun the GW calculations with LRPA= ' &
     &                 //' .TRUE., and do not change        |'
         WRITE(IU,'(A)') '|      ENCUTGW or NBANDS between runs.       ' &
     &                 //'                                 |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='aux WAVECAR') THEN
         WRITE(IU,'(A)') '|      An auxiliary WAVECAR file was read.  ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      W will be calculated using these auxi' &
     &                 //'liary wavefunctions.              |'
         WRITE(IU,'(A)') '|      The WAVEDER.XXX file must match the W' &
     &                 //'AVECAR file!                      |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='SPIN SPIRAL') THEN
         WRITE(IU,'(A)') '|      To represent the spin spiral you requ' &
     &                 //'ested, with a kinetic             |'
         WRITE(IU,'(A,F7.2,A,F7.2,A)') '|      energy cutoff of ENINI= ',RDAT(1),' eV,' &
     &                 //' choose ENMAX > ',RDAT(3),' eV          |'
         WRITE(IU,'(A,F7.2,A)') '|      Currently ENMAX= ',RDAT(2),' eV          ' &
     &                 //'                                  |'
      ENDIF
      
      IF (LIO.AND. TOPIC(1:LTOPIC)=='HIGHEST BANDS OCCUPIED') THEN
         WRITE(IU,'(A)') '|      Your highest band is occupied at some k-points! Unless you are         |'
         WRITE(IU,'(A)') '|      performing a calculation for an insulator or semiconductor, without    |'
         WRITE(IU,'(A)') '|      unoccupied bands, you have included TOO FEW BANDS!! Please increase    |'
         WRITE(IU,'(A)') '|      the parameter NBANDS in file INCAR to ensure that the highest band     |'
         WRITE(IU,'(A)') '|      is unoccupied at all k-points. It is always recommended to             |'
         WRITE(IU,'(A)') '|      include a few unoccupied bands to accelerate the convergence of        |'
         WRITE(IU,'(A)') '|      molecular dynamics runs (even for insulators or semiconductors).       |'
         WRITE(IU,'(A)') '|      Because the presence of unoccupied bands improves wavefunction         |'
         WRITE(IU,'(A)') "|      prediction, and helps to suppress 'band-crossings.'                    |"

         IF (NI==NR) THEN
            WRITE(IU,'(A)') '|      Following all k-points will be '// &
     &                'listed (with the Fermi weights of       |'
            WRITE(IU,'(A)') '|      the highest band given in '// &
     &           'paranthesis) ... :                           |'
            WRITE(IU,'(A)') '|                                     '// &
     &                '                                        |'
            DO 100 I=1,NI-1
               WRITE(IU,'(A,I5,A,F8.5,A)') &
     &              '|                      ',IDAT(I),'       (', &
     &              RDAT(I),')                                 |'
  100       CONTINUE
            WRITE(IU,'(A)') '|                                     '// &
     &                '                                        |'
            WRITE(IU,'(A,I5,A,F8.5,A)') &
     &              '|      The total occupancy of band no. ',IDAT(NI), &
     &                  ' is  ',RDAT(NI),' electrons ...       |'
         ENDIF
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='LVTOT') THEN
         WRITE(IU,'(A)') '|      For LVTOT=.TRUE. VASP.5.x writes the TOTAL local potential to          |'
         WRITE(IU,'(A)') '|      the file LOCPOT. If you want the Hartree contributions only, use       |'
         WRITE(IU,'(A)') '|      LVHAR=.TRUE. instead.                                                  |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='vdW-Klimes') THEN
         WRITE(IU,'(A)') '|      You have switched on vdW-DFT.        ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      This routine was written and supplied' &
     &                 //' by Jiri Klimes.                  |'
         WRITE(IU,'(A)') '|      We recommed that you carefully read a' &
     &                 //'nd cite the following             |'
         WRITE(IU,'(A)') '|      publication                          ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      J. Klimes, D.R. Bowler, A. Michelides' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|        J. Phys.: Cond Matt. 22 022201 (201' &
     &                 //'0)                                |'
         WRITE(IU,'(A)') '|      J. Klimes, D.R. Bowler, A. Michelides' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|        Phys. Rev. B. 83, 195131 (2011)    ' &
     &                 //'                                  |'
         WRITE(IU,'(A)') '|      and references therein.              ' &
     &                 //'                                  |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOKPAR') THEN
         WRITE(IU,'(A)') '|      You have enabled k-point parallelism ' &
                       //'(KPAR>1) but this does not work |'
         WRITE(IU,'(A)') '|      for the interpolation routines.      ' &
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPAR') THEN
         WRITE(IU,'(A)') '|      You have enabled k-point parallelism ' &
                       //'(KPAR>1).                         |'
         WRITE(IU,'(A)') '|      This developmental code was originally ' & 
                       //' written by Paul Kent at ORNL,  |'
         WRITE(IU,'(A)') '|      and carefully double checked in Vienna.' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      GW as well as linear response paralleli' & 
                       //'sm added by Martijn Marsman     |'
         WRITE(IU,'(A)') '|      and Georg Kresse.                      ' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      Carefully verify results versus KPAR=1.' &
                       //'                                |'
         WRITE(IU,'(A)') '|      Report problems to Paul Kent and Vienna' &
                       //'.                               |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OMEGAPAR') THEN
         WRITE(IU,'(A)') '|      You cannot create more omega groups tha' &
                       //'n CPUs available.               |'
         WRITE(IU,'(A)') '|      Number of omega groups set to number of' & 
                       //'CPUs.                           |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OMEGAPARNOMEGA') THEN
         WRITE(IU,'(A)') '|      You cannot create more omega groups tha' &
                       //'n omega points considered.      |'
         WRITE(IU,'(A)') '|      Number of omega groups set to NOMEGA.  ' & 
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='TAUPAR') THEN
         WRITE(IU,'(A)') '|      You cannot create more tau groups than ' &
                       //'CPUs available.                 |'
         WRITE(IU,'(A)') '|      Number of tau groups set to number of C' & 
                       //'PUs.                            |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='TAUPARNTAU') THEN
         WRITE(IU,'(A)') '|      You cannot create more tau groups than ' &
                       //'tau points considered.          |'
         WRITE(IU,'(A)') '|      Number of tau groups set to NOMEGA.    ' & 
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='TAUPARCPU') THEN
         WRITE(IU,'(A)') '|      TAUPAR and OMEGAPAR is not a 2D process' &
                       //'or grid. Please make            |'
         WRITE(IU,'(A)') '|      sure that TAUPAR * OMEGAPAR = NOMEGA ho' &
                       //'lds and that the                |'
         WRITE(IU,'(A)') '|      number of CPUs is dividable by TAUPAR a' &
                       //'nd OMEGAPAR.                    |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='chi_GG NOMEGA') THEN
         WRITE(IU,'(A)') '|      Presently the real time/ real space c' &
                       //'ode requires a minimum of         |'
         WRITE(IU,'(A)') '|      3 imaginary times.                     ' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      Please set NOMEGA to a sensible number.' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      Typically   8 imaginary times suffice. ' & 
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OMEGAGRID OMEGAMIN') THEN
         WRITE(IU,'(A)') '|      You have a very small band gap. This ' &
                       //'results in slow convergence       |'
         WRITE(IU,'(A)') '|      of the energy with respect to NOMEGA   ' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      Usually VASP defaults OMEGAMIN to the b' & 
                       //'andgap which guarantees an      |'
         WRITE(IU,'(A)') '|      optimal grid for OMEGAGRID=140.        ' & 
                       //'                                |'
         WRITE(IU,'(A,F7.4,A)') &
                         '|      The gap was found to be OMEGAMIN=',RDAT, & 
                         '                               |'
         WRITE(IU,'(A)') '|      Now OMEGAMIN is set to 0.02 instead, wh' & 
                       //'ich effectively removes some    |'
         WRITE(IU,'(A)') '|      excitations from the spectrum.         ' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      You can set OMEGAMIN manually in the IN' & 
                       //'CAR, however.                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='WANN_NOT_FOUND') THEN
         WRITE(IU,'(A)') '|      The file wannier90.win can not be found' & 
                       //'.                               |'
      ENDIF
     
      IF (LIO.AND. TOPIC(1:LTOPIC)=='TARGET_NOT_SET') THEN
         WRITE(IU,'(A)') '|      You have not defined target states for ' &
                       //'the projection onto the         |'
         WRITE(IU,'(A)') '|      subspace. VASP will continue and use al' & 
                       //'l states of the localized       |'
         WRITE(IU,'(A)') '|      basis as targets.                      ' &
                       //'                                |'
         WRITE(IU,'(A)') '|      You can set target states in the INCAR ' &
                       //'file with NTARGET_STATES        |' 
      ENDIF
    
      IF (LIO.AND. TOPIC(1:LTOPIC)=='NTARGET_STATES') THEN
         WRITE(IU,'(A)') '|      NTARGET_STATES contains more non-zero v' &
                       //'alues than target states        |'
         WRITE(IU,'(A)') '|      available. VASP will continue and negle' & 
                       //'ct screening effects of         |'
         WRITE(IU,'(A)') '|      ALL Wannier states! If this is not desi' &
                       //'red NTARGET_STATES has to       |'
         WRITE(IU,'(A)') '|      be set properly in the INCAR.          ' &
                       //'                                |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='NCRPALOW') THEN
         WRITE(IU,'(A)') '|      NCRPALOW is not set in INCAR.          ' &
                       //'                                |'
         WRITE(IU,'(A)') '|      NCRPALOW should correspond to the lowes' &
                       //'t band represented by           |'
         WRITE(IU,'(A)') '|      the localized basis.                   ' &
                       //'                                |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='DISENTANGLE') THEN
         WRITE(IU,'(A)') '|      CRPA disentanglement method needs a min' &
                       //'imal basis with same size       |'
         WRITE(IU,'(A,I2,A)') '|      as the target space. Your chosen target' &
                       //'s span a ',IDAT(2),' dimensional         |'
         WRITE(IU,'(A,I2,A)') '|      sub space in a space with ',IDAT(1),' dimensions.'&
                       //'                               |'
         WRITE(IU,'(A)') '|      Change either NTARGET_STATES or the loc' &
                       //'alized basis set.               |'
         WRITE(IU,'(A)') '|      Alternatively, use the standard CRPA or' &
                       //' the weighted method.           |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='LCAO_INCAR') THEN
         WRITE(IU,'(A)') '|      You have not specified an ion number WA' &
                       //'NPROJ_I and/or an angular       |'
         WRITE(IU,'(A)') '|      momentum WANPROJ_L in the INCAR file. T' & 
                       //'his is mandatory if you         |'
         WRITE(IU,'(A)') '|      want to use LCAOs!!                    ' & 
                       //'                                |'
      ENDIF
      IF (LIO.AND. TOPIC(1:LTOPIC)=='SHMEMCRPA') THEN
         WRITE(IU,'(A)') '|      NCSHMEM>1 is not supported for CRPA cal' &
                       //'culations.                      |'
         WRITE(IU,'(A)') '|      Please set NCSHMEM=1 for all CRPA calcu' &
                       //'lations.                        |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OMEGAGRID OMEGATL') THEN
         WRITE(IU,'(A)') '|      The maximum frequency in the frequenc' &
                       //'y grid is smaller than the        |'
         WRITE(IU,'(A)') '|      largest excitation energy. This can cau' & 
                       //'se errors in the                |'
         WRITE(IU,'(A)') '|      correlation energy. Please increase NOM' & 
                       //'EGA to make sure that the       |'
         WRITE(IU,'(A)') '|      error is sufficiently small.           ' & 
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NOMEGA 32') THEN
         WRITE(IU,'(A)') '|      OMEGAGRIRD = 140 does not allow for more ' &
     &                 //'than 32 grid points.          |'
         WRITE(IU,'(A)') '|      For this method the error decays e' &
     &                  //'xpontentially with the number of     |'
         WRITE(IU,'(A)') '|      grid points. Typically NOMEGA=8-16' &
     &                  //' is sufficient even for small gap    |'
         WRITE(IU,'(A)') '|      systems. Please make your own test' &
     &                  //'s.                                   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='OMEGAGRID140') THEN
         WRITE(IU,'(A)') '|      Presently the real time/ real space c' &
                       //'ode requires either               |'
         WRITE(IU,'(A)') '|      OMEGAGRID=140 (minimax grids) or' & 
                       //' OMEGAGRID=150 (least square grids).   |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NUPDOWN') THEN
         WRITE(IU,'(A)') '|      NUPDOWN is presently not supported beca' &
                       //'use it can result in a          |'
         WRITE(IU,'(A)') '|      different EFERMI for up and down electr' &
                       //'ons. Switching it off now       |'
         WRITE(IU,'(A)') '|      and continuing. Please check that occup' &
                       //'ancies are still ok by  e.g.    |'
         WRITE(IU,'(A)') '|      performing a one step KS calculation.  ' &
                       //'                                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='NPAR IALGO=8') THEN
         WRITE(IU,'(A)') '|      IALGO = 8 supports only parallelism o' &
                       //'ver k-points (KPAR) and           |'
         WRITE(IU,'(A)') '|      distribution of orbitals over cores.   ' & 
                       //'                                |'
         WRITE(IU,'(A)') '|      Specifically NPAR must be set to 1 an' &
                       //'d NCORE should not be set in      |'
         WRITE(IU,'(A)') '|      the INCAR file.                    ' & 
                       //'                                    |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='SHMEM ERROR') THEN
         WRITE(IU,'(A)') '|      Can not allocate shared memory.      ' &
                       //'                                  |'
         WRITE(IU,'(A)') '|      The simple solution is to run using NCS' & 
                       //'HMEM=1.                         |'
         WRITE(IU,'(A)') '|      However this can increase the memory ' &
                       //'requirements significantly.       |'
         WRITE(IU,'(A)') '|      If possible increase the availabe  ' & 
                       //'shared memory.                      |'
         WRITE(IU,'(A)') '|      The available memory can be set in ' & 
                       //'the kernel.                         |'
         WRITE(IU,'(A)') '|      /sbin/sysctl kernel{max,all,mni} ca' & 
                       //'n be used to determine the          |'
         WRITE(IU,'(A)') '|      available memory. max often needs t' & 
                       //'o be increased (half the physical   |'
         WRITE(IU,'(A)') '|      memory is often a good choice).    ' & 
                       //'                                    |'
      ENDIF


      IF (LIO.AND. TOPIC(1:LTOPIC)=='KPROJ') THEN
         WRITE(IU,'(A)') '|      The projection LKPROJ is presently only i' &
     &                 //'mplemented for NCORE=1.       |'
         WRITE(IU,'(A)') '|      Please set NCORE=1 (or NPAR to the' &
     &                  //' total number of cores)              |'
         WRITE(IU,'(A)') '|      and restart the calculations.     ' &
     &                  //'                                     |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GAMMA') THEN
         WRITE(IU,'(A)') '|      The file GAMMA is not compatible with the' &
     &                 //' present calculation.         |'
         WRITE(IU,'(A)') '|      The first line must specify the nu' &
     &                  //'mber of ions, bands and k-points.    |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='GAMMA2') THEN
         WRITE(IU,'(A)') '|      The file GAMMA can not be read. Each set ' &
     &                 //'must start with the k-point   |'
         WRITE(IU,'(A)') '|      number the start and final band fo' &
     &                  //'r which the correction of the        |'
         WRITE(IU,'(A,I6,A)') '|      density matrix is supplied.       ' &
     &                  //' NK=',IDAT,'                           |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='ALTERNANT') THEN
         WRITE(IU,'(A)') '|      Could not find all extrema of error funct' &
     &                 //'ion. Try to increase the      |'
         WRITE(IU,'(A)') '|      number of sampling points. Number of poin' &
     &                  //'ts used in this run:          |'
         WRITE(IU,'(A,I6,A)') '|      NMAXALT=',IDAT,'                    ' &
     &                 //'                                     |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='LBONEANDLZORA') THEN
         WRITE(IU,'(A)') '|      You have set both LBONE=.TRUE. as well as' &
     &                  //' LZORA=.TRUE. The former will |'
         WRITE(IU,'(A)') '|      take precedence, i.e., LZORA will now be ' &
     &                  //'set to .FALSE.                |'
      ENDIF

      IF (LIO.AND. TOPIC(1:LTOPIC)=='RANDOMSEED') THEN
         WRITE(IU,'(A)') '|      The Random Number Generator may be seeded' &
     &                  //' as follows:                  |'
         WRITE(IU,'(A)') '|        0 <= SEED1 < 900000000                 ' &
     &                  //'                              |'
         WRITE(IU,'(A)') '|        0 <= SEED2 < 1000000                   ' &
     &                  //'                              |'
         WRITE(IU,'(A)') '|        0 <= SEED3                             ' &
     &                  //'                              |'
         WRITE(IU,'(A,3I12,A)') &
     &                   '|      Your current RANDOM_SEED:',IDAT(1:3),'          |'
      ENDIF

! Here we have told the user all we can tell, bring it to an end ...:
99999 CONTINUE
! Hmmmm ..., very very fatal error --->  S T O P  !!
      IF (TYPE=='S') THEN
! Well announce the brute end and exit ...
         IF (LIO) WRITE(IU,5)
         STOP
      ENDIF
! Normal end of message ...
         IF (LIO) WRITE(IU,4)
! Bye ...
      RETURN
      END
