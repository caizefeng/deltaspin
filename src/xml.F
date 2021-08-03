#include "symbol.inc"
#define usgrid
!****************** PROGRAM VASP  Version 4.4 (f90)********************
! RCS:  
!
! this module implements the xml output of VASP
! the DDT can be found in vasp.xml
!
!**********************************************************************

      MODULE VASPXML
        USE prec
        IMPLICIT NONE
        INTEGER,SAVE ::  uixml  ! unit to which xml output is written
                                      ! if uixml is smaller 0 no output is written
        LOGICAL,SAVE ::  lxml   ! perform xml output

! the following table implements a stack of open tags
        INTEGER, PARAMETER,PRIVATE :: maxstringlen=20, maxdepth=20
        CHARACTER (LEN=maxstringlen),PRIVATE :: stack(maxdepth)
        INTEGER,SAVE,PRIVATE             :: stackposition=0
! automatic indention to make the file more easily readable
        LOGICAL,SAVE                     :: do_indent=.TRUE.
        INTEGER,SAVE                     :: inden
        CHARACTER (LEN=20)               :: blank="                    "

!=======================================================================
!
!  subroutine XML_TAG (TAG)
!    writes a single TAG to the xml
!    and pops the tag to a stack
!
!  subroutine XML_CLOSE_TAG
!    pop a TAG from the stack and write it to the xml file
!
!=======================================================================
      CONTAINS
        SUBROUTINE XML_TAG( tag, name, type, param, comment )
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag
          CHARACTER (LEN=*),OPTIONAL :: name
          CHARACTER (LEN=*),OPTIONAL :: type
          CHARACTER (LEN=*),OPTIONAL :: param
          CHARACTER (LEN=*),OPTIONAL :: comment

        ! check whether maxstringlen is sufficiently large
          IF (LEN(tag) > maxstringlen) THEN
             WRITE(0,*) 'internal WARNING in xml.F: increase maxstringlen to ',LEN(tag)
          ENDIF
          stackposition=stackposition+1
          IF (stackposition > maxdepth) THEN
             WRITE(0,*) 'internal ERROR in xml.F: increase maxdepth'
             STOP
          ENDIF

          stack(stackposition)=tag

          IF (lxml) THEN
          WRITE(uixml, '(A,"<",A)',ADVANCE="No") blank(1:inden),tag

          IF (PRESENT(name)) THEN
             WRITE(uixml, '(" name=",1H",A,1H"," ")',ADVANCE="No") name
          ENDIF
          IF (PRESENT(type)) THEN
             IF (type/="float") THEN
                WRITE(uixml, '(" type=",1H",A,1H"," ")',ADVANCE="No") type
             ENDIF
          ENDIF
          IF (PRESENT(param)) THEN
             WRITE(uixml, '(" param=",1H",A,1H")',ADVANCE="No")  param
          ENDIF
          IF (PRESENT(comment)) THEN
             WRITE(uixml, '(" comment=",1H",A,1H")',ADVANCE="No")  comment
          ENDIF
          WRITE(uixml, '(">")',ADVANCE="yes")
     
          ENDIF

          IF (do_indent) inden=stackposition

        END SUBROUTINE XML_TAG


        SUBROUTINE XML_CLOSE_TAG( tag)
          IMPLICIT NONE
          CHARACTER (LEN=*), OPTIONAL :: tag
          INTEGER :: l
          IF (stackposition < 1) THEN
             WRITE(0,*) 'internal ERROR in xml.F: xml stack exhausted, no more tags to close'
             IF (PRESENT(tag)) WRITE(0,*) 'closing tag is ',tag
             STOP
          ENDIF
          IF (do_indent) inden=(stackposition-1)

          l=LEN_TRIM(stack(stackposition))
          IF (lxml) WRITE(uixml, '(A,"</",A,">")')  blank(1:inden),stack(stackposition)(1:l)
          IF (PRESENT(tag)) THEN
             IF (tag /= stack(stackposition)(1:l)) THEN
                WRITE(0,*) 'XML_CLOSE_TAG internal error: not a matching tag',tag,' ',stack(stackposition)(1:l)
                STOP
             ENDIF
          ENDIF
          stackposition=stackposition-1
          
        END SUBROUTINE XML_CLOSE_TAG

!=======================================================================
!
! XML_FLUSH flush the file
!
!=======================================================================

        SUBROUTINE XML_FLUSH
          IF (lxml) CALL WFORCE(uixml)
        END SUBROUTINE XML_FLUSH

!=======================================================================
!
!  subroutine XML_TAG_STRING (TAG, STRING)
!  subroutine XML_TAG_REAL (TAG, STRING)
!  subroutine XML_TAG_INT (TAG, STRING)
!    writes an single entity to the the xml file, bracketing it between 
!    <TAG> entity <\TAG>
!
!=======================================================================

        SUBROUTINE XML_TAG_STRING( tag, string)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag, string

          IF (lxml) WRITE(uixml, '(A,"<i name=",1H",A,1H"," type=",1H","string",1H",">",A," </i>")')  blank(1:inden),tag,string

        END SUBROUTINE XML_TAG_STRING

        SUBROUTINE XML_TAG_REAL( tag, r)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag
          REAL(q)       :: r

          IF (lxml) WRITE(uixml, '(A,"<i name=",1H",A,1H",">",F16.8," </i>")')  blank(1:inden),tag,r

        END SUBROUTINE XML_TAG_REAL

        SUBROUTINE XML_TAG_INT( tag, i)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag
          INTEGER       :: i

          IF (lxml) WRITE(uixml, '(A,"<i name=",1H",A,1H"," type=",1H","int",1H",">",I8," </i>")')  blank(1:inden),tag,i

        END SUBROUTINE XML_TAG_INT


        SUBROUTINE XML_TAG_STRING_( tag, string)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag, string

          IF (lxml) WRITE(uixml, '(A,"<",A,">",A," </",A,">")')  blank(1:inden),tag,string,tag

        END SUBROUTINE XML_TAG_STRING_

        SUBROUTINE XML_TAG_REAL_( tag, r)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag
          REAL(q)       :: r

          IF (lxml) WRITE(uixml, '(A,"<",A,">",F16.8," </",A,">")')  blank(1:inden),tag,r,tag

        END SUBROUTINE XML_TAG_REAL_

        SUBROUTINE XML_TAG_INT_( tag, i)
          IMPLICIT NONE
          CHARACTER (LEN=*) :: tag
          INTEGER       :: i

          IF (lxml) WRITE(uixml, '(A,"<",A,">",I8," </",A,">")')  blank(1:inden),tag,i,tag

        END SUBROUTINE XML_TAG_INT_

!=======================================================================
!
!  subroutine XML_VEC_REAL
!  subroutine XML_VEC_INT
!  subroutine XML_VEC_LOG
!   write a vector with n entries
!   possibly add a name to the vector
!
!  <v>  entity1 entity2 entity3  </v>
!  <v name=supplied_name >  entity1 entity2 entity3  </v>
!
!=======================================================================

        SUBROUTINE XML_VEC_REAL( v, name, form)
          IMPLICIT NONE
          REAL(q)       :: v(:)
          CHARACTER (LEN=*),OPTIONAL :: name
          CHARACTER (LEN=*),OPTIONAL :: form
          INTEGER i
          CHARACTER (LEN=20) :: f

          IF (.NOT. lxml) RETURN

          IF (PRESENT(form)) THEN
             f=form
          ELSE
             f="(F16.8,' ')"
          ENDIF

          IF (PRESENT(name)) THEN
             WRITE(uixml, '(A,"<v name=",1H",A,1H",">")',ADVANCE="No")  blank(1:inden),name
          ELSE
             WRITE(uixml, '(A,"<v> ")',ADVANCE="No") blank(1:inden)
          ENDIF
          DO i=1,size(v)
             WRITE(uixml, f, ADVANCE="No") v(i)
          ENDDO
          WRITE(uixml, '("</v>")')

        END SUBROUTINE XML_VEC_REAL

        SUBROUTINE XML_VEC_INT( v, name, form)
          IMPLICIT NONE
          INTEGER       :: v(:)
          CHARACTER (LEN=*),OPTIONAL :: name
          CHARACTER (LEN=*),OPTIONAL  :: form
          CHARACTER (LEN=20) :: f
          INTEGER i

          IF (.NOT. lxml) RETURN

          IF (PRESENT(form)) THEN
             f=form
          ELSE
             f="(I8,' ')"
          ENDIF

          IF (PRESENT(name)) THEN
             WRITE(uixml, '(A,"<v type=",1H","int",1H"," name=",1H",A,1H",">")',ADVANCE="No")  blank(1:inden),name
          ELSE
             WRITE(uixml, '(A,"<v type=",1H","int",1H"," > ")',ADVANCE="No") blank(1:inden)
          ENDIF
          DO i=1,size(v)
             WRITE(uixml,f,ADVANCE="No") v(i)
          ENDDO
          WRITE(uixml, '("</v>")')

        END SUBROUTINE XML_VEC_INT

        SUBROUTINE XML_VEC_LOG( v, name)
          IMPLICIT NONE
          CHARACTER (LEN=*),OPTIONAL :: name
          LOGICAL       :: v(:)
          INTEGER i

          IF (.NOT. lxml) RETURN
          IF (PRESENT(name)) THEN
             WRITE(uixml, '(A,"<v type=",1H","logical",1H"," name=",1H",A,1H",">")',ADVANCE="No")  blank(1:inden),name
          ELSE
             WRITE(uixml, '(A,"<v type=",1H","logical",1H"," > ")',ADVANCE="No") blank(1:inden)
          ENDIF
          DO i=1,size(v)
             WRITE(uixml,"(L2,' ')",ADVANCE="No") v(i)
          ENDDO
          WRITE(uixml, '("</v>")')

        END SUBROUTINE XML_VEC_LOG

!=======================================================================
!
!  subroutine XML_TIMING
!   write a vector of timing entries 
!
!=======================================================================

        SUBROUTINE XML_TIMING( time1, time2, name )
          IMPLICIT NONE
          REAL(q)       :: time1, time2
          CHARACTER (LEN=*),OPTIONAL :: name

          IF (.NOT. lxml) RETURN

          IF (PRESENT(name)) THEN
             WRITE(uixml, '(A,"<time name=",1H",A,1H",">")',ADVANCE="No")  blank(1:inden),name
          ELSE
             WRITE(uixml, '(A,"<time> ")',ADVANCE="No") blank(1:inden)
          ENDIF
          WRITE(uixml,'(2F8.2)',ADVANCE="No") time1,time2
          WRITE(uixml, '("</time>")')

        END SUBROUTINE XML_TIMING

!=======================================================================
!
!  subroutine XML_ARRAY_REAL (TAG)
!  subroutine XML_ARRAY_INT  (TAG)
!  subroutine XML_ARRAY_LOG  (TAG)
!  write an array containing a single vectors in each row
!
!  <v>  entity1 entity2 entity3  </v>
!  <v>  entity1 entity2 entity3  </v>
!   ....
!=======================================================================


      SUBROUTINE XML_ARRAY_REAL(X, form1)
        IMPLICIT NONE
        REAL(q) :: X(:,:)
        INTEGER :: i
        CHARACTER (LEN=*),OPTIONAL  :: form1

        IF (PRESENT(form1)) THEN
           DO i=1,size(X,2)
              CALL XML_VEC_REAL(X(:,i),form=form1)
           ENDDO
        ELSE
           DO i=1,size(X,2)
              CALL XML_VEC_REAL(X(:,i))
           ENDDO
        ENDIF
           
      END SUBROUTINE XML_ARRAY_REAL

      SUBROUTINE XML_ARRAY_INT(X,form1)
        IMPLICIT NONE
        INTEGER :: X(:,:)
        INTEGER :: i
        CHARACTER (LEN=*),OPTIONAL  :: form1

        IF (PRESENT(form1)) THEN
           DO i=1,size(X,2)
              CALL XML_VEC_INT(X(:,i))
           ENDDO
        ELSE
           DO i=1,size(X,2)
              CALL XML_VEC_INT(X(:,i),form=form1)
           ENDDO
        ENDIF
      END SUBROUTINE XML_ARRAY_INT

      SUBROUTINE XML_ARRAY_LOG(X)
        IMPLICIT NONE
        LOGICAL :: X(:,:)
        INTEGER :: i

        DO i=1,size(X,2)
           CALL XML_VEC_LOG(X(:,i))
        ENDDO

      END SUBROUTINE XML_ARRAY_LOG


!=======================================================================
!
!  subroutine XML_ROW_DATA
!   write a vector with n entries in the data field of an array
!
!  <r>  entity1 entity2 entity3  </r>
!
!=======================================================================

        SUBROUTINE XML_ROW_DATA( v, form)
          IMPLICIT NONE
          REAL(q)       :: v(:)
          INTEGER i
          CHARACTER (LEN=*),OPTIONAL  :: form
          CHARACTER (LEN=20) :: f

          IF (.NOT. lxml) RETURN

          IF (PRESENT(form)) THEN
             f=form
          ELSE
             f="(F16.8)"
          ENDIF

          WRITE(uixml, '(A,"<r> ")',ADVANCE="No") blank(1:inden)
          DO i=1,size(v)
             WRITE(uixml, f, ADVANCE="No") v(i)
          ENDDO
          WRITE(uixml, '("</r>")')

        END SUBROUTINE XML_ROW_DATA

!=======================================================================
!
!  subroutine XML_VECARRAY (TAG)
!   write a field entry of the fields structure
!   possibly write also a type parameter, unit parameter or the scan
!   parameter
!
!=======================================================================

        SUBROUTINE XML_VECARRAY( name_, type_ )
          IMPLICIT NONE
          CHARACTER (LEN=*)          :: name_
          CHARACTER (LEN=*),OPTIONAL :: type_
          INTEGER i

          IF (PRESENT(type_)) THEN
             CALL XML_TAG("varray",name=name_, type=type_)
          ELSE
             CALL XML_TAG("varray",name=name_)
          ENDIF


        END SUBROUTINE XML_VECARRAY

!=======================================================================
!
!  subroutine XML_FIELD (TAG)
!   write a field entry of the fields structure
!   possibly write also a type parameter, unit parameter or the scan
!   parameter
!
!=======================================================================

        SUBROUTINE XML_FIELD( name, type, unit)
          IMPLICIT NONE
          CHARACTER (LEN=*)          :: name
          CHARACTER (LEN=*),OPTIONAL :: type
          CHARACTER (LEN=*),OPTIONAL :: unit
          INTEGER i

          IF (.NOT. lxml) RETURN
          
          WRITE(uixml, '(A,"<field")',ADVANCE="No") blank(1:inden)

          IF (PRESENT(type)) THEN
             IF (type/="float") THEN
                WRITE(uixml, '(" type=",1H",A,1H")',ADVANCE="No")  type
             ENDIF
          ENDIF
          IF (PRESENT(unit)) THEN
             WRITE(uixml, '(" unit=",1H",A,1H")',ADVANCE="No")  unit
          ENDIF
          WRITE(uixml, '(">")',ADVANCE="No")

          WRITE(uixml,"(A)",ADVANCE="No") name
          
          WRITE(uixml, '("</field>")')

        END SUBROUTINE XML_FIELD

!=======================================================================
!
!  subroutine XML_FIELD (TAG)
!   write a field entry of the fields structure
!   possibly write also a type parameter, unit parameter or the scan
!   parameter
!
!=======================================================================

        SUBROUTINE XML_DIMENSION(name, dim)
          IMPLICIT NONE
          CHARACTER (LEN=*)          :: name
          INTEGER, OPTIONAL          :: dim
          INTEGER i
          CHARACTER (LEN=40) :: strcounter

          IF (.NOT. lxml) RETURN
          
          WRITE(uixml, '(A,"<dimension")',ADVANCE="No") blank(1:inden)
          WRITE(strcounter,'(I3)') dim

          WRITE(uixml, '(" dim=",1H",A,1H")',ADVANCE="No") TRIM(ADJUSTL(strcounter))
          WRITE(uixml, '(">")',ADVANCE="No")

          WRITE(uixml,"(A)",ADVANCE="No") name
          
          WRITE(uixml, '("</dimension>")')

        END SUBROUTINE XML_DIMENSION

#define WT(x)  CALL XML_TAG(x)
#define WCT CALL XML_CLOSE_TAG

!=======================================================================
!
!  START_XML
!     subroutine opens the xml file and writes the initial header
!  STOP_XML
!     subroutine opens closes all open tags and closes the xml file
!  
!=======================================================================
      
      SUBROUTINE START_XML( uixml_SET, XMLFILE )
        IMPLICIT NONE
        INTEGER uixml_SET
        CHARACTER (LEN=*) XMLFILE

        uixml=uixml_SET
        IF ( uixml>= 0) THEN
           lxml=.TRUE.
        ELSE
           lxml=.FALSE.          
        END IF
        stackposition=0
        inden        =0 

        IF (lxml) THEN
           OPEN(uixml, FILE=XMLFILE)
           WRITE(uixml,'(A)') '<?xml version="1.0" encoding="ISO-8859-1"?>'
!          WRITE(uixml,'(A)') '<?xml-stylesheet type="text/xsl" href="vasp.xsl"?>'

        ENDIF
        
        WT("modeling")
      END SUBROUTINE START_XML

      SUBROUTINE STOP_XML
        IMPLICIT NONE
        INTEGER i

        WCT
        IF (stackposition /= 0) THEN
           WRITE(0,*) 'xml error: not all tags are closed in the xml file'
        ENDIF
        DO i=1,stackposition
           WCT
        ENDDO
        IF (lxml) CLOSE(uixml)
      END SUBROUTINE STOP_XML

!=======================================================================
! 
! XML_GENERATOR writes the generator tag
!  
!=======================================================================

      SUBROUTINE XML_GENERATOR
        IMPLICIT none
        CALL XML_TAG("generator")

      END SUBROUTINE XML_GENERATOR

      SUBROUTINE PARSE_GENERATOR_XML( INF )
        IMPLICIT none
        CHARACTER (LEN=*) INF
        CHARACTER (LEN=100) ST
        INTEGER i,il
        
        i =1
        il=INDEX(INF(i:),".")-1
        CALL XML_TAG_STRING("program", INF(i:il))
        i =il+2
        il=INDEX(INF(i:)," ")-1
        CALL XML_TAG_STRING("version", INF(i:i+il))
        CALL XML_TAG_STRING("subversion", INF(i+il+1:))
        
      END SUBROUTINE PARSE_GENERATOR_XML
        
!=======================================================================
! 
! XML_KPOINTS_LIST
!   write the generated k-points list
!
!=======================================================================

      SUBROUTINE XML_KPOINTS_LIST(VKPT, WTKPT, IDTET, VOLWGT)
        USE prec
        IMPLICIT NONE

        INTEGER NKPTS     ! number of k-points
        INTEGER NINTER    ! number of interpolation points
        REAL(q) :: VKPT(:, :)
        REAL(q) :: WTKPT(:)
        INTEGER,OPTIONAL :: IDTET(:,:)
        REAL(q),OPTIONAL :: VOLWGT
      ! local
        INTEGER i
        REAL(q) :: tmp(1)

        CALL XML_VECARRAY("kpointlist")
        CALL XML_ARRAY_REAL(VKPT)
        CALL XML_CLOSE_TAG("varray")

        CALL XML_VECARRAY("weights")
        DO i=1,size(WTKPT)
           tmp(1)=WTKPT(i)
           CALL XML_VEC_REAL(tmp)
        ENDDO
        CALL XML_CLOSE_TAG("varray")

        IF (PRESENT(IDTET).AND. PRESENT(VOLWGT)) THEN
           CALL XML_VECARRAY("tetrahedronlist","int")
           CALL XML_ARRAY_INT(IDTET, "(I7)")
           CALL XML_CLOSE_TAG("varray")
           CALL XML_TAG_REAL("volumeweight", VOLWGT)
        ENDIF

        
      END SUBROUTINE XML_KPOINTS_LIST

!=======================================================================
! 
! eigenvalues and weights 
!      
!=======================================================================

      SUBROUTINE XML_EIGENVALUES(CELTOT, FERTOT, NB_TOT, NKPTS, ISPIN)
        INTEGER NB_TOT                   ! number of bands
        INTEGER NKPTS                    ! number of k-points
        INTEGER ISPIN                    ! number of spins
        COMPLEX(q) :: CELTOT(:, :, :)
        REAL(q) :: FERTOT(:, :, :)
        
        CALL XML_TAG("eigenvalues")
        CALL XML_EIGENVAL_NOHEAD(CELTOT, FERTOT, NB_TOT, NKPTS, ISPIN)
        CALL XML_CLOSE_TAG

      END SUBROUTINE XML_EIGENVALUES

!=======================================================================
! 
! eigenvalues and weights 
! this version does not write the XML_TAG "eigenvalues"
!      
!=======================================================================

      SUBROUTINE XML_EIGENVAL_NOHEAD(CELTOT, FERTOT, NB_TOT, NKPTS, ISPIN)
        INTEGER NB_TOT                   ! number of bands
        INTEGER NKPTS                    ! number of k-points
        INTEGER ISPIN                    ! number of spins
        COMPLEX(q) :: CELTOT(:, :, :)
        REAL(q) :: FERTOT(:, :, :)
        CHARACTER (LEN=40) :: strcounter
        
        INTEGER n,nk,i
        REAL(q) :: TMP(2)

        CALL XML_TAG("array")
        CALL XML_DIMENSION("band",1)
        CALL XML_DIMENSION("kpoint",2)
        CALL XML_DIMENSION("spin",3)
        CALL XML_FIELD("eigene",type="float")
        CALL XML_FIELD("occ",type="float")
        CALL XML_TAG("set")

        DO i=1,ISPIN
           WRITE(strcounter,"(I2)") i
           CALL XML_TAG("set", comment="spin "//TRIM(ADJUSTL(strcounter)))
           DO nk=1,NKPTS
              WRITE(strcounter,"(I6)") nk
              CALL XML_TAG("set", comment="kpoint "//TRIM(ADJUSTL(strcounter)))
              DO n=1,NB_TOT
                 TMP(1)=CELTOT(n,nk,i)
                 TMP(2)=FERTOT(n,nk,i)
                 CALL XML_ROW_DATA(TMP, FORM='(F9.4,1X)')
              ENDDO
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")


      END SUBROUTINE XML_EIGENVAL_NOHEAD

!=======================================================================
! 
! eigenvalues and derivative
!      
!=======================================================================

      SUBROUTINE XML_EIGENVALUES_EXT( EIGENVAL, NDATA, NB_TOT, NKPTS, ISPIN)
        INTEGER NDATA                    ! number of data
        INTEGER NB_TOT                   ! number of bands
        INTEGER NKPTS                    ! number of k-points
        INTEGER ISPIN                    ! number of spins
        REAL(q) :: EIGENVAL( NDATA, NB_TOT, NKPTS, ISPIN)
        CHARACTER (LEN=40) :: strcounter
        
        INTEGER n,nk,i
        REAL(q) :: TMP(NDATA)

        CALL XML_TAG("eigenvalues")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("band",1)
        CALL XML_DIMENSION("kpoint",2)
        CALL XML_DIMENSION("spin",3)
        CALL XML_FIELD("eigene",type="float")
        DO I=1, NDATA-1
           CALL XML_FIELD("data",type="float")
        ENDDO
        CALL XML_TAG("set")

        DO i=1,ISPIN
           WRITE(strcounter,"(I2)") i
           CALL XML_TAG("set", comment="spin "//TRIM(ADJUSTL(strcounter)))
           DO nk=1,NKPTS
              WRITE(strcounter,"(I6)") nk
              CALL XML_TAG("set", comment="kpoint "//TRIM(ADJUSTL(strcounter)))
              DO n=1,NB_TOT
                 TMP(:)=EIGENVAL(:,n,nk,i)
                 CALL XML_ROW_DATA(TMP, FORM='(F9.4,1X)')
              ENDDO
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG


     END SUBROUTINE XML_EIGENVALUES_EXT


!=======================================================================
! 
! XML_DOS
! write density of states
!      
!=======================================================================

      SUBROUTINE XML_DOS(EFERMI, EMIN, EMAX, LPARTIAL, DOS, DOSI, DOSPAR, NEDOS, LDIM, NIONS, ISPIN , COMMENT)
        REAL(q) EFERMI
        REAL(q) EMIN, EMAX               ! minimal and maximal energy
        INTEGER NEDOS                    ! number of grid points
        INTEGER LDIM                     ! number of l-quantum numbers
        INTEGER NIONS                    ! number of ions
        INTEGER ISPIN                    ! number of spin degrees of freedom 
        LOGICAL LPARTIAL                 ! partial dos 
        LOGICAL LMDECOMPOSED             ! LM decomposed or not
        REAL(q) :: DOS(NEDOS, ISPIN), DOSI(NEDOS, ISPIN)
        REAL(q) :: DOSPAR(NEDOS, LDIM, NIONS, ISPIN)
        CHARACTER (LEN=*),OPTIONAL :: comment
      ! local
        INTEGER n,isp,l,ni,lmax
        REAL(q) :: E(NEDOS)
        REAL(q) :: DELTAE
        REAL(q), ALLOCATABLE :: TMP(:)
        CHARACTER (LEN=5) :: lmtype
        CHARACTER (LEN=40) :: strcounter

        IF (LPARTIAL) THEN
           ALLOCATE(TMP(MAX(3,1+LDIM)))
        ELSE
           ALLOCATE(TMP(3))
        ENDIF

        IF (PRESENT(COMMENT)) THEN
           CALL XML_TAG("dos", comment=comment)
        ELSE
           CALL XML_TAG("dos")
        ENDIF
        CALL XML_TAG_REAL("efermi",EFERMI)

        DELTAE=(EMAX-EMIN)/(NEDOS-1)

        DO n=1,NEDOS
           E(n)=EMIN+DELTAE*(n-1)
        ENDDO

        CALL XML_TAG("total")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_DIMENSION("spin",2)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("total",type="float")
        CALL XML_FIELD("integrated",type="float")
        CALL XML_TAG("set")
        DO isp=1,ISPIN
           WRITE(strcounter,"(I2)") isp
           CALL XML_TAG("set",comment="spin "//TRIM(ADJUSTL(strcounter)))

           DELTAE=(EMAX-EMIN)/(NEDOS-1)
           
           DO n=1,nedos
              TMP(1)= E(n)
              TMP(2)= DOS(n,isp)
              TMP(3)= DOSI(n,isp)
              ! CALL XML_ROW_DATA( TMP(1:3),form='(G12.4)')
              CALL XML_ROW_DATA( TMP(1:3),form='(F10.4,1X)')
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO
        
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("total")

        IF (LPARTIAL) THEN
           CALL XML_TAG("partial")
           CALL XML_TAG("array")
           CALL XML_DIMENSION("gridpoints",1)
           CALL XML_DIMENSION("spin",2)
           CALL XML_DIMENSION("ion",3)
           CALL XML_FIELD("energy",type="float")
           DO l=1,LDIM
              ni=0
              CALL SPHPRO_DESCRIPTION(ni, l, lmtype)
              IF (lmtype=="     ") THEN
                 EXIT
              ENDIF
              CALL SPHPRO_DESCRIPTION(ni, l, lmtype)
              IF (lmtype==" err ") THEN
                 WRITE(0,*) 'internal error: SPHPRO_DESCRIPTION returns an error for NI=',ni,' L',L
                 STOP
                 EXIT
              ENDIF
                 
              CALL XML_FIELD(lmtype, type="float")
           ENDDO
           lmax=l-1
           CALL XML_TAG("set")

           DO ni=1,NIONS
              WRITE(strcounter,"(I6)") ni
              CALL XML_TAG("set", comment="ion "//TRIM(ADJUSTL(strcounter)))
              DO isp=1,ISPIN
                 WRITE(strcounter,"(I2)") isp
                 CALL XML_TAG("set", comment="spin "//TRIM(ADJUSTL(strcounter)))

                 DO n=1,nedos
                    TMP(1)= E(n)
                    DO l=1,lmax
                       TMP(l+1)= DOSPAR(n, l, ni, isp)
                    ENDDO
                    ! CALL XML_ROW_DATA( TMP(1:lmax+1), form='(G12.4)')
                    CALL XML_ROW_DATA( TMP(1:lmax+1), form='(F10.4,1X)')
                 ENDDO
                 CALL XML_CLOSE_TAG("set")
              ENDDO
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
           CALL XML_CLOSE_TAG("array")
           CALL XML_CLOSE_TAG("partial")
        ENDIF

        CALL XML_CLOSE_TAG("dos")
        DEALLOCATE(TMP)

      END SUBROUTINE XML_DOS


     END MODULE VASPXML

!=======================================================================
! 
! XML_INCAR
!    is used write tags which have been read from the INCAR file
!
! the command prings N items
! since F90 does not support casts, the values to be dumped
! are supplied in different arrays INTRES ... STR
!  five types (given in the variable FORMAT) are supported
!                                                                      
!     -- 'S'    write some string       --> result goes to variable STR 
!     -- 'I'    write integer data list --> result goes to array INTRES 
!     -- 'F'    write real    data list --> result goes to array FLTRES 
!     -- 'C'    write complex data list --> result goes to array CMPRES
!     -- 'L'    write logical data list --> result goes to array LOGRES
!    
!=======================================================================


     SUBROUTINE XML_INCAR(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
        USE vaspxml
        USE prec
        IMPLICIT none
        CHARACTER (LEN=*) :: TAG
        CHARACTER (LEN=1) :: FORM
        
        INTEGER    :: INTRES(*)
        REAL(q)    :: FLTRES(*)
        COMPLEX(q) :: CMPRES(*)
        LOGICAL    :: LOGRES(*)
        CHARACTER (LEN=*) :: STR
        INTEGER N
! local variables
        INTEGER i

        IF (N==0) RETURN
        IF (N/=1 .AND. FORM /= "S") THEN
           WRITE(0,*) 'internal error: XML_INCAR called with a vector, use XML_INCAR_V instead ',TAG
           STOP
        ENDIF
        IF (.NOT. lxml) RETURN

        SELECT CASE (FORM)
           CASE ("S")
              i=LEN_TRIM(str)
              WRITE(uixml, '(A,"<i type=",1H","string",1H"," name=",1H",A,1H",">",A,"</i>")',ADVANCE="Yes")  blank(1:inden),tag,str(1:i)
           CASE("I")
              WRITE(uixml, '(A,"<i type=",1H","int",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(I6)',ADVANCE="No") INTRES(i)
              ENDDO
              WRITE(uixml, '("</i>")',ADVANCE="Yes") 
           CASE("F")
              WRITE(uixml, '(A,"<i name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(F16.8)',ADVANCE="No") FLTRES(i)
              ENDDO
              WRITE(uixml, '("</i>")',ADVANCE="Yes") 
           CASE("C")
              WRITE(uixml, '(A,"<i type=",1H","complex",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(2F16.8,2X)',ADVANCE="No") CMPRES(i)
              ENDDO
              WRITE(uixml, '("</i>")',ADVANCE="Yes") 
           CASE("L")
              WRITE(uixml, '(A,"<i type=",1H","logical",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(L2,2X)',ADVANCE="No") LOGRES(i)
              ENDDO
              WRITE(uixml, '("</i>")',ADVANCE="Yes") 

       END SELECT
#ifdef logflow
       CALL JOBFLOW(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
#endif

     END SUBROUTINE XML_INCAR

     SUBROUTINE XML_INCAR_V(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
        USE vaspxml
        USE prec
        IMPLICIT none
        CHARACTER (LEN=*) :: TAG
        CHARACTER (LEN=1) :: FORM
        
        INTEGER    :: INTRES(*)
        REAL(q)    :: FLTRES(*)
        COMPLEX(q) :: CMPRES(*)
        LOGICAL    :: LOGRES(*)
        CHARACTER (LEN=*) :: STR
        INTEGER N
! local variables
        INTEGER i
        INTEGER, PARAMETER :: break=20

        IF (N==0) RETURN
        IF (.NOT. lxml) RETURN

        SELECT CASE (FORM)
           CASE ("S")
              i=LEN_TRIM(str)
              WRITE(uixml, '(A,"<v type=",1H","string",1H"," name=",1H",A,1H",">",A,"</v>")',ADVANCE="Yes")  blank(1:inden),tag,str(1:i)
           CASE("I")
              WRITE(uixml, '(A,"<v type=",1H","int",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(I6)',ADVANCE="No") INTRES(i)
                 IF (MOD(i,break)==0) WRITE(uixml,*)
              ENDDO

              WRITE(uixml, '("</v>")',ADVANCE="Yes") 
           CASE("F")
              WRITE(uixml, '(A,"<v name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(F16.8)',ADVANCE="No") FLTRES(i)
                 IF (MOD(i,break)==0) WRITE(uixml,*)
              ENDDO
              WRITE(uixml, '("</v>")',ADVANCE="Yes") 
           CASE("C")
              WRITE(uixml, '(A,"<v type=",1H","complex",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(2F16.8,2X)',ADVANCE="No") CMPRES(i)
                 IF (MOD(i,break)==0) WRITE(uixml,*)
              ENDDO
              WRITE(uixml, '("</v>")',ADVANCE="Yes") 
           CASE("L")
              WRITE(uixml, '(A,"<v type=",1H","logical",1H"," name=",1H",A,1H",">")',ADVANCE="No")   blank(1:inden),tag
              DO i=1,N
                 WRITE(uixml,'(L2,2X)',ADVANCE="No") LOGRES(i)
                 IF (MOD(i,break)==0) WRITE(uixml,*)
              ENDDO
              WRITE(uixml, '("</v>")',ADVANCE="Yes") 

       END SELECT
#ifdef logflow
       CALL JOBFLOW_V(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
#endif

     END SUBROUTINE XML_INCAR_V
        
!=======================================================================
! 
! JOBFLOW writes the read INCAR tags to UNIT=19 if
! the corresponding unit is open
!    
!=======================================================================

     SUBROUTINE JOBFLOW(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
        USE vaspxml
        USE prec
        IMPLICIT none
        CHARACTER (LEN=*) :: TAG
        CHARACTER (LEN=1) :: FORM
        
        INTEGER    :: INTRES(*)
        REAL(q)    :: FLTRES(*)
        COMPLEX(q) :: CMPRES(*)
        LOGICAL    :: LOGRES(*)
        CHARACTER (LEN=*) :: STR
        INTEGER N
! local variables
        INTEGER i
        CHARACTER (LEN=9) :: act

        INQUIRE (19 , ACTION=act)
        IF (act=="UNDEFINED") RETURN

        IF (N==0) RETURN

        IF (N/=1 .AND. FORM /= "S") THEN
           WRITE(0,*) 'internal error: JOBFLOW called with a vector, use XML_INCAR_V instead ',TAG
           STOP
        ENDIF
        IF (.NOT. lxml) RETURN

        SELECT CASE (FORM)
           CASE ("S")
              i=LEN_TRIM(str)
              WRITE(19, '(A,"=",A)',ADVANCE="Yes")  tag,str(1:i)
           CASE("I")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(I6)',ADVANCE="No") INTRES(i)
              ENDDO
              WRITE(19,*) 
           CASE("F")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(F16.8)',ADVANCE="No") FLTRES(i)
              ENDDO
              WRITE(19,*) 
           CASE("C")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(2F16.8,2X)',ADVANCE="No") CMPRES(i)
              ENDDO
              WRITE(19,*) 
           CASE("L")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(L2,2X)',ADVANCE="No") LOGRES(i)
              ENDDO
              WRITE(19,*) 

       END SELECT

     END SUBROUTINE JOBFLOW

     SUBROUTINE JOBFLOW_V(TAG,FORM,INTRES,FLTRES,CMPRES,LOGRES,STR,N)
        USE vaspxml
        USE prec
        IMPLICIT none
        CHARACTER (LEN=*) :: TAG
        CHARACTER (LEN=1) :: FORM
        
        INTEGER    :: INTRES(*)
        REAL(q)    :: FLTRES(*)
        COMPLEX(q) :: CMPRES(*)
        LOGICAL    :: LOGRES(*)
        CHARACTER (LEN=*) :: STR
        INTEGER N
! local variables
        INTEGER i
        INTEGER, PARAMETER :: break=20
        CHARACTER :: BACKSL='\\'
        CHARACTER (LEN=9) :: act

        IF (N==0) RETURN

        INQUIRE (19 , ACTION=act)
        IF (act=="UNDEFINED") RETURN

        SELECT CASE (FORM)
           CASE ("S")
              i=LEN_TRIM(str)
              WRITE(19, '(A,"=",A)',ADVANCE="Yes")  tag,str(1:i)
           CASE("I")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(I6)',ADVANCE="No") INTRES(i)
                 IF (MOD(i,break)==0) WRITE(19,*) BACKSL
              ENDDO

              WRITE(19,*) 
           CASE("F")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(F16.8)',ADVANCE="No") FLTRES(i)
                 IF (MOD(i,break)==0) WRITE(19,*) BACKSL
              ENDDO
              WRITE(19,*) 
           CASE("C")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(2F16.8,2X)',ADVANCE="No") CMPRES(i)
                 IF (MOD(i,break)==0) WRITE(19,*) BACKSL
              ENDDO
              WRITE(19,*) 
           CASE("L")
              WRITE(19, '(A,"=")',ADVANCE="No") tag
              DO i=1,N
                 WRITE(19,'(L2,2X)',ADVANCE="No") LOGRES(i)
                 IF (MOD(i,break)==0) WRITE(19,*) BACKSL
              ENDDO
              WRITE(19,*) 

       END SELECT

     END SUBROUTINE JOBFLOW_V
        
!=======================================================================
! 
! XML_CRYSTAL
!  subroutine to write out the crystal structure
!  i.e. basis vectors and reciprocal basis vectors
!      
!=======================================================================

      SUBROUTINE XML_CRYSTAL(A, B, VOLUME)
        USE vaspxml
        USE prec
        IMPLICIT none
        REAL(q) A(3,3)
        REAL(q) B(3,3)
        REAL(q) VOLUME

        CALL XML_TAG( "crystal")

        CALL XML_VECARRAY( "basis")
          CALL XML_VEC_REAL( A(:,1))
          CALL XML_VEC_REAL( A(:,2))
          CALL XML_VEC_REAL( A(:,3))
        CALL XML_CLOSE_TAG

        CALL XML_TAG_REAL( "volume", volume)
        
        CALL XML_VECARRAY( "rec_basis")
          CALL XML_VEC_REAL( B(:,1))
          CALL XML_VEC_REAL( B(:,2))
          CALL XML_VEC_REAL( B(:,3))
        CALL XML_CLOSE_TAG

        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_CRYSTAL

!=======================================================================
! 
! XML_ATOMTYPES
!  subroutine to write information about the atom type, mass
!  and applied pseudpotential
!      
!=======================================================================

      SUBROUTINE XML_ATOMTYPES(NIONS, NTYP, NITYP, ITYP, ELEMENT, POMASS, ZVAL, STRING )
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER :: NIONS                 ! number of ions
        INTEGER :: NTYP                  ! number of types
        INTEGER :: NITYP(NTYP)           ! number of atoms per type
        INTEGER :: ITYP(NIONS)           ! type of each atom
        CHARACTER (LEN=2) :: ELEMENT(NTYP)   ! name of element
        REAL(q) :: POMASS(NTYP)          ! mass of element
        REAL(q) :: ZVAL(NTYP)            ! valency of element
        CHARACTER (LEN=40):: STRING(NTYP)    ! pseudopotential generation information
   ! local
        CHARACTER (LEN=2) :: tagr="rc"      
        CHARACTER (LEN=1) :: tagc="c"  
        INTEGER ni,nt

        CALL XML_TAG("atominfo")
        CALL XML_TAG_INT_("atoms", NIONS)
        CALL XML_TAG_INT_("types", NTYP)

        CALL XML_TAG("array", name="atoms")
        CALL XML_DIMENSION("ion",1)
        CALL XML_FIELD("element",type="string")
        CALL XML_FIELD("atomtype",type="int")
        CALL XML_TAG("set")
        
        IF (lxml) THEN
           DO ni=1,nions
              nt=ityp(ni)
              WRITE(uixml, '(A,"<",A,">")',ADVANCE="No")  blank(1:inden),tagr

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(A)',ADVANCE="No") element(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(I4)',ADVANCE="No") ityp(ni)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("</",A,">")')  tagr
           ENDDO
        ENDIF

        CALL XML_CLOSE_TAG
        CALL XML_CLOSE_TAG

        CALL XML_TAG("array", name="atomtypes")
        CALL XML_DIMENSION("type",1)
        CALL XML_FIELD("atomspertype",type="int")
        CALL XML_FIELD("element",type="string")
        CALL XML_FIELD("mass",type="float")
        CALL XML_FIELD("valence",type="float")
        CALL XML_FIELD("pseudopotential",type="string")
        CALL XML_TAG("set")

        IF (lxml) THEN
           DO nt=1,ntyp
              WRITE(uixml, '(A,"<",A,">")',ADVANCE="No")  blank(1:inden),tagr

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(I4)',ADVANCE="No") nityp(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(A)',ADVANCE="No") element(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(F16.8)',ADVANCE="No") pomass(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(F16.8)',ADVANCE="No") zval(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

              WRITE(uixml, '("<",A,">")',ADVANCE="No") tagc
              WRITE(uixml, '(A)',ADVANCE="No") string(nt)
              WRITE(uixml, '("</",A,">")',ADVANCE="No")  tagc

             WRITE(uixml, '("</",A,">")')  tagr
           ENDDO
        ENDIF

        CALL XML_CLOSE_TAG
        CALL XML_CLOSE_TAG

        CALL XML_CLOSE_TAG("atominfo")
      END SUBROUTINE XML_ATOMTYPES


!=======================================================================
! 
! position related items
! XML_POSITIONS
! XML_VEL
! XML_FORCES
! XML_LSDYN
!  subroutine to write the positions and velocities
!      
!=======================================================================

      SUBROUTINE XML_POSITIONS(NIONS, X)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER :: NIONS                 ! number of ions
        REAL(q) :: X(3,NIONS)            ! positions
   ! local
        CALL XML_VECARRAY("positions")
        CALL XML_ARRAY_REAL(X)
        CALL XML_CLOSE_TAG

      END SUBROUTINE XML_POSITIONS


      SUBROUTINE XML_VEL(NIONS, X)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER :: NIONS                 ! number of ions
        REAL(q) :: X(3,NIONS)            ! positions
   ! local
        CALL XML_VECARRAY("velocities")
        CALL XML_ARRAY_REAL(X)
        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_VEL

      SUBROUTINE XML_FORCES(NIONS, X)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER :: NIONS                 ! number of ions
        REAL(q) :: X(3,NIONS)            ! positions
   ! local
        CALL XML_VECARRAY("forces")
        CALL XML_ARRAY_REAL(X)
        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_FORCES

      SUBROUTINE XML_STRESS(X)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        REAL(q) :: X(3,3)                ! stress tensor

        CALL XML_VECARRAY("stress")
        CALL XML_ARRAY_REAL(X)
        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_STRESS


      SUBROUTINE XML_TENSOR(string,X)
        USE vaspxml
        USE prec
        IMPLICIT NONE
        CHARACTER (LEN=*) :: string

        REAL(q) :: X(3,3)                ! stress tensor

        CALL XML_VECARRAY(string)
        CALL XML_ARRAY_REAL(X)
        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_TENSOR


      SUBROUTINE XML_LSDYN(NIONS, L)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER :: NIONS                 ! number of ions
        LOGICAL :: L(3,NIONS)            ! positions
   ! local
        CALL XML_VECARRAY("selective", "logical")
        CALL XML_ARRAY_LOG(L)
        CALL XML_CLOSE_TAG

      END SUBROUTINE XML_LSDYN

      SUBROUTINE XML_NOSE( NOSE)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        REAL(q) :: NOSE(4)
   ! local
        CALL XML_TAG("nose")
        CALL XML_VEC_REAL(NOSE)
        CALL XML_CLOSE_TAG

      END SUBROUTINE XML_NOSE

!=======================================================================
! 
! energy tag that occur several times in the output
!      
!=======================================================================

      SUBROUTINE XML_ENERGY(E1, E2, E3)
        USE vaspxml
        USE prec
        IMPLICIT none
        REAL(q) E1, E2, E3
        
        CALL XML_TAG_REAL("e_fr_energy",E1)
        CALL XML_TAG_REAL("e_wo_entrp", E2)
        CALL XML_TAG_REAL("e_0_energy", E3)

      END SUBROUTINE XML_ENERGY

!=======================================================================
! 
! XML_EPSILON_W
! write the frequency dependent dielectric function 
!      
!=======================================================================

      SUBROUTINE XML_EPSILON_W(DELTAE, EPS_REAL, EPS_IMAG, NEDOS )
        USE vaspxml
        USE prec
        IMPLICIT none

        REAL(q) DELTAE
        INTEGER NEDOS                    ! number of grid points
        REAL(q) :: EPS_REAL(NEDOS, 3,3), EPS_IMAG(NEDOS, 3, 3)
      ! local
        INTEGER :: N
        REAL(q) :: E(NEDOS)
        REAL(q) :: tmp(7)
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("dielectricfunction")

        DO n=1,NEDOS
           E(n)=0+DELTAE*(n-1)
        ENDDO

        CALL XML_TAG("imag")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("xx", type="float")
        CALL XML_FIELD("yy", type="float")
        CALL XML_FIELD("zz", type="float")
        CALL XML_FIELD("xy", type="float")
        CALL XML_FIELD("yz", type="float")
        CALL XML_FIELD("zx", type="float")
        CALL XML_TAG("set")
        
        ! cap values to avoid ****** output
        DO n=1,nedos
           tmp(1)= E(n)
           tmp(2)= MIN(EPS_IMAG(n,1,1),99999._q)
           tmp(3)= MIN(EPS_IMAG(n,2,2),99999._q)
           tmp(4)= MIN(EPS_IMAG(n,3,3),99999._q)
           tmp(5)= MIN(EPS_IMAG(n,1,2),99999._q)
           tmp(6)= MIN(EPS_IMAG(n,2,3),99999._q)
           tmp(7)= MIN(EPS_IMAG(n,3,1),99999._q)
           CALL XML_ROW_DATA( tmp, form='(F10.4,1X)')
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("imag")


        CALL XML_TAG("real")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("xx", type="float")
        CALL XML_FIELD("yy", type="float")
        CALL XML_FIELD("zz", type="float")
        CALL XML_FIELD("xy", type="float")
        CALL XML_FIELD("yz", type="float")
        CALL XML_FIELD("zx", type="float")
        CALL XML_TAG("set")
        
        ! cap values to avoid ****** output
        DO n=1,nedos
           tmp(1)= E(n)
           tmp(2)= MIN(EPS_REAL(n,1,1),99999._q)
           tmp(3)= MIN(EPS_REAL(n,2,2),99999._q)
           tmp(4)= MIN(EPS_REAL(n,3,3),99999._q)
           tmp(5)= MIN(EPS_REAL(n,1,2),99999._q)
           tmp(6)= MIN(EPS_REAL(n,2,3),99999._q)
           tmp(7)= MIN(EPS_REAL(n,3,1),99999._q)
           CALL XML_ROW_DATA( tmp, form='(F10.4,1X)')
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("real")

        CALL XML_CLOSE_TAG("dielectricfunction")

      END SUBROUTINE XML_EPSILON_W



!=======================================================================
! 
! XML_EPSILON_W
! write the frequency dependent dielectric function at an 
! arbitrary frequency grid
!      
!=======================================================================

      SUBROUTINE XML_EPSILON_E(E, EPS_REAL, EPS_IMAG, NEDOS, COMMENT )
        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER NEDOS                    ! number of grid points
        REAL(q) E(NEDOS)
        REAL(q) :: EPS_REAL( 3, 3, NEDOS), EPS_IMAG( 3, 3, NEDOS)
        CHARACTER (LEN=*) :: COMMENT
      ! local
        INTEGER :: N
        REAL(q) :: tmp(7)
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("dielectricfunction",comment=COMMENT)

        CALL XML_TAG("imag")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("xx", type="float")
        CALL XML_FIELD("yy", type="float")
        CALL XML_FIELD("zz", type="float")
        CALL XML_FIELD("xy", type="float")
        CALL XML_FIELD("yz", type="float")
        CALL XML_FIELD("zx", type="float")
        CALL XML_TAG("set")
        
        DO n=1,nedos
           tmp(1)= E(n)
           tmp(2)= EPS_IMAG(1,1,n)
           tmp(3)= EPS_IMAG(2,2,n)
           tmp(4)= EPS_IMAG(3,3,n)
           tmp(5)= EPS_IMAG(1,2,n)
           tmp(6)= EPS_IMAG(2,3,n)
           tmp(7)= EPS_IMAG(3,1,n)
           CALL XML_ROW_DATA( tmp, form='(F10.4,1X)')
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("imag")


        CALL XML_TAG("real")
        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("xx", type="float")
        CALL XML_FIELD("yy", type="float")
        CALL XML_FIELD("zz", type="float")
        CALL XML_FIELD("xy", type="float")
        CALL XML_FIELD("yz", type="float")
        CALL XML_FIELD("zx", type="float")
        CALL XML_TAG("set")
        
        DO n=1,nedos
           tmp(1)= E(n)
           tmp(2)= EPS_REAL(1,1,n)
           tmp(3)= EPS_REAL(2,2,n)
           tmp(4)= EPS_REAL(3,3,n)
           tmp(5)= EPS_REAL(1,2,n)
           tmp(6)= EPS_REAL(2,3,n)
           tmp(7)= EPS_REAL(3,1,n)
           CALL XML_ROW_DATA( tmp, form='(F10.4,1X)')
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("real")

        CALL XML_CLOSE_TAG("dielectricfunction")

      END SUBROUTINE XML_EPSILON_E


!=======================================================================
! 
! XML_EPSILON_COND
! write the frequency dependent conductivity tensor
!      
!=======================================================================

      SUBROUTINE XML_EPSILON_COND(E, COND,  NEDOS, COMMENT )
        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER NEDOS                    ! number of grid points
        REAL(q) E(NEDOS)
        REAL(q) :: COND( NEDOS, 3 , 3)
        CHARACTER (LEN=*) :: COMMENT
      ! local
        INTEGER :: N
        REAL(q) :: tmp(7)
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("conductivity",comment=COMMENT)

        CALL XML_TAG("array")
        CALL XML_DIMENSION("gridpoints",1)
        CALL XML_FIELD("energy",type="float")
        CALL XML_FIELD("xx", type="float")
        CALL XML_FIELD("yy", type="float")
        CALL XML_FIELD("zz", type="float")
        CALL XML_FIELD("xy", type="float")
        CALL XML_FIELD("yz", type="float")
        CALL XML_FIELD("zx", type="float")
        CALL XML_TAG("set")
        
        DO n=1,nedos
           tmp(1)= E(n)
           tmp(2)= COND(n,1,1)
           tmp(3)= COND(n,2,2)
           tmp(4)= COND(n,3,3)
           tmp(5)= COND(n,1,2)
           tmp(6)= COND(n,2,3)
           tmp(7)= COND(n,3,1)
           CALL XML_ROW_DATA( tmp, form='(F10.4,1X)')
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")

        CALL XML_CLOSE_TAG("conductivity")

      END SUBROUTINE XML_EPSILON_COND


!=======================================================================
! 
! XML_PROCAR
!  presently there is some redundancy since the  
!  eigenvalues und occupancies are also written in the eigenvalue
!  list 
!
!=======================================================================

      SUBROUTINE XML_PROCAR(PAR, CELTOT, FERTOT, NB_TOT, NKPTS, LDIM , NIONS, ISPIN)

        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER NB_TOT                   ! number of bands
        INTEGER NKPTS                    ! number of k-points
        INTEGER LDIM                     ! number of l quantum numbers
        INTEGER NIONS                    ! number of ions
        INTEGER ISPIN                    ! number of spins
        COMPLEX(q) :: CELTOT(NB_TOT, NKPTS, ISPIN)
        REAL(q) :: FERTOT(NB_TOT, NKPTS, ISPIN)
        REAL(q) :: PAR(NB_TOT, NKPTS ,LDIM, NIONS, ISPIN)
     ! local
        INTEGER :: i, nk, n, ni, l, lmax
        REAL(q) :: TMP(LDIM+1)
        CHARACTER (LEN=5) :: lmtype
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("projected")
        CALL XML_EIGENVALUES(CELTOT, FERTOT, NB_TOT, NKPTS, MODULO(ISPIN,3))

        CALL XML_TAG("array")
        CALL XML_DIMENSION("ion",1)
        CALL XML_DIMENSION("band",2)
        CALL XML_DIMENSION("kpoint",3)
        CALL XML_DIMENSION("spin",4)

        DO l=1,LDIM
           ni=0
           CALL SPHPRO_DESCRIPTION(ni, l, lmtype)
           IF (lmtype=="     ") THEN
              EXIT
           ENDIF
           CALL SPHPRO_DESCRIPTION(ni, l, lmtype)
           IF (lmtype==" err ") THEN
              WRITE(0,*) 'internal error: SPHPRO_DESCRIPTION returns an error for NI=',ni,' L',L
              STOP
              EXIT
           ENDIF
           
           CALL XML_FIELD(lmtype, type="float")
        ENDDO
        lmax=l-1
        CALL XML_TAG("set")
        
        DO i=1,ISPIN
           WRITE(strcounter,"(I2)") i
           CALL XML_TAG("set", comment="spin"//TRIM(ADJUSTL(strcounter)))
           DO nk=1,NKPTS
              WRITE(strcounter,"(I6)") nk
              CALL XML_TAG("set", comment="kpoint "//TRIM(ADJUSTL(strcounter)))
              DO n=1,NB_TOT
                 WRITE(strcounter,"(I6)") n
                 CALL XML_TAG("set", comment="band "//TRIM(ADJUSTL(strcounter)))

                 DO ni=1,NIONS
                    DO l=1,lmax
                       TMP(l)= PAR(n, nk , l, ni, i)
                    ENDDO
                    CALL XML_ROW_DATA( TMP(1:lmax), form='(F7.4,1X)')
                 ENDDO

                 CALL XML_CLOSE_TAG("set")
              ENDDO
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("projected")
      END SUBROUTINE XML_PROCAR


!=======================================================================
! 
! XML_KPROJ
!   write for each stat
!
!=======================================================================

      SUBROUTINE XML_KPROJ(KAPPA, CELTOT, FERTOT, NB_TOT, NKPTS, ISPIN, NKPTS_IRZ)

        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER NB_TOT                   ! number of bands
        INTEGER NKPTS                    ! number of k-points
        INTEGER ISPIN                    ! number of spins
        INTEGER NKPTS_IRZ                ! number of k-points in IRZ of POSCAR.prim
        COMPLEX(q) :: CELTOT(NB_TOT, NKPTS, ISPIN)
        REAL(q) :: FERTOT(NB_TOT, NKPTS, ISPIN)
        REAL(q) :: KAPPA(NB_TOT, NKPTS , ISPIN, NKPTS_IRZ)
     ! local
        INTEGER :: i, nk, n, nk_irz
        REAL(q) :: TMP(NKPTS_IRZ)
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("kprojected")
        CALL XML_EIGENVALUES(CELTOT, FERTOT, NB_TOT, NKPTS, MODULO(ISPIN,3)) 

        CALL XML_TAG("array")
        CALL XML_DIMENSION("band",1)
        CALL XML_DIMENSION("kpoint",2)
        CALL XML_DIMENSION("spin",3)

        CALL XML_TAG("set")
        
        DO i=1,ISPIN
           WRITE(strcounter,"(I2)") i
           CALL XML_TAG("set", comment="spin"//TRIM(ADJUSTL(strcounter)))
           DO nk=1,NKPTS
              WRITE(strcounter,"(I6)") nk
              CALL XML_TAG("set", comment="kpoint "//TRIM(ADJUSTL(strcounter)))
              DO n=1,NB_TOT
                 WRITE(strcounter,"(I6)") n
                 CALL XML_TAG("set", comment="band "//TRIM(ADJUSTL(strcounter)))
                 DO nk_irz=1,NKPTS_IRZ
                    TMP(nk_irz)= KAPPA(n, nk, i, nk_irz)
                 ENDDO
                 CALL XML_ROW_DATA( TMP, form='(F7.4,1X)')
                 CALL XML_CLOSE_TAG("set")
              ENDDO
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO
        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")
        CALL XML_CLOSE_TAG("kprojected")
      END SUBROUTINE XML_KPROJ

!=======================================================================
! 
! XML_KPOINTS_1
!   write k-points, that have been read from the KPOINTS file
!   that is the simplest version, since it simply calls the
!   XML_KPOINTS_LIST routine
!
!=======================================================================

      SUBROUTINE XML_KPOINTS_1(NKPTS, VKPT, WTKPT, NTET, IDTET, VOLWGT)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER NKPTS             ! number of k-points
        REAL(q) :: VKPT(3, NKPTS) ! k-point coordinates
        REAL(q) :: WTKPT(NKPTS)   ! weigth for each k-point
        INTEGER :: NTET           ! number of tetrahedrons
        INTEGER :: IDTET(0:4,NTET)! list of tetrahedrons
        REAL(q) :: VOLWGT         ! volume weight for each tetrahedron

        CALL XML_TAG("kpoints")

      ! kpoint and tetrahedron linklist
        IF (NTET==0) THEN
           CALL XML_KPOINTS_LIST(VKPT, WTKPT)
        ELSE
           CALL XML_KPOINTS_LIST(VKPT, WTKPT, IDTET, VOLWGT)
        ENDIF
        CALL XML_CLOSE_TAG("kpoints")

        
      END SUBROUTINE XML_KPOINTS_1

!=======================================================================
! 
! XML_KPOINTS_2
!   write k-points generated automatically
!   this subroutine has to do more, since
!   all the parameters, that lead to this particular mesh are also
!   written to the xml file
!   this required some logic
!   CSEL describes the generation mode:
!   'a' 'A'   fully automatic (RKLEN is set)
!   'm' 'M'   Monkhorst-Pack
!   'g' 'G'   Gamma centered
!   other     gambler mode
!
!=======================================================================

      SUBROUTINE XML_KPOINTS_2(NKPTS, VKPT, WTKPT, NTET, IDTET, VOLWGT, &
           CSEL, RKLEN, NKPX, NKPY, NKPZ, SUPL_SHIFT, SHIFT, BK)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER NKPTS             ! number of k-points
        REAL(q) :: VKPT(3, NKPTS) ! k-point coordinates
        REAL(q) :: WTKPT(NKPTS)   ! weigth for each k-point
        INTEGER :: NTET           ! number of tetrahedrons
        INTEGER :: IDTET(0:4,NTET)! list of tetrahedrons
        REAL(q) :: VOLWGT         ! volume weight for each tetrahedron
        CHARACTER (LEN=1) :: CSEL ! selector 
        REAL(q) :: RKLEN          ! length for subdivisions
        INTEGER :: NKPX,NKPY,NKPZ ! subdivisions
        REAL(q) :: SUPL_SHIFT(3)  ! shift of mesh supplied by user
        REAL(q) :: SHIFT(3)       ! shift of mesh with respect to origin
        REAL(q) :: BK(3,3)        ! generating vectors
      ! local
        INTEGER i
        CHARACTER (LEN=15):: METHOD
        INTEGER :: NK(3)

        CALL XML_TAG("kpoints")

        SELECT CASE(CSEL)
        CASE('A','a')
           METHOD='Auto'
        CASE('M','m')
           METHOD='Monkhorst-Pack'
        CASE('G','g')
           METHOD='Gamma'
        CASE DEFAULT 
           METHOD='Manual'
        END SELECT
        CALL XML_TAG("generation",param=TRIM(METHOD))
              
        IF (CSEL=='A' .OR. CSEL== 'a') THEN
           CALL XML_TAG_REAL("subdivisionlength", RKLEN)
        ENDIF

        SELECT CASE(CSEL)
        CASE('A','a','M','m','G','g')
           NK(1)=NKPX
           NK(2)=NKPY
           NK(3)=NKPZ
           CALL XML_VEC_INT( NK, NAME="divisions")
        END SELECT
        CALL XML_VEC_REAL( SUPL_SHIFT, NAME="usershift")

        CALL XML_VEC_REAL(BK(:,1), NAME="genvec1")
        CALL XML_VEC_REAL(BK(:,2), NAME="genvec2")
        CALL XML_VEC_REAL(BK(:,3), NAME="genvec3")
        CALL XML_VEC_REAL( SHIFT, NAME="shift")

        CALL XML_CLOSE_TAG("generation")

      ! kpoint and tetrahedron linklist
        IF (NTET==0) THEN
           CALL XML_KPOINTS_LIST(VKPT, WTKPT)
        ELSE
           CALL XML_KPOINTS_LIST(VKPT, WTKPT, IDTET, VOLWGT)
        ENDIF
        CALL XML_CLOSE_TAG("kpoints")

        
      END SUBROUTINE XML_KPOINTS_2

!=======================================================================
! 
! XML_KPOINTS_3
!   write k-points generated by interpolation from a list of k-points
!
!=======================================================================

      SUBROUTINE XML_KPOINTS_3(NKPTS, VKPT, WTKPT, NINTER)
        USE vaspxml
        USE prec
        IMPLICIT NONE

        INTEGER NKPTS     ! number of k-points
        INTEGER NINTER    ! number of interpolation points
        REAL(q) :: VKPT(3, NKPTS)
        REAL(q) :: WTKPT(NKPTS)
      ! local
        INTEGER i

        CALL XML_TAG("kpoints")
        CALL XML_TAG("generation",param=TRIM("listgenerated"))


        CALL XML_TAG_INT("divisions", NINTER)
        DO i=1,NKPTS,NINTER
           CALL XML_VEC_REAL(VKPT(:,i))
        ENDDO
        CALL XML_VEC_REAL(VKPT(:,NKPTS))
        CALL XML_CLOSE_TAG("generation")

        CALL XML_KPOINTS_LIST(VKPT, WTKPT)
        CALL XML_CLOSE_TAG("kpoints")
        
      END SUBROUTINE XML_KPOINTS_3

!=======================================================================
! 
! XML_BORN_CHARGES
!   write Born effective charge tensors for ions
!
!=======================================================================

      SUBROUTINE XML_BORN_CHARGES(BORN_CHARGES, NIONS)
        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER NIONS     ! number of ions
        REAL(q) :: BORN_CHARGES(3,3,NIONS)
      ! local
        INTEGER :: n
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("array", "born_charges")
        CALL XML_DIMENSION("ion",1)
        DO n=1,NIONS
           CALL XML_TAG("set")
           CALL XML_VEC_REAL(BORN_CHARGES(1,:,n))
           CALL XML_VEC_REAL(BORN_CHARGES(2,:,n))
           CALL XML_VEC_REAL(BORN_CHARGES(3,:,n))
           CALL XML_CLOSE_TAG("set")
        END DO
        CALL XML_CLOSE_TAG
        
      END SUBROUTINE XML_BORN_CHARGES
      
! phonon_
!=======================================================================
! 
! XML_FORCE_CONSTANTS
!   write force constant tensors: forces on the ions in the
!   primitive cell (as defined by the NIONP entries of MAP),
!   induced by displacements of all ions in the supercell.
!
!=======================================================================

      SUBROUTINE XML_FORCE_CONSTANTS(NIONS,NIONP,MAP,SECOND_DERIV)
        USE vaspxml
        USE prec
        IMPLICIT none

        INTEGER :: NIONS,NIONP
        INTEGER :: MAP(NIONP)
        REAL(q) :: SECOND_DERIV(3*NIONS,3*NIONS) 
      ! local
        INTEGER :: n,np,m
        CHARACTER (LEN=40) :: strcounter

        CALL XML_TAG("array", "force_constants")
        CALL XML_DIMENSION("cart_super",1)
        CALL XML_DIMENSION("ion_super",2)
        CALL XML_DIMENSION("ion_primitive",3)
        CALL XML_FIELD("f_cart_prim_1",type="float")
        CALL XML_FIELD("f_cart_prim_2",type="float")
        CALL XML_FIELD("f_cart_prim_3",type="float")
        CALL XML_TAG("set")

        DO n=1,NIONP
           np=MAP(n)
           WRITE(strcounter,"(I6)") np
           CALL XML_TAG("set", comment="ionprim "//TRIM(ADJUSTL(strcounter)))
           DO m=1,NIONS
              WRITE(strcounter,"(I6)") m
              CALL XML_TAG("set", comment="ion "//TRIM(ADJUSTL(strcounter)))
              CALL XML_ROW_DATA(SECOND_DERIV(1+3*(np-1):3+3*(np-1),1+3*(m-1)))
              CALL XML_ROW_DATA(SECOND_DERIV(1+3*(np-1):3+3*(np-1),2+3*(m-1)))
              CALL XML_ROW_DATA(SECOND_DERIV(1+3*(np-1):3+3*(np-1),3+3*(m-1)))
              CALL XML_CLOSE_TAG("set")
           ENDDO
           CALL XML_CLOSE_TAG("set")
        ENDDO

! transposed version
!        CALL XML_DIMENSION("cart_prim",1)
!        CALL XML_DIMENSION("ion_primitive",2)
!        CALL XML_DIMENSION("ion_super",3)
!        CALL XML_FIELD("f_cart_super_1",type="float")
!        CALL XML_FIELD("f_cart_super_2",type="float")
!        CALL XML_FIELD("f_cart_super_3",type="float")
!        CALL XML_TAG("set")
!        DO m=1,NIONS
!           WRITE(strcounter,"(I6)") m
!           CALL XML_TAG("set", comment="ion "//TRIM(ADJUSTL(strcounter)))
!           DO n=1,NIONP
!              np=MAP(n)
!              WRITE(strcounter,"(I6)") np
!              CALL XML_TAG("set", comment="ionprim "//TRIM(ADJUSTL(strcounter)))
!              CALL XML_ROW_DATA(SECOND_DERIV(1+3*(np-1),1+3*(m-1):3*m))
!              CALL XML_ROW_DATA(SECOND_DERIV(2+3*(np-1),1+3*(m-1):3*m))
!              CALL XML_ROW_DATA(SECOND_DERIV(3+3*(np-1),1+3*(m-1):3*m))
!              CALL XML_CLOSE_TAG("set")
!           ENDDO
!           CALL XML_CLOSE_TAG("set")
!        ENDDO

        CALL XML_CLOSE_TAG("set")
        CALL XML_CLOSE_TAG("array")

      END SUBROUTINE XML_FORCE_CONSTANTS
! phonon_
