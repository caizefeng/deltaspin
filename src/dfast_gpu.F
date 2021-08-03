#include "symbol.inc"

MODULE dfast_gpu
  USE iso_c_binding
  USE prec
  USE mpimy
  USE dfast
  
 INTERFACE
      SUBROUTINE ORTH1_GPU(CSEL,GPU_CPTWFP,GPU_CFW,GPU_CPROJ,GPU_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL)
      USE iso_c_binding
      USE prec
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL

      CHARACTER*(*) CSEL
      INTEGER :: stat
    END SUBROUTINE ORTH1_GPU
 END INTERFACE

   INTERFACE
    SUBROUTINE ORTH2_GPU(GPU_CPTWFP,GPU_CFW,GPU_CPROJ,GPU_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL)
      USE iso_c_binding
      USE prec
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL

    END SUBROUTINE ORTH2_GPU
 END INTERFACE

CONTAINS

      SUBROUTINE LINCOM_GPU(MODE,GPU_CF,GPU_CPROF,GPU_CTRANS,NIN,NOUT,NPL, &
     &           NPRO,NPLDIM,NPROD,LDTRAN,GPU_CFA,GPU_CPROFA)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      
      INTEGER, PARAMETER :: STREAM_BLOCK_SIZE = 4

      CHARACTER*1 MODE
      INTEGER(c_intptr_t)  GPU_CF,GPU_CFA
      INTEGER(c_intptr_t)  GPU_CPROF,GPU_CPROFA
      INTEGER(c_intptr_t)  GPU_CTRANS
      INTEGER     TAILLE
      ! work array
      INTEGER(c_intptr_t) GPU_CBLOCK(STREAM_BLOCK_SIZE)
      COMPLEX(q) fakec
#ifdef TAU_PROF
!nvidia: some kind of hack to prevent compilation errors with tau
    Integer dummy
    dummy = 0
#endif

nv_profile_start(NVP_LINCOM)

      TAILLE = NBLK*LDTRAN
nv_profile_start(NVP_LINCOM_MALLOC)
      DO II=1, STREAM_BLOCK_SIZE
      call cublas_Alloc_safety (TAILLE,int(c_sizeof(fakec),c_size_t),GPU_CBLOCK(II))
      ENDDO
nv_profile_stop(NVP_LINCOM_MALLOC)

nv_profile_start(NVP_LINCOM_LINBAS)
      CALL LINBAS_GPU(MODE,GPU_CF,GPU_CBLOCK,GPU_CTRANS,NIN,NOUT,m_ NPL, &
     &            m_ NPLDIM,LDTRAN,m_ NBLK,GPU_CFA)
      IF (NPRO/=0) THEN
      CALL LINBAS_GPU(MODE,GPU_CPROF,GPU_CBLOCK,GPU_CTRANS,NIN,NOUT, NPRO, &
     &             NPROD,LDTRAN,m_ NBLK,GPU_CPROFA)
      ENDIF
nv_profile_stop(NVP_LINCOM_LINBAS)

      !call cublas_Free(GPU_CBLOCK)
      !call THREADSYNCHRONIZE()
nv_profile_start(NVP_LINCOM_MALLOC)
      DO II=1, STREAM_BLOCK_SIZE
      call cublas_free(GPU_CBLOCK(II))
      ENDDO
nv_profile_stop(NVP_LINCOM_MALLOC)

nv_profile_stop(NVP_LINCOM)
      RETURN
      END SUBROUTINE LINCOM_GPU

END MODULE dfast_gpu 

      SUBROUTINE LINBAS_GPU(MODE,GPU_CF,GPU_CBLOCK,GPU_CTRANS,NIN,NOUT,NPL, &
     &           NPLDIM,LDTRAN,NBLK,GPU_CFA)
      USE iso_c_binding
      USE cuda_interface
      USE prec
      USE gpu_data

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      IMPLICIT INTEGER (N)

      INTEGER, PARAMETER ::  STREAM_BLOCK_SIZE=4
      INTEGER(c_intptr_t)     GPU_CF,GPU_CFA
      INTEGER(c_intptr_t)     GPU_CBLOCK(STREAM_BLOCK_SIZE)
      INTEGER(c_intptr_t)     GPU_CTRANS
      GDEF,ALLOCATABLE, TARGET::CTRANS(:,:)
      GDEF,ALLOCATABLE::CTRANS_TMP(:,:)
      INTEGER LDTRAN
      REAL (q)  TV,TV1,TC,TC1
      GDEF fakeg

      PARAMETER(IUSETR=0)

      CHARACTER*1 MODE
      LOGICAL     LTRI,LADD,LBOTH,LTRANS
      INTEGER STREAM
      COMPLEX(q) fakec

nv_profile_start(NVP_LINBAS)
      !write(*,*) "Entering LinbasGPU loop..."

      IF (NOUT>NIN) THEN
         WRITE(*,1)
 1       FORMAT('internal error in routine LINBAS: wrong arguments, NOUT>NIN')
         STOP
      ENDIF

      ALLOCATE (CTRANS(LDTRAN,NIN),CTRANS_TMP(LDTRAN,NIN))

      LTRI =(MODE=='U').OR.(MODE=='u').OR. &
     &      (MODE=='L').OR.(MODE=='l')
      LADD =(MODE=='A').OR.(MODE=='a')
      LBOTH=(MODE=='B').OR.(MODE=='b')
      LTRANS=(MODE=='T').OR.(MODE=='t').OR.(MODE=='C').OR.(MODE=='c')
    
      IF (LTRI.AND.(IUSETR==0)) THEN
      !print *,"NIN =",NIN
nv_profile_start(NVP_LINBAS_MEMCPY)
      CALL cublas_Get_Matrix(LDTRAN,NIN,int(c_sizeof(fakeg),c_int),GPU_CTRANS,LDTRAN,c_loc(CTRANS),LDTRAN)
nv_profile_stop(NVP_LINBAS_MEMCPY)
         DO 4 N2=1,NIN
            IF ((MODE=='L').OR.(MODE=='l')) THEN
!DIR$ IVDEP
!OCL NOVREC
               DO N1=1,N2-1
               CTRANS(N1,N2)= zero
               ENDDO
            ELSE
!DIR$ IVDEP
!OCL NOVREC
               DO N1=N2+1,NIN
               CTRANS(N1,N2)= zero
               ENDDO
            ENDIF
    4    ENDDO
nv_profile_start(NVP_LINBAS_MEMCPY)
      CALL cublas_Set_Matrix(LDTRAN,NIN,int(c_sizeof(fakeg),c_int),c_loc(CTRANS),LDTRAN,GPU_CTRANS,LDTRAN)
nv_profile_stop(NVP_LINBAS_MEMCPY)
      ENDIF
    
! Try to get best load balance, maximum block size < NBLK ...
nv_profile_start(NVP_LINBAS_COMPUTE)
nv_profile_start(NVP_GEMM)
      NBLOCK=NBLK
      STREAM = 0
      DO 70 IBLOCK=0,NPL-1,NBLOCK
         ILENPL=MIN(NBLOCK,NPL-IBLOCK)
         IADDPL=MIN(IBLOCK,NPL-1)
         ILENPL=MAX(ILENPL,0)
         STREAM = MOD(STREAM, STREAM_BLOCK_SIZE)+1

         IF (LTRI.AND.(IUSETR/=0)) THEN
            print *,"LINBAS_GPU: LTRI.AND.(IUSETR/=0) not yet on GPU"
            call cuda_device_reset()
            stop
! 'Triangular update':
#ifdef gammareal
!            CALL DTRMM &
#else
!            CALL ZTRMM &
#endif
!     &                ('R',MODE,'N','N', ILENPL,NOUT,one, &
!     &                 CTRANS,LDTRAN,CF(1+IADDPL,1), NPLDIM)
         ELSE
            !write(*,*) "LINBAS_GPU...,STREAM=",STREAM
            !write(*,*) "LBOTH,LADD=",LBOTH,LADD
            !write(*,*) "M,N=",ILENPL,NIN
            !write(*,*) "Tx,Ty=",NPLDIM,NBLK
            !write(*,*) "shiftX=",IDX(1+IADDPL,1,NPLDIM)
            IF (LBOTH.OR.(.NOT.LADD)) THEN
               !write(*,*) "calling gpu_copyz_st..."
               call cuda_zsetmatrix(STREAM,ILENPL,NIN,GPU_CF+IDX(1+IADDPL,1,NPLDIM)*int(c_sizeof(fakec),c_size_t),NPLDIM,GPU_CBLOCK(STREAM),NBLK)
               !CALL gpu_copyz_st(STREAM, ILENPL,NIN,GPU_CF,NPLDIM,GPU_CBLOCK(STREAM),NBLK,IDX(1+IADDPL,1,NPLDIM),0)
               !call threadsynchronize()
               !call cuda_print("cblock.dat",'c',NIN*NBLK,GPU_CBLOCK(STREAM))
               !call cuda_device_reset()
               !stop
               IF (LTRANS) THEN
                 CALL CUBLAS_ZGEMMSH_ST(STREAM,'N', MODE, ILENPL, NOUT, NIN, one, &
     &                  GPU_CBLOCK(STREAM),IDX(1,1,NBLK), NBLK, GPU_CTRANS, IDX(1,1,LDTRAN), &
     &                  LDTRAN, zero, GPU_CF, IDX(IADDPL+1,1,NPLDIM), NPLDIM)
               ELSE
                 CALL CUBLAS_ZGEMMSH_ST(STREAM, 'N', 'N', ILENPL, NOUT, NIN, one, &
     &                  GPU_CBLOCK(STREAM),IDX(1,1,NBLK), NBLK, GPU_CTRANS, IDX(1,1,LDTRAN), &
     &                  LDTRAN, zero, GPU_CF, IDX(IADDPL+1,1,NPLDIM), NPLDIM)
               ENDIF
            ENDIF
            IF (LBOTH.OR.LADD) THEN

               IADDT=0
               IF (LBOTH) IADDT=NIN
               stop
               call cuda_zsetmatrix(STREAM,ILENPL,NIN,GPU_CFA+IADDPL*int(c_sizeof(fakec),c_size_t),NPLDIM,GPU_CBLOCK(STREAM),NBLK)
               !CALL gpu_copyz_st(ILENPL,NIN,GPU_CFA,NPLDIM,GPU_CBLOCK(STREAM),NBLK,IADDPL,0)
               
               CALL CUBLAS_ZGEMMSH_ST(STREAM, 'N', 'N', ILENPL, NOUT, NIN, one, &
     &               GPU_CBLOCK(STREAM),IDX(1,1,NBLK), NBLK, GPU_CTRANS, IDX(1+IADDT,1,LDTRAN), &
     &               LDTRAN, one, GPU_CF, IDX(IADDPL+1,1,NPLDIM), NPLDIM)
            ENDIF
         ENDIF
   70 CONTINUE
      call THREADSYNCHRONIZE()
nv_profile_stop(NVP_GEMM)
nv_profile_stop(NVP_LINBAS_COMPUTE)
      DEALLOCATE(CTRANS)

nv_profile_stop(NVP_LINBAS)
      RETURN
      END SUBROUTINE

            SUBROUTINE ORTH1_GPU(CSEL,GPU_CPTWFP,GPU_CFW,GPU_CPROJ,GPU_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL)
      USE iso_c_binding
      USE prec
      USE cuda_interface
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL

      CHARACTER*(*) CSEL
      INTEGER :: stat

      IF (NSTRIP+NPOS-1 > NBANDS) THEN
        WRITE(*,*)'internal error in ORTH1: dim=',NSTRIP+NPOS,NBANDS
        STOP
      ENDIF
      
!
! update of lower triangular part
!
    IF (CSEL(1:1) == 'L' .OR. CSEL(1:1) == 'l') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
      !TODO: CUBLAS using STREAM
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDIF

      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDIF
!
! update of upper triangular part
!
    ELSE IF (CSEL(1:1) == 'U' .OR. CSEL(1:1) == 'u') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,IDX(NPOSPR,1,NPROD),NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

    ELSE
      WRITE(*,*)'internal error in ORTH1: CSEL=',CSEL
    ENDIF
      
      RETURN
      END SUBROUTINE ORTH1_GPU
 
      SUBROUTINE ORTH1_GPU_STREAM(STREAM, CSEL,GPU_CPTWFP,GPU_CFW,GPU_CPROJ,GPU_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL)
      USE iso_c_binding
      USE prec
      USE cuda_interface
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)
      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL
      INTEGER STREAM

      CHARACTER*(*) CSEL
      INTEGER :: stat

      IF (NSTRIP+NPOS-1 > NBANDS) THEN
        WRITE(*,*)'internal error in ORTH1: dim=',NSTRIP+NPOS,NBANDS
        STOP
      ENDIF
      
!
! update of lower triangular part
!
    IF (CSEL(1:1) == 'L' .OR. CSEL(1:1) == 'l') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
      !TODO: CUBLAS using STREAM
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDIF

      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH_ST(STREAM,trans,'N',NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDIF
!
! update of upper triangular part
!
    ELSE IF (CSEL(1:1) == 'U' .OR. CSEL(1:1) == 'u') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,IDX(NPOSPR,1,NPROD),NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL CUBLAS_ZGEMMSH_ST(STREAM, trans,'N',NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

    ELSE
      WRITE(*,*)'internal error in ORTH1: CSEL=',CSEL
    ENDIF
      
      RETURN
      END SUBROUTINE ORTH1_GPU_STREAM
      


      SUBROUTINE ORTH1_GPU_SHIFT (CSEL,GPU_CPTWFP,SH_CPTWFP,GPU_CFW,SH_CFW,GPU_CPROJ,SH_CPROJ,GPU_CPROW,SH_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL,SH_COVL)
      USE iso_c_binding
      USE prec
      USE cuda_interface
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL
      integer    m_tile
      REAL (q)  TV,TV1,TC,TC1
#ifdef TAU_PROF
    integer, dimension (2,4) :: profile
    save profile
#endif

      INTEGER    SH_CPTWFP, SH_CFW, SH_CPROJ, SH_CPROW, SH_COVL

      CHARACTER*(*) CSEL
      INTEGER :: stat

      IF (NSTRIP+NPOS-1 > NBANDS) THEN
        WRITE(*,*)'internal error in ORTH1: dim=',NSTRIP+NPOS,NBANDS
        STOP
      ENDIF
!
! update of lower triangular part
!
    IF (CSEL(1:1) == 'L' .OR. CSEL(1:1) == 'l') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,1),'ORTH1_GPU_SHIFT_ZGEMM 1')
    call TAU_PROFILE_START(profile(1,1))
#endif
      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
      !TODO: CUBLAS using STREAM
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO


#if 1
      !cut large ZGEMM into smaller ZGEMMS along the m-dimension. This
      !increases performance (might be obsolete for future CUBLAS versions).

      m_tile = 1400
      num_m_tiles = (NBANDS-NPOS+1 + m_tile - 1 ) / m_tile
      do i = 0,(num_m_tiles-1)
         num_rows = min(m_tile, (NBANDS-NPOS+1) - (i * m_tile))
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH(trans,'N',num_rows ,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM) + m_tile * i * (m_ NPLDIM) ,m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS) + m_tile * i , NBANDS )
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",num_rows,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*Real(num_rows)*NSTRIP*(m_ NPL-NPOSPL+1)/Real(TC1-TC)/1e9 !REMOVE
      enddo 
#else
         CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
         CALL THREADSYNCHRONIZE ()!REMOVE
         CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*Real(NBANDS-NPOS+1)*NSTRIP*(m_ NPL-NPOSPL+1)/Real(TC1-TC)/1e9 !REMOVE
#endif
      
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,1))
#endif
      ENDIF

      IF (NPRO/=0) THEN

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,2),'ORTH1_GPU_SHIFT_ZGEMM 2')
    call TAU_PROFILE_START(profile(1,2))
#endif
      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM2:",NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,Real(TC1-TC),8.0*(NBANDS-NPOS+1)*NSTRIP*(NPRO-NPOSPR+1)/Real(TC1-TC)/1e9 !REMOVE
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,2))
#endif
      ENDIF
!
! update of upper triangular part
!
    ELSE IF (CSEL(1:1) == 'U' .OR. CSEL(1:1) == 'u') THEN
#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,3),'ORTH1_GPU_SHIFT_ZGEMM NPL U')
    call TAU_PROFILE_START(profile(1,3))
#endif
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*REAL((NSTRIP+NPOS-1)*NSTRIP *(NPL-NPOSPL+1))/Real(TC1-TC)/1e9 !REMOVE
      ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,3))
#endif

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,4),'ORTH1_GPU_SHIFT_ZGEMM NPRO U')
    call TAU_PROFILE_START(profile(1,4))
#endif
      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,1,NPROD),NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH(trans,'N',NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM2:",NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,Real(TC1-TC),8.0*REAL((NSTRIP+NPOS-1)*NSTRIP*(NPRO-NPOSPR+1))/Real(TC1-TC)/1e9 !REMOVE
      ENDIF

#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,4))
#endif

    ELSE
      WRITE(*,*)'internal error in ORTH1: CSEL=',CSEL
    ENDIF
      
      RETURN
      END SUBROUTINE ORTH1_GPU_SHIFT

   SUBROUTINE ORTH2_GPU(GPU_CPTWFP,GPU_CFW,GPU_CPROJ,GPU_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL)
      USE iso_c_binding
      USE prec
      USE cuda_interface

      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL

      IF (NPL/=0) THEN
      ! here external blocking can be done, but pretty useless
      NBLOCK=m_ NPL
      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
     !    CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
     !         GPU_CPTWFP,IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
     !         m_ NPLDIM,one,GPU_COVL,IDX(NPOS,NPOS,NBANDS),NBANDS)
      
      CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS,NSTRIP,NBLOCK,one, &
     &         GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
     &         m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
      CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS,NSTRIP,m_ NPL-NPOSPL+1,one, &
     &         GPU_CPTWFP,IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,IDX(NPOSPL,1,NPLDIM), &
     &         m_ NPLDIM,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

      IF (NPRO/=0) THEN
      ! here external blocking can be done, but pretty useless
      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
      CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS,NSTRIP,NBLOCK,one, &
     &         GPU_CPROJ,IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
     &          NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
      CALL CUBLAS_ZGEMMSH(trans,'N',NBANDS,NSTRIP,NPRO-NPOSPR+1,one, &
     &         GPU_CPROJ,IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,IDX(NPOSPR,1,NPROD), &
     &          NPROD,one,GPU_COVL,IDX(1,NPOS,NBANDS),NBANDS)
      ENDIF

      RETURN
    END SUBROUTINE ORTH2_GPU


#ifdef MIXED_PRECISION_GPU
! Mixed precision, not fully tested implemented by NVPS
      SUBROUTINE ORTH1_GPU_SHIFT_SP (CSEL,GPU_CPTWFP,SH_CPTWFP,GPU_CFW,SH_CFW,GPU_CPROJ,SH_CPROJ,GPU_CPROW,SH_CPROW,NBANDS, &
     &  NPOS,NSTRIP,NPL,NPRO,NPLDIM,NPROD,GPU_COVL,SH_COVL)
      USE iso_c_binding
      USE prec
      
      IMPLICIT COMPLEX(q) (C)
      IMPLICIT REAL(q) (A-B,D-H,O-Z)

      INTEGER(c_intptr_t) GPU_CPTWFP
      INTEGER(c_intptr_t) GPU_CFW
      INTEGER(c_intptr_t) GPU_CPROJ
      INTEGER(c_intptr_t) GPU_CPROW
      INTEGER(c_intptr_t) GPU_COVL
      integer    m_tile
      REAL (q)  TV,TV1,TC,TC1
#ifdef TAU_PROF
    integer, dimension (2,4) :: profile
    save profile
#endif

      INTEGER    SH_CPTWFP, SH_CFW, SH_CPROJ, SH_CPROW, SH_COVL

      CHARACTER*(*) CSEL
      INTEGER :: stat

      IF (NSTRIP+NPOS-1 > NBANDS) THEN
        WRITE(*,*)'internal error in ORTH1: dim=',NSTRIP+NPOS,NBANDS
        STOP
      ENDIF

!
! update of lower triangular part
!
    IF (CSEL(1:1) == 'L' .OR. CSEL(1:1) == 'l') THEN
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,1),'ORTH1_GPU_SHIFT_ZGEMM 1')
    call TAU_PROFILE_START(profile(1,1))
#endif
      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
      !TODO: CUBLAS using STREAM
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO


#if 1
      !cut large ZGEMM into smaller ZGEMMS along the m-dimension. This
      !increases performance (might be obsolete for future CUBLAS versions).

      m_tile = 1400
      num_m_tiles = (NBANDS-NPOS+1 + m_tile - 1 ) / m_tile
      do i = 0,(num_m_tiles-1)
         num_rows = min(m_tile, (NBANDS-NPOS+1) - (i * m_tile))
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',num_rows ,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM) + m_tile * i * (m_ NPLDIM) ,m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS) + m_tile * i , NBANDS )
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",num_rows,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*Real(num_rows)*NSTRIP*(m_ NPL-NPOSPL+1)/Real(TC1-TC)/1e9 !REMOVE
      enddo 
#else
         CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,NPOS,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
         CALL THREADSYNCHRONIZE ()!REMOVE
         CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",NBANDS-NPOS+1,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*Real(NBANDS-NPOS+1)*NSTRIP*(m_ NPL-NPOSPL+1)/Real(TC1-TC)/1e9 !REMOVE
#endif
      
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,1))
#endif
      ENDIF

      IF (NPRO/=0) THEN

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,2),'ORTH1_GPU_SHIFT_ZGEMM 2')
    call TAU_PROFILE_START(profile(1,2))
#endif
      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NBANDS-NPOS+1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
      ENDDO
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,NPOS,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(NPOS,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM2:",NBANDS-NPOS+1,NSTRIP,NPRO-NPOSPR+1,Real(TC1-TC),8.0*(NBANDS-NPOS+1)*NSTRIP*(NPRO-NPOSPR+1)/Real(TC1-TC)/1e9 !REMOVE
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,2))
#endif
      ENDIF
!
! update of upper triangular part
!
    ELSE IF (CSEL(1:1) == 'U' .OR. CSEL(1:1) == 'u') THEN
#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,3),'ORTH1_GPU_SHIFT_ZGEMM NPL U')
    call TAU_PROFILE_START(profile(1,3))
#endif
      IF (NPL/=0) THEN
      NBLOCK=m_ NPL

      DO NPOSPL=1,m_ NPL-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         !CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,one, &
              GPU_CPTWFP,SH_CPTWFP+IDX(NPOSPL,1,NPLDIM),m_ NPLDIM,GPU_CFW,SH_CFW+IDX(NPOSPL,1,NPLDIM), &
              m_ NPLDIM,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM1:",NSTRIP+NPOS-1,NSTRIP,m_ NPL-NPOSPL+1,Real(TC1-TC),8.0*REAL((NSTRIP+NPOS-1)*NSTRIP *(NPL-NPOSPL+1))/Real(TC1-TC)/1e9 !REMOVE
      ENDIF
#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,3))
#endif

#ifdef TAU_PROF
    call TAU_PROFILE_TIMER(profile(1,4),'ORTH1_GPU_SHIFT_ZGEMM NPRO U')
    call TAU_PROFILE_START(profile(1,4))
#endif
      IF (NPRO/=0) THEN

      NBLOCK=NPRO
      DO NPOSPR=1,NPRO-NBLOCK,NBLOCK
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NSTRIP+NPOS-1,NSTRIP,NBLOCK,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,1,NPROD),NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
      ENDDO
         CALL VTIME(TV,TC)       
         CALL CUBLAS_ZGEMMSH_SP(trans,'N',NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,one, &
              GPU_CPROJ,SH_CPROJ+IDX(NPOSPR,1,NPROD), NPROD,GPU_CPROW,SH_CPROW+IDX(NPOSPR,1,NPROD), &
              NPROD,one,GPU_COVL,SH_COVL+IDX(1,NPOS,NBANDS),NBANDS)
         !CALL THREADSYNCHRONIZE ()!REMOVE
         !CALL VTIME(TV1,TC1)       
         !print *, "C N ZGEMM2:",NSTRIP+NPOS-1,NSTRIP,NPRO-NPOSPR+1,Real(TC1-TC),8.0*REAL((NSTRIP+NPOS-1)*NSTRIP*(NPRO-NPOSPR+1))/Real(TC1-TC)/1e9 !REMOVE
      ENDIF

#ifdef TAU_PROF
    call TAU_PROFILE_STOP(profile(1,4))
#endif

    ELSE
      WRITE(*,*)'internal error in ORTH1: CSEL=',CSEL
    ENDIF
      
      RETURN
      END SUBROUTINE
#endif
