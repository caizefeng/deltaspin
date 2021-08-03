#include "symbol.inc"
MODULE nonl_high_gpu
  USE prec
  USE nonlr
  USE nonl
  USE nonl_high

CONTAINS 
  SUBROUTINE W1_PROJALL_GPU(WDES1, W1, NONLR_S, NONL_S, NMAX, GPU_CR, TimeTransfert)
    USE gpu_data
    IMPLICIT NONE
 
    TYPE (wavedes1) :: WDES1
    TYPE (wavefun1) :: W1(:)
    TYPE (nonlr_struct) NONLR_S
    TYPE (nonl_struct) NONL_S
    INTEGER, OPTIONAL :: NMAX
    !TYPE(gpu_type) GPU(SIZE(W1))
    INTEGER(c_intptr_t) GPU_CR(:)
  ! local
    INTEGER NMAX_, NP
    REAL(q) TimeTransfert,TV0,TC0,TV,TC

    IF (PRESENT(NMAX)) THEN
        NMAX_=NMAX
    ELSE
        NMAX_=SIZE(W1)
    ENDIF
 
    IF ( NONLR_S%LREAL ) THEN
        !IF (NMAX_ >1 ) THEN
        CALL RPROMU_GPU(NONLR_S,WDES1,W1,NMAX_,W1%LDO,GPU_CR,TimeTransfert,NV_NUM_BATCHES)
        !ELSE
        !DO NP=1,NMAX_
        !    IF (.NOT. W1(NP)%LDO) CYCLE
        !    call cuda_memcpydtoh(NULL_STREAM,c_loc(W1(NP)%CR),GPU_CR(NP),WDES1%GRID%MPLWV,int(c_sizeof(fakec),c_size_t))
        !    CALL RPRO1(NONLR_S,WDES1,W1(NP))
        !ENDDO
        !ENDIF
    ELSE
        DO NP=1,NMAX_
            IF (.NOT. W1(NP)%LDO) CYCLE
            CALL PROJ1(NONL_S,WDES1,W1(NP))
        ENDDO
    ENDIF
  END SUBROUTINE W1_PROJALL_GPU

END module
