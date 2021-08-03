#define IDX(i,j,ld)       ((((j)-1)*(ld))+(((i)-1)))
#define IDX3(i,j,k,ld,lp) (((ld)*(lp)*((k)-1))+(((j)-1)*(ld))+(((i)-1)))
!
!   FFT type
!
!Double to Double-complex FFT
#define FFT_R2C 106   
!Double-complex to Double FFT
#define FFT_C2R 108   
!Double-complex to double-complex FFT
#define FFT_C2C 105   
