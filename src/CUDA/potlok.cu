/*******************************************************************************
*
*                               WARNING
* 
*       This file implements the functions as used and called in the Fortran 
*       file: 'xclib_grad.F' for the code paths in which the value of 
*       LEXCH equals 7,8 or 9. Any changes that involve these code paths in 
*       the original 'xclib_grad.F' should also be made in this file to keep 
*       them consistent.
* 
*       If during testing you observe differences between the GPU and CPU implementations 
*       then this part of the GPU implementation can be disabled by defining the following 
*       in the Makefile: 'DO_NOT_USE_POTLOK_GPU' . This will force execution of the 
*       CPU code paths for all values of LEXCH. This choice is made in the file 'xcgrad.F'.
* 
* 
***********************************************************************************/

// File: potlok.cu
// C/Fortran interface to GPU port of xclib_grad.F.
// Disable with DO_NOT_USE_POTLOK_GPU flag in makefile.

// includes standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// includes cuda headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas.h"   /* CUBLAS public header file  */
//#include "cublasP.h"  /* CUBLAS private header file */
#include "Operator.h"


typedef size_t devptr_t;


__device__ __forceinline__ void gpu_gcor(const double A, const double A1, const double B1, const double B2, const double B3, const double B4, const double P, const double RS, double & GG, double & GGRS)
{
      const double P1 = P + 1.;
      const double Q0 = -2.*A*(1.+A1*RS);
      const double RS12 = sqrt(RS);
      const double RS32 = RS12*RS12*RS12;
      const double RSP = pow(RS,P);
      const double Q1 = 2.*A*(B1*RS12+B2*RS+B3*RS32+B4*RS*RSP);
      const double Q2 = log(1.+1./Q1);
      GG = Q0*Q2;
      const double Q3 = A*(B1/RS12+2.*B2+3.*B3*RS12+2.*B4*P1*RSP);
      GGRS = -2.*A*A1*Q2-Q0*Q3/(Q1*Q1+Q1);
}

__device__ __forceinline__ void gpu_corgga1(const double D, const double RS, const double T, double & EC1, double & EC1D, double & EC1DD, const double FK, const double SK, const double G, const double EC, const double ECRS)
{
      const double XNU=15.75592,CC0=0.004235,CX=-0.001667212,ALF=0.09;
      const double C1=0.002568,C2=0.023266,C3=7.389E-6,C4=8.723;
      const double C5=0.472,C6=7.389E-2,A4=100.0;
      const double THRD=1./3. ,SIXTH7=7./6.;
      const double BET = XNU*CC0;
      const double DELT = 2.0*ALF/BET;
      const double G3 = G*G*G;
      const double G4 = G3*G;
      const double PON = -DELT*EC/(G3*BET);
      const double B = DELT/(exp(PON)-1.0);
      const double B2 = B*B;
      const double T2 = T*T;
      const double T4 = T2*T2;
      const double T6 = T4*T2;
      const double RS2 = RS*RS;
      const double RS3 = RS2*RS;
      const double Q4 = 1.0+B*T2;
      const double Q5 = 1.0+B*T2+B2*T4;
      const double Q6 = C1+C2*RS+C3*RS2;
      const double Q7 = 1.0+C4*RS+C5*RS2+C6*RS3;
      const double CC = -CX + Q6/Q7;
      const double R0temp = SK/FK;
      const double R0 = R0temp*R0temp;
      const double R1 = A4*R0*G4;
      const double COEFF = CC-CC0-3.0*CX/7.0;
      const double R2 = XNU*COEFF*G3;
      const double R3 = exp(-R1*T2);
      const double H0 = G3*(BET/DELT)*log(1.0+DELT*Q4*T2/Q5);
      const double H1 = R3*R2*T2;
      const double H = H0+H1;
//!============================================================
      const double Q8 = Q5*Q5+DELT*Q4*Q5*T2;
      const double H0T = 2.0*BET*T*(1.0+2.0*B*T2)/Q8;
      const double H0B = -BET*T6*(2.0*B+B2*T2)/Q8;
      const double H0RS = H0B*B*ECRS*(B+DELT)/BET;
      const double H1T = 2.0*R3*R2*T*(1.0-R1*T2);
      const double CCRS = (C2+2.*C3*RS)/Q7 - Q6*(C4+2.*C5*RS+3.*C6*RS2)/(Q7*Q7);
      const double R1RS = 100.0*R0/RS;
      const double H1RS = XNU*T2*R3*(CCRS - COEFF*T2*R1RS);
//! = = = = = = = = = = = =
      const double HT = H0T + H1T;
      const double HRS = H0RS + H1RS;
      EC1 = D*H;
      EC1D = H-THRD*RS*HRS-SIXTH7*T*HT;
      EC1DD = 0.5*HT/SK;
}

__device__  __forceinline__ void gpu_gga91_wb(const double d, const double dd, double & exc,
                                              double & excd, double & excdd,
                                              const double AGGAC, const double AGGAX){
    if(d < 0)
    {
        exc = 0;
        excd = 0;
        excdd = 0;
        return;
    }
    const double thrd = 1./3.;
    const double g = 1.;
    const double pmid = M_PI*d;
    const double fk = pow(3*M_PI*pmid, thrd);
    const double sk = sqrt(4.*fk/M_PI);
    double s1, t1;
    if(d > 1e-10)
    {
        const double temp = 2.*d;
        s1 = dd/(temp*fk);
        t1 = dd/(temp*sk);
    }
    else
    {
        s1 = 0;
        t1 = 0;
    }
    
    const double s = s1, t = t1;
    //EXCH2 subroutine
    {
      const double A1=0.19645,A2=0.27430,A3=0.15084,A4=100.;
      const double AX=-0.7385588,A=7.7956,B1=0.004;
      const double THPITH=3.0936677262801;
      const double thrd4 = 4./3.;
      const double fac = AX*pow(d,thrd);
      const double s2 = s*s;
      const double s3 = s2*s;
      const double s4 = s2*s2;
      const double p0i = sqrt(1.0+A*A*s2);
      const double p0 = 1.0/p0i;
      const double p1 = log(A*s+p0i);
      const double p2 = exp(-A4*s2);
      const double p3 = 1.0/(1.0+A1*s*p1+B1*s4);
      const double p4 = 1.0+A1*s*p1+(A2-A3*p2)*s2;
      const double f = p3*p4;
      exc = fac*(f-1)*d;
      const double p5 = 2.0*(s*(A2-A3*p2)+A3*A4*s3*p2-2.0*B1*s3);
      const double p6 = (A1*(p1+A*s*p0)+4.0*B1*s3)*((A2-A3*p2)*s2-B1*s4);
      const double fs = (p5-p6*p3)*p3;
      excd = thrd4*fac*(f-s*fs-1);
      excdd = AX*fs*0.5/THPITH;
    }
    const double rs = pow(0.75/pmid, thrd);
    double eu, eurs;
   //end of EXCH2 subroutine
      gpu_gcor(0.0310907,0.21370,7.5957,3.5876,1.6382, 0.49294,1.00,rs,eu,eurs);
      double ec1, ec1d, ec1dd;
      gpu_corgga1(d,rs,t,ec1,ec1d,ec1dd, fk,sk,g,eu,eurs);
//      if(blockIdx.x == 0 && threadIdx.x == 1)
//          printf("Przed dodaniem: %e %e %e\n", exc, excd, excdd);
      
      //JBNV, as used in 5.2.12
      exc   = (exc*AGGAX)   + (ec1*AGGAC);   //EX*AGGAX + (EC+EC1)*AGGAC  (EC =0, EXC = EX)
      excd  = (excd*AGGAX)  + (ec1d*AGGAC);  //EXD*AGGAX + (ECD+EC1D)*AGGAC  (ECD =0, EXCD = EXD)
      excdd = (excdd*AGGAX) + (ec1dd*AGGAC); //EXDD*AGGAX + (EC1DD*AGGAC)  (EXDD =excdd)
}

__device__ __forceinline__ void gpu_exchpbe(const double rho, const double rhothrd,const double s, double & exlda,double & expbe,
                                            double & exdlda,double & exd,double & exdd, const double ukfactor, const int lexch)
{
    const double AX=-0.738558766382022405884230032680836;
    const double um=0.2195149727645171, uk1=0.8040, ul1=um/uk1;
//construct LDA exchange energy density
      const double exunif=AX*rhothrd;
      exlda=exunif*rho;
      exdlda=exunif*4./3.;
//----------------------------------------------------------------------
// construct PBE enhancement factor
      const double S2 = s*s;
//----------------------------------------------------------------------
      double FxPBE, Fs;
      //if (ukfactor != 0.0)
      if (lexch == 8)
      {
// These are the PBE96 and revPBE98 functionals
// scale uk with a factor
         const double uk = uk1*ukfactor;
         const double ul = ul1/ukfactor;
         const double P0=1.+ul*S2;
         FxPBE = 1.+uk-uk/P0;
         expbe = exlda*FxPBE;
//----------------------------------------------------------------------
//  ENERGY DONE. NOW THE POTENTIAL:
//  find first derivatives of Fx w.r.t s.
//         Fs=(1/s)*d FxPBE/ ds
         Fs=2.*um/(P0*P0);
      }
      else if(lexch == 9)
      {
// This is the RPBE functional [Hammer et al, PRB 59, 7413 (1999)]
         const double P0=exp(-ul1*S2);
         FxPBE = 1.+uk1*(1.0-P0);
         expbe = exlda*FxPBE;
//----------------------------------------------------------------------
//  ENERGY DONE. NOW THE POTENTIAL:
//  find first derivatives of Fx w.r.t s.
//  Fs=(1/s)*d FxPBE/ ds
         Fs=2.*um*P0;
      }
      else
      {
         //This is not implemented on the GPU
      }
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// calculate the partial derivatives of ex wrt n and |grad(n)|
//  0.3232409194=(3*pi^2)^(-1/3)
      exd =exunif*4./3.*(FxPBE-S2*Fs);
      exdd=0.5*AX*0.3232409194*s*Fs;
}

template <int IFLG, bool TREL>
__device__ __forceinline__ double gpu_EX(const double RS)
{
    const double CX[2] = {0.9163305865663, 1.1545041946774};
    const double CBETA = 0.0140;
    double ret=-CX[IFLG-1]/RS;
    if (TREL)
    {
         const double B=CBETA/RS;
         double F;
         if (B >= 1.e-5)
         {
         F=log(B+(sqrt(1+B*B)))/(B*B);
         F=(sqrt(1+B*B)/B)-F;
         }
         else
         {
//jF: the expression given above becomes numerically extremely instable for
//    very small values of B (small difference of two large numbers divided 
//    by small number = noise) therefore use following for reasons of safety:
         F=B*(2./3.-0.7*B*B);
         }
         ret=(1.-1.5*F*F)*ret;
     }
    return ret;
}

template<int IFLG>
__device__ __forceinline__ double gpu_EX_SR(const double RS, const double RMU)
{
      const double CX[2] = {0.9163305865663,1.1545041946774};
      const double EX_=-CX[IFLG-1]/RS;
      double ret;
      double QFAC;
      if (IFLG==2)
      {
         QFAC=pow((6.*M_PI*M_PI),(1./3.));
      }
      else
      {
         QFAC=pow((3.*M_PI*M_PI),(1./3.));
      }
      const double RHO = 3/(4.*M_PI)/(RS*RS*RS);
      const double QF=QFAC*pow(RHO,(1./3.));
      const double A=RMU/2./QF;
//      FRAC=(1 - (8._q/3._q)*A*( SQRT(PI)*ERRF(1/(2*A)) + (2*A-4*A*A*A)*EXP(-1/(4*A*A)) &
//     &        -3*A + 4*A*A*A ))
//      ret=EX_*FRAC
// IG Simple formula replaced by 

// Test on the value of A
      if (A < 1E-9)
      {
// Limit for small A
         ret=EX_;
      }
      else if (A <= 100.)
      {
// Intermediate Values of A
         const double FRAC=(1.0 - (8./3.)*A*( sqrt(M_PI)*erf(1./(2.*A)) + (2.*A-4.*A*A*A)*exp(-1./(4.*A*A))-3.*A + 4.*A*A*A ));
         ret=EX_*FRAC;
      }
      else if (A <= 1.E+9)
      {
// Development for large A
         const double FRAC=1./36./A/A;
         ret=EX_*FRAC;
      }
      else
         ret=0.;
      
      return ret;
}

template <int IFLG, bool TREL>
__device__ __forceinline__ double gpu_VX(const double RS)
{
      const double CX[2] = {1.2217741154217,1.5393389262365};
      const double CBETA = 0.0140;
      double ret;
      ret=-CX[IFLG-1]/RS;
      if (TREL)
      {
// Warning error in the paper of Bachelet et al. !!
         const double B=CBETA/RS;
         if(B >= 1.E-5)
         {
            const double F=log(B+(sqrt(1+B*B)))/B/sqrt(1+(B*B));
         ret=(-0.5+1.5*F)*ret;
         }
         else
         {
            const double F=1.-B*B*(2./3.-B*B*31./30.);
         ret=(-0.5+1.5*F)*ret;
         }
      }

      return ret;
}

template <int IFLG>
__device__ __forceinline__ double gpu_VX_SR(const double RS, const double RMU)
{
       const double CX[2] = {0.9163305865663,1.1545041946774};
       const double VX_=-CX[IFLG-1]/RS;
        double QFAC;
      if (IFLG==2) 
         QFAC=pow((6.*M_PI*M_PI),(1./3.));
      else
         QFAC=pow((3.*M_PI*M_PI),(1./3.));

      const double RHO = 3/(4.*M_PI)/(RS*RS*RS);
      const double QF=QFAC*pow(RHO,(1./3.));
      const double A=RMU/2./QF;

//      FRAC= 1._q - 32*A*A*A*A*EXP(-1/(4*A*A)) -8._q*A*A + (64._q/3._q)*A*A*A*A
//      VX_SR=EX_SR(RS,RMU,IFLG)+(1._q/3._q)*VX_*FRAC
// IG Simple formula replaced by 

// Test on the value of A
      if (A < 1E-9)
      {
// Limit for small A
         return (4./3.)*VX_;
      }
      else if (A <= 100.)
      {
// Intermediate Values of A
         const double FRAC= (1. - 32.*A*A*A*A*(exp(-1./(4.*A*A))-1.) -8.*A*A);
         return gpu_EX_SR<IFLG>(RS,RMU)+(1./3.)*VX_*FRAC;
      }
      else if (A <= 1.E+9)
      {
// Development for large A
         return -RMU/48./M_PI/A/A/A;
      }
      else
         return 0.;
}

template <int N>
__device__ __forceinline__ double gpu_EXPINT(const double X)
{
      const double EPS=1E-9,FPMIN=1E-30;
      const int MAXIT=100;
      const double EULER=0.577215664901532860606512;
     
      const int NM1=N-1;
      if (N==0)
         return exp(-X)/X;
      else if (X==0.0)
         return 1./NM1;
      else if (X>1.0)
      {
         double B=X+N;
         double C=1./FPMIN;
         double D=1.0/B;
         double H=D;
#pragma unroll
         for(int I=1; I<=MAXIT; ++I)
         {
            const double A=-I*(NM1+I);
            B=B+2.;
            D=1.0/(A*D+B);
            C=B+A/C;
            const double DEL=C*D;
            H=H*DEL;
            if (fabs(DEL-1.)< EPS)
               return H*exp(-X);
         }
         return H*exp(-X);
      }
      else
      {
         double ret;
         if (NM1!=0)
            ret = 1.0/NM1;
         else
            ret=-log(X)-EULER;
         
         double FACT=1.;
         for(int I=1; I <= MAXIT; ++I)
         {
            FACT=-FACT*X/I;
            double DEL;
            if (I!=NM1)
               DEL=-FACT/(I-NM1);
            else
            {
               double PSI=-EULER;
#pragma unroll
               for(int II=1; II <= NM1; ++II)
                  PSI=PSI+1./II;
               DEL=FACT*(log(X)+PSI);
            }
            ret=ret+DEL;
            if (fabs(DEL)<fabs(ret)*EPS) return ret;
         }
         return ret;
      }

}
__device__ __forceinline__ void gpu_EXCHWPBE_R(const double OMEGA, const double RHO, const double SW, double & FXWPBE_SR)
{
      const double F13=1./3.,F12=0.5,F14=0.25;
      const double F32=1.5,F34=0.75,F94=2.25,F98=1.125,F1516=0.9375;
// Constants  from the PBE hole      
      const double A=1.0161144,B=-0.37170836;
      const double C=-0.077215461,D=0.57786348;
      const double E=-0.051955731;
// Values for H(s)
      const double HA1=0.00979681,HA2=0.0410834,HA3=0.187440;
      const double HA4=0.00120824,HA5=0.0347188;
// Values for F(s)
      const double FC1=6.4753871,FC2=0.47965830;
// Coefficients of the erfc(x) expansion (eb1 set later depending on wcut)
      const double EA1=-1.128223946706117,EA2=1.452736265762971;
      const double EA3=-1.243162299390327,EA4=0.971824836115601;
      const double EA5=-0.568861079687373,EA6=0.246880514820192;
      const double EA7=-0.065032363850763,EA8=0.008401793031216;
      const double WCUT=14.0;
// Constants for polynomial expansion of EG for small s
      const double EGSCUT=0.08,EGA1=-0.02628417880,EGA2=-0.07117647788;
      const double EGA3=0.08534541323;
// Constants for large x in exp(x)*Ei(x)
      const double EXPCUT=700,EXEI1=4.03640,EXEI2=1.15198;
      const double EXEI3=5.03627,EXEI4=4.19160;
      
// General constants
      const double PI2=M_PI*M_PI;
      const double SRPI=sqrt(M_PI);
      const double F89M=-8./9.;

//----------------------------------------------------------------------
//construct modified-PBE enhancement factor
//----------------------------------------------------------------------
//     INTERMEDIATE VARIABLES

// Calculate prelim variables
      const double XKF=pow((3.*PI2*RHO),F13);
      const double A2=A*A, A3=A2*A, A12=sqrt(A), A32=A12*A, A52=A32*A;
      const double W=OMEGA/XKF, W2= W*W, W3=W2*W, W4=W2*W2, W5=W2*W3, W6=W3*W3;
      const double W7=W6*W, W8=W7*W;     

      const double S2=SW*SW;
      const double S4=S2*S2;
      const double S5=S4*SW;
      const double S6=S4*S2;

// Calculate H(s) and F(s) for the PBE hole
      const double HNUM=HA1*S2+HA2*S4;
      const double HDEN=1.0+HA3*S4+HA4*S5+HA5*S6;
      const double H=(HNUM)/(HDEN);
      const double F=FC1*H+FC2;
      
// Set exponent of the Gaussian in the approximation of the erfc function
      double EB1;
      if (W<WCUT)
      {
         EB1= 1.455915450052607;
      }
      else
      {
         EB1=2.0;
      }

// Calculate intermediate variables
      const double HSBW=S2*H+EB1*W2; 
      const double HSBW2=HSBW*HSBW; 
      const double HSBW3=HSBW2*HSBW;
      const double HSBW12=sqrt(HSBW); 
      const double HSBW32=HSBW12*HSBW; 
      const double HSBW52=HSBW32*HSBW;
    
      const double DHSB=D+S2*H+EB1*W2; 
      const double DHSB2=DHSB*DHSB; 
      const double DHSB3=DHSB2*DHSB;
      const double DHSB4=DHSB2*DHSB2; 
      const double DHSB12=sqrt(DHSB); 
      const double DHSB32=DHSB12*DHSB; 
      const double DHSB52=DHSB32*DHSB;
      const double DHSB72=DHSB52*DHSB; 
      
      const double HA94=F94*HSBW/A; 
      const double HA942=HA94*HA94; 
      const double HA943=HA942*HA94;
      const double HA945=HA943*HA942; 
      const double HA9412=sqrt(HA94);

      const double DHS=D+S2*H; 
      const double DHS2=DHS*DHS; 
      const double DHS3=DHS2*DHS; 
      const double DHS72=DHS3*sqrt(DHS); 

      const double DHSW=DHS +W2; 
      const double DHSW2=DHSW*DHSW; 
      const double DHSW52=sqrt(DHSW)*DHSW2;

//Calculate G(s) using expansion for small s if necessary
      double EG;
      if (SW>EGSCUT)
      {
         const double GA=SRPI*(15.*E+6.0*C*(1.0+F*S2)*DHS + 4.0*B*(DHS2) +8.0*A*(DHS3)) * (1.0/(16.*DHS72))
                         -F34*M_PI*sqrt(A)*exp(F94*H*S2/A)* (1.-erf(F32*SW*sqrt(H/A)));
         const double GB=F1516*SRPI*S2/DHS72;
         EG=-(F34*M_PI+GA)/GB;
      }
      else    
         EG=EGA1+EGA2*S2+EGA3*S4;
// calculate the terms needed in any case

      const double TM2=(DHS2*B + DHS*C +2.*E +DHS*S2*C*F +2.*S2*EG )/2./DHS3;
      const double TM3=-W*(4.*DHSW2*B +6.*DHSW*C + 15.*E + 6.0*DHSW*S2*C*F + 15.*S2*EG)/8./DHS/DHSW52;
      const double TM4=-W3*(DHSW*C + 5.*E + DHSW*S2*C*F + 5.0*S2*EG)/2./DHS2/DHSW52;
      const double TM5=-W5*(E+S2*EG)/DHS3/DHSW52; 
     
// Calculate t10 unless that would generate a division by zero
      double T10;
      if ((SW>0.0) || (W>0.0))
         T10=F12*A*log(HSBW/DHSB);
      
// Calculate exp(x)*f(x) depending on the size of x
      double EXER, EXEI;
      if (HA94<EXPCUT)
      {
         EXER=M_PI*exp(HA94)*(erfc(HA9412));
         const double EXHA94=exp(HA94);
         const double EIHA94=-gpu_EXPINT<1>(HA94);
         EXEI=EXHA94*EIHA94;
      }
      else
      {
         EXER=M_PI*(1./(SRPI*HA9412)-1./(2.*sqrt(M_PI*HA943))+ 3./(4.*sqrt(M_PI*HA945)));
         EXEI=-(1./HA94)*(HA942+EXEI1*HA94+EXEI2)/(HA942+EXEI3*HA94+EXEI4);      
      }  
      if (W==0.0)
      {
//Fall back to the PBE hole expression
         const double T1=-F12*A*EXEI;
         if (SW>0.0)
         {
            const double TM1=T1+T10;
            FXWPBE_SR=F89M*(TM1+TM2);
         }
         else
            FXWPBE_SR=1.;
         
      }
      else if(W>WCUT)
      {
// Use simple gaussian approximation for large w
         const double TM1=-F12*A*(EXEI+log(DHSB)-log(HSBW));
         FXWPBE_SR=F89M*(TM1+TM2+TM3+TM4+TM5);
      }
      else
      {
// For everything else use the full blown expression
//
// First calculate the polynomials for the first term                  
         const double PN1=-F32*EA1*A12*W + 27.*EA3*W3/(8.*A12)-243.*EA5*W5/(32.*A32) + 2187.*EA7*W7/(128.*A52);
         const double PN2=-A + F94*EA2*W2 - 81.*EA4*W4/(16.0*A) + 729.*EA6*W6/(64.*A2) - 6561.*EA8*W8/(256.*A3);
             
// The first term is 
         const double T1=F12*(PN1*EXER+PN2*EXEI);     
// The factors for the main polynomials in w
         const double F2= F12*EA1*SRPI*A/DHSB12;
         const double F3= F12*EA2*A/DHSB;
         const double F4= EA3*SRPI*(-F98/HSBW12+F14*A/DHSB32);
         const double F5= EA4*(1./128.)*(-144.*(1./HSBW)+64.*(1./DHSB2)*A);
         const double F6= EA5*(3.*SRPI*(3.*DHSB52*(9.0*HSBW-2.*A)+4.0*HSBW32*A2))/(32.*DHSB52*HSBW32*A);
         const double F7= EA6*(((32.*A)/DHSB3 + (-36.+(81.*S2*H)/A)/HSBW2))/32.;
         const double F8= EA7*(-3.*SRPI*(-40.*HSBW52*A3+9.0*DHSB72*(27.*HSBW2-6.0*HSBW*A+4.*A2)))/(128.*DHSB72*HSBW52*A2);
         const double F9= (324.*EA6*EB1*DHSB4*HSBW*A + EA8*(384.*HSBW3*A3+DHSB4*(-729.*HSBW2+324.*HSBW*A-288.*A2)))/(128.*DHSB4*HSBW3*A2);
  
         const double T2T9= F2*W+F3*W2+F4*W3+F5*W4+F6*W5+F7*W6+F8*W7+F9*W8;

// The final value of the first term for 0<omega<wcut is
         const double TM1= T1+ T2T9 +T10; 
         FXWPBE_SR=F89M*(TM1+TM2+TM3+TM4+TM5);
      }

}

__device__ __forceinline__ double gpu_SIGN(double x, double y)
{
    if(y >= 0) return fabs(x);
    else return -fabs(x);
}

__device__ __forceinline__ void gpu_WPBE_SPLIN2(const double *X, const double *Y, const int N,
                                                const double YP1,const double YPN, double *Y2)
{    
  //int I,K;
  double P,QN,SIG,UN;
  double U[6]; //NOTE do not call gpu_WPBE_SPLIN2 with N > 6

// First point
      if (YP1 > 0.99E30){
         Y2[1-1]=0;
         U[1-1]=0;
      }
      else {
//        Y2[1-1] =-0.5;
         Y2[1-1] = 0.5;
          U[1-1] = (3./(X[2-1]-X[1-1]))*((Y[2-1]-Y[1-1])/(X[2-1]-X[1-1])-YP1);
      }

// Decomposition loop for the tridiagonal alg
      for(int I=2; I <= N-1; I++)
      {
         SIG     = (X[I-1]-X[I-1-1])/(X[I-1+1]-X[I-1-1]);
         P       = SIG*Y2[I-1-1]+2.;
         Y2[I-1] = (SIG-1.)/P;
         U[I-1]  = (6.*((Y[I-1+1]-Y[I-1])/(X[I-1+1]-X[I-1])-(Y[I-1]-Y[I-1-1]) / (X[I-1]-X[I-1-1])) /(X[I-1+1]-X[I-1-1])-SIG*U[I-1-1])/P;
      }
// Last point
      if (YPN > 0.99E30){
         QN=0;
         UN=0;
      }
      else {      
         QN = 0.5;
         UN = (3./(X[N-1]-X[N-1-1]))*(YPN-(Y[N-1]-Y[N-1-1])/(X[N-1]-X[N-1-1]));
      }
      Y2[N-1] = (UN-QN*U[N-1-1])/(QN*Y2[N-1-1]+1.);
        // Backsubstitution loop of the tridiagonal alg
      for(int K=N-1; K >= 1; K--)
      {
        Y2[K-1] = Y2[K-1]*Y2[K+1-1]+U[K-1];
      }
      
      return;
}


//NOTE: The NRHO and NS values should match those defined in xclib_grad.F
#define NRHO 2000
#define NS   2000
 

#define  LOGRHO(I) ((MINLOGRHO+(I-1)*(MAXLOGRHO-MINLOGRHO)/(NRHO-1)))
#define  S(I) ((MINS+(I-1)*(MAXS-MINS)/(NS-1)))     
#define SPLIDX(x,y) ((y-1)*NS + (x-1))       
__device__ __forceinline__ void gpu_WPBE_SPLINE(double X1,double X2, double &Y,
                                                const double *FS_RS, 
						const double *SSPLINES_RHO_S,
						const double *SSPLINES_S_RHO,
                                                const  int LOGRHO0)
{      
      //double D1X1,D1X2;
      int    JLO,JHI,KLO,KHI,JTMP,KTMP,KMIN;

      double YYTMPX2[6],Y2X2[6],LOGRHOTMP[6];
      double YYTMPX1[6],Y2X1[6],STMP[6];
      double H,A,B;
      double D1HI1,D1LO2,DIFFX1,DIFFX2;
//      double XJ1_X_3_H;
      double YX2;

      //const int    NRHO=2000;
      //const int    NS=2000;
      const double STRANS=8.3;
      const double SMAX=8.5728844;
      const double SCONST=18.79622316;

      const double MAXLOGRHO= 22.5;
      const double MINLOGRHO=-33.0;
      const double MAXS=9.3;
      const double MINS= 0.0;
      const double LOGRHOSTEP=(MAXLOGRHO-MINLOGRHO)/(1.*(NRHO-1));
      const double SSTEP=MAXS/(1.*(NS-1));

      //const double LOGRHO0 = 1190; //TODO get this param from function call

      // Rescaled the S parameter to lie in the bounds:
      if (X2>STRANS) X2=SMAX-(SCONST/pow(X2,2));
      
      // Check whether X1 and X2 are within bounds
      if (X1<LOGRHO(1) || X1>LOGRHO(NRHO))
      {
        printf("WPBE_SPLINE: ERROR(1), LOG(RHO) out of bounds: %g %g %g \n", X1,LOGRHO(1),LOGRHO(NRHO));
      }
      if (X2<S(1)  || X2>S(NS))
      {
        printf("WPBE_SPLINE: ERROR(1), S out of bounds: %g %g %g %g \n", X2,S(1),S(NS),exp(X1));
      }
     
      // Find bracketing indices at the rho axis
      if (X1 >= 0) 
         JLO=((int)(X1/LOGRHOSTEP))+LOGRHO0;
      else
         JLO=(int)((abs((X1-MINLOGRHO)/LOGRHOSTEP)))+1;
      
      JHI=JLO+1;

      if ((LOGRHO(JLO) > X1) || (LOGRHO(JHI) < X1))
      {
         if ((LOGRHO(JLO) > X1))
            JLO=JLO-1;
         else
            JLO=JLO+1;
         JHI=JLO+1;
      }

      if ((LOGRHO(JLO) > X1) || (LOGRHO(JHI) < X1))
      {
         printf("WPBE_SPLINE: ERROR(2), LOG(RHO) out of bounds: %g %g %g %g %g %g \n", X1,LOGRHO(JLO),LOGRHO(JHI),JLO,JHI,LOGRHO0);
         return;
      }



      // Find bracketing indices at the S axis
      KLO=(int)(((X2/SSTEP)))+1;
      KHI=KLO+1;

      if (S(KLO) > X2)
      {
         KHI=KLO;
         KLO=KLO-1;
      }

      if ((S(KLO) > X2) || (S(KHI)-X2 < -1.0E-8))
      {
         printf("WPBE_SPLINE: ERROR(2), S out of bounds: %g %g %g %g %g \n",  X2,S(KLO),S(KHI),KLO,KHI);
         return;
         //if(S(KLO) > X2)
         //{
         //   KLO=NS-4;
         //   KHI=NS-3;
         //}
      }

      // Interpolate S on a section of the rows
      for(int J=1; J <= 6; J++)
      {
         // Calculate function value
         H=S(KHI)-S(KLO);
         A=(S(KHI)-X2)/H;
         B=(X2-S(KLO))/H; 

         JTMP=JLO-3+J;
         YYTMPX1[J-1]=A*FS_RS[SPLIDX(JTMP,KLO)]+B*FS_RS[SPLIDX(JTMP,KHI)]+ 
                      ((A*A*A-A)*SSPLINES_RHO_S[SPLIDX(JTMP,KLO)] +
                       (B*B*B-B)*SSPLINES_RHO_S[SPLIDX(JTMP,KHI)])*(H*H)/6.;
      }
//YYTMPX1 contains correct content

      // Calculate a spline from these 6 function values
      DIFFX1=LOGRHOSTEP;
      D1LO2 =((YYTMPX1[2-1]-YYTMPX1[1-1])/DIFFX1);
      D1HI1 =((YYTMPX1[6-1]-YYTMPX1[5-1])/DIFFX1);

      for(int J=1; J <= 6; J++){
        LOGRHOTMP[J-1]=LOGRHO(JLO-3+J);
      }
      gpu_WPBE_SPLIN2(LOGRHOTMP,YYTMPX1,6,D1LO2,D1HI1,Y2X1);
 
      // Now do the same thing along the other dimension. Take care of X2 close to zero
      if (KLO < 3) {
         KMIN=2;
         YYTMPX2[1-1]=1.;
         if (KLO  < 2) KMIN=3;         
      }
      else {
         KMIN=1;
      }

      for(int K=KMIN; K <= 6; K++)
      {
         H=LOGRHO(JHI)-LOGRHO(JLO);
         
	 A=(LOGRHO(JHI)-X1)/H;
         B=(X1-LOGRHO(JLO))/H;

         KTMP=KLO-3+K;
         YYTMPX2[K-1]=A*FS_RS[SPLIDX(JLO,KTMP)]+B*FS_RS[SPLIDX(JHI,KTMP)]+ 
                      ((A*A*A-A)*SSPLINES_S_RHO[SPLIDX(KTMP,JLO)]+ 
                       (B*B*B-B)*SSPLINES_S_RHO[SPLIDX(KTMP,JHI)])*(H*H)/6.;
     }


      if (KMIN > 1)
      {
         if (KMIN == 2) 
         {
            YYTMPX2[1-1]=YYTMPX2[3-1];
         }
         if (KMIN == 3)
         {
            YYTMPX2[2-1]=YYTMPX2[4-1];
            YYTMPX2[1-1]=YYTMPX2[5-1];
          }         
      }

      // Calculate a spline from these 6 function values
      DIFFX2=SSTEP;
      D1LO2 =((YYTMPX2[2-1]-YYTMPX2[1-1])/DIFFX2);
      D1HI1 =((YYTMPX2[6-1]-YYTMPX2[5-1])/DIFFX2);

      if (KMIN == 1)
      {
          for(int K=1; K <=6; K++) STMP[K-1]=S(KLO-3+K);
          gpu_WPBE_SPLIN2(STMP,YYTMPX2,6,D1LO2,D1HI1,Y2X2);
      }
      else
      {
         if (KLO == 2) STMP[1-1]=S(KLO-1)-SSTEP;

         if (KLO == 1) {
            STMP[1-1]=S(KLO)-2.*SSTEP;
            STMP[2-1]=S(KLO)-SSTEP;
         }
         for(int K=1; K <=6; K++)  STMP[K-1]=S(KLO-3+K);

	 gpu_WPBE_SPLIN2(STMP,YYTMPX2,6,D1LO2,D1HI1,Y2X2);
      }

      // Interpolate the first spline for X1 (LOGRHO)
      H=LOGRHOSTEP;
      // Derivative in X1 direction
      //D1A = -1./H;
//      D1B = -D1A;
      //H13 = pow((H*H*H),(-1./3.));

  //    XJ1_X_3_H = (LOGRHO(JHI)-X1)*(LOGRHO(JHI)-X1)*3.*H13;
      //X_XJ_3_H  = (X1-LOGRHO(JLO))*(X1-LOGRHO(JLO))*3.*H13;
      // dF/dRHO
      //      D1X1=(D1A*YYTMPX1(3)+D1B*YYTMPX1(4)+ &
      //     &      (H*H/6._q)*((-XJ1_X_3_H-D1A)*Y2X1(3)+ &
      //     &      (X_XJ_3_H-D1B)*Y2X1(4)))

      // Interpolate the second spline for X2 (S)
      H=SSTEP;
      A=(S(KHI)-X2)/H ;
      B=(X2-S(KLO))/H ;



      // F(rho,s)
      YX2=A*YYTMPX2[3-1]+B*YYTMPX2[4-1]+ ((A*A*A-A)*Y2X2[3-1]+(B*B*B-B)*Y2X2[4-1])*(H*H)/6.;

      // Derivative in X2 direction
//      D1A = -1./H;
 //     D1B = -D1A;
  //    H13 = pow((H*H*H),(-1./3.));

//      XJ1_X_3_H = (S(KHI)-X2)*(S(KHI)-X2)*3.*H13;
      //X_XJ_3_H  = (X2-S(KLO))*(X2-S(KLO))*3.*H13;
// dF/dS
//      D1X2=(D1A*YYTMPX2(3)+D1B*YYTMPX2(4)+ &
//     &      (H*H/6._q)*((-XJ1_X_3_H-D1A)*Y2X2(3)+ &
//     &      (X_XJ_3_H-D1B)*Y2X2(4)))

      Y=YX2;

//     IF (Y.LT.0._q) THEN
//        WRITE(*,*) 'WPBE_SPLINE: Y<1',X1,X2,Y
//     ENDIF
      return;
}








//Note in this function all calls to gpu_EXCHWPBE_R have been replaced 
//with calls to WPBE_SPLINE. This gives slightly different results 
//on a per value basis but the same results in the overall result
__device__ __forceinline__ void gpu_calc_exchwpbe_sp_spline(const double D,const double S, double & EXLDA,double & EXDLDA,double & EXLDA_SR,double & EXLDA_LR,double & EXDLDA_SR,double & EXDLDA_LR, double & EXWPBE_SR, double & EXWPBE_LR,double & EXWPBED_SR,double & EXWPBED_LR,double & EXWPBEDD_SR,double & EXWPBEDD_LR, const double LDASCREEN, const double AUTOA,
const double *fs_rs, const double *ssplines_rho_s, const double *ssplines_s_rho, const int LOGRHO0)
{
      const double AX=-0.738558766382022405884230032680836;
      const double UM=0.2195149727645171;
      const double UK=0.8040;
      const double UL=UM/UK; 
    
      const double SW=fabs(S);
      const double XKF=pow((3.*M_PI*M_PI*D),(1./3.));

      const double RS=pow((3./(4.*M_PI)/D),(1./3.));
      EXLDA=gpu_EX<1,false>(RS)*D/2.;
      EXLDA_SR=gpu_EX_SR<1>(RS,LDASCREEN*AUTOA)*D/2.;
      EXLDA_LR=EXLDA-EXLDA_SR;
      EXDLDA=gpu_VX<1,false>(RS)/2.;
      EXDLDA_SR=gpu_VX_SR<1>(RS,LDASCREEN*AUTOA)/2;
      EXDLDA_LR=gpu_VX<1,false>(RS)/2.-EXDLDA_SR;

// PBE quantities
      const double F=1.+UK-UK/(1.+UL*SW*SW);
      const double FS=2.*UM/(1.+UL*SW*SW)/(1.+UL*SW*SW);
      const double EXPBED=4./3.*(F-SW*SW*FS)*EXLDA/D;
      const double EXPBEDD=0.5*AX*0.3232409194*SW*FS;
// set some stuff to zero      
      EXWPBE_SR=0;
      EXWPBE_LR=0;
      EXWPBED_SR=0;
      EXWPBED_LR=0;
      EXWPBEDD_SR=0;
      EXWPBEDD_LR=0;
// Interpolate the short range part of the enhancement
// coefficient from the tables
      if (SW==0. || D<0.) return;
      
      double LOGRHO=log(D);

//    Cutoff criterion to enforce local Lieb-Oxford
//    this ensures that the enhancement factor does not exceed the
//    original one
          
      const double DD=SW*(2.*XKF*D);
      double FSR;

      gpu_WPBE_SPLINE(LOGRHO,SW,FSR,  fs_rs, ssplines_rho_s, ssplines_s_rho,LOGRHO0);
// get values from function directly (slower)
//      gpu_EXCHWPBE_R(LDASCREEN*AUTOA,D,SW,FSR);
// Numerical derivatives with interpolated values: 
// Firstly: derivatives w.r.t rho
      const double DMIN=D-0.0001*D;
      LOGRHO=log(DMIN);
      const double XKFMIN=pow(3.*M_PI*M_PI*DMIN,1./3.);
      double SMIN=DD/(DMIN*XKFMIN*2.);
      double FSR_MIN;
      //gpu_EXCHWPBE_R(LDASCREEN*AUTOA, DMIN, SMIN, FSR_MIN);
      gpu_WPBE_SPLINE(LOGRHO,SMIN,FSR_MIN,  fs_rs, ssplines_rho_s, ssplines_s_rho,LOGRHO0);

      const double DPLS=D+0.0001*D;
      LOGRHO=log(DPLS);
      const double XKFPLS=pow((3.*M_PI*M_PI*DPLS),(1./3.));
      double SPLS=DD/(DPLS*XKFPLS*2.);
      
      double FSR_PLS;
//      gpu_EXCHWPBE_R(LDASCREEN*AUTOA, DPLS, SPLS, FSR_PLS);
      gpu_WPBE_SPLINE(LOGRHO,SPLS,FSR_PLS, fs_rs, ssplines_rho_s, ssplines_s_rho,LOGRHO0);

      const double DFDR=(FSR_PLS-FSR_MIN)/2./D/0.0001;
// Secondly: derivatives w.r.t gradient(rho)
      LOGRHO=log(D);
      const double DDMIN=DD-0.0001*DD;
      const double DDPLS=DD+0.0001*DD;
      SMIN=DDMIN/(D*XKF*2.);
      SPLS=DDPLS/(D*XKF*2.);
      
//      CALL WPBE_SPLINE(LOGRHO,SMIN,FSR_MIN)
//      gpu_EXCHWPBE_R(LDASCREEN*AUTOA, D, SMIN, FSR_MIN);
//      CALL WPBE_SPLINE(LOGRHO,SPLS,FSR_PLS) 
//      gpu_EXCHWPBE_R(LDASCREEN*AUTOA,D, SPLS, FSR_PLS);
      gpu_WPBE_SPLINE(LOGRHO,SMIN,FSR_MIN, fs_rs, ssplines_rho_s, ssplines_s_rho,LOGRHO0);
      gpu_WPBE_SPLINE(LOGRHO,SPLS,FSR_PLS, fs_rs, ssplines_rho_s, ssplines_s_rho,LOGRHO0); 
      const double DFDDD=(FSR_PLS-FSR_MIN)/2./DD/0.0001;

// Find the complementary long range part
      const double FLR=F-FSR;
// E_sr = fsr*Exlda         
      EXWPBE_SR=FSR*EXLDA;
      EXWPBE_LR=FLR*EXLDA;
// dE_sr/drho = dfsr/drho*Exlda + fsr*dExlda/drho
      EXWPBED_SR=DFDR*EXLDA+4./3.*FSR*EXLDA/D;
      EXWPBED_LR=EXPBED-EXWPBED_SR;

// dE_sr/d(grad rho) = dfsr/dgrad(rho)*Exlda
//gK correct the sign
      EXWPBEDD_SR=DFDDD*EXLDA *gpu_SIGN(1.0,S);
      EXWPBEDD_LR=(EXPBEDD-DFDDD*EXLDA) *gpu_SIGN(1.0,S);
}


//Note in this function all calls to WPBE_SPLINE have been replaced 
//with calls to gpu_EXCHWPBE_R. This gives slightly different results 
//on a per value basis but the same results in the overall result
__device__ __forceinline__ void gpu_calc_exchwpbe_sp(const double D,const double S, double & EXLDA,double & EXDLDA,double & EXLDA_SR,double & EXLDA_LR,double & EXDLDA_SR,double & EXDLDA_LR, double & EXWPBE_SR, double & EXWPBE_LR,double & EXWPBED_SR,double & EXWPBED_LR,double & EXWPBEDD_SR,double & EXWPBEDD_LR, const double LDASCREEN, const double AUTOA)
{
      const double AX=-0.738558766382022405884230032680836;
      const double UM=0.2195149727645171;
      const double UK=0.8040;
      const double UL=UM/UK; 
    
      const double SW=fabs(S);
      const double XKF=pow((3.*M_PI*M_PI*D),(1./3.));

      const double RS=pow((3./(4.*M_PI)/D),(1./3.));
      EXLDA=gpu_EX<1,false>(RS)*D/2.;
      EXLDA_SR=gpu_EX_SR<1>(RS,LDASCREEN*AUTOA)*D/2.;
      EXLDA_LR=EXLDA-EXLDA_SR;
      EXDLDA=gpu_VX<1,false>(RS)/2.;
      EXDLDA_SR=gpu_VX_SR<1>(RS,LDASCREEN*AUTOA)/2;
      EXDLDA_LR=gpu_VX<1,false>(RS)/2.-EXDLDA_SR;

// PBE quantities
      const double F=1.+UK-UK/(1.+UL*SW*SW);
      const double FS=2.*UM/(1.+UL*SW*SW)/(1.+UL*SW*SW);
      const double EXPBED=4./3.*(F-SW*SW*FS)*EXLDA/D;
      const double EXPBEDD=0.5*AX*0.3232409194*SW*FS;
// set some stuff to zero      
      EXWPBE_SR=0;
      EXWPBE_LR=0;
      EXWPBED_SR=0;
      EXWPBED_LR=0;
      EXWPBEDD_SR=0;
      EXWPBEDD_LR=0;
// Interpolate the short range part of the enhancement
// coefficient from the tables
      if (SW==0. || D<0.) return;
      
//      const double LOGRHO=log(D);
//    Cutoff criterion to enforce local Lieb-Oxford
//    this ensures that the enhancement factor does not exceed the
//    original one
          
      const double DD=SW*(2.*XKF*D);
      double FSR;
// TEST: not use spline      
//      CALL WPBE_SPLINE(LOGRHO,SW,FSR)
// get values from function directly (slower)

      gpu_EXCHWPBE_R(LDASCREEN*AUTOA,D,SW,FSR);
// Numerical derivatives with interpolated values: 
// Firstly: derivatives w.r.t rho
      const double DMIN=D-0.0001*D;
//      const double LOGRHO=log(DMIN);
      const double XKFMIN=pow(3.*M_PI*M_PI*DMIN,1./3.);
      double SMIN=DD/(DMIN*XKFMIN*2.);
//      CALL WPBE_SPLINE(LOGRHO,SMIN,FSR_MIN) 
      double FSR_MIN;
      gpu_EXCHWPBE_R(LDASCREEN*AUTOA, DMIN, SMIN, FSR_MIN);
      const double DPLS=D+0.0001*D;
//      LOGRHO=LOG(DPLS)
      const double XKFPLS=pow((3.*M_PI*M_PI*DPLS),(1./3.));
      double SPLS=DD/(DPLS*XKFPLS*2.);
      
//      CALL WPBE_SPLINE(LOGRHO,SPLS,FSR_PLS)
      double FSR_PLS;
      gpu_EXCHWPBE_R(LDASCREEN*AUTOA, DPLS, SPLS, FSR_PLS);
      const double DFDR=(FSR_PLS-FSR_MIN)/2./D/0.0001;
// Secondly: derivatives w.r.t gradient(rho)
//      LOGRHO=LOG(D)
      const double DDMIN=DD-0.0001*DD;
      const double DDPLS=DD+0.0001*DD;
      SMIN=DDMIN/(D*XKF*2.);
      SPLS=DDPLS/(D*XKF*2.);
      
//      CALL WPBE_SPLINE(LOGRHO,SMIN,FSR_MIN)
      gpu_EXCHWPBE_R(LDASCREEN*AUTOA, D, SMIN, FSR_MIN);
//      CALL WPBE_SPLINE(LOGRHO,SPLS,FSR_PLS) 
      gpu_EXCHWPBE_R(LDASCREEN*AUTOA,D, SPLS, FSR_PLS);
      const double DFDDD=(FSR_PLS-FSR_MIN)/2./DD/0.0001;

// Find the complementary long range part
      const double FLR=F-FSR;
// E_sr = fsr*Exlda         
      EXWPBE_SR=FSR*EXLDA;
      EXWPBE_LR=FLR*EXLDA;
// dE_sr/drho = dfsr/drho*Exlda + fsr*dExlda/drho
      EXWPBED_SR=DFDR*EXLDA+4./3.*FSR*EXLDA/D;
      EXWPBED_LR=EXPBED-EXWPBED_SR;
// dE_sr/d(grad rho) = dfsr/dgrad(rho)*Exlda
//gK correct the sign
      EXWPBEDD_SR=DFDDD*EXLDA *gpu_SIGN(1.0,S);
      EXWPBEDD_LR=(EXPBEDD-DFDDD*EXLDA) *gpu_SIGN(1.0,S);
   
}

__device__ __forceinline__ void gpu_gcor2(const double A,const double A1,const double B1,const double B2,const double B3,const double B4,const double rtrs,double & GG,double & GGRS)
{
      const double Q0 = -2.*A*(1.+A1*rtrs*rtrs);
      const double Q1 = 2.*A*rtrs*(B1+rtrs*(B2+rtrs*(B3+B4*rtrs)));
      const double Q2 = log(1.+1./Q1);
      GG = Q0*Q2;
      const double Q3 = A*(B1/rtrs+2.*B2+rtrs*(3.*B3+4.*B4*rtrs));
      GGRS = -2.*A*A1*Q2-Q0*Q3/(Q1*(1.+Q1));

}


template <bool lgga>
__device__ __forceinline__ void gpu_CORunspPBE(const double RS, double & EC,  double & VC, const double sk, const double T, double & H, double & DVC, double & ecdd)
{
      const double gamma=0.03109069086965489503494086371273;
      const double BET=0.06672455060314922,DELT=BET/gamma;
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// find LSD energy contributions, using [c](10) and Table I[c].
// EU=unpolarized LSD correlation energy
// EURS=dEU/drs
// EP=fully polarized LSD correlation energy
// EPRS=dEP/drs
// ALFM=-spin stiffness, [c](3).
// ALFRSM=-dalpha/drs
// F=spin-scaling factor from [c](9).
// construct ec, using [c](8)
      const double rtrs=sqrt(RS);
      double EU, EURS;
      gpu_gcor2(0.0310907,0.21370,7.5957,3.5876,1.6382, 0.49294,rtrs,EU,EURS);
      EC = EU;
// check for zero energy, immediate return if true
      if (EC==0.)
      {
        H=0; DVC=0; ecdd=0;
        return;
      }
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// LSD potential from [c](A1)
// ECRS = dEc/drs [c](A2)
// ECZET=dEc/dzeta [c](A3)
// FZ = dF/dzeta [c](A4)
      const double ECRS = EURS;
      VC = EC -RS*ECRS/3.;
      if (!lgga) return;
//----------------------------------------------------------------------
// PBE correlation energy
// G=phi(zeta), given after [a](3)
// DELT=bet/gamma
// B=A of [a](8)
      const double PON=-EC/(gamma);
      const double B = DELT/(exp(PON)-1.);
      const double B2 = B*B;
      const double T2 = T*T;
      const double T4 = T2*T2;
      const double Q4 = 1.+B*T2;
      const double Q5 = 1.+B*T2+B2*T4;
      H = (BET/DELT)*log(1.+DELT*Q4*T2/Q5);
//----------------------------------------------------------------------
//----------------------------------------------------------------------
// ENERGY DONE. NOW THE POTENTIAL, using appendix E of [b].
      const double T6 = T4*T2;
      const double RSTHRD = RS/3.;
      const double FAC = DELT/B+1.;
      const double BEC = B2*FAC/(BET);
      const double Q8 = Q5*Q5+DELT*Q4*Q5*T2;
      const double Q9 = 1.+2.*B*T2;
      const double hB = -BET*B*T6*(2.+B*T2)/Q8;
      const double hRS = -RSTHRD*hB*BEC*ECRS;
      const double hT = 2.*BET*Q9/Q8;
      DVC = H+hRS-7.0*T2*hT/6.;
      ecdd=0.5/sk*T*hT;

}


template <int is_real>
__global__ void cuggaallgrid(const int lexch, const double *dcharg, double *dworkg,
		double *dwork, cuDoubleComplex *dworkz, double *excl, const int np,
		const double autoa, const double autoa3, const double autoa4,
		const double rytoev, const double ldascreen, const int luse_thomas_fermi,
		const int force_pbe,const double AGGAC, const double AGGAX,
		const int LUSE_LONGRANGE_HF, const double *fs_rs,
		const double *ssplines_rho_s, const double *ssplines_s_rho,
		const double LOGRHO0)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    double exc = 0;
    if(tid < np)
    {
        const double rho = dcharg[tid];
        const double old_dworkg = max(dworkg[tid], 1e-10);
        const double d = rho*autoa3;
        const double dd = dworkg[tid]*autoa4;
        double excd, excdd;

        if(lexch == 7)
        {
            //PW91   
            gpu_gga91_wb(d, dd, exc, excd, excdd, AGGAC, AGGAX);
            exc = 2.*exc/d;
            excd = 2.*excd;
            excdd = 2.*excdd;
        }
        else if(lexch == 8 || lexch == 9)
        {
            //Perdew Burke Ernzerhof and revised functional
             double ukfactor;
            if (lexch==8)
                ukfactor=1.0;
            else
                ukfactor=0.0;
       
            if(d<=0)
            {
                exc   = 0.;
                excd  = 0.;
                excdd = 0.;
            }
            else
            {
                const double THRD = 1./3.;
                const double DTHRD=exp(log(d)*THRD);
                const double RS=pow((0.75/M_PI),THRD)/DTHRD;
                const double FK=pow((3.*M_PI*M_PI),THRD)*DTHRD;
                const double SK = sqrt(4.0*FK/M_PI);
                double s1, t1;
                if(d>1.E-10)
                {
                    s1=dd/(d*FK*2.);
                    t1=dd/(d*SK*2.);
                }
                else
                {
                    s1=0.0;
                    t1=0.0;
                }
                const double S = s1, T = t1;
                double EXLDA, EXDLDA, EXLDA_SR, EXLDA_LR, EXDLDA_SR, EXDLDA_LR, EXWPBE_SR, EXWPBE_LR, EXWPBED_SR, EXWPBED_LR, EXWPBEDD_SR, EXWPBEDD_LR;
                // Iann Gerber: range separating in GGA exchange
                if (ldascreen==0. || luse_thomas_fermi == 1 || force_pbe == 1)
		{
                    gpu_exchpbe(d,DTHRD,S,EXLDA,exc,EXDLDA,excd,excdd, ukfactor, lexch);
		}
                else
                {
		    // WARNING: The fs_rs and spline arrays in xclib_grad.F have never been copied to device memory because they were never initialized in the CPU... this has to be fixed at some point by porting the initialization function into the GPU!
		   // enters this path for B.hR12 benchmark...
		    //return;

                    #if 0
                      //This function does not use SPLINE calls
                      gpu_calc_exchwpbe_sp(d,S, EXLDA,EXDLDA,EXLDA_SR,EXLDA_LR,EXDLDA_SR,EXDLDA_LR, EXWPBE_SR,EXWPBE_LR,EXWPBED_SR,EXWPBED_LR,EXWPBEDD_SR,EXWPBEDD_LR, ldascreen, autoa);                 
                    #else
                      //This function DOES use SPLINEs    , 
                      gpu_calc_exchwpbe_sp_spline(d,S, EXLDA,EXDLDA,EXLDA_SR,EXLDA_LR,EXDLDA_SR,EXDLDA_LR, EXWPBE_SR,EXWPBE_LR,EXWPBED_SR,EXWPBED_LR,EXWPBEDD_SR,EXWPBEDD_LR, ldascreen, autoa, fs_rs, ssplines_rho_s, ssplines_s_rho, LOGRHO0);
                    #endif
                }



                // Iann Gerber: end modification
                double ECLDA, ECDLDA, EC, ECD, ECDD;
                gpu_CORunspPBE<true>(RS,ECLDA,ECDLDA,SK, T,EC,ECD,ECDD);

                        //        ! Do not add LDA contributions
                        if (ldascreen==0. || luse_thomas_fermi == 1 || force_pbe == 1)
                        {
                            exc =EC  *AGGAC +(exc-EXLDA)/d*AGGAX;
                            excd =ECD *AGGAC +(excd-EXDLDA)*AGGAX;
                            excdd=ECDD*AGGAC + excdd*AGGAX;
                        }
                        else
                        {
                            if (LUSE_LONGRANGE_HF)
                            {
                                //              ! Iann Gerber's functionals
                                exc  =EC  *AGGAC+(EXWPBE_LR-EXLDA_LR)/d*AGGAX  +(EXWPBE_SR-EXLDA_SR)/d;
                                excd =ECD *AGGAC+(EXWPBED_LR-EXDLDA_LR)*AGGAX+(EXWPBED_SR-EXDLDA_SR); 
                                excdd=ECDD*AGGAC+ EXWPBEDD_LR*AGGAX +EXWPBEDD_SR;         
                            }
                            else
                            {
                                //              ! wPBE
                                exc  =EC  *AGGAC+(EXWPBE_SR-EXLDA_SR)/d*AGGAX  +(EXWPBE_LR-EXLDA_LR)/d;
                                excd =ECD *AGGAC+(EXWPBED_SR-EXDLDA_SR)*AGGAX+(EXWPBED_LR-EXDLDA_LR);
                                excdd=ECDD*AGGAC+ EXWPBEDD_SR*AGGAX +EXWPBEDD_LR;
                            }
                        }

        // Hartree -> Rydberg conversion
        exc = 2*exc;
        excd =2*excd;
        excdd=2*excdd;

            }
           
        }
        else if(lexch == 11 || lexch == 12)
        {
        }
        else if(lexch == 13)
        {

        }
        else if(lexch == 14)
        {

        }
        else
        {

        }
        exc = exc*rho;
	if(is_real)
            dwork[tid] = excdd/old_dworkg*rytoev*autoa;
	else
	    dworkz[tid] = make_cuDoubleComplex(excdd/old_dworkg*rytoev*autoa,0.0);
        dworkg[tid] = excd*rytoev;
    }
    __shared__ double excl_s[256];
    excl_s[threadIdx.x] = exc;
    __syncthreads();
#pragma unroll
    for(int s = 128; s >= 1; s =s/2)
    {
        if(threadIdx.x < s)
        {
            excl_s[threadIdx.x] += excl_s[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
        excl[blockIdx.x] = excl_s[0];
}


//Used by xclib_grad.F
extern "C"
void cuda_ggaallgrid_C(const devptr_t * dcharg_p, devptr_t * dworkg_p, devptr_t * dwork_p, devptr_t * excl_p, const int * np_p, const double * autoa_p, const double * autoa3_p, const double * autoa4_p, const double * rytoev_p, const int * lexch_p, double * exc_ret, const double * ldascreen_p, const int * luse_thomas_fermi_p, const int * force_pbe_p, const double * aggac_p, const double * aggax_p, const int * use_longrange_hf_p,
 const devptr_t * fs_rs_p, const devptr_t * ssplines_rho_s_p, const devptr_t * ssplines_s_rho_p, const int * LOGRHO0_, const int *DWORK_IS_REAL)
{
    const double * dcharg = (const double *)(*dcharg_p);
    double * dworkg = (double*)(*dworkg_p);
    double * dwork = (double*)(*dwork_p);
    cuDoubleComplex * dworkz = (cuDoubleComplex *)(*dwork_p);
    double * excl = (double*)(*excl_p);
    const int np = *np_p;
    const double autoa = *autoa_p;
    const double autoa3 = *autoa3_p;
    const double autoa4 = *autoa4_p;
    const double rytoev = *rytoev_p;
    const int lexch = *lexch_p;
    const double ldascreen = *ldascreen_p;
    const int luse_thomas_fermi = *luse_thomas_fermi_p;
    const int force_pbe = *force_pbe_p;
    const double aggac = *aggac_p;
    const double aggax = *aggax_p;
    const int use_longrange_hf = *use_longrange_hf_p;
    const int LOGRHO0 = *LOGRHO0_;

    double * fs_rs          = (double*)(*fs_rs_p);
    double * ssplines_rho_s = (double*)(*ssplines_rho_s_p);
    double * ssplines_s_rho = (double*)(*ssplines_s_rho_p);


    const int nThreads = 256;
    int blocks = (np+nThreads - 1)/nThreads;

    if((lexch != 7) && (lexch != 8) && (lexch != 9))
    {
        printf("Something went really wrong! You should not be there! GGA GPU does not work (yet) with the value %d!\n", lexch);
        cudaDeviceReset();
        exit(0);
    }

    if(*DWORK_IS_REAL)
    {
        cuggaallgrid<true><<<blocks,nThreads>>>(lexch,dcharg,dworkg,dwork,dworkz,
	    excl,np,autoa,autoa3,autoa4,rytoev,ldascreen,luse_thomas_fermi,force_pbe,
	    aggac,aggax,use_longrange_hf,fs_rs,ssplines_rho_s,ssplines_s_rho,LOGRHO0);
    }
    else
    {
        cuggaallgrid<false><<<blocks,nThreads>>>(lexch,dcharg,dworkg,dwork,dworkz,excl,
	    np,autoa,autoa3,autoa4,rytoev,ldascreen,luse_thomas_fermi,force_pbe,aggac,
	    aggax,use_longrange_hf,fs_rs,ssplines_rho_s,ssplines_s_rho,LOGRHO0);
    }

    //cudaDeviceSynchronize();       
    cudaError_t error = cudaGetLastError();      
    if (error != cudaSuccess)   
    {
       printf( "Error: %s in %s:%d\n", cudaGetErrorString( error ), __FILE__, __LINE__);      
       exit(-1);      
    }
    
    
    double * excl_h = new double[blocks];
    cudaMemcpy(excl_h, excl, blocks*sizeof(double), cudaMemcpyDeviceToHost);
    *exc_ret = 0;
    for(int i = 0; i < blocks; i++)
        *exc_ret += excl_h[i];
    delete [] excl_h;
    
    error = cudaGetLastError();      
    if (error != cudaSuccess)   
    {
       printf( "Error: %s in %s:%d\n", cudaGetErrorString( error ), __FILE__, __LINE__);      
       exit(-1);      
    }    
}
 
