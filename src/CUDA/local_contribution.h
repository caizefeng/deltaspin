/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Inspired by sdk */
/*
// USE_TEX 
#if (USE_TEX==1)
#undef fetchx
#undef fetchy
#undef fetchz
#define fetchx(i)  fetch_double(texX,parms.texXOfs+(i))
#define fetchy(i)  fetch_double(texY,parms.texYOfs+(i))
#define fetchz(i)  fetch_double(texY,parms.texYOfs+(i))
#else
#undef fetchx
#undef fetchy
#undef fetchz
#define fetchx(i)  W1_CR[i]
#define fetchy(i)  W2_CR[i]
#define fetchz(i)  SV[i]
#endif 
*/


    unsigned int i, n, tid, totalThreads, ctaStart, MM, MM_, ISPINOR, ISPINOR_, CSTE;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0f, 0.0f);
//#if (USE_TEX==0)
    const cuDoubleComplex *W1_CR;
    const cuDoubleComplex *W2_CR;
    const double *SV;
    const cuDoubleComplex *SVz;
//#endif
    /* wrapper must ensure that parms.n > 0 */
    tid      = threadIdx.x;
    n        = parms.n;
    ISPINOR  = parms.ISPINOR;
    ISPINOR_ = parms.ISPINOR_;
    CSTE     = parms.CSTE;
//#if (USE_TEX==0)
    W1_CR = parms.W1_CR;
    W2_CR = parms.W2_CR;
    if (parms.is_real)
        SV = parms.SVd;
    else
        SVz = parms.SVz;
//#endif
    totalThreads = gridDim.x * ZDOT_THREAD_COUNT;
    ctaStart = ZDOT_THREAD_COUNT * blockIdx.x;

    for (i = ctaStart + tid; i < n; i += totalThreads) {
        MM  = i + ISPINOR  * CSTE;
        MM_ = i + ISPINOR_ * CSTE;
        //sum = sum + fetchz(i) * fetchx(MM_) * cuConj(fetchy(MM));
        if(parms.is_real)
          sum = sum + SV[i] * W1_CR[MM_] * cuConj(W2_CR[MM]);
        else
          sum = sum + SVz[i] * W1_CR[MM_] * cuConj(W2_CR[MM]);
    }

    partialSum[tid] = sum;

#if (ZDOT_THREAD_COUNT & (ZDOT_THREAD_COUNT - 1))
#error code requires ZDOT_THREAD_COUNT to be a power of 2
#endif
#pragma unroll
    for (i = ZDOT_THREAD_COUNT >> 1; i > 0; i >>= 1) {
        __syncthreads(); 
        if (tid < i) {
            partialSum[tid] = partialSum[tid] + partialSum[tid + i]; 
        }
    }
    if (tid == 0) {
	//We either assign the value or we sum it with the result
	//of a previous iteration
	if(assign)
	{
        	parms.result[blockIdx.x] = partialSum[tid];
	}
	else
	{
		cuDoubleComplex temp = parms.result[blockIdx.x];
		if(normalize)
		{
			//Normalize the current value 
			temp.x = temp.x / parms.normalizeValue;
			temp.y = temp.y / parms.normalizeValue;
		}
        	parms.result[blockIdx.x] = temp + partialSum[tid];
	}
    }
