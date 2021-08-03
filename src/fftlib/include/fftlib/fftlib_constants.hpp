// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_CONSTANTS_HPP)
#define FFTLIB_CONSTANTS_HPP

// alignment for memory allocation for SSE, AVX, AVX-512.
#if !defined(FFTLIB_ALIGNMENT)
	#if defined(__MIC__) || defined(__AVX512F__)
		#define FFTLIB_ALIGNMENT (64)
	#else
		#define FFTLIB_ALIGNMENT (32)
	#endif
#endif

// maximum number of cache entries.
#if !defined(CACHE_SIZE)
	#define FFTLIB_CACHE_SIZE (8)
#endif

#define FFTLIB_UNDEFINED_INT32 (0xFFFFFFFF)

#endif
