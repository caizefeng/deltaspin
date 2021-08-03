// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_EXT_PLAN_HPP)
#define FFTLIB_EXT_PLAN_HPP

namespace fftlib_internal
{
	// this class serves as an 'fftw_plan' replacement.
	// it holds the actual FFTW plans (p_1, p_2, p_3, e.g. for composed FFT computation)
	// and additional information to partially recover the configuration at the time
	// of the plan creation call (we reuse the same plan(s) multiple times, 
	// but potentially for different inputs).
	class ext_plan
	{
	public:
		ext_plan();

		~ext_plan();

	public:	
		int d;

		int n[3];

		int howmany;

		int idist;

		int odist;

		void* in;
		
		void* out;
	       	
		const void* p_1;

		const void* p_2;

		const void* p_3;
		
		// this buffer is used for composed FFT computations.
		void** buffer;

		fftlib::transformation t;
		
		fftlib::backend b;
		
		fftlib::scheme s;	
	};
}

#include "fftlib_ext_plan_implementation.hpp"

#endif
