// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_PLAN_HPP)
#define FFTLIB_PLAN_HPP

namespace fftlib_internal
{ 
	using namespace fftlib;
	
	template<backend B, transformation T, std::int32_t D>
	class plan
	{
	public:
		plan(const configuration<B, T, D>& c, typename trafo<B, T>::in_t* in, typename trafo<B, T>::out_t* out);
		
		~plan();

	public:
		typename trafo<B, T>::plan_t p;	
	};
}

#include "fftlib_plan_implementation.hpp"

#endif
