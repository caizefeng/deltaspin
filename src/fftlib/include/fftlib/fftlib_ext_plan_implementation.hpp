// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	ext_plan::ext_plan()
		: in(NULL),
		  out(NULL),
		  p_1(NULL),
		  p_2(NULL),
		  p_3(NULL),
		  buffer(NULL)	
	{
	}

	ext_plan::~ext_plan()
	{
	}
}
