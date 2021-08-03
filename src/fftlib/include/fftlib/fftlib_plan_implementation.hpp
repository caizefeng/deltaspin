// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	using namespace fftlib;
       
	template<backend B, transformation T, std::int32_t D>
	plan<B, T, D>::plan(const configuration<B, T, D>& c, typename trafo<B, T>::in_t* in, typename trafo<B, T>::out_t* out)
	{
		dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
		// check if plan creation was sucessful.
		if ((p = dl.template create_plan<T>(D, c.n, c.howmany, in, c.inembed_ptr, c.istride, c.idist, out, c.onembed_ptr, c.ostride, c.odist, c.sign, c.flags)) == NULL)
			{
				FFTLIB_WARNING("plan constructor (thread=" << omp_get_thread_num() << ") failed to create plan");
			}
		else
			{
				FFTLIB_INFO("plan constructor (thread=" << omp_get_thread_num() << ") plan successfully created");
			}
	}
	
	template<backend B, transformation T, std::int32_t D>
	plan<B, T, D>::~plan()
	{
		FFTLIB_INFO("plan destructor (thread=" << omp_get_thread_num() << ")");

		// destroy plan only if it has been created sucessfully.
		if (p != NULL)
			{
				dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
				dl.destroy_plan(p);
			}
	}
}
