// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	using namespace fftlib;

	template<backend B, transformation T, std::int32_t D>
	configuration<B, T, D>::configuration(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, typename trafo<B, T>::in_t* in, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, typename trafo<B, T>::out_t* out, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags)
		: howmany(howmany),
		  istride(istride),
		  idist(idist),
		  ostride(ostride),
		  odist(odist),
  	  	  sign(sign),
		  flags(flags)
	{
		FFTLIB_INFO("configuration constructor (thread=" << omp_get_thread_num() << ")");

		// copy values of n[]
		for (std::size_t i = 0; i < D; ++i)
			this->n[i] = n[i];

		// store pointers 'inembed' and 'onembed': copy the values pointed to 
		// only if these pointers are different from NULL. otherwise, assign
		// FFTLIB_UNDEFINED_INT32.
		inembed_ptr = inembed;
		for (std::size_t i = 0; i < D; ++i)
			this->inembed[i] = (inembed != NULL ? inembed[i] : FFTLIB_UNDEFINED_INT32);
		
		onembed_ptr = onembed;
		for (std::size_t i = 0; i < D; ++i)
			this->onembed[i] = (onembed != NULL ? onembed[i] : FFTLIB_UNDEFINED_INT32);
	       
		// determine if pointers point to aligned memory or not, and
		// if the transformation is in-place or not.
		if (in != NULL && out == NULL)
			{
				aligned = ((reinterpret_cast<uint64_t>(in) % FFTLIB_ALIGNMENT) == 0);
				in_place = true;
			}
		else if (in != NULL && out != NULL)
			{
				if (reinterpret_cast<void*>(in) == reinterpret_cast<void*>(out))
					{			
						aligned = ((reinterpret_cast<uint64_t>(in) % FFTLIB_ALIGNMENT) == 0);
						in_place = true;	
					}
				else
					{
						aligned = ((reinterpret_cast<uint64_t>(in) % FFTLIB_ALIGNMENT) == 0) && ((reinterpret_cast<uint64_t>(out) % FFTLIB_ALIGNMENT) == 0);
						in_place = false;	
					}
			}
		else
			{
				aligned = false;
				in_place = false;
			}

		#if defined(FFTLIB_USE_MKL)
			// when using MKL's FFTW wrappers, the thread count at this point is immaterial and
			// not associated with the plan created for this configuration.
			// assign FFTLIB_UNDEFINED_INT32 in all cases.
			// the thread count is set when the plan is executed.
			nthreads = FFTLIB_UNDEFINED_INT32;
		#else
			// when using FFTW, the thread count at this point is relevant to the plan creation
			// and needs to be requested from 'dynamic_lib'.
			dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
			nthreads = dl.get_nthreads();
		#endif
		
		// the configuration itself stores its hash values that is used when looking up
		// pairs of <configuration,plan> in the internal map.
		hash = hasher::compute(this->n, this->howmany, this->inembed, this->istride, this->idist, this->onembed, this->ostride, this->odist, this->sign, this->flags, aligned, in_place, nthreads);		
	}
	
	template<backend B, transformation T, std::int32_t D>
	bool configuration<B, T, D>::operator==(const configuration& c) const
	{
		FFTLIB_INFO("configuration operator== (thread=" << omp_get_thread_num() << ")");

		if (hash != c.hash || sign != c.sign || nthreads != c.nthreads || howmany != c.howmany || aligned != c.aligned || in_place != c.in_place)
			return false;
		
		for (std::int32_t i = 0; i < D; ++i)
			if (n[i] != c.n[i])
				return false;

		for (std::int32_t i = 0; i < D; ++i)
			if (inembed[i] != c.inembed[i])
				return false;

		for (std::int32_t i = 0; i < D; ++i)
			if (onembed[i] != c.onembed[i])
				return false;	
		
		return (flags == c.flags && istride == c.istride && ostride == c.ostride && idist == c.idist && odist == c.odist);
	}
	
	template<backend B, transformation T, std::int32_t D>
	std::size_t configuration<B, T, D>::hasher::operator()(const configuration& c) const
	{
		FFTLIB_INFO("configuration::hasher operator() (thread=" << omp_get_thread_num() << ")");

		return c.hash;	
	}
	
	template<backend B, transformation T, std::int32_t D>
	std::size_t configuration<B, T, D>::hasher::compute(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags, const bool aligned, const bool in_place, const typename data<B, INT>::type_t nthreads)
	{
		FFTLIB_INFO("configuration::hasher compute (thread=" << omp_get_thread_num() << ")");

		std::size_t hash = 0x0;

		for (std::int32_t i = 0; i < D; ++i)
                        hash ^= ((i + 1) * n[i]);

		for (std::int32_t i = 0; i < D; ++i)
                        hash ^= ((i + 10) * inembed[i]);

		for (std::int32_t i = 0; i < D; ++i)
                        hash ^= ((i + 100) * onembed[i]);

                return (hash ^ sign ^ howmany ^ flags);
	}
	
	template<backend B, transformation T, std::int32_t D>
	bool configuration<B, T, D>::equal::operator()(const configuration& c_1, const configuration& c_2) const
	{
		FFTLIB_INFO("configuration::equal operator() (thread=" << omp_get_thread_num() << ")");

		return c_1 == c_2;
	}

}
