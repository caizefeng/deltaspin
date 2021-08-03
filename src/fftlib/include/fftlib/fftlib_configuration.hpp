// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFLIB_CONFIGURATION_HPP)
#define FFLIB_CONFIGURATION_HPP

namespace fftlib_internal
{
	using namespace fftlib;

	template<backend B, transformation T, std::int32_t D>
	class plan;

	template<backend B, transformation T, std::int32_t D>
	class configuration
	{
		// class 'plan' should have access to this class' protected attributes.
		friend class plan<B, T, D>;

	public:
		// this class is needed by std::unordered_map<const configuration,...,hasher,...> in 'fftlib.hpp'.
		class hasher
		{
		public:
			std::size_t operator()(const configuration& c) const;
			
			static std::size_t compute(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags, const bool aligned, const bool in_place, const typename data<B, INT>::type_t nthreads);
		};

	public:
		// this class is needed by std::unordered_map<const configuration,...,...,equal> in 'fftlib.hpp'.
		class equal
		{	
		public:
			bool operator()(const configuration& c_1, const configuration& c_2) const;	
		};
	
	public:
		configuration(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, typename trafo<B, T>::in_t* in, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, typename trafo<B, T>::out_t* out, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags);
	
		bool operator==(const configuration& c) const;

	protected:
		typename data<B, INT>::type_t n[D], inembed[D], onembed[D];

		const typename data<B, INT>::type_t* inembed_ptr;

		const typename data<B, INT>::type_t* onembed_ptr;

		typename data<B, INT>::type_t sign, istride, ostride, idist, odist, howmany, nthreads;

		typename data<B, UINT>::type_t flags;

		bool aligned, in_place;

		std::size_t hash;
	};
}

#include "fftlib_configuration_implementation.hpp"

#endif
