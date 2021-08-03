// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_HPP)
#define FFTLIB_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <omp.h>
#include <fftw3.h>

#include "fftlib_macros.hpp"
#include "fftlib_constants.hpp"
#include "fftlib_reader_writer_lock.hpp"
#include "fftlib_types.hpp"
#include "fftlib_dynamic_lib.hpp"
#include "fftlib_configuration.hpp"
#include "fftlib_plan.hpp"
#include "fftlib_ext_plan.hpp"

namespace fftlib_internal
{
	static std::unordered_map<std::uint64_t, std::uint64_t> plans;

	static fftlib_internal::reader_writer_lock lock_plans;

	template<backend B = FFTW>
	class fft_base
	{
	public:
		static int init_threads();

		static void plan_with_nthreads(const typename data<B, INT>::type_t nthreads);

		static void destroy_plan(typename trafo_base<B>::plan_t p);

		static void cleanup_threads();

		static void cleanup();
	};
}

namespace fftlib
{
	template<backend B = FFTW, transformation T = C2C_64, std::int32_t D = 3>
	class fft : public fftlib_internal::fft_base<B>
	{	
		using plan = typename fftlib_internal::plan<B, T, D>;

		using configuration = typename fftlib_internal::configuration<B, T, D>;

		using key_hash = typename fftlib_internal::configuration<B, T, D>::hasher;

		using key_equal = typename fftlib_internal::configuration<B, T, D>::equal;

		using unordered_map = typename std::unordered_map<const configuration, plan, key_hash, key_equal>;

		using unordered_map_const_iterator = typename unordered_map::const_iterator;

	public:
		~fft();

		static fft& get_instance();

		typename trafo<B, T>::plan_t create_plan(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, typename trafo<B, T>::in_t* in, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, typename trafo<B, T>::out_t* out, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags); 
		static void execute_plan(const typename trafo<B, T>::plan_t p, typename trafo<B, T>::in_t* in = NULL, typename trafo<B, T>::out_t* out = NULL);

		static void info();

		void clear();

	private:
		fft();

		const typename trafo<B, T>::plan_t* find_plan(const configuration& c);

		const typename trafo<B, T>::plan_t* create_plan(const configuration& c, typename trafo<B, T>::in_t* in, typename trafo<B, T>::out_t* out);

	private:
		unordered_map plans;

		std::vector<std::pair<const configuration*, const plan*>*> cache;

		std::vector<std::int32_t> next_slot, slots_used;
		
		fftlib_internal::reader_writer_lock lock_map;
	};  
}

#include "fftlib_implementation.hpp"

#endif
