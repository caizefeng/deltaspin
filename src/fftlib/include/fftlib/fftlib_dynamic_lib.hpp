// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_DYNAMIC_LIB_HPP)
#define FFTLIB_DYNAMIC_LIB_HPP

#include <string>
#include <map>
#include <vector>
#include <dlfcn.h>
#include <fftw3.h>

#if defined(FFTLIB_USE_MKL)
	#include <mkl.h>
#endif

namespace fftlib_internal
{
	class dynamic_loader
	{
		// this class implements the singleton pattern, that is,
		// only one instance of it can exist within the  process context.
	public:
		~dynamic_loader();

		static dynamic_loader& get_instance();

		void info() const;
	       
		std::int32_t get_symbol(const std::string& filename, void** f, const std::string& symbol);

		std::int32_t get_symbol(const char* filename, void** f, const char* symbol);

	private: 
		dynamic_loader();

		void* open(const std::string& filename);

		void* open(const char* filename);

	private:
		std::map<const std::string, void*> handles;
	};
}

namespace fftlib_internal
{
	using namespace fftlib;

	template<backend B>
	class dynamic_lib;

	template<>
	class dynamic_lib<DFTI>
	{
		// TODO: need to be implemented.
	};

	template<>
	class dynamic_lib<FFTW>
	{
		// this class implements the singleton pattern, that is,
		// only one instance of it can exist within the  process context.				
	public:
		~dynamic_lib();

		static dynamic_lib& get_instance();

		void info() const;

		typename data<FFTW, INT>::type_t init_threads(void);

		void plan_with_nthreads(typename data<FFTW, INT>::type_t nthreads);

		void cleanup_threads(void);
	       
		void cleanup(void) const;

		template<transformation T>
		typename trafo<FFTW, T>::plan_t create_plan(const typename data<FFTW, INT>::type_t d, const typename trafo<FFTW, T>::dim_t* n, const typename data<FFTW, INT>::type_t howmany, typename trafo<FFTW, T>::in_t* in, const typename trafo<FFTW, T>::dim_t* inembed, const typename data<FFTW, INT>::type_t istride, const typename data<FFTW, INT>::type_t idist, typename trafo<FFTW, T>::out_t* out, const typename trafo<FFTW, T>::dim_t* onembed, const typename data<FFTW, INT>::type_t ostride, const typename data<FFTW, INT>::type_t odist, const typename data<FFTW, INT>::type_t sign, const typename data<FFTW, UINT>::type_t flags);

		void execute_plan(const typename trafo_base<FFTW>::plan_t p) const;

		template<transformation T>
		void execute_plan(const typename trafo<FFTW, T>::plan_t p, typename trafo<FFTW, T>::in_t* in, typename trafo<FFTW, T>::out_t* out) const;

		void destroy_plan(const typename trafo_base<FFTW>::plan_t p) const;

		data<FFTW, INT>::type_t get_nthreads() const;
	       
	private:
		dynamic_lib();

	private:
		std::vector<data<FFTW, INT>::type_t> nthreads;

		bool plan_with_threads;

		bool initialize_threading;

		data<FFTW, INT>::type_t max_threads;

		typename data<FFTW, INT>::type_t (*xxx_init_threads)(void);

		void (*xxx_plan_with_nthreads)(const typename data<FFTW, INT>::type_t);

		void (*xxx_cleanup_threads)(void);

		void (*xxx_cleanup)(void);

		void (*xxx_execute)(const typename trafo_base<FFTW>::plan_t);

	        void (*xxx_execute_dft)(const typename trafo_base<FFTW>::plan_t, typename trafo<FFTW, C2C_64>::in_t*, typename trafo<FFTW, C2C_64>::out_t*);

		void (*xxx_execute_dft_r2c)(const typename trafo_base<FFTW>::plan_t, typename trafo<FFTW, R2C_64>::in_t*, typename trafo<FFTW, R2C_64>::out_t*);

		void (*xxx_execute_dft_c2r)(const typename trafo_base<FFTW>::plan_t, typename trafo<FFTW, C2R_64>::in_t*, typename trafo<FFTW, C2R_64>::out_t*);

		trafo_base<FFTW>::plan_t (*xxx_plan_many_dft)(data<FFTW, INT>::type_t, const typename trafo<FFTW, C2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, C2C_64>::in_t*, const typename trafo<FFTW, C2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, C2C_64>::out_t*, const typename trafo<FFTW, C2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, const typename data<FFTW, UINT>::type_t);
		
		trafo_base<FFTW>::plan_t (*xxx_plan_many_dft_r2c)(const typename data<FFTW, INT>::type_t, const typename trafo<FFTW, R2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, R2C_64>::in_t*, const typename trafo<FFTW, R2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, R2C_64>::out_t*, const typename trafo<FFTW, R2C_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, const typename data<FFTW, UINT>::type_t);

		trafo_base<FFTW>::plan_t (*xxx_plan_many_dft_c2r)(const typename data<FFTW, INT>::type_t, const typename trafo<FFTW, C2R_64>::dim_t*, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, C2R_64>::in_t*, const typename trafo<FFTW, C2R_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, typename trafo<FFTW, C2R_64>::out_t*, const typename trafo<FFTW, C2R_64>::dim_t*, const typename data<FFTW, INT>::type_t, const typename data<FFTW, INT>::type_t, const typename data<FFTW, UINT>::type_t);

		void (*xxx_destroy_plan)(const typename trafo_base<FFTW>::plan_t);
	};
}

#include "fftlib_dynamic_lib_implementation.hpp"

#endif
