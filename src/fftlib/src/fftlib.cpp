// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#include "fftlib/fftlib.hpp"

using namespace fftlib;

// threading
extern "C" data<FFTW, INT>::type_t fftw_init_threads(void)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_init_threads (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	return fft<FFTW>::init_threads();
}

extern "C" void dfftw_init_threads_(data<FFTW, INT>::type_t* iret)
{
	(*iret) = fftw_init_threads();
}

extern "C" void fftw_plan_with_nthreads(data<FFTW, INT>::type_t nthreads)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_with_nthreads (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::plan_with_nthreads(nthreads);
}

extern "C" void dfftw_plan_with_nthreads(data<FFTW, INT>::type_t* nthreads)
{
	fftw_plan_with_nthreads((*nthreads));
}

extern "C" void fftlib_dynamic_lib_info()
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftlib_dynamic_lib_info (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::info();
}

extern "C" void fftw_cleanup_threads(void)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_cleanup_threads (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::cleanup_threads();
}

extern "C" void dfftw_cleanup_threads_(void)
{
	fftw_cleanup_threads();
}

extern "C" void fftw_cleanup(void)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_cleanup (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::cleanup();
}

extern "C" void dfftw_cleanup_(void)
{
	fftw_cleanup();
}

// complex to complex
extern "C"
trafo_base<FFTW>::plan_t fftw_plan_many_dft(const data<FFTW, INT>::type_t d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t howmany, data<FFTW, C_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t istride, const data<FFTW, INT>::type_t idist, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t ostride, const data<FFTW, INT>::type_t odist, const data<FFTW, INT>::type_t sign, const data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_many_dft (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	switch(d)
		{	
                case 1:
			{
				fft<FFTW, C2C_64, 1>& my_fft = fft<FFTW, C2C_64, 1>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
			}
			break;

                case 2:
			{
				fft<FFTW, C2C_64, 2>& my_fft = fft<FFTW, C2C_64, 2>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
			}
			break;

                case 3:
			{
				fft<FFTW, C2C_64, 3>& my_fft = fft<FFTW, C2C_64, 3>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
			}
			break;
			
		default:
			{
				FFTLIB_WARNING("fftw_plan_many_dft (thread=" << omp_get_thread_num() << ") only d=1,2,3 is currently supported");
			}
			return NULL;
			
		}
}

extern "C"
void dfftw_plan_many_dft_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t* howmany, data<FFTW, C_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t* istride, const data<FFTW, INT>::type_t* idist, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t* ostride, const data<FFTW, INT>::type_t* odist, const data<FFTW, INT>::type_t* sign, const data<FFTW, UINT>::type_t* flags)
{	
	(*p) = fftw_plan_many_dft((*d), n, (*howmany), in, inembed, (*istride), (*idist), out, onembed, (*ostride), (*odist), (*sign), (*flags)); 
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_1d(data<FFTW, INT>::type_t n0, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, data<FFTW, INT>::type_t sign, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_1d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[1] = {n0}; 
	return fftw_plan_many_dft(1, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, sign, flags);
}

extern "C"
void dfftw_plan_dft_1d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* sign, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_1d((*n_0), in, out, (*sign), (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_2d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, data<FFTW, INT>::type_t sign, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_2d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[2] = {n0, n1}; 
	return fftw_plan_many_dft(2, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, sign, flags);
}

extern "C"
void dfftw_plan_dft_2d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* sign, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_2d((*n_1), (*n_0), in, out, (*sign), (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_3d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, INT>::type_t n2, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, int sign, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_3d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[3] = {n0, n1, n2}; 
	return fftw_plan_many_dft(3, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, sign, flags);
}

extern "C"
void dfftw_plan_dft_3d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, const data<FFTW, INT>::type_t* n_2, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* sign, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_3d((*n_2), (*n_1), (*n_0), in, out, (*sign), (*flags));	
}

extern "C"
void fftw_execute_dft(const trafo_base<FFTW>::plan_t p, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_execute_dft (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW, C2C_64>::execute_plan(p, in, out);
}

extern "C"
void dfftw_execute_dft_(const trafo_base<FFTW>::plan_t* p, data<FFTW, C_64>::type_t* in, data<FFTW, C_64>::type_t* out)
{
	fftw_execute_dft((*p), in, out);	
}

// complex to real
extern "C"
trafo_base<FFTW>::plan_t fftw_plan_many_dft_c2r(const data<FFTW, INT>::type_t d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t howmany, data<FFTW, C_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t istride, const data<FFTW, INT>::type_t idist, data<FFTW, R_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t ostride, const data<FFTW, INT>::type_t odist, const data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_many_dft_c2r (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	switch(d)
		{	

                case 1:
			{
				fft<FFTW, C2R_64, 1>& my_fft = fft<FFTW, C2R_64, 1>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;

                case 2:
			{
				fft<FFTW, C2R_64, 2>& my_fft = fft<FFTW, C2R_64, 2>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;

                case 3:
			{
				fft<FFTW, C2R_64, 3>& my_fft = fft<FFTW, C2R_64, 3>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;
			
		default:
			{
				FFTLIB_WARNING("fftw_plan_many_dft_c2r (thread=" << omp_get_thread_num() << ") only d=1,2,3 is currently supported");
			}
			return NULL;
			
		}
}

extern "C"
void dfftw_plan_many_dft_c2r_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t* howmany, data<FFTW, C_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t* istride, const data<FFTW, INT>::type_t* idist, data<FFTW, R_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t* ostride, const data<FFTW, INT>::type_t* odist, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_many_dft_c2r((*d), n, (*howmany), in, inembed, (*istride), (*idist), out, onembed, (*ostride), (*odist), (*flags)); 
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_c2r_1d(data<FFTW, INT>::type_t n0, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_c2r_1d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[1] = {n0}; 
	return fftw_plan_many_dft_c2r(1, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_c2r_1d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_c2r_1d((*n_0), in, out, (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_c2r_2d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_c2r_2d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[2] = {n0, n1}; 
	return fftw_plan_many_dft_c2r(2, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_c2r_2d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_c2r_2d((*n_1), (*n_0), in, out, (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_c2r_3d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, INT>::type_t n2, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_c2r_3d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[3] = {n0, n1, n2}; 
	return fftw_plan_many_dft_c2r(3, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_c2r_3d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, const data<FFTW, INT>::type_t* n_2, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_c2r_3d((*n_2), (*n_1), (*n_0), in, out, (*flags));	
}

extern "C"
void fftw_execute_dft_c2r(const trafo_base<FFTW>::plan_t p, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_execute_dft_c2r (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW, C2R_64>::execute_plan(p, in, out);
}

extern "C"
void dfftw_execute_dft_c2r_(const trafo_base<FFTW>::plan_t* p, data<FFTW, C_64>::type_t* in, data<FFTW, R_64>::type_t* out)
{
	fftw_execute_dft_c2r((*p), in, out);	
}

// real to complex
extern "C"
trafo_base<FFTW>::plan_t fftw_plan_many_dft_r2c(const data<FFTW, INT>::type_t d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t howmany, data<FFTW, R_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t istride, const data<FFTW, INT>::type_t idist, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t ostride, const data<FFTW, INT>::type_t odist, const data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_many_dft_r2c (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	switch(d)
		{	

                case 1:
			{
				fft<FFTW, R2C_64, 1>& my_fft = fft<FFTW, R2C_64, 1>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;

                case 2:
			{
				fft<FFTW, R2C_64, 2>& my_fft = fft<FFTW, R2C_64, 2>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;

                case 3:
			{
				fft<FFTW, R2C_64, 3>& my_fft = fft<FFTW, R2C_64, 3>::get_instance();
				return my_fft.create_plan(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, FFTW_FORWARD, flags);
			}
			break;
			
		default:
			{
				FFTLIB_WARNING("fftw_plan_many_dft_c2r (thread=" << omp_get_thread_num() << ") only d=1,2,3 is currently supported");
			}
			return NULL;
			
		}
}

extern "C"
void dfftw_plan_many_dft_r2c_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* d, const data<FFTW, INT>::type_t* n, const data<FFTW, INT>::type_t* howmany, data<FFTW, R_64>::type_t* in, const data<FFTW, INT>::type_t* inembed, const data<FFTW, INT>::type_t* istride, const data<FFTW, INT>::type_t* idist, data<FFTW, C_64>::type_t* out, const data<FFTW, INT>::type_t* onembed, const data<FFTW, INT>::type_t* ostride, const data<FFTW, INT>::type_t* odist, const data<FFTW, UINT>::type_t* flags)
{	
	(*p) = fftw_plan_many_dft_r2c((*d), n, (*howmany), in, inembed, (*istride), (*idist), out, onembed, (*ostride), (*odist), (*flags));
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_r2c_1d(data<FFTW, INT>::type_t n0, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_r2c_1d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[1] = {n0}; 
	return fftw_plan_many_dft_r2c(1, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_r2c_1d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_r2c_1d((*n_0), in, out, (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_r2c_2d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_r2c_2d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[2] = {n0, n1}; 
	return fftw_plan_many_dft_r2c(2, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_r2c_2d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_r2c_2d((*n_1), (*n_0), in, out, (*flags));	
}

extern "C" trafo_base<FFTW>::plan_t fftw_plan_dft_r2c_3d(data<FFTW, INT>::type_t n0, data<FFTW, INT>::type_t n1, data<FFTW, INT>::type_t n2, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, data<FFTW, UINT>::type_t flags)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_plan_dft_r2c_3d (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	const data<FFTW, INT>::type_t n[3] = {n0, n1, n2}; 
	return fftw_plan_many_dft_r2c(3, n, 1, in, NULL, 1, 0, out, NULL, 1, 0, flags);
}

extern "C"
void dfftw_plan_dft_r2c_3d_(trafo_base<FFTW>::plan_t* p, const data<FFTW, INT>::type_t* n_0, const data<FFTW, INT>::type_t* n_1, const data<FFTW, INT>::type_t* n_2, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out, const data<FFTW, UINT>::type_t* flags)
{
	(*p) = fftw_plan_dft_r2c_3d((*n_2), (*n_1), (*n_0), in, out, (*flags));	
}

extern "C"
void fftw_execute_dft_r2c(const trafo_base<FFTW>::plan_t p, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_execute_dft_r2c (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW, R2C_64>::execute_plan(p, in, out);
}

extern "C"
void dfftw_execute_dft_r2c_(const trafo_base<FFTW>::plan_t* p, data<FFTW, R_64>::type_t* in, data<FFTW, C_64>::type_t* out)
{	
	fftw_execute_dft_r2c((*p), in, out);	
}

extern "C"
void fftw_execute(const trafo_base<FFTW>::plan_t p)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_execute (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::execute_plan(p);
}

extern "C"
void dfftw_execute_(const trafo_base<FFTW>::plan_t* p)
{
	fftw_execute((*p));
}

extern "C"
void fftw_destroy_plan(const trafo_base<FFTW>::plan_t p)
{
	#if defined(FFTLIB_PRINT_INFO)
		static bool first_call = true;
		#pragma omp threadprivate(first_call)
		if (first_call)
			{
				FFTLIB_INFO("fftw_destroy_plan (thread=" << omp_get_thread_num() << ")");
				first_call = false;
			}
	#endif

	fft<FFTW>::destroy_plan(p);
}

extern "C"
void dfftw_destroy_plan_(trafo_base<FFTW>::plan_t* p)
{
	fftw_destroy_plan((*p));
}
