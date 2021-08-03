// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	using namespace fftlib;

	dynamic_lib<FFTW>::dynamic_lib() 
		: plan_with_threads(false),
		  initialize_threading(true)
	{
		// multi-threading: C++11 guarantees threadsafeness here, that is,
		// object instantiation goes along with an implicit barrier.
		//
		// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm
		FFTLIB_INFO("dynamic_lib constructor (thread=" << omp_get_thread_num() << ")");
		
		// determine the total number of threads allowed in the process context
		// (at this point we run sequentially).
		// 1.) request max_threads for the top-level parallel region.
		// 2.) if nested parallelism is enabled, reguest max_threads in the parallel region.
		max_threads = omp_get_max_threads();
		if (omp_get_nested())
			{
				#pragma omp parallel
				{
					#pragma omp single
					{
						// only one thread needs to execute this
						max_threads *= omp_get_max_threads();
					}
				}
			}
		
		// in the first instance it is assumed that multi-threading is not used.
		nthreads.assign(max_threads, 1);
	       
		// these dynamic libraries should be in the LD_LIBRARY_PATH when
		// running the application
		#if defined(FFTLIB_USE_MKL)
			const char filename[] = "libmkl_intel_lp64.so";
			const char filename_omp[] = "libmkl_intel_lp64.so";
		#else
			const char filename[] = "libfftw3.so";
			const char filename_omp[] = "libfftw3_omp.so";
		#endif

		dynamic_loader& loader =  dynamic_loader::get_instance();
		
		// load symbols
		if (loader.get_symbol(NULL, (void **)&xxx_init_threads, "fftw_init_threads"))
	       		loader.get_symbol(filename_omp, (void **)&xxx_init_threads, "fftw_init_threads");
		
		if (loader.get_symbol(NULL, (void **)&xxx_plan_with_nthreads, "fftw_plan_with_nthreads"))
			loader.get_symbol(filename_omp, (void **)&xxx_plan_with_nthreads, "fftw_plan_with_nthreads");

		if (loader.get_symbol(NULL, (void **)&xxx_cleanup_threads, "fftw_cleanup_threads"))
			loader.get_symbol(filename_omp, (void **)&xxx_cleanup_threads, "fftw_cleanup_threads");
	       	
		if (loader.get_symbol(NULL, (void **)&xxx_cleanup, "fftw_cleanup"))
			loader.get_symbol(filename, (void **)&xxx_cleanup, "fftw_cleanup");

		if (loader.get_symbol(NULL, (void **)&xxx_execute, "fftw_execute"))
			loader.get_symbol(filename, (void **)&xxx_execute, "fftw_execute");

		if (loader.get_symbol(NULL, (void **)&xxx_execute_dft, "fftw_execute_dft"))
			loader.get_symbol(filename, (void **)&xxx_execute_dft, "fftw_execute_dft");

		if (loader.get_symbol(NULL, (void **)&xxx_execute_dft_c2r, "fftw_execute_dft_c2r"))
			loader.get_symbol(filename, (void **)&xxx_execute_dft_c2r, "fftw_execute_dft_c2r");

		if (loader.get_symbol(NULL, (void **)&xxx_execute_dft_r2c, "fftw_execute_dft_r2c"))
			loader.get_symbol(filename, (void **)&xxx_execute_dft_r2c, "fftw_execute_dft_r2c");

		if (loader.get_symbol(NULL, (void **)&xxx_plan_many_dft, "fftw_plan_many_dft"))
			loader.get_symbol(filename, (void **)&xxx_plan_many_dft, "fftw_plan_many_dft");

		if (loader.get_symbol(NULL, (void **)&xxx_plan_many_dft_c2r, "fftw_plan_many_dft_c2r"))
			loader.get_symbol(filename, (void **)&xxx_plan_many_dft_c2r, "fftw_plan_many_dft_c2r");

		if (loader.get_symbol(NULL, (void **)&xxx_plan_many_dft_r2c, "fftw_plan_many_dft_r2c"))
			loader.get_symbol(filename, (void **)&xxx_plan_many_dft_r2c, "fftw_plan_many_dft_r2c");

		if (loader.get_symbol(NULL, (void **)&xxx_destroy_plan, "fftw_destroy_plan"))
			loader.get_symbol(filename, (void **)&xxx_destroy_plan, "fftw_destroy_plan");
	}

	dynamic_lib<FFTW>::~dynamic_lib()
	{
		// multi-threading: C++11 guarantees threadsafeness here, that is,
		// object destruction happens just once.
		FFTLIB_INFO("dynamic_lib destructor (thread=" << omp_get_thread_num() << ")");
	}

	dynamic_lib<FFTW>& dynamic_lib<FFTW>::get_instance()
	{
		// singleton patter: this method holds the only instance of
		// the dynamic_lib class.
		//
		// multi-threading (C++11): all threads obtain a valid reference
		// to the dynamic_lib in both the single-/multi-threaded context.
		FFTLIB_INFO("dynamic_lib get_instance (thread=" << omp_get_thread_num() << ")");

		static dynamic_lib instance;
		return instance;
	}
	
	void dynamic_lib<FFTW>::info() const
	{
		FFTLIB_INFO("dynamic_lib info (thread=" << omp_get_thread_num() << ")");

		std::string backend_name;
		#if defined(FFTLIB_USE_MKL)
			backend_name = "FFTW/MKL";
		#else
			backend_name = "FFTW";			
		#endif

		#pragma omp critical (FFTLIB_LOCK_INFO)
		{
			std::cout << "# dynamic_lib:" << std::endl;
			std::cout << "# +-------------> backend=" << backend_name << std::endl;
			std::cout << "# +-------------> plan_with_threads=" << (plan_with_threads ? "yes" : "no") << std::endl;
			std::cout << "# +-------------> max_threads=" << max_threads << std::endl;
			std::cout << "# +-------------> nthreads(self)=" << nthreads[omp_get_thread_num()] << std::endl;
			std::cout << "# +-------------> nthreads(master)=" << nthreads[0] << std::endl;
		}
	}

	typename data<FFTW, INT>::type_t dynamic_lib<FFTW>::get_nthreads() const
	{
		// return the current number of threads used for plan creation/execution
		//
		// multi-threading: in a parallel region, threads may use different
		// numbers of threads for plan creation/executin.
		if (!plan_with_threads || (omp_in_parallel() && !omp_get_nested()))
			return 1;
		else
			return nthreads[omp_get_thread_num()];
	}

	typename data<FFTW, INT>::type_t dynamic_lib<FFTW>::init_threads(void)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			return (*(xxx_init_threads))();
		#endif

		// return if threading has already been enabled.
		if (!initialize_threading)
			return 0;

		// the thread initialization routine should be called just once.
		if (omp_in_parallel())
			{
				// do not allow thread initialization in a parallel region.
				// throw a warning, but continue execution and return 'success'.
				#pragma omp single
				{
					FFTLIB_WARNING("dynamic_lib init_threads (thread=" << omp_get_thread_num() << ") called in parallel region (the program might fail if continued)");	
				}
				return 0;
			}
		else
			{
				// initialize threading.
				if ((*(xxx_init_threads))())
					{
						#if defined(FFTLIB_USE_MKL)
							// when using MKL's FFTW wrapppers, use 'max_threads' for plan creation in all cases.
							// the actual number of threads used for execution is set within the execute methods.
							(*(xxx_plan_with_nthreads))(max_threads);
							FFTLIB_INFO("dynamic_lib init_threads (thread=" << omp_get_thread_num() << ") with maximum number of threads=" << max_threads);
						#else
							FFTLIB_INFO("dynamic_lib init_threads (thread=" << omp_get_thread_num() << ")");
						#endif

						// extend the nthreads field to 'max_threads' entries and initialize then to 1.
						nthreads.assign(max_threads, 1);
						plan_with_threads = true;
						// thread initialization should happen just once.
						// there should not be any issues with multi-threading as we run sequentially here.
						initialize_threading = false;

						return 1;
					}
				else
					{
						FFTLIB_WARNING("dynamic_lib init_threads (thread=" << omp_get_thread_num() << ") some error occured");

						return 0;
					}
			}
	}

	void dynamic_lib<FFTW>::plan_with_nthreads(const typename data<FFTW, INT>::type_t nthreads)
	{	
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_plan_with_nthreads))(nthreads);
			return;
		#endif

		if (!plan_with_threads)
			return;

		if (omp_in_parallel())
			{
				// if nested parallelism is enabled, threads can adapt their thread count for
				// plan creation/execution individually.
				if (omp_get_nested())
					{
						FFTLIB_INFO("dynamic_lib plan_with_nthreads (thread=" << omp_get_thread_num() << ")");

						this->nthreads[omp_get_thread_num()] = nthreads;
					}
				// if not, but threads try to, threaded FFT computation is disabled by assigning 1.
				else if (!omp_get_nested() && nthreads != 1)
					{
						FFTLIB_WARNING("dynamic_lib plan_with_nthreads (thread=" << omp_get_thread_num() << ") nesting is disabled" << std::endl << "# +-------------> use 1 thread instead of " << nthreads << " as requested");

						this->nthreads[omp_get_thread_num()] = 1;
					}
			}
		else
			{
				FFTLIB_INFO("dynamic_lib plan_with_nthreads (thread=" << omp_get_thread_num() << ")");

				// the change of the thread count by the master thread (sequential context)
				// applies to all threads.
				this->nthreads.assign(max_threads, nthreads);
			}
	}

	void dynamic_lib<FFTW>::cleanup_threads(void)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_cleanup_threads))();
			return;
		#endif

		// continue only if multi-threading is used.
		if (!plan_with_threads)
			return;

		// do not call 'fftw_cleanup_threads' as it invalidates all plans in the map/cache.
		// instead, leave everything as is, and set the number of threads to be used for
		// FFT computation to 1 (this basically disables threading).
		if (omp_in_parallel())
			{
				FFTLIB_WARNING("dynamic_lib cleanup_threads (thread=" << omp_get_thread_num() << ") called in parallel region (the program might fail if continued)");

				nthreads[omp_get_thread_num()] = 1;
			}
		else
			{
				FFTLIB_INFO("dynamic_lib cleanup_threads (thread=" << omp_get_thread_num() << ")");

				nthreads.assign(max_threads, 1);
			}
	}	  

	void dynamic_lib<FFTW>::cleanup(void) const
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_cleanup))();
			return;
		#endif

		// do not call 'fftw_cleanup' as it invalidates all plans in the map/cache.
		if (omp_in_parallel())
			{
				#pragma omp single
				{
					FFTLIB_WARNING("dynamic_lib cleanup (thread=" << omp_get_thread_num() << ") called in parallel region (the program might fail if continued)");
				}
			}
		else
			{
				FFTLIB_INFO("dynamic_lib cleanup (thread=" << omp_get_thread_num() << ")");		
			}
	}

#define FFTLIB_FUNCTION_SIGNATURE(TRAFO) \
	template<> \
	typename trafo<FFTW, TRAFO>::plan_t dynamic_lib<FFTW>::create_plan<TRAFO>(const typename data<FFTW, INT>::type_t d, const typename trafo<FFTW, TRAFO>::dim_t* n, const typename data<FFTW, INT>::type_t howmany, typename trafo<FFTW, TRAFO>::in_t* in, const typename trafo<FFTW, TRAFO>::dim_t* inembed, const typename data<FFTW, INT>::type_t istride, const typename data<FFTW, INT>::type_t idist, typename trafo<FFTW, TRAFO>::out_t* out, const typename trafo<FFTW, TRAFO>::dim_t* onembed, const typename data<FFTW, INT>::type_t ostride, const typename data<FFTW, INT>::type_t odist, const typename data<FFTW, INT>::type_t sign, const typename data<FFTW, UINT>::type_t flags)

	FFTLIB_FUNCTION_SIGNATURE(C2C_64)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			return (*(xxx_plan_many_dft))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
		#endif

		FFTLIB_INFO("dynamic_lib create_plan<C2C_64> (thread=" << omp_get_thread_num() << ")");

		// use the current number of threads for plan creation.
		// the plan generated below is specific to this thread count.
		#if !defined(FFTLIB_USE_MKL)
			if (plan_with_threads)
				(*(xxx_plan_with_nthreads))(get_nthreads());
		#endif

		return (*(xxx_plan_many_dft))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags); 
	}

	FFTLIB_FUNCTION_SIGNATURE(R2C_64)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			return (*(xxx_plan_many_dft_r2c))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags);
		#endif

		FFTLIB_INFO("dynamic_lib create_plan<R2C_64> (thread=" << omp_get_thread_num() << ")");
		
		// use the current number of threads for plan creation.
		// the plan generated below is specific to this thread count.
		#if !defined(FFTLIB_USE_MKL)
			if (plan_with_threads)
				(*(xxx_plan_with_nthreads))(get_nthreads());
		#endif

		return (*(xxx_plan_many_dft_r2c))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags); 
	}

	FFTLIB_FUNCTION_SIGNATURE(C2R_64)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			return (*(xxx_plan_many_dft_c2r))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags);
		#endif

		FFTLIB_INFO("dynamic_lib create_plan<C2R_64> (thread=" << omp_get_thread_num() << ")");
		
		// use the current number of threads for plan creation.
		// the plan generated below is specific to this thread count.
		#if !defined(FFTLIB_USE_MKL)
			if (plan_with_threads)
				(*(xxx_plan_with_nthreads))(get_nthreads());
		#endif
	
		return (*(xxx_plan_many_dft_c2r))(d, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags); 
	}
#undef  FFTLIB_FUNCTION_SIGNATURE

	void dynamic_lib<FFTW>::execute_plan(const typename trafo_base<FFTW>::plan_t p) const
	{		
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_execute))(p);
			return;
		#endif

		// when using MKL's FFTW wrappers, the number of threads used for plan execution is set here
		// (the plans are not specific to any thread count).
		// in case of nested parallelism, we use 'mkl_set_num_threads_local' and recover the global
		// settings after plan execution using 'mkl_set_num_threads_local(0)'.
		#if defined(FFTLIB_USE_MKL)		
			if (omp_in_parallel())
				mkl_set_num_threads_local(get_nthreads());
		#endif

		FFTLIB_INFO("dynamic_lib execute_plan (thread=" << omp_get_thread_num() << ")");
				
		(*(xxx_execute))(p);

		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel() && omp_get_thread_num() == 0)
				mkl_set_num_threads_local(0);
		#endif
	}

	template<>
        void dynamic_lib<FFTW>::execute_plan<C2C_64>(const typename trafo<FFTW, C2C_64>::plan_t p, typename trafo<FFTW, C2C_64>::in_t* in, typename trafo<FFTW, C2C_64>::out_t* out) const
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_execute_dft))(p, in, out);
			return;
		#endif

		FFTLIB_INFO("dynamic_lib execute_plan<C2C_64> (thread=" << omp_get_thread_num() << ")");
		
		// when using MKL's FFTW wrappers, the number of threads used for plan execution is set here
		// (the plans are not specific to any thread count).
		// in case of nested parallelism, we use 'mkl_set_num_threads_local' and recover the global
		// settings after plan execution using 'mkl_set_num_threads_local(0)'.
		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel())
				mkl_set_num_threads_local(get_nthreads());
		#endif
		
		(*(xxx_execute_dft))(p, in, out);

		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel() && omp_get_thread_num() == 0)
				mkl_set_num_threads_local(0);
		#endif
	}

	template<>
        void dynamic_lib<FFTW>::execute_plan<R2C_64>(const typename trafo<FFTW, R2C_64>::plan_t p, typename trafo<FFTW, R2C_64>::in_t* in, typename trafo<FFTW, R2C_64>::out_t* out) const
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_execute_dft_r2c))(p, in, out);
			return;
		#endif

		FFTLIB_INFO("dynamic_lib execute_plan<R2C_64> (thread=" << omp_get_thread_num() << ")");
		
		// when using MKL's FFTW wrappers, the number of threads used for plan execution is set here
		// (the plans are not specific to any thread count).
		// in case of nested parallelism, we use 'mkl_set_num_threads_local' and recover the global
		// settings after plan execution using 'mkl_set_num_threads_local(0)'.
		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel())
				mkl_set_num_threads_local(get_nthreads());
		#endif
		
		(*(xxx_execute_dft_r2c))(p, in, out);

		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel() && omp_get_thread_num() == 0)
				mkl_set_num_threads_local(0);
		#endif
	}

	template<>
	void dynamic_lib<FFTW>::execute_plan<C2R_64>(const typename trafo<FFTW, C2R_64>::plan_t p, typename trafo<FFTW, C2R_64>::in_t* in, typename trafo<FFTW, C2R_64>::out_t* out) const
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_execute_dft_c2r))(p, in, out);
			return;
		#endif

		FFTLIB_INFO("dynamic_lib execute_plan<C2R_64> (thread=" << omp_get_thread_num() << ")");
		
		// when using MKL's FFTW wrappers, the number of threads used for plan execution is set here
		// (the plans are not specific to any thread count).
		// in case of nested parallelism, we use 'mkl_set_num_threads_local' and recover the global
		// settings after plan execution using 'mkl_set_num_threads_local(0)'.
		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel())
				mkl_set_num_threads_local(get_nthreads());
		#endif
		
		(*(xxx_execute_dft_c2r))(p, in, out);

		#if defined(FFTLIB_USE_MKL)
			if (omp_in_parallel() && omp_get_thread_num() == 0)
				mkl_set_num_threads_local(0);
		#endif
	}

	void dynamic_lib<FFTW>::destroy_plan(const typename trafo_base<FFTW>::plan_t p) const
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			(*(xxx_destroy_plan))(p);
			return;
		#endif

		FFTLIB_INFO("dynamic_lib destroy_plan (thread=" << omp_get_thread_num() << ")");

		(*(xxx_destroy_plan))(p);		
	}
}
