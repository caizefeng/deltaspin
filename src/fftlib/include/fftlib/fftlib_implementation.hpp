// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	template<backend B>
	int fft_base<B>::init_threads()
	{
		FFTLIB_INFO("fft_base init_threads (thread=" << omp_get_thread_num() << ")");

		dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
		return dl.init_threads();
        }

	template<backend B>
	void fft_base<B>::plan_with_nthreads(const typename data<B, INT>::type_t nthreads)
	{
		FFTLIB_INFO("fft_base plan_with_nthreads (thread=" << omp_get_thread_num() << ")");

		dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
		dl.plan_with_nthreads(nthreads);
        }

	template<backend B>
	void fft_base<B>::destroy_plan(typename trafo_base<B>::plan_t p)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
			dl.destroy_plan(p);
			return;
		#endif

		FFTLIB_INFO("fft_base destroy_plan (thread=" << omp_get_thread_num() << ")");

		#if defined(FFTLIB_UNSAFE_OPT)
			delete reinterpret_cast<ext_plan*>(p);
		#else
			// check if the map containing all 'ext_plan's holds this particular plan:
			// the address in main memory is used as key.
			std::uint64_t p_address = reinterpret_cast<std::uint64_t>(p);
			bool plan_found = false;

			// we just look up the plan: acquire the read lock for the map.
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				lock_plans.acquire_read_lock();
			#elif defined(FFTLIB_THREADSAFE)
				#pragma omp critical (FFTLIB_LOCK_PLANS)
			#endif
				{
					std::unordered_map<std::uint64_t, std::uint64_t>::const_iterator it = fftlib_internal::plans.find(p_address);
					if (it != fftlib_internal::plans.end())
						plan_found = true;
				}
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				lock_plans.release_read_lock();
			#endif

			// if the plan is in the map, remove it from the map.
			// otherwise destroy the plan using a 'direct' library call.
			if (plan_found)
				{
					FFTLIB_INFO("fft_base destroy_plan (thread=" << omp_get_thread_num() << ") plan still exists in fftlib's hash_map");
					
					// acquire the write lock for the map.
					#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
						lock_plans.acquire_write_lock();
					#elif defined(FFTLIB_THREADSAFE)
						#pragma omp critical (FFTLIB_LOCK_PLANS)
					#endif
						{
							// get a valid iterator: the former one might have been invalidated 
							// by other threads due to changing the map content (add/remove).
							std::unordered_map<std::uint64_t, std::uint64_t>::const_iterator it = fftlib_internal::plans.find(p_address);
							if (it != fftlib_internal::plans.end())
								{
									delete reinterpret_cast<ext_plan*>(p);
									fftlib_internal::plans.erase(it);	
								}
					}
					#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
						lock_plans.release_write_lock();
					#endif
				}
			else
				{
					dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
					dl.destroy_plan(p);
				}
		#endif
	}

	template<backend B>
	void fft_base<B>::cleanup_threads()
	{
		FFTLIB_INFO("fft_base cleanup_threads (thread=" << omp_get_thread_num() << ")");

		dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
		dl.cleanup_threads();
	}

	template<backend B>
	void fft_base<B>::cleanup()
	{
		FFTLIB_INFO("fft_base cleanup (thread=" << omp_get_thread_num() << ")");

		dynamic_lib<B>& dl = dynamic_lib<B>::get_instance();
		dl.cleanup();
	}
}

namespace fftlib
{
	template<backend B, transformation T, std::int32_t D>
	fft<B, T, D>::fft()
	{
		FFTLIB_INFO("fft constructor (thread=" << omp_get_thread_num() << ")");

		// determine the total number of threads allowed in the process context
		// (at this point we run sequentially).
		// 1.) request max_threads for the top-level parallel region.
		// 2.) if nested parallelism is enabled, reguest max_threads in the parallel region.
		std::int32_t max_threads = omp_get_max_threads();
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
		
		// create cache.
		cache.assign(max_threads, NULL);
		for (std::int32_t k = 0; k < max_threads; ++k)
			{
				// allocate memory.
				cache[k] = new std::pair<const configuration*, const plan*>[FFTLIB_CACHE_SIZE];
	
				// initialize the cache: empty -> (NULL, NULL) pairs
				for (std::int32_t i = 0; i < FFTLIB_CACHE_SIZE; ++i)
					cache[k][i] = std::pair<const configuration*, const plan*>(NULL, NULL);
			}

		slots_used.assign(max_threads, 0);
		next_slot.assign(max_threads, 0);
	}

	template<backend B, transformation T, std::int32_t D>
	fft<B, T, D>::~fft()
	{
		FFTLIB_INFO("fft destructor (thread=" << omp_get_thread_num() << ")");

		// reset the cache: empty -> (NULL, NULL) pairs
		for (std::int32_t k = 0; k < cache.size(); ++k)
			for (std::int32_t i = 0; i < FFTLIB_CACHE_SIZE; ++i)
				cache[k][i] = std::pair<const configuration*, const plan*>(NULL, NULL);
	}

	template<backend B, transformation T, std::int32_t D>
	fft<B, T, D>& fft<B, T, D>::get_instance()
	{
		// singleton patter: this method holds the only instance of
		// the fft<B, T, D> class.
		//
		// multi-threading (C++11): all threads obtain a valid reference
		// to the fft<B, T, D> in both the single-/multi-threaded context.
		FFTLIB_INFO("fft get_instance (thread=" << omp_get_thread_num() << ")");

		static fft instance;
		return instance;	
	}

	template<backend B, transformation T, std::int32_t D>
	const typename trafo<B, T>::plan_t* fft<B, T, D>::find_plan(const fftlib_internal::configuration<B, T, D>& c)
	{
		std::int32_t omp_id = omp_get_thread_num();

		// look up the plan in the cache.
		for (std::int32_t i = 0; i < slots_used[omp_id]; ++i)
			if (c == *(cache[omp_id][i].first))
				{
					FFTLIB_INFO("fft find_plan (thread=" << omp_id << ") plan found in cache");
					
					return &((cache[omp_id][i].second)->p);
				}

		// if the plan is not in the cache, look it up in the hash map.
		const typename trafo<B, T>::plan_t* p = NULL;
		const configuration* p_c = NULL;
		const plan* p_p = NULL;
		bool plan_found = false;
		#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
			lock_map.acquire_read_lock();
		#elif defined(FFTLIB_THREADSAFE)
			#pragma omp critical (FFTLIB_LOCK_MAP)
		#endif
			{
				unordered_map_const_iterator it = plans.find(c);
				if (it != plans.end())
					{
						// the plan is in the map: extract pointers from the iterator to the respective key-value pair.
						plan_found = true;
						p_c = &(it->first);
						p_p = &(it->second);
					}
			}
		#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
			lock_map.release_read_lock();
		#endif
		
		if (plan_found)
			{
				FFTLIB_INFO("fft find_plan (thread=" << omp_id << ") plan found in hash map"  << std::endl << "# +-------------> insert into cache");
				
				// the plan is in the hash map: insert the plan into the cache.
				cache[omp_id][next_slot[omp_id]++] = std::pair<const configuration*, const plan*>(p_c, p_p);
				next_slot[omp_id] = next_slot[omp_id] % FFTLIB_CACHE_SIZE;
				slots_used[omp_id] = std::min(FFTLIB_CACHE_SIZE, slots_used[omp_id] + 1);
				
				p = &(p_p->p);
			}
		else
			{
				// the plan is not in the hash map.
				FFTLIB_INFO("fft find_plan (thread=" << omp_id << ") plan not found");
			}

		return p;
	}

	template<backend B, transformation T, std::int32_t D>
	const typename trafo<B, T>::plan_t* fft<B, T, D>::create_plan(const configuration& c, typename trafo<B, T>::in_t* in, typename trafo<B, T>::out_t* out)
	{
		std::int32_t omp_id = omp_get_thread_num();

		FFTLIB_INFO("fft create plan (thread=" << omp_id << ") insert into hash map and cache");

		// create a new plan in-place when inserting it into the hash map.
		const configuration* p_c = NULL;
		const plan* p_p = NULL;
		#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
			lock_map.acquire_write_lock();
		#elif defined(FFTLIB_THREADSAFE)
			#pragma omp critical (FFTLIB_LOCK_MAP)
		#endif
			{
				std::pair<unordered_map_const_iterator, bool> p = plans.emplace(std::piecewise_construct_t(), std::tuple<const fftlib_internal::configuration<B, T, D>>(c), std::tuple<const fftlib_internal::configuration<B, T, D>&, typename trafo<B, T>::in_t*, typename trafo<B, T>::out_t*>(c, in, out));
				
				// extract pointers from the iterator to the respective key-value pair.
				p_c = &((*(p.first)).first);
				p_p = &((*(p.first)).second);
			}
		#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
			lock_map.release_write_lock();
		#endif
		
		// insert the plan into the cache. 
		cache[omp_id][next_slot[omp_id]++] = std::pair<const configuration*, const plan*>(p_c, p_p);
		next_slot[omp_id] = next_slot[omp_id] % FFTLIB_CACHE_SIZE;
		slots_used[omp_id] = std::min(FFTLIB_CACHE_SIZE, slots_used[omp_id] + 1);
		
		return &(p_p->p);
	}

	template<backend B, transformation T, std::int32_t D>
	typename trafo<B, T>::plan_t fft<B, T, D>::create_plan(const typename data<B, INT>::type_t* n, const typename data<B, INT>::type_t howmany, typename trafo<B, T>::in_t* in, const typename data<B, INT>::type_t* inembed, const typename data<B, INT>::type_t istride, const typename data<B, INT>::type_t idist, typename trafo<B, T>::out_t* out, const typename data<B, INT>::type_t* onembed, const typename data<B, INT>::type_t ostride, const typename data<B, INT>::type_t odist, const typename data<B, INT>::type_t sign, const typename data<B, UINT>::type_t flags)
	{
		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			fftlib_internal::dynamic_lib<B>& dl = fftlib_internal::dynamic_lib<B>::get_instance();
			return dl.template create_plan<T>(D, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
		#endif

		FFTLIB_INFO("fft create_plan (thread=" << omp_get_thread_num() << ")");
		
		// create a new 'ext_plan' object and set up its attribute.
		fftlib_internal::ext_plan* p = new fftlib_internal::ext_plan();

		p->b = B;
		p->d = D;
		p->t = T;
		for (std::int32_t i = 0; i < D; ++i)
			p->n[i] = n[i];
		p->howmany = howmany;
		p->idist = idist;
		p->odist = odist;
		p->in = reinterpret_cast<void *>(in);
		p->out = reinterpret_cast<void *>(out);

		// create a plan configuration object.
		configuration c = configuration(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);

		// look up this particular plan configuration in the hash map
		// and insert a respective plan (also in the cache) if not there.
		if ((p->p_1 = reinterpret_cast<const void*>(find_plan(c))) == NULL)
			p->p_1 = reinterpret_cast<const void*>(create_plan(c, in, out));

		#if defined(FFTLIB_UNSAFE_OPT)
			// return an 'fftw_plan' plan.
			return reinterpret_cast<typename trafo<B, T>::plan_t>(p);
		#else
			std::uint64_t p_address = reinterpret_cast<std::uint64_t>(p);

			// insert the 'ext_plan' object's address into the 'plans' hash map to
			// look it up in other methods, and so to find out whether the plan
			// given to these methods has been created by fftlib.
			//
			// note: fftw_plan is an opaque pointer type, and as such it is unique, that is,
			// no two plans from FFTW/MKL and fftlib will have the same address, and thus
			// can be distinguished via a hash map look up.
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				fftlib_internal::lock_plans.acquire_write_lock();
			#elif defined(FFTLIB_THREADSAFE)
				#pragma omp critical (FFTLIB_LOCK_PLANS)
			#endif
				{
					fftlib_internal::plans.emplace(p_address, p_address);
				}
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				fftlib_internal::lock_plans.release_write_lock();
			#endif

			// return an 'fftw_plan' plan.
			return reinterpret_cast<typename trafo<B, T>::plan_t>(p);
		#endif
	}

	template<backend B, transformation T, std::int32_t D>
	void fft<B, T, D>::execute_plan(const typename trafo<B, T>::plan_t p, typename trafo<B, T>::in_t* in, typename trafo<B, T>::out_t* out)
	{
		fftlib_internal::dynamic_lib<B>& dl = fftlib_internal::dynamic_lib<B>::get_instance();

		// bypass internal logic.
		#if defined(FFTLIB_BYPASS)
			if (in == NULL && out == NULL)
				dl.execute_plan(p);
			else
				dl.template execute_plan<T>(p, in, out);
			return;
		#endif

		FFTLIB_INFO("fft execute_plan (thread=" << omp_get_thread_num() << ")");

		#if !defined(FFTLIB_UNSAFE_OPT)
			//  look up the plan in the 'plans' hash map.
			std::uint64_t p_address = reinterpret_cast<std::uint64_t>(p);
			bool plan_found = false;
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				fftlib_internal::lock_plans.acquire_read_lock();
			#elif defined(FFTLIB_THREADSAFE)
				#pragma omp critical (FFTLIB_LOCK_PLANS)
			#endif
				{
					std::unordered_map<std::uint64_t, std::uint64_t>::const_iterator it = fftlib_internal::plans.find(p_address);
					if (it != fftlib_internal::plans.end())
						plan_found = true;
				}
			#if defined(FFTLIB_THREADSAFE) && defined(FFTLIB_OWN_LOCK)
				fftlib_internal::lock_plans.release_read_lock();
			#endif

			if (plan_found)
		#endif
				{
					FFTLIB_INFO("fft execute_plane (thread=" << omp_get_thread_num() << ") plan was created by fftlib");

					// if the plan has been found in the hash map, reinterpret its
					// attributes accordingly and execute it.
					fftlib_internal::ext_plan* pp = reinterpret_cast<fftlib_internal::ext_plan*>(p);
					
					if (in == NULL && out == NULL)
						#if defined(FFTLIB_UNSAFE_OPT) && defined(FFTLIB_USE_MKL)
							dl.template execute_plan<T>(*reinterpret_cast<const typename trafo<B, T>::plan_t*>(pp->p_1), reinterpret_cast<typename trafo<B, T>::in_t*>(pp->in), reinterpret_cast<typename trafo<B, T>::out_t*>(pp->out));
						#else
							switch (pp->t)
								{
								case C2C_64:
									dl.template execute_plan<C2C_64>(*reinterpret_cast<const typename trafo<B, C2C_64>::plan_t*>(pp->p_1), reinterpret_cast<typename trafo<B, C2C_64>::in_t*>(pp->in), reinterpret_cast<typename trafo<B, C2C_64>::out_t*>(pp->out));
									break;
									
								case R2C_64:
									dl.template execute_plan<R2C_64>(*reinterpret_cast<const typename trafo<B, R2C_64>::plan_t*>(pp->p_1), reinterpret_cast<typename trafo<B, R2C_64>::in_t*>(pp->in), reinterpret_cast<typename trafo<B, R2C_64>::out_t*>(pp->out));
									break;
									
								case C2R_64:
									dl.template execute_plan<C2R_64>(*reinterpret_cast<const typename trafo<B, C2R_64>::plan_t*>(pp->p_1), reinterpret_cast<typename trafo<B, C2R_64>::in_t*>(pp->in), reinterpret_cast<typename trafo<B, C2R_64>::out_t*>(pp->out));
									break;
									
								default:
									FFTLIB_WARNING("fft execute_plan (thread=" << omp_get_thread_num() << ") something went wrong");
								}
						#endif
					else
						dl.template execute_plan<T>(*reinterpret_cast<const typename trafo<B, T>::plan_t*>(pp->p_1), in, out);								
				}
		#if !defined(FFTLIB_UNSAFE_OPT)
			else
				{
					FFTLIB_INFO("fft fftw_execute_dft (thread=" << omp_get_thread_num() << ") plan was not created by fftlib");
				
					if (in == NULL && out == NULL)
						dl.execute_plan(p);
					else
						dl.template execute_plan<T>(p, in, out);
				}
		#endif
	}

	template<backend B, transformation T, std::int32_t D>
	void fft<B, T, D>::info()
	{
		FFTLIB_INFO("fft info (thread=" << omp_get_thread_num() << ")");

		fftlib_internal::dynamic_lib<B>& dl = fftlib_internal::dynamic_lib<B>::get_instance();
		dl.info();
	}
}
