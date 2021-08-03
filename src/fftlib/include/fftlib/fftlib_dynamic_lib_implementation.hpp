// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	dynamic_loader::dynamic_loader()
	{
		// multi-threading: C++11 guarantees threadsafeness here, that is,
		// object instantiation goes along with an implicit barrier.
		//
		// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm
		FFTLIB_INFO("dynamic_loader constructor (thread=" << omp_get_thread_num() << ")");
	}

	dynamic_loader::~dynamic_loader()
	{
		// multi-threading: C++11 guarantees threadsafeness here, that is,
		// object destruction happens just once.
		FFTLIB_INFO("dynamic_loader destructor (thread=" << omp_get_thread_num() << ")");

		// close all open library handles.
		std::map<const std::string, void*>::iterator it = handles.begin();		
		for (std::int32_t i = 0; i < handles.size(); ++i, ++it)
			{				
				if (it->second != NULL)
					{						
						dlclose(it->second);
						it->second = NULL;
					}				
			}		
		
		// clear the internal map.
		handles.clear();
	}
	  
	dynamic_loader& dynamic_loader::get_instance()
	{
		// singleton patter: this method holds the only instance of
		// the dynamic_loader class.
		//
		// multi-threading (C++11): all threads obtain a valid reference
		// to the dynamic_loader in both the single-/multi-threaded context.
		FFTLIB_INFO("dynamic_loader get_instance (thread=" << omp_get_thread_num() << ")");

		static dynamic_loader instance;
		return instance;
		
	}

	void dynamic_loader::info() const
	{
		FFTLIB_INFO("dynamic_loader info (thread=" << omp_get_thread_num() << ")");

		#pragma omp critical (FFTLIB_LOCK_INFO)
		{
			std::cout << "# dynamic_loader:" << std::endl;
			std::cout << "# +-------------> open library handles=" << handles.size() << std::endl;
		}
	}
	std::int32_t dynamic_loader::get_symbol(const std::string& filename, void** f, const std::string& symbol)
	{
		FFTLIB_INFO("dynamic_loader get_symbol(" << symbol << ") (thread=" << omp_get_thread_num() << ") success");

		std::int32_t status = 0;
		char* error_string = NULL;

		// if called in a multi-threaded context, acquire a
		// lock to make calls to 'open' and hence changes to
		// the internal map threadsafe.
		#if defined(FFTLIB_THREADSAFE)
			#pragma omp critical (FFTLIB_LOCK_GET_SYMBOL)
		#endif
			{	
				// RTLD_NEXT .. will find the next occurrence of a function
				// in the search order after the current library. this allows 
				// one to provide a wrapper around a function in another shared library.
				// 
				// http://linux.die.net/man/3/dlsym
				void* handle = (filename.empty() ? RTLD_NEXT : open(filename));
				
				if (handle != NULL)
					{
						// clear any existing error
						dlerror();
						// get address of the symbol
						(*f) = dlsym(handle, symbol.c_str());
						// when using RTLD_NEXT no error is generated when the
						// requested symbol cannot be found: test for NULL pointer.
						if ((*f) == NULL || (error_string = dlerror()) != NULL)
							status = 1;
					}
				else
					{
						status = 1;
					}
			}
			
			if (status == 0)
				{
				FFTLIB_INFO("dynamic_loader get_symbol(" << symbol << ") (thread=" << omp_get_thread_num() << ") found (" << (filename.empty() ? std::string("RTLD_NEXT") : filename) << ")");
				}
			else
				{
					FFTLIB_WARNING("dynamic_loader get_symbol(" << symbol << ") (thread=" << omp_get_thread_num() << ") not found" << error_string);
				}
			
			return status;
	}

	std::int32_t dynamic_loader::get_symbol(const char* filename, void** f, const char* symbol)
	{
		return get_symbol(std::string(filename == NULL ? "" : filename), f, std::string(symbol == NULL ? "" : symbol)); 
	}

	void* dynamic_loader::open(const std::string& filename)
	{
		// this method is always called in the destructor.
		std::map<const std::string, void*>::const_iterator it = handles.find(filename);

		// if the requested library has already been opened, return the
		// handle stored in the internal map.
		if (it != handles.end())
			{
				FFTLIB_INFO("dynamic_loader open(" << filename << ") (thread=" << omp_get_thread_num() << ") success");
				return it->second;
			}

		// otherwise, open the library: check if the library has already been loaded
		// via RTLD_NOLOAD (returns NULL if the library has not been loaded yet).
		// use lazy binding via RTLD_LAZY (resolve symbols only if actually referenced).
		void* handle = NULL;
		if ((handle = dlopen(filename.c_str(), RTLD_NOLOAD)) == NULL &&
		    (handle = dlopen(filename.c_str(), RTLD_LAZY)) == NULL)
			{				
				FFTLIB_WARNING("dynamic_loader open(" << filename << ") (thread=" << omp_get_thread_num() << ") not found");
			}
		else
			{
				FFTLIB_INFO("dynamic_loader open(" << filename << ") (thread=" << omp_get_thread_num() << ") success");
				handles.insert(std::pair<const std::string, void*>(filename, handle));
			}
		
		return handle;
	}

	void* dynamic_loader::open(const char* filename)
	{
		return open(std::string(filename));
	}

}

#include "fftlib_dynamic_lib_implementation_fftw.hpp"

// TODO: add DFTI implementation here
// #include "fftlib_dynamic_lib_implementation_dfti.hpp"
