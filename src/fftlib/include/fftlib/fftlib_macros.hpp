// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_MACROS_HPP)
#define FFTLIB_MACROS_HPP

#if defined(FFTLIB_DEBUG) || defined(FFTLIB_WARNINGS_ONLY)
	#include <sstream>
	#if !defined(FFTLIB_WARNINGS_ONLY)
		#define FFTLIB_PRINT_INFO
	#endif
	#define FFTLIB_PRINT_WARNINGS
#endif

static void print_info(const std::string& msg)
{
	#pragma omp critical (FFTLIB_LOCK_INFO)
	{
		std::cout << "# INFO fftlib : " << msg << std::endl;
	}
}

static void print_warning(const std::string& msg)
{
	#pragma omp critical (FFTLIB_LOCK_WARNING)
	{
		std::cerr << "# WARNING fftlib : " << msg << std::endl;
	}
}

#if defined(FFTLIB_PRINT_INFO)
	#define FFTLIB_INFO(MSG) \
	{ \
		std::ostringstream m; \
		m << MSG; \
		print_info(m.str()); \
	}
#else
	#define FFTLIB_INFO(MSG)
#endif

#if defined(FFTLIB_PRINT_WARNINGS)
	#define FFTLIB_WARNING(MSG) \
	{ \
		std::ostringstream m; \
		m << MSG; \
		print_warning(m.str());	\
	}
#else
	#define FFTLIB_WARNING(MSG)
#endif

#if defined(FFTLIB_PROFILING)
        #define FFTLIB_START_TIMER(T) double T = omp_get_wtime()
        #define FFTLIB_STOP_TIMER(T, V) V += (omp_get_wtime() - T)
#else
        #define FFTLIB_START_TIMER(T) 
        #define FFTLIB_STOP_TIMER(T, V) 
#endif

#endif
