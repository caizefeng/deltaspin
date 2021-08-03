// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_READER_WRITER_LOCK)
#define FFTLIB_READER_WRITER_LOCK

namespace fftlib_internal 
{
	class reader_writer_lock
	{
	public:
		reader_writer_lock();

		~reader_writer_lock();

		void acquire_read_lock();

		void release_read_lock();

		void acquire_write_lock();

		void release_write_lock();

	private:
		omp_lock_t lock_1, lock_2;
		
		std::int32_t num_readers, num_writers;
	};
}

#include "fftlib_reader_writer_lock_implementation.hpp"

#endif
