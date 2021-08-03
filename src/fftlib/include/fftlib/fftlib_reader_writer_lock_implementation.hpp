// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

namespace fftlib_internal
{
	reader_writer_lock::reader_writer_lock()
	{
		FFTLIB_INFO("reader_writer_lock constructor (thread=" << omp_get_thread_num() << ")");

		// initialize locks.
		omp_init_lock(&lock_1);
		omp_init_lock(&lock_2);
		
		// no readers and writes wait for acquiring the lock or already hold the lock.
		num_readers = 0;
		num_writers = 0;
	}

	reader_writer_lock::~reader_writer_lock()
	{
		FFTLIB_INFO("reader_writer_lock destructor (thread=" << omp_get_thread_num() << ")");

		// destroy locks.
		omp_destroy_lock(&lock_1);
		omp_destroy_lock(&lock_2);
	}

	void reader_writer_lock::acquire_read_lock()
	{
		FFTLIB_INFO("reader_writer_lock acquire_read_lock (thread=" << omp_get_thread_num() << ")");

		if (__sync_fetch_and_or(&num_writers, 0) == 0)
			{
				// no writers want to acquire 'lock_2': just increment the number of readers.
				// if this is the only reader at the current time, make it acquire 'lock_2'.
				if (__sync_fetch_and_add(&num_readers, 1) == 0)
					omp_set_lock(&lock_2);
			}
		else
			{
				// if there are already writers waiting for 'lock_2' wait for at least
		      		// the first writer to successfully acquire 'lock_2'.
				// it then will release 'lock_1' after having acquired 'lock_2' and writers and
				// readers will compete for 'lock_1' and then 'lock_2'.
				omp_set_lock(&lock_1);
				{
					if (__sync_fetch_and_add(&num_readers, 1) == 0)
						omp_set_lock(&lock_2);
				}
				omp_unset_lock(&lock_1);
			}
		// now 'lock_2' is on the readers' side.
	}
	
	void reader_writer_lock::release_read_lock()
	{
		FFTLIB_INFO("reader_writer_lock release_read_lock (thread=" << omp_get_thread_num() << ")");

		// reduce the number of readers by 1.
		// if this is the last reader, 'lock_2' can be release to allow
		// writers to acquire 'lock_2'.
		if (__sync_sub_and_fetch(&num_readers, 1) == 0)
			omp_unset_lock(&lock_2);
	}
	
	void reader_writer_lock::acquire_write_lock()
	{
		FFTLIB_INFO("reader_writer_lock acquire_write_lock (thread=" << omp_get_thread_num() << ")");

		// increment the number of writers by 1.
		__sync_fetch_and_add(&num_writers, 1);
		// acquire 'lock_1'.
		omp_set_lock(&lock_1);
		// now 'lock_1' is on the writers' side, and all possibly active readers
		// holding 'lock_2' finish their critical sections.
		// as no reader can acquire 'lock_1', the number of readers in critical 
		// sections reduces and finally 'lock_2' is released by the last reader.
		omp_set_lock(&lock_2);
		// now 'lock_2' is on the writers' side.
		// decrement the number of writers waiting for acquiring 'lock_2'.
		__sync_fetch_and_sub(&num_writers, 1);
		// release 'lock_1' so that other readers/writers become eligible to acquire 'lock_2'.
		omp_unset_lock(&lock_1);
	}

	void reader_writer_lock::release_write_lock()
	{
		FFTLIB_INFO("reader_writer_lock release_write_lock (thread=" << omp_get_thread_num() << ")");
		
		// release 'lock_2'.
		omp_unset_lock(&lock_2);
	}
}
