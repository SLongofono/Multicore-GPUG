/*
 * Barrier.h
 *
 * I adapted this code from a listing on Stackexchange to suit my needs.  I
 * chose to use this over the provided Barrier class because it had a good
 * solution for waiting (no need to keep track of how many threads are working
 * upstream).  The original code was authored by users Alexander Daryin and
 * Jinfeng Yang.  Their code was great except for getting cute with
 * pre-decrement and lambda expressions.  I imagine they are wonderful
 * obfuscators.
 *
 * URL:
 * https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11#24465624
 */

#ifndef BARR_H
#define BARR_H
#include <thread>
#include <mutex>
#include <condition_variable>

class Barrier{
	public:

		Barrier(int count);

		// Wait for the rest of the group to arrive
		void wait();

		// A thread has finished, and the group is now smaller (bound
		// below by 0)
		void decrement();

	private:
		std::mutex m_mutex;
		std::condition_variable m_condition;
		int m_count;
		int m_threshold;
};

#endif
