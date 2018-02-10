/*
 * Barrier.cpp
 *
 * I adapted this code from a listing on Stackexchange to suit my needs.  I
 * chose to use this over the provided Barrier class because it had a good
 * solution for waiting (no need to keep track of how many threads are working
 * upstream).  The original code was authored by users Alexander Daryin and
 * Jinfeng Yang.  Their code was great except for getting cute with
 * pre-decrement.  I imagine they are wonderful obfuscators.
 *
 * URL:
 * https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11#24465624
 */

#include "Barrier.h"


Barrier::Barrier(int count){
	m_count = count;
	m_threshold = count;
}

// Wait on all threads to arrive
void Barrier::wait(){
	// Take ownership of the mutex until scope resolves
	std::unique_lock<std::mutex> lock{m_mutex};
	

	// We have reached this point, so the count left to
	// move on is decremented
	m_count--;
	if(0 == m_count){
		// If all have arrived, reset count, and wake
		// everyone up to allow them to exit and move
		// on
		m_count = m_threshold;
		m_condition.notify_all();
	}
	else{
		/* 
		 * Otherwise, wait for the condition.  What is
		 * the condition?  In this case, being
		 * notified. Recall that the wait condition will
		 * unlock the mutex and stash this thread in a
		 * queue until notify is called.
		 */
		m_condition.wait(lock);
	}
}


// Decrement thread pool size (No less than 0)
void Barrier::decrement(){
	m_mutex.lock();
	if(m_threshold > 0){
		m_threshold--;
	}
	m_mutex.unlock();
}
