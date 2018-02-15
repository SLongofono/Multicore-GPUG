// Barrier.h - A class that implements a Barrier

#ifndef BARRIER_H
#define BARRIER_H

#include <condition_variable>
#include <mutex>
#include <iostream>

/* Usage:
	1. Create an instance of a Barrier class (called, say, "b") that
	   is accessible to, but outside the scope of any thread code that
	   needs to use it.
	2. In the thread code where barrier synchronization is to occur,
	   each thread in the "barrier group" must execute:

	   b.barrier(num); // where "num" is the number of threads in
	                   // the "barrier group"
*/

/* KNOWN DESIGN FLAW (2017-02-21):
	It is assumed that all threads in a barrier group will invoke the barrier
	method with the same (and correct) value for "numInBarrierGroup". If any
	thread calls the barrier method with a different value, various things may
	occur, depending on the value passed. For example, the barrier may
	immediately empty, or all threads may remain stuck at the barrier
	indefinitely.
*/

class Barrier
{
public:
	Barrier() : barrierCounter(0) {}
	virtual ~Barrier() {}

	void barrier(int numInBarrierGroup)
	{
		std::unique_lock<std::mutex> ulbm(barrierMutex);
		std::cout << "BARRIER\n";

		barrierCounter++;
		if (barrierCounter != numInBarrierGroup){
			barrierCV.wait(ulbm);
		}
		else
		{
			barrierCounter = 0;
			barrierCV.notify_all();
		}
		std::cout << "AWOKEN\n";
	}
private:
	int barrierCounter;
	std::mutex barrierMutex;
	std::condition_variable barrierCV;
};

#endif

