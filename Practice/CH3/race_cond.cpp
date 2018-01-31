// ThreadsAccessingGlobalVariables_Ex2.c++

#include <thread>
#include <iostream>
#include <math.h>

int nCallsToWork = 0;

double someSillyValue = 1.76;
double anotherSillyValue = 2.649;

void work(int assignment)
{
	double sr1 = sqrt(someSillyValue);
	double trigFcn = cos(anotherSillyValue);
	double myResult = pow(sr1, trigFcn);
	if (myResult < 0.0)
		std::cout << "Something weird happened.\n";
	// Since this global is not protected, it is possible (and
	// increasingly likely fior large numbers of threads) that it will not
	// increment properly.  The granularity of operations is at the
	// instruciton level rather than the expression level.  The increment
	// below is typically realized by the following intermediate code:
	//
	// t0 = nCallsToWork
	// t1 = t0 + 1
	// nCallsToWork = t1
	//
	// The scheduler could interrupt the thread after any of these
	// instructions.  If it were interrupted after the first instruction,
	// and another thread incremented nCallsToWork, the change would be
	// overwritten when control was returned to the first thread
	//
	nCallsToWork++;
}

int main(int argc, char* argv[])
{
	int nThreads = 10;
	if (argc > 1)
		nThreads = atoi(argv[1]);

	std::thread** t = new std::thread*[nThreads];
	for (int i=0 ; i<nThreads ; i++)
		t[i] = new std::thread(work, i);

	std::cout << "Hello from the parent thread.\n";
	for (int i=0 ; i<nThreads ; i++)
		t[i]->join();

	std::cout << "There were " << nCallsToWork << " calls to work.\n";
	return 0;
}

