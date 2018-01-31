#include <thread>
#include <iostream>

// Note the two different ways to query thread IDs. Be sure you understand
// why there are two different ways.

void work(int assignment)
{
	if (assignment == 3)
	{
		std::cout << "I am the only child thread allowed to talk.\n";
		std::cout << "My work assignment is: " << assignment << "\n";
		std::cout << "A thread can query its ID. I'll grab mine and tell you what it is!\n";
		std::thread::id ID = std::this_thread::get_id();
		std::cout << "OK, I got it.  Here is my ID: " << ID << "\n";
		std::cout << "My work here is done!\n";
	}
}

int main(int argc, char* argv[])
{
	// By default, we will create 10 threads:
	int nThreads = 10;
	if (argc > 1)
		nThreads = atoi(argv[1]);

	std::thread** t = new std::thread*[nThreads];
	std::thread::id* threadIDs = new std::thread::id[nThreads];
	for (int i=0 ; i<nThreads ; i++)
	{
		t[i] = new std::thread(work, i);
		// Let's save the ID of this thread.
		threadIDs[i] = t[i]->get_id();
	}

	for (int i=0 ; i<nThreads ; i++)
		t[i]->join();

	for (int i=0 ; i<nThreads ; i++)
		std::cout << "Thread " << i << " with ID = " << threadIDs[i]
		          << " has completed.\n";
	std::cout << "Hello (and goodbye) from the parent thread.\n";

	return 0;
}
