#include <thread>
#include <iostream>

void work(int assignment)
{
	std::cout << "I am a thread given work assignment: " << assignment << "\n";
}

int main(int argc, char* argv[])
{
	// By default, we will create 10 threads:
	int nThreads = 10;
	if (argc > 1)
		nThreads = atoi(argv[1]);

	std::thread** t = new std::thread*[nThreads];
	for (int i=0 ; i<nThreads ; i++)
		// All parameters to the std::thread constructor after the
		// first are passed as parameters to the function identified
		// by the first parameter:
		t[i] = new std::thread(work, i);

	std::cout << "Hello from the parent thread.\n";
	for (int i=0 ; i<nThreads ; i++)
		// Wait until the i-th thread completes. It will complete
		// when the given function ("work" in this case) exits.
		t[i]->join();

	return 0;
}
