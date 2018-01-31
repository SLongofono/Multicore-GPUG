#include <thread>
#include <iostream>

void work(int assignment)
{
	if (assignment == 3)
		std::cout << "I am the only thread allowed to tell you my work assignment: " << assignment << '\n';
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

	for (int i=0 ; i<nThreads ; i++)
		// Wait for the i-th thread to complete. It will complete when
		// the given function ("work" in this case) exits.
		t[i]->join();

	// Moving the parent thread output statement to here (instead of
	// before the previous loop) guarantees that the output will not be
	// interspersed with the output of the child with work assignment 3.
	std::cout << "Hello (and goodbye) from the parent thread.\n";

	return 0;
}
