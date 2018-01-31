// Listing 3.2 of Barlas

#include <thread>
#include <iostream>

void hello()
{
	std::cout << "Hello from the child thread.\n";
}

int main()
{
	std::thread t(hello);

	std::cout << "Hello from the parent thread.\n";
	t.join();
	return 0;
}
