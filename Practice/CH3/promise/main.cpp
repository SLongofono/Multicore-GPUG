#include <iostream>
#include <future>
#include <thread>
#include <chrono>

/* This function takes in a promised value, and when it receives it, it will
 * immediately resume processing.  Note: it blocks on the f.get() call as
 * usual
 */
int async_func(std::future<int>& f){
	int input;

	input = f.get();

	return input * input;
}

int main(){
	/* The promise class is used to store a value/exception which will be
	 * populated asynchronously sometime in the future.  The promise has
	 * an associated std::future object which is used in the typical
	 * manner.  It also includes protections and signalling (monitor-like
	 * conditions) for any workers which have a reference to the future.
	 * For programmers, we need only use the set_value() function to
	 * trigger any context waiting on the future value.
	 */
	std::promise<int> p;
	
	// We pass this to our workers
	std::future<int> fufu = p.get_future();

	std::future<int> work_result = std::async(std::launch::async, async_func, std::ref(fufu));

	// Spin for a while
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	// Populate the value to the promise's future object
	p.set_value(10);

	std::cout << 	"The promise's future returned the value " <<
			work_result.get() << 
			" to the work_result future." <<
			std::endl;


	return 0;
}
