#include <iostream>
#include <future>
#include <thread>
#include <mutex>

std::mutex mutex_result;

void doWork(int val1, int val2, double& myResult){
	double result = 0.5 * (val1 + val2);
	mutex_result.lock();
	std::cout << "The work: " << result << std::endl;
	myResult = result;
	mutex_result.unlock();
}

double doWorkSon(int val1, int val2){
	return 0.5 * (val1 + val2);
}

int main(){
	
	double myResult, martyMcFly;

	/* Creating a thread is great, but fetching the result can be
	 * cumbersome in c++. We need global locks, passing by reference, busy
	 * waiting, etc.
	 */
	std::thread t1(doWork, 3, 6, std::ref<double>(myResult));

	/* Instead, we can use futures.  Futures create a spot for variables
	 * which have yet to be creating/computed.  The async function will
	 * accept a work assignment and either run it in the current thread or
	 * generate a new one.  When the result is ready, it is returned to
	 * the future and some housekeeping is done.
	 *
	 * The blocking function std::future::get() retrieves the return value
	 * of the work function, and DESTROYS the future.
	 *
	 * We can enforce creation of threads with the optional arguments
	 * std::launch::async or std::launch::deferred, which spawn and do not
	 * spawn threads respectively.  Note that the latter will effectively
	 * do nothing until get() is called, emulating lazy evaluation.
	 */
	std::future<double> fufu = std::async(std::launch::async, doWorkSon, 3, 6);
	
	martyMcFly = fufu.get();

	std::cout << "If my calculations are correct, when this baby hits " <<
		     martyMcFly <<
		     " miles per hour, you're gonna see some serious shit."
		     << std::endl;

	t1.join();

	return 0;
}
