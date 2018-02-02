#include <iostream>
#include <thread> // Need this for std::ref because reasons
#include "myObj.h"


void useReference(myObj& thingy){
	std::cout << "The thingy has value: " << thingy.getVal() << std::endl;
}

void work(myObj& thingy){
	double result = 4 * thingy.getVal();
	std::cout << result << std::endl;
}


int main(){

	// Here is our class object
	myObj Ob;
	Ob.setVal(99.99);

	// To pass by reference, we can simply pass in the object itself.
	useReference(Ob);

	/*
	 * If we want to do stuff with it in a thread, we have a problem:
	 * thread will copy over the values, rather than give a reference like
	 * our work function is expecting.
	 *	std::thread t1(work, Ob);
	 *
	 * Instead, we need to create a reference wrapper, which is loosely
	 * speaking a class that behaves like a reference.  Thread happily
	 * copies over the pointer return of std::ref and still has a
	 * reference to the original object.  In this case, we are coercing
	 * the thread constructor's pass-by-value arguments to use our
	 * reference instead.
	 */

	std::thread t1(work, std::ref<myObj>(Ob));


	std::cout << "Back in main..." << std::endl;

	t1.join();

	//useReference(std::ref<myObj>(Ob));
	return 0;
}
