#include "myObj.h"

myObj::myObj(){
	myVal = 0;
}

double myObj::getVal() const {
	return myVal;
}

void myObj::setVal(double newVal){
	myVal = newVal;
}
