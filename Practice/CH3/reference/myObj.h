#ifndef MOBJ_H
#define MOBJ_H

class myObj{
	public:
		myObj();
		double getVal() const;
		void setVal(double newVal);
	private:
		double myVal;
};

#endif
