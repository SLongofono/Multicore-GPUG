// sample.c++: Code showing how to use ImageReader and Packed3DArray

#include "ImageReader.h"

void histo(int *r, int *g, int*b, int size);

void count(const cryph::Packed3DArray<unsigned char>* pa)
{
	int size = pa->getDim1() * pa -> getDim2();
	int rval, gval, bval, max=255;
	int *rlevels = new int[255]();	// Note: c++ equivalent of calloc using default constructor explicity
	int *blevels = new int[255]();
	int *glevels = new int[255]();

	// FOR EACH ROW OF THE IMAGE:
	for (int r=0 ; r<pa->getDim1() ; r++)
	{
		// FOR EACH COLUMN:
		for (int c=0 ; c<pa->getDim2() ; c++)
		{
			// FOR EACH CHANNEL (r, g, b):
			rval = pa->getDataElement(r,c,0);
			gval = pa->getDataElement(r,c,1);
			bval = pa->getDataElement(r,c,2);
			rlevels[rval]++;
			glevels[gval]++;
			blevels[bval]++;
			
			/*
			std::cout << "Read a point: (" << rval << ","
				  << gval << "," << bval << ")"
				  << std::endl;
			*/
		}
	}

	histo(rlevels,glevels,blevels, size);

	delete rlevels;
	delete glevels;
	delete blevels;
}

// Assumes 256 levels
void histo(int *rlevels, int *glevels, int *blevels, int numPixels){
	/* Since images may differ in size, we will express each value as a
	 * percentage of the count of pixels in the image used to generate the
	 * histogram
	 */
	float normalization = 1.0/numPixels;
	float rsum, gsum, bsum, val;
	rsum = gsum = bsum = val = 0.0;

	std::cout << "==================================================" << std::endl
		  << "                   Red Levels                     " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * rlevels[i];
		rsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << rsum << std::endl;
	
	std::cout << "==================================================" << std::endl
		  << "                   Blue Levels                    " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * glevels[i];
		gsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << gsum << std::endl;
	
	
	std::cout << "==================================================" << std::endl
		  << "                   Green Levels                   " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * blevels[i];
		bsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << bsum << std::endl;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		std::cerr << "Usage: " << argv[0] << " imageFileName\n";
	else
	{
		ImageReader* ir = ImageReader::create(argv[1]);
		if (ir == nullptr)
			std::cerr << "Could not open image file: " << argv[1] << '\n';
		else
			count(ir->getInternalPacked3DArrayImage());
	}
	return 0;
}
