// sample.c++: Code showing how to use ImageReader and Packed3DArray

#include "ImageReader.h"

int **count(const cryph::Packed3DArray<unsigned char>* pa)
{
	int size = pa->getDim1() * pa -> getDim2();
	int rval, gval, bval, max=255;
	int *rlevels = new int[256]();	// Note: c++ equivalent of calloc using default constructor explicity
	int *blevels = new int[256]();
	int *glevels = new int[256]();

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

			if(rval < 0 || gval < 0 || bval < 0){
				std::cout << "ERROR! negative value in count!" << std::endl;
			}

			/*
			std::cout << "Read a point: (" << rval << ","
				  << gval << "," << bval << ")"
				  << std::endl;
			*/
		}
	}

	int **counts = new int*[3];
	counts[0] = rlevels;
	counts[1] = glevels;
	counts[2] = blevels;

	return counts;

	//delete rlevels;
	//delete glevels;
	//delete blevels;
}

// Assumes 256 levels
float **histo(int *rlevels, int *glevels, int *blevels, int numPixels){
	/* Since images may differ in size, we will express each value as a
	 * percentage of the count of pixels in the image used to generate the
	 * histogram
	 */
	float normalization = 1.0/numPixels;
	float rsum, gsum, bsum, val;
	float **ret = new float*[3];
	for(int i = 0; i<3; ++i){
		ret[i] = new float[256];
	}
	rsum = gsum = bsum = val = 0.0;

	std::cout << "==================================================" << std::endl
		  << "                   Red Levels                     " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * rlevels[i];
		if(val < 0){
			std::cout << "ERROR! negative percentage in histo!" << std::endl;
			std::cout << "normalization: " << normalization << std::endl;
			std::cout << "Level: " << rlevels[i] << std::endl;
		}
		ret[0][i] = val;
		rsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << rsum << std::endl;
	
	std::cout << "==================================================" << std::endl
		  << "                   Blue Levels                    " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * glevels[i];
		if(val < 0){
			std::cout << "ERROR! negative percentage in histo!" << std::endl;
			std::cout << "normalization: " << normalization << std::endl;
			std::cout << "Level: " << glevels[i] << std::endl;
		}
		ret[1][i] = val;
		gsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << gsum << std::endl;
	
	
	std::cout << "==================================================" << std::endl
		  << "                   Green Levels                   " << std::endl
		  << "==================================================" << std::endl;
	for(int i =0; i<256; ++i){
		val = normalization * blevels[i];
		if(val < 0){
			std::cout << "ERROR! negative percentage in histo!" << std::endl;
			std::cout << "normalization: " << normalization << std::endl;
			std::cout << "Level: " << blevels[i] << std::endl;
		}
		ret[2][i] = val;
		bsum += val;
		std::cout << "Level " << i << ":\t\t" << val << std::endl;
	}
	std::cout << "Sanity check: sum of percentages is " << bsum << std::endl;

	return ret;
}

int main(int argc, char* argv[])
{
	if (argc < 2){
		std::cerr << "Usage: " << argv[0] << " imageFileName\n";
	}
	else
	{
		ImageReader* ir = ImageReader::create(argv[1]);
		ImageReader* ir2 = ImageReader::create(argv[2]);
		if (ir == nullptr || ir2 == nullptr){
			std::cerr << "Could not open image files: " << argv[1] << " " << argv[2] << '\n';
		}
		else{
			int dim11, dim12, dim21, dim22;
			dim11 = ir->getInternalPacked3DArrayImage()->getDim1();
			dim12 = ir->getInternalPacked3DArrayImage()->getDim2();
			dim21 = ir2->getInternalPacked3DArrayImage()->getDim1();
			dim22 = ir2->getInternalPacked3DArrayImage()->getDim2();
			int **counts = count(ir->getInternalPacked3DArrayImage());
			int **counts2 = count(ir2->getInternalPacked3DArrayImage());
			float **histogram = histo(counts[0], counts[1], counts[2], dim11*dim12);
			float **histogram2 = histo(counts2[0], counts2[1], counts2[2], dim21*dim22);
		
		float totalDiff = 0.0;
		float currDiff = 0.0;
		for(int i = 0; i<3; ++i){
			for(int j = 0; j<256; ++j){
				currDiff = histogram[i][j] - histogram2[i][j];
				totalDiff += currDiff > 0 ? currDiff: -1*currDiff;
			}
		}

		std::cout << "The total difference between the images is " << totalDiff << std::endl;
		
		
		}
	}
	return 0;
}
