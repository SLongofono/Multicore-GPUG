// sample.c++: Code showing how to use ImageReader and Packed3DArray

#include "ImageReader.h"

void count(const cryph::Packed3DArray<unsigned char>* pa)
{
	// This simple example shows counting the number of instances of 138
	// in the provided image.
	int count = 0;
	// FOR EACH ROW OF THE IMAGE:
	for (int r=0 ; r<pa->getDim1() ; r++)
	{
		// FOR EACH COLUMN:
		for (int c=0 ; c<pa->getDim2() ; c++)
		{
			// FOR EACH CHANNEL (r, g, b):
			for (int rgb=0 ; rgb<pa->getDim3() ; rgb++)
				if (pa->getDataElement(r, c, rgb) == 138)
					count++;
		}
	}
	std::cout << "There were " << count << " instances of 138 in the image.\n";
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
