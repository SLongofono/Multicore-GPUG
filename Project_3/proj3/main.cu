#include <iostream>
#include <cassert>
#include <fstream>
#include "ImageWriter.h"
#include <cuda.h>

#define DEBUG 1

#include "helpers.h"

using namespace std;

void launchMaxKernel();
void launchSumKernel1();
void launchSumKernel2();

int main(int argc, char **argv){
	
	/*
	 * Setup and housekeeping...
	 */
	if(argc < 7){
		cerr << "[ error ] Expected more arguments!" << endl;
		cerr << "Usage: ./project3 nRows nCols nSheets fileName projectionType outputFileNameBase" << endl;
		cerr << "Types: executable int int int string int string" << endl;
		cerr << "Exiting..." << endl;
		return -1;
	}

	int nRows, nCols, nSheets, nVals, projType;
	std::string fileType(".png");
	nRows = atoi(argv[1]);
	nCols = atoi(argv[2]);
	nSheets = atoi(argv[3]);
	projType = atoi(argv[5]);
	nVals = nRows * nCols * nSheets;
	unsigned char *rawImageData = new unsigned char[nVals];	
	ifstream infile(argv[4]);
	if(!infile.good()){
		cerr << "[ error ] Bad input filename.  Exiting..." << endl;
		return -1;
	}
	infile.read( reinterpret_cast<char *>(rawImageData), nVals);
	infile.close();

	switch(projType){
		case 1:
			cout << "Projection type " << projType << endl;
			break;
		case 2:
			cout << "Projection type " << projType << endl;
			break;
		case 3:
			cout << "Projection type " << projType << endl;
			break;
		case 4:
			cout << "Projection type " << projType << endl;
			break;
		case 5:
			cout << "Projection type " << projType << endl;
			break;
		case 6:
			cout << "Projection type " << projType << endl;
			break;
		default:
			cerr << "[ error ] '" << projType << "' is not a valid projection type, please select from [1,6]" << endl;
			delete [] rawImageData;
			return -1;
	}

#if DEBUG
	// Dump device information
	dumpDevices();
#endif

	/*
	 * Spit out an image, unadulterated
	 */
	writeFile( (argv[6] + fileType), nRows, nCols, rawImageData);

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	return 0;
}
