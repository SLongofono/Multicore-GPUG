#include <iostream>
#include <cassert>
#include <fstream>
#include "ImageWriter.h"
#include <cuda.h>
#include "helpers.h"
#include "kernels.h"

#define DEBUG 1
#define DEVICE_NUM 0

using namespace std;

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

#if DEBUG
	// Dump device information
	dumpDevices();
#endif

	int nRows, nCols, nSheets, nVals, projType;
	std::string fileType(".png");
	nRows = atoi(argv[1]);
	nCols = atoi(argv[2]);
	nSheets = atoi(argv[3]);
	projType = atoi(argv[5]);
	nVals = nRows * nCols * nSheets;
	unsigned char *rawImageData = new unsigned char[nVals];	
	unsigned char *d_voxels;
	unsigned char *d_maxImage;
	unsigned char *d_sumImage;
	float *d_localMax;

	
	ifstream infile(argv[4]);
	if(!infile.good()){
		cerr << "[ error ] Bad input filename.  Exiting..." << endl;
		return -1;
	}
	infile.read( reinterpret_cast<char *>(rawImageData), nVals);
	infile.close();


	/*
	 * Copy voxel data to the GPU
	 *
	 * We can assume that device 0 is valid if it exists.  Here, I
	 * explicitly set it to use the device I want, the Quadro 6000.  Not
	 * necessary, but I wanted to leave this here for future reference.
	 */
	cudaSetDevice(DEVICE_NUM);
	cudaMalloc((void **)&d_voxels, nVals*sizeof(unsigned char));
	

	/*
	 * Configure projection-specific details and launch kernels.  Rather
	 * than rely on the GPU to try and traverse differently based on the
	 * projection, we swap out the data in-place to the desired
	 * projection, allowing us to optimize memory access in terms of the
	 * resulting 2D image dimensions.
	 */
	switch(projType){
		case 1:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nCols*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nCols*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nSheets*sizeof(float));
			break;
		case 2:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nCols*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nCols*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nSheets*sizeof(float));
			break;
		case 3:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nSheets*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nSheets*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nCols*sizeof(float));
			break;
		case 4:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nSheets*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nSheets*nRows*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nCols*sizeof(float));
			break;
		case 5:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nCols*nSheets*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nCols*nSheets*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nRows*sizeof(float));
			break;
		case 6:
			cout << "Projection type " << projType << endl;
			cudaMalloc((void **)&d_maxImage, nCols*nSheets*sizeof(unsigned char));
			cudaMalloc((void **)&d_sumImage, nCols*nSheets*sizeof(unsigned char));
			cudaMalloc((void **)&d_localMax, nRows*sizeof(float));
			break;
		default:
			cerr << "[ error ] '" << projType << "' is not a valid projection type, please select from [1,6]" << endl;
			delete [] rawImageData;
			return -1;
	}

	//writeFile( (argv[6] + fileType), nRows, nCols, rawImageData);

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	
	return 0;
}
