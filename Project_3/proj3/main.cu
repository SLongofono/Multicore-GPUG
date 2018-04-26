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


/*
 * The max image kernel
 *
 * Each thread will collapse one pixel of the result matrix.  This apporach
 * allows the collective memory accesses to happen in a columnwise fashion and
 * be coalesced by the GPU memory manager.
 */
__global__ maxKernel(unsigned char *voxels, unsigned char *maxImage, float *maxes, int nRows, int nCols, int nSheets, int boundary){

	// Since the data is in column-major order, the thread IDs correspond
	// to the initial position in the first column of the first sheet
	int myStartOffset = blockDim.x * blockIdx.x + threadIdx;
	int localMax = -1;
	int globalMax = -1;
	if(myStartOffset < boundary){
		for(int c = 0; c < nCols; ++c){
			/* Step through the sheets, tracking the max along all sheets
			 *
			 * If we want to access column wise no matter what, we
			 * need to traverse each position across all sheets
			 * before moving on to the next row.
			 *
			 * This means we start at our column offset, increment
			 * by the dimensions of each sheet, and track the max
			 * as we go.
			 *
			 * Then, to get to the next row, we take our offset
			 * plus the number of rows times the current column
			 * index.  This allows each thread to step through the
			 * rows, while ensuring that all executing threads are
			 * accessing the same column synchronously.
			 *
			 * So long as we ensure that we have enough threads to
			 * cover a column in full (ideally more than that),
			 * and we check to make sure that we haven't run out
			 * of columns, then we should be fine
			 *
			 */
			for(int n = myStartOffset + c*nRows; n < nRows*nCols*nSheets; n += nRows*nCols){
				localMax = (localMax < voxels[n]) ? voxels[n] : localMax;
				globalMax = (globalMax < localMax) ? localMax : globalMax;
			}

			// Assign the max of each sheet position to the corresponding output image position
		}
	}

}

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
	unsigned char *d_maxImage, *h_maxImage;
	unsigned char *d_sumImage *h_sumImage;
	int resultSize;
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
	cudaMemcpy(d_voxels, rawImageData, nVals*sizeof(unsigned char),cudaMemcpyHostToDevice);

#if DEBUG
	writeFile("Original.png", nCols, nRows, rawImageData);
#endif

	/*
	 * Configure projection-specific details and launch kernels.  Rather
	 * than rely on the GPU to try and traverse differently based on the
	 * projection, we swap out the data in-place to the desired
	 * projection, allowing us to optimize memory access in terms of the
	 * resulting 2D image dimensions.
	 */
	
	// Re-flatten array per projection
	projection(rawImageData, nRows, nCols, nSheets, projType);

	switch(projType){
		case 1:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nSheets*sizeof(float));

			break;
		case 2:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nSheets*sizeof(float));

			break;
		case 3:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nCols*sizeof(float));

			break;
		case 4:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nCols*sizeof(float));

			break;
		case 5:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nRows*sizeof(float));

			break;
		case 6:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMax, nRows*sizeof(float));

			break;
		default:
			cerr << "[ error ] '" << projType << "' is not a valid projection type, please select from [1,6]" << endl;
			delete [] rawImageData;
			return -1;
	}

	/*
	 * Retrieve results
	 */
	cudaMemcpy(h_maxImage, d_maxImage, resultSize, cudaMemcpyDeviceToHost); 
	//cudaMemcpy(h_sumImage, d_sumImage, resultSize, cudaMemcpyDeviceToHost); 

	/*
	 * Write results
	 */
	writeImage("max.png", h_maxImage, projType, nRows, nCols, nSheets);
	//writeImage("sum.png", h_sumImage, projType, nRows, nCols, nSheets);
	

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	cudaFree(d_maxImage);
	cudaFree(d_sumImage);
	cudaFree(d_localMax);

	return 0;
}
