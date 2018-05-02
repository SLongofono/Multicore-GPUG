#include <iostream>
#include <cassert>
#include <fstream>
#include "ImageWriter.h"
#include <cuda.h>
#include "helpers.h"
#include <cmath>

#define DEBUG 1
#define DEVICE_NUM 0

using namespace std;


void __global__ kernelMaxImage(unsigned char *voxels, unsigned char *maxImage, float *weightedSums, float *globalMax, int nSheets){
	int myPos = blockIdx.x*blockDim.x + threadIdx.x;
	int localMax = 0;
	float norm = 1.0/nSheets;
	float weightedSum = 0.0;
	int imageSize = gridDim.x*blockDim.x;
	unsigned char curVal;

	for(int sh = 0; sh < nSheets; sh++){
		curVal = voxels[myPos + sh*imageSize];
		
		if(curVal > localMax){
			localMax = curVal;
		}
		weightedSum += norm * curVal * (sh+1);
		//weightedSum += norm * curVal*(nSheets - sh);
	}

	// Update output image for my pixel
	maxImage[myPos] = localMax;

	// Update weighted sums for my pixel
	weightedSums[myPos] = weightedSum;

	// Update global Max
	atomicMax(globalMax, weightedSum);
}


void __global__ kernelSumImage(float *weightedSums, unsigned char *sumImage, float *globalMax){
	int myPos = blockIdx.x*blockDim.x + threadIdx.x;
	float max = globalMax[0];
	float localMax = weightedSums[myPos];
	int result = (int)((localMax/max)*255.0);
	float diff = (localMax - max);
	if(diff < 0){
		diff *= -1.0;
	}
	//printf("Norm is %f, myVal is %f", norm, weightedSums[myPos]);
	// Formula : output = round( (p/globalMax)*255.0  )
	if((max - localMax) < 0.001){
		printf("MyPos: %d, globalMax: %f, localMax: %f, result: %d\n", myPos, max, localMax, result);
	}
	sumImage[myPos] = result;
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
	unsigned char *rawImageData = new unsigned char[nVals]();	
	unsigned char *d_voxels;
	unsigned char *d_maxImage, *h_maxImage;
	unsigned char *d_sumImage, *h_sumImage;
	int resultSize;
	float *d_weightedSums;
	float *d_globalMax;

	
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
	validate(cudaMalloc((void **)&d_voxels, nVals*sizeof(unsigned char)));

	/*
	 * Configure projection-specific details and launch kernels.  Rather
	 * than rely on the GPU to try and traverse differently based on the
	 * projection, we swap out the data in-place to the desired
	 * projection, allowing us to optimize memory access in terms of the
	 * resulting 2D image dimensions.
	 */
	
	// Re-flatten array per projection
	projection(rawImageData, nRows, nCols, nSheets, projType);
	validate(cudaMemcpy(d_voxels, rawImageData, nVals*sizeof(unsigned char),cudaMemcpyHostToDevice));

	// Issue kernels
	switch(projType){
		case 1:	// Note: need braces to restrict scope of the local variables
			{
				cout << "Projection type " << projType << endl;
				resultSize = nCols*nRows*sizeof(unsigned char);
				h_maxImage = new unsigned char[nCols*nRows];
				h_sumImage = new unsigned char[nCols*nRows];
				validate(cudaMalloc((void **)&d_maxImage, resultSize));
				validate(cudaMalloc((void **)&d_sumImage, resultSize));
				validate(cudaMalloc((void **)&d_weightedSums, nCols*nRows*sizeof(float)));
				validate(cudaMalloc((void **)&d_globalMax, sizeof(float)));
			
				kernelMaxImage<<<nCols, nRows>>>(d_voxels,d_maxImage, d_weightedSums, d_globalMax, nSheets);
				validate(cudaPeekAtLastError()); // Check invalid launch
				validate(cudaDeviceSynchronize()); // Check runtime error

				kernelSumImage<<<nCols, nRows>>>(d_weightedSums, d_sumImage, d_globalMax);
				validate(cudaPeekAtLastError());
				validate(cudaDeviceSynchronize());
			}
			break;
		case 2:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_weightedSums, nCols*nRows*sizeof(float));

			break;
		case 3:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_weightedSums, nSheets*nRows*sizeof(float));

			break;
		case 4:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_weightedSums, nSheets*nRows*sizeof(float));

			break;
		case 5:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_weightedSums, nCols*nSheets*sizeof(float));

			break;
		case 6:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_weightedSums, nCols*nSheets*sizeof(float));

			break;
		default:
			cerr << "[ error ] '" << projType << "' is not a valid projection type, please select from [1,6]" << endl;
			delete [] rawImageData;
			return -1;
	}

	/*
	 * Retrieve results
	 */
	validate(cudaMemcpy(h_maxImage, d_maxImage, resultSize, cudaMemcpyDeviceToHost)); 
	validate(cudaMemcpy(h_sumImage, d_sumImage, resultSize, cudaMemcpyDeviceToHost)); 

	/*
	 * Write results
	 */
	writeImage(argv[6] + std::string("_max.png"), h_maxImage, projType, nRows, nCols, nSheets);
	writeImage(argv[6] + std::string("_sum.png"), h_sumImage, projType, nRows, nCols, nSheets);

	/*
	 * Clean up
	 */
	cudaFree(d_maxImage);
	cudaFree(d_sumImage);
	cudaFree(d_weightedSums);
	cudaFree(d_globalMax);

	delete [] rawImageData;
	delete [] h_maxImage;
	delete [] h_sumImage;

	return 0;
}
