#include <iostream>
#include <cassert>
#include <fstream>
#include "ImageWriter.h"
#include <cuda.h>
#include "helpers.h"
#include "kernels.h"
#include <cmath>

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
void __global__ maxKernel(unsigned char *voxels, unsigned char *maxImage, float *localMaxes, float *globalMax, int nRows, int nCols, int nSheets){

	/*
	 * Since the data is in column-major order, the thread IDs correspond
	 * to the initial position in the first column of the first sheet.
	 * That is, each thread will have a different row, and that row will
	 * be constant across its work.
	 */
	int myRow = blockDim.x * blockIdx.x + threadIdx.x;
	int size = nRows*nCols;
	float norm = 1.0/nSheets;
	unsigned char val = 0;
	unsigned char localMax = 0;

	// In case we got rounded up, make sure we have a row to work on.
	if(myRow < nRows){

		/*
		 * The data is stored in column-major order, so we
		 * need to ensure that SIMD threads are accessing the
		 * data in the same way to exploit coalesced memory
		 * access.
		 *
		 * Thus we want each thread to be responsible for a
		 * row, and iterate over all columns, all sheets in
		 * that row.  Effectively, each thread will handle
		 * collapsing the slice formed by its row across all
		 * the sheets
		 *
		 * Each thread starts its work at the 0th column, in
		 * its row, which is naturally aligned to the thread
		 * index if we use a 1D kernel.
		 *
		 * Each thread will increment over columns, each
		 * column starting nRows from the start of the previous one.
		 *
		 * For each column, each thread will increment over all
		 * sheets, in increments of nRows*nCols.
		 *
		 * The position each thread is responsible for is determined
		 * from the rows and columns; Each thread has a fixed row, and
		 * as we progress through the columns the column is updated.
		 * We follow the column-major convention, so the position is
		 * determined as thisCol*nRows + thisRow.  The same is used to
		 * store the maximum for each position in the final image, for
		 * use in the sum kernel
		 *
		 */

		// TODO this is slightly off, I think I'm not getting all the
		// data.  Rewrite in terms of row, col, sheet and go from
		// there.

		for(int curPos = myRow; curPos < size; curPos += nRows){

			for(int sh = 0; sh < nSheets; ++sh){
				val = voxels[curPos + sh*size];

				// Fill in work for the max image
				if((int) val > (int)localMax){
					localMax = val;
				}
				//localMax = (int)localMax > (int)val ? localMax : val;

				// Fill in work for the sum image, the running
				// weighted sum along the collapsed dimension
				localMaxes[curPos] += norm * ((1 + sh)*(int)val);
			}

			// Fill in maxImage for this position
			maxImage[curPos] = localMax;

			// Adjust highest weighted sum seen if necessary
			globalMax[0] = globalMax[0] > localMaxes[curPos] ? globalMax[0] : localMaxes[curPos];
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
	unsigned char *d_sumImage, *h_sumImage;
	int resultSize;
	float *d_localMaxes;
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

	// Issue kernels
	switch(projType){
		case 1:	// Note: need braces to restrict scope of the local variables
			{
				cout << "Projection type " << projType << endl;
				resultSize = nCols*nRows*sizeof(unsigned char);
				h_maxImage = new unsigned char[nCols*nRows];
				cudaMalloc((void **)&d_maxImage, resultSize);
				cudaMalloc((void **)&d_sumImage, resultSize);
				cudaMalloc((void **)&d_localMaxes, nCols*nRows*sizeof(float));
				cudaMalloc((void **)&d_globalMax, sizeof(float));
			
				/*
				 * On selecting sizes
				 * 
				 * We really don't have enough work to make
				 * great use of the GPU here.  Ideally, we
				 * want enough blocks such that each SM has
				 * several blocks to run, some multiple of the
				 * number of schedulers.  However, we also
				 * need to balance the divison of work such
				 * that we don't incur too much overhead
				 * managing the blocks.  In this case, there
				 * isn't much work to be done, so the overhead
				 * is relatively large to the workload.
				 */

				// Threads per block should be a multiple of warp size
				int warpsize = 32;

				//  Number of warp schedulers is important
				int numWarpSchedulers = 2;

				// Cannot exceed this
				int maxThreadsPerSM = 1536;

				// We want each thread to do one row so it has
				// sufficient work
				int numThreads = nRows;

				// We have little internal memory and no
				// shared memory, so these are irrelevant
				
				// first approximation
				int threadsPerBlock = warpsize * numWarpSchedulers;
				
				// Bounds check
				threadsPerBlock = threadsPerBlock <= maxThreadsPerSM ? threadsPerBlock : maxThreadsPerSM;
					
				// Total number of threads divided by threads
				// per block dictates blocks per grid.
				int blocksPerGrid = ceil(numThreads/threadsPerBlock);

				cout << "Launching kernel..." << endl;
				cout << "Total threads: " << numThreads << endl;
				cout << "Threads per block: " << threadsPerBlock << endl;
				cout << "Number of blocks: " << blocksPerGrid << endl;

				maxKernel<<<blocksPerGrid, threadsPerBlock>>>(rawImageData, d_maxImage, d_localMaxes, d_globalMax, nRows, nCols, nSheets);
			}
			break;
		case 2:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMaxes, nCols*nRows*sizeof(float));

			break;
		case 3:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMaxes, nSheets*nRows*sizeof(float));

			break;
		case 4:
			cout << "Projection type " << projType << endl;
			resultSize = nSheets*nRows*sizeof(unsigned char);
			h_maxImage = new unsigned char[nSheets*nRows];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMaxes, nSheets*nRows*sizeof(float));

			break;
		case 5:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMaxes, nCols*nSheets*sizeof(float));

			break;
		case 6:
			cout << "Projection type " << projType << endl;
			resultSize = nCols*nSheets*sizeof(unsigned char);
			h_maxImage = new unsigned char[nCols*nSheets];
			cudaMalloc((void **)&d_maxImage, resultSize);
			cudaMalloc((void **)&d_sumImage, resultSize);
			cudaMalloc((void **)&d_localMaxes, nCols*nSheets*sizeof(float));

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
	writeImage(argv[6] + std::string("_max.png"), h_maxImage, projType, nRows, nCols, nSheets);
	//writeImage("sum.png", h_sumImage, projType, nRows, nCols, nSheets);
	

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	cudaFree(d_maxImage);
	cudaFree(d_sumImage);
	cudaFree(d_localMaxes);
	cudaFree(d_globalMax);

	return 0;
}
