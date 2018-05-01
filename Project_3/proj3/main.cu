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


/*
 * The max image kernel
 *
 * Each thread will collapse one pixel of the result matrix.  This apporach
 * allows the collective memory accesses to happen in a columnwise fashion and
 * be coalesced by the GPU memory manager.
 */
void __global__ maxKernel(unsigned char *voxels, unsigned char *maxImage, float *weightedSums, float *globalMax, int nRows, int nCols, int nSheets){

	/*
	 * Since the data is in column-major order, the thread IDs correspond
	 * to the initial position in the first column of the first sheet.
	 * That is, each thread will have a different row, and that row will
	 * be constant across its work.
	 */
	int myRow = blockDim.x * blockIdx.x + threadIdx.x;
	int myMaxPos = nRows*nCols + myRow;
	float norm = 1.0/nSheets;
	unsigned char val = (unsigned char)0;
	unsigned char localMax = (unsigned char)0;
	int size = nRows*nCols;

	int outputSize = size;
	int voxelSize = nRows*nCols*nSheets;

	// In case we got rounded up, make sure we have a row to work on.
//	if(myRow < nRows){
	if(myRow == 0){

#if 0
		// Do some sanity checks
		//Verify we can access every voxel
		for(int i = 0; i<nRows; ++i){
			for(int j = 0; j<nCols; ++j){
				for(int k = 0; k<nSheets; ++k){
					int arg  = 	k*nRows*nCols +
							j*nRows +
							i;
					int temp = voxels[arg];
					printf("Voxel %d OK. ", arg);
				}
			}
		}

		printf("We can access all voxels...\n");

		// Verify we can write every piece of the output image
		for(int i = 0; i<nRows; ++i){
			for(int j = 0; j<nCols; ++j){
				int arg = j*nRows + i;
				maxImage[arg] = 0;
				printf("Output %d OK/ ",arg);
			}
		}

		printf("We can access all pieces of the output image...\n");

		// Verify we can write every piece of the weighted sums
		for(int i = 0; i<nRows; ++i){
			for(int j = 0; j<nCols; ++j){
				int arg = j*nRows + i;
				weightedSums[arg] = 0.0;
				printf("Weighted sum %d OK. ", arg);
			}
		}
	
		printf("We can access all pieces of the weighted sums array...\n");
		printf("TESTS PASSED!\n");
#endif
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
		for(int c = 0; c < nCols; ++c){


		//for(int curPos = myRow; curPos < myMaxPos; curPos += nRows){
			// Current position wrt output image
			int curPos = myRow + c*nRows;

			if(curPos >= size){
				printf("ERROR!  curPos %d is invalid for limit %d!\n", curPos, size);
			}
			printf("Working on position (%d, %d)\n", myRow, curPos / nRows);

			printf("A!\n");

			// Tracking weighted sum
			weightedSums[curPos] = 0.0;
			printf("B!\n");

			for(int sh = 0; sh < nSheets; ++sh){
			
				if( (curPos + sh*size) >= voxelSize){
					printf("ERROR! voxel index %d is invalid for limit %d!\n", curPos+sh*size, voxelSize);
					
				}
				printf("Voxel index: %d, max %d\n", curPos + sh*nRows*nCols, voxelSize);

				val = voxels[curPos + sh*nRows*nCols];
				printf("C!\n");

				if(val > localMax){
					localMax = val;
				}

				// Fill in work for the sum image, the running
				// weighted sum along the collapsed dimension
				weightedSums[curPos] += norm * ((1 + sh)*(int)val);
				printf("D!\n");
			}

			if(myRow == 0){
				printf("Sanity check: curpos is %d, next curPos is %d, maxCurPos (non inclusive) is %d\n", curPos, curPos + nRows, myMaxPos);
			}
			printf("E!\n");

			// Fill in maxImage for this position
			maxImage[curPos] = localMax;
			printf("F!\n");

			localMax = 0;
			
			printf("G!\n");
			// Adjust highest weighted sum seen if necessary
			atomicMax(globalMax, weightedSums[curPos]);
			printf("H!\n");
		}
	}
	printf("KERNEL SUCCESS, THREAD %d\n", myRow);
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
	validate(
		cudaMalloc((void **)&d_voxels, nVals*sizeof(unsigned char))
	);

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
				validate(cudaMalloc((void **)&d_maxImage, resultSize));
				validate(cudaMalloc((void **)&d_sumImage, resultSize));
				validate(cudaMalloc((void **)&d_weightedSums, nCols*nRows*sizeof(float)));
				validate(cudaMalloc((void **)&d_globalMax, sizeof(float)));
			
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

				maxKernel<<<blocksPerGrid, threadsPerBlock>>>(rawImageData, d_maxImage, d_weightedSums, d_globalMax, nRows, nCols, nSheets);
				validate(cudaPeekAtLastError());
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
	//cudaMemcpy(h_sumImage, d_sumImage, resultSize, cudaMemcpyDeviceToHost); 

	/*
	 * Write results
	 */
	//writeImage(argv[6] + std::string("_max.png"), h_maxImage, projType, nCols, nRows,nSheets);
	writeImage(argv[6] + std::string("_max.png"), h_maxImage, projType, nRows, nCols, nSheets);
	//writeImage("sum.png", h_sumImage, projType, nRows, nCols, nSheets);
	

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	cudaFree(d_maxImage);
	cudaFree(d_sumImage);
	cudaFree(d_weightedSums);
	cudaFree(d_globalMax);

	return 0;
}
