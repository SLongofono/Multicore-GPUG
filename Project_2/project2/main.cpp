#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "ImageReader.h"

using namespace std;

/*
 * Populate destination with a normalized count of pixel weights (0-255) along dim3,
 * representing the percentage of pixels with each weight in each color
 * channel.  Following the convention of the Packed3DArray type, the data are
 * packed in triplets (r1,g1,b1,r2,g2,b3...).  Note we pass by reference since
 * heap allocation will change the address used in virtual memory
 */
void normalized_count(float *&dest, unsigned char *data, int dim1, int dim2){
	dest = new float[256*3];
	int arg;
	float norm = dim1 * dim2;
	
	// Accumulate pixel value counts
	int rcounts[256] = {0};
	int gcounts[256] = {0};
	int bcounts[256] = {0};

	for(int i = 0; i < dim1; ++i){
		for(int j = 0; j< dim2; ++j){
			arg = i*dim2*3 + j*3;
			rcounts[data[arg]]++;
			gcounts[data[arg + 1]]++;
			bcounts[data[arg + 2]]++;
		}
	}

	// Assign normalized percentage to each pixel value, in packed order
	// in destination array
	for(int i = 0, j=0; i < 256; ++i, j = j + 3){
		dest[j] = rcounts[i] / norm;
		dest[j+1] = rcounts[i] / norm;
		dest[j+2] = rcounts[i] / norm;
	}

}


/*
 * Compares the 256x(r,g,b) packed histogram in myHisto to the numNodes other
 * packed histograms in otherHistos to determine the most similar image.  In
 * this case, most similar means minimize the piecewise difference of the
 * histogram elements.  The histogram data has 3 channels of 256 values,
 * packed in triplets (r1,g1,b1,r2,g2,b2...).
 */
int compare_histos(float *&myHisto, float *&otherHistos, int myRank, int numNodes){
	float currDiff, minDiff=10000000.0;
	float lowRank;

	lowRank = myRank == 0 ? 1:0;

	// For each rank in the world...
	for(int n = 0; n < numNodes; ++n){
		if(n != myRank){
			// For each of my colors
			for(int i = 0; i<256; ++i){
				for(int j = 0; j < 3; ++j){
					currDiff += abs( otherHistos[n*256*3 + i*3 + j] - myHisto[i*3 + j] );
				}
			}
			if(currDiff < minDiff){
				lowRank = n;
				minDiff = currDiff;
			}
		}
	}
	return lowRank;
}


int main(int argc, char **argv){
	
	int numNodes, rank;
	cryph::Packed3DArray<unsigned char> *arr;
	int dims[2];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
	MPI_Status status;
	MPI_Request rq;

	if(rank){
		// We are not the root

		// Wait for dimensions information in a tag 0 message
		MPI_Recv(dims, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		cout << "NODE " << rank << " dimensions: (" << dims[0] << "," << dims[1] << ")" << endl;
			
		unsigned char rawData[dims[0]*dims[1]*3];

		// Wait for image data in a tag 1 message
		MPI_Recv(rawData, sizeof(rawData), MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);

		cout << "NODE " << rank << " got image with " << dims[0]*dims[1] << " pixels..." << endl;


		// Compute histograms
		float *channels;
		normalized_count(channels, rawData, dims[0], dims[1]);

		// All-to-all gather to get results from other ranks
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];
		MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

		// Determine the most similar rank, and report back to rank 0


	}
	else{
		// We are the root
		cout << "ROOT " << rank << endl;

		for(int n = 1; n < numNodes; ++n){
			cout << argv[n+1] << endl;

			// Prepare a packed array from the image
			ImageReader *ir = ImageReader::create(argv[n+1]);
			assert(nullptr != ir);
			arr = ir->getInternalPacked3DArrayImage();
			assert(nullptr != arr);

			// Send a message indicating the dimensions
			dims[0] = arr->getDim1();
			dims[1] = arr->getDim2();
			MPI_Isend(dims, 2, MPI_INT, n, 0, MPI_COMM_WORLD, &rq);

			// Flatten the data for transit
			unsigned char flatpack[dims[0]*dims[1]*3];
			for(int i = 0; i<dims[0]; ++i){
				for(int j =0; j<dims[1]; ++j){
					for(int k = 0; k<3; ++k){
					// 3D indexing: i*rows*cols + j*cols + k
						size_t arg = (i*dims[1]*3) + (j*3) + k;
						assert(arg <= sizeof(flatpack));
						flatpack[arg] = arr->getDataElement(i,j,k);
					}
				}
			}

			// Fire it off to the appropriate node
			MPI_Isend(flatpack, dims[0]*dims[1]*3, MPI_UNSIGNED_CHAR, n, 1, MPI_COMM_WORLD, &rq);
		}

		// Do our work while the rest complete

		// Prepare our image
		ImageReader *ir = ImageReader::create(argv[1]);
		arr = ir->getInternalPacked3DArrayImage();
		dims[0] = arr->getDim1();
		dims[1] = arr->getDim2();
		unsigned char flatpack[dims[0]*dims[1]*3];
		for(int i = 0; i<dims[0]; ++i){
			for(int j =0; j<dims[1]; ++j){
				for(int k = 0; k<3; ++k){
				// 3D indexing: i*rows*cols + j*cols + k
					size_t arg = (i*dims[1]*3) + (j*3) + k;
					assert(arg <= sizeof(flatpack));
					flatpack[arg] = arr->getDataElement(i,j,k);
				}
			}
		}


		// Compute our histogram
		float *channels;
		normalized_count(channels, flatpack, dims[0], dims[1]);	

		// All-to-all gather to get histograms from other ranks
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];
		MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

		// Determine the most similar image and report
		
		// Catch and report results from each other rank

	}

	

	MPI_Finalize();

}
