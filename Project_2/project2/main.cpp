#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include "ImageReader.h"

using namespace std;


// Populate destination with a normalized count of pixel weights (0-255) along dim3,
// representing the percentage of pixels with each weight in a single color
// channel.  Note we pass by reference since heap allocation will change the
// address used in virtual memory
void normalized_count(float *&dest, unsigned char *data, int dim1, int dim2, int dim3){
	dest = new float[256];
	int val;
	float norm;
	int counts[256] = {0};
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			val = data[i*dim2*3 + j*3 + dim3];
			counts[val]++;
		}
	}
	norm = dim1*dim2;
	float sanity = 0.0;
	for(int k = 0; k<256; ++k){
		dest[k] = counts[k]/norm;
		sanity += dest[k];
	}
	cout << "SANITY CHECK: channel percentage sums to " << sanity*100 << endl;
}

int max_report(float *data){
	float currmax = *data;
	int curpos = 0;
	for(int i = 0; i<256; ++i){
		if(data[i] > currmax){
			currmax = data[i];
			curpos = i;
		}
	}
	return curpos;
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
		float * rchannel;
		
		normalized_count(rchannel, rawData, dims[0], dims[1], 0);
		int maxpos = max_report(rchannel);
		cout << "NODE " << rank << " red channel has highest percentage of " << rchannel[maxpos] << " at value " << maxpos << endl;

		// All-to-all gather to get results from other ranks

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

			cout << "Root flattening image..." << endl;

			// Flatten the data for transit
			unsigned char flatpack[dims[0]*dims[1]*3];
			for(int i = 0; i<dims[0]; ++i){
				for(int j =0; j<dims[1]; ++j){
					for(int k = 0; k<3; ++k){
					// 3D indexing: i*rows*cols + j*cols + k
						int arg = (i*dims[1]*3) + (j*3) + k;
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
					int arg = (i*dims[1]*3) + (j*3) + k;
					assert(arg <= sizeof(flatpack));
					flatpack[arg] = arr->getDataElement(i,j,k);
				}
			}
		}


		// Compute our histogram
		
		// All-to-all gather to get histograms from other ranks
		
		// Determine the most similar image and report
		
		// Catch and report results from each other rank

	}

	

	MPI_Finalize();

}
