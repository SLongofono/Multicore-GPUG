#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include "ImageReader.h"

using namespace std;



int main(int argc, char **argv){
	
	int numNodes, rank;
	numNodes = argc-1;	// Implicit number of nodes is 12
	assert(numNodes == 12);	
	cryph::Packed3DArray<unsigned char> *arr;
	int dims[2];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	MPI_Request rq;

	if(rank){
		// We are not the root
		cout << "NODE " << rank << endl;

		// Wait for dimensions information in a tag 0 message
		MPI_Recv(dims, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		
		int rawData[dims[0]*dims[1]*3];

		// Wait for image data in a tag 1 message
		MPI_Recv(rawData, dims[0]*dims[1]*3, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);

	}
	else{
		// We are the root
		cout << "ROOT " << rank << endl;

		for(int i = 2; i<argc; ++i){
			cout << argv[i] << endl;

			// Prepare a packed array from the image
			string s = argv[i];
			ImageReader * ir = ImageReader::create(s);
			assert(nullptr != ir);
			arr = ir->getInternalPacked3DArrayImage();
			assert(nullptr != arr);

			// Send a message indicating the dimensions
			dims[0] = arr->getDim1();
			dims[1] = arr->getDim2();
			MPI_Isend(dims, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &rq);

			// Flatten the data for transit
			unsigned char flatpack[dims[0]*dims[1]*2];
			for(int i = 0; i<dims[0]; ++i){
				for(int j =0; j<dims[1]; ++j){
					for(int k = 0; k<3; ++k){
					// 3D indexing: i*rows*cols + j*cols + k
						flatpack[(i*dims[0]*dims[1]) + (j*dims[1]) + k ] = arr->getDataElement(i,j,k);
					}
				}
			}

			// Fire it off to the appropriate node
			MPI_Isend(flatpack, dims[0]*dims[1]*3, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, &rq);
		}

		// Do our work while the rest complete

	}

	

	MPI_Finalize();

}
