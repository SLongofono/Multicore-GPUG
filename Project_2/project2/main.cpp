#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include "ImageReader.h"

using namespace std;

int main(int argc, char **argv){
	
	int numNodes, rank;
	cryph::Packed3DArray<unsigned char> *arr;
	int dims[2];

	string s = "doop.txt";

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

		cout << "NODE " << rank << " got image..." << endl;

	}
	else{
		// We are the root
		cout << "ROOT " << rank << endl;

		for(int i = 1; i < numNodes; ++i){
			cout << argv[i+1] << endl;

			// Prepare a packed array from the image
			ImageReader * ir = ImageReader::create(argv[i]);
			assert(nullptr != ir);
			arr = ir->getInternalPacked3DArrayImage();
			assert(nullptr != arr);

			// Send a message indicating the dimensions
			dims[0] = arr->getDim1();
			dims[1] = arr->getDim2();
			MPI_Isend(dims, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &rq);

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

			cout << "Root sending " << sizeof(flatpack) << " chars to rank " << i << endl;

			// Fire it off to the appropriate node
			MPI_Isend(flatpack, dims[0]*dims[1]*3, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, &rq);
		}

		// Do our work while the rest complete

	}

	

	MPI_Finalize();

}
