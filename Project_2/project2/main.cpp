#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "ImageReader.h"

#define DIMS_MSG_TAG 0
#define DATA_MSG_TAG 1
#define DEBUG 0

using namespace std;

#if DEBUG

#include <cstdio>	// for string_format()
#include <memory>	// for string_format()
#include <fstream>	// for dump()


// Realizes sprintf() since C++ insists on making standard things complicated
// Thanks to stackOverflow user IFreilicht for this workaround.
// https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


// dumps packed channel percentages to a predefined filename
void dump(string fname, float *data){
	ofstream chan(fname);
	if(chan.good()){
		for(int i = 0; i<256*3; i+=3){
			chan << i/3 << ": (" << data[i] << "," << data[i+1] << "," << data[i+2] << ")" << endl;
		}
		chan.close();
	}
}

// dumps serialized counts
void serial_dump(string fname, int *data, int len){
	cout << "DUMPING..." << endl;
	ofstream chan(fname);
	if(chan.good()){
		for(int i = 0; i<len; i+=3){
			chan << i/3 << ":" << "(" << data[i] << "," << data[i+1] << "," << data[i+2] << ")" << endl;
		}
		chan.close();
	}
}



// dumps packed counts
void dump(int rank, int *data, int color){
	ofstream chan;
	if(0 == color){
		chan.open(string_format("node_%d_red.txt", rank));
	}
	else if (1 == color){
		chan.open(string_format("node_%d_green.txt", rank));
	}
	else{
		chan.open(string_format("node_%d_blue.txt", rank));
	}
	if(chan.good()){
		for(int i = 0; i<256; ++i){
			chan << i << ":" << data[i] << endl;
		}
		chan.close();
	}
}

void check_bounds(int n, int i, int j, int dim1, int dim2){
	if(n < 0 || n >= 256){
		cerr 	<< "ERROR: arg is " << n << ", i is " << i
			<< ", j is " << j << ", dim1 is " << dim1
			<< ", dim2 is " << dim2 << endl;
		assert(0);
			
	}
}

#endif


/*
 * Computes a histogram representing the percentage of pixels at each of the
 * 256 possible color values.  Uses the packing convention
 *     (r0,g0,b0,r1,g1,b1...) to represent the three color channels
 */
float * histogram(int *data, int dim1, int dim2, int rank){
	float* ret = new float[256*3];
	int arg;
	int red[256] = {0};
	int green[256] = {0};
	int blue[256] = {0};

	for(int i = 0; i<dim1*dim2*3; i+=3){
		red[ data[i] ]++;
		blue[ data[i+1] ]++;
		green[ data[i+2] ]++;
	}

#if DEBUG
	dump(rank, red, 0);
	dump(rank, green, 1);
	dump(rank, blue, 2);
	int sum = 0;
	for(int i = 0; i<256; ++i){
		sum += red[i];
	}
	assert(sum == dim1*dim2);
#endif
	float norm = 1.0/(dim1*dim2);

	arg = 0;
	// Normalize and assign
	for(int j = 0; j<256; ++j){
		ret[arg] = norm * red[j];
		ret[arg + 1] = norm * green[j];
		ret[arg + 2] = norm * blue[j];
		arg += 3;
	}
#if DEBUG
	// Ensure this is a percentage
	float rsum, gsum, bsum;
	rsum=gsum=bsum=0.0;
	for(int i = 0; i<256*3; i+=3){
		rsum += ret[i];
		gsum += ret[i+1];
		bsum += ret[i+2];
	}
	if(rsum <= 0.9999 || rsum > 1.0001){
		cout << "[ error ] NODE " << rank << ": red percentages sum to " << rsum << endl;
	}
	if(gsum <= 0.9999 || gsum > 1.0001){
		cout << "[ error ] NODE " << rank << ": green percentages sum to " << gsum << endl;
		
	}
	if(bsum <= 0.9999 || bsum > 1.0001){
		cout << "[ error ] NODE " << rank << ": blue percentages sum to " << bsum << endl;
	}
#endif
	return ret;
	
}


/*
 * Compares the 256x(r,g,b) packed histogram in myHisto to the numNodes other
 * packed histograms in otherHistos to determine the most similar image.  In
 * this case, most similar means minimize the piecewise difference of the
 * histogram elements.  The histogram data has 3 channels of 256 values,
 * packed in triplets (r0,g0,b0,r1,g1,b1,r2,g2,b2...).
 */
int compare_histos(float *myHisto, float *otherHistos, int myRank, int numNodes){
	float currDiff = 0.0;
	float minDiff = 10000000.0;
	float lowRank;

	lowRank = myRank == 0 ? 1:0;

	// For each rank in the world...
	for(int n = 0; n < numNodes; ++n){

#if DEBUG
		// verify that we still have a percentage
		float rsum, gsum, bsum;
		rsum=gsum=bsum=0.0;
		for(int i = 0; i<256*3; i+=3){
			rsum += myHisto[i];
			gsum += myHisto[i+1];
			bsum += myHisto[i+2];
		}
		if(rsum <= 0.9999 || rsum > 1.0001){
			cout << "[ error ] NODE " << myRank << ": red percentages sum to " << rsum << endl;
			assert(0);
		}
		if(gsum <= 0.9999 || gsum > 1.0001){
			cout << "[ error ] NODE " << myRank << ": green percentages sum to " << gsum << endl;	
			assert(0);
		}
		if(bsum <= 0.9999 || bsum > 1.0001){
			cout << "[ error ] NODE " << myRank << ": blue percentages sum to " << bsum << endl;
			assert(0);
		}
#endif

		currDiff = 0.0;
		if(n != myRank){
			// Since they are all packed the same way, compare
			// elementwise
			for(int i = 0; i<256*3; ++i){
				currDiff += abs( otherHistos[n*256*3 + i] - myHisto[i] );
			}
			if(currDiff < minDiff){
				lowRank = n;
				minDiff = currDiff;
			}
			cout << "NODE " << myRank << ": difference between me and node " << n << " is " << currDiff << endl;
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
		MPI_Recv(dims, 2, MPI_INT, 0, DIMS_MSG_TAG, MPI_COMM_WORLD, &status);
		
		// On heap since stack will overflow for many images
		int *rawData = new int[dims[0]*dims[1]*3];

		// Wait for image data in a tag 1 message
		MPI_Recv(rawData, dims[0]*dims[1]*3, MPI_INT, 0, DATA_MSG_TAG, MPI_COMM_WORLD, &status);

#if DEBUG
		cout << "NODE " << rank << ": dimensions: (" << dims[0] << "," << dims[1] << ")" << endl;
		cout << "NODE " << rank << ": got image with " << dims[0]*dims[1] << " pixels..." << endl;
		serial_dump(string_format("node_%d_after.txt", rank), rawData, dims[0]*dims[1]);
		cout << "NODE " << rank << ": computing histogram..." << endl;
#endif


		// Compute histograms
		float *channels = histogram(rawData, dims[0], dims[1], rank);

#if DEBUG
		dump(string_format("NODE_%d_CHANNELS.txt", rank), channels);
#endif
		//normalized_count(channels, rawData, dims[0], dims[1]);

		// All-to-all gather to get results from other ranks
		// Use blocking since we need everyone to be done before we
		// start to compare
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];
		
		// fill in my data
		for(int i = 0; i<256*3; ++i){
			histos[256*3*rank + i] = channels[i];
		}
	
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);
		// Without in place MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

#if DEBUG
		// Sanity check.  My data should start at 256*3*myRank
		int offset = 256 * 3 * rank;
		for(int k = 0; k<256*3; ++k){
			assert(channels[k] == histos[offset + k]);
		}

#endif

		// Determine the most similar rank
		int closest = compare_histos(channels, histos, rank, numNodes);

		// Report closest image back to root.  Need to wait so data
		// still exists
		MPI_Gather(&closest, 1, MPI_INT, nullptr, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);

		// Clean up
		delete [] channels;
		delete [] rawData;

		


	}
	else{
		// We are the root

		for(int n = 1; n < numNodes; ++n){
			cout << "ROOT: sending " << argv[n+1] << " to rank " << n << endl;

			// Prepare a packed array from the image
			ImageReader *ir = ImageReader::create(argv[n+1]);
			assert(nullptr != ir);
			arr = ir->getInternalPacked3DArrayImage();
			assert(nullptr != arr);


			// Send a message indicating the dimensions
			dims[0] = arr->getDim1();
			dims[1] = arr->getDim2();
			MPI_Isend(dims, 2, MPI_INT, n, DIMS_MSG_TAG, MPI_COMM_WORLD, &rq);

			// Flatten the data for transit
			// Use heap since the image may be large
			int *flatpack = new int[dims[0]*dims[1]*3];
			for(int i = 0; i<dims[0]; ++i){
				for(int j =0; j<dims[1]; ++j){
					for(int k = 0; k<3; ++k){
						// 3D indexing: i*cols*size + j*size + k
						int arg = (i*dims[1]*3) + (j*3) + k;
						flatpack[arg] = (int)arr->getDataElement(i,j,k);
					}
				}
			}
#if DEBUG
			serial_dump(string_format("node_%d_before.txt", n), flatpack, dims[0]*dims[1]);
#endif



			// Fire it off to the appropriate node.  Since it is
			// on the heap, use blocking to make sure it gets
			// there before deleting it.

			MPI_Send(flatpack, dims[0]*dims[1]*3, MPI_INT, n, DATA_MSG_TAG, MPI_COMM_WORLD);
			//MPI_Isend(flatpack, dims[0]*dims[1]*3, MPI_INT, n, DATA_MSG_TAG, MPI_COMM_WORLD, &rq);

			delete [] flatpack;
		}

		// Do our work while the rest complete

		// Prepare our image
		ImageReader *ir = ImageReader::create(argv[1]);
		arr = ir->getInternalPacked3DArrayImage();
		dims[0] = arr->getDim1();
		dims[1] = arr->getDim2();
		int *flatpack = new int[dims[0]*dims[1]*3];
		for(int i = 0; i<dims[0]; ++i){
			for(int j =0; j<dims[1]; ++j){
				for(int k = 0; k<3; ++k){
				// 3D indexing: i*cols*size + j*size + k
					int arg = (i*dims[1]*3) + (j*3) + k;
					flatpack[arg] = (int)arr->getDataElement(i,j,k);
				}
			}
		}

		// Compute our histogram
		float *channels = histogram(flatpack, dims[0], dims[1], 0);
		//normalized_count(channels, flatpack, dims[0], dims[1]);
		
#if DEBUG
		dump(string_format("NODE_%d_CHANNELS.txt", 0), channels);
#endif
		// All-to-all gather to get histograms from other ranks
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];

		// Copy my data into the appropriate area (since local work is
		// less expensive than communication)
		for(int i = 0; i<256*3; ++i){
			histos[i] = channels[i];
		}
		
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);
		//   Without in place: MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

#if DEBUG
		// Sanity check: my data should be at rank*256*3
		for(int i = 0; i<256*3; ++i){
			assert(channels[i] == histos[i + rank*256*3]);
		}
#endif

		// Declare buffer to catch results
		int results[numNodes];

		// Determine the most similar image and fill in root results
		results[0] = compare_histos(channels, histos, rank, numNodes);

		// Catch and report results from each other rank
		// Do so in a blocking fashon, since we need to report results
		// to the user
		MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, results, 1, MPI_INT, 0, MPI_COMM_WORLD);

		for(int i = 0; i<numNodes; ++i){
			cout << "NODE " << i << ": most similar to NODE " << results[i] << endl;
		}

		// Clean up
		delete [] channels;
		delete [] flatpack;
	}

	

	MPI_Finalize();

}
