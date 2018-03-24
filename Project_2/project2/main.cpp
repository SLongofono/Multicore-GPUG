#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "ImageReader.h"
#include <cstdio>
#include <memory>
#include <fstream>
#define DIMS_MSG_TAG 0
#define DATA_MSG_TAG 1
#define DEBUG 0

using namespace std;

// For testing only

#if DEBUG
#define COUNT 0
#endif

#define TEST4 1


template<typename ... Args>
string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


// dumps packed counts
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

// dumps channels
void dump(int rank, float *channels){
	ofstream rchan(string_format("node_%d_red.txt", rank));
	ofstream gchan(string_format("node_%d_green.txt", rank));
	ofstream bchan(string_format("node_%d_blue.txt", rank));
	if(rchan.good() && gchan.good() && bchan.good()){
		for(int i = 0; i<256*3; i += 3){
			rchan << channels[i] << endl;
			gchan << channels[i + 1] << endl;
			bchan << channels[i + 2] << endl;
		}
		rchan.close();
		gchan.close();
		bchan.close();
	}
}

void stats(int rank, float *channels){
	// Compute some stats for debugging purposes
	float rsum, gsum, bsum, rmax, gmax, bmax, rmin, gmin, bmin, rval, gval, bval;
	int rmaxpos, gmaxpos, bmaxpos, rminpos, gminpos, bminpos;
	rmaxpos = gmaxpos = bmaxpos = rminpos = gminpos = bminpos = 0;
	rsum = bsum = gsum = rmax = gmax = bmax = 0.0;
	rmin = gmin = bmin = 2;
	for(int triplet = 0; triplet < 256*3; triplet += 3){
		rval = channels[triplet];
		gval = channels[triplet + 1];
		bval = channels[triplet + 2];

		rsum += rval;
		gsum += gval;
		bsum += bval;

		if(rval < rmin){ rmin = rval; rminpos = triplet/3;}
		if(gval < gmin){ gmin = gval; gminpos = triplet/3;}
		if(bval < bmin){ bmin = bval; bminpos = triplet/3;}

		if(rval > rmax){ rmax = rval; rmaxpos = triplet/3;}
		if(gval > gmax){ gmax = gval; gmaxpos = triplet/3;}
		if(bval > bmax){ bmax = bval; bmaxpos = triplet/3;}
		
	}

	cout 	<< "NODE " << rank << ":" << endl << '\t'
		<< "Sums: " << "(" << rsum << " " << gsum << " " << bsum << ")" << endl <<'\t'
		<< "Maxs: " << "(" << rmaxpos << " " << gmaxpos << " " << bmaxpos << ")" << endl <<'\t'
		<< "Mins: " << "(" << rminpos << " " << gminpos << " " << bminpos << ")" << endl;

}

/*
int * count(int *data, int dim1, int dim2, int dim3){
	int *ret = new int[256]();

	for(int i = 0; i<256; ++i){
		assert(ret[i] == 0);
	}

	int arg;
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			arg = i*dim2*3 + j*3 + dim3;
			ret[data[arg]]++;
		}
	}
	return ret;
}	
*/

void check_bounds(int n, int i, int j, int dim1, int dim2){
	if(n < 0 || n >= 256){
		cerr 	<< "ERROR: arg is " << n << ", i is " << i
			<< ", j is " << j << ", dim1 is " << dim1
			<< ", dim2 is " << dim2 << endl;
		assert(0);
			
	}
}

float * histogram(int *data, int dim1, int dim2, int rank){
	float* ret = new float[256*3];
	int arg = 0;
	int red[256] = {0};
	int green[256] = {0};
	int blue[256] = {0};

	for(int i = 0; i<dim1*dim2; i+=3){
		red[ data[arg] ]++;
		blue[ data[arg+1] ]++;
		green[ data[arg+2] ]++;
	}

	/*
	for(int i = 0; i<256*3; i+=3){
		red[ data[i] ]++;
		green[ data[i + 1] ]++;
		blue[ data[i + 2 ] ]++;
	}
	*/

	/*
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			check_bounds(data[arg], i, j, dim1, dim2);
			check_bounds(data[arg+1], i, j, dim1, dim2);
			check_bounds(data[arg+2], i, j, dim1, dim2);

			arg = i*dim2*3 + j*3;
			red[ data[arg] ]++;
			green[ data[arg+1] ]++;
			blue[ data[arg+2] ]++;
		}
	}
	*/

	//red = count(data, dim1, dim2, 0);
	//green = count(data, dim1, dim2, 1);
	//blue = count(data, dim1, dim2, 2);
#if DEBUG
	dump(rank, red, 0);
	dump(rank, green, 1);
	dump(rank, blue, 2);
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
	

	/*	
	for(int i=0, j=0; i<256; ++i, j +=3){
		ret[j] = red[i]*norm;
		ret[j+1] = green[i]*norm;
		ret[j+2] = blue[i]*norm;
	}
	*/

	//delete [] red;
	//delete [] green;
	//delete [] blue;

	return ret;
	
}

/*
 * Populate destination with a normalized count of pixel weights (0-255) along dim3,
 * representing the percentage of pixels with each weight in each color
 * channel.  Following the convention of the Packed3DArray type, the data are
 * packed in triplets (r1,g1,b1,r2,g2,b3...).  Note we pass by reference since
 * heap allocation will change the address used in virtual memory
 */
/*
void normalized_count(float *&dest, int *data, int dim1, int dim2){
	dest = new float[256*3];
	int arg;
	float norm = 1.0/(dim1 * dim2);
	
	// Accumulate pixel value counts
	int rcounts[256] = {0};
	int gcounts[256] = {0};
	int bcounts[256] = {0};

	for(int i = 0; i < dim1; ++i){
		for(int j = 0; j< dim2; ++j){
			arg = i*dim2*3 + j*3;
			rcounts[(int)data[arg]]++;
			gcounts[(int)data[arg + 1]]++;
			bcounts[(int)data[arg + 2]]++;
		}
	}

	// Assign normalized percentage to each pixel value, in packed order
	// in destination array
	for(int i = 0, j=0; i < 256; ++i, j = j + 3){
		dest[j] = rcounts[i] * norm;
		dest[j+1] = gcounts[i] * norm;
		dest[j+2] = bcounts[i] * norm;
	}

}
*/

/*
 * Compares the 256x(r,g,b) packed histogram in myHisto to the numNodes other
 * packed histograms in otherHistos to determine the most similar image.  In
 * this case, most similar means minimize the piecewise difference of the
 * histogram elements.  The histogram data has 3 channels of 256 values,
 * packed in triplets (r1,g1,b1,r2,g2,b2...).
 */
int compare_histos(float *myHisto, float *otherHistos, int myRank, int numNodes){
	float currDiff = 0.0;
	float minDiff = 10000000.0;
	float lowRank;

	lowRank = myRank == 0 ? 1:0;

	// For each rank in the world...
	for(int n = 0; n < numNodes; ++n){
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
		cout << "NODE " << rank << ": dimensions: (" << dims[0] << "," << dims[1] << ")" << endl;
		
		// On heap since stack will overflow for many images
		int *rawData = new int[dims[0]*dims[1]*3];

		// Wait for image data in a tag 1 message
		MPI_Recv(rawData, dims[0]*dims[1]*3, MPI_INT, 0, DATA_MSG_TAG, MPI_COMM_WORLD, &status);

		cout << "NODE " << rank << ": got image with " << dims[0]*dims[1] << " pixels..." << endl;
#if DEBUG
		serial_dump(string_format("node_%d_after.txt", rank), rawData, dims[0]*dims[1]);
#endif

		cout << "NODE " << rank << ": computing histogram..." << endl;

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
	
		cout << "NODE " << rank << ": starting gather..." << endl;

		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

		// Without in place MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

#if DEBUG
		// Sanity check.  My data should start at 256*3*myRank
		int offset = 256 * 3 * rank;
		for(int k = 0; k<256*3; ++k){
			assert(channels[k] == histos[offset + k]);
		}

#endif

		// Determine the most similar rank, and report back to rank 0
		
		int closest = compare_histos(channels, histos, rank, numNodes);
		cout << "NODE " << rank << ": most similar to NODE " << closest << endl;

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
/*
			int *red, *green, *blue;
			red = count(flatpack, dims[0], dims[1], 0);
			green = count(flatpack, dims[0], dims[1], 1);
			blue = count(flatpack, dims[0], dims[1], 2);
			dump(100+n, red, 0);
			dump(100+n, green, 1);
			dump(100+n, blue, 2);
			delete []red;
			delete []green;
			delete []blue;
*/

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
		
		cout << "ROOT: starting gather..." << endl;

		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);
		//   Without in place: MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

		// Sanity check: my data should be at rank*256*3
		for(int i = 0; i<256*3; ++i){
			assert(channels[i] == histos[i + rank*256*3]);
		}

		// Determine the most similar image and report
		int closest = compare_histos(channels, histos, rank, numNodes);
		cout << "NODE " << rank << ": most similar to NODE " << closest << endl;
		
		// Catch and report results from each other rank
		

		// Clean up
		delete [] channels;
		delete [] flatpack;
	}

	

	MPI_Finalize();

}
