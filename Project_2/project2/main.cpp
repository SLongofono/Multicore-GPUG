#include <mpi.h>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "ImageReader.h"
#define DEBUG 1

// For testing only

#if DEBUG
#include <fstream>
#include <cstdio>
#include <memory>
#endif

#define TEST4 1

using namespace std;

template<typename ... Args>
string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf( new char[ size ] ); 
    snprintf( buf.get(), size, format.c_str(), args ... );
    return string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

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

void get_red_count(int *&dest, unsigned char *data, int dim1, int dim2){
	dest = new int[256]();
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			dest[data[i*dim2*3 + j*3]]++;
		}
	}
}

void get_green_count(int *&dest, unsigned char *data, int dim1, int dim2){
	dest = new int[256]();
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			dest[data[i*dim2*3 + j*3 + 1]]++;
		}
	}
}

void get_blue_count(int *&dest, unsigned char *data, int dim1, int dim2){
	dest = new int[256]();
	for(int i = 0; i<dim1; ++i){
		for(int j = 0; j<dim2; ++j){
			dest[data[i*dim2*3 + j*3 + 2]]++;
		}
	}
}

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
			rcounts[(int)data[arg]]++;
			gcounts[(int)data[arg + 1]]++;
			bcounts[(int)data[arg + 2]]++;
		}
	}

	cout << endl;

	// Assign normalized percentage to each pixel value, in packed order
	// in destination array
	for(int i = 0, j=0; i < 256; ++i, j = j + 3){
		dest[j] = rcounts[i] / norm;
		dest[j+1] = gcounts[i] / norm;
		dest[j+2] = bcounts[i] / norm;
	}

}


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
			currDiff = 0.0;
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
		cout << "NODE " << rank << ": dimensions: (" << dims[0] << "," << dims[1] << ")" << endl;
			
		unsigned char rawData[dims[0]*dims[1]*3];

		// Wait for image data in a tag 1 message
		MPI_Recv(rawData, sizeof(rawData), MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);

		cout << "NODE " << rank << ": got image with " << dims[0]*dims[1] << " pixels..." << endl;


		// Compute histograms
		float *channels;
		normalized_count(channels, rawData, dims[0], dims[1]);


		// All-to-all gather to get results from other ranks
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];
		MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

#if DEBUG
		// Sanity check.  My data should start at 256*3*myRank
		int offset = 256 * 3 * rank;
		for(int k = 0; k<256*3; ++k){
			assert(channels[k] == histos[offset + k]);
		}

		stats(rank, channels);
		dump(rank, channels);
		//cout << "NODE " << rank << ": " << 100*channels[512 + 253] << " percent of my pixels are max red" << endl;
		//cout << "NODE " << rank << ": " << 100*channels[512 + 254] << " percent of my pixels are max green" << endl;
		//cout << "NODE " << rank << ": " << 100*channels[512 + 255] << " percent of my pixels are max blue" << endl;
#endif

		// Determine the most similar rank, and report back to rank 0
		
		int closest;
		closest = compare_histos(channels, histos, rank, numNodes);
		cout << "NODE " << rank << ": most similar to NODE " << closest << endl;


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
						// 3D indexing: i*cols*size + j*size + k
						size_t arg = (i*dims[1]*3) + (j*3) + k;
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
				// 3D indexing: i*cols*size + j*size + k
					size_t arg = (i*dims[1]*3) + (j*3) + k;
					flatpack[arg] = arr->getDataElement(i,j,k);
				}
			}
		}


		// Compute our histogram
		float *channels;
		normalized_count(channels, flatpack, dims[0], dims[1]);	

#if DEBUG
		stats(0, channels);
		dump(0, channels);
#endif
		// All-to-all gather to get histograms from other ranks
		// Declare a buffer big enough for everything
		float histos[256*3*numNodes];
		MPI_Allgather(channels, 256*3, MPI_FLOAT, histos, 256*3, MPI_FLOAT, MPI_COMM_WORLD);

		// Determine the most similar image and report
		int closest;
		closest = compare_histos(channels, histos, rank, numNodes);
		cout << "NODE " << rank << ": most similar to NODE " << closest << endl;
		
		// Catch and report results from each other rank
		

	}

	

	MPI_Finalize();

}
