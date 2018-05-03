/*
 * A Collection of helper functions for the third project.
 */

/*
 * Hacky way to do atomic max with float arguments
 * from https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
__device__ static float atomicMax(float *address, float val){
	int * address_as_i = (int *)address;
	int old = *address_as_i, assumed;
	do{
		assumed = old;
		old = ::atomicCAS(	address_as_i, assumed,
					__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

/*
 * Dumps device information
 */
void dumpDevices(){

	int len;
	cudaGetDeviceCount(&len);	
	cudaDeviceProp * deviceProps = new cudaDeviceProp[len];

	std::cout << "Found " << len << " devices..." << std::endl;

	for(int i = 0; i < len; ++i){

		cudaGetDeviceProperties(&deviceProps[i], i);
	
		std::cout << "Device " << i << ":\t" << deviceProps[i].name << std::endl;
		
		// Grab interesting attributes here
		std::cout << "Global memory available:      " << deviceProps[i].totalGlobalMem << std::endl;
		std::cout << "Shared memory per block(max): " << deviceProps[i].sharedMemPerBlock << std::endl;
		std::cout << "Shared memory per SM:         " << deviceProps[i].sharedMemPerMultiprocessor << std::endl;
		std::cout << "Threads per SM (max):         " << deviceProps[i].maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Warp size:                    " << deviceProps[i].warpSize << std::endl;
		std::cout << "Registers per block (max):    " << deviceProps[i].regsPerBlock << std::endl;
		std::cout << "Compute capability:           " << deviceProps[i].major << "." << deviceProps[i].minor << std::endl;
		if(deviceProps[i].major < 2){
			std::cout << "Number of warp schedulers:    " << 1 << std::endl;
		}
		else if(deviceProps[i].major < 3){
			std::cout << "Number of warp schedulers:    " << 2 << std::endl;
		}
		else{
			std::cout << "Number of warp schedulers:    " << 4 << std::endl;
		}
		std::cout << std::endl;

	}
	
	delete [] deviceProps;

}


/*
 * Fetches device properties struct
 */
cudaDeviceProp *getDevice(int deviceNum){
	int len;
	cudaGetDeviceCount(&len);
	assert(deviceNum > 0);
	assert(deviceNum < len);
	cudaDeviceProp * deviceProps = new cudaDeviceProp[1];
	cudaGetDeviceProperties(deviceProps, deviceNum);
	return deviceProps;
}


// Error handler from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void gpuAssert(cudaError_t code, const char *filename, int line, bool abort=true){
	if(code != cudaSuccess){
		std::cout 	<< "GPU asserted an error on line "
				<< line << " from file " << filename
				<< ":" << std::endl << cudaGetErrorString(code)
				<< std::endl;
		if(abort){
			exit(-1);
		}
	}
	else{
#if DEBUG
		std::cout << "SUCCESS: line " << line << " in " << filename << std::endl;
#endif
	}
}

// Use a macro so we can access file and line information
#define validate(answer) { gpuAssert((answer),__FILE__, __LINE__); }


/*
 *  Projection of voxel data per rubric, column major from a different cardinal
 *  perspective.  The input and output array are packed in column major, one
 *  sheet at a time.
 */
void projection(unsigned char *data, int nRows, int nCols, int nSheets, int projection){
	
	if(1 == projection){
		// Nothing to do
		return;
	}
	
	int curPos, newPos;
	int newX, newY, newZ;
	int newRows, newCols, newSheets;
	int nVals = nRows*nCols*nSheets;
	unsigned char *work = new unsigned char[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int z = 0; z < nSheets; ++z){
		for(int y = 0; y < nRows; ++y){
			for(int x = 0; x < nCols; ++x){
				// Assign new coordinates, based on projection
				// type per rubric
				switch(projection){
					case 2:
						newRows = nRows;
						newCols = nCols;
						newSheets = nSheets;
						newX = newCols - x - 1;
						newY = y;
						newZ = newSheets - z - 1;
						break;
					case 3:
						newRows = nRows;
						newCols = nSheets;
						newSheets = nCols;
						newX = z;
						newY = y;
						newZ = newSheets - x - 1;
						break;
					case 4:
						newRows = nRows;
						newCols = nSheets;
						newSheets = nCols;
						newX = newCols - z - 1;
						newY = y;
						newZ = x;
						break;
					case 5:
						newRows = nSheets;
						newCols = nCols;
						newSheets = nRows;
						newX = x;
						newY = z;
						newZ = newSheets - y - 1;
						break;
					case 6:
						newRows = nSheets;
						newCols = nCols;
						newSheets = nRows;
						newX = x;
						newY = newRows - z - 1;
						newZ = y;
						break;
					default:
						assert("Invalid projection type in projection()" == "");
				}
				

				// column major, new orientation
				newPos = newZ*newRows*newCols + newX*newRows + newY;

				// Column major, old orientation
				curPos = z*nRows*nCols + x*nRows + y;

				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}


/*
 *  * Transposes flattened 2D matrix from column major to row major
 *   */
void transpose(unsigned char *data, int rows, int cols){
	int nVals = rows*cols;
	unsigned char *work = new unsigned char[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}
	for(int r = 0; r < rows; ++r){
		for(int c = 0; c < cols; ++c){
			work[r*cols + c] = data[c*rows + r];
			//data[c*rows + r] = work[r*cols + c];
		}
	}
																				delete [] work;
}


/*
void transpose(unsigned char *data, int N, int M){
	int nVals = N*M;
	unsigned char *work = new unsigned char[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}
	for(int n = 0; n < nVals; ++n){
		int i = n/N;
		int j = n%N;
		data[n] = work[M*j + i];
	}
	delete [] work;
}
*/


/*
 * Writes an image to the given filename
 * Assumes that the image is a greyscale with a single 8-bit value for each
 * pixel
 */
void writeFile(std::string fname, int xres, int yres, const unsigned char* imageBytes){
	unsigned char *row = new unsigned char[3 * xres];
	ImageWriter *writer = ImageWriter::create(fname, xres, yres);
	int next = 0;
	for(int r = 0; r < yres; ++r){
		for(int c = 0; c < 3*xres; c += 3){
			// The internal representation of the image is
			// (r,g,b), so write the same value for all three
			row[c] = row[c+1] = row[c+2] = imageBytes[next++];
		}
		writer->addScanLine(row);
	}
	writer->closeImageFile();
	delete writer;
	delete [] row;
}


/*
 * Writes an image result according to the projection.
 *
 * Note that the rubric contradicts itself - the specifications are correct,
 * the examples are incorrect.  The provided reference images are written with
 * xres as width and yres as height, but the example text states the converse.
 */
void writeImage(std::string fileName, unsigned char *data, int projection, int nRows, int nCols, int nSheets){
	switch(projection){
		case 1:
		case 2:
			transpose(data, nRows, nCols);
			writeFile(fileName, nCols, nRows, data);
			break;
		case 3:
		case 4:
			transpose(data, nRows, nSheets);
			writeFile(fileName, nSheets, nRows, data);
			break;
		case 5:
		case 6:
			transpose(data, nSheets, nCols);
			writeFile(fileName, nCols, nSheets, data);
			break;
	}
}


void dumpSheet(unsigned char *data, int projection, int nRows, int nCols, int nSheets, int sheetNo){
	unsigned char * work = new unsigned char[nRows * nCols];
	for(int r = 0; r < nRows; ++r){
		for(int c = 0; c < nCols; ++c){
			work[c*nRows + r] = data[c*nRows + r + sheetNo*nRows*nCols];
		}
	}
	writeImage(std::string("sheet_") + std::to_string(sheetNo) + std::string(".png"), work, projection, nRows, nCols, nSheets);
}
