/*
 * A Collection of helper functions for the third project.
 */

// Simplest logfile approach for debugging
void log(std::string filename, std::string s){
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);
	if(outfile.good()){
		outfile << s << std::endl;
	}
	outfile.close();
}

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
