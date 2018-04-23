/*
 * A Collection of helper functions for the third project.
 */

cuda::cudaDeviceProp *getDevice(int deviceNum){
	int len;
	cudaGetDeviceCount(&len);
	assert(deviceNum > 0);
	assert(deviceNum < len);
	cudaDeviceProp * deviceProps = new cudaDeviceProp[1];
	cudaGetDeviceProperties(&deviceProps[i], deviceNum);
	return deviceProps;
}


cuda::dim3 getGridGeometry(int nRows, int nCols, int nSheets, int projection){
	cuda::cudaDeviceProp = getDevice(0);
	int xDim, yDim, zDim;
	cuda::dim3 ret(xDim,yDim,zDim);
	return ret;
}


cuda::dim3 getBlockGeometry(int nRows, int nCols, int nSheets, int projection){
	cuda::cudaDeviceProp = getDevice(0);
	int xDim, yDim, zDim;
	cuda::dim3 ret(xDim,yDim,zDim);
	return ret;
}


// Quick and dirty matrix multiplication for transformations
// Where A is M rows by P columns, B is P rows by N columns, C will be M rows by N columns
void matrixMultiplyColumnMajor(int* A, int* B, int* C, int M, int P, int N){
	int sum = 0;
	for (int i = 0; i < M; ++i){
		for (int j = 0; j < N; ++j){
			for (int k = 0; k < P; ++k){
				sum += A[k*M + i] * B[j*P + k];
			}
			C[j*M + i] = sum;
		}
	}
}


/*
 * Given degrees (+/-90, 180) and a unit vector representing the axis of
 * rotation, returns a pointer to a column-major transformation matrix for R3
 * space
 */
int * getTransformationMatrix(int degrees, int l, int m, int n){
	int * transform = new int[9];
	int costheta, sintheta;
	
	switch(degrees){
		case 90:
			costheta = 0;
			sintheta = 1;
			break;
		case -90:
			costheta = 0;
			sintheta = -1;
			break;
		case 180:
		case -180:
			costheta = -1;
			sintheta = 0;
			break;
	}
	
	transform[0] = l*l*(1-costheta) + costheta;
	transform[1] = l*m*(1-costheta) + n*sintheta;
	transform[2] = l*n*(1-costheta) - m*sintheta;
	transform[3] = m*l*(1-costheta) - n*sintheta;
	transform[4] = m*m*(1-costheta) + costheta;
	transform[5] = m*n*(1-costheta) + l*sintheta;
	transform[6] = n*l*(1-costheta) + m*sintheta;
	transform[7] = n*m*(1-costheta) - l*sintheta;
	transform[8] = n*n*(1-costheta) + costheta;

	return transform;
}


/*
 * From wikipedia transformation matrix article
 * https://en.wikipedia.org/wiki/Transformation_matrix
 *
 * 3D Rotation matrix, rotation t about unit axis [l,m,n]
 *
 * ll(1-cos(t)) + cos(t)	ml(1-cos(t)) - nsin(t)	nl(1-cos(t)) + msin(t)
 * lm(1-cos(t)) + nsin(t)	mm(1-cos(t)) + cos(t)	nm(1-cos(t)) - lsin(t)
 * ln(1-cos(t)) - msin(t)	mn(1-cos(t)) + lsin(t)	nn(1-cos(t)) + cos(t)
 *
 */
void rotate(unsigned char *data, int nRows, int nCols, int nSheets, int projection){
	// Nothing to do
	if(1 == projection){ return; }

	int nVals = nRows*nCols*nSheets;
	unsigned char *work = new unsigned char[nVals];
	for(int i = 0; i<nVals; ++i){
		work[i] = data[i];
	}
	int oldIndex, newIndex, newRows, newCols, newSheets;
	int *curCoords = new int[3];
	int *newCoords = new int[3];
	int * transform;

	switch(projection){
		case 2:
			// Rotate 180 about original y axis
			tranform = getTransformationMatrix(180, 0, 1, 0);
			newRows = nRows;
			newCols = nCols;
			newSheets = nSheets;
		case 3:
			// Rotate -90 about original y axis
			tranform = getTransformationMatrix(-90, 0, 1, 0);
			newRows = nRows;
			newCols = nSheets;
			newSheets = nCols;
		case 4:
			// Rotate 90 about original y axis
			tranform = getTransformationMatrix(90, 0, 1, 0);
			newRows = nRows;
			newCols = nSheets;
			newSheets = nCols;
		case 5:
			// Rotate 90 about original x axis
			tranform = getTransformationMatrix(90, 1, 0, 0);
			newRows = nSheets;
			newCols = nCols;
			newSheets = nRows;
		case 6:
			// Rotate -90 about original x axis
			tranform = getTransformationMatrix(-90, 1, 0, 0);
			newRows = nSheets;
			newCols = nCols;
			newSheets = nRows;
	}

	for(int i = 0; i < nRows; ++i){
		for(int j = 0; j < nCols; ++j){
			for(int k = 0; k < nSheets; ++k){
				curCoords[0] = i;
				curCoords[1] = j;
				curCoords[2] = k;
				matrixMultiplyColumnMajor(transform, curCoords, newCoords, 3, 3, 1);
				oldIndex = k*nRows*nCols + j*nRows + i;
				newIndex = newCoords[2]*newRows*newCols +
					   newCoords[1]*newRows +
					   newCoords[0];
				data[newIndex] = work[oldIndex];
			}
		}
	}
}


/*
 *  Original orientation: x-> cols, y->rows, z->sheets
 *  Projects x as max to min column, y as min to max row, z as max to min sheet
 */
void projection2(unsigned char *data, int nRows, int nCols, int nSheets){
	int curPos, newPos, temp;
	int nVals = nRows*nCols*nSheets;
	int *work = new int[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int x = 0; x < nCols; ++x){
		for(int y = 0; y < nRows; ++y){
			for(int z = 0; z < nSheets; ++z){
				// column major, new orientation
				newPos = ((nSheets - z - 1)*nRows*nCols) +
					((nCols - x - 1)*nRows) +
					y;

				assert(newPos < nVals);

				// Column major, old orientation
				curPos = z*nRows*nCols + x*nRows + y;
				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}


/*
 *  Original orientation: x-> cols, y->rows, z->sheets
 *  Projects x as min to max sheet, y as min to max row, z as min to max column
 */
void projection3(unsigned char *data, int nRows, int nCols, int nSheets){
	int curPos, newPos, temp;
	int nVals = nRows*nCols*nSheets;
	int *work = new int[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int x = 0; x < nCols; ++x){
		for(int y = 0; y < nRows; ++y){
			for(int z = 0; z < nSheets; ++z){
				// column major, new orientation
				newPos = x*nRows*nSheets + z*nRows + y;
				
				assert(newPos < nVals);

				// Column major, old orientation
				curPos = z*nRows*nCols + x*nRows + y;
				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}


/*
 *  Original orientation: x-> cols, y->rows, z->sheets
 *  Projects x as max to min sheet, y as min to max row, z as max to min column
 */
void projection4(unsigned char *data, int nRows, int nCols, int nSheets){
	int curPos, newPos, temp;
	int nVals = nRows*nCols*nSheets;
	int *work = new int[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int x = 0; x < nCols; ++x){
		for(int y = 0; y < nRows; ++y){
			for(int z = 0; z < nSheets; ++z){
				// column major, new orientation
				newPos = ((nCols - 1 - x)*nRows*nCols) +
					 ((nSheets - 1 - z)*nRows) +
					 y;

				assert(newPos < nVals);

				// Column major, old orientation
				curPos = sh*nRows*nCols + c*nRows + r;
				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}

/*
 *  Original orientation: x-> cols, y->rows, z->sheets
 *  Projects x as min to max column, y as min to max sheet, z as max to min row
 */
void projection5(unsigned char *data, int nRows, int nCols, int nSheets){
	int curPos, newPos, temp;
	int nVals = nRows*nCols*nSheets;
	int *work = new int[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int r = 0; r < nRows; ++r){
		for(int c = 0; c < nCols; ++c){
			for(int sh = 0; sh < nSheets; ++sh){
				// column major, new orientation
				newPos = ((nSheets - 1 - sh)*nRows*nCols) +
					 ((nCols - 1 - c)*nRows) +
					 r;
				assert(newPos < nVals);

				// Column major, old orientation
				curPos = sh*nRows*nCols + c*nRows + r;
				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}

/*
 *  Original orientation: x-> cols, y->rows, z->sheets
 *  Projects x as min to max column, y as max to min sheet, z as min to max row
 */
void projection6(unsigned char *data, int nRows, int nCols, int nSheets){
	int curPos, newPos, temp;
	int nVals = nRows*nCols*nSheets;
	int *work = new int[nVals];
	for(int i = 0; i < nVals; ++i){
		work[i] = data[i];
	}

	for(int r = 0; r < nRows; ++r){
		for(int c = 0; c < nCols; ++c){
			for(int sh = 0; sh < nSheets; ++sh){
				// column major, new orientation
				newPos = ((nSheets - 1 - sh)*nRows*nCols) +
					 ((nCols - 1 - c)*nRows) +
					 r;
				assert(newPos < nVals);

				// Column major, old orientation
				curPos = sh*nRows*nCols + c*nRows + r;
				data[newPos] = work[curPos];
			}
		}
	}
	delete [] work;
}

// Swaps sheet order in place
void flipSheets(unsigned char *data, int nRows, int nCols, int nSheets){
	int temp, offset, sheet1, sheet2;
	for(int row = 0; row < nRows; ++row){
		for(int col = 0; col < nCols; ++col){
			offset = nRows*col + row;
			for(int sh = 0; sh < nSheets/2; ++sh){
				sheet1 = sh*nRows*nCols;
				sheet2 = (nSheets-1-sh)*nRows*nCols;
				temp = data[sheet1 + offset];
				data[sheet1 + offset] = data[sheet2 + offset];
				data[sheet2 + offset] = temp;
			}
		}
	}
}


// Swaps column order in place
void flipCols(unsigned char *data, int nRows, int nCols, int nSheets){
	int temp, sheet, column1, column2;
	for(int row = 0; row < nRows; ++row){
		for(int sh = 0; sh < nSheets; ++sh){
			sheet = sh*nRows*nCols;
			for(int col = 0; col < nCols/2; ++col){
				column1 = col*nRows + row;
				column2 = (nCols - 1 - col)*nRows + row;
				temp = data[sheet + column1];
				data[sheet + column1] = data[sheet + column2];
				data[sheet + column2] = temp;
			}
		}
	}
}


// Swaps row order in place
void flipRows(unsigned char *data, int nRows, int nCols, int nSheets){
	int temp, sheet, row1, row2;
	for(int sh = 0; sh < nSheets; ++sh){
		for(int col = 0; col < nCols; ++col){
			sheet = sh*nRows*nCols;
			for(int row = 0; row < nRows/2; ++row){
				row1 = col*nRows + row;
				row2 = col*nRows + (nRows -1 - row);
				temp = data[sheet + row1];
				data[sheet + row1] = data[sheet + row2];
				data[sheet + row2] = temp;
			}
		}
	}
}


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
