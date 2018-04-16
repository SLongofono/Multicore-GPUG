#include <iostream>
#include <cassert>
#include <fstream>
#include "ImageWriter.h"

using namespace std;

void writeFile(std::string fname, int xres, int yres, const unsigned char* imageBytes){
	unsigned char *row = new unsigned char[3 * xres];
	ImageWriter *writer = ImageWriter::create(fname, xres, yres);
	int next = 0;
	for(int r = 0; r < yres; ++r){
		for(int c = 0; c < 3*xres; c += 3){
			row[c] = row[c+1] = row[c+2] = imageBytes[next++];
		}
		writer->addScanLine(row);
	}
	writer->closeImageFile();
	delete writer;
	delete [] row;
}

int main(int argc, char **argv){
	
	/*
	 * Setup and housekeeping...
	 */
	if(argc < 7){
		cerr << "[ error ] Expected more arguments!" << endl;
		cerr << "Usage: ./project3 nRows nCols nSheets fileName projectionType outputFileNameBase" << endl;
		cerr << "Types: executable int int int string int string" << endl;
		cerr << "Exiting..." << endl;
		return -1;
	}

	int nRows, nCols, nSheets, nVals;
	std::string fileType(".png");
	nRows = atoi(argv[1]);
	nCols = atoi(argv[2]);
	nSheets = atoi(argv[3]);
	nVals = nRows * nCols * nSheets;
	unsigned char *rawImageData = new unsigned char[nVals];	
	ifstream infile(argv[4]);
	if(!infile.good()){
		cerr << "[ error ] Bad input filename.  Exiting..." << endl;
		return -1;
	}
	infile.read( reinterpret_cast<char *>(rawImageData), nVals);
	infile.close();

	/*
	 * Spit out an image, unadulterated
	 */
	writeFile( (argv[6] + fileType), nRows, nCols, rawImageData);

	/*
	 * Clean up
	 */
	delete [] rawImageData;
	return 0;
}
