// Error handler from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// No touchy touchy

inline void gpuAssert(cudaError_t code, const char *filename, int line, bool abort=true){
	if(code != cudaSuccess){
		cout 	<< "GPU asserted an error on line "
			<< line << ":" << endl << cudaGetErrorString(code)
			<< endl << " from file " << filename << endl;
		if(abort){
			exit(-1);
		}
	}	
}

#define validate(answer) { gpuAssert((answer),__FILE__, __LINE__); }

// end no touchy touchy


