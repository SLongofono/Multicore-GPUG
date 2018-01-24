int main(){

/*
 * Sequential application of a kernel matrix for convolutional image filtering
 *
 * Great candidate for parallelization, but this is the correct sequential
 * code.
 */

	int n = 3; // 3 nearest neighbors
	int kern[n][n]; // The kernel weights for adjacent pixels
	int n2 = n/2;
	int IMGX = 256; // Image dimensions in pixels
	int IMGY = 256;
	int img[IMGY][IMGX]; // Load image here
	int filt[IMGY+2][IMGX+2]; // Result here, note this reduces the size
	
	// Code to load image and intialize kernel here

	// Note we are mapping rows to y,j, columns to x,i
	for(int x = 1; x <= IMGX; ++x){
		for(int y = 1; y < IMGY; ++y){
			int newVal = 0;
			for(int i = -n2; i<= n2; ++i){
				for(int j = -n2; j <= n2; ++j){
					newVal += img[y-j][x-i]*kern[j+n2][i+n2];
				}	
			}
			filt[y-1][x-1] = newVal;
		}
	}

	// Code to dump filtered image here

}
