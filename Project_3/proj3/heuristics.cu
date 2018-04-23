// File : executionConfHeur.cu
#define min ( a , b ) ( ( a<b ) ? a : b )
#define max ( a , b ) ( ( a>b ) ? a : b )
//−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
// In : numberOfThreads , registersPerThread , sharedPerThread
// Out : bestThreadsPerBlock , bestTotal Blocks

void calcExecConf ( int numberOfThreads , int registersPerThread , int sharedPerThread , int &bestThreadsPerBlock , int &bestTotalBlocks ){
	cudaDeviceProp pr ;
	cudaGetDeviceProperties(&pr , 0) ; // replace 0 with appropriate ID in case of a multi −GPU system
	int maxRegs = pr.regsPerBlock ;
	int SM = pr.multiProcessorCount ;
	int warp = pr . warpSize ;
	int sharedMem = pr . sharedMemPerBlock ;
	int maxThreadsPerSM = pr . maxThreadsPerMultiProcessor ;
	int totalBlocks ;
	f l o a t imbalance , bestimbalance ;
	int threadsPerBlock ;

	int numWarpSchedulers ;
	switch(pr.major){
		case 1:
			numWarpSchedulers = 1;
			break ;
		case 2:
			numWarpSchedulers = 2;
			break ;
		default :
			numWarpSchedulers = 4;
			break ;
	}

	bestimbalance = SM ;

	// initially calculate the maximum possible threads per block. Incorporate limits imposed by :
	// 1) SM hardware
	threadsPerBlock = maxThreadsPerSM ;

	// 2) registers
	threadsPerBlock = min(threadsPerBlock , maxRegs / registersPerThread);

	// 3) shared memory size
	threadsPerBlock = min(threadsPerBlock , sharedMem / sharedPerThread);

	// make sure it is a multiple of warpSize
	int tmp = threadsPerBlock / warp ;
	threadsPerBlock = tmp ∗ warp ;

	for ( ; threadsPerBlock >= numWarpSchedulers ∗ warp && bestimbalance != 0 ; threadsPerBlock −= warp){
		totalBlocks = (int) ceil(1.0 ∗ numberOfThreads / threadsPerBlock);
		if(totalBlocks % SM == 0){
			imbalance = 0;
		}
		else{
			int blocksPerSM = totalBlocks / SM; // some SMs get this number and others get +1 block
			imbalance = (SM − (totalBlocks % SM)) / (blocksPerSM + 1.0);
		}

		if ( bestimbalance >= imbalance ){
			bestimbalance = imbalance ;
			bestThreadsPerBlock = threadsPerBlock;
			bestTotalBlocks = totalBlocks;
		}
	}
}
