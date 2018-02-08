/*
 * @file trains.cpp
 * @brief The entry point for the threaded train simulation project
 *
 * EECS690: Multicore and GPUG Programming
 * University of Kansas
 *
 * Author: 	Stephen Longofono
 * Created: 	2/7/2018
 *
 */

#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#define VERBOSE 0


using namespace std;

int main(int argc, char **argv){

	/*
	 * Parse trains, stations, routes from input file
	 */
	if(argc < 2){
		cerr << "Usage: ./main <input filepath>" << endl;
		cerr << "Expected an input file, exiting..." << endl;
		return -1;
	}

	ifstream infile(argv[1]);
	
	int nTrains, nStations, n;
	vector<int> *routes;
	string s;
	
	if(infile.good()){
		infile >> nTrains >> nStations;
		getline(infile, s); // Burn the whitespace of first line
		routes = new vector<int>[nTrains];
		for(int i = 0; i < nTrains; ++i){
			getline(infile, s);
			stringstream line(s);
			while(line >> n){
				routes[i].push_back(n);
			}
		}
	}
	else{
		cerr << "Input file error, exiting" << endl;
		return -1;
	}

	infile.close();

#if VERBOSE
	cout << "Got " << nTrains << " trains and " << nStations << " stations" << endl;

	cout << "Train routes:" << endl;

	for(int i = 0; i<nTrains; ++i){
		for(int n : routes[i]){
			cout << n << " ";
		}
		cout << endl;
	}
#endif

	/*
	 * Initialize finish times and a matrix of edges
	 */
	int finishTimes[nTrains];
	bool *tracks[nStations];
	for(int i = 0; i<nTrains; ++i){
		finishTimes[i] = 0;
	}

	for(int i = 0; i<nStations; ++i){
		tracks[i] = new bool[nStations];
		for(int j = 0; j<nStations; ++j){
			tracks[i][j] = false;
		}
	}

#if VERBOSE
	for(int i = 0; i<nStations; ++i){
		for(int j = 0; j< nStations; ++j){
			cout << tracks[i][j] << " ";
		}
		cout << endl;
	}

#endif
	cout << "Starting simulation..." << endl;

	/*
	 * Basic work design:
	 * 	Work loop while begins with a barrier call to enforce time
	 * 	while condition is currentLocation != destination
	 * 	acquire tracks lock
	 * 	check track(currentLocation, nextLocation) and release tracks
	 * 	give report, acquire tracks lock
	 * 	release track(currentLocation, nextLocation) and release
	 * 	tracks
	 * 	update currentLocation and nextLocation
	 *	Parameters: the route (vector<int>), the mutexes, a future
	 *
	 * Design helpers:
	 * 	cout mutex
	 * 	tracks mutex
	 * 	future to collect time complete results
	 */

	cout << "Simulation complete." << endl;

	/*
	 * Print results, clean up and exit
	 */

	for(int i = 0; i<nTrains; ++i){
		cout << "Train " << (char)(65+i) << " completed its route at time step " << finishTimes[i] << endl;
		routes[i].clear();
	}
	for(int i = 0; i<nStations; ++i){
		delete [] tracks[i];
	}
	delete [] routes;

	return 0;
}
