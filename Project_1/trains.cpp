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
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <chrono>
#include "Barrier.h"
#include <cassert>


using namespace std;

// Globals for synchronization
mutex ***tracks;	// Protect access to tracks as a matrix of edges
mutex *printLock = new mutex;	// Protect stdout
mutex *activeLock = new mutex;	// Protect number of trains
int nActiveTrains;	// Current number of trains
int nActiveTrainsNext;	// Number of trains t+1;
Barrier *lockstep = new Barrier;	// Barrier instance



/*
 * Safely acquire count of barrier calls
 */
int getActiveTrains(mutex *m){
	int ret;
	m->lock();
	ret = nActiveTrains;
	m->unlock();
	return ret;
}

void updateActiveTrains(mutex *m){
	m->lock();
	nActiveTrains = nActiveTrainsNext;
	m->unlock();
}

/*
 * Work function for threads.  Assumes at least pair of stations along the
 * route.  Returns the time in which it finished
 *
 * Notes 2/14
 *
 * We cant rely on longestRoute because contention make finish time
 * nondeterministic.  Instead, we need to keep track of the current number of
 * active trains, and the number of active trains at the next time instant.  
 *
 * By placing the barrier at the end of the main loop, we can have all active
 * trains using the same number for barrier calls.  When a train arrives, it
 * adjusts the NEXT number of active trains downward.  Back at the top of the
 * loop, we set the current number of active trains to the next number of
 * active trains, ensuring that the barrier count is always accurate.  This
 * also allows finished threads to kick out and be done.
 *
 */
int work(int name, vector<int> route){

	
	int currPos, endPos, time, src, dest;
	time = 0;
	currPos = 0;
	endPos = route.size() - 1;

	printLock->lock();
	cout << "Train stats: " << (char)(65+name) << endl;
	cout << "Current position: " << route.at(currPos) << endl;
	cout << "Next position: " << route.at(currPos+1) << endl;
	cout << "Final position: " << route.at(endPos) << endl;
	printLock->unlock();

	while(currPos != endPos){
		updateActiveTrains(activeLock);
		
		assert(getActiveTrains(activeLock) > 0);


		src = route.at(currPos);
		dest = route.at(currPos + 1); // <= endPos ? route.at(currPos + 1): route.at(endPos);

		printLock->lock();
		cout << "There are now " << getActiveTrains(activeLock) << " trains traveling" << endl;
		printLock->unlock();

		
		if(tracks[src][dest]-> try_lock()){
			printLock->lock();
			cout << "At time step: " << time << " train "
			     << (char)(65+name) << " is going from station "
			     << src << " to station " << dest << endl;
			printLock->unlock();

			// wait a moment so other threads definitely don't get
			// the lock this time instance
			this_thread::sleep_for(chrono::milliseconds(50));
			tracks[src][dest] -> unlock();
			currPos++;
		}
		else{
			printLock->lock();
			cout << "At time step: " << time << " train "
			     << (char)(65+name) << " must stay at station "
			     << src << endl;
			printLock->unlock();
		}
		time++;
		lockstep->barrier(getActiveTrains(activeLock));
	}
	
	// We have finished.  Adjust the next active trains
	activeLock->lock();
	nActiveTrainsNext--;
	activeLock->unlock();
	
	return time;


}


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


	nActiveTrains = nTrains;
	nActiveTrainsNext = nTrains;


	cout << "Got " << nTrains << " trains and " << nStations << " stations" << endl;

	cout << "Train routes:" << endl;

	for(int i = 0; i<nTrains; ++i){
		for(int n : routes[i]){
			cout << n << " ";
		}
		cout << endl;
	}

	/*
	 * Initialize a matrix of edges represented by atomic booleans
	 */

	tracks = new mutex **[nStations];
	for(int i = 0; i<nStations; ++i){
		tracks[i] = new mutex *[nStations];
		for(int j = i; j<nStations; ++j){
			mutex *m = new mutex;
			tracks[i][j] = m;
		}
	}

	// The references are reflected about the trace, since the track (0,1)
	// is identical to (1,0)
	for(int i = 0; i< nStations; ++i){
		for(int j = i; j< nStations; ++j){
			tracks[j][i] = tracks[i][j];
		}
	}

	cout << "Starting simulation..." << endl;

	std::vector<future<int>> results;
	int finishTimes[nTrains];
	for(int i = 0; i < nTrains; ++i){
		future<int> temp = async(launch::async, work, i, routes[i]);
		// "Move" the semantic record so the future ends up in the
		// right place
		results.push_back(move(temp));
	}

	// Gather results (blocks)
	for(int i = 0; i< nTrains; ++i){
		finishTimes[i] = results.at(i).get();
	}
	
	cout << "Simulation complete." << endl;

	/*
	 * Print results, clean up and exit
	 */

	for(int i = 0; i<nTrains; ++i){
		cout << "Train " << (char)(65+i) << " completed its route at time step " << finishTimes[i] << endl;
		routes[i].clear();
	}
	for(int i = 0; i<nStations; ++i){
		for(int j = i; j< nStations; ++j){
			delete tracks[i][j];
		}
		delete [] tracks[i];
	}
	delete [] tracks;
	delete [] routes;
	delete lockstep;

	return 0;
}
