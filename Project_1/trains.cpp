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

// Get stuff
#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>

// Hold stuff
#include <vector>
#include <string>

// Synchronize stuff
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include "Barrier.h"


using namespace std;

// Globals for synchronization
mutex ***tracks;		// Protect access to tracks as a matrix of edges
mutex *printLock = new mutex;	// Protect stdout
mutex *activeLock = new mutex;	// Protect number of trains
mutex *nextLock = new mutex; 	// Protect number of trains t+1
int nActiveTrains;		// Current number of trains
int nActiveTrainsNext;		// Number of trains t+1
Barrier *lockstep = new Barrier;// Barrier instance


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

/*
 * Safely update nActiveTrains to nActiveTrainsNext
 */
void updateActiveTrains(mutex *curr, mutex *next){
	curr->lock();
	next->lock();
	nActiveTrains = nActiveTrainsNext;
	next->unlock();
	curr->unlock();
}

/*
 * Helper function to print train names.  Assumes ASCII is OK on your system
 */
char getName(int n){
	return (char)(65+n);
}


/*
 * @brief Work function for threads.
 *
 * @description Assumes at least pair of stations along the
 * route.  Returns the time instant in which it is at its final station and
 * ready to respond.  Assumes that there is a section of track connecting any
 * consecutive stations in the route, and that this track can be traversed in
 * one time instant.
 *
 * @param name The thread ID, interpreted as the offset from 65 in the ASCII
 * alphabet
 *
 * @param route A vector representing the complete route that this train's
 * thread will run.
 *
 * @return The time instant at which this train is at its destination and
 * ready to move again.
 */
int work(int name, vector<int> route){

	
	int currPos, endPos, time, src, dest, havelock;
	time = 0;
	currPos = 0;
	endPos = route.size() - 1;
	havelock = 0;

	while(currPos != endPos){
		updateActiveTrains(activeLock, nextLock);

		lockstep->barrier(getActiveTrains(activeLock));

		src = route.at(currPos);
		dest = route.at(currPos + 1);

		if(tracks[src][dest]-> try_lock()){
			printLock->lock();
			cout << "At time step: " << time << " train "
			     << getName(name) << " is going from station "
			     << src << " to station " << dest << endl;
			printLock->unlock();
			currPos++;
			havelock = 1;
		}
		else{
			printLock->lock();
			cout << "At time step: " << time << " train "
			     << getName(name) << " must stay at station "
			     << src << endl;
			printLock->unlock();
		}
		time++;
		


		// Need to check if we are done here, because it will not
		// always work out that the number of active trains is updated
		// before trains that finish can update it.
		if(currPos == endPos){
			// Adjust nActiveTrainsNext so that the next round of
			// barriers works correctly.
			nextLock->lock();
			nActiveTrainsNext--;
			nextLock->unlock();
		}

		lockstep->barrier(getActiveTrains(activeLock));
		
		// If we have the lock, release it.  We need to hold it until
		// here to be sure that all threads have only one shot at the
		// track per instant of time.
		if(havelock){
			havelock = 0;
			tracks[src][dest] -> unlock();
		}
		
	}

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
	
	int nTrains, nStations, n, biggestStation = 0;
	vector<int> *routes;
	string s;
	
	if(infile.good()){
		infile >> nTrains >> nStations;
		getline(infile, s); // Burn the whitespace of first line
		routes = new vector<int>[nTrains];
		for(int i = 0; i < nTrains; ++i){
			getline(infile, s);
			stringstream line(s);

			// Burn station count, don't need it
			line >> n;

			while(line >> n){
				routes[i].push_back(n);

				// Assumes a monotonic numbering of stations
				if( n>biggestStation ){
					biggestStation = n;
				}
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

	/*
	 * Initialize a matrix of edges represented by atomic booleans
	 */

	tracks = new mutex **[biggestStation+1];
	for(int i = 0; i<= biggestStation; ++i){
		tracks[i] = new mutex *[biggestStation+1];
		for(int j = i; j<= biggestStation; ++j){
			mutex *f = new mutex;
			tracks[i][j] = f;
		}
	}

	// The references are reflected about the trace, since the track (0,1)
	// is identical to (1,0)
	for(int i = 0; i<= biggestStation; ++i){
		for(int j = i; j<= biggestStation; ++j){
			if(i != j){
				tracks[j][i] = tracks[i][j];
			}
		}
	}

	cout << "Starting simulation..." << endl;

	// Catch futures in a vector, since future pointers are wonky
	std::vector<future<int>> results;

	// Catch finish times separately since the futures are destroyed as
	// they resolve
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
		cout << "Train " << getName(i) << " completed its route at time step " << finishTimes[i] << endl;
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
	delete printLock;
	delete activeLock;
	delete nextLock;
	return 0;
}
