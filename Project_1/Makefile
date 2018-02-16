CXXFLAGS=-Wall -std=c++11 -lpthread -g
CC=g++

trains: trains.cpp Barrier.h 
	$(CC) trains.cpp $(CXXFLAGS) -o $@

clean:
	$(RM) *.o
	$(RM) main

.PHONY: clean trains
