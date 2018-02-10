CFLAGS=-Wall -std=c++11 -lpthread -g
CC=g++

trains: trains.cpp Barrier.o
	$(CC) trains.cpp Barrier.o $(CFLAGS) -o $@

Barrier.o: Barrier.cpp Barrier.h
	$(CC) $(CFLAGS) -c Barrier.cpp -o $@

clean:
	$(RM) *.o
	$(RM) main

.PHONY: clean main
