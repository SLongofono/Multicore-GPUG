CFLAGS=-Wall -std=c++11 -lpthread -g
CC=g++

main: trains.cpp
	$(CC) $+ $(CFLAGS) -o $@

clean:
	$(RM) *.o
	$(RM) main

.PHONY: clean main
