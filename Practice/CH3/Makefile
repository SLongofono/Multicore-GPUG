CFLAGS=-Wall -std=c++11 -lpthread
CC=g++

main: race_cond.cpp
	$(CC) $+ $(CFLAGS) -o $@

clean:
	$(RM) *.o
	$(RM) main

.PHONY: clean main
