CFLAGS=-Wall -std=c++11 -lpthread
CC=g++

main: main.o
	$(CC) main.o myObj.o $(CFLAGS) -o $@

main.o: main.cpp myObj.o
	$(CC) -c main.cpp $(CFLAGS)

myObj.o: myObj.cpp myObj.h
	$(CC) -c myObj.cpp $(CFLAGS)

clean:
	$(RM) *.o
	$(RM) main

.PHONY: clean main
