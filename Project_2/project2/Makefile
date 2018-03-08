CC=mpic++
CXXFLAGS=-Wall -g -std=c++11
LD=mpic++
TESTFILES=images/b1.jpg images/b2.jpg images/b3.jpg images/g1.jpg images/g2.jpg images/g3.jpg images/k1.jpg images/k2.jpg images/k3.jpg images/r1.jpg images/r2.jpg images/r3.jpg
OBJS=main.o ../lib/libCOGLImageReader.so

main: main.o ../lib/libCOGLImageReader.so
	$(CC) $(CXXFLAGS)  $(OBJS) -o $@

main.o: main.cpp
	$(CC) -c main.cpp $(CXXFLAGS) -I../Packed3DArray -I../ImageReader -o $@ 

../lib/libCOGLImageReader.so: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

test:
	mpirun -N 12 main $(TESTFILES)

clean:
	$(RM) *.o
	$(RM) main
	$(RM) ../lib/libCOGLImageReader.so

.PHONY: clean main
