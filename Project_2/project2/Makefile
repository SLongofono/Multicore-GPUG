CC=mpic++
CXXFLAGS=-Wall -g -std=c++11
LD=mpic++
TEST1=images/pure_green.png images/pure_green.png images/pure_green.png
TEST2=images/R1G255B1.png images/R1G255B1.png images/R1G255B1.png
TEST3=images/RED_PIXEL.png images/BLUE_PIXEL.png images/GREEN_PIXEL.png
TEST4=images/RED_PIXEL.png images/GREEN_PIXEL.png images/BLUE_PIXEL.png
TEST5=images/pure_green.png images/R1G255B1.png images/pure_red.png
TESTFILES=images/b1.jpg images/b2.jpg images/b3.jpg images/g1.jpg images/g2.jpg images/g3.jpg images/k1.jpg images/k2.jpg images/k3.jpg images/r1.jpg images/r2.jpg images/r3.jpg
OBJS=main.o ../lib/libCOGLImageReader.so

main: main.o ../lib/libCOGLImageReader.so
	$(CC) $(CXXFLAGS)  $(OBJS) -o $@

main.o: main.cpp
	$(CC) -c main.cpp $(CXXFLAGS) -I../Packed3DArray -I../ImageReader -o $@ 

../lib/libCOGLImageReader.so: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

test1:
	mpirun -N 3 main $(TEST1)

test2:
	mpirun -N 3 main $(TEST2)

test3:
	mpirun -N 3 main $(TEST3)

test4:
	mpirun -N 3 main $(TEST4)

test5:
	mpirun -N 3 main $(TEST5)

clean:
	$(RM) *.o
	$(RM) main
	$(RM) ../lib/libCOGLImageReader.so

.PHONY: clean main
