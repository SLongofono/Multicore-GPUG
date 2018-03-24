CC=mpic++
CXXFLAGS=-Wall -g -std=c++11
LD=mpic++
TEST0=images/R1G255B1.png images/R1G255B1.png images/R1G255B1.png images/R1G255B1.png images/R1G255B1.png
TEST1=images/pure_green.png images/pure_green.png images/pure_green.png
TEST2=images/b2.jpg images/b2.jpg images/b2.jpg
TEST3=images/R1G255B1.png images/R1G255B1.png 
TEST4=images/RED_PIXEL.png images/GREEN_PIXEL.png images/BLUE_PIXEL.png
TEST5=images/pure_green.png images/R1G255B1.png images/pure_red.png images/pure_red.png
TEST6=images/b1.jpg images/b2.jpg images/b3.jpg images/g1.jpg images/g2.jpg images/g3.jpg images/k1.jpg images/k2.jpg images/k3.jpg images/r1.jpg images/r2.jpg images/r3.jpg
OBJS=main.o ../lib/libCOGLImageReader.so

main: main.o ../lib/libCOGLImageReader.so
	$(CC) $(CXXFLAGS)  $(OBJS) -o $@

main.o: main.cpp
	$(CC) -c main.cpp $(CXXFLAGS) -I../Packed3DArray -I../ImageReader -o $@ 

../lib/libCOGLImageReader.so: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)

test0:
	mpirun -N 5 main $(TEST0)

test1:
	mpirun -N 3 main $(TEST1)

test2:
	mpirun -N 3 valgrind ./main $(TEST2)

test3:
	mpirun -N 2 main $(TEST3)

test4:
	mpirun -N 3 main $(TEST4)

test5:
	mpirun -N 4 main $(TEST5)

test6:
	mpirun -N 12 valgrind ./main $(TEST6)

clean:
	$(RM) *.o
	$(RM) main
	$(RM) ../lib/libCOGLImageReader.so

.PHONY: clean main
