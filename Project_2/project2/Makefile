sample: sample.o ../lib/libCOGLImageReader.so
	g++ -o sample sample.o ../lib/libCOGLImageReader.so
sample.o: sample.c++
	g++ -c -std=c++11 -I../Packed3DArray -I../ImageReader sample.c++
../lib/libCOGLImageReader.so: ../ImageReader/ImageReader.h ../ImageReader/ImageReader.c++ ../Packed3DArray/Packed3DArray.h
	(cd ../ImageReader; make)
