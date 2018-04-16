CPP=g++
CFLAGS=-g -std=c++11 -Wall
INCL=../ImageWriter
SODIR=../lib/OGL
OBJS=main.o $(SODIR)/libCOGLImageWriter.so
IMAGE1=256 256 256 ../voxeldata/aneurism.raw
IMAGE2=256 256 256 ../voxeldata/bonsai.raw
IMAGE3=256 256 178 ../voxeldata/BostonTeapot.raw
IMAGE4=256 256 128 ../voxeldata/engine.raw
IMAGE5=256 256 256 ../voxeldata/foot.raw
IMAGE6=64 64 64 ../voxeldata/fuel.raw
IMAGE7=128 128 128 ../voxeldata/hydrogenAtom.raw
IMAGE8=324 301 56 ../voxeldata/lobster.raw
IMAGE9=41 41 41 ../voxeldata/marschnerlobb.raw
IMAGE10=256 256 124 ../voxeldata/mri_ventricles.raw
IMAGE11=64 64 64 ../voxeldata/neghip.raw
IMAGE12=41 41 41 ../voxeldata/nucleon.raw
IMAGE13=64 64 512 ../voxeldata/shockwave.raw
IMAGE14=34 98 34 ../voxeldata/silicium.raw
IMAGE15=256 256 256 ../voxeldata/skull.raw
IMAGE16=341 341 93 ../voxeldata/statueLeg.raw
IMAGE17=512 512 174 ../voxeldata/bonsai.raw

project3: main.o
	$(CPP) $(CFLAGS) $(OBJS) -o $@

main.o:	main.cpp 
	$(CPP) $(CFLAGS) -I$(INCL) -c $+ -o $@

test1:
	@echo "============================================================\nTesting Image aneurism\n============================================================"
	./project3 $(IMAGE1) ${PT} aneurism
	@echo

test2:
	@echo "============================================================\nTesting Image bonsai\n============================================================"
	./project3 $(IMAGE2) ${PT} bonsai
	@echo

test3:
	@echo "============================================================\nTesting Image BostonTeapot\n============================================================"
	./project3 $(IMAGE3) ${PT} BostonTeapot
	@echo

test4:
	@echo "============================================================\nTesting Image engine\n============================================================"
	./project3 $(IMAGE4) ${PT} engine
	@echo

test5:
	@echo "============================================================\nTesting Image foot\n============================================================"
	./project3 $(IMAGE5) ${PT} foot
	@echo

test6:
	@echo "============================================================\nTesting Image fuel\n============================================================"
	./project3 $(IMAGE6) ${PT} fuel
	@echo

test7:
	@echo "============================================================\nTesting Image hydrogenAtom\n============================================================"
	./project3 $(IMAGE7) ${PT} hydrogenAtom
	@echo

test8:
	@echo "============================================================\nTesting Image lobster\n============================================================"
	./project3 $(IMAGE8) ${PT} lobster
	@echo

test9:
	@echo "============================================================\nTesting Image marschnerlobb\n============================================================"
	./project3 $(IMAGE9) ${PT} marschnerlobb
	@echo

test10:
	@echo "============================================================\nTesting Image mri_ventricles\n============================================================"
	./project3 $(IMAGE10) ${PT} mri_ventricles
	@echo

test11:
	@echo "============================================================\nTesting Image neghip\n============================================================"
	./project3 $(IMAGE11) ${PT} neghip
	@echo

test12:
	@echo "============================================================\nTesting Image nucleon\n============================================================"
	./project3 $(IMAGE12) ${PT} nucleon
	@echo

test13:
	@echo "============================================================\nTesting Image shockwave\n============================================================"
	./project3 $(IMAGE13) ${PT} shockwave
	@echo

test14:
	@echo "============================================================\nTesting Image silicium\n============================================================"
	./project3 $(IMAGE14) ${PT} silicium
	@echo

test15:
	@echo "============================================================\nTesting Image skull\n============================================================"
	./project3 $(IMAGE15) ${PT} skull
	@echo

test16:
	@echo "============================================================\nTesting Image statueLeg\n============================================================"
	./project3 $(IMAGE16) ${PT} statueLeg
	@echo

test17:
	@echo "============================================================\nTesting Image stent8\n============================================================"
	./project3 $(IMAGE17) ${PT} stent8
	@echo

clean:
	$(RM) *.o
	$(RM) project3
	$(RM) *.png

.PHONY: clean main
