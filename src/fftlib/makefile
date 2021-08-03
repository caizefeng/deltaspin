include makefile.include

CPPSOURCES_FFTLIB = 	src/fftlib.cpp

.PHONY: all

all:
	make fftlib.o

fftlib.o: $(CPPSOURCES_FFTLIB)
	$(CXX_FFTLIB) $(INCS_FFTLIB) -c src/fftlib.cpp
	cp fftlib.o ..
