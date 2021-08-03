.SUFFIXES:
.SUFFIXES: .mod .o .f90
.PRECIOUS: %.f90
# .SECONDEXPANSION:

FUFFIX=.F
SUFFIX=.f90

OFLAG_1=-O1
OFLAG_2=-O2
OFLAG_3=-O3
OFLAG_4=

OFLAG=$(OFLAG_2)
OFLAG_IN=$(OFLAG)

LIB=lib
LLIB=-Llib -ldmy

SRCDIR=../../src
BINDIR=../../bin

include .objects
include makefile.include

# FPP=$(CPP) $*$(FUFFIX) $*$(SUFFIX) $(CPP_OPTIONS)

FPP=$(filter-out -DwNGZhalf -DNGZhalf -DwNGXhalf -DNGXhalf,$(CPP))

DMPI=$(findstring DMPI,$(subst DMPI_,,$(CPP)))

LIB+=$(filter-out lib,$(LIBS))
LLIB+=$(filter-out -Llib -ldmy,$(LLIBS))

OBJS=$(SOURCE) $(OBJECTS)

ifndef VERSION
   ifeq ($(DMPI),DMPI)
      FPP+=-DNGZhalf
   else
      FPP+=-DNGXhalf
   endif
   EXE=vasp_std
else
ifeq ($(VERSION),std)
   ifeq ($(DMPI),DMPI)
      FPP+=-DNGZhalf
   else
      FPP+=-DNGXhalf
   endif
   EXE=vasp_std
else ifeq ($(VERSION),gam)
   ifeq ($(DMPI),DMPI)
      FPP+=-DNGZhalf -DwNGZhalf
   else
      FPP+=-DNGXhalf -DwNGXhalf
   endif
   EXE=vasp_gam
else ifeq ($(VERSION),ncl)
   EXE=vasp_ncl
else ifeq ($(VERSION),gpu)
   ifeq ($(DMPI),DMPI)
      FPP+=-DNGZhalf
   else
      FPP+=-DNGXhalf
   endif
   LIB  += CUDA
   LLIB += CUDA/lib/libCudaUtils_x86_64.a $(CUDA_LIB)
   EXE=vasp_gpu
   CPP_OPTIONS += $(CPP_GPU)
   OBJS=$(SOURCE_GPU) $(OBJECTS_GPU)
else ifeq ($(VERSION),gpu_ncl)
   EXE=vasp_gpu_ncl
   LIB  += CUDA
   LLIB += CUDA/lib/libCudaUtils_x86_64.a $(CUDA_LIB)
   CPP_OPTIONS += $(CPP_GPU)
   OBJS=$(SOURCE_GPU) $(OBJECTS_GPU)
else
   $(error $(VERSION) is not a valid target)
endif
endif

OBJS_O1=$(filter-out $(OBJECTS_O1) $(OBJECTS_O2) $(OBJECTS_O3) $(OBJECTS_IN),$(SOURCE_O1)) $(OBJECTS_O1)
OBJS_O2=$(filter-out $(OBJECTS_O1) $(OBJECTS_O2) $(OBJECTS_O3) $(OBJECTS_IN),$(SOURCE_O2)) $(OBJECTS_O2)
OBJS_O3=$(filter-out $(OBJECTS_O1) $(OBJECTS_O2) $(OBJECTS_O3) $(OBJECTS_IN),$(SOURCE_O3)) $(OBJECTS_O3)
OBJS_IN=$(filter-out $(OBJECTS_O1) $(OBJECTS_O2) $(OBJECTS_O3) $(OBJECTS_IN),$(SOURCE_IN)) $(OBJECTS_IN)

F90SRC_O1=$(OBJS_O1:.o=$(SUFFIX))
F90SRC_O2=$(OBJS_O2:.o=$(SUFFIX))
F90SRC_O3=$(OBJS_O3:.o=$(SUFFIX))
F90SRC_IN=$(OBJS_IN:.o=$(SUFFIX))

ifeq ($(FFLAGS_1),)
   FFLAGS_1=$(FFLAGS)
endif
ifeq ($(FFLAGS_2),)
   FFLAGS_2=$(FFLAGS)
endif
ifeq ($(FFLAGS_3),)
   FFLAGS_3=$(FFLAGS)
endif
ifeq ($(FFLAGS_IN),)
   FFLAGS_IN=$(FFLAGS)
endif

ifeq ($(OFLAG_1),)
   OFLAG_1=$(OFLAG)
endif
ifeq ($(OFLAG_2),)
   OFLAG_2=$(OFLAG)
endif
ifeq ($(OFLAG_3),)
   OFLAG_3=$(OFLAG)
endif

ifeq ($(INCS_1),)
   INCS_1=$(INCS)
endif
ifeq ($(INCS_2),)
   INCS_2=$(INCS)
endif
ifeq ($(INCS_3),)
   INCS_3=$(INCS)
endif
ifeq ($(INCS_IN),)
   INCS_IN=$(INCS)
endif


OBJCTS=$(filter %.o, $(OBJS) $(FFT3D))
#OBJCTS_f90=$(filter-out getshmem.o, $(OBJCTS))
F90SRC=$(OBJCTS:.o=$(SUFFIX))
F90SRC+=main.f90

-include .depend

# export

.PHONY: all cleanall clean sources dependencies depend libs $(LIB)

all: libs sources
	rm -f vasp ; $(MAKE) vasp ; cp vasp $(BINDIR)/$(EXE)
	
vasp: $(OBJS) $(FFT3D) $(INC) main.o
	$(FCL) -o vasp $(OBJS) main.o $(FFT3D) $(LLIB) $(LINK)

sources:
	rsync -u $(SRCDIR)/*.F $(SRCDIR)/*.inc .
#	-rsync -u $(SRCDIR)/*.f .

libs: $(LIB)

$(LIB):
	rsync -ru $(SRCDIR)/$@ .
	cp makefile.include $@
	$(MAKE) -C $@ -j1
#	$(MAKE) -C $@

dependencies: sources
	$(MAKE) depend

depend: $(F90SRC)
	makedepf90 $(F90SRC) > .depend

%$(SUFFIX): %$(FUFFIX)
	$(FPP)

%.o: %$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS) $(OFLAG) $(INCS) -c $*$(SUFFIX)

# $(OBJS_O1): $(F90SRC_O1)
$(OBJS_O1): %.o: %$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS_1) $(OFLAG_1) $(INCS_1) -c $*$(SUFFIX)

# $(OBJS_O2): $(F90SRC_O2)
$(OBJS_O2): %.o: %$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS_2) $(OFLAG_2) $(INCS_2) -c $*$(SUFFIX)

# $(OBJS_O3): $(F90SRC_O3)
$(OBJS_O3): %.o: %$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS_3) $(OFLAG_3) $(INCS_3) -c $*$(SUFFIX)

# $(OBJS_IN): $(F90SRC_IN)
$(OBJS_IN): %.o: %$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS_IN) $(OFLAG_IN) $(INCS_IN) -c $*$(SUFFIX)

cleanall: clean
	rm -rf $(LIB) *.F test.x

clean:
	rm -f *.o *.mod *.f90

#-----------------------------------------------------------------------
# special rules
#-----------------------------------------------------------------------

main.o : main$(SUFFIX)
	$(FC) $(FREE) $(FFLAGS) $(DEBUG) $(INCS) -c $*$(SUFFIX)

fft3dlib_f77.o: fft3dlib_f77$(SUFFIX)
	$(F77) $(FFLAGS_F77) -c $*$(SUFFIX)

