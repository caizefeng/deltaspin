.SUFFIXES:
.SUFFIXES: .inc .o .f .f90 .c
.PRECIOUS: %.f90

FUFFIX=.F
SUFFIX=.f90

include makefile.include

FPP=$(filter-out $(CPP_OPTIONS),$(CPP_LIB))

OBJS_STD = preclib.o timing_.o derrf_.o dclock_.o diolib.o dlexlib.o drdatab.o
OBJS = $(filter-out $(OBJECTS_LIB),$(OBJS_STD)) $(OBJECTS_LIB)

.PHONY: all

all:
	make libdmy.a

libdmy.a: $(OBJS)
	rm -f libdmy.a
	ar vq libdmy.a $(OBJS)

%$(SUFFIX): %.F
	$(FPP)

%.o: %$(SUFFIX)
	$(FC_LIB) $(FFLAGS_LIB) $(FREE_LIB) -c -o $@ $^

%.o: %.c
	$(CC_LIB) $(CFLAGS_LIB) -c -o $@ $^


# files which do not require autodouble 
lapack_min.o: lapack_min.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c lapack_min.f
lapack_double.o: lapack_double.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c lapack_double.f
lapack_single.o: lapack_single.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c lapack_single.f
lapack_atlas.o: lapack_atlas.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c lapack_atlas.f
linpack_double.o: linpack_double.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c linpack_double.f
linpack_single.o: linpack_single.f
	$(FC_LIB) $(FFLAGS_LIB) $(NOFREE) -c linpack_single.f


