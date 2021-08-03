include makefile.include

CPPSOURCES_PARS = sites.cpp functions.cpp radial.cpp basis.cpp
CSOURCES_PARS = lex.yy.c locproj.tab.c yywrap.c

CPPOBJ_PARS = $(CPPSOURCES_PARS:%.cpp=%.o)
COBJ_PARS = $(CSOURCES_PARS:%.c=%.o)

.PHONY: all

all:
	make libparser.a

libparser.a: $(CPPOBJ_PARS) $(COBJ_PARS) locproj.tab.h
	rm -f libparser.a
	ar vq libparser.a $(CPPOBJ_PARS) $(COBJ_PARS) locproj.tab.h

#locproj.tab.c locproj.tab.h: locproj.y
#	bison -d locproj.y

#lex.yy.c: locproj.l locproj.tab.h
#	flex locproj.l

%.o:	%.F90
	ifort -c $< -o $@

%.o:	%.c
	$(CXX_PARS) -D YY_parse_DEBUG=1 -c $< -o $@

%.o:	%.cpp
	$(CXX_PARS) -D YY_parse_DEBUG=1 -c $< -o $@

clean:
	rm *.o 

locproj_test:	call_from_fortran.o $(CPPOBJ_PARS) $(COBJ_PARS) locproj.tab.h
	ifort call_from_fortran.o $(CPPOBJ_PARS) $(COBJ_PARS)  -lstdc++ -o locproj_test
