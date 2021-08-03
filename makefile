VERSIONS = std gam ncl gpu gpu_ncl

.PHONY: all veryclean versions $(VERSIONS)

all: std gam ncl

versions: $(VERSIONS)

$(VERSIONS):
	mkdir build/$@ ; \
	cp src/makefile src/.objects makefile.include build/$@ ; \
	$(MAKE) -C build/$@ VERSION=$@ all
#	$(MAKE) -C build/$@ VERSION=$@ dependencies -j1 ; \
#	$(MAKE) -C build/$@ VERSION=$@ all


veryclean: 
	rm -rf build/*
