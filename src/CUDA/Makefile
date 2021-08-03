all: release 
clean: clean-release clean-debug clean-emurelease clean-emudebug

release:
	make -f Makefile_CUDASDK verbose=1
clean-release:
	rm -f lib/* obj/x86_64/release/*

debug:
	make -f Makefile_CUDASDK verbose=1 dbg=1
clean-debug:
	rm -f lib/debug/* obj/debug/*

emurelease:
	make -f Makefile_CUDASDK_EMU verbose=1 emu=1
clean-emurelease:
	rm -f lib/emurelease/* obj/emurelease/*

emudebug:
	make -f Makefile_CUDASDK_EMU verbose=1 emu=1 dbg=1
clean-emudebug:
	rm -f lib/emudebug/* obj/emudebug/*

