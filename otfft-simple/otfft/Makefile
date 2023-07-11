#CXXFLAGS = /GS /GL /W3 /Gy /Zc:wchar_t /Zi /Gm- /O2 /fp:fast /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_LIB" /D "_UNICODE" /D "UNICODE" /errorReport:prompt /WX- /Zc:forScope /arch:AVX2 /Gd /Oi /MT /openmp /EHsc /nologo
CXXFLAGS = /O2 /MT /arch:AVX2 /openmp /EHsc /Oi /GL /nologo /D "NDEBUG" /fp:fast

HEADERS1 = \
	otfft_misc.h \
	otfft_avxdif4.h \
	otfft_avxdif4omp.h \
	otfft_avxdif8.h \
	otfft_avxdif8omp.h \
	otfft_avxdif8exp.h \
	otfft_avxdit4.h \
	otfft_avxdit4omp.h \
	otfft_avxdit8.h \
	otfft_avxdit8omp.h \
	otfft_avxdit8exp.h \
	otfft_sixstep.h \
	otfft_sixstepsq.h \
	otfft_eightstep.h \
	otfft_avxdif16.h \
	otfft_avxdif16omp.h \
	otfft_avxdif16exp.h \
	otfft_avxdit16.h \
	otfft_avxdit16omp.h \
	otfft_avxdit16exp.h \
	otfft_mixedradix.h

HEADERS2 = \
	otfft_gen_new.h \
	otfft_gen_setup.h \
	otfft_gen_fwd.h \
	otfft_gen_fwd0.h \
	otfft_gen_fwdu.h \
	otfft_gen_inv.h \
	otfft_gen_invu.h \
	otfft_gen_invn.h \
	otfft_gen_delete.h

all: ffttune rewrite

clean:
	@del *.obj *.exe *.pdb *.lib

otfft.obj: otfft.cpp otfft.h $(HEADERS1) $(HEADERS2) Makefile
	$(CXX) $(CXXFLAGS) /c otfft.cpp

otfft.lib: otfft.obj
	lib /ltcg otfft.obj /nologo

ffttune: ffttune.cpp $(HEADERS1) msleep.h Makefile
	$(CXX) $(CXXFLAGS) ffttune.cpp

rewrite: rewrite.cpp Makefile
	$(CXX) $(CXXFLAGS) rewrite.cpp
