## ****************************************************************************
##   OTFFT C Header Version 11.5e
##
##   Copyright (c) 2016 OK Ojisan(Takuya OKAHISA)
##   Released under the MIT license
##   http://opensource.org/licenses/mit-license.php
## ****************************************************************************

# gcc -O3 -c hello.c && gcc hello.o otfft/otfft.o -lgomp -lm -lstdc++ -o hello && ./hello

when defined(gcc) and defined(no_prebuild):
    {.compile("otfft/otfft.cpp", "-lgomp -lm -lstdc++").}
    {.passL: "-lgomp -lm -lstdc++".}
elif defined(gcc):
    {.passL:"otfft/otfft.o".}
    {.passL:"-lgomp -lm -lstdc++".}
elif defined(vcc):
    {.passL:"otfft-simple/otfft/otfft.obj".}
    # {.passL:"/O2 /MT /arch:AVX2 /openmp /EHsc /Oi /GL /nologo /fp:fast".}
else:
    raise ValueError("Unknown compiler")



import std/complex

proc simd_malloc*(n: csize_t): pointer {.importc, cdecl.}
proc simd_free*(p: pointer) {.importc, cdecl.}

proc otfft_fft_new*(N: cint): pointer {.importc, cdecl.}
proc otfft_fft_delete*(p: pointer) {.importc, cdecl.}
proc otfft_fft_fwd*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_fwd0*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_fwdu*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_fwdn*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_inv*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_inv0*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_invu*(p: pointer; x: ptr Complex64) {.importc, cdecl.}
proc otfft_fft_invn*(p: pointer; x: ptr Complex64) {.importc, cdecl.}

# proc otfft_fft0_new*(N: cint): pointer
# proc otfft_fft0_delete*(p: pointer)
# proc otfft_fft0_fwd*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_fwd0*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_fwdu*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_fwdn*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_inv*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_inv0*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_invu*(p: pointer; x: ptr Complex64; y: ptr Complex64)
# proc otfft_fft0_invn*(p: pointer; x: ptr Complex64; y: ptr Complex64)

# proc otfft_rfft_new*(N: cint): pointer
# proc otfft_rfft_delete*(p: pointer)
# proc otfft_rfft_fwd*(p: pointer; x: ptr cdouble; y: ptr Complex64)
# proc otfft_rfft_fwd0*(p: pointer; x: ptr cdouble; y: ptr Complex64)
# proc otfft_rfft_fwdu*(p: pointer; x: ptr cdouble; y: ptr Complex64)
# proc otfft_rfft_fwdn*(p: pointer; x: ptr cdouble; y: ptr Complex64)
# proc otfft_rfft_inv*(p: pointer; x: ptr Complex64; y: ptr cdouble)
# proc otfft_rfft_inv0*(p: pointer; x: ptr Complex64; y: ptr cdouble)
# proc otfft_rfft_invu*(p: pointer; x: ptr Complex64; y: ptr cdouble)
# proc otfft_rfft_invn*(p: pointer; x: ptr Complex64; y: ptr cdouble)

# proc otfft_dct_new*(N: cint): pointer
# proc otfft_dct_delete*(p: pointer)
# proc otfft_dct_fwd*(p: pointer; x: ptr cdouble)
# proc otfft_dct_fwd0*(p: pointer; x: ptr cdouble)
# proc otfft_dct_fwdn*(p: pointer; x: ptr cdouble)
# proc otfft_dct_inv*(p: pointer; x: ptr cdouble)
# proc otfft_dct_inv0*(p: pointer; x: ptr cdouble)
# proc otfft_dct_invn*(p: pointer; x: ptr cdouble)

# proc otfft_dct0_new*(N: cint): pointer
# proc otfft_dct0_delete*(p: pointer)
# proc otfft_dct0_fwd*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)
# proc otfft_dct0_fwd0*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)
# proc otfft_dct0_fwdn*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)
# proc otfft_dct0_inv*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)
# proc otfft_dct0_inv0*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)
# proc otfft_dct0_invn*(p: pointer; x: ptr cdouble; y: ptr cdouble; z: ptr Complex64)

# proc otfft_bluestein_new*(N: cint): pointer
# proc otfft_bluestein_delete*(p: pointer)
# proc otfft_bluestein_fwd*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_fwd0*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_fwdu*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_fwdn*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_inv*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_inv0*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_invu*(p: pointer; x: ptr Complex64)
# proc otfft_bluestein_invn*(p: pointer; x: ptr Complex64)
