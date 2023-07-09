{.localPassC: "-I.".}
{.compile: "simple_fft2.cpp".}
{.pragma: otfft, header: "simple_fft2.h".}

type
  complex_t* {.importcpp: "complex_t", otfft.} = object
    Re*: cdouble
    Im*: cdouble

  const_complex_vector* = ptr complex_t
  complex_vector* = ptr complex_t

  FFT* {.importcpp: "struct FFT", otfft.} = object


# proc fwdbut*(N: cint; x: complex_vector; W: const_complex_vector) {.importcpp: "fwdbut(@)", otfft.}
# proc invbut*(N: cint; x: complex_vector; W: const_complex_vector) {.importcpp: "invbut(@)", otfft.}

proc create_fft*(n: cint): ptr FFT {.importcpp: "&FFT(@)", otfft.}

# proc destroy_fft*(fft: ptr FFT) {.importcpp: "FFT::fwd", otfft.}

proc fwd*(fft: ptr FFT; x: complex_vector) {.importcpp: "#.fwd(@)", otfft.}
# proc fft_fwd0*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}
# proc fft_fwdu*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}
# proc fft_fwdn*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}
proc inv*(fft: ptr FFT; x: complex_vector) {.importcpp: "#.inv(@)", otfft.}
# proc fft_inv0*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}
# proc fft_invu*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}
# proc fft_invn*(fft: ptr FFT; x: complex_vector) {.cdecl, importc.}

# proc simd_alloc*(N: cint): complex_vector {.importcpp: "simd_alloc(@)", otfft.}

when isMainModule:
  let size: cint = 16
  var fft = create_fft(size)
  var x = newSeq[complex_t](size)
  # var x = simd_alloc(size)
  
  for i in 0..<size:
    x[i].Re = i.float
    x[i].Im = 0.0

  echo "a"
  for i in 0..<size:
    echo x[i].Re, " ", x[i].Im

  echo "b"
  fft.fwd(x[0].addr)
  for i in 0..<size:
    echo x[i].Re, " ", x[i].Im

  # TODO: SIMD malloc

  echo "c"
  fft.inv(x[0].addr)
  for i in 0..<size:
    echo x[i].Re, " ", x[i].Im
#   destroy_fft(fft)

