import math, complex
import arraymancer, x86_simd/x86_avx

when isMainModule:
    import benchy, strformat, strutils, sequtils
    import audiofile/[vorbis, wavfile]

# {.experimental: "parallel".}
# import std/threadpool

# import pocketfft/pocketfft

# import fftw3

# {.emit: """#include <valgrind/callgrind.h>""".}

# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html

proc mulpz(ab: m256d, xy: m256d): m256d {.inline.} =
    ## AVX float64 complex number multiplication
    let aa = unpacklo_pd(ab, ab)
    let bb = unpackhi_pd(ab, ab)
    let yx = shuffle_pd(xy, xy, 5)
    return addsub_pd(mul_pd(aa, xy), mul_pd(bb, yx))

const thetaLutSize = 2048
const thetaLut = static:
    let step = 2.0*PI/float(thetaLutSize)
    var arr: array[thetaLutSize, array[2, float]]
    for k, v in mpairs(arr):
        v[0] = cos(step * float(k))
        v[1] = -sin(step * float(k))
    # var arr: array[thetaLutSize, Complex[float]]
    # for k, v in mpairs(arr):
    #     v = complex(cos(step * float(k)), -sin(step * float(k)))
    arr

# thetaLutSize = 8
# let (cval, sval) = case int(float(p)*thetaL0)
# of 0: (1.0, -0.0)
# of 1: (0.7071067811865476, -0.7071067811865475)
# of 2: (0.0, -1.0)
# of 3: (-0.7071067811865475, -0.7071067811865476)
# of 4: (-1.0, 0.0)
# of 5: (-0.7071067811865477, 0.7071067811865475)
# of 6: (0.0, 1.0)
# of 7: (0.7071067811865474, 0.7071067811865477)
# else: (1.0, 1.0)

type
    FFTArray = seq[Complex[float]] | Tensor[Complex[float]]

proc fft0*[T: FFTArray](n: int, s: int, eo: bool, x: var T, y: var T) =
    ## Fast Fourier Transform
    ##
    ## Inputs:
    ## - `n` Sequence length. **Must be power of two**
    ## - `s` Stride
    ## - `eo` x is output if eo == 0, y is output if eo == 1
    ## - `x` Input sequence(or output sequence if eo == 0)
    ## - `y` Work area(or output sequence if eo == 1)
    ##
    ## Returns:
    ## - Output sequence, either `x` or `y`

    let m: int = n div 2
    let theta0: float = 2.0*PI/float(n)
    if n == 1:
        if eo:
            for q in 0..<s:
                y[q] = x[q]
    else:
        for p in 0..<m:
            let fp = float(p)*theta0
            let wp = complex(cos(fp), -sin(fp))
            for q in 0..<s:
                let a = x[q + s*(p+0)]
                let b = x[q + s*(p+m)]
                y[q + s*(2*p+0)] =  a + b
                y[q + s*(2*p+1)] = (a - b) * wp

        fft0(n div 2, 2 * s, not eo, y, x)

proc fft0_avx*[T: FFTArray](n: int, s: int, eo: bool, x: var T, y: var T) =
    ## Fast Fourier Transform
    ##
    ## Inputs:
    ## - `n` Sequence length. **Must be power of two**
    ## - `s` Stride
    ## - `eo` x is output if eo == 0, y is output if eo == 1
    ## - `x` Input sequence(or output sequence if eo == 0)
    ## - `y` Work area(or output sequence if eo == 1)
    ##
    ## Returns:
    ## - Output sequence, either `x` or `y`
    
    # {.emit: """CALLGRIND_START_INSTRUMENTATION; CALLGRIND_TOGGLE_COLLECT;""".}

    let m = n div 2

    if n == 2:
        if eo:
            for q in 0..<s:
                let a = x[q + 0]
                let b = x[q + s]
                y[q + 0] = a + b
                y[q + s] = a - b
        else:
            for q in 0..<s:
                let a = x[q + 0]
                let b = x[q + s]
                x[q + 0] = a + b
                x[q + s] = a - b

    elif n >= 4:
        let theta0: float = 2.0*PI/float(n)
        let thetaL0: float = float(thetaLutSize)/float(n)
        if s == 1:
            for p in 0..<m:
                let fp = float(p)*theta0
                let wp = complex(cos(fp), -sin(fp))
                let a = x[p+0]
                let b = x[p+m]
                y[2*p+0] =  a + b
                y[2*p+1] = (a - b) * wp
        else:
            for p in 0..<m:
                let fpl  = thetaLut[int(float(p)*thetaL0)]

                # let fp = float(p)*theta0
                # let cval =  cos(fp)
                # let sval = -sin(fp)

                let o0 = s*(p+0)
                let o1 = s*(p+m)
                let o2 = s*(2*p+0)
                let o3 = s*(2*p+1)
                # let wp = setr_pd(cval, sval, cval, sval)
                let wp = setr_pd(fpl[0], fpl[1], fpl[0], fpl[1])
                for q in countup(0, s-1, 2):
                    let a = load_pd_256(x[q+o0].re.addr)
                    let b = load_pd_256(x[q+o1].re.addr)
                    store_pd(y[q+o2].re.addr,           add_pd(a, b))
                    store_pd(y[q+o3].re.addr, mulpz(wp, sub_pd(a, b)))

        fft0_avx(n div 2, s * 2, not eo, y, x)

    # {.emit: """CALLGRIND_TOGGLE_COLLECT; CALLGRIND_STOP_INSTRUMENTATION;""".}

proc fft_empty_array*(v: FFTArray): FFTArray =
    when v is seq:
        result = newSeq[Complex[float]](v.len)
    elif v is Tensor:
        result = zeros[Complex[float]](v.shape[0])

proc fft_empty_array_complex*(v: int): FFTArray =
    result = zeros[Complex[float]](v)

proc fft_array_len*(v: FFTArray): int =
    result = 0
    when v is seq:
        result = v.len
    elif v is Tensor:
        result = v.shape[0]

proc fft*(x: var FFTArray) =
    # n : sequence length
    # x : input/output sequence

    # Input length has to be a power of two
    let alen = fft_array_len(x)
    assert alen > 0
    assert alen.isPowerOfTwo()

    var y = fft_empty_array(x)
    fft0_avx(alen, 1, false, x, y)

proc ifft*(x: var FFTArray) =
    # n : sequence length
    # x : input/output sequence
    var n: int = fft_array_len(x)

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate

    var y = fft_empty_array(x)
    fft0_avx(n, 1, false, x, y)

    for p in 0..<n:
        x[p] = x[p].conjugate

proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

when isMainModule:
    echo "Running main module..."

    let tarr = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                -1.0,-1.0, 0.0, 0.0,-1.0,-1.0,-1.0,-1.0]

    let fft_target = @[2.0, 6.15, 4.13, 3.4 , 1.41, 2.27, 1.71, 1.22,
                       0.0, 1.22, 1.71, 2.27, 1.41, 3.4 , 4.13, 6.15]
   
    var ts = tarr.mapIt(complex(it, 0.0))
    fft(ts)
    var fft_result = ts.mapIt(round(abs(it), 2))
    echo fft_target
    echo fft_result
    doAssert fft_result == fft_target
    doAssert ts[0].re != tarr[0]
    ifft(ts)
    var tsr = ts.mapIt(round(it.re, 4))
    doAssert tsr == tarr

    var vsf = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo.mapIt(complex(it))
    echo "Sample len: ", fft_array_len(vsf)
    # var vsf2 = vsf

    var y = fft_empty_array(vsf)
    timeIt "fft":
        # Non-AVX is faster when -d:lto
        fft0(fft_array_len(vsf), 1, false, vsf, y)

    timeIt "fft_avx":
        # Faster when only -d:release or debug
        fft0_avx(fft_array_len(vsf), 1, false, vsf, y)

    # Benchmark pocketFFT
    # var dOut = newSeq[Complex[float64]](vsf2.len)
    # let dInDesc = DataDesc[Complex[float64]].init(vsf2[0].addr, [vsf2.len])
    # var dOutDesc = DataDesc[Complex[float64]].init(dOut[0].addr, [dOut.len])
    # let fft = FFTDesc[float64].init(axes=[0], forward=true)
    # timeIt "pocketfft":
    #     fft.apply(dOutDesc, dInDesc)

    # nim c -d:release -d:lto -d:strip -d:danger -r fft.nim
    # nim c --cc:clang -d:release -d:danger --passC:"-flto" --passL:"-flto" -d:strip -r fft.nim && ll fft
    # https://github.com/kraptor/nim_callgrind
    
    # nim c -d:release fft.nim && valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no --instr-atstart=no ./fft
    
    # import nimprof
    # nim c -r -d:release --profiler:on --stackTrace:on fft.nim  
