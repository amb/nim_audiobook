import math, complex

when isMainModule:
    import benchy, strformat, strutils, sequtils
    import audiofile/[vorbis, wavfile]
    # import plottings, nimview

# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html

when defined(fftSpeedy):
    # 2^11
    const thetaLutSize = 2048
    const thetaLut = static:
        let step = 2.0*PI/float(thetaLutSize)
        var arr: array[thetaLutSize, array[2, float]]
        for k, v in mpairs(arr):
            v[0] = cos(step * float(k))
            v[1] = -sin(step * float(k))
        arr

type
    FFTArray = seq[Complex[float]]

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

    ## n and s must always be powers of 2

    assert n.isPowerOfTwo
    assert s.isPowerOfTwo
    assert s > 0
    let m: int = n div 2
    # let s = 1 shl sp

    when not defined(fftSpeedy):
        let theta0: float = 2.0*PI/float(n)

    if n == 1:
        if eo:
            for q in 0..<s:
                y[q] = x[q]
    else:
        for p in 0..<n div 2:
            when defined(fftSpeedy):
                # let fpl = thetaLut[(p shl 11) div n]
                let fpl = thetaLut[(p*thetaLutSize) div n]
                let wp  = complex(fpl[0], fpl[1])
            else:
                let fp = float(p)*theta0
                let wp = complex(cos(fp), -sin(fp))
            for q in 0..<s:
                let a = x[q + s*(p+0)]
                let b = x[q + s*(p+m)]
                y[q + s*(2*p+0)] =  a + b
                y[q + s*(2*p+1)] = (a - b) * wp

        fft0(n div 2, s * 2, not eo, y, x)

proc fft_empty_array*(v: FFTArray): FFTArray =
    result = newSeq[Complex[float]](v.len)

# proc fft_empty_array_complex*(v: int): FFTArray =
    # result = zeros[Complex[float]](v)

proc fft_array_len*(v: FFTArray): int =
    return v.len

proc fft*(x: var FFTArray) =
    # n : sequence length
    # x : input/output sequence

    # Input length has to be a power of two
    let alen = fft_array_len(x)
    assert alen > 0
    assert alen.isPowerOfTwo()

    var y = fft_empty_array(x)
    fft0(alen, 1, false, x, y)

proc ifft*(x: var FFTArray) =
    # n : sequence length
    # x : input/output sequence
    var n: int = fft_array_len(x)

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate

    var y = fft_empty_array(x)
    fft0(n, 1, false, x, y)

    for p in 0..<n:
        x[p] = x[p].conjugate

proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

when isMainModule:
    echo "Running fft.nim"

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

    var audio = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo
    var caudio = audio.mapIt(complex(it))
    echo "Sample len: ", fft_array_len(caudio)

    var y = fft_empty_array(caudio)
    let caudio_len = fft_array_len(caudio)
    timeIt "fft":
        # Non-AVX is faster when -d:lto
        fft0(caudio_len, 1, false, caudio, y)

    # nim c -d:release -d:lto -d:strip -d:danger -r fft.nim
    # nim c --cc:clang -d:release -d:danger --passC:"-flto" --passL:"-flto" -d:strip -r fft.nim && ll fft
    # https://github.com/kraptor/nim_callgrind
    
    # nim c -d:release fft.nim && valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no --instr-atstart=no ./fft
    
    # import nimprof
    # nim c -r -d:release --profiler:on --stackTrace:on fft.nim  
    # nim c -d:danger -d:fftspeedy --debugger:native spectrum.nim

    # fft ............................... 17.335 ms     17.770 ms    ±0.333   x280
