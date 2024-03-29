import math, complex, nimsimd/avx

when isMainModule:
    import benchy, strformat, strutils, sequtils
    import audiofile/[vorbis, wavfile]
    import utils

when defined(gcc):
    {.passc: "-mavx".}

when compileOption("threads"):
    when defined(gcc):
        {.passc: "-fopenmp".}
        {.passl: "-lgomp".}
    when defined(vcc):
        {.passc: "/openmp".}


# Most ideas and solutions implemented from here:
# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html


proc log2(n: int): int =
    ## Returns the log2 of an integer
    var tn = n
    while tn > 1:
        tn = tn shr 1
        inc result


when defined(fftLUT):
    # 2^12
    const thetaLutMargin = 8
    const thetaLutSize = 4096
    const thetaLut = static:
        let step = 2.0*PI/float(thetaLutSize)
        var arr: array[thetaLutSize + thetaLutMargin, Complex[float]]
        for k in 0..<thetaLutSize + thetaLutMargin:
            arr[k] = complex(cos(step * float(k)), -sin(step * float(k)))
        arr


proc fft0*(n: int, s: int, eo: bool, x, y: var seq[Complex[float]]) =
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
    assert s == 0 or s.isPowerOfTwo
    let m: int = n div 2

    when not defined(fftLUT):
        let theta0: float = 2.0*PI/float(n)

    if n == 1:
        if eo:
            for q in 0..<s:
                y[q] = x[q]
    else:
        for p in 0..<m:
            when defined(fftLUT):
                let wp = thetaLut[(p*thetaLutSize) div n]
            else:
                let fp = float(p)*theta0
                let wp = complex(cos(fp), -sin(fp))

            let sp0 = s*(p+0)
            let spm = s*(p+m)
            let s2p0 = s*(2*p+0)
            let s2p1 = s*(2*p+1)

            for q in 0..<s:
                let a = x[q + sp0]
                let b = x[q + spm]
                y[q + s2p0] = a + b
                y[q + s2p1] = (a - b) * wp

        fft0(m, s * 2, not eo, y, x)


proc ffts0*(n: int, x, y: ptr UncheckedArray[Complex[float]]) =
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
    assert n >= 2

    var theta0: float = 2.0*PI/float(n)
    var nd = n
    var sd = 1
    var eod = false
    var fc = 0.0
    var ndd = log2(nd)

    # inc fftCalls
    # fftProcessed += n.uint64
    # if n <= 8:
    #     inc fftSmall

    template inner_piece(tx, ty: untyped): untyped =
        fc = 0.0
        theta0 = 2.0*PI/float(nd)
        nd = nd div 2
        for p in 0..<nd:
            when defined(fftLUT):
                let wp = thetaLut[(p*thetaLutSize) shr ndd]
            else:
                let fp = fc*theta0
                fc += 1.0
                let wp = complex(cos(fp), -sin(fp))

            let sp0 = sd*(p+0)
            let spm = sd*(p+nd)
            let s2p0 = sd*(2*p+0)
            let s2p1 = sd*(2*p+1)

            for q in 0..<sd:
                let a = tx[q + sp0]
                let b = tx[q + spm]
                ty[q + s2p0] = a + b
                ty[q + s2p1] = (a - b) * wp

        sd = sd * 2
        eod = not eod
        dec ndd

    while nd > 1:
        # butterfly
        inner_piece(x, y)
        if nd <= 1:
            break
        inner_piece(y, x)
    if nd == 1:
        if eod:
            for q in 0..<sd:
                x[q] = y[q]


proc mulpz(ab: M256d, xy: M256d): M256d {.inline.} =
    ## AVX float64 complex number multiplication
    let aa = mm256_unpacklo_pd(ab, ab)
    let bb = mm256_unpackhi_pd(ab, ab)
    let yx = mm256_shuffle_pd(xy, xy, 5)
    return mm256_addsub_pd(mm256_mul_pd(aa, xy), mm256_mul_pd(bb, yx))


proc fft0x*(n: int, x, y: ptr UncheckedArray[Complex[float]]) =
    ## Fast Fourier Transform
    ##
    ## Inputs:
    ## - `n` Sequence length. **Must be power of two**
    ## - `x` Input sequence
    ## - `y` Work area
    ##
    ## Returns:
    ## - Output sequence in `x`

    assert n.isPowerOfTwo

    var theta0: float = 2.0*PI/float(n)
    var nd = n
    var sd = 1
    var eod = false
    var fc = 0.0
    # var ndd = log2(nd)

    template inner_piece(tx, ty: untyped): untyped =
        fc = 0.0
        theta0 = 2.0*PI/float(nd)
        nd = nd div 2
        for p in 0..<nd:
            # when defined(fftLUT):
            #     let wpl = thetaLut[(p*thetaLutSize) shr ndd]
            #     let wp = mm256_setr_pd(wpl.re, wpl.im, wpl.re, wpl.im)
            # else:
            # Efficient sincos seems to happen through compiler optimization 
            let cval = cos(fc)
            let sval = -sin(fc)
            let wp = mm256_setr_pd(cval, sval, cval, sval)
            fc += theta0

            let sp0 = sd*p
            let spm = sp0 + sd*nd
            let s2p0 = sp0 * 2
            let s2p1 = s2p0 + sd

            for q in countup(0, sd-1, 2):
                let a = mm256_load_pd(tx[q+sp0].re.addr)
                let b = mm256_load_pd(tx[q+spm].re.addr)
                mm256_store_pd(ty[q+s2p0].re.addr, mm256_add_pd(a, b))
                mm256_store_pd(ty[q+s2p1].re.addr, mulpz(wp, mm256_sub_pd(a, b)))
        sd = sd * 2
        eod = not eod
        # dec ndd

    # First inner_piece where sd == 1
    nd = nd div 2
    for p in 0..<nd:
        let sp0 = sd*p
        let spm = sp0 + sd*nd
        let s2p0 = sp0 * 2
        let s2p1 = s2p0 + sd
        let fp = float(p)*theta0
        let a = x[0 + sp0]
        let b = x[0 + spm]
        y[0 + s2p0] = a + b
        y[0 + s2p1] = (a - b) * complex(cos(fp), -sin(fp))
    sd = sd * 2
    eod = not eod
    # dec ndd

    while nd > 1:
        # butterfly
        inner_piece(y, x)
        if nd <= 1:
            break
        inner_piece(x, y)
        
    if nd == 1 and eod:
        for q in 0..<sd:
            x[q] = y[q]


proc unsafeArray[T](x: seq[T], loc: int): ptr UncheckedArray[T] =
    return cast[ptr UncheckedArray[T]](x[loc].unsafeAddr)


proc sixstep_fft(log_N: int, x, y: var seq[Complex[float]]) =
    let N = 1 shl log_N
    let n = 1 shl (log_N div 2)

    # transpose x
    for k in 0||(n-1):
        for p in k+1..<n:
            swap(x[p + k*n], x[k + p*n])

    # FFT all p-line of x
    for p in 0||(n-1):
        fft0x(n, x.unsafeArray(p*n), y.unsafeArray(p*n))

    # multiply twiddle factor and transpose x
    for p in 0||(n-1):
        let theta0 = float(2 * p) * PI / float(N)
        for k in p..<n:
            let theta = float(k) * theta0
            let wkp = complex(cos(theta), -sin(theta))
            if k == p:
                x[p + p*n] *= wkp
            else:
                let a = x[k + p*n] * wkp
                let b = x[p + k*n] * wkp
                x[k + p*n] = b
                x[p + k*n] = a

    # FFT all k-line of x
    for p in 0||(n-1):
        fft0x(n, x.unsafeArray(p*n), y.unsafeArray(p*n))

    # transpose x
    for k in 0||(n-1):
        for p in k+1..<n:
            swap(x[p + k*n], x[k + p*n])


proc fft_singleshot*(x: var seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence

    # Input length has to be a power of two
    let alen = x.len
    assert alen > 0
    assert alen.isPowerOfTwo()
    let lalen = log2(alen)

    var y = newSeq[Complex[float]](x.len)
    if x.len < 512 or lalen mod 2 == 1:
        fft0x(alen, x.unsafeArray(0), y.unsafeArray(0))
    else:
        sixstep_fft(lalen, x, y)

# proc ifft*(x: var seq[Complex[float]]) =
#     # n : sequence length
#     # x : input/output sequence
#     var n: int = fft_array_len(x)

#     let fn = complex(1.0/float(n))
#     for p in 0..<n:
#         x[p] = (x[p]*fn).conjugate

#     var y = fft_empty_array(x)
#     fft0(n, 1, false, x, y)

#     for p in 0..<n:
#         x[p] = x[p].conjugate

when isMainModule:
    proc loadAudioSample(filename: string): seq[Complex[float]] =
        var audio = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo
        var caudio = newSeq[Complex[float]](audio.len)
        for i in 0..<audio.len:
            caudio[i] = complex(audio[i], 0.0)
        return caudio


    proc checkError(x, y: seq[Complex[float]]): float =
        var error = 0.0
        for i in 0..<x.len:
            error += abs(x[i] - y[i])
        return error / float(x.len)

    echo "Running fft.nim"

    doAssert log2(2) == 1
    doAssert log2(4) == 2
    doAssert log2(8) == 3
    doAssert log2(16) == 4
    doAssert log2(64) == 6

    let tarr = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]

    let fft_target = @[2.0, 6.15, 4.13, 3.4, 1.41, 2.27, 1.71, 1.22,
                       0.0, 1.22, 1.71, 2.27, 1.41, 3.4, 4.13, 6.15]

    var ts = tarr.map(proc(x: float): Complex[float] = complex(x, 0.0))
    ts.fft_singleshot
    var fft_result = ts.map(proc(x: Complex[float]): float = round(abs(x), 2))

    echo fft_target
    echo fft_result

    # doAssert fft_result == fft_target
    # doAssert ts[0].re != tarr[0]

    # ifft(ts)
    # var tsr = ts.mapIt(round(it.re, 4))
    # doAssert tsr == tarr

    var caudio = loadAudioSample("data/sample.ogg")
    let clog2 = log2(caudio.len)
    var caudio2 = caudio

    var y = newSeq[Complex[float]](caudio.len)
    fft0x(caudio2.len, caudio2.unsafeArray(0), y.unsafeArray(0))
    sixstep_fft(clog2, caudio, y)

    let err_val = checkError(caudio, caudio2)
    echo fmt"Six step error: {err_val:.6f}"
    # doAssert err_val < 0.0001

    timeIt "fft":
        sixstep_fft(clog2, caudio, y)
        # fft0x(caudio.len, caudio.unsafeArray(0), y.unsafeArray(0))
        # ffts0(caudio.len, caudio.unsafeArray(0), y.unsafeArray(0))

    # nim c -d:lto -d:strip -d:danger -r fft.nim
    # nim c --cc:clang -d:release -d:danger --passC:"-flto" --passL:"-flto" -d:strip -r fft.nim && ll fft
    # https://github.com/kraptor/nim_callgrind

    # nim c -d:release fft.nim && valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no --instr-atstart=no ./fft

    # import nimprof
    # nim c -r -d:release --profiler:on --stackTrace:on fft.nim
    # nim c -d:danger -d:fftspeedy --debugger:native spectrum.nim

    # with speedy:
    # fft ............................... 17.335 ms     17.770 ms    ±0.333   x280

    # superl & sleepy
    # nim --cc:vcc c -r -d:danger --stackTrace:on --debugger:native fft.nim

    # gcc needs --passc:"-mavx" for avx

    # openmp
    # nim c -r -d:danger --passc:"-fopenmp" --passl:"-lgomp" --threads:on fft.nim
    # nim --cc:vcc c -r -d:danger -d:lto -d:fftLUT --passc:"/openmp" --threads:on fft.nim