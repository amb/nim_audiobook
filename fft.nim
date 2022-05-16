import math, complex, strutils
import std/[sequtils]
import vorbis, wavfile
import benchy
import arraymancer

# import pocketfft/pocketfft
# import fftw3

# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html

# This is +20-50% improvement
const thetaLutSize = 2048
const thetaLut = static:
    var arr: array[thetaLutSize, Complex[float]]
    let step = 2.0*PI/float(thetaLutSize)
    for k, v in mpairs(arr):
        v = complex(cos(step * float(k)), -sin(step * float(k)))
    arr

proc fft0_seq(n: int, s: int, eo: bool, x: var seq[Complex[float]], y: var seq[Complex[float]]) =
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
    # let theta0: float = float(thetaLutSize)/float(n) 
    if n == 1:
        if eo:
            for q in 0..<s:
                y[q] = x[q]
    else:
        for p in 0..<m:
            let fp = float(p)*theta0
            let wp = complex(cos(fp), -sin(fp))
            # let fp = int(float(p)*theta0)
            # let wp = thetaLut[fp]
            for q in 0..<s:
                let a = x[q + s*(p+0)]
                let b = x[q + s*(p+m)]
                y[q + s*(2*p+0)] =  a + b
                y[q + s*(2*p+1)] = (a - b) * wp
        fft0_seq(n div 2, 2 * s, not eo, y, x)

proc fft_seq*(x: var seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  

    # Input length has to be a power of two
    assert x.len > 0
    assert x.len.isPowerOfTwo()
    
    var y: seq[Complex[float]] = newSeq[Complex[float]](x.len)
    fft0_seq(x.len, 1, false, x, y)

proc ifft_seq*(x: var seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  
    var n: int = x.len

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate

    var y: seq[Complex[float]] = newSeq[Complex[float]](n)
    fft0_seq(n, 1, false, x, y)

    for p in 0..<n:
        x[p] = x[p].conjugate

proc fft0(n: int, s: int, eo: bool, x: var Tensor[Complex[float]], y: var Tensor[Complex[float]]) =
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

    var 
        ln = n
        ls = s
        leo = eo

    while ln > 1:
        let theta0: float = 2.0*PI/float(ln) 
        ln = ln div 2
        for p in 0..<ln:
            let fp = float(p)*theta0
            let wp = complex(cos(fp), -sin(fp))
            let a = x[ls*(p+0)..<ls*(p+0+1)]
            let b = x[ls*(p+ln)..<ls*(p+ln+1)]
            y[ls*(2*p+0)..<ls*(2*p+1)] =  a + b
            y[ls*(2*p+1)..<ls*(2*p+2)] = (a - b) * wp

        ls = ls * 2
        leo = not leo
        (y, x) = (x, y)

    if ln == 1:
        if leo:
            for q in 0..<ls:
                y[q] = x[q]

proc fft*(x: var Tensor[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  

    # Input length has to be a power of two
    assert x.shape[0] > 0
    assert x.shape[0].isPowerOfTwo()
    
    var y: Tensor[Complex[float]] = zeros[Complex[float]](x.shape[0])
    fft0(x.shape[0], 1, false, x, y)

proc ifft*(x: var Tensor[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  
    var n: int = x.shape[0]

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate

    var y: Tensor[Complex[float]] = zeros[Complex[float]](n)
    fft0(n, 1, false, x, y)

    for p in 0..<n:
        x[p] = x[p].conjugate


proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

if isMainModule:
    echo "Running main module..."

    let tarr = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                -1.0,-1.0, 0.0, 0.0,-1.0,-1.0,-1.0,-1.0]

    let fft_target = @[2.0, 6.15, 4.13, 3.4 , 1.41, 2.27, 1.71, 1.22, 
                       0.0, 1.22, 1.71, 2.27, 1.41, 3.4 , 4.13, 6.15]

    var ts: Tensor[Complex[float]] = tarr.mapIt(complex(it, 0.0)).toTensor()
    fft(ts)
    var fft_result = ts.mapIt(round(abs(it), 2))
    doAssert fft_result == fft_target
    doAssert ts[0].re != tarr[0]
    ifft(ts)
    var tsr = ts.mapIt(round(it.re, 4))
    doAssert tsr == tarr

    # Use nextPowerOfTwo instead
    # doAssert maxPowerOfTwo(1048576) == 20
    # doAssert maxPowerOfTwo(2) == 1
    # doAssert maxPowerOfTwo(16) == 4

    var vsf = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo.mapIt(complex(it)).toTensor()
    var vsf2 = vsf

    var y: Tensor[Complex[float]] = zeros[Complex[float]](vsf.shape[0])
    timeIt "fft":
        fft0(vsf.shape[0], 1, false, vsf, y)
        #fft(vsf)

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
