import math, complex, strutils
import std/[sequtils, times, os]
import vorbis, wavfile
import benchy


proc maxPowerOfTwo(v: int): int =
    # How many time can <v> be divided by two until it's 1 or less
    var count = 0
    var tv = v
    while tv > 0:
        count += 1
        tv = tv shr 1
    if count > 0:
        count -= 1
    return count


# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html

# Increase this to a higher power of two to handle bigger FFT buffers
const thetaLutStep = 32

const thetaLut = static:
    var arr: array[thetaLutStep+1, Complex[float]]
    let step = 2.0*PI/float(1 shl thetaLutStep)
    for k, v in mpairs(arr):
        v = complex(cos(step * float(k)), -sin(step * float(k)))
    arr

proc fft0(n: int, s: int, eo: bool, x: var seq[Complex[float]], y: var seq[Complex[float]]) =
    # n  : sequence length
    # s  : stride
    # eo : x is output if eo == 0, y is output if eo == 1
    # x  : input sequence(or output sequence if eo == 0)
    # y  : work area(or output sequence if eo == 1)
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
                let a: Complex[float] = x[q + s*(p+0)]
                let b: Complex[float] = x[q + s*(p+m)]
                y[q + s*(2*p + 0)] =  a + b
                y[q + s*(2*p + 1)] = (a - b) * wp
        fft0(n div 2, 2 * s, not eo, y, x)

proc fft1(n: int, s: int, eo: bool, x: ptr seq[Complex[float]], y: ptr seq[Complex[float]]) =
    # n  : sequence length
    # s  : stride
    # eo : x is output if eo == 0, y is output if eo == 1
    # x  : input sequence(or output sequence if eo == 0)
    # y  : work area(or output sequence if eo == 1)

    var tn = n
    var ts = s
    var teo = eo

    var ptx = x
    var pty = y

    while tn > 1:
        let m = tn shr 1
        let theta0: float = 2.0*PI/float(tn) 
        for p in 0..<m:
            let fp = float(p)*theta0
            let wp = complex(cos(fp), -sin(fp))
            for q in 0..<ts:
                let a = ptx[q + ts*(p+0)]
                let b = ptx[q + ts*(p+m)]
                pty[q + ts*(2*p+0)] =  a + b
                pty[q + ts*(2*p+1)] = (a - b) * wp

        tn = tn shr 1
        ts = ts shl 1
        teo = not teo
        (pty, ptx) = (ptx, pty)

    if tn == 1 and teo:
        for q in 0..<ts:
            y[q] = x[q]


proc fft*(x: var seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  

    # Input length has to be a power of two
    assert x.len > 0
    assert x.len.isPowerOfTwo()
    
    var y: seq[Complex[float]] = newSeq[Complex[float]](x.len)
    fft1(x.len, 1, false, x.addr, y.addr)
    # fft0(x.len, 1, false, x, y)

proc ifft*(x: var seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  
    var n: int = x.len

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate

    var y: seq[Complex[float]] = newSeq[Complex[float]](n)
    fft1(n, 1, false, x.addr, y.addr)
    # fft0(n, 1, false, x, y)

    for p in 0..<n:
        x[p] = x[p].conjugate

proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

if isMainModule:
    let tarr = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                -1.0,-1.0, 0.0, 0.0,-1.0,-1.0,-1.0,-1.0]

    let fft_target = @[2.0, 6.15, 4.13, 3.4 , 1.41, 2.27, 1.71, 1.22, 
                       0.0, 1.22, 1.71, 2.27, 1.41, 3.4 , 4.13, 6.15]

    echo thetaLut.mapIt(abs(it))[0..10]

    var ts: seq[Complex[float]] = tarr.mapIt(complex(it, 0.0))
    fft(ts)
    var fft_result = ts.mapIt(round(abs(it), 2))
    doAssert fft_result == fft_target
    doAssert ts[0].re != tarr[0]
    ifft(ts)
    var tsr = ts.mapIt(round(it.re, 4))
    doAssert tsr == tarr

    assert maxPowerOfTwo(-5) == 0
    assert maxPowerOfTwo(0) == 0
    assert maxPowerOfTwo(1) == 0
    assert maxPowerOfTwo(2) == 1
    assert maxPowerOfTwo(3) == 1
    assert maxPowerOfTwo(4) == 2
    assert maxPowerOfTwo(5) == 2
    assert maxPowerOfTwo(32) == 5
    assert maxPowerOfTwo(62) == 5

    # assert maxPowerOfTwo(16) == 1 shl 4
    # assert maxPowerOfTwo(4) == 1 shl 2

    var vsf = loadVorbis("sample.ogg").toFloat.padPowerOfTwo.mapIt(complex(it))
    timeIt "fft":
       fft(vsf)
        
    # nim c -d:release -d:lto -d:strip -d:danger -r fft.nim   
    # https://github.com/kraptor/nim_callgrind
