import math, complex, strutils
import std/sequtils
 
# Works with floats and complex numbers as input
proc fft_slow*[T: float | Complex[float]](x: openarray[T]): seq[Complex[float]] =
    let n = x.len
    if n == 0: 
        return
 
    result.newSeq(n)

    if n == 1:
        result[0] = (when T is float: complex(x[0]) else: x[0])
        return
 
    var evens, odds = newSeq[T]()
    for i, v in x:
        if i mod 2 == 0: 
            evens.add v
        else: 
            odds.add v
    
    var (even, odd) = (fft_slow(evens), fft_slow(odds))

    let halfn = n div 2
    for k in 0 ..< halfn:
        let a = exp(complex(0.0, -2 * Pi * float(k) / float(n))) * odd[k]
        result[k] = even[k] + a
        result[k + halfn] = even[k] - a
 
# OTFFT library
# http://wwwa.pikara.ne.jp/okojisan/otfft-en/optimization1.html

proc fft0(n: int, s: int, eo: bool, x: ptr seq[Complex[float]], y: ptr seq[Complex[float]]) =
    # n  : sequence length
    # s  : stride
    # eo : x is output if eo == 0, y is output if eo == 1
    # x  : input sequence(or output sequence if eo == 0)
    # y  : work area(or output sequence if eo == 1)

    let m: int = n div 2
    let theta0: float = 2.0*PI/float(n) 

    # if n == 2:
    #     var z: ptr seq[Complex[float]] = if eo: y else: x
    #     for q in 0..<s:
    #         let a: Complex[float] = x[q + 0]
    #         let b: Complex[float] = x[q + s]
    #         z[q + 0] = a + b
    #         z[q + s] = a - b
    # elif n >= 4:
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

proc fft*(xi: seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  

    # Input length has to be a power of two
    assert xi.len > 0
    assert xi.len.isPowerOfTwo()
    # assert ceil(log2(float(xi.len))) == floor(log2(float(xi.len)))
    
    var n: int = xi.len
    var x = xi.unsafeAddr
    var y: seq[Complex[float]] = newSeq[Complex[float]](n)
    fft0(n, 1, false, x, y.addr)
    # let fn = complex(1.0/float(n))
    # for k in 0..<n:
    #     x[k] *= fn

proc ifft*(xi: seq[Complex[float]]) =
    # n : sequence length
    # x : input/output sequence  
    var n: int = xi.len
    var x = xi.unsafeAddr

    let fn = complex(1.0/float(n))
    for p in 0..<n:
        x[p] = (x[p]*fn).conjugate
        # x[p] = x[p].conjugate

    var y: seq[Complex[float]] = newSeq[Complex[float]](n)
    fft0(n, 1, false, x, y.addr)

    for p in 0..<n:
        x[p] = x[p].conjugate

proc powerOf2Pad*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

if isMainModule:
    let tarr = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                -1.0,-1.0, 0.0, 0.0,-1.0,-1.0,-1.0,-1.0]

    for i in fft_slow(tarr):
        echo formatFloat(abs(i), ffDecimal, 3)

    echo "-----"

    var ts: seq[Complex[float]] = tarr.mapIt(complex(it, 0.0))
    fft(ts)
    for i in ts:
        echo formatFloat(abs(i), ffDecimal, 3)
    ifft(ts)
    assert ts.mapIt(round(it.re, 4)) == tarr
