import std/[math, complex, times, strformat]
import otfft

const
    n_max = 22
    N_max = 1 shl n_max
    nN_max = n_max * N_max

proc imin(x, y: int): int =
    if x < y: x else: y

proc main() =
    var x = newSeq[Complex[float64]](N_max)
    for n in 1..n_max:
        stdout.write fmt"2^({n}): "        
        let N = 1 shl n
        let LOOPS = imin(70, n * 4) * (nN_max div (n * N))
        for p in 0..<N:
            let t = p.float / N.float
            x[p] = complex(
                10.0 * cos(3.0 * 2.0 * PI * t * t), 
                10.0 * sin(3.0 * 2.0 * PI * t * t))

        var obj = otfft_fft_new(N.cint)

        let t1 = epochTime()
        let xaddr = x[0].addr
        for i in 0..<LOOPS:
            obj.otfft_fft_fwd(xaddr)
            obj.otfft_fft_inv(xaddr)
        let t2 = epochTime()
        
        let tusec = float(t2 - t1) * 1000000.0 / float(LOOPS)
        echo fmt"{tusec:.3f} [usec], {LOOPS} loops"

when isMainModule:
    main()
