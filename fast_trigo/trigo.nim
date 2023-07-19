{.compile: "fasttrigo.c".}

import nimsimd/[avx, sse2]

proc ft_sqrt*(x: cfloat): cfloat {.importc.}
proc ft_sincos*(a: cfloat; s, c: ptr cfloat) {.importc.}

# ft_sincos_ps(__m128 angle, __m128 *sin, __m128 *cos)
proc ft_sincos_ps*(a: M128; s, c: ptr M128) {.importc.}

when isMainModule:
    echo ft_sqrt(9.0)
    var s, c: cfloat
    ft_sincos(0.0, addr s, addr c)
    echo s, c

    var ms, mc: M128
    ft_sincos_ps(mm_set_ps(0.3, 0.2, 0.1, 0.0), addr ms, addr mc)
    var floatArray: array[4, float32]
    mm_store_ps(addr floatArray, ms)
    echo floatArray
    mm_store_ps(addr floatArray, mc)
    echo floatArray