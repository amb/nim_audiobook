# Converted to Nim from http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

# Note in the original C-file

# SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

# Inspired by Intel Approximate Math library, and based on the
# corresponding algorithms of the cephes math library

# The default is to use the SSE1 version. If you define USE_SSE2 the
# the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
# not expect any significant performance improvement with SSE2.

# LICENSE of the original C-file:

# Copyright (C) 2007  Julien Pommier

# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.

# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:

# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

#   (this is the zlib license)

import nimsimd/sse2

proc sincos_ps(x: M128; ptr s: M128; ptr c: M128) =
    var xmm1, xmm2, xmm3 = mm_setzero_ps()
    var sign_bit_sin, y: M128
    var emm0, emm2, emm4: M128i

    sign_bit_sin = x
    # take the absolute value
    x = mm_and_ps(x, cast[M128](_ps_inv_sign_mask))
    # extract the sign bit (upper one)
    sign_bit_sin = mm_and_ps(sign_bit_sin, cast[M128](_ps_sign_mask))

    # scale by 4/Pi
    y = mm_mul_ps(x, cast[M128](_ps_cephes_FOPI))

    # store the integer part of y in emm2
    emm2 = mm_cvttps_epi32(y)

    # j=(j+1) & (~1) (see the cephes sources)
    emm2 = mm_add_epi32(emm2, cast[M128i](_pi32_1))
    emm2 = mm_and_si128(emm2, cast[M128i](_pi32_inv1))
    y = mm_cvtepi32_ps(emm2)

    emm4 = emm2

    # get the swap sign flag for the sine
    emm0 = mm_and_si128(emm2, cast[M128i](_pi32_4))
    emm0 = mm_slli_epi32(emm0, 29)
    var swap_sign_bit_sin: M128 = cast[M128](emm0)

    # get the polynom selection mask for the sine
    emm2 = mm_and_si128(emm2, cast[M128i](_pi32_2))
    emm2 = mm_cmpeq_epi32(emm2, mm_setzero_si128())
    var poly_mask: M128 = cast[M128](emm2)

    # The magic pass: "Extended precision modular arithmetic"
    # x = ((x - y * DP1) - y * DP2) - y * DP3;
    xmm1 = cast[M128](_ps_minus_cephes_DP1)
    xmm2 = cast[M128](_ps_minus_cephes_DP2)
    xmm3 = cast[M128](_ps_minus_cephes_DP3)
    xmm1 = mm_mul_ps(y, xmm1)
    xmm2 = mm_mul_ps(y, xmm2)
    xmm3 = mm_mul_ps(y, xmm3)
    x = mm_add_ps(x, xmm1)
    x = mm_add_ps(x, xmm2)
    x = mm_add_ps(x, xmm3)

    emm4 = mm_sub_epi32(emm4, cast[M128i](_pi32_2))
    emm4 = mm_andnot_si128(emm4, cast[M128i](_pi32_4))
    emm4 = mm_slli_epi32(emm4, 29)
    var sign_bit_cos: M128 = cast[M128](emm4)

    sign_bit_sin = mm_xor_ps(sign_bit_sin, swap_sign_bit_sin)

    # Evaluate the first polynom (0 <= x <= Pi/4)
    var z: M128 = mm_mul_ps(x, x)
    y = cast[M128](_ps_coscof_p0)

    y = mm_mul_ps(y, z)
    y = mm_add_ps(y, cast[M128](_ps_coscof_p1))
    y = mm_mul_ps(y, z)
    y = mm_add_ps(y, cast[M128](_ps_coscof_p2))
    y = mm_mul_ps(y, z)
    y = mm_mul_ps(y, z)
    var tmp: M128 = mm_mul_ps(z, cast[M128](_ps_0p5))
    y = mm_sub_ps(y, tmp)
    y = mm_add_ps(y, cast[M128](_ps_1))

    # Evaluate the second polynom (Pi/4 <= x <= 0)
    var y2: M128 = cast[M128](_ps_sincof_p0)
    y2 = mm_mul_ps(y2, z)
    y2 = mm_add_ps(y2, cast[M128](_ps_sincof_p1))
    y2 = mm_mul_ps(y2, z)
    y2 = mm_add_ps(y2, cast[M128](_ps_sincof_p2))
    y2 = mm_mul_ps(y2, z)
    y2 = mm_mul_ps(y2, x)
    y2 = mm_add_ps(y2, x)

    # select the correct result from the two polynoms
    xmm3 = poly_mask
    var ysin2: M128 = mm_and_ps(xmm3, y2)
    var ysin1: M128 = mm_andnot_ps(xmm3, y)
    y2 = mm_sub_ps(y2, ysin2)
    y = mm_sub_ps(y, ysin1)

    xmm1 = mm_add_ps(ysin1, ysin2)
    xmm2 = mm_add_ps(y, y2)

    # update the sign
    s = mm_xor_ps(xmm1, sign_bit_sin)
    c = mm_xor_ps(xmm2, sign_bit_cos)