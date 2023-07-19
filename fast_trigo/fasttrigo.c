#include "fasttrigo.h"

// Most ninja tricks used here:
// http://fastcpp.blogspot.fr/2011/03/changing-sign-of-float-values-using-sse.html
// http://www.songho.ca/misc/sse/sse.html
// http://markplusplus.wordpress.com/2007/03/14/fast-sse-select-operation/
// http://www.masmforum.com/board/index.php?PHPSESSID=786dd40408172108b65a5a36b09c88c0&topic=9515.0
// http://cbloomrants.blogspot.fr/2010/11/11-20-10-function-approximation-by_20.html
// http://assemblyrequired.crashworks.org/2009/10/16/timing-square-root/
// http://nghiaho.com/?p=997
// http://www.researchgate.net/publication/3321724_Efficient_approximations_for_the_arctangent_function
// http://www.ganssle.com/approx/approx.pdf
// http://forum.allaboutcircuits.com/newsgroups/viewtopic.php?t=68185

const float invtwopi = 0.1591549f;
const float twopi = 6.283185f;
const float threehalfpi = 4.7123889f;
const float pi = 3.141593f;
const float halfpi = 1.570796f;
const float quarterpi = 0.7853982f;
// static const __m128 SIGNMASK =  _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
// TODO: might be slow
#define SIGNMASK _mm_castsi128_ps(_mm_set1_epi32(0x80000000))

float ft_atan(float x);
float ft_cos_32s(float x);

__m128 ft_atan_ps(__m128 x);
__m128 ft_cos_32s_ps(__m128 x);

float ft_sqrt(float squared)
{
    static int csr = 0;
    if (!csr)
        csr = _mm_getcsr() | 0x8040; // DAZ,FTZ (divide by zero=0)
    _mm_setcsr(csr);
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(squared))) * squared;
}
float ft_length(float x, float y)
{
    return ft_sqrt(x * x + y * y);
}
// float ft_length(float x, float y, float z)
// {
//     return ft_sqrt(x*x+y*y+z*z);
// }

float ft_atan(float x)
{
    return quarterpi * x - x * (fabs(x) - 1) * (0.2447f + 0.0663f * fabs(x));
}
float ft_atan2(float y, float x)
{
    if (fabs(x) > fabs(y))
    {
        float atan = ft_atan(y / x);
        if (x > 0.f)
            return atan;
        else
            return y > 0.f ? atan + pi : atan - pi;
    }
    else
    {
        float atan = ft_atan(x / y);
        if (x > 0.f)
            return y > 0.f ? halfpi - atan : -halfpi - atan;
        else
            return y > 0.f ? halfpi + atan : -halfpi + atan;
    }
}

float ft_cos_32s(float x)
{
    const float c1 = 0.99940307f;
    const float c2 = -0.49558072f;
    const float c3 = 0.03679168f;
    float x2; // The input argument squared
    x2 = x * x;
    return (c1 + x2 * (c2 + c3 * x2));
}
float ft_cos(float angle)
{
    // clamp to the range 0..2pi
    angle = angle - floorf(angle * invtwopi) * twopi;
    angle = angle > 0.f ? angle : -angle;

    if (angle < halfpi)
        return ft_cos_32s(angle);
    if (angle < pi)
        return -ft_cos_32s(pi - angle);
    if (angle < threehalfpi)
        return -ft_cos_32s(angle - pi);
    return ft_cos_32s(twopi - angle);
}
float ft_sin(float angle)
{
    return ft_cos(halfpi - angle);
}
void ft_sincos(float angle, float *sin, float *cos)
{
    // clamp to the range 0..2pi
    angle = angle - floorf(angle * invtwopi) * twopi;
    float sinmultiplier = angle > 0.f && angle < pi ? 1.f : -1.f;
    angle = angle > 0.f ? angle : -angle;

    if (angle < halfpi)
    {
        *cos = ft_cos_32s(angle);
        *sin = sinmultiplier * ft_sqrt(1.f - *cos * *cos);
        return;
    }
    if (angle < pi)
    {
        *cos = -ft_cos_32s(pi - angle);
        *sin = sinmultiplier * ft_sqrt(1.f - *cos * *cos);
        return;
    }
    if (angle < threehalfpi)
    {
        *cos = -ft_cos_32s(angle - pi);
        *sin = sinmultiplier * ft_sqrt(1.f - *cos * *cos);
        return;
    }
    *cos = ft_cos_32s(twopi - angle);
    *sin = sinmultiplier * ft_sqrt(1.f - *cos * *cos);
    return;
}

// PACKED SCALAR
__m128 ft_sqrt_ps(__m128 squared)
{
    static int csr = 0;
    if (!csr)
        csr = _mm_getcsr() | 0x8040; // DAZ,FTZ (divide by zero=0)
    _mm_setcsr(csr);
    return _mm_mul_ps(_mm_rsqrt_ps(squared), squared);
}
__m128 ft_length_ps(__m128 x, __m128 y)
{
    return ft_sqrt_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)));
}
// __m128 ft_length_ps(__m128 x, __m128 y, __m128 z)
// {
//     return ft_sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x,x),_mm_mul_ps(y,y)),_mm_mul_ps(z,z)));
// }

__m128 ft_atan_ps(__m128 x)
{
    //                                      quarterpi*x
    //                                      - x*(fabs(x) - 1)
    //                                      *(0.2447f+0.0663f*fabs(x));
    return _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(quarterpi), x),
                      _mm_mul_ps(_mm_mul_ps(x, _mm_sub_ps(_mm_andnot_ps(SIGNMASK, x), _mm_set1_ps(1.f))),
                                 (_mm_add_ps(_mm_set1_ps(0.2447f), _mm_mul_ps(_mm_set1_ps(0.0663f), _mm_andnot_ps(SIGNMASK, x))))));
}
__m128 ft_atan2_ps(__m128 y, __m128 x)
{
    __m128 absxgreaterthanabsy = _mm_cmpgt_ps(_mm_andnot_ps(SIGNMASK, x), _mm_andnot_ps(SIGNMASK, y));
    __m128 ratio = _mm_div_ps(_mm_add_ps(_mm_and_ps(absxgreaterthanabsy, y), _mm_andnot_ps(absxgreaterthanabsy, x)),
                              _mm_add_ps(_mm_and_ps(absxgreaterthanabsy, x), _mm_andnot_ps(absxgreaterthanabsy, y)));
    __m128 atan = ft_atan_ps(ratio);

    __m128 xgreaterthan0 = _mm_cmpgt_ps(x, _mm_set1_ps(0.f));
    __m128 ygreaterthan0 = _mm_cmpgt_ps(y, _mm_set1_ps(0.f));

    atan = _mm_xor_ps(atan, _mm_andnot_ps(absxgreaterthanabsy, _mm_and_ps(xgreaterthan0, SIGNMASK))); // negate atan if absx<=absy & x>0

    __m128 shift = _mm_set1_ps(pi);
    shift = _mm_sub_ps(shift, _mm_andnot_ps(absxgreaterthanabsy, _mm_set1_ps(halfpi))); // substract halfpi if absx<=absy
    shift = _mm_xor_ps(shift, _mm_andnot_ps(ygreaterthan0, SIGNMASK));                  // negate shift if y<=0
    shift = _mm_andnot_ps(_mm_and_ps(absxgreaterthanabsy, xgreaterthan0), shift);       // null if abs>absy & x>0

    return _mm_add_ps(atan, shift);
}

__m128 ft_cos_32s_ps(__m128 x)
{
    const __m128 c1 = _mm_set1_ps(0.99940307f);
    const __m128 c2 = _mm_set1_ps(-0.49558072f);
    const __m128 c3 = _mm_set1_ps(0.03679168f);
    __m128 x2; // The input argument squared
    x2 = _mm_mul_ps(x, x);
    //               (c1+           x2*          (c2+           c3*x2));
    return _mm_add_ps(c1, _mm_mul_ps(x2, _mm_add_ps(c2, _mm_mul_ps(c3, x2))));
}
__m128 ft_cos_ps(__m128 angle)
{
    // clamp to the range 0..2pi

    // take absolute value
    angle = _mm_andnot_ps(SIGNMASK, angle);
    // fmod(angle,twopi)
    angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(invtwopi)))), _mm_set1_ps(twopi))); // simplied SSE2 fmod, must always operate on absolute value
    // if SSE4.1 is always available, comment the line above and uncomment the line below
    // angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(pi)), SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(twopi), angle))));

    __m128 result = ft_cos_32s_ps(cosangle);

    result = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(threehalfpi))), SIGNMASK));
    return result;
}
__m128 ft_sin_ps(__m128 angle)
{
    return ft_cos_ps(_mm_sub_ps(_mm_set1_ps(halfpi), angle));
}
void ft_sincos_ps(__m128 angle, __m128 *sin, __m128 *cos)
{
    __m128 anglesign = _mm_or_ps(_mm_set1_ps(1.f), _mm_and_ps(SIGNMASK, angle));

    // clamp to the range 0..2pi

    // take absolute value
    angle = _mm_andnot_ps(SIGNMASK, angle);
    // fmod(angle,twopi)
    angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(invtwopi)))), _mm_set1_ps(twopi))); // simplied SSE2 fmod, must always operate on absolute value
    // if SSE4.1 is always available, comment the line above and uncomment the line below
    // angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(pi)), SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(twopi), angle))));

    __m128 result = ft_cos_32s_ps(cosangle);

    result = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(threehalfpi))), SIGNMASK));
    *cos = result;

    __m128 sinmultiplier = _mm_mul_ps(anglesign, _mm_or_ps(_mm_set1_ps(1.f), _mm_and_ps(_mm_cmpgt_ps(angle, _mm_set1_ps(pi)), SIGNMASK)));
    *sin = _mm_mul_ps(sinmultiplier, ft_sqrt_ps(_mm_sub_ps(_mm_set1_ps(1.f), _mm_mul_ps(result, result))));

    return;
}

void ft_interleave_ps(__m128 x0x1x2x3, __m128 y0y1y2y3, __m128 *x0y0x1y1, __m128 *x2y2x3y3)
{
    *x0y0x1y1 = _mm_unpacklo_ps(x0x1x2x3, y0y1y2y3);
    *x2y2x3y3 = _mm_unpackhi_ps(x0x1x2x3, y0y1y2y3);
}
void ft_deinterleave_ps(__m128 x0y0x1y1, __m128 x2y2x3y3, __m128 *x0x1x2x3, __m128 *y0y1y2y3)
{
    *x0x1x2x3 = _mm_shuffle_ps(x0y0x1y1, x2y2x3y3, _MM_SHUFFLE(2, 0, 2, 0));
    *y0y1y2y3 = _mm_shuffle_ps(x0y0x1y1, x2y2x3y3, _MM_SHUFFLE(3, 1, 3, 1));
}
