/* FastTrigo 1.0 (c) 2013 Robin Lobel

  Fast yet accurate trigonometric functions

  Each namespace (FT, FTA) has 3 sets of functions:
  Scalar: standard trigonometric functions
  Packed Scalar: same functions computing 4 values at the same time (using SSE/SSE2/SSE3/SSE4.1 if available)
  Qt: convenience functions if using QVector2D/QVector3D classes from Qt

  FT Accuracy:
  FT::sqrt/sqrt_ps max error: 0.032% (average error: 0.0094%)
  FT::atan2/atan2_ps max error: 0.024% (0.0015 radians, 0.086 degrees)
  FT::cos/cos_ps max error: 0.06%
  FT::sin/sin_ps max error: 0.06%

  FT Speed up (MSVC2012 x64):
  FT::sqrt speed up: x2.5 (from standard sqrt)
  FT::atan2 speed up: x2.3 (from standard atan2)
  FT::sin/cos speed up: x1.9 (from standard sin/cos)
  FT::sincos speed up: x2.3 (from standard sin+cos)
  FT::sqrt_ps speed up: x8 (from standard sqrt)
  FT::atan2_ps speed up: x7.3 (from standard atan2)
  FT::sin_ps/cos_ps speed up: x4.9 (from standard sin/cos)
  FT::sincos_ps speed up: x6.2 (from standard sin+cos)

  Distributed under Revised BSD License
*/

#ifndef FASTTRIGO_H
#define FASTTRIGO_H

#include <math.h>
#include <intrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

float ft_sqrt(float squared);
float ft_length(float x, float y);
// float ft_length(float x, float y, float z);
float ft_atan2(float y, float x);
float ft_cos(float angle);
float ft_sin(float angle);
void  ft_sincos(float angle, float *sin, float *cos);

__m128 ft_sqrt_ps(__m128 squared);
__m128 ft_length_ps(__m128 x, __m128 y);
// __m128 ft_length_ps(__m128 x, __m128 y, __m128 z);
__m128 ft_atan2_ps(__m128 y, __m128 x);
__m128 ft_cos_ps(__m128 angle);
__m128 ft_sin_ps(__m128 angle);
void   ft_sincos_ps(__m128 angle, __m128 *sin, __m128 *cos);
void   ft_interleave_ps(__m128 x0x1x2x3, __m128 y0y1y2y3, __m128 *x0y0x1y1, __m128 *x2y2x3y3);
void   ft_deinterleave_ps(__m128 x0y0x1y1, __m128 x2y2x3y3, __m128 *x0x1x2x3, __m128 *y0y1y2y3);


#endif // FASTTRIGO_H
