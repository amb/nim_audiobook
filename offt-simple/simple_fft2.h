/******************************************************************************
*  Simple FFT 2(Cooley Tukey Radix-4)
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef simple_fft2_h
#define simple_fft2_h

#include <utility>
#include "otfft_misc.h"

#ifdef DO_SINGLE_THREAD
static const int OMP_THRESHOLD = 1<<30;
#else
static const int OMP_THRESHOLD = 1<<15;
#endif

using namespace OTFFT_Complex;
using namespace OTFFT_MISC;

typedef complex_t* __restrict const complex_vector;
typedef const complex_t* __restrict const const_complex_vector;

void fwdbut(int N, complex_vector x, const_complex_vector W) noexcept;
void invbut(int N, complex_vector x, const_complex_vector W) noexcept;

struct FFT
{
    const int N;
    simd_array<complex_t> weight;
    simd_array<int> table;
    complex_t* const W;
    int* bitrev;

    FFT(int n);

    void fwd(complex_vector x) const noexcept;
    void fwd0(complex_vector x) const noexcept;
    void fwdu(complex_vector x) const noexcept;
    void fwdn(complex_vector x) const noexcept;

    void inv(complex_vector x) const noexcept;
    void inv0(complex_vector x) const noexcept;
    void invu(complex_vector x) const noexcept;
    void invn(complex_vector x) const noexcept;
};

#endif // simple_fft2_h
