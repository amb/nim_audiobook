/******************************************************************************
*  OTFFT AVXDIF(Radix-8) of Exponential Version 11.5ev
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdif8exp_h
#define otfft_avxdif8exp_h

namespace OTFFT_AVXDIF8exp { //////////////////////////////////////////////////

using namespace OTFFT_MISC;

///////////////////////////////////////////////////////////////////////////////
// Forward Buffterfly Operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct fwdcore
{
    static constexpr int N  = n*s;
    static constexpr int N0 = 0;
    static constexpr int N1 = N/8;
    static constexpr int N2 = N1*2;
    static constexpr int N3 = N1*3;
    static constexpr int N4 = N1*4;
    static constexpr int N5 = N1*5;
    static constexpr int N6 = N1*6;
    static constexpr int N7 = N1*7;
    static constexpr int Ni = N1/8;
    static constexpr int h  = s/8;
    static constexpr double theta = -2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < Ni; i++) {
            const int p = i / h;
            const int q = i % h * 8;
            const int sp = s*p;
            const int s8p = 8*sp;
            complex_vector xq_sp  = x + q + sp;
            complex_vector yq_s8p = y + q + s8p;

            const amm w1p = duppz9(expj(theta*sp));

            const amm x0 = getpz8(xq_sp+N0);
            const amm x1 = getpz8(xq_sp+N1);
            const amm x2 = getpz8(xq_sp+N2);
            const amm x3 = getpz8(xq_sp+N3);
            const amm x4 = getpz8(xq_sp+N4);
            const amm x5 = getpz8(xq_sp+N5);
            const amm x6 = getpz8(xq_sp+N6);
            const amm x7 = getpz8(xq_sp+N7);

            const amm  a04 =       addpz8(x0, x4);
            const amm  s04 =       subpz8(x0, x4);
            const amm  a26 =       addpz8(x2, x6);
            const amm js26 = jxpz8(subpz8(x2, x6));
            const amm  a15 =       addpz8(x1, x5);
            const amm  s15 =       subpz8(x1, x5);
            const amm  a37 =       addpz8(x3, x7);
            const amm js37 = jxpz8(subpz8(x3, x7));
            const amm    a04_p1_a26 =        addpz8(a04,  a26);
            const amm    s04_mj_s26 =        subpz8(s04, js26);
            const amm    a04_m1_a26 =        subpz8(a04,  a26);
            const amm    s04_pj_s26 =        addpz8(s04, js26);
            const amm    a15_p1_a37 =        addpz8(a15,  a37);
            const amm w8_s15_mj_s37 = w8xpz8(subpz8(s15, js37));
            const amm  j_a15_m1_a37 =  jxpz8(subpz8(a15,  a37));
            const amm v8_s15_pj_s37 = v8xpz8(addpz8(s15, js37));

            const amm w2p = mulpz8(w1p,w1p);
            const amm w3p = mulpz8(w1p,w2p);
            const amm w4p = mulpz8(w2p,w2p);
            const amm w5p = mulpz8(w2p,w3p);
            const amm w6p = mulpz8(w3p,w3p);
            const amm w7p = mulpz8(w3p,w4p);

            setpz8(yq_s8p+s*0,             addpz8(a04_p1_a26,    a15_p1_a37));
            setpz8(yq_s8p+s*1, mulpz8(w1p, addpz8(s04_mj_s26, w8_s15_mj_s37)));
            setpz8(yq_s8p+s*2, mulpz8(w2p, subpz8(a04_m1_a26,  j_a15_m1_a37)));
            setpz8(yq_s8p+s*3, mulpz8(w3p, subpz8(s04_pj_s26, v8_s15_pj_s37)));
            setpz8(yq_s8p+s*4, mulpz8(w4p, subpz8(a04_p1_a26,    a15_p1_a37)));
            setpz8(yq_s8p+s*5, mulpz8(w5p, subpz8(s04_mj_s26, w8_s15_mj_s37)));
            setpz8(yq_s8p+s*6, mulpz8(w6p, addpz8(a04_m1_a26,  j_a15_m1_a37)));
            setpz8(yq_s8p+s*7, mulpz8(w7p, addpz8(s04_pj_s26, v8_s15_pj_s37)));
        }
    }
};

template <int N> struct fwdcore<N,1>
{
    static constexpr int N0 = 0;
    static constexpr int N1 = N/8;
    static constexpr int N2 = N1*2;
    static constexpr int N3 = N1*3;
    static constexpr int N4 = N1*4;
    static constexpr int N5 = N1*5;
    static constexpr int N6 = N1*6;
    static constexpr int N7 = N1*7;
    static constexpr double theta = -2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_8p = y + 8*p;

            const double t1 = theta*p;
            const double t2 = t1 + theta;
            const ymm w1p = cmplx2(expj(t1), expj(t2));

            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);

            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));

            const ymm w2p = mulpz2(w1p,w1p);
            const ymm w3p = mulpz2(w1p,w2p);
            const ymm w4p = mulpz2(w2p,w2p);
            const ymm w5p = mulpz2(w2p,w3p);
            const ymm w6p = mulpz2(w3p,w3p);
            const ymm w7p = mulpz2(w3p,w4p);
#if 0
            setpz3<8>(y_8p+0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
#else
            const ymm aA =             addpz2(a04_p1_a26,    a15_p1_a37);
            const ymm bB = mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm cC = mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm dD = mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm eE = mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37));
            const ymm fF = mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm gG = mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm hH = mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37));

            const ymm ab = catlo(aA, bB);
            setpz2(y_8p+ 0, ab);
            const ymm cd = catlo(cC, dD);
            setpz2(y_8p+ 2, cd);
            const ymm ef = catlo(eE, fF);
            setpz2(y_8p+ 4, ef);
            const ymm gh = catlo(gG, hH);
            setpz2(y_8p+ 6, gh);
            const ymm AB = cathi(aA, bB);
            setpz2(y_8p+ 8, AB);
            const ymm CD = cathi(cC, dD);
            setpz2(y_8p+10, CD);
            const ymm EF = cathi(eE, fF);
            setpz2(y_8p+12, EF);
            const ymm GH = cathi(gG, hH);
            setpz2(y_8p+14, GH);
#endif
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo, int mode> struct fwdend;

//-----------------------------------------------------------------------------

template <int s, bool eo, int mode> struct fwdend<8,s,eo,mode>
{
    static constexpr int N = 8*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        complex_vector z = eo ? y : x;
        #pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector zq = z + q;
            const ymm x0 = scalepz2<N,mode>(getpz2(xq+s*0));
            const ymm x1 = scalepz2<N,mode>(getpz2(xq+s*1));
            const ymm x2 = scalepz2<N,mode>(getpz2(xq+s*2));
            const ymm x3 = scalepz2<N,mode>(getpz2(xq+s*3));
            const ymm x4 = scalepz2<N,mode>(getpz2(xq+s*4));
            const ymm x5 = scalepz2<N,mode>(getpz2(xq+s*5));
            const ymm x6 = scalepz2<N,mode>(getpz2(xq+s*6));
            const ymm x7 = scalepz2<N,mode>(getpz2(xq+s*7));
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(zq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(zq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(zq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(zq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(zq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(zq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <bool eo, int mode> struct fwdend<8,1,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp single
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm x0 = scalepz<8,mode>(getpz(x[0]));
            const xmm x1 = scalepz<8,mode>(getpz(x[1]));
            const xmm x2 = scalepz<8,mode>(getpz(x[2]));
            const xmm x3 = scalepz<8,mode>(getpz(x[3]));
            const xmm x4 = scalepz<8,mode>(getpz(x[4]));
            const xmm x5 = scalepz<8,mode>(getpz(x[5]));
            const xmm x6 = scalepz<8,mode>(getpz(x[6]));
            const xmm x7 = scalepz<8,mode>(getpz(x[7]));
            const xmm  a04 =      addpz(x0, x4);
            const xmm  s04 =      subpz(x0, x4);
            const xmm  a26 =      addpz(x2, x6);
            const xmm js26 = jxpz(subpz(x2, x6));
            const xmm  a15 =      addpz(x1, x5);
            const xmm  s15 =      subpz(x1, x5);
            const xmm  a37 =      addpz(x3, x7);
            const xmm js37 = jxpz(subpz(x3, x7));
            const xmm    a04_p1_a26 =       addpz(a04,  a26);
            const xmm    s04_mj_s26 =       subpz(s04, js26);
            const xmm    a04_m1_a26 =       subpz(a04,  a26);
            const xmm    s04_pj_s26 =       addpz(s04, js26);
            const xmm    a15_p1_a37 =       addpz(a15,  a37);
            const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
            const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
            const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
            setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[1], addpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[2], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[3], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[5], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[6], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[7], addpz(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Forward FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo, int mode> struct fwdfft
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        fwdcore<n,s>()(x, y);
        fwdfft<n/8,8*s,!eo,mode>()(y, x);
    }
};

template <int s, bool eo, int mode> struct fwdfft<8,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        fwdend<8,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct fwdfft<4,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::fwdend<4,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::fwdend<2,s,eo,mode>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse Butterfly Operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct invcore
{
    static constexpr int N  = n*s;
    static constexpr int N0 = 0;
    static constexpr int N1 = N/8;
    static constexpr int N2 = N1*2;
    static constexpr int N3 = N1*3;
    static constexpr int N4 = N1*4;
    static constexpr int N5 = N1*5;
    static constexpr int N6 = N1*6;
    static constexpr int N7 = N1*7;
    static constexpr int Ni = N1/8;
    static constexpr int h  = s/8;
    static constexpr double theta = 2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < Ni; i++) {
            const int p = i / h;
            const int q = i % h * 8;
            const int sp = s*p;
            const int s8p = 8*sp;
            complex_vector xq_sp  = x + q + sp;
            complex_vector yq_s8p = y + q + s8p;

            const amm w1p = duppz9(expj(theta*sp));

            const amm x0 = getpz8(xq_sp+N0);
            const amm x1 = getpz8(xq_sp+N1);
            const amm x2 = getpz8(xq_sp+N2);
            const amm x3 = getpz8(xq_sp+N3);
            const amm x4 = getpz8(xq_sp+N4);
            const amm x5 = getpz8(xq_sp+N5);
            const amm x6 = getpz8(xq_sp+N6);
            const amm x7 = getpz8(xq_sp+N7);

            const amm  a04 =       addpz8(x0, x4);
            const amm  s04 =       subpz8(x0, x4);
            const amm  a26 =       addpz8(x2, x6);
            const amm js26 = jxpz8(subpz8(x2, x6));
            const amm  a15 =       addpz8(x1, x5);
            const amm  s15 =       subpz8(x1, x5);
            const amm  a37 =       addpz8(x3, x7);
            const amm js37 = jxpz8(subpz8(x3, x7));
            const amm    a04_p1_a26 =        addpz8(a04,  a26);
            const amm    s04_pj_s26 =        addpz8(s04, js26);
            const amm    a04_m1_a26 =        subpz8(a04,  a26);
            const amm    s04_mj_s26 =        subpz8(s04, js26);
            const amm    a15_p1_a37 =        addpz8(a15,  a37);
            const amm v8_s15_pj_s37 = v8xpz8(addpz8(s15, js37));
            const amm  j_a15_m1_a37 =  jxpz8(subpz8(a15,  a37));
            const amm w8_s15_mj_s37 = w8xpz8(subpz8(s15, js37));

            const amm w2p = mulpz8(w1p,w1p);
            const amm w3p = mulpz8(w1p,w2p);
            const amm w4p = mulpz8(w2p,w2p);
            const amm w5p = mulpz8(w2p,w3p);
            const amm w6p = mulpz8(w3p,w3p);
            const amm w7p = mulpz8(w3p,w4p);

            setpz8(yq_s8p+s*0,             addpz8(a04_p1_a26,    a15_p1_a37));
            setpz8(yq_s8p+s*1, mulpz8(w1p, addpz8(s04_pj_s26, v8_s15_pj_s37)));
            setpz8(yq_s8p+s*2, mulpz8(w2p, addpz8(a04_m1_a26,  j_a15_m1_a37)));
            setpz8(yq_s8p+s*3, mulpz8(w3p, subpz8(s04_mj_s26, w8_s15_mj_s37)));
            setpz8(yq_s8p+s*4, mulpz8(w4p, subpz8(a04_p1_a26,    a15_p1_a37)));
            setpz8(yq_s8p+s*5, mulpz8(w5p, subpz8(s04_pj_s26, v8_s15_pj_s37)));
            setpz8(yq_s8p+s*6, mulpz8(w6p, subpz8(a04_m1_a26,  j_a15_m1_a37)));
            setpz8(yq_s8p+s*7, mulpz8(w7p, addpz8(s04_mj_s26, w8_s15_mj_s37)));
        }
    }
};

template <int N> struct invcore<N,1>
{
    static constexpr int N0 = 0;
    static constexpr int N1 = N/8;
    static constexpr int N2 = N1*2;
    static constexpr int N3 = N1*3;
    static constexpr int N4 = N1*4;
    static constexpr int N5 = N1*5;
    static constexpr int N6 = N1*6;
    static constexpr int N7 = N1*7;
    static constexpr double theta = 2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_8p = y + 8*p;

            const double t1 = theta*p;
            const double t2 = t1 + theta;
            const ymm w1p = cmplx2(expj(t1), expj(t2));

            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);

            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));

            const ymm w2p = mulpz2(w1p,w1p);
            const ymm w3p = mulpz2(w1p,w2p);
            const ymm w4p = mulpz2(w2p,w2p);
            const ymm w5p = mulpz2(w2p,w3p);
            const ymm w6p = mulpz2(w3p,w3p);
            const ymm w7p = mulpz2(w3p,w4p);
#if 0
            setpz3<8>(y_8p+0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
#else
            const ymm aA =             addpz2(a04_p1_a26,    a15_p1_a37);
            const ymm bB = mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm cC = mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm dD = mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm eE = mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37));
            const ymm fF = mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm gG = mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm hH = mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37));

            const ymm ab = catlo(aA, bB);
            setpz2(y_8p+ 0, ab);
            const ymm cd = catlo(cC, dD);
            setpz2(y_8p+ 2, cd);
            const ymm ef = catlo(eE, fF);
            setpz2(y_8p+ 4, ef);
            const ymm gh = catlo(gG, hH);
            setpz2(y_8p+ 6, gh);
            const ymm AB = cathi(aA, bB);
            setpz2(y_8p+ 8, AB);
            const ymm CD = cathi(cC, dD);
            setpz2(y_8p+10, CD);
            const ymm EF = cathi(eE, fF);
            setpz2(y_8p+12, EF);
            const ymm GH = cathi(gG, hH);
            setpz2(y_8p+14, GH);
#endif
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo, int mode> struct invend;

//-----------------------------------------------------------------------------

template <int s, bool eo, int mode> struct invend<8,s,eo,mode>
{
    static constexpr int N = 8*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        complex_vector z = eo ? y : x;
        #pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector zq = z + q;
            const ymm x0 = scalepz2<N,mode>(getpz2(xq+s*0));
            const ymm x1 = scalepz2<N,mode>(getpz2(xq+s*1));
            const ymm x2 = scalepz2<N,mode>(getpz2(xq+s*2));
            const ymm x3 = scalepz2<N,mode>(getpz2(xq+s*3));
            const ymm x4 = scalepz2<N,mode>(getpz2(xq+s*4));
            const ymm x5 = scalepz2<N,mode>(getpz2(xq+s*5));
            const ymm x6 = scalepz2<N,mode>(getpz2(xq+s*6));
            const ymm x7 = scalepz2<N,mode>(getpz2(xq+s*7));
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(zq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(zq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(zq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(zq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(zq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(zq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <bool eo, int mode> struct invend<8,1,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp single
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm x0 = scalepz<8,mode>(getpz(x[0]));
            const xmm x1 = scalepz<8,mode>(getpz(x[1]));
            const xmm x2 = scalepz<8,mode>(getpz(x[2]));
            const xmm x3 = scalepz<8,mode>(getpz(x[3]));
            const xmm x4 = scalepz<8,mode>(getpz(x[4]));
            const xmm x5 = scalepz<8,mode>(getpz(x[5]));
            const xmm x6 = scalepz<8,mode>(getpz(x[6]));
            const xmm x7 = scalepz<8,mode>(getpz(x[7]));
            const xmm  a04 =      addpz(x0, x4);
            const xmm  s04 =      subpz(x0, x4);
            const xmm  a26 =      addpz(x2, x6);
            const xmm js26 = jxpz(subpz(x2, x6));
            const xmm  a15 =      addpz(x1, x5);
            const xmm  s15 =      subpz(x1, x5);
            const xmm  a37 =      addpz(x3, x7);
            const xmm js37 = jxpz(subpz(x3, x7));
            const xmm    a04_p1_a26 =       addpz(a04,  a26);
            const xmm    s04_pj_s26 =       addpz(s04, js26);
            const xmm    a04_m1_a26 =       subpz(a04,  a26);
            const xmm    s04_mj_s26 =       subpz(s04, js26);
            const xmm    a15_p1_a37 =       addpz(a15,  a37);
            const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
            const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
            const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
            setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[1], addpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[2], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[3], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[5], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[6], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[7], addpz(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo, int mode> struct invfft
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        invcore<n,s>()(x, y);
        invfft<n/8,8*s,!eo,mode>()(y, x);
    }
};

template <int s, bool eo, int mode> struct invfft<8,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        invend<8,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct invfft<4,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::invend<4,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::invend<2,s,eo,mode>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Power of 2 FFT Routine
///////////////////////////////////////////////////////////////////////////////

void fwd(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_length;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

void fwd0(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_1;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

void fwdu(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_unitary;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void fwdn(const int log_N, complex_vector x, complex_vector y) noexcept
{
    fwd(log_N, x, y);
}

///////////////////////////////////////////////////////////////////////////////

void inv(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_1;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void inv0(const int log_N, complex_vector x, complex_vector y) noexcept
{
    inv(log_N, x, y);
}

void invu(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_unitary;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

void invn(const int log_N, complex_vector x, complex_vector y) noexcept
{
    static const int mode = scale_length;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdif8exp_h
