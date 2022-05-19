import arraymancer, pixie, sugar, sequtils, math, base64, fft

const ok_purple = polarOklab(color(1.0, 0.0, 1.0, 1.0))
const ok_yellow = polarOklab(color(1.0, 1.0, 0.0, 1.0))

proc basicSine*(srate: int): Tensor[float] =
    let srate = float(srate)
    let dur = 1000
    let rr_dur = dur * 2
    let rr = toSeq(0..<rr_dur).mapIt(it.float).toTensor()
    var sample = sin(rr*(2.0*PI*440.0/srate))
    sample += sin(rr*(2.0*PI*440.0*2.0/srate)) * 0.5
    sample += sin(rr*(2.0*PI*440.0*4.0/srate))
    sample += sin(rr*(2.0*PI*440.0*8.0/srate)) * 0.5
    # With 11025 sample rate, this goes over Nyquist frequency
    # sample += sin(rr*(2.0*PI*440.0*16.0/srate))
    return sample *. sin(rr*(0.5*PI/float(dur)))

proc hann*[T](sz: int): Tensor[T] = sin(arange(sz.T)*PI/T(sz-1)).square
proc hamming*[T](sz: int): Tensor[T] = 0.54 -. 0.46*cos(arange(sz.T)*2.0*PI/T(sz-1)) 

proc freqToMel*[T](freq: T): T = 2595.0 * log10(1.0 + freq / 700.0)
proc melToFreq*[T](mel: T): T = 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
proc melToFreq*[T](mel: Tensor[T]): Tensor[T] = 
    return 700.0 * ((mel / 2595.0).map(x => pow(10.0, x)) -. 1.0)

proc dbToAmplitude*[T](data: Tensor[T]): Tensor[T] = data.map(x => pow(10.0, x/20.0))
proc amplitudeToDb*[T](data: Tensor[T]): Tensor[T] = data.map(x => log10(x) * 20.0) 

# Theres Arraymancer "clamp"
# proc clip*[T](data: Tensor[T], a, b: T): Tensor[T] = data.map(x => clamp(x, a, b))

# Same as Complex.phase really
# proc angle*[T: SomeFloat](z: Complex[T], degrees: bool = false): T = arctan2(z.im, z.re)

proc melPts*[T](r: int, fl, fh: T): Tensor[T] = linspace(fl.freqToMel, fh.freqToMel, r+2).melToFreq
proc melBanks*(rank, fft_size, srate: int, fh: float): Tensor[float] =
    result = zeros[float](rank, fft_size div 2)
    let bins = floor(fft_size.float * melPts(rank, 0.0, fh) / srate.float)
    for m in 1..rank:
        if bins[m-1] != bins[m] and bins[m] != bins[m+1]:
            for k in int(bins[m-1])..<int(bins[m]):
                result[m-1, k] = (k.float - bins[m-1]) / (bins[m] - bins[m-1])
            for k in int(bins[m])..<int(bins[m+1]):
                result[m-1, k] = (bins[m+1] - k.float) / (bins[m+1] - bins[m])
        else:
            # TODO: bilinear interpolation to fix the gaps
            result[m-1, int(bins[m])] = 1.0

proc stft*(data: Tensor[float], fft_size: int): Tensor[float] =
    # TODO: pad by half window size on both ends
    assert fft_size.isPowerOfTwo
    let hop_size = fft_size div 2
    let fft_half = fft_size div 2
    let total_segments = (data.shape[0] - fft_size) div hop_size

    result = zeros[float]([total_segments, fft_half])
    let window = hamming[float](fft_size)
    var windowed = zeros[Complex[float]]([fft_size])
    var y = fft_empty_array(windowed)
    for i in 0..<total_segments:
        let wl = hop_size * i
        for (j, val) in enumerate(data[wl..<(wl+fft_size)] *. window):
            windowed[j] = complex(val)
        fft0_avx(fft_size, 1, false, windowed, y)
        result[i, _] = (abs(windowed)[0..<fft_half]).reshape([1, fft_half])

proc istft*(data: Tensor[float]): Tensor[float] =
    assert data.shape[1].isPowerOfTwo
    let fft_size = data.shape[1] * 2
    let fft_half = fft_size div 2
    let hop_size = fft_half
    let data_len = data.shape[0] * hop_size + fft_size - hop_size + 1
    # let total_segments = (data_len - fft_size) div hop_size
    let total_segments = data.shape[0]

    var temp = zeros[float](data_len)

    # inverse hamming (can do because it's always > 0.0)
    let invert_window = hamming[float](fft_size).map(x => 1.0/x)
    var windowed = zeros[Complex[float]](fft_size)

    let fn = complex(1.0/float(fft_size))
    var y = fft_empty_array_complex(fft_size)
    for i in 0..<total_segments:
        windowed[0..<fft_half] = data[i, _].map(x => complex(x)).reshape(fft_half)

        # Copy FFT reverse side
        for j in fft_half..<fft_size:
            windowed[j] = windowed[fft_size-j]

        # ifft
        windowed.apply(x => (x*fn).conjugate)
        fft0_avx(fft_size, 1, false, windowed, y)
        windowed.apply(x => x.conjugate)

        let wl = hop_size * i
        temp[wl..<(wl+fft_size)] = temp[wl..<(wl+fft_size)] + (abs(windowed) *. invert_window)

    return temp