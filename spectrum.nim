import arraymancer, sugar, sequtils, math, fft

# {.experimental: "parallel".}
# import std/threadpool

when isMainModule:
    import plottings, audiofile/[vorbis, wavfile], benchy

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

# TODO: Replace pow with ^.

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

type
    STFT_DS_ref* = ref STFT_DS
    STFT_DS* = object
        fft_size*: int
        hop_size*: int
        total_segments*: int
        windowing_function*: Tensor[float]
        buffer_A*: Tensor[Complex[float]]
        buffer_B*: Tensor[Complex[float]]
        output*: Tensor[float]


proc newStft*(data: Tensor[float], fft_size: int): STFT_DS_ref =
    assert fft_size.isPowerOfTwo
    var ds = STFT_DS_ref()
    ds.fft_size = fft_size
    ds.hop_size = fft_size div 2
    ds.total_segments = (data.shape[0] - fft_size) div ds.hop_size
    ds.windowing_function = hamming[float](fft_size)
    ds.buffer_A = zeros[Complex[float]]([fft_size])
    ds.buffer_B = zeros[Complex[float]]([fft_size])
    ds.output = zeros[float]([ds.total_segments, fft_size div 2])
    return ds

proc stft0*(data: Tensor[float], ds: STFT_DS_ref) =
    let fft_half = ds.fft_size div 2
    for i in 0..<ds.total_segments:
        let wl = ds.hop_size * i
        # TODO: arraymancer version is slightly slower
        # ds.buffer_A = (data[wl..<(wl+ds.fft_size)] *. ds.windowing_function).map_inline(complex(x))
        for j in 0..<ds.fft_size:
            ds.buffer_A[j] = complex(data[wl+ds.fft_size] * ds.windowing_function[j])
        fft0(ds.fft_size, 0, false, ds.buffer_A, ds.buffer_B)
        for j in 0..<fft_half:
            ds.output[i, j] = abs(ds.buffer_A[j])
        # ds.output[i, _] = (abs(ds.buffer_A)[0..<fft_half]).reshape([1, fft_half])

proc stft*(data: Tensor[float], fft_size: int): Tensor[float] =
    var ds = newStft(data, fft_size)
    stft0(data, ds)
    return ds.output

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
        windowed[0..<fft_half] = data[i, _].map_inline(complex(x)).reshape(fft_half)

        # Copy FFT reverse side
        for j in fft_half..<fft_size:
            windowed[j] = windowed[fft_size-j]

        # ifft
        windowed.apply(x => (x*fn).conjugate)
        fft0(fft_size, 0, false, windowed, y)
        windowed.apply(x => x.conjugate)

        let wl = hop_size * i
        temp[wl..<(wl+fft_size)] = temp[wl..<(wl+fft_size)] + (abs(windowed) *. invert_window)

    return temp


when isMainModule:
    echo "Running spectrum.nim"

    let wave = loadWav("data/sample.wav")
    let fft_size = 512
    let mag_spectrum = wave.toFloat.toTensor()

    var ds = mag_spectrum.newStft(fft_size)
    timeIt "stft":
        mag_spectrum.stft0(ds)

    # let fft_size = 2048
    # let mag_spectrum = wave.toFloat.toTensor().stft(fft_size)

    # let pow_spectrum = mag_spectrum.square / fft_size.float
    # let mel_spectrum = mag_spectrum * melBanks(256, fft_size, wave.freq, wave.freq.float/2.0).transpose
    # html Image(mel_spectrum.amplitudeToDb.clip(-40.0, 200.0).plot2DArray)

    # Audio(wave.toFloat, wave.freq, false)

    # let inverse_stft = istft(mag_spectrum)
    # html Image(newImage(500,100).plot1D(inverse_stft.toSeq))
    # html Image(newImage(500,100).plot1D(wave.toFloat))
    # html Image(mag_spectrum.amplitudeToDb.clamp(-40.0, 200.0).plot2DArray)
