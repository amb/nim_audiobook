import arraymancer, sugar, sequtils, math, fft, random

# {.experimental: "parallel".}
# import std/threadpool

when isMainModule:
    import plottings, audiofile/[vorbis, wavfile], benchy, strformat, utils

proc hann*[T](sz: int): Tensor[T] = sin(arange(sz.T)*PI/T(sz-1)).square
proc hamming*[T](sz: int): Tensor[T] = 0.54 -. 0.46*cos(arange(sz.T)*2.0*PI/T(sz-1)) 

proc freqToMel*[T](freq: T): T = 2595.0 * log10(1.0 + freq / 700.0)
proc melToFreq*[T](mel: T): T = 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
proc melToFreq*[T](mel: Tensor[T]): Tensor[T] = 
    return 700.0 * ((mel / 2595.0).map(x => pow(10.0, x)) -. 1.0)

proc dbToAmplitude*[T](data: Tensor[T]): Tensor[T] = data.map(x => pow(10.0, x/20.0))
proc amplitudeToDb*[T](data: Tensor[T]): Tensor[T] = data.map(x => log10(x) * 20.0) 

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

type
    FFT_DS* = object
        A*: Tensor[Complex[float]]
        B*: Tensor[Complex[float]]

    STFT_DS_ref* = ref object
        fft_size*: int
        hop_size*: int
        data_size*: int
        total_segments*: int
        windowing_function*: Tensor[float]
        spectrum*: Tensor[Complex[float]]
        wave*: Tensor[float]
        fft_ds*: FFT_DS


proc newStft*(data_len: int, fft_size: int): STFT_DS_ref =
    assert fft_size.isPowerOfTwo
    var ds = STFT_DS_ref()
    ds.fft_size = fft_size
    ds.hop_size = fft_size div 2
    ds.data_size = data_len
    ds.total_segments = (data_len - fft_size) div ds.hop_size
    ds.windowing_function = hann[float](fft_size)

    ds.fft_ds.A = zeros[Complex[float]]([fft_size])
    ds.fft_ds.B = zeros[Complex[float]]([fft_size])
    ds.spectrum = zeros[Complex[float]]([ds.total_segments, fft_size])
    ds.wave = zeros[float]([data_len])
    return ds

proc newStft*(data: Tensor[float], fft_size: int): STFT_DS_ref =
    var ds = newStft(data.shape[0], fft_size)
    return ds

proc forward*(ds: STFT_DS_ref, data: Tensor[float]) =
    for i in 0..<ds.total_segments:
        let wl = ds.hop_size * i
        for j in 0..<ds.fft_size:
            ds.fft_ds.A[j] = complex(data[wl+j] * ds.windowing_function[j])
        fft0(ds.fft_size, 0, false, ds.fft_ds.A, ds.fft_ds.B)
        for j in 0..<ds.fft_size:
            ds.spectrum[i, j] = ds.fft_ds.A[j]

proc inverse*(ds: STFT_DS_ref, data: Tensor[Complex[float]]) =
    let rsize = complex(1.0/float(ds.fft_size))
    for i in 0..<ds.wave.shape[0]:
        ds.wave[i] = 0.0
    for i in 0..<ds.total_segments:
        let wl = ds.hop_size * i
        ds.fft_ds.A = data[i, _].reshape(ds.fft_size)
        ds.fft_ds.A.apply_inline((x*rsize).conjugate)
        fft0(ds.fft_size, 0, false, ds.fft_ds.A, ds.fft_ds.B)
        ds.wave[wl..<(wl+ds.fft_size)] = ds.wave[wl..<(wl+ds.fft_size)] + 
            (ds.fft_ds.A.map_inline(x.conjugate.re))

proc griffin_lim*(mag_spec: Tensor[float], iterations: int): Tensor[float] =
    ## Discover the phase information of a magnitude only spectrogram
    ## and output the original time series
    let fft_size = mag_spec.shape[1]
    let hop_size = fft_size div 2
    let samples_size = mag_spec.shape[0] * hop_size + fft_size
    var ds = newStft(samples_size, fft_size)

    let cone = complex(0.0, 1.0)
    let mag_comp = mag_spec.map_inline(x.complex)
    ds.wave = randomTensor(samples_size, 1.0)
    for n in countup(1, iterations):
        ds.forward(ds.wave)
        for i in 0..<ds.spectrum.shape[0]:
            for j in 0..<ds.spectrum.shape[1]:
                ds.spectrum[i, j] = exp(cone * phase(ds.spectrum[i, j]))
        ds.inverse(mag_comp *. ds.spectrum)

        # Iterate until satisfied, this should converge
        # let diff = sqrt(sum((guess - prev_guess)^.2.0)/float(samples_size))
        # echo fmt"Iteration: {n}/{iterations}, RMSE: {diff:.4f}"

    return ds.wave


when isMainModule:
    echo "Running spectrum.nim"

    let wave = loadWav("data/sample.wav")
    let fft_size = 512
    let wave_tensor = wave.toFloat.padPowerOfTwo.toTensor
    var stft_ds = newStft(wave_tensor.shape[0], fft_size)
    stft_ds.forward(wave_tensor)
    let mag_spec = stft_ds.spectrum.abs

    # let inverse = mag_spec.inverse0(fft_size)

    # var ds = newStft(wave_tensor.shape[0], fft_size)
    # timeIt "forward":
    #     mag_spec.forward0(ds)

    # timeIt "griffin_lim":
    #     discard griffin_lim(mag_spec, 32)
