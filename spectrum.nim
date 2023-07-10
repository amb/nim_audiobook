import sugar, sequtils, math, fft, random, std/complex

# {.experimental: "parallel".}
# import std/threadpool

when isMainModule:
    import audiofile/[vorbis, wavfile], benchy, strformat

type
    Array2D[T] = object
        data: seq[T]
        rows: int
        cols: int

proc newArray2D[T](rows, cols: int): Array2D[T] =
    result.data = newSeq[T](rows * cols)
    result.rows = rows
    result.cols = cols

proc `[]=`[T](arr: var Array2D[T], row, col: int, value: T) =
    arr.data[row * arr.cols + col] = value

proc `[]`[T](arr: Array2D[T], row, col: int): T =
    result = arr.data[row * arr.cols + col]

proc hann*[T](sz: int): seq[T] = 
    # sin(arange(sz.T)*PI/T(sz-1)).square
    collect(newSeq):
        for i in 0..<sz:
            let s0 = sin(i.T * PI/T(sz-1))
            s0 * s0

# proc hamming*[T](sz: int): seq[T] = 0.54 -. 0.46*cos(arange(sz.T)*2.0*PI/T(sz-1)) 

proc freqToMel*[T](freq: T): T = 2595.0 * log10(1.0 + freq / 700.0)
proc melToFreq*[T](mel: T): T = 700.0 * (pow(10.0, mel / 2595.0) - 1.0)

proc melToFreq*[T](mel: seq[T]): seq[T] = 
    return mel.map(x => (pow(10.0, x / 2595.0) - 1.0) * 700.0)

proc dbToAmplitude*[T](data: seq[T]): seq[T] = data.map(x => pow(10.0, x/20.0))
proc amplitudeToDb*[T](data: seq[T]): seq[T] = data.map(x => log10(x) * 20.0) 

proc melPts*[T](r: int, fl, fh: T): seq[T] = 
    #linspace(fl.freqToMel, fh.freqToMel, r+2).melToFreq
    collect(newSeq):
        let d = fh.freqToMel - fl.freqToMel
        for i in 0..<r+2:
            let v: T = fl.freqToMel + (d * i.T / (r+2).T)
            v.melToFreq

# proc melBanks*(rank, fft_size, srate: int, fh: float): seq[float] =
#     result = zeros[float](rank, fft_size div 2)
#     let bins = floor(fft_size.float * melPts(rank, 0.0, fh) / srate.float)
#     for m in 1..rank:
#         if bins[m-1] != bins[m] and bins[m] != bins[m+1]:
#             for k in int(bins[m-1])..<int(bins[m]):
#                 result[m-1, k] = (k.float - bins[m-1]) / (bins[m] - bins[m-1])
#             for k in int(bins[m])..<int(bins[m+1]):
#                 result[m-1, k] = (bins[m+1] - k.float) / (bins[m+1] - bins[m])

type
    # TODO: perhaps separate ISTFT and STFT data structs
    STFT_DS_ref* = ref object
        fft_size*: int
        hop_size*: int
        data_size*: int
        total_segments*: int
        windowing_function*: seq[float]
        buffer_A*: seq[Complex[float]]
        buffer_B*: seq[Complex[float]]
        stft*: Array2D[Complex[float]]
        wave*: seq[float]


proc newStft*(data_len: int, fft_size: int): STFT_DS_ref =
    assert fft_size.isPowerOfTwo
    var ds = STFT_DS_ref()
    ds.fft_size = fft_size
    ds.hop_size = fft_size div 2
    ds.data_size = data_len
    ds.total_segments = (data_len - fft_size) div ds.hop_size
    ds.windowing_function = hann[float](fft_size)
    ds.buffer_A = newSeq[Complex[float]](fft_size)
    ds.buffer_B = newSeq[Complex[float]](fft_size)
    ds.stft = newArray2D[Complex[float]](ds.total_segments, fft_size)
    # TODO: shouldn't this be complex???
    ds.wave = newSeq[float](data_len)

    return ds

proc stft0*(data: seq[float], ds: STFT_DS_ref) =
    for i in 0..<ds.total_segments:
        let wl = ds.hop_size * i
        for j in 0..<ds.fft_size:
            ds.buffer_A[j] = complex(data[wl+j] * ds.windowing_function[j])
        fft0(ds.fft_size, 0, false, ds.buffer_A, ds.buffer_B)
        for j in 0..<ds.fft_size:
            ds.stft[i, j] = ds.buffer_A[j]

proc stft*(data: seq[float], fft_size: int): Array2D[Complex[float]] =
    var ds = newStft(data.len, fft_size)
    stft0(data, ds)
    return ds.stft

proc istft0*(data: seq[Complex[float]], ds: STFT_DS_ref) =
    let rsize = complex(1.0/float(ds.fft_size))
    for i in 0..<ds.wave.len:
        ds.wave[i] = 0.0
    for i in 0..<ds.total_segments:
        let wl = ds.hop_size * i
        # ds.buffer_A = data[i, _].reshape(ds.fft_size)
        # ds.buffer_A.apply_inline((x*rsize).conjugate)
        for j in 0..<ds.fft_size:
            ds.buffer_A[j] = (data[i, j] * rsize).conjugate
        fft0(ds.fft_size, 0, false, ds.buffer_A, ds.buffer_B)
        # ds.wave[wl..<(wl+ds.fft_size)] = ds.wave[wl..<(wl+ds.fft_size)] + 
        #     (ds.buffer_A.map_inline(x.conjugate.re))
        for j in 0..<ds.fft_size:
            ds.wave[wl+j] = ds.wave[wl+j] + ds.buffer_A[j].conjugate.re

proc griffin_lim*(mag_spec: Array2D[float], iterations: int): seq[float] =
    ## Discover the phase information of a magnitude only spectrogram
    ## and output the original time series
    let fft_size = mag_spec.cols
    let hop_size = fft_size div 2
    let samples_size = mag_spec.rows * hop_size + fft_size
    var ds = newStft(samples_size, fft_size)

    let cone = complex(0.0, 1.0)
    ds.wave = newSeq[float](samples_size)
    for i in 0..<ds.wave.len:
        ds.wave[i] = rand(1.0)
    # var tmag = newSeq[float](mag_spec.len)
    var tmag = newArray2D[float](mag_spec.rows, mag_spec.cols)
    for n in countup(1, iterations):
        stft0(ds.wave, ds)
        for i in 0..<ds.stft.rows:
            for j in 0..<ds.stft.cols:
                ds.stft[i, j] = exp(cone * phase(ds.stft[i, j]))
        
        for i in 0..<ds.stft.rows:
            for j in 0..<ds.stft.cols:
                tmag[i, j] = mag_spec[i, j] * ds.stft[i, j].re

        # istft0(mag_spec *. ds.stft, ds)
        istft0(tmag, ds)

        # Iterate until satisfied, this should converge
        # let diff = sqrt(sum((guess - prev_guess)^.2.0)/float(samples_size))
        # echo fmt"Iteration: {n}/{iterations}, RMSE: {diff:.4f}"

    return ds.wave


when isMainModule:
    echo "Running spectrum.nim"

    let wave = loadWav("data/sample.wav")
    let fft_size = 512
    let wave_seq = wave.toFloat.toseq()
    let mag_spec = wave_seq.stft(fft_size).abs

    # var ds = newStft(wave_seq.shape[0], fft_size)
    # timeIt "stft":
    #     mag_spec.stft0(ds)

    timeIt "griffin_lim":
        discard griffin_lim(mag_spec, 32)
