import arraymancer, pixie, sequtils, std/[strutils, strformat, base64], fft

proc Image*(img: Image): string =
    return fmt"""<img src="data:image/png;base64, {img.encodeImage(PngFormat).encode}" />"""

proc plot1D*(img: Image, arr: openArray[float]): Image =
    var path: seq[string] = @[]

    let amax = arr[arr.maxIndex]
    let amin = arr[arr.minIndex]

    let mp = float(img.height)/(amax - amin)

    # For long arrays, draw <wmul> overlapping
    let wmul = 8
    if arr.len < img.width * wmul or false:
        for i in 0..<arr.len:
            let arr_v = arr[i]
            path.add(fmt"{int(float(i)*float(img.width)/float(arr.len))} {int((arr_v-amin) * mp)} ")
    else:
        for i in 0..<img.width * wmul:
            let arr_v = arr[int(float(i)*float(arr.len)/float(img.width)) div wmul]
            path.add(fmt"{i div wmul} {int((arr_v-amin) * mp)} ")

    img.strokePath(fmt"M {path[0]} L " & path.join(""), rgba(0, 0, 128, 255), strokeWidth = 1)
    return img

proc basicSine*(srate: int): Tensor[system.float] =
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

proc hann*(size: int): Tensor[float32] =
    let rr = toSeq(0..<size).mapIt(it.float32).toTensor()
    return cos(rr*2.0*PI/float(size))

proc stft*(data: Tensor[float32], freq: float): Tensor[float32] =
    # data: array containing the signal to be processed
    # freq: sampling frequency of the data

    let data_len = data.shape[0]
    let fft_size = 2048
    let hop_size = 1024

    # the last segment can overlap the end of the data array by no more than one window size
    let pad_end_size = fft_size
    let total_segments = int(ceil(data_len.float / hop_size.float))
    let t_max = data_len.float / freq.float

    # our half cosine window
    let window = hann(fft_size)

    # the zeros which will be used to double each segment size
    let inner_pad = zeros[float32]([fft_size])

    # the data to process
    let padding = zeros[float32]([pad_end_size])
    let process = concat([data, padding], 0)

    # space to hold the result
    result = zeros[float32]([total_segments, fft_size])
    echo result.shape

    # for each segment
    for i in 0..<total_segments:
        # figure out the current segment offset
        let current_hop = hop_size * i
        # get the current segment
        let segment = process[current_hop..(current_hop+fft_size)]
        # multiply by the half cosine function
        let windowed = segment * window
        # add 0s to double the length of the data
        let padded = concat([windowed, inner_pad], 0)
        # # take the Fourier Transform and scale by the number of samples
        # var spectrum: seq[Complex[float]] = padded.toSeq.mapIt(complex(float(it)))
        # fft(spectrum)
        # spectrum /= fft_size
        let spectrum = zeros[Complex[float32]]([padded.shape[0]])

        let spec_conjugate = spectrum[_.._].map(conjugate)
        # find the autopower spectrum
        let autopower = abs(spectrum * spec_conjugate)
        # append to the results array
        result[i, _] = autopower[_..fft_size]

    # scale to db and clip
    result = float32(20.0) * log10(result)
    result = clamp(result, -40.0, 200.0)

if isMainModule:
    var cten = zeros[Complex[float32]]([5])
    echo cten.shape, " ", cten
