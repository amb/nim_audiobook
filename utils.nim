import arraymancer, pixie
import sugar
import std/[strutils, strformat, sequtils, math, base64]
import fft

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

proc plot2DArray*(arr: Tensor[float]): Image =
    var array_out = newImage(arr.shape[0], arr.shape[1])
    for x in 0..<arr.shape[0]:
        for y in 0..<arr.shape[1]:
            let c = arr[x, y]
            array_out[x, arr.shape[1]-y] = color(c, c, c)
    return array_out

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

proc hann*(size: int): Tensor[float] =
    let rr = toSeq(0..<size).mapIt(it.float).toTensor()
    return cos(rr*2.0*PI/float(size))

proc freq2mel*(freq: float): float = 
    var value: float = 1.0 + freq / 700.0
    if value == 0.0:
        return 1.0/2048.0
    else:
        return 1127.0 * ln(value)

proc mel2freq*(mel: float): float = 
    return 700.0 * (exp(mel / 1127.0) - 1.0)

proc melBanks*[T: SomeFloat](rank, n, rate: int, fl, fh: T): Tensor[T] =
    var temp = newTensor[T](1, rank + 2)
    result = newTensor[T](rank, n)
    for i in 0 ..< rank + 2:
        temp[0, i] = T(n / rate) * mel2freq(freq2mel(fl) + T(i) * (freq2mel(fh) -
            freq2mel(fl)) / T(rank+1))
    for m in 1 .. rank:
        for k in 0 .. n:
            if temp[0, m - 1] <= T(k) and T(k) <= temp[0, m]:
                result[m - 1, k] = (T(k) - temp[0, m - 1]) / (temp[0, m] - temp[0, m - 1])
            elif temp[0, m] <= T(k) and T(k) <= temp[0, m + 1]:
                result[m - 1, k] = (temp[0, m + 1] - T(k)) / (temp[0, m + 1] - temp[0, m])

proc stft*(data: Tensor[float], freq: float, fft_size: int): Tensor[float] =
    let data_len = data.shape[0]
    let hop_size = fft_size div 2
    let fft_half = fft_size div 2
    let total_segments = (data_len - fft_size) div hop_size

    # for each segment
    result = zeros[float]([total_segments, fft_half])
    let window = hann(fft_size)
    var windowed = zeros[Complex[float]]([fft_size])
    for i in 0..<total_segments:
        let wl = hop_size * i
        for (j, val) in enumerate(data[wl..<(wl+fft_size)] *. window):
            windowed[j] = complex(val)
        fft(windowed)
        result[i, _] = (abs(windowed)[0..<fft_half]).reshape([1, fft_half])

if isMainModule:
    var cten = zeros[Complex[float]]([5])
    echo cten.shape, " ", cten
