# nim --cc:vcc cpp -d:release -r bench_pocket.nim
# gcc ony my system doesn't work because somehow it links to a wrong stdlibc++

# requires cpp compilation :(

import math, complex
import benchy, strformat, strutils, sequtils
import audiofile/[vorbis, wavfile]

import pocketfft/pocketfft

type
    FFTArray = seq[Complex[float]]

proc fft_array_len*(v: FFTArray): int =
    return v.len

proc fft_empty_array*(v: FFTArray): FFTArray =
    result = newSeq[Complex[float]](v.len)

proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo


echo "Running bench"

var audio = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo
var caudio = audio.mapIt(complex(it))
echo "Sample len: ", fft_array_len(caudio)

var y = fft_empty_array(caudio)
let caudio_len = fft_array_len(caudio)

# Benchmark pocketFFT
var dOut = newSeq[Complex[float64]](caudio.len)
let dInDesc = DataDesc[float64].init(audio[0].unsafeAddr, [audio.len])
var dOutDesc = DataDesc[Complex[float64]].init(dOut[0].addr, [dOut.len])
var fftd = FFTDesc[float64].init(axes=[0], forward=true)

echo "start."
timeIt "pocketfft":
    fftd.apply(dOutDesc, dInDesc)
echo "end."
