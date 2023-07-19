import math, complex
import benchy, strformat, strutils, sequtils
import audiofile/[vorbis, wavfile]

import otfft-simple/otfft

proc padPowerOfTwo*(arr: seq[float]): seq[float] =
    assert arr.len > 0
    result = arr
    for i in arr.len..<arr.len.nextPowerOfTwo:
        result.add(arr[0])
    assert result.len.isPowerOfTwo

var audio = loadVorbis("data/sample.ogg").toFloat.padPowerOfTwo
var caudio = audio.mapIt(complex(it))
echo "Sample len: ", caudio.len

let xaddr = caudio[0].addr
var offt = otfft_fft_new(caudio.len.cint)

timeIt "otfft":
    offt.otfft_fft_fwd(xaddr)
