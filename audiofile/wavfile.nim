import streams, std/base64, std/strformat

type
    WavFile* = object
        data*: seq[uint8]
        size*: int
        freq*: int
        bits*: int
        channels*: int

    wavHeaderObj* = object
        ChunkID*: array[4, char]
        ChunkSize: uint32
        Format: array[4, char]
        FmtChunkID: array[4, char]
        FmtChunkSize: uint32
        AudioFormat: uint16
        NumChannels: uint16
        SampleRate: uint32
        ByteRate: uint32
        BlockAlign: uint16
        BitsPerSample: uint16

    wavChunkObj* = object
        DataChunkID: string
        DataChunkSize*: uint32
        Data: string
        # Data: seq[uint8]


proc readDataChunk(f: FileStream): wavChunkObj =
    var chunk = wavChunkObj()
    chunk.DataChunkID = f.readStr(4)
    chunk.DataChunkSize = f.readUint32()
    chunk.Data = f.readStr(chunk.DataChunkSize.int)

    # TODO: this throws undebuggable error
    # var tbuf = newSeq[uint8](chunk.DataChunkSize)
    # discard f.readData(tbuf.addr, chunk.DataChunkSize.int)
    # chunk.Data = tbuf
    return chunk

proc wavWrite(f: Stream, wav: WavFile) =
    let datalen: uint32 = wav.data.len.uint32

    var header = wavHeaderObj()
    header.ChunkID = ['R', 'I', 'F', 'F']
    header.ChunkSize = datalen + sizeof(wavHeaderObj).uint32
    header.Format = ['W', 'A', 'V', 'E']
    header.FmtChunkID = ['f', 'm', 't', ' ']
    header.FmtChunkSize = 16.uint32
    header.AudioFormat = 1.uint16
    header.NumChannels = wav.channels.uint16
    header.SampleRate = wav.freq.uint32
    # TODO: calculate
    header.ByteRate = 44100.uint32
    header.BlockAlign = 2.uint16
    header.BitsPerSample = wav.bits.uint16

    f.writeData(addr(header), sizeof(wavHeaderObj))
    f.write(cast[array[4, char]](['d', 'a', 't', 'a']))
    f.write(datalen)

    f.writeData(wav.data[0].unsafeAddr, datalen.int)

proc wavSeq(iarr: seq[float], freq: int): WavFile =
    var clip: WavFile
    clip.size = iarr.len * 2
    clip.freq = freq
    clip.bits = 16
    clip.channels = 1
    clip.data = newSeq[uint8](clip.size)
    var arr = cast[ptr UncheckedArray[int16]](clip.data[0].addr)
    for i in 0..<iarr.len:
        # TODO: inaccurate
        arr[i] = int16(iarr[i]*32000.0)
    return clip


proc toFloat*(wav: WavFile): seq[float] =
    var rseq: seq[float] = @[]
    var arr = cast[ptr UncheckedArray[int16]](wav.data[0].unsafeAddr)
    let mpl = 1.0/32000.0
    for i in 0..<wav.size div 2:
        rseq.add(float(arr[i])*mpl)
    return rseq

proc loadWav*(filePath: string): WavFile =
    # Load PCM data from wav file.
    var f = newFileStream(filePath)

    var header = wavHeaderObj()
    var count = f.readData(addr(header), sizeof(wavHeaderObj))

    assert count == sizeof(wavHeaderObj)
    assert header.ChunkID == "RIFF"
    assert header.Format == "WAVE"
    assert header.FmtChunkID == "fmt "
    assert header.AudioFormat == 1

    var chunk = wavChunkObj()
    while chunk.DataChunkID != "data":
        chunk = f.readDataChunk()

    result.channels = int header.NumChannels
    result.size = chunk.Data.len
    result.freq = int header.SampleRate
    result.bits = int header.BitsPerSample
    result.data = cast[seq[uint8]](chunk.Data)

# TODO: saved data chunk-size is not the same as loaded chunk-size

proc saveWav*(wav: WavFile, filePath: string) =
    var f = newFileStream(filePath, fmWrite)
    if not isNil(f):
        f.wavWrite(wav)
    else:
        echo "Could not save .wav file: unable to open file for saving."

proc toHTML*(wav: WavFile, autoplay: bool): string =
    var f = newStringStream()
    f.wavWrite(wav)
    f.setPosition(0)
    var content = f.readAll()
    f.close()
    # This is the text/html part
    var aplay = (if autoplay: "autoplay=\"autoplay\"" else: "")
    return fmt"""
        <audio controls="controls" {aplay}>
        <source src="data:audio/wav;base64,{encode(content)}" type="audio/wav" />
        Your browser does not support the audio element.
        </audio>
        """


# proc audioBufferToHTMLStupidVSCodeVersion*(wav: WavFile): string =
#     var f = newStringStream()
#     f.wavWrite(wav)
#     f.setPosition(0)
#     var content = f.readAll()
#     f.close()
#     return fmt"""
#     <script>
#     var playAudio = function() {{
#         alert("foo!");
#         // console.log("Play");
#         // var snd = Audio("data:audio/wav;base64,{encode(content)}");
#         // snd.play();
#     }}
#     </script>
#     <button onclick="playAudio()">Play</button>
#     """

    # <script>
    # if (!window.audioContext) {{
    #     window.audioContext = new AudioContext();
    #     window.playAudio = function(audioChannels, sr) {
    #         const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
    #         for (let [channel, data] of audioChannels.entries()) {{
    #             buffer.copyToChannel(Float32Array.from(data), channel);
    #         }}
    
    #         const source = audioContext.createBufferSource();
    #         source.buffer = buffer;
    #         source.connect(audioContext.destination);
    #         source.start();
    #     }}
    # }}
    # </script>
    # <button onclick="playAudio(%s, %s)">Play</button>
    # """

proc Audio*(iarr: seq[float], freq: int, autoplay: bool): string =
    # Similar to Python Jupyter Audio
    return iarr.wavSeq(freq).toHTML(autoplay)

if isMainModule:
    echo ">> Loadwav"
    var sample_name = "sample.wav"
    var iwav = loadWav(sample_name)

    echo sample_name, ": ",
        iwav.size, " bytes, ",
        iwav.freq, " Hz, ",
        iwav.channels, " channels, ",
        iwav.bits, " bits, ",
        iwav.data.len, " length"

    echo ">> Savewav"
    saveWav(iwav, "out.wav")

    var iwav2 = loadWav("out.wav")

    echo iwav2.size, " bytes, ",
        iwav2.freq, " Hz, ",
        iwav2.channels, " channels, ",
        iwav2.bits, " bits"

    #echo toHTML(iwav)
