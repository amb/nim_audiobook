import arraymancer, pixie, chroma, sugar, strutils, strformat, sequtils, math, base64

# const ok_purple = polarOklab(color(1.0, 0.0, 1.0, 1.0))
# const ok_yellow = polarOklab(color(1.0, 1.0, 0.0, 1.0))

# proc magma(x: float): Color =
#     # TODO: colors don't seem to behave the way I think they do
#     # var c = polarOklab(i*255.0, ok_purple.C * (1.0 - x) + ok_yellow.C * x, ok_purple.h * (1.0 - x) + ok_yellow.h * x)
#     # result = color(c)
#     # let i=pow(x, 1.2)
#     result = color(hsv((1.0-x)*255.0, (1.0-x)*128.0+128.0, x*255.0))
#     result.a = 1.0

proc Image*(img: Image): string =
    return fmt"""<img src="data:image/png;base64, {img.encodeImage(PngFormat).encode}" />"""

proc plot1D*(img: Image, arr: openArray[float]): Image =
    var path: seq[string] = @[]
    let amax = arr[arr.maxIndex]
    let amin = arr[arr.minIndex]
    let mp = float(img.height)/(amax - amin)

    img.fill(rgb(255, 255, 255))

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

# proc plot1DC*(arr: openArray[float]): string =
#     return """
#         <canvas id="myCanvas" width="200" height="100" style="border:1px solid #000000;">
#         </canvas>"""

proc plot2DArray*(arr: Tensor[float]): Image =
    var array_out = newImage(arr.shape[0], arr.shape[1])
    let amin = arr.min()
    let amax = arr.max() - amin
    for x in 0..<arr.shape[0]:
        for y in 0..<arr.shape[1]:
            # let c = pow(arr[x, y], 0.5) * 0.95
            let c = (arr[x, y] - amin)/amax
            array_out[x, arr.shape[1]-y] = color(c, c, c)
            # array_out[x, arr.shape[1]-y] = magma(c)
    return array_out
