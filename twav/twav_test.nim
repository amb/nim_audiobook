import futhark, std/random, std/math

importc:
  sysPath "/usr/lib/clang/13.0.1/include"
  path "tinywav"
  "tinywav.h"

{.compile: "tinywav/tinywav.c".}

let
  NUM_CHANNELS = 1.int16
  SAMPLE_RATE = 48000.int32

var tw: Tinywav
discard tinywav_open_write(tw.addr, NUM_CHANNELS, SAMPLE_RATE, TW_FLOAT32, TW_INLINE, "output.wav".cstring)

# Problem:
# twav_test.nim(17, 4) Error: request to generate code for .compileTime proc: Lit

var wavefoo = tinywav_open_write(tw.addr, NUM_CHANNELS, SAMPLE_RATE, TW_FLOAT32, TW_INLINE, "output.wav".cstring)
if wavefoo != 0:
  echo "Can't open output file"
  quit 1

var samples: array[480, float32]
for i in 0..<samples.len:
  samples[i] = sin(float(i)*2*PI/480)

for i in 0..<100:
  doAssert tinywav_write_f(tw.addr, samples.addr, samples.len.cint) > 0

tinywav_close_write(tw.addr)
