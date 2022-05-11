#!/bin/sh

#nim c -d:danger --cc:clang --passC:"-flto -fprofile-instr-generate" --passL:"-flto -fprofile-instr-generate" fft.nim
#./fft
#llvm-profdata merge default.profraw -output data.profdata
#nim c -d:danger --cc:clang --passC:"-flto -fprofile-instr-use=data.profdata" --passL:"-flto -fprofile-instr-use=data.profdata" fft.nim
#./fft

nim c -d:danger --passC:"-flto -fprofile-generate" --passL:"-flto -fprofile-generate" fft.nim
./fft
llvm-profdata merge default.profraw -output data.profdata
nim c -d:danger --passC:"-flto -fprofile-use" --passL:"-flto -fprofile-use" fft.nim
./fft
