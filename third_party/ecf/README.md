# ECF decoder

`ecf_codec.cpp`, `ecf_codec.h` and `LICENSE` are unmodified copies from
[prophesee-ai/hdf5_ecf](https://github.com/prophesee-ai/hdf5_ecf) at commit
`8735ce7161a637cacc5f0affef48129a13ca5a3f` (Apache 2.0, Copyright Prophesee S.A.).
`ecf_c_api.cpp` and `ecf_codec_export.h` are ours.

Build `ecf.dll` from this folder (needs a 64-bit g++):

    g++ -O2 -std=c++14 -shared -static -o ecf.dll ecf_codec.cpp ecf_c_api.cpp
