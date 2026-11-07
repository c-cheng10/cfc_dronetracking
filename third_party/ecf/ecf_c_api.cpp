// Plain C wrapper so Python (ctypes) can call Prophesee's ECF decoder.
#include <cstdint>
#include "ecf_codec.h"

extern "C" {

// Number of events stored in one compressed chunk (read from its 4-byte header).
__declspec(dllexport) long long ecf_num_events(const std::uint8_t *data) {
    return static_cast<long long>(*reinterpret_cast<const std::uint32_t *>(data) >> 2);
}

// Decode one chunk into out. Returns the number of events, or -1 on failure.
__declspec(dllexport) long long ecf_decode(const std::uint8_t *data, long long nbytes, ECF::EventCD *out) {
    try {
        ECF::Decoder decoder;
        return static_cast<long long>(decoder(data, data + nbytes, out) / sizeof(ECF::EventCD));
    } catch (...) {
        return -1;
    }
}
}
