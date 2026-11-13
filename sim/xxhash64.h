// fireredsim: XXH64 (Yann Collet's xxHash, 64-bit variant), self-contained.
// Used for state hashes in determinism tests and for the snapshot build id.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pn {

namespace xxh64_detail {
constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t read64(const uint8_t *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
inline uint32_t read32(const uint8_t *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t round(uint64_t acc, uint64_t input)
{
    acc += input * P2;
    acc = rotl(acc, 31);
    return acc * P1;
}
inline uint64_t merge(uint64_t acc, uint64_t val)
{
    val = round(0, val);
    acc ^= val;
    return acc * P1 + P4;
}
} // namespace xxh64_detail

inline uint64_t xxh64(const void *data, size_t len, uint64_t seed = 0)
{
    using namespace xxh64_detail;
    const uint8_t *p = static_cast<const uint8_t *>(data);
    const uint8_t *end = p + len;
    uint64_t h;

    if (len >= 32) {
        const uint8_t *limit = end - 32;
        uint64_t v1 = seed + P1 + P2;
        uint64_t v2 = seed + P2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - P1;
        do {
            v1 = round(v1, read64(p)); p += 8;
            v2 = round(v2, read64(p)); p += 8;
            v3 = round(v3, read64(p)); p += 8;
            v4 = round(v4, read64(p)); p += 8;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1);
        h = merge(h, v2);
        h = merge(h, v3);
        h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += static_cast<uint64_t>(len);
    while (p + 8 <= end) {
        h ^= round(0, read64(p));
        h = rotl(h, 27) * P1 + P4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<uint64_t>(read32(p)) * P1;
        h = rotl(h, 23) * P2 + P3;
        p += 4;
    }
    while (p < end) {
        h ^= (*p) * P5;
        h = rotl(h, 11) * P1;
        p++;
    }
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

// Streaming XXH64: update() over several buffers gives the same digest as
// xxh64() over their concatenation (used to hash a list of memory regions).
class Xxh64Stream {
public:
    explicit Xxh64Stream(uint64_t seed = 0) : seed_(seed)
    {
        using namespace xxh64_detail;
        v_[0] = seed + P1 + P2;
        v_[1] = seed + P2;
        v_[2] = seed;
        v_[3] = seed - P1;
    }

    void update(const void *data, size_t len)
    {
        using namespace xxh64_detail;
        const uint8_t *p = static_cast<const uint8_t *>(data);
        const uint8_t *end = p + len;
        total_ += len;
        if (buf_len_ + len < 32) {
            std::memcpy(buf_ + buf_len_, p, len);
            buf_len_ += len;
            return;
        }
        if (buf_len_) {
            size_t fill = 32 - buf_len_;
            std::memcpy(buf_ + buf_len_, p, fill);
            consume(buf_);
            p += fill;
            buf_len_ = 0;
        }
        while (p + 32 <= end) {
            consume(p);
            p += 32;
        }
        buf_len_ = static_cast<size_t>(end - p);
        std::memcpy(buf_, p, buf_len_);
    }

    uint64_t digest() const
    {
        using namespace xxh64_detail;
        uint64_t h;
        if (total_ >= 32) {
            h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
            h = merge(h, v_[0]);
            h = merge(h, v_[1]);
            h = merge(h, v_[2]);
            h = merge(h, v_[3]);
        } else {
            h = seed_ + P5;
        }
        h += total_;
        const uint8_t *p = buf_;
        const uint8_t *end = buf_ + buf_len_;
        while (p + 8 <= end) {
            h ^= round(0, read64(p));
            h = rotl(h, 27) * P1 + P4;
            p += 8;
        }
        if (p + 4 <= end) {
            h ^= static_cast<uint64_t>(read32(p)) * P1;
            h = rotl(h, 23) * P2 + P3;
            p += 4;
        }
        while (p < end) {
            h ^= (*p) * P5;
            h = rotl(h, 11) * P1;
            p++;
        }
        h ^= h >> 33;
        h *= P2;
        h ^= h >> 29;
        h *= P3;
        h ^= h >> 32;
        return h;
    }

private:
    void consume(const uint8_t *p)
    {
        using namespace xxh64_detail;
        v_[0] = round(v_[0], read64(p));
        v_[1] = round(v_[1], read64(p + 8));
        v_[2] = round(v_[2], read64(p + 16));
        v_[3] = round(v_[3], read64(p + 24));
    }

    uint64_t seed_;
    uint64_t v_[4];
    uint64_t total_ = 0;
    uint8_t buf_[32];
    size_t buf_len_ = 0;
};

} // namespace pn
