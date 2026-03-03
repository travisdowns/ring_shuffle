#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif

// ---------- CRC32C ----------------------------------------------------------

#ifdef __SSE4_2__
static uint32_t crc32_bytes(const uint8_t* data, size_t len) {
    uint32_t crc = 0;
    for (size_t i = 0; i < len; ++i)
        crc = _mm_crc32_u8(crc, data[i]);
    return crc;
}
#else
// Software CRC32C fallback (Castagnoli polynomial).
static uint32_t crc32c_table[256];
static bool crc32c_table_init = [] {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j)
            c = (c >> 1) ^ (c & 1 ? 0x82F63B78u : 0);
        crc32c_table[i] = c;
    }
    return true;
}();
static uint32_t crc32_bytes(const uint8_t* data, size_t len) {
    (void)crc32c_table_init;
    uint32_t crc = 0;
    for (size_t i = 0; i < len; ++i)
        crc = (crc >> 8) ^ crc32c_table[(crc ^ data[i]) & 0xFF];
    return crc;
}
#endif

static uint32_t compute_row_crc(const uint8_t* row, int row_size) {
    return crc32_bytes(row, static_cast<size_t>(row_size));
}

// ---------- Data generation -------------------------------------------------

static void fill_random(uint8_t* p, size_t total, std::mt19937_64& rng) {
    size_t i = 0;
    for (; i + 8 <= total; i += 8) {
        uint64_t v = rng();
        std::memcpy(p + i, &v, 8);
    }
    if (i < total) {
        uint64_t v = rng();
        std::memcpy(p + i, &v, total - i);
    }
}

// ---------- Benchmark -------------------------------------------------------

int main(int argc, char* argv[]) {
    int row_size = 64;
    int num_rows = 8192;
    int repeats  = 5;

    if (argc > 1) row_size = std::atoi(argv[1]);
    if (argc > 2) num_rows = std::atoi(argv[2]);
    if (argc > 3) repeats  = std::atoi(argv[3]);

    size_t total_bytes = static_cast<size_t>(row_size) * num_rows;
    std::vector<uint8_t> data(total_bytes);

    std::mt19937_64 rng(42);
    fill_random(data.data(), total_bytes, rng);

    std::vector<double> times;
    times.reserve(repeats);
    uint32_t crc = 0;

    for (int i = 0; i < repeats; ++i) {
        uint32_t local_crc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < num_rows; ++r) {
            local_crc ^= compute_row_crc(data.data() + static_cast<size_t>(r) * row_size, row_size);
        }
        auto t1 = std::chrono::steady_clock::now();
        crc = local_crc;
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];
    double gb = static_cast<double>(total_bytes) / 1e9;
    double gbs = gb / (median_ms / 1000.0);

    std::printf("rows=%d  row_size=%d  total=%.2f MB  median=%.3f ms  %.2f GB/s  crc=%08x\n",
                num_rows, row_size, static_cast<double>(total_bytes) / 1e6,
                median_ms, gbs, crc);

    return 0;
}
