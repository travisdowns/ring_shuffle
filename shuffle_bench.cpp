// shuffle_bench.cpp — Standalone benchmark for three intra-process shuffle algorithms.
// Build: g++ -std=c++20 -O2 -pthread -o shuffle_bench shuffle_bench.cpp
//   (add -msse4.2 on x86 or -march=armv8-a+crc on aarch64 for hardware CRC32)
// Usage: ./shuffle_bench [variants] [M] [N] [rows_per_chunk] [row_size] [num_chunks] [repeats] [ring_k] [dist]
//   variants: string of letters selecting which algorithms to run (default: BCR)
//     B = Batch Partitioning, C = Channel Streaming, R = Ring-Buffer Streaming
//   dist: "flat" (default) or "normal" (row sizes ~ N(row_size, row_size/4))

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <algorithm>
#include <array>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Common types and utilities
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    int M            = 4;      // producer threads
    int N            = 4;      // consumer threads (partitions)
    int rows         = 4096;   // rows per chunk
    int row_size     = 64;     // bytes per row (mean for normal distribution)
    int num_chunks   = 10000;  // total chunks to produce
    int repeats      = 1;      // runs per algorithm (report median)
    int ring_k       = 2;      // ring buffer size (K)
    bool normal_dist = false;  // normal distribution for row sizes
    std::string variants = "BCR";
};

static void pin_thread(int core) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

struct Chunk {
    std::vector<uint8_t> data;
    std::vector<uint32_t> row_offsets;  // size = rows + 1; row i is data[row_offsets[i]..row_offsets[i+1])
    int num_rows;

    const uint8_t* row_ptr(int r) const { return data.data() + row_offsets[r]; }
    int row_len(int r) const { return static_cast<int>(row_offsets[r + 1] - row_offsets[r]); }
};
using ChunkPtr = std::shared_ptr<Chunk>;

static std::atomic<uint32_t> g_crc_sink{0};

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

static ChunkPtr generate_chunk(std::mt19937_64& rng, int rows, int row_size, bool normal_dist) {
    auto chunk = std::make_shared<Chunk>();
    chunk->num_rows = rows;
    chunk->row_offsets.resize(static_cast<size_t>(rows) + 1);

    if (!normal_dist) {
        // Fixed row sizes — flat layout.
        for (int r = 0; r <= rows; ++r)
            chunk->row_offsets[r] = static_cast<uint32_t>(r) * row_size;
    } else {
        // Normal distribution: mean = row_size, stddev = row_size / 4.
        std::normal_distribution<double> dist(row_size, row_size / 4.0);
        uint32_t offset = 0;
        for (int r = 0; r < rows; ++r) {
            chunk->row_offsets[r] = offset;
            int sz = std::max(1, static_cast<int>(std::round(dist(rng))));
            offset += static_cast<uint32_t>(sz);
        }
        chunk->row_offsets[rows] = offset;
    }

    size_t total = chunk->row_offsets[rows];
    chunk->data.resize(total);
    fill_random(chunk->data.data(), total, rng);
    return chunk;
}

#if defined(__SSE4_2__)
static inline uint32_t crc32_u8(uint32_t crc, uint8_t b) { return _mm_crc32_u8(crc, b); }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
static inline uint32_t crc32_u8(uint32_t crc, uint8_t b) { return __crc32cb(crc, b); }
#else
// Software CRC32C fallback (Castagnoli polynomial).
static constexpr auto crc32c_table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j)
            c = (c >> 1) ^ (c & 1 ? 0x82F63B78u : 0);
        t[i] = c;
    }
    return t;
}();
static inline uint32_t crc32_u8(uint32_t crc, uint8_t b) {
    return (crc >> 8) ^ crc32c_table[(crc ^ b) & 0xFF];
}
#endif

static uint32_t crc32_bytes(const uint8_t* data, size_t len) {
    uint32_t crc = 0;
    for (size_t i = 0; i < len; ++i)
        crc = crc32_u8(crc, data[i]);
    return crc;
}

static uint32_t partition_hash(const uint8_t* row, int row_size, int N) {
    size_t len = static_cast<size_t>(std::min(row_size, 4));
    return crc32_bytes(row, len) % static_cast<uint32_t>(N);
}

static uint32_t compute_row_crc(const uint8_t* row, int row_size) {
    return crc32_bytes(row, static_cast<size_t>(row_size));
}

// IndexedBatch: precomputed partition assignment via counting sort.
struct IndexedBatch {
    ChunkPtr chunk;
    std::vector<uint32_t> offsets;      // N+1 partition boundaries
    std::vector<uint32_t> permutation;  // row indices sorted by partition

    static std::shared_ptr<IndexedBatch> build(ChunkPtr c, int N) {
        auto ib = std::make_shared<IndexedBatch>();
        int rows = c->num_rows;
        ib->chunk = std::move(c);
        ib->offsets.resize(static_cast<size_t>(N + 1), 0);
        ib->permutation.resize(static_cast<size_t>(rows));

        const Chunk& ch = *ib->chunk;

        // Count rows per partition.
        for (int r = 0; r < rows; ++r)
            ++ib->offsets[partition_hash(ch.row_ptr(r), ch.row_len(r), N) + 1];

        // Prefix sum.
        for (int j = 1; j <= N; ++j)
            ib->offsets[j] += ib->offsets[j - 1];

        // Place rows (counting sort scatter).
        auto pos = ib->offsets; // copy
        for (int r = 0; r < rows; ++r) {
            uint32_t p = partition_hash(ch.row_ptr(r), ch.row_len(r), N);
            ib->permutation[pos[p]++] = static_cast<uint32_t>(r);
        }
        return ib;
    }

    std::span<const uint32_t> rows_for(int partition) const {
        return {permutation.data() + offsets[partition],
                permutation.data() + offsets[partition + 1]};
    }
};

using IndexedBatchPtr = std::shared_ptr<IndexedBatch>;

struct BenchResult {
    double time_ms;
    double throughput_gbs;
    uint32_t crc;
};

static BenchResult run_and_measure(double gb, int repeats, std::function<uint32_t()> fn) {
    std::vector<double> times;
    times.reserve(repeats);
    uint32_t crc = 0;
    for (int i = 0; i < repeats; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        crc = fn();
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    double median_ms = times[times.size() / 2];
    return {median_ms, gb / (median_ms / 1000.0), crc};
}

// Estimate total data volume by generating one chunk per producer and extrapolating.
static double estimate_total_gb(const Config& cfg) {
    if (!cfg.normal_dist)
        return static_cast<double>(cfg.num_chunks) * cfg.rows * cfg.row_size / 1e9;
    double total = 0;
    for (int p = 0; p < cfg.M; ++p) {
        std::mt19937_64 rng(p + 1);
        auto c = generate_chunk(rng, cfg.rows, cfg.row_size, true);
        int chunks_for_p = cfg.num_chunks / cfg.M + (p < cfg.num_chunks % cfg.M ? 1 : 0);
        total += static_cast<double>(c->data.size()) * chunks_for_p;
    }
    return total / 1e9;
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 1: Batch Partitioning
// ─────────────────────────────────────────────────────────────────────────────
namespace batch {

static uint32_t run(const Config& cfg) {
    int M = cfg.M, N = cfg.N;

    // Per-producer: N partition buffers, each a vector of ChunkPtrs.
    // Producers build an IndexedBatch per chunk (radix-partitioned), then append the
    // whole batch pointer to the appropriate partition bucket — no row-by-row copy.
    struct PartitionBucket {
        std::vector<IndexedBatchPtr> batches;
    };
    std::vector<std::vector<PartitionBucket>> partitions(M, std::vector<PartitionBucket>(N));

    auto chunks_for = [&](int p) -> std::pair<int, int> {
        int base = cfg.num_chunks / M;
        int rem  = cfg.num_chunks % M;
        int lo   = p * base + std::min(p, rem);
        int hi   = lo + base + (p < rem ? 1 : 0);
        return {lo, hi};
    };

    // Phase 1: Sink — each producer partitions chunks via IndexedBatch, no inter-thread communication.
    std::vector<std::thread> producers;
    producers.reserve(M);
    for (int p = 0; p < M; ++p) {
        producers.emplace_back([&, p] {
            pin_thread(p);
            std::mt19937_64 rng(p + 1);
            auto [lo, hi] = chunks_for(p);
            for (int c = lo; c < hi; ++c) {
                auto chunk = generate_chunk(rng, cfg.rows, cfg.row_size, cfg.normal_dist);
                auto ib = IndexedBatch::build(std::move(chunk), N);
                // Append the same batch pointer to every partition that has rows.
                for (int j = 0; j < N; ++j) {
                    if (ib->offsets[j] != ib->offsets[j + 1])
                        partitions[p][j].batches.push_back(ib);
                }
            }
        });
    }
    for (auto& t : producers) t.join();

    // Phase 2: Consume — each consumer reads its partition across all producers.
    std::atomic<uint32_t> crc_accum{0};
    std::vector<std::thread> consumers;
    consumers.reserve(N);
    for (int j = 0; j < N; ++j) {
        consumers.emplace_back([&, j] {
            pin_thread(j);
            uint32_t local_crc = 0;
            for (int p = 0; p < M; ++p) {
                for (auto& ib : partitions[p][j].batches) {
                    const Chunk& ch = *ib->chunk;
                    for (uint32_t idx : ib->rows_for(j))
                        local_crc ^= compute_row_crc(ch.row_ptr(idx), ch.row_len(idx));
                }
            }
            crc_accum.fetch_xor(local_crc);
        });
    }
    for (auto& t : consumers) t.join();
    return crc_accum.load();
}

} // namespace batch

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 2: Channel-Based Streaming
// ─────────────────────────────────────────────────────────────────────────────
namespace channel {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void push(T val) {
        std::unique_lock lk(mu_);
        cv_not_full_.wait(lk, [&] { return queue_.size() < cap_ || closed_; });
        if (closed_) return;
        queue_.push_back(std::move(val));
        cv_not_empty_.notify_one();
    }

    std::optional<T> pull() {
        std::unique_lock lk(mu_);
        cv_not_empty_.wait(lk, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return std::nullopt;
        T val = std::move(queue_.front());
        queue_.erase(queue_.begin());
        cv_not_full_.notify_one();
        return val;
    }

    void close() {
        std::lock_guard lk(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    size_t cap_;
    std::vector<T> queue_;
    std::mutex mu_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    bool closed_ = false;
};

static uint32_t run(const Config& cfg) {
    int M = cfg.M, N = cfg.N;

    std::vector<std::unique_ptr<BoundedQueue<IndexedBatchPtr>>> channels;
    channels.reserve(N);
    for (int j = 0; j < N; ++j)
        channels.push_back(std::make_unique<BoundedQueue<IndexedBatchPtr>>(M));

    std::atomic<int> producers_done{0};

    auto chunks_for = [&](int p) -> std::pair<int, int> {
        int base = cfg.num_chunks / M;
        int rem  = cfg.num_chunks % M;
        int lo   = p * base + std::min(p, rem);
        int hi   = lo + base + (p < rem ? 1 : 0);
        return {lo, hi};
    };

    // Launch all producers and consumers simultaneously.
    std::vector<std::thread> threads;
    threads.reserve(M + N);
    std::atomic<uint32_t> crc_accum{0};

    // Producers.
    for (int p = 0; p < M; ++p) {
        threads.emplace_back([&, p] {
            pin_thread(p);
            std::mt19937_64 rng(p + 1);
            auto [lo, hi] = chunks_for(p);
            for (int c = lo; c < hi; ++c) {
                auto chunk = generate_chunk(rng, cfg.rows, cfg.row_size, cfg.normal_dist);
                auto ib = IndexedBatch::build(std::move(chunk), N);
                for (int j = 0; j < N; ++j)
                    channels[j]->push(ib);
            }
            if (producers_done.fetch_add(1) + 1 == M) {
                for (int j = 0; j < N; ++j)
                    channels[j]->close();
            }
        });
    }

    // Consumers.
    for (int j = 0; j < N; ++j) {
        threads.emplace_back([&, j] {
            pin_thread(j);
            uint32_t local_crc = 0;
            while (auto ib = channels[j]->pull()) {
                const Chunk& ch = *(*ib)->chunk;
                for (uint32_t idx : (*ib)->rows_for(j)) {
                    local_crc ^= compute_row_crc(ch.row_ptr(idx), ch.row_len(idx));
                }
            }
            crc_accum.fetch_xor(local_crc);
        });
    }

    for (auto& t : threads) t.join();
    return crc_accum.load();
}

} // namespace channel

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 3: Ring-Buffer Streaming
// ─────────────────────────────────────────────────────────────────────────────
namespace ring {

struct BatchGroup {
    std::vector<IndexedBatchPtr> slots;
    std::atomic<size_t> writes_started{0};
    std::atomic<size_t> writes_done{0};
    std::atomic<size_t> readers_remaining;
    bool is_full = false;

    explicit BatchGroup(int G, int N)
        : slots(static_cast<size_t>(G)), readers_remaining(static_cast<size_t>(N)) {}
};

using BatchGroupPtr = std::shared_ptr<BatchGroup>;

// Per-producer container: holds a raw pointer to the current group,
// protected by a per-producer mutex. The publisher updates all containers
// when rotating to a new group.
struct InsertionBufferContainer {
    std::mutex mtx;
    BatchGroup* bg = nullptr;

    void replace(BatchGroup* new_bg) {
        std::lock_guard lk(mtx);
        bg = new_bg;
    }
};

struct ShuffleQueue {
    int G, N;
    size_t ring_size;

    std::vector<BatchGroupPtr> ring;
    std::atomic<size_t> published{0};
    size_t consumed_min = 0;

    BatchGroupPtr insertion_buffer;

    std::mutex mu;
    std::condition_variable cv_not_full;
    std::condition_variable cv_not_empty;
    std::atomic<uint64_t> buffer_generation{0};

    bool finished = false;

    std::vector<InsertionBufferContainer*> containers;

    ShuffleQueue(int G, int N, size_t ring_size)
        : G(G), N(N), ring_size(ring_size), ring(ring_size) {
        insertion_buffer = std::make_shared<BatchGroup>(G, N);
    }

    void register_container(InsertionBufferContainer* c) {
        c->bg = insertion_buffer.get();
        containers.push_back(c);
    }

    void insertion_buffer_ready(BatchGroupPtr replacement) {
        {
            std::unique_lock lk(mu);
            size_t pub = published.load(std::memory_order_relaxed);
            cv_not_full.wait(lk, [&] {
                return (pub - consumed_min) < ring_size || finished;
            });
            if (finished) return;
            ring[pub % ring_size] = std::move(insertion_buffer);
            published.store(pub + 1, std::memory_order_release);
            insertion_buffer = std::move(replacement);
        }
        // Broadcast new insertion buffer to all producers (per-producer lock each).
        BatchGroup* bg = insertion_buffer.get();
        for (auto* c : containers)
            c->replace(bg);
        ++buffer_generation;
        buffer_generation.notify_all();
        cv_not_empty.notify_all();
    }

    void finish_stream() {
        {
            std::unique_lock lk(mu);
            auto grp = insertion_buffer;
            size_t started = grp->writes_started.load(std::memory_order_acquire);
            if (started > 0) {
                lk.unlock();
                while (grp->writes_done.load(std::memory_order_acquire) < started)
                    std::this_thread::yield();
                lk.lock();
                size_t pub = published.load(std::memory_order_relaxed);
                cv_not_full.wait(lk, [&] {
                    return (pub - consumed_min) < ring_size;
                });
                ring[pub % ring_size] = std::move(grp);
                published.store(pub + 1, std::memory_order_release);
            }
            finished = true;
        }
        cv_not_empty.notify_all();
        ++buffer_generation;
        buffer_generation.notify_all();
    }

    // path: 0=cached, 1=atomic, 2=condvar
    BatchGroupPtr wait_for_group(size_t& read_pos, size_t& cached_pub, int& path) {
        if (read_pos < cached_pub) {
            path = 0;
            return ring[read_pos % ring_size];
        }
        cached_pub = published.load(std::memory_order_acquire);
        if (read_pos < cached_pub) {
            path = 1;
            return ring[read_pos % ring_size];
        }
        std::unique_lock lk(mu);
        cv_not_empty.wait(lk, [&] {
            cached_pub = published.load(std::memory_order_acquire);
            return read_pos < cached_pub || finished;
        });
        path = 2;
        if (read_pos >= cached_pub)
            return nullptr;
        return ring[read_pos % ring_size];
    }

    void release_group(size_t pos) {
        auto& grp = ring[pos % ring_size];
        size_t rem = grp->readers_remaining.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (rem == 0) {
            bool should_notify;
            {
                std::lock_guard lk(mu);
                grp.reset();
                consumed_min = pos + 1;
                size_t occupancy = published.load(std::memory_order_relaxed) - consumed_min;
                should_notify = occupancy <= ring_size / 2;
            }
            if (should_notify)
                cv_not_full.notify_one();
        }
    }
};

#ifdef RING_STATS
struct RingStatsData {
    struct ProducerStats {
        double generate_ms = 0, build_ms = 0, claim_ms = 0, complete_ms = 0;
        int claim_retries = 0;
    };
    struct ConsumerStats {
        double wait_ms = 0, process_ms = 0, release_ms = 0;
        double wait_cached_ms = 0, wait_atomic_ms = 0, wait_condvar_ms = 0;
        int groups_consumed = 0, cached_hits = 0, atomic_hits = 0, condvar_hits = 0;
    };
    std::vector<ProducerStats> producers;
    std::vector<ConsumerStats> consumers;

    void print(int M, int N) const {
        double gen_sum = 0, build_sum = 0, claim_sum = 0, complete_sum = 0;
        int retries_sum = 0;
        double gen_max = 0, build_max = 0, claim_max = 0, complete_max = 0;
        for (int i = 0; i < M; ++i) {
            auto& s = producers[i];
            gen_sum += s.generate_ms;   gen_max = std::max(gen_max, s.generate_ms);
            build_sum += s.build_ms;    build_max = std::max(build_max, s.build_ms);
            claim_sum += s.claim_ms;    claim_max = std::max(claim_max, s.claim_ms);
            complete_sum += s.complete_ms; complete_max = std::max(complete_max, s.complete_ms);
            retries_sum += s.claim_retries;
        }
        double wait_sum = 0, proc_sum = 0, rel_sum = 0;
        double wait_max = 0, proc_max = 0, rel_max = 0;
        int groups_sum = 0, cached_sum = 0, atomic_sum = 0, condvar_sum = 0;
        double wcached_sum = 0, watomic_sum = 0, wcondvar_sum = 0;
        for (int i = 0; i < N; ++i) {
            auto& s = consumers[i];
            wait_sum += s.wait_ms;    wait_max = std::max(wait_max, s.wait_ms);
            proc_sum += s.process_ms; proc_max = std::max(proc_max, s.process_ms);
            rel_sum += s.release_ms;  rel_max = std::max(rel_max, s.release_ms);
            groups_sum += s.groups_consumed;
            cached_sum += s.cached_hits;  atomic_sum += s.atomic_hits;  condvar_sum += s.condvar_hits;
            wcached_sum += s.wait_cached_ms;  watomic_sum += s.wait_atomic_ms;  wcondvar_sum += s.wait_condvar_ms;
        }
        std::printf("  Ring instrumentation:\n");
        std::printf("    Producer (avg / max over %d threads):\n", M);
        std::printf("      generate:  %7.1f / %7.1f ms\n", gen_sum / M, gen_max);
        std::printf("      build:     %7.1f / %7.1f ms\n", build_sum / M, build_max);
        std::printf("      claim:     %7.1f / %7.1f ms  (retries total: %d)\n", claim_sum / M, claim_max, retries_sum);
        std::printf("      complete:  %7.1f / %7.1f ms\n", complete_sum / M, complete_max);
        std::printf("    Consumer (avg / max over %d threads):\n", N);
        std::printf("      wait:      %7.1f / %7.1f ms\n", wait_sum / N, wait_max);
        std::printf("      process:   %7.1f / %7.1f ms\n", proc_sum / N, proc_max);
        std::printf("      release:   %7.1f / %7.1f ms\n", rel_sum / N, rel_max);
        std::printf("      groups:    %d total (%d per consumer avg)\n", groups_sum, groups_sum / N);
        int total_hits = cached_sum + atomic_sum + condvar_sum;
        std::printf("      wait path: cached=%d (%.0f%%)  atomic=%d (%.0f%%)  condvar=%d (%.0f%%)\n",
                    cached_sum, 100.0 * cached_sum / std::max(1, total_hits),
                    atomic_sum, 100.0 * atomic_sum / std::max(1, total_hits),
                    condvar_sum, 100.0 * condvar_sum / std::max(1, total_hits));
        std::printf("      wait time: cached=%.1f  atomic=%.1f  condvar=%.1f ms (avg per consumer)\n",
                    wcached_sum / N, watomic_sum / N, wcondvar_sum / N);
    }
};
#endif // RING_STATS

static uint32_t run(const Config& cfg, int G, size_t R) {
    int M = cfg.M, N = cfg.N;
    ShuffleQueue queue(G, N, R);

    std::atomic<int> producers_done{0};
#ifdef RING_STATS
    RingStatsData stats;
    stats.producers.resize(M);
    stats.consumers.resize(N);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
#endif

    auto chunks_for = [&](int p) -> std::pair<int, int> {
        int base = cfg.num_chunks / M;
        int rem  = cfg.num_chunks % M;
        int lo   = p * base + std::min(p, rem);
        int hi   = lo + base + (p < rem ? 1 : 0);
        return {lo, hi};
    };

    std::vector<std::thread> threads;
    threads.reserve(M + N);
    std::atomic<uint32_t> crc_accum{0};

    // Register per-producer containers.
    std::vector<InsertionBufferContainer> containers(M);
    for (int p = 0; p < M; ++p)
        queue.register_container(&containers[p]);

    // Producers.
    for (int p = 0; p < M; ++p) {
        threads.emplace_back([&, p] {
            pin_thread(p);
#ifdef RING_STATS
            auto& st = stats.producers[p];
#endif
            auto& container = containers[p];
            std::mt19937_64 rng(p + 1);
            auto replacement = std::make_shared<BatchGroup>(G, N);
            auto [lo, hi] = chunks_for(p);
            for (int c = lo; c < hi; ++c) {
#ifdef RING_STATS
                auto t0 = now();
#endif
                auto chunk = generate_chunk(rng, cfg.rows, cfg.row_size, cfg.normal_dist);
#ifdef RING_STATS
                st.generate_ms += ms_since(t0);
                t0 = now();
#endif
                auto ib = IndexedBatch::build(std::move(chunk), N);
#ifdef RING_STATS
                st.build_ms += ms_since(t0);
                t0 = now();
#endif
                // Push batch into current group (Oxla pattern).
                while (true) {
                    BatchGroup* bg;
                    size_t ticket;
                    bool is_full;
                    uint64_t gen = queue.buffer_generation.load(std::memory_order_acquire);
                    {
                        std::lock_guard lk(container.mtx);
                        bg = container.bg;
                        is_full = bg->is_full;
                        if (!is_full) {
                            ticket = bg->writes_started.fetch_add(1, std::memory_order_acq_rel);
                            if (ticket >= static_cast<size_t>(G)) {
#ifdef RING_STATS
                                ++st.claim_retries;
#endif
                                continue;  // overshoot, retry
                            }
                        }
                    }
                    if (is_full) {
                        // Wait for new buffer — lock-free futex wait.
                        queue.buffer_generation.wait(gen, std::memory_order_relaxed);
#ifdef RING_STATS
                        ++st.claim_retries;
#endif
                        continue;
                    }
#ifdef RING_STATS
                    st.claim_ms += ms_since(t0);
                    t0 = now();
#endif
                    bg->slots[ticket] = std::move(ib);
                    size_t done = bg->writes_done.fetch_add(1, std::memory_order_acq_rel) + 1;
                    if (done == static_cast<size_t>(G)) {
                        bg->is_full = true;
                        queue.insertion_buffer_ready(std::move(replacement));
                        replacement = std::make_shared<BatchGroup>(G, N);
                    }
#ifdef RING_STATS
                    st.complete_ms += ms_since(t0);
#endif
                    break;
                }
            }
            if (producers_done.fetch_add(1) + 1 == M) {
                queue.finish_stream();
            }
        });
    }

    // Consumers.
    for (int j = 0; j < N; ++j) {
        threads.emplace_back([&, j] {
            pin_thread(j);
#ifdef RING_STATS
            auto& st = stats.consumers[j];
#endif
            uint32_t local_crc = 0;
            size_t read_pos = 0;
            size_t cached_pub = 0;
            while (true) {
#ifdef RING_STATS
                int path = 0;
                auto t0 = now();
                auto grp = queue.wait_for_group(read_pos, cached_pub, path);
                double elapsed = ms_since(t0);
                st.wait_ms += elapsed;
#else
                int path;
                auto grp = queue.wait_for_group(read_pos, cached_pub, path);
#endif
                if (!grp) break;
#ifdef RING_STATS
                ++st.groups_consumed;
                if (path == 0)      { ++st.cached_hits;  st.wait_cached_ms  += elapsed; }
                else if (path == 1) { ++st.atomic_hits;  st.wait_atomic_ms  += elapsed; }
                else                { ++st.condvar_hits; st.wait_condvar_ms += elapsed; }
                t0 = now();
#endif
                for (auto& ib : grp->slots) {
                    if (!ib) continue;
                    const Chunk& ch = *ib->chunk;
                    for (uint32_t idx : ib->rows_for(j)) {
                        local_crc ^= compute_row_crc(ch.row_ptr(idx), ch.row_len(idx));
                    }
                }
#ifdef RING_STATS
                st.process_ms += ms_since(t0);
                t0 = now();
#endif
                queue.release_group(read_pos);
#ifdef RING_STATS
                st.release_ms += ms_since(t0);
#endif
                ++read_pos;
            }
            crc_accum.fetch_xor(local_crc);
        });
    }

    for (auto& t : threads) t.join();
#ifdef RING_STATS
    stats.print(M, N);
#endif
    return crc_accum.load();
}

} // namespace ring

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Config cfg;
    if (argc > 1) cfg.variants   = argv[1];
    if (argc > 2) cfg.M          = std::atoi(argv[2]);
    if (argc > 3) cfg.N          = std::atoi(argv[3]);
    if (argc > 4) cfg.rows       = std::atoi(argv[4]);
    if (argc > 5) cfg.row_size   = std::atoi(argv[5]);
    if (argc > 6) cfg.num_chunks = std::atoi(argv[6]);
    if (argc > 7) cfg.repeats    = std::atoi(argv[7]);
    if (argc > 8) cfg.ring_k     = std::atoi(argv[8]);
    if (argc > 9) cfg.normal_dist = (std::string(argv[9]) == "normal");

    auto has = [&](char c) { return cfg.variants.find(c) != std::string::npos; };

    double gb = estimate_total_gb(cfg);

    auto record = [&](const char* name, BenchResult result) {
        g_crc_sink.fetch_xor(result.crc);
        std::printf("  %-12s %7.1f ms  %6.2f GB/s\n", name, result.time_ms, result.throughput_gbs);
    };

    if (has('B'))
        record("Batch", run_and_measure(gb, cfg.repeats, [&] { return batch::run(cfg); }));
    if (has('C'))
        record("Channel", run_and_measure(gb, cfg.repeats, [&] { return channel::run(cfg); }));
    if (has('R')) {
        char label[32];
        std::snprintf(label, sizeof(label), "Ring K=%d", cfg.ring_k);
        record(label, run_and_measure(gb, cfg.repeats, [&] { return ring::run(cfg, cfg.M, cfg.ring_k); }));
    }

    return 0;
}
