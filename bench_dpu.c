// bench_dpu.c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <mram.h>
#include <defs.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

// Host-visible config & results
struct bench_cfg {
    uint32_t test_id;      // which test
    uint32_t bytes;        // transfer size per op
    uint32_t stride;       // bytes between blocks (0 = contiguous)
    uint32_t reps;         // repetitions per tasklet
    uint32_t flags;        // bit flags (0=default), bit0=misalign
};
struct bench_out {
    uint64_t cycles_total; // sum cycles over reps
    uint32_t ops_done;     // reps completed
    uint32_t pad;
};
__host struct bench_cfg CFG;
__host struct bench_out OUT[NR_TASKLETS];

// Shared barrier
BARRIER_INIT(bar_sync, NR_TASKLETS);

// --- sizing & alignment ---
#define MRAM_BUF_SIZE (8 * 1024 * 1024)  // 8 MiB scratch region in MRAM
__mram_noinit __attribute__((aligned(8))) uint8_t MRAM_BUF[MRAM_BUF_SIZE];

// WRAM scratch kept small to fit stacks (with 16 tasklets & 512B stacks)
#define CHUNK_MAX 2048
__dma_aligned static uint8_t wbuf[CHUNK_MAX];
__dma_aligned static uint8_t wbuf2[CHUNK_MAX];

static inline uint32_t align8_down_u32(uint32_t x) { return x & ~7u; }

static inline void do_barrier() { barrier_wait(&bar_sync); }

static void bench_mram_read(uint32_t tid, const struct bench_cfg *c, struct bench_out *o) {
    const uint32_t req = (c->bytes <= CHUNK_MAX ? c->bytes : CHUNK_MAX);
    const uint32_t sz   = align8_down_u32(req);
    const uint32_t stride = align8_down_u32(c->stride);
    const uint32_t reps = c->reps;
    if (sz == 0) { o->cycles_total = 0; o->ops_done = 0; return; }

    const uint32_t base = align8_down_u32(tid * (stride ? stride : sz));
    perfcounter_config(COUNT_CYCLES, true);
    const uint64_t start = perfcounter_get();
    for (uint32_t r = 0; r < reps; r++) {
        const uint32_t idx = base + (stride ? r * stride : r * sz);
        if (idx + sz > MRAM_BUF_SIZE) break;     // clamp
        mram_read(&MRAM_BUF[idx], wbuf, sz);
    }
    const uint64_t end = perfcounter_get();
    o->cycles_total = end - start;
    o->ops_done = reps;
}

static void bench_mram_write(uint32_t tid, const struct bench_cfg *c, struct bench_out *o) {
    const uint32_t req = (c->bytes <= CHUNK_MAX ? c->bytes : CHUNK_MAX);
    const uint32_t sz   = align8_down_u32(req);
    const uint32_t stride = align8_down_u32(c->stride);
    const uint32_t reps = c->reps;
    if (sz == 0) { o->cycles_total = 0; o->ops_done = 0; return; }

    for (uint32_t i = 0; i < sz; i++) wbuf[i] = (uint8_t)(i + tid);
    const uint32_t base = align8_down_u32(tid * (stride ? stride : sz));
    perfcounter_config(COUNT_CYCLES, true);
    const uint64_t start = perfcounter_get();
    for (uint32_t r = 0; r < reps; r++) {
        const uint32_t idx = base + (stride ? r * stride : r * sz);
        if (idx + sz > MRAM_BUF_SIZE) break;
        mram_write(wbuf, &MRAM_BUF[idx], sz);
    }
    const uint64_t end = perfcounter_get();
    o->cycles_total = end - start;
    o->ops_done = reps;
}


static void bench_wram_memcpy(uint32_t tid, const struct bench_cfg *c, struct bench_out *o) {
    const uint32_t sz = c->bytes <= CHUNK_MAX ? c->bytes : CHUNK_MAX;
    const uint32_t reps = c->reps;
    for (uint32_t i = 0; i < sz; i++) wbuf[i] = (uint8_t)(tid + i);
    perfcounter_config(COUNT_CYCLES, true);
    uint64_t start = perfcounter_get();
    for (uint32_t r = 0; r < reps; r++) {
        // simple copy
        for (uint32_t i = 0; i < sz; i++) wbuf2[i] = wbuf[i];
    }
    uint64_t end = perfcounter_get();
    o->cycles_total = end - start;
    o->ops_done = reps;
}

static void bench_barrier(uint32_t tid, const struct bench_cfg *c, struct bench_out *o) {
    const uint32_t reps = c->reps;
    perfcounter_config(COUNT_CYCLES, true);
    uint64_t start = perfcounter_get();
    for (uint32_t r = 0; r < reps; r++) {
        do_barrier();
    }
    uint64_t end = perfcounter_get();
    o->cycles_total = end - start;
    o->ops_done = reps;
}

static __attribute__((aligned(8))) uint32_t local_acc[NR_TASKLETS];

static void bench_reduce(uint32_t tid, const struct bench_cfg *c, struct bench_out *o) {
    const uint32_t reps = c->reps;

    // Each tasklet: accumulate locally in WRAM
    uint32_t acc = 0;
    perfcounter_config(COUNT_CYCLES, true);
    uint64_t start = perfcounter_get();
    for (uint32_t r = 0; r < reps; r++) {
        // simple arithmetic loop; mimics "work" without DMA
        acc += 1u;
    }
    local_acc[tid] = acc;

    // Synchronize then reduce on tasklet 0
    barrier_wait(&bar_sync);
    if (tid == 0) {
        uint32_t total = 0;
        for (uint32_t t = 0; t < NR_TASKLETS; t++) total += local_acc[t];
        (void)total;
    }
    uint64_t end = perfcounter_get();

    o->cycles_total = end - start;
    o->ops_done = reps;
}

int main() {
    const uint32_t tid = me();
    struct bench_cfg cfg = CFG;
    if (tid == 0) { /* zero outputs */ }
    do_barrier();

    switch (cfg.test_id) {
        case 1: bench_mram_read(tid, &cfg, &OUT[tid]);  break;
        case 2: bench_mram_write(tid, &cfg, &OUT[tid]); break;
        case 3: bench_wram_memcpy(tid, &cfg, &OUT[tid]);break;
        case 4: bench_barrier(tid, &cfg, &OUT[tid]);    break;
        case 5: bench_reduce(tid, &cfg, &OUT[tid]);     break;
        default: OUT[tid].cycles_total = 0; OUT[tid].ops_done = 0; break;
    }
    do_barrier();
    return 0;
}
