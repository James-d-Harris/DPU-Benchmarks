// bench_host.c
#define _POSIX_C_SOURCE 200809L
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

struct bench_cfg {
    uint32_t test_id, bytes, stride, reps, flags;
};
struct bench_out {
    uint64_t cycles_total;
    uint32_t ops_done;
    uint32_t pad;
};

#define MRAM_BUF_SIZE (8 * 1024 * 1024) // must match DPU side
static inline size_t align8_up(size_t x)   { return (x + 7) & ~(size_t)7; }
static inline size_t align8_down(size_t x) { return x & ~(size_t)7; }
static inline size_t clamp_to_buf(size_t x){ return x > MRAM_BUF_SIZE ? MRAM_BUF_SIZE : x; }


static void* xaligned(size_t align, size_t n) {
    void *p=NULL; if (posix_memalign(&p, align, n)) exit(2); return memset(p, 0xA5, n), p;
}

static void run_host_to_dpu(struct dpu_set_t dpus, size_t bytes, unsigned reps, FILE *csv) {
    size_t xfer = align8_down(clamp_to_buf(bytes));
    if (xfer < 8) xfer = 8;
    void *buf = xaligned(64, xfer);
    double t0 = now_ms();
    for (unsigned r=0;r<reps;r++) {
        DPU_ASSERT(dpu_broadcast_to(dpus, "MRAM_BUF", 0, buf, xfer, DPU_XFER_DEFAULT));
    }
    double t1 = now_ms();
    double total = (double)xfer*reps;
    fprintf(csv, "host_h2d,%zu,0,%u,%.6f,%.6f\n", xfer, reps, t1-t0, (total/1e6)/ (t1-t0));
    fflush(csv);
    free(buf);
}

// Count DPUs in a set
static unsigned count_dpus(struct dpu_set_t dpus) {
    struct dpu_set_t it; uint32_t idx; unsigned n = 0;
    DPU_FOREACH(dpus, it, idx) { (void)it; n++; }
    (void)idx;
    return n;
}

static void run_dpu_to_host(struct dpu_set_t dpus, size_t bytes, unsigned reps, FILE *csv) {
    size_t xfer = align8_down(clamp_to_buf(bytes));
    if (xfer < 8) xfer = 8;

    // one scratch buffer reused for each DPU
    void *buf = xaligned(64, xfer);

    // ensure MRAM has data before reading
    void *tmp = xaligned(64, xfer);
    memset(tmp, 0xA5, xfer);
    DPU_ASSERT(dpu_broadcast_to(dpus, "MRAM_BUF", 0, tmp, xfer, DPU_XFER_DEFAULT));
    free(tmp);

    const unsigned ndpus = count_dpus(dpus);

    double t0 = now_ms();
    for (unsigned r = 0; r < reps; r++) {
        struct dpu_set_t dpu; uint32_t each;
        DPU_FOREACH(dpus, dpu, each) {
            DPU_ASSERT(dpu_copy_from(dpu, "MRAM_BUF", 0, buf, xfer));
        } (void)each;
    }
    double t1 = now_ms();

    double total = (double)xfer * reps * ndpus;
    fprintf(csv, "host_d2h,%zu,0,%u,%.6f,%.6f\n", xfer, reps, t1 - t0, (total/1e6)/(t1 - t0));
    fflush(csv);

    free(buf);
}

// Prepare+Push Host -> DPU
static void run_prepare_push_h2d(struct dpu_set_t dpus, size_t bytes, unsigned reps, FILE *csv) {
    const unsigned ndpus = count_dpus(dpus);
    size_t xfer = align8_down(clamp_to_buf(bytes));
    if (xfer < 8) xfer = 8;

    void *buf = xaligned(64, xfer);

    // warm MRAM region once
    DPU_ASSERT(dpu_broadcast_to(dpus, "MRAM_BUF", 0, buf, xfer, DPU_XFER_DEFAULT));

    double t0 = now_ms();
    for (unsigned r = 0; r < reps; r++) {
        struct dpu_set_t dpu; uint32_t each;
        DPU_FOREACH(dpus, dpu, each) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, buf));
        } (void)each;
        DPU_ASSERT(dpu_push_xfer(dpus, DPU_XFER_TO_DPU, "MRAM_BUF", 0, xfer, DPU_XFER_DEFAULT));
    }
    double t1 = now_ms();

    double total = (double)xfer * reps * ndpus;
    fprintf(csv, "host_h2d_prepare_push,%zu,0,%u,%.6f,%.6f\n", xfer, reps, t1 - t0, (total/1e6)/(t1 - t0));
    fflush(csv);

    free(buf);
}


// Prepare+Push DPU -> Host
// replaces run_prepare_push_d2h
static void run_prepare_push_d2h(struct dpu_set_t dpus, size_t bytes, unsigned reps, FILE *csv) {
    const unsigned ndpus = count_dpus(dpus);
    size_t xfer = align8_down(clamp_to_buf(bytes));
    if (xfer < 8) xfer = 8;

    // batch size to cap memory; tune with --batch later if you want
    const unsigned BATCH = 64; // 64 * xfer bytes of RAM max

    // seed MRAM with data so reads are defined
    void *seed = xaligned(64, xfer); memset(seed, 0x5A, xfer);
    DPU_ASSERT(dpu_broadcast_to(dpus, "MRAM_BUF", 0, seed, xfer, DPU_XFER_DEFAULT));
    free(seed);

    double t0 = now_ms();
    for (unsigned r = 0; r < reps; r++) {
        // walk the DPU set in windows of size BATCH
        unsigned idx = 0;
        struct dpu_set_t dpu; uint32_t each;
        void **pool = NULL; unsigned pool_sz = 0;
\
        unsigned total = ndpus;

        // reset iterator
        DPU_FOREACH(dpus, dpu, each) {
            if (pool == NULL) {}
            (void)each;
        }

        // iterate again, actually process in chunks
        unsigned processed = 0;
        while (processed < total) {
            // allocate/reuse pool for a batch
            if (!pool) {
                pool_sz = (total - processed < BATCH) ? (total - processed) : BATCH;
                pool = (void **)calloc(pool_sz, sizeof(void *));
                for (unsigned i = 0; i < pool_sz; i++) pool[i] = xaligned(64, xfer);
            }

            // Prepare for this batch
            unsigned k = 0;
            DPU_FOREACH(dpus, dpu, each) {
                if (k < pool_sz && idx >= processed && idx < processed + pool_sz) {
                    DPU_ASSERT(dpu_prepare_xfer(dpu, pool[k++]));
                }
                idx++;
            } (void)each;

            // Push for this batch
            DPU_ASSERT(dpu_push_xfer(dpus, DPU_XFER_FROM_DPU, "MRAM_BUF", 0, xfer, DPU_XFER_DEFAULT));

            // Reset idx for next batch
            idx = 0;
            processed += pool_sz;

            // free batch buffers
            for (unsigned i = 0; i < pool_sz; i++) free(pool[i]);
            free(pool); pool = NULL; pool_sz = 0;
        }
    }
    double t1 = now_ms();

    double total_bytes = (double)xfer * reps * ndpus;
    fprintf(csv, "host_d2h_prepare_push,%zu,0,%u,%.6f,%.6f\n", xfer, reps, t1 - t0, (total_bytes/1e6)/(t1 - t0));
    fflush(csv);
}


static void run_on_dpu(struct dpu_set_t dpus, uint32_t test_id, size_t bytes, uint32_t stride, uint32_t reps, FILE *csv) {
    struct dpu_set_t dpu; uint32_t each;
    DPU_FOREACH(dpus, dpu, each) {
        struct bench_cfg cfg = { test_id, (uint32_t)bytes, stride, reps, 0 };
        DPU_ASSERT(dpu_copy_to(dpu, "CFG", 0, &cfg, sizeof(cfg)));
    } (void)each;

    // Warm up MRAM region if needed
    {
        size_t need = bytes;
        if (stride) {
            need = (size_t)NR_TASKLETS * stride + (size_t)stride * (reps ? reps - 1 : 0);
        } else {
            need = (size_t)NR_TASKLETS * (bytes ? bytes : 8) * (reps ? reps : 1);
        }
        size_t xfer = align8_up(need);
        xfer = clamp_to_buf(xfer);
        if (xfer < 8) xfer = 8;
    
        void *tmp = xaligned(64, xfer);
        DPU_ASSERT(dpu_broadcast_to(dpus, "MRAM_BUF", 0, tmp, xfer, DPU_XFER_DEFAULT));
        free(tmp);
    }

    double t0 = now_ms();
    DPU_ASSERT(dpu_launch(dpus, DPU_SYNCHRONOUS));
    double t1 = now_ms();

    struct bench_out outs[NR_TASKLETS];
    double sum_cycles = 0.0, sum_ops = 0.0;
    DPU_FOREACH(dpus, dpu, each) {
        DPU_ASSERT(dpu_copy_from(dpu, "OUT", 0, outs, sizeof(outs)));
        for (int t=0;t<NR_TASKLETS;t++){ sum_cycles += (double)outs[t].cycles_total; sum_ops += (double)outs[t].ops_done; }
    } (void)each;

    fprintf(csv, "dpu_test_%u,%zu,%u,%u,%.6f,%.0f,%.0f\n",
            test_id, bytes, stride, reps, t1-t0, sum_cycles, sum_ops);
    fflush(csv);

}

int main(int argc, char **argv) {
    size_t min_bytes = 64, max_bytes = 1<<26; // 64B..64MB
    unsigned factor = 2, reps = 64;
    unsigned which = 0xFFFFFFFFu; // all
    unsigned nr_dpus = 0; // auto

    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--min") && i+1<argc) min_bytes = strtoull(argv[++i],NULL,10);
        else if (!strcmp(argv[i],"--max") && i+1<argc) max_bytes = strtoull(argv[++i],NULL,10);
        else if (!strcmp(argv[i],"--factor") && i+1<argc) factor = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-r") && i+1<argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--dpus") && i+1<argc) nr_dpus = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--tests") && i+1<argc) which = strtoul(argv[++i],NULL,0);
    }

    struct dpu_set_t dpus;
    if (nr_dpus==0) { DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &dpus)); }
    else            { DPU_ASSERT(dpu_alloc(nr_dpus, NULL, &dpus)); }

    DPU_ASSERT(dpu_load(dpus, "./bench_dpu", NULL));

    FILE *csv = fopen("bench_results.csv","w");
    fprintf(csv, "name,bytes,stride,reps,host_ms,sum_cycles,sum_ops\n");
    fflush(csv);

    int i = 0;

    fprintf(stderr, "Starting for loop\n");
    for (size_t b=min_bytes; b<=max_bytes; b*=factor) {
        fprintf(stderr, "Starting loop number %u with b = %lu\n", i, b);
        if (which == 0xFFFFFFFFu || (which & 0x1))   run_host_to_dpu(dpus, b, reps, csv);
        fprintf(stderr, "Past run_host_to_dpu\n");
        if (which == 0xFFFFFFFFu || (which & 0x2))   run_dpu_to_host(dpus, b, reps, csv);
        fprintf(stderr, "Past run_dpu_to_host\n");
        if (which == 0xFFFFFFFFu || (which & 0x80))  run_prepare_push_h2d(dpus, b, reps, csv);
        fprintf(stderr, "Past run_prepare_push_h2d\n");
        if (which == 0xFFFFFFFFu || (which & 0x100)) run_prepare_push_d2h(dpus, b, reps, csv);
        fprintf(stderr, "Past run_prepare_push_d2h\n");
        if (which == 0xFFFFFFFFu || (which & 0x4))   run_on_dpu(dpus, 1, b, 0, reps, csv);
        fprintf(stderr, "Past run_on_dpu1\n");
        if (which == 0xFFFFFFFFu || (which & 0x8))   run_on_dpu(dpus, 2, b, 0, reps, csv);
        fprintf(stderr, "Past run_on_dpu2\n");
        if (which == 0xFFFFFFFFu || (which & 0x10))  run_on_dpu(dpus, 3, b, 0, reps, csv);
        fprintf(stderr, "Past run_on_dpu3\n");
        if (which == 0xFFFFFFFFu || (which & 0x20))  run_on_dpu(dpus, 4, 0, 0, reps*8, csv);
        fprintf(stderr, "Past run_on_dpu4\n");
        if (which == 0xFFFFFFFFu || (which & 0x40))  run_on_dpu(dpus, 5, 0, 0, reps*1024, csv);
        fprintf(stderr, "Past run_on_dpu5\n");
        i++;
        fprintf(stderr, "Loop %u finished\n", i);
    }
    fprintf(stderr, "For loop finished\n");

    fclose(csv);
    DPU_ASSERT(dpu_free(dpus));
    return 0;
}
