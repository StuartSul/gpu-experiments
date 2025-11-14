/*

    Observation:
      - exp auto-translates to __expf with fast_math
      - Plain arithmetic takes <4 cycles
      - Slow arithmetic: division, exp, log
      - exp2, log2, fma, and fma2 are surprisingly fast
      - TMEM is faster to access than SMEM only slightly; like 15% faster
    
    Results (Clock rate: 120 MHz, 8.33 ns per cycle):
        Empty Loop #0: 2 cycles
        Empty Loop #1: 5 cycles
        Empty Loop #2: 5 cycles
        Empty Loop #3: 2 cycles
        Empty Loop Average: 3.50 cycles
        TMA Load #0: 2597 cycles
        TMA Load #1: 709 cycles
        TMA Load #2: 831 cycles
        TMA Load #3: 705 cycles
        TMA Load Average: 1210.50 cycles
        Float32 Addition #0: 4 cycles
        Float32 Addition #1: 5 cycles
        Float32 Addition #2: 4 cycles
        Float32 Addition #3: 4 cycles
        Float32 Addition Average: 4.25 cycles
        Float32 Subtraction #0: 4 cycles
        Float32 Subtraction #1: 5 cycles
        Float32 Subtraction #2: 4 cycles
        Float32 Subtraction #3: 4 cycles
        Float32 Subtraction Average: 4.25 cycles
        Float32 Multiplication #0: 4 cycles
        Float32 Multiplication #1: 5 cycles
        Float32 Multiplication #2: 4 cycles
        Float32 Multiplication #3: 4 cycles
        Float32 Multiplication Average: 4.25 cycles
        Float32 Division (Fast math ON) #0: 23 cycles
        Float32 Division (Fast math ON) #1: 23 cycles
        Float32 Division (Fast math ON) #2: 23 cycles
        Float32 Division (Fast math ON) #3: 23 cycles
        Float32 Division (Fast math ON) Average: 23.00 cycles
        Float32 __fdividef #0: 23 cycles
        Float32 __fdividef #1: 23 cycles
        Float32 __fdividef #2: 23 cycles
        Float32 __fdividef #3: 23 cycles
        Float32 __fdividef Average: 23.00 cycles
        Float32 exp #0: 8 cycles
        Float32 exp #1: 8 cycles
        Float32 exp #2: 10 cycles
        Float32 exp #3: 17 cycles
        Float32 exp Average: 10.75 cycles
        Float32 __expf #0: 8 cycles
        Float32 __expf #1: 8 cycles
        Float32 __expf #2: 10 cycles
        Float32 __expf #3: 17 cycles
        Float32 __expf Average: 10.75 cycles
        Float32 exp2f #0: 2 cycles
        Float32 exp2f #1: 5 cycles
        Float32 exp2f #2: 5 cycles
        Float32 exp2f #3: 6 cycles
        Float32 exp2f Average: 4.50 cycles
        Float32 __logf #0: 20 cycles
        Float32 __logf #1: 19 cycles
        Float32 __logf #2: 19 cycles
        Float32 __logf #3: 22 cycles
        Float32 __logf Average: 20.00 cycles
        Float32 __log2f #0: 3 cycles
        Float32 __log2f #1: 5 cycles
        Float32 __log2f #2: 5 cycles
        Float32 __log2f #3: 6 cycles
        Float32 __log2f Average: 4.75 cycles
        Float32 abs #0: 2 cycles
        Float32 abs #1: 5 cycles
        Float32 abs #2: 5 cycles
        Float32 abs #3: 2 cycles
        Float32 abs Average: 3.50 cycles
        Float32 max #0: 2 cycles
        Float32 max #1: 6 cycles
        Float32 max #2: 2 cycles
        Float32 max #3: 6 cycles
        Float32 max Average: 4.00 cycles
        Float32 min #0: 2 cycles
        Float32 min #1: 6 cycles
        Float32 min #2: 2 cycles
        Float32 min #3: 6 cycles
        Float32 min Average: 4.00 cycles
        Float32 __fmaf_rn #0: 4 cycles
        Float32 __fmaf_rn #1: 5 cycles
        Float32 __fmaf_rn #2: 4 cycles
        Float32 __fmaf_rn #3: 4 cycles
        Float32 __fmaf_rn Average: 4.25 cycles
        Vector Float32 __ffma2_rn #0: 4 cycles
        Vector Float32 __ffma2_rn #1: 4 cycles
        Vector Float32 __ffma2_rn #2: 6 cycles
        Vector Float32 __ffma2_rn #3: 4 cycles
        Vector Float32 __ffma2_rn Average: 4.50 cycles
        warpgroup::sync(1) #0: 20 cycles
        warpgroup::sync(1) #1: 8 cycles
        warpgroup::sync(1) #2: 10 cycles
        warpgroup::sync(1) #3: 10 cycles
        warpgroup::sync(1) Average: 12.00 cycles

        Warning: below numbers are inaccurate due to register spills

        SM_BF16 -> RM_BF16 #0: 75 cycles
        SM_BF16 -> RM_BF16 #1: 503 cycles
        SM_BF16 -> RM_BF16 #2: 485 cycles
        SM_BF16 -> RM_BF16 #3: 477 cycles
        SM_BF16 -> RM_BF16 Average: 385.00 cycles
        RM_bf16 -> SM_BF16 #0: 324 cycles
        RM_bf16 -> SM_BF16 #1: 332 cycles
        RM_bf16 -> SM_BF16 #2: 324 cycles
        RM_bf16 -> SM_BF16 #3: 324 cycles
        RM_bf16 -> SM_BF16 Average: 326.00 cycles
        TM_FL32 -> RM_FL32 #0: 282 cycles
        TM_FL32 -> RM_FL32 #1: 282 cycles
        TM_FL32 -> RM_FL32 #2: 282 cycles
        TM_FL32 -> RM_FL32 #3: 282 cycles
        TM_FL32 -> RM_FL32 Average: 282.00 cycles
        TM_FL32 -> RM_BF16 #0: 398 cycles
        TM_FL32 -> RM_BF16 #1: 355 cycles
        TM_FL32 -> RM_BF16 #2: 516 cycles
        TM_FL32 -> RM_BF16 #3: 505 cycles
        TM_FL32 -> RM_BF16 Average: 443.50 cycles
        TM_BF16 -> RM_BF16 #0: 154 cycles
        TM_BF16 -> RM_BF16 #1: 338 cycles
        TM_BF16 -> RM_BF16 #2: 353 cycles
        TM_BF16 -> RM_BF16 #3: 347 cycles
        TM_BF16 -> RM_BF16 Average: 298.00 cycles
        RM_FL32 -> TM_FL32 #0: 283 cycles
        RM_FL32 -> TM_FL32 #1: 283 cycles
        RM_FL32 -> TM_FL32 #2: 283 cycles
        RM_FL32 -> TM_FL32 #3: 283 cycles
        RM_FL32 -> TM_FL32 Average: 283.00 cycles
        RM_BF16 -> TM_BF16 #0: 197 cycles
        RM_BF16 -> TM_BF16 #1: 476 cycles
        RM_BF16 -> TM_BF16 #2: 470 cycles
        RM_BF16 -> TM_BF16 #3: 387 cycles
        RM_BF16 -> TM_BF16 Average: 382.50 cycles
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int STATIC_SHARED_MEMORY = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 56;
    static constexpr int CONSUMER_REGISTERS = 224;

    static constexpr int PIPELINE_STAGES = 3;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    using tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;

    gl<bf16, 1, 1, -1, -1, tile> A;

    __host__ inline dim3 grid() { return dim3(1); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

constexpr int NUM_ITERS = 4;

__device__ inline void print_time(const char *name, volatile uint64_t *start, volatile uint64_t *end) {
    double avg = 0.0;

    for (int i = 0; i < NUM_ITERS; i++) {
        if (threadIdx.x == 0) printf("%s #%d: %llu cycles\n", name, i, end[i] - start[i]);
        avg += static_cast<double>(end[i] - start[i]);
    }

    avg /= static_cast<double>(NUM_ITERS);
    if (threadIdx.x == 0) printf("%s Average: %.2f cycles\n", name, avg);
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Allocate shared and tensor memory
    globals::tile &sm_bf = allocator.allocate<globals::tile>();
    tensor_allocator<1, 1> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();

    __shared__ uint64_t start[NUM_ITERS];
    __shared__ uint64_t end[NUM_ITERS];

    if (threadIdx.x == 0) {
        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        print_time("Empty Loop", start, end);
        asm volatile ("" ::: "memory");

        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
            tma::load_async(sm_bf, G.A, {0, 0}, inputs_arrived);
            wait(inputs_arrived, i % 2);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        print_time("TMA Load", start, end);

        float val = 0.0;

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = val + operand;
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 Addition", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = val - operand;
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 Subtraction", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = val * operand;
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 Multiplication", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = val / operand;
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 Division (Fast math ON)", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = __fdividef(val, operand);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 __fdividef", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = exp(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 exp", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = __expf(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 __expf", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = exp2f(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 exp2f", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = __logf(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 __logf", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = __log2f(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 __log2f", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = fabsf(val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 abs", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = max(val, operand);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 max", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = min(val, operand);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 min", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float operand1 = __bfloat162float(G.A.raw_ptr[(i * 4) % 13]);
            float operand2 = __bfloat162float(G.A.raw_ptr[(i * 3) % 17]);
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            val = __fmaf_rn(operand1, operand2, val);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Float32 __fmaf_rn", start, end);
        asm volatile ("" ::: "memory");

        asm volatile ("" ::: "memory");
        for (int i = 0; true; i++) {
            if (i == NUM_ITERS)
                break;
            float2 result = {val, val};
            float2 operand1 = {__bfloat162float(G.A.raw_ptr[(i * 4) % 13]), __bfloat162float(G.A.raw_ptr[(i * 6) % 19])};
            float2 operand2 = {__bfloat162float(G.A.raw_ptr[(i * 3) % 17]), __bfloat162float(G.A.raw_ptr[(i * 5) % 11])};
            asm volatile ("" ::: "memory");
            uint64_t t0 = clock64();
            asm volatile ("" ::: "memory");
            result = __ffma2_rn(operand1, operand2, result);
            asm volatile ("" ::: "memory");
            uint64_t t1 = clock64();
            start[i] = t0;
            end[i] = t1;
            val = result.x + result.y;
        }
        G.A.raw_ptr[((int)val * 100) % 2] = __float2bfloat16(val); // prevent compiler optimization
        print_time("Vector Float32 __ffma2_rn", start, end);
        asm volatile ("" ::: "memory");
    }
    __syncthreads();

    auto tm_fl = tm_allocator.allocate<tt<float, globals::BLOCK_SIZE, globals::BLOCK_SIZE>>(0);
    auto tm_bf = tm_allocator.allocate<tt<bf16, globals::BLOCK_SIZE, globals::BLOCK_SIZE>>(0);

    /*
        WARNING: Benchmarks below inaccurate due to register spills
    */

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
    }
    print_time("warpgroup::sync(1)", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_bf<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_bf;
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::load(rm_bf, sm_bf);
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
        warp::store(G.A, rm_bf, {i * 2, i});
    }
    print_time("SM_BF16 -> RM_BF16", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_bf<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_bf;
        warp::one(rm_bf);
        warp::add(rm_bf, rm_bf, __float2bfloat16((float)i));
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::store(sm_bf, rm_bf);
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
        warp::store(G.A, sm_bf, {i * 2, i});
    }
    print_time("RM_bf16 -> SM_BF16", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_fl<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_fl;
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::load_async(rm_fl, tm_fl);
        tensor_load_wait();
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
        warp::store(G.A, rm_fl, {i * 2, i});
    }
    print_time("TM_FL32 -> RM_FL32", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_bf<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_bf;
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::load_async(rm_bf, tm_fl);
        tensor_load_wait();
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
        warp::store(G.A, rm_bf, {i * 2, i});
    }
    print_time("TM_FL32 -> RM_BF16", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_bf<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_bf;
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::load_async(rm_bf, tm_bf);
        tensor_load_wait();
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
        warp::store(G.A, rm_bf, {i * 2, i});
    }
    print_time("TM_BF16 -> RM_BF16", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_fl<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_fl;
        warp::one(rm_fl);
        warp::add(rm_fl, rm_fl, (float)i);
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::store_async(tm_fl, rm_fl);
        tensor_store_wait();
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
    }
    print_time("RM_FL32 -> TM_FL32", start, end);
    asm volatile ("" ::: "memory");

    asm volatile ("" ::: "memory");
    for (int i = 0; true; i++) {
        rt_bf<globals::BLOCK_SIZE / 4, globals::BLOCK_SIZE> rm_bf;
        warp::one(rm_bf);
        warp::add(rm_bf, rm_bf, __float2bfloat16((float)i));
        if (i == NUM_ITERS)
            break;
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t0 = clock64();
        asm volatile ("" ::: "memory");
        warpgroup::store_async(tm_bf, rm_bf);
        tensor_store_wait();
        warpgroup::sync(1);
        asm volatile ("" ::: "memory");
        uint64_t t1 = clock64();
        start[i] = t0;
        end[i] = t1;
    }
    print_time("RM_BF16 -> TM_BF16", start, end);
    asm volatile ("" ::: "memory");
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel", &globals::A);
}
