/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A FP16 dense GEMM example for the NVIDIA Blackwell SM100 architecture using CUTLASS.

    This example demonstrates minimal set of changes needed to transition from a Hopper CUTLASS 3.x 
    GEMM kernel (see example 48_hopper_warp_specialized_gemm) to a Blackwell 3.x CUTLASS GEMM kernel.
    
    The Blackwell SM100 CUTLASS kernel uses of the following Blackwell SM100 features:

    1. New series of Tensor Core MMA Instructions (tcgen05) introduced on the Blackwell architecture (sm100a) 
    which have 2x throughput compared to Hopper Tensor Core MMA instructions (WGMMA). 
    
    Note that Hopper WGMMA Tensor Core MMA instructions are not compatible on Blackwell (See https://docs.nvidia.com/cuda/parallel-thread-execution). 

    2. A new per-SM memory called Tensor Memory (TMEM) introduced on the Blackwell architecture (sm100a). 
    Blackwell SM100 Tensor Core MMA instructions store their accumulation results in TMEM instead of the 
    Register File. (Please refer to CUDA 12.8 docs on https://docs.nvidia.com/cuda/).

    3. An extended flavor of the warp-specialized kernel design introduced in Hopper enabled by use of TMEM 
    which allows us to decouple the execution of MMA and epilogue into separate warps. 

    4. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution). 

    Usage:
      $ ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"


#include "cuda_runtime.h"
#include <iostream>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        CUDA_CHECK(cudaEventDestroy(_start));
        CUDA_CHECK(cudaEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        CUDA_CHECK(cudaEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (D = AB + C)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::bfloat16_t;                            // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::bfloat16_t;                            // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = void;                                            // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                       // Layout type for C and D matrix operands
constexpr int AlignmentC  = 0;                                               // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                             // Element type for C and D matrix operands
using         LayoutD     = cutlass::layout::RowMajor;                       // Layout type for C and D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;     // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0 
using MmaTileShape_MNK = Shape<cute::_256,cute::_256,cute::_64>;                          
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<cute::_2,cute::_1,cute::_1>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, 
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, float, void, float>
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    // cutlass::gemm::collective::KernelScheduleAuto
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100
  >::CollectiveOp;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                   // Default to ClusterLaunchControl (CLC) based tile scheduler 

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle;

  Options():
    help(false),
    m(8192), n(8192), k(8192),
    alpha(1.f), beta(0.f),
    iterations(10),
    swizzle(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "70_blackwell_fp16_gemm\n\n"
      << "  Blackwell FP16 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "70_blackwell_fp16_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, nullptr, stride_C, block_D.get(), stride_D}
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;

  arguments.hw_info.cluster_shape = dim3(2, 1, 1);  // Your ClusterShape_MNK
  arguments.hw_info.cluster_shape_fallback = dim3(2, 1, 1);  // Fallback cluster

  int device_id = 0;
  cudaGetDevice(&device_id);
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
  arguments.hw_info.sm_count = sm_count;
  arguments.hw_info.device_id = device_id;

  return arguments;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
    initialize(options);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    auto arguments = args_from_options(options);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    Result result;

    // Run profiling loop
    for (int iter = 0; iter < 500; ++iter) {
        CUTLASS_CHECK(gemm.run());
    }
    cudaDeviceSynchronize();

    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < 100; ++iter) {
        CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(100);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);


    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
    Options options;
    options.parse(argc, args);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    run<Gemm>(options);
    return 0;
}
