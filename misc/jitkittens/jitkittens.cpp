#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define CHECK_NVRTC(call)                                                \
    do {                                                                 \
        const nvrtcResult nvrtc_result = (call);                         \
        if (nvrtc_result != NVRTC_SUCCESS)                               \
            throw std::runtime_error(nvrtcGetErrorString(nvrtc_result)); \
    } while (0)

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        const CUresult cuda_result = (call);                                  \
        if (cuda_result != CUDA_SUCCESS) {                                    \
            const char *cuda_error = nullptr;                                 \
            cuGetErrorString(cuda_result, &cuda_error);                       \
            throw std::runtime_error(cuda_error ? cuda_error : "CUDA error"); \
        }                                                                     \
    } while (0)

struct NvrtcResult {
    std::vector<char> cubin;
    std::vector<std::string> mangled_names;
};

NvrtcResult compile_source_to_cubin(const std::string &source,
                                               const std::vector<std::string> &kernel_symbols, 
                                               int major, int minor,
                                               const std::vector<std::string> &include_directories,
                                               const std::vector<std::string> &additional_nvrtc_options,
                                               bool verbose = true) {
    // 1. Create the NVRTC program.
    nvrtcProgram nvrtc_program = nullptr;
    CHECK_NVRTC(nvrtcCreateProgram(&nvrtc_program, source.c_str(), "kernel.cu", 0, nullptr, nullptr));

    // 2. Register kernel symbol expressions.
    for (const std::string &kernel_symbol : kernel_symbols)
        CHECK_NVRTC(nvrtcAddNameExpression(nvrtc_program, kernel_symbol.c_str()));

    // 3. Prepare compiler flags and compile.
    std::vector<std::string> options = {"-DNDEBUG", "-lineinfo", "--std=c++20", "--use_fast_math", 
                                        "-Xptxas=--verbose", "-Xptxas=--warn-on-spills"};
    for (const std::string &include_directory : include_directories) options.push_back("-I" + include_directory);
    options.insert(options.end(), additional_nvrtc_options.begin(), additional_nvrtc_options.end());
    options.push_back("--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor) + (major >= 9 ? "a" : ""));

    std::vector<const char *> option_pointers;
    option_pointers.reserve(options.size());
    for (const std::string &option : options) option_pointers.push_back(option.c_str());
    const nvrtcResult compile_result = nvrtcCompileProgram(nvrtc_program, static_cast<int>(option_pointers.size()), option_pointers.data());

    // 4. Print the compiler log and check the compilation result.
    std::size_t log_size = 0;
    CHECK_NVRTC(nvrtcGetProgramLogSize(nvrtc_program, &log_size));
    std::string log(log_size, '\0');
    if (log_size != 0U) CHECK_NVRTC(nvrtcGetProgramLog(nvrtc_program, log.data()));
    if (log_size > 1U && (verbose || compile_result != NVRTC_SUCCESS)) {
        std::cout << log.c_str();
        if (log[log_size - 2U] != '\n') std::cout << '\n';
    }
    CHECK_NVRTC(compile_result);

    // 5. Copy the lowered (mangled) kernel names.
    NvrtcResult result;
    result.mangled_names.reserve(kernel_symbols.size());
    for (const std::string &kernel_symbol : kernel_symbols) {
        const char *lowered_name = nullptr;
        CHECK_NVRTC(nvrtcGetLoweredName(nvrtc_program, kernel_symbol.c_str(), &lowered_name));
        result.mangled_names.emplace_back(lowered_name);
    }

    // 6. Retrieve the compiled CUBIN.
    std::size_t cubin_size = 0;
    CHECK_NVRTC(nvrtcGetCUBINSize(nvrtc_program, &cubin_size));
    if (cubin_size == 0U)
        throw std::runtime_error("NVRTC returned no CUBIN");
    result.cubin.resize(cubin_size);
    CHECK_NVRTC(nvrtcGetCUBIN(nvrtc_program, result.cubin.data()));

    // 7. Destroy the NVRTC program.
    CHECK_NVRTC(nvrtcDestroyProgram(&nvrtc_program));

    return result;
}

std::vector<std::string> cuda_include_dirs() {
    const auto check_cuda_root = [](const std::filesystem::path &root) -> std::optional<std::filesystem::path> {
        const std::filesystem::path include = root / "include";
        if (std::filesystem::exists(include / "cuda_bf16.h")) return include;
        else return std::nullopt;
    };

    std::optional<std::filesystem::path> cuda_include;
    for (const char *env_var : {"CUDA_HOME", "CUDA_PATH"}) {
        const char *value = std::getenv(env_var);
        if (value != nullptr && *value != '\0') {
            cuda_include = check_cuda_root(value);
            if (cuda_include) break;
        }
    }
    for (const char *env_var : {"PATH", "LD_LIBRARY_PATH"}) {
        if (cuda_include) break;
        const char *value = std::getenv(env_var);
        std::string_view remaining = value == nullptr ? std::string_view{} : std::string_view{value};
        while (!remaining.empty()) {
            const std::size_t separator = remaining.find(':');
            const std::string_view entry = remaining.substr(0, separator);
            if (!entry.empty()) {
                cuda_include = check_cuda_root(std::filesystem::path(entry).parent_path());
                if (cuda_include) break;
            }
            if (separator == std::string_view::npos) break;
            remaining.remove_prefix(separator + 1);
        }
    }
    for (const std::filesystem::path &root : {std::filesystem::path("/usr/local/cuda"), std::filesystem::path("/usr/cuda")}) {
        if (cuda_include) break;
        cuda_include = check_cuda_root(root);
    }
    if (!cuda_include)
        throw std::runtime_error("Cannot find CUDA include directory.");

    const std::filesystem::path cccl = *cuda_include / "cccl";
    if (std::filesystem::exists(cccl)) // CUDA 13
        return {cuda_include->string(), cccl.string(), (cccl / "cuda" / "std").string()};
    else // CUDA 12
        return {cuda_include->string(), (*cuda_include / "cuda" / "std").string()};
}

CUmodule load_cubin_module(const std::vector<char> &cubin) {
    CUmodule module = nullptr;
    CHECK_CUDA(cuModuleLoadData(&module, cubin.data()));
    return module;
}

CUfunction get_kernel_from_cubin_module(CUmodule module, const std::string &kernel_name) {
    CUfunction function = nullptr;
    CHECK_CUDA(cuModuleGetFunction(&function, module, kernel_name.c_str()));
    return function;
}

void unload_cubin_module(CUmodule module) {
    CHECK_CUDA(cuModuleUnload(module));
}

void set_kernel_dynamic_smem(CUfunction function, int dynamic_smem_bytes) {
    CHECK_CUDA(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_smem_bytes));
}

CUtensorMap create_tma_descriptor(CUtensorMapDataType data_type, void *global_address,
                                  const std::vector<cuuint64_t> &gmem_shape, const std::vector<cuuint32_t> &smem_shape, 
                                  cuuint32_t swizzle_bytes, int swizzle_axis) {
    if (global_address == nullptr || (reinterpret_cast<std::uintptr_t>(global_address) & 0xf) != 0)
        throw std::invalid_argument("TMA global address must be 16-byte aligned");
    if (gmem_shape.size() != 4 || (smem_shape.size() != 1 && smem_shape.size() != 2))
        throw std::invalid_argument("TMA shapes must be 4D global and 1D or 2D shared");
    for (cuuint64_t dimension : gmem_shape)
        if (dimension == 0) throw std::invalid_argument("TMA global dimensions must be nonzero");
    for (cuuint32_t dimension : smem_shape)
        if (dimension == 0) throw std::invalid_argument("TMA shared dimensions must be nonzero");

    cuuint32_t dtype_bytes = 0;
    switch (data_type) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8:
            dtype_bytes = 1;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_UINT16:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
            dtype_bytes = 2;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_UINT32:
        case CU_TENSOR_MAP_DATA_TYPE_INT32:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
            dtype_bytes = 4;
            break;
        case CU_TENSOR_MAP_DATA_TYPE_UINT64:
        case CU_TENSOR_MAP_DATA_TYPE_INT64:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
            dtype_bytes = 8;
            break;
        default:
            throw std::invalid_argument("Packed TMA data types are unsupported; use their packed storage type");
    }

    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    switch (swizzle_bytes) {
    case 0:
        break;
    case 32:
        swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
        break;
    case 64:
        swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
        break;
    case 128:
        swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        break;
    default:
        throw std::invalid_argument("TMA swizzle must be 0, 32, 64, or 128 bytes");
    }

    std::array<cuuint64_t, 5> encoded_gmem_shape{};
    std::array<cuuint64_t, 4> encoded_gmem_strides{};
    std::array<cuuint32_t, 5> encoded_smem_shape{1, 1, 1, 1, 1};
    std::array<cuuint32_t, 5> encoded_smem_strides{1, 1, 1, 1, 1};
    cuuint32_t rank = 0;

    if (smem_shape.size() == 1) {
        if (swizzle_bytes != 0 || swizzle_axis != -1)
            throw std::invalid_argument("Vector TMA requires no swizzle and axis -1");
        const cuuint32_t length = smem_shape[0];
        if (length % 16 != 0 || (length > 256 && (static_cast<cuuint64_t>(length) * dtype_bytes) % 128 != 0))
            throw std::invalid_argument("Invalid vector length for TMA");
        cuuint32_t inner_length = 16;
        for (int divider = 16; divider >= 2; --divider) {
            const cuuint32_t candidate = 16 * divider;
            if (length % candidate == 0 && (length < 256 || (static_cast<cuuint64_t>(candidate) * dtype_bytes) % 128 == 0)) {
                inner_length = candidate;
                break;
            }
        }
        rank = 4;
        encoded_gmem_shape = {gmem_shape[3], gmem_shape[2], gmem_shape[1], gmem_shape[0], 0};
        encoded_gmem_strides = {gmem_shape[3] * dtype_bytes, gmem_shape[2] * gmem_shape[3] * dtype_bytes,
                                gmem_shape[1] * gmem_shape[2] * gmem_shape[3] * dtype_bytes, 0};
        encoded_smem_shape = {inner_length, 1, 1, 1, 1};
    } else if (swizzle_bytes == 0) {
        if (swizzle_axis != 2) throw std::invalid_argument("Non-swizzled tile TMA requires axis 2");
        rank = 4;
        encoded_gmem_shape = {gmem_shape[3], gmem_shape[2], gmem_shape[1], gmem_shape[0], 0};
        encoded_gmem_strides = {gmem_shape[3] * dtype_bytes, gmem_shape[2] * gmem_shape[3] * dtype_bytes,
                                gmem_shape[1] * gmem_shape[2] * gmem_shape[3] * dtype_bytes, 0};
        encoded_smem_shape = {smem_shape[1], smem_shape[0], 1, 1, 1};
    } else {
        if (swizzle_axis < 0 || swizzle_axis > 2 || swizzle_bytes % dtype_bytes != 0)
            throw std::invalid_argument("Invalid tile TMA swizzle");
        const cuuint32_t swizzle_elements = swizzle_bytes / dtype_bytes;
        if (smem_shape[1] % swizzle_elements != 0)
            throw std::invalid_argument("Shared tile width must be divisible by the swizzle width");
        rank = 5;
        if (swizzle_axis == 2) {
            encoded_gmem_shape = {swizzle_elements, gmem_shape[2], (gmem_shape[3] + swizzle_elements - 1) / swizzle_elements, gmem_shape[1], gmem_shape[0]};
            encoded_gmem_strides = {gmem_shape[3] * dtype_bytes, swizzle_bytes, gmem_shape[2] * gmem_shape[3] * dtype_bytes,
                                    gmem_shape[1] * gmem_shape[2] * gmem_shape[3] * dtype_bytes};
        } else if (swizzle_axis == 1) {
            encoded_gmem_shape = {swizzle_elements, gmem_shape[1], (gmem_shape[3] + swizzle_elements - 1) / swizzle_elements, gmem_shape[2], gmem_shape[0]};
            encoded_gmem_strides = {gmem_shape[2] * gmem_shape[3] * dtype_bytes, swizzle_bytes, gmem_shape[3] * dtype_bytes,
                                    gmem_shape[1] * gmem_shape[2] * gmem_shape[3] * dtype_bytes};
        } else {
            encoded_gmem_shape = {swizzle_elements, gmem_shape[0], (gmem_shape[3] + swizzle_elements - 1) / swizzle_elements, gmem_shape[2], gmem_shape[1]};
            encoded_gmem_strides = {gmem_shape[1] * gmem_shape[2] * gmem_shape[3] * dtype_bytes, swizzle_bytes,
                                    gmem_shape[3] * dtype_bytes, gmem_shape[2] * gmem_shape[3] * dtype_bytes};
        }
        encoded_smem_shape = {swizzle_elements, smem_shape[0], smem_shape[1] / swizzle_elements, 1, 1};
    }

    for (cuuint32_t i = 0; i + 1 < rank; ++i)
        if (encoded_gmem_strides[i] % 16 != 0) throw std::invalid_argument("TMA global strides must be 16-byte aligned");
    for (cuuint32_t i = 0; i < rank; ++i)
        if (encoded_smem_shape[i] == 0 || encoded_smem_shape[i] > 256) throw std::invalid_argument("TMA shared dimensions must be between 1 and 256");
    if ((static_cast<cuuint64_t>(encoded_smem_shape[0]) * dtype_bytes) % 16 != 0)
        throw std::invalid_argument("TMA innermost shared dimension must span a multiple of 16 bytes");

    CUtensorMap descriptor{};
    CHECK_CUDA(cuTensorMapEncodeTiled(&descriptor, data_type, rank, global_address, encoded_gmem_shape.data(), encoded_gmem_strides.data(), 
                                      encoded_smem_shape.data(), encoded_smem_strides.data(), 
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return descriptor;
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

class KernelArgs {
public:
    explicit KernelArgs(std::vector<std::vector<CUtensorMap>> values) // CUtensormap used for 128B alignment
        : arguments(std::move(values)), argument_pointers(arguments.size()) {
        for (std::size_t i = 0; i < arguments.size(); ++i)
            argument_pointers[i] = arguments[i].data();
    }

    KernelArgs(const KernelArgs &) = delete;
    KernelArgs &operator=(const KernelArgs &) = delete;
    KernelArgs(KernelArgs &&) = default;
    KernelArgs &operator=(KernelArgs &&) = default;

    std::uintptr_t data_ptr() const {
        return reinterpret_cast<std::uintptr_t>(argument_pointers.data());
    }

    std::size_t size() const {
        return argument_pointers.size();
    }

private:
    std::vector<std::vector<CUtensorMap>> arguments;
    std::vector<void *> argument_pointers;
};

std::vector<CUtensorMap> create_gl_argument(CUtensorMapDataType data_type, std::uintptr_t global_address,
                                            const std::vector<cuuint64_t> &runtime_shape,
                                            const std::vector<int> &compile_shape,
                                            const std::vector<std::vector<cuuint32_t>> &tma_shapes,
                                            const std::vector<cuuint32_t> &swizzle_bytes,
                                            const std::vector<int> &swizzle_axes) {
    if (runtime_shape.size() != 4 || compile_shape.size() != 4)
        throw std::invalid_argument("GL runtime and compile shapes must contain four values");
    if (tma_shapes.size() != swizzle_bytes.size() || tma_shapes.size() != swizzle_axes.size())
        throw std::invalid_argument("GL TMA descriptor fields must have equal lengths");

    std::array<std::size_t, 4> dimension_offsets{};
    std::size_t field_offset = sizeof(std::uintptr_t); // to store the global addr
    for (std::size_t i = 0; i < 4; ++i) {
        if (compile_shape[i] == -1) {
            field_offset = align_up(field_offset, alignof(std::size_t));
            dimension_offsets[i] = field_offset;
            field_offset += sizeof(std::size_t);
        } else {
            if (compile_shape[i] <= 0 || static_cast<cuuint64_t>(compile_shape[i]) != runtime_shape[i])
                throw std::invalid_argument("Compile-time GL dimension does not match runtime shape");
            field_offset += 1;
        }
    }

    const std::size_t descriptor_count = tma_shapes.size();
    const std::size_t descriptor_base = descriptor_count == 0 ? 0 : align_up(field_offset, alignof(CUtensorMap));
    const std::size_t argument_size = descriptor_count == 0 ? align_up(field_offset + 1, alignof(std::uintptr_t))
                                                            : descriptor_base + (descriptor_count + 1) * sizeof(CUtensorMap);

    std::vector<CUtensorMap> argument((argument_size + sizeof(CUtensorMap) - 1) / sizeof(CUtensorMap));
    std::byte *packed = reinterpret_cast<std::byte *>(argument.data());
    std::memcpy(packed, &global_address, sizeof(global_address));
    for (std::size_t i = 0; i < 4; ++i) {
        if (compile_shape[i] == -1)
            std::memcpy(packed + dimension_offsets[i], &runtime_shape[i], sizeof(runtime_shape[i]));
    }

    for (std::size_t i = 0; i < descriptor_count; ++i) {
        CUtensorMap descriptor = create_tma_descriptor(data_type, reinterpret_cast<void *>(global_address),
                                                       runtime_shape, tma_shapes[i], swizzle_bytes[i], swizzle_axes[i]);
        std::memcpy(packed + descriptor_base + i * sizeof(CUtensorMap), &descriptor, sizeof(descriptor));
    }
    return argument;
}

KernelArgs create_gl_arguments(const std::vector<CUtensorMapDataType> &data_types,
                               const std::vector<std::uintptr_t> &global_addresses,
                               const std::vector<std::vector<cuuint64_t>> &runtime_shapes,
                               const std::vector<std::vector<int>> &compile_shapes,
                               const std::vector<std::vector<std::vector<cuuint32_t>>> &tma_shapes,
                               const std::vector<std::vector<cuuint32_t>> &swizzle_bytes,
                               const std::vector<std::vector<int>> &swizzle_axes) {
    const std::size_t count = global_addresses.size();
    if (count == 0 || data_types.size() != count || runtime_shapes.size() != count ||
        compile_shapes.size() != count || tma_shapes.size() != count ||
        swizzle_bytes.size() != count || swizzle_axes.size() != count)
        throw std::invalid_argument("GL argument fields must have equal nonzero lengths");

    std::vector<std::vector<CUtensorMap>> arguments;
    arguments.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        arguments.push_back(create_gl_argument(data_types[i], global_addresses[i], runtime_shapes[i],
                                               compile_shapes[i], tma_shapes[i], swizzle_bytes[i], swizzle_axes[i]));
    return KernelArgs(std::move(arguments));
}

CUlaunchConfig create_launch_config(dim3 grid, dim3 block, unsigned int dynamic_smem_bytes, CUstream stream,
                                    std::vector<CUlaunchAttribute> &attributes,
                                    std::optional<dim3> cluster = std::nullopt, bool pdl = false) {
    CUlaunchConfig config{
        .gridDimX = grid.x,
        .gridDimY = grid.y,
        .gridDimZ = grid.z,
        .blockDimX = block.x,
        .blockDimY = block.y,
        .blockDimZ = block.z,
        .sharedMemBytes = dynamic_smem_bytes,
        .hStream = stream,
        .attrs = nullptr,
        .numAttrs = 0,
    };
    attributes.clear();
    attributes.reserve(3);

    if (cluster) {
        CUlaunchAttribute preferred_cluster{};
        preferred_cluster.id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
        preferred_cluster.value.preferredClusterDim.x = cluster->x;
        preferred_cluster.value.preferredClusterDim.y = cluster->y;
        preferred_cluster.value.preferredClusterDim.z = cluster->z;
        attributes.push_back(preferred_cluster);

        CUlaunchAttribute required_cluster{};
        required_cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        required_cluster.value.clusterDim.x = cluster->x;
        required_cluster.value.clusterDim.y = cluster->y;
        required_cluster.value.clusterDim.z = cluster->z;
        attributes.push_back(required_cluster);
    }

    if (pdl) {
        CUlaunchAttribute pdl_attribute{};
        pdl_attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        pdl_attribute.value.programmaticStreamSerializationAllowed = 1;
        attributes.push_back(pdl_attribute);
    }

    config.attrs = attributes.empty() ? nullptr : attributes.data();
    config.numAttrs = static_cast<unsigned int>(attributes.size());
    return config;
}

void launch_kernel(const CUlaunchConfig &config, const CUfunction function, void **args) {
    CHECK_CUDA(cuLaunchKernelEx(&config, function, args, nullptr));
}
