#include <cuda.h>
#include <nvrtc.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
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
