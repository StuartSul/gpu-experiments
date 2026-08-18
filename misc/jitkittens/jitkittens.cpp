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
