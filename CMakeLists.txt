cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0119)
    cmake_policy(SET CMP0119 NEW)
endif()
project(tiny-llm
    VERSION 2.0.2
    DESCRIPTION "Lightweight LLM inference engine with W8A16 quantization"
    LANGUAGES CXX
)

# check_language / enable_language 的编译器检测会把 CMAKE_CUDA_ARCHITECTURES
# 填成 nvcc 默认（52），必须在此之前捕获用户的显式指定（-D / toolchain），见
# .agents/notes/implemented/bug-fix/2026-11-05-cuda-arch-default-never-applies.md
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_TLLM_USER_CUDA_ARCH TRUE)
else()
    set(_TLLM_USER_CUDA_ARCH FALSE)
endif()

include(CheckLanguage)
include(FetchContent)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA toolkit with nvcc is required to build tiny-llm. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH.")
endif()
enable_language(CUDA)

# ── C++ / CUDA 标准 ──────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── 外部依赖 ─────────────────────────────────────────────────────

# spdlog (logging library)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

# ── CUDA 配置 ────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

if(NOT _TLLM_USER_CUDA_ARCH AND NOT DEFINED ENV{CUDAARCHS})
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
    endif()
endif()

# ── 构建类型 ─────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── 主库 ─────────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES "src/*.cpp" "kernels/*.cu")
# Exclude main.cpp / benchmark.cpp from library (they are standalone executables)
list(FILTER SOURCES EXCLUDE REGEX "src/main\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "src/benchmark\\.cpp$")
# 显式追加（GLOB 在增量构建时不会发现新文件；保留原 GLOB 不动，见 NOTE）
list(APPEND SOURCES src/execution_common.cpp)
list(APPEND SOURCES kernels/transpose_weights.cu)
list(APPEND SOURCES kernels/paged_kv.cu)
list(APPEND SOURCES kernels/sampling.cu)

add_library(tiny_llm STATIC ${SOURCES})
add_library(tiny_llm::tiny_llm ALIAS tiny_llm)

target_include_directories(tiny_llm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_link_libraries(tiny_llm PUBLIC CUDA::cudart spdlog::spdlog)

target_compile_options(tiny_llm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# ── 主程序 ───────────────────────────────────────────────────────
add_executable(tiny_llm_demo src/main.cpp)
target_include_directories(tiny_llm_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(tiny_llm_demo PRIVATE tiny_llm)

# ── benchmark ────────────────────────────────────────────────────
add_executable(tiny_llm_bench src/benchmark.cpp)
target_include_directories(tiny_llm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(tiny_llm_bench PRIVATE tiny_llm)

# ── kernel microbenchmark ────────────────────────────────────────
# Decode-path kernel-level timing (replaces ncu/nsys on hosts where those
# profilers are unavailable). See docs/performance/benchmark-methodology.md.
add_executable(tiny_llm_kernel_bench src/kernel_bench.cpp)
target_include_directories(tiny_llm_kernel_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(tiny_llm_kernel_bench PRIVATE tiny_llm)

# ── 测试 ─────────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
    add_executable(tiny_llm_tests ${TEST_SOURCES})
    # These test files use CUDA runtime headers from host-side GTest code only.
    # Compile them as C++ to avoid nvcc/GCC host-compiler incompatibilities.
    set(TEST_HOST_SOURCES
        tests/test_inference_engine.cu
        tests/test_integration.cu
        tests/test_kernels.cu
        tests/test_transformer.cu
        tests/test_w8a16_matmul.cu
    )
    set_source_files_properties(${TEST_HOST_SOURCES} PROPERTIES LANGUAGE CXX)
    target_link_libraries(tiny_llm_tests PRIVATE
        tiny_llm
        GTest::gtest_main
        CUDA::cudart
    )
    target_include_directories(tiny_llm_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_compile_options(tiny_llm_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    include(GoogleTest)
    gtest_discover_tests(tiny_llm_tests)
endif()
