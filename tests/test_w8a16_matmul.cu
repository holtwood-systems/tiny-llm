#include "tiny_llm/cuda_utils.h"
#include "transpose_weights.cuh"
#include "w8a16_matmul.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
// #include <rapidcheck.h>
// NOTE: rapidcheck/gtest is disabled in .cu tests due to GCC 11/12 + nvcc
// std::function compatibility issues during CI builds.
// #include <rapidcheck/gtest.h>
#include <vector>

using namespace tiny_llm;
using namespace tiny_llm::kernels;

// Helper to check if CUDA device is available
static bool hasCudaDevice() {
    static bool checked = false;
    static bool has_device = false;
    if (!checked) {
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        has_device = (err == cudaSuccess && device_count > 0);
        checked = true;
    }
    return has_device;
}

// Helper class for GPU test fixtures
class W8A16MatMulTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int         device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        cudaSetDevice(0);
    }

    void TearDown() override { cudaDeviceSynchronize(); }

    // Generate random FP16 matrix
    static std::vector<half> randomFP16(int rows, int cols, float scale = 1.0f) {
        std::vector<half>                     data(rows * cols);
        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Generate random INT8 weights
    static std::vector<int8_t> randomINT8(int rows, int cols) {
        std::vector<int8_t>                data(rows * cols);
        std::mt19937                       gen(123);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto &v : data) {
            v = static_cast<int8_t>(dist(gen));
        }
        return data;
    }

    // Generate random scales
    static std::vector<half> randomScales(int num_groups, int cols) {
        std::vector<half>                     data(num_groups * cols);
        std::mt19937                          gen(456);
        std::uniform_real_distribution<float> dist(0.001f, 0.1f);
        for (auto &v : data) {
            v = __float2half(dist(gen));
        }
        return data;
    }

    // Compute relative error
    float computeRelativeError(const std::vector<half> &a, const std::vector<half> &b) {
        float max_diff = 0.0f;
        float max_val = 0.0f;

        for (size_t i = 0; i < a.size(); ++i) {
            float va = __half2float(a[i]);
            float vb = __half2float(b[i]);
            max_diff = std::max(max_diff, std::abs(va - vb));
            max_val = std::max(max_val, std::max(std::abs(va), std::abs(vb)));
        }

        return max_val > 0 ? max_diff / max_val : 0.0f;
    }
};

// Unit tests
TEST_F(W8A16MatMulTest, SmallMatrixCorrectness) {
    int M = 4, N = 8, K = 16;
    int group_size = 8;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    // Allocate device memory
    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // Run W8A16 kernel
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    // Dequantize and run FP16 reference
    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    // Check relative error
    float rel_error = computeRelativeError(output, output_ref);
    EXPECT_LT(rel_error, 0.01f) << "Relative error too high: " << rel_error;
}

TEST_F(W8A16MatMulTest, EdgeCaseM1) {
    int M = 1, N = 64, K = 128;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    EXPECT_NO_THROW({
        w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                     group_size);
        cudaDeviceSynchronize();
    });
}

TEST_F(W8A16MatMulTest, EdgeCaseN1) {
    int M = 64, N = 1, K = 128;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    EXPECT_NO_THROW({
        w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                     group_size);
        cudaDeviceSynchronize();
    });
}

TEST_F(W8A16MatMulTest, NonAlignedDimensions) {
    int M = 17, N = 33, K = 65;
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(output, output_ref);
    EXPECT_LT(rel_error, 0.01f);
}

#if 0
// Property-based tests
// Feature: tiny-llm-inference-engine, Property 1: W8A16 MatMul Numerical
// Accuracy Validates: Requirements 2.5, 2.6
// NOTE: Disabled in CUDA translation units due to GCC 11/12 + nvcc
// compatibility issues with rapidcheck's GTest integration.

class W8A16PropertyTest : public W8A16MatMulTest {
  protected:
    void SetUp() override {
        if (hasCudaDevice()) {
            cudaSetDevice(0);
        }
    }

    void TearDown() override {
        if (hasCudaDevice()) {
            cudaDeviceSynchronize();
        }
    }
};

RC_GTEST_FIXTURE_PROP(W8A16PropertyTest, NumericalAccuracyProperty,
                      (int m_raw, int n_raw, int k_raw)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Constrain dimensions to reasonable ranges
    int M = 1 + (std::abs(m_raw) % 128);
    int N = 8 + (std::abs(n_raw) % 256);
    int K = 32 + (std::abs(k_raw) % 512);
    int group_size = 32;
    int num_groups = (K + group_size - 1) / group_size;

    // Generate random data
    auto input = randomFP16(M, K, 0.5f);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    // Allocate device memory
    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // Run W8A16 kernel
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    // Dequantize and run FP16 reference
    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    // Property: relative error should be < 1%
    float rel_error = computeRelativeError(output, output_ref);
    RC_ASSERT(rel_error < 0.01f);
}

RC_GTEST_FIXTURE_PROP(W8A16PropertyTest, DifferentGroupSizes,
                      (int m_raw, int n_raw, int k_raw, int gs_raw)) {
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available";
    }
    int M = 1 + (std::abs(m_raw) % 64);
    int N = 8 + (std::abs(n_raw) % 128);
    int K = 64 + (std::abs(k_raw) % 256);

    // Test different group sizes: 32, 64, 128
    int group_sizes[] = {32, 64, 128};
    int group_size = group_sizes[std::abs(gs_raw) % 3];
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K, 0.5f);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_weight_fp16(K * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);

    dequantize_weights(d_weight.data(), d_scales.data(), d_weight_fp16.data(), K, N, group_size);
    fp16_matmul_reference(d_input.data(), d_weight_fp16.data(), d_output_ref.data(), M, N, K);

    cudaDeviceSynchronize();

    std::vector<half> output(M * N);
    std::vector<half> output_ref(M * N);
    d_output.copyToHost(output.data(), M * N);
    d_output_ref.copyToHost(output_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(output, output_ref);
    RC_ASSERT(rel_error < 0.01f);
}
#endif

// 大矩阵（M*N >= 4096）走 tiled kernel 分支；小矩阵测试只覆盖 reference。
// 真实模型（1×896×896、group 128）必须与 reference 差分对齐。
TEST_F(W8A16MatMulTest, LargeMatrixTiledMatchesReference) {
    int M = 1, N = 896, K = 896;
    int group_size = 128;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_output(M * N);
    DeviceBuffer<half>   d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // tiled 分支（M*N = 896 >= 4096）
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_output.data(), M, N, K,
                 group_size);
    // reference 分支
    w8a16_matmul_reference(d_input.data(), d_weight.data(), d_scales.data(), d_output_ref.data(), M,
                           N, K, group_size);
    cudaDeviceSynchronize();

    std::vector<half> out(M * N);
    std::vector<half> out_ref(M * N);
    d_output.copyToHost(out.data(), M * N);
    d_output_ref.copyToHost(out_ref.data(), M * N);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i) {
        float a = __half2float(out[i]);
        float b = __half2float(out_ref[i]);
        EXPECT_NEAR(a, b, std::max(1e-2f, std::abs(b) * 1e-2f))
            << "mismatch at " << i << ": tiled=" << a << " ref=" << b;
    }
}

// fp16_matmul 的 M==1 decode 快速路径：与 GPU reference 及 CPU float 参考对齐。
// lm_head 形状 [1, hidden] @ [hidden, vocab]，N 可能很大（Qwen2.5 为 151936）。
TEST_F(W8A16MatMulTest, Fp16MatmulM1MatchesReference) {
    int M = 1, K = 128, N = 1024;

    auto input = randomFP16(M, K);
    auto weight = randomFP16(K, N);

    DeviceBuffer<half> d_input(M * K);
    DeviceBuffer<half> d_weight(K * N);
    DeviceBuffer<half> d_output(M * N);
    DeviceBuffer<half> d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);

    // M==1 走 warp-per-column 快速路径
    fp16_matmul(d_input.data(), d_weight.data(), d_output.data(), M, N, K);
    // reference kernel（测试基准）
    fp16_matmul_reference(d_input.data(), d_weight.data(), d_output_ref.data(), M, N, K);
    cudaDeviceSynchronize();

    std::vector<half> out(M * N);
    std::vector<half> out_ref(M * N);
    d_output.copyToHost(out.data(), M * N);
    d_output_ref.copyToHost(out_ref.data(), M * N);
    cudaDeviceSynchronize();

    // CPU float 参考（float 累加后转 half），与 GPU 快速路径逐元素比较
    for (int col = 0; col < N; ++col) {
        float cpu_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_sum += __half2float(input[k]) * __half2float(weight[k * N + col]);
        }
        EXPECT_NEAR(__half2float(out[col]), cpu_sum, 1e-1f)
            << "fast path mismatch vs CPU at col " << col;
        EXPECT_NEAR(__half2float(out[col]), __half2float(out_ref[col]), 1e-1f)
            << "fast path mismatch vs gpu reference at col " << col;
    }
}

// fp16_matmul 的 M>1 路径必须回退到 reference 并保持正确。
TEST_F(W8A16MatMulTest, Fp16MatmulM4FallsBackToReference) {
    int M = 4, K = 128, N = 256;

    auto input = randomFP16(M, K);
    auto weight = randomFP16(K, N);

    DeviceBuffer<half> d_input(M * K);
    DeviceBuffer<half> d_weight(K * N);
    DeviceBuffer<half> d_output(M * N);
    DeviceBuffer<half> d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);

    fp16_matmul(d_input.data(), d_weight.data(), d_output.data(), M, N, K);
    fp16_matmul_reference(d_input.data(), d_weight.data(), d_output_ref.data(), M, N, K);
    cudaDeviceSynchronize();

    std::vector<half> out(M * N);
    std::vector<half> out_ref(M * N);
    d_output.copyToHost(out.data(), M * N);
    d_output_ref.copyToHost(out_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(out, out_ref);
    EXPECT_LT(rel_error, 0.01f) << "M>1 fp16_matmul diverged from reference";
}

TEST_F(W8A16MatMulTest, Fp16MatmulTransposedM4MatchesReference) {
    int M = 4, K = 128, N = 1024;

    auto              input = randomFP16(M, K);
    auto              weight = randomFP16(K, N);
    std::vector<half> weight_t(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            weight_t[static_cast<size_t>(n) * K + k] = weight[static_cast<size_t>(k) * N + n];
        }
    }

    DeviceBuffer<half> d_input(M * K);
    DeviceBuffer<half> d_weight(K * N);
    DeviceBuffer<half> d_weight_t(N * K);
    DeviceBuffer<half> d_output(M * N);
    DeviceBuffer<half> d_output_ref(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_weight_t.copyFromHost(weight_t.data(), N * K);

    fp16_matmul(d_input.data(), d_weight.data(), d_weight_t.data(), d_output.data(), M, N, K);
    fp16_matmul_reference(d_input.data(), d_weight.data(), d_output_ref.data(), M, N, K);
    cudaDeviceSynchronize();

    std::vector<half> out(M * N);
    std::vector<half> out_ref(M * N);
    d_output.copyToHost(out.data(), M * N);
    d_output_ref.copyToHost(out_ref.data(), M * N);
    cudaDeviceSynchronize();

    float rel_error = computeRelativeError(out, out_ref);
    EXPECT_LT(rel_error, 0.01f) << "transposed M>1 fp16_matmul diverged from reference";
}

// ─────────────────────────────────────────────────────────────────
// 任务 C1：转置权重 M==1 快路径
// ─────────────────────────────────────────────────────────────────

// 对单个 (N, K, group_size) 形状做转置快路径 vs 旧 m1 kernel vs CPU 参考的差分。
static void expectTransposedW8A16Matches(int N, int K, int group_size) {
    int M = 1;
    int num_groups = (K + group_size - 1) / group_size;

    // 本地随机数据（与 fixture 生成逻辑一致，避免依赖 fixture 的 protected 访问）
    std::mt19937                          gen_in(42);
    std::uniform_real_distribution<float> dist_in(-1.0f, 1.0f);
    std::vector<half>                     input(M * K);
    for (auto &v : input)
        v = __float2half(dist_in(gen_in));

    std::mt19937                       gen_w(123);
    std::uniform_int_distribution<int> dist_w(-127, 127);
    std::vector<int8_t>                weight(static_cast<size_t>(K) * N);
    for (auto &v : weight)
        v = static_cast<int8_t>(dist_w(gen_w));

    std::mt19937                          gen_s(456);
    std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);
    std::vector<half>                     scales(static_cast<size_t>(num_groups) * N);
    for (auto &v : scales)
        v = __float2half(dist_s(gen_s));

    // 构造转置布局 [N, K] / [N, scale_rows]（host 端转置）
    std::vector<int8_t> weight_t(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight_t[static_cast<size_t>(n) * K + k] = weight[static_cast<size_t>(k) * N + n];
    std::vector<half> scales_t(static_cast<size_t>(N) * num_groups);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < num_groups; ++g)
            scales_t[static_cast<size_t>(n) * num_groups + g] =
                scales[static_cast<size_t>(g) * N + n];

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<int8_t> d_weight_t(N * K);
    DeviceBuffer<half>   d_scales_t(N * num_groups);
    DeviceBuffer<half>   d_old(M * N); // 旧 m1 kernel
    DeviceBuffer<half>   d_new(M * N); // 转置快路径

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);
    d_weight_t.copyFromHost(weight_t.data(), N * K);
    d_scales_t.copyFromHost(scales_t.data(), N * num_groups);

    // 旧路径（无转置指针）
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_old.data(), M, N, K,
                 group_size);
    // 新路径（转置指针）
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_weight_t.data(),
                 d_scales_t.data(), d_new.data(), M, N, K, group_size);
    cudaDeviceSynchronize();

    std::vector<half> out_old(M * N);
    std::vector<half> out_new(M * N);
    d_old.copyToHost(out_old.data(), M * N);
    d_new.copyToHost(out_new.data(), M * N);
    cudaDeviceSynchronize();

    // CPU float 参考：output[col] = sum_k input[k] * weight[k*N+col] * scales[(k/gs)*N+col]
    // 容差：绝对值 1e-1（小形状），大 K 归约数值量级放大时退化为 1% 相对容差
    // （与 LargeMatrixTiledMatchesReference 口径一致）。
    for (int col = 0; col < N; ++col) {
        float cpu_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_sum += __half2float(input[k]) * static_cast<float>(weight[k * N + col]) *
                       __half2float(scales[(k / group_size) * N + col]);
        }
        float tol_cpu = std::max(1e-1f, std::abs(cpu_sum) * 1e-2f);
        EXPECT_NEAR(__half2float(out_new[col]), cpu_sum, tol_cpu)
            << "transposed fast path vs CPU at col " << col << " (N=" << N << ",K=" << K << ")";
        EXPECT_NEAR(__half2float(out_old[col]), cpu_sum, tol_cpu)
            << "old m1 kernel vs CPU at col " << col << " (N=" << N << ",K=" << K << ")";
        EXPECT_NEAR(__half2float(out_new[col]), __half2float(out_old[col]), 1e-2f)
            << "transposed fast path vs old m1 kernel at col " << col << " (N=" << N << ",K=" << K
            << ")";
    }
}

TEST_F(W8A16MatMulTest, TransposedFastPathMatchesCpuReference) {
    // 随机 M=1、K=128、N=1024（以及一个 down-proj 形状 N=896, K=4864）
    expectTransposedW8A16Matches(1024, 128, 128);
    expectTransposedW8A16Matches(896, 4864, 128);
}

// ─────────────────────────────────────────────────────────────────
// DEV-11：小 M 批量转置快路径（M ∈ {2,4,8}）
// ─────────────────────────────────────────────────────────────────

// 对 (M, N, K, group_size) 做新 dispatch（带转置缓冲）vs CPU float 参考的逐行差分。
static void expectSmallMW8A16Matches(int M, int N, int K, int group_size) {
    int num_groups = (K + group_size - 1) / group_size;

    std::mt19937                          gen_in(42 + M);
    std::uniform_real_distribution<float> dist_in(-1.0f, 1.0f);
    std::vector<half>                     input(static_cast<size_t>(M) * K);
    for (auto &v : input)
        v = __float2half(dist_in(gen_in));

    std::mt19937                       gen_w(123);
    std::uniform_int_distribution<int> dist_w(-127, 127);
    std::vector<int8_t>                weight(static_cast<size_t>(K) * N);
    for (auto &v : weight)
        v = static_cast<int8_t>(dist_w(gen_w));

    std::mt19937                          gen_s(456);
    std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);
    std::vector<half>                     scales(static_cast<size_t>(num_groups) * N);
    for (auto &v : scales)
        v = __float2half(dist_s(gen_s));

    std::vector<int8_t> weight_t(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight_t[static_cast<size_t>(n) * K + k] = weight[static_cast<size_t>(k) * N + n];
    std::vector<half> scales_t(static_cast<size_t>(N) * num_groups);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < num_groups; ++g)
            scales_t[static_cast<size_t>(n) * num_groups + g] =
                scales[static_cast<size_t>(g) * N + n];

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<int8_t> d_weight_t(N * K);
    DeviceBuffer<half>   d_scales_t(N * num_groups);
    DeviceBuffer<half>   d_out(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);
    d_weight_t.copyFromHost(weight_t.data(), N * K);
    d_scales_t.copyFromHost(scales_t.data(), N * num_groups);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_weight_t.data(),
                 d_scales_t.data(), d_out.data(), M, N, K, group_size);
    cudaDeviceSynchronize();

    std::vector<half> out(M * N);
    d_out.copyToHost(out.data(), M * N);
    cudaDeviceSynchronize();

    for (int m = 0; m < M; ++m) {
        for (int col = 0; col < N; ++col) {
            float cpu_sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                cpu_sum += __half2float(input[static_cast<size_t>(m) * K + k]) *
                           static_cast<float>(weight[k * N + col]) *
                           __half2float(scales[(k / group_size) * N + col]);
            }
            float tol = std::max(1e-1f, std::abs(cpu_sum) * 1e-2f);
            EXPECT_NEAR(__half2float(out[static_cast<size_t>(m) * N + col]), cpu_sum, tol)
                << "small-M path vs CPU at row " << m << " col " << col << " (M=" << M
                << ",N=" << N << ",K=" << K << ",gs=" << group_size << ")";
        }
    }
}

TEST_F(W8A16MatMulTest, SmallMTransposedMatchesCpuReference) {
    // 真实层形状 × B ∈ {2,4,8}：q/o 896→896、k/v 896→128、gate/up 896→4864、down 4864→896
    for (int M : {2, 4, 8}) {
        expectSmallMW8A16Matches(M, 896, 896, 128);
        expectSmallMW8A16Matches(M, 128, 896, 128);
        expectSmallMW8A16Matches(M, 4864, 896, 128);
        expectSmallMW8A16Matches(M, 896, 4864, 128);
    }
}

TEST_F(W8A16MatMulTest, SmallMTransposedTailsAndGroupBoundaries) {
    // N/K 尾与非整除 group：N=130（非 warp 块倍数）、K=100（非 32 倍数）、
    // K=130/gs=64（部分组）、K=128 边界整除
    expectSmallMW8A16Matches(4, 130, 896, 128);
    expectSmallMW8A16Matches(2, 896, 100, 128);
    expectSmallMW8A16Matches(4, 64, 130, 64);
    expectSmallMW8A16Matches(8, 64, 128, 128);
}

TEST_F(W8A16MatMulTest, SmallMNonTemplateMValuesFallBack) {
    // M=3、M=5 非模板实例：交回既有 reference/tiled 路由，结果仍须正确
    expectSmallMW8A16Matches(3, 896, 896, 128);
    expectSmallMW8A16Matches(5, 4864, 896, 128);
    // M=16 超出 kSmallMMax：交回 tiled 路径
    expectSmallMW8A16Matches(16, 512, 896, 128);
}

TEST_F(W8A16MatMulTest, SmallMZeroAndExtremeInputs) {
    // 全零输入 → 全零输出；含 0/正负混合由上方随机分布覆盖
    int M = 4, N = 896, K = 896, group_size = 128;
    int num_groups = K / group_size;
    std::vector<half>   input(M * K, __float2half(0.0f));
    std::vector<int8_t> weight_t(N * K, 127);
    std::vector<half>   scales_t(N * num_groups, __float2half(0.5f));

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N); // 未使用但须传非空
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<int8_t> d_weight_t(N * K);
    DeviceBuffer<half>   d_scales_t(N * num_groups);
    DeviceBuffer<half>   d_out(M * N);

    std::vector<int8_t> weight(K * N, 127);
    std::vector<half>   scales(num_groups * N, __float2half(0.5f));
    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);
    d_weight_t.copyFromHost(weight_t.data(), N * K);
    d_scales_t.copyFromHost(scales_t.data(), N * num_groups);

    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_weight_t.data(),
                 d_scales_t.data(), d_out.data(), M, N, K, group_size);
    cudaDeviceSynchronize();
    std::vector<half> out(M * N);
    d_out.copyToHost(out.data(), M * N);
    for (auto v : out)
        EXPECT_EQ(__half2float(v), 0.0f);
}

TEST_F(W8A16MatMulTest, TransposedFastPathFallsBackWithoutBuffers) {
    // 不传 data_t/scales_t 时，新重载必须与旧路径行为一致（bitwise）。
    int M = 1, K = 128, N = 512, group_size = 64;
    int num_groups = (K + group_size - 1) / group_size;

    auto input = randomFP16(M, K);
    auto weight = randomINT8(K, N);
    auto scales = randomScales(num_groups, N);

    DeviceBuffer<half>   d_input(M * K);
    DeviceBuffer<int8_t> d_weight(K * N);
    DeviceBuffer<half>   d_scales(num_groups * N);
    DeviceBuffer<half>   d_a(M * N);
    DeviceBuffer<half>   d_b(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_scales.copyFromHost(scales.data(), num_groups * N);

    // 旧签名
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), d_a.data(), M, N, K, group_size);
    // 新重载、转置指针为 nullptr
    w8a16_matmul(d_input.data(), d_weight.data(), d_scales.data(), nullptr, nullptr, d_b.data(), M,
                 N, K, group_size);
    cudaDeviceSynchronize();

    std::vector<half> a(M * N);
    std::vector<half> b(M * N);
    d_a.copyToHost(a.data(), M * N);
    d_b.copyToHost(b.data(), M * N);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(a[i], b[i]) << "fallback path diverged at " << i;
    }
}

// fp16_matmul 转置快路径：与旧 m1 路径一致（bitwise，同一归约顺序）。
TEST_F(W8A16MatMulTest, Fp16MatmulTransposedMatchesUntransposed) {
    int M = 1, K = 128, N = 1024;

    auto              input = randomFP16(M, K);
    auto              weight = randomFP16(K, N);
    std::vector<half> weight_t(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            weight_t[static_cast<size_t>(n) * K + k] = weight[static_cast<size_t>(k) * N + n];

    DeviceBuffer<half> d_input(M * K);
    DeviceBuffer<half> d_weight(K * N);
    DeviceBuffer<half> d_weight_t(N * K);
    DeviceBuffer<half> d_a(M * N);
    DeviceBuffer<half> d_b(M * N);

    d_input.copyFromHost(input.data(), M * K);
    d_weight.copyFromHost(weight.data(), K * N);
    d_weight_t.copyFromHost(weight_t.data(), N * K);

    fp16_matmul(d_input.data(), d_weight.data(), d_a.data(), M, N, K);
    fp16_matmul(d_input.data(), d_weight.data(), d_weight_t.data(), d_b.data(), M, N, K);
    cudaDeviceSynchronize();

    std::vector<half> a(M * N);
    std::vector<half> b(M * N);
    d_a.copyToHost(a.data(), M * N);
    d_b.copyToHost(b.data(), M * N);
    cudaDeviceSynchronize();

    for (int i = 0; i < M * N; ++i) {
        EXPECT_EQ(a[i], b[i]) << "fp16 transposed vs untransposed diverged at " << i;
    }
}
