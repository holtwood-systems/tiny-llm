#include "tiny_llm/cuda_utils.h"
#include "w8a16_matmul.cuh"
#include "warp_utils.cuh"

namespace tiny_llm {
namespace kernels {

// 诊断开关：强制走 reference kernel（用于区分 tiled kernel 与权重/上层逻辑问题）
bool g_force_reference = false;

// Dequantization kernel
__global__ void dequantize_kernel(const int8_t *__restrict__ weight_int8,
                                  const half *__restrict__ scales, half *__restrict__ weight_fp16,
                                  int K, int N, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;

    if (idx < total) {
        int k = idx / N;
        int n = idx % N;

        int group_idx = k / group_size;
        int scale_idx = group_idx * N + n;

        float w_int = static_cast<float>(weight_int8[idx]);
        float s = __half2float(scales[scale_idx]);
        weight_fp16[idx] = __float2half(w_int * s);
    }
}

void dequantize_weights(const int8_t *weight_int8, const half *scales, half *weight_fp16, int K,
                        int N, int group_size, cudaStream_t stream) {
    if (K <= 0 || N <= 0 || group_size <= 0) {
        return;
    }

    int total = K * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    dequantize_kernel<<<grid_size, block_size, 0, stream>>>(weight_int8, scales, weight_fp16, K, N,
                                                            group_size);
}

// Simple FP16 matmul reference kernel
__global__ void fp16_matmul_kernel(const half *__restrict__ A, const half *__restrict__ B,
                                   half *__restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = __float2half(sum);
    }
}

void fp16_matmul_reference(const half *input, const half *weight, half *output, int M, int N, int K,
                           cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    fp16_matmul_kernel<<<grid, block, 0, stream>>>(input, weight, output, M, N, K);
}

// Decode-optimized FP16 GEMM for M == 1.  Same warp-per-output-column scheme
// as w8a16_matmul_m1_kernel; this keeps the fp16 lm_head path from becoming
// the decode bottleneck for large vocabularies.
__global__ void fp16_matmul_m1_kernel(const half *__restrict__ input,
                                      const half *__restrict__ weight, half *__restrict__ output,
                                      int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;

    const int lane = threadIdx.x & 31;
    float     sum = 0.0f;

    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = __half2float(weight[k * N + col]);
        sum += a * w;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[col] = __float2half(sum);
    }
}

// 行并行快路径（转置权重布局 [N, K]）：每个 warp 负责一个输出列，blockIdx.y
// 选择 batch row。32 个 lane 沿 K 并行 reduce；weight_t[col*K + k] 在 warp 内按 k
// 连续，访存完全 coalesced（对比旧 kernel 的 weight[k*N + col] 跨 lane stride=N）。
__global__ void fp16_matmul_transposed_rows_kernel(const half *__restrict__ input,
                                                   const half *__restrict__ weight_t,
                                                   half *__restrict__ output, int M, int N, int K) {
    const int row = blockIdx.y;
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (row >= M || col >= N) return;

    const int lane = threadIdx.x & 31;
    float     sum = 0.0f;

    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[static_cast<size_t>(row) * K + k]);
        float w = __half2float(weight_t[(size_t)col * K + k]);
        sum += a * w;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[static_cast<size_t>(row) * N + col] = __float2half(sum);
    }
}

void fp16_matmul(const half *input, const half *weight, half *output, int M, int N, int K,
                 cudaStream_t stream) {
    fp16_matmul(input, weight, /*weight_t=*/nullptr, output, M, N, K, stream);
}

void fp16_matmul(const half *input, const half *weight, const half *weight_t, half *output, int M,
                 int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    constexpr int WARPS_PER_BLOCK = 4; // 128 threads
    dim3          block(WARPS_PER_BLOCK * 32);
    dim3          grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, M);
    if (weight_t != nullptr) {
        fp16_matmul_transposed_rows_kernel<<<grid, block, 0, stream>>>(input, weight_t, output, M,
                                                                       N, K);
        return;
    }

    if (M == 1) {
        dim3 single_row_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        fp16_matmul_m1_kernel<<<single_row_grid, block, 0, stream>>>(input, weight, output, N, K);
        return;
    }

    fp16_matmul_reference(input, weight, output, M, N, K, stream);
}

// W8A16 reference kernel (simple but correct)
__global__ void w8a16_matmul_reference_kernel(const half *__restrict__ input,
                                              const int8_t *__restrict__ weight,
                                              const half *__restrict__ scales,
                                              half *__restrict__ output, int M, int N, int K,
                                              int group_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            // Load input
            float a = __half2float(input[row * K + k]);

            // Load and dequantize weight
            int8_t w_int = weight[k * N + col];
            int    group_idx = k / group_size;
            float  scale = __half2float(scales[group_idx * N + col]);
            float  w = static_cast<float>(w_int) * scale;

            sum += a * w;
        }

        output[row * N + col] = __float2half(sum);
    }
}

void w8a16_matmul_reference(const half *input, const int8_t *weight, const half *scales,
                            half *output, int M, int N, int K, int group_size,
                            cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) {
        return;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    w8a16_matmul_reference_kernel<<<grid, block, 0, stream>>>(input, weight, scales, output, M, N,
                                                              K, group_size);
}

// Optimized W8A16 kernel with shared memory tiling
// Each thread block computes a TILE_M x TILE_N output tile
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int BLOCK_K = 32;

__global__ void w8a16_matmul_tiled_kernel(const half *__restrict__ input,
                                          const int8_t *__restrict__ weight,
                                          const half *__restrict__ scales,
                                          half *__restrict__ output, int M, int N, int K,
                                          int group_size) {
    // Shared memory for input tile and weight tile
    __shared__ half   smem_input[TILE_M][BLOCK_K + 1];  // +1 to avoid bank conflicts
    __shared__ int8_t smem_weight[BLOCK_K][TILE_N + 4]; // +4 for alignment

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread computes multiple output elements
    // Thread block is (TILE_N, TILE_M/4) = (64, 16)
    // Each thread handles 4 rows
    int global_col = bx * TILE_N + tx;

    // Thread-local accumulators for 4 rows
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Number of tiles along K dimension
    int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_start = k_tile * BLOCK_K;

        // Cooperative load of input tile
        // Each thread loads multiple elements
        for (int i = ty; i < TILE_M; i += blockDim.y) {
            for (int j = tx; j < BLOCK_K; j += blockDim.x) {
                int global_row = by * TILE_M + i;
                int global_k = k_start + j;

                if (global_row < M && global_k < K) {
                    smem_input[i][j] = input[global_row * K + global_k];
                } else {
                    smem_input[i][j] = __float2half(0.0f);
                }
            }
        }

        // Cooperative load of weight tile
        for (int i = ty; i < BLOCK_K; i += blockDim.y) {
            for (int j = tx; j < TILE_N; j += blockDim.x) {
                int global_k = k_start + i;
                int g_col = bx * TILE_N + j;

                if (global_k < K && g_col < N) {
                    smem_weight[i][j] = weight[global_k * N + g_col];
                } else {
                    smem_weight[i][j] = 0;
                }
            }
        }

        __syncthreads();

        // Compute partial products for 4 rows per thread
        if (global_col < N) {
#pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                int global_k = k_start + k;
                if (global_k < K) {
                    // Get the correct scale for this k position
                    int   group_idx = global_k / group_size;
                    float scale = __half2float(scales[group_idx * N + global_col]);
                    float w = static_cast<float>(smem_weight[k][tx]) * scale;

// Accumulate for 4 rows
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        int local_row = ty * 4 + r;
                        int global_row = by * TILE_M + local_row;
                        if (global_row < M && local_row < TILE_M) {
                            float a = __half2float(smem_input[local_row][k]);
                            acc[r] += a * w;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write output for 4 rows
    if (global_col < N) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int local_row = ty * 4 + r;
            int global_row = by * TILE_M + local_row;
            if (global_row < M) {
                output[global_row * N + global_col] = __float2half(acc[r]);
            }
        }
    }
}

// Decode-optimized W8A16 matmul for M == 1.
// The generic dispatch used the slow reference kernel for small M*N (which is
// every single-token decode GEMM).  Here one warp owns one output column and
// the 32 lanes reduce over K in parallel, which is substantially faster for
// the [1, K] @ [K, N] shape that dominates autoregressive decode.
__global__ void w8a16_matmul_m1_kernel(const half *__restrict__ input,
                                       const int8_t *__restrict__ weight,
                                       const half *__restrict__ scales, half *__restrict__ output,
                                       int N, int K, int group_size) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;

    const int lane = threadIdx.x & 31;
    float     sum = 0.0f;

    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = static_cast<float>(weight[k * N + col]);
        float s = __half2float(scales[(k / group_size) * N + col]);
        sum += a * w * s;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[col] = __float2half(sum);
    }
}

// M==1 decode 快路径（转置权重布局 [N, K]）：lane 沿 K 连续读
// weight_t[col*K + k]，跨 lane 地址 stride=1，完全 coalesced；
// 对比旧 kernel 的 weight[k*N + col] 跨 lane stride=N×2B（非 coalesced）。
__global__ void w8a16_matmul_m1_transposed_kernel(const half *__restrict__ input,
                                                  const int8_t *__restrict__ weight_t,
                                                  const half *__restrict__ scales_t,
                                                  half *__restrict__ output, int N, int K,
                                                  int group_size) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;

    const int lane = threadIdx.x & 31;
    float     sum = 0.0f;
    const int scale_rows = (K + group_size - 1) / group_size;

    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = static_cast<float>(weight_t[(size_t)col * K + k]);
        float s = __half2float(scales_t[(size_t)col * scale_rows + (k / group_size)]);
        sum += a * w * s;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[col] = __float2half(sum);
    }
}

// 小 M decode 批量路径（转置权重 [N, K]）：一个 warp 负责一列，寄存器持有
// M 个行累加器——weight_t[col,k] 每 lane 只读一次，摊薄到全部 M 行；
// input 行读取为 warp 内广播（每行同地址），无额外带宽放大。
// 适用 M ∈ [2, kSmallMMax]；更大 M 交回 tiled 路径。
inline constexpr int kSmallMMax = 8;

template <int MVAL>
__global__ void w8a16_matmul_smallm_transposed_kernel(const half *__restrict__ input,
                                                      const int8_t *__restrict__ weight_t,
                                                      const half *__restrict__ scales_t,
                                                      half *__restrict__ output, int N, int K,
                                                      int group_size) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;

    const int lane = threadIdx.x & 31;
    const int scale_rows = (K + group_size - 1) / group_size;
    float      acc[MVAL];
#pragma unroll
    for (int m = 0; m < MVAL; ++m) acc[m] = 0.0f;

    for (int k = lane; k < K; k += 32) {
        const float w = static_cast<float>(weight_t[(size_t)col * K + k]) *
                        __half2float(scales_t[(size_t)col * scale_rows + (k / group_size)]);
#pragma unroll
        for (int m = 0; m < MVAL; ++m) {
            acc[m] += __half2float(input[(size_t)m * K + k]) * w;
        }
    }

#pragma unroll
    for (int m = 0; m < MVAL; ++m) {
        const float sum = warp_reduce_sum(acc[m]);
        if (lane == 0) output[(size_t)m * N + col] = __float2half(sum);
    }
}

void w8a16_matmul(const half *input, const int8_t *weight, const half *scales, half *output, int M,
                  int N, int K, int group_size, cudaStream_t stream) {
    w8a16_matmul(input, weight, scales, /*weight_t=*/nullptr, /*scales_t=*/nullptr, output, M, N, K,
                 group_size, stream);
}

void w8a16_matmul(const half *input, const int8_t *weight, const half *scales,
                  const int8_t *weight_t, const half *scales_t, half *output, int M, int N, int K,
                  int group_size, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) {
        return;
    }

    if (g_force_reference) {
        w8a16_matmul_reference(input, weight, scales, output, M, N, K, group_size, stream);
        return;
    }

    // Autoregressive decode fast path: [1, K] @ [K, N].
    if (M == 1) {
        constexpr int WARPS_PER_BLOCK = 4; // 128 threads
        dim3          block(WARPS_PER_BLOCK * 32);
        dim3          grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        if (weight_t != nullptr && scales_t != nullptr) {
            w8a16_matmul_m1_transposed_kernel<<<grid, block, 0, stream>>>(input, weight_t, scales_t,
                                                                          output, N, K, group_size);
        } else {
            w8a16_matmul_m1_kernel<<<grid, block, 0, stream>>>(input, weight, scales, output, N, K,
                                                               group_size);
        }
        return;
    }

    // 小 M 批量 decode：转置布局可用时走 small-M 快路径（权重读取摊薄 M 行）。
    if (M > 1 && M <= kSmallMMax && weight_t != nullptr && scales_t != nullptr) {
        constexpr int WARPS_PER_BLOCK = 4; // 128 threads
        dim3          block(WARPS_PER_BLOCK * 32);
        dim3          grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        switch (M) {
        case 2:
            w8a16_matmul_smallm_transposed_kernel<2><<<grid, block, 0, stream>>>(
                input, weight_t, scales_t, output, N, K, group_size);
            return;
        case 4:
            w8a16_matmul_smallm_transposed_kernel<4><<<grid, block, 0, stream>>>(
                input, weight_t, scales_t, output, N, K, group_size);
            return;
        case 8:
            w8a16_matmul_smallm_transposed_kernel<8><<<grid, block, 0, stream>>>(
                input, weight_t, scales_t, output, N, K, group_size);
            return;
        default:
            break; // 非 {2,4,8}：交回下方既有 reference/tiled 路由
        }
    }

    if (M * N < 4096) {
        w8a16_matmul_reference(input, weight, scales, output, M, N, K, group_size, stream);
        return;
    }

    dim3 block(TILE_N, TILE_M / 4);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    w8a16_matmul_tiled_kernel<<<grid, block, 0, stream>>>(input, weight, scales, output, M, N, K,
                                                          group_size);
}

} // namespace kernels
} // namespace tiny_llm
