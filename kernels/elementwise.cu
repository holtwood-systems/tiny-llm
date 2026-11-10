#include "elementwise.cuh"

namespace tiny_llm {
namespace kernels {

namespace {
__global__ void add_inplace_kernel(half *data, const half *add, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float a = __half2float(data[idx]);
        float b = __half2float(add[idx]);
        data[idx] = __float2half(a + b);
    }
}

__global__ void silu_mul_inplace_kernel(half *gate, const half *up, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu = g / (1.0f + expf(-g));
        gate[idx] = __float2half(silu * u);
    }
}

__global__ void gather_embeddings_kernel(const int *tokens, const half *embedding, half *output,
                                         int num_tokens, int hidden_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;
    if (idx >= total) {
        return;
    }

    int token_idx = idx / hidden_dim;
    int hidden_idx = idx % hidden_dim;
    int token_id = tokens[token_idx];

    if (token_id >= 0 && token_id < vocab_size) {
        output[idx] = embedding[token_id * hidden_dim + hidden_idx];
    } else {
        output[idx] = __float2half(0.0f);
    }
}
__global__ void add_bias_kernel(half *data, const half *bias, int cols, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return; // 尾块线程越界防护（grid 按 ceil(total/256) 启动）
    float b = __half2float(bias[idx % cols]);
    data[idx] = __float2half(__half2float(data[idx]) + b);
}
__global__ void copy_rows_indexed_kernel(half *dst, const half *src, const int *dst_rows,
                                        int num_rows, int row_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * row_elems;
    if (idx >= total) {
        return;
    }
    const int r = idx / row_elems;
    const int c = idx - r * row_elems;
    const int s = dst_rows[r];
    dst[static_cast<size_t>(s) * row_elems + c] = src[static_cast<size_t>(r) * row_elems + c];
}
} // namespace

void copy_rows_indexed(half *dst, const half *src, const int *dst_rows, int num_rows,
                       int row_elems, cudaStream_t stream) {
    if (num_rows <= 0 || row_elems <= 0 || !dst || !src || !dst_rows) {
        return;
    }
    const int total = num_rows * row_elems;
    const int block_size = 256;
    copy_rows_indexed_kernel<<<(total + block_size - 1) / block_size, block_size, 0, stream>>>(
        dst, src, dst_rows, num_rows, row_elems);
}

void add_bias_inplace(half *data, const half *bias, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || data == nullptr || bias == nullptr) return;
    int total = rows * cols;
    int block = 256;
    add_bias_kernel<<<(total + block - 1) / block, block, 0, stream>>>(data, bias, cols, total);
}

void add_inplace(half *data, const half *add, int num_elements, cudaStream_t stream) {
    if (num_elements <= 0) return;
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(data, add, num_elements);
}

void silu_mul_inplace(half *gate, const half *up, int num_elements, cudaStream_t stream) {
    if (num_elements <= 0) return;
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    silu_mul_inplace_kernel<<<grid_size, block_size, 0, stream>>>(gate, up, num_elements);
}

void gather_embeddings(const int *tokens, const half *embedding, half *output, int num_tokens,
                       int hidden_dim, int vocab_size, cudaStream_t stream) {
    if (num_tokens <= 0 || hidden_dim <= 0 || !tokens || !embedding || !output) {
        return;
    }
    int total = num_tokens * hidden_dim;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gather_embeddings_kernel<<<grid_size, block_size, 0, stream>>>(
        tokens, embedding, output, num_tokens, hidden_dim, vocab_size);
}

// 任务 3.2：KV append（device 写位置版本）。
// 每个线程负责一个 (token, head, dim) 元素；写位置从 device int 读取，使该 kernel
// 可被 CUDA Graph 重放（host 在每次 decode 前更新 device 值）。
// 写 num_tokens 行（D2e 修复：prefill 多 token 时逐行写入）。
__global__ void append_kv_at_kernel(const half *__restrict__ new_k, const half *__restrict__ new_v,
                                    half *__restrict__ dst_k, half *__restrict__ dst_v,
                                    const int *__restrict__ device_write_pos, int num_tokens,
                                    int kv_heads, int head_dim) {
    const int write_pos = *device_write_pos;
    const int per_token = kv_heads * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * per_token;
    if (idx >= total) return;
    const int t = idx / per_token;
    const int j = idx - t * per_token;
    const int offset = (write_pos + t) * per_token + j;
    dst_k[offset] = new_k[idx];
    dst_v[offset] = new_v[idx];
}

void append_kv_at(const half *new_k, const half *new_v, half *dst_k, half *dst_v,
                  const int *device_write_pos, int num_tokens, int kv_heads, int head_dim,
                  cudaStream_t stream) {
    if (!new_k || !new_v || !dst_k || !dst_v || !device_write_pos || num_tokens <= 0 ||
        kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    const int total = num_tokens * kv_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    append_kv_at_kernel<<<grid_size, block_size, 0, stream>>>(
        new_k, new_v, dst_k, dst_v, device_write_pos, num_tokens, kv_heads, head_dim);
}

} // namespace kernels
} // namespace tiny_llm
