#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// In-place elementwise add: data[i] += add[i]
void add_inplace(half *data, const half *add, int num_elements, cudaStream_t stream = 0);

// 广播加 bias：data[r*cols + c] += bias[c]（每行加同一 bias 向量）
void add_bias_inplace(half *data, const half *bias, int rows, int cols, cudaStream_t stream = 0);

// In-place SiLU + multiply: gate[i] = silu(gate[i]) * up[i]
void silu_mul_inplace(half *gate, const half *up, int num_elements, cudaStream_t stream = 0);

// Embedding gather: output[token_idx, hidden_idx] = embedding[token_id, hidden_idx]
void gather_embeddings(const int *tokens, const half *embedding, half *output, int num_tokens,
                       int hidden_dim, int vocab_size, cudaStream_t stream = 0);

// 行索引 gather：dst[dst_rows[r], :] = src[r, :]。dst_rows 是 device int[num_rows]——
// 行→槽映射作为 device 数据读取，CUDA Graph 重放时只需更新该缓冲（DEV-14）。
void copy_rows_indexed(half *dst, const half *src, const int *dst_rows, int num_rows,
                       int row_elems, cudaStream_t stream = 0);

// 任务 3.2：KV append 的 device 写位置版本。把当前 step 的 K/V 写入 cache 的
// *device_write_pos 位置（CUDA Graph 重放：host 更新 device 位置后重放即可，
// 不像 host 计算的目标地址会被 graph 固化）。new_k/new_v 都是 [num_tokens, kv_heads*D]；
// 写 num_tokens 行（D2e 修复：prefill 多 token 时逐行写入，与 host 版语义一致）。
void append_kv_at(const half *new_k, const half *new_v, half *dst_k, half *dst_v,
                  const int *device_write_pos, int num_tokens, int kv_heads, int head_dim,
                  cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
