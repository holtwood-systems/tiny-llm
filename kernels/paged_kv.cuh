#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tiny_llm {
namespace kernels {

// 把连续 [num_tokens, chunk_dim] 的 src 按块表散布到 pool。
// 绝对位置 = position + t；块内偏移 = abs % block_size；
// pool 传入的是"本层"指针（调用方负责 layer 偏移）。
// block_table 是 device 端 int 数组（visible_blocks 个物理块 id）。
// max_num_blocks：块 id 合法值域 [0, max_num_blocks)；越界 id 的元素被跳过
// （scatter 不写 / gather 写 0），防止坏块表造成越界访存毒化 CUDA 上下文。
void paged_scatter_blocks(const half *src, half *pool, const int *block_table, int num_tokens,
                          int position, int block_size, int chunk_dim, int max_num_blocks,
                          cudaStream_t stream = 0);

// 从块表把 [visible_tokens, chunk_dim] 连续读回 dst。
void paged_gather_blocks(half *dst, const half *pool, const int *block_table, int visible_tokens,
                         int block_size, int chunk_dim, int max_num_blocks,
                         cudaStream_t stream = 0);

// device 参数变体（CUDA Graph 可重放）：scatter 的起始绝对位置、
// gather 的可见长度从 device int 读取，graph 重放时随 H2D 更新生效。
// gather_dev 的 grid 按 max_visible_tokens 满配发射（kernel 内早退）。
void paged_scatter_blocks_dev(const half *src, half *pool, const int *block_table, int num_tokens,
                              const int *position_dev, int block_size, int chunk_dim,
                              int max_num_blocks, cudaStream_t stream = 0);

void paged_gather_blocks_dev(half *dst, const half *pool, const int *block_table,
                             const int *visible_dev, int block_size, int chunk_dim,
                             int max_num_blocks, int max_visible_tokens, cudaStream_t stream = 0);

} // namespace kernels
} // namespace tiny_llm
