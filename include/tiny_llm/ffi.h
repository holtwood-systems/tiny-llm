// C ABI 契约：paged-serving（Rust）与 tiny-llm（C++/CUDA）之间的执行后端桥接。
//
// 契约定义与 paged-serving 的 `src/tiny_llm_ffi.rs` 保持逐字段一致：
//   - TinyLlmConfig 为 9 个 int 的 repr(C) 布局（ABI v2；Rust 侧有布局守卫测试）
//   - 步进式执行：每步处理一个 batch（prefill/decode 混合）
//   - KV 生命周期由本后端管理：
//       策略 1（分页 KV，max_num_blocks > 0）：block_tables / num_blocks 生效
//       策略 2（连续 KV，max_num_blocks == 0）：block_tables / num_blocks 忽略
//   - 策略 1 已启用：paged-serving 默认走分页 KV（max_num_blocks = 引擎配置），
//     `PAGED_SERVING_TINY_LLM_STRATEGY=2` 可回退到策略 2。
//
// 数据布局约定：
//   - seq_ids 显式给出每序列 id；input_tokens / positions 是扁平化数组
//     （seq_lens 描述每序列切分，与 seq_ids 对齐），positions 不得为空。
//     positions 不是可选提示：后端会校验它——prefill 段必须是 0..len-1
//     的连续绝对位置，decode 段必须等于后端内部位置计数（即调用方的
//     context_len - 1），decode 的 seq_len 必须为 1；不一致返回错误，
//     避免 RoPE 位置 / KV 槽位静默错位。
//   - block_tables 是扁平化数组；num_blocks[i] = 第 i 个序列的 block_tables
//     长度，扁平化 block_tables 的总长为 sum(num_blocks)
//   - logprobs_k == 0 不输出；否则 logprobs 至少容纳
//     num_sequences * logprobs_k * 2 个 float，按 (token_id, logprob) 交错；
//     token_id 也以 float 表示。logprobs_k 必须位于 [0, vocab_size]，
//     logprobs_k > 0 时 logprobs 不得为 nullptr
#ifndef TINY_LLM_FFI_H
#define TINY_LLM_FFI_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TinyLlmHandle TinyLlmHandle;

typedef struct TinyLlmConfig {
    // 6 个几何字段：0 = 由 GGUF 提取；非 0 时必须与 GGUF 一致，
    // 否则 tinyllm_load 返回 NULL 并写入不匹配字段。
    int hidden_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int block_size;     // 分页块大小（策略 2 下忽略，保留对齐）
    int max_batch_size; // 引擎同时管理的最大序列数
    int max_num_blocks; // 分页 KV 池的物理块总数；0 = 策略 2（连续 KV）
    // ↑ 该字段使 TinyLlmConfig 变为 9 个 int 的 repr(C) 布局（ABI v2）。
} TinyLlmConfig;

// 加载 GGUF 模型；成功返回句柄，失败返回 NULL 并写入 err_buf。
TinyLlmHandle *tinyllm_load(const char *model_path, const TinyLlmConfig *config, char *err_buf,
                            int err_buf_len);

// 单步执行一个 batch（prefill/decode 混合），逐序列输出下一 token 与 logprobs。
// seq_ids 显式给出每个序列的 id（由 tinyllm_allocate_sequence 分配），
// 支持任意 id 的序列混批（返回 0 成功，非 0 错误码）。
// num_blocks[i] = 第 i 个序列的 block_tables 长度；扁平化 block_tables 的
// 总长为 sum(num_blocks)。策略 2 下二者忽略（传 nullptr 即可）。
int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                 const int *positions, const int *seq_lens, const int *block_tables,
                 const int *num_blocks, const unsigned char *is_prefill, int num_sequences,
                 int *next_tokens, float *logprobs, int logprobs_k);

// KV 生命周期：分配/释放一个序列的缓存。
int tinyllm_allocate_sequence(TinyLlmHandle *handle, int seq_id, int num_tokens);
int tinyllm_free_sequence(TinyLlmHandle *handle, int seq_id);

// 释放句柄及其全部资源。
void tinyllm_free(TinyLlmHandle *handle);

#ifdef __cplusplus
}
#endif

#endif // TINY_LLM_FFI_H
