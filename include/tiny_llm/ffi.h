// C ABI 契约：paged-serving（Rust）与 tiny-llm（C++/CUDA）之间的执行后端桥接。
//
// 契约定义与 paged-serving 的 `src/tiny_llm_ffi.rs` 保持逐字段一致：
//   - TinyLlmConfig 为 9 个 int 的 repr(C) 布局（ABI v2 起；Rust 侧有布局守卫测试）
//   - 步进式执行：每步处理一个 batch（prefill/decode 混合）
//   - KV 生命周期由本后端管理：
//       策略 1（分页 KV，max_num_blocks > 0）：block_tables / num_blocks 生效
//       策略 2（连续 KV，max_num_blocks == 0）：block_tables / num_blocks 忽略
//   - 策略 1 已启用：paged-serving 默认走分页 KV（max_num_blocks = 引擎配置），
//     `PAGED_SERVING_TINY_LLM_STRATEGY=2` 可回退到策略 2。
//
// ABI v3 变更（相对 v2）：
//   - tinyllm_step 新增出参 `seq_status`（长度 num_sequences）：逐序列状态码，
//     `seq_status[i] == 0` 表示该序列本步成功、next_tokens[i] 有效；
//     `< 0` 表示该序列失败（next_tokens[i] 无效）。单个序列失败不再使整个
//     调用失败——调用方按 seq_status 做逐序列归因与终态处理。
//   - 返回值语义随之收窄：TLLM_OK(0) 仅表示"调用本身已执行"（含逐序列结果）；
//     TLLM_ERR 仅用于调用级违约（空指针、logprobs_k 越界、内部缓冲分配失败、
//     批量后处理失败等无法归属到单个序列的错误）。
//
// 数据布局约定：
//   - seq_ids 显式给出每序列 id；input_tokens 是扁平化数组（seq_lens 描述
//     每序列切分，与 seq_ids 对齐）。decode 的 seq_len 必须为 1，否则
//     该序列按逐序列失败处理（seq_status），不算调用级违约。
//   - positions 是签名保留的惰性参数：当前实现不读取、不校验，序列位置
//     由后端 SeqState 内部跟踪（见下方"已声明但未生效的参数"）。调用方
//     仍按现有布局传入真实值，但传入值不影响执行。
//   - block_tables 是扁平化数组；num_blocks[i] = 第 i 个序列的 block_tables
//     长度，扁平化 block_tables 的总长为 sum(num_blocks)
//   - logprobs_k == 0 不输出；否则 logprobs 至少容纳
//     num_sequences * logprobs_k * 2 个 float，按 (token_id, logprob) 交错；
//     token_id 也以 float 表示。logprobs_k 必须位于 [0, vocab_size]，
//     logprobs_k > 0 时 logprobs 不得为 nullptr
//
// 已声明但未生效的参数（诚实标注，避免调用方误读契约）：
//   - positions：当前两种策略下均被忽略。序列位置由后端 SeqState 内部跟踪
//     （prefill 恒从位置 0 写起、decode 取后端计数器）；调用方传入的 positions
//     仅作布局占位，不代表"写入位置控制权"。如需 chunked prefill / 位置控制，
//     必须先修订本契约。
//   - tinyllm_allocate_sequence 的 num_tokens：策略 1（分页）下仅登记序列、
//     不参与任何容量预留——分页 KV 的真实上限是共享块池（max_num_blocks ×
//     block_size）与 max_seq_len；策略 2（连续 KV）下才作为每序列预留容量生效。
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

// 逐序列状态码（ABI v3）：seq_status[i] 的值域。
#define TLLM_SEQ_OK  0
#define TLLM_SEQ_ERR (-1)

// 加载 GGUF 模型；成功返回句柄，失败返回 NULL 并写入 err_buf。
TinyLlmHandle *tinyllm_load(const char *model_path, const TinyLlmConfig *config, char *err_buf,
                            int err_buf_len);

// 单步执行一个 batch（prefill/decode 混合），逐序列输出下一 token 与 logprobs。
// seq_ids 显式给出每个序列的 id（由 tinyllm_allocate_sequence 分配），
// 支持任意 id 的序列混批。
// num_blocks[i] = 第 i 个序列的 block_tables 长度；扁平化 block_tables 的
// 总长为 sum(num_blocks)。策略 2 下二者忽略（传 nullptr 即可）。
//
// ABI v3：seq_status 必须非空（长度 num_sequences）。返回 TLLM_OK 时，
// seq_status[i] 逐序列给出本步结果：TLLM_SEQ_OK(0) 成功 / TLLM_SEQ_ERR(<0)
// 失败。失败序列不影响同 batch 其他序列；其 KV 写入按"层内失败不推进
// position"保持序列级原子性。返回 TLLM_ERR 表示调用级违约，此时
// seq_status/next_tokens 内容均不可信。
int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                 const int *positions, const int *seq_lens, const int *block_tables,
                 const int *num_blocks, const unsigned char *is_prefill, int num_sequences,
                 int *next_tokens, float *logprobs, int logprobs_k, int *seq_status);

// KV 生命周期：分配/释放一个序列的缓存。
// num_tokens 语义按策略区分（见头部"已声明但未生效的参数"）：
//   策略 1：仅登记序列（num_tokens 不参与分配，但必须 > 0）；
//   策略 2：作为该序列的连续 KV 预留容量生效。
int tinyllm_allocate_sequence(TinyLlmHandle *handle, int seq_id, int num_tokens);
int tinyllm_free_sequence(TinyLlmHandle *handle, int seq_id);

// 释放句柄及其全部资源。
void tinyllm_free(TinyLlmHandle *handle);

#ifdef __cplusplus
}
#endif

#endif // TINY_LLM_FFI_H
