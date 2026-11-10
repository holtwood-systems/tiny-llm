// C ABI 实现：paged-serving 与 tiny-llm 的执行后端桥接（ABI v3）。
//
// 策略 2（连续 KV）：block_tables 忽略；序列位置由引擎内部跟踪
// （KV 写入位置以 slot.current_len 为准，与调用方传入 position 解耦）。
// 逐序列执行（step 的第 s 项对应 seq_id = s），多序列语义正确。
// 正常 greedy 请求把末端 RMSNorm / LM head / argmax 组成 device batch，并在 step
// 末尾一次性回传；Transformer layer 的计算融合仍是后续里程碑。
//
// ABI v3：tinyllm_step 带 `seq_status` 出参做逐序列归因——单序列失败
// 标记 seq_status[s]=TLLM_SEQ_ERR 并继续处理其余序列，不再整批失败。
// 返回值仅表达调用级违约（空指针/范围非法/内部缓冲失败）。
//
// 策略 1 的 decode 默认启用 CUDA Graph（TLLM_FFI_CUDA_GRAPHS=0 关闭）：
// embed + 逐层 forwardPaged 记录为一张 handle 级 graph（地址绑定共享缓冲，
// 序列差异由每步 H2D 内容表达），首次 decode 先直接执行再捕获，其后重放。

#include "tiny_llm/ffi.h"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "sampling.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/execution_common.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h" // sampleGreedy
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/model_loader.h"
#include "tiny_llm/transformer.h"
#include "tiny_llm/validator.h"
#include "w8a16_matmul.cuh"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int TLLM_OK = 0;
constexpr int TLLM_ERR = -1;

struct SeqState {
    int  position = 0; // 已处理的 token 数（下一 decode 的绝对位置）
    bool allocated = false;
};

struct TinyLlmHandleImpl {
    tiny_llm::ModelConfig                                    config;
    tiny_llm::ModelWeights                                   weights;
    std::vector<std::unique_ptr<tiny_llm::TransformerLayer>> layers;
    std::unique_ptr<tiny_llm::KVCacheManager>                kv_cache;
    tiny_llm::LayerWorkspace                                 workspace;
    float                                                   *rope_cos = nullptr;
    float                                                   *rope_sin = nullptr;
    half                        *hidden_buf = nullptr; // [max_seq_len * hidden]
    half                        *logits_buf = nullptr; // [vocab_size]
    cudaStream_t                 stream = 0;
    int                          max_batch_size = 1;
    tiny_llm::DeviceBuffer<int>  d_tokens;   // token ids 上传缓冲（gather 需 device 指针）
    tiny_llm::DeviceBuffer<int>  decode_len; // 任务 3.1：decode 可见 KV 长度 (device int)
    tiny_llm::DeviceBuffer<int>  rope_pos;   // 任务 3.2：RoPE 起始位置 (device int)
    tiny_llm::DeviceBuffer<int>  d_next_tokens;  // normal greedy 的逐序列 device 输出
    tiny_llm::DeviceBuffer<half> d_batch_hidden; // [batch, hidden] 的末层状态
    tiny_llm::DeviceBuffer<half> d_batch_logits; // [batch, vocab] 的末端 logits
    std::unordered_map<int, SeqState> sequences;

    // ── 分页 KV（策略 1，max_num_blocks > 0）──
    bool  paged_kv = false;
    int   max_num_blocks = 0;
    int   max_visible_tokens = 0;
    int   block_size = 0;         // paged 模式下保存的分页块大小
    half *paged_k_pool = nullptr; // [L * max_num_blocks * block_size * kv_dim]
    half *paged_v_pool = nullptr;
    half *paged_k_scratch = nullptr; // [max_visible_tokens * kv_dim]
    half *paged_v_scratch = nullptr;
    tiny_llm::DeviceBuffer<int> d_block_tables; // 容量 max_num_blocks

    // 资源释放放在析构里：tinyllm_load 在任一中间步骤失败（GGUF 解析、权重
    // 加载、分页参数校验、KV cache 创建、cudaMalloc 异常）都会直接返回
    // nullptr，裸指针与 ModelWeights 没有其他所有者，旧实现会把已分配的
    // GPU 内存永久泄漏。unique_ptr<TinyLlmHandleImpl> 在失败路径离开作用域
    // 时由本析构统一回收。
    ~TinyLlmHandleImpl() {
        // FFI decode graph（exec 先于 graph 销毁）
        if (ffi_graph_exec != nullptr) cudaGraphExecDestroy(ffi_graph_exec);
        if (ffi_decode_graph != nullptr) cudaGraphDestroy(ffi_decode_graph);
        if (g_hidden_row) cudaFree(g_hidden_row);
        if (rope_cos) cudaFree(rope_cos);
        if (rope_sin) cudaFree(rope_sin);
        if (hidden_buf) cudaFree(hidden_buf);
        if (logits_buf) cudaFree(logits_buf);
        if (paged_k_pool) cudaFree(paged_k_pool);
        if (paged_v_pool) cudaFree(paged_v_pool);
        if (paged_k_scratch) cudaFree(paged_k_scratch);
        if (paged_v_scratch) cudaFree(paged_v_scratch);
        if (stream) cudaStreamDestroy(stream);
        tiny_llm::ModelLoader::freeWeights(weights);
    }

    // ── FFI decode CUDA Graph（仅策略 1 的 decode；TLLM_FFI_CUDA_GRAPHS=0 关闭）──
    // 设计：所有 graph 固化的地址均为 handle 级共享缓冲（token_in / rope_pos /
    // decode_len / d_block_tables / g_hidden_row），graph 只绑定地址不绑定
    // 内容，因此同一张 graph 可服务任意 seq_id 的 decode——序列差异完全由
    // 每步 H2D 上传的内容表达。策略 2（连续 KV）的 appendKV 以 host seq_id
    // 为参数，不可重放，不在本机制范围内。
    bool                        ffi_graphs_enabled = false;
    bool                        ffi_graph_captured = false;
    cudaGraph_t                 ffi_decode_graph = nullptr;
    cudaGraphExec_t             ffi_graph_exec = nullptr;
    tiny_llm::DeviceBuffer<int> g_token_in;             // [1] decode 输入 token（地址固定）
    tiny_llm::DeviceBuffer<int> g_decode_len;           // [1] decode 可见长度（append 后）
    tiny_llm::DeviceBuffer<int> g_rope_pos;             // [1] decode 绝对位置
    half                       *g_hidden_row = nullptr; // [hidden] graph 固定工作行
    // H2D 源（成员变量保证 host 指针跨调用稳定）
    int g_tok_host = 0;
    int g_len_host = 0;
    int g_pos_host = 0;

    // ── DEV-12 decode 行批量前向（TLLM_BATCH_DECODE=1，opt-in eager）──
    // 行数上限与 kernel 模板实例一致（{2,4,8}）；eager 路径不建/不碰 graph。
    tiny_llm::DeviceBuffer<int>  d_batch_token_in;     // [8]
    tiny_llm::DeviceBuffer<int>  d_batch_rope_pos;     // [8]
    tiny_llm::DeviceBuffer<int>  d_batch_decode_len;   // [8]
    tiny_llm::DeviceBuffer<int>  d_batch_block_tables; // [8 * max_num_blocks]
    tiny_llm::DeviceBuffer<half> d_batch_work;         // [8 * hidden]
};

// ── 内部工具 ─────────────────────────────────────────────

void set_err(char *err_buf, int err_buf_len, const std::string &msg) {
    if (err_buf && err_buf_len > 0) {
        std::strncpy(err_buf, msg.c_str(), static_cast<size_t>(err_buf_len) - 1);
        err_buf[err_buf_len - 1] = '\0';
    }
}

void embed(TinyLlmHandleImpl *h, const int *tokens, int num_tokens, half *output) {
    // gather_embeddings 的 tokens 是 device 指针：先上传主机 token ids
    if (h->d_tokens.size() < static_cast<size_t>(num_tokens)) {
        h->d_tokens = tiny_llm::DeviceBuffer<int>(static_cast<size_t>(num_tokens));
    }
    h->d_tokens.copyFromHost(tokens, static_cast<size_t>(num_tokens), h->stream);
    tiny_llm::kernels::gather_embeddings(h->d_tokens.data(), h->weights.token_embedding, output,
                                         num_tokens, h->config.hidden_dim, h->config.vocab_size,
                                         h->stream);
}

/// final norm + lm_head + greedy 采样，返回下一 token id。
int sample_from_hidden(TinyLlmHandleImpl *h, half *hidden) {
    // 任务 4.3：final norm + lm_head 走共享 helper（与 InferenceEngine 一致）
    tiny_llm::finalNormAndComputeLogits(hidden, h->weights, h->config, h->logits_buf, h->stream);
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    std::vector<half> h_logits(static_cast<size_t>(h->config.vocab_size));
    CUDA_CHECK(cudaMemcpy(h_logits.data(), h->logits_buf,
                          static_cast<size_t>(h->config.vocab_size) * sizeof(half),
                          cudaMemcpyDeviceToHost));
    return tiny_llm::InferenceEngine::sampleGreedy(h_logits.data(), h->config.vocab_size);
}

// 每个序列 layer forward 完成后立即保存末层 hidden，避免同一 shared hidden_buf 被
// 下一序列覆盖；同一 stream 保证后续 batch postprocess 读取到完整结果。
void queue_hidden_for_batch(TinyLlmHandleImpl *h, half *hidden, int batch_index) {
    const size_t offset = static_cast<size_t>(batch_index) * h->config.hidden_dim;
    CUDA_CHECK(cudaMemcpyAsync(h->d_batch_hidden.data() + offset, hidden,
                               static_cast<size_t>(h->config.hidden_dim) * sizeof(half),
                               cudaMemcpyDeviceToDevice, h->stream));
}

// 策略 1 decode 的可捕获 device 路径：embed → 逐层 forwardPaged。
// 只含 kernel/memcpy（可捕获）；块表上传、参数 H2D、hidden 出列、
// position 推进均在 capture 之外由调用方驱动。
// 要求 view.position_dev / view.decode_len 指向 graph 稳定 device int，
// 使 scatter/gather 的位置与可见长度在重放时从 device 内存读取。
tiny_llm::Result<void> runFfiPagedDecode(TinyLlmHandleImpl                *h,
                                         const tiny_llm::PagedKVCacheView &view) {
    tiny_llm::kernels::gather_embeddings(h->g_token_in.data(), h->weights.token_embedding,
                                         h->g_hidden_row, 1, h->config.hidden_dim,
                                         h->config.vocab_size, h->stream);
    for (auto &layer : h->layers) {
        auto r = layer->forwardPaged(h->g_hidden_row, view, 1, h->g_rope_pos.data(), h->rope_cos,
                                     h->rope_sin, h->stream);
        if (r.isErr()) {
            return r;
        }
    }
    return tiny_llm::Result<void>::ok();
}

// ── DEV-12 decode 行批量前向（策略 1 纯 decode、logprobs_k==0、B∈{2,4,8}）──
// 行压缩在 tinyllm_step 预扫描完成；本函数负责 metadata 上传、B 行 embed、
// 逐层 forwardPagedBatch、末层 hidden 映回原槽。kernel/device 级失败为
// 调用级失败：返回 err 时本批所有行记失败（与 ABI v3 单调用语义一致）。
struct FfiBatchDecodeRow {
    int        s;      // 原始输入槽位
    int        seq_id;
    int        pos;    // decode 绝对位置
    int        nb;     // 本行块表长度
    const int *bt;     // host 块表切片
    int        token;
};

constexpr int kFfiBatchDecodeMax = 8;

tiny_llm::Result<void> runFfiPagedDecodeBatch(TinyLlmHandleImpl                     *h,
                                              const std::vector<FfiBatchDecodeRow> &rows) {
    const int    B = static_cast<int>(rows.size());
    const int    hidden = h->config.hidden_dim;
    const size_t bt_stride = static_cast<size_t>(h->max_num_blocks);
    if (h->d_batch_token_in.size() < kFfiBatchDecodeMax) {
        h->d_batch_token_in = tiny_llm::DeviceBuffer<int>(kFfiBatchDecodeMax);
        h->d_batch_rope_pos = tiny_llm::DeviceBuffer<int>(kFfiBatchDecodeMax);
        h->d_batch_decode_len = tiny_llm::DeviceBuffer<int>(kFfiBatchDecodeMax);
        h->d_batch_block_tables =
            tiny_llm::DeviceBuffer<int>(static_cast<size_t>(kFfiBatchDecodeMax) * bt_stride);
        h->d_batch_work = tiny_llm::DeviceBuffer<half>(
            static_cast<size_t>(kFfiBatchDecodeMax) * static_cast<size_t>(hidden));
    }

    int toks[kFfiBatchDecodeMax], poss[kFfiBatchDecodeMax], lens[kFfiBatchDecodeMax];
    for (int r = 0; r < B; ++r) {
        toks[r] = rows[r].token;
        poss[r] = rows[r].pos;
        lens[r] = rows[r].pos + 1;
    }
    h->d_batch_token_in.copyFromHost(toks, static_cast<size_t>(B), h->stream);
    h->d_batch_rope_pos.copyFromHost(poss, static_cast<size_t>(B), h->stream);
    h->d_batch_decode_len.copyFromHost(lens, static_cast<size_t>(B), h->stream);
    for (int r = 0; r < B; ++r) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_batch_block_tables.data() + r * bt_stride, rows[r].bt,
                                   static_cast<size_t>(rows[r].nb) * sizeof(int),
                                   cudaMemcpyHostToDevice, h->stream));
    }

    tiny_llm::kernels::gather_embeddings(h->d_batch_token_in.data(), h->weights.token_embedding,
                                         h->d_batch_work.data(), B, hidden, h->config.vocab_size,
                                         h->stream);

    std::vector<tiny_llm::PagedKVCacheView> views(static_cast<size_t>(B));
    for (int r = 0; r < B; ++r) {
        auto &v = views[static_cast<size_t>(r)];
        v.k_pool = h->paged_k_pool;
        v.v_pool = h->paged_v_pool;
        v.block_table = h->d_batch_block_tables.data() + r * bt_stride;
        v.k_scratch = h->paged_k_scratch;
        v.v_scratch = h->paged_v_scratch;
        v.visible_blocks = rows[r].nb;
        v.block_size = h->block_size;
        v.max_num_blocks = h->max_num_blocks;
        v.max_visible_tokens = h->max_visible_tokens;
        v.position = rows[r].pos; // host 校验与 visible_blocks 推导用
        v.decode_len = h->d_batch_decode_len.data() + r;
        v.position_dev = nullptr; // scatter 位置走 d_batch_rope_pos（forwardPagedBatch 参数）
    }

    for (auto &layer : h->layers) {
        auto r = layer->forwardPagedBatch(h->d_batch_work.data(), views.data(), B,
                                          h->d_batch_rope_pos.data(), h->rope_cos, h->rope_sin,
                                          h->stream);
        if (r.isErr()) {
            return r;
        }
    }
    // 末层 hidden 映回原始槽位（与 queue_hidden_for_batch 同一 stream 语义）
    for (int r = 0; r < B; ++r) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_batch_hidden.data() +
                                       static_cast<size_t>(rows[r].s) * hidden,
                                   h->d_batch_work.data() + static_cast<size_t>(r) * hidden,
                                   static_cast<size_t>(hidden) * sizeof(half),
                                   cudaMemcpyDeviceToDevice, h->stream));
        h->sequences[rows[r].seq_id].position = rows[r].pos + 1;
    }
    return tiny_llm::Result<void>::ok();
}

// 策略 1 decode：graph 已捕获则重放；未捕获则先直接执行本步，再捕获并实例化
// （capture 不执行 kernel，本步结果来自直接执行）。任何捕获失败都关闭
// ffi_graphs_enabled 回退常规路径，不让 capture 异常毒化后续步骤。
tiny_llm::Result<void> runFfiPagedDecodeGraphed(TinyLlmHandleImpl                *h,
                                                const tiny_llm::PagedKVCacheView &view) {
    if (h->ffi_graphs_enabled && h->ffi_graph_captured) {
        CUDA_CHECK(cudaGraphLaunch(h->ffi_graph_exec, h->stream));
        return tiny_llm::Result<void>::ok();
    }
    if (!h->ffi_graphs_enabled) {
        return runFfiPagedDecode(h, view);
    }

    // 首次 decode：直接执行本步，同步后捕获一张新 graph。
    auto direct = runFfiPagedDecode(h, view);
    if (direct.isErr()) {
        return direct;
    }
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    bool capture_ok = false;
    try {
        cudaError_t begin_err = cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeThreadLocal);
        if (begin_err == cudaSuccess) {
            auto        capture_result = runFfiPagedDecode(h, view);
            cudaError_t end_err = cudaStreamEndCapture(h->stream, &h->ffi_decode_graph);
            capture_ok = capture_result.isOk() && end_err == cudaSuccess;
            if (!capture_ok && h->ffi_decode_graph != nullptr) {
                cudaGraphDestroy(h->ffi_decode_graph);
                h->ffi_decode_graph = nullptr;
            }
        }
    } catch (const tiny_llm::CudaException &) {
        // capture 期间的 CUDA 异常必须结束 capture，否则 stream 停在 capture 态。
        cudaStreamEndCapture(h->stream, &h->ffi_decode_graph);
        if (h->ffi_decode_graph != nullptr) {
            cudaGraphDestroy(h->ffi_decode_graph);
            h->ffi_decode_graph = nullptr;
        }
        capture_ok = false;
    }

    if (!capture_ok) {
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] paged decode graph capture failed; falling back\n");
        }
        h->ffi_graphs_enabled = false;
        return tiny_llm::Result<void>::ok(); // 本步已由直接执行完成
    }
    cudaError_t inst_err =
        cudaGraphInstantiate(&h->ffi_graph_exec, h->ffi_decode_graph, nullptr, nullptr, 0);
    if (inst_err != cudaSuccess) {
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] paged decode graph instantiate failed: %s\n",
                    cudaGetErrorString(inst_err));
        }
        cudaGraphDestroy(h->ffi_decode_graph);
        h->ffi_decode_graph = nullptr;
        h->ffi_graphs_enabled = false;
        return tiny_llm::Result<void>::ok();
    }
    h->ffi_graph_captured = true;
    if (std::getenv("TLLM_FFI_DEBUG")) {
        fprintf(stderr, "  [ffi] paged decode graph captured\n");
    }
    return tiny_llm::Result<void>::ok();
}

/// 对最后一步采样的 logits 计算 top-k (token_id, logprob)，写入交错数组。
/// out 布局：s * logprobs_k * 2 起，每候选 (token_id, logprob)。
void compute_logprobs(TinyLlmHandleImpl *h, int seq_idx, int logprobs_k, float *out) {
    if (logprobs_k <= 0 || out == nullptr) return;

    std::vector<half> h_logits(static_cast<size_t>(h->config.vocab_size));
    CUDA_CHECK(cudaMemcpy(h_logits.data(), h->logits_buf,
                          static_cast<size_t>(h->config.vocab_size) * sizeof(half),
                          cudaMemcpyDeviceToHost));

    // softmax（数值稳定）
    float max_l = -1e30f;
    for (const auto &v : h_logits)
        max_l = std::max(max_l, __half2float(v));
    double             sum_exp = 0.0;
    std::vector<float> probs(static_cast<size_t>(h->config.vocab_size));
    for (size_t i = 0; i < h_logits.size(); ++i) {
        probs[i] = std::exp(__half2float(h_logits[i]) - max_l);
        sum_exp += probs[i];
    }
    for (auto &p : probs)
        p = static_cast<float>(p / sum_exp);

    // top-k 选择（插入排序，k 通常很小）
    const int                          k = logprobs_k;
    std::vector<std::pair<int, float>> top;
    for (int v = 0; v < h->config.vocab_size; ++v) {
        if (probs[v] <= 0.0f) continue;
        std::pair<int, float> cand{v, probs[v]};
        bool                  inserted = false;
        for (size_t i = 0; i < top.size(); ++i) {
            if (cand.second > top[i].second) {
                top.insert(top.begin() + static_cast<long>(i), cand);
                inserted = true;
                break;
            }
        }
        if (!inserted && top.size() < static_cast<size_t>(k)) top.push_back(cand);
        if (top.size() > static_cast<size_t>(k)) top.pop_back();
    }

    float *dst = out + static_cast<size_t>(seq_idx) * k * 2;
    for (int i = 0; i < k; ++i) {
        if (i < static_cast<int>(top.size())) {
            dst[i * 2] = static_cast<float>(top[i].first);
            dst[i * 2 + 1] = std::log(top[i].second);
        } else {
            dst[i * 2] = -1.0f;
            dst[i * 2 + 1] = 0.0f;
        }
    }
}

} // namespace

// ── C ABI 入口 ───────────────────────────────────────────

extern "C" {

TinyLlmHandle *tinyllm_load(const char *model_path, const TinyLlmConfig *config, char *err_buf,
                            int err_buf_len) {
    if (model_path == nullptr) {
        set_err(err_buf, err_buf_len, "tinyllm_load: null model_path");
        return nullptr;
    }
    try {
        auto h = std::make_unique<TinyLlmHandleImpl>();

        // 解析 GGUF 提取真实模型配置
        tiny_llm::GGUFParser parser(model_path);
        auto                 parse_result = parser.parse();
        if (parse_result.isErr()) {
            set_err(err_buf, err_buf_len, "GGUF parse: " + parse_result.error());
            return nullptr;
        }
        auto cfg_result = parser.extractModelConfig();
        if (cfg_result.isErr()) {
            set_err(err_buf, err_buf_len, "config: " + cfg_result.error());
            return nullptr;
        }
        h->config = cfg_result.value();

        // C ABI 兼容契约：6 个几何字段允许传 0（表示“由 GGUF 提取”），
        // 非 0 时必须与 GGUF 一致。旧实现完全忽略这些字段，调用方以为
        // 自己指定了模型几何、实际却按文件里的另一套配置运行。
        if (config != nullptr) {
            struct DimField {
                const char *name;
                int         passed;
                int         actual;
            };
            const DimField dim_fields[] = {
                {"hidden_dim", config->hidden_dim, h->config.hidden_dim},
                {"num_layers", config->num_layers, h->config.num_layers},
                {"num_heads", config->num_heads, h->config.num_heads},
                {"num_kv_heads", config->num_kv_heads, h->config.num_kv_heads},
                {"head_dim", config->head_dim, h->config.head_dim},
                {"vocab_size", config->vocab_size, h->config.vocab_size},
            };
            for (const auto &f : dim_fields) {
                if (f.passed != 0 && f.passed != f.actual) {
                    set_err(err_buf, err_buf_len,
                            std::string("tinyllm_load: config.") + f.name + "=" +
                                std::to_string(f.passed) + " does not match GGUF value " +
                                std::to_string(f.actual));
                    return nullptr;
                }
            }
        }

        // 修复：C ABI 此前不校验模型几何，而 attention kernel 中
        // kv_head(q_head) = q_head / (num_heads / num_kv_heads) 是整数除法。
        // num_heads 不被 num_kv_heads 整除时 kv_head 会超出 [0, num_kv_heads)，
        // 最后一个 token 的 K/V 读越过缓冲末尾——Compute Sanitizer 可复现的
        // illegal address，会毒化整个 CUDA 上下文（不是仅这一次请求失败）。
        // GGUFParser::extractModelConfig 对这类元数据是"补默认值"而非报错，
        // 因此必须在进入 kernel 前显式拒绝。InferenceEngine::Load 早已校验同一组
        // 条件（Validator::validateModelConfig），此处补齐 C ABI 路径。
        auto config_valid = tiny_llm::Validator::validateModelConfig(h->config);
        if (config_valid.isErr()) {
            set_err(err_buf, err_buf_len,
                    "tinyllm_load: invalid model config: " + config_valid.error());
            return nullptr;
        }

        // 加载权重（GGUF -> W8A16/FP16 上传 GPU）
        auto w_result = tiny_llm::ModelLoader::loadGGUF(model_path, h->config);
        if (w_result.isErr()) {
            set_err(err_buf, err_buf_len, "load weights: " + w_result.error());
            return nullptr;
        }
        h->weights = std::move(w_result.value());

        // KV 生命周期（多序列，max_batch_size 由调用方配置）。
        int max_batch = config ? std::max(1, config->max_batch_size) : 1;
        h->max_batch_size = max_batch;

        const bool use_paged = (config != nullptr && config->max_num_blocks > 0);
        if (use_paged) {
            // 策略 1：分页 KV 池。校验 + 分配 K/V pool 与 scratch；跳过
            // KVCacheManager（h->kv_cache 保持 nullptr），块表由调用方传入。
            if (config->block_size <= 0 || config->max_num_blocks <= 0) {
                set_err(err_buf, err_buf_len,
                        "tinyllm_load: paged KV requires block_size > 0 and max_num_blocks > 0");
                return nullptr;
            }
            h->paged_kv = true;
            h->max_num_blocks = config->max_num_blocks;
            h->max_visible_tokens = config->max_num_blocks * config->block_size;
            h->block_size = config->block_size;
            const int    kv_dim = h->config.num_kv_heads * h->config.head_dim;
            const size_t pool_elems =
                static_cast<size_t>(h->config.num_layers) * static_cast<size_t>(h->max_num_blocks) *
                static_cast<size_t>(config->block_size) * static_cast<size_t>(kv_dim);
            const size_t scratch_elems =
                static_cast<size_t>(h->max_visible_tokens) * static_cast<size_t>(kv_dim);
            CUDA_CHECK(cudaMalloc(&h->paged_k_pool, pool_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_v_pool, pool_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_k_scratch, scratch_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_v_scratch, scratch_elems * sizeof(half)));
            h->d_block_tables = tiny_llm::DeviceBuffer<int>(static_cast<size_t>(h->max_num_blocks));

            // 策略 1 decode 的固定地址缓冲：无论是否启用 graph，decode 路径
            // 都经 g_* 缓冲执行（graph 只是其上的捕获/重放优化）。
            h->g_token_in = tiny_llm::DeviceBuffer<int>(1);
            h->g_decode_len = tiny_llm::DeviceBuffer<int>(1);
            h->g_rope_pos = tiny_llm::DeviceBuffer<int>(1);
            CUDA_CHECK(cudaMalloc(&h->g_hidden_row,
                                  static_cast<size_t>(h->config.hidden_dim) * sizeof(half)));

            // TLLM_FFI_CUDA_GRAPHS=0 显式关闭 decode graph 捕获/重放；
            // 常规 device 路径不受影响。
            const char *graph_env = std::getenv("TLLM_FFI_CUDA_GRAPHS");
            h->ffi_graphs_enabled =
                !(graph_env != nullptr && graph_env[0] == '0' && graph_env[1] == '\0');
        } else {
            // 策略 2：连续 KV（KVCacheManager）
            tiny_llm::KVCacheConfig kv_config;
            kv_config.num_layers = h->config.num_layers;
            kv_config.num_kv_heads = h->config.num_kv_heads;
            kv_config.head_dim = h->config.head_dim;
            kv_config.max_seq_len = h->config.max_seq_len;
            kv_config.max_batch_size = max_batch;
            auto kv_result = tiny_llm::KVCacheManager::create(kv_config);
            if (kv_result.isErr()) {
                set_err(err_buf, err_buf_len, "KV cache: " + kv_result.error());
                return nullptr;
            }
            h->kv_cache = std::move(kv_result.value());
        }

        // 共享激活工作区 + 层
        h->workspace.allocate(h->config);
        h->layers.reserve(static_cast<size_t>(h->config.num_layers));
        for (int i = 0; i < h->config.num_layers; ++i) {
            h->layers.push_back(std::make_unique<tiny_llm::TransformerLayer>(
                i, h->weights.layers[i], h->config, &h->workspace));
        }

        // 任务 3.1/3.2：decode 可见长度与 RoPE 位置（device int，graph 前置）
        h->decode_len = tiny_llm::DeviceBuffer<int>(1);
        h->rope_pos = tiny_llm::DeviceBuffer<int>(1);
        // 修复（跨流数据竞争）：先创建工作流，RoPE 预计算在其上 launch 并同步。
        // 此前预计算在 cudaStreamCreate 之前执行（h->stream 尚为默认流 0），
        // 而末尾 sync 的是新流，等于从未等待预计算完成——首次推理可能读到
        // 未写完的 cos/sin 表。
        CUDA_CHECK(cudaStreamCreate(&h->stream));

        // RoPE cache + hidden/logits 缓冲
        int half_d = h->config.head_dim / 2;
        CUDA_CHECK(cudaMalloc(&h->rope_cos,
                              static_cast<size_t>(h->config.max_seq_len) * half_d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h->rope_sin,
                              static_cast<size_t>(h->config.max_seq_len) * half_d * sizeof(float)));
        tiny_llm::kernels::rope_precompute_cache(h->rope_cos, h->rope_sin, h->config.max_seq_len,
                                                 h->config.head_dim, h->config.rope_theta,
                                                 h->stream);
        CUDA_CHECK(cudaMalloc(&h->hidden_buf, static_cast<size_t>(h->config.max_seq_len) *
                                                  h->config.hidden_dim * sizeof(half)));
        CUDA_CHECK(
            cudaMalloc(&h->logits_buf, static_cast<size_t>(h->config.vocab_size) * sizeof(half)));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));

        return reinterpret_cast<TinyLlmHandle *>(h.release());
    } catch (const std::exception &e) {
        set_err(err_buf, err_buf_len, std::string("tinyllm_load: ") + e.what());
        return nullptr;
    }
}

int tinyllm_allocate_sequence(TinyLlmHandle *handle, int seq_id, int num_tokens) {
    if (handle == nullptr || num_tokens <= 0) return TLLM_ERR;
    auto *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    try {
        if (h->paged_kv) {
            // 策略 1：分页 KV 由调用方管理块表；这里只登记序列。
            // 修复：重复分配同一 seq_id 会覆盖登记并泄漏旧序列状态，显式拒绝。
            if (h->sequences.count(seq_id) != 0) {
                if (std::getenv("TLLM_FFI_DEBUG")) {
                    fprintf(stderr, "  [ffi] allocate_sequence(%d) already allocated\n", seq_id);
                }
                return TLLM_ERR;
            }
            SeqState st;
            st.allocated = true;
            h->sequences[seq_id] = st;
            if (std::getenv("TLLM_FFI_DEBUG")) {
                fprintf(stderr, "  [ffi] paged allocate_sequence(%d, %d) ok\n", seq_id, num_tokens);
            }
            return TLLM_OK;
        }
        auto alloc = h->kv_cache->allocateSequence(seq_id, num_tokens);
        if (alloc.isErr()) {
            if (std::getenv("TLLM_FFI_DEBUG")) {
                fprintf(stderr, "  [ffi] allocate_sequence(%d,%d) fail: %s\n", seq_id, num_tokens,
                        alloc.error().c_str());
            }
            return TLLM_ERR;
        }
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] allocate_sequence(%d, %d) ok\n", seq_id, num_tokens);
        }
        SeqState st;
        st.allocated = true;
        h->sequences[seq_id] = st;
        return TLLM_OK;
    } catch (const std::exception &e) {
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] allocate_sequence(%d) threw: %s\n", seq_id, e.what());
        }
        return TLLM_ERR;
    }
}

int tinyllm_free_sequence(TinyLlmHandle *handle, int seq_id) {
    if (handle == nullptr) return TLLM_ERR;
    auto *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    try {
        if (h->paged_kv) {
            // 策略 1：无连续 KV 可释放，仅移除序列登记。
            h->sequences.erase(seq_id);
            return TLLM_OK;
        }
        auto rel = h->kv_cache->releaseSequence(seq_id);
        h->sequences.erase(seq_id);
        return rel.isErr() ? TLLM_ERR : TLLM_OK;
    } catch (...) {
        return TLLM_ERR;
    }
}

int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                 const int *positions, const int *seq_lens, const int *block_tables,
                 const int *num_blocks, const unsigned char *is_prefill, int num_sequences,
                 int *next_tokens, float *logprobs, int logprobs_k, int *seq_status) {
    // ABI v3：seq_status 为必填出参；调用级违约才返回 TLLM_ERR，
    // 逐序列结果写入 seq_status（TLLM_SEQ_OK / TLLM_SEQ_ERR）。
    if (handle == nullptr || seq_ids == nullptr || input_tokens == nullptr || seq_lens == nullptr ||
        is_prefill == nullptr || next_tokens == nullptr || seq_status == nullptr ||
        num_sequences <= 0) {
        return TLLM_ERR;
    }
    auto     *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    const int hidden = h->config.hidden_dim;

    // logprobs_k 的范围与输出缓冲区必须和双源 ABI 契约一致。
    if (logprobs_k < 0 || logprobs_k > h->config.vocab_size ||
        (logprobs_k > 0 && logprobs == nullptr)) {
        return TLLM_ERR;
    }

    for (int s = 0; s < num_sequences; ++s) {
        seq_status[s] = TLLM_SEQ_ERR;
    }

    const bool batch_device_postprocess = logprobs_k == 0;
    if (batch_device_postprocess) {
        try {
            const size_t batch_size = static_cast<size_t>(num_sequences);
            if (h->d_next_tokens.size() < batch_size) {
                h->d_next_tokens = tiny_llm::DeviceBuffer<int>(batch_size);
            }
            if (h->d_batch_hidden.size() < batch_size * h->config.hidden_dim) {
                h->d_batch_hidden = tiny_llm::DeviceBuffer<half>(batch_size * h->config.hidden_dim);
            }
            if (h->d_batch_logits.size() < batch_size * h->config.vocab_size) {
                h->d_batch_logits = tiny_llm::DeviceBuffer<half>(batch_size * h->config.vocab_size);
            }
        } catch (...) {
            return TLLM_ERR;
        }
    }

    // ── DEV-12：decode 行批量前向（TLLM_BATCH_DECODE=1 候选开关，默认关闭）──
    // 预扫描与主循环同一套 ABI 语义：收集策略 1 纯 decode、logprobs_k==0、
    // 校验通过的行；行数恰为 {2,4,8} 时整体走 forwardPagedBatch（eager，不建
    // graph），否则全部行回落既有逐序列路径，行为与开关关闭完全一致。
    std::vector<char> batch_handled(static_cast<size_t>(num_sequences), 0);
    bool              batch_ok = false;
    {
        const char *bd_env = std::getenv("TLLM_BATCH_DECODE");
        const bool  batch_decode_enabled =
            h->paged_kv && batch_device_postprocess && bd_env != nullptr &&
            std::string(bd_env) == "1";
        if (batch_decode_enabled) {
            std::vector<FfiBatchDecodeRow> rows;
            int                            off2 = 0, toff2 = 0;
            for (int s = 0; s < num_sequences; ++s) {
                const int len = seq_lens[s];
                if (len <= 0) continue;
                if (!is_prefill[s] && len != 1) continue;
                const int *toks = input_tokens + off2;
                off2 += len;
                const int *bt = nullptr;
                int        nb = 0;
                if (num_blocks != nullptr && block_tables != nullptr) {
                    nb = num_blocks[s];
                    if (nb > 0) {
                        bt = block_tables + toff2;
                        toff2 += nb;
                    }
                }
                if (is_prefill[s]) continue;
                auto it = h->sequences.find(seq_ids[s]);
                if (it == h->sequences.end() || !it->second.allocated) continue;
                const auto &st = it->second;
                const int   block_size = h->block_size;
                bool        bad = (bt == nullptr || nb <= 0 || nb > h->max_num_blocks);
                for (int i = 0; !bad && i < nb; ++i) {
                    bad = (bt[i] < 0 || bt[i] >= h->max_num_blocks);
                }
                const int need_blocks = (st.position + 1 + block_size - 1) / block_size;
                if (bad || nb < need_blocks || st.position >= h->config.max_seq_len) continue;
                rows.push_back({s, seq_ids[s], st.position, nb, bt, toks[0]});
            }
            const size_t bsz = rows.size();
            if (bsz == 2 || bsz == 4 || bsz == 8) {
                batch_ok = runFfiPagedDecodeBatch(h, rows).isOk();
                for (const auto &r : rows) batch_handled[static_cast<size_t>(r.s)] = 1;
                if (std::getenv("TLLM_FFI_DEBUG")) {
                    fprintf(stderr, "  [ffi] TLLM_BATCH_DECODE: batched %zu decode rows, ok=%d\n",
                            bsz, (int)batch_ok);
                }
            } else if (bsz > 0 && std::getenv("TLLM_FFI_DEBUG")) {
                fprintf(stderr,
                        "  [ffi] TLLM_BATCH_DECODE: %zu rows -> fallback (B not in {2,4,8})\n",
                        bsz);
            }
        }
    }

    int offset = 0;
    int table_offset = 0; // 扁平化 block_tables 中前面序列块数之和（策略 1）
    int ok_count = 0;
    for (int s = 0; s < num_sequences; ++s) {
        const int len = seq_lens[s];
        if (len <= 0) {
            continue; // 非法长度：本序列失败，不消耗 token 切片
        }
        // decode 必须恰好 1 个 token；ABI v3 下同样不消耗切片、仅记本序列
        // 失败（seq_status 保持 TLLM_SEQ_ERR），不株连 batch。
        if (!is_prefill[s] && len != 1) continue;
        const int seq_id = seq_ids[s];
        auto      it = h->sequences.find(seq_id);
        if (it == h->sequences.end() || !it->second.allocated) {
            continue; // 未分配/已释放的 seq_id：本序列失败
        }
        auto &st = it->second;

        const int *toks = input_tokens + offset;
        offset += len;

        bool seq_ok = false;
        try {
            if (std::getenv("TLLM_FFI_DEBUG") && !h->paged_kv) {
                auto c = h->kv_cache->getCache(seq_id, 0);
                fprintf(stderr, "  [ffi] seq %d hasSeq=%d cache0=%p kv_seq_len=%d\n", seq_id,
                        (int)h->kv_cache->hasSequence(seq_id), (void *)c.first,
                        h->kv_cache->getSeqLen(seq_id));
            }
            if (h->paged_kv) {
                // ── 策略 1：分页 KV ──
                // 先按调用方 num_blocks 推进 table_offset（扁平化布局对齐），
                // 再做本序列校验——校验失败不再连累后续序列的块表切片。
                const int *bt = nullptr;
                int        nb = 0;
                if (num_blocks != nullptr && block_tables != nullptr) {
                    nb = num_blocks[s];
                    if (nb > 0) {
                        bt = block_tables + table_offset;
                        table_offset += nb;
                    }
                }
                // 批量路径已处理（成功或调用级失败）的 decode 行：offset 与
                // table_offset 已按同一规则推进，这里只记状态不再重复执行。
                if (batch_handled[static_cast<size_t>(s)]) {
                    seq_ok = batch_ok;
                    goto seq_done;
                }
                if (bt == nullptr || nb <= 0 || nb > h->max_num_blocks) {
                    goto seq_done; // 块表缺失/块数非法
                }
                const int block_size = h->block_size;
                {
                    // host 侧块 id 值域校验：坏块表必须显式失败，而不是依赖
                    // kernel 的值域防护把错误吞成静默错误输出。
                    bool bad_block = false;
                    for (int i = 0; i < nb; ++i) {
                        if (bt[i] < 0 || bt[i] >= h->max_num_blocks) {
                            bad_block = true;
                            break;
                        }
                    }
                    const int cur_pos = is_prefill[s] ? 0 : st.position;
                    const int need_blocks = is_prefill[s]
                                                ? (len + block_size - 1) / block_size
                                                : (cur_pos + 1 + block_size - 1) / block_size;
                    if (bad_block || nb < need_blocks) {
                        goto seq_done;
                    }
                }

                // 块表上传（逐序列，同一 stream 上顺序执行）
                h->d_block_tables.copyFromHost(bt, static_cast<size_t>(nb), h->stream);

                // 构造视图
                tiny_llm::PagedKVCacheView view;
                view.k_pool = h->paged_k_pool;
                view.v_pool = h->paged_v_pool;
                view.block_table = h->d_block_tables.data();
                view.k_scratch = h->paged_k_scratch;
                view.v_scratch = h->paged_v_scratch;
                view.visible_blocks = nb;
                view.block_size = block_size;
                view.max_num_blocks = h->max_num_blocks;
                view.max_visible_tokens = h->max_visible_tokens;

                if (is_prefill[s]) {
                    // 修复：prefill 长度必须 <= max_seq_len，否则 hidden_buf /
                    // RoPE 表越界（hidden_buf 按 max_seq_len 分配）。
                    if (len > h->config.max_seq_len) goto seq_done;
                    // R5: 不支持对同一序列二次 prefill（KV 从位置 0 重写覆盖
                    // 旧上下文、current_len 双倍累加会静默错乱），显式拒绝。
                    if (st.position != 0) goto seq_done;
                    // Prefill：从绝对位置 0 写入整个 prompt
                    view.position = 0;
                    view.decode_len = nullptr;
                    const int zero_pos = 0;
                    h->rope_pos.copyFromHost(&zero_pos, 1, h->stream);
                    embed(h, toks, len, h->hidden_buf);
                    {
                        int  li = 0;
                        bool layer_ok = true;
                        for (auto &layer : h->layers) {
                            auto r =
                                layer->forwardPaged(h->hidden_buf, view, len, h->rope_pos.data(),
                                                    h->rope_cos, h->rope_sin, h->stream);
                            if (r.isErr()) {
                                if (std::getenv("TLLM_FFI_DEBUG")) {
                                    fprintf(stderr, "  [ffi] paged layer %d err: %s\n", li,
                                            r.error().c_str());
                                }
                                layer_ok = false;
                                break;
                            }
                            ++li;
                        }
                        if (!layer_ok) goto seq_done;
                    }
                    if (batch_device_postprocess) {
                        queue_hidden_for_batch(h, h->hidden_buf + (len - 1) * hidden, s);
                    } else {
                        next_tokens[s] = sample_from_hidden(h, h->hidden_buf + (len - 1) * hidden);
                    }
                    st.position = len;
                    seq_ok = true;
                } else {
                    // Decode：单个新 token（绝对位置 = st.position）。
                    // 工作行使用 graph 固定地址 g_hidden_row，scatter/gather
                    // 的位置与可见长度走 device int——同一张 graph 可重放
                    // 任意序列的 decode。
                    const int pos = st.position;
                    // 修复：绝对位置必须 < max_seq_len（RoPE 表越界防护）。
                    if (pos >= h->config.max_seq_len) goto seq_done;
                    h->g_tok_host = toks[0];
                    h->g_len_host = pos + 1;
                    h->g_pos_host = pos;
                    h->g_token_in.copyFromHost(&h->g_tok_host, 1, h->stream);
                    h->g_decode_len.copyFromHost(&h->g_len_host, 1, h->stream);
                    h->g_rope_pos.copyFromHost(&h->g_pos_host, 1, h->stream);
                    view.position = pos;                      // host 校验仍用
                    view.position_dev = h->g_rope_pos.data(); // scatter 走 device
                    view.decode_len = h->g_decode_len.data(); // gather/可见长度走 device
                    if (runFfiPagedDecodeGraphed(h, view).isErr()) goto seq_done;
                    if (batch_device_postprocess) {
                        queue_hidden_for_batch(h, h->g_hidden_row, s);
                    } else {
                        next_tokens[s] = sample_from_hidden(h, h->g_hidden_row);
                    }
                    st.position = pos + 1;
                    seq_ok = true;
                }
            } else if (is_prefill[s]) {
                // Prefill：处理完整 prompt，输出最后一个 hidden 的下一 token
                // 修复：prefill 长度必须 <= max_seq_len（hidden_buf 越界防护）。
                if (len > h->config.max_seq_len) goto seq_done;
                // R5: 不支持对同一序列二次 prefill（KV 从位置 0 重写覆盖旧
                // 上下文、advanceSeqLen 累加后 decode 错乱），显式拒绝。
                if (st.position != 0) goto seq_done;
                {
                    bool dbg = std::getenv("TLLM_FFI_DEBUG") != nullptr;
                    embed(h, toks, len, h->hidden_buf);
                    if (dbg) {
                        cudaError_t e = cudaGetLastError();
                        fprintf(stderr, "  [ffi] embed: %s\n", cudaGetErrorString(e));
                    }
                    // 任务 3.2：prefill 起始位置 0、append 写位置 = 当前可见长度
                    const int zero_pos = 0;
                    h->rope_pos.copyFromHost(&zero_pos, 1, h->stream);
                    h->kv_cache->setAppendPos(0, h->stream);
                    int  li = 0;
                    bool layer_ok = true;
                    for (auto &layer : h->layers) {
                        auto r = [&]() -> tiny_llm::Result<void> {
                            try {
                                return layer->forwardPrefill(h->hidden_buf, *h->kv_cache, seq_id,
                                                             len, h->rope_pos.data(), h->rope_cos,
                                                             h->rope_sin, h->stream);
                            } catch (const std::exception &e) {
                                if (std::getenv("TLLM_FFI_DEBUG")) {
                                    fprintf(stderr, "  [ffi] layer %d threw: %s\n", li, e.what());
                                }
                                return tiny_llm::Result<void>::err(std::string("threw: ") +
                                                                   e.what());
                            }
                        }();
                        if (r.isErr()) {
                            if (std::getenv("TLLM_FFI_DEBUG")) {
                                fprintf(stderr, "  [ffi] layer %d forwardPrefill err: %s\n", li,
                                        r.error().c_str());
                            }
                            layer_ok = false;
                            break;
                        }
                        if (dbg) {
                            cudaError_t e = cudaDeviceSynchronize();
                            fprintf(stderr, "  [ffi] layer %d: %s\n", li, cudaGetErrorString(e));
                            if (e != cudaSuccess) break;
                        }
                        ++li;
                    }
                    if (!layer_ok) goto seq_done;
                }
                if (h->kv_cache->advanceSeqLen(seq_id, len).isErr()) goto seq_done;
                if (batch_device_postprocess) {
                    queue_hidden_for_batch(h, h->hidden_buf + (len - 1) * hidden, s);
                } else {
                    next_tokens[s] = sample_from_hidden(h, h->hidden_buf + (len - 1) * hidden);
                }
                st.position = len;
                seq_ok = true;
            } else {
                // Decode：单个新 token（绝对位置由引擎跟踪，与 KV 写入一致）
                const int pos = st.position;
                // 修复：绝对位置必须 < max_seq_len（hidden_buf / RoPE 表越界防护）。
                if (pos >= h->config.max_seq_len) goto seq_done;
                embed(h, toks, 1, h->hidden_buf + static_cast<size_t>(pos) * hidden);
                half *token_state = h->hidden_buf + static_cast<size_t>(pos) * hidden;
                // 任务 3.1/3.2：把 appendKV 后的可见长度、RoPE 位置与
                // append 写位置写入 device 缓冲
                const int decode_len = h->kv_cache->getSeqLen(seq_id) + 1;
                h->decode_len.copyFromHost(&decode_len, 1, h->stream);
                h->rope_pos.copyFromHost(&pos, 1, h->stream);
                h->kv_cache->setAppendPos(h->kv_cache->getSeqLen(seq_id), h->stream);
                {
                    bool layer_ok = true;
                    for (auto &layer : h->layers) {
                        auto r = layer->forward(token_state, *h->kv_cache, seq_id, pos,
                                                h->decode_len.data(), h->rope_pos.data(),
                                                h->rope_cos, h->rope_sin, h->stream);
                        if (r.isErr()) {
                            layer_ok = false;
                            break;
                        }
                    }
                    if (!layer_ok) goto seq_done;
                }
                if (h->kv_cache->advanceSeqLen(seq_id, 1).isErr()) goto seq_done;
                if (batch_device_postprocess) {
                    queue_hidden_for_batch(h, token_state, s);
                } else {
                    next_tokens[s] = sample_from_hidden(h, token_state);
                }
                st.position = pos + 1;
                seq_ok = true;
            }
        } catch (...) {
            seq_ok = false;
        }

    seq_done:
        if (seq_ok) {
            seq_status[s] = TLLM_SEQ_OK;
            ++ok_count;
            if (logprobs_k > 0 && logprobs != nullptr) {
                compute_logprobs(h, s, logprobs_k, logprobs);
            }
        }
    }

    // 全部序列失败时跳过批量后处理（next_tokens 无有效槽位）。
    if (batch_device_postprocess && ok_count > 0) {
        try {
            tiny_llm::finalNormAndComputeLogitsBatch(h->d_batch_hidden.data(), num_sequences,
                                                     h->weights, h->config,
                                                     h->d_batch_logits.data(), h->stream);
            tiny_llm::kernels::greedy_argmax_batch(h->d_batch_logits.data(), num_sequences,
                                                   h->config.vocab_size, h->d_next_tokens.data(),
                                                   h->stream);
            CUDA_CHECK(cudaMemcpyAsync(next_tokens, h->d_next_tokens.data(),
                                       static_cast<size_t>(num_sequences) * sizeof(int),
                                       cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
        } catch (...) {
            return TLLM_ERR;
        }
    }

    // positions 当前两种策略均不使用（序列位置由后端 SeqState 内部跟踪；
    // 见 ffi.h"已声明但未生效的参数"）。block_tables/num_blocks 仅策略 1
    // 在循环内使用，策略 2 忽略。
    (void)positions;
    (void)block_tables;
    (void)num_blocks;
    return TLLM_OK;
}

void tinyllm_free(TinyLlmHandle *handle) {
    if (handle == nullptr) return;
    // 全部释放逻辑在 TinyLlmHandleImpl::~TinyLlmHandleImpl()，
    // 与 tinyllm_load 的失败路径共用同一套清理。
    delete reinterpret_cast<TinyLlmHandleImpl *>(handle);
}

} // extern "C"
