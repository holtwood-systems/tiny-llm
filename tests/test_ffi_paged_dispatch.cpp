// TLLM-P0-005：direct paged decode 经真实 C ABI 的差分门禁。
//
// 栈内已有证据停在 C++ 层（test_paged_direct.cpp kernel 级、
// test_paged_dispatch.cpp layer 级）；本文件把门禁推进到 C ABI：
// tinyllm_load → tinyllm_allocate_sequence → tinyllm_step(prefill+decode)
// → tinyllm_free_sequence。不依赖真实模型——合成 GGUF 由本文件按
// GGUF v3 规范构造（F16 tensor、确定权重），走的就是生产加载路径
// （parse → extractModelConfig → validateModelConfig → loadGGUF）。
//
// 断言分级（与 kernel/layer 级门禁口径一致）：
//  - legacy / direct / auto / splitkv=1 之间逐位等价 → 逐步 token id 严格相等；
//  - splitkv>1 允许 fp32 归约序差异（kernel 级实测 max|diff| ≈ 6e-5），
//    token id 可能在小间距处翻转——改为逐步比较完整输出概率分布
//    （|Δprob| ≤ 0.02），翻转不判负、分布被真实破坏则判负；
//  - decode 喂固定 token（解耦轨迹：单步 argmax 翻转不会污染后续步的可比性）。
//
// 其余验证点：策略 2（max_num_blocks == 0）不受开关影响；块表校验失败返回
// TLLM_ERR 且序列仍可用；TLLM_ATTN_SPLITKV 非法值在 decode 步干净失败。

#include "tiny_llm/ffi.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

// ABI v3: 逐序列状态出参（本文件测试统一用共享缓冲）
static int seq_status[256];

namespace {

bool hasCudaDevice() {
    static bool checked = false;
    static bool has_device = false;
    if (!checked) {
        int         n = 0;
        cudaError_t e = cudaGetDeviceCount(&n);
        has_device = (e == cudaSuccess && n > 0);
        checked = true;
    }
    return has_device;
}

// ── 合成 GGUF 构造 ─────────────────────────────────────────────
// 几何：hidden=128, layers=2, heads=4(GQA kv=2), head_dim=32,
// ffn=128, vocab=64, ctx=256。全部通过 validateModelConfig 与
// attention kernel 的 head_dim ∈ {32,64,128} 约束。
constexpr int kHidden = 128;
constexpr int kLayers = 2;
constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 32;
constexpr int kKvDim = kKvHeads * kHeadDim; // 64
constexpr int kInter = 128;
constexpr int kVocab = 64;
constexpr int kCtxLen = 256;

constexpr uint32_t U32_TYPE = 4;
constexpr uint32_t F32_TYPE = 6;
constexpr uint32_t STR_TYPE = 8;
constexpr uint32_t ARR_TYPE = 9;
constexpr uint32_t F16_GGML = 1;

void pushU32(std::vector<uint8_t> &b, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void pushU64(std::vector<uint8_t> &b, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void pushStr(std::vector<uint8_t> &b, const std::string &s) {
    pushU64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void pushU32Kv(std::vector<uint8_t> &b, const std::string &key, uint32_t value) {
    pushStr(b, key);
    pushU32(b, U32_TYPE);
    pushU32(b, value);
}
void pushF32Kv(std::vector<uint8_t> &b, const std::string &key, float value) {
    pushStr(b, key);
    pushU32(b, F32_TYPE);
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    pushU32(b, bits);
}
void pushStrKv(std::vector<uint8_t> &b, const std::string &key, const std::string &value) {
    pushStr(b, key);
    pushU32(b, STR_TYPE);
    pushStr(b, value);
}

// 确定性伪随机权重：splitmix32 哈希 -> [-0.08, 0.08]。
uint16_t f16Bits(uint64_t i) {
    uint32_t x = static_cast<uint32_t>(i) * 2654435761u;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    const float     v = ((x & 0xFFFF) / 65535.0f - 0.5f) * 0.16f;
    const __half    h = __float2half(v);
    const uint16_t *p = reinterpret_cast<const uint16_t *>(&h);
    return *p;
}

std::vector<uint8_t> f16TensorData(size_t n, uint64_t seed_salt, float constant = 0.0f) {
    std::vector<uint8_t> d(n * 2);
    for (size_t i = 0; i < n; ++i) {
        uint16_t bits;
        if (constant != 0.0f) {
            const __half h = __float2half(constant);
            bits = *reinterpret_cast<const uint16_t *>(&h);
        } else {
            bits = f16Bits(seed_salt + i);
        }
        d[2 * i] = static_cast<uint8_t>(bits & 0xFF);
        d[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return d;
}

struct Tensor {
    std::string           name;
    std::vector<uint64_t> dims; // GGUF 序：dims[0] = 最内层（= 矩阵输入维）
    std::vector<uint8_t>  data;
};

Tensor mkTensor(std::string name, std::vector<uint64_t> dims, size_t n, uint64_t salt,
                float constant = 0.0f) {
    return {std::move(name), std::move(dims), f16TensorData(n, salt, constant)};
}

// 组装完整 GGUF v3 二进制：header → kv → tensor infos → 32B 对齐 data 区。
std::vector<uint8_t> buildGguf(const std::vector<Tensor> &tensors) {
    constexpr uint64_t   kAlign = 32;
    std::vector<uint8_t> b;
    pushU32(b, 0x46554747u); // "GGUF"
    pushU32(b, 3);           // version
    pushU64(b, tensors.size());
    constexpr uint64_t kKvCount = 10;
    pushU64(b, kKvCount);

    pushStrKv(b, "general.architecture", "qwen2");
    pushU32Kv(b, "qwen2.embedding_length", kHidden);
    pushU32Kv(b, "qwen2.block_count", kLayers);
    pushU32Kv(b, "qwen2.attention.head_count", kHeads);
    pushU32Kv(b, "qwen2.attention.head_count_kv", kKvHeads);
    pushU32Kv(b, "qwen2.context_length", kCtxLen);
    pushU32Kv(b, "qwen2.feed_forward_length", kInter);
    pushF32Kv(b, "qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
    pushF32Kv(b, "qwen2.rope.freq_base", 10000.0f);
    // vocab_size 由 tokens 数组长度派生，必须 == kVocab
    pushStr(b, "tokenizer.ggml.tokens");
    pushU32(b, ARR_TYPE);
    pushU32(b, STR_TYPE);
    pushU64(b, kVocab);
    for (uint64_t i = 0; i < kVocab; ++i)
        pushStr(b, "t" + std::to_string(i));

    // tensor infos：offset 相对 data 区起点，逐个 32B 对齐
    uint64_t offset = 0;
    for (const auto &t : tensors) {
        pushStr(b, t.name);
        pushU32(b, static_cast<uint32_t>(t.dims.size()));
        for (uint64_t d : t.dims)
            pushU64(b, d);
        pushU32(b, F16_GGML);
        pushU64(b, offset);
        offset += t.data.size();
        offset = (offset + kAlign - 1) & ~(kAlign - 1);
    }

    // data 区起点按 32 对齐，再顺序写入 tensor 数据（保持声明的 offset）
    const size_t data_start = (b.size() + kAlign - 1) & ~(kAlign - 1);
    b.resize(data_start, 0);
    uint64_t cursor = 0;
    for (const auto &t : tensors) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        cursor += t.data.size();
        const uint64_t next = (cursor + kAlign - 1) & ~(kAlign - 1);
        b.resize(b.size() + (next - cursor), 0);
        cursor = next;
    }
    return b;
}

// 2 层 qwen2 最小模型：每层 9 个必需 tensor + 全局 2 个。
// 无 output.weight → lm_head 走 tied token_embd 路径。
std::vector<uint8_t> syntheticModelGguf() {
    std::vector<Tensor> ts;
    uint64_t            salt = 1;
    auto                rnd = [&](std::string name, std::vector<uint64_t> dims, size_t n) {
        ts.push_back(mkTensor(std::move(name), std::move(dims), n, salt++ * 7919));
    };
    auto ones = [&](std::string name, uint64_t n) {
        ts.push_back(mkTensor(std::move(name), {n}, n, 0, 1.0f));
    };

    rnd("token_embd.weight", {kHidden, kVocab}, kHidden * kVocab);
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ones(p + "attn_norm.weight", kHidden);
        rnd(p + "attn_q.weight", {kHidden, kHidden}, kHidden * kHidden);
        rnd(p + "attn_k.weight", {kHidden, kKvDim}, kHidden * kKvDim);
        rnd(p + "attn_v.weight", {kHidden, kKvDim}, kHidden * kKvDim);
        rnd(p + "attn_output.weight", {kHidden, kHidden}, kHidden * kHidden);
        ones(p + "ffn_norm.weight", kHidden);
        rnd(p + "ffn_gate.weight", {kHidden, kInter}, kHidden * kInter);
        rnd(p + "ffn_down.weight", {kInter, kHidden}, kInter * kHidden);
        rnd(p + "ffn_up.weight", {kHidden, kInter}, kHidden * kInter);
    }
    ones("output_norm.weight", kHidden);
    return buildGguf(ts);
}

std::string writeTempFile(const std::vector<uint8_t> &buf, const char *tag) {
    const std::string path =
        std::string("/tmp/tiny_llm_ffidisp_") + tag + "_" + std::to_string(::getpid()) + ".gguf";
    std::ofstream f(path, std::ios::binary);
    EXPECT_TRUE(f.is_open()) << "cannot write " << path;
    f.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    f.close();
    return path;
}

// ── env 守护（dispatch 每次调用解析 env，不缓存，见 PR-3 设计）──
class EnvGuard {
  public:
    explicit EnvGuard(const char *name) : name_(name) {
        const char *v = std::getenv(name);
        if (v) {
            had_ = true;
            old_ = v;
        }
    }
    ~EnvGuard() { restore(); }
    void set(const char *v) { ::setenv(name_.c_str(), v, 1); }
    void unset() { ::unsetenv(name_.c_str()); }
    void restore() {
        if (had_) {
            ::setenv(name_.c_str(), old_.c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

  private:
    std::string name_, old_;
    bool        had_ = false;
};

// ── 固定输入：prefill 48 token（3 块）+ 20 步 decode ──
// decode 第 17 步（visible 65）跨进第 5 块：块表在 decode 期间增长，
// 正是生产路径的常态事件。
constexpr int kPromptLen = 48;
constexpr int kDecodeSteps = 20;
constexpr int kBlockSize = 16;
constexpr int kMaxBlocks = 32;
constexpr int kFeedToken = 7; // decode 步固定喂入（轨迹与 env 解耦）

std::vector<int> promptTokens() {
    std::vector<int> t(kPromptLen);
    for (int i = 0; i < kPromptLen; ++i)
        t[i] = static_cast<int>((i * 7 + 3) % kVocab);
    return t;
}
std::vector<int> promptPositions() {
    std::vector<int> p(kPromptLen);
    for (int i = 0; i < kPromptLen; ++i)
        p[i] = i;
    return p;
}

// 非连续物理块表：暴露"逻辑窗口 ≠ 物理布局"的取址差异
const std::vector<int> kBlocks = {7, 2, 11, 5, 9};

struct HandleDeleter {
    void operator()(TinyLlmHandle *h) const {
        if (h) tinyllm_free(h);
    }
};
using HandlePtr = std::unique_ptr<TinyLlmHandle, HandleDeleter>;

HandlePtr loadPaged(const char *path) {
    char           err[512] = {0};
    TinyLlmConfig  cfg = {0, 0, 0, 0, 0, 0, kBlockSize, 2, kMaxBlocks};
    TinyLlmHandle *h = tinyllm_load(path, &cfg, err, sizeof(err));
    EXPECT_NE(h, nullptr) << "tinyllm_load failed: " << err;
    return HandlePtr(h);
}

// 单次 decode 步（策略 1，可选 logprobs）。
// ABI v3：positions 为惰性参数（后端 SeqState 内部跟踪），传值仅作记录。
int decodeStep(TinyLlmHandle *h, int tok, int nb, int pos, int *next, float *lp, int lp_k) {
    int           seq_ids[] = {0}, len1[] = {1};
    unsigned char dec[] = {0};
    return tinyllm_step(h, seq_ids, &tok, &pos, len1, kBlocks.data(), &nb, dec, 1, next, lp, lp_k,
                        seq_status);
}

struct SeqOut {
    std::vector<int>                  tokens;
    std::vector<std::map<int, float>> probs;    // 每步 id->prob（lp_k>0 时非空）
    std::vector<int>                  steps_rc; // 每步返回值（诊断用）
};

// 跑一次完整序列：prefill 48 token（3 块），decode 固定喂 kFeedToken、
// 块表随可见窗口增长（第 17 步跨进第 5 块）。lp_k>0 时每步取 top-k
// (id, logprob) 并转 prob map。
SeqOut runPagedSeq(TinyLlmHandle *h, int lp_k) {
    SeqOut             out;
    const auto         prompt = promptTokens();
    const auto         pos0 = promptPositions();
    int                seq_ids[] = {0}, seq_lens[] = {kPromptLen};
    unsigned char      pre[] = {1};
    int                next = -1;
    std::vector<float> lp(static_cast<size_t>(std::max(lp_k, 0)) * 2, 0.0f);

    auto collect = [&]() {
        if (lp_k <= 0) return;
        std::map<int, float> m;
        for (int i = 0; i < lp_k; ++i) {
            const int id = static_cast<int>(lp[2 * i]);
            // top.size() < k 时 ABI 写哨兵 (id=-1, logprob=0)——不过滤会当成
            // prob=1.0 的伪条目，且只在 prob 恰好下溢为 0 的一侧出现。
            if (id < 0) continue;
            m[id] = std::exp(static_cast<double>(lp[2 * i + 1]));
        }
        out.probs.push_back(std::move(m));
    };

    EXPECT_EQ(tinyllm_allocate_sequence(h, 0, kPromptLen + kDecodeSteps + 8), 0);
    int nb = 3;
    int rc = tinyllm_step(h, seq_ids, prompt.data(), pos0.data(), seq_lens, kBlocks.data(), &nb,
                          pre, 1, &next, lp.data(), lp_k, seq_status);
    out.steps_rc.push_back(rc);
    if (rc != 0 || next < 0) {
        tinyllm_free_sequence(h, 0);
        return out;
    }
    out.tokens.push_back(next);
    collect();

    // 块表随可见窗口增长：visible = 48+i+1，第 17 步起需要第 5 块
    for (int i = 0; i < kDecodeSteps; ++i) {
        const int need = (kPromptLen + i + 1 + kBlockSize - 1) / kBlockSize;
        rc = decodeStep(h, kFeedToken, need, kPromptLen + i, &next, lp.data(), lp_k);
        out.steps_rc.push_back(rc);
        if (rc != 0 || next < 0) break;
        out.tokens.push_back(next);
        collect();
    }
    EXPECT_EQ(out.tokens.size(), static_cast<size_t>(kDecodeSteps + 1));
    EXPECT_EQ(tinyllm_free_sequence(h, 0), 0);
    return out;
}

// 概率分布比较：所有出现的 id（两边并集）|Δprob| ≤ tol。
// splitkv>1 的归约序差异会穿过整条链路放大到 logit/prob 层；
// 0.02 足够覆盖该噪声，又远小于真实寻址错误造成的分布位移。
::testing::AssertionResult probsNear(const SeqOut &a, const SeqOut &b, float tol) {
    if (a.probs.size() != b.probs.size())
        return ::testing::AssertionFailure()
               << "step count differs: " << a.probs.size() << " vs " << b.probs.size();
    for (size_t s = 0; s < a.probs.size(); ++s) {
        for (const auto &[id, pa] : a.probs[s]) {
            const auto  it = b.probs[s].find(id);
            const float pb = (it == b.probs[s].end()) ? 0.0f : it->second;
            // NaN 不会通过 >tol 比较暴露（NaN>x 恒 false），必须显式拒绝。
            if (!std::isfinite(pa) || !std::isfinite(pb))
                return ::testing::AssertionFailure() << "step " << s << " token " << id
                                                     << ": non-finite prob " << pa << " vs " << pb;
            if (std::fabs(pa - pb) > tol)
                return ::testing::AssertionFailure()
                       << "step " << s << " token " << id << ": prob " << pa << " vs " << pb;
        }
        for (const auto &[id, pb] : b.probs[s]) {
            if (a.probs[s].count(id) == 0) {
                if (!std::isfinite(pb) || pb > tol)
                    return ::testing::AssertionFailure()
                           << "step " << s << " token " << id << ": prob 0 vs " << pb;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

class FfiPagedDispatchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!hasCudaDevice()) GTEST_SKIP() << "No CUDA device available";
        // 每个用例重写（TearDown 会删除），内容确定，路径按 pid 区分
        model_path_ = writeTempFile(syntheticModelGguf(), "model");
    }
    void        TearDown() override { std::remove(model_path_.c_str()); }
    std::string model_path_;
};

} // namespace

// 核心门禁：同一合成模型、同一 C ABI 路径，仅切换 TLLM_PAGED_ATTENTION
// （+ TLLM_ATTN_SPLITKV）。逐位等价模式比 token id；splitkv>1 比概率分布。
TEST_F(FfiPagedDispatchTest, DirectAutoAndSplitKvMatchLegacyThroughCAbi) {
    EnvGuard paged("TLLM_PAGED_ATTENTION"), splitkv("TLLM_ATTN_SPLITKV");
    auto     h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    paged.set("legacy");
    const SeqOut base_tok = runPagedSeq(h.get(), 0);
    ASSERT_EQ(base_tok.tokens.size(), static_cast<size_t>(kDecodeSteps + 1));

    // 逐位等价模式：token id 必须严格相等
    for (const char *mode : {"direct", "auto"}) {
        paged.set(mode);
        splitkv.unset();
        EXPECT_EQ(runPagedSeq(h.get(), 0).tokens, base_tok.tokens) << "mode=" << mode;
    }
    paged.set("direct");
    splitkv.set("1");
    EXPECT_EQ(runPagedSeq(h.get(), 0).tokens, base_tok.tokens) << "splitkv=1 anchor";

    // 加强：逐位等价模式的**概率分布**也必须逐位相同（probs 由 logits 经
    // host fp32 softmax 推出——logits 逐位 ⇒ probs 逐位）。token id 只证明
    // argmax 一致，无法区分"同分布"与"概率在小数点后漂移"；容差 0 即逐位比较。
    paged.set("legacy");
    splitkv.unset();
    const SeqOut base_exact = runPagedSeq(h.get(), kVocab);
    for (const char *mode : {"direct", "auto"}) {
        paged.set(mode);
        splitkv.unset();
        EXPECT_TRUE(probsNear(base_exact, runPagedSeq(h.get(), kVocab), 0.0f))
            << "bitwise mode=" << mode;
    }
    paged.set("direct");
    splitkv.set("1");
    EXPECT_TRUE(probsNear(base_exact, runPagedSeq(h.get(), kVocab), 0.0f))
        << "bitwise splitkv-env=1";

    // splitkv>1：逐步比较完整概率分布（容忍 fp32 归约序差异）。
    // 开关同时作用于 direct 与 legacy 两条入口——只测 direct 会让 legacy
    // 侧的 partial 接线回归漏检（master 上正是两条入口同样空转）。
    paged.set("legacy");
    splitkv.unset();
    const SeqOut base_lp = runPagedSeq(h.get(), kVocab);
    for (const char *mode : {"direct", "legacy"}) {
        paged.set(mode);
        for (const char *ns : {"2", "4", "16"}) {
            splitkv.set(ns);
            const SeqOut cur = runPagedSeq(h.get(), kVocab);
            EXPECT_TRUE(probsNear(base_lp, cur, 0.02f)) << "mode=" << mode << " splitkv=" << ns;
        }
    }
}

// 策略 2（max_num_blocks == 0，连续 KV）不受 decode dispatch 开关影响。
TEST_F(FfiPagedDispatchTest, StrategyTwoIgnoresAttentionSwitch) {
    EnvGuard      paged("TLLM_PAGED_ATTENTION");
    char          err[512] = {0};
    TinyLlmConfig cfg2 = {0, 0, 0, 0, 0, 0, kBlockSize, 2, 0};
    HandlePtr     h(tinyllm_load(model_path_.c_str(), &cfg2, err, sizeof(err)));
    ASSERT_NE(h, nullptr) << err;

    auto run = [&]() {
        std::vector<int> tokens;
        const auto       prompt = promptTokens();
        const auto       pos0 = promptPositions();
        int              seq_ids[] = {0}, seq_lens[] = {kPromptLen};
        unsigned char    pre[] = {1};
        int              next = -1;
        EXPECT_EQ(tinyllm_allocate_sequence(h.get(), 0, kPromptLen + kDecodeSteps + 8), 0);
        EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos0.data(), seq_lens, nullptr,
                               nullptr, pre, 1, &next, nullptr, 0, seq_status),
                  0);
        tokens.push_back(next);
        int pos = kPromptLen;
        for (int i = 0; i < kDecodeSteps; ++i) {
            int           in = kFeedToken, len1[] = {1};
            unsigned char dec[] = {0};
            EXPECT_EQ(tinyllm_step(h.get(), seq_ids, &in, &pos, len1, nullptr, nullptr, dec, 1,
                                   &next, nullptr, 0, seq_status),
                      0);
            tokens.push_back(next);
            ++pos;
        }
        EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
        return tokens;
    };

    paged.set("legacy");
    const auto base = run();
    paged.set("direct");
    EXPECT_EQ(run(), base);
}

// 块表不足时（ABI v3）调用级仍返回 TLLM_OK，逐序列状态 seq_status[0]
// 标记失败，且序列不被毒化：释放重分配后继续跑，仍与 legacy 基线一致。
TEST_F(FfiPagedDispatchTest, ShortBlockTableFailsThenSequenceSurvives) {
    EnvGuard paged("TLLM_PAGED_ATTENTION");
    paged.set("direct");
    auto h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    const auto    prompt = promptTokens();
    const auto    pos0 = promptPositions();
    int           seq_ids[] = {0}, seq_lens[] = {kPromptLen}, next = -1;
    unsigned char pre[] = {1};
    int           too_few = 1; // 48 token 需 3 块，只给 1 块

    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 0, 64), 0);
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos0.data(), seq_lens, kBlocks.data(),
                           &too_few, pre, 1, &next, nullptr, 0, seq_status),
              0)
        << "call-level rc stays OK under ABI v3";
    EXPECT_EQ(seq_status[0], TLLM_SEQ_ERR) << "undersized block table must fail this sequence";
    EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);

    const auto direct_tokens = runPagedSeq(h.get(), 0);
    paged.set("legacy");
    EXPECT_EQ(direct_tokens.tokens, runPagedSeq(h.get(), 0).tokens);
}

// TLLM_ATTN_SPLITKV 非法值必须在 decode 分支干净失败，不得崩溃或静默
// 回退；恢复合法值后同一句柄继续可用。
TEST_F(FfiPagedDispatchTest, InvalidSplitKvEnvFailsCleanly) {
    EnvGuard paged("TLLM_PAGED_ATTENTION"), splitkv("TLLM_ATTN_SPLITKV");
    paged.set("direct");
    auto h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    const auto    prompt = promptTokens();
    const auto    pos0 = promptPositions();
    int           seq_ids[] = {0}, seq_lens[] = {kPromptLen}, next = -1;
    unsigned char pre[] = {1};
    int           nb = 3;
    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 0, 64), 0);

    // env 在每次 attentionPaged 入口解析（不缓存）：非法值对 prefill 同样
    // 显式失败——先用合法环境完成 prefill，再注入非法值验证 decode 干净失败。
    ASSERT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos0.data(), seq_lens, kBlocks.data(),
                           &nb, pre, 1, &next, nullptr, 0, seq_status),
              0);

    splitkv.set("garbage");
    nb = 4;
    int bad_next = -1;
    EXPECT_EQ(decodeStep(h.get(), kFeedToken, nb, kPromptLen, &bad_next, nullptr, 0), 0)
        << "call-level rc stays OK under ABI v3";
    EXPECT_EQ(seq_status[0], TLLM_SEQ_ERR)
        << "invalid TLLM_ATTN_SPLITKV must fail this sequence at decode dispatch";

    splitkv.unset();
    nb = 4;
    int good_next = -1;
    EXPECT_EQ(decodeStep(h.get(), kFeedToken, nb, kPromptLen, &good_next, nullptr, 0), 0)
        << "handle must stay usable after env restored";
    EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);
    EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
}
// ── ABI v3：逐序列状态 ─────────────────────────────────────────
// 混合 batch：seq 0 合法、seq 1 块表不足。调用级返回 TLLM_OK，
// seq 0 正常产出 next_token，seq 1 仅 seq_status 标记失败——不再株连。
// 且失败序列 position 不推进：随后用合法块表重试 prefill 必须成功。
TEST_F(FfiPagedDispatchTest, PartialBatchFailureDoesNotCascade) {
    EnvGuard paged("TLLM_PAGED_ATTENTION");
    paged.set("direct");
    auto h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    const auto    prompt = promptTokens();
    const auto    pos0 = promptPositions();
    int           seq_ids[] = {0, 1};
    int           seq_lens[] = {kPromptLen, kPromptLen};
    unsigned char pre[] = {1, 1};
    int           next[2] = {-1, -1};
    // seq0 用满 5 块切片，seq1 只声明 1 块（48 token 需 3 块 → 失败）
    int nb[2] = {5, 1};

    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 0, 64), 0);
    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 1, 64), 0);

    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos0.data(), seq_lens, kBlocks.data(),
                           nb, pre, 2, next, nullptr, 0, seq_status),
              0);
    EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);
    EXPECT_EQ(seq_status[1], TLLM_SEQ_ERR);
    EXPECT_GE(next[0], 0) << "seq0 must produce a token despite seq1 failure";

    // seq1 的 position 必须仍为 0（失败原子性）：合法块表重试 prefill 成功。
    {
        int           sid1[] = {1}, len1[] = {kPromptLen}, next1 = -1;
        unsigned char pre1[] = {1};
        int           nb1 = 3;
        EXPECT_EQ(tinyllm_step(h.get(), sid1, prompt.data(), pos0.data(), len1, kBlocks.data(),
                               &nb1, pre1, 1, &next1, nullptr, 0, seq_status),
                  0);
        EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);
        EXPECT_GE(next1, 0);
    }

    EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
    EXPECT_EQ(tinyllm_free_sequence(h.get(), 1), 0);
}

// 未分配 seq_id 混入 batch：同样只败自己、不株连。
TEST_F(FfiPagedDispatchTest, UnallocatedSeqIdFailsOnlyItself) {
    EnvGuard paged("TLLM_PAGED_ATTENTION");
    paged.set("legacy");
    auto h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    const auto    prompt = promptTokens();
    const auto    pos0 = promptPositions();
    int           seq_ids[] = {0, 9}; // 9 从未 allocate
    int           seq_lens[] = {kPromptLen, kPromptLen};
    unsigned char pre[] = {1, 1};
    int           next[2] = {-1, -1};
    int           nb[2] = {3, 3};

    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 0, 64), 0);
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos0.data(), seq_lens, kBlocks.data(),
                           nb, pre, 2, next, nullptr, 0, seq_status),
              0);
    EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);
    EXPECT_EQ(seq_status[1], TLLM_SEQ_ERR);
    EXPECT_GE(next[0], 0);
    EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
}

// C ABI 契约：几何字段非 0 时必须与 GGUF 一致，否则 load 显式失败，
// 而不是让调用方以为自己的配置生效、实际按另一套几何运行。
TEST_F(FfiPagedDispatchTest, TinyLlmLoadRejectsMismatchedGeometry) {
    char          err[512] = {0};
    TinyLlmConfig cfg = {kHidden + 1, 0, 0, 0, 0, 0, kBlockSize, 1, kMaxBlocks};
    EXPECT_EQ(tinyllm_load(model_path_.c_str(), &cfg, err, sizeof(err)), nullptr);
    EXPECT_NE(std::string(err).find("hidden_dim"), std::string::npos) << err;
}

// ABI v3：positions 为惰性参数（后端 SeqState 内部跟踪）——与后端记账
// 不一致的值不再导致失败；decode 的 seq_len != 1 仍是违约，但只记逐序列
// 失败（seq_status）、不株连 batch，也不消耗 token 切片。
TEST_F(FfiPagedDispatchTest, TinyLlmStepInertPositionsAndPerSeqLenCheck) {
    auto h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);
    ASSERT_EQ(tinyllm_allocate_sequence(h.get(), 0, kPromptLen + 8), 0);

    const auto       prompt = promptTokens();
    auto             pos = promptPositions();
    int              seq_ids[] = {0}, seq_lens[] = {kPromptLen}, nb = 3, next = -1;
    unsigned char    pre[] = {1};
    std::vector<int> blocks = kBlocks;

    // positions 惰性：prefill 位置不从 0 开始不影响调用与本序列状态
    pos[0] = 1;
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos.data(), seq_lens, blocks.data(),
                           &nb, pre, 1, &next, nullptr, 0, seq_status),
              0);
    EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);

    int           tok = kFeedToken;
    int           len1[] = {1}, need = 4;
    unsigned char dec[] = {0};

    // decode 位置与后端记账不一致同样被忽略（SeqState 权威）
    int wrong_pos = kPromptLen + 1;
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, &tok, &wrong_pos, len1, blocks.data(), &need, dec, 1,
                           &next, nullptr, 0, seq_status),
              0);
    EXPECT_EQ(seq_status[0], TLLM_SEQ_OK);

    // decode 的 seq_len != 1：调用级仍 TLLM_OK，本序列记失败
    int good_pos = kPromptLen + 1;
    int len2[] = {2};
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, &tok, &good_pos, len2, blocks.data(), &need, dec, 1,
                           &next, nullptr, 0, seq_status),
              0);
    EXPECT_EQ(seq_status[0], TLLM_SEQ_ERR);

    EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
}

// ─────────────────────────────────────────────────────────────────
// DEV-12：TLLM_BATCH_DECODE 行批量前向 —— 与逐序列路径逐位差分。
// 4 条序列同一 step 调用（B=4），prompt 长度不同 → 位置/可见长度/块表
// 均不同；块表 id 全池唯一且乱序 → 覆盖非连续块表与行映射。
// ─────────────────────────────────────────────────────────────────
TEST_F(FfiPagedDispatchTest, BatchDecodeMatchesScalarThroughCAbi) {
    EnvGuard bd("TLLM_BATCH_DECODE"), paged("TLLM_PAGED_ATTENTION");
    auto     h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    constexpr int kSeqs = 4;
    constexpr int kSteps = 12;
    // 各序列不同 prompt 长度（16 的倍数对齐块）：48/32/16/48 → decode 起始
    // 绝对位置 {48,32,16,48}，可见块数 {3,2,1,3}，随步数可跨进下一块。
    const int prompt_lens[kSeqs] = {48, 32, 16, 48};
    // 每条序列 5 个互不相同的物理块（共享 pool，id 必须唯一、乱序且
    // < kMaxBlocks=32）。
    const int seq_blocks[kSeqs][5] = {{3, 17, 9, 31, 22},
                                      {6, 28, 14, 30, 1},
                                      {11, 25, 4, 20, 19},
                                      {24, 8, 26, 13, 27}};

    auto run = [&]() {
        std::vector<std::array<int, kSeqs>> steps_tokens;
        // prefill：逐序列独立调用（prefill 不进批量路径）
        for (int s = 0; s < kSeqs; ++s) {
            EXPECT_EQ(tinyllm_allocate_sequence(h.get(), s, 96), 0);
            int                 seq_ids[] = {s}, seq_lens[] = {prompt_lens[s]}, next = -1;
            unsigned char       pre[] = {1};
            std::vector<int>    prompt(prompt_lens[s]);
            std::vector<int>    pos(prompt_lens[s]);
            for (int i = 0; i < prompt_lens[s]; ++i) {
                prompt[i] = (i * 7 + s) % 128;
                pos[i] = i;
            }
            int nb = prompt_lens[s] / kBlockSize;
            EXPECT_EQ(tinyllm_step(h.get(), seq_ids, prompt.data(), pos.data(), seq_lens,
                                   seq_blocks[s], &nb, pre, 1, &next, nullptr, 0, seq_status),
                      0);
            EXPECT_EQ(seq_status[0], TLLM_SEQ_OK) << "prefill seq " << s;
        }
        // decode：4 序列同一 step 调用
        for (int step = 0; step < kSteps; ++step) {
            int           seq_ids[kSeqs], toks[kSeqs], pos[kSeqs], lens[kSeqs], nbs[kSeqs],
                next[kSeqs];
            unsigned char dec[kSeqs];
            std::vector<int> flat_blocks;
            for (int s = 0; s < kSeqs; ++s) {
                seq_ids[s] = s;
                toks[s] = kFeedToken + s;
                pos[s] = prompt_lens[s] + step;
                lens[s] = 1;
                dec[s] = 0;
                nbs[s] = (prompt_lens[s] + step + 1 + kBlockSize - 1) / kBlockSize;
                EXPECT_LE(nbs[s], 5);
                for (int b = 0; b < nbs[s]; ++b) flat_blocks.push_back(seq_blocks[s][b]);
                next[s] = -1;
            }
            EXPECT_EQ(tinyllm_step(h.get(), seq_ids, toks, pos, lens, flat_blocks.data(), nbs,
                                   dec, kSeqs, next, nullptr, 0, seq_status),
                      0);
            for (int s = 0; s < kSeqs; ++s)
                EXPECT_EQ(seq_status[s], TLLM_SEQ_OK) << "step " << step << " seq " << s;
            std::array<int, kSeqs> row;
            for (int s = 0; s < kSeqs; ++s) row[s] = next[s];
            steps_tokens.push_back(row);
        }
        for (int s = 0; s < kSeqs; ++s)
            EXPECT_EQ(tinyllm_free_sequence(h.get(), s), 0);
        return steps_tokens;
    };

    bd.set("0"); // DEV-14 起默认启用：baseline 须显式关闭批量路径
    const auto scalar = run();
    bd.set("1");
    const auto batched = run();

    // 逐位等价：argmax token 必须严格一致（DEV-10 oracle 口径；
    // argmax 边界分歧按规格须保留现场提交审阅，不放宽容差）
    EXPECT_EQ(batched, scalar);
}

// 批量失败语义：B∉{2,4,8}（3 条）回落逐序列路径，结果与基线一致；
// B=4 但含非法行的混合批中合法行仍正常产出。
TEST_F(FfiPagedDispatchTest, BatchDecodeFallsBackForUnsupportedCounts) {
    EnvGuard bd("TLLM_BATCH_DECODE"), paged("TLLM_PAGED_ATTENTION");
    auto     h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);
    bd.set("1");

    constexpr int kSeqs = 3; // 不在 {2,4,8}：应整体回落
    const int     seq_blocks[kSeqs][5] = {{3, 17, 9, 31, 22}, {6, 28, 14, 30, 1}, {11, 25, 4, 20, 19}};
    int           seq_ids[kSeqs], toks[kSeqs], pos[kSeqs], lens[kSeqs], nbs[kSeqs], next[kSeqs];
    unsigned char dec[kSeqs];
    for (int s = 0; s < kSeqs; ++s) {
        ASSERT_EQ(tinyllm_allocate_sequence(h.get(), s, 64), 0);
        int              sid[] = {s}, plen[] = {kPromptLen}, nx = -1;
        unsigned char    pre[] = {1};
        const auto       prompt = promptTokens();
        const auto       p0 = promptPositions();
        int              nb = 3;
        ASSERT_EQ(tinyllm_step(h.get(), sid, prompt.data(), p0.data(), plen, seq_blocks[s], &nb,
                               pre, 1, &nx, nullptr, 0, seq_status),
                  0);
    }
    std::vector<int> flat;
    for (int s = 0; s < kSeqs; ++s) {
        seq_ids[s] = s;
        toks[s] = kFeedToken;
        pos[s] = kPromptLen;
        lens[s] = 1;
        dec[s] = 0;
        nbs[s] = (kPromptLen + 1 + kBlockSize - 1) / kBlockSize;
        for (int b = 0; b < nbs[s]; ++b) flat.push_back(seq_blocks[s][b]);
        next[s] = -1;
    }
    EXPECT_EQ(tinyllm_step(h.get(), seq_ids, toks, pos, lens, flat.data(), nbs, dec, kSeqs, next,
                           nullptr, 0, seq_status),
              0);
    for (int s = 0; s < kSeqs; ++s) {
        EXPECT_EQ(seq_status[s], TLLM_SEQ_OK) << "seq " << s;
        EXPECT_GE(next[s], 0);
    }
    for (int s = 0; s < kSeqs; ++s)
        EXPECT_EQ(tinyllm_free_sequence(h.get(), s), 0);
}

// ─────────────────────────────────────────────────────────────────
// DEV-14：批量 decode graph 分桶 —— 同一 B 图服务任意行集。
// 场景：B=4 decode 数步 → 释放 seq1、新分配 seq4（同 B 不同行集/槽位
// 映射/块表）→ 继续 B=4 → 释放两条降为 B=2。批量路径必须经
// d_batch_slots device 索引映回而非固化槽位；逐位 oracle 同 DEV-12。
// ─────────────────────────────────────────────────────────────────
TEST_F(FfiPagedDispatchTest, BatchDecodeGraphBucketsMatchScalar) {
    EnvGuard bd("TLLM_BATCH_DECODE"), paged("TLLM_PAGED_ATTENTION");
    auto     h = loadPaged(model_path_.c_str());
    ASSERT_NE(h, nullptr);

    const int init_lens[4] = {48, 32, 16, 48};
    const int all_blocks[5][5] = {{3, 17, 9, 31, 22},
                                  {6, 28, 14, 30, 1},
                                  {11, 25, 4, 20, 19},
                                  {24, 8, 26, 13, 27},
                                  {2, 15, 21, 29, 5}}; // seq4 后续替换 seq1 用

    auto do_prefill = [&](int slot, int seq_id, int len, const int *blocks) {
        (void)slot; // 槽位仅是调用方数组下标；分配/释放按 seq_id
        EXPECT_EQ(tinyllm_allocate_sequence(h.get(), seq_id, 96), 0);
        int              sid[] = {seq_id}, lens[] = {len}, nx = -1;
        unsigned char    pre[] = {1};
        std::vector<int> prompt(len), pos(len);
        for (int i = 0; i < len; ++i) {
            prompt[i] = (i * 7 + seq_id) % 128;
            pos[i] = i;
        }
        int nb = len / kBlockSize;
        EXPECT_EQ(tinyllm_step(h.get(), sid, prompt.data(), pos.data(), lens, blocks, &nb, pre, 1,
                               &nx, nullptr, 0, seq_status),
                  0);
        EXPECT_EQ(seq_status[0], TLLM_SEQ_OK) << "prefill seq " << seq_id;
        return nx;
    };

    auto run = [&]() {
        std::vector<std::vector<int>> tokens_log;
        int                           next_tok[5] = {0};
        for (int s = 0; s < 4; ++s) next_tok[s] = do_prefill(s, s, init_lens[s], all_blocks[s]);
        int cur_len[5] = {init_lens[0], init_lens[1], init_lens[2], init_lens[3], 0};

        // decode 一段：按当前 seq_ids/位置推进 n 步
        auto decode_steps = [&](const std::vector<int> &seq_ids, int steps) {
            const int n = static_cast<int>(seq_ids.size());
            for (int step = 0; step < steps; ++step) {
                int              sids[8], toks[8], pos[8], slens[8], nbs[8], next[8];
                unsigned char    dec[8];
                std::vector<int> flat;
                for (int s = 0; s < n; ++s) {
                    const int id = seq_ids[s];
                    sids[s] = id;
                    toks[s] = next_tok[id] >= 0 ? next_tok[id] : kFeedToken;
                    pos[s] = cur_len[id];
                    slens[s] = 1;
                    dec[s] = 0;
                    nbs[s] = (pos[s] + 1 + kBlockSize - 1) / kBlockSize;
                    EXPECT_LE(nbs[s], 5);
                    for (int b = 0; b < nbs[s]; ++b) flat.push_back(all_blocks[id][b]);
                    next[s] = -1;
                }
                EXPECT_EQ(tinyllm_step(h.get(), sids, toks, pos, slens, flat.data(), nbs, dec, n,
                                       next, nullptr, 0, seq_status),
                          0);
                std::vector<int> row(n);
                for (int s = 0; s < n; ++s) {
                    EXPECT_EQ(seq_status[s], TLLM_SEQ_OK)
                        << "step " << step << " seq " << seq_ids[s];
                    row[s] = next[s];
                    next_tok[seq_ids[s]] = next[s];
                }
                tokens_log.push_back(row);
            }
            for (int id : seq_ids) cur_len[id] += steps;
        };

        decode_steps({0, 1, 2, 3}, 6); // B=4 桶：首次 eager + 捕获，后续重放

        // seq1 退出 → 新 seq4 占槽1（不同 prompt 长、不同块表）：行集变化
        // 但 B 不变 → 同一张 B=4 图必须仍正确（device 槽位索引映回）。
        EXPECT_EQ(tinyllm_free_sequence(h.get(), 1), 0);
        cur_len[4] = 16;
        next_tok[4] = do_prefill(1, 4, 16, all_blocks[4]);
        decode_steps({0, 4, 2, 3}, 6); // 复用 B=4 graph，行集 {0,4,2,3}

        // 降为 B=2：释放 seq4 与 seq0 → seq{2,3} 继续
        EXPECT_EQ(tinyllm_free_sequence(h.get(), 4), 0);
        EXPECT_EQ(tinyllm_free_sequence(h.get(), 0), 0);
        decode_steps({2, 3}, 4); // B=2 桶

        for (int id : {2, 3}) EXPECT_EQ(tinyllm_free_sequence(h.get(), id), 0);
        return tokens_log;
    };

    bd.set("0"); // baseline 走逐序列路径（默认批量显式关闭）
    const auto scalar = run();
    bd.set("1");
    const auto batched = run();

    EXPECT_EQ(batched, scalar);
}
