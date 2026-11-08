// Q4_0 / Q5_0 / Q4_K / Q6_K 反量化的单元测试。
//
// 期望值的来源：
// 1. 合成块测试的期望值由 Python `gguf` 包的参考实现
//    （gguf.quants.Q4_0/Q5_0/Q4_K/Q6_K.dequantize）对相同字节序列计算得出；
// 2. 真实模型测试（环境变量 TLLM_GGUF_TEST_MODEL 门控）的期望值由同一参考
//    实现对 Qwen2.5-0.5B-Instruct Q4_K_M 文件的首块反量化结果得出。
// 布局依据见 ggml/gguf 官方格式，不依赖本仓库自身实现计算期望值。

#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/quantization.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace tiny_llm;

namespace {

std::vector<float> toFloats(const std::vector<half> &hs) {
    std::vector<float> out(hs.size());
    for (size_t i = 0; i < hs.size(); ++i) {
        out[i] = __half2float(hs[i]);
    }
    return out;
}

void expectNearF16(float actual, float expected, const char *ctx) {
    const float tol = std::abs(expected) * 1e-3f + 1e-6f;
    EXPECT_NEAR(actual, expected, tol) << ctx;
}

} // namespace

TEST(DequantizeQ4_0, NullPointerFails) {
    auto r = dequantizeQ4_0(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ4_0, MatchesGGMLSplitHalfLayout) {
    // 合成块: d=1.0, qs[i] = (i << 4) | (15 - i)。
    // GGML 布局：值 i (i<16) 取 qs[i] 低 nibble，值 i+16 取 qs[i] 高 nibble。
    // 低 nibble 依次为 15..0 -> -1..-8；高 nibble 依次为 0..15 -> 0..7。
    // 旧的交错布局（2j 低 / 2j+1 高）会得到不同的顺序。
    uint8_t block[18];
    half    d = __float2half(1.0f);
    std::memcpy(block, &d, sizeof(d));
    for (int i = 0; i < 16; ++i) {
        block[2 + i] = static_cast<uint8_t>((i << 4) | (15 - i));
    }

    auto r = dequantizeQ4_0(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 32u);
    for (int i = 0; i < 16; ++i) {
        const int q = (15 - i) > 7 ? (15 - i) - 16 : (15 - i);
        EXPECT_FLOAT_EQ(v[i], static_cast<float>(q)) << "low nibble index " << i;
    }
    for (int i = 0; i < 16; ++i) {
        const int q = i > 7 ? i - 16 : i;
        EXPECT_FLOAT_EQ(v[16 + i], static_cast<float>(q)) << "high nibble index " << i;
    }
}

TEST(DequantizeQ5_0, NullPointerFails) {
    auto r = dequantizeQ5_0(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ5_0, MatchesReferenceOnSyntheticBlock) {
    // 合成块: d=2.0, qh=全 1（所有第 5 位置位）, qs=0x5A x 16
    // 值 0..15: 低 nibble 0xA=10 | (1<<4) = 26 -> (26-16)*2 = 20
    // 值 16..31: 高 nibble 0x5 | (1<<4) = 21 -> (21-16)*2 = 10
    uint8_t block[22];
    half    d = __float2half(2.0f);
    std::memcpy(block, &d, sizeof(d));
    uint32_t qh = 0xFFFFFFFFu;
    std::memcpy(block + 2, &qh, sizeof(qh));
    std::memset(block + 6, 0x5A, 16);

    auto r = dequantizeQ5_0(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 32u);
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(v[i], 20.0f) << "index " << i;
    }
    for (int i = 16; i < 32; ++i) {
        EXPECT_FLOAT_EQ(v[i], 10.0f) << "index " << i;
    }
}

TEST(DequantizeQ4_K, NullPointerFails) {
    auto r = dequantizeQ4_K(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ4_K, MatchesReferenceOnSyntheticBlock) {
    // 合成块: d=1.0, dmin=0.5, scales=[0..11], qs=[0..127]
    uint8_t block[144];
    half    d = __float2half(1.0f);
    half    dmin = __float2half(0.5f);
    std::memcpy(block, &d, sizeof(d));
    std::memcpy(block + 2, &dmin, sizeof(dmin));
    for (int i = 0; i < 12; ++i)
        block[4 + i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 128; ++i)
        block[16 + i] = static_cast<uint8_t>(i);

    auto r = dequantizeQ4_K(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 256u);

    // Python gguf 参考实现输出
    for (int i = 0; i < 8; ++i)
        expectNearF16(v[i], -2.0f, "sub-block 0");
    for (int i = 32; i < 40; ++i)
        expectNearF16(v[i], -2.5f, "sub-block 1");
    const float sb2[8] = {-3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    for (int i = 0; i < 8; ++i)
        expectNearF16(v[64 + i], sb2[i], "sub-block 2");
    for (int i = 248; i < 256; ++i)
        expectNearF16(v[i], 77.0f, "sub-block 7");
}

TEST(DequantizeQ6_K, NullPointerFails) {
    auto r = dequantizeQ6_K(nullptr, 1);
    EXPECT_TRUE(r.isErr());
}

TEST(DequantizeQ6_K, MatchesReferenceOnSyntheticBlock) {
    // 合成块: ql=[0..127], qh=[0..63], scales=[-8..7], d=0.25
    uint8_t block[210];
    for (int i = 0; i < 128; ++i)
        block[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 64; ++i)
        block[128 + i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 16; ++i)
        block[192 + i] = static_cast<uint8_t>(static_cast<int8_t>(i - 8));
    half d = __float2half(0.25f);
    std::memcpy(block + 208, &d, sizeof(d));

    auto r = dequantizeQ6_K(block, 1);
    ASSERT_FALSE(r.isErr()) << r.error();
    auto v = toFloats(r.value());
    ASSERT_EQ(v.size(), 256u);

    // Python gguf 参考实现输出
    const float first8[8] = {64.0f, 30.0f, -4.0f, -38.0f, 56.0f, 22.0f, -12.0f, -46.0f};
    for (int i = 0; i < 8; ++i)
        expectNearF16(v[i], first8[i], "first 8");
    const float at16[8] = {56.0f, 26.25f, -3.5f, -33.25f, 49.0f, 19.25f, -10.5f, -40.25f};
    for (int i = 0; i < 8; ++i)
        expectNearF16(v[16 + i], at16[i], "[16:24]");
    for (int i = 128; i < 136; ++i)
        expectNearF16(v[i], 0.0f, "[128:136]");
    for (int i = 248; i < 256; ++i)
        expectNearF16(v[i], -43.75f, "last 8");
}

// ── 真实模型验证（门控）──────────────────────────────────────────
// 设置 TLLM_GGUF_TEST_MODEL 指向 Qwen2.5-0.5B-Instruct Q4_K_M GGUF 运行。
// 期望值由 Python gguf 参考实现计算，见文件头注释。

class GGUFRealModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const char *path = std::getenv("TLLM_GGUF_TEST_MODEL");
        if (path == nullptr) {
            GTEST_SKIP() << "set TLLM_GGUF_TEST_MODEL to a GGUF file to enable";
        }
        path_ = path;
        parser_ = std::make_unique<GGUFParser>(path_);
        auto r = parser_->parse();
        ASSERT_FALSE(r.isErr()) << r.error();
    }

    std::vector<half> dequantFirstBlock(const std::string &name, GGMLType type, size_t block_bytes,
                                        size_t block_vals) {
        const auto *tensor = parser_->getTensorByName(name);
        EXPECT_NE(tensor, nullptr) << name;
        if (!tensor) return {};
        EXPECT_EQ(static_cast<uint32_t>(tensor->type), static_cast<uint32_t>(type)) << name;
        auto raw = parser_->readTensorData(*tensor);
        EXPECT_FALSE(raw.isErr()) << raw.error();
        if (raw.isErr()) return {};
        switch (type) {
        case GGMLType::Q5_0:
            return dequantizeQ5_0(raw.value().data(), 1).value();
        case GGMLType::Q4_K:
            return dequantizeQ4_K(raw.value().data(), 1).value();
        case GGMLType::Q6_K:
            return dequantizeQ6_K(raw.value().data(), 1).value();
        default:
            ADD_FAILURE() << "unexpected type";
            return {};
        }
    }

    std::string                 path_;
    std::unique_ptr<GGUFParser> parser_;
};

TEST_F(GGUFRealModelTest, ConfigMatchesQwen25HalfB) {
    auto cr = parser_->extractModelConfig();
    ASSERT_FALSE(cr.isErr()) << cr.error();
    const auto &cfg = cr.value();

    // 仅当确实是 Qwen2.5-0.5B 时断言精确值（允许该门控测试用于其他模型做冒烟检查）
    if (cfg.hidden_dim == 896 && cfg.num_layers == 24) {
        EXPECT_EQ(cfg.num_heads, 14);
        EXPECT_EQ(cfg.num_kv_heads, 2);
        EXPECT_EQ(cfg.vocab_size, 151936);
        EXPECT_EQ(cfg.intermediate_dim, 4864);
        EXPECT_EQ(cfg.max_seq_len, 32768);
        EXPECT_NEAR(cfg.rope_theta, 1000000.0f, 1.0f);
        EXPECT_NEAR(cfg.rms_norm_eps, 1e-6f, 1e-9f);
        EXPECT_EQ(cfg.eos_token_id, 151645);
    } else {
        EXPECT_GT(cfg.hidden_dim, 0);
        EXPECT_GT(cfg.vocab_size, 0);
    }
}

TEST_F(GGUFRealModelTest, FirstBlocksMatchPythonReference) {
    const auto &cfg = parser_->extractModelConfig().value();
    if (!(cfg.hidden_dim == 896 && cfg.num_layers == 24)) {
        GTEST_SKIP() << "reference values are for Qwen2.5-0.5B-Instruct Q4_K_M";
    }

    {
        auto v = toFloats(dequantFirstBlock("token_embd.weight", GGMLType::Q5_0, 22, 32));
        ASSERT_EQ(v.size(), 32u);
        const float exp8[8] = {-0.0101929f, 0.0407715f,   0.0101929f, -0.0f,
                               -0.0280304f, -0.00254822f, -0.0f,      -0.0203857f};
        for (int i = 0; i < 8; ++i)
            expectNearF16(v[i], exp8[i], "token_embd Q5_0");
    }
    {
        auto v = toFloats(dequantFirstBlock("blk.0.ffn_down.weight", GGMLType::Q6_K, 210, 256));
        ASSERT_EQ(v.size(), 256u);
        const float exp8[8] = {-0.00611287f, 0.00611287f,  0.0279446f,  0.00436634f,
                               -0.00523961f, -0.00698614f, -0.0183386f, 0.0026198f};
        for (int i = 0; i < 8; ++i)
            expectNearF16(v[i], exp8[i], "ffn_down Q6_K first8");
        const float exp128[4] = {0.00285149f, -0.00142574f, -0.00142574f, 0.0456238f};
        for (int i = 0; i < 4; ++i)
            expectNearF16(v[128 + i], exp128[i], "ffn_down Q6_K [128:132]");
    }
    {
        auto v = toFloats(dequantFirstBlock("blk.23.ffn_down.weight", GGMLType::Q4_K, 144, 256));
        ASSERT_EQ(v.size(), 256u);
        const float exp8[8] = {0.00669551f,  0.00373709f,  0.00669551f, 0.00373709f,
                               -0.00513816f, 0.000778675f, -0.011055f,  0.000778675f};
        for (int i = 0; i < 8; ++i)
            expectNearF16(v[i], exp8[i], "blk.23 Q4_K first8");
        const float exp128[4] = {0.00363874f, 0.0179163f, -0.00825924f, 0.00601834f};
        for (int i = 0; i < 4; ++i)
            expectNearF16(v[128 + i], exp128[i], "blk.23 Q4_K [128:132]");
    }
}

// 完整权重管线往返：GGUF 反量化 -> 转置(列主[in,out] -> 行主[K,N]) -> W8A16 量化
// -> 反量化重建。验证 loadGGUF 的权重转换不会破坏权重值（真实模型门控）。
TEST_F(GGUFRealModelTest, WeightW8A16RoundTripPreservesValues) {
    const auto &cfg = parser_->extractModelConfig().value();
    if (!(cfg.hidden_dim == 896 && cfg.num_layers == 24)) {
        GTEST_SKIP() << "reference values are for Qwen2.5-0.5B-Instruct Q4_K_M";
    }

    const auto *tensor = parser_->getTensorByName("blk.0.attn_q.weight");
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(static_cast<uint32_t>(tensor->type), static_cast<uint32_t>(GGMLType::Q5_0));

    int in_f = static_cast<int>(tensor->dimensions[0]);  // K
    int out_f = static_cast<int>(tensor->dimensions[1]); // N
    ASSERT_EQ(in_f, 896);
    ASSERT_EQ(out_f, 896);
    const int group_size = 128;

    auto raw = parser_->readTensorData(*tensor);
    ASSERT_FALSE(raw.isErr()) << raw.error();

    size_t num_blocks = (tensor->numElements() + 31) / 32;
    auto   f16 = dequantizeQ5_0(raw.value().data(), num_blocks);
    ASSERT_FALSE(f16.isErr());
    ASSERT_EQ(f16.value().size(), static_cast<size_t>(in_f) * out_f);

    // 复现 model_loader 的转置：列主 [out, in] -> 行主 [in, out]
    std::vector<half> dst(static_cast<size_t>(in_f) * out_f);
    for (int o = 0; o < out_f; ++o) {
        for (int i = 0; i < in_f; ++i) {
            dst[static_cast<size_t>(i) * out_f + o] =
                f16.value()[static_cast<size_t>(o) * in_f + i];
        }
    }

    auto q = quantizeF16ToW8A16(dst.data(), in_f, out_f, group_size);
    ASSERT_FALSE(q.isErr());
    const auto &int8_data = q.value().first;
    const auto &scales = q.value().second;

    // 反量化重建：val = q * scale。量化误差应受限于该组步长 scale/2。
    double max_abs_err = 0.0;
    double worst_ratio = 0.0; // 绝对误差 / scale，应 <= 0.5 + 少许
    size_t max_err_idx = 0;
    size_t bad_count = 0;
    for (int k = 0; k < in_f; ++k) {
        for (int n = 0; n < out_f; ++n) {
            size_t idx = static_cast<size_t>(k) * out_f + n;
            int    group = k / group_size;
            float  scale = __half2float(scales[static_cast<size_t>(group) * out_f + n]);
            float  rebuilt = static_cast<float>(int8_data[idx]) * scale;
            float  orig = __half2float(dst[idx]);
            float  abs_err = std::abs(rebuilt - orig);
            float  ratio = abs_err / (scale + 1e-30f);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_err_idx = idx;
            }
            if (ratio > worst_ratio) worst_ratio = ratio;
            if (ratio > 0.75) ++bad_count;
        }
    }
    EXPECT_LT(worst_ratio, 0.75) << "worst abs_err/scale ratio " << worst_ratio << " at idx "
                                 << max_err_idx << " (k=" << max_err_idx / out_f
                                 << ", n=" << max_err_idx % out_f << ")";
    // 量化应覆盖绝大多数元素；允许少数极小值被量化到相邻档位
    EXPECT_LT(static_cast<double>(bad_count) / (static_cast<double>(in_f) * out_f), 1e-4)
        << "bad_count=" << bad_count;
}

// R7: 组内仅有极小非零值时，scale 不应被钳到 1.0f（那会把整组量化成 0）。
// 应保持一个可表示的非零 scale（≥ fp16 最小正常数），保留量级信息。
TEST(W8A16Quantize, TinyNonZeroGroupKeepsNonUnitScale) {
    std::vector<half> data(128, __float2half(0.0f));
    data[0] = __float2half(1e-9f);
    auto r = quantizeF16ToW8A16(data.data(), 128, 1, 128);
    ASSERT_TRUE(r.isOk());
    const auto &pair = r.value();
    ASSERT_EQ(pair.second.size(), 1u);
    const float scale = __half2float(pair.second[0]);
    EXPECT_NE(scale, 1.0f) << "scale should not be clamped to 1.0 for tiny groups";
    EXPECT_GT(scale, 0.0f) << "scale must stay representable in fp16";
}
