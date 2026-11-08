#include "tiny_llm/inference_engine.h"
#include "tiny_llm/model_loader.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
// #include <rapidcheck.h>
// NOTE: rapidcheck/gtest disabled due to GCC 11/12 std::function bug
// in CI builds.
// #include <rapidcheck/gtest.h>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace tiny_llm;

// Helper to create temporary files
class TempFile {
  public:
    TempFile(const std::string &suffix = ".bin") {
#ifdef _WIN32
        int pid = _getpid();
#else
        int pid = getpid();
#endif
        path_ = "/tmp/tiny_llm_test_" + std::to_string(pid) + "_" +
                std::to_string(std::random_device{}()) + suffix;
    }

    ~TempFile() { std::remove(path_.c_str()); }

    const std::string &path() const { return path_; }

    void write(const void *data, size_t size) {
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char *>(data), size);
    }

    template <typename T>
    void write(const T &value) {
        write(&value, sizeof(T));
    }

    void writeBytes(const std::vector<uint8_t> &bytes) const {
        std::ofstream file(path_, std::ios::binary);
        file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }

  private:
    std::string path_;
};

namespace {

void appendBytes(std::vector<uint8_t> &out, const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T>
void appendValue(std::vector<uint8_t> &out, const T &value) {
    appendBytes(out, &value, sizeof(T));
}

void appendString(std::vector<uint8_t> &out, const std::string &value) {
    uint64_t size = value.size();
    appendValue(out, size);
    appendBytes(out, value.data(), value.size());
}

struct GGUFFileBuilder {
    std::vector<uint8_t> metadata_entries;
    std::vector<uint8_t> tensor_entries;
    uint64_t             metadata_count = 0;
    uint64_t             tensor_count = 0;

    template <typename T>
    void addScalarMetadata(const std::string &key, GGUFType type, const T &value) {
        appendString(metadata_entries, key);
        uint32_t raw_type = static_cast<uint32_t>(type);
        appendValue(metadata_entries, raw_type);
        appendValue(metadata_entries, value);
        ++metadata_count;
    }

    void addArrayMetadataHeader(const std::string &key, GGUFType elem_type, uint64_t count) {
        appendString(metadata_entries, key);
        uint32_t raw_type = static_cast<uint32_t>(GGUFType::ARRAY);
        appendValue(metadata_entries, raw_type);
        uint32_t raw_elem_type = static_cast<uint32_t>(elem_type);
        appendValue(metadata_entries, raw_elem_type);
        appendValue(metadata_entries, count);
        ++metadata_count;
    }

    void writeTo(const TempFile &file) const {
        std::vector<uint8_t> bytes;
        appendValue(bytes, GGUF_MAGIC);
        uint32_t version = 3;
        appendValue(bytes, version);
        appendValue(bytes, tensor_count);
        appendValue(bytes, metadata_count);
        bytes.insert(bytes.end(), metadata_entries.begin(), metadata_entries.end());
        bytes.insert(bytes.end(), tensor_entries.begin(), tensor_entries.end());

        constexpr uint64_t kAlignment = 32;
        size_t             aligned_size = (bytes.size() + kAlignment - 1) & ~(kAlignment - 1);
        bytes.resize(aligned_size, 0);
        file.writeBytes(bytes);
    }
};

// ============================================================================
// 任务 A4：完整最小 GGUF 构造器 —— header + metadata + tensor info + F32 数据区，
// 让 ModelLoader::loadGGUF 能真正成功加载（校验 bias / tied embedding 契约路径）。
// ============================================================================
struct GGUFTensorSpec {
    std::string           name;
    std::vector<uint64_t> dims; // GGUF 维度顺序 [dim0, dim1, ...]
    std::vector<float>    data; // F32，row-major，元素数 = dims 乘积
};

class MinimalGGUFBuilder {
  public:
    int hidden = 8, layers = 1, heads = 1, kv_heads = 1, vocab = 4, context = 16, intermediate = 8;

    std::vector<GGUFTensorSpec> tensors;

    // 生成完整 GGUF 文件字节（含对齐后的 F32 数据区）。
    std::vector<uint8_t> build() const {
        std::vector<uint8_t> metadata;
        uint64_t             metadata_count = 0;
        auto                 add_int = [&](const std::string &key, int32_t v) {
            appendString(metadata, key);
            appendValue(metadata, uint32_t(static_cast<uint32_t>(GGUFType::INT32)));
            appendValue(metadata, v);
            ++metadata_count;
        };
        add_int("llama.embedding_length", hidden);
        add_int("llama.block_count", layers);
        add_int("llama.attention.head_count", heads);
        add_int("llama.attention.head_count_kv", kv_heads);
        add_int("llama.context_length", context);
        add_int("llama.feed_forward_length", intermediate);
        // tokenizer.ggml.tokens（STRING 数组）→ extractModelConfig 派生 vocab_size
        appendString(metadata, "tokenizer.ggml.tokens");
        appendValue(metadata, uint32_t(static_cast<uint32_t>(GGUFType::ARRAY)));
        appendValue(metadata, uint32_t(static_cast<uint32_t>(GGUFType::STRING)));
        appendValue(metadata, uint64_t(vocab));
        for (int i = 0; i < vocab; ++i)
            appendString(metadata, "t" + std::to_string(i));
        ++metadata_count;

        // tensor info 条目（offset 相对数据区起始，数据按声明顺序连续写入）
        std::vector<uint8_t> entries;
        uint64_t             data_off = 0;
        for (const auto &t : tensors) {
            appendString(entries, t.name);
            appendValue(entries, uint32_t(t.dims.size()));
            for (auto d : t.dims)
                appendValue(entries, uint64_t(d));
            appendValue(entries, uint32_t(static_cast<uint32_t>(GGMLType::F32)));
            appendValue(entries, uint64_t(data_off));
            data_off += t.data.size() * sizeof(float);
        }

        // 组装：header + metadata + tensor info，32 字节对齐后接数据区
        std::vector<uint8_t> bytes;
        appendValue(bytes, GGUF_MAGIC);
        appendValue(bytes, uint32_t(3));
        appendValue(bytes, uint64_t(tensors.size()));
        appendValue(bytes, metadata_count);
        bytes.insert(bytes.end(), metadata.begin(), metadata.end());
        bytes.insert(bytes.end(), entries.begin(), entries.end());

        constexpr uint64_t kAlignment = 32;
        size_t             aligned = (bytes.size() + kAlignment - 1) & ~(kAlignment - 1);
        bytes.resize(aligned, 0);
        for (const auto &t : tensors) {
            appendBytes(bytes, t.data.data(), t.data.size() * sizeof(float));
        }
        return bytes;
    }
};

} // namespace

// ============================================================================
// 任务 A4：模型权重契约 —— bias / tied output embedding 处理路径。
// 构造完整最小 GGUF（真实 F32 数据）走 ModelLoader::loadGGUF 全路径。
// ============================================================================
class ModelLoaderContractTest : public ::testing::Test {
  protected:
    static bool hasCuda() {
        int         count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return err == cudaSuccess && count > 0;
    }

    static std::vector<float> ramp(size_t n, float base) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i)
            v[i] = base + static_cast<float>(i) * 0.01f;
        return v;
    }

    // 写一个完整最小 GGUF。include_output=false 时省略 output.weight（tied 场景）；
    // include_qwen_bias=true 时带 blk.0.attn_q/k/v.bias；extra 追加额外张量。
    void writeMinimalGGUF(TempFile &file, bool include_output, bool include_qwen_bias,
                          const std::vector<GGUFTensorSpec> &extra = {}) const {
        MinimalGGUFBuilder b;
        auto add_weight = [&](const char *name, uint64_t in, uint64_t out, float base) {
            b.tensors.push_back({name, {in, out}, ramp(in * out, base)});
        };
        add_weight("token_embd.weight", 8, 4, 0.1f);
        add_weight("blk.0.attn_q.weight", 8, 8, 0.2f);
        add_weight("blk.0.attn_k.weight", 8, 8, 0.3f);
        add_weight("blk.0.attn_v.weight", 8, 8, 0.4f);
        add_weight("blk.0.attn_output.weight", 8, 8, 0.5f);
        add_weight("blk.0.ffn_gate.weight", 8, 8, 0.6f);
        add_weight("blk.0.ffn_down.weight", 8, 8, 0.7f);
        add_weight("blk.0.ffn_up.weight", 8, 8, 0.8f);
        b.tensors.push_back({"blk.0.attn_norm.weight", {8}, ramp(8, 1.0f)});
        b.tensors.push_back({"blk.0.ffn_norm.weight", {8}, ramp(8, 1.1f)});
        b.tensors.push_back({"output_norm.weight", {8}, ramp(8, 1.2f)});
        if (include_output) add_weight("output.weight", 8, 4, 0.9f);
        if (include_qwen_bias) {
            b.tensors.push_back({"blk.0.attn_q.bias", {8}, ramp(8, 2.0f)});
            b.tensors.push_back({"blk.0.attn_k.bias", {8}, ramp(8, 2.1f)});
            b.tensors.push_back({"blk.0.attn_v.bias", {8}, ramp(8, 2.2f)});
        }
        for (const auto &t : extra)
            b.tensors.push_back(t);
        file.writeBytes(b.build());
    }
};

TEST_F(ModelLoaderContractTest, NonTiedOutputWeightLoads) {
    if (!hasCuda()) GTEST_SKIP() << "No CUDA device available";
    TempFile file(".gguf");
    writeMinimalGGUF(file, /*include_output=*/true, /*include_qwen_bias=*/false);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);
    ASSERT_TRUE(result.isOk()) << result.error();
    auto &weights = result.value();
    EXPECT_TRUE(weights.lm_head.isValid());
    EXPECT_EQ(weights.lm_head.rows, config.hidden_dim);
    EXPECT_EQ(weights.lm_head.cols, config.vocab_size);
    EXPECT_NE(weights.lm_head_fp16, nullptr);
    ModelLoader::freeWeights(weights);
}

TEST_F(ModelLoaderContractTest, TiedOutputEmbeddingFallsBackToTokenEmbedding) {
    if (!hasCuda()) GTEST_SKIP() << "No CUDA device available";
    TempFile file(".gguf");
    // 无 output.weight / lm_head.weight → tied output embedding 路径
    writeMinimalGGUF(file, /*include_output=*/false, /*include_qwen_bias=*/false);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);
    ASSERT_TRUE(result.isOk()) << result.error();
    auto &weights = result.value();
    EXPECT_TRUE(weights.lm_head.isValid());
    EXPECT_EQ(weights.lm_head.rows, config.hidden_dim);
    EXPECT_EQ(weights.lm_head.cols, config.vocab_size);
    ASSERT_NE(weights.lm_head_fp16, nullptr);

    // 差分验证：lm_head_fp16 [hidden, vocab] = transpose(token_embd [vocab, hidden])
    const int         hidden = config.hidden_dim;
    const int         vocab = config.vocab_size;
    std::vector<half> lm_host(static_cast<size_t>(hidden) * vocab);
    cudaMemcpy(lm_host.data(), weights.lm_head_fp16, lm_host.size() * sizeof(half),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    const auto emb_host = ramp(static_cast<size_t>(hidden) * vocab, 0.1f);
    for (int s = 0; s < hidden; ++s) {
        for (int v = 0; v < vocab; ++v) {
            // token_embd GGUF row-major [vocab, hidden]；lm_head 转置为 [hidden, vocab]
            EXPECT_NEAR(__half2float(lm_host[static_cast<size_t>(s) * vocab + v]),
                        emb_host[static_cast<size_t>(v) * hidden + s], 1e-3f)
                << "hidden " << s << " vocab " << v;
        }
    }
    ModelLoader::freeWeights(weights);
}

TEST_F(ModelLoaderContractTest, QwenAttentionBiasTensorsLoad) {
    if (!hasCuda()) GTEST_SKIP() << "No CUDA device available";
    TempFile file(".gguf");
    writeMinimalGGUF(file, /*include_output=*/true, /*include_qwen_bias=*/true);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);
    ASSERT_TRUE(result.isOk()) << result.error();
    auto &weights = result.value();
    ASSERT_EQ(weights.layers.size(), 1u);
    EXPECT_NE(weights.layers[0].wq_bias, nullptr);
    EXPECT_NE(weights.layers[0].wk_bias, nullptr);
    EXPECT_NE(weights.layers[0].wv_bias, nullptr);
    ModelLoader::freeWeights(weights);
}

// 任务 C1：loadGGUF 后每个 QuantizedWeight 必须已构建转置副本（data_t/scales_t），
// 且 data_t 确实等于 data 的转置（抽查一个 8×4 子块）。
TEST_F(ModelLoaderContractTest, TransposedWeightCopiesBuiltAndCorrect) {
    if (!hasCuda()) GTEST_SKIP() << "No CUDA device available";
    TempFile file(".gguf");
    writeMinimalGGUF(file, /*include_output=*/true, /*include_qwen_bias=*/false);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);
    ASSERT_TRUE(result.isOk()) << result.error();
    auto &weights = result.value();

    // 逐层所有 QuantizedWeight + lm_head 都必须有转置副本
    ASSERT_EQ(weights.layers.size(), 1u);
    auto &lw = weights.layers[0];
    for (const QuantizedWeight *qw :
         {&lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.w1, &lw.w2, &lw.w3, &weights.lm_head}) {
        EXPECT_TRUE(qw->hasTransposed()) << "missing transposed copy";
    }
    EXPECT_NE(weights.lm_head_fp16_t, nullptr);

    // 抽查 wq（8×8）：D2H 读回 data 与 data_t，比较 data_t[i*rows+j] == data[j*cols+i]
    const QuantizedWeight &qw = lw.wq;
    const int              rows = qw.rows;
    const int              cols = qw.cols;
    ASSERT_EQ(rows, 8);
    ASSERT_EQ(cols, 8);
    std::vector<int8_t> data_host(static_cast<size_t>(rows) * cols);
    std::vector<int8_t> data_t_host(static_cast<size_t>(rows) * cols);
    cudaMemcpy(data_host.data(), qw.data, data_host.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_t_host.data(), qw.data_t, data_t_host.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < 8; ++i) {     // data_t 行（= data 列）
        for (int j = 0; j < 4; ++j) { // data_t 列（= data 行），抽查 8×4 子块
            EXPECT_EQ(data_t_host[static_cast<size_t>(i) * cols + j],
                      data_host[static_cast<size_t>(j) * cols + i])
                << "transpose mismatch at data_t[" << i << "][" << j << "]";
        }
    }

    ModelLoader::freeWeights(weights);
}

TEST_F(ModelLoaderContractTest, UnsupportedBiasTensorReturnsExplicitError) {
    TempFile file(".gguf");
    // 不支持的 bias（output.bias）→ 显式报错而非静默忽略（无需 GPU，校验在加载前）
    writeMinimalGGUF(file, /*include_output=*/true, /*include_qwen_bias=*/false,
                     {GGUFTensorSpec{"output.bias", {8}, ramp(8, 3.0f)}});

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);
    ASSERT_TRUE(result.isErr()) << "unsupported bias must be rejected explicitly";
    EXPECT_NE(result.error().find("bias"), std::string::npos)
        << "unexpected error: " << result.error();
    EXPECT_NE(result.error().find("output.bias"), std::string::npos)
        << "unexpected error: " << result.error();
}

// Unit tests for GGUF header parsing
class GGUFHeaderTest : public ::testing::Test {};

TEST(GGUFTensorInfoTest, NumElementsReturnsZeroOnOverflow) {
    GGUFTensorInfo info;
    info.dimensions = {std::numeric_limits<uint64_t>::max(), 2};

    EXPECT_EQ(info.numElements(), 0u);
}

TEST(GGUFTensorInfoTest, CalculateSizeReturnsZeroOnOverflow) {
    GGUFTensorInfo info;
    info.type = GGMLType::F32;
    info.dimensions = {std::numeric_limits<uint64_t>::max(), 2};

    EXPECT_EQ(info.calculateSize(), 0u);
}

TEST_F(GGUFHeaderTest, OversizedArrayReturnsErrorInsteadOfThrowing) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addArrayMetadataHeader("bad.array", GGUFType::UINT32,
                                std::numeric_limits<uint64_t>::max() / sizeof(uint32_t));
    gguf.writeTo(file);

    GGUFParser parser(file.path());

    EXPECT_NO_THROW({
        auto result = parser.parse();
        EXPECT_TRUE(result.isErr());
        if (result.isErr()) {
            EXPECT_NE(result.error().find("Array"), std::string::npos);
        }
    });
}

TEST_F(GGUFHeaderTest, ExtractModelConfigIgnoresNumericArchitectureMetadata) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addScalarMetadata("general.architecture", GGUFType::UINT32, uint32_t{1234});
    gguf.writeTo(file);

    GGUFParser parser(file.path());
    auto       parse_result = parser.parse();
    ASSERT_TRUE(parse_result.isOk()) << parse_result.error();

    auto config_result = parser.extractModelConfig();
    ASSERT_TRUE(config_result.isOk()) << config_result.error();
    EXPECT_EQ(config_result.value().hidden_dim, ModelConfig{}.hidden_dim);
}

TEST_F(GGUFHeaderTest, IncompleteRuntimeGGUFReturnsStructuredErrorWithoutThrowing) {
    TempFile        file(".gguf");
    GGUFFileBuilder gguf;
    gguf.addScalarMetadata("llama.embedding_length", GGUFType::UINT32, uint32_t{8});
    gguf.addScalarMetadata("llama.block_count", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.attention.head_count", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.attention.head_count_kv", GGUFType::UINT32, uint32_t{1});
    gguf.addScalarMetadata("llama.context_length", GGUFType::UINT32, uint32_t{16});
    gguf.addScalarMetadata("llama.feed_forward_length", GGUFType::UINT32, uint32_t{16});
    gguf.addScalarMetadata("tokenizer.ggml.model.vocab_size", GGUFType::UINT32, uint32_t{32});
    gguf.writeTo(file);

    ModelConfig config;

    EXPECT_NO_THROW({
        auto result = ModelLoader::loadGGUF(file.path(), config);
        EXPECT_TRUE(result.isErr());
        if (result.isErr()) {
            EXPECT_TRUE(result.error().find("missing") != std::string::npos ||
                        result.error().find("required") != std::string::npos ||
                        result.error().find("unsupported") != std::string::npos);
        }
    });
}

TEST_F(GGUFHeaderTest, ValidHeaderTruncatedTensorData) {
    // 合法 header + 完整 metadata/tensor info，但 tensor 数据区被截断
    // （声明的数据大小超过文件长度）。parse() 只读 info 应成功；
    // readTensorData() 必须干净地返回错误且不崩溃。
    GGUFFileBuilder builder;
    // 手写一条合法 string 元数据（GGUF：key + type + uint64 长度 + 字节）
    appendString(builder.metadata_entries, "general.architecture");
    appendValue(builder.metadata_entries, uint32_t(static_cast<uint32_t>(GGUFType::STRING)));
    appendString(builder.metadata_entries, "qwen2");
    builder.metadata_count = 1;
    builder.tensor_count = 1;
    appendString(builder.tensor_entries, "blk.0.weight");
    appendValue(builder.tensor_entries, uint32_t(1));    // n_dims
    appendValue(builder.tensor_entries, uint32_t(4096)); // dim
    appendValue(builder.tensor_entries, uint32_t(static_cast<uint32_t>(GGMLType::F32)));
    appendValue(builder.tensor_entries, uint64_t(0)); // offset（无对齐、无数据）

    TempFile file(".gguf");
    builder.writeTo(file); // 只写 header+metadata+tensor info，不写 tensor 数据

    GGUFParser parser(file.path());
    auto       parse_result = parser.parse();
    ASSERT_TRUE(parse_result.isOk()) << parse_result.error();

    ASSERT_EQ(parser.getTensors().size(), 1u);
    auto data_result = parser.readTensorData(parser.getTensors()[0]);
    EXPECT_TRUE(data_result.isErr()) << "truncated tensor data must fail cleanly";
    EXPECT_TRUE(data_result.error().find("Failed to read tensor") != std::string::npos ||
                data_result.error().find("seek") != std::string::npos ||
                data_result.error().find("exceeds remaining file bytes") != std::string::npos)
        << "unexpected error message: " << data_result.error();
}

TEST_F(GGUFHeaderTest, ValidHeaderTruncatedMetadata) {
    // 合法 header 但 metadata 在中间被截断 → parse() 返回错误且不崩溃。
    std::vector<uint8_t> bytes;
    appendValue(bytes, GGUF_MAGIC);
    appendValue(bytes, uint32_t(3));
    appendValue(bytes, uint64_t(0)); // tensor_count
    appendValue(bytes, uint64_t(5)); // metadata_count = 5，但只写了 2 个
    // key 1
    appendString(bytes, "general.architecture");
    appendValue(bytes, uint32_t(static_cast<uint32_t>(GGUFType::STRING)));
    appendString(bytes, "qwen2");
    // key 2 只写 key、不写 type/value —— 截断
    appendString(bytes, "general.name");

    TempFile file(".gguf");
    file.writeBytes(bytes);

    GGUFParser parser(file.path());
    auto       result = parser.parse();
    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("metadata") != std::string::npos ||
                result.error().find("Failed") != std::string::npos)
        << "unexpected error message: " << result.error();
}

// 恶意/损坏 GGUF：tensor_count / metadata_kv_count 为天文数字时，parse()
// 必须返回结构化错误，而不是 reserve() 抛 length_error/bad_alloc 穿透。
TEST_F(GGUFHeaderTest, HugeEntryCountsFailCleanlyWithoutThrowing) {
    for (uint64_t huge : {uint64_t(1) << 40, std::numeric_limits<uint64_t>::max()}) {
        TempFile             file(".gguf");
        std::vector<uint8_t> bytes;
        appendValue(bytes, GGUF_MAGIC);
        appendValue(bytes, uint32_t(3));
        appendValue(bytes, huge);        // tensor_count
        appendValue(bytes, uint64_t(0)); // metadata_kv_count
        file.writeBytes(bytes);

        GGUFParser parser(file.path());
        EXPECT_NO_THROW({
            auto result = parser.parse();
            EXPECT_TRUE(result.isErr()) << "tensor_count=" << huge << " must be rejected";
            if (result.isErr()) {
                EXPECT_TRUE(result.error().find("count") != std::string::npos ||
                            result.error().find("limit") != std::string::npos ||
                            result.error().find("Failed") != std::string::npos)
                    << "unexpected error message: " << result.error();
            }
        });
    }
}

// metadata_kv_count 同理：超大计数必须被上界校验拒绝（而非逐条读到流死）。
TEST_F(GGUFHeaderTest, HugeMetadataCountFailsCleanlyWithoutThrowing) {
    TempFile             file(".gguf");
    std::vector<uint8_t> bytes;
    appendValue(bytes, GGUF_MAGIC);
    appendValue(bytes, uint32_t(3));
    appendValue(bytes, uint64_t(0));             // tensor_count
    appendValue(bytes, uint64_t(1) << 40);       // metadata_kv_count
    appendString(bytes, "general.architecture"); // 一条残缺 entry 后截断
    file.writeBytes(bytes);

    GGUFParser parser(file.path());
    EXPECT_NO_THROW({
        auto result = parser.parse();
        EXPECT_TRUE(result.isErr());
    });
}

TEST_F(GGUFHeaderTest, ValidHeader) {
    TempFile file(".gguf");

    // Write valid GGUF header
    std::vector<uint8_t> header;

    // Magic: "GGUF" = 0x46554747
    uint32_t magic = GGUF_MAGIC;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&magic),
                  reinterpret_cast<uint8_t *>(&magic) + 4);

    // Version: 3
    uint32_t version = 3;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&version),
                  reinterpret_cast<uint8_t *>(&version) + 4);

    // Tensor count: 10
    uint64_t tensor_count = 10;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&tensor_count),
                  reinterpret_cast<uint8_t *>(&tensor_count) + 8);

    // Metadata KV count: 5
    uint64_t metadata_count = 5;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&metadata_count),
                  reinterpret_cast<uint8_t *>(&metadata_count) + 8);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    // Should fail with partial implementation message or parsing error (not crash)
    EXPECT_TRUE(result.isErr());
}

TEST_F(GGUFHeaderTest, InvalidMagic) {
    TempFile file(".gguf");

    std::vector<uint8_t> header;
    uint32_t             bad_magic = 0x12345678;
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&bad_magic),
                  reinterpret_cast<uint8_t *>(&bad_magic) + 4);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    EXPECT_TRUE(result.isErr());
    // Error message should indicate header or magic issue
    EXPECT_TRUE(result.error().find("header") != std::string::npos ||
                result.error().find("magic") != std::string::npos ||
                result.error().find("Invalid") != std::string::npos);
}

TEST_F(GGUFHeaderTest, UnsupportedVersion) {
    TempFile file(".gguf");

    std::vector<uint8_t> header;
    uint32_t             magic = GGUF_MAGIC;
    uint32_t             bad_version = 99;

    header.insert(header.end(), reinterpret_cast<uint8_t *>(&magic),
                  reinterpret_cast<uint8_t *>(&magic) + 4);
    header.insert(header.end(), reinterpret_cast<uint8_t *>(&bad_version),
                  reinterpret_cast<uint8_t *>(&bad_version) + 4);

    file.writeBytes(header);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    EXPECT_TRUE(result.isErr());
    // Error message should indicate version issue
    EXPECT_TRUE(result.error().find("version") != std::string::npos ||
                result.error().find("Unsupported") != std::string::npos ||
                result.error().find("header") != std::string::npos);
}

// Unit tests for binary format
class BinLoaderTest : public ::testing::Test {};

TEST_F(BinLoaderTest, InvalidMagic) {
    TempFile file(".bin");

    uint32_t bad_magic = 0x12345678;
    file.write(bad_magic);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("magic") != std::string::npos);
}

TEST_F(BinLoaderTest, FileNotFound) {
    ModelConfig config;
    auto        result = ModelLoader::loadBin("/nonexistent/path/model.bin", config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("open") != std::string::npos ||
                result.error().find("Failed") != std::string::npos);
}

TEST_F(BinLoaderTest, GGUFLoadFailsOnMissingFile) {
    // GGUF 运行时加载已实现（ModelLoader::loadGGUF）；
    // 不存在的文件应返回解析错误而非旧的"不支持"拒绝
    ModelConfig config;
    auto        result = InferenceEngine::load("missing-model.gguf", config);

    EXPECT_TRUE(result.isErr());
    EXPECT_NE(result.error().find("Failed to parse GGUF"), std::string::npos);
}

TEST_F(BinLoaderTest, TruncatedHeader) {
    TempFile file(".bin");

    // Write only magic, no version
    uint32_t magic = BIN_MAGIC;
    file.write(magic);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    EXPECT_TRUE(result.isErr());
}

// Property-based tests for corrupted file handling
// Feature: tiny-llm-inference-engine, Property 9: Corrupted File Error Handling
// Validates: Requirements 1.5

#if 0
// NOTE: Property-based tests are temporarily disabled due to GCC 11/12
// compatibility issues with rapidcheck's GTest integration.

RC_GTEST_PROP(CorruptedFileProperty, RandomBytesReturnError, (std::vector<uint8_t> random_bytes)) {
    // Skip empty files
    RC_PRE(!random_bytes.empty());

    TempFile file(".bin");
    file.writeBytes(random_bytes);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    // Property: random bytes should always result in error, never crash
    // The result should be an error (unless by extreme chance it's valid)
    if (random_bytes.size() < sizeof(BinHeader)) {
        RC_ASSERT(result.isErr());
    }
    // Even if it passes magic check, it should fail somewhere
}

RC_GTEST_PROP(CorruptedFileProperty, TruncatedFilesReturnError, (int truncate_at)) {
    // Create a minimal valid-looking header
    std::vector<uint8_t> data;

    uint32_t    magic = BIN_MAGIC;
    uint32_t    version = BIN_VERSION;
    ModelConfig config;

    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&version),
                reinterpret_cast<uint8_t *>(&version) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&config),
                reinterpret_cast<uint8_t *>(&config) + sizeof(config));

    // Truncate at random position
    truncate_at = std::abs(truncate_at) % (data.size() + 1);
    data.resize(truncate_at);

    TempFile file(".bin");
    file.writeBytes(data);

    auto result = ModelLoader::loadBin(file.path(), config);

    // Property: truncated files should return error
    RC_ASSERT(result.isErr());
}

RC_GTEST_PROP(CorruptedFileProperty, WrongVersionReturnError, (uint32_t bad_version)) {
    // Ensure version is not valid
    RC_PRE(bad_version != BIN_VERSION);

    std::vector<uint8_t> data;

    uint32_t magic = BIN_MAGIC;
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&bad_version),
                reinterpret_cast<uint8_t *>(&bad_version) + 4);

    TempFile file(".bin");
    file.writeBytes(data);

    ModelConfig config;
    auto        result = ModelLoader::loadBin(file.path(), config);

    // Property: wrong version should return error
    RC_ASSERT(result.isErr());
    RC_ASSERT(result.error().find("version") != std::string::npos);
}

RC_GTEST_PROP(CorruptedFileProperty, GGUFRandomBytesReturnError,
              (std::vector<uint8_t> random_bytes)) {
    RC_PRE(!random_bytes.empty());

    TempFile file(".gguf");
    file.writeBytes(random_bytes);

    ModelConfig config;
    auto        result = ModelLoader::loadGGUF(file.path(), config);

    // Property: random bytes should result in error for GGUF too
    // Unless by chance they form valid GGUF magic
    if (random_bytes.size() < 4 ||
        *reinterpret_cast<const uint32_t *>(random_bytes.data()) != GGUF_MAGIC) {
        RC_ASSERT(result.isErr());
    }
}
#endif

// Test dimension mismatch
TEST_F(BinLoaderTest, DimensionMismatch) {
    TempFile file(".bin");

    std::vector<uint8_t> data;

    uint32_t magic = BIN_MAGIC;
    uint32_t version = BIN_VERSION;

    // Create config with different dimensions
    ModelConfig stored_config;
    stored_config.hidden_dim = 256;
    stored_config.num_layers = 2;
    stored_config.vocab_size = 1000;

    data.insert(data.end(), reinterpret_cast<uint8_t *>(&magic),
                reinterpret_cast<uint8_t *>(&magic) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&version),
                reinterpret_cast<uint8_t *>(&version) + 4);
    data.insert(data.end(), reinterpret_cast<uint8_t *>(&stored_config),
                reinterpret_cast<uint8_t *>(&stored_config) + sizeof(stored_config));

    file.writeBytes(data);

    // Try to load with different config
    ModelConfig expected_config;
    expected_config.hidden_dim = 512; // Different!
    expected_config.num_layers = 2;
    expected_config.vocab_size = 1000;

    auto result = ModelLoader::loadBin(file.path(), expected_config);

    EXPECT_TRUE(result.isErr());
    EXPECT_TRUE(result.error().find("mismatch") != std::string::npos);
}
