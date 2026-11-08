// GGUFParser 健壮性测试（不需要真实模型文件）。
//
// R3: 张量数据区对齐应读取 general.alignment（而非硬编码 32）；
// R4: GGUF v1 应被拒绝（与 llama.cpp 一致），而非仅 WARN 后按 v2 解析；
// R10: 嵌套数组（数组的元素类型为 ARRAY）应显式拒绝，而非递归解析。

#include "tiny_llm/gguf_parser.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace tiny_llm;

namespace {

constexpr uint32_t U32_TYPE = 4; // GGUF_METADATA_VALUE_TYPE_UINT32
constexpr uint32_t STR_TYPE = 8; // GGUF_METADATA_VALUE_TYPE_STRING
constexpr uint32_t ARR_TYPE = 9; // GGUF_METADATA_VALUE_TYPE_ARRAY
constexpr uint32_t I32_TYPE = 5; // GGUF_METADATA_VALUE_TYPE_INT32

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

void writeFile(const std::string &path, const std::vector<uint8_t> &buf) {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "cannot write " << path;
    f.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
    ASSERT_TRUE(f.good());
}

std::string tmpPath(const char *tag) {
    return std::string("/tmp/tiny_llm_gguf_") + tag + "_" + std::to_string(::getpid()) + ".gguf";
}

// header：magic + version + tensor_count + metadata_kv_count
std::vector<uint8_t> header(uint32_t version, uint64_t n_kv) {
    std::vector<uint8_t> b;
    pushU32(b, GGUF_MAGIC);
    pushU32(b, version);
    pushU64(b, 0); // tensor_count
    pushU64(b, n_kv);
    return b;
}

// 带 tensor_count 的 header（张量健壮性测试用）
std::vector<uint8_t> headerWithTensors(uint32_t version, uint64_t n_tensors) {
    std::vector<uint8_t> b;
    pushU32(b, GGUF_MAGIC);
    pushU32(b, version);
    pushU64(b, n_tensors);
    pushU64(b, 0); // metadata_kv_count
    return b;
}

// 追加一条 tensor info（不写数据区；形状校验只读 info，不读数据）
void pushTensorInfo(std::vector<uint8_t> &b, const std::string &name,
                    std::initializer_list<uint64_t> dims) {
    pushStr(b, name);
    pushU32(b, static_cast<uint32_t>(dims.size()));
    for (uint64_t d : dims)
        pushU64(b, d);
    pushU32(b, 0); // type = F32
    pushU64(b, 0); // offset
}

ModelConfig smallConfig() {
    ModelConfig cfg;
    cfg.hidden_dim = 8;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.num_kv_heads = 1;
    cfg.vocab_size = 32;
    cfg.intermediate_dim = 16;
    return cfg;
}

} // namespace

// R4: GGUF v1 必须被拒绝（version < 2 不再仅 WARN）
TEST(GGUFParser, RejectsVersion1) {
    auto              buf = header(1, 0);
    const std::string path = tmpPath("v1");
    writeFile(path, buf);

    GGUFParser p(path);
    EXPECT_TRUE(p.parse().isErr()) << "GGUF v1 should be rejected";

    std::remove(path.c_str());
}

// R3: 张量数据区按 general.alignment 对齐（本测试用 alignment=64）
//
// 字节布局（tellg 刻意凑成 78，使 32 对齐=96、64 对齐=128）：
//   header 24B
//   kv1 "general.alignment"(17) -> u32 64   : 8+17+4+4 = 33B
//   kv2 "x"(1) -> string ""                 : 8+1+4+8  = 21B
//   tellg = 24+33+21 = 78
TEST(GGUFParser, HonorsGeneralAlignment) {
    auto buf = header(3, 2);
    pushStr(buf, "general.alignment");
    pushU32(buf, U32_TYPE);
    pushU32(buf, 64);
    pushStr(buf, "x");
    pushU32(buf, STR_TYPE);
    pushU64(buf, 0); // 空字符串

    const std::string path = tmpPath("align");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk());
    EXPECT_EQ(p.getDataOffset(), 128u)
        << "data offset should be aligned to general.alignment=64 (not 32)";

    std::remove(path.c_str());
}

// R10: 数组元素类型为 ARRAY 时显式拒绝（不再递归解析）
TEST(GGUFParser, RejectsNestedArray) {
    auto buf = header(3, 1);
    pushStr(buf, "nested");
    pushU32(buf, ARR_TYPE);
    pushU32(buf, ARR_TYPE); // 外层数组的元素类型 = ARRAY
    pushU64(buf, 1);        // 1 个元素
    // 内层数组：INT32, count=1, value=42（若被递归解析会"成功"，旧实现返回 ok）
    pushU32(buf, I32_TYPE);
    pushU64(buf, 1);
    pushU32(buf, 42);

    const std::string path = tmpPath("nested");
    writeFile(path, buf);

    GGUFParser p(path);
    EXPECT_TRUE(p.parse().isErr()) << "nested arrays must be rejected explicitly";

    std::remove(path.c_str());
}

// 健壮性（与 llama.cpp#26366/#26978 同类审计）：
// n_dims 文件可控，超过 GGML_MAX_DIMS(4) 必须拒绝——否则恶意值
// （如 0xFFFFFFFF）会让 dimensions.resize 尝试 ~32GB 分配，
// 未捕获的 bad_alloc 直接 abort。
TEST(GGUFParser, RejectsExcessiveTensorDims) {
    for (uint32_t bad_dims : {5u, 0xFFFFFFFFu}) {
        auto buf = headerWithTensors(3, 1);
        pushStr(buf, "t0");
        pushU32(buf, bad_dims); // n_dims 越界；维度数据本身无需写出
        pushU32(buf, 0);        // type = F32
        pushU64(buf, 0);        // offset

        const std::string path = tmpPath("ndims");
        writeFile(path, buf);

        GGUFParser p(path);
        EXPECT_TRUE(p.parse().isErr())
            << "n_dims=" << bad_dims << " must be rejected before resize";

        std::remove(path.c_str());
    }
}

// 健壮性（与 llama.cpp#26978 的 GGML_PAD 回绕同类）：
// tensor.offset 文件可控（64 位），与 data_offset_ 相加回绕后
// seekg 会落到错误偏移读垃圾数据（静默损坏）。必须在相加前拒绝。
TEST(GGUFParser, RejectsTensorOffsetOverflow) {
    auto buf = headerWithTensors(3, 1);
    pushStr(buf, "t0");
    pushU32(buf, 1);              // n_dims = 1
    pushU64(buf, 1);              // dims = [1]（F32 单元素，4 字节）
    pushU32(buf, 0);              // type = F32
    pushU64(buf, UINT64_MAX - 2); // offset：与 data_offset_ 相加必然回绕

    const std::string path = tmpPath("offovf");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk()) << "tensor info 本身结构合法，parse 应通过";
    ASSERT_EQ(p.getTensors().size(), 1u);
    EXPECT_TRUE(p.readTensorData(p.getTensors()[0]).isErr())
        << "offset 加法回绕必须被拒绝，而不是 seek 到错误偏移";

    std::remove(path.c_str());
}

// 健壮性：tensor 声明的尺寸超过文件剩余字节时，必须在分配前拒绝。
// 否则一个几百字节的伪造 GGUF 可声明巨大 dimensions，先分配数 GB 再读失败，
// 在 Linux 上表现为 OOM-kill 而不是解析错误。
TEST(GGUFParser, RejectsTensorDataExceedingFileRemainder) {
    auto buf = headerWithTensors(3, 1);
    pushStr(buf, "t0");
    pushU32(buf, 1);    // n_dims = 1
    pushU64(buf, 1024); // 1024 个 F32 = 4096 字节，但文件在 tensor info 后结束
    pushU32(buf, 0);    // type = F32
    pushU64(buf, 0);    // offset = 0

    const std::string path = tmpPath("truncated");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk()) << "tensor info 本身结构合法，parse 应通过";
    ASSERT_EQ(p.getTensors().size(), 1u);
    auto r = p.readTensorData(p.getTensors()[0]);
    ASSERT_TRUE(r.isErr()) << "声明尺寸大于文件剩余字节必须在分配前拒绝";
    EXPECT_NE(r.error().find("remaining"), std::string::npos) << r.error();

    std::remove(path.c_str());
}

// 形状与 ModelConfig 交叉校验：合法维度（含 blk./layers. 两种前缀）必须通过。
TEST(GGUFParser, ValidatesTensorShapesAgainstConfig) {
    auto buf = headerWithTensors(3, 8);
    pushTensorInfo(buf, "token_embd.weight", {8, 32});
    pushTensorInfo(buf, "blk.0.attn_q.weight", {8, 8});
    pushTensorInfo(buf, "blk.0.attn_k.weight", {8, 4});
    pushTensorInfo(buf, "blk.0.ffn_down.weight", {16, 8});
    pushTensorInfo(buf, "blk.0.attn_norm.weight", {8});
    pushTensorInfo(buf, "output.weight", {8, 32});
    pushTensorInfo(buf, "blk.0.attn_q.bias", {8});
    pushTensorInfo(buf, "layers.0.attn_v.bias", {4});

    const std::string path = tmpPath("shape_ok");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk());
    auto r = p.validateTensorShapes(smallConfig());
    EXPECT_TRUE(r.isOk()) << (r.isErr() ? r.error() : "");

    std::remove(path.c_str());
}

// 维度写反（ffn_down 的 in/out 颠倒）元素总数不变，但布局错误，必须拒绝。
TEST(GGUFParser, RejectsTransposedTensorShape) {
    auto buf = headerWithTensors(3, 1);
    pushTensorInfo(buf, "blk.0.ffn_down.weight", {8, 16});

    const std::string path = tmpPath("shape_transposed");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk());
    auto r = p.validateTensorShapes(smallConfig());
    ASSERT_TRUE(r.isErr()) << "shape [8,16] 与 ffn_down 期望 [16,8] 不符，必须拒绝";
    EXPECT_NE(r.error().find("blk.0.ffn_down.weight"), std::string::npos) << r.error();
    EXPECT_NE(r.error().find("[8, 16]"), std::string::npos) << r.error();
    EXPECT_NE(r.error().find("[16, 8]"), std::string::npos) << r.error();

    std::remove(path.c_str());
}

// 词表维必须与 config.vocab_size 精确相等：lm_head 的 FP16 转置副本
// 按 vocab_size 定步长，padding/截断都会让转置读错行。
TEST(GGUFParser, RejectsMismatchedVocabDimension) {
    for (uint64_t bad_vocab : {40u, 16u}) {
        auto buf = headerWithTensors(3, 1);
        pushTensorInfo(buf, "token_embd.weight", {8, bad_vocab});
        const std::string path = tmpPath("shape_vocab");
        writeFile(path, buf);

        GGUFParser p(path);
        ASSERT_TRUE(p.parse().isOk());
        auto r = p.validateTensorShapes(smallConfig());
        ASSERT_TRUE(r.isErr()) << "vocab=" << bad_vocab << " 与 config 32 不符，必须拒绝";
        EXPECT_NE(r.error().find("token_embd.weight"), std::string::npos) << r.error();
        EXPECT_NE(r.error().find("[8, 32]"), std::string::npos) << r.error();
        std::remove(path.c_str());
    }
}

// 配置不完整时不得进行形状校验（缺键回落默认值可能把错误形状判成匹配）。
TEST(GGUFParser, RejectsShapeValidationWithIncompleteConfig) {
    auto buf = headerWithTensors(3, 0);

    const std::string path = tmpPath("shape_cfg");
    writeFile(path, buf);

    GGUFParser p(path);
    ASSERT_TRUE(p.parse().isOk());
    ModelConfig cfg = smallConfig();
    cfg.num_layers = 0;
    EXPECT_TRUE(p.validateTensorShapes(cfg).isErr());

    std::remove(path.c_str());
}
