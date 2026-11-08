#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/logger.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <system_error>

namespace tiny_llm {

namespace {

bool safeMultiplySize(size_t lhs, size_t rhs, size_t &out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        out = 0;
        return false;
    }
    out = lhs * rhs;
    return true;
}

template <typename T>
bool canAllocateArray(uint64_t count) {
    return count <= std::numeric_limits<size_t>::max() / sizeof(T);
}

size_t remainingBytes(std::ifstream &file) {
    const std::streampos current = file.tellg();
    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    file.seekg(current);
    if (current < 0 || end < current) {
        return 0;
    }
    return static_cast<size_t>(end - current);
}

} // namespace

GGUFParser::GGUFParser(const std::string &path) : path_(path) {}

Result<void> GGUFParser::parse() {
    TLLM_INFO("Parsing GGUF file: {}", path_);

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return Result<void>::err("Failed to open file: " + path_);
    }

    // Parse header
    auto header_result = parseHeader(file);
    if (header_result.isErr()) {
        return header_result;
    }

    TLLM_INFO("GGUF header: version={}, tensors={}, metadata_entries={}", header_.version,
              header_.tensor_count, header_.metadata_kv_count);

    // Parse metadata
    auto meta_result = parseMetadata(file);
    if (meta_result.isErr()) {
        return meta_result;
    }

    TLLM_DEBUG("Parsed {} metadata entries", metadata_.kv.size());

    // R3: 读取 general.alignment（GGUF 规范，未指定默认 32，必须是 2 的幂）
    if (auto it = metadata_.kv.find("general.alignment"); it != metadata_.kv.end()) {
        if (const auto *v = std::get_if<uint32_t>(&it->second)) {
            const uint64_t a = *v;
            if (a != 0 && (a & (a - 1)) == 0) {
                alignment_ = a;
                TLLM_INFO("GGUF general.alignment = {}", alignment_);
            } else {
                TLLM_WARN("Ignoring invalid general.alignment {} (must be a power of 2)", a);
            }
        }
    }

    // Parse tensor info
    auto tensor_result = parseTensorInfo(file);
    if (tensor_result.isErr()) {
        return tensor_result;
    }

    TLLM_DEBUG("Parsed {} tensor entries", tensors_.size());

    // Calculate data offset with alignment
    data_offset_ = alignOffset(static_cast<uint64_t>(file.tellg()));

    TLLM_INFO("GGUF parsing complete. Data offset: {}", data_offset_);

    return Result<void>::ok();
}

Result<void> GGUFParser::parseHeader(std::ifstream &file) {
    file.read(reinterpret_cast<char *>(&header_.magic), 4);
    file.read(reinterpret_cast<char *>(&header_.version), 4);
    file.read(reinterpret_cast<char *>(&header_.tensor_count), 8);
    file.read(reinterpret_cast<char *>(&header_.metadata_kv_count), 8);

    if (!file) {
        return Result<void>::err("Failed to read GGUF header");
    }

    if (header_.magic != GGUF_MAGIC) {
        return Result<void>::err("Invalid GGUF magic number. Expected 0x" +
                                 std::to_string(GGUF_MAGIC) + ", got 0x" +
                                 std::to_string(header_.magic));
    }

    // 只支持 v2/v3：v1 布局虽与 v2 相同，但 llama.cpp 已弃用 v1（格式内容
    // 未标准化），继续解析对未知文件存在静默错读风险，这里显式拒绝。
    if (header_.version < 2) {
        return Result<void>::err("GGUF version 1 is no longer supported (please use v2/v3): " +
                                 std::to_string(header_.version));
    }
    if (header_.version > 3) {
        TLLM_WARN("GGUF version {} may not be fully supported", header_.version);
    }

    // 上界校验：损坏/恶意文件的计数是攻击者可控的 uint64，直接
    // tensors_.reserve(tensor_count) 会抛 length_error/bad_alloc 穿透 parse()，
    // 逐条循环 metadata_kv_count 则白耗 IO。1M 对真实模型绰绰有余
    // （Qwen2.5-0.5B 约 300 个张量 + 数十元数据）。
    constexpr uint64_t MAX_GGUF_ENTRY_COUNT = uint64_t(1) << 20;
    if (header_.tensor_count > MAX_GGUF_ENTRY_COUNT) {
        return Result<void>::err("GGUF tensor_count exceeds sanity limit (" +
                                 std::to_string(MAX_GGUF_ENTRY_COUNT) +
                                 "): " + std::to_string(header_.tensor_count));
    }
    if (header_.metadata_kv_count > MAX_GGUF_ENTRY_COUNT) {
        return Result<void>::err("GGUF metadata_kv_count exceeds sanity limit (" +
                                 std::to_string(MAX_GGUF_ENTRY_COUNT) +
                                 "): " + std::to_string(header_.metadata_kv_count));
    }

    return Result<void>::ok();
}

Result<void> GGUFParser::parseMetadata(std::ifstream &file) {
    for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
        // Read key
        auto key_result = readString(file);
        if (key_result.isErr()) {
            return Result<void>::err("Failed to read metadata key: " + key_result.error());
        }
        std::string key = key_result.value();

        // Read value type
        uint32_t type_val;
        file.read(reinterpret_cast<char *>(&type_val), 4);
        if (!file) {
            return Result<void>::err("Failed to read metadata type for key: " + key);
        }

        // Read value
        auto value_result = readValue(file, static_cast<GGUFType>(type_val));
        if (value_result.isErr()) {
            return Result<void>::err("Failed to read metadata value for key: " + key + ": " +
                                     value_result.error());
        }

        metadata_.kv[key] = value_result.value();
        TLLM_TRACE("Metadata: {} = <value>", key);
    }

    return Result<void>::ok();
}

Result<void> GGUFParser::parseTensorInfo(std::ifstream &file) {
    tensors_.reserve(header_.tensor_count);

    for (uint64_t i = 0; i < header_.tensor_count; ++i) {
        auto tensor_result = readTensorInfoEntry(file);
        if (tensor_result.isErr()) {
            return Result<void>::err("Failed to read tensor info " + std::to_string(i) + ": " +
                                     tensor_result.error());
        }

        tensors_.push_back(tensor_result.value());
        tensor_name_map_[tensors_.back().name] = tensors_.size() - 1;

        TLLM_TRACE("Tensor: {} dims={} type={} offset={}", tensors_.back().name,
                   tensors_.back().dimensions.size(), static_cast<uint32_t>(tensors_.back().type),
                   tensors_.back().offset);
    }

    return Result<void>::ok();
}

Result<GGUFTensorInfo> GGUFParser::readTensorInfoEntry(std::ifstream &file) {
    GGUFTensorInfo info;

    // Read name
    auto name_result = readString(file);
    if (name_result.isErr()) {
        return Result<GGUFTensorInfo>::err(name_result.error());
    }
    info.name = name_result.value();

    // Read number of dimensions
    uint32_t n_dims;
    file.read(reinterpret_cast<char *>(&n_dims), 4);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor n_dims");
    }
    // 健壮性：GGUF 张量至多 4 维（GGML_MAX_DIMS）。n_dims 文件可控，
    // 无上界时恶意值（如 0xFFFFFFFF）会让 resize 尝试 ~32GB 分配，
    // bad_alloc 未被捕获即 abort。与 llama.cpp#26366/#26978 同类的
    // 解析器健壮性问题，先于分配拒绝。
    if (n_dims > 4) {
        return Result<GGUFTensorInfo>::err("Tensor n_dims " + std::to_string(n_dims) +
                                           " exceeds GGML_MAX_DIMS(4)");
    }

    // Read dimensions
    info.dimensions.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
        file.read(reinterpret_cast<char *>(&info.dimensions[d]), 8);
        if (!file) {
            return Result<GGUFTensorInfo>::err("Failed to read tensor dimension " +
                                               std::to_string(d));
        }
    }

    // Read type
    uint32_t type_val;
    file.read(reinterpret_cast<char *>(&type_val), 4);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor type");
    }
    info.type = static_cast<GGMLType>(type_val);

    // Read offset
    file.read(reinterpret_cast<char *>(&info.offset), 8);
    if (!file) {
        return Result<GGUFTensorInfo>::err("Failed to read tensor offset");
    }

    return Result<GGUFTensorInfo>::ok(info);
}

Result<GGUFValue> GGUFParser::readValue(std::ifstream &file, GGUFType type) {
    switch (type) {
    case GGUFType::UINT8: {
        uint8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read uint8_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT8: {
        int8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read int8_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT16: {
        uint16_t v;
        file.read(reinterpret_cast<char *>(&v), 2);
        if (!file) return Result<GGUFValue>::err("Failed to read uint16_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT16: {
        int16_t v;
        file.read(reinterpret_cast<char *>(&v), 2);
        if (!file) return Result<GGUFValue>::err("Failed to read int16_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT32: {
        uint32_t v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read uint32_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT32: {
        int32_t v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read int32_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::UINT64: {
        uint64_t v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read uint64_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::INT64: {
        int64_t v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read int64_t");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::FLOAT32: {
        float v;
        file.read(reinterpret_cast<char *>(&v), 4);
        if (!file) return Result<GGUFValue>::err("Failed to read float");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::FLOAT64: {
        double v;
        file.read(reinterpret_cast<char *>(&v), 8);
        if (!file) return Result<GGUFValue>::err("Failed to read double");
        return Result<GGUFValue>::ok(GGUFValue{v});
    }
    case GGUFType::BOOL: {
        uint8_t v;
        file.read(reinterpret_cast<char *>(&v), 1);
        if (!file) return Result<GGUFValue>::err("Failed to read bool");
        return Result<GGUFValue>::ok(GGUFValue{static_cast<bool>(v)});
    }
    case GGUFType::STRING: {
        auto str_result = readString(file);
        if (str_result.isErr()) {
            return Result<GGUFValue>::err(str_result.error());
        }
        return Result<GGUFValue>::ok(GGUFValue{str_result.value()});
    }
    case GGUFType::ARRAY: {
        auto arr_result = readArray(file);
        if (arr_result.isErr()) {
            return Result<GGUFValue>::err(arr_result.error());
        }
        return Result<GGUFValue>::ok(GGUFValue{arr_result.value()});
    }
    default:
        return Result<GGUFValue>::err("Unsupported GGUF type: " +
                                      std::to_string(static_cast<uint32_t>(type)));
    }
}

Result<GGUFValue> GGUFParser::readArray(std::ifstream &file) {
    // Read element type
    uint32_t type_val;
    file.read(reinterpret_cast<char *>(&type_val), 4);
    if (!file) {
        return Result<GGUFValue>::err("Failed to read array element type");
    }
    GGUFType elem_type = static_cast<GGUFType>(type_val);

    // Read count
    uint64_t count;
    file.read(reinterpret_cast<char *>(&count), 8);
    if (!file) {
        return Result<GGUFValue>::err("Failed to read array count");
    }

    TLLM_TRACE("Reading array of {} elements, type {}", count, static_cast<uint32_t>(elem_type));
    const size_t remaining = remainingBytes(file);

    // Handle different array types
    switch (elem_type) {
    case GGUFType::UINT32: {
        if (!canAllocateArray<uint32_t>(count) || count > remaining / sizeof(uint32_t)) {
            return Result<GGUFValue>::err("Array too large: " + std::to_string(count));
        }
        std::vector<uint32_t> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read uint32 array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::INT32: {
        if (!canAllocateArray<int32_t>(count) || count > remaining / sizeof(int32_t)) {
            return Result<GGUFValue>::err("Array too large: " + std::to_string(count));
        }
        std::vector<int32_t> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read int32 array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::FLOAT32: {
        if (!canAllocateArray<float>(count) || count > remaining / sizeof(float)) {
            return Result<GGUFValue>::err("Array too large: " + std::to_string(count));
        }
        std::vector<float> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 4);
        if (!file) return Result<GGUFValue>::err("Failed to read float array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::FLOAT64: {
        if (!canAllocateArray<double>(count) || count > remaining / sizeof(double)) {
            return Result<GGUFValue>::err("Array too large: " + std::to_string(count));
        }
        std::vector<double> arr(count);
        file.read(reinterpret_cast<char *>(arr.data()), count * 8);
        if (!file) return Result<GGUFValue>::err("Failed to read double array");
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    case GGUFType::STRING: {
        if (count > remaining / sizeof(uint64_t)) {
            return Result<GGUFValue>::err("Array too large: " + std::to_string(count));
        }
        std::vector<std::string> arr;
        arr.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            auto r = readString(file);
            if (r.isErr()) {
                return Result<GGUFValue>::err(r.error());
            }
            arr.push_back(r.value());
        }
        return Result<GGUFValue>::ok(GGUFValue{arr});
    }
    default:
        // R10: 数组元素类型为 ARRAY（嵌套数组）——GGUFValue 无法表示，
        // 且递归解析会随恶意文件深度增长，显式拒绝。
        if (elem_type == GGUFType::ARRAY) {
            return Result<GGUFValue>::err(
                "Nested arrays are not supported (array element type ARRAY)");
        }
        // Skip unsupported array types
        TLLM_WARN("Unsupported array element type: {}, skipping {} elements",
                  static_cast<uint32_t>(elem_type), count);
        // Skip the data
        for (uint64_t i = 0; i < count; ++i) {
            auto r = readValue(file, elem_type);
            if (r.isErr()) {
                return Result<GGUFValue>::err(r.error());
            }
        }
        return Result<GGUFValue>::ok(GGUFValue{std::vector<uint8_t>()});
    }
}

Result<std::string> GGUFParser::readString(std::ifstream &file) {
    uint64_t length;
    file.read(reinterpret_cast<char *>(&length), 8);

    if (!file) {
        return Result<std::string>::err("Failed to read string length");
    }

    // Sanity check
    if (length > 1024 * 1024) {
        return Result<std::string>::err("String too long: " + std::to_string(length));
    }

    if (length == 0) {
        return Result<std::string>::ok({});
    }

    std::string str(length, '\0');
    file.read(&str[0], length);

    if (!file) {
        return Result<std::string>::err("Failed to read string data");
    }

    return Result<std::string>::ok(str);
}

uint64_t GGUFParser::alignOffset(uint64_t offset) const {
    return (offset + alignment_ - 1) & ~(alignment_ - 1);
}

Result<ModelConfig> GGUFParser::extractModelConfig() const {
    ModelConfig config;

    // Helper lambdas for getting metadata values
    auto get_int = [this](const std::string &key, int &out) {
        auto it = metadata_.kv.find(key);
        if (it != metadata_.kv.end()) {
            if (auto *val = std::get_if<int32_t>(&it->second)) {
                out = *val;
                return true;
            }
            if (auto *val = std::get_if<uint32_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
            if (auto *val = std::get_if<int64_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
            if (auto *val = std::get_if<uint64_t>(&it->second)) {
                out = static_cast<int>(*val);
                return true;
            }
        }
        return false;
    };

    auto get_float = [this](const std::string &key, float &out) {
        auto it = metadata_.kv.find(key);
        if (it != metadata_.kv.end()) {
            if (auto *val = std::get_if<float>(&it->second)) {
                out = *val;
                return true;
            }
            if (auto *val = std::get_if<double>(&it->second)) {
                out = static_cast<float>(*val);
                return true;
            }
        }
        return false;
    };

    // GGUF 标准：配置键以 general.architecture 声明的架构名为前缀
    // （qwen2.embedding_length / llama.block_count / phi3.* ...）。
    // 依次尝试 {arch}.* -> llama.*（历史兼容）-> general.*（非标准兜底）。
    std::string arch = "llama";
    if (auto it = metadata_.kv.find("general.architecture"); it != metadata_.kv.end()) {
        if (const auto *arch_str = std::get_if<std::string>(&it->second)) {
            arch = *arch_str;
        }
    }

    auto get_arch_int = [&](const std::string &name, int &out) {
        return get_int(arch + "." + name, out) || get_int("llama." + name, out) ||
               get_int("general." + name, out);
    };
    auto get_arch_float = [&](const std::string &name, float &out) {
        return get_float(arch + "." + name, out) || get_float("llama." + name, out);
    };

    get_arch_int("embedding_length", config.hidden_dim);
    get_arch_int("block_count", config.num_layers);
    get_arch_int("attention.head_count", config.num_heads);
    // 修复：head_count_kv 缺失时（MHA 老 GGUF）显式回退 num_heads，
    // 避免静默保持 ModelConfig 默认值 32 导致 wk/wv 维度错配。
    if (!get_arch_int("attention.head_count_kv", config.num_kv_heads)) {
        config.num_kv_heads = config.num_heads;
    }
    get_arch_int("context_length", config.max_seq_len);
    get_arch_int("feed_forward_length", config.intermediate_dim);
    get_arch_float("attention.layer_norm_rms_epsilon", config.rms_norm_eps);
    get_arch_float("rope.freq_base", config.rope_theta);

    // 词表大小：GGUF 中不存在独立的 vocab_size 键，
    // 从 tokenizer.ggml.tokens 数组长度派生
    if (auto it = metadata_.kv.find("tokenizer.ggml.tokens"); it != metadata_.kv.end()) {
        if (const auto *tokens = std::get_if<std::vector<std::string>>(&it->second)) {
            config.vocab_size = static_cast<int>(tokens->size());
        }
    }

    // Tokenizer 特殊 token
    get_int("tokenizer.ggml.eos_token_id", config.eos_token_id);
    get_int("tokenizer.ggml.bos_token_id", config.bos_token_id);

    // Calculate derived values
    if (config.num_heads > 0) {
        config.head_dim = config.hidden_dim / config.num_heads;
    }

    // Validate essential fields
    if (config.hidden_dim <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.vocab_size <= 0) {
        TLLM_WARN("GGUF metadata may be incomplete. Using defaults for missing fields.");
        // Set reasonable defaults if missing
        if (config.hidden_dim <= 0) config.hidden_dim = 4096;
        if (config.num_layers <= 0) config.num_layers = 32;
        if (config.num_heads <= 0) config.num_heads = 32;
        if (config.num_kv_heads <= 0) config.num_kv_heads = config.num_heads;
        if (config.vocab_size <= 0) config.vocab_size = 32000;
        if (config.head_dim <= 0) config.head_dim = config.hidden_dim / config.num_heads;
        if (config.intermediate_dim <= 0) config.intermediate_dim = config.hidden_dim * 8 / 3;
    }

    TLLM_INFO("Extracted model config: hidden_dim={}, num_layers={}, num_heads={}, "
              "num_kv_heads={}, vocab_size={}",
              config.hidden_dim, config.num_layers, config.num_heads, config.num_kv_heads,
              config.vocab_size);

    return Result<ModelConfig>::ok(config);
}

Result<void> GGUFParser::validateTensorShapes(const ModelConfig &config) const {
    // 配置来自 metadata，缺失键时 extractModelConfig() 可能回落默认值；
    // 用不完整的配置做形状校验会把错误张量判成“匹配”，必须显式拒绝。
    if (config.hidden_dim <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.num_kv_heads <= 0 || config.head_dim <= 0 || config.vocab_size <= 0 ||
        config.intermediate_dim <= 0) {
        return Result<void>::err(
            "Cannot validate GGUF tensor shapes: incomplete ModelConfig "
            "(hidden_dim/num_layers/num_heads/num_kv_heads/head_dim/vocab_size/"
            "intermediate_dim must all be > 0)");
    }

    const uint64_t hidden = static_cast<uint64_t>(config.hidden_dim);
    const uint64_t vocab = static_cast<uint64_t>(config.vocab_size);
    const uint64_t inter = static_cast<uint64_t>(config.intermediate_dim);
    const uint64_t q_dim =
        static_cast<uint64_t>(config.num_heads) * static_cast<uint64_t>(config.head_dim);
    const uint64_t kv_dim =
        static_cast<uint64_t>(config.num_kv_heads) * static_cast<uint64_t>(config.head_dim);

    auto format_dims = [](const std::vector<uint64_t> &dims) {
        std::string out = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) out += ", ";
            out += std::to_string(dims[i]);
        }
        out += "]";
        return out;
    };

    std::vector<std::string> mismatches;
    auto                     mismatch = [&](const GGUFTensorInfo &t, const std::string &expected) {
        mismatches.push_back("tensor " + t.name + " has shape " + format_dims(t.dimensions) +
                                                 ", expected " + expected);
    };
    auto expect_exact = [&](const GGUFTensorInfo &t, std::vector<uint64_t> expected) {
        if (t.dimensions != expected) {
            mismatch(t, format_dims(expected));
        }
    };
    // 词表维必须是精确值：lm_head_fp16 的转置副本按 config.vocab_size 定步长，
    // 多出的 padding 列会让转置读错行（数值全错），少则越界/截断。
    auto expect_vocab = [&](const GGUFTensorInfo &t) {
        if (t.dimensions.size() != 2 || t.dimensions[0] != hidden || t.dimensions[1] != vocab) {
            mismatch(t, "[" + std::to_string(hidden) + ", " + std::to_string(vocab) + "]");
        }
    };

    // 去掉 "blk.N." / "layers.N." 前缀；层号越界或非逐层张量返回空（跳过）。
    auto layer_suffix = [&](const std::string &name) -> std::string {
        for (const char *prefix : {"blk.", "layers."}) {
            if (name.rfind(prefix, 0) != 0) continue;
            const size_t begin = std::strlen(prefix);
            const size_t dot = name.find('.', begin);
            if (dot == std::string::npos) return {};
            long long   idx = 0;
            const char *first = name.data() + begin;
            const char *last = name.data() + dot;
            auto [ptr, ec] = std::from_chars(first, last, idx);
            if (ec != std::errc() || ptr != last || idx < 0 || idx >= config.num_layers) {
                return {};
            }
            return name.substr(dot + 1);
        }
        return {};
    };

    for (const auto &t : tensors_) {
        if (t.name == "token_embd.weight" || t.name == "tok_embeddings.weight" ||
            t.name == "output.weight" || t.name == "lm_head.weight") {
            expect_vocab(t);
            continue;
        }
        if (t.name == "output_norm.weight" || t.name == "norm.weight") {
            expect_exact(t, {hidden});
            continue;
        }

        std::string suffix = layer_suffix(t.name);
        if (suffix.empty()) continue;

        if (suffix == "attn_q.weight") {
            expect_exact(t, {hidden, q_dim});
        } else if (suffix == "attn_k.weight" || suffix == "attn_v.weight") {
            expect_exact(t, {hidden, kv_dim});
        } else if (suffix == "attn_output.weight" || suffix == "attn_out.weight") {
            expect_exact(t, {q_dim, hidden});
        } else if (suffix == "ffn_gate.weight" || suffix == "ffn_up.weight") {
            expect_exact(t, {hidden, inter});
        } else if (suffix == "ffn_down.weight") {
            expect_exact(t, {inter, hidden});
        } else if (suffix == "attn_norm.weight" || suffix == "ffn_norm.weight") {
            expect_exact(t, {hidden});
        } else if (suffix == "attn_q.bias") {
            expect_exact(t, {q_dim});
        } else if (suffix == "attn_k.bias" || suffix == "attn_v.bias") {
            expect_exact(t, {kv_dim});
        }
    }

    if (!mismatches.empty()) {
        std::string joined;
        for (size_t i = 0; i < mismatches.size(); ++i) {
            if (i > 0) joined += "; ";
            joined += mismatches[i];
        }
        return Result<void>::err("GGUF tensor shape mismatch: " + joined);
    }
    return Result<void>::ok();
}

const GGUFTensorInfo *GGUFParser::getTensorByName(const std::string &name) const {
    auto it = tensor_name_map_.find(name);
    if (it != tensor_name_map_.end()) {
        return &tensors_[it->second];
    }
    return nullptr;
}

Result<std::vector<uint8_t>> GGUFParser::readTensorData(const GGUFTensorInfo &tensor) {
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to open file: " + path_);
    }

    // 溢出守卫：tensor.offset 文件可控（64 位），data_offset_ + tensor.offset
    // 回绕后 seekg 会落到错误偏移读到垃圾数据（静默损坏，比崩溃更糟）。
    // 与 llama.cpp#26978（GGML_PAD 回绕）同类的尺寸/偏移溢出问题。
    if (tensor.offset > UINT64_MAX - data_offset_) {
        return Result<std::vector<uint8_t>>::err("Tensor data offset overflows for: " +
                                                 tensor.name);
    }
    uint64_t read_offset = data_offset_ + tensor.offset;
    file.seekg(read_offset);

    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to seek to tensor data at offset " +
                                                 std::to_string(read_offset));
    }

    size_t size = tensor.calculateSize();
    if (!tensor.dimensions.empty() && size == 0) {
        return Result<std::vector<uint8_t>>::err("Tensor size overflow or unsupported type for: " +
                                                 tensor.name);
    }
    // 尺寸先与文件剩余字节比较，避免伪造 dimensions 触发先分配后读取：
    // 旧路径会对一个几百字节的文件按声明尺寸分配数 GB，构造的 GGUF 可直接
    // 触发 OOM-kill，而不是返回解析错误。readArray 已有同型检查。
    const size_t remaining = remainingBytes(file);
    if (size > remaining) {
        return Result<std::vector<uint8_t>>::err("Tensor data exceeds remaining file bytes for: " +
                                                 tensor.name);
    }
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size);

    if (!file) {
        return Result<std::vector<uint8_t>>::err("Failed to read tensor data");
    }

    return Result<std::vector<uint8_t>>::ok(data);
}

// GGUFMetadata methods
bool GGUFMetadata::has(const std::string &key) const { return kv.find(key) != kv.end(); }

// GGUFTensorInfo methods
size_t GGUFTensorInfo::numElements() const {
    if (dimensions.empty()) return 0;
    size_t n = 1;
    for (auto d : dimensions) {
        size_t next = 0;
        if (!safeMultiplySize(n, static_cast<size_t>(d), next)) {
            return 0;
        }
        n = next;
    }
    return n;
}

size_t GGUFTensorInfo::calculateSize() const {
    size_t num_elem = numElements();
    if (!dimensions.empty() && num_elem == 0) {
        return 0;
    }

    auto scaledSize = [&](size_t bytes_per_elem) -> size_t {
        size_t size = 0;
        return safeMultiplySize(num_elem, bytes_per_elem, size) ? size : 0;
    };
    auto blockScaledSize = [&](size_t block_size, size_t bytes_per_block) -> size_t {
        if (block_size == 0) {
            return 0;
        }
        const size_t blocks = (num_elem + block_size - 1) / block_size;
        size_t       size = 0;
        return safeMultiplySize(blocks, bytes_per_block, size) ? size : 0;
    };

    // Bytes per element for each type
    switch (type) {
    case GGMLType::F32:
        return scaledSize(4);
    case GGMLType::F16:
        return scaledSize(2);
    case GGMLType::I8:
        return num_elem;
    case GGMLType::I16:
        return scaledSize(2);
    case GGMLType::I32:
        return scaledSize(4);
    case GGMLType::I64:
        return scaledSize(8);
    case GGMLType::F64:
        return scaledSize(8);
    case GGMLType::Q8_0:
        // Q8_0: 32 values per block, each block has 32 int8 + 1 half scale
        return blockScaledSize(32, 32 + 2);
    case GGMLType::Q4_0:
        // Q4_0: 32 values per block, each block has 16 int8 + 1 half scale
        return blockScaledSize(32, 16 + 2);
    case GGMLType::Q4_1:
        // Q4_1: 32 values per block, each block has 16 int8 + 2 half (scale + min)
        return blockScaledSize(32, 16 + 4);
    case GGMLType::Q5_0:
        // Q5_0: 32 values per block = 2B scale + 4B high bits + 16B low quants
        return blockScaledSize(32, 22);
    case GGMLType::Q5_1:
        // Q5_1: 32 values per block = 2B scale + 2B min + 4B high bits + 16B low quants
        return blockScaledSize(32, 24);
    case GGMLType::Q2_K:
        return blockScaledSize(256, 84);
    case GGMLType::Q3_K:
        return blockScaledSize(256, 110);
    case GGMLType::Q4_K:
        return blockScaledSize(256, 144);
    case GGMLType::Q5_K:
        return blockScaledSize(256, 176);
    case GGMLType::Q6_K:
        return blockScaledSize(256, 210);
    default:
        // 未知/未支持的类型：显式失败，绝不按 FP16 估算造成静默错位读取
        TLLM_ERROR("Unsupported GGML tensor type {}", static_cast<uint32_t>(type));
        return 0;
    }
}

} // namespace tiny_llm
