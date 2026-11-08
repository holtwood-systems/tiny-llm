#pragma once

#include "tiny_llm/result.h"
#include "tiny_llm/types.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tiny_llm {

// GGUF magic number
constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

// GGUF value types (as defined in GGUF spec)
enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// GGML tensor types
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4, // Removed, keep for compatibility
    Q4_3 = 5, // Removed, keep for compatibility
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
};

/**
 * @brief GGUF metadata value (variant type)
 */
using GGUFValue =
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float,
                 double, bool, std::string, std::vector<uint8_t>, std::vector<int8_t>,
                 std::vector<uint16_t>, std::vector<int16_t>, std::vector<uint32_t>,
                 std::vector<int32_t>, std::vector<uint64_t>, std::vector<int64_t>,
                 std::vector<float>, std::vector<double>, std::vector<std::string>>;

/**
 * @brief GGUF metadata container
 */
struct GGUFMetadata {
    std::unordered_map<std::string, GGUFValue> kv;

    /**
     * @brief Check if key exists
     */
    bool has(const std::string &key) const;

    /**
     * @brief Get value with type conversion
     * @tparam T Expected type
     * @param key Key to look up
     * @return Result with value or error
     */
    template <typename T>
    Result<T> get(const std::string &key) const;

    /**
     * @brief Get value with default
     * @tparam T Expected type
     * @param key Key to look up
     * @param default_value Default if not found
     * @return Value or default
     */
    template <typename T>
    T getOr(const std::string &key, T default_value) const;
};

/**
 * @brief GGUF tensor information
 */
struct GGUFTensorInfo {
    std::string           name;
    std::vector<uint64_t> dimensions;
    GGMLType              type;
    uint64_t              offset;

    /**
     * @brief Calculate tensor size in bytes
     */
    size_t calculateSize() const;

    /**
     * @brief Get number of elements
     */
    size_t numElements() const;
};

/**
 * @brief GGUF file header
 */
struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

/**
 * @brief GGUF file parser
 *
 * Parses GGUF format files and extracts model configuration and tensor info.
 *
 * Usage:
 * @code
 * GGUFParser parser("model.gguf");
 * auto result = parser.parse();
 * if (result.isOk()) {
 *   auto config = parser.extractModelConfig();
 *   const auto& tensors = parser.getTensors();
 *   // ...
 * }
 * @endcode
 */
class GGUFParser {
  public:
    explicit GGUFParser(const std::string &path);

    /**
     * @brief Parse the GGUF file
     * @return Result indicating success or error
     */
    Result<void> parse();

    /**
     * @brief Get the parsed metadata
     */
    const GGUFMetadata &getMetadata() const { return metadata_; }

    /**
     * @brief Get the tensor information
     */
    const std::vector<GGUFTensorInfo> &getTensors() const { return tensors_; }

    /**
     * @brief Get the file header
     */
    const GGUFHeader &getHeader() const { return header_; }

    /**
     * @brief Extract model configuration from metadata
     * @return ModelConfig or error
     */
    Result<ModelConfig> extractModelConfig() const;

    /**
     * @brief Validate tensor dimensions against the model configuration
     *
     * GGUF 的 ne[] 是反序维度：dimensions[0] 是最内层（in_features），
     * dimensions[1] 是 out_features。校验必须在分配 GPU 内存前完成，
     * 否则维度写反或截断的文件会按错误布局静默加载。
     *
     * 只校验文件中存在的已知张量；缺失张量由 ModelLoader 统一报告。
     * 词表维要求与 config.vocab_size 精确相等：lm_head 的 FP16 转置副本
     * 按 config.vocab_size 定步长，padding 会导致转置读错行。
     *
     * @param config 由同一文件 extractModelConfig() 得到的配置
     * @return Result indicating success or a list of mismatches
     */
    Result<void> validateTensorShapes(const ModelConfig &config) const;

    /**
     * @brief Get tensor by name
     * @param name Tensor name
     * @return Pointer to tensor info or nullptr if not found
     */
    const GGUFTensorInfo *getTensorByName(const std::string &name) const;

    /**
     * @brief Read tensor data from file
     * @param tensor Tensor information
     * @return Raw tensor data or error
     */
    Result<std::vector<uint8_t>> readTensorData(const GGUFTensorInfo &tensor);

    /**
     * @brief Get the data section offset in file
     */
    uint64_t getDataOffset() const { return data_offset_; }

  private:
    // Parse methods
    Result<void> parseHeader(std::ifstream &file);
    Result<void> parseMetadata(std::ifstream &file);
    Result<void> parseTensorInfo(std::ifstream &file);

    // Read methods
    Result<GGUFValue>      readValue(std::ifstream &file, GGUFType type);
    Result<GGUFValue>      readArray(std::ifstream &file);
    Result<std::string>    readString(std::ifstream &file);
    Result<GGUFTensorInfo> readTensorInfoEntry(std::ifstream &file);

    // Calculate alignment padding
    uint64_t alignOffset(uint64_t offset) const;

    std::string                             path_;
    GGUFHeader                              header_{};
    GGUFMetadata                            metadata_;
    std::vector<GGUFTensorInfo>             tensors_;
    std::unordered_map<std::string, size_t> tensor_name_map_;
    uint64_t                                data_offset_ = 0;
    // Alignment (GGUF spec 默认 32；若 metadata 含 general.alignment 则覆盖，
    // 且必须为 2 的幂)。数据区起点按此对齐。
    uint64_t alignment_ = 32;
};

// Template implementations
template <typename T>
Result<T> GGUFMetadata::get(const std::string &key) const {
    auto it = kv.find(key);
    if (it == kv.end()) {
        return Result<T>::err("Key not found: " + key);
    }

    if (auto *val = std::get_if<T>(&it->second)) {
        return Result<T>::ok(*val);
    }

    return Result<T>::err("Type mismatch for key: " + key);
}

template <typename T>
T GGUFMetadata::getOr(const std::string &key, T default_value) const {
    auto result = get<T>(key);
    return result.isOk() ? result.value() : default_value;
}

} // namespace tiny_llm
