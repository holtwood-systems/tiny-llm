#include "tiny_llm/model_loader.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/logger.h"
#include "tiny_llm/quantization.h"
#include "transpose_weights.cuh"

#include <algorithm>
#include <cstring>

namespace tiny_llm {

Result<ModelWeights> ModelLoader::loadGGUF(const std::string &path, ModelConfig &config) {
    TLLM_INFO("Loading GGUF model from: {}", path);

    // Use the new GGUFParser
    GGUFParser parser(path);
    auto       parse_result = parser.parse();
    if (parse_result.isErr()) {
        TLLM_ERROR("Failed to parse GGUF: {}", parse_result.error());
        return Result<ModelWeights>::err("Failed to parse GGUF: " + parse_result.error());
    }

    // Extract model config
    auto config_result = parser.extractModelConfig();
    if (config_result.isErr()) {
        TLLM_ERROR("Failed to extract model config: {}", config_result.error());
        return Result<ModelWeights>::err("Failed to extract model config: " +
                                         config_result.error());
    }
    config = config_result.value();

    TLLM_INFO("Model config: hidden_dim={}, num_layers={}, vocab_size={}", config.hidden_dim,
              config.num_layers, config.vocab_size);

    const auto &tensors = parser.getTensors();
    TLLM_DEBUG("Found {} tensors", tensors.size());

    // Build tensor name map for quick lookup
    std::unordered_map<std::string, const GGUFTensorInfo *> tensor_map;
    for (const auto &t : tensors) {
        tensor_map[t.name] = &t;
    }

    // Helper function to find tensor
    auto find_tensor = [&](const std::string &name) -> const GGUFTensorInfo * {
        auto it = tensor_map.find(name);
        if (it != tensor_map.end()) {
            return it->second;
        }
        return nullptr;
    };

    auto find_any_tensor =
        [&](std::initializer_list<const char *> names) -> const GGUFTensorInfo * {
        for (const char *name : names) {
            if (const auto *tensor = find_tensor(name)) {
                return tensor;
            }
        }
        return nullptr;
    };

    std::vector<std::string> missing_tensors;
    auto                     require_tensor = [&](std::initializer_list<const char *> names) {
        if (!find_any_tensor(names)) {
            std::string joined;
            for (const char *name : names) {
                if (!joined.empty()) {
                    joined += " | ";
                }
                joined += name;
            }
            missing_tensors.push_back(std::move(joined));
        }
    };

    require_tensor({"token_embd.weight", "tok_embeddings.weight"});
    for (int layer = 0; layer < config.num_layers; ++layer) {
        const std::string layer_prefix = "blk." + std::to_string(layer) + ".";
        const std::string llama_prefix = "layers." + std::to_string(layer) + ".";
        auto              require_layer_tensor = [&](std::initializer_list<const char *> suffixes) {
            for (const char *suffix : suffixes) {
                if (find_tensor(layer_prefix + suffix) || find_tensor(llama_prefix + suffix)) {
                    return;
                }
            }
            std::string joined;
            for (const char *suffix : suffixes) {
                if (!joined.empty()) {
                    joined += " | ";
                }
                joined += layer_prefix + suffix;
                joined += " | ";
                joined += llama_prefix + suffix;
            }
            missing_tensors.push_back(std::move(joined));
        };

        require_layer_tensor({"attn_q.weight"});
        require_layer_tensor({"attn_k.weight"});
        require_layer_tensor({"attn_v.weight"});
        require_layer_tensor({"attn_output.weight", "attn_out.weight"});
        require_layer_tensor({"ffn_gate.weight"});
        require_layer_tensor({"ffn_down.weight"});
        require_layer_tensor({"ffn_up.weight"});
        require_layer_tensor({"attn_norm.weight"});
        require_layer_tensor({"ffn_norm.weight"});
    }

    require_tensor({"output_norm.weight", "norm.weight"});
    // 任务 A4：tied output embedding 兼容 —— GGUF 中无 output.weight/lm_head.weight
    // 但存在 token_embd.weight 时，lm_head 复用 token_embd.weight（tied）。
    // 只有当两者都不存在时才报 missing（token_embd 缺失本身已在上面单独报告）。
    if (!find_any_tensor({"output.weight", "lm_head.weight"}) &&
        !find_any_tensor({"token_embd.weight", "tok_embeddings.weight"})) {
        missing_tensors.push_back("output.weight | lm_head.weight");
    }

    // 任务 A4：显式校验不支持的 bias 张量（不静默忽略）。
    // 当前仅支持 Qwen2 系逐层 attention bias（blk.*/layers.* 的 attn_q/k/v.bias），
    // 其余任何 *.bias（如 output.bias / token_embd.bias / ffn_*.bias）直接报错。
    for (const auto &t : tensors) {
        if (t.name.find(".bias") == std::string::npos) continue;
        bool supported = false;
        for (int layer = 0; layer < config.num_layers && !supported; ++layer) {
            const std::string pfx_blk = "blk." + std::to_string(layer) + ".";
            const std::string pfx_llama = "layers." + std::to_string(layer) + ".";
            for (const char *b : {"attn_q.bias", "attn_k.bias", "attn_v.bias"}) {
                if (t.name == pfx_blk + b || t.name == pfx_llama + b) {
                    supported = true;
                    break;
                }
            }
        }
        if (!supported) {
            return Result<ModelWeights>::err(
                "Unsupported bias tensor (only blk.*/layers.* attn_q/k/v.bias are supported): " +
                t.name);
        }
    }

    if (!missing_tensors.empty()) {
        std::string missing;
        for (size_t i = 0; i < missing_tensors.size(); ++i) {
            if (i > 0) {
                missing += ", ";
            }
            missing += missing_tensors[i];
        }
        return Result<ModelWeights>::err("GGUF runtime tensors missing: " + missing);
    }

    // 形状与配置交叉校验：维度写反（如 ffn_down 的 [in,out] 颠倒）的元素总数
    // 往往仍然“合法”，但会按错误布局加载并产出错误数值。必须在分配 GPU
    // 内存前拒绝。
    if (auto shape_check = parser.validateTensorShapes(config); shape_check.isErr()) {
        TLLM_ERROR("GGUF tensor shape validation failed: {}", shape_check.error());
        return Result<ModelWeights>::err(shape_check.error());
    }

    // 所有必要 tensor 已验证存在且形状匹配，开始读取、转换并上传 GPU
    ModelWeights weights;
    bool         success = false;
    auto         cleanup_on_error = [&]() {
        if (!success) {
            freeWeights(weights);
        }
    };

    try {
        const int group_sz = 128;

        // 读取 tensor 原始数据并反量化为 FP16
        auto load_f16 = [&](const GGUFTensorInfo *tensor) -> Result<std::vector<half>> {
            auto data_result = parser.readTensorData(*tensor);
            if (data_result.isErr()) {
                return Result<std::vector<half>>::err(data_result.error());
            }
            const auto &raw = data_result.value();
            size_t      num = tensor->numElements();

            switch (tensor->type) {
            case GGMLType::F16:
                return Result<std::vector<half>>::ok(
                    std::vector<half>(reinterpret_cast<const half *>(raw.data()),
                                      reinterpret_cast<const half *>(raw.data()) + num));
            case GGMLType::F32:
                return convertF32ToF16(reinterpret_cast<const float *>(raw.data()), num);
            case GGMLType::Q4_0:
                return dequantizeQ4_0(raw.data(), (num + 31) / 32);
            case GGMLType::Q8_0:
                return dequantizeQ8_0(raw.data(), (num + 31) / 32);
            case GGMLType::Q5_0:
                return dequantizeQ5_0(raw.data(), (num + 31) / 32);
            case GGMLType::Q4_K:
                return dequantizeQ4_K(raw.data(), (num + 255) / 256);
            case GGMLType::Q6_K:
                return dequantizeQ6_K(raw.data(), (num + 255) / 256);
            default:
                return Result<std::vector<half>>::err(
                    "Unsupported GGML type for tensor " + tensor->name +
                    ": supported types are F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K");
            }
        };

        // 转置: GGUF 行主序 [out, in] -> tiny-llm 行主序 [in, out]
        auto transpose_f16 = [](std::vector<half> &data, int out_f, int in_f) {
            std::vector<half> dst(static_cast<size_t>(in_f) * out_f);
            for (int o = 0; o < out_f; ++o)
                for (int i = 0; i < in_f; ++i)
                    dst[static_cast<size_t>(i) * out_f + o] =
                        data[static_cast<size_t>(o) * in_f + i];
            data = std::move(dst);
        };

        // 反量化 -> 转置 -> 重量化 W8A16 -> 上传 GPU
        auto load_quantized = [&](const GGUFTensorInfo *tensor) -> Result<QuantizedWeight> {
            if (!tensor) return Result<QuantizedWeight>::err("Tensor not found");
            // 修复：损坏/恶意 GGUF 的权重 tensor 不足 2 维时，直接下标访问
            // dimensions[1] 是未定义行为，这里显式校验。
            if (tensor->dimensions.size() < 2) {
                return Result<QuantizedWeight>::err(
                    "Tensor " + tensor->name +
                    " has <2 dimensions (dims=" + std::to_string(tensor->dimensions.size()) + ")");
            }

            int in_f = static_cast<int>(tensor->dimensions[0]);
            int out_f = static_cast<int>(tensor->dimensions[1]);

            auto f16_result = load_f16(tensor);
            if (f16_result.isErr()) return Result<QuantizedWeight>::err(f16_result.error());
            auto f16_data = f16_result.value();

            transpose_f16(f16_data, out_f, in_f);

            auto q = quantizeF16ToW8A16(f16_data.data(), in_f, out_f, group_sz);
            if (q.isErr()) return Result<QuantizedWeight>::err(q.error());
            auto [int8_data, scales] = q.value();

            QuantizedWeight qw;
            qw.rows = in_f;
            qw.cols = out_f;
            qw.group_size = group_sz;

            CUDA_CHECK(cudaMalloc(&qw.data, int8_data.size()));
            CUDA_CHECK(cudaMalloc(&qw.scales, scales.size() * sizeof(half)));
            CUDA_CHECK(
                cudaMemcpy(qw.data, int8_data.data(), int8_data.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(qw.scales, scales.data(), scales.size() * sizeof(half),
                                  cudaMemcpyHostToDevice));

            // 任务 C1：构建 [cols, rows] 转置副本，供 M==1 decode 快路径使用。
            // data_t 是 int8 数组：元素数==字节数，但显式写 sizeof(int8_t) 表明
            // 单位是字节，避免未来换量化元素类型时踩坑。
            CUDA_CHECK(cudaMalloc(&qw.data_t, qw.weightElements() * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&qw.scales_t, qw.scaleElements() * sizeof(half)));
            kernels::transpose_int8(qw.data, qw.data_t, qw.rows, qw.cols, /*stream=*/0);
            kernels::transpose_scales(qw.scales, qw.scales_t, qw.scaleRows(), qw.cols,
                                      /*stream=*/0);

            return Result<QuantizedWeight>::ok(qw);
        };

        // 反量化 -> 直接上传 FP16 到 GPU
        auto load_fp16 = [&](const GGUFTensorInfo *tensor) -> Result<half *> {
            if (!tensor) return Result<half *>::err("Tensor not found");
            auto f16_result = load_f16(tensor);
            if (f16_result.isErr()) return Result<half *>::err(f16_result.error());
            const auto &f16_data = f16_result.value();
            half       *d_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&d_ptr, f16_data.size() * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(d_ptr, f16_data.data(), f16_data.size() * sizeof(half),
                                  cudaMemcpyHostToDevice));
            return Result<half *>::ok(d_ptr);
        };

        // --- Token embedding (FP16) ---
        auto *emb = find_any_tensor({"token_embd.weight", "tok_embeddings.weight"});
        auto  emb_r = load_fp16(emb);
        if (emb_r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("token embedding: " + emb_r.error());
        }
        weights.token_embedding = emb_r.value();

        // --- 每层权重 ---
        weights.layers.resize(config.num_layers);
        for (int layer = 0; layer < config.num_layers; ++layer) {
            auto             &lw = weights.layers[layer];
            const std::string pfx_blk = "blk." + std::to_string(layer) + ".";
            const std::string pfx_llama = "layers." + std::to_string(layer) + ".";

            auto find_l =
                [&](std::initializer_list<const char *> suffixes) -> const GGUFTensorInfo * {
                for (const char *s : suffixes) {
                    if (auto *t = find_tensor(pfx_blk + s)) return t;
                    if (auto *t = find_tensor(pfx_llama + s)) return t;
                }
                return nullptr;
            };

            auto load_qw = [&](QuantizedWeight                    &target,
                               std::initializer_list<const char *> names) -> Result<void> {
                auto *t = find_l(names);
                auto  r = load_quantized(t);
                if (r.isErr()) {
                    std::string n;
                    for (const char *nm : names) {
                        if (!n.empty()) n += "|";
                        n += nm;
                    }
                    return Result<void>::err("Layer " + std::to_string(layer) + " " + n + ": " +
                                             r.error());
                }
                target = r.value();
                return Result<void>::ok();
            };

            // Qwen2 系 attention 带 bias（Llama 系无此 tensor，缺失时保持 nullptr）
            // R12: tensor 存在但加载失败（IO/内存错误）必须报错——静默降级为
            // nullptr 会让推理带着缺失的 bias 跑出错误数值。
            auto load_bias = [&](const char *name) -> Result<half *> {
                auto *t = find_l({name});
                if (!t) return Result<half *>::ok(nullptr);
                auto r = load_fp16(t);
                if (r.isErr()) {
                    return Result<half *>::err("layer " + std::to_string(layer) + " bias " + name +
                                               " load failed: " + r.error());
                }
                return Result<half *>::ok(r.value());
            };

            if (auto r = load_qw(lw.wq, {"attn_q.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.wk, {"attn_k.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.wv, {"attn_v.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.wo, {"attn_output.weight", "attn_out.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.w1, {"ffn_gate.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.w2, {"ffn_down.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }
            if (auto r = load_qw(lw.w3, {"ffn_up.weight"}); r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(r.error());
            }

            // Attention bias（Qwen2 系）
            auto bq = load_bias("attn_q.bias");
            if (bq.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(bq.error());
            }
            lw.wq_bias = bq.value();
            auto bk = load_bias("attn_k.bias");
            if (bk.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(bk.error());
            }
            lw.wk_bias = bk.value();
            auto bv = load_bias("attn_v.bias");
            if (bv.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err(bv.error());
            }
            lw.wv_bias = bv.value();

            // RMSNorm 权重 (FP16)
            auto att_r = load_fp16(find_l({"attn_norm.weight"}));
            if (att_r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " attn_norm: " + att_r.error());
            }
            lw.rms_att_weight = att_r.value();

            auto ffn_r = load_fp16(find_l({"ffn_norm.weight"}));
            if (ffn_r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " ffn_norm: " + ffn_r.error());
            }
            lw.rms_ffn_weight = ffn_r.value();
        }

        // --- Final norm (FP16) ---
        auto fn_r = load_fp16(find_any_tensor({"output_norm.weight", "norm.weight"}));
        if (fn_r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("final norm: " + fn_r.error());
        }
        weights.final_norm_weight = fn_r.value();

        // --- LM head ---
        // 量化版本（W8A16）作为后备；同时加载 FP16 版本用于 logits 精度（output 层不量化）
        auto lm_t = find_any_tensor({"output.weight", "lm_head.weight"});
        if (!lm_t) {
            // 任务 A4：tied output embedding —— lm_head 复用 token_embd.weight。
            // 值相同但布局不同：embedding 表为 [vocab, hidden]，lm_head 需转置为
            // [hidden, vocab]，因此下面仍按独立副本加载（非指针别名），无双重释放问题。
            lm_t = emb;
            TLLM_INFO("Tied output embedding detected: lm_head uses token_embd.weight");
        }
        if (lm_t->dimensions.size() < 2) {
            cleanup_on_error();
            return Result<ModelWeights>::err("LM head tensor " + lm_t->name + " has <2 dimensions");
        }
        auto lm_r = load_quantized(lm_t);
        if (lm_r.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("LM head: " + lm_r.error());
        }
        weights.lm_head = lm_r.value();

        // FP16 lm_head：load_fp16 输出列主 [in, out]，需转置为行主 [K, N]（与量化路径一致）
        auto lm_f16_raw = load_f16(lm_t);
        if (lm_f16_raw.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("LM head fp16: " + lm_f16_raw.error());
        }
        auto lm_f16_vec = lm_f16_raw.value();
        transpose_f16(lm_f16_vec, static_cast<int>(lm_t->dimensions[1]),
                      static_cast<int>(lm_t->dimensions[0]));
        half *lm_fp16_d = nullptr;
        CUDA_CHECK(cudaMalloc(&lm_fp16_d, lm_f16_vec.size() * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(lm_fp16_d, lm_f16_vec.data(), lm_f16_vec.size() * sizeof(half),
                              cudaMemcpyHostToDevice));
        weights.lm_head_fp16 = lm_fp16_d;

        // 任务 C1：构建 lm_head_fp16 的转置副本 [vocab, hidden]（M==1 decode 快路径）。
        // lm_f16_vec 此时为行主 [hidden, vocab]（上面 transpose_f16 已转）。
        const int hidden_f = static_cast<int>(config.hidden_dim);
        const int vocab_f = static_cast<int>(config.vocab_size);
        CUDA_CHECK(cudaMalloc(&weights.lm_head_fp16_t,
                              static_cast<size_t>(hidden_f) * vocab_f * sizeof(half)));
        kernels::transpose_fp16(weights.lm_head_fp16, weights.lm_head_fp16_t, hidden_f, vocab_f,
                                /*stream=*/0);

        // 修复（跨流数据竞争）：上面的 transpose kernel 在默认流上异步执行，而推理
        // kernel 在引擎私有 stream 上消费 data_t/scales_t/lm_head_fp16_t。加载完成
        // 与首次使用之间没有其他跨流同步点，必须在此显式等待默认流排空，否则理论上
        // 首次 decode 可读到未写完的转置权重。
        CUDA_CHECK(cudaStreamSynchronize(0));

        success = true;
        return Result<ModelWeights>::ok(std::move(weights));

    } catch (...) {
        // 修复：CUDA_CHECK(cudaMalloc/cudaMemcpy) 抛异常时同样清理已上传
        // 的 GPU 权重（原 cleanup_on_error 只在显式 isErr() 返回路径执行，
        // 异常会直接穿透导致 GPU 内存泄漏）。
        cleanup_on_error();
        throw;
    }
}

Result<ModelWeights> ModelLoader::loadBin(const std::string &path, const ModelConfig &config) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Result<ModelWeights>::err("Failed to open file: " + path);
    }

    // Read and validate header
    BinHeader header;
    file.read(reinterpret_cast<char *>(&header.magic), sizeof(header.magic));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary header magic");
    }

    if (header.magic != BIN_MAGIC) {
        return Result<ModelWeights>::err("Invalid binary magic number: expected 0x" +
                                         std::to_string(BIN_MAGIC) + ", got 0x" +
                                         std::to_string(header.magic));
    }

    file.read(reinterpret_cast<char *>(&header.version), sizeof(header.version));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read binary version");
    }

    if (header.version != BIN_VERSION) {
        return Result<ModelWeights>::err("Unsupported binary version: " +
                                         std::to_string(header.version));
    }

    // Read stored config
    file.read(reinterpret_cast<char *>(&header.config), sizeof(header.config));
    if (!file) {
        return Result<ModelWeights>::err("Failed to read model config");
    }

    // Validate dimensions match
    if (header.config.hidden_dim != config.hidden_dim ||
        header.config.num_layers != config.num_layers ||
        header.config.vocab_size != config.vocab_size) {
        return Result<ModelWeights>::err(
            "Model config mismatch: expected hidden_dim=" + std::to_string(config.hidden_dim) +
            ", num_layers=" + std::to_string(config.num_layers) +
            ", vocab_size=" + std::to_string(config.vocab_size) +
            ", got hidden_dim=" + std::to_string(header.config.hidden_dim) +
            ", num_layers=" + std::to_string(header.config.num_layers) +
            ", vocab_size=" + std::to_string(header.config.vocab_size));
    }

    ModelWeights weights;
    bool         success = false;
    auto         cleanup_on_error = [&]() {
        if (!success) {
            freeWeights(weights);
        }
    };

    try {
        // Read token embedding [vocab_size, hidden_dim] as FP16
        size_t            embed_size = static_cast<size_t>(config.vocab_size) * config.hidden_dim;
        std::vector<half> embed_host(embed_size);
        file.read(reinterpret_cast<char *>(embed_host.data()), embed_size * sizeof(half));
        if (!file) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read token embedding");
        }

        // Allocate and transfer embedding to GPU
        CUDA_CHECK(cudaMalloc(&weights.token_embedding, embed_size * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(weights.token_embedding, embed_host.data(), embed_size * sizeof(half),
                              cudaMemcpyHostToDevice));

        // Read layer weights
        weights.layers.resize(config.num_layers);
        int group_size = 128; // Default group size

        for (int layer = 0; layer < config.num_layers; ++layer) {
            auto &lw = weights.layers[layer];

            // Read attention weights
            auto load_qweight = [&](QuantizedWeight &qw, int rows, int cols) -> Result<void> {
                auto result = loadQuantizedTensor(file, rows, cols, group_size);
                if (result.isErr()) {
                    return Result<void>::err(result.error());
                }
                qw = result.value();
                return Result<void>::ok();
            };

            int hidden = config.hidden_dim;
            int kv_dim = config.num_kv_heads * config.head_dim;
            int inter = config.intermediate_dim;

            // Q, K, V, O projections
            auto r = load_qweight(lw.wq, hidden, hidden);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " wq: " + r.error());
            }

            r = load_qweight(lw.wk, hidden, kv_dim);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " wk: " + r.error());
            }

            r = load_qweight(lw.wv, hidden, kv_dim);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " wv: " + r.error());
            }

            r = load_qweight(lw.wo, hidden, hidden);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " wo: " + r.error());
            }

            // FFN weights
            r = load_qweight(lw.w1, hidden, inter);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " w1: " + r.error());
            }

            r = load_qweight(lw.w2, inter, hidden);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " w2: " + r.error());
            }

            r = load_qweight(lw.w3, hidden, inter);
            if (r.isErr()) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Layer " + std::to_string(layer) +
                                                 " w3: " + r.error());
            }

            // RMSNorm weights (FP16)
            std::vector<half> norm_host(hidden);

            file.read(reinterpret_cast<char *>(norm_host.data()), hidden * sizeof(half));
            if (!file) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Failed to read rms_att_weight for layer " +
                                                 std::to_string(layer));
            }
            CUDA_CHECK(cudaMalloc(&lw.rms_att_weight, hidden * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(lw.rms_att_weight, norm_host.data(), hidden * sizeof(half),
                                  cudaMemcpyHostToDevice));

            file.read(reinterpret_cast<char *>(norm_host.data()), hidden * sizeof(half));
            if (!file) {
                cleanup_on_error();
                return Result<ModelWeights>::err("Failed to read rms_ffn_weight for layer " +
                                                 std::to_string(layer));
            }
            CUDA_CHECK(cudaMalloc(&lw.rms_ffn_weight, hidden * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(lw.rms_ffn_weight, norm_host.data(), hidden * sizeof(half),
                                  cudaMemcpyHostToDevice));
        }

        // Read final norm weight
        std::vector<half> final_norm_host(config.hidden_dim);
        file.read(reinterpret_cast<char *>(final_norm_host.data()),
                  config.hidden_dim * sizeof(half));
        if (!file) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read final norm weight");
        }
        CUDA_CHECK(cudaMalloc(&weights.final_norm_weight, config.hidden_dim * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(weights.final_norm_weight, final_norm_host.data(),
                              config.hidden_dim * sizeof(half), cudaMemcpyHostToDevice));

        // Read LM head
        auto lm_result =
            loadQuantizedTensor(file, config.hidden_dim, config.vocab_size, group_size);
        if (lm_result.isErr()) {
            cleanup_on_error();
            return Result<ModelWeights>::err("Failed to read LM head: " + lm_result.error());
        }
        weights.lm_head = lm_result.value();

        // 修复（跨流数据竞争）：loadQuantizedTensor 内的 transpose 在默认流异步执行，
        // 返回前显式等待，保证调用方拿到句柄时转置权重已就绪。
        CUDA_CHECK(cudaStreamSynchronize(0));

        success = true;
        return Result<ModelWeights>::ok(std::move(weights));

    } catch (...) {
        cleanup_on_error();
        throw;
    }
}

Result<QuantizedWeight> ModelLoader::loadQuantizedTensor(std::ifstream &file, int rows, int cols,
                                                         int group_size) {

    QuantizedWeight qw;
    qw.rows = rows;
    qw.cols = cols;
    qw.group_size = group_size;

    // Read INT8 weights
    size_t              weight_size = qw.weightElements();
    std::vector<int8_t> weight_host(weight_size);
    file.read(reinterpret_cast<char *>(weight_host.data()), weight_size);
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read quantized weights");
    }

    // Read scales
    size_t            scale_size = qw.scaleElements();
    std::vector<half> scale_host(scale_size);
    file.read(reinterpret_cast<char *>(scale_host.data()), scale_size * sizeof(half));
    if (!file) {
        return Result<QuantizedWeight>::err("Failed to read scale factors");
    }

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&qw.data, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales, scale_size * sizeof(half)));

    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(qw.data, weight_host.data(), weight_size * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(qw.scales, scale_host.data(), scale_size * sizeof(half),
                          cudaMemcpyHostToDevice));

    // 任务 C1：构建 [cols, rows] 转置副本，供 M==1 decode 快路径使用。
    CUDA_CHECK(cudaMalloc(&qw.data_t, weight_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&qw.scales_t, scale_size * sizeof(half)));
    kernels::transpose_int8(qw.data, qw.data_t, qw.rows, qw.cols, /*stream=*/0);
    kernels::transpose_scales(qw.scales, qw.scales_t, qw.scaleRows(), qw.cols, /*stream=*/0);

    // 修复（跨流数据竞争）：本方法是公开 API，调用方可能在任意 stream 上消费
    // data_t/scales_t；返回前等待默认流，保证转置副本已就绪。
    CUDA_CHECK(cudaStreamSynchronize(0));

    return Result<QuantizedWeight>::ok(qw);
}

void ModelLoader::freeWeights(ModelWeights &weights) {
    if (weights.token_embedding) {
        cudaFree(weights.token_embedding);
        weights.token_embedding = nullptr;
    }

    for (auto &layer : weights.layers) {
        // Free quantized weights
        auto free_qw = [](QuantizedWeight &qw) {
            if (qw.data) {
                cudaFree(qw.data);
                qw.data = nullptr;
            }
            if (qw.scales) {
                cudaFree(qw.scales);
                qw.scales = nullptr;
            }
            // 任务 C1：转置副本（[cols, rows] 布局）
            if (qw.data_t) {
                cudaFree(qw.data_t);
                qw.data_t = nullptr;
            }
            if (qw.scales_t) {
                cudaFree(qw.scales_t);
                qw.scales_t = nullptr;
            }
        };

        free_qw(layer.wq);
        free_qw(layer.wk);
        free_qw(layer.wv);
        free_qw(layer.wo);
        free_qw(layer.w1);
        free_qw(layer.w2);
        free_qw(layer.w3);

        if (layer.rms_att_weight) {
            cudaFree(layer.rms_att_weight);
            layer.rms_att_weight = nullptr;
        }
        if (layer.rms_ffn_weight) {
            cudaFree(layer.rms_ffn_weight);
            layer.rms_ffn_weight = nullptr;
        }
        auto free_bias = [](half *&b) {
            if (b) {
                cudaFree(b);
                b = nullptr;
            }
        };
        free_bias(layer.wq_bias);
        free_bias(layer.wk_bias);
        free_bias(layer.wv_bias);
    }
    weights.layers.clear();

    if (weights.final_norm_weight) {
        cudaFree(weights.final_norm_weight);
        weights.final_norm_weight = nullptr;
    }

    if (weights.lm_head_fp16) {
        cudaFree(weights.lm_head_fp16);
        weights.lm_head_fp16 = nullptr;
    }
    // 任务 C1：lm_head_fp16 转置副本（[vocab, hidden]）
    if (weights.lm_head_fp16_t) {
        cudaFree(weights.lm_head_fp16_t);
        weights.lm_head_fp16_t = nullptr;
    }

    if (weights.lm_head.data) {
        cudaFree(weights.lm_head.data);
        weights.lm_head.data = nullptr;
    }
    if (weights.lm_head.scales) {
        cudaFree(weights.lm_head.scales);
        weights.lm_head.scales = nullptr;
    }
    // 任务 C1：lm_head W8A16 转置副本
    if (weights.lm_head.data_t) {
        cudaFree(weights.lm_head.data_t);
        weights.lm_head.data_t = nullptr;
    }
    if (weights.lm_head.scales_t) {
        cudaFree(weights.lm_head.scales_t);
        weights.lm_head.scales_t = nullptr;
    }
}

} // namespace tiny_llm
