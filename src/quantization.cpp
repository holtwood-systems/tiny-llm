#include "tiny_llm/quantization.h"
#include "tiny_llm/logger.h"

#include <cmath>
#include <cstring>

namespace tiny_llm {

Result<std::vector<half>> convertF32ToF16(const float *f32_data, size_t num_elements) {
    if (f32_data == nullptr) {
        return Result<std::vector<half>>::err("convertF32ToF16: null pointer");
    }

    std::vector<half> f16_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        f16_data[i] = __float2half(f32_data[i]);
    }

    return Result<std::vector<half>>::ok(std::move(f16_data));
}

Result<std::vector<half>> convertF32ToF16(const std::vector<float> &f32_data) {
    return convertF32ToF16(f32_data.data(), f32_data.size());
}

Result<std::vector<half>> dequantizeQ4_0(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ4_0: null pointer");
    }

    // Q4_0: 32 values per block, each block has 16 bytes (32 x 4-bit) + 2 bytes (half scale)
    // Total: 18 bytes per block -> 32 FP16 outputs
    constexpr size_t  BLOCK_SIZE = 32;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        // Each block: scale (half) + 16 bytes of packed 4-bit values
        const half    *scale = reinterpret_cast<const half *>(data + b * 18);
        const uint8_t *packed = data + b * 18 + 2;

        float scale_f = __half2float(*scale);

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            // GGML packs value i in the low nibble of qs[i] for i < 16, and
            // value i+16 in the high nibble of qs[i]. This is the same layout
            // as Q5_0's low 4 bits, not an interleaved low/high stream.
            const uint8_t packed_byte = packed[i < 16 ? i : i - 16];
            const uint8_t nibble = (i < 16) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            // Convert from unsigned 4-bit to signed: 0-15 -> -8 to 7
            const int8_t value = static_cast<int8_t>(nibble > 7 ? nibble - 16 : nibble);

            float dequantized = scale_f * static_cast<float>(value);
            result[b * BLOCK_SIZE + i] = __float2half(dequantized);
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::vector<half>> dequantizeQ8_0(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ8_0: null pointer");
    }

    // Q8_0: 32 values per block, each block has 32 bytes (32 x 8-bit) + 2 bytes (half scale)
    // Total: 34 bytes per block -> 32 FP16 outputs
    constexpr size_t  BLOCK_SIZE = 32;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        // Each block: scale (half) + 32 bytes of int8 values
        const half   *scale = reinterpret_cast<const half *>(data + b * 34);
        const int8_t *values = reinterpret_cast<const int8_t *>(data + b * 34 + 2);

        float scale_f = __half2float(*scale);

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            float dequantized = scale_f * static_cast<float>(values[i]);
            result[b * BLOCK_SIZE + i] = __float2half(dequantized);
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

namespace {

// 从 Q4_K 的 12 字节打包 scale 区提取子块 j (0..7) 的 6-bit scale 与 4-bit min。
// 位布局与 GGML 参考实现一致：
//   j < 4:  sc = scales[j] & 63;          m = scales[j+4] & 63
//   j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//           m  = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
void getScaleMinK4(int j, const uint8_t *scales, uint8_t &sc, uint8_t &m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0x0F) | static_cast<uint8_t>((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | static_cast<uint8_t>((scales[j] >> 6) << 4);
    }
}

} // namespace

Result<std::vector<half>> dequantizeQ5_0(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ5_0: null pointer");
    }

    // Q5_0: 32 values per block = d (2B) + qh (4B, bit i = 值 i 的第 5 位) + qs (16B 低 4 位)
    constexpr size_t  BLOCK_SIZE = 32;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t *block = data + b * 22;
        const float    d = __half2float(*reinterpret_cast<const half *>(block));
        uint32_t       qh;
        std::memcpy(&qh, block + 2, sizeof(qh));
        const uint8_t *qs = block + 6;

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            const uint8_t low = (i < 16) ? (qs[i] & 0x0F) : (qs[i - 16] >> 4);
            const uint8_t high = (qh >> i) & 1u;
            const int     q = static_cast<int>(low | (high << 4)) - 16;
            result[b * BLOCK_SIZE + i] = __float2half(d * static_cast<float>(q));
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::vector<half>> dequantizeQ4_K(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ4_K: null pointer");
    }

    // Q4_K: 256 values per 144-byte block = d (2B) + dmin (2B) + scales (12B) + qs (128B)
    // qs 的 4 个 32 字节组各拆成低/高 nibble 两个子块，共 8 个子块 x 32 值
    constexpr size_t  BLOCK_SIZE = 256;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t *block = data + b * 144;
        const float    d = __half2float(*reinterpret_cast<const half *>(block));
        const float    dmin = __half2float(*reinterpret_cast<const half *>(block + 2));
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;

        half *out = result.data() + b * BLOCK_SIZE;
        for (int g = 0; g < 4; ++g) {
            const uint8_t *q = qs + 32 * g;
            for (int nibble = 0; nibble < 2; ++nibble) {
                uint8_t sc, m;
                getScaleMinK4(g * 2 + nibble, scales, sc, m);
                const float dl = d * sc;
                const float ml = dmin * m;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t v = (nibble == 0) ? (q[l] & 0x0F) : (q[l] >> 4);
                    *out++ = __float2half(dl * v - ml);
                }
            }
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::vector<half>> dequantizeQ6_K(const uint8_t *data, size_t num_blocks) {
    if (data == nullptr) {
        return Result<std::vector<half>>::err("dequantizeQ6_K: null pointer");
    }

    // Q6_K: 256 values per 210-byte block = ql (128B 低 4 位) + qh (64B 高 2 位)
    //       + scales (16 x int8, 每 16 值一个) + d (2B)
    constexpr size_t  BLOCK_SIZE = 256;
    std::vector<half> result(num_blocks * BLOCK_SIZE);

    for (size_t b = 0; b < num_blocks; ++b) {
        const uint8_t *block = data + b * 210;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const auto    *scales = reinterpret_cast<const int8_t *>(block + 192);
        const float    d = __half2float(*reinterpret_cast<const half *>(block + 208));

        half *out = result.data() + b * BLOCK_SIZE;
        // 8 行 x 32 quant。行 r 与输出位置的映射由参考实现展平顺序推出：
        // 输出下标 = s*16 + l%16, 其中 s = 2*r + l/16
        for (int r = 0; r < 8; ++r) {
            const int ql_base = (r % 2) * 32 + (r / 4) * 64;
            const int qh_base = (r / 4) * 32;
            const int ql_shift = ((r / 2) % 2) * 4;
            const int qh_shift = (r % 4) * 2;
            for (int l = 0; l < 32; ++l) {
                const int low = (ql[ql_base + l] >> ql_shift) & 0x0F;
                const int high = (qh[qh_base + l] >> qh_shift) & 0x03;
                const int q = (low | (high << 4)) - 32;
                const int s = 2 * r + l / 16;
                out[s * 16 + l % 16] = __float2half(d * scales[s] * q);
            }
        }
    }

    return Result<std::vector<half>>::ok(std::move(result));
}

Result<std::pair<std::vector<int8_t>, std::vector<half>>>
quantizeF16ToW8A16(const half *f16_data, int rows, int cols, int group_size) {
    if (f16_data == nullptr) {
        return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::err(
            "quantizeF16ToW8A16: null pointer");
    }

    if (rows <= 0 || cols <= 0 || group_size <= 0) {
        return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::err(
            "quantizeF16ToW8A16: invalid dimensions");
    }

    size_t total_elements = static_cast<size_t>(rows) * cols;
    int    scale_rows = (rows + group_size - 1) / group_size;
    size_t scale_elements = static_cast<size_t>(scale_rows) * cols;

    std::vector<int8_t> quantized(total_elements);
    std::vector<half>   scales(scale_elements);

    // Process each column independently
    for (int c = 0; c < cols; ++c) {
        // Process groups within the column
        for (int g = 0; g < scale_rows; ++g) {
            int group_start = g * group_size;
            int group_end = std::min(group_start + group_size, rows);

            // Find max absolute value in group
            float max_abs = 0.0f;
            for (int r = group_start; r < group_end; ++r) {
                float val = __half2float(f16_data[r * cols + c]);
                max_abs = std::max(max_abs, std::abs(val));
            }

            // Calculate scale (avoid division by zero)
            // R7: 极小/零组保持 scale >= fp16 最小正常数，而非钳到 1.0——
            // 后者会把整组量化为 round(v/1.0)=0，静默丢失量级信息。
            constexpr float kHalfMinNormal = 6.1035156e-5f; // 2^-14
            float           scale = max_abs / 127.0f;
            if (scale < kHalfMinNormal) {
                scale = kHalfMinNormal; // fp16 可表示且非零
            }

            scales[static_cast<size_t>(g) * cols + c] = __float2half(scale);

            // Quantize values
            for (int r = group_start; r < group_end; ++r) {
                float val = __half2float(f16_data[r * cols + c]);
                int   quantized_val = static_cast<int>(std::round(val / scale));
                // Clamp to int8 range
                quantized_val = std::max(-128, std::min(127, quantized_val));
                quantized[static_cast<size_t>(r) * cols + c] = static_cast<int8_t>(quantized_val);
            }
        }
    }

    return Result<std::pair<std::vector<int8_t>, std::vector<half>>>::ok(
        {std::move(quantized), std::move(scales)});
}

} // namespace tiny_llm
