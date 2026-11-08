# Agent Note: Q4_0 反量化改为 GGML split-half 布局

Status: implemented

## Problem

`dequantizeQ4_0` 把 32 个 4-bit 值按「交错」顺序展开：

```cpp
for (size_t i = 0; i < BLOCK_SIZE; ++i) {
    uint8_t packed_byte = packed[i / 2];
    if (i % 2 == 0) value = packed_byte & 0x0F;
    else            value = packed_byte >> 4;
}
```

即输出位置 `2j` 取 `qs[j]` 低 nibble、`2j+1` 取高 nibble。GGML/GGUF 的
`block_q4_0` 与 `block_q5_0` 使用同一约定：低 nibble 是值 `j`，高 nibble 是值
`j+16`。同文件的 `dequantizeQ5_0` 已经是正确布局：

```cpp
const uint8_t low = (i < 16) ? (qs[i] & 0x0F) : (qs[i - 16] >> 4);
```

因此 Q4_0 的每个 32 值块都被静默重排。任何 Q4_0 GGUF 经
`model_loader` 加载后，权重 K 维索引与激活不一致，attention/FFN/LM head 的数值
全部偏离参考实现：模型能跑、不报错，但输出错误。

测试覆盖也印证了这一点：`tests/test_quantization.cpp` 只覆盖 Q5_0/Q4_K/Q6_K，
真实模型 fixture 是 Q4_K_M，不经过 Q4_0。README 与
`docs/performance/index.md` 却把 Q4_0 列在「与 Python gguf 参考差分对比 ✅」内。

## Decision

把 Q4_0 的取值改为与 GGML/Q5_0 一致的 split-half 布局：

```cpp
const uint8_t packed_byte = packed[i < 16 ? i : i - 16];
const uint8_t nibble = (i < 16) ? (packed_byte & 0x0F) : (packed_byte >> 4);
const int8_t value = static_cast<int8_t>(nibble > 7 ? nibble - 16 : nibble);
```

同时新增 `DequantizeQ4_0.MatchesGGMLSplitHalfLayout`：用 `d=1.0`、
`qs[i] = (i << 4) | (15 - i)` 的合成块覆盖 0..15 所有 nibble；期望值由 GGML
布局直接给出（低 nibble 依次 15..0 → -1..-8，高 nibble 依次 0..15 → 0..7）。
旧交错实现会让该测试失败。

文档中的能力表述改为「Q4_0/Q5_0/Q4_K/Q6_K 有合成块参考测试，真实模型
Q4_K_M 覆盖 Q5_0/Q4_K/Q6_K」，不再把 Q4_0 写成真实模型已验证。

## Alternatives considered

- **只改测试期望对齐现有交错实现。** 那会把错误布局固化成契约，并继续产出
  与 llama.cpp / gguf 参考实现不同的权重；否决。
- **在 `model_loader` 里加载后重排 Q4_0。** 多一次全量拷贝与转换，且把
  格式知识放进 loader 而不是量化模块；否决。
- **等真实 Q4_0 GGUF fixture 再修。** 合成块已经能逐 nibble 证明布局；真实
  fixture 可作为后续端到端验证，但不阻塞正确性修复；否决等待。
- **同时补 Q8_0 测试。** Q8_0 是逐字节 int8 直通，布局风险低；本轮只修有
  证据的 Q4_0，避免扩大范围。

## Consequences

Benefits:

- Q4_0 GGUF 的权重不再被静默重排；加载后数值与 GGML 参考实现一致。
- Q4_0 首次有独立于本仓库实现的合成块测试。

Costs and limits:

- 本机无 nvcc/GPU，未执行 `tiny_llm_tests`；新增 GTest 只在有 CUDA 的 CI/本机
  才会真正运行。
- 只覆盖单个 32 值块的布局，未覆盖真实 Q4_0 模型的端到端 token 差分。
- `npm run verify-notes` 未运行：仓库无 `node_modules`，离线无法执行。

## Verification

- 独立 host 复现：把 `src/quantization.cpp` 与最小 `cuda_fp16.h` shim 一起编译
  （不链接 CUDA），对 `d=1.0, qs[i]=(i<<4)|(15-i)` 的块调用真实的
  `dequantizeQ4_0`。修复前 `mismatches vs GGML split-half layout: 30 / 32`；
  修复后 `0 / 32`。
- 新增 GTest 用例的期望值与上述 harness 相同；CUDA 12.2.2 devel 容器完整
  编译 `tiny_llm_tests`（261 项），`GGUFParser.*` / `DequantizeQ4_0.*` 等
  18 项 host-only 用例在无 GPU 环境全部通过。
- `git diff --check`：通过。
