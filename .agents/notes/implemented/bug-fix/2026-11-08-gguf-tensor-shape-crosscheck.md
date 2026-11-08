# Agent Note: GGUF 张量形状与 ModelConfig 交叉校验

Status: implemented

## Problem

`ModelLoader::loadGGUF` 只校验必要张量“存在”，不校验它们的 dimensions 是否与
`ModelConfig` 一致。`load_quantized` 直接取 `dimensions[0]` / `dimensions[1]`
作为 `in_f` / `out_f` 并转置：

```cpp
int in_f = static_cast<int>(tensor->dimensions[0]);
int out_f = static_cast<int>(tensor->dimensions[1]);
```

维度写反（如 `ffn_down.weight` 的 `[in, out]` 颠倒）或词表维与 tokenizer 长度
不一致的文件，元素总数往往仍然“合法”，于是会按错误布局完成反量化、转置、
W8A16 重量化并上传 GPU，直到推理产出错误数值才暴露。

词表维尤其隐蔽：`lm_head_fp16_t` 的转置副本按 `config.vocab_size` 定步长，
padding 列会让 `transpose_f16` 读错行，数值全错但不报错。

## Decision

在 `GGUFParser` 上新增 host-only 的 `validateTensorShapes(const ModelConfig &)`，
按 GGUF 的反序维度约定（`dimensions[0]` = 最内层 / in_features）校验已知张量：

| 张量 | 期望 dimensions |
| --- | --- |
| `token_embd` / `output` / `lm_head` | `[hidden, vocab]`（精确相等） |
| `attn_q.weight` | `[hidden, num_heads * head_dim]` |
| `attn_k/v.weight` | `[hidden, num_kv_heads * head_dim]` |
| `attn_output.weight` | `[num_heads * head_dim, hidden]` |
| `ffn_gate/up.weight` | `[hidden, intermediate]` |
| `ffn_down.weight` | `[intermediate, hidden]` |
| `attn_norm` / `ffn_norm` / `output_norm` | `[hidden]` |
| `attn_q.bias` | `[num_heads * head_dim]` |
| `attn_k/v.bias` | `[num_kv_heads * head_dim]` |

`blk.N.` 与 `layers.N.` 两种前缀都识别；缺失张量仍由 `ModelLoader` 的统一
缺失列表报告，本函数只校验文件中实际存在的已知张量，未知张量跳过。配置字段
不完整时直接拒绝校验，避免 `extractModelConfig()` 的默认回退把错误形状判成
匹配。

`ModelLoader::loadGGUF` 在缺失检查之后、任何 GPU 分配之前调用：

```cpp
if (auto shape_check = parser.validateTensorShapes(config); shape_check.isErr()) {
    return Result<ModelWeights>::err(shape_check.error());
}
```

错误信息一次性列出全部不匹配项，例如：

```text
GGUF tensor shape mismatch: tensor blk.0.ffn_down.weight has shape [8, 16], expected [16, 8]
```

## Alternatives considered

- **只校验元素总数（audit 建议的 `expectElements`）。** 维度写反时元素总数
  不变，恰好漏掉最危险的布局错误；否决。
- **在 `ModelLoader` 内用 lambda 校验。** 无法脱离 CUDA 编译/测试；放进
  `GGUFParser`（纯 host 代码）后可用 shim 直接跑真实 GTest。采用后者。
- **允许词表维 padding（tensor 行数 > `config.vocab_size`）。** `lm_head_fp16_t`
  的转置按 `config.vocab_size` 定步长，padding 会让转置读错行；在支持 padding
  之前必须精确相等；否决。
- **在 `parse()` 阶段强制所有张量形状正确。** `parse()` 是通用解析器，不应
  绑定具体模型架构；形状校验需要 `ModelConfig`，放在单独 API 更清晰。

## Consequences

Benefits:

- 维度颠倒、截断或词表不一致的 GGUF 在分配 GPU 内存之前失败，错误可定位到
  具体张量与期望形状。
- 校验逻辑是纯 host 代码，可独立于 CUDA 工具链测试。

Costs and limits:

- 只覆盖当前加载器认识的张量名；新增架构/张量时需要同步扩展表。
- 允许 `dimensions[1]` 更大的词表 padding 模型会被拒绝（当前引擎不支持该
  布局，拒绝优于静默错误）。
- 层号超出 `num_layers` 的逐层张量被跳过（加载器本来也忽略它们），不在本轮
  范围内。

## Verification

- 用最小 logger / cuda_fp16 shim 编译真实 `src/gguf_parser.cpp` 与
  `tests/test_gguf_parser.cpp`，链接 miniconda 的 GTest：
  **10/10 通过**，其中新增 4 条：
  `ValidatesTensorShapesAgainstConfig`、`RejectsTransposedTensorShape`、
  `RejectsMismatchedVocabDimension`、`RejectsShapeValidationWithIncompleteConfig`。
- 独立 host harness 覆盖：合法形状通过、`ffn_down` 写反被拒、词表 40/16 均被拒、
  一维 norm 长度错误被拒、未知张量跳过、配置不完整被拒；全部通过。
- 本机无 nvcc/GPU，`ModelLoader::loadGGUF` 的调用点未编译、未跑真实 GGUF；
  `tiny_llm_tests` 待有 CUDA 的环境执行。
- `git diff --check`：通过。
