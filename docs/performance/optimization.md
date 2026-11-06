# 优化

本页区分**已实现的优化**与**计划中的优化**。计划项的落地顺序见 [ROADMAP](https://github.com/holtwood-systems/tiny-llm/blob/master/ROADMAP.md)。

## 已实现的优化

### 1. W8A16 量化推理路径

- INT8 权重 + FP16 激活，分组量化（`QuantizationParams::group_size`，默认 128，对称量化无零点）
- 权重显存占用约为 FP16 的 1/2（不含 scale 开销）
- 实现：`kernels/w8a16_matmul.cu`、`src/quantization.cpp`

### 2. 显存预分配的 KV Cache

- `KVCacheManager` 按 `KVCacheConfig`（层数、头数、head_dim、max_seq_len、max_batch_size）一次性预分配
- 避免 decode 阶段动态分配；显存上界在启动时即可计算
- 每序列 KV 显存估算公式：`2 × num_layers × num_kv_heads × head_dim × max_seq_len × 2 bytes`

### 3. 共享内存与 warp 级 kernel 模式

- `kernels/` 下的 attention / rmsnorm / elementwise kernel 使用共享内存分块与 warp 级归约
- GQA 支持：`ModelConfig::num_kv_heads` 与 `num_heads` 可不同

### 4. M==1 转置权重 GEMM 快路径（decode 核心，C1）

- 根因：M==1 kernel 的 `lane=k` 映射 × 权重 `[K,N]` 布局 → 跨 lane 地址
  `stride=N×2B`（非 coalesced）。
- 解法：加载时构建 `[N,K]` 转置权重（`data_t`/`scales_t`，`QuantizedWeight`
  新增字段），M==1 走 `w8a16_matmul_m1_transposed_kernel` /
  `fp16_matmul_m1_transposed_kernel`，lane 沿 k 连续读，coalesced。
- 效果：lm_head FP16 ~10.0 → ~0.98 ms；W8A16 N=4864 ~0.163 → ~0.049 ms。
- 实现：`kernels/transpose_weights.cu`、`kernels/w8a16_matmul.cu`、
  `src/model_loader.cpp`（加载时构建转置副本 + freeWeights 释放）。

### 5. CUDA Graphs decode 默认开启（C2）

- decode 的确定性 device 序列（`runDecodeDevicePath`）捕获为 CUDA Graph，
  step 变化值（visible_len / RoPE pos / append pos / token id）改为 device 端
  int + 固定缓冲，host 每次重放前更新。
- 默认开启；`TLLM_CUDA_GRAPHS=0` 显式关闭（opt-out）。捕获失败自动回退。
- 设计细节见 `docs/performance/cuda-graphs.md`。

## 计划中的优化（按 ROADMAP 顺序）

| 优化项 | 预期收益 | 状态 |
|--------|----------|------|
| 真实模型端到端 + 基准脚本 | 使所有优化可度量 | ✅ 已完成（`tiny_llm_bench`） |
| CUDA Graphs | 降低 decode 阶段 kernel launch 开销 | ✅ 已完成（默认开启） |
| M==1 转置权重 GEMM 快路径 | 消解 decode GEMM 非 coalesced 访存 | ✅ 已完成（C1） |
| Tensor Core WMMA（FP16/INT8） | GEMM 再提速（未做 `mma.sync`） | 待做 |
| 分页 KV（策略 1） | block table + scatter/gather，与连续 KV 逐 token 差分 | ✅ 已完成 |
| FlashDecoding | 长上下文 decode attention | 待做；只复用算法经验，不把 cuflash 接入 generate 路径 |
| fused kernel（rmsnorm/RoPE/QKV/bias） | 减少 kernel 数与中间显存 | 待做 |
| KV Cache swapping/offload | 支持超出显存的并发序列 | 待做 |
| 连续批处理 | 吞吐导向场景 | 待做（调度层设计见 [paged-serving](https://github.com/holtwood-systems/paged-serving)） |

> 已完成项的开关和复现方式以代码与对应结果页为准；待做项在实现前不写虚构开关或
> 预估加速比。

## 内存估算方法（理论值，非实测）

推理时显存主要由三部分组成：

| 组件 | 估算公式 |
|------|----------|
| 模型权重 (W8A16) | `num_params × 1 byte + scale 开销` |
| KV Cache | `2 × num_layers × num_kv_heads × head_dim × max_seq_len × batch × 2 bytes` |
| 激活与临时缓冲 | 与 hidden_dim、seq_len 相关，按实际分配统计 |

`tiny_llm_bench` 当前报告加载前与运行后的**常驻显存差值**；这不是进程峰值。
真实峰值需要用外部采样器或 profiler 记录运行期间的最高占用，并单独注明采样频率。

## 通用测量建议

1. 用 `nvidia-smi dmon` 观察 GPU 利用率与显存带宽
2. 用 Nsight Systems 看时间线空洞，用 Nsight Compute 看 kernel 级瓶颈（见 [分析](./profiling)）
3. 测量时固定时钟与功率限制，避免 boost 抖动影响对比
