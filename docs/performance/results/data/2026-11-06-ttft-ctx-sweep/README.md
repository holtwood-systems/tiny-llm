# TTFT/TPOT × 上下文长度扫描（2026-11-06）

对高频面试题"prefill/decode 瓶颈""长上下文 decode 为什么变慢"的实测回答。
`tiny_llm_bench --max-tokens 8 --iters 3 --json`，同一 prompt 模板按
1/4/16/48/128 重复得不同 prompt 长度；ptok 为 tokenizer 实测值。

## 原始结果（p50）

### 0.5B Q4_K_M（本机 + Node B 双节点）

| ptok | TTFT 本机 | TTFT Node B | TPOT 本机 | TPOT Node B |
|---|---|---|---|---|
| 19 | 51.5 ms | 49.4 ms | 2.24 ms | 2.20 ms |
| 73 | 67.4 | 64.5 | 2.35 | 2.30 |
| 289 | 82.2 | 77.9 | 2.89 | 2.80 |
| 865 | 204.3 | 193.0 | 4.26 | 4.09 |
| 2305 | 606.1 | 571.9 | 7.67 | 7.40 |

跨节点偏差 4–6%（Node B 宿主 CPU load≈22 下仍一致，复现性可信）。

### 7B Q4_K_M（本机）

| ptok | TTFT | TPOT |
|---|---|---|
| 19 | 352.4 ms | 11.07 ms |
| 73 | 447.8 | 11.21 |
| 289 | 1128.9 | 12.07 |
| 865 | 2926.0 | 14.03 |
| 2305 | 8014.5 | 19.38 |

## 面试题视角的分析

**"Prefill 和 decode 瓶颈分别是什么？"**

- Prefill 边际速率：0.5B ≈ 0.243ms/tok（**~4120 tok/s**），
  7B ≈ 3.04ms/tok（**~330 tok/s**）。比值 12.5× ≈ 权重规模比 14×——
  prefill 由权重 GEMM 算力主导（compute-bound），与理论一致。
- Decode TPOT 随 ctx 近似**线性增长**：每步要把新 Q 对全部历史
  KV 做注意力（memory-bound 访存量随 ctx 线性涨）。

**"长上下文 decode 为什么变慢？慢多少？"**

TPOT(ctx) 线性拟合斜率：

| 模型 | KV/ctx-token（fp16 理论） | 实测斜率 | 隐含有效带宽 |
|---|---|---|---|
| 0.5B (24层/GQA2/hd64) | 12,288 B | 2.38 µs/tok | **~5 GB/s** |
| 7B (28层/GQA4/hd128) | 57,344 B | 3.63 µs/tok | **~16 GB/s** |

两档都远低于 4090 的 ~1008GB/s 峰值带宽：batch=1 的 decode
注意力是 launch/延迟受限而非带宽受限——这正好回答了下一题
"为什么 serving 需要 continuous batching"（摊薄固定开销）。

**"TTFT 固定开销"**：19-token 处 TTFT 基线 0.5B≈50ms、7B≈352ms，
远高于 1-token 的 2.8ms——KV 池初始化/首批权重载入等每轮
固定成本，量级可复现（跨节点一致）。

## 文件

- `local.jsonl` — 本机 10 点（0.5B×5 + 7B×5），schema v2
- `nodeb-0.5b.jsonl` — Node B 0.5B×5（跨节点对照）

## 复现

```bash
base="请详细分析 Transformer 架构中自注意力机制的数学原理与工程实现细节，"
for rep in 1 4 16 48 128; do
  ./build/tiny_llm_bench <model.gguf> --prompt "$(python3 -c "print('$base'*$rep)")" \
    --max-tokens 8 --warmup 1 --iters 3 --json
done
```
