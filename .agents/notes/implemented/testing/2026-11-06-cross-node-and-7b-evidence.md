# 跨节点复现 + 7B 扩展性证据（2026-11-06）

## Problem

全部 GPU 证据来自单一节点（本机 12.8.61）；模型证据上限 0.5B，
跨节点可复现性与更大模型路径均未验证。

## Decision

1. 同 commit `9afe0c0`（HEAD `338bbcf`，docs-only 差异）在第二台
   RTX 4090/同驱动/CUDA 12.8.93 节点重跑 gate 与 decode bench。
2. 以 Qwen2.5-7B-Instruct Q4_K_M 跑 `tiny_llm_bench`，记录 TTFT/
   TPOT/tok/s/RSS。
3. 镜像 compat 驱动库导致的 CUDA 804 作为环境坑证据保留
   （修复前后双 gate 日志）。

## Consequences

- Node B gate 252/252（修复后，72.84s）；decode TPOT 2.281ms vs
  本机 2.267ms（+0.6%，噪声内）→ 补丁版 toolkit 差异不影响结论。
- **环境坑可复用结论**：CUDA 官方镜像 `cuda-*/compat/libcuda.so.*`
  是数据中心卡专用 forward-compat 驱动，经 ldconfig 劫持后消费卡
  cudaGetDeviceCount 即 804；`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
  绕回真驱动即可。文档见
  `docs/performance/results/data/2026-11-06-nodeb-rtx4090-cuda128.93/README.md`。
- 7B 实测：TPOT 11.26ms / 88.8 tok/s / RSS +23GB；GGUF 解析与
  W8A16+Graphs 路径在 7B/GQA 下无需改动。
- 限制：Node B 为租用临时节点（可能已回收），复现命令已写入
  证据 README；7B 跑时有外部租户占 3.5GiB 显存。

## Verification

- `gate-nodeb-fixed.log`：252/252，72.84s
- `bench-nodeb-graphs.json`：tpot 2.281ms、438.4 tok/s
- `2026-11-06-rtx4090-7b-graphs.json`：schema v2
- 修复前对照：`gate-nodeb-broken-804.log`（4 FFI 失败+GPU 测试 skip）
