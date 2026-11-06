# 2026-08-23 CUDA Graph clean-commit A/B（RTX 3060 Laptop）

> 这是 benchmark schema v2 的正式本地基准。10 个进程均从 clean commit
> `565da79bd8f2d11005c04130dbe942c87e810586` 启动，原始输出与机器可读汇总随报告
> 一并归档。结论只覆盖本文固定的硬件、模型与 64-token greedy decode。

## 1. 环境与边界

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-23 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（`nvidia-smi` 6144 MiB；CUDA API 报告 6143 MiB） |
| Windows 驱动 | 610.88 |
| CUDA Toolkit / runtime | nvcc 12.0.140 / runtime 12000；driver API 13030 |
| 被测 commit | `565da79bd8f2d11005c04130dbe942c87e810586` |
| 工作区 | 每个进程启动前为 clean；`cmake --build build --target tiny_llm_bench` 显示无待构建项 |
| 模型 | Qwen2.5-0.5B-Instruct GGUF Q4_K_M；加载后重量化为 W8A16 |
| 模型 SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`（491400032 bytes） |
| prompt / 输出 | `你好`（1 token）/ 最多 64 新 token，greedy |
| 每个进程 | 1 次 Graph capture 预热 + 3 warmup + 10 timed iterations |
| 重复 | 5 组独立进程对，按 on→off、off→on、on→off、off→on、on→off 交错 |
| 频率 | 未锁频；记录真实 Laptop/WSL2 波动，不外推为数据中心 GPU 结果 |

WSL 默认 loader 被 Linux `libcuda` 包遮蔽，本机命令显式使用
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`。这只修正驱动库选择，不改变 benchmark 参数。

## 2. 复现命令与数据

```bash
MODEL=/home/shane/github/open-infra-ai/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  ./build/tiny_llm_bench "$MODEL" --prompt "你好" \
  --max-tokens 64 --warmup 3 --iters 10 --json --graphs

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  ./build/tiny_llm_bench "$MODEL" --prompt "你好" \
  --max-tokens 64 --warmup 3 --iters 10 --json --no-graphs
```

- [10 个进程的原始 JSONL](https://github.com/holtwood-systems/tiny-llm/blob/master/docs/performance/results/data/2026-08-23-cuda-graphs-ab.raw.jsonl)
- [机器可读聚合与配对差值](data/2026-08-23-cuda-graphs-ab.summary.json)

Graph on 五组均为 `enabled=true,captured=true`，off 五组均为
`enabled=false,captured=false`；每组均生成 64 token。

## 3. 十组原始汇总

| 组 | 顺序 | Graph | TTFT mean / p50 (ms) | TPOT mean / p50 (ms) | decode tok/s | 常驻显存差值 (MB) |
|----|------|-------|----------------------|----------------------|--------------|---------------------|
| 1 | 先 | on | 10.029 / 9.496 | 5.225 / 5.205 | 191.384 | 3368 |
| 1 | 后 | off | 9.621 / 9.185 | 8.304 / 8.272 | 120.426 | 3364 |
| 2 | 先 | off | 9.023 / 9.072 | 8.322 / 8.336 | 120.168 | 3364 |
| 2 | 后 | on | 8.871 / 8.746 | 5.064 / 5.056 | 197.466 | 3368 |
| 3 | 先 | on | 10.100 / 10.144 | 5.845 / 5.713 | 171.097 | 3368 |
| 3 | 后 | off | 8.634 / 8.853 | 8.513 / 8.612 | 117.467 | 3364 |
| 4 | 先 | off | 9.505 / 9.645 | 8.430 / 8.500 | 118.621 | 3364 |
| 4 | 后 | on | 9.377 / 8.822 | 5.209 / 5.213 | 191.976 | 3368 |
| 5 | 先 | on | 8.729 / 8.683 | 5.307 / 5.324 | 188.413 | 3368 |
| 5 | 后 | off | 8.426 / 8.335 | 8.146 / 7.903 | 122.766 | 3364 |

## 4. 聚合结论

主表对每种模式取五个“进程内统计值”的中位数：

| 指标 | Graph off | Graph on | 变化 |
|------|-----------|----------|------|
| TTFT p50 的跨进程中位数 | 9.072 ms | 8.822 ms | -2.8% |
| TPOT mean 的跨进程中位数 | 8.322 ms | 5.225 ms | **-37.2%** |
| decode tok/s 的跨进程中位数 | 120.168 | 191.384 | **+59.3%** |
| 常驻显存差值 | 3364 MB | 3368 MB | +4 MB（非峰值） |

再对五组一一配对后取变化率中位数：TTFT **+3.4%**、TPOT **-37.1%**、decode
吞吐 **+58.9%**。TTFT 的单组变化范围为 -8.5%～+14.6%，且两种聚合方法方向相反，
因此不能声称 Graph 改善了 TTFT；它在本机噪声范围内。TPOT 与吞吐则 5/5 组方向一致。

结论只支持：在这台 RTX 3060 Laptop、这个模型与 64-token decode 上，Graph 重放显著
减少当前实现的 launch/host 调度开销，TPOT 降低约 37%，decode 吞吐提高约 59%，常驻
显存差值增加 4 MB。它不支持外推到其他 GPU、模型、batch 或长上下文，也不能写成
“整体推理加速 59%”。

## 5. 正确性与 profiler 状态

- `InferenceEngineTest.CudaGraphsGenerateMatchesNonGraph` 用真实模型跑 1 项通过，
  on/off 16 token 逐 token 相等，并断言同请求 TTFT 已记录。
- 本轮首轮提交前默认配置 193/193 测试通过；真实 FFI 4/4 与 Graph 1/1 门控测试通过。
- `ncu` 2022.4.1 返回 `ERR_NVGPUCTRPERM`，没有采到性能计数器。
- `nsys profile` 2022.4.2 可生成 `.qdstrm`，但本机缺 importer，不能转换报告或运行 stats。

因此本报告证明 clean commit 的端到端 A/B、输出一致性与重复性；尚不能用正式
timeline/warp-stall 数据完成因果归因。后续在 profiler 权限完整的 Linux 云 GPU 上补
nsys/ncu，并在更多模型、上下文与 batch 上复测，才可扩大结论边界。
