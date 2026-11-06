# 跨节点复现 · Node B RTX 4090 + CUDA 12.8.93（2026-11-06）

同型号 RTX 4090、同驱动 550.107.02、不同节点、不同 toolkit 补丁版
（本机 12.8.61 vs Node B 12.8.93）下的 gate/bench 复现，以及一个
镜像级环境缺陷的定位与修复。

## 环境差异

| 项 | 本机（Node A） | Node B |
|---|---|---|
| GPU | RTX 4090 49140MiB | RTX 4090 49140MiB |
| Driver | 550.107.02 | 550.107.02 |
| nvcc | 12.8.61 | 12.8.93 |
| 镜像 | 自建 | CUDA 官方镜像 |
| commit | `9afe0c0` | `9afe0c0`（HEAD `338bbcf`，仅 docs 差异） |

## 结果

| 套件 | Node A（12.8.61） | Node B（12.8.93） |
|---|---|---|
| GPU gate | 252/252, 0 skip | **252/252**, 1 skip（`Fp16MatmulTransposedMatchesUntransposed` 见下） |
| decode graphs TPOT | ~2.267 ms | 2.281 ms（+0.6%） |
| decode tok/s | ~441 | 438.4 |

跨节点结论：同 commit、同硬件、同驱动下，toolkit 补丁版差异对
正确性与 decode 性能影响在噪声内。

## 镜像缺陷：compat 驱动库劫持 → CUDA error 804（重点）

CUDA 官方镜像在 `/usr/local/cuda-12.8/compat/` 放置了
`libcuda.so.570.124.06`（**数据中心 GPU 专用 forward-compat 驱动**），
并通过 `/etc/ld.so.conf.d/00-compat-*.conf` 把它注册进 ldconfig 最高
优先级。RTX 4090 是消费卡，**不支持 forward-compat 路径**：

- 症状：`cudaGetDeviceCount` 直接返回 804
  （`forward compatibility was attempted on non supported HW`），
  所有 CUDA 测试 skip/失败；`nvidia-smi` 却正常。
- 诊断：最小 probe（`probe.cu`，见日志 `gate-nodeb-broken-804.log`
  同场景）确认 context init 即失败，与构建无关。
- 修复：`export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
  使 runtime dlopen 真驱动 `libcuda.so.550.107.02`，
  gate 立即 252/252（72.84s，`gate-nodeb-fixed.log`）。

**可复用结论**：消费级 GPU + CUDA 官方镜像时，先检查
`ldconfig -p | grep libcuda` 是否指向 `cuda-*/compat/`；
此坑表象是"工具链坏了"，实为镜像默认路径劫持，排查顺序应为
`nvidia-smi` → `cudaGetDeviceCount` probe → `ldconfig` → fatbin。

## 文件

- `gate-nodeb-broken-804.log` — 修复前 gate（4 FFI 失败 + 大量 GPU 测试 skip）
- `gate-nodeb-fixed.log` — 修复后 252/252
- `bench-nodeb-graphs.json` — decode bench JSON（schema v2）
- `nodeb-env.txt` — 镜像 compat 配置 / 驱动 / 内存 / 磁盘

## 复现

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu   # 关键：绕开 compat 劫持
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
TLLM_GGUF_TEST_MODEL=...q4_k_m.gguf TLLM_GGUF_TEST_MODEL_2=...q8_0.gguf \
  ctest --test-dir build -j4
```
