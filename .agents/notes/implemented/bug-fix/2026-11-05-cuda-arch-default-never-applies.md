# Agent Note: CUDA 架构默认值从未生效——CMake 先注入 52，`NOT DEFINED` 守护空转

Status: implemented

## Problem

2026-11-05 在新机器（RTX 4090，sm_89，nvcc 12.8，驱动 550.107.02 / CUDA
UMD 12.4，cmake 4.0.2）上跑 `scripts/run-gpu-gate.sh`，252 个测试中
约 90 个失败：几乎所有触自研 CUDA kernel 的用例（split-KV、attention、
paged-kv、RoPE、w8a16 matmul、sampling）全挂，失败模式为输出全零
（`has_nonzero=false`、argmax 返回 0）。

两层根因：

1. **arch 默认逻辑从未生效**。`check_language(CUDA)` / `enable_language(CUDA)`
   的编译器检测先把
   `CMAKE_CUDA_ARCHITECTURES` 初始化为 nvcc 默认（52），之后
   `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` 永远跳过——`native`
   默认值自写入起没有在任何机器上赋过值（CMP0104=NEW 下 cmake 4.0.2
   行为相同，最小复现已验证）。于是整库编成 `compute_52` PTX +
   `sm_52` cubin。
2. **PTX 版本被驱动拒绝**。sm_52 cubin 不能在 sm_89 上跑（cubin 不跨
   大版本兼容），只能走 PTX JIT；而 nvcc 12.8 生成的 PTX ISA 版本高于
   驱动 550（CUDA UMD 12.4）JIT 支持的版本 → 每次 launch 报
   `cudaErrorUnsupportedPtxVersion (222)`（compute-sanitizer 实测），
   kernel 从未执行，测试侧缺少 launch 错误检查，拿到全零输出。
   3060 Laptop（驱动 610 / UMD 13.3 + toolkit 12.0）全绿是
   "驱动新于工具链"恰好成立，不是设计。

同一份代码 `-DCMAKE_CUDA_ARCHITECTURES=89` 重编后，曾失败的用例全部
通过（sm_89 cubin 直接执行，不经 JIT）。

## Decision

在 `check_language(CUDA)` 之前把"用户显式传入"这一事实落到普通变量
`_TLLM_USER_CUDA_ARCH`（命令行 `-D` 与 toolchain 文件在此可见；
`check_language` 自身就会先把变量填成 52，放它后面同样失效——已实测）；
`enable_language` 之后按该哨兵判断：用户未传且未设 `CUDAARCHS`
环境变量时，`native`（cmake ≥3.24）或 `70;75;80;86;89`（旧 cmake）。

显式 `-DCMAKE_CUDA_ARCHITECTURES=` 与环境变量 `CUDAARCHS` 均视为用户
指定，不再覆盖。

测试层的 launch 错误检查缺口（kernel 未启动被读成全零数据）留作独立
改进；门禁侧先保证按目标架构构建，不把 PTX JIT 当作可移植性前提。

## Alternatives considered

- **`if(NOT DEFINED OR STREQUAL "")`**：最直觉的补丁，但实测值是 52
  而非空串，判断不到；等于没修。
- **只在门禁/CI 显式传 `-DCMAKE_CUDA_ARCHITECTURES`**：只修触发路径，
  任何手工 `cmake -B build` 的贡献者仍会拿到 52 构建；守护代码留着
  继续说谎。拒绝——修默认逻辑本身。
- **删除默认逻辑、要求所有构建显式指定**：最诚实但把成本转嫁给每台
  新机器/每位贡献者；`native` 在常规开发机上足够好。

## Consequences

- `build/` 的 CMakeCache 已含 `ARCHITECTURES:STRING=52`，旧缓存不会因
  本改动自动刷新；`cmake -S -B` 重复 configure 保留缓存值。换机或首跑
  前需用干净构建目录（本次验证用 `build-sm89/` 隔离）。
- 证据链：失败批日志 `docs/performance/results/data/2026-11-05-gpu-gate.log`
  （sm_52 + JIT 被拒，93 项失败）；compute-sanitizer 输出确认
  `cudaErrorUnsupportedPtxVersion (222)`；sm_89 构建全绿批随后落盘。
- **经验**：驱动 CUDA UMD 版本 < 工具链版本的机器上，任何依赖 PTX JIT
  的"兜底 arch"都不是兜底而是全灭；`CUDAARCHS` 列表里最后一个
  `compute_XX` 只救得了驱动 ≥ 工具链的场景。
- 本机另有进程常驻 ~42.7 GiB 显存（nvidia-smi 可见），大显存用例
  （如整模型 KV cache 分配）可能 OOM，属环境约束而非代码缺陷。
