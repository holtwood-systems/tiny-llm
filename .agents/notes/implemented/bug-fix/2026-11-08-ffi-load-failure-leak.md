# Agent Note: tinyllm_load 失败路径不再泄漏 GPU 资源

Status: implemented

## Problem

`tinyllm_load` 把资源直接挂在 `TinyLlmHandleImpl` 的裸指针上，只有成功返回后
调用 `tinyllm_free` 才会释放。中间步骤失败时函数直接 `return nullptr`：

- GGUF 解析 / 配置提取 / `Validator::validateModelConfig` 失败（此时尚未分配，
  无泄漏）；
- `ModelLoader::loadGGUF` 成功之后的分页参数校验 `return nullptr`；
- `KVCacheManager::create` 失败；
- `LayerWorkspace::allocate`、`cudaMalloc`、`cudaStreamCreate` 抛异常（进入
  `catch`）。

这些路径上 `std::unique_ptr<TinyLlmHandleImpl> h` 会析构，但
`TinyLlmHandleImpl` 没有析构函数，`weights`、`rope_cos/rope_sin`、
`hidden_buf`、`logits_buf`、分页 K/V pool 与 scratch 全部泄漏。对长驻服务
（paged-serving 反复加载/卸载模型）这是可持续累积的显存泄漏。

## Decision

把 `tinyllm_free` 中的释放逻辑移入 `TinyLlmHandleImpl::~TinyLlmHandleImpl()`：

```cpp
~TinyLlmHandleImpl() {
    if (rope_cos) cudaFree(rope_cos);
    ...
    if (stream) cudaStreamDestroy(stream);
    tiny_llm::ModelLoader::freeWeights(weights);
}
```

`LayerWorkspace`、`DeviceBuffer`、`KVCacheManager`、`TransformerLayer` 已有
RAII，成员析构自动接管；裸指针与 `ModelWeights` 由析构函数回收。
`tinyllm_free` 只剩 `delete reinterpret_cast<TinyLlmHandleImpl *>(handle)`。

保持原有释放顺序：先销毁 stream，再 `freeWeights`（与旧 `tinyllm_free`
一致）；`cudaFree` 自身与设备工作同步，不存在释放后仍有 kernel 读取的窗口。

## Alternatives considered

- **在每个失败分支手写 cleanup（局部 lambda / goto）。** 分支多且未来会新增，
  容易漏；析构函数是单点，失败与正常释放共用同一套逻辑；否决。
- **让 `TinyLlmHandleImpl` 的裸指针改为 `DeviceBuffer` / `unique_ptr`。**
  更彻底，但要改 `ModelWeights`（公共结构）与 `cudaFree` 调用约定，超出本次
  修复范围；析构函数已消除泄漏。
- **在 `tinyllm_load` 的 `catch` 中手动调用 `tinyllm_free(h.release())`。**
  只能覆盖异常路径，显式 `return nullptr` 仍漏；且 `tinyllm_free` 需要完整
  句柄语义；否决。

## Consequences

Benefits:

- 任何加载失败路径都会释放已分配的权重与缓冲区；错误信息不变。
- `tinyllm_free` 与失败路径不再各维护一份释放列表，后续新增资源只需改析构。

Costs and limits:

- 本机无 nvcc/GPU，`src/ffi.cpp` 未编译、未用 Compute Sanitizer 验证；
  需在有 CUDA 的环境跑一次“故意让 `KVCacheManager::create` 失败”的泄漏对照。
- 析构中忽略 `cudaFree` 返回值（与旧实现一致）；CUDA 上下文已销毁时属预期。

## Verification

- 静态核对 `src/ffi.cpp` 全部 `cudaMalloc` / `cudaStreamCreate` 句柄：
  `paged_k_pool`、`paged_v_pool`、`paged_k_scratch`、`paged_v_scratch`、
  `rope_cos`、`rope_sin`、`hidden_buf`、`logits_buf`、`stream` 均在析构中释放；
  `d_*` 为 `DeviceBuffer` RAII，`workspace`/`kv_cache`/`layers` 为 RAII。
- `ModelLoader::freeWeights` 对空指针安全，权重加载失败（`h->weights` 仍为
  默认值）时析构不会二次释放或崩溃。
- `git diff --check`：通过。
- clang 18 CUDA host-only 语法检查（Triton CUDA 13 头 + `nv/target` 与
  logger shim，`-fgpu-defer-diag`）：`src/ffi.cpp` 通过，析构与错误路径可编译。
- 未运行：nvcc 设备代码编译、`tiny_llm_tests` 与 GPU 泄漏对照
  （本机无 CUDA 工具链 / GPU）。
