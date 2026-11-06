# 入门指南

欢迎使用 Tiny-LLM！本指南将帮助您快速入门。

## 什么是 Tiny-LLM？

Tiny-LLM 是一个极简的教育性 LLM 推理引擎，使用 CUDA C++ 实现。它提供：

- **W8A16 量化**：INT8 权重量化，16 位激活值
- **CUDA 内核**：优化的注意力机制、前馈网络等内核
- **KV 缓存**：使用预分配缓存实现高效的自回归生成
- **无异常设计**：基于 `Result\<T\>` 的错误传播

## 系统要求

开始之前，请确保您拥有：

- **CUDA 工具包** 11.0 或更高版本
- **CMake** 3.18 或更高版本
- **C++17** 兼容编译器（GCC 9+、Clang 10+、MSVC 2019+）
- **NVIDIA GPU**，计算能力 7.0+（Volta 或更新）

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/holtwood-systems/tiny-llm.git
cd tiny-llm

# 配置并构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行测试
ctest --test-dir build --output-on-failure
```

## 下一步

- [安装指南](/guide/installation) - 详细的安装说明
- [快速入门教程](/guide/quickstart) - 构建您的第一个推理
- [架构概览](/architecture/inference-engine) - 了解内部原理
