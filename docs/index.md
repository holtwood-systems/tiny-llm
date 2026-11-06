---
layout: home

hero:
  name: "Tiny-LLM"
  text: "CUDA 原生推理引擎"
  tagline: 面向聚焦型 Transformer 推理，提供 W8A16 kernel、显式 KV Cache 管理，以及更低维护成本的仓库表面。
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: 开始使用
      link: /guide/getting-started
    - theme: alt
      text: API 参考
      link: /api/

features:
  - icon: "⚡"
    title: W8A16 推理
    details: INT8 权重 + FP16 激活，保持精简的 CUDA 运行时路径。
  - icon: "📦"
    title: 二进制运行时路径
    details: "`InferenceEngine::load()` 面向当前引擎支持的二进制运行时格式。"
  - icon: "🔎"
    title: GGUF 解析工具
    details: 保留 GGUF 解析、元数据提取和 tensor 检查能力，但不再假装它就是运行时加载路径。
  - icon: "🧠"
    title: 显式 KV Cache
    details: 预分配序列槽位，让自回归解码更可预测。
  - icon: "🛡️"
    title: "Result<T> API"
    details: 宿主侧失败显式可见，便于排查和收敛。
  - icon: "🧪"
    title: 核心测试覆盖
    details: GoogleTest 与 RapidCheck 覆盖核心加载、缓存和生成路径。
---

## 运行时边界

| 能力 | 状态 | 说明 |
|------|------|------|
| 二进制运行时加载 | 支持 | `InferenceEngine::load()` 使用这条路径 |
| GGUF 解析与检查 | 支持 | 使用 `GGUFParser` 读取元数据和 tensor |
| 直接 GGUF 运行时加载 | 不支持 | 运行时加载器会拒绝 `.gguf` 路径 |

## 架构示意

```mermaid
flowchart LR
    A[二进制运行时模型] --> B[InferenceEngine::load()]
    C[GGUF 文件] --> D[GGUFParser]
    B --> E[Transformer 层]
    E --> F[KV Cache]
    E --> G[W8A16 矩阵乘法]
    G --> H[采样]
    H --> I[输出 Tokens]
    D --> J[元数据 / Tensor 检查]

    style B fill:#00D4AA,stroke:#00C49A,color:#fff
    style G fill:#76B900,stroke:#5a8f00,color:#fff
    style D fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

## 从这里开始

| 资源 | 用途 |
|------|------|
| [开始使用](/guide/getting-started) | 最小构建与首次运行路径 |
| [架构说明](/architecture/) | 运行时结构与职责划分 |
| [API 参考](/api/) | 公共头文件与类型 |
| [性能](/performance/) | 基准和优化说明 |
| [GitHub 仓库](https://github.com/holtwood-systems/tiny-llm) | 源码、Issue、Release |
