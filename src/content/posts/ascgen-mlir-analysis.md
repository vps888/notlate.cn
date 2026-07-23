---
title: "Ascgen融入MLIR的收益分析"
description: "Author: 牛玉虎 Date: 2026 01 28 MLIR 是一套提供AI编译器基础设施的框架，其提供了丰富、易用、易扩展的能力。本文主要挑选一些相对Ascgen来说不具备，但是用处比较大的能力来进行介绍。 注：这些能力基本上具备通用性，至于是否适用于Ascend NPU，还需要深入分…"
slug: "ascgen-mlir-analysis"
legacyId: 19540575
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19540575"
pubDate: 2026-01-27
updatedDate: 2026-02-06
category: "AI 编译器"
tags: ["AI 编译器","MLIR"]
featured: true
---

> Author: 牛玉虎
>
> Date: 2026-01-28

MLIR 是一套提供AI编译器基础设施的框架，其提供了丰富、易用、易扩展的能力。本文主要挑选一些相对Ascgen来说不具备，但是用处比较大的能力来进行介绍。

*注：这些能力基本上具备通用性，至于是否适用于Ascend NPU，还需要深入分析。*

## 1. TileAndFuse机制

**Tile-and-Fuse** 是 MLIR 中一种**组合型循环变换**：

* 先对算子进行 tiling（分块）
* 再在 tile 粒度上进行 producer–consumer fusion（生产者–消费者融合）

此机制理论上可以实现**任意实现了TilingInterface接口的Op**之间的融合，在目前Ascgen的架构中，缺乏类似机制。这类机制的使用场景如下：

#### 1.1 Tile-based编程

[如何基于MLIR实现Tile-based编程？](https://notlate.cn/blog/how-to-mlir-implement-tile-based)

#### 1.2. 生成Micro-Kernel的关键前置步骤

尤其是需要高数据局部性和复杂依赖的算子，比如Attention、Optimizer优化器等

[IREE的Flow方言如何高效计算QKV？](https://notlate.cn/blog/iree-flow-dialect-how-to-efficient-qkv)

[如何基于MLIR高效实现FlashAttention？](https://notlate.cn/blog/mlir-how-to-triton-efficient-implement-attention)

## 2. 基于Loop的分析变换能力

MLIR 提供了丰富的基于循环的分析和变换能力，现在介绍三种能力：

*  **Loop-carried Dependency Analysis（循环携带依赖分析）**：用于识别循环迭代间的依赖关系，为安全的循环重排、融合和并行化提供基础；
*  **Loop Unrolling（循环展开）**：通过展开小循环，减少分支开销，提高指令级并行性；
*  **Affine Loop Invariant Code Motion（循环不变代码外提）** ：将循环内不变的计算提取到循环外，减少冗余计算；

等等，这些能力在Ascgen中是难以表达实现（**缺乏对循环迭代空间和 SSA 依赖的精细控制能力**）。

详见：[如何充分发挥MLIR中Loop的优化特性？](https://notlate.cn/blog/how-to-mlir-loop)

## 3. 自动化Tensor Packing

MLIR中提供了自动化的 **Tensor Packing/Unpacking** 和 **Swizzled Layout(布局重排)** 等数据表达能力，这些能力有助于：

* **Cache Line Utilization：**将 tensor 元素按访问模式重新排列，使连续访问尽量落在同一 cache line 上，减少 cache miss；
* **Vectorization Friendliness**：将数据布局调整为向量化友好的连续块，便于 SIMD 指令的高效加载与存储
* **Bank Conflict Reduction：**对内存进行布局优化，减少 bank 冲突，提高并行访问效率；
* **Subtensor / Tile Access Optimization：**配合 tiling 或 micro-kernel，可以快速提取连续子块（subtensor）进行局部计算，减少中间 tensor 物化；

在现有的Ascgen的架构中，这些能力基本依赖隐含在ATT的tiling求解上，没有一套原生机制来实现和控制。

详见：[如何使用MLIR的linalg.pack实现性能大幅提升？](https://notlate.cn/blog/how-to-mlir-linalg-pack-implement)

## 4. 丰富的基础方言

MLIR提供了丰富的方言，分别负责不同场景的优化，自定义方言完全可以利用它们。比如：

* SCF：用于表达可组合的控制流结构，如 `if`、`for`、`while`，同时保证 SSA 可分析性，为循环变换和融合提供基础；
* Affine：支持静态可推导循环的各种变换，并提供多面体分析的能力，用于高级循环优化和自动并行化；

这些都是Ascgen目前不擅长的表达！不具体介绍。

## 5. 轻松构建Python API

MLIR 提供了丰富的 Python API 绑定，使用户能够在 Python 中：

* 构建和操作 MLIR Module、Op 和 Type；
* 定义自定义 Dialect 和算子，从而快速构建 DSL；
* 调用优化 Pass 并进行 IR 转换；
* Lower 高层算子到硬件相关的低级 IR。

通过 Python API，用户可以快速进行**实验和原型开发**，兼顾灵活性和性能。这对于开发领域特定语言（DSL）、快速验证优化策略，以及硬件感知算子生成尤其有用，**例如在 LLM 领域构建高效的 Attention 算子**。  

借助这一机制，**完全可以实现类似 Triton 或 TileLang 的 DSL 产品**，实现 tile-level kernel 设计和自动化优化。

## 6. 为什么会有如此大的差异？设计理念不同！

上述差异的本质来源于**编译器在 IR 设计上的根本理念**：

* IR：**为了让编译器能分析和优化程序**而设计的。

* 一个编译器中，通常会同时使用多种 IR，每种 IR 专注解决不同层次的问题。

现代编译器技术中，与 AI 编译器性能优化相关的两大 IR 是：

* Graph IR：擅长算子级优化
* SSA IR：擅长Loop级 / 数据局部性 / Micro-Kernel 级的优化

我们Ascgen中使用的ascir，是基于 Ascend IR 设计，实现上进行了属性扩展，本质上还是和 Ascend IR 一样，**属于Graph IR**。所表达的 Loop 等信息都是隐式的。

而 MLIR-based IR，尤其是 Linalg 及更低级硬件相关 IR，都是**属于 SSA IR**。可以精细管理循环、迭代空间、数据重用等，便于进行 **Tile-and-Fuse、Micro-kernel 生成** 等性能优化。

详见：[Graph IR vs SSA IR：为什么AI编译器离不开 SSA？](https://notlate.cn/blog/graph-ir-vs-ssa-ir-ai-ssa)

## 结论：如果要贴近硬件做极致性能优化，更推荐基于 SSA IR 进行设计。
