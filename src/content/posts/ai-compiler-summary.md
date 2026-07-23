---
title: "AI编译器融合技术系统化分类总结"
description: "请移步GitBook： AI编译器融合技术系统化分类总结 (https://notlate cn.github.io/aicompiler/) 目录及核心内容如下： 0. 全篇概述与阅读建议 0.1 全篇概述 本文旨在深入探讨 编译器融合技术 ，特别是如何在硬件架构 (如GPU 和 NPU) …"
slug: "ai-compiler-summary"
legacyId: 19530126
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19530126"
pubDate: 2026-01-25
updatedDate: 2026-02-07
category: "AI 编译器"
tags: ["AI 编译器"]
featured: true
---

请移步GitBook：[AI编译器融合技术系统化分类总结](https://notlate-cn.github.io/aicompiler/)

目录及核心内容如下：

## 0. 全篇概述与阅读建议

### 0.1 全篇概述

本文旨在深入探讨**编译器融合技术**，特别是如何在硬件架构 (如GPU 和 NPU) 上实现高效的计算。我们将从不同的层次和角度分析优化技术，包括算法层面的算子融合 (Fusion)、内存布局优化、硬件适配，以及全局优化策略。每一章都旨在帮助读者理解如何通过优化编译器的各个环节来最大化硬件资源的利用，从而提升计算性能。

本文内容的核心思想是：**优化不仅仅是局部算子的改进**，而是**通过跨层次的全局优化策略，解决不同优化手段之间的冲突，以实现最优的硬件适配和性能表现**。特别是随着 AI 处理器 (如GPU 和 Ascend NPU) 的发展，编译器需要更加智能地处理算子调度、内存管理、计算与存储的权衡等复杂问题。

### 0.2 Fusion的本质

算子**融合** (Fusion)是 AI 编译器优化技术中的一个核心概念。Fusion不仅仅是将多个算子合并为一个操作，它的本质在于通过**改变计算边界、重组执行时序和重映射数据生存期**，以最小化数据传输、最大化硬件利用率，从而提升整体计算效率。

具体来说，Fusion的本质包括：

* **改变计算边界 (Compute Boundary)**：通过将多个操作融合到一个计算图中，减少不必要的计算拆分，使得计算边界尽可能靠近硬件资源。
* **重组执行时序 (Execution Order)**：通过合理的调度顺序，避免冗余的内存访问和计算，提高计算资源的利用率。
* **重映射数据生存期 (Data Lifetime)**：通过延长某些数据的生命周期和优化数据的存储方式，减少不必要的数据传输和缓存读写。

Fusion的核心目标是减少数据移动开销，并最大化硬件的计算能力。这一思想贯穿本文的各个章节，特别是在算子层级和内存优化方面，它是实现高效编译器优化的关键技术之一。

### 0.3 章节结构

* 第 1 章 (图)：先看宏观的图结构，决定谁和谁能连在一起。
* 第 2 章 (环)：进入算子内部，看循环怎么写效率最高。
* 第 3 章 (数)：看数据怎么摆放，算得最顺手。
* 第 4 章 (存)：数据摆好了，怎么在不同速度的存储器 (HBM/L2/Reg) 之间搬运。
* 第 5 章 (并)：单核算好了，怎么利用多核、多芯片、集群搞并行。
* 第 6 章 (衡)：硬件资源有限 (寄存器/专用指令)，怎么做取舍 (Trade-off)。
* 第 7 章 (变)：形状或流程变了 (动态性)，怎么通过特化和符号化稳住性能。
* 第 8 章 (全)：以上全是局部招数，最后用全局视野 (Cost Model) 做最终决策。

---

## 1. 依赖拓扑 (Dependency Topology)
### 1.1 垂直融合 (Vertical Fusion)
### 1.2 水平融合 (Horizontal Fusion)
### 1.3 模式融合 (Pattern Fusion)

## 2. 循环与迭代空间优化 (Loop & Iteration Space Optimization)
### 2.1 循环融合 (Loop Fusion)
### 2.2 跨迭代状态优化 (Cross-iteration State Optimization)
### 2.3 循环展开与流水线 (Loop Unrolling & Pipelining)

## 3. 数据布局与表示 (Data Layout & Representation)
### 3.1 全局布局优化 (Global Layout Optimization)
### 3.2 数据打包与微布局 (Data Packing & Micro-layout)
### 3.3 填充与对齐 (Padding & Alignment)
### 3.4 缓冲区化与原地更新 (Bufferization & In-place)

## 4. 内存层次与多级分块 (Memory Hierarchy & Tiling)
### 4.1 多级分块 (Multi-level Tiling)
### 4.2 显式内存层级管理 (Explicit Hierarchy Management)
### 4.3 内存生命周期优化 (Lifetime Optimization)

## 5. 并行性与分布式融合 (Parallelism & Distributed Fusion)
### 5.1 指令与向量级并行 (Instruction/Vector Parallelism)
### 5.2 线程级并行融合 (Thread-level Parallelism)
### 5.3 分布式与张量并行 (Distributed & Tensor Parallelism)
### 5.4 任务级与异构并行 (Task & Heterogeneous Parallelism)

## 6. 硬件适配与计算-内存权衡 (Hardware Adaptation & Compute-Memory Trade-off)
### 6.1 资源感知内核融合 (Resource-Aware Kernel Fusion)
### 6.2 推测性融合与重计算 (Speculative Fusion & Rematerialization)
### 6.3 专用硬件指令映射 (Accelerator Intrinsic Mapping)
### 6.4 混合精度与量化融合 (Mixed Precision & Quantization Fusion)
### 6.5 指令级数据打包 (Instruction-Specific Packing)

## 7. 控制流与动态性 (Control-flow & Dynamism)
### 7.1 控制流扁平化与谓词融合 (Control-Flow Flattening & Predication)
### 7.2 动态形状融合 (Dynamic Shape Fusion)
### 7.3 稀疏性与不规则融合 (Sparsity & Irregular Fusion)
### 7.4 运行时特化 (Runtime Specialization)

## 8. 跨层次全局优化 (Cross-layer Global Optimization)
### 8.1 全局布局与Buffer传播 (Global Layout & Buffer Propagation)
### 8.2 代价模型驱动的融合决策 (Cost-Model Driven Fusion)
### 8.3 自动调优与调度分离 (Auto-tuning & Schedule Separation)

## 9. 特殊应用场景的融合策略映射
### 9.1 大语言模型优化 (LLM / Transformer)
### 9.2 混合专家模型 (Mixture of Experts, MoE)
### 9.3 稀疏计算 (Sparse Computing)
### 9.4 量化部署 (Quantization)
### 9.5 推荐系统 (DLRM)
### 9.6 状态空间模型 (Mamba / SSM)
### 9.7 边缘设备 (Edge / Mobile)
### 9.8 动态 Batch (Dynamic Batching)

## 附：主流编译器
### XLA (XLA: Accelerated Linear Algebra)
### TVM (Tensor Virtual Machine)
### TensorRT
### Triton
### MLIR (Multi-Level Intermediate Representation)
### TorchInductor
### CANN (Compute Architecture for Neural Networks)

## 参考资料
