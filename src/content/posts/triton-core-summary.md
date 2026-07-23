---
title: "【纯干货】Triton 发展历程核心总结"
description: "1. Triton 1.0：从实验室走向OpenAI工程化落地 资料： 1. 2019 Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (https://dl.acm.org…"
slug: "triton-core-summary"
legacyId: 19827341
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19827341"
pubDate: 2026-04-07
updatedDate: 2026-05-14
category: "AI 编译器"
tags: ["AI 编译器","Triton"]
featured: false
---

## 1. Triton 1.0：从实验室走向OpenAI工程化落地

资料：

1. [2019 Triton: An Intermediate Language and Compiler for
   Tiled Neural Network Computations](https://dl.acm.org/doi/epdf/10.1145/3315508.3329973)

2. [2021 Introducing Triton: Open-source GPU programming for neural networks](https://openai.com/index/triton/)

### 1.1 背景

论文指出，当时（2019年）的深度学习算子开发主要存在两种路径：

* 手写算子：性能极高，但是开发效率很低，而且移植性差。
* AI编译器：支持自动化生成算子，又分为两类技术：
  * 基于多面体技术的编译器：具有完备的数学理论，但数学约束非常复杂，适合处理仿射表达式，难以处理非仿射（no-affine）的稀疏计算。代表工作有：Tensor Comprehensions等。
  * 基于循环调度的编译器：实现简单，将计算与调度分离，但是生成算子依赖人工预定义模板和搜索空间，实际生成高性能算子的难度很大，而且搜索时间长。代表工作有：Halide、AutoTVM（2018第一代）等。

### 1.2 方案

Triton的切入点是折中上述两种路径：提供一种Tile-based的编程范式，开发者只需要处理好分块逻辑，编译器负责处理硬件底层最繁琐的逻辑。换句话说：开发者操作对象从**单个标量**转移到**分块张量**。

* 高性能：开发者负责将算子实现为以块（Tile）为单位的计算逻辑，这一部分是性能的关键，可以大大降低传统AI编译器在庞大搜索空间的寻优难度。
* 高效率：相比手写算子，屏蔽掉了硬件底层复杂逻辑，内存相关优化完全可以自动化，这部分由编译器负责，大大降低算子开发难度，提高开发效率。

> CUDA 对比 Triton
>
> |                          | CUDA   | TRITON    |
> | ------------------------ | ------ | --------- |
> | Memory Coalescing        | Manual | Automatic |
> | Shared Memory Management | Manual | Automatic |
> | Scheduling (Within SMs)  | Manual | Automatic |
> | Scheduling (Across SMs)  | Manual | Manual    |

### 1.3 编程模型

* Tile-first，基本操作单元是固定Shape的矩阵块
* 通过掩码处理边界对齐问题
* 隐式并发，开发者编写单线程针对单一块的Kernel，编译器负责将其分发到SM上并行执行。

### 1.4 Triton架构(1.0)

![triton-overview](https://img2024.cnblogs.com/blog/3599704/202604/3599704-20260407002310313-1526099716.png)


1. Triton-C

   一种类似于C语言的DSL，作为Triton的前端，主要目的是向上提供一种稳定的接口。

2. Triton-IR

   一种基于LLVM的IR，是Triton的核心表示组件，提供了基于分块语义的中间表示。由Triton-C通过解析得到。

3. Triton-JIT：

   将Triton-IR编译优化得到高效的机器码（PTX），这是Triton的核心优化组件，主要负责处理复杂的硬件映射。这个组件主要有三部分组成：

   **（1）硬件无关优化**

   * 自动预取（Auto Pre-fetching）：自动检测并前置下一次迭代所需的Load指令
   * 代数化简：比如两次转置等于自身。

   **（2）硬件相关优化**

   这部分主要是针对GPU硬件模型进行优化。

   * **层级化分块（Hierarchical Tiling）**：将大块（Tile）进一步分解为Micro-Tile和Nano-Tile，用来匹配GPU的硬件模型（从SM->SIMD->寄存器）。
   * **内存合并（Memory Coalescing）**：GPU 访问显存时，如果相邻线程访问连续地址，效率最高。Triton 后端会自动重新排列Micro-Tile内的线程执行顺序，确保即使在逻辑上是不连续的访问，在底层也能尽可能实现合并访问。
   * **共享内存分配（Shared Memory Allocation）**：分析变量的**生命周期（Live Range）**，并使用线性时间算法自动规划共享内存的布局。

   * **共享内存同步（Shared Memory Synchronization）**：自动检测数据流的读写依赖关系，并自动插同步原语。

   （3）自动调优（Auto-tuner）

   （2018年的时候）传统的自动调优需要依赖手写模板，Triton则是直接从IR中提取优化参数。为什么？因为人已经通过Triton提供的DSL写好了计算逻辑，也就是模板。编译器只需要寻优Tile参数即可。

   Tile参数的搜索，Triton采用了枚举法，因为它限制了搜索范围以提高效率，约束所有Tile参数必须是2的幂。不同层级的Tile块范围不同：

   * Tile：32~128
   * Micro-Tile：8~32
   * Nano-Tile：1~4

   所有，综合起来每个维度的参数不超过3个，所以搜索速度会很快。

### 1.5 总结

​	Triton的本质是在LLVM-IR的基础上增加了数据流和控制流的扩展。通过这种扩展，不仅吸收了传统编译器自动化的能力（屏蔽硬件内存模型和指令映射），而且保证了算子性能。



## 2. Triton 2.0：全面重构，融入MLIR
未完待续...
