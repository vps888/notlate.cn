---
title: "Graph IR vs SSA IR：为什么现代 AI 编译器离不开 SSA？"
description: "在编译器和 AI 编译领域，经常会听到 Graph IR 和 SSA IR 。 它们并不是“新旧关系”，而是 解决不同问题的两类中间表示（IR） 。 本文用 直观的方式 解释它们的区别，以及各自适合做什么。 1. 什么是 Graph IR？ 直观理解 Graph IR 像一张“算子连接图” 节…"
slug: "graph-ir-vs-ssa-ir-ai-ssa"
legacyId: 19545289
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19545289"
pubDate: 2026-01-28
category: "AI 编译器"
tags: ["AI 编译器"]
featured: true
---

在编译器和 AI 编译领域，经常会听到 **Graph IR** 和 **SSA IR**。
它们并不是“新旧关系”，而是**解决不同问题的两类中间表示（IR）**。

本文用**直观的方式**解释它们的区别，以及各自适合做什么。

------

## 1. 什么是 Graph IR？

### 直观理解

> **Graph IR 像一张“算子连接图”**

- 节点（Node）是算子（MatMul、Add、Softmax）
- 边（Edge）是 Tensor
- 整体是一个 **有向无环图（DAG）**

### 一个简单示例

```text
Q ──┐
    ├─ MatMul ── Softmax ── MatMul ── Out
K ──┘                      ▲
                           │
                           V
```

这在 ONNX、TensorFlow Graph、Pytorch 中看到的形式。

------

### Graph IR 擅长什么？

- 算子级调度
- 算子融合（Op Fusion）
- 跨设备/跨算子执行规划
- 粗粒度内存规划

**一句话：擅长“调度算子”**

------

### Graph IR 的局限

Graph IR **不知道**：

- 算子内部是否有循环
- 哪一维是 reduction
- 是否可以做 tiling
- 中间结果能否复用
- 是否存在循环携带依赖

> 在 Graph IR 中，算子内部是 **黑盒**

------

## 2. 什么是 SSA IR？

### 直观理解

> **SSA IR 是“一段可分析、可重写的程序”**

SSA（Static Single Assignment）要求：

- 每个值只定义一次
- 所有依赖关系显式可追踪

### 一个简单示例

```llvm
%a = load %A
%b = load %B
%c = arith.mulf %a, %b
%d = arith.addf %c, %bias
```

这里的每个 `%x` 都是一个**有生命周期的变量**。

------

### SSA IR 擅长什么？

- 循环分析与变换
- Tiling / Fusion / Unrolling
- Reduction 重写
- 数据复用分析
- 生成 micro-kernel

**一句话：擅长“重写程序”**

------

## 3. 一个关键区别：算子 vs 程序

| 对比维度       | Graph IR | SSA IR      |
| -------------- | -------- | ----------- |
| 表达单元       | 算子     | 语句 / 循环 |
| 控制流         | ❌        | ✅           |
| Loop           | ❌        | ✅           |
| Reduction 语义 | ❌        | ✅           |
| 数据依赖       | 隐式     | 显式        |
| 程序重写       | ❌        | ✅           |

**本质区别在于：**

> Graph IR 表达的是 **“做什么算子”**
> SSA IR 表达的是 **“怎么算”**

------

## 4. 用 Attention 举一个直观对比

### 在 Graph IR 中

```text
MatMul(Q, K)
 → Scale
 → Softmax
 → MatMul(·, V)
```

编译器只能：

- **调整算子顺序**
- **尝试算子融合**

------

### 在 SSA IR 中

Attention 会被展开成：

- 显式的多重循环
- 明确的 reduction 维度
- 可分析的中间变量

因此可以：

- Tile Q/K/V
- 将 Softmax 改写为 Online Softmax
- 融合成一个 micro-kernel（如 FlashAttention）

> **FlashAttention 在 Graph IR 中几乎不可表达，在 SSA IR 中是一次程序重写。**

------

## 5. 为什么现代 AI 编译器需要两者？

现实中的编译器并不是“二选一”，而是：

```text
Graph IR  →  SSA IR  →  低级 IR
```

### 各自分工明确

- **Graph IR**
  - 模型结构表达
  - 算子级优化
  - 全局调度
- **SSA IR**
  - 循环与数据局部性优化
  - Tiling / Fusion
  - Micro-kernel 生成

------

## 6. 一句话总结

> **Graph IR 是“算子调度的语言”，
> SSA IR 是“程序优化的语言”。**

如果只想“连算子”，Graph IR 足够；
如果想“榨干硬件性能”，SSA IR 不可或缺。
