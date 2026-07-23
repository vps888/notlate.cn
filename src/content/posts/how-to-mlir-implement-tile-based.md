---
title: "如何基于MLIR实现Tile-based编程？"
description: "本文档介绍 MLIR 中的 Tiling 技术，这是优化计算性能的核心方法。即使你对 MLIR 不熟悉，也可以通过本文档学习 Tiling 的概念和用法。 目录 1. 前置知识：MLIR 基础概念 ( 1 前置知识mlir 基础概念) 2. 什么是 Tiling？ ( 2 什么是 tiling…"
slug: "how-to-mlir-implement-tile-based"
legacyId: 19537345
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19537345"
pubDate: 2026-01-27
category: "AI 编译器"
tags: ["AI 编译器","MLIR"]
featured: true
---

> 本文档介绍 MLIR 中的 Tiling 技术，这是优化计算性能的核心方法。即使你对 MLIR 不熟悉，也可以通过本文档学习 Tiling 的概念和用法。

---

## 目录

1. [前置知识：MLIR 基础概念](#1-前置知识mlir-基础概念)
2. [什么是 Tiling？](#2-什么是-tiling)
3. [示例 1：最简单的循环 Tiling](#3-示例-1最简单的循环-tiling)
4. [MLIR 中的 Tiling 机制](#4-mlir-中的-tiling-机制)
5. [示例 2：矩阵乘法的 Tiling](#5-示例-2矩阵乘法的-tiling)
6. [进阶：Tile and Fuse 模式](#6-进阶tile-and-fuse-模式)
7. [高级特性介绍](#7-高级特性介绍)
8. [术语表](#8-术语表)
9. [参考资源](#9-参考资源)

---

## 1. 前置知识：MLIR 基础概念

在深入 Tiling 之前，先了解一些 MLIR 的基本概念。

### 1.1 MLIR 是什么？

**MLIR (Multi-Level Intermediate Representation)** 是一个编译器基础设施，用于构建可重用、可扩展的编译器。

简单理解：MLIR 是一种**中间表示语言**，介于高级语言（如 TensorFlow、C++）和底层代码（如 LLVM IR、机器码）之间。

```
高级语言
   ↓
  MLIR  ←── 我们在这里工作
   ↓
底层代码
```

### 1.2 MLIR 的核心概念

#### Dialect（方言）

MLIR 由多个 **Dialect** 组成，每个 Dialect 专注于特定领域：

| Dialect     | 用途         | 示例操作                                      |
| ----------- | ------------ | --------------------------------------------- |
| `arith`     | 算术运算     | `arith.add`, `arith.constant`                 |
| `scf`       | 结构化控制流 | `scf.for`, `scf.if`                           |
| `linalg`    | 线性代数运算 | `linalg.matmul`, `linalg.generic`             |
| `affine`    | Affine 循环  | `affine.for`                                  |
| `tensor`    | 张量操作     | `tensor.extract_slice`, `tensor.insert_slice` |
| `transform` | 变换脚本     | `transform.structured.tile`                   |

#### Tensor 类型

MLIR 使用 `tensor` 类型表示多维数组：

```cpp
// 语法：tensor<维度1x维度2x...x数据类型>
%A = tensor<512x512xf32>      // 512x512 的浮点数矩阵
%B = tensor<1024xi32>          // 长度为 1024 的整数向量
%C = tensor<3x4x5xf64>         // 3x4x5 的双精度浮点数张量
```

#### SSA 形式

MLIR 使用 SSA（静态单赋值）形式，每个值只定义一次：

```cpp
// 定义一个常量
%c0 = arith.constant 0 : index

// 使用它
%result = arith.addi %a, %b : index
```

---

### 1.3 MLIR 代码结构示例

这是一个简单的 MLIR 函数，计算矩阵加法：

```cpp
module {
  func.func @matrix_add(%A: tensor<512x512xf32>,
                        %B: tensor<512x512xf32>)
                        -> tensor<512x512xf32> {
    // 定义一个零张量作为输出
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<512x512xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<512x512xf32>)
      -> tensor<512x512xf32>

    // 执行矩阵加法
    %result = linalg.generic
      {indexing_maps = [
        affine_map<(i, j) -> (i, j)>,  // A 的索引映射
        affine_map<(i, j) -> (i, j)>,  // B 的索引映射
        affine_map<(i, j) -> (i, j)>   // 输出的索引映射
      ],
      iterator_types = ["parallel", "parallel"]}
      ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%fill : tensor<512x512xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<512x512xf32>

    return %result : tensor<512x512xf32>
  }
}
```

---

## 2. 什么是 Tiling？

### 2.1 直观理解

**Tiling（分块）** 是将大计算任务分解为小块的技术。

想象你要粉刷一面 10m × 10m 的墙：

- **未 Tiling**：一次粉刷整面墙 → 刷子需要蘸很多颜料，容易干掉
- **Tiling**：把墙分成 1m × 1m 的小块，逐块粉刷 → 每次只需少量颜料，效率更高

### 2.2 为什么需要 Tiling？

#### 问题：缓存局部性

现代 CPU 有多级缓存（L1/L2/L3），速度差异巨大：

```
┌─────────────────────────────────────────┐
│  寄存器: ~1ns     | 最快，容量最小      │
│  L1 缓存: ~3ns    | ~32KB              │
│  L2 缓存: ~10ns   | ~256KB             │
│  L3 缓存: ~50ns   | ~8MB               │
│  主内存: ~100ns   | 最慢，容量最大      │
└─────────────────────────────────────────┘
```

如果计算数据无法放入 L1 缓存，CPU 需要等待从 L2/L3/主存加载数据，性能急剧下降。

#### Tiling 的效果

通过 Tiling，让每个小计算块的数据放入 L1 缓存：

```
未 Tiling:
  计算 C = A × B（512×512 矩阵）
  需要反复在 L3 缓存和主存间传输数据

Tiling 后（tile_size = 8）:
  将 512×512 分成 64×64 个 8×8 的小块
  每个小块计算时，数据完全放入 L1 缓存
  访问速度提升 10-100 倍
```

### 2.3 Tiling 的基本模式

以三重循环为例：

```cpp
// 原始代码
for (int i = 0; i < 1024; i++)
  for (int j = 0; j < 1024; j++)
    for (int k = 0; k < 1024; k++)
      C[i][j] += A[i][k] * B[k][j];
```

**Tiling 后**（tile_size = 32）：

```cpp
// 外层：遍历每个 tile
for (int ti = 0; ti < 1024; ti += 32)
  for (int tj = 0; tj < 1024; tj += 32)
    for (int tk = 0; tk < 1024; tk += 32)
      // 内层：计算单个 tile
      for (int i = ti; i < ti + 32; i++)
        for (int j = tj; j < tj + 32; j++)
          for (int k = tk; k < tk + 32; k++)
            C[i][j] += A[i][k] * B[k][j];
```

---

## 3. 示例 1：最简单的循环 Tiling

让我们从一个最简单的例子开始，不涉及 MLIR 的复杂特性。

### 3.1 原始代码

```cpp
func.func @loop_tiling() {
  // 三重嵌套循环
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}
```

**说明**：

- `affine.for` 是 MLIR 的一种循环结构
- `%i`, `%j`, `%k` 是循环变量（SSA 值）
- `"test.foo"` 是一个占位操作，代表任意计算

### 3.2 应用 Tiling

使用 MLIR 的 `affine-loop-tile` pass：

```bash
mlir-opt input.mlir --affine-loop-tile="tile-size=32" -o output.mlir
```

### 3.3 Tiling 后的代码

```cpp
func.func @loop_tiling() {
  // 外层：tile 循环（步长 32）
  affine.for %ti = 0 to 256 step 32 {
    affine.for %tj = 0 to 512 step 32 {
      affine.for %tk = 0 to 1024 step 32 {

        // 内层：点循环（处理单个 tile）
        affine.for %i = %ti to min(%ti + 32, 256) {
          affine.for %j = %tj to min(%tj + 32, 512) {
            affine.for %k = %tk to min(%tk + 32, 1024) {
              "test.foo"(%i, %j, %k) : (index, index, index) -> ()
            }
          }
        }
      }
    }
  }
  return
}
```

### 3.4 变化对比

| 方面     | 原始代码         | Tiling 后                |
| -------- | ---------------- | ------------------------ |
| 循环层数 | 3 层             | 6 层（3层 tile + 3层点） |
| 外层循环 | 无               | 遍历 tile（步长 32）     |
| 内层循环 | 遍历全部         | 遍历单个 tile            |
| 边界处理 | 简单（固定边界） | 需要 `min()` 处理不对齐  |

### 3.5 可视化理解

```
原始循环（256 × 512）：
┌────────────────────────────────────┐
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  │
│ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊  │
└────────────────────────────────────┘

Tiling 后（tile_size = 32）：
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ tile│ tile│ tile│ tile│ tile│ tile│  ← 外层循环遍历
├─────┼─────┼─────┼─────┼─────┼─────┤
│ tile│ tile│ tile│ tile│ tile│ tile│
├─────┼─────┼─────┼─────┼─────┼─────┤
│ ┊   │ ┊   │ ┊   │ ┊   │ ┊   │ ┊   │
└─────┴─────┴─────┴─────┴─────┴─────┘
  ↓
每个 tile 内部有内层循环
```

---

## 4. MLIR 中的 Tiling 机制

现在介绍 MLIR 如何统一处理不同 Dialect 的 Tiling。

### 4.1 问题：不同 Dialect 的 Tiling 逻辑相似

无论是 `affine.for`、`linalg.matmul` 还是自定义操作，Tiling 的核心逻辑是一样的：

1. 确定循环维度
2. 分成外层（tile）和内层（点）
3. 提取子区域数据

但每种操作有自己的实现方式，导致代码重复。

### 4.2 解决方案：TilingInterface

MLIR 通过 **接口** 机制统一 Tiling：

```cpp
// 伪代码：TilingInterface 的定义
interface TilingInterface {
  // 方法 1：返回循环迭代器类型
  // 例如：[parallel, parallel, reduction]
  getLoopIteratorTypes();

  // 方法 2：返回迭代空间（每个维度的范围）
  // 例如：[{0, 512}, {0, 512}, {0, 512}]
  getIterationDomain();

  // 方法 3：生成 Tiling 后的实现
  getTiledImplementation(tileSizes, loopRanges);
}
```

### 4.3 各 Dialect 实现 TilingInterface

| Dialect  | 操作                       | 实现            |
| -------- | -------------------------- | --------------- |
| `linalg` | `matmul`, `generic` 等     | ✅ 已实现        |
| `tensor` | `extract_slice`, `pack` 等 | ✅ 已实现        |
| `scf`    | `for` 循环                 | ✅ 已实现        |
| `affine` | `affine.for`               | 通过专门的 pass |

这意味着你可以用统一的 API 对所有这些操作进行 Tiling！

### 4.4 Transform Dialect

MLIR 提供了 **Transform Dialect**，用于编写变换脚本：

```
┌─────────────────────────────────────────┐
│  Transform IR (变换脚本)                  │
│  - 描述"如何变换"                         │
│  - 操作类型: transform.*                 │
└──────────────┬──────────────────────────┘
               │ applies to
               ▼
┌─────────────────────────────────────────┐
│  Payload IR (待优化的计算代码)            │
│  - 描述"计算什么"                         │
│  - 操作类型: linalg.*, scf.*, arith.*    │
└─────────────────────────────────────────┘
```

**关键概念**：

- **Payload IR**：你想要优化的计算代码（如矩阵乘法）
- **Transform IR**：描述如何优化的脚本（如"tile 尺寸为 8"）

---

## 5. 示例 2：矩阵乘法的 Tiling

现在看一个实际例子：使用 Transform Dialect 对矩阵乘法进行 Tiling。

### 5.1 Payload IR：原始矩阵乘法

```cpp
func.func @matmul(%A: tensor<25x34xf32>,
                  %B: tensor<34x25xf32>,
                  %C: tensor<25x25xf32>)
                  -> tensor<25x25xf32> {
  %result = linalg.matmul
    ins(%A, %B : tensor<25x34xf32>, tensor<34x25xf32>)
    outs(%C : tensor<25x25xf32>)
    -> tensor<25x25xf32>
  return %result : tensor<25x25xf32>
}
```

**说明**：

- `linalg.matmul` 是 MLIR 的矩阵乘法操作
- `ins(...)`：输入操作数（A 和 B）
- `outs(...)`：输出操作数（C，作为累加器）
- `tensor<25x34xf32>`：25×34 的浮点数张量

### 5.2 Transform IR：Tiling 脚本

```cpp
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {

    // Step 1: 匹配 linalg.matmul 操作
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // Step 2: 执行 tiling
    %tiled_op, %loop = transform.structured.tile_using_for %matmul
      tile_sizes [9]          // 第一维的 tile 大小为 9
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

**逐步解释**：

1. **`transform.named_sequence`**：定义一个可复用的变换序列
2. **`%root: !transform.any_op`**：输入参数，指向任意 payload 操作
3. **`transform.structured.match`**：找到 payload IR 中的 `linalg.matmul` 操作
4. **`transform.structured.tile_using_for`**：执行 tiling，使用 `scf.for` 循环
5. **`tile_sizes [9]`**：第一维（25）分成 9 的小块

### 5.3 完整模块（Payload + Transform）

```cpp
module {
  // ═════════════════════════════════════════════════════════
  // Payload IR: 计算代码
  // ═════════════════════════════════════════════════════════
  func.func @matmul(%A: tensor<25x34xf32>,
                    %B: tensor<34x25xf32>,
                    %C: tensor<25x25xf32>)
                    -> tensor<25x25xf32> {
    %result = linalg.matmul
      ins(%A, %B : tensor<25x34xf32>, tensor<34x25xf32>)
      outs(%C : tensor<25x25xf32>)
      -> tensor<25x25xf32>
    return %result : tensor<25x25xf32>
  }

  // ═════════════════════════════════════════════════════════
  // Transform IR: 变换脚本
  // ═════════════════════════════════════════════════════════
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%root: !transform.any_op) {
      %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      %tiled_op, %loop = transform.structured.tile_using_for %matmul
        tile_sizes [9]
      transform.yield
    }
  }
}
```

### 5.4 运行变换

```bash
mlir-opt input.mlir --transform-interpreter -o output.mlir
```

**`--transform-interpreter`**：执行 Transform IR，修改 Payload IR

### 5.5 变换后的代码

```cpp
func.func @matmul(%A: tensor<25x34xf32>,
                  %B: tensor<34x25xf32>,
                  %C: tensor<25x25xf32>)
                  -> tensor<25x25xf32> {
  %c9 = arith.constant 9 : index
  %c0 = arith.constant 0 : index

  // 外层：tile 循环
  %result = scf.for %idx = %c0 to 25 step %c9
    iter_args(%out = %C) -> (tensor<25x25xf32>) {

    // 提取 A 的切片 [idx:idx+9, 0:34]
    %A_tile = tensor.extract_slice %A[%idx, 0] [9, 34] [1, 1]
      : tensor<25x34xf32> to tensor<9x34xf32>

    // 提取 C 的切片 [idx:idx+9, 0:25]
    %C_tile = tensor.extract_slice %out[%idx, 0] [9, 25] [1, 1]
      : tensor<25x25xf32> to tensor<9x25xf32>

    // 在 tile 上执行 matmul（9×34 × 34×25 = 9×25）
    %tile_result = linalg.matmul
      ins(%A_tile, %B : tensor<9x34xf32>, tensor<34x25xf32>)
      outs(%C_tile : tensor<9x25xf32>)
      -> tensor<9x25xf32>

    // 将结果插入回输出张量
    %inserted = tensor.insert_slice %tile_result into %out[%idx, 0] [9, 25] [1, 1]
      : tensor<9x25xf32> into tensor<25x25xf32>

    scf.yield %inserted : tensor<25x25xf32>
  }

  return %result : tensor<25x25xf32>
}
```

### 5.6 关键变化

| 方面     | 变换前               | 变换后                    |
| -------- | -------------------- | ------------------------- |
| 循环结构 | 无循环               | 外层 `scf.for` 遍历 tiles |
| 数据访问 | 完整的 25×34×25 矩阵 | 每个 tile 是 9×34×25      |
| 内存效率 | 可能超出 L1 缓存     | 每个 tile 适合 L1 缓存    |
| 并行性   | 单线程               | `scf.for` 可并行化        |

---

## 6. 进阶：Tile and Fuse 模式

当有多个连续(任意实现了TilingInterface的)操作时，**Tile and Fuse** 可以进一步提升性能。

### 6.1 问题场景

考虑一个神经网络层：矩阵乘法 + 加偏置 + ReLU

```cpp
func.func @fc_relu(%lhs: tensor<512x512xf32>,
                   %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>,
                   %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // 操作 1: 矩阵乘法
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  // 操作 2: 加偏置 (逐元素加法)
  %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  // 操作 3: ReLU (逐元素 max(x, 0))
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>)
    -> tensor<512x512xf32>

  return %relued : tensor<512x512xf32>
}
```

**问题**：每个操作的中间结果都写入内存

```
内存访问（未优化）:
matmul   → 写入 512×512 = 262K 元素
add      → 写入 262K 元素
max      → 写入 262K 元素
总计: 786K 元素写入
```

### 6.2 Tile and Fuse 的思路

```
原始数据流:
lhs, rhs → matmul → C (写内存) → add → D (写内存) → max → E (写内存)

Tile and Fuse:
对于每个 tile:
  tile(lhs), tile(rhs) → tile(matmul) → 直接给 tile(add) → 直接给 tile(max) → 写回
                          ↑_______________________↑
                          全部在 L1/寄存器中完成
```

### 6.3 Transform 脚本

```cpp
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %root: !transform.any_op,                    // ← 绑定到整个 module
       %matmul: !transform.op<"linalg.matmul">,     // ← 绑定到 @fc_relu 中的 linalg.matmul
       %elementwise: !transform.op<"linalg.elementwise">) {  // ← 绑定到 @fc_relu 中的 2 个 elementwise

    // ═════════════════════════════════════════════════════════
    // Step 0: ↑↑↑↑↑↑↑↑↑↑ 理解入参 ↑↑↑↑↑↑↑↑↑↑
    // ═════════════════════════════════════════════════════════
    // %elementwise_handle 关联了模块中 ALL 的 linalg.elementwise 操作！
    // 在我们的例子中，它关联了两个操作：
    //   1. %biased = linalg.elementwise kind=<add>    (加偏置)
    //   2. %relued = linalg.elementwise kind=<max>    (ReLU)
    //
    // 因为 transform dialect 的 match 操作会匹配所有符合条件的操作，
    // 所以 %elementwise_handle 是一个包含 2 个操作的列表。
    //
    // 数据流关系：
    //   %matmul → %biased (add) → %relued (max)
    //   我们需要从最后的 max 开始 tile，然后向上融合 add 和 matmul
    // ═════════════════════════════════════════════════════════


    %add, %max = transform.split_handle %elementwise
      : (!transform.op<"linalg.elementwise">)
      -> (!transform.any_op, !transform.any_op)
    // ═════════════════════════════════════════════════════════
    // Step 1: ↑↑↑↑↑↑↑↑↑↑ 分离 elementwise handle ↑↑↑↑↑↑↑↑↑↑
    // ═════════════════════════════════════════════════════════
    // 将 [add, max] 分成两个独立的 handles
    // 现在：
    //   %add_handle  → 指向 linalg.elementwise kind=<add>
    //   %max_handle  → 指向 linalg.elementwise kind=<max>
    // ═════════════════════════════════════════════════════════

    %tiled_max, %loop = transform.structured.tile_using_forall %max
      tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // ═════════════════════════════════════════════════════════
    // Step 2: ↑↑↑↑↑↑↑↑↑↑ Tile 最后一个操作 (max/ReLU) ↑↑↑↑↑↑↑↑↑↑
    // ═════════════════════════════════════════════════════════
    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %max_tile = linalg.elementwise kind=max on [ti*8:ti*8+8, tj*32:tj*32+32]
    // }
    // ═════════════════════════════════════════════════════════

    %add_fused, %loop = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op)
    // ═════════════════════════════════════════════════════════
    // Step 3: ↑↑↑↑↑↑↑↑↑↑ Fuse add 操作到循环中 ↑↑↑↑↑↑↑↑↑↑
    // ═════════════════════════════════════════════════════════
    // Fusion 做什么：
    // 1. 找到 add 操作的输入（来自 matmul 的输出）
    // 2. 将 add 操作移动到循环内部
    // 3. 在循环内对输入做 extract_slice
    // 4. add 的输出直接作为 max 的输入，不写回内存

    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %biased_tile = linalg.elementwise kind=add on tile
    //   %max_tile = linalg.elementwise kind=max on tile (使用 %biased_tile)
    // }
    // ═════════════════════════════════════════════════════════
    
    %matmul_fused, %loop = transform.structured.fuse_into_containing_op %matmul into %loop
      : (!transform.op<"linalg.matmul">, !transform.any_op)
    // ═════════════════════════════════════════════════════════
    // Step 4: ↑↑↑↑↑↑↑↑↑↑ Fuse matmul 操作到循环中 ↑↑↑↑↑↑↑↑↑↑
    // ═════════════════════════════════════════════════════════
    // 原始 matmul 操作 (在循环外):
    //   %matmul = linalg.matmul
    //     ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
    //     outs(%output : tensor<512x512xf32>)
    //
    // 目标循环 (Step 3 之后):
    //   scf.forall (%ti, %tj) in (64, 16) {
    //     %add_tile = linalg.elementwise kind=add on tile
    //     %max_tile = linalg.elementwise kind=max on tile
    //   }
    //
    // fuse_into_containing_op 的执行流程：
    // ┌─────────────────────────────────────────────────────────┐
    // │ 1. 分析数据依赖                                          │
    // │    - 找到 matmul 的 consumer: %add 操作                 │
    // │    - 检测 %add 在循环内的 tiled 版本: %add_tile         │
    // │                                                          │
    // │ 2. 调用 TilingInterface (linalg.matmul 实现)            │
    // │                                                          │
    // │    getIterationDomain() 返回:                           │
    // │      - dim 0: [0, 512)  并行迭代                        │
    // │      - dim 1: [0, 512)  并行迭代                        │
    // │      - dim 2: [0, 512)  归约迭代 (K 维度)               │
    // │                                                          │
    // │    getLoopIteratorTypes() 返回:                         │
    // │      - [parallel, parallel, reduction]                  │
    // │                                                          │
    // │ 3. 匹配循环迭代空间                                      │
    // │    - 循环外层: scf.forall (%ti, %tj)                    │
    // │      %ti 对应 matmul 的 dim 0 (M 维度)                  │
    // │      %tj 对应 matmul 的 dim 1 (N 维度)                  │
    // │    - matmul 的 dim 2 (K 维度) 是归约，不需要外层循环    │
    // │                                                          │
    // │ 4. 计算需要提取的 tile 区域                              │
    // │    - 输入 %lhs: 从 [ti*8, 0] 提取 [8, 512]              │
    // │      (M 维度 tiling, K 维度完整)                         │
    // │    - 输入 %rhs: 从 [0, tj*32] 提取 [512, 32]            │
    // │      (K 维度完整, N 维度 tiling)                         │
    // │    - 输出: 从 add_tile 的输出位置继承                   │
    // │                                                          │
    // │ 5. 在循环内生成 tiled matmul                             │
    // │    %lhs_tile = tensor.extract_slice %lhs[ti*8, 0][8,512]│
    // │    %rhs_tile = tensor.extract_slice %rhs[0, tj*32][512,32]│
    // │    %matmul_tile = linalg.matmul                         │
    // │      ins(%lhs_tile, %rhs_tile)                          │
    // │      outs(%output_tile)                                 │
    // │                                                          │
    // │ 6. 更新数据流                                           │
    // │    原来使用 %add 的输入 %biased (来自循环外的 matmul)    │
    // │    现在使用 %matmul_tile (循环内的 tiled 版本)           │
    // └─────────────────────────────────────────────────────────┘
    //
    // 关键点：为什么 matmul 的输入提取方式不同？
    // ┌─────────────────────────────────────────────────────────┐
    // │ %lhs: tensor<512x512xf32>  (M=512, K=512)               │
    // │   提取 [ti*8:ti*8+8, 0:512]  → tensor<8x512xf32>        │
    // │   M 维度被 tiling (分成 8), K 维度保持完整             │
    // │                                                          │
    // │ %rhs: tensor<512x512xf32>  (K=512, N=512)               │
    // │   提取 [0:512, tj*32:tj*32+32]  → tensor<512x32xf32>    │
    // │   K 维度保持完整 (用于归约), N 维度被 tiling (分成 32)  │
    // │                                                          │
    // │ 这是由 matmul 的语义决定的：                             │
    // │   C[i,j] = sum_k(A[i,k] * B[k,j])                      │
    // │   每个输出 tile C[ti*8:ti*8+8, tj*32:tj*32+32] 需要：   │
    // │     - A[ti*8:ti*8+8, 0:512]  (完整的 K 列)              │
    // │     - B[0:512, tj*32:tj*32+32]  (完整的 K 行)           │
    // └─────────────────────────────────────────────────────────┘
    //
    // 生成代码结构：
    // scf.forall (%ti, %tj) in (64, 16) {
    //   %lhs_tile = extract_slice %lhs[ti*8, 0][8, 512]
    //   %rhs_tile = extract_slice %rhs[0, tj*32][512, 32]
    //   %matmul_tile = linalg.matmul on tiles
    //   %add_tile = linalg.elementwise kind=add (使用 %matmul_tile)
    //   %max_tile = linalg.elementwise kind=max (使用 %add_tile)
    // }
    // ═════════════════════════════════════════════════════════
    
    
    // ═════════════════════════════════════════════════════════
    // 要是 K 过大怎么办？见 "7.5 处理大 K 维度"
    // ═════════════════════════════════════════════════════════
    transform.yield
  }
}
```

### 6.4 融合后的代码

```cpp
func.func @fc_relu_tiled_fused(...) -> tensor<512x512xf32> {
  %result = scf.forall (%ti, %tj) in (64, 16)
      shared_outs(%output_arg = %output) -> (tensor<512x512xf32>) {

    // 提取 tiles
    %lhs_tile = tensor.extract_slice %lhs[%ti*8, 0] [8, 512] [1, 1]
    %rhs_tile = tensor.extract_slice %rhs[0, %tj*32] [512, 32] [1, 1]
    %bias_tile = tensor.extract_slice %bias[%ti*8, %tj*32] [8, 32] [1, 1]
    %output_tile = tensor.extract_slice %output_arg[%ti*8, %tj*32] [8, 32] [1, 1]

    // 三个操作在 tile 上完成，数据保持在寄存器/L1
    %matmul_tile = linalg.matmul
      ins(%lhs_tile, %rhs_tile)
      outs(%output_tile)
      -> tensor<8x32xf32>

    %add_tile = linalg.elementwise kind=add
      ins(%matmul_tile, %bias_tile)
      outs(%matmul_tile)
      -> tensor<8x32xf32>

    %max_tile = linalg.elementwise kind=max_signed
      ins(%add_tile, %c0f)
      outs(%add_tile)
      -> tensor<8x32xf32>

    // 只写回最终结果
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %max_tile into %output_arg[%ti*8, %tj*32] [8, 32]
    }
  }
  return %result
}
```

### 6.5 性能对比

| 指标       | 未优化    | Tile + Fuse        |
| ---------- | --------- | ------------------ |
| 内存写入   | 786K 元素 | 16K 元素           |
| 缓存命中率 | 低        | 高                 |
| 并行性     | 低        | 高（`scf.forall`） |

---

## 7. 高级特性介绍

### 7.1 不同类型的 Tiling

| 类型                | 循环结构         | 用途            |
| ------------------- | ---------------- | --------------- |
| `tile_using_for`    | `scf.for`        | 通用 tiling     |
| `tile_using_forall` | `scf.forall`     | 并行 tiling     |
| Partial reduction   | `scf.for` + 累加 | 归约维度 tiling |

### 7.2 参数化 Tiling

Tile size 可以在运行时确定：

```cpp
#map = affine_map<()[s0] -> (s0 ceildiv 32)>
affine.for %ti = 0 to %N step #map()[%N] {
  // tile size 根据输入大小动态计算
}
```

### 7.3 循环交换（Loop Interchange）

```cpp
options.interchange = {1, 0};  // 交换i,j循环顺序
```

### 7.4 Multi-way Split Tiling

```cpp
// 将25分割为 [18, 7], 再将7分割为 [4, 3]
%tile_sizes, %chunk_sizes = transform.structured.continuous_tile_sizes %0
  { dimension = 0, target_size = 9 }
%linalg_splits = transform.structured.split %0 after %chunk_sizes
  { dimension = 0, multiway }
```


### 7.5 处理大 K 维度

#### 问题场景

```cpp
// 假设 K 维度非常大 (例如 4096 或更大)
func.func @large_k_matmul(%A: tensor<512x4096xf32>,
                          %B: tensor<4096x512xf32>,
                          %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %matmul = linalg.matmul
    ins(%A, %B : tensor<512x4096xf32>, tensor<4096x512xf32>)
    outs(%C : tensor<512x512xf32>)
    -> tensor<512x512xf32>
  return %matmul
}
```

**问题**：按照之前的融合方式，每个 tile 需要加载：

- `%lhs_tile`: 8 x 4096 = 32K 元素 (~128KB)
- `%rhs_tile`: 4096 x 32 = 131K 元素 (~512KB)

这会超出 L1 缓存（通常 32-64KB），导致性能下降。

---

#### 解决方案 1：对 K 维度进行 Tiling

使用 **Partial Reduction Tiling** 或 **Pack + Tiling** 策略：

```cpp
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %root: !transform.any_op,
       %matmul: !transform.op<"linalg.matmul">) {

    // ═════════════════════════════════════════════════════════
    // Step 1: 先对归约维度 (K) 进行 tiling
    // ═════════════════════════════════════════════════════════
    // 使用 tile_reduction 进行 K 维度 tiling
    %tiled_k, %k_loop = transform.structured.tile_using_for %matmul
      tile_sizes [64]              // 只 tile K 维度 (索引2)
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

    // 生成代码结构：
    // scf.for %tk = 0 to 4096 step 64 {
    //   // K 维度被分成 64 大小的块
    //   %partial = linalg.matmul on partial K
    //   // 注意：这是部分归约，需要累加
    // }

    // ═════════════════════════════════════════════════════════
    // Step 2: 再对 M, N 维度进行 tiling
    // ═════════════════════════════════════════════════════════
    %tiled_mn, %mn_loop = transform.structured.tile_using_forall %tiled_k
      tile_sizes [8, 32]           // Tile M 和 N 维度
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

**生成的代码结构**：

```cpp
func.func @large_k_matmul_tiled(...) -> tensor<512x512xf32> {
  // M, N 维度的外层循环 (并行)
  %result = scf.forall (%ti, %tj) in (64, 16)
      iter_args(%C_accum = %C) -> (tensor<512x512xf32>) {

    %C_tile = tensor.extract_slice %C_accum[%ti*8, %tj*32] [8, 32] [1, 1]

    // K 维度的内层循环 (归约)
    %final_tile = scf.for %tk = 0 to 4096 step 64
        iter_args(%accum = %C_tile) -> (tensor<8x32xf32>) {

      // 提取 K 维度的 tile
      %A_tile = tensor.extract_slice %A[%ti*8, %tk] [8, 64] [1, 1]
        : tensor<512x4096xf32> to tensor<8x64xf32>

      %B_tile = tensor.extract_slice %B[%tk, %tj*32] [64, 32] [1, 1]
        : tensor<4096x512xf32> to tensor<64x32xf32>

      // 在小 tile 上执行 matmul (8x64 * 64x32 = 8x32)
      %partial = linalg.matmul
        ins(%A_tile, %B_tile : tensor<8x64xf32>, tensor<64x32xf32>)
        outs(%accum : tensor<8x32xf32>)
        -> tensor<8x32xf32>

      scf.yield %partial : tensor<8x32xf32>
    }

    // 将最终 tile 写回
    tensor.parallel_insert_slice %final_tile
      into %C_accum[%ti*8, %tj*32] [8, 32] [1, 1]
  }
  return %result
}
```

**内存使用对比**：

| 方案           | 每个 tile 的内存使用                  |
| -------------- | ------------------------------------- |
| 不 tile K 维度 | 8x4096 + 4096x32 = 163K 元素 (~640KB) |
| Tile K=64      | 8x64 + 64x32 = 2.5K 元素 (~10KB)      |

---

#### 解决方案 2：Pack + Tiling (数据重排)

> tensor.pack不懂没关系，见单独的介绍文章：[待补充](https://notlate.cn)

对于非常不规则的数据访问模式，可以先 Pack 数据：

```cpp
// 先对 A 和 B 进行 pack 重排
%A_packed = tensor.pack %A
  inner_dims_pos = [0, 1]           // 保持 M, K 维度顺序
  inner_tiles = [8, 64]              //打包成 8x64 的小块
  into tensor<64x8x64xf32>           // [K/64, M/8, 8, 64]

%B_packed = tensor.pack %B
  inner_dims_pos = [0, 1]           // 保持 K, N 维度顺序
  inner_tiles = [64, 32]             // 打包成 64x32 的小块
  into tensor<64x16x64x32xf32>       // [K/64, N/32, 64, 32]

// 然后在 packed 数据上进行 matmul
// 这可以让缓存预取更有效
```

---

#### 解决方案 3：使用 Pack 算子

MLIR 提供专门的 `tensor.pack` 操作用于优化数据布局：

```cpp
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%matmul: !transform.op<"linalg.matmul">) {

    // 在 tiling 之前先 pack
    %packed = transform.structured.pack_matrices %matmul
      packing_factors [8, 64, 32]     // [M_tile, K_tile, N_tile]
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op)

    // 对 packed matmul 进行 tiling
    %tiled, %loop = transform.structured.tile_using_forall %packed
      tile_sizes [1, 1, 1]           // 每个 tile 已经是 packed 大小
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

---

#### 关键点总结

| 场景              | 解决方案      | Tiling 策略                   |
| ----------------- | ------------- | ----------------------------- |
| K 适度大小 (< L2) | 标准 tile     | 只 tile M, N，保持 K 完整     |
| K 很大 (> L2)     | K 维度 tiling | 先 tile K (归约)，再 tile M,N |
| 访问不规则        | Pack + Tile   | 先 pack 数据布局，再 tile     |
| 寄存器优化        | 微内核        | tile 到更小尺寸 (4x4, 8x8)    |

---

## 8. 术语表

| 术语                | 解释                                      |
| ------------------- | ----------------------------------------- |
| **Tiling**          | 将大计算分解为小块的技术                  |
| **Tile**            | 分解后的小计算块                          |
| **Fuse**            | 将多个操作合并到同一个循环中              |
| **Dialect**         | MLIR 的方言，针对特定领域的操作集合       |
| **SSA**             | 静态单赋值形式，每个值只定义一次          |
| **Payload IR**      | 待优化的计算代码                          |
| **Transform IR**    | 描述如何优化的脚本                        |
| **Handle**          | Transform IR 中指向 Payload IR 操作的引用 |
| **TilingInterface** | MLIR 的统一 Tiling 接口                   |
| **Cache Locality**  | 缓存局部性，数据访问的聚集程度            |

---

## 9. 参考资源

### 官方文档

- [MLIR 官方文档](https://mlir.llvm.org/)
- [Transform Dialect 教程](https://mlir.llvm.org/docs/Tutorials/transform/)
- [Linalg Dialect 文档](https://mlir.llvm.org/docs/Dialects/Linalg/)

### 源码位置

- **接口定义**: `mlir/include/mlir/Interfaces/TilingInterface.td`
- **SCF 实现**: `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
- **Linalg 实现**: `mlir/lib/Dialect/Linalg/Transforms/Tiling.cpp`
- **测试文件**: `mlir/test/Dialect/Linalg/`

### 命令行工具

```bash
# 基本用法
mlir-opt input.mlir --pass-name -o output.mlir

# 常用 pass
mlir-opt input.mlir --affine-loop-tile="tile-size=32"
mlir-opt input.mlir --transform-interpreter
mlir-opt input.mlir --convert-linalg-to-loops
```
