---
title: "【MLIR】Linalg中FusePadOpWithLinalgProducer优化分析"
description: "【MLIR】Linalg中FusePadOpWithLinalgProducer优化分析 本文档基于 MLIR 版本 : 21.1.8分析，代码路径： mlir/lib/Dialect/Linalg/Transforms/FusePadOpWithLinalgProducer.cpp 1. 概…"
slug: "mlirlinalg-fusepadopwithlinalgproducer-analysis"
legacyId: 19544761
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19544761"
pubDate: 2026-01-28
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Linalg"]
featured: true
---

#【MLIR】Linalg中FusePadOpWithLinalgProducer优化分析

本文档基于**MLIR 版本**: 21.1.8分析，代码路径：`mlir/lib/Dialect/Linalg/Transforms/FusePadOpWithLinalgProducer.cpp` 

## 1. 概述

`FusePadOpWithLinalgProducer` 是 MLIR 中的一个优化转换，它将**张量填充操作（padding）**与其**上游计算操作（producer）**融合在一起，从而减少内存分配和数据复制的开销。

### 1.1 功能介绍

**简单类比**：想象你要在一张照片周围加上相框（padding），然后再对照片进行处理（比如调整亮度）。传统方法是：

1. 先给照片加相框（分配新内存，复制数据）
2. 再对整张带相框的照片处理

优化后的方法是：

1. 准备一个带相框大小的空画布
2. 直接在画布中心区域处理原始照片
3. 相框部分保持填充色

这样就避免了中间步骤的内存复制。

### 1.2 适用场景

- 深度学习模型编译（TensorFlow、PyTorch 等）
- 卷积神经网络中的 padding（填充）操作
- 图像处理 Pipeline
- 任何需要张量填充的数值计算

---

## 2. 背景与动机

### 2.1 为什么需要 Padding？

在张量计算中，padding 是一个基础操作，常见于：

1. **卷积神经网络（CNN）**

   ```
   原始图像: 28x28
   经过卷积: 26x26  (尺寸减小)
   加 padding: 28x28  (恢复尺寸)
   ```

   Padding 保证卷积后图像尺寸不变，防止边界信息丢失。

2. **内存对齐**

   ```
   原始数据: 15 个元素
   硬件要求: 16 的倍数对齐
   填充后:   16 个元素 (末尾填充 1 个)
   ```

   现代 CPU/GPU/NPU 的 SIMD 指令通常要求数据对齐。

3. **批处理**

   ```
   句子1: [5 个单词]
   句子2: [8 个单词]
   句子3: [6 个单词]
   填充后: 全部变成 [8 个单词]，方便批量处理
   ```

### 2.2 传统实现的问题

**未优化的代码逻辑**：

```llvm
%input = ...                          // 输入: tensor<10x10xf32>
%computed = linalg.generic ... %input // 计算: tensor<10x10xf32>
%padded = tensor.pad %computed ...    // 填充: tensor<14x14xf32>
```

**存在的问题**：

1. **额外内存分配**：`tensor.pad` 需要分配新的 14x14 张量
2. **数据复制开销**：将 10x10 的数据复制到新张量中心
3. **缓存效率低**：两次独立的内存操作，破坏了数据局部性
4. **无法进一步优化**：计算和填充分离，编译器难以联合优化

**性能影响示例**（假设场景）：

```
输入张量: 1024x1024xf32 (4MB)
计算: 元素级操作（100 GFLOPS）
填充: pad to 1280x1280xf32 (6.4MB)

未优化:
  - 计算时间: 40μs
  - 内存分配: 6.4MB (约 50μs)
  - 数据复制: 4MB (约 30μs)
  - 总耗时: ~120μs

优化后:
  - 一次性分配: 6.4MB (约 50μs)
  - 直接计算写入: 40μs
  - 填充区域已初始化: 0μs
  - 总耗时: ~90μs (提升 25%)
```

### 2.3 优化的核心思想

既然最终需要一个填充后的张量，为什么不直接在目标位置计算？

**优化后的代码逻辑**：

```llvm
%target = tensor.empty ...              // 1. 创建目标大小的空张量
%filled = linalg.fill %target ...       // 2. 用填充值初始化整个张量
%slice = extract_slice %filled ...      // 3. 提取中心区域（不复制数据）
%computed = linalg.generic ... %slice   // 4. 直接在中心区域计算
%result = insert_slice %computed ...    // 5. 结果已在正确位置
```

**优势**：

- 只分配一次内存
- 计算直接写入最终位置
- 消除了中间张量
- 为后续优化（如 tiling）创造条件

---

## 3. 核心概念

### 3.1 MLIR 与 Linalg 方言简介

**MLIR** 是一个编译器基础设施，支持多层次的中间表示：

```
高层: TensorFlow/PyTorch 模型
  ↓
中层: Linalg 方言（结构化的线性代数操作）
  ↓
底层: LLVM IR（接近机器码）
```

**Linalg 方言**提供了结构化的张量操作表示：

- `linalg.generic`: 通用的结构化计算
- `linalg.fill`: 填充张量
- `linalg.matmul`: 矩阵乘法
- ...

### 3.2 关键操作解释

#### 3.2.1 `tensor.pad` - 张量填充

```llvm
%padded = tensor.pad %source low [2, 2] high [1, 2] {
  ^bb0(%i: index, %j: index):
    tensor.yield %const : f32
} : tensor<10x10xf32> to tensor<13x14xf32>
```

**含义**：

- 在 `%source` 周围添加填充
- `low [2, 2]`: 第一维前面加 2 行，第二维前面加 2 列
- `high [1, 2]`: 第一维后面加 1 行，第二维后面加 2 列
- 填充值为 `%const`

**视觉表示**：

```
原始 (10x10):          填充后 (13x14):
┌──────────┐          ┌───────────────┐
│          │          │ P P P P P P P │ ← 2 行填充
│  DATA    │    →     │ P P P P P P P │
│          │          │ P P       P P │
└──────────┘          │ P P DATA  P P │ ← 2列填充 + 数据 + 2列填充
                      │ P P       P P │
                      │ P P       P P │
                      │ P P P P P P P │ ← 1 行填充
                      └───────────────┘
                         ↑         ↑
                      2列填充    2列填充
```

#### 3.2.2 `linalg.generic` - 通用计算

```llvm
%result = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // 输入映射
    affine_map<(d0, d1) -> (d0, d1)>   // 输出映射
  ],
  iterator_types = ["parallel", "parallel"]  // 迭代类型
} ins(%input : tensor<?x?xf32>)
  outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %squared = arith.mulf %in, %in : f32
    linalg.yield %squared : f32
} -> tensor<?x?xf32>
```

**含义**：对输入张量的每个元素求平方

**关键属性**：

- `indexing_maps`: 定义输入/输出的访问模式
- `iterator_types`:
  - `"parallel"`: 可以并行执行（如元素级操作）
  - `"reduction"`: 需要归约（如求和、求最大值）

#### 3.2.3 切片操作

```llvm
// 提取切片（不复制数据，只是创建视图）
%slice = tensor.extract_slice %tensor[2, 2][10, 10][1, 1]
  : tensor<13x14xf32> to tensor<10x10xf32>
// 含义: 从位置 [2,2] 开始，提取大小 [10,10]，步长 [1,1]

// 插入切片（将数据写回）
%result = tensor.insert_slice %computed into %target[2, 2][10, 10][1, 1]
  : tensor<10x10xf32> into tensor<13x14xf32>
```

---

## 4. 技术原理

### 4.1 转换模式详解

#### 4.1.1 原始代码模式

```llvm
// 步骤 1: 执行某种计算
%computed = linalg.generic {
  indexing_maps = [...],
  iterator_types = ["parallel", "parallel"]
} ins(%input : tensor<10x10xf32>)
  outs(%init : tensor<10x10xf32>) {
  ^bb0(%in: f32, %out: f32):
    // 某种计算，如: %result = %in * %in
    linalg.yield %result : f32
} -> tensor<10x10xf32>

// 步骤 2: 对结果进行填充
%padded = tensor.pad %computed low [2, 3] high [1, 2] {
  ^bb0(%i: index, %j: index):
    tensor.yield %pad_value : f32
} : tensor<10x10xf32> to tensor<13x15xf32>
```

#### 4.1.2 优化后的代码模式

```llvm
// 步骤 1: 创建目标大小的空张量
%empty = tensor.empty(13, 15) : tensor<13x15xf32>

// 步骤 2: 用填充值初始化整个张量
%filled = linalg.fill ins(%pad_value : f32)
                       outs(%empty : tensor<13x15xf32>)
          -> tensor<13x15xf32>

// 步骤 3: 提取中心区域（实际数据将写入的位置）
%slice = tensor.extract_slice %filled[2, 3][10, 10][1, 1]
  : tensor<13x15xf32> to tensor<10x10xf32>

// 步骤 4: 在切片上执行计算（直接写入目标位置）
%computed = linalg.generic {
  indexing_maps = [...],
  iterator_types = ["parallel", "parallel"]
} ins(%input : tensor<10x10xf32>)
  outs(%slice : tensor<10x10xf32>) {  // 注意：输出是切片
  ^bb0(%in: f32, %out: f32):
    linalg.yield %result : f32
} -> tensor<10x10xf32>

// 步骤 5: 将计算结果插回填充后的张量
%result = tensor.insert_slice %computed into %filled[2, 3][10, 10][1, 1]
  : tensor<10x10xf32> into tensor<13x15xf32>
```

### 4.2 为什么这样更高效？

#### 4.2.1 内存访问对比

**未优化版本**：

```
内存操作序列:
1. 分配: tensor<10x10xf32> (400 字节) - 用于 linalg.generic 输出
2. 写入: 计算结果写入 400 字节
3. 分配: tensor<13x15xf32> (780 字节) - 用于 tensor.pad 输出
4. 写入: 填充值写入边界区域 (380 字节)
5. 复制: 中心数据从旧张量复制到新张量 (400 字节)

总内存操作: 1180 字节分配 + 780 字节写入 + 400 字节复制 = 2360 字节
```

**优化版本**：

```
内存操作序列:
1. 分配: tensor<13x15xf32> (780 字节) - 最终大小
2. 写入: 填充值写入整个张量 (780 字节)
3. 写入: 计算结果直接写入中心区域 (400 字节，覆盖填充值)

总内存操作: 780 字节分配 + 1180 字节写入 = 1960 字节
节省: (2360 - 1960) / 2360 = 17% 内存流量
```

#### 4.2.2 数据流图示

**未优化**：

```
     输入数据
        ↓
   [linalg.generic] ← 分配临时内存 A
        ↓
     中间结果
        ↓
    [tensor.pad] ← 分配目标内存 B
        ↓           复制 A → B
      最终结果
```

**优化后**：

```
     输入数据              填充值
        ↓                    ↓
        ↓              [linalg.fill] ← 分配目标内存
        ↓                    ↓
        ↓          [extract_slice] (无复制)
        ↓                    ↓
        └→ [linalg.generic] ←┘ (直接写入目标)
                 ↓
         [insert_slice] (逻辑操作)
                 ↓
              最终结果
```

### 4.3 应用条件分析

这个优化**不是总能生效**，需要满足特定条件：

#### 4.3.1 条件 1: 常量填充值

```llvm
// ✅ 可以优化 - 常量填充
%padded = tensor.pad %source ... {
  tensor.yield %c0_f32 : f32  // 常量
}

// ❌ 不能优化 - 动态填充值
%padded = tensor.pad %source ... {
  ^bb0(%i: index, %j: index):
    %val = some_computation(%i, %j)  // 依赖于位置的动态值
    tensor.yield %val : f32
}
```

**原因**：`linalg.fill` 只能用单一常量填充整个张量。

#### 4.3.2 条件 2: 全并行迭代器

```llvm
// ✅ 可以优化 - 全并行
linalg.generic {
  iterator_types = ["parallel", "parallel"]
  // 可以任意顺序执行，结果相同
}

// ❌ 不能优化 - 包含归约
linalg.generic {
  iterator_types = ["parallel", "reduction"]
  // 例如：矩阵乘法，归约维度的顺序影响中间结果
}
```

**原因**：归约操作会破坏切片的独立性，可能导致错误结果。

**技术细节**（源码 `FusePadOpWithLinalgProducer.cpp:58-62`）：

```cpp
// All iterator types need to be parallel.
if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops()) {
  return rewriter.notifyMatchFailure(
      padOp, "only supported for ops with all parallel iterator types");
}
```

#### 4.3.3 条件 3: Linalg Generic 操作

```llvm
// ✅ 可以优化
%result = linalg.generic ...

// ❌ 当前不支持（但理论上可以扩展）
%result = linalg.matmul ...
%result = linalg.conv_2d ...
```

**原因**：当前实现保守，只支持 `linalg.generic`。注释表明可以扩展到其他 Linalg 操作。

### 4.4 转换算法步骤

**输入**：

- `padOp`: 一个 `tensor.pad` 操作
- `linalgOp`: pad 的源操作（`linalg.generic`）

**输出**：

- 等价的 `fill + extract_slice + generic + insert_slice` 序列

**算法流程**（对应源码行号）：

```
步骤 1: 验证条件 (L46-68)
├─ 检查填充值是否为常量
├─ 检查源操作是否为 linalg.generic
├─ 检查是否全并行迭代器
└─ 推导 pad 结果的形状

步骤 2: 创建填充后的目标张量 (L72-82)
├─ 计算目标张量大小（源大小 + 填充大小）
├─ 创建 tensor.empty
└─ 创建 linalg.fill 初始化

步骤 3: 计算切片参数 (L88-102)
├─ offsets = 低位填充值 [low_pad_0, low_pad_1, ...]
├─ sizes = 源张量形状 [src_dim_0, src_dim_1, ...]
│   ├─ 静态维度：直接使用常量
│   └─ 动态维度：使用 tensor.dim 查询
└─ strides = [1, 1, ...] (连续访问)

步骤 4: 提取切片 (L103-104)
└─ tensor.extract_slice 创建中心区域视图

步骤 5: 克隆并重定向计算 (L107-109)
├─ 克隆原始 linalg.generic 操作
└─ 将输出重定向到切片

步骤 6: 插入结果 (L112-114)
└─ tensor.insert_slice 将结果放回目标张量
```

---

## 5. 实例详解

### 5.1 实例 1: 动态形状的图像处理

#### 5.1.1 场景描述

假设你正在处理一批图像，每张图像的尺寸可能不同（动态形状）：

- 对每个像素应用某种滤波（平方操作）
- 然后在图像周围添加边框（padding）

#### 5.1.2 原始 MLIR 代码

```llvm
func.func @dynamic_pad_fusion(
    %input : tensor<?x?xf32>,      // 输入图像（动态尺寸）
    %low_pad_y : index,             // 上边框高度
    %low_pad_x : index,             // 左边框宽度
    %high_pad_y : index,            // 下边框高度
    %high_pad_x : index,            // 右边框宽度
    %border_color : f32             // 边框颜色
) -> tensor<?x?xf32> {

  // 获取输入图像尺寸
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %height = tensor.dim %input, %c0 : tensor<?x?xf32>
  %width = tensor.dim %input, %c1 : tensor<?x?xf32>

  // 创建输出缓冲区
  %output_buffer = tensor.empty(%height, %width) : tensor<?x?xf32>

  // 滤波操作：对每个像素求平方
  %filtered = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // 输入: input[i][j]
      affine_map<(d0, d1) -> (d0, d1)>   // 输出: output[i][j]
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>)
    outs(%output_buffer : tensor<?x?xf32>) {
    ^bb0(%pixel_in: f32, %pixel_out: f32):
      %squared = arith.mulf %pixel_in, %pixel_in : f32
      linalg.yield %squared : f32
  } -> tensor<?x?xf32>

  // 添加边框
  %with_border = tensor.pad %filtered
    low [%low_pad_y, %low_pad_x]
    high [%high_pad_y, %high_pad_x] {
    ^bb0(%y: index, %x: index):
      tensor.yield %border_color : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>

  return %with_border : tensor<?x?xf32>
}
```

#### 5.1.3 执行示例（具体数值）

假设调用参数：

```
input: 10x20 的图像
low_pad_y = 2, low_pad_x = 3
high_pad_y = 1, high_pad_x = 2
border_color = 0.0
```

**执行流程**：

```
1. 输入图像: 10x20
   ┌────────────────────┐
   │   原始像素数据       │
   │   (10 行 x 20 列)   │
   └────────────────────┘

2. 滤波后: 10x20 (分配新内存)
   ┌────────────────────┐
   │   每个像素平方后     │
   └────────────────────┘

3. 添加边框: 13x25
   ┌─────────────────────────┐
   │ 0 0 0 0 0 0 0 0 0 0 0 0 │ ← 2 行边框 (low_pad_y)
   │ 0 0 0 0 0 0 0 0 0 0 0 0 │
   │ 0 0 0 [滤波数据 10x20] 0 │ ← 3 列边框 + 数据 + 2 列边框
   │ 0 0 0                 0 │
   │  ...  (10 行数据)    ... │
   │ 0 0 0                 0 │
   │ 0 0 0 0 0 0 0 0 0 0 0 0 │ ← 1 行边框 (high_pad_y)
   └─────────────────────────┘
```

#### 5.1.4 优化后的 MLIR 代码

```llvm
func.func @dynamic_pad_fusion_optimized(...) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // 执行滤波（代码相同）
  %filtered = linalg.generic ... -> tensor<?x?xf32>

  // === 优化开始 ===

  // 计算最终尺寸
  %filtered_h = tensor.dim %filtered, %c0 : tensor<?x?xf32>
  %filtered_w = tensor.dim %filtered, %c1 : tensor<?x?xf32>

  // target_height = filtered_h + low_pad_y + high_pad_y
  #map = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
  %target_h = affine.apply #map()[%filtered_h, %low_pad_y, %high_pad_y]
  %target_w = affine.apply #map()[%filtered_w, %low_pad_x, %high_pad_x]

  // 1. 创建目标大小的张量并填充边框色
  %empty = tensor.empty(%target_h, %target_w) : tensor<?x?xf32>
  %filled_border = linalg.fill
    ins(%border_color : f32)
    outs(%empty : tensor<?x?xf32>)
    -> tensor<?x?xf32>

  // 2. 提取中心区域（数据将写入的位置）
  %center_region = tensor.extract_slice %filled_border
    [%low_pad_y, %low_pad_x]      // 偏移量
    [%filtered_h, %filtered_w]    // 大小
    [1, 1]                        // 步长
    : tensor<?x?xf32> to tensor<?x?xf32>

  // 3. 在中心区域执行滤波（直接写入最终位置）
  %filtered_in_place = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>)
    outs(%center_region : tensor<?x?xf32>) {  // 输出到中心区域
    ^bb0(%pixel_in: f32, %pixel_out: f32):
      %squared = arith.mulf %pixel_in, %pixel_in : f32
      linalg.yield %squared : f32
  } -> tensor<?x?xf32>

  // 4. 将结果插回（逻辑操作，实际已在正确位置）
  %result = tensor.insert_slice %filtered_in_place into %filled_border
    [%low_pad_y, %low_pad_x]
    [%filtered_h, %filtered_w]
    [1, 1]
    : tensor<?x?xf32> into tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}
```

#### 5.1.5 优化前后对比

**内存分配**：

```
未优化:
  - 第 1 次分配: tensor<10x20xf32> = 800 字节 (滤波输出)
  - 第 2 次分配: tensor<13x25xf32> = 1300 字节 (边框输出)
  - 总计: 2100 字节

优化后:
  - 唯一分配: tensor<13x25xf32> = 1300 字节
  - 总计: 1300 字节
  - 节省: 38% 内存分配
```

**数据移动**：

```
未优化:
  - 写入滤波结果: 800 字节
  - 复制到带边框张量: 800 字节
  - 写入边框值: 500 字节
  - 总计: 2100 字节数据移动

优化后:
  - 写入边框值: 1300 字节
  - 写入滤波结果: 800 字节 (覆盖中心区域)
  - 总计: 2100 字节数据移动
  - 注: 虽然总量相同，但消除了复制操作，提升缓存局部性
```

#### 5.1.6 动态形状处理要点

代码中的关键技术：

1. **运行时尺寸查询**：

```llvm
%height = tensor.dim %input, %c0 : tensor<?x?xf32>
```

在运行时获取张量的动态维度。

2. **仿射表达式计算**：

```llvm
#map = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
%target_h = affine.apply #map()[%filtered_h, %low_pad_y, %high_pad_y]
```

编译器可以优化这些表达式，在某些情况下编译时求值。

3. **灵活的切片操作**：

```llvm
%slice = tensor.extract_slice %filled[%offset_y, %offset_x][%h, %w][1, 1]
```

支持动态偏移和大小。

---

### 5.2 实例 2: 混合静态/动态维度的转置操作

#### 5.2.1 场景描述

处理一批文本嵌入向量，其中：

- 词汇表大小固定为 42（静态）
- 批量大小动态变化
- 需要转置并添加 padding

#### 5.2.2 原始代码

```llvm
func.func @mixed_pad_fusion(
    %input : tensor<?x42xf32>,  // 批量大小未知 x 固定嵌入维度
    %low_pad_dynamic : index,    // 动态低位填充
    %high_pad_dynamic : index,   // 动态高位填充
    %pad_value : f32             // 填充值
) -> tensor<49x?xf32> {         // 输出: 49 (= 42 + 3 + 4) x 动态

  %c0 = arith.constant 0 : index
  %batch_size = tensor.dim %input, %c0 : tensor<?x42xf32>

  // 初始化转置后的输出缓冲区
  %transposed_init = tensor.empty(%batch_size) : tensor<42x?xf32>

  // 转置 + 平方操作
  %transposed = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,  // 输入: [batch, 42]
      affine_map<(d0, d1) -> (d1, d0)>   // 输出: [42, batch] (转置)
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x42xf32>)
    outs(%transposed_init : tensor<42x?xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %squared = arith.mulf %in_val, %in_val : f32
      linalg.yield %squared : f32
  } -> tensor<42x?xf32>

  // 填充: 第一维 +3+4=7，第二维动态填充
  %padded = tensor.pad %transposed
    low [3, %low_pad_dynamic]
    high [4, %high_pad_dynamic] {
    ^bb0(%i: index, %j: index):
      tensor.yield %pad_value : f32
  } : tensor<42x?xf32> to tensor<49x?xf32>

  return %padded : tensor<49x?xf32>
}
```

#### 5.2.3 维度分析

**第一维（静态）**：

```
原始: 42 (固定的嵌入维度)
低位填充: 3 (常量)
高位填充: 4 (常量)
结果: 42 + 3 + 4 = 49 (编译时已知)
```

**第二维（动态）**：

```
原始: batch_size (运行时确定)
低位填充: %low_pad_dynamic (运行时传入)
高位填充: %high_pad_dynamic (运行时传入)
结果: batch_size + low_pad_dynamic + high_pad_dynamic (运行时计算)
```

#### 5.2.4 优化后的代码

```llvm
func.func @mixed_pad_fusion_optimized(...) -> tensor<49x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // 执行转置操作（代码相同）
  %transposed = linalg.generic ... -> tensor<42x?xf32>

  // === 优化开始 ===

  // 计算最终尺寸
  // 第一维: 静态计算 49 = 42 + 3 + 4
  // 第二维: 动态计算
  %transposed_dim1 = tensor.dim %transposed, %c1
  #map = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
  %target_dim1 = affine.apply #map()[
    %transposed_dim1,
    %low_pad_dynamic,
    %high_pad_dynamic
  ]

  // 1. 创建目标张量 (第一维静态 49，第二维动态)
  %empty = tensor.empty(%target_dim1) : tensor<49x?xf32>
  %filled = linalg.fill
    ins(%pad_value : f32)
    outs(%empty : tensor<49x?xf32>)
    -> tensor<49x?xf32>

  // 2. 提取中心区域
  // 注意: [3, %low_pad_dynamic] - 混合静态/动态偏移
  //      [42, %transposed_dim1] - 混合静态/动态大小
  %center = tensor.extract_slice %filled
    [3, %low_pad_dynamic]           // 偏移: 静态 3，动态 low_pad
    [42, %transposed_dim1]          // 大小: 静态 42，动态 batch_size
    [1, 1]
    : tensor<49x?xf32> to tensor<42x?xf32>

  // 3. 在中心区域执行转置计算
  %result_center = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1, d0)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x42xf32>)
    outs(%center : tensor<42x?xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %squared = arith.mulf %in_val, %in_val : f32
      linalg.yield %squared : f32
  } -> tensor<42x?xf32>

  // 4. 插回结果
  %result = tensor.insert_slice %result_center into %filled
    [3, %low_pad_dynamic]
    [42, %transposed_dim1]
    [1, 1]
    : tensor<42x?xf32> into tensor<49x?xf32>

  return %result : tensor<49x?xf32>
}
```

#### 5.2.5 编译器优化机会

对于静态维度，编译器可以进行更激进的优化：

**静态维度的循环展开**：

```cpp
// 编译后的伪代码
for (int i = 0; i < 3; i++) {      // 低位填充（静态）
  for (int j = 0; j < dim1; j++) {  // 动态维度
    output[i][j] = pad_value;
  }
}
// 这个循环可能被展开为 3 个独立循环

for (int i = 3; i < 45; i++) {     // 数据区域（静态 42 行）
  for (int j = low_pad; j < low_pad + dim1; j++) {
    output[i][j] = compute(...);    // 转置计算
  }
}

for (int i = 45; i < 49; i++) {    // 高位填充（静态）
  for (int j = 0; j < target_dim1; j++) {
    output[i][j] = pad_value;
  }
}
```

#### 5.2.6 测试用例验证

**测试输入**（`pad_fusion.mlir:54`）：

```bash
# 编译并运行测试
mlir-opt -test-linalg-pad-fusion pad_fusion.mlir | FileCheck pad_fusion.mlir
```

**预期输出检查**（`pad_fusion.mlir:74-93`）：

```
✓ 验证: 创建正确大小的 tensor.empty
✓ 验证: 使用 linalg.fill 初始化
✓ 验证: extract_slice 使用正确的偏移 [3, %low_pad_dynamic]
✓ 验证: generic 操作输出重定向到切片
✓ 验证: insert_slice 参数与 extract_slice 匹配
✓ 验证: 最终返回类型为 tensor<49x?xf32>
```

---

## 6. 性能分析

### 6.1 理论性能模型

#### 6.1.1 时间复杂度

假设张量大小为 `N x M`，填充后为 `(N+P) x (M+Q)`：

**未优化版本**：

```
T_unoptimized = T_compute + T_alloc + T_copy + T_pad_write
  = O(N×M) + O(1) + O(N×M) + O(P×M + Q×N + P×Q)
  = O(N×M) + 常数
```

**优化版本**：

```
T_optimized = T_alloc + T_fill + T_compute_inplace
  = O(1) + O((N+P)×(M+Q)) + O(N×M)
  = O(N×M + N×P + M×Q + P×Q)
```

**分析**：

- 当填充量相对于数据量较小时（P << N, Q << M），优化版本略慢（多了填充整个张量的开销）
- 但消除了数据复制，提升了缓存局部性
- 为后续优化（如 tiling、循环融合）创造了条件

#### 6.1.2 空间复杂度

**未优化版本**：

```
S_unoptimized = S_input + S_intermediate + S_output
  = N×M + N×M + (N+P)×(M+Q)
  = 2×N×M + (N+P)×(M+Q)
```

**优化版本**：

```
S_optimized = S_input + S_output
  = N×M + (N+P)×(M+Q)
```

**节省**：

```
ΔS = N×M (消除了中间张量)
```

### 6.2 真实场景性能估算

#### 6.2.1 场景 1: 小图像（224x224 RGB）

```
输入: 224 x 224 x 3 (float32) = 602KB
填充: pad 2 像素 → 228 x 228 x 3 = 624KB

未优化:
  - 中间张量: 602KB
  - 最终张量: 624KB
  - 峰值内存: 1226KB
  - 数据复制: 602KB

优化后:
  - 最终张量: 624KB
  - 峰值内存: 624KB
  - 节省: 49% 峰值内存

预计加速: 5-10% (主要得益于缓存优化)
```

#### 6.2.2 场景 2: 大批量 NLP（batch=256, seq=512）

```
输入: 256 x 512 (float32) = 512KB
填充: pad to 256 x 1024 = 1MB

未优化:
  - 中间张量: 512KB
  - 最终张量: 1MB
  - 峰值内存: 1.5MB
  - 数据复制: 512KB

优化后:
  - 最终张量: 1MB
  - 峰值内存: 1MB
  - 节省: 33% 峰值内存

预计加速: 15-20% (大张量，复制开销显著)
```

#### 6.2.3 场景 3: 高分辨率图像（4K）

```
输入: 3840 x 2160 x 3 (float32) = 99MB
填充: pad 16 像素 → 3872 x 2192 x 3 = 102MB

未优化:
  - 中间张量: 99MB
  - 最终张量: 102MB
  - 峰值内存: 201MB
  - 数据复制: 99MB (约 10ms on DDR4-3200)

优化后:
  - 最终张量: 102MB
  - 峰值内存: 102MB
  - 节省: 49% 峰值内存

预计加速: 20-30% (大数据量，内存带宽瓶颈)
```

### 6.3 与后续优化的协同效应

#### 6.3.1 与 Tiling 融合

优化后的代码结构更容易进行 tiling：

```llvm
// 优化前: 难以 tile（需要处理中间张量）
%intermediate = linalg.generic ...
%padded = tensor.pad %intermediate ...

// 优化后: 可以直接在 fill 结果上 tile
%filled = linalg.fill ...
scf.for %i = ... {
  scf.for %j = ... {
    %tile = extract_slice %filled[%i, %j] ...
    %computed_tile = linalg.generic ... outs(%tile)
    insert_slice %computed_tile into %filled[%i, %j] ...
  }
}
```

**性能提升**：

- 减少缓存缺失（tile 可以放入 L1/L2 缓存）
- 提升并行度（不同 tile 可以并行处理）
- 预计额外加速: 2-5x（取决于硬件）

#### 6.3.2 与向量化结合

```cpp
// 编译器可以生成 SIMD 指令
// 例如: AVX-512 可以一次处理 16 个 float32

// 填充操作 → broadcast + store
__m512 pad_vec = _mm512_set1_ps(pad_value);
for (int i = 0; i < size; i += 16) {
  _mm512_store_ps(&output[i], pad_vec);
}

// 计算操作 → vectorized computation
for (int i = 0; i < size; i += 16) {
  __m512 data = _mm512_load_ps(&input[i]);
  __m512 result = _mm512_mul_ps(data, data);
  _mm512_store_ps(&output[offset + i], result);
}
```

**预计加速**: 4-8x（在支持 AVX-512 的 CPU 上）

---

## 7. 源码解析

### 7.1 文件结构

**路径**: `mlir/lib/Dialect/Linalg/Transforms/FusePadOpWithLinalgProducer.cpp`

**依赖**:

```cpp
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // 变换接口
#include "mlir/Dialect/Linalg/IR/Linalg.h"              // Linalg 操作定义
```

### 7.2 核心类：FusePadOp

#### 7.2.1 类定义 (L40-41)

```cpp
struct FusePadOp : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
```

**设计模式**：

- 继承自 `OpRewritePattern<tensor::PadOp>` - 专门匹配和重写 `tensor.pad` 操作
- 使用 MLIR 的模式重写框架

#### 7.2.2 主方法：matchAndRewrite (L43-116)

**签名**：

```cpp
LogicalResult matchAndRewrite(
    tensor::PadOp padOp,           // 待优化的 pad 操作
    PatternRewriter &rewriter      // 重写器（用于创建新操作）
) const override
```

**返回值**：

- `success()`: 成功应用优化
- `rewriter.notifyMatchFailure(...)`: 无法应用（附带原因）

#### 7.2.3 详细步骤解析

**步骤 1: 验证填充值 (L45-48)**

```cpp
// Only works on padding op that sets the padded value to a constant.
Value padValue = padOp.getConstantPaddingValue();
if (!padValue)
  return rewriter.notifyMatchFailure(padOp, "non constant padding");
```

**关键 API**：

- `getConstantPaddingValue()`: 如果填充区域使用常量，返回该常量；否则返回 null

**失败案例**：

```llvm
// ❌ 这种情况会失败
%padded = tensor.pad %source ... {
  ^bb0(%i: index, %j: index):
    %dynamic = arith.addi %i, %j  // 非常量
    %float_val = arith.index_cast %dynamic : index to f32
    tensor.yield %float_val : f32
}
```

**步骤 2: 验证源操作 (L52-57)**

```cpp
Value source = padOp.getSource();
auto linalgOp = source.getDefiningOp<linalg::GenericOp>();
if (!linalgOp) {
  return rewriter.notifyMatchFailure(
      padOp, "expected source to be linalg.generic op");
}
```

**关键检查**：

- 源必须是 `linalg.generic` 操作
- 通过 `getDefiningOp<T>()` 进行类型检查和转换

**步骤 3: 验证迭代器类型 (L58-62)**

```cpp
// All iterator types need to be parallel.
if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops()) {
  return rewriter.notifyMatchFailure(
      padOp, "only supported for ops with all parallel iterator types");
}
```

**为什么需要全并行？**
考虑包含归约的情况：

```llvm
// ❌ 包含归约 - 不能应用优化
%result = linalg.generic {
  iterator_types = ["parallel", "reduction"]
} ins(%A, %B) outs(%C) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %c, %prod : f32
    linalg.yield %sum : f32
}
// 归约操作的中间状态依赖于执行顺序
// 在切片上执行会破坏正确性
```

**步骤 4: 推导输出形状 (L63-68)**

```cpp
ReifiedRankedShapedTypeDims resultShape;
if (failed(reifyResultShapes(rewriter, padOp, resultShape)) ||
    resultShape.size() != 1) {
  return rewriter.notifyMatchFailure(
      padOp, "failed to get shape of pad op result");
}
```

**形状推导**：

- `reifyResultShapes()`: 将抽象形状具体化为 MLIR 值（SSA values）
- 例如：`tensor<?x?xf32>` → `[%dim0, %dim1]`（运行时值）

**步骤 5: 创建目标张量 (L72-82)**

```cpp
Location loc = padOp.getLoc();
RankedTensorType padResultType = padOp.getResultType();
auto resultSizes = resultShape[0];

// Create the tensor of same size as output of the pad op.
auto emptyTensor = rewriter.create<tensor::EmptyOp>(
    loc, resultSizes, padResultType.getElementType());

// Fill the tensor with the pad value.
auto fillTensor = rewriter.create<linalg::FillOp>(
    loc, padValue, emptyTensor.getResult());
```

**生成的 IR**：

```llvm
%empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
%filled = linalg.fill ins(%pad_value : f32) outs(%empty : tensor<?x?xf32>)
  -> tensor<?x?xf32>
```

**关键点**：

- `tensor.empty`: 不初始化内存（undef 值），只分配空间
- `linalg.fill`: 将整个张量填充为常量值

**TODO 注释 (L79-80)**：

```cpp
// TODO: There is an option to fill only the boundaries. For now just
// filling the whole tensor.
```

**优化机会**：当前实现填充整个张量，但实际上只需要填充边界区域。未来可以优化为：

```llvm
// 当前: 填充整个 13x15 张量
%filled = linalg.fill ins(%pad_value) outs(%empty_13x15)

// 潜在优化: 只填充边界
%filled = linalg.fill ins(%pad_value) outs(%empty_13x15)
  region = boundaries_only  // 伪代码：只填充边界
```

**步骤 6: 计算切片参数 (L84-102)**

```cpp
// Construct a slice of the fill result that is to be replaced with the
// result of the generic op. The low pad values are the offsets, the size of
// the source is the size of the slice.
unsigned resultNumber = cast<OpResult>(source).getResultNumber();
SmallVector<OpFoldResult> offsets = padOp.getMixedLowPad();
SmallVector<OpFoldResult> sizes;
sizes.reserve(offsets.size());

for (const auto &shape :
     llvm::enumerate(cast<RankedTensorType>(source.getType()).getShape())) {
  if (ShapedType::isDynamic(shape.value())) {
    // 动态维度：运行时查询
    sizes.push_back(
        rewriter.create<tensor::DimOp>(loc, source, shape.index())
            .getResult());
  } else {
    // 静态维度：编译时常量
    sizes.push_back(rewriter.getIndexAttr(shape.value()));
  }
}

SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
```

**数据类型解释**：

- `OpFoldResult`: 可以是编译时常量（Attribute）或运行时值（Value）
- 这种设计允许编译器在可能时进行常量折叠

**示例**：

```
源张量: tensor<10x20xf32>
低位填充: [2, 3]

offsets = [2, 3]             // 切片起始位置
sizes = [10, 20]             // 切片大小（源张量形状）
strides = [1, 1]             // 连续访问
```

**步骤 7: 提取切片 (L103-104)**

```cpp
auto slice = rewriter.create<tensor::ExtractSliceOp>(
    loc, fillTensor.getResult(0), offsets, sizes, strides);
```

**生成的 IR**：

```llvm
%slice = tensor.extract_slice %filled[2, 3][10, 20][1, 1]
  : tensor<13x25xf32> to tensor<10x20xf32>
```

**重要特性**：

- `extract_slice` 是一个**视图操作**（view operation）
- 不复制数据，只创建对原张量子区域的引用
- 零开销抽象（zero-cost abstraction）

**步骤 8: 克隆并重定向 generic 操作 (L106-109)**

```cpp
// Clone the generic op.
auto clonedOp =
    cast<linalg::GenericOp>(rewriter.clone(*linalgOp.getOperation()));
clonedOp.setDpsInitOperand(resultNumber, slice.getResult());
```

**关键 API**：

- `clone()`: 深拷贝操作及其所有属性
- `setDpsInitOperand()`: 设置 DPS（Destination Passing Style）的输出操作数

**DPS 解释**：

```llvm
// DPS: 输出张量作为参数传入
linalg.generic ... outs(%output_tensor) {
  // 计算直接写入 %output_tensor
}

// 非 DPS（传统风格）：
%result = some_op(...)  // 操作分配输出内存
```

**步骤 9: 插入结果 (L112-114)**

```cpp
// Insert it back into the result of the fill.
rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
    padOp, clonedOp.getResult(resultNumber), fillTensor.getResult(0),
    offsets, sizes, strides);
return success();
```

**生成的 IR**：

```llvm
%result = tensor.insert_slice %computed into %filled[2, 3][10, 20][1, 1]
  : tensor<10x20xf32> into tensor<13x25xf32>
```

**语义**：

- 将 `%computed` 的内容写入 `%filled` 的指定区域
- 参数与 `extract_slice` 完全对应

**关键方法**：

- `replaceOpWithNewOp()`: 创建新操作并替换旧操作（原子操作）
- 确保 MLIR IR 始终有效（SSA 形式）

### 7.3 注册接口 (L120-123)

```cpp
void mlir::linalg::populateFuseTensorPadWithProducerLinalgOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FusePadOp>(patterns.getContext());
}
```

**使用方式**：

```cpp
// 在 Pass 中注册模式
RewritePatternSet patterns(&getContext());
populateFuseTensorPadWithProducerLinalgOpPatterns(patterns);

// 应用贪心重写
if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
  signalPassFailure();
```

### 7.4 测试 Pass 实现

**文件**: `mlir/test/lib/Dialect/Linalg/TestPadFusion.cpp`

```cpp
struct TestPadFusion : public PassWrapper<TestPadFusion, OperationPass<>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFuseTensorPadWithProducerLinalgOpPatterns(patterns);

    // 贪心应用所有匹配的模式
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
```

**贪心策略**：

- 不断尝试应用模式，直到没有模式可以匹配
- 适合融合类优化（可能创造新的融合机会）

---

## 8. 局限性与展望

### 8.1 当前限制

#### 8.1.1 仅支持 linalg.generic

**代码** (L50):

```cpp
// This pattern could work for any Linalg op. For now restrict it to generic
// ops.
```

**影响**：

- 不支持 `linalg.matmul`、`linalg.conv_2d` 等命名操作
- 需要手动将这些操作转换为 `linalg.generic` 形式

**扩展方案**：

```cpp
// 可以添加更多模式
struct FusePadWithMatmul : OpRewritePattern<tensor::PadOp> {
  LogicalResult matchAndRewrite(...) {
    auto matmulOp = source.getDefiningOp<linalg::MatmulOp>();
    if (!matmulOp) return failure();
    // 类似的融合逻辑
  }
};

void populateFuseTensorPadWithProducerLinalgOpPatterns(...) {
  patterns.add<FusePadOp, FusePadWithMatmul, FusePadWithConv>(context);
}
```

#### 8.1.2 填充整个张量

**代码(L79)**:

```cpp
// TODO: There is an option to fill only the boundaries. For now just
// filling the whole tensor.
```

**当前行为**：

```
填充区域:  ████████████████
          ██░░░░░░░░░░░██
          ██░ DATA    ░██
          ██░░░░░░░░░░░██
          ████████████████
█ = 填充值写入
░ = 将被覆盖的区域（浪费的写入）
```

**优化后**：

```
填充区域:  ████████████████
          ██            ██
          ██   DATA     ██
          ██            ██
          ████████████████
█ = 填充值写入（仅边界）
  = 未初始化（将被 DATA 覆盖）
```

**实现挑战**：

- 需要生成更复杂的循环结构
- 边界填充可能需要多个 `linalg.fill` 操作或自定义循环

#### 8.1.3 常量填充值限制

**原因**：

- `linalg.fill` 只接受常量值
- 动态填充值需要不同的实现策略

**替代方案**（未实现）：

```llvm
// 对于动态填充值，可以使用 linalg.generic
%padded = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} outs(%empty : tensor<?x?xf32>) {
  ^bb0(%out: f32):
    %is_boundary = compute_if_boundary(...)
    %value = scf.if %is_boundary {
      %pad_val = compute_dynamic_pad_value(...)
      scf.yield %pad_val
    } else {
      %data_val = load_from_source(...)
      scf.yield %data_val
    }
    linalg.yield %value : f32
}
```

#### 8.1.4 单一 Producer 限制

**当前假设**：pad 操作只有一个定义操作（producer）

**无法处理的情况**：

```llvm
// 多个使用者
%computed = linalg.generic ...
%padded1 = tensor.pad %computed ...  // 第一个 pad
%padded2 = tensor.pad %computed ...  // 第二个 pad（不同的填充参数）

// 如果融合，%computed 会被复制两次
```

**解决方案**：添加使用者数量检查

```cpp
if (!source.hasOneUse()) {
  return rewriter.notifyMatchFailure(
      padOp, "source has multiple uses");
}
```

### 8.2 未来演进方向

#### 8.2.1 扩展到更多 Linalg 操作

**roadmap**：

```
Phase 1: linalg.generic ✅ (已实现)
Phase 2: linalg.matmul, linalg.matvec
Phase 3: linalg.conv_2d, linalg.pooling
Phase 4: 自动识别可融合的自定义操作
```

#### 8.2.2 智能填充策略

**研究方向**：

- **自适应填充**: 根据填充比例选择策略

  ```
  if (padding_ratio < 0.1) {
    // 填充量小 → 只填充边界
    fill_boundaries_only();
  } else {
    // 填充量大 → 填充整个张量（代码更简单，编译器优化更好）
    fill_entire_tensor();
  }
  ```

- **延迟填充**: 在首次访问时才填充

  ```cpp
  // 使用页错误机制（需要运行时支持）
  allocate_with_lazy_init(size, pad_value);
  ```

#### 8.2.3 与其他优化的联合应用

**Tiling + Fusion**：

```llvm
// 当前: 两个独立的 Pass
Pass 1: 融合 pad
Pass 2: tile 融合后的操作

// 未来: 联合优化
Pass: tile-and-fuse-with-padding
  - 在 tiling 时考虑 padding
  - 每个 tile 独立处理边界
  - 避免全局填充
```

**向量化 + Masking**：

```cpp
// 使用 SIMD 掩码处理边界
for (int i = 0; i < size; i += 16) {
  if (i + 16 <= size) {
    // 完整向量：无掩码
    __m512 data = _mm512_load_ps(&input[i]);
    _mm512_store_ps(&output[i], process(data));
  } else {
    // 边界：使用掩码
    __mmask16 mask = (1 << (size - i)) - 1;
    __m512 data = _mm512_maskz_load_ps(mask, &input[i]);
    _mm512_mask_store_ps(&output[i], mask, process(data));
  }
}
```

#### 8.2.4 跨层次融合

**MLIR 的层次化设计**：

```
High-level Dialect (e.g., TensorFlow)
  ↓ 融合 pad 操作
Linalg Dialect ← 当前优化工作在这里
  ↓ 进一步融合
SCF Dialect (循环)
  ↓ 向量化
Vector Dialect
  ↓ 降级
LLVM Dialect
```

**跨层次优化示例**：

```llvm
// Linalg 层: 融合 pad + generic
%filled = linalg.fill ...
%result = linalg.generic ...

// 降级到 SCF 层: 生成循环
scf.for %i = ... {
  scf.for %j = ... {
    // 循环体可以进一步优化
  }
}

// 降级到 Vector 层: 向量化
vector.transfer_write %vec, %mem[%i, %j]
```

#### 8.2.5 自动调优

**机器学习驱动的优化选择**：

```python
# 伪代码
def should_fuse_pad(op_profile):
    features = extract_features(op_profile)
    # 特征: 张量大小、填充比例、硬件信息等

    decision = ml_model.predict(features)
    # 使用预训练的模型预测是否应该融合

    return decision > threshold
```

**AutoTVM/Ansor 集成**：

- 自动搜索最佳融合策略
- 考虑硬件特性（缓存大小、内存带宽）
- 生成特定硬件的优化代码

---

## 9. 总结

### 9.1 核心价值

`FusePadOpWithLinalgProducer` 优化体现了现代编译器优化的关键原则：

1. **算子融合（Operator Fusion）**
   - 将多个操作合并，减少中间结果的物化
   - 提升数据局部性，提高缓存命中率

2. **内存优化（Memory Optimization）**
   - 消除不必要的内存分配
   - 减少数据复制和移动

3. **结构化变换（Structured Transformation）**
   - 利用 Linalg 的结构化表示
   - 保持代码的可分析性和可优化性

4. **渐进式降级（Progressive Lowering）**
   - 在高层次进行优化，保留语义信息
   - 为后续低层次优化创造机会

### 9.2 适用场景总结

✅ **适用场景**：

- 深度学习模型推理（CNN、Transformer）
- 图像处理管线（滤波 + 边界处理）
- 科学计算（有限元分析的边界条件）
- 批处理系统（数据对齐和填充）

❌ **不适用场景**：

- 动态填充值（每个位置填充值不同）（暂不支持）
- 包含归约操作的 producer（暂不支持）
- 多个消费者共享同一 producer（暂不支持）
- 极小的张量（优化开销大于收益）

### 9.3 相关资源

**MLIR 官方文档**：

- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Tensor Semantics](https://mlir.llvm.org/docs/Rationale/Rationale/#tensor-types)

**代码仓库**：

- [MLIR Examples](https://github.com/llvm/llvm-project/tree/main/mlir/examples)

**社区资源**：

- [MLIR Discourse](https://discourse.llvm.org/c/mlir/)
- [MLIR Discord](https://discord.gg/xS7Z362)
- [LLVM Developers' Meeting Videos](https://www.youtube.com/c/LLVMPROJ/videos)

---

## 10. 附录：完整代码流程图

```
输入: %source = linalg.generic ... -> tensor<10x10xf32>
      %padded = tensor.pad %source [2,3][1,2] -> tensor<13x15xf32>

      ┌─────────────────────────────────────────┐
      │   FusePadOp::matchAndRewrite(padOp)     │
      └──────────────────┬──────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 1. 获取填充值: padValue = %const_0.0       │
      │    检查: 是否为常量?                       │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 2. 获取源操作: linalgOp = linalg.generic   │
      │    检查: 是否为 generic?                   │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 3. 检查迭代器类型                          │
      │    parallel 循环数 = 2                    │
      │    总循环数 = 2                           │
      │    检查: 全并行?                           │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 4. 推导输出形状                            │
      │    resultShape = [13, 15]                │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 5. 创建空张量                             │
      │    %empty = tensor.empty() : <13x15xf32> │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 6. 填充张量                               │
      │    %filled = linalg.fill                 │
      │      ins(%const_0.0) outs(%empty)        │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 7. 计算切片参数                            │
      │    offsets = [2, 3]  (lowPad)            │
      │    sizes = [10, 10]  (源形状)             │
      │    strides = [1, 1]                      │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 8. 提取切片                               │
      │    %slice = extract_slice %filled        │
      │      [2,3][10,10][1,1]                   │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 9. 克隆 generic 操作                       │
      │    %cloned = clone(linalgOp)             │
      │    cloned.setOutput(%slice)              │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 10. 插入结果                              │
      │     %result = insert_slice %cloned       │
      │       into %filled [2,3][10,10][1,1]     │
      └──────────────────┬───────────────────────┘
                         │
      ┌──────────────────▼───────────────────────┐
      │ 11. 替换原 pad 操作                        │
      │     rewriter.replaceOp(padOp, %result)   │
      └──────────────────┬───────────────────────┘
                         │
                      return success()

输出: %filled = linalg.fill ... -> tensor<13x15xf32>
      %slice = extract_slice %filled ... -> tensor<10x10xf32>
      %result_slice = linalg.generic ... outs(%slice)
      %final = insert_slice %result_slice into %filled
```
