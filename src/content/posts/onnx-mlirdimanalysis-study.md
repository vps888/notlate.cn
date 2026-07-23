---
title: "【onnx-mlir】DimAnalysis功能学习"
description: "DimAnalysis 动态维度分析 1. 概述 DimAnalysis 是 onnx mlir 中用于在 编译时分析动态维度之间等价关系 的工具类。它能够判断两个动态维度在运行时是否相等，从而帮助编译器做出更优的代码生成决策。 2. 核心作用 2.1 消除不必要的广播代码 当两个张量的动态维…"
slug: "onnx-mlirdimanalysis-study"
legacyId: 19419609
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19419609"
pubDate: 2025-12-30
category: "AI 编译器"
tags: ["AI 编译器","MLIR","ONNX-MLIR"]
featured: true
---

## DimAnalysis 动态维度分析

### 1. 概述

`DimAnalysis` 是` onnx-mlir `中用于在**编译时分析动态维度之间等价关系**的工具类。它能够判断两个动态维度在运行时是否相等，从而帮助编译器做出更优的代码生成决策。

### 2. 核心作用

#### 2.1 消除不必要的广播代码

当两个张量的动态维度已知相等时，可以避免生成处理广播规则的运行时代码：

```cpp
%0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5xf32>, tensor<?x3x5xf32>) -> tensor<?x3x5xf32>
```

如果能在编译时确定` %arg0 `和 `%arg1 `的第一维相等，则无需生成广播处理代码。

#### 2.2 加速器适配决策

帮助判断某个操作是否可以卸载到加速器（如 NNPA），因为某些加速器不支持广播操作。

### 3. 核心数据结构

```cpp
// 维度类型：一个张量和一个维度轴组成的对
using DimT = std::pair<mlir::Value, uint64_t>;

// 维度集合：一组被认为相等的动态维度
using DimSetT = llvm::SmallDenseSet<DimT, 4>;

// 维度集合映射：将集合ID映射到维度集合
using DimSetMapT = llvm::SmallDenseMap<uint64_t, DimSetT, 4>;
```

### 4. 分析算法

DimAnalysis 使用不动点迭代算法，包含两个阶段：

#### 4.1 扩展阶段 (Expand)

通过以下方式发现相同的动态维度：

- ShapeHelper：利用操作的形状推导逻辑
- 消费者操作分析：分析使用该张量的操作
- 形状输入分析：针对 ConstantOfShape、Reshape 等操作

#### 4.2 合并阶段 (Merge)

将有共同元素的维度集合合并为单一集合。

```cpp
void DimAnalysis::analyze() {
  bool continued = true;
  while (continued) {
    // 本地搜索并更新每个维度集合
    continued = updateDimSets();
    if (continued)
      // 合并有共同维度的集合
      mergeDimSets();
  }
}
```

### 5. 主要 API

| 方法                                       | 作用                                      |
| ------------------------------------------ | ----------------------------------------- |
| sameDim(tensor1, axis1, tensor2, axis2)    | 判断两个维度是否相等（含静态维度比较）    |
| sameDynDim(tensor1, axis1, tensor2, axis2) | 判断两个动态维度是否相等                  |
| sameShape(tensor1, tensor2)                | 判断两个张量形状是否完全相同              |
| sameDynShape(tensor1, tensor2)             | 判断两个张量的动态维度是否相同            |
| broadcastLastDim(tensor1, tensor2)         | 判断 tensor1 是否按最后一维广播到 tensor2 |

### 6. 支持的操作类型

分析过程能识别以下操作的维度关系：

#### 6.1 矩阵运算

- MatMul / MatMulInteger：A[M×N] × B[N×P] → dimA[1] == dimB[0]
- Gemm：考虑转置属性

#### 6.2 RNN 类操作

- LSTM / GRU / RNN：识别 batch_size 维度在不同输入间的等价性

#### 6.3 形状操作

- Concat：非连接轴上的维度相等
- Reshape：通过静态维度乘积推导动态维度
- ConstantOfShape / Expand / Tile：从形状输入推导

#### 6.4 二元广播操作

- Add / Mul / Sub / Div / Where 等：当输入维度相同时，输出维度也相同

### 7. 使用示例

```cpp
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

ModuleOp moduleOp = getOperation();

// 构造分析器
onnx_mlir::DimAnalysis dimAnalysis(moduleOp);

// 执行分析
dimAnalysis.analyze();

// 查询两个动态维度是否相等
bool isSame = dimAnalysis.sameDynDim(tensor1, 0, tensor2, 0);

// 查询两个张量形状是否相同
bool shapeEqual = dimAnalysis.sameShape(tensor1, tensor2);
```

### 8. dim_params 支持

支持通过 ONNX 的 dim_params 属性来标记相同的维度参数：

// onnx.dim_params = "0:batch,1:seq_len"
// 具有相同 dim_param 名称的维度会被归入同一集合

### 9. 调试支持

提供 dump() 方法输出分析结果，以及通过 --onnx-dim-analysis pass 插入 onnx.DimGroup 操作用于测试验证。

---

文件位置：

- 头文件：src/Dialect/ONNX/ONNXDimAnalysis.hpp
- 实现：src/Dialect/ONNX/ONNXDimAnalysis.cpp
- 文档：docs/DynamicDimensionAnalysis.md
