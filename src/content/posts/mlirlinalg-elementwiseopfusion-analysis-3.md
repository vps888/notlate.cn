---
title: "【MLIR】Linalg中ElementwiseOpFusion优化分析（三）"
description: "本文介绍 mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp 中其他三种关键优化模式： 1. populateFoldReshapeOpsByExpansionPatterns 2. tensor::populateBubble…"
slug: "mlirlinalg-elementwiseopfusion-analysis-3"
legacyId: 19500691
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19500691"
pubDate: 2026-01-19
updatedDate: 2026-01-28
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Linalg"]
featured: true
---

本文介绍 `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp` 中其他三种关键优化模式：

1. `populateFoldReshapeOpsByExpansionPatterns`
2. `tensor::populateBubbleUpExpandShapePatterns`
3. `populateConstantFoldLinalgOperations`

---

## 1. populateFoldReshapeOpsByExpansionPatterns

### 核心原理

**维度扩展融合**：通过扩展 Linalg 操作的循环维度来融合 reshape 操作，消除中间的 reshape/collapse_shape 节点，减少内存访问开销。

该模式包含两个核心 Pattern：

- `FoldWithProducerReshapeOpByExpansion`：把 producer 的 `collapse_shape` 融合到 consumer 中
- `FoldReshapeWithGenericOpByExpansion`：把 `expand_shape` 融合到 producer 的 `generic op` 中

### 触发条件

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp:563
static bool isFusableWithReshapeByDimExpansion(LinalgOp linalgOp,
                                               OpOperand *fusableOpOperand) {
  // 条件1: 所有 indexing maps 必须是 projected permutation
  // 条件2: 融合的张量不能是标量
  // 条件3: 必须是纯 tensor 语义
  return linalgOp.hasPureTensorSemantics() &&
         llvm::all_of(linalgOp.getIndexingMaps().getValue(),
                      [](AffineMap map) { return map.isProjectedPermutation(); });
}
```

> *1. 何为 **ProjectedPermutation**？*
>
> **数学定义**：
>
>  组合了两种操作：                                                                  
>
> 1. **Projection (投影)**: 从输入维度中选择子集                                                     
> 2. **Permutation (排列)**: 对选中的维度重新排序                                                     
>
>  **形式化表示**                                                                     
>
>  给定输入维度$(d_0, d_1, ..., d_n)$，投影排列是映射：                                                  
>
>  $ f: (d_0, d_1, ..., d_n) -> (d_{i1}, d_{i2}, ..., d_{ik}) $                                       
>
>  其中：                                                                       
>
> -  $k ≤ n+1$ (结果数不超过输入数)                                                           
>
> - 每个 $d_{ix}$ 是不同的输入维度（无重复）                                                       
>
> - 顺序可以任意                                                                   
>
>  **实际例子**                                                                      
>
>  // 3D -> 2D 投影排列                                                                
>
> $ (d0, d1, d2) -> (d2, d0) $  // 选择 $d2,d0$，并重排                                                
>
>  // 纯投影（选择子集）                                                                
>
> $ (d0, d1, d2) -> (d1) $    // 只选 $d1$                                                       
>
>  // 纯排列                                                                      
>
> $ (d0, d1, d2) -> (d2, d1, d0)$  // 反转顺序                                                      
>
>  // 恒等映射                                                                     
>
>  $(d0, d1) -> (d0, d1) $    // 特殊情况                                                      
>
>  **反例（非投影排列）**                                                                 
>
> $ (d0, d1) -> (d0, d0) $    // ✗ 重复维度                                                     
>
> $ (d0, d1) -> (d0 + d1) $   // ✗ 计算表达式                                                    
>
> $ (d0, d1) -> (d0, d1, d2)$   // ✗ 结果维度多于输入                                                 
>
> $ (d0, d1)[s0] -> (d0, s0) $  // ✗ 包含符号                                                     
>
>  **为什么重要？**
>
>  在 Linalg 变换中，投影排列保证：                                                          
>
>  \- **维度关系简单**：无计算，只有索引重映射                                                       
>
>  \- **可逆性**：容易推导逆映射                                                              
>
>  \- **融合安全**：reshape 融合时不会产生复杂依赖     
>
> *2. **isProjectedPermutation** 的作用*
>
> **函数概念**: 判断一个 AffineMap 是否为ProjectedPermutation                                        
>
> **判定条件**:                                                                      
>
> 1. **无符号变量**: getNumSymbols() == 0                                                         
> 2. **结果数 ≤ 输入数**: getNumResults() <= getNumInputs()    (否则必有重复或零)                                                                 
>
> 3. **每个结果表达式**必须是:                                                              
>
>   \- **维度表达式** (AffineDimExpr): 且每个输入维度**最多出现一次**                                             
>
>   \- **常量零** (仅当 allowZeroInResults=true 时)                                                    
>
>  **例子**:                                                                        
>
>  (d0, d1, d2) -> (d1, d0)   ✓ 投影排列(选择+置换)                                                 
>
>  (d0, d1, d2) -> (d2)     ✓ 投影排列(仅选择)                                                  
>
>  (d0, d1) -> (d1, d0, 0)    ✓ (allowZeroInResults=true时)                                             
>
>  (d0, d1) -> (d0, d0)     ✗ d0重复                                                       
>
>  (d0, d1) -> (d0 + d1)     ✗ 非单纯维度表达式     

### 示例讲解

#### Before (未优化)

```text
// 原始: 2D tensor (16x64) -> collapse -> (16x1) -> generic op
%0 = tensor.collapse_shape %arg0 [[0, 1]] // 16x64 -> 16
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>,
                   affine_map<(d0) -> (d0)>]
  ins(%0 : tensor<16xf32>)
  outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
}
```

#### After (优化后)

```text
// 融合后: generic op 直接在原始 2D tensor 上操作
%1 = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // 扩展为 2D
                   affine_map<(d0, d1) -> (d0)>]        // 输出保持 1D
  ins(%arg0 : tensor<16x64xf32>)
  outs(%init : tensor<16xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
}
```

### 优化效果

- **消除内存拷贝**：不需要生成 collapse_shape 的中间张量
- **提升并行度**：扩展后的循环维度可以更好地利用并行硬件
- **减少访存**：融合后的操作可以复用缓存行

---

## 2. tensor::populateBubbleUpExpandShapePatterns

### 核心原理

**上浮变换**：当 `expand_shape` 的 producer 是 `collapse_shape` 时，如果两者的 reassociation 索引是parallel，则可以交换它们的位置，使 `expand_shape` 向上移动。

这样做的目的是让 `expand_shape` 能够与其他模式（如上述的扩展融合模式）配合，进一步优化。

### 触发条件

```cpp
// mlir/lib/Dialect/Tensor/Transforms/ReshapePatterns.cpp:169
// 两个 reshape 操作平行的条件：
// 1. reassociation 索引大小相同，或
// 2. collapse 或 expand 的 reassociation 大小为 1
for (auto [expandReassociation, collapseReassociation] :
     llvm::zip_equal(expandReInds, collapseReInds)) {
  if (collapseReassociation.size() == expandReassociation.size()) {
    // 验证静态形状是否一致
    continue;
  }
  if (collapseReassociation.size() != 1 && expandReassociation.size() != 1)
    return failure();  // 不平行，无法上浮
}
```

### 示例讲解

#### 输入形状

```text
tensor<4x?x4x32x4x?xf16> 
```

#### Before (未优化)

```text
%collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3, 4], [5]] : tensor<4x?x4x32x4x?xf16> into tensor<?x128x?xf16>        
%expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3], [4, 5]] : tensor<?x128x?xf16> into tensor<4x?x4x128x?x32xf16>   
```

#### After (上浮后)

```text
%expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3], [4], [5, 6]] : 
                             tensor<4x?x4x32x4x?xf16> into tensor<4x?x4x32x4x?x32xf16>                                         
%collapsed = tensor.collapse_shape %expanded [[0], [1], [2], [3, 4], [5], [6]] : 
                             tensor<4x?x4x32x4x?x32xf16> into tensor<4x?x4x128x?x32xf16>     
```

#### 形状变化

```text
原始:   [4, ?, 4, 32, 4, ?]                                                                                                   
expand: [4, ?, 4, 32, 4, ?, 32]  ← 最后维度被展开                                                                             
最终:   [4, ?, 4, 128, ?, 32]    ← [3,4] 被 collapse 
```

### 优化效果

- **为融合创造条件**：上浮后的 expand_shape 可能与更上层的 linalg 操作融合
- **消除冗余 reshape**：某些情况下 collapse 和 expand 会相互抵消

---

## 3. populateConstantFoldLinalgOperations

### 核心原理

**常量折叠**：当 Linalg 操作（如 transpose）的输入全部是编译时常量时，直接在编译期计算出结果常量，替换整个计算操作。

目前实现的 Pattern：

- `FoldConstantTranspose`：专门处理 transpose 操作

### 实现机制

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ConstantFold.cpp:265
struct FoldConstantTranspose : public FoldConstantBase<FoldConstantTranspose> {
  // 1. 验证 indexing maps 只有一个输入和一个输出
  // 2. 验证 region 只包含 yield op
  // 3. yield 直接返回输入（无实际计算）
  // 4. 根据 indexing maps 重排常量元素
}
```

常量重排使用**索引去线性化**技术：

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ConstantFold.cpp:181
auto computeRemappedLinearIndex = [&](int linearIndex) {
  // 线性索引 -> 多维索引
  for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
    indices[dim] = totalCount % loopBounds[dim];
    totalCount /= loopBounds[dim];
  }
  // 根据 indexing maps 映射到输入/输出的多维索引
  // 再转回线性索引进行访问
};
```

### 示例讲解

#### Before (未优化)

```text
// 编译期常量转置
%0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
%1 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d1, d0)>,  // transpose: (i,j) -> (j,i)
    affine_map<(d0, d1) -> (d0, d1)>
  ]
  ins(%0 : tensor<2x2xf32>)
  outs(%init : tensor<2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
}
```

#### After (常量折叠后)

```text
// 直接计算好的转置结果
%0 = arith.constant dense<[[1.0, 3.0], [2.0, 4.0]]> : tensor<2x2xf32>
```

### 优化效果

- **零运行时开销**：整个操作在编译期完成
- **减少代码体积**：消除循环和控制流
- **便于后续优化**：常量传播可继续向上传递

---

## 三种优化的协同工作

在 `LinalgElementwiseOpFusionPass` 中，这些模式按以下顺序应用：

```cpp
// mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp:2301
populateElementwiseOpsFusionPatterns(patterns, defaultControlFn);
populateFoldReshapeOpsByExpansionPatterns(patterns, defaultControlFn);
tensor::populateBubbleUpExpandShapePatterns(patterns);      // 上浮 expand
// ... canonicalization ...
populateConstantFoldLinalgOperations(patterns, defaultControlFn);  // 常量折叠
```

### 协同流程示例

#### 步骤 0: 未优化的原始代码

```text
%cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

%transpose = linalg.transpose
    ins(%cst : tensor<2x4xf32>)
    outs(%init_4x2 : tensor<4x2xf32>)
    permutation = [1, 0]
// 结果: [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]

%reshape = tensor.expand_shape %transpose [[0, 1], [2]]
    output_shape [2, 2, 2]
    : tensor<4x2xf32> into tensor<2x2x2xf32>
// 4x2 -> 2x2x2

%result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reshape : tensor<2x2x2xf32>)
    outs(%out : tensor<2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mul = arith.mulf %in, %in : f32
    linalg.yield %mul : f32
} -> tensor<2x2x2xf32>
```

#### 问题

- 常量在运行时执行 transpose（浪费）
- reshape 阻碍了与 generic 的融合
- 多次内存布局转换

#### 优化步骤 1：常量折叠 (Constant Folding)

**触发条件**: transpose 的输入是编译时常量

```text
// Pass: -canonicalize 或 -sccp

// 之前:
%cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
%transpose = linalg.transpose ins(%cst : ...) permutation = [1, 0]

// 之后: 直接折叠为转置后的常量
%cst_folded = arith.constant dense<[[1.0, 5.0],
                                     [2.0, 6.0],
                                     [3.0, 7.0],
                                     [4.0, 8.0]]> : tensor<4x2xf32>

// transpose 操作被消除！
%reshape = tensor.expand_shape %cst_folded [[0, 1], [2]] ...
%result = linalg.generic { ... }
```

**效果**:

- 消除运行时 transpose 计算
- 减少一次内存读写
- 暴露更多优化机会给后续 Pass

#### 优化步骤 2：扩展融合 (Reshape by Expansion Fusion)

**触发条件**: expand_shape 的生产者是 generic op，且满足 `isFusableWithReshapeByDimExpansion`

```text
// Pass: -linalg-fuse-elementwise-ops

// 假设常量折叠后，我们有一个 generic 生产者：
%producer = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%cst_folded : tensor<4x2xf32>)
    outs(%init : tensor<4x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %in : f32
    linalg.yield %add : f32
} -> tensor<4x2xf32>

%reshape = tensor.expand_shape %producer [[0, 1], [2]]
    : tensor<4x2xf32> into tensor<2x2x2xf32>

// ========== 融合后 ==========

%fused = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]  // 维度扩展！
  } ins(%input_expanded : tensor<2x2x2xf32>)  // 输入也被 expand
    outs(%init_3d : tensor<2x2x2xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %in : f32  // 计算逻辑不变
    linalg.yield %add : f32
} -> tensor<2x2x2xf32>

// reshape 操作被消除！
```

**关键机制**:

- `FoldReshapeWithGenericOpByExpansion` 模式匹配
- 通过 `fuseWithReshapeByExpansion` 将循环维度从 2D 扩展到 3D
- indexing map 从 `(d0, d1)` 变为 `(d0, d1, d2)`
- 原 reshape 的 `[[0, 1], [2]]` 映射到新的迭代空间

**效果**:

- 消除 reshape 操作
- 减少中间张量分配
- 计算直接在目标形状上进行

#### 优化步骤 3：上浮 expand (Bubble Up Expand Shape)

**触发条件**: expand 的生产者是 collapse，且满足Parallel Reassociation

> *Parallel Reassociation*: 
>
> **解释：**collapse和expand互为逆操作
>
> **条件**：
>
> 1. **Map 结构相同**：collapse 和 expand 的 reassociation maps 互为逆操作
> 2. **维度数量匹配**：collapse 的输入维度数 = expand 的输出维度数
> 3. **中间维度兼容**：collapse 的输出 shape = expand 的输入 shape

**中间操作不依赖原始维度场景**

```text
// Pass: -test-tensor-transform-patterns=test-expand-shape-bubbling

%input : tensor<2x3x4x5xf32>

// 步骤1：为了某个操作，需要合并维度
%collapsed = tensor.collapse_shape %input [[0, 1], [2], [3]]
  : tensor<2x3x4x5xf32> into tensor<6x4x5xf32>

// 步骤2：在 collapsed 形状上做 linalg 操作（不关心具体维度分解）
%result = linalg.matmul ins(%collapsed, ...) 
  : tensor<6x4x5xf32> ...

// 步骤3：expand 回原始维度供下游使用
%final = tensor.expand_shape %result [[0, 1], [2], [3]]
  : tensor<6x4x5xf32> into tensor<2x3x4x5xf32>

// ========== 上浮后 ==========
// 直接操作原始张量，跳过 reshape
%final = linalg.matmul ins(%input, ...) 
  : tensor<2x3x4x5xf32> ...
```

**更典型的场景**（expand 直接与 generic 相邻）:

```text
// 上浮前:
%collapsed = tensor.collapse_shape %input [[0], [1, 2], [3]]
    : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>

%expanded = tensor.expand_shape %collapsed [[0], [1], [2, 3]]
    : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>

%result = linalg.generic { ... } ins(%expanded : ...) ...

// 上浮后:
%expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
    : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?xf32>

%collapsed = tensor.collapse_shape %expanded [[0], [1, 2], [3], [4]]
    : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>

%result = linalg.generic { ... } ins(%collapsed : ...) ...

// 此时可以触发步骤 2 的融合！
```

**效果**:

- 调整 reshape 顺序，暴露融合机会
- 有时可以消除冗余的 reshape 对
- 为步骤 2 的扩展融合创造前置条件

### 实际代码对比

#### 优化前 (原始)

```text
func.func @before(%out: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>

  %transpose = linalg.transpose
      ins(%cst : tensor<2x4xf32>)
      outs(%t_init : tensor<4x2xf32>)
      permutation = [1, 0]

  %reshape = tensor.expand_shape %transpose [[0, 1], [2]]
      output_shape [2, 2, 2]
      : tensor<4x2xf32> into tensor<2x2x2xf32>

  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%reshape : tensor<2x2x2xf32>)
      outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  } -> tensor<2x2x2xf32>

  return %result : tensor<2x2x2xf32>
}

// 内存操作: 3 次分配 (transpose结果, reshape结果, generic结果)
// 计算次数: transpose + reshape + generic
```

#### 优化后 (全部 Pass 应用)

```text
func.func @after(%out: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // 常量已经是最终形状（折叠 + 融合的结果）
  %cst = arith.constant dense<[[[1.0, 5.0],
                                 [2.0, 6.0]],
                                [[3.0, 7.0],
                                 [4.0, 8.0]]]> : tensor<2x2x2xf32>

  // 直接计算，无中间步骤
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%cst : tensor<2x2x2xf32>)
      outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in: f32, %out_elem: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  } -> tensor<2x2x2xf32>

  return %result : tensor<2x2x2xf32>
}

// 内存操作: 1 次分配 (仅 generic 结果)
// 计算次数: 仅 generic (transpose和reshape被消除)
```

#### 相关 Pass 命令

```bash
# 常量折叠
mlir-opt --canonicalize input.mlir

# 上浮 expand shape
mlir-opt --test-tensor-transform-patterns=test-expand-shape-bubbling input.mlir

# Linalg 元素级融合（包含扩展融合）
mlir-opt --linalg-fuse-elementwise-ops input.mlir

# 完整优化流程
mlir-opt --canonicalize \
         --test-tensor-transform-patterns=test-expand-shape-bubbling \
         --linalg-fuse-elementwise-ops \
         input.mlir
```

---

## 总结

| 优化模式               | 核心技术     | 目标                  | 适用场景             |
| ---------------------- | ------------ | --------------------- | -------------------- |
| FoldReshapeByExpansion | 维度扩展融合 | 消除 reshape 中间结果 | Linalg + reshape 链  |
| BubbleUpExpandShape    | 变换上浮     | 为融合创造条件        | collapse + expand 链 |
| ConstantFold           | 编译期求值   | 消除运行时计算        | 常量输入的 Linalg    |

这三种优化通过减少内存访问、消除冗余操作和编译期求值，共同提升 MLIR 程序的性能。
