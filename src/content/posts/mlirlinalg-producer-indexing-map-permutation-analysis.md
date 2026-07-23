---
title: "【MLIR】Linalg融合中Producer输出Indexing Map的Permutation检查分析"
description: "1. 问题背景 代码位置 核心问题 这个检查在做什么？ 获取producer输出的indexing map，检查它是否是permutation，为什么？ 2. Permutation是什么？ 定义 Permutation：维度的一一对应重排，每个输入维度恰好对应一个输出维度，每个输出维度恰好来…"
slug: "mlirlinalg-producer-indexing-map-permutation-analysis"
legacyId: 19449364
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19449364"
pubDate: 2026-01-06
updatedDate: 2026-01-19
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Linalg"]
featured: true
---

# 1. 问题背景

## 代码位置

```cpp
// ElementwiseOpFusion.cpp:175-178
AffineMap producerResultIndexMap =
    producer.getMatchingIndexingMap(producer.getDpsInitOperand(0));
if (!producerResultIndexMap.isPermutation())
  return false;
```

## 核心问题

这个检查在做什么？

获取producer输出的indexing map，检查它是否是permutation，为什么？

# 2. Permutation是什么？

## 定义

Permutation：维度的一一对应重排，每个输入维度恰好对应一个输出维度，每个输出维度恰好来自一个输入维度。

## 数学表达

排列映射的数学表达式为：$(d_0, d_1, d_2, \dots, d_n) \rightarrow (d_{\pi(0)}, d_{\pi(1)}, \dots, d_{\pi(n)})$

其中 $\pi$ 是一种permutation。

## 代码示例

✅ 合法的Permutation映射

```text
// 1. Identity（恒等映射）
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// 2. Transpose（转置）
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// 3. 复杂排列
#map2 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>  // 循环移位
#map3 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>  // 另一种排列
```

❌ 非法的非Permutation映射

```text
// 1. Broadcast（维度缺失）
#map4 = affine_map<(d0, d1) -> (d0)>  // d1维度消失了

// 2. Projection（投影）
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>  // d2维度消失了

// 3. Duplication（重复使用）
#map6 = affine_map<(d0, d1) -> (d0, d0)>  // d0使用了两次

// 4. Constant（常量索引）
#map7 = affine_map<(d0, d1) -> (0, d1)>  // 第一个维度是常量

// 5. Affine expression（仿射表达式）
#map8 = affine_map<(d0, d1) -> (d0 + d1, d0)>  // 使用了加法
```

# 3. 为什么必须是Permutation？核心原因

## 原因1：需要计算逆映射（Inverse Map）

融合算法的关键步骤：

```cpp
// ElementwiseOpFusion.cpp:57-59
AffineMap invProducerResultIndexMap =
    inversePermutation(producerResultIndexMap);
assert(invProducerResultIndexMap &&
       "expected producer result indexing map to be invertible");
```

### 数学原理

融合需要回答的问题： 给定consumer的循环索引 (i, j)，则producer的循环索引 (p, q) 应该是多少？

解决方法：通过逆映射转换坐标系

## 原因2：坐标系转换链

融合的IndexingMap转换公式：

$$ fusedMap = producerArgMap ∘ invProducerResultMap ∘ consumerArgMap $$                   

其中，$invProducerResultMap$需要逆映射！

### 具体过程（看不懂没关系，下一节案例详细解释计算流程）

```cpp
// ElementwiseOpFusion.cpp:47-74
static AffineMap getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
    OpOperand *producerOpOperand, 
    AffineMap producerResultIndexMap,
    AffineMap fusedConsumerArgIndexMap) {

  // 步骤1: 计算逆映射
  AffineMap invProducerResultIndexMap = inversePermutation(producerResultIndexMap);
  //    																^^^^^^^^^^^^^^^
  //    																如果不是permutation，这里会返回空！

  assert(invProducerResultIndexMap &&
         "expected producer result indexing map to be invertible");

  // 步骤2: 获取producer的arg map（输入操作数的映射map）
  LinalgOp producer = cast<LinalgOp>(producerOpOperand->getOwner());
  AffineMap argMap = producer.getMatchingIndexingMap(producerOpOperand);

  // 步骤3: 组合映射
  // argMap: producer loop -> producer arg
  // invProducerResultIndexMap: producer result -> producer loop
  AffineMap t1 = argMap.compose(invProducerResultIndexMap);

  // 步骤4: 最终映射
  // fusedConsumerArgIndexMap: consumer loop -> producer result
  return t1.compose(fusedConsumerArgIndexMap);
}
```

# 4. 详细案例分析

## 案例1：合法的Permutation（Transpose）

```text
// Producer: 转置操作
%producer = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // input
    affine_map<(d0, d1) -> (d1, d0)>   // output (transpose)
    //         ^^^^^^^^^^^^^^^^^^^^^^^^
    //         这是一个permutation！
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%A) outs(%init) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
} -> tensor<?x?xf32>

// Consumer
%consumer = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,  // input (%producer)
    affine_map<(d0, d1) -> (d0, d1)>   // output
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%producer) outs(%init2) {
  ^bb0(%in: f32, %out: f32):
    %mul = arith.mulf %in, %in : f32
    linalg.yield %mul : f32
}
```

### 转换步骤

1. $ producerResultIndexMap = (d0, d1) -> (d1, d0) $ 
2. 计算逆映射：

$$
\begin{align} invProducerResultIndexMap &= inversePermutation((d0, d1) -> (d1, d0))    \\                      
 &= (r0, r1) -> (r1, r0)    ✅ 成功！因为是permutation \\
 \end{align}
$$

3. $ producerArgMap = (d0, d1) -> (d0, d1)  $ [input的map] 

4. $ consumerArgMap = (d0, d1) -> (d0, d1)  $ [consumer访问producer]

5. 计算融合后的map： 

$$
\begin{align}
Step A: t1 &= argMap ∘ invProducerResultMap\\ 
&= (d0, d1) -> (d0, d1) ∘ (r0, r1) -> (r1, r0)\\ 
&= (r0, r1) -> (r1, r0)\\
Step B: fusedMap &= t1 ∘ consumerArgMap\\
&= (r0, r1) -> (r1, r0) ∘ (d0, d1) -> (d0, d1)\\     
&= (d0, d1) -> (d1, d0)\\
\end{align}
$$

6. 融合后的indexing map：        

```cpp
 linalg.generic {
   indexing_maps = [
     affine_map<(d0, d1) -> (d1, d0)>  // A需要转置访问！
     affine_map<(d0, d1) -> (d0, d1)>  // output
   ]
 }
```

## 案例2：非法的Broadcast（不是Permutation）

```text
// Producer: Broadcast操作
%producer = linalg.generic {
  indexing_maps = [
    affine_map<(d0) -> (d0)>,        // input: 1D
    affine_map<(d0) -> (d0)>         // output: 1D
  ],
  iterator_types = ["parallel"]
} ins(%A) outs(%init) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
} -> tensor<?xf32>

// Consumer: 将1D broadcast到2D
%consumer = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0)>,     // input (%producer) - broadcast!
    //         ^^^^^^^^^^^^^^^^^^^^
    //         这不是permutation！d1维度丢失了
    affine_map<(d0, d1) -> (d0, d1)>  // output: 2D
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%producer) outs(%init2) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
}
```

### 问题分析

假设producer的result map是 $ (d0) -> (d0)  $，但consumer访问它的方式是$  (d0, d1) -> (d0) $

### 尝试转换

1. $ producerResultIndexMap = (d0) -> (d0)  $  ✅ 这个是permutation
2. $ consumerArgMap = (d0, d1) -> (d0)  $ ❌ 这个不是permutation！ 但这个map在consumer，不是在producer result

### 实际问题

- Producer是1D空间：只有d0
- Consumer是2D空间：有d0和d1
- 维度不匹配！

检查会在这里失败：

```cpp
// ElementwiseOpFusion.cpp:169-171
AffineMap consumerIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
if (consumerIndexMap.getNumResults() != producer.getNumLoops())
  return false;  // 1 != 2，维度不匹配！
```

## 案例3：更复杂的非Permutation

```text
// Producer: 输出使用了仿射表达式
%producer = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,       // input
    affine_map<(d0, d1) -> (d0 + d1, d0)>   // output: 仿射表达式！
    //         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //         不是permutation！第一个结果是 d0+d1
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%A) outs(%init) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
}
```

### 问题

$ producerResultIndexMap = (d0, d1) -> (d0 + d1, d0) $

尝试计算逆映射：

$ invProducerResultIndexMap = inversePermutation((d0, d1) -> (d0 + d1, d0))                          = nullptr $ ❌ 失败！

### 原因

- 给定输出$ (r0, r1) $

- 需要找到 $ (d0, d1) $使得：    
  $$
  \begin{align}
  d0 + d1 &= r0 \\
  d0 &= r1 \\
  \end{align}
  $$

- 解得：$ d0 = r1, d1 = r0 - r1 $

虽然数学上可解，但这不是简单的维度重排！ inversePermutation() 只能处理维度的排列，不能处理仿射运算。

# 5. 数学上的必要性

## Permutation的关键性质

### 性质1：可逆性

若排列映射为：$ f: (d_0, d_1, \dots, d_n) \rightarrow (d_{\pi(0)}, d_{\pi(1)}, \dots, d_{\pi(n)})$

则存在逆映射：$f^{-1}: (r_0, r_1, \dots, r_n) \rightarrow (d_{\pi^{-1}(0)}, d_{\pi^{-1}(1)}, \dots, d_{\pi^{-1}(n)})$

### 性质2：一一对应

* 每个输入维度 → 恰好一个输出维度 

* 每个输出维度 ← 恰好一个输入维度

### 性质3：信息无损

* 维度数量不变：输入n维 → 输出n维 

* 没有维度丢失（如broadcast） 

* 没有维度合并（如d0+d1）

## 为什么其他映射不行？

### 1. Broadcast（维度丢失）

$ map = (d0, d1) -> (d0) $

问题：给定输出索引 r0，无法唯一确定 (d0, d1) 

- 可能是 (r0, 0)
- 可能是 (r0, 1) 
- 可能是 (r0, 任何值)

逆映射不存在！

### 2. Affine Expression（仿射运算）

$ map = (d0, d1) -> (d0 + d1, d0) $

虽然数学上可逆：  
$$
\begin{align}
d0 &= r1 \\
d1 &= r0 - r1 \\
\end{align}
$$


但这不是"permutation"，而是仿射变换！ inversePermutation() API 不支持这种情况。

# 6. 实现细节：inversePermutation

## API定义

```cpp
// 在 AffineMap.h 中
AffineMap inversePermutation(AffineMap map);

// 返回值：
// - 如果map是permutation，返回逆映射
// - 否则返回空的AffineMap
```

## 实现逻辑（简化版）

```cpp
AffineMap inversePermutation(AffineMap map) {
  unsigned numDims = map.getNumDims();
  unsigned numResults = map.getNumResults();

  // 检查1: 维度数必须相等
  if (numDims != numResults)
    return AffineMap();

  // 检查2: 每个result必须是单个dim表达式
  SmallVector<unsigned> permutation(numDims);
  for (unsigned i = 0; i < numResults; ++i) {
    auto expr = map.getResult(i);

    // 必须是AffineDimExpr，不能是加法、乘法等
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return AffineMap();  // 不是单个维度

    permutation[i] = dimExpr.getPosition();
  }

  // 检查3: permutation必须是双射
  SmallVector<bool> seen(numDims, false);
  for (unsigned p : permutation) {
    if (seen[p])
      return AffineMap();  // 重复使用了某个维度
    seen[p] = true;
  }

  // 构造逆映射
  SmallVector<AffineExpr> invExprs;
  for (unsigned i = 0; i < numDims; ++i) {
    // 找到哪个位置映射到i
    for (unsigned j = 0; j < numDims; ++j) {
      if (permutation[j] == i) {
        invExprs.push_back(getAffineDimExpr(j, map.getContext()));
        break;
      }
    }
  }

  return AffineMap::get(numDims, 0, invExprs, map.getContext());
}
```

## 示例

输入：(d0, d1, d2) -> (d2, d0, d1)

permutation = [2, 0, 1]

逆映射构造： 

- 位置0在原来的位置1 → invExprs[0] = d1 
- 位置1在原来的位置2 → invExprs[1] = d2 
- 位置2在原来的位置0 → invExprs[2] = d0

输出：(d0, d1, d2) -> (d1, d2, d0)

# 7. 实际影响

## 限制了什么操作？

* 不能作为Producer融合的操作：

```text
// ❌ 1. Reduction（降维）
%sum = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0)>      // 丢失了d1维度
  ],
  iterator_types = ["parallel", "reduction"]
} -> tensor<?xf32>

// ❌ 2. Reshape/Collapse
%collapsed = tensor.collapse_shape %A [[0, 1]]
// 内部的indexing map不是permutation

// ❌ 3. Gather/Scatter（间接索引）
// 输出索引不是输入索引的简单重排
```

* 可以作为Producer融合的操作：

```text
// ✅ 1. Transpose
%transposed = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d1, d0)>  // permutation
  ]
}

// ✅ 2. Identity
%copy = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>  // permutation
  ]
}

// ✅ 3. 复杂的维度重排
%shuffled = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>  // permutation
  ]
}
```

# 8. 总结

## 为什么必须是Permutation？

| 原因            | 说明                      | 后果             |
| :-------------- | :------------------------ | :--------------- |
| 1. 逆映射可计算 | inversePermutation() 需要 | 无法转换坐标系   |
| 2. 维度一一对应 | 输入输出维度相同          | 保证信息完整     |
| 3. 无信息丢失   | 每个维度都被保留          | 可以恢复原始索引 |
| 4. 简单高效     | 不需要解方程              | 编译时可计算     |

## 核心公式

融合的IndexingMap计算：$ fusedMap = producerArgMap ∘ invProducerResultMap ∘ consumerArgMap $                        

其中$invProducerResultMap$必须可逆！

可逆的充要条件：$producerResultMap $是 permutation

## 代码检查的本质

```cpp
if (!producerResultIndexMap.isPermutation())
  return false;
```

如果producer的输出映射不是简单的维度重排，我们无法计算逆映射，因此无法将producer的输入转换到consumer的坐标系中，所以不能融合。

## 实际例子对比

✅ 可以融合 

Producer: (d0, d1) -> (d1, d0)  [transpose, 是permutation] 

Consumer使用它，可以计算出正确的indexing map

❌ 不能融合 Producer: (d0, d1) -> (d0)      [reduction, 不是permutation] 

Consumer使用它，无法计算逆映射

这个限制确保了融合算法的正确性和可实现性！
