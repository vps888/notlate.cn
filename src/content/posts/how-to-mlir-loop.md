---
title: "如何充分发挥MLIR中Loop的优化特性？"
description: "本文通过一个完整的矩阵乘法示例，讲解MLIR中三个核心Loop优化技术： 1. Loop carried Dependency 分析 识别循环间依赖 2. Loop Unrolling 循环展开 3. Affine Loop LICM 循环不变代码外提 场景：矩阵乘法优化 考虑这个经典的热点场…"
slug: "how-to-mlir-loop"
legacyId: 19539174
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19539174"
pubDate: 2026-01-27
category: "AI 编译器"
tags: ["AI 编译器","MLIR"]
featured: true
---

本文通过一个完整的矩阵乘法示例，讲解MLIR中三个核心Loop优化技术：

1. **Loop-carried Dependency 分析** - 识别循环间依赖
2. **Loop Unrolling** - 循环展开
3. **Affine Loop LICM** - 循环不变代码外提

---

## 场景：矩阵乘法优化

考虑这个经典的热点场景：C = A × B，其中A、B、C都是N×N矩阵。

### 初始MLIR代码

```cpp
#map0 = affine_map<(d0, d1) -> (d0 * N + d1)>

func.func @matmul(%A: memref<NxNxf32>, %B: memref<NxNxf32>,
                 %C: memref<NxNxf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32

  // 三层嵌套循环
  affine.for %i = 0 to N {
    affine.for %j = 0 to N {
      // 初始化累加器 - 这是循环不变代码！
      %sum_init = arith.constant 0.0 : f32

      affine.for %k = 0 to N {
        %a = affine.load %A[%i, %k] : memref<NxNxf32>
        %b = affine.load %B[%k, %j] : memref<NxNxf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %sum_init, %prod : f32
        %sum_init = %sum  // 循环携带依赖！
      }
      affine.store %sum_init, %C[%i, %j] : memref<NxNxf32>
    }
  }
  return
}
```

---

## 1. Loop-carried Dependency（循环携带依赖）

### 什么是循环携带依赖？

当循环的某次迭代**依赖**于前一次迭代的结果时，就存在循环携带依赖。

**在上面的代码中：**

```cpp
%sum_init = arith.constant 0.0 : f32  // 初始化
affine.for %k = 0 to N {
  %sum = arith.addf %sum_init, %prod : f32
  %sum_init = %sum  // 第k次迭代使用第k-1次的结果
}
```

这里 `%sum_init` 是一个**循环携带的迭代参数**，每次迭代都基于上一次的结果。

### MLIR如何检测循环携带依赖？

**核心代码位置：** `mlir/lib/Dialect/Affine/Analysis/AffineAnalysis.cpp:611-660`

```cpp
// 检查两个内存访问之间是否存在依赖
DependenceResult mlir::affine::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth,
    FlatAffineValueConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents,
    bool allowRAR) {

  // 1. 检查是否访问同一memref
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;

  // 2. 创建访问关系
  // 将访问模式转换为Presburger关系
  IntegerRelation srcRel, dstRel;
  srcAccess.getAccessRelation(srcRel);
  dstAccess.getAccessRelation(dstRel);

  // 3. 计算依赖关系
  // 通过求交集来判断是否存在依赖
  // ... (使用Presburger库进行精确的数学分析)
}
```

**判断循环是否可并行化：**

```cpp
bool mlir::affine::isLoopParallel(AffineForOp forOp) {
  // 1. 检查SSA循环携带依赖（iter_args）
  if (forOp.getNumIterOperands() > 0) {
    // 有iter_args说明有循环携带依赖
    // 但如果是reduction操作，仍然可以并行
    if (!isSupportedReduction(forOp))
      return false;
  }

  // 2. 检查内存依赖
  return isLoopMemoryParallel(forOp);
}
```

### 对矩阵乘法的影响

**问题：** `%k` 循环有循环携带依赖（累加操作）

**解决方案：** 使用 `iter_args` 显式声明reduction

```cpp
affine.for %k = 0 to N iter_args(%sum = %sum_init) -> f32 {
  %a = affine.load %A[%i, %k] : memref<NxNxf32>
  %b = affine.load %B[%k, %j] : memref<NxNxf32>
  %prod = arith.mulf %a, %b : f32
  %next_sum = arith.addf %sum, %prod : f32
  affine.yield %next_sum : f32  // 传递给下一次迭代
}
```

这样MLIR可以识别这是一个**可并行的reduction**。

---

## 高级应用：Reduce-to-Elementwise Fusion

### 场景描述

在实际应用中，经常遇到这种模式：

1. **Reduction循环**：沿某个维度累加（如行求和）
2. **Elementwise循环**：对reduction结果逐元素操作（如乘以系数）

**问题：** 两个循环分离导致中间结果写入内存，破坏局部性。

### 示例：行求和 + 标量乘法

```cpp
// 原始代码：两个分离的循环
func.func @reduce_then_elementwise(%A: memref<10x10xf32>,
                                   %B: memref<10xf32>,
                                   %C: memref<10xf32>) {
  %cf7 = arith.constant 7.0 : f32

  // 循环1：Reduction - 每行求和到B[i]
  affine.for %i = 0 to 10 {
    %sum_init = arith.constant 0.0 : f32
    affine.for %j = 0 to 10 {
      %a = affine.load %A[%i, %j] : memref<10x10xf32>
      %sum = arith.addf %sum_init, %a : f32
      %sum_init = %sum  // 循环携带依赖
    }
    affine.store %sum_init, %B[%i] : memref<10xf32>
  }

  // 循环2：Elementwise - B[i] * 7.0
  affine.for %i = 0 to 10 {
    %b = affine.load %B[%i] : memref<10xf32>
    %c = arith.mulf %b, %cf7 : f32
    affine.store %c, %C[%i] : memref<10xf32>
  }

  return
}
```

### 为什么可以融合？

**依赖分析：**

```cpp
// MLIR检查两个循环之间的依赖关系
DependenceResult result = checkMemrefAccessDependence(
    srcAccess,  // affine.store %sum_init, %B[%i]
    dstAccess,  // affine.load %b, %B[%i]
    loopDepth
);

// 结果：存在RAW（Read-After-Write）依赖
// 但这是一个producer-consumer关系，可以融合！
```

**关键条件：**

1. Producer循环只写入 `%B`（single-writer）
2. Consumer循环只读取 `%B`（single-reader）
3. 访问模式兼容（都是affine访问）

### 融合后的代码

```bash
mlir-opt -affine-loop-fusion input.mlir
```

```cpp
func.func @reduce_then_elementwise_fused(%A: memref<10x10xf32>,
                                         %C: memref<10xf32>) {
  %cf7 = arith.constant 7.0 : f32

  // 融合后的单层循环
  affine.for %i = 0 to 10 {
    // 私有临时变量（无需写回内存！）
    %sum_init = arith.constant 0.0 : f32

    // Reduction部分
    affine.for %j = 0 to 10 {
      %a = affine.load %A[%i, %j] : memref<10x10xf32>
      %sum = arith.addf %sum_init, %a : f32
      %sum_init = %sum
    }

    // Elementwise部分（直接使用寄存器中的值）
    %c = arith.mulf %sum_init, %cf7 : f32
    affine.store %c, %C[%i] : memref<10xf32>
  }

  return
}
```

### MLIR融合实现细节

**核心代码位置：** `mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp`

**融合策略判断：**

```cpp
// 检查是否满足融合条件
static bool canFuseLoops(AffineForOp srcLoop, AffineForOp dstLoop,
                        const MemRefDependenceGraph &mdg) {
  // 1. 检查memref依赖
  // srcLoop写入的memref，dstLoop是否只读取？
  if (!hasSingleWriterMemRef(srcLoop, dstLoop))
    return false;

  // 2. 检查访问模式兼容性
  // 能否找到fusion slice（融合切片）？
  ComputationSliceState slice;
  if (failed(getComputationSliceState(srcLoop, dstLoop, &slice)))
    return false;

  // 3. 检查reduction的循环携带依赖
  // 融合后的循环是否仍然保持正确的依赖关系？
  if (!validateDependencesAfterFusion(srcLoop, dstLoop, slice))
    return false;

  return true;
}
```

**Fusion Slice计算：**

```cpp
// 确定如何将src循环插入到dst循环中
// 例如：src是二维循环(i,j)，dst是一维循环(i)
// fusion slice告诉我们只需要融合src的内层循环(j)

struct ComputationSliceState {
  // 要融合的循环迭代次数
  SmallVector<uint64_t> loopTripCounts;

  // 循环界限（可能是表达式）
  SmallVector<Value> lbs, ubs;

  // 是否是maximal fusion（完全融合）
  std::optional<bool> isMaximal;
};
```

### 性能影响分析

**融合前：**

```
内存访问：
- 写入B[0..9]: 10次store
- 读取B[0..9]: 10次load
- 总共：20次内存访问
```

**融合后：**

```
内存访问：
- B被完全消除（使用寄存器）
- 总共：0次额外的内存访问
```

**实际性能提升：**

- 减少50%的内存访问
- 提高cache命中率
- 暴露更多指令级并行机会

### 更复杂的例子：嵌套Reduction融合

```cpp
// 三阶段：Reduce -> Reduce -> Elementwise
func.func @multi_stage_fusion(%A: memref<10x10x10xf32>,
                               %D: memref<10xf32>) {
  %cf2 = arith.constant 2.0 : f32

  // 阶段1: A[i,j,k] -> B[i,j]  (沿k维reduction)
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      %sum = arith.constant 0.0 : f32
      affine.for %k = 0 to 10 {
        %a = affine.load %A[%i, %j, %k]
        %sum = arith.addf %sum, %a
      }
      affine.store %sum, %B[%i, %j]
    }
  }

  // 阶段2: B[i,j] -> C[i]  (沿j维reduction)
  affine.for %i = 0 to 10 {
    %sum = arith.constant 0.0 : f32
    affine.for %j = 0 to 10 {
      %b = affine.load %B[%i, %j]
      %sum = arith.addf %sum, %b
    }
    affine.store %sum, %C[%i]
  }

  // 阶段3: C[i] * 2.0 -> D[i]  (elementwise)
  affine.for %i = 0 to 10 {
    %c = affine.load %C[%i]
    %d = arith.mulf %c, %cf2
    affine.store %d, %D[%i]
  }
}
```

**完全融合后：**

```cpp
func.func @multi_stage_fused(%A: memref<10x10x10xf32>,
                             %D: memref<10xf32>) {
  %cf2 = arith.constant 2.0 : f32

  affine.for %i = 0 to 10 {
    // 所有中间变量都在寄存器中！
    %c_sum = arith.constant 0.0 : f32

    affine.for %j = 0 to 10 {
      %b_sum = arith.constant 0.0 : f32

      affine.for %k = 0 to 10 {
        %a = affine.load %A[%i, %j, %k]
        %b_sum = arith.addf %b_sum, %a  // 阶段1的累加
      }
      // %b_sum完成，但直接用于外层累加

      %c_sum = arith.addf %c_sum, %b_sum  // 阶段2的累加
    }
    // %c_sum完成，直接用于elementwise

    %d = arith.mulf %c_sum, %cf2  // 阶段3
    affine.store %d, %D[%i]  // 只写最终结果
  }
}
```

### 依赖分析的关键作用

在融合过程中，依赖分析确保：

1. **正确的迭代顺序**：

   ```cpp
   // 融合前：内层循环先完成
   for k:
       B[i] += A[i,k]
   // 融合后：保持相同的顺序
   for i:
       for k:
           temp += A[i,k]  // temp对应B[i]
   ```

2. **循环携带依赖的保持**：

   ```cpp
   // 检查融合后的reduction是否正确
   // 每个temp变量的累加顺序是否保持一致？
   bool validateReductionAfterFusion(
       const ComputationSliceState &slice) {
     // 确保reduction的迭代顺序不变
     // 确保没有race condition
   }
   ```

3. **内存访问的合法性**：

   ```cpp
   // 融合后的访问模式是否仍然正确？
   // 是否引入了新的依赖冲突？
   checkMemrefAccessDependence(...);
   ```

---

## 2. Loop Unrolling（循环展开）

### 什么是循环展开？

将循环体复制多次，减少循环控制开销，增加指令级并行机会。

**展开前（假设N=4）：**

```cpp
affine.for %k = 0 to 4 {
  %a = affine.load %A[%i, %k]
  %b = affine.load %B[%k, %j]
  %prod = arith.mulf %a, %b
  %sum = arith.addf %sum, %prod
}
```

**展开后（unroll factor = 4）：**

```cpp
// 第一次迭代
%a0 = affine.load %A[%i, 0]
%b0 = affine.load %B[0, %j]
%prod0 = arith.mulf %a0, %b0
%sum0 = arith.addf %sum, %prod0

// 第二次迭代
%a1 = affine.load %A[%i, 1]
%b1 = affine.load %B[1, %j]
%prod1 = arith.mulf %a1, %b1
%sum1 = arith.addf %sum0, %prod1

// 第三次迭代
%a2 = affine.load %A[%i, 2]
%b2 = affine.load %B[2, %j]
%prod2 = arith.mulf %a2, %b2
%sum2 = arith.addf %sum1, %prod2

// 第四次迭代
%a3 = affine.load %A[%i, 3]
%b3 = affine.load %B[3, %j]
%prod3 = arith.mulf %a3, %b3
%sum3 = arith.addf %sum2, %prod3
```

### MLIR如何实现循环展开？

**核心代码位置：** `mlir/lib/Dialect/Affine/Transforms/LoopUnroll.cpp`

**Pass入口：**

```cpp
struct LoopUnroll : public affine::impl::AffineLoopUnrollBase<LoopUnroll> {
  void runOnOperation() override {
    // 收集最内层循环
    SmallVector<AffineForOp, 4> loops;
    gatherInnermostLoops(func, loops);

    // 对每个循环应用展开
    for (auto forOp : loops) {
      if (unrollFull)
        loopUnrollFull(forOp);  // 完全展开
      else
        loopUnrollByFactor(forOp, unrollFactor);  // 按因子展开
    }
  }
};
```

**关键函数：** `loopUnrollByFactor`（位于 `mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp`）

```cpp
LogicalResult loopUnrollByFactor(AffineForOp forOp, uint64_t unrollFactor,
                                 const LoopUnrollOptions &options) {
  // 1. 计算展开次数
  std::optional<uint64_t> tripCount = getConstantTripCount(forOp);

  // 2. 生成cleanup loop处理余数
  // 例如：tripCount=10, unrollFactor=4
  // 主循环：0-8 (step 4)，cleanup循环：8-10

  // 3. 复制循环体
  OpBuilder b(forOp.getBody());
  for (unsigned i = 0; i < unrollFactor; ++i) {
    // 克隆操作
    for (Operation &op : forOp.getBody()->without_terminator()) {
      Operation *clonedOp = b.clone(op);

      // 替换induction variable
      // 如果原IV是 %k，替换为 %k + i
      replaceIV(clonedOp, forOp.getInductionVar(), i);
    }
  }

  // 4. 处理iter_args（reduction）
  if (forOp.getNumIterOperands() > 0) {
    // 链接各个展开块的reduction结果
    chainReductions(unrolledOps, iterArgs);
  }
}
```

### 展开策略

1. **完全展开（Full Unroll）：** 当trip count很小且已知

   ```bash
   mlir-opt matmul.mlir -affine-loop-unroll{unroll-full=true}
   ```

2. **按因子展开（Unroll by Factor）：** 指定展开因子

   ```bash
   mlir-opt matmul.mlir -affine-loop-unroll{unroll-factor=4}
   ```

3. **带有cleanup循环：** 处理非整除情况

   ```cpp
   // tripCount=10, unrollFactor=4
   affine.for %k = 0 to 8 step 4 {  // 主循环
     // 展开为4次迭代
   }
   affine.for %k = 8 to 10 {  // cleanup循环
     // 剩余2次迭代
   }
   ```

### 性能影响

**优点：**

- 减少分支预测失败
- 增加指令级并行（ILP）
- 暴露更多优化机会

**缺点：**

- 代码膨胀
- 可能降低寄存器利用率

---

## 3. Affine Loop LICM（循环不变代码外提）

### 什么是循环不变代码？

在循环内部**每次迭代都执行相同结果**的代码。

**在矩阵乘法中：**

```cpp
affine.for %i = 0 to N {
  affine.for %j = 0 to N {
    // 这些常量对于整个嵌套循环是不变的
    %sum_init = arith.constant 0.0 : f32  // 不变！

    affine.for %k = 0 to N {
      // ...
    }
  }
}
```

### MLIR如何实现LICM？

**核心代码位置：** `mlir/lib/Dialect/Affine/Transforms/AffineLoopInvariantCodeMotion.cpp:63-135`

**判断不变性的核心逻辑：**

```cpp
static bool isOpLoopInvariant(Operation &op, AffineForOp loop,
                              SmallPtrSetImpl<Operation *> &opsWithUsers,
                              SmallPtrSetImpl<Operation *> &opsToHoist) {
  Value iv = loop.getInductionVar();

  // 1. 检查操作类型
  if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
    // 递归检查if分支内的所有操作
    return checkInvarianceOfNestedIfOps(ifOp, loop, ...);
  }

  // 2. 检查副作用
  if (!isMemoryEffectFree(&op) &&
      !isa<AffineReadOpInterface, AffineWriteOpInterface>(&op)) {
    return false;  // 有副作用的操作不能外提
  }

  // 3. 特殊处理affine load/store
  if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
    // 检查是否存在依赖
    // 如果有其他store操作写入同一位置，则不能外提
    if (hasConflictingStore(op, loop))
      return false;
  }

  // 4. 检查操作数
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    // 如果操作数是循环IV，则不是不变代码
    if (iv == op.getOperand(i))
      return false;

    // 如果操作数是iter_arg，则不是不变代码
    if (llvm::is_contained(loop.getRegionIterArgs(), op.getOperand(i)))
      return false;
  }

  opsToHoist.insert(&op);
  return true;
}
```

**Pass主逻辑：**

```cpp
void LoopInvariantCodeMotion::runOnAffineForOp(AffineForOp forOp) {
  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;

  // 遍历循环体中的所有操作
  for (Operation &op : *forOp.getBody()) {
    if (!isa<AffineYieldOp>(op)) {
      if (isOpLoopInvariant(op, forOp, opsWithUsers, opsToHoist)) {
        opsToMove.push_back(&op);
      }
    }
  }

  // 将不变代码移动到循环之前
  OpBuilder b(forOp.getOperation());
  for (auto *op : opsToMove) {
    op->moveBefore(forOp);  // 移动到循环前
  }
}
```

**自底向上遍历：** 先处理内层循环，再处理外层

```cpp
void LoopInvariantCodeMotion::runOnOperation() {
  // 从内层到外层处理
  getOperation().walk([&](AffineForOp op) {
    runOnAffineForOp(op);
  });
}
```

### LICM效果示例

**优化前：**

```cpp
affine.for %i = 0 to N {
  affine.for %j = 0 to N {
    %c0 = arith.constant 0.0 : f32  // 执行N×N次！
    affine.for %k = 0 to N {
      // ...
    }
  }
}
```

**优化后（-affine-loop-invariant-code-motion）：**

```cpp
%c0_hoisted = arith.constant 0.0 : f32  // 只执行1次！

affine.for %i = 0 to N {
  affine.for %j = 0 to N {
    affine.for %k = 0 to N {
      // 使用 %c0_hoisted
    }
  }
}
```

---

## 实际测试用例参考

MLIR代码库中的完整测试示例：

1. **LICM测试：** `mlir/test/Dialect/Affine/affine-loop-invariant-code-motion.mlir`
2. **Unroll测试：** `mlir/test/Dialect/Affine/unroll.mlir`
3. **依赖分析测试：** `mlir/test/Dialect/Affine/loop-fusion-dependence-check.mlir`

---

## 性能影响总结

| 优化技术              | 适用场景     | 性能提升     | 代码膨胀 | 寄存器压力 |
| --------------------- | ------------ | ------------ | -------- | ---------- |
| Loop-carried Analysis | 所有并行化   | 识别并行机会 | 无       | 无         |
| Loop Unrolling        | 小trip count | 20-50%       | 高       | 高         |
| LICM                  | 有不变代码   | 10-30%       | 无       | 低         |

**组合效果：** 对于N=64的矩阵乘法，这些优化组合可以带来**2-3倍**的性能提升。

---

## 扩展阅读：识别可并行Reduction有什么用处？

在前面的内容中，我们看到MLIR可以通过循环携带依赖分析识别出"可并行的reduction"。读者可能会问：**这有什么实际用处？**

答案是：这是**高性能优化的基础**。一旦识别出可并行的reduction，MLIR可以应用一系列强大的优化转换。

### 1. 并行化执行

最直接的用处是将串行循环转换为并行执行：

**原始串行代码：**

```cpp
affine.for %k = 0 to N iter_args(%sum = %sum_init) -> f32 {
  %a = affine.load %A[%i, %k]
  %b = affine.load %B[%k, %j]
  %prod = arith.mulf %a, %b
  %next_sum = arith.addf %sum, %prod
  affine.yield %next_sum : f32
}
```

**转换为 `affine.parallel`：**

```cpp
affine.parallel (%k) = (0) to (N) reduce (%sum_init = arith.addf) {
  %a = affine.load %A[%i, %k]
  %b = affine.load %B[%k, %j]
  %prod = arith.mulf %a, %b
  affine.yield %prod : f32
}
```

### 2. SIMD向量化

识别为reduction后，可以使用SIMD指令一次处理多个元素：

**标量版本：**

```cpp
affine.for %k = 0 to N {
  %v = affine.load %A[%k]
  %sum = arith.addf %sum, %v
}
```

**向量化后（AVX2，一次处理8个f32）：**

```cpp
affine.for %k = 0 to N step 8 {
  %vec = vector.load %A[%k] : vector<8xf32>
  %vec_sum = vector.add %vec_sum, %vec : vector<8xf32>
}
%final = vector.reduction <add>, %vec_sum : vector<8xf32> into f32
```

### 3. GPU加速

对于大规模计算，可以生成GPU kernel：

```cpp
gpu.launch blocks(%i, %j) in (%grid = <32, 32>)
              threads(%k_thread) in (%block = <32>) {
  // 每个线程处理部分k维度
  %private_sum = arith.constant 0.0 : f32

  affine.for %k = %k_thread to N step 1024 {
    %a = affine.load %A[%i, %k]
    %b = affine.load %B[%k, %j]
    %prod = arith.mulf %a, %b
    %private_sum = arith.addf %private_sum, %prod
  }

  // 跨线程归约
  %final_sum = gpu.all_reduce %private_sum (arith.addf)
  affine.store %final_sum, %C[%i, %j]
}
```

### 4. 循环交换与融合

知道是reduction后，可以安全地进行循环交换以改善局部性：

**原始顺序：** i → j → k（k是最内层reduction）
**交换为：** k → i → j（将reduction外提，改善cache利用率）

### 5. 软件流水线

重叠执行多个迭代阶段，隐藏内存延迟：

```cpp
// 同时执行：
// - 当前迭代的计算
// - 下一次迭代的内存加载
// - 下下次迭代的预取

affine.for %k = 0 to N-2 {
  %a_curr = affine.load %A[%k]      // 当前
  %a_next = affine.load %A[%k+1]    // 预取
  %sum_curr = arith.addf %sum, %a_curr
  %sum_next = arith.addf %sum_curr, %a_next
}
```

### 6. 硬件原子操作

对于共享内存的reduction，使用无锁原子操作：

```cpp
// 不需要锁！
affine.parallel (%k) = (0) to (N) {
  %v = compute(%k)
  %old = atomicrmw @global_sum, %v, arith.addf
}
```

### 7. 分布式计算

在分布式系统中，将reduction分布到多个节点：

```cpp
// 节点0: k in [0, N/4)
// 节点1: k in [N/4, N/2)
// 节点2: k in [N/2, 3N/4)
// 节点3: k in [3N/4, N)
// 最后通过AllReduce合并
```

---

### 完整优化链示例

展示一个行求和函数如何逐步优化：

```cpp
// 阶段0：原始代码
func.func @row_sum(%A: memref<1024x1024xf32>, %B: memref<1024xf32>) {
  affine.for %i = 0 to 1024 {
    %sum_init = arith.constant 0.0 : f32
    affine.for %j = 0 to 1024 iter_args(%sum = %sum_init) -> f32 {
      %a = affine.load %A[%i, %j]
      %sum_next = arith.addf %sum, %a
      affine.yield %sum_next : f32
    }
    affine.store %sum, %B[%i]
  }
}

// 阶段1：向量化（-affine-vectorize）
affine.for %i = 0 to 1024 {
  %vec_sum = vector.splat 0.0 : vector<8xf32>
  affine.for %j = 0 to 1024 step 8 {
    %vec = vector.load %A[%i, %j] : vector<8xf32>
    %vec_sum = vector.add %vec_sum, %vec : vector<8xf32>
  }
  %scalar_sum = vector.reduction <add>, %vec_sum
  affine.store %scalar_sum, %B[%i]
}

// 阶段2：并行化（-affine-parallelize）
affine.parallel (%i) = (0) to (1024) {
  %vec_sum = vector.splat 0.0 : vector<8xf32>
  affine.for %j = 0 to 1024 step 8 {
    %vec = vector.load %A[%i, %j] : vector<8xf32>
    %vec_sum = vector.add %vec_sum, %vec : vector<8xf32>
  }
  %scalar_sum = vector.reduction <add>, %vec_sum
  affine.store %scalar_sum, %B[%i]
}

// 阶段3：GPU offload（-convert-affine-to-gpu）
gpu.launch blocks(%i_grid) in (%grid = <32>)
              threads(%j_thread) in (%block = <32>) {
  // 每个block处理一行，使用warp shuffle优化reduction
}
```

---

### 性能提升对比

| 优化技术              | 性能提升 | 适用硬件    |
| --------------------- | -------- | ----------- |
| 并行化（8核）         | 6-8x     | 多核CPU     |
| SIMD向量化（AVX-512） | 8-16x    | CPU向量单元 |
| GPU加速（RTX 4090）   | 50-100x  | GPU         |
| 分布式（4节点）       | 3-4x     | 集群        |

**组合效果：** 在合适的硬件上，可并行reduction的识别可以带来**上百倍**的性能提升。

---

### 关键要点

> 如果MLIR不能识别这是一个可并行的reduction，所有这些优化都无法应用，代码只能串行执行！

这就是为什么循环携带依赖分析如此重要——它是高性能优化的**入场券**。

---

### 相关MLIR Pass

- `-affine-parallelize`：转换为affine.parallel
- `-affine-vectorize`：生成向量代码
- `-convert-affine-to-gpu`：生成GPU kernel
- `-affine-loop-fusion`：与其他循环融合
- `-affine-loop-tile`：应用分块优化
