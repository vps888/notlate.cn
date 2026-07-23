---
title: "【MLIR】Affine 方言深入研究"
description: "本文档基于 Claude Code + GLM4.7 (https://zhetengxia.com/) + CodeReaderSkills (https://zhetengxia.com/)完成。 1. 快速概览 1.1 代码统计 目录结构: 总计: 约 14,332 行 C++ 代码（不…"
slug: "mliraffine-dialect-deep-dive"
legacyId: 19612954
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19612954"
pubDate: 2026-02-13
updatedDate: 2026-03-13
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Affine"]
featured: true
---

本文档基于[Claude Code + GLM4.7](https://zhetengxia.com/) + [CodeReaderSkills](https://zhetengxia.com/)完成。

## 1. 快速概览

### 1.1 代码统计

**目录结构:**

```
mlir/lib/Dialect/Affine/
├── IR/
│   ├── AffineOps.cpp (5523 行)          # 核心操作实现
│   ├── AffineValueMap.cpp              # Affine 值映射
│   └── ValueBoundsOpInterfaceImpl.cpp   # 值边界接口实现
├── Analysis/
│   ├── AffineAnalysis.cpp (729 行)      # 依赖分析、并行性检测
│   ├── AffineStructures.cpp             # 多面体结构
│   ├── LoopAnalysis.cpp                # 循环分析
│   └── Utils.cpp                       # 工具函数
├── Transforms/
│   ├── LoopTiling.cpp (222 行)         # 循环分块
│   ├── LoopFusion.cpp (1594 行)        # 循环融合
│   ├── LoopUnroll.cpp (155 行)         # 循环展开
│   ├── AffineDataCopyGeneration.cpp    # 数据拷贝优化
│   └── ...其他变换
└── Utils/
    ├── LoopUtils.cpp                  # 循环工具
    └── LoopFusionUtils.cpp            # 融合工具

mlir/include/mlir/Dialect/Affine/
├── IR/
│   ├── AffineOps.h (563 行)           # 操作定义头文件
│   ├── AffineOps.td (1268 行)         # TableGen 操作定义
│   └── AffineValueMap.h              # 值映射头文件
├── Analysis/
│   ├── AffineAnalysis.h             # 分析接口
│   └── AffineStructures.h           # 多面体结构接口
└── Transforms/
    └── Transforms.h                 # 变换接口
```

**总计:** 约 14,332 行 C++ 代码（不含注释和空行）

### 1.2 核心依赖

```
Affine Dialect
    ├── IR 基础设施
    │   ├── AffineMap/IntegerSet     # 仿射映射和整数集
    │   └── Value/Operation          # MLIR 核心概念
    ├── Arith Dialect                # 算术运算
    ├── MemRef Dialect               # 内存引用
    ├── Presburger 库                # 多面体分析
    │   ├── IntegerRelation          # 整数关系
    │   ├── Matrix                   # 矩阵运算
    │   └── Polyhedron               # 多面体
    └── 通用分析
        ├── DataFlow                 # 数据流分析
        └── Dominance                # 支配关系
```

### 1.3 设计目标

Affine 方言是 MLIR 中用于**多面体编译**的核心方言，主要目标包括：

1. **精确的依赖分析**: 通过仿射约束表示循环边界和内存访问
2. **程序变换**: 支持循环分块、融合、重排等高级优化
3. **并行化检测**: 自动识别可并行化的循环
4. **内存优化**: 数据局部性分析和缓存优化

---

## 2. 背景与动机

### 2.1 为什么 MLIR 需要 Affine 方言

**WHY 1: 传统循环的局限性**

```text
// SCF (Structured Control Flow) 方言 - 灵活但难以分析
scf.for %i = 0 to %n step 1 {
  %idx = arith.addi %i, %offset : index  // ← 复杂的索引计算
  %val = memref.load %A[%idx] : memref<?xf32>
  // 无法静态分析访问模式
}
```

**WHY 2: Affine 提供精确的数学表示**

```text
// Affine 方言 - 约束明确，易于分析
affine.for %i = 0 to %n {
  %idx = affine.apply affine_map<(d0) -> (d0 + 5)> (%i)
  %val = affine.load %A[%idx] : memref<100xf32>
  // 访问模式: A[i+5]，可以精确分析依赖关系
}
```

**WHY 3: 多面体编译的数学基础**

多面体模型将程序表示为：

- **迭代空间**: 由不等式定义的多面体
- **访问映射**: 从迭代空间到数据空间的仿射映射
- **依赖关系**: 可以通过整数规划精确计算

### 2.2 与其他方言的对比

| 特性       | Affine     | SCF         | Linalg       |
| ---------- | ---------- | ----------- | ------------ |
| 循环边界   | 仿射表达式 | 任意 SSA 值 | 任意 SSA 值  |
| 内存访问   | 仿射索引   | 任意索引    | 语义化操作   |
| 并行性分析 | 自动精确   | 需手动标注  | 部分自动     |
| 优化空间   | 多面体变换 | 通用优化    | 高级算子融合 |
| 适用场景   | 规则计算   | 通用控制流  | 张量计算     |

### 2.3 多面体编译理论基础

**整数集合和关系:**

```
迭代空间表示: {(i, j) | 0 <= i < N, 0 <= j < M}
访问映射:    (i, j) -> (i + j, i - j)
依赖检查:    ∃(i,j),(i',j') : src(i,j) = dst(i',j') AND (i,j) < (i',j')
```

**Presburger 算术:**

- 一阶逻辑扩展了整数加法和比较
- 可判定性: 所有公式都有算法可以验证真值
- MLIR 使用 Presburger 库进行约束求解

---

## 3. 核心概念

### 3.1 仿射映射 (Affine Maps)

**定义:**

```
affine_map<(d0, d1)[s0, s1] -> (d0 + s0, d1 * 2 + s1)>
```

**WHY 分析:**

1. **WHY 分离维度和符号?**
   - 维度 (d0, d1): 循环归纳变量，随迭代变化
   - 符号 (s0, s1): 编译时常量或循环不变量
   - 分离后可以更精确地分析迭代空间

2. **WHY 限制为仿射表达式?**
   - 只允许: +, -, *, 常数 (无乘法中的两个变量)
   - 保证可逆性: 可以从访问地址反推迭代点
   - 保证可分析性: 依赖分析可以在多项式时间内完成

3. **WHY 支持多元仿射映射?**
   - 可以同时计算多个索引
   - 例如: `(i, j) -> (i+j, i-j)` 用于转置访问

**代码实现:**

```cpp
// AffineMap 是不可变的，在 Context 中唯一
class AffineMap {
  unsigned numDims;      // 维度数量
  unsigned numSymbols;   // 符号数量
  ArrayRef<AffineExpr> results;  // 结果表达式列表
};
```

### 3.2 整数集合 (Integer Sets)

**定义:**

```
affine_set<(d0, d1)[s0] : (d0 >= 0, d0 < s0, d1 = d0 + 1)>
```

**WHY 分析:**

1. **WHY 需要整数集合?**
   - 表示条件分支的执行条件
   - 表示循环的有效迭代空间
   - 表示数据依赖的约束

2. **WHY 使用等式和不等式?**
   - 等式 (=): 精确约束，如 `j = i + 1`
   - 不等式 (>=, <): 范围约束，如 `i >= 0`
   - 组合可以表示任意凸多面体

**affine.if 操作:**

```text
affine.if #set(%i, %j)[%N] {
  // then 分支: 当约束满足时执行
} else {
  // else 分支
}
```

### 3.3 Affine.for 操作

**语法:**

```
affine.for %i = max affine_map to min affine_map step constant {
  // 循环体
}
```

**WHY 分析:**

1. **WHY step 必须是正整数常量?**
   - 保证每次迭代前进固定的量
   - 简化依赖分析（无需考虑动态步长）
   - 便于计算精确的迭代次数

2. **WHY 支持多结果下界/上界?**
   - `max(a, b, c)`: 取多个下界的最大值
   - `min(x, y, z)`: 取多个上界的最小值
   - 可以表示复杂的边界条件

3. **WHY 需要 iter_args?**
   - 支持循环携带的归约变量
   - 可以返回最终值
   - 便于表示累加、最大值等操作

**代码分析 (AffineOps.td 第 121-337 行):**

```llvm
def AffineForOp : Affine_Op<"for", [...]> {
  let arguments = (ins
    Variadic<Index>:$lowerBoundOperands,
    Variadic<Index>:$upperBoundOperands,
    Variadic<AnyType>:$inits,
    AffineMapAttr:$lowerBoundMap,
    AffineMapAttr:$upperBoundMap,
    IndexAttr:$step
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
}
```

**关键方法:**

```cpp
class AffineForOp {
  BlockArgument getInductionVar();           // 获取循环变量
  AffineBound getLowerBound();               // 获取下界信息
  AffineBound getUpperBound();               // 获取上界信息
  int64_t getStepAsInt();                    // 获取步长
  bool hasConstantBounds();                  // 检查边界是否为常数
};
```

### 3.4 Affine.if 操作

**WHY 分析:**

1. **WHY 使用整数集合而不是布尔表达式?**
   - 整数集合可以编码多维约束
   - 便于与循环的多面体表示统一
   - 可以进行更精确的分析

2. **WHY 支持返回值?**
   - 可以在条件分支中计算值
   - 支持条件初始化
   - 便于表示边缘填充等操作

**示例:**

```llvm
#interior = affine_set<(i, j) : (i >= 1, j >= 1, 10 - i >= 0, 10 - j >= 0)>

%val = affine.if #interior (%i, %j) {
  %v = affine.load %A[%i - 1, %j - 1]
  affine.yield %v
} else {
  %v = arith.constant 0.0 : f32
  affine.yield %v
}
```

### 3.5 内存操作

**Affine.load:**

```text
%val = affine.load %A[%i + 3, %j * 2 + 1] : memref<100x100xf32>
```

**Affine.store:**

```text
affine.store %val, %A[%i, %j] : memref<100x100xf32>
```

**WHY 分析:**

1. **WHY 索引必须是仿射表达式?**
   - 保证可以静态计算访问模式
   - 依赖分析需要精确的访问函数
   - 便于应用多面体变换

2. **WHY 与标准 load/store 分离?**
   - Affine 版本可以进行更激进的优化
   - 可以证明内存访问的安全性
   - 便于向量化、分块等变换

**内存访问表示 (MemRefAccess 结构):**

```cpp
struct MemRefAccess {
  Value memref;                      // 被访问的内存
  Operation *opInst;                 // load/store 操作
  SmallVector<Value, 4> indices;     // 索引值

  // 获取访问关系: 迭代空间 -> 数据空间
  LogicalResult getAccessRelation(IntegerRelation &rel);
};
```

### 3.6 Affine.parallel 操作

**WHY 分析:**

1. **WHY 需要单独的 parallel 操作?**
   - 明确表示并行循环
   - 支持归约操作
   - 可以直接生成并行代码

2. **WHY 支持多维并行?**
   - 表示循环并行
   - 便于 GPU 等并行硬件映射
   - 可以优化线程块大小

**示例:**

```text
affine.parallel (%i, %j) = (0, 0) to (N, M) step (32, 32) {
  // 并行执行
} reduce ("addf", "mulf") -> (f32, f32)
```

### 3.7 Polyhedral 模型

**核心概念:**

1. **迭代空间 (Iteration Space)**

   ```
   循环: for i = 0 to N { for j = 0 to M { ... } }
   空间: {(i, j) ∈ Z² | 0 ≤ i < N, 0 ≤ j < M}
   ```

2. **访问映射 (Access Map)**

   ```
   访问: A[i+1][j*2]
   映射: (i, j) -> (i+1, j*2)
   ```

3. **依赖关系 (Dependence)**

   ```
   RAW 依赖: ∃(i,j),(i',j'): src(i,j) = dst(i',j')
   顺序约束: (i,j) < (i',j') (字典序)
   ```

**依赖分析算法 (AffineAnalysis.cpp 第 611-695 行):**

```cpp
DependenceResult checkMemrefAccessDependence(
    const MemRefAccess &srcAccess,
    const MemRefAccess &dstAccess,
    unsigned loopDepth) {

  // 1. 构建访问关系
  IntegerRelation srcRel, dstRel;
  srcAccess.getAccessRelation(srcRel);
  dstAccess.getAccessRelation(dstRel);

  // 2. 组合源访问和目标访问的逆
  dstRel.inverse();
  dstRel.mergeAndCompose(srcRel);

  // 3. 添加顺序约束
  addOrderingConstraints(srcDomain, dstDomain, loopDepth);

  // 4. 检查是否为空
  if (dependenceDomain.isEmpty())
    return NoDependence;

  return HasDependence;
}
```

---

## 4. 基础能力介绍

### 4.1 维度和符号验证 (isValidDim/isValidSymbol)

**WHY 分析 - 为什么要区分维度和符号?**

维度 (Dimension) 和符号 (Symbol) 是 Affine 方言的两大核心概念：

- **维度 (d0, d1, ...)**: 循环归纳变量，随迭代变化
- **符号 (s0, s1, ...)**: 编译时常量或循环不变量

**WHY 分离?**

1. 依赖分析需要区分迭代相关和无关的值
2. 符号可以在分析时当作常量处理
3. 简化迭代空间的数学表示

---

#### 4.1.1 isValidDim - 验证维度有效性

**源码位置:** `mlir/lib/Dialect/Affine/IR/AffineOps.cpp:291-344`

```cpp
// 步骤 1: 无 region 版本调用有 region 版本
bool mlir::affine::isValidDim(Value value) {
  // 场景 1: 类型必须是 index
  if (!value.getType().isIndex())
    return false;
    // WHY 非 index 类型不能作为仿射表达式的一部分
    // Affine 运算仅针对索引类型（表示数组索引、循环计数器等）

  // 场景 2: 值由操作定义，获取其 AffineScope 进行验证
  if (auto *defOp = value.getDefiningOp())
    return isValidDim(value, getAffineScope(defOp));
    // WHY 需要获取 AffineScope
    // 有效性是相对于特定区域的，不同区域规则不同

  // 场景 3: 值是块参数（如函数参数、循环归纳变量）
  // 步骤 2: 检查是否是仿射循环的归纳变量
  if (isAffineInductionVar(value))
    return true;
    // WHY 循环归纳变量是有效维度
    // 它们代表迭代空间中的坐标点

  // 步骤 3: 检查父操作是否具有 AffineScope 特征
  auto *parentOp = llvm::cast<BlockArgument>(value).getOwner()->getParentOp();
  return parentOp && parentOp->hasTrait<OpTrait::AffineScope>();
  // WHY AffineScope 内定义的值是符号
  // 这些值在该区域的所有仿射表达式中都是常量
}

// 步骤 4: 有 region 版本的完整验证逻辑
bool mlir::affine::isValidDim(Value value, Region *region) {
  // 场景 1: 类型检查
  if (!value.getType().isIndex())
    return false;
    // WHY 非索引类型直接拒绝
    // 保证类型安全的仿射表达式

  // 场景 2: 所有有效符号也是有效维度
  // WHY 符号是维度的超集
  if (isValidSymbol(value, region))
    return true;
    // 此时：value 是常量或循环不变量
    // 可以在仿射表达式中作为"参数"使用

  // 场景 3: 值由操作定义，递归检查
  auto *op = value.getDefiningOp();
  if (!op) {
    // 场景 3.1: 没有定义操作，必须是块参数
    // 步骤 5: 检查是否是仿射归纳变量
    return isAffineInductionVar(value);
    // WHY 块参数中没有定义操作时，只可能是归纳变量
    // 其他块参数（如 scf.for iter_args）不是有效维度
  }

  // 场景 4: 值由 affine.apply 操作定义
  if (auto applyOp = dyn_cast<AffineApplyOp>(op))
    return applyOp.isValidDim(region);
    // WHY 递归检查操作数
    // 如果所有输入都是有效维度，则输出也是
    // 例如: %idx = affine.apply (d0) -> (d0 * 2 + 1) (%i)
    //       若 %i 是有效维度，则 %idx 也是

  // 场景 5: 值由索引变换操作定义
  if (isa<AffineDelinearizeIndexOp, AffineLinearizeIndexOp>(op))
    return llvm::all_of(op->getOperands(),
                        [&](Value arg) { return ::isValidDim(arg, region); });
    // WHY 检查所有操作数
    // delinearize/linearize 是特殊的索引重排列操作

  // 场景 6: 值由 dim 操作定义（获取动态大小）
  if (auto dimOp = dyn_cast<ShapedDimOpInterface>(op))
    return isTopLevelValue(dimOp.getShapedValue());
    // WHY 顶层值是有效的
    // 动态大小如果在循环外定义，则是循环不变的符号

  // 场景 7: 不认识的操作，拒绝
  return false;
    // WHY 保守策略
    // 无法证明安全的情况下，认为无效
}
```

**执行流程示例 - 追踪具体值:**

```text
// 示例代码
func.func @example(%N: index, %M: index) {
  // 场景: %N, %M 是函数参数（块参数）
  // 验证 isValidDim(%N):
  //   步骤 1: %N 类型是 index ✓
  //   步骤 2: %N 没有定义操作（是块参数）
  //   步骤 3: isAffineInductionVar(%N) = false（不是归纳变量）
  //   步骤 4: parentOp 是 func.func，没有 AffineScope
  //   结果: isValidDim(%N) = false
  //   解释: 函数参数不是维度，但可能是符号

  affine.for %i = 0 to %N {
    // 场景: %i 是循环归纳变量
    // 验证 isValidDim(%i):
    //   步骤 1: %i 类型是 index ✓
    //   步骤 2: %i 是块参数（循环的归纳变量参数）
    //   步骤 3: isAffineInductionVar(%i) = true
    //   结果: isValidDim(%i) = true ✓

    %j = affine.apply affine_map<(d0) -> (d0 * 2)> (%i)
    // 场景: %j 由 affine.apply 定义
    // 验证 isValidDim(%j):
    //   步骤 1: %j 类型是 index ✓
    //   步骤 2: %j 有定义操作 (affine.apply)
    //   步骤 3: 进入场景 4，检查 applyOp.isValidDim(region)
    //         → 递归检查操作数 %i
    //         → isValidDim(%i) = true
    //   结果: isValidDim(%j) = true ✓

    %idx = affine.apply affine_map<(d0, s0) -> (d0 + s0)> (%i, %N)
    // 场景: %idx 使用了维度 %i 和符号 %N
    // 验证 isValidDim(%idx):
    //   步骤 1: %idx 类型是 index ✓
    //   步骤 2: %idx 有定义操作
    //   步骤 3: applyOp.isValidDim(region)
    //         → 检查操作数 %i（维度）→ true
    //         → 检查操作数 %N（符号）→ isValidSymbol(%N) = true
    //   结果: isValidDim(%idx) = true ✓
  }
}
```

**易错点标注:**

1. ⚠️ **函数参数不是维度**: `%N` 是符号而非维度
2. ⚠️ **递归深度限制**: 嵌套过深的 `affine.apply` 链可能导致性能问题
3. ⚠️ **跨 region 使用**: 在一个 region 有效的维度在另一个 region 可能无效

---

#### 4.1.2 isValidSymbol - 验证符号有效性

**源码位置:** `mlir/lib/Dialect/Affine/IR/AffineOps.cpp:405-429`

```cpp
bool mlir::affine::isValidSymbol(Value value) {
  // 步骤 1: 空值检查
  if (!value)
    return false;
    // WHY 防御性编程
    // 处理空指针情况

  // 步骤 2: 类型检查
  if (!value.getType().isIndex())
    return false;
    // WHY 符号也必须是 index 类型
    // 保证仿射表达式类型一致性

  // 步骤 3: 检查是否是顶层值
  if (isTopLevelValue(value))
    return true;
    // WHY 顶层值总是有效符号
    // 顶层值 = 在 AffineScope 区域顶层定义的值
    // 例如: 函数参数、常量

  // 步骤 4: 值由操作定义，获取 region 进行验证
  if (auto *defOp = value.getDefiningOp())
    return isValidSymbol(value, getAffineScope(defOp));
    // WHY 需要确定验证的上下文

  // 步骤 5: 值是块参数但没有定义操作
  return false;
    // WHY 无法验证
    // 例如: scf.for 的 iter_args 不是有效仿射符号
}
```

**执行流程示例:**

```text
func.func @symbol_example(%N: index, %M: index) -> index {
  // 场景: 验证函数参数
  // isValidSymbol(%N):
  //   步骤 1: value = %N, 非空 ✓
  //   步骤 2: 类型是 index ✓
  //   步骤 3: isTopLevelValue(%N)
  //         → %N 是函数的块参数
  //         → parentRegion = 函数体
  //         → %N 在该 region 定义
  //         → 返回 true
  //   结果: isValidSymbol(%N) = true ✓

  %c42 = arith.constant 42 : index
  // 场景: 常量总是有效符号
  // isValidSymbol(%c42):
  //   步骤 3: isTopLevelValue(%c42) = true
  //         → %c42 由 arith.constant 定义
  //         → 定义在函数体顶层
  //         → 返回 true
  //   结果: isValidSymbol(%c42) = true ✓

  affine.for %i = 0 to %N {
    // 场景: 归纳变量不是符号
    // isValidSymbol(%i):
    //   步骤 3: isTopLevelValue(%i) = false
    //         → %i 在循环体内，不在顶层
    //   步骤 4: %i 是块参数，没有定义操作
    //   步骤 5: 返回 false
    //   结果: isValidSymbol(%i) = false ✓
    //   解释: 归纳变量是维度，不是符号

    %dim = memref.dim %alloc, %i : memref<?xf32>
    // 场景: 循环依赖的 dim 操作
    // isValidSymbol(%dim):
    //   步骤 3: isTopLevelValue(%dim) = false
    //         → %dim 定义在循环内
    //   步骤 4: 有定义操作 (memref.dim)
    //         → 调用 isValidSymbol(%dim, region)
    //         → 检查 dim 操作的特殊规则
    //         → %i 不是顶层值
    //   结果: isValidSymbol(%dim) = false
    //   解释: 依赖循环变量的值不是符号
  }

  return %c42
}
```

**WHY 区分维度和符号的关键点:**

| 值类型                            | 是否有效维度 | 是否有效符号 | WHY                  |
| --------------------------------- | ------------ | ------------ | -------------------- |
| 循环归纳变量 `%i`                 | ✅ 是         | ❌ 否         | 随迭代变化，不是常量 |
| 函数参数 `%N`                     | ❌ 否         | ✅ 是         | 在整个函数内不变     |
| 常量 `%c42`                       | ❌ 否         | ✅ 是         | 编译时常量           |
| `affine.apply` 结果（输入是维度） | ✅ 是         | ❌ 否         | 继承输入的属性       |
| `affine.apply` 结果（输入是符号） | ❌ 否         | ✅ 是         | 继承输入的属性       |
| `memref.dim`（顶层）              | ❌ 否         | ✅ 是         | 循环不变             |
| `memref.dim`（循环依赖）          | ❌ 否         | ❌ 否         | 循环变化             |

---

### 4.2 依赖分析 - checkMemrefAccessDependence

**源码位置:** `mlir/lib/Dialect/Affine/Analysis/AffineAnalysis.cpp:611-695`

**WHY 依赖分析是 Affine 方言的核心优势?**

依赖分析回答：两个内存访问是否可能访问相同位置，以及访问的顺序关系。这是循环变换（重排、融合、并行化）的基础。

```cpp
DependenceResult mlir::affine::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess,    // 源访问（先执行）
    const MemRefAccess &dstAccess,    // 目标访问（后执行）
    unsigned loopDepth,               // 检查深度
    FlatAffineValueConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents,
    bool allowRAR) {

  // ========== 阶段 1: 前置检查 ==========

  // 步骤 1: 检查是否访问相同 memref
  // 场景 1: 不同的 memref
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;
    // WHY 不同 memref 不可能有依赖
    // 每个独立的 memref 有独立的地址空间
    // 例如: A[i] 和 B[j] 不会有冲突

  // 步骤 2: 检查是否有写操作
  // 场景 2: 两个都是读操作（RAR）
  if (!allowRAR && !isa<AffineWriteOpInterface>(srcAccess.opInst) &&
      !isa<AffineWriteOpInterface>(dstAccess.opInst))
    return DependenceResult::NoDependence;
    // WHY 读-读依赖不影响变换
    // 只需要关心 RAW, WAR, WAW

  // 步骤 3: 检查分析范围
  // 场景 3: 不同的 affine scope
  if (getAffineAnalysisScope(srcAccess.opInst) !=
      getAffineAnalysisScope(dstAccess.opInst))
    return DependenceResult::Failure;
    // WHY 无法跨 scope 分析
    // 不同 scope 的循环结构没有已知的关联

  // 步骤 4: 检查公共块
  if (!getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst))
    return DependenceResult::Failure;
    // WHY 需要在同一控制流
    // 否则无法确定执行顺序

  // ========== 阶段 2: 构建访问关系 ==========

  // 步骤 5: 创建访问关系
  PresburgerSpace space = PresburgerSpace::getRelationSpace();
  IntegerRelation srcRel(space), dstRel(space);
  // WHY 使用 Presburger 算术
  // 可以精确表示仿射映射和迭代空间

  // 场景 4: 构建 srcAccess 的访问关系
  // 例如: affine.load %A[i*2 + j, i - j]
  //       srcRel: (i, j) -> (i*2 + j, i - j)
  if (failed(srcAccess.getAccessRelation(srcRel)))
    return DependenceResult::Failure;
    // WHY 可能失败
    // 访问映射太复杂（非仿射）无法表示

  // 场景 5: 构建 dstAccess 的访问关系
  // 例如: affine.store %val, %A[i + 1, j*2]
  //       dstRel: (i', j') -> (i' + 1, j'*2)
  if (failed(dstAccess.getAccessRelation(dstRel)))
    return DependenceResult::Failure;

  // 步骤 6: 提取迭代空间约束
  FlatAffineValueConstraints srcDomain(srcRel.getDomainSet());
  FlatAffineValueConstraints dstDomain(dstRel.getDomainSet());
  // WHY 分离域和值
  // 域 = 迭代空间（循环边界）
  // 值 = 访问的内存位置

  // ========== 阶段 3: 顺序约束检查 ==========

  // 步骤 7: 检查字典序顺序
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  // 场景 6: loopDepth > numCommonLoops
  if (!allowRAR && loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess)) {
    return DependenceResult::NoDependence;
    // WHY 检查源是否在目标之前
    // 如果源在目标之后，不可能有 src->dst 的依赖
  }

  // ========== 阶段 4: 构建依赖多面体 ==========

  // 步骤 8: 组合访问关系
  // 目标: 找到 (i,j,i',j') 使得 src(i,j) = dst(i',j')
  dstRel.inverse();
  // WHY 反转 dstRel
  // dstRel: (i',j') -> (x',y')
  // inverse: (x',y') -> (i',j')
  // 这样可以组合 srcRel 和 inverse(dstRel)

  // 场景 7: 组合关系
  // srcRel: (i,j) -> (i*2+j, i-j)
  // inverse(dstRel): (x,y) -> (i'-1, j'/2)
  // 组合后: (i,j) -> (i*2+j, i-j) -> (i', j')
  //         使得 i*2+j = i'+1 且 i-j = j'/2
  dstRel.mergeAndCompose(srcRel);

  // 步骤 9: 转换变量种类
  dstRel.convertVarKind(VarKind::Domain, 0, dstRel.getNumDomainVars(),
                        VarKind::Range, 0);
  IntegerPolyhedron dependenceDomain(dstRel);
  // WHY 域变为值
  // 组合后我们关心的是迭代对之间的约束

  // ========== 阶段 5: 添加顺序约束 ==========

  // 步骤 10: 添加 src < dst 的约束
  addOrderingConstraints(srcDomain, dstDomain, loopDepth, &dependenceDomain);
  // WHY 需要顺序约束
  // 即使访问相同位置，如果 src 在 dst 之后，也不是依赖
  // 字典序: (i,j) < (i',j') 当且仅当 i < i' 或 (i = i' 且 j < j')

  // ========== 阶段 6: 检查依赖是否存在 ==========

  // 步骤 11: 检查解空间是否为空
  if (dependenceDomain.isEmpty())
    return DependenceResult::NoDependence;
    // WHY 空集表示无依赖
    // 没有满足所有约束的 (i,j,i',j') 元组

  // 步骤 12: 计算方向向量
  if (dependenceComponents != nullptr)
    computeDirectionVector(srcDomain, dstDomain, loopDepth, &dependenceDomain,
                           dependenceComponents);
    // WHY 方向向量描述依赖类型
    // [0] = 同一次迭代
    // [1] = 相邻迭代
    // [>0] = 长距离依赖

  return DependenceResult::HasDependence;
}
```

**完整执行流程示例 - 具体数据追踪:**

```text
// 示例: 检查以下代码的依赖
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %v1 = affine.load %A[%i, %j]          // srcAccess: S1
    affine.store %v1, %A[%i + 1, %j]      // dstAccess: S2
  }
}
```

**分析步骤追踪:**

```
========== 输入 ==========
srcAccess: S1 = load %A[i, j]
dstAccess: S2 = store %A[i+1, j]
loopDepth = 2

问题: S1 读取的数据被 S2 写入覆盖了吗？(WAR 依赖)

========== 阶段 1: 前置检查 ==========
步骤 1: srcAccess.memref (%A) == dstAccess.memref (%A) ✓
步骤 2: S1 是读, S2 是写 → 检查 WAR 依赖 ✓
步骤 3: scope 相同 ✓
步骤 4: 在同一块 ✓

========== 阶段 2: 构建访问关系 ==========
步骤 5: 构建访问映射
  srcRel: (i, j) -> (i, j)      [S1 在迭代 (i,j) 读取 A[i,j]]
  dstRel: (i', j') -> (i'+1, j')  [S2 在迭代 (i',j') 写入 A[i'+1,j']]

========== 阶段 3: 组合关系 ==========
步骤 8: 反转并组合，找到访问相同位置的迭代对
  目标: 找到 (i,j,i',j') 使得 A[i,j] = A[i'+1,j']

  约束推导:
    i = i' + 1  AND  j = j'
    → i' = i - 1  AND  j' = j

  解释: S1 在 (i,j) 读取的位置，等于 S2 在 (i-1,j) 写入的位置

========== 阶段 4: 添加顺序约束 ==========
步骤 10: 检查是否存在 (i,j) < (i',j') 的解

  字典序定义: (i,j) < (i',j') 当且仅当
    i < i'  OR  (i = i' AND j < j')

  从访问关系: i' = i - 1

  检查顺序约束:
    (i, j) < (i-1, j)  成立吗?
    → i < i-1?  NO
    → 结论: 不成立!

  代入具体数值验证:
    当 (i=5, j=10) 时:
      S1: load %A[5, 10]    ← 读取 A[5][10]
      S2: store %A[6, 10]   ← 写入 A[6][10]

    当 (i=4, j=10) 时:
      S1: load %A[4, 10]    ← 读取 A[4][10]
      S2: store %A[5, 10]   ← 写入 A[5][10]

    检查 A[5][10] 的依赖:
      S2 在 (4, 10) 写入 A[5][10]
      S1 在 (5, 10) 读取 A[5][10]

      顺序: (4, 10) < (5, 10)?  YES (4 < 5)

  结论: S1→S2 方向 NoDependence
       但存在 S2→S1 的反向依赖! (WAR)

  解释: S2 → S1 表示 "S1 依赖于 S2"
        = S2 必须在 S1 之前执行
        = S1 读取的值来自 S2 的写入

  具体例子:
    S2 在 (4, 10) 写入 A[5][10]
    S1 在 (5, 10) 读取 A[5][10]
    → S1 需要等待 S2 完成

========== 更清晰的依赖分析示例 ==========

示例 1: 明显的 RAW 依赖

affine.for %i = 0 to 99 {
  %v1 = affine.load %A[%i]           // S1: 读取 A[i]
  affine.store %v1, %A[%i + 1]       // S2: 写入 A[i+1]
}

执行追踪:
  i=0: S1 读 A[0], S2 写 A[1]
  i=1: S1 读 A[1], S2 写 A[2]
  ...

依赖: S2 在 i=0 写入 A[1]，S1 在 i=1 读取 A[1]
检查: (0) < (1)? YES
方向向量: [1]


示例 2: 对角线依赖

affine.for %i = 0 to 99 {
  affine.for %j = 0 to 99 {
    %v1 = affine.load %A[%i, %j]          // S1
    affine.store %v1, %A[%i + 1, %j + 1]  // S2
  }
}

访问关系:
  S1: (i, j) -> (i, j)
  S2: (i', j') -> (i'+1, j'+1)

相等约束: i = i'+1, j = j'+1
顺序检查: (i,j) < (i-1, j-1)?  NO (i > i-1)

结论: S1→S2 无依赖，S2→S1 有依赖


示例 3: 真正的 S1→S2 依赖

affine.for %i = 0 to 99 {
  %v1 = affine.load %A[%i + 1]      // S1: 读取 A[i+1]
  affine.store %v1, %A[%i]          // S2: 写入 A[i]
}

访问关系:
  S1: (i) -> (i+1)
  S2: (i') -> (i')

相等约束: i+1 = i'  →  i' = i+1
顺序检查: (i) < (i+1)?  YES (i < i+1)

结论: S1→S2 有依赖，方向向量 [1]

========== 核心理解 ==========

依赖分析检查的是:
1. 相同位置: src 访问的位置 = dst 访问的位置
2. 执行顺序: src 的迭代点 < dst 的迭代点（字典序）

对于原始代码:
  S1 读取 A[i][j], S2 写入 A[i+1][j]

  S1(i,j) 和 S2(i-1,j) 访问相同位置
  但 (i,j) > (i-1,j)，所以是 S2→S1 依赖，不是 S1→S2
```

**易错点标注:**

1. ⚠️ **方向向量符号**: 正数表示正向依赖，负数表示反向
2. ⚠️ **边界条件**: 循环边界可能切断依赖
3. ⚠️ **复杂映射**: 非仿射访问（如 `A[i*j]`）会失败

---

## 5. 变换操作详解

### 5.1 循环分块 (Loop Tiling)

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp` (200 行)
- **测试文件：** `mlir/test/Dialect/Affine/loop-tiling.mlir`

#### WHAT：循环分块是什么？

**循环分块** (Loop Tiling) 将嵌套循环的 **迭代空间划分为小块** (tiles)，每个小块能放入缓存。

**直观理解：**

- 原始：处理整个矩阵 256x256
- 分块 32x32：处理 8x8 个小块，每块 32x32

**示例：**

**分块前：**

```text
affine.for %i = 0 to 256 {
  affine.for %j = 0 to 512 {
    affine.for %k = 0 to 1024 {
      "test.foo"(%i, %j, %k)
    }
  }
}
```

**分块后 (tileSize=32)：**

```text
affine.for %i_outer = 0 to 256 step 32 {
  affine.for %j_outer = 0 to 512 step 32 {
    affine.for %k_outer = 0 to 1024 step 32 {
      affine.for %i = %i_outer to min(%i_outer + 32, 256) {
        affine.for %j = %j_outer to min(%j_outer + 32, 512) {
          affine.for %k = %k_outer to min(%k_outer + 32, 1024) {
            "test.foo"(%i, %j, %k)
          }
        }
      }
    }
  }
}
```

#### WHY：为什么需要分块？

**问题：缓存未命中**

考虑矩阵乘法 `C[i,j] += A[i,k] * B[k,j]`：

| 访问模式 | 局部性 | 问题               |
| -------- | ------ | ------------------ |
| `A[i,k]` | 好     | i 固定，k 顺序访问 |
| `B[k,j]` | **差** | k 跳跃访问，j 固定 |
| `C[i,j]` | 差     | 每次都写入不同位置 |

**WHY B[k,j] 的访问模式差？**

- 第 1 次迭代：`B[0,0], B[1,0], B[2,0], ...`
- 第 2 次迭代：`B[0,1], B[1,1], B[2,1], ...`
- 缓存行未被充分利用

**分块的效果：**

```cpp
// 分块后：外层处理块，内层处理块内元素
for (ii = 0; ii < N; ii += 32)        // 块行索引
  for (jj = 0; jj < N; jj += 32)      // 块列索引
    for (kk = 0; kk < N; kk += 32)    // 块深度索引
      for (i = ii; i < ii+32; i++)    // 块内行
        for (j = jj; j < jj+32; j++)  // 块内列
          for (k = kk; k < kk+32; k++) // 块内深度
            C[i][j] += A[i][k] * B[k][j];
```

**WHY 这样有效？**

- 块内的 `B[k,j]` 访问是 **局部** 的
- 整个块在处理期间保持在 **L2/L1 缓存** 中
- 缓存命中率大幅提升

#### Tiling流程（LoopTiling.cpp）

**1. 顶层入口**

```
  runOnOperation()                                                                                    
  ├── getTileableBands(func, &bands)                                                                  
  │     → 扫描整个 FuncOp，收集所有"可平铺循环带"（完美嵌套的 AffineForOp 序列，也就是forOp的循环体中只有forOp）     
  │                                                                                                 
  ├── for each band:
  │   ├── [约束] isTilingValid(band) // 判断能否实施Tiling
  │   │         → 通过 checkMemrefAccessDependence 检查依赖关系，若循环间存在阻止 tiling 的数据依赖则跳过
  │   │           （emit remark: "tiling nest is invalid due to dependences"）
  │   │
  │   ├── getTileSizes(band, &tileSizes)
  │   │     → 计算每层循环的 tile size（见下节）
  │   │
  │   ├── tilePerfectlyNested(band, tileSizes, &tiledNest)
  │   │     → 核心变换：将 N 层循环展开成 2N 层（外层 tile 循环 + 内层 intra-tile 循环）
  │   │     → [约束] 若变换失败则跳过（空 band 保证成功）
  │   │
  │   └── [可选] if separate:
  │         separateFullTiles(intraTileLoops)
  │           → 将 tiledNest 后半段（intra-tile 循环）分离成 full tile 和 partial tile 两部分
  │           → intraTileLoops = tiledNest.drop_front(band.size())（去掉外层 tile 循环）
```

**2. Tile Size计算**

```
// 优先级从高到低，命中即返回
getTileSizes(band, tileSizes)  
│
├── [优先级1] 命令行指定了 --tile-size=N
│     → 所有维度统一使用 N
│
├── [优先级2] 命令行指定了 --tile-sizes=a,b,c,...
│     → 按序赋值；不足的维度用 kDefaultTileSize(=4) 补齐
│
├── [约束] band 为空 → 直接返回
│
├── [约束] cacheSizeInKiB == 0 → 所有维度设为 1（最小合法 tile size）
│
├── getMemoryFootprintBytes(band[0], 0)
│   ├── 若 footprint 未知（返回 nullopt）：
│   │     → 所有维度填 kDefaultTileSize(=4)
│   │     → [约束] avoidMaxMinBounds=true 时，调用 adjustToDivisorsOfTripCounts()
│   │
│   └── 若 footprint 已知（根据缓存空间大小计算开N次根号）：
│       ├── excessFactor = ceil(footprint / cacheSize)
│       ├── [约束] excessFactor <= 1 → footprint 已经装得下，所有维度设为 1，不需要 tiling
│       │
│       └── excessFactor > 1 → 需要缩减
│             → tSize = floor(excessFactor ^ (1/N))  // N 维均分 excess
│             → 前 N-1 维均设为 tSize，最后一维 = excessFactor / 前面累积乘积
│             → [约束] avoidMaxMinBounds=true 时，调用 adjustToDivisorsOfTripCounts()

```

**3. Tile Size调整**

```
  adjustToDivisorsOfTripCounts(band, tileSizes)
  ├── for each loop[i] in band:
  │   ├── getConstantTripCount(loop[i])
  │   │   └── 若 trip count 未知 → 跳过该维度
  │   │
  │   └── 若 trip count 已知（= T）：
  │       ├── [约束] T > 1 且 tSize > T/2 → 先将 tSize 缩到 T/2
  │       └── while T % tSize != 0: tSize--
  │             → 保证 tile size 整除 trip count，避免边界处产生 max/min 表达式
```

**4. 关键约束汇总**

| 约束                     | 位置                                                | 效果                                           |
| ------------------------ | --------------------------------------------------- | ---------------------------------------------- |
| 依赖检查失败             | `isTilingValid()`                                   | 整个 band 跳过，不 tile                        |
| cacheSizeInKiB == 0      | `getTileSizes()`                                    | 所有维度 tile size = 1                         |
| footprint <= cache       | `getTileSizes()`                                    | 所有维度 tile size = 1（无需 tile）            |
| footprint 未知           | `getTileSizes()`                                    | 退回 kDefaultTileSize = 4                      |
| avoidMaxMinBounds        | `getTileSizes()` + `adjustToDivisorsOfTripCounts()` | tile size 向下取整除                           |
| trip count 的最大因子    |                                                     |                                                |
| tilePerfectlyNested 失败 | `runOnOperation()`                                  | 该 band 跳过，继续下一个                       |
| separate 模式            | `runOnOperation()`                                  | 额外调用 separateFullTiles，分离完整/边界 tile |

#### 源代码级深度解析

**核心算法 1：`getTileSizes` - 分块大小计算**

```cpp
// 来源: LoopTiling.cpp (99-176 行)
// 智能计算分块大小的完整实现
void LoopTiling::getTileSizes(ArrayRef<AffineForOp> band,
                              SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  // === 策略 1：命令行固定大小 ===
  if (tileSize) {
    tileSizes->assign(band.size(), tileSize);
    return;
  }

  // === 策略 2：用户提供的大小列表 ===
  if (!this->tileSizes.empty()) {
    tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());
    tileSizes->resize(band.size(), kDefaultTileSize);  // 填充默认值
    return;
  }

  tileSizes->resize(band.size());

  // === 策略 3：无缓存信息 → 最小有效大小 ===
  if (cacheSizeInKiB == 0) {
    llvm::fill(*tileSizes, 1);  // WHY：1 是有效的最小分块大小
    return;
  }

  // === 策略 4：基于缓存大小的自动计算 ===
  // 获取内存足迹
  std::optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
  if (!fp) {
    // 未知足迹：使用默认值并调整为 trip count 约数
    llvm::fill(*tileSizes, LoopTiling::kDefaultTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    return;
  }

  // 计算需要缩小的倍数
  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;
  uint64_t excessFactor = llvm::divideCeil(*fp, cacheSizeBytes);

  // 如果已经能放入缓存：不需要分块
  if (excessFactor <= 1) {
    llvm::fill(*tileSizes, 1);
    return;
  }

  // === 策略 5：在各维度平均分配缩放因子 ===
  // WHY：n 维循环 → 计算 excessFactor 的 n 次方根
  // 例如：256×256×256 = 16,777,216，缓存 32KB
  //      excessFactor ≈ 512，3D → tSize = 8
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));

  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // 最后一个维度：覆盖剩余部分
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }

  // 可选：调整为 trip count 约数 (避免 min/max 边界)
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}
```

**分块大小选择策略**

| 策略         | 方法                  | 优点         | 缺点               |
| ------------ | --------------------- | ------------ | ------------------ |
| **固定大小** | 命令行指定            | 可控、可复现 | 需要手动调优       |
| **缓存感知** | 基于缓存大小计算      | 自动适应硬件 | 依赖准确的足迹计算 |
| **默认值**   | 使用 kDefaultTileSize | 简单         | 可能不是最优       |

**WHY 默认值是 4？**

- 典型的 L1 缓存行大小是 64 字节
- float (4 bytes) × 16 = 64 字节
- 4×4 块 × 4 bytes = 64 字节 (刚好一个缓存行)

**核心算法 2：`adjustToDivisorsOfTripCounts` - 避免 min/max 边界**

```cpp
// 来源: LoopTiling.cpp (75-91 行)
// 将分块大小调整为 trip count 的约数
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];

    // 获取常量 trip count
    std::optional<uint64_t> mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;  // 非常量：无法调整

    uint64_t constTripCount = *mayConst;

    // WHY 限制为 tripCount/2：
    // 避免 tile size 接近 trip count (无意义的大分块)
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;

    // WHY 向下递减寻找约数：
    // 保证 tripCount % tileSize == 0
    // 例如：tripCount = 100，tileSize = 32 → 调整为 25
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}
```

**WHY 避免 min/max 边界？**

```text
// 有 min/max 的分块 (性能较差)
affine.for %i_outer = 0 to 256 step 32 {
  affine.for %j_outer = 0 to 256 step 32 {
    affine.for %i = %i_outer to min(%i_outer + 32, 256) {  // 运行时检查
      affine.for %j = %j_outer to min(%j_outer + 32, 256) {  // 运行时检查
        // ...
      }
    }
  }
}

// 无 min/max 的分块 (性能更好)
affine.for %i_outer = 0 to 256 step 32 {
  affine.for %j_outer = 0 to 256 step 32 {
    affine.for %i = %i_outer to %i_outer + 32 {  // 编译期确定
      affine.for %j = %j_outer to %j_outer + 32 {  // 编译期确定
        // ...
      }
    }
  }
}
// 注意：只有当 256 % 32 == 0 时才能这样生成
```

**核心算法 3：`tilePerfectlyNestedLoops` - 执行分块**

```cpp
// 来源: LoopUtils.cpp (实际分块实现)
LogicalResult tilePerfectlyNestedLoops(
    ArrayRef<AffineForOp> band,
    ArrayRef<unsigned> tileSizes,
    SmallVector<AffineForOp, 6> *tiledNest) {

  // WHY 从内向外处理：内层循环先分块
  // 分块后：原循环变为 intra-tile 循环，新循环为 tile 循环
  for (size_t i = band.size(); i > 0; i--) {
    AffineForOp forOp = band[i - 1];
    unsigned tileSize = tileSizes[i - 1];

    // === 步骤 1：提取循环信息 ===
    uint64_t step = forOp.getStep();
    AffineMap lbMap = forOp.getLowerBoundMap();
    AffineMap ubMap = forOp.getUpperBoundMap();
    ValueRange lbOperands = forOp.getLowerBoundOperands();
    ValueRange ubOperands = forOp.getUpperBoundOperands();

    // === 步骤 2：创建 tile 循环 (外层) ===
    // WHY 使用 OpBuilder：确保正确的插入位置
    OpBuilder b(forOp.getOperation());
    Location loc = forOp.getLoc();

    // 创建 tile 循环：lb to ub step tileSize
    AffineForOp tileLoop = b.create<AffineForOp>(
        loc, lbMap, lbOperands, ubMap, ubOperands, step * tileSize);
    tileLoop.getBody()->getOperations().splice(
        tileLoop.getBody()->begin(),
        forOp.getBody()->getOperations());

    // === 步骤 3：创建 intra-tile 循环 (内层) ===
    // WHY 插入在 tile 循环体内
    b.setInsertionPointToStart(tileLoop.getBody());

    // 计算 intra-tile 下界：tileIV * (step * tileSize) + lb
    // 简化：tileIV * tileSize (假设 step=1)
    AffineMap intraLbMap = b.getAffineMapVarResults();
    SmallVector<Value> intraLbOperands;

    // 创建 intra-tile 循环
    AffineForOp intraTileLoop = b.create<AffineForOp>(
        loc, intraLbMap, intraLbOperands,
        /*ubMap=*/..., /*ubOperands=*/...,
        /*step=*/step);

    // === 步骤 4：移动循环体到 intra-tile 循环 ===
    intraTileLoop.getBody()->getOperations().splice(
        intraTileLoop.getBody()->begin(),
        tileLoop.getBody()->getOperations());

    // === 步骤 5：替换 IV 使用 ===
    // WHY：原循环的操作引用了原 IV，需要替换为 intra-tile IV
    forOp.getInductionVar().replaceAllUsesWith(intraTileLoop.getInductionVar());

    // === 步骤 6：删除原循环 ===
    forOp.erase();

    // === 步骤 7：更新 band 引用 ===
    // 分块后，原循环被 tileLoop 和 intraTileLoop 替代
    // 后续迭代需要使用新的循环
    tiledNest->push_back(tileLoop);
    tiledNest->push_back(intraTileLoop);
  }

  return success();
}
```

**执行流程示例：3D 矩阵分块**

```cpp
// === 原始代码 ===
affine.for %i = 0 to 256 {
  affine.for %j = 0 to 256 {
    affine.for %k = 0 to 256 {
      %v = "compute"(%i, %j, %k) : f32
      "use"(%v)
    }
  }
}

// === 分块后 (tileSize = 32) ===
affine.for %i_tile = 0 to 256 step 32 {          // i tile 循环
  affine.for %j_tile = 0 to 256 step 32 {        // j tile 循环
    affine.for %k_tile = 0 to 256 step 32 {      // k tile 循环
      affine.for %i = %i_tile to %i_tile + 32 {  // i intra-tile
        affine.for %j = %j_tile to %j_tile + 32 {  // j intra-tile
          affine.for %k = %k_tile to %k_tile + 32 {  // k intra-tile
            %v = "compute"(%i, %j, %k) : f32
            "use"(%v)
          }
        }
      }
    }
  }
}

// 内存分析：
// 原始：每个 iteration 访问不同位置
// 分块：每个 32³ 块内的访问是局部的
//      块大小 = 32³ × 4 bytes = 128KB (可放入 L2 缓存)
```

**WHY 分块顺序重要？**

```cpp
// 好的分块：外层是 tile，内层是 intra-tile
for (i_tile)          // 缓慢变化：块间迭代
  for (j_tile)
    for (k_tile)
      for (i)         // 快速变化：块内迭代
        for (j)
          for (k)
            compute(i, j, k)

// 坏的分块：tile 和 intra-tile 混合
for (i_tile)          // 块迭代
  for (i)             // 块内迭代
    for (j_tile)      // 又是块迭代！
      for (j)
        // ...
// WHY 不好：频繁切换块，破坏局部性
```





### 5.2 循环融合 (Loop Fusion)

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp` (~1000+ 行，最复杂的 Pass 之一)
- **头文件：** `mlir/include/mlir/Dialect/Affine/LoopFusionUtils.h`
- **测试文件：** `mlir/test/Dialect/Affine/loop-fusion.mlir` 及多个变体

#### WHAT：循环融合是什么？

**循环融合** (Loop Fusion) 将多个独立的循环嵌套 **合并为一个**，以减少内存访问和改善局部性。

**示例：**

**融合前：**

```text
// 生产者循环：写入 B
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 10 {
    %v = affine.load %A[%i, %j] : memref<10x10xf32>
    %r = arith.addf %v, %cst : f32
    affine.store %r, %B[%i] : memref<10xf32>
  }
}

// 消费者循环：读取 B
affine.for %i = 0 to 10 {
  %v = affine.load %B[%i] : memref<10xf32>
  affine.store %v, %C[%i] : memref<10xf32>
}
```

**融合后：**

```text
affine.for %i = 0 to 10 {
  // 生产者循环体融合进来
  affine.for %j = 0 to 10 {
    %v = affine.load %A[%i, %j] : memref<10x10xf32>
    %r = arith.addf %v, %cst : f32
    affine.store %r, %B_local[0] : memref<1xf32>  // 使用局部缓冲
  }
  // 消费者循环体融合进来
  %v2 = affine.load %B_local[0] : memref<1xf32>
  affine.store %v2, %C[%i] : memref<10xf32>
}
```

#### WHY：为什么需要循环融合？

| 收益               | 解释               | 底层原理                            |
| ------------------ | ------------------ | ----------------------------------- |
| **减少内存访问**   | 不再写入主内存     | B 从 memref 变为寄存器/栈上局部变量 |
| **改善缓存局部性** | 数据在缓存中保持   | 生产后立即消费，充分利用临时局部性  |
| **消除同步点**     | 不需要等待写入完成 | 融合后的计算可以流水线执行          |
| **减少循环开销**   | 循环数量减少       | 减少分支和控制逻辑                  |

**WHY 适用于仿射循环？**

- 仿射约束可以 **精确分析依赖**
- 可以安全地 **切片融合** (slice fusion)
- 可以自动 **生成局部缓冲**

#### HOW：融合原理

**核心概念：依赖图 (MemRef Dependence Graph)**

```cpp
struct MemRefDependenceGraph {
  // 节点：循环嵌套或操作序列
  struct Node {
    unsigned id;
    Operation *op;
    SmallVector<Operation *> loads;   // 该节点的加载操作
    SmallVector<Operation *> stores;  // 该节点的存储操作
  };

  // 边：依赖关系
  struct Edge {
    unsigned id;    // 目标节点 ID
    Value memref;   // 依赖的 memref
  };

  DenseMap<unsigned, Node> nodes;
  DenseMap<unsigned, SmallVector<Edge>> inEdges;   // 入边
  DenseMap<unsigned, SmallVector<Edge>> outEdges;  // 出边
};
```

**融合类型：**

| 类型                  | 描述              | 示例                                   |
| --------------------- | ----------------- | -------------------------------------- |
| **Producer-Consumer** | 生产者-消费者依赖 | `A[i] = ...; ... = B[i]` 其中 B 依赖 A |
| **Sibling**           | 兄弟循环融合      | 两个都写入相同的输出                   |
| **Greedy**            | 贪心融合 (默认)   | 尝试所有可能的融合                     |

#### 融合流程 (LoopFusion.cpp)

**1. 入口**

```
runOnBlock(block)
  → g.init()              // 构建 MDG：遍历 block，为每个 forOp 建节点，分析 load/store 建依赖边
  → GreedyFusion(g, ...)
      → runGreedyFusion() / runProducerConsumerFusionOnly() / runSiblingFusionOnly()
```

**2. ProducerConsumer 融合**

```
fuseProducerConsumerNodes(maxSrcUserCount)
  └─ performFusionsIntoDest(dstId)
       │
       ├─ sinkSequentialLoops(dstNode)
       │    // 把顺序循环下沉，提升可融合深度
       │
       ├─ getProducerCandidates(dstId)
       │    // 根据 dst 读的 memref → 找出写该 memref 的 src 节点
       │
       └─ for each srcId (逆序遍历):
            │
            ├─ [约束] srcNode 不能有返回值
            │
            ├─ gatherProducerConsumerMemrefs(src, dst)
            │    // = src.stores ∩ dst.loads（按 memref 取交集）
            │
            ├─ [约束] outEdgeCount(src, memref) <= maxSrcUserCount
            │    // 限制 src 的消费者数量
            │
            ├─ gatherEscapingMemrefs(src)
            │    // 找出 src 中写到 block 外部的 memref
            │
            ├─ getFusedLoopNestInsertionPoint(src, dst)
            │    // src和dst是两个并行（兄弟节点）的forOp，其各自前后可能存在其他Op，且src和dst之间也可能存在其他Op
            │    // 在 (src,dst) 之间找合法插入点
            │    // firstDepOpA = 第一个依赖 src 的 op
            │    // lastDepOpB  = 最后一个 dst 依赖的 op
            │    // [约束] firstDepOpA > lastDepOpB，否则返回 nullptr
            │
            ├─ 计算 dstLoopDepthTest
            │    // dst 中（for循环体中）存在其他forOp或其他读写memref的Op
            │    // 找出访问 producer-consumer memref 的 op 的最内公共循环深度，也就是可以融合的最大for循环层数
            │    // - numSurroundingLoops
            │
            ├─ for depth = 1 to dstLoopDepthTest:
            │    canFuseLoops(src, dst, depth)  // 逐个判断两个forOp的第depth层能否融合
            │    ├─ getFusedLoopNestInsertionPoint   // 再次检查插入点
            │    ├─ [约束] getMaxLoopDepth(opsA,opsB) >= dstLoopDepth
            │    │    // dst 内部依赖不被破坏的最大深度
            │    │    // 检查 targetDstOps 内所有 op 对的依赖，取最小安全深度（从外层开始最小的可融合层数）
            │    └─ computeSliceUnion(strategyOpsA, opsB, depth)
            │         // strategyOpsA = src 的 stores（ProducerConsumer策略）
            │         // 对所有依赖对计算切片约束并取并集
            │         // → srcSlice（src 需要执行的迭代子集）
            │
            ├─ [约束] hasCyclicDependence(src)?
            │    // 有循环依赖时：
            │    //   maximalFusion: fraction必须==0
            │    //   普通模式: computeTolerance强制=0
            │
            ├─ isFusionProfitable(src, dst, depthSliceUnions)
            │    // 非 maximalFusion 时执行
            │    // 计算各深度的冗余计算比例
            │    // [约束] 冗余比例 <= computeToleranceThreshold
            │    // → bestDstLoopDepth
            │
            ├─ canRemoveSrcNodeAfterFusion(src, dst, slice, insPoint)
            │    // 满足以下任一可删除 src：
            │    // 1. 无出边依赖 + 无 escaping memref
            │    // 2. 无出边依赖 + 有 escaping memref + slice maximal
            │    // 3. 有出边依赖 + slice maximal + 插入点在所有依赖之前
            │
            ├─ canCreatePrivateMemRef(memref, ...)
            │    // 对每个 p-c memref 判断能否私有化：
            │    // [约束1] 元素大小可计算
            │    // [约束2] 不能(逃逸 且 (src被删 或 dst也写它))
            │    // [约束3] src 无该 memref 的入边，dst 无该 memref 的出边
            │    // [约束4] src被删时，该 memref 不能有 dst 以外的消费者
            │
            └─ 执行融合
                 ├─ fuseLoops(src, dst, bestSlice)
                 ├─ dstForOp->moveBefore(fusedLoopInsPoint)
                 ├─ createPrivateMemRef(...)  // 为 privateMemrefs 创建局部 alloc
                 ├─ mdg->updateEdges(...)
                 └─ removeSrcNode ? mdg->removeNode(src) + src->erase()

```

**2. Sibling 融合**

```
fuseSiblingNodes()
  └─ for each dstNode:
       │
       ├─ findSiblingNodeToFuse(dstNode)
       │    └─ canFuseWithSibNode(sibNode, memref)
       │         ├─ [约束] sibNode.getLoadOpCount(memref) == 1
       │         │    // 对共享 memref 只有一个 load
       │         ├─ [约束] 无依赖路径（双向）
       │         │    // !hasDependencePath(sib,dst) && !hasDependencePath(dst,sib)
       │         └─ [约束] sibNode 中既读又写的 memref，不能有外部写入者
       │              // getIncomingMemRefAccesses(sib, memref) == 0
       │
       ├─ getFusedLoopNestInsertionPoint(sib, dst)
       │
       ├─ 计算 dstLoopDepthTest（同 ProducerConsumer）
       │
       ├─ for depth = 1 to dstLoopDepthTest:
       │    canFuseLoops(sib, dst, depth, Sibling策略)
       │    └─ computeSliceUnion(strategyOpsA, opsB, depth)
       │         // strategyOpsA = sib 中读共享 memref 的 load（Sibling策略）
       │
       ├─ [约束] bestSlice.isMaximal() == true
       │    // Sibling 融合要求切片必须是 maximal，否则不融合
       │
       └─ 执行融合
            ├─ fuseLoops(sib, dst, bestSlice)
            ├─ dstForOp->moveBefore(insertPointInst)
            ├─ updateStateAfterSiblingFusion(sib, dst)
            └─ mdg->removeNode(sib) + sib->erase()

```

**3. GreedyFusion 融合**

```
fuseProducerConsumerNodes(maxSrcUserCount=1)
    // 只处理 p-c memref 的唯一消费者，最安全
fuseSiblingNodes()
    // 第一轮融合后图更稀疏，兄弟机会更多
fuseProducerConsumerNodes(maxSrcUserCount=MAX)
    // 放开限制，处理多消费者情况
eraseUnusedMemRefAllocations()
    // 清理不再引用的 alloc
```

**4. 核心数据结构**

```
MDG
 ├─ nodes: id → Node { op, loads[], stores[] }
 ├─ inEdges:  id → [Edge{ id=上游, value=memref }]
 ├─ outEdges: id → [Edge{ id=下游, value=memref }]
 └─ block: 所有节点所在的顶层 Block

ComputationSliceState（srcSlice）
 └─ src 在融合后需要执行的迭代约束（仿射表达式）

DependenceComponent { lb, ub }
 └─ 每层公共循环上的依赖距离范围
```

**5. 依赖分析核心（checkMemrefAccessDependence）**

```
checkMemrefAccessDependence(src, dst, loopDepth)
  │
  ├─ 构建访问关系 srcRel / dstRel
  │    // (IV...) → (memref下标...)
  │
  ├─ 组合依赖关系: dstRel.inverse() ∘ srcRel
  │    // 得到 src迭代域 → dst迭代域 中访问同一地址的关系
  │
  ├─ addOrderingConstraints(loopDepth)
  │    // depth=1:       i_dst - i_src >= 1
  │    // depth=2:       i_src==i_dst,  j_dst - j_src >= 1
  │    // depth=k(最大): 前k-1层相等,   第k层 >= 1 或程序顺序
  │
  ├─ dependenceDomain.isEmpty() → NoDependence
  │
  └─ computeDirectionVector()
       // 每个公共循环维度输出 DependenceComponent { lb, ub }
       // lb>0: 正向  ub<0: 反向  跨0: 方向不确定
```

#### 完整样例演示

**输入 MLIR**

```text
func.func @example(%out: memref<10xf32>) {
  %A = memref.alloc() : memref<10xf32>
  %B = memref.alloc() : memref<10xf32>

  // Loop 1 (id=1, src): 计算 A[i] = i * 2
  affine.for %i = 0 to 10 {
    %v = arith.constant 2.0 : f32
    affine.store %v, %A[%i] : memref<10xf32>
  }

  // Loop 2 (id=2, dst): B[i] = A[i] + 1, out[i] = B[i]
  affine.for %i = 0 to 10 {
    %a = affine.load %A[%i] : memref<10xf32>
    %one = arith.constant 1.0 : f32
    %b = arith.addf %a, %one : f32
    affine.store %b, %B[%i] : memref<10xf32>
    %bv = affine.load %B[%i] : memref<10xf32>
    affine.store %bv, %out[%i] : memref<10xf32>
  }
  return
}
```

------

**Step 1：构建 MDG**

Nodes:

| ID     | 说明                                                         |
| ------ | ------------------------------------------------------------ |
| `id=0` | `alloc %A`                                                   |
| `id=1` | Loop1（stores: `[store %A]`）                                |
| `id=2` | Loop2（loads: `[load %A, load %B]`，stores: `[store %B, store %out]`） |

**Edges:**

```
outEdges[0] = [{id=1, value=%A}]   // alloc %A → Loop1 (SSA 依赖)
outEdges[1] = [{id=2, value=%A}]   // Loop1 写 A，Loop2 读 A
inEdges[2]  = [{id=1, value=%A}]
```

**Step 2：performFusionsIntoDest(dstId=2)**

```
getProducerCandidates(dstId=2)` → dst 读 `%A`，找写 `%A` 的节点 → `srcIdCandidates = [1]
```

**Step 3：评估 src=1, dst=2**

**`gatherProducerConsumerMemrefs`：**

```
srcStoreMemRefs            = {%A}
dstLoadMemRefs             = {%A, %B}
producerConsumerMemrefs    = {%A}
```

- `outEdgeCount(%A) = 1`（只有 dst），不超过 `maxSrcUserCount`，继续。
- `srcEscapingMemRefs`：`%A` 是内部 alloc，不逃逸 → `{}`

**`getFusedLoopNestInsertionPoint`：**

- `(Loop1, Loop2)` 之间无其他 op
- `firstDepOpA = null`（Loop1 无其他消费者）
- `lastDepOpB = null`（Loop2 无其他生产者）
- → 插入点 = Loop2 本身

**Step 4：计算可融合深度**

```
dstMemrefOps        = Loop2 中访问 %A 的 op = [load %A[%i]]
innermostCommonLoopDepth([load %A]) = 1   // 只有一层 %i
numSurroundingLoops = 0
dstLoopDepthTest    = 1 - 0 = 1
```

**尝试 `i=1`：`canFuseLoops(Loop1, Loop2, depth=1)`**

```
getMaxLoopDepth(opsA, opsB):
  targetDstOps = [load %A]（只有 load，无 store）
  → 全是 load，返回 innermostCommonLoopDepth = 1

getMaxLoopDepth = 1 >= dstLoopDepth = 1  ✓
```

**`computeSliceUnion`：**

- `(store %A[%i], load %A[%j])` 访问相同 memref
- 依赖约束：`%i == %j`
- slice：`%i_src = %i_dst`（1:1 对应）
- → Success，`maxLegalFusionDepth = 1`

**Step 5：Profitability 分析**

```
producerStores = [store %A]
isFusionProfitable(depth=1):
  sliceCost  = 10（src 跑 10 次，与原来相同）
  fusedCost  = 10（无冗余）
  冗余比例   = 0%  ≤ threshold
→ bestDstLoopDepth = 1
```

**Step 6：canRemoveSrcNodeAfterFusion**

```
outEdges[1] = [{id=2, value=%A}]，融合后 dst=2 消除
→ hasOutDepsAfterFusion = false
  escapingMemRefs = {}
→ 条件 1 满足，removeSrcNode = true
```

**Step 7：canCreatePrivateMemRef(%A)**

| 条件                                                         | 结果 |
| ------------------------------------------------------------ | ---- |
| 元素大小可计算                                               | ✓    |
| `%A` 不逃逸                                                  | ✓    |
| `inEdges of src on %A`（alloc→src 是 SSA 边，`getIncomingMemRefAccesses=0`） | ✓    |
| `outEdges of dst on %A`：无                                  | ✓    |
| `removeSrcNode=true`，`%A` 的 outEdge 只有 dst               | ✓    |

→ `privateMemrefs = {%A}`

------

**Step 8：执行融合**

`fuseLoops(Loop1, Loop2, slice=%i_src=%i_dst)`：

```text
affine.for %i = 0 to 10 {
  // 插入 src 的切片
  %v = arith.constant 2.0 : f32
  affine.store %v, %A[%i]        // 先 store
  // 原 dst 的内容
  %a = affine.load %A[%i]        // 立即 load
  %one = arith.constant 1.0 : f32
  %b = arith.addf %a, %one : f32
  affine.store %b, %B[%i]
  %bv = affine.load %B[%i]
  affine.store %bv, %out[%i]
}
```

------

**Step 9：创建私有 memref**

`createPrivateMemRef` 将 `%A` 替换为局部小 buffer（大小由切片决定，这里 1 元素即可）：

```text
affine.for %i = 0 to 10 {
  %A_private = memref.alloc() : memref<1xf32>  // 私有 buffer
  %v = arith.constant 2.0 : f32
  affine.store %v, %A_private[0]
  %a = affine.load %A_private[0]
  %one = arith.constant 1.0 : f32
  %b = arith.addf %a, %one : f32
  affine.store %b, %B[%i]
  %bv = affine.load %B[%i]
  affine.store %bv, %out[%i]
}
// Loop1 被删除，%A 的 alloc 也随后被 eraseUnusedMemRefAllocations 清理
```

------

**最终效果对比**

| 指标           | 融合前          | 融合后           |
| -------------- | --------------- | ---------------- |
| 循环数         | 2               | 1                |
| `%A` 内存      | `10×f32` 全量   | `1×f32` 寄存器级 |
| cache locality | store 完再 load | store/load 相邻  |
| Loop1          | 保留            | 删除             |

#### 性能模型

**融合收益计算：**

```cpp
// 计算融合带来的额外计算比例
std::optional<double> getAdditionalComputeFraction(
    AffineForOp srcForOp, AffineForOp dstForOp, unsigned depth) {

  // 原始成本
  uint64_t srcCost = getComputeCost(srcForOp);
  uint64_t dstCost = getComputeCost(dstForOp);

  // 融合后成本（可能增加计算）
  uint64_t sliceCost = getSliceCost(srcForOp, dstForOp, depth);
  uint64_t fusedCost = dstCost + sliceCost;

  // 额外计算比例
  return (double)(fusedCost - dstCost) / (double)(srcCost + dstCost);
}
```

**WHY 需要计算额外计算？**

- 融合可能增加 **冗余计算** (如切片导致重复计算)
- 如果额外计算过多，融合可能不值得
- 需要在 **局部性收益** 和 **计算成本** 之间权衡

> **疑问：是否要求srcForOp和dstForOp对应深度循环的上下界一致？**
>
> **答：**不需要强制要求上下界一致，因为**切片机制**已经处理了不一致的情况。
>   **computeSliceUnion**计算的是：在给定 `dstLoopDepth` 下，dst 的每个迭代需要 src 执行哪些迭代。这个切片约束本身就是仿射表达式，可以处理上下界不同的情况。
>
> ```text
> affine.for %i = 0 to 10 {         // src，10次
>   affine.store %v, %A[%i * 2]
> }
> 
> affine.for %j = 0 to 20 {         // dst，20次，上界不同
>   %a = affine.load %A[%j]
> }
> ```
>
> 切片约束：`%i_src = %j / 2`，src 的迭代范围由 dst 的迭代推导出来，不要求两者上下界相同。
>
> **但有间接约束：**
>
>     1. `getMaxLoopDepth` 检查的是依赖方向，不检查上下界
>     2. `isFusionProfitable` 会计算切片的计算量，上下界不同会影响冗余比例，可能导致融合被判断为不划算
>     3. `isMaximal` 检查切片是否覆盖 src 的全部迭代空间——上下界不一致时切片通常不是 maximal，影响 src
>        能否被删除

#### 源代码级深度解析

**核心数据结构：MemRefDependenceGraph**

```cpp
// 来源: LoopFusion.cpp (依赖图构建的核心结构)
struct MemRefDependenceGraph {
  /// 节点：代表一个循环嵌套或操作序列
  struct Node {
    unsigned id;                                // 唯一标识符
    Operation *op;                              // 对应的 Operation
    SmallVector<Operation *, 4> loads;          // 该节点的加载操作
    SmallVector<Operation *, 4> stores;         // 该节点的存储操作
    DenseMap<Value, unsigned> memrefAccessCounts; // memref 访问计数

    // 辅助函数：获取特定 memref 的存储操作数
    unsigned getStoreOpCount(Value memref) const {
      unsigned count = 0;
      for (Operation *op : stores)
        if (cast<AffineWriteOpInterface>(op).getMemRef() == memref)
          ++count;
      return count;
    }
  };

  /// 边：代表依赖关系
  struct Edge {
    unsigned id;    // 目标节点 ID
    Value value;    // 依赖的 memref
  };

  // WHY 使用 DenseMap：O(1) 查找，节点 ID 通常是紧凑的整数
  DenseMap<unsigned, Node> nodes;
  DenseMap<unsigned, SmallVector<Edge>> inEdges;   // 入边 (谁是生产者)
  DenseMap<unsigned, SmallVector<Edge>> outEdges;  // 出边 (谁是消费者)
  Block &block;

  // 辅助函数：获取节点对特定 memref 的入边访问数
  unsigned getIncomingMemRefAccesses(unsigned nodeId, Value memref) const;

  // 辅助函数：获取节点对特定 memref 的出边数
  unsigned getOutEdgeCount(unsigned nodeId, Value memref) const;
};
```

**核心算法 1：`canFuseLoops` - 融合合法性检查**

```cpp
// 来源: LoopFusionUtils.cpp (352行核心融合检查函数)
// 返回值：FusionResult 枚举 (Success, FailBlock,FailPrecondition, etc.)
FusionResult canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                          unsigned dstLoopDepth,
                          FusionStrategy fusionStrategy,
                          ComputationSliceState *sliceUnion) {

  // === 步骤 1：基本结构检查 ===
  // WHY：确保融合不会破坏程序结构
  if (srcForOp->getParentRegion() != dstForOp->getParentRegion())
    return FusionResult::FailBlock;  // 必须在同一区域

  // === 步骤 2：依赖分析检查 ===
  // 获取 src 和 dst 之间的所有依赖
  SmallVector<DependenceComponent, 2> depComps;
  getDependenceComponents(srcForOp, dstForOp, &depComps);

  // WHY：检查循环携带依赖的方向
  // 如果有反向依赖，融合会破坏依赖关系
  for (auto &dep : depComps) {
    // WHY 检查 <= 0：依赖必须从 src 到 dst (正向)
    if (dep dependenceDirection == DependenceDirection::LT)
      return FusionResult::FailDependence;  // 反向依赖：不能融合
  }

  // === 步骤 3：计算融合切片 ===
  // WHY 切片：源循环可能只需要融合部分迭代
  // 例如：src 写 A[0:10]，dst 读 A[2:8]，只需要融合 src 的 [2:8] 部分
  ComputationSliceState slice;
  if (failed(computeSliceUnion(srcForOp, dstForOp, dstLoopDepth,
                                fusionStrategy, &slice)))
    return FusionResult::FailSliceComputation;

  // === 步骤 4：检查切片是否有效 ===
  // 确保 IV 映射是仿射变换
  if (!slice.isValid)
    return FusionResult::FailSliceInvalid;

  // === 步骤 5：逃逸分析 ===
  // 检查 src 写入的 memref 是否被外部使用
  DenseSet<Value> srcEscapingMemRefs;
  getEscapingMemRefs(srcForOp, &srcEscapingMemRefs);

  if (!srcEscappingMemRefs.empty() && !slice.isMaximal())
    return FusionResult::FailEscaping;  // 有逃逸且非最大切片：不安全

  // === 步骤 6：内存空间检查 ===
  for (Value memref : producerConsumerMemrefs) {
    if (memref.getType().cast<MemRefType>().getMemorySpace()
        != fastMemorySpace)
      return FusionResult::FailMemorySpaceMismatch;
  }

  // === 步骤 7：循环嵌套兼容性 ===
  // 确保 dst 的循环深度足够容纳切片
  unsigned dstNumLoops = getNumAffineForOps(dstForOp);
  if (dstLoopDepth > dstNumLoops)
    return FusionResult::FailLoopDepthExceeded;

  return FusionResult::Success;
}
```

**核心算法 2：`isFusionProfitable` - 融合收益分析**

```cpp
// 来源: LoopFusion.cpp (200+ 行的成本模型)
// 返回值：true 表示融合值得，storageReduction 输出存储节省百分比
static bool isFusionProfitable(
    AffineForOp srcForOp, AffineForOp dstForOp, unsigned dstLoopDepth,
    const ComputationSliceState &slice,
    std::optional<unsigned> fastMemorySpace,
    unsigned localBufSizeThreshold, double computeToleranceThreshold,
    double *storageReduction) {

  // === 步骤 1：计算原始内存占用 ===
  // 获取 src 和 dst 循环嵌套的内存足迹
  std::optional<int64_t> srcMemSize = getMemoryFootprintBytes(srcForOp);
  std::optional<int64_t> dstMemSize = getMemoryFootprintBytes(dstForOp);

  if (!srcMemSize || !dstMemSize)
    return false;  // 无法计算：保守地不融合

  // === 步骤 2：计算融合后的内存占用 ===
  // WHY 融合后可能更小：局部缓冲可以复用
  // 切片内存估计：只计算需要融合的部分
  std::optional<int64_t> sliceMemEstimate =
      getSliceMemoryFootprintBytes(srcForOp, dstForOp, slice, dstLoopDepth);

  if (!sliceMemEstimate)
    return false;

  auto fusedMem = *dstMemSize + *sliceMemEstimate;

  // === 步骤 3：内存收益检查 ===
  // WHY 融合必须减少内存占用
  // 如果融合后更大，说明局部性收益抵不上内存开销
  if (static_cast<long>(fusedMem) > *srcMemSize + *dstMemSize) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion not profitable: memory increases\n");
    return false;
  }

  // 计算存储节省百分比
  *storageReduction = 100.0 * (1.0 - fusedMem /
      (static_cast<double>(*srcMemSize) + *dstMemSize));

  // === 步骤 4：计算成本 ===
  uint64_t srcLoopNestCost = getComputeCost(srcForOp);
  uint64_t dstLoopNestCost = getComputeCost(dstForOp);
  uint64_t minFusedLoopNestCost =
      getFusedComputeCost(srcForOp, dstForOp, slice, dstLoopDepth);

  // === 步骤 5：计算额外计算百分比 ===
  // WHY 切片可能导致重复计算
  // 例如：src[i] 被多个 dst 迭代使用，切片可能重复计算
  double additionalComputeFraction =
      100.0 * (minFusedLoopNestCost /
               (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) - 1);

  LLVM_DEBUG({
    llvm::dbgs() << "Additional compute: " << additionalComputeFraction << "%\n";
    llvm::dbgs() << "Storage reduction: " << *storageReduction << "%\n";
  });

  // === 步骤 6：成本收益权衡 ===
  // 条件 1：额外计算在容忍范围内
  // 条件 2：存储节省足够大
  if (additionalComputeFraction > computeToleranceThreshold) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion not profitable: too much redundant compute\n");
    return false;
  }

  // WHY 有存储阈值：小缓冲区不值得分配
  if (fusedMem > localBufSizeThreshold * 1024) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion not profitable: buffer too large\n");
    return false;
  }

  return true;
}
```

**核心算法 3：`createPrivateMemRef` - 局部缓冲生成**

```cpp
// 来源: LoopFusion.cpp (局部 memref 创建)
// 创建融合后使用的局部缓冲区
static Value createPrivateMemRef(OpBuilder &b, Location loc, Value memref,
                                ArrayRef<Operation *> sliceOps,
                                AffineForOp dstForOp) {
  MemRefType memrefType = cast<MemRefType>(memref.getType());

  // === 步骤 1：确定局部缓冲的大小 ===
  // WHY：只分配实际需要的大小，节省内存
  SmallVector<int64_t, 4> privateShape;

  // 分析 sliceOps 对 memref 的访问模式
  // 例如：访问 A[i+1:i+10] → 大小为 9
  for (unsigned dim = 0; dim < memrefType.getRank(); ++dim) {
    int64_t dimSize = 1;  // 默认为标量

    // 检查该维度的访问范围
    std::optional<int64_t> range = getAccessRange(sliceOps, memref, dim);
    if (range)
      dimSize = *range;
    else
      dimSize = memrefType.getShape()[dim];  // 保守：使用原始大小

    privateShape.push_back(dimSize);
  }

  // === 步骤 2：创建新的 memref 类型 ===
  // WHY 使用静态形状：编译期可知，更好的优化
  MemRefType privateType =
      MemRefType::Builder(memrefType).setShape(privateShape);

  // === 步骤 3：分配缓冲区 ===
  // 在 dstForOp 之前分配 (WHY：避免每次迭代都分配)
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(dstForOp);

  Value privateMemref = b.create<memref::AllocOp>(loc, privateType);

  // === 步骤 4：处理快速内存空间 ===
  // 如果指定了 fastMemorySpace，在快速内存中分配
  if (fastMemorySpace) {
    privateType = MemRefType::Builder(privateType)
                      .setMemorySpace(*fastMemorySpace);
    privateMemref = b.create<memref::AllocOp>(loc, privateType);
  }

  return privateMemref;
}
```

**核心算法 4：`performFusionsIntoDest` - 贪心融合主循环**

```cpp
// 来源: LoopFusion.cpp (300+ 行的融合执行逻辑)
void GreedyFusion::performFusionsIntoDest(unsigned dstId,
                                         unsigned maxSrcUserCount) {
  // === 前置检查 ===
  if (mdg->nodes.count(dstId) == 0)
    return;  // 节点已被移除 (之前融合过)

  auto *dstNode = mdg->getNode(dstId);
  if (!isa<AffineForOp>(dstNode->op))
    return;  // 只处理循环嵌套

  if (dstNode->op->getNumResults() > 0)
    return;  // TODO: 不支持有返回值的循环

  // === 循环变换准备 ===
  // WHY 下沉顺序循环：增加融合深度
  // 顺序循环下移后，并行循环上移，可以在更深层融合
  sinkSequentialLoops(dstNode);
  auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

  // === 贪心融合循环 ===
  bool dstNodeChanged;
  do {
    dstNodeChanged = false;

    // 收集所有生产者候选
    SmallVector<unsigned, 16> srcIdCandidates;
    getProducerCandidates(dstId, *mdg, srcIdCandidates);

    // WHY 反向遍历：程序序的逆序，减少迭代次数
    for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
      auto *srcNode = mdg->getNode(srcId);
      auto srcAffineForOp = cast<AffineForOp>(srcNode->op);

      // === 检查 1：用户数限制 ===
      // WHY：如果 memref 被多个消费者使用，融合可能不划算
      for (Value memref : getProducerConsumerMemrefs(srcId, dstId, *mdg)) {
        if (mdg->getOutEdgeCount(srcId, memref) > maxSrcUserCount)
          continue;  // 跳过：用户太多
      }

      // === 检查 2：逃逸 memref ===
      DenseSet<Value> srcEscapingMemRefs;
      getEscapingMemRefs(srcNode->op, &srcEscapingMemRefs);

      // === 检查 3：融合深度搜索 ===
      // 尝试不同的融合深度，找到最优的
      std::optional<unsigned> bestDstLoopDepth;
      ComputationSliceState bestSlice;
      double bestStorageReduction = 0.0;

      // WHY 从深到浅搜索：深层融合通常收益更大
      for (unsigned dstLoopDepth = getNumAffineForOps(dstAffineForOp);
           dstLoopDepth > 0; --dstLoopDepth) {

        ComputationSliceState slice;
        if (failed(computeSlice(srcAffineForOp, dstAffineForOp,
                               dstLoopDepth, &slice)))
          continue;

        // 检查融合收益
        double storageReduction;
        if (!isFusionProfitable(srcAffineForOp, dstAffineForOp, dstLoopDepth,
                               slice, fastMemorySpace,
                               localBufSizeThreshold,
                               computeToleranceThreshold,
                               &storageReduction))
          continue;  // 不划算：跳过

        // 更新最优深度
        if (storageReduction > bestStorageReduction) {
          bestStorageReduction = storageReduction;
          bestDstLoopDepth = dstLoopDepth;
          bestSlice = slice;
        }
      }

      // === 执行融合 ===
      if (bestDstLoopDepth) {
        // 检查是否可以创建局部缓冲
        DenseMap<Value, Value> privateMemRefs;
        for (Value memref : getProducerConsumerMemrefs(srcId, dstId, *mdg)) {
          if (canCreatePrivateMemRef(memref, srcEscapingMemRefs,
                                    srcId, dstId,
                                    /*removeSrcNode=*/true)) {
            // 创建局部缓冲
            Value privateMemref = createPrivateMemRef(
                b, srcAffineForOp.getLoc(), memref,
                bestSlice.sliceOps, dstAffineForOp);
            privateMemRefs[memref] = privateMemref;
          }
        }

        // 执行实际的融合操作
        FusionResult result = fuseLoops(srcAffineForOp, dstAffineForOp,
                                       *bestDstLoopDepth, &bestSlice,
                                       privateMemRefs);

        if (success(result)) {
          // 更新依赖图
          mdg->updateAfterFusion(srcId, dstId, bestSlice, privateMemRefs);

          // 检查是否可以删除源循环
          if (canRemoveSrcNodeAfterFusion(srcId, dstId, bestSlice,
                                         dstAffineForOp, srcEscapingMemRefs,
                                         *mdg)) {
            mdg->eraseNode(srcId);  // 从图中删除
            srcAffineForOp.erase();  // 删除 IR
          }

          dstNodeChanged = true;  // 继续迭代：可能有新的融合机会
        }
      }
    }
  } while (dstNodeChanged);  // 直到不动点
}
```

**执行流程示例：矩阵乘法融合**

```cpp
// === 原始代码 ===
// 循环 1：A * B^T = C (生产者)
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    %v = affine.load %A[%i, %j] : memref<1024x1024xf32>
    affine.store %v, %C[%i, %j] : memref<1024x1024xf32>
  }
}

// 循环 2：C + D = E (消费者)
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    %v1 = affine.load %C[%i, %j] : memref<1024x1024xf32>
    %v2 = affine.load %D[%i, %j] : memref<1024x1024xf32>
    %r = arith.addf %v1, %v2 : f32
    affine.store %r, %E[%i, %j] : memref<1024x1024xf32>
  }
}

// === 融合后代码 ===
affine.for %i = 0 to 1024 {
  affine.for %j = 0 to 1024 {
    // 生产者代码融合进来
    %v = affine.load %A[%i, %j] : memref<1024x1024xf32>
    %c_local = affine.load %D[%i, %j] : memref<1024x1024xf32>
    %tmp = arith.addf %v, %c_local : f32

    // 消费者代码融合进来
    affine.store %tmp, %E[%i, %j] : memref<1024x1024xf32>
  }
}
// 注意：C 被完全消除了！
```

**WHY 融合成功？**

1. **依赖检查**：循环 1 是 C 的生产者，循环 2 是消费者 → 正向依赖
2. **切片计算**：两个循环范围相同 [0, 1024) → 最大切片
3. **内存收益**：C 是 1024×1024×4 = 4MB，融合后为 0 → 100% 节省
4. **计算成本**：没有额外计算 → 0% 冗余

### 5.3 循环展开 (Loop Unroll)

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/LoopUnroll.cpp` (156 行)
- **头文件：** `mlir/include/mlir/Dialect/Affine/Passes.h`
- **测试文件：** `mlir/test/Dialect/Affine/unroll.mlir`

#### WHAT：循环展开是什么？

**循环展开** (Loop Unrolling) 是一种编译器优化技术，通过 **复制循环体多次** 来减少循环控制开销。

**示例：**

**展开前：**

```text
affine.for %i = 0 to 12 {
  %v = affine.load %A[%i] : memref<12xf32>
  "use"(%v) : (f32) -> ()
}
```

**展开 4 倍后：**

```text
affine.for %i = 0 to 12 step 4 {
  %v0 = affine.load %A[%i] : memref<12xf32>
  "use"(%v0) : (f32) -> ()
  %v1 = affine.load %A[%i + 1] : memref<12xf32>
  "use"(%v1) : (f32) -> ()
  %v2 = affine.load %A[%i + 2] : memref<12xf32>
  "use"(%v2) : (f32) -> ()
  %v3 = affine.load %A[%i + 3] : memref<12xf32>
  "use"(%v3) : (f32) -> ()
}
affine.for %i = 12 to 12 {  // 清理循环 (cleanup loop)
  // 处理剩余元素
}
```

#### WHY：为什么需要循环展开？

| 优势               | 解释                        | 收益来源              |
| ------------------ | --------------------------- | --------------------- |
| **减少分支开销**   | 循环条件检查减少 N 倍       | N 是展开因子          |
| **增加 ILP**       | 更多独立指令供 CPU 并行执行 | 现代 CPU 是超标量架构 |
| **改善寄存器重用** | 变量保存在寄存器中          | 减少内存访问          |
| **指令缓存友好**   | 更少的分支指令              | 减少流水线停顿        |

**WHY 不是展开越多越好？**

- **代码体积膨胀**：展开 8 倍 = 代码 8 倍
- **寄存器压力**：变量增多可能导致溢出到内存
- **指令缓存**：代码过大会导致缓存失效

#### 展开流程

**1. 顶层入口：`runOnOperation()`**

```
runOnOperation()                                                                                    
  │                                                                                                   
  ├── [约束] func.isExternal() → 直接返回，不处理外部函数声明                                         
  │                                                                            
  ├── 模式一：全展开 + threshold 模式（unrollFull && unrollFullThreshold 有值）
  │   │   trip count = ceil((ub - lb) / s)    // (上界 - 下界) / 步长，即实际执行次数
  │   │   目标：把 trip count 小的循环直接全部展开（用于测试外层循环展开）
  │   │
  │   ├── post-order walk 收集所有 forOp：
  │   │   ├── getConstantTripCount(forOp)
  │   │   │   └── [约束] trip count 必须是常量且 <= unrollFullThreshold 才收集
  │   │   └── 用 post-order 保证先收集内层再收集外层
  │   │       （避免外层展开时删掉已收集的内层）
  │   │
  │   ├── for each forOp: loopUnrollFull(forOp)
  │   └── return（不走模式二）
  │
  └── 模式二：按因子展开最内层循环（默认模式）
      │   目标：反复找最内层循环，按指定 factor 展开
      │
      ├── for i in range(numRepetitions)，或有 getUnrollFactor 回调时无限循环：
      │   ├── gatherInnermostLoops(func, loops)
      │   │   └── walk 所有 forOp → isInnermostAffineForOp() 过滤
      │   │       → body 内无嵌套 AffineForOp 的才算最内层
      │   │
      │   ├── [约束] loops 为空 → break
      │   │
      │   ├── for each forOp: runOnAffineForOp(forOp)
      │   │
      │   └── [约束] 本轮没有任何一个 forOp 展开成功 → break
      │         （防止死循环：有 getUnrollFactor 时靠此条件退出）
```

**2. 单个循环展开：`runOnAffineForOp(forOp)`**

```
// 优先级从高到低，命中即返回
runOnAffineForOp(forOp)
│
├── [优先级1] getUnrollFactor 回调不为 null
│     → loopUnrollByFactor(forOp, getUnrollFactor(forOp), cleanUpUnroll)
│       调用方自定义每个 forOp 的展开因子
│
├── [优先级2] unrollFull == true（且无 threshold，否则已在 runOnOperation 提前处理）
│     → loopUnrollFull(forOp)
│       将循环体完全复制 tripCount 次，消除循环
│
├── [优先级3] unrollUpToFactor == true
│     → loopUnrollUpToFactor(forOp, unrollFactor)
│       展开到最多 unrollFactor 次，不产生余数循环
│       （若 trip count < factor，则按实际 trip count 展开）
│
└── [默认] loopUnrollByFactor(forOp, unrollFactor, cleanUpUnroll)
      按固定 factor 展开，产生余数循环处理边界
```

**3. 辅助函数**

| 函数                                | 作用                                        |
| ----------------------------------- | ------------------------------------------- |
| `isInnermostAffineForOp(op)`        | walk body，遇到任何嵌套 AffineForOp 立即    |
| interrupt；未中断则说明是最内层     |                                             |
| `gatherInnermostLoops(func, loops)` | walk 整个 func，过滤出所有最内层 forOp      |
| `loopUnrollFull`                    | 将循环完全展开（外部实现，在 LoopUtils 中） |
| `loopUnrollByFactor`                | 按因子展开，生成余数循环（外部实现）        |
| `loopUnrollUpToFactor`              | 按因子展开但不超过 trip count（外部实现）   |

**4. 关键约束汇总**

| 约束                                         | 位置                        | 效果                                                 |
| -------------------------------------------- | --------------------------- | ---------------------------------------------------- |
| `func.isExternal()`                          | `runOnOperation()`          | 跳过函数声明                                         |
| `unrollFull && unrollFullThreshold` 同时成立 | `runOnOperation()`          | 进入 threshold 模式，只展开                          |
| trip count ≤ threshold 的循环                |                             |                                                      |
| trip count 必须是常量                        | threshold 模式收集阶段      | 无法静态确定 trip count 的循环不被收集               |
| post-order walk                              | threshold 模式              | 保证内层先于外层被收集，避免外层展开时内层 op 被删除 |
| `loops.empty()`                              | 模式二循环体                | 无最内层循环时退出迭代                               |
| 本轮无任何展开成功                           | 模式二循环体                | 防止无限循环（尤其是有 getUnrollFactor 回调时）      |
| `getUnrollFactor` 回调优先级最高             | `runOnAffineForOp()`        | 外部回调覆盖所有命令行参数                           |
| `cleanUpUnroll` flag                         | `loopUnrollByFactor` 调用处 | 控制展开后是否做清理（如消除多余的 iv 计算）         |

  #### 源代码级深度解析

**核心展开实现在 `LoopUtils.cpp` 中：**

```cpp
// 概念性实现 (实际代码更复杂)
LogicalResult loopUnrollByFactor(AffineForOp forOp, uint64_t unrollFactor,
                                 ...) {
  // === 步骤 1：检查前置条件 ===
  std::optional<uint64_t> tripCount = getConstantTripCount(forOp);
  if (!tripCount)
    return failure();  // 非常量 tripCount：无法展开

  if (*tripCount == 1)
    return success();  // 单次迭代：无需展开

  // === 步骤 2：计算清理循环下界 ===
  AffineMap cleanupLbMap;
  SmallVector<Value, 4> cleanupLbOperands;
  getCleanupLoopLowerBound(forOp, unrollFactor,
                          cleanupLbMap, cleanupLbOperands);

  // === 步骤 3：生成主循环 (展开版本) ===
  // WHY 使用 step * unrollFactor：跳过已展开的迭代
  AffineForOp mainLoop;
  if (*tripCount >= unrollFactor) {
    mainLoop = replaceForOpWithNewLoop(affine.for, /*lb=*/forOp.getLowerBound(),
                                       /*ub=*/cleanupLbMap,
                                       /*step=*/forOp.getStep() * unrollFactor);

    // 在主循环体内展开 unrollFactor 次
    forOp.getBody()->clear();
    for (uint64_t i = 0; i < unrollFactor; ++i) {
      // 克隆循环体
      IRMapping mapper;
      forOp.getBody()->cloneInto(mainLoop.getBody(), mapper);

      // 调整 IV 使用：iv, iv+1, iv+2, ...
      forOperation *clone = ...;
      for (Operation *op : clone) {
        for (Value operand : op.getOperands()) {
          if (operand == forOp.getInductionVar()) {
            // 替换为 iv + i * step
            Value adjustedIV = createAffineApplyOp(iv, i, step);
            operand.replaceAllUsesWith(adjustedIV);
          }
        }
      }
    }
  }

  // === 步骤 4：生成清理循环 (剩余迭代) ===
  // WHY：当 tripCount % unrollFactor != 0 时需要
  if (cleanupLbMap && *tripCount % unrollFactor != 0) {
    AffineForOp cleanupLoop = replaceForOpWithNewLoop(
        affine.for, /*lb=*/cleanupLbMap, /*ub=*/forOp.getUpperBound(),
        /*step=*/forOp.getStep());
    // 移动循环体到清理循环
  }

  // === 步骤 5：处理 iter_args (归约变量) ===
  // WHY：归约变量需要在迭代间传递
  if (forOp.getNumIterOperands() > 0) {
    // 主循环：每次展开需要正确传递 iter_arg
    // 清理循环：使用主循环的最终结果作为初始值
  }

  // === 步骤 6：删除原循环 ===
  forOp.erase();

  return success();
}
```

**展开示例：tripCount = 10, unrollFactor = 4**

```text
// === 原始循环 ===
affine.for %i = 0 to 10 {
  %v = affine.load %A[%i] : memref<10xf32>
  "use"(%v) : (f32) -> ()
}

// === 展开后 ===
// 主循环：处理 0-8 (step = 4)
affine.for %i = 0 to 8 step 4 {
  // 迭代 0
  %v0 = affine.load %A[%i] : memref<10xf32>
  "use"(%v0) : (f32) -> ()
  // 迭代 1
  %v1 = affine.load %A[%i + 1] : memref<10xf32>
  "use"(%v1) : (f32) -> ()
  // 迭代 2
  %v2 = affine.load %A[%i + 2] : memref<10xf32>
  "use"(%v2) : (f32) -> ()
  // 迭代 3
  %v3 = affine.load %A[%i + 3] : memref<10xf32>
  "use"(%v3) : (f32) -> ()
}

// 清理循环：处理 8-10
affine.for %i = 8 to 10 {
  %v = affine.load %A[%i] : memref<10xf32>
  "use"(%v) : (f32) -> ()
}
```

**`getCleanupLoopLowerBound` - 计算清理循环边界**

```cpp
// 来源: LoopUtils.cpp (43-98 行)
// 计算展开后清理循环的下界 (也是主循环的上界)
static void getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
                                     AffineMap &cleanupLbMap,
                                     SmallVectorImpl<Value> &cleanupLbOperands) {
  // === 步骤 1：获取 trip count ===
  AffineMap tripCountMap;
  SmallVector<Value, 4> tripCountOperands;
  getTripCountMapAndOperands(forOp, &tripCountMap, &tripCountOperands);

  if (!tripCountMap) {
    cleanupLbMap = AffineMap();  // 无法计算：返回空
    return;
  }

  OpBuilder b(forOp);
  auto lbMap = forOp.getLowerBoundMap();
  auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap,
                                    forOp.getLowerBoundOperands());

  // === 步骤 2：为每个上界表达式计算"bump" ===
  // WHY 处理多个上界：affine.for 可以有 min(ub1, ub2, ...)
  // 示例：for i = 0 to min(100, N) step 1
  SmallVector<AffineExpr, 4> bumpExprs(tripCountMap.getNumResults());
  SmallVector<Value, 4> bumpValues(tripCountMap.getNumResults());
  int64_t step = forOp.getStepAsInt();

  for (unsigned i = 0, e = tripCountMap.getNumResults(); i < e; i++) {
    auto tripCountExpr = tripCountMap.getResult(i);

    // WHY 减去余数：向下取整到 unrollFactor 的倍数
    // 例如：tripCount = 10, unrollFactor = 4
    //       bump = (10 - 10 % 4) * step = 8, 此处的step是展开之前的step
    bumpExprs[i] = (tripCountExpr - tripCountExpr % unrollFactor) * step;

    auto bumpMap = AffineMap::get(tripCountMap.getNumDims(),
                                  tripCountMap.getNumSymbols(), bumpExprs[i]);
    // 创建下界=8的AffineApplyOp，清理循环的范围是[8, 10)
    bumpValues[i] = b.create<AffineApplyOp>(forOp.getLoc(), bumpMap,
                                             tripCountOperands);
  }

  // === 步骤 3：构建清理循环下界映射 ===
  // cleanupLb = lb + bump1 + bump2 + ...
  SmallVector<AffineExpr, 4> newUbExprs(tripCountMap.getNumResults());
  for (unsigned i = 0, e = bumpExprs.size(); i < e; i++)
    newUbExprs[i] = b.getAffineDimExpr(0) + b.getAffineDimExpr(i + 1);

  // 重新构建新的操作数
  cleanupLbOperands.clear();
  cleanupLbOperands.push_back(lb); // 原始的下界0
  cleanupLbOperands.append(bumpValues.begin(), bumpValues.end()); // 能够整除展开因子的偏移量值
  // 得到清理循环的下界就是 0+8=8
  cleanupLbMap = AffineMap::get(1 + tripCountMap.getNumResults(), 0,
                                newUbExprs, b.getContext());

  // === 步骤 4：简化映射 ===
  // WHY：合并常量，消除冗余操作
  fullyComposeAffineMapAndOperands(&cleanupLbMap, &cleanupLbOperands);
  cleanupLbMap = simplifyAffineMap(cleanupLbMap);
  canonicalizeMapAndOperands(&cleanupLbMap, &cleanupLbOperands);

  // 清理死代码
  for (auto v : bumpValues)
    if (v.use_empty())
      v.getDefiningOp()->erase();
  if (lb.use_empty())
    lb.erase();
}
```

### 5.4 循环并行（Affine Parallel）

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/AffineParallelize.cpp` (95 行)
- **测试文件：** `mlir/test/Dialect/Affine/parallelize.mlir`

#### WHAT：并行化 Pass 是什么？

将 `affine.for` 循环转换为 `affine.parallel` 操作，使其能够并行执行。

**变换前：**

```text
affine.for %i = 0 to 100 {
  %v = affine.load %A[%i] : memref<100xf32>
  %r = arith.addf %v, %cst : f32
  affine.store %r, %B[%i] : memref<100xf32>
}
```

**变换后：**

```text
affine.parallel (%i) = (0) to (100) {
  %v = affine.load %A[%i] : memref<100xf32>
  %r = arith.addf %v, %cst : f32
  affine.store %r, %B[%i] : memref<100xf32>
}
```

#### WHY：为什么需要并行化？

| 收益          | 解释                  |
| ------------- | --------------------- |
| **多核利用**  | 现代 CPU 有多个核心   |
| **SIMD 友好** | 并行循环更容易向量化  |
| **GPU 映射**  | 可直接映射到 GPU 线程 |

#### HOW：实现解析

```cpp
// 来源: AffineParallelize.cpp (62-94 行)
void AffineParallelize::runOnOperation() {
  func::FuncOp f = getOperation();

  // === 步骤 1：收集可并行化的循环 ===
  // WHY 前序遍历：先处理外层循环，控制嵌套深度
  std::vector<ParallelizationCandidate> parallelizableLoops;
  f.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
    SmallVector<LoopReduction> reductions;

    // 检查循环是否可并行化
    // 如果支持归约，同时检测归约模式
    if (isLoopParallel(loop, parallelReductions ? &reductions : nullptr))
      parallelizableLoops.emplace_back(loop, std::move(reductions));
  });

  // === 步骤 2：执行并行化 (控制嵌套深度) ===
  for (const ParallelizationCandidate &candidate : parallelizableLoops) {
    unsigned numParentParallelOps = 0;
    AffineForOp loop = candidate.loop;

    // 计算父级中已有的 parallel 操作数
    // WHY 遍历到 AffineScope：只计算直接父级
    for (Operation *op = loop->getParentOp();
         op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
         op = op->getParentOp()) {
      if (isa<AffineParallelOp>(op))
        ++numParentParallelOps;
    }

    // WHY 限制嵌套深度：
    // 1. 避免过度并行化 (线程创建开销)
    // 2. 硬件限制 (如 GPU 的 grid/block 层级)
    // 3. 编译器/运行时限制
    if (numParentParallelOps < maxNested) {
      if (failed(affineParallelize(loop, candidate.reductions))) {
        LLVM_DEBUG(llvm::dbgs() << "failed to parallelize\n" << loop);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "too many nested parallel loops\n" << loop);
    }
  }
}
```

**并行化判定：`isLoopParallel`**

```cpp
// 来源: LoopAnalysis.cpp (依赖分析)
// 检查循环是否可以安全地并行化
bool isLoopParallel(AffineForOp forOp,
                    SmallVectorImpl<LoopReduction> *parallelReductions) {
  // === 检查 1：内存依赖 ===
  // 如果有任何循环携带依赖，不能并行化
  // WHY：依赖意味着迭代顺序重要
  if (!isLoopMemoryParallel(forOp))
    return false;

  // === 检查 2：iter_args (归约变量) ===
  // WHY 归约特殊处理：
  // 虽然有跨迭代依赖，但可以通过原子操作或归约原语实现
  if (forOp.getNumIterOperands() > 0) {
    if (!parallelReductions)
      return false;  // 不支持归约：不能并行化

    // 检查每个 iter_arg 是否是归约模式
    for (unsigned i = 0, e = forOp.getNumIterOperands(); i < e; ++i) {
      Value iterArg = forOp.getRegionIterArg(i);
      ValueOperand operand = forOp.getIterOperands()[i];

      // 分析 yield 操作
      SmallVector<Operation *, 4> yieldUsers;
      for (Operation *user : iterArg.getUsers())
        if (auto affineIf = dyn_cast<AffineIfOp>(user))
          yieldUsers.append(affineIf.getBody()->begin(),
                           affineIf.getBody()->end());

      // 检查是否是归约模式
      LoopReduction reduction;
      if (isReductionLoop(iterArg, operand, yieldUsers, &reduction)) {
        parallelReductions->push_back(reduction);
      } else {
        return false;  // 不是归约：不能并行化
      }
    }
  }

  return true;
}
```

**内存并行性检查 - `isLoopMemoryParallel`** 

```cpp
// 来源: AffineAnalysis.cpp (内存依赖分析)
// 检查循环是否有循环携带的内存依赖
bool isLoopMemoryParallel(AffineForOp forOp) {
  // 收集循环中的所有内存访问操作
  SmallVector<Operation *, 4> loads;
  SmallVector<Operation *, 4> stores;

  for (Operation &op : *forOp.getBody()) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op))
      loads.push_back(&op);
    else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op))
      stores.push_back(&op);
  }

  // 检查所有 store-load 对
  for (Operation *store : stores) {
    for (Operation *load : loads) {
      // 获取依赖向量
      SmallVector<DependenceComponent, 2> depComps;
      llvm::Optional<unsigned> commonLoopDepth =
          getCommonLoopDepth(forOp, cast<AffineReadOpInterface>(load),
                            cast<AffineWriteOpInterface>(store));

      // 检查依赖方向
      DependenceResult result = checkDependence(
          cast<AffineReadOpInterface>(load),
          cast<AffineWriteOpInterface>(store),
          /*loopDepth=*/commonLoopDepth ? *commonLoopDepth : 1,
          &depComps);

      if (result.hasValue()) {
        // 检查是否有循环携带依赖
        // WHY：如果有任何组件是 LT/GT (不是 EQ)，则有序依赖
        for (const auto &dep : depComps) {
          if (dep dependenceDirection == DependenceDirection::LT ||
              dep.dependenceDirection == DependenceDirection::GT) {
            // 找到序依赖：不能并行化
            return false;
          }
        }
      }
    }
  }

  return true;
}
```

**归约处理：**

支持的归约操作 (`AtomicRMWKind`)：

- `add`, `minimum`, `maximum`, `andi`, `ori`, `xori`

**归约检测示例：**

```cpp
// === 归约循环 ===
// 可以并行化：最终结果是所有迭代的总和
%sum = affine.for %i = 0 to 100 iter_args(%arg0 = %c0) -> f32 {
  %v = affine.load %A[%i] : memref<100xf32>
  %new = arith.addf %arg0, %v : f32
  affine.yield %new : f32
} // 返回总和

// === 非归约循环 ===
// 不能并行化：每次迭代依赖前一次的结果
%fib = affine.for %i = 0 to 100 iter_args(%arg0 = %c0, %arg1 = %c1) -> (i32, i32) {
  %next = arith.addi %arg0, %arg1 : i32
  affine.yield %arg1, %next : i32, i32
} // 斐波那契数列
```

**并行化变换示例：**

```text
// === 变换前：串行循环 ===
affine.for %i = 0 to 1024 {
  %v = affine.load %A[%i] : memref<1024xf32>
  %r = arith.addf %v, %cst : f32
  affine.store %r, %B[%i] : memref<1024xf32>
}

// === 变换后：并行循环 ===
affine.parallel (%i) = (0) to (1024) {
  %v = affine.load %A[%i] : memref<1024xf32>
  %r = arith.addf %v, %cst : f32
  affine.store %r, %B[%i] : memref<1024xf32>
}
```

**WHY affine.parallel 更适合并行执行？**

1. **明确的并行语义**：不保证迭代顺序
2. **减少同步**：不需要屏障
3. **编译器友好**：更容易映射到硬件线程



### 5.5 数据传输流水线（Data Transfer Pipeline）

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/PipelineDataTransfer.cpp` (380 行)
- **测试文件：** `mlir/test/Dialect/Affine/pipeline-data-transfer.mlir`

#### WHAT：数据传输流水线是什么？

**目标：** 重叠 DMA (Direct Memory Access) 数据传输与计算。

**场景：** 加速器/异构计算中，数据需要在不同内存层级间传输：

- CPU 主机内存 ↔ GPU 设备内存
- DRAM ↔ 片上缓存 (SRAM)

**WHY 需要流水线？**

- DMA 传输是异步的
- 可以在传输数据的同时处理之前的数据

#### HOW：实现解析

**核心函数：`doubleBuffer`**

```cpp
// 来源: PipelineDataTransfer.cpp (75-136 行)
// 将 memref 扩展为 2 倍大小，第一维度作为缓冲区索引
static bool doubleBuffer(Value oldMemRef, AffineForOp forOp) {
  auto *forBody = forOp.getBody();
  OpBuilder bInner(forBody, forBody->begin());

  // === 步骤 1：修改 memref 形状 ===
  // WHY 添加前导维度 2：双缓冲需要两个独立的缓冲区
  auto doubleShape = [&](MemRefType oldMemRefType) -> MemRefType {
    ArrayRef<int64_t> oldShape = oldMemRefType.getShape();
    SmallVector<int64_t, 4> newShape(1 + oldMemRefType.getRank());
    newShape[0] = 2;  // 双缓冲：索引 0 和 1
    std::copy(oldShape.begin(), oldShape.end(), newShape.begin() + 1);
    return MemRefType::Builder(oldMemRefType).setShape(newShape).setLayout({});
  };

  auto oldMemRefType = cast<MemRefType>(oldMemRef.getType());
  auto newMemRefType = doubleShape(oldMemRefType);

  // === 步骤 2：分配新的双缓冲 memref ===
  // WHY 在循环外分配：避免每次迭代都分配
  OpBuilder bOuter(forOp);
  SmallVector<Value, 4> allocOperands;

  // 处理动态维度
  for (const auto &dim : llvm::enumerate(oldMemRefType.getShape())) {
    if (dim.value() == ShapedType::kDynamic)  // -1 表示动态
      allocOperands.push_back(bOuter.createOrFold<memref::DimOp>(
          forOp.getLoc(), oldMemRef, dim.index()));
  }

  // 在 forOp 之前创建分配
  Value newMemRef = bOuter.create<memref::AllocOp>(
      forOp.getLoc(), newMemRefType, allocOperands);

  // === 步骤 3：创建 "iv mod 2" 索引 ===
  // WHY 使用 mod 2：在两个缓冲区之间交替
  // 迭代 0 → 索引 0, 迭代 1 → 索引 1, 迭代 2 → 索引 0, ...
  auto d0 = bInner.getAffineDimExpr(0);
  int64_t step = forOp.getStepAsInt();
  auto modTwoMap =
      AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                     d0.floorDiv(step) % 2);

  // 在循环体开始创建 affine.apply 操作
  auto ivModTwoOp = bInner.create<AffineApplyOp>(
      forOp.getLoc(), modTwoMap, forOp.getInductionVar());

  // === 步骤 4：替换所有 memref 使用 ===
  // WHY 需要支配过滤器：确保替换后的操作仍然合法
  auto userFilterFn = [&](Operation *user) {
    auto domInfo = std::make_unique<DominanceInfo>(
        forOp->getParentOfType<FunctionOpInterface>());
    return domInfo->dominates(&*forOp.getBody()->begin(), user);
  };

  if (failed(replaceAllMemRefUsesWith(oldMemRef, newMemRef,
                                      /*extraIndices=*/{ivModTwoOp},
                                      /*indexRemap=*/AffineMap(),
                                      /*extraOperands=*/{},
                                      /*symbolOperands=*/{},
                                      userFilterFn))) {
    // 替换失败：回滚
    LLVM_DEBUG(forOp.emitError("memref replacement failed"));
    ivModTwoOp.erase();
    return false;
  }

  // === 步骤 5：插入 dealloc ===
  // WHY 在循环后释放：双缓冲在整个循环期间都有效
  bOuter.setInsertionPointAfter(forOp);
  bOuter.create<memref::DeallocOp>(forOp.getLoc(), newMemRef);

  return true;
}
```

**WHY 使用 `iv mod 2` 而不是布尔标志？**

```cpp
// 方案 1：使用 mod 2 (MLIR 采用)
// 优势：纯仿射表达式，可以静态分析
// 循环变量：0, 1, 2, 3, 4, 5, ...
// 缓冲索引：0, 1, 0, 1, 0, 1, ...
buffer[iv % 2]  // 仿射映射

// 方案 2：使用布尔标志 (伪代码)
// 劣势：需要条件分支，破坏仿射性质
bool flag = false;
for (int i = 0; i < N; i++) {
  buffer[flag ? 1 : 0] = ...;  // 条件选择
  flag = !flag;
}
```

**流水线变换流程：**

1. **识别 DMA start/wait 对**

   ```cpp
   findMatchingStartFinishInsts(forOp, startWaitPairs);
   ```

2. **对数据缓冲区双缓冲**

   ```cpp
   for (auto &pair : startWaitPairs) {
     Value oldMemRef = ...;
     doubleBuffer(oldMemRef, forOp);
   }
   ```

3. **对 tag 缓冲区双缓冲**

   ```cpp
   for (auto &pair : startWaitPairs) {
     Value oldTagMemRef = ...;
     doubleBuffer(oldTagMemRef, forOp);
   }
   ```

4. **操作倾斜 (Op Skewing)**

   - 将操作移到不同的迭代
   - DMA start 移到迭代 `i`
   - 计算移到迭代 `i+1`
   - DMA wait 移到迭代 `i+1`

**效果：**

```cpp
// 变换前
for (int i = 0; i < N; i++) {
  dma_start(tag[i]);   // 启动传输
  dma_wait(tag[i]);    // 等待完成
  compute(data[i]);    // 计算
}

// 变换后
dma_start(tag[0]);     // prologue: 启动第一次传输
for (int i = 0; i < N-1; i++) {
  dma_wait(tag[i]);     // 等待上一次传输
  compute(data[i]);     // 计算
  dma_start(tag[i+1]);  // 启动下一次传输 (与计算重叠)
}
dma_wait(tag[N-1]);     // epilogue: 等待最后一次传输
compute(data[N-1]);
```

**备注**：`Tag memref` 本质是一个**信号量/完成标志**的存储位置，DMA 硬件写入它表示传输完成，`dma_wait` 读取它来阻塞等待。

#### 源代码级深度解析

**核心算法 1：`findMatchingStartFinishInsts` - DMA 配对查找**

```cpp
// 来源: PipelineDataTransfer.cpp (175-242 行)
// 查找循环中所有匹配的 DMA start/wait 对
static void findMatchingStartFinishInsts(
    AffineForOp forOp,
    SmallVectorImpl<std::pair<Operation *, Operation *>> &startWaitPairs) {

  // === 步骤 1：收集 outgoing DMA 操作 ===
  // WHY 需要检查依赖：outgoing DMA 可能与 incoming DMA 冲突
  SmallVector<AffineDmaStartOp, 4> outgoingDmaOps;
  for (auto &op : *forOp.getBody()) {
    auto dmaStartOp = dyn_cast<AffineDmaStartOp>(op);
    if (dmaStartOp && dmaStartOp.isSrcMemorySpaceFaster())
      outgoingDmaOps.push_back(dmaStartOp);
  }

  SmallVector<Operation *, 4> dmaStartInsts, dmaFinishInsts;

  // === 步骤 2：收集所有 DMA 操作 ===
  for (auto &op : *forOp.getBody()) {
    // 收集 DMA wait 操作
    if (isa<AffineDmaWaitOp>(op)) {
      dmaFinishInsts.push_back(&op);
      continue;
    }

    auto dmaStartOp = dyn_cast<AffineDmaStartOp>(op);
    if (!dmaStartOp)
      continue;

    // === 步骤 3：只处理 incoming DMA ===
    // WHY：只有 incoming 可以流水线化
    // incoming：从慢内存传输到快内存 (可以与计算重叠)
    // outgoing：从快内存传输到慢内存 (通常不能重叠)
    if (!dmaStartOp.isDestMemorySpaceFaster())
      continue;

    // === 步骤 4：检查与 outgoing DMA 的依赖 ===
    // WHY 保守检查：避免数据竞争
    auto *it = outgoingDmaOps.begin();
    for (; it != outgoingDmaOps.end(); ++it) {
      if (it->getDstMemRef() == dmaStartOp.getSrcMemRef())
        break;  // 找到依赖：跳过
    }
    if (it != outgoingDmaOps.end())
      continue;

    // === 步骤 5：检查缓冲区逃逸 ===
    // WHY：如果缓冲区在循环外使用，不能安全地双缓冲
    auto memref = dmaStartOp.getOperand(dmaStartOp.getFasterMemPos());
    bool escapingUses = false;
    for (auto *user : memref.getUsers()) {
      // dealloc 可以忽略
      if (isa<memref::DeallocOp>(user))
        continue;

      // 检查使用是否在循环体内
      if (!forOp.getBody()->findAncestorOpInBlock(*user)) {
        LLVM_DEBUG(llvm::dbgs() << "can't pipeline: buffer escapes\n");
        escapingUses = true;
        break;
      }
    }

    if (!escapingUses)
      dmaStartInsts.push_back(&op);
  }

  // === 步骤 6：配对 start 和 wait 操作 ===
  // WHY：通过 tag memref 匹配
  for (auto *dmaStartOp : dmaStartInsts) {
    for (auto *dmaFinishOp : dmaFinishInsts) {
      if (checkTagMatch(cast<AffineDmaStartOp>(dmaStartOp),
                        cast<AffineDmaWaitOp>(dmaFinishOp))) {
        startWaitPairs.push_back({dmaStartOp, dmaFinishOp});
        break;
      }
    }
  }
}
```

**核心算法 3：`runOnAffineForOp` - 流水线主流程**

```cpp
// 来源: PipelineDataTransfer.cpp (247-380+ 行)
void PipelineDataTransfer::runOnAffineForOp(AffineForOp forOp) {
  // === 前置检查：trip count ===
  auto mayBeConstTripCount = getConstantTripCount(forOp);
  if (!mayBeConstTripCount) {
    LLVM_DEBUG(forOp.emitRemark("won't pipeline: unknown trip count"));
    return;
  }

  // === 步骤 1：查找 DMA start/wait 对 ===
  SmallVector<std::pair<Operation *, Operation *>, 4> startWaitPairs;
  findMatchingStartFinishInsts(forOp, startWaitPairs);

  if (startWaitPairs.empty()) {
    LLVM_DEBUG(forOp.emitRemark("No dma start/finish pairs\n"));
    return;
  }

  // === 步骤 2：对数据缓冲区双缓冲 ===
  for (auto &pair : startWaitPairs) {
    auto *dmaStartOp = pair.first;

    // 获取快速内存空间的 memref (目标缓冲区)
    Value oldMemRef = dmaStartOp->getOperand(
        cast<AffineDmaStartOp>(dmaStartOp).getFasterMemPos());

    if (!doubleBuffer(oldMemRef, forOp)) {
      LLVM_DEBUG(llvm::dbgs() << "double buffering failed\n");
      return;
    }

    // 清理旧的分配 (如果不再使用)
    if (auto *allocOp = oldMemRef.getDefiningOp()) {
      if (oldMemRef.use_empty()) {
        allocOp->erase();
      } else if (oldMemRef.hasOneUse()) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(*oldMemRef.user_begin())) {
          dealloc.erase();
          allocOp->erase();
        }
      }
    }
  }

  // === 步骤 3：对 tag 缓冲区双缓冲 ===
  for (auto &pair : startWaitPairs) {
    auto *dmaFinishOp = pair.second;
    Value oldTagMemRef = dmaFinishOp->getOperand(getTagMemRefPos(*dmaFinishOp));

    if (!doubleBuffer(oldTagMemRef, forOp)) {
      LLVM_DEBUG(llvm::dbgs() << "tag double buffering failed\n");
      return;
    }

    // 清理旧的 tag 分配
    if (auto *tagAllocOp = oldTagMemRef.getDefiningOp()) {
      if (oldTagMemRef.use_empty()) {
        tagAllocOp->erase();
      } else if (oldTagMemRef.hasOneUse()) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(*oldTagMemRef.user_begin())) {
          dealloc.erase();
          tagAllocOp->erase();
        }
      }
    }
  }

  // === 步骤 4：重新查找 (双缓冲后 IR 已改变) ===
  startWaitPairs.clear();
  findMatchingStartFinishInsts(forOp, startWaitPairs);

  // === 步骤 5：计算每个 Op 的时间偏移量，分三步 ===
  DenseMap<Operation *, unsigned> instShiftMap;
  // 第1步：DMA start 相关 op 设为 shift=0， instShiftMap[op] = 0
  for (auto &pair : startWaitPairs) {                                                                 
    instShiftMap[dmaStartOp] = 0;                                                                

    // 尝试把 dmaStartOp 的 affine 操作数提取成独立的 AffineApplyOp slice
    createAffineComputationSlice(dmaStartOp, &sliceOps);
    if (!sliceOps.empty()) {
        for (auto sliceOp : sliceOps)
            instShiftMap[sliceOp] = 0;  // slice 也是 shift=0
    } else {
        // 没有生成 slice，找所有可达的 affine.apply op
        getReachableAffineApplyOps(operands, affineApplyInsts);
        for (auto *op : affineApplyInsts)
            instShiftMap[op] = 0;
    }
  }
  
  // 第2步：其余所有 op 设为 shift=1
  for (auto &op : forOp.getBody()->without_terminator())
    // try_emplace 的关键：只在 key 不存在时插入，所以 shift=0 的 op 不会被改成 1。
    instShiftMap.try_emplace(&op, 1);  // 已在 map 中的（shift=0）不会被覆盖

  // 第3步：按顺序转成数组作为 affineForOpBodySkew 的输入，它只接受一个扁平数组（按位置索引），而不是 map
  for (auto &op : forOp.getBody()->without_terminator())
    shifts[s++] = instShiftMap[&op];  // 按 block 中的顺序读出

  
  // === 步骤 6：检查 shift 不违反依赖 ===
  if (!isOpwiseShiftValid(forOp, shifts)) {
    // Violates dependences.
    LLVM_DEBUG(llvm::dbgs() << "Shifts invalid - unexpected\n";);
    return;
  }

  // === 步骤 7：执行 Op 倾斜 ===
  // affineForOpBodySkew 会把原 forOp 删掉，替换为三段结构：
  // - prologue：执行 shift=0 的 op（第一次 dma_start）
  // - 主循环：每轮同时跑 shift=0（下一轮 dma_start）和 shift=1（当前轮 compute+wait）
  // - epilogue：执行最后一轮 shift=1 的 op
  if (failed(affineForOpBodySkew(forOp, shifts))) {
    LLVM_DEBUG(llvm::dbgs() << "op body skewing failed - unexpected\n";);
    return;
  }
}
```

**流水线变换后的 IR 结构**

```text
// === 原始循环 ===
affine.for %i = 0 to 100 {
  %tag = affine.dma_start %A[%i] to %B[%i], tag(%tag_buf[%i]) : memref<100xf32>
  affine.dma_wait %tag_buf[%i], %tag : memref<1xi32>
  %v = affine.load %B[%i] : memref<100xf32>
  "use"(%v) : (f32) -> ()
}

// === 流水线后 ===
// Prologue: 启动第一次传输
%tag_0 = affine.dma_start %A[0] to %B[0], tag(%tag_buf[0 mod 2]) : ...

// Steady-state: 主循环
affine.for %i = 0 to 99 {
  // 等待上一次传输 (已完成)
  affine.dma_wait %tag_buf[%i mod 2], %tag : ...

  // 使用已传输的数据
  %v = affine.load %B[%i] : ...
  "use"(%v) : ...

  // 启动下一次传输 (与计算重叠)
  %tag_next = affine.dma_start %A[%i + 1] to %B[%i + 1],
                             tag(%tag_buf[(%i + 1) mod 2]) : ...
}

// Epilogue: 等待最后一次传输并使用数据
affine.dma_wait %tag_buf[99 mod 2], %tag_99 : ...
%v_last = affine.load %B[99] : ...
"use"(%v_last) : ...
```

### 5.6 简化Min/Max（Simplify Min Max）

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/SimplifyAffineMinMax.cpp` (265 行)
- **测试文件：** `mlir/test/Dialect/Affine/simplify-min-max-ops.mlir`

#### WHAT：Min/Max 简化是什么？

简化 `affine.min` 和 `affine.max` 操作，通过 **边界分析** 消除不必要的比较。

**示例：**

**简化前：**

```text
// 已知：i >= 0
%v = affine.min affine_map<(d0) -> (d0, 0)>(%i)
// 结果总是 0，因为 i >= 0
```

**简化后：**

```text
%c0 = arith.constant 0 : index
// 直接使用常量 0
```

#### WHY：为什么需要简化？

| 收益               | 解释               |
| ------------------ | ------------------ |
| **消除运行时比较** | 编译期确定结果     |
| **减少指令数**     | min/max 操作被删除 |
| **便于后续优化**   | 常量更容易传播     |

#### HOW：实现解析

**关键思想：** 使用 **值边界分析** (Value Bounds Analysis) 确定变量间的偏序关系。

1.  **构建变量（每个 map result 对应一个 Variable）**

```
affine.min(s0, s0 + 4, 100)  →  variables = [s0, s0+4, 100]
```

2. **用并查集找"被同一个 bound 控制的等价类"**

  对每对 `(i, j)`，问约束求解器：`v_i < v_j`（对 min）是否恒成立？

  - 如果成立：`i` 和 `j` 合并到同一等价类，bound 保留较小的（对 min）
  - 最终如果所有 result 都在同一个等价类，说明找到了一个全局最小值

3. **替换**

```
// 如果能证明 s0 < s0+4 且 s0 < 100（在当前约束下）
affine.min(s0, s0+4, 100)  →  直接替换为 s0
```

**关键数据结构**

  - **IntEqClasses boundedClasses**：并查集，合并已知有公共 bound 的 result
  - **DenseMap<unsigned, Variable*> bounds**：每个等价类的当前 bound（最小值候选）
  - **ComparisonOperator**：min 用 LT，max 用 GT

**约束来源**

**ValueBoundsConstraintSet** 会收集 IR 中已知的约束（loop bounds、assume等），在这个上下文中判断大小关系是否恒成立。

#### 源码级深度解析

```cpp
template <typename AffineOp>
static bool simplifyAffineMinMaxOp(RewriterBase &rewriter, AffineOp affineOp) {
  using Variable = ValueBoundsConstraintSet::Variable;
  using ComparisonOperator = ValueBoundsConstraintSet::ComparisonOperator;

  AffineMap affineMap = affineOp.getMap();
  ValueRange operands = affineOp.getOperands();
  static constexpr bool isMin = std::is_same_v<AffineOp, AffineMinOp>;

  // 步骤 1：构建变量列表
  SmallVector<Variable> variables = llvm::map_to_vector(
      llvm::iota_range<unsigned>(0u, affineMap.getNumResults(), false),
      [&](unsigned i) {
        return Variable(affineMap.getSliceMap(i, 1), operands);
      });

  // 步骤 2：获取比较操作
  ComparisonOperator cmpOp = isMin ? ComparisonOperator::LT
                                    : ComparisonOperator::GT;

  // 步骤 3：使用并查集合并可比较的变量
  llvm::IntEqClasses boundedClasses(variables.size());
  DenseMap<unsigned, Variable *> bounds;

  for (auto &&[i, v] : llvm::enumerate(variables)) {
    unsigned eqClass = boundedClasses.findLeader(i);
    if (bounds.contains(eqClass))
      continue;

    Variable *bound = &v;

    // 检查与其他变量的关系
    for (size_t j = i + 1; j < variables.size(); ++j) {
      unsigned jEqClass = boundedClasses.findLeader(j);
      if (jEqClass == eqClass)
        continue;

      Variable *nv = bounds.lookup_or(jEqClass, &variables[j]);

      // 比较：bound < nv ?
      FailureOr<bool> cmpResult =
          ValueBoundsConstraintSet::strongCompare(*bound, cmpOp, *nv);

      if (failed(cmpResult))
        continue;  // 无法比较

      if (*cmpResult) {
        // bound < nv，合并
        boundedClasses.join(eqClass, jEqClass);
      } else {
        // bound >= nv，更新 bound
        bound = nv;
        boundedClasses.join(eqClass, jEqClass);
      }
    }
    bounds[boundedClasses.findLeader(i)] = bound;
  }

  // 步骤 4：如果成功简化，更新 affine map
  if (bounds.size() >= affineMap.getNumResults())
    return false;  // 没有简化

  SmallVector<AffineExpr> results;
  results.reserve(bounds.size());
  for (auto [k, bound] : bounds)
    results.push_back(bound->getMap().getResult(0));

  affineMap = AffineMap::get(0, affineMap.getNumSymbols() + affineMap.getNumDims(),
                             results, rewriter.getContext());

  rewriter.modifyOpInPlace(affineOp, [&]() { affineOp.setMap(affineMap); });
  return true;
}
```

### 5.7 向量化（Super Vectorize）

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/SuperVectorize.cpp` (~2500 行，最大的 Pass)
- **测试目录：** `mlir/test/Dialect/Affine/SuperVectorize/`

#### WHAT：超向量化是什么？

将循环和操作转换为 **n-D 向量操作**，利用 SIMD 指令 (如 AVX-512)。

**示例：**

**向量化前：**

```text
affine.for %i = 0 to 1024 {
  %v = affine.load %A[%i] : memref<1024xf32>
  %r = arith.addf %v, %cst : f32
  affine.store %r, %B[%i] : memref<1024xf32>
}
```

**向量化后 (向量宽度 4)：**

```text
affine.for %i = 0 to 1024 step 4 {
  %v = vector.load %A[%i] : memref<1024xf32>, vector<4xf32>
  %r = arith.addf %v, %bcst : vector<4xf32>
  vector.store %r, %B[%i] : memref<1024xf32>, vector<4xf32>
}
```

#### WHY：为什么需要向量化？

| 收益           | 解释                         |
| -------------- | ---------------------------- |
| **SIMD 利用**  | 一条指令处理多个数据         |
| **带宽效率**   | 减少内存访问次数             |
| **吞吐量提升** | 4-16 倍性能提升 (取决于硬件) |

#### 向量化策略

**1. 平铺向量化 (Tiled Vectorization)**

- 适用于：规则嵌套循环
- 策略：将循环空间划分为 **tiles**
- 示例：矩阵乘法分块后向量化

**2. 最外层向量化 (Outer Loop Vectorization)**

- 适用于：最外层循环可并行
- 策略：向量化最外层循环
- 优势：减少循环开销

**3. 收缩减轻 (Reduction Vectorization)**

- 适用于：归约操作
- 策略：向量归约 + 树形归约
- 示例：`sum += A[i]` → 向量加 + 水平归约

#### 向量化流程

**1. 入口：`runOnOperation()`**

```
runOnOperation                                                                                      
  │
  ├─ [约束] fastestVaryingPattern.size() != vectorSizes.size() → 失败                               
  ├─ [约束] any(vectorSizes <= 0) → 失败                                                          
  ├─ [约束] vectorizeReductions && vectorSizes.size() != 1 → 失败
  │
  ├─ walk FuncOp，收集 parallel loops
  ├─ 若 vectorizeReductions：识别规约循环，构建 ReductionLoopMap
  │
  └─ vectorizeLoops
       │
       ├─ makePattern
       │    ├─ [约束] vectorRank ∈ {1,2,3}
       │    └─ isVectorizableLoopPtrFactory（用 fastestVaryingPattern 过滤合法循环）
       │
       ├─ pattern->match()（找所有满足 pattern 的嵌套循环组合）
       │
       ├─ computeIntersectionBuckets
       │    └─ 把有祖先/后代关系的 match 放同一桶，避免冲突
       │
       └─ for each 桶中互不相交的 match:
            │
            ├─ build VectorizationStrategy（vectorSizes + reductionLoops）
            │
            ├─ analyzeProfitability（递归）
            │    └─ vectorizeLoopIfProfitable
            │         └─ [条件] patternDepth - depthInPattern <= vectorSizes.size()
            │              → 为该层循环分配向量维度
            │
            └─ vectorizeRootMatch
                 │
                 ├─ getMatchedAffineLoops（从 NestedMatch 提取 AffineForOp 层次）
                 │
                 └─ vectorizeLoopNest
                      │
                      ├─ [约束] !isVectorizableLoopBody → 失败
                      │
                      ├─ walk PreOrder → vectorizeOneOperation（逐 op dispatch）
                      │    ├─ AffineForOp → vectorizeAffineForOp
                      │    │    ├─ [约束] isVecDim && iterOperands>0 && step!=1 → 失败
                      │    │    ├─ newStep = step * vectorFactor
                      │    │    └─ 规约维度：createInitialVector + createMask
                      │    ├─ AffineLoadOp  → vectorizeAffineLoad  → vector.transfer_read
                      │    ├─ AffineStoreOp → vectorizeAffineStore → vector.transfer_write
                      │    │    ├─ [约束] operands 含 AffineApplyOp → 失败
                      │    │    ├─ [约束] isIVMappedToMultipleIndices → 失败
                      │    │    └─ [约束] 无法构建 permutationMap → 失败
                      │    ├─ AffineYieldOp → vectorizeAffineYieldOp
                      │    ├─ ConstantOp   → vectorizeConstant（splat）
                      │    ├─ AffineApplyOp→ vectorizeAffineApplyOp
                      │    ├─ 有 Region 的其他 op → 失败
                      │    └─ 其他 → widenOp
                      │         └─ vectorizeOperand（按优先级：缓存→常量→均匀值→失败）
                      │
                      ├─ 成功 → 替换规约结果（vector.reduce）→ erase scalar loop
                      └─ 失败 → erase vector loop（回滚，保留原标量循环）

```

**2. 单 Op 向量化 dispatch**

 **`vectorizeOneOperation()`** 

| 操作类型            | 处理函数                   |
| ------------------- | -------------------------- |
| `AffineLoadOp`      | `vectorizeAffineLoad()`    |
| `AffineStoreOp`     | `vectorizeAffineStore()`   |
| `AffineForOp`       | `vectorizeAffineForOp()`   |
| `AffineYieldOp`     | `vectorizeAffineYieldOp()` |
| `arith::ConstantOp` | `vectorizeConstant()`      |
| `AffineApplyOp`     | `vectorizeAffineApplyOp()` |
| 有 Region 的其他 op | → `nullptr`（不支持）      |
| 其他                | `widenOp()`                |

**3. 关键子函数**

**`vectorizeAffineForOp()`**

  **约束：**

  - `isVecDim && numIterOperands > 0 && step != 1` → 失败（规约循环步长须为1）
  - 若向量化该维度：`newStep = step * vectorFactor`（步长放大）
  - 若为规约维度：创建初始向量 `createInitialVector()`，可能创建 mask `createMask()`

**`vectorizeAffineLoad()` / `vectorizeAffineStore()`** 

  **约束（均适用）：**

  - operands 中含 `AffineApplyOp` → 失败
  - `isIVMappedToMultipleIndices()` → 失败
  - 无法构建 `permutationMap` → 失败

  生成 `vector.transfer_read` / `vector.transfer_write`

 **`vectorizeOperand()`** 

  **优先级：**

    1. 已有向量替换 → 直接返回缓存
    2. 常量 → `vectorizeConstant()`（splat）
    3. 循环外均匀值 → `vectorizeUniform()`（broadcast）
    4. 其他 → `nullptr`

 **`createMask()`** 

  **约束：**

  - 仅支持 1-D
  - `vecForOp.step == vectorSizes[0]`
  - 若循环边界为常数且 trip count 整除向量长度 → 不需要 mask，返回 `nullptr`

**4. 状态管理**

  `VectorizationState` 维护四张替换表：

| 表                            | 含义                            |
| ----------------------------- | ------------------------------- |
| `opVectorReplacement`         | 标量 op → 向量 op               |
| `valueVectorReplacement`      | 标量 value → 向量 value         |
| `valueScalarReplacement`      | 向量循环中的新标量值（如新 IV） |
| `loopResultScalarReplacement` | 规约结果 → 规约后标量           |

  `finishVectorizationPattern()` 在成功时用这些表做最终替换并清理。

### 5.8 生成数据搬运操作（Data Copy Generation）

#### 文件位置

- **源文件：** `mlir/lib/Dialect/Affine/Transforms/AffineDataCopyGeneration.cpp` (400 行)
- **测试文件：** `mlir/test/Dialect/Affine/affine-data-copy.mlir`

#### WHAT：数据搬运操作是什么？

自动将 **慢速内存空间** 的数据复制到 **快速内存空间**，并在计算完成后写回。

**场景：**

- GPU：全局内存 → 共享内存
- CPU：主内存 → 缓存/片上内存
- 加速器：DRAM → SRAM

**示例：矩阵乘法的数据复制**

**复制前：**

```text
func.func @matmul(%A: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {                   
  // 外层 tile 循环，step=32（非单步）                                                    
  affine.for %i0 = 0 to 1024 step 32 {
    affine.for %i1 = 0 to 1024 step 32 {
      // 内层计算循环
      affine.for %i = %i0 to %i0+32 {
        affine.for %j = %i1 to %i1+32 {
          %a = affine.load %A[%i, %j] : memref<1024x1024xf32>
          %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
          %sum = arith.addf %a, %c : f32
          affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
        }
      }
    }
  }
}

```

**复制后：**

```text
affine.for %i0 = 0 to 1024 step 32 {
  affine.for %i1 = 0 to 1024 step 32 {

    // === copy-in nest（自动生成，记入 copyNests）===
    %bufA = memref.alloc() : memref<32x32xf32, 1>  // 快速内存 space=1
    %bufC = memref.alloc() : memref<32x32xf32, 1>
    affine.dma_start %A[%i0+%i, %i1+%j], %bufA[%i,%j], %tag[0], %c1024
      : memref<...>, memref<...>, memref<1xi32, 2>
    affine.dma_wait %tag[0], %c1024 : memref<1xi32, 2>

    // === 原计算循环（load/store 已被重写到 bufA/bufC）===
    affine.for %i = 0 to 32 {
      affine.for %j = 0 to 32 {
        %a = affine.load %bufA[%i, %j] : memref<32x32xf32, 1>
        %c = affine.load %bufC[%i, %j] : memref<32x32xf32, 1>
        %sum = arith.addf %a, %c : f32
        affine.store %sum, %bufC[%i, %j] : memref<32x32xf32, 1>
      }
    }

    // === copy-out nest ===
    affine.dma_start %bufC[%i,%j], %C[%i0+%i, %i1+%j], %tag2[0], %c1024
      : memref<...>, memref<...>, memref<1xi32, 2>
    affine.dma_wait %tag2[0], %c1024 : memref<1xi32, 2>
    memref.dealloc %bufA : memref<32x32xf32, 1>
    memref.dealloc %bufC : memref<32x32xf32, 1>
  }
}

```

#### WHY：为什么需要显式数据搬运？

| 传统方法             | 显式复制         |
| -------------------- | ---------------- |
| 依赖硬件缓存         | 软件管理缓存     |
| 不可控               | 完全可控         |
| 跨越内存边界可能失效 | 保证在快速内存中 |

**WHY 仿射循环适合显式复制？**

- 访问模式 **静态可分析**
- 可以精确计算 **需要复制的数据区域**
- 可以在编译期插入 **复制循环**

#### HOW：实现解析

**核心算法：**

```cpp
void AffineDataCopyGeneration::runOnBlock(Block *block,
                                          DenseSet<Operation *> &copyNests) {
  AffineCopyOptions copyOptions = {
    generateDma,           // 使用 DMA 还是点对点复制
    slowMemorySpace,       // 源内存空间 ID
    fastMemorySpace,       // 目标内存空间 ID
    tagMemorySpace,        // DMA 标签内存空间
    fastMemCapacityBytes   // 快速内存容量限制
  };

  // 遍历基本块，识别需要复制的区域
  auto curBegin = findFirstLoadStoreOrFor(block);
  auto it = curBegin;

  while (it != block->end()) {
    AffineForOp forOp;
    if ((forOp = dyn_cast<AffineForOp>(&*it))) {
      // 检查内存足迹
      auto footprint = getMemoryFootprintBytes(forOp);

      if (footprint && *footprint > fastMemCapacityBytes) {
        // 超过容量，递归到内层
        runOnBlock(forOp.getBody(), copyNests);
      } else {
        // 足够小，在当前层复制
        affineDataCopyGenerate(curBegin, std::next(it), copyOptions, ...);
      }

      curBegin = findNextLoadStoreOrFor(std::next(it), block->end());
      it = curBegin;
    } else {
      ++it;
    }
  }
}
```

**内存足迹计算：**

```cpp
std::optional<int64_t> getMemoryFootprintBytes(AffineForOp rootForOp,
                                                unsigned memorySpace) {
  // 递归计算所有被访问的 memref 的总大小
  int64_t footprint = 0;

  rootForOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      Value memref = loadOp.getMemRef();
      if (getMemorySpace(memref) == memorySpace) {
        footprint += getMemRefSize(memref);
      }
    }
    // 类似处理 store 操作
  });

  return footprint;
}
```

**复制生成：**

```cpp
LogicalResult affineDataCopyGenerate(
    Block::iterator begin, Block::iterator end,
    const AffineCopyOptions &options,
    std::optional<filterMemRefFunc> memrefFilter,
    DenseSet<Operation *> &copyNests) {

  // 步骤 1：分析访问，确定需要复制的 memref
  DenseMap<Value, MemRefRegion> regions;
  for (Operation *op = begin; op != end; ++op) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      Value memref = loadOp.getMemRef();
      if (shouldCopy(memref, options, memrefFilter)) {
        regions[memref].unionRegion(op, ...);
      }
    }
    // 类似处理 store 操作
  }

  // 步骤 2：为每个 memref 分配快速内存缓冲
  for (auto &[memref, region] : regions) {
    // 计算缓冲区大小
    SmallVector<int64_t> bufferShape = region.getConstantShape();

    // 分配
    Value fastBuffer = createAllocOp(bufferShape, fastMemorySpace);

    // 步骤 3：生成 copy-in 循环
    createCopyInLoop(memref, fastBuffer, region);

    // 步骤 4：替换原始访问
    replaceMemRefUses(memref, fastBuffer, begin, end);

    // 步骤 5：生成 copy-out 循环
    if (region.isStored()) {
      createCopyOutLoop(fastBuffer, memref, region);
    }
  }

  return success();
}
```

**DMA vs 点对点复制：**

```cpp
if (options.generateDma) {
  // 使用 DMA 操作
  createDmaStart(srcMemref, dstBuffer, size, tag);
  createDmaWait(tag);
} else {
  // 使用点对点加载/存储
  affine.for %i = 0 to size {
    %v = affine.load srcMemref[i]
    affine.store %v, dstBuffer[i]
  }
}
```

### 5.9 循环不变代码提升

**WHY 分析 - 为什么要提升不变代码?**

如果计算不依赖循环变量，应该在循环外计算一次。

```cpp
// AffineLoopInvariantCodeMotion.cpp
LogicalResult mlir::affine::hoistLoopInvariantCode(AffineForOp forOp) {
  // WHY 1: 分析操作的操作数
  for (Operation &op : forOp.getBody()->withoutTerminator()) {
    // WHY 2: 检查是否所有操作数都是循环不变的
    bool isInvariant = llvm::all_of(op.getOperands(), [&](Value operand) {
      return isDefinedOutsideOfLoop(operand, forOp);
    });

    // WHY 3: 检查是否有副作用
    // 只有无副作用的操作才能提升
    if (isInvariant && isMemoryEffectFree(&op)) {
      // WHY 4: 移动到循环前
      op.moveBefore(forOp);
    }
  }
}
```

---

## 6. 测试用例分析

### 6.1 基本操作测试 (ops.mlir)

**测试维度和符号验证:**

```text
// test3: 显式使用 symbol 关键字
func.func @test3(%arg0 : index, %arg1 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // WHY: symbol() 明确标记符号操作数
      %1 = affine.load %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
    }
  }
}
```

**测试嵌套仿射表达式:**

```text
// test4: 复杂的仿射表达式
func.func @test4(%arg0 : index, %arg1 : index) {
  %0 = memref.alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // WHY: 支持嵌套表达式和整数除法
      %1 = affine.load %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7]
    }
  }
}
```

### 6.2 循环分块测试 (loop-tiling.mlir)

**基本分块:**

```text
// CHECK-LABEL: @loop_tiling()
func.func @loop_tiling() {
  // 原始: 三重循环
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k)
      }
    }
  }
}

// 分块后:
// affine.for %ii = 0 to 256 step 32 {    // 外块循环
//   affine.for %jj = 0 to 512 step 32 {
//     affine.for %kk = 0 to 1024 step 32 {
//       affine.for %i = %ii to %ii + 32 {   // 内块循环
//         affine.for %j = %jj to %jj + 32 {
//           affine.for %k = %kk to %kk + 32 {
//             "test.foo"(%i, %j, %k)
//           }
//         }
//       }
//     }
//   }
// }
```

**边界处理:**

```text
// WHY: 部分块处理边界
func.func @tile_using_symbolic_loop_upper_bounds(%arg0: memref<?x?xf32>, ...) {
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 0 to %0 {    // 动态边界
    affine.for %i1 = 0 to %0 {
      ...
    }
  }
}

// 分块后需要 min 表达式:
// affine.for %ii = 0 to %0 step 32 {
//   affine.for %i = %ii to min(%ii + 32, %0) {  // 处理边界
//     ...
//   }
// }
```

### 6.3 循环融合测试 (loop-fusion.mlir)

**生产者-消费者融合:**

```text
// 原始程序: 两个独立的循环
func.func @producer_consumer(%A: memref<100xf32>, %B: memref<100xf32>) {
  affine.for %i = 0 to 100 {
    affine.store %val, %A[%i]    // 生产者
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i]      // 消费者
    affine.store %v, %B[%i]
  }
}

// 融合后:
// affine.for %i = 0 to 100 {
//   affine.store %val, %A[%i]    // 生产
//   %v = affine.load %A[%i]      // 立即消费
//   affine.store %v, %B[%i]
// }
```

**切片融合:**

```text
// 只使用部分生产者输出
func.func @slice_fusion() {
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      affine.store ..., %A[%i, %j]    // 生产 100x100
    }
  }
  affine.for %i = 0 to 100 {
    %v = affine.load %A[%i, %i+1]      // 只使用对角线
    affine.store %v, %B[%i]
  }
}

// 融合后: 只计算需要的部分
// affine.for %i = 0 to 100 {
//   affine.for %j = %i to %i+1 {      // 只计算对角线
//     affine.store ..., %A[%i, %j]
//   }
//   %v = affine.load %A[%i, %i+1]
//   affine.store %v, %B[%i]
// }
```

---

## 7. 执行流程示例

### 7.1 矩阵乘法的完整优化流程

**原始代码:**

```text
// C = A × B
// A: 128 × 64, B: 64 × 96, C: 128 × 96
func.func @matmul(%A: memref<128x64xf32>,
                  %B: memref<64x96xf32>,
                  %C: memref<128x96xf32>) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 96 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<128x64xf32>
        %b = affine.load %B[%k, %j] : memref<64x96xf32>
        %c = affine.load %C[%i, %j] : memref<128x96xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<128x96xf32>
      }
    }
  }
  return
}
```

**维度说明:**

- A[128, 64]: 128 行，64 列
- B[64, 96]: 64 行，96 列
- C[128, 96]: 128 行，96 列
- 矩阵乘法: $C[i,j] = \sum_{k=0}^{63} A[i,k] × B[k,j]$

**步骤 1: 循环分块**

```
输入: 三个嵌套循环 (128, 96, 64)
分块大小: (32, 32, 16)
输出:
  - 外块循环: (0..128 step 32, 0..96 step 32, 0..64 step 16)
  - 内块循环: (ii..ii+32, jj..jj+32, kk..kk+16)
```

```text
affine.for %ii = 0 to 128 step 32 {
  affine.for %jj = 0 to 96 step 32 {
    affine.for %kk = 0 to 64 step 16 {
      affine.for %i = %ii to min(%ii + 32, 128) {
        affine.for %j = %jj to min(%jj + 32, 96) {
          affine.for %k = %kk to min(%kk + 16, 64) {
            // 原始计算
          }
        }
      }
    }
  }
}
```

**WHY 分块大小不同:**

- i, j 维度较大 (128, 96)，分块 32
- k 维度较小 (64)，分块 16
- 块大小: 32 × 32 × 16 = 16,384 次迭代

**步骤 2: 寄存器分块**

```
目的: 在寄存器中累加，减少存储次数
方法: 引入临时变量，内层 k 循环后存储

伪代码:
  for ii, jj, kk:  // 外块循环
    for i in block_ii:
      for j in block_jj:
        acc = 0  // 寄存器中的累加器
        for k in block_kk:  // 内层 k 循环
          acc += A[i,k] * B[k,j]
        C[i,j] = acc  // 只在 k 循环结束后存储一次
```

**步骤 3: 循环交换**

```
原顺序: i -> j -> k
  - A[i,k]: 按行访问（i固定，k变化）→ 连续 ✓
  - B[k,j]: 按列访问（k,j都变化）→ 跳跃 ✗
  - C[i,j]: 按行访问（i固定，j变化）→ 连续 ✓

交换后: i -> k -> j
  - A[i,k]: 按行访问（i固定，k变化）→ 连续 ✓
  - B[k,j]: 按行访问（k固定，j变化）→ 连续 ✓
  - C[i,j]: 跳跃访问（k,j都变化）→ 可能跳跃

目的: 提高 B（右矩阵）的局部性
```

**详细分析:**

原始顺序 i→j→k:

```
for i:
  for j:
    for k:
      A[i,k]  // i固定，k递增 → 连续访问行i
      B[k,j]  // k递增，j固定 → 跳跃访问不同行的第j列
              // 例如: B[0,j], B[1,j], B[2,j], ... B[127,j]
```

交换后 i→k→j:

```
for i:
  for k:
    for j:
      A[i,k]  // i固定，k递增 → 连续访问行i
      B[k,j]  // k固定，j递增 → 连续访问行k
              // 例如: B[k,0], B[k,1], B[k,2], ... B[k,127]
```

**WHY 交换提高性能:**

- B 的访问从按列变为按行，与行主序存储匹配
- 每次内层 j 循环，B 的一整行被连续访问
- 缓存行被充分利用

**步骤 4: 向量化**

```
交换后内层是 j 循环，检查其向量化可行性:

对于固定的 i, k:
  A[i,k]: 常量（可广播到向量）
  B[k,j]: j = 0,1,2,...,95 → 连续内存访问 ✓
  C[i,j]: j = 0,1,2,...,95 → 连续内存访问 ✓

向量化策略（新维度 128×64×96）:
  affine.for %i = 0 to 128 {
    affine.for %k = 0 to 64 {
      affine.for %j = 0 to 96 step 4 {      // 向量宽度 = 4
        %a = affine.load %A[%i, %k]         // A[i,k]: 标量广播
        %b_vec = vector.load %B[%k, %j]     // B[k,j]: 连续向量加载
        %c_vec = vector.load %C[%i, %j]     // C[i,j]: 连续向量加载
        %a_vec = vector.broadcast %a        // 广播 %a 到 4 个元素
        %p_vec = vector.fma %a_vec, %b_vec, %c_vec  // %p = %a×%b + %c（向量化）
        vector.store %p_vec, %C[%i, %j]
      }
    }
  }
}

// 内存访问分析（k=32 时）:
//   B[32, 0:3], B[32, 4:7], B[32, 8:11], ... B[32, 92:95] → 连续！
//   C[64, 0:3], C[64, 4:7], C[64, 8:11], ... C[64, 92:95] → 连续！

// WHY 可以向量化
1. B[k, j] 在 j 维度连续 → 可以向量加载
2. C[i, j] 在 j 维度连续 → 可以向量存储
3. A[i, k] 在 j 循环中不变 → 可以广播
4. j 循环各次迭代无依赖 → 可以并行执行

// vector.store 工作原理:
%p_vec = vector<4xf32>  // 包含 4 个元素
vector.store %p_vec, %C[%i, %j] 的含义: 将 %p_vec 的 4 个元素连续存储到从 C[%i, %j] 开始的 4 个位置

示例（i=64, j=0）:
  vector.store %p_vec[0,1,2,3], C[64, 0]
  → C[64,0] = %p_vec[0]
  → C[64,1] = %p_vec[1]
  → C[64,2] = %p_vec[2]
  → C[64,3] = %p_vec[3]

示例（i=64, j=4）:
  vector.store %p_vec[0,1,2,3], C[64, 4]
  → C[64,4] = %p_vec[0]
  → C[64,5] = %p_vec[1]
  → C[64,6] = %p_vec[2]
  → C[64,7] = %p_vec[3]
```

关键: 向量化发生在 **j 循环内**，i 和 k 在此时是**常量索引**

**向量化代码结构:**

```text
affine.for %i = 0 to 128 {            // 外层循环
  affine.for %k = 0 to 64 {           // 中层循环
    affine.for %j = 0 to 96 step 4 {  // 内层循环（被向量化）

      // i=64, k=32 时，j=0,4,8 的各次迭代:

      // ============ 外层变量的角色 ============
      // %i: 固定为 64，用于选择 A 和 C 的行
      // %k: 固定为 32，用于选择 A 的列和 B 的行
      // 它们不参与向量化计算

      %a = affine.load %A[64, 32]    // A[行64, 列32] → 标量
      %b_vec = vector.load %B[32, 0:3]  // B[行32, 列0-3] → 向量4
      %c_vec = vector.load %C[64, 0:3]  // C[行64, 列0-3] → 向量4

      // ============ 向量化计算 ============
      %a_vec = vector.broadcast %a        // 标量 → 向量(复制4份)

      // 向量 FMA: p[m] = a[m] × b[m] + c[m]
      //   其中 m 是 j 循环内的向量索引 (0, 1, 2, 3)
      //   每次计算 4 个结果，对应 j=0,4,8,12 时的存储

      vector.store %p_vec, %C[64, 0]  // 存储 4 个元素到 C[行64]
    }  // j 循环结束
  }  // k 循环结束
}  // i 循环结束
```

**WHY i 和 k 不需要向量化:**

- 它们是外层循环的归纳变量
- 在 j 循环内保持**不变**（是常量）
- 用于**索引** A、B、C 的维度
- 向量化的是 j 维度的**元素级并行**

**向量化效果:**

- 原 j 循环：96 次迭代，每次处理 1 个元素
- 向量化后：24 次迭代（96/4），每次处理 4 个元素
- 理论加速：约 4x（假设向量宽度与硬件匹配）

---

**扩展：i-k 并行化的详细分析**

### 问题的本质：循环携带依赖

原始三重嵌套循环:

```
for i in [0, 128):
  for k in [0, 64):
    C[i,j] += A[i,k] * B[k,j]  // C[i,j] 需要上一次迭代的值
```

**WHY 有依赖:**

- C[i,j] 在内层 j 循环中被累加
- 每次迭代需要读取上一次写入的 C[i,j]
- 这是**跨迭代的依赖**（循环携带依赖）

### 方案 1: affine.parallel 的限制

```text
affine.parallel (%i, %k) = (0, 0) to (128, 64) {
  affine.for %j = 0 to 96 {
    // 问题: C[i,j] 的归约在哪里?
    // affine.parallel 支持 reductions，但:
    //  - 归约只在循环结束后执行一次
    //  - 无法在内层循环中使用部分归约结果
  }
}
```

**WHY 不够:**

- `affine.parallel` 的归约操作在**最外层循环结束后**执行
- 无法满足 `C[i,j] += A[i,k] * B[k,j]` 的模式（需要在内层循环中使用部分结果）

### 方案 2: 循环交换 + 归约重组（实际可行）

**第 1 步: 循环交换解除依赖**

原始顺序: i → j → k

```
for i in [0, 128):        // 外层循环
  for j in [0, 96):       // 中层循环
    for k in [0, 64):     // 内层循环
      C[i,j] += A[i,k] * B[k,j]  // 依赖上一次迭代的 C[i,j]
```

**WHY 原始有依赖:**

- C[i,j] 在内层 k 循环中累加
- 每次迭代需要读取上一次写入的 C[i,j]
- k 循环有**循环携带依赖**

交换后顺序: k → j → i

```
for k in [0, 64):        // 外层循环 (原内层)
  for j in [0, 96):       // 中层循环 (原中层)
    sum = 0               // 可以在 i 循环前初始化
    for i in [0, 128):    // 内层循环 (原外层)
      sum += A[i,k] * B[k,j]  // 不再有循环携带依赖!
    C[k,j] = sum           // 在 i 循环结束后存储
```

**WHY 交换后可以并行:**

- k 和 j 在外层和中层（可以并行）
- i 在内层，循环结束时产生**完整的** sum（不需要跨迭代传递）
- C[k,j] 在 i 循环结束后一次性写入（无循环携带依赖）

**第 2 步: 应用 affine.parallel**

```text
affine.parallel (%k, %j) = (0, 0) to (64, 96) {
  // k 和 j 现在可以并行执行
  // 因为它们的迭代是独立的

  // 内层 i 循环带归约
  %sum = affine.for %i = 0 to 128 iter_args(%acc = %zero) {
    %a = affine.load %A[%i, %k]
    %b = affine.load %B[%k, %j]
    %p = arith.mulf %a, %b
    %new_acc = arith.addf %acc, %p
    affine.yield %new_acc  // 传递给下一次迭代
  }

  // 在 k,j 并行循环结束后存储
  affine.store %sum, %C[%k, %j]
}
```

### 完整的变换序列

```
原始: i → j → k (串行，有循环携带依赖)
  ↓
交换: k → j → i (依赖解除)
  ↓
并行: affine.parallel(k, j) (k 和 j 可以并行)
  ↓
向量化: 内层 i 循环向量化 (可以与并行化同时使用！)
```

### 并行化 + 向量化同时使用

**重要: 并行化和向量化是互补的优化技术，可以同时应用！**

```text
// 完整的并行 + 向量化版本
affine.parallel (%k, %j) = (0, 0) to (64, 96) {
  // ============ 并行层级 ============
  // 不同的 (k, j) 对在不同线程/核心上执行

  // ============ 向量化层级 ============
  // 每个 (k, j) 对内部，i 循环使用 SIMD
  affine.for %i = 0 to 128 step 4 {      // 向量宽度 = 4
    %a_vec = vector.load %A[%i, %k]     // A[i:i+4, k] 向量加载
    %b = affine.load %B[%k, %j]         // B[k, j] 标量加载
    %c_vec = vector.load %C[%i, %j]     // C[i:i+4, j] 向量加载

    %b_vec = vector.broadcast %b, 4     // 标量广播为向量

    %p_vec = vector.fma %a_vec, %b_vec, %c_vec  // 向量 FMA
    vector.store %p_vec, %C[%i, %j]
  }
}
```

**硬件执行模型:**

```
┌─────────────────────────────────────────────────────────┐
│                    多核处理器 + SIMD                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 核心 0    │  │ 核心 1   │  │ 核心 2    │  │ 核心 3    │  │
│  │(k=0,j=0) │  │(k=0,j=1) │  │(k=0,j=2) │  │(k=0,j=3) │  │
│  │          │  │          │  │          │  │          │  │
│  │SIMD:4×f32│  │SIMD:4×f32│  │SIMD:4×f32│  │SIMD:4×f32│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────┘

并行化: 4 个核心同时处理不同的 (k, j)
向量化: 每个核心用 SIMD 处理 4 个 i 元素
```

**加速效果:**

- 并行: 4x (假设 4 个核心)
- 向量化: 4x (向量宽度 4)
- 总加速: 16x (理想情况，无其他瓶颈)

### 实际应用案例

**这种并行化 + 向量化方案在工业界广泛使用！**

#### 1. 高性能矩阵乘法库

| 库               | 应用         | 优化技术         |
| ---------------- | ------------ | ---------------- |
| **Intel oneDNN** | 深度学习推理 | OpenMP + AVX-512 |
| **OpenBLAS**     | 科学计算     | pthread + AVX2   |
| **BLIS**         | HPC          | OpenMP + ARM SVE |

**实际性能：** 相比朴素实现可达 50-100x 加速

#### 2. 深度学习框架

```
PyTorch/TensorFlow 的矩阵乘法调用链:

torch.matmul / tf.matmul
  ↓
oneDNN / cuBLAS / Eigen
  ↓
多线程 + SIMD 内核 (类似上述方案)
  ↓
硬件执行
```

#### 3. MLIR 在工业界的应用

**使用 MLIR Affine 优化循环的项目：**

1. **TensorFlow/XLA**
   - 使用 Affine dialect 进行循环优化
   - 自动 lowering 到 GPU/CPU 代码

2. **IREE (Google)**
   - MLIR 作为核心 IR
   - Affine → GPU (Vulkan/SPIRV/CUDA)

3. **torch-mlir**
   - PyTorch 程序的 MLIR 编译器
   - 自动应用分块、并行化、向量化

#### 4. LLVM Polly 优化器

```bash
# Polly 是 LLVM 的多面体优化框架
# 自动应用本文档讨论的所有变换

clang -O3 -mllvm -polly \
       -mllvm -polly-vectorizer=stripmine \
       matmul.c

# Polly 自动执行:
# 1. 依赖分析 (checkMemrefAccessDependence)
# 2. 循环分块 (LoopTiling)
# 3. 循环交换 (LoopInterchange)
# 4. OpenMP 并行化
# 5. SIMD 向量化
```

#### 5. Intel oneDNN 的矩阵乘法内核 (简化)

```cpp
// 文件: src/cpu/x64/jit_gemm_s8u8s32_avx512_core.cpp

// 外层：OpenMP 并行
#pragma omp parallel for collapse(2)
for (int k = 0; k < K; k += k_block) {
  for (int j = 0; j < N; j += j_block) {
    // 中层：分块
    for (int i = 0; i < M; i += i_block) {
      // 内层：AVX-512 向量化
      __m512i sum = _mm512_setzero_si512();
      for (int k_inner = 0; k_inner < k_block; k_inner += 16) {
        // 向量加载 (16 × int8)
        __m512i a = _mm512_loadu_si512(...);
        __m512i b = _mm512_loadu_si512(...);
        // 向量 FMA (融合乘加)
        sum = _mm512_dpbusd_epi32(sum, a, b);  // AVX-512 VNNI
      }
      _mm512_storeu_si512(..., sum);
    }
  }
}
```



### 关键理解

**WHY 循环交换能解除依赖:**

- 原始：C[i,j] 需要上一次迭代的值（跨 k 依赖）
- 交换后：C[k,j] 在内层 i 循环结束后计算完成（无跨迭代依赖）

**WHY affine.parallel 需要交换:**

- 原始结构有循环携带依赖（串行性质）
- 交换后结构可以分解为独立的并行任务
- 需要先做**分块**和**循环交换**来解除依赖

**实际编译器的做法:**

1. 首先进行循环分块
2. 然后应用循环交换
3. 最后在最内层使用向量化
4. 如果有 `affine.parallel` 支持，直接在最内层使用

****

### 7.2 依赖分析示例

**代码:**

```text
affine.for %i = 1 to 99 {
  affine.for %j = 1 to 99 {
    %v1 = affine.load %A[%i, %j]        // S1
    %v2 = affine.load %A[%i+1, %j]      // S2
    affine.store %v1, %A[%i, %j+1]      // S3
  }
}
```

**依赖分析:**

```
访问函数:
  S1.read:  (i, j) -> (i, j)
  S2.read:  (i, j) -> (i+1, j)
  S3.write: (i, j) -> (i, j+1)

依赖检查:
  S1 -> S3:
    约束: (i1, j1) = (i2, j2+1)
    顺序: (i1, j1) < (i2, j2)
    解: j1 = j2 + 1 且 (i1 < i2 或 (i1 = i2 且 j1 < j2))
         = j1 = j2 + 1 且 j1 < j2
         无解!
         结论: 无依赖

  S2 -> S3:
    约束: (i+1, j) = (i', j'+1)
    顺序: (i, j) < (i', j')
    解: i+1 = i', j = j'+1
         且 (i < i' 或 (i = i' 且 j < j'))
         i < i+1: 恒成立
         方向向量: [1, -1] (i 跨 1 步, j 跨 -1 步)

并行性:
  外层 i 循环: 有依赖 (跨步)，不能并行
  内层 j 循环: 有依赖 (反向)，不能并行
```

### 7.3 Affine 执行流程

**编译阶段:**

```
1. 解析 (Parser)
   ├─ 识别 affine.for/if/parallel
   ├─ 解析仿射映射
   └─ 验证维度/符号约束

2. 验证 (Verifier)
   ├─ isValidDim: 检查维度有效性
   ├─ isValidSymbol: 检查符号有效性
   └─ 检查操作数与映射一致性

3. 分析 (Analysis)
   ├─ 构建依赖图
   ├─ 计算迭代空间
   └─ 检测并行性

4. 变换 (Transforms)
   ├─ 应用优化 Pass
   ├─ 保持语义正确性
   └─ 更新依赖信息

5. 降低 (Lowering)
   ├─ affine.for -> scf.for
   ├─ affine.if -> scf.if
   └─ affine.load/store -> 标准操作
```

**运行时阶段:**

```
Affine 本身不引入运行时开销
所有分析都在编译时完成
生成的代码与手写循环性能相当或更优
```

---

## 总结

### Affine 方言的核心价值

1. **精确的数学表示**: 通过仿射约束精确表示程序行为
2. **强大的分析能力**: 依赖分析、并行性检测自动完成
3. **丰富的优化空间**: 多面体变换提供广阔的优化空间
4. **渐进式降低**: 可以逐步降低到更底层的方言

### 关键设计原则

1. **WHY 约束仿射表达式?**
   - 保证可分析性
   - 支持精确的依赖分析
   - 可以在编译时完全求值

2. **WHY 分离维度和符号?**
   - 区分迭代相关和无关的值
   - 简化分析算法
   - 提高优化精度

3. **WHY 提供专门的循环结构?**
   - 编码更多优化信息
   - 自动验证优化合法性
   - 简化变换实现

### 未来发展

1. **更丰富的分析**: 跨函数、跨模块分析
2. **自动调优**: 基于性能模型的自动参数选择
3. **GPU 支持**: 更好的 GPU 映射和优化
4. **与其他方言协同**: 与 Linalg、Vector 等方言的深度集成

---

**参考文献:**

1. MLIR Documentation: https://mlir.llvm.org/docs/Dialects/Affine/
2. Polyhedral compilation: "Polyhedral Compilation" by Louis-Noel Pouchet
3. Affine transformations: "Affine Transformations" in LLVM
4. Presburger arithmetic: "Decision Methods for the Algebra of Theory of Real Fields" by Tarski
