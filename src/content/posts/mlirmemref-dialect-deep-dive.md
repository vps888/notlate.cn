---
title: "【MLIR】MemRef方言深入研究"
description: "本文档基于 Claude Code + Sonnet4.6 (https://zhetengxia.com/) + CodeReaderSkills (https://zhetengxia.com/)完成。 一、MemRef方言概述 设计目的和核心概念 MemRef（Memory Refere…"
slug: "mlirmemref-dialect-deep-dive"
legacyId: 19760130
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19760130"
pubDate: 2026-03-23
updatedDate: 2026-03-24
category: "AI 编译器"
tags: ["AI 编译器","MLIR","MemRef"]
featured: true
---

> 本文档基于[Claude Code + Sonnet4.6](https://zhetengxia.com/) + [CodeReaderSkills](https://zhetengxia.com/)完成。

## 一、MemRef方言概述

### 设计目的和核心概念

MemRef（Memory Reference）方言是MLIR中用于表示内存引用的核心方言。它提供了一种抽象的方式来描述和操作多维内存缓冲区，而不依赖于具体的内存分配方式或硬件细节。

**核心概念**：

1. **MemRef类型**：表示内存引用的抽象类型，包含以下关键属性：
   - `element type`：元素类型（如f32, i32等）
   - `shape`：形状信息（静态维度和动态维度）
   - `layout map`：布局映射（通常用仿射表示）
   - `address space`：地址空间
   - `memory space`：内存空间

2. **布局映射**：使用仿射映射描述内存布局，支持：
   - 恒等布局：`affine_map<(d0, d1) -> (d0, d1)>`
   - 压缩布局：`affine_map<(i, j) -> (i * 8 + j)>`
   - 分块布局：`affine_map<(i) -> (i floordiv 4, i mod 4)>`

3. **内存空间**：支持不同的内存空间（如GPU的共享内存、全局内存等）

### 与MLIR整体架构的关系

MemRef方言在MLIR架构中处于核心位置：

1. **中间表示层**：提供了高级的内存抽象，位于具体硬件实现之上
2. **平台无关性**：不依赖于特定的内存分配器（如malloc/alloca）
3. **可优化性**：通过布局映射和变换能力，支持编译时优化
4. **多方言协同**：与Affine、Arith、Vector、SCF、Func等方言紧密协作

### 与Affine方言的关系

MemRef与Affine方言的协作是MLIR内存管理的核心：

1. **地址计算**：Affine方言计算访问索引，MemRef提供被访问的内存缓冲区
2. **布局映射**：MemRef类型包含仿射布局映射，Affine操作使用这些映射进行地址转换
3. **优化协同**：共享循环变换和分析能力，联合进行内存访问模式分析和并行化

## 二、核心Operations

### 操作分类

#### 1. 内存分配操作

- `memref.alloc`：堆内存分配
- `memref.alloca`：栈内存分配
- `memref.realloc`：重新分配内存
- `memref.dealloc`：释放内存

#### 2. 内存访问操作

- `memref.load`：从缓冲区加载数据
- `memref.store`：向缓冲区存储数据

#### 3. 内存视图操作

- `memref.subview`：创建子视图（rank-reducing）
- `memref.reinterpret_cast`：重新解释内存布局
- `memref.cast`：类型转换
- `memref.reshape`：改变形状（不复制数据）
- `memref.expand_shape`：扩展维度
- `memref.collapse_shape`：合并维度
- `memref.transpose`：转置视图

#### 4. 元数据查询操作

- `memref.dim`：查询维度大小
- `memref.rank`：查询张量秩

#### 5. 元数据提取操作

- `memref.extract_strided_metadata`：提取步幅和偏移量元数据

#### 6. 内存空间操作

- `memref.memory_space_cast`：内存空间转换

#### 7. DMA操作（异构系统）

- `memref.dma_start`：开始DMA传输
- `memref.dma_wait`：等待DMA完成

#### 8. 全局变量

- `memref.global`：声明全局变量
- `memref.get_global`：获取全局变量引用

#### 9. 原子操作

- `memref.atomic_rmw`：原子读-改-写操作
- `memref.generic_atomic_rmw`：通用的原子RMW操作

#### 10. 复制操作

- `memref.copy`：内存复制

#### 11. 假设操作

- `memref.assume_alignment`：假设对齐
- `memref.memory_space_cast`：内存空间转换

#### 12. 其他操作

- `memref.alloc_time`：获取分配时间
- `memref.prefetch`：数据预取

## 三、Pass详解（重点）

### 3.1 整体学习路线

MemRef方言的Transforms目录下有17个源文件，建议按照下述顺序学习：

![MemRef Transforms整体学习路线](https://img2024.cnblogs.com/blog/3599704/202603/3599704-20260324091243609-889428452.svg)

### 3.2 AllocationOpInterfaceImpl

**功能**：为MemRef分配操作提供`AllocationOpInterface`实现，接口的价值是让 pass 对 op 类型无感知。

**核心实现**（位于`AllocationOpInterfaceImpl.cpp`）：

```cpp
struct DefaultAllocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::AllocOp> {
  // 为 AllocOp/ReallocOp 构建构建 DeallocOp 操作
  static std::optional<Operation *> buildDealloc(OpBuilder &builder, Value alloc) {
    return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc).getOperation();
  }

  // 构建clone操作
  static std::optional<Value> buildClone(OpBuilder &builder, Value alloc) {
    return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc).getResult();
  }

  // 获取提升类型
  static HoistingKind getHoistingKind() {
    return HoistingKind::Loop | HoistingKind::Block;
  }

  // 构建提升后的alloc
  static std::optional<Operation *> buildPromotedAlloc(OpBuilder &builder, Value alloc) {
    return builder.create<memref::AllocaOp>(...);
  }
};
```

**示例 - buildDealloc**:

```cpp
// 对于 AllocOp
Operation *DefaultAllocationInterface::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc);
}

// 使用场景
%alloc = memref.alloc() : memref<100xf32>
// ... 使用 %alloc ...
%dealloc = AllocationOpInterface::buildDealloc(builder, %alloc)
// %dealloc 是 memref.dealloc %alloc
```

**示例 - buildClone**:

```cpp
Value DefaultAllocationInterface::buildClone(OpBuilder &builder, Value alloc) {
  MemRefType type = cast<MemRefType>(alloc.getType());
  Operation *clone = builder.create<memref::AllocOp>(
      alloc.getLoc(), type,
      getAsOpFoldResult(alloc.getDefiningOp()->getOperands()));
  builder.create<memref::CopyOp>(alloc.getLoc(), alloc, clone->getResult(0));
  return clone->getResult(0);
}

// 使用场景
%original = memref.alloc() : memref<100xf32>
%cloned = AllocationOpInterface::buildClone(builder, %original)
// %cloned 是 %original 的副本
```

**示例 - buildPromotedAlloc**:

```cpp
Value DefaultAllocationInterface::buildPromotedAlloc(
    OpBuilder &builder, Value alloc) {
  MemRefType type = cast<MemRefType>(alloc.getType());
  return builder.create<memref::AllocaOp>(alloc.getLoc(), type);
}

// 使用场景: 将堆分配提升为栈分配
%heap_alloc = memref.alloc() : memref<100xf32>
%stack_alloc = AllocationOpInterface::buildPromotedAlloc(builder, %heap_alloc)
// %stack_alloc 是 memref.alloca() : memref<100xf32>
```

**使用场景**:

- 自动内存管理
- 分配提升优化
- 克隆操作生成
- 与Bufferization pass集成

**优化效果**：

- 支持分配提升优化
- 支持内存到寄存器提升
- 与bufferization框架集成

### 3.3 BufferViewFlowOpInterface

**文件**: `BufferViewFlowOpInterfaceImpl.cpp`

**作用**: 为 ReallocOp 实现 `BufferViewFlowOpInterface`

**依赖关系**: realloc 的结果可能依赖于源操作数

**终端缓冲区**: realloc 可能返回新分配的缓冲区，因此是终端缓冲区

**接口方法**:

```cpp
struct ReallocOpInterface
    : public BufferViewFlowOpInterface::ExternalModel<...> {
  void populateDependencies(Operation *op,
                           BufferViewFlowAnalysis::DependenyMap &dependencies);

  bool mayBeTerminalBuffer(Operation *op, Value value);
};
```

**populateDependencies**:

```cpp
void ReallocOpInterface::populateDependencies(
    Operation *op,
    BufferViewFlowAnalysis::DependencyMap &dependencies) {
  auto reallocOp = cast<memref::ReallocOp>(op);
  // realloc 的结果依赖于:
  // 1. 源操作数（如果重用）
  // 2. 新分配（如果重新分配）
  dependencies[reallocOp.getResult()].push_back(reallocOp.getSource());
}
```

**mayBeTerminalBuffer**:

```cpp
bool ReallocOpInterface::mayBeTerminalBuffer(Operation *op, Value value) {
  auto reallocOp = cast<memref::ReallocOp>(op);
  // 如果realloc返回新分配，则是终端缓冲区
  return true;
}
```

**使用场景**:

- 缓冲区生命周期分析
- 别名分析
- 内存优化
- 死代码消除

**接口用法演示（内存Inplace优化）**

`BufferViewFlowAnalysis::build()` 构建一张依赖图：`Value → Set<Value>`，表示"这个值可能依赖于哪些 buffer"。

演示代码：

```cpp
// BufferViewFlowAnalysis.cpp:82-118
void BufferViewFlowAnalysis::build(Operation *op) {
  // 步骤 0: 定义 registerDependencies 函数
  auto registerDependencies = [&](ValueRange values, ValueRange dependencies) {
    for (auto [value, dep] : llvm::zip_equal(values, dependencies)) {
      this->dependencies[value].insert(dep);      // value 依赖于 dep
      this->reverseDependencies[dep].insert(value); // dep 被 value 依赖
    }
  };
  //          ↓ 调用后效果
  // dependencies[result] = {true_value, false_value}

  op->walk([&](Operation *op) {
    // 步骤 1: 检查 op 是否实现 BufferViewFlowOpInterface
    if (auto bufferViewFlowOp = dyn_cast<BufferViewFlowOpInterface>(op)) {
      // 步骤 2: 让 op 自己注册依赖关系
      bufferViewFlowOp.populateDependencies(registerDependencies);
      //                 ↓ 对于 arith.select，内部会执行：
      //                 registerDependenciesFn(true_value, result);
      //                 registerDependenciesFn(false_value, result);

      // 步骤 3: 检查结果是否是 terminal（分配终点，不能再追溯）
      for (Value v : op->getResults())
        if (isa<BaseMemRefType>(v.getType()) &&
            bufferViewFlowOp.mayBeTerminalBuffer(v))  // realloc 返回 true
          this->terminals.insert(v);
      return;
    }

    // 步骤 4: 回退方案 - 用其他接口推断
    if (auto viewInterface = dyn_cast<ViewLikeOpInterface>(op)) {
      // subview/slice 等：结果依赖于 source
      registerDependencies(viewInterface.getViewSource(),
                           viewInterface->getResult(0));
      return;
    }
    // ... BranchOpInterface, RegionBranchOpInterface 类似处理
  });
}
```

输入 IR：

```text
func.func @example(%cond: i1) {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %sel = arith.select %cond, %a, %b : memref<10xf32>
  %new = memref.realloc %sel : memref<10xf32>
  return
}
```

构建过程：

| 遍历到的 op         | 接口实现                                 | 注册的依赖                    | terminals |
| ------------------- | ---------------------------------------- | ----------------------------- | --------- |
| memref.alloc %a     | -                                        | -                             | %a ✅      |
| memref.alloc %b     | -                                        | -                             | %b ✅      |
| arith.select %sel   | SelectOpInterface::populateDependencies  | dependencies[%sel] = {%a, %b} | -         |
| memref.realloc %new | ReallocOpInterface::populateDependencies | dependencies[%new] = {%sel}   | %new ✅    |

最终数据结构：

```
dependencies = {
  %sel:  {%a, %b},
  %new: {%sel}
}

terminals = {%a, %b, %new}
```

调用 `analysis.resolve(%new)` 递归展开：

```
// → {%new, %sel, %a, %b}
```

为什么需要这张图？

Bufferize pass 需要知道 "%new inplace 修改安全吗？"

1. 调用 `analysis.resolve(%new)` → `{%new, %sel, %a, %b}`
2. 发现 %a、%b 也在这集合里
3. 如果 %a 还在被用 → 不能 inplace，必须 clone

这就是 `one-shot-bufferize-analysis.mlir` 测试的逻辑：

```text
// 测试意图：证明 %2 可能 alias %1，所以 %1 不能 inplace
%1 = linalg.fill ... outs(%0)
%2 = "dummy.dummy_op"(%1)  // 通过 BufferViewFlowOpInterface 声称 alias %1
%3 = linalg.fill ... outs(%2)  // 如果 %1 inplace，会错误地改掉 %2

// 输出检查：
// CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
//                                    ↑ 这个 false 就是基于依赖分析得出的
```

### 3.4  RuntimeOpVerification

**文件**: `RuntimeOpVerification.cpp`

**作用**: 运行时操作验证 - 为 MemRef 操作生成运行时验证代码

**边界检查**: 生成 `0 <= index < dim_size` 的断言

**对齐验证**: 检查指针对齐

**类型验证**: 检查秩、维度大小、偏移和步长

**SubView 验证**: 验证偏移和切片不越界

**使用 `cf::AssertOp`**: 生成运行时断言

**接口实现**:

```cpp
struct AssumeAlignmentOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct CastOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct LoadStoreOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};

struct SubViewOpInterface {
  void generateVerification(Operation *op, OpBuilder &builder);
};
```

**示例 - Load验证**:

转换前:

```text
%val = memref.load %memref[%i, %j] : memref<100x100xf32>
```

转换后:

```text
// 运行时边界检查
%dim0 = memref.dim %memref, 0 : memref<100x100xf32>
%dim1 = memref.dim %memref, 1 : memref<100x100xf32>
%check0 = arith.cmpi slt, %i, %dim0 : index
%check1 = arith.cmpi slt, %j, %dim1 : index
%check2 = arith.cmpi sge, %i, 0 : index
%check3 = arith.cmpi sge, %j, 0 : index
%valid = arith.andi %check0, %check1, %check2, %check3 : i1
cf.assert %valid, "index out of bounds" : i1

%val = memref.load %memref[%i, %j] : memref<100x100xf32>
```

**示例 - SubView验证**:

转换前:

```text
%subview = memref.subview %base[%off0, %off1] [%size0, %size1] [1, 1]
    : memref<100x100xf32> to memref<10x20xf32>
```

转换后:

```text
// 验证偏移不越界
%dim0 = memref.dim %base, 0 : memref<100x100xf32>
%dim1 = memref.dim %base, 1 : memref<100x100xf32>

%off0_ok = arith.cmpi sle, %off0, %dim0 : index
%off1_ok = arith.cmpi sle, %off1, %dim1 : index
%size0_ok = arith.cmpi sle, (%off0 + %size0), %dim0 : index
%size1_ok = arith.cmpi sle, (%off1 + %size1), %dim1 : index

%all_ok = arith.andi %off0_ok, %off1_ok, %size0_ok, %size1_ok : i1
cf.assert %all_ok, "subview out of bounds" : i1

%subview = memref.subview %base[%off0, %off1] [%size0, %size1] [1, 1]
    : memref<100x100xf32> to memref<10x20xf32>
```

**使用场景**:

- 调试
- 安全检查
- 动态验证
- 边界条件检测

### 3.5 ReifyResultShapes

**文件**: `ReifyResultShapes.cpp`

**作用**: 具体化结果形状 - 为 `ReifyRankedShapedTypeOpInterface` 操作具体化结果形状

**形状具体化**: 调用 `reifyResultShapes` 获取形状

**类型更新**: 根据具体化的形状更新结果类型

**操作克隆**: 克隆操作并更新结果类型

**转换插入**: 插入 `cast` 操作以保持 IR 一致性

**限制**: 当前只支持 `tensor::PadOp` 和 `tensor::ConcatOp`

**核心流程**:

```cpp
LogicalResult reifyOpResultShapes(
    Operation *op,
    ReificationCallbackFn reificationCallback);

// 对于每个操作结果:
// 1. 最终调用具体 Op 自身的 reifyResultShapes 获取形状值
// 2. 更新结果类型为静态形状
// 3. 插入 cast 从动态类型到静态类型
```

**示例**:

转换前:

```text
%padded = tensor.pad %source low[0] high[%pad_amount] {
  ^bb0(%arg0: index):
    tensor.yield %c0 : f32
} : tensor<?xf32> to tensor<?xf32>
%dim = tensor.dim %padded, 0 : tensor<?xf32>
```

转换后:

```text
// 形状被具体化为计算值
%original_dim = tensor.dim %source, 0 : tensor<?xf32>
%static_size = arith.addi %original_dim, %pad_amount : index

// 结果类型变为静态
%padded_static = tensor.pad %source low[0] high[%pad_amount] {
  ^bb0(%arg0: index):
    tensor.yield %c0 : f32
} : tensor<?xf32> to tensor<100xf32>  // 具体化大小

// Cast保持类型兼容
%padded = tensor.cast %padded_static : tensor<100xf32> to tensor<?xf32>

// dim操作可以被折叠
%dim = arith.constant 100 : index  // 替换原始的 tensor.dim
```

**使用场景**:

- 形状推断
- 类型静态化
- 边界检查消除
- 优化循环边界

### 3.6 NormalizeMemRefsPass

**文件**: `NormalizeMemRefs.cpp`

**作用**: 将 memref 转换为恒等布局映射

**函数间分析**:

1. 识别所有可规范化的函数
2. 调用/被调用非规范化函数的函数也被视为不可规范化

**规范化过程**:

1. 更新函数参数类型
2. 规范化 `AllocOp`, `AllocaOp`, `ReinterpretCastOp`
3. 更新函数返回类型
4. 更新调用点

**布局映射**: 使用 AffineMap 处理非恒等布局

**核心算法**：

1. **可规范化性分析**：

```cpp
bool areMemRefsNormalizable(func::FuncOp funcOp) {
  // 检查函数中所有MemRef类型是否可规范化
  // 只有load/store/dealloc/call/return等操作的use才能规范化
}
```

2. **函数签名更新**：

```cpp
void updateFunctionSignature(func::FuncOp funcOp, ModuleOp moduleOp) {
  // 更新函数参数和返回类型的MemRef布局
  // 需要同时更新所有调用点
}
```

3. **操作结果规范化**：

```cpp
Operation *createOpResultsNormalized(func::FuncOp funcOp, Operation *oldOp) {
  // 为操作的MemRef结果创建恒等布局版本
}
```

**转换示例**：

```text
// 转换前
#map = affine_map<(i) -> (i floordiv 4, i mod 4)>
%alloc = memref.alloc() : memref<16xf32, #map>

// 转换后
%alloc = memref.alloc() : memref<4x4xf32>
%flat = affine.apply affine_map<(i, j) -> (i * 4 + j)> (%i, %j)
```

**优化效果**：

- 简化后续分析（恒等布局更容易分析）
- 为向量化、并行化等优化铺平道路

### 3.7 FlattenMemRefsPass

**文件**: `FlattenMemRefs.cpp`

**功能**：将多维MemRef操作转换为一维MemRef操作。

**操作重写**:

- **AllocOp/AllocaOp**: 创建一维分配，然后用 reinterpret_cast 恢复原始类型
- **LoadOp/StoreOp**: 使用线性化索引访问一维 memref
- **Vector 操作**: 类似处理

**限制**:

- 要求 identity 或 strided 布局
- Transfer 操作要求 inbounds 访问和 identity/minor_identity 排列映射

**核心算法：

1. **线性化计算**：

```cpp
static std::pair<Value, Value> getFlattenMemrefAndOffset(
    OpBuilder &rewriter, Location loc, Value source, ValueRange indices) {
  // 提取步幅元数据
  memref::ExtractStridedMetadataOp stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);

  // 计算线性化索引
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(...);

  // 创建一维reinterpret_cast
  return std::make_pair(
      rewriter.create<memref::ReinterpretCastOp>(...),
      getValueFromOpFoldResult(rewriter, loc, linearizedIndices));
}
```

2. **操作重写**：

- Load/Store操作：添加线性化索引
- SubView操作：计算新的offset/size/stride
- Copy操作：更新源和目标

**转换示例**：

```text
// 转换前
%0 = memref.alloc() : memref<4x8xf32>
%1 = memref.load %0[%i, %j] : memref<4x8xf32>

// 转换后
%0 = memref.alloc() : memref<32xf32>
%idx = arith.muli %i, %c8 : index
%linear_idx = arith.addi %idx, %j : index
%1 = memref.load %0[%linear_idx] : memref<32xf32>
```

**优化效果**：

- 简化地址计算
- 提高缓存利用率
- 为SIMD向量化铺路

### 3.8 ExpandStridedMetadataPass

**文件**: `ExpandStridedMetadata.cpp`

**功能**：将修改MemRef元数据的操作展开为显式的元数据计算序列。

**技术原理**:

使用 affine 表达式显式计算步长、偏移和大小，使元数据操作的效果可被分析。

**Subview 展开**:

```cpp
// 新步长: newStrides#i = baseStrides#i * subStrides#i
// 新偏移: offset = baseOffset + sum(subOffsets#i * baseStrides#i)
// 新大小: sizes = subSizes
```

**ExpandShape 展开**:

```cpp
// 扩展大小: expandedSizes#i = baseSizes#groupId / product(expandShapeSizes#j for j != i)
// 扩展步长: expandedStrides#i = origStrides#reassDim * product(expandShapeSizes#j for j <= i)
```

**CollapseShape 展开**:

```cpp
// 折叠大小: collapsedSize = prod(origSizes#i in group)
// 折叠步长: collapsedStride = 最内层维度的步长
```

**Alloc 展开**: 计算恒等步长布局

**核心数据结构：**

```cpp
struct StridedMetadata {
  Value basePtr;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};
```

**优化效果**：

- 使元数据计算显式化
- 为后续优化提供更多分析信息
- 简化后端代码生成

**示例1 - SubView**:

转换前:

```text
%0 = memref.alloc() : memref<10x20xf32>                                                             
%1 = memref.subview %0[5, 3][3, 4][1, 1]                                                            
     : memref<10x20xf32> to memref<3x4xf32, offset: [?], strides: [20, 1]>   
```

转换过程：

```
// 1. 提取源的元数据                                                                                
%base, %offset_base, %sizes_base, %strides_base =                                                   
    memref.extract_strided_metadata %0                                                              
//     ↑ base_ptr                                                                                   
//         ↑ offset (0)                                                                             
//             ↑ sizes [10, 20]                                                                     
//                 ↑ strides [20, 1]                                                                

// 2. 计算新的 strides（每个维度：base_stride * sub_stride）                                        
%new_stride0 = affine.apply (s0 * s1) (%sub_stride0, %stride_base0)                              
//            = affine.apply (s0 * s1) (1, 20) = 20                                                 
%new_stride1 = affine.apply (s0 * s1) (%sub_stride1, %stride_base1)                                 
//            = affine.apply (s0 * s1) (1, 1) = 1                                                   

// 3. 计算新的 offset                                                                               
// offset = base_offset + sub_offsets[0] * base_strides[0] + sub_offsets[1] * base_strides[1]       
%new_offset = affine.apply (s0 + s1*s2 + s3*s4)                                                     
              (%offset_base, %sub_offset0, %stride_base0, %sub_offset1, %stride_base1)              
//            = affine.apply (s0 + s1*s2 + s3*s4) (0, 5, 20, 3, 1)                                  
//            = 5 * 20 + 3 * 1 = 103                                                                

// 4. 新的 sizes 直接来自 subview 的 sizes                                                          
// sizes = [3, 4]                                                                                   

// 5. 用 reinterpret_cast 重建 memref                                                               
%1 = memref.reinterpret_cast %base           
      offset [%new_offset], sizes [3, 4], strides [%new_stride0, %new_stride1]                      
      : memref<10x20xf32> to memref<3x4xf32, offset: [103], strides: [20, 1]>   
```

转换后:

```text
%base, %offset_0, %size_0, %size_1, %stride_0, %stride_1 =                                          
    memref.extract_strided_metadata %0                                                              

%offset_new = affine.apply (d0) -> (d0 + 5*20 + 3*1) (%offset_0)                                    

%1 = memref.reinterpret_cast %base                                                                  
      offset [%offset_new], sizes [3, 4], strides [20, 1]                                        
      : memref<10x20xf32> to memref<3x4xf32, ...>      
```

**示例2 - ExpandShape**:

转换前：

```llvm
// 把一维 [12] 展开成二维 [3, 4]                                                                 
%0 = memref.alloc() : memref<12xf32>                                                                
%1 = memref.expand_shape %0 [[0], [1]]                                                              
     : memref<12xf32> into memref<3x4xf32>   
```

转换过程：

```llvm
// 1. 提取元数据                                                                                    
%base, %offset, %size_0, %stride_0 =                                                                
    memref.extract_strided_metadata %0                                                              
// size_0 = 12, stride_0 = 1                                                                        

// 2. 计算新的 sizes                                                                                
// size_0_new = size_0 / size_1_new = 12 / 4 = 3                                                    
// size_1_new = 4                                                                                   
%size_0_new = affine.apply (d0) -> (d0 / 4) (%size_0)  // 3                                         
%size_1_new = 4                                                                                     

// 3. 计算新的 strides                                                                              
// stride_0_new = stride_0 * size_1_new = 1 * 4 = 4                                                 
// stride_1_new = stride_0 = 1                                                                      
%stride_0_new = affine.apply (d0) -> (d0 * 4) (%stride_0)  // 4                                     
%stride_1_new = %stride_0  // 1                                                                     

// 4. 重建 memref                                                                                   
%1 = memref.reinterpret_cast %base                                                                  
      offset [%offset], sizes [3, 4], strides [4, 1]                                                
      : memref<12xf32> into memref<3x4xf32>     
```

**示例3 - CollapseShape**:

转换前：

```llvm
// 把二维 [3, 4] 折叠成一维 [12]
%0 = memref.alloc() : memref<3x4xf32>
%1 = memref.collapse_shape %0 [[0, 1]]
     : memref<3x4xf32> into memref<12xf32>   
```

转换过程：

```llvm
// 1. 提取元数据
%base, %offset, %size_0, %size_1, %stride_0, %stride_1 =
    memref.extract_strided_metadata %0
// size_0 = 3, size_1 = 4, stride_0 = 4, stride_1 = 1

// 2. 计算新的 size（折叠维度的乘积）
%size_new = affine.apply (d0, d1) -> (d0 * d1) (%size_0, %size_1)  // 12

// 3. 计算新的 stride（最内层维度的 stride）
%stride_new = %stride_1  // 1

// 4. 重建 memref
%1 = memref.reinterpret_cast %base
      offset [%offset], sizes [12], strides [1]
      : memref<3x4xf32> into memref<12xf32>
```

**示例4 - ExtractStridedMetadata 优化**:

转换前：

```llvm
%0 = memref.alloc() : memref<10x20xf32>
%1 = memref.subview %0[5, 3][3, 4][1, 1] : ...
%base, %offset, %sizes, %strides = memref.extract_strided_metadata %1
```

转换过程：

```llvm
// 直接展开（跳过 subview）

// 不先创建 subview，直接计算 subview 的元数据
%base, %offset_0, %size_0, %size_1, %stride_0, %stride_1 =
    memref.extract_strided_metadata %0

%offset_new = affine.apply (d0 + d1*d2 + d3*d4)
              (%offset_0, %c5, %stride_0, %c3, %stride_1)
//    = 0 + 5*20 + 3*1 = 103

%sizes_new = [3, 4]
%strides_new = [20, 1]

// 结果：%base, %offset_new, %sizes_new, %strides_new
```

这个优化相比示例1是**更激进的优化**，完全消除了中间的 subview op，直接暴露底层计算。

### 3.9 ResolveShapedTypeResultDimsPass

**文件**: `ResolveShapedTypeResultDims.cpp`

**功能**：通过`InferShapedTypeOpInterface`解析`memref.dim`操作。

**Dim 折叠**: 使用 `reifyReturnTypeShapes` 获取形状，然后提取维度。

**迭代参数**: 在 `scf.forall` 中，将 `%arg0` 的 dim 替换为对应初始参数的 dim

**核心算法**：

```cpp
// 对于实现 InferShapedTypeOpInterface 的操作
struct DimOfShapedTypeOpInterface
    : public OpRewritePattern<memref::DimOp> {
  LogicalResult matchAndRewrite(memref::DimOp dimOp, ...) {
    // 调用 reifyReturnTypeShapes
    // 提取维度
    // 替换 dim 操作
  }
};

// 对于 scf.forall 的迭代参数
struct IterArgsToInitArgs : public OpRewritePattern<memref::DimOp> {
  // 将 iter_args 的 dim 替换为 init_args 的 dim
};
```

**优化效果**：

- 消除运行时dim查询
- 使形状信息在编译时可用
- 提高类型推断能力

**示例**:

转换前:

```text
%alloc = memref.alloc(%size) : memref<?xf32>
%d = memref.dim %alloc, 0 : memref<?xf32>
```

转换后:

```text
// dim操作直接使用size值
%d = %size  // 假设size是索引值

// 或如果size需要转换:
%d = arith.index_cast %size : i32 to index
```

**Forall迭代参数示例**:

转换前:

```text
scf.forall (%arg0) in (%size) shared_outs(%init = %output) -> (memref<?xf32>) {
  %d = memref.dim %arg0, 0 : memref<?xf32>
  // 使用 %d
  scf.forall.in_parallel {
    tensor.parallel_insert %val into %output[...]
  }
}
```

转换后:

```text
scf.forall (%arg0) in (%size) shared_outs(%init = %output) -> (memref<?xf32>) {
  %d = %size  // 直接使用初始大小
  // 使用 %d
  scf.forall.in_parallel {
    tensor.parallel_insert %val into %output[...]
  }
}
```

**使用场景**:

- 消除冗余的dim操作
- 形状传播优化
- 静态形状推断

### 3.10 ComposeSubView

**功能**：将嵌套的SubView操作组合为单个SubView。

**限制条件**:

- 源 SubViewOp 不能是降秩操作（rank-reducing）
- 只支持静态大小
- 支持静态和动态偏移量

**核心算法**（位于`ComposeSubView.cpp`）：

1. **模式匹配**：

```cpp
struct ComposeSubViewOpPattern : public OpRewritePattern<memref::SubViewOp> {
  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 检查源是否是SubView
    auto sourceOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceOp) return failure();
  }
};
```

2. **组合计算**：

```cpp
// 步幅：strides[i] = sourceStrides[i] * opStrides[i]
// 偏移：offset[i] = sourceOffset[i] + opOffset[i] * sourceStrides[i]
// 大小：取最终的大小（最小）
```

**转换示例**：

```text
// 转换前
%0 = memref.subview %base[10, 20] [5, 5] [1, 1] : ...
%1 = memref.subview %0[2, 3] [2, 2] [1, 1] : ...

// 转换后
%1 = memref.subview %base[12, 23] [2, 2] [1, 1] : ...
```

**优化效果**：

- 减少中间SubView操作
- 简化访问路径
- 提高分析效率

### 3.11 FoldMemRefAliasOps

**功能**：将对子视图的加载/存储折叠为对原始MemRef的加载/存储。

**核心模式**（位于`FoldMemRefAliasOps.cpp`）：

**示例 - SubView折叠**:

转换前:

```text
%subview = memref.subview %base[10, 20] [30, 40] [1, 1]
    : memref<100x100xf32> to memref<30x40xf32>
%val = memref.load %subview[%i, %j]
    : memref<30x40xf32>
```

转换后:

```text
%val = memref.load %base[10 + %i, 20 + %j]
    : memref<100x100xf32>
```

**示例 - ExpandShape折叠**:

转换前:

```text
%expanded = memref.expand_shape %base [[0, 1], [2]]
    : memref<12x4xf32> into memref<3x4x4xf32>
%val = memref.load %expanded[%i, %j, %k]
```

转换后:

```text
%linear_idx = %i * 16 + %j * 4 + %k  // 线性化索引
%val = memref.load %base[%linear_idx / 4, %linear_idx % 4]
```

**算法流程**：

```cpp
static LogicalResult resolveSourceIndicesExpandShape(
    Location loc, PatternRewriter &rewriter,
    memref::ExpandShapeOp expandShapeOp, ValueRange indices,
    SmallVectorImpl<Value> &sourceIndices, bool startsInbounds) {
  // 遍历reassociation groups
  // 对每个group计算线性化索引
  // 使用affine.linearize_index op
}
```

**优化效果**：

- 减少间接访问层级
- 提高内存访问效率
- 为后续优化提供更清晰的访问模式

### 3.12 ExtractAddressComputations

**功能**：将有偏移量的 Load/Store 重写为对Subview的 Load/Store，其中offset 是全0开始。此 Pass 与 **FoldMemRefAliasOps** 是逆操作。

```cpp
// FoldMemRefAliasOps                                                                                   
load(subview(src)[i, j]) → load(src[i+offset, j+offset])                                             
消除 subview，把 offset 吸收进索引。                                                                 

// ExtractAddressComputations                                                                           
load(src[i+offset, j+offset]) → load(subview(src[offset,...][1,1][1,1])[0, 0])                       
把索引中的 offset 提取出来，生成一个 subview，访问时索引归零。                                       

// 互为逆变换，用于不同场景：                                                                           
- FoldMemRefAliasOps：消除间接层，让访问更直接，利于分析和 lowering                                  
- ExtractAddressComputations：统一访问模式为 [0,0,...]，方便某些 backend（如 NVGPU）要求 load/store  
的索引必须为零   
```

**示例**：

```text
// 转换前
%val = memref.load %base[%off0, %off1, %i, %j]

// 转换后
%subview = memref.subview %base[%off0, %off1] [1, 1] [1, 1]
%val = memref.load %subview[0, 0, %i, %j]
```

**支持的Op**：

- `memref::LoadOp`, `memref::StoreOp`
- `nvgpu::LdMatrixOp`
- `vector::TransferReadOp`, `vector::TransferWriteOp`

**优化效果**：

- 分离地址计算和数据访问
- 提高地址计算重用机会
- 为向量化优化提供基础

### 3.13 IndependenceTransforms

**文件**: `IndependenceTransforms.cpp`

**功能**：使操作独立于特定的依赖值。包括三个核心功能：

* buildIndependentOp / replaceWithIndependentOp  

```
核心功能：把一个 alloca 的动态 size 替换为不依赖某些值（independencies）的上界。                     

// 假设 %n 是循环变量，alloca 依赖它                                                                 
%buf = memref.alloca(%n) : memref<?xf32>                                                             

// 转换后：用 %n 的上界 %ub 分配更大的 alloca，再 subview 出实际大小                                 
%big = memref.alloca(%ub) : memref<?xf32>                                    
%buf = memref.subview %big[0][%n][1] : ...                                                           

用途：把 alloca 提升到循环外——循环内的 alloca size                                                   
依赖循环变量，提升后用上界做一次性分配，避免每轮迭代重新分配栈内存。  
```

* replaceAndPropagateMemRefType 

```
把旧 op 替换为新 op 后，用 unrealized_conversion_cast 桥接类型差异，并尽量将 cast 向下推（propagate  
through subview 等），最终清理掉无用的 cast。

这是替换操作时处理 memref type 变化的工具函数。  
```

* allocToAlloca 

```
 简单地把 alloc + dealloc 对替换为 alloca（栈分配）。前提是在同一 block 内能找到对应的 dealloc。
```

**核心算法**：

```cpp
static FailureOr<OpFoldResult> makeIndependent(
    OpBuilder &b, Location loc, OpFoldResult ofr, ValueRange independencies) {
  // 使用ValueBoundsConstraintSet计算独立边界
  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeIndependentBound(
          boundMap, mapOperands, presburger::BoundType::UB,
          ofr, independencies, /*closedUB=*/true)))
    return failure();

  // 物化计算出的边界
  return affine::materializeComputedBound(b, loc, boundMap, mapOperands);
}

// 应用示例：使Alloca大小独立于循环变量
FailureOr<Value> buildIndependentOp(OpBuilder &b, memref::AllocaOp allocaOp,
                                   ValueRange independencies) {
  // 计算独立的上界大小
  // 创建新的Alloca
  // 包装在SubView中
}
```

**优化效果**：

- 允许分配提升到循环外
- 减少分配次数
- 提高并行性

**示例**:

转换前:

```text
%size = "unknown_size"() : () -> index
%alloc = memref.alloc(%size) : memref<?xf32>
```

转换后:

```text
// 计算独立上界
%static_size = arith.constant 100 : index
%independent_alloc = memref.alloc(%static_size) : memref<100xf32>

// 使用subview包装获得原始类型
%cast = memref.subview %independent_alloc[0][%size][1]
    : memref<100xf32> to memref<?xf32>
```

**Alloc到Alloca转换**:

转换前:

```text
%alloc = memref.alloc() : memref<100xf32>
// ... 使用 %alloc ...
memref.dealloc %alloc : memref<100xf32>
```

转换后:

```text
%alloca = memref.alloca() : memref<100xf32>
// ... 使用 %alloca ...
// (不需要dealloc)
```

**使用场景**:

- 依赖分析
- 生命周期优化
- 栈分配优化
- 并行化准备

### 3.14 ExpandOpsPass

**功能**：将高层次的MemRef操作转换为更基础的操作。

**主要转换**：

- `memref.reshape` → `memref.reinterpret_cast`（当形状静态时）

**主要代码**：

```cpp
// 转换reshape为reinterpret_cast
struct MemRefReshapeOpConverter : public OpRewritePattern<memref::ReshapeOp> {
  LogicalResult matchAndRewrite(memref::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    // 计算sizes和strides
    // 使用affine表达式计算stride
    // 创建reinterpret_cast操作
  }
};
```

**优化效果**：

- 将动态形状计算转换为显式的大小和步幅计算
- 为后续优化提供更清晰的IR表示

**示例**:

转换前:

```text
%shape = arith.constant [2, 3, 4] : index
%reshaped = memref.reshape %src(%shape)
    : (memref<24xf32>) -> memref<2x3x4xf32>
```

转换后:

```text
%reshaped = memref.reinterpret_cast %src to
    offset: [0], sizes: [2, 3, 4], strides: [12, 4, 1]
    : memref<24xf32> to memref<2x3x4xf32>
```

**使用场景**:

- 将高级reshape操作降级为底层操作
- 为代码生成做准备
- 简化分析

### 3.15 ExpandRealloc

**文件**: `ExpandRealloc.cpp`

**作用**: 扩展 memref.realloc 操作 - 将 realloc 分解为其组成操作

**条件分配和复制**:

1. 比较当前缓冲区大小与请求大小
2. 如果旧缓冲区较小，分配新缓冲区，复制数据，释放旧缓冲区
3. 如果旧缓冲区足够大，使用 reinterpret_cast 调整大小

**实现**: 使用 `scf.if` 实现条件逻辑

**核心模式**:

```cpp
struct ExpandReallocOpPattern : OpRewritePattern<memref::ReallocOp> {
  LogicalResult matchAndRewrite(memref::ReallocOp op, ...) {
    // 提取源缓冲区大小
    // 创建条件: if (source_size < requested_size)
    // then分支: 分配新缓冲区，复制数据
    // else分支: reinterpret_cast
  }
};
```

**示例**:

转换前:

```text
%result = memref.realloc %source[%new_size]
    : (memref<?xf32>, index) -> memref<?xf32>
```

转换后:

```text
%source_size = memref.dim %source, 0 : memref<?xf32>
%result = scf.if (%source_size < %new_size) -> (memref<?xf32>) {
  // 需要重新分配
  %new_alloc = memref.alloc(%new_size) : memref<?xf32>
  memref.copy %source, %new_alloc : memref<?xf32>
  memref.dealloc %source : memref<?xf32>
  scf.yield %new_alloc : memref<?xf32>
} else {
  // 可以重用现有缓冲区
  %view = memref.reinterpret_cast %source
      to offset: [0], sizes: [%new_size], strides: [1]
  scf.yield %view : memref<?xf32>
}
```

**使用场景**:

- 代码生成（目标平台不支持realloc）
- 显式内存管理
- 优化分析

### 3.16 MultiBufferPass

**功能**：通过数组扩展消除循环迭代之间的临时分配依赖。

**核心算法**（位于`MultiBuffer.cpp`）：

1. **候选识别**：

```cpp
// 查找在循环内分配的MemRef
// 检查是否有完整的写覆盖（overrideBuffer）
// 检查是否可以使用多缓冲
```

2. **缓冲区扩展**：

```cpp
FailureOr<memref::AllocOp> multiBuffer(
    RewriterBase &rewriter, memref::AllocOp allocOp,
    unsigned multiBufferingFactor, bool skipOverrideAnalysis) {

  // 1. 获取原始分配大小
  SmallVector<OpFoldResult> originalSizes = allocOp.getMixedSizes();

  // 2. 创建新的分配（多倍大小）
  // 在新维度上扩展
  SmallVector<OpFoldResult> newSizes = originalSizes;
  newSizes.insert(newSizes.begin(),
                  rewriter.getIndexAttr(multiBufferingFactor));

  // 3. 创建新分配并包装在subview中
  // 返回原始大小的subview
}
```

3. **索引更新**：

```cpp
// 更新所有使用点，添加循环归纳变量作为索引
// %new_alloc = memref.alloc(%factor, %size) : memref<?xsize>
// %subview = memref.subview %new_alloc[%iv, 0] [%c1, %size] [1, 1]
```

**转换示例**：

```text
// 转换前
affine.for %i = 0 to 100 {
  %temp = memref.alloc() : memref<128xf32>
  // 使用%temp...
  memref.dealloc %temp : memref<128xf32>
}

// 转换后（factor=2）
%temp = memref.alloc() : memref<2x128xf32>
affine.for %i = 0 to 100 {
  %idx = arith.remsi %i, %c2 : index
  %subview = memref.subview %temp[%idx, 0] [1, 128] [1, 1]
  // 使用%subview...
}
memref.dealloc %temp : memref<2x128xf32>
```

**优化效果**：

- 减少内存分配/释放开销
- 消除迭代间依赖，提高并行性
- 软件流水化的基础

**使用场景**：

主要被 `MemRefTransformOps` 调用，即通过 Transform dialect 的 `transform.memref.multibuffer` op 使用：

```
// 在 transform IR 里                                                                                
transform.memref.multibuffer %alloc {factor = 2 : i64}                                               
  : (!transform.any_op) -> !transform.any_op      
```

 这是推荐的使用方式——通过 Transform dialect 驱动，而不是硬编码在某个 pass 里。

## 四、Pass依赖关系

### 典型的Pass流水线

```
1. ExpandStridedMetadataPass
   ↓ (使元数据显式化)
2. ExpandOpsPass
   ↓ (展开高层次操作)
3. ComposeSubView
   ↓ (组合视图操作)
4. NormalizeMemRefsPass
   ↓ (规范化布局)
5. FoldMemRefAliasOpsPass
   ↓ (折叠别名操作)
6. FlattenMemRefsPass
   ↓ (扁平化内存)
7. MultiBufferPass
   ↓ (多缓冲优化)
8. ExtractAddressComputations
   ↓ (提取地址计算)
9. (后续向量化、并行化等优化)
```

### Pass选择建议

| 优化目标     | 推荐Pass                                        |
| ------------ | ----------------------------------------------- |
| 简化内存布局 | NormalizeMemRefsPass                            |
| 减少间接访问 | FoldMemRefAliasOpsPass                          |
| 提高缓存效率 | FlattenMemRefsPass                              |
| 循环并行化   | MultiBufferPass                                 |
| 向量化准备   | ExtractAddressComputations + FlattenMemRefsPass |
| 分配优化     | AllocationOpInterface + IndependenceTransforms  |

## 五、接口和Trait

### 关键接口

1. **InferShapedTypeOpInterface**
   - 推断形状类型操作的结果形状
   - 用于动态形状的编译时推断

2. **AllocationOpInterface**
   - 内存分配操作的统一接口
   - 支持分配提升、克隆等优化

3. **OffsetSizeAndStrideOpInterface**
   - 描述偏移-大小-步长模式的接口
   - 用于Subview等操作

4. **ViewLikeOpInterface**
   - 视图操作的通用接口
   - 支持视图操作的统一处理

5. **BufferViewFlowOpInterface**
   - 缓冲区视图流分析
   - 用于别名分析

### 关键Trait

1. **MemRefsNormalizable**
   - 标记可规范化的操作
   - NormalizeMemRefsPass使用

2. **SameOperandsAndResultShape**
   - 操作数和结果形状相同
   - 用于形状推断

3. **OperandsAreShapeConvertible**
   - 操作数可转换为形状
   - 用于reshape等操作

## 六、测试用例解析

### 重要测试场景

#### 1. 基本操作测试

- `ops.mlir`：所有基本操作
- `expand-ops.mlir`：操作扩展测试
- `expand-strided-metadata.mlir`：元数据扩展测试

#### 2. Pass测试

- `canonicalize.mlir`：标准化测试
- `normalize-memrefs.mlir`：规范化测试
- `fold-memref-alias-ops.mlir`：别名折叠测试
- `multibuffer.mlir`：多缓冲测试
- `flattened-memref.mlir`：扁平化测试

#### 3. 优化效果测试

- `mem2reg.mlir`：内存到寄存器提升
- `compose-subview.mlir`：Subview组合测试
- `resolve-shaped-type-result-dims.mlir`：形状解析测试

### 使用模式示例

#### 多缓冲优化

```text
// 优化前
affine.for %i = 0 to 100 {
  %temp = memref.alloc() : memref<128xf32>
  // ... 使用%temp
  memref.dealloc %temp
}

// 应用MultiBufferPass后
%temp = memref.alloc() : memref<2x128xf32>
affine.for %i = 0 to 100 {
  %idx = arith.uremi %i, %c2 : index
  %sub = memref.subview %temp[%idx] [1] [1] : ... to memref<128xf32>
  // ... 使用%sub
}
memref.dealloc %temp
```

## 七、总结

MLIR MemRef方言的Pass系统提供了全面的内存优化能力：

1. **层次化设计**：从高层次的Normalize到低层次的Flatten
2. **渐进式优化**：每个Pass专注于特定转换
3. **可组合性**：Pass可以灵活组合形成优化流水线
4. **与Affine深度集成**：共享分析和优化能力
5. **丰富的接口**：支持定制化扩展

通过合理使用这些Pass，可以显著提高内存访问效率，为后续的向量化、并行化等优化奠定基础。
