---
title: "【MLIR】Linalg的通用Fusion优化分析"
description: "【MLIR】Linalg的通用Fusion优化分析 从宏观到微观，逐步深入理解 Linalg 的 Fusion 优化 1. Fusion 是什么？ Fusion（融合） = 把多个操作合并成一个，减少中间结果的内存读写。 直观示例 ： 相关文件 ： 2. Fusion 的核心概念 2.1 生产…"
slug: "mlirlinalg-fusion-analysis"
legacyId: 19558642
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19558642"
pubDate: 2026-01-31
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Linalg"]
featured: true
---

#【MLIR】Linalg的通用Fusion优化分析

> 从宏观到微观，逐步深入理解 Linalg 的 Fusion 优化

---

## 1. Fusion 是什么？

**Fusion（融合）** = 把多个操作合并成一个，减少中间结果的内存读写。

**直观示例**：

```
融合前: 两个操作，中间结果写内存
┌─────────────┐    %1     ┌─────────────┐    %2     ┌─────────────┐
│   MatMul    │ ────────> │    Add      │ ────────> │   Output    │
│  A × B → C  │           │   C + 1.0   │           │     D       │
└─────────────┘           └─────────────┘           └─────────────┘
     写内存                  读/写内存                  读内存

融合后: 一个操作，中间结果留寄存器
┌─────────────────────────────────────────┐
│         Fused MatMul + Add              │
│    (A × B) + 1.0 → D                    │
│  (无需中间张量 C 的内存读写)                │
└─────────────────────────────────────────┘
```

**相关文件**：

```
mlir/lib/Dialect/Linalg/Transforms/
├── Fusion.cpp                      # 核心 Tensor 融合
├── ElementwiseOpFusion.cpp         # 逐元素操作融合
└── FusePadOpWithLinalgProducer.cpp # Pad 操作融合
```

---

## 2. Fusion 的核心概念

### 2.1 生产者-消费者关系

```
生产者 (Producer)           消费者 (Consumer)
┌──────────────┐             ┌──────────────┐
│ %1 = add(A)  │ ──── %1 ──> │ %2 = mul(%1) │
└──────────────┘             └──────────────┘
     产生数据                     使用数据
```

**融合条件**：

1. 生产者只有一个消费者（或消费者可以内联）
2. 操作的计算模式兼容（索引映射能对得上就能融合，见第 2.3 节）
3. 融合后能减少内存访问

### 2.2 Buffer vs Tensor 语义

| 特性     | Buffer (MemRef)  | Tensor                 |
| -------- | ---------------- | ---------------------- |
| 语义     | 可变内存         | 不可变值               |
| 切片操作 | `memref.subview` | `tensor.extract_slice` |
| 内存分配 | 不分配新内存     | 返回新张量值           |
| 修改方式 | 原地修改         | 生成新值               |

```llvm
// Buffer 语义
%subview = memref.subview %buffer[0, 0][32, 32][1, 1]  // 指向原 buffer 的一部分
%1 = linalg.generic ins(%subview) outs(%output)        // 原地修改

// Tensor 语义
%slice = tensor.extract_slice %tensor[0, 0][32, 32][1, 1]  // 新的张量值
%1 = linalg.generic ins(%slice) outs(%output)              // 不修改原 tensor
```

### 2.3 索引映射 (Indexing Maps)

描述操作数如何映射到循环迭代空间：

```cpp
indexing_maps = [
  affine_map<(i, j) -> (i, j)>,  // 输入 A: 使用 (i, j)
  affine_map<(i, j) -> (j)>,     // 输入 B: 使用 j
  affine_map<(i, j) -> (i)>      // 输出 C: 使用 i
]
```

**示例解读**：`affine_map<(i, j) -> (i + j, j)>`

- 左侧 `(i, j)` = 循环索引
- 右侧 `(i + j, j)` = Tensor 索引

等价代码：

```cpp
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        B[i + j][j] = A[i][j];  // 注意：索引是 i+j
    }
}
// 结果 B 的 Shape = [M + N - 1, N]
```

**为什么需要索引映射？**
它让 LinalgOp 能够描述"哪个操作数用到哪些循环维度"，从而判断两个操作能否融合。 

---

## 3. TileAndFuse 工作流

### 3.1 整体流程

#### 流程-文字版（Greedy Fusion 算法）

```
1. 收集阶段
   ┌─────────────────────────────────────┐
   │ %1 = linalg.generic { ... }         │  → linalgOps[0]
   │ %2 = linalg.matmul ins(%1, %B)      │  → linalgOps[1]
   │ %3 = linalg.generic ins(%2) outs(%C)│  → linalgOps[2]
   └─────────────────────────────────────┘

2. 反向遍历 reverse(linalgOps) = [%3, %2, %1]
   注意：遍历顺序固定，但操作内容会更新

   第一轮: linalgOp = %3 (消费者)
     └─> 遍历操作数，发现 %2 是 Tensor
         └─> 尝试 fuseProducerOfTensor(%2)
             └─> 如果 %2 有 extract_slice → 融合成功
                 └─> 更新: linalgOps[1] = %2_fused
             └─> 如果 %2 无 extract_slice → 跳过

   第二轮: linalgOp = %2 或 %2_fused
     └─> 遍历操作数，%1 是 Tensor
         └─> 尝试 fuseProducerOfTensor(%1)
             └─> 如果 %1 有 extract_slice → 融合成功
                 └─> 更新: linalgOps[0] = %1_fused
             └─> 如果 %1 无 extract_slice → 跳过

   第三轮: linalgOp = %1 或 %1_fused
     └─> 遍历操作数，%A 是函数参数
         └─> fuseProducerOfTensor(%A) 返回 failure
             └─> 跳过

3. 最终状态: linalgOps = [%1_fused, %2_fused, %3]
   所有可融合的操作都融合到循环内
```

**关键点**：

- `reverse(linalgOps)` 的遍历**顺序**是固定的：`%3` → `%2` → `%1`
- 操作的**内容**会在融合后更新：`%2` → `%2_fused`
- **只有存在 `extract_slice` 时才能融合**（Tiling 后才有）
- 第二轮处理的是融合后的版本，它本身的输入可能也需要融合

---

#### 流程-样例版（Greedy Fusion + Tiling）

**前置条件**: 先对消费者 `%3` 进行 Tiling，生成 `extract_slice`

```
原始代码:
%1 = linalg.generic ins(%A) outs(%init1) { add }
%2 = linalg.matmul ins(%1, %B) outs(%init2)
%3 = linalg.generic ins(%2) outs(%C) { mul }

Step 1: Tiling 消费者 %3（生成 extract_slice）
┌─────────────────────────────────────┐
│ scf.for %ii = 0 to 128 step 32 {    │
│   scf.for %jj = 0 to 128 step 32 {  │
│     %2_tile = extract_slice %2...   │  ← 生成切片
│     %3_tile = generic ins(%2_fused) │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘

Step 2: 融合生产者（重复直到无法继续）

第一轮融合: 处理 %3，发现 %2_tile 通过 extract_slice 引用 %2
  └─> fuseProducerOfTensor 融合 %2 (matmul)
      └─> 将 %2 搬进循环，生成 %2_fused

第二轮融合: 处理 %2_fused，发现 %1_tile 通过 extract_slice 引用 %1
  └─> fuseProducerOfTensor 融合 %1 (generic add)
      └─> 将 %1 搬进循环，生成 %1_fused

第三轮融合: 处理 %1_fused，输入是 %A（函数参数）
  └─> 无法融合，结束
┌───────────────────────────────────────────────┐
│ scf.for %ii = 0 to 128 step 32 {              │
│   scf.for %jj = 0 to 128 step 32 {            │
│     %1_tile = extract_slice %1...             │  ← 融合 matmul (%2) 的结果
│     %B_tile = extract_slice %B...             │
│     %2_fused = matmul ins(%1_tile, %B_tile)   │
│                                               │
│     %A_tile = extract_slice %A...             │  ← 融合 generic (%1) 的结果
│     %1_fused = generic ins(%A_tile)           │
│                                               │
│     %3_tile = generic ins(%2_fused)           │
│   }                                           │
│ }                                             │
└───────────────────────────────────────────────┘
```

#### 流程-源码版（Greedy Fusion）

**fuseLinalgOpsGreedily - 核心算法**，源码位置: `mlir/test/lib/Dialect/Linalg/TestLinalgFusionTransforms.cpp`

```cpp
static LogicalResult fuseLinalgOpsGreedily(func::FuncOp f) {
  OpBuilder b(f);

  // 步骤 1: 收集所有 Linalg 操作
  SmallVector<LinalgOp, 8> linalgOps;
  f.walk([&](LinalgOp op) {
    if (op->getNumResults() <= 1)  // 只支持单结果
      linalgOps.push_back(op);
  });

  // 步骤 2: 反向遍历（从消费者到生产者）
  bool changed = false;
  for (LinalgOp linalgOp : llvm::reverse(linalgOps)) {
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      // 跳过 MemRef 类型
      if (isa<MemRefType>(opOperand.get().getType()))
        continue;

      // 只处理 Tensor 类型输入
      if (isa<RankedTensorType>(opOperand.get().getType())) {
        // 跳过输出操作数，详见第3.3节
        if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())
          continue;

        // 尝试融合，入参是 linalgOp 的输入，详见第5.1节
        auto info = fuseProducerOfTensor(b, opOperand);
        if (failed(info)) continue;

        // 更新操作列表
        auto *originalOp = info->originalProducer.getOperation();
        auto *it = llvm::find(linalgOps, originalOp);
        *it = info->fusedProducer;

        changed = true;
      }
    }
  }

  return changed ? success() : failure();
}
```

#### 关键点

1. **反向遍历**：从消费者向生产者融合，保证融合顺序正确
2. **更新操作列表**：融合后替换原操作，避免重复融合
3. **不立即删除**：保留原操作让 DCE 处理，避免 use-def 链断裂
4. **迭代执行**：与规范化 Pass 配合，持续融合直到稳定
5. **只融合输入**：跳过 `DpsInit`（输出操作数），避免复杂依赖

### 3.2 为什么需要 Tiling？

**问题**：直接融合整个大张量 → 内存占用大，缓存利用率低

**解决**：先 Tiling 分块，再融合到循环内 → 每次只处理一小块数据

```
不 Tiling 直接融合:
融合后操作处理整个 128×128 → 内存压力大

Tile 后再融合:
融合后操作只处理 32×32 → 数据可放入 L1 缓存
```

### 3.3 "跳过输出操作数"是什么？

本小节详细介绍下3.1中源码片段中的第21行**条件**：`if (opOperand.getOperandNumber() >= linalgOp.getNumDpsInputs())`

#### DpsInputs 是什么？

```
DpsInit = Destination-Passing Style Initializer (输出缓冲区)
DpsInputs = Destination-Passing Style Inputs (输入张量)

在 LinalgOp 中：
┌────────────────────────────────────────────┐
│ %result = linalg.generic                   │
│   ins(%A, %B : tensor<128x128xf32>)        │  ← DpsInputs (输入)
│   outs(%init : tensor<128x128xf32>) {      │  ← DpsInit (输出)
│   ^bb0(%a: f32, %b: f32, %out: f32):       │
│     ...                                    │
│     linalg.yield %res                      │
│ }                                          │
└────────────────────────────────────────────┘
```

#### getOperandNumber() 返回什么？

```
操作数索引:
┌────────────────────────────────────────┐
│ ins(%A, %B) outs(%init)                │
│   0    1        2                      │
│   ↑    ↑        ↑                      │
│   └────┴────────┴─ getOperandNumber()  │
└────────────────────────────────────────┘

getNumDpsInputs() = 2  (%A, %B)
getNumDpsInits() = 1   (%init)
```

#### 条件判断逻辑

```cpp
// 遍历所有操作数
for (OpOperand &opOperand : linalgOp->getOpOperands()) {
  unsigned opNum = opOperand.getOperandNumber();
  unsigned numInputs = linalgOp.getNumDpsInputs();

  // 如果: opNum >= numInputs
  // 说明: 这是输出操作数，跳过

  if (opNum >= numInputs)
    continue;  // 不融合输出操作数
}
```

#### 为什么只融合输入？

```
输入操作数: 可以融合其生产者
┌────────────────────────────────────────┐
│ %1 = linalg.generic ins(%A) { ... }    │  ← %1 的生产者
│ %2 = linalg.matmul ins(%1, %B)         │  ← 融合 %1
└────────────────────────────────────────┘

输出操作数: 不能融合（它就是要写入的地方）
┌────────────────────────────────────────┐
│ %result = linalg.generic               │
│   ins(%A) outs(%init)                  │  ← %init 是输出目标
│                                        │     不是要融合的对象
└────────────────────────────────────────┘
```

#### 为什么 for 循环条件不使用 getNumDpsInputs() ？

若使用 getNumDpsInputs()，则隐含假设：

> DPS inputs 一定排在 operand 列表的最前面

虽然 **当前 Linalg 确实如此**，但这是 **dialect-level 约定**，不是 `Operation` / `Pass` 的通用保证。

#### 具体示例

```cpp
// 假设有这个操作
%result = linalg.generic
  ins(%A : tensor<128x128xf32>,      // 操作数 0
       %B : tensor<128x128xf32>)     // 操作数 1
  outs(%init : tensor<128x128xf32>)  // 操作数 2

// 此时，getNumDpsInputs() = 2

// 遍历操作数:
opOperand = %A, getOperandNumber() = 0
  if (0 >= 2) → false → 处理 ✓

opOperand = %B, getOperandNumber() = 1
  if (1 >= 2) → false → 处理 ✓

opOperand = %init, getOperandNumber() = 2
  if (2 >= 2) → true → 跳过 ✗
```

---

## 4. 核心数据结构

### 4.1 ShapeDimension

记录"哪个张量的哪个维度"决定循环范围：

```cpp
struct ShapeDimension {
  Value shape;         // 张量
  unsigned dimension;  // 维度索引
};
```

**示例**：

```llvm
%A: tensor<100x200xf32>

ShapeDimension { shape: %A, dimension: 0 }  // 第 0 维，范围 100
ShapeDimension { shape: %A, dimension: 1 }  // 第 1 维，范围 200
```

### 4.2 FusionInfo

记录融合前后的操作：

```cpp
struct FusionInfo {
  LinalgOp originalProducer;  // 融合前的生产者
  LinalgOp fusedProducer;     // 融合后的生产者
};
```

### 4.3 Range

表示循环的范围：

```cpp
struct Range {
  OpFoldResult offset;  // 起始偏移
  OpFoldResult size;    // 大小
  OpFoldResult stride;  // 步长
};
```

---

## 5. 核心函数详解

本节示例：以融合 `%2` (matmul) 为例

```
原始代码（融合前）:
%A: tensor<128x128xf32>
%B: tensor<128x128xf32>
%C: tensor<128x128xf32>

%1 = linalg.generic ins(%A) outs(%init1) { A + 1.0 }   // 生产者1: add
%2 = linalg.matmul ins(%1, %B) outs(%init2)            // 生产者2: matmul
%3 = linalg.generic ins(%2) outs(%C) { ... }           // 消费者

Tile后（只看融合%2的场景）:
%2_tile = tensor.extract_slice %2[%ii, %jj][32, 32][1, 1]  // 偏移由循环变量决定
%3 = linalg.generic ins(%2_tile) outs(%C_tile) { ... }

融合目标: 把 %2 (matmul) 搬进循环内，只计算需要的 32×32 块
```

**操作链**: `%1` (add) → `%2` (matmul) → `%3` (consumer)

### 5.1 fuseProducerOfTensor - 公共API

**作用**：融合的主要入口

```cpp
 FailureOr<FusionInfo> mlir::linalg::fuseProducerOfTensor(
     OpBuilder &b, OpOperand &consumerOpOperand)
```

#### 代码解读

```cpp
// ========== 输入：Linalg Op 的入参 ==========
// 假设正在处理：%3 = linalg.generic ins(%2_tile) outs(%C_tile) { ... }
// 此时：consumerOpOperand = %3 的第 0 个入参 (即 %2_tile)

// 步骤 1: 查找生产者
// 此时：inputTensor = %2_tile
Value inputTensor = consumerOpOperand.get();
// 根据入参 %2_tile 获取其 Linalg Op 类型的 Producer，见第5.2节
getProducerOfTensor(inputTensor, producerOpResult);
if (!producerOpResult) return failure();

// 步骤 2: 验证操作类型
// inputTensor 的间接定义 Op 必须是 LinalgOp，本例中是：%2 的定义Op (linalg.matmul)
auto producerOp = dyn_cast<LinalgOp>(producerOpResult.getOwner());
auto consumerOp = dyn_cast<LinalgOp>(consumerOpOperand.getOwner());
if (!producerOp || !consumerOp) return failure();

// 步骤 3: inputTensor 的直接定义 Op 必须是 ExtractSliceOp
auto sliceOp = inputTensor.getDefiningOp<tensor::ExtractSliceOp>();
if (!sliceOp) return failure();

// 步骤 4: 检查是否已融合
if (consumerOpOperand.get().getParentBlock() ==
   producerOpResult.getParentBlock())
 return failure(); // 已经在同一基本块中，说明已经融合过了，直接返回

// 步骤 5: 执行融合
OpBuilder::InsertionGuard g(b);   // 保存插入位置
b.setInsertionPoint(consumerOp);  // 在 consumerOp 之前插入

// 获取生产者的输出操作数（dpsInit）
// producerOpResult.getResultNumber() = 0 （是结果的索引）
OpOperand *opOperand =
    producerOp.getDpsInitOperand(producerOpResult.getResultNumber());
// opOperand = %init2（matmul 的输出操作数）

// 执行融合，详见第5.3节
LinalgOp fusedProducer = fuse(b, producerOp,
                             producerOp.getMatchingIndexingMap(opOperand),
                             consumerOpOperand);
// 调用 fuse 函数，创建融合后的操作：
// %2_fused = linalg.matmul ins(%1_tile, %B_tile) outs(%output_tile)

// 步骤 6: 处理 Rank Reduction（如果需要）
Value def = fusedProducer->getResult(producerOpResult.getResultNumber());
// def = %2_fused（类型：tensor<32x32xf32>）
Type consumerType = consumerOpOperand.get().getType();
// consumerType = %2_tile 的类型（tensor<32x32xf32>）

if (cast<ShapedType>(consumerType).getRank() !=
    cast<ShapedType>(def.getType()).getRank()) {
  // 维度数量不匹配，说明发生了降维
  // 例如：tensor<32x32x1xf32> -> tensor<32x32xf32>
  // 本例：2 != 2 为 false，跳过
  llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
  def = tensor::dropGivenUnitDims(b, fusedProducer.getLoc(),
                                  def, droppedDims);
}

// 步骤 7: 类型转换（如果需要）
if (consumerType != def.getType())
  // 类型不匹配，插入转换操作
  // 本例：tensor<32x32xf32> == tensor<32x32xf32>，跳过
  def = b.create<tensor::CastOp>(fusedProducer.getLoc(),
                                 consumerType, def);

// 步骤 8: 替换使用
consumerOpOperand.set(def);
// 消费者操作的第 0 个输入从 %2_tile 替换为 %2_fused

return FusionInfo{producerOp, fusedProducer};
```

### 5.2 getProducerOfTensor - 查找生产者

**作用**：沿着 use-def 链向上追溯，找到张量的生产者。

```cpp
static void getProducerOfTensor(Value tensor, OpResult &opResult)
```

#### 代码解读

```cpp
while (true) {
  // 场景 1: 若 tensor 由 LinalgOp 定义，直接返回
  if (auto linalgOp = tensor.getDefiningOp<LinalgOp>()) {
    opResult = cast<OpResult>(tensor);
    // while 只循环 1 次
    return;
  }

  // 按照本节示例，首次调用本函数时：tensor = %2_tile
  // 场景 2: 若 tensor 通过 ExtractSliceOp 链接，继续追溯源
  if (auto sliceOp = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
    tensor = sliceOp.getSource();
    // 此时：tensor = %2，由 linalg.matmul 定义
    // 执行第二次 while 循环，会进入场景 1 分支 (linalg.matmul 是 LinalgOp)
    continue;
  }

  // 场景 3: 通过 scf.for 的迭代参数（单独示例）
  // %1 = linalg.generic ins(%A) outs(%init) { ... }
  // %2 = scf.for %i = 0 to 10 iter_args(%arg = %1) {
  //   %3 = linalg.generic ins(%arg) outs(%init2) { ... }
  //   scf.yield %3
  // }
  // getProducerOfTensor(%arg)
  if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
    // 第一次 while 循环：tensor = %arg，是 BlockArgument
    if (auto forOp = blockArg.getDefiningOp<scf::ForOp>()) {
      // %arg 由 scf.for 定义，获取循环的初始值：%1
      // blockArg.getArgNumber() = 0（%arg 是第 0 个迭代参数）
      // forOp.getInitArgs()[0] = %1
      tensor = forOp.getInitArgs()[blockArg.getArgNumber()];
      // 此时：tensor = %1，由 linalg.generic 定义
      // 执行第二次 while 循环，会进入场景 1 分支
      continue;
    }
  }

  return;  // 找不到（可能是函数参数）
}
```

### 5.3 fuse - 融合函数

**作用**：创建只计算指定范围的"克隆版本"。

```cpp
// 重载版本
static LinalgOp fuse(OpBuilder &b, LinalgOp producerOp, AffineMap producerMap,
                     OpOperand &consumerOpOperand)
// 主函数
static LinalgOp fuse(OpBuilder &b, LinalgOp producer,
                     const DenseMap<unsigned, Range> &fusedLoopsAndRanges)
```

#### 代码解读 - 重载版本（推断融合范围）

```cpp
// ========== 输入 ==========
// 生产者（输出维度到循环的映射）：
producerMap = affine_map<(i, j) -> (i, j)>
// 解释：matmul 的第 0 个输出维度对应循环 i，第 1 个对应循环 j

// 消费者使用的切片：
// %2_tile = tensor.extract_slice %2[32, 32][32, 32][1, 1]

// ========== 执行过程 ==========
DenseMap<unsigned, Range> fusedLoopsAndRanges;
Value shapedOperand = consumerOpOperand.get();  // shapedOperand = %2_tile

// 遍历生产者 IndexMap(affine_map右侧) 的每个结果
// producerMap.getResults() = [i, j]
for (const auto &en : llvm::enumerate(producerMap.getResults())) {
  // en.index() 是结果索引（0, 1, ...）
  // en.value() 是结果表达式（如 i, j）

  // 第一次循环：en.index() = 0, en.value() = i
  // 获取这个表达式对应的循环位置
  unsigned posInProducerLoop =
      cast<AffineDimExpr>(en.value()).getPosition();
  // i.getPosition() = 0
  // posInProducerLoop = 0

  // 从消费者切片中获取对应维度的范围
  // en.index() = 0，获取切片的第 0 维范围
  fusedLoopsAndRanges[posInProducerLoop] =
      getRangeFromOperandShape(  // 见第5.5节
          b, consumerOpOperand.getOwner()->getLoc(),
          shapedOperand,           // %2_tile
          en.index());             // 0
  // fusedLoopsAndRanges[0] = {offset: 32, size: 32, stride: 1}

  // 第二次循环：en.index() = 1, en.value() = j
  // j.getPosition() = 1
  // posInProducerLoop = 1
  // 获取切片的第 1 维范围
  // fusedLoopsAndRanges[1] = {offset: 32, size: 32, stride: 1}
}

// fusedLoopsAndRanges = {
//   0: {offset: 32, size: 32, stride: 1},
//   1: {offset: 32, size: 32, stride: 1}
// }

// 调用主函数执行融合
return fuse(b, producerOp, fusedLoopsAndRanges);
// 结果：创建只计算 [32:64][32:64] 的克隆版本
```

#### 代码解读 - 主函数（执行融合）

```cpp
// 先经过重载版本的fuse函数，产生如下输入：
// fusedLoopsAndRanges = {
//   0: {offset: 32, size: 32, stride: 1},
//   1: {offset: 32, size: 32, stride: 1}
// }

// 步骤 1: 准备循环信息
for (unsigned i = 0; i < producer.getNumLoops(); ++i) {
  // i = 0: 获取第 0 层循环由哪个操作数的哪个维度决定
  //       调用 getShapeDefiningLoopRange(producer, 0)，详见第5.4节
  //       返回 { shape: %A, dimension: 0 }
  auto shapeDim = getShapeDefiningLoopRange(producer, i);

  // 获取这个维度的实际大小
  // createFoldedDimOp(b, loc, %A, 0) 返回 %A 第 0 维的大小，即 128
  // 生成结果类似：%dim = tensor.dim %A, %c0
  OpFoldResult dim = createFoldedDimOp(b, loc, shapeDim.shape, shapeDim.dimension);
  sizeBounds.push_back(dim);  // 第一次循环：sizeBounds = [128]

  // 检查这个循环是否要被融合
  if (fusedLoopsAndRanges.contains(i)) {
    // 这是一个被融合的循环，使用融合范围
    ivs.push_back(it->second.offset);     // ivs = [32]
    tileSizes.push_back(it->second.size); // tileSizes = [32]
    loopRanges.push_back(it->second);     // loopRanges = [{32, 32, 1}]
  } else {
    // 未被融合的循环，使用完整范围
    tileSizes.push_back(b.getIndexAttr(0));  // 0 表示不融合
    loopRanges.push_back(Range{b.getIndexAttr(0), dim,
                                b.getIndexAttr(1)});
  }
  // i = 1: 同样处理
  //       getShapeDefiningLoopRange(producer, 1) 返回 { shape: %A, dimension: 1 }
  //       dim = 128
  //       sizeBounds = [128, 128]
  //       ivs = [32, 32]
  //       tileSizes = [32, 32]
  //       loopRanges = [{32, 32, 1}, {32, 32, 1}]
}
// ivs = [32, 32]
// tileSizes = [32, 32]
// sizeBounds = [128, 128]
// loopRanges = [{32, 32, 1}, {32, 32, 1}]

// 步骤 2: 为每个操作数创建切片
clonedShapes = makeTiledShapes(b, loc, producer, getTiledOperands(producer),
                               ivs, tileSizes, sizeBounds, ...);
// 对于 matmul (%2)，getTiledOperands 返回 [%1, %B, %init2]
// %1_tile = tensor.extract_slice %1[%ii, 0][32, 128][1, 1]
// %B_tile = tensor.extract_slice %B[0, %jj][128, 32][1, 1]
// %output_tile = tensor.extract_slice %init2[%ii, %jj][32, 32][1, 1]
// clonedShapes = [%1_tile, %B_tile, %output_tile]

// 步骤 3: 克隆操作，替换操作数
clonedOp = clone(b, producer, resultTypes, clonedShapes);
// %2_fused = linalg.matmul ins(%1_tile, %B_tile) outs(%output_tile) { ... }

// 步骤 4: 调整索引偏移（见下方）
offsetIndices(b, clonedOp, allIvs);

return clonedOp;
```

#### 为什么需要调整索引偏移？

```cpp
// 原始操作: i 范围 [0, 128)
%i = linalg.index 0
%cond = arith.cmpi sgt, %i, %c64  // i > 64

// 融合后: i 范围变成 [0, 32)（局部）
// 如果不调整，条件永远不满足

// offsetIndices 修复:
%i_local = linalg.index 0           // [0, 32)
%i = arith.addi %i_local, %c32      // [32, 64) ← 全局索引
%cond = arith.cmpi sgt, %i, %c64    // 正确
```

### 5.4 getShapeDefiningLoopRange - 获取循环范围来源

**作用**：找出哪个操作数的哪个维度决定循环范围。

```cpp
static ShapeDimension getShapeDefiningLoopRange(
    LinalgOp op,
    unsigned loopDepth,
    bool fromSubViewOpOnly = false)
```

**关键参数** `fromSubViewOpOnly`：

| 值      | 用途            | 获取范围      | 典型调用时机 |
| ------- | --------------- | ------------- | ------------ |
| `false` | Consumer Tiling | 完整 [0, 128) | Tiling 前    |
| `true`  | Fuse Producer   | 局部 [32, 64) | Tiling 后    |

**为什么需要区分？**

```cpp
// Tiling 前：需要完整迭代空间
getShapeDefiningLoopRange(consumerOp, 0, false)
// 返回 {shape: %T, dimension: 0} → 范围 [0, 128)

// Tiling 后：需要 tile 后的局部范围
%tile = tensor.extract_slice %T[32, 32][32, 32][1, 1]
getShapeDefiningLoopRange(consumerOp, 0, true)
// 返回 {shape: %tile, dimension: 0} → 范围 [32, 64)
```

### 5.5 getRangeFromOperandShape - 从切片提取范围

**作用**：从切片操作数中提取范围信息。

```cpp
static Range getRangeFromOperandShape(OpBuilder &b, Location loc,
                                      Value shapedOperand, unsigned dim)
```

**示例**：

```llvm
// 示例：获取 %2_tile 的范围
%2_tile = tensor.extract_slice %2[%ii, %jj][32, 32][1, 1]
// 从 %2[%ii:%ii+32][%jj:%jj+32] 提取

getRangeFromOperandShape(b, loc, %2_tile, 0)
// 返回: {offset: %ii, size: 32, stride: 1}

getRangeFromOperandShape(b, loc, %2_tile, 1)
// 返回: {offset: %jj, size: 32, stride: 1}
```

---

## 6. Greedy Fusion 完整调用逻辑

源码位置: `mlir/test/lib/Dialect/Linalg/TestLinalgFusionTransforms.cpp`

```cpp
struct TestLinalgGreedyFusion : public PassWrapper<...> {
  void runOnOperation() override {
    // 准备规范化模式
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);

    // 准备 Pass Pipeline
    OpPassManager pm(func::FuncOp::getOperationName());
    pm.addPass(createLoopInvariantCodeMotionPass());  // 循环不变量外提
    pm.addPass(createCanonicalizerPass());            // 规范化
    pm.addPass(createCSEPass());                      // 公共子表达式消除

    // 迭代融合直到无法继续
    do {
      applyPatternsGreedily(getOperation(), patterns);
      runPipeline(pm, getOperation());
    } while (succeeded(fuseLinalgOpsGreedily(getOperation())));
  }
};
```

---

## 7. 完整示例

### 7.1 原始代码

```llvm
func.func @example(%A: tensor<128x128xf32>,
                   %B: tensor<128x128xf32>,
                   %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // 生产者 1: A + 1.0
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>]
    ins(%A : tensor<128x128xf32>)
    outs(%init1 : tensor<128x128xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %tmp = arith.addf %arg0, %cst : f32
        linalg.yield %tmp : f32
  }

  // 生产者 2: (A+1.0) × B
  %2 = linalg.matmul ins(%1, %B) outs(%init2)

  // 消费者: result × 2.0
  %3 = linalg.generic ins(%2) outs(%C) {
    ^bb0(%arg0: f32, %arg1: f32):
      %res = arith.mulf %arg0, %cst2 : f32
      linalg.yield %res : f32
  }

  return %3 : tensor<128x128xf32>
}
```

### 7.2 步骤 1: Tile 消费者 (tile_size = 32)

```llvm
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    %2_tile = tensor.extract_slice %2[%ii, %jj][32, 32][1, 1]
    %C_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]

    %3_tile = linalg.generic ins(%2_tile) outs(%C_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}
```

### 7.3 步骤 2: 融合生产者 2 (matmul)

```llvm
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    // 融合的 matmul（只计算需要的 tile）
    %1_tile = tensor.extract_slice %1[%ii, 0][32, 128][1, 1]
    %B_tile = tensor.extract_slice %B[0, %jj][128, 32][1, 1]
    %output_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]

    %2_tile = linalg.matmul ins(%1_tile, %B_tile) outs(%output_tile)

    %3_tile = linalg.generic ins(%2_tile) outs(%output_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}
```

### 7.4 步骤 3: 融合生产者 1 (generic add)

```llvm
%3 = scf.for %ii = 0 to 128 step 32 iter_args(%arg4 = %C) {
  %result2 = scf.for %jj = 0 to 128 step 32 iter_args(%arg5 = %arg4) {
    // 融合的 add
    %A_tile = tensor.extract_slice %A[%ii, 0][32, 128][1, 1]
    %init_tile = tensor.empty() : tensor<32x128xf32>
    %1_tile = linalg.generic ins(%A_tile) outs(%init_tile) {
      ^bb0(%arg0: f32, %arg1: f32):
        %tmp = arith.addf %arg0, %cst : f32
        linalg.yield %tmp : f32
    }

    // 融合的 matmul
    %B_tile = tensor.extract_slice %B[0, %jj][128, 32][1, 1]
    %output_tile = tensor.extract_slice %arg5[%ii, %jj][32, 32][1, 1]
    %2_tile = linalg.matmul ins(%1_tile, %B_tile) outs(%output_tile)

    // 消费者
    %3_tile = linalg.generic ins(%2_tile) outs(%output_tile) { ... }

    %result3 = tensor.insert_slice %3_tile into %arg5[%ii, %jj][32, 32][1, 1]
    scf.yield %result3
  }
  scf.yield %result2
}
```

### 7.5 最终效果

```
原始: 3 个操作，每个处理 128×128
  └─> 内存访问: 3 次 128×128 的读写

融合后: 3 个操作融合到循环内，每次只处理 32×32
  └─> 内存访问: 32×32 块可放入 L1 缓存
  └─> 中间结果不需要写回内存
  └─> 整体性能提升数倍
```

---

## 8. Fusion 决策树

```
问题: 应该使用哪种 Fusion 策略?

1. 是否存在 producer–consumer 关系，且 consumer 支持 TilingInterface ?
   YES → 进入 Tile-and-Fuse 主路径，使用 fuseProducerOfTensor （Fusion.cpp）
   参数：--linalg-tile-and-fuse

2. 是否为纯逐元素算子链（elementwise + 无边界）?
   YES → 检查 areElementwiseOpsFusable
         如果可融合 → 使用 ElementwiseOpFusion（ElementwiseOpFusion.cpp）
       	 --linalg-elementwise-op-fusion

3. 是否涉及 Pad / 边界扩展?
   YES → 检查生产者是否为全并行 GenericOp
         如果是 → 使用 FusePadOpWithLinalgProducer（FusePadOpWithLinalgProducer.cpp）
         --linalg-fuse-pad-with-producer

4. consumer 是否被 tile ?
   YES → 自动形成 partial compute fusion
   NO  → 等价于 full fusion
```

* ElementwiseOpFusion.cpp：参见 [【MLIR】Linalg中ElementwiseOpFusion优化分析（总）](https://www.cnblogs.com/notlate-cn/articles/19358995)

* FusePadOpWithLinalgProducer.cpp：参见 [【MLIR】Linalg中FusePadOpWithLinalgProducer优化分析](https://www.cnblogs.com/notlate-cn/articles/19544761)
