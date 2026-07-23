---
title: "【MLIR】Tensor vs MemRef"
description: "1. 基本概念对比 | 特性 | Tensor | MemRef | | | | | | 语义 | 值语义（Value Semantics） | 引用语义（Reference Semantics） | | 可变性 | 不可变（Immutable） | 可变（Mutable） | | 内存模型 …"
slug: "mlirtensor-vs-memref"
legacyId: 19449240
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19449240"
pubDate: 2026-01-06
category: "AI 编译器"
tags: ["AI 编译器","MLIR","MemRef"]
featured: true
---

## 1. 基本概念对比

| 特性     | Tensor                    | MemRef                          |
| -------- | ------------------------- | ------------------------------- |
| 语义     | 值语义（Value Semantics） | 引用语义（Reference Semantics） |
| 可变性   | 不可变（Immutable）       | 可变（Mutable）                 |
| 内存模型 | 抽象的，不关心内存        | 具体的内存缓冲区                |
| 别名     | 不存在别名                | 可能存在别名                    |
| SSA      | 严格遵守SSA               | 可以原地修改                    |
| 优化     | 容易优化（无副作用）      | 需要别名分析                    |

## 2. 代码示例对比

```text
// ========== Tensor语义（值语义）==========
func.func @tensor_example(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  
  // 每个操作产生新的tensor值
  %1 = linalg.generic ins(%arg0) outs(%0) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
  } -> tensor<4xf32>
  
  // %arg0 和 %1 是不同的值
  // %arg0 没有被修改
  return %1 : tensor<4xf32>
}

// ========== MemRef语义（引用语义）==========
func.func @memref_example(%arg0: memref<4xf32>) {
  %alloc = memref.alloc() : memref<4xf32>

  // 直接修改内存
  linalg.generic ins(%arg0) outs(%alloc) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
  }

  // %alloc指向的内存被修改了
  // 如果其他地方也引用这块内存，会看到变化
  return
}
```

## 3. 别名问题详解

### 什么是别名（Aliasing）？

别名：两个或多个变量指向同一块内存区域。

```cpp
// C++中的别名示例
float* a = new float[4];
float* b = a;  // b和a是别名，指向同一块内存

a[0] = 1.0f;
// b[0] 也是 1.0f，因为a和b指向同一块内存
```

### MLIR MemRef中的别名示例

```text
func.func @memref_alias_example() {
  %base = memref.alloc() : memref<10xf32>

  // 创建两个子视图，可能重叠！
  %view1 = memref.subview %base[0][5][1] : memref<10xf32> to memref<5xf32>
  %view2 = memref.subview %base[2][5][1] : memref<10xf32> to memref<5xf32>
  //                              ^^^ 从index 2开始

  // %view1 和 %view2 有重叠部分！
  // %view1[2], %view1[3], %view1[4] 和 %view2[0], %view2[1], %view2[2]
  // 指向同一块内存

  // 危险：如果融合下面两个操作...
  linalg.generic ins(...) outs(%view1) { ... }  // 写入view1
  linalg.generic ins(%view2) outs(...) { ... }  // 读取view2
}
```

## 4. 为什么MemRef别名会破坏融合的正确性？

### 场景1：读写冲突

```text
func.func @alias_read_write_conflict(%base: memref<10xf32>) {
  %view1 = memref.subview %base[0][5][1] : memref<10xf32> to memref<5xf32>
  %view2 = memref.subview %base[3][5][1] : memref<10xf32> to memref<5xf32>
  
  // Producer: 写入 view1
  linalg.generic outs(%view1) {
    ^bb0(%out: f32):
      %c = arith.constant 42.0 : f32
      linalg.yield %c : f32
  }
  
  // Consumer: 读取 view2（可能与view1重叠！）
  %result = linalg.generic ins(%view2) outs(%other) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> memref<5xf32>
}
```

如果融合：

```text
// 错误的融合！
linalg.generic ins(%view2) outs(%view1, %other) {
  ^bb0(%in: f32, %out1: f32, %out2: f32):
    %c = arith.constant 42.0 : f32
    linalg.yield %c, %in : f32, f32
    //           ^^  ^^^
    //           |    └─ 期望读取原始的view2值
    //           └─ 但view2可能已经被%c覆盖了！
}
```

**问题**：

- 原始代码：先完整写入view1，再读取view2
- 融合后：边写入view1边读取view2
- 如果view1和view2重叠，读取的值可能已经被修改了！

### 场景2：输出别名问题

```text
func.func @output_alias(%arg0: memref<4xf32>) {
  // Producer: 修改 arg0
  linalg.generic ins(%input) outs(%arg0) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
  }
  
  // Consumer: 使用 arg0 作为输入，写入到 arg0（原地操作）
  linalg.generic ins(%arg0) outs(%arg0) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  }
}
```

**问题**：

- Producer的输出和Consumer的输入/输出可能是同一块内存
- 融合会破坏执行顺序
- 编译器难以判断是否安全

## 5. Tensor如何避免别名问题？

**核心原则**：值语义 + SSA

```text
func.func @tensor_no_alias(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %init1 = tensor.empty() : tensor<4xf32>

  // Producer: 产生新的tensor值 %1
  %1 = linalg.generic ins(%arg0) outs(%init1) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
  } -> tensor<4xf32>

  %init2 = tensor.empty() : tensor<4xf32>

  // Consumer: 使用 %1，产生新的tensor值 %2
  %2 = linalg.generic ins(%1) outs(%init2) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %in : f32
      linalg.yield %mul : f32
  } -> tensor<4xf32>

  return %2 : tensor<4xf32>
}
```

**保证**：

1. %arg0, %1, %2 是三个不同的SSA值
2. 每个tensor值不可变
3. 不存在别名：%1 绝对不会和 %arg0 或 %2 重叠
4. 可以安全融合

融合后：

```text
func.func @tensor_fused(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  
  %2 = linalg.generic ins(%arg0) outs(%init) {
    ^bb0(%in: f32, %out: f32):
      // Producer计算
      %add = arith.addf %in, %in : f32
      // Consumer计算
      %mul = arith.mulf %add, %add : f32
      linalg.yield %mul : f32
  } -> tensor<4xf32>
  
  return %2 : tensor<4xf32>
}
```

**正确性保证**：

- 因为tensor不可变，%arg0不会被修改
- 中间结果%add只存在于寄存器中
- 没有内存别名的可能性

## 6. 代码中的检查

在融合条件检查中：

```cpp
// ElementwiseOpFusion.cpp:153-155
if (!producer.hasPureTensorSemantics() ||
    !isa<RankedTensorType>(fusedOperand->get().getType()))
  return false;
```

**检查内容**：

1. hasPureTensorSemantics()：
   1. 所有operands都是tensor类型
   2. 没有memref
2. isa<RankedTensorType>(...)：
   1. 被融合的operand必须是tensor
   2. 必须有已知的rank（维度数量）

**实际判断逻辑**：

```cpp
// 在LinalgOp.cpp中
bool LinalgOp::hasPureTensorSemantics() {
  return llvm::all_of(getOperands(), [](Value operand) {
    return isa<RankedTensorType>(operand.getType());
  });
}
```

**含义**：只有当操作的所有输入和输出都是tensor时，才有纯tensor语义。

## 7. 对比总结

### MemRef融合的风险

```text
// ⚠️ 危险：无法保证安全
func.func @memref_fusion_risk(%a: memref<4xf32>, %b: memref<4xf32>) {
  // 问题1: %a 和 %b 可能指向同一块内存
  // 问题2: producer可能修改了consumer需要的数据
  // 问题3: 编译器需要复杂的别名分析

  linalg.generic ins(...) outs(%a) { ... }  // Producer
  linalg.generic ins(%a) outs(%b) { ... }   // Consumer

  // 如果 %a == %b，融合会导致错误结果！
}
```

### Tensor融合的安全性

```text
// ✅ 安全：值语义保证
func.func @tensor_fusion_safe(%a: tensor<4xf32>) -> tensor<4xf32> {
  // 保证1: %temp1 是新值，不会与 %a 别名
  // 保证2: %temp2 是新值，不会与 %temp1 或 %a 别名
  // 保证3: 无需别名分析

  %temp1 = linalg.generic ins(%a) outs(...) { ... } -> tensor<4xf32>
  %temp2 = linalg.generic ins(%temp1) outs(...) { ... } -> tensor<4xf32>

  return %temp2 : tensor<4xf32>
}
```

## 8. 实际应用场景

### 为什么MLIR同时支持Tensor和MemRef？

**Tensor（高层抽象）**：

- 用于优化阶段
- 方便做数学变换
- 适合融合、常量折叠等
- 代表"什么计算"

**MemRef（低层实现）**：

- 用于代码生成阶段
- 控制内存布局
- 适合lowering到LLVM IR
- 代表"如何存储"

**典型的编译流程**：

```text
Tensor IR (高层)
    ↓ 融合、优化
Tensor IR (优化后)
    ↓ Bufferization
MemRef IR (低层)
    ↓ 代码生成
LLVM IR
```

### Bufferization过程

**输入：Tensor IR**

```text
func.func @example(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = linalg.generic ins(%arg0) outs(%0) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
  } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
```

**输出：MemRef IR（经过Bufferization）**

```text
func.func @example(%arg0: memref<4xf32>) -> memref<4xf32> {
  %alloc = memref.alloc() : memref<4xf32>
  linalg.generic ins(%arg0) outs(%alloc) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %in : f32
      linalg.yield %add : f32
  }
  return %alloc : memref<4xf32>
}
```

## 9. 总结

1. Tensor 基于值语义、不可变特性，天然避免别名问题，是MLIR优化阶段（如算子融合）的首选；MemRef 基于引用语义、可变特性，存在别名风险，融合时需复杂的别名分析。
2. MLIR 中 `hasPureTensorSemantics()` 检查的核心是确保操作的所有输入输出均为 RankedTensorType，以此保证融合优化的安全性。
3. Tensor 用于高层抽象和优化，MemRef 用于底层内存布局和代码生成，两者通过 Bufferization 过程完成转换，构成MLIR完整的编译链路。
