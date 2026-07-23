---
title: "【MLIR】Transform 方言深入研究"
description: "本文档基于 Claude Code + GLM4.7&Sonnet4.6 (https://zhetengxia.com/) + CodeReaderSkills (https://zhetengxia.com/)完成。 方言完整地图 目录结构 文件规模概览 | 文件 | 行数 | 职责 | …"
slug: "mlirtransform-dialect-deep-dive"
legacyId: 19718284
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19718284"
pubDate: 2026-03-14
category: "AI 编译器"
tags: ["AI 编译器","MLIR"]
featured: true
---

> 本文档基于[Claude Code + GLM4.7&Sonnet4.6](https://zhetengxia.com/) + [CodeReaderSkills](https://zhetengxia.com/)完成。

---

## 方言完整地图

### 目录结构

```
mlir/
├── include/mlir/Dialect/Transform/
│   ├── IR/
│   │   ├── TransformDialect.h/td       # 方言注册、扩展机制
│   │   ├── TransformOps.h/td           # 核心操作定义（sequence/foreach等）
│   │   ├── TransformTypes.h/td         # Handle类型（AnyOp/Operation/Param等）
│   │   ├── TransformAttrs.h/td         # 属性（FailurePropagationMode等）
│   │   └── Utils.h                     # IR工具
│   ├── Interfaces/
│   │   ├── TransformInterfaces.h/td    # 核心接口（TransformOpInterface等）
│   │   └── MatchInterfaces.h/td        # 匹配接口
│   ├── Transforms/
│   │   ├── Passes.h/td                 # Pass声明
│   │   └── TransformInterpreterUtils.h # 解释器工具
│   ├── Utils/
│   │   ├── DiagnosedSilenceableFailure.h  # 三态错误类型
│   │   ├── RaggedArray.h               # 不规则二维数组
│   │   └── Utils.h
│   ├── DebugExtension/                 # 调试扩展（emit_remark等）
│   ├── LoopExtension/                  # 循环变换扩展
│   ├── PDLExtension/                   # PDL模式匹配扩展
│   ├── TuneExtension/                  # 可调参数扩展
│   └── IRDLExtension/                  # IRDL扩展
│
├── lib/Dialect/Transform/
│   ├── IR/                             # 核心IR实现
│   ├── Interfaces/                     # 接口实现
│   ├── Transforms/                     # Pass实现
│   ├── Utils/                          # 工具实现
│   └── [DebugExtension/.../TuneExtension/]  # 各扩展实现
│
└── test/Dialect/Transform/
    ├── ops.mlir                        # 基础操作测试
    ├── ops-invalid.mlir                # 48个非法用法案例
    ├── interpreter.mlir                # 解释器执行测试
    ├── foreach-match.mlir              # 模式匹配迭代测试
    ├── selective-targeting.mlir        # 选择性目标测试
    ├── check-use-after-free.mlir       # Handle生命周期安全测试
    ├── test-interpreter.mlir           # 完整解释器测试（70KB+）
    ├── apply-foreach-nested.mlir       # 嵌套foreach测试
    ├── infer-effects.mlir              # 自动推断副作用测试
    └── include/                        # 外部库测试素材
```

### 文件规模概览

| 文件                                        | 行数 | 职责                                      |
| ------------------------------------------- | ---- | ----------------------------------------- |
| `lib/Interfaces/TransformInterfaces.cpp`    | 2045 | 执行引擎核心（TransformState、apply流程） |
| `lib/IR/TransformOps.cpp`                   | 3138 | 所有内置Transform操作实现                 |
| `include/Interfaces/TransformInterfaces.h`  | 1624 | TransformState、接口声明                  |
| `include/IR/TransformOps.td`                | 1398 | 操作定义（TableGen）                      |
| `include/Interfaces/TransformInterfaces.td` | 413  | 接口定义（TableGen）                      |

## 0. 快速入门

### 0.1 Hello World：最简单的 Transform 示例

**目标：** 使用 Transform方言打印一个操作的名称。

```text
// ============================================================
// Payload IR - 被转换的目标 IR
// ============================================================
module {
  func.func @hello_world() {
    %0 = arith.constant 42 : i32
    return
  }
}

// ============================================================
// Transform IR - 控制转换逻辑的 IR
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 找到所有函数
  %funcs = transform.loop.match "func.func" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印找到的函数
  transform.print %funcs { name = "Found functions" }

  transform.yield
}
```

**预期输出：**

```
// Found functions
func.func @hello_world
```

### 0.2 从头到尾的执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Transform Hello World 执行流程                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  步骤 1: 解析阶段                                                        │
│    ├── 加载 Transform 方言                                               │
│    ├── 解析 Transform IR                                                 │
│    └── 验证类型正确性                                                    │
│                                                                         │
│  步骤 2: 状态初始化                                                      │
│    ├── 创建 TransformState                                              │
│    └── 映射 %root → 顶层模块操作                                         │
│                                                                         │
│  步骤 3: 执行 transform.loop.match                                       │
│    ├── 从 %root 获取 Payload 操作                                        │
│    ├── 遍历查找所有 func.func 操作                                       │
│    └── 创建 %funcs Handle，指向 [func.func@hello_world]                  │
│                                                                         │
│  步骤 4: 执行 transform.print                                           │
│    ├── 从 %funcs 获取 Payload 操作                                       │
│    ├── 打印操作到 stdout                                                 │
│    └── 输出: "Found functions: func.func @hello_world"                   │
│                                                                         │
│  步骤 5: transform.yield 结束                                            │
│    └── 返回成功状态                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 0.3 完整可运行的示例

```text
// ============================================================
// 完整示例：匹配并打印所有算术操作
// ============================================================
module {
  func.func @example(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %c0 = arith.constant 0.0 : f32
    %1 = arith.addf %arg0, %arg0 : tensor<10xf32>
    %2 = arith.mulf %1, %1 : tensor<10xf32>
    return %2 : tensor<10xf32>
  }
}

// ============================================================
// Transform 序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 匹配所有 arith.addf 操作
  %add_ops = transform.loop.match "arith.addf" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印匹配结果
  transform.print %add_ops { name = "Add operations" }

  // 匹配所有 arith.mulf 操作
  %mul_ops = transform.loop.match "arith.mulf" in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印匹配结果
  transform.print %mul_ops { name = "Mul operations" }

  transform.yield
}
```

### 0.4 关键概念速览

| 概念               | 说明                                   | 示例                                         |
| ------------------ | -------------------------------------- | -------------------------------------------- |
| **Payload IR**     | 被转换的目标 IR                        | `func.func`, `arith.addf`, `scf.for`         |
| **Transform IR**   | 控制转换逻辑的 IR                      | `transform.sequence`, `transform.loop.match` |
| **Handle**         | Transform IR 中指向 Payload IR 的引用  | `%funcs : !transform.any_op`                 |
| **TransformState** | 维护 Handle ↔ Payload 映射的运行时状态 | 内部对象，用户不可见                         |

---

## 1. 快速概览

### 1.1 基本信息

**Transform 方言**是 MLIR 中用于**精细控制 IR 转换**的方言。

| 属性             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| **方言名称**     | `transform`                                                  |
| **C++ 命名空间** | `::mlir::transform`                                          |
| **核心文件**     | `mlir/lib/Dialect/Transform/IR/TransformDialect.cpp`         |
| **ODS 定义**     | `mlir/include/mlir/Dialect/Transform/IR/TransformOps.td`     |
| **接口定义**     | `mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.td` |

### 1.2 核心设计理念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Transform 方言架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐         ┌──────────────────┐                     │
│   │  Transform IR    │         │   Payload IR     │                     │
│   │  (控制转换逻辑)    │──────▶  │   (被转换的IR)    │                      │
│   └──────────────────┘         └──────────────────┘                     │
│          │                                                              │
│          │ 通过 Handle 关联                                               │
│          ▼                                                              │
│   ┌──────────────────┐                                                  │
│   │  Handle 类型系统  │                                                  │
│   │ • OperationHandle│                                                  │
│   │ • ValueHandle    │                                                  │
│   │ • ParamHandle    │                                                  │
│   └──────────────────┘                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 背景与动机

### 2.1 问题本质

**要解决的问题：**传统的MLIR Pass无法精细控制"对哪些操作应用哪种变换"。官方文档对此的描述是：

> The main use case is orchestrating fine-grain transformations on individual IR objects — finding loop-like operations with specific properties (e.g., large size) and applying loop tiling to **those and only those operations**.

**考虑以下编译优化场景：**

```text
// 假设有以下循环嵌套需要优化
scf.for %i = 0 to 1024 {
  scf.for %j = 0 to 1024 {
    scf.for %k = 0 to 1024 {
      // 一些计算
    }
  }
}
```

传统Pass是"全局扫描、统一应用"的模型。例如 `--linalg-tile` 会对所有 `linalg.*` 操作进行平铺，参数全局统一。这造成：

1. **无法区分**：Pass 对所有 循环Op 应用相同转换
2. **无法组合**：想要"先切分最内层循环，再切分次内层"需要编写新 Pass
3. **无法回退**：某种转换失败时，无法尝试备选方案
4. **调试困难**：哪些操作被变换了，哪些没有，完全不透明

### 2.2 方案选择

**WHY 选择嵌入式DSL（Transform IR）而不是Python脚本？**

| 方案                   | 优势                                   | 劣势                                   |
| ---------------------- | -------------------------------------- | -------------------------------------- |
| Transform IR（现方案） | 类型安全、可验证、与MLIR编译器同步运行 | 需要学习新语法                         |
| Python脚本             | 灵活、生态丰富                         | 与IR的类型系统脱节，运行时错误难以定位 |
| 配置文件（如YAML）     | 简单                                   | 表达能力有限，无法动态决策             |

Transform方言的设计选择是：**让Transform本身也是MLIR IR**，这带来了最大的好处：

- Transform IR可以被分析、验证、优化
- Transform IR可以被其他程序生成
- Transform IR可以作为库函数被重用（named_sequence）

**WHY 不用JIT/动态执行？**

Transform IR是**声明式的、可静态分析的**。这允许在执行前验证handle消费约束、副作用声明等，而不是等到运行时才报错。

### 2.3 Transform 方言的解决方案

```text
// 使用 Transform 方言精细控制转换
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // 1. 找到所有循环
  %loops = transform.loop.structure %arg0 : (!transform.any_op) -> !transform.any_op

  // 2. 只对最内层循环应用切分
  %innermost = transform.loop.get_innermost %loops
  transform.loop.unroll %innermost { factor = 4 }

  // 3. 对次内层循环应用不同转换
  %middle = transform.loop.get_middle %loops
  transform.loop.tile %middle { tile_sizes = [8, 8] }
}
```

### 2.4 应用场景

**适用场景：**

- 编译pipeline中需要精细控制的阶段（ML编译器、HPC优化）
- 算子库开发（需要对特定尺寸的矩阵乘法使用特定变换策略）
- 自动调优（通过TuneExtension探索变换参数空间）

**不适用场景：**

- 全局统一的简单变换（用普通Pass更简单）
- 不需要跨操作协调的变换（单独的Pattern更合适）

---

## 3. 核心概念

### 3.1 概念清单

**概念 1：Payload IR vs Transform IR**

- **是什么：** 两个独立的IR世界。Payload IR是"被变换的代码"（用户的程序），Transform IR是"描述如何变换的规则"（变换脚本）。
- **WHY 需要：** 将"做什么"与"怎么做"分离。没有这种分离，变换逻辑会硬编码到每个Pass中，无法重用和组合。
- **WHY 这样实现：** Transform IR与Payload IR共存于同一个MLIRContext但在不同的Operation树中，由TransformState维护它们之间的映射关系。

**概念 2：Handle（句柄）**

- **是什么：** Transform IR中的一个值（`%handle : !transform.any_op`），指向Payload IR中的一组操作或值。类比于数据库中的游标，或C++中指向容器元素的迭代器。
- **WHY 需要：** Transform op需要一种方式指定"作用在哪些Payload操作上"，handle就是这个"指针"。
- **WHY 不用操作名字符串：** 字符串无类型安全，一个Payload程序中可能有多个同名操作，无法区分。Handle指向特定的操作实例。

**概念 3：TransformState**

- **是什么：** 执行Transform IR时的运行时状态，维护所有handle到Payload操作的映射。是整个执行引擎的中枢。
- **WHY 需要：** Transform操作之间需要共享映射信息——前一个操作产生的handle，后一个操作才能使用。TransformState就是这个共享的黑板（blackboard）。
- **WHY 双向映射：** 正向（handle→ops）用于查询；反向（op→handles）用于失效传播（O(1)而非O(n)）。

**概念 4：DiagnosedSilenceableFailure**

- **是什么：** 三态错误类型（成功/可沉默失败/确定失败），是所有Transform操作的返回值类型。
- **WHY 需要三态：** 编译优化中"优化无法应用"不等于"程序有错误"。Silenceable failure让容器操作可以选择忽略并继续，而definite failure则表示不可恢复的错误。
- **WHY 不用异常：** MLIR全面使用基于返回值的错误处理，避免异常带来的控制流复杂性。

**概念 5：内存效应（Memory Effects）**

- **是什么：** Transform操作对handle和payload的副作用声明（Allocate/Free/Read/Write）。
- **WHY 需要：** 系统需要知道某个操作是否"消费"（销毁/修改）了handle，以便跟踪handle的有效性。没有效应声明，就无法检测use-after-free。
- **WHY 不用自动推断：** 只有操作的实现者才知道它的副作用，自动推断会导致保守的过度标注（所有操作都视为消费）。

#### 概念关系矩阵

| 关系类型  | 概念 A                      | 概念 B     | WHY 这样关联                                               |
| --------- | --------------------------- | ---------- | ---------------------------------------------------------- |
| 包含/管理 | TransformState              | Handle映射 | TransformState是所有映射的唯一权威                         |
| 产生/消费 | Transform操作               | Handle     | 操作通过TransformResults产生handle，通过operands消费handle |
| 约束/检查 | 内存效应                    | Handle消费 | 效应声明决定哪些操作会失效handle                           |
| 包装/传播 | DiagnosedSilenceableFailure | 错误诊断   | 包装Diagnostic对象，强制显式处理                           |
| 作用于    | Transform IR                | Payload IR | Transform操作修改Payload，但两者物理分离                   |

### 3.2 TransformState 类

**TransformState是整个Transform方言的"执行上下文"**，维护Transform IR值与Payload IR实体之间的多对多映射。

#### 内部数据结构

```cpp
// 每个Region有独立的Mappings，形成作用域栈
struct Mappings {
  TransformOpMapping direct;     // Value -> [Operation*]  正向映射
  TransformOpReverseMapping reverse; // Operation* -> [Value]  反向映射
  ParamMapping params;           // Value -> [Attribute]  参数映射
  ValueMapping values;           // Value -> [Value]       值映射
  ValueMapping reverseValues;    // Value -> [Value]       反向值映射
};

// 每个Region有独立的映射（隔离作用域）
DenseMap<Region*, std::unique_ptr<Mappings>> mappings;
std::vector<RegionScope*> regionStack;

// 顶级额外映射（传给顶级op的参数）
RaggedArray<MappedValue> topLevelMappedValues;
```

**为什么双向映射？**

正向映射（handle → ops）：Transform op问"我的operand对应哪些payload ops"。
反向映射（op → handles）：当一个op被删除时，需要快速找出所有指向它的handles并使其失效。如果只有正向映射，必须遍历所有handles（O(n)）；有了反向映射，只需查表（O(1)）。

**为什么每个Region有独立的Mappings？**

Transform IR中的Region（如`transform.foreach`的循环体）定义了新的块参数，这些块参数的映射生命周期应局限于该Region。当Region处理完毕，RegionScope析构函数自动清理所有映射，防止内存泄漏和映射污染。

#### 关键方法

```cpp
// 查询handle对应的payload操作（自动跳过nullptr/已删除操作）
auto getPayloadOps(Value handle) const {
  return llvm::make_filter_range(view, [](Operation *op) {
    return op != nullptr;  // 延迟紧凑化：不立即删除，只跳过
  });
}

// 为块参数关联payload实体（进入Region时调用）
LogicalResult mapBlockArgument(BlockArgument arg,
                               ArrayRef<MappedValue> values);
```

**延迟紧凑化的权衡**：用内存换O(1)删除速度。被删除操作留nullptr占位，迭代时自动跳过，避免频繁重新分配。

### 3.3 Transform 类型接口详解

#### 3.3.1 TransformTypeInterfaceBase - 基础类型接口

```cpp
// TransformInterfaces.td
template <typename DerivedTy, typename PayloadTy>
class TransformTypeInterfaceBase : public TypeInterface<DerivedTy> {
public:
  virtual DiagnosedSilenceableFailure checkPayload(
      Location loc,
      ArrayRef<PayloadTy> payload) = 0;
};
```

#### 3.3.2 TransformHandleTypeInterface - 操作句柄接口

```cpp
// TransformInterfaces.td
def TransformHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformHandleTypeInterface",
                                 "::mlir::Operation *"> {
  let description = [{
    Types that can be used for the Transform dialect operation handle values.
  }];
}
```

**实现示例：**

```cpp
DiagnosedSilenceableFailure OperationType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) {
  for (Operation *op : payload) {
    if (op->getName().getStringRef() != getOperationName()) {
      return emitSilenceableError(loc)
             << "expected '" << getOperationName() << "' operation, "
             << "but found '" << op->getName() << "'";
    }
  }
  return DiagnosedSilenceableFailure::success();
}
```

#### 3.3.3 TransformValueHandleTypeInterface - 值句柄接口

```cpp
// TransformInterfaces.td
def TransformValueHandleTypeInterface
    : TransformTypeInterfaceBase<"TransformValueHandleTypeInterface",
                                 "::mlir::Value"> {
  let description = [{
    Types that can be used for the Transform dialect handle values pointing to
    Payload IR values.
  }];
}
```

#### 3.3.4 TransformParamTypeInterface - 参数句柄接口

```cpp
// TransformInterfaces.td
def TransformParamTypeInterface
    : TransformTypeInterfaceBase<"TransformParamTypeInterface",
                                 "::mlir::Attribute"> {
  let description = [{
    Types that can be used for the Transform dialect parameter values.
  }];
}
```

### 3.4 TransformOpInterface - 操作接口

**概念：** 所有 Transform 操作必须实现的核心接口。

#### 3.4.1 核心方法：apply

```cpp
virtual DiagnosedSilenceableFailure apply(
    TransformRewriter &rewriter,
    TransformResults &results,
    TransformState &state) = 0;
```

**参数说明：**

| 参数       | 职责                  | WHY 这样设计                              |
| ---------- | --------------------- | ----------------------------------------- |
| `rewriter` | 所有IR修改必须通过它  | 支持撤销、冲突检测、通知机制              |
| `results`  | 报告新handle的映射    | 与state分离，支持原子性：失败时不提交结果 |
| `state`    | 查询输入的payload映射 | 只读访问，防止apply()直接修改全局状态     |

**实现示例：**

```cpp
DiagnosedSilenceableFailure MyTransformOp::apply(
    TransformRewriter &rewriter,
    TransformResults &results,
    TransformState &state) {

  // 步骤 1: 获取目标操作
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  if (targets.empty()) {
    return emitSilenceableError() << "no operations to transform";
  }

  // 步骤 2: 对每个目标应用转换
  SmallVector<Operation *> transformedOps;
  for (Operation *target : targets) {
    FailureOr<Operation *> result = applyMyTransform(rewriter, target);
    if (failed(result)) {
      return emitDefaultSilenceableFailure(target);
    }
    transformedOps.push_back(*result);
  }

  // 步骤 3: 设置结果
  results.set(cast<OpResult>(getResult()), transformedOps);

  return DiagnosedSilenceableFailure::success();
}
```

### 3.5 RaggedArray：不规则二维数组

用于`topLevelMappedValues`的存储，每行长度可以不同：

```cpp
class RaggedArray<T> {
  SmallVector<std::pair<size_t, size_t>> slices;  // (offset, length) 偏移量表
  SmallVector<T> storage;                          // 所有元素的连续存储
};
```

**为什么不用`Vec<Vec<T>>`？**

单一连续存储 + 偏移量表比多次动态分配：

- 减少内存碎片和分配开销
- 缓存友好，CPU预取效率更高
- 支持`replace(pos, elements)`就地修改，自动更新后续偏移量

### 3.6 DiagnosedSilenceableFailure：三态错误处理

#### 三种状态

```
成功（success）
  ↓ 所有变换正常完成
可沉默失败（silenceable failure）
  ↓ 变换无法应用，但不是IR错误（如优化条件不满足）
  ↓ 容器操作可以选择忽略并继续
确定失败（definite failure）
  ↓ 不可恢复的错误（如IR非法、类型不匹配）
  ↓ 必须立即中止
```

#### 内部设计

```cpp
class DiagnosedSilenceableFailure {
  SmallVector<Diagnostic, 1> diagnostics;  // 诊断消息（非空 = silenceable failure）
  LogicalResult result;                     // success/failure（只在diagnostics为空时有意义）
};
```

**状态判断逻辑：**

- `succeeded()` = `result == success && diagnostics.empty()`
- `isDefiniteFailure()` = `result == failure && diagnostics.empty()`
- `isSilenceableFailure()` = `!diagnostics.empty()`

**WHY 使用`[[nodiscard]]`：** 强制调用者显式处理返回值，防止silenceable failure被悄悄丢弃（这是常见的安全漏洞）。

**两种"处理"方式：**

```cpp
// 方式1：升级为错误并报告
LogicalResult result = failure.checkAndReport();

// 方式2：忽略（消除）silenceable failure
(void)failure.silence();
```

---

## 4. 类型系统

### 4.1 类型层次结构

```
Transform 类型系统
├── TransformHandleTypeInterface (操作句柄)
│   ├── AnyOpType (!transform.any_op)
│   └── OperationType<!transform.op<"op_name">>
├── TransformValueHandleTypeInterface (值句柄)
│   └── AnyValueType (!transform.any_value)
└── TransformParamTypeInterface (参数句柄)
    ├── AnyParamType (!transform.any_param)
    ├── ParamType<!transform.param<Type>>
    ├── AffineMapParamType (!transform.affine_map)
    └── TypeParamType (!transform.type)
```

### 4.2 操作句柄类型 (Operation Handle Types)

#### 4.2.1 AnyOpType

**语法：** `!transform.any_op`

**定义：**

```text
def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let mnemonic = "any_op";
}
```

**WHY 设计 AnyOpType：**

- **灵活性**：可以指向任何操作
- **类型安全最小化**：不进行类型约束验证
- **适用场景**：操作类型未知或多样化

#### 4.2.2 OperationType

**语法：** `!transform.op<"operation_name">`

**定义：**

```text
def Transform_OperationType : TypeDef<Transform_Dialect, "Operation",
    [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let mnemonic = "op";
  let parameters = (ins
    StringRefParameter<"Name of the allowed payload operation">:$operation_name
  );
}
```

**WHY 设计 OperationType：**

- **类型安全**：编译时和运行时都验证操作类型
- **操作特化**：某些转换只适用于特定操作
- **文档作用**：清晰表达句柄的预期内容

### 4.3 值句柄类型 (Value Handle Types)

#### 4.3.1 AnyValueType

**语法：** `!transform.any_value`

**定义：**

```text
def Transform_AnyValue : TypeDef<Transform_Dialect, "AnyValue",
    [DeclareTypeInterfaceMethods<TransformValueHandleTypeInterface>]> {
  let mnemonic = "any_value";
}
```

**WHY 需要值句柄：**

| 特性     | 操作句柄            | 值句柄                 |
| -------- | ------------------- | ---------------------- |
| 指向对象 | Operation           | SSA Value              |
| 用途     | 操作转换            | 值追踪/转换            |
| 示例     | `!transform.any_op` | `!transform.any_value` |

### 4.4 参数句柄类型 (Parameter Handle Types)

#### 4.4.1 ParamType

**语法：** `!transform.param<Type>`

**定义：**

```text
def Transform_ParamType : TypeDef<Transform_Dialect, "Param",
    [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let mnemonic = "param";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Underlying type of the parameter">:$type
  );
}
```

**WHY 需要参数句柄接口：**

| 需求           | 说明                            | 示例                           |
| -------------- | ------------------------------- | ------------------------------ |
| **运行时参数** | 支持在 Transform 执行时传递参数 | 切分大小、向量宽度等           |
| **类型安全**   | 确保参数类型正确                | 防止将字符串传给期望整数的转换 |
| **灵活性**     | 允许动态配置转换行为            | 根据运行时信息选择参数         |

### 4.5 类型验证机制

每个 Transform 类型都实现了对应的接口，提供 `checkPayload` 方法进行运行时验证。

```cpp
// 接口定义
virtual DiagnosedSilenceableFailure checkPayload(
    Location loc,
    ArrayRef<Operation *> payload) = 0;
```

**WHY 需要运行时验证：**

| 验证时机 | 验证内容                 | 原因                      |
| -------- | ------------------------ | ------------------------- |
| 解析时   | 类型语法正确性           | MLIR 类型系统保证         |
| 执行时   | Payload 对象符合类型约束 | 运行时才知道 Payload 对象 |

### 4.6 类型系统设计决策

#### 4.6.1 WHY 使用参数化类型？

- **类型安全**：编译时就知道句柄指向的操作类型
- **优化机会**：编译器可以基于类型信息优化
- **文档作用**：类型本身就表达了约束

#### 4.6.2 WHY 分离三种 Handle 类型？

```
操作句柄 → 指向 Operation（可执行单元）
  ↓
值句柄 → 指向 Value（数据流）
  ↓
参数句柄 → 指向 Attribute（编译时常量）
```

**WHY 这样分离：**

1. **语义清晰**：操作转换、数据追踪、配置参数各司其职
2. **类型安全**：混用会导致类型混乱
3. **验证分离**：每种类型有不同的验证规则

---

## 5. 核心操作详解

### 5.1 transform.sequence / transform.named_sequence - 转换序列

#### 5.1.1 transform.sequence（旧版本，不建议使用）

**sequence** 是基础的顺序执行容器，**按顺序执行**一组变换操作，`failure_propagation_mode`控制错误处理：

```text
// Propagate模式（严格）：任何失败立即中止
transform.sequence %root : !transform.any_op failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ... in %arg0 : ...
  transform.structured.tile_using_for %0 tile_sizes [4, 4] : ...
}

// Suppress模式（容错）：忽略silenceable failure继续执行
transform.sequence %root : !transform.any_op failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // 如果这里失败（silenceable），继续执行下一个操作
  transform.optional_optimization %arg0 : !transform.any_op
}
```

**WHY两种模式？** 用于不同场景：

- Propagate：关键变换，任何失败都不可接受
- Suppress：试探性优化，部分失败可以接受

**SequenceOp::apply 执行流程：**

```
1. 创建RegionScope（隔离block argument命名空间）
2. 映射block arguments到payload
3. 顺序执行body中的每个Transform op：
   a. definite failure → 立即返回
   b. silenceable failure + propagate → 转发失败，停止执行
   c. silenceable failure + suppress → 忽略，继续
4. 转发yield操作数（只在成功时）
```

#### 5.1.2 transform.named_sequence（新版本，建议使用）

**named_sequence** 是可复用的Transform库函数，相当于**定义一个函数**，可以被多次调用：

```text
module attributes { transform.with_named_sequence } {
  // 定义库函数
  transform.named_sequence @tile_and_vectorize(
      %op: !transform.any_op {transform.consumed}) {
    %tiled, %loops = transform.structured.tile_using_for %op tile_sizes [4] : ...
    transform.structured.vectorize %tiled : !transform.any_op
    transform.yield
  }

  // 主入口点
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : ...
    // 调用库函数
    transform.include @tile_and_vectorize failures(propagate) (%matmul) : ...
    transform.yield
  }
}
```

**WHY 禁止递归？**

1. Transform的handle消费验证依赖静态分析，递归会导致无穷展开
2. 解释器不需要处理栈溢出或循环检测，设计更简单
3. 禁止递归不影响大多数实际使用场景（变换通常是有界的）

### 5.2 transform.foreach / transform.foreach_match - 遍历

#### 5.2.1 transform.foreach

Handle（句柄）可能指向**多个 op**，`foreach` 对每一个**单独处理**：

```text
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 匹配所有 linalg.generic（可能有多个）
  %generics = transform.structured.match ops{["linalg.generic"]} in %root
    : (!transform.any_op) -> !transform.any_op

  // 对每一个 generic 单独 tile
  transform.foreach %generics : !transform.any_op {
  ^bb0(%single_generic: !transform.any_op):
    %tiled, %loops = transform.structured.tile_using_for %single_generic [2, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  }
}
```

**为什么需要 foreach？** 

```
%generics 可能指向：
  [linalg.generic #1,
   linalg.generic #2,
   linalg.generic #3]
--> 不用 foreach → 整体处理，tile 参数对所有 op 统一生效
--> 用 foreach   → 每个 op 独立进入 body，可以单独决策
```

**ForeachOp::apply 的关键设计：**

```cpp
// 关键1：提前快照所有payloads，防止迭代中映射被修改
SmallVector<SmallVector<MappedValue>> payloads;
detail::prepareValueMappings(payloads, getTargets(), state);

for (size_t i = 0; i < numIterations; i++) {
  // 关键2：每次迭代创建独立scope，防止跨迭代污染
  auto scope = state.make_region_scope(getBody());

  // 关键3：每次迭代只映射单个元素
  state.mapBlockArgument(blockArg, {payloads[argIdx][i]});

  // 执行body...

  // 关键4：累积yield结果（append，不是覆盖）
  llvm::append_range(resTuple, state.getPayloadOps(yieldOperand));
}
```

#### 5.2.2 transform.foreach_match

**foreach_match** 是模式驱动的迭代，用于"匹配特定类型op，然后对每个匹配应用变换"：

```text
// 匹配器：检查是否是scf.for
transform.named_sequence @match_for(
    %arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
  transform.yield %arg0 : !transform.any_op
}

// 动作：对scf.for应用循环分割
transform.named_sequence @peel(
    %arg0: !transform.op<"scf.for"> {transform.consumed}) {
  transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> ...
  transform.yield
}

// 主变换
transform.foreach_match in %root
    @match_for -> @peel
    : (!transform.any_op) -> !transform.any_op
```

**foreach_match 的执行语义：**

```
for each op in root (post-order walk):
  for each (matcher, action) pair:
    if matcher(op) succeeds:
      apply action(matcher_results)
      break  // 跳过其他matcher
```

#### 综合对比上述4个Op

```
假如有一堆 op 需要变换：

sequence        → 写死步骤，全部统一处理
                  [match → tile → vectorize → ...]

named_sequence  → 把上面的步骤封装成函数，哪里需要哪里调用
                  @my_pipeline(%op) { ... }

foreach         → 有多个 op，想对每个单独跑同一套逻辑
                  for each op in handles: { tile(op) }

foreach_match   → 有多个 op，不同类型走不同逻辑
                  for each op in root:
                    if matmul → @handle_matmul
                    if conv   → @handle_conv
```

### 5.3 transform.match.ops / transform.structured.match - 操作匹配

**语法：**

```text
// 核心方言
%results = transform.match ops {["op_name1", "op_name2"]} in %target
    : (!transform.any_op) -> !transform.any_op

// Linalg 方言
%results = transform.structured.match ops {["linalg.matmul"]} in %target
    : (!transform.any_op) -> !transform.any_op
```

**参数：**

| 参数                 | 类型       | 说明             |
| -------------------- | ---------- | ---------------- |
| `target`             | Handle     | 在其中搜索的操作 |
| `ops`                | 字符串数组 | 要匹配的操作名称 |
| `interface`          | 可选属性   | 匹配特定接口     |
| `attributes`         | 可选属性   | 匹配特定属性     |
| `filter_result_type` | 可选属性   | 过滤结果类型     |

**返回值：** 指向所有匹配操作的 Handle

**WHY 需要匹配：**

- 选择要转换的目标操作
- 支持复杂查询条件
- 类型安全的操作选择

**使用示例：**

```text
// 匹配所有 matmul 操作
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 匹配多种操作
%ops = transform.match ops {["scf.for", "affine.for"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 使用接口匹配
%loops = transform.structured.match interface {LoopLikeInterface} in %root
    : (!transform.any_op) -> !transform.any_op
```

### 5.4 transform.structured.tile - 循环分块

**语法：**

```text
%tiled, %loops = transform.structured.tile %target [tile_sizes]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

**参数：**

| 参数          | 类型     | 说明               |
| ------------- | -------- | ------------------ |
| `target`      | Handle   | 要分块的操作       |
| `tile_sizes`  | 整数数组 | 每个维度的分块大小 |
| `interchange` | 可选数组 | 维度置换           |

**返回值：**

| 返回值  | 说明         |
| ------- | ------------ |
| `tiled` | 分块后的操作 |
| `loops` | 新生成的循环 |

**WHY 需要分块：**

- 提高缓存局部性
- 启用向量化
- 减少内存访问延迟

**使用示例：**

```text
// 矩阵乘法分块
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

// 分块：M=64, N=32, K=16
%tiled, %loops = transform.structured.tile %matmuls [64, 32, 16]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### 5.5 transform.structured.vectorize - 向量化

**语法：**

```text
// 自动推断向量大小
transform.structured.vectorize %target : !transform.any_op

// 指定向量大小
transform.structured.vectorize %target vector_sizes [4, 8]
    : (!transform.any_op) -> !transform.any_op
```

**参数：**

| 参数             | 类型     | 说明           |
| ---------------- | -------- | -------------- |
| `target`         | Handle   | 要向量化的操作 |
| `vector_sizes`   | 可选数组 | 向量形状       |
| `scalable_sizes` | 可选数组 | 可扩展向量标志 |

**返回值：** 无（消费目标 Handle）

**WHY 需要向量化：**

- 利用 SIMD 指令
- 提高并行度
- 减少指令数量

**使用示例：**

```text
// 先分块再向量化
%matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
    : (!transform.any_op) -> !transform.any_op

%tiled, %loops = transform.structured.tile %matmuls [16, 16]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// 向量化最内层循环
transform.structured.vectorize %tiled vector_sizes [4, 8]
    : (!transform.any_op) -> !transform.any_op
```

### 5.6 transform.print - 调试输出

**语法：**

```text
transform.print %target { name = "Debug output" }
    : !transform.any_op
```

**参数：**

| 参数              | 类型        | 说明           |
| ----------------- | ----------- | -------------- |
| `target`          | 可选 Handle | 要打印的操作   |
| `name`            | 可选字符串  | 打印前缀       |
| `assume_verified` | 可选属性    | 跳过验证       |
| `use_local_scope` | 可选属性    | 局部作用域打印 |
| `skip_regions`    | 可选属性    | 跳过子区域     |

**返回值：** 无

**WHY 需要 print：**

- 调试转换序列
- 验证中间结果
- 理解转换流程

**使用示例：**

```text
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 打印匹配结果
  transform.print %ops { name = "Matched loops" }

  // 应用转换
  %tiled = transform.loop.tile %ops [32]

  // 打印转换结果
  transform.print %tiled { name = "After tiling" }

  transform.yield
}
```

### 5.7 transform.verify - 验证 IR

**语法：**

```text
transform.verify %target : !transform.any_op
```

**参数：**

| 参数     | 类型   | 说明         |
| -------- | ------ | ------------ |
| `target` | Handle | 要验证的操作 |

**返回值：** 无

**WHY 需要 verify：**

- 确保 IR 合法性
- 类似断言的作用
- 捕获转换错误

**使用示例：**

```text
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transform" }

  %tiled = transform.loop.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After transform" }

  transform.yield
}
```

### 5.8 transform.alternatives - 备选方案

**语法：**

```text
%result = transform.alternatives %scope : !transform.any_op {
^bb0(%arg0: !transform.any_op):
  // 备选方案 1
  %r1 = transform.try_vectorize %arg0
  transform.yield %r1 : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 备选方案 2
  %r2 = transform.scalar_optimize %arg0
  transform.yield %r2 : !transform.any_op
}
```

**参数：**

| 参数           | 类型        | 说明             |
| -------------- | ----------- | ---------------- |
| `scope`        | 可选 Handle | 备选方案的作用域 |
| `alternatives` | 区域列表    | 备选方案区域     |

**返回值：** 第一个成功方案的返回值

**WHY 需要 alternatives：**

- 尝试多种优化策略
- 提供回退机制
- 提高转换成功率

**使用示例：**

```text
// 尝试不同的向量化策略
%result = transform.alternatives %loops : !transform.any_op {
^bb0(%arg0: !transform.any_op):
  // 策略 1: 向量化 + 循环分发
  %v = transform.vectorize %arg0 vector_sizes [256]
  %d = transform.distribute %v
  transform.yield %d : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 2: 分块 + 向量化
  %t = transform.tile %arg0 [32]
  %v = transform.vectorize %t vector_sizes [32]
  transform.yield %v : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 3: 仅向量化（回退）
  %v = transform.vectorize %arg0 vector_sizes [128]
  transform.yield %v : !transform.any_op
}
```

---

## 6. 解释器（InterpreterPass）与 Pass 系统

### 6.1 Transform 解释器架构

```
用户调用
  ↓
InterpreterPass::runOnOperation()
  ├─ 获取预加载的Transform库（getPreloadedTransformModule）
  ├─ 定位Payload根（debugPayloadRootTag 或 Pass锚点操作）
  ├─ 查找Transform入口点（findTransformEntryPoint → @__transform_main）
  ├─ 解析绑定参数（parseArguments → RaggedArray<MappedValue>）
  └─ 执行Transform（applyTransformNamedSequence）
       ├─ 必要时合并符号表（mergeSymbolsInto）
       └─ 创建TransformState，逐op解释执行
```

### 6.2 findTransformEntryPoint：入口点查找

```cpp
// 两层搜索策略
NamedSequenceOp findTransformEntryPoint(Operation *root,
                                        ModuleOp module,
                                        StringRef entryPoint) {
  NamedSequenceOp found;

  // 第一层：在root中前序遍历查找
  root->walk<WalkOrder::PreOrder>([&](NamedSequenceOp op) {
    if (op.getSymName() == entryPoint) {
      found = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // 第二层：在外部module（预加载库）中查找
  if (!found && module)
    module->walk<WalkOrder::PreOrder>([&](NamedSequenceOp op) { ... });

  return found;
}
```

**默认入口点名称：** `__transform_main`（常量`kTransformEntryPointSymbolName`）

### 6.3 PreloadLibraryPass：库预加载

**WHY 需要预加载？**

Transform库（命名序列集合）可能跨多个`.mlir`文件定义，解释器需要统一符号表来解析序列调用。相比在Pass执行时逐次加载，预加载避免重复解析。

**执行流程：**

```
1. expandPathsToMLIRFiles：展开目录路径，收集所有.mlir文件
2. 逐文件解析：parseTransformModuleFromFile
3. 合并符号表：mergeSymbolsInto（创建虚拟根模块 __transform）
4. 加载到方言：TransformDialect::loadIntoLibraryModule
```

**跨文件符号解析：** 库A中的named_sequence可以调用库B中的sequence，通过合并后的统一符号表实现。

### 6.4 CheckUses Pass：use-after-free 检测

Transform Handle类似于C++中的指针——当被操作"消费"（delete）后，就不应再使用。CheckUses Pass进行静态分析：

**核心算法（TransformOpMemFreeAnalysis）：**

```
1. 收集所有"释放点"：freedBy[handle] = {可能释放它的操作集合}

2. 对每个handle的每个使用点，检查isUseLive：
   a. 建立从定义点到使用点的祖先链
   b. 检查祖先链中是否存在释放操作
   c. 在控制流中：如果任意分支释放了handle，汇聚点视为失效

3. 保守性：may-free即报警（宁可false positive，不遗漏true positive）
```

**控制流示例：**

```text
^bb1:
  transform.consume %0 : !transform.any_op  // 在bb1中释放
  cf.br ^bb3

^bb2:
  cf.br ^bb3

^bb3:
  // 警告：%0可能在bb1路径中被释放
  transform.use %0 : !transform.any_op  // use-after-free!
```

### 6.5 InferEffects Pass：自动推断副作用

手动为每个named_sequence参数标注`{transform.readonly}`或`{transform.consumed}`容易出错。InferEffects Pass自动分析：

```
1. 遍历所有FunctionOpInterface操作（named_sequence）
2. 对每个块参数：
   - 检查是否被传入某个"消费性"操作的operand
   - 如果是 → 标注 {transform.consumed}
   - 否则 → 标注 {transform.readonly}
3. 设置对应属性
```

---

## 7. 扩展机制：TransformDialectExtension

Transform方言可以通过`TransformDialectExtension`机制向其他方言注入额外的操作，解耦**核心方言**与**特定变换**。

### 7.1 扩展机制背景与动机

**问题：** Transform 方言需要扩展？

- 不同方言需要特定的转换操作
- 核心方言不应依赖特定方言
- 需要延迟加载和类型安全

**设计目标：**

1. **延迟加载**：扩展只在需要时加载
2. **解耦合**：Transform 方言不依赖特定方言
3. **类型安全**：自动验证扩展操作的接口实现
4. **易用性**：简单的 API 注册操作和类型

### 7.2 扩展机制设计原理

#### 7.2.1 CRTP 模式

```cpp
// TransformDialect.h
template <typename DerivedTy>
class TransformDialectExtension 
    : public DialectExtension<DerivedTy, TransformDialect, ExtraDialects...> {
public:
  void apply(MLIRContext *context, TransformDialect *transformDialect,
             ExtraDialects*...) const {
    // 加载dependent dialects
    for (const DialectLoader &loader : dialectLoaders)
      loader(context);

    // 执行initializers（注册新操作）
    for (const Initializer &init : initializers)
      init(transformDialect);
  }

protected:
  explicit TransformDialectExtension(bool buildOnly = false)
      : buildOnly(buildOnly) {
    // 调用派生类的实现
    static_cast<DerivedTy *>(this)->init();
  } 

  // 注册操作
  template <typename... OpTys>
  void registerTransformOps() {
    dialect->addOperations<OpTys...>();
  }

  // 声明依赖方言
  template <typename DialectTy>
  void declareDependentDialect() {
    dialect->declareDependentDialect<DialectTy>();
  }

protected:
  TransformDialect *dialect;
};

// Build-Only模式（只构建IR，不执行，不会产生未声明依赖的方言）
template <typename DerivedTy>
class BuildOnly : public DerivedTy {
  BuildOnly() : DerivedTy(/*buildOnly=*/true) {}
};
```

**注入的操作必须：**

1. 实现`TransformOpInterface`（或`PatternDescriptorOpInterface`等等价接口）
2. 实现`MemoryEffectsOpInterface`
3. 使用点分前缀命名（如`transform.affine.reschedule`）

**WHY 使用 CRTP：**

- 编译时多态，避免虚函数开销
- 类型安全的扩展注册
- 简洁的 API 设计

#### 7.2.2 初始化流程

```cpp
// TransformDialect.cpp
void TransformDialect::initialize() {
  // 步骤 1: 注册核心操作
  addOperations<
#define GET_OP_LIST
#include "TransformOps.cpp.inc"
  >();

  // 步骤 2: 注册扩展
  for (const ExtensionInitialization &entry : extensionsToInitialize) {
    entry.initialize(*this);
  }
}
```

### 7.3 核心组件详解

#### 7.3.1 registerTransformOps - 注册操作

```cpp
// 使用示例
void MyExtension::init() {
  registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
  >();
}
```

#### 7.3.2 declareDependentDialect vs declareGeneratedDialect

```cpp
// 依赖方言：扩展操作使用的类型
declareDependentDialect<LinalgDialect>();

// 生成方言：转换可能产生的操作
declareGeneratedDialect<SCFDialect>();
declareGeneratedDialect<VectorDialect>();
```

**WHY 区分两者：**

- **依赖方言**：必须在加载扩展前加载
- **生成方言**：转换执行时需要加载

#### 7.3.3 registerTypes - 注册类型

```cpp
// 注册自定义类型
void registerTypes() {
  dialect->addTypes<
#define GET_TYPEDEF_LIST
#include "MyTransformTypes.cpp.inc"
  >();
}
```

### 7.4 Transform官方内置扩展

#### 7.4.1 DebugExtension

提供Transform程序内部的观测能力：

```text
// 在操作位置发出Remark
transform.debug.emit_remark_at %op, "found target" : !transform.any_op

// 将参数值作为Remark输出
transform.debug.emit_param_as_remark %tile_size, "tile size" at %op
    : !transform.param<i64>, !transform.any_op
```

**用途：** 调试Transform脚本，无需停止执行即可观察中间状态。

#### 7.4.2 LoopExtension

```text
// 循环不变量外提
transform.loop.hoist_loop_invariant_subsets %loop
    : (!transform.op<"scf.for">) -> !transform.op<"scf.for">

// 循环分割（将最后一次迭代分离）
%main, %remainder = transform.loop.peel %loop
    : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

// 循环展开
transform.loop.unroll %loop { factor = 4 }
    : !transform.op<"scf.for">
```

#### 7.4.3 PDLExtension

将PDL（Pattern Description Language）与Transform方言集成，支持声明式的模式匹配：

```text
transform.with_pdl_patterns %root : !transform.any_op {
^bb0(%arg: !transform.any_op):
  // 定义PDL模式
  pdl.pattern @match_matmul_attrA : benefit(1) {
    %attr = attribute
    %0 = operation "linalg.matmul" {"test.attrA" = %attr} -> ...
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg failures(propagate) {
  ^bb1(%root: !transform.any_op):
    // 使用PDL模式匹配
    %matches = pdl_match @match_matmul_attrA in %root : ...
    // 对匹配的op应用变换
    transform.structured.tile_using_for %matches tile_sizes [4,4] : ...
  }
}
```

**PDLExtension的内部机制：**

1. `PatternApplicatorExtension`：延迟编译PDL模式（首次请求时才编译）
2. `PDLMatchOp`：遍历payload，应用PDL模式，收集匹配的操作

#### 7.4.4 TuneExtension

支持超参数搜索和自动调优：

```text
// 声明一个可调参数
%tile_size = transform.tune.knob<"tile_size"> = #16
    from options = [8, 16, 32] -> !transform.param<i64>

// 使用该参数
transform.structured.tile_using_for %op tile_sizes [%tile_size] : ...
```

**使用场景：** 外部搜索框架（如AutoTVM）遍历不同的`selected`值，评估编译结果的性能。

#### 7.4.5 IRDLExtension

IRDL（IR Definition Language）是MLIR中用于**动态描述操作约束**的方言。IRDLExtension将IRDL与Transform方言结合，提供无需注册操作即可按约束匹配的能力。

**唯一操作：`transform.irdl.collect_matching`**

```text
// 用IRDL描述约束，收集所有满足条件的操作
%matched = transform.irdl.collect_matching in %root
    : (!transform.any_op) -> !transform.any_op {
^bb0(%arg: !transform.any_op):
  irdl.dialect @test {
    irdl.operation @whatever {
      // 约束：结果类型必须是 i32 或 i64
      %t_i32 = irdl.is i32
      %t_i64 = irdl.is i64
      %t_any  = irdl.any_of(%t_i32, %t_i64)
      irdl.results(foo: %t_any)
    }
  }
}
// %matched 持有所有结果类型为 i32 或 i64 的 test.whatever 操作
```

来自测试用例 `irdl.mlir`：上面的Transform脚本对两个 `test.whatever` 操作发出"matched"备注——结果类型为 `f32` 的那个不匹配，被自动排除。

**与PDLExtension的区别：**

| 维度           | PDLExtension                 | IRDLExtension             |
| -------------- | ---------------------------- | ------------------------- |
| 描述方式       | PDL语言（图形匹配）          | IRDL语言（类型/属性约束） |
| 匹配粒度       | 操作+操作数+结果的整体图匹配 | 单个操作的类型约束匹配    |
| 适用场景       | 复杂的数据流模式             | 按类型系统约束筛选op      |
| 是否需要注册op | 否                           | 否                        |

**WHY需要IRDLExtension？**

PDL擅长匹配"结构"（操作之间的连接关系），而IRDL擅长描述"约束"（类型系统层面的限定）。当只需要按类型约束筛选操作时，IRDL比PDL更简洁直观。例如"找出所有输出为向量类型且元素为浮点数的操作"，用IRDL的类型约束比PDL的图模式更自然。

**内部实现：**

```cpp
// IRDLCollectMatchingOp::apply 的核心逻辑
DiagnosedSilenceableFailure IRDLCollectMatchingOp::apply(...) {
  // 1. 从body中提取IRDL操作描述（DialectOp + OperationOp）
  auto dialect = cast<irdl::DialectOp>(getBody().front().front());
  irdl::OperationOp operation = *body.getOps<irdl::OperationOp>().begin();

  // 2. 根据IRDL描述创建验证器（不注册操作，只创建约束检查器）
  auto verifier = irdl::createVerifier(operation, {}, {});

  // 3. 注册空的诊断handler（抑制约束不匹配时产生的诊断，视为正常的"不匹配"）
  auto handlerID = getContext()->getDiagEngine().registerHandler(
      [](Diagnostic &) { return success(); });  // 吞掉所有诊断

  // 4. 遍历payload，逐个尝试验证
  SmallVector<Operation *> matched;
  for (Operation *payload : state.getPayloadOps(getRoot()))
    payload->walk([&](Operation *target) {
      if (succeeded(verifier(target)))
        matched.push_back(target);  // 满足约束则收集
    });

  getContext()->getDiagEngine().eraseHandler(handlerID);
  results.set(cast<OpResult>(getMatched()), matched);
  return DiagnosedSilenceableFailure::success();
}
```

**关键设计：空诊断handler**

IRDL验证器在约束不满足时会发出诊断（错误信息）。IRDLCollectMatchingOp需要的是"静默的约束测试"——不匹配不是错误，只是"筛掉了"。因此注册一个空handler吞掉所有诊断，验证失败只用返回值判断，不产生任何噪音。

**当前限制（代码注释中标注）：**

- body中只允许一个 `irdl.dialect` 操作
- `irdl.dialect` 中只允许一个 `irdl.operation`
- 暂不支持 `irdl.type` 和 `irdl.attribute`（TODO注释）

### 7.5 TransformDialectData - 扩展间通信机制

```cpp
// TransformState.h
class TransformDialectData {
public:
  template <typename T>
  T &get() {
    TypeID id = TypeID::get<T>();
    auto it = data.find(id);
    if (it == data.end()) {
      it = data.emplace(id, std::make_unique<T>()).first;
    }
    return static_cast<T &>(*it->second);
  }

private:
  DenseMap<TypeID, std::unique_ptr<Extension>> data;
};
```

**WHY 需要扩展间通信：**

- 共享转换状态
- 避免重复计算
- 支持协作式转换

### 7.6 扩展自动加载机制详解

#### 7.6.1 完整加载流程

```
┌─────────────────────────────────────────────────────────────┐
│                    扩展自动加载流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 应用启动                                                 │
│     ├── 注册所有扩展（不加载）                               │
│     └── 记录扩展 → 操作映射                                  │
│                                                             │
│  2. Pass 运行                                               │
│     ├── 解析 Transform IR                                    │
│     ├── 识别使用的操作                                       │
│     └── 触发扩展加载                                         │
│                                                             │
│  3. 扩展加载                                                │
│     ├── 加载依赖方言                                         │
│     ├── 注册操作和类型                                       │
│     └── 应用扩展到 Dialect                                  │
│                                                             │
│  4. 转换执行                                                │
│     └── 所有操作可用                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 7.6.2 扩展应用逻辑

```cpp
// TransformDialect.cpp
void TransformDialect::loadAvailableExtensions() {
  for (const auto &entry : extensionRegistry) {
    if (isExtensionRequired(entry)) {
      // 加载扩展
      entry.create(*this);
      loadedExtensions.insert(entry.typeid);
    }
  }
}

bool TransformDialect::isExtensionRequired(
    const ExtensionEntry &entry) {
  // 检查是否需要此扩展
  for (Operation &op : getOperations()) {
    if (entry.providedOps.contains(op.getName())) {
      return true;
    }
  }
  return false;
}
```

**WHY 自动加载：**

- 用户体验：无需手动加载
- 按需加载：只加载需要的扩展
- 避免循环依赖：延迟加载机制

### 7.7 完整扩展示例：LinalgTransformDialectExtension

```cpp
// LinalgTransformDialectExtension.h
class LinalgTransformDialectExtension
    : public ::mlir::transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgTransformDialectExtension)

  using Base::Base;

  void init() {
    // 声明依赖
    declareDependentDialect<LinalgDialect>();
    declareGeneratedDialect<SCFDialect>();

    // 注册操作
    registerTransformOps<
#define GET_OP_LIST
#include "LinalgTransformOps.cpp.inc"
    >();
  }
};
```

---

## 8. 执行模型 (Execution Model)

### 8.1 执行流程概述

Transform 方言的执行是一个**逐步应用转换**的过程，每个 Transform 操作通过 `TransformOpInterface::apply()` 方法实现。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Transform 执行流程                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 解析与验证阶段                                                         │
│     ├── 解析 Transform IR                                                │
│     ├── 验证类型约束                                                      │
│     └── 检查操作定义                                                      │
│                                                                         │
│  2. 状态初始化阶段                                                         │
│     ├── 创建 TransformState                                              │
│     ├── 建立 Payload IR 根映射                                            │
│     └── 初始化 Handle → Payload 对象映射                                   │
│                                                                          │
│  3. 转换执行阶段                                                           │
│     ├── 调用 TransformOpInterface::apply()                               │
│     │   ├── 场景 A：成功 → 更新 Handle 映射                                │
│     │   ├── 场景 B：Silenceable Failure → 回滚，尝试备选                   │
│     │   └── 场景 C：Definite Failure → 立即停止                           │
│     └── 处理 Handle 失效                                                 │
│                                                                         │
│  4. 清理阶段                                                              │
│     ├── 移除 nullptr 操作                                                 │
│     ├── 压缩 Handle 映射                                                  │
│     └── 验证最终状态                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 详细执行步骤

#### 8.2.1 解析与验证 Transform IR

```cpp
// TransformInterpreterUtils.cpp
LogicalResult transform::applyTransformNamedSequence(
    RaggedArray<MappedValue> bindings, TransformOpInterface transformRoot,
    ModuleOp transformModule, const TransformOptions &options) {

  // 步骤 1: 创建 TransformState
  TransformState state(transformRoot->getRegion(), /*payloadRoot=*/nullptr,
                      bindings, options);

  // 步骤 2: 应用 Transform 操作
  DiagnosedSilenceableFailure result = state.applyTransform(transformRoot);

  // 步骤 3: 检查执行结果
  if (failed(result.checkAndReport())) {
    return failure();
  }

  return success();
}
```

#### 8.2.2 应用单个 Transform 操作

```cpp
// TransformInterfaces.cpp
DiagnosedSilenceableFailure TransformState::applyTransform(
    TransformOpInterface transform) {
  // 步骤 1: 创建 TransformRewriter
  TransformRewriter rewriter(transform->getContext());

  // 步骤 2: 设置 Rewriter 的监听器
  auto listener = createTrackingListener(rewriter);
  rewriter.setListener(listener.get());

  // 步骤 3: 创建 TransformResults 容器
  TransformResults results(transform->getNumResults());

  // 步骤 4: 调用操作的 apply 方法
  DiagnosedSilenceableFailure result =
      transform.apply(rewriter, results, *this);

  // 步骤 5: 处理执行结果
  if (succeeded(result.isSuccess())) {
    if (failed(updateStateFromResults(results, transform->getResults()))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    recordOpHandleInvalidations(transform);
  }

  return result;
}
```

### 8.3 失败处理机制

#### 8.3.1 Silenceable Failure（可恢复失败）

**定义：** 转换未能应用，但 Payload IR 未被修改，可以尝试其他转换。

**特征：**

- 转换未修改 Payload IR（原子性保证）
- 可以尝试备选转换
- 延迟报告错误

**使用场景：**

```text
// 场景 1: 尝试多种转换策略
transform.alternatives {
^bb0(%arg0: !transform.any_op):
  // 策略 A：尝试向量化
  %v = transform.try_vectorize %arg0
  transform.yield %v : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // 策略 B：向量化失败，尝试标量优化
  %s = transform.scalar_optimize %arg0
  transform.yield %s : !transform.any_op
}
```

**WHY 这样设计：**

- **灵活性**：允许"尽力而为"的转换策略
- **容错性**：某个转换失败不影响整个流程
- **探索性**：尝试多种优化，选择最佳的

#### 8.3.2 Definite Failure（不可恢复失败）

**定义：** Payload IR 可能处于不一致状态，必须立即停止。

**特征：**

- Payload IR 可能已被部分修改
- 必须立即停止，不能继续执行
- 立即报告错误

**WHY 区分两种失败：**

| 特性            | Silenceable    | Definite          |
| --------------- | -------------- | ----------------- |
| Payload IR 状态 | 未修改         | 可能不一致        |
| 后续操作        | 可以继续       | 必须停止          |
| 错误报告        | 可延迟         | 立即报告          |
| 典型场景        | 前置条件不满足 | 内部错误/约束违反 |

### 8.4 Handle 失效规则 (Handle Invalidation)

当 Transform 操作消费或修改 Payload 操作时，相关的 Handle 会自动失效。

#### 8.4.1 失效触发条件

```cpp
void TransformState::recordOpHandleInvalidations(
    TransformOpInterface transform) {
  // 步骤 1: 获取被消费的 Handle 操作数
  SmallVector<OpOperand *> consumedOperands =
      getConsumedHandleOpOperands(transform);

  // 步骤 2: 检查每个被消费的 Handle
  for (OpOperand *operand : consumedOperands) {
    Value handle = operand->get();
    ArrayRef<Operation *> payloadOps = getPayloadOpsView(handle);

    for (Operation *payloadOp : payloadOps) {
      if (payloadOp->isDead()) {
        invalidatedHandles.insert(handle);
      }
    }
  }
}
```

#### 8.4.2 失效规则图解

```
Handle 失效规则
├── 消费 OperationHandle
│   ├── ✓ 该 Handle 本身失效
│   ├── ✓ 指向嵌套操作的 Handle 失效
│   └── ✓ 指向操作结果的 ValueHandle 失效
│
└── 消费 ValueHandle
    ├── ✓ 产生该值的操作 Handle 失效
    ├── ✓ 指向嵌套操作的 Handle 失效
    └── ✓ 指向包含该值的块参数的 Handle 失效
```

**WHY 这样设计：**

- **安全性**：防止引用已删除/替换的操作
- **一致性**：确保 Handle 指向有效的 Payload IR
- **可预测性**：明确的失效规则，易于理解

### 8.5 TransformRewriter 与 TrackingListener

#### 8.5.1 TransformRewriter 的特殊功能

```cpp
class TransformRewriter : public PatternRewriter {
public:
  void replaceOp(Operation *op, ValueRange newValues) override {
    if (listener) {
      listener->notifyOperationReplaced(op, newValues);
    }
    PatternRewriter::replaceOp(op, newValues);
  }

  void eraseOp(Operation *op) override {
    if (listener) {
      listener->notifyOperationErased(op);
    }
    PatternRewriter::eraseOp(op);
  }
};
```

#### 8.5.2 TrackingListener 的映射更新逻辑

```cpp
class TrackingListener : public RewriterBase::Listener {
public:
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    SmallVector<Value> handles;
    (void)state.getHandlesForPayloadOp(op, handles);

    for (Value handle : handles) {
      if (!newValues.empty()) {
        Operation *newOp = newValues[0].getDefiningOp();
        if (newOp) {
          state.updateMapping(handle, op, newOp);
        }
      } else {
        state.invalidateHandle(handle);
      }
    }
  }

private:
  TransformState &state;
};
```

---

## 9. 扩展开发完整教程

### 9.1 扩展开发步骤概览

```
扩展开发流程
│
├── 步骤 1: 定义扩展类
│   └── 继承 TransformDialectExtension
│
├── 步骤 2: 使用 TableGen 定义操作
│   ├── 继承 TransformDialectOp
│   ├── 实现 TransformOpInterface
│   └── 定义操作参数和结果
│
├── 步骤 3: 实现 C++ 类
│   ├── 实现 apply 方法
│   ├── 实现 getEffects 方法
│   └── 处理错误情况
│
├── 步骤 4: 注册扩展
│   └── 通过 DialectRegistry 注册
│
└── 步骤 5: 测试扩展
    └── 编写单元测试和集成测试
```

### 9.2 步骤 1：定义扩展类

```cpp
// MyTransformOps.h
#pragma once

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace my {
namespace transform {

class MyTransformDialectExtension
    : public ::mlir::transform::TransformDialectExtension<
          MyTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MyTransformDialectExtension)

  using Base::Base;

  void init() {
    // 声明依赖方言
    declareDependentDialect<MyDialect>();

    // 声明生成方言
    declareGeneratedDialect<::mlir::scf::SCFDialect>();
    declareGeneratedDialect<::mlir::vector::VectorDialect>();

    // 注册 Transform 操作
    registerTransformOps<
#define GET_OP_LIST
#include "MyTransformOps.cpp.inc"
    >();
  }
};

} // namespace transform
} // namespace my
```

### 9.3 步骤 2：使用 TableGen 定义操作

```text
// MyTransformOps.td
#ifndef MY_TRANSFORM_OPS
#define MY_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MyCustomTransformOp : TransformDialectOp<"my_custom",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Applies my custom transformation to target operations";

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    OptionalAttr<I64Attr>$param,
    UnitAttr:$verbose
  );

  let results = (outs
    TransformHandleTypeInterface:$result
  );

  let assemblyFormat = [{
    $target `(` $param^ `,` `verbose` $verbose^?`)` attr-dict
      `:` type($target) `->` type($result)
  }];

  let hasVerifier = 1;
}

#endif // MY_TRANSFORM_OPS
```

### 9.4 步骤 3：实现 C++ 类

```cpp
// MyTransformOps.cpp
#include "MyTransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::transform;

namespace {

struct MyCustomTransformOp
    : public Op<MyCustomTransformOp,
               TransformOpInterface::Trait,
               MemoryEffectsOpInterface::Trait> {
  using Op::Op;

  DiagnosedSilenceableFailure apply(
      TransformRewriter &rewriter,
      TransformResults &results,
      TransformState &state) override {

    // 步骤 1: 获取目标操作
    ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

    if (targets.empty()) {
      return emitSilenceableError()
             << "no operations found to transform";
    }

    // 步骤 2: 获取可选参数
    int64_t param = 0;
    if (auto paramAttr = getParam()) {
      param = paramAttr.getInt();
    }

    bool verbose = getVerboseAttr().hasValue();

    // 步骤 3: 对每个目标应用转换
    SmallVector<Operation *> transformedOps;
    transformedOps.reserve(targets.size());

    for (Operation *target : targets) {
      if (!isValidTarget(target)) {
        return emitDefaultSilenceableFailure(target);
      }

      FailureOr<Operation *> result = applyMyTransform(
          rewriter, target, param, verbose);

      if (failed(result)) {
        if (result.error.isSilenceable()) {
          return result.error;
        } else {
          return emitDefiniteFailure()
                 << "internal error during transformation";
        }
      }

      transformedOps.push_back(*result);
    }

    // 步骤 4: 设置结果
    results.set(cast<OpResult>(getResult()), transformedOps);

    return DiagnosedSilenceableFailure::success();
  }
};

} // namespace
```

### 9.5 步骤 4：注册扩展

```cpp
// MyTransformDialectExtension.cpp
#include "MyTransformOps.h"

using namespace mlir;
using namespace my::transform;

// 注册扩展到 DialectRegistry
void registerMyTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<
      MyTransformDialectExtension
  >();
}
```

### 9.6 步骤 5：测试扩展

```cpp
// unittests/MyTransformOpsTest.cpp

class MyTransformTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<transform::TransformDialect>();
    context.loadDialect<MyDialect>();

    DialectRegistry registry;
    my::transform::registerMyTransformDialectExtension(registry);
    context.appendDialectRegistry(registry);
  }

  MLIRContext context;
};

TEST_F(MyTransformTest, BasicTransform) {
  // 构造测试 IR
  Builder builder(&context);
  auto moduleOp = builder.create<ModuleOp>(builder.getUnknownLoc());

  // 构造 Transform IR
  auto transformModule = parseTransformModule(R"(
    transform.sequence {
    ^bb0(%root: !transform.any_op):
      %ops = transform.my_custom %root { param = 64 : i64 }
      transform.yield %ops : !transform.any_op
    }
  )");

  // 应用 Transform
  TransformOptions options;
  auto result = applyTransformNamedSequence(
      moduleOp, entryPoint, transformModule, options);

  EXPECT_TRUE(succeeded(result));
}
```

### 9.7 最佳实践

#### 9.7.1 编写 Transform 序列

**DO（推荐做法）：**

```text
// 使用 named_sequence 组织代码
transform.named_sequence @optimize_op(%arg: !transform.any_op) {
  %1 = transform.tile %arg [32]
  %2 = transform.vectorize %1
  transform.yield %2
}

// 使用 include 复用
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  %result = transform.include @optimize_op(%ops)
  transform.yield %result
}

// 添加错误处理
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"my.op"} in %root
  %result = transform.apply_patterns to %ops { ... }
  transform.yield %result
}
```

**DON'T（不推荐做法）：**

```text
// 过长的内联序列
transform.sequence {
^bb0(%root: !transform.any_op):
  %1 = transform.step1 %root
  %2 = transform.step2 %1
  // ... 50 多行 ...
  %50 = transform.step50 %49
}

// 重复代码
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"op1"} in %root
  %tiled1 = transform.tile %ops1 [32]
  %vect1 = transform.vectorize %tiled1

  %ops2 = transform.match.ops{"op2"} in %root
  %tiled2 = transform.tile %ops2 [32]  // 重复
  %vect2 = transform.vectorize %tiled2  // 重复
}
```

#### 9.7.2 调试技巧

**技巧 1：使用 print 调试**

```text
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  transform.print %ops { name = "Matched loops" }

  %tiled = transform.tile %ops [32]
  transform.print %tiled { name = "After tiling" }
}
```

**技巧 2：使用 verify 确保正确性**

```text
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transform" }

  %tiled = transform.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After tiling" }
}
```

#### 9.7.3 性能考虑

**考虑 1：减少 Handle 查找**

```text
// 不推荐：重复查找
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops1 = transform.match.ops{"scf.for"} in %root
  // 使用 %ops1
  %ops2 = transform.match.ops{"scf.for"} in %root  // 重复查找
}

// 推荐：复用 Handle
transform.sequence {
^bb0(%root: !transform.any_op):
  %ops = transform.match.ops{"scf.for"} in %root
  // 使用 %ops
}
```

**考虑 2：批量操作**

```text
// 推荐：一次处理所有操作
transform.sequence {
^bb0(%root: !transform.any_op):
  %all_ops = transform.match.ops{"scf.for"} in %root
  transform.tile %all_ops [32]
}
```

---

## 10. 实战案例

### 10.1 案例1：矩阵乘法优化（tile + vectorize）

**目标：** 优化矩阵乘法运算，提高缓存局部性和并行度。

```text
// ============================================================
// 初始 Payload IR
// ============================================================
module {
  func.func @matmul(%A: tensor<1024x1024xf32>,
                    %B: tensor<1024x1024xf32>,
                    %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = linalg.matmul ins(%A, %B: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                         outs(%C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}

// ============================================================
// Transform IR：完整的优化序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // --------------------------------------------------------
  // 步骤 1: 匹配所有 matmul 操作
  // --------------------------------------------------------
  %matmuls = transform.structured.match ops {["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // 打印找到的操作（调试）
  transform.print %matmuls { name = "Found matmuls" }

  // --------------------------------------------------------
  // 步骤 2: 应用多层级分块
  // WHY: 提高缓存局部性，减少内存访问延迟
  // --------------------------------------------------------
  // 第一层分块：较大块（L2 缓存友好）
  %tiled_l1, %loops_l1 = transform.structured.tile %matmuls [64, 64, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // 打印第一层分块结果
  transform.print %tiled_l1 { name = "After L1 tiling" }

  // 第二层分块：较小块（L1 缓存友好）
  %tiled_l2, %loops_l2 = transform.structured.tile %tiled_l1 [8, 8, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // --------------------------------------------------------
  // 步骤 3: 向量化最内层循环
  // WHY: 利用 SIMD 指令，提高并行度
  // --------------------------------------------------------
  transform.structured.vectorize %tiled_l2 vector_sizes [4, 8]
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 4: 应用公共子表达式消除
  // WHY: 减少冗余计算
  // --------------------------------------------------------
  transform.apply_cse %root : !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 验证最终 IR
  // --------------------------------------------------------
  transform.verify %root : !transform.any_op

  transform.yield
}
```

**优化效果：**

- **缓存局部性**：通过多层级分块，提高数据重用
- **向量化**：利用 SIMD 指令，提高计算吞吐量
- **代码简化**：CSE 消除冗余计算

### 10.2 案例2：循环嵌套优化（多层级tile）

**目标：** 优化深层嵌套循环结构。

```text
// ============================================================
// 初始 Payload IR：三层嵌套循环
// ============================================================
module {
  func.func @nested_loops(%arg0: tensor<1024x1024x1024xf32>)
      -> tensor<1024x1024x1024xf32> {
    %0 = tensor.empty() : tensor<1024x1024x1024xf32>
    scf.for %i = 0 to 1024 {
      scf.for %j = 0 to 1024 {
        scf.for %k = 0 to 1024 {
          %1 = tensor.extract %arg0[%i, %j, %k] : tensor<1024x1024x1024xf32>
          %2 = arith.addf %1, %1 : f32
          %3 = tensor.insert %2 into %0[%i, %j, %k] : tensor<1024x1024x1024xf32>
        }
      }
    }
    return %0 : tensor<1024x1024x1024xf32>
  }
}

// ============================================================
// Transform IR：多层级循环优化
// ============================================================
transform.named_sequence @optimize_nested_loops(%root: !transform.any_op) {
  // --------------------------------------------------------
  // 步骤 1: 匹配所有 scf.for 循环
  // --------------------------------------------------------
  %all_loops = transform.match ops {["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 2: 获取最外层循环
  // --------------------------------------------------------
  %outer_loops = transform.loop.get_outermost %all_loops
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 3: 应用多层级循环分块
  // WHY: 每一层对应不同的缓存层级
  // --------------------------------------------------------

  // L3 缓存层：大分块
  %tiled_l3, %loops_l3 = transform.loop.tile %outer_loops [256, 256, 256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // L2 缓存层：中等分块
  %tiled_l2, %loops_l2 = transform.loop.tile %tiled_l3 [64, 64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // L1 缓存层：小分块
  %tiled_l1, %loops_l1 = transform.loop.tile %tiled_l2 [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // --------------------------------------------------------
  // 步骤 4: 循环展开（最内层）
  // WHY: 减少循环控制开销
  // --------------------------------------------------------
  %innermost = transform.loop.get_innermost %loops_l1
      : (!transform.any_op) -> !transform.any_op

  transform.loop.unroll %innermost { factor = 4 }
      : !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 向量化
  // --------------------------------------------------------
  transform.loop.vectorize %tiled_l1 vector_sizes [4, 4]
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 6: 应用 LICM（循环不变代码外提）
  // --------------------------------------------------------
  transform.apply_licm %root : !transform.any_op

  transform.yield %root : !transform.any_op
}

// ============================================================
// 主序列：应用优化
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %result = transform.include @optimize_nested_loops(%root)
      : (!transform.any_op) -> !transform.any_op

  transform.yield %result : !transform.any_op
}
```

**优化策略说明：**

| 层级 | 分块大小 | 目标缓存 | WHY                  |
| ---- | -------- | -------- | -------------------- |
| L1   | 256      | L3 缓存  | 最大化 L3 缓存利用率 |
| L2   | 64       | L2 缓存  | 适应 L2 缓存大小     |
| L3   | 16       | L1 缓存  | 适应 L1 缓存大小     |
| 展开 | 4        | 寄存器   | 减少分支开销         |

### 10.3 案例3：GPU映射案例

**目标：** 将计算映射到 GPU 执行。

```text
// ============================================================
// 初始 Payload IR：简单的并行计算
// ============================================================
module {
  func.func @vector_add(%a: tensor<1024x1024xf32>,
                        %b: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = scf.for %i = 0 to 1024 iter_args(%acc = %0) -> tensor<1024x1024xf32> {
      %2 = scf.for %j = 0 to 1024 iter_args(%inner_acc = %acc) -> tensor<1024x1024xf32> {
        %3 = tensor.extract %a[%i, %j] : tensor<1024x1024xf32>
        %4 = tensor.extract %b[%i, %j] : tensor<1024x1024xf32>
        %5 = arith.addf %3, %4 : f32
        %6 = tensor.insert %5 into %inner_acc[%i, %j] : tensor<1024x1024xf32>
        scf.yield %6 : tensor<1024x1024xf32>
      }
      scf.yield %2 : tensor<1024x1024xf32>
    }
    return %1 : tensor<1024x1024xf32>
  }
}

// ============================================================
// Transform IR：GPU 映射序列
// ============================================================
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // --------------------------------------------------------
  // 步骤 1: 匹配目标函数
  // --------------------------------------------------------
  %funcs = transform.match ops {["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 2: 匹配循环操作
  // --------------------------------------------------------
  %loops = transform.match ops {["scf.for"]} in %root
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 3: GPU 映射策略
  // WHY: 将循环映射到 GPU 线程层级
  // --------------------------------------------------------

  // 方案 A：使用 forall 并行构造
  %result = transform.alternatives %loops : !transform.any_op {
  ^bb0(%arg0: !transform.any_op):
    // 尝试转换为 forall + GPU 映射
    %forall = transform.loop.to_forall %arg0
        : (!transform.any_op) -> !transform.any_op

    %gpu = transform.gpu.map %forall
        { grid_dims = [32, 32], block_dims = [16, 16] }
        : (!transform.any_op) -> !transform.any_op

    transform.yield %gpu : !transform.any_op
  }, {
  ^bb0(%arg0: !transform.any_op):
    // 方案 B：直接 GPU 映射（回退）
    %gpu = transform.gpu.launch %root
        { blocks = [32, 32, 1], threads = [16, 16, 1] }
        : (!transform.any_op) -> !transform.any_op

    transform.yield %gpu : !transform.any_op
  }

  // --------------------------------------------------------
  // 步骤 4: 向量化（使用 GPU 向量宽度）
  // --------------------------------------------------------
  %vectors = transform.gpu.vectorize %result vector_size = 128
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 5: 内存优化
  // --------------------------------------------------------
  // 共享内存优化
  %shared = transform.gpu.use_shared_memory %vectors
      { buffer_size = 4096 }
      : (!transform.any_op) -> !transform.any_op

  // --------------------------------------------------------
  // 步骤 6: 验证 GPU IR
  // --------------------------------------------------------
  transform.verify %root : !transform.any_op

  transform.print %root { name = "Final GPU IR" }

  transform.yield %root : !transform.any_op
}
```

**GPU 映射策略：**

| GPU 层级 | 映射目标      | 线程数          |
| -------- | ------------- | --------------- |
| Grid     | 整个计算域    | 32 x 32 blocks  |
| Block    | 单个 block 内 | 16 x 16 threads |
| Vector   | SIMD 宽度     | 128             |

---

## 11. 调试与排错

### 11.1 常见错误类型及解决方案

#### 11.1.1 Handle 类型不匹配

**错误示例：**

```
error: 'transform.loop.tile' op operand type mismatch:
  expected '!transform.op<"scf.for">', got '!transform.any_op'
```

**原因：** 操作期望特定类型的 Handle，但提供了通用类型。

**解决方案：**

```text
// 不正确
%loops = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
transform.loop.tile %loops [32]  // 错误：类型不匹配

// 正确：先进行类型转换
%loops = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
%typed_loops = transform.cast %loops to !transform.op<"scf.for">
    : (!transform.any_op) -> !transform.op<"scf.for">
transform.loop.tile %typed_loops [32]
```

#### 11.1.2 Handle 被重复消费

**错误示例：**

```
error: handle has already been consumed
```

**原因：** Handle 被消费后仍被使用。

**解决方案：**

```text
// 不正确
%handle = transform.match ops {["scf.for"]} in %root
%tiled1 = transform.loop.tile %handle [32]
%tiled2 = transform.loop.tile %handle [32]  // 错误：已被消费

// 正确：克隆 Handle
%handle = transform.match ops {["scf.for"]} in %root
%handle1, %handle2 = transform.split_handle %handle
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
%tiled1 = transform.loop.tile %handle1 [32]
%tiled2 = transform.loop.tile %handle2 [32]
```

#### 11.1.3 Transform 操作失败

**错误示例：**

```
error: transform failed: silenceable failure at location
```

**原因：** 转换前置条件不满足。

**解决方案：**

```text
// 添加错误处理
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root

  // 尝试转换
  %result = transform.loop.tile %ops [32]
      : (!transform.any_op) -> !transform.any_op
      or {
        // 回退方案
        transform.yield %ops : !transform.any_op
      }

  transform.yield %result : !transform.any_op
}
```

### 11.2 调试工具使用

#### 11.2.1 使用 print 调试

```text
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 调试：打印每步中间结果
  %ops1 = transform.match ops {["scf.for"]} in %root
  transform.print %ops1 { name = "Step 1: Matched loops" }

  %ops2 = transform.loop.tile %ops1 [32]
  transform.print %ops2 { name = "Step 2: After tiling" }

  %ops3 = transform.loop.vectorize %ops2
  transform.print %ops3 { name = "Step 3: After vectorize" }

  transform.yield
}
```

#### 11.2.2 使用 verify 检查

```text
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  // 在关键点验证 IR
  %ops = transform.match ops {["scf.for"]} in %root

  // 转换前验证
  transform.verify %ops { name = "Before transformation" }

  // 应用转换
  %tiled = transform.loop.tile %ops [32]

  // 转换后验证
  transform.verify %tiled { name = "After transformation" }

  transform.yield
}
```

#### 11.2.3 使用 mlir-transform-opt 工具

```bash
# 应用 Transform 并打印结果
mlir-opt input.mlir \
  --transform-interpreter \
  --transform-spec-library=transform.mlir

# 调试模式
mlir-opt input.mlir \
  --transform-interpreter=debug \
  --transform-spec-library=transform.mlir

# 只验证 Transform IR（不执行）
mlir-opt transform.mlir --verify-diagnostics
```

### 11.3 问题诊断流程

```
┌─────────────────────────────────────────────────────────────┐
│                    问题诊断流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 确认问题类型                                            │
│     ├── 编译错误？→ 检查类型和语法                           │
│     ├── 运行时错误？→ 检查 Handle 映射                       │
│     └── 转换失败？→ 检查前置条件                             │
│                                                             │
│  2. 收集信息                                                │
│     ├── 添加 print 语句                                      │
│     ├── 添加 verify 检查                                     │
│     └── 启用调试输出                                        │
│                                                             │
│  3. 定位问题                                                │
│     ├── 哪个 Transform 操作失败？                            │
│     ├── 哪个 Payload 操作导致失败？                          │
│     └── Handle 指向正确的操作吗？                           │
│                                                             │
│  4. 解决问题                                                │
│     ├── 修复类型不匹配                                       │
│     ├── 添加条件检查                                         │
│     └── 使用 alternatives 提供回退                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. 性能与最佳实践

### 12.1 性能考虑

#### 12.1.1 减少不必要的 Handle 查找

```text
// 不推荐：重复查找
%ops = transform.match ops {["scf.for"]} in %root
%tiled1 = transform.loop.tile %ops [32]

%ops2 = transform.match ops {["scf.for"]} in %root  // 重复
%tiled2 = transform.loop.tile %ops2 [16]

// 推荐：复用 Handle
%ops = transform.match ops {["scf.for"]} in %root
%h1, %h2 = transform.split_handle %ops
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
%tiled1 = transform.loop.tile %h1 [32]
%tiled2 = transform.loop.tile %h2 [16]
```

#### 12.1.2 批量操作优于单个操作

```text
// 推荐：批量处理
%all_ops = transform.match ops {["linalg.matmul"]} in %root
transform.structured.tile %all_ops [64, 64, 16]
transform.structured.vectorize %all_ops
```

#### 12.1.3 使用 named_sequence 提高复用性

```text
// 定义可复用的优化序列
transform.named_sequence @optimize_linalg_op(%op: !transform.any_op) {
  %tiled = transform.structured.tile %op [32, 32]
  transform.structured.vectorize %tiled
  transform.apply_cse %tiled
  transform.yield
}

// 应用
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.structured.match ops {["linalg.matmul"]} in %root
  %result = transform.foreach %ops
      iter_args(%op: !transform.any_op)
      -> (!transform.any_op) {
  ^bb0(%op: !transform.any_op):
    %optimized = transform.include @optimize_linalg_op(%op)
        : (!transform.any_op) -> !transform.any_op
    transform.yield %optimized : !transform.any_op
  }
  transform.yield
}
```

### 12.2 最佳实践

#### 12.2.1 错误处理策略

```text
// 策略 1：使用 suppress 模式处理可选转换
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  // 尝试优化，失败时继续
  %try = transform.loop.unroll %ops { factor = 4 }
      or transform.yield %ops
}

// 策略 2：使用 alternatives 提供回退
%result = transform.alternatives %ops {
^bb0(%arg0):
  // 乐观策略
  %fast = transform.fast_path %arg0
  transform.yield %fast
}, {
^bb0(%arg0):
  // 保守回退
  %safe = transform.safe_path %arg0
  transform.yield %safe
}
```

#### 12.2.2 模块化 Transform 序列

```text
// 将复杂转换分解为小模块
module {
  // 模块 1：循环优化
  transform.named_sequence @optimize_loops(%root: !transform.any_op) {
    // ...
  }

  // 模块 2：向量化
  transform.named_sequence @vectorize(%root: !transform.any_op) {
    // ...
  }

  // 模块 3：内存优化
  transform.named_sequence @optimize_memory(%root: !transform.any_op) {
    // ...
  }

  // 主序列：组合模块
  transform.named_sequence @full_optimization(%root: !transform.any_op) {
    %r1 = transform.include @optimize_loops(%root)
    %r2 = transform.include @vectorize(%r1)
    %r3 = transform.include @optimize_memory(%r2)
    transform.yield %r3
  }
}
```

#### 12.2.3 文档和注释

```text
// ============================================================
// 优化序列：矩阵乘法
// 目标：提高缓存局部性，启用向量化
// ============================================================
transform.named_sequence @optimize_matmul(%op: !transform.any_op) {
  // 步骤 1: 分块 L2 缓存
  // WHY: 提高数据重用，减少内存访问
  %tiled_l2 = transform.structured.tile %op [64, 64, 16]

  // 步骤 2: 分块 L1 缓存
  %tiled_l1 = transform.structured.tile %tiled_l2 [8, 8, 4]

  // 步骤 3: 向量化最内层
  // WHY: 利用 SIMD 指令
  transform.structured.vectorize %tiled_l1 vector_sizes [4, 8]

  transform.yield
}
```

---

## 13. 常见问题FAQ

### Q1: Handle类型不匹配怎么办？

**问题：** 收到 "operand type mismatch" 错误。

**解决方案：**

```text
// 方案 1：使用 cast 转换类型
%any_handle = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.any_op
%typed_handle = transform.cast %any_handle to !transform.op<"scf.for">
    : (!transform.any_op) -> !transform.op<"scf.for">

// 方案 2：使用正确的匹配类型
%typed_handle = transform.match ops {["scf.for"]} in %root
    : (!transform.any_op) -> !transform.op<"scf.for">
```

### Q2: Transform操作执行失败如何调试？

**问题：** 转换失败但不知道原因。

**调试步骤：**

```text
// 1. 添加 print 调试
transform.sequence failures(propagate) {
^bb0(%root: !transform.any_op):
  %ops = transform.match ops {["scf.for"]} in %root
  transform.print %ops { name = "Before transform" }

  %tiled = transform.loop.tile %ops [32]
  transform.print %tiled { name = "After transform" }
}

// 2. 添加 verify 检查
transform.verify %ops { name = "Verification failed" }

// 3. 使用 suppress 模式继续执行
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  // 即使某些转换失败，继续执行
}
```

### Q3: 如何选择Transform还是Pass？

**对比：**

| 特性         | Transform 方言 | 传统 Pass |
| ------------ | -------------- | --------- |
| **粒度**     | 精细控制       | 粗粒度    |
| **组合性**   | 灵活组合       | 固定顺序  |
| **调试性**   | 可观察         | 难以定位  |
| **性能**     | 略有开销       | 高效      |
| **适用场景** | 实验性/研究    | 生产环境  |

**选择建议：**

- **使用 Transform**：需要精细控制、实验新优化、条件化转换
- **使用 Pass**：已知优化序列、性能关键、生产环境

### Q4: 扩展加载失败的常见原因

**问题：** 自定义 Transform 操作无法使用。

**常见原因：**

```cpp
// 1. 忘记注册扩展
// 错误：
// DialectRegistry registry;
// context.loadDialect<TransformDialect>();

// 正确：
DialectRegistry registry;
my::transform::registerMyTransformDialectExtension(registry);
context.appendDialectRegistry(registry);

// 2. 依赖方言未加载
// 确保声明所有依赖方言
void MyExtension::init() {
  declareDependentDialect<LinalgDialect>();
  declareDependentDialect<SCFDialect>();
  // ...
}

// 3. 操作名称拼写错误
// 检查 TableGen 定义与使用是否一致
def MyOp : TransformDialectOp<"my_op"> { ... }
// 使用: transform.my_op (不是 transform.myOp)
```

---

## 14. 参考资料

### 14.1 官方文档

- [Transform Dialect - Overview](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)

### 14.2 源代码位置

| 组件             | 路径                                              |
| ---------------- | ------------------------------------------------- |
| 核心方言定义     | `mlir/include/mlir/Dialect/Transform/IR/`         |
| 核心实现         | `mlir/lib/Dialect/Transform/IR/`                  |
| Linalg Transform | `mlir/lib/Dialect/Linalg/TransformOps/`           |
| SCF Transform    | `mlir/lib/Dialect/SCF/TransformOps/`              |
| 接口定义         | `mlir/include/mlir/Dialect/Transform/Interfaces/` |

### 14.3 术语表

| 术语                | 英文                | 解释                                      |
| ------------------- | ------------------- | ----------------------------------------- |
| Payload IR          | Payload IR          | 被转换的目标 IR                           |
| Transform IR        | Transform IR        | 控制转换逻辑的 IR                         |
| Handle              | Handle              | Transform IR 中指向 Payload IR 对象的引用 |
| Silenceable Failure | Silenceable Failure | 可恢复失败                                |
| Definite Failure    | Definite Failure    | 不可恢复失败                              |

### 14.4 延伸阅读

- [MLIR 编写转换指南](https://mlir.llvm.org/docs/Transformations/)
- [Linalg 结构化操作](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [SCF 结构化控制流](https://mlir.llvm.org/docs/Dialects/SCF/)
