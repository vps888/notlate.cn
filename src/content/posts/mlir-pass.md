---
title: "一文读懂MLIR的Pass机制"
description: "本文选取 MLIR 的 Pass Manager （ mlir/lib/Pass/Pass.cpp 及相关头文件）作为解读对象。Pass Manager 是整个编译器的\"调度中枢\"，它的实现不算很长——核心代码约 2000 行——但几乎每一个设计决策背后都有值得细品的权衡。 一、为什么选这段代…"
slug: "mlir-pass"
legacyId: 19729533
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19729533"
pubDate: 2026-03-17
category: "AI 编译器"
tags: ["AI 编译器","MLIR","编译器 Pass"]
featured: true
---

> 本文选取 MLIR 的 **Pass Manager**（`mlir/lib/Pass/Pass.cpp` 及相关头文件）作为解读对象。Pass Manager 是整个编译器的"调度中枢"，它的实现不算很长——核心代码约 2000 行——但几乎每一个设计决策背后都有值得细品的权衡。

---

## 一、为什么选这段代码？

MLIR（Multi-Level Intermediate Representation）是 LLVM 生态里的新一代编译器基础设施，由 Google 发起，现在是整个社区在一起维护。它想解决的问题说起来其实挺直接：传统编译器里有太多互相不兼容的 IR，各自为政，很难复用。MLIR 的思路是提供一套统一的框架，让不同层次的 IR 可以在同一套工具链下共存。

Pass Manager 在这套框架里负责的事情是：按照用户定义的顺序，把一系列变换（Pass）依次施加在 IR 上。听起来简单，但要做到支持嵌套调度、并发执行、分析缓存管理、还有完善的调试支持，里面的设计就很有嚼头了。

---

## 二、先看整体文件布局

```
mlir/include/mlir/Pass/
├── Pass.h                  # Pass 基类与接口定义
├── PassManager.h           # PassManager / OpPassManager 接口
└── PassInstrumentation.h   # 插桩接口

mlir/lib/Pass/
├── Pass.cpp                # 核心实现，约 1800 行
├── PassManagerOptions.cpp
└── PassTiming.cpp          # 计时插桩的具体实现
```

接下来我们从四个角度来看这段代码：**架构设计**、**设计模式**、**并发模型**、**性能细节**。

---

## 三、架构设计：让 Pass 树和 IR 树长一个样

### 3.1 核心思路

MLIR 的 IR 是树状的：一个 `Operation` 里可以嵌套 `Region`，`Region` 里有 `Block`，`Block` 里又有更多 `Operation`。Pass Manager 的结构跟这棵树对齐——不同层级的 IR 对应不同层级的 `OpPassManager`，整体也是嵌套的树形结构。

```cpp
// mlir/include/mlir/Pass/PassManager.h（有所简化）

class OpPassManager {
public:
  // 往当前层加一个 Pass
  void addPass(std::unique_ptr<Pass> pass);

  // 嵌套一层：为某种子 Op 类型单独配一套 pipeline
  OpPassManager &nest(StringRef nestedOpName);

  // 跑起来
  LogicalResult run(Operation *op, AnalysisManager am);

private:
  SmallVector<std::unique_ptr<PassConcept>> passes;
  std::optional<StringAttr> opName;  // 这一层对应哪种 Op
};

class PassManager : public OpPassManager {
  // 顶层入口，持有线程池、instrumentation 等全局资源
  MLIRContext *context;
  std::unique_ptr<PassInstrumentor> instrumentor;
  // ...
};
```

`PassManager` 直接继承 `OpPassManager`，本身就是嵌套结构的根节点。跑起来的时候，外层遍历 IR 树，找到跟 `opName` 匹配的 Operation，就交给对应层的 pipeline 处理。

画出来长这样：

```
PassManager (顶层，比如 "builtin.module")
├── Pass A          ← 作用于整个模块
├── OpPassManager ("func.func")
│   ├── Pass B      ← 只作用于每个函数
│   └── Pass C
└── Pass D          ← 又回到模块级别
```

这样一来，"模块级优化"和"函数级优化"可以自然地写在同一个 pipeline 里，不需要任何特殊处理，结构本身就把这件事说清楚了。

### 3.2 Pass 基类设计得很轻

```cpp
// mlir/include/mlir/Pass/Pass.h（核心片段）

class Pass {
public:
  virtual ~Pass() = default;

  // 用户只需要实现这一个方法
  virtual void runOnOperation() = 0;

  // 这个 Pass 处理哪种 Op
  virtual StringRef getOpName() = 0;

  // 并发时需要克隆出独立副本
  virtual std::unique_ptr<Pass> clone() const = 0;

protected:
  // 在 runOnOperation 里，就这样拿分析结果
  template <typename AnalysisT>
  AnalysisT &getAnalysis() {
    return getAnalysisManager().getAnalysis<AnalysisT>();
  }

private:
  // 运行时由框架注入，Pass 自己不持有
  detail::PassExecutionState *passState = nullptr;
};
```

`Pass` 基类本身不持有任何 IR 的引用。`passState` 是每次执行时由框架临时"塞进来"的，执行完就清掉。这个设计让 Pass 对象天然可以被复用，也可以安全地克隆给其他线程。

---

## 四、四个值得细说的设计模式

### 4.1 Concept/Model 模式：让框架"夹在中间"做事

MLIR 内部用了一种在 C++ 里不算常见但很好用的手法——Concept/Model 模式，本质上是一种类型擦除：

```cpp
// mlir/lib/Pass/Pass.cpp（简化展示）

// Concept：框架内部看到的接口，签名跟用户的 runOnOperation 不同
struct PassConcept {
  virtual ~PassConcept() = default;
  virtual LogicalResult run(Operation *op, AnalysisManager am,
                            PassInstrumentor *pi, ...) = 0;
  virtual std::unique_ptr<PassConcept> clone() const = 0;
  virtual Pass *getPass() = 0;
};

// Model：真正持有用户 Pass 的壳，负责"中间那层"的工作
template <typename PassT>
struct PassModel : public PassConcept {
  explicit PassModel(std::unique_ptr<PassT> pass)
      : pass(std::move(pass)) {}

  LogicalResult run(Operation *op, AnalysisManager am,
                    PassInstrumentor *pi, ...) override {
    // ① 注入运行时上下文
    pass->passState = {op, am, ...};
    // ② 通知所有观察者：Pass 要开始了
    if (pi) pi->runBeforePass(pass.get(), op);
    // ③ 终于到用户代码了
    pass->runOnOperation();
    // ④ Pass 结束，再通知一次
    if (pi) pi->runAfterPass(pass.get(), op);
    // ⑤ 清掉临时状态
    pass->passState = nullptr;
    return success();
  }

  std::unique_ptr<PassT> pass;
};
```

你可能会问：直接让用户 Pass 继承一个虚函数接口不就行了，绕这一圈干嘛？

关键在于 `PassConcept::run` 和 `Pass::runOnOperation` 的签名完全不同。`PassModel` 夹在中间，把**状态注入、instrumentation 触发、错误处理**这些"横切关注点"统一在这里处理，用户完全不用操心，只需要写 `runOnOperation` 里的逻辑。框架控制生命周期，用户只管自己的活。

### 4.2 责任链：失败了就停下来

Pass 一个接一个地跑，形成一条责任链：

```cpp
LogicalResult OpPassManager::run(Operation *op, AnalysisManager am) {
  for (auto &pass : passes) {
    // 哪个 Pass 失败了，整条链就断在这里
    if (failed(pass->run(op, am, ...)))
      return failure();

    // 成功后，根据这个 Pass 自己声明的"我保留了哪些分析结果"
    // 来决定哪些缓存可以继续用，哪些要失效
    am.invalidate(pass->getPreservedAnalyses());
  }
  return success();
}
```

配合 `--mlir-pass-pipeline-crash-reproducer` 这个选项，如果链上某个 Pass 直接崩溃了，框架会自动把崩溃前的 IR 状态和 pipeline 配置写到文件里，方便之后复现。这个调试体验在工业级编译器里算是挺贴心的。

### 4.3 观察者模式：让调试能力"插进来"而不是"写进去"

插桩系统（Instrumentation）是观察者模式的教科书实现：

```cpp
// mlir/include/mlir/Pass/PassInstrumentation.h

class PassInstrumentation {
public:
  virtual ~PassInstrumentation() = default;

  // 每个回调都有默认空实现，按需重写就好
  virtual void runBeforePass(Pass *pass, Operation *op) {}
  virtual void runAfterPass(Pass *pass, Operation *op) {}
  virtual void runAfterPassFailed(Pass *pass, Operation *op) {}
  virtual void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) {}
  virtual void runAfterAnalysis(StringRef name, TypeID id, Operation *op) {}
};

// 聚合多个观察者，统一广播事件
class PassInstrumentor {
  SmallVector<std::unique_ptr<PassInstrumentation>, 2> instrumentations;
public:
  void runBeforePass(Pass *pass, Operation *op) {
    for (auto &i : instrumentations)
      i->runBeforePass(pass, op);
  }
  // ...
};
```

框架内置了三种实现，日常工作中你可能都用过：

| 实现类                           | 对应什么                                           |
| -------------------------------- | -------------------------------------------------- |
| `PassTiming`                     | `-mlir-timing`，统计每个 Pass 花了多少时间         |
| `IRPrinterInstrumentation`       | `--mlir-print-ir-after-all`，每个 Pass 前后打印 IR |
| `CrashReproducerInstrumentation` | 崩溃时自动生成可复现文件                           |

如果你想加自己的统计或追踪逻辑，调一下 `PassManager::addInstrumentation()` 就行，一行核心代码都不用改。

### 4.4 享元模式：Analysis 只算一次

支配树、别名分析这类东西算起来很贵，如果每个 Pass 都重新算一遍就太浪费了。MLIR 的 `AnalysisManager` 把结果缓存起来，按需取用，用完了再按规则失效：

```cpp
class AnalysisMap {
public:
  template <typename AnalysisT>
  AnalysisT &getAnalysis(Operation *op, PassInstrumentor *pi) {
    TypeID id = TypeID::get<AnalysisT>();

    // 有缓存？直接拿
    auto it = analyses.find(id);
    if (it != analyses.end())
      return *static_cast<AnalysisT *>(it->second.get());

    // 没有才算，算完存起来
    if (pi) pi->runBeforeAnalysis(AnalysisT::name(), id, op);
    auto *result = new AnalysisT(op);
    analyses.insert({id, ...});
    if (pi) pi->runAfterAnalysis(AnalysisT::name(), id, op);
    return *result;
  }

  // Pass 跑完之后，精准清掉它破坏了的那些分析
  void invalidate(const PreservedAnalyses &pa) {
    for (auto it = analyses.begin(); it != analyses.end();) {
      if (!pa.isPreserved(it->first))
        it = analyses.erase(it);
      else
        ++it;
    }
  }

private:
  DenseMap<TypeID, std::unique_ptr<void, ...>> analyses;
};
```

每个 Pass 可以声明 `preserveAll()`（说明我没破坏任何分析结果）或者列出具体保留了哪些，框架据此精准失效，不该扔的不扔，该扔的一个不留。

---

## 五、并发：用"克隆"换"无锁"

### 5.1 数据并行的思路

一个模块里通常有很多函数，理论上可以对它们并行地跑同一套 Pass pipeline。MLIR 确实这么做了：

```cpp
// mlir/lib/Pass/Pass.cpp（核心并发逻辑，简化）

LogicalResult runOnRegionsWithMultiThreading(...) {

  // 把所有要处理的 Operation 收集起来
  SmallVector<Operation *> opsToProcess;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &childOp : block)
        if (childOp.getName() == opPMs[0]->getOpName())
          opsToProcess.push_back(&childOp);

  std::atomic<bool> hasFailure(false);
  llvm::parallelFor(0, opsToProcess.size(), [&](size_t i) {
    // 每个线程拿到的是独立克隆出来的 pipeline
    OpPassManager localPM = opPMs[0]->clone();
    // 分析缓存也是独立的
    AnalysisManager localAM = am.nest(opsToProcess[i]);

    if (failed(localPM.run(opsToProcess[i], localAM)))
      hasFailure.store(true);
  });

  return failure(hasFailure.load());
}
```

线程安全靠三条规则撑起来：**Pass 实例各自克隆，互不共享；每个 Operation 有自己独立的分析缓存；MLIRContext 在并发阶段只读，写入只在单线程阶段做。** 没有一把锁，靠的是"根本不共享可变状态"。

### 5.2 用 CRTP 帮用户把 clone() 写对

并发能跑起来，前提是每个 Pass 都能被正确克隆。MLIR 通过 CRTP 基类把这件事变成了"继承就自动获得"：

```cpp
// 用户只要这样写，clone() 就自动有了正确的实现
class MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "my-pass"; }
  void runOnOperation() override { /* 你的变换逻辑 */ }
  // clone() 不用写，PassWrapper 帮你生成了
};
```

克隆的语义是：复制配置参数，但不复制运行时状态。`PassWrapper` 通过 CRTP 把这个约束变成了默认行为，用户想写错都很难。

---

## 六、性能上的一些小讲究

### 6.1 SmallVector 随处可见

```cpp
// 大多数 pipeline 不超过几个 Pass，用 SmallVector 在栈上存
SmallVector<std::unique_ptr<PassConcept>, 4> passes;

// 一个模块里的函数数量通常也有限
SmallVector<Operation *, 8> opsToProcess;
```

`SmallVector` 在元素少的时候直接在栈上分配，不走堆。编译器里这类短生命周期的小列表到处都是，积少成多，效果很明显。

### 6.2 TypeID：用指针比较代替字符串比较

`AnalysisMap` 需要按类型查找缓存。传统 RTTI 用 `dynamic_cast` 涉及字符串比较，性能不太可控。MLIR 自己搞了一套 `TypeID`：

```cpp
class TypeID {
public:
  template <typename T>
  static TypeID get() {
    // 每个类型 T 对应唯一的一个静态对象
    // 程序生命周期内地址不变，直接拿地址作为 ID
    static detail::TypeIDResolver<T> resolver;
    return TypeID(&resolver);
  }

  bool operator==(TypeID other) const { return storage == other.storage; }

private:
  const void *storage;
};
```

查找和比较都变成了指针比较，O(1)，比字符串 hash 要快得多也稳得多。

### 6.3 Analysis 懒得不能更懒

```cpp
template <typename AnalysisT>
AnalysisT &getAnalysis() {
  // 只有真的被调用到，才会去算
  // 没有 Pass 需要的分析，一次都不会运行
  return analysisMap.getAnalysis<AnalysisT>(getOperation(), pi);
}
```

惰性求值做到底：分析结果只在第一次被请求时计算，之后复用缓存。如果整个 pipeline 里没有任何 Pass 需要某个分析，那个分析一次都不会运行。

### 6.4 正确性和性能分开管

`enableVerifier(false)` 可以关掉 IR 校验，`--mlir-disable-threading` 可以回退单线程，生产构建用 `-DLLVM_ENABLE_ASSERTIONS=OFF` 关掉断言。这些都是编译期或运行时的开关，不影响核心路径。

调试的时候打开校验，看到完整错误；发布的时候全部关掉，跑最快的路径。这种分离本身就是工程成熟的标志。

---

## 七、错误处理：把"失败"和"崩溃"分清楚

### 7.1 LogicalResult：不用异常也能好好报错

MLIR 里没有异常，错误靠 `LogicalResult` 一路往上传：

```cpp
// 只要有一个 Pass 失败，后面的就不跑了
if (failed(pass->run(op, am, ...)))
  return failure();
```

`LogicalResult` 本质就是个 bool 包装，没有运行时开销，也没有栈展开的不确定性。更重要的是，它让"这个 Pass 告诉我转换失败了"和"这个 Pass 自己挂掉了"在语义上泾渭分明。

### 7.2 崩溃复现器：线上问题不再难追

```cpp
// 一行开启，崩溃时自动留证据
pm.enableCrashReproducerGeneration("crash-reproducer.mlir");

// 内部逻辑大概是这样：
// - 每个 Pass 跑之前，把当前 IR 状态序列化到内存
// - 如果 Pass 导致崩溃，在信号处理器里把快照写到文件
// - 下次用这个文件就能复现崩溃，不需要完整的原始输入
```

这个特性在实际工作中救过不少人——线上环境的输入往往很大很复杂，但崩溃复现器生成的文件只包含触发问题的那个 Pass 和它看到的最小 IR，拿到就能直接调试。

---

## 八、最后：这段代码在说什么？

把这些设计放在一起，会发现它们都在做同一件事：**让用户很难犯错，让框架的能力对用户透明**。

CRTP 基类让 `clone()` 不需要用户手写；Instrumentation 让调试能力可以插拔而不用改核心代码；惰性 Analysis 让"按需计算"变成默认行为而不是用户的责任；`LogicalResult` 让错误传播清晰而轻量。

| 设计维度 | 具体做法                         | 好处是什么                           |
| -------- | -------------------------------- | ------------------------------------ |
| 架构同构 | Pipeline 树和 IR 树结构一致      | 没有额外的概念需要学，天然支持嵌套   |
| 类型擦除 | Concept/Model 模式               | 横切逻辑在框架里统一处理，用户不用管 |
| 扩展点   | Observer 模式的 Instrumentation  | 加调试能力不需要碰核心代码           |
| 并发     | Clone + 数据独立，不共享可变状态 | 没有锁，天然线程安全                 |
| 性能     | TypeID、SmallVector、惰性计算    | 零开销抽象，快的东西还是快           |
| 可靠性   | LogicalResult + 崩溃复现器       | 失败和崩溃语义清晰，出问题有据可查   |

这种"让正确的路最省力"的设计哲学，是 MLIR 在工业落地中越来越受信任的重要原因之一。

---

## 参考资料

- MLIR 官方源码：[github.com/llvm/llvm-project/tree/main/mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
- MLIR 文档：[mlir.llvm.org/docs/PassManagement](https://mlir.llvm.org/docs/PassManagement/)
- Chris Lattner, Jacques Pienaar 等，*MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*，CGO 2021
