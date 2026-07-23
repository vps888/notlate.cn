---
title: "IREE的Flow方言如何高效计算QKV？"
description: "一、问题动机：为什么 QKV 是 必须 做 Multi output Fusion 的场景 以 Transformer 中最典型的结构为例： $$ Q = X W Q,\\quad K = X W K,\\quad V = X W V $$ 朴素实现的问题 在“算子级”视角下，这是 三个独立 Ma…"
slug: "iree-flow-dialect-how-to-efficient-qkv"
legacyId: 19518938
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19518938"
pubDate: 2026-01-22
updatedDate: 2026-01-27
category: "AI 编译器"
tags: ["AI 编译器","IREE","Attention"]
featured: true
---

## 一、问题动机：为什么 QKV 是 *必须* 做 Multi-output Fusion 的场景

以 Transformer 中最典型的结构为例：
$$
[
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V
]
$$
### 朴素实现的问题

在“算子级”视角下，这是 **三个独立 MatMul**：

```
X -> MatMul(Wq) -> Q
X -> MatMul(Wk) -> K
X -> MatMul(Wv) -> V
```

这在 GPU 上会导致：

1. **X 被加载 3 次**
2. 每个 MatMul 都有独立的：**global → shared → register 的数据搬运**
3. 无法在一个 thread block 内复用 X 的 tile
4. 写回 Q/K/V 三次 global memory

在 Attention 这种 **memory-bound + tile-friendly** 的模式中，这是灾难性的。

---

## 二、IREE 的核心思路：不是“算子融合”，而是“调度融合”

> 关键点：
> **IREE 并不是在 linalg 层简单做 op fusion，而是在 Flow 层统一调度（flow control）**

### 核心目标

> **让 Q/K/V 的生成在同一个 tiled dispatch 中完成**

这意味着：

* 同一个 thread block
* 同一个 X tile
* 同一次 global load
* 同时产出 3 个输出 tile

---

## 三、MLIR 中相关方言的职责分工（非常关键）

| 层级 | 方言         | 负责什么                      |
| -- | ---------- | ------------------------- |
| 高层 | linalg     | 表达 *计算语义*（MatMul、Generic） |
| 中层 | flow       | **调度、分块、dispatch 边界、多输出** |
| 低层 | hal / llvm | 设备相关 lowering             |

**Multi-output Fusion 发生的关键位置：Flow 方言**

---

## 四、QKV 在 MLIR 中的“理想语义表达”

在 linalg 层，QKV 本质上是：

```text
%Q = linalg.matmul ins(%X, %Wq) outs(%Q)
%K = linalg.matmul ins(%X, %Wk) outs(%K)
%V = linalg.matmul ins(%X, %Wv) outs(%V)
```

### 注意

* **语义上它们是三个 op**
* **但调度上它们可以共享 iteration space**

这是 Flow 能介入的前提。

---

## 五、Flow 方言如何做 Multi-output Fusion（核心）

### 1. Flow 的核心抽象：`flow.dispatch.workgroups`

Flow 并不关心你是 MatMul 还是 Reduce，它关心的是：

* 一个 dispatch：

  * 输入 buffers
  * 输出 buffers（**可以是多个！**）
  * 一个 tiled 的 workgroup 执行体

**这就是 Multi-output Fusion 的载体**

---

### 2. 从 3 个 linalg.matmul → 1 个 flow.dispatch

IREE 在 Flow lowering 时，会：

1. **识别这些 op：**

   * 相同输入 `%X`
   * 相同 iteration space（M×N×K）
   * 相同 tiling 方式
2. **构建一个 fused dispatch region**
3. dispatch 的 signature 变成：

```text
flow.dispatch.workgroups
  ins(%X, %Wq, %Wk, %Wv)
  outs(%Q, %K, %V)
```

> 注意：
> **Flow dispatch 是天然支持多输出的**

---

### 3. 在 dispatch 内部：单一 tiled loop nest

在 dispatch region 中，逻辑类似：

```text
for m_tile
  for n_tile
    load X_tile
    load Wq_tile
    load Wk_tile
    load Wv_tile

    Q_tile += X_tile @ Wq_tile
    K_tile += X_tile @ Wk_tile
    V_tile += X_tile @ Wv_tile

    store Q_tile
    store K_tile
    store V_tile
```

### GPU 视角（极其重要）

* **一个 thread block**
* **一次 X_tile 的 shared memory load**
* **3 次 dot，但完全复用数据**
* register / shared 级别的 reuse

---

## 六、为什么这不是普通的 Op Fusion

| 维度     | 传统 op fusion        | Flow Multi-output Fusion |
| ------ | ------------------- | ------------------------ |
| 融合单位   | producer → consumer | **并列 producer**          |
| IR 层级  | linalg / affine     | **flow（调度层）**            |
| 输出数    | 单输出                 | **多输出**                  |
| 驱动力    | 消除中间 tensor         | **共享 tile / 调度空间**       |
| GPU 映射 | 指令级                 | **thread-block 级**       |

**QKV 融合不是“消掉中间结果”，而是“改变执行形态”**

---

## 七、Flow Tiling + Distribution 的角色

### Flow 不只是 fusion，还做：

1. **Tiling**

   * 决定 M/N/K tile size
2. **Distribution**

   * tile → workgroup
   * tile 内 → threads
3. **Dispatch boundary**

   * 决定 kernel 级别的融合范围

### 这使得：

* Q/K/V **在同一个 kernel 内**
* 且 tile 边界完全一致
* 确保不会引入跨 kernel 同步

---

## 八、和 Triton / FlashAttention 的关系

### 思想层面对齐

| Triton                    | IREE Flow               |
| ------------------------- | ----------------------- |
| 一个 program_id 产出多个 tensor | 一个 dispatch 产出多个 buffer |
| 显式 tile reuse             | IR 层自动 tile + reuse     |
| programmer 控制             | compiler 决策             |

你可以认为：

> **Flow Multi-output Fusion = 编译器自动生成的 Triton-style kernel**

---

## 九、为什么这种能力对 AI 编译器“非常硬核”

因为它要求：

1. **跨算子分析 iteration space**
2. **允许多输出 kernel**
3. **调度层与计算语义解耦**
4. **IR 能表达“一个 kernel 多结果”**

这正是：

* XLA HLO 做不到
* 传统 LLVM 做不到
* 但 **MLIR + Flow 能做到的原因**

---

## 十、一句话总结

> **IREE 中 QKV 的 Multi-output Fusion 本质上不是算子融合，而是一次“以 Tile 为中心的调度融合”：
> 编译器在 Flow 层构造一个 multi-output dispatch，使得一个 GPU thread block 在一次 X tile 加载中，同时生成 Q/K/V。**
