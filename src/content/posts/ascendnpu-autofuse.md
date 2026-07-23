---
title: "AscendNPU AutoFuse技术分享"
description: "1. 背景 1.1 AI模型演进趋势 现代AI模型架构日益复杂，呈现两个显著特点： 算子细粒度化 ：MoE（混合专家）、多模态模型中大量使用动态小算子组合 结构灵活多变 ：如DeepSeek中的hcPre、hcPost等模块，组合方式多样 1.2 传统方案痛点 手写融合算子面临两大挑战： 开发…"
slug: "ascendnpu-autofuse"
legacyId: 19608695
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19608695"
pubDate: 2026-02-12
updatedDate: 2026-05-14
category: "AI 编译器"
tags: ["AI 编译器","Ascend"]
featured: true
---

## 1. 背景

### 1.1 AI模型演进趋势

现代AI模型架构日益复杂，呈现两个显著特点：

- **算子细粒度化**：MoE（混合专家）、多模态模型中大量使用动态小算子组合
- **结构灵活多变**：如DeepSeek中的hcPre、hcPost等模块，组合方式多样

### 1.2 传统方案痛点

手写融合算子面临两大挑战：

- **开发效率低**：无法跟上模型快速迭代
- **维护成本高**：每个新组合都需要重新开发和优化

### 1.3 PyTorch Inductor的机遇与挑战

PyTorch 2.0引入的Inductor提供基础算子融合能力，但其设计面向GPU：

- GPU：基于 **SIMT 的吞吐型执行模型**，通过大量线程并行与 **隐式缓存层次** 隐藏访存延迟、提升计算单元利用率
- NPU：基于 **向量/矩阵计算单元的吞吐型执行模型**，通过 **显式数据搬运与流水并行** 提高带宽利用率与数据复用效率

直接沿用GPU优化策略无法充分发挥NPU硬件潜力。

### 1.4 问题抽象

NPU的硬件架构决定了其编程模型必须显示管理**数据搬运**与**流水并行**。以向量计算单元为例，其执行过程可抽象为三个阶段：

* 数据载入（Load, GM-->UB）
* 数据计算（Vector Compute）
* 数据载出（Store, UB-->GM）

因此，编译器需要精确确定并协同优化：

* 每次载入与载出的数据规模
* 并行核的分配方式
* 核内迭代空间规模及其展开策略

其目标是通过 **数据搬运（DMA）与计算（Compute）的流水并行重叠**，最大化计算单元利用率，从而充分释放 NPU 的吞吐能力。

### 1.5 AutoFuse的定位

AutoFuse 是 CANN 技术栈中针对 NPU 硬件架构深度优化的JIT算子融合组件：

- **支持Inductor接入**：保持生态兼容
- **架构感知优化**：针对 NPU 分级存储和并行机制
- **自动化代码生成**：从计算图到 AscendC Kernel 的全流程自动化

---

## 2. AutoFuse整体架构

### 2.1 架构介绍

#### 架构图

<p align="center">
  <img src="https://img2024.cnblogs.com/blog/3599704/202602/3599704-20260212153905963-1437135457.png" alt="AutoFuse架构图" width="1000"  loading="lazy" decoding="async"/>
</p>


#### 流程图

<p align="center">
  <img src="https://img2024.cnblogs.com/blog/3599704/202602/3599704-20260212153925078-1656405872.png" alt="AutoFuse流程图" width="1000"  loading="lazy" decoding="async"/>
</p>


#### 术语表

| 术语             | 含义                                                         |
| ---------------- | ------------------------------------------------------------ |
| **AscendLoopIR** | 计算图的统一表达，是 AutoFuse 计算优化的中间表示，目的是用于连接Tensor语义与硬件执行。通过有限的Op定义，实现NPU硬件能力的完备表达。 |
| **HintGraph**    | 原始计算图（融合算子）                                       |
| **ImplGraph**    | 经过 Schedule 模块处理后，硬件感知优化的计算图               |

### 2.2 核心模块

| 模块         | 输入      | 输出          | 核心能力                   |
| ------------ | --------- | ------------- | -------------------------- |
| **Schedule** | HintGraph | 多个ImplGraph | 架构感知的多策略调度       |
| **ATT**      | ImplGraph | Tiling代码    | 基于硬件性能建模的参数求解 |
| **Codegen**  | ImplGraph | Kernel代码    | AscendC Kernel代码生成     |

---

## 3. Schedule：架构感知的多策略调度

Schedule模块通过循环变换（Loop Transform）、并行化（Parallelization）以及缓存管理（Buffer Management）等技术，在不改变计算语义的前提下，将 HintGraph 转换为多组架构感知优化的 ImplGraph，是实现NPU吞吐能力最大化的关键模块。

### 3.1 循环变换

Schedule模块的输入是一个融合算子（一张融合子图，表示为HintGraph），是由一组有相同迭代空间的算子组成，即可以用相同循环结构的Scalar表达。

循环变换则是编译器中一类在不改变程序语义的前提下，对循环的结构、层次和执行顺序进行重构的技术，**目标是**更好地匹配硬件执行模型，以提升并行性、数据局部性和执行效率。

在Schedule中，循环变换模块主要包括以下能力：

* **循环交换（Loop Interchange）**
* **循环合并（Loop Fusion）**
* **循环切分（Loop Tiling / Blocking）**
* **向量化（Vectorization）**

下面介绍一下整体的技术原理。

#### 3.1.1 计算类型（ComputeType）抽象

即使AscendLoopIR所定义的Op是有限的，但是数量依然非常多。虽然Op的计算逻辑不同，但是可以从计算方式的角度进行归类。目前Schedule对计算类型抽象为：

* Elementwise - 逐元素计算类
* Broadcast - 广播类
* Reduce - 规约类
* Transpose - 转置类
* Concat - 拼接类
* Split/Slice - 切分类
* Gather - 跳读类
* Scatter - 跳写类

按照计算类型分类后，可以极大的简化循环变换的难度。

#### 3.1.2 轴分组

> 循环轴（Loop Axis），表示的是Tensor中的某个维度，Schedule的输入就是基于Tensor表达的计算图。

计算类型减少之后，还面临一个问题：每个Op的循环轴的个数不固定，少则1个，多则可能数十个。

为了解决这个问题，我们参考业界的做法，从循环轴的迭代类型分析，将其归纳为3类：

* Parallel：并行类，支持任意维度切分，无数据依赖
* Reduction：规约类，切分需特殊处理，存在简单数据依赖
* Other：其他类，或不建议切分的场景，往往需要具体情况具体分析

由此，Schedule引入了轴分组（Axis Group）技术，即把相同迭代类型的轴分到同一个组。分组规则为：

* Y组：所有并行类的轴
* R组：所有规约类的轴
* N组：其他类型的轴或不可切分的轴

以最常见的几类计算进行介绍：

| ComputeType     | 典型算子   | 迭代类型                 | 轴分组                                   |
| --------------- | ---------- | ------------------------ | ---------------------------------------- |
| **Elementwise** | Add, Mul等 | 并行类                   | 全部划分到Y组                            |
| **Broadcast**   | Broadcast  | 并行类，广播轴需特殊处理 | 全部划分到Y组                            |
| **Reduce**      | Sum, Max等 | 归约类，规约轴需特殊处理 | 归约(R)轴划分到R组，非归约轴划分到Y组    |
| **Concat**      | Concat     | 其他类，拼接轴不可切分   | 拼接轴及其内轴划分到N组，其余的划分到Y组 |

那么，轴分组都有哪些作用呢？有两个作用：

* **Op融合**：不同Op之间可以进行分组合并（Merge Group），统一成相同的循环结构，以实现Op之间的融合；
* **循环合并**：同组中若存在**连续**的轴，则可以将其进行合并，减少轴的数量。

#### 3.1.3 分组合并

先来看轴分组的第一个作用，如果所有Op都可以合并成相同的轴分组，则意味着当前子图可以融合成一个算子。

假设有两个Op，分别是Target和Source，每个Op可以按照上一节进行轴分组，分组的结果有三种情况：Y、R、N。

现在考虑一下：这两个Op能否合并成同一套循环结构？如何合并呢？

**规则如下表**

| #    | Target | Source | 合并条件                                                     | 结果       |
| ---- | ------ | ------ | ------------------------------------------------------------ | ---------- |
| 1    | Y      | Y      | `target.y_group == src.y_group`                              | 保持Y组    |
| 2    | Y      | YR     | `target.y_group == (src.y_group ∪ src.r_group)`              | 升级为YR组 |
| 3    | YR     | Y      | `src.y_group == (target.y_group ∪ target.r_group)`           | 保持YR组   |
| 4    | YR     | YR     | `target.y_group == src.y_group` 且 `target.r_group == src.r_group` | 保持YR组   |
| 5    | Y      | R      | `target.y_group == src.r_group`                              | 降级为R组  |
| 6    | R      | Y      | `target.r_group == src.y_group`                              | 保持R组    |

**合并优先级**

```
R > YR > Y
```

- **R组**优先级最高（Reduce优先）
- **YR组**次之（包含Reduce）
- **Y组**优先级最低（纯并行计算）

##### 关键规则说明

**Y + YR → YR**：Y组的轴必须等于YR组中Y∪R的所有轴，表示Y组可以被"拆分"为Y和R两组。

**YR + YR → YR**：两组的Y和R必须分别相同（考虑轴顺序），这是最严格的合并条件。

**Y + R → R**：Y组完全退化为R组，表示原来认为是并行计算的轴实际上是归约轴。

以第2条规则为例解释含义：

> 假设Target是Add，Shape=[s0, s1, s2]，其属于Elementwise类型，所有轴分到Y组，表示为 Y=[s0, s1, s2], R=[], N=[]；
>
> 假设Source是ReduceSum，Shape=[s0, s1, s2], reduce_axis=s2，其属于Reduction类型，划分结果为 Y=[s0, s1], R=[s2], N=[]；
>
> 若要将Add 和 ReduceSum 合并成相同的循环结构，则可以将Add的轴分组按照ReduceSum的轴分组进行调整即可。

伪代码表示：

```python
# 合并前:
# 	Add Op用Scalar表达的循环结构如下：
for i in s0:
  for j in s1:
    for k in s2:
      add_out[k] = data0[k] + data1[k]

# 	ReduceSum Op用Scalar表达的循环结构如下：
for i in s0:
  for j in s1:
    sum = 0
    for k in s2:
      sum += add_out[k]
    out[j] = sum

# 合并后：
for i in s0:
  for j in s1:
    sum = 0
    for k in s2:
      add_out = data0[k] + data1[k]   # 把Add Op的s2轴合并到ReduceSum的s2轴表达的循环结构中
      sum += add_out
    out[j] = sum
```

#### 3.1.4 循环合并

轴分组的第二个作用是循环合并：将同组中连续的轴进行合并，减少轴的数量，即把连续的循环合并成一层循环。

**为什么要循环合并**？

* **减少循环开销**
* **增强并行性**：合并后的循环体更长，更容易调度多核并行
* **提高硬件利用率**：合并后单次循环处理的数据量更大，可以提高带宽利用率和计算单元利用率

所以关键是：如何判断两个轴是连续的。这在编译器领域是一个非常基础而又核心的问题。

**定义**：连续轴是指在存储和迭代顺序上，相邻的两个轴访问的元素在内存上是连续的或者可以直接按 stride = 1 映射。

常用的判断方法有两种：

**基于存储布局**

> 以行存储为例，假设：Tensor Shape: `[d0, d1, d2]`，Stride: `[d0 * d1 * d2, d2, 1]`
>
> 则连续的判断条件为：**`Stride[i] = Stride[i+1] * Shape[i+1]`**

**基于迭代空间**

> 假设两层循环如下：
>
> ```python
> for i in 0..M:
> for j in 0..N:
>  C[i, j] = A[i, j] + B[j, i]
> ```
>
> 最内层循环 `i` 或 `j` 是否连续，取决于 **内存访问模式**，假设A和B矩阵以行存储为例：
>
> - 访问`A[i, j]`时，`j`内层连续，`i` 则需要跳`Stride = N`。
> - 访问`B[j, i]`时，`i`内层连续，`j` 则需要跳`Stride = M`。

这两种判断方法本质上是等价的。

#### 3.1.5 循环切分

循环切分是将循环的迭代空间划分为若干 **小块（tile / block）**，再对每个块内进行循环迭代的一种优化技术。其主要作用是适配NPU编程模型，这是Schedule调度策略的核心点。可以回顾1.4节提到的关键点：

> 编译器需要精确确定并协同优化：
>
> * 每次载入与载出的数据规模
> * 并行核的分配方式
> * 核内迭代空间规模及其展开策略

为了解决上述问题，业界常用的方法则是**循环切分**，接下来从两个角度解释一下原理：

**基于迭代空间表示**

> 假设原始轴：$z_0 ∈ [0, Z_0)$，切分大小：tile size = $T$，则切分后：
>
> $z_0 = z_0T * T + z_0t$
>
> 其中：
>
> * $z_0T ∈ [0, \lceil \frac{Z_0}{T} \rceil]$             —— **outer (外)轴**，可以用于空间分核（Block），或核内调度轴（Tile）
> * $z_0t ∈ [0, T)$                    —— **inner (内)轴**，可以用于局部计算或向量化
>
> * $z_0T * T + z_0t < Z_0$     —— **边界条件**

**基于Loop表示**

>原始循环伪代码：
>
>```python
>for z0 in 0..Z0:
>	body(z0)
>```
>
>按照tile size = T 切分后伪代码：
>
>```python
>for z0T in 0..ceil(Z0 / T):
> for z0t in 0..T:
>     z0 = z0T * T + z0t
>     if z0 < Z0:
>         body(z0)
>```
>
>其中，$z_0T$和$z_0t$表示含义和约束与第一种表示方式一致。

在Schedule中，基于循环切分原理设计了两种切分方式：

| 切分类型       | 说明      | 表示                       | 解释                                                         |
| -------------- | --------- | -------------------------- | ------------------------------------------------------------ |
| **TileSplit**  | UB切分    | $z_0 = z_0T * T + z_0t$    | UB内单次循环处理数据量$T$，共需循环$\lceil \frac{Z_0}{T} \rceil$次，尾块数据量$Z_0 \% T$。 |
| **BlockSplit** | Block切分 | $z_0T = z_0TB * B + z_0Tb$ | $令Z_0T=\lceil \frac{Z_0}{T} \rceil$，单核循环$B$次，共需$\lceil \frac{Z_0T}{B} \rceil$个核并行，尾核循环$Z_0T \% B$ 次。 |

**其中上述表格中的T和B则是需要求解的参数。**

接下来我们来看一下如何把这两种切分方式作用到合并后的轴分组上。

#### 3.1.6 模板生成

N组设计为不可切分组，因此只需要遍历Y组和R组中的每个合并后的轴进行组合，形成的组合我们称为TilingCase（切分模板），伪代码如下：

```cpp
// 伪代码
for (auto y_id : axes_group.y_group) {
    for (auto r_id : axes_group.r_group) {
        TilingCase case;
        case.ub_tiling_id_y = y_id;  // Y组UB切分轴
        case.ub_tiling_id_r = r_id;  // R组UB切分轴
        cases.push_back(case);
    }
}
```

> 示例：
>
> 假设`Y=[z0z1, z3]`, `R=[z2]`，其中`z0z1`表示原始轴是`z0`和`z1`，因为同组且连续，所以合并成了`z0z1`。
>
> 则组合后共有两种情况：
>
> * TilingCase1 = {ub_tiling_id_y=z0z1, ub_tiling_id_r=z2}
> * TilingCase2 = {ub_tiling_id_y=z3, ub_tiling_id_r=z2}

接下来就是遍历每一个TilingCase，对其y轴和r轴进行切分。

> 以TilingCase1为例：
>
> * TileSplit：z0z1 --> (z0z1T, z0z1t)，z2 --> (z2T, z2t)
> * BlockSplit: z0z1T --> (z0z1TB, z0z1Tb)， z2T --> (z2TB, z2Tb)
>
> 切分之后完整的轴表示为：[z0z1TB, z0z1Tb, z0z1t, z2TB, z2Tb, z2t, z3]
>
> 接着会做一次轴调序，将tile轴排到最内侧，结果表示为：[z0z1TB, z2TB, z0z1Tb, z2Tb, z0z1t, z2t, z3]

这样的一个TilingCase就称为一个模板。在经过优化处理的ImplGraph上，对图中的每个Op进行相同的轴切分动作。切不同的轴则会生成不同的新的ImplGraph，这个ImplGraph就是模板。

#### 3.1.7 向量化

**向量化**是把"对单个标量的重复计算"，重写为"对一组数据的单指令并行计算"，以充分利用硬件的 **SIMD / Vector Compute单元**。

**为什么要向量化**？

* **提升算力利用率**：并行执行的最小数据通道称为lane，标量指令只能用到1个lane，而NPU的向量指令一次可以驱动128个lane(FP16)；
* **摊薄访存开销**：向量化的Load/Store可以实现一条指令搬运连续数据块，从而减少指令数，提高带宽利用率；
* **为流水并行创造条件**：固定且规则的 tile 内计算模式便于DMA和Compute的流水并行重叠。

**向量化的条件**

* 内存访问连续

> 经过上述几步的循环变换，比如对轴分组中的Y组和R组进行了切分等，会将切分后的$z_0t$及剩下的N组内所有轴调整到最内侧，同时为了最大化硬件性能，部分场景还需要对这些轴做32Bytes对齐操作，以确保内存访问连续

* 无跨lane数据依赖

> 数据依赖问题已经通过轴分组合并的方式提前消除

所以，接下来Schedule的逻辑较为简单，只需要将选择最内侧的N个轴（$z_0t$及其内轴）标记为向量化轴即可。

### 3.2 并行化

**并行化**是将一个计算任务拆分成多个**独立或可同时执行的子任务**，让硬件上的多个计算单元同时工作，从而提升整体吞吐量。

**并行化的作用**

* 提升吞吐量：通过核级、UB级Tile并行，充分利用计算单元算力；
* 掩盖延迟（Latency Hiding）DMA和Compute的流水调度重叠；

Schedule主要是通过迭代空间分解（循环合并和切分），将计算任务映射到执行单元：

* **核级并行**：外层Tile（$z_0TB$）分配给不同核

* **UB 内循环并行**：Tile内循环向量化

* **流水调度**：DMA与Compute并行

### 3.3 缓存管理

**缓存管理**（Buffer Management）是指**编译器在生成代码时，显式或隐式地管理数据在不同存储层级之间的调度和使用策略**。以NPU的Vector计算编程模型为例，需要显式处理多层存储：**GM→(L2→)UB→Registers**。其作用体现在：

* **提升算力利用率**：避免计算单元空转，确保数据就绪；与向量化&并行化配合，实现流水并行；
* **减少内存访问带宽压力**：高复用的数据尽量放入快速缓存（UB/寄存器），从而减少重复访问GM；
* **支持流水化和DoubleBuffer**：为 DMA和Compute并行创造条件，让UB内Tile与Vector Compute并行；
* **降低延迟波动**：避免Bank冲突等。

在Schedule中，主要管理两层存储：GM和UB，核心处理技术主要包括Bufferization、Double Buffer、缓存复用等。

#### 3.3.1 Bufferization

Bufferization是将高层抽象的张量操作或循环计算映射到具体的内存缓冲区（Buffer）上。简单的理解就是将Tensor表达的IR转成Buffer表达的IR，将计算逻辑映射到硬件Buffer上。

对于AutoFuse而言，我们设计思路是融合子图中的中间结果数据不出UB。所以对于Schedule而言，Bufferization的主要逻辑包括：

* 为Load/Store节点设置数据搬运方向，这个决定了映射指令。例如Load表示GM-->UB，因此使用MTE2指令。
* 为所有节点分配逻辑上的UB空间，包括输出空间、临时空间等。

#### 3.3.2 Double Buffer

无论硬件是GPU还是NPU，都存在GM / DDR / HBM 延迟远高于计算的问题，就会出现计算等数据（算力空转）现象。

**Double Buffer** 是一种 **软件流水（software pipelining）技术**，使用两个等价的缓冲区，让**数据搬运（DMA）**和 **计算（compute）** 在时间上重叠，从而 **隐藏内存访问延迟**。这种技术在业界也称为Ping-Pong。

**基本流程**

> 1. 首先将NPU Vector计算流水线抽象为三阶段：
>    * Stage 0： Load
>    * Stage 1： Compute
>    * Stage 2： Store
> 2. 然后申请两块等价的UB Buffer，假设为：buffer[0], buffer[1]
> 3. 使用时交替循环：
>
> | 时间 | buffer[0] | buffer[1] |
> | ---- | --------- | --------- |
> | T0   | Load 0    | Idle      |
> | T1   | Compute 0 | Load 1    |
> | T2   | Store 0   | Compute 1 |
> | T3   | Load 2    | Store 1   |
>
> 效果是：任何时刻至少一个Buffer在计算，使计算单元不闲置。

**Double Buffer的约束与代价**

* UB容量压力：需要两块等价空间，若Tile块过大则放不下，所以影响Tile切分。

* 尾块处理逻辑复杂：

  * 不满Vector Width，数据读写长度需要特殊处理，会减少收益。
  * 最后一个Tile计算时，无需再预取下一个块（因为已经没有数据了），因此计算逻辑需要特殊处理，会减少收益。

  > 业界常用的尾块处理策略有3种：
  >
  > 策略1：条件保护（最通用）
  >
  > ```python
  > for i in range(NT + 1):
  >  if i < NT:
  >      load tile i into buffer[i % 2]
  >  if i > 0:
  >      compute tile i-1 from buffer[(i-1) % 2]
  > ```
  >
  > 策略2：循环拆解成三段 Prologue / Steady / Epilogue（最经典）
  >
  > 使用一套指令参数，尾块通过Mask等手段避免越界。
  >
  > ```python
  > # Prologue（预热）
  > load tile 0
  > 
  > # Steady State（满流水）
  > for i in range(1, NT):
  >  load tile i
  >  compute tile i-1
  > 
  > # Epilogue（收尾）
  > compute tile NT-1
  > ```
  >
  > 策略3：尾块单独处理（NPU常用）
  >
  > 综合策略1和策略2，将循环拆成两段，使用两套指令参数，主函数只处理满Tile，尾函数专门处理尾块。
  >
  > ```python
  > # 主函数（使用满块的指令参数）
  > def main_func(...):
  > Prologue
  > Steady(double buffer, no tail)
  > 
  > # 尾函数（使用尾块的指令参数）
  > def tail_func(...):
  > 	if N % T != 0:
  > 	Epilogue
  > ```

* 并非所有场景多有收益：若Compute很慢，而DMA很快，则收益有限。

综上，Schedule在经过上一步的Bufferization之后，会全图遍历，按照一定的规则，在Load/Store节点的IR属性上设置DoubleBuffer是否启用的参数。

#### 3.3.3 缓存复用

缓存复用指的是同一份数据在被逐出高速缓存（UB/shared）之前，被多次用于计算，从而减少对慢速存储（GM/HBM）的重复访问。在 AI 编译器中，缓存复用是所有高性能 Schedule 的中心目标。

**缓存复用的类型**

* **时间复用（Temporal Reuse）**：同一数据，不同时间点使用，又称为跨Loop复用；

  ```python
  # 最典型的是Broadcast运算
  for i:
    load A[i] -> UB    # --> 只需要在外层循环加载一次，内层循环可以重复使用
    for j_tile:
      load B[i, j_tile]
      compute using A[i]
  ```

* **空间复用（Spatial Reuse）**：相邻地址一次搬运，多次使用。这主要体现在Vector运算上，搬运一段连续数据，在向量寄存器或UB中被多个vector lane并行使用，从而摊薄访存开销、提升带宽利用率。

* **跨Op复用（Inner-Op Reuse）**：Op融合后，中间结果不出缓存，直接被下一个Op使用，这也是本项目AutoFuse的最根本的设计原理。

* **跨Tile复用（Tile Reuse）**：一个Tile被多个Tile消费，复用粒度是Tile块，通常发生在特定运算的Op中，比如MatMul、LayerNorm等。

缓存复用不是天然存在的，而是由Schedule **主动创造**：

* **Loop Tiling（最核心操作）**

  由于缓存空间有限，必须把大循环切成合适的小块，才能放进缓存。

  ```python
  for i0 in tiles:
    load tile(i0) into UB
    for i1 in tile:
      compute using UB
  ```

* **Loop Reordering（循环调序，创造复用机会）**

  ```python
  # 以A=(1, 10), B=(20, 10), C=A+B隐式广播为例：
  
  # 优化前：
  for i in 0..20:
      for j in 0..10:
          TA = Load A[0, j]
          TB = Load B[i, j]
          C[i,j] = TA + TB
  # 分析：
  #    1. A[0,j] 每次循环 i 都被重复用
  #    2. 跨 i 轴存在时间复用潜力
  
  # 优化后：第一步轴调序
  for j in 0..10:
      for i in 0..20:
          TA = Load A[0, j]
          TB = Load B[i, j]
          C[i,j] = TA + TB
  # 优化后：第二步空间跨Loop复用
  for j in 0..10:
      TA = Load A[0, j]
      for i in 0..20:
          TB = Load B[i, j]
          C[i,j] = TA + TB
  # 分析：A[0,j] 在内层循环里只访问一次，所以放到外层循环，只需要加载一次，空间跨Loop复用。
  ```

* **显式缓存管理**

  **生命周期分析（Lifetime Analysis）**是编译器在buffer/寄存器管理中最核心的步骤之一。其目的是确定每个中间数据（tensor/Op 输出）在片上缓存或寄存器中需要存在的时间区间，以便于分配有限的高速缓存（UB/Registers）、避免冗余搬运、支持多层复用（跨 tile / vector / Op）。

  **核心思想**

  > 1. **扫描 IR / DAG**，确定每个中间张量出现的定义依赖（Def-Use）链；
  >
  > 2. **确定时间区间**，记录 **first_use / last_use**（表达形式可借鉴LLVM 的 **live interval**）；
  >
  > 3. **合并 / 分配 buffer**：
  >
  > - 两个 tensor 的生命周期不重叠 → 可以共用同一 UB buffer
  > - 两个 tensor 生命周期重叠 → 必须单独分配

  **实现方法**

  > **1. 基于 DAG 的分析**：全局Op层级，指导缓存的分配和复用
  >
  > - 遍历 Op DAG（计算图）
  > - 对每个 tensor 记录 **first_use / last_use**
  > - 得到 live interval
  >
  > 2. **基于Loop / Tile 的分析**：局部层级，循环感知，实现Tile复用，例如MatMul的左矩阵Tile
  >
  > - 对每个 tile 或 loop
  > - 分析 tensor 在 tile 内的 **Definition / Last Use**
  > - 生成 UB / vector register 的分配策略
  >
  > **3. Buffer 重用算法**：在缓存有限的约束下，实现物理层级的分配优化
  >
  > - 按 live interval 做 **贪心分配**
  > - 允许非重叠 tensor 共享同一 UB
  > - 可以结合 double buffer / ping-pong 技术

  Schedule基于上述核心思想，综合三种实现方法对融合子图进行生命周期分析（Lifetime Analysis）。

---

## 4. Codegen：AscendC Kernel代码生成

Codegen模块负责将Schedule生成的ImplGraph转换为可执行的AscendC Kernel代码。其核心挑战在于如何将抽象的IR表达正确映射到硬件感知的API调用和循环结构。

### 4.1 API调用生成

Codegen通过**工厂模式**将IR节点映射到具体的AscendC API调用。

#### 4.1.1 工厂模式架构

```cpp
// API工厂注册
class ApiCallFactory {
    static ApiCallFactory& Instance();
    void Register(const string& type, CreatorFunc creator);
    ApiCall* Create(const string& type, const string& name);
};

// 自动注册宏
#define REGISTER_API_CALL(ClassName) \
    static ApiCallRegister<ClassName> reg_##ClassName(#ClassName)
```

#### 4.1.2 API Call层次结构

```
ApiCall (基类)
    ├── UnaryApiCall (一元运算: Abs, Sin, Cos, ...)
    ├── BinaryApiCall (二元运算: Add, Mul, Sub, ...)
    ├── LoadApiCall / StoreApiCall (数据搬运)
    ├── ReduceApiCall (归约: Sum, Max, Min, ...)
    ├── BroadcastApiCall (广播)
    ├── ConcatApiCall (拼接)
    ├── TransposeApiCall (转置)
    └── GatherApiCall (索引)
```

### 4.2 生成内容

#### Kernel函数（多模板路由）

```cpp
extern "C" __global__ __aicore__ void kernel_name(
    GM_ADDR input, GM_ADDR output,
    GM_ADDR workspace, AutofuseTilingData param) {

    if (TILING_KEY_IS(0)) {
        // 模板0：通用切分
        impl_graph_0_general(input, output, workspace, t);
    } else if (TILING_KEY_IS(1)) {
        // 模板1：非对齐切分
        impl_graph_1_unaligned(input, output, workspace, t);
    }
}
```

#### 模板函数实现

```cpp
// 模板0：对z1轴切分
for (int z0z1Tb = 0; z0z1Tb < z0z1Tb_loop_size; z0z1Tb++) {
    DataCopyPadExtend(ub, gm[...], block_count, block_len, ...);
    Abs(y_local[0], x_local[0], z1t_actual_size);
}

// 模板1：对z0轴切分
for (int z0Tb = 0; z0Tb < z0Tb_loop_size; z0Tb++) {
    DataCopyPadExtend(ub, gm[...], block_count, block_len, ...);
    Abs(y_local[0], x_local[0], z0t_actual_size * z1_axis_size);
}
```

---

## 5. ATT：面向硬件感知的自动Tiling优化框架

经过Schedule模块处理，生成了多个模板，每个模板的Tiling均是未知。如何选择模板？Tiling大小如何设定？这项工作由ATT（Auto Tiling）模块完成。

ATT 模块将 Tiling 参数选择问题抽象为 **硬件感知的约束满足优化问题（CSP+INLP）**，在编译阶段实现计算图与底层硬件的紧耦合调优。其核心理念是：在保证求解精度的前提下，同时控制求解耗时，通过约束优化和性能建模，实现对流水线资源、缓存层次及多核并行度的近似最优配置。

### 5.1 问题抽象

#### 5.1.1 Pipeline 与依赖

在 NPU 执行模型中，单个 Tile 的执行通常包含三个阶段：

- **MTE2**：Load
- **VEC**：Compute
- **MTE3**：Store

由于数据依赖关系存在，三个 Pipe 之间并非完全独立执行，而满足如下先后约束：$\text{MTE2} \rightarrow \text{VEC} \rightarrow \text{MTE3}$。

因此，该问题本质上是一个带 precedence 约束的 makespan 最小化问题，本质归类为：$P \mid prec \mid C_{\max}$。

在实际工程中，算子通常通过 DoubleBuffer 执行：
$$
\text{Tile}_i: \text{MTE2} \rightarrow \text{VEC} \rightarrow \text{MTE3}
$$

$$
\text{Tile}_{i+1}: \text{MTE2} \rightarrow \text{VEC} \rightarrow \text{MTE3}
$$

当流水线进入稳定阶段后，各 Pipe 可实现跨 Tile 重叠执行。此时系统吞吐由最慢 Pipe 决定，整体 Makespan 可近似为：

$$
\text{StageLatency}(\mathbf{T}, \mathbf{C}) =
\max(
\text{StageLatency}_{\text{MTE2}},
\ \text{StageLatency}_{\text{VEC}},
\ \text{StageLatency}_{\text{MTE3}}
)
$$

**解释**：StageLatency 最大值对应稳定阶段吞吐的瓶颈 Pipe，是 ATT 的性能评价函数。

#### 5.1.2 问题定义

给定计算图 $G$ 和硬件环境 $H$，寻找最优 Tiling 参数集合 $\mathbf{T}$ 和多核切分方案 $\mathbf{C}$：

$$
\min_{\mathbf{T}, \mathbf{C}} \text{StageLatency}(\mathbf{T}, \mathbf{C})
$$

约束条件：

$$
(\mathbf{T}, \mathbf{C}) \in \mathcal{F}
$$
其中：

- $\mathbf{T}$：Tiling 参数空间  
- $\mathbf{C}$：多核切分参数  
- $\text{StageLatency}(\mathbf{T}, \mathbf{C})$：算子总执行延迟 （makespan） 
- $\mathcal{F}$：总可行域，由硬件约束、语义约束和性能启发式约束组成

**Makespan 形式等价转换**

引入辅助变量 $M$，问题可等价转化为 makespan 最小化形式：

$$
\min M
$$

$$
\text{s.t. } \text{StageLatency}_{\text{pipe}}(\mathbf{T}, \mathbf{C}) \le M,
\quad \forall \text{pipe} \in \{\text{MTE2}, \text{MTE3}, \text{VEC}\}
$$

由于变量离散、函数非线性，该问题属于 **整数非线性规划（INLP）**，一般为 NP-Hard。

### 5.2 数学建模

#### 5.2.1 StageLatency 建模

在 NPU 中，以 Vector 类计算为例，Load / Store / Vector Compute 分别对应不同 pipe：

| Pipe | 对应阶段       | 延迟符号                            |
| ---- | -------------- | ----------------------------------- |
| MTE2 | Load           | $\text{StageLatency}_{\text{MTE2}}$ |
| MTE3 | Store          | $\text{StageLatency}_{\text{MTE3}}$ |
| VEC  | Vector Compute | $\text{StageLatency}_{\text{VEC}}$  |

##### (1) MTE2：Load Pipe 建模

MTE2 延迟由三部分构成：

1. 数据传输时间  
2. 指令头开销  
3. Pipe 启动头开销  

$$
\text{StageLatency}_{\text{MTE2}} =
T_{\text{transfer}}^{\text{MTE2}}
+
T_{\text{instruction}}^{\text{MTE2}}
+
H_{\text{MTE2}}
$$

其中：

$$
T_{\text{transfer}}^{\text{MTE2}} =
\frac{data\_elements \times DataSize}{Bandwidth}
$$

$$
T_{\text{instruction}}^{\text{MTE2}} =
Count_{ins}^{\text{MTE2}} \cdot h^{\text{MTE2}}_{ins}
$$

因此：

$$
\text{StageLatency}_{\text{MTE2}} =
\frac{data\_elements \times DataSize}{Bandwidth}
+
Count_{ins}^{\text{MTE2}} \cdot h^{\text{MTE2}}_{ins}
+
H_{\text{MTE2}}
$$

##### (2) MTE3：Store Pipe 建模

Store 与 Load 结构对称：

$$
\text{StageLatency}_{\text{MTE3}} =
\frac{data\_elements \times DataSize}{Bandwidth}
+
Count_{ins}^{\text{MTE3}} \cdot h^{\text{MTE3}}_{ins}
+
H_{\text{MTE3}}
$$

##### (3) VEC：Compute Pipe 建模

$$
\text{StageLatency}_{\text{VEC}} =
\frac{FLOPs}{Throughput}
+
Count_{ins}^{\text{VEC}} \cdot h^{\text{VEC}}_{ins}
+
H_{\text{VEC}}
$$

##### (4) 算子总延迟（Makespan）

由于三个 pipe 并行执行，算子总延迟为公式（3）：

$$
\text{StageLatency}(\mathbf{T}, \mathbf{C}) =
\max(
\text{StageLatency}_{\text{MTE2}},
\ \text{StageLatency}_{\text{MTE3}},
\ \text{StageLatency}_{\text{VEC}}
)
$$

**符号说明**

- $data\_elements$：Tile 中数据元素数量  
- $Bandwidth$：DMA 带宽  
- $h^{\text{PIPE}}_{ins}$：对应 Pipe 的单条指令头开销  
- $Count^{\text{PIPE}}_{ins}$：对应 Pipe 指令调用次数（与 TileSize 离散相关）  
- $H_{\text{PIPE}}$：对应 Pipe 启动头开销  

#### 5.2.2 约束类型

ATT 的约束可分为三类：

1. **硬件约束（必满足）**  
2. **语义约束（合法性）**  
3. **性能启发式约束（可选剪枝）**

##### (1) 硬件约束

| 类型     | 表达式                              | 描述                                              |
| -------- | ----------------------------------- | ------------------------------------------------- |
| UB 容量  | $UB(\mathbf{T}) \le UB_\text{max}$  | Tile 及临时 Buffer 占用的 UB 空间不能超过最大容量 |
| 核数限制 | $Core(\mathbf{C}) \le C_\text{max}$ | 多核切分方案使用的核数不能超过总核数              |

其中：
$$
UB(\mathbf{T})
=
\sum_{b \in \mathcal{B}}
\left(
\prod_{i \in \mathcal{D}_b} T_i
\right)
\cdot size_b \ , \ 
\mathcal{B}=\{ input, \ output, \ tempbuffer \}
$$

$$
Core(\mathbf{C}) =
\sum_{k \in \mathcal{K}} C_k , \ 
\mathcal{K} = \{1,2,\dots,m\} \ , \ C_k \in \{0,1\}是否使用第k个核心
$$

严格可行域定义为：
$$
\mathcal{F}_H = \left\{ (\mathbf{T}, \mathbf{C}) 
\mid UB(\mathbf{T}) \le UB_\text{max},\ 
Core(\mathbf{C}) \le C_\text{max}
\right\}
$$

##### (2) 语义约束

| 类型       | 表达式                                         | 描述                             |
| ---------- | ---------------------------------------------- | -------------------------------- |
| 整除约束   | $D_i = k_i \cdot T_i + r_i, \ 0 \le r_i < T_i$ | 确保 Tile 切分不破坏原始张量维度 |
| 父子轴关系 | $T_i \ge T_j$                                  | 保持多层 Tile 的局部性           |

语义可行域为：
$$
\mathcal{F}_S
=
\left\{
\mathbf{T}
\ \middle|\
D_i = k_i T_i + r_i,\ 0 \le r_i < T_i,\ \forall i
\right\}
$$

##### (3) 性能启发式约束（可选）

| 类型            | 表达式                                                       | 描述                               |
| --------------- | ------------------------------------------------------------ | ---------------------------------- |
| Cache line 下限 | $\left( \prod_{i \in \mathcal{D}_b} T_i \right) s_b \ge L_{\text{cache}}$ | 避免 Bank Conflict，提高缓存利用率 |
| 硬件对齐        | $T_i \bmod A = 0, \quad \forall i \in \mathcal{D}$           | DMA / SIMD 对齐要求                |

性能约束记为：$\mathcal{F}_P$

##### (4) 总可行域

$$
\mathcal{F} = \mathcal{F}_H \cap \mathcal{F}_S \cap \mathcal{F}_P
$$

#### 5.2.3 优化目标

带约束的 makespan 最小化：

$$
\min_{\mathbf{T}, \mathbf{C}} \text{StageLatency}(\mathbf{T}, \mathbf{C}) 
\quad \text{s.t. } (\mathbf{T}, \mathbf{C}) \in \mathcal{F}
$$

### 5.3 问题求解

#### 5.3.1 问题难点

ATT面临的是一个极具挑战性的优化问题，难点主要体现在以下几个方面：

**1. NP-Hard问题**

这是一个整数非线性规划（INLP）问题，具体表现为：

* 双线性约束（变量相乘）导致搜索空间非凸

* 非线性的整除约束破坏连续性

* 约束耦合导致变量不能独立优化

**2. 编译性能要求** 

产品目标: 单次求解 < 1ms
实际: Schedule产生多个模板，每个模板可能有几十个节点

- 编译时间直接影响用户开发体验
- 不能为了解的质量牺牲编译速度
- 需要稳定、可预测的求解时间

**3. 约束复杂多样** 

| 约束类型       | 数学性质           | 求解难度 |
| -------------- | ------------------ | -------- |
| UB容量约束     | 双线性不等式       | 高       |
| 核数约束       | 二次不等式         | 中       |
| 整除约束       | 取模运算（不连续） | 高       |
| 对齐约束       | 取模运算           | 中       |
| Cache Line检查 | 线性不等式         | 低       |

**(4) 动态Shape**

- 不能针对特定形状硬编码解
- 需要生成通用的求解代码
- 求解质量跨形状保持一致

**(5) 性能模型不精确**

- 模型误差 → 选错Tile → 性能下降
- 过拟合 → 新场景失效
- 建模成本 → 需要大量实测数据

**(6) 多目标权衡**

优化目标多：

* 最小化执行时间 (主要目标)
* 最大化UB利用率 (避免浪费)
* 最大化多核利用率 (避免空闲)
* 最小化代码大小 (工程考虑)

目标间的冲突：

* 高多核利用率 → 核间通信开销增加
* 高UB利用率 → 可能限制Tile大小

### 5.3.2 分阶段启发式求解

ATT 通过 **分阶段启发式** 将 NP-Hard 问题高效近似求解：

**阶段 1：UB 受限优化**  

- 主约束 UB 容量  
- 由于 $\text{StageLatency}_\text{compute}$ 随 Tile 增大单调下降，而 UB 使用量单调上升，最优 Tile 通常位于 UB 边界。

**阶段 2：Core 扩展近似**  

- 在单核结构确定后，多核扩展近似线性：  

$$
\text{TotalTime} \approx \frac{\text{Work}}{\text{CoreCount}}
$$

- 可采用 **Block Coordinate Descent** 先优化 Tile，再优化 Core。

**阶段 3：档位离散搜索**  

- 性能函数为分段结构（向量宽度固定、DMA burst 对齐）  
- 搜索仅在结构切换点、容量边界、对齐点进行  

#### 轴排序求解器

**定位**：
Tiling求解器的目标是基于输入Shape确定合适的分核及分块大小，以获得尽可能好的Kernel性能。轴排序求解器支持的启发式优先级排序规则（`优先`是指优先不切的轴）及基础算法（如`UB占用核数占用权衡算法`、`UB优先贪心算法`、`对称切分`）可以很好地保证基础解的性能，另一方面结合性能建模的表达可以提升Kernel性能的上限。

**实现**：

- **确定切分轴的优先级**：首先需要基于API来确定切分轴的`优先级`，顺序如下：

  - 父轴优先级高于子轴（功能性硬约束）；
  - 规约化类轴高于非规约化类轴（启发式规则）；
  - 广播轴高于非广播轴（启发式规则）；
  - 非最内轴高于最内轴（启发式规则）；
  - 搬运API的尾轴具有同等优先级（启发式规则）。

- **分核与核内切分**：其次分成两个部分进行切分，包括`核内Tiling`和`分核Tiling`。以`UB优先贪心算法`为例（适合API头开销较大的场景）：

  - **核内Tiling**：按照轴排序的逆序依次遍历，优先将变量调整至最大值，判断是否符合硬件约束条件。若不满足，则通过二分法调整该变量，直到符合硬件约束条件为止，随后调整下一个核内Tiling变量，直至所有变量均满足硬件约束条件。例如，s1tt2、s1tt、s1t、s2t是Tiling相关轴，其切分流程如下：

    <p align="center">
      <img src="https://img2024.cnblogs.com/blog/3599704/202602/3599704-20260212154137186-1202869316.png" alt="核内Tiling切分流程" width="1000"  loading="lazy" decoding="async"/>
    </p>

  - **分核Tiling**：识别与多核相关的变量，按从大到小的顺序遍历这些变量，找到核数占用更大的记录，若超出物理核数则返回。如下图所示，bngs1T是多核切分轴，其切分流程如下。选择策略为：当核数占用不同时，优先选择占用核数更大的记录。根据上述策略，最终选定的占用核数为47。

    <p align="center">
      <img src="https://img2024.cnblogs.com/blog/3599704/202602/3599704-20260212154150988-1971581908.png" alt="分核Tiling切分流程" width="1000"  loading="lazy" decoding="async"/>
    </p>


    具体流程为：
    优先遍历s2t，调至最大值1024，符合硬件约束条件；
    然后遍历s1t，调至最大值256，符合硬件约束条件；
    再然后依次调整下个变量s1tt>s1tt2，直到所有变量均满足硬件约束条件。

- **其他算法**：
  - `UB占用核数占用权衡算法`：适合需要精细控制核数占用及UB占用的场景（有预设值，可根据性能公式动态调整核数）。
  - `对称切分`：适合尾轴转置场景，需要保证作为搬运类API的多个尾轴可以按照同等优先级切分。

### 5.3.3 PGO（Profile-Guided Optimization）

PGO是一种基于运行时性能反馈的优化技术，通过生成多个候选Tiling参数并在运行时根据实际性能选择最优方案，进一步提升求解质量。

**核心思想**

传统求解器基于静态性能模型生成单一解，但由于模型误差和硬件复杂性，静态解可能并非最优。PGO通过：

1. **编译时**：生成多个候选Tiling参数组合
2. **运行时**：根据实际执行性能选择最优方案
3. **反馈**：可选地将性能数据反馈回编译器

**候选解生成策略**

ATT采用指数步长递归生成候选解：

```cpp
// 伪代码
void GenerateCandidates(index, current_solution, candidates, step_max) {
    if (index >= num_variables) {
        if (SatisfyConstraints(current_solution)) {
            candidates.push_back(current_solution);
        }
        return;
    }

    variable = variables[index];
    min_value = variable.align;
    max_value = UpperBound(variable);

    // 指数步长遍历：1, 2, 4, 8, ..., step_max, step_max, ...
    step = variable.align;
    while (variable.value < max_value) {
        if (step <= step_max && variable.value < step_max) {
            variable.value = step;
        } else {
            variable.value += step;
        }
        variable.value = min(variable.value, max_value);

        if (SatisfyConstraints(variable)) {
            current_solution[index] = variable.value;
            GenerateCandidates(index + 1, current_solution, candidates, step_max);
        }

        step = min(step_max, step * 2);
    }
}
```

**档位离散化**

为控制搜索空间大小，核数采用档位离散化策略：

| 核数范围 | 档位间隔 | 说明                     |
| -------- | -------- | ------------------------ |
| [1, 4]   | 1        | 精细调节，避免多核开销   |
| (4, 16)  | 2        | 适度调节                 |
| [16, ∞)  | 4        | 粗粒度调节，减少搜索空间 |

**多模板选择**

运行时通过性能比较选择最优模板：

```python
for each candidate in candidates:
    kernel_time = ExecuteWithTiling(candidate)
    RecordPerformance(candidate, kernel_time)

SelectBestCandidate()
```

**优势与局限**

| 优势             | 局限           |
| ---------------- | -------------- |
| 克服静态模型误差 | 运行时开销增加 |
| 适应不同硬件特性 | 候选解数量受限 |
| 支持性能反馈学习 | 编译时间增加   |

---

## 6. 模型收益与使能

### 6.1 性能收益

- **整网收益**：Kernel耗时降低约15%
- **局部收益**：hc_post融合Kernel耗时降低4倍

### 6.2 使能方式

前提条件

- GCC版本：要求9.5.0以上，建议使用9.5.0版本；
- CMake版本：要求3.20.0版本以上，建议使用3.20.0版本；

```python
# 在torch脚本开头添加
import inductor_npu_ext
```

### 6.3 开源信息

源码已开源：https://gitcode.com/cann/ge/tree/master/compiler/graph/optimize/autofuse
