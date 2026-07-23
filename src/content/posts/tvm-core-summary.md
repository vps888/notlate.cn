---
title: "【纯干货】TVM 演进历程核心总结（一）"
description: "声明：本文纯手写 1. 第一代：AutoTVM 一种端到端的自动优化编译器 资料 2018 TVM: An Automated End to End Optimizing Compiler for Deep Learning (https://arxiv.org/abs/1802.04799)…"
slug: "tvm-core-summary"
legacyId: 19792168
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19792168"
pubDate: 2026-03-29
updatedDate: 2026-05-14
category: "AI 编译器"
tags: ["AI 编译器","TVM"]
featured: false
---

> 声明：本文纯手写

## 1. 第一代：AutoTVM - 一种端到端的自动优化编译器

**资料**

[2018 TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)

[2018 Learning to Optimize Tensor Programs](https://arxiv.org/pdf/1805.08166)

### 1.1 背景

1. 传统机器学习框架严重需要依赖高性能算子库（比如cuDNN），这种库往往只能针对特定硬件，因此框架需要付出巨大努力才能支持各种硬件。

2. 还有个弊端就是这些算子库往往是滞后于算子演进的。

### 1.2 方案

#### 高级图优化

TVM引入Tensor Expression（TE）中间表达，用于描述算子语义和计算逻辑，与硬件无关。基于TE可以做：

1. 承接上层框架多样的算子
2. 图优化：算子融合、常量折叠、静态内存规划、数据布局转换等
3. 向下转成有限的硬件指令抽象。

#### 自动优化框架

TVM支持自动生成算子，并能够枚举所有可能的算子实现方案（搜索空间），然后**TVM提了一种学习驱动的自动优化框架**：

1. 引入**机器学习模型**，输入一种算子实现方案（Low-level program），预测其执行耗时，以指导在搜索空间中进行寻优。
2. 使用**可迁移的表示（transferable representations）**，能够加速其他算子的泛化搜索（迁移学习，transfer learning）。

### 1.3 核心思路

#### 高级图优化

1. 算子融合：单一映射类（Elementwise、Transpose等）、Reduction类、复杂算子后融合Elementwise类，其余的归为不可融合类。
2. 数据布局转换：为每个算子选择最优Layout，在producer和consumer之间插入Layout Transform。
3. Tensorization：把硬件指令与Schedule解耦合，抽象出一层Tensor指令，使TVM可扩展不同硬件。

#### 计算语义与执行策略分离

1. **索引表达式（Index Expression）**：用索引表达式表达算子计算语义（不同的算子表达式不同）。表达式可以保留底层实现细节，比如循环顺序、内存层次和并行化方法等。
2. **变换（Schedule/Transformation）**：同一个索引表达式，经过不同的变换（执行策略），可以生成不同的Kernel代码，即一种具体的算子实现（Low-level  program）。

#### 可学习的搜索问题

通过上述思想，TVM将问题形式化为可学习的搜索问题。这虽然与传统的超参优化问题（HPO）类似，但相比HPO却有3个**重大优势**：

1. 实验成本低：机器学习模型训练推理快，可以在线收集大量真实数据；
2. 结构化信息：生成的代码结构化信息强，有限的IR、AST、嵌套循环、内存访问模式等的组合。
3. 具备可迁移性：大量的相似任务，不同的Shape、Layout、Batch等等，可以通过迁移学习降低搜索成本。

**如何保证搜索到的结果能够和手搓的算子性能媲美？**

1. **搜索空间要大**，能够覆盖手搓算子水平的实现方案，则从理论上一定能匹配手搓版本。
2. **搜索效率要高**，否则根本无法实用。

那搜索空间如何设计呢？先来对比一下主流的设计方法：

1. **多面体模型（Polyhedral Model）**：用整数线性约束描述循环（迭代）域，优点是理论完备，表达能力强；但缺点是搜索空间指数级且不规则，难以参数化，搜索不友好，优化困难。
2. **Halide**：用变换原语（Schedule Primitives）显示表达变换，比如split、reorder、tile、bind等。将搜索空间变成结构化和可参数化，非常适合自动搜索，且更贴近硬件执行模型。缺点是完备性差，且依赖人工设计原语。

TVM最终选择了Halide的路子，因为其更关注的是能否在合理的时间内找到一个足够好的解。

### 1.4 具体实现

#### 搜索空间设计

1. 每根轴进行多级Tiling（Multi-level tiling on earch loop axis）
2. 循环顺序（Loop ordering）
3. 多级缓存（Shared Memory caching）
4. 循环展开（Unrolling）
5. 向量化（Vectorization）

#### 模型设计

1. 基于人工提取特征的传统机器学习模型
   * 模型：XGBoost
   * 特征：**循环结构信息**（比如内存访问数量、数据复用率等）和**优化标记**（比如向量化、循环展开、线程绑定等）。
   * 缺点：依赖特征工程，泛化能力有限。

2. 基于表示学习的神经网络模型
   * 模型：TreeGRU
   * 不需要手动设计特征，直接输入Low-level program。
   * 缺点：训练成本高，推理慢。

#### 目标函数

对于搜索空间中的每个Low-level program，不需要精准预测执行耗时，只需要能够按照耗时长短正确排序即可（这类似推荐算法），于是Rank loss function为：

$$
\sum_{i,j}{\log (1 + e ^ {-sign(c_{i} - c_{j}) · (\hat{f}(x_i) - \hat{f}(x_j)) })}
$$
其中：$x_i$表示第$i$个program， $c_i$表示第$i$个program的实际耗时，$\hat{f}(x_i)$表示第$i$个program的预测耗时。

解释：当$c_i < c_j$，即$x_i$应该排的靠前，此时$-sign(c_{i} - c_{j})=1$，因此$\hat{f}(x_i)$比$\hat{f}(x_j)$小的越多，整体损失函数就越小。

#### 搜索算法

1. **候选生成**（并行）

* **目的**：生成一批候选schedules，记为$Q$。
* **方法**：并行执行模拟退火（Simulated Annealing，SA）算法。假设N个线程，则每个线程随机或启发式初始化一个状态（schedule），经过一步变换（比如改变循环顺序、改变切分大小等）生成邻居候选状态（candidate schedule）。使用代价模型$\hat{f}(x)$计算能量，根据**接受概率**（新状态耗时短直接接受，否则按照概率接受）选择是否更新状态，根据**退火策略**（线性、指数或自适应等）更新温度$T$。重复若干步（通常是设置最大步数或温度参数$T$降到阈值）。汇总N个线程各自产生的1个或多个局部最优解。
* **核心思想**：用代价模型引导搜索方向。

2. **贪心次模最优化（Greedy Submodular Optimization）**

* **目的**：从 $Q$ 中挑选一部分schedule作为下一批上板实测的候选集。

* **方法**：使用**次模贪心策略**（每次从$Q$中选择一个使得当前集合$S$的公式2最大的一个schedule）选择$(1 - ε)b$个候选，然后再**随机**选择$εb$个候选。目的是既要保证性能潜力（exploitation），又要保证多样性（exploration）。
  $$
  L(S) = - \sum_{s} \hat{f(g(e, s))} + \alpha \sum_{j=1}^{m}|U_s\{{s_j}\}|
  $$
  公式2看起来很费劲，改一个好理解的版本：
  $$
  L(S) = - \sum_{s} CostTime(s) + \alpha \ Diversity(S)
  $$
  也就是公式3由两部分组成：耗时最小+多样性最大，即最大化这个目标（多样性简单理解就是每个Schedule重复的特征越少越好）。

* **核心思想**：兼顾探索与利用

3. **硬件实测**

将上一步得到的集合$S$上板测量真实性能，加入到历史数据集合$D$中。用于修正代价模型预估偏差。

4. **更新代价模型**

用更新后的真实数据集$D$更新代价模型

5. **结束条件：搜索次数达到阈值**

6. **输出实测性能最优的schedule**

### 1.5 迁移学习

#### 背景

每个算子单独调优成本非常高，而同一个深度学习模型会包含很多结构相似的算子，所以要想办法利用历史数据，加速新算子的优化。

#### 思路

**可迁移表示（Transferable Representation）**，要做迁移学习，必须要找到一个跨算子不变（invariant）的表示。

TVM选择用Low-level program （AST）表示，一个矩阵乘法的AST表示如下：

```python
for y in range(8):
  for x in range(8):
    C[y][x] = 0
    for k in range(8):
      C[y][x] += A[k][y] * B[k][x]
```

#### 实现

如何把AST变成可学习表示？针对之前的两种类型模型分别有两条路线。

1. **GBT：Context Relation Features**

（1）提取上下文特征（context feature）

每一层循环提取多个特征组成一个特征向量（feature vector），比如循环长度、内存访问量、数据复用率等，最终表示成一个矩阵$Z_{k,i}$。

* $k$：第$k$层循环
* $i$：第$i$个特征

（2）构造"关系特征"（关键创新）

论文中的表达式比较复杂，简单说就是：寻找两个特征经过分桶聚合之后的关系，其实是一个二阶交叉特征。举例：

> | loop | loop length | memory touched |
> | ---- | ----------- | -------------- |
> | L1   | 64          | 1024           |
> | L2   | 8           | 128            |
> | L3   | 4           | 32             |
>
> 这种逐层特征不容易泛化，因为其他算子可能是2层循环，也可以是5层循环，何况顺序还可能不同，模型很难学习到那种结构更好。
>
> **核心思路**：不是看某一层，而是看特征之间的关系（分桶聚合交叉特征）
>
> 比如：小 loop + 小 memory touched 是不是效果好？举例介绍下如何分桶聚合特征。
>
> Step1：选一个条件，比如 memory torched < 阈值=200 的 loop，得到L2和L3。
>
> Step2：再找loop length最大值是多少。 得到L2的length最大=8。
>
> 这就得到了一组关系特征：当Memory小的时候，loop length有多大。再组合上实际耗时，即可让模型学习到：
>
> * 小内存 + 大循环 --> 耗时短 --> 学习到：数据复用高 
> * 大内存 + 小循环 --> 耗时长 --> 学习到：cache miss大 

2. **TreeGRU：Context Encoded TreeGRU**

（1）和GBT第一步一样，提取每一层循环的上下文特征向量，记为$h \in \mathbb{R}^d$

（2）loop embedding：$out_i = softmax( W^T·h )_i · h$，其中$W \in \mathbb{R}^{d \times m}$，$softmax(x) \in \mathbb{R}^{m}$，结果$out \in \mathbb{R}^{m \times d}$。

（3）final embedding：把所有的循环按照上面两步得到多个$out^{(l)} \in \mathbb{R}^{m \times d}$，将他们进行元素求和，再按照$m$维度聚合（sum、mean等），得到
$$
final_{embedding} = ReduceSum(OUT, axis=0), \ f \in \mathbb{R}^d
$$

3. **迁移学习的最终形式**

$$
\hat{f}(x) = \hat{f}_{global}(x) + \hat{f}_{local}(x)
$$

其中：

* global部分，用历史数据训练，提供初始预测能力，在冷启动场景提供关键作用；
* local部分，在线实时训练，在调优阶段被选中的program，负责精细拟合。

4. **整体流程总结**

* 根据历史数据训练出global模型
* 遇到新算子时，用global模型预测，指导初始搜索
* 收集新的实测数据
* 训练local模型
* 使用global+local进行更准确的预测。

### 1.6 总结

SA算法是搜索策略，local/global模型近似$f(x)$。
