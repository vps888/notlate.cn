---
title: "【纯干货】TVM 演进历程核心总结（三）"
description: "声明：本文纯手写 3. 第三代：MetaSchedule 基于概率编程的全自动框架 资料： Tensor Program Optimization with Probabilistic Programs (https://arxiv.org/abs/2205.13603) 3.1 背景 在Te…"
slug: "tvm-core-summary-3"
legacyId: 19822217
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19822217"
pubDate: 2026-04-05
updatedDate: 2026-05-14
category: "AI 编译器"
tags: ["AI 编译器","TVM"]
featured: false
---

> 声明：本文纯手写

## 3. 第三代：MetaSchedule - 基于概率编程的全自动框架

资料：[Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603)

### 3.1 背景

在Tensor Program优化场景，比如给定初始program $e_0$，一个典型的优化框架通常分成两部分：

* 根据预定义变换规则$S(e_0)$生成一堆计算语义等价的候选集
* 从这些候选集中选出最优program $e^*$

存在的问题：

* 已有方法的调度变换规则是确定性的，且强依赖专家定义。
* 搜索空间需要额外构建机制，与变换规则割裂。
* 自动调优方法的搜索成本高。

### 3.2 方案

面对上述问题，TVM团队提出了一种领域特定概率语言，把变换规则与搜索空间构建统一到了一个数学框架内。

#### 3.2.1 构建随机搜索空间

传统方法通常预定义固定的变换规则，通过枚举等方法组合应用这些规则可以生成丰富的搜索空间。然而这些规则与算子类型和硬件强相关，因此灵活性不够好。

作者的解决方案是用概率化方式构建搜索空间把规则定义和搜索空间构建统一起来。具体方案是：

**1. 由固定变换 --> 参数化变换**

* 每一步变换$t_i$通过参数化控制，具体取值如Split、Parallelize、Vectorize等；
* 对初始$e_0$ 应用参数化变换序列$𝜏$，最终得到$e_n$

**2. 由枚举 --> 随机变量**

* 具体变换的参数由随机变量$\theta$决定，表示为Sample-Tile、Sample-Compute-Location等。以Sample-Tile为例，表示切分某个轴，那么切分哪个循环？切成几块？每块切多大？都建模为随机变量。比如$Sample-Tile(i, parts=2)$就是对第$i$个循环切分成$2$块，每块大小再由随机变量$\theta_0、\theta_1$决定。
* 每个变换因此是随机的，符合一定概率分布的。
* 搜索空间自然可以表示完备的变换序列。

**3. 依赖结构状态**

* 用随机变量生成变换序列时，每个变量的可选值是依赖前面采样结果（当前program的状态）的，这样能保证搜索空间的结构合理性和高效性。
* 捕获长序列结构算术依赖，不是简单地枚举超参组合。

举例：

变换序列：$𝜏 = [t_1, t_2, t_3, ...]$

| 变换 (t_i) | 随机变量 θ     | 可能取值               |
| ---------- | -------------- | ---------------------- |
| $t_1$      | $θ_{loop}$     | $i, j$                 |
|            | $θ_{parts}$    | $1, 2, 4, ...$         |
|            | $θ_0, θ_1$     | …                      |
|            | $θ_{relu}$     | $i_{block}, j_{block}$ |
| $t_2$      | $θ_{parallel}$ | $0,1,2, …$             |
|            | $θ_{vec}$      | $128,256, …$           |
| $t_3$      | …              | …                      |

**总结**：通过定义随机变量的概率化方式，把规则定义和参数填充统一起来，把搜索空间构造定义为概率分布拟合。

#### 3.2.2 模块化搜索空间组合

传统的搜索空间构造方法，构造出的单个program通常是一条很长变换序列，不仅理解困难而且没法复用，尤其是不同算子之间无法共享优化策略。

MetaSchedule引入了Transformation module，定义了变换原语。每个module定义为原子的随机变换或是由程序分析、采样和更小的变换组合而成。这种设计可以让变换模块可复用、可组合，非常灵活。

以优化Dense + Relu 为例，对比两种方法：

**1. 传统方法的变换序列：**

```
t1: Split i → θ0, θ1
t2: Split j → θ2, θ3
t3: Parallelize i0
t4: Vectorize j1
t5: Fuse ReLU → θ_relu
```

问题：

* 变换步骤长且复杂

* 不易理解每个 θ 的作用

* 如果想复用这些变换到 conv2d 或 conv3d，需要重新分析并写新的序列
* 每次搜索空间都是完整的笛卡尔积，结构依赖不直观

**2. 模块化方法：**

| 模块名                 | 功能                                        |
| ---------------------- | ------------------------------------------- |
| Multi-Level-Tiling     | 分析循环 → 随机切分 → 重排序（5 级 tiling） |
| Auto-Inline            | 元素级操作自动内联，提高带宽效率            |
| Cross-Thread Reduction | 跨线程 reduction 优化                       |
| Use-Tensor-Core        | 针对硬件 TensorCore 优化                    |

组合方式：

```python
for location in program:
    m ∼ Sample([Multi-Level-Tiling, Auto-Inline, Cross-Thread Reduction])
    program ← m(location, program)
```

优势：

* 每个模块可复用
* 每个模块内部封装随机变量（参数），且自动管理依赖关系
* 容易扩展
* 开发者只需要理解模块功能，无需关注每个模块的参数细节

> **与其他Tensor Program优化方法对比**
>
> | 传统方法                                                     | MetaSchedule 实现方式                                        |
> | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | 基于确定性DSL的手动优化方法（Halide、早期TVM、Tiramisu、TACO） | 不使用随机变量时，MetaSchedule 等价于普通 DSL，支持手动优化  |
> | 基于模块的自动调优方法，开发者预定义搜索空间（AutoTVM）      | 在 MetaSchedule 中，可提前定义所有随机变量，搜索空间固定，不依赖程序状态 |
> | 自动调度方法Auto-scheduling（Ansor）                         | 通过概率变换模块生成搜索空间，实现同样的可编程性和功能       |
>
> **概括**：MetaSchedule 通过概率变换模块和模块化搜索空间，将现有 DSL、模板搜索和自动调度方法统一在一个可扩展、可复用的框架中。

#### 3.2.3 学习驱动的寻优框架

MetaSchedule把搜索定义寻找最优变换序列的问题：假设初始程序为$e_0$，从所有变换序列空间中，找到最优的变换序列$\tau^*$，使得到的程序性能最好。

##### 1. 目标函数

$$
\tau^* = \arg\max_\tau P(\tau | e_0) \propto e^{-f(g(e_0, \tau))} \cdot P(\tau)
$$

其中：

* 变换序列$\tau$来自程序指定的先验分布$P(\tau)$

* 经过变换的程序是$g(e_0, \tau)$

* $f(·)$表示程序实际性能
* 优化后程序的后验概率为：$P(\tau \mid e_0) \propto e^{-f(g(e_0, \tau))} \cdot P(\tau)$，含义是：性能越好，该序列被选中的概率越大

即通过**最大后验估计（MAP）**的方法找到最优变换序列。

##### 2. 执行追踪

为了方便领域专家使用变换模块（Transformation Modules）表达优化知识，TVM团队把MetaSchedule嵌入到了Python中，于是引入 **执行追踪（tracing）**来降低python重复执行的开销。

该模块只**记录采样和变换操作**，忽略 Python 的控制流等其他语法。

另外trace结果可以重复执行。

以优化矩阵乘为例介绍，假设原始程序为：

```python
C[i, j] += A[i, k] * B[k, j]
```

基于变换模块的调度（schedule）代码：

```python
def schedule(sch, M):
    i, j, k = sch.get_loops("C")

    # 控制流（不会进入 trace）
    if M > 1024:
        ti = sch.sample_perfect_tile(i, n=2)  # 随机采样
    else:
        ti = [16, 1]  # 固定值

    # 变换（会进入 trace）
    i0, i1 = sch.split(i, factors=ti)

    # 再来一个采样
    tj = sch.sample_perfect_tile(j, n=2)
    j0, j1 = sch.split(j, factors=tj)

    sch.reorder(i0, j0, i1, j1, k)
```

Trace结果：

```python
ti = sample_perfect_tile(loop=i, n=2)
split(loop=i, factors=ti)

tj = sample_perfect_tile(loop=j, n=2)
split(loop=j, factors=tj)

reorder(i0, j0, i1, j1, k)
```

那原schedule代码中的else固定值去哪了呢？答案是在sample_perfect_tile中的随机变量中，理论上可以包括所有场景。

那接下来如何使用和复用Trace？固定Trace，改变采样值：

比如第一次采样：

```
ti = 16
tj = 16
u  = 2
```

得到程序=tile(16, 16) + unroll(2)

第二次采样（同一个Trace）：

```
ti = 32
tj = 8
u  = 4
```

得到程序=tile(32, 8) + unroll(4)

**总结**：可以把Trace看作是Ansor中Sketch+Annotation，是一个可重复采样的程序模板。

##### 3. 端到端搜索

工作流程如下（对应 Figure 7）：

> 图片资源未包含在博客园数据库备份中：picture7

1. Trace初始程序变换，得到初始变换集合；
2. 基于Traces集合，迭代执行进化搜索算法来探索：
   - 使用 **进化搜索（evolutionary search）**，对 trace 中的随机变量进行突变（mutation），生成候选程序。
   - 使用Validator对生成的程序进行合法性校验。
   - 基于代价模型预测和带随机性的接受策略（类似annealed MH）决策接受还是拒绝。
   - 如果接受则真实测量性能
   - 更新代价模型

另外，这个框架也支持将进化搜索算法替换为 **贝叶斯优化或强化学习** 等选择策略。

##### 4. 代价模型

- 可以使用预训练的成本模型，默认是 **树提升（tree boosting）** 模型。
- 特征集沿用了之前张量优化工作的常用特征。

##### 5. Trace Validation有效性验证

随机变量选择可能超出硬件限制，或者引入非法的执行顺序，引入 **Validator** 对 trace 进行正确性验证。

### 3.3 总结

MetaSchedule 的本质是：把 Tensor Program 优化问题转化为一个概率程序上的搜索问题，再用学习模型高效近似这个搜索：

* 用概率编程抽象构造具有丰富搜索空间的程序
* 允许专家以模块化的方式将专业知识定义为变换模块，可以自有组合，而不需要关注具体细节。
* 通过端到端的学习驱动的框架来搜索最优程序。

MetaSchedule 的核心创新在于：

1. 用概率程序统一表示“变换规则 + 搜索空间”
2. 用 trace 作为优化决策的载体（结构 + 参数一体化）
3. 用学习模型（cost model）降低搜索成本
4. 用进化搜索在 trace 空间中高效探索

从而把：

* 规则设计问题转换为表达问题
* 搜索问题转化为学习驱动优化问题
