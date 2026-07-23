---
title: "竞价形势（Bid Landscape）预估总结"
description: "市场竞价预估 市场竞价预估的目标是预测每一次请求的市场价格，通常ADX和DSP都使用二价计费，因此市场价格也就是二价（计费价）。 准确地预测市场价格是一个重要的任务。对于ADX来说，每一次广告的请求的计费价都是可知的，因为可以轻易地用回归的方法拟合出。ADX可以用来预估每个DSP的竞价，从而选…"
slug: "bid-landscape-bid-landscape-summary"
legacyId: 18706034
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18706034"
pubDate: 2025-02-09
category: "计算广告"
tags: ["计算广告"]
featured: false
---

### 市场竞价预估

市场竞价预估的目标是预测每一次请求的市场价格，通常ADX和DSP都使用二价计费，因此市场价格也就是二价（计费价）。

准确地预测市场价格是一个重要的任务。对于ADX来说，每一次广告的请求的计费价都是可知的，因为可以轻易地用回归的方法拟合出。ADX可以用来预估每个DSP的竞价，从而选择要发送的DSP组合（比每次全部请求节约成本），还可以作为低价。但是对于DSP而言，只能获取到竞胜请求的计费价，对于竞价失败的请求，只能知道计费价比自己的竞价要高，这部分数据学术上称为删失数据，所以准确预测是十分困难的。

接下来总结关于模型的四个方法：

### 1. 只假设对数正态分布（2011 KDD）

参考文献：[Bid landscape forecasting in online ad exchange marketplace](http://wnzhang.net/share/rtb-papers/bid-lands.pdf)

本文方法先用Fast-Correlation Based Filtering方法挑选出重要特征，然后提出了星树（star tree）结构存储每条样本的特征和竞价信息。除最后一层叶节点外，每一层表示一个特征，每个节点存储出现次数大于一定阈值的特征值，其余未出现的或次数较少的用星号（star）节点代替。最后一层的叶节点存储本条特征链路上的竞价数据的均值和方差。

上述特征树创建好之后，就相当于生成了一份新的数据集，然后用GBDT方拟合历史竞价，学习每条路径的特征和竞价信息的关系。

在线推理时，根据请求特征从树中寻找到最匹配的叶节点，取出期均值和方差。然后假设竞价服从对数正态分布，代入取到的均值和方差就可以计算出竞价预估值。对于广告活动粒度的预估，论文假设其为每个广告任务的混合分布（FMM），也就是将广告活动包含的样本预估结果进行聚合即可。

### 2. 假设正态分布+机器学习（2015 KDD）

参考文献：[Predicting Winning Price in Real Time Bidding with Censored Data](http://wnzhang.net/share/rtb-papers/win-price-pred.pdf)

总体思路：线性回归模型拟合竞胜数据，删失回归模型拟合竞输数据，实时竞价时使用两者混合后的模型。

#### 2.1 线性回归模型拟合竞胜数据

表达式：$v_i ≈ \beta^T x_i + \varepsilon_i$，注意，本文中的winning price表示市场价（计费二价）。

其中，下标$i$表示样本序号；$v_i$是市场价；$\beta^T x_i$表示市场价的均值；$\varepsilon_i$假设服从均值为$0$，方差为$\sigma^2$的正态分布，用来学习删失信息。

#### 2.2 删失回归模型拟合竞输数据

用上述公式拟合出竞胜均价之后，再用下述公式表示单次竞胜的概率：
$$
P(v_i < b_i) = P(\varepsilon_i < (b_i - \beta^T x_i)) = \Phi(\frac{b_i - \beta^T x_i}{\sigma})
$$
其中$\Phi$表示累计概率分布，最终实际计算时，还是使用逻辑回归改写了这个公式：
$$
P(v_i < b_i) = \frac{1}{1 + e^{-\beta_{lr}^T x_i}}
$$

#### 2.3 损失函数

在竞胜数据上，损失函数目标是逼近市场价，最小化残差

在竞输数据上，公式表示竞胜概率，取负对数作为损失函数，也是最小此公式

#### 2.4 在线预测

预测时，使用混合模型，竞胜率 \* 竞胜均价 + 竞输率 \* 竞输均价

### 3. 假设多个分布+深度学习（2018 KDD）

参考文献：[Deep Censored Learning of the Winning Price in the Real Time Bidding](https://github.com/notlate-cn/tech-blogs/blob/main/papers/Bidding Landscape/2018-Deep Censored Learning of the Winning Price in the Real Time Bidding.pdf)

与第2条是同一个作者，本文主要变动点：把原来的线性回归替换成通用函数$g$，主要采用深度学习网络结构。

假设的分布用$f$和$F$表示，可以灵活替换成正态分布、对数正态分布和Gumbel分布。

### 4. 不假设任何分布，直接预估分布（2019 KDD）

参考文献：[Deep Landscape Forecasting for Real-time Bidding Advertising](https://arxiv.org/abs/1905.03028)

大致梳理一下论文思路，因为原论文中应该是有几处公式错误，可以参考文章《[论文复现Deep Landscape Forecasting for Real-time Bidding Advertising](https://blog.csdn.net/w55100/article/details/90401199)》，本文推导的公式与论文和CSDN文章均略有差异。

#### 4.1 生存分析法

本文基于生存分析法（KM）进行问题分析，可阅读文章进一步了解《[KM生存曲线的原理及画法](https://zhuanlan.zhihu.com/p/160186178)》。下面简述一下：

KM法是这样估计生存曲线：首先计算出活过一定时期的病人再活过下一时期的概率（即生存概率），然后将逐个生存概率相乘，即为相应时段的生存率。需要对观察对象一直持续保持关注，但是很难做到终生关注，中间可能会丢失。当观察到结束事件时（比如死亡）就停止记录，其中生存率常用$S$表示。

则对应到市场竞价预估中，**结束事件（比如死亡）**用**竞胜**表示。因为竞胜后，就不需要继续分析后续出价范围了。同样的，**生存事件**则用**竞输**表示。所以生存率$S = \prod_{每个周期}p_{竞输}$

#### 4.2 在连续空间上表示竞胜率和竞输率

假设市场价（计费二价）$z$的分布概率密度函数为$p(z)$，累积概率密度函数为$P(z)$，出价为$b$时：

* 竞胜率表示为：$W(b) = P(z < b) = \int_0^b p(z) dz$，含义是：当DSP的出价大于市场价时才能竞胜。

* 竞输率表示为：$S(b) = P(z ≥ b) = 1 - W(b) = 1 - \int_0^b p(z) dz$ 

上述表示好理解，但是没法计算，接下来就把市场价离散化，重新表示。

#### 4.3 在离散空间上表示竞胜率和竞输率

把出价离散化，可以用计费的最小单位（分）表示，例如：$0 = b_0 < b_1 < b_2 < ... < b_{l-1} < b_l$，那么相邻两个价格组成的区间记为$V_0=[b_0, b_1), V_1=[b_1, b_2), ... , V_{l-1}=[v_{l-1}, v), V_l=[b_l, b_{l+1})$，则：

* 新增定义 市场价$z$恰好落到价格区间$V_l$的概率为：$p_l = P(z \in V_l)$

* 竞胜率重新表示为：$W(b_l) = P(z < b) = \sum_{i=0}^{l-1}P(z \in V_i)$
* 竞输率重新表示为：$S(b_l) = P(z ≥ b) = 1 - W(b_l) = 1 - \sum_{i=0}^{l-1}P(z \in V_i)$

综上可得：
$$
p_l = P(z \in V_l) = W(b_{l+1}) - W(b_l) = S(b_l) - S(b_{l + 1}) \tag{5}
$$

#### 4.4 引入RNN

此时就可以把离线的竞胜率和竞输率转换成模型，然后使用三元组$(x, b, z)$样本数据计算概率分布$p(z)$。其中$x$是输入特征，$b$为实际出价，$z$为市场价。当本次竞价获胜时，$z$为计费价；否则$z$为0（因为竞输时，DSP拿不到计费价）。

但是目前$p(z)$还是不好用模型结构表示，所以作者巧妙的构造了辅助变量来解决这个问题。

定义：**在已知出价为$b_{l-1}$竞输的条件下，出价为$b_l$时恰好获胜的概率**为$h_l$，则：
$$
h_l = P(z \in V_{l-1} | z ≥ b_{l-1}) \xlongequal{贝叶斯公式} \frac{P(z \in V_{l-1}, z ≥ b_{l-1})}{P(z ≥ b_{l-1})} = \frac{P(z \in V_{l-1})}{P(z ≥ b_{l-1})} = \frac{p_{l-1}}{S(b_{l-1})} \tag{6}
$$
其中$z ≥ b_{l-1}$包含范围$z \in V_{l-1}=[b_{l-1}, b_l)$，其交集为$z \in V_{l-1}$。

由新的辅助变量可得：计算$b_l$出价的恰好获胜概率只需要出价为$b_{l-1}$的竞输率和市场价格正好落在$V_{l-1}$的概率。所以引入RNN模型（本文使用的是LSTM结构），用$f_\theta$表示，则公式$(6)$可改写为：
$$
h_l^i = P(z \in V_{l-1} | z ≥ b_{l-1}, \pmb{x^i}; \theta) = f_\theta (\pmb{x^i}, b_l | \pmb{r_{l-1}})   \tag{7}
$$
公式$(7)$的含义就是：在上一个出价区间竞输（$\pmb{r_{l-1}}$）的条件下，本次出价$b_l$竞胜的概率，所以：
$$
\pmb{r_{l-1}} = 1 - h_{l-1}^i \tag{7-1 本文作者补充}
$$
基于公式$(6)$和$(7)$，重写竞输率和竞胜率为公式$(8)$：
$$
\begin{aligned} 
S(b_l | \pmb{x^i}; \theta) &= P(z ≥ b_l | \pmb{x^i}; \theta) \\
&= P(z \notin V_0, z \notin V_1, z \notin V_2, ... , z \notin V_{l-1} | \pmb{x^i}; \theta) \quad (8.1)\\
&= P(z \notin V_0 | \pmb{x^i}; \theta) * P(z \notin V_1 | z \notin V_0,\pmb{x^i}; \theta) * P(z \notin V_2 | z \notin V_0, z \notin V_1,\pmb{x^i}; \theta) * ... * P(z \notin V_{l-1} | z \notin V_0,\pmb{x^i}, ... , z \notin V_{l-2}; \theta) \quad (8.2)\\
&= 1 * P(z \notin V_1 | z ≥ b_1, \pmb{x^i}; \theta) * P(z \notin V_2 | z ≥ b_2, \pmb{x^i}; \theta) \, * \, ... \,*\, P(z \notin V_{l-1} | z ≥ b_{l-1}, \pmb{x^i}; \theta) \quad (8.3) \\
&= \prod_{k=1}^{l-1} P(z \notin V_k | z ≥ b_k, \pmb{x^i}; \theta) \quad (8.4) \\
&= \prod_{k=1}^{l-1} \Big(\, 1 - P(z \in V_k | z ≥ b_k, \pmb{x^i}; \theta) \,\Big) \quad (8.5) \\
&= \prod_{k=1}^{l-1} (\, 1 - h_{k+1}^i \,) \quad (8.6) \\
&= \prod_{k=2}^{l} (\, 1 - h_{k}^i \,) \quad (8.7) \\
&= \prod_{k=1}^{l} (\, 1 - h_{k}^i \,) \quad (8.8) \\
\end{aligned}
$$

上式$(8.7) \rightarrow (8.8)$解释：当$k=1$时，$h_1^i = P(z \in V_0 | z ≥ b_0, \pmb{x^i}; \theta) = 1$。
$$
\begin{aligned} 
W(b_l^i| \pmb{x^i}; \theta) &= 1 - S(b_l^i | \pmb{x^i}; \theta) \\
&= 1 - \prod_{k=1}^{l} (\, 1 - h_k^i \,)  \quad (8.9)
\end{aligned}
$$
再由公式$(6)$得到，对于第$i$个样本来说，$z^i$正好落在区间$V_{l-1}$的概率为：
$$
p_{l-1}^i = h_l^i * S(b_{l-1}^i) = h_l^i \, \prod_{k=1}^{l-1} (\, 1 - h_{k}^i \,) \tag{9}
$$
#### 4.5 损失函数

本文定义的损失函数形式类似于第3篇文章，采用了两种损失函数加权和的方式。

第一种方式是用**市场价概率分布函数**拟合**竞胜数据**。当已知给定的样本都是竞胜时，则优化目标可定义为**最大化市场价**$z$**恰好落在$V_l$的概率，最好为$1$**。定义损失函数为负对数似然函数，则最小化以下公式即可：
$$
\begin{aligned}
L_1 &= -\log \, \bigg( \prod_{\pmb{x^i},z^i \in D_{win}} \, P(z^i \in V_l | \pmb{x^i}; \theta) \bigg) \\
&= -\log \, \bigg( \prod_{\pmb{x^i},z^i \in D_{win}} \, p_l^i \bigg) \\
&= -\log \, \bigg( \prod_{\pmb{x^i},z^i \in D_{win}} \, \Big( h_{l+1}^i \, \prod_{k=1}^l (\, 1 - h_{k}^i \,) \Big)\bigg) \\
&= - \bigg( \sum_{\pmb{x^i},z^i \in D_{win}} \Big( \log h_{l+1}^i + \log \prod_{k=1}^l (\, 1 - h_{k}^i) \Big) \bigg) \\
&= - \sum_{\pmb{x^i},z^i \in D_{win}} \Big( \log h_{l+1}^i + \sum_{k=1}^l \log (\, 1 - h_{k}^i) \Big) \\
&= - \sum_{\pmb{x^i},z^i \in D_{win}} \Big( \log h_{l+1}^i + \sum_{k:k≤l} \log (\, 1 - h_{k}^i) \Big) \quad (10)\\
\end{aligned}
$$
第二种方式是用市场价**累积概率分布函数**同时拟合**竞胜和竞输**数据。对于竞胜数据，我们希望$P(z^i < b_l^i | \pmb{x^i}; \theta) \rightarrow 1$，对于竞输数据，我们希望$P(z^i ≥ b_l^i | \pmb{x^i}; \theta) \rightarrow 1$。采用负对数似然函数定义损失函数如下：
$$
\begin{aligned}
L_{win} &= - \log \Big( \prod_{\pmb{x^i},b^i \in D_{win}} P(z < b_l^i | \pmb{x^i}; \theta) \Big) \\
&= - \log \Big( \prod_{\pmb{x^i},b^i \in D_{win}} W(b_l^i | \pmb{x^i}; \theta )  \Big) \\
&= - \log \bigg( \prod_{\pmb{x^i},b^i \in D_{win}} \Big( 1 - \prod_{k=1}^{l} (\, 1 - h_{k}^i \,) \Big) \bigg) \\
&= - \sum_{\pmb{x^i},b^i \in D_{win}} \log \Big( 1 - \prod_{k=1}^{l} (\, 1 - h_{k}^i \,) \Big)  \quad (11.1) \\
&= - \sum_{\pmb{x^i},b^i \in D_{win}} \log \Big( 1 - \prod_{k:k≤l} (\, 1 - h_{k}^i \,) \Big)  \quad (11.2) \\
\end{aligned}
$$

$$
\begin{aligned}
L_{lose} &= - \log \Big( \prod_{\pmb{x^i},b^i \in D_{lose}} P(z ≥ b_l^i | \pmb{x^i}; \theta) \Big) \\
&= - \log \Big( \prod_{\pmb{x^i},b^i \in D_{lose}} S(b_l^i | \pmb{x^i}; \theta )  \Big) \\
&= - \log \bigg( \prod_{\pmb{x^i},b^i \in D_{lose}} \Big( \prod_{k=1}^{l} (\, 1 - h_{k}^i \,) \Big) \bigg) \\
&= - \sum_{\pmb{x^i},b^i \in D_{lose}} \log \Big( \prod_{k=1}^{l} (\, 1 - h_{k}^i \,) \Big) \\
&= - \sum_{\pmb{x^i},b^i \in D_{lose}} \sum_{k=1}^{l} \log (\, 1 - h_{k}^i \,) \quad (12.1) \\
&= - \sum_{\pmb{x^i},b^i \in D_{lose}} \sum_{k:k≤l} \log (\, 1 - h_{k}^i \,)  \quad (12.2) \\
\end{aligned}
$$

因为这两个损失函数是通过竞胜还是竞输区分的，所以作者设计了一个指示函数：
$$
w^i=
\begin{cases}
1, \quad if \quad b^i > z^i, \\
0, \quad otherwise
\end{cases}
\tag{13}
$$
就可以把$L_{win}$和$L_{lose}$合并成：
$$
\begin{aligned}
L_2 &= L_{win} + L_{lose} \\
&= - \log \Big( \prod_{\pmb{x^i},b^i \in D_{win}} P(z < b_l^i | \pmb{x^i}; \theta) \Big) - \log \Big( \prod_{\pmb{x^i},b^i \in D_{lose}} P(z ≥ b_l^i | \pmb{x^i}; \theta) \Big)\\
&= - \log \bigg( \prod_{\pmb{x^i},b^i \in D} \Big( W(b_l^i | \pmb{x^i}; \theta ) \Big)^{w^i} \Big( 1 - W(b_l^i | \pmb{x^i}; \theta ) \Big)^{1 - w^i}  \bigg) \quad (14.1) \\
&= - \sum_{\pmb{x^i},b^i \in D} \bigg( w^i \, \log \Big( W(b_l^i | \pmb{x^i}; \theta ) \Big) + (1 - w^i) \log \Big( 1 - W(b_l^i | \pmb{x^i}; \theta ) \Big)  \bigg) \quad (14.2) \\
\end{aligned}
$$
最终的损失函数为：
$$
\underset{\theta}{argmin} \, \alpha L_1 + (1-\alpha)L_2 \tag{15}
$$
公式实在太多了，可累坏了。
