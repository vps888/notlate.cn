---
title: "移动广告工作总结-ADX"
description: "本文主要记录计算广告领域ADX的基础功能和工作中用到的策略。 询价 低价（市场保留价，Market Reserve Price） 公开底价 密封底价 多重低价 静态多重低价 动态多重低价 智能询价（Selective Call Out） 随着接入的DSP越来越多，媒体方的每次广告请求，对于AD…"
slug: "mobile-advertising-work-summary-adx"
legacyId: 18706031
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18706031"
pubDate: 2025-02-09
category: "推荐系统与机器学习"
tags: ["推荐系统与机器学习","计算广告"]
featured: false
---

本文主要记录计算广告领域ADX的基础功能和工作中用到的策略。

## 询价

### 低价（市场保留价，Market Reserve Price）

* 公开底价

* 密封底价

* 多重低价

* 静态多重低价

* 动态多重低价

### 智能询价（Selective Call Out）

随着接入的DSP越来越多，媒体方的每次广告请求，对于ADX来说要发送N倍的请求给DSP，成本非常高。在带宽和服务成本的约束下，如何获得最优的出价和参与度呢？

**我们的策略是：**基于ADX的用户信息、广告位信息、竞价信息等数据对各DSP的出价分布进行建模，实时预估各DSP对每次请求的出价能力，再结合各DSP耗时统计，动态调整请求的DSP组合。

更多文献：《[Selective Call Out and Real Time Bidding - 2010](http://wnzhang.net/share/rtb-papers/select-callout.pdf)》

### 动态底价

对于同一个请求，不同的DSP有不同的出价策略和预算，同一DSP在不同时间的出价策略和预算也不相同，所以动态低价的目标是提高ADX和媒体方的利润。论文《[An Empirical Study of Reserve Price Optimisation in Real-Time Bidding - 2014](http://wnzhang.net/share/rtb-papers/reserve-price.pdf)》提出了三种方法：

* 基于**贝叶斯推断**的底价估计算法，这种方法是数学理论最完备的，但是要求每次竞价的最高出价符合对数高斯分布。另外一点是这个算法优化的是单次收入最优，而非全局收入最优。

* 基于**均值统计**的底价估计算法，这种方法的思想就是把历史平均收入作为本次请求的底价。如果考虑时间序列的化，可以使用加权平均的方式，比如距离越远的收入权重越小。这种方法实现最为简单。

* 基于**经验的One-Shot底价**调整算法，这种方法虽然没有数学理论指导，但是是一种值得尝试的方法。一来实现也比较简单；二来这种方法也比较符合常规理解。主要思路是：当低价**小于**最高出价时，则**缓慢提高**低价；当低价**大于**最高出价时，则**迅速降低**低价。具体的调整幅度由不同的系数控制，完全基于上一次的最高出价和低价，计算下一次的底价。

上述论文可以参考文章《[拍卖与博弈：计算广告中的底价问题](https://embolismsoil.github.io/2019/07/21/%E6%8B%8D%E5%8D%96%E4%B8%8E%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%AE%97%E5%B9%BF%E5%91%8A%E4%B8%AD%E7%9A%84%E5%BA%95%E4%BB%B7%E9%97%AE%E9%A2%98/)》

**我们的策略：**在每个时间段内为各DSP设置不同的底价，统计各DSP在不同低价下的出价分布和参与度，挑选出使得各DSP出价和参与度联合最优的底价作为当前时间段各DSP的底价。对于同一个请求，不同的DSP低价不同。

更多文献：

《[Learning Algorithms for Second-Price Auctions with Reserve - 2016](https://www.jmlr.org/papers/volume17/14-499/14-499.pdf)》

《[Optimal Reserve Prices in Upstream Auctions: Empirical Application on Online Video Advertising - 2016](https://www.kdd.org/kdd2016/papers/files/rpp1142-alcobendas-lisbonaA.pdf)》

《[A Dynamic Pricing Model for Unifying Programmatic Guarantee and Real-Time Bidding in Display Advertising - 2014](http://arxiv.org/pdf/1405.5189.pdf)

### 超时熔断

对于SSP的每次广告请求，ADX会并行给多个DSP，而各家DSP的性能不同，也就是返回时间不同，因此ADX容易出现为了等某些性能差的DSP响应而导致本次SSP请求超时，错失一次填充机会。所以我们的目标是在出价和参与度的约束下，最小化端到端时延。

**我们的策略是**：对每次请求**实时预估**各DSP的**出价能力和响应速度**，在已有DSP返回达到综合最优的一定比例的条件下，及时熔断尚未返回的DSP请求，也就是不一定非要等到媒体最大时延才熔断，可以有条件的提前熔断。

## 竞价

### 竞价系数

* **提权**。通常ADX都是大型媒体构建的，同时也会构建自有DSP。相比三方DSP，自有DSP更容易受自己ADX的倾斜保护。通常采用对自有DSP出价提权的策略，即实时计算每一次请求的提价系数（**大于1**），然后乘以自有DSP原始出价的结果去竞价。
* **降权**。为了**打击**在技术和市场角度都具有绝对优势的DSP通过抬高出价但是低计费拿量的现象，可以使用降权策略，即周期性统计三方DSP的出价溢价率作为降价系数（**小于1**），然后乘以其原始出价的结果去竞价。

### 竞胜策略

通常采用**价高者得**的策略。

## 计费

### 计费策略（主要针对返回1个广告的场景）

* **一价计费**：出价即计费价。特点是：广告主出价**不**说实话；**非**占优策略激励兼容；**非**社会福利最大化。
* **二价计费**：次高价为计费价。特点是：广告主说实话；占优策略激励兼容；**非**社会福利最大化。
* **VCG计费**：因胜者而导致其他参与方的损失总和为计费价。特点是：广告主说实话；占优策略激励兼容；社会福利最大化。

自从2019年开始，越来越多的ADX采用一价计费。主要是因为现有的广告竞拍模式至少是两阶段的，即广告主先在DSP内部竞争；竞胜后以计费价作为出价再在ADX内进行竞争，这就很可能导致在DSP内出价高的广告主在ADX内竞争失败的情况出现。但是DSP内部通常还是二价计费。VCG计费策略比较复杂，不好向客户解释，所以实际使用的比较少。

更多文献：

《[Boosted Second Price Auctions: Revenue Optimization for Heterogeneous Bidders- 2021](https://dl.acm.org/doi/pdf/10.1145/3447548.3467454)》

《[Optimal Auctions through Deep Learning - 2019](http://proceedings.mlr.press/v97/duetting19a/duetting19a.pdf)》

《[Repeated Auctions with Budgets in Ad Exchanges: Approximations and Design - 2014](http://wnzhang.net/share/rtb-papers/repeat-auction.pdf)》

### 动态计费

通常DSP与广告主采用二价计费，因为这种方式简单易解释且为占优策略。但是ADX向DSP计费时，多数采用一价计费或者是基于二价的改进策略，并非完全二价计费，这样可以确保媒体和ADX的利益。试想某一ADX其市场竞争不激烈，最高价和次高价相差可能很大，若完全按照二价计费，则媒体和ADX利益严重受损；若抬高低价，则可能只有1家DSP出价，甚至没有DSP参与，对媒体和ADX来说也不能保障利益。综上即因为出价和计费价差距非常大，导致媒体和ADX的收益大大受损。

**我们的策略是：**在设置较低底价的同时，动态设置计费比例(0~1)。若DSP出价，则$计费价=\max(低价, 二价, 出价*计费比例)$。

## 流控

### 智能流控

在服务器负载较高和流量高峰期间，容易出现请求超时现象。智能流控的目的是在成本和超时率的约束下，最大化收入。

* **广告位优先**：实时统计各媒体广告位价值，对低价值的**广告位请求**进行流控，优先保证高价值广告位请求超时率不增。
* **DSP优先**：实时预估各DSP的价值，对低价值的**DSP请求**进行流控，优先保证高价值DSP的请求超时率不增。

## 建议阅读

1. [竞价广告的竞价策略的变迁（GSP-GSF-HD-BS-BC）](https://www.ichdata.com/changes-in-the-ad-exchange-bid-strategy.html)

   

## 欢迎各位留言交流探讨
