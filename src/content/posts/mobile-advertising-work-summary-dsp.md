---
title: "移动广告工作总结-DSP"
description: "本文主要记录计算广告领域DSP的基础功能和工作中用到的技术与策略简述。由于内容太多，具体方法总结会另写文章介绍。 合约广告（Guaranteed Delivery） 流量预测（Traffic Forecasting） 给定一组受众标签组合和一个ECPM阈值，预估将来某个时间段内符合这些受众标签…"
slug: "mobile-advertising-work-summary-dsp"
legacyId: 18706032
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18706032"
pubDate: 2025-02-09
category: "计算广告"
tags: ["计算广告"]
featured: false
---

本文主要记录计算广告领域DSP的基础功能和工作中用到的技术与策略简述。由于内容太多，具体方法总结会另写文章介绍。

## 合约广告（Guaranteed Delivery）

### 流量预测（Traffic Forecasting）

给定一组受众标签组合和一个ECPM阈值，预估将来某个时间段内符合这些受众标签组合的条件并且市场价格在该ECPM阈值以下的广告展示量

* **基于统计的方法**：统计历史数据，基于反向索引的方法拟合未来流量，参考自《计算广告》合约广告核心技术章节。
* **基于回归模型的方法**：选择简单易加工的特征，使用回归模型预测大盘流量，参考阅读《[Predicting Traffic of Online Advertising in Real-time Bidding Systems from Perspective of Demand-Side Platforms - 2016](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Predicting%20Traffic%20of%20Online%20Advertising%20in%20Real-time%20Bidding%20Systems%20from%20Perspective%20of%20Demand-Side%20Platforms.pdf)》

### 频次控制（Frequency Capping）

按照不同的分类维度（比如同一广告主、同一广告类目、同一广告素材等）控制广告出现的频次，即为了保障用户体验，也为了保障广告主的转化成本。

相关阅读：

《[Soft Frequency Capping for Improved Ad Click Prediction in Yahoo Gemini Native - 2019](https://github.com/notlate-cn/tech-blogs/blob/main/papers/Frequency%20Capping/2019-Soft%20Frequency%20Capping%20for%20Improved%20Ad%20Click%20Prediction%20in%20Yahoo%20Gemini%20Native.pdf)》

《[Frequency Capping in Online Advertising - 2011](https://theory.epfl.ch/moranfe/Publications/WADS2011.pdf)》

### 在线分配（Online Allocation）

主要参考自Hulu的技术博客《[第7期:经典广告流量匹配算法](https://mp.weixin.qq.com/s?__biz=MzA5NzQyNTcxMA==&mid=2656436030&idx=1&sn=8915a7e0158af593bacca9d3f13d3a86&scene=19#wechat_redirect)》，这篇文章介绍的非常清楚。通过对每一次广告展示进行实时在线决策，从而达到满足某些量的约束的前提下，优化广告产品整体收益的过程。一般将此问题简化为二部图匹配问题

* **DUAL**：最早由Yahoo提出，主要思路是将原优化问题转化为拉格朗日对偶问题后直接求解。因为二部图的**供给节点**（**用户**，亿级）和**需求节点**（**广告**，万级）的数量庞大，整体计算完存储量非常大。因此实际工程中往往采用坐标下降、梯度下降等方法**离线**计算并存储需求节点（数量相对少）的对偶变量；**在线**时，对某用户而言，检索出所有能投放给他的广告订单，实时计算出满足约束的参数，继而得到投放概率。此方法缺点是：**离线和在线都需要大量迭代，计算复杂度高**
* **HWM（High Water Mark）**：启发式算法，参考文献《[Ad Serving Using a Compact Allocation Plan - 2012](https://arxiv.org/abs/1203.3593)》。此方法缺点是：不能找到问题的最优解
* **SHALE**：结合了DUAL和HWM方法，兼具了性能和效果。参考文献《[SHALE: An Efficient Algorithm for Allocation of Guaranteed Display Advertising - 2012](https://arxiv.org/abs/1203.3619)》

### [投放控制（Pacing）](#)

在线分配只能优化单次流量的投放，而从整体流量投放来看，往往不是最优的方案。比如广告主预算消耗过快或过慢，用户广告体验单一等问题。投放控制主要有两个目标：

* 成本控制：控制广告主的平均（转化）成本不超过预期，或者是ROI（投资回报比）不超标
* 预算控制：控制广告主的预算消耗速度，既不要过早消耗完，也不能消耗过慢，更不能超标。

实现这个目标，通常只需要解决两个问题：

* 要不要参与竞价？通过参竞率来控制广告投放速度
* 出多少价？通过调整出价来控制竞胜率和投放速度，同时也可以把成本作为优化目标纳入考虑。

解决这些问题的方法，最经典的非PID方法莫属。建议先阅读文章**了解基础原理**《[PID控制算法原理（抛弃公式，从本质上真正理解PID控制）](https://zhuanlan.zhihu.com/p/39573490)》，然后再阅读Hulu技术博客**了解如何实战**《[第8期:广告流量匹配算法在Hulu/Disney Streaming平台的实战](https://mp.weixin.qq.com/s?__biz=MzA5NzQyNTcxMA==&mid=2656436132&idx=1&sn=0aedee0ea56550abbb867feeececa147&scene=19#wechat_redirect)》。

更多阅读：

《[Multiplicative Pacing Equilibria in Auction Markets - 2022](https://arxiv.org/abs/1706.07151)》

《[Smart Pacing for Effective Online Ad Campaign Optimization - 2015](https://arxiv.org/abs/1506.05851)》

《[**Budget Pacing for Targeted Online Advertisements at LinkedIn - 2014**](https://github.com/wzhe06/Ad-papers/blob/master/Budget%20Control/Budget%20Pacing%20for%20Targeted%20Online%20Advertisements%20at%20LinkedIn.pdf)》

《[Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising - 2013](https://arxiv.org/abs/1305.3011)》

## 竞价广告（Real Time Bidding）

### 特征工程（Feature Engineering）

#### 特征选择（Feature Selection）

在众多的特征中高效率的挑选出对目标区分能力最强的特征

* MCR：训练模型好之后，在测试阶段，每次shuffle一个特征，AUC下降越明显，说明特征越重要
* FSCD：在训练阶段，直接学习出每个特征的重要性，这是一篇来自阿里妈妈2021年的工作[FSCD-PreRank](https://zhuanlan.zhihu.com/p/375943741)

#### ID编码（ID Mapping）

ID类特征（比如用户ID）的特征取值上亿，即维度很高，若采用one-hot方式编码，模型空间占用会非常大，如何实现ID类特征的无损或高质量降维呢？

* CityHash
* Frequency Hash：为了尽量减少损失，高频特征值保持原值，低频特征值进行hash映射降维
* [Deep Hash Embedding - 2020](https://arxiv.org/abs/2010.10784)：来自谷歌。总的来说就是：用$K$个hash函数把高维ID映射成$K(1024)$维向量，这是固定的不可训练的常量，也不需要存储。然后经过多层全连接学习出低维（32维）表示，替代One-Hot编码中的需要超大参数存储的查表（look-up）操作。不过缺点是推理性能要比查表慢很多。可以参考知乎文章《[推荐系统里，你是怎么Embedding的](https://zhuanlan.zhihu.com/p/397600084)》。

### 召回（Retrieval）

因为排序模块性能较差，所以引入召回模块，作用是从物品库中筛选出最相关的万级或千级的物品，本模块需要极高的性能。

#### 规则召回

最常用的方法，效率高，可解释性强。通常使用多路规则并行召回，聚合多路结果为一个集合后，送入下游排序模块处理。

#### 定向召回

常用方法是倒排索引（Inverted Index），建议阅读文章《[广告索引（定向）的布尔表达式](https://www.cnblogs.com/chenny7/p/14765412.html)》，通过详细的绘图介绍的很清楚。

#### 向量召回

也就是Embedding技术，通过计算用户和广告之间的距离（向量内积）描述两者之间的相关性。基于模型生成Embedding的技术非常多，也必然是符合技术发展潮流的。

* 传统模型，比如FM、FFM等
* 深度模型，比如[Youtebe经典之作 - 2016](https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/45530.pdf)、[微软之作Item2Vec - 2017](https://arxiv.org/abs/1603.04259)、 [谷歌之作DNN双塔 - 2019](https://github.com/tangxyw/RecSysPapers/blob/main/Match/%5B2019%5D%5BGoogle%5D%20Sampling-Bias-Corrected%20Neural%20Modeling%20for%20Large%20Corpus%20Item%20Recommendations.pdf)、[阿里妈妈面向下一代的粗排排序系统COLD - 2020](https://arxiv.org/abs/2007.16122)等
* 用户行为序列模型，比如[GRU - 2018](https://arxiv.org/pdf/1706.03847.pdf)、[CNN - 2018](https://arxiv.org/abs/1809.07426)、[Transformer - 2018](https://arxiv.org/abs/1808.09781)、[DIEN - 2018](https://arxiv.org/abs/1809.03672v1)等。
* 用户多兴趣拆分模型，比如[MIND - 2019](https://arxiv.org/abs/1904.08030)等
* 知识图谱融合模型，比如[RippleNet - 2018](https://arxiv.org/pdf/1803.03467.pdf%C2%A0)、[KGAT - 2019](https://arxiv.org/abs/1905.07854)等
* 图神经网络模型，比如[GraphSAGE - 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)、[PinSage - 2018](https://arxiv.org/abs/1806.01973)等
* 多模态召回，必然的发展趋势

#### 向量索引

向量召回实际应用时通常在离线生成用户和物品的Embedding，把用户Embedding存储在Redis内存数据库中，物品Embedding存储在专用的向量数据库中（因为用户Embedding与大量的物品Embedding计算相似度非常耗时）。

在线推断时，根据用户ID找到用户Embedding；再从大量的物品Embedding中快速找出与之匹配topK的物品。那么如果快速检索呢？下述部分方法参考自文章《[谈快速检索embedding](https://zhuanlan.zhihu.com/p/421616703)》和《[广告深度学习计算：向量召回索引的演进以及工程实现](https://zhuanlan.zhihu.com/p/604748988)》

* **暴力检索**：逐个计算用户和每个物品的相似度，然后取topK

* **局部敏感哈希（LSH）**：让相邻的点落入同一个桶，这样在k近邻搜索时仅需在一个桶或相邻的几个桶内搜索。可以参考B站介绍视频《[召回层算法(3)-LSH_局部敏感哈希](https://www.bilibili.com/video/BV1Zy4y1B71R/?spm_id_from=333.999.0.0)》

* **树索引**：TDM的训练大致分为两步：树的学习和模型的学习《[Learning Tree-based Deep Model for Recommender Systems - 2018](https://arxiv.org/abs/1801.02294)》

  JTM对树的学习改进《[Joint Optimization of Tree-based Index and Deep Model for Recommender Systems - 2019](https://arxiv.org/abs/1902.07565)》

  OTM对模型的学习改进《[Learning Optimal Tree Models under Beam Search - 2020](https://arxiv.org/abs/2006.15408)》

  上述三篇文章的详细解读《[阿里妈妈深度树检索技术（TDM）及应用框架的探索实践](https://zhuanlan.zhihu.com/p/78488485)》

* **图索引**：HNSW《[Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation - 2022](https://arxiv.org/abs/2202.10226)》

* **多类目+多层次的图索引**

常用的向量数据库有：Milvus、Faiss、HnswLib等，其中Faiss检索工具比较常用，支持三种索引方法：

* **精确索引**：暴力检索

* **倒排快速索引（实际上是倒排+聚类）**：建立倒排索引，将数据库内的向量聚类。查询时先找到最相近的几个类，然后在这些类里做k近邻搜索。好处是查询时不再检索那些明显不在k近邻范围内的类，减少了计算量

* **乘积量化索引：**这种方法直接对所有向量做了量化（类似于神经网络中模型压缩的量化）。经过量化后原有的向量丢失，只剩下了量化后的近似向量

#### 建议阅读本人笔记《[小红书推荐系统公开课学习笔记01-召回](https://notlate.cn/p/07689bd69c9a0a6f/)》

### 粗排（Pre-ranking）

粗排的作用是继续筛选召回结果，输出百级物品给精排模块，因此同样需要高性能，但是本质上和精排模型是类似的。

#### 粗排发展

* **传统模型**：线性模型LR等，计算性能足够，但是模型过于简单，表达能力偏弱
* **深度模型**：依然是[DNN双塔]((https://github.com/tangxyw/RecSysPapers/blob/main/Match/%5B2019%5D%5BGoogle%5D%20Sampling-Bias-Corrected%20Neural%20Modeling%20for%20Large%20Corpus%20Item%20Recommendations.pdf))、**三塔**(为了提升性能，把统计特征单独成塔)等

#### 粗排优化

* **知识蒸馏**：是一种模型训练方式，分为**特征蒸馏**和**模型蒸馏**。特征蒸馏是Teacher和Student模型具有相同模型结构，但是Student模型使用更少更简单的特征，是为了通过Teacher模型学习到复杂特征有效信息的前提下，简化复杂特征，从而提高网络性能。模型蒸馏主要是两个模型使用相同的特征，但是Student模型的网络结构更加简单，也是为了提高性能。**通过知识蒸馏得到的粗排模型也保证了与精排模型一致的优化目标。**
* **特征裁剪**：阿里妈妈2020年提出的新一代粗排系统COLD（[论文](https://arxiv.org/abs/2007.16122)、[官方解读](https://zhuanlan.zhihu.com/p/186320100)），引入了SE block，能够学习到每个特征的重要性，从而可以实现特征裁剪，并配合网络剪枝和工程优化，可以实现精度和性能之间的权衡。

### 精排（Ranking）

精排是推荐领域最具有技术含量的模块，相应的文章也非常多。深度学习的模型发展路线图，建议阅读《[A Brief History of Recommender Systems - 2022](https://arxiv.org/pdf/2209.01860.pdf)》。下面简要梳理一下现有工作：

* 传统方法：协同过滤（CF）、矩阵分解（MF）、FM、FFM等

* 深度学习：[Wide&Deep - 2016](https://arxiv.org/abs/1606.07792)、[DeepFM - 2017](https://arxiv.org/abs/1703.04247)、[Deep&Cross - 2017](https://arxiv.org/abs/1708.05123)、[AutoInt - 2018](https://arxiv.org/abs/1810.11921)、[XDeepFM - 2018](https://arxiv.org/abs/1803.05170)、[FiBiNet - 2019](https://arxiv.org/abs/1905.09433)等

* 行为序列：[DIN - 2017](https://arxiv.org/abs/1706.06978)、[SIM - 2020](https://arxiv.org/pdf/2006.05639.pdf)等

* 多目标优化：

  [MMoE - 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)：使模型在不同相关程度的多任务目标上获得较好泛化能力；尽量少的参数以保证性能。但是存在极化问题，解决方案可阅读《[Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf)》。

  [ESMM - 2018](https://dl.acm.org/doi/pdf/10.1145/3209978.3210104)：阿里提出的一种简单高效实用的方法，利用CTCVR和CTR的监督信息来训练网络，隐式地学习CVR，可以参考解读文章《[阿里CVR预估模型之ESMM](https://zhuanlan.zhihu.com/p/57481330)》

  [PE-LTR - 2019](http://ofey.me/papers/Pareto.pdf)：一种基于帕累托有效算法解决推荐中多目标优化问题的方法，可以参考解读文章《[推荐系统的多目标优化(4)-PE-LTR](https://haiping.vip/2020/05/04/%E5%B8%95%E7%B4%AF%E6%89%98%E6%9C%80%E4%BC%98/)》

  [PLE - 2020](https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising/blob/master/5_Multi-task/2020%20(Tencent)%20(Recsys)%20%5BPLE%5D%20Progressive%20Layered%20Extraction%20(PLE)%20-%20A%20Novel%20Multi-Task%20Learning%20(MTL)%20Model%20for%20Personalized%20Recommendations.pdf)：多目标优化时容易出现跷跷板效应，本文优化多任务之间的共享机制和网络结构，以提升效果[PEPNet - 2023](https://arxiv.org/abs/2302.01115)：快手借鉴LHUC中的个性化参数配置，提出的一种大模型。PPNet输入UserID等特征实现用户个性化，EPNet输入场景ID等特征实现场景个性化。可以参考解读文章《[PEPNet：融合个性化先验信息的多场景多任务网络](https://zhuanlan.zhihu.com/p/611400673)》

* 长短期兴趣分离：

  《[Neural News Recommendation with Long- and Short-term User Representations - 2019](https://aclanthology.org/P19-1033.pdf)》

* 多模态融合：《[Image Matters: Visually Modeling User Behaviors Using Advanced Model Server - 2018](https://www.researchgate.net/profile/Xiaoqiang-Zhu-7/publication/328439173_Image_Matters_Visually_Modeling_User_Behaviors_Using_Advanced_Model_Server/links/5beba4ca299bf1124fd0f147/Image-Matters-Visually-Modeling-User-Behaviors-Using-Advanced-Model-Server.pdf)》

* 强化学习：

  《[Top-K Off-Policy Correction for a REINFORCE Recommender System - 2019](https://www.alexbeutel.com/papers/wsdm2019_reinforce_recs.pdf)》

  《[Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology - 2019](https://arxiv.org/abs/1905.12767)》

#### 建议阅读本人笔记《[小红书推荐系统公开课学习笔记02-排序](https://notlate.cn/?p=304)》、《[小红书推荐系统公开课学习笔记03-特征交叉](https://notlate.cn/p/16fca92af3f530cf/)》、《[小红书推荐系统公开课学习笔记04-行为序列](https://notlate.cn/p/773215a847ca8a36/)》

### 重排（Re-ranking）

推荐系统中排序常用的优化目标或损失函数定义方式有三种：Point Wise、Pair Wise和List Wise。

我们最常见的是Point Wise方式，也就是优化目标中只考虑单个样本，输出其打分然后排序，不需要考虑多个物品之间的顺序关系。

Pair Wise方式会考虑两个物品的顺序关系，比如优化目标是物品A排序要高于物品B等。这种方式在推荐领域使用非常广泛且很有效，比如BPR损失。

List Wise方式更关注整个列表中物品的顺序关系来优化模型。目前因为构造数据难、推理速度慢等原因使用场景比较少。这种方式做重排会有更好的效果，因为重排的目标就是通过调整列表中物品的顺序来实现全局最优。因为优化目标需要考虑输入的序列性，所以RNN或Transformer等序列模型是重排的首选。做法是：排序Top结果的物品有序，作为重排模型的输入，可以考虑在特征级别，融合当前物品上下文，也就是排序列表中其它物品的特征，来从列表整体评估效果。重排模型每个输入位置经过特征融合，再次输出预测得分，按照新预测的得分重新对物品排序，就完成了融合上下文信息，进行重新排序的目的。（参考自文章《[推荐系统技术演进趋势：从召回到排序再到重排](https://zhuanlan.zhihu.com/p/100019681)》）

更多阅读：

《[Personalized Re-ranking for Recommendation - 2019](https://www.researchgate.net/profile/Fei-Sun-41/publication/332439435_Personalized_Context-aware_Re-ranking_for_E-commerce_Recommender_Systems/links/6047567b4585154e8c87e02e/Personalized-Context-aware-Re-ranking-for-E-commerce-Recommender-Systems.pdf)》

《[Learning a Deep Listwise Context Model for Ranking Refinement - 2018](https://dl.acm.org/doi/pdf/10.1145/3209978.3209985)》

#### 建议阅读本人笔记《[小红书推荐系统公开课学习笔记05-重排](https://notlate.cn/p/eb73eb0901934624/)》

### 模型校准（Calibration）

对于推荐问题来说，只需要排对相对顺序就可以了。但是对于广告来说，是需要精确预估的，因为涉及计算出价。建议阅读文章《[模型预估（广告点击率/转化率预估等）校准算法资料](https://zhuanlan.zhihu.com/p/350744424)》和《[阿里妈妈展示广告预估校准技术演进之路](https://zhuanlan.zhihu.com/p/398235467)》。本文主要介绍几种常用的方法：

* Platt scaling - 1999，用LR拟合的方法校准，建议阅读文章《[通俗理解Platt scaling/Platt缩放/普拉特缩放](https://blog.csdn.net/qq_36158230/article/details/128590183)》

* Isotonic Regression - 2002，保序回归，又叫PAV算法，大致方法是先分桶，再把逆序的桶合并成均值桶，建议阅读文章《[使用 Isotonic Regression 校准分类器](http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/21/classifier-calibration-with-isotonic-regression)》

* [Facebook公式校准 - 2014](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BGBDT%2BLR%5D%20Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook%20(Facebook%202014).pdf)，一个很简单的校准公式，$q=\frac{p}{p+(1-p)/w}$，其中$p$是预测值，$w$是校准系数，$q$是校准后的值。在实际应用中，$q$和$p$都是已知的，所以可比较简单的通过检索的方式找到符合要求的校准系数供在线使用。建议阅读《[面向稀有事件的 Logistic Regression 模型校准](http://vividfree.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2015/12/15/model-calibration-for-logistic-regression-in-rare-events-data)》。
* Smoothed Isotonic Regression - 2020，保序回归平滑校准算法，结合了Binning、Isotonic Regression和线性Scaling方法。大致方法是先分桶，对逆序桶合并成小桶，然后使用scaling系数提拉。
* Bayes-SIR - 2020，引入贝叶斯平滑思想，对不置信的后验CTR做贝叶斯平滑，解决冷启动问题
* [PCCEM - 2020](https://github.com/tangxyw/RecSysPapers/blob/main/Calibration/Calibrating%20User%20Response%20Predictions%20in%20Online%20Advertising.pdf)，后链路预估值校准，本方法和SIR一起提出，主要解决面临数据稀疏和延迟转化问题的模型校准。主要思路是构建用户点击后的短期行为（如浏览、停留时长等）与后链路指标之间的关系来预测用户长期的转化行为

### [投放控制](#投放控制（Pacing）)（Pacing）

### 市场竞价预估（Bidding Landscape）

在二价交易机制下实时预估每一次交易市场价格的分布。可以参考本人简单总结的文章《[Bid Landscape总结](https://notlate.cn/p/cff1ae52c7af5559/)》。

* 启发式函数假设

  《[Bidding Machine: Learning to Bid for Directly Optimizing Profits in Display Advertising - 2017](https://arxiv.org/abs/1803.02194)》

  《[User Response Learning for Directly Optimizing Campaign Performance in Display Advertising - 2016](https://discovery.ucl.ac.uk/id/eprint/1524035/1/wang_p679-ren.pdf)》

* 预设概率分布函数

  《[Deep Censored Learning of the Winning Price in the Real Time Bidding - 2018](https://github.com/notlate-cn/tech-blogs/blob/main/papers/Bidding%20Landscape/2018-Deep%20Censored%20Learning%20of%20the%20Winning%20Price%20in%20the%20Real%20Time%20Bidding.pdf)》

  《[Predicting Winning Price in Real Time Bidding with Censored Data - 2015](http://wnzhang.net/share/rtb-papers/win-price-pred.pdf)》

  《[Bid landscape forecasting in online ad exchange marketplace - 2011](http://wnzhang.net/share/rtb-papers/bid-lands.pdf)》

* 不假设任何概率分布函数，直接预估分布

  《[Deep Landscape Forecasting for Real-time Bidding Advertising - 2019](https://arxiv.org/abs/1905.03028)》

相关阅读：

《[市场竞价预估（Bid Landscape）总结](https://notlate.cn/p/a7a3a81cb28cc06c/)》

### 竞价策略（Bidding Strategies）

#### 线性出价

* **固定出价**：所有请求出价相同，通过人为调整价格来影响投放成本和速度。优点：简单；缺点：没有考虑流量优劣。
* **随机出价**：在一定出价范围内每次随机出价。
* **真实出价**：不考虑预算限制，估算该展示的真实出价。二价机制保证了DSP按照真实价值出价也有套利机会。
* **受限点击成本出价**：在CPC模式下，广告主一般会设定可接受的最高点击成本CPC，通过计算CPC*pCTR来出价。若pCTR预估准确，该策略可以保证广告主收益。

相关阅读：

《[Bid optimizing and inventory scoring in targeted online advertising - 2012](http://wnzhang.net/share/rtb-papers/lin-bid.pdf)》

#### 非线性出价

该出价模型在有预算限制的情况下最大化收益（点击数或转化数），建议参考[张伟楠的RTB Papers列表](https://github.com/wnzhang/rtb-papers/blob/master/README.md#bidding-strategies)

相关阅读：

《[Optimal real-time bidding for display advertising - 2016](https://discovery.ucl.ac.uk/id/eprint/1496878/1/weinan-zhang-phd-2016.pdf)》

### 最大化竞价收益（Bid Shading）

由于ADX越来越多地采用一价计费，因此DSP若要保持自己的利益，则需要调整自己的出价策略，在保证一定胜率的情况下，尽量压低自己的出价，才能使得收益最大化。

相关阅读：

《[An Efficient Deep Distribution Network for Bid Shading in First-Price Auctions - 2021](https://arxiv.org/abs/2107.06650)》

《[Adaptive Bid Shading Optimization of First-Price Ad Inventory - 2021](https://ieeexplore.ieee.org/document/9482665)》

### 出价模式

普通出价模式：CPM、CPC、CPA

智能出价模式：oCPC、oCPM。

建议仔细阅读下述文章，读完之后基本彻底搞懂oCPX：

《[申探社：深入互联网广告中的出价模式（上）— 基础出价模式](https://zhuanlan.zhihu.com/p/87606755)》

《[申探社：深入互联网广告中的出价模式（下） — 联盟，RTB和RTA](https://zhuanlan.zhihu.com/p/139635658)》

《[申探社：深入互联网广告中的出价模式（补充篇）](https://zhuanlan.zhihu.com/p/159329979)》

《[申探社：再谈oCPX中的双出价](https://zhuanlan.zhihu.com/p/419366881)》

## 用户体验（User Experience）

### [频次控制](#频次控制（Frequency Capping）)

### 黑名单

屏蔽所有广告或某些行业、品牌、标题等。

### 素材美观度

根据不同的机型，设置最低可以展示的素材分数等。

### 负反馈

排除掉用户负反馈的广告主或广告

### 用户忍耐度

综合考虑用户的使用时长、曝光/点击/转化次数、负反馈次数等各种数据，对用户的广告忍耐度进行建模，实时对每一次请求打分，此推理数据可以作为CTR/CVR模型的特征，也可以直接根据这个值判定是否填充广告。

## 防作弊（Fraud Detection）

防作弊也是一个非常重要的课题。相关阅读：

《[Independent Auditing of Online Display Advertising Campaigns - 2016](http://www.it.uc3m.es/~rcuevas/papers/p120-callejo.pdf)》

《[Impression Fraud in On-line Advertising via Pay-Per-View Networks - 2013](http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/12305-sec13-paper_springborn.pdf)》

《[Understanding Fraudulent Activities in Online Ad Exchanges - 2011](http://conferences.sigcomm.org/imc/2011/docs/p279.pdf)》



## 欢迎各位留言交流探讨。
