---
title: "【烂笔头系列】推荐系统笔记05-Embedding技术"
description: "1. Embedding是什么 Embedding 就是用一个数值向量“表示”一个对象（Object）的方法 解读1：左边例子，从 king 到 queen 的向量和从 man 到 woman 的向量，无论从方向还是尺度来说它们都非常接近。 解读2：右边例子也很典型，从 walking 到 w…"
slug: "notes-recommender-system-05-embedding"
legacyId: 18705997
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18705997"
pubDate: 2025-02-09
category: "推荐系统"
tags: ["推荐系统","Embedding"]
featured: false
---

### 1. Embedding是什么

Embedding 就是用一个数值向量“表示”一个对象（Object）的方法

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221242798.png" alt="Embedding示意图" style="zoom:50%;"  loading="lazy" decoding="async"/>

解读1：左边例子，从 king 到 queen 的向量和从 man 到 woman 的向量，无论从方向还是尺度来说它们都非常接近。

解读2：右边例子也很典型，从 walking 到 walked 和从 swimming 到 swam 的向量基本一致，这说明词向量揭示了词之间的时态关系

### 2. Embedding技术的重要性

#### （1）处理稀疏特征的利器

1）大量使用 One-hot 编码会导致样本特征向量极度稀疏

2）深度学习的结构特点又不利于稀疏特征向量的处理，原因如下：

​		① 特征过于稀疏会导致整个网络的收敛非常慢，因为每一个样本的学习只有极少数的权重会得到更新，这在样本数量有限的情况下会导致模型不收敛。

​		② One-hot 类稀疏特征的维度往往非常地大，可能会达到千万甚至亿的级别，如果直接连接进入深度学习网络，那整个模型的参数数量会非常庞大，这对于一般公司的算力开销来说都是吃不消的。

3） 因此由 Embedding 层负责将稀疏高维特征向量转换成稠密低维特征向量。

#### （2）可以融合大量有价值信息，本身就是极其重要的特征向量

1）相比由原始信息直接处理得来的特征向量，Embedding 的表达能力更强

2）Graph Embedding 技术被提出后，Embedding 几乎可以引入任何信息进行编码，使其本身就包含大量有价值的信息

### 3. Embedding的实现技术

#### （1）Word2Vec：首次成功应用

1）Word2Vec，2013年由谷歌提出。模型分为两种形式：CBOW(连续词袋模型：由相邻词预测中间词)和Skip-gram(跳词模型：由当前词预测前后相邻词)。

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221254567.png" alt="Word2Vec的两种训练方式" style="zoom:50%;"  loading="lazy" decoding="async"/>

2）训练方法：

​	① 准备语料

​	② 分词，去掉停用词等无实际含义词

​	③ 生成词序列

​	④ 选取滑动窗口N，通过截取词组的方式生成训练样本

​	⑤ 模型训练（可以基于开源项目）

3）模型结构：本质是一个三层神经网络

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221303479.png" alt="Word2Vec模型结构" style="zoom:50%;"  loading="lazy" decoding="async"/>

​	① 隐层激活函数：没有或者说输入即输出的恒等函数

​	② 输出激活函数： softmax 

4）词向量：

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221312355.png" alt="词向量训练结果" style="zoom:50%;"  loading="lazy" decoding="async"/>

​	① 输入层到隐层的权重矩阵$W_{V*N}$（输入向量矩阵） 的每一个行向量对应的就是我们要找的“词向量”。同理输出向量矩阵也可以表示，但是通常习惯使用输入向量矩阵表示“词向量”。

​	② 把输入向量矩阵转换成词向量查找表（Lookup table）

5）延伸：Word2vec还有非常多的知识点值得细细挖掘，比如：模型结构、目标函数、负采样方法、负采样中的目标函数等。建议看一下《动手学深度学习》的相关内容：[10.1词嵌入](http://zh.gluon.ai/chapter_natural-language-processing/word2vec.html)和[10.2近似计算](http://zh.gluon.ai/chapter_natural-language-processing/approx-training.html)。

#### （2）Item2Vec：万物皆Embedding

1）Item2Vec，2015年由微软提出，它是对 Word2vec 方法的推广，使 Embedding 方法适用于几乎所有的**序列数据**。

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221321966.png" alt="Item2Vec与Word2Vec对比" style="zoom:50%;"  loading="lazy" decoding="async"/>

2）Item2Vec 模型的技术细节几乎和 Word2vec 完全一致，只要能够用序列数据的形式把要表达的对象表示出来，再把序列数据“喂”给 Word2vec 模型，就能够得到任意物品的 Embedding了。

#### （3）Graph Embedding

1）互联网的数据可不仅仅是序列数据那么简单，越来越多的数据被我们以图的形式展现出来。典型的图结构数据示意图：

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221333959.png" alt="图结构数据" style="zoom:50%;"  loading="lazy" decoding="async"/>

​	① **社交关系**：从社交网络中，我们可以发现意见领袖，可以发现社区，再根据这些“社交”特性进行社交化的推荐。**如果我们可以对社交网络中的节点进行 Embedding 编码，社交化推荐的过程将会非常方便**。

​	② 知识图谱：知识图谱中包含了不同类型的知识主体（如人物、地点等），附着在知识主体上的属性（如人物描述，物品特点），以及主体和主体之间、主体和属性之间的关系。**如果我们能够对知识图谱中的主体进行 Embedding 化，就可以发现主体之间的潜在关系，这对于基于内容和知识的推荐系统是非常有帮助的**。

​	③ **行为关系**：由用户和物品组成的“二部图”，借助这样的关系图，我们自然能够**利用 Embedding 技术发掘出物品和物品之间、用户和用户之间，以及用户和物品之间的关系**，从而应用于推荐系统的进一步推荐。

#### 2）Deep Walk：基于随机游走的 Graph Embedding 方法

​	① Deep Walk，2014年由美国石溪大学的研究者提出。

​	② 主要思想：由物品组成的图结构上进行随机游走，产生大量物品序列，然后将这些物品序列作为训练样本输入 Word2vec 进行训练，最终得到物品的 Embedding。

![图结构数据转成序列数据的方法](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221342914.png)

​	③ 跳转概率：就是遍历 vi 的邻接点 vj 的概率。

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221348626.png" alt="跳转概率" style="zoom:50%;"  loading="lazy" decoding="async"/>

<1> 有向有权图：$N_+(v_i)$是节点$v_i$所有的出边集合，$M_{ij}$是节点$v_i$到节点$v_j$的边的权重，即跳转概率是跳转边的权重占所有相关出边权重之和的比例

<2> 无向无权图：是上述公式的特例，$M_{ij}=1$，$N_+(v_i)$是节点$v_i$所有的边集合。<!--没看懂-->

#### 3） Node2Vec：在同质性和结构性间权衡的方法

① Node2Vec，2016年由斯坦福大学的研究者提出。

② 主要思想：基于Deep Walk，Node2vec 通过调整随机游走跳转概率的方法，让 Graph Embedding 的结果在网络的**同质性（Homophily）**和**结构性（Structural Equivalence）**中进行权衡，可以进一步把不同的 Embedding 输入推荐模型，让推荐系统学习到不同的网络结构特点。

![Node2Vec结构示意图](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221355654.png)

③ 同质性：距离相近节点的 Embedding 应该尽量近似，**让游走的过程更倾向于 DFS**。示例：节点 u 与其相连的节点 s1、s2、s3、s4的 Embedding 表达应该是接近的。

④ 结构性：结构上相似的节点的 Embedding 应该尽量接近，**让随机游走要更倾向于 BFS**。示例：节点 u 和节点 s6都是各自局域网络的中心节点，它们在结构上相似，所以它们的 Embedding 表达也应该近似。

⑤ 跳转概率：$\pi_{vx}=\alpha_{pq}(t,x)·w_{vx}$

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221404937.png" alt="跳转概率" style="zoom:50%;"  loading="lazy" decoding="async"/>

<1> $w_{vx}$是$vx$的原始权重，$\alpha_{pq}(t,x)$如上图所示，$d_{tx}$表示节点$t$和距离节点$x$(节点$v$的下一个节点)的距离。

<2> 参数 p 被称为返回参数（Return Parameter），p 越小，随机游走回节点 t 的可能性越大，Node2vec 就更注重表达网络的结构性

<3> 参数 q 被称为进出参数（In-out Parameter），q 越小，随机游走到远方节点的可能性越大，Node2vec 更注重表达网络的同质性。

<4> **计算出的概率需要做归一化，使节点$v$到所有下一个节点的概率和为1。**

### 4. Embedding的应用方法

#### 1）直接应用

① 利用物品 Embedding 间的相似性实现相似物品推荐

② 利用物品 Embedding 和用户 Embedding 的相似性实现“猜你喜欢”等经典推荐功能

③ 利用物品 Embedding 实现推荐系统中的召回层

#### 2）预训练应用

把这些 Embedding 向量作为特征向量的一部分，跟其余的特征向量拼接起来，作为推荐模型的输入参与训练

#### 3）End2End应用：端到端训练

① 概念：不预先训练 Embedding，而是把 Embedding 的训练与深度学习推荐模型结合起来，采用统一的、端到端的方式一起训练，直接得到包含 Embedding 层的推荐模型

② 案例：微软的` Deep Crossing`，UCL 提出的 `FNN` 和 Google 的 `Wide&Deep`

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20210128221412660.png" alt="端到端训练" style="zoom:59%;"  loading="lazy" decoding="async"/>

#### 4）常用的向量相似度计算法方法

请参考[《计算向量间相似度的常用方法》](https://cloud.tencent.com/developer/article/1668762)。



### 5. 经典问答

#### 1. 比较： 预训练与端到端训练区别

Embedding预训练的优点：

① 更快。因为对于End2End的方式，Embedding层的优化还受推荐算法的影响，这会增加计算量。

② 难收敛。推荐算法是以Embedding为前提的，在端到端的方式中，训练初期由于Embedding层的结果没有意义，所以推荐模块的优化也可能不太有意义，可能无法有效收敛。

Embedding端到端的优点：

① 能够找到Embedding层在这个模型结构下的最优解。因为端到端将Embedding训练和推荐算法连接起来训练，那么Embedding层可以学习到最有利于推荐目标的Embedding结果。

#### 2. Deep walk的优点和特点

① 去掉多余噪音信息，关注主要矛盾，所以一般要生成比原样本更少的样本量

② deep walk的抽样过程保留了转移矩阵的“主要框架”，但同时当抽样次数不太高的时候，item embedding的覆盖率反而没有item2vec好

#### 3. AutoEncoder和Word2vec的关系是什么？

没找到特别好的材料，欢迎留言，参考 [SVD分解(三)：连Word2Vec都只不过是个SVD？](https://spaces.ac.cn/archives/4233)中的说法：

> 结构上：Word2vec与AutoEncoder和SVD是一样的；
>
> 实现上：Word2Vec最后接的是softmax来预测概率，也就是说实现了一个非线性变换，而自编码器或者SVD并没有。

### 6. 扩展阅读

强烈建议大家阅读下王喆推荐的[Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184)。

### 参考资料

《深度学习推荐系统实战》 -- 极客时间，王喆
