---
title: "【烂笔头系列】推荐系统笔记09-深度学习推荐模型发展脉络"
description: "1. 深度学习模型拟合能力更强 特征交叉方式中，点积等方式过于简单，在样本数据比较复杂的情况下，容易欠拟合。而深度学习可以大大提高模型的拟合能力，比如在 NeuralCF（神经网络协同过滤）模型中，点积层被替换为多层神经网络，理论上多层神经网络具备拟合任意函数的能力，所以我们通过增加神经网络层…"
slug: "notes-recommender-system-09-study"
legacyId: 18706001
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18706001"
pubDate: 2025-02-09
category: "推荐系统与机器学习"
tags: ["推荐系统与机器学习","推荐系统"]
featured: false
---

### 1. 深度学习模型拟合能力更强

特征交叉方式中，点积等方式过于简单，在样本数据比较复杂的情况下，容易欠拟合。而深度学习可以大大提高模型的拟合能力，比如在 NeuralCF（神经网络协同过滤）模型中，点积层被替换为多层神经网络，理论上多层神经网络具备拟合任意函数的能力，所以我们通过增加神经网络层的方式就可以解决欠拟合的问题了。

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/7063d223da013845534d3c84b7ab9409-20210116205913169.jpg" alt="NeuralCF模型结构图" style="zoom:33%;"  loading="lazy" decoding="async"/>

### 2. 深度学习模型结构更加灵活

（1）深度学习模型结构不尽相同，多数是可以通过堆叠不同作用的网络层，最简单的是串联结构，有的像网状结构，有的像金字塔结构等等。

（2）典型案例是阿里巴巴的 DIN（深度兴趣网络，下图左）和 DIEN（深度兴趣进化网络，下图右），通过在模型结构中引入**注意力机制**和**模拟兴趣进化的序列模型**，来更好地模拟用户行为。

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/202cfa968b1aa6fa4349722bb4ab4332.jpg" alt="DIN和DIEN模型结构图" style="zoom:33%;"  loading="lazy" decoding="async"/>

其中，DIN 模型在神经网络中增加了一个“激活单元“结构，是为了模仿人类的注意力机制。其改进版 DIEN 模型不仅引入了注意力机制，还用AUGRU单元模拟了用户兴趣随时间的演化过程。

这些改进都是基于实际业务洞察分析的演进，所以需要正确、全面地掌握不同深度学习模型的特点以及发展关系非常重要。

### 3. 深度学习模型发展关系图

<img src="https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/10e8105911823d96348dc7288d4d26c5.jpg" style="zoom:33%;"  loading="lazy" decoding="async"/>

（1）**核心结构**：多层感知机（MultiLayer Perception，MLP）。

（2）**基础结构**：AutoRec，一种单隐层的神经网络模型，将自编码器（AutoEncoder）的思想与协同过滤结合。

（3）**经典结构**：Deep Crossing，在原始特征和MLP之间加入了Embedding层，把输入的稀疏特征先转换成稠密 Embedding 向量，再输入到MLP进行训练，这就解决了MLP不善于处理稀疏特征的问题。**因此Embedding+MLP结构是最经典的深度学习推荐模型结构。**

（4）**广泛应用结构**：Wide&Deep，模型分为两部分：Wide部分是浅层的神经网络结构，让模型具备很好的记忆性；Deep部分是深层MLP，让模型具备良好的泛化性，最终把两者结合起来。凭借着易实现、易改造的特点，获得了业界广泛应用。同时还衍生出了诸多变种，比如通过改造 Wide 部分提出的Deep&Cross和DeepFM，通过改造Deep部分提出的AFM、NFM等等。

（5）**与其他机器学习子领域交叉**：

① 深度学习和**注意力机制**结合，比如阿里的DIN，浙大和新加坡国立提出的AFM等；

② 把**序列模型**引入Embedding+MLP的经典结构，比如阿里的DIEN等；

③ 深度学习和**强化学习**结合，比如微软的DRN（深度强化学习网络），以及包括[美团-猜你喜欢](https://tech.meituan.com/2018/11/15/reinforcement-learning-in-mt-recommend-system.html)、[阿里-强化学习在阿里的技术演进与业务创新](https://alitech-private.oss-cn-beijing.aliyuncs.com/1517812754285/reinforcement_learning.pdf?Expires=1615452000&OSSAccessKeyId=LTAI4G7JAotCoNVvbmrLZNtj&Signature=qwQTj4C%2FcScl1NJz%2FvknvkRgTrk%3D)在内的非常有价值的业界应用。

### 4. 演进规律

#### （1）改变神经网络的复杂程度

从最简单的单层神经网络模型 AutoRec，到经典的深度神经网络结构 Deep Crossing，它们主要的进化方式在于**增加了深度神经网络的层数和结构复杂度**。

#### （2）改变特征交叉方式

这种演进方式的要点在于**大大提高了深度学习网络中特征交叉的能力**。比如改变了用户向量和物品向量互操作方式的NeuralCF，定义了多种特征向量交叉操作的 PNN 等等。

#### （3）把多种模型组合应用

组合模型主要指的就是以 Wide&Deep 模型为代表的一系列**把不同结构组合在一起**的改进思路。它通过组合两种甚至多种不同特点、优势互补的深度学习网络，来提升模型的综合能力。

#### （4）让深度推荐模型和其他领域进行交叉

我们从 DIN、DIEN、DRN 等模型中可以看出，深度推荐模型无时无刻不在从其他研究领域汲取新的知识。从今年的推荐系统顶会 Recsys2020 中可以看到，NLP 领域的著名模型 Bert 又与推荐模型结合起来，并且产生了非常好的效果。一般来说，自然语言处理、图像处理、强化学习这些领域都是推荐系统经常汲取新知识的地方。



下一篇详细记录深度学习模型。

### 参考资料

《深度学习推荐系统实战》 -- 极客时间，王喆
