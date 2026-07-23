---
title: "【烂笔头系列】小红书推荐系统学习笔记05-重排"
description: "重排是精排的后处理操作。 物品多样性 相似度度量 基于物品属性标签 基于物品向量表征 （1）双塔模型的物品塔，但是因为头部效应问题导致学不好物品向量表征 （2） 基于图文内容学习 CLIP 基于图文内容的物品向量表征 原理 对于图片 文本二元组数据进行对比学习，预测图文是否匹配。优点是：无需人…"
slug: "notes-xiaohongshu-recommender-system-study-05"
legacyId: 18705971
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18705971"
pubDate: 2025-02-09
category: "推荐系统与机器学习"
tags: ["推荐系统与机器学习","推荐系统"]
featured: false
---

重排是精排的后处理操作。

## 物品多样性

### 相似度度量

* 基于物品属性标签
* 基于物品向量表征
  
  （1）双塔模型的物品塔，但是因为头部效应问题导致学不好物品向量表征
  
  （2）**基于图文内容学习**

### CLIP - 基于图文内容的物品向量表征

#### 原理

对于图片-文本二元组数据进行对比学习，预测图文是否匹配。优点是：无需人工标注。参考文献《[Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf)》和解读文章《[对Connecting Text and Images的理解](https://mileistone.github.io/work/2021/01/14/thought-on-connecting-text-and-images/)》。

### 正样本

同一个物品中的图片和文字二元组数据构成正样本。

### 负样本

同batch内，正样本的图片与其他样本的文字组成的二元组数据构成负样本。

![image-20230416004943510](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416004943510.png)

## MMR多样性算法

### 原理

MMR算法中需要计算两个物品的相似度，这个相似度计算就用上一小节提到的CLIP方法学习到的物品Embedding向量的余弦相似度计算即可。

![image-20230416005624631](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416005624631.png)

### 步骤

![image-20230416005759680](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416005759680.png)

### 滑动窗口解决S集合过大问题

核心思想就是只考虑待排列表中最后一个窗口范围内的物品无相似（多样性好）即可。

![image-20230416005938631](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416005938631.png)

## 业务规则控制多样性

通常是MMR+规则控制多样性

## DPP

### 数学原理

![image-20230416010339177](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416010339177.png)

![image-20230416010408983](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416010408983.png)

### 计算思路

![image-20230416010542425](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416010542425.png)

### 求解方法

#### 暴力方法

![image-20230416011056075](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416011056075.png)

#### Hulu方法

![image-20230416011146962](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416011146962.png)

### DPP+滑动窗口

![image-20230416011239145](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416011239145.png)

## 参考文献

公开课地址：[GitHub](https://github.com/wangshusen/RecommenderSystem)
