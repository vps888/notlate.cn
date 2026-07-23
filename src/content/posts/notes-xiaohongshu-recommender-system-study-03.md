---
title: "【烂笔头系列】小红书推荐系统学习笔记03-特征交叉"
description: "FM 线性模型 设有$d$个特征，记为：$\\pmb{X} = x 1, x 2, ... , x d $，则线性模型的表达式为： $$ p = b + \\sum {i=1}^d w i · x i $$ 其中，$b$为偏置，总共$d+1$个模型参数。$p$是预测结果，也就是特征和权重参数的加权…"
slug: "notes-xiaohongshu-recommender-system-study-03"
legacyId: 18705969
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18705969"
pubDate: 2025-02-09
category: "推荐系统"
tags: ["推荐系统"]
featured: false
---

## FM

### 线性模型

设有$d$个特征，记为：$\pmb{X} = [x_1, x_2, ... , x_d]$，则线性模型的表达式为：
$$
p = b + \sum_{i=1}^d w_i · x_i
$$
其中，$b$为偏置，总共$d+1$个模型参数。$p$是预测结果，也就是特征和权重参数的加权和。因为没有乘法操作，所以特征之间没有交叉。

### 线性模型+二阶交叉

$$
p = b + \sum_{i=1}^d w_i · x_i + \sum_{i=1}^d \sum_{j=i+1}^d u_{ij} · x_i · x_j
$$

其中，$w_i$是为每个特征分配一个权重参数，而$u_{ij}$则是为每一组交叉特征分配一个权重参数，那么其参数量则为$O(d^2)$。若特征比较多，则参数量会非常多。

优化思路：把$u_{ij}$这个方阵近似用两个向量内积表示，即$\pmb{U} ≈ \pmb{V} · \pmb{V^T}$。参数量从$O(d^2) \rightarrow O(kd)$。

![image-20230415233604273](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230415233604273.png)

这就是FM。

## DCN

### 交叉层

![image-20230415234947465](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230415234947465.png)

![image-20230415235052524](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230415235052524.png)

### 交叉网络(Cross Network)

![image-20230415235159373](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230415235159373.png)
$$
X_1 = X_0 \circ (W_0 * X_0 + b_0) + X_0 \\
X_2 = X_0 \circ (W_1 * X_1 + b_1) + X_1 \\
X_3 = X_0 \circ (W_2 * X_2 + b_2) + X_2 \\
$$

### 深度交叉网络（DCN）

![image-20230415235736561](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230415235736561.png)

DCN既可以用于召回，也可以用于排序。

双塔模型中的用户塔和物品塔都可以是DCN。

多目标中的Shared Bottoms和MMoE中的专家网络也可以是DCN。

## LHUC

[LHUC - 2016](https://arxiv.org/pdf/1601.02828.pdf)起源于语音识别，只能用于精排。[快手将其用于推荐精排，称为PPNet](https://www.51cto.com/article/644214.html)。

### LHUC应用于推荐系统

![image-20230416000906812](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416000906812.png)

### 快手PPNet结构

![img](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/026923bbefc6bf9355fbaae4c7c441f9.jpg)

## SENet

### 特征内加权

![image-20230416001137195](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416001137195.png)

上图是SENet结构图，其中输入的m个离散特征的Embedding向量长度可以不同。

SENet的本质是对离散特征的filed-wise加权。

### 特征间加权

![image-20230416001512557](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416001512557.png)

![image-20230416001538523](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416001538523.png)

![image-20230416001623453](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416001623453.png)

### FiBiNet模型

![image-20230416001736344](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/image-20230416001736344.png)

## 参考文献

公开课地址：[GitHub](https://github.com/wangshusen/RecommenderSystem)
