---
title: "【烂笔头系列】推荐系统笔记01-推荐系统概要"
description: "1. 深度学习推荐系统基础概念 2. 从0到1搭建深度学习推荐系统 开源项目Sparrow RecSys实操（以Mac为例） （1）安装Scala 2.11（务必是2.11大版本，否则与开源项目设置的版本号不匹配，会有执行失败的问题） （2）下载开源项目Sparrow RecSys git c…"
slug: "notes-recommender-system-01-recommender-system"
legacyId: 18705995
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/18705995"
pubDate: 2025-02-09
category: "推荐系统"
tags: ["推荐系统"]
featured: false
---

### 1. 深度学习推荐系统基础概念

![推荐系统](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/01.%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F.png)

### 2. 从0到1搭建深度学习推荐系统  -- 开源项目Sparrow RecSys实操（以Mac为例）

#### （1）安装Scala 2.11（务必是2.11大版本，否则与开源项目设置的版本号不匹配，会有执行失败的问题）

```
brew update
brew install scala@2.11
```

#### （2）下载开源项目Sparrow RecSys

`git clone https://github.com/wzhe06/SparrowRecSys.git`

项目地址：https://github.com/wzhe06/SparrowRecSys

#### （3）安装IDEA和JDK

① 下载IDEA（https://www.jetbrains.com/idea/download/#section=mac）

② 下载JDK（https://www.oracle.com/java/technologies/javase-jdk15-downloads.html）

③ 安装IDEA和JDK（JDK的路径~/Library/Java/JavaVirtualMachines/openjdk-15.0.1-1）

#### （4）导入工程&运行

① 打开IDEA，打开File->Project Strucure->Project->Project JDK(我的好像会自动识别)。若没有识别（显示jdk15.1）,点击三角号，自己添加，步骤Add SDK->JDK->选择上面提到的JDK路径选择。

② 在pom.xml点击右键，设置为maven project->'Reload project'。耐心等待，这个很费时间。

③ 然后找到SparrowRecSys/src/main/java/com/SparrowRecSys/online/RecSysServer,右击选择"Run 'RecSysServer.main()'",程序就执行起来了.

④ 浏览器中输入http://localhost:6010/即可打开首页

#### （5）SparrowRecsys涵盖的技术

![](https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/8cee6a7eeebda9745bfbe1b6yy18c59e.jpg)

### 3. 推荐系统相关知识扩充

#### （1）书籍推荐

① 深度学习推荐系统

② 西瓜书

③ 蒲公英书

④ 百面机器学习

⑤ 数学之美（吴军）

#### （2）实践工具相关

#### Spark

① [形象理解Hadoop、Hive、Spark](https://www.zhihu.com/question/27974418)

② [根据官网写一个Spark Hello World 程序](https://spark.apache.org/docs/2.4.3/quick-start.html)

③ [初步了解Spark MLlib](https://spark.apache.org/docs/2.4.3/ml-guide.html)

#### Tensorflow

① [介绍 TensorFlow 和 Keras 的基本概念的文章](https://blog.csdn.net/li528405176/article/details/83857286)

② [Keras 写一个 Hello World](https://www.tensorflow.org/tutorials/quickstart/beginner)

③ [官方教程](https://www.tensorflow.org/tutorials)

#### Redis

① [Redis基本介绍](http://www.redis.cn/)

② [Redis基本操作](http://www.redis.cn/download.html)

#### 经典问答

#### 1. 问：对于电影推荐系统来讲，哪些数据对生成用户个性化推荐结果最有帮助？

答：（1）内容相关特征：电影种类，演员，电影内容，电影质量等；（2）用户行为特征：用户历史浏览记录、观看记录等；

#### 2. 问：召回层单独优化新增特征，在排序层没有，如何处理？

答：在设计召回层和排序层的时候一般要联合设计，召回层要特别关注召回率指标。如果新增特征对结果影响比较大，排序层模型训练的时候同步引入这两个特征。

### 参考资料

《深度学习推荐系统实战》 -- 极客时间，王喆
