---
title: "【MLIR】Linalg中ElementwiseOpFusion优化分析（总）"
description: "通过此命令可以查看MLIR中关于linalg的所有Pass，本篇主要分析： linalg fuse elementwise ops （ 基于llvm 21.1.8版本 (https://github.com/llvm/llvm project/tree/llvmorg 21.1.8)）。 1.…"
slug: "mlirlinalg-elementwiseopfusion-analysis-4"
legacyId: 19358995
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19358995"
pubDate: 2026-01-19
updatedDate: 2026-01-28
category: "AI 编译器"
tags: ["AI 编译器","MLIR","Linalg"]
featured: false
---

``` shell
./mlir-opt -h | grep linalg
```

通过此命令可以查看MLIR中关于linalg的所有Pass，本篇主要分析：`linalg-fuse-elementwise-ops`（[基于llvm 21.1.8版本](https://github.com/llvm/llvm-project/tree/llvmorg-21.1.8)）。

## 1. 介绍

### 1.1 代码介绍

`linalg-fuse-elementwise-ops`是 Linalg 中关于 Elementwise 类算子融合的优化 Pass。从源代码中全文检索此关键字，在`mlir/include/mlir/Dialect/Linalg/Passes.td:73`中找到了`LinalgElementwiseOpFusionPass`定义。定义非常简单，只声明了依赖的三种方言，如下：
![LinalgElementwiseOpFusionPass定义](https://img2024.cnblogs.com/blog/3599704/202512/3599704-20251229151019987-1101858257.png "LinalgElementwiseOpFusionPass定义")

继续检索`LinalgElementwiseOpFusionPass`关键字，在`mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp:2284`中找到了具体实现，代码如下：
![LinalgElementwiseOpFusionPass实现](https://img2024.cnblogs.com/blog/3599704/202512/3599704-20251229154702352-1197483950.png "LinalgElementwiseOpFusionPass实现")

该类继承自`LinalgElementwiseOpFusionPassBase`类，跳转进去后发现是使用 `mlir-tblgen`工具生成的代码，如下：
![LinalgElementwiseOpFusionPassBase基类](https://img2024.cnblogs.com/blog/3599704/202512/3599704-20251229155031293-1718754296.png "LinalgElementwiseOpFusionPassBase基类")

该类最重要的作用是实现了Pass机制中的虚函数`runOnOperation`，也就是该优化的核心功能，见上图中红框部分`populate`关键词开头的几个函数。

### 1.2 Pass核心流程图

```mermaid
 graph LR
      A[LinalgElementwiseOpFusionPass<br/>Linalg元素级操作融合Pass] --> B[runOnOperation<br/>执行优化变换]

      B --> B1[populateElementwiseOpsFusionPatterns<br/>注册元素级融合模式]
      B --> B2[populateFoldReshapeOpsByExpansionPatterns<br/>注册扩展融合模式]
      B --> B3[populateFoldReshapeOpsByCollapsingPatterns<br/>注册折叠融合模式]
      B --> B4[Canonicalization & Constant Folding<br/>规范化和常量折叠]

      subgraph S1["元素级融合 - 合并连续的Generic操作"]
          C1[FuseElementwiseOps<br/>匹配并融合元素级操作]
          C2[FoldFillWithGenericOp<br/>将Fill操作内联到Generic]
          C3[FoldScalarOrSplatConstant<br/>折叠标量/Splat常量]
          C4[RemoveOutsDependency<br/>移除未使用的输出依赖]

          D1{areElementwiseOpsFusable<br/>检查融合前置条件}
          D2[fuseElementwiseOps<br/>执行融合变换]

          E1[getPreservedProducerResults<br/>确定需要保留的Producer结果]
          E2[generateFusedElementwiseOpRegion<br/>生成融合后的操作体]
          E3[getIndexingMapOfProducerOperands<br/>计算融合坐标系中的索引映射]

          F1[isOpOperandCanBeDropped<br/>判断操作数能否被删除]
      end

      subgraph S2["维度扩展融合 - 通过扩展迭代空间消除Reshape"]
          G1[FoldReshapeWithGenericOpByExpansion<br/>折叠Producer的Reshape]
          G2[FoldWithProducerReshapeOpByExpansion<br/>折叠Consumer的Reshape]
          G3[FoldPadWithProducerReshapeOpByExpansion<br/>处理Pad+Reshape组合]

          H1{isFusableWithReshapeByDimExpansion<br/>检查是否可通过扩展融合}
          H2[fuseWithReshapeByExpansion<br/>执行扩展融合]

          I1[ExpansionInfo::compute<br/>计算维度扩展映射]
          I2[createExpandedOp<br/>创建扩展后的Linalg操作]

          J1[createExpandedGenericOp<br/>创建扩展的GenericOp]
          J2[createExpandedTransposeOp<br/>创建扩展的TransposeOp]
          J3[updateExpandedGenericOpRegion<br/>修正扩展后的index操作]
      end

      subgraph S3["维度折叠融合 - 通过折叠迭代空间消除Reshape"]
          K1[FoldWithProducerReshapeOpByCollapsing<br/>折叠Consumer的ExpandShape]
          K2[FoldReshapeWithGenericOpByCollapsing<br/>折叠Producer的CollapseShape]
          K3[FoldPadWithProducerReshapeOpByCollapsing<br/>处理Pad+ExpandShape组合]

          L1[getCollapsableIterationSpaceDims<br/>计算可折叠的迭代维度]
          L2[collapseOpIterationDims<br/>执行维度折叠]

          M1[CollapsingInfo::initialize<br/>初始化折叠映射]
          M2[createCollapsedOp<br/>创建折叠后的Linalg操作]
          M3[generateCollapsedIndexingRegion<br/>通过除法和取模恢复原始索引]

          N1[cloneToCollapsedOp<br/>克隆并调整操作]
          N2[collapseOperandsAndResults<br/>折叠操作数和结果张量]

          P1[isDimSequencePreserved<br/>检查维度序列在映射中是否保持]
      end

      B1 --> C1
      B1 --> C2
      B1 --> C3
      B1 --> C4

      C1 --> D1
      C1 --> D2
      D2 --> E1
      D2 --> E2
      D2 --> E3
      E1 --> F1
      E2 --> E3

      B2 --> G1
      B2 --> G2
      B2 --> G3

      G1 --> H1
      G1 --> H2
      G2 --> H1
      G2 --> H2
      G3 --> H2

      H2 --> I1
      H2 --> I2
      I2 --> J1
      I2 --> J2
      J1 --> J3

      B3 --> K1
      B3 --> K2
      B3 --> K3

      K1 --> L1
      K1 --> L2
      K2 --> L1
      K2 --> L2
      K3 --> L2

      L2 --> M1
      L2 --> M2
      L2 --> M3
      M2 --> N1
      N1 --> N2
      L1 --> P1

      style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
      style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
      style B1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
      style B2 fill:#ffe0b2,stroke:#e65100,stroke-width:2px
      style B3 fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
      style C1 fill:#ffcdd2
      style G1 fill:#fff9c4
      style K1 fill:#e1bee7
      style D2 fill:#a5d6a7,stroke:#2e7d32,stroke-width:2px
      style H2 fill:#ffcc80,stroke:#e65100,stroke-width:2px
      style L2 fill:#ce93d8,stroke:#880e4f,stroke-width:2px
```

## 2 重点功能分析

###  [【MLIR】Linalg中ElementwiseOpFusion的优化模式技术分析（一）](https://notlate.cn/blog/mlirlinalg-elementwiseopfusion-analysis)

### [【MLIR】Linalg中ElementwiseOpFusion的优化模式技术分析（二）](https://notlate.cn/blog/mlirlinalg-elementwiseopfusion-analysis-2)

### [【MLIR】Linalg中ElementwiseOpFusion的优化模式技术分析（三）](https://www.cnblogs.com/notlate-cn/articles/19500691)
