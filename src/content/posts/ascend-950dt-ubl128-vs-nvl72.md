---
title: "昇腾 950DT 超节点（UBL128）与 NVIDIA NVL72 体系结构对比总结"
description: "昇腾 950DT UBL128 与 NVIDIA NVL72 的互联、内存层级、微架构及模型部署策略对比。"
slug: "ascend-950dt-ubl128-vs-nvl72"
legacyId: 19827342
pubDate: 2026-10-16
category: "AI 编译器"
tags: ["昇腾", "NVIDIA", "AI基础设施", "超节点", "大模型"]
featured: false
---

## 1. 核心概念与扩展范式解析（ScaleUp vs. ScaleOut）

在大模型分布式训练与超大规模 AI 集群的互联语境下，ScaleUp 与 ScaleOut 构成了集群分层通信的两大核心维度：

* **ScaleUp（纵向扩展 / 向上扩展）**：
* **定义**：利用私有高带宽、低延迟的专用总线或交换矩阵，将多个计算芯片（NPU/GPU）深度聚合为在逻辑上近乎单一的“超大计算/内存节点”（Supernode / 超节点）。
* **承载流量**：面向超高频、大吞吐量、对延迟抖动极度敏感的细粒度通信，如 **TP（张量并行）**、**EP（专家并行 / AllToAll 动态 Token 路由）**。
* **代表技术**：NVIDIA NVLink 5.0 / NVSwitch、华为灵衢总线（Unified Bus / UB）。


* **ScaleOut（横向扩展 / 向外扩展）**：
* **定义**：在超节点达到物理尺寸、供电或布线极限后，利用标准数据中心以太网或 InfiniBand 网络将多个 ScaleUp 节点横向串联，形成千卡至十万卡算力池。
* **承载流量**：通信量相对较小、通信步调规律、容易与前向/反向计算形成流水重叠（Overlap）的粗粒度通信，如 **DP / FSDP（数据并行 / 权重分片）**、**PP（流水线并行）**。
* **代表技术**：RoCEv2、UBoE（UB over Ethernet）、InfiniBand。


* **系统设计核心定律（通信局部性）**：
* **高频通信局域化**：把流量最大、频率最高、对延迟最敏感的通信（如 MoE 路由）放到 ScaleUp 域内。
* **低频通信走外部**：将能够被 MatMul/Vector 计算平滑掩盖的通信放到 ScaleOut 网络上，避免低速网络拖垮计算硬件利用率（MFU）。



---

## 2. NVIDIA NVL72 vs. 昇腾 950DT 超节点（UBL128）全规格横向对比

| 体系结构维度 | NVIDIA GB200 NVL72 | 华为昇腾 950DT 超节点 (UBL128) |
| --- | --- | --- |
| **超节点物理形态** | **单机柜极密一体化**（18 个 Compute Trays + 9 个 NVSwitch Trays，背板铜缆直连） | **模块化多机机群**（8 台标准 PC16 / Atlas 850e 服务器，每台 16 卡，灵衢线缆/背板组网） |
| **ScaleUp 域规模** | **72 卡**（36 颗 Grace CPU + 72 颗 B200 GPU） | **128 卡**（128 × Ascend 950DT） |
| **ScaleUp 域内总线** | **NVLink 5.0**（私有硬件协议） | **灵衢总线（Unified Bus / UB 3.2Tbps）**（开放统一互联协议） |
| **单卡 ScaleUp 物理端口** | **18 个端口 (Links)** | **8 个端口（8 × 400G UB 端口）** |
| **单卡 ScaleUp 有效带宽** | **900 GB/s**（单向）/ **1800 GB/s (1.8 TB/s)**（双向） | **400 GB/s**（单向）/ **800 GB/s**（双向） |
| **ScaleOut 域外互联** | 800G / 1.6T RoCEv2 或 InfiniBand（约 100~200 GB/s/卡） | **UBoE / Ethernet**（**50 GB/s / 卡**，形成严格的 8× 带宽分层） |
| **协议语义支持** | 专有 NVLink 内存语义 + SHARP 机内网络规约 | **三位一体**：UB Memory（同步访存）、URMA（异步拷贝）、UBoE（以太网直通） |
| **片上内存 (HBM)** | **13.5 TB**（192GB HBM3e/卡，**8.0 TB/s** 单卡带宽） | **18.0 TB**（144GB 片上内存/卡，**4.0 TB/s** 单卡带宽） |
| **片上 L2 Cache** | **128 MB** 片上共享 L2，带宽 $>20\text{ TB/s}$（另含 36.5MB SM SRAM） | **192 MB** 片上共享 L2，带宽约 $16\sim 20\text{ TB/s}$（支持 Sector Cache 与 CMO） |
| **Host 内存形态** | **17 TB 板载 LPDDR5X**（每 CPU 480/512GB，单 CPU 带宽 546 GB/s） | **12 TB 标准插槽 DDR5**（8 台 PC16，每台配备 1.5 TB Host DDR） |
| **HBM $\leftrightarrow$ DDR 互联通路** | **NVLink-C2C**（封装级短距直连） | **PCIe 5.0 x16**（标准主板/背板走线） |
| **HBM $\leftrightarrow$ DDR 单卡带宽** | **~400 - 450 GB/s**（理论单向 450 GB/s，双向 900 GB/s） | **~58 - 60 GB/s**（理论单向 64 GB/s，双向 128 GB/s） |
| **芯片微架构与执行模型** | 纯 SIMT 架构（Streaming Multiprocessor + Tensor Core + TMA 异步引擎） | 第三代达芬奇分离架构：Cube Core (AIC) + Vector Core (AIV) + NDDMA + STARS 2.0 硬件调度 |
| **原生精度格式** | FP4, FP8, FP16, BF16, TF32, FP32 | 原生支持 **MXFP4, MXFP8, HiF8**, BF16, FP16 |
| **供电与散热约束** | 单机柜功耗约 **120 kW**，深度绑定全液冷（DLC） | 单机 16P 模块化设计，支持风液混合或整机柜液冷，机房适配度更灵活 |

---

## 3. 互联物理层与网络协议机制深析

### 3.1 速率与端口换算

* **灵衢 3.2Tbps 的来源**：

$$8 \text{ 端口} \times 400\text{ Gbps/端口} = 3200\text{ Gbps} = 3.2\text{ Tbps (物理总线速率)}$$



转换为存储与访存体系的单向有效数据带宽：

$$3200\text{ Gbps} \div 8\text{ bits/Byte} = 400\text{ GB/s/卡}$$


* **NVLink 5.0 速率口径**：
单张 Blackwell B200 引出 18 条 NVLink 链路，采用 224G SerDes（PAM4 调制），单链路单向 400 Gbps（50 GB/s）。

$$\text{单向带宽} = 18 \times 50\text{ GB/s} = 900\text{ GB/s}$$


$$\text{双向聚合带宽} = 18 \times 100\text{ GB/s} = 1800\text{ GB/s} = 1.8\text{ TB/s}$$



（注：业界常说的“双向”即指此 1.8 TB/s，并非误传的百分比缩写）。

### 3.2 交换拓扑模式：内嵌式（华为） vs. 物理外置式（NVIDIA）

* **昇腾 950DT：双层内生交换架构**
* **卡内（UB On-Chip Switch）**：直接做入 950DT 的 **IO Die** 内部，具备全向 Crossbar、无状态硬件转发表（128 项 SRAM/TCAM）、Cut-Through 纯硬件线速穿透（单跳 30~80ns）与 Credit-Based 流控能力。芯片自身兼具计算终端与微型交换节点属性。
* **板载（On-Board Switch）**：在 PC16 服务器基板上集成汇聚交换单元，将机内 16 张卡汇聚为线速无阻塞的 Clos 拓扑（机内呈现逻辑 1-Hop 互联），剩余端口对外引出，与其余 7 台 PC16 连成 128 卡全互联超节点。
* **对等直连支持度**：950DT 片上 Switch 亦原生支持在无板载交换芯片的环境下，依靠芯片自身端口按 2D/3D-Torus 或超立方体构建多跳纯直连对等网。


* **NVIDIA NVL72：严格解耦的集中星型交换**
* GPU（B200）内部不具备任何包中继中转功能（无 Transit 能力）。
* 18 条 NVLink 连线全部从计算托盘引出，物理接入机柜中部的 **9 个独立 NVSwitch 托盘（含 18 颗专用 NVSwitch 芯片）**，以两级集中式 Crossbar 支撑 72 张卡地位均等的多对多直联。



### 3.3 跨超节点通信协议：UBoE 与 RoCEv2 的本质差异

* **跨机物理层**：跨超节点采用标准数据中心以太网交换机，单卡引出有效带宽为 **50 GB/s**（400Gbps），相比 ScaleUp 域内形成 8× 带宽落差。
* **RoCEv2 机制与痛点**：将 InfiniBand 传输报文封装进 UDP/IP（Port 4791），高度依赖以太网交换机开启 PFC（基于优先级的流控）构建无损网络，在万卡大规模 AllToAll 场景下易诱发 PFC 死锁、拥塞风暴蔓延与 QP Cache 缺失。
* **UBoE 机制与优化**：把灵衢总线原生传输单元（Flit/Packet）直接封装到以太网帧中，结合灵衢底层的 CTP（紧凑模式）与 RTP（可靠模式），配合硬件端到端 Credit 流控与快反算法，摆脱对复杂 PFC 状态机与 IB Verbs 软件协议栈的强依赖，数据可由远端 IO Die 硬件直接解包注入 HBM/L2 Cache。

---

## 4. 存储层级、搬运控制与微架构分工

### 4.1 内存形态与互联通道对比

* **NVIDIA（GB200）**：
* **介质形态**：CPU 搭配高密板载焊接的 **LPDDR5X**（提供 546 GB/s 单节点带宽）。
* **互联管道**：通过 **NVLink-C2C**（Chip-to-Chip）超短距高密封装互联，提供双向 **900 GB/s** 的物理吞吐与硬件双向缓存一致性，消除显存与内存之间的软件搬运边界。


* **昇腾 950DT（UBL128）**：
* **介质形态**：Host 侧配备通用标准插槽式 **DDR5 RDIMM**，PC16 单机聚合 **1.5 TB**，8 台机汇聚为 **12TB 全局 Host 内存池**。
* **互联管道**：NPU 芯片通过 **PCIe 5.0 x16** 总线连接 Host CPU，实际单向有效搬运吞吐为 **~58-60 GB/s**。



### 4.2 搬运主体与硬件引擎职责划分（NDDMA vs. SDMA）

* **NDDMA（N-dimensional Direct Memory Access）**：
* **物理位置**：集成于 **AI Core（计算核）** 内部。
* **职责范围**：负责在片上 **HBM / L2 Cache** 与 AI Core 内部的 **Unified Buffer（UB / Scratchpad）** 之间搬运数据，专精于计算近端的多维张量 Layout 变换、切片与矩阵自动转置，不负责跨 PCIe 搬运。


* **SDMA（System Direct Memory Access）**：
* **物理位置**：集成于 **IO Die** 内部。
* **职责范围**：作为 PCIe 总线的主设备（Bus Master），真正执行 **H2D / D2H（DDR $\leftrightarrow$ HBM）** 的物理 DMA 搬运动作。


* **H2D 搬运流转链条**：
1. Host 运行时通过驱动下发搬运任务至芯片命令队列（SQ）；
2. 片上 **STARS 2.0 硬件调度器** 解析依赖并向 SDMA 发派任务；
3. **SDMA 引擎** 发起 PCIe 5.0 事务从 Host DDR 拉取数据写入 HBM；
4. 搬运完成后产生事件信号，AI Core 内部的 **NDDMA** 方才介入，将 HBM 激活值拉入 Unified Buffer 交由 Cube/Vector Core 执行计算。



### 4.3 跨节点访存通路与有效带宽实测汇总

| 访问路径 | 采用协议 | 关键经过的硬件链路 | 实际有效单向带宽 | 核心物理瓶颈点 |
| --- | --- | --- | --- | --- |
| **跨机 HBM $\leftrightarrow$ HBM** | **URMA** | 本卡 IO Die $\rightarrow$ 灵衢总线 $\rightarrow$ 远端 IO Die $\rightarrow$ 远端 HBM | **400 GB/s** | 灵衢 8×400G UB 接口物理线速 |
| **节点内 HBM $\leftrightarrow$ DDR** | **PCIe + SDMA** | 本卡 IO Die $\rightarrow$ PCIe 5.0 x16 $\rightarrow$ 本机 CPU/DDR | **~58-60 GB/s** | 本机单卡 PCIe 5.0 x16 物理吞吐 |
| **跨节点 HBM $\leftrightarrow$ DDR** | **URMA + SDMA** | 本卡 IO Die $\rightarrow$ 灵衢总线 $\rightarrow$ 远端 IO Die $\rightarrow$ 远端 PCIe 5.0 $\rightarrow$ 远端 DDR | **~58-60 GB/s** | **远端单卡 PCIe 5.0 x16 物理吞吐**（灵衢总线 400GB/s 会在远端被截流） |

---

## 5. 超大规模模型（550B / 1.6T / 10T）部署切分策略与工程落地哲学

### 5.1 分级模型切分路线设计

* **550B 级（DeepSeek V4.1 Flash）**：
* **硬件配比**：单超节点（128 卡 950DT，8 台 PC16）。
* **并行范式**：Dense FSDP128 + EP64 / eFSDP2。高频 AllToAll 专家通信完全内聚在 400 GB/s 灵衢总线内，单层通信耗时被压低至 0.2ms 级。


* **1.6T 级（DeepSeek V4 Pro）**：
* **硬件配比**：2 个超节点（256 卡 950DT）。
* **并行范式**：**EP128 × FSDP2**。绝不使用跨节点 PP（流水并行），彻底消除流水残差通信与气泡（Bubble）。跨超节点仅使用 50 GB/s 以太网络执行低频 FSDP2 权重聚合，并依靠计算 Overlap 彻底平摊。


* **10T 级超大规模外推**：
* **硬件配比**：8 个超节点（1024 卡 950DT）。
* **并行范式**：**EP128 × FSDP4 × PP2**。在此规模下引入粗粒度 PP2，核心目的在于切断并削减单个 Stage 内临时聚合参数的显存峰值。



### 5.2 两种系统哲学的终极权衡

* **NVIDIA NVL72 的“单机柜极致暴力”模式**：
* 依托超高物理带宽（HBM 8 TB/s + NVLink 1.8 TB/s + C2C 900 GB/s），为开发者提供扁平无感的统一度量空间。但在万亿 MoE 超出 72 卡规模后，EP 通信必须穿透外部网络，对外部高密交换矩阵提出严苛挑战。


* **昇腾 950DT 超节点的“大拓扑域协同掩盖”模式**：
* 不盲目在单卡追求极致总线数值，而是利用灵衢总线直接拉出 **128 卡统一通信与内存 ScaleUp 域**，从结构上保证万亿级 MoE 的高频 Token 交换不跨出超节点；
* 面对 DDR $\leftrightarrow$ HBM 仅 ~60 GB/s 的客观物理约束，实施严格的显式存储分层：**HBM（18TB）专供高频前向反向计算与交换区，DDR（12TB）作为冷状态池承接优化器动量与 Engram 表**，利用后台异步 DMA 搬运与密集计算的时间窗口完全隐藏带宽短板，在更具工程可行性与自主可控的供应链条件下实现系统吞吐的全局最优。