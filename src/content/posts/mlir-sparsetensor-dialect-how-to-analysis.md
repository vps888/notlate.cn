---
title: "MLIR中的SparseTensor方言是如何分析矩阵的稀疏性的？"
description: "在传统编程中， CSR 通常是手动维护的三个数组（ row ptr , col indices , values ）。但在 MLIR 中， 稀疏性 被设计成为 类型系统 (Type System)的一种属性，而不是具体的数据结构实现细节。 1. CSR是什么？ 在传统高性能计算和深度学习系统中…"
slug: "mlir-sparsetensor-dialect-how-to-analysis"
legacyId: 19525701
sourceUrl: "https://www.cnblogs.com/notlate-cn/p/19525701"
pubDate: 2026-01-24
category: "AI 编译器"
tags: ["AI 编译器","MLIR"]
featured: true
---

在传统编程中，`CSR` 通常是手动维护的三个数组（`row_ptr`, `col_indices`, `values`）。但在 MLIR 中，**稀疏性**被设计成为**类型系统**(Type System)的一种属性，而不是具体的数据结构实现细节。

## 1. CSR是什么？

在传统高性能计算和深度学习系统中，稀疏张量通常以固定格式存在，例如 CSR、CSC、COO、ELL、DIA 等。

而**CSR (Compressed Sparse Row)** 和 **CSC (Compressed Sparse Column)** 是稀疏矩阵（Sparse Matrix）最经典、最常用的两种存储格式。

它们的核心目的都是：**只存储非零元素**，从而节省内存并加速计算。区别在于它们是**行优先**还是**列优先**来组织数据。

为了方便理解，我们统一使用下面这个 $4 \times 4$ 的稀疏矩阵 $M$ 作为例子：

$$
M = \begin{pmatrix}
1 & 0 & 0 & 2 \\
0 & 3 & 0 & 0 \\
0 & 0 & 4 & 0 \\
5 & 6 & 0 & 7
\end{pmatrix}
$$

**矩阵属性**：

*   行数（Rows）：4
*   列数（Cols）：4
*   非零元素数量（NNZ, Number of Non-Zeros）：7
*   非零元素值：1, 2, 3, 4, 5, 6, 7

### 1.1 CSR (Compressed Sparse Row) - 压缩稀疏行

**核心思想**：按**行**顺序存储非零元素。类似于 C/C++ 的 Row-major 布局。

#### 存储结构（三个数组）

CSR 使用三个一维数组来表示矩阵：

1.  **`values` (数值数组)**：
    *   按**从左到右，从上到下**的顺序，存储所有非零元素的值。
    *   长度 = NNZ (7)   -- 表示非零元素个数
    *   **示例**：`[1, 2, 3, 4, 5, 6, 7]`

2.  **`col_indices` (列索引数组)**：
    *   对应 `values` 中每个元素的**列坐标**。
    *   长度 = NNZ (7)。
    *   **示例**：
        *   1 在第0列，2 在第3列 $\rightarrow$ `[0, 3]`
        *   3 在第1列 $\rightarrow$ `[1]`
        *   4 在第2列 $\rightarrow$ `[2]`
        *   5, 6, 7 分别在 0, 1, 3 列 $\rightarrow$ `[0, 1, 3]`
    *   **最终数组**：`[0, 3, 1, 2, 0, 1, 3]`

3.  **`row_ptr` (行偏移数组/行指针)**：**这是理解 CSR 的关键**
    *   存储每一行在 `values` 数组中的**起始位置（索引）**。
    *   长度 = 行数 + 1 (4 + 1 = 5)。
    *   最后一个元素通常存储 NNZ 的总数。
    *   **示例推导**：
        *   第 0 行起始于 index **0**。
        *   第 1 行起始于 index **2**（因为第 0 行有 2 个元素）。
        *   第 2 行起始于 index **3**（因为第 1 行有 1 个元素：$2+1=3$）。
        *   第 3 行起始于 index **4**（因为第 2 行有 1 个元素：$3+1=4$）。
        *   结束位置（第 4 行不存在）是 **7**（因为第 3 行有 3 个元素：$4+3=7$）。
    *   **最终数组**：`[0, 2, 3, 4, 7]`

#### 如何读取 CSR？

要恢复第 $i$ 行的数据：

1.  读取 `start = row_ptr[i]` 和 `end = row_ptr[i+1]`。
2.  遍历 `values` 和 `col_indices` 数组中下标从 `start` 到 `end-1` 的部分。

#### 优点与场景

*   **行切片（Row Slicing）极快**：可以瞬间定位到某一行。
*   **SpMV（稀疏矩阵-向量乘法）高效**：$Ax = y$ 计算中，计算 $y$ 的第 $i$ 个元素只需遍历矩阵的第 $i$ 行，与 CSR 内存布局完美契合。
*   **深度学习主流**：PyTorch (`torch.sparse_csr_tensor`), SciPy 等默认多用 CSR。

### 1.2 CSC (Compressed Sparse Column) - 压缩稀疏列

**核心思想**：按**列**顺序存储非零元素。类似于 Fortran 或 MATLAB 的 Column-major 布局。

#### 存储结构（三个数组）

CSC 也使用三个数组，逻辑与 CSR 对称：

1.  **`values` (数值数组)**：
    *   按**从上到下，从左到右**的顺序（即先读第一列，再读第二列...），存储非零元素。
    *   **示例**：
        *   第 0 列：1, 5
        *   第 1 列：3, 6
        *   第 2 列：4
        *   第 3 列：2, 7
    *   **最终数组**：`[1, 5, 3, 6, 4, 2, 7]` （注意顺序变了）

2.  **`row_indices` (行索引数组)**：
    *   对应 `values` 中每个元素的**行坐标**。
    *   **示例**：
        *   1(行0), 5(行3) $\rightarrow$ `[0, 3]`
        *   3(行1), 6(行3) $\rightarrow$ `[1, 3]`
        *   4(行2) $\rightarrow$ `[2]`
        *   2(行0), 7(行3) $\rightarrow$ `[0, 3]`
    *   **最终数组**：`[0, 3, 1, 3, 2, 0, 3]`

3.  **`col_ptr` (列偏移数组/列指针)**：
    *   存储每一列在 `values` 数组中的**起始位置**。
    *   长度 = 列数 + 1 (5)。
    *   **示例推导**：
        *   第 0 列起始于 **0**。
        *   第 1 列起始于 **2**（第 0 列有 2 个元素）。
        *   第 2 列起始于 **4**（第 1 列有 2 个元素）。
        *   第 3 列起始于 **5**（第 2 列有 1 个元素）。
        *   结束位置 **7**。
    *   **最终数组**：`[0, 2, 4, 5, 7]`

#### 优点与场景

*   **列切片（Column Slicing）极快**。
*   **特定算法优势**：在某些线性代数求解器（如 LU 分解）或图算法中，按列访问更自然。
*   **MLIR 视角**：CSC 其实就是 CSR 的转置（维度顺序交换）。

### 1.3 CSR vs CSC 对比总结

| 特性          | CSR (行压缩)                       | CSC (列压缩)                                    |
| :------------ | :--------------------------------- | :---------------------------------------------- |
| **遍历顺序**  | 先行后列                           | 先列后行                                        |
| **指针数组**  | `row_ptr` (大小 = 行数+1)          | `col_ptr` (大小 = 列数+1)                       |
| **索引数组**  | `col_indices` (存列号)             | `row_indices` (存行号)                          |
| **访问优势**  | 快速获取**某一行**的所有非零元     | 快速获取**某一列**的所有非零元                  |
| **典型应用**  | 矩阵-向量乘法 (SpMV), 深度学习推理 | 矩阵-矩阵乘法 (SpGEMM) 的一部分, 科学计算求解器 |
| **MLIR 映射** | `(d0: dense, d1: compressed)`      | `(d1: dense, d0: compressed)`                   |

---

## 2. MLIR中如何表达CSR（TACO理论）？

在编译器层面表达稀疏矩阵存储方式存在根本性问题：

- 稀疏格式是“名字级”的抽象，而非可组合的语义模型
- 算法与格式强耦合，新格式需要手写新 kernel
- 难以支持混合稀疏（如 Block + Sparse）
- 编译器无法统一分析、变换和优化稀疏计算

`TACO (Tensor Algebra Compiler) ` 的目标正是解决这一问题：

>  **用统一的张量代数和存储抽象，描述所有稀疏/稠密格式，并由编译器自动生成高效代码。**

其核心思想是：

>  **张量的"格式"不是一个整体概念，而是由每个维度对应的"层级（Level）"及其存储与遍历语义共同决定的。**

这被称为 **Level-based Sparse Tensor Model（基于维度层级的稀疏张量模型）**。

### 2.1 维度层级（Dimension Level）理论

#### 从维度到层级

数学上，一个 N 维张量表示为：
$$
[ A \in \mathbb{R}^{I_0 \times I_1 \times \cdots \times I_{n-1}} ]
$$
TACO 并未停留在"维度大小"这一层面，而是将每个维度映射为一个 **Level**：

```
Dimension 0 → Level 0
Dimension 1 → Level 1
...
```

关键不在映射本身，而在于：

>  **每个 Level 都可以独立选择其存储格式和遍历规则。**

#### Level Format（层级格式）

TACO 定义了一组基础的 Level 类型，用于描述稀疏或稠密结构：

| Level 类型 | 含义                               |
| ---------- | ---------------------------------- |
| Dense      | 该维度是连续稠密的                 |
| Compressed | 稀疏维度，使用 offset + index 表示 |
| Singleton  | 每个父坐标只有一个子坐标           |
| Hashed     | 使用哈希表存储                     |
| Ordered    | 子坐标有序                         |
| Unordered  | 子坐标无序                         |

这些 Level 类型可以**自由组合**，从而表达复杂的稀疏结构。

#### Level = 存储 + 遍历语义

每个 Level 同时定义两件事：

1. **存储方式**：是否显式存储坐标（如 `pos` / `crd` 数组）
2. **遍历语义**：给定父 Level 的一个坐标，如何枚举子 Level 的坐标集合

这使得 TACO 能够在不依赖具体“格式名称”的前提下，推导出正确的循环结构。

### 2.2 CSR 在 TACO 理论中的定义

#### 传统视角下的 CSR

CSR（Compressed Sparse Row）通常被描述为：

- `row_ptr`：行偏移
- `col_indices`：列索引
- `values`：非零值

这是一种**内存布局定义**。

#### TACO 视角下的 CSR

在 TACO 中，并不存在“CSR”这个一等概念。CSR 被视为一个二维张量，其层级配置为：

```
Level 0（row）: Dense
Level 1（col）: Compressed
```

即：

> **CSR = Dense × Compressed**

这是一个纯粹的语义描述，而非格式名称。

#### 存储含义

| Level | 类型       | 存储含义                    |
| ----- | ---------- | --------------------------- |
| L0    | Dense      | 行索引隐式存在（0..nrow-1） |
| L1    | Compressed | `pos` / `crd` / `values`    |

这与传统 CSR 的内存结构完全等价，但抽象层次更高。

### 2.3 TACO 的张量代数与访问路径

TACO 从数学表达式出发，例如：
$$
[ C_{ij} = \sum_k A_{ik} B_{kj} ]
$$
关键差异在于：

> **元素访问不再是直接的索引操作，而是由 Level 遍历驱动。**

访问 `A(i, k)` 的语义是：

```
for i in Level0(A):
  for k in children(Level1(A), i):
    use A(i, k)
```

循环结构完全由 Level Format 推导。

### 2.4 Merge Lattice：稀疏计算的核心机制

#### 稀疏计算的本质

TACO 观察到：

> **所有稀疏张量计算，本质上都是多个 Level iterator 的合并（merge）问题。**

例如：

- SpMV：合并 row 与 col iterator
- SpMM：在共享维度上做 intersection
- Masked compute：条件化 merge

#### Merge Lattice

TACO 构建了一个 **Merge Lattice**，用于系统性地决定：

- 循环嵌套结构
- `if` 条件
- iterator 的推进顺序

这正是编译器能够自动生成高效稀疏循环代码的理论基础。

### 2.5 TACO 理论如何映射到 MLIR SparseTensor

#### Encoding Attribute 的含义

在 MLIR 中，一个 CSR 张量可以表示为：

```text
tensor<1024x1024xf32,
  #sparse_tensor.encoding<
    { dimLevelType = ["dense", "compressed"] }
  >
>
```

这里的 `dimLevelType` 几乎是 **TACO Level Model 的直接映射**。

#### TACO 与 MLIR 的对应关系

| TACO 概念     | MLIR SparseTensor       |
| ------------- | ----------------------- |
| Level         | Dimension               |
| Level Format  | `dimLevelType`          |
| pos / crd     | Sparse storage spec     |
| Merge Lattice | Sparsification pass     |
| Codegen       | `scf` / `llvm` lowering |

MLIR 编译器并不“认识 CSR”，而是理解 **Level 组合的语义**。

### 2.6 更复杂的示例：Blocked CSR

使用 TACO/MLIR 的 Level Model，可以自然表示 BSR：

```
L0: Dense        (block row)
L1: Compressed   (block col)
L2: Dense        (in-block row)
L3: Dense        (in-block col)
```

无需引入新的 IR 或专用 kernel，这正是 Level-based 模型的威力所在。

### 2.7 总结

TACO 理论的核心贡献在于：

> **它将稀疏张量格式从“具体内存布局”提升为“按维度分层的、可组合的存储与遍历语义模型”。**

在这一模型下：

- CSR、CSC、COO 等只是 Level 组合的特例
- 稀疏计算被统一为 Level iterator 的 merge 问题
- 编译器能够自动生成正确且高效的稀疏代码

MLIR SparseTensor Dialect 正是这一理论在现代编译器基础设施中的工程化落地。


---

## 3. MLIR 示例

在 MLIR 中，CSR 不是通过手动分配内存创建的，而是通过给 Tensor 类型附加一个 **Encoding Attribute（编码属性）** 来定义的。

编译器利用 **TACO的理论，通过**维度层级**(Level Formats)来描述稀疏格式。

### MLIR 代码解析

```cpp
// 1. 定义 CSR 编码属性
// map: 定义维度的存储方式
// d0 (行): dense (稠密/不压缩) -> 意味着每一行都存在，对应 CSR 的 row_ptr 数组
// d1 (列): compressed (压缩)   -> 意味着只存储非零元素，对应 col_ind 和 values
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// 2. 使用该属性定义 Tensor 类型
// 编译器看到这个类型，就知道它在内存中不是一块连续的 float 数组，
// 而是由特定的元数据（metadata）和值数组组成的结构。
%sp_mat: tensor<?x?xf32, #CSR>
```

### 数据是如何“灌入”这个格式的？

在运行时，通常通过 **Conversion（转换）** 操作将数据从外部格式（如稠密 Tensor 或 COO 列表）转换为 MLIR 的稀疏格式。

```cpp
// 假设 %dense_data 是一个标准的稠密张量 (包含很多 0)
%dense_data = ... : tensor<10x10xf32>

// 使用 convert 操作将数据 "Pack" 进 CSR 格式
// 这一步，编译器生成的代码会扫描 %dense_data，
// 丢弃 0 值，构建 row_ptr, col_ind 和 values 数组，
// 并返回一个指向这些结构的 opaque 指针 (%sparse_data)。
%sparse_data = sparse_tensor.convert %dense_data 
    : tensor<10x10xf32> to tensor<10x10xf32, #CSR>
```

---

## 4. 编译器如何知道哪些是稀疏的？

编译器在**编译期（Compile Time）**并不知道具体的数值（哪一个元素是 0），它知道的是**结构（Structure）**。具体的非零元素位置是在 **Runtime** 阶段通过读取 CSR 的元数据数组确定的。

MLIR 的稀疏编译器（Sparse Compiler）工作流程如下：

### 4.1. 静态分析（类型驱动）

编译器看到 `tensor<..., #CSR>`，通过解析 `#CSR` 属性：

1.  **第 0 维是 `dense`**：编译器知道需要生成一个从 `0` 到 `N` 的标准 `for` 循环（遍历行）。
2.  **第 1 维是 `compressed`**：编译器知道不能生成标准 `for (j=0; j<M)` 循环，而是必须生成一个**间接访问循环**。它需要读取 CSR 的 `row_ptr` 数组来获取当前行的起止位置，然后遍历 `col_ind`。

### 4.2. 代码生成（Lowering）

当 `linalg.generic` 遇到稀疏类型时，编译器会自动将通用的循环逻辑“重写”为稀疏迭代逻辑。

**伪代码对比：**

**如果是稠密矩阵 (Dense):**

```cpp
// 编译器生成的代码
for (int i = 0; i < N; i++) {       // d0: dense
  for (int j = 0; j < M; j++) {     // d1: dense
    float val = A[i * M + j];
    compute(val);
  }
}
```

**如果是稀疏矩阵 (CSR):**

```cpp
// 编译器根据 #CSR 属性自动生成的代码
// d0: dense -> 标准循环
for (int i = 0; i < N; i++) {
  // d1: compressed -> 查表循环
  // 编译器自动插入读取 metadata 的代码
  int start = pointers[i]; // row_ptr[i]
  int end   = pointers[i+1];
  
  for (int p = start; p < end; p++) {
    int j = indices[p];    // col_ind[p] -> 真实的列坐标
    float val = values[p]; // 真实的非零值
    
    // 此时编译器"知道"坐标 (i, j) 处有值 val
    compute(val);
  }
}
```

## 总结

1.  **创建方式**：通过 `#sparse_tensor.encoding` 属性声明类型，并在运行时通过 `sparse_tensor.convert` 或 `sparse_tensor.new` 算子将数据“打包”成该格式。
2.  **识别稀疏性**：
    *   **编译期**：编译器通过属性（`dense` vs `compressed`）决定生成哪种类型的循环（直接遍历 vs 查表遍历）。
    *   **运行时**：生成的代码通过读取底层的 `pointers` 和 `indices` 数组，精确地跳过零值，只访问非零元素。

这种设计的强大之处在于，如果你想换成 **CSC (列压缩)**，只需要改一行代码（修改 map 映射顺序），编译器就会自动重新生成完全不同的遍历循环，而无需手动重写算法。
