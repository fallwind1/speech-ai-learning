
# W8D1：Blank 与 CTC 路径折叠

## 1. CTC 的基本概念

### 1.1 什么是 CTC Path？
- CTC Path是模型在每一个时间帧上预测得到的Token序列
例如，模型经过若干时间帧的预测，得到：a a _ a _ b
其中：
- a、b：普通 Token。
- _：Blank Token。
CTC Path是模型的逐帧预测结果，而不是最终识别文本。需要经过路径折叠才能得到最终输出

### 1.2 什么是 Blank？
- Blank是CTC引入的特殊类别，表示当前时间帧不输出普通token。
- Blank 与普通 Token 的区别：
不属于最终识别文本，重要作用是允许相同两个Token在原始路径中被分隔开，从而使最终输出能够包含连续重复字符。

## 2. CTC 路径折叠（Collapse）

### 2.1 路径折叠的两个步骤

第一步：合并相邻重复 Token
- 将原始路径中连续出现的相同Token合并为一个

第二步：删除 Blank
- 将第一部处理结果中的所有Blank删除。

### 2.2 为什么不能全局去重？
- 因为两个相同Token之间可能存在Blank，Blank的存在意味着这两个Token在原始路径中并不相邻。

### 2.3 CTC 路径折叠规则
- 记原始路径为 (\pi)，最终输出为 (y)，则：

$$[
y=B(\pi)
]$$

其中 (B) 表示 CTC 路径折叠操作，具体包括：
1. 合并相邻重复 Token。
2. 删除 Blank。
两个步骤必须按照上述顺序执行。

## 3. 手算例子

### Example 1

原始路径：
a a _ a _ b

中间结果（合并相邻重复 Token）：
a _ a _ b

最终结果（删除 Blank）：
a a b

### Example 2

原始路径：
a _ a

中间结果（合并相邻重复 Token）：
a _ a

最终结果（删除 Blank）：
a a

### Example 3

原始路径：
a a b b _ b

中间结果（合并相邻重复 Token）：
a b _ b

最终结果（删除 Blank）：
a b b

## 4. 核心问题

### 4.1 为什么 a_blank_a 与 aa 的输出不同？

- 核心区别是两个相同Token在原始路径周是否相邻，第一个两个a在原始路径中之间存在blank，故不相邻

### 4.2 为什么删除 Blank 后不能再次合并重复 Token？

- 因为删除Blank会改变Token之间的相邻关系，原本被Blank分隔的两个相同Token，在删除Blank后会变成相邻Token。

## 5. 本模块总结

- CTC 使用逐帧预测得到 Path，再通过路径折叠得到最终文本。
路径折叠包含两个步骤：
1. 合并原始路径中的相邻重复Token
2. 删除Blank
Blank的核心作用是允许相同Token在最终输出中连续出现。