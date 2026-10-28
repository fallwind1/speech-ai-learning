# W6D4：测试集标注 3

## 1. 候选队列

W6D4 候选数量：10条

候选 selection_rank：21-30

是否排除已校对样本：是

候选选择方法及随机 seed：
沿用已建立的固定候选队列，按照selection_rank顺序选择21-30
本次没有重新随机抽样。原始候选的sampling_method与sampling_seed保留在test_candidates.csv中，不应为本次批次重新编造随机seed。

## 2. 人工听音校对

新增校对数量：
10条

新增 verified 数量：
10条

needs_review 数量：
0

术语原文是否保留：
本次转写包含植物、动物分类名称及地理名称等专有词语，例如：

- 连山耳草
- 东湖尾孢虫
- 束花石斛
- 居延海沙地
- 萨尔加迪纽

已按人工提交的文本写入 raw_text。
这些术语的准确性应以实际音频核查结果为准，字段检查本身不能独立证明听写准确。

数字是否完整保留：
本批次提交的10条 raw_text 中没有明显的阿拉伯数字形式。

开头与结尾是否完整：
10条均已标记 verified，表示本次校对流程中未保留待处理状态。

边界不清晰的样本及原因：
本次没有提交具体的边界不清晰案例。

## 3. 文本字段

raw_text 是否忠实于人工听音：
10条样本的 raw_text 均已填写。其准确性依据人工听音校对，不应仅以字段非空作为判断标准。

score_text 是否遵循项目归一化规则：
10条样本的 score_text 均已填写。当前提交中，raw_text 与 score_text 内容一致。

不确定内容是否被正确标记：
本批次全部为 verified，没有留下 needs_review 状态。

是否存在未经核实的猜测：
本次提交未报告未经核实的内容。

## 4. 累计验收

累计 verified：
30条。

是否达到30条：
是

raw_text 是否完整：
W6D4本批10条均非空。

score_text 是否完整：
W6D4本批10条均非空。

review_minutes 是否记录：
10条全部记录。

候选 UID 是否重复：
此前对98条候选执行的 candidate_uid 重复检查结果为0。

## 5. 结果文件

test_candidates.csv：
metadata/test_candidates.csv

标注记录保存位置：
在 test_candidates.csv 中更新 review_batch、raw_text、score_text、status 和 review_minutes 等已有字段。