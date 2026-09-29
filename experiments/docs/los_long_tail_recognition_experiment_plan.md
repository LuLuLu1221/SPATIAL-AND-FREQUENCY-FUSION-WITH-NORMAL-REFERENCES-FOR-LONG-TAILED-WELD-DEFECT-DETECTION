# LOS（ICLR 2025）长尾识别与检测迁移实验方案

## 1. 术语核验与实验边界

LOS 指 ICLR 2025 论文 **Rethinking Classifier Re-Training in Long-Tailed Recognition: Label Over-Smooth Can Balance**，其全称为 **Label Over-Smooth**。LOS 是图像**分类**方法，不是目标检测方法。

其公开代码的基准为：

| 数据集 | 任务 | 标签形式 | LOS 设置 |
|---|---|---|---|
| CIFAR100-LT | 单标签图像分类 | 每图一个类别 | 人工长尾，imbalance ratio `10/50/100` |
| ImageNet-LT | 单标签图像分类 | 每图一个类别 | 自然长尾 |
| iNaturalist 2018 | 单标签图像分类 | 每图一个类别 | 自然长尾 |

我们的核心模型是 D-FINE 检测器，并依赖 detection query、候选框以及 IBO 的内部—边界—外部区域证据。因此，**不能将完整的 D-FINE + IBO + 频率证据模型直接运行在 LOS 的分类数据集上**：这些数据没有目标框，无法构造候选内部、边界环和外部上下文，也无法计算检测 AP。

本方案据此拆成三条相互独立的实验线，任何论文表格均不得混报。

| 轨道 | 目的 | 数据 | 是否使用完整核心方法 |
|---|---|---|---|
| L1：LOS 复现 | 确认 LOS 官方方法与代码在分类任务上的行为 | CIFAR100-LT，后续 ImageNet-LT/iNat | 否，仅 LOS |
| L2：LOS 思想检测迁移 | 测试 label over-smoothing 对 D-FINE 检测分类头的作用 | LVIS v1 | 否，仅 LOS 风格分类头消融 |
| L3：核心方法主实验 | 测试 IBO、频率、可靠性融合和 query 纠错 | LVIS v1 与焊接数据 | 是 |

L1 的结果只能证明 LOS 的分类复现；L2 只能证明 LOS 风格分类头重训练是否有助于检测；L3 才是本文方法的主证据。

## 2. LOS 方法概述

LOS 采用解耦训练：

1. **Stage 1：表征学习**。使用原始长尾训练分布学习 backbone 与分类器。
2. **Stage 2：分类器重训练**。固定或主要固定 Stage 1 学到的表征，对分类器进行再训练。
3. **Label Over-Smooth**。不再使用 one-hot 标签，也不同于传统仅对正确类占主导的 label smoothing。设类别总数为 `K`、过平滑系数为 `s`，正确类目标概率略高于 `1/K`，其余类别略低于 `1/K`，即整体接近均匀分布。

LOS 试图平衡各类别的 logits magnitude，并降低训练扰动，而不是通过显式类别频次权重增大尾类梯度。

## 3. 研究问题与假设

### RQ-L1：LOS 是否可复现？

在 CIFAR100-LT 的 `IR=10/50/100` 上，LOS 的两阶段训练和 label over-smoothing 是否能稳定改善 many/medium/few 类别的 Top-1 accuracy？

### RQ-L2：LOS 的分类头机制能否迁移至检测？

将 LOS 风格目标分布施加在 D-FINE 的 query 分类头重训练阶段，能否提升 LVIS rare 类 `AP_r`，同时维持 frequent 类 `AP_f`？

### RQ-L3：LOS 与本文方法是否互补？

LOS 调整全局分类决策边界；本文方法利用候选级空间—频率证据、可靠性融合和局部混淆纠错。二者是否能产生互补增益，还是 LOS 已能解释主要收益？

## 4. L1：LOS 官方分类复现实验

### 4.1 数据与优先级

先复现 CIFAR100-LT；它规模小、公开代码直接支持 `IR=10/50/100`，可先验证环境与两阶段流程。ImageNet-LT 和 iNaturalist2018 作为后续扩展，不与当前 D-FINE/LVIS 训练并行执行。

| 优先级 | 数据集 | 目的 | 资源策略 |
|---|---|---|---|
| P0 | CIFAR100-LT IR=100 | LOS 机制核验 | 先执行 |
| P1 | CIFAR100-LT IR=10, 50 | 检查不同失衡强度 | P0 成功后执行 |
| P2 | ImageNet-LT | 大规模自然长尾验证 | LVIS 主实验完成后执行 |
| P3 | iNaturalist2018 | 超大类别数自然长尾验证 | 仅在有明确论文需求时执行 |

### 4.2 L1 实验组

| 组别 | Stage 1 | Stage 2 分类目标 | 目的 |
|---|---|---|---|
| C0 | 官方默认 | 不重训练 | Stage 1 基线 |
| C1 | 与 C0 相同 | 常规交叉熵重训练 | 分离“分类器重训练”贡献 |
| C2 | 与 C0 相同 | 标准 label smoothing | 区分标准平滑与 LOS |
| C3 | 与 C0 相同 | LOS label over-smooth | 复现 LOS 主结果 |

`C0--C3` 必须共享同一 Stage 1 checkpoint，禁止每一组重新学习特征，以免分类器重训练效果被表征差异掩盖。

### 4.3 L1 指标与输出

- Overall Top-1 accuracy。
- Many-shot、medium-shot、few-shot Top-1 accuracy，组划分严格沿用 LOS/数据集约定。
- 每类 accuracy、类别平均 accuracy 和 worst-group accuracy。
- logit magnitude 的类别均值与标准差。
- LOS 论文定义的 Regularized Standard Deviation（RSD）；实现细节必须在复现前从论文/代码核对。
- 三个随机种子，报告均值 ± 标准差。

保存内容：数据长尾划分文件、随机种子、Stage 1 与 Stage 2 checkpoint、每类预测 CSV、logit 统计 CSV、完整训练日志与环境版本。

## 5. L2：将 LOS 分类头重训练迁移到 LVIS 检测

### 5.1 迁移原则

L2 不是“在 LOS 数据集上运行本文检测方法”，而是将 LOS 的**分类头再训练思想**置于正确的检测数据上测试。使用 LVIS v1，因其有边界框、长尾类别和官方 `AP_r/AP_c/AP_f` 指标。

使用已训练的 D-FINE-S A0 checkpoint：

1. 冻结 backbone、encoder、decoder 和 box regression 分支。
2. 保留 query-to-GT 匹配机制与背景/无目标 query 的处理。
3. 仅更新分类 head，或在预实验确认必要后更新分类 head 前的最后投影层；其余参数保持冻结。
4. 对 matched positive queries 使用 LOS 目标分布；背景 query 保持原始背景监督，不将背景视作均匀前景类别。
5. 使用独立的 validation split 调节 `s`，不得在 test/val 指标上反复选择后只报告最佳数值。

### 5.2 检测重训练的目标定义

对于 `K` 个前景类别、真类别 `y` 和 LOS 系数 `s`：

`t_y = s`，`t_c = (1-s)/(K-1), c != y`。

在实际实现前，需用 LOS 原论文/官方代码核对其确切参数化是否等价于上述形式；若不等价，以官方定义为准并在运行配置中保存公式。

检测分类损失应只替换正样本 query 的类别目标。box L1/GIoU/FGL、denoising 和背景损失保持 D-FINE 原定义，确保比较只反映分类边界变化。

### 5.3 L2 实验组

| 组别 | D-FINE 主训练 | 重训练方式 | 作用 |
|---|---|---|---|
| D0 | A0 baseline | 无 | 原始检测基线 |
| D1 | 与 D0 相同 | 冻结特征 + 原始分类损失重训练 | 分类头重训练基线 |
| D2 | 与 D0 相同 | 冻结特征 + 标准 label smoothing | 平滑基线 |
| D3 | 与 D0 相同 | 冻结特征 + LOS | LOS 检测迁移 |
| D4 | 与 D0 相同 | LOS + 本文 query 混淆纠错 | 检验两个分类层机制是否互补 |

初步阶段不把 IBO、频率和可靠性模块与 LOS 一并训练。待 D3 的独立结果明确后，才运行 D4；否则无法区分 LOS 与本文模块的贡献。

### 5.4 L2 主指标

- LVIS 官方：`AP_box`、`AP_r`、`AP_c`、`AP_f`、`AP50`、`AP75`、`AP_s/m/l`。
- 按 r/c/f 分组的 logit magnitude、logit 方差与类别校准指标 ECE/Brier。
- 高频混淆类别对的错选率。
- Candidate Recall 与 Wrong-class Rate：将“候选未出现”和“候选类别选错”分开。

预注册选择规则：优先最大化 `AP_r`，条件为 `AP_f >= AP_f(D0)-0.5` 个百分点；如不满足，报告为“尾类改善伴随头类代价”，不可写为整体改进。

## 6. L3：与本文完整方法的关系

| 模块 | 作用层级 | 是否属于 LOS | 是否属于本文核心方法 |
|---|---|---|---|
| Stage 2 classifier retraining | 全局分类头 | 是 | 否 |
| Label over-smooth | 类别决策边界 | 是 | 否 |
| IBO 内部—边界—外部证据 | 候选区域 | 否 | 是 |
| 局部频率证据 | 候选区域 | 否 | 是 |
| 可靠性融合 | 候选级双域证据 | 否 | 是 |
| Query 混淆类对比纠错 | 局部类别重排序 | 否 | 是 |

论文主线仍然是 L3。LOS 可作为有价值的对照：若 D3 提升有限而 A6 显著提升，支持“仅调整全局分类边界不足以处理候选级证据缺失和混淆”；若 D4 优于 D3 与 A5，则可表述二者具有互补性。

## 7. 执行顺序与资源安排

1. 等待当前 SimLTD/LVIS pilot10k 的 D-FINE-S 100 epoch 训练结束，避免显存与 CPU 内存竞争。
2. 下载/核验 LOS 官方仓库和 CIFAR100-LT，生成数据版本与 hash 记录。
3. 运行 L1-P0（CIFAR100-LT, IR=100, C0--C3）。
4. 固定 LOS 实现后，完成 LVIS A0 正式 D-FINE-S 训练。
5. 运行 L2 的 D0--D3，再决定是否运行 D4。
6. 完成 L3 的 A0--A6 核心消融。
7. 仅在有额外算力和明确论文需求时，扩展至 ImageNet-LT 与 iNaturalist2018。

不在当前 D-FINE 训练仍占用 GPU 时并行训练 LOS。CIFAR100-LT 虽小，但同时运行增加断训与内存错误风险。

## 8. 可发表结论边界

可以表述：

> LOS 的全局分类边界重训练可作为长尾检测中的分类头对照；本文方法进一步通过候选级区域证据、频率证据与可靠性融合处理仅靠分类头校准无法区分的漏检和局部混淆。

不应表述：

> 本文完整检测方法在 CIFAR100-LT/ImageNet-LT/iNaturalist2018 上优于 LOS。

原因是后三者为图像分类任务，没有检测框，无法承载完整的 IBO 或 D-FINE detection query 模块。

## 9. 参考与复现入口

- LOS 官方代码：`https://github.com/bilibili/LOS`
- LOS 官方设置：先以 `CIFAR100-LT, IR=100` 的 Stage 1/Stage 2 流程作为最小可复现单元。
- 检测迁移数据：LVIS v1 官方 train/val 与官方 evaluator。

