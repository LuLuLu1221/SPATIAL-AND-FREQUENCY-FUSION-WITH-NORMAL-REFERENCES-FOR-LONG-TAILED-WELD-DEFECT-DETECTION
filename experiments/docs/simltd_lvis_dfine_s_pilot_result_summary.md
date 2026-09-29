# SimLTD/LVIS 长尾检测实验阶段性总结：D-FINE-S 底座结果

## 1. 实验目的

本阶段实验的目的不是验证最终方法性能，而是先完成 SimLTD/LVIS 数据上的基础检测器训练，得到后续 IBO 区域证据、正常参照距离、频率证据和混淆类对比纠错所需的候选框生成器。

当前阶段只评估 D-FINE-S no-stretch 底座，不包含以下模块：

- IBO 内部-边界-外部区域证据；
- 正常参照距离；
- 局部频率证据；
- 可靠性融合；
- 混淆类对比纠错；
- 长尾覆盖度分布校准。

因此，本阶段结果应被理解为“pipeline 打通与底座候选框质量检查”，不能作为本文最终方法结果。

## 2. 数据集与划分

使用 LVIS v1 数据构造 SimLTD 对齐实验的 pilot 子集。

| 项目 | 设置 |
|---|---:|
| 原始数据集 | LVIS v1 |
| 类别数 | 1203 |
| 训练子集 | 10,000 张图像 |
| 验证子集 | 2,000 张图像 |
| 训练标注数 | 117,605 |
| 验证标注数 | 20,891 |
| 标注格式 | LVIS/COCO-style JSON |
| 类别编号 | 转换为 D-FINE 适配的 zero-based category id，范围 0–1202 |

数据位置：

- `D:\1\项目论文\public_datasets\LVIS\images\train2017`
- `D:\1\项目论文\public_datasets\LVIS\images\val2017`
- `D:\1\项目论文\public_datasets\LVIS\dfine_annotations_pilot10k`

## 3. 模型与训练设置

| 项目 | 设置 |
|---|---|
| 基础检测器 | D-FINE-S |
| 训练方式 | 只训练基础检测器 |
| 输入尺寸 | 640 × 640 |
| 图像变换 | no-stretch，最长边 resize 后 padding 到 640 |
| batch size | 4 |
| epoch | 3 |
| AMP | 启用 |
| GPU | NVIDIA RTX 4060 Laptop GPU，8GB |
| 参数量 | 11,102,201 |

配置文件：

- `D:\1\项目论文\zhwk_project\dfine_configs\dfine_hgnetv2_s_lvis_v1_no_stretch_640_pilot10k.yml`
- `D:\1\项目论文\zhwk_project\dfine_configs\lvis_v1_dfine_no_stretch_detection_pilot10k.yml`

输出目录：

- `D:\1\项目论文\zhwk_runs\simltd_lvis_dfine_s_no_stretch_640_pilot10k_bs4_20260916_seed0`

## 4. 训练完成情况

训练已经完成 3 个 epoch，并保存了完整 checkpoint。

| 文件 | 含义 |
|---|---|
| `checkpoint0000.pth` | epoch 0 checkpoint |
| `checkpoint0001.pth` | epoch 1 checkpoint |
| `checkpoint0002.pth` | epoch 2 checkpoint |
| `last.pth` | 最后一轮模型 |
| `best_stg1.pth` | 当前最佳模型 |
| `log.txt` | 训练与验证指标日志 |

当前最佳模型为：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_dfine_s_no_stretch_640_pilot10k_bs4_20260916_seed0\best_stg1.pth
```

最佳 epoch 为 epoch 1。

## 5. D-FINE-S 验证结果

### 5.1 COCO-style bbox 指标

`test_coco_eval_bbox` 的 12 个数值对应 COCO bbox 评估常用指标：

1. AP@[0.50:0.95]
2. AP@0.50
3. AP@0.75
4. AP small
5. AP medium
6. AP large
7. AR@1
8. AR@10
9. AR@100
10. AR small
11. AR medium
12. AR large

| Epoch | AP@[.50:.95] | AP50 | AP75 | APs | APm | APl | AR@1 | AR@10 | AR@100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000333 | 0.001140 | 0.000122 | 0.000033 | 0.000575 | 0.002121 | 0.003440 | 0.006295 | 0.007945 |
| 1 | **0.000645** | **0.002143** | **0.000238** | **0.000119** | 0.001021 | 0.003844 | 0.005415 | 0.009985 | 0.013001 |
| 2 | 0.000585 | 0.001905 | 0.000215 | 0.000084 | **0.001086** | **0.004312** | **0.006113** | **0.011292** | **0.014394** |

从 AP 指标看，epoch 1 是当前最佳；从召回类指标看，epoch 2 的 AR 略高，但 AP 没有继续提升。

### 5.2 最后一轮验证集检测统计

epoch 2 的验证统计如下：

| 指标 | 数值 |
|---|---:|
| F1 | 0.012137 |
| Precision | 0.351499 |
| Recall | 0.006175 |
| IoU | 0.005262 |
| TP | 129 |
| FP | 238 |
| FN | 20,762 |

## 6. 当前结果解读

当前 D-FINE-S 底座结果较低，主要原因包括：

1. 该实验只使用 `pilot10k` 子集，而不是完整 LVIS/SimLTD 训练集；
2. 训练只进行了 3 个 epoch，属于流程验证与候选框生成器预训练阶段；
3. LVIS 类别数高达 1203，长尾极强，少量训练轮数下尾类学习不足；
4. 当前模型未加入本文方法的关键模块，包括 IBO 区域证据、正常参照距离、频率证据与混淆类纠错；
5. no-stretch 设置保留了原图比例，更符合本文方法要求，但在短训练下可能收敛更慢。

因此，当前结果不用于宣称最终性能，只用于说明：

- 数据转换正确；
- D-FINE-S 可以在该数据配置下完成训练；
- no-stretch 训练流程已经打通；
- 已得到可用于后续候选框提取的 `best_stg1.pth`。

## 7. 与后续本文方法的关系

下一阶段应以 `best_stg1.pth` 作为候选框生成器，在候选区域上继续执行本文方法：

1. 使用 D-FINE-S 输出候选框；
2. 在候选框周围执行 150px 循环滑窗；
3. 构造 IBO 区域：
   - I：候选框内部；
   - B：候选框边界；
   - O：候选框外部邻近背景或正常区域；
4. 提取空间证据、正常参照距离和局部频率证据；
5. 对混淆类别进行对比纠错；
6. 使用 SimLTD 论文一致的评价指标，并补充本文自己的长尾指标。

## 8. 阶段性结论

本阶段已完成 D-FINE-S 在 SimLTD/LVIS pilot10k 子集上的 no-stretch 训练。实验成功产出 `best_stg1.pth`，可作为后续 IBO 与双域证据模块的基础候选框生成器。

当前性能仍处于很低水平，说明仅依赖基础检测器难以处理 LVIS/SimLTD 的强长尾分布。这也为后续引入 IBO 区域证据、正常参照距离、频率证据和混淆类纠错提供了实验动机。

后续实验不应把当前 AP 作为最终方法性能，而应将其作为 D-FINE-S baseline/pilot 结果，与加入本文模块后的结果进行对比。

## 9. IBO 候选区域构建结果（验证集）

在 D-FINE-S `best_stg1.pth` 基础上，已完成 pilot10k 验证集的 IBO 候选区域构建。

本阶段修正了旧 IBO 脚本的推理预处理：旧脚本直接将图像 resize 到 640 × 640，会造成拉伸；本次 SimLTD/LVIS 实验使用与训练一致的 no-stretch 方式，即最长边 resize 到 640 后补边，再将预测框从 padded 640 坐标回投到原图坐标后裁剪 IBO crop。

IBO 专用脚本：

```text
D:\1\项目论文\zhwk_project\scripts\build_ibo_lvis_nostretch_from_dfine_candidates.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_nostretch_pilot10k_20260916\val_full_thr025
```

### 9.1 IBO 构建设置

| 项目 | 设置 |
|---|---:|
| split | val |
| 验证图像数 | 2,000 |
| GT 实例数 | 20,891 |
| 候选阈值 | 0.25 |
| 高置信阈值 | 0.40 |
| 匹配 IoU | 0.50 |
| GT 覆盖阈值 | 0.20 |
| context scale | 2.0 |
| min crop size | 32 |
| batch size | 4 |
| 导出 IBO crop 数 | 43,882 |

### 9.2 候选覆盖率

| 指标 | 覆盖实例数 | 覆盖率 |
|---|---:|---:|
| 低阈值同类候选覆盖，GT 覆盖 ≥ 0.20 | 3,801 / 20,891 | 18.19% |
| 低阈值任意类别候选覆盖，GT 覆盖 ≥ 0.20 | 10,098 / 20,891 | 48.34% |
| 低阈值同类候选 IoU ≥ 0.50 | 2,725 / 20,891 | 13.04% |
| 低阈值任意类别候选 IoU ≥ 0.50 | 4,534 / 20,891 | 21.70% |
| 高置信同类候选覆盖，GT 覆盖 ≥ 0.20 | 435 / 20,891 | 2.08% |
| 高置信任意类别候选覆盖，GT 覆盖 ≥ 0.20 | 1,066 / 20,891 | 5.10% |

### 9.3 候选框分布

| 项目 | 数值 |
|---|---:|
| 总候选数 | 43,882 |
| 高置信候选数 | 1,792 |
| 召回保护候选数 | 42,090 |
| 分数最小值 | 0.250004 |
| 分数中位数 | 0.284969 |
| 分数 P90 | 0.360909 |
| 分数 P95 | 0.390933 |
| 分数 P99 | 0.457228 |
| 分数最大值 | 0.804589 |

候选框按匹配结果统计：

| 候选结果 | 数量 |
|---|---:|
| correct | 11,210 |
| wrong_class | 10,670 |
| overlaps_other_class | 5,549 |
| partial_same_class | 2,824 |
| background | 13,629 |

### 9.4 阶段性观察

低阈值“任意类别候选覆盖率”为 48.34%，明显高于“同类候选覆盖率”的 18.19%。这说明当前 D-FINE-S 底座并非完全无法定位目标区域；相当一部分 GT 能被某些候选框覆盖，但候选类别不正确或置信度不足。

因此，后续 IBO 模块应重点解决两个问题：

1. 对已经覆盖到目标区域但类别错误的候选框，利用 IBO 区域证据、正常参照距离和频率证据进行重分类；
2. 对高置信候选覆盖率过低的问题，保留 recall-protect 低阈值候选，再通过可靠性融合和混淆类纠错过滤背景候选。

这与本文方法设定一致：D-FINE-S 在此阶段主要作为候选区域生成器，而不是最终分类器。

## 10. IBO 候选区域构建结果（训练集）

在验证集 IBO 构建完成后，使用相同设置继续构建 pilot10k 训练集 IBO 候选区域。

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_nostretch_pilot10k_20260916\train_full_thr025
```

### 10.1 train split 构建设置

| 项目 | 设置 |
|---|---:|
| split | train |
| 训练图像数 | 10,000 |
| GT 实例数 | 117,605 |
| 候选阈值 | 0.25 |
| 高置信阈值 | 0.40 |
| 匹配 IoU | 0.50 |
| GT 覆盖阈值 | 0.20 |
| context scale | 2.0 |
| min crop size | 32 |
| batch size | 4 |
| 导出 IBO crop 数 | 221,120 |

### 10.2 train split 候选覆盖率

| 指标 | 覆盖实例数 | 覆盖率 |
|---|---:|---:|
| 低阈值同类候选覆盖，GT 覆盖 ≥ 0.20 | 21,659 / 117,605 | 18.42% |
| 低阈值任意类别候选覆盖，GT 覆盖 ≥ 0.20 | 58,100 / 117,605 | 49.40% |
| 低阈值同类候选 IoU ≥ 0.50 | 15,151 / 117,605 | 12.88% |
| 低阈值任意类别候选 IoU ≥ 0.50 | 24,415 / 117,605 | 20.76% |
| 高置信同类候选覆盖，GT 覆盖 ≥ 0.20 | 2,760 / 117,605 | 2.35% |
| 高置信任意类别候选覆盖，GT 覆盖 ≥ 0.20 | 7,279 / 117,605 | 6.19% |

### 10.3 train split 候选框分布

| 项目 | 数值 |
|---|---:|
| 总候选数 | 221,120 |
| 高置信候选数 | 9,936 |
| 召回保护候选数 | 211,184 |
| 分数最小值 | 0.250000 |
| 分数中位数 | 0.285921 |
| 分数 P90 | 0.363947 |
| 分数 P95 | 0.395344 |
| 分数 P99 | 0.464915 |
| 分数最大值 | 0.785006 |

候选框按匹配结果统计：

| 候选结果 | 数量 |
|---|---:|
| correct | 62,419 |
| wrong_class | 57,301 |
| overlaps_other_class | 26,207 |
| partial_same_class | 14,396 |
| background | 60,797 |

### 10.4 train/val 一致性观察

train 与 val 的候选覆盖模式高度一致：

| split | 同类覆盖率 | 任意类别覆盖率 | 同类 IoU50 | 任意类别 IoU50 |
|---|---:|---:|---:|---:|
| train | 18.42% | 49.40% | 12.88% | 20.76% |
| val | 18.19% | 48.34% | 13.04% | 21.70% |

这说明当前现象不是验证集偶然波动，而是 D-FINE-S pilot 底座在 LVIS/SimLTD 长尾设置下的稳定行为：候选框经常覆盖到目标区域，但类别证据不可靠，且高置信候选覆盖不足。因此下一阶段应优先训练一个候选级证据融合/重打分模块，而不是继续单纯提高 D-FINE 分类阈值。

后续实验将以 train split 的 221,120 个 IBO crop 训练候选级证据融合模型，并在 val split 的 43,882 个 IBO crop 上评估其对 correct、wrong-class、background 候选的区分能力。

## 11. IBO 正常参照—频率证据可靠性融合结果

在 train/val IBO crop 构建完成后，训练 LVIS/SimLTD no-stretch 专用的候选级证据融合模型。该模型不使用环形焊道 ROI，不做 cyclic merge，而是直接读取 IBO crop，提取：

- I/B/O 空间过渡证据；
- Haar 局部频率过渡证据；
- training background IBO candidates 构成的正常参照距离；
- D-FINE 候选分数与候选框几何信息；
- ExtraTrees 可靠性融合分数。

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\train_lvis_ibo_evidence_fusion.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_evidence_fusion_20260916\full_train221120_val43882_resize256_r1
```

### 11.1 训练与验证规模

| 项目 | 数值 |
|---|---:|
| train candidates | 221,120 |
| val candidates | 43,882 |
| train positive / negative | 76,815 / 144,305 |
| val positive / negative | 14,034 / 29,848 |
| normal reference samples | 12,000 |
| normal reference source | training background IBO candidates |

### 11.2 候选级可靠性融合指标

| 指标 | 数值 |
|---|---:|
| Val AUROC | 0.682308 |
| Val AUPR | 0.520936 |
| Val Brier score | 0.205746 |

候选级最佳 F1 阈值为 0.40：

| 指标 | 数值 |
|---|---:|
| Precision | 0.435352 |
| Recall | 0.658116 |
| F1 | 0.524043 |
| TP | 9,236 |
| FP | 11,979 |
| TN | 17,869 |
| FN | 4,798 |
| selected candidates | 21,215 |

### 11.3 解释

候选级结果说明：IBO 空间过渡、频率证据和正常参照距离能够对 D-FINE 候选框可靠性形成有效区分。AUPR 从正样本比例约 31.98% 提升到 52.09%，说明融合证据确实增加了 correct/partial_same_class 候选与 wrong/background 候选之间的可分性。

但该阶段仍是候选级可靠性评估，不等价于最终检测 AP。它只判断“已有候选是否可靠”，不能生成 D-FINE 完全没有提出的新候选。

## 12. IBO 可靠性分数回投验证集评估

在候选级融合完成后，将 `reliability_fusion_score` 回投到 val split 的实例级 GT 上，比较原始 D-FINE 高置信候选与 IBO reliability 筛选候选的 correct / wrong / miss。

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_reliability_backproject_20260916\val_full_thr_sweep
```

评估口径：

- 同类候选覆盖 GT 面积 ≥ 0.20，记为 correct；
- 无同类正确覆盖，但存在异类候选覆盖，记为 wrong；
- 否则记为 miss。

### 12.1 总体实例级结果

| 方法 | 阈值 | 候选数 | correct / wrong / miss | 实例准确率 | image correct / wrong / miss | 图片准确率 |
|---|---:|---:|---:|---:|---:|---:|
| D-FINE score baseline | 0.40 | 1,792 | 433 / 626 / 19,832 | 2.07% | 26 / 29 / 1,926 | 1.31% |
| IBO reliability 最佳实例阈值 | 0.15 | 43,859 | 3,761 / 6,132 / 10,998 | 18.00% | 124 / 420 / 1,437 | 6.26% |
| IBO reliability 最佳图片阈值 | 0.35 | 29,163 | 3,096 / 5,300 / 12,495 | 14.82% | 129 / 362 / 1,490 | 6.51% |

### 12.2 阈值扫描观察

| reliability threshold | selected | correct | wrong | miss | instance accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 43,882 | 3,761 | 6,132 | 10,998 | 18.00% |
| 0.15 | 43,859 | 3,761 | 6,132 | 10,998 | 18.00% |
| 0.20 | 43,633 | 3,759 | 6,126 | 11,006 | 17.99% |
| 0.30 | 36,976 | 3,523 | 5,804 | 11,564 | 16.86% |
| 0.35 | 29,163 | 3,096 | 5,300 | 12,495 | 14.82% |
| 0.40 | 21,215 | 2,564 | 4,586 | 13,741 | 12.27% |
| 0.50 | 9,503 | 1,516 | 2,886 | 16,489 | 7.26% |

### 12.3 阶段性结论

与原始 D-FINE 高置信候选相比，IBO reliability 回投显著提高了实例级 correct 数量：

```text
433 -> 3,761
```

同时 miss 数量明显下降：

```text
19,832 -> 10,998
```

但 wrong 数量也从 626 增加到 6,132。这说明当前融合证据主要提升了“候选召回与可靠性识别”，但还没有充分解决类别混淆问题。

因此下一阶段不宜只继续降低阈值，而应进入：

1. 混淆类对比纠错；
2. 类别/频率/正常参照条件下的阈值校准；
3. 头类约束下的尾类 AP 或尾类召回优化。

这与本文路线一致：IBO 与正常参照频率证据先提升候选可靠性，随后需要分布校准与混淆纠错来压低 wrong-class。

## 13. 类别条件阈值校准实验

为进一步压低 wrong-class，本阶段尝试用 train split 的候选级标签学习类别条件 reliability 阈值，再应用到 val split 做实例级回投评估。

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\calibrate_lvis_ibo_class_thresholds.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_class_threshold_calibration_20260916\train_learned_val_eval
```

### 13.1 策略对比

| 策略 | 候选数 | correct / wrong / miss | 实例准确率 |
|---|---:|---:|---:|
| global_default，阈值 0.15 | 43,859 | 3,761 / 6,132 / 10,998 | 18.00% |
| global_f1 | 3,598 | 751 / 1,302 / 18,838 | 3.59% |
| global_f0.5 | 1,940 | 474 / 779 / 19,638 | 2.27% |
| global_precision_guard | 1,940 | 474 / 779 / 19,638 | 2.27% |
| class_f1 | 5,852 | 1,171 / 1,728 / 17,992 | 5.61% |
| class_f0.5 | 3,713 | 855 / 1,017 / 19,019 | 4.09% |
| class_precision_guard | 3,713 | 855 / 1,017 / 19,019 | 4.09% |

### 13.2 结论

类别条件阈值校准没有超过全局低阈值策略。原因是当前 D-FINE pilot 底座和 IBO reliability 的主要收益来自 recall-protect 候选。如果按 train split 学到较保守的类别阈值，wrong 会下降，但 correct 下降更明显，导致实例级准确率下降。

这说明：当前问题不是“每个类别设一个更高阈值”就能解决，而是大量候选已经覆盖到目标区域，但类别标签不可靠。下一步应从候选框筛选转向候选框重分类，即：

1. 对 wrong_class / overlaps_other_class 候选分析混淆类别对；
2. 对高频混淆类别训练二阶段 class corrector；
3. 在 correction 后再重新做 reliability 阈值与实例级回投。

## 14. IBO reliability 后的高频混淆类别对

在类别阈值校准未能提升总体结果后，进一步统计 reliability ≥ 0.15 的 val 候选中，预测类别与 GT 类别不一致但存在覆盖关系的高频混淆对。

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\analyze_lvis_ibo_confusion_pairs.py
```

输出文件：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_ibo_class_threshold_calibration_20260916\train_learned_val_eval\confusion_pairs_reliability_ge_0p15.csv
```

### 14.1 Top-20 混淆对

| 排名 | 预测类别 | GT 类别 | 次数 |
|---:|---|---|---:|
| 1 | cow | sheep | 615 |
| 2 | sheep | cow | 312 |
| 3 | cabinet | drawer | 302 |
| 4 | tomato | carrot | 295 |
| 5 | cabinet | refrigerator | 282 |
| 6 | apple | banana | 256 |
| 7 | apple | doughnut | 244 |
| 8 | laptop_computer | monitor_(computer_equipment) computer_monitor | 241 |
| 9 | tomato | apple | 236 |
| 10 | sheep | elephant | 227 |
| 11 | cow | elephant | 224 |
| 12 | cabinet | mirror | 212 |
| 13 | pillow | sofa | 188 |
| 14 | horse | cow | 184 |
| 15 | elephant | cow | 170 |
| 16 | elephant | sheep | 164 |
| 17 | cabinet | handle | 162 |
| 18 | horse | sheep | 157 |
| 19 | clock | clock_tower | 147 |
| 20 | orange_(fruit) | doughnut | 141 |

### 14.2 阶段性判断

错分高度集中在外观、上下文或语义相近类别之间，例如 cow / sheep / horse / elephant，cabinet / drawer / refrigerator / mirror / handle，以及 apple / tomato / banana / doughnut 等。

因此，下一步不建议继续只做单变量阈值调参。更合理的推进方式是训练一个轻量二阶段混淆纠错器：

1. 输入仍使用当前 IBO 候选的 crop、IBO 区域证据、正常参照距离、频率证据和 D-FINE 原始类别分数；
2. 只对高频混淆簇内类别做局部重判别，而不是重训完整检测器；
3. 在纠错后重新生成 val 预测，再按 SimLTD 评价指标和本文自定义 correct / wrong / miss 指标共同评估。

这一步对应本文方案中的“混淆类对比纠错”，用于弥补当前 evidence fusion 主要改善候选可靠性、但类别边界仍不稳的问题。

## 15. 二阶段混淆类纠错 pilot

基于上一节的混淆对统计，进一步训练轻量二阶段混淆纠错器。该纠错器不改变检测框，只尝试修正候选类别。

### 15.1 设置

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\train_lvis_confusion_corrector.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_confusion_corrector_20260916\extratrees_cov020_rel010_apply015_p035_m010_light
```

训练标签只使用 train split 中覆盖 GT 面积 ≥ 0.20 的候选，避免使用 val GT 作为模型输入。输入特征包括：

1. D-FINE 原始预测类别与预测分数；
2. IBO reliability 分数；
3. 正常参照距离；
4. 频率证据分数；
5. 原始候选框与局部 IBO 框的几何特征。

本轮没有引入 crop 图像视觉 embedding，因此它是一个轻量元特征纠错器。

训练规模与 holdout 表现：

| 项目 | 数值 |
|---|---:|
| train candidates | 48,000 |
| holdout candidates | 12,000 |
| holdout accuracy | 14.85% |
| holdout balanced accuracy | 6.37% |
| val candidates | 43,882 |
| 初始应用纠错数 | 1,309 |

### 15.2 回投结果

与未纠错 IBO reliability 的最佳实例阈值结果相比：

| 方法 | 应用条件 | 纠错候选数 | correct / wrong / miss | 实例准确率变化 |
|---|---|---:|---:|---:|
| 未纠错 IBO reliability | threshold 0.15 | 0 | 3,761 / 6,132 / 10,998 | baseline |
| 轻量纠错器 | p≥0.35, margin≥0.10 | 1,309 | 3,721 / 6,172 / 10,998 | -40 correct |
| 阈值扫描 | p≥0.50, margin≥0.15 | 317 | 3,752 / 6,141 / 10,998 | -9 correct |
| 阈值扫描 | p≥0.60, margin≥0.20 | 137 | 3,756 / 6,137 / 10,998 | -5 correct |
| 阈值扫描 | p≥0.70, margin≥0.25 | 62 | 3,757 / 6,136 / 10,998 | -4 correct |

### 15.3 结论

当前轻量二阶段纠错器没有超过未纠错 IBO reliability。它说明一个重要问题：对于 LVIS 这类 1000+ 类长尾检测数据，混淆类别纠错不能只依赖 D-FINE 分数、IBO 几何、正常参照距离和频率证据这类元特征；这些特征能判断“候选是否可靠”，但不足以稳定地区分 cow / sheep / horse / elephant、cabinet / drawer / refrigerator、apple / tomato / banana / doughnut 等细粒度类别。

因此后续混淆类对比纠错应调整为：

1. 在高频混淆簇内训练局部纠错器，而不是全 1000+ 类统一纠错；
2. 引入 crop 图像视觉 embedding 或 D-FINE query embedding，补充类别语义信息；
3. 保留当前 reliability fusion 作为候选筛选，不把它直接当作类别重判别器；
4. 最终对比时应报告“IBO 提升候选召回与可靠性，但类别纠错需要视觉语义特征”的消融结论。

## 16. 高频混淆簇 crop 视觉表征纠错实验

针对第 15 节“浅层元特征纠错不足”的问题，本阶段进一步实现 train-defined 高频混淆簇内的 crop 视觉表征纠错。该实验对应本文方法中的“混淆类对比纠错”，不是独立于方法之外的额外技巧。

### 16.1 设置

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\train_lvis_cluster_crop_corrector.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_cluster_crop_corrector_20260916\sourcecrop_unicode_top30_p045_m010
```

关键实现：

1. 只用 train split 统计高频 pred→GT 混淆边，构造混淆簇；
2. 只用 train split 中覆盖 GT 面积 ≥ 0.20 的候选作为纠错训练样本；
3. val split 只用于最后回投评估，不用 val GT 参与簇发现或训练；
4. crop 图片没有直接使用已清理的候选 crop 文件，而是从 LVIS 原图按 `crop_xyxy` 即时裁剪；
5. Windows 中文路径下 OpenCV `imread` 失效，已改为 `np.fromfile + cv2.imdecode` 读取。

输入特征包括：

- 候选 crop 的颜色、梯度、DCT 频率和纹理统计；
- D-FINE 原始预测类别和分数；
- IBO reliability 分数；
- 正常参照距离；
- 频率证据分数；
- 候选框几何特征。

### 16.2 train-defined 混淆簇

| 簇 | 类别 | train candidates | val candidates | holdout acc | holdout balanced acc |
|---:|---|---:|---:|---:|---:|
| 0 | apple / banana / carrot / doughnut / orange_(fruit) / tomato | 12,000 | 6,267 | 98.58% | 97.84% |
| 1 | cow / elephant / horse / sheep | 12,000 | 5,779 | 98.75% | 98.73% |
| 2 | cabinet / cupboard / drawer | 8,792 | 4,901 | 98.75% | 97.94% |
| 3 | bicycle / motorcycle | 5,016 | 1,320 | 99.50% | 99.26% |
| 4 | cushion / pillow | 1,827 | 1,351 | 99.73% | 99.82% |
| 5 | laptop_computer / monitor_(computer_equipment) computer_monitor | 2,614 | 1,036 | 99.24% | 99.25% |
| 6 | signboard / street_sign | 1,474 | 1,079 | 96.27% | 96.26% |
| 7 | sink / toilet | 2,695 | 981 | 99.63% | 99.60% |

val 上共有 22,714 个候选进入混淆簇预测，其中 4,714 个候选满足纠错阈值并被实际改类。

### 16.3 回投结果

| 方法 | correct / wrong / miss | 相对未纠错 IBO |
|---|---:|---:|
| 未纠错 IBO reliability | 3,761 / 6,132 / 10,998 | baseline |
| 全 1000+ 类浅层元特征纠错 | 3,721 / 6,172 / 10,998 | -40 correct, +40 wrong |
| 高频混淆簇 crop 视觉纠错 | 3,769 / 6,124 / 10,998 | +8 correct, -8 wrong |

### 16.4 结论

与第 15 节的浅层元特征纠错相比，加入候选 crop 视觉表征并限制在高频混淆簇内后，纠错方向变为正向：correct 增加、wrong 下降、miss 不变。提升幅度目前较小，但它证明了一个关键判断：

> IBO、正常参照距离和频率证据适合判断候选是否可靠；真正的类别纠错仍需要候选的视觉/语义表征，并且应在高频混淆簇内做局部重判别。

因此正式方法中，“混淆类对比纠错”应优先写成 detection query / crop embedding 驱动的局部对比纠错，而不是只依赖分数、几何、正常参照距离和频率证据。对于本文焊接缺陷场景，推荐优先使用 D-FINE detection query 表征；当 query 导出不方便时，候选 crop 视觉 embedding 可作为工程替代。

## 17. D-FINE detection query 表征混淆簇纠错实验

第 16 节使用 crop 视觉统计特征作为工程替代。本阶段进一步导出 D-FINE decoder 的 detection query embedding，用 query 表征替代 crop 统计特征，验证更贴近本文方法主线的“基于 detection query 的混淆类局部对比纠错”。

### 17.1 Query 特征导出

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\extract_lvis_dfine_query_features.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_dfine_query_features_20260917\full
```

实现方式：

1. 使用与 IBO 候选构建完全一致的 D-FINE-S checkpoint、no-stretch resize+pad 推理流程；
2. D-FINE decoder 在 eval 输出中已有 `query_feats`；
3. D-FINE postprocessor 通过 top-k 分数得到 `query_index`；
4. 对每个已导出的 IBO candidate，用同一个 `query_index` gather 对应 256 维 query embedding；
5. 与已有 `ibo_id` 顺序逐项对齐，不重建 crop，不覆盖原始 manifest。

导出结果：

| split | 候选数 | query 维度 | alignment mismatch |
|---|---:|---:|---:|
| train | 221,120 | 256 | 0 |
| val | 43,882 | 256 | 0 |

这说明现有 IBO candidate 与 D-FINE detection query 可以一一对应。该结果支持论文中的方法假设：D-FINE 的 detection query 可以作为“一个候选缺陷的空间—频率证据载体”。

### 17.2 Query embedding 混淆簇纠错设置

脚本：

```text
D:\1\项目论文\zhwk_project\scripts\train_lvis_cluster_query_corrector.py
```

输出目录：

```text
D:\1\项目论文\zhwk_runs\simltd_lvis_cluster_query_corrector_20260917\query_top30_p045_m010
```

输入特征包括：

- D-FINE detection query embedding，256 维；
- D-FINE 原始预测类别和预测分数；
- IBO reliability 分数；
- 正常参照距离；
- 频率证据分数；
- 候选框几何特征；
- 混淆簇内原始预测类别 one-hot。

混淆簇仍只由 train split 的高频 pred→GT 错分类别对发现，val split 只用于最终评估。

### 17.3 Query 纠错器训练表现

| 簇 | 类别 | train candidates | val candidates | holdout acc | holdout balanced acc |
|---:|---|---:|---:|---:|---:|
| 0 | apple / banana / carrot / doughnut / orange_(fruit) / tomato | 12,000 | 6,267 | 97.42% | 96.44% |
| 1 | cow / elephant / horse / sheep | 12,000 | 5,779 | 95.75% | 95.75% |
| 2 | cabinet / cupboard / drawer | 8,792 | 4,901 | 93.69% | 89.44% |
| 3 | bicycle / motorcycle | 5,016 | 1,320 | 97.41% | 95.44% |
| 4 | cushion / pillow | 1,827 | 1,351 | 96.45% | 92.26% |
| 5 | laptop_computer / monitor_(computer_equipment) computer_monitor | 2,614 | 1,036 | 93.50% | 93.49% |
| 6 | signboard / street_sign | 1,474 | 1,079 | 96.95% | 96.97% |
| 7 | sink / toilet | 2,695 | 981 | 98.14% | 97.92% |

val 上共有 22,714 个候选进入 query 混淆簇预测。默认阈值 p≥0.45、margin≥0.10 时，实际改类 4,193 个候选。

### 17.4 回投结果与阈值扫描

| 方法 | 应用条件 | 实际改类候选数 | correct / wrong / miss | 相对未纠错 IBO |
|---|---|---:|---:|---:|
| 未纠错 IBO reliability | threshold 0.15 | 0 | 3,761 / 6,132 / 10,998 | baseline |
| crop 视觉统计混淆簇纠错 | p≥0.45, margin≥0.10 | 4,714 | 3,769 / 6,124 / 10,998 | +8 correct, -8 wrong |
| query 混淆簇纠错 | p≥0.45, margin≥0.10 | 4,193 | 3,761 / 6,132 / 10,998 | 0 |
| query 混淆簇纠错 | p≥0.55, margin≥0.15 | 2,580 | 3,784 / 6,109 / 10,998 | +23 correct, -23 wrong |
| query 混淆簇纠错 | p≥0.65, margin≥0.20 | 1,258 | 3,780 / 6,113 / 10,998 | +19 correct, -19 wrong |
| query 混淆簇纠错 | p≥0.75, margin≥0.25 | 484 | 3,765 / 6,128 / 10,998 | +4 correct, -4 wrong |

### 17.5 阶段性结论

query embedding 版混淆簇纠错在合适阈值下优于 crop 视觉统计版：

```text
crop 统计版最佳：+8 correct / -8 wrong
query embedding 版最佳：+23 correct / -23 wrong
```

这说明 D-FINE detection query 比手工 crop 视觉统计更适合作为候选缺陷的语义表征。与此同时，默认低阈值 p≥0.45 时 query 纠错正负抵消，说明 query 纠错需要可靠性门控或 margin 约束，不能无条件覆盖原始类别。

因此本文方法中，“混淆类对比纠错”可明确表述为：

> 在高频混淆类簇内，以 D-FINE detection query 为候选语义载体，并结合 IBO 区域证据、正常参照距离和频率证据进行局部对比重判别；仅当纠错置信度和类别间隔同时满足可靠性门控时，才覆盖原始类别预测。

这与本文主线一致：IBO 与正常参照频率证据先判断候选可靠性，D-FINE query 负责补足类别语义边界，最终通过可靠性门控避免过度纠错。
