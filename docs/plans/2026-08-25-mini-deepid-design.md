# Mini-DeepID 教学复现设计

日期：2026-08-25

## 1. 项目目标

在 Windows 本地使用 RTX 5060 Laptop GPU，以 LFW 为唯一数据源，完成一个可解释、可重复的 Mini-DeepID 教学实验。系统从 LFW 自动选择照片最多的 10 个身份，每个身份固定抽取 50 张灰度人脸图像，训练一个四层卷积网络，并把 160 维隐藏层激活作为 DeepID 特征。

项目最终输出身份分类结果、训练曲线、混淆矩阵、预测样例和 PCA 特征分布图。它复现的是 DeepID 的核心思想，不宣称复现 CVPR 2014 论文的完整规模或 LFW verification 指标。

## 2. 依据与资料

- 原论文：Yi Sun, Xiaogang Wang, Xiaoou Tang, “Deep Learning Face Representation from Predicting 10,000 Classes,” CVPR 2014。
- 官方论文页面：https://openaccess.thecvf.com/content_cvpr_2014/html/Sun_Deep_Learning_Face_2014_CVPR_paper.html
- 官方论文 PDF：https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf
- LFW 加载接口：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
- PyTorch Windows 安装入口：https://pytorch.org/get-started/locally/

实施时将论文 PDF、引用信息和中文阅读提纲保存到 `docs/references/`。阅读提纲重点解释身份分类如何形成判别特征、160 维信息瓶颈、多区域特征的原论文动机，以及 Mini 版本省略这些模块的原因。

## 3. 本地环境决策

已实测本机存在以下解释器：

- `C:\Users\pengjian\AppData\Local\Programs\Python\Python312\python.exe`：Python 3.12.10。
- `D:\虚拟C盘\暑假项目\1\Ai量化\.venv\Scripts\python.exe`：Python 3.14.5。
- `D:\虚拟C盘\暑假项目\1\Ai量化\Kronos\.venv\Scripts\python.exe`：Python 3.12.13，已有 PyTorch，但属于其他项目。

Mini-DeepID 使用 C 盘 Python 3.12.10 在项目根目录创建专属 `.venv`。不复用两个 D 盘环境，防止依赖升级相互影响。当前普通系统 PowerShell 可通过 `PATH` 找到 C 盘 Python；此前受限终端无法解析该命令是执行隔离现象，不是 Python 缺失或磁盘不同步。

GPU 已识别为 NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB，计算能力 12.0，驱动 610.88。PyTorch 使用官方稳定 CUDA 构建。安装后生成锁定依赖清单。GPU 验收必须包含 `cuda:0` 上的真实张量计算和模型前向、反向传播；CUDA 失败时停止，不静默回退 CPU。

## 4. 数据设计

使用 `fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=False, funneled=True)` 下载并加载 LFW。统计标签数量后自动选择照片最多的前 10 个身份。LFW 前 10 个身份中第 10 名只有约 53 张，因此每人固定抽取 50 张，而不是以 60 张为上限形成不平衡数据。

固定随机种子为 42。每个身份的 50 张图片划分为：

- 训练集 35 张；
- 验证集 7 张；
- 测试集 8 张。

最终共有 500 张图片，其中训练集 350 张、验证集 70 张、测试集 80 张。scikit-learn 返回的 funneled 灰度人脸统一缩放为 `64×64`。导出目录按清洗后的身份名称组织，同时生成：

- `data/manifests/identities.json`：模型标签到身份名称的固定映射；
- `data/manifests/split.csv`：原始索引、身份、标签、分区和导出路径；
- `outputs/data_summary.json`：数量和图像规格摘要；
- `outputs/class_distribution.png`：类别分布图。

manifest 一旦生成，后续训练直接复用，不重新随机选择。训练增强仅在读取训练集时动态执行；验证集和测试集不增强。三个分区的原始索引必须互斥。

## 5. 模型设计

模型输入为 `[B, 1, 64, 64]`，由四个卷积块组成：

```text
Input                    1×64×64
Conv1  1→32  + Pool     32×32×32
Conv2 32→64  + Pool     64×16×16
Conv3 64→128 + Pool    128×8×8
Conv4 128→128 + Pool   128×4×4
```

Conv3 输出经自适应池化变为 `128×4×4`，与 Conv4 的 `128×4×4` 输出拼接。拼接特征经过全连接层、ReLU 和 Dropout，得到 `[B, 160]` 的 DeepID embedding；分类器再把 160 维特征映射为 `[B, 10]` logits。

模型前向接口同时返回 `(embedding, logits)`。训练使用 logits 计算交叉熵；评估和可视化直接使用 embedding。第一版不使用预训练模型、Triplet Loss、ArcFace、verification loss、多 face patch、多 CNN 集成或 Joint Bayesian。

## 6. 训练策略

- 最大轮数：80；
- batch size：32；
- 优化器：AdamW；
- 初始学习率：`1e-3`；
- weight decay：`1e-4`；
- 损失：`CrossEntropyLoss`；
- early stopping patience：12；
- 设备：`cuda:0`；
- 随机种子：42。

训练增强包括概率 0.5 的水平翻转、正负 8 度旋转、轻微平移和 0.95 至 1.05 缩放。验证和测试只执行统一尺寸、张量转换及归一化。

每轮记录训练和验证的 loss、accuracy。根据验证集准确率保存最佳检查点；测试集只在模型选择完成后运行一次。检查点包含模型状态、优化器状态、当前轮数、最佳指标、身份映射、随机种子和配置快照。

## 7. 组件和目录

```text
MINI DEEP ID/
├── README.md
├── requirements.txt
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── visualize_embeddings.py
├── verify_environment.py
├── prepare_lfw.py
├── tests/
├── docs/
│   ├── plans/
│   └── references/
├── data/
│   ├── cache/
│   ├── processed/
│   └── manifests/
├── checkpoints/
└── outputs/
```

所有入口读取 `config.py`，确保图像尺寸、归一化、随机种子、embedding 维数和身份顺序一致。每次训练结果保存到独立的 `outputs/<run_id>/`，不覆盖历史结果。

## 8. 错误处理边界

数据准备完成前必须验证：前 10 个身份均至少有 50 张、每类恰好导出 50 张、标签连续为 0 至 9、三个分区互斥、图片全部为有限值灰度 `64×64`。下载失败时保留官方缓存并明确提示重试，不写入最终 manifest。

训练中如 CUDA 不可用、真实 CUDA 运算失败、loss 非有限、检查点损坏或身份映射不匹配，程序立即停止并给出具体错误。检查点先写临时文件，完整写入后再替换目标文件。

预测属于闭集分类。输入必须属于既定的 10 个身份之一；Softmax confidence 不代表可靠的陌生人检测概率。第一版不提供 `UNKNOWN`，不用于实际门禁、身份认证或其他高风险场景。

## 9. 测试与验收

单元测试覆盖数据选择确定性、每类数量、分区互斥、图像张量、模型输出形状、梯度传播、检查点往返和身份映射。冒烟测试使用少量数据训练两轮，要求 loss 有限、参数发生更新、保存后重新载入得到相同输出。

完整实验必须生成：

- `loss_curve.png`；
- `accuracy_curve.png`；
- `confusion_matrix.png`；
- `predictions.png`；
- `embeddings_pca.png`；
- `metrics.json`；
- `environment_report.json`；
- `data_summary.json`。

技术验收要求环境、数据、前向、反向、检查点及全部输出通过。10 类随机准确率为 10%；50% 以上视为明显超过随机基线，70% 为教学目标而非保证。未达到目标时保留真实结果并分析过拟合、类别混淆和数据差异，不修改测试划分或反复使用测试集调参。

## 10. 实施顺序

1. 归档论文、链接、引用和阅读提纲。
2. 创建项目 `.venv`，安装并锁定依赖，完成真实 CUDA 验收。
3. 下载 LFW，生成固定的 10 人、每人 50 张数据集及 manifest。
4. 生成数据摘要和样本网格，完成数据验收。
5. 先写形状、梯度和检查点测试，再实现模型。
6. 完成两轮冒烟训练。
7. 正式训练并保存最佳验证模型。
8. 冻结模型选择，执行一次最终测试。
9. 生成全部图表、指标和使用说明。

## 11. 非目标

- 不完整复刻论文的约 10,000 身份、60 个网络和多 patch 流程；
- 不报告论文级 LFW verification 结果；
- 不识别训练身份之外的陌生人；
- 不收集或训练用户私人照片；
- 不构建 Web、桌面 GUI、摄像头实时识别或生产部署系统。
