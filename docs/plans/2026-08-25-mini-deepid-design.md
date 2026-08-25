# Mini-DeepID 教学复现设计

日期：2026-08-25

> **Mini-DeepID educational reproduction inspired by CVPR 2014 DeepID, not a reproduction of the paper's reported LFW 97.45% result.**

## 1. 项目目标

在 Windows 本地使用 RTX 5060 Laptop GPU，以 LFW 为唯一数据源，完成一个可解释、可重复的 Mini-DeepID 教学实验。系统从 LFW 自动选择照片最多的 10 个身份，每个身份固定抽取 50 张灰度人脸图像，训练一个四层卷积网络，并把 160 维隐藏层激活作为 DeepID 特征。

项目最终输出身份分类结果、训练曲线、混淆矩阵、预测样例和 PCA 特征分布图。原论文先利用约 10,000 个身份和多区域 ConvNet 学习 DeepID，再在 LFW 上做 verification；本项目直接在 LFW 的 10 个身份上做 closed-set identification。因此它是 DeepID 的思想级教学复现，不是论文 benchmark 或 97.45% 指标复现。README 第一行必须保留上面的英文声明。

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

GPU 已识别为 NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB，计算能力 12.0，驱动 610.88。环境基线锁定为 Python 3.12、PyTorch 2.12.1、torchvision 0.27.1 和官方 CUDA 13.0 wheel。PyTorch 2.12 已不再把 CUDA 12.8 作为标准 wheel，并建议 Blackwell 使用 CUDA 13.0 以上；本机驱动高于 Windows 最低 580.88。

安装后同时保存 `requirements.txt`、完整解析后的 `requirements-lock.txt`、便于人阅读的 `environment.txt` 和机器可读的 `outputs/environment_report.json`。两种环境报告均记录 Python、PyTorch、torchvision、CUDA runtime、GPU、驱动、计算能力和随机种子。GPU 验收必须包含 `cuda:0` 上的 `2000×2000` 矩阵乘法以及模型前向、反向传播；CUDA 失败时停止，不静默回退 CPU。

## 4. 数据设计

使用 `fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=False, funneled=True)` 下载并加载 LFW。统计标签数量后自动选择照片最多的前 10 个身份。LFW 前 10 个身份中第 10 名只有约 53 张，因此每人固定抽取 50 张，而不是以 60 张为上限形成不平衡数据。

固定随机种子为 42。每个身份的 50 张图片划分为：

- 训练集 35 张；
- 验证集 7 张；
- 测试集 8 张。

最终共有 500 张图片，其中训练集 350 张、验证集 70 张、测试集 80 张。scikit-learn 返回的 funneled 灰度人脸统一缩放为 `64×64`。导出目录按清洗后的身份名称组织，同时生成：

- `data/manifests/identities.json`：模型标签到身份名称的固定映射；
- `data/manifests/split_manifest.csv`：原始索引、文件名、身份、标签、分区和导出路径；
- `outputs/data_summary.json`：数量和图像规格摘要；
- `outputs/class_distribution.png`：类别分布图。

`split_manifest.csv` 一旦生成即视为冻结实验协议。此后的 dataset、训练、验证、测试、预测和可视化代码只能读取该清单，禁止再次调用 `random_split` 或重新随机划分。数据流程严格为“先划分，再增强”：训练增强仅在读取 `train` 行时动态执行；`val` 和 `test` 只允许 resize、tensor conversion 和 normalize，禁止 RandomRotation、RandomCrop、RandomFlip。三个分区的原始索引必须互斥。

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

每轮记录训练和验证的 loss、accuracy。根据验证集准确率保存最佳检查点；测试集只在模型、学习率、Dropout、早停策略和唯一最佳检查点全部冻结后运行一次。最终评估生成 `final_test_receipt.json`，记录 checkpoint 与 manifest 的 SHA-256、协议版本、时间和指标；该文件存在时同一协议拒绝再次评估。看到测试结果后不得修改模型并重测。检查点包含模型状态、优化器状态、当前轮数、最佳指标、身份映射、随机种子和配置快照。

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

单元测试覆盖数据选择确定性、每类数量、分区互斥、图像张量、模型输出形状、梯度传播、检查点往返和身份映射。正式冒烟训练前新增 Tiny-set Overfit Test：从冻结的训练清单中按身份确定性选取 32 张，关闭随机增强、Dropout 和 weight decay，最多训练 300 轮；通过线为训练准确率达到 95%，目标为 100%。若未通过则停止，优先检查标签、预处理、loss、梯度、optimizer 和学习率，不得进入正式训练。随后再用完整管线训练两轮，验证 CUDA、loss、checkpoint 和 resume。

完整实验必须生成：

- `loss_curve.png`；
- `accuracy_curve.png`；
- `confusion_matrix.png`；
- `predictions.png`；
- `embeddings_pca.png`；
- `metrics.json`；
- `environment_report.json`；
- `data_summary.json`；
- `tiny_overfit.json`；
- `tiny_overfit_curve.png`；
- `final_test_receipt.json`。

技术验收要求环境、数据、前向、反向、检查点及全部输出通过。报告必须并列给出 `Random Guess Accuracy = 10.00%` 与 Mini-DeepID 的真实测试准确率；50% 以上视为明显超过随机基线，70% 为教学目标而非保证。PCA 只能在训练 embeddings 上 `fit`，再对保存的测试 embeddings 执行 `transform`，不得在测试集上拟合投影。未达到目标时保留真实结果并分析过拟合、类别混淆和数据差异，不修改模型、测试划分或反复使用测试集调参。

验收按 G0–G14 记录：实验声明、资料、环境、500 张数据、固定 350/70/80 清单、数据验收、shape、gradient、Tiny Overfit、Smoke Train、正式训练、唯一 best checkpoint、一次 Final Test、报告和空环境 README。任一 gate 失败都必须记录证据并停止后续阶段。

| Gate | 内容 | 必须通过的验收 |
| --- | --- | --- |
| G0 | 实验协议 | 明确 Mini-DeepID 不等于原论文 benchmark |
| G1 | 资料归档 | 原论文 PDF、BibTeX、阅读提纲齐全 |
| G2 | 环境 | Python 3.12、锁定依赖、RTX 5060 真 CUDA 运算成功 |
| G3 | 数据构建 | 10×50，共 500 张，seed 42 |
| G4 | 数据冻结 | `split_manifest.csv` 固定为 350/70/80 |
| G5 | 数据验收 | 无重叠、标签正确、样本网格正常 |
| G6 | 模型单测 | `[B,1,64,64]→[B,160]→[B,10]` |
| G7 | 梯度测试 | 所有应训练参数均有有限 gradient |
| G8 | Tiny Overfit | 32 张达到至少 95%，目标 100% |
| G9 | Smoke Train | GPU、loss、checkpoint、resume 正常 |
| G10 | 正式训练 | 不超过 80 epochs，只使用 train/val |
| G11 | 模型冻结 | 确定唯一 `mini_deepid_best.pth` |
| G12 | Final Test | 冻结测试集只进行一次正式推理 |
| G13 | 报告 | 随机基线、Accuracy、Confusion Matrix、Prediction、PCA |
| G14 | README | 从空环境可以重现，并保留实验性质声明 |

所有 gate 统一记录在 `outputs/gate_status.json`，每项包含状态、时间、关键指标和证据文件路径。只有前一 gate 为 `passed` 才允许后一 gate 启动；失败记录不得被删除，只能在修复后追加新的尝试结果。

## 10. 实施顺序

1. 归档论文、链接、引用和阅读提纲。
2. 创建项目 `.venv`，安装并锁定依赖，完成真实 CUDA 验收。
3. 下载 LFW，生成固定的 10 人、每人 50 张数据集及 manifest。
4. 生成数据摘要和样本网格，完成数据验收。
5. 先写形状、梯度和检查点测试，再实现模型。
6. 用 32 张冻结训练图片通过 Tiny-set Overfit Test。
7. 完成两轮冒烟训练和 checkpoint resume。
8. 把已验收程序交给用户在 PyCharm 中亲手启动正式训练。
9. 用户训练完成后冻结唯一最佳模型，再执行一次最终测试并生成报告。

## 11. PyCharm 亲手启动交接

代码实现阶段只运行单元测试、真实 CUDA 验证、32 图 Tiny Overfit 和两轮 Smoke Train；代理不得提前运行正式 80 轮训练。所有关卡通过后提供无命令行参数的 `run_in_pycharm.py` 和 `docs/PYCHARM_RUN.md`，并明确告诉用户“程序已准备好”。

用户在 PyCharm 中把项目解释器设置为 `E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe`，打开 `run_in_pycharm.py`，点击绿色运行按钮。控制台第一段必须打印 `sys.executable`、GPU、数据数量、已通过 gate 和 `PYCHARM_FORMAL_TRAIN_START`，随后逐轮显示 train/val loss 与 accuracy。启动脚本只训练并选择最佳 checkpoint，不自动触碰测试集。训练结束后用户回到任务中，再共同确认唯一 best model，并明确授权一次 Final Test。

## 12. 非目标

- 不完整复刻论文的约 10,000 身份、60 个网络和多 patch 流程；
- 不报告论文级 LFW verification 结果；
- 不识别训练身份之外的陌生人；
- 不收集或训练用户私人照片；
- 不构建 Web、桌面 GUI、摄像头实时识别或生产部署系统。
