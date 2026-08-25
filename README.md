> **Mini-DeepID educational reproduction inspired by CVPR 2014 DeepID, not a reproduction of the paper's reported LFW 97.45% result.**

# Mini-DeepID 教学复现

本项目是 LFW 10 人 closed-set identification 教学实验：从 LFW 自动选择照片最多的 10 个身份，每个身份固定 50 张灰度人脸（共 500 张），训练一个四层卷积网络，把 160 维隐藏层激活作为 DeepID 特征，做 10 类身份分类。它是 DeepID 的**思想级**教学复现，不是 CVPR 2014 DeepID 论文的完整复现，也不是论文报告的 LFW 97.45% verification benchmark 复现。

## 项目性质

- 10 个身份 × 50 张 = 500 张灰度 64×64 人脸（随机种子 42）。
- Train / Val / Test = 350 / 70 / 80（每人 35 / 7 / 8）。
- 输入 `1×64×64`；四层卷积；160D DeepID embedding；10 logits。
- 训练设备 `cuda:0`；CUDA 失败立即停止，**绝不静默回退 CPU**。

本项目**不是**：陌生人识别系统、实际门禁或生产身份认证系统、论文 benchmark 复现。

## 环境

- Python 3.12；基解释器 `C:\Users\pengjian\AppData\Local\Programs\Python\Python312\python.exe`。
- 项目专属解释器：`E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe`（不复用其他项目环境）。
- 依赖（`requirements.txt`，锁定见 `requirements-lock.txt`）：torch 2.12.1 / torchvision 0.27.1（CUDA 13.0 wheel）、numpy、pandas、Pillow、scikit-learn、matplotlib、seaborn、tqdm、pytest。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU（驱动 610.88、计算能力 12.0、显存 8151 MiB）。

### 从空环境重现

```powershell
# 1. 创建专属虚拟环境
& 'C:\Users\pengjian\AppData\Local\Programs\Python\Python312\python.exe' -m venv '.venv'

# 2. 安装依赖（官方 cu130 wheel）
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt

# 3. 真实 CUDA 验证（GPU 名称 / runtime / 计算能力 / 2000x2000 matmul / 模型前向+反向+有限梯度）
& '.\.venv\Scripts\python.exe' verify_environment.py

# 4. 下载 LFW 并生成冻结数据（若大档案下载停滞，可用 curl 续传 lfw-funneled.tgz）
& '.\.venv\Scripts\python.exe' prepare_lfw.py

# 5. 数据验收（无重叠、标签、图像规格、样本网格）
& '.\.venv\Scripts\python.exe' verify_data.py

# 6. 单元测试
& '.\.venv\Scripts\python.exe' -m pytest -q

# 7. 32 张 Tiny Overfit（必须 >=95%，目标 100%）
& '.\.venv\Scripts\python.exe' tiny_overfit.py

# 8. 两轮冒烟训练（checkpoint 保存与 resume）
& '.\.venv\Scripts\python.exe' train.py --epochs 2

# 9. 正式训练（<=80 epochs，由用户在 PyCharm 亲手启动）
#    在 PyCharm 中选择 .venv 解释器，打开 run_in_pycharm.py，点击绿色 Run。

# 10. 一次性 Final Test（冻结测试集 80 张，仅一次）
& '.\.venv\Scripts\python.exe' evaluate.py

# 11. PCA 可视化（只在 train embeddings 上 fit，test 只 transform）
& '.\.venv\Scripts\python.exe' visualize_embeddings.py
```

## 冻结数据协议

`data/manifests/split_manifest.csv` 一旦生成即视为**冻结**实验协议。此后的 dataset、训练、验证、测试、预测、可视化**只能读取该清单**，禁止 `random_split` 或重新随机划分。Val/Test 只允许 resize、tensor 转换与 normalize，禁止任何随机增强；随机增强仅对 Train 动态执行。三分区 source_index 互斥。

## 预期成功标记

- `MINI_DEEPID_CUDA_SMOKE_OK device=cuda:0`
- `LFW_PREPARE_OK classes=10 samples=500 train=350 val=70 test=80`
- `MINI_DEEPID_DATA_ACCEPT_OK classes=10 samples=500 images=500`
- `MINI_DEEPID_TINY_OVERFIT_OK ...`
- `MINI_DEEPID_TRAIN_OK ...`
- `MINI_DEEPID_EVAL_OK samples=80`
- `MINI_DEEPID_PCA_OK fit=train transform=test dimensions=160->2 samples=80`

## 实测结果

| 指标 | 值 |
| --- | --- |
| Random Guess Accuracy | 10.00% |
| Mini-DeepID Test Accuracy | 77.50%（62/80） |
| 最佳验证准确率（epoch 75） | 80.00% |
| Macro Precision / Recall / F1 | 0.7927 / 0.7750 / 0.7758 |

## 输出位置

- 冻结清单/身份映射：`data/manifests/`（`split_manifest.csv`、`identities.json`）
- 导出灰度图：`data/processed/`
- 最佳检查点：`checkpoints/mini_deepid_best.pth`
- 报告/图件：`outputs/`（`metrics.json`、`confusion_matrix.png`、`predictions.png`、`embeddings_pca.png`、`final_test_receipt.json`、`data_summary.json`、`tiny_overfit.json`、`tiny_overfit_curve.png`、`environment_report.json`）
- 训练曲线：`outputs/<run_id>/loss_curve.png`、`accuracy_curve.png`
- Gate 台账：`outputs/gate_status.json`

## 已知故障模式

- CUDA 不可用 → `verify_environment.py` / 训练立即报错，绝不静默回退 CPU。
- LFW 大档案下载停滞 → 用 `curl -L -C - --retry ...` 续传官方 `lfw-funneled.tgz` 到 `data/cache/lfw_home/`（SHA256 校验后重跑 `prepare_lfw.py`）。
- loss 非有限 → `run_epoch` 抛错停止。
- checkpoint 身份映射不符 → `load_checkpoint` 抛 `ValueError`。
- `final_test_receipt.json` 已存在 → `evaluate.py` 拒绝二次 Final Test。

## DeepID 与 Mini-DeepID 的区别

| | 原论文 DeepID | Mini-DeepID |
| --- | --- | --- |
| 规模 | 约 10,000 身份 | 10 身份 |
| 结构 | 多 face patch、多 ConvNet 集成 | 单四层卷积 + 多尺度融合 |
| 度量 | Joint Bayesian + LFW verification（97.45%） | closed-set identification |
| 特征 | 160D DeepID | 160D DeepID |

## 闭集限制

预测属于闭集分类，输入必须属于既定的 10 个身份之一。Softmax confidence **不代表**可靠的陌生人检测概率；本项目不提供 `UNKNOWN`，不用于实际门禁、身份认证或其他高风险场景。
