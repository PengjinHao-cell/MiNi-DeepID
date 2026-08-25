# 在 PyCharm 中亲手启动正式训练

本项目的正式训练由**你**在 PyCharm 中亲手启动，代理不会代跑。

## 1. 选择解释器

1. 用 PyCharm 打开项目目录：`E:\Self Experiment\MINI DEEP ID`
2. 打开 `File → Settings → Project → Python Interpreter`
3. 选择（或新增）本项目专属解释器：

   ```
   E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe
   ```

## 2. 启动训练

1. 打开 `run_in_pycharm.py`
2. 点击右上角绿色 Run 三角形

## 3. 控制台预期输出

第一段（预检）会打印：

```
sys.executable=E:\Self Experiment\MINI DEEP ID\.venv\Scripts\python.exe
gpu=NVIDIA GeForce RTX 5060 Laptop GPU
data_samples=500
passed_gates=G0,G1,G2,G3,G4,G5,G6,G7,G8,G9
PYCHARM_TRAIN_READY
PYCHARM_FORMAL_TRAIN_START
```

随后逐轮打印：

```
epoch=1/80 train_loss=... train_acc=... val_loss=... val_acc=...
epoch=2/80 ...
```

## 4. 训练说明

- 最大 80 轮；验证集准确率 12 轮无提升则早停。
- 训练只使用 train/val，**不触碰测试集**。
- 最佳验证准确率检查点保存到：`checkpoints\mini_deepid_best.pth`
- 本轮记录（history/曲线）保存到：`outputs\run_<时间戳>\`
- 训练结束后，控制台最后打印 `MINI_DEEPID_TRAIN_OK ...`

## 5. 训练结束后

请回到对话中报告训练完成。届时我们会一起确认唯一 best model，并**明确授权一次** Final Test（G12）。在授权之前，请勿自行运行 `evaluate.py`。
