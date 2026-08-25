# 参考资料归档（Primary References）

## 原论文

- 标题：Deep Learning Face Representation from Predicting 10,000 Classes
- 作者：Yi Sun, Xiaogang Wang, Xiaoou Tang
- 会议：IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2014

## 官方链接

- HTML 页面：https://openaccess.thecvf.com/content_cvpr_2014/html/Sun_Deep_Learning_Face_2014_CVPR_paper.html
- PDF（本地归档）：`Sun_Deep_Learning_Face_2014_CVPR_paper.pdf`
- PDF 官方地址：https://openaccess.thecvf.com/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf

## BibTeX（取自官方页面）

```bibtex
@InProceedings{Sun_2014_CVPR,
author = {Sun, Yi and Wang, Xiaogang and Tang, Xiaoou},
title = {Deep Learning Face Representation from Predicting 10,000 Classes},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2014}
}
```

## 数据接口

- LFW 加载接口（scikit-learn）：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html

## Mini-DeepID 与论文的关系（保留与省略）

本项目是 DeepID 的思想级教学复现，**不是**论文报告的 LFW 97.45% verification benchmark 复现。

**保留的核心思想：**

- 身份分类（closed-set identification）：以 softmax 交叉熵为监督信号，用分类任务驱动判别特征学习。
- 160 维信息瓶颈：以 160 维隐藏层激活作为 DeepID embedding。
- 多尺度特征融合：在单网络内拼接 Conv3 与 Conv4 输出，保留"多尺度互补"的思想。

**省略的模块：**

- 原论文约 10,000 个身份的训练规模；
- 多个 face patch、多个 ConvNet 集成；
- Joint Bayesian 度量学习后端；
- LFW verification 协议（因此不报告 97.45% 指标）。
