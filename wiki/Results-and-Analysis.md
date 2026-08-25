# Results and Analysis

## Overall metrics

| Metric | Value |
| --- | ---: |
| Random baseline | 10.00% |
| Final test accuracy | **77.50% (62/80)** |
| Macro Precision | 0.7927 |
| Macro Recall | 0.7750 |
| Macro F1 | **0.7758** |
| Best validation accuracy | 80.00% at epoch 75 |

![Training accuracy](https://raw.githubusercontent.com/PengjinHao-cell/MiNi-DeepID/master/docs/assets/accuracy_curve.png)

## Class-wise observations

Junichiro Koizumi and Jean Chretien achieved 8/8 correct predictions. Tony Blair was the most difficult class with 4/8 correct and recall 0.50. Gerhard Schroeder reached recall 0.75 but precision 0.5455, indicating that samples from other identities were frequently attracted to this class.

![Confusion matrix](https://raw.githubusercontent.com/PengjinHao-cell/MiNi-DeepID/master/docs/assets/confusion_matrix.png)

## Embedding view

PCA is a two-dimensional linear projection and supporting evidence only. It cannot prove separability in the full 160D space or open-set recognition ability.

![Embedding PCA](https://raw.githubusercontent.com/PengjinHao-cell/MiNi-DeepID/master/docs/assets/embeddings_pca.png)
