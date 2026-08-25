# Mini-DeepID Wiki

Mini-DeepID is an educational, small-scale reproduction of the representation-learning ideas in the CUHK DeepID paper. It learns a 160D face embedding by predicting ten known LFW identities.

## Verified result

- 10 identities × 50 images
- Frozen split: 350 train / 70 validation / 80 test
- Best validation accuracy: 80.00%
- Final test accuracy: **77.50% (62/80)**
- Macro F1: **0.7758**

This is closed-set identification, not the paper's 97.45% LFW verification benchmark.

## Documentation

- [[Methodology]]
- [[Reproduction]]
- [[Results and Analysis|Results-and-Analysis]]
- [[Limitations and Future Work|Limitations-and-Future-Work]]
- [Full Chinese report](https://github.com/PengjinHao-cell/MiNi-DeepID/blob/master/docs/Mini-DeepID完整实验报告.md)

![Confusion matrix](https://raw.githubusercontent.com/PengjinHao-cell/MiNi-DeepID/master/docs/assets/confusion_matrix.png)
