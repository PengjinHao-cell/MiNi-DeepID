# Methodology

## Core idea

DeepID uses identity prediction as supervision:

$$
\mathbf{z}=f_\theta(x)\in\mathbb{R}^{160},\qquad
\hat{\mathbf{p}}=\operatorname{softmax}(W\mathbf{z}+\mathbf{b}).
$$

Mini-DeepID preserves identity classification, a 160D bottleneck, and complementary multi-scale features. It omits the original system's approximately 10,000 identities, multiple face-patch ConvNets, Joint Bayesian backend, and verification protocol.

## Architecture

1. Input: grayscale `1×64×64` face.
2. Four `3×3 Conv → ReLU → 2×2 MaxPool` blocks.
3. Adaptive pooling aligns Conv3 with Conv4 at `4×4`.
4. The two 128-channel maps are concatenated.
5. A fully connected layer produces the 160D embedding.
6. A linear classifier produces ten logits.

The model has 897,386 trainable parameters and uses cross-entropy, dropout 0.4, and weight decay $10^{-4}$.

## Data protocol

The LFW subset is balanced at 50 images per identity. Seed 42 creates 35/7/8 train/validation/test samples per identity. Augmentation is train-only and every consumer reads the same frozen manifest.
