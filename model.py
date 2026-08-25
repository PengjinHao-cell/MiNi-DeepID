"""Mini-DeepID: four-convolution network with a 160D DeepID embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniDeepID(nn.Module):
    """Four convolution blocks with multi-scale fusion, a 160D DeepID
    embedding, and a 10-class classifier. Forward returns ``(embedding, logits)``.

    Channels: 1 -> 32 -> 64 -> 128 -> 128, each block conv(3x3)+ReLU+2x2 max pool.
    The third pooled feature map is adaptive-average-pooled to 4x4 and
    concatenated with the fourth pooled feature map (multi-scale fusion).
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 160,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_embed = nn.Linear(256 * 4 * 4, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.pool(F.relu(self.conv1(x)))  # 32x32x32
        x2 = self.pool(F.relu(self.conv2(x1)))  # 64x16x16
        x3 = self.pool(F.relu(self.conv3(x2)))  # 128x8x8
        x4 = self.pool(F.relu(self.conv4(x3)))  # 128x4x4
        fused = torch.cat([self.adaptive_pool(x3), x4], dim=1)  # 256x4x4
        embedding = self.dropout(F.relu(self.fc_embed(fused.flatten(1))))  # 160
        logits = self.classifier(embedding)  # 10
        return embedding, logits
